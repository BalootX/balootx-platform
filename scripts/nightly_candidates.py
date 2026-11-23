"""Generate and qualify a bounded batch of synthetic decision clips."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
from tempfile import TemporaryDirectory
import time

from baloot_platform.analysis.deductions import derive_hard_deductions
from baloot_platform.analysis.models import InformationState
from baloot_platform.challenge.interestingness_v4 import (
    ActionOutcome,
    ContextFeatures,
    score_interestingness,
)
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.research.calibration.score import engine_interest_v1
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.proof_service import UnifiedScenarioProofServiceV1
from baloot_platform.research.scenario_contracts import (
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.video.media_validation import (
    APPROVED_LAYOUT_TABLE_SHA256,
    media_tool,
    run_media_command,
)

from scripts import render_research_notice as notice
from scripts import viewer_seat_proof as viewer


REMOTION = notice.ROOT / "src/baloot_platform/video/remotion"
PLAY_INDICES = range(24, 28)
PUBLICATION_PENDING = {"V_history_unavailable", "publication_qa_unverified"}


class Rejected(ValueError):
    """A known exclusion with an artifact-safe reason code."""


class SearchBudgetExhausted(Exception):
    """The nightly search reached its wall-clock limit."""


def budget_seconds():
    seconds = int(os.environ.get("BALOOT_NIGHTLY_BUDGET_SECONDS", "1080"))
    if seconds <= 0:
        raise ValueError("nightly budget must be a positive number of seconds")
    cutoff = os.environ.get("BALOOT_NIGHTLY_JOB_DEADLINE")
    return min(seconds, int(cutoff) - time.time() - 15) if cutoff else seconds


def stop_search(_signal=None, _frame=None):
    raise SearchBudgetExhausted


@contextmanager
def search_deadline(seconds):
    if seconds <= 0:
        stop_search()
    deadline = time.monotonic() + seconds
    if not hasattr(signal, "setitimer"):
        yield deadline
        return
    previous = signal.signal(signal.SIGALRM, stop_search)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def write_json(path, value):
    atomic_write_text(path, viewer.receipt_bytes(value).decode("utf-8"))
    return file_receipt(path)


def file_receipt(path):
    return {"file": path.name, "sha256": notice._sha256(path)}


def synthetic_prefixes(engine, seed, contract):
    rng = random.Random(seed)
    dealer, buyer = Seat(rng.randrange(4)), Seat(rng.randrange(4))
    state = engine.create_round(seed=seed, dealer=dealer)
    engine.shuffle_and_initial_deal(state, cut_position=rng.randrange(1, 32))
    auction = BiddingMachine(
        dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(engine.rules.data),
        state.purchase_card.rank,
    )
    while auction.result is None:
        bid = (
            contract.value
            if auction.actor == buyer and contract.value in auction.legal_actions()
            else "pass"
        )
        trump = state.purchase_card.suit if bid == "hokum" else None
        auction.submit(bid, trump=trump)
    if not engine.complete_deal_from_bid_result(state, auction.result):
        raise Rejected("auction_incomplete")
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=seed))
    prefix = {
        "schemaVersion": 1,
        "deckOrder": list(state.shuffle.deck_order),
        "dealer": dealer.name.lower(),
        "roundNumber": state.round_number,
        "matchScore": list(state.match_score),
        "auctionActions": [
            {
                "actor": bid.seat.name.lower(),
                "action": bid.action,
                **(
                    {"trumpSuit": bid.trump.name.lower()}
                    if bid.trump is not None
                    else {}
                ),
            }
            for bid in state.bid_history
        ],
    }
    plays = []
    for index in range(PLAY_INDICES.stop):
        if index in PLAY_INDICES:
            yield {**prefix, "playActions": plays.copy()}
        play = engine.play_card(state, bot.choose(state), ekka=False)
        plays.append({"actor": play.seat.name.lower(), "card": play.card.code})


def winning_fixture(exact, prefix, proof_service, profile_sha, candidate_id):
    request = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "complete_state",
        "claim": "winning_root_action_count",
        "targetTeam": exact.round_state.current_player.team.name.lower(),
        "utility": "round_win",
        "successThreshold": 1,
        "budgetProfile": "late_game_exact_v1",
    }
    proof = proof_service.prove(
        exact,
        ScenarioProofRequestV1.build(request),
        bundle_hash=profile_sha,
    ).to_dict()
    if proof["completionStatus"] != "exact_complete":
        raise Rejected("round_win_proof_incomplete")
    if proof["winningRootActionCount"] != 1:
        raise Rejected("no_unique_round_winner")
    action = proof["selectedAction"]
    variation = proof["principalVariation"]
    if len(prefix["playActions"]) + len(variation) != 32 or any(
        "+" in play for play in variation
    ):
        raise Rejected("unsupported_render_continuation")
    return {
        "schemaVersion": 1,
        "scenarioId": candidate_id,
        "exactTracePrefix": prefix,
        "rulesMeaning": {"actingSeat": exact.next_actor},
        "expected": {
            "selectedAction": action,
            "winningRootActionCount": 1,
            "principalVariation": variation,
        },
        "proof": {
            **request,
            "claim": "selected_action_is_unique_winner",
            "selectedAction": action,
        },
    }, {row["action"]: int(row["value"]) for row in proof["rootActions"]}


def exact_viewer_inputs(proof, *, rules_sha):
    if proof["status"] != "exact_complete":
        raise Rejected(proof["reason"])
    if proof["rulesSha256"] != rules_sha:
        raise Rejected("viewer_proof_binding_failed")
    inputs = proof["scorerInputs"]
    return inputs


def candidate_score(
    exact, proof, inputs, proof_sha, round_win, margin, engine, solver, profile
):
    state = exact.round_state
    result = solver.solve(
        state, maximum_seconds=None, maximum_nodes=2_000_000, all_root_values=True
    )
    if result["status"] != "exact":
        raise Rejected("actual_deal_margin_proof_incomplete")
    values = {row["action"]: row["value"] for row in result["rootActions"]}
    action = proof["selectedAction"]
    if values.keys() != round_win.keys():
        raise Rejected("actual_deal_root_binding_failed")
    naive = HeuristicBot(engine, BotProfile(deterministic_seed=0)).choose(state).code
    shares = {row["action"]: row["bestShare"] for row in proof["moves"]}
    info = InformationState.from_complete(state, state.current_player)
    deductions = derive_hard_deductions(info)
    # The difficulty prior uses the proven round outcome as its win signal.
    prior = score_interestingness(
        [
            ActionOutcome(
                move,
                value / 40,
                float(round_win[move] > 0),
                value,
                best_deal_share=shares[move],
                is_obvious_heuristic_choice=move == naive,
            )
            for move, value in values.items()
        ],
        ContextFeatures(
            contract=state.contract.value,
            tricks_remaining=8 - len(state.completed_tricks),
        ),
    ).difficulty_prior
    beats = notice.setup_beats(
        {
            "bidHistory": exact.state["biddingHistory"],
            "dealer": exact.state["dealer"],
            "purchaseRecipient": exact.state["purchaseRecipient"],
            "escalationHistory": exact.state["escalationHistory"],
        }
    )
    measurements = {
        "exactStateId": exact.exact_state_id,
        "action": action,
        "legal": sorted(values),
        "roundWin": round_win,
        "pointMargins": values,
        "ddGap": values[action]
        - max(value for move, value in values.items() if move != action),
        "ddSwing": max(values.values()) - min(values.values()),
        "worldCoverageComplete": inputs["worldsComplete"],
        "bestShare": inputs["bestShare"],
        "strictShare": inputs["strictShare"],
        "worldMeanGap": inputs["worldMeanGap"],
        "baselineRegret": None,
        "naiveRegret": max(values.values()) - values[naive],
        "naiveAction": naive,
        "interestingnessV4": {"difficulty_prior": prior},
        "legalReplayVerified": True,
        "hardDeductions": len(deductions),
        "exactActorProofVerified": True,
        "exactActorProofSha256": proof_sha,
        "finalMargin": margin,
        "playIndex": len(state.played_cards),
        "auctionSetupBeats": len(beats),
        "durationSeconds": profile["exports"]["decision_clip"]["target_seconds"],
    }
    return engine_interest_v1(measurements), measurements


def check_hook(bundle_file, bundle_sha, output):
    renderer = REMOTION / "scripts/render-decision-clip.mjs"
    subprocess.run(
        ["node", str(renderer), "--check-hook", str(bundle_file), str(output)],
        cwd=REMOTION,
        check=True,
        timeout=120,
        capture_output=True,
    )
    evidence = output / "hook-layout-evidence.json"
    data = json.loads(evidence.read_bytes())
    (hook,) = data["items"]
    png = output / "layout-hook.png"
    table = output / "layout-table.json"
    binding = data["binding"]
    if not (
        data["status"] == "stills_passed"
        and data["scope"] == "hook"
        and data["postingAuthorized"] is False
        and binding["mode"] == "qualification"
        and binding["bundleSha256"] == bundle_sha
        and binding["bundleFileSha256"] == notice._sha256(bundle_file)
        and binding["tableSha256"]
        == notice._sha256(table)
        == APPROVED_LAYOUT_TABLE_SHA256
        and hook["phase"] == "hook"
        and hook["validation"]["status"] == "passed"
        and hook["source"] == png.name
        and hook["pngSha256"] == notice._sha256(png)
    ):
        raise Rejected("hook_layout_binding_failed")


def render_candidate(bundle, renderer, scratch, ffmpeg):
    artifacts = renderer.execute(
        bundle,
        export_preset="decision_clip",
        output_dir=scratch,
        basename="candidate",
        playback_evidence=True,
    )
    manifest = json.loads(artifacts.manifest.read_bytes())
    layout, evidence_files = notice._release_layout(
        manifest, artifacts.video, bundle.bundle_sha256
    )
    run_media_command(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(artifacts.video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return (
        artifacts.video,
        evidence_files,
        {
            "status": "passed",
            "videoSha256": layout["videoSha256"],
            "bundleSha256": bundle.bundle_sha256,
            "fullDecode": "passed",
            "layoutGeometry": layout,
            "mediaProbe": manifest["mediaProbe"],
            "loudness": manifest["audioFinishing"]["loudness"],
            "qrValidation": manifest["qrValidation"]["passed"],
            "playbackCheckpoints": len(manifest["playbackEvidence"]),
        },
    )


def qualify(
    row,
    prefix,
    directory,
    *,
    engine,
    solver,
    profile,
    proof_service,
    renderer,
    source,
    library,
    ffmpeg,
):
    stage = "input"
    try:
        exact = ExactStateNormalizerV1(engine.rules).normalize(prefix)
        state = exact.round_state
        row["exactStateId"] = exact.exact_state_id
        if not 2 <= len(engine.legal_actions(state)) <= 4:
            raise Rejected("G1_clarity")
        fixture, round_win = winning_fixture(
            exact, prefix, proof_service, canonical_sha256(profile), row["id"]
        )
        fixture_path = directory / "fixture.json"
        row["fixture"] = write_json(fixture_path, fixture)

        stage = "viewer_proof"
        proof = viewer.prove_fixture(
            fixture_path.read_bytes(),
            fixture_sha=row["fixture"]["sha256"],
            source_sha=source,
            maximum_allocations=4096,
            maximum_nodes=2_000_000,
            cp_work=1,
            backend="native",
            native_library=library,
        )
        row["viewerProof"] = write_json(directory / "viewer-proof.json", proof)
        inputs = exact_viewer_inputs(proof, rules_sha=engine.rules.sha256)

        stage = "input"
        final = notice.engine_round_result(engine.rules, fixture_path)
        points = final["roundScore"]["awardedPoints"]
        stage = "score"
        card, measurements = candidate_score(
            exact,
            proof,
            inputs,
            row["viewerProof"]["sha256"],
            round_win,
            abs(points[0] - points[1]),
            engine,
            solver,
            profile,
        )
        row["scorecard"] = card
        row["scoreInputs"] = measurements
        failures = [
            reason
            for reason in card["gateReasons"]
            if reason not in PUBLICATION_PENDING
        ]
        if failures:
            row.update(status="rejected", stage=stage, reasons=failures)
            return False

        stage = "input"
        bundle = notice.build_notice_bundle(
            engine.rules, profile, fixture=fixture_path, golden=row["id"]
        )
        notice.validate_notice_bundle(bundle, final, public=True, fixture=fixture_path)
        bundle_file = directory / "bundle.json"
        row["bundle"] = write_json(bundle_file, bundle.to_dict())

        stage = "hook_layout"
        hook = directory / "hook"
        try:
            check_hook(bundle_file, bundle.bundle_sha256, hook)
        finally:
            evidence = hook / "hook-layout-evidence.json"
            if evidence.is_file():
                row["hookLayout"] = {
                    "directory": hook.name,
                    "files": [
                        file_receipt(path)
                        for path in sorted(hook.iterdir())
                        if path.is_file()
                    ],
                }

        stage = "media_preflight"
        with TemporaryDirectory() as scratch:
            video, evidence_files, media = render_candidate(
                bundle, renderer, Path(scratch), ffmpeg
            )
            for path in evidence_files:
                shutil.copyfile(path, directory / path.name)
            row["fullLayout"] = [
                file_receipt(directory / path.name) for path in evidence_files
            ]
            row["mediaPreflight"] = write_json(directory / "media.json", media)
            row["video"] = file_receipt(shutil.move(video, directory / video.name))
        row.update(
            status="passed", stage=stage, reasons=["candidate_preflights_passed"]
        )
        return True
    except SearchBudgetExhausted:
        (directory / "candidate.mp4").unlink(missing_ok=True)
        row.update(status="rejected", stage=stage, reasons=["time_budget"])
        raise
    except (
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        reason = str(error) if isinstance(error, Rejected) else type(error).__name__
        row.update(status="rejected", stage=stage, reasons=[reason])
        return False


def generate(output, *, run_id, source, tree, library, rounds, target):
    output.mkdir(parents=True, exist_ok=False)
    index = {
        "schemaVersion": 1,
        "sourceCommit": source,
        "sourceTree": tree,
        "runId": run_id,
        "roundBudget": rounds,
        "target": target,
        "synthetic": True,
        "postingAuthorized": False,
        "publicationHistory": "NOT_CHECKED",
        "publicationQA": "NOT_CHECKED",
        "status": "failed",
        "stage": "setup",
        "reason": "generation_interrupted",
        "candidates": [],
        "qualified": [],
    }
    index_path = output / "candidate-index.json"
    write_json(index_path, index)
    qualified = index["qualified"]
    solver = None
    try:
        try:
            with search_deadline(budget_seconds()) as deadline:
                rules = RuleBook.load(viewer.RULES_PATH)
                engine = BalootEngine(rules)
                profile = notice.notice_profile(public=True)
                proof_service = UnifiedScenarioProofServiceV1.from_repository(
                    notice.ROOT
                )
                renderer = notice.RemotionProfessionalRenderer(
                    profile,
                    repository_root=notice.ROOT,
                    timeout_seconds=notice.render_timeout_seconds(),
                )
                ffmpeg = media_tool(notice.ROOT, "ffmpeg")
                media_tool(notice.ROOT, "ffprobe")
                solver = viewer.DoubleDummySolver(rules, library)
                index["stage"] = "search"
                write_json(index_path, index)
                search(
                    output,
                    index,
                    deadline,
                    index_path,
                    library,
                    engine,
                    solver,
                    profile,
                    proof_service,
                    renderer,
                    ffmpeg,
                )
        except SearchBudgetExhausted:
            index["stopReason"] = "time_budget"
        finally:
            if solver is not None:
                solver.close()
    except Exception as error:
        index["reason"] = type(error).__name__
        return 1
    else:
        if qualified:
            index["status"] = "passed"
            del index["reason"]
        else:
            index["reason"] = "no_qualified_candidates"
        return 0 if qualified else 1
    finally:
        write_json(index_path, index)


def search(
    output,
    index,
    deadline,
    index_path,
    library,
    engine,
    solver,
    profile,
    proof_service,
    renderer,
    ffmpeg,
):
    rng = random.Random(index["runId"])
    for trial in range(index["roundBudget"]):
        if time.monotonic() >= deadline:
            stop_search()
        seed = rng.getrandbits(64)
        contract = (Contract.SUN, Contract.HOKUM)[trial % 2]
        for prefix in synthetic_prefixes(engine, seed, contract):
            if time.monotonic() >= deadline:
                stop_search()
            play_index = len(prefix["playActions"])
            row = {
                "id": f"candidate-{seed:016x}-{play_index}",
                "seed": f"{seed:016x}",
                "contract": contract.value,
                "playIndex": play_index,
                "status": "rejected",
                "stage": "search",
                "reasons": ["generation_interrupted"],
            }
            index["candidates"].append(row)
            directory = output / row["id"]
            directory.mkdir()
            passed = qualify(
                row,
                prefix,
                directory,
                engine=engine,
                solver=solver,
                profile=profile,
                proof_service=proof_service,
                renderer=renderer,
                source=index["sourceCommit"],
                library=library,
                ffmpeg=ffmpeg,
            )
            if passed:
                index["qualified"].append(row["id"])
            write_json(index_path, index)
            if passed:
                break
        if len(index["qualified"]) == index["target"]:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    source, tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"],
        cwd=notice.ROOT,
        text=True,
        timeout=10,
    ).splitlines()
    if source != args.source or not 1 <= args.target <= args.rounds or args.run_id <= 0:
        parser.error("the checkout or batch inputs do not match the workflow run")
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=notice.ROOT,
        check=True,
        timeout=10,
    )
    return generate(
        args.out.resolve(),
        run_id=args.run_id,
        source=source,
        tree=tree,
        library=args.native_library,
        rounds=args.rounds,
        target=args.target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
