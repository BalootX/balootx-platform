import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import build_nightly_shadow_day as shadow
from scripts import nightly_candidates as nightly
from scripts import pick_nightly_candidates as picker


@pytest.fixture(scope="module")
def sample(rules, tmp_path_factory):
    engine = nightly.BalootEngine(rules)
    profile = nightly.notice.notice_profile(public=True)
    profile["identity"] = nightly.synthetic_identity()
    source = "a" * 40
    seed = 140250512009021371
    prefixes = list(nightly.synthetic_prefixes(engine, seed, nightly.Contract.SUN))
    assert prefixes == list(
        nightly.synthetic_prefixes(engine, seed, nightly.Contract.SUN)
    )
    prefix = next(row for row in prefixes if len(row["playActions"]) == 25)
    exact = nightly.ExactStateNormalizerV1(rules).normalize(prefix)
    service = nightly.UnifiedScenarioProofServiceV1.from_repository(nightly.notice.ROOT)
    scenario_id = "synthetic-candidate"
    fixture, wins = nightly.winning_fixture(
        exact, prefix, service, nightly.canonical_sha256(profile), scenario_id
    )
    assert fixture["expected"]["selectedAction"] == "HA"
    assert wins == {"HK": -1, "HA": 1}
    raw = nightly.viewer.receipt_bytes(fixture)
    digest = nightly.write_json(
        tmp_path_factory.mktemp("scenario") / "fixture.json", fixture
    )["sha256"]
    receipt = nightly.viewer.prove_fixture(
        raw,
        fixture_sha=digest,
        source_sha=source,
        maximum_allocations=4096,
        maximum_nodes=2_000_000,
        cp_work=1,
        backend="ortools",
    )
    assert receipt["status"] == "exact_complete" and receipt["R"] == 1
    margins, _ = nightly.viewer.cp_sat_values(engine, exact.round_state, 2_000_000, 1)
    return SimpleNamespace(
        engine=engine,
        profile=profile,
        source=source,
        prefix=prefix,
        exact=exact,
        service=service,
        scenario_id=scenario_id,
        raw=raw,
        digest=digest,
        receipt=receipt,
        margins=margins,
    )


def test_viewer_score_requires_exact_status_and_matching_rules(sample, rules):
    assert (
        nightly.exact_viewer_inputs(sample.receipt, rules_sha=rules.sha256)
        == sample.receipt["scorerInputs"]
    )
    with pytest.raises(nightly.Rejected, match="viewer_proof_binding_failed"):
        nightly.exact_viewer_inputs(sample.receipt, rules_sha="d" * 64)
    with pytest.raises(nightly.Rejected, match="maximum_nodes"):
        nightly.exact_viewer_inputs(
            {"status": "INCOMPLETE", "reason": "maximum_nodes"}, rules_sha=rules.sha256
        )


@pytest.mark.parametrize(
    "failure", ["input", "hook_layout", "media_preflight", "time_budget", None]
)
def test_only_passing_media_enters_the_artifact(tmp_path, monkeypatch, sample, failure):
    def prove(raw, **options):
        assert (raw, options["fixture_sha"], options["source_sha"]) == (
            sample.raw,
            sample.digest,
            sample.source,
        )
        return sample.receipt

    def hook(bundle_file, bundle_sha, output):
        assert json.loads(bundle_file.read_bytes())["bundleSha256"] == bundle_sha
        nightly.write_json(output / "hook-layout-evidence.json", {"synthetic": True})
        if failure == "hook_layout":
            raise subprocess.CalledProcessError(1, "synthetic-hook")

    def media(bundle, renderer, scratch, ffmpeg):
        video = scratch / "candidate.mp4"
        video.write_bytes(b"synthetic media")
        if failure == "media_preflight":
            raise ValueError("synthetic failed decode")
        if failure == "time_budget":
            raise nightly.SearchBudgetExhausted
        return video, [], {"bundleSha256": bundle.bundle_sha256, "status": "passed"}

    monkeypatch.setattr(nightly.viewer, "prove_fixture", prove)
    monkeypatch.setattr(nightly, "check_hook", hook)
    monkeypatch.setattr(nightly, "render_candidate", media)
    solver = SimpleNamespace(
        solve=lambda *args, **kwargs: {
            "status": "exact",
            "rootActions": [
                {"action": key, "value": value} for key, value in sample.margins.items()
            ],
        }
    )
    prefix = sample.prefix if failure != "input" else {**sample.prefix, "deckOrder": []}
    row = {"id": sample.scenario_id}

    def qualify():
        return nightly.qualify(
            row,
            prefix,
            tmp_path,
            engine=sample.engine,
            solver=solver,
            profile=sample.profile,
            proof_service=sample.service,
            renderer=None,
            source=sample.source,
            library=None,
            ffmpeg=None,
        )

    if failure == "time_budget":
        with pytest.raises(nightly.SearchBudgetExhausted):
            qualify()
        assert row["status"] == "rejected" and row["reasons"] == ["time_budget"]
        assert row["stage"] == "media_preflight"
        assert not (tmp_path / "candidate.mp4").exists()
        return
    passed = qualify()

    assert passed is (failure is None)
    assert (tmp_path / "candidate.mp4").exists() is passed
    assert row["status"] == ("passed" if passed else "rejected")
    assert row["stage"] == (failure or "media_preflight")
    if failure != "input":
        assert row["scorecard"]["score"] >= 60
        assert row["scorecard"]["postAllowed"] is False
        assert row["scorecard"]["exactActorProofSha256"] == row["viewerProof"]["sha256"]
        assert {entry["file"] for entry in row["hookLayout"]["files"]} == {
            "hook-layout-evidence.json"
        }
    if passed:
        assert row["video"] == nightly.file_receipt(tmp_path / "candidate.mp4")


def test_hook_check_requires_the_receipt_for_the_exact_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.json"
    bundle.write_bytes(b"synthetic bundle")
    hook = tmp_path / "hook"
    table = nightly.REMOTION / "scripts/approved-layout-adjacencies.json"
    receipt = {
        "scope": "hook",
        "status": "stills_passed",
        "postingAuthorized": False,
        "binding": {
            "mode": "qualification",
            "bundleSha256": "a" * 64,
            "bundleFileSha256": nightly.file_receipt(bundle)["sha256"],
            "tableSha256": nightly.APPROVED_LAYOUT_TABLE_SHA256,
        },
    }

    def run(args, **kwargs):
        assert args[2:] == ["--check-hook", str(bundle), str(hook)]
        hook.mkdir(exist_ok=True)
        (hook / "layout-table.json").write_bytes(table.read_bytes())
        png = hook / "layout-hook.png"
        png.write_bytes(b"synthetic png")
        receipt["items"] = [
            {
                "phase": "hook",
                "source": png.name,
                "pngSha256": nightly.file_receipt(png)["sha256"],
                "validation": {"status": "passed"},
            }
        ]
        nightly.write_json(hook / "hook-layout-evidence.json", receipt)

    monkeypatch.setattr(nightly.subprocess, "run", run)
    nightly.check_hook(bundle, "a" * 64, hook)
    bundle.write_bytes(b"changed")
    with pytest.raises(nightly.Rejected, match="hook_layout_binding_failed"):
        nightly.check_hook(bundle, "a" * 64, hook)


@pytest.fixture
def batch(tmp_path, monkeypatch, rules):
    output = tmp_path / "candidates"
    monkeypatch.delenv("BALOOT_NIGHTLY_BUDGET_SECONDS", raising=False)
    monkeypatch.delenv("BALOOT_NIGHTLY_JOB_DEADLINE", raising=False)
    monkeypatch.setattr(nightly, "signal", SimpleNamespace())
    monkeypatch.setattr(nightly.RuleBook, "load", lambda path: rules)
    monkeypatch.setattr(nightly.notice, "notice_profile", lambda **kwargs: {})
    monkeypatch.setattr(
        nightly.UnifiedScenarioProofServiceV1, "from_repository", lambda path: None
    )
    monkeypatch.setattr(nightly, "media_tool", lambda *args: None)
    monkeypatch.setattr(
        nightly.viewer,
        "DoubleDummySolver",
        lambda *args: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        nightly,
        "synthetic_prefixes",
        lambda *args: [
            {"playActions": []},
            {"playActions": [{}]},
        ],
    )

    def run(*, renderer_root=None):
        monkeypatch.setattr(
            nightly.notice,
            "RemotionProfessionalRenderer",
            lambda *args, **kwargs: SimpleNamespace(renderer_root=renderer_root),
        )
        if renderer_root is None:
            monkeypatch.setattr(nightly, "check_renderer_runtime", lambda *args: None)
        code = nightly.generate(
            output,
            run_id=1,
            source="a" * 40,
            tree="b" * 40,
            library=None,
            rounds=3,
            target=2,
        )
        return code, json.loads((output / "candidate-index.json").read_bytes())

    return run


@pytest.mark.parametrize("qualified", [False, True])
def test_budget_stops_before_the_next_candidate(batch, monkeypatch, qualified):
    clock = 0
    attempts = []
    monkeypatch.setattr(nightly, "time", SimpleNamespace(monotonic=lambda: clock))
    assert nightly.budget_seconds() == 18 * 60

    def qualify(row, *args, **kwargs):
        nonlocal clock
        attempts.append(row["id"])
        clock += 18 * 60
        row["status"] = "passed" if qualified else "rejected"
        return qualified

    monkeypatch.setattr(nightly, "qualify", qualify)
    code, index = batch()
    assert len(attempts) == 1
    assert code == (0 if qualified else 1)
    assert index["status"] == ("passed" if qualified else "failed")
    assert index["stopReason"] == "time_budget"
    assert index["qualified"] == (attempts if qualified else [])


def test_job_deadline_reserves_cleanup_time(monkeypatch):
    monkeypatch.setenv("BALOOT_NIGHTLY_BUDGET_SECONDS", "20")
    monkeypatch.setenv("BALOOT_NIGHTLY_JOB_DEADLINE", "130")
    monkeypatch.setattr(nightly, "time", SimpleNamespace(time=lambda: 100))
    assert nightly.budget_seconds() == 15
    monkeypatch.setenv("BALOOT_NIGHTLY_BUDGET_SECONDS", "10")
    assert nightly.budget_seconds() == 10


@pytest.mark.parametrize("stage", ["setup", "search"])
def test_unexpected_failure_finalizes_index(batch, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise AssertionError("synthetic failure")

    if stage == "setup":
        monkeypatch.setattr(nightly, "media_tool", fail)
    else:
        monkeypatch.setattr(nightly, "qualify", fail)
    code, index = batch()
    assert code == 1
    assert index["status"] == "failed"
    assert index["stage"] == stage
    assert index["reason"] == "AssertionError"


def test_r8_synthetic_fixtures_replay_to_the_exact_expected_scores(rules):
    from baloot_platform.research.lesson_tags import classify_public_mechanism_v1

    fixture_dir = nightly.notice.ROOT / "examples" / "research-r8"
    fixtures = sorted(fixture_dir.glob("*.json"))
    assert len(fixtures) == 10
    profile = nightly.notice.notice_profile(public=True)
    normalizer = nightly.ExactStateNormalizerV1(rules)
    deal_hashes = set()
    families = set()

    for fixture in fixtures:
        data = json.loads(fixture.read_bytes())
        expected = data["expected"]
        state = normalizer.normalize(data["exactTracePrefix"])
        assert state.next_actor == expected["actingSeat"]
        assert state.state["buyer"] == expected["buyer"]
        assert [row["code"] for row in state.legal_actions] == expected[
            "legalRootActions"
        ]
        mechanism = classify_public_mechanism_v1(
            rules=rules,
            current_trick=state.state["currentTrick"] or {"plays": []},
            contract=state.state["contract"],
            trump=state.state.get("trumpSuit"),
            action=expected["selectedAction"],
        )
        assert mechanism == data["rulesMeaning"]["mechanism"]
        assert "caption" not in data
        families.add(data["scenarioFamily"])
        deal_hashes.add(state.round_state.shuffle.deck_commitment)

        final = nightly.notice.engine_round_result(rules, fixture)
        assert all(
            final["roundScore"][key] == value
            for key, value in expected["roundScore"].items()
        )
        bundle = nightly.notice.build_notice_bundle(
            rules, profile, fixture=fixture, golden=data["scenarioId"]
        )
        nightly.notice.validate_notice_bundle(
            bundle, final, public=True, fixture=fixture
        )

    assert len(deal_hashes) == len(families) == len(fixtures)


def test_nightly_identity_rejects_generic_labels():
    seed = "synthetic-fixture"
    names = nightly.notice.seeded_player_names(nightly.synthetic_identity(), seed)
    nightly.check_names({"gameSerial": seed, "players": names})
    with pytest.raises(nightly.Rejected, match="generic_or_unapproved_player_names"):
        nightly.check_names(
            {
                "gameSerial": seed,
                "players": {**names, "north": "Player 01"},
            }
        )


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0"], ids=["stale", "matching"])
def test_nightly_checks_renderer_runtime_before_search(
    batch, tmp_path, monkeypatch, version
):
    monkeypatch.setattr(nightly, "REMOTION", tmp_path)
    package = tmp_path / "node_modules" / "renderer"
    package.mkdir(parents=True)
    lock = tmp_path / "package-lock.json"
    lock.write_text(
        json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {"node_modules/renderer": {"version": "2.0.0"}},
            }
        ),
        encoding="utf-8",
    )
    (package / "package.json").write_text(
        json.dumps({"version": version}), encoding="utf-8"
    )
    calls = []

    def search(_output, index, *_args):
        calls.append("search")
        index["qualified"].append("synthetic")

    monkeypatch.setattr(nightly, "media_tool", lambda *_: calls.append("media"))
    monkeypatch.setattr(nightly, "search", search)

    code, index = batch(renderer_root=tmp_path)

    if version == "1.0.0":
        assert code == 1
        assert index["stage"] == "setup"
        assert index["reason"] == "stale_render_runtime"
        assert calls == []
    else:
        assert code == 0
        assert index["qualified"] == ["synthetic"]
        assert calls == ["media", "media", "search"]


def test_early_caption_selection_excludes_recent_unapproved_and_retired():
    def check_caption(text, private, pool):
        if text == "unapproved":
            return ["CAPTION_NOT_IN_APPROVED_POOL"]
        return ["BANNED_CAPTION_PHRASE"] if text in private else []

    records = [{"text": f"approved-{i}", "sha256": f"id-{i}"} for i in range(10)]
    records += [{"text": text, "sha256": text} for text in ("unapproved", "retired")]
    gate = SimpleNamespace(check_caption=check_caption)
    result = picker.caption_options(
        gate, {"captions": records}, {"retired"}, {"id-0", "id-1"}
    )
    assert result == [f"id-{i}" for i in range(2, 10)]
    with pytest.raises(picker.PlanError, match="INSUFFICIENT_APPROVED_CAPTIONS"):
        picker.caption_options(
            gate, {"captions": records}, {"retired"}, {"id-0", "id-1", "id-2"}
        )


def test_picker_applies_live_variety_before_filling_two_days():
    live_history = [{"mechanism": "lead", "scenarioFamily": "used-family"}]

    def check_history(known, captions, latest, forbidden, binding, chronology):
        assert known is latest is chronology is live_history
        assert captions == {"recent"} and forbidden == []
        if binding["opener"]["family"] == latest[-1]["scenarioFamily"]:
            return ["CONSECUTIVE_FAMILY"], 0
        return (
            (["CONSECUTIVE_MECHANISM"], 0)
            if binding["opener"]["mechanism"] == latest[-1]["mechanism"]
            else ([], 15)
        )

    gate = SimpleNamespace(
        variety_windows=lambda chronology: (chronology, chronology),
        check_history=check_history,
        check_score=lambda base, variety, *args: ["SCORE_BELOW_60"]
        if base + variety < 60
        else [],
    )
    rows = [
        {
            "id": str(i),
            "video": {"sha256": "test"},
            "scoreInputs": {
                "mechanism": ("lead", "ruff", "discard")[i % 3],
                "scenarioFamily": "used-family" if i == 1 else f"fresh-{i}",
                "visibleOpenerSha256": str(i),
            },
            "scorecard": {
                "parts": {"D": 25, "S": 10, "R": 10, "C": 10, "V": 15},
                "difficultyPrior": 50,
                "mechanismFrequency": 0,
                "exactStateId": str(i),
            },
        }
        for i in range(18)
    ]
    days, fallbacks = picker.select(
        gate,
        rows,
        [str(i) for i in range(8)],
        (live_history, {"recent"}, [], [], live_history, []),
    )
    picks = days[0] + days[1]
    assert list(map(len, days)) == [4, 4]
    assert all(
        item["id"] != "1" and item["mechanism"] != "lead" for item in picks + fallbacks
    )
    assert all(
        left["mechanism"] != right["mechanism"] for left, right in zip(picks, picks[1:])
    )
    assert len({item["id"] for item in picks + fallbacks}) == len(picks + fallbacks)


def test_sealed_pool_binds_media_and_exact_run(tmp_path):
    shards = tmp_path / "shards"
    shard = shards / "nightly-candidate-shard-0"
    rows = []
    for i in range(nightly.MINIMUM_POOL_SIZE):
        key = f"candidate-{i:016x}-24"
        folder = shard / key
        folder.mkdir(parents=True)
        video = folder / "candidate.mp4"
        video.write_bytes(bytes([i]))
        rows.append(
            {
                "id": key,
                "status": "passed",
                "video": nightly.file_receipt(video),
                "scoreInputs": {
                    "mechanism": i % 3,
                    **{
                        field: str(i)
                        for field in (
                            "exactStateId",
                            "dealHash",
                            "scenarioFamily",
                            "visibleOpenerSha256",
                        )
                    },
                },
            }
        )
    source, tree, run_id = "a" * 40, "b" * 40, 123
    nightly.write_json(
        shard / "candidate-index.json",
        {
            "status": "passed",
            "postingAuthorized": False,
            "sourceCommit": source,
            "sourceTree": tree,
            "runId": run_id,
            "qualified": [row["id"] for row in rows],
            "candidates": rows,
        },
    )
    output = tmp_path / "pool"
    nightly.seal_pool(shards, output, source=source, tree=tree, run_id=run_id)
    index, seal = picker.verified_pool(
        output, {"databaseId": run_id, "headSha": source}
    )
    assert index["status"] == "NEVER_POST" and len(index["qualified"]) == 14
    assert len(seal) == 64
    with pytest.raises(picker.PlanError, match="POOL_RUN_BINDING_MISMATCH"):
        picker.verified_pool(output, {"databaseId": run_id + 1, "headSha": source})
    (output / rows[0]["id"] / "candidate.mp4").write_bytes(b"tampered")
    with pytest.raises(picker.PlanError, match="POOL_HASH_MISMATCH"):
        picker.verified_pool(output)


def test_shadow_checks_approved_names_and_notice(monkeypatch):
    seed = "shadow-test"
    names = nightly.notice.seeded_player_names(nightly.synthetic_identity(), seed)
    monkeypatch.setattr(shadow.RenderBundleV1, "verify", lambda self: True)
    bundle = {
        "identity": {"gameSerial": seed, "players": names},
        "localizations": {
            "video.notice.sentence": nightly.notice.notice_profile(public=True)[
                "localizations"
            ]["video.notice.sentence"]
        },
        "presentation": {
            "majlis": {"research_notice": True, "theme": "red"},
            "presentation": {
                "show_justification": False,
                "show_justification_text": False,
            },
        },
        "audioCues": [],
    }
    shadow.check_presentation(bundle)
    with pytest.raises(nightly.Rejected, match="generic_or_unapproved_player_names"):
        shadow.check_presentation(
            {
                **bundle,
                "identity": {
                    "gameSerial": seed,
                    "players": {**names, "north": "Player 01"},
                },
            }
        )
    bundle["presentation"]["majlis"]["research_notice"] = False
    with pytest.raises(picker.PlanError, match="INVALID_SHADOW_NOTICE"):
        shadow.check_presentation(bundle)


def test_shadow_rejects_missing_or_out_of_range_sound():
    from baloot_platform.video.media_validation import RELEASE_TARGET_LUFS

    media = {
        "fullDecode": "passed",
        "mediaProbe": {"audio": {"channels": 2}},
        "loudness": {
            "integratedLufs": RELEASE_TARGET_LUFS,
            "targetLufs": RELEASE_TARGET_LUFS,
            "truePeakDbtp": -2.0,
        },
    }
    shadow.check_audio(media)
    media["loudness"]["truePeakDbtp"] = -0.5
    with pytest.raises(ValueError, match="governed audio loudness"):
        shadow.check_audio(media)
    media["mediaProbe"]["audio"]["channels"] = 0
    with pytest.raises(picker.PlanError, match="SHADOW_AUDIO_MISSING"):
        shadow.check_audio(media)


def test_shadow_slots_require_one_ordered_utc_day():
    slots = [{"slotUtc": f"2026-01-01T{hour:02}:00:00Z"} for hour in (9, 12, 15, 18)]
    assert shadow.checked_slots({"slots": slots}) == [
        entry["slotUtc"] for entry in slots
    ]
    with pytest.raises(picker.PlanError, match="INVALID_REFERENCE_SLOTS"):
        shadow.checked_slots({"slots": [slots[0], slots[0], *slots[2:]]})


def test_shadow_combination_proof_checks_prior_fallback_and_never_post(monkeypatch):
    from datetime import UTC, datetime

    def record(gate, route, post_id):
        return {
            "deal": route.binding["deal"],
            "mechanism": route.mechanism,
            "captionSha256": route.binding["captionSha256"],
            "postedAtUtc": route.binding["slotUtc"],
        }

    def check_history(known, used, latest, forbidden, binding, chronology):
        reasons = []
        if binding["opener"]["mechanism"] == latest[-1]["mechanism"]:
            reasons.append("CONSECUTIVE_MECHANISM")
        if any(row["deal"] == binding["deal"] for row in known):
            reasons.append("LIFETIME_DEAL_REPEAT")
        return reasons, 15

    monkeypatch.setattr(shadow, "simulated_post", record)
    gate = SimpleNamespace(
        variety_windows=lambda chronology: ([x[2] for x in chronology], []),
        check_history=check_history,
        check_score=lambda *args: [],
    )
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    past = {"deal": "past", "mechanism": "A"}
    live = ([past], set(), [past], [], [(stamp, stamp, past)], [])
    picks = [
        {"id": "first", "mechanism": "B", "captionSha256": "one"},
        {"id": "second", "mechanism": "C", "captionSha256": "two"},
    ]
    reserves = [{"id": "reserve", "mechanism": "C"}]
    candidates = {
        item["id"]: {
            "id": item["id"],
            "scoreInputs": {
                "mechanism": item["mechanism"],
                "scenarioFamily": item["id"],
                "visibleOpenerSha256": "opener",
                "dealHash": "deal",
                "exactStateId": "state",
            },
            "video": {"sha256": "media"},
            "scorecard": {"parts": {"S": 10, "D": 20}},
        }
        for item in picks + reserves
    }
    slots = ["2026-01-01T09:00:00Z", "2026-01-01T12:00:00Z"]
    combo, clips = shadow.combination_checks(
        gate, live, [], candidates, {"one", "two"}, slots, picks, reserves
    )
    branches = combo["allCandidateChecksByReachablePriorPostingCombination"]
    fallback = next(
        row
        for row in branches
        if row["priorSlotPostings"] == [{"slotUtc": slots[0], "clipId": "reserve"}]
    )
    assert fallback["candidates"]["second"]["captionCheck"] == "PASS"
    assert fallback["candidates"]["second"]["policyCCheck"] == "HOLD"
    assert clips["reserve"]["eligibleInSlotCombination"] == "PASS"
    with pytest.raises(picker.PlanError, match="SHADOW_CLIP_IN_CANONICAL_LEDGER"):
        shadow.combination_checks(
            gate,
            live,
            [{"artifactSha256": "media"}],
            candidates,
            {"one", "two"},
            slots,
            picks,
            reserves,
        )


def test_shadow_viewer_requires_complete_exact_receipt_at_ninety_percent():
    proof = {
        "status": "exact_complete",
        "coverageComplete": True,
        "scenarioSha256": "fixture",
        "sourceCommit": "source",
        "rulesSha256": "rules",
        "R": 0.90,
    }
    binding = dict(fixture_sha="fixture", source="source", rules_sha="rules")
    shadow.check_viewer(proof, **binding)
    with pytest.raises(picker.PlanError, match="SHADOW_VIEWER_PROOF_MISMATCH"):
        shadow.check_viewer({**proof, "R": 0.8999}, **binding)
    with pytest.raises(picker.PlanError, match="SHADOW_VIEWER_PROOF_MISMATCH"):
        shadow.check_viewer({**proof, "scenarioSha256": "another fixture"}, **binding)


def test_shadow_day_holds_with_unresolved_canonical_history():
    policy = {"clip": {"eligibleInSlotCombination": "PASS"}}
    assert shadow.day_qa_blockers({"unresolvedCanonicalRows": 0}, policy) == []
    assert shadow.day_qa_blockers({"unresolvedCanonicalRows": 1}, policy) == [
        "LEDGER_HISTORY_UNRESOLVED"
    ]
