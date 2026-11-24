"""Build and verify a private research-only shadow day from a nightly pick."""

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from baloot_platform.research.slot_sequencer import (
    PlanError,
    SLOT_COUNT,
    load_gate,
    require,
    simulated_post,
)
from baloot_platform.video.media_validation import validate_loudness
from baloot_platform.video.render_bundle import (
    APPROVED_V4_NOTICE_SENTENCE_SHA256,
    RenderBundleV1,
)

from scripts.nightly_candidates import (
    Rejected,
    check_names,
    file_receipt,
    notice,
    write_json,
)
from scripts.pick_nightly_candidates import (
    SEAL,
    candidate_binding,
    history,
    verified_pool,
)


def checked_slots(reference):
    times = [datetime.fromisoformat(row["slotUtc"]) for row in reference["slots"]]
    require(
        len(times) == SLOT_COUNT
        and all(t.tzinfo == UTC for t in times)
        and times == sorted(set(times))
        and len({t.date() for t in times}) == 1,
        "INVALID_REFERENCE_SLOTS",
    )
    return [value.isoformat().replace("+00:00", "Z") for value in times]


def check_presentation(bundle):
    check_names(bundle["identity"])
    sentence = bundle["localizations"]["video.notice.sentence"]
    majlis = bundle["presentation"]["majlis"]
    presentation = bundle["presentation"]["presentation"]
    require(
        majlis["research_notice"] is True
        and majlis["theme"] == "red"
        and sha256(sentence.encode()).hexdigest() == APPROVED_V4_NOTICE_SENTENCE_SHA256,
        "INVALID_SHADOW_NOTICE",
    )
    require(
        presentation["show_justification"] is False
        and presentation["show_justification_text"] is False
        and all(
            cue.get("assetId") != "voice-explanation" for cue in bundle["audioCues"]
        ),
        "SHADOW_JUSTIFICATION_ENABLED",
    )
    require(RenderBundleV1(bundle).verify(), "INVALID_SHADOW_BUNDLE")


def check_audio(media):
    audio = media["mediaProbe"]["audio"]
    require(
        media["fullDecode"] == "passed" and audio["channels"] > 0,
        "SHADOW_AUDIO_MISSING",
    )
    validate_loudness(media["loudness"])


def check_viewer(proof, *, fixture_sha, source, rules_sha):
    require(
        proof["status"] == "exact_complete"
        and proof["coverageComplete"] is True
        and (proof["scenarioSha256"], proof["sourceCommit"], proof["rulesSha256"])
        == (fixture_sha, source, rules_sha)
        and type(proof["R"]) in (int, float)
        and 0.90 <= proof["R"] <= 1,
        "SHADOW_VIEWER_PROOF_MISMATCH",
    )


def verify_clip(directory, candidate, source):
    video = directory / "candidate.mp4"
    require(candidate["video"] == file_receipt(video), "SHADOW_VIDEO_HASH_MISMATCH")
    bundle_path = directory / "bundle.json"
    media_path = directory / "media.json"
    require(
        candidate["bundle"] == file_receipt(bundle_path)
        and candidate["mediaPreflight"] == file_receipt(media_path),
        "SHADOW_EVIDENCE_HASH_MISMATCH",
    )
    bundle = json.loads(bundle_path.read_bytes())
    media = json.loads(media_path.read_bytes())
    check_presentation(bundle)
    check_audio(media)
    viewer_path = directory / "viewer-proof.json"
    require(
        candidate["viewerProof"] == file_receipt(viewer_path)
        and candidate["fixture"] == file_receipt(directory / "fixture.json"),
        "SHADOW_VIEWER_EVIDENCE_HASH_MISMATCH",
    )
    viewer = json.loads(viewer_path.read_bytes())
    check_viewer(
        viewer,
        fixture_sha=candidate["fixture"]["sha256"],
        source=source,
        rules_sha=bundle["source"]["rulesHash"],
    )
    require(
        media["status"] == "passed"
        and media["videoSha256"] == candidate["video"]["sha256"]
        and media["bundleSha256"] == bundle["bundleSha256"],
        "SHADOW_MEDIA_BINDING_MISMATCH",
    )
    files = {
        row["file"]: {"name": row["file"], "sha256": row["sha256"]}
        for row in candidate["fullLayout"]
    }
    layout, _ = notice._release_layout(
        {
            "layoutGeometry": media["layoutGeometry"],
            "files": {
                "layoutTable": files["candidate-layout-table.json"],
                "layoutGeometry": files["candidate-layout.json"],
                **{
                    f"layout:{phase}": files[f"candidate-layout-{phase}.png"]
                    for phase in media["layoutGeometry"]["phases"]
                },
            },
        },
        video,
        bundle["bundleSha256"],
    )
    frames = json.loads((directory / "candidate-layout.json").read_bytes())["items"]
    require(
        len(frames) == len(layout["phases"])
        and {row["phase"] for row in frames} == set(layout["phases"])
        and all(row["roles"]["researchNotice"]["matchedCount"] == 1 for row in frames),
        "SHADOW_NOTICE_LAYOUT_MISSING",
    )
    return {
        "names": "PASS",
        "researchNotice": "PASS",
        "noJustification": "PASS",
        "viewerSeatAtLeast90Percent": "PASS",
        "viewerSeatRatio": viewer["R"],
        "viewerProofSha256": candidate["viewerProof"]["sha256"],
        "sound": "PASS",
        "loudness": media["loudness"],
        "videoSha256": media["videoSha256"],
        "bundleSha256": media["bundleSha256"],
        "layoutSha256": layout["evidenceSha256"],
    }


def combination_checks(
    gate, live, records, candidates, captions, slots, primaries, reserves
):
    known, last_eight, _, forbidden, chronology, unresolved = live
    checks = []
    complete_paths = 0
    history_keys = {
        "exactStateId": "exactStateId",
        "dealHash": "dealHash",
        "deal": "deal",
        "artifactSha256": "mediaSha256",
        "visibleOpenerSha256": "visibleOpenerSha256",
    }
    clip_expected = {}
    for choice in primaries + reserves:
        clip_id = choice["id"]
        binding = candidate_binding(candidates[clip_id], "")
        evidence = [
            row
            for row in records
            if any(
                row.get(key) == binding[value] for key, value in history_keys.items()
            )
        ]
        require(not evidence, "SHADOW_CLIP_IN_CANONICAL_LEDGER")
        clip_expected[clip_id] = {"canonicalLedgerNeverPosted": "PASS"}

    def scenarios(index, prior):
        nonlocal complete_paths
        if index == len(slots):
            complete_paths += 1
            return
        slot = slots[index]
        primary = primaries[index]
        stamp = datetime.fromisoformat(slot)
        simulated = [post for _, post in prior]
        actual = [*known, *simulated]
        projected = [*chronology, *((time, time, post) for time, post in prior)]
        latest, _ = gate.variety_windows(projected)
        used = last_eight | {post["captionSha256"] for post in simulated}
        outcomes = {}
        eligible = []
        caption = primary["captionSha256"]
        for choice in (primary, *reserves):
            key = choice["id"]
            binding = candidate_binding(candidates[key], caption)
            reasons, variety = gate.check_history(
                actual, used, latest, forbidden, binding, projected
            )
            parts = candidates[key]["scorecard"]["parts"]
            reasons += gate.check_score(
                sum(value for tag, value in parts.items() if tag != "V"),
                variety,
                parts["D"],
                parts["S"],
            )
            if key != primary["id"] and choice["mechanism"] == primary["mechanism"]:
                reasons.append("FALLBACK_MATCHES_PRIMARY_MECHANISM")
            caption_reasons = [
                reason for reason in reasons if reason == "RECENT_CAPTION_REPEAT"
            ]
            if caption not in captions:
                caption_reasons.append("CAPTION_POOL_OR_RETIRED_PHRASE")
            policy_reasons = [
                reason for reason in reasons if reason != "RECENT_CAPTION_REPEAT"
            ]
            reasons = policy_reasons + caption_reasons
            outcomes[key] = {
                "status": "HOLD" if reasons else "PASS",
                "reasons": reasons,
                "captionSha256": caption,
                "captionCheck": "HOLD" if caption_reasons else "PASS",
                "policyCCheck": "HOLD" if policy_reasons else "PASS",
                "policyCVariety": float(variety),
            }
            if not reasons:
                b = {**binding, "account": "shadow", "slotUtc": slot}
                route = SimpleNamespace(binding=b, mechanism=choice["mechanism"])
                eligible.append(simulated_post(gate, route, str(9 * 10**18 + index)))
        checks.append(
            {
                "slotUtc": slot,
                "priorSlotPostings": [
                    {"slotUtc": post["postedAtUtc"], "clipId": post["deal"]}
                    for post in simulated
                ],
                "candidates": outcomes,
            }
        )
        if [post["deal"] for post in simulated] == [
            pick["id"] for pick in primaries[:index]
        ]:
            require(
                outcomes[primary["id"]]["status"] == "PASS",
                "SHADOW_PLANNED_PRIMARY_FAILED_POLICY_C",
            )
        for post in eligible:
            scenarios(index + 1, (*prior, (stamp, post)))
        scenarios(index + 1, prior)

    scenarios(0, ())
    for clip_id, item in clip_expected.items():
        outcomes = [
            row["candidates"][clip_id] for row in checks if clip_id in row["candidates"]
        ]
        item["captionPoolLastEightRetiredPhrases"] = (
            "PASS"
            if any(outcome["captionCheck"] == "PASS" for outcome in outcomes)
            else "HOLD"
        )
        item["policyC"] = (
            "PASS"
            if any(outcome["policyCCheck"] == "PASS" for outcome in outcomes)
            else "HOLD"
        )
        item["eligibleInSlotCombination"] = (
            "PASS"
            if any(outcome["status"] == "PASS" for outcome in outcomes)
            else "HOLD"
        )
    return {
        "status": "NEVER_POST",
        "postingAuthorized": False,
        "checkedCanonicalPosts": len(known),
        "reachableCompletePaths": complete_paths,
        "unresolvedCanonicalRows": len(unresolved),
        "allCandidateChecksByReachablePriorPostingCombination": checks,
    }, clip_expected


def build_day(
    pick_path,
    ledger,
    reference,
    reference_sha,
    day_index,
    output,
    *,
    gate,
    caption_pool,
):
    require(
        not output.exists()
        and not any((parent / ".git").exists() for parent in (output, *output.parents)),
        "FRESH_PRIVATE_SHADOW_DIRECTORY_REQUIRED",
    )
    reference_raw = reference.read_bytes()
    require(
        sha256(reference_raw).hexdigest() == reference_sha,
        "REFERENCE_SLOT_DIGEST_MISMATCH",
    )
    reference_plan = json.loads(reference_raw)
    slots = checked_slots(reference_plan)
    require(ledger.resolve() == gate.LEDGER.resolve(), "CANONICAL_LEDGER_PATH_REQUIRED")
    ledger_raw = ledger.read_bytes()
    ledger_digest = sha256(ledger_raw).hexdigest()
    ledger_value = gate.parse(ledger_raw)
    inputs = gate.Inputs()
    live = history(gate, ledger_value, datetime.now(UTC), inputs)
    captions_raw = inputs.read(caption_pool, gate.CAPTION_POOL_SHA)
    captions = gate.parse(captions_raw)
    private = gate.parse(inputs.read(gate.PRIVATE, gate.PRIVATE_SHA))
    approved = {
        row["sha256"]
        for row in captions["captions"]
        if not gate.check_caption(row["text"], private, captions)
    }
    pick_raw = pick_path.read_bytes()
    pick = json.loads(pick_raw)
    require(
        pick["schemaVersion"] == 1
        and pick["status"] == "NEVER_POST"
        and pick["postingAuthorized"] is False
        and pick["ledgerSha256"] == ledger_digest
        and pick["captionPoolSha256"] == gate.CAPTION_POOL_SHA,
        "SHADOW_PICK_BINDING_MISMATCH",
    )
    pool = pick_path.parent / pick["pool"]
    index, seal = verified_pool(pool)
    require(
        (pick["poolSealSha256"], pick["runId"], pick["sourceCommit"])
        == (seal, index["runId"], index["sourceCommit"]),
        "SHADOW_POOL_BINDING_MISMATCH",
    )
    candidates = {row["id"]: row for row in index["candidates"]}
    primaries, reserves = pick["days"][day_index], pick["fallbacks"]
    choices = primaries + reserves
    require(
        len(primaries) == SLOT_COUNT
        and len(reserves) >= 2
        and len({row["id"] for row in choices}) == len(choices)
        and all(
            row["mechanism"] == candidates[row["id"]]["scoreInputs"]["mechanism"]
            for row in choices
        )
        and all(
            re.fullmatch(r"[0-9a-f]{64}", row["captionSha256"]) for row in primaries
        ),
        "INVALID_SHADOW_PICK",
    )

    combinations, policy = combination_checks(
        gate,
        live,
        ledger_value["records"],
        candidates,
        approved,
        slots,
        primaries,
        reserves,
    )
    combinations["ledgerSha256"] = ledger_digest
    combinations["captionPoolSha256"] = gate.CAPTION_POOL_SHA
    combinations["poolSealSha256"] = seal
    combinations["primaryPlan"] = [row["id"] for row in primaries]

    with TemporaryDirectory(prefix=".shadow-day-", dir=output.parent) as scratch:
        day = Path(scratch) / "day"
        day.mkdir()
        proof = {}
        for choice in choices:
            key = choice["id"]
            directory = pool / key
            proof[key] = {
                **verify_clip(directory, candidates[key], index["sourceCommit"]),
                **policy[key],
                "byteMatchToPoolSeal": "PASS",
            }
            video = day / f"{key}.mp4"
            shutil.copyfile(directory / "candidate.mp4", video)
            require(
                notice._sha256(video) == proof[key]["videoSha256"],
                "SHADOW_COPY_HASH_MISMATCH",
            )
        require(
            notice._sha256(ledger) == pick["ledgerSha256"]
            and notice._sha256(reference) == reference_sha,
            "SHADOW_INPUT_CHANGED",
        )
        combinations_receipt = write_json(
            day / "SLOT-COMBINATION-CHECKS.json", combinations
        )
        write_json(
            day / "shadow-day.json",
            {
                "schemaVersion": 1,
                "status": "NEVER_POST",
                "postingAuthorized": False,
                "platform": "TikTok",
                "runId": index["runId"],
                "sourceCommit": index["sourceCommit"],
                "builderSha256": notice._sha256(Path(__file__)),
                "poolSealSha256": seal,
                "ledgerSha256": ledger_digest,
                "captionPoolSha256": gate.CAPTION_POOL_SHA,
                "slotCombinationChecks": combinations_receipt,
                "qaStatus": "PASS"
                if all(
                    item["eligibleInSlotCombination"] == "PASS"
                    for item in policy.values()
                )
                else "HOLD",
                "pickSha256": sha256(pick_raw).hexdigest(),
                "referenceSlotManifestSha256": reference_sha,
                "slots": [
                    {
                        "slotUtc": slot,
                        "id": row["id"],
                        "mechanism": row["mechanism"],
                        "captionSha256": row["captionSha256"],
                    }
                    for slot, row in zip(slots, primaries, strict=True)
                ],
                "reserves": [
                    {"id": row["id"], "mechanism": row["mechanism"]} for row in reserves
                ],
                "qa": proof,
            },
        )
        (day / SEAL).write_text(
            "".join(
                f"{notice._sha256(path)}  {path.name}\n"
                for path in sorted(day.iterdir())
            ),
            encoding="ascii",
            newline="\n",
        )
        day.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "pick",
        "ledger",
        "reference",
        "output",
        "gate",
        "policy",
        "caption-pool",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--reference-sha", required=True)
    parser.add_argument("--day-index", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    try:
        build_day(
            args.pick,
            args.ledger,
            args.reference,
            args.reference_sha,
            args.day_index,
            args.output.resolve(),
            gate=load_gate(args.gate, args.policy),
            caption_pool=args.caption_pool,
        )
    except PlanError as error:
        print(f"SHADOW_HELD {error.reason}")
        return 1
    except (Rejected, ValueError, OSError, KeyError, IndexError, TypeError):
        print("SHADOW_HELD invalid media evidence or local input")
        return 1
    print("SHADOW_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
