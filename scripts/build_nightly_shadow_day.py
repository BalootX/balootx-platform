"""Build and verify a private research-only shadow day from a nightly pick."""

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from baloot_platform.research.slot_sequencer import PlanError, SLOT_COUNT, require
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
from scripts.pick_nightly_candidates import SEAL, verified_pool


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
    require(
        bundle["presentation"]["majlis"]["research_notice"] is True
        and sha256(sentence.encode()).hexdigest() == APPROVED_V4_NOTICE_SENTENCE_SHA256,
        "INVALID_SHADOW_NOTICE",
    )
    require(RenderBundleV1(bundle).verify(), "INVALID_SHADOW_BUNDLE")


def check_audio(media):
    audio = media["mediaProbe"]["audio"]
    require(
        media["fullDecode"] == "passed" and audio["channels"] > 0,
        "SHADOW_AUDIO_MISSING",
    )
    validate_loudness(media["loudness"])


def verify_clip(directory, candidate):
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
        "sound": "PASS",
        "loudness": media["loudness"],
        "videoSha256": media["videoSha256"],
        "bundleSha256": media["bundleSha256"],
        "layoutSha256": layout["evidenceSha256"],
    }


def build_day(pick_path, ledger, reference, reference_sha, day_index, output):
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
    pick_raw = pick_path.read_bytes()
    pick = json.loads(pick_raw)
    require(
        pick["schemaVersion"] == 1
        and pick["status"] == "NEVER_POST"
        and pick["postingAuthorized"] is False
        and pick["ledgerSha256"] == notice._sha256(ledger),
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

    with TemporaryDirectory(prefix=".shadow-day-", dir=output.parent) as scratch:
        day = Path(scratch) / "day"
        day.mkdir()
        proof = {}
        for choice in choices:
            key = choice["id"]
            directory = pool / key
            proof[key] = verify_clip(directory, candidates[key])
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
        write_json(
            day / "shadow-day.json",
            {
                "schemaVersion": 1,
                "status": "NEVER_POST",
                "postingAuthorized": False,
                "runId": index["runId"],
                "sourceCommit": index["sourceCommit"],
                "builderSha256": notice._sha256(Path(__file__)),
                "poolSealSha256": seal,
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
    for key in ("pick", "ledger", "reference", "output"):
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
