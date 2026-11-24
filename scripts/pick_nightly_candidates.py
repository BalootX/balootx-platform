"""Pick two days of NEVER_POST candidates against PC-local publication inputs."""

import argparse
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from baloot_platform.research.calibration.score import engine_interest_rank_key_v1
from baloot_platform.research.slot_sequencer import (
    PlanError,
    SLOT_COUNT,
    load_gate,
    require,
    write_handoff,
)

from scripts.nightly_candidates import CANDIDATE_ID, MINIMUM_POOL_SIZE, notice


ARTIFACT = "nightly-synthetic-candidates"
WORKFLOW = "nightly-candidates.yml"
SEAL = "SHA256SUMS.txt"
INDEX = "candidate-index.json"
PICKS = 2 * SLOT_COUNT
MAXIMUM_POOL_AGE = timedelta(hours=36)


def gh(*args, timeout=30):
    # An explicit PC-local credential avoids falling back to a user's CLI login.
    require(bool(os.environ.get("GH_TOKEN")), "ARTIFACT_READ_CREDENTIAL_REQUIRED")
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        check=True,
        timeout=timeout,
    )
    return result.stdout


def download_pool(repo, output, now):
    runs = json.loads(
        gh(
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            WORKFLOW,
            "--branch",
            "main",
            "--event",
            "schedule",
            "--status",
            "success",
            "--limit",
            "1",
            "--json",
            "databaseId,headSha,createdAt",
        )
    )
    require(len(runs) == 1, "SCHEDULED_POOL_UNAVAILABLE")
    run = runs[0]
    require(
        timedelta()
        <= now - datetime.fromisoformat(run["createdAt"])
        <= MAXIMUM_POOL_AGE,
        "SCHEDULED_POOL_STALE",
    )
    gh(
        "run",
        "download",
        str(run["databaseId"]),
        "--repo",
        repo,
        "--name",
        ARTIFACT,
        "--dir",
        str(output),
        timeout=180,
    )
    return run


def verified_pool(directory, run=None):
    seal = (directory / SEAL).read_bytes()
    paths = set()
    for line in seal.decode("ascii").splitlines():
        digest, path = line.split("  ", 1)
        parts = PurePosixPath(path).parts
        require(
            re.fullmatch("[0-9a-f]{64}", digest)
            and path == "/".join(parts)
            and all(
                re.fullmatch(r"[a-zA-Z0-9_.-]+", part) and part not in {".", ".."}
                for part in parts
            )
            and path not in paths
            and path != SEAL,
            "INVALID_POOL_SEAL",
        )
        member = directory / path
        require(
            member.resolve().is_relative_to(directory.resolve()), "INVALID_POOL_PATH"
        )
        require(
            not member.is_symlink() and notice._sha256(member) == digest,
            "POOL_HASH_MISMATCH",
        )
        paths.add(path)
    require(
        paths
        == {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if p.is_file() and p != directory / SEAL
        }
        and INDEX in paths,
        "POOL_SEAL_COVERAGE_MISMATCH",
    )
    index = json.loads((directory / INDEX).read_bytes())
    require(
        index["schemaVersion"] == 1
        and index["status"] == "NEVER_POST"
        and index["postingAuthorized"] is False
        and index["synthetic"] is True
        and (
            run is None
            or (index["runId"], index["sourceCommit"])
            == (run["databaseId"], run["headSha"])
        ),
        "POOL_RUN_BINDING_MISMATCH",
    )
    rows = index["candidates"]
    ids = [row["id"] for row in rows]
    require(
        len(ids) >= MINIMUM_POOL_SIZE
        and ids == index["qualified"]
        and len(ids) == len(set(ids))
        and all(
            CANDIDATE_ID.fullmatch(row["id"])
            and row["status"] == "passed"
            and row["video"]
            == {
                "file": "candidate.mp4",
                "sha256": notice._sha256(directory / row["id"] / "candidate.mp4"),
            }
            for row in rows
        ),
        "POOL_CANDIDATE_BINDING_MISMATCH",
    )
    return index, sha256(seal).hexdigest()


def caption_options(gate, pool, private, recent_captions):
    available = []
    for row in pool["captions"]:
        if row["sha256"] in recent_captions or gate.check_caption(
            row["text"], private, pool
        ):
            continue
        available.append(row["sha256"])
    require(len(set(available)) >= PICKS, "INSUFFICIENT_APPROVED_CAPTIONS")
    return available


def history(gate, ledger, now, inputs):
    records = ledger["records"]
    known = [row for row in records if row["status"] in gate.POST_STATUSES]
    accounts = {
        urlsplit(row["url"]).path.split("/")[1].removeprefix("@")
        for row in known
        if row["platform"] == "TikTok" and row["status"] == gate.CONFIRMED
    }
    require(len(accounts) == 1, "CANONICAL_HISTORY_ACCOUNT_AMBIGUOUS")
    binding = {
        "account": accounts.pop(),
        "knownMinimumIds": [row["id"] for row in known],
    }
    return gate.history(ledger, binding, now, inputs)


def select(gate, rows, captions, live):
    known, recent_captions, _, forbidden, chronology, _ = live
    latest, _ = gate.variety_windows(chronology)
    eligible = []
    for row in rows:
        measures = row["scoreInputs"]
        opener = {
            "mechanism": measures["mechanism"],
            "family": measures["scenarioFamily"],
            "sha256": measures["visibleOpenerSha256"],
        }
        binding = {
            **measures,
            "opener": opener,
            "deal": row["id"],
            "mediaSha256": row["video"]["sha256"],
            "captionSha256": captions[0],
        }
        reasons, variety = gate.check_history(
            known,
            recent_captions,
            latest,
            forbidden,
            binding,
            chronology,
        )
        # Other positions may follow a different mechanism. The first pick
        # must differ from the latest observed post.
        if set(reasons) - {"CONSECUTIVE_MECHANISM"}:
            continue
        card = row["scorecard"]
        parts = {**card["parts"], "V": float(variety)}
        base = sum(value for key, value in parts.items() if key != "V")
        if gate.check_score(base, variety, parts["D"], parts["S"]):
            continue
        rank = engine_interest_rank_key_v1({**card, "parts": parts})
        eligible.append((rank, row, opener))
    eligible.sort(key=lambda item: item[0])

    previous = latest[-1]["mechanism"]
    picks = []
    for caption in captions[:PICKS]:
        candidates = [item for item in eligible if item[2]["mechanism"] != previous]
        if not candidates:
            raise PlanError("INSUFFICIENT_DIVERSE_CANDIDATES")
        # Prefer the most supplied eligible mechanism to leave alternatives
        # for the remaining picks. Score breaks ties within equal supply.
        _, row, opener = choice = min(
            candidates,
            key=lambda item: (
                -sum(
                    other[2]["mechanism"] == item[2]["mechanism"] for other in eligible
                ),
                item[0],
            ),
        )
        eligible.remove(choice)
        picks.append(
            {
                "id": row["id"],
                "mechanism": opener["mechanism"],
                "captionSha256": caption,
            }
        )
        previous = opener["mechanism"]

    fallbacks = [
        {"id": row["id"], "mechanism": opener["mechanism"]}
        for _, row, opener in eligible
    ]
    require(
        len(fallbacks) >= 2 and len({item["mechanism"] for item in fallbacks}) >= 2,
        "INSUFFICIENT_DISTINCT_FALLBACKS",
    )
    return [picks[:SLOT_COUNT], picks[SLOT_COUNT:]], fallbacks


def prepare(args, gate):
    output = args.output.resolve()
    require(
        output.is_dir()
        and not any((parent / ".git").exists() for parent in (output, *output.parents))
        and not output.is_relative_to(args.policy.parent.resolve()),
        "PRIVATE_OUTPUT_DIRECTORY_REQUIRED",
    )
    require(
        args.ledger.resolve() == gate.LEDGER.resolve(), "CANONICAL_LEDGER_PATH_REQUIRED"
    )
    now = datetime.now(UTC)
    inputs = gate.Inputs()
    ledger_raw = args.ledger.read_bytes()
    live = history(gate, gate.parse(ledger_raw), now, inputs)
    pool = gate.parse(inputs.read(args.caption_pool, gate.CAPTION_POOL_SHA))
    private = gate.parse(inputs.read(gate.PRIVATE, gate.PRIVATE_SHA))
    captions = caption_options(gate, pool, private, live[1])
    with TemporaryDirectory(prefix=".nightly-pick-", dir=output) as temporary:
        artifact = Path(temporary) / "pool"
        if args.artifact is None:
            run = download_pool(args.repo, artifact, now)
        else:
            run, artifact = None, args.artifact.resolve()
        index, seal = verified_pool(artifact, run)
        days, fallbacks = select(gate, index["candidates"], captions, live)
        require(args.ledger.read_bytes() == ledger_raw, "LEDGER_CHANGED_DURING_PICK")
        result = output / f"pick-{index['runId']}"
        if result.exists():
            require(
                (result / "candidate-pick.json").is_file(),
                "PRIVATE_PICK_ALREADY_EXISTS",
            )
            return
        complete = Path(temporary) / "complete"
        complete.mkdir()
        if run is not None:
            artifact.rename(complete / "pool")
        write_handoff(
            complete / "candidate-pick.json",
            {
                "schemaVersion": 1,
                "status": "NEVER_POST",
                "postingAuthorized": False,
                "runId": index["runId"],
                "sourceCommit": index["sourceCommit"],
                "pool": "pool" if run is not None else str(artifact),
                "poolSealSha256": seal,
                "ledgerSha256": sha256(ledger_raw).hexdigest(),
                "captionPoolSha256": gate.CAPTION_POOL_SHA,
                "days": days,
                "fallbacks": fallbacks,
            },
        )
        # Windows rename refuses to replace a concurrently published pick.
        complete.rename(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "caption-pool", "gate", "policy", "output", "artifact"):
        parser.add_argument(f"--{name}", type=Path, required=name != "artifact")
    parser.add_argument("--repo", default="BalootX/balootx-platform")
    args = parser.parse_args()
    try:
        prepare(args, load_gate(args.gate, args.policy))
    except PlanError as error:
        print(f"NIGHTLY_PICK_HELD {error.reason}")
        return 1
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        IndexError,
        subprocess.SubprocessError,
    ):
        # The local gate and downloader may include private values in exceptions.
        print("NIGHTLY_PICK_HELD")
        return 1
    print("NIGHTLY_PICK_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
