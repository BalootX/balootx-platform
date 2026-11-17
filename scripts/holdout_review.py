#!/usr/bin/env python3
# ruff: noqa: E402
"""Prepare, seal, and finalize independent holdout reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.benchmark import (
    IndependentAdjudication,
    canonical_sha256,
    load_candidate_jsonl,
    submission_hash,
    validate_reviewer_attestation,
)
from baloot_platform.config import build_bundle, platform_root


ROOT = platform_root()
DEFAULT_CANDIDATES = (
    ROOT / "private-inputs/strength-candidates.jsonl"
)


def prepare_template(
    candidates: dict[str, Any],
    *,
    reviewer_id: str,
    authority_id: str,
    attestation: str,
) -> dict[str, Any]:
    reviewer_id = reviewer_id.strip()
    authority_id = authority_id.strip()
    attestation = attestation.strip()
    if not reviewer_id or not authority_id or not attestation:
        raise ValueError(
            "reviewer, authority, and independence attestation are required"
        )
    return {
        "schemaVersion": 1,
        "datasetId": candidates["datasetId"],
        "datasetSha256": candidates["datasetSha256"],
        "reviewer": {
            "id": reviewer_id,
            "authorityId": authority_id,
            "independenceAttestation": attestation,
            "independenceAttestationSha256": canonical_sha256(
                {
                    "reviewerId": reviewer_id,
                    "authorityId": authority_id,
                    "attestation": attestation,
                }
            ),
        },
        "instructions": {
            "blindReview": True,
            "observerStateOnly": True,
            "hiddenCardsAllowed": False,
            "engineRecommendationAllowed": False,
            "requiredFields": [
                "action",
                "confidence",
                "actionRegrets",
            ],
            "completion": (
                "Fill every label using only the matching observer-safe "
                "candidate. Then run the seal command."
            ),
        },
        "labels": [
            {
                "candidateId": item["candidateId"],
                "archetype": item["archetype"],
                "legalActions": list(item["legalActions"]),
                "action": None,
                "confidence": None,
                "actionRegrets": {
                    str(action): None
                    for action in item["legalActions"]
                },
            }
            for item in candidates["positions"]
        ],
        "submissionHash": None,
    }


def seal_submission(
    candidates: dict[str, Any],
    draft: dict[str, Any],
) -> dict[str, Any]:
    if str(draft.get("datasetId", "")) != str(
        candidates["datasetId"]
    ):
        raise ValueError("review draft dataset identity mismatch")
    if str(draft.get("datasetSha256", "")) != str(
        candidates["datasetSha256"]
    ):
        raise ValueError("review draft dataset hash mismatch")
    validate_reviewer_attestation(dict(draft.get("reviewer") or {}))
    candidate_by_id = {
        str(item["candidateId"]): item
        for item in candidates["positions"]
    }
    labels = list(draft.get("labels") or ())
    if len(labels) != len(candidate_by_id):
        raise ValueError("review draft must cover the complete corpus")
    sealed_labels = []
    seen: set[str] = set()
    for item in labels:
        value = dict(item)
        candidate_id = str(value.get("candidateId", "")).strip()
        if candidate_id not in candidate_by_id or candidate_id in seen:
            raise ValueError("review draft candidate identity is invalid")
        seen.add(candidate_id)
        action = str(value.get("action", "")).strip()
        legal = {
            str(candidate)
            for candidate in candidate_by_id[candidate_id]["legalActions"]
        }
        if action not in legal:
            raise ValueError("review draft action must be legal")
        try:
            confidence = float(value.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("review confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("review confidence must be in [0, 1]")
        raw_regrets = dict(value.get("actionRegrets") or {})
        if set(str(key) for key in raw_regrets) != legal:
            raise ValueError(
                "review action regrets must cover every legal action"
            )
        regrets: dict[str, float] = {}
        for candidate_action in sorted(legal):
            try:
                regret = float(raw_regrets[candidate_action])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "review action regrets must be numeric"
                ) from exc
            if not math.isfinite(regret) or regret < 0.0:
                raise ValueError(
                    "review action regrets must be finite and nonnegative"
                )
            regrets[candidate_action] = regret
        if min(regrets.values()) != 0.0 or regrets[action] != 0.0:
            raise ValueError(
                "review best action must have zero regret"
            )
        sealed_labels.append(
            {
                "candidateId": candidate_id,
                "action": action,
                "confidence": confidence,
                "actionRegrets": regrets,
            }
        )
    if seen != set(candidate_by_id):
        raise ValueError("review draft must cover the complete corpus")
    sealed_labels.sort(key=lambda item: str(item["candidateId"]))
    sealed = {
        "schemaVersion": 1,
        "datasetId": candidates["datasetId"],
        "datasetSha256": candidates["datasetSha256"],
        "reviewer": dict(draft.get("reviewer") or {}),
        "labels": sealed_labels,
    }
    sealed["submissionHash"] = submission_hash(sealed)
    return sealed


def serialize_holdout(value: dict[str, Any]) -> str:
    manifest = {
        key: item for key, item in value.items() if key != "positions"
    }
    return "".join(
        json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in [manifest, *value["positions"]]
    )


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_reviewer_package(
    candidates: dict[str, Any],
    *,
    reviewer_id: str,
    authority_id: str,
    attestation: str,
    output: Path,
) -> dict[str, Any]:
    """Build a deterministic blind-review ZIP with no answer material."""

    draft = prepare_template(
        candidates,
        reviewer_id=reviewer_id,
        authority_id=authority_id,
        attestation=attestation,
    )
    files = {
        "candidates.jsonl": serialize_holdout(candidates).encode("utf-8"),
        "review-template.json": _json_bytes(draft),
        "README.md": (
            "# BalootX blind holdout review\n\n"
            "راجع كل حالة باستخدام معلومات المراقب والخيارات القانونية فقط. "
            "لا تستخدم توصيات المحرك أو الأوراق المخفية ولا تنسق الإجابات "
            "مع أي مراجع آخر.\n\n"
            "Review every position using only the observer-safe state and "
            "legal actions. Do not use engine recommendations, hidden cards, "
            "or coordinate labels with another reviewer. Fill every action, "
            "confidence, and actionRegrets field in review-template.json.\n"
        ).encode("utf-8"),
    }
    manifest = {
        "schemaVersion": 1,
        "purpose": "independent_blind_permanent_holdout_review",
        "datasetId": candidates["datasetId"],
        "datasetSha256": candidates["datasetSha256"],
        "candidateCount": len(candidates["positions"]),
        "reviewer": dict(draft["reviewer"]),
        "files": {
            name: {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            for name, data in sorted(files.items())
        },
        "explicitlyExcluded": [
            "trusted replay recipes",
            "complete hidden deals",
            "exact labels",
            "engine recommendations",
            "other reviewer submissions",
        ],
        "proofBoundary": (
            "This package proves deterministic observer-safe review input "
            "assembly only. Reviewer independence, expertise, completion, "
            "agreement, and label correctness require later verification."
        ),
    }
    files["manifest.json"] = _json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(_zip_info(name), data)
    return {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "manifest": manifest,
    }


def finalize_holdout(
    candidates: dict[str, Any],
    submissions: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    exact_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return IndependentAdjudication(profile).finalize(
        candidates,
        submissions,
        exact_evidence=exact_evidence,
    ).to_dict()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--reviewer-id", required=True)
    prepare.add_argument("--authority-id", required=True)
    prepare.add_argument("--attestation", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    package = commands.add_parser("package")
    package.add_argument("--candidates", type=Path, required=True)
    package.add_argument("--reviewer-id", required=True)
    package.add_argument("--authority-id", required=True)
    package.add_argument("--attestation", required=True)
    package.add_argument("--output", type=Path, required=True)

    seal = commands.add_parser("seal")
    seal.add_argument("--candidates", type=Path, required=True)
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    finalize.add_argument("--profile", type=Path)
    finalize.add_argument(
        "--submission",
        type=Path,
        action="append",
        required=True,
    )
    finalize.add_argument("--exact-labels", type=Path)
    finalize.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    candidates = load_candidate_jsonl(args.candidates)
    if args.command == "prepare":
        value = prepare_template(
            candidates,
            reviewer_id=args.reviewer_id,
            authority_id=args.authority_id,
            attestation=args.attestation,
        )
        _write_json(args.output, value)
    elif args.command == "package":
        result = build_reviewer_package(
            candidates,
            reviewer_id=args.reviewer_id,
            authority_id=args.authority_id,
            attestation=args.attestation,
            output=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    elif args.command == "seal":
        value = seal_submission(candidates, _read_json(args.draft))
        _write_json(args.output, value)
    else:
        profile = (
            yaml.safe_load(args.profile.read_text(encoding="utf-8"))
            if args.profile is not None
            else build_bundle(ROOT).component("benchmark")
        )
        value = finalize_holdout(
            candidates,
            [_read_json(path) for path in args.submission],
            profile,
            exact_evidence=(
                _read_json(args.exact_labels)
                if args.exact_labels is not None
                else None
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            serialize_holdout(value),
            encoding="utf-8",
            newline="\n",
        )
    print(args.output)


if __name__ == "__main__":
    main()
