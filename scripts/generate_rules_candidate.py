#!/usr/bin/env python3
# ruff: noqa: E402
"""Bind governed engine-conformance evidence to the rules candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.rule_source_catalog import load_bound_rule_source_catalog
from baloot_platform.filesystem import atomic_write_text

ROOT = platform_root()
DEFAULT_OUTPUT = ROOT / "release/rules-certification-candidate.json"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_rules_identity(
    traceability: dict[str, Any], fixtures: dict[str, Any], bundle: Any
) -> None:
    expected = {
        "rulesProfile": bundle.rules.id,
        "rulesVersion": bundle.rules.version,
        "rulesSha256": bundle.rules.sha256,
    }
    for field, value in expected.items():
        if traceability.get(field) != value:
            raise ValueError(f"rule traceability {field} is stale")
        if fixtures.get(field) != value:
            raise ValueError(f"rule fixtures {field} is stale")


def build(
    conformance: dict[str, Any], *, bundle_path: str | Path | None = None
) -> dict[str, Any]:
    bundle = build_bundle(ROOT, bundle_path=bundle_path)
    certification = bundle.component("certification")
    gates = certification["software_conformance"]
    fixtures = json.loads(
        (ROOT / str(certification["golden_corpus"])).read_text(
            encoding="utf-8"
        )
    )
    source_catalog = load_bound_rule_source_catalog(
        certification,
        bundle.rules,
        ROOT,
    )
    traceability = None
    if source_catalog is None:
        traceability = json.loads(
            (ROOT / "docs/rules/rule-traceability.json").read_text(
                encoding="utf-8"
            )
        )
    conformance_payload = {
        key: value for key, value in conformance.items() if key != "evidenceSha256"
    }
    if conformance.get("evidenceSha256") != canonical_sha256(conformance_payload):
        raise ValueError("engine-conformance evidence hash mismatch")
    if conformance.get("bundleSha256") != bundle.sha256:
        raise ValueError("engine-conformance bundle identity is stale")
    if conformance.get("rulesSha256") != bundle.rules.sha256:
        raise ValueError("engine-conformance rules identity is stale")
    if str(certification["candidate_profile"]) != bundle.components["rules"].path.removeprefix(
        "configs/"
    ):
        raise ValueError("certification candidate profile does not match bundle rules")
    if traceability is not None:
        validate_rules_identity(traceability, fixtures, bundle)
    else:
        validate_rules_identity(
            {
                "rulesProfile": bundle.rules.id,
                "rulesVersion": bundle.rules.version,
                "rulesSha256": bundle.rules.sha256,
            },
            fixtures,
            bundle,
        )
    required = {
        "rounds": "minimum_seeded_replay_rounds",
        "replayRounds": "minimum_seeded_replay_rounds",
        "nativePositionsCompared": "minimum_python_native_positions",
        "invalidCasesRejected": "minimum_invalid_input_fuzz_cases",
    }
    for actual_key, policy_key in required.items():
        if int(conformance.get(actual_key, 0)) < int(gates[policy_key]):
            raise ValueError(f"engine-conformance gate failed: {actual_key}")
    if conformance.get("profileEnforced") is not True:
        raise ValueError("engine-conformance profile was not enforced")
    if conformance.get("nativeEnabled") is not True:
        raise ValueError("native semantic parity was not exercised")
    if int(fixtures["caseCount"]) < int(gates["minimum_executable_fixtures"]):
        raise ValueError("executable rule fixture gate failed")
    unresolved = (
        [
            clause["id"]
            for clause in traceability["clauses"]
            if clause.get("status") != "existing_executable_evidence"
        ]
        if traceability is not None
        else []
    )
    if unresolved:
        raise ValueError(f"unresolved rule traceability clauses: {unresolved}")
    value = {
        "schemaVersion": 3 if source_catalog is not None else 2,
        "release": "BalootX",
        "status": "passed",
        "certificationStatus": str(certification["profile"]["status"]),
        "certificationEligible": bool(
            certification.get("certification_eligible", True)
        ),
        "official": False,
        "automaticRuleActivation": False,
        "externalSignatures": [],
        "bundleSha256": bundle.sha256,
        "rulesSha256": bundle.rules.sha256,
        "rulesProfile": bundle.rules.id,
        "rulesVersion": bundle.rules.version,
        "fixtureCount": fixtures["caseCount"],
        "traceabilityClauseCount": (
            len(traceability["clauses"])
            if traceability is not None
            else len(source_catalog.bindings)
        ),
        "unresolvedTraceabilityClauses": [],
        "conformance": conformance,
        "proofBoundary": (
            "Software conformance is complete for the declared candidate profile. "
            "Official Saudi Baloot certification still requires independent external signatures."
            if certification.get("certification_eligible", True)
            else "Software conformance does not make this configured variant eligible for official certification."
        ),
    }
    if source_catalog is not None:
        value["ruleSourceCatalog"] = {
            "id": str(source_catalog.data["catalog"]["id"]),
            "version": str(source_catalog.data["catalog"]["version"]),
            "sha256": source_catalog.sha256,
            "profileSha256": str(
                source_catalog.data["catalog"]["profile_sha256"]
            ),
            "retainedSourceReceipts": [
                dict(item) for item in source_catalog.verify_retained_sources(ROOT)
            ],
        }
    value["candidateSha256"] = canonical_sha256(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformance", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    conformance = json.loads(args.conformance.read_text(encoding="utf-8"))
    value = build(conformance, bundle_path=args.bundle)
    atomic_write_text(
        args.output,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(args.output)


if __name__ == "__main__":
    main()
