from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import jsonschema
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from baloot_platform.benchmark import IntelligenceBenchmarkV18
from baloot_platform.certification import (
    build_certification_candidate,
    certification_approval_payload,
    verify_certification_candidate,
)
from baloot_platform.operations import SloEvaluatorV18


def test_rules_certification_candidate_is_tamper_evident(root: Path):
    candidate = build_certification_candidate(
        rules_path=root / "configs/rules/saudi_common_v1.yaml",
        adjudication_path=root / "configs/adjudication/saudi_researched_v1.yaml",
        golden_cases_path=root / "examples/rules-certification-golden-v18.json",
    ).to_dict()
    assert verify_certification_candidate(candidate)["valid"] is True
    tampered = copy.deepcopy(candidate)
    tampered["goldenCases"][0]["expected"] = "tampered"
    assert verify_certification_candidate(tampered)["valid"] is False
    assert candidate["automaticRuleActivation"] is False


def test_official_rules_status_requires_external_signers(root: Path):
    unsigned = build_certification_candidate(
        rules_path=root / "configs/rules/saudi_common_v1.yaml",
        adjudication_path=root / "configs/adjudication/saudi_researched_v1.yaml",
        golden_cases_path=root / "examples/rules-certification-golden-v18.json",
    ).to_dict()
    assert unsigned["official"] is False
    referee = Ed25519PrivateKey.generate()
    authority = Ed25519PrivateKey.generate()
    approval_payload = certification_approval_payload(unsigned)
    signatures = [
        {
            "signerId": signer_id,
            "role": role,
            "algorithm": "ed25519",
            "signature": base64.b64encode(key.sign(approval_payload)).decode("ascii"),
        }
        for signer_id, role, key in (
            ("referee-1", "certified_referee", referee),
            ("authority-1", "rules_authority", authority),
        )
    ]
    signed = build_certification_candidate(
        rules_path=root / "configs/rules/saudi_common_v1.yaml",
        adjudication_path=root / "configs/adjudication/saudi_researched_v1.yaml",
        golden_cases_path=root / "examples/rules-certification-golden-v18.json",
        external_signatures=signatures,
    ).to_dict()
    trusted = [
        {
            "signer_id": signer_id,
            "role": role,
            "public_key_base64": base64.b64encode(
                key.public_key().public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
            ).decode("ascii"),
        }
        for signer_id, role, key in (
            ("referee-1", "certified_referee", referee),
            ("authority-1", "rules_authority", authority),
        )
    ]
    assert signed["official"] is False
    assert verify_certification_candidate(signed)["official"] is False
    verified = verify_certification_candidate(signed, trusted_signers=trusted)
    assert verified["official"] is True
    ineligible = verify_certification_candidate(
        signed,
        trusted_signers=trusted,
        certification_eligible=False,
    )
    assert ineligible["valid"] is True
    assert ineligible["signaturesValid"] is True
    assert ineligible["certificationEligible"] is False
    assert ineligible["official"] is False
    assert verified["verifiedSignerCount"] == 2

    role_strings_only = copy.deepcopy(signed)
    role_strings_only["externalSignatures"] = [
        {"role": "certified_referee", "signature": "sig-ref"},
        {"role": "rules_authority", "signature": "sig-authority"},
    ]
    assert verify_certification_candidate(role_strings_only, trusted_signers=trusted)[
        "official"
    ] is False


def test_catalog_bound_candidate_rejects_wrong_configured_authority(root: Path):
    catalog_identity = {
        "id": "saudi_federation_2023_sources",
        "version": "1.0.0",
        "sha256": "a5fcbd89d2451c436be97967547d1b88044cccabd38c4e693679ecd4714816b3",
    }
    paths = {
        "rules_path": root / "configs/rules/saudi_federation_2023.yaml",
        "adjudication_path": root / "configs/adjudication/saudi_researched_v1.yaml",
        "golden_cases_path": root
        / "examples/rule-fixtures-saudi-federation-2023.json",
        "source_catalog_identity": catalog_identity,
    }
    unsigned = build_certification_candidate(**paths).to_dict()
    keys = {
        "referee-1": ("certified_referee", Ed25519PrivateKey.generate()),
        "authority-1": ("rules_authority", Ed25519PrivateKey.generate()),
    }
    approval_payload = certification_approval_payload(unsigned)
    signatures = [
        {
            "signerId": signer_id,
            "role": role,
            "algorithm": "ed25519",
            "signature": base64.b64encode(key.sign(approval_payload)).decode("ascii"),
        }
        for signer_id, (role, key) in keys.items()
    ]
    signed = build_certification_candidate(
        **paths, external_signatures=signatures
    ).to_dict()
    trusted = [
        {
            "signer_id": signer_id,
            "role": role,
            "public_key_base64": base64.b64encode(
                key.public_key().public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
            ).decode("ascii"),
        }
        for signer_id, (role, key) in keys.items()
    ]
    profile_identity = {
        "id": signed["profileId"],
        "version": signed["profileVersion"],
        "sha256": signed["profileHash"],
    }

    verified = verify_certification_candidate(
        signed,
        trusted_signers=trusted,
        expected_profile_identity=profile_identity,
        expected_source_catalog_identity=catalog_identity,
    )
    assert verified["official"] is True
    wrong_catalog = {**catalog_identity, "sha256": "0" * 64}
    rejected = verify_certification_candidate(
        signed,
        trusted_signers=trusted,
        expected_profile_identity=profile_identity,
        expected_source_catalog_identity=wrong_catalog,
    )
    assert rejected["valid"] is True
    assert rejected["authorityIdentityValid"] is False
    assert rejected["official"] is False


def test_intelligence_benchmark_reports_calibration_and_never_auto_promotes(root: Path):
    profile = yaml.safe_load((root / "configs/benchmark/intelligence_v18.yaml").read_text(encoding="utf-8"))
    profile["promotion_thresholds"]["minimum_examples"] = 2
    benchmark = IntelligenceBenchmarkV18(profile)
    result = benchmark.evaluate("MODEL-X", [
        {"legalActions": ["C7", "C8"], "referenceAction": "C8", "predictedAction": "C8", "rankedActions": ["C8", "C7"], "referenceProbability": 0.9, "predictedRegret": 0.0, "matchValueLoss": 0.0, "archetype": "endgame"},
        {"legalActions": ["H7", "H8"], "referenceAction": "H8", "predictedAction": "H8", "rankedActions": ["H8", "H7"], "referenceProbability": 0.8, "predictedRegret": 0.2, "matchValueLoss": 0.1, "archetype": "opening_lead"},
    ]).to_dict()
    assert result["policyAccuracy"] == 1.0 and result["illegalActionRate"] == 0.0
    assert result["promotionEligible"] is True
    assert "resultHash" in result
    # Eligibility is evidence only; the benchmark has no promotion operation.
    assert not hasattr(benchmark, "promote")


def test_slo_evaluator_is_configurable(root: Path):
    profile = yaml.safe_load((root / "configs/operations/slo_v18.yaml").read_text(encoding="utf-8"))
    evaluator = SloEvaluatorV18(profile)
    metrics = {
        "action_ack_p95_ms": 120,
        "room_event_p95_ms": 220,
        "analysis_standard_p95_ms": 3000,
        "replay_availability": 0.9999,
        "job_success_rate": 0.999,
        "data_durability": 0.9999999,
    }
    passed = evaluator.evaluate(metrics)
    assert passed["status"] == "passed" and passed["automaticProductionChange"] is False
    metrics["action_ack_p95_ms"] = 900
    assert evaluator.evaluate(metrics)["status"] == "failed"


def test_standalone_governance_profiles_are_closed(root: Path):
    bundle = yaml.safe_load(
        (root / "configs/bundle.yaml").read_text(encoding="utf-8")
    )
    for family in ("hitl", "certification", "benchmark", "operations"):
        schema = json.loads((root / f"schemas/profiles/{family}.schema.json").read_text(encoding="utf-8"))
        assert schema.get("additionalProperties") is False
        config_path = root / "configs" / bundle["components"][family]
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(value)
        value["unknownField"] = True
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
        assert errors
