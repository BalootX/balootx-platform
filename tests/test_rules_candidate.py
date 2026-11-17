from __future__ import annotations

import copy
import json

import pytest

from baloot_platform.config import build_bundle
from scripts.generate_rules_candidate import build, canonical_sha256, validate_rules_identity


def _conformance(root, bundle_path=None):
    bundle = build_bundle(root, bundle_path=bundle_path)
    value = {
        "schemaVersion": 1,
        "bundleSha256": bundle.sha256,
        "rulesSha256": bundle.rules.sha256,
        "certificationProfile": {
            "id": "rules_certification_v21",
            "status": "candidate",
            "version": "21.0.0",
        },
        "profileEnforced": True,
        "rounds": 10000,
        "plays": 320000,
        "replayRounds": 10000,
        "nativeEnabled": True,
        "nativePositionsCompared": 320000,
        "applyUndoByteIdentityChecks": 10000,
        "observerBoundaryChecks": 40000,
        "invalidCasesRejected": 10000,
        "invalidCategories": {
            "malformed": 2500,
            "negative": 2500,
            "not_legal": 2500,
            "overflow": 2500,
        },
        "externalCertification": False,
    }
    value["evidenceSha256"] = canonical_sha256(value)
    return value


def test_v21_rules_candidate_is_complete_but_explicitly_unsigned(root):
    result = build(_conformance(root))

    assert result["status"] == "passed"
    assert result["official"] is False
    assert result["externalSignatures"] == []
    assert result["fixtureCount"] >= 250
    assert result["unresolvedTraceabilityClauses"] == []


def test_v21_rules_candidate_rejects_stale_or_under_scale_conformance(root):
    stale = copy.deepcopy(_conformance(root))
    stale["bundleSha256"] = "0" * 64
    stale["evidenceSha256"] = canonical_sha256(
        {key: value for key, value in stale.items() if key != "evidenceSha256"}
    )
    with pytest.raises(ValueError, match="bundle identity is stale"):
        build(stale)

    under_scale = copy.deepcopy(_conformance(root))
    under_scale["rounds"] = 9999
    under_scale["evidenceSha256"] = canonical_sha256(
        {
            key: value
            for key, value in under_scale.items()
            if key != "evidenceSha256"
        }
    )
    with pytest.raises(ValueError, match="rounds"):
        build(under_scale)


def test_v21_rules_candidate_rejects_stale_traceability_identity(root):
    bundle = build_bundle(root)
    traceability = json.loads(
        (root / "docs/rules/rule-traceability.json").read_text(encoding="utf-8")
    )
    fixtures = json.loads(
        (root / "examples/rule-fixtures.json").read_text(encoding="utf-8")
    )
    traceability["rulesVersion"] = "stale"
    with pytest.raises(ValueError, match="rule traceability rulesVersion is stale"):
        validate_rules_identity(traceability, fixtures, bundle)


def test_native_v3_rules_candidate_binds_profile_and_source_catalog(root):
    bundle_path = "configs/bundle_ortools_research_native_v3.yaml"
    result = build(_conformance(root, bundle_path), bundle_path=bundle_path)

    assert result["status"] == "passed"
    assert result["official"] is False
    assert result["rulesProfile"] == "saudi_federation_2023"
    assert result["ruleSourceCatalog"] == {
        **{
            "id": "saudi_federation_2023_sources",
            "version": "1.0.0",
            "sha256": "fdcdbb515ea9754ef0ba6178937f907f88328a021c6fc2e7f3a37a8b65046473",
            "profileSha256": result["rulesSha256"],
        },
        "retainedSourceReceipts": result["ruleSourceCatalog"][
            "retainedSourceReceipts"
        ],
    }
    assert [
        item["sourceId"]
        for item in result["ruleSourceCatalog"]["retainedSourceReceipts"]
        if item["status"] == "verified"
    ] == ["balootx_configured_gap_2026"]
