from __future__ import annotations

import pytest

import hashlib
import importlib.util

from baloot_platform.benchmark import canonical_sha256
from baloot_platform.config import build_bundle
from baloot_platform.release_scope import SOURCE_EXCLUDED_PATHS, SOURCE_SCOPE
from baloot_platform.research.contracts import ResearchConfigurationAuthorityV1


def test_v4_performance_receipt_is_bound_qualified_and_reproducible(root):
    path = root / "scripts/benchmark_research_scenario_v4.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_research_scenario_v4_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_identity = {
        "revision": None,
        "tree": None,
        "scope": SOURCE_SCOPE,
        "excludedPaths": list(SOURCE_EXCLUDED_PATHS),
        "binding": "unbound_test_invocation",
    }

    evidence = module.run_benchmark(root, source_identity=source_identity)
    bundle = build_bundle(root)

    assert evidence["status"] == "passed"
    assert evidence["evidenceSha256"] == canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidenceSha256"}
    )
    assert evidence["source"] == source_identity
    assert evidence["source"]["binding"] == "unbound_test_invocation"
    assert evidence["bundle"]["sha256"] == bundle.sha256
    assert evidence["bundle"]["rulesSha256"] == bundle.rules.sha256
    authority = ResearchConfigurationAuthorityV1.build(
        evidence["configurationAuthority"]
    )
    assert evidence["bundle"]["revisionId"] == authority.bundle_revision_id
    assert evidence["bundle"]["semanticRuntimeSha256"] == (
        authority.semantic_bundle_hash
    )
    snapshot_bytes = module.canonical_json_bytes(evidence["configurationSnapshot"])
    assert (
        authority.snapshot_artifact["sha256"]
        == hashlib.sha256(snapshot_bytes).hexdigest()
    )
    assert authority.snapshot_artifact["bytes"] == len(snapshot_bytes)
    assert evidence["configurationSnapshot"]["contentSha256"] == (
        authority.effective_runtime_config_hash
    )
    assert evidence["environment"]["cpuMeasurementBoundary"] == ("whole_process_cpu_v1")
    assert (
        evidence["proofBudgetProfileSha256"]
        == hashlib.sha256(
            (root / "configs/research/proof_budgets_v1.yaml").read_bytes()
        ).hexdigest()
    )

    exact = evidence["workloads"]["exactNoSalvage"]
    assert exact["nodes"] > 0
    assert exact["cacheEntries"] > 0
    assert exact["hiddenWorlds"] == 1

    for workload in evidence["workloads"].values():
        assert workload["scenarioIdentityHash"] == (
            module.qualified_scenario_identity_hash_v1(
                {
                    "scenarioProjection": {
                        "scenarioTemplateId": workload["scenarioTemplateId"],
                        "exactStateId": workload["exactStateId"],
                    }
                }
            )
        )
        receipt = workload["receipt"]
        assert workload["receiptSha256"] == canonical_sha256(
            {key: value for key, value in receipt.items() if key != "sha256"}
        )
        assert receipt["bundleHash"] == authority.semantic_bundle_hash
        assert workload["result"]["disposition"] == "qualified"
        assert workload["result"]["qualification"]["qualified"] is True
        assert workload["result"]["qualification"]["observerSafe"] is True
        budget = workload["proofBudget"]
        assert (
            budget["registrySha256"]
            == evidence["proofBudgetAuthority"]["registrySha256"]
        )
        assert (
            evidence["proofBudgetAuthority"]["profiles"][budget["profileId"]]
            == budget["profileSha256"]
        )
        reproducibility = workload["reproducibility"]
        assert reproducibility["repeatCount"] == 2
        assert reproducibility["scope"] == (
            "semantic_outcome_excluding_resource_timings_v1"
        )
        assert reproducibility["matched"] is True
        assert (
            reproducibility["semanticReceiptSha256"]
            == workload["semanticReceiptSha256"]
        )
        assert reproducibility["fullReceiptByteIdentityExpected"] is False
        assert len(reproducibility["fullReceiptSha256"]) == 2
        assert workload["runtime"]["sampleCount"] == 2
        assert workload["runtime"]["cpuMeasurementBoundary"] == ("whole_process_cpu_v1")

    generated = evidence["workloads"]["generatedSunHundredNoSalvage"]
    assert generated["funnel"] == {
        "attempted": 1,
        "structurallyFeasible": 1,
        "legalReplay": 1,
        "semanticSatisfied": 1,
        "proofCompleted": 1,
        "qualified": 1,
    }
    summary = evidence["summary"]
    assert summary["measurementSampleScope"] == "first_sample_per_workload_v1"
    assert summary["generatedFunnel"] == generated["funnel"]
    assert summary["qualifiedGeneratedResultCount"] == 1
    assert summary["qualifiedDistinctGeneratedScenarioCount"] == 1
    assert summary["qualifiedDistinctGeneratedScenarioHashes"] == [
        generated["scenarioIdentityHash"]
    ]
    assert summary["qualifiedDistinctGeneratedResultHashes"] == [
        generated["result"]["resultHash"]
    ]
    assert (
        exact["result"]["resultHash"]
        not in summary["qualifiedDistinctGeneratedResultHashes"]
    )
