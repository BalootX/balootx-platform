from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.agents.qualification import (
    FinalistAgentPilotEvidenceV1,
    FoundryAgentQualificationBundleV1,
    FoundryAgentReleaseV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from scripts.run_local_finalist_preanalysis import build_workspace_manifest


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _result(value):
    document = dict(value)
    document["resultHash"] = _hash(document)
    return document


def _lineup(registry, profile_id: str, model_hash: str):
    return registry.snapshot_lineup(
        {seat: profile_id for seat in ("north", "west", "south", "east")},
        model_snapshot={"modelReleaseHash": model_hash},
    )


def _pilot_documents(root, profile):
    manifest = build_workspace_manifest(
        root,
        extra_paths=(root / "tests/fixtures/synthetic-candidates.jsonl",),
    )
    evidence = {
        "schemaVersion": 2,
        "kind": "observer_safe_exact_subset_preanalysis",
        "status": "passed",
        "source": {
            "binding": "local_workspace_manifest",
            "workspaceManifestSha256": manifest["sha256"],
        },
        "localPilot": {
            "candidatePolicy": profile.id,
            "baselinePolicy": "strong",
            "sourceManifestSha256": manifest["sha256"],
            "qualificationEligible": False,
        },
        "automaticPromotion": False,
        "promotionDecision": {"decision": "not_eligible"},
        "policies": {
            profile.id: {
                "policyId": profile.id,
                "policySnapshotSha256": _hash(profile.to_dict()),
                "positions": 120,
                "top1Accuracy": 1.0,
                "expectedRegret": 0.0,
                "illegalActionRate": 0.0,
                "allRankedActionsLegal": True,
                "latencyMs": {"p95": 2990.0},
            },
            "strong": {
                "policyId": "strong",
                "positions": 120,
                "top1Accuracy": 0.72,
                "expectedRegret": 2.9,
            },
        },
    }
    evidence["resultHash"] = _hash(evidence)
    return evidence, manifest


def _write_pilot(tmp_path, evidence, manifest):
    evidence_path = tmp_path / "finalist-preanalysis.json"
    manifest_path = tmp_path / "finalist-source-manifest.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return evidence_path, manifest_path


def test_local_finalist_pilot_is_content_bound_and_never_qualifies_release(root, tmp_path):
    registry = AgentProfileRegistry.from_bundle(root)
    profile = registry.get("finalist_candidate")
    benchmark = yaml.safe_load(
        (root / "configs/benchmark/strength.yaml").read_text(encoding="utf-8")
    )
    evidence, manifest = _pilot_documents(root, profile)
    evidence_path, manifest_path = _write_pilot(tmp_path, evidence, manifest)

    pilot = FinalistAgentPilotEvidenceV1.load(
        root=root,
        profile=profile,
        benchmark_profile=benchmark,
        evidence_path=evidence_path,
        source_manifest_path=manifest_path,
    )

    assert pilot.passed
    assert pilot.positions == 120
    assert pilot.policy_accuracy == 1.0
    assert pilot.expected_regret == 0.0
    assert pilot.qualification_eligible is False


def test_local_finalist_pilot_rejects_tampering_and_stale_sources(root, tmp_path):
    registry = AgentProfileRegistry.from_bundle(root)
    profile = registry.get("finalist_candidate")
    benchmark = yaml.safe_load(
        (root / "configs/benchmark/strength.yaml").read_text(encoding="utf-8")
    )
    evidence, manifest = _pilot_documents(root, profile)
    tampered = json.loads(json.dumps(evidence))
    tampered["policies"][profile.id]["top1Accuracy"] = 0.1
    evidence_path, manifest_path = _write_pilot(tmp_path, tampered, manifest)
    invalid = FinalistAgentPilotEvidenceV1.load(
        root=root,
        profile=profile,
        benchmark_profile=benchmark,
        evidence_path=evidence_path,
        source_manifest_path=manifest_path,
    )
    assert invalid.status == "invalid"
    assert not invalid.passed

    stale_manifest = json.loads(json.dumps(manifest))
    stale_manifest["files"][0]["bytes"] += 1
    stale_manifest["sha256"] = _hash(
        {key: value for key, value in stale_manifest.items() if key != "sha256"}
    )
    stale_evidence = json.loads(json.dumps(evidence))
    stale_evidence["source"]["workspaceManifestSha256"] = stale_manifest["sha256"]
    stale_evidence["localPilot"]["sourceManifestSha256"] = stale_manifest["sha256"]
    stale_evidence["resultHash"] = _hash(
        {key: value for key, value in stale_evidence.items() if key != "resultHash"}
    )
    evidence_path, manifest_path = _write_pilot(tmp_path, stale_evidence, stale_manifest)
    stale = FinalistAgentPilotEvidenceV1.load(
        root=root,
        profile=profile,
        benchmark_profile=benchmark,
        evidence_path=evidence_path,
        source_manifest_path=manifest_path,
    )
    assert stale.status == "stale"
    assert not stale.passed


def test_foundry_agent_release_separates_configuration_from_qualification(root):
    registry = AgentProfileRegistry.from_bundle(root)
    benchmark = yaml.safe_load(
        (root / "configs/benchmark/strength.yaml").read_text(encoding="utf-8")
    )
    model_hash = "a" * 64

    strong = FoundryAgentReleaseV1.build(
        release_id="strong-throughput-candidate",
        profile=registry.get("strong"),
        lineup_snapshot=_lineup(registry, "strong", model_hash),
        benchmark_profile=benchmark,
        model_release_hash=model_hash,
    )
    assert not strong.zero_mistake_search_ready
    assert strong.status == "rejected_for_expert_generation"
    assert not strong.expert_generation_allowed

    candidate = FoundryAgentReleaseV1.build(
        release_id="expert-configured-candidate",
        profile=registry.get("expert_deterministic"),
        lineup_snapshot=_lineup(registry, "expert_deterministic", model_hash),
        benchmark_profile=benchmark,
        model_release_hash=model_hash,
    )
    assert candidate.zero_mistake_search_ready
    assert candidate.status == "configured_candidate"
    assert not candidate.expert_generation_allowed
    assert "permanent_holdout_missing" in candidate.issues


def test_expert_generation_requires_complete_benchmark_and_professional_evidence(root):
    registry = AgentProfileRegistry.from_bundle(root)
    profile = registry.get("finalist_candidate")
    benchmark = yaml.safe_load(
        (root / "configs/benchmark/strength.yaml").read_text(encoding="utf-8")
    )
    model_hash = "b" * 64
    archetypes = {
        name: {"examples": 50, "policyAccuracy": 0.8, "expectedRegret": 0.5}
        for name in benchmark["strata"]
    }
    holdout = _result({
        "schemaVersion": 2,
        "modelId": "finalist_candidate",
        "examples": 500,
        "policyAccuracy": 0.8,
        "top3Accuracy": 0.95,
        "expectedRegret": 0.5,
        "illegalActionRate": 0.0,
        "labelProvenance": {"exact": 300, "independentlyAdjudicated": 200},
        "archetypes": archetypes,
        "searchCompletion": {"exact": 500},
    })
    paired = _result({
        "schemaVersion": 1,
        "candidateId": "finalist_candidate",
        "baselineId": "foundry_balanced",
        "pairs": 100,
        "decisiveWinRate": 0.6,
        "confidence95": {"decisiveWinRate": [0.51, 0.69]},
        "regretImprovement": 0.3,
        "promotionTargetMet": True,
    })
    symmetry = _result({
        "schemaVersion": 1,
        "profileId": profile.id,
        "profileHash": _hash(profile.to_dict()),
        "passed": True,
        "transforms": ["seat_rotation", "team_swap", "suit_relabel"],
        "comparedDecisions": 12,
        "maximumDelta": 0.0,
    })
    professional_review = {
        "schemaVersion": 1,
        "approved": True,
        "reviewerId": "professional-reviewer-1",
        "roles": ["professional_baloot"],
        "reviewedEvidenceHashes": [
            holdout["resultHash"], paired["resultHash"], symmetry["resultHash"]
        ],
        "attestation": "Reviewed independently against the frozen evidence.",
    }
    professional_review["attestationHash"] = _hash(professional_review)
    release = FoundryAgentReleaseV1.build(
        release_id="expert-release-1",
        profile=profile,
        lineup_snapshot=_lineup(registry, "finalist_candidate", model_hash),
        benchmark_profile=benchmark,
        permanent_holdout=holdout,
        paired_comparison=paired,
        symmetry_evidence=symmetry,
        professional_review=professional_review,
    )
    assert release.expert_generation_allowed
    assert release.status == "expert_qualified"
    assert not release.issues

    qualification_bundle = FoundryAgentQualificationBundleV1.build(
        release_id="expert-release-1",
        profile=profile,
        lineup_snapshot=_lineup(registry, "finalist_candidate", model_hash),
        benchmark_profile=benchmark,
        permanent_holdout=holdout,
        paired_comparison=paired,
        symmetry_evidence=symmetry,
        professional_review=professional_review,
    )
    qualification_path = root / ".codex-tmp" / "agent-qualification-test.json"
    qualification_path.parent.mkdir(parents=True, exist_ok=True)
    qualification_path.write_text(
        json.dumps(qualification_bundle.to_dict()), encoding="utf-8"
    )
    loaded = FoundryAgentQualificationBundleV1.load(
        qualification_path,
        profile=profile,
        lineup_snapshot=_lineup(registry, "finalist_candidate", model_hash),
        benchmark_profile=benchmark,
    )
    assert loaded.release == release

    bundle = build_bundle(root)
    simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=profile,
        agent_release=release,
        generation_tier="expert_validation",
    )
    assert simulator.agent_release == release.sha256
    assert len(set(simulator.equal_seat_agents)) == 1

    incomplete = FoundryAgentReleaseV1.build(
        release_id="expert-incomplete",
        profile=profile,
        lineup_snapshot=_lineup(registry, "expert_deterministic", model_hash),
        benchmark_profile=benchmark,
    )
    with pytest.raises(ValueError, match="fully qualified"):
        NaturalLaffaSimulatorV1(
            bundle.rules,
            bundle.convention,
            bundle_hash=bundle.sha256,
            configuration_snapshot=bundle.configuration_snapshot(),
            agent_profile=profile,
            agent_release=incomplete,
            generation_tier="expert_validation",
        )
