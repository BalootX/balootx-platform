from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.evidence_v3 import canonical_sha256
from scripts import run_bounded_campaign_proof as command


class _CampaignControl:
    def __init__(self) -> None:
        self.campaigns: dict[str, dict[str, str]] = {}

    def get_campaign(self, campaign_hash: str):
        return self.campaigns.get(campaign_hash)

    def create_campaign(self, spec) -> None:
        self.campaigns[spec.sha256] = {"status": "queued"}

    def complete(self, campaign_hash: str) -> None:
        self.campaigns[campaign_hash]["status"] = "completed"

    def audit_trace_storage_metrics(self, campaign_hash: str):
        assert self.campaigns[campaign_hash]["status"] == "completed"
        return {
            "schemaVersion": 1,
            "campaignHash": campaign_hash,
            "occurrenceCount": 1_002,
            "uniqueTraceCount": 1_000,
            "uniqueArtifactCount": 1_000,
            "logicalBytes": 6_000_000,
            "uniqueBytes": 5_000_000,
        }


class _CampaignProofs:
    def __init__(self) -> None:
        self.profile = SimpleNamespace(
            minimum_organic_rounds=1_000,
            required_worker_counts=(1, 2, 4),
            required_failure_schedules=("none", "out_of_order", "retry_reclaim"),
            sha256="a" * 64,
        )
        self.executions = ()

    def create(self, executions):
        self.executions = executions
        return {
            "record": {"kind": "campaign_equivalence_proof_v1"},
            "document": {
                "status": "qualified",
                "unmetQualificationGates": [],
                "normalizedEvidenceHash": "b" * 64,
            },
        }


def test_default_bounded_proof_emits_coherent_storage_and_timing_contract(
    tmp_path,
    monkeypatch,
) -> None:
    control = _CampaignControl()
    proofs = _CampaignProofs()
    container = SimpleNamespace(
        campaign_control=control,
        campaign_proofs=proofs,
        shutdown=lambda: None,
    )
    requested_rounds: list[int] = []

    def campaign(_container, *, seed_start: int, rounds: int, workers: int):
        assert seed_start == 910_000
        requested_rounds.append(rounds)
        return SimpleNamespace(
            sha256=f"campaign-{workers}",
            shards=lambda: tuple(range(workers)),
        )

    def complete(_container, spec) -> None:
        control.complete(spec.sha256)

    output = tmp_path / "bounded-proof.json"
    monkeypatch.setattr(command, "build_container", lambda: container)
    monkeypatch.setattr(command, "_campaign", campaign)
    monkeypatch.setattr(command, "_run_none", complete)
    monkeypatch.setattr(command, "_run_out_of_order", complete)
    monkeypatch.setattr(command, "_run_retry_reclaim", complete)
    monkeypatch.setattr(command, "_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run_bounded_campaign_proof.py", "--output", str(output)])

    assert command.main() == 0
    assert requested_rounds == [1_000, 1_000, 1_000]
    assert [item["declaredWorkerCount"] for item in proofs.executions] == [1, 2, 4]

    evidence = json.loads(output.read_text(encoding="utf-8"))[
        "boundedExecutionEvidence"
    ]
    normalized = evidence["normalizedStorageMetrics"]
    assert evidence["normalizedStorageSha256"] == canonical_sha256(normalized)
    assert normalized == {
        "schemaVersion": 1,
        "occurrenceCount": 1_002,
        "uniqueTraceCount": 1_000,
        "uniqueArtifactCount": 1_000,
        "logicalBytes": 6_000_000,
        "uniqueBytes": 5_000_000,
        "deduplicatedBytes": 1_000_000,
        "dedupeRatio": 0.166666666667,
        "logicalToUniqueRatio": 1.2,
        "logicalBytesPerRequestedRound": 6_000.0,
        "uniqueBytesPerRequestedRound": 5_000.0,
    }
    matrix = evidence["matrixStorageMetrics"]
    assert matrix["requestedExecutionRounds"] == 3_000
    assert matrix["occurrenceCount"] == 3_006
    assert matrix["logicalBytes"] == 18_000_000
    assert matrix["uniqueBytes"] == 5_000_000
    assert matrix["deduplicatedBytes"] == 13_000_000
    assert matrix["dedupeRatio"] == 0.722222222222
    assert matrix["logicalToUniqueRatio"] == 3.6
    assert all(item["storage"] == normalized for item in evidence["executions"])
    assert all(item["timing"]["wallSeconds"] >= 0 for item in evidence["executions"])
    assert all(
        item["timing"]["processCpuSeconds"] >= 0
        for item in evidence["executions"]
    )
    assert "timing" not in normalized

    changed_timing = copy.deepcopy(evidence["executions"])
    changed_timing[0]["timing"]["wallSeconds"] += 99
    repeated = command._bounded_execution_evidence(
        changed_timing,
        requested_rounds=1_000,
    )
    assert repeated["normalizedStorageSha256"] == evidence["normalizedStorageSha256"]


def test_bounded_proof_storage_contract_fails_closed() -> None:
    control = _CampaignControl()
    control.campaigns["campaign"] = {"status": "completed"}
    coherent = command._normalized_storage_metrics(
        control,
        "campaign",
        requested_rounds=1_000,
    )
    drifted = copy.deepcopy(coherent)
    drifted["uniqueBytes"] -= 1

    with pytest.raises(RuntimeError, match="differ across worker schedules"):
        command._bounded_execution_evidence(
            [
                {"storage": coherent},
                {"storage": drifted},
            ],
            requested_rounds=1_000,
        )

    original = control.audit_trace_storage_metrics

    def incomplete(campaign_hash: str):
        return {**original(campaign_hash), "occurrenceCount": 999}

    control.audit_trace_storage_metrics = incomplete
    with pytest.raises(RuntimeError, match="cover every requested round"):
        command._normalized_storage_metrics(
            control,
            "campaign",
            requested_rounds=1_000,
        )


def test_bounded_proof_rejects_round_counts_that_collapse_worker_layouts(
    monkeypatch,
) -> None:
    container = SimpleNamespace(
        campaign_proofs=_CampaignProofs(),
        shutdown=lambda: None,
    )
    monkeypatch.setattr(command, "build_container", lambda: container)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bounded_campaign_proof.py", "--rounds", "4"],
    )

    with pytest.raises(SystemExit, match="2"):
        command.main()
