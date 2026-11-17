"""Built-in multi-perspective evidence analyzer plugin."""
from __future__ import annotations

from typing import Any, Mapping

from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import LaffaEvidenceV2, ScenarioGraphV1, ScenarioNodeV1, canonical_sha256
from .perspective_pipeline import build_decision_perspective_evidence


class MultiPerspectiveFoundryV1:
    descriptor = PluginDescriptor(
        id="multi_perspective_foundry_v1",
        kind="perspective_analyzer",
        version="1.0.0",
        description="All-decisions actor truth and isolated four-seat experience evidence",
        capabilities=(
            "all_decisions",
            "four_isolated_observers",
            "seat_journeys",
            "team_journeys",
            "laffa_evidence_v2",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=300,
        maximum_memory_mb=2048,
    )

    def __init__(
        self,
        store: Any,
        *,
        critical_threshold: float,
        minimum_critical_confidence: float = 0.75,
        perspective_policy: Any | None = None,
    ):
        if not 0.0 <= float(critical_threshold) <= 100.0:
            raise ValueError("foundry critical threshold must be between 0 and 100")
        self.store = store
        self.critical_threshold = float(critical_threshold)
        if not 0.0 <= float(minimum_critical_confidence) <= 1.0:
            raise ValueError("foundry minimum confidence must be between zero and one")
        self.minimum_critical_confidence = float(minimum_critical_confidence)
        self.perspective_policy = perspective_policy

    def analyze_decision(self, decision: Mapping[str, Any], audit: Mapping[str, Any], **_: Any):
        prompts = (
            self.perspective_policy.seat_experience_prompts
            if self.perspective_policy is not None
            else (
                "predict_opponent_play",
                "read_partner_signal",
                "identify_threat",
                "plan_next_response",
                "infer_distribution",
                "post_reveal_analysis",
            )
        )
        return build_decision_perspective_evidence(
            self.store,
            decision,
            audit,
            seat_experience_prompts=prompts,
        )

    def analyze_laffa(
        self,
        identifier: str,
        audit_report: Mapping[str, Any],
        *,
        critical_threshold: float | None = None,
    ) -> LaffaEvidenceV2:
        ledger = self.store.decision_ledger(identifier)
        decisions = {str(item["decisionId"]): item for item in self.store.list_decisions(identifier, limit=10000)}
        audits = {str(item["decisionId"]): item for item in audit_report.get("decisions", ())}
        expected = {item.decision_id for item in ledger.records}
        if set(decisions) != expected or set(audits) != expected:
            raise ValueError("foundry analysis requires one decision and one audit for every ledger record")
        perspectives = tuple(
            evidence
            for record in ledger.records
            for evidence in self.analyze_decision(decisions[record.decision_id], audits[record.decision_id])
        )
        nodes = tuple(
            ScenarioNodeV1(
                node_id=f"decision:{record.ordinal}",
                ordinal=record.ordinal,
                kind="decision",
                decision_id=record.decision_id,
                evidence_hash=canonical_sha256(audits[record.decision_id]),
            )
            for record in ledger.records
        )
        # Causal edges are intentionally absent until a tactical detector
        # supplies proof. Temporal adjacency must never be mislabeled causal.
        graph = ScenarioGraphV1.build(nodes, ())
        return LaffaEvidenceV2.build(
            ledger,
            perspectives,
            critical_threshold=(
                self.critical_threshold
                if critical_threshold is None
                else float(critical_threshold)
            ),
            minimum_critical_confidence=self.minimum_critical_confidence,
            require_four_projections=True,
            scenario_graph=graph,
        )
