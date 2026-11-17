"""Integrate verified post-reveal tactics without changing observer-safe truth."""
from __future__ import annotations

from typing import Iterable

from .evidence_v3 import (
    DecisionLedgerV1,
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    ScenarioEdgeV1,
    ScenarioGraphV1,
    ScenarioNodeV1,
    TruthLabel,
    TruthMode,
    canonical_sha256,
)
from .oracle import TacticalVerificationResultV1
from .tactical_evidence import TacticalKind


_OUTCOME_KINDS = {
    TacticalKind.ONLY_WINNING_MOVE,
    TacticalKind.KABOUT_CREATION,
    TacticalKind.KABOUT_PRESERVATION,
    TacticalKind.KABOUT_PREVENTION,
    TacticalKind.CONTRACT_RESCUE,
    TacticalKind.PROJECT_BIDDING,
}
_DISASTER_KINDS = {
    TacticalKind.ONLY_WINNING_MOVE,
    TacticalKind.KABOUT_PREVENTION,
    TacticalKind.CONTRACT_RESCUE,
}
_DEPTH_KINDS = {
    TacticalKind.SMALL_CARD_BIG_GAIN,
    TacticalKind.LARGE_CARD_SACRIFICE,
    TacticalKind.ENTRY_UNBLOCKING,
    TacticalKind.TRUMP_CONTROL,
    TacticalKind.PARTNER_ENABLEMENT,
}


def _maximum(values: Iterable[float]) -> float:
    return max((float(value) for value in values), default=0.0)


def _oracle_truth_label(verification: TacticalVerificationResultV1) -> TruthLabel:
    oracle = verification.oracle
    if verification.status != "verified" or not oracle.complete or not oracle.actions:
        return TruthLabel.UNRESOLVED
    recommendation_hash = canonical_sha256(dict(oracle.observer_recommendation))
    selected = next(
        (
            item
            for item in oracle.actions
            if canonical_sha256(dict(item.action)) == recommendation_hash
        ),
        None,
    )
    margins = [item.final_margin for item in oracle.actions]
    if selected is None or selected.final_margin is None or any(item is None for item in margins):
        return TruthLabel.UNRESOLVED
    return (
        TruthLabel.EXACT_OPTIMAL
        if float(selected.final_margin) == max(float(item) for item in margins)
        else TruthLabel.UNRESOLVED
    )


def _oracle_dimensions(verification: TacticalVerificationResultV1) -> dict[str, float]:
    evidence = tuple(verification.evidence)
    dimensions: dict[str, float] = {
        "confidence": 100.0,
        "explanationFidelity": 100.0,
    }

    def magnitude(*kinds: TacticalKind) -> float:
        selected = set(kinds)
        return _maximum(item.magnitude for item in evidence if item.kind in selected)

    uniqueness = magnitude(TacticalKind.ONLY_WINNING_MOVE)
    if uniqueness:
        dimensions["uniqueness"] = uniqueness
    outcome = _maximum(item.magnitude for item in evidence if item.kind in _OUTCOME_KINDS)
    if outcome:
        dimensions["outcomeImpact"] = outcome
    disaster = _maximum(item.magnitude for item in evidence if item.kind in _DISASTER_KINDS)
    if disaster:
        dimensions["disasterAvoidance"] = disaster
    kabout = magnitude(
        TacticalKind.KABOUT_CREATION,
        TacticalKind.KABOUT_PRESERVATION,
        TacticalKind.KABOUT_PREVENTION,
    )
    if kabout:
        dimensions["kaboutImpact"] = kabout
    sacrifice = magnitude(TacticalKind.LARGE_CARD_SACRIFICE)
    if sacrifice:
        dimensions["sacrificeValue"] = sacrifice
    delayed = magnitude(
        TacticalKind.SMALL_CARD_BIG_GAIN,
        TacticalKind.LARGE_CARD_SACRIFICE,
        TacticalKind.ENTRY_UNBLOCKING,
        TacticalKind.PARTNER_ENABLEMENT,
    )
    if delayed:
        dimensions["delayedGain"] = delayed
    depth_items = tuple(item for item in evidence if item.kind in _DEPTH_KINDS)
    if depth_items:
        dimensions["tacticalDepth"] = max(
            _maximum(item.magnitude for item in depth_items),
            min(100.0, _maximum(item.payoff_depth for item in depth_items) * 20.0),
        )
        dimensions["executionDifficulty"] = dimensions["tacticalDepth"]
    debate = magnitude(TacticalKind.DEBATE)
    if debate:
        dimensions["executionDifficulty"] = max(
            debate,
            dimensions.get("executionDifficulty", 0.0),
        )
    return dimensions


def integrate_verified_tactical_evidence(
    base: LaffaEvidenceV2,
    ledger: DecisionLedgerV1,
    verifications: Iterable[TacticalVerificationResultV1],
    *,
    critical_threshold: float,
) -> LaffaEvidenceV2:
    """Add isolated oracle records and hash-only causal nodes to one لفة.

    Acting-player recommendations, observer projections, and seat-experience
    records are copied unchanged. Actual-deal actions and continuations remain
    only in the separately governed tactical-verification artifact.
    """
    if base.ledger_hash != ledger.sha256:
        raise ValueError("tactical integration ledger does not match the base Laffa")
    if not 0.0 <= float(critical_threshold) <= 100.0:
        raise ValueError("tactical integration critical threshold is out of range")
    records = {item.decision_id: item for item in ledger.records}
    actor_evidence = {
        item.decision_id: item
        for item in base.perspectives
        if item.truth_mode == TruthMode.ACTOR_OBSERVER
    }
    graph = base.scenario_graph
    if graph is None:
        raise ValueError("tactical integration requires the base decision scenario graph")
    decision_nodes = {
        item.decision_id: item
        for item in graph.nodes
        if item.kind == "decision" and item.decision_id is not None
    }
    if set(decision_nodes) != set(records):
        raise ValueError("tactical integration requires one graph node per ledger decision")

    raw_verifications = tuple(verifications)
    identities = [item.oracle.decision_id for item in raw_verifications]
    if len(set(identities)) != len(identities):
        raise ValueError("tactical integration received duplicate decision verification")
    unknown = set(identities) - set(records)
    if unknown:
        raise ValueError("tactical integration references an unknown decision")
    ordered = sorted(
        raw_verifications,
        key=lambda item: (records[item.oracle.decision_id].ordinal, item.sha256),
    )

    perspectives = list(base.perspectives)
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    for verification in ordered:
        decision_id = verification.oracle.decision_id
        if verification.status != "verified":
            continue
        if not verification.oracle.complete:
            raise ValueError("verified tactical evidence requires a complete exact oracle")
        actor = actor_evidence[decision_id]
        perspectives.append(
            DecisionPerspectiveEvidenceV3(
                decision_id=decision_id,
                acting_seat=actor.acting_seat,
                observer_seat=actor.acting_seat,
                truth_mode=TruthMode.ACTUAL_DEAL_ORACLE,
                projection_hash=verification.oracle.sha256,
                forced=actor.forced,
                truth_label=_oracle_truth_label(verification),
                confidence=1.0,
                evidence=_oracle_dimensions(verification),
                recommended_action=None,
                prompt_contract="post_reveal_analysis",
            )
        )
        source = decision_nodes[decision_id]
        oracle_node_id = f"oracle:{decision_id}"
        nodes.append(
            ScenarioNodeV1(
                oracle_node_id,
                source.ordinal,
                "actual_deal_oracle_verification",
                decision_id,
                verification.sha256,
            )
        )
        edges.append(
            ScenarioEdgeV1(
                source.node_id,
                oracle_node_id,
                "post_reveal_verification",
                100.0,
            )
        )
        for item in sorted(
            verification.evidence,
            key=lambda value: (value.kind.value, value.causal_proof_hash),
        ):
            payoff_node_id = f"tactical:{decision_id}:{item.kind.value}"
            nodes.append(
                ScenarioNodeV1(
                    payoff_node_id,
                    source.ordinal + max(1, int(item.payoff_depth)),
                    "counterfactual_tactical_payoff",
                    decision_id,
                    item.causal_proof_hash,
                )
            )
            edges.append(
                ScenarioEdgeV1(
                    oracle_node_id,
                    payoff_node_id,
                    item.kind.value,
                    item.magnitude,
                )
            )

    enriched_graph = ScenarioGraphV1.build(
        sorted(nodes, key=lambda item: (item.ordinal, item.node_id)),
        sorted(edges, key=lambda item: (item.source, item.target, item.relation)),
    )
    return LaffaEvidenceV2.build(
        ledger,
        perspectives,
        critical_threshold=float(critical_threshold),
        minimum_critical_confidence=base.minimum_critical_confidence,
        require_four_projections=True,
        scenario_graph=enriched_graph,
    )
