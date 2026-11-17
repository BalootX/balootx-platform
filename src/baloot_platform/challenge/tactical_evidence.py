"""Causal tactical detectors over solver-produced action continuations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.tactics import TacticalDetectorProfileV1, TacticalKind

from .evidence_v3 import canonical_sha256


@dataclass(frozen=True, slots=True)
class CausalActionLineV1:
    action_id: str
    card_strength: float
    immediate_margin: float
    final_margin: float
    preserves_outcome: bool
    kabout_probability: float
    opponent_kabout_probability: float
    payoff_depth: int
    mechanisms: tuple[str, ...]
    principal_variation: tuple[str, ...]
    contract_success_probability: float = 0.0
    project_consequence: float = 0.0
    hidden_deal_disagreement: float = 0.0
    causal_source_hash: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.card_strength <= 100.0:
            raise ValueError("card strength must be between 0 and 100")
        for value in (self.kabout_probability, self.opponent_kabout_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("kabout probabilities must be between zero and one")
        for value in (self.contract_success_probability, self.hidden_deal_disagreement):
            if not 0.0 <= value <= 1.0:
                raise ValueError("causal probabilities must be between zero and one")
        if self.payoff_depth < 0:
            raise ValueError("payoff depth cannot be negative")
        if self.causal_source_hash is not None and (
            len(self.causal_source_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.causal_source_hash)
        ):
            raise ValueError("causal source hash must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionId": self.action_id,
            "cardStrength": self.card_strength,
            "immediateMargin": self.immediate_margin,
            "finalMargin": self.final_margin,
            "preservesOutcome": self.preserves_outcome,
            "kaboutProbability": self.kabout_probability,
            "opponentKaboutProbability": self.opponent_kabout_probability,
            "payoffDepth": self.payoff_depth,
            "mechanisms": list(self.mechanisms),
            "principalVariation": list(self.principal_variation),
            "contractSuccessProbability": self.contract_success_probability,
            "projectConsequence": self.project_consequence,
            "hiddenDealDisagreement": self.hidden_deal_disagreement,
            "causalSourceHash": self.causal_source_hash,
        }


@dataclass(frozen=True, slots=True)
class TacticalEvidenceV1:
    kind: TacticalKind
    selected_action: str
    alternative_actions: tuple[str, ...]
    magnitude: float
    payoff_depth: int
    mechanisms: tuple[str, ...]
    criteria: tuple[tuple[str, float], ...]
    causal_proof_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": self.kind.value,
            "selectedAction": self.selected_action,
            "alternativeActions": list(self.alternative_actions),
            "magnitude": self.magnitude,
            "payoffDepth": self.payoff_depth,
            "mechanisms": list(self.mechanisms),
            "criteria": {key: value for key, value in self.criteria},
            "causalProofHash": self.causal_proof_hash,
        }


def _evidence(
    kind: TacticalKind,
    selected: CausalActionLineV1,
    alternatives: Iterable[CausalActionLineV1],
    magnitude: float,
    criteria: Mapping[str, float],
) -> TacticalEvidenceV1:
    other = tuple(alternatives)
    frozen_criteria = tuple(sorted((str(key), float(value)) for key, value in criteria.items()))
    proof = {
        "kind": kind.value,
        "selected": selected.to_dict(),
        "alternatives": [item.to_dict() for item in other],
        "criteria": {key: value for key, value in frozen_criteria},
    }
    return TacticalEvidenceV1(
        kind,
        selected.action_id,
        tuple(item.action_id for item in other),
        max(0.0, min(100.0, float(magnitude))),
        selected.payoff_depth,
        selected.mechanisms,
        frozen_criteria,
        canonical_sha256(proof),
    )


def detect_tactical_evidence(
    selected: CausalActionLineV1,
    alternatives: Iterable[CausalActionLineV1],
    *,
    minimum_delayed_gain: float = 8.0,
    kabout_probability_delta: float = 0.20,
    low_card_max_strength: float = 35.0,
    high_card_min_strength: float = 70.0,
    minimum_causal_gain: float = 0.01,
    contract_probability_delta: float = 0.20,
    project_consequence_delta: float = 5.0,
    debate_margin_gap: float = 2.0,
    debate_disagreement: float = 0.25,
) -> tuple[TacticalEvidenceV1, ...]:
    others = tuple(alternatives)
    if not others:
        return ()
    best_other = max(others, key=lambda item: item.final_margin)
    result: list[TacticalEvidenceV1] = []

    if selected.preserves_outcome and all(not item.preserves_outcome for item in others):
        result.append(_evidence(
            TacticalKind.ONLY_WINNING_MOVE, selected, others, 100.0,
            {"outcomeBoundaryCrossed": 1.0},
        ))

    delayed_gain = selected.final_margin - best_other.final_margin
    immediate_advantage = selected.immediate_margin - max(
        item.immediate_margin for item in others
    )
    delayed_component = delayed_gain - max(0.0, immediate_advantage)
    causal_mechanisms = {"entry", "promotion", "forced_return", "partner_enablement", "trump_exhaustion"}
    has_delayed_cause = (
        selected.causal_source_hash is not None
        and bool(causal_mechanisms.intersection(selected.mechanisms))
        and selected.payoff_depth > 0
    )
    if (
        selected.card_strength <= low_card_max_strength
        and has_delayed_cause
        and delayed_component >= minimum_delayed_gain
    ):
        result.append(_evidence(
            TacticalKind.SMALL_CARD_BIG_GAIN, selected, (best_other,),
            min(100.0, delayed_component * 5),
            {
                "delayedComponent": delayed_component,
                "immediateAdvantage": immediate_advantage,
                "lowCardMaxStrength": low_card_max_strength,
                "minimumDelayedGain": minimum_delayed_gain,
            },
        ))

    immediate_concession = best_other.immediate_margin - selected.immediate_margin
    if selected.card_strength >= high_card_min_strength and has_delayed_cause and immediate_concession > 0 and delayed_gain >= minimum_delayed_gain:
        result.append(_evidence(
            TacticalKind.LARGE_CARD_SACRIFICE, selected, (best_other,),
            min(100.0, delayed_gain * 5),
            {
                "delayedGain": delayed_gain,
                "immediateConcession": immediate_concession,
                "highCardMinStrength": high_card_min_strength,
                "minimumDelayedGain": minimum_delayed_gain,
            },
        ))

    kabout_gain = selected.kabout_probability - max(item.kabout_probability for item in others)
    if kabout_gain >= kabout_probability_delta:
        result.append(_evidence(
            TacticalKind.KABOUT_CREATION, selected, others, kabout_gain * 100,
            {"minimumProbabilityDelta": kabout_probability_delta},
        ))
    preservation_gain = selected.kabout_probability - min(
        item.kabout_probability for item in others
    )
    if (
        selected.kabout_probability > 0
        and preservation_gain >= kabout_probability_delta
        and kabout_gain < kabout_probability_delta
    ):
        result.append(_evidence(
            TacticalKind.KABOUT_PRESERVATION,
            selected,
            others,
            preservation_gain * 100,
            {"minimumProbabilityDelta": kabout_probability_delta},
        ))
    prevention_gain = min(item.opponent_kabout_probability for item in others) - selected.opponent_kabout_probability
    if prevention_gain >= kabout_probability_delta:
        result.append(_evidence(
            TacticalKind.KABOUT_PREVENTION, selected, others, prevention_gain * 100,
            {"minimumProbabilityDelta": kabout_probability_delta},
        ))

    if delayed_gain >= minimum_causal_gain and has_delayed_cause and {"entry", "forced_return", "partner_enablement"}.intersection(selected.mechanisms):
        result.append(_evidence(
            TacticalKind.ENTRY_UNBLOCKING, selected, (best_other,),
            min(100.0, delayed_gain * 4), {"minimumCausalGain": minimum_causal_gain},
        ))
    if (
        delayed_gain >= minimum_causal_gain
        and selected.causal_source_hash is not None
        and selected.payoff_depth > 0
        and {"trump_extraction", "trump_preservation", "overtrump_preparation", "trump_exhaustion"}.intersection(selected.mechanisms)
    ):
        result.append(_evidence(
            TacticalKind.TRUMP_CONTROL, selected, (best_other,),
            min(100.0, delayed_gain * 4), {"minimumCausalGain": minimum_causal_gain},
        ))
    contract_gain = selected.contract_success_probability - max(item.contract_success_probability for item in others)
    if contract_gain >= contract_probability_delta:
        result.append(_evidence(
            TacticalKind.CONTRACT_RESCUE, selected, others, contract_gain * 100,
            {"minimumProbabilityDelta": contract_probability_delta},
        ))
    if (
        "partner_enablement" in selected.mechanisms
        and selected.causal_source_hash is not None
        and selected.payoff_depth > 0
        and delayed_gain >= minimum_causal_gain
    ):
        result.append(_evidence(
            TacticalKind.PARTNER_ENABLEMENT, selected, (best_other,),
            min(100.0, delayed_gain * 4), {"minimumCausalGain": minimum_causal_gain},
        ))
    project_gain = selected.project_consequence - max(item.project_consequence for item in others)
    if project_gain >= project_consequence_delta:
        result.append(_evidence(
            TacticalKind.PROJECT_BIDDING, selected, others,
            min(100.0, project_gain * 5),
            {
                "minimumProjectConsequence": project_consequence_delta,
                "projectConsequenceGain": project_gain,
            },
        ))
    final_gap = selected.final_margin - best_other.final_margin
    disagreement = max(
        selected.hidden_deal_disagreement,
        best_other.hidden_deal_disagreement,
    )
    if abs(final_gap) <= debate_margin_gap and disagreement >= debate_disagreement:
        result.append(_evidence(
            TacticalKind.DEBATE, selected, (best_other,), disagreement * 100,
            {"maximumMarginGap": debate_margin_gap, "minimumDisagreement": debate_disagreement},
        ))
    return tuple(result)


class CausalTacticalDetectorV1:
    """Built-in governed boundary for causal tactical evidence."""

    descriptor = PluginDescriptor(
        id="causal_tactical_detector_v1",
        kind="tactical_detector",
        version="1.1.0",
        description="Causal tactical labels derived from paired solver continuations",
        capabilities=tuple(item.value for item in TacticalKind),
        resource_class="cpu",
        deterministic=True,
        contract_id="tactical_proof",
        contract_version="1",
        safety_guards=(
            "observer_safe",
            "hard_gate_preservation",
            "frozen_provenance",
        ),
    )

    def __init__(self, profile: TacticalDetectorProfileV1 | None = None):
        self.profile = profile or TacticalDetectorProfileV1.defaults()

    def detect(
        self,
        selected: CausalActionLineV1,
        alternatives: Iterable[CausalActionLineV1],
    ) -> tuple[TacticalEvidenceV1, ...]:
        return detect_tactical_evidence(
            selected,
            alternatives,
            minimum_delayed_gain=self.profile.minimum_delayed_gain,
            kabout_probability_delta=self.profile.kabout_probability_delta,
            low_card_max_strength=self.profile.low_card_max_strength,
            high_card_min_strength=self.profile.high_card_min_strength,
            minimum_causal_gain=self.profile.minimum_causal_gain,
            contract_probability_delta=self.profile.contract_probability_delta,
            project_consequence_delta=self.profile.project_consequence_delta,
            debate_margin_gap=self.profile.debate_margin_gap,
            debate_disagreement=self.profile.debate_disagreement,
        )
