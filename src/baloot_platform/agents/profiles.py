from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from baloot_platform.analysis.bidding import BidChoice, analyze_bid, legal_bid_choices
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.decision_scores import (
    BID_DECISION_OBJECTIVE_V1,
    BID_DECISION_TIE_BREAK_V1,
    PLAY_DECISION_TIE_BREAK_V1,
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
    PrimaryDecisionScoreEvidenceV1,
    bid_score_rows_v1,
    canonical_json_bytes_v1,
    canonical_sha256_v1,
    legal_action_set_hash_v1,
    play_score_rows_v1,
    primary_decision_score_confidence_v1,
    primary_decision_score_evidence_v1,
    validate_primary_decision_score_evidence_v1,
)
from baloot_platform.analysis.evaluator import (
    ANALYSIS_OBJECTIVES,
    AnalysisPolicy,
    analyze,
)
from baloot_platform.analysis.match_value_v7 import MatchValueConfig
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search_backend import (
    ExactSearchBackend,
    PythonReferenceSearchBackend,
)
from baloot_platform.core.bidding import _choose_action
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import RoundState
from baloot_platform.core.shuffle import SplitMix64
from baloot_platform.ml.actions import ActionCodec
from baloot_platform.ml.features import encode_bid_state, encode_information_state
from baloot_platform.ml.models import (
    ModelPlugin,
    model_runtime_identity,
    rank_legal_actions,
)
from baloot_platform.plugins.base import PluginDescriptor


_PRIMARY_PLAY_AUTHORITY_KEYS_V1 = frozenset({
    "schemaVersion",
    "decisionKind",
    "actor",
    "publicPrefixHash",
    "legalActionSetHash",
    "rulesHash",
    "analysisIdentityHash",
    "traceAnalysisProfileHash",
    "cohortSeedPolicy",
    "cohortIndex",
    "cohortSeed",
    "objective",
    "tieBreak",
    "backend",
})


def _canonical_sha256_digest_v1(value: Any, *, field: str) -> str:
    digest = str(value)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return digest


def _normalize_primary_play_score_authority_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(value)
    if set(raw) != _PRIMARY_PLAY_AUTHORITY_KEYS_V1:
        raise ValueError(
            "primary play score authority is incomplete or contains unknown keys"
        )
    if raw["schemaVersion"] != 1 or isinstance(raw["schemaVersion"], bool):
        raise ValueError("unsupported primary play score authority schema")
    if raw["decisionKind"] != "play":
        raise ValueError("primary play score authority must govern play")
    try:
        actor = Seat.parse(str(raw["actor"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("primary play score authority actor is invalid") from exc
    if str(raw["actor"]) != actor.name.lower():
        raise ValueError("primary play score authority actor is not canonical")
    for key in (
        "publicPrefixHash",
        "legalActionSetHash",
        "rulesHash",
        "analysisIdentityHash",
        "traceAnalysisProfileHash",
    ):
        raw[key] = _canonical_sha256_digest_v1(raw[key], field=key)
    if raw["cohortSeedPolicy"] != PRIMARY_DECISION_COHORT_SEED_POLICY_V1:
        raise ValueError("primary play score authority seed policy is unsupported")
    cohort_index = raw["cohortIndex"]
    if (
        isinstance(cohort_index, bool)
        or not isinstance(cohort_index, int)
        or not 0 <= cohort_index <= 2
    ):
        raise ValueError("primary play score authority cohort index must be in [0, 2]")
    cohort_seed = raw["cohortSeed"]
    if (
        isinstance(cohort_seed, bool)
        or not isinstance(cohort_seed, int)
        or not 0 <= cohort_seed < 2**64
    ):
        raise ValueError("primary play score authority cohort seed is invalid")
    objective = str(raw["objective"])
    if objective not in ANALYSIS_OBJECTIVES:
        raise ValueError("primary play score authority objective is unsupported")
    if raw["tieBreak"] != PLAY_DECISION_TIE_BREAK_V1:
        raise ValueError("primary play score authority tie-break is unsupported")
    if not isinstance(raw["backend"], str) or not raw["backend"].strip():
        raise ValueError("primary play score authority backend is required")
    return raw


def primary_play_decision_score_authority_v1(
    *,
    actor: Seat | str,
    public_prefix_hash: str,
    legal_cards: tuple[Card, ...],
    rules_hash: str,
    analysis_identity_hash: str,
    trace_analysis_profile_hash: str,
    cohort_index: int,
    cohort_seed: int,
    objective: str,
    backend: str,
) -> dict[str, Any]:
    """Build the closed authority passed into one exact play analysis."""

    try:
        seat = actor if isinstance(actor, Seat) else Seat.parse(actor)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("primary play score authority actor is invalid") from exc
    if not legal_cards:
        raise ValueError("primary play score authority requires legal cards")
    return _normalize_primary_play_score_authority_v1({
        "schemaVersion": 1,
        "decisionKind": "play",
        "actor": seat.name.lower(),
        "publicPrefixHash": public_prefix_hash,
        "legalActionSetHash": legal_action_set_hash_v1(
            card.to_dict() for card in legal_cards
        ),
        "rulesHash": rules_hash,
        "analysisIdentityHash": analysis_identity_hash,
        "traceAnalysisProfileHash": trace_analysis_profile_hash,
        "cohortSeedPolicy": PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
        "cohortIndex": int(cohort_index),
        "cohortSeed": int(cohort_seed),
        "objective": objective,
        "tieBreak": PLAY_DECISION_TIE_BREAK_V1,
        "backend": backend,
    })


def _forced_decision_legality_proof_v1(
    authority: Mapping[str, Any],
    selected_action: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schemaVersion": 1,
        "proofType": "authoritative_forced_legal_action_v1",
        "decisionKind": authority["decisionKind"],
        "actor": authority["actor"],
        "publicPrefixHash": authority["publicPrefixHash"],
        "rulesHash": authority["rulesHash"],
        "legalActionCount": 1,
        "legalActionSetHash": authority["legalActionSetHash"],
        "selectedAction": dict(selected_action),
        "selectedActionHash": canonical_sha256_v1(dict(selected_action)),
    }
    return {**core, "sha256": canonical_sha256_v1(core)}


def validate_primary_play_decision_score_trace_v1(
    trace: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    selected_action: Mapping[str, Any],
    forced: bool,
) -> PrimaryDecisionScoreEvidenceV1 | None:
    """Fail closed unless a play trace exactly preserves its score authority."""

    normalized = _normalize_primary_play_score_authority_v1(authority)
    raw = dict(trace)
    selected = dict(selected_action)
    if canonical_json_bytes_v1(raw.get("selectedAction")) != canonical_json_bytes_v1(
        selected
    ):
        raise ValueError("primary play trace selected action drifted")
    if forced:
        if raw.get("primaryDecisionScoreEvidence") is not None:
            raise ValueError("forced play must not contain analyzer score evidence")
        expected = _forced_decision_legality_proof_v1(normalized, selected)
        if canonical_json_bytes_v1(raw.get("forcedLegalityProof")) != (
            canonical_json_bytes_v1(expected)
        ):
            raise ValueError("forced play legality proof is missing or drifted")
        return None
    evidence_value = raw.get("primaryDecisionScoreEvidence")
    if not isinstance(evidence_value, Mapping):
        raise ValueError("non-forced primary play score evidence is required")
    evidence = validate_primary_decision_score_evidence_v1(
        evidence_value,
        decision_kind="play",
        actor=normalized["actor"],
        public_prefix_hash=normalized["publicPrefixHash"],
        legal_action_set_hash=normalized["legalActionSetHash"],
        rules_hash=normalized["rulesHash"],
        analysis_identity_hash=normalized["analysisIdentityHash"],
        trace_analysis_profile_hash=normalized["traceAnalysisProfileHash"],
        cohort_seed_policy=normalized["cohortSeedPolicy"],
        cohort_index=normalized["cohortIndex"],
        cohort_seed=normalized["cohortSeed"],
    )
    evidence_mapping = evidence.to_dict()
    for evidence_key, authority_key in (
        ("objective", "objective"),
        ("tieBreak", "tieBreak"),
        ("backend", "backend"),
    ):
        if evidence_mapping[evidence_key] != normalized[authority_key]:
            raise ValueError(f"primary play score evidence drifted {evidence_key}")
    if canonical_json_bytes_v1(evidence.recommended_action) != canonical_json_bytes_v1(
        selected
    ):
        raise ValueError("primary play selection differs from its recommendation")
    if canonical_json_bytes_v1(raw.get("intendedBestAction")) != (
        canonical_json_bytes_v1(selected)
    ):
        raise ValueError("primary play intended action differs from its recommendation")
    if canonical_json_bytes_v1(raw.get("confidence")) != canonical_json_bytes_v1(
        primary_decision_score_confidence_v1(evidence)
    ):
        raise ValueError("primary play trace confidence drifted from score evidence")
    scalar_bindings = {
        "primaryDecisionScoreEvidenceHash": evidence.sha256,
        "decisionScoreRowsHash": evidence.rows_hash,
        "legalActionSetHash": evidence.legal_action_set_hash,
        "publicPrefixHash": evidence.public_prefix_hash,
        "rulesHash": evidence.rules_hash,
        "analysisIdentityHash": evidence.analysis_identity_hash,
        "traceAnalysisProfileHash": evidence.trace_analysis_profile_hash,
        "cohortSeedPolicy": evidence.cohort_seed_policy,
        "cohortIndex": evidence.cohort_index,
        "cohortSeed": evidence.cohort_seed,
        "analysisObjective": evidence.objective,
        "analysisTieBreak": evidence.tie_break,
        "analysisBackend": evidence.backend,
        "analysisMethod": evidence.method,
        "analysisHiddenDeals": evidence.hidden_deal_support,
        "analysisEffectiveSampleSize": evidence.effective_sample_size,
    }
    mismatched = [
        key for key, expected in scalar_bindings.items() if raw.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "primary play trace score bindings drifted: " + ", ".join(mismatched)
        )
    return evidence


def _validate_primary_play_score_authority_against_state_v1(
    value: Mapping[str, Any],
    *,
    engine: BalootEngine,
    state: RoundState,
    legal_cards: tuple[Card, ...],
    seed: int,
    exact_search_backend: ExactSearchBackend | None,
) -> dict[str, Any]:
    authority = _normalize_primary_play_score_authority_v1(value)
    actor = state.current_player
    if actor is None or authority["actor"] != actor.name.lower():
        raise ValueError("primary play score authority actor differs from runtime")
    actual_legal = tuple(engine.legal_moves(state, actor))
    actual_legal_hash = legal_action_set_hash_v1(
        card.to_dict() for card in actual_legal
    )
    supplied_legal_hash = legal_action_set_hash_v1(
        card.to_dict() for card in legal_cards
    )
    if supplied_legal_hash != actual_legal_hash:
        raise ValueError("supplied primary play legal actions differ from the engine")
    if authority["legalActionSetHash"] != actual_legal_hash:
        raise ValueError("primary play score authority legal actions drifted")
    information = InformationState.from_complete(state, actor)
    if authority["publicPrefixHash"] != canonical_sha256_v1(information.to_dict()):
        raise ValueError("primary play score authority public prefix drifted")
    if authority["rulesHash"] != engine.rules.sha256:
        raise ValueError("primary play score authority rules drifted")
    if authority["cohortSeed"] != int(seed):
        raise ValueError("primary play score authority seed drifted")
    backend = (
        PythonReferenceSearchBackend.descriptor.id
        if exact_search_backend is None
        else str(exact_search_backend.descriptor.id)
    )
    if authority["backend"] != backend:
        raise ValueError("primary play score authority backend drifted")
    return authority


def actor_observer_bid_public_state_v1(
    state: RoundState,
    machine: BiddingMachine,
    actor: Seat,
) -> dict[str, Any]:
    """Canonical actor-visible auction projection shared by generation lanes."""

    return {
        "rulesHash": state.rules_hash,
        "observer": actor.name.lower(),
        "observerHand": [card.to_dict() for card in state.hand(actor)],
        "purchaseCard": state.purchase_card.to_dict() if state.purchase_card else None,
        "dealer": state.dealer.name.lower(),
        "matchScore": list(state.match_score),
        "bidding": {
            "round": machine.round_index,
            "actor": machine.actor.name.lower() if machine.actor is not None else None,
            "inRecall": machine.in_recall,
            "legalActions": [item.to_dict() for item in legal_bid_choices(machine)],
            "history": [item.to_dict() for item in machine.records],
        },
    }


def _normalize_primary_bid_score_authority_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(value)
    if set(raw) != _PRIMARY_PLAY_AUTHORITY_KEYS_V1:
        raise ValueError(
            "primary bid score authority is incomplete or contains unknown keys"
        )
    if raw["schemaVersion"] != 1 or isinstance(raw["schemaVersion"], bool):
        raise ValueError("unsupported primary bid score authority schema")
    if raw["decisionKind"] != "bid":
        raise ValueError("primary bid score authority must govern bidding")
    try:
        actor = Seat.parse(str(raw["actor"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("primary bid score authority actor is invalid") from exc
    if str(raw["actor"]) != actor.name.lower():
        raise ValueError("primary bid score authority actor is not canonical")
    for key in (
        "publicPrefixHash",
        "legalActionSetHash",
        "rulesHash",
        "analysisIdentityHash",
        "traceAnalysisProfileHash",
    ):
        raw[key] = _canonical_sha256_digest_v1(raw[key], field=key)
    if raw["cohortSeedPolicy"] != PRIMARY_DECISION_COHORT_SEED_POLICY_V1:
        raise ValueError("primary bid score authority seed policy is unsupported")
    cohort_index = raw["cohortIndex"]
    if (
        isinstance(cohort_index, bool)
        or not isinstance(cohort_index, int)
        or not 0 <= cohort_index <= 2
    ):
        raise ValueError("primary bid score authority cohort index must be in [0, 2]")
    cohort_seed = raw["cohortSeed"]
    if (
        isinstance(cohort_seed, bool)
        or not isinstance(cohort_seed, int)
        or not 0 <= cohort_seed < 2**64
    ):
        raise ValueError("primary bid score authority cohort seed is invalid")
    if raw["objective"] != BID_DECISION_OBJECTIVE_V1:
        raise ValueError("primary bid score authority objective is unsupported")
    if raw["tieBreak"] != BID_DECISION_TIE_BREAK_V1:
        raise ValueError("primary bid score authority tie-break is unsupported")
    if not isinstance(raw["backend"], str) or not raw["backend"].strip():
        raise ValueError("primary bid score authority backend is required")
    return raw


def primary_bid_decision_score_authority_v1(
    *,
    actor: Seat | str,
    public_prefix_hash: str,
    legal_choices: tuple[BidChoice, ...],
    rules_hash: str,
    analysis_identity_hash: str,
    trace_analysis_profile_hash: str,
    cohort_index: int,
    cohort_seed: int,
    backend: str,
) -> dict[str, Any]:
    try:
        seat = actor if isinstance(actor, Seat) else Seat.parse(actor)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("primary bid score authority actor is invalid") from exc
    if not legal_choices:
        raise ValueError("primary bid score authority requires legal choices")
    return _normalize_primary_bid_score_authority_v1({
        "schemaVersion": 1,
        "decisionKind": "bid",
        "actor": seat.name.lower(),
        "publicPrefixHash": public_prefix_hash,
        "legalActionSetHash": legal_action_set_hash_v1(
            choice.to_dict() for choice in legal_choices
        ),
        "rulesHash": rules_hash,
        "analysisIdentityHash": analysis_identity_hash,
        "traceAnalysisProfileHash": trace_analysis_profile_hash,
        "cohortSeedPolicy": PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
        "cohortIndex": int(cohort_index),
        "cohortSeed": int(cohort_seed),
        "objective": BID_DECISION_OBJECTIVE_V1,
        "tieBreak": BID_DECISION_TIE_BREAK_V1,
        "backend": backend,
    })


def _validate_primary_bid_score_authority_against_state_v1(
    value: Mapping[str, Any],
    *,
    state: RoundState,
    machine: BiddingMachine,
    rules: Any,
    seed: int,
    exact_search_backend: ExactSearchBackend | None,
) -> dict[str, Any]:
    authority = _normalize_primary_bid_score_authority_v1(value)
    actor = machine.actor
    if actor is None or authority["actor"] != actor.name.lower():
        raise ValueError("primary bid score authority actor differs from runtime")
    legal = tuple(legal_bid_choices(machine))
    if authority["legalActionSetHash"] != legal_action_set_hash_v1(
        choice.to_dict() for choice in legal
    ):
        raise ValueError("primary bid score authority legal actions drifted")
    public = actor_observer_bid_public_state_v1(state, machine, actor)
    if authority["publicPrefixHash"] != canonical_sha256_v1(public):
        raise ValueError("primary bid score authority public prefix drifted")
    if authority["rulesHash"] != str(getattr(rules, "sha256", "")):
        raise ValueError("primary bid score authority rules drifted")
    if authority["cohortSeed"] != int(seed):
        raise ValueError("primary bid score authority seed drifted")
    backend = (
        PythonReferenceSearchBackend.descriptor.id
        if exact_search_backend is None
        else str(exact_search_backend.descriptor.id)
    )
    if authority["backend"] != backend:
        raise ValueError("primary bid score authority backend drifted")
    return authority


def validate_primary_bid_decision_score_trace_v1(
    trace: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    selected_action: Mapping[str, Any],
    forced: bool,
) -> PrimaryDecisionScoreEvidenceV1 | None:
    normalized = _normalize_primary_bid_score_authority_v1(authority)
    raw = dict(trace)
    selected = dict(selected_action)
    if canonical_json_bytes_v1(raw.get("selectedAction")) != canonical_json_bytes_v1(
        selected
    ):
        raise ValueError("primary bid trace selected action drifted")
    if forced:
        if raw.get("primaryDecisionScoreEvidence") is not None:
            raise ValueError("forced bid must not contain analyzer score evidence")
        expected = _forced_decision_legality_proof_v1(normalized, selected)
        if canonical_json_bytes_v1(raw.get("forcedLegalityProof")) != (
            canonical_json_bytes_v1(expected)
        ):
            raise ValueError("forced bid legality proof is missing or drifted")
        return None
    evidence_value = raw.get("primaryDecisionScoreEvidence")
    if not isinstance(evidence_value, Mapping):
        raise ValueError("non-forced primary bid score evidence is required")
    evidence = validate_primary_decision_score_evidence_v1(
        evidence_value,
        decision_kind="bid",
        actor=normalized["actor"],
        public_prefix_hash=normalized["publicPrefixHash"],
        legal_action_set_hash=normalized["legalActionSetHash"],
        rules_hash=normalized["rulesHash"],
        analysis_identity_hash=normalized["analysisIdentityHash"],
        trace_analysis_profile_hash=normalized["traceAnalysisProfileHash"],
        cohort_seed_policy=normalized["cohortSeedPolicy"],
        cohort_index=normalized["cohortIndex"],
        cohort_seed=normalized["cohortSeed"],
    )
    evidence_mapping = evidence.to_dict()
    for key in ("objective", "tieBreak", "backend"):
        if evidence_mapping[key] != normalized[key]:
            raise ValueError(f"primary bid score evidence drifted {key}")
    if canonical_json_bytes_v1(evidence.recommended_action) != canonical_json_bytes_v1(
        selected
    ):
        raise ValueError("primary bid selection differs from its recommendation")
    if canonical_json_bytes_v1(raw.get("intendedBestAction")) != (
        canonical_json_bytes_v1(selected)
    ):
        raise ValueError("primary bid intended action differs from its recommendation")
    if canonical_json_bytes_v1(raw.get("confidence")) != canonical_json_bytes_v1(
        primary_decision_score_confidence_v1(evidence)
    ):
        raise ValueError("primary bid trace confidence drifted from score evidence")
    scalar_bindings = {
        "primaryDecisionScoreEvidenceHash": evidence.sha256,
        "decisionScoreRowsHash": evidence.rows_hash,
        "legalActionSetHash": evidence.legal_action_set_hash,
        "publicPrefixHash": evidence.public_prefix_hash,
        "rulesHash": evidence.rules_hash,
        "analysisIdentityHash": evidence.analysis_identity_hash,
        "traceAnalysisProfileHash": evidence.trace_analysis_profile_hash,
        "cohortSeedPolicy": evidence.cohort_seed_policy,
        "cohortIndex": evidence.cohort_index,
        "cohortSeed": evidence.cohort_seed,
        "analysisObjective": evidence.objective,
        "analysisTieBreak": evidence.tie_break,
        "analysisBackend": evidence.backend,
        "analysisMethod": evidence.method,
        "analysisHiddenDeals": evidence.hidden_deal_support,
        "analysisEffectiveSampleSize": evidence.effective_sample_size,
    }
    mismatched = [
        key for key, expected in scalar_bindings.items() if raw.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "primary bid trace score bindings drifted: " + ", ".join(mismatched)
        )
    return evidence


@dataclass(frozen=True, slots=True)
class AgentProfile:
    id: str
    display_name_ar: str
    skill: float
    buying_strength: float
    play_search_budget: int
    play_search_mode: str
    play_search_objective: str
    play_search_max_remaining: int
    play_search_samples: int
    play_exact_allocation_limit: int
    play_exact_search_remaining: int
    play_maximum_sampling_attempts: int
    risk_tolerance: float
    mistake_probability: float
    deception_probability: float
    signal_discipline: float
    use_production_model: bool
    model_weight: float
    minimum_model_confidence: float
    bid_samples: int
    maximum_deception_regret: float
    maximum_deception_win_drop: float

    @classmethod
    def from_mapping(cls, identifier: str, value: Mapping[str, Any]) -> "AgentProfile":
        def bounded(name: str, default: float) -> float:
            return max(0.0, min(1.0, float(value.get(name, default))))

        legacy_budget = max(
            0,
            int(value.get("play_search_budget", 0)),
        )
        search_mode = str(
            value.get(
                "play_search_mode",
                (
                    "paired_information_set"
                    if legacy_budget >= 5_000
                    else "disabled"
                ),
            )
        ).strip()
        if search_mode not in {"disabled", "paired_information_set"}:
            raise ValueError(
                f"unsupported play search mode {search_mode}"
            )
        search_objective = str(
            value.get("play_search_objective", "match_point_margin")
        ).strip()
        if search_objective not in ANALYSIS_OBJECTIVES:
            raise ValueError(
                f"unsupported play search objective {search_objective}"
            )
        return cls(
            id=identifier,
            display_name_ar=str(value.get("display_name_ar", identifier)),
            skill=bounded("skill", 0.5),
            buying_strength=bounded("buying_strength", 0.5),
            play_search_budget=legacy_budget,
            play_search_mode=search_mode,
            play_search_objective=search_objective,
            play_search_max_remaining=max(
                1,
                int(value.get("play_search_max_remaining", 16)),
            ),
            play_search_samples=max(
                1,
                int(
                    value.get(
                        "play_search_samples",
                        max(4, min(48, legacy_budget // 750)),
                    )
                ),
            ),
            play_exact_allocation_limit=max(
                1,
                int(
                    value.get(
                        "play_exact_allocation_limit",
                        max(
                            100,
                            min(20_000, legacy_budget * 2),
                        ),
                    )
                ),
            ),
            play_exact_search_remaining=max(
                1,
                int(value.get("play_exact_search_remaining", 12)),
            ),
            play_maximum_sampling_attempts=max(
                1,
                int(
                    value.get(
                        "play_maximum_sampling_attempts",
                        max(2_000, legacy_budget * 4),
                    )
                ),
            ),
            risk_tolerance=bounded("risk_tolerance", 0.5),
            mistake_probability=bounded("mistake_probability", 0.0),
            deception_probability=bounded("deception_probability", 0.0),
            signal_discipline=bounded("signal_discipline", 0.5),
            use_production_model=bool(value.get("use_production_model", True)),
            model_weight=bounded("model_weight", 0.65),
            minimum_model_confidence=bounded("minimum_model_confidence", 0.36),
            bid_samples=max(0, int(value.get("bid_samples", 0))),
            maximum_deception_regret=max(0.0, float(value.get("maximum_deception_regret", 1.75))),
            maximum_deception_win_drop=max(0.0, min(1.0, float(value.get("maximum_deception_win_drop", 0.02)))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayNameAr": self.display_name_ar,
            "skill": self.skill,
            "buyingStrength": self.buying_strength,
            "playSearchBudget": self.play_search_budget,
            "playSearchMode": self.play_search_mode,
            "playSearchObjective": self.play_search_objective,
            "playSearchMaxRemaining": self.play_search_max_remaining,
            "playSearchSamples": self.play_search_samples,
            "playExactAllocationLimit": (
                self.play_exact_allocation_limit
            ),
            "playExactSearchRemaining": (
                self.play_exact_search_remaining
            ),
            "playMaximumSamplingAttempts": (
                self.play_maximum_sampling_attempts
            ),
            "riskTolerance": self.risk_tolerance,
            "mistakeProbability": self.mistake_probability,
            "deceptionProbability": self.deception_probability,
            "signalDiscipline": self.signal_discipline,
            "useProductionModel": self.use_production_model,
            "modelWeight": self.model_weight,
            "minimumModelConfidence": self.minimum_model_confidence,
            "bidSamples": self.bid_samples,
            "maximumDeceptionRegret": self.maximum_deception_regret,
            "maximumDeceptionWinDrop": self.maximum_deception_win_drop,
        }


class AgentProfileRegistry:
    def __init__(self, profiles: Mapping[str, AgentProfile], *, default_profile: str, default_lineup: Mapping[str, str]):
        self._profiles = dict(profiles)
        if default_profile not in self._profiles:
            raise ValueError(f"unknown default agent profile {default_profile}")
        self.default_profile = default_profile
        self.default_lineup = dict(default_lineup)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentProfileRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "AgentProfileRegistry":
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
            raise ValueError("agent profile file must define a profiles mapping")
        profiles = {
            str(identifier): AgentProfile.from_mapping(str(identifier), value)
            for identifier, value in data["profiles"].items()
        }
        return cls(
            profiles,
            default_profile=str(data.get("default_profile", "strong")),
            default_lineup=data.get("default_lineup", {}),
        )

    @classmethod
    def from_bundle(
        cls,
        root: str | Path | None = None,
    ) -> "AgentProfileRegistry":
        from baloot_platform.config import build_bundle

        return cls.from_mapping(
            build_bundle(
                Path(root) if root is not None else None
            ).component("agents")
        )

    def get(self, identifier: str | None = None) -> AgentProfile:
        key = identifier or self.default_profile
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise KeyError(f"unknown agent profile {key}") from exc

    def list(self) -> list[dict[str, Any]]:
        return [self._profiles[key].to_dict() for key in sorted(self._profiles)]

    @staticmethod
    def _snapshot_profile(profile: AgentProfile) -> dict[str, Any]:
        return profile.to_dict()

    def snapshot_lineup(
        self, lineup: Mapping[str, str] | None = None, *,
        model_snapshot: Mapping[str, Any] | None = None,
        seed_policy: str = "state_seed_version_actor_v1",
    ) -> dict[str, Any]:
        """Freeze every seat's complete bot tuning and model identity.

        A shared baseline must never store only friendly profile names.  The
        immutable snapshot includes every effective parameter, so later edits
        to the profile registry cannot change an already-published game.
        """
        import hashlib
        import json
        resolved = dict(self.default_lineup)
        for seat, profile_id in dict(lineup or {}).items():
            resolved[str(seat).lower()] = str(profile_id)
        seats = {}
        for seat in ("north", "west", "south", "east"):
            profile_id = resolved.get(seat, self.default_profile)
            profile = self.get(profile_id)
            seats[seat] = {
                "profileId": profile_id,
                "tunings": self._snapshot_profile(profile),
            }
        payload = {
            "schemaVersion": 1,
            "seedPolicy": str(seed_policy),
            "seats": seats,
            "modelSnapshot": dict(model_snapshot or {"mode": "disabled_for_shared_environment"}),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["lineupHash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "AgentProfileRegistry":
        import hashlib
        import json
        payload = {key: value for key, value in dict(snapshot).items() if key != "lineupHash"}
        expected = str(snapshot.get("lineupHash") or "")
        actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if expected != actual:
            raise ValueError("agent lineup snapshot hash mismatch")
        profiles: dict[str, AgentProfile] = {}
        lineup: dict[str, str] = {}
        for seat, entry in dict(snapshot.get("seats") or {}).items():
            profile_id = str(entry.get("profileId"))
            tuning = dict(entry.get("tunings") or {})
            # Snapshot dictionaries use API camelCase; convert back to source fields.
            source = {
                "display_name_ar": tuning.get("displayNameAr", profile_id),
                "skill": tuning.get("skill", 0.5),
                "buying_strength": tuning.get("buyingStrength", 0.5),
                "play_search_budget": tuning.get("playSearchBudget", 0),
                "play_search_mode": tuning.get(
                    "playSearchMode",
                    "disabled",
                ),
                "play_search_max_remaining": tuning.get(
                    "playSearchMaxRemaining",
                    16,
                ),
                "play_search_samples": tuning.get(
                    "playSearchSamples",
                    4,
                ),
                "play_exact_allocation_limit": tuning.get(
                    "playExactAllocationLimit",
                    100,
                ),
                "play_exact_search_remaining": tuning.get(
                    "playExactSearchRemaining",
                    12,
                ),
                "play_maximum_sampling_attempts": tuning.get(
                    "playMaximumSamplingAttempts",
                    2_000,
                ),
                "risk_tolerance": tuning.get("riskTolerance", 0.5),
                "mistake_probability": tuning.get("mistakeProbability", 0.0),
                "deception_probability": tuning.get("deceptionProbability", 0.0),
                "signal_discipline": tuning.get("signalDiscipline", 0.5),
                "use_production_model": tuning.get("useProductionModel", True),
                "model_weight": tuning.get("modelWeight", 0.65),
                "minimum_model_confidence": tuning.get("minimumModelConfidence", 0.36),
                "bid_samples": tuning.get("bidSamples", 0),
                "maximum_deception_regret": tuning.get("maximumDeceptionRegret", 1.75),
                "maximum_deception_win_drop": tuning.get("maximumDeceptionWinDrop", 0.02),
            }
            profiles.setdefault(profile_id, AgentProfile.from_mapping(profile_id, source))
            lineup[str(seat).lower()] = profile_id
        if not profiles:
            raise ValueError("agent lineup snapshot is empty")
        default = next(iter(profiles))
        return cls(profiles, default_profile=default, default_lineup=lineup)


def _model_identity(model: ModelPlugin | None) -> str | None:
    if model is None:
        return None
    identifier = getattr(model, "id", model.__class__.__name__)
    version = getattr(model, "version", None)
    return f"{identifier}@{version}" if version else str(identifier)


def _softmax_confidence(values: list[float]) -> float:
    if not values:
        return 0.0
    top = max(values)
    weights = [math.exp(max(-40.0, value - top)) for value in values]
    return weights[values.index(top)] / max(1e-12, sum(weights))


def canonical_plain_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one defensive JSON-compatible copy of a governed profile.

    Research execution contexts expose recursively immutable mappings. Policy
    consumers retain their own plain copy so hashing and downstream analyzers
    never depend on a mutable caller object or ``mappingproxy`` details.
    """

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError("governed profile keys must be strings")
                normalized[key] = plain(nested)
            return normalized
        if isinstance(item, (list, tuple)):
            return [plain(nested) for nested in item]
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise TypeError(
            "governed profiles must contain only JSON-compatible values"
        )

    normalized = plain(value)
    assert isinstance(normalized, dict)
    json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return normalized


def _canonical_policy_hash(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        canonical_plain_profile(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _analysis_policy_payload(policy: AnalysisPolicy) -> dict[str, Any]:
    return {
        "profileId": policy.profile_id,
        "profileVersion": policy.profile_version,
        "requestedSamples": policy.requested_samples,
        "exactAllocationLimit": policy.exact_allocation_limit,
        "exactSearchMaxRemaining": policy.exact_search_max_remaining,
        "maximumSamplingAttempts": policy.maximum_sampling_attempts,
        "objective": policy.objective,
    }


def _belief_policy_payload(policy: BeliefPolicy) -> dict[str, Any]:
    return {
        "profileId": policy.profile_id,
        "profileVersion": policy.profile_version,
        "requestedParticles": policy.requested_particles,
        "maximumParticles": policy.maximum_particles,
        "exactAllocationLimit": policy.exact_allocation_limit,
        "maximumExactAllocationLimit": policy.maximum_exact_allocation_limit,
        "minimumEffectiveFraction": policy.minimum_effective_fraction,
    }


class AdaptivePlayStrategy:
    """Observer-safe configurable playing policy with immutable decision traces.

    The trace records what the policy actually knew and why it selected an
    action. It is not a post-hoc claim about human intent. Rules and legal masks
    remain authoritative; models, search, heuristics, controlled mistakes and
    safe deception only rank already-legal actions.
    """

    descriptor = PluginDescriptor(
        "adaptive_observer_safe_v2",
        "play_strategy",
        "2.3.0",
        "Observer-safe strategy with governed objective, search-node budget, and auditable bounded variation",
        (
            "observer_safe",
            "configurable_skill",
            "analysis_budget",
            "enforced_search_node_budget",
            "bounded_mistakes",
            "safe_deception",
            "policy_trace",
            "deterministic",
        ),
    )

    def __init__(
        self,
        profile: AgentProfile,
        convention: dict[str, Any] | None = None,
        model: ModelPlugin | None = None,
        exact_search_backend: ExactSearchBackend | None = None,
        *,
        analysis_policy: AnalysisPolicy | None = None,
        belief_policy: BeliefPolicy | None = None,
        match_value_profile: Mapping[str, Any] | None = None,
    ):
        self.profile = profile
        self.convention = dict(convention or {})
        self.model = model
        self.exact_search_backend = exact_search_backend
        supplied = (
            analysis_policy is not None,
            belief_policy is not None,
            match_value_profile is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "explicit adaptive analysis requires analysis, belief, and "
                "match-value profiles together"
            )
        self.analysis_policy = analysis_policy
        self.belief_policy = belief_policy
        self._match_value_profile = (
            canonical_plain_profile(match_value_profile)
            if match_value_profile is not None
            else None
        )
        if analysis_policy is None:
            self._policy_authority = {
                "mode": "legacy_active_compatibility",
                "analysis": None,
                "belief": None,
                "matchValue": None,
            }
        else:
            assert belief_policy is not None
            assert self._match_value_profile is not None
            analysis_policy.validate_against(belief_policy)
            MatchValueConfig.from_profile(self._match_value_profile)
            match_metadata = dict(
                self._match_value_profile.get("profile") or {}
            )
            match_profile_id = str(match_metadata.get("id") or "").strip()
            match_profile_version = str(
                match_metadata.get("version") or ""
            ).strip()
            if not match_profile_id or not match_profile_version:
                raise ValueError("match-value profile identity is required")
            analysis_payload = _analysis_policy_payload(analysis_policy)
            belief_payload = _belief_policy_payload(belief_policy)
            self._policy_authority = {
                "mode": "explicit_governed",
                "analysis": {
                    "profileId": analysis_policy.profile_id,
                    "profileVersion": analysis_policy.profile_version,
                    "sha256": _canonical_policy_hash(analysis_payload),
                },
                "belief": {
                    "profileId": belief_policy.profile_id,
                    "profileVersion": belief_policy.profile_version,
                    "sha256": _canonical_policy_hash(belief_payload),
                },
                "matchValue": {
                    "profileId": match_profile_id,
                    "profileVersion": match_profile_version,
                    "sha256": _canonical_policy_hash(
                        self._match_value_profile
                    ),
                },
            }

    @property
    def match_value_profile(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._match_value_profile)

    @property
    def policy_authority(self) -> dict[str, Any]:
        return copy.deepcopy(self._policy_authority)

    def choose_with_trace(
        self,
        engine: BalootEngine,
        state: RoundState,
        legal_cards: tuple[Card, ...] | None = None,
        *,
        timeout_fallback: bool = False,
        **kwargs: Any,
    ) -> tuple[Card, dict[str, Any]]:
        legal = tuple(legal_cards or engine.legal_moves(state))
        if not legal:
            raise ValueError("no legal card")
        seed = int(kwargs.get("seed", state.seed + len(state.played_cards) * 8191 + int(state.current_player or 0)))
        raw_score_authority = kwargs.get("primary_decision_score_authority")
        score_authority = (
            _validate_primary_play_score_authority_against_state_v1(
                raw_score_authority,
                engine=engine,
                state=state,
                legal_cards=legal,
                seed=seed,
                exact_search_backend=self.exact_search_backend,
            )
            if isinstance(raw_score_authority, Mapping)
            else None
        )
        if raw_score_authority is not None and score_authority is None:
            raise TypeError("primary play score authority must be a mapping")
        if score_authority is not None:
            if self.profile.play_search_mode != "paired_information_set":
                raise ValueError(
                    "primary play score evidence requires paired information-set analysis"
                )
            if self.profile.play_search_objective != score_authority["objective"]:
                raise ValueError("primary play score authority objective drifted")
            if self.profile.mistake_probability != 0.0:
                raise ValueError("primary play score evidence forbids configured mistakes")
            if self.profile.deception_probability != 0.0:
                raise ValueError("primary play score evidence forbids configured deception")
            if self.profile.use_production_model:
                raise ValueError("primary play score evidence requires analyzer-owned ranking")
        analysis_budget = dict(kwargs.get("analysis_budget") or {})
        allowed_budget_keys = {
            "requested_samples",
            "exact_allocation_limit",
            "exact_search_max_remaining",
            "maximum_sampling_attempts",
            "exact_search_max_nodes",
            "exact_search_max_seconds",
            "exact_search_max_depth",
            "exact_search_max_cache_entries",
        }
        unknown_budget_keys = set(analysis_budget) - allowed_budget_keys
        if unknown_budget_keys:
            raise ValueError(
                f"unknown adaptive play analysis budget keys: {sorted(unknown_budget_keys)}"
            )
        optional_search_budget = {
            key: analysis_budget[key]
            for key in (
                "exact_search_max_seconds",
                "exact_search_max_depth",
                "exact_search_max_cache_entries",
            )
            if key in analysis_budget
        }
        if len(legal) == 1:
            selected = legal[0]
            trace = {
                "schemaVersion": 2,
                "source": "bot",
                "profileId": self.profile.id,
                "profile": self.profile.to_dict(),
                "strategyId": self.descriptor.id,
                "strategyVersion": self.descriptor.version,
                "decisionSource": "only_legal_action",
                "analysisMethod": None,
                "analysisHiddenDeals": None,
                "analysisEffectiveSampleSize": None,
                "analysisCompletionStatuses": [],
                "analysisExhaustiveSupport": True,
                "analysisPopulationCoverageComplete": True,
                "analysisSolverCompletion": "only_legal_action",
                "analysisTruthClass": "exact_legal_forced",
                "modelId": _model_identity(self.model),
                "modelIdentity": model_runtime_identity(self.model),
                "observerSafe": True,
                "seed": seed,
                "confidence": 1.0,
                "legalActionCount": 1,
                "intendedBestAction": selected.to_dict(),
                "selectedAction": selected.to_dict(),
                "selectedRank": 0,
                "mistakeApplied": False,
                "deceptionApplied": False,
                "maximumDeceptionRegret": self.profile.maximum_deception_regret,
                "maximumDeceptionWinDrop": self.profile.maximum_deception_win_drop,
                "rankedActions": [{
                    "action": selected.to_dict(),
                    "rank": 1,
                    "forced": True,
                }],
                "reasonTags": ["only_legal_action"],
                "analysisBudget": analysis_budget,
                "analysisPolicyAuthority": copy.deepcopy(
                    self._policy_authority
                ),
            }
            if score_authority is not None:
                trace.update({
                    "publicPrefixHash": score_authority["publicPrefixHash"],
                    "rulesHash": score_authority["rulesHash"],
                    "analysisIdentityHash": score_authority["analysisIdentityHash"],
                    "traceAnalysisProfileHash": score_authority[
                        "traceAnalysisProfileHash"
                    ],
                    "cohortSeedPolicy": score_authority["cohortSeedPolicy"],
                    "cohortIndex": score_authority["cohortIndex"],
                    "cohortSeed": score_authority["cohortSeed"],
                    "legalActionSetHash": score_authority["legalActionSetHash"],
                    "primaryDecisionScoreEvidence": None,
                    "primaryDecisionScoreEvidenceHash": None,
                    "decisionScoreRowsHash": None,
                    "forcedLegalityProof": _forced_decision_legality_proof_v1(
                        score_authority,
                        selected.to_dict(),
                    ),
                })
                validate_primary_play_decision_score_trace_v1(
                    trace,
                    score_authority,
                    selected_action=selected.to_dict(),
                    forced=True,
                )
            return selected, trace
        rng = SplitMix64(seed)
        heuristic = HeuristicBot(engine, BotProfile(seed))
        ranked: list[Card] = []
        ranking: list[dict[str, Any]] = []
        method = "heuristic"
        confidence = 0.35
        reason_tags: list[str] = []
        analysis_method: str | None = None
        analysis_hidden_deals: int | None = None
        analysis_effective_samples: float | None = None
        analysis_completion_statuses: list[str] = []
        analysis_exhaustive_support = False
        analysis_population_coverage_complete = False
        analysis_solver_completion = "not_analyzed"
        analysis_truth_class = "unresolved"
        score_evidence: PrimaryDecisionScoreEvidenceV1 | None = None

        if self.model is not None and self.profile.use_production_model and state.current_player is not None:
            try:
                info = InformationState.from_complete(state, state.current_player)
                features = encode_information_state(info)
                mask = ActionCodec.legal_play_mask(legal)
                output = self.model.infer(features, mask)
                cards_by_action = {
                    ActionCodec.play(card): card for card in legal
                }
                legal_pairs = tuple(
                    (score, cards_by_action[action_id])
                    for action_id, score in rank_legal_actions(
                        output,
                        tuple(cards_by_action),
                    )
                )
                if legal_pairs:
                    confidence = _softmax_confidence([item[0] for item in legal_pairs])
                    if confidence >= self.profile.minimum_model_confidence:
                        ranked = [item[1] for item in legal_pairs]
                        ranking = [
                            {
                                "action": card.to_dict(),
                                "rank": index + 1,
                                "policyScore": score,
                            }
                            for index, (score, card) in enumerate(legal_pairs)
                        ]
                        method = "production_model"
                        reason_tags.extend(("model_policy", "legal_mask_applied"))
            except Exception as exc:
                reason_tags.append(f"model_fallback:{exc.__class__.__name__}")
                ranked = []
                ranking = []

        remaining = sum(hand.bit_count() for hand in state.hands)
        if (
            not ranked
            and self.profile.play_search_mode
            == "paired_information_set"
            and state.current_player is not None
            and remaining <= self.profile.play_search_max_remaining
        ):
            try:
                info = InformationState.from_complete(state, state.current_player)
                result = analyze(
                    info,
                    engine.rules,
                    self.convention,
                    requested_samples=int(analysis_budget.get(
                        "requested_samples", self.profile.play_search_samples
                    )),
                    exact_allocation_limit=(
                        int(analysis_budget.get(
                            "exact_allocation_limit",
                            self.profile.play_exact_allocation_limit,
                        ))
                    ),
                    exact_search_max_remaining=(
                        int(analysis_budget.get(
                            "exact_search_max_remaining",
                            self.profile.play_exact_search_remaining,
                        ))
                    ),
                    maximum_sampling_attempts=(
                        int(analysis_budget.get(
                            "maximum_sampling_attempts",
                            self.profile.play_maximum_sampling_attempts,
                        ))
                    ),
                    exact_search_max_nodes=max(
                        1,
                        int(analysis_budget.get(
                            "exact_search_max_nodes",
                            self.profile.play_search_budget,
                        )),
                    ),
                    seed=seed,
                    objective=self.profile.play_search_objective,
                    cancel_check=kwargs.get("cancel_check"),
                    exact_search_backend=self.exact_search_backend,
                    analysis_policy=self.analysis_policy,
                    belief_policy=self.belief_policy,
                    match_value_profile=self._match_value_profile,
                    **optional_search_budget,
                )
                if score_authority is not None:
                    if result.observer != state.current_player:
                        raise ValueError(
                            "primary play analyzer observer differs from runtime"
                        )
                    if result.objective != score_authority["objective"]:
                        raise ValueError("primary play analyzer objective drifted")
                    score_evidence = primary_decision_score_evidence_v1(
                        decision_kind="play",
                        actor=score_authority["actor"],
                        public_prefix_hash=score_authority["publicPrefixHash"],
                        rules_hash=score_authority["rulesHash"],
                        analysis_identity_hash=score_authority[
                            "analysisIdentityHash"
                        ],
                        trace_analysis_profile_hash=score_authority[
                            "traceAnalysisProfileHash"
                        ],
                        cohort_seed_policy=score_authority["cohortSeedPolicy"],
                        cohort_index=score_authority["cohortIndex"],
                        cohort_seed=score_authority["cohortSeed"],
                        objective=result.objective,
                        tie_break=score_authority["tieBreak"],
                        backend=score_authority["backend"],
                        method=result.method,
                        hidden_deal_support=result.hidden_deals,
                        effective_sample_size=result.effective_sample_size,
                        recommended_action=result.recommended_card.to_dict(),
                        rows=play_score_rows_v1(result.actions),
                        legal_action_set_hash=score_authority[
                            "legalActionSetHash"
                        ],
                    )
                ranked = [item.card for item in result.actions if item.card in legal]
                ranking = [
                    {
                        "action": item.card.to_dict(),
                        "rank": index + 1,
                        "expectedMargin": item.expected_match_margin,
                        "winProbability": item.win_probability,
                        "matchWinProbability": item.match_win_probability,
                        "downsideCvar10": item.downside_cvar,
                        "bestShare": item.best_share,
                        "exactShare": item.exact_share,
                    }
                    for index, item in enumerate(result.actions)
                    if item.card in legal
                ]
                method = "hidden_information_search"
                analysis_method = result.method
                analysis_hidden_deals = int(result.hidden_deals)
                analysis_effective_samples = float(result.effective_sample_size)
                analysis_completion_statuses = sorted({
                    str(getattr(item, "search_completion_status", "unknown"))
                    for item in result.actions
                })
                analysis_exhaustive_support = bool(
                    str(result.method).startswith("exhaustive_")
                    and analysis_completion_statuses == ["exact"]
                    and all(float(item.exact_share) >= 1.0 for item in result.actions)
                )
                analysis_population_coverage_complete = bool(
                    str(result.method).startswith("exhaustive_")
                    and analysis_completion_statuses in (
                        ["exact"], ["heuristic_complete"]
                    )
                )
                analysis_solver_completion = (
                    analysis_completion_statuses[0]
                    if len(analysis_completion_statuses) == 1
                    else "mixed_or_unknown"
                )
                analysis_truth_class = (
                    "exact_optimal"
                    if analysis_exhaustive_support
                    else (
                        "full_population_best_known"
                        if analysis_population_coverage_complete
                        else "unresolved"
                    )
                )
                confidence = max(0.25, min(1.0, result.effective_sample_size / max(8.0, float(result.hidden_deals or 1))))
                reason_tags.extend(("paired_hidden_deals", "observer_safe_search"))
            except InterruptedError:
                raise
            except TimeoutError as exc:
                if not timeout_fallback or score_authority is not None:
                    raise
                analysis_solver_completion = "time_censored"
                analysis_completion_statuses = ["time_censored"]
                reason_tags.append(f"search_fallback:{exc.__class__.__name__}")
            except Exception as exc:
                if score_authority is not None:
                    raise ValueError(
                        "primary play decision analysis/evidence failed closed"
                    ) from exc
                reason_tags.append(f"search_fallback:{exc.__class__.__name__}")
                ranked = []
                ranking = []

        if score_authority is not None and score_evidence is None:
            raise ValueError("non-forced primary play score evidence is missing")
        if not ranked:
            first = heuristic.choose(state)
            ranked = [first, *(card for card in legal if card != first)]
            ranking = [
                {"action": card.to_dict(), "rank": index + 1, "heuristicPreferred": index == 0}
                for index, card in enumerate(ranked)
            ]
            method = "transparent_heuristic"
            reason_tags.extend(("heuristic_card_control", "observer_safe"))

        intended = ranked[0]
        selected = intended
        selected_index = 0
        mistake_applied = False
        deception_applied = False
        error_rate = self.profile.mistake_probability
        roll = rng.next() / float(2**64)
        if len(ranked) > 1 and roll < error_rate:
            selected_index = 1 + int(rng.next() % (len(ranked) - 1))
            selected = ranked[selected_index]
            mistake_applied = True
            reason_tags.append("controlled_mistake")
        elif len(ranked) > 1 and self.profile.deception_probability > 0:
            deception_roll = rng.next() / float(2**64)
            if deception_roll < self.profile.deception_probability:
                best_metrics = ranking[0] if ranking else {}
                safe: list[int] = []
                for index, item in enumerate(ranking[1:], start=1):
                    if "expectedMargin" not in best_metrics or "expectedMargin" not in item:
                        continue
                    regret = float(best_metrics["expectedMargin"]) - float(item["expectedMargin"])
                    win_drop = float(best_metrics.get("winProbability", 0.0)) - float(item.get("winProbability", 0.0))
                    if regret <= self.profile.maximum_deception_regret and win_drop <= self.profile.maximum_deception_win_drop:
                        safe.append(index)
                if safe:
                    selected_index = safe[int(rng.next() % len(safe))]
                    selected = ranked[selected_index]
                    deception_applied = True
                    reason_tags.append("safe_deception")

        trace = {
            "schemaVersion": 2,
            "source": "bot",
            "profileId": self.profile.id,
            "profile": self.profile.to_dict(),
            "strategyId": self.descriptor.id,
            "strategyVersion": self.descriptor.version,
            "decisionSource": method,
            "analysisMethod": analysis_method,
            "analysisHiddenDeals": analysis_hidden_deals,
            "analysisEffectiveSampleSize": analysis_effective_samples,
            "analysisCompletionStatuses": analysis_completion_statuses,
            "analysisExhaustiveSupport": analysis_exhaustive_support,
            "analysisPopulationCoverageComplete": (
                analysis_population_coverage_complete
            ),
            "analysisSolverCompletion": analysis_solver_completion,
            "analysisTruthClass": analysis_truth_class,
            "modelId": _model_identity(self.model),
            "modelIdentity": model_runtime_identity(self.model),
            "observerSafe": True,
            "seed": seed,
            "confidence": (
                primary_decision_score_confidence_v1(score_evidence)
                if score_evidence is not None
                else round(confidence, 6)
            ),
            "legalActionCount": len(legal),
            "intendedBestAction": intended.to_dict(),
            "selectedAction": selected.to_dict(),
            "selectedRank": selected_index,
            "mistakeApplied": mistake_applied,
            "deceptionApplied": deception_applied,
            "maximumDeceptionRegret": self.profile.maximum_deception_regret,
            "maximumDeceptionWinDrop": self.profile.maximum_deception_win_drop,
            "rankedActions": ranking,
            "reasonTags": reason_tags,
            "analysisBudget": analysis_budget,
            "analysisPolicyAuthority": copy.deepcopy(
                self._policy_authority
            ),
        }
        if score_evidence is not None:
            if canonical_json_bytes_v1(selected.to_dict()) != canonical_json_bytes_v1(
                score_evidence.recommended_action
            ):
                raise ValueError(
                    "primary play selection differs from canonical score recommendation"
                )
            trace.update({
                "publicPrefixHash": score_evidence.public_prefix_hash,
                "rulesHash": score_evidence.rules_hash,
                "analysisIdentityHash": score_evidence.analysis_identity_hash,
                "traceAnalysisProfileHash": (
                    score_evidence.trace_analysis_profile_hash
                ),
                "cohortSeedPolicy": score_evidence.cohort_seed_policy,
                "cohortIndex": score_evidence.cohort_index,
                "cohortSeed": score_evidence.cohort_seed,
                "analysisObjective": score_evidence.objective,
                "analysisTieBreak": score_evidence.tie_break,
                "analysisBackend": score_evidence.backend,
                "legalActionSetHash": score_evidence.legal_action_set_hash,
                "primaryDecisionScoreEvidence": score_evidence.to_dict(),
                "primaryDecisionScoreEvidenceHash": score_evidence.sha256,
                "decisionScoreRowsHash": score_evidence.rows_hash,
                "forcedLegalityProof": None,
            })
            validate_primary_play_decision_score_trace_v1(
                trace,
                score_authority,
                selected_action=selected.to_dict(),
                forced=False,
            )
        return selected, trace

    def choose_card(
        self,
        engine: BalootEngine,
        state: RoundState,
        legal_cards: tuple[Card, ...] | None = None,
        **kwargs: Any,
    ) -> Card:
        card, trace = self.choose_with_trace(engine, state, legal_cards, **kwargs)
        require_uncensored_play_trace(trace)
        return card


class AdaptiveBiddingStrategy:
    """Observer-safe buy/pass strategy with immutable ranking provenance."""

    descriptor = PluginDescriptor(
        "adaptive_observer_safe_bidding_v2",
        "bidding_strategy",
        "2.0.0",
        "Observer-safe buy/pass strategy with model, paired simulation, safe variation and policy traces",
        ("observer_safe", "buy_pass", "shared_cohort", "model_optional", "policy_trace", "deterministic"),
    )

    def __init__(
        self,
        profile: AgentProfile,
        model: ModelPlugin | None = None,
        *,
        exact_search_backend: ExactSearchBackend | None = None,
    ):
        self.profile = profile
        self.model = model
        self.exact_search_backend = exact_search_backend

    def choose_with_trace(
        self,
        state: RoundState,
        machine: BiddingMachine,
        rules: Any,
        *,
        seed: int,
        cancel_check: Any = None,
        primary_decision_score_authority: Mapping[str, Any] | None = None,
    ) -> tuple[str, Any | None, dict[str, Any]]:
        actor = machine.actor
        if actor is None:
            raise ValueError("auction has no current actor")
        choices = legal_bid_choices(machine)
        if not choices:
            raise ValueError("auction has no legal action")
        score_authority = (
            _validate_primary_bid_score_authority_against_state_v1(
                primary_decision_score_authority,
                state=state,
                machine=machine,
                rules=rules,
                seed=seed,
                exact_search_backend=self.exact_search_backend,
            )
            if primary_decision_score_authority is not None
            else None
        )
        if score_authority is not None:
            if self.profile.skill != 1.0:
                raise ValueError("primary bid score evidence requires full configured skill")
            if self.profile.mistake_probability != 0.0:
                raise ValueError("primary bid score evidence forbids configured mistakes")
            if self.profile.deception_probability != 0.0:
                raise ValueError("primary bid score evidence forbids configured deception")
            if self.profile.use_production_model:
                raise ValueError("primary bid score evidence requires analyzer-owned ranking")
            if self.profile.bid_samples < 2:
                raise ValueError("primary bid score evidence requires paired bid analysis")
        if score_authority is not None and len(choices) == 1:
            selected = choices[0]
            trace = {
                "schemaVersion": 2,
                "source": "bot",
                "profileId": self.profile.id,
                "profile": self.profile.to_dict(),
                "strategyId": self.descriptor.id,
                "strategyVersion": self.descriptor.version,
                "decisionSource": "only_legal_action",
                "analysisMethod": None,
                "analysisHiddenDeals": None,
                "analysisEffectiveSampleSize": None,
                "modelId": _model_identity(self.model),
                "modelIdentity": model_runtime_identity(self.model),
                "observerSafe": True,
                "seed": int(seed),
                "confidence": 1.0,
                "legalActionCount": 1,
                "intendedBestAction": selected.to_dict(),
                "selectedAction": selected.to_dict(),
                "selectedRank": 0,
                "mistakeApplied": False,
                "deceptionApplied": False,
                "rankedActions": [{
                    **selected.to_dict(),
                    "rank": 1,
                    "forced": True,
                }],
                "reasonTags": ["only_legal_action"],
                "publicPrefixHash": score_authority["publicPrefixHash"],
                "rulesHash": score_authority["rulesHash"],
                "analysisIdentityHash": score_authority["analysisIdentityHash"],
                "traceAnalysisProfileHash": score_authority[
                    "traceAnalysisProfileHash"
                ],
                "cohortSeedPolicy": score_authority["cohortSeedPolicy"],
                "cohortIndex": score_authority["cohortIndex"],
                "cohortSeed": score_authority["cohortSeed"],
                "legalActionSetHash": score_authority["legalActionSetHash"],
                "primaryDecisionScoreEvidence": None,
                "primaryDecisionScoreEvidenceHash": None,
                "decisionScoreRowsHash": None,
                "forcedLegalityProof": _forced_decision_legality_proof_v1(
                    score_authority,
                    selected.to_dict(),
                ),
            }
            validate_primary_bid_decision_score_trace_v1(
                trace,
                score_authority,
                selected_action=selected.to_dict(),
                forced=True,
            )
            return selected.action, selected.trump_suit, trace
        ranked = list(choices)
        ranking: list[dict[str, Any]] = []
        method = "transparent_heuristic"
        confidence = 0.35
        reason_tags: list[str] = []
        analysis_hidden_deals: int | None = None
        analysis_effective_samples: float | None = None
        score_evidence: PrimaryDecisionScoreEvidenceV1 | None = None

        if self.model is not None and self.profile.use_production_model:
            try:
                features = encode_bid_state(state, machine, actor)
                mask_values = [0] * 76
                for choice in choices:
                    mask_values[ActionCodec.bid(choice.action, choice.trump_suit)] = 1
                output = self.model.infer(features, tuple(mask_values))
                choices_by_action = {
                    ActionCodec.bid(choice.action, choice.trump_suit): choice
                    for choice in choices
                }
                scored = tuple(
                    (score, choices_by_action[action_id])
                    for action_id, score in rank_legal_actions(
                        output,
                        tuple(choices_by_action),
                    )
                )
                confidence = _softmax_confidence([item[0] for item in scored])
                if confidence >= self.profile.minimum_model_confidence:
                    ranked = [item[1] for item in scored]
                    ranking = [
                        {**choice.to_dict(), "rank": index + 1, "policyScore": score}
                        for index, (score, choice) in enumerate(scored)
                    ]
                    method = "production_model"
                    reason_tags.extend(("model_policy", "legal_mask_applied"))
            except Exception as exc:
                reason_tags.append(f"model_fallback:{exc.__class__.__name__}")
                ranking = []

        if method != "production_model" and self.profile.bid_samples >= 2:
            try:
                analysis = analyze_bid(
                    state,
                    machine,
                    actor,
                    rules,
                    requested_samples=max(2, self.profile.bid_samples),
                    seed=seed,
                    cancel_check=cancel_check,
                    exact_search_backend=self.exact_search_backend,
                )
                if score_authority is not None:
                    if analysis.actor != actor:
                        raise ValueError("primary bid analyzer actor drifted")
                    score_evidence = primary_decision_score_evidence_v1(
                        decision_kind="bid",
                        actor=actor,
                        public_prefix_hash=score_authority["publicPrefixHash"],
                        rules_hash=score_authority["rulesHash"],
                        analysis_identity_hash=score_authority[
                            "analysisIdentityHash"
                        ],
                        trace_analysis_profile_hash=score_authority[
                            "traceAnalysisProfileHash"
                        ],
                        cohort_seed_policy=score_authority["cohortSeedPolicy"],
                        cohort_index=score_authority["cohortIndex"],
                        cohort_seed=score_authority["cohortSeed"],
                        objective=BID_DECISION_OBJECTIVE_V1,
                        tie_break=BID_DECISION_TIE_BREAK_V1,
                        backend=score_authority["backend"],
                        method=analysis.method,
                        hidden_deal_support=analysis.hidden_deals,
                        effective_sample_size=float(analysis.hidden_deals),
                        recommended_action=analysis.recommended.to_dict(),
                        rows=bid_score_rows_v1(analysis.choices),
                        legal_action_set_hash=score_authority[
                            "legalActionSetHash"
                        ],
                    )
                ranked = [item.choice for item in analysis.choices]
                ranking = [
                    {
                        **item.choice.to_dict(),
                        "rank": index + 1,
                        "expectedMargin": item.expected_match_margin,
                        "winProbability": item.team_win_probability,
                        "contractSuccessProbability": item.contract_success_probability,
                        "downsideCvar10": item.downside_cvar10,
                        "bestShare": item.best_share,
                    }
                    for index, item in enumerate(analysis.choices)
                ]
                method = "paired_bid_simulation"
                confidence = max(0.25, min(1.0, analysis.hidden_deals / max(12.0, float(self.profile.bid_samples))))
                analysis_hidden_deals = int(analysis.hidden_deals)
                analysis_effective_samples = float(analysis.hidden_deals)
                reason_tags.extend(("shared_hidden_deal_cohort", "match_aware_buying"))
                best = analysis.choices[0]
                if (
                    score_authority is None
                    and ranked[0].action != "pass"
                    and best.expected_match_margin
                    < (0.5 - self.profile.buying_strength) * 18.0
                ):
                    pass_index = next((index for index, choice in enumerate(ranked) if choice.action == "pass"), None)
                    if pass_index is not None:
                        passed = ranked.pop(pass_index)
                        passed_metrics = ranking.pop(pass_index)
                        ranked.insert(0, passed)
                        ranking.insert(0, passed_metrics)
                        reason_tags.append("buy_threshold_shifted_to_pass")
            except InterruptedError:
                raise
            except Exception as exc:
                if score_authority is not None:
                    raise ValueError(
                        "primary bid decision analysis/evidence failed closed"
                    ) from exc
                reason_tags.append(f"bid_search_fallback:{exc.__class__.__name__}")

        if score_authority is not None and score_evidence is None:
            raise ValueError("non-forced primary bid score evidence is missing")
        if not ranking:
            action, trump = _choose_action(
                machine,
                {actor: tuple(state.hand(actor))},
                state.purchase_card,
                rules,
            )
            preferred_index = next(
                (index for index, choice in enumerate(choices) if choice.action == action and choice.trump_suit == trump),
                0,
            )
            preferred = choices[preferred_index]
            ranked = [preferred, *(choice for index, choice in enumerate(choices) if index != preferred_index)]
            ranking = [{**choice.to_dict(), "rank": index + 1, "heuristicPreferred": index == 0} for index, choice in enumerate(ranked)]
            reason_tags.extend(("five_card_public_heuristic", "observer_safe"))

        rng = SplitMix64(seed)
        intended = ranked[0]
        selected = intended
        selected_index = 0
        mistake_applied = False
        deception_applied = False
        error_rate = max(self.profile.mistake_probability, (1.0 - self.profile.skill) * 0.10)
        if len(ranked) > 1 and rng.next() / float(2**64) < error_rate:
            selected_index = 1 + int(rng.next() % (len(ranked) - 1))
            selected = ranked[selected_index]
            mistake_applied = True
            reason_tags.append("controlled_mistake")
        elif len(ranked) > 1 and self.profile.deception_probability > 0 and rng.next() / float(2**64) < self.profile.deception_probability:
            best_metrics = ranking[0]
            safe: list[int] = []
            for index, item in enumerate(ranking[1:], start=1):
                if "expectedMargin" not in best_metrics or "expectedMargin" not in item:
                    continue
                regret = float(best_metrics["expectedMargin"]) - float(item["expectedMargin"])
                win_drop = float(best_metrics.get("winProbability", 0.0)) - float(item.get("winProbability", 0.0))
                if regret <= self.profile.maximum_deception_regret and win_drop <= self.profile.maximum_deception_win_drop:
                    safe.append(index)
            if safe:
                selected_index = safe[int(rng.next() % len(safe))]
                selected = ranked[selected_index]
                deception_applied = True
                reason_tags.append("safe_deception")

        trace = {
            "schemaVersion": 2,
            "source": "bot",
            "profileId": self.profile.id,
            "profile": self.profile.to_dict(),
            "strategyId": self.descriptor.id,
            "strategyVersion": self.descriptor.version,
            "decisionSource": method,
            "modelId": _model_identity(self.model),
            "modelIdentity": model_runtime_identity(self.model),
            "observerSafe": True,
            "seed": int(seed),
            "confidence": (
                primary_decision_score_confidence_v1(score_evidence)
                if score_evidence is not None
                else round(confidence, 6)
            ),
            "analysisMethod": (
                score_evidence.method if score_evidence is not None else None
            ),
            "analysisHiddenDeals": analysis_hidden_deals,
            "analysisEffectiveSampleSize": analysis_effective_samples,
            "legalActionCount": len(choices),
            "intendedBestAction": intended.to_dict(),
            "selectedAction": selected.to_dict(),
            "selectedRank": selected_index,
            "mistakeApplied": mistake_applied,
            "deceptionApplied": deception_applied,
            "rankedActions": ranking,
            "reasonTags": reason_tags,
        }
        if score_evidence is not None:
            if canonical_json_bytes_v1(selected.to_dict()) != canonical_json_bytes_v1(
                score_evidence.recommended_action
            ):
                raise ValueError(
                    "primary bid selection differs from canonical score recommendation"
                )
            trace.update({
                "publicPrefixHash": score_evidence.public_prefix_hash,
                "rulesHash": score_evidence.rules_hash,
                "analysisIdentityHash": score_evidence.analysis_identity_hash,
                "traceAnalysisProfileHash": (
                    score_evidence.trace_analysis_profile_hash
                ),
                "cohortSeedPolicy": score_evidence.cohort_seed_policy,
                "cohortIndex": score_evidence.cohort_index,
                "cohortSeed": score_evidence.cohort_seed,
                "analysisObjective": score_evidence.objective,
                "analysisTieBreak": score_evidence.tie_break,
                "analysisBackend": score_evidence.backend,
                "legalActionSetHash": score_evidence.legal_action_set_hash,
                "primaryDecisionScoreEvidence": score_evidence.to_dict(),
                "primaryDecisionScoreEvidenceHash": score_evidence.sha256,
                "decisionScoreRowsHash": score_evidence.rows_hash,
                "forcedLegalityProof": None,
            })
            validate_primary_bid_decision_score_trace_v1(
                trace,
                score_authority,
                selected_action=selected.to_dict(),
                forced=False,
            )
        return selected.action, selected.trump_suit, trace

    def choose(
        self,
        state: RoundState,
        machine: BiddingMachine,
        rules: Any,
        *,
        seed: int,
    ) -> tuple[str, Any | None]:
        action, trump, _trace = self.choose_with_trace(state, machine, rules, seed=seed)
        return action, trump
