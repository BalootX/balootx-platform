"""Canonical actor-observer decision-score evidence.

The projections consume already ordered analyzer output; they never evaluate
or re-rank actions.  Raw hidden-world data is represented only by hashes and
counts.  Final Baloot scores, CP-SAT structural objectives, and content scores
are separate authorities and are intentionally absent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit
from baloot_platform.research.contracts import (
    BID_DECISION_OBJECTIVE_V1,
    RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1 as BID_DECISION_TIE_BREAK_V1,
)

from .models import ActionEvaluation, BidChoice, BidChoiceEvaluation, SEARCH_COMPLETION_STATUSES


_TYPE = "primary_decision_score_evidence_v1"
_KINDS = frozenset({"bid", "play"})
_MAX_COHORT_INDEX = (1 << 31) - 1
_MAX_SEED = (1 << 64) - 1
PRIMARY_DECISION_COHORT_SEED_POLICY_V1 = "actor_observer_cohort_seed_v1"
PLAY_DECISION_TIE_BREAK_V1 = (
    "configured_objective_then_match_win_probability_then_expected_round_margin_"
    "then_contract_success_probability_then_round_lead_probability_then_best_share_"
    "then_card_id_v1"
)
_ENVELOPE_KEYS = frozenset({
    "schemaVersion", "evidenceType", "truthMode", "decisionKind", "actor",
    "publicPrefixHash", "legalActionSetHash", "rulesHash",
    "analysisIdentityHash", "traceAnalysisProfileHash", "cohortSeedPolicy",
    "cohortIndex", "cohortSeed", "objective", "tieBreak", "backend",
    "method", "hiddenDealSupport", "effectiveSampleSize", "recommendedAction",
    "rows", "rowsHash", "sha256",
})
_BID_ROW_KEYS = frozenset({
    "rank", "actionId", "action", "expectedRoundMargin",
    "roundLeadProbability", "contractSuccessProbability", "downsideCvar10",
    "volatility", "bestShare", "samples",
})
_PLAY_ROW_KEYS = frozenset({
    "rank", "actionId", "action", "expectedRoundMargin",
    "expectedTeamAwardedPoints", "expectedOpponentAwardedPoints",
    "roundLeadProbability", "drawProbability", "roundLeadConfidence95",
    "bestShare", "samples", "effectiveSamples", "exactShare",
    "contractSuccessProbability", "matchWinProbability",
    "matchWinConfidence95", "matchValueModel", "supportMarginsHash",
    "supportMarginCount", "downsideCvar10", "volatility",
    "principalVariationHash", "principalVariationLength", "searchNodes",
    "searchCompletionStatus",
})


def canonical_json_bytes_v1(value: Any) -> bytes:
    """Return the exact UTF-8 representation used by this contract."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256_v1(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v1(value)).hexdigest()


def primary_decision_score_confidence_v1(
    evidence: Mapping[str, Any] | "PrimaryDecisionScoreEvidenceV1",
) -> float:
    """Derive confidence exclusively from the validated score-row authority."""

    frozen = (
        evidence
        if isinstance(evidence, PrimaryDecisionScoreEvidenceV1)
        else PrimaryDecisionScoreEvidenceV1.from_mapping(evidence)
    )
    denominator = 8.0 if frozen.decision_kind == "bid" else 10.0
    sampled = min(
        1.0,
        math.sqrt(max(1.0, float(frozen.effective_sample_size))) / denominator,
    )
    if frozen.decision_kind != "play" or not str(frozen.method).startswith(
        "exhaustive_"
    ):
        return sampled
    statuses = {str(item["searchCompletionStatus"]) for item in frozen.rows}
    if statuses == {"heuristic_complete"}:
        return 1.0
    if statuses == {"exact"}:
        exact_coverage = min(
            max(0.0, min(1.0, float(item["exactShare"])))
            for item in frozen.rows
        )
        return max(sampled, exact_coverage)
    return sampled


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes_v1(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "decision-score evidence must contain finite canonical JSON values"
        ) from exc


def _closed(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    raw = _json_copy(dict(value))
    unknown = sorted(set(raw) - keys)
    missing = sorted(keys - set(raw))
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown keys: {unknown}")
        if missing:
            parts.append(f"missing keys: {missing}")
        raise ValueError(f"{label} is not closed ({'; '.join(parts)})")
    return raw


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip() or not value.isprintable():
        raise ValueError(f"{label} must be a canonical printable string")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a canonical lowercase SHA-256 digest")
    return value


def _integer(value: Any, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its supported range")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its supported range")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} is above its supported range")
    return value


def _probability(value: Any, label: str) -> int | float:
    return _number(value, label, minimum=0.0, maximum=1.0)


def _interval(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    low = _probability(value[0], f"{label}[0]")
    high = _probability(value[1], f"{label}[1]")
    if low > high:
        raise ValueError(f"{label} lower bound exceeds its upper bound")


def _bid_action(value: Any) -> tuple[dict[str, Any], str]:
    raw = _closed(value, frozenset({"action", "trumpSuit"}), "bid action")
    action = _text(raw["action"], "bid action.action")
    if raw["trumpSuit"] is None:
        choice = BidChoice(action)
    else:
        try:
            choice = BidChoice(action, Suit.parse(raw["trumpSuit"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bid action.trumpSuit is invalid") from exc
    if raw != choice.to_dict():
        raise ValueError("bid action is not canonical")
    return raw, choice.id


def _play_action(value: Any) -> tuple[dict[str, Any], str]:
    keys = frozenset({"id", "code", "suit", "rank", "symbol", "label"})
    raw = _closed(value, keys, "play action")
    try:
        card = Card.parse(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("play action is invalid") from exc
    if raw != card.to_dict():
        raise ValueError("play action is not canonical")
    return raw, f"card:{card.id}"


def bid_score_rows_v1(
    evaluations: Sequence[BidChoiceEvaluation],
) -> tuple[dict[str, Any], ...]:
    """Project authoritative bid evaluations in their existing order."""

    rows = tuple({
        "rank": rank,
        "actionId": item.choice.id,
        "action": item.choice.to_dict(),
        "expectedRoundMargin": float(item.expected_match_margin),
        "roundLeadProbability": float(item.team_win_probability),
        "contractSuccessProbability": float(item.contract_success_probability),
        "downsideCvar10": float(item.downside_cvar10),
        "volatility": float(item.volatility),
        "bestShare": float(item.best_share),
        "samples": int(item.samples),
    } for rank, item in enumerate(evaluations, start=1))
    return _validate_rows("bid", rows)


def play_score_rows_v1(
    evaluations: Sequence[ActionEvaluation],
) -> tuple[dict[str, Any], ...]:
    """Project authoritative play evaluations in their existing order."""

    rows = tuple({
        "rank": rank,
        "actionId": f"card:{item.card.id}",
        "action": item.card.to_dict(),
        "expectedRoundMargin": float(item.expected_round_margin),
        "expectedTeamAwardedPoints": float(item.expected_team_awarded_points),
        "expectedOpponentAwardedPoints": float(item.expected_opponent_awarded_points),
        "roundLeadProbability": float(item.round_lead_probability),
        "drawProbability": float(item.draw_probability),
        "roundLeadConfidence95": [float(item.confidence_low), float(item.confidence_high)],
        "bestShare": float(item.best_share),
        "samples": int(item.samples),
        "effectiveSamples": float(item.effective_samples),
        "exactShare": float(item.exact_share),
        "contractSuccessProbability": float(item.contract_success_probability),
        "matchWinProbability": float(item.match_win_probability),
        "matchWinConfidence95": [
            float(item.match_confidence_low), float(item.match_confidence_high)
        ],
        "matchValueModel": str(item.match_value_model),
        "supportMarginsHash": canonical_sha256_v1(
            [float(value) for value in item.deal_margins]
        ),
        "supportMarginCount": len(item.deal_margins),
        "downsideCvar10": float(item.downside_cvar),
        "volatility": float(item.volatility),
        "principalVariationHash": canonical_sha256_v1(list(item.principal_variation)),
        "principalVariationLength": len(item.principal_variation),
        "searchNodes": int(item.search_nodes),
        "searchCompletionStatus": item.search_completion_status,
    } for rank, item in enumerate(evaluations, start=1))
    return _validate_rows("play", rows)


def _validate_rows(
    decision_kind: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if decision_kind not in _KINDS:
        raise ValueError(f"decisionKind must be one of {sorted(_KINDS)}")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("decision score rows must be a non-empty list or tuple")
    normalized = []
    for rank, value in enumerate(rows, start=1):
        keys = _BID_ROW_KEYS if decision_kind == "bid" else _PLAY_ROW_KEYS
        row = _closed(value, keys, f"{decision_kind} score row {rank}")
        if _integer(row["rank"], "rank") != rank:
            raise ValueError("score row ranks must preserve analyzer order")
        action, action_id = (
            _bid_action(row["action"])
            if decision_kind == "bid"
            else _play_action(row["action"])
        )
        if row["actionId"] != action_id:
            raise ValueError("score row actionId does not match its action")
        row["action"] = action
        for field in ("expectedRoundMargin", "downsideCvar10"):
            _number(row[field], field)
        for field in ("roundLeadProbability", "contractSuccessProbability", "bestShare"):
            _probability(row[field], field)
        _number(row["volatility"], "volatility", minimum=0.0)
        _integer(row["samples"], "samples")
        if decision_kind == "play":
            for field in ("expectedTeamAwardedPoints", "expectedOpponentAwardedPoints"):
                _number(row[field], field)
            for field in ("drawProbability", "exactShare", "matchWinProbability"):
                _probability(row[field], field)
            _interval(row["roundLeadConfidence95"], "roundLeadConfidence95")
            _interval(row["matchWinConfidence95"], "matchWinConfidence95")
            _number(row["effectiveSamples"], "effectiveSamples", minimum=0.0)
            _text(row["matchValueModel"], "matchValueModel")
            _hash(row["supportMarginsHash"], "supportMarginsHash")
            _integer(row["supportMarginCount"], "supportMarginCount")
            _hash(row["principalVariationHash"], "principalVariationHash")
            _integer(row["principalVariationLength"], "principalVariationLength")
            _integer(row["searchNodes"], "searchNodes")
            status = _text(row["searchCompletionStatus"], "searchCompletionStatus")
            if status not in SEARCH_COMPLETION_STATUSES:
                raise ValueError("searchCompletionStatus is unsupported")
        normalized.append(row)
    action_hashes = [canonical_sha256_v1(row["action"]) for row in normalized]
    if len(set(action_hashes)) != len(action_hashes):
        raise ValueError("decision score rows contain duplicate actions")
    return tuple(normalized)


def legal_action_set_hash_v1(actions: Iterable[Mapping[str, Any]]) -> str:
    """Hash an action set independently of ranking order."""

    hashes = sorted(canonical_sha256_v1(dict(action)) for action in actions)
    if not hashes or len(hashes) != len(set(hashes)):
        raise ValueError("legal action set must be non-empty and contain no duplicates")
    return canonical_sha256_v1(hashes)


def decision_score_rows_hash_v1(
    decision_kind: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    normalized = _validate_rows(decision_kind, rows)
    return canonical_sha256_v1({
        "schemaVersion": 1,
        "decisionKind": decision_kind,
        "rows": list(normalized),
    })


@dataclass(frozen=True, slots=True)
class PrimaryDecisionScoreEvidenceV1:
    decision_kind: str
    actor: str
    public_prefix_hash: str
    legal_action_set_hash: str
    rules_hash: str
    analysis_identity_hash: str
    trace_analysis_profile_hash: str
    cohort_seed_policy: str
    cohort_index: int
    cohort_seed: int
    objective: str
    tie_break: str
    backend: str
    method: str
    hidden_deal_support: int
    effective_sample_size: int | float
    recommended_action: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    rows_hash: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrimaryDecisionScoreEvidenceV1":
        raw = _closed(value, _ENVELOPE_KEYS, "decision score evidence")
        if raw["schemaVersion"] != 1 or isinstance(raw["schemaVersion"], bool):
            raise ValueError("unsupported decision score evidence schema")
        if raw["evidenceType"] != _TYPE or raw["truthMode"] != "actor_observer":
            raise ValueError("unsupported decision score evidence authority")
        kind = _text(raw["decisionKind"], "decisionKind")
        if kind not in _KINDS:
            raise ValueError(f"decisionKind must be one of {sorted(_KINDS)}")
        seats = {seat.name.lower() for seat in Seat}
        if raw["actor"] not in seats:
            raise ValueError("actor must be a canonical lowercase Baloot seat")
        rows = _validate_rows(kind, raw["rows"])
        rows_hash = decision_score_rows_hash_v1(kind, rows)
        if _hash(raw["rowsHash"], "rowsHash") != rows_hash:
            raise ValueError("decision score rows checksum mismatch")
        legal_hash = legal_action_set_hash_v1(row["action"] for row in rows)
        if _hash(raw["legalActionSetHash"], "legalActionSetHash") != legal_hash:
            raise ValueError("decision score legal action set checksum mismatch")
        recommended, _ = (
            _bid_action(raw["recommendedAction"])
            if kind == "bid"
            else _play_action(raw["recommendedAction"])
        )
        if canonical_json_bytes_v1(recommended) != canonical_json_bytes_v1(rows[0]["action"]):
            raise ValueError("recommendedAction must be the first ordered score row")
        hashes = {
            key: _hash(raw[key], key)
            for key in (
                "publicPrefixHash", "rulesHash", "analysisIdentityHash",
                "traceAnalysisProfileHash",
            )
        }
        policy = _text(raw["cohortSeedPolicy"], "cohortSeedPolicy")
        cohort_index = _integer(raw["cohortIndex"], "cohortIndex", _MAX_COHORT_INDEX)
        cohort_seed = _integer(raw["cohortSeed"], "cohortSeed", _MAX_SEED)
        for key in ("objective", "tieBreak", "backend", "method"):
            _text(raw[key], key)
        hidden_support = _integer(raw["hiddenDealSupport"], "hiddenDealSupport")
        effective_support = _number(
            raw["effectiveSampleSize"], "effectiveSampleSize", minimum=0.0
        )
        core = {key: raw[key] for key in _ENVELOPE_KEYS if key != "sha256"}
        digest = canonical_sha256_v1(core)
        if _hash(raw["sha256"], "sha256") != digest:
            raise ValueError("decision score evidence checksum mismatch")
        return cls(
            kind, raw["actor"], hashes["publicPrefixHash"], legal_hash,
            hashes["rulesHash"], hashes["analysisIdentityHash"],
            hashes["traceAnalysisProfileHash"], policy, cohort_index, cohort_seed,
            raw["objective"], raw["tieBreak"], raw["backend"], raw["method"],
            hidden_support, effective_support, recommended, rows, rows_hash, digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "evidenceType": _TYPE,
            "truthMode": "actor_observer",
            "decisionKind": self.decision_kind,
            "actor": self.actor,
            "publicPrefixHash": self.public_prefix_hash,
            "legalActionSetHash": self.legal_action_set_hash,
            "rulesHash": self.rules_hash,
            "analysisIdentityHash": self.analysis_identity_hash,
            "traceAnalysisProfileHash": self.trace_analysis_profile_hash,
            "cohortSeedPolicy": self.cohort_seed_policy,
            "cohortIndex": self.cohort_index,
            "cohortSeed": self.cohort_seed,
            "objective": self.objective,
            "tieBreak": self.tie_break,
            "backend": self.backend,
            "method": self.method,
            "hiddenDealSupport": self.hidden_deal_support,
            "effectiveSampleSize": self.effective_sample_size,
            "recommendedAction": copy.deepcopy(dict(self.recommended_action)),
            "rows": [copy.deepcopy(dict(row)) for row in self.rows],
            "rowsHash": self.rows_hash,
            "sha256": self.sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes_v1(self.to_dict())


def primary_decision_score_evidence_v1(
    *,
    decision_kind: str,
    actor: Seat | str,
    public_prefix_hash: str,
    rules_hash: str,
    analysis_identity_hash: str,
    trace_analysis_profile_hash: str,
    cohort_seed_policy: str,
    cohort_index: int,
    cohort_seed: int,
    objective: str,
    tie_break: str,
    backend: str,
    method: str,
    hidden_deal_support: int,
    effective_sample_size: float,
    recommended_action: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    legal_action_set_hash: str | None = None,
) -> PrimaryDecisionScoreEvidenceV1:
    """Build a closed, self-validating primary decision-score envelope."""

    normalized = _validate_rows(decision_kind, rows)
    computed_legal_hash = legal_action_set_hash_v1(row["action"] for row in normalized)
    if legal_action_set_hash is not None and legal_action_set_hash != computed_legal_hash:
        raise ValueError("supplied legal action set hash does not match score rows")
    try:
        seat = actor if isinstance(actor, Seat) else Seat.parse(actor)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("actor is invalid") from exc
    core = {
        "schemaVersion": 1,
        "evidenceType": _TYPE,
        "truthMode": "actor_observer",
        "decisionKind": decision_kind,
        "actor": seat.name.lower(),
        "publicPrefixHash": public_prefix_hash,
        "legalActionSetHash": computed_legal_hash,
        "rulesHash": rules_hash,
        "analysisIdentityHash": analysis_identity_hash,
        "traceAnalysisProfileHash": trace_analysis_profile_hash,
        "cohortSeedPolicy": cohort_seed_policy,
        "cohortIndex": cohort_index,
        "cohortSeed": cohort_seed,
        "objective": objective,
        "tieBreak": tie_break,
        "backend": backend,
        "method": method,
        "hiddenDealSupport": hidden_deal_support,
        "effectiveSampleSize": effective_sample_size,
        "recommendedAction": copy.deepcopy(dict(recommended_action)),
        "rows": list(normalized),
        "rowsHash": decision_score_rows_hash_v1(decision_kind, normalized),
    }
    return PrimaryDecisionScoreEvidenceV1.from_mapping({
        **core, "sha256": canonical_sha256_v1(core)
    })


def primary_decision_score_evidence_from_mapping_v1(
    value: Mapping[str, Any],
) -> PrimaryDecisionScoreEvidenceV1:
    return PrimaryDecisionScoreEvidenceV1.from_mapping(value)


def validate_primary_decision_score_evidence_v1(
    value: Mapping[str, Any] | PrimaryDecisionScoreEvidenceV1,
    **expected_bindings: Any,
) -> PrimaryDecisionScoreEvidenceV1:
    """Validate self-hashes and any independently supplied envelope bindings."""

    evidence = PrimaryDecisionScoreEvidenceV1.from_mapping(
        value.to_dict() if isinstance(value, PrimaryDecisionScoreEvidenceV1) else value
    )
    field_names = {
        "decision_kind": "decisionKind",
        "actor": "actor",
        "public_prefix_hash": "publicPrefixHash",
        "legal_action_set_hash": "legalActionSetHash",
        "rules_hash": "rulesHash",
        "analysis_identity_hash": "analysisIdentityHash",
        "trace_analysis_profile_hash": "traceAnalysisProfileHash",
        "cohort_seed_policy": "cohortSeedPolicy",
        "cohort_index": "cohortIndex",
        "cohort_seed": "cohortSeed",
    }
    unknown = sorted(set(expected_bindings) - set(field_names))
    if unknown:
        raise TypeError(f"unknown expected decision-score bindings: {unknown}")
    actual = evidence.to_dict()
    mismatched = []
    for argument, expected in expected_bindings.items():
        if isinstance(expected, Seat):
            expected = expected.name.lower()
        if expected is not None and actual[field_names[argument]] != expected:
            mismatched.append(field_names[argument])
    if mismatched:
        raise ValueError("decision score evidence binding mismatch: " + ", ".join(mismatched))
    return evidence


__all__ = [
    "BID_DECISION_OBJECTIVE_V1",
    "BID_DECISION_TIE_BREAK_V1",
    "PLAY_DECISION_TIE_BREAK_V1",
    "PRIMARY_DECISION_COHORT_SEED_POLICY_V1",
    "PrimaryDecisionScoreEvidenceV1",
    "bid_score_rows_v1",
    "canonical_json_bytes_v1",
    "canonical_sha256_v1",
    "decision_score_rows_hash_v1",
    "legal_action_set_hash_v1",
    "play_score_rows_v1",
    "primary_decision_score_confidence_v1",
    "primary_decision_score_evidence_from_mapping_v1",
    "primary_decision_score_evidence_v1",
    "validate_primary_decision_score_evidence_v1",
]
