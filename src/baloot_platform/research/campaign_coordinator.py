"""Deterministic CP-SAT shard orchestration and Baloot variant diversity."""

from __future__ import annotations

import hashlib
import json
from itertools import permutations, zip_longest
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit

from .scenario_contracts import canonical_json, canonical_sha256
from .sharding import ResearchShardSpecV1


# Leave process/receipt bookkeeping inside the governed 250 ms acknowledgement gate.
_CANCELLATION_ACKNOWLEDGEMENT_SECONDS = 0.225


@dataclass(frozen=True, slots=True)
class CampaignStageTimingV1:
    stage: str
    wall_time_ns: int
    cpu_time_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("campaign timing stage must not be empty")
        if (
            type(self.wall_time_ns) is not int
            or type(self.cpu_time_ns) is not int
            or self.wall_time_ns < 0
            or self.cpu_time_ns < 0
        ):
            raise ValueError("campaign timing values must be non-negative integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "wallTimeNs": self.wall_time_ns,
            "cpuTimeNs": self.cpu_time_ns,
        }


@dataclass(frozen=True, slots=True)
class CampaignVariantV1:
    """A semantic candidate plus execution-only timing evidence."""

    semantic_json: str
    semantic_hash: str
    symmetry_json: str
    symmetry_hash: str
    stage_timings: tuple[CampaignStageTimingV1, ...] = ()

    def __post_init__(self) -> None:
        try:
            semantic = json.loads(self.semantic_json)
            symmetry = json.loads(self.symmetry_json)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "campaign variant documents must be canonical JSON"
            ) from error
        if not isinstance(semantic, dict) or not isinstance(symmetry, dict):
            raise ValueError("campaign variant documents must be JSON objects")
        if (
            canonical_json(semantic) != self.semantic_json
            or canonical_sha256(semantic) != self.semantic_hash
            or canonical_json(symmetry) != self.symmetry_json
            or canonical_sha256(symmetry) != self.symmetry_hash
        ):
            raise ValueError("campaign variant identity does not match its documents")
        if any(
            not isinstance(item, CampaignStageTimingV1) for item in self.stage_timings
        ):
            raise TypeError(
                "campaign variant timings must be CampaignStageTimingV1 values"
            )

    @classmethod
    def build(
        cls,
        semantic_document: Mapping[str, Any],
        *,
        symmetry_document: Mapping[str, Any] | None = None,
        stage_timings: tuple[CampaignStageTimingV1, ...] = (),
    ) -> CampaignVariantV1:
        semantic_json = canonical_json(dict(semantic_document))
        symmetry_json = canonical_json(
            dict(symmetry_document)
            if symmetry_document is not None
            else dict(semantic_document)
        )
        return cls(
            semantic_json,
            canonical_sha256(json.loads(semantic_json)),
            symmetry_json,
            canonical_sha256(json.loads(symmetry_json)),
            tuple(stage_timings),
        )

    @property
    def semantic_document(self) -> dict[str, Any]:
        return json.loads(self.semantic_json)

    @property
    def symmetry_document(self) -> dict[str, Any]:
        return json.loads(self.symmetry_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semanticHash": self.semantic_hash,
            "semanticDocument": self.semantic_document,
            "symmetryHash": self.symmetry_hash,
            "symmetryDocument": self.symmetry_document,
            "stageTimings": [item.to_dict() for item in self.stage_timings],
        }


@dataclass(frozen=True, slots=True)
class CampaignShardOutcomeV1:
    variants: tuple[CampaignVariantV1, ...]
    stage_timings: tuple[CampaignStageTimingV1, ...] = ()
    partial_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(item, CampaignVariantV1) for item in self.variants):
            raise TypeError("campaign shard variants must be CampaignVariantV1 values")
        if any(
            not isinstance(item, CampaignStageTimingV1) for item in self.stage_timings
        ):
            raise TypeError(
                "campaign shard timings must be CampaignStageTimingV1 values"
            )
        if self.partial_evidence is not None:
            object.__setattr__(
                self,
                "partial_evidence",
                json.loads(canonical_json(dict(self.partial_evidence))),
            )


class CampaignCancellationTokenV1:
    """Cooperative cancellation token passed to pure shard solver callables."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def _cancel(self) -> None:
        self._event.set()


class CampaignAttemptProgressV1:
    """Thread-safe execution telemetry; never part of semantic identity."""

    __slots__ = ("_json", "_lock", "_checkpoints", "_worker_cpu_ns")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._json = canonical_json({"currentStage": "queued"})
        self._checkpoints: dict[str, str] = {}
        self._worker_cpu_ns: int | None = None

    def sample_worker_cpu(self, cpu_time_ns: int) -> None:
        if type(cpu_time_ns) is not int or cpu_time_ns < 0:
            raise ValueError("worker CPU sample requires nonnegative integer nanoseconds")
        with self._lock:
            self._worker_cpu_ns = max(self._worker_cpu_ns or 0, cpu_time_ns)

    def update(self, evidence: Mapping[str, Any]) -> None:
        document = dict(evidence)
        if (
            not isinstance(document.get("currentStage"), str)
            or not document["currentStage"]
        ):
            raise ValueError("campaign progress requires a current stage")
        with self._lock:
            # Stage telemetry replaces stale solver details, but attempted work
            # must survive until this attempt's hash-bound terminal receipt.
            previous_count = json.loads(self._json).get("completedCandidates")
            count = document.get("completedCandidates", 0)
            if type(previous_count) is int and previous_count >= 0 and type(count) is int and count >= 0:
                document["completedCandidates"] = max(previous_count, count)
            self._json = canonical_json(document)

    def checkpoint(self, evidence: Mapping[str, Any] | str) -> None:
        # Owned process workers serialize once, outside the coordinator's GIL.
        # Snapshot decoding still rejects malformed transport evidence.
        value = evidence if isinstance(evidence, str) else canonical_json(dict(evidence))
        identity = hashlib.sha256(value.encode("utf-8")).hexdigest()
        with self._lock:
            self._checkpoints[identity] = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value = self._json
            checkpoints = tuple(sorted(self._checkpoints.items()))
            worker_cpu_ns = self._worker_cpu_ns
        return {
            **json.loads(value),
            **({"workerCpuAccounting": {"cpuTimeNs": worker_cpu_ns,
                "coverage": "sampled_lower_bound"}} if worker_cpu_ns is not None else {}),
            **({"qualityCheckpoints": {
                identity: json.loads(document) for identity, document in checkpoints
            }} if checkpoints else {}),
        }


@dataclass(frozen=True, slots=True)
class CampaignShardRequestV1:
    shard: ResearchShardSpecV1
    attempt: int
    cancellation: CampaignCancellationTokenV1
    accepted_variants: tuple[CampaignVariantV1, ...] = ()
    cp_sat_worker_count: int = 1
    progress_callback: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, repr=False, compare=False
    )
    progress: CampaignAttemptProgressV1 = field(
        default_factory=CampaignAttemptProgressV1, repr=False, compare=False
    )
    cancellation_acknowledgement: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    evidence_transfer_lock: Any = field(
        default_factory=threading.RLock, repr=False, compare=False
    )
    deadline_ns: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or not (
            1 <= self.attempt <= self.shard.maximum_attempts
        ):
            raise ValueError("campaign shard attempt is outside its governed range")
        if type(self.cp_sat_worker_count) is not int or self.cp_sat_worker_count != 1:
            raise ValueError("campaign shards must declare one CP-SAT worker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard.to_dict(),
            "attempt": self.attempt,
            "replenishmentAcceptedSemanticHashes": [
                item.semantic_hash for item in self.accepted_variants
            ],
            "cpSatWorkerCount": self.cp_sat_worker_count,
        }

    def report_progress(self, evidence: Mapping[str, Any]) -> None:
        value = {
            **dict(evidence),
            "shardIndex": self.shard.shard_index,
            "shardId": self.shard.shard_id,
            "attempt": self.attempt,
        }
        self.progress.update(value)
        if self.progress_callback is not None:
            try:
                self.progress_callback(value)
            except Exception:  # Progress is operational evidence, never solver authority.
                pass

    def progress_snapshot(self) -> dict[str, Any]:
        # An acknowledged cancellation must retain any already-started private frame.
        with self.evidence_transfer_lock:
            return self.progress.snapshot()

    def report_checkpoint(self, evidence: Mapping[str, Any] | str) -> None:
        """Retain private worker evidence without calling the live UI callback."""
        with self.evidence_transfer_lock:
            self.progress.checkpoint(evidence)

    def report_worker_cpu_sample(self, cpu_time_ns: int) -> None:
        self.progress.sample_worker_cpu(cpu_time_ns)

    def cancellation_requested(self) -> bool:
        cancelled = self.cancellation.is_cancelled
        if cancelled:
            self.cancellation_acknowledgement.set()
        return cancelled


@dataclass(frozen=True, slots=True)
class CampaignBudgetV1:
    result_count: int
    shard_worker_count: int
    maximum_wall_time_seconds: float
    cancellation_poll_milliseconds: int = 25
    cp_sat_workers_per_solve: int = 1

    def __post_init__(self) -> None:
        if type(self.result_count) is not int or self.result_count < 1:
            raise ValueError("campaign result count must be a positive integer")
        if type(self.shard_worker_count) is not int or self.shard_worker_count < 1:
            raise ValueError("campaign shard worker count must be positive")
        if (
            isinstance(self.maximum_wall_time_seconds, bool)
            or not isinstance(self.maximum_wall_time_seconds, (int, float))
            or not math.isfinite(self.maximum_wall_time_seconds)
            or self.maximum_wall_time_seconds <= 0
        ):
            raise ValueError("campaign wall-time budget must be positive")
        if type(self.cancellation_poll_milliseconds) is not int or not (
            1 <= self.cancellation_poll_milliseconds <= 250
        ):
            raise ValueError("campaign cancellation polling must be in [1, 250] ms")
        if (
            type(self.cp_sat_workers_per_solve) is not int
            or self.cp_sat_workers_per_solve != 1
        ):
            raise ValueError("campaign CP-SAT worker count must be exactly one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resultCount": self.result_count,
            "shardWorkerCount": self.shard_worker_count,
            "maximumWallTimeSeconds": self.maximum_wall_time_seconds,
            "cancellationPollMilliseconds": self.cancellation_poll_milliseconds,
            "cpSatWorkersPerSolve": self.cp_sat_workers_per_solve,
        }


class RetryableCampaignShardError(RuntimeError):
    def __init__(self, code: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("retryable campaign shard error code must not be empty")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CampaignAttemptReceiptV1:
    shard_index: int
    shard_id: str
    attempt: int
    status: str
    variant_hashes: tuple[str, ...]
    stage_timings: tuple[CampaignStageTimingV1, ...]
    error_code: str | None = None
    partial_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.partial_evidence is not None:
            object.__setattr__(
                self,
                "partial_evidence",
                json.loads(canonical_json(dict(self.partial_evidence))),
            )

    def to_dict(self) -> dict[str, Any]:
        document = {
            "shardIndex": self.shard_index,
            "shardId": self.shard_id,
            "attempt": self.attempt,
            "status": self.status,
            "variantHashes": list(self.variant_hashes),
            "stageTimings": [item.to_dict() for item in self.stage_timings],
            "errorCode": self.error_code,
        }
        if self.partial_evidence is not None:
            document["partialEvidence"] = json.loads(
                canonical_json(dict(self.partial_evidence))
            )
        return document


@dataclass(frozen=True, slots=True)
class CampaignVariantReceiptV1:
    semantic_hash: str
    symmetry_hash: str
    shard_index: int
    attempt: int
    accepted: bool
    rejection_reason: str | None
    duplicate_key_hash: str
    symmetry_key_hash: str | None
    stage_timings: tuple[CampaignStageTimingV1, ...]
    nearest_accepted_distance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "semanticHash": self.semantic_hash,
            "symmetryHash": self.symmetry_hash,
            "shardIndex": self.shard_index,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "rejectionReason": self.rejection_reason,
            "duplicateKeyHash": self.duplicate_key_hash,
            "symmetryKeyHash": self.symmetry_key_hash,
            "nearestAcceptedDistance": self.nearest_accepted_distance,
            "stageTimings": [item.to_dict() for item in self.stage_timings],
        }


@dataclass(frozen=True, slots=True)
class CampaignReceiptV1:
    status: str
    shards: tuple[ResearchShardSpecV1, ...]
    budget: CampaignBudgetV1
    duplicate_policy_id: str
    symmetry_policy_id: str | None
    distance_policy_id: str | None
    minimum_distance: int | None
    attempts: tuple[CampaignAttemptReceiptV1, ...]
    variants: tuple[CampaignVariantReceiptV1, ...]
    accepted_variants: tuple[CampaignVariantV1, ...]
    semantic_sha256: str | None
    wall_time_ns: int
    cpu_time_ns: int
    receipt_sha256: str
    cpu_time_coverage: str | None = None

    def _semantic_core_document(self) -> dict[str, Any] | None:
        if self.status not in {"completed", "not_found_within_budget"}:
            return None
        return {
            "schemaVersion": 1,
            "status": self.status,
            "recipeHash": self.shards[0].recipe_hash,
            "shardPartitions": [
                {
                    "shardIndex": item.shard_index,
                    "seedStart": item.seed_start,
                    "seedCount": item.seed_count,
                    "maximumAttempts": item.maximum_attempts,
                }
                for item in self.shards
            ],
            "requestedResultCount": self.budget.result_count,
            "duplicatePolicyId": self.duplicate_policy_id,
            "symmetryPolicyId": self.symmetry_policy_id,
            "distancePolicyId": self.distance_policy_id,
            "minimumDistance": self.minimum_distance,
            "targetSelectionPolicyId": _TARGET_PREFIX_POLICY,
            "acceptedSemanticHashes": [
                item.semantic_hash for item in self.accepted_variants
            ],
        }

    def semantic_core(self) -> dict[str, Any] | None:
        return (
            self._semantic_core_document()
            if self.semantic_sha256 is not None
            else None
        )

    def _receipt_core(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "shards": [item.to_dict() for item in self.shards],
            "budget": self.budget.to_dict(),
            "workerDeclaration": {
                "shardWorkerCount": self.budget.shard_worker_count,
                "cpSatWorkersPerSolve": 1,
            },
            "duplicatePolicyId": self.duplicate_policy_id,
            "symmetryPolicyId": self.symmetry_policy_id,
            "distancePolicyId": self.distance_policy_id,
            "minimumDistance": self.minimum_distance,
            "targetSelectionPolicyId": _TARGET_PREFIX_POLICY,
            "attempts": [item.to_dict() for item in self.attempts],
            "variants": [item.to_dict() for item in self.variants],
            "acceptedVariants": [item.to_dict() for item in self.accepted_variants],
            "semanticSha256": self.semantic_sha256,
            "wallTimeNs": self.wall_time_ns,
            "cpuTimeNs": self.cpu_time_ns,
            **({"cpuTimeCoverage": self.cpu_time_coverage}
               if self.cpu_time_coverage is not None else {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._receipt_core(), "receiptSha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class _AttemptExecution:
    request: CampaignShardRequestV1
    outcome: CampaignShardOutcomeV1 | None
    retryable: bool
    error_code: str | None
    timing: CampaignStageTimingV1


CampaignShardSolverV1 = Callable[[CampaignShardRequestV1], CampaignShardOutcomeV1]
CampaignVariantKeyV1 = Callable[[CampaignVariantV1], str]
CampaignVariantDistanceV1 = Callable[[CampaignVariantV1, CampaignVariantV1], int]


def _semantic_key(variant: CampaignVariantV1) -> str:
    return variant.semantic_hash


_RULES_PRESERVING_POLICY = "rules_preserving_v1"
_DECISION_DISTANCE_POLICY = "decision_state_card_hamming_v1"
SEMANTIC_DISTANCE_POLICY_V1 = (
    "decision_state_and_principal_variation_thresholds_v1"
)
_TARGET_PREFIX_POLICY = "anchor_descending_partition_prefix_v1"


def _seat_offset(value: Any, acting_seat: Seat) -> int:
    return (int(Seat.parse(value)) - int(acting_seat)) % 4


def _decision_actor_v1(
    source: Mapping[str, Any],
    acting_seat: Seat | str | int | None,
) -> Seat:
    bidding = source.get("bidding")
    actor_value = acting_seat
    if actor_value is None:
        actor_value = (
            source.get("nextActor")
            or source.get("currentPlayer")
            or (dict(bidding).get("actor") if isinstance(bidding, Mapping) else None)
        )
    if actor_value is None:
        raise ValueError("decision diversity requires a current actor")
    return Seat.parse(actor_value)


def _decision_card_statuses_v1(
    source: Mapping[str, Any], actor: Seat
) -> tuple[list[int], list[tuple[int, int]]]:
    hands = source.get("hands")
    if not isinstance(hands, Mapping) or set(hands) != {
        seat.name.lower() for seat in Seat
    }:
        raise ValueError("decision diversity requires all four canonical hands")

    statuses = [8] * 32
    active_cards: set[int] = set()
    for seat in Seat:
        cards = hands[seat.name.lower()]
        if not isinstance(cards, list):
            raise TypeError("decision diversity hands must be card lists")
        for value in cards:
            card_id = Card.parse(value).id
            if card_id in active_cards:
                raise ValueError("decision diversity state duplicates an active card")
            active_cards.add(card_id)
            statuses[card_id] = _seat_offset(seat, actor)

    current = source.get("currentTrick")
    current_plays = [] if current is None else dict(current).get("plays")
    if not isinstance(current_plays, list):
        raise TypeError("decision diversity current trick plays must be a list")
    parsed_plays: list[tuple[int, int]] = []
    for row in current_plays:
        play = dict(row)
        card_id = Card.parse(play.get("card")).id
        if card_id in active_cards:
            raise ValueError("decision diversity state duplicates an active card")
        active_cards.add(card_id)
        relative_seat = _seat_offset(play.get("seat"), actor)
        statuses[card_id] = 4 + relative_seat
        parsed_plays.append((relative_seat, card_id))

    purchase = source.get("purchaseCard")
    if purchase is not None:
        purchase_id = Card.parse(purchase).id
        if purchase_id not in active_cards:
            statuses[purchase_id] = 9
    return statuses, parsed_plays


def decision_state_card_vector_v1(
    state: Mapping[str, Any],
    *,
    acting_seat: Seat | str | int | None = None,
) -> tuple[int, ...]:
    source = dict(state)
    actor = _decision_actor_v1(source, acting_seat)
    statuses, _plays = _decision_card_statuses_v1(source, actor)
    return tuple(statuses)


def build_rules_preserving_diversity_document_v1(
    state: Mapping[str, Any],
    *,
    acting_seat: Seat | str | int | None = None,
) -> dict[str, Any]:
    """Canonicalize a complete decision state under legal seat/suit renaming."""

    source = dict(state)
    actor = _decision_actor_v1(source, acting_seat)
    statuses, parsed_plays = _decision_card_statuses_v1(source, actor)

    contract = str(source.get("contract") or "").lower()
    if contract not in {"", "sun", "hokum", "ashkal"}:
        raise ValueError("decision diversity requires a supported contract")
    trump = source.get("trumpSuit")
    trump_suit = Suit.parse(trump) if trump is not None else None
    if contract == "hokum" and trump_suit is None:
        raise ValueError("Hokum decision diversity requires trumpSuit")

    candidates: list[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = []
    for order in permutations(range(4)):
        suit_map = {original: canonical for canonical, original in enumerate(order)}
        if trump_suit is not None and suit_map[int(trump_suit)] != 0:
            continue
        vector = [8] * 32
        for card_id, status in enumerate(statuses):
            canonical_id = suit_map[card_id // 8] * 8 + card_id % 8
            vector[canonical_id] = status
        plays = tuple(
            (relative_seat, suit_map[card_id // 8], card_id % 8)
            for relative_seat, card_id in parsed_plays
        )
        candidates.append((tuple(vector), plays))
    vector, plays = min(candidates)

    current = source.get("currentTrick")
    current_leader = None if current is None else dict(current).get("leader")
    return {
        "schemaVersion": 1,
        "policyId": _RULES_PRESERVING_POLICY,
        "distancePolicyId": _DECISION_DISTANCE_POLICY,
        "contract": "sun" if contract == "ashkal" else contract or "none",
        "buyerOffset": (
            None
            if source.get("buyer") is None
            else _seat_offset(source["buyer"], actor)
        ),
        "dealerOffset": _seat_offset(source.get("dealer"), actor),
        "currentLeaderOffset": (
            None if current_leader is None else _seat_offset(current_leader, actor)
        ),
        "decisionStateCardVector": list(vector),
        "currentTrickVector": [list(value) for value in plays],
    }


def rules_preserving_symmetry_key_v1(variant: CampaignVariantV1) -> str:
    document = variant.symmetry_document
    _validated_decision_vector(document)
    return variant.symmetry_hash


def _validated_decision_vector(document: Mapping[str, Any]) -> tuple[int, ...]:
    required = {
        "schemaVersion",
        "policyId",
        "distancePolicyId",
        "contract",
        "buyerOffset",
        "dealerOffset",
        "currentLeaderOffset",
        "decisionStateCardVector",
        "currentTrickVector",
    }
    if set(document) != required:
        raise ValueError("campaign rules-preserving diversity shape is invalid")
    if (
        document.get("schemaVersion") != 1
        or document.get("policyId") != _RULES_PRESERVING_POLICY
        or document.get("distancePolicyId") != _DECISION_DISTANCE_POLICY
        or document.get("contract") not in {"none", "sun", "hokum"}
    ):
        raise ValueError("campaign rules-preserving diversity identity is invalid")
    if any(
        value is not None and (type(value) is not int or not 0 <= value < 4)
        for value in (
            document.get("buyerOffset"),
            document.get("dealerOffset"),
            document.get("currentLeaderOffset"),
        )
    ):
        raise ValueError("campaign rules-preserving seat offsets are invalid")
    vector = document.get("decisionStateCardVector")
    if (
        not isinstance(vector, list)
        or len(vector) != 32
        or any(type(value) is not int or not 0 <= value <= 9 for value in vector)
    ):
        raise ValueError("campaign decision-state vectors must contain 32 cards")
    plays = document.get("currentTrickVector")
    if not isinstance(plays, list) or any(
        not isinstance(play, list)
        or len(play) != 3
        or any(type(value) is not int for value in play)
        or not (0 <= play[0] < 4 and 0 <= play[1] < 4 and 0 <= play[2] < 8)
        for play in plays
    ):
        raise ValueError("campaign current-trick vector is invalid")
    return tuple(vector)


def decision_state_card_hamming_distance_v1(
    left: CampaignVariantV1,
    right: CampaignVariantV1,
) -> int:
    vectors = tuple(
        _validated_decision_vector(variant.symmetry_document)
        for variant in (left, right)
    )
    return sum(a != b for a, b in zip(vectors[0], vectors[1], strict=True))


def principal_variation_action_hamming_distance_v1(
    left: CampaignVariantV1,
    right: CampaignVariantV1,
) -> int:
    values = []
    for variant in (left, right):
        actions = variant.semantic_document.get("principalVariationActionIds")
        if not isinstance(actions, list) or any(
            not isinstance(action, str) or not action for action in actions
        ):
            raise ValueError(
                "campaign principal variation must contain stable action IDs"
            )
        values.append(actions)
    return sum(
        left_action != right_action
        for left_action, right_action in zip_longest(values[0], values[1])
    )


def semantic_threshold_distance_v1(
    left: CampaignVariantV1,
    right: CampaignVariantV1,
    *,
    minimum_decision_state_distance: int,
    minimum_principal_variation_distance: int,
) -> int:
    """Count how many governed semantic-distance gates the pair satisfies."""

    return int(
        decision_state_card_hamming_distance_v1(left, right)
        >= minimum_decision_state_distance
    ) + int(
        principal_variation_action_hamming_distance_v1(left, right)
        >= minimum_principal_variation_distance
    )


def select_campaign_variants_v1(
    occurrences: tuple[tuple[int, int, CampaignVariantV1], ...],
    *,
    result_count: int,
    duplicate_key: CampaignVariantKeyV1 = _semantic_key,
    symmetry_key: CampaignVariantKeyV1 | None = None,
    distance: CampaignVariantDistanceV1 | None = None,
    minimum_distance: int | None = None,
) -> tuple[tuple[CampaignVariantV1, ...], tuple[CampaignVariantReceiptV1, ...]]:
    """Select a canonical, duplicate-free campaign result set."""

    if type(result_count) is not int or result_count < 1:
        raise ValueError("campaign result count must be a positive integer")
    if (distance is None) != (minimum_distance is None):
        raise ValueError("campaign distance hook and minimum must be supplied together")
    if minimum_distance is not None and (
        type(minimum_distance) is not int or minimum_distance < 1
    ):
        raise ValueError("campaign minimum distance must be a positive integer")
    ordered = sorted(
        (
            (variant.semantic_hash, shard_index, attempt, variant)
            for shard_index, attempt, variant in occurrences
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    documents_by_hash: dict[str, str] = {}
    duplicate_keys: set[str] = set()
    symmetry_keys: set[str] = set()
    accepted: list[CampaignVariantV1] = []
    receipts: list[CampaignVariantReceiptV1] = []
    for semantic_hash, shard_index, attempt, variant in ordered:
        known_document = documents_by_hash.setdefault(
            semantic_hash, variant.semantic_json
        )
        if known_document != variant.semantic_json:
            raise RuntimeError("campaign semantic hash collision detected")
        candidate_duplicate_key = duplicate_key(variant)
        if not isinstance(candidate_duplicate_key, str) or not candidate_duplicate_key:
            raise ValueError("campaign duplicate hook must return a non-empty string")
        candidate_symmetry_key = symmetry_key(variant) if symmetry_key else None
        if candidate_symmetry_key is not None and (
            not isinstance(candidate_symmetry_key, str) or not candidate_symmetry_key
        ):
            raise ValueError("campaign symmetry hook must return a non-empty string")
        candidate_distances = (
            tuple(distance(variant, existing) for existing in accepted)
            if distance is not None
            else ()
        )
        if any(type(value) is not int or value < 0 for value in candidate_distances):
            raise ValueError("campaign distance hook must return non-negative integers")
        nearest_distance = min(candidate_distances, default=None)
        if candidate_duplicate_key in duplicate_keys:
            rejection = "duplicate"
        elif (
            candidate_symmetry_key is not None
            and candidate_symmetry_key in symmetry_keys
        ):
            rejection = "symmetry"
        elif nearest_distance is not None and nearest_distance < minimum_distance:
            rejection = "decision_state_distance"
        elif len(accepted) >= result_count:
            rejection = "result_count_reached"
        else:
            rejection = None
            duplicate_keys.add(candidate_duplicate_key)
            if candidate_symmetry_key is not None:
                symmetry_keys.add(candidate_symmetry_key)
            accepted.append(variant)
        receipts.append(
            CampaignVariantReceiptV1(
                semantic_hash,
                variant.symmetry_hash,
                shard_index,
                attempt,
                rejection is None,
                rejection,
                canonical_sha256({"key": candidate_duplicate_key}),
                (
                    canonical_sha256({"key": candidate_symmetry_key})
                    if candidate_symmetry_key is not None
                    else None
                ),
                variant.stage_timings,
                nearest_distance,
            )
        )
    return tuple(accepted), tuple(receipts)


class DeterministicCpSatCampaignCoordinatorV1:
    """Run fixed independent shards and merge only after their terminal epoch."""

    def __init__(
        self,
        *,
        duplicate_key: CampaignVariantKeyV1 = _semantic_key,
        duplicate_policy_id: str = "semantic_hash_v1",
        symmetry_key: CampaignVariantKeyV1 | None = None,
        symmetry_policy_id: str | None = None,
        distance: CampaignVariantDistanceV1 | None = None,
        minimum_distance: int | None = None,
        distance_policy_id: str | None = None,
    ) -> None:
        if not isinstance(duplicate_policy_id, str) or not duplicate_policy_id:
            raise ValueError("campaign duplicate policy identity must not be empty")
        if (symmetry_key is None) != (symmetry_policy_id is None):
            raise ValueError(
                "campaign symmetry hook and identity must be supplied together"
            )
        if symmetry_policy_id is not None and (
            not isinstance(symmetry_policy_id, str) or not symmetry_policy_id
        ):
            raise ValueError("campaign symmetry policy identity must not be empty")
        if (
            len(
                {distance is None, minimum_distance is None, distance_policy_id is None}
            )
            != 1
        ):
            raise ValueError(
                "campaign distance hook, minimum, and identity must be supplied together"
            )
        if minimum_distance is not None and (
            type(minimum_distance) is not int or minimum_distance < 1
        ):
            raise ValueError("campaign minimum distance must be a positive integer")
        if distance_policy_id is not None and (
            not isinstance(distance_policy_id, str) or not distance_policy_id
        ):
            raise ValueError("campaign distance policy identity must not be empty")
        self._duplicate_key = duplicate_key
        self._duplicate_policy_id = duplicate_policy_id
        self._symmetry_key = symmetry_key
        self._symmetry_policy_id = symmetry_policy_id
        self._distance = distance
        self._minimum_distance = minimum_distance
        self._distance_policy_id = distance_policy_id

    def run(
        self,
        *,
        shards: tuple[ResearchShardSpecV1, ...],
        solver: CampaignShardSolverV1,
        budget: CampaignBudgetV1,
        cancellation_event: threading.Event | None = None,
        cancellation_check: Callable[[], bool] | None = None,
        replenish_diversity_rejections: bool = False,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> CampaignReceiptV1:
        ordered_shards = self._validate_shards(shards)
        started_wall = time.perf_counter_ns()
        started_cpu = time.process_time_ns()
        deadline = started_wall + int(budget.maximum_wall_time_seconds * 1e9)
        poll_seconds = budget.cancellation_poll_milliseconds / 1000.0
        token = CampaignCancellationTokenV1()
        executor = ThreadPoolExecutor(
            max_workers=min(budget.shard_worker_count, len(ordered_shards)),
            thread_name_prefix="research-cp-sat-shard",
        )
        active: dict[Future[_AttemptExecution], CampaignShardRequestV1] = {}
        attempts: list[CampaignAttemptReceiptV1] = []
        completed: dict[
            tuple[str, int],
            tuple[CampaignShardRequestV1, CampaignShardOutcomeV1],
        ] = {}
        failed = False
        stop_status: str | None = None
        target_shard_ids: frozenset[str] | None = None
        replenishment_started = False
        canonical_target_order = (
            ordered_shards[:1] + tuple(reversed(ordered_shards[1:]))
        )

        def canonical_target_prefix() -> frozenset[str] | None:
            prefix: set[str] = set()
            for shard in canonical_target_order:
                prefix.add(shard.shard_id)
                if not any(
                    request.shard.shard_id == shard.shard_id
                    for request, _outcome in completed.values()
                ):
                    return None
                prefix_completed = {
                    key: value
                    for key, value in completed.items()
                    if value[0].shard.shard_id in prefix
                }
                accepted, _receipts = self._merge(
                    prefix_completed, budget.result_count
                )
                if len(accepted) >= budget.result_count:
                    return frozenset(prefix)
            return None

        def submit(
            shard: ResearchShardSpecV1,
            attempt: int,
            accepted_variants: tuple[CampaignVariantV1, ...] = (),
        ) -> None:
            nonlocal replenishment_started
            if attempt > 1:
                replenishment_started = True
            request = CampaignShardRequestV1(
                shard,
                attempt,
                token,
                accepted_variants,
                progress_callback=progress_callback,
                deadline_ns=deadline,
            )
            request.report_progress({"currentStage": "queued"})
            active[executor.submit(self._invoke, solver, request)] = request

        for shard in canonical_target_order:
            submit(shard, 1)

        try:
            while True:
                while active:
                    if ((cancellation_event is not None and cancellation_event.is_set())
                            or (cancellation_check is not None and cancellation_check())):
                        stop_status = "cancelled"
                        break
                    if time.perf_counter_ns() >= deadline:
                        stop_status = "budget_exhausted"
                        break
                    done, _ = wait(
                        tuple(active), timeout=poll_seconds, return_when=FIRST_COMPLETED
                    )
                    for future in sorted(
                        done,
                        key=lambda item: (
                            active[item].shard.shard_index,
                            active[item].attempt,
                        ),
                    ):
                        request = active.pop(future)
                        execution = future.result()
                        if execution.outcome is not None:
                            outcome = execution.outcome
                            attempts.append(
                                CampaignAttemptReceiptV1(
                                    request.shard.shard_index,
                                    request.shard.shard_id,
                                    request.attempt,
                                    "completed",
                                    tuple(
                                        item.semantic_hash for item in outcome.variants
                                    ),
                                    (*outcome.stage_timings, execution.timing),
                                    partial_evidence=outcome.partial_evidence,
                                )
                            )
                            completed[(request.shard.shard_id, request.attempt)] = (
                                request,
                                outcome,
                            )
                            if progress_callback is not None:
                                _accepted, selection_receipts = self._merge(
                                    completed, budget.result_count
                                )
                                for receipt in selection_receipts:
                                    if (
                                        receipt.shard_index
                                        == request.shard.shard_index
                                        and receipt.attempt == request.attempt
                                        and not receipt.accepted
                                    ):
                                        request.report_progress(
                                            {
                                                "currentStage": "variant_selection_terminal",
                                                "candidateStatus": receipt.rejection_reason,
                                                "candidateHash": receipt.semantic_hash,
                                                "nearestAcceptedDistance": receipt.nearest_accepted_distance,
                                            }
                                        )
                            partial = dict(outcome.partial_evidence or {})
                            retry_eligible = (
                                partial.get("proofRetryEligible") is True
                                or partial.get("attemptRetryEligible") is True
                            )
                            if (
                                retry_eligible
                                and request.attempt < request.shard.maximum_attempts
                                and not (
                                    replenish_diversity_rejections
                                    and outcome.variants
                                    and partial.get("attemptRetryEligible") is True
                                    and partial.get("proofRetryEligible") is not True
                                )
                            ):
                                retry_stage = (
                                    "proof_retry_scheduled"
                                    if partial.get("proofRetryEligible") is True
                                    else "attempt_retry_scheduled"
                                )
                                request.report_progress(
                                    {
                                        "currentStage": retry_stage,
                                        "retryAttempt": request.attempt + 1,
                                        "candidateStatus": (
                                            "proof_incomplete"
                                            if partial.get("proofRetryEligible") is True
                                            else "search_incomplete"
                                        ),
                                    }
                                )
                                submit(
                                    request.shard,
                                    request.attempt + 1,
                                    request.accepted_variants,
                                )
                        else:
                            can_retry = (
                                execution.retryable
                                and request.attempt < request.shard.maximum_attempts
                            )
                            attempts.append(
                                CampaignAttemptReceiptV1(
                                    request.shard.shard_index,
                                    request.shard.shard_id,
                                    request.attempt,
                                    "retryable_failure" if can_retry else "failed",
                                    (),
                                    (execution.timing,),
                                    execution.error_code,
                                    request.progress_snapshot(),
                                )
                            )
                            if can_retry:
                                submit(
                                    request.shard,
                                    request.attempt + 1,
                                    request.accepted_variants,
                                )
                            else:
                                failed = True
                    if replenish_diversity_rejections and not replenishment_started:
                        target_shard_ids = canonical_target_prefix()
                        if target_shard_ids is not None:
                            break
                if (
                    stop_status is not None
                    or target_shard_ids is not None
                    or not replenish_diversity_rejections
                ):
                    break
                preview_accepted, preview_receipts = self._merge(
                    completed, budget.result_count
                )
                if len(preview_accepted) >= budget.result_count:
                    break
                completed_by_shard = {
                    shard.shard_id: tuple(
                        request
                        for request, _outcome in completed.values()
                        if request.shard.shard_id == shard.shard_id
                    )
                    for shard in ordered_shards
                }
                latest_outcome_by_shard = {
                    shard.shard_id: max(
                        (
                            (request.attempt, outcome)
                            for request, outcome in completed.values()
                            if request.shard.shard_id == shard.shard_id
                        ),
                        key=lambda item: item[0],
                    )[1]
                    for shard in ordered_shards
                    if completed_by_shard[shard.shard_id]
                }
                rejected_indexes = {
                    item.shard_index
                    for item in preview_receipts
                    if not item.accepted
                    and item.rejection_reason
                    in {"duplicate", "symmetry", "decision_state_distance"}
                }
                retryable_shards = tuple(
                    shard
                    for shard in sorted(
                        ordered_shards,
                        key=lambda item: (
                            item.shard_index not in rejected_indexes,
                            item.shard_index,
                        ),
                    )
                    if shard.shard_id in latest_outcome_by_shard
                    and len(completed_by_shard[shard.shard_id]) < shard.seed_count
                    and latest_outcome_by_shard[shard.shard_id].variants
                    and not bool(
                        dict(
                            latest_outcome_by_shard[
                                shard.shard_id
                            ].partial_evidence
                            or {}
                        ).get("candidateBudgetExhausted")
                    )
                    and max(
                        (
                            request.attempt
                            for request in completed_by_shard[shard.shard_id]
                        ),
                        default=0,
                    )
                    < shard.maximum_attempts
                )
                if not retryable_shards:
                    break
                replenishment_width = min(len(retryable_shards), 2)
                replenishment_shards: list[ResearchShardSpecV1] = []
                used_residues: set[int] = set()
                for shard in retryable_shards:
                    residue = shard.shard_index % replenishment_width
                    if residue in used_residues:
                        continue
                    used_residues.add(residue)
                    replenishment_shards.append(shard)
                    if len(replenishment_shards) == replenishment_width:
                        break
                for shard in replenishment_shards:
                    submit(
                        shard,
                        max(
                            request.attempt
                            for request in completed_by_shard[shard.shard_id]
                        )
                        + 1,
                        preview_accepted,
                    )
        finally:
            if stop_status is not None or target_shard_ids is not None:
                cancellation_started = time.perf_counter_ns()
                token._cancel()
                cancelled_before_start = {
                    future: future.cancel() for future in active
                }
                # Polling controls how quickly the coordinator notices cancellation;
                # acknowledgement has its separately governed 250 ms performance gate.
                acknowledgement_deadline = (
                    time.perf_counter() + _CANCELLATION_ACKNOWLEDGEMENT_SECONDS
                )
                acknowledgement_latencies: dict[Future[_AttemptExecution], int] = {}

                def acknowledgements_complete() -> bool:
                    for future, request in active.items():
                        if future not in acknowledgement_latencies and (
                            cancelled_before_start[future]
                            or future.done()
                            or request.cancellation_acknowledgement.is_set()
                        ):
                            acknowledgement_latencies[future] = (
                                time.perf_counter_ns() - cancellation_started
                            )
                    return len(acknowledgement_latencies) == len(active)

                while (
                    not acknowledgements_complete()
                    and time.perf_counter() < acknowledgement_deadline
                ):
                    wait(
                        tuple(active),
                        timeout=min(
                            0.005,
                            max(0.0, acknowledgement_deadline - time.perf_counter()),
                        ),
                        return_when=FIRST_COMPLETED,
                    )
                if target_shard_ids is not None:
                    while time.perf_counter() < acknowledgement_deadline and not all(
                        cancelled_before_start[future] or future.done()
                        for future in active
                    ):
                        wait(
                            tuple(active),
                            timeout=min(
                                0.005,
                                max(
                                    0.0,
                                    acknowledgement_deadline - time.perf_counter(),
                                ),
                            ),
                            return_when=FIRST_COMPLETED,
                        )
                    # A successful campaign must not leak owned shard work into the
                    # next deterministic solve.  This join is intentionally stronger
                    # than explicit user cancellation, which only promises prompt
                    # acknowledgement and may unwind in the background.
                    executor.shutdown(wait=True, cancel_futures=True)
                for future, request in sorted(
                    active.items(),
                    key=lambda item: (item[1].shard.shard_index, item[1].attempt),
                ):
                    was_cancelled_before_start = cancelled_before_start[future]
                    worker_terminated = was_cancelled_before_start or future.done()
                    acknowledged = (
                        worker_terminated
                        or request.cancellation_acknowledgement.is_set()
                    )
                    timing = CampaignStageTimingV1(
                        "cancellation_acknowledgement",
                        acknowledgement_latencies.get(
                            future, time.perf_counter_ns() - cancellation_started
                        ),
                    )
                    stage_timings: tuple[CampaignStageTimingV1, ...] = (timing,)
                    variant_hashes: tuple[str, ...] = ()
                    partial_evidence = request.progress_snapshot()
                    if worker_terminated and not was_cancelled_before_start:
                        execution = future.result()
                        stage_timings = (
                            *(
                                execution.outcome.stage_timings
                                if execution.outcome is not None
                                else ()
                            ),
                            execution.timing,
                            timing,
                        )
                        if execution.outcome is not None:
                            variant_hashes = tuple(
                                item.semantic_hash
                                for item in execution.outcome.variants
                            )
                            if execution.outcome.partial_evidence is not None:
                                partial_evidence = {
                                    **partial_evidence,
                                    **dict(execution.outcome.partial_evidence),
                                }
                    partial_evidence = {
                        **partial_evidence,
                        "cancellation": {
                            "acknowledged": acknowledged,
                            "workerTerminated": worker_terminated,
                            "latencyNs": timing.wall_time_ns,
                            "pollLimitMilliseconds": (
                                budget.cancellation_poll_milliseconds
                            ),
                        },
                    }
                    # An interrupted candidate is operationally incomplete even
                    # when no immutable execution document returned. Never relay
                    # the private checkpoint payload through the public callback.
                    ordinal = partial_evidence.get("completedCandidates")
                    if (not variant_hashes and partial_evidence.get("currentStage") == "playingQuality"
                            and type(ordinal) is int and ordinal > 0
                            and isinstance(partial_evidence.get("candidateHash"), str)):
                        request.report_progress({
                            "currentStage": "candidate_terminal",
                            "candidateStatus": "incomplete",
                            "candidateHash": partial_evidence["candidateHash"],
                            "executionOrdinal": ordinal,
                            "completedCandidates": ordinal,
                            "candidateAttemptReason": stop_status or "target_satisfied",
                            "incompleteCompilation": {"stage": "playingQuality",
                                "exhaustionReason": stop_status or "target_satisfied"},
                        })
                    attempts.append(
                        CampaignAttemptReceiptV1(
                            request.shard.shard_index,
                            request.shard.shard_id,
                            request.attempt,
                            (
                                "cancelled"
                                if acknowledged
                                else "cancellation_unacknowledged"
                            ),
                            variant_hashes,
                            stage_timings,
                            stop_status or "target_satisfied",
                            partial_evidence,
                        )
                    )
                if target_shard_ids is None:
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        canonical_completed = (
            completed
            if target_shard_ids is None
            else {
                key: value
                for key, value in completed.items()
                if value[0].shard.shard_id in target_shard_ids
            }
        )
        if target_shard_ids is not None:
            attempts = [
                (
                    replace(
                        item,
                        status="not_required",
                        variant_hashes=(),
                        error_code="target_satisfied",
                    )
                    if item.status == "completed"
                    and item.shard_id not in target_shard_ids
                    else item
                )
                for item in attempts
            ]
        accepted, variants = self._merge(canonical_completed, budget.result_count)
        if target_shard_ids is not None:
            status = "completed"
        elif stop_status is not None:
            status = stop_status
        elif failed or len(
            {request.shard.shard_id for request, _outcome in completed.values()}
        ) != len(ordered_shards):
            status = "incomplete"
        else:
            status = (
                "completed"
                if len(accepted) == budget.result_count
                else "not_found_within_budget"
            )

        ended_wall = time.perf_counter_ns()
        ended_cpu = time.process_time_ns()
        attempts_tuple = tuple(
            sorted(attempts, key=lambda item: (item.shard_index, item.attempt))
        )
        provisional = CampaignReceiptV1(
            status,
            ordered_shards,
            budget,
            self._duplicate_policy_id,
            self._symmetry_policy_id,
            self._distance_policy_id,
            self._minimum_distance,
            attempts_tuple,
            variants,
            accepted,
            None,
            ended_wall - started_wall,
            ended_cpu - started_cpu,
            "",
        )
        semantic_core = provisional._semantic_core_document()
        provisional = replace(
            provisional,
            semantic_sha256=(
                canonical_sha256(semantic_core) if semantic_core else None
            ),
        )
        return replace(
            provisional,
            receipt_sha256=canonical_sha256(provisional._receipt_core()),
        )

    @staticmethod
    def _invoke(
        solver: CampaignShardSolverV1, request: CampaignShardRequestV1
    ) -> _AttemptExecution:
        started_wall = time.perf_counter_ns()
        started_cpu = time.thread_time_ns()
        outcome: CampaignShardOutcomeV1 | None = None
        retryable = False
        error_code: str | None = None
        try:
            outcome = solver(request)
            if not isinstance(outcome, CampaignShardOutcomeV1):
                raise TypeError("campaign shard solver returned an invalid outcome")
        except RetryableCampaignShardError as error:
            retryable = True
            error_code = error.code
            progress = request.progress_snapshot()
            progress.pop("qualityCheckpoints", None)
            request.report_progress(
                {**progress, "currentStage": "retryable_failure", "errorCode": error_code}
            )
        except Exception as error:  # noqa: BLE001 - receipt owns solver failures
            error_code = f"{type(error).__module__}.{type(error).__qualname__}"
            progress = request.progress_snapshot()
            progress.pop("qualityCheckpoints", None)
            request.report_progress(
                {
                    **progress,
                    "currentStage": (
                        "cancelled" if request.cancellation_requested() else "worker_failed"
                    ),
                    "errorCode": error_code,
                    "errorDetail": str(error)[:512],
                }
            )
        timing = CampaignStageTimingV1(
            "shard_solver",
            time.perf_counter_ns() - started_wall,
            time.thread_time_ns() - started_cpu,
        )
        return _AttemptExecution(request, outcome, retryable, error_code, timing)

    def _merge(
        self,
        completed: Mapping[str, tuple[CampaignShardRequestV1, CampaignShardOutcomeV1]],
        result_count: int,
    ) -> tuple[tuple[CampaignVariantV1, ...], tuple[CampaignVariantReceiptV1, ...]]:
        return select_campaign_variants_v1(
            tuple(
                (request.shard.shard_index, request.attempt, variant)
                for request, outcome in completed.values()
                for variant in outcome.variants
            ),
            result_count=result_count,
            duplicate_key=self._duplicate_key,
            symmetry_key=self._symmetry_key,
            distance=self._distance,
            minimum_distance=self._minimum_distance,
        )

    @staticmethod
    def _validate_shards(
        shards: tuple[ResearchShardSpecV1, ...],
    ) -> tuple[ResearchShardSpecV1, ...]:
        if not shards:
            raise ValueError("campaign requires at least one shard")
        if any(not isinstance(item, ResearchShardSpecV1) for item in shards):
            raise TypeError("campaign shards must be ResearchShardSpecV1 values")
        ordered = tuple(sorted(shards, key=lambda item: item.shard_index))
        first = ordered[0]
        if any(
            item.run_code != first.run_code
            or item.run_hash != first.run_hash
            or item.recipe_hash != first.recipe_hash
            for item in ordered
        ):
            raise ValueError("campaign shards must share one run and recipe authority")
        if len({item.shard_index for item in ordered}) != len(ordered):
            raise ValueError("campaign shard indexes must be unique")
        by_seed = sorted(ordered, key=lambda item: item.seed_start)
        if any(
            left.seed_end_exclusive > right.seed_start
            for left, right in zip(by_seed, by_seed[1:], strict=False)
        ):
            raise ValueError("campaign shard seed ranges must not overlap")
        return ordered


__all__ = [
    "CampaignAttemptReceiptV1",
    "CampaignBudgetV1",
    "CampaignCancellationTokenV1",
    "CampaignReceiptV1",
    "CampaignShardOutcomeV1",
    "CampaignShardRequestV1",
    "CampaignStageTimingV1",
    "CampaignVariantReceiptV1",
    "CampaignVariantV1",
    "CampaignVariantDistanceV1",
    "DeterministicCpSatCampaignCoordinatorV1",
    "RetryableCampaignShardError",
    "build_rules_preserving_diversity_document_v1",
    "decision_state_card_vector_v1",
    "decision_state_card_hamming_distance_v1",
    "rules_preserving_symmetry_key_v1",
    "select_campaign_variants_v1",
]
