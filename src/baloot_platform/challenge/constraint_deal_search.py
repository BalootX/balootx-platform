"""Optional deterministic CP-SAT deal construction for guided discovery.

The solver owns only structural card placement.  The Baloot engine, equal-seat
agents, observer-safe analysis, and finalist gates remain authoritative.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.projects import project_card_patterns
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.shuffle import shuffle_and_cut
from baloot_platform.core.structural_ir import StructuralDomainIRV1
from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import canonical_sha256


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_sha256(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return value


def _require_exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _range_map(value: Mapping[str, Any] | None) -> tuple[tuple[Seat, Suit, int, int], ...]:
    rows: list[tuple[Seat, Suit, int, int]] = []
    for raw_seat, raw_suits in dict(value or {}).items():
        seat = Seat.parse(raw_seat)
        if not isinstance(raw_suits, Mapping):
            raise ValueError("deal suit ranges must be grouped by seat")
        for raw_suit, raw_range in raw_suits.items():
            suit = Suit.parse(raw_suit)
            if not isinstance(raw_range, Mapping):
                raise ValueError("a deal suit range must be an object")
            unknown = set(raw_range) - {"minimum", "maximum"}
            if unknown:
                raise ValueError(f"unknown deal suit range keys: {sorted(unknown)}")
            minimum = int(raw_range.get("minimum", 0))
            maximum = int(raw_range.get("maximum", 8))
            if not 0 <= minimum <= maximum <= 8:
                raise ValueError("deal suit count range is invalid")
            rows.append((seat, suit, minimum, maximum))
    return tuple(sorted(rows, key=lambda item: (int(item[0]), int(item[1]))))


def _required_card_map(
    value: Mapping[str, Any] | None,
    *,
    unique_globally: bool = True,
) -> tuple[tuple[Seat, tuple[int, ...]], ...]:
    rows: list[tuple[Seat, tuple[int, ...]]] = []
    claimed: set[int] = set()
    for raw_seat, raw_cards in dict(value or {}).items():
        if not isinstance(raw_cards, (list, tuple)):
            raise ValueError("required cards must be grouped as a list per seat")
        cards = tuple(sorted({Card.parse(item).id for item in raw_cards}))
        if len(cards) != len(raw_cards):
            raise ValueError("required card lists cannot contain duplicates")
        if unique_globally and claimed.intersection(cards):
            raise ValueError("one required card cannot belong to multiple seats")
        claimed.update(cards)
        rows.append((Seat.parse(raw_seat), cards))
    return tuple(sorted(rows, key=lambda item: int(item[0])))


@dataclass(frozen=True, slots=True)
class ProjectPatternConstraintV1:
    seat: Seat
    contract: Contract
    kind: str
    value: int
    high_rank: int

    @classmethod
    def build(cls, value: Mapping[str, Any] | None) -> "ProjectPatternConstraintV1 | None":
        if not value:
            return None
        allowed = {"seat", "contract", "kind", "value", "highRank"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown project pattern keys: {sorted(unknown)}")
        kind = str(value.get("kind") or "").strip()
        if kind not in {"siri", "fifty", "hundred_sequence", "hundred_four_kind", "four_hundred"}:
            raise ValueError("unsupported project pattern kind")
        return cls(
            Seat.parse(value["seat"]),
            Contract.parse(value["contract"]),
            kind,
            int(value["value"]),
            int(Rank.parse(value["highRank"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat.name.lower(),
            "contract": self.contract.value,
            "kind": self.kind,
            "value": self.value,
            "highRank": Rank(self.high_rank).name.lower(),
        }


@dataclass(frozen=True, slots=True)
class DealConstraintSpecV1:
    profile_id: str
    revision: str
    enabled: bool
    backend: str
    assumed_buyer: Seat | None
    purchase_suits: tuple[Suit, ...]
    initial_required_cards: tuple[tuple[Seat, tuple[int, ...]], ...]
    final_required_cards: tuple[tuple[Seat, tuple[int, ...]], ...]
    final_forbidden_cards: tuple[tuple[Seat, tuple[int, ...]], ...]
    initial_suit_ranges: tuple[tuple[Seat, Suit, int, int], ...]
    final_suit_ranges: tuple[tuple[Seat, Suit, int, int], ...]
    project_pattern: ProjectPatternConstraintV1 | None
    maximum_solutions: int
    maximum_conflicts: int
    safety_timeout_seconds: float
    solver_workers: int
    seed_offset: int
    tags: tuple[str, ...]
    tactical_objective_weights: tuple[tuple[str, int], ...]
    objective_target_seat: Seat | None
    tactical_deal_balance: tuple[int, int, int, int] | None
    tactical_contract: Contract | None
    tactical_trump: Suit | None
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any] | None) -> "DealConstraintSpecV1":
        raw = dict(value or {})
        allowed = {
            "schemaVersion", "id", "revision", "enabled", "backend",
            "assumedBuyer", "purchaseSuits", "initialSuitCountRanges",
            "initialRequiredCards", "finalRequiredCards", "finalForbiddenCards",
            "finalSuitCountRanges", "projectPattern", "maximumSolutions",
            "maximumConflicts", "safetyTimeoutSeconds", "solverWorkers",
            "seedOffset", "tags",
            "tacticalObjectiveWeights", "objectiveTargetSeat",
            "tacticalDealBalance", "tacticalContract", "tacticalTrump",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown deal constraint profile keys: {sorted(unknown)}")
        if int(raw.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported deal constraint schema version")
        profile_id = str(raw.get("id") or "").strip()
        revision = str(raw.get("revision") or "").strip()
        if not profile_id or not revision:
            raise ValueError("deal constraint profile identity is required")
        backend = str(raw.get("backend") or "")
        if backend != "ortools_cp_sat_v1":
            raise ValueError("unsupported deal constraint backend")
        assumed_buyer = (
            Seat.parse(raw["assumedBuyer"])
            if raw.get("assumedBuyer") is not None
            else None
        )
        purchase_suits = tuple(sorted({Suit.parse(item) for item in raw.get("purchaseSuits") or ()}, key=int))
        initial_required_cards = _required_card_map(raw.get("initialRequiredCards"))
        final_required_cards = _required_card_map(raw.get("finalRequiredCards"))
        final_forbidden_cards = _required_card_map(
            raw.get("finalForbiddenCards"), unique_globally=False
        )
        required_claims = {
            (seat, card)
            for seat, cards in final_required_cards
            for card in cards
        }
        forbidden_claims = {
            (seat, card)
            for seat, cards in final_forbidden_cards
            for card in cards
        }
        if required_claims.intersection(forbidden_claims):
            raise ValueError("one card cannot be both required and forbidden for a seat")
        initial_ranges = _range_map(raw.get("initialSuitCountRanges"))
        final_ranges = _range_map(raw.get("finalSuitCountRanges"))
        project = ProjectPatternConstraintV1.build(raw.get("projectPattern"))
        if (final_required_cards or final_forbidden_cards or final_ranges or project is not None) and assumed_buyer is None:
            raise ValueError("final-hand constraints require an assumed buyer")
        maximum_solutions = int(raw.get("maximumSolutions", 0))
        maximum_conflicts = int(raw.get("maximumConflicts", 0))
        safety_timeout = float(raw.get("safetyTimeoutSeconds", 0.0))
        workers = int(raw.get("solverWorkers", 0))
        if not 1 <= maximum_solutions <= 100:
            raise ValueError("deal constraint solution count must be in [1, 100]")
        if not 1 <= maximum_conflicts <= 10_000_000:
            raise ValueError("deal constraint conflict budget is invalid")
        if not 0.1 <= safety_timeout <= 900.0:
            raise ValueError("deal constraint safety timeout is invalid")
        if workers != 1:
            raise ValueError("deterministic CP-SAT deal search requires one worker")
        tags = tuple(sorted({str(item).strip() for item in raw.get("tags") or () if str(item).strip()}))
        objective_weights = tuple(sorted(
            (
                str(kind), int(weight)
            )
            for kind, weight in dict(
                raw.get("tacticalObjectiveWeights") or {}
            ).items()
            if int(weight) > 0
        ))
        if any(not kind.strip() or not 1 <= weight <= 10_000 for kind, weight in objective_weights):
            raise ValueError("tactical objective weights must be named integers in [1, 10000]")
        objective_target = (
            Seat.parse(raw["objectiveTargetSeat"])
            if raw.get("objectiveTargetSeat") is not None
            else assumed_buyer
        )
        if objective_weights and objective_target is None:
            raise ValueError("tactical objectives require an objective target seat")
        tactical_contract = (
            Contract.parse(raw["tacticalContract"])
            if raw.get("tacticalContract") is not None
            else None
        )
        tactical_trump = (
            Suit.parse(raw["tacticalTrump"])
            if raw.get("tacticalTrump") is not None
            else None
        )
        balance_raw = raw.get("tacticalDealBalance")
        tactical_balance: tuple[int, int, int, int] | None = None
        if balance_raw is not None:
            balance = dict(balance_raw)
            allowed_balance = {
                "maximumTeamHighCardDifference",
                "minimumHighCardsPerSeat",
                "maximumHighCardsPerSeat",
                "minimumTargetLowCards",
            }
            if set(balance) != allowed_balance:
                raise ValueError("tactical deal balance is incomplete or contains unknown keys")
            tactical_balance = (
                int(balance["maximumTeamHighCardDifference"]),
                int(balance["minimumHighCardsPerSeat"]),
                int(balance["maximumHighCardsPerSeat"]),
                int(balance["minimumTargetLowCards"]),
            )
            difference, minimum_high, maximum_high, minimum_low = tactical_balance
            if not 0 <= difference <= 12:
                raise ValueError("tactical team high-card difference is invalid")
            if not 0 <= minimum_high <= 3 or not 3 <= maximum_high <= 8:
                raise ValueError("tactical per-seat high-card bounds are invalid")
            if minimum_high > maximum_high or not 0 <= minimum_low <= 3:
                raise ValueError("tactical deal balance bounds are inconsistent")
            if objective_target is None:
                raise ValueError("tactical deal balance requires an objective target seat")
        contract_sensitive = bool(objective_weights or tactical_balance is not None)
        if contract_sensitive and tactical_contract is None:
            raise ValueError(
                "contract-sensitive tactical construction requires an exact contract"
            )
        if tactical_contract is None and tactical_trump is not None:
            raise ValueError("a tactical trump requires an exact tactical contract")
        if tactical_contract is not None:
            if tactical_contract.effective == Contract.HOKUM and tactical_trump is None:
                raise ValueError("Hokum tactical construction requires an exact trump suit")
            if tactical_contract.effective != Contract.HOKUM and tactical_trump is not None:
                raise ValueError("Sun and Ashkal tactical construction cannot declare trump")
        core = {
            "schemaVersion": 1,
            "id": profile_id,
            "revision": revision,
            "enabled": bool(raw.get("enabled", False)),
            "backend": backend,
            "assumedBuyer": assumed_buyer.name.lower() if assumed_buyer is not None else None,
            "purchaseSuits": [item.name.lower() for item in purchase_suits],
            "initialRequiredCards": {
                seat.name.lower(): [Card(card_id).code for card_id in cards]
                for seat, cards in initial_required_cards
            },
            "finalRequiredCards": {
                seat.name.lower(): [Card(card_id).code for card_id in cards]
                for seat, cards in final_required_cards
            },
            "finalForbiddenCards": {
                seat.name.lower(): [Card(card_id).code for card_id in cards]
                for seat, cards in final_forbidden_cards
            },
            "initialSuitCountRanges": cls._ranges_dict(initial_ranges),
            "finalSuitCountRanges": cls._ranges_dict(final_ranges),
            "projectPattern": project.to_dict() if project is not None else None,
            "maximumSolutions": maximum_solutions,
            "maximumConflicts": maximum_conflicts,
            "safetyTimeoutSeconds": safety_timeout,
            "solverWorkers": workers,
            "seedOffset": int(raw.get("seedOffset", 0)),
            "tags": list(tags),
            "tacticalObjectiveWeights": dict(objective_weights),
            "objectiveTargetSeat": (
                objective_target.name.lower()
                if objective_target is not None else None
            ),
            "tacticalDealBalance": (
                {
                    "maximumTeamHighCardDifference": tactical_balance[0],
                    "minimumHighCardsPerSeat": tactical_balance[1],
                    "maximumHighCardsPerSeat": tactical_balance[2],
                    "minimumTargetLowCards": tactical_balance[3],
                }
                if tactical_balance is not None else None
            ),
            "tacticalContract": (
                tactical_contract.value if tactical_contract is not None else None
            ),
            "tacticalTrump": (
                tactical_trump.name.lower() if tactical_trump is not None else None
            ),
        }
        return cls(
            profile_id, revision, core["enabled"], backend, assumed_buyer,
            purchase_suits, initial_required_cards, final_required_cards,
            final_forbidden_cards, initial_ranges, final_ranges, project,
            maximum_solutions, maximum_conflicts, safety_timeout, workers,
            core["seedOffset"], tags, objective_weights, objective_target,
            tactical_balance, tactical_contract, tactical_trump,
            canonical_sha256(core),
        )

    @staticmethod
    def _ranges_dict(
        rows: tuple[tuple[Seat, Suit, int, int], ...],
    ) -> dict[str, dict[str, dict[str, int]]]:
        result: dict[str, dict[str, dict[str, int]]] = {}
        for seat, suit, minimum, maximum in rows:
            result.setdefault(seat.name.lower(), {})[suit.name.lower()] = {
                "minimum": minimum,
                "maximum": maximum,
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "enabled": self.enabled,
            "backend": self.backend,
            "assumedBuyer": self.assumed_buyer.name.lower() if self.assumed_buyer is not None else None,
            "purchaseSuits": [item.name.lower() for item in self.purchase_suits],
            "initialRequiredCards": {
                seat.name.lower(): [Card(card_id).code for card_id in cards]
                for seat, cards in self.initial_required_cards
            },
            "finalRequiredCards": {
                seat.name.lower(): [Card(card_id).code for card_id in cards]
                for seat, cards in self.final_required_cards
            },
            "finalForbiddenCards": {
                seat.name.lower(): [Card(card_id).code for card_id in cards]
                for seat, cards in self.final_forbidden_cards
            },
            "initialSuitCountRanges": self._ranges_dict(self.initial_suit_ranges),
            "finalSuitCountRanges": self._ranges_dict(self.final_suit_ranges),
            "projectPattern": self.project_pattern.to_dict() if self.project_pattern is not None else None,
            "maximumSolutions": self.maximum_solutions,
            "maximumConflicts": self.maximum_conflicts,
            "safetyTimeoutSeconds": self.safety_timeout_seconds,
            "solverWorkers": self.solver_workers,
            "seedOffset": self.seed_offset,
            "tags": list(self.tags),
            "tacticalObjectiveWeights": dict(self.tactical_objective_weights),
            "objectiveTargetSeat": (
                self.objective_target_seat.name.lower()
                if self.objective_target_seat is not None else None
            ),
            "tacticalDealBalance": (
                {
                    "maximumTeamHighCardDifference": self.tactical_deal_balance[0],
                    "minimumHighCardsPerSeat": self.tactical_deal_balance[1],
                    "maximumHighCardsPerSeat": self.tactical_deal_balance[2],
                    "minimumTargetLowCards": self.tactical_deal_balance[3],
                }
                if self.tactical_deal_balance is not None else None
            ),
            "tacticalContract": (
                self.tactical_contract.value
                if self.tactical_contract is not None else None
            ),
            "tacticalTrump": (
                self.tactical_trump.name.lower()
                if self.tactical_trump is not None else None
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ConstraintDealCandidateV1:
    ordinal: int
    deck_order: tuple[int, ...]
    deck_commitment: str
    ordering_score: int
    structural_objective_score: int
    profile_hash: str
    model_hash: str
    backend_version: str
    solver_status: str
    no_good_cut_type: str
    candidate_hash: str
    search_identity_hash: str = ""
    preceding_no_good_cut_hashes: tuple[str, ...] = ()

    def verify(self) -> "ConstraintDealCandidateV1":
        """Reject a forged or non-canonical generated candidate."""

        ordinal = _require_exact_int(self.ordinal, "candidate ordinal")
        _require_exact_int(self.ordering_score, "candidate orderingScore")
        _require_exact_int(
            self.structural_objective_score,
            "candidate structuralObjectiveScore",
        )
        if ordinal < 0 or type(self.deck_order) is not tuple or any(
            type(card) is not int for card in self.deck_order
        ) or sorted(self.deck_order) != list(range(32)):
            raise ValueError("candidate deck must be a canonical 32-card permutation")
        if type(self.preceding_no_good_cut_hashes) is not tuple or len(
            self.preceding_no_good_cut_hashes
        ) != ordinal:
            raise ValueError("candidate lost its ordered no-good prefix")
        for digest, field in (
            (self.deck_commitment, "candidate deckCommitment"),
            (self.profile_hash, "candidate profileHash"),
            (self.model_hash, "candidate modelHash"),
            (self.search_identity_hash, "candidate searchIdentityHash"),
            (self.candidate_hash, "candidateHash"),
            *(
                (digest, "candidate noGoodCutHash")
                for digest in self.preceding_no_good_cut_hashes
            ),
        ):
            _require_sha256(digest, field)
        if self.no_good_cut_type not in {
                "complete_deck_assignment_v1",
                "final_hand_allocation_v1",
        }:
            raise ValueError("candidate no-good cut authority is unsupported")
        if (
            type(self.backend_version) is not str
            or not self.backend_version
            or self.solver_status != "optimal"
            or hashlib.sha256(bytes(self.deck_order)).hexdigest()
            != self.deck_commitment
        ):
            raise ValueError("candidate integrity is invalid")
        if canonical_sha256(self.integrity_core()) != self.candidate_hash:
            raise ValueError("candidate integrity hash is invalid")
        return self

    def integrity_core(self) -> dict[str, Any]:
        """Return the complete, canonical identity bound by ``candidate_hash``."""
        return {
            "schemaVersion": 1,
            "pluginVersion": OrToolsCpSatDealGeneratorV1.descriptor.version,
            "ordinal": self.ordinal,
            "deckOrder": list(self.deck_order),
            "deckCommitment": self.deck_commitment,
            "orderingScore": self.ordering_score,
            "structuralObjectiveScore": self.structural_objective_score,
            "profileHash": self.profile_hash,
            "modelHash": self.model_hash,
            "searchIdentityHash": self.search_identity_hash,
            "backendVersion": self.backend_version,
            "solverStatus": self.solver_status,
            "noGoodCutType": self.no_good_cut_type,
            "precedingNoGoodCutHashes": list(
                self.preceding_no_good_cut_hashes
            ),
        }

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "pluginId": "ortools_cp_sat_deal_generator_v1",
            "pluginVersion": OrToolsCpSatDealGeneratorV1.descriptor.version,
            "profileHash": self.profile_hash,
            "modelHash": self.model_hash,
            "backendVersion": self.backend_version,
            "solverStatus": self.solver_status,
            "solutionOrdinal": self.ordinal,
            "searchIdentityHash": self.search_identity_hash,
            "precedingNoGoodCutCount": len(
                self.preceding_no_good_cut_hashes
            ),
            "precedingNoGoodCutHashes": list(
                self.preceding_no_good_cut_hashes
            ),
            "noGoodCutType": self.no_good_cut_type,
            "orderingScore": self.ordering_score,
            "structuralObjectiveScore": self.structural_objective_score,
            "deckCommitment": self.deck_commitment,
            "candidateHash": self.candidate_hash,
        }


@dataclass(frozen=True, slots=True)
class ConstraintDealContinuationReceiptV1:
    """Immutable proof of a real solve after an ordered rejection prefix.

    ``maximumSolutions`` limits candidates emitted for semantic evaluation.
    This receipt exposes only a commitment when the same CP-SAT model has a
    further solution, so the continuation witness cannot become an unevaluated
    game or leak a replacement deck into the qualification lane.
    """

    profile_hash: str
    model_hash: str
    backend_version: str
    solver_status: str
    solution_ordinal: int
    search_identity_hash: str
    preceding_no_good_cut_hashes: tuple[str, ...]
    no_good_cut_type: str
    exhausted: bool
    witness_candidate_hash: str | None
    witness_deck_commitment: str | None
    witness_ordering_score: int | None
    structural_objective_score: int
    receipt_hash: str

    def verify(self) -> "ConstraintDealContinuationReceiptV1":
        """Reject a forged or non-canonical continuation solve receipt."""

        ordinal = _require_exact_int(
            self.solution_ordinal,
            "continuation solutionOrdinal",
        )
        _require_exact_int(
            self.structural_objective_score,
            "continuation structuralObjectiveScore",
        )
        if type(self.exhausted) is not bool:
            raise ValueError("continuation exhausted must be a boolean")
        if type(self.preceding_no_good_cut_hashes) is not tuple or len(
            self.preceding_no_good_cut_hashes
        ) != ordinal:
            raise ValueError("continuation lost its ordered no-good prefix")
        for digest, field in (
            (self.profile_hash, "continuation profileHash"),
            (self.model_hash, "continuation modelHash"),
            (self.search_identity_hash, "continuation searchIdentityHash"),
            (self.receipt_hash, "continuation sha256"),
            *(
                (digest, "continuation noGoodCutHash")
                for digest in self.preceding_no_good_cut_hashes
            ),
        ):
            _require_sha256(digest, field)
        witness = (
            self.witness_candidate_hash,
            self.witness_deck_commitment,
            self.witness_ordering_score,
        )
        if (self.exhausted and any(value is not None for value in witness)) or (
            not self.exhausted and not all(value is not None for value in witness)
        ):
            raise ValueError("continuation witness evidence is inconsistent")
        if not self.exhausted:
            _require_sha256(self.witness_candidate_hash, "witnessCandidateHash")
            _require_sha256(self.witness_deck_commitment, "witnessDeckCommitment")
            _require_exact_int(self.witness_ordering_score, "witnessOrderingScore")
        if (
            ordinal < 0
            or type(self.backend_version) is not str
            or not self.backend_version
            or self.solver_status not in {"optimal", "infeasible"}
            or self.exhausted != (self.solver_status == "infeasible")
            or self.no_good_cut_type not in {
                "complete_deck_assignment_v1",
                "final_hand_allocation_v1",
            }
            or canonical_sha256(self.integrity_core()) != self.receipt_hash
        ):
            raise ValueError("continuation receipt integrity is invalid")
        return self

    def integrity_core(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "cp_sat_candidate_space_probe_receipt_v1",
            "pluginId": OrToolsCpSatDealGeneratorV1.descriptor.id,
            "pluginVersion": OrToolsCpSatDealGeneratorV1.descriptor.version,
            "profileHash": self.profile_hash,
            "modelHash": self.model_hash,
            "backendVersion": self.backend_version,
            "solverStatus": self.solver_status,
            "solutionOrdinal": self.solution_ordinal,
            "searchIdentityHash": self.search_identity_hash,
            "precedingNoGoodCutCount": len(
                self.preceding_no_good_cut_hashes
            ),
            "precedingNoGoodCutHashes": list(
                self.preceding_no_good_cut_hashes
            ),
            "finalNoGoodCutHash": (
                self.preceding_no_good_cut_hashes[-1]
                if self.preceding_no_good_cut_hashes else None
            ),
            "noGoodCutType": self.no_good_cut_type,
            "solverInvokedAfterFinalCut": bool(
                self.preceding_no_good_cut_hashes
            ),
            "candidateEmitted": False,
            "candidateEvaluated": False,
            "candidatePublished": False,
            "exhausted": self.exhausted,
            "witnessCandidateHash": self.witness_candidate_hash,
            "witnessDeckCommitment": self.witness_deck_commitment,
            "witnessOrderingScore": self.witness_ordering_score,
            "structuralObjectiveScore": self.structural_objective_score,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.integrity_core(), "sha256": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class _ConstraintDealGenerationResultV1:
    candidates: tuple[ConstraintDealCandidateV1, ...]
    continuation_receipt: ConstraintDealContinuationReceiptV1 | None = None


class OrToolsCpSatDealGeneratorV1:
    # Each governed strength table is a permutation of 0..7.  The bottom and
    # top three values are structural construction bands only; tactical truth
    # is still proved later by the observer-safe causal audit.
    _LOW_STRENGTH_MAX = 2
    _HIGH_STRENGTH_MIN = 5

    descriptor = PluginDescriptor(
        id="ortools_cp_sat_deal_generator_v1",
        kind="generator",
        version="1.5.0",
        description="Deterministic structural deal candidate construction; engine validation required",
        capabilities=(
            "complete_deck_permutation",
            "purchase_suit_constraints",
            "seat_suit_shape_constraints",
            "rule_derived_project_pattern",
            "weighted_tactical_structure_objective",
            "contract_aware_tactical_strength",
            "complete_assignment_no_good_cuts",
            "initial_and_post_rejection_continuation_receipts",
            "domain_structural_semantics",
            "engine_validation_required",
        ),
        deterministic=True,
        resource_class="cpu",
        timeout_seconds=900,
        maximum_memory_mb=2048,
    )

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("ortools.sat.python.cp_model") is not None

    @staticmethod
    def compile_structural_domain(**kwargs):
        """Compile governed deck/seating/dealing semantics through CP-SAT."""
        from baloot_platform.research.structural_compiler import (
            OrToolsCpSatStructuralCompilerV1,
        )

        return OrToolsCpSatStructuralCompilerV1().compile(**kwargs)

    def generate(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: RuleBook,
        deadline_monotonic: float | None = None,
    ) -> tuple[ConstraintDealCandidateV1, ...]:
        """Return the existing deterministic batch of structural candidates."""
        return self._generate(
            spec,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            rejected_candidates=(),
            maximum_new_solutions=spec.maximum_solutions,
            deadline_monotonic=deadline_monotonic,
            continuation_probe=False,
        ).candidates

    def generate_next(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: RuleBook,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...] = (),
        deadline_monotonic: float | None = None,
    ) -> ConstraintDealCandidateV1 | None:
        """Return the next candidate after an exact, ordered rejection prefix.

        The rejection tuple is stateless continuation evidence.  Every member
        is integrity checked against this exact governed search before its own
        declared no-good cut is applied.  The returned candidate therefore
        cannot silently cross a profile, model, seed, dealer, rules, backend,
        or cut representation boundary.
        """
        if not isinstance(rejected_candidates, tuple):
            raise TypeError("rejected candidates must be an ordered tuple")
        result = self._generate(
            spec,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            rejected_candidates=rejected_candidates,
            maximum_new_solutions=1,
            deadline_monotonic=deadline_monotonic,
            continuation_probe=False,
        )
        return result.candidates[0] if result.candidates else None

    def probe_continuation(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: RuleBook,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        deadline_monotonic: float | None = None,
    ) -> ConstraintDealContinuationReceiptV1:
        """Solve once after an exact rejection prefix, including the empty prefix.

        The probe reuses the exact model builder, seed, worker count, deadline,
        structural optimum, and ordered rejection prefix.  A feasible witness
        is commitment-only and is never returned as a candidate.
        """

        if not isinstance(rejected_candidates, tuple):
            raise TypeError("rejected candidates must be an ordered tuple")
        if not 0 <= len(rejected_candidates) <= spec.maximum_solutions:
            raise ValueError(
                "continuation probe prefix must remain within maximumSolutions"
            )
        result = self._generate(
            spec,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            rejected_candidates=rejected_candidates,
            maximum_new_solutions=1,
            deadline_monotonic=deadline_monotonic,
            continuation_probe=True,
        )
        if result.candidates or result.continuation_receipt is None:
            raise AssertionError(
                "continuation probe emitted a candidate or lost its receipt"
            )
        return result.continuation_receipt

    def _generate(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: RuleBook,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        maximum_new_solutions: int,
        deadline_monotonic: float | None,
        continuation_probe: bool,
    ) -> _ConstraintDealGenerationResultV1:
        if not spec.enabled:
            if rejected_candidates:
                raise ValueError(
                    "rejected candidates cannot continue a disabled deal search"
                )
            return _ConstraintDealGenerationResultV1(())
        if len(rejected_candidates) > spec.maximum_solutions:
            raise ValueError(
                "rejected candidate count exceeds maximumSolutions"
            )
        self._require_deadline(deadline_monotonic)
        if not self.available():
            raise RuntimeError("OR-Tools is required by the active deal constraint profile")
        import ortools
        from ortools.sat.python import cp_model

        layout = DealLayoutV1.build(
            rules.structural_domain,
            dealer=dealer,
            buyer=spec.assumed_buyer,
        )
        if self._seat_allocation_compatible(spec):
            return self._generate_seat_allocations(
                spec,
                root_seed=root_seed,
                layout=layout,
                rules=rules,
                cp_model=cp_model,
                backend_version=str(ortools.__version__),
                rejected_candidates=rejected_candidates,
                maximum_new_solutions=maximum_new_solutions,
                deadline_monotonic=deadline_monotonic,
                continuation_probe=continuation_probe,
            )
        model = cp_model.CpModel()
        variables = {
            (card, position): model.new_bool_var(f"c{card}_p{position}")
            for card in range(32)
            for position in range(32)
        }
        for card in range(32):
            model.add_exactly_one(variables[card, position] for position in range(32))
        for position in range(32):
            model.add_exactly_one(variables[card, position] for card in range(32))
        if spec.purchase_suits:
            allowed_cards = [card for card in range(32) if Suit(card // 8) in spec.purchase_suits]
            model.add(sum(variables[card, layout.purchase_position] for card in allowed_cards) == 1)
        for seat, cards in spec.initial_required_cards:
            positions = layout.positions_for(seat, final=False)
            for card in cards:
                model.add(sum(variables[card, position] for position in positions) == 1)
        for seat, cards in spec.final_required_cards:
            positions = layout.positions_for(seat, final=True)
            for card in cards:
                model.add(sum(variables[card, position] for position in positions) == 1)
        for seat, cards in spec.final_forbidden_cards:
            positions = layout.positions_for(seat, final=True)
            for card in cards:
                model.add(sum(variables[card, position] for position in positions) == 0)
        self._add_suit_ranges(model, variables, layout, spec.initial_suit_ranges, final=False)
        self._add_suit_ranges(model, variables, layout, spec.final_suit_ranges, final=True)
        if spec.project_pattern is not None:
            self._add_project_pattern(model, variables, layout, spec.project_pattern, rules)
        self._add_position_symmetry_breaking(model, variables, layout)

        tactical_coefficients = self._tactical_coefficients(spec, rules=rules)
        tactical_membership = {}
        membership_keys = set(tactical_coefficients)
        if spec.tactical_deal_balance is not None:
            high_cards, low_cards = self._tactical_card_bands(spec, rules=rules)
            membership_keys.update(
                (card, seat)
                for card in (*high_cards, *low_cards)
                for seat in Seat
            )
        for card, seat in sorted(
            membership_keys, key=lambda item: (item[0], int(item[1]))
        ):
            membership = model.new_bool_var(f"c{card}_s{int(seat)}")
            seat_positions = layout.positions_for(
                seat, final=spec.assumed_buyer is not None
            )
            model.add(
                membership
                == sum(variables[card, position] for position in seat_positions)
            )
            tactical_membership[card, seat] = membership
        self._add_tactical_balance(
            model,
            tactical_membership,
            spec,
            rules=rules,
        )

        costs = {
            key: int(canonical_sha256({
                "kind": "cp_sat_deal_cost_v1",
                "rootSeed": int(root_seed),
                "profileHash": spec.sha256,
                "card": key[0],
                "position": key[1],
            })[:8], 16) % 1_000_003
            for key in variables
        }
        ordered_keys = sorted(costs, key=lambda key: (costs[key], key[0], key[1]))
        ordered_variables = [variables[key] for key in ordered_keys]
        if tactical_coefficients:
            objective_variables = [
                tactical_membership[key]
                for key in sorted(
                    tactical_coefficients,
                    key=lambda key: (
                        -tactical_coefficients[key],
                        costs[key[0], layout.positions_for(
                            key[1], final=spec.assumed_buyer is not None
                        )[0]],
                        key[0],
                        int(key[1]),
                    ),
                )
            ]
            model.add_decision_strategy(
                objective_variables,
                cp_model.CHOOSE_FIRST,
                cp_model.SELECT_MAX_VALUE,
            )
        model.add_decision_strategy(
            ordered_variables,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )
        tactical_expression = sum(
            coefficient * tactical_membership[key]
            for key, coefficient in tactical_coefficients.items()
        )
        if tactical_coefficients:
            model.maximize(tactical_expression)
        fixed_hint_deck = None
        if (
            spec.maximum_solutions == 1
            and not spec.purchase_suits
            and not spec.initial_required_cards
            and not spec.final_required_cards
            and not spec.final_forbidden_cards
            and not spec.initial_suit_ranges
            and not spec.final_suit_ranges
            and spec.project_pattern is None
            and not tactical_coefficients
            and spec.tactical_deal_balance is None
        ):
            fixed_hint_deck = self._seeded_unconstrained_deck(
                spec=spec,
                root_seed=root_seed,
                ordinal=len(rejected_candidates),
                dealer=dealer,
                layout=layout,
                rules=rules,
            )
            for position, selected_card in enumerate(fixed_hint_deck):
                for card in range(32):
                    model.add_hint(
                        variables[card, position],
                        int(card == selected_card),
                    )
        model_hash = canonical_sha256({
            "schemaVersion": 1,
            "kind": "cp_sat_deal_model_v1",
            "pluginVersion": self.descriptor.version,
            "profileHash": spec.sha256,
            "rulesHash": rules.sha256,
            "playDomainIrHash": rules.play_domain.sha256,
            "tacticalContract": (
                spec.tactical_contract.value
                if spec.tactical_contract is not None else None
            ),
            "tacticalTrump": (
                spec.tactical_trump.name.lower()
                if spec.tactical_trump is not None else None
            ),
            "tacticalCoefficients": [
                [card, seat.name.lower(), coefficient]
                for (card, seat), coefficient in sorted(
                    tactical_coefficients.items(),
                    key=lambda item: (item[0][0], int(item[0][1])),
                )
            ],
            "dealer": dealer.name.lower(),
            "rootSeed": int(root_seed),
            "costs": [costs[key] for key in sorted(costs)],
            **(
                {
                    "selectionPolicy": "seeded_governed_shuffle_fixed_hint_v1",
                }
                if fixed_hint_deck is not None
                else {}
            ),
        })
        backend_version = str(ortools.__version__)
        search_identity_hash = self._search_identity_hash(
            spec=spec,
            model_hash=model_hash,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            backend_version=backend_version,
            no_good_cut_type="complete_deck_assignment_v1",
        )
        frozen_structural_score: int | None = None
        if rejected_candidates and tactical_coefficients:
            structural_solver = self._full_solver(
                cp_model,
                spec,
                root_seed,
                0,
                deadline_monotonic=deadline_monotonic,
            )
            status = structural_solver.solve(model)
            self._require_deadline(deadline_monotonic)
            self._require_optimal(cp_model, status)
            frozen_structural_score = int(
                structural_solver.value(tactical_expression)
            )
            model.add(tactical_expression == frozen_structural_score)
            model.clear_objective()

        preceding_cut_hashes: list[str] = []
        for ordinal, rejected in enumerate(rejected_candidates):
            if not isinstance(rejected, ConstraintDealCandidateV1):
                raise TypeError(
                    "rejected candidates must be ConstraintDealCandidateV1 values"
                )
            if (
                len(rejected.deck_order) != 32
                or sorted(rejected.deck_order) != list(range(32))
            ):
                raise ValueError(
                    "rejected candidate deck must be a complete permutation"
                )
            expected_ordering_score = sum(
                costs[card, position]
                for position, card in enumerate(rejected.deck_order)
            )
            relevant_hands = {
                seat: {
                    rejected.deck_order[position]
                    for position in layout.positions_for(
                        seat,
                        final=spec.assumed_buyer is not None,
                    )
                }
                for seat in Seat
            }
            actual_structural_score = sum(
                coefficient
                for (card, seat), coefficient in tactical_coefficients.items()
                if card in relevant_hands[seat]
            )
            if actual_structural_score != int(frozen_structural_score or 0):
                raise ValueError(
                    "rejected candidate is not from the proved structural optimum"
                )
            cut_hash = self._validate_rejected_candidate(
                rejected,
                ordinal=ordinal,
                expected_profile_hash=spec.sha256,
                expected_model_hash=model_hash,
                expected_search_identity_hash=search_identity_hash,
                expected_backend_version=backend_version,
                expected_cut_type="complete_deck_assignment_v1",
                expected_preceding_cut_hashes=tuple(preceding_cut_hashes),
                expected_ordering_score=expected_ordering_score,
                expected_structural_score=(
                    actual_structural_score
                ),
                layout=layout,
            )
            model.add(
                sum(
                    variables[card, position]
                    for position, card in enumerate(rejected.deck_order)
                )
                <= 31
            )
            preceding_cut_hashes.append(cut_hash)

        results: list[ConstraintDealCandidateV1] = []
        continuation_receipt: ConstraintDealContinuationReceiptV1 | None = None
        first_ordinal = len(rejected_candidates)
        stop_ordinal = min(
            spec.maximum_solutions + int(continuation_probe),
            first_ordinal + maximum_new_solutions,
        )
        for ordinal in range(first_ordinal, stop_ordinal):
            self._require_deadline(deadline_monotonic)
            solver = self._full_solver(
                cp_model,
                spec,
                root_seed,
                ordinal,
                deadline_monotonic=deadline_monotonic,
                fix_hints=fixed_hint_deck is not None,
            )
            status = solver.solve(model)
            self._require_deadline(deadline_monotonic)
            if status == cp_model.INFEASIBLE:
                if continuation_probe:
                    continuation_receipt = self._build_continuation_receipt(
                        profile_hash=spec.sha256,
                        model_hash=model_hash,
                        backend_version=backend_version,
                        solver_status="infeasible",
                        solution_ordinal=ordinal,
                        search_identity_hash=search_identity_hash,
                        preceding_no_good_cut_hashes=tuple(
                            preceding_cut_hashes
                        ),
                        no_good_cut_type="complete_deck_assignment_v1",
                        exhausted=True,
                        witness=None,
                        structural_objective_score=int(
                            frozen_structural_score or 0
                        ),
                    )
                break
            if status != cp_model.OPTIMAL:
                raise RuntimeError("CP-SAT deal construction did not prove an optimal deterministic result")
            by_position = [-1] * 32
            chosen_literals = []
            for card in range(32):
                positions = [
                    position
                    for position in range(32)
                    if solver.boolean_value(variables[card, position])
                ]
                if len(positions) != 1:
                    raise AssertionError("CP-SAT returned an invalid card assignment")
                position = positions[0]
                by_position[position] = card
                chosen_literals.append(variables[card, position])
            deck_order = tuple(by_position)
            self._require_deadline(deadline_monotonic)
            if sorted(deck_order) != list(range(32)):
                raise AssertionError("CP-SAT returned an invalid deck permutation")
            commitment = hashlib.sha256(bytes(deck_order)).hexdigest()
            ordering_score = sum(costs[card, position] for position, card in enumerate(deck_order))
            structural_score = (
                int(solver.value(tactical_expression))
                if tactical_coefficients else 0
            )
            if tactical_coefficients and frozen_structural_score is None:
                frozen_structural_score = structural_score
                model.add(tactical_expression == structural_score)
                model.clear_objective()
            candidate = self._build_candidate(
                ordinal=ordinal,
                deck_order=deck_order,
                deck_commitment=commitment,
                ordering_score=ordering_score,
                structural_objective_score=structural_score,
                profile_hash=spec.sha256,
                model_hash=model_hash,
                search_identity_hash=search_identity_hash,
                backend_version=backend_version,
                no_good_cut_type="complete_deck_assignment_v1",
                preceding_no_good_cut_hashes=tuple(preceding_cut_hashes),
            )
            if continuation_probe:
                if any(
                    candidate.candidate_hash == rejected.candidate_hash
                    or candidate.deck_commitment
                    == rejected.deck_commitment
                    for rejected in rejected_candidates
                ):
                    raise RuntimeError(
                        "continuation witness repeated a rejected candidate"
                    )
                continuation_receipt = self._build_continuation_receipt(
                    profile_hash=spec.sha256,
                    model_hash=model_hash,
                    backend_version=backend_version,
                    solver_status="optimal",
                    solution_ordinal=ordinal,
                    search_identity_hash=search_identity_hash,
                    preceding_no_good_cut_hashes=tuple(
                        preceding_cut_hashes
                    ),
                    no_good_cut_type="complete_deck_assignment_v1",
                    exhausted=False,
                    witness=candidate,
                    structural_objective_score=structural_score,
                )
                break
            results.append(candidate)
            preceding_cut_hashes.append(
                self._no_good_cut_hash(candidate, layout=layout)
            )
            model.add(sum(chosen_literals) <= 31)
        return _ConstraintDealGenerationResultV1(
            tuple(results), continuation_receipt
        )

    @staticmethod
    def _seat_allocation_compatible(spec: DealConstraintSpecV1) -> bool:
        """Use the compact model when only final hands govern feasibility."""
        return (
            spec.assumed_buyer is not None
            and bool(
                spec.tactical_objective_weights
                or spec.tactical_deal_balance is not None
                or spec.project_pattern
            )
            and not spec.purchase_suits
            and not spec.initial_required_cards
            and not spec.initial_suit_ranges
        )

    def _generate_seat_allocations(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        layout: DealLayoutV1,
        rules: RuleBook,
        cp_model,
        backend_version: str,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        maximum_new_solutions: int,
        deadline_monotonic: float | None,
        continuation_probe: bool,
    ) -> _ConstraintDealGenerationResultV1:
        """Optimize card-to-seat allocation before deterministic dealing.

        This removes the 8! positional symmetry inside each final hand.  It is
        used only when the recipe has no initial-hand or purchase-card
        constraint; the emitted deck is still replayed by the Baloot engine.
        """
        model = cp_model.CpModel()
        membership = {
            (card, seat): model.new_bool_var(f"c{card}_s{int(seat)}")
            for card in range(32)
            for seat in Seat
        }
        for card in range(32):
            model.add_exactly_one(membership[card, seat] for seat in Seat)
        for seat in Seat:
            model.add(sum(membership[card, seat] for card in range(32)) == 8)
        for seat, cards in spec.final_required_cards:
            for card in cards:
                model.add(membership[card, seat] == 1)
        for seat, cards in spec.final_forbidden_cards:
            for card in cards:
                model.add(membership[card, seat] == 0)
        for seat, suit, minimum, maximum in spec.final_suit_ranges:
            cards = range(int(suit) * 8, int(suit) * 8 + 8)
            count = sum(membership[card, seat] for card in cards)
            model.add(count >= minimum)
            model.add(count <= maximum)
        self._add_tactical_balance(model, membership, spec, rules=rules)
        if spec.project_pattern is not None:
            patterns = project_card_patterns(
                kind=spec.project_pattern.kind,
                value=spec.project_pattern.value,
                high_rank=spec.project_pattern.high_rank,
                contract=spec.project_pattern.contract,
                rules=rules,
            )
            if not patterns:
                raise ValueError("the active rules expose no matching project pattern")
            selected = []
            for index, mask in enumerate(patterns):
                literal = model.new_bool_var(f"project_pattern_{index}")
                selected.append(literal)
                for card in range(32):
                    if mask & (1 << card):
                        model.add(membership[card, spec.project_pattern.seat] >= literal)
            model.add_exactly_one(selected)

        coefficients = self._tactical_coefficients(spec, rules=rules)
        tactical_expression = sum(
            coefficient * membership[key]
            for key, coefficient in coefficients.items()
        )
        tie_costs = {
            key: int(canonical_sha256({
                "kind": "cp_sat_seat_tie_v1",
                "rootSeed": int(root_seed),
                "profileHash": spec.sha256,
                "card": key[0],
                "seat": key[1].name.lower(),
            })[:8], 16) % 1_000_003
            for key in membership
        }
        tie_expression = sum(tie_costs[key] * membership[key] for key in membership)
        if coefficients:
            model.maximize(tactical_expression)
        else:
            model.minimize(tie_expression)
        model_hash = canonical_sha256({
            "schemaVersion": 1,
            "kind": "cp_sat_final_hand_allocation_v1",
            "pluginVersion": self.descriptor.version,
            "profileHash": spec.sha256,
            "rulesHash": rules.sha256,
            "playDomainIrHash": rules.play_domain.sha256,
            "tacticalContract": (
                spec.tactical_contract.value
                if spec.tactical_contract is not None else None
            ),
            "tacticalTrump": (
                spec.tactical_trump.name.lower()
                if spec.tactical_trump is not None else None
            ),
            "tacticalCoefficients": [
                [card, seat.name.lower(), coefficient]
                for (card, seat), coefficient in sorted(
                    coefficients.items(),
                    key=lambda item: (item[0][0], int(item[0][1])),
                )
            ],
            "dealer": layout.dealer.name.lower(),
            "rootSeed": int(root_seed),
            "tieCosts": [tie_costs[key] for key in sorted(tie_costs, key=lambda item: (item[0], int(item[1])))],
        })
        search_identity_hash = self._search_identity_hash(
            spec=spec,
            model_hash=model_hash,
            root_seed=root_seed,
            dealer=layout.dealer,
            rules=rules,
            backend_version=backend_version,
            no_good_cut_type="final_hand_allocation_v1",
        )
        frozen_structural_score: int | None = None
        if coefficients:
            self._require_deadline(deadline_monotonic)
            structural_solver = self._solver(
                cp_model,
                spec,
                root_seed,
                0,
                deadline_monotonic=deadline_monotonic,
            )
            status = structural_solver.solve(model)
            self._require_deadline(deadline_monotonic)
            self._require_optimal(cp_model, status)
            frozen_structural_score = int(
                structural_solver.value(tactical_expression)
            )
            model.add(tactical_expression == frozen_structural_score)
            model.clear_objective()
            model.minimize(tie_expression)

        preceding_cut_hashes: list[str] = []
        for ordinal, rejected in enumerate(rejected_candidates):
            if not isinstance(rejected, ConstraintDealCandidateV1):
                raise TypeError(
                    "rejected candidates must be ConstraintDealCandidateV1 values"
                )
            hands = self._hands_from_deck(rejected.deck_order, layout=layout)
            canonical_deck = self._compile_hands_to_deck(
                hands,
                layout=layout,
                root_seed=root_seed,
                profile_hash=spec.sha256,
                rules=rules,
                tactical_contract=spec.tactical_contract,
                tactical_trump=spec.tactical_trump,
            )
            if rejected.deck_order != canonical_deck:
                raise ValueError(
                    "rejected compact candidate deck is not canonically compiled"
                )
            expected_ordering_score = sum(
                tie_costs[card, seat]
                for seat, cards in hands.items()
                for card in cards
            )
            actual_structural_score = sum(
                coefficient
                for (card, seat), coefficient in coefficients.items()
                if card in hands[seat]
            )
            if actual_structural_score != int(frozen_structural_score or 0):
                raise ValueError(
                    "rejected candidate is not from the proved structural optimum"
                )
            cut_hash = self._validate_rejected_candidate(
                rejected,
                ordinal=ordinal,
                expected_profile_hash=spec.sha256,
                expected_model_hash=model_hash,
                expected_search_identity_hash=search_identity_hash,
                expected_backend_version=backend_version,
                expected_cut_type="final_hand_allocation_v1",
                expected_preceding_cut_hashes=tuple(preceding_cut_hashes),
                expected_ordering_score=expected_ordering_score,
                expected_structural_score=actual_structural_score,
                layout=layout,
            )
            model.add(
                sum(
                    membership[card, seat]
                    for seat, cards in hands.items()
                    for card in cards
                )
                <= 31
            )
            preceding_cut_hashes.append(cut_hash)

        results: list[ConstraintDealCandidateV1] = []
        continuation_receipt: ConstraintDealContinuationReceiptV1 | None = None
        first_ordinal = len(rejected_candidates)
        stop_ordinal = min(
            spec.maximum_solutions + int(continuation_probe),
            first_ordinal + maximum_new_solutions,
        )
        for ordinal in range(first_ordinal, stop_ordinal):
            self._require_deadline(deadline_monotonic)
            solver = self._solver(
                cp_model,
                spec,
                root_seed,
                ordinal,
                deadline_monotonic=deadline_monotonic,
            )
            status = solver.solve(model)
            self._require_deadline(deadline_monotonic)
            if status == cp_model.INFEASIBLE:
                if continuation_probe:
                    continuation_receipt = self._build_continuation_receipt(
                        profile_hash=spec.sha256,
                        model_hash=model_hash,
                        backend_version=backend_version,
                        solver_status="infeasible",
                        solution_ordinal=ordinal,
                        search_identity_hash=search_identity_hash,
                        preceding_no_good_cut_hashes=tuple(
                            preceding_cut_hashes
                        ),
                        no_good_cut_type="final_hand_allocation_v1",
                        exhausted=True,
                        witness=None,
                        structural_objective_score=int(
                            frozen_structural_score or 0
                        ),
                    )
                break
            self._require_optimal(cp_model, status)
            hands = {
                seat: tuple(
                    card for card in range(32)
                    if solver.boolean_value(membership[card, seat])
                )
                for seat in Seat
            }
            deck_order = self._compile_hands_to_deck(
                hands,
                layout=layout,
                root_seed=root_seed,
                profile_hash=spec.sha256,
                rules=rules,
                tactical_contract=spec.tactical_contract,
                tactical_trump=spec.tactical_trump,
            )
            self._require_deadline(deadline_monotonic)
            commitment = hashlib.sha256(bytes(deck_order)).hexdigest()
            ordering_score = int(solver.value(tie_expression))
            structural_score = int(frozen_structural_score or 0)
            candidate = self._build_candidate(
                ordinal=ordinal,
                deck_order=deck_order,
                deck_commitment=commitment,
                ordering_score=ordering_score,
                structural_objective_score=structural_score,
                profile_hash=spec.sha256,
                model_hash=model_hash,
                search_identity_hash=search_identity_hash,
                backend_version=backend_version,
                no_good_cut_type="final_hand_allocation_v1",
                preceding_no_good_cut_hashes=tuple(preceding_cut_hashes),
            )
            if continuation_probe:
                if any(
                    candidate.candidate_hash == rejected.candidate_hash
                    or candidate.deck_commitment
                    == rejected.deck_commitment
                    for rejected in rejected_candidates
                ):
                    raise RuntimeError(
                        "continuation witness repeated a rejected candidate"
                    )
                continuation_receipt = self._build_continuation_receipt(
                    profile_hash=spec.sha256,
                    model_hash=model_hash,
                    backend_version=backend_version,
                    solver_status="optimal",
                    solution_ordinal=ordinal,
                    search_identity_hash=search_identity_hash,
                    preceding_no_good_cut_hashes=tuple(
                        preceding_cut_hashes
                    ),
                    no_good_cut_type="final_hand_allocation_v1",
                    exhausted=False,
                    witness=candidate,
                    structural_objective_score=structural_score,
                )
                break
            results.append(candidate)
            preceding_cut_hashes.append(
                self._no_good_cut_hash(candidate, layout=layout)
            )
            model.add(sum(membership[card, seat] for seat, cards in hands.items() for card in cards) <= 31)
        return _ConstraintDealGenerationResultV1(
            tuple(results), continuation_receipt
        )

    @classmethod
    def _build_candidate(
        cls,
        *,
        ordinal: int,
        deck_order: tuple[int, ...],
        deck_commitment: str,
        ordering_score: int,
        structural_objective_score: int,
        profile_hash: str,
        model_hash: str,
        search_identity_hash: str,
        backend_version: str,
        no_good_cut_type: str,
        preceding_no_good_cut_hashes: tuple[str, ...],
    ) -> ConstraintDealCandidateV1:
        candidate = ConstraintDealCandidateV1(
            ordinal=ordinal,
            deck_order=deck_order,
            deck_commitment=deck_commitment,
            ordering_score=ordering_score,
            structural_objective_score=structural_objective_score,
            profile_hash=profile_hash,
            model_hash=model_hash,
            backend_version=backend_version,
            solver_status="optimal",
            no_good_cut_type=no_good_cut_type,
            candidate_hash="",
            search_identity_hash=search_identity_hash,
            preceding_no_good_cut_hashes=preceding_no_good_cut_hashes,
        )
        return replace(
            candidate,
            candidate_hash=canonical_sha256(candidate.integrity_core()),
        ).verify()

    @classmethod
    def _build_continuation_receipt(
        cls,
        *,
        profile_hash: str,
        model_hash: str,
        backend_version: str,
        solver_status: str,
        solution_ordinal: int,
        search_identity_hash: str,
        preceding_no_good_cut_hashes: tuple[str, ...],
        no_good_cut_type: str,
        exhausted: bool,
        witness: ConstraintDealCandidateV1 | None,
        structural_objective_score: int,
    ) -> ConstraintDealContinuationReceiptV1:
        _require_exact_int(solution_ordinal, "continuation solutionOrdinal")
        _require_exact_int(
            structural_objective_score,
            "continuation structuralObjectiveScore",
        )
        if type(exhausted) is not bool:
            raise ValueError("continuation exhausted must be a boolean")
        if solver_status not in {"optimal", "infeasible"}:
            raise ValueError(
                "a continuation receipt requires a conclusive solver status"
            )
        if exhausted != (solver_status == "infeasible"):
            raise ValueError(
                "continuation exhaustion must match the solver status"
            )
        if bool(witness) == exhausted:
            raise ValueError(
                "continuation witness presence must match feasibility"
            )
        if len(preceding_no_good_cut_hashes) != solution_ordinal:
            raise ValueError(
                "continuation receipt lost the complete ordered cut prefix"
            )
        if witness is not None:
            witness.verify()
            if witness.ordinal != solution_ordinal:
                raise ValueError("continuation witness ordinal is invalid")
            if (
                witness.profile_hash != profile_hash
                or witness.model_hash != model_hash
                or witness.backend_version != backend_version
                or witness.search_identity_hash != search_identity_hash
                or witness.no_good_cut_type != no_good_cut_type
                or witness.preceding_no_good_cut_hashes
                != preceding_no_good_cut_hashes
                or witness.structural_objective_score
                != structural_objective_score
            ):
                raise ValueError(
                    "continuation witness changed the governed search identity"
                )
        receipt = ConstraintDealContinuationReceiptV1(
            profile_hash=profile_hash,
            model_hash=model_hash,
            backend_version=backend_version,
            solver_status=solver_status,
            solution_ordinal=solution_ordinal,
            search_identity_hash=search_identity_hash,
            preceding_no_good_cut_hashes=preceding_no_good_cut_hashes,
            no_good_cut_type=no_good_cut_type,
            exhausted=exhausted,
            witness_candidate_hash=(
                witness.candidate_hash if witness is not None else None
            ),
            witness_deck_commitment=(
                witness.deck_commitment if witness is not None else None
            ),
            witness_ordering_score=(
                witness.ordering_score if witness is not None else None
            ),
            structural_objective_score=structural_objective_score,
            receipt_hash="",
        )
        return replace(
            receipt,
            receipt_hash=canonical_sha256(receipt.integrity_core()),
        ).verify()

    @classmethod
    def _deal_layout_document(
        cls,
        layout: DealLayoutV1,
        *,
        structural_domain_hash: str,
    ) -> dict[str, Any]:
        core = {
            "schemaVersion": 1,
            "kind": "cp_sat_deal_layout_commitment_v1",
            "structuralDomainIrHash": structural_domain_hash,
            "dealer": layout.dealer.name.lower(),
            "buyer": (
                layout.buyer.name.lower()
                if layout.buyer is not None
                else None
            ),
            "initialHandPositions": {
                seat.name.lower(): list(layout.positions_for(seat, final=False))
                for seat in Seat
            },
            "finalHandPositions": (
                {
                    seat.name.lower(): list(layout.positions_for(seat, final=True))
                    for seat in Seat
                }
                if layout.buyer is not None
                else None
            ),
            "purchasePosition": layout.purchase_position,
            "undealtPositions": list(layout.undealt_positions),
        }
        return {**core, "sha256": canonical_sha256(core)}

    @classmethod
    def deal_layout_document(
        cls,
        *,
        spec: DealConstraintSpecV1,
        dealer: Seat,
        rules: RuleBook,
    ) -> dict[str, Any]:
        """Return the canonical governed deal-layout commitment."""

        return cls._deal_layout_document(
            DealLayoutV1.build(
                rules.structural_domain,
                dealer=dealer,
                buyer=spec.assumed_buyer,
            ),
            structural_domain_hash=rules.structural_domain.sha256,
        )

    @classmethod
    def validate_deal_layout_document(
        cls,
        value: Mapping[str, Any],
        *,
        structural_domain_document: Mapping[str, Any],
        dealer: str,
        buyer: str | None,
    ) -> dict[str, Any]:
        """Rebuild and exact-compare a persisted governed deal layout."""

        if not isinstance(value, Mapping):
            raise ValueError("deal layout commitment must be an object")
        if dealer not in {"east", "south", "west", "north"} or buyer not in {
            None, "east", "south", "west", "north",
        }:
            raise ValueError("deal layout dealer or buyer is invalid")
        structural_domain = StructuralDomainIRV1.from_dict(
            structural_domain_document
        )
        expected = cls._deal_layout_document(
            DealLayoutV1.build(
                structural_domain,
                dealer=Seat.parse(dealer),
                buyer=Seat.parse(buyer) if buyer is not None else None,
            ),
            structural_domain_hash=structural_domain.sha256,
        )
        if canonical_sha256(dict(value)) != canonical_sha256(expected):
            raise ValueError("deal layout commitment is not canonical")
        return expected

    @classmethod
    def search_identity_document(
        cls,
        *,
        spec: DealConstraintSpecV1,
        model_hash: str,
        root_seed: int,
        dealer: Seat,
        rules: RuleBook,
        backend_version: str,
        no_good_cut_type: str,
    ) -> dict[str, Any]:
        """Return the canonical governed-search identity and its hash."""

        core = {
            "schemaVersion": 1,
            "kind": "cp_sat_deal_search_identity_v1",
            "pluginId": cls.descriptor.id,
            "pluginVersion": cls.descriptor.version,
            "backend": spec.backend,
            "backendVersion": backend_version,
            "profileId": spec.profile_id,
            "profileRevision": spec.revision,
            "profileHash": spec.sha256,
            "modelHash": model_hash,
            "rootSeed": int(root_seed),
            "dealer": dealer.name.lower(),
            "rulesHash": rules.sha256,
            "structuralDomainIrHash": rules.structural_domain.sha256,
            "structuralDomain": rules.structural_domain.to_dict(),
            "playDomainIrHash": rules.play_domain.sha256,
            "dealLayout": cls.deal_layout_document(
                spec=spec,
                dealer=dealer,
                rules=rules,
            ),
            "noGoodCutType": no_good_cut_type,
        }
        return {**core, "sha256": canonical_sha256(core)}

    @classmethod
    def _search_identity_hash(
        cls,
        *,
        spec: DealConstraintSpecV1,
        model_hash: str,
        root_seed: int,
        dealer: Seat,
        rules: RuleBook,
        backend_version: str,
        no_good_cut_type: str,
    ) -> str:
        return str(cls.search_identity_document(
            spec=spec,
            model_hash=model_hash,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            backend_version=backend_version,
            no_good_cut_type=no_good_cut_type,
        )["sha256"])

    @staticmethod
    def _hands_from_deck(
        deck_order: tuple[int, ...],
        *,
        layout: DealLayoutV1,
    ) -> dict[Seat, tuple[int, ...]]:
        if len(deck_order) != 32 or sorted(deck_order) != list(range(32)):
            raise ValueError(
                "rejected candidate deck must be a complete permutation"
            )
        return {
            seat: tuple(sorted(
                deck_order[position]
                for position in layout.positions_for(seat, final=True)
            ))
            for seat in Seat
        }

    @classmethod
    def exact_no_good_cut_document(
        cls,
        candidate: ConstraintDealCandidateV1,
        *,
        layout: DealLayoutV1,
    ) -> dict[str, Any]:
        """Return the canonical exact-cut preimage and its integrity hash."""

        candidate.verify()
        if candidate.no_good_cut_type == "complete_deck_assignment_v1":
            exact_assignment: Any = list(candidate.deck_order)
        elif candidate.no_good_cut_type == "final_hand_allocation_v1":
            if not isinstance(layout, DealLayoutV1):
                raise TypeError("final-hand no-good cuts require DealLayoutV1")
            hands = cls._hands_from_deck(candidate.deck_order, layout=layout)
            exact_assignment = {
                seat.name.lower(): list(hands[seat]) for seat in Seat
            }
        else:
            raise ValueError("rejected candidate declares an unsupported no-good cut")
        core = {
            "schemaVersion": 1,
            "kind": "cp_sat_exact_candidate_no_good_v1",
            "candidateHash": candidate.candidate_hash,
            "searchIdentityHash": candidate.search_identity_hash,
            "profileHash": candidate.profile_hash,
            "modelHash": candidate.model_hash,
            "ordinal": candidate.ordinal,
            "noGoodCutType": candidate.no_good_cut_type,
            "exactAssignment": exact_assignment,
        }
        cut_hash = canonical_sha256(core)
        document_core = {
            **core,
            "cutSha256": cut_hash,
            "validationFinalHandPositions": (
                {
                    seat.name.lower(): list(layout.positions_for(seat, final=True))
                    for seat in Seat
                }
                if candidate.no_good_cut_type == "final_hand_allocation_v1"
                else None
            ),
            "validationDealer": (
                layout.dealer.name.lower()
                if candidate.no_good_cut_type == "final_hand_allocation_v1"
                else None
            ),
            "validationBuyer": (
                layout.buyer.name.lower()
                if candidate.no_good_cut_type == "final_hand_allocation_v1"
                and layout.buyer is not None
                else None
            ),
        }
        return {**document_core, "sha256": canonical_sha256(document_core)}

    @classmethod
    def _no_good_cut_hash(
        cls,
        candidate: ConstraintDealCandidateV1,
        *,
        layout: DealLayoutV1,
    ) -> str:
        return str(
            cls.exact_no_good_cut_document(candidate, layout=layout)["cutSha256"]
        )

    @classmethod
    def _validate_rejected_candidate(
        cls,
        candidate: ConstraintDealCandidateV1,
        *,
        ordinal: int,
        expected_profile_hash: str,
        expected_model_hash: str,
        expected_search_identity_hash: str,
        expected_backend_version: str,
        expected_cut_type: str,
        expected_preceding_cut_hashes: tuple[str, ...],
        expected_ordering_score: int,
        expected_structural_score: int,
        layout: DealLayoutV1,
    ) -> str:
        if not isinstance(candidate, ConstraintDealCandidateV1):
            raise TypeError(
                "rejected candidates must be ConstraintDealCandidateV1 values"
            )
        candidate.verify()
        if candidate.ordinal != ordinal:
            raise ValueError(
                "rejected candidates must be the exact ordered ordinal prefix"
            )
        if candidate.profile_hash != expected_profile_hash:
            raise ValueError(
                "rejected candidate belongs to a different governed profile"
            )
        if candidate.search_identity_hash != expected_search_identity_hash:
            raise ValueError(
                "rejected candidate belongs to a different governed search identity"
            )
        if candidate.model_hash != expected_model_hash:
            raise ValueError(
                "rejected candidate belongs to a different governed model"
            )
        if candidate.backend_version != expected_backend_version:
            raise ValueError(
                "rejected candidate belongs to a different solver backend"
            )
        if candidate.no_good_cut_type != expected_cut_type:
            raise ValueError(
                "rejected candidate no-good cut does not match the governed model"
            )
        if (
            candidate.preceding_no_good_cut_hashes
            != expected_preceding_cut_hashes
        ):
            raise ValueError(
                "rejected candidate no-good provenance is not the ordered prefix"
            )
        if candidate.ordering_score != expected_ordering_score:
            raise ValueError("rejected candidate ordering score is invalid")
        if candidate.structural_objective_score != expected_structural_score:
            raise ValueError(
                "rejected candidate is not from the proved structural optimum"
            )
        return cls._no_good_cut_hash(candidate, layout=layout)

    @staticmethod
    def _full_solver(
        cp_model,
        spec: DealConstraintSpecV1,
        root_seed: int,
        ordinal: int,
        *,
        deadline_monotonic: float | None,
        fix_hints: bool = False,
    ):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = spec.solver_workers
        solver.parameters.random_seed = (
            int(root_seed) + spec.seed_offset + ordinal
        ) % 2_147_483_647
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.max_number_of_conflicts = spec.maximum_conflicts
        solver.parameters.max_time_in_seconds = (
            OrToolsCpSatDealGeneratorV1._remaining_solver_seconds(
                spec,
                deadline_monotonic,
            )
        )
        solver.parameters.fix_variables_to_their_hinted_value = bool(
            fix_hints
        )
        return solver

    @staticmethod
    def _solver(
        cp_model,
        spec: DealConstraintSpecV1,
        root_seed: int,
        ordinal: int,
        *,
        deadline_monotonic: float | None,
    ):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = spec.solver_workers
        solver.parameters.random_seed = (
            int(root_seed) + spec.seed_offset + ordinal
        ) % 2_147_483_647
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        solver.parameters.max_number_of_conflicts = spec.maximum_conflicts
        solver.parameters.max_time_in_seconds = (
            OrToolsCpSatDealGeneratorV1._remaining_solver_seconds(
                spec,
                deadline_monotonic,
            )
        )
        return solver

    @staticmethod
    def _require_deadline(deadline_monotonic: float | None) -> None:
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            raise InterruptedError("CP-SAT deal generation deadline exceeded")

    @staticmethod
    def _remaining_solver_seconds(
        spec: DealConstraintSpecV1,
        deadline_monotonic: float | None,
    ) -> float:
        if deadline_monotonic is None:
            return spec.safety_timeout_seconds
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise InterruptedError("CP-SAT deal generation deadline exceeded")
        return min(spec.safety_timeout_seconds, remaining)

    @staticmethod
    def _require_optimal(cp_model, status: int) -> None:
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "CP-SAT deal construction did not prove an optimal deterministic result"
            )

    @staticmethod
    def _compile_hands_to_deck(
        hands: Mapping[Seat, tuple[int, ...]],
        *,
        layout: DealLayoutV1,
        root_seed: int,
        profile_hash: str,
        rules: RuleBook,
        tactical_contract: Contract | None,
        tactical_trump: Suit | None,
    ) -> tuple[int, ...]:
        by_position = [-1] * 32
        buyer = layout.buyer
        if buyer is None:
            raise ValueError("final-hand compilation requires an assumed buyer")
        play_domain = rules.play_domain.require_integrity(
            rules.structural_domain
        )

        def ordering_strength(card_id: int) -> int:
            if tactical_contract is None:
                # With no selected contract this is pure structural ordering;
                # card id is deterministic and makes no contract claim.
                return int(card_id)
            return play_domain.strength(
                Card(card_id),
                tactical_contract,
                tactical_trump,
            )

        for seat in Seat:
            cards = list(hands[seat])
            initial_positions = tuple(sorted(layout.positions_for(seat, final=False)))
            if seat == buyer:
                purchase = min(
                    cards,
                    key=lambda card: (
                        ordering_strength(card),
                        canonical_sha256({
                            "kind": "purchase_card_choice_v1",
                            "rootSeed": int(root_seed),
                            "profileHash": profile_hash,
                            "card": card,
                        }),
                    ),
                )
                by_position[layout.purchase_position] = purchase
                cards.remove(purchase)
            initial_cards = sorted(
                cards,
                key=(
                    (lambda card: (-ordering_strength(card), card))
                    if seat == buyer
                    else (lambda card: (ordering_strength(card), card))
                ),
            )[: len(initial_positions)]
            extras = sorted(set(cards) - set(initial_cards))
            for position, card in zip(initial_positions, sorted(initial_cards)):
                by_position[position] = card
            extra_positions = tuple(
                position
                for position in sorted(layout.positions_for(seat, final=True))
                if position not in set(initial_positions)
                and position != layout.purchase_position
            )
            for position, card in zip(extra_positions, extras):
                by_position[position] = card
        deck = tuple(by_position)
        if sorted(deck) != list(range(32)):
            raise AssertionError("seat allocation did not compile to a complete deck")
        return deck

    @staticmethod
    def _add_position_symmetry_breaking(model, variables, layout) -> None:
        """Canonicalize permutations that have identical Baloot semantics.

        Cards dealt into the same private hand packet are unordered from the
        engine's perspective.  Ordering them by card id removes a very large
        CP-SAT symmetry without changing the represented deal, purchase card,
        bids, projects, legal play, or replay.
        """
        for group in OrToolsCpSatDealGeneratorV1._position_symmetry_groups(
            layout
        ):
            card_values = [
                sum(card * variables[card, position] for card in range(32))
                for position in group
            ]
            for left, right in zip(card_values, card_values[1:]):
                model.add(left < right)

    @staticmethod
    def _position_symmetry_groups(layout) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = [
            tuple(sorted(layout.positions_for(seat, final=False)))
            for seat in Seat
        ]
        if layout.final_hand_positions is not None:
            for seat in Seat:
                initial = set(layout.positions_for(seat, final=False))
                extras = tuple(
                    position
                    for position in sorted(layout.positions_for(seat, final=True))
                    if position not in initial
                    and position != layout.purchase_position
                )
                if extras:
                    groups.append(extras)
        return tuple(groups)

    @classmethod
    def _seeded_unconstrained_deck(
        cls,
        *,
        spec: DealConstraintSpecV1,
        root_seed: int,
        ordinal: int,
        dealer: Seat,
        layout: DealLayoutV1,
        rules: RuleBook,
    ) -> tuple[int, ...]:
        hint_seed = int(
            canonical_sha256(
                {
                    "kind": "cp_sat_unconstrained_deal_hint_seed_v1",
                    "rootSeed": int(root_seed),
                    "ordinal": int(ordinal),
                    "profileHash": spec.sha256,
                    "dealer": dealer.name.lower(),
                    "rulesHash": rules.sha256,
                }
            )[:16],
            16,
        )
        deck = list(
            shuffle_and_cut(
                hint_seed,
                None,
                dealer=dealer,
                structural_domain=rules.structural_domain,
            ).record.deck_order
        )
        for group in cls._position_symmetry_groups(layout):
            cards = sorted(deck[position] for position in group)
            for position, card in zip(group, cards):
                deck[position] = card
        return tuple(deck)

    @staticmethod
    def _tactical_coefficients(
        spec: DealConstraintSpecV1,
        *,
        rules: RuleBook,
    ) -> dict[tuple[int, Seat], int]:
        if not spec.tactical_objective_weights or spec.objective_target_seat is None:
            return {}
        target = spec.objective_target_seat
        partner = target.partner
        team = {target, partner}
        opponents = set(Seat) - team
        strengths = OrToolsCpSatDealGeneratorV1._tactical_card_strengths(
            spec,
            rules=rules,
        )
        coefficients: dict[tuple[int, Seat], int] = {}

        def add(seats: set[Seat], card: int, value: int) -> None:
            if value == 0:
                return
            for seat in seats:
                key = (card, seat)
                coefficients[key] = coefficients.get(key, 0) + value

        for kind, configured_weight in spec.tactical_objective_weights:
            weight = int(configured_weight)
            for card_id in range(32):
                card = Card(card_id)
                strength = strengths[card_id]
                strong = strength * 2
                low = max(
                    0,
                    (
                        OrToolsCpSatDealGeneratorV1._LOW_STRENGTH_MAX
                        + 1
                        - strength
                    )
                    * 2,
                )
                if kind == "only_winning_move":
                    # Structural guidance only: concentrate resources in the
                    # actor while leaving enough opposing strength to create a
                    # narrow decision. Causal proof remains a hard later gate.
                    add({target}, card_id, weight * (strong * 2 + low))
                    add({partner}, card_id, weight * low)
                    add(opponents, card_id, weight * strong)
                elif kind in {"kabout_creation", "kabout_preservation", "contract_rescue"}:
                    add({target}, card_id, weight * strong * 2)
                    add(set(Seat) - {target}, card_id, -weight * strong)
                elif kind == "kabout_prevention":
                    add({target}, card_id, weight * strong * 2)
                    add(set(Seat) - {target}, card_id, -weight * strong)
                elif kind == "small_card_big_gain":
                    add({target}, card_id, weight * low)
                    add({partner}, card_id, weight * strong)
                elif kind == "large_card_sacrifice":
                    add({target}, card_id, weight * strong)
                    add({partner}, card_id, weight * max(0, strong - 2))
                elif kind in {"entry_unblocking", "partner_enablement"}:
                    add({target}, card_id, weight * low)
                    add({partner}, card_id, weight * strong)
                elif kind == "trump_control":
                    trump_resource = int(
                        spec.tactical_contract is not None
                        and spec.tactical_contract.effective == Contract.HOKUM
                        and spec.tactical_trump is not None
                        and card.suit == spec.tactical_trump
                    ) * 5
                    add(team, card_id, weight * (strong + trump_resource))
                elif kind == "project_bidding":
                    add({target}, card_id, weight * strong)
                elif kind == "debate":
                    # Prefer two opposing hands with comparable middle/high
                    # resources. The information-set analyzer, not this proxy,
                    # proves whether supported alternatives are genuinely close.
                    balanced = max(0, 10 - abs(strong - 5))
                    add({target, target.next()}, card_id, weight * balanced)
                    add({partner}, card_id, weight * low)
                else:
                    raise ValueError(
                        f"unsupported tactical construction objective: {kind}"
                    )
        return coefficients

    @staticmethod
    def _add_tactical_balance(
        model,
        membership,
        spec: DealConstraintSpecV1,
        *,
        rules: RuleBook,
    ) -> None:
        if spec.tactical_deal_balance is None:
            return
        if spec.objective_target_seat is None:
            raise ValueError("tactical deal balance requires an objective target")
        difference, minimum_high, maximum_high, minimum_low = (
            spec.tactical_deal_balance
        )
        high_cards, low_cards = (
            OrToolsCpSatDealGeneratorV1._tactical_card_bands(
                spec,
                rules=rules,
            )
        )
        high_counts = {}
        for seat in Seat:
            values = [membership[card_id, seat] for card_id in high_cards]
            high_counts[seat] = sum(values)
            model.add(high_counts[seat] >= minimum_high)
            model.add(high_counts[seat] <= maximum_high)
        target = spec.objective_target_seat
        model.add(
            sum(membership[card_id, target] for card_id in low_cards)
            >= minimum_low
        )
        target_team_high = high_counts[target] + high_counts[target.partner]
        opponent_team_high = (
            high_counts[target.next()] + high_counts[target.previous()]
        )
        model.add(target_team_high - opponent_team_high <= difference)
        model.add(opponent_team_high - target_team_high <= difference)

    @staticmethod
    def _tactical_card_strengths(
        spec: DealConstraintSpecV1,
        *,
        rules: RuleBook,
    ) -> dict[int, int]:
        contract = spec.tactical_contract
        if contract is None:
            raise ValueError(
                "contract-sensitive tactical construction requires an exact contract"
            )
        trump = spec.tactical_trump
        if contract.effective == Contract.HOKUM and trump is None:
            raise ValueError("Hokum tactical construction requires an exact trump suit")
        play_domain = rules.play_domain.require_integrity(
            rules.structural_domain
        )
        return {
            card_id: play_domain.strength(Card(card_id), contract, trump)
            for card_id in range(32)
        }

    @staticmethod
    def _tactical_card_bands(
        spec: DealConstraintSpecV1,
        *,
        rules: RuleBook,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        strengths = OrToolsCpSatDealGeneratorV1._tactical_card_strengths(
            spec,
            rules=rules,
        )
        high_cards = tuple(
            card_id
            for card_id, strength in strengths.items()
            if strength >= OrToolsCpSatDealGeneratorV1._HIGH_STRENGTH_MIN
        )
        low_cards = tuple(
            card_id
            for card_id, strength in strengths.items()
            if strength <= OrToolsCpSatDealGeneratorV1._LOW_STRENGTH_MAX
        )
        if len(high_cards) != 12 or len(low_cards) != 12:
            raise AssertionError(
                "governed strength tables must expose twelve high and twelve low cards"
            )
        return high_cards, low_cards

    @staticmethod
    def _add_suit_ranges(model, variables, layout, rows, *, final: bool) -> None:
        for seat, suit, minimum, maximum in rows:
            positions = layout.positions_for(seat, final=final)
            cards = range(int(suit) * 8, int(suit) * 8 + 8)
            count = sum(variables[card, position] for card in cards for position in positions)
            model.add(count >= minimum)
            model.add(count <= maximum)

    @staticmethod
    def _add_project_pattern(model, variables, layout, constraint, rules) -> None:
        patterns = project_card_patterns(
            kind=constraint.kind,
            value=constraint.value,
            high_rank=constraint.high_rank,
            contract=constraint.contract,
            rules=rules,
        )
        if not patterns:
            raise ValueError("the active rules expose no matching project pattern")
        positions = layout.positions_for(constraint.seat, final=True)
        selected = []
        for index, mask in enumerate(patterns):
            literal = model.new_bool_var(f"project_pattern_{index}")
            selected.append(literal)
            for card in range(32):
                if mask & (1 << card):
                    model.add(
                        sum(variables[card, position] for position in positions) >= literal
                    )
        model.add_exactly_one(selected)
