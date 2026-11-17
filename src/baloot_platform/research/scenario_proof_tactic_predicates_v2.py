"""Engine-free lowerings for proof- and tactic-derived scenario claims."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import FULL_DECK
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.enums import Rank, Seat, Team
from baloot_platform.tactics import DELAYED_PAYOFF_TACTIC_IDS, TacticalDetectorProfileV1, TacticalKind

from .ortools_proof import OrToolsGraphProofOutcomeV1
from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_predicate_lowering_v2 import CpSatPredicateLoweringV1
from .scenario_registry_v2 import scenario_predicate_definitions_v2


_COMPLETE_PROOF_STATUSES = frozenset(
    {"exact_complete", "bounded_complete", "policy_complete"}
)
TACTICAL_DETECTOR_IDS_V1 = frozenset(item.value for item in TacticalKind)
_TACTICAL_DECISION_KINDS_V1 = ("bid", "escalation", "play", "project")
_TACTICAL_ROOT_BINDINGS_V1 = (
    ("root_action_graph", 2),
    ("root_actor_target_relation", 1),
    ("selected_root_action", 1),
)


def lower_native_proof_predicate_v3(
    predicate: ScenarioPredicateV1,
    context: Any,
    *,
    manifest: Any,
    root_selection_variables: Sequence[Any],
) -> CpSatPredicateLoweringV1:
    """Bind implicit tactical selection to the proof model's canonical root."""
    if not predicate.kind.startswith("tactic.") or context.proof.selected_action is not None:
        return manifest.lower_predicate(predicate, context)
    selections = tuple(root_selection_variables)
    roots = context.proof.root_actions
    if not selections or len(selections) != len(roots):
        raise ValueError("native canonical root selection is unavailable")
    lowerings = []
    matches = []
    for index, root in enumerate(roots):
        selected_context = copy.copy(context)
        selected_context.proof = copy.copy(context.proof)
        selected_context.proof.selected_action = root.action_code
        selected_context.selected_root_action_index = index
        selected_context.context_id = f"{context.context_id}-root-{index}"
        selected_context.sha256 = canonical_sha256({
            "contextHash": context.sha256, "selectedRootIndex": index,
            "selectedRootAction": root.action_code,
        })
        lowering = manifest.lower_predicate(predicate, selected_context)
        lowerings.append(lowering)
        matches.append(ProofTacticPredicateLoweringV2._and(
            context.model, (selections[index], lowering.literal),
            f"{selected_context.context_id}-{predicate.predicate_id}-selected",
        ))
    literal = ProofTacticPredicateLoweringV2._or(
        context.model, matches, f"{context.context_id}-{predicate.predicate_id}-selected"
    )

    def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
        selected = [index for index, choice in enumerate(selections) if solver.boolean_value(choice)]
        if len(selected) != 1:
            raise ValueError("native canonical root selection is not unique")
        return lowerings[selected[0]].extract(solver)["observed"]

    return CpSatPredicateLoweringV1(
        predicate.predicate_id, predicate.kind, literal, lowerings[0].evidence_extractor_id,
        canonical_sha256({"predicateHash": predicate.sha256,
            "rootSelectionVariables": [variable.name for variable in selections],
            "rootLoweringHashes": [lowering.lowering_hash for lowering in lowerings]}), extract,
    )


@dataclass(frozen=True, slots=True)
class TacticalDetectorPrerequisiteV1:
    """Closed producer requirements for one native tactical detector."""

    tactic_id: str
    decision_kinds: tuple[str, ...]
    required_utility: str | None
    required_proof_bindings: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.tactic_id not in TACTICAL_DETECTOR_IDS_V1:
            raise ValueError("tactical detector prerequisite has an unknown tactic")
        if not self.decision_kinds or len(self.decision_kinds) != len(
            set(self.decision_kinds)
        ):
            raise ValueError("tactical detector decision kinds must be unique")
        if not set(self.decision_kinds).issubset(_TACTICAL_DECISION_KINDS_V1):
            raise ValueError("tactical detector decision kind is unsupported")
        names = [name for name, _minimum in self.required_proof_bindings]
        if len(names) != len(set(names)) or any(
            not name or type(minimum) is not int or minimum < 1
            for name, minimum in self.required_proof_bindings
        ):
            raise ValueError("tactical detector proof bindings are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tacticId": self.tactic_id,
            "decisionKinds": list(self.decision_kinds),
            "requiredUtility": self.required_utility,
            "requiredProofBindings": [
                {"binding": binding, "minimumCount": minimum}
                for binding, minimum in self.required_proof_bindings
            ],
        }


def _tactical_prerequisite(
    tactic: TacticalKind,
    *,
    decision_kinds: tuple[str, ...] = _TACTICAL_DECISION_KINDS_V1,
    required_utility: str | None = None,
    bindings: tuple[tuple[str, int], ...] = (),
) -> TacticalDetectorPrerequisiteV1:
    return TacticalDetectorPrerequisiteV1(
        tactic.value,
        tuple(sorted(decision_kinds)),
        required_utility,
        tuple(sorted((*_TACTICAL_ROOT_BINDINGS_V1, *bindings))),
    )


TACTICAL_DETECTOR_PREREQUISITES_V1 = MappingProxyType(
    {
        TacticalKind.ONLY_WINNING_MOVE.value: _tactical_prerequisite(
            TacticalKind.ONLY_WINNING_MOVE
        ),
        TacticalKind.CONTRACT_RESCUE.value: _tactical_prerequisite(
            TacticalKind.CONTRACT_RESCUE,
            required_utility="contract_success",
        ),
        TacticalKind.PROJECT_BIDDING.value: _tactical_prerequisite(
            TacticalKind.PROJECT_BIDDING,
            decision_kinds=("bid", "project"),
        ),
        TacticalKind.DEBATE.value: _tactical_prerequisite(TacticalKind.DEBATE),
        **{
            tactic.value: _tactical_prerequisite(
                tactic,
                required_utility="kabout",
                bindings=(("base_trick_variables", 2),),
            )
            for tactic in (
                TacticalKind.KABOUT_CREATION,
                TacticalKind.KABOUT_PRESERVATION,
                TacticalKind.KABOUT_PREVENTION,
            )
        },
        **{
            tactic.value: _tactical_prerequisite(
                tactic,
                decision_kinds=("play",),
                bindings=(("principal_trick_winner_seat_variables", 2),),
            )
            for tactic in (
                TacticalKind.ENTRY_UNBLOCKING,
                TacticalKind.PARTNER_ENABLEMENT,
            )
        },
        TacticalKind.TRUMP_CONTROL.value: _tactical_prerequisite(
            TacticalKind.TRUMP_CONTROL,
            decision_kinds=("play",),
            bindings=(
                ("principal_variation_has_continuation_variable", 1),
                ("trump_suit_variable", 1),
            ),
        ),
        TacticalKind.SMALL_CARD_BIG_GAIN.value: _tactical_prerequisite(
            TacticalKind.SMALL_CARD_BIG_GAIN,
            decision_kinds=("play",),
            bindings=(
                ("card_strength_by_id", 32),
                ("principal_variation_has_continuation_variable", 1),
            ),
        ),
        TacticalKind.LARGE_CARD_SACRIFICE.value: _tactical_prerequisite(
            TacticalKind.LARGE_CARD_SACRIFICE,
            decision_kinds=("play",),
            bindings=(
                ("card_strength_by_id", 32),
                ("first_principal_trick_winner_team_variable", 1),
                ("principal_variation_has_continuation_variable", 1),
            ),
        ),
    }
)

if set(TACTICAL_DETECTOR_PREREQUISITES_V1) != TACTICAL_DETECTOR_IDS_V1:
    raise RuntimeError("tactical detector prerequisite coverage is incomplete")


def tactical_detector_prerequisite_catalog_v1() -> dict[str, Any]:
    core = {
        "schemaVersion": 1,
        "detectors": [
            TACTICAL_DETECTOR_PREREQUISITES_V1[tactic_id].to_dict()
            for tactic_id in sorted(TACTICAL_DETECTOR_PREREQUISITES_V1)
        ],
    }
    return {**core, "sha256": canonical_sha256(core)}

PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2 = MappingProxyType(
    {
        "proof.winning_root_action_count": "cp_sat.proof_root_count_evidence_v2",
        "proof.winning_root_card_rank_count": (
            "cp_sat.proof_root_card_rank_count_evidence_v1"
        ),
        "proof.winning_continuation_line_count": (
            "cp_sat.proof_continuation_line_count_evidence_v1"
        ),
        "proof.principal_variation_trick_winner": (
            "cp_sat.proof_principal_trick_winner_evidence_v1"
        ),
        "proof.principal_variation_trick_team_rank_count": (
            "cp_sat.proof_principal_trick_team_rank_count_evidence_v1"
        ),
        "tactic.verified": "cp_sat.registered_tactic_verified_evidence_v1",
        "tactic.applicable": "cp_sat.registered_tactic_applicable_evidence_v1",
    }
)


def _strict_sha256(value: Any, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class PrincipalVariationPlayEvidenceV1:
    action_code: str
    actor_seat: str

    @classmethod
    def build(
        cls,
        action_code: str,
        actor_seat: str,
    ) -> "PrincipalVariationPlayEvidenceV1":
        action = LegalPlayActionV1.from_code(action_code)
        seat = Seat.parse(actor_seat).name.lower()
        return cls(action.code, seat)

    def to_dict(self) -> dict[str, Any]:
        action = LegalPlayActionV1.from_code(self.action_code)
        return {
            "action": action.code,
            "card": action.card.to_dict(),
            "actorSeat": self.actor_seat,
            "actorTeam": Seat.parse(self.actor_seat).team.name.lower(),
        }


@dataclass(frozen=True, slots=True)
class PrincipalVariationTrickEvidenceV1:
    trick_index: int
    plays: tuple[PrincipalVariationPlayEvidenceV1, ...]
    winner_position: int
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        trick_index: int,
        plays: Sequence[PrincipalVariationPlayEvidenceV1],
        winner_position: int,
    ) -> "PrincipalVariationTrickEvidenceV1":
        if type(trick_index) is not int or not 0 <= trick_index <= 7:
            raise ValueError("principal-variation trickIndex must be in [0, 7]")
        frozen = tuple(plays)
        if len(frozen) != 4 or any(
            not isinstance(item, PrincipalVariationPlayEvidenceV1) for item in frozen
        ):
            raise ValueError("principal-variation trick requires four typed plays")
        if len({item.action_code for item in frozen}) != 4:
            raise ValueError("principal-variation trick cards must be unique")
        if type(winner_position) is not int or not 0 <= winner_position < 4:
            raise ValueError("principal-variation winnerPosition must be in [0, 3]")
        core = {
            "schemaVersion": 1,
            "trickIndex": trick_index,
            "plays": [item.to_dict() for item in frozen],
            "winnerPosition": winner_position,
        }
        return cls(trick_index, frozen, winner_position, canonical_sha256(core))

    @property
    def winner(self) -> PrincipalVariationPlayEvidenceV1:
        return self.plays[self.winner_position]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "trickIndex": self.trick_index,
            "plays": [item.to_dict() for item in self.plays],
            "winnerPosition": self.winner_position,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RegisteredTacticalEvidenceV1:
    detector_id: str
    detector_version: str
    evidence_id: str
    applicable: bool
    verified: bool
    score: float
    acting_seat: str | None
    observer: str | None
    target_seat: str | None
    target_team: str | None
    causal_proof_hash: str | None
    source_proof_hash: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        detector_id: str,
        detector_version: str,
        evidence_id: str,
        applicable: bool,
        verified: bool,
        score: int | float,
        source_proof_hash: str,
        acting_seat: str | None = None,
        observer: str | None = None,
        target_seat: str | None = None,
        target_team: str | None = None,
        causal_proof_hash: str | None = None,
    ) -> "RegisteredTacticalEvidenceV1":
        if detector_id not in TACTICAL_DETECTOR_IDS_V1:
            raise ValueError("tactical detector is not registered")
        if type(detector_version) is not str or not detector_version.strip():
            raise ValueError("tactical detector version is required")
        if type(evidence_id) is not str or not evidence_id.strip():
            raise ValueError("tactical evidence ID is required")
        if type(applicable) is not bool or type(verified) is not bool:
            raise TypeError("tactical applicable and verified must be booleans")
        if verified and not applicable:
            raise ValueError("verified tactical evidence must be applicable")
        if type(score) not in {int, float} or not 0 <= float(score) <= 100:
            raise ValueError("tactical evidence score must be in [0, 100]")
        seats = {
            name: Seat.parse(value).name.lower() if value is not None else None
            for name, value in {
                "actingSeat": acting_seat,
                "observer": observer,
                "targetSeat": target_seat,
            }.items()
        }
        if target_team is not None and target_team not in {
            "north_south",
            "west_east",
        }:
            raise ValueError("tactical targetTeam is invalid")
        proof_hash = _strict_sha256(source_proof_hash, "sourceProofHash")
        causal_hash = (
            _strict_sha256(causal_proof_hash, "causalProofHash")
            if causal_proof_hash is not None
            else None
        )
        core = {
            "schemaVersion": 1,
            "detectorId": detector_id,
            "detectorVersion": detector_version.strip(),
            "evidenceId": evidence_id.strip(),
            "applicable": applicable,
            "verified": verified,
            "score": float(score),
            **seats,
            "targetTeam": target_team,
            "causalProofHash": causal_hash,
            "sourceProofHash": proof_hash,
        }
        return cls(
            detector_id,
            detector_version.strip(),
            evidence_id.strip(),
            applicable,
            verified,
            float(score),
            seats["actingSeat"],
            seats["observer"],
            seats["targetSeat"],
            target_team,
            causal_hash,
            proof_hash,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "detectorId": self.detector_id,
            "detectorVersion": self.detector_version,
            "evidenceId": self.evidence_id,
            "applicable": self.applicable,
            "verified": self.verified,
            "score": self.score,
            "actingSeat": self.acting_seat,
            "observer": self.observer,
            "targetSeat": self.target_seat,
            "targetTeam": self.target_team,
            "causalProofHash": self.causal_proof_hash,
            "sourceProofHash": self.source_proof_hash,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ProofTacticPredicateContextV1:
    model: cp_model.CpModel
    context_id: str
    proof: OrToolsGraphProofOutcomeV1
    principal_variation_tricks: tuple[PrincipalVariationTrickEvidenceV1, ...]
    tactical_evidence: tuple[RegisteredTacticalEvidenceV1, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("proof/tactic context requires a CpModel")
        if type(self.context_id) is not str or not self.context_id:
            raise ValueError("proof/tactic context ID is required")
        if not isinstance(self.proof, OrToolsGraphProofOutcomeV1):
            raise TypeError("proof/tactic context requires an OR-Tools proof outcome")
        proof_dict = self.proof.to_dict()
        declared = proof_dict.pop("sha256")
        if declared != canonical_sha256(proof_dict):
            raise ValueError("OR-Tools proof outcome checksum mismatch")
        tricks = tuple(self.principal_variation_tricks)
        evidence = tuple(self.tactical_evidence)
        if any(
            not isinstance(item, PrincipalVariationTrickEvidenceV1) for item in tricks
        ):
            raise TypeError("principal-variation tricks must use typed evidence")
        if any(not isinstance(item, RegisteredTacticalEvidenceV1) for item in evidence):
            raise TypeError("tactical evidence must use registered typed evidence")
        if len({item.trick_index for item in tricks}) != len(tricks):
            raise ValueError("principal-variation trick indices must be unique")
        if any(item.source_proof_hash != self.proof.sha256 for item in evidence):
            raise ValueError("tactical evidence belongs to a different proof")
        self._validate_principal_variation(tricks)
        core = {
            "schemaVersion": 1,
            "contextId": self.context_id,
            "proofHash": self.proof.sha256,
            "principalVariationTricks": [
                item.to_dict()
                for item in sorted(tricks, key=lambda row: row.trick_index)
            ],
            "tacticalEvidence": [
                item.to_dict() for item in sorted(evidence, key=lambda row: row.sha256)
            ],
        }
        object.__setattr__(self, "principal_variation_tricks", tricks)
        object.__setattr__(self, "tactical_evidence", evidence)
        object.__setattr__(self, "sha256", canonical_sha256(core))

    def _validate_principal_variation(
        self,
        tricks: tuple[PrincipalVariationTrickEvidenceV1, ...],
    ) -> None:
        principal = self.proof.principal_variation
        if not tricks:
            return
        flattened = [
            play.action_code
            for trick in sorted(tricks, key=lambda row: row.trick_index)
            for play in trick.plays
        ]
        if len(flattened) != len(set(flattened)):
            raise ValueError("principal-variation card evidence must be unique")
        principal_set = set(principal)
        mapped = [code for code in flattened if code in principal_set]
        if tuple(mapped) != principal:
            raise ValueError("principal-variation trick evidence does not match proof")
        first = next(
            index for index, code in enumerate(flattened) if code in principal_set
        )
        if any(code not in principal_set for code in flattened[first:]):
            raise ValueError("only the first principal trick may contain prefix plays")


class ProofTacticPredicateLoweringV2:
    """Lower exact immutable proof facts into composable CP-SAT literals."""

    def __init__(self) -> None:
        definitions = {item.kind: item for item in scenario_predicate_definitions_v2()}
        self._definitions = {
            kind: definitions[kind] for kind in PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2
        }

    @property
    def registrations(self) -> Mapping[str, str]:
        return PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2

    def lower_predicate(
        self,
        predicate: ScenarioPredicateV1,
        context: ProofTacticPredicateContextV1,
    ) -> CpSatPredicateLoweringV1:
        definition = self._definitions.get(predicate.kind)
        if definition is None or predicate.version != definition.version:
            raise ValueError(
                f"predicate has no proof/tactic lowering: {predicate.kind}@{predicate.version}"
            )
        definition.validate_arguments(predicate.arguments)
        literal, observed = self._lower_native(predicate, context)
        solver_bools = observed.pop("_solverBools", {})
        solver_ints = observed.pop("_solverInts", {})
        solver_scaled_ints = observed.pop("_solverScaledInts", {})
        solver_decimal_digits = observed.pop("_solverDecimalDigits", {})
        extractor_id = PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2[predicate.kind]
        lowering_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "predicateHash": predicate.sha256,
                "contextHash": context.sha256,
                "evidenceExtractor": extractor_id,
            }
        )
        frozen_observed = copy.deepcopy(observed)

        def extract(solver: cp_model.CpSolver) -> dict[str, Any]:
            result = copy.deepcopy(frozen_observed)
            result.update(
                {
                    key: bool(solver.boolean_value(variable))
                    for key, variable in solver_bools.items()
                }
            )
            result.update(
                {key: solver.value(variable) for key, variable in solver_ints.items()}
            )
            result.update(
                {
                    key: solver.value(variable) / scale
                    for key, (variable, scale) in solver_scaled_ints.items()
                }
            )
            result.update(
                {
                    key: str(sum(
                        int(solver.value(digit)) * (1_000_000**limb)
                        for limb, digit in enumerate(digits)
                    ))
                    for key, digits in solver_decimal_digits.items()
                }
            )
            return result

        return CpSatPredicateLoweringV1(
            predicate.predicate_id,
            predicate.kind,
            literal,
            extractor_id,
            lowering_hash,
            extract,
        )

    @staticmethod
    def _and(model: Any, values: Sequence[Any], name: str) -> Any:
        literal = model.new_bool_var(name)
        model.add_bool_and(list(values)).only_enforce_if(literal)
        model.add_bool_or([value.Not() for value in values]).only_enforce_if(
            literal.Not()
        )
        return literal

    @staticmethod
    def _or(model: Any, values: Sequence[Any], name: str) -> Any:
        literal = model.new_bool_var(name)
        model.add_bool_or(list(values)).only_enforce_if(literal)
        model.add_bool_and([value.Not() for value in values]).only_enforce_if(
            literal.Not()
        )
        return literal

    @classmethod
    def _range_literal(
        cls,
        model: Any,
        value: Any,
        minimum: int,
        maximum: int,
        name: str,
    ) -> Any:
        if minimum > maximum:
            raise ValueError("predicate minimum cannot exceed maximum")
        lower = model.new_bool_var(f"{name}_minimum")
        upper = model.new_bool_var(f"{name}_maximum")
        model.add(value >= minimum).only_enforce_if(lower)
        model.add(value < minimum).only_enforce_if(lower.Not())
        model.add(value <= maximum).only_enforce_if(upper)
        model.add(value > maximum).only_enforce_if(upper.Not())
        return cls._and(model, (lower, upper), f"{name}_range")

    @classmethod
    def _lex_bound_literal(
        cls,
        model: Any,
        digits_lsf: Sequence[Any],
        bound: int,
        *,
        greater: bool,
        name: str,
    ) -> Any:
        base = 1_000_000
        bound_digits: list[int] = []
        remaining = int(bound)
        for _ in digits_lsf:
            remaining, digit = divmod(remaining, base)
            bound_digits.append(digit)
        if remaining:
            return model.new_constant(int(not greater))
        prefix_equal = model.new_constant(1)
        winning: list[Any] = []
        for index in range(len(digits_lsf) - 1, -1, -1):
            digit = digits_lsf[index]
            target = bound_digits[index]
            equal = model.new_bool_var(f"{name}_equal_{index}")
            better = model.new_bool_var(f"{name}_better_{index}")
            model.add(digit == target).only_enforce_if(equal)
            model.add(digit != target).only_enforce_if(equal.Not())
            if greater:
                model.add(digit > target).only_enforce_if(better)
                model.add(digit <= target).only_enforce_if(better.Not())
            else:
                model.add(digit < target).only_enforce_if(better)
                model.add(digit >= target).only_enforce_if(better.Not())
            winning.append(
                cls._and(model, (prefix_equal, better), f"{name}_wins_{index}")
            )
            prefix_equal = cls._and(
                model, (prefix_equal, equal), f"{name}_prefix_{index}"
            )
        return cls._or(model, (*winning, prefix_equal), f"{name}_result")

    @classmethod
    def _decimal_range_literal(
        cls,
        model: Any,
        value: str,
        minimum: int,
        maximum: int,
        name: str,
    ) -> Any:
        if minimum > maximum:
            raise ValueError("predicate minimum cannot exceed maximum")
        integer = int(value)
        digits = []
        while integer or not digits:
            integer, digit = divmod(integer, 1_000_000)
            digits.append(model.new_constant(digit))
        lower = cls._lex_bound_literal(
            model, digits, minimum, greater=True, name=f"{name}_minimum"
        )
        upper = cls._lex_bound_literal(
            model, digits, maximum, greater=False, name=f"{name}_maximum"
        )
        return cls._and(model, (lower, upper), f"{name}_range")

    def _lower_native(
        self,
        predicate: ScenarioPredicateV1,
        context: ProofTacticPredicateContextV1,
    ) -> tuple[Any, dict[str, Any]]:
        complete = bool(getattr(context, "native_proof_bindings", False)) or (
            context.proof.completion_status in _COMPLETE_PROOF_STATUSES
            and context.proof.solver_status == "optimal"
        )
        if not complete:
            raise ValueError(
                "proof predicate evidence is unavailable because proof is incomplete"
            )
        model = context.model
        live = bool(getattr(context, "native_proof_bindings", False))
        arguments = dict(predicate.arguments)
        name = "proof_" + predicate.sha256[:12]
        minimum = int(arguments.get("minimum", 0))
        maximum = int(arguments.get("maximum", minimum))
        if predicate.kind == "proof.winning_root_action_count":
            root_actions = tuple(getattr(context, "root_actions", ()))
            values = (tuple(getattr(context, "root_winning_variables", ())) if live
                      else [item.winning for item in root_actions])
            if len(values) != len(context.proof.root_actions) or any(
                item is None for item in values
            ):
                raise ValueError("native root winning bindings are unavailable")
            count = model.new_int_var(0, len(values), f"{name}_count")
            model.add(count == sum(values))
            return self._range_literal(model, count, minimum, maximum, name), {
                "proofComplete": True,
                "winningCount": (
                    None if live else context.proof.winning_root_action_count
                ),
                "proofHash": context.proof.sha256,
                "_solverInts": {"winningCount": count},
            }
        if predicate.kind == "proof.winning_root_card_rank_count":
            ranks = frozenset(str(item) for item in arguments["rank"])
            matches = [] if live else [
                row
                for row in context.proof.root_actions
                if row.satisfies
                and LegalPlayActionV1.from_code(row.action_code).card.rank.name.lower()
                in ranks
            ]
            root_actions = tuple(getattr(context, "root_actions", ()))
            if len(root_actions) != len(context.proof.root_actions) or any(
                item.winning is None for item in root_actions
            ):
                raise ValueError("native root action bindings are unavailable")
            allowed = [
                (card.id,)
                for card in (
                    LegalPlayActionV1.from_code(row.action_code).card
                    for row in context.proof.root_actions
                )
                if card.rank.name.lower() in ranks
            ]
            values = []
            for index, action in enumerate(root_actions):
                rank_ok = model.new_bool_var(f"{name}_rank_{index}")
                model.add_allowed_assignments([action.card], allowed).only_enforce_if(
                    rank_ok
                )
                model.add_forbidden_assignments([action.card], allowed).only_enforce_if(
                    rank_ok.Not()
                )
                values.append(
                    self._and(
                        model,
                        (action.winning, rank_ok),
                        f"{name}_winning_rank_{index}",
                    )
                )
            count = model.new_int_var(0, len(values), f"{name}_count")
            model.add(count == sum(values))
            return self._range_literal(model, count, minimum, maximum, name), {
                "proofComplete": True,
                "ranks": sorted(ranks),
                "matchingCount": len(matches),
                **({} if live else {"matchingActions": [row.to_dict() for row in matches]}),
                "proofHash": context.proof.sha256,
                "_solverInts": {"matchingCount": count},
            }
        if predicate.kind == "proof.winning_continuation_line_count":
            if live:
                digits = tuple(getattr(context, "continuation_line_digits", ()))
                requested = str(arguments["lineUniverse"])
                if not digits or getattr(context, "continuation_line_universe", None) != requested:
                    raise ValueError("native continuation-line count bindings are unavailable")
                lower = self._lex_bound_literal(
                    model, digits, minimum, greater=True, name=f"{name}_minimum"
                )
                upper = self._lex_bound_literal(
                    model, digits, maximum, greater=False, name=f"{name}_maximum"
                )
                return self._and(model, (lower, upper), f"{name}_range"), {
                    "proofComplete": True,
                    "lineUniverse": requested,
                    "proofHash": context.proof.sha256,
                    "_solverDecimalDigits": {"exactCount": digits},
                }
            lines = context.proof.continuation_lines
            requested = str(arguments["lineUniverse"])
            if (
                lines is None
                or not lines.complete
                or lines.line_universe != requested
                or lines.exact_count is None
                or not lines.exact_count.isdigit()
            ):
                raise ValueError("complete continuation-line evidence is unavailable")
            return self._decimal_range_literal(
                model, lines.exact_count, minimum, maximum, name
            ), {
                "proofComplete": True,
                "lineUniverse": requested,
                "exactCount": lines.exact_count,
                "continuationLines": lines.to_dict(),
                "proofHash": context.proof.sha256,
            }
        if predicate.kind == "proof.principal_variation_trick_winner":
            if live:
                indexes = tuple(
                    getattr(context, "principal_trick_indexes", ())
                )
                requested = int(arguments["trickIndex"])
                if requested not in indexes:
                    raise ValueError(
                        "requested native principal-variation trick is unavailable"
                    )
                index = indexes.index(requested)
                winners = tuple(
                    getattr(context, "principal_trick_winner_seat_variables", ())
                )
                cards = tuple(
                    getattr(context, "principal_trick_winning_card_variables", ())
                )
                if len(winners) != len(indexes) or len(cards) != len(indexes):
                    raise ValueError("native principal winner bindings are unavailable")
                team_ok = model.new_bool_var(f"{name}_team")
                team_value = int(Team[str(arguments["team"]).upper()])
                model.add_modulo_equality(team_value, winners[index], 2).only_enforce_if(
                    team_ok
                )
                model.add_modulo_equality(1 - team_value, winners[index], 2).only_enforce_if(
                    team_ok.Not()
                )
                rank_ok = model.new_bool_var(f"{name}_rank")
                allowed_ids = [
                    (card.id,)
                    for card in FULL_DECK
                    if card.rank.name.lower() in set(arguments["rank"])
                ]
                model.add_allowed_assignments([cards[index]], allowed_ids).only_enforce_if(
                    rank_ok
                )
                model.add_forbidden_assignments([cards[index]], allowed_ids).only_enforce_if(
                    rank_ok.Not()
                )
                return self._and(model, (team_ok, rank_ok), name), {
                    "proofComplete": True,
                    "trickIndex": requested,
                    "proofHash": context.proof.sha256,
                }
            trick = self._trick(arguments, context)
            if trick is None:
                raise ValueError("requested principal-variation trick is unavailable")
            winner = trick.winner
            action = LegalPlayActionV1.from_code(winner.action_code)
            team = Seat.parse(winner.actor_seat).team
            team_var = model.new_constant(int(team))
            rank_var = model.new_constant(int(action.card.rank))
            team_ok = model.new_bool_var(f"{name}_team")
            model.add(
                team_var == int(Team[str(arguments["team"]).upper()])
            ).only_enforce_if(team_ok)
            model.add(
                team_var != int(Team[str(arguments["team"]).upper()])
            ).only_enforce_if(team_ok.Not())
            rank_ok = model.new_bool_var(f"{name}_rank")
            allowed = [(int(Rank[str(item).upper()]),) for item in arguments["rank"]]
            model.add_allowed_assignments([rank_var], allowed).only_enforce_if(rank_ok)
            model.add_forbidden_assignments([rank_var], allowed).only_enforce_if(
                rank_ok.Not()
            )
            return self._and(model, (team_ok, rank_ok), name), {
                "proofComplete": True,
                "trick": trick.to_dict(),
                "winner": winner.to_dict(),
                "proofHash": context.proof.sha256,
            }
        if predicate.kind == "proof.principal_variation_trick_team_rank_count":
            if live:
                trick_indexes = tuple(
                    getattr(context, "principal_action_trick_indexes", ())
                )
                cards = tuple(
                    getattr(context, "principal_action_card_variables", ())
                )
                actors = tuple(
                    getattr(context, "principal_action_actor_variables", ())
                )
                present = tuple(
                    getattr(context, "principal_action_present_variables", ())
                )
                if not (
                    len(trick_indexes) == len(cards) == len(actors) == len(present)
                ):
                    raise ValueError("native principal action bindings are unavailable")
                requested = int(arguments["trickIndex"])
                allowed_ids = [
                    (card.id,)
                    for card in FULL_DECK
                    if card.rank.name.lower() in set(arguments["rank"])
                ]
                team_value = int(Team[str(arguments["team"]).upper()])
                matches = []
                for index, trick_index in enumerate(trick_indexes):
                    if trick_index != requested:
                        continue
                    team_ok = model.new_bool_var(f"{name}_team_{index}")
                    model.add_modulo_equality(team_value, actors[index], 2).only_enforce_if(
                        team_ok
                    )
                    model.add_modulo_equality(1 - team_value, actors[index], 2).only_enforce_if(
                        team_ok.Not()
                    )
                    rank_ok = model.new_bool_var(f"{name}_rank_{index}")
                    model.add_allowed_assignments([cards[index]], allowed_ids).only_enforce_if(
                        rank_ok
                    )
                    model.add_forbidden_assignments([cards[index]], allowed_ids).only_enforce_if(
                        rank_ok.Not()
                    )
                    matches.append(
                        self._and(
                            model,
                            (present[index], team_ok, rank_ok),
                            f"{name}_match_{index}",
                        )
                    )
                if not matches:
                    raise ValueError(
                        "requested native principal-variation trick is unavailable"
                    )
                count = model.new_int_var(0, len(matches), f"{name}_count")
                model.add(count == sum(matches))
                return self._range_literal(model, count, minimum, maximum, name), {
                    "proofComplete": True,
                    "trickIndex": requested,
                    "proofHash": context.proof.sha256,
                    "_solverInts": {"matchingCount": count},
                }
            trick = self._trick(arguments, context)
            if trick is None:
                raise ValueError("requested principal-variation trick is unavailable")
            ranks = set(arguments["rank"])
            team_name = str(arguments["team"])
            matches = [
                play
                for play in trick.plays
                if Seat.parse(play.actor_seat).team.name.lower() == team_name
                and LegalPlayActionV1.from_code(play.action_code).card.rank.name.lower()
                in ranks
            ]
            values = [
                model.new_constant(
                    int(
                        Seat.parse(play.actor_seat).team.name.lower() == team_name
                        and LegalPlayActionV1.from_code(
                            play.action_code
                        ).card.rank.name.lower()
                        in ranks
                    )
                )
                for play in trick.plays
            ]
            count = model.new_int_var(0, len(values), f"{name}_count")
            model.add(count == sum(values))
            return self._range_literal(model, count, minimum, maximum, name), {
                "proofComplete": True,
                "trickIndex": trick.trick_index,
                "matchingCount": len(matches),
                "matchingPlays": [item.to_dict() for item in matches],
                "proofHash": context.proof.sha256,
            }
        if predicate.kind in {"tactic.verified", "tactic.applicable"}:
            tactic = str(arguments["kind"])
            prerequisite = TACTICAL_DETECTOR_PREREQUISITES_V1.get(tactic)
            if prerequisite is None:
                raise ValueError(f"native tactical detector is unavailable: {tactic}")
            minimum_roots = dict(prerequisite.required_proof_bindings)["root_action_graph"]
            native_authority = getattr(context, "native_cp_sat_authority", live)
            native_project = native_authority and tactic == TacticalKind.PROJECT_BIDDING.value
            if native_project and not live:
                raise ValueError("native project consequence requires owned live proof bindings")
            if live and tactic == TacticalKind.PROJECT_BIDDING.value:
                consequences = tuple(getattr(context, "root_project_consequence_variables", ()))
                if (getattr(context, "decision_kind", None) != "project"
                        or getattr(context, "proof_utility", None) != "awarded_point_margin"
                        or (len(context.proof.root_actions) >= minimum_roots and (
                            len(consequences) != len(context.proof.root_actions)
                            or not consequences or any(item is None for item in consequences)))):
                    raise ValueError("native project-consequence proof bindings are unavailable")
            actor_stopping = (live and tactic == TacticalKind.KABOUT_PREVENTION.value
                and getattr(context.proof, "proof_engine_id", None) == "ortools_actor_information_set_cegis_v1")
            native_trump = live and native_authority
            native_partner = native_authority and tactic == TacticalKind.PARTNER_ENABLEMENT.value
            native_entry = native_authority and tactic == TacticalKind.ENTRY_UNBLOCKING.value
            if (native_authority and tactic in {TacticalKind.TRUMP_CONTROL.value, TacticalKind.PARTNER_ENABLEMENT.value, TacticalKind.ENTRY_UNBLOCKING.value}
                    and len(context.proof.root_actions) >= minimum_roots
                    and (not live or getattr(context, "causal_gain_terminal_awarded_scope", None) is not True)):
                raise ValueError(f"native {tactic} requires exact complete-state terminal awarded-point proof")
            native_small = native_authority and tactic == TacticalKind.SMALL_CARD_BIG_GAIN.value
            native_delayed = native_authority and tactic in DELAYED_PAYOFF_TACTIC_IDS
            delayed_profile = getattr(context, "tactical_detector_profile", None)
            paired_delayed = (native_delayed and live
                and len(context.proof.root_actions) >= minimum_roots
                and isinstance(delayed_profile, TacticalDetectorProfileV1)
                and delayed_profile.native_delayed_gain_metric is not None
                and getattr(context, "causal_gain_terminal_awarded_scope", None) is True)
            root_actions = tuple(getattr(context, "root_actions", ()))
            root_winning = tuple(
                getattr(context, "root_winning_variables", ())
            ) or tuple(item.winning for item in root_actions)
            root_values = tuple(getattr(context, "root_value_variables", ())) or tuple(
                item.value for item in root_actions
            )
            selected_index = getattr(context, "selected_root_action_index", None)
            strengths = tuple(getattr(context, "card_strength_by_id", ()))
            if (
                len(root_actions) != len(context.proof.root_actions)
                or len(root_winning) != len(root_actions)
                or len(root_values) != len(root_actions)
                or not root_actions
                or (not live and len(root_actions) < minimum_roots)
                or any(item is None for item in (*root_winning, *root_values))
            ):
                raise ValueError("native tactical root bindings are unavailable")
            if (
                type(selected_index) is not int
                or not 0 <= selected_index < len(root_actions)
                or context.proof.selected_action is None
                or context.proof.root_actions[selected_index].action_code
                != context.proof.selected_action
            ):
                raise ValueError("native selected-root binding is unavailable")
            else:
                if getattr(context, "decision_kind", None) not in (
                    prerequisite.decision_kinds
                ):
                    if tactic == TacticalKind.PROJECT_BIDDING.value:
                        raise ValueError(
                            "project_bidding requires a bid or project decision"
                        )
                    raise ValueError(
                        f"{tactic} requires one of the declared decision kinds"
                    )
                if prerequisite.required_utility is not None and getattr(
                    context, "proof_utility", None
                ) != prerequisite.required_utility:
                    if tactic == TacticalKind.CONTRACT_RESCUE.value:
                        raise ValueError(
                            "contract_rescue requires contract_success utility"
                        )
                    raise ValueError("Kabout tactics require kabout utility")
                selected = root_actions[selected_index]
                alternatives = tuple(
                    item
                    for index, item in enumerate(root_actions)
                    if index != selected_index
                )
                winning_count = model.new_int_var(
                    0, len(root_winning), f"{name}_winning_count"
                )
                model.add(winning_count == sum(root_winning))

                actor = selected.actor
                actor_team = model.new_int_var(0, 1, f"{name}_actor_team")
                model.add_modulo_equality(actor_team, actor, 2)
                actor_is_target = getattr(
                    context, "root_actor_is_target_variable", None
                )
                if actor_is_target is None:
                    raise ValueError(
                        "native tactical root-target binding is unavailable"
                    )
                context_filters = []
                for field in ("actingSeat", "observer"):
                    if arguments.get(field) is not None:
                        matches = model.new_bool_var(f"{name}_{field}")
                        expected = int(Seat.parse(arguments[field]))
                        model.add(actor == expected).only_enforce_if(matches)
                        model.add(actor != expected).only_enforce_if(matches.Not())
                        context_filters.append(matches)
                if arguments.get("targetTeam") is not None:
                    target_team = model.new_int_var(0, 1, f"{name}_target_team_value")
                    model.add(target_team == actor_team).only_enforce_if(
                        actor_is_target
                    )
                    model.add(target_team + actor_team == 1).only_enforce_if(
                        actor_is_target.Not()
                    )
                    matches = model.new_bool_var(f"{name}_target_team")
                    expected = int(Team[str(arguments["targetTeam"]).upper()])
                    model.add(target_team == expected).only_enforce_if(matches)
                    model.add(target_team != expected).only_enforce_if(matches.Not())
                    context_filters.append(matches)
                if arguments.get("targetSeat") is not None:
                    target_seat = getattr(context, "target_seat_variable", None)
                    if target_seat is None:
                        raise ValueError(
                            "native tactical target-seat binding is unavailable"
                        )
                    matches = model.new_bool_var(f"{name}_target_seat")
                    expected = int(Seat.parse(arguments["targetSeat"]))
                    model.add(target_seat == expected).only_enforce_if(matches)
                    model.add(target_seat != expected).only_enforce_if(matches.Not())
                    context_filters.append(matches)
                context_matches = self._and(model, context_filters, f"{name}_context")

                applicable = model.new_bool_var(f"{name}_root_available")
                model.add(applicable == 1)
                verified = model.new_bool_var(f"{name}_verified")
                strictly_better_values = []
                not_worse_values = []
                tied_values = []
                for index, alternative in enumerate(root_values):
                    if index == selected_index:
                        continue
                    strict = model.new_bool_var(f"{name}_strict_{index}")
                    not_worse = model.new_bool_var(f"{name}_not_worse_{index}")
                    tied = model.new_bool_var(f"{name}_tied_{index}")
                    greater = model.new_bool_var(f"{name}_greater_{index}")
                    lower = model.new_bool_var(f"{name}_lower_{index}")
                    at_least = model.new_bool_var(f"{name}_at_least_{index}")
                    at_most = model.new_bool_var(f"{name}_at_most_{index}")
                    model.add(root_values[selected_index] > alternative).only_enforce_if(
                        greater
                    )
                    model.add(root_values[selected_index] <= alternative).only_enforce_if(
                        greater.Not()
                    )
                    model.add(root_values[selected_index] < alternative).only_enforce_if(
                        lower
                    )
                    model.add(root_values[selected_index] >= alternative).only_enforce_if(
                        lower.Not()
                    )
                    model.add(root_values[selected_index] >= alternative).only_enforce_if(
                        at_least
                    )
                    model.add(root_values[selected_index] < alternative).only_enforce_if(
                        at_least.Not()
                    )
                    model.add(root_values[selected_index] <= alternative).only_enforce_if(
                        at_most
                    )
                    model.add(root_values[selected_index] > alternative).only_enforce_if(
                        at_most.Not()
                    )
                    strict_target = self._and(
                        model, (actor_is_target, greater), f"{name}_strict_target_{index}"
                    )
                    strict_opponent = self._and(
                        model,
                        (actor_is_target.Not(), lower),
                        f"{name}_strict_opponent_{index}",
                    )
                    model.add(
                        strict
                        == self._or(
                            model,
                            (strict_target, strict_opponent),
                            f"{name}_strict_direction_{index}",
                        )
                    )
                    best_target = self._and(
                        model, (actor_is_target, at_least), f"{name}_best_target_{index}"
                    )
                    best_opponent = self._and(
                        model,
                        (actor_is_target.Not(), at_most),
                        f"{name}_best_opponent_{index}",
                    )
                    model.add(
                        not_worse
                        == self._or(
                            model,
                            (best_target, best_opponent),
                            f"{name}_best_direction_{index}",
                        )
                    )
                    model.add(
                        root_values[selected_index] == alternative
                    ).only_enforce_if(tied)
                    model.add(
                        root_values[selected_index] != alternative
                    ).only_enforce_if(tied.Not())
                    strictly_better_values.append(strict)
                    not_worse_values.append(not_worse)
                    tied_values.append(tied)
                strategically_better = self._and(
                    model, strictly_better_values, f"{name}_strategically_better"
                )
                strategically_best = self._and(
                    model, not_worse_values, f"{name}_strategically_best"
                )
                has_tied_alternative = self._or(
                    model, tied_values, f"{name}_tied_alternative"
                )
                has_causal_advantage = self._or(
                    model, strictly_better_values, f"{name}_causal_advantage"
                )
                evidence_values = list(strictly_better_values)
                trump_extraction = None
                card_strength_bindings = None
                project_threshold = None
                trump_threshold = None
                paired_threshold = None
                paired_comparator = None
                paired_winners = ()
                entry_variables = ()
                payoff_depth = None
                mechanism_flags = {}
                delayed_gain = delayed_raw_gain = delayed_concession = None
                delayed_threshold = None
                if len(root_actions) < minimum_roots:
                    root_count = model.new_int_var(len(root_actions), len(root_actions), f"{name}_root_count")
                    applicable = model.new_bool_var(f"{name}_required_roots_present")
                    model.add(root_count >= minimum_roots).only_enforce_if(applicable)
                    model.add(root_count < minimum_roots).only_enforce_if(applicable.Not())
                    model.add(verified == applicable)
                    evidence_values = []
                elif tactic == TacticalKind.ONLY_WINNING_MOVE.value:
                    alternative_losses = tuple(
                        item.Not()
                        for index, item in enumerate(root_winning)
                        if index != selected_index
                    )
                    evidence_values = list(alternative_losses)
                    applicable = root_winning[selected_index]
                    sole_selected_winner = self._and(
                        model,
                        (
                            applicable,
                            self._and(
                                model,
                                alternative_losses,
                                f"{name}_all_alternatives_lose",
                            ),
                        ),
                        f"{name}_sole_selected_winner",
                    )
                    model.add(verified == sole_selected_winner)
                elif tactic == TacticalKind.CONTRACT_RESCUE.value:
                    alternative_losses = tuple(
                        item.Not()
                        for index, item in enumerate(root_winning)
                        if index != selected_index
                    )
                    evidence_values = list(alternative_losses)
                    applicable = root_winning[selected_index]
                    alternative_loses = self._or(
                        model,
                        alternative_losses,
                        f"{name}_alternative_loses",
                    )
                    native_verified = self._and(
                        model,
                        (
                            applicable,
                            alternative_loses,
                        ),
                        f"{name}_verified_conditions",
                    )
                    model.add(verified == native_verified)
                elif native_partner or native_entry or paired_delayed:
                    paired_entry = native_entry or paired_delayed
                    profile = getattr(context, "tactical_detector_profile", None)
                    scale = getattr(context, "causal_gain_scale", None)
                    rows = tuple(getattr(context, "root_raw_payoff_variables", ()))
                    if not isinstance(profile, TacticalDetectorProfileV1):
                        raise ValueError(f"native {tactic} requires a frozen tactical detector profile")
                    profile.validate()
                    if type(scale) is not int or scale < 1:
                        raise ValueError(f"native {tactic} margin scale is unavailable")
                    if len(rows) != len(root_actions):
                        raise ValueError(f"native {tactic} requires paired root winner bindings")
                    keys = tuple(sorted(key for key in rows[selected_index] if key.endswith("WinnerSeat")))
                    if not keys or any(tuple(sorted(key for key in row if key.endswith("WinnerSeat"))) != keys for row in rows):
                        raise ValueError(f"native {tactic} requires matching paired tricks")
                    paired_winners = tuple({key: row[key] for key in keys} for row in rows)
                    if paired_entry:
                        entry_keys = tuple(key.removesuffix("WinnerSeat") + suffix for key in keys
                                           for suffix in ("LeaderSeat", "LeadCard", "CardRawMargin"))
                        if any(not set(entry_keys).issubset(row) for row in rows):
                            raise ValueError("native entry requires paired leader, lead-card and card-point bindings")
                        entry_variables = tuple({key: row[key] for key in entry_keys} for row in rows)
                        payoff_depth = model.new_int_var(1, len(keys), f"{name}_payoff_depth")
                        selected_suit = model.new_int_var(0, 3, f"{name}_selected_suit")
                        model.add_division_equality(selected_suit, selected.card, 8)
                    flag_witnesses = {key: [] for key in ("entry", "forcedReturn", "partnerEnablement")}
                    if paired_delayed:
                        control_keys = tuple(key.removesuffix("WinnerSeat") + suffix for key in keys
                            for suffix in ("WinningCard", "OpponentTrumpsRemaining"))
                        if any(not set(control_keys).issubset(row) for row in rows):
                            raise ValueError("native delayed payoff requires paired winning-card and trump-count bindings")
                        card_boundary, card_strength_bindings, cutoff = self._delayed_strength_boundary(
                            model, selected.card, strengths, profile, native_small, name)
                        flag_witnesses.update(promotion=[], trumpExhaustion=[])
                        raw_bound = 2 * sum(max(abs(value) for row in rows for value in row[
                            key.removesuffix("WinnerSeat") + "CardRawMargin"].proto.domain) for key in keys)
                        awarded_bound = max(bound for value in root_values for bound in value.proto.domain) - min(
                            bound for value in root_values for bound in value.proto.domain)
                        metric_scale = 1 if profile.native_delayed_gain_metric == "raw_card_points" else scale
                        delayed_threshold = max(1, ceil(Fraction(str(profile.native_delayed_minimum_gain)) * metric_scale))
                        gain_bound = raw_bound if metric_scale == 1 and profile.native_delayed_gain_metric == "raw_card_points" else awarded_bound
                        delayed_gain = model.new_int_var(-gain_bound, gain_bound, f"{name}_native_delayed_gain")
                        delayed_raw_gain = model.new_int_var(-raw_bound, raw_bound, f"{name}_later_card_margin_gain")
                        delayed_concession = model.new_int_var(-raw_bound, raw_bound, f"{name}_first_trick_concession")
                    partner = model.new_int_var(0, 3, f"{name}_partner")
                    model.add_allowed_assignments([actor, partner], ((0, 2), (1, 3), (2, 0), (3, 1)))
                    paired_threshold = max(1, ceil(Fraction(str(profile.minimum_causal_gain)) * scale))
                    gains, best_choices, mechanisms = [], [], []
                    for index, alternative in enumerate(root_values):
                        if index == selected_index:
                            continue
                        gain = model.new_bool_var(f"{name}_configured_gain_{index}")
                        difference = root_values[selected_index] - alternative
                        model.add(difference >= paired_threshold).only_enforce_if([actor_is_target, gain])
                        model.add(difference < paired_threshold).only_enforce_if([actor_is_target, gain.Not()])
                        model.add(difference <= -paired_threshold).only_enforce_if([actor_is_target.Not(), gain])
                        model.add(difference > -paired_threshold).only_enforce_if([actor_is_target.Not(), gain.Not()])
                        gains.append(gain)
                        comparisons = []
                        for other_index, other in enumerate(root_values):
                            if other_index in {selected_index, index}:
                                continue
                            best = model.new_bool_var(f"{name}_alternative_{index}_beats_{other_index}")
                            # Stable root order breaks equal-valued alternative ties.
                            gap = int(other_index < index)
                            model.add(alternative >= other + gap).only_enforce_if([actor_is_target, best])
                            model.add(alternative < other + gap).only_enforce_if([actor_is_target, best.Not()])
                            model.add(alternative <= other - gap).only_enforce_if([actor_is_target.Not(), best])
                            model.add(alternative > other - gap).only_enforce_if([actor_is_target.Not(), best.Not()])
                            comparisons.append(best)
                        chosen = self._and(model, comparisons, f"{name}_best_alternative_{index}")
                        best_choices.append((index, chosen))
                        changes, entries, returns, promotions, exhaustions = [], [], [], [], []
                        if paired_entry:
                            paid = self._or(model, (), f"{name}_no_payoff_{index}")
                            cumulative = 0
                            depth = model.new_int_var(1, len(keys), f"{name}_payoff_depth_{index}")
                        for ordinal, key in enumerate(keys):
                            selected_partner = model.new_bool_var(f"{name}_{key}_selected_partner_{index}")
                            other_not_partner = model.new_bool_var(f"{name}_{key}_alternative_not_partner_{index}")
                            model.add(rows[selected_index][key] == partner).only_enforce_if(selected_partner)
                            model.add(rows[selected_index][key] != partner).only_enforce_if(selected_partner.Not())
                            model.add(rows[index][key] != partner).only_enforce_if(other_not_partner)
                            model.add(rows[index][key] == partner).only_enforce_if(other_not_partner.Not())
                            if not paired_delayed or ordinal:
                                changes.append(self._and(model, (selected_partner, other_not_partner), f"{name}_{key}_changed_{index}"))
                            if paired_entry:
                                label = key.removesuffix("WinnerSeat")
                                cumulative += rows[selected_index][label + "CardRawMargin"] - rows[index][label + "CardRawMargin"]
                                positive = model.new_bool_var(f"{name}_{label}_positive_{index}")
                                model.add(cumulative > 0).only_enforce_if(positive)
                                model.add(cumulative <= 0).only_enforce_if(positive.Not())
                                first_payoff = self._and(model, (paid.Not(), positive), f"{name}_{label}_first_payoff_{index}")
                                model.add(depth == ordinal + 1).only_enforce_if(first_payoff)
                                paid = self._or(model, (paid, positive), f"{name}_{label}_paid_{index}")
                                team_wins = []
                                for root_index in (selected_index, index):
                                    team = model.new_int_var(0, 1, f"{name}_{label}_team_{root_index}_{index}")
                                    model.add_modulo_equality(team, rows[root_index][key], 2)
                                    won = model.new_bool_var(f"{name}_{label}_team_won_{root_index}_{index}")
                                    model.add(team == actor_team).only_enforce_if(won)
                                    model.add(team != actor_team).only_enforce_if(won.Not())
                                    team_wins.append(won)
                                if not paired_delayed or ordinal:
                                    entries.append(self._and(model, (team_wins[0], team_wins[1].Not(),
                                        paid if ordinal < len(keys) - 1 else self._and(model, (), f"{name}_last_payoff_{index}")),
                                        f"{name}_{label}_entry_{index}"))
                                if ordinal:
                                    leader_team = model.new_int_var(0, 1, f"{name}_{label}_leader_team_{index}")
                                    model.add_modulo_equality(leader_team, rows[selected_index][label + "LeaderSeat"], 2)
                                    opponent_lead = model.new_bool_var(f"{name}_{label}_opponent_lead_{index}")
                                    model.add(leader_team != actor_team).only_enforce_if(opponent_lead)
                                    model.add(leader_team == actor_team).only_enforce_if(opponent_lead.Not())
                                    suit_matches = []
                                    for root_index in (selected_index, index):
                                        suit = model.new_int_var(0, 3, f"{name}_{label}_lead_suit_{root_index}_{index}")
                                        model.add_division_equality(suit, rows[root_index][label + "LeadCard"], 8)
                                        matches = model.new_bool_var(f"{name}_{label}_return_suit_{root_index}_{index}")
                                        model.add(suit == selected_suit).only_enforce_if(matches)
                                        model.add(suit != selected_suit).only_enforce_if(matches.Not())
                                        suit_matches.append(matches)
                                    different_return = self._or(model, (suit_matches[1].Not(), team_wins[1].Not()),
                                        f"{name}_{label}_different_return_{index}")
                                    returns.append(self._and(model, (opponent_lead, suit_matches[0], team_wins[0], different_return),
                                        f"{name}_{label}_forced_return_{index}"))
                                    if paired_delayed and native_small:
                                        winning_suits = []
                                        for root_index in (selected_index, index):
                                            suit = model.new_int_var(0, 3, f"{name}_{label}_winning_suit_{root_index}_{index}")
                                            model.add_division_equality(suit, rows[root_index][label + "WinningCard"], 8)
                                            same_suit = model.new_bool_var(f"{name}_{label}_winning_root_suit_{root_index}_{index}")
                                            model.add(suit == selected_suit).only_enforce_if(same_suit)
                                            model.add(suit != selected_suit).only_enforce_if(same_suit.Not())
                                            winning_suits.append(same_suit)
                                        low_winner, _, _ = self._delayed_strength_boundary(model,
                                            rows[selected_index][label + "WinningCard"], strengths, profile, True,
                                            f"{name}_{label}_promotion_{index}")
                                        promotions.append(self._and(model, (team_wins[0], winning_suits[0], low_winner,
                                            self._or(model, (team_wins[1].Not(), winning_suits[1].Not()),
                                                f"{name}_{label}_different_promoted_control_{index}")),
                                            f"{name}_{label}_promotion_{index}"))
                                if paired_delayed:
                                    exhausted = model.new_bool_var(f"{name}_{label}_exhausted_{index}")
                                    model.add(rows[selected_index][label + "OpponentTrumpsRemaining"] == 0).only_enforce_if(exhausted)
                                    model.add(rows[selected_index][label + "OpponentTrumpsRemaining"] > 0).only_enforce_if(exhausted.Not())
                                    other_remaining = model.new_bool_var(f"{name}_{label}_other_trumps_{index}")
                                    model.add(rows[index][label + "OpponentTrumpsRemaining"] > 0).only_enforce_if(other_remaining)
                                    model.add(rows[index][label + "OpponentTrumpsRemaining"] == 0).only_enforce_if(other_remaining.Not())
                                    exhaustions.append(self._and(model, (exhausted, other_remaining),
                                        f"{name}_{label}_trump_exhaustion_{index}"))
                        if paired_entry:
                            model.add(depth == len(keys)).only_enforce_if(paid.Not())
                            model.add(payoff_depth == depth).only_enforce_if(chosen)
                            for flag, witnesses in (("entry", entries), ("forcedReturn", returns), ("partnerEnablement", changes),
                                    *((("promotion", promotions), ("trumpExhaustion", exhaustions)) if paired_delayed else ())):
                                flag_witnesses[flag].append(self._and(model, (chosen, self._or(model, witnesses,
                                    f"{name}_{flag}_{index}")), f"{name}_selected_{flag}_{index}"))
                        if paired_delayed:
                            labels = tuple(key.removesuffix("WinnerSeat") for key in keys)
                            raw_gain = sum(rows[selected_index][label + "CardRawMargin"] - rows[index][label + "CardRawMargin"]
                                for label in labels)
                            first_gain = rows[selected_index][labels[0] + "CardRawMargin"] - rows[index][labels[0] + "CardRawMargin"]
                            model.add(delayed_raw_gain == raw_gain - first_gain).only_enforce_if(chosen)
                            model.add(delayed_concession == -first_gain).only_enforce_if(chosen)
                            if profile.native_delayed_gain_metric == "raw_card_points":
                                model.add(delayed_gain == raw_gain).only_enforce_if(chosen)
                            else:
                                model.add(delayed_gain == difference).only_enforce_if([chosen, actor_is_target])
                                model.add(delayed_gain == -difference).only_enforce_if([chosen, actor_is_target.Not()])
                        mechanisms.append(self._and(model, (chosen, self._or(model, changes + entries + returns + promotions + exhaustions,
                            f"{name}_partner_change_{index}")), f"{name}_paired_mechanism_{index}"))
                    if paired_entry:
                        mechanism_flags = {key: self._or(model, witnesses, f"{name}_{key}") for key, witnesses in flag_witnesses.items()}
                    paired_comparator = model.new_int_var(0, len(root_actions) - 1, f"{name}_best_alternative")
                    model.add(sum(chosen for _, chosen in best_choices) == 1)
                    model.add(paired_comparator == sum(index * chosen for index, chosen in best_choices))
                    applicable = self._or(model, mechanisms, f"{name}_paired_partner_entry")
                    if paired_delayed:
                        later_gain = model.new_bool_var(f"{name}_later_gain")
                        model.add(delayed_raw_gain > 0).only_enforce_if(later_gain)
                        model.add(delayed_raw_gain <= 0).only_enforce_if(later_gain.Not())
                        conditions = [applicable, card_boundary, later_gain]
                        if not native_small:
                            concedes = model.new_bool_var(f"{name}_concedes_first_trick")
                            model.add(delayed_concession > 0).only_enforce_if(concedes)
                            model.add(delayed_concession <= 0).only_enforce_if(concedes.Not())
                            loses = model.new_bool_var(f"{name}_loses_first_trick")
                            first_winner = context.first_principal_trick_winner_team_variable
                            model.add(actor_team != first_winner).only_enforce_if(loses)
                            model.add(actor_team == first_winner).only_enforce_if(loses.Not())
                            conditions.extend((concedes, loses))
                        applicable = self._and(model, conditions, f"{name}_delayed_mechanism")
                        configured_gain = model.new_bool_var(f"{name}_native_gain_threshold")
                        model.add(delayed_gain >= delayed_threshold).only_enforce_if(configured_gain)
                        model.add(delayed_gain < delayed_threshold).only_enforce_if(configured_gain.Not())
                        gains.append(configured_gain)
                    evidence_values = [applicable]
                    model.add(verified == self._and(model, (applicable, self._and(model, gains,
                        f"{name}_configured_gain")), f"{name}_verified_conditions"))
                elif tactic in {
                    TacticalKind.KABOUT_CREATION.value,
                    TacticalKind.KABOUT_PRESERVATION.value,
                    TacticalKind.KABOUT_PREVENTION.value,
                }:
                    base_tricks = tuple(
                        getattr(context, "base_trick_variables", ())
                    )
                    if len(base_tricks) != 2:
                        raise ValueError(
                            "native tactical base-trick bindings are unavailable"
                        )
                    actor_base = model.new_int_var(0, 8, f"{name}_actor_base")
                    opponent_base = model.new_int_var(0, 8, f"{name}_opponent_base")
                    model.add_element(actor_team, base_tricks, actor_base)
                    other_team = model.new_int_var(0, 1, f"{name}_other_team")
                    model.add(actor_team + other_team == 1)
                    model.add_element(other_team, base_tricks, opponent_base)
                    target_base = model.new_int_var(0, 8, f"{name}_target_base")
                    target_opponent_base = model.new_int_var(
                        0, 8, f"{name}_target_opponent_base"
                    )
                    model.add(target_base == actor_base).only_enforce_if(
                        actor_is_target
                    )
                    model.add(target_base == opponent_base).only_enforce_if(
                        actor_is_target.Not()
                    )
                    model.add(target_opponent_base == opponent_base).only_enforce_if(
                        actor_is_target
                    )
                    model.add(target_opponent_base == actor_base).only_enforce_if(
                        actor_is_target.Not()
                    )
                    if tactic == TacticalKind.KABOUT_CREATION.value:
                        no_base_tricks = model.new_bool_var(f"{name}_no_base_tricks")
                        model.add(sum(base_tricks) == 0).only_enforce_if(
                            no_base_tricks
                        )
                        model.add(sum(base_tricks) != 0).only_enforce_if(
                            no_base_tricks.Not()
                        )
                        applicable = self._and(
                            model,
                            (actor_is_target, no_base_tricks),
                            f"{name}_applicable",
                        )
                        alternative_failures = tuple(
                            item.Not()
                            for index, item in enumerate(root_winning)
                            if index != selected_index
                        )
                        evidence_values = list(alternative_failures)
                        alternatives_fail = self._and(
                            model,
                            alternative_failures,
                            f"{name}_alternatives_fail",
                        )
                        conditions = (
                            applicable,
                            root_winning[selected_index],
                            alternatives_fail,
                        )
                    elif tactic == TacticalKind.KABOUT_PRESERVATION.value:
                        target_started = model.new_bool_var(f"{name}_target_started")
                        opponent_empty = model.new_bool_var(f"{name}_opponent_empty")
                        model.add(target_base > 0).only_enforce_if(target_started)
                        model.add(target_base == 0).only_enforce_if(
                            target_started.Not()
                        )
                        model.add(target_opponent_base == 0).only_enforce_if(
                            opponent_empty
                        )
                        model.add(target_opponent_base != 0).only_enforce_if(
                            opponent_empty.Not()
                        )
                        applicable = self._and(
                            model,
                            (actor_is_target, target_started, opponent_empty),
                            f"{name}_applicable",
                        )
                        alternative_failures = tuple(
                            item.Not()
                            for index, item in enumerate(root_winning)
                            if index != selected_index
                        )
                        evidence_values = list(alternative_failures)
                        alternative_fails = self._or(
                            model,
                            alternative_failures,
                            f"{name}_alternative_fails",
                        )
                        conditions = (
                            applicable,
                            root_winning[selected_index],
                            alternative_fails,
                        )
                    else:
                        opposing_side_empty = model.new_bool_var(
                            f"{name}_opposing_side_empty"
                        )
                        model.add(target_opponent_base == 0).only_enforce_if(
                            opposing_side_empty
                        )
                        model.add(target_opponent_base != 0).only_enforce_if(
                            opposing_side_empty.Not()
                        )
                        applicable = self._and(
                            model,
                            (actor_is_target.Not(), opposing_side_empty),
                            f"{name}_applicable",
                        )
                        if actor_stopping:
                            stopping = tuple(getattr(context, "root_stopping_variables", ()))
                            if len(stopping) != len(root_winning):
                                raise ValueError("native opponent stopping bindings are unavailable")
                            successes = []
                            for index, value in enumerate(root_values):
                                if index == selected_index:
                                    continue
                                allows = model.new_bool_var(f"{name}_target_kabout_{index}")
                                model.add(value == 1).only_enforce_if(allows)
                                model.add(value != 1).only_enforce_if(allows.Not())
                                successes.append(allows)
                            alternative_successes = tuple(successes)
                            selected_prevents = stopping[selected_index]
                        else:
                            alternative_successes = tuple(item for index, item in enumerate(root_winning)
                                if index != selected_index)
                            selected_prevents = root_winning[selected_index].Not()
                        evidence_values = list(alternative_successes)
                        alternative_allows = self._or(
                            model,
                            alternative_successes,
                            f"{name}_alternative_allows",
                        )
                        conditions = (
                            applicable,
                            selected_prevents,
                            alternative_allows,
                        )
                    native_verified = self._and(
                        model, conditions, f"{name}_verified_conditions"
                    )
                    model.add(verified == native_verified)
                elif tactic == TacticalKind.PROJECT_BIDDING.value:
                    if live:
                        if native_project:
                            profile = getattr(context, "tactical_detector_profile", None)
                            scale = getattr(context, "project_consequence_scale", None)
                            if not isinstance(profile, TacticalDetectorProfileV1):
                                raise ValueError("native project tactic requires a frozen tactical detector profile")
                            profile.validate()
                            if type(scale) is not int or scale < 1:
                                raise ValueError("native project consequence scale is unavailable")
                            project_threshold = max(1, ceil(Fraction(str(profile.project_consequence_delta)) * scale))
                        evidence_values = []
                        for index, delta in enumerate(consequences):
                            if index == selected_index:
                                continue
                            benefit = model.new_bool_var(f"{name}_project_benefit_{index}")
                            difference = consequences[selected_index] - delta
                            minimum_gain = project_threshold if native_project else 1
                            model.add(difference >= minimum_gain).only_enforce_if([actor_is_target, benefit])
                            model.add(difference < minimum_gain).only_enforce_if([actor_is_target, benefit.Not()])
                            model.add(difference <= -minimum_gain).only_enforce_if([actor_is_target.Not(), benefit])
                            model.add(difference > -minimum_gain).only_enforce_if([actor_is_target.Not(), benefit.Not()])
                            evidence_values.append(benefit)
                        has_causal_advantage = (self._and if native_project else self._or)(
                            model, evidence_values, f"{name}_project_causal")
                        applicable = self._and(model, (strategically_better, has_causal_advantage),
                                               f"{name}_project_applicable")
                    else:
                        applicable = strategically_better
                    native_verified = self._and(
                        model,
                        (applicable,),
                        f"{name}_verified_conditions",
                    )
                    model.add(verified == native_verified)
                elif tactic == TacticalKind.DEBATE.value:
                    evidence_values = list(tied_values)
                    applicable = has_tied_alternative
                    native_verified = self._and(
                        model,
                        (applicable, strategically_best, has_tied_alternative),
                        f"{name}_verified_conditions",
                    )
                    model.add(verified == native_verified)
                elif tactic in {
                    TacticalKind.ENTRY_UNBLOCKING.value,
                    TacticalKind.PARTNER_ENABLEMENT.value,
                }:
                    winner_seats = tuple(
                        getattr(
                            context, "principal_trick_winner_seat_variables", ()
                        )
                    )
                    if len(winner_seats) < 2:
                        raise ValueError(
                            "native principal-variation winner bindings are unavailable"
                        )
                    first_winner_team = model.new_int_var(
                        0, 1, f"{name}_first_winner_team"
                    )
                    model.add_modulo_equality(
                        first_winner_team, winner_seats[0], 2
                    )
                    loses_first = model.new_bool_var(f"{name}_loses_first")
                    model.add(actor_team != first_winner_team).only_enforce_if(
                        loses_first
                    )
                    model.add(actor_team == first_winner_team).only_enforce_if(
                        loses_first.Not()
                    )
                    partner = model.new_int_var(0, 3, f"{name}_partner")
                    model.add_allowed_assignments(
                        [actor, partner],
                        ((0, 2), (1, 3), (2, 0), (3, 1)),
                    )
                    later_matches = []
                    target_actor = (
                        actor
                        if tactic == TacticalKind.ENTRY_UNBLOCKING.value
                        else partner
                    )
                    for index, winner in enumerate(winner_seats[1:]):
                        matches = model.new_bool_var(f"{name}_later_winner_{index}")
                        model.add(winner == target_actor).only_enforce_if(matches)
                        model.add(winner != target_actor).only_enforce_if(
                            matches.Not()
                        )
                        later_matches.append(matches)
                    later_entry = self._or(
                        model, later_matches, f"{name}_later_entry"
                    )
                    conditions = [later_entry]
                    if tactic == TacticalKind.ENTRY_UNBLOCKING.value:
                        conditions.append(loses_first)
                    applicable = self._and(
                        model, conditions, f"{name}_applicable"
                    )
                    native_verified = self._and(
                        model,
                        (applicable, strategically_better),
                        f"{name}_verified_conditions",
                    )
                    model.add(verified == native_verified)
                elif tactic == TacticalKind.TRUMP_CONTROL.value:
                    configured_gain = strategically_better
                    if native_trump:
                        profile = getattr(context, "tactical_detector_profile", None)
                        scale = getattr(context, "causal_gain_scale", None)
                        if not isinstance(profile, TacticalDetectorProfileV1):
                            raise ValueError("native trump control requires a frozen tactical detector profile")
                        profile.validate()
                        if type(scale) is not int or scale < 1:
                            raise ValueError("native trump control margin scale is unavailable")
                        trump_threshold = max(1, ceil(Fraction(str(profile.minimum_causal_gain)) * scale))
                        evidence_values = []
                        for index, alternative in enumerate(root_values):
                            if index == selected_index:
                                continue
                            gain = model.new_bool_var(f"{name}_configured_gain_{index}")
                            difference = root_values[selected_index] - alternative
                            model.add(difference >= trump_threshold).only_enforce_if([actor_is_target, gain])
                            model.add(difference < trump_threshold).only_enforce_if([actor_is_target, gain.Not()])
                            model.add(difference <= -trump_threshold).only_enforce_if([actor_is_target.Not(), gain])
                            model.add(difference > -trump_threshold).only_enforce_if([actor_is_target.Not(), gain.Not()])
                            evidence_values.append(gain)
                        configured_gain = self._and(model, evidence_values, f"{name}_configured_gain")
                    trump = getattr(context, "trump_suit_variable", None)
                    continuation = getattr(
                        context,
                        "principal_variation_has_continuation_variable",
                        None,
                    )
                    if trump is None or continuation is None:
                        raise ValueError(
                            "native trump/continuation bindings are unavailable"
                        )
                    selected_suit = model.new_int_var(0, 3, f"{name}_selected_suit")
                    model.add_division_equality(selected_suit, selected.card, 8)
                    is_trump = model.new_bool_var(f"{name}_is_trump")
                    model.add(selected_suit == trump).only_enforce_if(is_trump)
                    model.add(selected_suit != trump).only_enforce_if(
                        is_trump.Not()
                    )
                    applicable = self._and(
                        model,
                        (is_trump, continuation),
                        f"{name}_applicable",
                    )
                    mechanism = []
                    if native_trump:
                        suffix = tuple(getattr(context, "trump_control_suffix_actions", ()))
                        if not suffix or any(not isinstance(row, tuple) or len(row) != 4
                                or any(variable is None for variable in row[:3])
                                or type(row[3]) is not int or not 0 <= row[3] < 8 for row in suffix):
                            raise ValueError("native trump-extraction suffix bindings are unavailable")
                        first_trick = min(row[3] for row in suffix)
                        witnesses = []
                        for index, (card, seat, present, trick) in enumerate(suffix):
                            if trick != first_trick:
                                continue
                            suit = model.new_int_var(0, 3, f"{name}_extraction_suit_{index}")
                            team = model.new_int_var(0, 1, f"{name}_extraction_team_{index}")
                            model.add_division_equality(suit, card, 8)
                            model.add_modulo_equality(team, seat, 2)
                            matches = model.new_bool_var(f"{name}_extraction_trump_{index}")
                            opponent = model.new_bool_var(f"{name}_extraction_opponent_{index}")
                            model.add(suit == trump).only_enforce_if(matches)
                            model.add(suit != trump).only_enforce_if(matches.Not())
                            model.add(team != actor_team).only_enforce_if(opponent)
                            model.add(team == actor_team).only_enforce_if(opponent.Not())
                            witnesses.append(self._and(model, (present, matches, opponent),
                                f"{name}_extraction_witness_{index}"))
                        trump_extraction = self._and(model, (is_trump,
                            self._or(model, witnesses, f"{name}_opponent_trump")), f"{name}_trump_extraction")
                        mechanism.append(trump_extraction)
                    native_verified = self._and(
                        model,
                        (applicable, configured_gain, *mechanism),
                        f"{name}_verified_conditions",
                    )
                    model.add(verified == native_verified)
                elif (
                    selected is not None
                    and alternatives
                    and len(strengths) == 32
                    and all(item.value is not None for item in root_actions)
                ):
                    selected_strength = model.new_int_var(
                        min(strengths), max(strengths), f"{name}_selected_strength"
                    )
                    model.add_element(selected.card, strengths, selected_strength)
                    alternative_strengths = []
                    for index, action in enumerate(alternatives):
                        strength = model.new_int_var(
                            min(strengths),
                            max(strengths),
                            f"{name}_alt_strength_{index}",
                        )
                        model.add_element(action.card, strengths, strength)
                        alternative_strengths.append(strength)
                    relative = []
                    for index, strength in enumerate(alternative_strengths):
                        comparison = model.new_bool_var(f"{name}_relative_{index}")
                        if tactic == TacticalKind.SMALL_CARD_BIG_GAIN.value:
                            model.add(selected_strength < strength).only_enforce_if(
                                comparison
                            )
                            model.add(selected_strength >= strength).only_enforce_if(
                                comparison.Not()
                            )
                        else:
                            model.add(selected_strength > strength).only_enforce_if(
                                comparison
                            )
                            model.add(selected_strength <= strength).only_enforce_if(
                                comparison.Not()
                            )
                        relative.append(comparison)
                    relative_card = self._or(model, relative, f"{name}_relative_card")

                    continuation = getattr(
                        context,
                        "principal_variation_has_continuation_variable",
                        None,
                    )
                    if continuation is None:
                        raise ValueError(
                            "native principal-variation continuation binding is unavailable"
                        )
                    tactic_conditions = [relative_card, continuation]
                    if native_delayed:
                        profile = getattr(context, "tactical_detector_profile", None)
                        if not isinstance(profile, TacticalDetectorProfileV1):
                            raise ValueError("native delayed-payoff tactic requires a frozen tactical detector profile")
                        profile.validate()
                        boundary, card_strength_bindings, cutoff = self._delayed_strength_boundary(
                            model, selected.card, strengths, profile, native_small, name)
                        tactic_conditions.append(boundary)
                    if tactic == TacticalKind.LARGE_CARD_SACRIFICE.value:
                        first_winner = getattr(
                            context,
                            "first_principal_trick_winner_team_variable",
                            None,
                        )
                        if first_winner is None:
                            raise ValueError(
                                "native first-trick winner binding is unavailable"
                            )
                        loses = model.new_bool_var(f"{name}_loses_first_trick")
                        model.add(actor_team != first_winner).only_enforce_if(loses)
                        model.add(actor_team == first_winner).only_enforce_if(
                            loses.Not()
                        )
                        tactic_conditions.append(loses)
                    applicable = self._and(
                        model, tactic_conditions, f"{name}_applicable"
                    )
                    native_verified = self._and(
                        model,
                        (applicable, strategically_better),
                        f"{name}_verified_conditions",
                    )
                    model.add(verified == native_verified)
                else:
                    model.add(verified == 0)

                qualifying_evidence = [
                    self._and(
                        model,
                        (verified, item),
                        f"{name}_qualifying_evidence_{index}",
                    )
                    for index, item in enumerate(evidence_values)
                ]
                evidence_count = model.new_int_var(
                    0, len(qualifying_evidence), f"{name}_evidence_count"
                )
                model.add(evidence_count == sum(qualifying_evidence))
                score = model.new_int_var(0, 100_000_000, f"{name}_score_value")
                model.add_division_equality(
                    score,
                    evidence_count * 100_000_000,
                    max(1, len(qualifying_evidence)),
                )
                if native_delayed and not paired_delayed:
                    # shortcut: paired mechanism/world certificates are missing; enable causality only with those bindings.
                    causal_proven = model.new_bool_var(f"{name}_causal_proven")
                    model.add(causal_proven == 0)
                else:
                    causal_proven = self._and(
                        model,
                        (verified, has_causal_advantage),
                        f"{name}_causal_proven",
                    )
                if predicate.kind == "tactic.applicable":
                    expected = bool(arguments["applicable"])
                    occurrence = applicable if expected else applicable.Not()
                    literal = self._and(
                        model,
                        (context_matches, occurrence),
                        f"{name}_applicable_result",
                    )
                else:
                    score_ok = model.new_bool_var(f"{name}_score")
                    threshold = round(float(arguments["minimumScore"]) * 1_000_000)
                    model.add(score >= threshold).only_enforce_if(score_ok)
                    model.add(score < threshold).only_enforce_if(score_ok.Not())
                    required_evidence = int(arguments["minimumEvidence"])
                    enough_evidence = model.new_bool_var(f"{name}_enough_evidence")
                    model.add(evidence_count >= required_evidence).only_enforce_if(
                        enough_evidence
                    )
                    model.add(evidence_count < required_evidence).only_enforce_if(
                        enough_evidence.Not()
                    )
                    causal_gate = causal_proven
                    if not bool(arguments.get("causalRequired", False)):
                        causal_gate = model.new_bool_var(f"{name}_causal_not_required")
                        model.add(causal_gate == 1)
                    literal = self._and(
                        model,
                        (
                            context_matches,
                            applicable,
                            verified,
                            score_ok,
                            enough_evidence,
                            causal_gate,
                        ),
                        f"{name}_native_evidence",
                    )
                return literal, {
                    "proofComplete": True,
                    "detectorId": tactic,
                    "detectorVersion": "3.1.0" if paired_delayed else "3.0.1" if native_delayed or native_project or trump_threshold is not None or paired_threshold is not None else "3.0.0" if live and (tactic == TacticalKind.PROJECT_BIDDING.value
                        or tactic == TacticalKind.TRUMP_CONTROL.value and native_trump) else "2.0.0",
                    "scorePolicy": "paired_best_alternative_mechanism_v1" if paired_threshold is not None else "qualifying_root_witness_ratio_v1",
                    "requestedMinimumScore": float(arguments.get("minimumScore", 0)),
                    "requestedMinimumEvidence": int(
                        arguments.get("minimumEvidence", 1)
                    ),
                    "causalRequired": bool(arguments.get("causalRequired", False)),
                    **({"pairedRawPayoffScope": "canonical_complete_state_continuations",
                        "pairedRawPayoffUnit": "raw_card_points"}
                       if (native_delayed or paired_threshold is not None) and getattr(context, "root_raw_payoff_variables", ()) else {}),
                    **({"causalProofStatus": "paired_native_delayed_payoff" if paired_delayed
                        else "missing_native_delayed_payoff_witness"} if native_delayed else {}),
                    **({"nativeDelayedPayoff": {
                        "gainMetric": profile.native_delayed_gain_metric,
                        "minimumGain": profile.native_delayed_minimum_gain},
                        "nativeGainScale": metric_scale,
                        "minimumNativeGainScaled": delayed_threshold,
                        "rawPayoffUnit": "raw_card_points_excluding_last_trick_bonus",
                        "continuationObjective": "awarded_point_margin"}
                       if paired_delayed else {}),
                    **({"pairedDelayedControlVariables": [{key: value.name for key, value in row.items()
                        if key.endswith(("WinningCard", "OpponentTrumpsRemaining"))}
                        for row in context.root_raw_payoff_variables]} if paired_delayed else {}),
                    **({"detectorProfileHash": profile.sha256,
                        ("lowCardMaxStrength" if native_small else "highCardMinStrength"): cutoff,
                        "strengthNormalization": "within_suit_0_100"} if card_strength_bindings is not None else {}),
                    **({"detectorProfileHash": profile.sha256,
                        "projectConsequenceDelta": profile.project_consequence_delta,
                        "projectConsequenceUnit": "awarded_point_margin",
                        "projectConsequenceScale": scale,
                        "minimumProjectConsequenceScaled": project_threshold}
                       if project_threshold is not None else {}),
                    **({"detectorProfileHash": profile.sha256,
                        "minimumCausalGain": profile.minimum_causal_gain,
                        "causalGainUnit": "awarded_point_margin",
                        "causalGainScale": scale,
                        "minimumCausalGainScaled": trump_threshold if trump_threshold is not None else paired_threshold}
                       if trump_threshold is not None or paired_threshold is not None else {}),
                    **({"mechanismKind": tactic,
                        "mechanismScope": "paired_canonical_continuations",
                        "comparisonPolicy": "best_actor_alternative_stable_root_order",
                        "pairedWinnerVariables": [{key: value.name for key, value in row.items()} for row in paired_winners]}
                       if paired_threshold is not None else {}),
                    **({"pairedEntryVariables": [{key: value.name for key, value in row.items()} for row in entry_variables],
                        "payoffPointUnit": "raw_card_points_excluding_last_trick_bonus"}
                       if entry_variables else {}),
                    "qualificationSource": (
                        "native_cp_sat_root_action_prerequisite_variables"
                        if len(root_actions) < minimum_roots
                        else
                        "native_cp_sat_opponent_stop_and_public_trick_variables"
                        if actor_stopping
                        else
                        "native_cp_sat_root_win_and_public_trick_variables"
                        if live and getattr(context, "base_tricks", ()) is None
                        and tactic in {TacticalKind.KABOUT_CREATION.value, TacticalKind.KABOUT_PRESERVATION.value}
                        else
                        "native_cp_sat_paired_project_consequence_variables"
                        if live and tactic == TacticalKind.PROJECT_BIDDING.value
                        else "native_cp_sat_paired_delayed_payoff_variables" if paired_delayed
                        else "native_cp_sat_paired_entry_variables" if entry_variables
                        else "native_cp_sat_paired_partner_winner_variables"
                        if paired_threshold is not None
                        else "native_cp_sat_root_values_and_trump_extraction"
                        if trump_extraction is not None
                        else "native_cp_sat_root_win_variables"
                        if tactic == TacticalKind.ONLY_WINNING_MOVE.value
                        else "native_cp_sat_root_value_variables"
                    ),
                    "rootActionCount": len(root_winning),
                    **({"inapplicabilityReason": "required_alternative_absent",
                        "minimumRequiredRootActions": minimum_roots}
                       if len(root_actions) < minimum_roots else {}),
                    "maximumEvidenceCount": len(qualifying_evidence),
                    "selectedRootAction": context.proof.selected_action,
                    "selectedValueVariable": (root_values[selected_index].name),
                    "winningCountVariable": winning_count.name,
                    "evidenceCountVariable": evidence_count.name,
                    "scoreVariable": score.name,
                    "winningVariables": [item.name for item in root_winning],
                    "rootValueVariables": [item.name for item in root_values],
                    **({"mechanismKind": "trump_extraction",
                        "mechanismScope": "selected_canonical_continuation",
                        "trumpExtractionVariable": trump_extraction.name}
                       if trump_extraction is not None else {}),
                    "proofHash": context.proof.sha256,
                    **({"alternativeGuaranteeScope": "no_uniform_target_kabout_guarantee"}
                       if live and getattr(context, "base_tricks", ()) is None
                       and tactic in {TacticalKind.KABOUT_CREATION.value, TacticalKind.KABOUT_PRESERVATION.value}
                       else {}),
                    **({"selectedGuaranteeScope": "uniform_opponent_no_target_kabout_policy",
                        "alternativeGuaranteeScope": "uniform_target_kabout_guarantee",
                        "stoppingVariables": [variable.name for variable in context.root_stopping_variables]}
                       if actor_stopping else {}),
                    "_solverBools": {
                        "applicable": applicable,
                        "verified": verified,
                        "causalProven": causal_proven,
                        "contextMatches": context_matches,
                        **mechanism_flags,
                        **({"trumpExtraction": trump_extraction} if trump_extraction is not None else {}),
                    },
                    "_solverInts": {
                        "matchingEvidenceCount": evidence_count,
                        **({"bestAlternativeRootIndex": paired_comparator} if paired_comparator is not None else {}),
                        **({"payoffDepth": payoff_depth} if payoff_depth is not None else {}),
                        **({"nativeGain": delayed_gain, "laterCardMarginGain": delayed_raw_gain,
                            "firstTrickConcession": delayed_concession} if paired_delayed else {}),
                        **(card_strength_bindings or {}),
                        **({f"root{index}{key[0].upper() + key[1:]}": variable
                            for index, row in enumerate(context.root_raw_payoff_variables)
                            for key, variable in row.items()}
                           if (native_delayed or paired_threshold is not None) and getattr(context, "root_raw_payoff_variables", ()) else {}),
                        **({"baseNorthSouthTricks": context.base_trick_variables[0],
                            "baseWestEastTricks": context.base_trick_variables[1]}
                           if live and getattr(context, "base_tricks", ()) is None
                           and tactic in {TacticalKind.KABOUT_CREATION.value, TacticalKind.KABOUT_PRESERVATION.value,
                               TacticalKind.KABOUT_PREVENTION.value}
                           else {}),
                        **({f"rootProjectConsequence{index}": value for index, value in enumerate(consequences)}
                           if live and tactic == TacticalKind.PROJECT_BIDDING.value else {}),
                    },
                    "_solverScaledInts": {"score": (score, 1_000_000)},
                }
        raise AssertionError("registered proof/tactic predicate disappeared")

    @staticmethod
    def _delayed_strength_boundary(model, card, strengths, profile, small, name):
        if len(strengths) != 32:
            raise ValueError("native delayed payoff requires the complete card-strength table")
        cutoff = profile.low_card_max_strength if small else profile.high_card_min_strength
        threshold_key = "allowedNumerator" if small else "requiredNumerator"
        numerators, spans, cutoffs = [], [], []
        for card_id, strength in enumerate(strengths):
            suit_strengths = strengths[8 * (card_id // 8):8 * (card_id // 8 + 1)]
            span = max(1, max(suit_strengths) - min(suit_strengths))
            numerators.append(100 * (strength - min(suit_strengths)))
            spans.append(span)
            scaled = Fraction(str(cutoff)) * span
            cutoffs.append(int(scaled) if small else ceil(scaled))
        bindings = {}
        for key, values in (("strengthNumerator", numerators), ("strengthSpan", spans), (threshold_key, cutoffs)):
            variable = model.new_int_var(min(values), max(values), f"{name}_{key}")
            model.add_element(card, values, variable)
            bindings[key] = variable
        boundary = model.new_bool_var(f"{name}_configured_low_card" if small else f"{name}_configured_high_card")
        numerator, threshold = bindings["strengthNumerator"], bindings[threshold_key]
        if small:
            model.add(numerator <= threshold).only_enforce_if(boundary)
            model.add(numerator > threshold).only_enforce_if(boundary.Not())
        else:
            model.add(numerator >= threshold).only_enforce_if(boundary)
            model.add(numerator < threshold).only_enforce_if(boundary.Not())
        return boundary, bindings, cutoff

    @staticmethod
    def _trick(
        arguments: Mapping[str, Any],
        context: ProofTacticPredicateContextV1,
    ) -> PrincipalVariationTrickEvidenceV1 | None:
        index = int(arguments["trickIndex"])
        return next(
            (
                item
                for item in context.principal_variation_tricks
                if item.trick_index == index
            ),
            None,
        )


__all__ = [
    "PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2",
    "TACTICAL_DETECTOR_IDS_V1",
    "PrincipalVariationPlayEvidenceV1",
    "PrincipalVariationTrickEvidenceV1",
    "ProofTacticPredicateContextV1",
    "ProofTacticPredicateLoweringV2",
    "RegisteredTacticalEvidenceV1",
]
