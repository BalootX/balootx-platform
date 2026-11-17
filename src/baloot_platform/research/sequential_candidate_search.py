"""Product-owned orchestration for sequential structural candidate search.

The generator owns CP-SAT construction and exact no-good cuts.  The injected
evaluator owns semantic replay and qualification.  This module coordinates
those authorities without importing the pilot runner or duplicating either
authority's policy semantics.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from baloot_platform.challenge.constraint_deal_search import (
    ConstraintDealCandidateV1,
    ConstraintDealContinuationReceiptV1,
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_hash(value: Any, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    digest = value
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return digest


def _require_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _require_fields(
    value: Mapping[str, Any],
    field: str,
    expected: tuple[str, ...],
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{field} has an unknown or missing field")


def _canonical_strings(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if any(type(value) is not str for value in values):
        raise ValueError(f"{field} must contain strings")
    normalized = tuple(sorted({value.strip() for value in values}))
    if any(not value for value in normalized):
        raise ValueError(f"{field} cannot contain an empty value")
    return normalized


class SequentialCandidateOutcomeKindV1(str, Enum):
    ORDINARY_REJECTION = "ordinary_rejection"
    QUALIFIED = "qualified"
    FATAL_QUARANTINE = "fatal_quarantine"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SequentialCandidateEvaluationV1:
    """Typed, immutable evaluator outcome."""

    outcome: SequentialCandidateOutcomeKindV1
    candidate_hash: str
    evaluation_evidence_hashes: tuple[str, ...]
    failure_codes: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        outcome: SequentialCandidateOutcomeKindV1 | str,
        candidate_hash: str,
        evaluation_evidence_hashes: tuple[str, ...],
        failure_codes: tuple[str, ...] = (),
    ) -> "SequentialCandidateEvaluationV1":
        parsed = SequentialCandidateOutcomeKindV1(outcome)
        candidate_digest = _require_hash(candidate_hash, "candidateHash")
        evidence = _canonical_strings(
            evaluation_evidence_hashes,
            "evaluationEvidenceHashes",
        )
        for digest in evidence:
            _require_hash(digest, "evaluationEvidenceHash")
        failures = _canonical_strings(failure_codes, "failureCodes")
        if parsed in {
            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
            SequentialCandidateOutcomeKindV1.QUALIFIED,
        } and not evidence:
            raise ValueError(
                "a complete rejection or qualification requires evaluation evidence"
            )
        if parsed == SequentialCandidateOutcomeKindV1.QUALIFIED:
            if failures:
                raise ValueError("a qualified evaluation cannot declare failures")
        elif not failures:
            raise ValueError("a non-qualified evaluation requires a failure code")
        core = {
            "schemaVersion": 1,
            "kind": "research_sequential_candidate_evaluation_v1",
            "outcome": parsed.value,
            "candidateHash": candidate_digest,
            "evaluationEvidenceHashes": list(evidence),
            "failureCodes": list(failures),
        }
        return cls(parsed, candidate_digest, evidence, failures, canonical_sha256(core))

    def integrity_core(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "research_sequential_candidate_evaluation_v1",
            "outcome": self.outcome.value,
            "candidateHash": self.candidate_hash,
            "evaluationEvidenceHashes": list(self.evaluation_evidence_hashes),
            "failureCodes": list(self.failure_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.integrity_core(), "sha256": self.sha256}

    def verify(self) -> "SequentialCandidateEvaluationV1":
        if canonical_sha256(self.integrity_core()) != _require_hash(
            self.sha256,
            "evaluation sha256",
        ):
            raise ValueError("candidate evaluation integrity is invalid")
        rebuilt = self.build(
            outcome=self.outcome,
            candidate_hash=self.candidate_hash,
            evaluation_evidence_hashes=self.evaluation_evidence_hashes,
            failure_codes=self.failure_codes,
        )
        if rebuilt != self:
            raise ValueError("candidate evaluation is not canonical")
        return self


@dataclass(frozen=True, slots=True)
class SequentialCandidateSearchReceiptV1:
    """Immutable canonical per-seed receipt bundle."""

    _canonical_core_json: str
    sha256: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "SequentialCandidateSearchReceiptV1":
        document = copy.deepcopy(dict(value))
        known_hash = _require_hash(document.pop("sha256", ""), "receipt sha256")
        _validate_receipt_links(document)
        if canonical_sha256(document) != known_hash:
            raise ValueError("sequential candidate search receipt integrity is invalid")
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(encoded, known_hash)

    @classmethod
    def _build(cls, core: Mapping[str, Any]) -> "SequentialCandidateSearchReceiptV1":
        document = copy.deepcopy(dict(core))
        return cls.from_dict({**document, "sha256": canonical_sha256(document)})

    def to_dict(self) -> dict[str, Any]:
        return {**json.loads(self._canonical_core_json), "sha256": self.sha256}

    def verify(self) -> "SequentialCandidateSearchReceiptV1":
        if self.from_dict(self.to_dict()) != self:
            raise ValueError("sequential candidate search receipt integrity is invalid")
        return self


@dataclass(frozen=True, slots=True)
class SequentialCandidateSearchResultV1:
    qualified_candidate: ConstraintDealCandidateV1 | None
    qualified_evaluation: SequentialCandidateEvaluationV1 | None
    receipt: SequentialCandidateSearchReceiptV1


def _search_authority(value: Any) -> dict[str, Any]:
    return {
        "profileHash": value.profile_hash,
        "modelHash": value.model_hash,
        "searchIdentityHash": value.search_identity_hash,
        "backendVersion": value.backend_version,
        "noGoodCutType": value.no_good_cut_type,
        "structuralObjectiveScore": value.structural_objective_score,
    }


def _candidate_snapshot(
    candidate: Any,
    *,
    descriptor: PluginDescriptor,
    expected_profile_hash: str,
    expected_search_hash: str | None,
    ordinal: int,
    rejected: tuple[ConstraintDealCandidateV1, ...],
    feedback: list[dict[str, Any]],
    authority: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(candidate, ConstraintDealCandidateV1):
        raise TypeError("generator returned an unsupported candidate contract")
    candidate.verify()
    core = copy.deepcopy(candidate.integrity_core())
    if candidate.profile_hash != expected_profile_hash:
        raise ValueError("candidate changed the governed profile")
    if expected_search_hash is not None and candidate.search_identity_hash != expected_search_hash:
        raise ValueError("candidate changed the expected search identity")
    if candidate.ordinal != ordinal:
        raise ValueError("candidate ordinal is not sequential")
    prefix = tuple(candidate.preceding_no_good_cut_hashes)
    if len(prefix) != ordinal or len(prefix) != len(rejected):
        raise ValueError("candidate lost the ordered no-good prefix")
    for index, cut_hash in enumerate(prefix):
        expected_cut = feedback[index]["noGoodCut"]["cutSha256"]
        if expected_cut != cut_hash:
            raise ValueError("candidate changed an earlier no-good cut")
    if any(
        candidate.candidate_hash == previous.candidate_hash
        or candidate.deck_commitment == previous.deck_commitment
        for previous in rejected
    ):
        raise ValueError("candidate repeated a rejected candidate or deck")
    candidate_authority = _search_authority(candidate)
    if authority is not None and candidate_authority != authority:
        raise ValueError("candidate changed the frozen search authority")
    provenance = candidate.provenance_dict()
    if provenance.get("pluginId") != descriptor.id or provenance.get(
        "candidateHash"
    ) != candidate.candidate_hash:
        raise ValueError("candidate provenance linkage is invalid")
    return core, candidate_authority


def _evaluation_receipt(
    candidate: ConstraintDealCandidateV1,
    evaluation: SequentialCandidateEvaluationV1,
    *,
    attempt: str,
) -> dict[str, Any]:
    evaluation.verify()
    if evaluation.candidate_hash != candidate.candidate_hash:
        raise ValueError("candidate evaluation changed the candidate identity")
    core = {
        "schemaVersion": 1,
        "kind": "research_sequential_candidate_evaluation_receipt_v1",
        "candidateOrdinal": candidate.ordinal,
        "attempt": attempt,
        "candidateIntegrityCore": copy.deepcopy(candidate.integrity_core()),
        "candidateProvenance": copy.deepcopy(candidate.provenance_dict()),
        "evaluation": evaluation.to_dict(),
    }
    return {**core, "sha256": canonical_sha256(core)}


def _feedback_core(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(row[field])
        for field in (
            "schemaVersion",
            "kind",
            "rejectionOrdinal",
            "candidateHash",
            "evaluationReceiptHash",
            "evaluationEvidenceHashes",
            "failureCodes",
            "noGoodCut",
        )
    }


def _feedback_state(
    core: Mapping[str, Any],
    *,
    asked: bool,
    cut_hash: str | None = None,
    application: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feedback_core = copy.deepcopy(dict(core))
    state = {
        **feedback_core,
        "feedbackHash": canonical_sha256(feedback_core),
        "askedForNextSolution": asked,
        "appliedToSubsequentSolve": application is not None,
        "noGoodCutHash": cut_hash,
        "applicationReceipt": copy.deepcopy(application),
    }
    return {**state, "stateHash": canonical_sha256(state)}


def _new_feedback(
    candidate: ConstraintDealCandidateV1,
    evaluation_receipt: Mapping[str, Any],
    no_good_cut: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = evaluation_receipt["evaluation"]
    core = {
        "schemaVersion": 1,
        "kind": "research_exact_candidate_no_good_feedback_v1",
        "rejectionOrdinal": candidate.ordinal,
        "candidateHash": candidate.candidate_hash,
        "evaluationReceiptHash": evaluation_receipt["sha256"],
        "evaluationEvidenceHashes": list(evaluation["evaluationEvidenceHashes"]),
        "failureCodes": list(evaluation["failureCodes"]),
        "noGoodCut": copy.deepcopy(dict(no_good_cut)),
    }
    return _feedback_state(core, asked=False)


def _no_good_cut_document(
    candidate: ConstraintDealCandidateV1,
    *,
    spec: DealConstraintSpecV1,
    dealer: Seat,
    rules: RuleBook,
) -> dict[str, Any]:
    layout = None
    if candidate.no_good_cut_type == "final_hand_allocation_v1":
        layout = DealLayoutV1.build(
            rules.structural_domain,
            dealer=dealer,
            buyer=spec.assumed_buyer,
        )
    return OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
        candidate,
        layout=layout,
    )


def _application(
    feedback: Mapping[str, Any],
    *,
    cut_hash: str,
    evidence_kind: str,
    evidence_hash: str,
    solve_ordinal: int,
    solver_status: str,
    next_candidate_hash: str | None,
) -> dict[str, Any]:
    core = {
        "schemaVersion": 1,
        "kind": "research_exact_candidate_no_good_application_v1",
        "feedbackHash": feedback["feedbackHash"],
        "noGoodCutHash": _require_hash(cut_hash, "noGoodCutHash"),
        "solveEvidenceKind": evidence_kind,
        "solveEvidenceHash": _require_hash(evidence_hash, "solveEvidenceHash"),
        "solveOrdinal": solve_ordinal,
        "solverStatus": solver_status,
        "nextCandidateHash": (
            _require_hash(next_candidate_hash, "nextCandidateHash")
            if next_candidate_hash is not None
            else None
        ),
    }
    return {**core, "sha256": canonical_sha256(core)}


def _ask_for_next(feedback: list[dict[str, Any]]) -> None:
    if not feedback:
        return
    final = feedback[-1]
    if final["applicationReceipt"] is not None:
        raise ValueError("final no-good feedback was already applied")
    feedback[-1] = _feedback_state(_feedback_core(final), asked=True)


def _apply_next_candidate(
    candidate: ConstraintDealCandidateV1,
    feedback: list[dict[str, Any]],
) -> None:
    if not feedback:
        return
    prefix = tuple(candidate.preceding_no_good_cut_hashes)
    final = feedback[-1]
    if len(prefix) != len(feedback) or not final["askedForNextSolution"]:
        raise ValueError("candidate no-good application prefix is invalid")
    expected_prefix = tuple(row["noGoodCut"]["cutSha256"] for row in feedback)
    if prefix != expected_prefix:
        raise ValueError("candidate changed the exact no-good cut prefix")
    application = _application(
        final,
        cut_hash=expected_prefix[-1],
        evidence_kind="next_candidate",
        evidence_hash=candidate.candidate_hash,
        solve_ordinal=candidate.ordinal,
        solver_status=candidate.solver_status,
        next_candidate_hash=candidate.candidate_hash,
    )
    feedback[-1] = _feedback_state(
        _feedback_core(final),
        asked=True,
        cut_hash=expected_prefix[-1],
        application=application,
    )


def _apply_continuation(
    receipt: Any,
    *,
    rejected: tuple[ConstraintDealCandidateV1, ...],
    feedback: list[dict[str, Any]],
    authority: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(receipt, ConstraintDealContinuationReceiptV1):
        raise TypeError("generator returned an unsupported continuation receipt")
    receipt.verify()
    receipt_hash = _require_hash(receipt.receipt_hash, "continuation sha256")
    receipt_authority = _search_authority(receipt)
    prefix = tuple(receipt.preceding_no_good_cut_hashes)
    expected_prefix = tuple(row["noGoodCut"]["cutSha256"] for row in feedback)
    if (
        (authority is not None and receipt_authority != authority)
        or receipt.solution_ordinal != len(rejected)
        or prefix != expected_prefix
    ):
        raise ValueError("continuation receipt search linkage is invalid")
    if receipt.witness_candidate_hash is not None:
        if any(
            receipt.witness_candidate_hash == candidate.candidate_hash
            or receipt.witness_deck_commitment == candidate.deck_commitment
            for candidate in rejected
        ):
            raise ValueError("continuation witness repeated a rejected candidate")
    if feedback:
        final = feedback[-1]
        if not final["askedForNextSolution"] or final["applicationReceipt"] is not None:
            raise ValueError("terminal no-good feedback state is invalid")
        application = _application(
            final,
            cut_hash=expected_prefix[-1],
            evidence_kind="post_rejection_continuation_receipt",
            evidence_hash=receipt_hash,
            solve_ordinal=receipt.solution_ordinal,
            solver_status=receipt.solver_status,
            next_candidate_hash=receipt.witness_candidate_hash,
        )
        feedback[-1] = _feedback_state(
            _feedback_core(final),
            asked=True,
            cut_hash=expected_prefix[-1],
            application=application,
        )
    return receipt.to_dict()


def _terminal_probe(
    generator: Any,
    spec: DealConstraintSpecV1,
    *,
    root_seed: int,
    dealer: Seat,
    rules: RuleBook,
    deadline_monotonic: float | None,
    rejected: tuple[ConstraintDealCandidateV1, ...],
    feedback: list[dict[str, Any]],
    authority: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ConstraintDealContinuationReceiptV1]:
    """Run the evidence-only solve after the current ordered cut prefix."""

    _ask_for_next(feedback)
    receipt = generator.probe_continuation(
        spec,
        root_seed=root_seed,
        dealer=dealer,
        rules=rules,
        rejected_candidates=rejected,
        deadline_monotonic=deadline_monotonic,
    )
    return (
        _apply_continuation(
            receipt,
            rejected=rejected,
            feedback=feedback,
            authority=authority,
        ),
        receipt,
    )


def _evaluate(
    evaluator: Callable[
        [ConstraintDealCandidateV1, int],
        SequentialCandidateEvaluationV1,
    ],
    candidate: ConstraintDealCandidateV1,
    index: int,
) -> SequentialCandidateEvaluationV1:
    try:
        result = evaluator(candidate, index)
    except InterruptedError:
        result = SequentialCandidateEvaluationV1.build(
            outcome=SequentialCandidateOutcomeKindV1.CANCELLED,
            candidate_hash=candidate.candidate_hash,
            evaluation_evidence_hashes=(),
            failure_codes=("candidate_evaluation_cancelled",),
        )
    if not isinstance(result, SequentialCandidateEvaluationV1):
        raise TypeError("evaluator must return SequentialCandidateEvaluationV1")
    return result.verify()


def run_sequential_candidate_search_v1(
    generator: Any,
    spec: DealConstraintSpecV1,
    *,
    root_seed: int,
    dealer: Seat,
    rules: RuleBook,
    evaluate_candidate: Callable[
        [ConstraintDealCandidateV1, int],
        SequentialCandidateEvaluationV1,
    ],
    deadline_monotonic: float | None = None,
    expected_search_identity_hash: str | None = None,
) -> SequentialCandidateSearchResultV1:
    """Evaluate at most ``maximumSolutions`` distinct emitted candidates."""

    if not isinstance(spec, DealConstraintSpecV1):
        raise TypeError("spec must be DealConstraintSpecV1")
    if not isinstance(dealer, Seat):
        raise TypeError("dealer must be a Seat")
    root_seed = _require_int(root_seed, "rootSeed")
    rules_hash = _require_hash(getattr(rules, "sha256", ""), "rulesHash")
    expected_search = (
        _require_hash(expected_search_identity_hash, "expectedSearchIdentityHash")
        if expected_search_identity_hash is not None
        else None
    )
    descriptor = getattr(generator, "descriptor", None)
    if not isinstance(descriptor, PluginDescriptor) or descriptor.kind != "generator":
        raise TypeError("generator must expose a generator PluginDescriptor")
    descriptor_document = descriptor.to_dict()
    if (
        not descriptor.deterministic
        or descriptor_document
        != OrToolsCpSatDealGeneratorV1.descriptor.to_dict()
    ):
        raise ValueError("generator must implement the canonical deterministic deal contract")
    descriptor_hash = canonical_sha256(descriptor_document)
    request_core = {
        "schemaVersion": 1,
        "kind": "research_sequential_candidate_search_request_v1",
        "rootSeed": int(root_seed),
        "dealer": dealer.name.lower(),
        "rulesHash": rules_hash,
        "profileId": spec.profile_id,
        "profileRevision": spec.revision,
        "profileHash": spec.sha256,
        "assumedBuyer": (
            spec.assumed_buyer.name.lower()
            if spec.assumed_buyer is not None
            else None
        ),
        "maximumSolutions": spec.maximum_solutions,
        "generatorDescriptorHash": descriptor_hash,
        "expectedSearchIdentityHash": expected_search,
    }
    search_request = {**request_core, "sha256": canonical_sha256(request_core)}

    rejected: tuple[ConstraintDealCandidateV1, ...] = ()
    feedback: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    authority: dict[str, Any] | None = None
    emitted = 0
    evaluated = 0
    qualified_candidate = None
    qualified_evaluation = None
    continuation_document = None
    search_identity_document = None
    stop_reason = "candidates_exhausted"
    extra_failures: list[str] = []

    while evaluated < spec.maximum_solutions:
        _ask_for_next(feedback)
        candidate = generator.generate_next(
            spec,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            rejected_candidates=rejected,
            deadline_monotonic=deadline_monotonic,
        )
        if candidate is None:
            continuation_document, continuation = _terminal_probe(
                generator,
                spec,
                root_seed=root_seed,
                dealer=dealer,
                rules=rules,
                deadline_monotonic=deadline_monotonic,
                rejected=rejected,
                feedback=feedback,
                authority=authority,
            )
            if not continuation.exhausted:
                raise ValueError("candidate and continuation solves disagree")
            continuation_authority = _search_authority(continuation)
            continuation_search_identity = (
                OrToolsCpSatDealGeneratorV1.search_identity_document(
                    spec=spec,
                    model_hash=continuation.model_hash,
                    root_seed=root_seed,
                    dealer=dealer,
                    rules=rules,
                    backend_version=continuation.backend_version,
                    no_good_cut_type=continuation.no_good_cut_type,
                )
            )
            if (
                continuation_search_identity["sha256"]
                != continuation.search_identity_hash
                or (
                    expected_search is not None
                    and continuation.search_identity_hash != expected_search
                )
                or (
                    search_identity_document is not None
                    and continuation_search_identity != search_identity_document
                )
            ):
                raise ValueError("continuation search identity preimage is invalid")
            authority = authority or continuation_authority
            search_identity_document = continuation_search_identity
            stop_reason = "candidates_exhausted"
            extra_failures.append("candidate_space_exhausted")
            break

        snapshot, candidate_authority = _candidate_snapshot(
            candidate,
            descriptor=descriptor,
            expected_profile_hash=spec.sha256,
            expected_search_hash=expected_search,
            ordinal=evaluated,
            rejected=rejected,
            feedback=feedback,
            authority=authority,
        )
        candidate_search_identity = OrToolsCpSatDealGeneratorV1.search_identity_document(
            spec=spec,
            model_hash=candidate.model_hash,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            backend_version=candidate.backend_version,
            no_good_cut_type=candidate.no_good_cut_type,
        )
        if candidate_search_identity["sha256"] != candidate.search_identity_hash:
            raise ValueError("candidate search identity preimage is invalid")
        if (
            search_identity_document is not None
            and candidate_search_identity != search_identity_document
        ):
            raise ValueError("candidate changed the frozen search identity preimage")
        search_identity_document = candidate_search_identity
        authority = authority or candidate_authority
        _apply_next_candidate(candidate, feedback)
        emitted += 1
        evaluation = _evaluate(evaluate_candidate, candidate, evaluated + 1)
        if candidate.integrity_core() != snapshot:
            raise ValueError("evaluator modified the generated candidate")
        primary = _evaluation_receipt(candidate, evaluation, attempt="primary")
        evaluations.append(primary)
        evaluated += 1

        if evaluation.outcome == SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION:
            feedback.append(_new_feedback(
                candidate,
                primary,
                _no_good_cut_document(
                    candidate,
                    spec=spec,
                    dealer=dealer,
                    rules=rules,
                ),
            ))
            rejected = (*rejected, candidate)
            if evaluated < spec.maximum_solutions:
                continue
            continuation_document, continuation = _terminal_probe(
                generator,
                spec,
                root_seed=root_seed,
                dealer=dealer,
                rules=rules,
                deadline_monotonic=deadline_monotonic,
                rejected=rejected,
                feedback=feedback,
                authority=authority,
            )
            if continuation.exhausted:
                stop_reason = "candidates_exhausted"
                extra_failures.append("candidate_space_exhausted")
            else:
                stop_reason = "maximum_solutions_reached"
                extra_failures.append("maximum_solutions_reached")
            break

        if evaluation.outcome == SequentialCandidateOutcomeKindV1.QUALIFIED:
            repeated = _evaluate(evaluate_candidate, candidate, evaluated)
            if candidate.integrity_core() != snapshot:
                raise ValueError("qualified repeat modified the generated candidate")
            evaluations.append(
                _evaluation_receipt(candidate, repeated, attempt="qualified_repeat")
            )
            if repeated == evaluation:
                stop_reason = "qualified"
                qualified_candidate = candidate
                qualified_evaluation = evaluation
            elif repeated.outcome == SequentialCandidateOutcomeKindV1.CANCELLED:
                stop_reason = "cancelled"
                extra_failures.append("qualified_repeat_cancelled")
            elif repeated.outcome == SequentialCandidateOutcomeKindV1.INCOMPLETE:
                stop_reason = "incomplete"
                extra_failures.append("qualified_repeat_incomplete")
            else:
                stop_reason = "fatal_quarantine"
                extra_failures.append("qualified_repeat_mismatch")
            break

        stop_reason = evaluation.outcome.value
        break

    failures = sorted({
        *extra_failures,
        *(
            code
            for receipt in evaluations
            for code in receipt["evaluation"]["failureCodes"]
        ),
    })
    authority_document = copy.deepcopy(authority) if authority is not None else {
        "profileHash": spec.sha256,
        "modelHash": None,
        "searchIdentityHash": expected_search,
        "backendVersion": None,
        "noGoodCutType": None,
        "structuralObjectiveScore": None,
    }
    receipt = SequentialCandidateSearchReceiptV1._build({
        "schemaVersion": 1,
        "kind": "research_sequential_candidate_search_receipt_v1",
        "rootSeed": int(root_seed),
        "dealer": dealer.name.lower(),
        "rulesHash": rules_hash,
        "profileId": spec.profile_id,
        "profileRevision": spec.revision,
        "assumedBuyer": (
            spec.assumed_buyer.name.lower()
            if spec.assumed_buyer is not None
            else None
        ),
        "maximumSolutions": spec.maximum_solutions,
        "searchRequest": search_request,
        "searchIdentityDocument": search_identity_document,
        "generatorDescriptor": descriptor_document,
        "generatorDescriptorHash": descriptor_hash,
        **authority_document,
        "emittedCandidateCount": emitted,
        "evaluatedCandidateCount": evaluated,
        "rejectedCandidateCount": len(feedback),
        "evaluationReceipts": evaluations,
        "rejectionFeedbackRows": feedback,
        "terminalContinuationReceipt": continuation_document,
        "qualifiedCandidateHash": (
            qualified_candidate.candidate_hash if qualified_candidate is not None else None
        ),
        "stopReason": stop_reason,
        "failureCodes": failures,
    })
    return SequentialCandidateSearchResultV1(
        qualified_candidate,
        qualified_evaluation,
        receipt,
    )


def _require_document_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = copy.deepcopy(dict(value))
    known = _require_hash(core.pop("sha256", ""), f"{field} sha256")
    if canonical_sha256(core) != known:
        raise ValueError(f"{field} integrity is invalid")
    return core


def _evaluation_from_document(value: Mapping[str, Any]) -> SequentialCandidateEvaluationV1:
    core = _require_document_hash(value, "evaluation")
    if (
        _require_int(core.get("schemaVersion"), "evaluation schemaVersion") != 1
        or core.get("kind") != "research_sequential_candidate_evaluation_v1"
    ):
        raise ValueError("unsupported evaluation contract")
    rebuilt = SequentialCandidateEvaluationV1.build(
        outcome=core.get("outcome", ""),
        candidate_hash=str(core.get("candidateHash", "")),
        evaluation_evidence_hashes=tuple(core.get("evaluationEvidenceHashes") or ()),
        failure_codes=tuple(core.get("failureCodes") or ()),
    )
    if rebuilt.integrity_core() != core or rebuilt.to_dict() != dict(value):
        raise ValueError("evaluation document is not canonical")
    return rebuilt


def _candidate_from_documents(
    core: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ConstraintDealCandidateV1:
    try:
        candidate = ConstraintDealCandidateV1(
            ordinal=core["ordinal"],
            deck_order=tuple(core["deckOrder"]),
            deck_commitment=core["deckCommitment"],
            ordering_score=core["orderingScore"],
            structural_objective_score=core["structuralObjectiveScore"],
            profile_hash=core["profileHash"],
            model_hash=core["modelHash"],
            backend_version=core["backendVersion"],
            solver_status=core["solverStatus"],
            no_good_cut_type=core["noGoodCutType"],
            candidate_hash=provenance["candidateHash"],
            search_identity_hash=core["searchIdentityHash"],
            preceding_no_good_cut_hashes=tuple(
                core["precedingNoGoodCutHashes"]
            ),
        ).verify()
    except (KeyError, TypeError) as exc:
        raise ValueError("candidate document is invalid") from exc
    if (
        canonical_sha256(candidate.integrity_core()) != canonical_sha256(core)
        or canonical_sha256(candidate.provenance_dict())
        != canonical_sha256(provenance)
    ):
        raise ValueError("candidate document is not canonical")
    return candidate


def _validate_layout_document(
    value: Mapping[str, Any],
    *,
    structural_domain_document: Mapping[str, Any],
    dealer: str,
    buyer: str | None,
) -> dict[str, Any]:
    document = OrToolsCpSatDealGeneratorV1.validate_deal_layout_document(
        value,
        structural_domain_document=structural_domain_document,
        dealer=dealer,
        buyer=buyer,
    )
    return {key: child for key, child in document.items() if key != "sha256"}


def _validate_no_good_cut_document(
    value: Mapping[str, Any],
    provenance: Mapping[str, Any],
    candidate_core: Mapping[str, Any],
    *,
    expected_dealer: str,
    expected_buyer: str | None,
    expected_final_positions: Mapping[str, Any] | None,
) -> str:
    document_core = _require_document_hash(value, "exact no-good cut document")
    _require_fields(
        document_core,
        "exact no-good cut document",
        (
            "schemaVersion", "kind", "candidateHash", "searchIdentityHash",
            "profileHash", "modelHash", "ordinal", "noGoodCutType",
            "exactAssignment", "cutSha256", "validationFinalHandPositions",
            "validationDealer", "validationBuyer",
        ),
    )
    if (
        _require_int(
            document_core.get("schemaVersion"),
            "exact no-good cut schemaVersion",
        )
        != 1
        or document_core.get("kind") != "cp_sat_exact_candidate_no_good_v1"
    ):
        raise ValueError("unsupported exact no-good cut document")
    layout_positions = document_core.pop("validationFinalHandPositions", None)
    validation_dealer = document_core.pop("validationDealer", None)
    validation_buyer = document_core.pop("validationBuyer", None)
    cut_hash = _require_hash(
        document_core.pop("cutSha256", ""),
        "exact no-good cut sha256",
    )
    if canonical_sha256(document_core) != cut_hash:
        raise ValueError("exact no-good cut preimage is invalid")
    core = document_core
    for cut_field, provenance_field in (
        ("candidateHash", "candidateHash"),
        ("searchIdentityHash", "searchIdentityHash"),
        ("profileHash", "profileHash"),
        ("modelHash", "modelHash"),
        ("ordinal", "solutionOrdinal"),
        ("noGoodCutType", "noGoodCutType"),
    ):
        if core.get(cut_field) != provenance.get(provenance_field):
            raise ValueError("exact no-good cut candidate linkage is invalid")
    assignment = core.get("exactAssignment")
    _require_int(core.get("ordinal"), "exact no-good cut ordinal")
    if core.get("noGoodCutType") == "complete_deck_assignment_v1":
        deck = list(assignment or ())
        if (
            any(type(card) is not int for card in deck)
            or sorted(deck) != list(range(32))
            or deck != list(candidate_core.get("deckOrder") or ())
            or layout_positions is not None
            or validation_dealer is not None
            or validation_buyer is not None
        ):
            raise ValueError("complete-deck no-good assignment is invalid")
    elif core.get("noGoodCutType") == "final_hand_allocation_v1":
        hands = dict(assignment or {})
        cards = [
            card
            for seat in ("east", "south", "west", "north")
            for card in hands.get(seat, ())
        ]
        positions = dict(layout_positions or {})
        position_values = [
            position
            for seat in ("east", "south", "west", "north")
            for position in positions.get(seat, ())
        ]
        deck = list(candidate_core.get("deckOrder") or ())
        if (
            set(hands) != {"east", "south", "west", "north"}
            or set(positions) != set(hands)
            or any(len(hands[seat]) != 8 for seat in hands)
            or any(len(positions[seat]) != 8 for seat in positions)
            or any(type(card) is not int for card in cards)
            or any(type(position) is not int for position in position_values)
            or sorted(cards) != list(range(32))
            or sorted(position_values) != list(range(32))
            or validation_dealer != expected_dealer
            or validation_buyer != expected_buyer
            or expected_buyer is None
            or positions != dict(expected_final_positions or {})
            or any(
                sorted(hands[seat])
                != sorted(deck[position] for position in positions[seat])
                for seat in hands
            )
        ):
            raise ValueError("final-hand no-good assignment is invalid")
    else:
        raise ValueError("unsupported exact no-good cut type")
    return cut_hash


def _validate_continuation_document(value: Mapping[str, Any]) -> dict[str, Any]:
    core = _require_document_hash(value, "continuation receipt")
    _require_fields(
        core,
        "continuation receipt",
        (
            "schemaVersion", "kind", "pluginId", "pluginVersion", "profileHash",
            "modelHash", "backendVersion", "solverStatus", "solutionOrdinal",
            "searchIdentityHash", "precedingNoGoodCutCount",
            "precedingNoGoodCutHashes", "finalNoGoodCutHash", "noGoodCutType",
            "solverInvokedAfterFinalCut", "candidateEmitted", "candidateEvaluated",
            "candidatePublished", "exhausted", "witnessCandidateHash",
            "witnessDeckCommitment", "witnessOrderingScore",
            "structuralObjectiveScore",
        ),
    )
    prefix = list(core.get("precedingNoGoodCutHashes") or ())
    for digest in prefix:
        _require_hash(digest, "continuation noGoodCutHash")
    schema_version = _require_int(
        core.get("schemaVersion"),
        "continuation schemaVersion",
    )
    solution_ordinal = _require_int(
        core.get("solutionOrdinal"),
        "continuation solutionOrdinal",
    )
    prefix_count = _require_int(
        core.get("precedingNoGoodCutCount"),
        "continuation precedingNoGoodCutCount",
    )
    _require_int(
        core.get("structuralObjectiveScore"),
        "continuation structuralObjectiveScore",
    )
    if type(core.get("exhausted")) is not bool:
        raise ValueError("continuation exhausted must be a boolean")
    if (
        schema_version != 1
        or core.get("kind") != "cp_sat_candidate_space_probe_receipt_v1"
        or core.get("solverStatus") not in {"optimal", "infeasible"}
        or core["exhausted"] != (core.get("solverStatus") == "infeasible")
        or solution_ordinal != len(prefix)
        or prefix_count != len(prefix)
        or core.get("finalNoGoodCutHash")
        != (prefix[-1] if prefix else None)
        or core.get("solverInvokedAfterFinalCut") is not bool(prefix)
        or core.get("candidateEmitted") is not False
        or core.get("candidateEvaluated") is not False
        or core.get("candidatePublished") is not False
    ):
        raise ValueError("continuation receipt status, cut, or exposure evidence is invalid")
    witness = (
        core.get("witnessCandidateHash"),
        core.get("witnessDeckCommitment"),
        core.get("witnessOrderingScore"),
    )
    if (core["exhausted"] and any(item is not None for item in witness)) or (
        not core["exhausted"] and not all(item is not None for item in witness)
    ):
        raise ValueError("continuation witness evidence is invalid")
    if not core["exhausted"]:
        _require_hash(witness[0], "witnessCandidateHash")
        _require_hash(witness[1], "witnessDeckCommitment")
        _require_int(witness[2], "witnessOrderingScore")
    return core


def _validate_receipt_links(core: Mapping[str, Any]) -> None:
    _require_fields(
        core,
        "sequential candidate search receipt",
        (
            "schemaVersion", "kind", "rootSeed", "dealer", "rulesHash",
            "profileId", "profileRevision", "assumedBuyer", "maximumSolutions",
            "searchRequest", "searchIdentityDocument", "generatorDescriptor",
            "generatorDescriptorHash", "profileHash", "modelHash",
            "searchIdentityHash", "backendVersion", "noGoodCutType",
            "structuralObjectiveScore", "emittedCandidateCount",
            "evaluatedCandidateCount", "rejectedCandidateCount",
            "evaluationReceipts", "rejectionFeedbackRows",
            "terminalContinuationReceipt", "qualifiedCandidateHash", "stopReason",
            "failureCodes",
        ),
    )
    schema_version = _require_int(core.get("schemaVersion"), "receipt schemaVersion")
    if schema_version != 1 or core.get(
        "kind"
    ) != "research_sequential_candidate_search_receipt_v1":
        raise ValueError("unsupported sequential candidate search receipt")
    maximum = _require_int(core.get("maximumSolutions"), "maximumSolutions")
    emitted = _require_int(core.get("emittedCandidateCount"), "emittedCandidateCount")
    evaluated = _require_int(
        core.get("evaluatedCandidateCount"),
        "evaluatedCandidateCount",
    )
    _require_int(core.get("rootSeed"), "rootSeed")
    if not 1 <= maximum <= 100 or not 0 <= emitted == evaluated <= maximum:
        raise ValueError("candidate counts exceed maximumSolutions")
    if core.get("dealer") not in {"east", "south", "west", "north"}:
        raise ValueError("dealer must be a canonical seat")
    descriptor = dict(core.get("generatorDescriptor") or {})
    if (
        descriptor != OrToolsCpSatDealGeneratorV1.descriptor.to_dict()
        or canonical_sha256(descriptor) != _require_hash(
            core.get("generatorDescriptorHash"),
            "generatorDescriptorHash",
        )
    ):
        raise ValueError("generator descriptor integrity is invalid")
    search_request = _require_document_hash(
        core.get("searchRequest") or {},
        "search request",
    )
    _require_fields(
        search_request,
        "search request",
        (
            "schemaVersion", "kind", "rootSeed", "dealer", "rulesHash",
            "profileId", "profileRevision", "profileHash", "assumedBuyer",
            "maximumSolutions", "generatorDescriptorHash",
            "expectedSearchIdentityHash",
        ),
    )
    _require_int(search_request.get("rootSeed"), "search request rootSeed")
    _require_int(
        search_request.get("maximumSolutions"),
        "search request maximumSolutions",
    )
    for field in (
        "rootSeed", "dealer", "rulesHash", "profileId", "profileRevision",
        "profileHash", "assumedBuyer", "maximumSolutions", "generatorDescriptorHash",
    ):
        if search_request.get(field) != core.get(field):
            raise ValueError("search request root identity linkage is invalid")
    if (
        _require_int(
            search_request.get("schemaVersion"),
            "search request schemaVersion",
        )
        != 1
        or search_request.get("kind")
        != "research_sequential_candidate_search_request_v1"
    ):
        raise ValueError("unknown sequential candidate search request")
    for field in ("rulesHash", "profileHash"):
        _require_hash(core.get(field), field)
    primary: list[tuple[dict[str, Any], SequentialCandidateEvaluationV1]] = []
    repeats: list[tuple[dict[str, Any], SequentialCandidateEvaluationV1]] = []
    for value in list(core.get("evaluationReceipts") or ()):
        evaluation_receipt = _require_document_hash(value, "evaluation receipt")
        _require_fields(
            evaluation_receipt,
            "evaluation receipt",
            (
                "schemaVersion", "kind", "candidateOrdinal", "attempt",
                "candidateIntegrityCore", "candidateProvenance", "evaluation",
            ),
        )
        evaluation = _evaluation_from_document(evaluation_receipt["evaluation"])
        candidate_core = dict(evaluation_receipt.get("candidateIntegrityCore") or {})
        provenance = dict(evaluation_receipt.get("candidateProvenance") or {})
        _require_fields(
            candidate_core,
            "candidate integrity core",
            (
                "schemaVersion", "pluginVersion", "ordinal", "deckOrder",
                "deckCommitment", "orderingScore", "structuralObjectiveScore",
                "profileHash", "modelHash", "searchIdentityHash",
                "backendVersion", "solverStatus", "noGoodCutType",
                "precedingNoGoodCutHashes",
            ),
        )
        _require_fields(
            provenance,
            "candidate provenance",
            (
                "schemaVersion", "pluginId", "pluginVersion", "profileHash",
                "modelHash", "backendVersion", "solverStatus", "solutionOrdinal",
                "searchIdentityHash", "precedingNoGoodCutCount",
                "precedingNoGoodCutHashes", "noGoodCutType", "orderingScore",
                "structuralObjectiveScore", "deckCommitment", "candidateHash",
            ),
        )
        receipt_schema = _require_int(
            evaluation_receipt.get("schemaVersion"),
            "evaluation receipt schemaVersion",
        )
        candidate_schema = _require_int(
            candidate_core.get("schemaVersion"),
            "candidate integrity schemaVersion",
        )
        provenance_schema = _require_int(
            provenance.get("schemaVersion"),
            "candidate provenance schemaVersion",
        )
        candidate = _candidate_from_documents(candidate_core, provenance)
        ordinal = _require_int(
            evaluation_receipt.get("candidateOrdinal"),
            "candidateOrdinal",
        )
        if (
            receipt_schema != 1
            or evaluation_receipt.get("kind")
            != "research_sequential_candidate_evaluation_receipt_v1"
            or candidate_schema != 1
            or provenance_schema != 1
            or evaluation.candidate_hash != candidate.candidate_hash
            or ordinal != candidate.ordinal
            or any(_search_authority(candidate).get(field) != core.get(field) for field in (
                "profileHash", "modelHash", "searchIdentityHash", "backendVersion",
                "noGoodCutType", "structuralObjectiveScore",
            ))
        ):
            raise ValueError("evaluation candidate linkage is invalid")
        target = {
            "primary": primary,
            "qualified_repeat": repeats,
        }.get(evaluation_receipt["attempt"])
        if target is None:
            raise ValueError("unknown evaluation attempt")
        target.append((evaluation_receipt, evaluation))
    attempts = [item["attempt"] for item in core.get("evaluationReceipts") or ()]
    if (
        len(primary) != evaluated
        or len(repeats) > 1
        or attempts != ["primary"] * evaluated + ["qualified_repeat"] * len(repeats)
        or [item[0]["candidateOrdinal"] for item in primary] != list(range(evaluated))
    ):
        raise ValueError("evaluation receipt counts are invalid")
    expected_search_hash = search_request.get("expectedSearchIdentityHash")
    if expected_search_hash is not None:
        _require_hash(expected_search_hash, "expectedSearchIdentityHash")
    if core.get("assumedBuyer") not in {None, "east", "south", "west", "north"}:
        raise ValueError("search request assumed buyer is invalid")
    if evaluated or core.get("terminalContinuationReceipt") is not None:
        _require_hash(core.get("modelHash"), "modelHash")
        _require_hash(core.get("searchIdentityHash"), "searchIdentityHash")
        _require_int(
            core.get("structuralObjectiveScore"),
            "structuralObjectiveScore",
        )
        if (
            type(core.get("backendVersion")) is not str
            or not core["backendVersion"]
            or core.get("noGoodCutType") not in {
                "complete_deck_assignment_v1",
                "final_hand_allocation_v1",
            }
        ):
            raise ValueError("search backend or no-good authority is invalid")
    if evaluated or core.get("terminalContinuationReceipt") is not None:
        if (
            expected_search_hash is not None
            and core.get("searchIdentityHash") != expected_search_hash
        ):
            raise ValueError("search changed its expected identity")
    else:
        raise ValueError("empty search lost its conclusive initial solve receipt")
    candidate_hashes = [item[0]["candidateProvenance"]["candidateHash"] for item in primary]
    deck_hashes = [item[0]["candidateProvenance"]["deckCommitment"] for item in primary]
    if len(candidate_hashes) != len(set(candidate_hashes)) or len(deck_hashes) != len(
        set(deck_hashes)
    ):
        raise ValueError("evaluated candidates or decks are not distinct")
    search_identity_value = core.get("searchIdentityDocument")
    search_identity: dict[str, Any] | None = None
    layout_document: dict[str, Any] | None = None
    if evaluated or core.get("terminalContinuationReceipt") is not None:
        if not isinstance(search_identity_value, Mapping):
            raise ValueError("evaluated search lost its identity preimage")
        search_identity = _require_document_hash(
            search_identity_value,
            "search identity",
        )
        _require_int(
            search_identity.get("rootSeed"),
            "search identity rootSeed",
        )
        _require_fields(
            search_identity,
            "search identity",
            (
                "schemaVersion", "kind", "pluginId", "pluginVersion", "backend",
                "backendVersion", "profileId", "profileRevision", "profileHash",
                "modelHash", "rootSeed", "dealer", "rulesHash",
                "structuralDomainIrHash", "structuralDomain", "playDomainIrHash",
                "dealLayout", "noGoodCutType",
            ),
        )
        if (
            search_identity_value["sha256"] != core.get("searchIdentityHash")
            or _require_int(
                search_identity.get("schemaVersion"),
                "search identity schemaVersion",
            )
            != 1
            or search_identity.get("kind") != "cp_sat_deal_search_identity_v1"
            or search_identity.get("backend") != "ortools_cp_sat_v1"
            or search_identity.get("pluginId") != descriptor.get("id")
            or search_identity.get("pluginVersion") != descriptor.get("version")
            or any(search_identity.get(field) != core.get(field) for field in (
                "rootSeed", "dealer", "rulesHash", "profileId", "profileRevision",
                "profileHash", "modelHash", "backendVersion", "noGoodCutType",
            ))
        ):
            raise ValueError("search identity preimage linkage is invalid")
        _require_hash(
            search_identity.get("structuralDomainIrHash"),
            "structuralDomainIrHash",
        )
        _require_hash(search_identity.get("playDomainIrHash"), "playDomainIrHash")
        layout_document = _validate_layout_document(
            search_identity.get("dealLayout") or {},
            structural_domain_document=search_identity.get("structuralDomain") or {},
            dealer=str(core.get("dealer")),
            buyer=(
                str(core["assumedBuyer"])
                if core.get("assumedBuyer") is not None
                else None
            ),
        )
        if (
            layout_document.get("structuralDomainIrHash")
            != search_identity.get("structuralDomainIrHash")
        ):
            raise ValueError("search identity structural domain linkage is invalid")
    elif search_identity_value is not None:
        raise ValueError("empty search claimed an evaluated search identity")
    feedback = list(core.get("rejectionFeedbackRows") or ())
    rejected_count = _require_int(
        core.get("rejectedCandidateCount"),
        "rejectedCandidateCount",
    )
    if len(feedback) != rejected_count or rejected_count > evaluated:
        raise ValueError("rejection feedback count is invalid")
    feedback_rows: list[dict[str, Any]] = []
    for index, row_value in enumerate(feedback):
        row = copy.deepcopy(dict(row_value))
        state_hash = _require_hash(row.pop("stateHash", ""), "feedback stateHash")
        _require_fields(
            row,
            "rejection feedback",
            (
                "schemaVersion", "kind", "rejectionOrdinal", "candidateHash",
                "evaluationReceiptHash", "evaluationEvidenceHashes", "failureCodes",
                "noGoodCut", "feedbackHash", "askedForNextSolution",
                "appliedToSubsequentSolve", "noGoodCutHash", "applicationReceipt",
            ),
        )
        if canonical_sha256(row) != state_hash:
            raise ValueError("feedback state integrity is invalid")
        if (
            _require_int(row.get("schemaVersion"), "feedback schemaVersion") != 1
            or row.get("kind") != "research_exact_candidate_no_good_feedback_v1"
        ):
            raise ValueError("unsupported rejection feedback contract")
        feedback_core = _feedback_core(row)
        _require_int(row.get("rejectionOrdinal"), "rejectionOrdinal")
        if canonical_sha256(feedback_core) != row["feedbackHash"]:
            raise ValueError("feedback integrity is invalid")
        primary_receipt, primary_evaluation = primary[index]
        provenance = primary_receipt["candidateProvenance"]
        cut_hash = _validate_no_good_cut_document(
            row["noGoodCut"],
            provenance,
            primary_receipt["candidateIntegrityCore"],
            expected_dealer=str(core.get("dealer")),
            expected_buyer=(
                str(core["assumedBuyer"])
                if core.get("assumedBuyer") is not None
                else None
            ),
            expected_final_positions=(
                layout_document.get("finalHandPositions")
                if layout_document is not None
                else None
            ),
        )
        if (
            row.get("rejectionOrdinal") != index
            or row.get("candidateHash") != provenance["candidateHash"]
            or row.get("evaluationReceiptHash")
            != dict(core["evaluationReceipts"][index])["sha256"]
            or row.get("evaluationEvidenceHashes")
            != list(primary_evaluation.evaluation_evidence_hashes)
            or row.get("failureCodes") != list(primary_evaluation.failure_codes)
            or primary_evaluation.outcome
            != SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION
            or provenance["precedingNoGoodCutHashes"]
            != [prior["noGoodCut"]["cutSha256"] for prior in feedback_rows]
            or row.get("askedForNextSolution") is not True
        ):
            raise ValueError("feedback evaluation or cut linkage is invalid")
        applied = row.get("applicationReceipt") is not None
        if row.get("appliedToSubsequentSolve") is not applied or (
            applied and row.get("noGoodCutHash") != cut_hash
        ) or (not applied and row.get("noGoodCutHash") is not None):
            raise ValueError("feedback application state is invalid")
        if row.get("applicationReceipt") is not None:
            application_core = _require_document_hash(
                row["applicationReceipt"],
                "application receipt",
            )
            _require_fields(
                application_core,
                "application receipt",
                (
                    "schemaVersion", "kind", "feedbackHash", "noGoodCutHash",
                    "solveEvidenceKind", "solveEvidenceHash", "solveOrdinal",
                    "solverStatus", "nextCandidateHash",
                ),
            )
            if (
                _require_int(
                    application_core.get("schemaVersion"),
                    "application schemaVersion",
                )
                != 1
                or application_core.get("kind")
                != "research_exact_candidate_no_good_application_v1"
                or application_core["feedbackHash"] != row["feedbackHash"]
                or application_core["noGoodCutHash"] != row["noGoodCutHash"]
                or _require_int(
                    application_core.get("solveOrdinal"),
                    "application solveOrdinal",
                )
                != index + 1
            ):
                raise ValueError("application receipt linkage is invalid")
        feedback_rows.append({**row, "stateHash": state_hash})
    expected_cuts = [row["noGoodCut"]["cutSha256"] for row in feedback_rows]
    for ordinal, (evaluation_receipt, _) in enumerate(primary):
        if evaluation_receipt["candidateProvenance"][
            "precedingNoGoodCutHashes"
        ] != expected_cuts[:ordinal]:
            raise ValueError("candidate lost the complete exact no-good cut prefix")
    continuation = core.get("terminalContinuationReceipt")
    continuation_core = None
    if continuation is not None:
        continuation_core = _validate_continuation_document(continuation)
        if continuation_core["precedingNoGoodCutHashes"] != [
            row["noGoodCut"]["cutSha256"] for row in feedback_rows
        ]:
            raise ValueError("continuation cut linkage is invalid")
        if any(continuation_core[field] != core[field] for field in (
            "profileHash", "modelHash", "searchIdentityHash", "backendVersion",
            "noGoodCutType", "structuralObjectiveScore",
        )):
            raise ValueError("continuation authority linkage is invalid")
        if (
            continuation_core.get("pluginId") != descriptor.get("id")
            or continuation_core.get("pluginVersion") != descriptor.get("version")
            or any(
                continuation_core.get("witnessCandidateHash")
                == item[0]["candidateProvenance"]["candidateHash"]
                or continuation_core.get("witnessDeckCommitment")
                == item[0]["candidateProvenance"]["deckCommitment"]
                for item in primary
            )
        ):
            raise ValueError("continuation plugin or witness identity is invalid")
    for index, row in enumerate(feedback_rows):
        application = row.get("applicationReceipt")
        if application is None:
            raise ValueError("ordinary rejection lost its no-good application receipt")
        application_core = dict(application)
        kind = application_core.get("solveEvidenceKind")
        if kind == "next_candidate":
            if index + 1 >= len(primary):
                raise ValueError("no-good application lost its next candidate")
            next_provenance = primary[index + 1][0]["candidateProvenance"]
            if (
                application_core.get("solverStatus") != "optimal"
                or application_core.get("solveEvidenceHash")
                != next_provenance["candidateHash"]
                or application_core.get("nextCandidateHash")
                != next_provenance["candidateHash"]
                or next_provenance["precedingNoGoodCutHashes"][index]
                != row["noGoodCut"]["cutSha256"]
            ):
                raise ValueError("next-candidate application evidence is invalid")
        elif kind == "post_rejection_continuation_receipt":
            if (
                continuation_core is None
                or index != len(feedback_rows) - 1
                or application_core.get("solveEvidenceHash") != continuation["sha256"]
                or application_core.get("solverStatus") != continuation_core["solverStatus"]
                or application_core.get("nextCandidateHash")
                != continuation_core["witnessCandidateHash"]
            ):
                raise ValueError("continuation application evidence is invalid")
        else:
            raise ValueError("unknown no-good application evidence kind")
    qualified_hash = core.get("qualifiedCandidateHash")
    stop_reason = str(core.get("stopReason"))
    outcomes = [item[1].outcome for item in primary]
    if outcomes[:rejected_count] != [
        SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION
    ] * rejected_count or len(outcomes) - rejected_count > 1:
        raise ValueError("primary evaluation outcome order is invalid")
    terminal = primary[-1][1] if rejected_count < evaluated else None
    synthetic_failures: set[str] = set()
    if repeats:
        repeated_receipt, repeated = repeats[0]
        final_receipt, final_evaluation = primary[-1]
        if (
            final_evaluation.outcome != SequentialCandidateOutcomeKindV1.QUALIFIED
            or repeated_receipt["candidateOrdinal"] != final_receipt["candidateOrdinal"]
            or repeated_receipt["candidateIntegrityCore"]
            != final_receipt["candidateIntegrityCore"]
            or repeated_receipt["candidateProvenance"]
            != final_receipt["candidateProvenance"]
        ):
            raise ValueError("qualified repeat candidate linkage is invalid")
        if repeated == final_evaluation:
            expected_stop = "qualified"
        elif repeated.outcome == SequentialCandidateOutcomeKindV1.CANCELLED:
            expected_stop = "cancelled"
            synthetic_failures.add("qualified_repeat_cancelled")
        elif repeated.outcome == SequentialCandidateOutcomeKindV1.INCOMPLETE:
            expected_stop = "incomplete"
            synthetic_failures.add("qualified_repeat_incomplete")
        else:
            expected_stop = "fatal_quarantine"
            synthetic_failures.add("qualified_repeat_mismatch")
    elif terminal is not None:
        if terminal.outcome == SequentialCandidateOutcomeKindV1.QUALIFIED:
            raise ValueError("qualified primary evaluation lost its exact repeat")
        expected_stop = terminal.outcome.value
    elif stop_reason == "maximum_solutions_reached":
        if (
            evaluated != maximum
            or rejected_count != evaluated
            or continuation_core is None
            or continuation_core["exhausted"]
        ):
            raise ValueError("maximumSolutions stop evidence is invalid")
        expected_stop = stop_reason
        synthetic_failures.add("maximum_solutions_reached")
    elif stop_reason == "candidates_exhausted":
        if (
            rejected_count != evaluated
            or continuation_core is None
            or not continuation_core["exhausted"]
        ):
            raise ValueError("candidate exhaustion evidence is invalid")
        expected_stop = stop_reason
        synthetic_failures.add("candidate_space_exhausted")
    else:
        raise ValueError("unknown sequential candidate stop reason")
    if stop_reason != expected_stop or (stop_reason != "qualified" and qualified_hash is not None):
        raise ValueError("terminal stop reason is not linked to its exact evaluation")
    if stop_reason == "qualified":
        expected_hash = primary[-1][0]["candidateProvenance"]["candidateHash"]
        if qualified_hash != expected_hash or continuation_core is not None:
            raise ValueError("qualified result identity is invalid")
    elif terminal is not None and continuation_core is not None:
        raise ValueError("typed terminal outcome cannot carry a continuation witness")
    failures = list(core.get("failureCodes") or ())
    evaluation_failures = {
        code for _, evaluation in (*primary, *repeats) for code in evaluation.failure_codes
    }
    if failures != sorted(evaluation_failures | synthetic_failures):
        raise ValueError("receipt failure code linkage is invalid")


__all__ = [
    "SequentialCandidateEvaluationV1",
    "SequentialCandidateOutcomeKindV1",
    "SequentialCandidateSearchReceiptV1",
    "SequentialCandidateSearchResultV1",
    "run_sequential_candidate_search_v1",
]
