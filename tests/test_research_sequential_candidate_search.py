from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from baloot_platform.challenge.constraint_deal_search import (
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.config import build_bundle
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.research.sequential_candidate_search import (
    SequentialCandidateEvaluationV1,
    SequentialCandidateOutcomeKindV1,
    SequentialCandidateSearchReceiptV1,
    run_sequential_candidate_search_v1,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_RULES = build_bundle().rules


def _spec(
    *,
    maximum_solutions: int = 2,
    assumed_buyer: str | None = None,
) -> DealConstraintSpecV1:
    return DealConstraintSpecV1.build({
        "schemaVersion": 1,
        "id": "sequential_product_test_v1",
        "revision": "1",
        "enabled": True,
        "backend": "ortools_cp_sat_v1",
        "assumedBuyer": assumed_buyer,
        "purchaseSuits": [],
        "initialRequiredCards": {},
        "finalRequiredCards": {},
        "finalForbiddenCards": {},
        "initialSuitCountRanges": {},
        "finalSuitCountRanges": {},
        "projectPattern": None,
        "maximumSolutions": maximum_solutions,
        "maximumConflicts": 1000,
        "safetyTimeoutSeconds": 1.0,
        "solverWorkers": 1,
        "seedOffset": 0,
        "tags": ["engine_validation_required"],
    })


def _search_hash(
    spec: DealConstraintSpecV1,
    *,
    root_seed: int,
    dealer: Seat,
    no_good_cut_type: str = "complete_deck_assignment_v1",
    rules=_RULES,
) -> str:
    return OrToolsCpSatDealGeneratorV1.search_identity_document(
        spec=spec,
        model_hash=_hash("model"),
        root_seed=root_seed,
        dealer=dealer,
        rules=rules,
        backend_version="9.15.6755",
        no_good_cut_type=no_good_cut_type,
    )["sha256"]


def _candidate(
    spec: DealConstraintSpecV1,
    ordinal: int,
    cuts: tuple[str, ...],
    *,
    search_hash: str | None = None,
    profile_hash: str | None = None,
    no_good_cut_type: str = "complete_deck_assignment_v1",
):
    deck = tuple(range(ordinal, 32)) + tuple(range(ordinal))
    return OrToolsCpSatDealGeneratorV1._build_candidate(
        ordinal=ordinal,
        deck_order=deck,
        deck_commitment=hashlib.sha256(bytes(deck)).hexdigest(),
        ordering_score=900,
        structural_objective_score=42,
        profile_hash=profile_hash or spec.sha256,
        model_hash=_hash("model"),
        search_identity_hash=search_hash or _hash("search"),
        backend_version="9.15.6755",
        no_good_cut_type=no_good_cut_type,
        preceding_no_good_cut_hashes=cuts,
    )


class _Generator:
    descriptor = OrToolsCpSatDealGeneratorV1.descriptor

    def __init__(
        self,
        spec: DealConstraintSpecV1,
        *,
        candidate_count: int,
        root_seed: int,
        dealer: Seat,
        continuation_exhausted: bool = False,
        no_good_cut_type: str = "complete_deck_assignment_v1",
        rules=_RULES,
    ):
        search_hash = _search_hash(
            spec,
            root_seed=root_seed,
            dealer=dealer,
            no_good_cut_type=no_good_cut_type,
            rules=rules,
        )
        layout = (
            DealLayoutV1.build(
                rules.structural_domain,
                dealer=dealer,
                buyer=spec.assumed_buyer,
            )
            if no_good_cut_type == "final_hand_allocation_v1"
            else None
        )
        cuts: list[str] = []
        candidates = []
        for index in range(candidate_count):
            candidate = _candidate(
                spec,
                index,
                tuple(cuts),
                search_hash=search_hash,
                no_good_cut_type=no_good_cut_type,
            )
            candidates.append(candidate)
            cuts.append(
                OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
                    candidate,
                    layout=layout,
                )["cutSha256"]
            )
        self.cuts = tuple(cuts)
        self.candidates = tuple(candidates)
        self.witness = _candidate(
            spec,
            candidate_count,
            self.cuts,
            search_hash=search_hash,
            no_good_cut_type=no_good_cut_type,
        )
        self.continuation_exhausted = continuation_exhausted
        self._authority_candidate = self.witness
        self.generate_calls: list[tuple[str, ...]] = []
        self.probe_calls: list[tuple[str, ...]] = []

    def generate_next(self, _spec, *, rejected_candidates, **_kwargs):
        self.generate_calls.append(
            tuple(candidate.candidate_hash for candidate in rejected_candidates)
        )
        ordinal = len(rejected_candidates)
        return self.candidates[ordinal] if ordinal < len(self.candidates) else None

    def probe_continuation(self, _spec, *, rejected_candidates, **_kwargs):
        self.probe_calls.append(
            tuple(candidate.candidate_hash for candidate in rejected_candidates)
        )
        prefix = self.cuts[:len(rejected_candidates)]
        exhausted = self.continuation_exhausted or not self.candidates
        authority = rejected_candidates[0] if rejected_candidates else self._authority_candidate
        return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
            profile_hash=authority.profile_hash,
            model_hash=authority.model_hash,
            backend_version=authority.backend_version,
            solver_status="infeasible" if exhausted else "optimal",
            solution_ordinal=len(rejected_candidates),
            search_identity_hash=authority.search_identity_hash,
            preceding_no_good_cut_hashes=prefix,
            no_good_cut_type=authority.no_good_cut_type,
            exhausted=exhausted,
            witness=None if exhausted else self.witness,
            structural_objective_score=authority.structural_objective_score,
        )

def _evaluation(candidate, outcome, *, suffix: str = "stable", failures=()):
    return SequentialCandidateEvaluationV1.build(
        outcome=outcome,
        candidate_hash=candidate.candidate_hash,
        evaluation_evidence_hashes=(_hash(f"evidence-{candidate.ordinal}-{suffix}"),),
        failure_codes=tuple(failures),
    )


def test_ordinary_rejection_applies_exact_cut_then_qualified_repeat_is_exact():
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=2, root_seed=17, dealer=Seat.EAST)
    calls: list[str] = []

    def evaluate(candidate, _index):
        calls.append(candidate.candidate_hash)
        if candidate.ordinal == 0:
            return _evaluation(
                candidate,
                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                failures=("not_interesting",),
            )
        return _evaluation(candidate, SequentialCandidateOutcomeKindV1.QUALIFIED)

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=17,
        dealer=Seat.EAST,
        rules=_RULES,
        evaluate_candidate=evaluate,
        expected_search_identity_hash=generator.candidates[0].search_identity_hash,
    )
    document = result.receipt.verify().to_dict()

    assert result.qualified_candidate == generator.candidates[1]
    assert result.qualified_evaluation is not None
    assert calls == [
        generator.candidates[0].candidate_hash,
        generator.candidates[1].candidate_hash,
        generator.candidates[1].candidate_hash,
    ]
    assert generator.generate_calls == [(), (generator.candidates[0].candidate_hash,)]
    assert generator.probe_calls == []
    assert document["stopReason"] == "qualified"
    assert document["evaluatedCandidateCount"] == 2
    assert document["emittedCandidateCount"] == 2
    assert [row["attempt"] for row in document["evaluationReceipts"]] == [
        "primary",
        "primary",
        "qualified_repeat",
    ]
    feedback = document["rejectionFeedbackRows"][0]
    assert feedback["noGoodCutHash"] == generator.cuts[0]
    assert feedback["applicationReceipt"]["solveEvidenceKind"] == "next_candidate"
    assert feedback["applicationReceipt"]["nextCandidateHash"] == (
        generator.candidates[1].candidate_hash
    )

    rerun = run_sequential_candidate_search_v1(
        _Generator(spec, candidate_count=2, root_seed=17, dealer=Seat.EAST),
        spec,
        root_seed=17,
        dealer=Seat.EAST,
        rules=_RULES,
        evaluate_candidate=evaluate,
        expected_search_identity_hash=generator.candidates[0].search_identity_hash,
    )
    assert rerun.receipt.sha256 == result.receipt.sha256


@pytest.mark.parametrize(
    ("exhausted", "stop_reason"),
    [(False, "maximum_solutions_reached"), (True, "candidates_exhausted")],
)
def test_maximum_solutions_is_distinct_candidate_cap_and_probe_is_evidence_only(
    exhausted: bool,
    stop_reason: str,
):
    spec = _spec(maximum_solutions=2)
    generator = _Generator(
        spec,
        candidate_count=2,
        root_seed=18,
        dealer=Seat.NORTH,
        continuation_exhausted=exhausted,
    )
    evaluated: list[str] = []

    def evaluate(candidate, _index):
        evaluated.append(candidate.candidate_hash)
        return _evaluation(
            candidate,
            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
            failures=(f"rejected_{candidate.ordinal}",),
        )

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=18,
        dealer=Seat.NORTH,
        rules=_RULES,
        evaluate_candidate=evaluate,
    )
    document = result.receipt.to_dict()

    assert result.qualified_candidate is None
    assert evaluated == [candidate.candidate_hash for candidate in generator.candidates]
    assert generator.generate_calls == [(), (generator.candidates[0].candidate_hash,)]
    assert generator.probe_calls == [
        tuple(candidate.candidate_hash for candidate in generator.candidates)
    ]
    assert generator.witness.candidate_hash not in evaluated
    assert document["evaluatedCandidateCount"] == spec.maximum_solutions
    assert document["emittedCandidateCount"] == spec.maximum_solutions
    assert document["stopReason"] == stop_reason
    continuation = document["terminalContinuationReceipt"]
    assert continuation["candidateEmitted"] is False
    assert continuation["candidateEvaluated"] is False
    assert continuation["candidatePublished"] is False
    final = document["rejectionFeedbackRows"][-1]
    assert final["applicationReceipt"]["solveEvidenceKind"] == (
        "post_rejection_continuation_receipt"
    )
    assert final["applicationReceipt"]["solveEvidenceHash"] == continuation["sha256"]


@pytest.mark.parametrize(
    "outcome",
    [
        SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
        SequentialCandidateOutcomeKindV1.INCOMPLETE,
        SequentialCandidateOutcomeKindV1.CANCELLED,
    ],
)
def test_nonordinary_terminal_outcome_stops_without_cut_or_probe(outcome):
    spec = _spec(maximum_solutions=3)
    generator = _Generator(spec, candidate_count=3, root_seed=19, dealer=Seat.SOUTH)

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=19,
        dealer=Seat.SOUTH,
        rules=_RULES,
        evaluate_candidate=lambda candidate, _index: _evaluation(
            candidate,
            outcome,
            failures=(f"{outcome.value}_failure",),
        ),
    )
    document = result.receipt.to_dict()

    assert result.qualified_candidate is None
    assert document["stopReason"] == outcome.value
    assert document["rejectionFeedbackRows"] == []
    assert document["terminalContinuationReceipt"] is None
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []


def test_qualified_repeat_mismatch_fails_closed_without_cut():
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=2, root_seed=20, dealer=Seat.WEST)
    call_count = 0

    def evaluate(candidate, _index):
        nonlocal call_count
        call_count += 1
        return _evaluation(
            candidate,
            SequentialCandidateOutcomeKindV1.QUALIFIED,
            suffix=str(call_count),
        )

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=20,
        dealer=Seat.WEST,
        rules=_RULES,
        evaluate_candidate=evaluate,
    )
    document = result.receipt.to_dict()

    assert result.qualified_candidate is None
    assert document["stopReason"] == "fatal_quarantine"
    assert "qualified_repeat_mismatch" in document["failureCodes"]
    assert document["rejectionFeedbackRows"] == []
    assert generator.probe_calls == []


def test_empty_structural_search_emits_canonical_exhaustion_receipt():
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=0, root_seed=21, dealer=Seat.SOUTH)

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=21,
        dealer=Seat.SOUTH,
        rules=_RULES,
        evaluate_candidate=lambda *_args: pytest.fail("no candidate may be evaluated"),
    )
    document = result.receipt.verify().to_dict()

    assert document["stopReason"] == "candidates_exhausted"
    assert document["emittedCandidateCount"] == 0
    assert document["evaluationReceipts"] == []
    assert document["terminalContinuationReceipt"]["exhausted"] is True
    assert generator.probe_calls == [()]


def test_candidate_profile_order_search_and_integrity_fail_closed():
    spec = _spec(maximum_solutions=2)
    valid_search_hash = _search_hash(spec, root_seed=22, dealer=Seat.SOUTH)
    unknown_cut = replace(
        _candidate(spec, 0, (), search_hash=valid_search_hash),
        no_good_cut_type="unknown_cut_v1",
        candidate_hash="",
    )
    unknown_cut = replace(
        unknown_cut,
        candidate_hash=canonical_sha256(unknown_cut.integrity_core()),
    )
    cases = [
        _candidate(
            spec,
            0,
            (),
            search_hash=valid_search_hash,
            profile_hash=_hash("foreign-profile"),
        ),
        _candidate(
            spec,
            1,
            (_hash("prior-cut"),),
            search_hash=valid_search_hash,
        ),
        _candidate(spec, 0, (), search_hash=_hash("foreign-search")),
        unknown_cut,
        replace(
            _candidate(spec, 0, (), search_hash=valid_search_hash),
            candidate_hash=_hash("tampered"),
        ),
    ]
    expected_messages = [
        "profile", "ordinal", "expected search", "cut authority", "integrity hash",
    ]

    for candidate, expected in zip(cases, expected_messages, strict=True):
        generator = _Generator(spec, candidate_count=1, root_seed=22, dealer=Seat.SOUTH)
        generator.candidates = (candidate,)
        with pytest.raises(ValueError, match=expected):
            run_sequential_candidate_search_v1(
                generator,
                spec,
                root_seed=22,
                dealer=Seat.SOUTH,
                rules=_RULES,
                evaluate_candidate=lambda current, _index: _evaluation(
                    current,
                    SequentialCandidateOutcomeKindV1.QUALIFIED,
                ),
                expected_search_identity_hash=valid_search_hash,
            )


def test_cut_and_continuation_link_tampering_fail_closed():
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=2, root_seed=23, dealer=Seat.SOUTH)
    generator.candidates = (
        generator.candidates[0],
        _candidate(
            spec,
            1,
            (_hash("wrong-cut"),),
            search_hash=_search_hash(spec, root_seed=23, dealer=Seat.SOUTH),
        ),
    )

    with pytest.raises(ValueError, match="cut"):
        run_sequential_candidate_search_v1(
            generator,
            spec,
            root_seed=23,
            dealer=Seat.SOUTH,
            rules=_RULES,
            evaluate_candidate=lambda candidate, _index: _evaluation(
                candidate,
                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                failures=("ordinary",),
            ),
        )


def test_receipt_rejects_hash_tampering_and_canonically_rehashed_link_tampering():
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=2, root_seed=24, dealer=Seat.SOUTH)

    def evaluate(candidate, _index):
        if candidate.ordinal == 0:
            return _evaluation(
                candidate,
                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                failures=("ordinary",),
            )
        return _evaluation(candidate, SequentialCandidateOutcomeKindV1.QUALIFIED)

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=24,
        dealer=Seat.SOUTH,
        rules=_RULES,
        evaluate_candidate=evaluate,
    )
    with pytest.raises(ValueError, match="integrity"):
        SequentialCandidateSearchReceiptV1.from_dict({
            **result.receipt.to_dict(),
            "sha256": _hash("wrong"),
        })

    tampered = result.receipt.to_dict()
    tampered.pop("sha256")
    tampered["rejectionFeedbackRows"][0]["noGoodCutHash"] = _hash("foreign")
    tampered["sha256"] = canonical_sha256(tampered)
    with pytest.raises(ValueError, match="feedback state integrity"):
        SequentialCandidateSearchReceiptV1.from_dict(tampered)


def test_evaluation_contract_requires_complete_rejection_and_exact_candidate_link():
    candidate_hash = _hash("candidate")
    with pytest.raises(ValueError, match="evaluation evidence"):
        SequentialCandidateEvaluationV1.build(
            outcome=SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
            candidate_hash=candidate_hash,
            evaluation_evidence_hashes=(),
            failure_codes=("ordinary",),
        )
    with pytest.raises(ValueError, match="failure code"):
        SequentialCandidateEvaluationV1.build(
            outcome=SequentialCandidateOutcomeKindV1.INCOMPLETE,
            candidate_hash=candidate_hash,
            evaluation_evidence_hashes=(),
        )

    spec = _spec(maximum_solutions=1)
    generator = _Generator(spec, candidate_count=1, root_seed=25, dealer=Seat.SOUTH)
    with pytest.raises(ValueError, match="candidate identity"):
        run_sequential_candidate_search_v1(
            generator,
            spec,
            root_seed=25,
            dealer=Seat.SOUTH,
            rules=_RULES,
            evaluate_candidate=lambda _candidate, _index: (
                SequentialCandidateEvaluationV1.build(
                    outcome=SequentialCandidateOutcomeKindV1.QUALIFIED,
                    candidate_hash=_hash("foreign"),
                    evaluation_evidence_hashes=(_hash("evidence"),),
                )
            ),
        )


def _rehash(document: dict, field: str = "sha256") -> None:
    document.pop(field, None)
    document[field] = canonical_sha256({
        key: value for key, value in document.items() if key != field
    })


def _rehash_feedback(row: dict) -> None:
    row["feedbackHash"] = canonical_sha256({
        field: row[field]
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
    })
    _rehash(row, "stateHash")


def _qualified_receipt() -> dict:
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=2, root_seed=28, dealer=Seat.EAST)

    def evaluate(candidate, _index):
        if candidate.ordinal == 0:
            return _evaluation(
                candidate,
                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                failures=("ordinary",),
            )
        return _evaluation(candidate, SequentialCandidateOutcomeKindV1.QUALIFIED)

    return run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=28,
        dealer=Seat.EAST,
        rules=_RULES,
        evaluate_candidate=evaluate,
    ).receipt.to_dict()


def _rewrite_candidate_receipts(
    document: dict,
    ordinal: int,
    *,
    core_updates: dict,
    provenance_updates: dict | None = None,
) -> tuple[str, str]:
    matching = [
        receipt
        for receipt in document["evaluationReceipts"]
        if receipt["candidateOrdinal"] == ordinal
    ]
    old_hash = matching[0]["candidateProvenance"]["candidateHash"]
    new_hash = ""
    for receipt in matching:
        receipt["candidateIntegrityCore"].update(core_updates)
        new_hash = canonical_sha256(receipt["candidateIntegrityCore"])
        receipt["candidateProvenance"].update(provenance_updates or {})
        receipt["candidateProvenance"]["candidateHash"] = new_hash
        receipt["evaluation"]["candidateHash"] = new_hash
        _rehash(receipt["evaluation"])
        _rehash(receipt)
    if document["qualifiedCandidateHash"] == old_hash:
        document["qualifiedCandidateHash"] = new_hash
    return old_hash, new_hash


def _sync_terminal_continuation(document: dict) -> None:
    continuation = document["terminalContinuationReceipt"]
    _rehash(continuation)
    row = document["rejectionFeedbackRows"][-1]
    application = row["applicationReceipt"]
    application["solveEvidenceHash"] = continuation["sha256"]
    application["nextCandidateHash"] = continuation["witnessCandidateHash"]
    _rehash(application)
    _rehash(row, "stateHash")


def _bounded_rejection_receipt():
    spec = _spec(maximum_solutions=2)
    generator = _Generator(spec, candidate_count=2, root_seed=26, dealer=Seat.SOUTH)
    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=26,
        dealer=Seat.SOUTH,
        rules=_RULES,
        evaluate_candidate=lambda candidate, _index: _evaluation(
            candidate,
            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
            failures=(f"rejected_{candidate.ordinal}",),
        ),
    )
    return result.receipt.to_dict()


@pytest.mark.parametrize(
    "mutation",
    ["outcome", "stop", "application", "continuation_exposure"],
)
def test_canonically_rehashed_semantic_and_link_tampering_fails_closed(mutation):
    document = _bounded_rejection_receipt()
    if mutation == "outcome":
        evaluation_receipt = document["evaluationReceipts"][0]
        evaluation = evaluation_receipt["evaluation"]
        evaluation["outcome"] = "cancelled"
        _rehash(evaluation)
        _rehash(evaluation_receipt)
    elif mutation == "stop":
        document["stopReason"] = "cancelled"
        document["failureCodes"] = []
    elif mutation == "application":
        row = document["rejectionFeedbackRows"][-1]
        application = row["applicationReceipt"]
        application["solveEvidenceHash"] = _hash("foreign-evidence")
        _rehash(application)
        _rehash(row, "stateHash")
    else:
        continuation = document["terminalContinuationReceipt"]
        continuation["candidateEvaluated"] = True
        _rehash(continuation)
    _rehash(document)

    with pytest.raises(ValueError):
        SequentialCandidateSearchReceiptV1.from_dict(document)


def test_runtime_continuation_rejects_canonically_hashed_impossible_status():
    spec = _spec(maximum_solutions=1)
    generator = _Generator(spec, candidate_count=1, root_seed=27, dealer=Seat.SOUTH)
    original_probe = generator.probe_continuation

    def invalid_probe(*args, **kwargs):
        receipt = original_probe(*args, **kwargs)
        invalid = replace(receipt, solver_status="garbage", receipt_hash="")
        return replace(
            invalid,
            receipt_hash=canonical_sha256(invalid.integrity_core()),
        )

    generator.probe_continuation = invalid_probe
    with pytest.raises(ValueError, match="continuation receipt"):
        run_sequential_candidate_search_v1(
            generator,
            spec,
            root_seed=27,
            dealer=Seat.SOUTH,
            rules=_RULES,
            evaluate_candidate=lambda candidate, _index: _evaluation(
                candidate,
                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                failures=("ordinary",),
            ),
        )


def test_direct_noncanonical_evaluation_construction_fails_verification():
    candidate_hash = _hash("direct-candidate")
    core = {
        "schemaVersion": 1,
        "kind": "research_sequential_candidate_evaluation_v1",
        "outcome": "ordinary_rejection",
        "candidateHash": candidate_hash,
        "evaluationEvidenceHashes": [_hash("z"), _hash("a"), _hash("z")],
        "failureCodes": ["z_failure", "a_failure", "z_failure"],
    }
    evaluation = SequentialCandidateEvaluationV1(
        SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
        candidate_hash,
        tuple(core["evaluationEvidenceHashes"]),
        tuple(core["failureCodes"]),
        canonical_sha256(core),
    )

    with pytest.raises(ValueError, match="canonical"):
        evaluation.verify()


@pytest.mark.parametrize(
    ("contract", "field"),
    [
        ("root", "schemaVersion"),
        ("root", "kind"),
        ("search_request", "schemaVersion"),
        ("search_request", "kind"),
        ("search_identity", "schemaVersion"),
        ("search_identity", "kind"),
        ("evaluation_receipt", "schemaVersion"),
        ("evaluation_receipt", "kind"),
        ("candidate_core", "schemaVersion"),
        ("candidate_provenance", "schemaVersion"),
        ("evaluation", "schemaVersion"),
        ("evaluation", "kind"),
        ("feedback", "schemaVersion"),
        ("feedback", "kind"),
        ("no_good_cut", "schemaVersion"),
        ("no_good_cut", "kind"),
        ("application", "schemaVersion"),
        ("application", "kind"),
        ("continuation", "schemaVersion"),
        ("continuation", "kind"),
    ],
)
def test_canonically_rehashed_nested_contract_relabels_fail_closed(
    contract: str,
    field: str,
):
    document = (
        _qualified_receipt()
        if contract in {
            "evaluation_receipt",
            "candidate_core",
            "candidate_provenance",
            "evaluation",
        }
        else _bounded_rejection_receipt()
    )
    foreign = 99 if field == "schemaVersion" else "foreign_contract_v1"
    if contract == "root":
        document[field] = foreign
    elif contract == "search_request":
        document["searchRequest"][field] = foreign
        _rehash(document["searchRequest"])
    elif contract == "search_identity":
        document["searchIdentityDocument"][field] = foreign
        _rehash(document["searchIdentityDocument"])
    elif contract == "evaluation_receipt":
        document["evaluationReceipts"][1][field] = foreign
        _rehash(document["evaluationReceipts"][1])
    elif contract == "candidate_core":
        _rewrite_candidate_receipts(
            document,
            1,
            core_updates={field: foreign},
        )
    elif contract == "candidate_provenance":
        for receipt in document["evaluationReceipts"]:
            if receipt["candidateOrdinal"] == 1:
                receipt["candidateProvenance"][field] = foreign
                _rehash(receipt)
    elif contract == "evaluation":
        receipt = document["evaluationReceipts"][1]
        receipt["evaluation"][field] = foreign
        _rehash(receipt["evaluation"])
        _rehash(receipt)
    elif contract == "feedback":
        row = document["rejectionFeedbackRows"][0]
        row[field] = foreign
        _rehash_feedback(row)
    elif contract == "no_good_cut":
        row = document["rejectionFeedbackRows"][0]
        cut = row["noGoodCut"]
        cut[field] = foreign
        cut["cutSha256"] = canonical_sha256({
            key: value
            for key, value in cut.items()
            if key not in {
                "sha256",
                "cutSha256",
                "validationFinalHandPositions",
                "validationDealer",
                "validationBuyer",
            }
        })
        _rehash(cut)
        _rehash_feedback(row)
    elif contract == "application":
        row = document["rejectionFeedbackRows"][0]
        row["applicationReceipt"][field] = foreign
        _rehash(row["applicationReceipt"])
        _rehash(row, "stateHash")
    else:
        document["terminalContinuationReceipt"][field] = foreign
        _sync_terminal_continuation(document)
    _rehash(document)

    with pytest.raises(ValueError):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [("deterministic", False), ("kind", "renderer")],
)
def test_canonically_rehashed_foreign_generator_descriptor_fails_closed(field, value):
    document = _bounded_rejection_receipt()
    document["generatorDescriptor"][field] = value
    descriptor_hash = canonical_sha256(document["generatorDescriptor"])
    document["generatorDescriptorHash"] = descriptor_hash
    document["searchRequest"]["generatorDescriptorHash"] = descriptor_hash
    _rehash(document["searchRequest"])
    _rehash(document)

    with pytest.raises(ValueError, match="descriptor"):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    ["plugin_version", "ordering_score", "structural_score", "solver_status", "deck"],
)
def test_canonically_rehashed_candidate_authority_tampering_fails_closed(mutation):
    document = _qualified_receipt()
    second = document["evaluationReceipts"][1]
    first = document["evaluationReceipts"][0]
    if mutation == "plugin_version":
        core_updates = {"pluginVersion": "999.0.0"}
        provenance_updates = {}
    elif mutation == "ordering_score":
        core_updates = {"orderingScore": second["candidateIntegrityCore"]["orderingScore"] + 1}
        provenance_updates = {}
    elif mutation == "structural_score":
        score = second["candidateIntegrityCore"]["structuralObjectiveScore"] + 1
        core_updates = {"structuralObjectiveScore": score}
        provenance_updates = {"structuralObjectiveScore": score}
    elif mutation == "solver_status":
        core_updates = {"solverStatus": "feasible"}
        provenance_updates = {"solverStatus": "feasible"}
    else:
        core_updates = {
            "deckOrder": first["candidateIntegrityCore"]["deckOrder"],
            "deckCommitment": first["candidateProvenance"]["deckCommitment"],
        }
        provenance_updates = {
            "deckCommitment": first["candidateProvenance"]["deckCommitment"],
        }
    _rewrite_candidate_receipts(
        document,
        1,
        core_updates=core_updates,
        provenance_updates=provenance_updates,
    )
    _rehash(document)

    with pytest.raises(ValueError):
        SequentialCandidateSearchReceiptV1.from_dict(document)


def test_canonically_rehashed_candidate_full_prefix_tampering_fails_closed():
    document = _qualified_receipt()
    _, new_hash = _rewrite_candidate_receipts(
        document,
        1,
        core_updates={"precedingNoGoodCutHashes": [_hash("foreign-cut")]},
        provenance_updates={"precedingNoGoodCutHashes": [_hash("foreign-cut")]},
    )
    row = document["rejectionFeedbackRows"][0]
    application = row["applicationReceipt"]
    application["solveEvidenceHash"] = new_hash
    application["nextCandidateHash"] = new_hash
    _rehash(application)
    _rehash(row, "stateHash")
    _rehash(document)

    with pytest.raises(ValueError, match="prefix"):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize("mutation", ["plugin", "witness"])
def test_canonically_rehashed_continuation_identity_tampering_fails_closed(mutation):
    document = _bounded_rejection_receipt()
    continuation = document["terminalContinuationReceipt"]
    if mutation == "plugin":
        continuation["pluginVersion"] = "999.0.0"
    else:
        rejected = document["evaluationReceipts"][0]["candidateProvenance"]
        continuation["witnessCandidateHash"] = rejected["candidateHash"]
        continuation["witnessDeckCommitment"] = rejected["deckCommitment"]
    _sync_terminal_continuation(document)
    _rehash(document)

    with pytest.raises(ValueError, match="plugin|witness"):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    ["root_seed", "maximum_count", "repeat_primary", "missing_repeat", "failure_codes"],
)
def test_canonically_rehashed_root_and_stop_semantics_fail_closed(mutation):
    document = (
        _bounded_rejection_receipt()
        if mutation in {"root_seed", "maximum_count", "failure_codes"}
        else _qualified_receipt()
    )
    if mutation == "root_seed":
        document["rootSeed"] += 1
    elif mutation == "maximum_count":
        document["maximumSolutions"] += 1
        document["searchRequest"]["maximumSolutions"] += 1
        _rehash(document["searchRequest"])
    elif mutation == "repeat_primary":
        receipt = document["evaluationReceipts"][1]
        replacement = SequentialCandidateEvaluationV1.build(
            outcome=SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
            candidate_hash=receipt["candidateProvenance"]["candidateHash"],
            evaluation_evidence_hashes=(_hash("fatal-evidence"),),
            failure_codes=("fatal_primary",),
        )
        receipt["evaluation"] = replacement.to_dict()
        _rehash(receipt)
        document["failureCodes"] = ["fatal_primary"]
    elif mutation == "missing_repeat":
        document["evaluationReceipts"].pop()
        document["qualifiedCandidateHash"] = None
        document["stopReason"] = "fatal_quarantine"
    else:
        document["failureCodes"].append("foreign_failure")
        document["failureCodes"].sort()
    _rehash(document)

    with pytest.raises(ValueError):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    ["assignment", "arbitrary_partition", "negative_positions", "dealer", "buyer"],
)
def test_compact_final_hand_cut_is_bound_to_deck_and_layout(root, mutation):
    rules = build_bundle(root).rules
    spec = _spec(maximum_solutions=1, assumed_buyer="south")
    generator = _Generator(
        spec,
        candidate_count=1,
        root_seed=29,
        dealer=Seat.WEST,
        no_good_cut_type="final_hand_allocation_v1",
        rules=rules,
    )
    document = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=29,
        dealer=Seat.WEST,
        rules=rules,
        evaluate_candidate=lambda candidate, _index: _evaluation(
            candidate,
            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
            failures=("ordinary",),
        ),
    ).receipt.to_dict()
    row = document["rejectionFeedbackRows"][0]
    cut = row["noGoodCut"]
    if mutation == "assignment":
        east = cut["exactAssignment"]["east"]
        south = cut["exactAssignment"]["south"]
        east[0], south[0] = south[0], east[0]
        cut["cutSha256"] = canonical_sha256({
            key: value
            for key, value in cut.items()
            if key not in {
                "sha256",
                "cutSha256",
                "validationFinalHandPositions",
                "validationDealer",
                "validationBuyer",
            }
        })
    elif mutation == "arbitrary_partition":
        positions = cut["validationFinalHandPositions"]
        positions["east"][0], positions["south"][0] = (
            positions["south"][0],
            positions["east"][0],
        )
        deck = document["evaluationReceipts"][0]["candidateIntegrityCore"][
            "deckOrder"
        ]
        cut["exactAssignment"] = {
            seat: sorted(deck[position] for position in seat_positions)
            for seat, seat_positions in positions.items()
        }
        cut["cutSha256"] = canonical_sha256({
            key: value
            for key, value in cut.items()
            if key not in {
                "sha256",
                "cutSha256",
                "validationFinalHandPositions",
                "validationDealer",
                "validationBuyer",
            }
        })
    elif mutation == "negative_positions":
        for positions in cut["validationFinalHandPositions"].values():
            positions[:] = [position - 32 for position in positions]
    elif mutation == "dealer":
        cut["validationDealer"] = "east"
    else:
        cut["validationBuyer"] = "north"
    _rehash(cut)
    _rehash_feedback(row)
    if mutation == "arbitrary_partition":
        row["noGoodCutHash"] = cut["cutSha256"]
        continuation = document["terminalContinuationReceipt"]
        continuation["precedingNoGoodCutHashes"] = [cut["cutSha256"]]
        continuation["finalNoGoodCutHash"] = cut["cutSha256"]
        _rehash(continuation)
        application = row["applicationReceipt"]
        application["feedbackHash"] = row["feedbackHash"]
        application["noGoodCutHash"] = cut["cutSha256"]
        application["solveEvidenceHash"] = continuation["sha256"]
        _rehash(application)
        _rehash(row, "stateHash")
    _rehash(document)

    with pytest.raises(ValueError, match="final-hand no-good assignment"):
        SequentialCandidateSearchReceiptV1.from_dict(document)


def test_fully_cascaded_compact_layout_forgery_fails_closed(root):
    rules = build_bundle(root).rules
    spec = _spec(maximum_solutions=1, assumed_buyer="east")
    generator = _Generator(
        spec,
        candidate_count=1,
        root_seed=35,
        dealer=Seat.NORTH,
        no_good_cut_type="final_hand_allocation_v1",
        rules=rules,
    )
    document = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=35,
        dealer=Seat.NORTH,
        rules=rules,
        evaluate_candidate=lambda candidate, _index: _evaluation(
            candidate,
            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
            failures=("ordinary",),
        ),
    ).receipt.to_dict()

    search_identity = document["searchIdentityDocument"]
    layout = search_identity["dealLayout"]
    initial = layout["initialHandPositions"]
    final = layout["finalHandPositions"]
    west_packet = sorted(set(final["west"]) - set(initial["west"]))
    south_packet = sorted(set(final["south"]) - set(initial["south"]))
    assert len(west_packet) == len(south_packet) == 3
    swaps = {
        **dict(zip(west_packet, south_packet, strict=True)),
        **dict(zip(south_packet, west_packet, strict=True)),
    }
    final["west"] = [swaps.get(position, position) for position in final["west"]]
    final["south"] = [swaps.get(position, position) for position in final["south"]]
    _rehash(layout)
    _rehash(search_identity)
    search_hash = search_identity["sha256"]
    document["searchIdentityHash"] = search_hash

    evaluation_receipt = document["evaluationReceipts"][0]
    candidate_core = evaluation_receipt["candidateIntegrityCore"]
    deck = candidate_core["deckOrder"]
    for west_position, south_position in zip(
        west_packet,
        south_packet,
        strict=True,
    ):
        deck[west_position], deck[south_position] = (
            deck[south_position],
            deck[west_position],
        )
    deck_commitment = hashlib.sha256(bytes(deck)).hexdigest()
    candidate_core["deckCommitment"] = deck_commitment
    candidate_core["searchIdentityHash"] = search_hash
    candidate_hash = canonical_sha256(candidate_core)
    provenance = evaluation_receipt["candidateProvenance"]
    provenance["deckCommitment"] = deck_commitment
    provenance["searchIdentityHash"] = search_hash
    provenance["candidateHash"] = candidate_hash
    evaluation_receipt["evaluation"]["candidateHash"] = candidate_hash
    _rehash(evaluation_receipt["evaluation"])
    _rehash(evaluation_receipt)

    row = document["rejectionFeedbackRows"][0]
    cut = row["noGoodCut"]
    cut["candidateHash"] = candidate_hash
    cut["searchIdentityHash"] = search_hash
    cut["validationFinalHandPositions"] = final
    cut["exactAssignment"] = {
        seat: sorted(deck[position] for position in positions)
        for seat, positions in final.items()
    }
    cut["cutSha256"] = canonical_sha256({
        key: value
        for key, value in cut.items()
        if key not in {
            "sha256",
            "cutSha256",
            "validationFinalHandPositions",
            "validationDealer",
            "validationBuyer",
        }
    })
    _rehash(cut)
    row["candidateHash"] = candidate_hash
    row["evaluationReceiptHash"] = evaluation_receipt["sha256"]
    row["noGoodCutHash"] = cut["cutSha256"]
    _rehash_feedback(row)

    continuation = document["terminalContinuationReceipt"]
    continuation["searchIdentityHash"] = search_hash
    continuation["precedingNoGoodCutHashes"] = [cut["cutSha256"]]
    continuation["finalNoGoodCutHash"] = cut["cutSha256"]
    _rehash(continuation)
    application = row["applicationReceipt"]
    application["feedbackHash"] = row["feedbackHash"]
    application["noGoodCutHash"] = cut["cutSha256"]
    application["solveEvidenceHash"] = continuation["sha256"]
    _rehash(application)
    _rehash(row, "stateHash")
    _rehash(document)

    with pytest.raises(ValueError, match="deal layout commitment"):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("modelHash", _hash("foreign-model")),
        ("backendVersion", "foreign-backend"),
        ("noGoodCutType", "final_hand_allocation_v1"),
        ("structuralObjectiveScore", 1),
        ("searchIdentityHash", _hash("foreign-search")),
    ],
)
def test_canonically_rehashed_empty_search_authority_relabel_fails_closed(
    field,
    foreign,
):
    spec = _spec(maximum_solutions=1)
    generator = _Generator(spec, candidate_count=0, root_seed=30, dealer=Seat.NORTH)
    document = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=30,
        dealer=Seat.NORTH,
        rules=_RULES,
        evaluate_candidate=lambda *_args: pytest.fail("no evaluation expected"),
    ).receipt.to_dict()
    document[field] = foreign
    _rehash(document)

    with pytest.raises(ValueError):
        SequentialCandidateSearchReceiptV1.from_dict(document)


def test_empty_search_freezes_optional_expected_search_identity():
    spec = _spec(maximum_solutions=1)
    expected = _search_hash(spec, root_seed=31, dealer=Seat.NORTH)
    document = run_sequential_candidate_search_v1(
        _Generator(spec, candidate_count=0, root_seed=31, dealer=Seat.NORTH),
        spec,
        root_seed=31,
        dealer=Seat.NORTH,
        rules=_RULES,
        evaluate_candidate=lambda *_args: pytest.fail("no evaluation expected"),
        expected_search_identity_hash=expected,
    ).receipt.verify().to_dict()

    assert document["searchRequest"]["expectedSearchIdentityHash"] == expected
    assert document["searchIdentityHash"] == expected

    document["searchRequest"]["expectedSearchIdentityHash"] = _hash("foreign")
    _rehash(document["searchRequest"])
    _rehash(document)
    with pytest.raises(ValueError, match="expected identity"):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    [
        "continuation_exhausted",
        "continuation_score",
        "maximum",
        "counts",
        "application_ordinal",
    ],
)
def test_canonically_rehashed_json_type_aliases_fail_closed(mutation):
    document = _bounded_rejection_receipt()
    if mutation == "continuation_exhausted":
        document["terminalContinuationReceipt"]["exhausted"] = 0
        _sync_terminal_continuation(document)
    elif mutation == "continuation_score":
        document["terminalContinuationReceipt"]["witnessOrderingScore"] = "foreign"
        _sync_terminal_continuation(document)
    elif mutation == "maximum":
        document["maximumSolutions"] = "2"
        document["searchRequest"]["maximumSolutions"] = "2"
        _rehash(document["searchRequest"])
    elif mutation == "counts":
        document["emittedCandidateCount"] = "2"
        document["evaluatedCandidateCount"] = "2"
        document["rejectedCandidateCount"] = "2"
    else:
        row = document["rejectionFeedbackRows"][0]
        row["applicationReceipt"]["solveOrdinal"] = "1"
        _rehash(row["applicationReceipt"])
        _rehash(row, "stateHash")
    _rehash(document)

    with pytest.raises(ValueError):
        SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    ["ordinal_bool", "boolean_cards", "ordering_string", "structural_string"],
)
def test_runtime_candidate_exact_types_delegate_to_generator_contract(mutation):
    spec = _spec(maximum_solutions=1)
    generator = _Generator(spec, candidate_count=1, root_seed=32, dealer=Seat.WEST)
    candidate = generator.candidates[0]
    if mutation == "ordinal_bool":
        candidate = replace(candidate, ordinal=False, candidate_hash="")
    elif mutation == "boolean_cards":
        candidate = replace(
            candidate,
            deck_order=(False, True, *range(2, 32)),
            candidate_hash="",
        )
    elif mutation == "ordering_string":
        candidate = replace(candidate, ordering_score="900", candidate_hash="")
    else:
        candidate = replace(
            candidate,
            structural_objective_score="42",
            candidate_hash="",
        )
    candidate = replace(
        candidate,
        candidate_hash=canonical_sha256(candidate.integrity_core()),
    )
    generator.candidates = (candidate,)

    with pytest.raises(ValueError):
        run_sequential_candidate_search_v1(
            generator,
            spec,
            root_seed=32,
            dealer=Seat.WEST,
            rules=_RULES,
            evaluate_candidate=lambda current, _index: _evaluation(
                current,
                SequentialCandidateOutcomeKindV1.QUALIFIED,
            ),
        )


def test_empty_search_rejects_rehashed_dealer_and_unevidenced_cancellation():
    spec = _spec(maximum_solutions=1)
    generator = _Generator(spec, candidate_count=0, root_seed=33, dealer=Seat.NORTH)
    document = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=33,
        dealer=Seat.NORTH,
        rules=_RULES,
        evaluate_candidate=lambda *_args: pytest.fail("no evaluation expected"),
    ).receipt.to_dict()

    invalid_dealer = {**document, "dealer": "foreign"}
    invalid_dealer["searchRequest"] = {
        **document["searchRequest"],
        "dealer": "foreign",
    }
    _rehash(invalid_dealer["searchRequest"])
    _rehash(invalid_dealer)
    with pytest.raises(ValueError, match="dealer"):
        SequentialCandidateSearchReceiptV1.from_dict(invalid_dealer)

    relabelled = document
    relabelled["terminalContinuationReceipt"] = None
    relabelled["searchIdentityDocument"] = None
    for field in (
        "modelHash",
        "searchIdentityHash",
        "backendVersion",
        "noGoodCutType",
        "structuralObjectiveScore",
    ):
        relabelled[field] = None
    relabelled["stopReason"] = "cancelled"
    relabelled["failureCodes"] = ["candidate_generation_cancelled"]
    _rehash(relabelled)
    with pytest.raises(ValueError, match="initial solve receipt"):
        SequentialCandidateSearchReceiptV1.from_dict(relabelled)


@pytest.mark.parametrize("stage", ["generation", "probe", "unknown_probe"])
def test_zero_candidate_nonconclusive_generator_outcomes_do_not_emit_receipts(stage):
    spec = _spec(maximum_solutions=1)
    generator = _Generator(spec, candidate_count=0, root_seed=34, dealer=Seat.NORTH)
    if stage == "generation":
        generator.generate_next = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InterruptedError("cancelled")
        )
    elif stage == "probe":
        generator.probe_continuation = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InterruptedError("cancelled")
        )
    else:
        original_probe = generator.probe_continuation

        def unknown_probe(*args, **kwargs):
            receipt = original_probe(*args, **kwargs)
            invalid = replace(receipt, solver_status="unknown", receipt_hash="")
            return replace(
                invalid,
                receipt_hash=canonical_sha256(invalid.integrity_core()),
            )

        generator.probe_continuation = unknown_probe

    expected = InterruptedError if stage != "unknown_probe" else ValueError
    with pytest.raises(expected):
        run_sequential_candidate_search_v1(
            generator,
            spec,
            root_seed=34,
            dealer=Seat.NORTH,
            rules=_RULES,
            evaluate_candidate=lambda *_args: pytest.fail("no evaluation expected"),
        )
