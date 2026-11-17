from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from baloot_platform.analysis.search import SearchTimeBudgetExceeded
from baloot_platform.challenge.analysis_pipeline import TraceAnalysisProfileV1
from baloot_platform.challenge.constraint_deal_search import (
    ConstraintDealCandidateV1,
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research import service as service_module
from baloot_platform.research.audit_trace import ResearchAuditTraceRecordV1
from baloot_platform.research.contracts import build_research_recipe
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.service.container import build_container
from scripts.run_strategic_play_pilot import _recipe


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _Harness:
    container: Any
    research: Any
    simulation: Any
    simulator: Any


@pytest.fixture(scope="module")
def service_harness(tmp_path_factory: pytest.TempPathFactory) -> _Harness:
    container = build_container(
        writable_root=tmp_path_factory.mktemp("research-audit-service")
    )
    research = container.research
    simulator = research.simulator
    simulation = simulator.simulate(1, dealer=Seat.EAST, match_score=(0, 0))
    assert simulation.publishable_laffa is True
    assert simulation.final_trace.trusted_final_state["phase"] == GamePhase.SCORED.value
    try:
        yield _Harness(container, research, simulation, simulator)
    finally:
        container.shutdown()


class _FixedSimulator:
    def __init__(self, base: Any, simulation: Any) -> None:
        self.rules = base.rules
        self.agent_release = base.agent_release
        self.exact_search_backend = getattr(base, "exact_search_backend", None)
        self._simulation = simulation

    def simulate(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._simulation

    def simulate_preordered_deck(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._simulation


class _SequenceSimulator:
    def __init__(self, base: Any, simulations: tuple[Any, ...]) -> None:
        self._base = base
        self._simulations = iter(simulations)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def simulate_preordered_deck(self, *_args: Any, **_kwargs: Any) -> Any:
        return next(self._simulations)


class _SequentialGeneratorStub:
    descriptor = OrToolsCpSatDealGeneratorV1.descriptor

    def __init__(self, label: str, *, candidate_count: int = 1) -> None:
        self.label = label
        self.candidate_count = candidate_count
        self.generate_calls: list[tuple[str, ...]] = []
        self.probe_calls: list[tuple[str, ...]] = []
        self.legacy_generate_calls = 0
        self.candidates: dict[int, ConstraintDealCandidateV1] = {}

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def _cut_hashes(
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
    ) -> tuple[str, ...]:
        return tuple(
            OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
                candidate,
                layout=None,
            )["cutSha256"]
            for candidate in rejected_candidates
        )

    def _candidate(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: Any,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
    ) -> ConstraintDealCandidateV1:
        ordinal = len(rejected_candidates)
        deck = tuple(range(ordinal, 32)) + tuple(range(ordinal))
        model_hash = _hash(f"service-generator-model:{self.label}")
        backend_version = "fixture-backend-v1"
        search_hash = OrToolsCpSatDealGeneratorV1.search_identity_document(
            spec=spec,
            model_hash=model_hash,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            backend_version=backend_version,
            no_good_cut_type="complete_deck_assignment_v1",
        )["sha256"]
        candidate = OrToolsCpSatDealGeneratorV1._build_candidate(
            ordinal=ordinal,
            deck_order=deck,
            deck_commitment=hashlib.sha256(bytes(deck)).hexdigest(),
            ordering_score=100 - ordinal,
            structural_objective_score=1,
            profile_hash=spec.sha256,
            model_hash=model_hash,
            search_identity_hash=search_hash,
            backend_version=backend_version,
            no_good_cut_type="complete_deck_assignment_v1",
            preceding_no_good_cut_hashes=self._cut_hashes(rejected_candidates),
        )
        self.candidates[ordinal] = candidate
        return candidate

    def generate_next(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: Any,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        **_kwargs: Any,
    ) -> ConstraintDealCandidateV1 | None:
        self.generate_calls.append(
            tuple(candidate.candidate_hash for candidate in rejected_candidates)
        )
        if len(rejected_candidates) >= self.candidate_count:
            return None
        return self._candidate(
            spec,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            rejected_candidates=rejected_candidates,
        )

    def probe_continuation(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        rules: Any,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        **_kwargs: Any,
    ) -> Any:
        self.probe_calls.append(
            tuple(candidate.candidate_hash for candidate in rejected_candidates)
        )
        exhausted = len(rejected_candidates) >= self.candidate_count
        witness = (
            None
            if exhausted
            else self._candidate(
                spec,
                root_seed=root_seed,
                dealer=dealer,
                rules=rules,
                rejected_candidates=rejected_candidates,
            )
        )
        authority = rejected_candidates[0] if rejected_candidates else witness
        assert authority is not None
        return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
            profile_hash=spec.sha256,
            model_hash=authority.model_hash,
            backend_version=authority.backend_version,
            solver_status="infeasible" if exhausted else "optimal",
            solution_ordinal=len(rejected_candidates),
            search_identity_hash=authority.search_identity_hash,
            preceding_no_good_cut_hashes=self._cut_hashes(rejected_candidates),
            no_good_cut_type=authority.no_good_cut_type,
            exhausted=exhausted,
            witness=witness,
            structural_objective_score=authority.structural_objective_score,
        ).verify()

    def generate(self, *_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        self.legacy_generate_calls += 1
        raise AssertionError("Research service must not use legacy batch generation")


def _deal_spec(
    label: str,
    *,
    assumed_buyer: Seat | None,
    maximum_solutions: int,
) -> DealConstraintSpecV1:
    return DealConstraintSpecV1.build({
        "schemaVersion": 1,
        "id": f"service_{label.replace('-', '_')}_v1",
        "revision": "1",
        "enabled": True,
        "backend": "ortools_cp_sat_v1",
        "assumedBuyer": (
            assumed_buyer.name.lower() if assumed_buyer is not None else None
        ),
        "purchaseSuits": [],
        "initialRequiredCards": {},
        "finalRequiredCards": {},
        "finalForbiddenCards": {},
        "initialSuitCountRanges": {},
        "finalSuitCountRanges": {},
        "projectPattern": None,
        "maximumSolutions": maximum_solutions,
        "maximumConflicts": 1_000,
        "safetyTimeoutSeconds": 1.0,
        "solverWorkers": 1,
        "seedOffset": 0,
        "tags": ["service_fixture"],
    })


def _stub_ortools_compilers(
    research: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    label: str,
    spec: DealConstraintSpecV1,
) -> None:
    auction = SimpleNamespace(
        actions=(),
        sha256=_hash(f"auction:{label}"),
        model_hash=_hash(f"auction-model:{label}"),
        solver_status="optimal",
    )
    escalation = SimpleNamespace(actions=())
    project = SimpleNamespace(sha256=_hash(f"project:{label}"))
    play = SimpleNamespace(
        sha256=_hash(f"play:{label}"),
        model_hash=_hash(f"play-model:{label}"),
        solver_status="optimal",
        tactical_precursor=None,
    )
    monkeypatch.setattr(research, "_compile_deal_spec", lambda *_a, **_k: spec)
    monkeypatch.setattr(research, "_compile_auction", lambda *_a, **_k: auction)
    monkeypatch.setattr(
        research,
        "_compile_escalation_plan",
        lambda *_a, **_k: escalation,
    )
    monkeypatch.setattr(research, "_played_line", lambda *_a, **_k: ((), ()))
    monkeypatch.setattr(
        research,
        "_compile_complete_play",
        lambda *_a, **_k: (project, play, ()),
    )
    monkeypatch.setattr(
        research,
        "_compiled_play_mismatches",
        lambda *_a, **_k: [],
    )


def _payload(
    research: Any,
    *,
    label: str,
    lane: str,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = int(_hash(label)[:8], 16)
    value = _recipe(
        research,
        mode="actor_safe_agent_verified",
        attempts=1,
        seed_start=seed,
    )
    if lane == "organic_runtime":
        value["resources"] = {
            "generationLane": lane,
            "seedStart": seed,
            "seedCount": 1,
            "seedsPerShard": 1,
            "analysisMode": "screening",
        }
    else:
        value["resources"].update(
            {
                "generationLane": "ortools",
                "seedStart": seed,
                "seedCount": 1,
                "seedsPerShard": 1,
                "auctionConstructionMode": "actor_safe_agent_verified",
                "playConstructionMode": "actor_safe_agent_verified",
            }
        )
    value["resources"].update(copy.deepcopy(resources or {}))
    return value


def _claim_case(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    label: str,
    payload: dict[str, Any],
    simulation: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    research = harness.research
    recipe = build_research_recipe(payload)
    research.store.put_recipe(recipe.to_dict())
    run_hash = _hash(f"audit-service-run:{label}")
    deal_generator_identity = (
        research._deal_generator_identity()
        if recipe.resources["generationLane"] == "ortools"
        else None
    )
    run = research.store.create_run(
        {
            "schemaVersion": 1,
            "runCode": f"RUN-{run_hash[:12].upper()}",
            "runHash": run_hash,
            "recipeCode": recipe.code,
            "recipeHash": recipe.sha256,
            "generationLane": recipe.resources["generationLane"],
            "dealGeneratorIdentity": copy.deepcopy(deal_generator_identity),
            "dealGeneratorDescriptorHash": (
                deal_generator_identity["descriptorHash"]
                if deal_generator_identity is not None
                else None
            ),
            "status": "queued",
            "progress": 0.0,
            "resultCount": 0,
            "quarantineCount": 0,
            "events": [],
            "createdAt": "2026-08-10T00:00:00+00:00",
        }
    )
    shards = partition_research_shards(
        run_code=run["runCode"],
        run_hash=run_hash,
        recipe_hash=recipe.sha256,
        seed_start=int(recipe.resources["seedStart"]),
        seed_count=int(recipe.resources["seedCount"]),
        seeds_per_shard=int(recipe.resources["seedsPerShard"]),
    )
    research.store.create_shards(tuple(item.to_dict() for item in shards))
    worker_id = f"audit-service-worker:{label}"
    claim = research.store.claim_shard(
        run["runCode"], worker_id=worker_id, lease_seconds=30
    )
    assert claim is not None
    monkeypatch.setattr(
        research,
        "_build_recipe_and_context",
        lambda _value: (recipe, None),
    )
    monkeypatch.setattr(
        research,
        "simulator",
        _FixedSimulator(harness.simulator, simulation or harness.simulation),
    )
    return run, claim, recipe


def _audit_items(research: Any, run_code: str) -> list[dict[str, Any]]:
    return research.list_audit_traces(run_code=run_code, limit=20)["items"]


def _result_items(research: Any, run_code: str) -> list[dict[str, Any]]:
    return research.list_results(run_code=run_code, limit=20)["items"]


def _store_validated_legacy_recipe(
    research: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Seed a historical V3 fixture without exercising new-authoring APIs."""

    validated = research.validate_recipe(payload)
    assert validated["valid"] is True, validated["issues"]
    return research.store.put_recipe(validated["recipe"])


def _search_receipt_items(
    research: Any,
    run_code: str,
    *,
    admitted_only: bool = True,
) -> list[dict[str, Any]]:
    return research.store.list_sequential_search_receipts(
        run_code=run_code,
        admitted_only=admitted_only,
        limit=20,
    )["items"]


def _completed_events(research: Any, claim: dict[str, Any]) -> list[dict[str, Any]]:
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["status"] == "completed"
    return list((shard.get("result") or {}).get("events") or ())


def _completed_all_pass_simulation(simulator: Any) -> Any:
    scripted_bids = tuple(
        {
            "actor": actor,
            "roundIndex": round_index,
            "action": "pass",
            "trumpSuit": None,
        }
        for round_index in (1, 2)
        for actor in ("north", "west", "south", "east")
    )
    return simulator.simulate_preordered_deck(
        91,
        tuple(range(32)),
        deal_generator_provenance={
            "schemaVersion": 1,
            "pluginId": "service_all_pass_fixture_v1",
            "profileHash": _hash("service-all-pass-profile"),
        },
        dealer=Seat.EAST,
        match_score=(0, 0),
        scripted_bids=scripted_bids,
    )


def _prepare_ortools_branch(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    label: str,
    resources: dict[str, Any] | None = None,
    situation: dict[str, Any] | None = None,
    assumed_buyer: Seat | None = None,
    simulation: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    research = harness.research
    payload = _payload(
        research,
        label=label,
        lane="ortools",
        resources=resources,
    )
    analysis_mode = str(payload["resources"]["analysisMode"])
    payload["profiles"]["analysisProfileHash"] = (
        research.analyzers[analysis_mode].profile.sha256
    )
    payload["situation"].update(copy.deepcopy(situation or {}))
    spec = _deal_spec(
        label,
        assumed_buyer=assumed_buyer,
        maximum_solutions=int(payload["resources"]["candidatesPerSeed"]),
    )
    monkeypatch.setattr(
        research,
        "deal_generator",
        _SequentialGeneratorStub(label, candidate_count=spec.maximum_solutions),
    )
    run, claim, _ = _claim_case(
        harness,
        monkeypatch,
        label=label,
        payload=payload,
        simulation=simulation,
    )
    _stub_ortools_compilers(
        research,
        monkeypatch,
        label=label,
        spec=spec,
    )
    return run, claim


def test_ortools_submission_freezes_canonical_generator_identity(
    service_harness: _Harness,
) -> None:
    research = service_harness.research
    payload = _payload(
        research,
        label="submission-generator-identity",
        lane="ortools",
    )
    payload["schemaVersion"] = 3
    payload.pop("profiles")
    payload["tactics"] = []
    payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": service_harness.container.configuration.resolve()[
            "bundle"
        ]["revisionId"],
    }
    recipe = _store_validated_legacy_recipe(research, payload)

    run = research._submit_recipe_run(
        recipe["code"],
        idempotency_key="submission-generator-identity",
        schedule=False,
    )

    identity = research._deal_generator_identity()
    assert run["generationLane"] == "ortools"
    assert run["dealGeneratorIdentity"] == identity
    assert run["dealGeneratorDescriptorHash"] == identity["descriptorHash"]


def test_retry_rejects_tampered_run_generator_identity_before_heartbeat(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, first_claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="tampered-generator-retry",
        payload=_payload(
            research,
            label="tampered-generator-retry",
            lane="ortools",
        ),
    )
    research.store.fail_shard(
        first_claim["shardId"],
        worker_id=first_claim["leaseOwner"],
        fencing_token=first_claim["fencingToken"],
        error={"type": "fixture_retry", "message": "retry identity check"},
        retryable=True,
    )
    retry_claim = research.store.claim_shard(
        run["runCode"],
        worker_id="audit-service-worker:tampered-generator-retry-2",
        lease_seconds=30,
    )
    assert retry_claim is not None
    assert retry_claim["attemptCount"] == 2
    tampered_identity = copy.deepcopy(run["dealGeneratorIdentity"])
    tampered_identity["sha256"] = "0" * 64
    research.store.update_run(
        run["runCode"],
        dealGeneratorIdentity=tampered_identity,
    )
    heartbeat_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_heartbeat = research.store.heartbeat_shard

    def heartbeat_spy(*args: Any, **kwargs: Any) -> Any:
        heartbeat_calls.append((args, copy.deepcopy(kwargs)))
        return original_heartbeat(*args, **kwargs)

    monkeypatch.setattr(research.store, "heartbeat_shard", heartbeat_spy)

    with pytest.raises(ValueError, match="deal-generator identity mismatch"):
        research._execute_claimed_shard(
            run["runCode"],
            retry_claim,
            retry_claim["leaseOwner"],
            context=None,
        )

    assert heartbeat_calls == []
    assert _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    ) == []


def test_retry_after_result_put_reuses_identical_result_payload(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, first_claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="retry-after-result-put",
        payload=_payload(
            research,
            label="retry-after-result-put",
            lane="organic_runtime",
        ),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])
    result_payloads: list[dict[str, Any]] = []
    completion_calls = 0
    original_put_result = research.store.put_result
    original_complete_shard = research.store.complete_shard

    def capture_result(payload: dict[str, Any], **kwargs: Any) -> Any:
        result_payloads.append(copy.deepcopy(payload))
        return original_put_result(payload, **kwargs)

    def interrupt_first_completion(*args: Any, **kwargs: Any) -> Any:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            raise InterruptedError("fixture interruption after result put")
        return original_complete_shard(*args, **kwargs)

    monkeypatch.setattr(research.store, "put_result", capture_result)
    monkeypatch.setattr(
        research.store,
        "complete_shard",
        interrupt_first_completion,
    )

    research._execute_claimed_shard(
        run["runCode"],
        first_claim,
        first_claim["leaseOwner"],
        context=None,
    )

    interrupted = research.store.get_shard(first_claim["shardId"])
    assert interrupted is not None
    assert interrupted["status"] == "queued"
    assert interrupted["retryCount"] == 1
    assert len(result_payloads) == 1
    assert result_payloads[0]["createdAt"] == run["createdAt"]
    assert research.get_result(result_payloads[0]["resultCode"]) is None

    retry_claim = research.store.claim_shard(
        run["runCode"],
        worker_id="audit-service-worker:retry-after-result-put-2",
        lease_seconds=30,
    )
    assert retry_claim is not None
    assert retry_claim["attemptCount"] == first_claim["attemptCount"] + 1
    assert retry_claim["fencingToken"] == first_claim["fencingToken"] + 1

    research._execute_claimed_shard(
        run["runCode"],
        retry_claim,
        retry_claim["leaseOwner"],
        context=None,
    )

    completed = research.store.get_shard(retry_claim["shardId"])
    assert completed is not None and completed["status"] == "completed"
    assert completion_calls == 2
    assert len(result_payloads) == 2
    assert result_payloads[0] == result_payloads[1]
    admitted = _result_items(research, run["runCode"])
    assert admitted == [result_payloads[0]]
    records = _audit_items(research, run["runCode"])
    assert len(records) == 2
    assert {
        int(record["source"]["attemptCount"]): record["attemptStatus"]
        for record in records
    } == {
        first_claim["attemptCount"]: "retry_scheduled",
        retry_claim["attemptCount"]: "completed",
    }


def test_scored_trace_is_retained_before_semantic_quarantine(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="semantic-quarantine",
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    call_order: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    original_put_receipt = research.store.put_sequential_search_receipt
    original_put_disagreement = research.store.put_disagreement
    original_finalize = research.audit_traces.finalize
    original_complete = research.store.complete_shard

    def put_receipt_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("receipt", args, copy.deepcopy(kwargs)))
        return original_put_receipt(*args, **kwargs)

    def put_disagreement_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("disagreement", args, copy.deepcopy(kwargs)))
        return original_put_disagreement(*args, **kwargs)

    def finalize_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("finalize", args, copy.deepcopy(kwargs)))
        return original_finalize(*args, **kwargs)

    def complete_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("complete", args, copy.deepcopy(kwargs)))
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(
        research.store,
        "put_sequential_search_receipt",
        put_receipt_spy,
    )
    monkeypatch.setattr(
        research.store,
        "put_disagreement",
        put_disagreement_spy,
    )
    monkeypatch.setattr(research.audit_traces, "finalize", finalize_spy)
    monkeypatch.setattr(research.store, "complete_shard", complete_spy)
    monkeypatch.setattr(
        research,
        "_semantic_mismatches",
        lambda *_a, **_k: ["runtime_transition_defect"],
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == service_harness.simulation.final_trace.sha256
    assert record["disposition"] == "quarantined", research.store.get_shard(
        claim["shardId"]
    )
    assert record["auditOnly"] is True
    assert record["publicationEligible"] is False
    assert {
        "stage_semantic_parity_verification",
        "runtime_transition_defect",
    }.issubset(record["failureCategories"])
    assert len(record["disagreementReferences"]) == 1
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert [item["disagreementId"] for item in disagreements] == [
        record["disagreementReferences"][0]["disagreementId"]
    ]
    assert _result_items(research, run["runCode"]) == []
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["result"]["quarantineCount"] == 1
    receipt = _search_receipt_items(research, run["runCode"])[0]
    assert receipt["attemptCount"] == claim["attemptCount"]
    assert receipt["fencingToken"] == claim["fencingToken"]
    assert receipt["admitted"] is True
    bundle = receipt["receiptBundle"]
    assert bundle["stopReason"] == "fatal_quarantine"
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0
    names = [item[0] for item in call_order]
    assert names.count("receipt") == 1
    assert names.count("disagreement") == 1
    assert names.count("finalize") == 1
    assert names.count("complete") == 1
    assert names.index("receipt") < names.index("disagreement")
    assert names.index("receipt") < names.index("finalize")
    assert names.index("receipt") < names.index("complete")
    receipt_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "receipt"
    )
    disagreement_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "disagreement"
    )
    complete_args, complete_kwargs = next(
        (args, kwargs)
        for name, args, kwargs in call_order
        if name == "complete"
    )
    expected_execution = {
        "run_code": run["runCode"],
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
        "seed": claim["seedStart"],
    }
    assert receipt_kwargs == expected_execution
    assert disagreement_kwargs == {
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
    }
    assert complete_args == (claim["shardId"],)
    assert complete_kwargs["worker_id"] == claim["leaseOwner"]
    assert complete_kwargs["fencing_token"] == claim["fencingToken"]
    assert record["source"] | {
        "runCode": run["runCode"],
        "shardId": claim["shardId"],
        "attemptCount": claim["attemptCount"],
        "fencingToken": claim["fencingToken"],
        "workerId": claim["leaseOwner"],
        "seed": claim["seedStart"],
    } == record["source"]


@pytest.mark.real_clock
def test_analysis_wall_clock_exhaustion_is_retained_as_incomplete_without_retry(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="analysis-wall-clock-incomplete",
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    analyzer_type = type(research.analyzers["screening"])

    def exhaust_wall_clock(*_args: Any, **_kwargs: Any) -> Any:
        raise SearchTimeBudgetExceeded(
            "exact search exceeded its wall-clock budget without canonical output"
        )

    monkeypatch.setattr(analyzer_type, "analyze_trace", exhaust_wall_clock)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["status"] == "completed"
    assert shard["attemptCount"] == claim["attemptCount"]
    assert generator.generate_calls == [()]
    assert _result_items(research, run["runCode"]) == []
    receipt = _search_receipt_items(research, run["runCode"])[0]
    assert receipt["receiptBundle"]["stopReason"] == "incomplete"
    assert receipt["receiptBundle"]["failureCodes"] == [
        "analysis_wall_clock_budget_exhausted"
    ]
    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "rejected"
    assert {
        "stage_candidate_analysis",
        "candidate_analysis_incomplete",
        "analysis_wall_clock_budget_exhausted",
    }.issubset(records[0]["failureCategories"])


def test_qualified_repeat_source_conflict_quarantines_primary_audit(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    selection_value = research.decision_score_profile.to_dict()
    selection_value.pop("sha256")
    selection_value.update({
        "id": "qualified_repeat_fixture",
        "revision": "1",
        "minimumScore": 0.0,
        "requireVerifiedTacticalPlay": False,
        "minimumSurroundingQuality": 0.0,
        "minimumSurroundingConfidence": 0.0,
        "minimumSurroundingMateriality": 0.0,
        "maximumSurroundingMaterialMistakes": 2_147_483_647,
        "maximumSurroundingUnverifiedDecisions": 2_147_483_647,
        "maximumSurroundingCatastrophicCrossings": 2_147_483_647,
    })
    decision_score_profile = SelectionProfileV1.build(selection_value)
    monkeypatch.setattr(
        research,
        "decision_score_profile",
        decision_score_profile,
    )
    monkeypatch.setattr(
        research,
        "decision_score_profile_hash",
        decision_score_profile.sha256,
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="qualified-repeat-success",
        resources={"analysisMode": "screening"},
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    base_qualification = research._qualification
    base_evaluate_decision_selection = service_module.evaluate_decision_selection_v1
    qualification_calls: list[dict[str, Any]] = []
    decision_selection_calls: list[dict[str, Any]] = []

    def qualify_with_completed_content_gates(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        value = base_qualification(*args, **kwargs)
        qualification_calls.append(copy.deepcopy(value))
        return {
            **value,
            "qualified": True,
            "truthLabel": "provisional",
            "failures": [],
        }

    def record_decision_selection(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        value = base_evaluate_decision_selection(*args, **kwargs)
        decision_selection_calls.append(copy.deepcopy(value))
        return value

    call_order: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    original_put_receipt = research.store.put_sequential_search_receipt
    original_put_result = research.store.put_result
    original_put_disagreement = research.store.put_disagreement
    original_finalize = research.audit_traces.finalize
    original_complete = research.store.complete_shard

    def put_receipt_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("receipt", args, copy.deepcopy(kwargs)))
        return original_put_receipt(*args, **kwargs)

    def put_result_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("result", args, copy.deepcopy(kwargs)))
        return original_put_result(*args, **kwargs)

    def put_disagreement_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("disagreement", args, copy.deepcopy(kwargs)))
        return original_put_disagreement(*args, **kwargs)

    def finalize_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("finalize", args, copy.deepcopy(kwargs)))
        return original_finalize(*args, **kwargs)

    def complete_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("complete", args, copy.deepcopy(kwargs)))
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(
        research,
        "_qualification",
        qualify_with_completed_content_gates,
    )
    monkeypatch.setattr(
        service_module,
        "evaluate_decision_selection_v1",
        record_decision_selection,
    )
    monkeypatch.setattr(
        research.store,
        "put_sequential_search_receipt",
        put_receipt_spy,
    )
    monkeypatch.setattr(research.store, "put_result", put_result_spy)
    monkeypatch.setattr(
        research.store,
        "put_disagreement",
        put_disagreement_spy,
    )
    monkeypatch.setattr(research.audit_traces, "finalize", finalize_spy)
    monkeypatch.setattr(research.store, "complete_shard", complete_spy)

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=None,
    )

    results = _result_items(research, run["runCode"])
    assert len(results) == 1
    assert results[0]["disposition"] == "qualified"
    receipts = _search_receipt_items(research, run["runCode"])
    assert len(receipts) == 1
    bundle = receipts[0]["receiptBundle"]
    assert bundle["stopReason"] == "qualified"
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert [
        item["attempt"] for item in bundle["evaluationReceipts"]
    ] == ["primary", "qualified_repeat"]
    primary, repeated = bundle["evaluationReceipts"]
    assert primary["candidateIntegrityCore"] == repeated[
        "candidateIntegrityCore"
    ]
    assert primary["evaluation"] == repeated["evaluation"]
    assert bundle["qualifiedCandidateHash"] == results[0]["provenance"][
        "candidateHash"
    ]
    assert len(qualification_calls) == 2
    assert qualification_calls[0] == qualification_calls[1]
    coverage = research.compiled_domain.compiled.coverage
    required_base_failures = {
        "decision_material_counterexample",
        "decision_recommendation_mismatch",
        *(
            ("domain_executable_binding_coverage_incomplete",)
            if not coverage.executable_binding_complete
            else ()
        ),
        *(
            ("domain_behavioral_or_invariant_coverage_unproven",)
            if not coverage.behavioral_or_invariant_complete
            else ()
        ),
    }
    assert required_base_failures <= set(qualification_calls[0]["failures"])
    assert len(decision_selection_calls) == 2
    assert decision_selection_calls[0] == decision_selection_calls[1]
    assert decision_selection_calls[0]["profileHash"] == decision_score_profile.sha256
    assert decision_selection_calls[0]["passed"] is True
    assert decision_selection_calls[0]["evaluatedDecisionCount"] == 25
    assert decision_selection_calls[0]["qualifiedDecisionCount"] > 0
    assert decision_selection_calls[0]["rejectionReasons"] == []
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0
    names = [item[0] for item in call_order]
    assert names.count("receipt") == 1
    assert names.count("result") == 1
    assert names.count("finalize") == 1
    assert names.count("complete") == 1
    assert names.index("receipt") < names.index("result")
    assert names.index("receipt") < names.index("finalize")
    assert names.index("receipt") < names.index("complete")
    receipt_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "receipt"
    )
    result_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "result"
    )
    result_args = next(
        args for name, args, _kwargs in call_order if name == "result"
    )
    assert receipt_kwargs == {
        "run_code": run["runCode"],
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
        "seed": claim["seedStart"],
    }
    assert result_args[0]["provenance"]["researchShardId"] == claim["shardId"]
    assert result_kwargs == {
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
    }
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["result"]["resultCount"] == 1
    assert shard["result"]["qualifiedCount"] == 1
    assert shard["result"]["rejectedCount"] == 0
    assert shard["result"]["quarantineCount"] == 0

    call_order.clear()
    qualification_calls.clear()
    decision_selection_calls.clear()
    secondary = service_harness.simulator.simulate(
        2,
        dealer=Seat.EAST,
        match_score=(0, 0),
    )
    assert secondary.final_trace.trusted_final_state["phase"] in {
        GamePhase.COMPLETE.value,
        GamePhase.SCORED.value,
    }
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="qualified-repeat-source-conflict",
        resources={"analysisMode": "screening"},
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    monkeypatch.setattr(
        research,
        "simulator",
        _SequenceSimulator(
            service_harness.simulator,
            (service_harness.simulation, secondary),
        ),
    )

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=None,
    )

    assert _result_items(research, run["runCode"]) == []
    receipts = _search_receipt_items(research, run["runCode"])
    assert len(receipts) == 1
    bundle = receipts[0]["receiptBundle"]
    assert bundle["stopReason"] == "fatal_quarantine"
    assert [
        item["evaluation"]["outcome"]
        for item in bundle["evaluationReceipts"]
    ] == ["qualified", "fatal_quarantine"]
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    disagreement = disagreements[0]
    assert disagreement["category"] == "runtime_transition_defect"
    assert disagreement["runtimeEvidence"]["stage"] == (
        "candidate_output_determinism_admission"
    )
    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == service_harness.simulation.final_trace.sha256
    assert record["disposition"] == "quarantined"
    assert {
        "stage_qualified_repeat_determinism",
        "runtime_transition_defect",
    }.issubset(record["failureCategories"])
    assert record["disagreementReferences"] == [{
        "disagreementId": disagreement["disagreementId"],
        "sha256": disagreement["sha256"],
        "category": disagreement["category"],
        "status": disagreement["status"],
    }]
    names = [item[0] for item in call_order]
    assert names.count("receipt") == 1
    assert names.count("result") == 0
    assert names.count("disagreement") == 1
    assert names.count("finalize") == 1
    assert names.count("complete") == 1
    assert names.index("receipt") < names.index("disagreement")
    assert names.index("receipt") < names.index("finalize")
    assert names.index("receipt") < names.index("complete")
    assert len(qualification_calls) == 1
    assert len(decision_selection_calls) == 1
    assert decision_selection_calls[0]["profileHash"] == decision_score_profile.sha256
    assert decision_selection_calls[0]["passed"] is True
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0


def test_completed_trace_survives_construction_target_rejection(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    actual_buyer = Seat.parse(
        service_harness.simulation.final_trace.trusted_final_state["buyer"]
    )
    assumed_buyer = next(seat for seat in Seat if seat != actual_buyer)
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="construction-target",
        assumed_buyer=assumed_buyer,
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "rejected"
    assert {
        "stage_actor_safe_policy_target_verification",
        "construction_target_miss",
        "buyer_target_miss",
    }.issubset(records[0]["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "construction_target_miss"
        for item in _completed_events(research, claim)
    )
    receipt = _search_receipt_items(research, run["runCode"])[0]
    bundle = receipt["receiptBundle"]
    candidate = generator.candidates[0]
    expected_cut = OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
        candidate,
        layout=None,
    )
    assert bundle["stopReason"] == "candidates_exhausted"
    assert len(bundle["rejectionFeedbackRows"]) == 1
    feedback = bundle["rejectionFeedbackRows"][0]
    assert feedback["noGoodCut"] == expected_cut
    assert feedback["noGoodCutHash"] == expected_cut["cutSha256"]
    continuation = bundle["terminalContinuationReceipt"]
    assert continuation["exhausted"] is True
    assert continuation["precedingNoGoodCutHashes"] == [
        expected_cut["cutSha256"]
    ]
    assert feedback["applicationReceipt"]["solveEvidenceKind"] == (
        "post_rejection_continuation_receipt"
    )
    assert feedback["applicationReceipt"]["solveEvidenceHash"] == continuation[
        "sha256"
    ]
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["result"]["optimalityCutCount"] == 0
    assert generator.generate_calls == [()]
    assert generator.probe_calls == [(candidate.candidate_hash,)]
    assert generator.legacy_generate_calls == 0


def test_actor_safe_contract_target_miss_is_rejected_not_quarantined(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    actual_contract = str(
        service_harness.simulation.final_trace.trusted_final_state["contract"]
    )
    requested_contract = next(
        item for item in ("sun", "hokum", "ashkal")
        if item != actual_contract
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="contract-target-miss",
        situation={"contracts": [requested_contract]},
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "rejected"
    assert {
        "stage_actor_safe_policy_target_verification",
        "construction_target_miss",
        "contract_target_miss",
    }.issubset(records[0]["failureCategories"])
    assert records[0]["disagreementReferences"] == []
    assert research.store.list_disagreements(run_code=run["runCode"]) == []
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["result"]["quarantineCount"] == 0
    event = next(
        item for item in _completed_events(research, claim)
        if item["type"] == "construction_target_miss"
    )
    assert event["reasons"] == ["contract_target_miss"]
    assert event["expected"]["contracts"] == [requested_contract]
    assert event["actual"]["contract"] == actual_contract


def test_compiled_replay_mismatch_wins_over_actor_safe_target_miss(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    actual_contract = str(
        service_harness.simulation.final_trace.trusted_final_state["contract"]
    )
    requested_contract = next(
        item for item in ("sun", "hokum", "ashkal")
        if item != actual_contract
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="contract-target-plus-play-parity",
        situation={"contracts": [requested_contract]},
    )
    monkeypatch.setattr(
        research,
        "_compiled_play_mismatches",
        lambda *_args, **_kwargs: ("compiler_parity_defect",),
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "quarantined"
    assert {
        "stage_semantic_parity_verification",
        "compiler_parity_defect",
    }.issubset(records[0]["failureCategories"])
    assert "construction_target_miss" not in records[0]["failureCategories"]
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "compiler_parity_defect"


def test_final_match_score_parity_mismatch_is_quarantined(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    trusted_state = service_harness.simulation.final_trace.trusted_final_state
    play_plan = research.play_compiler.verify_trace(
        trusted_state,
        rules=service_harness.simulator.rules,
    )
    tampered_state = copy.deepcopy(trusted_state)
    tampered_state["matchScore"][0] += 1
    tampered_simulation = SimpleNamespace(
        final_trace=SimpleNamespace(trusted_final_state=tampered_state)
    )
    mismatch = research._compiled_play_mismatches(
        play_plan,
        tampered_simulation,
        scoring_domain=ScoringDomainIRV1.from_rules(
            service_harness.simulator.rules.data
        ),
        expected_dealer=Seat.EAST,
        starting_match_score=(0, 0),
    )
    assert mismatch == ("compiler_parity_defect",)

    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="final-match-score-parity",
    )
    monkeypatch.setattr(
        research,
        "_compiled_play_mismatches",
        lambda *_args, **_kwargs: mismatch,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "quarantined"
    assert {
        "stage_semantic_parity_verification",
        "compiler_parity_defect",
    }.issubset(records[0]["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "compiler_parity_defect"


def test_completed_trace_survives_tactical_viability_rejection(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    profile = {
        "schemaVersion": 1,
        "id": "exact_tactical_viability_screen_v1",
        "revision": "1",
        "enabled": True,
        "maximumDecisions": 4,
        "requiredTacticsOnly": True,
        "rejectWhenExactAbsent": True,
    }
    payload = _payload(
        research,
        label="tactical-pre-screen",
        lane="organic_runtime",
        resources={"tacticalViabilityPreScreen": profile},
    )
    payload["tactics"][0]["mode"] = "qualification_gate"
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="tactical-pre-screen",
        payload=payload,
    )
    evidence = SimpleNamespace(
        rejected=True,
        required_tactics=("small_card_big_gain",),
        found_tactics=(),
        probe=SimpleNamespace(status="not_found"),
        sha256=_hash("tactical-pre-screen-evidence"),
        to_dict=lambda: {
            "schemaVersion": 1,
            "rejected": True,
            "requiredTactics": ["small_card_big_gain"],
            "foundTactics": [],
        },
    )
    monkeypatch.setattr(
        service_module,
        "TacticalViabilityPreScreenV1",
        lambda *_a, **_k: SimpleNamespace(evaluate=lambda *_x, **_y: evidence),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    record = _audit_items(research, run["runCode"])[0]
    assert record["disposition"] == "rejected"
    assert {
        "stage_tactical_viability_pre_screen",
        "tactical_viability_pre_screen_rejected",
        "required_tactic_absent:small_card_big_gain",
    }.issubset(record["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "tactical_viability_pre_screen" and item["rejected"]
        for item in _completed_events(research, claim)
    )


def test_completed_trace_survives_construction_pre_screen_rejection(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    profile = {
        "schemaVersion": 1,
        "id": "actor_safe_pre_screen_v1",
        "revision": "1",
        "enabled": True,
        "maximumPlayDecisions": 4,
        "maximumBidRegret": 0,
        "maximumPlayRegret": 0,
        "maximumProbabilityDrop": 0,
        "minimumConfidence": 0,
        "rejectCatastrophicCrossing": True,
    }
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="construction-pre-screen",
        resources={"constructionPreScreen": profile},
    )
    evidence = SimpleNamespace(
        rejected=True,
        rejection_reasons=("catastrophic_threshold_crossing",),
        sha256=_hash("construction-pre-screen-evidence"),
        to_dict=lambda: {
            "schemaVersion": 1,
            "rejected": True,
            "rejectionReasons": ["catastrophic_threshold_crossing"],
        },
    )
    monkeypatch.setattr(
        service_module,
        "ConstructionPreScreenV1",
        lambda *_a, **_k: SimpleNamespace(evaluate=lambda *_x, **_y: evidence),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    record = _audit_items(research, run["runCode"])[0]
    assert record["disposition"] == "rejected"
    assert {
        "stage_construction_pre_screen",
        "construction_pre_screen_rejected",
        "catastrophic_threshold_crossing",
    }.issubset(record["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "construction_pre_screen" and item["rejected"]
        for item in _completed_events(research, claim)
    )


def test_late_qualification_rejection_reuses_retained_complete_trace_reference(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="late-qualification",
        payload=_payload(
            research,
            label="late-qualification",
            lane="organic_runtime",
        ),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])
    qualification = research._qualification

    def reject_late(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = qualification(*args, **kwargs)
        return {
            **value,
            "qualified": False,
            "truthLabel": "rejected",
            "failures": [*value["failures"], "service_fixture_late_gate"],
        }

    monkeypatch.setattr(research, "_qualification", reject_late)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    results = _result_items(research, run["runCode"])
    assert len(records) == len(results) == 1
    record = records[0]
    result = results[0]
    assert record["disposition"] == result["disposition"] == "rejected"
    assert "stage_content_qualification" in record["failureCategories"]
    assert "service_fixture_late_gate" in record["failureCategories"]
    trace_reference = result["artifacts"]["completeTrace"]
    assert {
        key: value
        for key, value in trace_reference.items()
        if key not in {"ownerKind", "ownerHash"}
    } == record["completeTrace"]
    assert trace_reference["ownerKind"] == "complete_trace_v1"
    assert trace_reference["ownerHash"] == result["traceHash"]
    assert result["artifacts"]["analysis"]["ownerKind"] == "trace_analysis_v1"
    assert result["artifacts"]["auditAnalysis"]["ownerKind"] == (
        "trace_analysis_v1"
    )
    assert result["artifacts"]["decisionAudit"]["ownerKind"] == (
        "decision_audit_v2"
    )
    expected_owned_reference_keys = {
        "uri",
        "sha256",
        "bytes",
        "mediaType",
        "name",
        "ownerKind",
        "ownerHash",
    }
    for artifact_name in (
        "completeTrace",
        "analysis",
        "auditAnalysis",
        "decisionAudit",
    ):
        assert set(result["artifacts"][artifact_name]) == (
            expected_owned_reference_keys
        )

    analysis_document = json.loads(
        research.artifact_store.get(result["artifacts"]["analysis"]["uri"])
    )
    audit_analysis_document = json.loads(
        research.artifact_store.get(result["artifacts"]["auditAnalysis"]["uri"])
    )
    audit_document = json.loads(
        research.artifact_store.get(result["artifacts"]["decisionAudit"]["uri"])
    )
    assert result["artifacts"]["analysis"]["ownerHash"] == analysis_document[
        "sha256"
    ]
    assert result["artifacts"]["auditAnalysis"]["ownerHash"] == (
        audit_analysis_document["sha256"]
    )
    assert result["provenance"]["auditAnalysisHash"] == audit_analysis_document[
        "sha256"
    ]
    assert result["artifacts"]["decisionAudit"]["ownerHash"] == audit_document[
        "sha256"
    ]
    decisions = research.get_decisions(result["resultCode"])
    assert decisions == audit_document["decisions"]
    admitted_page = research.audit_traces.list_records(
        run_code=result["runCode"],
        offset=0,
        limit=500,
    )
    retained_occurrence = next(
        item
        for item in admitted_page["items"]
        if item["traceHash"] == result["traceHash"]
        and item["source"]["candidateHash"]
        == result["provenance"]["candidateHash"]
    )
    later_occurrence = copy.deepcopy(retained_occurrence)
    later_occurrence["source"]["seed"] = int(
        later_occurrence["source"]["seed"]
    ) + 1
    later_occurrence["source"]["candidateHash"] = _hash(
        "later-same-trace-candidate"
    )
    later_occurrence["source"]["candidateOrdinal"] = int(
        later_occurrence["source"]["candidateOrdinal"]
    ) + 1
    later_occurrence["createdAt"] = "2026-08-10T01:00:00+00:00"
    later_occurrence["updatedAt"] = "2026-08-10T01:00:00+00:00"
    later_occurrence["recordHash"] = canonical_sha256(
        {
            "schemaVersion": 1,
            "traceHash": later_occurrence["traceHash"],
            "source": later_occurrence["source"],
        }
    )
    later_occurrence["auditTraceCode"] = (
        f"LAF-AUD-{later_occurrence['recordHash'][:10].upper()}"
    )
    later_occurrence = ResearchAuditTraceRecordV1.build(
        later_occurrence
    ).to_dict()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.audit_traces,
            "list_records",
            lambda **_kwargs: {
                **copy.deepcopy(admitted_page),
                "items": [
                    copy.deepcopy(later_occurrence),
                    *copy.deepcopy(admitted_page["items"]),
                ],
                "nextOffset": None,
            },
        )
        assert research.get_decisions(result["resultCode"]) == decisions
    selected = next(
        item
        for item in decisions
        if item["decisionKind"] == "play" and item["forced"] is False
    )
    prepared = service_harness.container.research_renders.prepare(
        result["resultCode"],
        decision_ordinal=int(selected["ordinal"]),
        export_preset="hooked_flagship",
    )
    source = prepared["source"]
    assert source["resultHash"] == result["resultHash"]
    assert source["traceHash"] == result["traceHash"]
    assert source["analysisHash"] == analysis_document["sha256"]
    assert source["auditHash"] == audit_document["sha256"]
    assert source["analysisProfileHash"] == analysis_document["profileHash"]
    assert source["decisionId"] == selected["decisionId"]
    assert source["exactStateId"] == selected["decisionId"]
    serialized_source = json.dumps(source, sort_keys=True)
    assert "constructionObjective" not in serialized_source
    assert "structuralObjective" not in serialized_source
    assert "solverObjective" not in serialized_source

    bundle_document = json.loads(
        research.artifact_store.get(prepared["bundleArtifact"]["uri"])
    )
    public_lifecycle = next(
        layer["content"]
        for scene in bundle_document["scenes"]
        if scene["stage"] == "continuation"
        for layer in scene["layers"]
        if layer["id"] == "public-lifecycle"
    )
    trusted_final_state = service_harness.simulation.final_trace.trusted_final_state
    assert public_lifecycle["matchScore"] == trusted_final_state["matchScore"]
    assert public_lifecycle["roundScore"] == trusted_final_state["roundScore"]
    serialized_bundle = json.dumps(bundle_document, sort_keys=True)
    assert "structuralObjectiveScore" not in serialized_bundle
    assert "constructionObjective" not in serialized_bundle

    def put_owned(
        document: dict[str, Any], *, media_type: str, owner_kind: str
    ) -> dict[str, Any]:
        payload = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        reference = research.artifact_store.put(
            payload,
            media_type=media_type,
            name=f"{document['sha256']}.json",
        )
        return {
            **reference,
            "ownerKind": owner_kind,
            "ownerHash": document["sha256"],
        }

    def rehash_result(value: dict[str, Any]) -> None:
        provenance = dict(value["provenance"])
        identity = {
            "schemaVersion": 1,
            "runCode": value["runCode"],
            "shardId": provenance["researchShardId"],
            "shardIndex": provenance["researchShardIndex"],
            "recipeHash": provenance["recipeHash"],
            "traceHash": value["traceHash"],
            "analysisHash": value["artifacts"]["analysis"]["ownerHash"],
            "auditAnalysisHash": value["artifacts"]["auditAnalysis"][
                "ownerHash"
            ],
            "auditHash": value["artifacts"]["decisionAudit"]["ownerHash"],
            "auctionPlanHash": provenance["auctionCompilerPlanHash"],
        }
        if provenance["decisionScoreParityStatus"] == "matched":
            identity.update(
                {
                    "primaryDecisionScoreAnalysisHash": provenance[
                        "primaryDecisionScoreAnalysisHash"
                    ],
                    "decisionScoreParityHash": provenance[
                        "decisionScoreParityHash"
                    ],
                }
            )
        value["resultHash"] = canonical_sha256(identity)
        value["resultCode"] = f"LAF-{value['resultHash'][:10].upper()}"

    def expect_rejected(value: dict[str, Any], message: str) -> None:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                research.store,
                "get_result",
                lambda _code: copy.deepcopy(value),
            )
            with pytest.raises(ValueError, match=message):
                research.get_decisions(value["resultCode"])

    ownerless = copy.deepcopy(result)
    ownerless["artifacts"]["completeTrace"].pop("ownerKind")
    ownerless["artifacts"]["completeTrace"].pop("ownerHash")
    expect_rejected(ownerless, "not closed or owned")

    false_owner = copy.deepcopy(result)
    false_owner["artifacts"]["completeTrace"]["ownerHash"] = "0" * 64
    expect_rejected(false_owner, "semantic owner identity mismatch")

    missing_audit_analysis = copy.deepcopy(result)
    missing_audit_analysis["artifacts"].pop("auditAnalysis")
    expect_rejected(missing_audit_analysis, "not closed or owned")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.store,
            "get_result",
            lambda _code: copy.deepcopy(missing_audit_analysis),
        )
        with pytest.raises(ValueError, match="not closed or owned"):
            service_harness.container.research_renders.prepare(
                missing_audit_analysis["resultCode"],
                decision_ordinal=int(selected["ordinal"]),
                export_preset="hooked_flagship",
            )

    false_audit_analysis_owner = copy.deepcopy(result)
    false_audit_analysis_owner["artifacts"]["auditAnalysis"]["ownerHash"] = (
        "1" * 64
    )
    expect_rejected(
        false_audit_analysis_owner,
        "audit analysis semantic owner identity mismatch",
    )

    null_audit_analysis_owner = copy.deepcopy(result)
    null_audit_analysis_owner["artifacts"]["auditAnalysis"]["ownerKind"] = None
    null_audit_analysis_owner["artifacts"]["auditAnalysis"]["ownerHash"] = None
    expect_rejected(null_audit_analysis_owner, "semantic owner kind mismatch")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.store,
            "get_result",
            lambda _code: copy.deepcopy(null_audit_analysis_owner),
        )
        with pytest.raises(ValueError, match="semantic owner"):
            service_harness.container.research_renders.prepare(
                null_audit_analysis_owner["resultCode"],
                decision_ordinal=int(selected["ordinal"]),
                export_preset="hooked_flagship",
            )

    false_analysis_owner = copy.deepcopy(result)
    false_analysis_owner["artifacts"]["analysis"]["ownerHash"] = "2" * 64
    expect_rejected(false_analysis_owner, "analysis semantic owner identity mismatch")

    false_audit_owner = copy.deepcopy(result)
    false_audit_owner["artifacts"]["decisionAudit"]["ownerHash"] = "3" * 64
    expect_rejected(
        false_audit_owner,
        "DecisionAuditV2 semantic owner identity mismatch",
    )

    substituted_trace_membership = copy.deepcopy(result)
    substituted_trace_membership["traceHash"] = "f" * 64
    rehash_result(substituted_trace_membership)
    expect_rejected(substituted_trace_membership, "complete-trace membership mismatch")

    substituted_analysis_membership = copy.deepcopy(result)
    substituted_analysis_membership["scoreEvidence"][
        "decisionSelectionQualification"
    ]["analysisHash"] = "e" * 64
    substituted_analysis_membership["scoreEvidence"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in substituted_analysis_membership[
                "scoreEvidence"
            ].items()
            if key != "sha256"
        }
    )
    expect_rejected(substituted_analysis_membership, "analysis membership mismatch")

    substituted_analysis = copy.deepcopy(analysis_document)
    substituted_analysis["analyzerId"] = "substituted_trace_analyzer_v1"
    analysis_identity = {
        "schemaVersion": substituted_analysis["schemaVersion"],
        "traceHash": substituted_analysis["traceHash"],
        "profileHash": substituted_analysis["profileHash"],
        "analyzerId": substituted_analysis["analyzerId"],
        "analyzerVersion": substituted_analysis["analyzerVersion"],
        "decisionAnalysisHashes": [
            item["analysisHash"] for item in substituted_analysis["decisions"]
        ],
        "laffaEvidenceHash": substituted_analysis["laffaEvidence"]["sha256"],
    }
    substituted_analysis["sha256"] = canonical_sha256(analysis_identity)
    substituted_audit = copy.deepcopy(audit_document)
    for item in substituted_audit["decisions"]:
        if item["decisionId"] in {
            decision["decisionId"] for decision in substituted_analysis["decisions"]
        }:
            item["provenance"]["analyzerId"] = substituted_analysis["analyzerId"]
    substituted_audit["sha256"] = canonical_sha256(
        {key: value for key, value in substituted_audit.items() if key != "sha256"}
    )
    same_game_substitution = copy.deepcopy(result)
    same_game_substitution["artifacts"]["analysis"] = put_owned(
        substituted_analysis,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    same_game_substitution["artifacts"]["decisionAudit"] = put_owned(
        substituted_audit,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    rehash_result(same_game_substitution)
    expect_rejected(same_game_substitution, "analysis authority mismatch")

    substituted_audit = copy.deepcopy(audit_document)
    substituted_row = next(
        item
        for item in substituted_audit["decisions"]
        if item["decisionId"] == selected["decisionId"]
    )
    substituted_row["recommendedAction"] = next(
        action
        for action in substituted_row["legalActions"]
        if action != substituted_row["recommendedAction"]
    )
    substituted_row["actualMatchesRecommendation"] = (
        substituted_row["actualAction"] == substituted_row["recommendedAction"]
    )
    substituted_audit["sha256"] = canonical_sha256(
        {key: value for key, value in substituted_audit.items() if key != "sha256"}
    )
    audit_substitution = copy.deepcopy(result)
    audit_substitution["artifacts"]["decisionAudit"] = put_owned(
        substituted_audit,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    rehash_result(audit_substitution)
    expect_rejected(audit_substitution, "does not match its parsed analysis")

    governed_audit = copy.deepcopy(audit_document)
    governed_row = next(
        item
        for item in governed_audit["decisions"]
        if item["decisionKind"] in {"project", "escalation"}
    )
    governed_row["actualAction"] = {
        **governed_row["actualAction"],
        "eventType": "substituted_governed_event",
    }
    governed_audit["sha256"] = canonical_sha256(
        {key: value for key, value in governed_audit.items() if key != "sha256"}
    )
    governed_substitution = copy.deepcopy(result)
    governed_substitution["artifacts"]["decisionAudit"] = put_owned(
        governed_audit,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    rehash_result(governed_substitution)
    expect_rejected(governed_substitution, "does not match its trace event")

    divergent_analysis = copy.deepcopy(analysis_document)
    divergent_laffa = divergent_analysis["laffaEvidence"]
    divergent_laffa["classification"] = (
        f"{divergent_laffa['classification']}_substituted"
    )
    divergent_laffa["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in divergent_laffa.items()
            if key not in {"sha256", "scenarioGraph"}
        }
    )
    divergent_analysis["sha256"] = canonical_sha256(
        {
            "schemaVersion": divergent_analysis["schemaVersion"],
            "traceHash": divergent_analysis["traceHash"],
            "profileHash": divergent_analysis["profileHash"],
            "analyzerId": divergent_analysis["analyzerId"],
            "analyzerVersion": divergent_analysis["analyzerVersion"],
            "decisionAnalysisHashes": [
                item["analysisHash"] for item in divergent_analysis["decisions"]
            ],
            "laffaEvidenceHash": divergent_laffa["sha256"],
        }
    )
    coherent_same_trace_substitution = copy.deepcopy(result)
    divergent_reference = put_owned(
        divergent_analysis,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    coherent_same_trace_substitution["artifacts"]["analysis"] = copy.deepcopy(
        divergent_reference
    )
    coherent_same_trace_substitution["artifacts"]["auditAnalysis"] = (
        copy.deepcopy(divergent_reference)
    )
    coherent_same_trace_substitution["provenance"]["auditAnalysisHash"] = (
        divergent_analysis["sha256"]
    )
    coherent_same_trace_substitution["classification"] = divergent_laffa[
        "classification"
    ]
    coherent_selection = coherent_same_trace_substitution["scoreEvidence"][
        "decisionSelectionQualification"
    ]
    coherent_selection["analysisHash"] = divergent_analysis["sha256"]
    coherent_same_trace_substitution["scoreEvidence"]["sha256"] = (
        canonical_sha256(
            {
                key: value
                for key, value in coherent_same_trace_substitution[
                    "scoreEvidence"
                ].items()
                if key != "sha256"
            }
        )
    )
    rehash_result(coherent_same_trace_substitution)
    expect_rejected(
        coherent_same_trace_substitution,
        "not the canonical analyzer output",
    )

    substituted_simulation = service_harness.simulator.simulate(
        2,
        dealer=Seat.EAST,
        match_score=(0, 0),
    )
    substituted_trace = substituted_simulation.final_trace
    substituted_trace_document = substituted_trace.to_dict(trusted=True)
    substitution_analyzer = research.analyzers["screening"]
    profile = substitution_analyzer.profile
    heuristic_profile = TraceAnalysisProfileV1.build(
        bid_samples=profile.bid_samples,
        play_samples=profile.play_samples,
        exact_allocation_limit=profile.exact_allocation_limit,
        exact_search_max_remaining=0,
        maximum_sampling_attempts=profile.maximum_sampling_attempts,
        exact_search_max_nodes=profile.exact_search_max_nodes,
        exact_search_max_seconds=profile.exact_search_max_seconds,
        exact_search_max_depth=profile.exact_search_max_depth,
        exact_search_max_cache_entries=profile.exact_search_max_cache_entries,
        search_backend_id=profile.search_backend_id,
        objective=profile.objective,
        use_conventions=profile.use_conventions,
        belief_policy=substitution_analyzer.belief_policy,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(substitution_analyzer, "profile", heuristic_profile)
        substituted_analysis_result = substitution_analyzer.analyze_trace(
            substituted_trace
        )
    substituted_analysis_document = substituted_analysis_result.to_dict()
    substituted_audit_document = research._decision_audit(
        substituted_trace,
        substituted_analysis_result,
    )
    coordinated_substitution = copy.deepcopy(result)
    coordinated_substitution["traceHash"] = substituted_trace.sha256
    coordinated_substitution["gameHash"] = substituted_trace.game_hash
    coordinated_substitution["artifacts"]["completeTrace"] = put_owned(
        substituted_trace_document,
        media_type="application/vnd.balootx.complete-trace-v1+json",
        owner_kind="complete_trace_v1",
    )
    coordinated_substitution["artifacts"]["analysis"] = put_owned(
        substituted_analysis_document,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    coordinated_substitution["artifacts"]["auditAnalysis"] = put_owned(
        substituted_analysis_document,
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        owner_kind="trace_analysis_v1",
    )
    coordinated_substitution["artifacts"]["decisionAudit"] = put_owned(
        substituted_audit_document,
        media_type="application/vnd.balootx.decision-audit-v2+json",
        owner_kind="decision_audit_v2",
    )
    coordinated_substitution["provenance"]["auditAnalysisHash"] = (
        substituted_analysis_result.sha256
    )
    coordinated_substitution["decisionCount"] = substituted_audit_document[
        "decisionCount"
    ]
    selection_qualification = coordinated_substitution["scoreEvidence"][
        "decisionSelectionQualification"
    ]
    selection_qualification["analysisHash"] = substituted_analysis_result.sha256
    coordinated_substitution["scoreEvidence"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in coordinated_substitution["scoreEvidence"].items()
            if key != "sha256"
        }
    )
    rehash_result(coordinated_substitution)
    expect_rejected(
        coordinated_substitution,
        "retained audit occurrence is unavailable",
    )


def test_retained_ortools_screening_result_uses_deep_audit_analysis_for_render(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    label = "ortools-screening-audit-render"
    payload = _payload(
        research,
        label=label,
        lane="ortools",
        resources={"analysisMode": "screening"},
    )
    payload["schemaVersion"] = 3
    payload.pop("profiles")
    payload["tactics"] = []
    payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": service_harness.container.configuration.resolve()[
            "bundle"
        ]["revisionId"],
    }
    spec = _deal_spec(
        label,
        assumed_buyer=None,
        maximum_solutions=1,
    )
    monkeypatch.setattr(
        research,
        "deal_generator",
        _SequentialGeneratorStub(label, candidate_count=1),
    )
    created = _store_validated_legacy_recipe(research, payload)
    stored_recipe = research.store.get_recipe(created["code"])
    assert stored_recipe is not None
    recipe, context = research._build_recipe_and_context(stored_recipe)
    assert context is not None
    run = research._submit_recipe_run(
        recipe.code,
        idempotency_key=label,
        schedule=False,
    )
    claim = research.store.claim_shard(
        run["runCode"],
        worker_id=f"audit-service-worker:{label}",
        lease_seconds=60,
    )
    assert claim is not None
    _stub_ortools_compilers(
        research,
        monkeypatch,
        label=label,
        spec=spec,
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])
    base_qualification = research._qualification

    def retain_as_rejected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = base_qualification(*args, **kwargs)
        return {
            **value,
            "qualified": False,
            "truthLabel": "rejected",
            "failures": [*value["failures"], "service_fixture_late_gate"],
        }

    monkeypatch.setattr(research, "_qualification", retain_as_rejected)

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=context,
    )

    results = _result_items(research, run["runCode"])
    assert len(results) == 1
    result = results[0]
    assert result["provenance"]["generationLane"] == "ortools"
    assert result["provenance"]["finalistPrimaryPolicy"] is True
    assert result["provenance"]["decisionScoreParityStatus"] == "matched"
    assert result["provenance"]["structuralObjectiveScore"] == 1
    selected_analysis = json.loads(
        research.artifact_store.get(result["artifacts"]["analysis"]["uri"])
    )
    audit_analysis = json.loads(
        research.artifact_store.get(result["artifacts"]["auditAnalysis"]["uri"])
    )
    audit_document = json.loads(
        research.artifact_store.get(result["artifacts"]["decisionAudit"]["uri"])
    )
    assert selected_analysis["profileHash"] == context.analyzers[
        "screening"
    ].profile.sha256
    assert audit_analysis["profileHash"] == context.analyzers["deep"].profile.sha256
    assert selected_analysis["sha256"] != audit_analysis["sha256"]
    assert result["provenance"]["auditAnalysisHash"] == audit_analysis["sha256"]
    assert result["provenance"]["primaryDecisionScoreAnalysisHash"] == (
        audit_analysis["sha256"]
    )
    assert result["artifacts"]["auditAnalysis"]["ownerHash"] == audit_analysis[
        "sha256"
    ]
    expected_identity = {
        "schemaVersion": 1,
        "runCode": result["runCode"],
        "shardId": result["provenance"]["researchShardId"],
        "shardIndex": result["provenance"]["researchShardIndex"],
        "recipeHash": result["provenance"]["recipeHash"],
        "traceHash": result["traceHash"],
        "analysisHash": selected_analysis["sha256"],
        "auditAnalysisHash": audit_analysis["sha256"],
        "auditHash": audit_document["sha256"],
        "primaryDecisionScoreAnalysisHash": result["provenance"][
            "primaryDecisionScoreAnalysisHash"
        ],
        "decisionScoreParityHash": result["provenance"][
            "decisionScoreParityHash"
        ],
        "auctionPlanHash": result["provenance"]["auctionCompilerPlanHash"],
    }
    assert result["resultHash"] == canonical_sha256(expected_identity)
    assert result["resultCode"] == (
        f"LAF-{result['resultHash'][:10].upper()}"
    )

    decisions = research.get_decisions(result["resultCode"])
    assert decisions == audit_document["decisions"]
    selected = next(
        item
        for item in decisions
        if item["decisionKind"] == "play" and item["forced"] is False
    )
    prepared = service_harness.container.research_renders.prepare(
        result["resultCode"],
        decision_ordinal=int(selected["ordinal"]),
        export_preset="hooked_flagship",
    )
    source = prepared["source"]
    assert source["resultHash"] == result["resultHash"]
    assert source["traceHash"] == result["traceHash"]
    assert source["analysisHash"] == audit_analysis["sha256"]
    assert source["analysisProfileHash"] == audit_analysis["profileHash"]
    assert source["analyzerId"] == audit_analysis["analyzerId"]
    assert source["analyzerVersion"] == audit_analysis["analyzerVersion"]
    assert source["auditHash"] == audit_document["sha256"]
    assert source["decisionId"] == selected["decisionId"]
    serialized_source = json.dumps(source, sort_keys=True)
    assert "structuralObjective" not in serialized_source
    assert "constructionObjective" not in serialized_source
    assert "solverObjective" not in serialized_source

    bundle_document = json.loads(
        research.artifact_store.get(prepared["bundleArtifact"]["uri"])
    )
    public_lifecycle = next(
        layer["content"]
        for scene in bundle_document["scenes"]
        if scene["stage"] == "continuation"
        for layer in scene["layers"]
        if layer["id"] == "public-lifecycle"
    )
    trusted = json.loads(
        research.artifact_store.get(result["artifacts"]["completeTrace"]["uri"])
    )["trustedFinalState"]
    assert public_lifecycle["matchScore"] == trusted["matchScore"]
    assert public_lifecycle["roundScore"] == trusted["roundScore"]
    serialized_bundle = json.dumps(bundle_document, sort_keys=True)
    assert "structuralObjectiveScore" not in serialized_bundle
    assert "constructionObjective" not in serialized_bundle

    def put_cas(
        document: dict[str, Any],
        *,
        media_type: str,
    ) -> dict[str, Any]:
        payload_bytes = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return dict(
            research.artifact_store.put(
                payload_bytes,
                media_type=media_type,
                name=f"{document['sha256']}.json",
            )
        )

    def rehash_ortools_result(value: dict[str, Any]) -> None:
        provenance = value["provenance"]
        identity = {
            "schemaVersion": 1,
            "runCode": value["runCode"],
            "shardId": provenance["researchShardId"],
            "shardIndex": provenance["researchShardIndex"],
            "recipeHash": provenance["recipeHash"],
            "traceHash": value["traceHash"],
            "analysisHash": value["artifacts"]["analysis"]["ownerHash"],
            "auditAnalysisHash": value["artifacts"]["auditAnalysis"][
                "ownerHash"
            ],
            "auditHash": value["artifacts"]["decisionAudit"]["ownerHash"],
            "primaryDecisionScoreAnalysisHash": provenance[
                "primaryDecisionScoreAnalysisHash"
            ],
            "decisionScoreParityHash": provenance["decisionScoreParityHash"],
            "auctionPlanHash": provenance["auctionCompilerPlanHash"],
        }
        value["resultHash"] = canonical_sha256(identity)
        value["resultCode"] = f"LAF-{value['resultHash'][:10].upper()}"

    def expect_ortools_rejected(value: dict[str, Any], message: str) -> None:
        def unexpected_canonical_reanalysis(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError(
                "incoherent retained artifacts must fail before canonical reanalysis"
            )

        with monkeypatch.context() as scoped:
            scoped.setattr(
                research.store,
                "get_result",
                lambda _code: copy.deepcopy(value),
            )
            for analyzer_type in {
                type(analyzer) for analyzer in context.analyzers.values()
            }:
                scoped.setattr(
                    analyzer_type,
                    "analyze_trace",
                    unexpected_canonical_reanalysis,
                )
            with pytest.raises(ValueError, match=message):
                research.get_decisions(value["resultCode"])

    primary_substitution = copy.deepcopy(result)
    primary_substitution["artifacts"]["primaryDecisionScoreAnalysis"] = put_cas(
        selected_analysis,
        media_type=(
            "application/vnd.balootx."
            "primary-decision-score-analysis-v1+json"
        ),
    )
    expect_ortools_rejected(
        primary_substitution,
        "primary decision-score analysis mismatch",
    )

    parity_document = json.loads(
        research.artifact_store.get(
            result["artifacts"]["decisionScoreParity"]["uri"]
        )
    )
    substituted_parity = copy.deepcopy(parity_document)
    substituted_parity["structuralObjectiveCompared"] = True
    substituted_parity["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in substituted_parity.items()
            if key != "sha256"
        }
    )
    parity_substitution = copy.deepcopy(result)
    parity_substitution["artifacts"]["decisionScoreParity"] = put_cas(
        substituted_parity,
        media_type="application/vnd.balootx.decision-score-parity-v1+json",
    )
    parity_substitution["provenance"]["decisionScoreParityHash"] = (
        substituted_parity["sha256"]
    )
    rehash_ortools_result(parity_substitution)
    expect_ortools_rejected(
        parity_substitution,
        "decision-score parity evidence mismatch",
    )

    drifted_run = copy.deepcopy(research.store.get_run(run["runCode"]))
    assert drifted_run is not None
    drifted_run["configurationAuthority"] = copy.deepcopy(
        drifted_run["configurationAuthority"]
    )
    drifted_run["configurationAuthority"]["sha256"] = "0" * 64
    with monkeypatch.context() as scoped:
        scoped.setattr(
            research.store,
            "get_run",
            lambda _code: copy.deepcopy(drifted_run),
        )
        with pytest.raises(ValueError, match="configuration authority mismatch"):
            research.get_decisions(result["resultCode"])


def test_completed_all_pass_trace_is_retained_as_rejected_audit_evidence(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    all_pass = _completed_all_pass_simulation(service_harness.simulator)
    assert all_pass.publishable_laffa is False
    assert all_pass.final_trace.trusted_final_state["phase"] == (
        GamePhase.COMPLETE.value
    )
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="all-pass-complete",
        payload=_payload(
            research,
            label="all-pass-complete",
            lane="organic_runtime",
        ),
        simulation=all_pass,
    )

    def forbid_sequential_search(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("organic Research must not enter sequential OR search")

    monkeypatch.setattr(
        service_module,
        "run_sequential_candidate_search_v1",
        forbid_sequential_search,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == all_pass.final_trace.sha256
    assert record["status"] == "finalized"
    assert record["disposition"] == "rejected"
    assert record["attemptStatus"] == "completed"
    assert {
        "stage_organic_simulation",
        "organic_all_pass_complete_trace",
    }.issubset(record["failureCategories"])
    assert record["auditOnly"] is True
    assert record["publicationEligible"] is False
    assert record["replayValidation"]["status"] == "verified"
    assert record["observerSafety"]["status"] == "verified"
    assert _result_items(research, run["runCode"]) == []
    assert _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    ) == []
    assert any(
        item["type"] == "recipe_infeasible"
        and item["reason"] == "organic_runtime_all_pass"
        for item in _completed_events(research, claim)
    )


def test_constructed_completed_all_pass_trace_is_retained_as_rejected_audit_evidence(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    all_pass = _completed_all_pass_simulation(service_harness.simulator)
    assert all_pass.publishable_laffa is False
    assert all_pass.final_trace.trusted_final_state["phase"] == (
        GamePhase.COMPLETE.value
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="constructed-all-pass-complete",
        simulation=all_pass,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == all_pass.final_trace.sha256
    assert record["status"] == "finalized"
    assert record["disposition"] == "rejected"
    assert record["attemptStatus"] == "completed"
    assert {
        "stage_candidate_simulation",
        "all_pass_unscored_complete_trace",
    }.issubset(record["failureCategories"])
    assert record["auditOnly"] is True
    assert record["publicationEligible"] is False
    assert record["replayValidation"]["status"] == "verified"
    assert record["observerSafety"]["status"] == "verified"
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "recipe_infeasible"
        and item["reason"] == "constructed_auction_all_pass"
        for item in _completed_events(research, claim)
    )


def test_incomplete_unscored_trace_is_not_admitted_or_publishable(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    evaluations: list[Any] = []
    original_build_evaluation = service_module.SequentialCandidateEvaluationV1.build

    def build_evaluation_spy(_cls: type[Any], **kwargs: Any) -> Any:
        evaluation = original_build_evaluation(**kwargs)
        evaluations.append(evaluation)
        return evaluation

    monkeypatch.setattr(
        service_module.SequentialCandidateEvaluationV1,
        "build",
        classmethod(build_evaluation_spy),
    )
    unscored_trace = SimpleNamespace(
        engine_version=service_harness.simulation.final_trace.engine_version,
        sha256=_hash("unscored-all-pass-trace"),
        trusted_final_state={"phase": GamePhase.BIDDING.value},
    )
    all_pass = SimpleNamespace(
        publishable_laffa=False,
        final_trace=unscored_trace,
    )
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="all-pass-unscored",
        payload=_payload(
            research,
            label="all-pass-unscored",
            lane="organic_runtime",
        ),
        simulation=all_pass,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    assert _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    ) == []
    assert len(evaluations) == 1
    assert evaluations[0].outcome.value == "incomplete"
    assert evaluations[0].failure_codes == ("authoritative_replay_incomplete",)
    assert not any(
        item["type"] == "recipe_infeasible"
        for item in _completed_events(research, claim)
    )
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["result"]["resultCount"] == 0
    assert shard["result"]["qualifiedCount"] == 0
    assert shard["result"]["rejectedCount"] == 0
    assert shard["result"]["quarantineCount"] == 0


def test_constructed_incomplete_trace_stops_without_no_good_cut(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    incomplete_trace = SimpleNamespace(
        engine_version=service_harness.simulation.final_trace.engine_version,
        sha256=_hash("constructed-incomplete-trace"),
        trusted_final_state={"phase": GamePhase.BIDDING.value},
    )
    incomplete = SimpleNamespace(
        publishable_laffa=True,
        final_trace=incomplete_trace,
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="constructed-incomplete",
        simulation=incomplete,
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    receipt = _search_receipt_items(research, run["runCode"])[0]
    bundle = receipt["receiptBundle"]
    assert bundle["stopReason"] == "incomplete"
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0


@pytest.mark.real_clock
def test_cancelled_candidate_evaluation_persists_unadmitted_receipt_without_effects(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="constructed-cancelled",
        resources={
            "auctionConstructionMode": "cp_sat_feasibility_line",
            "playConstructionMode": "cp_sat_feasibility_line",
        },
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    effect_calls: list[str] = []
    receipt_calls: list[dict[str, Any]] = []
    original_put_receipt = research.store.put_sequential_search_receipt
    original_put_result = research.store.put_result
    original_put_disagreement = research.store.put_disagreement
    original_finalize = research.audit_traces.finalize
    original_complete = research.store.complete_shard

    def put_receipt_spy(*args: Any, **kwargs: Any) -> Any:
        receipt_calls.append(copy.deepcopy(kwargs))
        return original_put_receipt(*args, **kwargs)

    def put_result_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("result")
        return original_put_result(*args, **kwargs)

    def put_disagreement_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("disagreement")
        return original_put_disagreement(*args, **kwargs)

    def finalize_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("finalize")
        return original_finalize(*args, **kwargs)

    def complete_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("complete")
        return original_complete(*args, **kwargs)

    def cancel_during_auction(*_args: Any, **_kwargs: Any) -> Any:
        raise InterruptedError("service fixture cancellation")

    monkeypatch.setattr(
        research.store,
        "put_sequential_search_receipt",
        put_receipt_spy,
    )
    monkeypatch.setattr(research.store, "put_result", put_result_spy)
    monkeypatch.setattr(
        research.store,
        "put_disagreement",
        put_disagreement_spy,
    )
    monkeypatch.setattr(research.audit_traces, "finalize", finalize_spy)
    monkeypatch.setattr(research.store, "complete_shard", complete_spy)
    monkeypatch.setattr(research, "_compile_auction", cancel_during_auction)

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=None,
    )

    receipts = _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    )
    assert len(receipts) == 1
    persisted = receipts[0]
    assert persisted["admitted"] is False
    assert persisted["shardId"] == claim["shardId"]
    assert persisted["attemptCount"] == claim["attemptCount"]
    assert persisted["fencingToken"] == claim["fencingToken"]
    assert persisted["seed"] == claim["seedStart"]
    assert receipt_calls == [{
        "run_code": run["runCode"],
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
        "seed": claim["seedStart"],
    }]
    bundle = persisted["receiptBundle"]
    assert bundle["stopReason"] == "cancelled"
    assert bundle["failureCodes"] == ["candidate_evaluation_cancelled"]
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert bundle["evaluationReceipts"][0]["evaluation"]["outcome"] == (
        "cancelled"
    )
    assert effect_calls == []
    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    assert research.store.list_disagreements(run_code=run["runCode"]) == []
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["status"] == "running"
    assert shard.get("result") is None
    current_run = research.store.get_run(run["runCode"])
    assert current_run is not None
    assert current_run["resultCount"] == 0
    assert current_run["quarantineCount"] == 0
    assert "solverIncomplete" not in current_run
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0


class _ReplayInvalidTrace:
    def __init__(self, valid: Any) -> None:
        self._valid = valid
        self.sha256 = valid.sha256
        self.engine_version = valid.engine_version

    def to_dict(self, *, trusted: bool = False) -> dict[str, Any]:
        value = copy.deepcopy(self._valid.to_dict(trusted=True))
        value["stateHash"] = "0" * 64
        return value


def test_replay_invalid_trace_is_quarantined_without_admission_or_result(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    invalid = SimpleNamespace(
        publishable_laffa=True,
        final_trace=_ReplayInvalidTrace(service_harness.simulation.final_trace),
    )
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="replay-invalid",
        payload=_payload(
            research,
            label="replay-invalid",
            lane="organic_runtime",
        ),
        simulation=invalid,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "runtime_transition_defect"
    assert disagreements[0]["runtimeEvidence"]["stage"] == (
        "authoritative_complete_trace_replay"
    )
    assert any(
        item["type"] == "semantic_replay_quarantine"
        and item["category"] == "runtime_transition_defect"
        for item in _completed_events(research, claim)
    )
