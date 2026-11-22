from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from baloot_platform.challenge.constraint_deal_search import (
    ConstraintDealCandidateV1,
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.core.enums import GamePhase, Seat
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
