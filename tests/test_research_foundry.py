from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from service_test_client import OperatorClient as TestClient

from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.research import (
    ResearchRecipeV1,
    ResearchRecipeV2,
    SQLiteResearchStoreV1,
    build_research_recipe,
)
from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1
from baloot_platform.challenge.constraint_deal_search import (
    ConstraintDealCandidateV1,
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.research.decision_audit import (
    validate_governed_decision_audit_v2,
)
from baloot_platform.research.service import ResearchFoundryServiceV1
from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
from baloot_platform.research.result_diversity import select_research_results
from baloot_platform.service.container import build_container


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _SequentialDealGeneratorStub:
    descriptor = OrToolsCpSatDealGeneratorV1.descriptor

    def __init__(self, label: str) -> None:
        self.label = label
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
        rules,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
    ) -> ConstraintDealCandidateV1:
        ordinal = len(rejected_candidates)
        deck = tuple(range(ordinal, 32)) + tuple(range(ordinal))
        model_hash = _hash(f"foundry-generator-model:{self.label}")
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
        rules,
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        **_kwargs,
    ) -> ConstraintDealCandidateV1 | None:
        self.generate_calls.append(
            tuple(candidate.candidate_hash for candidate in rejected_candidates)
        )
        if rejected_candidates:
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
        rejected_candidates: tuple[ConstraintDealCandidateV1, ...],
        **_kwargs,
    ):
        self.probe_calls.append(
            tuple(candidate.candidate_hash for candidate in rejected_candidates)
        )
        assert rejected_candidates
        authority = rejected_candidates[0]
        return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
            profile_hash=spec.sha256,
            model_hash=authority.model_hash,
            backend_version=authority.backend_version,
            solver_status="infeasible",
            solution_ordinal=len(rejected_candidates),
            search_identity_hash=authority.search_identity_hash,
            preceding_no_good_cut_hashes=self._cut_hashes(rejected_candidates),
            no_good_cut_type=authority.no_good_cut_type,
            exhausted=True,
            witness=None,
            structural_objective_score=authority.structural_objective_score,
        ).verify()

    def generate(self, *_args, **_kwargs):
        self.legacy_generate_calls += 1
        raise AssertionError("Research service must not use legacy batch generation")


def _ortools_run_identity(research) -> dict[str, object]:
    identity = research._deal_generator_identity()
    return {
        "generationLane": "ortools",
        "dealGeneratorIdentity": copy.deepcopy(identity),
        "dealGeneratorDescriptorHash": identity["descriptorHash"],
    }


def _store_validated_legacy_recipe(
    research: ResearchFoundryServiceV1,
    payload: dict,
) -> dict:
    """Seed an immutable historical recipe without reopening legacy authoring."""

    validated = research.validate_recipe(payload)
    assert validated["valid"] is True, validated["issues"]
    recipe = validated["recipe"]
    stored = research.store.put_recipe(recipe)
    return {
        **stored,
        "lifecycleStatus": research.store.recipe_status(stored["code"]),
    }


def test_research_perspective_loader_returns_one_isolated_observer():
    payload = {
        "laffaEvidence": {
            "perspectives": [
                {"decisionId": "D-2", "observerSeat": "south", "actingSeat": "east"},
                {"decisionId": "D-1", "observerSeat": "north", "actingSeat": "north"},
                {"decisionId": "D-1", "observerSeat": "south", "actingSeat": "north"},
            ]
        }
    }
    service = object.__new__(ResearchFoundryServiceV1)
    service.get_result = lambda code: {
        "schemaVersion": 1,
        "artifacts": {"analysis": {"uri": "cas://analysis"}},
    } if code == "LAF-TEST" else None
    service.artifact_store = SimpleNamespace(
        get=lambda uri: json.dumps(payload).encode("utf-8")
    )
    rows = service.get_perspectives("LAF-TEST", "south")
    assert [item["decisionId"] for item in rows] == ["D-1", "D-2"]
    assert {item["observerSeat"] for item in rows} == {"south"}


def _recipe(**overrides):
    value = {
        "schemaVersion": 1,
        "titleAr": "ورقة صغيرة تفتح مكسب كبير",
        "output": {"kind": "laffa", "resultCount": 2},
        "profiles": {
            "domainHash": _hash("domain"),
            "bundleHash": _hash("bundle"),
            "rulesHash": _hash("rules"),
            "conventionHash": _hash("convention"),
            "agentRelease": "agent@frozen",
            "analysisProfileHash": _hash("analysis"),
            "scoreProfileHash": _hash("score"),
        },
        "situation": {
            "dealer": "east",
            "buyer": "south",
            "contracts": ["sun"],
            "matchScore": [120, 118],
            "requiredCards": {"south": ["SA"]},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [{
            "kind": "small_card_big_gain",
            "mode": "qualification_gate",
            "minimumScore": 70,
            "weight": 1,
            "perspectives": ["south"],
        }],
        "qualification": {
            "minimumTruth": "robust_best_known",
            "minimumConfidence": 0.8,
            "minimumSurroundingQuality": 75,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0,
        },
        "resources": {
            "seedStart": 100,
            "seedCount": 20,
            "seedsPerShard": 5,
            "candidatesPerSeed": 2,
            "solverTimeoutSeconds": 10,
            "maximumConflicts": 250_000,
            "analysisMode": "cascaded",
        },
        "status": "published",
    }
    value.update(overrides)
    return value


def test_research_recipe_is_closed_hashed_and_round_trippable():
    recipe = ResearchRecipeV1.build(_recipe())
    assert recipe.code.startswith("RCP-")
    assert recipe.situation["matchScore"] == [120, 118]
    assert recipe.tactics[0]["kind"] == "small_card_big_gain"
    assert recipe.resources["deepVerificationPolicy"] == "required_gates"
    assert "playConstructionMode" not in recipe.resources
    assert ResearchRecipeV1.build(recipe.to_dict()) == recipe

    invalid = _recipe()
    invalid["hiddenOverride"] = True
    with pytest.raises(ValueError, match="unknown research recipe keys"):
        ResearchRecipeV1.build(invalid)

    explicit = _recipe()
    explicit["resources"]["playConstructionMode"] = (
        "actor_safe_agent_verified"
    )
    assert ResearchRecipeV1.build(explicit).resources[
        "playConstructionMode"
    ] == "actor_safe_agent_verified"

    invalid = _recipe()
    invalid["resources"]["playConstructionMode"] = "omniscient_magic"
    with pytest.raises(ValueError, match="play-construction mode"):
        ResearchRecipeV1.build(invalid)


def test_research_run_submission_revalidates_frozen_authority_before_scheduling():
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = SimpleNamespace(get_recipe=lambda code: _recipe())
    service.validate_recipe = lambda payload: {
        "valid": False,
        "recipe": payload,
        "issues": [{
            "severity": "error",
            "code": "stale_rulesHash",
            "field": "profiles.rulesHash",
        }],
    }

    with pytest.raises(ValueError, match="stale_rulesHash"):
        service.submit("RCP-STALE01", idempotency_key="stale-authority")


def test_research_recipe_v2_freezes_operational_diversity_and_on_demand_rendering():
    value = _recipe(schemaVersion=2)
    value["diversity"] = {
        "schemaVersion": 1,
        "cellDimensions": ["contract", "acting_seat", "source"],
        "topKPerCell": 2,
        "rejectedAuditModulus": 50,
    }
    value["renderPolicy"] = {"mode": "on_demand"}
    recipe = ResearchRecipeV2.build(value)
    assert recipe.code.startswith("RCP-")
    assert recipe.diversity["topKPerCell"] == 2
    assert recipe.render_policy == {"mode": "on_demand"}
    assert build_research_recipe(recipe.to_dict()) == recipe

    changed = copy.deepcopy(value)
    changed["diversity"]["topKPerCell"] = 3
    assert ResearchRecipeV2.build(changed).code != recipe.code

    invalid = copy.deepcopy(value)
    invalid["renderPolicy"] = {"mode": "automatic"}
    with pytest.raises(ValueError, match="on-demand only"):
        ResearchRecipeV2.build(invalid)

    invalid = copy.deepcopy(value)
    invalid["diversity"]["inventedDimension"] = True
    with pytest.raises(ValueError, match="unknown research diversity policy"):
        ResearchRecipeV2.build(invalid)


def test_research_recipe_v2_freezes_relationship_card_and_distribution_constraints():
    value = _recipe(schemaVersion=2)
    value["situation"].update({
        "buyer": None,
        "objectiveSeat": "west",
        "constructionBuyerRelation": "partner",
        "purchaseSuits": ["hearts", "clubs"],
        "requiredCards": {"west": ["SA", "H7"]},
        "forbiddenCards": {"west": ["C8"]},
        "finalSuitCountRanges": {
            "west": {
                "spades": {"minimum": 2, "maximum": 5},
                "hearts": {"minimum": 1, "maximum": 4},
            },
        },
    })
    value["diversity"] = {
        "schemaVersion": 1,
        "cellDimensions": ["contract", "acting_seat"],
        "topKPerCell": 2,
        "rejectedAuditModulus": 50,
    }
    value["renderPolicy"] = {"mode": "on_demand"}
    recipe = ResearchRecipeV2.build(value)
    assert recipe.situation["objectiveSeat"] == "west"
    assert recipe.situation["constructionBuyerRelation"] == "partner"
    assert recipe.situation["purchaseSuits"] == ["clubs", "hearts"]
    assert recipe.situation["finalSuitCountRanges"]["west"]["spades"] == {
        "minimum": 2, "maximum": 5,
    }

    service = object.__new__(ResearchFoundryServiceV1)
    spec = service._compile_deal_spec(recipe, seed=99)
    assert spec.objective_target_seat == Seat.WEST
    assert spec.assumed_buyer == Seat.EAST
    assert [item.name.lower() for item in spec.purchase_suits] == ["hearts", "clubs"]
    assert any(
        seat == Seat.WEST and suit.name.lower() == "spades" and (low, high) == (2, 5)
        for seat, suit, low, high in spec.final_suit_ranges
    )

    screened_value = copy.deepcopy(value)
    screened_value["resources"]["deepVerificationPolicy"] = "required_gates"
    screened_value["resources"]["tacticalViabilityPreScreen"] = {
        "schemaVersion": 1,
        "id": "exact_tactical_viability_screen_v1",
        "revision": "1",
        "enabled": True,
        "maximumDecisions": 8,
        "requiredTacticsOnly": True,
        "rejectWhenExactAbsent": True,
    }
    screened_recipe = ResearchRecipeV2.build(screened_value)
    screened_spec = service._compile_deal_spec(screened_recipe, seed=99)
    assert screened_spec.sha256 == spec.sha256

    ambiguous = copy.deepcopy(value)
    ambiguous["situation"]["buyer"] = "south"
    with pytest.raises(ValueError, match="explicit buyer"):
        ResearchRecipeV2.build(ambiguous)

    duplicate_card = copy.deepcopy(value)
    duplicate_card["situation"]["requiredCards"] = {
        "west": ["SA"], "east": ["SA"],
    }
    with pytest.raises(ValueError, match="multiple seats"):
        ResearchRecipeV2.build(duplicate_card)

    impossible_range = copy.deepcopy(value)
    impossible_range["situation"]["finalSuitCountRanges"] = {
        "west": {
            "spades": {"minimum": 5, "maximum": 8},
            "hearts": {"minimum": 4, "maximum": 8},
        },
    }
    with pytest.raises(ValueError, match="eight-card hand"):
        ResearchRecipeV2.build(impossible_range)

    invalid_card_container = copy.deepcopy(value)
    invalid_card_container["situation"]["requiredCards"] = {"west": "SA"}
    with pytest.raises(ValueError, match="card lists"):
        ResearchRecipeV2.build(invalid_card_container)


def test_research_deal_spec_freezes_exact_tactical_contract_and_trump():
    service = object.__new__(ResearchFoundryServiceV1)

    sun = service._compile_deal_spec(
        ResearchRecipeV1.build(_recipe()),
        seed=101,
    )
    assert sun.tactical_contract == Contract.SUN
    assert sun.tactical_trump is None

    hokum_value = _recipe()
    hokum_value["situation"]["contracts"] = ["hokum"]
    hokum_value["situation"]["trumpSuit"] = "diamonds"
    hokum = service._compile_deal_spec(
        ResearchRecipeV1.build(hokum_value),
        seed=101,
    )
    assert hokum.tactical_contract == Contract.HOKUM
    assert hokum.tactical_trump == Suit.DIAMONDS

    for contracts in ([], ["sun", "hokum"]):
        unbound_value = _recipe()
        unbound_value["situation"]["contracts"] = contracts
        unbound = ResearchRecipeV1.build(unbound_value)
        with pytest.raises(ValueError, match="exactly one requested contract"):
            service._compile_deal_spec(unbound, seed=101)


def test_research_v2_diversity_selection_is_real_deterministic_and_bounded():
    profile = QualityDiversityProfileV1.build({
        "schemaVersion": 1,
        "cellDimensions": ["contract", "acting_seat"],
        "topKPerCell": 2,
        "rejectedAuditModulus": 10,
    })

    def result(code: str, game: str, contract: str, seat: str, score: float):
        return {
            "resultCode": code,
            "resultHash": _hash(f"result:{code}"),
            "traceHash": _hash(f"trace:{code}"),
            "gameHash": game,
            "score": score,
            "confidence": 0.9,
            "disposition": "qualified",
            "qualification": {
                "qualified": True,
                "minimumSurroundingQuality": 80,
            },
            "libraryProjection": {
                "contract": contract,
                "actingSeats": [seat],
                "observerSeats": [seat],
                "generationLane": "ortools",
            },
        }

    values = [
        result("LAF-A", _hash("game-a"), "sun", "north", 91),
        result("LAF-B", _hash("game-b"), "hokum", "south", 82),
        result("LAF-C", _hash("game-c"), "sun", "north", 80),
        result("LAF-A2", _hash("game-a"), "sun", "north", 70),
    ]
    first = select_research_results(values, profile=profile, requested=2)
    second = select_research_results(
        reversed(values), profile=profile, requested=2
    )
    assert first.to_dict() == second.to_dict()
    assert first.selected_result_codes == ("LAF-B", "LAF-A")
    assert first.duplicate_count == 1
    assert {item["reason"] for item in first.rejected} == {
        "duplicate_fingerprint",
        "diversity_or_result_limit",
    }
    audit_all = select_research_results(
        values,
        profile=QualityDiversityProfileV1.build({
            "schemaVersion": 1,
            "cellDimensions": ["contract", "acting_seat"],
            "topKPerCell": 2,
            "rejectedAuditModulus": 1,
        }),
        requested=2,
    )
    assert set(audit_all.rejected_audit_sample) == {
        item["resultCode"] for item in audit_all.rejected
    }

def test_research_recipe_rejects_unknown_tactics_and_unfrozen_profiles():
    invalid = _recipe()
    invalid["tactics"][0]["kind"] = "magic_guess"
    with pytest.raises(ValueError, match="tactical kind"):
        ResearchRecipeV1.build(invalid)

    invalid = _recipe()
    invalid["profiles"]["rulesHash"] = "latest"
    with pytest.raises(ValueError, match="SHA-256"):
        ResearchRecipeV1.build(invalid)


def test_research_recipe_governed_bounds_and_precursor_fail_closed():
    invalid_trump = _recipe()
    invalid_trump["situation"]["contracts"] = []
    invalid_trump["situation"]["trumpSuit"] = "hearts"
    with pytest.raises(ValueError, match="requires only the Hokum contract"):
        ResearchRecipeV1.build(invalid_trump)

    invalid_weight = _recipe()
    invalid_weight["tactics"][0]["weight"] = 100.01
    with pytest.raises(ValueError, match="tactical score or weight"):
        ResearchRecipeV1.build(invalid_weight)

    unsupported_precursor = _recipe()
    unsupported_precursor["tactics"][0]["kind"] = "only_winning_move"
    unsupported_precursor["resources"].update({
        "generationLane": "ortools",
        "playConstructionMode": "cp_sat_feasibility_line",
        "tacticalConstruction": {
            "schemaVersion": 1,
            "id": "tactical_play_precursor_v1",
            "revision": 1,
            "enabled": True,
            "maximumPayoffTricks": 4,
            "minimumLaterRawPoints": 10,
            "requirePrecursorForGatedTactics": True,
        },
    })
    with pytest.raises(ValueError, match="gated small-card-big-gain tactic"):
        ResearchRecipeV1.build(unsupported_precursor)

    invalid_outcome = _recipe()
    invalid_outcome["resources"].update({
        "generationLane": "ortools",
        "playConstructionMode": "cp_sat_actor_safe_late_game_v1",
        "strategicPlay": {
            "schemaVersion": 1,
            "minimumPlayedCards": 24,
            "requestedSamples": 32,
            "exactAllocationLimit": 500,
            "maximumSamplingAttempts": 500,
            "maximumWorlds": 500,
            "maximumContinuationSteps": 20_000,
            "maximumPlies": 8,
            "maximumSelectionSeconds": 30,
            "requireExactForQualification": True,
            "continuationPolicy": "observer_safe_heuristic_v1",
            "policyProfileId": "maximin_margin",
            "policyRevision": 1,
            "outcomeBoundary": 10_001,
            "utility": "actor_team_awarded_point_margin",
            "tieBreak": "maximin_then_total_then_card_id",
        },
    })
    with pytest.raises(ValueError, match="outcome boundary"):
        ResearchRecipeV1.build(invalid_outcome)

    invalid_diversity = _recipe(schemaVersion=2)
    invalid_diversity["diversity"] = {
        "schemaVersion": 1,
        "cellDimensions": ["contract"],
        "topKPerCell": 1_001,
        "rejectedAuditModulus": 100,
    }
    with pytest.raises(ValueError, match="governed limits"):
        ResearchRecipeV2.build(invalid_diversity)

    invalid_diversity["diversity"].update({
        "topKPerCell": 3,
        "rejectedAuditModulus": 1_000_001,
    })
    with pytest.raises(ValueError, match="governed limits"):
        ResearchRecipeV2.build(invalid_diversity)


def test_research_api_models_match_governed_nested_bounds():
    from pydantic import ValidationError

    from baloot_platform.service.api import (
        ResearchDiversityRequest,
        ResearchSituationRequest,
        ResearchStrategicPlayRequest,
        ResearchTacticRequest,
        ResearchTacticalViabilityPreScreenRequest,
    )

    assert ResearchTacticalViabilityPreScreenRequest().enabled is False
    with pytest.raises(ValidationError):
        ResearchSituationRequest(matchScore=[0, 10_001])
    with pytest.raises(ValidationError):
        ResearchStrategicPlayRequest(outcomeBoundary=10_001)
    with pytest.raises(ValidationError):
        ResearchTacticRequest(kind="debate", weight=100.01)
    with pytest.raises(ValidationError):
        ResearchDiversityRequest(topKPerCell=1_001)
    with pytest.raises(ValidationError):
        ResearchDiversityRequest(rejectedAuditModulus=1_000_001)


def test_deep_tactical_verification_is_explicit_and_cannot_disable_hard_gates():
    preference = _recipe(
        tactics=[{
            "kind": "small_card_big_gain",
            "mode": "preference",
            "minimumScore": 70,
            "weight": 1,
            "perspectives": ["south"],
        }],
        resources={
            "seedStart": 100,
            "seedCount": 20,
            "seedsPerShard": 5,
            "candidatesPerSeed": 2,
            "solverTimeoutSeconds": 10,
            "maximumConflicts": 250_000,
            "analysisMode": "cascaded",
            "deepVerificationPolicy": "off",
        },
    )
    preference_recipe = ResearchRecipeV1.build(preference)
    assert not ResearchFoundryServiceV1._tactical_verification_required(
        preference_recipe
    )

    required = copy.deepcopy(preference)
    required["resources"]["deepVerificationPolicy"] = "required_gates"
    required_recipe = ResearchRecipeV1.build(required)
    assert ResearchFoundryServiceV1._tactical_verification_required(
        required_recipe
    )

    all_candidates = copy.deepcopy(preference)
    all_candidates["resources"]["deepVerificationPolicy"] = "all_candidates"
    all_candidates["tactics"] = []
    assert ResearchFoundryServiceV1._tactical_verification_required(
        ResearchRecipeV1.build(all_candidates)
    )

    service = object.__new__(ResearchFoundryServiceV1)
    service.maximum_tactical_decisions = 3
    service.validate_recipe = lambda payload: {
        "valid": True,
        "recipe": required_recipe.to_dict(),
        "issues": [],
    }
    service._build_recipe_and_context = lambda payload: (required_recipe, None)
    estimate = service.estimate(required_recipe.to_dict())
    assert estimate["candidateCount"] == 40
    assert estimate["tacticalVerificationDecisionUpperBound"] == 120

    invalid = _recipe()
    invalid["resources"]["deepVerificationPolicy"] = "off"
    with pytest.raises(ValueError, match="cannot disable deep verification"):
        ResearchRecipeV1.build(invalid)


def test_research_recipe_rejects_impossible_four_hundred_contracts():
    mismatch = _recipe()
    mismatch["situation"]["contracts"] = ["hokum"]
    mismatch["situation"]["projects"] = [{
        "seat": "south", "contract": "sun", "kind": "four_hundred",
        "value": 400, "highRank": "ace", "mode": "hard",
    }]
    with pytest.raises(ValueError, match="project contract must match"):
        ResearchRecipeV1.build(mismatch)

    wrong_contract = _recipe()
    wrong_contract["situation"]["contracts"] = []
    wrong_contract["situation"]["projects"] = [{
        "seat": "south", "contract": "hokum", "kind": "four_hundred",
        "value": 400, "highRank": "ace", "mode": "hard",
    }]
    with pytest.raises(ValueError, match="requires sun contract"):
        ResearchRecipeV1.build(wrong_contract)

    malformed = _recipe()
    malformed["situation"]["projects"] = [{
        "seat": "south", "contract": "sun", "kind": "four_hundred",
        "value": 100, "highRank": "ace", "mode": "hard",
    }]
    with pytest.raises(ValueError, match="four aces worth 400"):
        ResearchRecipeV1.build(malformed)


def test_research_store_keeps_immutable_recipes_ranked_results_and_quarantine(tmp_path, monkeypatch):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    recipe = ResearchRecipeV1.build(_recipe()).to_dict()
    store.put_recipe(recipe)
    assert store.get_recipe(recipe["code"]) == recipe
    assert store.recipe_status(recipe["code"]) == "published"

    changed = copy.deepcopy(recipe)
    changed["titleAr"] = "عنوان مختلف"
    with pytest.raises(ValueError, match="immutable"):
        store.put_recipe(changed)
    retired = store.retire_recipe(recipe["code"])
    assert retired["lifecycleStatus"] == "retired"
    assert store.recipe_status(recipe["code"]) == "retired"

    run = {
        "schemaVersion": 1,
        "runCode": "RUN-TEST00000001",
        "runHash": _hash("run"),
        "recipeCode": recipe["code"],
        "recipeHash": recipe["sha256"],
        "status": "running",
        "progress": 0.5,
        "events": [],
    }
    store.create_run(run)
    assert store.run_status(run["runCode"]) == "running"
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    service._lock = threading.RLock()
    service._cancelled = set()
    with monkeypatch.context() as patcher:
        patcher.setattr(store, "get_run", lambda _code: pytest.fail("cancel poll decoded run payload"))
        assert service._cancel_check(run["runCode"]) is False
    for code, score in (("LAF-LOW", 50), ("LAF-HIGH", 90)):
        store.put_result({
            "resultCode": code,
            "resultHash": _hash(code),
            "runCode": run["runCode"],
            "truthLabel": "robust_best_known",
            "confidence": 0.9,
            "score": score,
            "disposition": "qualified",
        })
    ranked = store.list_results(run_code=run["runCode"], limit=10)
    assert [item["resultCode"] for item in ranked["items"]] == ["LAF-HIGH", "LAF-LOW"]

    disagreement = {
        "disagreementId": "DSG-TEST000001",
        "sha256": _hash("disagreement"),
        "runCode": run["runCode"],
        "category": "constraint_mismatch",
        "status": "open",
    }
    store.put_disagreement(disagreement)
    assert store.list_disagreements(run_code=run["runCode"])[0] == disagreement
    stats = store.aggregate_stats(run_code=run["runCode"])
    assert stats["resultCount"] == 2
    assert stats["resultDisposition"] == {"qualified": 2}
    assert stats["truth"] == {"robust_best_known": 2}
    assert stats["averageScore"] == 70
    assert stats["semanticDefects"] == {}
    assert stats["historicalRecords"] == {
        "historical_record_requires_readjudication": 1
    }
    assert store.run_status("RUN-MISSING") is None
    store.update_run(run["runCode"], status="cancelled")
    assert store.run_status(run["runCode"]) == "cancelled"
    assert service._cancel_check(run["runCode"]) is True


def test_research_api_is_connected_to_the_governed_runtime():
    from baloot_platform.service import api

    with TestClient(api.app) as client:
        capabilities = client.get("/api/v2/domain-capabilities")
        assert capabilities.status_code == 200, capabilities.text
        payload = capabilities.json()
        assert payload["schemaVersion"] == 2
        assert payload["research"]["solverAvailable"] is True
        assert payload["research"]["playCompiler"]["id"] == "ortools_cp_sat_play_compiler_v1"
        assert payload["research"]["generationLanes"] == [
            "ortools", "organic_runtime"
        ]
        assert payload["research"]["defaultAgentRelease"]
        assert payload["research"]["analysisProfiles"]["cascaded"]
        recipe_configuration = payload["research"]["recipeConfiguration"]
        assert recipe_configuration["recipeSchemaVersion"] == 3
        assert recipe_configuration["supportedRecipeSchemaVersions"] == [1, 2, 3]
        assert recipe_configuration["legacyReadSchemaVersions"] == [1, 2]
        assert recipe_configuration["unsupportedBehaviorRequiresPlugin"] is True
        assert len(recipe_configuration["sha256"]) == 64
        assert recipe_configuration["fields"]["situation.requiredCards"]["control"] == "seat_card_picker"
        assert recipe_configuration["fields"]["resources.strategicPlay.maximumSelectionSeconds"]["maximum"] == 30
        assert recipe_configuration["fields"]["resources.strategicPlay.policyRevision"]["default"] == 1
        assert recipe_configuration["fields"]["resources.strategicPlay.outcomeBoundary"]["minimum"] == -10_000
        assert recipe_configuration["fields"]["resources.tacticalViabilityPreScreen.maximumDecisions"]["maximum"] == 16
        assert recipe_configuration["fields"]["resources.tacticalViabilityPreScreen.requiredTacticsOnly"]["valueType"] == "boolean"
        assert recipe_configuration["fields"]["resources.tacticalDealBalance.enabled"]["valueType"] == "boolean"
        assert recipe_configuration["fields"]["qualification.minimumSurroundingQuality"]["default"] == 70
        assert recipe_configuration["fields"]["resources.seedCount"]["default"] == 10
        assert recipe_configuration["fields"]["resources.seedsPerShard"]["default"] == 10
        assert recipe_configuration["fields"]["resources.constructionPreScreen.enabled"]["default"] is False
        assert recipe_configuration["fields"]["resources.constructionPreScreen.minimumConfidence"]["default"] == 0
        assert recipe_configuration["fields"]["resources.tacticalViabilityPreScreen.enabled"]["default"] is False
        relationship_codes = {
            item["code"] for item in recipe_configuration["relationships"]
        }
        assert {
            "actor_safe_auction_requires_actor_safe_play",
            "actor_safe_auction_forbids_escalation",
            "strategic_play_samples_fit_worlds",
            "strategic_play_exact_limit_fits_worlds",
            "strategic_play_attempts_cover_samples",
            "strategic_play_plies_cover_tail",
            "tactical_viability_required_only_requires_gate",
            "tactical_deal_balance_bounds_ordered",
        } <= relationship_codes
        assert {
            item["kind"] for item in recipe_configuration["projectOptions"]
        } == {
            "siri", "fifty", "hundred_sequence", "hundred_four_kind",
            "four_hundred",
        }
        assert len(recipe_configuration["projectDomainHash"]) == 64
        assert payload["research"]["comparisonProfile"]["minimumYieldMultiplier"] == 2.0
        assert len(payload["research"]["comparisonProfile"]["sha256"]) == 64

        stats = client.get("/api/v2/research-stats")
        assert stats.status_code == 200, stats.text
        assert stats.json()["scope"] == "all"
        assert "funnel" in stats.json()
        assert "cpuMeasuredRunCount" in stats.json()
        assert "qualifiedPerWallHour" in stats.json()
        qualified_per_cpu_hour = stats.json()["qualifiedPerCpuHour"]
        assert qualified_per_cpu_hour is None or qualified_per_cpu_hour >= 0
        if qualified_per_cpu_hour is not None:
            assert stats.json()["cpuMeasuredRunCount"] > 0
        assert "byGenerationLane" in stats.json()
        comparison_campaigns = client.get("/api/v2/comparison-campaigns")
        assert comparison_campaigns.status_code == 200, comparison_campaigns.text
        assert "items" in comparison_campaigns.json()
        missing_render = client.post(
            "/api/v2/research-results/LAF-0000000000:prepare-render",
            json={},
        )
        assert missing_render.status_code == 404, missing_render.text
        missing_perspective = client.get(
            "/api/v2/research-results/LAF-0000000000/perspectives",
            params={"observerSeat": "south"},
        )
        assert missing_perspective.status_code == 404, missing_perspective.text
        perspective_operation = api.app.openapi()["paths"][
            "/api/v2/research-results/{code}/perspectives"
        ]["get"]
        perspective_schema = perspective_operation["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert perspective_schema["$ref"].endswith(
            "/ResearchPerspectiveListResponse"
        )

        primitives = client.get("/api/v2/rule-primitives")
        assert primitives.status_code == 200, primitives.text
        assert all(
            {"runtimeCoverage", "solverCoverage", "documentationCoverage", "testCoverage"}
            <= set(item)
            for item in primitives.json()["items"]
        )

        selected_revision = api.CONTAINER.configuration.resolve()["bundle"][
            "revisionId"
        ]
        scoped_capabilities = client.get(
            "/api/v2/domain-capabilities",
            params={"bundleRevisionId": selected_revision},
        )
        assert scoped_capabilities.status_code == 200, scoped_capabilities.text
        assert scoped_capabilities.json()["research"]["configurationSelection"][
            "ready"
        ] is True

        recipe = _recipe(
            schemaVersion=3,
            output={"kind": "laffa_collection", "resultCount": 2},
        )
        recipe.pop("profiles")
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": selected_revision,
        }
        validated = client.post("/api/v2/research-recipes:validate", json=recipe)
        assert validated.status_code == 200, validated.text
        assert validated.json()["valid"] is True
        frozen_recipe = validated.json()["recipe"]
        authority = frozen_recipe["configuration"]
        identities = authority["identities"]
        assert frozen_recipe["profiles"] == {
            "domainHash": identities["domain"]["sha256"],
            "bundleHash": authority["semanticBundleHash"],
            "rulesHash": identities["rules"]["sha256"],
            "conventionHash": identities["convention"]["sha256"],
            "agentRelease": identities["agent"]["id"],
            "analysisProfileHash": identities["analyzer"]["sha256"],
            "scoreProfileHash": identities["score"]["sha256"],
        }
        estimate = client.post("/api/v2/research-recipes:estimate", json=recipe)
        assert estimate.status_code == 200, estimate.text
        assert estimate.json()["candidateCount"] == 40
        rejected_create = client.post(
            "/api/v2/research-recipes",
            json=recipe,
            headers={"Idempotency-Key": "research-api-create-v3"},
        )
        assert rejected_create.status_code == 422, rejected_create.text
        api.RESEARCH.store.put_recipe(frozen_recipe)
        historical_read = client.get(
            f"/api/v2/research-recipes/{frozen_recipe['code']}"
        )
        assert historical_read.status_code == 200, historical_read.text
        assert historical_read.json()["schemaVersion"] == 3
        assert historical_read.json()["profiles"] == frozen_recipe["profiles"]
        assert historical_read.json()["configuration"] == authority

        for legacy_version in (1, 2):
            legacy = _recipe(schemaVersion=legacy_version)
            if legacy_version == 2:
                legacy["diversity"] = {
                    "schemaVersion": 1,
                    "cellDimensions": ["contract", "acting_seat", "source"],
                    "topKPerCell": 3,
                    "rejectedAuditModulus": 100,
                }
                legacy["renderPolicy"] = {"mode": "on_demand"}
            legacy_validation = client.post(
                "/api/v2/research-recipes:validate", json=legacy
            )
            assert legacy_validation.status_code == 422
            assert legacy_validation.json()["detail"] == (
                "historical_record_requires_readjudication"
            )
            assert client.post(
                "/api/v2/research-recipes", json=legacy
            ).status_code == 422

        strategic_recipe = copy.deepcopy(recipe)
        strategic_recipe["resources"].update({
            "playConstructionMode": "cp_sat_actor_safe_late_game_v1",
            "strategicPlay": {
                "schemaVersion": 1,
                "minimumPlayedCards": 24,
                "requestedSamples": 32,
                "exactAllocationLimit": 500,
                "maximumSamplingAttempts": 500,
                "maximumWorlds": 500,
                "maximumContinuationSteps": 20_000,
                "maximumPlies": 8,
                "maximumSelectionSeconds": 30,
                "requireExactForQualification": True,
                "continuationPolicy": "observer_safe_heuristic_v1",
                "policyProfileId": "maximin_margin",
                "policyRevision": 1,
                "outcomeBoundary": 1,
                "utility": "actor_team_awarded_point_margin",
                "tieBreak": "maximin_then_total_then_card_id",
            },
        })
        strategic_validation = client.post(
            "/api/v2/research-recipes:validate", json=strategic_recipe
        )
        assert strategic_validation.status_code == 200, strategic_validation.text
        assert strategic_validation.json()["valid"] is True
        canonical_strategic = strategic_validation.json()["recipe"]["resources"][
            "strategicPlay"
        ]
        assert canonical_strategic["solverBackend"] == "batched_cp_sat"
        assert len(canonical_strategic["sha256"]) == 64

        organic_recipe = copy.deepcopy(recipe)
        organic_recipe["resources"]["generationLane"] = "organic_runtime"
        for solver_field in (
            "candidatesPerSeed",
            "solverTimeoutSeconds",
            "maximumConflicts",
        ):
            organic_recipe["resources"].pop(solver_field, None)
        organic_recipe["situation"]["requiredCards"] = {}
        organic_recipe["situation"]["forbiddenCards"] = {}
        organic_estimate = client.post(
            "/api/v2/research-recipes:estimate", json=organic_recipe
        )
        assert organic_estimate.status_code == 200, organic_estimate.text
        assert organic_estimate.json()["candidateCount"] == 20
        assert organic_estimate.json()["maximumSolverSeconds"] == 0

        project_recipe = copy.deepcopy(recipe)
        project_recipe["situation"]["buyer"] = "south"
        project_recipe["situation"]["contracts"] = ["sun"]
        project_recipe["situation"]["projects"] = [{
            "seat": "south", "contract": "sun", "kind": "four_hundred",
            "value": 400, "highRank": "ace", "mode": "hard",
        }]
        project_recipe["tactics"] = []
        project_validation = client.post(
            "/api/v2/research-recipes:validate", json=project_recipe
        )
        assert project_validation.status_code == 200, project_validation.text
        canonical_project_payload = project_validation.json()["recipe"]
        canonical_project_payload.pop("lifecycleStatus", None)
        canonical_project_recipe = build_research_recipe(
            canonical_project_payload
        )
        compiled = api.RESEARCH._compile_deal_spec(
            canonical_project_recipe, seed=1
        )
        assert compiled.project_pattern is not None
        assert compiled.project_pattern.value == 40
        assert api.RESEARCH._project_matches(
            canonical_project_recipe,
            SimpleNamespace(trusted_final_state={
                "contract": "sun",
                "projects": [{
                    "owner": "south",
                    "kind": "four_hundred",
                    "value": 40,
                    "highRank": 7,
                }],
            }),
        ) is True

        frozen_recipe.pop("lifecycleStatus", None)
        canonical_recipe = build_research_recipe(frozen_recipe)
        tactical_compiled = api.RESEARCH._compile_deal_spec(
            canonical_recipe, seed=1
        )
        assert tactical_compiled.tactical_objective_weights == (
            ("small_card_big_gain", 100),
        )
        assert tactical_compiled.objective_target_seat.name.lower() == "south"

        balanced_value = copy.deepcopy(frozen_recipe)
        balanced_value.pop("sha256", None)
        balanced_value["resources"]["tacticalDealBalance"] = {
            "schemaVersion": 1,
            "id": "balanced_tactical_deal_v1",
            "revision": 1,
            "enabled": True,
            "maximumTeamHighCardDifference": 2,
            "minimumHighCardsPerSeat": 2,
            "maximumHighCardsPerSeat": 4,
            "minimumTargetLowCards": 1,
        }
        balanced_recipe = build_research_recipe(balanced_value)
        balanced_compiled = api.RESEARCH._compile_deal_spec(
            balanced_recipe, seed=1
        )
        assert balanced_compiled.tactical_deal_balance == (2, 2, 4, 1)
        assert balanced_recipe.resources["auctionConstructionMode"] == (
            "actor_safe_agent_verified"
        )
        assert balanced_recipe.resources["playConstructionMode"] == (
            "actor_safe_agent_verified"
        )

        unbound_tactical = copy.deepcopy(recipe)
        unbound_tactical["situation"]["buyer"] = None
        unbound_tactical["tactics"][0]["perspectives"] = []
        unbound_validation = client.post(
            "/api/v2/research-recipes:validate", json=unbound_tactical
        )
        assert unbound_validation.status_code == 200, unbound_validation.text
        canonical_unbound_payload = unbound_validation.json()["recipe"]
        canonical_unbound_payload.pop("lifecycleStatus", None)
        canonical_unbound = build_research_recipe(canonical_unbound_payload)
        first_target = api.RESEARCH._compile_deal_spec(
            canonical_unbound, seed=1
        )
        second_target = api.RESEARCH._compile_deal_spec(
            canonical_unbound, seed=2
        )
        assert first_target.objective_target_seat.name.lower() == "west"
        assert first_target.assumed_buyer == first_target.objective_target_seat.partner
        assert second_target.objective_target_seat.name.lower() == "south"
        assert second_target.assumed_buyer == second_target.objective_target_seat.partner

        for unsupported_kind in ("contextualized_laffa", "sakka"):
            unsupported = copy.deepcopy(recipe)
            unsupported["output"]["kind"] = unsupported_kind
            response = client.post(
                "/api/v2/research-recipes:validate", json=unsupported
            )
            assert response.status_code == 422, response.text

        escalation_recipe = copy.deepcopy(recipe)
        escalation_recipe["situation"]["escalations"] = ["double"]
        escalation_response = client.post(
            "/api/v2/research-recipes:validate", json=escalation_recipe
        )
        assert escalation_response.status_code == 422, escalation_response.text
        assert "does not yet support required escalation" in (
            escalation_response.json()["detail"]
        )


def test_every_replay_valid_result_is_renderable_but_only_strong_results_publish():
    rejected = {
        "disposition": "rejected",
        "truthLabel": "provisional",
        "confidence": 0.1,
        "qualification": {
            "failures": ["surrounding_agent_regret"],
            "minimumSurroundingQuality": 0,
        },
    }
    assert ResearchRenderPreparationServiceV1._publication_eligibility(rejected) == (
        False,
        "rejected",
        ["surrounding_agent_regret"],
    )
    qualified = {
        "disposition": "qualified",
        "truthLabel": "robust_best_known",
        "confidence": 0.8,
        "qualification": {"failures": [], "minimumSurroundingQuality": 75},
    }
    assert ResearchRenderPreparationServiceV1._publication_eligibility(qualified) == (
        True,
        "qualified",
        [],
    )
    audits = [
        {"ordinal": 1, "decisionId": "bid", "decisionKind": "bid", "forced": False},
        {"ordinal": 2, "decisionId": "play", "decisionKind": "play", "forced": False,
         "confidence": 0.2, "tacticalEvidence": []},
    ]
    chosen = ResearchRenderPreparationServiceV1._select_audit(
        {"criticalDecisionIds": []}, audits, None
    )
    assert chosen["decisionId"] == "play"


def test_actor_safe_debate_and_bid_tactics_do_not_require_actual_deal_oracle(tmp_path):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        debate = research._observer_tactical_evidence(
            SimpleNamespace(forced=False, decision_kind="play"),
            SimpleNamespace(
                sha256="a" * 64,
                alternatives=(
                    {
                        "rank": 1,
                        "action": {"suit": "hearts", "rank": "seven"},
                        "expectedMargin": 10.0,
                        "bestShare": 0.50,
                    },
                    {
                        "rank": 2,
                        "action": {"suit": "hearts", "rank": "eight"},
                        "expectedMargin": 9.0,
                        "bestShare": 0.40,
                    },
                ),
            ),
        )
        assert [item["kind"] for item in debate] == ["debate"]
        assert debate[0]["truthMode"] == "actor_observer"
        assert debate[0]["evidenceBasis"] == "paired_observer_safe_alternatives"

        bidding = research._observer_tactical_evidence(
            SimpleNamespace(forced=False, decision_kind="bid"),
            SimpleNamespace(
                sha256="b" * 64,
                alternatives=(
                    {
                        "rank": 1,
                        "action": {"action": "sun"},
                        "expectedMargin": 12.0,
                        "contractSuccessProbability": 0.80,
                        "bestShare": 0.80,
                    },
                    {
                        "rank": 2,
                        "action": {"action": "pass"},
                        "expectedMargin": 4.0,
                        "contractSuccessProbability": 0.40,
                        "bestShare": 0.20,
                    },
                ),
            ),
        )
        assert [item["kind"] for item in bidding] == ["project_bidding"]
        assert bidding[0]["truthMode"] == "actor_observer"
        assert bidding[0]["mechanisms"] == ["purchase_or_contract_choice"]
    finally:
        container.shutdown()


def test_rule_primitive_capabilities_link_real_evidence_and_distinguish_native_cp_sat(root):
    from baloot_platform.service import api

    primitives = api.RESEARCH.capabilities()["rulePrimitives"]
    assert primitives
    assert any(item["cpSatNative"] for item in primitives)
    assert any(
        item["solverCoverage"] and not item["cpSatNative"]
        for item in primitives
    )
    for primitive in primitives:
        evidence = primitive["coverageEvidence"]
        assert evidence["runtime"]
        assert evidence["solver"]
        assert evidence["documentation"]
        assert evidence["tests"]
        for relative in (
            *evidence["runtime"], *evidence["solver"],
            *evidence["documentation"], *evidence["tests"],
        ):
            assert (root / relative).is_file(), (primitive["id"], relative)
        if primitive["cpSatNative"]:
            assert evidence["proofBoundary"] == "direct_cp_sat_constraint"
        else:
            assert evidence["proofBoundary"] == "deterministic_post_solve_semantic_gate"


def test_optimality_cut_identifies_only_material_non_forced_counterexamples():
    candidate = SimpleNamespace(candidate_hash=_hash("candidate"), ordinal=3)
    event = ResearchFoundryServiceV1._optimality_cut_event(
        candidate=candidate,
        trace_hash=_hash("trace"),
        maximum_regret=0.25,
        audit={"decisions": [
            {"decisionId": "forced", "forced": True, "expectedRegret": 99},
            {"decisionId": "within", "forced": False, "expectedRegret": 0.25},
            {"decisionId": "violation", "forced": False, "expectedRegret": 0.251},
        ]},
    )

    assert event["cutType"] == "complete_deck_assignment_v1"
    assert event["candidateOrdinal"] == 3
    assert event["violatingDecisionIds"] == ["violation"]


def test_infeasible_auction_retains_only_current_solver_evidence(tmp_path):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        recipe = _recipe(
            schemaVersion=3,
            titleAr="اختبار مزاد غير قابل للتحقيق",
            output={"kind": "laffa", "resultCount": 1},
            situation={
                "dealer": "east", "buyer": None, "contracts": [],
                "matchScore": [0, 0], "requiredCards": {},
                "forbiddenCards": {}, "projects": [], "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "seedStart": 1, "seedCount": 1, "seedsPerShard": 1,
                "candidatesPerSeed": 1, "solverTimeoutSeconds": 1,
                "maximumConflicts": 10, "analysisMode": "screening",
                "auctionConstructionMode": "cp_sat_feasibility_line",
                "playConstructionMode": "cp_sat_feasibility_line",
            },
        )
        recipe.pop("profiles")
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": container.configuration.resolve()["bundle"][
                "revisionId"
            ],
        }
        created = _store_validated_legacy_recipe(research, recipe)
        frozen_recipe = build_research_recipe(
            research.store.get_recipe(created["code"])
        )
        frozen_context = research.execution_context_factory.resolve_recipe(
            frozen_recipe
        )
        generator = _SequentialDealGeneratorStub("auction-infeasible")
        research.deal_generator = generator
        run_code = "RUN-AUCTIONNONE1"
        research.store.create_run({
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": _hash("auction-infeasible-run"),
            "recipeCode": created["code"],
            "recipeHash": created["sha256"],
            "configurationAuthority": copy.deepcopy(created["configuration"]),
            "researchContextHash": frozen_context.context_hash,
            **_ortools_run_identity(research),
                "status": "queued",
                "progress": 0,
                "events": [],
            })
        original_simulator = research.simulator
        simulation_calls: list[tuple[object, ...]] = []

        def fail_if_simulated(*args, **kwargs):
            simulation_calls.append(args)
            raise AssertionError("an infeasible auction must not reach runtime replay")

        research.simulator = SimpleNamespace(
            rules=original_simulator.rules,
            simulate_preordered_deck=fail_if_simulated,
        )
        research._compile_auction = lambda *args, **kwargs: None

        research._execute(run_code)

        candidate = generator.candidates[0]
        finished = research.get_run(run_code)
        assert finished is not None
        assert finished["status"] == "completed", finished.get("events")
        assert finished["resultCount"] == 0
        assert finished["quarantineCount"] == 0
        assert simulation_calls == []
        assert research.list_disagreements(run_code=run_code) == []
        shard = research.store.list_shards(run_code)[0]
        event = next(
            item for item in shard["result"]["events"]
            if item["type"] == "recipe_infeasible"
        )
        assert event["reason"] == "auction_infeasible"
        assert event["candidateHash"] == candidate.candidate_hash
        assert event["solverEvidence"]["candidateHash"] == candidate.candidate_hash
        assert event["solverEvidence"]["stage"] == "auction_construction"
        assert "runtimeEvidence" not in event
        assert "traceHash" not in event
    finally:
        container.shutdown()


def test_runtime_compiler_mismatch_is_quarantined_without_a_result(tmp_path):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        recipe = _recipe(
            schemaVersion=3,
            titleAr="اختبار الحجر عند اختلاف المترجم والمحرك",
            output={"kind": "laffa", "resultCount": 1},
            situation={
                "dealer": "east", "buyer": None, "contracts": [],
                "matchScore": [0, 0], "requiredCards": {},
                "forbiddenCards": {}, "projects": [], "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "seedStart": 1, "seedCount": 1, "seedsPerShard": 1,
                "candidatesPerSeed": 1, "solverTimeoutSeconds": 1,
                "maximumConflicts": 10, "analysisMode": "screening",
                "auctionConstructionMode": "cp_sat_feasibility_line",
                "playConstructionMode": "cp_sat_feasibility_line",
            },
        )
        recipe.pop("profiles")
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": container.configuration.resolve()["bundle"][
                "revisionId"
            ],
        }
        created = _store_validated_legacy_recipe(research, recipe)
        frozen_recipe = build_research_recipe(
            research.store.get_recipe(created["code"])
        )
        frozen_context = research.execution_context_factory.resolve_recipe(
            frozen_recipe
        )
        research.deal_generator = _SequentialDealGeneratorStub("runtime-mismatch")
        run_code = "RUN-QUARANTINE01"
        research.store.create_run({
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": _hash("quarantine-run"),
            "recipeCode": created["code"],
            "recipeHash": created["sha256"],
            "configurationAuthority": copy.deepcopy(created["configuration"]),
            "researchContextHash": frozen_context.context_hash,
            **_ortools_run_identity(research),
            "status": "queued",
            "progress": 0,
            "events": [],
        })
        original_simulator = research.simulator
        fixed_simulation = original_simulator.simulate(
            1,
            dealer=Seat.EAST,
            match_score=(0, 0),
        )
        fixed_simulator = SimpleNamespace(
            rules=original_simulator.rules,
            structural_domain=original_simulator.structural_domain,
            exact_search_backend=original_simulator.exact_search_backend,
            agent_release=original_simulator.agent_release,
            simulate_preordered_deck=lambda *args, **kwargs: fixed_simulation,
        )
        test_context = replace(frozen_context, simulator=fixed_simulator)
        research.simulator = fixed_simulator
        research._build_recipe_and_context = lambda _payload: (
            frozen_recipe,
            test_context,
        )
        # Keep this test focused on the semantic-quarantine boundary. Complete
        # play construction and parity have their own protocol tests.
        research._compile_complete_play = lambda *args, **kwargs: (
            SimpleNamespace(sha256=_hash("mismatch-project-plan")),
                SimpleNamespace(
                    solver_status="optimal",
                    model_hash=_hash("mismatch-play-model"),
                    sha256=_hash("mismatch-play-plan"),
                    tricks=(),
                    tactical_precursor=None,
                ),
            (),
        )
        research._compiled_play_mismatches = lambda *args, **kwargs: ()
        research._semantic_mismatches = lambda *args, **kwargs: [
            "hard_constraint_violation",
            "runtime_transition_defect",
        ]

        research._execute(run_code)

        finished = research.get_run(run_code)
        assert finished is not None
        assert finished["status"] == "completed", finished.get("events")
        assert finished["resultCount"] == 0
        assert finished["quarantineCount"] == 2
        assert research.list_results(run_code=run_code, limit=10)["total"] == 0
        disagreements = research.list_disagreements(run_code=run_code)
        assert {item["category"] for item in disagreements} == {
            "hard_constraint_violation",
            "runtime_transition_defect",
        }
    finally:
        container.shutdown()


def test_research_auction_is_compiled_and_replayed_without_buyer_assumption(tmp_path):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        capabilities = research.capabilities()
        active = capabilities["research"]
        recipe = ResearchRecipeV1.build(_recipe(
            profiles={
                "domainHash": capabilities["sha256"],
                "bundleHash": capabilities["bundleHash"],
                "rulesHash": capabilities["rulesHash"],
                "conventionHash": capabilities["conventionHash"],
                "agentRelease": active["defaultAgentRelease"],
                "analysisProfileHash": active["analysisProfiles"]["screening"],
                "scoreProfileHash": active["defaultScoreProfileHash"],
            },
            situation={
                "dealer": "east",
                "buyer": "north",
                "contracts": ["hokum"],
                "purchaseRound": 2,
                "trumpSuit": "spades",
                "matchScore": [0, 0],
                "requiredCards": {},
                "forbiddenCards": {},
                "projects": [],
                "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional",
                "minimumConfidence": 0,
                "minimumSurroundingQuality": 0,
                "requireObserverSafety": True,
                "requireAllDecisionAudit": True,
                "maximumRegret": 999,
            },
            resources={
                "seedStart": 1,
                "seedCount": 1,
                "seedsPerShard": 1,
                "candidatesPerSeed": 1,
                "solverTimeoutSeconds": 5,
                "maximumConflicts": 50_000,
                "analysisMode": "screening",
            },
        ))
        deck_order = tuple(range(32))
        auction_plan = research._compile_auction(recipe, deck_order)

        assert auction_plan is not None
        scripted_bids = tuple(item.to_dict() for item in auction_plan.actions)
        assert any(item["action"] == "hokum" for item in scripted_bids)
        simulation = research.simulator.simulate_preordered_deck(
            1,
            deck_order,
            deal_generator_provenance={
                "backend": "ortools_cp_sat_v1",
                "profileHash": _hash("auction-integration-profile"),
                "candidateHash": _hash("auction-integration-candidate"),
            },
            dealer=Seat.EAST,
            match_score=(0, 0),
            scripted_bids=scripted_bids,
        )
        final_state = simulation.final_trace.trusted_final_state
        assert final_state["buyer"] == "north"
        assert final_state["contract"] == "hokum"
        assert final_state["trumpSuit"] == "spades"
        assert research._semantic_mismatches(recipe, simulation) == []
    finally:
        container.shutdown()


def test_research_escalation_is_compiled_and_replayed_before_play(tmp_path):
    from baloot_platform.challenge.replay import CompleteTraceReplayerV1

    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        capabilities = research.capabilities()
        active = capabilities["research"]
        recipe = ResearchRecipeV1.build(_recipe(
            profiles={
                "domainHash": capabilities["sha256"],
                "bundleHash": capabilities["bundleHash"],
                "rulesHash": capabilities["rulesHash"],
                "conventionHash": capabilities["conventionHash"],
                "agentRelease": active["defaultAgentRelease"],
                "analysisProfileHash": active["analysisProfiles"]["screening"],
                "scoreProfileHash": active["defaultScoreProfileHash"],
            },
            situation={
                "dealer": "east", "buyer": "north", "contracts": ["hokum"],
                "purchaseRound": 2, "trumpSuit": "spades", "matchScore": [0, 0],
                "requiredCards": {}, "forbiddenCards": {}, "projects": [],
                "escalations": ["gahwa"],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "seedStart": 1, "seedCount": 1, "seedsPerShard": 1,
                "candidatesPerSeed": 1, "solverTimeoutSeconds": 5,
                "maximumConflicts": 50_000, "analysisMode": "screening",
            },
        ))
        deck_order = tuple(range(32))
        auction = research._compile_auction(recipe, deck_order)
        assert auction is not None
        escalation = research._compile_escalation(recipe, auction)
        assert escalation is not None
        escalation_plan = research._compile_escalation_plan(recipe, auction)
        assert escalation_plan is not None
        project_plan, play_plan, scripted_plays = research._compile_complete_play(
            recipe,
            deck_order,
            auction,
            escalation_plan,
        )
        simulation = research.simulator.simulate_preordered_deck(
            1,
            deck_order,
            deal_generator_provenance={
                "backend": "ortools_cp_sat_v1",
                "profileHash": _hash("escalation-integration-profile"),
                "candidateHash": _hash("escalation-integration-candidate"),
            },
            dealer=Seat.EAST,
            match_score=(0, 0),
            scripted_bids=tuple(item.to_dict() for item in auction.actions),
            scripted_escalations=escalation,
            scripted_plays=scripted_plays,
            scripted_play_provenance={
                "schemaVersion": 1,
                "pluginId": research.play_compiler.descriptor.id,
                "modelHash": play_plan.model_hash,
                "planHash": play_plan.sha256,
                "projectPlanHash": project_plan.sha256,
                "solverStatus": play_plan.solver_status,
            },
        )
        trace = simulation.final_trace
        assert trace.trusted_final_state["escalation"] == "gahwa"
        assert research._compiled_play_mismatches(play_plan, simulation) == ()
        assert trace.trusted_final_state["generation"]["scriptedPlayCompiler"][
            "planHash"
        ] == play_plan.sha256
        assert all(
            item["trace"]["method"] == "ortools_cp_sat_complete_play_v1"
            for item in trace.trusted_final_state["decisionPolicyTraces"]
            if item["ordinal"] > len(auction.actions)
        )
        replay = CompleteTraceReplayerV1(research.simulator.rules).replay(trace)
        assert replay.trace_hash == trace.sha256
        assert research._semantic_mismatches(recipe, simulation) == []
        analysis = research.analyzers["screening"].analyze_trace(trace)
        audit = research._decision_audit(trace, analysis)
        validate_governed_decision_audit_v2(
            audit,
            trace=trace,
            rules=research.simulator.rules,
        )
        escalation_row = next(
            item
            for item in audit["decisions"]
            if item["decisionKind"] == "escalation"
        )
        assert len(escalation_row["legalActions"]) > 1

        def rehash_audit(value: dict) -> None:
            value["sha256"] = canonical_sha256(
                {key: item for key, item in value.items() if key != "sha256"}
            )

        legal_set_substitution = copy.deepcopy(audit)
        substituted_row = next(
            item
            for item in legal_set_substitution["decisions"]
            if item["decisionKind"] == "escalation"
        )
        substituted_row["legalActions"] = substituted_row["legalActions"][:-1]
        rehash_audit(legal_set_substitution)
        with pytest.raises(ValueError, match="does not match its trace event"):
            validate_governed_decision_audit_v2(
                legal_set_substitution,
                trace=trace,
                rules=research.simulator.rules,
            )

        phase_substitution = copy.deepcopy(audit)
        next(
            item
            for item in phase_substitution["decisions"]
            if item["decisionKind"] == "escalation"
        )["phase"] = "projects"
        rehash_audit(phase_substitution)
        with pytest.raises(ValueError, match="does not match its trace event"):
            validate_governed_decision_audit_v2(
                phase_substitution,
                trace=trace,
                rules=research.simulator.rules,
            )

        team_substitution = copy.deepcopy(audit)
        next(
            item
            for item in team_substitution["decisions"]
            if item["decisionKind"] == "escalation"
        )["actingTeam"] = "opponents"
        rehash_audit(team_substitution)
        with pytest.raises(ValueError, match="does not match its trace event"):
            validate_governed_decision_audit_v2(
                team_substitution,
                trace=trace,
                rules=research.simulator.rules,
            )
    finally:
        container.shutdown()


def test_actor_safe_agent_line_is_verified_by_complete_cp_sat_play(tmp_path):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        capabilities = research.capabilities()
        active = capabilities["research"]
        assert active["recommendedPlayConstructionMode"] == (
            "actor_safe_agent_verified"
        )
        assert active["playConstructionModes"] == [
            "actor_safe_agent_verified",
            "cp_sat_actor_safe_late_game_v1",
            "cp_sat_feasibility_line",
        ]
        recipe = ResearchRecipeV1.build(_recipe(
            profiles={
                "domainHash": capabilities["sha256"],
                "bundleHash": capabilities["bundleHash"],
                "rulesHash": capabilities["rulesHash"],
                "conventionHash": capabilities["conventionHash"],
                "agentRelease": active["defaultAgentRelease"],
                "analysisProfileHash": active["analysisProfiles"]["screening"],
                "scoreProfileHash": active["defaultScoreProfileHash"],
            },
            situation={
                "dealer": "east", "buyer": "north", "contracts": ["hokum"],
                "purchaseRound": 2, "trumpSuit": "spades", "matchScore": [0, 0],
                "requiredCards": {}, "forbiddenCards": {}, "projects": [],
                "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "seedStart": 1, "seedCount": 1, "seedsPerShard": 1,
                "candidatesPerSeed": 1, "solverTimeoutSeconds": 5,
                "maximumConflicts": 50_000, "analysisMode": "screening",
                "playConstructionMode": "actor_safe_agent_verified",
            },
        ))
        deck_order = tuple(range(32))
        auction = research._compile_auction(recipe, deck_order)
        assert auction is not None
        escalation_plan = research._compile_escalation_plan(recipe, auction)
        assert escalation_plan is not None
        simulation = research.simulator.simulate_preordered_deck(
            1,
            deck_order,
            deal_generator_provenance={
                "backend": "ortools_cp_sat_v1",
                "profileHash": _hash("agent-verified-profile"),
                "candidateHash": _hash("agent-verified-candidate"),
                "playConstructionMode": "actor_safe_agent_verified",
                "playPolicyRelease": research.simulator.agent_release,
            },
            dealer=Seat.EAST,
            match_score=(0, 0),
            scripted_bids=tuple(item.to_dict() for item in auction.actions),
            scripted_escalations=tuple(
                item.to_dict() for item in escalation_plan.actions
            ),
        )
        required_cards, required_ekka_leads = research._played_line(simulation)
        _project_plan, play_plan, verified_actions = (
            research._compile_complete_play(
                recipe,
                deck_order,
                auction,
                escalation_plan,
                required_cards=required_cards,
                required_ekka_leads=required_ekka_leads,
            )
        )
        state = simulation.final_trace.trusted_final_state
        assert "scriptedPlayCompiler" not in state["generation"]
        assert len(verified_actions) == 32
        assert play_plan.solver_status == "optimal"
        assert research._compiled_play_mismatches(play_plan, simulation) == ()
        assert all(
            item["trace"].get("method") != "ortools_cp_sat_complete_play_v1"
            for item in state["decisionPolicyTraces"]
            if item["ordinal"] > len(auction.actions)
        )
    finally:
        container.shutdown()


def test_declared_non_finalist_ortools_line_is_replayable_but_not_admitted(
    tmp_path,
):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        recipe = _recipe(
            schemaVersion=3,
            titleAr="اختبار لعب الوكلاء والتحقق من المسار",
            output={"kind": "laffa", "resultCount": 1},
            situation={
                "dealer": "east", "buyer": "north", "contracts": ["hokum"],
                "purchaseRound": 2, "trumpSuit": "spades", "matchScore": [0, 0],
                "requiredCards": {}, "forbiddenCards": {}, "projects": [],
                "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "generationLane": "ortools", "seedStart": 1, "seedCount": 1,
                "seedsPerShard": 1, "candidatesPerSeed": 1,
                "solverTimeoutSeconds": 15, "maximumConflicts": 250_000,
                "analysisMode": "screening",
                # This fixture isolates actor-safe play verification; natural
                # actor-safe auction target misses are covered separately.
                "auctionConstructionMode": "cp_sat_feasibility_line",
                "playConstructionMode": "actor_safe_agent_verified",
            },
        )
        recipe.pop("profiles")
        selected_revision = container.configuration.resolve()["bundle"][
            "revisionId"
        ]
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": selected_revision,
        }
        created = _store_validated_legacy_recipe(research, recipe)
        frozen_recipe = build_research_recipe(
            research.store.get_recipe(created["code"])
        )
        frozen_context = research.execution_context_factory.resolve_recipe(
            frozen_recipe
        )
        generator = _SequentialDealGeneratorStub("agent-verified")
        research.deal_generator = generator
        run_code = "RUN-AGENTVERIFY1"
        research.store.create_run({
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": _hash("agent-verified-run"),
            "recipeCode": created["code"],
            "recipeHash": created["sha256"],
            "configurationAuthority": copy.deepcopy(created["configuration"]),
            "researchContextHash": frozen_context.context_hash,
            **_ortools_run_identity(research),
            "status": "queued",
            "progress": 0,
            "events": [],
        })
        research._execute(run_code)

        finished = research.get_run(run_code)
        assert finished is not None
        assert finished["status"] == "completed", research.store.list_shards(
            run_code
        )
        assert finished["semanticReplayPassed"] == 1
        assert finished["playCompilerOptimal"] == 1
        assert finished["quarantineCount"] == 0
        assert finished["resultCount"] == 0
        assert finished["qualifiedCount"] == 0
        assert finished["rejectedCount"] == 0
        assert research.list_results(run_code=run_code, limit=10)["items"] == []
        audit = research.list_audit_traces(run_code=run_code, limit=10)["items"]
        assert len(audit) == 1
        assert audit[0]["disposition"] == "rejected"
        assert {
            "stage_finalist_primary_policy_admission",
            "finalist_primary_policy_required",
        }.issubset(audit[0]["failureCategories"])
        assert audit[0]["disagreementReferences"] == []
        shard = research.store.list_shards(run_code)[0]
        assert shard["result"]["solverIncomplete"] == 0
        assert shard["result"]["resultCount"] == 0
        assert shard["result"]["qualifiedCount"] == 0
        assert shard["result"]["rejectedCount"] == 0
        assert shard["result"]["quarantineCount"] == 0
        receipts = research.store.list_sequential_search_receipts(
            run_code=run_code,
            limit=10,
        )["items"]
        assert len(receipts) == 1
        receipt = receipts[0]["receiptBundle"]
        assert receipt["stopReason"] == "incomplete"
        assert receipt["failureCodes"] == [
            "finalist_primary_policy_unavailable"
        ]
        assert receipt["emittedCandidateCount"] == 1
        assert receipt["evaluatedCandidateCount"] == 1
        assert receipt["rejectedCandidateCount"] == 0
        assert receipt["rejectionFeedbackRows"] == []
        assert receipt["terminalContinuationReceipt"] is None
        assert receipt["evaluationReceipts"][0]["evaluation"][
            "outcome"
        ] == "incomplete"
        assert generator.generate_calls == [()]
        assert generator.probe_calls == []
        assert generator.legacy_generate_calls == 0
        events = shard["result"]["events"]
        assert any(
            item["type"] == "finalist_primary_policy_required"
            and item["auctionConstructionMode"] == "cp_sat_feasibility_line"
            and item["playConstructionMode"] == "actor_safe_agent_verified"
            for item in events
        )
    finally:
        container.shutdown()


def test_finalist_decision_score_mismatch_is_fatally_quarantined(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        recipe = _recipe(
            schemaVersion=3,
            titleAr="اختبار حجر عدم تطابق نتيجة القرار",
            output={"kind": "laffa", "resultCount": 1},
            situation={
                "dealer": "east", "buyer": None, "contracts": [],
                "matchScore": [0, 0], "requiredCards": {},
                "forbiddenCards": {}, "projects": [], "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "generationLane": "ortools", "seedStart": 1, "seedCount": 1,
                "seedsPerShard": 1, "candidatesPerSeed": 1,
                "solverTimeoutSeconds": 15, "maximumConflicts": 250_000,
                "analysisMode": "screening",
                "auctionConstructionMode": "actor_safe_agent_verified",
                "playConstructionMode": "actor_safe_agent_verified",
            },
        )
        recipe.pop("profiles")
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": container.configuration.resolve()["bundle"][
                "revisionId"
            ],
        }
        created = _store_validated_legacy_recipe(research, recipe)
        frozen_recipe = build_research_recipe(
            research.store.get_recipe(created["code"])
        )
        frozen_context = research.execution_context_factory.resolve_recipe(
            frozen_recipe
        )
        research.deal_generator = _SequentialDealGeneratorStub("score-parity")
        run_code = "RUN-SCOREPARITY1"
        research.store.create_run({
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": _hash("score-parity-run"),
            "recipeCode": created["code"],
            "recipeHash": created["sha256"],
            "configurationAuthority": copy.deepcopy(created["configuration"]),
            "researchContextHash": frozen_context.context_hash,
            **_ortools_run_identity(research),
            "status": "queued",
            "progress": 0,
            "events": [],
        })
        analysis_hash = _hash("direct-deep-analysis")
        direct_analysis = SimpleNamespace(
            sha256=analysis_hash,
            to_dict=lambda: {
                "schemaVersion": 3,
                "sha256": analysis_hash,
                "source": "direct_deep_test_authority",
            },
        )
        fake_analyzer = SimpleNamespace(
            analyze_trace=lambda *args, **kwargs: direct_analysis
        )
        test_context = replace(
            frozen_context,
            analyzers={"screening": fake_analyzer, "deep": fake_analyzer},
        )
        monkeypatch.setattr(
            research,
            "_build_recipe_and_context",
            lambda _payload: (frozen_recipe, test_context),
        )
        generation_hash = _hash("generation-decision-score")
        audit_hash = _hash("audit-decision-score")
        parity_core = {
            "schemaVersion": 1,
            "authority": "finalist_primary_decision_score_parity_v1",
            "traceHash": _hash("same-frozen-trace"),
            "deepAnalysisHash": analysis_hash,
            "matched": False,
            "mismatches": [{
                "decisionId": "D-TEST",
                "ordinal": 1,
                "decisionKind": "bid",
                "reason": "primary_decision_score_evidence_bytes_mismatch",
            }],
            "decisions": [{
                "decisionId": "D-TEST",
                "ordinal": 1,
                "generationEvidenceHash": generation_hash,
                "auditEvidenceHash": audit_hash,
                "matched": False,
            }],
            "crossLaneComparison": False,
            "structuralObjectiveCompared": False,
            "officialBalootScoreCompared": False,
        }
        parity = {**parity_core, "sha256": _hash("score-parity-evidence")}
        monkeypatch.setattr(
            research,
            "_decision_score_parity_evidence",
            lambda *args, **kwargs: parity,
        )

        research._execute(run_code)

        finished = research.get_run(run_code)
        assert finished is not None
        assert finished["status"] == "completed"
        assert finished["quarantineCount"] == 1
        assert research.list_results(run_code=run_code, limit=10)["items"] == []
        audit = research.list_audit_traces(run_code=run_code, limit=10)["items"]
        assert len(audit) == 1
        assert audit[0]["disposition"] == "quarantined"
        assert {
            "stage_decision_score_parity",
            "compiler_parity_defect",
            (
                "decision_score_parity:"
                "primary_decision_score_evidence_bytes_mismatch"
            ),
        }.issubset(audit[0]["failureCategories"])
        disagreements = research.store.list_disagreements(run_code=run_code)
        assert len(disagreements) == 1
        assert disagreements[0]["category"] == "compiler_parity_defect"
        assert disagreements[0]["solverEvidence"]["stage"] == (
            "decision_score_parity"
        )
        assert disagreements[0]["runtimeEvidence"]["deepAnalysisHash"] == (
            analysis_hash
        )
        shard = research.store.list_shards(run_code)[0]
        event = next(
            item for item in shard["result"]["events"]
            if item["type"] == "compiler_parity_defect"
        )
        assert event["stage"] == "decision_score_parity"
        assert event["mismatchCount"] == 1
        retained_parity = json.loads(
            research.artifact_store.get(event["parityEvidence"]["uri"])
        )
        assert retained_parity["decisions"][0] == {
            "decisionId": "D-TEST",
            "ordinal": 1,
            "generationEvidenceHash": generation_hash,
            "auditEvidenceHash": audit_hash,
            "matched": False,
        }
        assert retained_parity["crossLaneComparison"] is False
    finally:
        container.shutdown()


@pytest.mark.real_clock
def test_organic_runtime_lane_uses_the_same_audit_pipeline_without_solver_quarantine(tmp_path):
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        recipe = _recipe(
            schemaVersion=3,
            titleAr="خط أساس طبيعي للمقارنة",
            output={"kind": "laffa", "resultCount": 1},
            situation={
                "dealer": "east", "buyer": None, "contracts": [],
                "matchScore": [0, 0], "requiredCards": {},
                "forbiddenCards": {}, "projects": [], "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "generationLane": "organic_runtime",
                "seedStart": 1, "seedCount": 1, "seedsPerShard": 1,
                "analysisMode": "screening",
            },
            diversity={
                "schemaVersion": 1,
                "cellDimensions": ["contract", "acting_seat"],
                "topKPerCell": 1,
                "rejectedAuditModulus": 10,
            },
            renderPolicy={"mode": "on_demand"},
        )
        recipe.pop("profiles")
        selected_revision = container.configuration.resolve()["bundle"][
            "revisionId"
        ]
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": selected_revision,
        }
        created = _store_validated_legacy_recipe(research, recipe)
        frozen_recipe = build_research_recipe(
            research.store.get_recipe(created["code"])
        )
        frozen_context = research.execution_context_factory.resolve_recipe(
            frozen_recipe
        )
        run_code = "RUN-ORGANIC00001"
        research.store.create_run({
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": _hash("organic-run"),
            "recipeCode": created["code"],
            "recipeHash": created["sha256"],
            "configurationAuthority": copy.deepcopy(created["configuration"]),
            "researchContextHash": frozen_context.context_hash,
            "generationLane": "organic_runtime",
            "status": "queued",
            "progress": 0,
            "events": [],
            "createdAt": "2026-08-10T00:00:00+00:00",
        })

        research._execute(run_code)

        finished = research.get_run(run_code)
        assert finished is not None
        assert finished["status"] == "completed"
        assert finished["generationLane"] == "organic_runtime"
        assert finished["quarantineCount"] == 0
        assert finished["solverOptimal"] == 0
        assert finished["runtimeParityPassed"] == 0
        assert finished["semanticReplayPassed"] == 1
        results = research.list_results(run_code=run_code, limit=10)["items"]
        assert len(results) == 1
        assert results[0]["provenance"]["generationLane"] == "organic_runtime"
        assert results[0]["provenance"]["generatorPlugin"] == "natural_laffa_generator_v1"
        frozen = created["configuration"]
        assert results[0]["provenance"]["configurationAuthorityHash"] == frozen[
            "sha256"
        ]
        assert results[0]["provenance"]["effectiveRuntimeConfigHash"] == frozen[
            "effectiveRuntimeConfigHash"
        ]
        assert results[0]["provenance"]["snapshotArtifact"] == frozen[
            "snapshotArtifact"
        ]
        assert results[0]["provenance"]["executionIdentities"] == frozen[
            "identities"
        ]
        assert results[0]["provenance"]["domainHash"] == frozen["identities"][
            "domain"
        ]["sha256"]
        assert results[0]["provenance"]["compiledDomainHash"] == frozen[
            "identities"
        ]["domain"]["sha256"]
        assert results[0]["provenance"]["legacyDomainV2Hash"] != frozen[
            "identities"
        ]["domain"]["sha256"]
        assert results[0]["scoreEvidence"]["configurationAuthorityHash"] == frozen[
            "sha256"
        ]
        assert results[0]["scoreEvidence"]["researchContextHash"] == (
            frozen_context.context_hash
        )
        audit_reference = results[0]["artifacts"]["decisionAudit"]
        audit = json.loads(
            research.artifact_store.get(audit_reference["uri"]).decode("utf-8")
        )
        assert audit["configurationProvenance"][
            "configurationAuthorityHash"
        ] == frozen["sha256"]
        assert all(
            decision["provenance"]["effectiveRuntimeConfigHash"]
            == frozen["effectiveRuntimeConfigHash"]
            for decision in audit["decisions"]
        )
        tactical_reference = results[0]["artifacts"]["tacticalEvidence"]
        tactical_payload = json.loads(
            research.artifact_store.get(tactical_reference["uri"]).decode("utf-8")
        )
        assert tactical_payload["configurationProvenance"][
            "configurationAuthorityHash"
        ] == frozen["sha256"]
        projection = results[0]["libraryProjection"]
        assert projection["generationLane"] == "organic_runtime"
        assert projection["contract"] in {"sun", "hokum", "ashkal"}
        assert projection["actingSeats"]
        assert projection["observerSeats"]
        filtered = research.list_results(
            run_code=run_code,
            contract=projection["contract"],
            generation_lane="organic_runtime",
            observer_seat=projection["observerSeats"][0],
            limit=10,
        )
        assert [item["resultCode"] for item in filtered["items"]] == [
            results[0]["resultCode"]
        ]
        # Complete compiler coverage and semantic replay are not content
        # qualification. The natural line remains a normal, non-quarantined
        # result, but its explicit audit counterexamples reject publication.
        assert results[0]["disposition"] == "rejected"
        assert {
            "decision_material_counterexample",
            "decision_recommendation_mismatch",
        } <= set(results[0]["qualification"]["failures"])
        decision_selection = results[0]["qualification"]["decisionSelection"]
        score_selection = results[0]["scoreEvidence"][
            "decisionSelectionQualification"
        ]
        assert decision_selection["profileHash"] == frozen["identities"][
            "decisionScore"
        ]["sha256"]
        assert decision_selection["evidenceHash"] == score_selection["sha256"]
        assert decision_selection["passed"] == score_selection["passed"]
        if not decision_selection["passed"]:
            assert results[0]["disposition"] == "rejected"
            assert any(
                item.startswith("decision_score_profile_gate:")
                for item in results[0]["qualification"]["failures"]
            )
        behavioral_or_invariant_complete = (
            frozen_context.compiled_domain.compiled.coverage
            .behavioral_or_invariant_complete
        )
        assert (
            "domain_behavioral_or_invariant_coverage_unproven"
            in results[0]["qualification"]["failures"]
        ) is not behavioral_or_invariant_complete
        assert results[0]["provenance"]["compilerExecutableBindingComplete"] is True
        assert results[0]["provenance"][
            "compilerBehavioralOrInvariantComplete"
        ] is behavioral_or_invariant_complete
        stats = research.statistics(run_code=run_code)
        organic = stats["byGenerationLane"]["organic_runtime"]
        assert organic["runCount"] == 1
        assert organic["attemptedSeeds"] == 1
        assert organic["generatedCandidates"] == 1
        assert organic["semanticReplayPassed"] == 1
        assert organic["quarantineCount"] == 0
        assert organic["tacticalVerificationCandidates"] == 0
        assert organic["tacticalVerificationDecisions"] == 0
        assert organic["tacticalVerificationSeconds"] == 0
        assert organic["constructionSeconds"] >= 0
        assert organic["simulationSeconds"] > 0
        assert organic["semanticParitySeconds"] >= 0
        assert organic["analysisSeconds"] > 0
        assert organic["averageSecondsPerAttempt"] > 0
        assert stats["stageSeconds"]["simulationSeconds"] > 0
        assert stats["stageSeconds"]["analysisSeconds"] > 0
    finally:
        container.shutdown()


def test_comparison_campaign_freezes_one_recipe_and_schedules_both_lanes(tmp_path):
    class DeferredExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, *args):
            self.calls.append((function, args))

    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        deferred = DeferredExecutor()
        research.executor = deferred
        executed: list[tuple[str, str, str, str]] = []

        def complete_scheduled_run(run_code: str) -> None:
            run = research.store.get_run(run_code)
            assert run is not None
            executed.append((
                run_code,
                str(run["generationLane"]),
                str(run["recipeHash"]),
                str(run["researchContextHash"]),
            ))
            research.store.update_run(
                run_code,
                status="completed",
                progress=1.0,
                attemptedSeeds=1,
                structuralCandidates=1,
                qualifiedCount=0,
                selectedCount=0,
                rejectedCount=1,
                quarantineCount=0,
                elapsedSeconds=0.1,
                cpuSeconds=0.1,
                events=[*run.get("events", []), {"type": "completed"}],
            )

        research._execute = complete_scheduled_run
        capabilities = research.capabilities()
        active = capabilities["research"]
        recipe = _recipe(
            schemaVersion=3,
            titleAr="مقارنة متطابقة بين طريقتي التوليد",
            output={"kind": "laffa", "resultCount": 1},
            profiles={
                "domainHash": capabilities["sha256"],
                "bundleHash": capabilities["bundleHash"],
                "rulesHash": capabilities["rulesHash"],
                "conventionHash": capabilities["conventionHash"],
                "agentRelease": active["defaultAgentRelease"],
                "analysisProfileHash": active["analysisProfiles"]["screening"],
                "scoreProfileHash": active["defaultScoreProfileHash"],
            },
            situation={
                "dealer": "east", "buyer": None, "contracts": [],
                "matchScore": [0, 0], "requiredCards": {},
                "forbiddenCards": {}, "projects": [], "escalations": [],
            },
            tactics=[],
            qualification={
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            resources={
                "generationLane": "ortools", "seedStart": 7, "seedCount": 1,
                "seedsPerShard": 1, "candidatesPerSeed": 1,
                "solverTimeoutSeconds": 2, "maximumConflicts": 50_000,
                "analysisMode": "screening",
                "auctionConstructionMode": "actor_safe_agent_verified",
                "playConstructionMode": "actor_safe_agent_verified",
            },
            diversity={
                "schemaVersion": 1,
                "cellDimensions": ["contract", "source"],
                "topKPerCell": 2,
                "rejectedAuditModulus": 100,
            },
            renderPolicy={"mode": "on_demand"},
        )
        recipe.pop("profiles")
        recipe["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": container.configuration.resolve()["bundle"][
                "revisionId"
            ],
        }
        base = _store_validated_legacy_recipe(research, recipe)
        campaign = research.create_comparison_campaign(
            base["code"], idempotency_key="comparison-test"
        )
        assert campaign["status"] == "queued"
        assert set(campaign["runCodes"]) == {"ortools", "organic_runtime"}
        assert campaign["matchedInputs"]["seedStart"] == 7
        assert campaign["matchedInputs"]["seedCount"] == 1
        assert campaign["matchedInputs"]["recipeSchemaVersion"] == 3
        assert campaign["matchedInputs"]["renderPolicy"] == {
            "mode": "on_demand"
        }
        assert campaign["matchedInputs"]["generationLaneDelta"] == {
            "ortools": {
                "generationLane": "ortools",
                "removedResourceKeys": [],
            },
                "organic_runtime": {
                    "generationLane": "organic_runtime",
                    "removedResourceKeys": [
                        "candidatesPerSeed",
                        "solverTimeoutSeconds",
                        "maximumConflicts",
                        "auctionConstructionMode",
                        "playConstructionMode",
                    ],
            },
        }
        assert len(deferred.calls) == 2
        organic_recipe = research.get_recipe(
            campaign["recipeCodes"]["organic_runtime"]
        )
        assert organic_recipe is not None
        assert organic_recipe["schemaVersion"] == 3
        assert "auctionConstructionMode" not in organic_recipe["resources"]
        assert "playConstructionMode" not in organic_recipe["resources"]
        for function, args in deferred.calls:
            function(*args)
        assert {item[1] for item in executed} == {"ortools", "organic_runtime"}
        assert len({item[2] for item in executed}) == 2
        assert len({item[3] for item in executed}) == 1
        completed = research.get_comparison_campaign(campaign["campaignCode"])
        assert completed is not None
        assert completed["status"] == "completed", completed["runs"]
        assert completed["runs"]["ortools"]["attemptedSeeds"] == 1
        assert completed["runs"]["organic_runtime"]["attemptedSeeds"] == 1
        assert completed["runs"]["organic_runtime"]["quarantineCount"] == 0
        assert completed["comparison"]["profileHash"] == research.comparison_profile_hash
        assert completed["comparison"]["recommendation"] in {
            "promote_ortools", "retain_current", "insufficient_evidence"
        }
        assert completed["comparison"]["minimumYieldMultiplier"] == 2.0
        assert completed["comparison"]["minimumCostReduction"] == 0.4
        assert "stageSeconds" in completed["runs"]["ortools"]
        assert "secondsPerQualified" in completed["runs"]["organic_runtime"]
        assert completed["runs"]["ortools"]["cpuSeconds"] is not None
        assert "qualifiedPerCpuHour" in completed["runs"]["ortools"]
        assert "cpuSecondsPerQualified" in completed["runs"]["organic_runtime"]
        listed = research.list_comparison_campaigns()
        assert listed[0]["campaignCode"] == campaign["campaignCode"]
        events = research.comparison_campaign_events(campaign["campaignCode"])
        assert {event["generationLane"] for event in events} == {"ortools", "organic_runtime"}
    finally:
        container.shutdown()


def test_comparison_campaign_preflights_both_lanes_before_persisting_or_starting():
    recipe = ResearchRecipeV2.build(_recipe(
        schemaVersion=2,
        tactics=[
            {
                "kind": "small_card_big_gain",
                "mode": "qualification_gate",
                "minimumScore": 70,
                "weight": 1,
                "perspectives": ["south"],
            },
            {
                "kind": "only_winning_move",
                "mode": "qualification_gate",
                "minimumScore": 70,
                "weight": 1,
                "perspectives": ["south"],
            },
        ],
        resources={
            "generationLane": "ortools",
            "seedStart": 7,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 2,
            "maximumConflicts": 50_000,
            "analysisMode": "screening",
            "playConstructionMode": "cp_sat_feasibility_line",
            "tacticalConstruction": {
                "schemaVersion": 1,
                "id": "tactical_play_precursor_v1",
                "revision": 1,
                "enabled": True,
                "maximumPayoffTricks": 4,
                "minimumLaterRawPoints": 10,
                "requirePrecursorForGatedTactics": False,
            },
            "tacticalDealBalance": {
                "schemaVersion": 1,
                "id": "balanced_tactical_deal_v1",
                "revision": 1,
                "enabled": True,
                "maximumTeamHighCardDifference": 2,
                "minimumHighCardsPerSeat": 2,
                "maximumHighCardsPerSeat": 4,
                "minimumTargetLowCards": 1,
            },
            "constructionPreScreen": {
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
            },
            "tacticalViabilityPreScreen": {
                "schemaVersion": 1,
                "id": "exact_tactical_viability_screen_v1",
                "revision": "1",
                "enabled": True,
                "maximumDecisions": 4,
                "requiredTacticsOnly": True,
                "rejectWhenExactAbsent": True,
            },
        },
        diversity={
            "schemaVersion": 1,
            "cellDimensions": ["contract", "source"],
            "topKPerCell": 2,
            "rejectedAuditModulus": 100,
        },
        renderPolicy={"mode": "on_demand"},
    ))
    mutations: list[str] = []

    def reject_mutation(name):
        def fail(*args, **kwargs):
            mutations.append(name)
            raise AssertionError(f"unexpected comparison mutation: {name}")

        return fail

    service = object.__new__(ResearchFoundryServiceV1)
    service.execution_context_factory = None
    service._workflow_copy = {
        "research_comparison_lane_organic": "organic runtime",
    }
    service.store = SimpleNamespace(
        get_recipe=lambda code: recipe.to_dict(),
        recipe_status=lambda code: "published",
        get_comparison_campaign=lambda code: None,
        put_recipe=reject_mutation("recipe"),
        create_run=reject_mutation("run"),
        put_comparison_campaign=reject_mutation("campaign"),
    )
    organic_preflighted = False

    def validate(payload):
        nonlocal organic_preflighted
        if dict(payload["resources"])["generationLane"] == "organic_runtime":
            organic_preflighted = True
            assert "tacticalConstruction" not in payload["resources"]
            assert "tacticalDealBalance" not in payload["resources"]
            assert payload["resources"]["constructionPreScreen"]["enabled"]
            assert payload["resources"]["tacticalViabilityPreScreen"]["enabled"]
            return {
                "valid": False,
                "recipe": copy.deepcopy(dict(payload)),
                "issues": [{
                    "severity": "error",
                    "code": "organic_clone_rejected",
                    "field": "resources.generationLane",
                }],
            }
        built = build_research_recipe(payload)
        return {"valid": True, "recipe": built.to_dict(), "issues": []}

    service.validate_recipe = validate

    with pytest.raises(ValueError, match="organic_clone_rejected"):
        service.create_comparison_campaign(
            recipe.code, idempotency_key="comparison-preflight"
        )

    assert mutations == []
    assert organic_preflighted is True


def test_comparison_campaign_rejects_lane_authority_drift_before_mutation():
    recipe = ResearchRecipeV1.build(_recipe(
        tactics=[],
        resources={
            "generationLane": "ortools",
            "seedStart": 7,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 2,
            "maximumConflicts": 50_000,
            "analysisMode": "screening",
        },
    ))
    mutations: list[str] = []

    def reject_mutation(name):
        def fail(*args, **kwargs):
            mutations.append(name)
            raise AssertionError(f"unexpected comparison mutation: {name}")

        return fail

    service = object.__new__(ResearchFoundryServiceV1)
    service.execution_context_factory = None
    service._workflow_copy = {
        "research_comparison_lane_organic": "organic runtime",
    }
    service.store = SimpleNamespace(
        get_recipe=lambda code: recipe.to_dict(),
        recipe_status=lambda code: "published",
        get_comparison_campaign=lambda code: None,
        put_recipe=reject_mutation("recipe"),
        create_run=reject_mutation("run"),
        put_comparison_campaign=reject_mutation("campaign"),
    )

    def validate(payload):
        raw = copy.deepcopy(dict(payload))
        if dict(raw["resources"])["generationLane"] == "organic_runtime":
            raw["profiles"]["rulesHash"] = _hash("drifted-rules")
        built = build_research_recipe(raw)
        return {"valid": True, "recipe": built.to_dict(), "issues": []}

    service.validate_recipe = validate

    with pytest.raises(ValueError, match="configuration authority mismatch"):
        service.create_comparison_campaign(
            recipe.code, idempotency_key="comparison-authority-drift"
        )

    assert mutations == []
