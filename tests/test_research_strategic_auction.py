from __future__ import annotations

from dataclasses import replace

import pytest

import baloot_platform.research.strategic_auction as strategic_auction_module
from baloot_platform.analysis.bidding import (
    BidAnalysisResult,
    BidChoice,
    BidChoiceEvaluation,
    legal_bid_choices,
)
from baloot_platform.analysis.decision_scores import (
    PrimaryDecisionScoreEvidenceV1,
    canonical_sha256_v1,
)
from baloot_platform.analysis.search_backend import PythonReferenceSearchBackend
from baloot_platform.challenge.analysis_pipeline import TraceAnalysisProfileV1
from baloot_platform.challenge.decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.challenge.simulation import observer_safe_bid_state_v1
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.research.contracts import ResearchRecipeV1
from baloot_platform.research.strategic_auction import (
    ActorSafeCpSatAuctionProfileV1,
    ActorSafeCpSatAuctionSelectorV1,
)
from baloot_platform.service.container import build_container
from scripts.run_actor_safe_auction_pilot import _context_bound_auction_selector
from scripts.run_strategic_play_pilot import _recipe, _wait


def _profile(*, samples: int = 8) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "requestedSamples": samples,
        "integerScale": 1_000,
        "maximumConflicts": 50_000,
        "maximumSelectionSeconds": 5.0,
        "objective": "expected_round_margin",
        "tieBreak": (
            "round_lead_probability_then_best_share_then_legal_action_order_v1"
        ),
        "requireOptimalForQualification": True,
    }


def _deep_profile(
    *,
    bid_samples: int = 8,
    backend_id: str = "python_reference_exact_search_v1",
) -> TraceAnalysisProfileV1:
    return TraceAnalysisProfileV1.build(
        bid_samples=bid_samples,
        play_samples=12,
        exact_allocation_limit=128,
        exact_search_max_remaining=9,
        maximum_sampling_attempts=1_000,
        exact_search_max_nodes=80_000,
        exact_search_max_seconds=0.4,
        exact_search_max_depth=32,
        exact_search_max_cache_entries=100_000,
        search_backend_id=backend_id,
        objective="match_win_probability",
        use_conventions=False,
    )


def _auction_state(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=73_001)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    return state, machine


def _analysis(actor, samples: int, machine) -> BidAnalysisResult:
    choices = []
    for index, choice in enumerate(legal_bid_choices(machine)):
        if choice.action == "sun":
            metrics = (3.50, 0.65, 0.70, -1.0, 1.0, 0.75)
        elif choice.action == "pass":
            metrics = (-1.25, 0.40, 0.0, -4.0, 1.5, 0.25)
        else:
            metrics = (-2.0 - index, 0.30, 0.25, -5.0, 2.0, 0.10)
        choices.append(BidChoiceEvaluation(choice, *metrics, samples))
    choices.sort(
        key=lambda item: (
            item.expected_match_margin,
            item.team_win_probability,
            item.best_share,
        ),
        reverse=True,
    )
    frozen_choices = tuple(choices)
    return BidAnalysisResult(
        actor,
        frozen_choices[0].choice,
        frozen_choices,
        "paired_observer_safe_bid_reference_v2",
        samples,
        50,
        50,
        (),
    )


def _primary_bid_seed(state, machine, identity_hash: str) -> tuple[str, int]:
    actor = machine.actor
    assert actor is not None
    public_hash = canonical_sha256_v1(observer_safe_bid_state_v1(state, machine, actor))
    return public_hash, derive_actor_observer_cohort_seed_v1(
        public_prefix_hash=public_hash,
        decision_kind="bid",
        acting_seat=actor,
        cohort_authority_hash=identity_hash,
        cohort_index=0,
        governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
    )


def _analysis_with_recommended_action(
    actor,
    samples: int,
    machine,
    recommended: BidChoice,
) -> BidAnalysisResult:
    base = _analysis(actor, samples, machine)
    return BidAnalysisResult(
        base.actor,
        recommended,
        base.choices,
        base.method,
        base.hidden_deals,
        base.difficulty,
        base.debate_score,
        base.explanation,
    )


def _analysis_with_illegal_action(actor, samples: int, machine) -> BidAnalysisResult:
    base = _analysis(actor, samples, machine)
    illegal = BidChoiceEvaluation(
        BidChoice("hokum"),
        -9.0,
        0.1,
        0.1,
        -10.0,
        2.0,
        0.0,
        samples,
    )
    return BidAnalysisResult(
        base.actor,
        base.recommended,
        (base.choices[0], *base.choices[1:-1], illegal),
        base.method,
        base.hidden_deals,
        base.difficulty,
        base.debate_score,
        base.explanation,
    )


def test_actor_safe_cp_sat_auction_profile_is_closed_and_hashable() -> None:
    profile = ActorSafeCpSatAuctionProfileV1.build(_profile())
    assert ActorSafeCpSatAuctionProfileV1.build(profile.to_dict()) == profile
    assert len(profile.sha256) == 64
    assert ActorSafeCpSatAuctionSelectorV1.descriptor.version == "1.1.0"
    assert "canonical_decision_score_evidence" in (
        ActorSafeCpSatAuctionSelectorV1.descriptor.capabilities
    )

    invalid = _profile()
    invalid["hiddenDefault"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        ActorSafeCpSatAuctionProfileV1.build(invalid)


def test_deep_analysis_binding_uses_exact_bid_samples_and_freezes_both_hashes(
    monkeypatch,
    rules,
) -> None:
    deep_profile = _deep_profile()
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    backend = PythonReferenceSearchBackend()
    captured: dict[str, object] = {}

    def fake_analyze_bid(state, machine, actor, rulebook, **kwargs):
        captured.update(kwargs)
        return _analysis(actor, int(kwargs["requested_samples"]), machine)

    monkeypatch.setattr(
        strategic_auction_module,
        "analyze_bid",
        fake_analyze_bid,
    )
    state, machine = _auction_state(rules)
    identity_hash = "a" * 64
    selector = ActorSafeCpSatAuctionSelectorV1(
        selector_profile,
        exact_search_backend=backend,
        trace_analysis_profile=deep_profile,
        analysis_identity_hash=identity_hash,
    )
    public_hash, seed = _primary_bid_seed(state, machine, identity_hash)

    action, trump, trace = selector.choose_with_trace(
        state,
        machine,
        rules,
        seed=seed,
    )

    assert (action, trump) == ("sun", None)
    assert captured["requested_samples"] == deep_profile.bid_samples
    assert trace["analysisRequestedSamples"] == deep_profile.bid_samples
    assert trace["traceAnalysisProfileHash"] == deep_profile.sha256
    assert trace["analysisIdentityHash"] == identity_hash
    assert trace["publicPrefixHash"] == public_hash
    assert trace["cohortSeedPolicy"] == "actor_observer_cohort_seed_v1"
    assert trace["cohortIndex"] == 0
    assert trace["cohortSeed"] == seed
    assert trace["bidObjective"] == "expected_round_margin"
    assert trace["playObjective"] == "match_win_probability"
    assert all(
        "fixedPointSelectionObjective" not in item for item in trace["rankedActions"]
    )
    assert all(
        set(item)
        == {
            "rank",
            "actionId",
            "action",
            "fixedPointExpectedRoundMargin",
            "fixedPointRoundLeadProbability",
            "fixedPointBestShare",
            "deterministicActionPreference",
            "fixedPointSelectionObjective",
        }
        for item in trace["cpSatObjectiveRows"]
    )
    legal_order = [choice.id for choice in legal_bid_choices(machine)]
    preference_by_action = {
        item["actionId"]: item["deterministicActionPreference"]
        for item in trace["cpSatObjectiveRows"]
    }
    assert preference_by_action == {
        action_id: len(legal_order) - index
        for index, action_id in enumerate(legal_order)
    }
    evidence = PrimaryDecisionScoreEvidenceV1.from_mapping(
        trace["primaryDecisionScoreEvidence"]
    )
    assert evidence.sha256 == trace["primaryDecisionScoreEvidenceHash"]
    assert evidence.rows_hash == trace["decisionScoreRowsHash"]
    assert evidence.public_prefix_hash == public_hash
    assert evidence.rules_hash == rules.sha256
    assert evidence.analysis_identity_hash == identity_hash
    assert evidence.trace_analysis_profile_hash == deep_profile.sha256
    assert evidence.cohort_seed == seed
    assert evidence.objective == "expected_round_margin"
    assert evidence.tie_break == (
        "round_lead_probability_then_best_share_then_legal_action_order_v1"
    )
    assert evidence.tie_break == selector_profile.tie_break
    assert evidence.backend == deep_profile.search_backend_id
    assert evidence.hidden_deal_support == deep_profile.bid_samples
    assert evidence.effective_sample_size == float(deep_profile.bid_samples)
    assert evidence.recommended_action == {"action": "sun", "trumpSuit": None}
    assert trace["cpSatCanonicalInputHash"] == canonical_sha256_v1(
        {
            "schemaVersion": 1,
            "authority": "primary_decision_score_evidence_v1",
            "decisionScoreEvidenceHash": evidence.sha256,
            "decisionScoreRowsHash": evidence.rows_hash,
            "legalActionSetHash": evidence.legal_action_set_hash,
            "integerScale": selector_profile.integer_scale,
            "objective": selector_profile.objective,
            "tieBreak": selector_profile.tie_break,
            "lexicographicWeights": {
                "fixedPointExpectedRoundMargin": 4_008_004,
                "fixedPointRoundLeadProbability": 4_004,
                "fixedPointBestShare": 4,
                "deterministicActionPreference": 1,
            },
            "orderedFixedPointRows": trace["cpSatObjectiveRows"],
        }
    )
    assert trace["cpSatObjectiveScope"] == (
        "deterministic_fixed_point_argmax_only_not_baloot_score_or_decision_estimate"
    )
    assert "objectiveValue" not in trace
    assert "roundScore" not in trace
    assert "balootScore" not in trace


def test_real_bid_analyzer_emits_self_validating_primary_score_evidence(rules) -> None:
    deep_profile = _deep_profile(bid_samples=2)
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    identity_hash = "e" * 64
    selector = ActorSafeCpSatAuctionSelectorV1(
        selector_profile,
        exact_search_backend=PythonReferenceSearchBackend(),
        trace_analysis_profile=deep_profile,
        analysis_identity_hash=identity_hash,
    )
    state, machine = _auction_state(rules)
    public_hash, seed = _primary_bid_seed(state, machine, identity_hash)

    action, trump, trace = selector.choose_with_trace(
        state,
        machine,
        rules,
        seed=seed,
    )

    evidence = PrimaryDecisionScoreEvidenceV1.from_mapping(
        trace["primaryDecisionScoreEvidence"]
    )
    assert {"action": action, "trumpSuit": trump.name.lower() if trump else None} == (
        evidence.recommended_action
    )
    assert evidence.public_prefix_hash == public_hash
    assert evidence.cohort_seed == seed
    assert evidence.hidden_deal_support == 2
    assert len(evidence.rows) == len(legal_bid_choices(machine))
    assert trace["solverStatus"] == "optimal"


def test_analysis_identity_is_part_of_the_cp_sat_proof_hash(
    monkeypatch,
    rules,
) -> None:
    deep_profile = _deep_profile()
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    backend = PythonReferenceSearchBackend()

    def fake_analyze_bid(state, machine, actor, rulebook, **kwargs):
        return _analysis(actor, int(kwargs["requested_samples"]), machine)

    monkeypatch.setattr(
        strategic_auction_module,
        "analyze_bid",
        fake_analyze_bid,
    )
    state, machine = _auction_state(rules)
    traces = []
    for identity_hash in ("1" * 64, "2" * 64):
        selector = ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            exact_search_backend=backend,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash=identity_hash,
        )
        _, seed = _primary_bid_seed(state, machine, identity_hash)
        traces.append(selector.choose_with_trace(state, machine, rules, seed=seed)[2])

    assert traces[0]["selectedAction"] == traces[1]["selectedAction"]
    assert traces[0]["modelHash"] != traces[1]["modelHash"]
    assert (
        traces[0]["primaryDecisionScoreEvidenceHash"]
        != traces[1]["primaryDecisionScoreEvidenceHash"]
    )


def test_selection_requires_deep_authority_and_exact_primary_seed(rules) -> None:
    deep_profile = _deep_profile()
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    backend = PythonReferenceSearchBackend()
    state, machine = _auction_state(rules)

    legacy = ActorSafeCpSatAuctionSelectorV1(
        selector_profile,
        exact_search_backend=backend,
    )
    with pytest.raises(ValueError, match="requires frozen deep analysis authority"):
        legacy.choose_with_trace(state, machine, rules, seed=1)

    identity_hash = "b" * 64
    selector = ActorSafeCpSatAuctionSelectorV1(
        selector_profile,
        exact_search_backend=backend,
        trace_analysis_profile=deep_profile,
        analysis_identity_hash=identity_hash,
    )
    _, seed = _primary_bid_seed(state, machine, identity_hash)
    with pytest.raises(TypeError, match="seed must be an integer"):
        selector.choose_with_trace(state, machine, rules, seed=str(seed))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="primary actor-observer cohort"):
        selector.choose_with_trace(state, machine, rules, seed=seed + 1)
    drifted_state = state.clone()
    drifted_state.rules_hash = "0" * 64
    with pytest.raises(ValueError, match="state rules do not match"):
        selector.choose_with_trace(drifted_state, machine, rules, seed=seed)


@pytest.mark.parametrize(
    ("analysis_factory", "message"),
    (
        (
            lambda actor, samples, machine: _analysis_with_recommended_action(
                actor,
                samples,
                machine,
                BidChoice("pass"),
            ),
            "recommendedAction must be the first ordered score row",
        ),
        (
            _analysis_with_illegal_action,
            "supplied legal action set hash does not match score rows",
        ),
    ),
)
def test_canonical_bid_evidence_rejects_action_authority_drift(
    monkeypatch,
    rules,
    analysis_factory,
    message: str,
) -> None:
    deep_profile = _deep_profile()
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    identity_hash = "c" * 64
    selector = ActorSafeCpSatAuctionSelectorV1(
        selector_profile,
        exact_search_backend=PythonReferenceSearchBackend(),
        trace_analysis_profile=deep_profile,
        analysis_identity_hash=identity_hash,
    )
    state, machine = _auction_state(rules)
    _, seed = _primary_bid_seed(state, machine, identity_hash)

    def fake_analyze_bid(state, machine, actor, rulebook, **kwargs):
        return analysis_factory(actor, int(kwargs["requested_samples"]), machine)

    monkeypatch.setattr(strategic_auction_module, "analyze_bid", fake_analyze_bid)
    with pytest.raises(ValueError, match=message):
        selector.choose_with_trace(state, machine, rules, seed=seed)


def test_cp_sat_argmax_must_match_canonical_bid_recommendation(
    monkeypatch,
    rules,
) -> None:
    deep_profile = _deep_profile()
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    identity_hash = "d" * 64
    selector = ActorSafeCpSatAuctionSelectorV1(
        selector_profile,
        exact_search_backend=PythonReferenceSearchBackend(),
        trace_analysis_profile=deep_profile,
        analysis_identity_hash=identity_hash,
    )
    state, machine = _auction_state(rules)
    _, seed = _primary_bid_seed(state, machine, identity_hash)

    def fake_analyze_bid(state, machine, actor, rulebook, **kwargs):
        samples = int(kwargs["requested_samples"])
        legal = legal_bid_choices(machine)
        sun = next(choice for choice in legal if choice.action == "sun")
        passed = next(choice for choice in legal if choice.action == "pass")
        rows = [
            BidChoiceEvaluation(sun, 0.0004, 0.0, 0.5, -1.0, 1.0, 0.0, samples),
            BidChoiceEvaluation(passed, 0.0003, 1.0, 0.5, -1.0, 1.0, 1.0, samples),
        ]
        rows.extend(
            BidChoiceEvaluation(choice, -10.0, 0.0, 0.0, -10.0, 1.0, 0.0, samples)
            for choice in legal
            if choice not in {sun, passed}
        )
        return BidAnalysisResult(
            actor,
            sun,
            tuple(rows),
            "paired_observer_safe_bid_reference_v2",
            samples,
            50,
            50,
            (),
        )

    monkeypatch.setattr(strategic_auction_module, "analyze_bid", fake_analyze_bid)
    with pytest.raises(ValueError, match="argmax does not match"):
        selector.choose_with_trace(state, machine, rules, seed=seed)


def test_deep_analysis_binding_fails_closed_on_partial_or_drifted_authority() -> None:
    deep_profile = _deep_profile()
    selector_profile = ActorSafeCpSatAuctionProfileV1.build(
        _profile(samples=deep_profile.bid_samples)
    )
    backend = PythonReferenceSearchBackend()

    with pytest.raises(ValueError, match="checksum mismatch"):
        ActorSafeCpSatAuctionSelectorV1(
            replace(selector_profile, sha256="0" * 64),
            exact_search_backend=backend,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="unsupported actor-safe auction tie-break"):
        ActorSafeCpSatAuctionSelectorV1(
            replace(
                selector_profile,
                tie_break="win_probability_then_best_share_then_action_id",
            ),
            exact_search_backend=backend,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash="a" * 64,
        )

    with pytest.raises(ValueError, match="injected together"):
        ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            exact_search_backend=backend,
            trace_analysis_profile=deep_profile,
        )
    with pytest.raises(ValueError, match="injected together"):
        ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            exact_search_backend=backend,
            analysis_identity_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="canonical SHA-256"):
        ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            exact_search_backend=backend,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash="not-a-sha",
        )
    with pytest.raises(ValueError, match="profile checksum mismatch"):
        ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            exact_search_backend=backend,
            trace_analysis_profile=replace(deep_profile, sha256="0" * 64),
            analysis_identity_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="explicit exact-search backend"):
        ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="does not match"):
        ActorSafeCpSatAuctionSelectorV1(
            selector_profile,
            exact_search_backend=backend,
            trace_analysis_profile=_deep_profile(backend_id="another_backend"),
            analysis_identity_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="must mirror"):
        ActorSafeCpSatAuctionSelectorV1(
            ActorSafeCpSatAuctionProfileV1.build(_profile(samples=7)),
            exact_search_backend=backend,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash="a" * 64,
        )


def test_recipe_requires_the_strategic_auction_profile_for_cp_sat_mode(
    tmp_path,
) -> None:
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        payload = _recipe(
            container.research,
            mode="actor_safe_agent_verified",
            attempts=1,
            seed_start=202608080001,
        )
        payload["resources"]["auctionConstructionMode"] = (
            "cp_sat_actor_safe_decision_v1"
        )
        with pytest.raises(ValueError, match="strategic auction profile"):
            ResearchRecipeV1.build(payload)

        payload["resources"]["strategicAuction"] = _profile()
        built = ResearchRecipeV1.build(payload)
        assert built.resources["strategicAuction"]["sha256"]
    finally:
        container.shutdown()


def test_cp_sat_auction_is_verified_but_cannot_bypass_finalist_primary_policy(
    tmp_path,
) -> None:
    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        payload = _recipe(
            research,
            mode="actor_safe_agent_verified",
            attempts=1,
            seed_start=202608080001,
        )
        selected_bundle = container.configuration.resolve()["bundle"]
        _, selected_context = (
            research.execution_context_factory.resolve_bundle_revision(
                selected_bundle["revisionId"]
            )
        )
        deep_bid_samples = selected_context.analyzers["deep"].profile.bid_samples
        pilot_selector = _context_bound_auction_selector(
            selected_context,
            ActorSafeCpSatAuctionProfileV1.build(
                _profile(samples=deep_bid_samples)
            ),
        )
        assert (
            pilot_selector.trace_analysis_profile
            is selected_context.analyzers["deep"].profile
        )
        assert pilot_selector.analysis_identity_hash == (
            selected_context.analysis_identity_hashes["deep"]
        )
        assert pilot_selector.exact_search_backend is (
            selected_context.exact_search_backend
        )
        payload["schemaVersion"] = 3
        payload.pop("profiles")
        payload["configuration"] = {
            "schemaVersion": 1,
            "bundleRevisionId": selected_bundle["revisionId"],
        }
        payload["tactics"] = []
        payload["resources"].update(
            {
                "auctionConstructionMode": "cp_sat_actor_safe_decision_v1",
                "strategicAuction": _profile(samples=deep_bid_samples),
            }
        )
        payload["qualification"] = {
            "minimumTruth": "provisional",
            "minimumConfidence": 0,
            "minimumSurroundingQuality": 0,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 999,
        }
        validated = research.validate_recipe(payload)
        assert validated["valid"] is True, validated["issues"]
        created = research.store.put_recipe(validated["recipe"])
        submitted = research.submit(
            created["code"],
            idempotency_key="cp-sat-actor-safe-auction-integration",
        )
        run = _wait(research, submitted["runCode"], timeout=900)

        assert run["status"] == "completed"
        assert run["auctionFeasible"] == 1
        assert run["quarantineCount"] == 0
        results = research.list_results(run_code=run["runCode"], limit=10)["items"]
        receipts = research.store.list_sequential_search_receipts(
            run_code=run["runCode"], limit=10
        )["items"]
        assert results == []
        assert run["semanticReplayPassed"] == 1
        assert run["runtimeParityPassed"] == 1
        assert len(receipts) == 1
        receipt = receipts[0]["receiptBundle"]
        assert receipt["failureCodes"] == ["finalist_primary_policy_unavailable"]
        assert receipt["stopReason"] == "incomplete"
        assert receipt["evaluationReceipts"][0]["evaluation"]["outcome"] == (
            "incomplete"
        )
    finally:
        container.shutdown()
