from __future__ import annotations

from dataclasses import replace

import pytest

from baloot_platform.challenge.sakka_context import (
    ContextualizedLaffaV1,
    CuratedSakkaSignerV1,
    CuratedSakkaV1,
    NaturalSakkaV1,
    SakkaContextV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.enums import Seat


def _played(simulator, seed, *, dealer, match_score, round_number):
    result = simulator.simulate(
        seed,
        dealer=dealer,
        match_score=match_score,
        round_number=round_number,
    )
    assert result.publishable_laffa
    return result.final_trace


def test_natural_sakka_requires_uninterrupted_authoritative_match_continuity(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    traces = []
    score = (0, 0)
    dealer = Seat.EAST
    round_number = 1
    seed = 20
    while max(score) < 152 and len(traces) < 20:
        trace = _played(
            simulator,
            seed,
            dealer=dealer,
            match_score=score,
            round_number=round_number,
        )
        traces.append(trace)
        score = tuple(trace.trusted_final_state["matchScore"])
        dealer = Seat.parse(trace.trusted_final_state["dealer"]).next()
        round_number = int(trace.trusted_final_state["roundNumber"]) + 1
        seed += 1

    sakka = NaturalSakkaV1.build(traces, target=152)
    repeated = NaturalSakkaV1.build(tuple(traces), target=152)
    assert sakka.sha256 == repeated.sha256
    assert sakka.provenance == "natural_sakka"
    assert sakka.final_score == score and max(score) >= 152
    assert len(sakka.rounds) == len(traces)

    broken = list(traces)
    second_start = broken[1].trusted_final_state["generation"]
    broken[1] = _played(
        simulator,
        int(second_start["rootSeed"]),
        dealer=Seat.parse(second_start["dealer"]),
        match_score=(99, 99),
        round_number=int(second_start["roundNumber"]),
    )
    with pytest.raises(ValueError, match="score continuity"):
        NaturalSakkaV1.build(broken, target=152)


def test_contextualized_laffa_proves_same_deal_and_full_context_resimulation(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    parent = _played(
        simulator,
        2,
        dealer=Seat.EAST,
        match_score=(0, 0),
        round_number=1,
    )
    context = SakkaContextV1.build(
        starting_score=(100, 100),
        dealer=Seat.EAST,
        round_number=10,
        target=152,
        reachability_mode="verified_enumeration",
        reachability_proof_hash="d" * 64,
    )
    descendant = _played(
        simulator,
        2,
        dealer=Seat.EAST,
        match_score=context.starting_score,
        round_number=context.round_number,
    )
    parent_decision = next(
        item for item in parent.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    descendant_decision = next(
        item for item in descendant.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    contextualized = ContextualizedLaffaV1.build(
        parent=parent,
        descendant=descendant,
        context=context,
        original_decision_id=parent_decision.decision_id,
        descendant_decision_id=descendant_decision.decision_id,
        analysis_hash="e" * 64,
        context_amplification={"matchProbabilitySwing": 35.0, "climax": 20.0},
        strategy_changed=(
            parent_decision.chosen_action != descendant_decision.chosen_action
        ),
        recommendation_stable=True,
    )
    assert contextualized.provenance == "contextualized_laffa"
    assert parent.seed_provenance["deckCommitment"] == descendant.seed_provenance["deckCommitment"]

    with pytest.raises(ValueError, match="unstable recommendations"):
        ContextualizedLaffaV1.build(
            parent=parent,
            descendant=descendant,
            context=context,
            original_decision_id=parent_decision.decision_id,
            descendant_decision_id=descendant_decision.decision_id,
            analysis_hash="e" * 64,
            context_amplification={"matchProbabilitySwing": 35.0},
            strategy_changed=False,
            recommendation_stable=False,
        )


def test_curated_sakka_requires_full_descendant_continuity_and_signature(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    score = (100, 100)
    dealer = Seat.EAST
    round_number = 10
    seed = 300
    contexts = []
    descendants = []
    while max(score) < 152 and len(descendants) < 10:
        parent = _played(
            simulator,
            seed,
            dealer=dealer,
            match_score=(0, 0),
            round_number=1,
        )
        context = SakkaContextV1.build(
            starting_score=score,
            dealer=dealer,
            round_number=round_number,
            target=152,
            reachability_mode="verified_enumeration",
            reachability_proof_hash=f"{seed:064x}",
        )
        descendant = _played(
            simulator,
            seed,
            dealer=dealer,
            match_score=score,
            round_number=round_number,
        )
        parent_decision = next(
            item for item in parent.decision_ledger.records
            if item.decision_kind == "play" and not item.forced
        )
        descendant_decision = next(
            item for item in descendant.decision_ledger.records
            if item.decision_kind == "play" and not item.forced
        )
        contexts.append(ContextualizedLaffaV1.build(
            parent=parent,
            descendant=descendant,
            context=context,
            original_decision_id=parent_decision.decision_id,
            descendant_decision_id=descendant_decision.decision_id,
            analysis_hash=f"{seed + 1:064x}",
            context_amplification={"matchProbabilitySwing": 10.0},
            strategy_changed=(
                parent_decision.chosen_action != descendant_decision.chosen_action
            ),
            recommendation_stable=True,
        ))
        descendants.append(descendant)
        score = tuple(descendant.trusted_final_state["matchScore"])
        dealer = Seat.parse(descendant.trusted_final_state["dealer"]).next()
        round_number = int(descendant.trusted_final_state["roundNumber"]) + 1
        seed += 1

    signer = CuratedSakkaSignerV1.from_secret(
        b"curated-sakka-test-signing-secret-at-least-32-bytes"
    )
    curated = CuratedSakkaV1.build(
        contexts,
        descendants,
        target=152,
        signer=signer,
    )
    assert curated.provenance == "curated_sakka"
    assert curated.final_score == score
    assert curated.verify(signer)
    assert len(curated.rounds) == len(descendants)
    assert all("descendantTraceHash" in item for item in curated.rounds)
    assert not replace(curated, final_score=(999, 0)).verify(signer)

    with pytest.raises(ValueError, match="reuse a dealt-card fingerprint"):
        CuratedSakkaV1.build(
            (contexts[0], contexts[0]),
            (descendants[0], descendants[0]),
            target=152,
            signer=signer,
        )
