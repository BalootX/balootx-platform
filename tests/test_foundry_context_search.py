from __future__ import annotations

from dataclasses import dataclass

import pytest

from baloot_platform.challenge.analysis_pipeline import TraceDecisionAnalysisV3
from baloot_platform.challenge.context_search import (
    CriticalContextSearchProfileV1,
    CriticalContextSearchV1,
    NaturalHistoryContextVerifierV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.sakka_context import SakkaContextV1
from baloot_platform.challenge.score_reachability import (
    AuthoritativeRoundOutcomeCatalogV1,
    AuthoritativeScoreReachabilityV1,
    ScoreReachabilityProfileV1,
    VerifiedEnumeratedContextV1,
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


@dataclass(frozen=True)
class _Result:
    decisions: tuple[TraceDecisionAnalysisV3, ...]
    sha256: str


class _ActorSafeAnalyzer:
    def __init__(self, simulator):
        self.rules = simulator.rules
        self.profile = type("Profile", (), {"sha256": "a" * 64})()

    def analyze_trace(self, trace, *, cancel_check=None):
        start = tuple(trace.trusted_final_state["generation"]["matchScore"])
        contextualized = start != (0, 0)
        decisions = []
        for record in trace.decision_ledger.records:
            chosen = dict(record.chosen_action)
            if record.decision_kind == "play" and not record.forced:
                alternative = next(
                    dict(item) for item in record.legal_actions
                    if canonical_sha256(dict(item)) != canonical_sha256(chosen)
                )
                low, high = ((0.2, 0.8) if contextualized else (0.4, 0.6))
                rows = (
                    {
                        "action": chosen,
                        "expectedMargin": 10.0,
                        "winProbability": 0.7,
                        "matchWinProbability": high,
                    },
                    {
                        "action": alternative,
                        "expectedMargin": 0.0,
                        "winProbability": 0.3,
                        "matchWinProbability": low,
                    },
                )
                supported = ("match_win_probability",)
                confidence = 0.95
            else:
                rows = ({
                    "action": chosen,
                    "expectedMargin": 0.0,
                    "winProbability": 0.5,
                },)
                supported = ("legality",)
                confidence = 1.0
            decisions.append(TraceDecisionAnalysisV3(
                record.decision_id,
                record.ordinal,
                record.acting_seat.name.lower(),
                record.decision_kind,
                record.forced,
                chosen,
                chosen,
                rows,
                "test_actor_safe_analysis",
                16,
                16.0,
                confidence,
                50.0,
                0.0,
                0.0,
                0.0,
                False,
                supported,
                (),
                canonical_sha256({"trace": trace.sha256, "decision": record.decision_id}),
            ))
        return _Result(
            tuple(decisions),
            canonical_sha256({"trace": trace.sha256, "profile": self.profile.sha256}),
        )


def _profile():
    return CriticalContextSearchProfileV1.build({
        "schemaVersion": 1,
        "id": "critical_context_test_v1",
        "revision": "1",
        "minimumAnalysisConfidence": 0.8,
        "minimumMatchProbabilityAmplification": 0.2,
        "requireSameRecommendation": True,
        "maximumContexts": 8,
    })


def test_natural_history_context_is_proven_from_exact_continuity(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    first = _played(
        simulator,
        20,
        dealer=Seat.EAST,
        match_score=(0, 0),
        round_number=1,
    )
    first_score = tuple(first.trusted_final_state["matchScore"])
    second = _played(
        simulator,
        21,
        dealer=Seat.parse(first.trusted_final_state["dealer"]).next(),
        match_score=first_score,
        round_number=2,
    )
    verified = NaturalHistoryContextVerifierV1.build_context((first, second), target=1000)
    context = verified.context
    assert context.starting_score == tuple(second.trusted_final_state["matchScore"])
    assert context.round_number == 3
    assert context.reachability_mode == "natural_history"
    assert context.previous_round_hashes == (first.sha256, second.sha256)
    assert verified.history_trace_hashes == context.previous_round_hashes

    with pytest.raises(ValueError, match="continuity"):
        NaturalHistoryContextVerifierV1.build_context((second, first), target=1000)


def test_context_search_resimulates_exact_cards_and_accepts_only_amplified_actor_truth(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    history = _played(
        simulator,
        20,
        dealer=Seat.EAST,
        match_score=(0, 0),
        round_number=1,
    )
    verified = NaturalHistoryContextVerifierV1.build_context((history,), target=1000)
    context = verified.context
    parent = _played(
        simulator,
        2,
        dealer=context.dealer,
        match_score=(0, 0),
        round_number=1,
    )
    critical = next(
        item for item in parent.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    result = CriticalContextSearchV1(
        simulator,
        _ActorSafeAnalyzer(simulator),
        _profile(),
    ).search(parent, decision_id=critical.decision_id, contexts=(verified,))

    assert len(result.accepted) == 1
    assert result.selected_hash == result.accepted[0].sha256
    assert result.evaluated[0]["matchProbabilityAmplification"] == pytest.approx(0.4)
    assert result.descendant_traces[0].seed_provenance["deckCommitment"] == (
        parent.seed_provenance["deckCommitment"]
    )
    assert result.accepted[0].provenance == "contextualized_laffa"


def test_context_search_rejects_unproven_enumerated_scores(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    parent = _played(
        simulator,
        2,
        dealer=Seat.EAST,
        match_score=(0, 0),
        round_number=1,
    )
    critical = next(
        item for item in parent.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    unproven = SakkaContextV1.build(
        starting_score=(100, 100),
        dealer=Seat.EAST,
        round_number=10,
        target=152,
        reachability_mode="verified_enumeration",
        reachability_proof_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="verified reachability"):
        CriticalContextSearchV1(
            simulator,
            _ActorSafeAnalyzer(simulator),
            _profile(),
        ).search(parent, decision_id=critical.decision_id, contexts=(unproven,))


def test_context_search_accepts_engine_witnessed_enumerated_context(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    east = _played(
        simulator, 1, dealer=Seat.EAST, match_score=(0, 0), round_number=1
    )
    north = _played(
        simulator, 2, dealer=Seat.NORTH, match_score=(0, 0), round_number=1
    )
    goal = tuple(
        int(east.trusted_final_state["roundScore"]["awardedPoints"][index])
        + int(north.trusted_final_state["roundScore"]["awardedPoints"][index])
        for index in range(2)
    )
    reachability_profile = ScoreReachabilityProfileV1.build({
        "schemaVersion": 1,
        "id": "context_search_reachability_test_v1",
        "revision": "1",
        "maximumWitnessTraces": 10,
        "witnessesPerOutcome": 4,
        "maximumRounds": 10,
        "maximumStates": 10000,
    })
    catalog = AuthoritativeRoundOutcomeCatalogV1.build(
        (east, north), rules=simulator.rules, profile=reachability_profile
    )
    proof = AuthoritativeScoreReachabilityV1(
        catalog, simulator.rules, reachability_profile
    ).prove(goal, initial_dealer=Seat.EAST, target_dealer=Seat.WEST)
    verified = VerifiedEnumeratedContextV1.build(proof)
    parent = _played(
        simulator,
        11,
        dealer=Seat.WEST,
        match_score=(0, 0),
        round_number=1,
    )
    critical = next(
        item for item in parent.decision_ledger.records
        if item.decision_kind == "play" and not item.forced
    )
    result = CriticalContextSearchV1(
        simulator, _ActorSafeAnalyzer(simulator), _profile()
    ).search(parent, decision_id=critical.decision_id, contexts=(verified,))
    assert len(result.accepted) == 1
    assert result.accepted[0].context_hash == verified.context.sha256
    assert result.accepted[0].provenance == "contextualized_laffa"
