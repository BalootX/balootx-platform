from __future__ import annotations

from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.research.reference_scoring import (
    ReferenceScoringInterpreterV1,
)
from baloot_platform.service.container import build_container


def test_reference_scoring_matches_runtime_complete_rounds():
    container = build_container()
    try:
        reference = ReferenceScoringInterpreterV1.from_rules(
            container.research.simulator.rules.data
        )
        for seed in range(1, 13):
            state = container.research.simulator.simulate(
                seed
            ).final_trace.trusted_final_state
            actual = dict(state["roundScore"])
            score = reference.score(
                contract=Contract.parse(state["contract"]),
                declarer=Seat.parse(state["declarer"]),
                escalation=str(state["escalation"]),
                trick_raw_points=[
                    int(trick["rawPoints"])
                    for trick in state["completedTricks"]
                ],
                trick_winners=[
                    Seat.parse(trick["winner"])
                    for trick in state["completedTricks"]
                ],
                project_points=actual["projectPoints"],
                baloot_points=actual["balootPoints"],
                counting_team=Team[str(actual["countingTeam"]).upper()],
            ).to_dict()
            for key in (
                "rawCardPoints",
                "cardGamePoints",
                "projectPoints",
                "balootPoints",
                "totalBeforeAward",
                "winner",
                "awardedPoints",
                "countingTeam",
                "multiplier",
                "kabout",
                "gahwa",
                "matchWinner",
            ):
                assert score[key] == actual[key]
    finally:
        container.shutdown()


def test_reference_scoring_conversion_boundaries_are_explicit():
    container = build_container()
    try:
        reference = ReferenceScoringInterpreterV1.from_rules(
            container.research.simulator.rules.data
        )
        assert reference.convert(25, Contract.SUN) == 5
        assert reference.convert(26, Contract.SUN) == 6
        assert reference.convert(25, Contract.HOKUM) == 2
        assert reference.convert(26, Contract.HOKUM) == 3
    finally:
        container.shutdown()


def test_scoring_behavioral_receipts_compare_all_three_targets():
    container = build_container()
    try:
        compiled = container.research.compiled_domain.compiled
        proofs = compiled.to_dict(include_payloads=True)["targetPayloads"][
            "behavioralProofs"
        ]
        expected = {
            "rules.scoring.last_trick_bonus_raw",
            "rules.scoring.card_game_point_totals.sun",
            "rules.scoring.card_game_point_totals.hokum",
            "rules.scoring.escalation.project_multiplier_cap",
            "rules.scoring.escalation.baloot_multiplier_cap",
            "rules.scoring.kabout.enabled",
            "rules.scoring.kabout.base_game_points.sun",
            "rules.scoring.kabout.base_game_points.hokum",
            "rules.scoring.match.target",
        }

        for primitive_id in expected:
            proof = proofs[primitive_id]
            assert proof["status"] == "proven_behavioral"
            assert proof["targetsObserved"] == ["reference", "runtime", "cpSat"]
            assert proof["referenceIndependent"] is True
            assert proof["rulesHash"] == container.research.simulator.rules.sha256
            assert proof["counterfactualRulesHash"] != proof["rulesHash"]
            assert proof["observationHash"]
            assert proof["proofHash"]

        assert proofs["rules.scoring.last_trick_bonus_raw"]["jointDependencies"] == [
            "rules.scoring.raw_card_totals.sun",
            "rules.scoring.raw_card_totals.hokum",
        ]
    finally:
        container.shutdown()
