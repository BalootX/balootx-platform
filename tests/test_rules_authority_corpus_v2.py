from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.models import Project
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1


_PROFILES = (
    (
        "saudi_federation_2023_sweep.yaml",
        "rule-fixtures-saudi-federation-2023-sweep.json",
    ),
    (
        "saudi_federation_2023_v2.yaml",
        "rule-fixtures-saudi-federation-2023-v2.json",
    ),
    (
        "saudi_federation_2023_v8.yaml",
        "rule-fixtures-saudi-federation-2023-v8.json",
    ),
    (
        "saudi_project_optional_unratified_v1.yaml",
        "rule-fixtures-saudi-project-optional-unratified-v1.json",
    ),
)


def _case(corpus: dict, action_kind: str) -> dict:
    return next(
        item for item in corpus["cases"] if item["action"]["kind"] == action_kind
    )


def _hundred_sequence(rules: RuleBook) -> Project:
    domain = rules.project_domain
    return Project(
        Seat.NORTH,
        "hundred_sequence",
        tuple(
            Card.of(Suit.HEARTS, rank)
            for rank in (Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING)
        ),
        10,
        int(Rank.KING),
        domain.category_priorities["hundred_sequence"],
        declared=True,
        declaration_order=2,
        scoring=True,
    )


def _hundred_quartet(rules: RuleBook) -> Project:
    domain = rules.project_domain
    return Project(
        Seat.WEST,
        "hundred_four_kind",
        tuple(Card.of(suit, Rank.TEN) for suit in Suit),
        10,
        int(Rank.TEN),
        domain.category_priorities["hundred_four_kind"],
        declared=True,
        declaration_order=1,
        scoring=True,
    )


@pytest.mark.parametrize(("rules_name", "corpus_name"), _PROFILES)
def test_successor_rule_authority_corpus_is_current_and_executable(
    root: Path, rules_name: str, corpus_name: str
) -> None:
    command = [
        sys.executable,
        "scripts/generate_rule_fixtures.py",
        "--check",
        "--rules-path",
        f"configs/rules/{rules_name}",
        "--adjudication-path",
        "configs/adjudication/research_events_v2.yaml",
        "--output",
        f"examples/{corpus_name}",
    ]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr

    rules = RuleBook.load(root / "configs/rules" / rules_name)
    corpus = json.loads((root / "examples" / corpus_name).read_text(encoding="utf-8"))
    adjudication = yaml.safe_load(
        (root / "configs/adjudication/research_events_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert corpus["rulesProfile"] == rules.id
    assert corpus["rulesVersion"] == rules.version
    assert corpus["rulesSha256"] == rules.sha256

    exposed = _case(corpus, "verify_exposed_ace_recall")
    auction = AuctionDomainIRV1.from_rules(rules.data)
    machine = BiddingMachine(Seat.EAST, Suit.HEARTS, auction, Rank.ACE)
    for action in exposed["initialState"]["auctionPrefix"]:
        machine.submit(action)
    assert list(machine.legal_actions()) == exposed["resultingState"]["legalActions"]

    ashkal = _case(corpus, "verify_ashkal_roles")
    resolved = auction.resolve_action(
        "ashkal",
        round_index=1,
        actor=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        trump=None,
    )
    assert resolved.buyer.name.lower() == ashkal["resultingState"]["buyer"]
    assert resolved.purchase_recipient.name.lower() == ashkal["resultingState"][
        "purchaseRecipient"
    ]

    boundary = _case(corpus, "verify_sun_double_boundary")
    escalation = EscalationDomainIRV1.from_rules(rules.data)
    transition = escalation.transition(
        escalation.initial_state(),
        declarer=Seat.SOUTH,
        contract=Contract.SUN,
        match_score=(101, 100),
        actor=Seat.EAST,
        action="double",
    )
    assert transition.state.level == boundary["resultingState"]["level"]

    project_case = _case(corpus, "verify_hundred_priority")
    sequence = _hundred_sequence(rules)
    quartet = _hundred_quartet(rules)
    winner = rules.project_domain.comparison_winner(
        (sequence, quartet), second_trick_leader=Seat.NORTH
    )
    assert winner is not None
    assert winner.kind == project_case["resultingState"]["winnerKind"]

    baloot_case = _case(corpus, "verify_baloot_in_hundred")
    counts, blocked = rules.project_domain.baloot_project_state(
        (sequence,), owner=Seat.NORTH, trump_suit=Suit.HEARTS
    )
    assert counts is baloot_case["resultingState"]["counts"]
    assert blocked is False
    assert baloot_case["resultingState"]["verbalDeclarationAuthorized"] is False

    kabout_case = _case(corpus, "verify_doubled_kabout")
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    if scoring.kabout_winner == "trick_sweeper":
        from baloot_platform.research.ortools_scoring_stage import OrToolsScoringStageV1
        sweep = _case(corpus, "verify_kabout_sweeper")
        inputs = sweep["initialState"]
        solved = OrToolsScoringStageV1(scoring, native_conversion=True).solve(
            contract=Contract(inputs["contract"]), buyer=Seat.NORTH,
            escalation="normal", last_escalating_team=None,
            raw_points=inputs["rawPoints"], project_points=inputs["projectPoints"],
            baloot_points=(0, 0), trick_winner_teams=inputs["trickWinnerTeams"],
            starting_match_score=(0, 0),
        )
        assert solved.status == "satisfied"
        assert solved.score["roundWinner"] == sweep["resultingState"]["roundWinner"]
        assert solved.score["gamePoints"] == sweep["resultingState"]["awardedPoints"]
    for contract in (Contract.SUN, Contract.HOKUM):
        award = scoring.round_award(
            contract=contract,
            winner=Team.NORTH_SOUTH,
            declarer_team=Team.WEST_EAST,
            escalation="double",
            total_before_award=(0, 0),
            project_points=(0, 0),
            baloot_points=(0, 0),
            kabout=True,
        )
        assert award == (kabout_case["resultingState"][contract.value], 0)

    declaration = _case(corpus, "verify_project_declaration_mode")
    assert rules.project_domain.declaration_mode == declaration["resultingState"]["mode"]

    adjudication_cases = {
        item["initialState"]["eventId"]: item
        for item in corpus["cases"]
        if item["action"]["kind"] == "verify_adjudication_boundary"
    }
    assert set(adjudication_cases) == set(adjudication["events"])
    assert "false_ekka" not in adjudication_cases
    for event_id, event in adjudication["events"].items():
        expected = adjudication_cases[event_id]["resultingState"]
        assert expected == {
            "ordinaryTrace": event["ordinaryTrace"],
            "hardPredicate": event["hardPredicate"],
            "requiresTableReferee": event["requiresTableReferee"],
            "remedies": event["availableRemedies"],
        }
