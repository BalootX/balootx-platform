from __future__ import annotations

from pathlib import Path

import pytest

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.project_ir import project_id_v1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.serialization import replay_trusted_context
from baloot_platform.research.ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
)
from baloot_platform.research.ortools_exact_state import (
    OrToolsExactStateNormalizerV1,
)
from baloot_platform.research.recipe_v4 import research_recipe_v4_authoring_catalog
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
from baloot_platform.service.api import (
    ExactTraceLegalActionV1Response,
    ExactTracePrefixV1Request,
)


def _rules(root: Path, name: str) -> RuleBook:
    return RuleBook.load(root / f"configs/rules/{name}.yaml")


def _hands() -> dict[Seat, tuple[Card, ...]]:
    north = tuple(
        Card.parse(code)
        for code in ("S7", "S8", "S9", "H7", "H8", "H9", "DA", "CK")
    )
    remainder = tuple(card for card in FULL_DECK if card not in north)
    return {
        Seat.NORTH: north,
        Seat.WEST: remainder[:8],
        Seat.SOUTH: remainder[8:16],
        Seat.EAST: remainder[16:],
    }


def _complete_actions(domain, projects, dealer: Seat, *, declare_first: bool):
    state = domain.declaration_state(projects, dealer=dealer)
    while not state.complete:
        legal = domain.legal_declaration_actions(state, trump_suit=None)
        chosen = (
            next((action for action in legal if action.action == "declare"), legal[0])
            if declare_first and not state.actions
            else next(action for action in legal if action.action == "pass")
        )
        state, _announced = domain.apply_declaration_action(
            state,
            chosen,
            trump_suit=None,
        )
    return state


def test_mandatory_profile_keeps_one_forced_transition(root: Path) -> None:
    rules = _rules(root, "saudi_federation_2023_v2")
    domain = rules.project_domain
    projects = domain.detect(_hands(), Contract.SUN, Seat.EAST)
    state = domain.declaration_state(projects, dealer=Seat.EAST)

    legal = domain.legal_declaration_actions(state, trump_suit=None)

    assert domain.declaration_mode == "mandatory"
    assert len(legal) == 1
    assert legal[0].action == "declare"
    with pytest.raises(ValueError, match="illegal"):
        domain.apply_declaration_action(state, "pass", trump_suit=None)


def test_optional_profile_exposes_ordered_pass_and_stable_declare(root: Path) -> None:
    rules = _rules(root, "saudi_project_optional_unratified_v1")
    domain = rules.project_domain
    projects = domain.detect(_hands(), Contract.SUN, Seat.EAST)
    state = domain.declaration_state(projects, dealer=Seat.EAST)

    legal = domain.legal_declaration_actions(state, trump_suit=None)

    assert domain.declaration_mode == "optional"
    assert [action.action for action in legal] == ["pass", "declare"]
    assert legal[1].project_id == project_id_v1(
        max(
            (project for project in projects if project.owner == Seat.NORTH),
            key=lambda project: (
                project.value,
                project.category_priority,
                project.high_rank,
            ),
        )
    )
    assert set(legal[1].to_dict()) == {"actor", "action", "projectId"}
    assert all("cards" not in action.to_dict() for action in legal)


def test_optional_pass_declare_and_award_lifecycle_matches_cp_sat(root: Path) -> None:
    rules = _rules(root, "saudi_project_optional_unratified_v1")
    domain = rules.project_domain
    hands = _hands()
    projects = domain.detect(hands, Contract.SUN, Seat.EAST)
    declared_state = _complete_actions(
        domain,
        projects,
        Seat.EAST,
        declare_first=True,
    )
    domain_scored, domain_winner = domain.score_team(declared_state.projects)
    model = OrToolsProjectLifecycleModelV1()

    declared = model.solve(
        hands=hands,
        contract=Contract.SUN,
        dealer=Seat.EAST,
        trump=None,
        domain=domain,
        required_declaration_actions=declared_state.actions,
        required_declaration_actions_complete=True,
    )
    passed = model.solve(
        hands=hands,
        contract=Contract.SUN,
        dealer=Seat.EAST,
        trump=None,
        domain=domain,
        required_declaration_actions=(
            {"actor": "north", "action": "pass"},
        ),
    )

    assert declared.winning_team == domain_winner == Team.NORTH_SOUTH
    assert {
        item.project_id for item in declared.projects if item.declared
    } == {
        project_id_v1(item) for item in domain_scored if item.declared
    }
    assert declared.combined_awarded_value == domain.project_points(domain_scored)
    assert sum(item.selected for item in declared.projects if item.owner == Seat.NORTH) == 2
    assert sum(item.declared for item in declared.projects if item.owner == Seat.NORTH) == 1
    assert all(
        not item.declared for item in passed.projects if item.owner == Seat.NORTH
    )
    assert declared.model_hash == model.solve(
        hands=dict(reversed(tuple(hands.items()))),
        contract="sun",
        dealer="east",
        trump=None,
        domain=domain,
        required_declaration_actions=declared_state.actions,
        required_declaration_actions_complete=True,
    ).model_hash


def test_optional_actions_reject_wrong_order_and_project_id(root: Path) -> None:
    rules = _rules(root, "saudi_project_optional_unratified_v1")
    domain = rules.project_domain
    projects = domain.detect(_hands(), Contract.SUN, Seat.EAST)
    state = domain.declaration_state(projects, dealer=Seat.EAST)

    with pytest.raises(ValueError, match="actor mismatch"):
        domain.apply_declaration_action(
            state,
            {"actor": "west", "action": "pass"},
            trump_suit=None,
        )
    with pytest.raises(ValueError, match="unknown project"):
        OrToolsProjectLifecycleModelV1().solve(
            hands=_hands(),
            contract="sun",
            dealer="east",
            trump=None,
            domain=domain,
            required_declaration_actions=(
                {
                    "actor": "north",
                    "action": "declare",
                    "projectId": "PROJECT-0000000000000000",
                },
            ),
        )


def test_runtime_replay_consumes_explicit_optional_project_action(root: Path) -> None:
    rules = _rules(root, "saudi_project_optional_unratified_v1")
    hands = _hands()
    projects = rules.project_domain.detect(hands, Contract.SUN, Seat.EAST)
    project = max(
        (item for item in projects if item.owner == Seat.NORTH),
        key=lambda item: (item.value, item.category_priority, item.high_rank),
    )
    context = {
        "dealer": "east",
        "contract": "sun",
        "buyer": "north",
        "declarer": "north",
        "firstLeader": "north",
        "initialHands": {
            seat.name.lower(): [card.code for card in cards]
            for seat, cards in hands.items()
        },
        "currentTrick": {
            "plays": [{"seat": "north", "card": "S7", "ekka": False}]
        },
        "projectActions": [
            {
                "actor": "north",
                "action": "declare",
                "projectId": project_id_v1(project),
            }
        ],
    }

    replayed = replay_trusted_context(context, rules)

    assert replayed.current_trick is not None
    assert len(replayed.current_trick.plays) == 1
    assert any(item.declared for item in replayed.projects)
    with pytest.raises(ValueError, match="required before playing"):
        replay_trusted_context({**context, "projectActions": []}, rules)


def test_exact_trace_project_actions_are_typed_without_rewriting_legacy_hash() -> None:
    base = {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {},
    }
    legacy = ExactTracePrefixV1.build(base)
    typed = ExactTracePrefixV1.build(
        {
            **base,
            "projectActions": [
                {"actor": "north", "action": "pass"},
                {
                    "actor": "west",
                    "action": "declare",
                    "projectId": "PROJECT-0000000000000000",
                },
            ],
        }
    )

    assert "projectActions" not in legacy.to_dict()
    assert ExactTracePrefixV1.build(legacy.to_dict()).sha256 == legacy.sha256
    assert typed.to_dict()["projectActions"] == [
        {"actor": "north", "action": "pass"},
        {
            "actor": "west",
            "action": "declare",
            "projectId": "PROJECT-0000000000000000",
        },
    ]
    assert "projectActions" not in ExactTracePrefixV1Request.model_validate(
        base
    ).model_dump(exclude_none=True)
    ExactTracePrefixV1Request.model_validate(typed.to_dict())
    ExactTraceLegalActionV1Response.model_validate(
        {
            "stage": "project",
            "code": "pass",
            "actor": "north",
            "action": "pass",
        }
    )
    with pytest.raises(ValueError, match="cannot contain projectId"):
        ExactTracePrefixV1.build(
            {
                **base,
                "projectActions": [
                    {
                        "actor": "north",
                        "action": "pass",
                        "projectId": "PROJECT-0000000000000000",
                    }
                ],
            }
        )


def test_v4_authoring_catalog_exposes_typed_project_actions() -> None:
    catalog = research_recipe_v4_authoring_catalog(
        execution_capabilities={
            "combinations": [],
            "decisionSelectorSchemas": {},
        },
        policy_catalog={},
        default_policy_profiles={},
        proof_budgets={"profiles": {"late_game_exact_v1": {}}},
        authoring_defaults={},
        predicate_kinds=(),
    )
    project_actions = catalog["exactTrace"]["projectActions"]
    item = project_actions["items"]

    assert project_actions["maxItems"] == 4
    assert item["additionalProperties"] is False
    assert item["required"] == ["actor", "action"]
    assert item["properties"]["action"]["enum"] == ["pass", "declare"]
    assert item["properties"]["projectId"]["pattern"] == (
        r"^PROJECT-[0-9A-F]{16}$"
    )
    assert catalog["localizationKeys"]["values"]["declare"] == (
        "research_project_declared"
    )
    assert catalog["localizationKeys"]["exactArguments"]["projectId"] == (
        "research_scenario_argument_project_kind"
    )


def _optional_exact_prefix() -> dict:
    return {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [
            {"action": action}
            for action in ("pass", "pass", "sun", "pass", "sun")
        ],
        "escalationActions": [],
        "projectActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {},
    }


def test_optional_exact_normalization_uses_shared_legal_action_universe(
    root: Path,
) -> None:
    rules = _rules(root, "saudi_project_optional_unratified_v1")
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    prefix = _optional_exact_prefix()
    decision = normalizer.normalize(
        prefix,
        decision_point={"kind": "project", "selector": {"owner": "north"}},
    )
    assert [action["action"] for action in decision.legal_actions] == [
        "pass",
        "declare",
    ]

    declare = dict(decision.legal_actions[1])
    declare.pop("code")
    prefix["projectActions"] = [declare]
    first = normalizer.normalize(
        prefix,
        decision_point={"kind": "play", "selector": {"actingSeat": "north"}},
    )
    second = normalizer.normalize(
        prefix,
        decision_point={"kind": "play", "selector": {"actingSeat": "north"}},
    )

    assert first.to_dict() == second.to_dict()
    assert first.project_declarations_verified == 1
    assert any(
        event["type"] == "project_declared" and event["owner"] == "north"
        for event in first.state["events"]
    )
    assert all("cards" not in action for action in decision.legal_actions)


def test_optional_exact_normalization_rejects_out_of_order_action(root: Path) -> None:
    rules = _rules(root, "saudi_project_optional_unratified_v1")
    prefix = _optional_exact_prefix()
    prefix["projectActions"] = [{"actor": "west", "action": "pass"}]

    with pytest.raises(ValueError, match="actor mismatch"):
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            prefix,
            decision_point={"kind": "play", "selector": {"actingSeat": "north"}},
        )
