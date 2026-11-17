from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import baloot_platform.research.ortools_exact_state as exact_state_module
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactStateNormalizerV1,
)
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256


@pytest.mark.real_clock
@pytest.mark.parametrize("filename,complete,measured_clock", [
    ("intelligent_research.yaml", False, True),
    ("intelligent_research_validation_5s.yaml", True, True),
    ("intelligent_research_validation_5s.yaml", True, False),
])
def test_immutable_validation_budget_covers_retained_position(
    root, monkeypatch, record_property, filename, complete, measured_clock,
):
    import time

    from baloot_platform.config import load_profile_payload
    from baloot_platform.research.scenario_backend import ScenarioResearchAuthorityV1

    profile = load_profile_payload("challenge", root / "configs/challenge" / filename, root=root)
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(profile)
    budget = authority.search_budgets["profiles"]["interactive_validation_v1"]["maximumWallSeconds"]
    fixture = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text(encoding="utf-8"))
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    assert rules.sha256 == fixture["rulesHash"]
    raw = fixture["exactTracePrefix"]
    actor = raw["playActions"][24]["actor"]
    raw["playActions"] = raw["playActions"][:24]
    clock, solves = [0.0], []
    if measured_clock:
        original_solve = exact_state_module.cp_model.CpSolver.solve

        def timed_solve(solver, *args, **kwargs):
            result = original_solve(solver, *args, **kwargs)
            solves.append(solver.parameters.max_time_in_seconds)
            clock[0] += 2.148 / 7
            return result

        # Replay measured cost deterministically; every CP-SAT query still runs.
        monkeypatch.setattr(exact_state_module.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(exact_state_module.cp_model.CpSolver, "solve", timed_solve)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    started = time.perf_counter()
    if complete:
        exact = normalizer.normalize(raw, decision_point={"kind": "play", "selector": {"actingSeat": actor}},
                                     maximum_seconds=budget)
        exact.require_integrity()
        assert exact.state_hash == "371d4be12470159159122a8ab7e393e03c719b6f1d048bdc52a7cedfbff1f6e3"
        if measured_clock:
            assert len(solves) == 7 and 1 < clock[0] < budget
    else:
        with pytest.raises(ExactNormalizationStopped) as stopped:
            normalizer.normalize(raw, decision_point={"kind": "play", "selector": {"actingSeat": actor}},
                                 maximum_seconds=budget)
        assert stopped.value.partial_receipt["exhaustionReason"] == "maximum_seconds"
        assert 1 < clock[0] < 5
    record_property("profile", profile["profile"]["version"])
    record_property("budgetSeconds", budget)
    record_property("clock", "measured_replay" if measured_clock else "real")
    record_property("elapsedSeconds", time.perf_counter() - started)


def test_native_normalization_binds_new_project_visibility_without_changing_legacy(rules):
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    legacy = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    data = copy.deepcopy(rules.data)
    data["projects"]["top_rank_visibility"] = "comparison_only"
    revised = RuleBook(data)
    current = OrToolsExactStateNormalizerV1(revised, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    assert "projectRankVisibility" not in legacy.state
    assert current.state["projectRankVisibility"] == {
        "mode": "comparison_only", "categoryPriorities": dict(revised.project_domain.category_priorities)}
    assert current.state_hash != legacy.state_hash
    assert current.normalization_receipt.to_dict()["rulesHash"] == revised.sha256 != rules.sha256


def test_new_rules_native_exact_observer_hides_uncompared_project_rank(root):
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v3.yaml")
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    raw["playActions"] = raw["playActions"][:3]
    raw["projectAssertions"] = []
    raw["stateAssertions"] = {}
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        ExactTracePrefixV1.build(raw))
    exact.require_integrity()
    observed = build_canonical_observer_information_v1(
        complete_state=exact.state, acting_seat=exact.next_actor)
    assert exact.next_actor != "north"
    assert observed["publicState"]["projects"]
    assert next(row["highRank"] for row in observed["publicState"]["projects"] if row["owner"] == "north") is None
    assert all(row["highRank"] is None for row in observed["publicState"]["projects"]
               if row["owner"] != exact.next_actor)


@pytest.mark.parametrize("prefix_length", [0, 4, 32])
def test_sun_baloot_projection_preserves_zero_event_history(root, prefix_length):
    from baloot_platform.config import build_bundle
    from baloot_platform.core.enums import Contract, Seat

    rules = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").rules
    for compact in (False, True):
        model = exact_state_module.cp_model.CpModel()
        owners = tuple(card // 8 for card in range(32))
        fragment = exact_state_module.attach_project_lifecycle_fragment_v1(
            model, owner_of_card=tuple(model.new_int_var(owner, owner, f"owner_{card}")
                                      for card, owner in enumerate(owners)),
            fixed_card_owners=owners, contract=Contract.SUN, dealer=Seat.EAST,
            trump=None, domain=rules.project_domain, name="sun_projection",
            compact_fixed_states=compact)
        assert all(list(value.proto.domain) == [0, 1]
                   for value in fragment.baloot_seat)
        slots = [SimpleNamespace(actor=model.new_constant(owners[card]), card=model.new_constant(card))
                 for card in range(prefix_length)]
        def slot(step):
            return slots[step]
        auto, declared, teams = exact_state_module._native_baloot_prefix_projection(
            model=model, fragment=fragment, play_slot=slot, prefix_length=prefix_length,
            contract=Contract.SUN, trump=None, domain=rules.project_domain)
        solver = exact_state_module.cp_model.CpSolver()
        assert solver.solve(model) == exact_state_module.cp_model.OPTIMAL
        assert len(auto) == len(declared) == prefix_length
        assert [solver.value(value) for value in (*auto, *declared, *teams)] == [0] * (2 * prefix_length + 2)


def test_legacy_project_prefix_does_not_preempt_future_owner_declaration(root):
    from baloot_platform.config import build_bundle
    from baloot_platform.core.auction_ir import AuctionDomainIRV1
    from baloot_platform.core.bidding_machine import BiddingMachine
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.enums import Seat
    from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped

    rules = build_bundle(root).rules
    engine = BalootEngine(rules)
    state = engine.create_round(seed=43, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=(43 * 17 % 31) + 1)
    machine = BiddingMachine(state.dealer, state.purchase_card.suit,
                             AuctionDomainIRV1.from_rules(rules.data))
    engine.complete_deal_from_bid_result(state, machine.submit("sun"))
    engine.prepare_projects(state)
    base = {"schemaVersion": 1, "deckOrder": list(state.shuffle.deck_order),
            "dealer": "east", "auctionActions": [{"action": "sun"}],
            "escalationActions": [], "projectDeclarations": [], "playActions": []}
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    root_state = normalizer.normalize(ExactTracePrefixV1.build(base),
                                      decision_point={"kind": "play"})
    assert "SA" in {row["code"] for row in root_state.legal_actions}

    engine.play_card(state, "SA")
    one = ExactTracePrefixV1.build({**base, "playActions": [{"actor": "north", "card": "SA"}]})
    before_west = normalizer.normalize(one, decision_point={"kind": "play"})
    assert before_west.next_actor == "west"
    assert before_west.normalization_receipt.to_dict()["uniquenessSolverStatus"] == "infeasible"
    assert not any(row["type"] == "project_declared"
                   for row in before_west.state["events"])

    engine.play_card(state, "S8")
    declarations = [p.public_declaration() for p in state.projects if p.declared]
    assert any(row["owner"] == "west" for row in declarations)
    two_raw = {**base, "playActions": [*one.play_actions, {"actor": "west", "card": "S8"}],
               "projectDeclarations": declarations}
    after_west = normalizer.normalize(ExactTracePrefixV1.build(two_raw),
                                      decision_point={"kind": "play"})
    assert any(row["type"] == "project_declared" and row["owner"] == "west"
               for row in after_west.state["events"])
    with pytest.raises(PlayTailSolveStopped):
        normalizer.normalize(ExactTracePrefixV1.build({**two_raw, "projectDeclarations": []}),
                             decision_point={"kind": "play"})


@pytest.mark.parametrize("left,right", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_native_boolean_folding_preserves_solver_truth(left, right):
    cp_model = exact_state_module.cp_model
    model = cp_model.CpModel()
    x, y = model.new_bool_var("x"), model.new_bool_var("y")
    conjunction, disjunction = exact_state_module._bool_and, exact_state_module._bool_or
    folded = [conjunction(model, [], "empty_and"), disjunction(model, [], "empty_or"),
              conjunction(model, [x, x.Not(), y], "contradiction")]
    assert len(model.proto.constraints) == 0
    assert folded[1].index == folded[2].index
    outputs = [*folded, conjunction(model, [x, y.Not()], "and"),
               disjunction(model, [x.Not(), y], "or")]
    model.add(x == left)
    model.add(y == right)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert [solver.value(value) for value in outputs] == [
        1, 0, 0, int(left and not right), int(not left or right)]


def test_project_followup_choices_match_native_successor_and_reject_wrong_prefix(root):
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    parent = normalizer.normalize_play_or_project(prefix)
    snapshot = parent.normalization_receipt.to_dict()
    for action in parent.legal_actions:
        child = normalizer.append_normalized_action(prefix, parent, action["code"])
        choices = exact_state_module._certified_project_followup_rows(parent, child)
        successor = normalizer.normalize_play_or_project(child)
        assert {row["code"] for row in choices} == {row["code"] for row in successor.legal_actions}
        assert all(row["actor"] == successor.next_actor for row in choices)
    assert parent.normalization_receipt.to_dict() == snapshot
    wrong = child.to_dict()
    wrong.pop("sha256")
    wrong["matchScore"] = [1, 0]
    with pytest.raises(ValueError, match="parent differs"):
        exact_state_module._certified_project_followup_rows(parent, ExactTracePrefixV1.build(wrong))
    wrong = child.to_dict()
    wrong.pop("sha256")
    wrong["projectActions"][-1] = {"actor": parent.next_actor, "action": "declare",
                                  "projectId": "PROJECT-0000000000000000"}
    with pytest.raises(ValueError, match="uncertified declaration"):
        exact_state_module._certified_project_followup_rows(parent, ExactTracePrefixV1.build(wrong))
    with pytest.raises(ValueError, match="one certified declaration"):
        exact_state_module._certified_project_followup_rows(parent, prefix)


def test_native_project_parent_reuse_rejects_invalid_worlds_before_construction(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1
    rules, original = _early_hokum_prefix(root)
    raw = original.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exact = normalizer.normalize_play_or_project(prefix)
    wrong_prefix = ExactTracePrefixV1.build({**raw, "matchScore": [1, 0]})
    wrong = normalizer.normalize_play_or_project(wrong_prefix)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable", "targetTeam": "north_south", "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "test", "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "append_normalized_action",
                        lambda *args: pytest.fail("invalid parent reached model construction"))
    for parents, maximum, message in (
        ((), 2, "declared coverage"),
        (((exact, prefix), (exact, prefix)), 1, "declared coverage"),
        (((exact, prefix), (exact, prefix)), 2, "ownership or observation"),
        (((wrong, prefix),), 2, "ownership or observation"),
        (((wrong, wrong_prefix),), 2, "ownership or observation"),
    ):
        with pytest.raises(ValueError, match=message):
            compiler._prepare_observed_project_action(exact, prefix, request, exact.legal_actions[0]["code"],
                maximum_worlds=maximum, maximum_attempts=1000, maximum_model_bytes=30_000_000,
                maximum_seconds=30, _parents=parents)


def test_integrity_check_does_not_export_or_mutate_receipt(rules, monkeypatch):
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exact = normalizer.normalize(ExactTracePrefixV1.build(_prefix(actions=[])))
    original = copy.deepcopy(exact.normalization_receipt.payload)
    exported = exact.normalization_receipt.to_dict()
    nested_key = next(key for key, value in exported.items() if isinstance(value, dict))
    exported[nested_key]["not_in_receipt"] = True
    assert exact.normalization_receipt.payload == original
    monkeypatch.setattr(exact_state_module.OrToolsExactNormalizationReceiptV1, "to_dict",
                        lambda self: pytest.fail("integrity checking must not export a full receipt"))
    exact.require_integrity()
    assert exact.normalization_receipt.payload == original


@pytest.mark.parametrize("target", ["state", "actions", "receipt", "receipt_owner"])
def test_action_append_rejects_modified_normalization(rules, target):
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    prefix = ExactTracePrefixV1.build(_prefix(actions=[]))
    exact = normalizer.normalize(prefix)
    exact.require_integrity()
    code = exact.legal_actions[0]["code"]
    if target == "state":
        state = copy.deepcopy(dict(exact.state))
        state["matchScore"] = [1, 1]
        modified = replace(exact, state=state)
    elif target == "actions":
        modified = replace(exact, legal_actions=())
    else:
        receipt = exact.normalization_receipt.to_dict()
        receipt["prefixHash"] = "0" * 64
        if target == "receipt_owner":
            receipt.pop("sha256")
            receipt["sha256"] = canonical_sha256(receipt)
        modified = replace(exact, normalization_receipt=exact_state_module.OrToolsExactNormalizationReceiptV1(receipt))
    with pytest.raises(ValueError, match="exact normalization .* identity mismatch"):
        normalizer.append_normalized_action(prefix, modified, code)
    normalizer.append_normalized_action(prefix, exact, code)


@pytest.mark.parametrize("contract,locked", [("sun", False), ("hokum", False), ("hokum", True)])
def test_boundary_legality_uses_one_trick_with_full_tail_parity(rules, monkeypatch, contract, locked):
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Contract, Seat, Suit
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1
    cards = tuple(Card.parse(code) for code in ("SA", "S7", "HA", "H7", "DA", "D7", "CA", "C7"))
    owners = {card: Seat(index % 4) for index, card in enumerate(cards)}
    parsed = Contract.parse(contract)
    trump = Suit.HEARTS if contract == "hokum" else None
    full = OrToolsPlayTailStageV1().solve(cards=cards, ownership=owners, contract=parsed,
        trump=trump, first_leader=Seat.NORTH, locked=locked, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, maximum_time_seconds=5)
    assert full.solver_status == "optimal"
    original, calls = OrToolsPlayTailStageV1.solve, []
    def limited(self, **kwargs):
        assert kwargs["play_step_count"] == len(Seat)
        result = original(self, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", limited)
    for length in range(4):
        prefix = full.plays[:length]
        played = {action.card for action in prefix}
        state = {"hands": {seat.name.lower(): [card.code for card in cards if owners[card] == seat and card not in played]
                           for seat in Seat}, "locked": locked,
                 "currentTrick": {"leader": "north", "plays": [action.to_dict() for action in prefix]}}
        legal = exact_state_module._native_boundary_legal_play_actions(
            state, rules=rules, contract=parsed, trump=trump, maximum_seconds=5)
        assert legal == full.legal_play_actions_by_step[length]
    assert len(calls) == 4
    assert all(len(result.plays) == 4 for result in calls)


@pytest.mark.parametrize("reverse_pair", [False, True])
@pytest.mark.parametrize("prefix_length", [1, 31, 32])
def test_native_baloot_running_history_matches_card_history(root, reverse_pair, prefix_length):
    from types import SimpleNamespace
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Contract, Rank, Seat, Suit

    rules, prefix = _early_hokum_prefix(root)
    cp_model = exact_state_module.cp_model
    model = cp_model.CpModel()
    cards = [Card(card_id) for card_id in range(32)]
    king, queen = Card.of(Suit.SPADES, Rank.KING), Card.of(Suit.SPADES, Rank.QUEEN)
    pair = [king, queen] if not reverse_pair else [queen, king]
    cards = [pair[0], *(card for card in cards if card not in pair), pair[1]]
    owners = [int(Card(card_id).suit) for card_id in range(32)]
    plays = [{"actor": Seat(owners[card.id]).name.lower(), "card": card.id} for card in cards]
    raw = prefix.to_dict()
    raw.pop("sha256", None)
    raw["playActions"] = plays[:prefix_length]
    raw["projectActions"] = [{"actor": seat.name.lower(), "action": "pass"}
                             for seat in rules.project_domain.declaration_seats(Seat.EAST)]
    spec = ExactTracePrefixV1.build(raw)
    fragment = exact_state_module.attach_project_lifecycle_fragment_v1(
        model, owner_of_card=tuple(model.new_int_var(owner, owner, f"owner_{card_id}")
                                   for card_id, owner in enumerate(owners)),
        fixed_card_owners=tuple(owners), contract=Contract.HOKUM, dealer=Seat.EAST,
        trump=Suit.SPADES, domain=rules.project_domain, name="history_test",
    )
    # Test the project fragment independently; whole-play legality is covered
    # by the normalizer suite. Both orders must award only on the second card.
    slots = [SimpleNamespace(card=model.new_constant(card.id),
                             actor=model.new_constant(owners[card.id])) for card in cards]
    binding = exact_state_module._native_project_prefix_projection(
        build=SimpleNamespace(model=model, play_state_context=SimpleNamespace(play_slots=slots),
                              selected_play_indexes=tuple(slot.card for slot in slots)),
        spec=spec, fragment=fragment, dealer=Seat.EAST, contract=Contract.HOKUM,
        trump=Suit.SPADES, domain=rules.project_domain,
    )
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert [solver.value(value) for value in binding.declared_events] == ([0] * 31 + [1])[:prefix_length]
    names = [variable.name for variable in model.proto.variables]
    assert sum(name.startswith("exact_baloot_seen_") for name in names) == 2 * 4 * prefix_length
    assert not any(name.startswith("exact_baloot_prior_king_") for name in names)


def _early_hokum_prefix(root):
    from baloot_platform.core.cards import Card
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.core.enums import Seat
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    deck = [None] * 32
    for position, code in zip(layout.initial_positions[Seat.NORTH], ("SJ", "S9", "SA", "S10", "SK"), strict=True):
        deck[position] = Card.parse(code).id
    deck[layout.purchase_position] = Card.parse("SQ").id
    remaining = iter(card for card in range(32) if card not in deck)
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "dealer": "east", "roundNumber": 1, "matchScore": [0, 0],
        "deckOrder": [next(remaining) if card is None else card for card in deck],
        "auctionActions": [{"actor": "north", "action": "hokum", "trumpSuit": "spades"},
                           *({"actor": seat, "action": "pass"} for seat in ("west", "south", "east"))],
        "escalationActions": [], "projectActions": [{"actor": "north", "action": "pass"}],
        "playActions": [], "stateAssertions": {},
    })
    return rules, prefix


@pytest.mark.parametrize("certify_roots", [False, True])
def test_native_strong_hokum_project_master_diagnostic(root, record_property, monkeypatch, certify_roots, deterministic_work):
    import time
    from baloot_platform.core.enums import Team
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1, ScenarioProofBudgetV1, NativeObservedProofLimitsV1
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    compiler = OrToolsActorInformationSetCompilerV1(
        rules, native_cp_sat=True, prune_own_project_pass_worlds=True,
    )
    phase_seconds = {name: 0.0 for name in (
        "worldEnumerationSeconds", "hiddenAllocationSeconds",
        "projectRebindingSeconds", "exactNormalizationSeconds", "pathModelSeconds",
    )}

    def timed(name, original):
        def call(*args, **kwargs):
            start = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                phase_seconds[name] += time.monotonic() - start
        return call

    monkeypatch.setattr(compiler, "enumerate_graph_world_inputs", timed(
        "worldEnumerationSeconds", compiler.enumerate_graph_world_inputs))
    monkeypatch.setattr(compiler, "_enumerate_hidden_groups", timed(
        "hiddenAllocationSeconds", compiler._enumerate_hidden_groups))
    monkeypatch.setattr(compiler, "_rebind_project_actions", timed(
        "projectRebindingSeconds", compiler._rebind_project_actions))
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", timed(
        "exactNormalizationSeconds", OrToolsExactStateNormalizerV1.normalize))
    monkeypatch.setattr(compiler.graph_compiler, "_path_oracle", timed(
        "pathModelSeconds", compiler.graph_compiler._path_oracle))
    started = time.monotonic()
    measurements = {"diagnosticOnly": True, "maximumQueries": 1000, "maximumPrefixes": 1000,
                    "maximumSeconds": 30, "originalGraphNodeGate": "not_tested",
                    "compiledPlaySteps": 4, "decisionScope": "first_trick_after_north_pass"}
    frontier = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "claim": "outcome_reachable", "targetTeam": "north_south",
        "strategyClass": "pure_deterministic",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    work_started = deterministic_work.read("monotonic")
    def remaining():
        return 30 - (deterministic_work.read("monotonic") - work_started)
    try:
        exact = normalizer.normalize_play_or_project(prefix, maximum_seconds=remaining())
        parent, parent_prefix = exact, prefix
        prefix = normalizer.append_normalized_action(prefix, exact, "pass")
        exact = normalizer.normalize_play_or_project(prefix, maximum_seconds=remaining())
        fixed = compiler._public_fixed_ownership(exact)
        assert compiler._observer_project_facts_fixed(exact, fixed, ())
        assert not OrToolsActorInformationSetCompilerV1(
            rules, native_cp_sat=True,
        )._observer_project_facts_fixed(exact, fixed, ())
        other_pass = replace(exact, state={**exact.state, "events": [
            *exact.state["events"], {"type": "project_passed", "owner": "west"},
        ]})
        assert not compiler._observer_project_facts_fixed(other_pass, fixed, ())
        contexts = {}
        if certify_roots:
            with monkeypatch.context() as patch:
                patch.setattr(compiler, "enumerate_graph_world_inputs",
                    lambda *args, **kwargs: pytest.fail("invalid project root reached enumeration"))
                with pytest.raises(ValueError, match="owned legal root"):
                    compiler._prepare_observed_project_action(parent, parent_prefix, frontier,
                        "not-a-project-action", maximum_worlds=2, maximum_attempts=1000,
                        maximum_model_bytes=30_000_000, maximum_seconds=remaining())
            worlds, contexts, action_preparation = compiler._prepare_observed_project_action(
                parent, parent_prefix, frontier, "pass", maximum_worlds=2,
                maximum_attempts=1000, maximum_model_bytes=30_000_000, maximum_seconds=remaining())
            assert action_preparation["sourceStateHash"] == parent.state_hash
            assert action_preparation["action"] == "pass"
            preparation = action_preparation["preparation"]
        else:
            worlds, preparation = compiler._prepare_observed_play_worlds(exact, frontier,
                maximum_worlds=2, maximum_attempts=1000, maximum_model_bytes=30_000_000,
                maximum_seconds=remaining(), project_prefix=prefix, project_contexts=contexts)
        assert preparation["allocationModel"]["worldCount"] == 2
        assert contexts and all("projectCoverage" in row for row in preparation["models"])
        measurements.update(constructionSeconds=time.monotonic() - started, models=len(worlds),
            variables=sum(len(oracle.build.model.proto.variables) for _, oracle, _ in worlds),
            constraints=sum(len(oracle.build.model.proto.constraints) for _, oracle, _ in worlds))
        proof_started = time.monotonic()
        if certify_roots:
            for _, oracle, _ in worlds:
                monkeypatch.setattr(oracle, "enumerate_paths", lambda **kwargs:
                    pytest.fail("five-branch first-trick proof attempted exhaustive enumeration"))
            roots = compiler._solve_observed_root_actions(worlds, Team.NORTH_SOUTH,
                native_limits=NativeObservedProofLimitsV1.build({"schemaVersion": 1,
                    "maximumIterations": 1000, "maximumQueries": 1000,
                    "maximumObservations": 1000, "maximumBoundPrefixes": 1000,
                    "maximumCachedPrefixesPerBranch": 1000}),
                maximum_seconds=remaining(), project_contexts=contexts)
            outcome = compiler._observed_root_outcome(roots, frontier, maximum_seconds=remaining())
            assert outcome.completion_status == "bounded_complete"
            assert set(roots["roots"]) == {row["code"] for row in exact.legal_actions}
            assert all(root["scope"]["declarationCoverage"] for root in roots["roots"].values())
            measurements.update(status=roots["status"], receiptHash=roots["sha256"],
                rootCount=len(roots["roots"]), proofSeconds=time.monotonic() - proof_started)
            return
        result = compiler._solve_observed_worlds(worlds, Team.NORTH_SOUTH, maximum_iterations=1000,
            maximum_queries=1000, maximum_prefixes=1000, maximum_seconds=remaining(), project_contexts=contexts)
        measurements.update(status=result["status"], proofSeconds=time.monotonic() - proof_started,
            queries=len(result["queryHashes"]), observations=len(result["domains"]),
            worlds=result["worldCount"], branches=result["branchCount"], receiptHash=result["sha256"])
        assert result["status"] == "converged" and result["worldCount"] == 2
        budget = ScenarioProofBudgetV1.build("test", {
            "maximumNodes": 1000, "maximumDepth": 8, "maximumHorizonActions": 8,
            "maximumHiddenWorlds": 2, "maximumWallSeconds": 30, "maximumCpuSeconds": 30,
            "maximumCacheEntries": 1000, "maximumCacheBytes": 30_000_000,
            "cancellationCheckIntervalNodes": 1, "maximumWorkers": 1,
            "maximumLineCountNodes": 1000, "maximumStateSpace": 1000,
            "nativeObserved": {"schemaVersion": 1, "maximumIterations": 1000,
                "maximumQueries": 1000, "maximumObservations": 1000,
                "maximumBoundPrefixes": 1000, "maximumCachedPrefixesPerBranch": 1000}})
        def prepared(source, request, **kwargs):
            assert source == exact and request == frontier and kwargs["project_prefix"] == prefix
            kwargs["project_contexts"].update(contexts)
            return worlds, preparation
        def stopped_roots(supplied, target, **kwargs):
            assert supplied == worlds and target == Team.NORTH_SOUTH
            assert kwargs["project_contexts"] == contexts
            raise InformationSetCompilationStopped("cancelled")
        with monkeypatch.context() as patch:
            patch.setattr(compiler, "_prepare_observed_play_worlds", prepared)
            patch.setattr(compiler, "_solve_observed_root_actions", stopped_roots)
            with pytest.raises(InformationSetCompilationStopped) as stopped:
                compiler._prove_observed_play(exact, frontier, budget=budget,
                    maximum_seconds=remaining(), project_prefix=prefix)
            assert stopped.value.partial_receipt["evidence"]["preparation"] == preparation
    except InformationSetCompilationStopped as exc:
        measurements.update(status="incomplete", partialReceipt=exc.partial_receipt)
        raise
    finally:
        measurements["totalSeconds"] = time.monotonic() - started
        measurements.update(phase_seconds)
        record_property("nativeStrongHokumDiagnostic", json.dumps(measurements, sort_keys=True))


@pytest.mark.parametrize("real_solver", [pytest.param(False, marks=pytest.mark.real_clock), True])
def test_native_project_root_loop_preserves_shared_budget(root, monkeypatch, record_property, real_solver):
    import time
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1, ScenarioProofRequestV1
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert len(exact.legal_actions) > 1
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable", "targetTeam": "north_south", "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "test", "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    limits = NativeObservedProofLimitsV1.build({"schemaVersion": 1, "maximumIterations": 100,
        "maximumQueries": 100, "maximumObservations": 100,
        "maximumBoundPrefixes": 100, "maximumCachedPrefixesPerBranch": 100})
    if real_solver:
        measured_seconds = 900.0  # Sulaiman p99 464.237s; capped at the 900s file lane.
        stage_times = []
        active_action = [None]
        original_prepare = compiler._prepare_observed_project_action
        original_prove = compiler._solve_observed_worlds
        def timed_prepare(*args, **kwargs):
            active_action[0] = args[3]
            started = time.monotonic()
            try:
                return original_prepare(*args, **kwargs)
            finally:
                stage_times.append((active_action[0], "prepare", time.monotonic() - started))
        def timed_prove(*args, **kwargs):
            started = time.monotonic()
            try:
                return original_prove(*args, **kwargs)
            finally:
                stage_times.append((active_action[0], "proof", time.monotonic() - started))
        monkeypatch.setattr(compiler, "_prepare_observed_project_action", timed_prepare)
        monkeypatch.setattr(compiler, "_solve_observed_worlds", timed_prove)
        limits = replace(limits, maximum_iterations=1000, maximum_queries=1000,
            maximum_observations=1000, maximum_bound_prefixes=1000,
            maximum_cached_prefixes_per_branch=1000)
        from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
        budget = ScenarioProofBudgetV1.build("test", {
            "maximumNodes": 1000, "maximumDepth": 8, "maximumHorizonActions": 8,
            "maximumHiddenWorlds": 2, "maximumWallSeconds": measured_seconds, "maximumCpuSeconds": measured_seconds,
            "maximumCacheEntries": 1000, "maximumCacheBytes": 30_000_000,
            "cancellationCheckIntervalNodes": 1, "maximumWorkers": 1,
            "maximumLineCountNodes": 1000, "maximumStateSpace": 1000, "nativeObserved": limits.to_dict()})
        try:
            executed, execution = compiler._prove_observed_project(exact, prefix, request,
                budget=budget, maximum_seconds=measured_seconds)
        finally:
            record_property("nativeProjectRootStageSeconds", json.dumps(stage_times))
        assert execution["budgetHash"] == budget.sha256
        assert execution["outcome"] == executed.to_dict()
        result = execution["rootArtifact"]
        assert result["status"] == "complete"
        assert set(result["roots"]) == {row["code"] for row in exact.legal_actions}
        assert all(proof["status"] == "converged" for proof in result["roots"].values())
        assert len(result["parentStateHashes"]) == 2
        claim = compiler._certify_observed_project_actions(result, exact, request, maximum_seconds=5)
        assert claim["status"] in {"satisfied", "proof_failed"}
        assert type(claim["winningRootActionCount"]) is int
        assert claim["actionQuality"]["actualActionOptimal"] is True
        assert claim["actionQuality"]["regret"] == 0
        assert set(claim["actionQuality"]["optimalActions"]) <= set(result["legalActions"])
        outcome = compiler._observed_project_outcome(result, exact, request, maximum_seconds=5)
        assert outcome.completion_status in {"bounded_complete", "proof_failed"}
        assert outcome.graph_hash == result["sha256"]
        assert outcome.model_document()["rootActorSeat"] == exact.next_actor
        swapped = copy.deepcopy(result)
        left, right = sorted(swapped["roots"])[:2]
        swapped["roots"][left] = copy.deepcopy(swapped["roots"][right])
        swapped["sha256"] = canonical_sha256({k: v for k, v in swapped.items() if k != "sha256"})
        with pytest.raises(ValueError, match="declaration model binding"):
            compiler._certify_observed_project_actions(swapped, exact, request, maximum_seconds=5)
        altered = copy.deepcopy(result)
        altered["roots"][next(iter(altered["roots"]))]["value"] += 1
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="nested evidence checksum"):
            compiler._certify_observed_project_actions(altered, exact, request, maximum_seconds=5)
        return
    calls = []
    monkeypatch.setattr(compiler, "enumerate_graph_world_inputs", lambda *args, **kwargs: ((exact, prefix),))
    monkeypatch.setattr(compiler, "_prepare_observed_project_action", lambda *args, **kwargs:
        ((), {}, {"parents": [{"stateHash": exact.state_hash}]}))
    def solve(*args, **kwargs):
        calls.append(kwargs["native_limits"].maximum_queries)
        return {"status": "converged", "value": 1, "domains": {}, "resources": {"usage": {
            "counterexampleQueries": 2, "masterIterations": 1}}}
    monkeypatch.setattr(compiler, "_solve_observed_worlds", solve)
    result = compiler._solve_observed_project_actions(exact, prefix, request, native_limits=limits,
        maximum_worlds=2, maximum_attempts=100, maximum_model_bytes=1000, maximum_seconds=10)
    assert set(result["roots"]) == {row["code"] for row in exact.legal_actions}
    assert calls == [100 - 2 * index for index in range(len(calls))]
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_observed_project_actions(exact, prefix, request,
            native_limits=replace(limits, maximum_queries=2), maximum_worlds=2,
            maximum_attempts=100, maximum_model_bytes=1000, maximum_seconds=10)
    assert len(stopped.value.partial_receipt["evidence"]["roots"]) == 1
    assert stopped.value.partial_receipt["evidence"]["usage"] == {
        "counterexampleQueries": 2, "masterIterations": 1, "observations": 0}
    for completed_first in (False, True):
        attempts = []
        def interrupted(*args, **kwargs):
            attempts.append(kwargs["native_limits"].maximum_queries)
            if completed_first and len(attempts) == 1:
                return solve(*args, **kwargs)
            raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_worlds",
                evidence={"domains": {"observation": ["S7"]}, "resources": {"usage": {
                    "counterexampleQueries": 3, "masterIterations": 2}}})
        monkeypatch.setattr(compiler, "_solve_observed_worlds", interrupted)
        with pytest.raises(InformationSetCompilationStopped) as partial:
            compiler._solve_observed_project_actions(exact, prefix, request, native_limits=limits,
                maximum_worlds=2, maximum_attempts=100, maximum_model_bytes=1000, maximum_seconds=10)
        evidence = partial.value.partial_receipt["evidence"]
        assert partial.value.reason == "maximum_seconds"
        assert len(evidence["roots"]) == int(completed_first)
        assert evidence["usage"] == {"counterexampleQueries": 3 + 2 * completed_first,
            "masterIterations": 2 + completed_first, "observations": 1}
        assert evidence["causeReceipt"]["stage"] == "native_observed_worlds"
    monkeypatch.setattr(compiler, "_solve_observed_worlds", lambda *args, **kwargs:
        {"status": "incomplete", "value": None})
    with pytest.raises(InformationSetCompilationStopped) as incomplete:
        compiler._solve_observed_project_actions(exact, prefix, request, native_limits=limits,
            maximum_worlds=2, maximum_attempts=100, maximum_model_bytes=1000, maximum_seconds=10)
    assert incomplete.value.reason == "project_root_incomplete"
    assert len(incomplete.value.partial_receipt["evidence"]["roots"]) == 1


@pytest.mark.parametrize("contract", ["sun", "hokum"])
def test_native_all_pass_projects_allow_later_play_proof(root, contract):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    if contract == "sun":
        raw["auctionActions"] = [{"actor": "north", "action": "sun"}]
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exact = normalizer.normalize_play_or_project(prefix)
    while len(prefix.play_actions) < 4:
        action = "pass" if exact.decision_kind == "project" else exact.legal_actions[0]["code"]
        prefix = normalizer.append_normalized_action(prefix, exact, action)
        exact = normalizer.normalize_play_or_project(prefix)
    assert exact.state["projects"] and exact.state["projectWinningTeam"] is None
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "complete_state", "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True).graph_compiler
    compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=request)
    if contract == "hokum":
        terminal_raw = request.to_dict()
        terminal_raw.pop("sha256")
        terminal_raw["horizon"] = {"unit": "tricks", "value": 7}
        terminal_start = compiler._parse_start(exact, expected_rules_hash=rules.sha256,
            request=ScenarioProofRequestV1.build(terminal_raw))
        assert terminal_start.terminal_project_domain == rules.project_domain
        assert len(terminal_start.terminal_project_owners) == 32
        assert terminal_start.terminal_project_declarations == ()
    missing_pass = copy.deepcopy(exact.state)
    missing_pass["events"] = [event for event in missing_pass["events"]
        if not (event.get("type") == "project_passed" and event.get("owner") == "north")]
    with pytest.raises(ValueError, match="resolved project comparison"):
        compiler._parse_start(replace(exact, state=missing_pass, state_hash=canonical_sha256(missing_pass)),
            expected_rules_hash=rules.sha256, request=request)


def test_ratified_project_continuations_do_not_resolve_a_future_tie(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    raw = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    declarations = raw["projectActions"]
    raw.update(projectActions=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    source = normalizer.normalize_play_or_project(prefix)
    assert source.state["completedTricks"] == []
    assert source.state["projectWinningTeam"] is None
    receipt = normalizer._enumerate_project_continuations(source, prefix,
        maximum_sequences=64, maximum_seconds=10)
    assert receipt["status"] == "complete"
    assert receipt["sequences"] == [declarations]


def test_native_project_continuations_match_legal_declaration_tree(root, monkeypatch):
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    source = normalizer.normalize_play_or_project(prefix)
    receipt = normalizer._enumerate_project_continuations(source, prefix,
        maximum_sequences=64, maximum_seconds=10)
    expected = []
    queue = [(prefix, source)]
    while queue:
        branch, state = queue.pop()
        if len(branch.project_actions) == 4:
            expected.append([dict(action) for action in branch.project_actions])
            continue
        choices = state.legal_actions if state.decision_kind == "project" else state.legal_actions[:1]
        for action in choices:
            child = normalizer.append_normalized_action(branch, state, action["code"])
            queue.append((child, normalizer.normalize_play_or_project(child)))
    assert len(expected) > 1
    assert receipt["sequences"] == sorted(expected, key=canonical_sha256)
    passed = normalizer.append_normalized_action(prefix, source, "pass")
    child = normalizer.normalize_play_or_project(passed)
    conditional = normalizer._enumerate_project_continuations(child, passed,
        maximum_sequences=64, maximum_seconds=10)
    assert conditional["sequences"] == sorted(
        [sequence[1:] for sequence in expected if sequence[0]["action"] == "pass"], key=canonical_sha256)
    assert conditional["scope"]["sourceStateHash"] != receipt["scope"]["sourceStateHash"]
    original = exact_state_module._solve_cp_sat_with_cancel_v1
    calls = 0
    def unknown_after_first(*args):
        nonlocal calls
        calls += 1
        return original(*args) if calls == 1 else exact_state_module.cp_model.UNKNOWN
    monkeypatch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", unknown_after_first)
    with pytest.raises(ExactNormalizationStopped) as stopped:
        normalizer._enumerate_project_continuations(source, prefix, maximum_sequences=64, maximum_seconds=10)
    partial = stopped.value.partial_receipt
    assert partial["status"] == "incomplete" and partial["exhaustionReason"] == "unknown"
    assert len(partial["evidence"]["sequences"]) == 1
    assert partial["evidence"]["queries"][-1]["status"] == "unknown"


@pytest.mark.parametrize("opening_index", range(8))
def test_one_native_witness_projects_all_first_trick_play_checkpoints(root, monkeypatch, opening_index):
    rules, prefix = _early_hokum_prefix(root)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    opening = normalizer.normalize_play_or_project(prefix)
    assert len(opening.legal_actions) == 8
    prefix = normalizer.append_normalized_action(prefix, opening, opening.legal_actions[opening_index]["code"])
    captured = []
    original = exact_state_module._native_fixed_play_prefix

    def capture(**kwargs):
        result = original(**kwargs)
        captured.append(result.result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(exact_state_module, "_native_fixed_play_prefix", capture)
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert len(captured) == 1
    witness = captured[0]
    order = rules.project_domain.declaration_seats(exact_state_module.Seat.EAST)
    for step in range(4):
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["projectActions"] = [{"actor": seat.name.lower(), "action": "pass"}
                                 for seat in order[:step + 1]]
        raw["playActions"] = [{"actor": action.actor.name.lower(), "card": action.card.id,
                               "ekka": action.ekka} for action in witness.plays[:step]]
        exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            ExactTracePrefixV1.build(raw))
        projected = witness.at_prefix(step)
        assert projected.next_actor.name.lower() == exact.next_actor
        assert exact_state_module._legal_play_rows(projected.legal_play_actions) == exact.legal_actions
        assert {seat.name.lower(): [card.to_dict() for card in projected.remaining_hands[int(seat)]]
                for seat in exact_state_module.Seat} == exact.state["hands"]
        if step == 3:
            receipt = exact.normalization_receipt.to_dict()
            margins = receipt["nativePlayNormalization"]["witnessRawMargins"]
            derived = receipt["derivedPlay"]
            trick = derived["tricks"][0]
            winner = exact_state_module.Seat.parse(trick["winner"]).team
            assert margins["modelHash"] == derived["modelHash"]
            assert margins["values"] == [
                trick["rawPoints"] if team == winner else -trick["rawPoints"]
                for team in exact_state_module.Team
            ]




def test_native_render_rejects_short_project_evidence_before_mutation(root, monkeypatch):
    rules, prefix = _early_hokum_prefix(root)
    captured = []
    original = exact_state_module._native_fixed_play_prefix

    def capture(**kwargs):
        result = original(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(exact_state_module, "_native_fixed_play_prefix", capture)
    OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    native = captured[-1]
    state = {}
    with pytest.raises(ValueError, match="does not cover the play prefix"):
        exact_state_module._render_native_play_prefix(state, play=native.result.at_prefix(1),
            projection=native.project_projection, trump=None)
    assert state == {}


def test_terminal_escalation_certificate_reused_and_bound(root, monkeypatch):
    rules, prefix = _early_hokum_prefix(root)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    parent = normalizer.normalize_play_or_project(prefix)
    advanced = normalizer.append_normalized_action(prefix, parent, parent.legal_actions[0]["code"])
    expected = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(advanced)

    def forbidden(*args, **kwargs):
        raise LookupError("new escalation solve")

    monkeypatch.setattr(exact_state_module.OrToolsEscalationStageV1, "solve", forbidden)
    assert normalizer.normalize_play_or_project(advanced).to_dict() == expected.to_dict()
    changed = advanced.to_dict()
    changed.pop("sha256")
    changed["matchScore"] = [1, 0]
    with pytest.raises(LookupError, match="new escalation solve"):
        normalizer.normalize_play_or_project(ExactTracePrefixV1.build(changed))
    key, result, digest = normalizer._terminal_escalation_certificate
    normalizer._terminal_escalation_certificate = (key, result, "corrupt-" + digest)
    with pytest.raises(ValueError, match="escalation certificate integrity"):
        normalizer.normalize_play_or_project(advanced)


def test_first_trick_frame_reuses_across_distinct_branches(root, monkeypatch):
    rules, prefix = _early_hokum_prefix(root)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    parent = normalizer.normalize_play_or_project(prefix)
    actions = [row for row in parent.legal_actions if "+ekka" not in row["code"]][:2]
    assert len(actions) == 2
    branches = [normalizer.append_normalized_action(prefix, parent, row["code"]) for row in actions]
    first = normalizer.normalize_play_or_project(branches[0])
    expected = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(branches[1])
    frame = normalizer._first_trick_templates[1]

    def forbidden(*args, **kwargs):
        pytest.fail("same-position branch rebuilt its native frame")

    monkeypatch.setattr(exact_state_module.OrToolsPlayTailStageV1, "build_model", forbidden)
    second = normalizer.normalize_play_or_project(branches[1])
    assert second.to_dict() == expected.to_dict()
    assert first.state_hash != second.state_hash
    assert normalizer._first_trick_templates[1] is frame
    assert set(normalizer._first_trick_templates) == {0, 1}
    frame[2].model.add_bool_or([])
    with pytest.raises(ValueError, match="template integrity"):
        normalizer.normalize_play_or_project(branches[1])


@pytest.mark.parametrize("declare", [False, True])
def test_symbolic_project_checkpoints_match_independent_prefixes(root, declare):
    from baloot_platform.core.enums import Contract, Seat, Suit
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    parent = normalizer.normalize_play_or_project(prefix)
    action = next(row for row in parent.legal_actions if (row["code"] != "pass") == declare)
    child = normalizer.append_normalized_action(prefix, parent, action["code"])
    _, _, frame, fragment = parent._play_template
    build = replace(frame, model=frame.model.clone())
    checkpoints = []
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    bind = exact_state_module._native_project_action_literals
    typed = ProjectDeclarationActionV1.build(action["code"], actor=Seat.NORTH)
    literals = bind(fragment, typed)
    assert all(literal.index < 0 for literal in literals) if not declare else len(literals) == 1
    with pytest.raises(TypeError, match="typed declaration"):
        bind(fragment, action["code"])
    if declare:
        with pytest.raises(ValueError, match="unknown project"):
            bind(fragment, ProjectDeclarationActionV1(Seat.WEST, "declare", typed.project_id))
    for step in range(1, 4):
        raw = child.to_dict()
        raw.pop("sha256")
        raw["projectActions"].extend({"actor": seat.name.lower(), "action": "pass"}
                                     for seat in (Seat.WEST, Seat.SOUTH)[:step - 1])
        authored = ExactTracePrefixV1.build(raw)
        binding = exact_state_module._native_project_prefix_projection(
            build=build, spec=authored, fragment=fragment, dealer=Seat.EAST,
            contract=Contract.HOKUM, trump=Suit.SPADES, domain=rules.project_domain,
            checkpoint_id=f"future_{step}", play_prefix_length=step)
        checkpoints.append((authored, binding))
    solver = exact_state_module.cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    witness = build.solve(maximum_time_seconds=10, solver=solver)
    for step, (authored, binding) in enumerate(checkpoints, 1):
        unique = build.model.clone()
        for variable, play in zip(build.selected_play_indexes, witness.plays, strict=True):
            unique.add(unique.get_int_var_from_proto_index(variable.index) == play.card.id)
        indexes = sorted({variable.index for variable in binding.semantic_variables})
        unique.add_forbidden_assignments([unique.get_int_var_from_proto_index(index) for index in indexes],
            [[solver.value(build.model.get_int_var_from_proto_index(index)) for index in indexes]])
        checker = exact_state_module.cp_model.CpSolver()
        checker.parameters.num_search_workers = 1
        assert checker.solve(unique) == exact_state_module.cp_model.INFEASIBLE
        arguments = dict(binding=binding, solver=solver, model_hash=build.model_hash,
            prefix_length=step, uniqueness_model_hash=canonical_sha256(str(unique.proto)),
            uniqueness_solver_status="infeasible")
        _, projected = exact_state_module._extract_native_project_prefix_projection(**arguments)
        with pytest.raises(ValueError, match="does not cover"):
            exact_state_module._extract_native_project_prefix_projection(**{**arguments, "prefix_length": step - 1})
        raw = authored.to_dict()
        raw.pop("sha256")
        raw["playActions"] = [{"actor": play.actor.name.lower(), "card": play.card.id, "ekka": play.ekka}
                              for play in witness.plays[:step]]
        reference = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(raw))
        expected = reference.normalization_receipt.to_dict()["derivedProjectPrefix"]
        ignored = {"modelHash", "evidenceHash", "uniquenessModelHash", "uniquenessSolverStatus"}
        assert {key: value for key, value in projected.to_dict().items() if key not in ignored} == {
            key: value for key, value in expected.items() if key not in ignored}


@pytest.mark.parametrize("declare", [False, True])
def test_shared_project_model_keeps_prior_declaration_hidden(root, declare):
    from baloot_platform.core.enums import Contract, Seat, Suit
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    parent = normalizer.normalize_play_or_project(prefix)
    action = next(row for row in parent.legal_actions if (row["code"] != "pass") == declare)
    child = normalizer.append_normalized_action(prefix, parent, action["code"])
    _, _, frame, fragment = parent._play_template
    build = replace(frame, model=frame.model.clone())
    bindings = [exact_state_module._native_project_prefix_projection(
        build=build, spec=stage, fragment=fragment, dealer=Seat.EAST,
        contract=Contract.HOKUM, trump=Suit.SPADES, domain=rules.project_domain,
        checkpoint_id=checkpoint,
    ) for checkpoint, stage in (("before", prefix), ("after", child))]
    assert bindings[0].winning_team.name == "before:exact_project_prefix_winner"
    assert bindings[1].winning_team.name == "after:exact_project_prefix_winner"
    names = [variable.name for variable in build.model.proto.variables if variable.name]
    assert len(names) == len(set(names))
    original_size = len(build.model.proto.variables)
    with pytest.raises(ValueError, match="already exists"):
        exact_state_module._native_project_prefix_projection(
            build=build, spec=prefix, fragment=fragment, dealer=Seat.EAST,
            contract=Contract.HOKUM, trump=Suit.SPADES, domain=rules.project_domain,
            checkpoint_id="before")
    assert len(build.model.proto.variables) == original_size
    solver = exact_state_module.cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(build.model) == exact_state_module.cp_model.OPTIMAL
    before, after = bindings
    assert before.project_phase_open and not after.project_phase_open
    assert all(solver.value(value) == 0 for value in before.declared_now.values())
    assert any(solver.value(value) for value in after.declared_now.values()) == declare
    for binding in bindings:
        assert all(solver.value(value) == 0 for value in binding.awarded_now.values())
        assert all(solver.value(value) == 0 for value in binding.revealed_now.values())

    # The same frame must also switch counterfactual declarations by assumption,
    # not accumulate contradictory hard bindings or leak a prior solve's choice.
    conditional = replace(frame, model=frame.model.clone())
    foreign = exact_state_module.cp_model.CpModel().new_bool_var("foreign_guard")
    untouched = str(conditional.model.proto)
    with pytest.raises(ValueError, match="same-model Boolean guard"):
        exact_state_module._native_project_prefix_projection(
            build=conditional, spec=child, fragment=fragment, dealer=Seat.EAST,
            contract=Contract.HOKUM, trump=Suit.SPADES, domain=rules.project_domain,
            checkpoint_id="foreign", declaration_guard=foreign)
    assert str(conditional.model.proto) == untouched
    alternatives = []
    for index, row in enumerate(parent.legal_actions):
        guard = conditional.model.new_bool_var(f"choose_{index}")
        branch = normalizer.append_normalized_action(prefix, parent, row["code"])
        binding = exact_state_module._native_project_prefix_projection(
            build=conditional, spec=branch, fragment=fragment, dealer=Seat.EAST,
            contract=Contract.HOKUM, trump=Suit.SPADES, domain=rules.project_domain,
            checkpoint_id=f"choice_{index}", declaration_guard=guard)
        alternatives.append((guard, binding, row["code"]))
    assert len(alternatives) == 2
    for selected in (0, 1, 0):
        conditional.model.clear_assumptions()
        conditional.model.add_assumptions([
            guard if index == selected else guard.Not()
            for index, (guard, _, _) in enumerate(alternatives)])
        assert solver.solve(conditional.model) == exact_state_module.cp_model.OPTIMAL
        _, binding, code = alternatives[selected]
        assert any(solver.value(value) for value in binding.declared_now.values()) == (code != "pass")
        uniqueness = conditional.model.clone()
        indexes = sorted({variable.index for variable in binding.semantic_variables})
        uniqueness.add_forbidden_assignments(
            [uniqueness.get_int_var_from_proto_index(index) for index in indexes],
            [[solver.value(conditional.model.get_int_var_from_proto_index(index)) for index in indexes]])
        checker = exact_state_module.cp_model.CpSolver()
        checker.parameters.num_search_workers = 1
        assert checker.solve(uniqueness) == exact_state_module.cp_model.INFEASIBLE
        _, projected = exact_state_module._extract_native_project_prefix_projection(
            binding=binding, solver=solver, model_hash=conditional.model_hash, prefix_length=0,
            uniqueness_model_hash=canonical_sha256(str(uniqueness.proto)),
            uniqueness_solver_status="infeasible")
        independent = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            normalizer.append_normalized_action(prefix, parent, code))
        expected = independent.normalization_receipt.to_dict()["derivedProjectPrefix"]
        actual = projected.to_dict()
        for field in ("projects", "comparisonWinnerIds", "winningTeam", "balootTeams", "balootEvents", "lifecycleById"):
            assert actual[field] == expected[field]
        with pytest.raises(ValueError, match="inactive declaration checkpoint"):
            exact_state_module._extract_native_project_prefix_projection(
                binding=alternatives[1 - selected][1], solver=solver,
                model_hash="unused", prefix_length=0,
                uniqueness_model_hash="unused", uniqueness_solver_status="infeasible")
        with pytest.raises(ValueError, match="inactive declaration checkpoint"):
            exact_state_module._native_project_legal_actions(
                model=conditional.model, binding=alternatives[1 - selected][1],
                solved=solver, seed=0, maximum_seconds=1)
    conditional.model.clear_assumptions()
    conditional.model.add_assumptions([guard for guard, _, _ in alternatives])
    assert solver.solve(conditional.model) == exact_state_module.cp_model.INFEASIBLE


def test_prior_declaration_can_force_a_pass_without_fixed_zero_domains(root):
    rules, prefix = _early_hokum_prefix(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    opening = normalizer.normalize_play_or_project(prefix)
    declaration = next(row["code"] for row in opening.legal_actions if row["action"] == "declare")
    prefix = normalizer.append_normalized_action(prefix, opening, declaration)
    conditional_passes = []
    for _ in range(3):
        play = normalizer.normalize_play_or_project(prefix)
        prefix = normalizer.append_normalized_action(prefix, play, play.legal_actions[0]["code"])
        project = normalizer.normalize_play_or_project(prefix)
        raw_forced = project.normalization_receipt.to_dict()["nativePlayNormalization"]["forcedProjectPasses"]["seats"]
        if [row["code"] for row in project.legal_actions] == ["pass"] and project.next_actor not in raw_forced:
            certificate = project.normalization_receipt.to_dict()["nativePlayNormalization"]["conditionalForcedProjectPasses"]
            assert project.next_actor in certificate["seats"]
            assert certificate["keepAllFeasibleSolutionsInPresolve"] is True
            assert certificate["feasibilityOnly"] is True
            assert certificate["modelHash"] == project.normalization_receipt.to_dict()["derivedPlay"]["modelHash"]
            child_prefix = normalizer.append_normalized_action(prefix, project, "pass")
            child = normalizer.normalize_play_or_project(child_prefix)
            fresh = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(child_prefix)
            assert child.to_dict() == fresh.to_dict()
            assert child.normalization_receipt.to_dict() == fresh.normalization_receipt.to_dict()
            control = prefix.to_dict()
            control.pop("sha256")
            control["projectActions"][0] = {"actor": "north", "action": "pass"}
            alternative = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(control))
            assert any(row["action"] == "declare" for row in alternative.legal_actions)
            alternative_certificate = alternative.normalization_receipt.to_dict()["nativePlayNormalization"]["conditionalForcedProjectPasses"]
            assert alternative.next_actor not in alternative_certificate["seats"]
            conditional_passes.append(project.next_actor)
        prefix = normalizer.append_normalized_action(prefix, project, "pass")
    assert conditional_passes


def test_forced_pass_alias_matches_independent_cp_checkpoint(root, monkeypatch):
    rules, prefix = _early_hokum_prefix(root)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    opening = normalizer.normalize_play_or_project(prefix)
    prefix = normalizer.append_normalized_action(prefix, opening, opening.legal_actions[0]["code"])
    original_bind = exact_state_module._native_project_prefix_projection
    original_extract = exact_state_module._extract_native_project_prefix_projection
    reference = []
    compared = []

    def bind(**kwargs):
        binding = original_bind(**kwargs)
        following = kwargs["spec"].to_dict()
        following.pop("sha256")
        following["projectActions"].append({"actor": "west", "action": "pass"})
        reference.append(original_bind(**{**kwargs, "spec": ExactTracePrefixV1.build(following),
                                          "checkpoint_id": "independent_after_pass"}))
        return binding

    def extract(**kwargs):
        actual = original_extract(**kwargs)
        if not kwargs["binding"].project_phase_open:
            expected = original_extract(**{**kwargs, "binding": reference[0]})
            assert [item.to_dict() for item in actual] == [item.to_dict() for item in expected]
            compared.append(True)
        return actual

    monkeypatch.setattr(exact_state_module, "_native_project_prefix_projection", bind)
    monkeypatch.setattr(exact_state_module, "_extract_native_project_prefix_projection", extract)
    normalized = normalizer.normalize_play_or_project(prefix)
    assert [row["code"] for row in normalized.legal_actions] == ["pass"]
    assert compared == [True]


def test_forced_project_pass_preserves_native_play_facts(root, monkeypatch):
    rules, prefix = _early_hokum_prefix(root)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    play = normalizer.normalize_play_or_project(prefix)
    prefix = normalizer.append_normalized_action(prefix, play, play.legal_actions[0]["code"])
    parent = normalizer.normalize_play_or_project(prefix)
    assert parent.decision_kind == "project"
    assert [action["code"] for action in parent.legal_actions] == ["pass"]
    receipt = parent.normalization_receipt.to_dict()
    forced = receipt["nativePlayNormalization"]["forcedProjectPasses"]
    assert forced["modelHash"] == receipt["derivedPlay"]["modelHash"]
    assert "west" in forced["seats"]
    assert "north" not in forced["seats"]  # Its eligible project is not a forced pass.
    child_prefix = normalizer.append_normalized_action(prefix, parent, "pass")
    assert len(normalizer._paired_pass_cache) == 1
    with monkeypatch.context() as patch:
        def repeated_native_solve(*args, **kwargs):
            pytest.fail("jointly certified forced-pass child was solved again")
        def repeated_native_clone(*args, **kwargs):
            pytest.fail("jointly certified forced-pass child cloned its model again")
        patch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", repeated_native_solve)
        patch.setattr(exact_state_module.cp_model.CpModel, "clone", repeated_native_clone)
        child = normalizer.normalize_play_or_project(child_prefix)
    key, (digest, paired) = next(iter(normalizer._paired_pass_cache.items()))
    normalizer._paired_pass_cache[key] = ("corrupt", paired)
    with pytest.raises(ValueError, match="paired-pass certificate integrity"):
        normalizer.normalize_play_or_project(child_prefix)
    normalizer._paired_pass_cache[key] = (digest, paired)
    with pytest.raises(ExactNormalizationStopped):
        normalizer.normalize_play_or_project(child_prefix, cancel_check=lambda: True)
    fresh_child = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(child_prefix)
    assert child.to_dict() == fresh_child.to_dict()
    assert child.normalization_receipt.to_dict() == fresh_child.normalization_receipt.to_dict()
    nullable_pass = child_prefix.to_dict()
    nullable_pass.pop("sha256")
    nullable_pass["projectActions"][-1]["projectId"] = None
    nullable_child = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        ExactTracePrefixV1.build(nullable_pass))
    assert nullable_child.next_actor == child.next_actor
    assert nullable_child.legal_actions == child.legal_actions
    assert nullable_child.normalization_receipt.to_dict()["derivedProjectPrefix"] == child.normalization_receipt.to_dict()["derivedProjectPrefix"]
    assert nullable_child.to_dict() == normalizer.normalize_play_or_project(ExactTracePrefixV1.build(nullable_pass)).to_dict()
    def repeated_deal(*args, **kwargs):
        pytest.fail("unchanged deal/auction certificate was solved again")
    with monkeypatch.context() as patch:
        patch.setattr(exact_state_module.OrToolsScenarioModelV1, "solve", repeated_deal)
        patch.setattr(exact_state_module, "_native_role_projection", repeated_deal)
        assert normalizer.normalize_play_or_project(child_prefix).to_dict() == fresh_child.to_dict()
    assert parent._play_template is not None
    original_model_hash = parent._play_template[2].model_hash
    def rebuilt(*args, **kwargs):
        pytest.fail("matching parent play template was rebuilt")
    with monkeypatch.context() as patch:
        patch.setattr(exact_state_module.OrToolsPlayTailStageV1, "build_model", rebuilt)
        patch.setattr(exact_state_module, "attach_project_lifecycle_fragment_v1", rebuilt)
        for _ in range(2):
            reused = normalizer.normalize(child_prefix, decision_point={"kind": "play"},
                                          _play_template=parent._play_template)
            assert reused.to_dict() == child.to_dict()
            assert reused._play_template is None  # The next play changes the prefix.
            assert reused.normalization_receipt.to_dict() == child.normalization_receipt.to_dict()
            assert parent._play_template[2].model_hash == original_model_hash
    original_build = exact_state_module.OrToolsPlayTailStageV1.build_model
    rebuilds = []
    def counted_build(*args, **kwargs):
        rebuilds.append(True)
        return original_build(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(exact_state_module.OrToolsPlayTailStageV1, "build_model", counted_build)
        mismatch = ("different-input", *parent._play_template[1:])
        rebuilt_child = normalizer.normalize(child_prefix, decision_point={"kind": "play"}, _play_template=mismatch)
        assert rebuilt_child.to_dict() == child.to_dict()
        assert len(rebuilds) == 1
    parent._play_template[2].model.add_bool_or([])
    with pytest.raises(ValueError, match="native play template integrity mismatch"):
        normalizer.normalize(child_prefix, decision_point={"kind": "play"},
                             _play_template=parent._play_template)
    parent.require_integrity()
    child.require_integrity()
    assert child.decision_kind == "play"
    assert parent.next_actor == child.next_actor
    assert {key for key in parent.state if parent.state[key] != child.state[key]} == {"seed", "shuffle", "events"}
    assert parent.normalization_receipt.to_dict()["nativePlayNormalization"]["legalPlayActions"] == child.normalization_receipt.to_dict()["nativePlayNormalization"]["legalPlayActions"]
    parent_events = copy.deepcopy(parent.state["events"])
    for event in parent_events:
        if event["type"] in {"round_created", "deck_shuffled"}:
            event["seed"] = child.state["seed"]
    assert child.state["events"] == [*parent_events, {
        "sequence": len(parent_events), "type": "project_passed", "owner": parent.next_actor,
    }]
    key, values, digest = normalizer._deal_prefix_certificate
    normalizer._deal_prefix_certificate = ("different-deal", values, digest)
    original_solve = exact_state_module.OrToolsScenarioModelV1.solve
    solves = []
    def counted_solve(*args, **kwargs):
        solves.append(True)
        return original_solve(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(exact_state_module.OrToolsScenarioModelV1, "solve", counted_solve)
        assert normalizer.normalize_play_or_project(child_prefix).to_dict() == fresh_child.to_dict()
        assert len(solves) == 2
    normalizer._deal_prefix_certificate = (key, values, "corrupted")
    with pytest.raises(ValueError, match="native deal certificate integrity mismatch"):
        normalizer.normalize_play_or_project(child_prefix)


def test_native_early_hokum_joint_legality_matches_boundary(root):
    from baloot_platform.core.enums import Contract, Suit
    rules, prefix = _early_hokum_prefix(root)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"})
    joint = normalized.normalization_receipt.to_dict()["nativePlayNormalization"]["legalPlayActions"]
    boundary = exact_state_module._native_boundary_legal_play_actions(
        normalized.state, rules=rules, contract=Contract.HOKUM,
        trump=Suit.SPADES, maximum_seconds=10,
    )
    assert {(row["card"], row["ekka"]) for row in joint} == {
        (action.card.code, action.ekka) for action in boundary
    }
    assert {(row["card"], row["ekka"]) for row in joint} == {
        (row["code"].removesuffix("+ekka"), row["ekka"]) for row in normalized.legal_actions
    }


@pytest.mark.parametrize("stage", ["role", "play"])
def test_native_uniqueness_unknown_preserves_model_receipt(root, rules, monkeypatch, stage):
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    original_hash = exact_state_module._cp_model_hash
    original_solve = exact_state_module._solve_cp_sat_with_cancel_v1
    selected = {}
    def record_model(model, *, authority):
        value = original_hash(model, authority=authority)
        if ((stage == "role" and authority.get("purpose") == "semantic_uniqueness")
                or (stage == "play" and authority.get("stage") == "exact_play_semantic_uniqueness")):
            selected.update(model=model, sha256=value)
        return value
    def stop(solver, model, cancel_check):
        if model is selected.get("model"):
            return exact_state_module.cp_model.UNKNOWN
        return original_solve(solver, model, cancel_check)
    monkeypatch.setattr(exact_state_module, "_cp_model_hash", record_model)
    monkeypatch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", stop)
    with pytest.raises(ExactNormalizationStopped) as stopped:
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            fixture["exactTracePrefix"], decision_point={"kind": "play"})
    receipt = stopped.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == "unknown"
    assert receipt["evidence"]["sourceStage"] == f"{stage}_semantic_uniqueness"
    assert receipt["evidence"]["modelHash"] == selected["sha256"]
    assert receipt["evidence"]["sourceModelHash"] != selected["sha256"]
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.parametrize("operation", ["build_model", "solve"])
@pytest.mark.parametrize("status", ["unknown", "cancelled", "infeasible", "invalid_input"])
def test_native_prefix_preserves_solver_stop_receipt(root, monkeypatch, status, operation):
    rules, prefix = _early_hokum_prefix(root)
    failure = (ValueError("invalid authored prefix") if status == "invalid_input" else
        exact_state_module.PlayTailSolveStopped(status, evidence={"branches": 7}))
    def stopped(*args, **kwargs):
        raise failure
    owner = (exact_state_module.OrToolsPlayTailStageV1 if operation == "build_model" else
             exact_state_module.OrToolsPlayTailBuildV1)
    monkeypatch.setattr(owner, operation, stopped)
    expected = (ValueError if status == "invalid_input" else
        exact_state_module.PlayTailSolveStopped if status == "infeasible" else ExactNormalizationStopped)
    with pytest.raises(expected) as caught:
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    if status in {"infeasible", "invalid_input"}:
        assert caught.value is failure
    else:
        receipt = caught.value.partial_receipt
        assert receipt["exhaustionReason"] == status
        assert receipt["evidence"]["sourceStage"] == "exact_play_prefix"
        assert receipt["evidence"]["causeReceipt"] == failure.partial_receipt
        assert caught.value.__cause__ is failure


@pytest.mark.parametrize("status", ["unknown", "cancelled", "infeasible"])
def test_boundary_legality_preserves_solver_exhaustion(rules, monkeypatch, status):
    from baloot_platform.core.enums import Contract, Seat
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1, PlayTailSolveStopped
    failure = PlayTailSolveStopped(status, evidence={"branches": 7})
    def stopped(*args, **kwargs):
        raise failure
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", stopped)
    state = {"hands": {seat.name.lower(): [int(seat)] for seat in Seat},
             "currentTrick": {"leader": "north", "plays": []}}
    with pytest.raises(PlayTailSolveStopped if status == "infeasible" else ExactNormalizationStopped) as caught:
        exact_state_module._native_boundary_legal_play_actions(
            state, rules=rules, contract=Contract.SUN, trump=None, maximum_seconds=5)
    if status == "infeasible":
        assert caught.value is failure
    else:
        receipt = caught.value.partial_receipt
        assert receipt["exhaustionReason"] == status
        assert receipt["evidence"]["sourceStage"] == "boundary_legal_actions"
        assert receipt["evidence"]["causeReceipt"] == failure.partial_receipt


@pytest.mark.parametrize("chosen", [None, 0, 1])
@pytest.mark.parametrize("forced", [False, True])
def test_project_legal_actions_reuse_only_the_primary_solver_witness(monkeypatch, chosen, forced):
    from types import SimpleNamespace
    from baloot_platform.core.enums import Seat

    cp = exact_state_module.cp_model
    model = cp.CpModel()
    declared = [model.new_bool_var(f"declare_{index}") for index in range(2)]
    declarable = model.new_bool_var("declarable")
    model.add(declarable == 1)
    model.add_at_most_one(declared)
    if forced:
        model.add_assumptions([item.Not() for item in declared] if chosen is None else [declared[chosen]])
    primary = model.clone()
    if chosen is None:
        primary.add(sum(primary.get_bool_var_from_proto_index(item.index) for item in declared) == 0)
    else:
        primary.add(primary.get_bool_var_from_proto_index(declared[chosen].index) == 1)
    solver = cp.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(primary) == cp.OPTIMAL
    projects = tuple(SimpleNamespace(owner=Seat.NORTH, declared=item, declarable=declarable,
                                     project_id=f"project-{index}") for index, item in enumerate(declared))
    binding = SimpleNamespace(project_phase_open=True, reached_action_count=0, declaration_guard=None,
                              declaration_seats=(Seat.NORTH,), fragment=SimpleNamespace(projects=projects))
    calls = []
    probes = []
    source_proto = str(model.proto)
    original = exact_state_module._solve_cp_sat_with_cancel_v1

    def counted(*args):
        calls.append(1)
        probes.append(args[1])
        return original(*args)

    monkeypatch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", counted)
    actions = exact_state_module._native_project_legal_actions(
        model=model, binding=binding, solved=solver, seed=0, maximum_seconds=5,
    )
    expected = [
        ("pass", None), ("declare", "project-0"), ("declare", "project-1"),
    ]
    if forced:
        expected = [expected[0 if chosen is None else chosen + 1]]
    assert [(action.action, action.project_id) for action in actions] == expected
    assert len(calls) == 2
    assert probes[0] is probes[1]
    assert probes[0] is not model
    assert str(model.proto) == source_proto
    with pytest.raises(ExactNormalizationStopped):
        exact_state_module._native_project_legal_actions(
            model=model, binding=binding, solved=solver, seed=0, maximum_seconds=5,
            cancel_check=lambda: True,
        )
    assert len(calls) == 2


    monkeypatch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", lambda *args: cp.UNKNOWN)
    with pytest.raises(ExactNormalizationStopped) as stopped:
        exact_state_module._native_project_legal_actions(
            model=model, binding=binding, solved=solver, seed=0, maximum_seconds=5,
        )
    receipt = stopped.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == "unknown"
    assert receipt["evidence"]["sourceStage"] == "project_legal_actions"
    assert receipt["evidence"]["actor"] == "north"
    assert len(receipt["evidence"]["modelHash"]) == 64
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


def _prefix(*, actions: list[dict] | None = None) -> dict:
    return {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": actions or [],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {},
    }


def _terminal_sun_actions(*, with_actors: bool = False) -> list[dict]:
    actions = ["pass", "pass", "sun", "pass", "sun"]
    actors = ["north", "west", "south", "west", "north"]
    return [
        {"action": action, **({"actor": actor} if with_actors else {})}
        for action, actor in zip(actions, actors, strict=True)
    ]


@pytest.mark.parametrize("terminal", [False, True])
def test_native_normalization_does_not_enumerate_auction_tables(rules, monkeypatch, terminal):
    import baloot_platform.research.ortools_scenario_model as auction_module
    prefix = _prefix(actions=_terminal_sun_actions() if terminal else [{"action": "pass"}])
    def forbidden(*args, **kwargs):
        raise AssertionError("native normalization reached the legacy auction table")
    for name in ("_enumerate_auction", "_advance_auction", "_auction_legal_choices"):
        monkeypatch.setattr(auction_module, name, forbidden)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize(prefix, maximum_seconds=15)
    assert normalized.next_actor == (None if terminal else "west")
    assert normalizer.normalize(prefix, maximum_seconds=15).to_dict() == normalized.to_dict()
    receipt = normalized.normalization_receipt.to_dict()
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})


@pytest.mark.parametrize("seconds", [True, None, "10", 0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("method", ["normalize", "preview_legal_next_actions"])
def test_exact_normalization_rejects_invalid_wall_limits(rules, seconds, method):
    with pytest.raises(ValueError, match="finite and positive"):
        getattr(OrToolsExactStateNormalizerV1(rules), method)(_prefix(), maximum_seconds=seconds)


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["cancelled", "budget"])
@pytest.mark.parametrize("method", ["normalize", "preview_legal_next_actions"])
def test_exact_normalization_stops_before_uniqueness_after_exhaustion(rules, monkeypatch, reason, method):
    clock = [0.0]
    cancelled = [False]
    solves = []
    original = exact_state_module.OrToolsScenarioModelV1.solve

    def solve(model, **kwargs):
        solves.append(kwargs["maximum_seconds"])
        result = original(model, **kwargs)
        clock[0] = 10.0 if reason == "budget" else 0.0
        cancelled[0] = reason == "cancelled"
        return result

    monkeypatch.setattr(exact_state_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(exact_state_module.OrToolsScenarioModelV1, "solve", solve)
    with pytest.raises(ExactNormalizationStopped, match="cancelled" if reason == "cancelled" else "within budget") as stopped:
        getattr(OrToolsExactStateNormalizerV1(rules), method)(
            _prefix(), maximum_seconds=10, cancel_check=lambda: cancelled[0],
        )
    assert solves == [10.0]
    receipt = stopped.value.partial_receipt
    assert receipt["exhaustionReason"] == ("cancelled" if reason == "cancelled" else "maximum_seconds")
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})


@pytest.mark.parametrize("stage", ["deal_auction_witness", "deal_auction_uniqueness"])
def test_native_deal_auction_unknown_is_incomplete_not_invalid(rules, monkeypatch, stage):
    original = exact_state_module.OrToolsScenarioModelV1.solve
    calls = [0]

    def solve(model, **kwargs):
        calls[0] += 1
        if calls[0] == (1 if stage == "deal_auction_witness" else 2):
            return SimpleNamespace(status="not_found_within_budget", solver_status="unknown")
        return original(model, **kwargs)

    monkeypatch.setattr(exact_state_module.OrToolsScenarioModelV1, "solve", solve)
    with pytest.raises(ExactNormalizationStopped) as stopped:
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(_prefix())
    receipt = stopped.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == "unknown"
    assert receipt["evidence"]["sourceStage"] == stage
    assert calls[0] == (1 if stage == "deal_auction_witness" else 2)


@pytest.mark.parametrize("method", ["normalize", "preview_legal_next_actions"])
def test_native_exact_normalization_shares_clock_across_every_solve(root, rules, monkeypatch, method):
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    clock = [0.0]
    budgets = []
    original = exact_state_module.cp_model.CpSolver.solve

    def solve(solver, *args, **kwargs):
        budgets.append(solver.parameters.max_time_in_seconds)
        result = original(solver, *args, **kwargs)
        clock[0] += 1.0
        return result

    monkeypatch.setattr(exact_state_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(exact_state_module.cp_model.CpSolver, "solve", solve)
    result = getattr(OrToolsExactStateNormalizerV1(rules, native_cp_sat=True), method)(
        fixture["exactTracePrefix"], maximum_seconds=100,
        **({"decision_point": {"kind": "play"}} if method == "normalize" else {}),
    )
    assert (result.phase == "playing") if method == "normalize" else (result["currentStage"] == "play")
    assert len(budgets) == 7  # No preliminary hint solve; all uniqueness stages remain.
    assert budgets == [100.0 - index for index in range(len(budgets))]


@pytest.mark.parametrize("method", ["normalize", "preview_legal_next_actions"])
def test_native_escalation_unknown_is_incomplete_not_invalid(rules, monkeypatch, method):
    monkeypatch.setattr(exact_state_module.OrToolsEscalationStageV1, "solve",
        lambda *args, **kwargs: SimpleNamespace(
            status="not_found_within_budget", solver_status="unknown", observation=None))
    kwargs = {"decision_point": {"kind": "play"}} if method == "normalize" else {}
    with pytest.raises(ExactNormalizationStopped) as stopped:
        getattr(OrToolsExactStateNormalizerV1(rules, native_cp_sat=True), method)(
            _prefix(actions=[{"action": "sun"}]), **kwargs)
    assert stopped.value.partial_receipt["exhaustionReason"] == "unknown"
    assert stopped.value.partial_receipt["evidence"]["sourceStage"] == "exact_escalation"


def test_cp_exact_normalizes_fixed_deck_and_open_auction_deterministically(rules):
    prefix = _prefix()
    normalizer = OrToolsExactStateNormalizerV1(rules)

    first = normalizer.normalize(prefix)
    second = normalizer.normalize(copy.deepcopy(prefix))

    assert first.to_dict() == second.to_dict()
    assert (
        first.normalization_receipt.to_dict() == second.normalization_receipt.to_dict()
    )
    assert first.phase == "bidding"
    assert first.decision_kind == "bid"
    assert first.next_actor == "north"
    assert first.state["shuffle"]["deckOrder"] == list(range(32))
    assert first.state["purchaseCard"]["id"] == 20
    assert first.state["purchaseCard"]["code"] == "DJ"
    assert first.state["buyer"] is None
    assert first.state["contract"] is None
    assert first.state["trumpSuit"] is None
    assert [
        len(first.state["hands"][seat]) for seat in ("north", "west", "south", "east")
    ] == [5, 5, 5, 5]
    assert all(
        first.state["initialHands"][seat] == []
        for seat in ("north", "west", "south", "east")
    )
    assert {row["code"] for row in first.legal_actions} == {
        "pass",
        "sun",
        "hokum:diamonds",
    }
    receipt = first.normalization_receipt.to_dict()
    assert receipt["backendId"] == "ortools_cp_sat_v2"
    assert receipt["solverStatus"] == "optimal"
    assert receipt["uniquenessStatus"] == "proven_unique"
    assert receipt["uniquenessSolverStatus"] == "infeasible"
    assert receipt["derivedAuction"] == {
        "prefixTerminal": False,
        "buyer": None,
        "contract": None,
        "trumpSuit": None,
        "purchaseRound": None,
        "currentActor": "north",
        "currentRound": 1,
        "legalNextActions": ["hokum", "pass", "sun"],
        "legalNextChoices": ["hokum:diamonds", "pass", "sun"],
        "traceHash": receipt["derivedAuction"]["traceHash"],
    }


def test_cp_exact_open_auction_preserves_legacy_neutral_wire(rules):
    prefix = _prefix()
    cp_normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    legacy_normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )

    assert cp_normalized.to_dict() == legacy_normalized.to_dict()


def test_cp_exact_terminal_auction_derives_purchase_roles_and_complete_hands(rules):
    normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        _prefix(actions=_terminal_sun_actions())
    )

    assert normalized.phase == "final_dealt"
    assert normalized.next_actor is None
    assert normalized.legal_actions == ()
    assert normalized.state["buyer"] == "north"
    assert normalized.state["contractOwner"] == "north"
    assert normalized.state["purchaseRecipient"] == "north"
    assert normalized.state["declarer"] == "north"
    assert normalized.state["contract"] == "sun"
    assert normalized.state["trumpSuit"] is None
    assert (
        normalized.normalization_receipt.to_dict()["derivedAuction"]["purchaseRound"]
        == 1
    )
    assert (
        normalized.normalization_receipt.to_dict()["derivedAuction"]["prefixTerminal"]
        is True
    )
    assert normalized.state["firstLeader"] == "north"
    assert [
        len(normalized.state["hands"][seat])
        for seat in ("north", "west", "south", "east")
    ] == [8, 8, 8, 8]
    purchase_id = normalized.state["purchaseCard"]["id"]
    assert purchase_id == 20
    assert any(card["id"] == purchase_id for card in normalized.state["hands"]["north"])
    assert all(
        not any(card["id"] == purchase_id for card in normalized.state["hands"][seat])
        for seat in ("west", "south", "east")
    )
    all_cards = [
        card["id"]
        for seat in ("north", "west", "south", "east")
        for card in normalized.state["hands"][seat]
    ]
    assert sorted(all_cards) == list(range(32))
    assert normalized.state["hands"] == normalized.state["initialHands"]


def test_cp_exact_derives_first_round_hokum_trump_from_purchase_card(rules):
    normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        _prefix(
            actions=[
                {"action": "hokum"},
                {"action": "pass"},
                {"action": "pass"},
                {"action": "pass"},
            ]
        )
    )
    auction = normalized.normalization_receipt.to_dict()["derivedAuction"]

    assert normalized.state["purchaseCard"]["suit"] == "diamonds"
    assert normalized.state["contract"] == "hokum"
    assert normalized.state["trumpSuit"] == "diamonds"
    assert auction["purchaseRound"] == 1


def test_cp_exact_optional_actor_is_validation_only_for_identity(rules):
    normalizer = OrToolsExactStateNormalizerV1(rules)
    derived = normalizer.normalize(_prefix(actions=_terminal_sun_actions()))
    asserted = normalizer.normalize(
        _prefix(actions=_terminal_sun_actions(with_actors=True))
    )

    assert derived.prefix_hash == asserted.prefix_hash
    assert derived.exact_state_id == asserted.exact_state_id
    assert derived.to_dict() == asserted.to_dict()


def test_cp_exact_rejects_wrong_explicit_auction_actor(rules):
    with pytest.raises(ValueError, match="deal or auction prefix is infeasible"):
        OrToolsExactStateNormalizerV1(rules).normalize(
            _prefix(actions=[{"actor": "west", "action": "pass"}])
        )


def test_cp_exact_state_assertions_are_strict_and_do_not_change_identity(rules):
    normalizer = OrToolsExactStateNormalizerV1(rules)
    plain = normalizer.normalize(_prefix())
    asserted_prefix = _prefix()
    asserted_prefix["stateAssertions"] = {
        "phase": "bidding",
        "bidding.actor": "north",
        "purchaseCard.id": 20,
    }
    asserted = normalizer.normalize(asserted_prefix)

    assert asserted.exact_state_id == plain.exact_state_id
    assert asserted.prefix_hash == plain.prefix_hash

    bad = _prefix()
    bad["stateAssertions"] = {"roundNumber": 1.0}
    with pytest.raises(ValueError, match="assertion mismatch at roundNumber"):
        normalizer.normalize(bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "projectDeclarations",
            [
                {
                    "owner": "north",
                    "kind": "sequence_three",
                    "value": 20,
                    "highRank": 4,
                    "declarationOrder": 1,
                }
            ],
        ),
        ("projectAssertions", [{"state": "eligible"}]),
        ("playActions", [{"card": 0}]),
    ],
)
def test_cp_exact_rejects_project_or_play_input_at_bid_decision(rules, field, value):
    prefix = _prefix()
    prefix[field] = value
    with pytest.raises(ValueError, match="bid decision trace cannot contain"):
        OrToolsExactStateNormalizerV1(rules).normalize(prefix)


def test_cp_exact_rejects_completed_match_score(rules):
    prefix = _prefix()
    prefix["matchScore"] = [152, 100]
    with pytest.raises(ValueError, match="completed match"):
        OrToolsExactStateNormalizerV1(rules).normalize(prefix)


def test_cp_exact_rejects_unintegrated_decision_until_its_cp_stage_exists(rules):
    with pytest.raises(ValueError, match="decision kind is unsupported"):
        OrToolsExactStateNormalizerV1(rules).normalize(
            _prefix(),
            decision_point={"kind": "deal", "selector": {}},
        )


@pytest.mark.parametrize(
    ("actions", "acting_seat"),
    [
        ([], "west"),
        ([{"actor": "west", "action": "double", "mode": "open"}], "north"),
    ],
)
def test_cp_exact_escalation_prefix_and_legal_actions_match_legacy(
    rules,
    actions,
    acting_seat,
):
    prefix = _prefix(
        actions=[
            {"action": "hokum"},
            {"action": "pass"},
            {"action": "pass"},
            {"action": "pass"},
        ]
    )
    prefix["escalationActions"] = actions
    decision = {
        "kind": "escalation",
        "selector": {"actingSeat": acting_seat},
    }

    cp_normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision,
    )
    legacy_normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision,
    )

    assert cp_normalized.to_dict() == legacy_normalized.to_dict()
    receipt = cp_normalized.normalization_receipt.to_dict()
    assert receipt["derivedEscalation"]["observation"]["actions"] == [
        {
            "actor": item["actor"],
            "team": "west_east",
            "action": item["action"],
            "requestedMode": item.get("mode"),
            "resolvedMode": item.get("mode") or "open",
        }
        for item in actions
    ]
    assert receipt["uniquenessStages"]["escalation"] == (
        "domain_ir_deterministic_transition"
    )


def test_cp_exact_escalation_requires_terminal_auction_and_actor(rules):
    normalizer = OrToolsExactStateNormalizerV1(rules)
    with pytest.raises(ValueError, match="terminal purchased auction"):
        normalizer.normalize(
            _prefix(),
            decision_point={
                "kind": "escalation",
                "selector": {"actingSeat": "west"},
            },
        )
    prefix = _prefix(actions=_terminal_sun_actions())
    with pytest.raises(ValueError, match="requires selector.actingSeat"):
        normalizer.normalize(
            prefix,
            decision_point={"kind": "escalation", "selector": {}},
        )


def _flagship_prefix(root: Path) -> dict:
    return json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )["exactTracePrefix"]


def test_cp_exact_flagship_play_projects_and_identity_match_frozen_legacy(
    rules,
    root: Path,
):
    prefix = _flagship_prefix(root)
    decision = {"kind": "play", "selector": {}}

    cp_normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision,
    )
    legacy_normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision,
    )

    assert cp_normalized.to_dict() == legacy_normalized.to_dict()
    assert cp_normalized.exact_state_id == "STATE-C1540EA736F0D5920BA3"
    assert cp_normalized.project_declarations_verified == 1
    assert cp_normalized.project_assertions_verified == 1
    assert cp_normalized.state["phase"] == "playing"
    assert len(cp_normalized.state["completedTricks"]) == 5
    assert len(cp_normalized.state["currentTrick"]["plays"]) == 3
    assert cp_normalized.next_actor == "west"
    receipt = cp_normalized.normalization_receipt.to_dict()
    assert receipt["uniquenessStages"]["projects"] == ("cp_sat_fixed_hands_optimal")
    assert receipt["uniquenessStages"]["play"] == ("cp_sat_fixed_prefix_optimal")


def test_cp_exact_flagship_state_stops_at_prefix_and_receipt_is_observer_safe(
    rules,
    root: Path,
):
    prefix = _flagship_prefix(root)
    normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point={"kind": "play", "selector": {}},
    )

    public_events = normalized.state["events"]
    assert sum(event["type"] == "card_played" for event in public_events) == len(
        prefix["playActions"]
    )
    assert len(normalized.state["completedTricks"]) == 5
    assert len(normalized.state["currentTrick"]["plays"]) == 3

    receipt = normalized.normalization_receipt.to_dict()
    # Exact validation compiles only through the next completed trick; the
    # authored 23-action prefix therefore needs one CP-SAT continuation action.
    assert len(receipt["derivedPlay"]["plays"]) == 24
    assert [
        {"actor": item["actor"], "card": item["card"]}
        for item in receipt["derivedPlay"]["plays"][: len(prefix["playActions"])]
    ] == prefix["playActions"]
    observer = receipt["observerInformation"]
    assert observer["actingSeat"] == normalized.next_actor == "west"
    assert len(observer["observerInformationHash"]) == 64
    hands = observer["publicState"]["hands"]
    assert isinstance(hands["west"], list)
    for seat in ("north", "south", "east"):
        assert hands[seat] == {"count": len(normalized.state["hands"][seat])}
    encoded_observer = json.dumps(observer, ensure_ascii=False, sort_keys=True)
    for private_key in ("initialHands", "shuffle", "dealTrace"):
        assert private_key not in encoded_observer


def test_cp_exact_play_actors_are_optional_validation_aids(rules, root: Path):
    authored = _flagship_prefix(root)
    actor_free = copy.deepcopy(authored)
    for action in actor_free["playActions"]:
        action.pop("actor")
    decision = {"kind": "play", "selector": {}}

    first = OrToolsExactStateNormalizerV1(rules).normalize(
        authored,
        decision_point=decision,
    )
    second = OrToolsExactStateNormalizerV1(rules).normalize(
        actor_free,
        decision_point=decision,
    )

    assert first.to_dict() == second.to_dict()
    assert first.normalization_receipt.to_dict() == (
        second.normalization_receipt.to_dict()
    )

    wrong_actor = copy.deepcopy(authored)
    wrong_actor["playActions"][0]["actor"] = "west"
    with pytest.raises(ValueError, match="play action 0 actor mismatch"):
        OrToolsExactStateNormalizerV1(rules).normalize(
            wrong_actor,
            decision_point=decision,
        )


def test_cp_exact_project_decision_matches_frozen_legacy(rules, root: Path):
    prefix = _flagship_prefix(root)
    prefix["playActions"] = []
    prefix["projectDeclarations"] = []
    prefix["projectAssertions"] = []
    prefix["stateAssertions"] = {}
    decision = {
        "kind": "project",
        "selector": {"owner": "north", "projectIndex": 0},
    }

    cp_normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision,
    )
    legacy_normalized = ExactStateNormalizerV1(rules).normalize(
        prefix,
        decision_point=decision,
    )

    assert cp_normalized.to_dict() == legacy_normalized.to_dict()
    assert cp_normalized.exact_state_id == "STATE-925FD116BCDD145C34EB"
    assert cp_normalized.legal_actions == (
        {
            "code": "declare:hundred_sequence:20:7",
            "action": "declare",
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "highRank": 7,
        },
    )


def test_cp_exact_complete_round_scores_and_matches_frozen_legacy(
    rules,
    root: Path,
):
    prefix = _flagship_prefix(root)
    decision = {"kind": "play", "selector": {}}
    full = copy.deepcopy(prefix)
    full["playActions"] = [
        *prefix["playActions"],
        {"actor": "west", "card": "D7", "ekka": False},
        {"actor": "east", "card": "C8", "ekka": False},
        {"actor": "north", "card": "C7", "ekka": False},
        {"actor": "west", "card": "CA", "ekka": False},
        {"actor": "south", "card": "S7", "ekka": False},
        {"actor": "west", "card": "D8", "ekka": False},
        {"actor": "south", "card": "SA", "ekka": False},
        {"actor": "east", "card": "C9", "ekka": False},
        {"actor": "north", "card": "DA", "ekka": False},
    ]
    full["stateAssertions"] = {}

    cp_normalized = OrToolsExactStateNormalizerV1(rules).normalize(
        full,
        decision_point=decision,
    )
    legacy_normalized = ExactStateNormalizerV1(rules).normalize(
        full,
        decision_point=decision,
    )

    assert cp_normalized.to_dict() == legacy_normalized.to_dict()
    assert cp_normalized.exact_state_id == "STATE-93157C5FC624EACE1DA5"
    assert cp_normalized.phase == "scored"
    assert cp_normalized.next_actor is None
    assert cp_normalized.legal_actions == ()
    assert cp_normalized.state["matchScore"] == [30, 16]
    assert cp_normalized.state["roundScore"]["rawCardPoints"] == [46, 84]


def test_native_cp_exact_derives_flagship_without_python_semantic_replay(
    rules,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixed_owner_calls = []
    attach_projects = exact_state_module.attach_project_lifecycle_fragment_v1

    def capture_fixed_owners(*args, fixed_card_owners=None, **kwargs):
        fixed_owner_calls.append(tuple(fixed_card_owners or ()))
        return attach_projects(
            *args,
            fixed_card_owners=fixed_card_owners,
            **kwargs,
        )

    monkeypatch.setattr(
        exact_state_module,
        "attach_project_lifecycle_fragment_v1",
        capture_fixed_owners,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy Python semantic replay was called")

    for name in (
        "_derive_play_prefix",
        "_replay_play_prefix",
        "_verify_project_assertions",
    ):
        monkeypatch.setattr(exact_state_module, name, forbidden)
    monkeypatch.setattr(
        type(rules.play_domain),
        "resolve_first_leader",
        forbidden,
    )
    for name in (
        "announce_for_turn",
        "reveal_for_turn",
        "score_team",
        "baloot_project_state",
        "project_points",
        "declaration_state",
        "apply_declaration_action",
        "legal_declaration_actions",
        "should_announce",
    ):
        monkeypatch.setattr(type(rules.project_domain), name, forbidden)
    monkeypatch.setattr(
        exact_state_module.OrToolsProjectLifecycleModelV1,
        "solve",
        forbidden,
    )

    normalizer = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    )
    prefix = _flagship_prefix(root)
    normalized = normalizer.normalize(
        prefix,
        decision_point={"kind": "play", "selector": {}},
    )
    preview = normalizer.preview_legal_next_actions(prefix)

    assert normalized.exact_state_id == "STATE-C1540EA736F0D5920BA3"
    assert normalized.next_actor == "west"
    expected_owners = {
        card["id"]: seat
        for seat, cards in normalized.state["initialHands"].items()
        for card in cards
    }
    assert fixed_owner_calls
    assert all(
        tuple(owner.name.lower() for owner in owners)
        == tuple(expected_owners[card_id] for card_id in range(32))
        for owners in fixed_owner_calls
    )
    receipt = normalized.normalization_receipt.to_dict()
    assert receipt["backendId"] == "ortools_cp_sat_v3"
    assert receipt["uniquenessStatus"] == "proven_unique"
    assert receipt["uniquenessStages"] == {
        "dealAuction": "cp_sat_second_solve_infeasible",
        "roles": "cp_sat_second_solve_infeasible",
        "escalation": "cp_sat_fixed_trace_optimal",
        "projects": "cp_sat_fixed_hands_and_prefix_unique",
        "play": "cp_sat_fixed_cards_second_solve_infeasible",
        "scoring": "not_applicable",
    }
    assert receipt["derivedRoles"]["uniquenessSolverStatus"] == "infeasible"
    assert receipt["nativePlayNormalization"]["uniquenessSolverStatus"] == "infeasible"
    assert receipt["derivedProjectPrefix"]["uniquenessSolverStatus"] == "infeasible"
    assert (
        receipt["derivedProjects"]["modelHash"]
        == receipt["derivedPlay"]["modelHash"]
        == receipt["derivedProjectPrefix"]["modelHash"]
    )
    assert preview["currentStage"] == "play"
    assert receipt["nativePlayNormalization"]["legalPlayActions"] == [
        {
            "actor": normalized.next_actor,
            "card": action["code"].removesuffix("+ekka"),
            "ekka": action["ekka"],
        }
        for action in normalized.legal_actions
    ]


@pytest.mark.parametrize("prefix_length", range(23, 33))
def test_native_cp_exact_complete_round_uses_native_scoring_binding(
    rules,
    root: Path,
    monkeypatch,
    prefix_length,
):
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailBuildV1

    solve = OrToolsPlayTailBuildV1.solve
    solves = []
    def counted_solve(self, **kwargs):
        solves.append(self)
        return solve(self, **kwargs)
    monkeypatch.setattr(OrToolsPlayTailBuildV1, "solve", counted_solve)

    def forbidden(*args, **kwargs):
        pytest.fail("native exact scoring called interpreted raw conversion")

    monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    monkeypatch.setattr(ScoringDomainIRV1, "tie_winner", forbidden)
    monkeypatch.setattr(ScoringDomainIRV1, "resolve_counting_team", forbidden)
    monkeypatch.setattr(ScoringDomainIRV1, "multiplier", forbidden)
    prefix = _flagship_prefix(root)
    full = copy.deepcopy(prefix)
    full["playActions"] = [
        *prefix["playActions"],
        {"actor": "west", "card": "D7", "ekka": False},
        {"actor": "east", "card": "C8", "ekka": False},
        {"actor": "north", "card": "C7", "ekka": False},
        {"actor": "west", "card": "CA", "ekka": False},
        {"actor": "south", "card": "S7", "ekka": False},
        {"actor": "west", "card": "D8", "ekka": False},
        {"actor": "south", "card": "SA", "ekka": False},
        {"actor": "east", "card": "C9", "ekka": False},
        {"actor": "north", "card": "DA", "ekka": False},
    ]
    full["stateAssertions"] = {}
    continuation = full["playActions"]
    full["playActions"] = full["playActions"][:prefix_length]

    normalized = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(full, decision_point={"kind": "play", "selector": {}})

    if prefix_length < 32:
        # One joint solve; semantic uniqueness uses its separate CP solver.
        assert len(solves) == 1
        assert normalized.next_actor == continuation[prefix_length]["actor"]
        assert continuation[prefix_length]["card"] in [action["code"] for action in normalized.legal_actions]
        from baloot_platform.core.enums import Contract
        assert exact_state_module._legal_play_rows(
            exact_state_module._native_boundary_legal_play_actions(
                normalized.state, rules=rules, contract=Contract.SUN, trump=None, maximum_seconds=5)
        ) == normalized.legal_actions
        receipt = normalized.normalization_receipt.to_dict()
        assert receipt["nativePlayNormalization"]["constructionHintStatus"] == "joint_solution_continuation"
        assert receipt["nativePlayNormalization"]["constructionHintModelHash"] is None
        assert receipt["nativePlayNormalization"]["uniquenessSolverStatus"] == "infeasible"
        return
    assert normalized.exact_state_id == "STATE-93157C5FC624EACE1DA5"
    assert len(solves) == 1
    assert normalized.phase == "scored"
    assert normalized.state["matchScore"] == [30, 16]
    receipt = normalized.normalization_receipt.to_dict()
    assert receipt["nativePlayNormalization"]["constructionHintStatus"] == "not_required_complete_prefix"
    assert receipt["nativePlayNormalization"]["constructionHintModelHash"] is None
    assert receipt["uniquenessStages"]["scoring"] == ("cp_sat_fixed_inputs_optimal")
    assert receipt["derivedScoring"]["solverStatus"] == "optimal"
    assert receipt["derivedScoring"]["score"]["rawPoints"] == [46, 84]
    assert (
        receipt["derivedScoring"]["modelHash"]
        == receipt["derivedProjects"]["modelHash"]
        == receipt["derivedPlay"]["modelHash"]
        == receipt["derivedProjectPrefix"]["modelHash"]
    )


def test_native_cp_exact_partial_round_keeps_every_declared_project(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = _prefix(
        actions=[
            {"actor": "north", "action": "pass"},
            {"actor": "west", "action": "hokum", "trumpSuit": "diamonds"},
            {"actor": "south", "action": "sun"},
            {"actor": "west", "action": "pass"},
            {"actor": "north", "action": "sun"},
        ]
    )
    prefix["deckOrder"] = [
        31, 28, 14, 30, 3, 13, 21, 22, 25, 10, 11, 12, 16, 18, 15, 23,
        4, 0, 6, 2, 17, 19, 20, 24, 26, 27, 1, 5, 7, 29, 8, 9,
    ]
    prefix["projectDeclarations"] = [
        {
            "owner": "north",
            "kind": "hundred_sequence",
            "value": 20,
            "highRank": 4,
            "declarationOrder": 1,
        },
        {
            "owner": "east",
            "kind": "hundred_sequence",
            "value": 20,
            "highRank": 4,
            "declarationOrder": 2,
        },
    ]
    prefix["playActions"] = [
        {"card": card, "ekka": False}
        for card in (
            "CJ", "CK", "C8", "CQ", "DA", "DK", "H10", "D8",
            "S10", "SQ", "SK", "D10", "HQ", "S7", "HJ", "HK",
            "D7", "C9", "DQ", "H9", "S8", "S9", "D9",
        )
    ]

    normalized = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(prefix, decision_point={"kind": "play", "selector": {}})

    assert normalized.project_declarations_verified == 2
    assert [
        event["owner"]
        for event in normalized.state["events"]
        if event["type"] == "project_declared"
    ] == ["north", "east"]


def test_native_cp_exact_keeps_official_ashkal_roles_distinct(root: Path):
    official = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    prefix = _prefix(
        actions=[
            {"action": "pass"},
            {"action": "pass"},
            {"action": "ashkal"},
            {"action": "pass"},
            {"action": "pass"},
        ]
    )

    normalized = OrToolsExactStateNormalizerV1(
        official,
        native_cp_sat=True,
    ).normalize(prefix)

    assert normalized.state["winningBidder"] == "south"
    assert normalized.state["buyer"] == "south"
    assert normalized.state["contractOwner"] == "south"
    assert normalized.state["declarer"] == "south"
    assert normalized.state["purchaseRecipient"] == "north"
    assert normalized.state["purchaseCard"] in normalized.state["hands"]["north"]
    roles = normalized.normalization_receipt.to_dict()["derivedRoles"]
    assert roles["buyer"] == roles["contractOwner"] == roles["declarer"] == ("south")
    assert roles["purchaseRecipient"] == roles["firstLeader"] == "north"


@pytest.mark.parametrize(
    ("rules_name", "buyer", "recipient", "declarer"),
    (
        ("saudi_federation_2023_v2.yaml", "south", "north", "south"),
        ("saudi_common_v1.yaml", "north", "north", "north"),
    ),
)
def test_ashkal_exact_roles_match_runtime_reference_and_native_cp_sat(
    root: Path,
    rules_name: str,
    buyer: str,
    recipient: str,
    declarer: str,
) -> None:
    rules = RuleBook.load(root / "configs/rules" / rules_name)
    prefix = _prefix(
        actions=[
            {"action": "pass"},
            {"action": "pass"},
            {"action": "ashkal"},
            {"action": "pass"},
            {"action": "pass"},
        ]
    )
    runtime = ExactStateNormalizerV1(rules).normalize(prefix)
    reference = OrToolsExactStateNormalizerV1(rules).normalize(prefix)
    native = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(prefix)
    expected = {
        "winningBidder": "south",
        "buyer": buyer,
        "contractOwner": buyer,
        "purchaseRecipient": recipient,
        "declarer": declarer,
    }
    for normalized in (runtime, reference, native):
        assert {
            key: normalized.state[key] for key in expected
        } == expected

    derived = reference.normalization_receipt.to_dict()["derivedAuction"]
    assert (derived["caller"], derived["buyer"], derived["purchaseRecipient"], derived["declarer"]) == (
        "south",
        buyer,
        recipient,
        declarer,
    )
    native_roles = native.normalization_receipt.to_dict()["derivedRoles"]
    assert (
        native_roles["caller"],
        native_roles["buyer"],
        native_roles["purchaseRecipient"],
        native_roles["declarer"],
    ) == ("south", buyer, recipient, declarer)


def test_cp_exact_requires_exactly_one_copy_of_every_card(rules):
    prefix = _prefix()
    prefix["deckOrder"][-1] = 0
    with pytest.raises(ValueError, match="permutation of 0..31"):
        OrToolsExactStateNormalizerV1(rules).normalize(prefix)


def test_cp_exact_preserves_neutral_wire_and_binds_hashes(rules):
    normalized = OrToolsExactStateNormalizerV1(rules).normalize(_prefix())
    wire = normalized.to_dict()
    assert set(wire) == {
        "schemaVersion",
        "exactStateId",
        "prefixHash",
        "rulesHash",
        "stateHash",
        "phase",
        "decisionKind",
        "nextActor",
        "legalActions",
        "state",
        "projectDeclarationsVerified",
        "projectAssertionsVerified",
        "eventChainHash",
        "sha256",
    }
    assert wire["stateHash"] == canonical_sha256(wire["state"])
    assert wire["eventChainHash"] == canonical_sha256(wire["state"]["events"])
    exact_core = {
        key: wire[key]
        for key in (
            "schemaVersion",
            "prefixHash",
            "rulesHash",
            "stateHash",
            "phase",
            "decisionKind",
            "nextActor",
            "legalActions",
            "eventChainHash",
        )
    }
    assert wire["sha256"] == canonical_sha256(exact_core)
    assert wire["exactStateId"] == f"STATE-{wire['sha256'][:20].upper()}"
    receipt = normalized.normalization_receipt.to_dict()
    receipt_core = {key: value for key, value in receipt.items() if key != "sha256"}
    assert receipt["sha256"] == canonical_sha256(receipt_core)
    assert receipt["exactStateId"] == wire["exactStateId"]
    assert receipt["stateHash"] == wire["stateHash"]


@pytest.mark.parametrize("kind", ("bid", "escalation", "project"))
def test_append_normalized_nonplay_action_is_bound_and_replayable(rules, root, kind):
    raw = _prefix()
    decision = {"kind": kind}
    field = {"bid": "auctionActions", "escalation": "escalationActions", "project": "projectActions"}[kind]
    if kind == "escalation":
        raw["auctionActions"] = [{"action": action} for action in ("hokum", "pass", "pass", "pass")]
        decision["selector"] = {"actingSeat": "west"}
    elif kind == "project":
        rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
        raw["auctionActions"] = [{"actor": "north", "action": "sun"}]
        raw["projectActions"] = [{"actor": actor, "action": "pass"} for actor in ("north", "west", "south")]
        raw["playActions"] = [{"actor": actor, "card": card} for actor, card in (("north", "S7"), ("west", "S10"), ("south", "SK"))]
        decision["selector"] = {"owner": "east", "projectIndex": 0}
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize(prefix, decision_point=decision, maximum_seconds=15)
    selected = next(row["code"] for row in normalized.legal_actions
                    if row["action"] == ("double" if kind == "escalation" else "pass"))
    advanced = normalizer.append_normalized_action(prefix, normalized, selected)
    assert len(advanced.to_dict()[field]) == len(raw.get(field, [])) + 1
    assert prefix.to_dict()[field] == raw.get(field, [])
    after = normalizer.normalize(advanced, decision_point={"kind": "bid" if kind == "bid" else "play"}, maximum_seconds=15)
    assert after.legal_actions
    if kind != "bid":
        assert after.normalization_receipt.to_dict()["derivedEscalation"]["observation"]["terminal"] is True
    with pytest.raises(ValueError, match="unique legal action"):
        normalizer.append_normalized_action(prefix, normalized, "not-an-action")
    stale = prefix.to_dict()
    stale.pop("sha256")
    stale["roundNumber"] += 1
    with pytest.raises(ValueError, match="different prefix or rules"):
        normalizer.append_normalized_action(ExactTracePrefixV1.build(stale), normalized, selected)
    if kind == "escalation":
        from baloot_platform.core.escalation_ir import EscalationDomainIRV1
        closure = EscalationDomainIRV1.from_rules(rules.data).closure_action
        with pytest.raises(ValueError, match="explicit stage-transition record"):
            normalizer.append_normalized_action(prefix, normalized, closure)


@pytest.mark.parametrize("passes", [0, 4, 8])
def test_native_bid_append_reaches_purchase_or_all_pass_boundary(rules, passes):
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    prefix = ExactTracePrefixV1.build(_prefix())
    for _ in range(passes):
        before = normalizer.normalize(prefix, decision_point={"kind": "bid"})
        selected = next(row["code"] for row in before.legal_actions if row["action"] == "pass")
        prefix = normalizer.append_normalized_action(prefix, before, selected)
    before = normalizer.normalize(prefix, decision_point={"kind": "bid"})
    if passes == 8:
        assert before.next_actor is None
        assert not before.legal_actions
        assert before.phase == "complete"
        assert before.state["buyer"] is None and before.state["contract"] is None
        assert before.state["matchScore"] == list(prefix.match_score)
        initial = normalizer.normalize(ExactTracePrefixV1.build(_prefix()), decision_point={"kind": "bid"})
        assert set(initial.state) - {"bidding"} <= set(before.state)
        assert before.state["dealTrace"] == initial.state["dealTrace"]
        assert before.state["shuffle"]["deckOrder"] == list(prefix.deck_order)
        assert before.state["roundScore"] is None
        assert before.state["projects"] == []
        assert before.state["events"][-1]["type"] == "all_pass_redeal"
        events = before.state["events"]
        assert [event["sequence"] for event in events] == list(range(len(events)))
        assert events[0]["type"] == "round_created"
        assert sum(event["count"] for event in events if event["type"] == "cards_dealt") == 20
        assert next(event for event in events if event["type"] == "purchase_card_exposed")["card"] == before.state["purchaseCard"]
        bids = [{key: value for key, value in event.items() if key not in {"sequence", "type"}}
                for event in events if event["type"] == "bid_made"]
        assert bids == before.state["biddingHistory"]
        assert [(row["round"], row["turn"]) for row in bids] == [
            (round_index, turn) for round_index in (1, 2) for turn in range(4)
        ]
        assert all(row["action"] == "pass" and row["buyer"] is None for row in bids)
        receipt = before.normalization_receipt.to_dict()
        assert receipt["uniquenessModelHash"] != receipt["modelHash"]
        assert receipt["structuralDomainIrHash"] == rules.structural_domain.sha256
        assert receipt["variableCount"] > 0 and receipt["constraintCount"] > 0
        assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
        assert normalizer.normalize(prefix, decision_point={"kind": "bid"}).to_dict() == before.to_dict()
        preview = normalizer.preview_legal_next_actions(prefix)
        assert preview["terminal"] is True
        assert preview["nextActor"] is None
        assert preview["legalActions"] == []
        return
    selected = next(row["code"] for row in before.legal_actions if row["action"] == "sun")
    prefix = normalizer.append_normalized_action(prefix, before, selected)
    after = normalizer.normalize(prefix, decision_point={"kind": "bid"})
    assert after.state["contract"] == "sun"
    assert after.state["buyer"] == before.next_actor
    assert not after.legal_actions


@pytest.mark.parametrize("mutation", ["actor", "trump", "play", "assertion"])
@pytest.mark.parametrize("method", ["normalize", "preview_legal_next_actions"])
def test_native_all_pass_rejects_invalid_prefix_and_assertions(rules, mutation, method):
    raw = _prefix(actions=[{"action": "pass"} for _ in range(8)])
    if mutation == "actor":
        raw["auctionActions"][0]["actor"] = "south"
    elif mutation == "trump":
        raw["auctionActions"][0]["trumpSuit"] = "spades"
    elif mutation == "play":
        raw["playActions"] = [{"actor": "north", "card": "S7"}]
    else:
        raw["stateAssertions"] = {"buyer": "north"}
    with pytest.raises(ValueError):
        getattr(OrToolsExactStateNormalizerV1(rules, native_cp_sat=True), method)(raw)


@pytest.mark.parametrize("stop_at", [1, 2])
def test_native_all_pass_unknown_cannot_certify_uniqueness(rules, monkeypatch, stop_at):
    original = exact_state_module._solve_cp_sat_with_cancel_v1
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == stop_at:
            return exact_state_module.cp_model.UNKNOWN
        return original(*args, **kwargs)

    monkeypatch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", interrupted)
    with pytest.raises(ExactNormalizationStopped):
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            _prefix(actions=[{"action": "pass"} for _ in range(8)])
        )
    assert calls == stop_at


def test_native_all_pass_cancelled_before_launch_never_solves(rules, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cancelled normalization launched a solver")

    monkeypatch.setattr(exact_state_module, "_solve_cp_sat_with_cancel_v1", forbidden)
    with pytest.raises(ExactNormalizationStopped):
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            _prefix(actions=[{"action": "pass"} for _ in range(8)]), cancel_check=lambda: True
        )


@pytest.mark.parametrize("declare", [False, True])
def test_native_routes_interleaved_first_trick_without_skipping_declarations(root, declare, monkeypatch):
    extract = exact_state_module._extract_native_project_prefix_projection
    awarded_totals = []
    def checked_extract(**kwargs):
        binding, solver = kwargs["binding"], kwargs["solver"]
        totals = tuple(int(solver.value(value)) for value in binding.combined_awarded_value)
        assert totals == tuple(
            sum(project.value for project in binding.fragment.projects
                if project.owner.team == team and solver.boolean_value(binding.awarded_now[project.project_id]))
            for team in exact_state_module.Team
        )
        assert {value.index for value in binding.combined_awarded_value} <= {value.index for value in binding.semantic_variables}
        if kwargs["prefix_length"] < 4:
            assert totals == (0, 0)
        awarded_totals.append(totals)
        return extract(**kwargs)
    monkeypatch.setattr(exact_state_module, "_extract_native_project_prefix_projection", checked_extract)
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _prefix(actions=[{"actor": "north", "action": "sun"}])
    raw["projectActions"] = []
    source = ExactTracePrefixV1.build(raw)
    prefix = source
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    order = rules.project_domain.declaration_seats(exact_state_module.Seat.parse(prefix.dealer))
    declarations = 0
    for actor in order:
        project = normalizer.normalize_play_or_project(prefix)
        assert project.decision_kind == "project"
        assert project.next_actor == actor.name.lower()
        choices = [row for row in project.legal_actions if row["action"] == "declare"]
        selected = choices[0] if declare and choices else next(
            row for row in project.legal_actions if row["action"] == "pass"
        )
        declarations += selected["action"] == "declare"
        prefix = normalizer.append_normalized_action(prefix, project, selected["code"])
        play = normalizer.normalize_play_or_project(prefix)
        assert play.decision_kind == "play"
        assert play.next_actor == actor.name.lower()
        assert play.state["events"][-1]["type"] in {"project_passed", "project_declared"}
        assert play.state["events"][-1]["owner"] == actor.name.lower()
        prefix = normalizer.append_normalized_action(prefix, play, play.legal_actions[0]["code"])
    after = normalizer.normalize_play_or_project(prefix)
    assert after.decision_kind == "play"
    assert len(prefix.project_actions) == len(prefix.play_actions) == len(order)
    assert len(source.project_actions) == len(source.play_actions) == 0
    assert bool(declarations) == declare
    assert bool(sum(awarded_totals[-1])) == declare
    timeline = [event for event in after.state["events"]
                if event["type"] in {"project_passed", "project_declared", "card_played"}]
    assert len(timeline) == 8
    for index, actor in enumerate(order):
        declaration, played = timeline[index * 2:index * 2 + 2]
        assert declaration["type"] in {"project_passed", "project_declared"}
        assert declaration["owner"] == played["seat"] == actor.name.lower()
        assert played["type"] == "card_played"
    broken = prefix.to_dict()
    broken.pop("sha256")
    broken["projectActions"].pop()
    with pytest.raises(ValueError, match="project actions must occur in order"):
        normalizer.normalize_play_or_project(ExactTracePrefixV1.build(broken))
    with pytest.raises(ExactNormalizationStopped):
        normalizer.normalize_play_or_project(source, cancel_check=lambda: True)


@pytest.mark.parametrize("native_cp_sat", [False, True])
def test_cp_exact_prefix_preview_restricts_each_stage(
    rules, root: Path, native_cp_sat, monkeypatch,
):
    original_solve = exact_state_module.OrToolsPlayTailStageV1.solve
    conversion_modes = []

    def record_conversion(stage, **kwargs):
        conversion_modes.append(stage.native_conversion)
        return original_solve(stage, **kwargs)

    monkeypatch.setattr(
        exact_state_module.OrToolsPlayTailStageV1, "solve", record_conversion,
    )
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=native_cp_sat)

    auction = normalizer.preview_legal_next_actions(_prefix())
    assert auction["currentStage"] == "auction"
    assert auction["nextActor"] == "north"
    assert {item["code"] for item in auction["legalActions"]} == {
        "pass",
        "sun",
        "hokum:diamonds",
    }
    assert {item["stage"] for item in auction["legalActions"]} == {"auction"}

    hokum = _prefix(
        actions=[
            {"action": "hokum"},
            {"action": "pass"},
            {"action": "pass"},
            {"action": "pass"},
        ]
    )
    escalation = normalizer.preview_legal_next_actions(hokum)
    assert escalation["currentStage"] == "escalation"
    assert {item["stage"] for item in escalation["legalActions"]} == {
        "escalation",
        "play",
    }
    assert {
        item["actor"]
        for item in escalation["legalActions"]
        if item["stage"] == "escalation"
    } == {
        "west",
        "east",
    }

    play_prefix = _flagship_prefix(root)
    play_prefix["playActions"] = []
    play_prefix["projectDeclarations"] = []
    play_prefix["projectAssertions"] = []
    play_prefix["stateAssertions"] = {}
    project = normalizer.preview_legal_next_actions(play_prefix)
    assert project["currentStage"] == "project"
    assert {item["stage"] for item in project["legalActions"]} == {"play"}
    assert all(item["projectDeclarations"] for item in project["legalActions"])
    assert {
        item["projectDeclarations"][0]["owner"] for item in project["legalActions"]
    } == {project["nextActor"]}
    assert conversion_modes and set(conversion_modes) == {native_cp_sat}


def test_cp_exact_prefix_preview_fails_closed_on_illegal_cross_stage_prefix(rules):
    prefix = _prefix()
    prefix["playActions"] = [{"card": 0}]
    with pytest.raises(ValueError, match="before purchase"):
        OrToolsExactStateNormalizerV1(rules).preview_legal_next_actions(prefix)


def test_cp_exact_module_has_no_runtime_research_or_engine_imports(root: Path):
    path = root / "src/baloot_platform/research/ortools_exact_state.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden = {
        "baloot_platform.core.engine",
        "baloot_platform.core.bidding_machine",
        "baloot_platform.core.bots",
        "baloot_platform.research.exact_state",
        "baloot_platform.research.proof_service",
        "baloot_platform.research.reference_play",
        "reference_play",
        "baloot_platform.research.scenario_evidence",
    }
    assert imported.isdisjoint(forbidden)
