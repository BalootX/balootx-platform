from __future__ import annotations

import copy

import pytest

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.rules import RuleBook
from baloot_platform.config import build_bundle
from baloot_platform.research import OrToolsCpSatAuctionCompilerV1
from baloot_platform.research.reference_auction import ReferenceAuctionInterpreterV1
from baloot_platform.service.container import build_container


@pytest.fixture(scope="module")
def published_rules(root):
    return build_bundle(root).rules


@pytest.mark.parametrize("machine_type", (BiddingMachine, ReferenceAuctionInterpreterV1))
def test_auction_enumeration_reuses_frozen_domain_without_changing_branches(
    published_rules, monkeypatch, machine_type,
):
    domain = AuctionDomainIRV1.from_rules(published_rules.data)
    root = machine_type(Seat.EAST, Suit.HEARTS, domain)
    root.submit("hokum")
    unchanged = copy.deepcopy(root)
    compiler = OrToolsCpSatAuctionCompilerV1()
    expected = compiler._enumerate_auctions(
        root, dealer=Seat.EAST, purchase_suit=Suit.HEARTS,
    )

    def reject_domain_copy(self, memo):
        raise AssertionError("immutable auction rules must not be copied per branch")

    monkeypatch.setattr(AuctionDomainIRV1, "__deepcopy__", reject_domain_copy, raising=False)
    actual = compiler._enumerate_auctions(
        root, dealer=Seat.EAST, purchase_suit=Suit.HEARTS,
    )

    assert actual == expected
    assert root == unchanged
    assert root.domain is domain


def test_auction_tables_reuse_only_identical_immutable_inputs(published_rules):
    enumerated = []

    class CountingCompiler(OrToolsCpSatAuctionCompilerV1):
        @classmethod
        def _enumerate_auctions(cls, root, **kwargs):
            enumerated.append(type(root))
            return super()._enumerate_auctions(root, **kwargs)

    options = dict(dealer=Seat.EAST, purchase_suit=Suit.HEARTS, rules=published_rules)
    reference = CountingCompiler._legal_auctions(**options)
    runtime = CountingCompiler._runtime_legal_auctions(**options)
    assert reference == runtime
    copied_rules = {**options, "rules": RuleBook(copy.deepcopy(published_rules.data))}
    assert CountingCompiler._legal_auctions(**copied_rules) is reference
    assert CountingCompiler._runtime_legal_auctions(**copied_rules) is runtime
    assert enumerated == [ReferenceAuctionInterpreterV1, BiddingMachine]

    changed = copy.deepcopy(published_rules.data)
    ashkal = changed["bidding"]["ashkal"]
    ashkal["contract_owner"] = (
        "taker" if ashkal.get("contract_owner", "taker") == "bidder" else "bidder"
    )
    for updates in (
        {"rules": RuleBook(changed)},
        {"dealer": Seat.NORTH},
        {"purchase_suit": Suit.CLUBS},
        {"purchase_rank": Rank.ACE},
    ):
        before = len(enumerated)
        actual = {**options, **updates}
        assert CountingCompiler._legal_auctions(**actual) == (
            CountingCompiler._runtime_legal_auctions(**actual)
        )
        assert enumerated[before:] == [ReferenceAuctionInterpreterV1, BiddingMachine]


def test_compiler_proves_first_and_second_purchase_with_runtime_parity(published_rules):
    rules = published_rules
    compiler = OrToolsCpSatAuctionCompilerV1()
    first = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_buyer=Seat.SOUTH,
        required_contracts=(Contract.HOKUM,),
        required_purchase_round=1,
    )
    assert first is not None
    assert first.buyer == Seat.SOUTH
    assert first.contract == Contract.HOKUM
    assert first.purchase_round == 1
    assert first.trump == Suit.HEARTS
    assert first.solver_status == "optimal"

    second = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_buyer=Seat.NORTH,
        required_contracts=(Contract.HOKUM,),
        required_purchase_round=2,
        required_trump=Suit.SPADES,
    )
    assert second is not None
    assert second.buyer == Seat.NORTH
    assert second.purchase_round == 2
    assert second.trump == Suit.SPADES


def test_compiler_proves_ashkal_buyer_and_is_deterministic(published_rules):
    rules = published_rules
    compiler = OrToolsCpSatAuctionCompilerV1()
    first = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.CLUBS,
        rules=rules,
        required_buyer=Seat.NORTH,
        required_contracts=(Contract.ASHKAL,),
        required_purchase_round=1,
    )
    second = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.CLUBS,
        rules=rules,
        required_buyer=Seat.NORTH,
        required_contracts=(Contract.ASHKAL,),
        required_purchase_round=1,
    )
    assert first == second
    assert first is not None
    assert first.contract == Contract.ASHKAL
    assert first.buyer == Seat.NORTH
    assert any(item.action == "ashkal" and item.actor == Seat.SOUTH for item in first.actions)


def test_compiler_returns_infeasible_for_impossible_first_round_trump(published_rules):
    rules = published_rules
    result = OrToolsCpSatAuctionCompilerV1().compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_contracts=(Contract.HOKUM,),
        required_purchase_round=1,
        required_trump=Suit.SPADES,
    )
    assert result is None


def test_compiler_rejects_invalid_purchase_round(published_rules):
    rules = published_rules
    with pytest.raises(ValueError, match="purchase round"):
        OrToolsCpSatAuctionCompilerV1().compile(
            dealer=Seat.EAST,
            purchase_suit=Suit.HEARTS,
            rules=rules,
            required_purchase_round=3,
        )


def test_compiler_proves_one_exact_actor_safe_auction_trace(published_rules):
    rules = published_rules
    compiler = OrToolsCpSatAuctionCompilerV1()
    selected = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    assert selected is not None

    verified = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_actions=tuple(item.to_dict() for item in selected.actions),
    )

    assert verified is not None
    assert verified.actions == selected.actions
    assert verified.buyer == selected.buyer
    assert verified.contract == selected.contract
    assert verified.trump == selected.trump
    assert verified.model_hash != selected.model_hash


def test_compiler_accepts_a_required_auction_prefix(published_rules):
    rules = published_rules
    compiler = OrToolsCpSatAuctionCompilerV1()
    selected = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    assert selected is not None
    required_prefix = tuple(
        {
            "actor": item.actor.name.lower(),
            "action": item.action,
            **(
                {"trumpSuit": item.trump.name.lower()}
                if item.trump is not None
                else {}
            ),
        }
        for item in selected.actions[:2]
    )

    verified = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_actions=required_prefix,
        required_actions_complete=False,
    )

    assert verified is not None
    assert tuple(
        (item.actor.name.lower(), item.action)
        for item in verified.actions[: len(required_prefix)]
    ) == tuple((item["actor"], item["action"]) for item in required_prefix)
    assert len(verified.actions) >= len(required_prefix)


def test_compiler_rejects_an_exact_trace_that_conflicts_with_required_outcome(published_rules):
    rules = published_rules
    compiler = OrToolsCpSatAuctionCompilerV1()
    selected = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_contracts=(Contract.SUN,),
    )
    assert selected is not None

    assert compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
        required_contracts=(Contract.HOKUM,),
        required_actions=selected.actions,
    ) is None


def test_compiler_rejects_non_contiguous_mapping_trace_ordinals(published_rules):
    rules = published_rules
    compiler = OrToolsCpSatAuctionCompilerV1()
    selected = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    assert selected is not None
    actions = [item.to_dict() for item in selected.actions]
    actions[0]["ordinal"] = 2

    with pytest.raises(ValueError, match="contiguous ordinals"):
        compiler.compile(
            dealer=Seat.EAST,
            purchase_suit=Suit.HEARTS,
            rules=rules,
            required_actions=actions,
        )


def test_compiled_auction_ir_is_identical_in_all_three_domain_targets():
    container = build_container()
    compiled = container.research.compiled_domain.compiled
    targets = compiled.target_payloads
    runtime = targets["runtimePlan"]["compiledAuction"]
    reference = targets["referencePlan"]["compiledAuction"]
    cp_sat = targets["cpSatBlueprint"]["compiledAuction"]
    assert runtime == reference == cp_sat
    assert runtime["sha256"] == AuctionDomainIRV1.from_rules(
        container.research.domain.rules
    ).sha256


def test_auction_ir_mutation_influences_runtime_reference_and_cp_sat(published_rules):
    baseline_payload = copy.deepcopy(dict(published_rules.data))
    mutated_payload = copy.deepcopy(baseline_payload)
    mutated_payload["contracts"]["allowed"] = ["sun", "hokum"]
    mutated_payload["bidding"]["first_round"]["allowed"] = [
        "pass", "sun", "hokum"
    ]
    mutated_payload["bidding"]["second_round"]["allowed"] = [
        "pass", "sun", "hokum"
    ]
    baseline = RuleBook(baseline_payload)
    mutated = RuleBook(mutated_payload)

    def legal_after_two_passes(rulebook):
        domain = AuctionDomainIRV1.from_rules(rulebook.data)
        runtime = BiddingMachine(Seat.EAST, Suit.HEARTS, domain)
        reference = ReferenceAuctionInterpreterV1(Seat.EAST, Suit.HEARTS, domain)
        for machine in (runtime, reference):
            machine.submit("pass")
            machine.submit("pass")
        assert runtime.legal_actions() == reference.legal_actions()
        return runtime.legal_actions()

    assert "ashkal" in legal_after_two_passes(baseline)
    assert "ashkal" not in legal_after_two_passes(mutated)

    compiler = OrToolsCpSatAuctionCompilerV1()
    assert compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=baseline,
        required_buyer=Seat.NORTH,
        required_contracts=(Contract.ASHKAL,),
        required_purchase_round=1,
    ) is not None
    assert compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=mutated,
        required_buyer=Seat.NORTH,
        required_contracts=(Contract.ASHKAL,),
        required_purchase_round=1,
    ) is None


def test_auction_ir_rejects_conflicting_pass_configuration(published_rules):
    payload = copy.deepcopy(dict(published_rules.data))
    payload["bidding"]["pass_allowed"] = False
    with pytest.raises(ValueError, match="pass action conflicts"):
        AuctionDomainIRV1.from_rules(payload)


def test_auction_ir_rejects_unsupported_ashkal_resolution(published_rules):
    payload = copy.deepcopy(dict(published_rules.data))
    payload["bidding"]["ashkal"]["resolves_to"] = "ashkal"
    with pytest.raises(ValueError, match="only Sun resolution"):
        AuctionDomainIRV1.from_rules(payload)


def test_auction_ir_rejects_duplicate_contracts(published_rules):
    payload = copy.deepcopy(dict(published_rules.data))
    payload["contracts"]["allowed"] = ["sun", "sun"]
    with pytest.raises(ValueError, match="unique and non-empty"):
        AuctionDomainIRV1.from_rules(payload)


def test_runtime_rejects_the_legacy_parallel_auction_configuration():
    with pytest.raises(TypeError, match="compiled AuctionDomainIRV1"):
        BiddingMachine(Seat.EAST, Suit.HEARTS, {Contract.SUN})


@pytest.mark.parametrize("dealer", tuple(Seat))
@pytest.mark.parametrize("purchase_suit", tuple(Suit))
def test_domain_ir_reference_matches_every_runtime_auction_trace(
    dealer, purchase_suit, published_rules
):
    rules = published_rules
    proof = OrToolsCpSatAuctionCompilerV1.reference_runtime_parity(
        dealer=dealer,
        purchase_suit=purchase_suit,
        rules=rules,
    )
    assert proof["targetsObserved"] == ["reference", "runtime"]
    assert proof["traceCount"] > 0
    assert len(proof["traceSetHash"]) == 64
    assert len(proof["proofHash"]) == 64


@pytest.mark.parametrize(
    "allowed",
    [
        ("sun",),
        ("hokum",),
        ("ashkal",),
        ("sun", "hokum"),
        ("sun", "hokum", "ashkal"),
    ],
)
def test_domain_ir_reference_matches_supported_contract_profiles(
    allowed, published_rules
):
    baseline = published_rules
    payload = copy.deepcopy(dict(baseline.data))
    payload["contracts"]["allowed"] = list(allowed)
    payload["bidding"]["first_round"]["allowed"] = ["pass", *allowed]
    payload["bidding"]["second_round"]["allowed"] = [
        "pass", *[item for item in allowed if item != "ashkal"]
    ]
    hokum_enabled = "hokum" in allowed
    payload["bidding"]["first_round"]["hokum_uses_purchase_suit"] = hokum_enabled
    payload["bidding"]["second_round"][
        "hokum_must_differ_from_purchase_suit"
    ] = hokum_enabled
    rules = RuleBook(payload)
    proof = OrToolsCpSatAuctionCompilerV1.reference_runtime_parity(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    assert proof["traceCount"] > 0


@pytest.mark.parametrize("contract", ("sun", "hokum", "ashkal"))
def test_preselected_contract_profile_matches_reference_runtime_and_cp_sat(
    contract, published_rules
):
    payload = copy.deepcopy(dict(published_rules.data))
    payload["contracts"].update(
        {"allowed": [contract], "default": contract, "require_preselected": True}
    )
    payload["bidding"]["first_round"]["allowed"] = ["pass", contract]
    payload["bidding"]["second_round"]["allowed"] = [
        "pass", *([] if contract == "ashkal" else [contract])
    ]
    hokum_enabled = contract == "hokum"
    payload["bidding"]["first_round"]["hokum_uses_purchase_suit"] = hokum_enabled
    payload["bidding"]["second_round"][
        "hokum_must_differ_from_purchase_suit"
    ] = hokum_enabled
    rules = RuleBook(payload)
    compiler = OrToolsCpSatAuctionCompilerV1()
    proof = compiler.reference_runtime_parity(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    plan = compiler.compile(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    assert proof["traceCount"] == 1
    assert plan is not None
    assert plan.contract == Contract.parse(contract)
    assert plan.purchase_round == 1
    assert plan.solver_status == "optimal"
