"""CP-SAT foundations for the OR-Tools-only Research V4 backend."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from fractions import Fraction
from math import floor, isfinite, lcm
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.rules import RuleBook

from .scenario_contracts import ScenarioExpressionV1, canonical_sha256
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1

if TYPE_CHECKING:
    from .ortools_play_tail import OrToolsPlayTailBuildV1, TailPrefixActionV1
    from .ortools_project_fragment import OrToolsProjectLifecycleFragmentV1
    from .ortools_scoring_stage import OrToolsScoringFragmentV1
    from .scenario_predicate_auction_escalation_v2 import CpSatEscalationPredicateContextV1


_CONTRACT_IDS = {
    Contract.SUN: 0,
    Contract.HOKUM: 1,
    Contract.ASHKAL: 2,
}
_NO_TRUMP = len(Suit)
NATIVE_OBSERVED_AUCTION_POLICY_IDS = (
    "prefer_legal_sun_then_pass_v1", "prefer_root_actor_sun_partner_pass_v1")


def validate_native_observed_auction_policy_id_v1(policy_id):
    if type(policy_id) is not str or policy_id not in NATIVE_OBSERVED_AUCTION_POLICY_IDS:
        raise ValueError("unsupported native observed auction policy")
    return policy_id


@dataclass(frozen=True, slots=True)
class OrToolsNativeRoundFragmentV1:
    """One purchased-round branch, not a strategy or exhaustive branch proof."""

    play: OrToolsPlayTailBuildV1
    escalation: CpSatEscalationPredicateContextV1
    projects: OrToolsProjectLifecycleFragmentV1
    scoring: OrToolsScoringFragmentV1
    scope: Mapping[str, Any]
    escalation_opportunities: tuple[Mapping[str, Any], ...]
    public_baloot_binding: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    public_project_bindings: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class OrToolsAuctionHandValuesV1:
    """Native fixed-point heuristic values, not probabilities or game outcomes."""

    scale: int
    sun: cp_model.IntVar
    hokum: Mapping[Suit, cp_model.IntVar]
    thresholds: Mapping[str, int]
    profile_hash: str


def attach_auction_hand_values_v1(
    model: cp_model.CpModel, rules: RuleBook, *,
    visible_cards: Sequence[cp_model.IntVar], name: str = "auction_hand",
) -> OrToolsAuctionHandValuesV1:
    """Lower the frozen bidding heuristic over symbolic actor-visible cards.

    The caller supplies visibility membership, not a hidden complete hand.
    This fragment does not select actions or qualify playing strength.
    """
    if len(visible_cards) != 32:
        raise ValueError("auction hand evaluation requires 32 card membership variables")
    profile = rules.data["bidding"]["heuristics"]
    rank_keys = ("sun_rank_weights", "hokum_trump_rank_weights")
    count_keys = ("sun_extra_suit_card_after", "hokum_extra_trump_card_after")
    coefficients = {}
    for key, raw in profile.items():
        if key in count_keys:
            if type(raw) is not int or not 0 <= raw <= cp_model.INT_MAX // 128:
                raise ValueError("auction hand count thresholds must be non-negative integers")
            continue
        values = raw if key in rank_keys else [raw]
        if key in rank_keys and len(values) != len(Rank):
            raise ValueError("auction rank weights must cover every rank")
        if any(type(value) not in {int, float} for value in values):
            raise ValueError("auction hand coefficients must be numeric")
        fractions = tuple(Fraction(str(value)) for value in values)
        if any(value < 0 for value in fractions):
            raise ValueError("auction hand coefficients must be non-negative")
        coefficients[key] = fractions
    scale = lcm(*(value.denominator for values in coefficients.values() for value in values))
    weights = {key: tuple(int(value * scale) for value in values) for key, values in coefficients.items()}
    largest = max(value for values in weights.values() for value in values)
    if max(scale, largest) > cp_model.INT_MAX // 128:
        raise ValueError("auction hand coefficient precision exceeds CP-SAT integer bounds")
    for card in visible_cards:
        model.add_linear_constraint(card, 0, 1)
    cards = tuple(Card(index) for index in range(32))

    def excess(suit, key):
        result = model.new_int_var(0, len(Rank), f"{name}_{key}_{int(suit)}")
        count = sum(visible_cards[card.id] for card in cards if card.suit == suit)
        model.add_max_equality(result, [0, count - profile[key]])
        return result

    sun = model.new_int_var(0, largest * 64, f"{name}_sun")
    model.add(sun == sum(visible_cards[card.id] * weights["sun_rank_weights"][int(card.rank)] for card in cards)
              + weights["sun_extra_suit_card_weight"][0] * sum(excess(suit, count_keys[0]) for suit in Suit))
    hokum = {}
    for suit in Suit:
        value = model.new_int_var(0, largest * 64, f"{name}_hokum_{int(suit)}")
        model.add(value == sum(visible_cards[card.id] * weights["hokum_trump_rank_weights"][int(card.rank)]
                               for card in cards if card.suit == suit)
                  + weights["hokum_extra_trump_card_weight"][0] * excess(suit, count_keys[1])
                  + weights["hokum_off_trump_ace_ten_weight"][0] * sum(
                      visible_cards[card.id] for card in cards
                      if card.suit != suit and card.rank in {Rank.ACE, Rank.TEN}))
        hokum[suit] = value
    return OrToolsAuctionHandValuesV1(
        scale, sun, hokum,
        {key: values[0] for key, values in weights.items() if key.endswith("threshold")},
        canonical_sha256(profile),
    )


def attach_auction_action_values_v1(
    model: cp_model.CpModel, rules: RuleBook, *,
    visible_cards: Sequence[cp_model.IntVar], purchase_card: Card,
    round_index: cp_model.IntVar, name: str = "auction_action_value",
) -> tuple[OrToolsAuctionHandValuesV1, Mapping[str, cp_model.IntVar]]:
    """Heuristic bid margins using only five own cards and the exposed card.

    Legal-action filtering belongs to the native auction fragment. These values
    are not terminal utility, calibrated probabilities, or strategic proof.
    """
    purchase = Card.parse(purchase_card)
    hand = attach_auction_hand_values_v1(model, rules, visible_cards=visible_cards, name=name)
    model.add(sum(visible_cards) == sum(rules.structural_domain.initial_packets))
    model.add(visible_cards[purchase.id] == 0)
    model.add_linear_constraint(round_index, 1, 2)
    public_cards = list(visible_cards)
    public_cards[purchase.id] = model.new_constant(1)
    # Ashkal evaluates the caller's support, never their partner's hidden hand.
    support = attach_auction_hand_values_v1(
        model, rules, visible_cards=public_cards, name=f"{name}_ashkal_support",
    )
    first = model.new_bool_var(f"{name}_first_round")
    model.add(round_index == 1).only_enforce_if(first)
    model.add(round_index == 2).only_enforce_if(first.Not())
    values = {"pass": model.new_constant(0)}

    def margin(code, score, first_threshold, second_threshold):
        lower = min(score.proto.domain) - max(first_threshold, second_threshold)
        upper = max(score.proto.domain)
        value = model.new_int_var(lower, upper, f"{name}_{code}")
        model.add(value == score - first_threshold).only_enforce_if(first)
        model.add(value == score - second_threshold).only_enforce_if(first.Not())
        values[code] = value

    margin("sun", hand.sun, hand.thresholds["sun_threshold"], hand.thresholds["second_sun_threshold"])
    for suit in Suit:
        margin(f"hokum:{suit.name.lower()}", hand.hokum[suit],
               hand.thresholds["hokum_threshold"], hand.thresholds["second_hokum_threshold"])
    margin("ashkal", support.sun, hand.thresholds["ashkal_partner_threshold"], hand.thresholds["ashkal_partner_threshold"])
    return hand, values


def _auction_logic_v1(model, name):
    def equal(variable, value, label):
        result = model.new_bool_var(f"{name}_{label}")
        model.add(variable == value).only_enforce_if(result)
        model.add(variable != value).only_enforce_if(result.Not())
        return result

    def conjunction(label, *terms):
        result = model.new_bool_var(f"{name}_{label}")
        model.add_min_equality(result, terms or (1,))
        return result

    def disjunction(label, *terms):
        result = model.new_bool_var(f"{name}_{label}")
        model.add_max_equality(result, terms or (0,))
        return result
    return equal, conjunction, disjunction


def attach_auction_legal_actions_v1(
    model: cp_model.CpModel, domain: AuctionDomainIRV1, *,
    round_index: cp_model.IntVar, position: cp_model.IntVar,
    in_recall: cp_model.IntVar, terminal: cp_model.IntVar,
    candidate_contract: cp_model.IntVar, purchase_rank: cp_model.IntVar,
    purchase_suit: cp_model.IntVar, prior_nonpass_count: cp_model.IntVar,
    name: str = "auction_native",
) -> tuple[Mapping[str, cp_model.IntVar], Mapping[str, cp_model.IntVar]]:
    """Lower legal choices from symbolic state; candidate -1 means no purchase.

    This fragment does not certify reachability of the supplied state variables.
    The transition fragment must bind them; no interpreted legality is an input.
    """
    if not isinstance(model, cp_model.CpModel) or not isinstance(domain, AuctionDomainIRV1):
        raise TypeError("auction lowering requires CP-SAT and AuctionDomainIRV1")
    inputs = ((round_index, 1, domain.rounds), (position, 1, len(Seat)),
              (in_recall, 0, 1), (terminal, 0, 1),
              (candidate_contract, -1, len(_CONTRACT_IDS) - 1),
              (purchase_rank, 0, len(Rank) - 1), (purchase_suit, 0, len(Suit) - 1))
    for variable, lower, upper in inputs:
        if not isinstance(variable, cp_model.IntVar):
            raise TypeError("auction state inputs must be solver variables")
        model.add(variable >= lower)
        model.add(variable <= upper)
    if not isinstance(prior_nonpass_count, cp_model.IntVar):
        raise TypeError("auction history count must be a solver variable")
    model.add(prior_nonpass_count >= 0)

    equal, conjunction, disjunction = _auction_logic_v1(model, name)
    first = equal(round_index, 1, "first")
    no_candidate = equal(candidate_contract, -1, "no_candidate")
    sun_candidate = equal(candidate_contract, _CONTRACT_IDS[Contract.SUN], "sun_candidate")
    hokum_candidate = equal(candidate_contract, _CONTRACT_IDS[Contract.HOKUM], "hokum_candidate")
    ace = equal(purchase_rank, int(Rank.ACE), "ace")
    right = equal(position, 1, "right")
    restricted = conjunction("restricted", first, ace, hokum_candidate, right.Not(),
                             int(domain.exposed_ace_hokum_sun_overcall == "dealer_right_only"))
    sun_state = disjunction("sun_state",
        conjunction("recall_sun", in_recall, restricted.Not()),
        conjunction("ordinary_sun", in_recall.Not(), sun_candidate.Not(), restricted.Not()))
    eligible = disjunction("ashkal_position", 0, *(
        equal(position, value, f"ashkal_position_{value}") for value in domain.ashkal_priority_positions))
    ashkal_state = conjunction("ashkal_state", in_recall.Not(), no_candidate, eligible,
                              equal(prior_nonpass_count, 0, "prior_passes"), int(domain.ashkal_enabled))
    states = {"pass": int(domain.pass_allowed), "sun": sun_state,
              "hokum": conjunction("hokum_state", in_recall.Not(), no_candidate), "ashkal": ashkal_state}
    actions = {}
    for action, state in states.items():
        allowed = disjunction(f"{action}_allowed",
            conjunction(f"{action}_first", first, int(action in domain.first_round_allowed)),
            conjunction(f"{action}_second", first.Not(), int(action in domain.second_round_allowed)))
        actions[action] = conjunction(f"legal_{action}", terminal.Not(), allowed, state)
    choices = {action: literal for action, literal in actions.items() if action != "hokum"}
    for suit in Suit:
        same = equal(purchase_suit, int(suit), f"purchase_{suit.name.lower()}")
        permitted = disjunction(f"trump_{suit.name.lower()}",
            conjunction(f"first_trump_{suit.name.lower()}", first, same),
            conjunction(f"second_trump_{suit.name.lower()}", first.Not(), same.Not()))
        choices[f"hokum:{suit.name.lower()}"] = conjunction(
            f"legal_hokum_{suit.name.lower()}", actions["hokum"], permitted)
    return actions, choices


def attach_auction_path_v1(
    model: cp_model.CpModel, domain: AuctionDomainIRV1, *,
    dealer: cp_model.IntVar, purchase_suit: cp_model.IntVar, purchase_rank: cp_model.IntVar,
    maximum_actions: int, name: str = "auction_path",
    legal_rows: list[Mapping[str, cp_model.IntVar]] | None = None,
) -> tuple[tuple[Mapping[str, cp_model.IntVar], ...], tuple[Mapping[str, cp_model.IntVar], ...]]:
    """Native finite auction transitions; callers constrain actions or terminal.

    Closed padding freezes the terminal state. No complete-trace table or Python
    state interpreter supplies transition truth. Preselected profiles emit their
    automatic recorded prefix, not player decision opportunities.
    """
    if not isinstance(model, cp_model.CpModel) or not isinstance(domain, AuctionDomainIRV1):
        raise TypeError("auction path requires CP-SAT and AuctionDomainIRV1")
    preselected_position = 0
    if domain.require_preselected:
        if domain.default_contract is None or domain.default_contract.value not in domain.first_round_allowed:
            raise ValueError("preselected auction requires an allowed default contract")
        if domain.default_contract == Contract.ASHKAL:
            preselected_position = 2 if domain.ashkal_taker == "bidder_partner" else 0
            if not domain.ashkal_enabled or preselected_position + 1 not in domain.ashkal_priority_positions:
                raise ValueError("preselected Ashkal requires an eligible bidder")
    if type(maximum_actions) is not int or not 1 <= maximum_actions <= 2 * len(Seat):
        raise ValueError("maximum_actions must be an integer from one to eight")
    if not isinstance(dealer, cp_model.IntVar):
        raise TypeError("dealer must be a solver variable")
    model.add(dealer >= 0)
    model.add(dealer < len(Seat))
    bounds = {"round": (1, domain.rounds), "cursor": (0, 4), "recall": (-1, 2),
              "terminal": (0, 1), "allPassed": (0, 1), "nonpass": (0, maximum_actions),
              "contract": (-1, 2), "trump": (0, _NO_TRUMP), "candidatePosition": (-1, 3),
              "caller": (-1, 3), "buyer": (-1, 3), "declarer": (-1, 3),
              "recipient": (-1, 3), "purchaseRound": (0, 2)}
    states = tuple({key: model.new_int_var(low, high, f"{name}_{index}_{key}")
                    for key, (low, high) in bounds.items()} for index in range(maximum_actions + 1))
    initial = {"round": 1, "cursor": 0, "recall": -1, "terminal": 0, "allPassed": 0,
               "nonpass": 0, "contract": -1, "trump": _NO_TRUMP, "candidatePosition": -1,
               "caller": -1, "buyer": -1, "declarer": -1, "recipient": -1, "purchaseRound": 0}
    for key, value in initial.items():
        model.add(states[0][key] == value)
    selections = []
    for index, (before, after) in enumerate(zip(states, states[1:])):
        stage = f"{name}_{index}"
        equal, both, either = _auction_logic_v1(model, stage)
        live = before["terminal"].Not()
        recall = model.new_bool_var(f"{stage}_in_recall")
        model.add(recall == equal(before["recall"], -1, "ordinary").Not())
        first = equal(before["round"], 1, "first_round")
        position = model.new_int_var(1, 4, f"{stage}_position")
        model.add(position == before["recall"] + 1).only_enforce_if([live, recall])
        model.add(position == before["cursor"] + 1).only_enforce_if([live, recall.Not()])
        model.add(position == 1).only_enforce_if(before["terminal"])
        actor = model.new_int_var(0, 3, f"{stage}_actor")
        model.add_modulo_equality(actor, dealer + position, len(Seat))
        before["actor"] = actor
        before["position"] = position
        before["inRecall"] = recall
        _, legal = attach_auction_legal_actions_v1(model, domain, round_index=before["round"],
            position=position, in_recall=recall, terminal=before["terminal"],
            candidate_contract=before["contract"], purchase_rank=purchase_rank,
            purchase_suit=purchase_suit, prior_nonpass_count=before["nonpass"], name=f"{stage}_legal")
        if legal_rows is not None:
            legal_rows.append(legal)
        chosen = {code: model.new_bool_var(f"{stage}_choose_{code}") for code in (*legal, "closed")}
        model.add_exactly_one(chosen.values())
        model.add(chosen["closed"] == before["terminal"])
        if domain.require_preselected:
            if index < preselected_position:
                model.add(chosen["pass"] == 1)
            elif index > preselected_position:
                model.add(chosen["closed"] == 1)
            elif domain.default_contract == Contract.HOKUM:
                for suit in Suit:
                    model.add(chosen[f"hokum:{suit.name.lower()}"] == equal(purchase_suit, int(suit), f"default_{suit.name.lower()}"))
            else:
                model.add(chosen[domain.default_contract.value] == 1)
        else:
            for code, literal in legal.items():
                model.add_implication(chosen[code], literal)
        selections.append(chosen)
        sunlike = either("sunlike", chosen["sun"], chosen["ashkal"])
        purchase = either("purchase", sunlike, *(chosen[f"hokum:{s.name.lower()}"] for s in Suit))
        partner = model.new_int_var(0, 3, f"{stage}_partner")
        model.add_modulo_equality(partner, actor + 2, len(Seat))
        for code, selected in chosen.items():
            if code in {"pass", "closed"}:
                for key in ("contract", "trump", "candidatePosition", "caller", "buyer", "declarer", "recipient", "purchaseRound"):
                    model.add(after[key] == before[key]).only_enforce_if(selected)
                continue
            contract = Contract.HOKUM if code.startswith("hokum:") else Contract.parse(code)
            recipient = partner if contract == Contract.ASHKAL and domain.ashkal_taker == "bidder_partner" else actor
            buyer = actor if contract == Contract.ASHKAL and domain.ashkal_contract_owner == "bidder" else recipient
            values = {"contract": _CONTRACT_IDS[contract], "candidatePosition": position - 1,
                      "caller": actor, "recipient": recipient, "buyer": buyer, "declarer": buyer,
                      "purchaseRound": before["round"],
                      "trump": int(Suit.parse(code.split(":")[1])) if contract == Contract.HOKUM else _NO_TRUMP}
            for key, value in values.items():
                model.add(after[key] == value).only_enforce_if(selected)

        def move(guard, round_value, cursor, recall_value, terminal_value, all_passed=0, reset=False):
            values = {"round": round_value, "cursor": cursor, "recall": recall_value,
                      "terminal": terminal_value, "allPassed": all_passed,
                      "nonpass": 0 if reset else before["nonpass"] + purchase}
            for key, value in values.items():
                model.add(after[key] == value).only_enforce_if(guard)

        move(before["terminal"], before["round"], before["cursor"], before["recall"], 1, before["allPassed"])
        if domain.require_preselected:
            move(live, 1, before["cursor"] + int(index < preselected_position), -1,
                 int(index >= preselected_position))
            if index == preselected_position:
                first_actor = model.new_int_var(0, 3, f"{stage}_default_buyer")
                model.add_modulo_equality(first_actor, dealer + 1, len(Seat))
                model.add(after["buyer"] == first_actor)
            continue
        move(both("recalling", live, recall), before["round"], before["cursor"], before["recall"] - 1,
             equal(before["recall"], 0, "last_recall"))
        ordinary = both("live_ordinary", live, recall.Not())
        move(both("first_sun", ordinary, first, sunlike), 1, before["cursor"], before["cursor"] - 1,
             equal(before["cursor"], 0, "first_position"))
        last = equal(before["cursor"], 3, "last_position")
        advancing = both("first_advancing", ordinary, first, sunlike.Not())
        move(both("first_next", advancing, last.Not()), 1, before["cursor"] + 1, -1, 0)
        no_candidate = equal(after["contract"], -1, "no_candidate")
        end = both("first_end", advancing, last)
        move(both("first_passed", end, no_candidate), domain.rounds, 0 if domain.rounds == 2 else 4,
             -1, int(domain.rounds == 1), int(domain.rounds == 1), reset=domain.rounds == 2)
        move(both("first_recall", end, no_candidate.Not()), 1, 4, after["candidatePosition"] - 1,
             equal(after["candidatePosition"], 0, "candidate_first"))
        move(both("second_sun", ordinary, first.Not(), sunlike), 2, before["cursor"], -1, 1)
        move(both("second_next", ordinary, first.Not(), sunlike.Not()), 2, before["cursor"] + 1,
             -1, last, both("second_all_passed", last, no_candidate))
    return states, tuple(selections)


def attach_optimal_auction_choices_v1(
    model: cp_model.CpModel, rules: RuleBook, *,
    position_of_card: Sequence[cp_model.IntVar], purchase_position: int,
    states: Sequence[Mapping[str, cp_model.IntVar]],
    choices: Sequence[Mapping[str, cp_model.IntVar]],
    legal_rows: Sequence[Mapping[str, cp_model.IntVar]],
    dealer: Seat, name: str = "auction_quality",
) -> None:
    """Require maximal frozen visible-hand values, including tied choices."""
    if len(position_of_card) != 32 or len(states) != len(choices) + 1 or len(legal_rows) != len(choices):
        raise ValueError("auction quality path inputs are inconsistent")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer)
    purchase = []
    for card, position in enumerate(position_of_card):
        literal = model.new_bool_var(f"{name}_purchase_{card}")
        model.add(position == purchase_position).only_enforce_if(literal)
        model.add(position != purchase_position).only_enforce_if(literal.Not())
        purchase.append(literal)
    hands = {}
    support = {}
    for seat in Seat:
        positions = frozenset(layout.positions_for(seat, final=False))
        visible = []
        for card, position in enumerate(position_of_card):
            literal = model.new_bool_var(f"{name}_{seat.name.lower()}_{card}")
            model.add_allowed_assignments(
                [position, literal],
                [(index, int(index in positions)) for index in range(32)],
            )
            visible.append(literal)
        hands[seat] = attach_auction_hand_values_v1(
            model, rules, visible_cards=visible, name=f"{name}_{seat.name.lower()}",
        )
        support[seat] = attach_auction_hand_values_v1(
            model, rules, visible_cards=[visible[index] + purchase[index] for index in range(32)],
            name=f"{name}_{seat.name.lower()}_support",
        )

    thresholds = hands[Seat.NORTH].thresholds
    for index, (state, selected, legal) in enumerate(zip(states[:-1], choices, legal_rows, strict=True)):
        actor = state["actor"]
        first = model.new_bool_var(f"{name}_{index}_first")
        model.add(state["round"] == 1).only_enforce_if(first)
        model.add(state["round"] == 2).only_enforce_if(first.Not())
        values = {"pass": model.new_constant(0)}
        scores = {
            "sun": [hands[seat].sun for seat in Seat],
            "ashkal": [support[seat].sun for seat in Seat],
            **{f"hokum:{suit.name.lower()}": [hands[seat].hokum[suit] for seat in Seat] for suit in Suit},
        }
        for code, seat_scores in scores.items():
            score = model.new_int_var(0, max(max(item.proto.domain) for item in seat_scores), f"{name}_{index}_{code}_score")
            model.add_element(actor, seat_scores, score)
            first_key, second_key = (
                ("sun_threshold", "second_sun_threshold") if code == "sun"
                else ("ashkal_partner_threshold", "ashkal_partner_threshold") if code == "ashkal"
                else ("hokum_threshold", "second_hokum_threshold")
            )
            lower = -max(thresholds[first_key], thresholds[second_key])
            value = model.new_int_var(lower, max(score.proto.domain), f"{name}_{index}_{code}_margin")
            model.add(value == score - thresholds[first_key]).only_enforce_if(first)
            model.add(value == score - thresholds[second_key]).only_enforce_if(first.Not())
            values[code] = value
        for code, chosen in selected.items():
            if code == "closed":
                continue
            for other, permitted in legal.items():
                guard = model.new_bool_var(f"{name}_{index}_{code}_over_{other}")
                model.add_min_equality(guard, [chosen, permitted])
                model.add(values[code] >= values[other]).only_enforce_if(guard)


@dataclass(frozen=True, slots=True)
class OrToolsAuctionActionV1:
    round_index: int
    actor: Seat
    action: str
    trump: Suit | None
    contract: Contract | None
    buyer: Seat | None
    purchase_recipient: Seat | None = None
    caller: Seat | None = None
    declarer: Seat | None = None

    @property
    def choice_code(self) -> str:
        return (
            f"hokum:{self.trump.name.lower()}"
            if self.action == "hokum" and self.trump is not None
            else self.action
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "roundIndex": self.round_index,
            "actor": self.actor.name.lower(),
            "action": self.action,
            "trumpSuit": self.trump.name.lower() if self.trump is not None else None,
        }
        if self.contract is not None:
            if None in {
                self.caller,
                self.buyer,
                self.purchase_recipient,
                self.declarer,
            }:
                raise AssertionError("purchased auction action lacks typed roles")
            value.update(
                {
                    "caller": self.caller.name.lower(),
                    "buyer": self.buyer.name.lower(),
                    "purchaseRecipient": self.purchase_recipient.name.lower(),
                    "declarer": self.declarer.name.lower(),
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class _RequiredAuctionActionV1:
    actor: Seat | None
    action: str
    trump: Suit | None
    trump_specified: bool
    round_index: int | None

    def matches(self, action: OrToolsAuctionActionV1) -> bool:
        return (
            (self.actor is None or action.actor == self.actor)
            and action.action == self.action
            and (self.round_index is None or action.round_index == self.round_index)
            and (not self.trump_specified or action.trump == self.trump)
        )


@dataclass(frozen=True, slots=True)
class _AuctionStateV1:
    round_index: int
    cursor: int
    records: tuple[OrToolsAuctionActionV1, ...]
    candidate: OrToolsAuctionActionV1 | None
    recall_queue: tuple[Seat, ...]
    recall_position: int
    terminal: bool
    all_passed: bool
    contract: Contract | None
    buyer: Seat | None
    trump: Suit | None


@dataclass(frozen=True, slots=True)
class _AuctionPlanV1:
    purchase_suit: Suit
    purchase_rank: Rank | None
    actions: tuple[OrToolsAuctionActionV1, ...]
    caller: Seat
    buyer: Seat
    purchase_recipient: Seat
    declarer: Seat
    contract: Contract
    trump: Suit | None
    purchase_round: int
    prefix_actions: tuple[OrToolsAuctionActionV1, ...]
    prefix_caller: Seat | None
    prefix_buyer: Seat | None
    prefix_purchase_recipient: Seat | None
    prefix_declarer: Seat | None
    prefix_contract: Contract | None
    prefix_trump: Suit | None
    prefix_purchase_round: int | None
    current_actor: Seat | None
    current_round: int
    prefix_terminal: bool
    legal_next_actions: tuple[str, ...]
    legal_next_choices: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class OrToolsScenarioAuctionObservationV1:
    actions: tuple[OrToolsAuctionActionV1, ...]
    buyer: Seat | None
    contract: Contract | None
    trump: Suit | None
    purchase_round: int | None
    current_actor: Seat | None
    current_round: int
    prefix_terminal: bool
    legal_next_actions: tuple[str, ...]
    legal_next_choices: tuple[str, ...]
    trace_hash: str
    completion_actions: tuple[OrToolsAuctionActionV1, ...]
    completion_buyer: Seat | None
    completion_contract: Contract | None
    completion_trump: Suit | None
    completion_purchase_round: int | None
    prefix_purchase_recipient: Seat | None = None
    completion_purchase_recipient: Seat | None = None
    caller: Seat | None = None
    declarer: Seat | None = None
    completion_caller: Seat | None = None
    completion_declarer: Seat | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "actions": [action.to_dict() for action in self.actions],
            "buyer": self.buyer.name.lower() if self.buyer is not None else None,
            "contract": self.contract.value if self.contract is not None else None,
            "trumpSuit": self.trump.name.lower() if self.trump is not None else None,
            "purchaseRound": self.purchase_round,
            "currentActor": (
                self.current_actor.name.lower()
                if self.current_actor is not None
                else None
            ),
            "currentRound": self.current_round,
            "prefixTerminal": self.prefix_terminal,
            "legalNextActions": list(self.legal_next_actions),
            "legalNextChoices": list(self.legal_next_choices),
            "traceHash": self.trace_hash,
            "completion": {
                "actions": [action.to_dict() for action in self.completion_actions],
                "buyer": self.completion_buyer.name.lower() if self.completion_buyer is not None else None,
                "contract": self.completion_contract.value if self.completion_contract is not None else None,
                "trumpSuit": (
                    self.completion_trump.name.lower()
                    if self.completion_trump is not None
                    else None
                ),
                "purchaseRound": self.completion_purchase_round,
            },
        }
        if (
            self.prefix_purchase_recipient is not None
            and self.prefix_purchase_recipient != self.buyer
        ):
            value["purchaseRecipient"] = self.prefix_purchase_recipient.name.lower()
        if (
            self.completion_purchase_recipient is not None
            and self.completion_purchase_recipient != self.completion_buyer
        ):
            value["completion"]["purchaseRecipient"] = (
                self.completion_purchase_recipient.name.lower()
            )
        if self.caller is not None:
            value["caller"] = self.caller.name.lower()
        if self.declarer is not None:
            value["declarer"] = self.declarer.name.lower()
        if self.completion_caller is not None:
            value["completion"]["caller"] = self.completion_caller.name.lower()
        if self.completion_declarer is not None:
            value["completion"]["declarer"] = (
                self.completion_declarer.name.lower()
            )
        return value


@dataclass(frozen=True, slots=True)
class OrToolsScenarioAuctionFragmentV1:
    plans: tuple[_AuctionPlanV1, ...]
    selected: tuple[cp_model.IntVar, ...]
    contract: cp_model.IntVar
    trump: cp_model.IntVar
    purchase_round: cp_model.IntVar
    caller: cp_model.IntVar
    buyer: cp_model.IntVar
    purchase_recipient: cp_model.IntVar
    declarer: cp_model.IntVar
    terminal: cp_model.IntVar
    legal_next_action_literals: Mapping[str, cp_model.IntVar]
    legal_next_choice_literals: Mapping[str, cp_model.IntVar]
    domain_hash: str

    def extract(self, solver: cp_model.CpSolver) -> OrToolsScenarioAuctionObservationV1:
        chosen = [
            index
            for index, literal in enumerate(self.selected)
            if solver.boolean_value(literal)
        ]
        if len(chosen) != 1:
            raise AssertionError("auction fragment did not select one terminal plan")
        plan = self.plans[chosen[0]]
        legal_actions = tuple(
            action
            for action, literal in sorted(self.legal_next_action_literals.items())
            if solver.boolean_value(literal)
        )
        legal_choices = tuple(
            choice
            for choice, literal in sorted(self.legal_next_choice_literals.items())
            if solver.boolean_value(literal)
        )
        expected_trump = (
            _NO_TRUMP if plan.trump is None else list(Suit).index(plan.trump)
        )
        extracted_values = (
            solver.value(self.contract),
            solver.value(self.trump),
            solver.value(self.purchase_round),
            solver.value(self.caller),
            solver.value(self.buyer),
            solver.value(self.purchase_recipient),
            solver.value(self.declarer),
            bool(solver.boolean_value(self.terminal)),
            legal_actions,
            legal_choices,
        )
        expected_values = (
            _CONTRACT_IDS[plan.contract],
            expected_trump,
            plan.purchase_round,
            int(plan.caller),
            int(plan.buyer),
            int(plan.purchase_recipient),
            int(plan.declarer),
            plan.prefix_terminal,
            tuple(sorted(plan.legal_next_actions)),
            tuple(sorted(plan.legal_next_choices)),
        )
        if extracted_values != expected_values:
            raise AssertionError(
                "auction fragment variables disagree with selected plan"
            )
        trace_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "actions": [action.to_dict() for action in plan.prefix_actions],
                "caller": (
                    plan.prefix_caller.name.lower()
                    if plan.prefix_caller is not None
                    else None
                ),
                "buyer": (
                    plan.prefix_buyer.name.lower()
                    if plan.prefix_buyer is not None
                    else None
                ),
                "purchaseRecipient": (
                    plan.prefix_purchase_recipient.name.lower()
                    if plan.prefix_purchase_recipient is not None
                    else None
                ),
                "declarer": (
                    plan.prefix_declarer.name.lower()
                    if plan.prefix_declarer is not None
                    else None
                ),
                "contract": (
                    plan.prefix_contract.value
                    if plan.prefix_contract is not None
                    else None
                ),
                "trumpSuit": (
                    plan.prefix_trump.name.lower()
                    if plan.prefix_trump is not None
                    else None
                ),
                "purchaseRound": plan.prefix_purchase_round,
                "currentActor": (
                    plan.current_actor.name.lower()
                    if plan.current_actor is not None
                    else None
                ),
                "currentRound": plan.current_round,
                "prefixTerminal": plan.prefix_terminal,
                "legalNextActions": list(legal_actions),
                "legalNextChoices": list(legal_choices),
            }
        )
        return OrToolsScenarioAuctionObservationV1(
            actions=plan.prefix_actions,
            buyer=plan.prefix_buyer,
            contract=plan.prefix_contract,
            trump=plan.prefix_trump,
            purchase_round=plan.prefix_purchase_round,
            current_actor=plan.current_actor,
            current_round=plan.current_round,
            prefix_terminal=bool(solver.boolean_value(self.terminal)),
            legal_next_actions=legal_actions,
            legal_next_choices=legal_choices,
            trace_hash=trace_hash,
            completion_actions=plan.actions,
            completion_buyer=plan.buyer,
            completion_contract=plan.contract,
            completion_trump=plan.trump,
            completion_purchase_round=plan.purchase_round,
            prefix_purchase_recipient=plan.prefix_purchase_recipient,
            completion_purchase_recipient=plan.purchase_recipient,
            caller=plan.prefix_caller,
            declarer=plan.prefix_declarer,
            completion_caller=plan.caller,
            completion_declarer=plan.declarer,
        )


@dataclass(frozen=True, slots=True)
class OrToolsNativeAuctionFragmentV1:
    states: tuple[Mapping[str, cp_model.IntVar], ...]
    choices: tuple[Mapping[str, cp_model.IntVar], ...]
    prefix_length: int
    legal_next_action_literals: Mapping[str, cp_model.IntVar]
    legal_next_choice_literals: Mapping[str, cp_model.IntVar]
    domain_hash: str
    public_prefix_bound: bool = False
    legal_choices_by_step: tuple[Mapping[str, cp_model.IntVar], ...] = ()

    @property
    def contract(self):
        return self.states[-1]["contract"]

    @property
    def trump(self):
        return self.states[-1]["trump"]

    @property
    def purchase_round(self):
        return self.states[-1]["purchaseRound"]

    @property
    def caller(self):
        return self.states[-1]["caller"]

    @property
    def buyer(self):
        return self.states[-1]["buyer"]

    @property
    def purchase_recipient(self):
        return self.states[-1]["recipient"]

    @property
    def declarer(self):
        return self.states[-1]["declarer"]

    @property
    def terminal(self):
        return self.states[self.prefix_length]["terminal"]

    def extract_actions(
        self, solver: cp_model.CpSolver, *, length: int | None = None,
    ) -> tuple[OrToolsAuctionActionV1, ...]:
        """Read solved records without requiring a purchased completion."""
        if length is None:
            length = len(self.choices)
        if type(length) is not int or not 0 <= length <= len(self.choices):
            raise ValueError("auction extraction length is outside the modeled path")
        contracts = {value: key for key, value in _CONTRACT_IDS.items()}
        complete = []
        for before, after, row in zip(self.states, self.states[1:], self.choices[:length]):
            code, = [code for code, literal in row.items() if solver.boolean_value(literal)]
            if code == "closed":
                break
            action = code.split(":")[0]
            purchased = action != "pass"
            complete.append(OrToolsAuctionActionV1(
                round_index=solver.value(before["round"]), actor=Seat(solver.value(before["actor"])),
                action=action, trump=Suit(solver.value(after["trump"])) if action == "hokum" else None,
                contract=contracts[solver.value(after["contract"])] if purchased else None,
                buyer=Seat(solver.value(after["buyer"])) if purchased else None,
                purchase_recipient=Seat(solver.value(after["recipient"])) if purchased else None,
                caller=Seat(solver.value(after["caller"])) if purchased else None,
                declarer=Seat(solver.value(after["declarer"])) if purchased else None))
        return tuple(complete)

    def extract(self, solver: cp_model.CpSolver) -> OrToolsScenarioAuctionObservationV1:
        contracts = {value: key for key, value in _CONTRACT_IDS.items()}
        complete = self.extract_actions(solver)
        prefix = complete[:self.prefix_length]
        state, final = self.states[self.prefix_length], self.states[-1]
        terminal = bool(solver.value(state["terminal"]))
        purchased = terminal and not solver.boolean_value(state["allPassed"])
        completed_purchase = not solver.boolean_value(final["allPassed"])
        roles = {key: Seat(solver.value(state[key])) if purchased else None
                 for key in ("buyer", "caller", "declarer", "recipient")}
        contract = contracts[solver.value(state["contract"])] if purchased else None
        trump = Suit(solver.value(state["trump"])) if terminal and solver.value(state["trump"]) != _NO_TRUMP else None
        actor = None if terminal else Seat(solver.value(state["actor"]))
        purchase_round = solver.value(state["purchaseRound"]) if purchased else None
        actions = tuple(sorted(key for key, literal in self.legal_next_action_literals.items() if solver.boolean_value(literal)))
        choices = tuple(sorted(key for key, literal in self.legal_next_choice_literals.items() if solver.boolean_value(literal)))
        trace_hash = canonical_sha256({"schemaVersion": 1, "actions": [row.to_dict() for row in prefix],
            **{key: roles[field].name.lower() if roles[field] is not None else None
               for key, field in {"caller": "caller", "buyer": "buyer", "purchaseRecipient": "recipient", "declarer": "declarer"}.items()},
            "contract": contract.value if contract is not None else None,
            "trumpSuit": trump.name.lower() if trump is not None else None, "purchaseRound": purchase_round,
            "currentActor": actor.name.lower() if actor is not None else None, "currentRound": solver.value(state["round"]),
            "prefixTerminal": terminal, "legalNextActions": list(actions), "legalNextChoices": list(choices)})
        return OrToolsScenarioAuctionObservationV1(actions=prefix, buyer=roles["buyer"], caller=roles["caller"],
            declarer=roles["declarer"], contract=contract, trump=trump, purchase_round=purchase_round,
            current_actor=actor, current_round=solver.value(state["round"]), prefix_terminal=terminal,
            legal_next_actions=actions, legal_next_choices=choices, trace_hash=trace_hash,
            completion_actions=tuple(complete), completion_buyer=Seat(solver.value(final["buyer"])) if completed_purchase else None,
            completion_contract=contracts[solver.value(final["contract"])] if completed_purchase else None,
            completion_trump=Suit(solver.value(final["trump"])) if solver.value(final["trump"]) != _NO_TRUMP else None,
            completion_purchase_round=solver.value(final["purchaseRound"]) if completed_purchase else None, prefix_purchase_recipient=roles["recipient"],
            completion_purchase_recipient=Seat(solver.value(final["recipient"])) if completed_purchase else None,
            completion_caller=Seat(solver.value(final["caller"])) if completed_purchase else None,
            completion_declarer=Seat(solver.value(final["declarer"])) if completed_purchase else None)


def _auction_order(dealer: Seat) -> tuple[Seat, ...]:
    first = dealer.next()
    return (first, first.next(), first.next().next(), dealer)


def _initial_auction_state(
    domain: AuctionDomainIRV1,
    *,
    dealer: Seat,
    purchase_suit: Suit,
) -> _AuctionStateV1:
    if not domain.require_preselected:
        return _AuctionStateV1(1, 0, (), None, (), 0, False, False, None, None, None)

    preselected = domain.preselected_bid(
        dealer=dealer,
        purchase_suit=purchase_suit,
    )
    resolution = preselected.resolution
    if resolution.contract is None or resolution.buyer is None:
        raise AssertionError("preselected auction did not purchase")
    records: list[OrToolsAuctionActionV1] = []
    for actor in _auction_order(dealer):
        if actor == preselected.bidder:
            records.append(
                OrToolsAuctionActionV1(
                    1,
                    actor,
                    resolution.contract.value,
                    resolution.trump,
                    resolution.contract,
                    resolution.buyer,
                    resolution.purchase_recipient,
                    resolution.caller,
                    resolution.declarer,
                )
            )
            break
        records.append(OrToolsAuctionActionV1(1, actor, "pass", None, None, None))
    return _AuctionStateV1(
        1,
        0,
        tuple(records),
        records[-1],
        (),
        0,
        True,
        False,
        resolution.contract,
        resolution.buyer,
        resolution.trump,
    )


def _auction_actor(state: _AuctionStateV1, order: tuple[Seat, ...]) -> Seat:
    if state.recall_queue and state.recall_position < len(state.recall_queue):
        return state.recall_queue[state.recall_position]
    return order[state.cursor]


def _auction_legal_choices(
    domain: AuctionDomainIRV1,
    state: _AuctionStateV1,
    *,
    dealer: Seat,
    purchase_suit: Suit,
    purchase_rank: Rank | None = None,
) -> tuple[tuple[str, Suit | None], ...]:
    if state.terminal:
        return ()
    order = _auction_order(dealer)
    actor = _auction_actor(state, order)
    in_recall = bool(state.recall_queue) and state.recall_position < len(
        state.recall_queue
    )
    actions = domain.legal_actions(
        round_index=state.round_index,
        position=order.index(actor) + 1,
        in_recall=in_recall,
        candidate_contract=(
            state.candidate.contract if state.candidate is not None else None
        ),
        prior_round_actions=tuple(
            record.action
            for record in state.records
            if record.round_index == state.round_index
        ),
        purchase_rank=purchase_rank,
    )
    choices: list[tuple[str, Suit | None]] = []
    for action in actions:
        if action != "hokum":
            choices.append((action, None))
        elif state.round_index == 1:
            choices.append((action, purchase_suit))
        else:
            choices.extend((action, suit) for suit in Suit if suit != purchase_suit)
    return tuple(choices)


def _finish_auction(
    state: _AuctionStateV1,
    candidate: OrToolsAuctionActionV1,
) -> _AuctionStateV1:
    if candidate.contract is None or candidate.buyer is None:
        raise AssertionError("auction cannot finish without a purchased candidate")
    return _AuctionStateV1(
        state.round_index,
        state.cursor,
        state.records,
        candidate,
        state.recall_queue,
        state.recall_position,
        True,
        False,
        candidate.contract,
        candidate.buyer,
        candidate.trump,
    )


def _advance_auction(
    domain: AuctionDomainIRV1,
    state: _AuctionStateV1,
    *,
    dealer: Seat,
    purchase_suit: Suit,
    purchase_rank: Rank | None = None,
    action: str,
    trump: Suit | None,
) -> _AuctionStateV1:
    order = _auction_order(dealer)
    actor = _auction_actor(state, order)
    choices = _auction_legal_choices(
        domain,
        state,
        dealer=dealer,
        purchase_suit=purchase_suit,
        purchase_rank=purchase_rank,
    )
    if (action, trump) not in choices:
        raise ValueError("auction transition received an illegal action")
    resolution = domain.resolve_action(
        action,
        round_index=state.round_index,
        actor=actor,
        purchase_suit=purchase_suit,
        trump=trump,
    )
    record = OrToolsAuctionActionV1(
        state.round_index,
        actor,
        action,
        resolution.trump,
        resolution.contract,
        resolution.buyer,
        resolution.purchase_recipient,
        resolution.caller,
        resolution.declarer,
    )
    state = _AuctionStateV1(
        state.round_index,
        state.cursor,
        (*state.records, record),
        state.candidate,
        state.recall_queue,
        state.recall_position,
        False,
        False,
        None,
        None,
        None,
    )

    in_recall = bool(state.recall_queue) and state.recall_position < len(
        state.recall_queue
    )
    if in_recall:
        candidate = record if resolution.contract == Contract.SUN else state.candidate
        recall_position = state.recall_position + 1
        recalled = _AuctionStateV1(
            state.round_index,
            state.cursor,
            state.records,
            candidate,
            state.recall_queue,
            recall_position,
            False,
            False,
            None,
            None,
            None,
        )
        if recall_position >= len(state.recall_queue):
            if candidate is None:
                raise AssertionError("auction recall ended without a candidate")
            return _finish_auction(recalled, candidate)
        return recalled

    candidate = state.candidate
    cursor = state.cursor
    if state.round_index == 1:
        if resolution.contract == Contract.HOKUM:
            candidate = record
            cursor += 1
        elif resolution.contract in {Contract.SUN, Contract.ASHKAL}:
            candidate = record
            queue = tuple(order[index] for index in range(cursor - 1, -1, -1))
            recalled = _AuctionStateV1(
                1,
                cursor,
                state.records,
                candidate,
                queue,
                0,
                False,
                False,
                None,
                None,
                None,
            )
            return _finish_auction(recalled, candidate) if not queue else recalled
        else:
            cursor += 1
        progressed = _AuctionStateV1(
            1,
            cursor,
            state.records,
            candidate,
            (),
            0,
            False,
            False,
            None,
            None,
            None,
        )
        if cursor < 4:
            return progressed
        if candidate is None:
            if domain.rounds == 1:
                return _AuctionStateV1(
                    1,
                    cursor,
                    state.records,
                    None,
                    (),
                    0,
                    True,
                    True,
                    None,
                    None,
                    None,
                )
            return _AuctionStateV1(
                2, 0, state.records, None, (), 0, False, False, None, None, None
            )
        bidder_index = order.index(candidate.actor)
        queue = tuple(order[index] for index in range(bidder_index - 1, -1, -1))
        recalled = _AuctionStateV1(
            1,
            cursor,
            state.records,
            candidate,
            queue,
            0,
            False,
            False,
            None,
            None,
            None,
        )
        return _finish_auction(recalled, candidate) if not queue else recalled

    if resolution.contract in {Contract.SUN, Contract.ASHKAL}:
        return _finish_auction(state, record)
    if resolution.contract == Contract.HOKUM:
        candidate = record
    cursor += 1
    progressed = _AuctionStateV1(
        2,
        cursor,
        state.records,
        candidate,
        (),
        0,
        False,
        False,
        None,
        None,
        None,
    )
    if cursor < 4:
        return progressed
    if candidate is None:
        return _AuctionStateV1(
            2,
            cursor,
            state.records,
            None,
            (),
            0,
            True,
            True,
            None,
            None,
            None,
        )
    return _finish_auction(progressed, candidate)


@lru_cache(maxsize=128)
def _enumerate_auction(
    domain: AuctionDomainIRV1,
    *,
    dealer: Seat,
    purchase_suit: Suit,
    purchase_rank: Rank | None = None,
) -> tuple[
    tuple[_AuctionStateV1, ...],
    Mapping[tuple[tuple[int, int, str, int], ...], _AuctionStateV1],
]:
    terminal: list[_AuctionStateV1] = []
    states: dict[tuple[tuple[int, int, str, int], ...], _AuctionStateV1] = {}
    root = _initial_auction_state(
        domain,
        dealer=dealer,
        purchase_suit=purchase_suit,
    )

    def key(records: Sequence[OrToolsAuctionActionV1]):
        return tuple(
            (
                record.round_index,
                int(record.actor),
                record.action,
                int(record.trump) if record.trump is not None else _NO_TRUMP,
            )
            for record in records
        )

    def visit(state: _AuctionStateV1) -> None:
        states[key(state.records)] = state
        if state.terminal:
            terminal.append(state)
            return
        for action, trump in _auction_legal_choices(
            domain,
            state,
            dealer=dealer,
            purchase_suit=purchase_suit,
            purchase_rank=purchase_rank,
        ):
            visit(
                _advance_auction(
                    domain,
                    state,
                    dealer=dealer,
                    purchase_suit=purchase_suit,
                    purchase_rank=purchase_rank,
                    action=action,
                    trump=trump,
                )
            )

    visit(root)
    if domain.require_preselected:
        states[()] = root
    return tuple(terminal), states


def reify_scenario_expression(
    model: cp_model.CpModel,
    expression: ScenarioExpressionV1,
    predicate_literals: Mapping[str, cp_model.IntVar],
) -> cp_model.IntVar:
    """Return one exact Boolean for a closed scenario expression.

    Missing leaves are rejected before the model is changed. This prevents the
    legacy partial-``all`` behavior from becoming a qualification authority.
    """

    required = {item.predicate_id for item in expression.predicates()}
    missing = sorted(required - set(predicate_literals))
    if missing:
        raise ValueError(
            "scenario expression is missing predicate literals: " + ", ".join(missing)
        )

    serial = 0

    def next_name(prefix: str, node: ScenarioExpressionV1) -> str:
        nonlocal serial
        serial += 1
        return f"{prefix}_{node.sha256[:12]}_{serial}"

    def lower(node: ScenarioExpressionV1) -> cp_model.IntVar:
        if node.predicate is not None:
            return predicate_literals[node.predicate.predicate_id]

        children = [lower(child) for child in node.children]
        result = model.new_bool_var(next_name("scenario_expr", node))
        if node.op == "all":
            model.add_min_equality(result, children)
        elif node.op == "any":
            model.add_max_equality(result, children)
        elif node.op == "not":
            if len(children) != 1:
                raise ValueError("scenario not expression requires one child")
            model.add(result + children[0] == 1)
        elif node.op == "count":
            if node.minimum is None or node.maximum is None:
                raise ValueError("scenario count expression requires bounds")
            count = model.new_int_var(
                0,
                len(children),
                next_name("scenario_count", node),
            )
            model.add(count == sum(children))
            model.add_allowed_assignments(
                [count, result],
                [
                    (value, int(node.minimum <= value <= node.maximum))
                    for value in range(len(children) + 1)
                ],
            )
        else:
            raise ValueError(f"unsupported scenario expression operator: {node.op}")
        return result

    return lower(expression)


@dataclass(frozen=True, slots=True)
class OrToolsScenarioDealSolutionV1:
    status: str
    solver_status: str
    dealer: Seat
    buyer: Seat | None
    deck_order: tuple[int, ...] | None
    position_of_card: tuple[int, ...] | None
    owner_at_position: tuple[Seat, ...] | None
    owner_of_card: tuple[Seat, ...] | None
    purchase_position: int
    purchase_card: Card | None
    purchase_suit: Suit | None
    no_good_hashes: tuple[str, ...]
    minimum_deck_hamming_distance: int
    model_hash: str
    solution_hash: str | None
    branches: int
    conflicts: int
    auction: OrToolsScenarioAuctionObservationV1 | None = None
    purchase_recipient: Seat | None = None
    variable_count: int = 0
    constraint_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "status": self.status,
            "solverStatus": self.solver_status,
            "dealer": self.dealer.name.lower(),
            "buyer": self.buyer.name.lower() if self.buyer is not None else None,
            "deckOrder": list(self.deck_order) if self.deck_order is not None else None,
            "positionOfCard": (
                list(self.position_of_card)
                if self.position_of_card is not None
                else None
            ),
            "ownerAtPosition": (
                [seat.name.lower() for seat in self.owner_at_position]
                if self.owner_at_position is not None
                else None
            ),
            "ownerOfCard": (
                [seat.name.lower() for seat in self.owner_of_card]
                if self.owner_of_card is not None
                else None
            ),
            "purchasePosition": self.purchase_position,
            "purchaseCard": (
                self.purchase_card.code if self.purchase_card is not None else None
            ),
            "purchaseSuit": (
                self.purchase_suit.name.lower()
                if self.purchase_suit is not None
                else None
            ),
            "noGoodHashes": list(self.no_good_hashes),
            "minimumDeckHammingDistance": self.minimum_deck_hamming_distance,
            "modelHash": self.model_hash,
            "solutionHash": self.solution_hash,
            "branches": self.branches,
            "conflicts": self.conflicts,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
            "auction": self.auction.to_dict() if self.auction is not None else None,
        }
        if (
            self.purchase_recipient is not None
            and self.purchase_recipient != self.buyer
        ):
            value["purchaseRecipient"] = self.purchase_recipient.name.lower()
        return value


class OrToolsScenarioModelV1:
    """Integrated deal/purchase/ownership foundation for Research V4."""

    def __init__(
        self,
        *,
        rules: RuleBook,
        dealer: Seat | str | int,
        required_buyer: Seat | str | int | None = None,
        excluded_deck_orders: Sequence[Sequence[int]] = (),
        minimum_deck_hamming_distance: int = 1,
        canonicalize_deal_packets: bool = False,
    ) -> None:
        if not isinstance(rules, RuleBook):
            raise TypeError("OR-Tools scenario model requires a RuleBook")
        self.rules = rules
        self.dealer = Seat.parse(dealer)
        self.required_buyer = (
            Seat.parse(required_buyer) if required_buyer is not None else None
        )
        if type(canonicalize_deal_packets) is not bool:
            raise TypeError("scenario deal-packet canonicalization must be boolean")
        self.canonicalize_deal_packets = canonicalize_deal_packets
        self.excluded_deck_orders = self._normalize_decks(excluded_deck_orders)
        if (
            type(minimum_deck_hamming_distance) is not int
            or not 1 <= minimum_deck_hamming_distance <= 32
        ):
            raise ValueError("scenario deck Hamming distance must be in 1..32")
        self.minimum_deck_hamming_distance = minimum_deck_hamming_distance
        self.no_good_hashes = tuple(
            canonical_sha256(list(deck)) for deck in self.excluded_deck_orders
        )

        domain = rules.structural_domain.require_integrity()
        layouts = {
            buyer: DealLayoutV1.build(
                domain,
                dealer=self.dealer,
                buyer=buyer,
            )
            for buyer in Seat
        }
        purchase_positions = {layout.purchase_position for layout in layouts.values()}
        if len(purchase_positions) != 1:
            raise AssertionError("purchase position must not depend on the buyer")
        self.purchase_position = purchase_positions.pop()

        self.model = cp_model.CpModel()
        self.card_at_position = [
            self.model.new_int_var(0, 31, f"card_at_position_{position}")
            for position in range(32)
        ]
        self.position_of_card = [
            self.model.new_int_var(0, 31, f"position_of_card_{card}")
            for card in range(32)
        ]
        self.model.add_inverse(self.card_at_position, self.position_of_card)

        if self.canonicalize_deal_packets:
            layout = layouts[self.required_buyer if self.required_buyer is not None else Seat.NORTH]
            groups = []
            for seat in Seat:
                initial = layout.positions_for(seat, final=False)
                groups.append(initial)
                final_additions = tuple(
                    position
                    for position in layout.positions_for(seat, final=True)
                    if position not in initial and position != self.purchase_position
                )
                if self.required_buyer is not None:
                    groups.append(final_additions)
            if self.required_buyer is None:
                groups.extend(layout.purchase_invariant_undealt_groups(domain))
            for group in groups:
                for left, right in zip(group, group[1:]):
                    self.model.add(self.card_at_position[left] < self.card_at_position[right])

        self.buyer = self.model.new_int_var(0, 3, "buyer")
        if self.required_buyer is not None:
            self.model.add(self.buyer == int(self.required_buyer))

        owners_by_recipient: dict[Seat, tuple[int, ...]] = {}
        for recipient, layout in layouts.items():
            owner_by_position = [-1] * 32
            for seat in Seat:
                for position in layout.positions_for(seat, final=True):
                    owner_by_position[position] = int(seat)
            if any(owner < 0 for owner in owner_by_position):
                raise AssertionError("final deal layout must own every position")
            owners_by_recipient[recipient] = tuple(owner_by_position)

        self.owner_at_position = [
            self.model.new_int_var(0, 3, f"owner_at_position_{position}")
            for position in range(32)
        ]
        self.purchase_recipient = self.model.new_int_var(0, 3, "purchase_recipient")
        self._owners_by_recipient = owners_by_recipient
        self._ownership_bound = False

        self.owner_of_card = [
            self.model.new_int_var(0, 3, f"owner_of_card_{card}") for card in range(32)
        ]
        for card, owner in enumerate(self.owner_of_card):
            self.model.add_element(
                self.position_of_card[card],
                self.owner_at_position,
                owner,
            )

        self.purchase_card = self.card_at_position[self.purchase_position]
        self.purchase_suit = self.model.new_int_var(0, 3, "purchase_suit")
        self.model.add_element(
            self.purchase_card,
            [int(Card(card).suit) for card in range(32)],
            self.purchase_suit,
        )
        self.purchase_rank = self.model.new_int_var(0, 7, "purchase_rank")
        self.model.add_element(
            self.purchase_card,
            [int(Card(card).rank) for card in range(32)],
            self.purchase_rank,
        )

        for cut_index, excluded in enumerate(self.excluded_deck_orders):
            self.add_deck_distance_cut(excluded, _name=f"deck_cut_{cut_index}")

        self.model.add_decision_strategy(
            [self.buyer, *self.card_at_position],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )
        self.auction_fragment: OrToolsScenarioAuctionFragmentV1 | OrToolsNativeAuctionFragmentV1 | None = None
        self._observed_auction_bindings: dict[str, dict[str, Any]] = {}
        self._observed_escalation_bindings: dict[str, dict[str, Any]] = {}
        self._observed_project_bindings: dict[str, dict[str, Any]] = {}
        self._observed_play_bindings: dict[str, dict[str, Any]] = {}
        self._complete_information_deck: tuple[int, ...] | None = None
        self._recorded_escalation_count = 0
        self._recorded_project_count = 0
        self._recorded_play_count = 0
        self._native_play_root = None
        self._native_project_root = None
        self._native_escalation_root = None
        self._native_escalation_root_position = None
        self._native_auction_policy_root = None
        self._native_play_source_query = None
        self._native_round_fragment: OrToolsNativeRoundFragmentV1 | None = None
        self._native_redeal_scope: Mapping[str, Any] | None = None
        self._native_redeal_awarded_points: tuple[Any, ...] | None = None
        self._bind_purchase_recipient_ownership()
        self._recipient_authority_bound = False
        auction_domain = AuctionDomainIRV1.from_rules(self.rules.data)
        if auction_domain.ashkal_contract_owner == "taker":
            self.model.add(self.purchase_recipient == self.buyer)
            self._recipient_authority_bound = True

    def _bind_purchase_recipient_ownership(self) -> None:
        if self._ownership_bound:
            return
        for position, owner in enumerate(self.owner_at_position):
            self.model.add_allowed_assignments(
                [self.purchase_recipient, owner],
                [
                    (int(recipient), owners[position])
                    for recipient, owners in self._owners_by_recipient.items()
                ],
            )
        self._ownership_bound = True

    def add_deck_distance_cut(self, deck, *, _name=None) -> None:
        """Enumerate a different proposal in this model; not an UNSAT claim."""
        normalized, = self._normalize_decks((deck,))
        name = _name or f"deck_distance_{len(self.model.proto.constraints)}"
        matches = []
        for position, card in enumerate(normalized):
            match = self.model.new_bool_var(f"{name}_position_{position}_matches")
            self.model.add(self.card_at_position[position] == card).only_enforce_if(match)
            self.model.add(self.card_at_position[position] != card).only_enforce_if(match.Not())
            matches.append(match)
        self.model.add(sum(matches) <= 32-self.minimum_deck_hamming_distance)

    @staticmethod
    def _normalize_decks(
        values: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        decks: set[tuple[int, ...]] = set()
        for index, raw in enumerate(values):
            if any(type(card) is not int for card in raw):
                raise TypeError(f"scenario no-good deck {index} must contain integers")
            deck = tuple(raw)
            if len(deck) != 32 or sorted(deck) != list(range(32)):
                raise ValueError(
                    f"scenario no-good deck {index} is not a permutation of 0..31"
                )
            decks.add(deck)
        return tuple(sorted(decks))

    @staticmethod
    def _normalize_auction_prefix(
        values: Sequence[OrToolsAuctionActionV1 | Mapping[str, Any]],
        *,
        domain: AuctionDomainIRV1,
    ) -> tuple[_RequiredAuctionActionV1, ...]:
        active_actions = {
            *domain.first_round_allowed,
            *domain.second_round_allowed,
        }
        normalized: list[_RequiredAuctionActionV1] = []
        for index, raw in enumerate(values, start=1):
            if isinstance(raw, OrToolsAuctionActionV1):
                actor = raw.actor
                action = raw.action
                trump = raw.trump
                trump_specified = True
                round_index = raw.round_index
            else:
                item = dict(raw)
                unknown = set(item) - {
                    "ordinal",
                    "roundIndex",
                    "actor",
                    "action",
                    "trumpSuit",
                }
                if unknown:
                    raise ValueError(
                        f"auction prefix action {index} has unknown fields: "
                        + ", ".join(sorted(unknown))
                    )
                if item.get("ordinal") is not None and item["ordinal"] != index:
                    raise ValueError("auction prefix ordinals must be contiguous")
                if item.get("action") is None:
                    raise ValueError("auction prefix actions require an action")
                actor = (
                    Seat.parse(item["actor"]) if item.get("actor") is not None else None
                )
                action = str(item["action"]).strip().lower()
                trump_specified = "trumpSuit" in item
                trump = (
                    Suit.parse(item["trumpSuit"])
                    if item.get("trumpSuit") is not None
                    else None
                )
                round_index = (
                    int(item["roundIndex"])
                    if item.get("roundIndex") is not None
                    else None
                )
            if action not in active_actions:
                raise ValueError(f"auction prefix action {action!r} is inactive")
            if round_index not in {None, 1, 2}:
                raise ValueError("auction prefix roundIndex must be one or two")
            if action != "hokum" and trump_specified and trump is not None:
                raise ValueError("only a Hokum auction action may specify trump")
            normalized.append(
                _RequiredAuctionActionV1(
                    actor,
                    action,
                    trump,
                    trump_specified,
                    round_index,
                )
            )
        return tuple(normalized)

    def add_native_auction_fragment(
        self, *, required_actions: Sequence[OrToolsAuctionActionV1 | Mapping[str, Any]] = (),
        required_actions_complete: bool = False,
        required_purchase_card: Card | int | str | Mapping[str, Any] | None = None,
        allowed_contracts: Sequence[Contract | str] | None = None,
        prefix_length: int | None = None,
        require_optimal_actions: bool = False,
        allow_all_passed: bool = False,
    ) -> OrToolsNativeAuctionFragmentV1:
        """Bind deal ownership and exact-prefix observations to native transitions."""
        if self.auction_fragment is not None:
            raise ValueError("scenario model already contains an auction fragment")
        if type(required_actions_complete) is not bool:
            raise TypeError("required_actions_complete must be boolean")
        if type(require_optimal_actions) is not bool:
            raise TypeError("require_optimal_actions must be boolean")
        if type(allow_all_passed) is not bool:
            raise TypeError("allow_all_passed must be boolean")
        if allow_all_passed and self.required_buyer is not None:
            raise ValueError("all-pass auction cannot require a buyer")
        if isinstance(required_purchase_card, bool):
            raise TypeError("required_purchase_card must be a card")
        domain = AuctionDomainIRV1.from_rules(self.rules.data)
        prefix = self._normalize_auction_prefix(required_actions, domain=domain)
        prefer_live = prefix_length is not None and not prefix and not required_actions_complete
        count = len(prefix) if prefix_length is None else prefix_length
        if type(count) is not int or not len(prefix) <= count <= 2 * len(Seat):
            raise ValueError("native auction prefix length must cover actions and fit eight steps")
        if required_actions_complete and count != len(prefix):
            raise ValueError("a complete auction requires its exact action length")
        if domain.require_preselected:
            count = max(count, 3 if domain.default_contract == Contract.ASHKAL and domain.ashkal_taker == "bidder_partner" else 1)
        if required_purchase_card is not None:
            self.model.add(self.purchase_card == Card.parse(required_purchase_card).id)
        legal_rows: list[Mapping[str, cp_model.IntVar]] = []
        states, choices = attach_auction_path_v1(self.model, domain,
            dealer=self.model.new_constant(int(self.dealer)), purchase_suit=self.purchase_suit,
            purchase_rank=self.purchase_rank, maximum_actions=2 * len(Seat),
            legal_rows=legal_rows)
        if require_optimal_actions:
            attach_optimal_auction_choices_v1(
                self.model, self.rules, position_of_card=self.position_of_card,
                purchase_position=self.purchase_position, states=states,
                choices=choices, legal_rows=legal_rows, dealer=self.dealer,
            )
        final = states[-1]
        self.model.add(final["terminal"] == 1)
        if not allow_all_passed:
            self.model.add(final["contract"] >= 0)
        contracts = tuple(Contract) if allowed_contracts is None else tuple(Contract.parse(value) for value in allowed_contracts)
        self.model.add_allowed_assignments([final["contract"]],
            [(_CONTRACT_IDS[value],) for value in contracts] + ([(-1,)] if allow_all_passed else []))
        for index in range(count):
            self.model.add(choices[index]["closed"] == 0)
        for index, action in enumerate(prefix):
            matches = [literal for code, literal in choices[index].items()
                       if code.split(":")[0] == action.action
                       and (not action.trump_specified or action.action != "hokum"
                            or (action.trump is not None and code == f"hokum:{action.trump.name.lower()}"))]
            self.model.add(sum(matches) == 1)
            if action.actor is not None:
                self.model.add(states[index]["actor"] == int(action.actor))
            if action.round_index is not None:
                self.model.add(states[index]["round"] == action.round_index)
        state = states[count]
        if required_actions_complete:
            self.model.add(state["terminal"] == 1)
        if prefer_live:
            self.model.add(state["terminal"] == 0)
        legal_actions, legal_choices = attach_auction_legal_actions_v1(self.model, domain,
            round_index=state["round"], position=state.get("position", self.model.new_constant(1)),
            in_recall=state.get("inRecall", self.model.new_constant(0)), terminal=state["terminal"],
            candidate_contract=state["contract"], purchase_rank=self.purchase_rank,
            purchase_suit=self.purchase_suit, prior_nonpass_count=state["nonpass"], name="auction_observation")
        if allow_all_passed:
            self.model.add(self.buyer == final["buyer"]).only_enforce_if(final["allPassed"].Not())
            self.model.add(self.purchase_recipient == final["recipient"]).only_enforce_if(final["allPassed"].Not())
            # These allocation variables are unused when no final deal occurs.
            self.model.add(self.buyer == 0).only_enforce_if(final["allPassed"])
            self.model.add(self.purchase_recipient == 0).only_enforce_if(final["allPassed"])
        else:
            self.model.add(self.buyer == final["buyer"])
            self.model.add(self.purchase_recipient == final["recipient"])
        self._recipient_authority_bound = True
        self._bind_purchase_recipient_ownership()
        fragment = OrToolsNativeAuctionFragmentV1(states, choices, count, legal_actions, legal_choices,
            domain.sha256, public_prefix_bound=count == len(prefix),
            legal_choices_by_step=tuple(legal_rows))
        self.auction_fragment = fragment
        return fragment

    def native_auction_observation_literal(
        self, *, actor: Seat, visible_cards: Sequence[Card | str | int], purchase_card: Card | str | int,
        public_choices: Sequence[str] | None = None,
    ) -> cp_model.IntVar:
        """Match private cards at a fixed root or explicit later public history.

        This is an observation constraint, not a strategy or optimality claim.
        Future actions and other seats' cards are deliberately absent.
        """
        fragment = self.auction_fragment
        if not isinstance(fragment, OrToolsNativeAuctionFragmentV1):
            raise ValueError("observation matching requires a native auction fragment")
        if not fragment.public_prefix_bound:
            raise ValueError("observation matching requires a fixed public auction prefix")
        history = None if public_choices is None else tuple(public_choices)
        index = fragment.prefix_length if history is None else len(history)
        if history is not None and (
            isinstance(public_choices, (str, bytes))
            or not fragment.prefix_length <= index < len(fragment.choices)
            or any(type(code) is not str or code == "closed" or code not in fragment.choices[i]
                   for i, code in enumerate(history))
        ):
            raise ValueError("observation requires a valid explicit public auction history")
        actor = Seat.parse(actor)
        cards = tuple(sorted(Card.parse(card).id for card in visible_cards))
        purchase = Card.parse(purchase_card)
        layout = DealLayoutV1.build(self.rules.structural_domain, dealer=self.dealer)
        positions = layout.positions_for(actor, final=False)
        if len(cards) != len(positions) or len(set(cards)) != len(cards) or purchase.id in cards:
            raise ValueError("auction observation requires a unique initial hand and separate purchase card")
        name = "auction_observation_" + canonical_sha256({
            "actor": int(actor), "cards": cards, "purchase": purchase.id,
            "prefixLength": fragment.prefix_length,
            "publicChoices": history,
        })[:20]
        equal, conjunction, _ = _auction_logic_v1(self.model, name)
        state = fragment.states[index]
        terms = [equal(state["actor"], int(actor), "actor"),
                 equal(state["terminal"], 0, "live"),
                 equal(self.purchase_card, purchase.id, "purchase")]
        if history is not None:
            terms.extend(fragment.choices[i][code] for i, code in enumerate(history))
        terms.extend(self._initial_hand_literals(actor, cards, name))
        return conjunction("matches", *terms)

    def _initial_hand_literals(self, actor, cards, name):
        positions = DealLayoutV1.build(self.rules.structural_domain, dealer=self.dealer).positions_for(actor, final=False)
        terms = []
        for card in cards:
            owns = self.model.new_bool_var(f"{name}_owns_{card}")
            self.model.add_allowed_assignments([self.position_of_card[card], owns],
                [(position, int(position in positions)) for position in self.rules.structural_domain.card_ids])
            terms.append(owns)
        return terms

    def bind_native_auction_observed_choice(
        self, *, actor: Seat, visible_cards: Sequence[Card | str | int],
        purchase_card: Card | str | int, action: str,
        public_choices: Sequence[str],
        maximum_seconds: float, maximum_conflicts: int,
        cancel_check: Callable[[], bool] | None = None,
        _prefix_legality: bool = False,
    ) -> dict[str, Any]:
        """Bind one observation-local action only after native legality proof.

        A bound receipt is neither reachability nor strategic coverage evidence.
        Unknown/cancelled legality queries never install the action constraint.
        """
        if type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("policy binding requires a finite positive time budget")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("policy binding requires a positive conflict budget")
        if type(_prefix_legality) is not bool:
            raise TypeError("auction prefix legality must be bool")
        deadline = time.monotonic() + maximum_seconds
        fragment = self.auction_fragment
        if not isinstance(fragment, OrToolsNativeAuctionFragmentV1) or not fragment.public_prefix_bound:
            raise ValueError("policy binding requires a fixed native auction prefix")
        if action not in fragment.legal_next_choice_literals:
            raise ValueError("policy binding action is unsupported at this auction root")
        if isinstance(public_choices, (str, bytes)):
            raise ValueError("observation requires a valid explicit public auction history")
        history = tuple(public_choices)
        cards = tuple(visible_cards)
        observation_hash = self._auction_observation_hash(actor, cards, purchase_card, history)
        previous = self._observed_auction_bindings.get(observation_hash)
        if previous is not None:
            if previous["action"] != action:
                raise ValueError("conflicting actions for one auction observation")
            return dict(previous)
        matches = self.native_auction_observation_literal(
            actor=actor, visible_cards=cards, purchase_card=purchase_card,
            public_choices=history)
        index = len(history)
        legal_choices = fragment.legal_choices_by_step[index]
        probe = None
        if _prefix_legality:
            # Auction legality depends on the public path and purchase rank/suit,
            # not future deals, projects or play. This broader universe can only
            # certify a sufficient legality bound, never reachability or utility.
            purchase = Card.parse(purchase_card)
            probe, rows = cp_model.CpModel(), []
            states, choices = attach_auction_path_v1(probe, AuctionDomainIRV1.from_rules(self.rules.data),
                dealer=probe.new_constant(int(self.dealer)),
                purchase_suit=probe.new_constant(int(purchase.suit)),
                purchase_rank=probe.new_constant(int(purchase.rank)),
                maximum_actions=index + 1, legal_rows=rows)
            for step, code in enumerate(history):
                probe.add(choices[step][code] == 1)
            probe.add(states[index]["actor"] == int(Seat.parse(actor)))
            probe.add(states[index]["terminal"] == 0)
            probe.add(rows[index][action] == 0)
        return self._bind_native_policy_choice(observation_hash=observation_hash, action=action,
            choices=((matches, fragment.choices[index][action], legal_choices[action]),),
            kind="auction", bindings=self._observed_auction_bindings, deadline=deadline,
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, legality_probe=probe)

    def _bind_native_policy_choice(self, *, observation_hash, action, choices, kind,
            bindings, deadline, maximum_conflicts, cancel_check, record_model_bytes=False, record_bound_model_hash=False,
            legality_probe=None, _defer_owner_hash=False):
        if type(record_model_bytes) is not bool:
            raise TypeError("native model-byte recording must be boolean")
        if type(record_bound_model_hash) is not bool:
            raise TypeError("native policy model-hash recording must be boolean")
        if (type(_defer_owner_hash) is not bool or _defer_owner_hash
                and (kind != "play" or record_model_bytes or record_bound_model_hash)):
            raise ValueError("deferred owner hashing requires a play binding batch")
        if legality_probe is not None and (kind not in {"play", "auction"} or not isinstance(legality_probe, cp_model.CpModel)):
            raise ValueError("relaxed legality probes require a native play or auction model")
        probe = legality_probe
        if probe is None:
            probe = self.model.clone()
            _, both, _ = _auction_logic_v1(probe, kind + "_policy_legality")
            failures = []
            prior_valid = probe.new_constant(1) if kind == "constant_auction" and action != "pass" else None
            for index, (matches, selected, legal) in enumerate(choices):
                if prior_valid is None:
                    failures.append(both(str(index), matches, legal.Not()))
                    continue
                # An earlier purchase restricts later auction histories. Do not
                # reject this policy using a history that its own choice excludes.
                # Stop enforcing after the first illegal opportunity so the
                # probe can still witness that failure instead of hiding it.
                failure = both(str(index), prior_valid, matches, legal.Not())
                probe.add(selected == 1).only_enforce_if([prior_valid, matches, legal])
                prior_valid = both(f"prior_valid_{index}", prior_valid, failure.Not())
                failures.append(failure)
            probe.add_bool_or(failures)
        probe_proto = str(probe.proto).encode("utf-8")
        probe_hash, probe_bytes = hashlib.sha256(probe_proto).hexdigest(), len(probe_proto)
        del probe_proto
        cache = getattr(self, "_native_policy_legality_cache", None)
        cache_key = (self.rules.sha256, probe_hash)
        cached = None if cache is None else cache.get(cache_key)
        if cached is not None and (not isinstance(cached, Mapping) or cached.get("status") != "bound"
                or cached.get("solverStatus") != "INFEASIBLE"
                or cached.get("rulesHash") != self.rules.sha256
                or cached.get("legalityModelHash") != probe_hash
                or cached.get("sha256") != canonical_sha256(
                    {key: value for key, value in cached.items() if key != "sha256"})):
            raise ValueError("native policy legality cache identity mismatch")
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        remaining = deadline - time.monotonic()
        status = cp_model.UNKNOWN
        if remaining > 0:
            solver.parameters.max_time_in_seconds = remaining
            status = cp_model.INFEASIBLE if cached is not None else _solve_cp_sat_with_cancel_v1(
                solver, probe, cancel_check)
        if status == cp_model.MODEL_INVALID:
            raise ValueError("native policy legality model is invalid")
        if time.monotonic() >= deadline or (cancel_check is not None and cancel_check()):
            status = cp_model.UNKNOWN
        binding_status = "bound" if status == cp_model.INFEASIBLE else (
            "invalid" if legality_probe is None and status in {cp_model.FEASIBLE, cp_model.OPTIMAL} else "incomplete")
        if binding_status == "bound":
            for matches, selected, _ in choices:
                self.model.add(selected == 1).only_enforce_if(matches)
        owner_hash = None
        if not _defer_owner_hash:
            owner_proto = str(self.model.proto).encode("utf-8")
            owner_hash, owner_bytes = hashlib.sha256(owner_proto).hexdigest(), len(owner_proto)
            del owner_proto
        # The refiner consumes this digest immediately, before another mutation.
        self._last_policy_binding_proto_hash = owner_hash
        core = {"schemaVersion": 1, "kind": f"native_{kind}_policy_binding_v1",
            "status": binding_status, "observationHash": observation_hash, "action": action,
            "rulesHash": self.rules.sha256, "solverStatus": solver.status_name(status),
            "legalityModelHash": probe_hash}
        if owner_hash is not None:
            core["modelHash"] = self._model_hash(search_branching="fixed", proto_hash=owner_hash)
        if legality_probe is not None:
            core["legalityScope"] = f"actor_visible_{kind}_prefix_relaxation_v1"
        if record_bound_model_hash:
            core["boundModelHash"] = self._model_hash(search_branching="automatic", proto_hash=owner_hash)
        if record_model_bytes:
            core["modelBytes"] = {"unit": "utf8_cp_model_proto",
                "owner": owner_bytes, "legalityProbe": probe_bytes}
        receipt = {**core, "sha256": canonical_sha256(core)}
        if binding_status == "bound":
            bindings[observation_hash] = receipt
            if cache is not None:
                # Execution-local exact queries only; fresh readback receives no cache.
                cache[cache_key] = dict(receipt)
        return dict(receipt)

    def _auction_observation_hash(self, actor, cards, purchase, history):
        return canonical_sha256({"actor": int(Seat.parse(actor)),
            "cards": sorted(Card.parse(card).id for card in cards),
            "purchase": Card.parse(purchase).id, "publicChoices": tuple(history),
            **({"completeDeck": list(self._complete_information_deck)}
               if self._complete_information_deck is not None else {})})

    def _extract_native_auction_observations(self, solver):
        """Read each bidder's information and legal choices, never a full hand."""
        fragment = self.auction_fragment
        if not isinstance(fragment, OrToolsNativeAuctionFragmentV1):
            raise ValueError("auction observation extraction requires a native fragment")
        domain = AuctionDomainIRV1.from_rules(self.rules.data)
        if domain.require_preselected:
            return []  # Automatic profile actions are not player decisions.
        layout = DealLayoutV1.build(self.rules.structural_domain, dealer=self.dealer)
        purchase = solver.value(self.purchase_card)
        history, observations = [], []
        for index, action in enumerate(fragment.extract_actions(solver)):
            code = action.action + (f":{action.trump.name.lower()}" if action.trump is not None else "")
            cards = sorted(solver.value(self.card_at_position[position])
                           for position in layout.positions_for(action.actor, final=False))
            observations.append({"decisionIndex": index, "actor": action.actor.name.lower(),
                "visibleCards": cards, "purchaseCard": purchase, "publicChoices": list(history),
                "legalActions": sorted(key for key, literal in fragment.legal_choices_by_step[index].items()
                                       if solver.boolean_value(literal)),
                "selectedAction": code, "authoredPrefix": index < fragment.prefix_length,
                "observationHash": self._auction_observation_hash(action.actor, cards, purchase, history)})
            history.append(code)
        return observations

    def _normalize_escalation_information(self, information, *, completed=False):
        from .ortools_nonplay_graphs import EscalationGraphActionV1, _escalation_window_key

        if self._native_round_fragment is None:
            raise ValueError("escalation observation requires a native round branch")
        fields = {"actor", "initialCards", "visibleCards", "purchaseCard", "publicChoices",
                  "publicAnnouncements", "ownDeclinedWindows"}
        if self._complete_information_deck is not None:
            fields.add("completePriorWindowDeclines")
            fields.add("completeWindowDeclineOrder")
        if not isinstance(information, Mapping) or set(information) != fields:
            raise ValueError("escalation information must contain only actor-visible fields")
        for key in fields - {"actor", "purchaseCard", "completePriorWindowDeclines"}:
            if not isinstance(information[key], (list, tuple)):
                raise ValueError("escalation information sequences must be lists or tuples")
        if type(information["actor"]) not in {Seat, str, int}:
            raise ValueError("escalation observation actor is invalid")
        actor = Seat.parse(information["actor"])
        if any(type(card) not in {Card, str, int} for card in
               [*information["initialCards"], *information["visibleCards"], information["purchaseCard"]]):
            raise ValueError("escalation observation cards are invalid")
        initial = sorted(Card.parse(card).id for card in information["initialCards"])
        cards = sorted(Card.parse(card).id for card in information["visibleCards"])
        purchase = Card.parse(information["purchaseCard"]).id
        layout = DealLayoutV1.build(self.rules.structural_domain, dealer=self.dealer)
        if (len(initial) != len(layout.positions_for(actor, final=False))
                or len(cards) != self.rules.structural_domain.cards_per_player
                or len(set(initial)) != len(initial) or len(set(cards)) != len(cards)
                or not set(initial) <= set(cards) or purchase in initial):
            raise ValueError("escalation observation requires unique initial and final hands")
        history = information["publicChoices"]
        auction = self.auction_fragment
        if (not 0 < len(history) <= len(auction.choices)
                or any(type(code) is not str or code == "closed" or code not in auction.choices[index]
                       for index, code in enumerate(history))):
            raise ValueError("escalation observation requires a complete public auction history")
        announcements = information["publicAnnouncements"]
        native = self._native_round_fragment.escalation.native
        if len(announcements) > len(native["actionSlots"]):
            raise ValueError("escalation observation has too many announcements")
        parsed = []
        for row in announcements:
            if (not isinstance(row, (list, tuple)) or len(row) != 3
                    or type(row[0]) not in {Seat, str, int} or type(row[1]) is not str
                    or row[1] not in native["actions"]
                    or (row[2] not in ("open", "locked") if row[1] in {"double", "four"} else row[2] is not None)):
                raise ValueError("escalation observation announcement is invalid")
            parsed.append(EscalationGraphActionV1(Seat.parse(row[0]), row[1], row[2]))
        windows = [_escalation_window_key(parsed[:index]) for index in range(len(parsed) + int(completed))]
        recall = information["ownDeclinedWindows"]
        if (any(type(value) is not str for value in recall)
                or list(recall) != [value for value in windows if value in recall]):
            raise ValueError("escalation observation private recall must match prior public windows")
        complete_recall = {}
        if self._complete_information_deck is not None:
            raw = information["completePriorWindowDeclines"]
            if (not isinstance(raw, Mapping) or set(raw) != {seat.name.lower() for seat in Seat}
                    or any(not isinstance(rows, (list, tuple))
                           or any(type(value) is not str for value in rows)
                           or list(rows) != [window for window in windows if window in rows]
                           for rows in raw.values())
                    or list(raw[actor.name.lower()]) != list(recall)):
                raise ValueError("complete escalation recall must match prior windows and own recall")
            complete_recall = {"completePriorWindowDeclines": {seat.name.lower(): list(raw[seat.name.lower()]) for seat in Seat}}
            order = information["completeWindowDeclineOrder"]
            seats = {seat.name.lower() for seat in Seat}
            if (len(order) != len(parsed) + 1
                    or any(not isinstance(rows, (list, tuple))
                           or any(type(seat) is not str or seat not in seats for seat in rows)
                           or len(rows) != len(set(rows)) for rows in order)
                    or any(parsed[index].actor.name.lower() in rows for index, rows in enumerate(order[:-1]))
                    or (not completed and actor.name.lower() in order[-1])
                    or any(list(raw[seat]) != [window for index, window in enumerate(windows)
                                              if seat in order[index]] for seat in seats)):
                raise ValueError("complete escalation order must match observed declines")
            complete_recall["completeWindowDeclineOrder"] = [list(rows) for rows in order]
        return {"actor": actor.name.lower(), "initialCards": initial, "visibleCards": cards,
            "purchaseCard": purchase, "publicChoices": list(history),
            "publicAnnouncements": [[row.actor.name.lower(), row.action, row.mode] for row in parsed],
            "ownDeclinedWindows": list(recall), **complete_recall}, windows

    def _escalation_observation_hash(self, information):
        scope = self._native_round_fragment.scope
        # Final lock mode, opportunity order, hidden cards, and selected/future
        # actions are not known at this decision and must not enter its key.
        return canonical_sha256({"kind": "escalation", "rulesHash": self.rules.sha256,
            "dealer": self.dealer.name.lower(), "contract": scope["contract"],
            "trumpSuit": scope["trumpSuit"], "matchScore": scope["matchScore"],
            **({"completeDeck": list(self._complete_information_deck)}
               if self._complete_information_deck is not None else {}), **information})

    def _extract_native_escalation_observations(self, solver, opportunities):
        layout = DealLayoutV1.build(self.rules.structural_domain, dealer=self.dealer)
        history = [row.action + (f":{row.trump.name.lower()}" if row.trump is not None else "")
                   for row in self.auction_fragment.extract_actions(solver)]
        observations = []
        for ordinal, row in enumerate(opportunities):
            actor = Seat.parse(row["actor"])
            information, _ = self._normalize_escalation_information({
                "actor": row["actor"], "initialCards": [solver.value(self.card_at_position[position])
                    for position in layout.positions_for(actor, final=False)],
                "visibleCards": [card for card, owner in enumerate(self.owner_of_card)
                                 if solver.value(owner) == int(actor)],
                "purchaseCard": solver.value(self.purchase_card), "publicChoices": history,
                "publicAnnouncements": row["publicAnnouncements"], "ownDeclinedWindows": row["ownDeclinedWindows"],
                **({"completePriorWindowDeclines": {seat.name.lower(): [prior["windowHash"] for prior in opportunities
                    if prior["window"] < row["window"] and prior["actor"] == seat.name.lower()
                    and prior["selectedAction"] == self._native_round_fragment.escalation.stage.domain.closure_action]
                    for seat in Seat}, "completeWindowDeclineOrder": [
                        [prior["actor"] for prior in opportunities[:ordinal] if prior["window"] == window
                         and prior["selectedAction"] == self._native_round_fragment.escalation.stage.domain.closure_action]
                        for window in range(row["window"] + 1)]} if self._complete_information_deck is not None else {})})
            observations.append({"actor": row["actor"], "information": information,
                "observationHash": self._escalation_observation_hash(information),
                **({"authoredPrefix": True} if self._native_play_root is not None or self._native_project_root is not None
                    or (row["window"] * len(Seat) + row["ordinal"] < solver.value(self._native_escalation_root_position)
                        if self._native_escalation_root_position is not None else ordinal < self._recorded_escalation_count) else {}),
                "legalActions": list(row["legalActions"]), "selectedAction": row["selectedAction"]})
        return observations

    def _native_escalation_information_literal(self, info, windows, name):
        """Match own cards, public announcements, and private recall in one owner."""
        branch, auction = self._native_round_fragment, self.auction_fragment
        slots, native = branch.escalation_opportunities, branch.escalation.native
        equal, both, either = _auction_logic_v1(self.model, name)
        actor = Seat.parse(info["actor"])
        terms = self._initial_hand_literals(actor, info["initialCards"], name)
        terms.extend(equal(self.owner_of_card[card], int(actor), f"card_{card}") for card in info["visibleCards"])
        terms.append(equal(self.purchase_card, info["purchaseCard"], "purchase"))
        terms.extend(auction.choices[index][code] for index, code in enumerate(info["publicChoices"]))
        terms.append(equal(auction.states[len(info["publicChoices"])]["terminal"], 1, "auction_terminal"))
        for index, (seat, announcement, mode) in enumerate(info["publicAnnouncements"]):
            terms.extend((equal(native["actionSlots"][index][0], int(Seat.parse(seat)), f"actor_{index}"),
                          equal(native["actionSlots"][index][1], native["actions"][announcement], f"action_{index}")))
            if mode is not None:
                terms.append(equal(native["stateLocked"][index + 1], int(mode == "locked"), f"mode_{index}"))
        for index, window_hash in enumerate(windows):
            declined = either(f"recall_{index}", *(both(f"own_{index}_{slot['ordinal']}", slot["declined"],
                equal(slot["actor"], int(actor), f"recall_actor_{index}_{slot['ordinal']}"))
                for slot in slots if slot["window"] == index))
            terms.append(declined if window_hash in info["ownDeclinedWindows"] else declined.Not())
            if self._complete_information_deck is not None:
                for seat in Seat:
                    if seat == actor:
                        continue
                    other = either(f"complete_recall_{index}_{int(seat)}", *(
                        both(f"complete_{index}_{int(seat)}_{slot['ordinal']}", slot["declined"],
                             equal(slot["actor"], int(seat), f"complete_actor_{index}_{int(seat)}_{slot['ordinal']}"))
                        for slot in slots if slot["window"] == index))
                    terms.append(other if window_hash in info["completePriorWindowDeclines"][seat.name.lower()] else other.Not())
        if self._complete_information_deck is not None:
            for window, order in enumerate(info["completeWindowDeclineOrder"]):
                window_slots = [slot for slot in slots if slot["window"] == window]
                positions = {}
                for seat in Seat:
                    position = self.model.new_int_var(0, len(window_slots) - 1, f"{name}_order_{window}_{int(seat)}")
                    self.model.add_element(position, [slot["actor"] for slot in window_slots], int(seat))
                    positions[seat.name.lower()] = position
                for index, (first, second) in enumerate(zip(order, order[1:])):
                    before = self.model.new_bool_var(f"{name}_ordered_{window}_{index}")
                    self.model.add(positions[first] < positions[second]).only_enforce_if(before)
                    self.model.add(positions[first] >= positions[second]).only_enforce_if(before.Not())
                    terms.append(before)
                if window == len(windows):  # Only the current, unfinished window.
                    for seat in Seat:
                        before = self.model.new_bool_var(f"{name}_prior_{window}_{int(seat)}")
                        self.model.add(positions[seat.name.lower()] < positions[actor.name.lower()]).only_enforce_if(before)
                        self.model.add(positions[seat.name.lower()] >= positions[actor.name.lower()]).only_enforce_if(before.Not())
                        declined = either(f"ordered_decline_{window}_{int(seat)}", *(
                            both(f"ordered_slot_{window}_{int(seat)}_{slot['ordinal']}", before, slot["declined"],
                                 equal(slot["actor"], int(seat), f"ordered_actor_{window}_{int(seat)}_{slot['ordinal']}"))
                            for slot in window_slots))
                        terms.append(declined if seat.name.lower() in order else declined.Not())
        return both("common", *terms)

    def constrain_native_escalation_observation(self, *, information):
        """Start at the actor's visible opportunity, never a private-history ordinal."""
        info, windows = self._normalize_escalation_information(information)
        if (self._complete_information_deck is not None or self._native_play_root is not None
                or self._native_project_root is not None or self._recorded_escalation_count
                or self.auction_fragment.prefix_length != len(info["publicChoices"])):
            raise ValueError("observed escalation root requires observer visibility and the complete public auction prefix")
        core = {"schemaVersion": 1, "observationHash": self._escalation_observation_hash(info),
            "window": len(info["publicAnnouncements"])}
        root = {**core, "sha256": canonical_sha256(core)}
        if self._native_escalation_root is not None:
            if self._native_escalation_root != root:
                raise ValueError("symbolic escalation root is already fixed to another observation")
            return dict(root)
        name = "escalation_root_" + core["observationHash"][:20]
        common = self._native_escalation_information_literal(info, windows, name)
        equal, both, _ = _auction_logic_v1(self.model, name + "_position")
        slots = self._native_round_fragment.escalation_opportunities
        matches = [(index, both(f"matches_{index}", common, slot["active"],
            equal(slot["actor"], int(Seat.parse(info["actor"])), f"actor_{index}")))
            for index, slot in enumerate(slots) if slot["window"] == core["window"]]
        self.model.add(sum(literal for _, literal in matches) == 1)
        position = self.model.new_int_var(0, len(slots) - 1, name + "_slot")
        self.model.add(position == sum(index * literal for index, literal in matches))
        self._native_escalation_root = root
        self._native_escalation_root_position = position
        return dict(root)

    def bind_native_escalation_observed_choice(self, *, information, action,
            maximum_seconds, maximum_conflicts, cancel_check=None):
        """Bind the same action across indistinguishable escalation opportunities."""
        if type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("policy binding requires a finite positive time budget")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("policy binding requires a positive conflict budget")
        deadline = time.monotonic() + maximum_seconds
        info, windows = self._normalize_escalation_information(information)
        branch = self._native_round_fragment
        slots = branch.escalation_opportunities
        if type(action) is not str or action not in slots[0]["legal"]:
            raise ValueError("policy binding action is unsupported for escalation")
        key = self._escalation_observation_hash(info)
        previous = self._observed_escalation_bindings.get(key)
        if previous is not None:
            if previous["action"] != action:
                raise ValueError("conflicting actions for one escalation observation")
            return dict(previous)
        name = "escalation_observation_" + key[:20]
        common = self._native_escalation_information_literal(info, windows, name)
        equal, both, _ = _auction_logic_v1(self.model, name + "_choice")
        actor, window = Seat.parse(info["actor"]), len(info["publicAnnouncements"])
        choices = []
        for slot in slots:
            if slot["window"] != window:
                continue
            matches = both(f"matches_{slot['ordinal']}", common, slot["active"],
                equal(slot["actor"], int(actor), f"actor_{slot['ordinal']}"))
            selected = self._native_escalation_selected_literal(slot, action, name)
            choices.append((matches, selected, slot["legal"][action]))
        return self._bind_native_policy_choice(observation_hash=key, action=action, choices=choices,
            kind="escalation", bindings=self._observed_escalation_bindings, deadline=deadline,
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)

    def _native_escalation_selected_literal(self, slot, action, name, escalation=None):
        escalation = escalation if escalation is not None else self._native_round_fragment.escalation
        native, window = escalation.native, slot["window"]
        if action == escalation.stage.domain.closure_action:
            return slot["declined"]
        if window == len(native["actionSlots"]):
            return self.model.new_constant(0)  # No further announcement slot.
        equal, both, _ = _auction_logic_v1(self.model, name + "_choice")
        code, _, mode = action.partition(":")
        return both(f"selected_{slot['ordinal']}", slot["announced"],
            equal(native["actionSlots"][window][1], native["actions"][code], f"selected_action_{slot['ordinal']}"),
            *([equal(native["stateLocked"][window + 1], int(mode == "locked"), f"selected_mode_{slot['ordinal']}")] if mode else []))

    def _extract_native_project_observations(self, solver, opportunities, escalation, play, decisions):
        """Read first-trick declaration information, without future or hidden facts."""
        from .ortools_nonplay_observation import _project_rows
        from baloot_platform.core.project_ir import ProjectDeclarationActionV1

        branch = self._native_round_fragment
        layout = DealLayoutV1.build(self.rules.structural_domain, dealer=self.dealer)
        auction = [row.action + (f":{row.trump.name.lower()}" if row.trump is not None else "")
                   for row in self.auction_fragment.extract_actions(solver)]
        announcements = [[row.actor.name.lower(), row.action,
            row.resolved_mode if row.action in {"double", "four"} else None] for row in escalation.observation.actions]
        project_lookup = {row.project_id: row for row in branch.projects.projects}
        declared, observations = [], []
        for index, decision in enumerate(decisions):
            actor = Seat.parse(decision["actor"])
            if play.plays[index].actor != actor:
                raise ValueError("project declaration order disagrees with first-trick play order")
            information = {
                "actor": actor.name.lower(),
                "initialCards": sorted(solver.value(self.card_at_position[position])
                    for position in layout.positions_for(actor, final=False)),
                "visibleCards": [card for card, owner in enumerate(self.owner_of_card)
                                 if solver.value(owner) == int(actor)],
                "purchaseCard": solver.value(self.purchase_card), "publicChoices": list(auction),
                "publicAnnouncements": announcements,
                "ownDeclinedWindows": [row["windowHash"] for row in opportunities
                    if row["actor"] == actor.name.lower()
                    and row["selectedAction"] == branch.escalation.stage.domain.closure_action],
                "publicProjects": list(_project_rows(declared, project_lookup, start_ordinal=0)),
                "publicPlays": [row.to_dict() for row in play.plays[:index]],
            }
            if self._complete_information_deck is not None:
                information["completePriorWindowDeclines"] = {seat.name.lower(): [row["windowHash"] for row in opportunities
                    if row["actor"] == seat.name.lower() and row["selectedAction"] == branch.escalation.stage.domain.closure_action]
                    for seat in Seat}
                information["completeWindowDeclineOrder"] = [[row["actor"] for row in opportunities
                    if row["window"] == window and row["selectedAction"] == branch.escalation.stage.domain.closure_action]
                    for window in range(len(announcements) + 1)]
                information["completeProjectActions"] = [row["selectedAction"] for row in decisions[:index]]
            information, _ = self._normalize_project_information(information)
            observations.append({**decision, "information": information,
                **({"authoredPrefix": True} if index < self._recorded_project_count else {}),
                "observationHash": self._project_observation_hash(information)})
            declared.append(ProjectDeclarationActionV1.build(decision["selectedAction"], actor=actor).to_dict())
        return observations

    def _normalize_project_information(self, information):
        from .ortools_nonplay_observation import _strict_public_history
        from .ortools_play_tail import TailPrefixActionV1

        extra = {"publicProjects", "publicPlays"}
        if self._complete_information_deck is not None:
            extra.add("completeProjectActions")
        if not isinstance(information, Mapping) or not extra <= set(information):
            raise ValueError("project information requires public project and play histories")
        info, windows = self._normalize_escalation_information(
            {key: value for key, value in information.items() if key not in extra}, completed=True)
        if any(not isinstance(information[key], (list, tuple)) for key in extra):
            raise ValueError("project information histories must be lists or tuples")
        actor = Seat.parse(info["actor"])
        order = self.rules.project_domain.declaration_seats(self.dealer)
        index = order.index(actor)
        if any(not isinstance(row, Mapping) or type(row.get("actor")) not in {Seat, str, int}
               or row.get("action") not in {"pass", "declare"}
               for row in information["publicProjects"]):
            raise ValueError("project information declaration is invalid")
        projects = _strict_public_history(information["publicProjects"])
        plays = information["publicPlays"]
        if len(projects) != index or len(plays) != index:
            raise ValueError("project information must stop at the acting seat's first turn")
        parsed_plays = []
        for ordinal, (project, play) in enumerate(zip(projects, plays)):
            if (project["stage"] != "project" or project["actor"] != order[ordinal].name.lower()
                    or project["action"] not in {"pass", "declare"}):
                raise ValueError("project information declaration actor or action is invalid")
            if project["action"] == "declare" and not any(
                    row.owner == order[ordinal] and (row.kind, row.value, row.high_rank) ==
                    (project["kind"], project["value"], project["highRank"])
                    for row in self._native_round_fragment.projects.projects):
                raise ValueError("project information announcement is unsupported")
            if (not isinstance(play, Mapping) or set(play) != {"actor", "card", "ekka"}
                    or type(play["actor"]) not in {Seat, str, int}
                    or type(play["card"]) not in {Card, str, int} or type(play["ekka"]) is not bool):
                raise ValueError("project information play is invalid")
            parsed = TailPrefixActionV1.build(play["actor"], play["card"], ekka=play["ekka"])
            if parsed.actor != order[ordinal] or parsed.card.id in info["visibleCards"]:
                raise ValueError("project information play actor or ownership is invalid")
            parsed_plays.append(parsed.to_dict())
        if len({row["card"] for row in parsed_plays}) != len(parsed_plays):
            raise ValueError("project information repeats a played card")
        return {**info, "publicProjects": list(projects), "publicPlays": parsed_plays,
                **self._complete_project_history(information, index)}, windows

    def _complete_project_history(self, information, count):
        if self._complete_information_deck is None:
            return {}
        rows = information.get("completeProjectActions")
        order = self.rules.project_domain.declaration_seats(self.dealer)
        actions = self._native_round_fragment.projects.declaration_actions
        if (not isinstance(rows, (list, tuple)) or len(rows) != count
                or any(type(action) is not str or action not in actions[int(order[index])]
                       for index, action in enumerate(rows))):
            raise ValueError("complete project history must contain exactly the prior legal action identifiers")
        return {"completeProjectActions": list(rows)}

    def _project_observation_hash(self, information):
        return self._round_observation_hash("project", information)

    def _round_observation_hash(self, kind, information):
        scope = self._native_round_fragment.scope
        return canonical_sha256({"kind": kind, "rulesHash": self.rules.sha256,
            "dealer": self.dealer.name.lower(), "contract": scope["contract"],
            "trumpSuit": scope["trumpSuit"], "locked": scope["locked"],
            "matchScore": scope["matchScore"], **information,
            **({"completeDeck": list(self._complete_information_deck)}
               if self._complete_information_deck is not None else {})})

    def _extract_native_play_observations(self, solver, play, decisions, project_observations, baloot_events):
        """Read perfect-recall play information without future or private opponent data."""
        from .ortools_nonplay_observation import _project_rows
        from baloot_platform.core.project_ir import ProjectDeclarationActionV1

        branch = self._native_round_fragment
        lookup = {row.project_id: row for row in branch.projects.projects}
        by_actor = {row["actor"]: row for row in project_observations}
        public_projects = list(_project_rows([
            ProjectDeclarationActionV1.build(row["selectedAction"], actor=Seat.parse(row["actor"])).to_dict()
            for row in project_observations], lookup, start_ordinal=0))
        observations = []
        for index, decision in enumerate(decisions):
            own = by_actor[decision["actor"]]
            view = branch.public_project_bindings[min(index, len(branch.public_project_bindings) - 1)]
            compared = solver.boolean_value(view["compared"])
            info = {**own["information"], "ownProjectAction": own["selectedAction"],
                "publicProjects": public_projects[:min(index + 1, len(public_projects))],
                "publicPlays": [row.to_dict() for row in play.plays[:index]],
                "projectComparison": ([team.name.lower() for team in Team
                    if solver.boolean_value(branch.projects.winning_team[int(team)])] if compared else None),
                "projectPoints": ([solver.value(value) for value in branch.projects.combined_awarded_value]
                                  if compared else None),
                "revealedProjects": sorted(project_id for project_id, literal in view["revealed"].items()
                                           if solver.boolean_value(literal)),
                "publicBalootEvents": [dict(row) for row in baloot_events if row["playIndex"] < index]}
            if self._complete_information_deck is not None:
                info["completeProjectActions"] = [row["selectedAction"] for row in
                    project_observations[:min(index + 1, len(project_observations))]]
            info, _ = self._normalize_play_information(info)
            observations.append({**decision, "information": info,
                **({"authoredPrefix": True} if index < self._recorded_play_count else {}),
                "observationHash": self._round_observation_hash("play", info)})
        return observations

    def constrain_native_project_observation(self, *, information):
        """Fix only the declaring actor's observation, not hidden project identities."""
        info, windows = self._normalize_project_information(information)
        if (self._complete_information_deck is not None or self._native_play_root is not None
                or self._native_escalation_root is not None
                or self.auction_fragment.prefix_length != len(info["publicChoices"])):
            raise ValueError("observed project root requires observer visibility and complete public auction history")
        core = {"schemaVersion": 1, "observationHash": self._project_observation_hash(info),
            "playIndex": len(info["publicPlays"]), "projectPrefixLength": len(info["publicProjects"]),
            "escalationStageComplete": True}
        root = {**core, "sha256": canonical_sha256(core)}
        if self._native_project_root is not None:
            if self._native_project_root != root:
                raise ValueError("symbolic project root is already fixed to another observation")
            return dict(root)
        name = "project_observation_" + core["observationHash"][:20]
        terms = self._native_round_history_literals(info, windows, name)
        _, both, _ = _auction_logic_v1(self.model, name + "_root")
        self.model.add(both("matches", *terms) == 1)
        self._native_project_root = root
        self._recorded_play_count = core["playIndex"]
        self._recorded_project_count = core["projectPrefixLength"]
        return dict(root)

    def bind_native_project_observed_choice(self, *, information, action,
            maximum_seconds, maximum_conflicts, cancel_check=None):
        """Enforce a legal declaration only in matching actor-information states."""
        if type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("policy binding requires a finite positive time budget")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("policy binding requires a positive conflict budget")
        deadline = time.monotonic() + maximum_seconds
        info, windows = self._normalize_project_information(information)
        branch, actor = self._native_round_fragment, Seat.parse(info["actor"])
        actions = branch.projects.declaration_actions
        if actions is None or type(action) is not str or action not in actions[int(actor)]:
            raise ValueError("policy binding action is unsupported for projects")
        key = self._project_observation_hash(info)
        previous = self._observed_project_bindings.get(key)
        if previous is not None:
            if previous["action"] != action:
                raise ValueError("conflicting actions for one project observation")
            return dict(previous)
        name = "project_observation_" + key[:20]
        terms = self._native_round_history_literals(info, windows, name)
        equal, both, either = _auction_logic_v1(self.model, name + "_choice")
        selected = self._native_project_selected_literal(actor, action, both)
        return self._bind_native_policy_choice(observation_hash=key, action=action,
            choices=((both("matches", *terms), selected, actions[int(actor)][action]),),
            kind="project", bindings=self._observed_project_bindings, deadline=deadline,
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)

    def constrain_native_completed_escalation(self, events):
        """Fix all completed opportunities, including private declines."""
        self.constrain_native_escalation_prefix(events, complete=True)

    def constrain_native_escalation_prefix(self, events, *, complete=False, next_actor=None):
        """Bind recorded opportunities without choosing unplayed responses."""
        branch = self._native_round_fragment
        if (branch is None or not isinstance(events, (tuple, list)) or type(complete) is not bool
                or next_actor is not None and (complete or type(next_actor) is not str
                    or next_actor not in {seat.name.lower() for seat in Seat})):
            raise ValueError("native escalation history requires ordered events")
        slots = branch.escalation_opportunities
        codes = set(slots[0]["legal"]) if slots else set()
        if (len(events) > len(slots) or any(not isinstance(event, Mapping)
                or set(event) != {"actor", "actionCode"}
                or type(event["actor"]) is not str or event["actor"] not in {seat.name.lower() for seat in Seat}
                or type(event["actionCode"]) is not str or event["actionCode"] not in codes
                for event in events)):
            raise ValueError("native escalation history requires ordered events")
        count = sum(slot["active"] for slot in slots)
        self.model.add(count == len(events) if complete else count >= len(events) + int(next_actor is not None))
        previous = []
        for index, slot in enumerate(slots):
            equal, both, _ = _auction_logic_v1(self.model, f"recorded_escalation_{index}")
            for ordinal, event in enumerate(events):
                selected = both(f"event_{ordinal}", slot["active"], equal(sum(previous), ordinal, f"index_{ordinal}"))
                self.model.add(slot["actor"] == int(Seat.parse(event["actor"]))).only_enforce_if(selected)
                self.model.add(self._native_escalation_selected_literal(
                    slot, event["actionCode"], f"recorded_escalation_{index}_{ordinal}") == 1).only_enforce_if(selected)
            if next_actor is not None:
                root = both("next_actor", slot["active"], equal(sum(previous), len(events), "next_index"))
                self.model.add(slot["actor"] == int(Seat.parse(next_actor))).only_enforce_if(root)
            previous.append(slot["active"])
        self._recorded_escalation_count = len(events)

    def constrain_native_project_prefix(self, actions):
        """Fix recorded declarations; legality remains a native constraint."""
        from baloot_platform.core.project_ir import ProjectDeclarationActionV1
        branch = self._native_round_fragment
        order = self.rules.project_domain.declaration_seats(self.dealer)
        if (branch is None or not isinstance(actions, (tuple, list)) or len(actions) > len(order)
                or any(not isinstance(action, ProjectDeclarationActionV1)
                    or action.actor != order[index]
                    or action.code not in branch.projects.declaration_actions[int(action.actor)]
                    for index, action in enumerate(actions))):
            raise ValueError("native project prefix requires ordered declaration actions")
        for index, action in enumerate(actions):
            _, both, _ = _auction_logic_v1(self.model, f"recorded_project_{index}")
            self.model.add(branch.projects.declaration_actions[int(action.actor)][action.code] == 1)
            self.model.add(self._native_project_selected_literal(action.actor, action.code, both) == 1)
        self._recorded_project_count = len(actions)

    def _native_project_selected_literal(self, actor, action, both):
        projects = self._native_round_fragment.projects.projects
        return (both("selected_pass", *(project.declared.Not() for project in projects if project.owner == actor))
            if action == "pass" else next(project.declared for project in projects
                if project.owner == actor and "declare:" + project.project_id == action))

    def _native_round_history_literals(self, info, windows, name):
        branch = self._native_round_fragment
        common = self._native_escalation_information_literal(info, windows, name)
        equal, both, either = _auction_logic_v1(self.model, name + "_history")
        terms = [common, equal(branch.escalation.native["length"], len(info["publicAnnouncements"]), "escalation_length")]
        for index, row in enumerate(info["publicProjects"]):
            owned = [project for project in branch.projects.projects if project.owner == Seat.parse(row["actor"])]
            if row["action"] == "pass":
                terms.append(both(f"passed_{index}", *(project.declared.Not() for project in owned)))
            else:
                terms.append(either(f"declared_{index}", *(project.declared for project in owned
                    if (project.kind, project.value, project.high_rank) == (row["kind"], row["value"], row["highRank"]))))
        if self._complete_information_deck is not None:
            order = self.rules.project_domain.declaration_seats(self.dealer)
            for index, action in enumerate(info["completeProjectActions"]):
                owned = [project for project in branch.projects.projects if project.owner == order[index]]
                terms.append(both(f"complete_pass_{index}", *(project.declared.Not() for project in owned))
                    if action == "pass" else next(project.declared for project in owned
                                                 if "declare:" + project.project_id == action))
        for index, row in enumerate(info["publicPlays"]):
            slot = branch.play.play_state_context.play_slots[index]
            terms.extend((equal(slot.actor, int(Seat.parse(row["actor"])), f"actor_{index}"),
                equal(slot.card, Card.parse(row["card"]).id, f"card_{index}"),
                equal(slot.ekka, int(row["ekka"]), f"ekka_{index}")))
        return terms

    def _normalize_play_information(self, information):
        from .ortools_nonplay_observation import _strict_public_history
        from .ortools_play_tail import TailPrefixActionV1

        extra = {"ownProjectAction", "publicProjects", "publicPlays", "projectComparison",
                 "projectPoints", "revealedProjects", "publicBalootEvents"}
        if self._complete_information_deck is not None:
            extra.add("completeProjectActions")
        if not isinstance(information, Mapping) or not extra <= set(information):
            raise ValueError("play information requires complete public histories")
        info, windows = self._normalize_escalation_information(
            {key: value for key, value in information.items() if key not in extra}, completed=True)
        if any(not isinstance(information[key], (list, tuple)) for key in
               ("publicProjects", "publicPlays", "revealedProjects", "publicBalootEvents")):
            raise ValueError("play information histories must be lists or tuples")
        branch, actor = self._native_round_fragment, Seat.parse(info["actor"])
        raw_plays = information["publicPlays"]
        if not 0 <= len(raw_plays) < len(branch.play._extract.actors):
            raise ValueError("play information must stop before a compiled decision")
        plays = []
        for row in raw_plays:
            if (not isinstance(row, Mapping) or set(row) != {"actor", "card", "ekka"}
                    or type(row["actor"]) not in {Seat, str, int}
                    or type(row["card"]) not in {Card, str, int} or type(row["ekka"]) is not bool):
                raise ValueError("play information card history is invalid")
            parsed = TailPrefixActionV1.build(row["actor"], row["card"], ekka=row["ekka"])
            if (parsed.actor == actor) != (parsed.card.id in info["visibleCards"]):
                raise ValueError("play information contradicts own card ownership")
            plays.append(parsed.to_dict())
        if len({row["card"] for row in plays}) != len(plays):
            raise ValueError("play information repeats a card")
        raw_projects = information["publicProjects"]
        if any(not isinstance(row, Mapping) or type(row.get("actor")) not in {Seat, str, int}
               or row.get("action") not in {"pass", "declare"} for row in raw_projects):
            raise ValueError("play information declaration is invalid")
        projects = _strict_public_history(raw_projects)
        order = tuple(self.rules.project_domain.declaration_seats(self.dealer))
        if len(projects) != min(len(plays) + 1, len(order)):
            raise ValueError("play information declaration count is invalid")
        for index, row in enumerate(projects):
            if row["stage"] != "project" or row["actor"] != order[index].name.lower():
                raise ValueError("play information declaration order is invalid")
            if row["action"] == "declare" and not any(project.owner == order[index]
                    and (project.kind, project.value, project.high_rank) ==
                    (row.get("kind"), row.get("value"), row.get("highRank")) for project in branch.projects.projects):
                raise ValueError("play information declaration is unsupported")
        own_action = information["ownProjectAction"]
        if type(own_action) is not str or own_action not in branch.projects.declaration_actions[int(actor)]:
            raise ValueError("play information own declaration is invalid")
        comparison, points = information["projectComparison"], information["projectPoints"]
        if len(plays) < 4:
            if comparison is not None or points is not None:
                raise ValueError("play information reveals future project comparison")
        elif (not isinstance(comparison, (list, tuple)) or len(comparison) > 1
                or any(type(team) is not str or team not in {value.name.lower() for value in Team} for team in comparison)
                or not isinstance(points, (list, tuple)) or len(points) != len(Team)
                or any(type(value) is not int or not 0 <= value <= max(map(int, variable.proto.domain))
                       for value, variable in zip(points, branch.projects.combined_awarded_value))):
            raise ValueError("play information project comparison is invalid")
        revealed = information["revealedProjects"]
        if (any(type(value) is not str for value in revealed) or list(revealed) != sorted(set(revealed))
                or not set(revealed) <= {project.project_id for project in branch.projects.projects}
                or (len(plays) <= 4 and revealed)):
            raise ValueError("play information revealed projects are invalid")
        events = information["publicBalootEvents"]
        if any(not isinstance(row, Mapping) or set(row) != {"playIndex", "actor", "kind"}
               or type(row["playIndex"]) is not int or not 0 <= row["playIndex"] < len(plays)
               or row["actor"] != plays[row["playIndex"]]["actor"]
               or row["kind"] not in {"included", "declared"} for row in events):
            raise ValueError("play information Baloot history is invalid")
        if [row["playIndex"] for row in events] != sorted({row["playIndex"] for row in events}):
            raise ValueError("play information Baloot history must be unique and ordered")
        return {**info, "ownProjectAction": own_action, "publicProjects": list(projects), "publicPlays": plays,
            "projectComparison": None if comparison is None else list(comparison),
            "projectPoints": None if points is None else list(points), "revealedProjects": list(revealed),
            "publicBalootEvents": [dict(row) for row in events],
            **self._complete_project_history(information, len(projects))}, windows

    def _native_play_observation_terms(self, info, windows, name):
        branch, actor = self._native_round_fragment, Seat.parse(info["actor"])
        index = len(info["publicPlays"])
        terms = self._native_round_history_literals(info, windows, name)
        equal, both, _ = _auction_logic_v1(self.model, name + "_choice")
        terms.append(equal(branch.play._extract.actors[index], int(actor), "actor"))
        own = info["ownProjectAction"]
        terms.append(both("own_pass", *(project.declared.Not() for project in branch.projects.projects if project.owner == actor))
            if own == "pass" else next(project.declared for project in branch.projects.projects
                if project.owner == actor and "declare:" + project.project_id == own))
        view = branch.public_project_bindings[min(index, len(branch.public_project_bindings) - 1)]
        terms.append(view["compared"] if info["projectComparison"] is not None else view["compared"].Not())
        if info["projectComparison"] is not None:
            terms.extend(equal(branch.projects.winning_team[int(team)], int(team.name.lower() in info["projectComparison"]),
                               f"winning_team_{int(team)}") for team in Team)
            terms.extend(equal(variable, value, f"points_{team}") for team, (variable, value) in
                         enumerate(zip(branch.projects.combined_awarded_value, info["projectPoints"], strict=True)))
        terms.extend(literal if project_id in info["revealedProjects"] else literal.Not()
                     for project_id, literal in view["revealed"].items())
        events = {(row["playIndex"], row["kind"]) for row in info["publicBalootEvents"]}
        for kind, literals in zip(("included", "declared"), branch.public_baloot_binding[:2], strict=True):
            terms.extend(literal if (step, kind) in events else literal.Not()
                         for step, literal in enumerate(literals[:index]))
        return terms, equal, both

    def native_play_observation_literal(self, *, information):
        """Match the actor's full visible history without fixing hidden hands or a move."""
        info, windows = self._normalize_play_information(information)
        name = "play_observation_" + self._round_observation_hash("play", info)[:20]
        terms, _, both = self._native_play_observation_terms(info, windows, name)
        return both("matches", *terms)

    def constrain_native_play_observation(self, *, information):
        """Start a symbolic continuation at a visible play decision, not its past."""
        info, _ = self._normalize_play_information(information)
        if (self._complete_information_deck is not None or self._native_project_root is not None
                or self._native_escalation_root is not None
                or self.auction_fragment.prefix_length != len(info["publicChoices"])):
            raise ValueError("observed play root requires observer visibility and the complete public auction prefix")
        core = {"schemaVersion": 1, "observationHash": self._round_observation_hash("play", info),
            "playIndex": len(info["publicPlays"]), "projectPrefixLength": len(info["publicProjects"]),
            "escalationStageComplete": True}
        root = {**core, "sha256": canonical_sha256(core)}
        if self._native_play_root is not None:
            if self._native_play_root != root:
                raise ValueError("symbolic play root is already fixed to another observation")
            return dict(root)
        self.model.add(self.native_play_observation_literal(information=info) == 1)
        self._native_play_root = root
        self._recorded_play_count = core["playIndex"]
        self._recorded_project_count = core["projectPrefixLength"]
        return dict(root)

    def bind_native_play_observed_choice(self, *, information, action,
            maximum_seconds, maximum_conflicts, cancel_check=None, _prefix_legality=False,
            _defer_owner_hash=False):
        """Bind a play across worlds sharing the actor's complete visible history."""
        from baloot_platform.core.models import LegalPlayActionV1

        if type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("policy binding requires a finite positive time budget")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("policy binding requires a positive conflict budget")
        if type(_prefix_legality) is not bool:
            raise TypeError("prefix legality selection must be boolean")
        if type(_defer_owner_hash) is not bool or _defer_owner_hash and not _prefix_legality:
            raise ValueError("deferred play hashing requires prefix legality")
        deadline = time.monotonic() + maximum_seconds
        info, windows = self._normalize_play_information(information)
        branch = self._native_round_fragment
        index = len(info["publicPlays"])
        legal = branch.play.native_legal_action_literals(index)
        if type(action) is not str or action not in legal:
            raise ValueError("policy binding play action is unsupported")
        key = self._round_observation_hash("play", info)
        previous = self._observed_play_bindings.get(key)
        if previous is not None:
            if previous["action"] != action:
                raise ValueError("conflicting actions for one play observation")
            return dict(previous)
        terms, equal, both = self._native_play_observation_terms(info, windows, "play_observation_" + key[:20])
        parsed = LegalPlayActionV1.from_code(action)
        slot = branch.play.play_state_context.play_slots[index]
        selected = both("selected", equal(slot.card, parsed.card.id, "selected_card"),
                        equal(slot.ekka, int(parsed.ekka), "selected_ekka"))
        probe = None
        if _prefix_legality and time.monotonic() < deadline and not (cancel_check is not None and cancel_check()):
            from .ortools_play_tail import OrToolsPlayTailStageV1, PlayTailSolveStopped, TailPrefixActionV1
            def stopped():
                return time.monotonic() >= deadline or (cancel_check is not None and cancel_check())
            try:
                # Legality depends on the current trick and remaining cards,
                # not completed tricks. Keep the full-history guard in the owner;
                # this broader probe never supplies reachability or strategic truth.
                completed = index // len(Seat) * len(Seat)
                removed = {Card.parse(row["card"]).id for row in info["publicPlays"][:completed]}
                current = info["publicPlays"][completed:]
                prefix_build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
                    cards=tuple(card for card in FULL_DECK if card.id not in removed),
                    ownership={card: Seat.parse(info["actor"]) for card in info["visibleCards"] if card not in removed},
                    contract=branch.scope["contract"], trump=branch.scope["trumpSuit"],
                    first_leader=current[0]["actor"] if current else info["actor"], locked=branch.scope["locked"],
                    play_domain=self.rules.play_domain, structural_domain=self.rules.structural_domain,
                    prefix=tuple(TailPrefixActionV1.build(row["actor"], row["card"], ekka=row["ekka"])
                        for row in current),
                    play_step_count=len(Seat), decision_prefix_length=len(current),
                    decision_actor=info["actor"], _export_predicate_contexts=False, cancel_check=stopped)
                probe = prefix_build.model
                probe.add(prefix_build.native_legal_action_literals(len(current)).get(action, 0) == 0)
            except PlayTailSolveStopped:
                pass  # The original binder records cancellation/deadline without solving.
        return self._bind_native_policy_choice(observation_hash=key, action=action,
            choices=((both("matches", *terms), selected, legal[action]),),
            kind="play", bindings=self._observed_play_bindings, deadline=deadline,
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, legality_probe=probe,
            _defer_owner_hash=_defer_owner_hash)

    def _bind_native_play_overrides(self, rows, *, maximum_seconds, maximum_conflicts, cancel_check=None):
        """Prove each choice; record their shared owner once, not per choice."""
        if type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("play override batch requires a finite positive deadline")
        deadline = time.monotonic() + maximum_seconds
        bindings, matches = [], {}
        for row in rows:
            binding = self.bind_native_play_observed_choice(**row,
                maximum_seconds=max(deadline - time.monotonic(), 1e-9), maximum_conflicts=maximum_conflicts,
                cancel_check=cancel_check, _prefix_legality=True, _defer_owner_hash=True)
            bindings.append(binding)
            if (binding["status"] != "bound" or time.monotonic() >= deadline
                    or cancel_check is not None and cancel_check()):
                break
            # Preserve the original variable/constraint order for frozen models.
            matches.setdefault(len(row["information"]["publicPlays"]), []).append(
                self.native_play_observation_literal(information=row["information"]))
        if bindings:
            owner_hash = hashlib.sha256(str(self.model.proto).encode("utf-8")).hexdigest()
            self._last_policy_binding_proto_hash = owner_hash
            for index, binding in enumerate(bindings):
                if "modelHash" in binding:
                    continue
                core = {key: value for key, value in binding.items() if key != "sha256"}
                core["modelHash"] = self._model_hash(search_branching="fixed", proto_hash=owner_hash)
                bindings[index] = {**core, "sha256": canonical_sha256(core)}
                if binding["status"] == "bound":
                    self._observed_play_bindings[binding["observationHash"]] = bindings[index]
        return bindings, matches

    def _validate_round_match_score(self, match_score):
        from baloot_platform.core.scoring_ir import ScoringDomainIRV1

        scoring_domain = ScoringDomainIRV1.from_rules(self.rules.data)
        if (not isinstance(match_score, tuple) or len(match_score) != 2
                or any(type(value) is not int or value < 0 for value in match_score)
                or (max(match_score) >= scoring_domain.match_target and match_score[0] != match_score[1])):
            raise ValueError("round continuation requires two nonterminal match scores")
        return scoring_domain

    def _native_round_query_context(self):
        if self._native_round_fragment is not None:
            scope = self._native_round_fragment.scope
            if (self._native_play_root is not None or self._native_project_root is not None
                    or self._native_escalation_root is not None):
                core = {key: value for key, value in scope.items() if key != "sha256"}
                if scope.get("sha256") != canonical_sha256(core):
                    raise ValueError("native round scope integrity mismatch")
                if self._native_play_root is not None:
                    core["playRoot"] = dict(self._native_play_root)
                if self._native_project_root is not None:
                    core["projectRoot"] = dict(self._native_project_root)
                if self._native_escalation_root is not None:
                    core["escalationRoot"] = dict(self._native_escalation_root)
                scope = {**core, "sha256": canonical_sha256(core)}
            return scope, self._native_round_fragment.scoring.awarded_points
        if self._native_redeal_scope is not None and self._native_redeal_awarded_points is not None:
            return self._native_redeal_scope, self._native_redeal_awarded_points
        raise ValueError("round outcome query requires a native round continuation")

    def add_native_redeal_continuation(self, *, match_score: tuple[int, int]):
        """Compile the all-pass deal boundary; the next deal is not analyzed."""
        auction = self.auction_fragment
        if not isinstance(auction, OrToolsNativeAuctionFragmentV1):
            raise ValueError("redeal continuation requires a native auction fragment")
        if self._native_round_fragment is not None or self._native_redeal_scope is not None:
            raise ValueError("scenario model already contains a round continuation")
        if not self.rules.structural_domain.all_pass_redeal:
            raise ValueError("rules do not support all-pass redeal")
        scoring_domain = self._validate_round_match_score(match_score)
        self.model.add(auction.states[-1]["allPassed"] == 1)
        self.model.add_all_different(self.card_at_position)
        # This utility is the current deal's award delta, not the value of the
        # subsequent redeal. An all-pass deal awards no round points.
        self._native_redeal_awarded_points = tuple(self.model.new_constant(0) for _ in Team)
        scope = {"schemaVersion": 1, "kind": "native_all_pass_deal_branch_v1",
            "rulesHash": self.rules.sha256, "auctionDomainHash": auction.domain_hash,
            "scoringDomainHash": scoring_domain.sha256, "matchScore": list(match_score),
            "continuationBoundary": "current_deal_completion", "nextDealAnalyzed": False,
            "allPassedCovered": True, "strategicCoverage": False}
        self._native_redeal_scope = {**scope, "sha256": canonical_sha256(scope)}
        return dict(self._native_redeal_scope)

    def _add_native_round_escalation(self, *, contract, trump, first_leader, locked, match_score):
        """Shared legal prefix used by both reachability and full-round models."""
        from baloot_platform.core.escalation_ir import EscalationDomainIRV1
        from .ortools_escalation_stage import OrToolsEscalationStageV1
        from .scenario_predicate_auction_escalation_v2 import CpSatEscalationPredicateContextV1

        auction = self.auction_fragment
        if not isinstance(auction, OrToolsNativeAuctionFragmentV1):
            raise ValueError("round continuation requires a native auction fragment")
        if self._native_round_fragment is not None or self._native_redeal_scope is not None:
            raise ValueError("scenario model already contains a round continuation")
        contract = Contract.parse(contract)
        trump = Suit.parse(trump) if trump is not None else None
        first_leader = Seat.parse(first_leader)
        if (contract.effective == Contract.HOKUM) != (trump is not None):
            raise ValueError("round continuation trump disagrees with contract")
        if type(locked) is not bool:
            raise TypeError("round continuation locked must be boolean")
        scoring_domain = self._validate_round_match_score(match_score)
        play_domain = self.rules.play_domain.require_integrity(self.rules.structural_domain)
        if play_domain.first_leader == "dealer_right":
            expected_leader = self.rules.structural_domain.dealer_right(self.dealer)
        elif play_domain.first_leader == "configured":
            expected_leader = play_domain.configured_first_leader
        elif play_domain.first_leader == "buyer":
            expected_leader = None
        else:
            raise ValueError("first-leader rule lacks native round binding")
        if expected_leader is not None and first_leader != expected_leader:
            raise ValueError("round continuation leader disagrees with rules")
        model = self.model
        # OR-Tools 9.15 inverse expansion can falsely reject the partially fixed
        # full-round model. Explicit bijection propagation preserves presolve
        # and the exact universe (the allocation inverse already implies this).
        # Keep this local to the new round model; historical deal models retain
        # their original representation. See the unresolved-Sun parity fixture.
        model.add_all_different(self.card_at_position)
        model.add(auction.contract == _CONTRACT_IDS[contract])
        model.add(auction.trump == (int(trump) if trump is not None else _NO_TRUMP))
        if expected_leader is None:
            model.add(auction.buyer == int(first_leader))
        escalation = CpSatEscalationPredicateContextV1.attach(
            OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(self.rules.data),
                declarer=self.dealer, contract=contract, match_score=match_score, native_cp_sat=True),
            model=model, context_id="round_escalation", declarer_variable=auction.declarer)
        model.add(escalation.native["terminal"] == 1)
        model.add(escalation.native["locked"] == int(locked))
        return escalation, scoring_domain, play_domain

    def add_native_round_continuation(
        self, *, contract: Contract | str, trump: Suit | str | None,
        first_leader: Seat | str, locked: bool, match_score: tuple[int, int],
        prefix: Sequence[TailPrefixActionV1] = (),
        opportunity_order: Sequence[Seat | str | int] | None = None,
        fixed_card_owners: Sequence[Seat | str | int | None] | None = None,
        all_project_actions_passed: bool = False,
        policy_prefix_steps: int = 0,
        policy_id: str = "observer_safe_trick_control_v1",
        cancel_check: Callable[[], bool] | None = None,
        decision_prefix_length: int | None = None,
    ) -> OrToolsNativeRoundFragmentV1:
        """Connect a selected auction branch to every legal play and terminal utility."""
        from .ortools_play_tail import OrToolsPlayTailStageV1, PlayTailSolveStopped
        from .ortools_exact_state import _native_baloot_prefix_projection, _native_project_visibility_projection
        from .ortools_project_fragment import attach_project_lifecycle_fragment_v1
        from .ortools_scoring_stage import CpSatScoringInputsV1, attach_scoring_fragment_v1, scoring_contract_id_v1

        if decision_prefix_length is not None and (
                type(decision_prefix_length) is not int or not 0 <= decision_prefix_length <= 32):
            raise ValueError("decision_prefix_length lies outside the play trace")
        if cancel_check is not None and cancel_check():
            raise PlayTailSolveStopped("cancelled")
        contract = Contract.parse(contract)
        trump = Suit.parse(trump) if trump is not None else None
        first_leader = Seat.parse(first_leader)
        escalation, scoring_domain, play_domain = self._add_native_round_escalation(
            contract=contract, trump=trump, first_leader=first_leader, locked=locked, match_score=match_score)
        model, auction = self.model, self.auction_fragment
        escalation_opportunities = escalation.attach_opportunity_path(order=opportunity_order)
        if fixed_card_owners is not None:
            fixed_card_owners = tuple(fixed_card_owners)
        second_trick_leader = (
            model.new_int_var(0, 3, "round_second_trick_leader")
            if self.rules.project_domain.tie_break_order == "second_trick_reveal" else None
        )
        try:
            projects = attach_project_lifecycle_fragment_v1(model,
                owner_of_card=self.owner_of_card, contract=contract, dealer=self.dealer,
                trump=trump, domain=self.rules.project_domain, name="round_projects",
                fixed_card_owners=fixed_card_owners, compact_fixed_states=fixed_card_owners is not None,
                tighten_value_bounds=fixed_card_owners is not None,
                include_declaration_actions=True, second_trick_leader=second_trick_leader,
                cancel_check=cancel_check)
        except InterruptedError as exc:
            raise PlayTailSolveStopped("cancelled") from exc
        if all_project_actions_passed:
            for project in projects.projects:
                model.add(project.declared == 0)
        play = OrToolsPlayTailStageV1(native_conversion=True).build_model(
            cards=tuple(Card(index) for index in self.rules.structural_domain.card_ids),
            ownership=({Card(index): owner for index, owner in enumerate(fixed_card_owners)
                        if owner is not None} if fixed_card_owners is not None else None),
            contract=contract, trump=trump, first_leader=first_leader,
            play_domain=play_domain, structural_domain=self.rules.structural_domain,
            scoring_domain=scoring_domain, prefix=prefix, locked=locked,
            decision_prefix_length=decision_prefix_length,
            policy_prefix_steps=policy_prefix_steps, policy_id=policy_id,
            shared_model=model, shared_owner_of_card=self.owner_of_card,
            cancel_check=cancel_check)
        if second_trick_leader is not None:
            model.add(second_trick_leader == play._extract.winner_seats[0])
        public_baloot_binding = _native_baloot_prefix_projection(model=model, fragment=projects,
            play_slot=play.play_state_context.play_slots.__getitem__,
            prefix_length=len(play._extract.actors), contract=contract, trump=trump,
            domain=self.rules.project_domain)
        public_project_bindings = []
        declaration_seats = tuple(self.rules.project_domain.declaration_seats(self.dealer))
        # After the second trick, project visibility no longer changes. Reuse
        # that final view rather than compiling the same constraints 24 times.
        for prefix_length in range(9):
            if cancel_check is not None and cancel_check():
                raise PlayTailSolveStopped("cancelled")
            if all_project_actions_passed:
                public_project_bindings.append({"compared": model.new_constant(int(prefix_length >= 4)),
                    "revealed": {project.project_id: model.new_constant(0) for project in projects.projects}})
                continue
            name = f"round_public_projects_{prefix_length}"
            equal, _, either = _auction_logic_v1(model, name)
            def owner_seen(owner, start, stop, label):
                return either(label, *(equal(play._extract.actors[step], int(owner), f"{label}_{step}")
                                      for step in range(start, min(stop, prefix_length))))
            seen_first = {seat: model.new_constant(int(seat in declaration_seats[:min(prefix_length + 1, 4)]))
                          for seat in Seat}
            compared, _, _, _, _, revealed = _native_project_visibility_projection(
                model=model, fragment=projects, seen_first=seen_first, owner_seen=owner_seen,
                prefix_length=prefix_length, declaration_seats=declaration_seats, name=name,
                reveal_scope=self.rules.project_domain.reveal_scope)
            public_project_bindings.append({"compared": compared, "revealed": revealed})
        baloot = tuple(model.new_int_var(0, projects.baloot_value, f"round_baloot_{team}")
                       for team in range(2))
        for team, variable in enumerate(baloot):
            model.add(variable == projects.baloot_value * projects.baloot_team[team])
        scope = {"schemaVersion": 1, "kind": "native_purchased_round_branch_v1",
            "rulesHash": self.rules.sha256, "auctionDomainHash": auction.domain_hash,
            "projectFragmentHash": projects.fragment_hash, "playRequestHash": play.request_hash,
            "scoringDomainHash": scoring_domain.sha256, "escalationDomainHash": escalation.stage.domain.sha256,
            "contract": contract.value, "trumpSuit": trump.name.lower() if trump is not None else None,
            "firstLeader": first_leader.name.lower(), "locked": locked,
            "escalationOpportunityOrder": ("symbolic_per_window" if opportunity_order is None
                else [Seat.parse(seat).name.lower() for seat in opportunity_order]),
            "matchScore": list(match_score), "allPassedCovered": False, "strategicCoverage": False}
        if all_project_actions_passed:
            scope["projectVisibilitySpecialization"] = "all_pass_v1"
        scoring = attach_scoring_fragment_v1(model, scoring_domain, CpSatScoringInputsV1(
            contract=model.new_constant(scoring_contract_id_v1(contract)), buyer_seat=auction.buyer,
            escalation=escalation.native["level"], last_escalating_team=escalation.native["lastTeam"],
            raw_points=play.raw_team_variables, project_points=projects.combined_awarded_value,
            baloot_points=baloot, trick_winner_teams=play.winner_team_variables,
            starting_match_score=tuple(model.new_constant(value) for value in match_score)),
            source_hash=canonical_sha256(scope), native_conversion=True)
        fragment = OrToolsNativeRoundFragmentV1(play, escalation, projects, scoring,
            {**scope, "sha256": canonical_sha256(scope)}, escalation_opportunities, public_baloot_binding,
            tuple(public_project_bindings))
        self._native_round_fragment = fragment
        return fragment

    def constrain_native_auction_root_choice(self, action, *, actor):
        """Fix the decision being proved without applying its continuation policy to it."""
        fragment = self.auction_fragment
        if (not isinstance(fragment, OrToolsNativeAuctionFragmentV1) or not isinstance(actor, Seat)
                or type(action) is not str or action == "closed"
                or not fragment.prefix_length < len(fragment.choices)
                or action not in fragment.choices[fragment.prefix_length]):
            raise ValueError("native auction root requires an actor and a legal action code")
        root = (fragment.prefix_length, action, actor.team)
        if self._native_auction_policy_root is not None:
            if self._native_auction_policy_root != root:
                raise ValueError("native auction root cannot be replaced")
            return
        self.model.add(fragment.choices[fragment.prefix_length][action] == 1)
        self._native_auction_policy_root = root

    def _native_constant_nonplay_choices(self, policy_team, kind, action, *, escalation=None, opportunities=()):
        """Same policy literals in prefix reachability and full-round queries."""
        if kind == "escalation" and (self._native_play_root is not None or self._native_project_root is not None):
            return []
        name = "constant_" + kind + "_" + policy_team.name.lower()
        equal, both, either = _auction_logic_v1(self.model, name)
        choices = []
        if kind == "auction":
            fragment = self.auction_fragment
            for i in range(fragment.prefix_length, len(fragment.choices)):
                if (self._native_auction_policy_root is not None
                        and i == self._native_auction_policy_root[0]
                        and action != self._native_auction_policy_root[1]
                        and policy_team == self._native_auction_policy_root[2]):
                    continue
                state = fragment.states[i]
                matches = both(f"active_{i}", state["terminal"].Not(), either(f"team_{i}",
                    *(equal(state["actor"], int(s), f"actor_{i}_{int(s)}") for s in Seat if s.team == policy_team)))
                choices.append((matches, fragment.choices[i][action], fragment.legal_choices_by_step[i][action]))
        else:
            previous = []
            for position, slot in enumerate(opportunities):
                i = slot["ordinal"]
                matches = both(f"active_{i}", slot["active"], either(f"team_{i}",
                    *(equal(slot["actor"], int(s), f"actor_{i}_{int(s)}") for s in Seat if s.team == policy_team)))
                if self._native_escalation_root_position is not None:
                    future = self.model.new_bool_var(f"{name}_root_future_{position}")
                    self.model.add(self._native_escalation_root_position <= position).only_enforce_if(future)
                    self.model.add(self._native_escalation_root_position > position).only_enforce_if(future.Not())
                    matches = both(f"root_future_active_{position}", matches, future)
                elif self._recorded_escalation_count:
                    future = self.model.new_bool_var(f"{name}_future_{i}")
                    self.model.add(sum(previous) >= self._recorded_escalation_count).only_enforce_if(future)
                    self.model.add(sum(previous) < self._recorded_escalation_count).only_enforce_if(future.Not())
                    matches = both(f"future_active_{i}", matches, future)
                selected = self._native_escalation_selected_literal(slot, action, name, escalation)
                choices.append((matches, selected, slot["legal"][action]))
                previous.append(slot["active"])
        return choices

    def _native_visible_auction_policy_choices(self, policy_team, policy_id):
        """One observer-safe action from native legal choices, preserving the fixed root."""
        validate_native_observed_auction_policy_id_v1(policy_id)
        if not isinstance(policy_team, Team) or any(
                not {"sun", "pass"} <= row.keys() for row in self.auction_fragment.choices):
            raise ValueError("native auction policy requires Sun/Pass action variables and a team")
        root_actor_only = policy_id == "prefer_root_actor_sun_partner_pass_v1"
        if root_actor_only and (self._native_auction_policy_root is None
                or self._native_auction_policy_root[2] != policy_team):
            raise ValueError("root-actor auction policy requires a fixed root owned by its team")
        equal, both, either = _auction_logic_v1(self.model, "visible_auction_policy")
        rows = {row["sun"].index: (row["pass"], legal["pass"], state["actor"])
                for row, legal, state in zip(self.auction_fragment.choices,
                    self.auction_fragment.legal_choices_by_step,
                    self.auction_fragment.states[:-1], strict=True)}
        choices = []
        for index, (matches, selected_sun, legal_sun) in enumerate(
                self._native_constant_nonplay_choices(policy_team, "auction", "sun")):
            selected_pass, legal_pass, actor = rows[selected_sun.index]
            choose_sun = legal_sun
            if root_actor_only:
                root_actor = self.auction_fragment.states[self._native_auction_policy_root[0]]["actor"]
                choose_sun = both(f"root_sun_{index}", legal_sun,
                    equal(actor, root_actor, f"root_actor_{index}"))
            selected = either(f"selected_{index}", both(f"sun_{index}", choose_sun, selected_sun),
                both(f"pass_{index}", choose_sun.Not(), selected_pass))
            available = either(f"legal_{index}", choose_sun,
                both(f"pass_legal_{index}", choose_sun.Not(), legal_pass)) if root_actor_only else either(
                    f"legal_{index}", legal_sun, legal_pass)
            choices.append((matches, selected, available))
        return choices

    def find_native_round_with_constant_policy_below(
        self, *, target: Team, policy_team: Team, auction_action: str | None,
        escalation_action: str | None, threshold: int, maximum_seconds: float,
        maximum_conflicts: int, cancel_check: Callable[[], bool] | None = None,
        project_actions: Mapping[str, str] | None = None,
        native_project_policy_id: str | None = None,
        native_auction_policy_id: str | None = None,
        play_policies: Mapping[str, Mapping[str, Any]] | None = None,
        native_play_policy_id: str | None = None,
        other_native_play_policy_id: str | None = None,
        native_play_overrides: Sequence[Mapping[str, Any]] | None = None,
        escalation_observations: Sequence[Mapping[str, Any]] | None = None,
        record_model_bytes: bool = False,
        optimize_margin: bool = True,
        record_policy_model_hashes: bool = False,
    ) -> dict[str, Any]:
        """Query one branch under a natively validated constant team strategy.

        All future information sets of the chosen team share the same supplied
        nonplay action per stage. Optional play templates choose the first legal
        action in their governed order. Other actions remain unrestricted. This is a bound
        ingredient, not branch coverage, minimax, or publication certification.
        Binding both native play policies constructs a conditioned joint line;
        its scope cannot certify a guarantee against unrestricted opponents.
        Guarded play overrides refine a native fallback at exact information
        states. Their distinct receipt is not accepted by legacy proof readers.
        The source model is never changed, including on invalidity/exhaustion.
        """
        if not isinstance(target, Team) or not isinstance(policy_team, Team) or type(threshold) is not int:
            raise ValueError("constant policy query requires teams and an integer threshold")
        if type(record_model_bytes) is not bool:
            raise TypeError("native model-byte recording must be boolean")
        if type(optimize_margin) is not bool:
            raise TypeError("constant policy optimization selection must be boolean")
        if type(record_policy_model_hashes) is not bool:
            raise TypeError("constant policy model-hash recording must be boolean")
        if record_policy_model_hashes and (project_actions or native_auction_policy_id or play_policies or escalation_observations
                                           or other_native_play_policy_id is not None):
            raise ValueError("policy model hashes require constant nonplay and unmixed native project/play authorities")
        if record_model_bytes and escalation_observations:
            raise ValueError("model-byte recording does not yet cover observed escalation bindings")
        if (type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds)
                or maximum_seconds <= 0 or type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("constant policy query requires positive finite budgets")
        if cancel_check is not None and not callable(cancel_check):
            raise TypeError("constant policy cancellation check must be callable")
        if project_actions is not None and (not isinstance(project_actions, Mapping)
                or any(type(seat) is not str or seat not in {s.name.lower() for s in Seat}
                       or type(action) is not str for seat, action in project_actions.items())):
            raise ValueError("constant project policy requires canonical seat/action strings")
        project_actions = dict(sorted((project_actions or {}).items()))
        if native_auction_policy_id is not None:
            validate_native_observed_auction_policy_id_v1(native_auction_policy_id)
            if auction_action is not None:
                raise ValueError("native auction policy cannot mix auction authorities")
        if native_project_policy_id is not None:
            from .ortools_project_fragment import NATIVE_FULL_HAND_PROJECT_POLICY_ID
            if native_project_policy_id != NATIVE_FULL_HAND_PROJECT_POLICY_ID or project_actions:
                raise ValueError("native project policy requires one symbolic policy authority")
        policies = {}
        native_policy = None
        other_native_policy = None
        if other_native_play_policy_id is not None and native_play_policy_id is None:
            raise ValueError("paired native play requires both policy identities")
        if native_play_policy_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            if play_policies is not None or escalation_observations is not None:
                raise ValueError("native visible play cannot mix policy authorities")
            native_policy = native_observer_safe_policy_identity_v1(native_play_policy_id, policy_team)
            if other_native_play_policy_id is not None:
                other_native_policy = native_observer_safe_policy_identity_v1(other_native_play_policy_id, policy_team.other)
        overrides = {}
        if native_play_overrides is not None:
            if (native_policy is None or other_native_policy is not None
                    or record_model_bytes or record_policy_model_hashes
                    or not isinstance(native_play_overrides, (list, tuple))):
                raise ValueError("guarded play overrides require one native policy and no legacy model accounting")
            for row in native_play_overrides:
                if (not isinstance(row, Mapping) or set(row) != {"information", "action"}
                        or type(row["action"]) is not str):
                    raise ValueError("guarded play overrides require information and action")
                info, _ = self._normalize_play_information(row["information"])
                key = self._round_observation_hash("play", info)
                if (Seat.parse(info["actor"]).team != policy_team or key in overrides
                        or len(info["publicPlays"]) < self._recorded_play_count):
                    raise ValueError("guarded play overrides must be unique future policy-team observations")
                overrides[key] = {"information": info, "action": row["action"]}
            overrides = dict(sorted(overrides.items()))
        if play_policies is not None:
            from .ortools_proof import DeclarativeActionPolicyV1

            if (not isinstance(play_policies, Mapping)
                    or set(play_policies) != {seat.name.lower() for seat in Seat if seat.team == policy_team}):
                raise ValueError("ranked play policies must cover exactly the policy team")
            policies = {seat: DeclarativeActionPolicyV1.build(seat, value)
                        for seat, value in sorted(play_policies.items())}
            if any(policy.choices or policy.materialization is None for policy in policies.values()):
                raise ValueError("ranked play requires unmaterialized declarative policy templates")
        observed = []
        if escalation_observations is not None:
            if (escalation_action is not None or not isinstance(escalation_observations, (list, tuple))
                    or self._native_round_fragment is None
                    or len(escalation_observations) > len(self._native_round_fragment.escalation_opportunities)):
                raise ValueError("observed escalation requires bounded rows and no constant escalation action")
            hashes = set()
            for row in escalation_observations:
                if (not isinstance(row, Mapping) or set(row) != {"information", "action"}
                        or type(row["action"]) is not str):
                    raise ValueError("observed escalation rows require information and action")
                info, _ = self._normalize_escalation_information(row["information"])
                key = self._escalation_observation_hash(info)
                if Seat.parse(info["actor"]).team != policy_team or key in hashes:
                    raise ValueError("observed escalation must be unique and owned by the policy team")
                hashes.add(key)
                observed.append({"information": info, "action": row["action"]})
        if (auction_action is None and native_auction_policy_id is None and escalation_action is None and not project_actions and native_project_policy_id is None
                and not policies and not observed and native_policy is None):
            raise ValueError("constant policy requires an action")
        fragment, branch = self.auction_fragment, self._native_round_fragment
        scope, _ = self._native_round_query_context()
        if scope.get("sha256") != canonical_sha256({k: v for k, v in scope.items() if k != "sha256"}):
            raise ValueError("constant policy branch scope integrity mismatch")
        if (not isinstance(fragment, OrToolsNativeAuctionFragmentV1)
                or any((self._observed_auction_bindings, self._observed_escalation_bindings,
                        self._observed_project_bindings, self._observed_play_bindings))):
            raise ValueError("constant policy requires an unbound native round branch")
        if auction_action is not None and (type(auction_action) is not str
                or auction_action not in fragment.legal_next_choice_literals):
            raise ValueError("unsupported constant auction action")
        escalation_codes = (branch.escalation_opportunities[0]["legal"] if branch is not None
            else (EscalationDomainIRV1.from_rules(self.rules.data).closure_action,))
        if escalation_action is not None and (type(escalation_action) is not str
                or escalation_action not in escalation_codes):
            raise ValueError("unsupported constant escalation action")
        for seat, action in project_actions.items():
            actor = Seat.parse(seat)
            if (actor.team != policy_team or branch is None
                    or branch.projects.declaration_actions is None
                    or action not in branch.projects.declaration_actions[int(actor)]):
                raise ValueError("unsupported constant project action or policy team")
        deadline = time.monotonic() + maximum_seconds
        source_proto = str(self.model.proto).encode("utf-8")
        source_hash, source_bytes = hashlib.sha256(source_proto).hexdigest(), len(source_proto)
        del source_proto
        core = {"schemaVersion": 1, "kind": "native_constant_policy_round_query_v1",
            "strategicCoverage": False, "sourceModelHash": self._model_hash(search_branching="automatic", proto_hash=source_hash),
            "scope": dict(scope), "rulesHash": self.rules.sha256,
            "targetTeam": target.name.lower(), "threshold": threshold,
            "policy": {"team": policy_team.name.lower(), "auctionAction": auction_action,
                "escalationAction": escalation_action, "auctionPrefixLength": fragment.prefix_length,
                "scope": "all_future_nonplay_information_sets"},
            "bindings": [], "query": None}
        if (auction_action is not None and self._native_auction_policy_root is not None
                and auction_action != self._native_auction_policy_root[1]
                and policy_team == self._native_auction_policy_root[2]):
            core["policy"]["auctionContinuationStart"] = self._native_auction_policy_root[0] + 1
        if project_actions:
            core["policy"]["projectActions"] = project_actions
        if native_project_policy_id is not None:
            core["policy"]["nativeProjectPolicyId"] = native_project_policy_id
        if native_auction_policy_id is not None:
            core["policy"]["nativeAuctionPolicyId"] = native_auction_policy_id
        if policies:
            core["policy"]["scope"] = "future_nonplay_constants_and_ranked_play"
            core["policy"]["playPolicies"] = {seat: policy.identity_dict() for seat, policy in policies.items()}
            core["policy"]["playPrefixLength"] = self._recorded_play_count
        if native_policy is not None:
            core["policy"].update(scope="future_nonplay_constants_and_native_visible_play",
                nativePlayPolicy=native_policy, playPrefixLength=self._recorded_play_count)
            if other_native_policy is not None:
                core["policy"].update(scope="future_nonplay_constants_and_paired_native_visible_play",
                    otherNativePlayPolicy=other_native_policy)
            if overrides:
                # Distinct authority: legacy constant-policy readers must not
                # reinterpret a refined strategy as the unchanged default.
                core["kind"] = "native_guarded_play_policy_round_query_v1"
                core["policy"].update(scope="future_nonplay_constants_and_guarded_native_visible_play",
                    playOverrides=overrides)
        if observed:
            core["policy"]["scope"] = "nonplay_constants_observation_guards_and_optional_ranked_play"
            core["policy"]["escalationObservations"] = observed
        if self._recorded_escalation_count or self._recorded_project_count:
            core["policy"].update(escalationPrefixLength=self._recorded_escalation_count,
                                  projectPrefixLength=self._recorded_project_count)

        def finish(status):
            result = {**core, "status": status}
            if record_model_bytes:
                phases = [source_bytes * 2]
                if "nativePlayPolicyModelBytes" in core["policy"]:
                    phases.append(source_bytes + core["policy"]["nativePlayPolicyModelBytes"])
                if "nativeProjectPolicyModelBytes" in core["policy"]:
                    phases.append(source_bytes + core["policy"]["nativeProjectPolicyModelBytes"])
                for binding in core["bindings"]:
                    sizes = binding["modelBytes"]
                    phases.append(source_bytes + sizes["owner"] + sizes["legalityProbe"])
                if core["query"] is not None:
                    sizes = core["query"]["modelBytes"]
                    phases.append(source_bytes + sizes["owner"] + sizes["outcomeQuery"])
                result["modelBytes"] = {"unit": "utf8_cp_model_proto", "source": source_bytes,
                    "retainedUpperBound": max(phases), "includesSolverWorkingMemory": False}
            if status == "incomplete":
                result["exhaustionReason"] = ("cancelled" if cancel_check is not None and cancel_check()
                    else "maximum_seconds" if time.monotonic() >= deadline
                    else (core["query"] or {}).get("exhaustionReason") or "policy_legality_incomplete")
            return {**result, "sha256": canonical_sha256(result)}

        def exhausted():
            return time.monotonic() >= deadline or (cancel_check is not None and cancel_check())

        if exhausted():
            return finish("incomplete")
        owned = copy.copy(self)
        owned.model = self.model.clone()
        owned._observed_escalation_bindings = {}
        owned._observed_play_bindings = {}
        if native_auction_policy_id is not None:
            choices = owned._native_visible_auction_policy_choices(policy_team, native_auction_policy_id)
            binding = owned._bind_native_policy_choice(
                observation_hash=canonical_sha256({"policy": core["policy"], "stage": "auction"}),
                action=native_auction_policy_id, choices=choices, kind="visible_auction", bindings={},
                deadline=deadline, maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                **({"record_model_bytes": True} if record_model_bytes else {}))
            core["bindings"].append(binding)
            if binding["status"] != "bound":
                return finish("invalid_policy" if binding["status"] == "invalid" else "incomplete")
            if exhausted():
                return finish("incomplete")
        for row in observed:
            if exhausted():
                return finish("incomplete")
            binding = owned.bind_native_escalation_observed_choice(**row,
                maximum_seconds=max(deadline - time.monotonic(), 1e-9),
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
            core["bindings"].append(binding)
            if binding["status"] != "bound":
                return finish("invalid_policy" if binding["status"] == "invalid" else "incomplete")
        for kind, action in (("auction", auction_action), ("escalation", escalation_action)):
            if action is None or (kind == "escalation" and branch is None):
                continue
            choices = owned._native_constant_nonplay_choices(policy_team, kind, action,
                escalation=branch.escalation if branch is not None else None,
                opportunities=branch.escalation_opportunities if branch is not None else ())
            binding = owned._bind_native_policy_choice(
                observation_hash=canonical_sha256({"policy": core["policy"], "stage": kind}),
                action=action, choices=choices, kind="constant_" + kind, bindings={},
                deadline=deadline, maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                **({"record_bound_model_hash": True} if record_policy_model_hashes else {}),
                **({"record_model_bytes": True} if record_model_bytes else {}))
            core["bindings"].append(binding)
            if binding["status"] != "bound":
                return finish("invalid_policy" if binding["status"] == "invalid" else "incomplete")
            if exhausted():
                return finish("incomplete")
        for seat, action in project_actions.items():
            actor = Seat.parse(seat)
            if actor in self.rules.project_domain.declaration_seats(self.dealer)[:self._recorded_project_count]:
                continue
            _, both, _ = _auction_logic_v1(owned.model, "constant_project_" + seat)
            selected = owned._native_project_selected_literal(actor, action, both)
            binding = owned._bind_native_policy_choice(
                observation_hash=canonical_sha256({"policy": core["policy"], "stage": "project", "actor": seat}),
                action=action, choices=((both("matches"), selected,
                    branch.projects.declaration_actions[int(actor)][action]),),
                kind="constant_project", bindings={}, deadline=deadline,
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                **({"record_model_bytes": True} if record_model_bytes else {}))
            core["bindings"].append(binding)
            if binding["status"] != "bound":
                return finish("invalid_policy" if binding["status"] == "invalid" else "incomplete")
            if exhausted():
                return finish("incomplete")
        if native_project_policy_id is not None and branch is not None:
            before = hashlib.sha256(str(owned.model.proto).encode("utf-8")).hexdigest()
            branch.projects.constrain_native_visible_policy(owned.model, team=policy_team,
                policy_id=native_project_policy_id,
                skipped_actors=self.rules.project_domain.declaration_seats(self.dealer)[:self._recorded_project_count])
            binding = {"schemaVersion": 1, "kind": "native_visible_project_policy_binding_v1",
                "policyId": native_project_policy_id, "team": policy_team.name.lower(),
                "sourceCpModelProtoHash": before,
                "boundCpModelProtoHash": hashlib.sha256(str(owned.model.proto).encode("utf-8")).hexdigest()}
            core["policy"]["nativeProjectPolicyBinding"] = {**binding, "sha256": canonical_sha256(binding)}
            if record_model_bytes:
                core["policy"]["nativeProjectPolicyModelBytes"] = len(str(owned.model.proto).encode("utf-8"))
            if exhausted():
                return finish("incomplete")
        if policies and branch is not None:
            if not replace(branch.play, model=owned.model).constrain_declarative_play_policies(
                    team=policy_team, policies=play_policies, start_step=self._recorded_play_count,
                    cancel_check=exhausted):
                return finish("incomplete")
            core["policy"]["playPolicyModelHash"] = owned._model_hash(search_branching="automatic")
        if native_policy is not None and branch is not None:
            if exhausted():
                return finish("incomplete")
            before = hashlib.sha256(str(owned.model.proto).encode("utf-8")).hexdigest()
            batch, override_matches = owned._bind_native_play_overrides(overrides.values(),
                maximum_seconds=max(deadline - time.monotonic(), 1e-9),
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
            core["bindings"].extend(batch)
            for binding in batch:
                if binding["status"] != "bound":
                    return finish("invalid_policy" if binding["status"] == "invalid" else "incomplete")
                if exhausted():
                    return finish("incomplete")
            # The fragment is shared; bind to a detached play view of the clone.
            # Mutating branch.play.model would change the caller's universe.
            if self._recorded_play_count < len(branch.play._extract.actors):
                replace(branch.play, model=owned.model).constrain_native_observer_safe_policy(
                    team=policy_team, policy_id=native_play_policy_id, start_step=self._recorded_play_count,
                    **({"_override_matches": override_matches} if override_matches else {}))
            from .ortools_play_tail import NATIVE_COMPACT_PARTNER_FEED_POLICY_ID
            if native_play_policy_id == NATIVE_COMPACT_PARTNER_FEED_POLICY_ID:
                from baloot_platform.core.scoring_ir import ScoringDomainIRV1
                from .ortools_scoring_stage import scoring_level_id_v1

                domain = ScoringDomainIRV1.from_rules(self.rules.data)
                if all(domain.kabout_base_game_points[contract] >= domain.card_game_point_totals[contract]
                       for contract in domain.card_game_point_totals):
                    # Redundant native bounds bridge award-case selectors during
                    # presolve. They do not select cards or assume a winner.
                    score = branch.scoring
                    equal, _, _ = _auction_logic_v1(owned.model, "compact_policy_award")
                    normal = equal(score.inputs.escalation, scoring_level_id_v1("normal"), "normal")
                    winner_ns = equal(score.round_winner, int(Team.NORTH_SOUTH), "winner_ns")
                    for team, wins in ((0, winner_ns), (1, winner_ns.Not())):
                        owned.model.add(score.awarded_points[team] >= score.total_before_award[team]
                            ).only_enforce_if([normal, wins])
                        owned.model.add(score.awarded_points[1-team] <= score.total_before_award[1-team]
                            ).only_enforce_if([normal, wins])
            proto = str(owned.model.proto).encode("utf-8")
            binding = {"schemaVersion": 1, "kind": "native_visible_play_policy_binding_v1",
                "policy": native_policy, "sourceCpModelProtoHash": before,
                "boundCpModelProtoHash": hashlib.sha256(proto).hexdigest()}
            if overrides:
                binding.update(kind="native_guarded_play_policy_binding_v1",
                    overridesHash=canonical_sha256(overrides))
            core["policy"]["nativePlayPolicyBinding"] = {**binding, "sha256": canonical_sha256(binding)}
            if other_native_policy is not None:
                if exhausted():
                    return finish("incomplete")
                before = binding["boundCpModelProtoHash"]
                if self._recorded_play_count < len(branch.play._extract.actors):
                    replace(branch.play, model=owned.model).constrain_native_observer_safe_policy(
                        team=policy_team.other, policy_id=other_native_play_policy_id,
                        start_step=self._recorded_play_count)
                proto = str(owned.model.proto).encode("utf-8")
                binding = {"schemaVersion": 1, "kind": "native_visible_play_policy_binding_v1",
                    "policy": other_native_policy, "sourceCpModelProtoHash": before,
                    "boundCpModelProtoHash": hashlib.sha256(proto).hexdigest()}
                core["policy"]["otherNativePlayPolicyBinding"] = {**binding, "sha256": canonical_sha256(binding)}
            core["policy"]["playPolicyModelHash"] = owned._model_hash(
                search_branching="automatic", proto_hash=binding["boundCpModelProtoHash"])
            if record_model_bytes:
                core["policy"]["nativePlayPolicyModelBytes"] = len(proto)
            if exhausted():
                return finish("incomplete")
        core["query"] = owned.find_native_round_margin_below(target=target, threshold=threshold,
            maximum_seconds=max(deadline - time.monotonic(), 1e-9),
            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
            **({"optimize_margin": False} if not optimize_margin else {}),
            **({"record_model_bytes": True} if record_model_bytes else {}))
        # A completed query remains evidence after its local allowance expires.
        # The caller still owns the campaign deadline; cancellation is distinct.
        cancelled = cancel_check is not None and cancel_check()
        return finish("incomplete" if cancelled or core["query"]["status"] == "incomplete" else "queried")

    def find_native_continuation_depth_above(
        self, *, maximum_depth: int, maximum_seconds: float, maximum_conflicts: int,
        cancel_check: Callable[[], bool] | None = None,
        record_model_bytes: bool = False,
        _source_proto: bytes | None = None,
    ) -> dict[str, Any]:
        """Find an over-depth legal continuation without truncating the universe."""
        if (type(record_model_bytes) is not bool or type(maximum_depth) is not int or maximum_depth < 1
                or type(maximum_conflicts) is not int or maximum_conflicts < 1
                or type(maximum_seconds) not in {int, float}
                or not isfinite(maximum_seconds) or maximum_seconds <= 0):
            raise ValueError("native depth query requires positive finite limits")
        if _source_proto is not None and type(_source_proto) is not bytes:
            raise TypeError("owned depth-query source must be serialized bytes")
        scope, _ = self._native_round_query_context()
        auction = self.auction_fragment
        deadline = time.monotonic() + maximum_seconds
        # Preparation may reuse its bytes only in the same, unmodified model state.
        source_proto = str(self.model.proto).encode("utf-8") if _source_proto is None else _source_proto
        source_bytes = len(source_proto)
        source_hash = self._model_hash(search_branching="automatic", proto_hash=hashlib.sha256(source_proto).hexdigest())
        del source_proto
        branch = self._native_round_fragment
        escalation_rows = (branch.escalation_opportunities if branch is not None
            and self._native_play_root is None and self._native_project_root is None else ())
        fixed = -auction.prefix_length
        if branch is not None:
            fixed += len(branch.projects.declaration_actions) + len(branch.play._extract.actors)
            fixed -= self._recorded_project_count + self._recorded_play_count
            if self._native_play_root is None and self._native_project_root is None:
                fixed -= self._recorded_escalation_count
        depth_upper = len(auction.choices) + len(escalation_rows) + max(0, fixed)
        # Dropping round constraints enlarges the universe. Only use this domain
        # relaxation when its slot bound excludes every over-depth witness;
        # tighter horizons still need the complete legal continuation model.
        relaxed = maximum_depth >= depth_upper
        model = cp_model.CpModel() if relaxed else self.model.clone()
        projected = {}
        def variable(value):
            if not relaxed:
                return model.get_int_var_from_proto_index(value.index)
            if value.index not in projected:
                projected[value.index] = model.new_int_var_from_domain(
                    cp_model.Domain.from_flat_intervals(value.proto.domain), f"depth_source_{value.index}")
            return projected[value.index]
        terms = [1 - variable(row["closed"]) for row in auction.choices]
        for index, row in enumerate(escalation_rows):
            active = variable(row["active"])
            if self._native_escalation_root_position is not None:
                _, both, _ = _auction_logic_v1(model, f"depth_escalation_root_{index}")
                future = model.new_bool_var(f"depth_escalation_future_{index}")
                model.add(variable(self._native_escalation_root_position) <= index).only_enforce_if(future)
                model.add(variable(self._native_escalation_root_position) > index).only_enforce_if(future.Not())
                active = both("active", active, future)
            terms.append(active)
        depth = model.new_int_var(0, depth_upper, "continuation_action_depth")
        model.add(depth == sum(terms) + fixed)
        model.add(depth > maximum_depth)
        model.clear_objective()
        query_proto = str(model.proto).encode("utf-8")
        query_hash, query_bytes = hashlib.sha256(query_proto).hexdigest(), len(query_proto)
        del query_proto
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        remaining = deadline - time.monotonic()
        stopped = "cancelled" if cancel_check is not None and cancel_check() else None
        if remaining <= 0:
            stopped = stopped or "maximum_seconds"
        status = cp_model.UNKNOWN
        if stopped is None:
            solver.parameters.max_time_in_seconds = remaining
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            if cancel_check is not None and cancel_check():
                stopped = "cancelled"
        if status == cp_model.MODEL_INVALID:
            raise ValueError("native continuation depth model is invalid")
        witness = stopped is None and status in {cp_model.FEASIBLE, cp_model.OPTIMAL}
        if relaxed and witness:
            raise ValueError("native depth domain relaxation contradicts its slot bound")
        core = {"schemaVersion": 1, "kind": "native_continuation_depth_query_v1",
            "sourceModelHash": source_hash, "queryModelHash": query_hash, "scopeHash": scope["sha256"],
            **({"queryScope": "decision_slot_domain_relaxation_v1"} if relaxed else {}),
            "maximumDepth": maximum_depth, "unit": "remaining_decision_actions",
            "includes": ["auction", "escalation", "project", "play"],
            "status": "over_depth_witness" if witness else "depth_bound_proven"
                if stopped is None and status == cp_model.INFEASIBLE else "incomplete",
            "solverStatus": solver.status_name(status).lower(),
            "witnessDepth": int(solver.value(depth)) if witness else None,
            "exhaustionReason": stopped, "maximumSeconds": maximum_seconds,
            "maximumConflicts": maximum_conflicts, "workers": 1,
            **({"modelBytes": {"unit": "utf8_cp_model_proto",
                "owner": source_bytes, "depthQuery": query_bytes}}
               if record_model_bytes else {})}
        return {**core, "sha256": canonical_sha256(core)}

    def find_native_round_margin_below(
        self, *, target: Team, threshold: int, maximum_seconds: float,
        maximum_conflicts: int, cancel_check: Callable[[], bool] | None = None,
        optimize_margin: bool = True,
        record_model_bytes: bool = False,
        deck_hint: Sequence[int] | None = None,
        play_card_hint: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Query an adverse legal outcome over the branch's symbolic deals.

        This is existential, not minimax: without a complete observation-keyed
        policy, a witness may include a target mistake. Infeasibility covers
        only this model's explicit branch and constraints, never all auctions.
        Objective bounds cover the threshold-filtered query. An unrestricted
        floor requires a full-domain threshold or accounting for excluded values.
        """
        if not isinstance(target, Team) or type(threshold) is not int:
            raise ValueError("round outcome query requires a team and integer threshold")
        if type(optimize_margin) is not bool:
            raise TypeError("round outcome optimization selection must be boolean")
        if type(record_model_bytes) is not bool:
            raise TypeError("native model-byte recording must be boolean")
        if deck_hint is not None and (not isinstance(deck_hint, (tuple, list))
                or len(deck_hint) != len(self.card_at_position)
                or any(type(card) is not int for card in deck_hint)
                or set(deck_hint) != set(self.rules.structural_domain.card_ids)):
            raise ValueError("native round deck hint requires one complete legal deck")
        if (type(maximum_seconds) not in {int, float} or not isfinite(maximum_seconds)
                or maximum_seconds <= 0 or type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("round outcome query requires positive finite resource bounds")
        branch = self._native_round_fragment
        if play_card_hint is not None and (branch is None
                or not isinstance(play_card_hint, (tuple, list))
                or len(play_card_hint) != len(branch.play.play_state_context.play_slots)
                or any(type(card) is not int for card in play_card_hint)
                or set(play_card_hint) != set(self.rules.structural_domain.card_ids)):
            raise ValueError("native round play hint requires one complete legal card sequence")
        scope, awarded_points = self._native_round_query_context()
        if scope.get("sha256") != canonical_sha256(
                {key: value for key, value in scope.items() if key != "sha256"}):
            raise ValueError("native round branch scope integrity mismatch")
        deadline = time.monotonic() + maximum_seconds
        source_proto = str(self.model.proto).encode("utf-8")
        source_bytes = len(source_proto)
        base_hash = self._model_hash(search_branching="automatic", proto_hash=hashlib.sha256(source_proto).hexdigest())
        del source_proto
        stopped = "cancelled" if cancel_check is not None and cancel_check() else None
        model = self.model.clone()
        hints = dict(zip(model.proto.solution_hint.vars, model.proto.solution_hint.values, strict=True))
        if deck_hint is not None:
            for variable, card in zip(self.card_at_position, deck_hint, strict=True):
                hints[variable.index] = card
        if play_card_hint is not None:
            for slot, card in zip(branch.play.play_state_context.play_slots, play_card_hint, strict=True):
                hints[slot.card.index] = card
        if deck_hint is not None or play_card_hint is not None:
            # Explicit hints replace the owner's value for that variable;
            # duplicate hint indices are invalid CP-SAT, not stronger guidance.
            model.clear_hints()
            for index, card in sorted(hints.items()):
                model.add_hint(model.get_int_var_from_proto_index(index), card)
        points = tuple(model.get_int_var_from_proto_index(value.index) for value in awarded_points)
        maximum = max(max(map(int, value.proto.domain)) for value in points)
        margin = model.new_int_var(-maximum, maximum, "round_query_awarded_margin")
        model.add(margin == points[int(target)] - points[int(target.other)])
        model.add(margin < threshold)
        if optimize_margin:
            model.minimize(margin)
        else:
            model.clear_objective()
        query_proto = str(model.proto).encode("utf-8")
        query_hash, query_bytes = hashlib.sha256(query_proto).hexdigest(), len(query_proto)
        del query_proto
        solver = cp_model.CpSolver()
        objective_bounds = []
        if optimize_margin:
            # UNKNOWN during presolve can have an unset/default zero bound.
            # Only a notified bound or a feasible/optimal response owns one.
            solver.best_bound_callback = objective_bounds.append
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        # Match the native tail witness search. Probing can consume the whole
        # deadline before branching, including on objective-bearing queries.
        # It strengthens neither a counterexample nor an OPTIMAL certificate.
        solver.parameters.cp_model_probing_level = 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stopped = stopped or "maximum_seconds"
        status = cp_model.UNKNOWN
        fixed_trace_probe = None
        fixed_world_probe = None
        search_hint_repair = None
        probe_conflicts = 0
        if (stopped is None and not optimize_margin
                and deck_hint is not None and play_card_hint is not None):
            # Validate a proposed counterexample natively before rediscovering it.
            # Fixed-hint infeasibility is never evidence about the unrestricted model.
            solver.parameters.max_time_in_seconds = remaining
            solver.parameters.fix_variables_to_their_hinted_value = True
            try:
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            finally:
                solver.parameters.fix_variables_to_their_hinted_value = False
            try:
                probe_conflicts, probe_branches = int(solver.num_conflicts), int(solver.num_branches)
            except RuntimeError:
                # Cancellation can precede solver initialization.
                probe_conflicts = probe_branches = 0
            if cancel_check is not None and cancel_check():
                stopped = "cancelled"
            fixed_trace_probe = {"solverStatus": solver.status_name(status).lower(),
                "scope": "fixed_hint_witness", "conflicts": probe_conflicts,
                "branches": probe_branches,
                "accepted": stopped is None and status in {cp_model.FEASIBLE, cp_model.OPTIMAL}}
            if status == cp_model.MODEL_INVALID:
                raise ValueError("native round outcome model is invalid")
            if not fixed_trace_probe["accepted"]:
                status = cp_model.UNKNOWN
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stopped = stopped or "maximum_seconds"
                elif probe_conflicts >= maximum_conflicts:
                    stopped = stopped or "maximum_conflicts"
                if stopped is None:
                    solver.parameters.max_number_of_conflicts = maximum_conflicts - probe_conflicts
        if stopped is None and fixed_trace_probe and not fixed_trace_probe["accepted"]:
            # A changed policy can refute the old line without escaping its
            # deal. Re-search that deal natively; absence is only local to it.
            world = model.clone()
            world.clear_hints()
            for variable, card in zip(self.card_at_position, deck_hint, strict=True):
                world.add_hint(world.get_int_var_from_proto_index(variable.index), card)
            world_hash = hashlib.sha256(str(world.proto).encode("utf-8")).hexdigest()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stopped = "maximum_seconds"
            else:
                # Share the existing allowance with unrestricted search.
                solver.parameters.max_time_in_seconds = remaining / 2
                solver.parameters.max_number_of_conflicts = max(1, (maximum_conflicts - probe_conflicts) // 2)
                solver.parameters.fix_variables_to_their_hinted_value = True
                previous_repair, previous_hint_limit = solver.parameters.repair_hint, solver.parameters.hint_conflict_limit
                # Complete the retained deal's changed line under this probe's
                # existing allowance, not the solver's ten-conflict hint default.
                solver.parameters.repair_hint = True
                solver.parameters.hint_conflict_limit = solver.parameters.max_number_of_conflicts
                repair_limit = solver.parameters.hint_conflict_limit
                try:
                    status = _solve_cp_sat_with_cancel_v1(solver, world, cancel_check)
                finally:
                    solver.parameters.fix_variables_to_their_hinted_value = False
                    solver.parameters.repair_hint = previous_repair
                    solver.parameters.hint_conflict_limit = previous_hint_limit
                try:
                    conflicts, branches = int(solver.num_conflicts), int(solver.num_branches)
                except RuntimeError:
                    conflicts = branches = 0
                probe_conflicts += conflicts
                if cancel_check is not None and cancel_check():
                    stopped = "cancelled"
                fixed_world_probe = {"solverStatus": solver.status_name(status).lower(),
                    "scope": "fixed_hint_world_witness", "modelHash": world_hash,
                    "conflicts": conflicts, "branches": branches,
                    "hintRepair": {"enabled": True, "conflictLimit": repair_limit},
                    "accepted": stopped is None and status in {cp_model.FEASIBLE, cp_model.OPTIMAL}}
                if status == cp_model.MODEL_INVALID:
                    raise ValueError("native round outcome model is invalid")
                if not fixed_world_probe["accepted"]:
                    status = cp_model.UNKNOWN
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        stopped = stopped or "maximum_seconds"
                    elif probe_conflicts >= maximum_conflicts:
                        stopped = stopped or "maximum_conflicts"
                    if stopped is None:
                        solver.parameters.max_number_of_conflicts = maximum_conflicts - probe_conflicts
        if stopped is None and not (fixed_trace_probe and fixed_trace_probe["accepted"]
                                    or fixed_world_probe and fixed_world_probe["accepted"]):
            solver.parameters.max_time_in_seconds = remaining
            if not optimize_margin and model.proto.solution_hint.vars:
                previous_repair, previous_hint_limit = solver.parameters.repair_hint, solver.parameters.hint_conflict_limit
                solver.parameters.repair_hint = True
                solver.parameters.hint_conflict_limit = solver.parameters.max_number_of_conflicts
                search_hint_repair = {"enabled": True, "conflictLimit": solver.parameters.hint_conflict_limit}
            try:
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            finally:
                if search_hint_repair is not None:
                    solver.parameters.repair_hint = previous_repair
                    solver.parameters.hint_conflict_limit = previous_hint_limit
            if cancel_check is not None and cancel_check():
                stopped = "cancelled"
        if status == cp_model.UNKNOWN and stopped is None:
            try:
                conflicts_exhausted = solver.num_conflicts + probe_conflicts >= maximum_conflicts
            except RuntimeError:
                # No response exists when a solve stops before initialization.
                conflicts_exhausted = False
            stopped = ("maximum_seconds" if time.monotonic() >= deadline else
                       "maximum_conflicts" if conflicts_exhausted else
                       "solver_unknown")
        if status == cp_model.MODEL_INVALID:
            raise ValueError("native round outcome model is invalid")
        if optimize_margin and status in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
            objective_bounds.append(solver.best_objective_bound)
        objective_lower = None
        if objective_bounds and status != cp_model.INFEASIBLE:
            if any(not isfinite(value) or not -maximum <= value <= maximum for value in objective_bounds):
                raise ValueError("native round objective bound is invalid")
            objective_lower = floor(max(objective_bounds))
        candidate = None
        has_witness = stopped is None and status in {cp_model.FEASIBLE, cp_model.OPTIMAL}
        if has_witness and branch is None:
            candidate = {"deckOrder": [solver.value(value) for value in self.card_at_position],
                "allPassed": bool(solver.value(self.auction_fragment.states[-1]["allPassed"])),
                "auctionActions": [row.to_dict() for row in self.auction_fragment.extract_actions(solver)],
                "auctionObservations": self._extract_native_auction_observations(solver),
                "escalationObservations": [], "projectObservations": [], "playObservations": [],
                "awardedPoints": [solver.value(value) for value in points], "utility": solver.value(margin),
                "continuationBoundary": scope["continuationBoundary"], "nextDealAnalyzed": False}
        elif has_witness:
            play = branch.play._extract(solver, status, query_hash)
            from baloot_platform.core.models import LegalPlayActionV1
            play_decisions = [{"playIndex": index, "actor": row.actor.name.lower(),
                "legalActions": [code for code, literal in branch.play.native_legal_action_literals(index).items()
                                 if solver.boolean_value(literal)],
                "selectedAction": LegalPlayActionV1(row.card, row.ekka).code}
                for index, row in enumerate(play.plays)]
            if any(row["selectedAction"] not in row["legalActions"] for row in play_decisions):
                raise ValueError("native play choice disagrees with legal action variables")
            auto, declared, _ = branch.public_baloot_binding
            baloot_events = [{"playIndex": index, "actor": row.actor.name.lower(), "kind": kind}
                for index, row in enumerate(play.plays)
                for kind, literal in (("included", auto[index]), ("declared", declared[index]))
                if solver.boolean_value(literal)]
            escalation = branch.escalation.extract_stage_result(solver, model_hash=query_hash,
                solver_status=solver.status_name(status).lower(),
                branches=int(solver.num_branches), conflicts=int(solver.num_conflicts))
            opportunities = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
            project_decisions = [branch.projects.extract_declaration_decision(solver, actor)
                for actor in self.rules.project_domain.declaration_seats(self.dealer)]
            project_observations = self._extract_native_project_observations(
                solver, opportunities, escalation, play, project_decisions)
            candidate = {"deckOrder": [solver.value(value) for value in self.card_at_position],
                "auctionActions": [row.to_dict() for row in self.auction_fragment.extract_actions(solver)],
                "auctionObservations": self._extract_native_auction_observations(solver),
                "escalationOpportunities": opportunities,
                "escalationObservations": self._extract_native_escalation_observations(solver, opportunities),
                "projectDecisions": project_decisions,
                "projectObservations": project_observations,
                "playDecisions": play_decisions, "publicBalootEvents": baloot_events,
                "playObservations": self._extract_native_play_observations(
                    solver, play, play_decisions, project_observations, baloot_events),
                "escalation": escalation.to_dict(), "projects": branch.projects.extract(solver).to_dict(),
                "play": play.to_dict(), "awardedPoints": [solver.value(value) for value in points],
                "utility": solver.value(margin)}
        if candidate is not None and objective_lower is not None and objective_lower > candidate["utility"]:
            raise ValueError("native round objective bound contradicts its witness: "
                f"bound={objective_lower}, utility={candidate['utility']}, "
                f"rawBounds={objective_bounds!r}, status={solver.status_name(status)}")
        if candidate is not None and optimize_margin and status == cp_model.OPTIMAL:
            # The solved integer objective is exact; its double bound can be one
            # representable value below it (e.g. 32.99999999999999 for 33).
            objective_lower = candidate["utility"]
        query_status = ("witness_found" if candidate is not None else "no_matching_outcome"
                        if stopped is None and status == cp_model.INFEASIBLE else "incomplete")
        core = {"schemaVersion": 1, "kind": "native_round_outcome_query_v1", "status": query_status,
            "scope": dict(scope), "baseModelHash": base_hash, "queryModelHash": query_hash,
            "targetTeam": target.name.lower(), "utility": "awarded_point_margin", "threshold": threshold,
            "auctionPolicyBindings": [dict(self._observed_auction_bindings[key])
                                      for key in sorted(self._observed_auction_bindings)],
            "escalationPolicyBindings": [dict(self._observed_escalation_bindings[key])
                                         for key in sorted(self._observed_escalation_bindings)],
            "projectPolicyBindings": [dict(self._observed_project_bindings[key])
                                      for key in sorted(self._observed_project_bindings)],
            "playPolicyBindings": [dict(self._observed_play_bindings[key])
                                   for key in sorted(self._observed_play_bindings)],
            "solverStatus": solver.status_name(status).lower(), "exhaustionReason": stopped,
            "candidate": candidate,
            **({"deckHintHash": canonical_sha256(list(deck_hint))} if deck_hint is not None else {}),
            **({"playCardHintHash": canonical_sha256(list(play_card_hint))} if play_card_hint is not None else {}),
            **({"fixedTraceProbe": fixed_trace_probe} if fixed_trace_probe is not None else {}),
            **({"fixedWorldProbe": fixed_world_probe} if fixed_world_probe is not None else {}),
            **({"hintRepair": search_hint_repair} if search_hint_repair is not None else {}),
            **({"objectiveLowerBound": objective_lower} if objective_lower is not None else {}),
            **({"queryMode": "feasibility_witness"} if not optimize_margin else {})}
        if record_model_bytes:
            core["modelBytes"] = {"unit": "utf8_cp_model_proto",
                "owner": source_bytes, "outcomeQuery": query_bytes}
        return {**core, "sha256": canonical_sha256(core)}

    def add_auction_fragment(
        self,
        *,
        required_actions: Sequence[OrToolsAuctionActionV1 | Mapping[str, Any]] = (),
        required_actions_complete: bool = False,
        required_purchase_card: Card | int | str | Mapping[str, Any] | None = None,
        allowed_contracts: Sequence[Contract | str] | None = None,
        prefix_length: int | None = None,
        canonical_completion_only: bool = False,
    ) -> OrToolsScenarioAuctionFragmentV1:
        """Add the complete Domain-IR auction table to this integrated model."""

        if self.auction_fragment is not None:
            raise ValueError("scenario model already contains an auction fragment")
        if type(required_actions_complete) is not bool:
            raise TypeError("required_actions_complete must be boolean")
        if type(canonical_completion_only) is not bool:
            raise TypeError("canonical_completion_only must be boolean")
        if isinstance(required_purchase_card, bool):
            raise TypeError("required_purchase_card must be a card")
        purchase_card = (
            Card.parse(required_purchase_card)
            if required_purchase_card is not None
            else None
        )
        if purchase_card is not None:
            self.model.add(self.purchase_card == purchase_card.id)

        domain = AuctionDomainIRV1.from_rules(self.rules.data)
        prefix = self._normalize_auction_prefix(required_actions, domain=domain)
        prefer_live_prefix = (
            prefix_length is not None
            and not prefix
            and not required_actions_complete
        )
        if prefix_length is None:
            prefix_length = len(prefix)
        if type(prefix_length) is not int or prefix_length < len(prefix):
            raise ValueError(
                "auction prefix length must cover every required action"
            )
        if required_actions_complete and prefix_length != len(prefix):
            raise ValueError(
                "a complete required auction uses its exact action length"
            )
        contracts = (
            frozenset(Contract)
            if allowed_contracts is None
            else frozenset(Contract.parse(item) for item in allowed_contracts)
        )
        plans: list[_AuctionPlanV1] = []

        def state_key(records: Sequence[OrToolsAuctionActionV1]):
            return tuple(
                (
                    record.round_index,
                    int(record.actor),
                    record.action,
                    int(record.trump) if record.trump is not None else _NO_TRUMP,
                )
                for record in records
            )

        purchase_suits = (
            (purchase_card.suit,) if purchase_card is not None else tuple(Suit)
        )
        purchase_ranks: tuple[Rank | None, ...] = (
            (purchase_card.rank,)
            if purchase_card is not None
            else (
                tuple(Rank)
                if domain.exposed_ace_hokum_sun_overcall == "dealer_right_only"
                else (None,)
            )
        )
        for purchase_suit in purchase_suits:
            for purchase_rank in purchase_ranks:
                terminal, states = _enumerate_auction(
                    domain,
                    dealer=self.dealer,
                    purchase_suit=purchase_suit,
                    purchase_rank=purchase_rank,
                )
                for state in terminal:
                    if state.all_passed:
                        continue
                    if state.contract is None or state.buyer is None:
                        raise AssertionError("purchased auction lacks a result")
                    if (
                        state.candidate is None
                        or state.candidate.caller is None
                        or state.candidate.purchase_recipient is None
                        or state.candidate.declarer is None
                    ):
                        raise AssertionError("purchased auction lacks typed roles")
                    if state.contract not in contracts:
                        continue
                    if (
                        self.required_buyer is not None
                        and state.buyer != self.required_buyer
                    ):
                        continue
                    if len(state.records) < prefix_length or any(
                        not required.matches(actual)
                        for required, actual in zip(prefix, state.records, strict=False)
                    ):
                        continue
                    if required_actions_complete and len(state.records) != len(prefix):
                        continue
                    prefix_records = state.records[:prefix_length]
                    prefix_state = states.get(state_key(prefix_records))
                    if prefix_state is None:
                        raise AssertionError("auction prefix state was not enumerated")
                    if prefix_state.terminal and prefer_live_prefix:
                        continue
                    choices = _auction_legal_choices(
                        domain,
                        prefix_state,
                        dealer=self.dealer,
                        purchase_suit=purchase_suit,
                        purchase_rank=purchase_rank,
                    )
                    legal_actions = tuple(sorted({action for action, _ in choices}))
                    legal_choices = tuple(
                        sorted(
                            f"hokum:{trump.name.lower()}"
                            if action == "hokum" and trump is not None
                            else action
                            for action, trump in choices
                        )
                    )
                    prefix_recipient = (
                        prefix_state.candidate.purchase_recipient
                        if prefix_state.terminal and prefix_state.candidate is not None
                        else None
                    )
                    prefix_caller = (
                        prefix_state.candidate.caller
                        if prefix_state.terminal and prefix_state.candidate is not None
                        else None
                    )
                    prefix_declarer = (
                        prefix_state.candidate.declarer
                        if prefix_state.terminal and prefix_state.candidate is not None
                        else None
                    )
                    core = {
                        "schemaVersion": 1,
                        "auctionDomainIrHash": domain.sha256,
                        "dealer": self.dealer.name.lower(),
                        "purchaseSuit": purchase_suit.name.lower(),
                        "actions": [action.to_dict() for action in state.records],
                        "caller": state.candidate.caller.name.lower(),
                        "buyer": state.buyer.name.lower(),
                        "purchaseRecipient": (
                            state.candidate.purchase_recipient.name.lower()
                        ),
                        "declarer": state.candidate.declarer.name.lower(),
                        "contract": state.contract.value,
                        "trumpSuit": (
                            state.trump.name.lower() if state.trump is not None else None
                        ),
                        "prefixLength": prefix_length,
                        "prefixTerminal": prefix_state.terminal,
                        "legalNextActions": list(legal_actions),
                        "legalNextChoices": list(legal_choices),
                    }
                    if purchase_rank is not None:
                        core["purchaseRank"] = purchase_rank.name.lower()
                    plans.append(
                        _AuctionPlanV1(
                            purchase_suit=purchase_suit,
                            purchase_rank=purchase_rank,
                            actions=state.records,
                            caller=state.candidate.caller,
                            buyer=state.buyer,
                            purchase_recipient=state.candidate.purchase_recipient,
                            declarer=state.candidate.declarer,
                            contract=state.contract,
                            trump=state.trump,
                            purchase_round=state.records[-1].round_index,
                            prefix_actions=prefix_records,
                            prefix_caller=prefix_caller,
                            prefix_buyer=(
                                prefix_state.buyer if prefix_state.terminal else None
                            ),
                            prefix_purchase_recipient=prefix_recipient,
                            prefix_declarer=prefix_declarer,
                            prefix_contract=(
                                prefix_state.contract if prefix_state.terminal else None
                            ),
                            prefix_trump=(
                                prefix_state.trump if prefix_state.terminal else None
                            ),
                            prefix_purchase_round=(
                                prefix_state.records[-1].round_index
                                if prefix_state.terminal and prefix_state.records
                                else None
                            ),
                            current_actor=(
                                None
                                if prefix_state.terminal
                                else _auction_actor(
                                    prefix_state,
                                    _auction_order(self.dealer),
                                )
                            ),
                            current_round=prefix_state.round_index,
                            prefix_terminal=prefix_state.terminal,
                            legal_next_actions=legal_actions,
                            legal_next_choices=legal_choices,
                            fingerprint=canonical_sha256(core),
                        )
                    )

        plans.sort(key=lambda plan: plan.fingerprint)
        if canonical_completion_only:
            canonical_plans: dict[tuple[Any, ...], _AuctionPlanV1] = {}
            for plan in plans:
                key = (
                    plan.purchase_suit,
                    plan.purchase_rank,
                    plan.caller,
                    plan.buyer,
                    plan.purchase_recipient,
                    plan.declarer,
                    plan.contract,
                    plan.trump,
                    plan.purchase_round,
                    plan.prefix_actions,
                    plan.prefix_caller,
                    plan.prefix_buyer,
                    plan.prefix_purchase_recipient,
                    plan.prefix_declarer,
                    plan.prefix_contract,
                    plan.prefix_trump,
                    plan.prefix_purchase_round,
                    plan.current_actor,
                    plan.current_round,
                    plan.prefix_terminal,
                    plan.legal_next_actions,
                    plan.legal_next_choices,
                )
                canonical_plans.setdefault(key, plan)
            plans = list(canonical_plans.values())
        selected = tuple(
            self.model.new_bool_var(f"auction_plan_{index}")
            for index in range(len(plans))
        )
        if selected:
            self.model.add_exactly_one(selected)
        else:
            self.model.add_bool_or([])

        contract = self.model.new_int_var(0, len(_CONTRACT_IDS) - 1, "auction_contract")
        trump = self.model.new_int_var(0, _NO_TRUMP, "auction_trump")
        purchase_round = self.model.new_int_var(1, 2, "auction_purchase_round")
        caller = self.model.new_int_var(0, 3, "auction_caller")
        declarer = self.model.new_int_var(0, 3, "auction_declarer")
        terminal = self.model.new_bool_var("auction_prefix_terminal")
        if selected:
            self.model.add(
                caller
                == sum(
                    int(plan.caller) * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self.model.add(
                self.buyer
                == sum(
                    int(plan.buyer) * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self.model.add(
                self.purchase_suit
                == sum(
                    int(plan.purchase_suit) * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            if plans[0].purchase_rank is not None:
                self.model.add(
                    self.purchase_rank
                    == sum(
                        int(plan.purchase_rank) * selected[index]
                        for index, plan in enumerate(plans)
                        if plan.purchase_rank is not None
                    )
                )
            self.model.add(
                self.purchase_recipient
                == sum(
                    int(plan.purchase_recipient) * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self.model.add(
                declarer
                == sum(
                    int(plan.declarer) * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self._recipient_authority_bound = True
            self.model.add(
                contract
                == sum(
                    _CONTRACT_IDS[plan.contract] * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self.model.add(
                trump
                == sum(
                    (int(plan.trump) if plan.trump is not None else _NO_TRUMP)
                    * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self.model.add(
                purchase_round
                == sum(
                    plan.purchase_round * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            self.model.add(
                terminal
                == sum(
                    int(plan.prefix_terminal) * selected[index]
                    for index, plan in enumerate(plans)
                )
            )
            partner_rows = [(int(seat), int(seat.partner)) for seat in Seat]
            for index, plan in enumerate(plans):
                if plan.contract != Contract.ASHKAL:
                    continue
                selected_plan = selected[index]
                self.model.add(declarer == self.buyer).only_enforce_if(
                    selected_plan
                )
                if domain.ashkal_contract_owner == "bidder":
                    self.model.add(caller == self.buyer).only_enforce_if(
                        selected_plan
                    )
                else:
                    self.model.add(
                        self.buyer == self.purchase_recipient
                    ).only_enforce_if(selected_plan)
                if domain.ashkal_taker == "bidder_partner":
                    self.model.add_allowed_assignments(
                        [caller, self.purchase_recipient], partner_rows
                    ).only_enforce_if(selected_plan)
                else:
                    self.model.add(
                        caller == self.purchase_recipient
                    ).only_enforce_if(selected_plan)

        action_names = sorted(
            {
                *domain.first_round_allowed,
                *domain.second_round_allowed,
            }
        )
        choice_names = sorted(
            {action for action in action_names if action != "hokum"}
            | {f"hokum:{suit.name.lower()}" for suit in Suit}
        )
        legal_action_literals = {
            action: self.model.new_bool_var(f"auction_legal_action_{action}")
            for action in action_names
        }
        legal_choice_literals = {
            choice: self.model.new_bool_var(
                "auction_legal_choice_" + choice.replace(":", "_")
            )
            for choice in choice_names
        }
        for action, literal in legal_action_literals.items():
            self.model.add(
                literal
                == sum(
                    selected[index]
                    for index, plan in enumerate(plans)
                    if action in plan.legal_next_actions
                )
            )
        for choice, literal in legal_choice_literals.items():
            self.model.add(
                literal
                == sum(
                    selected[index]
                    for index, plan in enumerate(plans)
                    if choice in plan.legal_next_choices
                )
            )
        if selected:
            self.model.add_decision_strategy(
                selected,
                cp_model.CHOOSE_FIRST,
                cp_model.SELECT_MAX_VALUE,
            )

        fragment = OrToolsScenarioAuctionFragmentV1(
            plans=tuple(plans),
            selected=selected,
            contract=contract,
            trump=trump,
            purchase_round=purchase_round,
            caller=caller,
            buyer=self.buyer,
            purchase_recipient=self.purchase_recipient,
            declarer=declarer,
            terminal=terminal,
            legal_next_action_literals=legal_action_literals,
            legal_next_choice_literals=legal_choice_literals,
            domain_hash=domain.sha256,
        )
        self._bind_purchase_recipient_ownership()
        self.auction_fragment = fragment
        return fragment

    def _model_hash(self, *, search_branching: str, proto_hash: str | None = None) -> str:
        # Callers may reuse a digest only at the same, unmodified model state.
        if proto_hash is None:
            proto_hash = hashlib.sha256(str(self.model.proto).encode("utf-8")).hexdigest()
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "ortools_scenario_deal_model_v1",
                "rulesHash": self.rules.sha256,
                "structuralDomainIrHash": self.rules.structural_domain.sha256,
                "dealer": self.dealer.name.lower(),
                "requiredBuyer": (
                    self.required_buyer.name.lower()
                    if self.required_buyer is not None
                    else None
                ),
                "noGoodHashes": list(self.no_good_hashes),
                "minimumDeckHammingDistance": self.minimum_deck_hamming_distance,
                "canonicalizeDealPackets": self.canonicalize_deal_packets,
                "searchBranching": search_branching,
                "auctionDomainIrHash": (
                    self.auction_fragment.domain_hash
                    if self.auction_fragment is not None
                    else None
                ),
                "cpModelProtoHash": proto_hash,
                **({"nativePlayRoot": self._native_play_root} if self._native_play_root is not None else {}),
                **({"nativeProjectRoot": self._native_project_root} if self._native_project_root is not None else {}),
                **({"nativeEscalationRoot": self._native_escalation_root} if self._native_escalation_root is not None else {}),
                **({"completeInformationDeckHash": canonical_sha256(list(self._complete_information_deck))}
                   if self._complete_information_deck is not None else {}),
                **({"authoredPrefixLengths": {"escalation": self._recorded_escalation_count,
                    "project": self._recorded_project_count, "play": self._recorded_play_count}}
                   if any((self._recorded_escalation_count, self._recorded_project_count,
                           self._recorded_play_count)) else {}),
            }
        )

    def solve(
        self,
        *,
        seed: int = 0,
        maximum_seconds: float = 15.0,
        maximum_conflicts: int = 250_000,
        search_branching: str = "fixed",
        cancel_check: Callable[[], bool] | None = None,
        solver: cp_model.CpSolver | None = None,
    ) -> OrToolsScenarioDealSolutionV1:
        if type(seed) is not int:
            raise TypeError("scenario solver seed must be an integer")
        if type(maximum_seconds) not in {int, float} or isinstance(
            maximum_seconds, bool
        ):
            raise TypeError("scenario solver maximum_seconds must be numeric")
        if maximum_seconds <= 0:
            raise ValueError("scenario solver maximum_seconds must be positive")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("scenario solver maximum_conflicts must be positive")
        if search_branching not in {"fixed", "automatic"}:
            raise ValueError("scenario solver search_branching is unsupported")

        if not self._recipient_authority_bound:
            self.model.add(self.purchase_recipient == self.buyer)
            self._recipient_authority_bound = True
        model_hash = self._model_hash(search_branching=search_branching)
        solver = solver if solver is not None else cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed & 0x7FFFFFFF
        solver.parameters.search_branching = (
            cp_model.FIXED_SEARCH
            if search_branching == "fixed"
            else cp_model.AUTOMATIC_SEARCH
        )
        solver.parameters.max_time_in_seconds = float(maximum_seconds)
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        cancelled_before_solve = None

        def checked_cancel():
            nonlocal cancelled_before_solve
            cancelled = bool(cancel_check and cancel_check())
            if cancelled_before_solve is None:
                cancelled_before_solve = cancelled
            return cancelled

        status = _solve_cp_sat_with_cancel_v1(solver, self.model,
            checked_cancel if cancel_check is not None else None)
        solver_status = solver.status_name(status).lower()
        # A pre-start cancellation has no response, or only a supplied solver's stale response.
        branches = 0 if cancelled_before_solve else int(solver.num_branches)
        conflicts = 0 if cancelled_before_solve else int(solver.num_conflicts)

        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            return OrToolsScenarioDealSolutionV1(
                "infeasible"
                if status == cp_model.INFEASIBLE
                else "not_found_within_budget",
                solver_status,
                self.dealer,
                None,
                None,
                None,
                None,
                None,
                self.purchase_position,
                None,
                None,
                self.no_good_hashes,
                self.minimum_deck_hamming_distance,
                model_hash,
                None,
                branches,
                conflicts,
                None,
                None,
                len(self.model.proto.variables),
                len(self.model.proto.constraints),
            )

        deck = tuple(int(solver.value(item)) for item in self.card_at_position)
        positions = tuple(int(solver.value(item)) for item in self.position_of_card)
        owners_at_positions = tuple(
            Seat(int(solver.value(item))) for item in self.owner_at_position
        )
        owners_of_cards = tuple(
            Seat(int(solver.value(item))) for item in self.owner_of_card
        )
        buyer = Seat(int(solver.value(self.buyer)))
        purchase_card = Card(int(solver.value(self.purchase_card)))
        purchase_suit = Suit(int(solver.value(self.purchase_suit)))
        purchase_recipient = Seat(int(solver.value(self.purchase_recipient)))
        auction = (
            self.auction_fragment.extract(solver)
            if self.auction_fragment is not None
            else None
        )
        if auction is not None and auction.completion_contract is None:
            buyer = purchase_recipient = None
            owners_at_positions = owners_of_cards = None
        core = {
            "schemaVersion": 1,
            "modelHash": model_hash,
            "dealer": self.dealer.name.lower(),
            "buyer": buyer.name.lower() if buyer is not None else None,
            "deckOrder": list(deck),
            "positionOfCard": list(positions),
            "ownerAtPosition": [seat.name.lower() for seat in owners_at_positions] if owners_at_positions is not None else None,
            "ownerOfCard": [seat.name.lower() for seat in owners_of_cards] if owners_of_cards is not None else None,
            "purchasePosition": self.purchase_position,
            "purchaseCard": purchase_card.id,
            "purchaseSuit": purchase_suit.name.lower(),
            "noGoodHashes": list(self.no_good_hashes),
            "minimumDeckHammingDistance": self.minimum_deck_hamming_distance,
            "auction": auction.to_dict() if auction is not None else None,
        }
        if purchase_recipient != buyer:
            core["purchaseRecipient"] = purchase_recipient.name.lower()
        return OrToolsScenarioDealSolutionV1(
            "satisfied",
            solver_status,
            self.dealer,
            buyer,
            deck,
            positions,
            owners_at_positions,
            owners_of_cards,
            self.purchase_position,
            purchase_card,
            purchase_suit,
            self.no_good_hashes,
            self.minimum_deck_hamming_distance,
            model_hash,
            canonical_sha256(core),
            branches,
            conflicts,
            auction,
            purchase_recipient,
            len(self.model.proto.variables),
            len(self.model.proto.constraints),
        )


__all__ = [
    "OrToolsAuctionActionV1",
    "OrToolsScenarioAuctionFragmentV1",
    "OrToolsScenarioAuctionObservationV1",
    "OrToolsScenarioDealSolutionV1",
    "OrToolsScenarioModelV1",
    "reify_scenario_expression",
]
