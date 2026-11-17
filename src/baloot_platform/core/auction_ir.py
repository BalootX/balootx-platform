from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .enums import Contract, Rank, Seat, Suit


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    # Local import keeps the core state machine independent of Research modules.
    import hashlib
    import json

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AuctionResolutionV1:
    contract: Contract | None
    trump: Suit | None
    caller: Seat | None
    purchase_recipient: Seat | None
    declarer: Seat | None
    buyer: Seat | None


@dataclass(frozen=True, slots=True)
class AuctionPreselectionV1:
    bidder: Seat
    resolution: AuctionResolutionV1


@dataclass(frozen=True, slots=True)
class AuctionDomainIRV1:
    """Validated auction semantics shared by runtime, reference, and CP-SAT."""

    allowed_contracts: tuple[Contract, ...]
    require_preselected: bool
    default_contract: Contract | None
    rounds: int
    pass_allowed: bool
    first_round_allowed: tuple[str, ...]
    second_round_allowed: tuple[str, ...]
    first_round_hokum_uses_purchase_suit: bool
    second_round_hokum_must_differ: bool
    ashkal_enabled: bool
    ashkal_priority_positions: tuple[int, ...]
    ashkal_taker: str
    ashkal_contract_owner: str
    ashkal_resolves_to: str
    exposed_ace_hokum_sun_overcall: str
    sha256: str

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "AuctionDomainIRV1":
        try:
            contracts = dict(rules["contracts"])
            bidding = dict(rules["bidding"])
            first = dict(bidding["first_round"])
            second = dict(bidding["second_round"])
            ashkal = dict(bidding["ashkal"])
            core = {
                "schemaVersion": 1,
                "allowedContracts": [str(item) for item in contracts["allowed"]],
                "requirePreselected": bool(contracts["require_preselected"]),
                "defaultContract": contracts["default"],
                "rounds": int(bidding["rounds"]),
                "passAllowed": bool(bidding["pass_allowed"]),
                "firstRoundAllowed": [str(item) for item in first["allowed"]],
                "secondRoundAllowed": [str(item) for item in second["allowed"]],
                "firstRoundHokumUsesPurchaseSuit": bool(
                    first["hokum_uses_purchase_suit"]
                ),
                "secondRoundHokumMustDiffer": bool(
                    second["hokum_must_differ_from_purchase_suit"]
                ),
                "ashkalEnabled": bool(ashkal["enabled"]),
                "ashkalPriorityPositions": [
                    int(item) for item in ashkal["eligible_priority_positions"]
                ],
                "ashkalTaker": str(ashkal["taker"]),
                "ashkalContractOwner": str(ashkal.get("contract_owner", "taker")),
                "ashkalResolvesTo": str(ashkal["resolves_to"]),
                "exposedAceHokumSunOvercall": str(
                    first.get("exposed_ace_hokum_sun_overcall", "priority_recall")
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("incomplete or invalid auction domain configuration") from exc

        allowed_contracts = tuple(
            Contract.parse(item) for item in core["allowedContracts"]
        )
        if not allowed_contracts or len(allowed_contracts) != len(
            set(allowed_contracts)
        ):
            raise ValueError("allowed contracts must be unique and non-empty")
        default_contract = (
            Contract.parse(core["defaultContract"])
            if core["defaultContract"] is not None
            else None
        )
        allowed_names = {item.value for item in allowed_contracts}
        valid_actions = {"pass", *allowed_names}
        for key in ("firstRoundAllowed", "secondRoundAllowed"):
            values = core[key]
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{key} must contain unique auction actions")
            unknown = set(values) - valid_actions
            if unknown:
                raise ValueError(f"{key} contains unsupported actions: {sorted(unknown)}")
            if ("pass" in values) != core["passAllowed"]:
                raise ValueError(f"{key} pass action conflicts with passAllowed")
        if core["rounds"] not in {1, 2}:
            raise ValueError("auction rounds must be one or two")
        if core["requirePreselected"]:
            if default_contract is None or default_contract not in allowed_contracts:
                raise ValueError("preselected auction requires an allowed default contract")
            if default_contract.value not in core["firstRoundAllowed"]:
                raise ValueError("preselected contract must be legal in the first round")
        if (
            "hokum" in core["firstRoundAllowed"]
        ) != core["firstRoundHokumUsesPurchaseSuit"]:
            raise ValueError("first-round Hokum action conflicts with its suit rule")
        if (
            "hokum" in core["secondRoundAllowed"]
        ) != core["secondRoundHokumMustDiffer"]:
            raise ValueError("second-round Hokum action conflicts with its suit rule")
        positions = tuple(core["ashkalPriorityPositions"])
        if len(positions) != len(set(positions)) or any(
            item not in {1, 2, 3, 4} for item in positions
        ):
            raise ValueError("Ashkal priority positions must be unique values from one to four")
        if core["ashkalTaker"] not in {"bidder", "bidder_partner"}:
            raise ValueError("unsupported Ashkal taker policy")
        if core["ashkalContractOwner"] not in {"bidder", "taker"}:
            raise ValueError("unsupported Ashkal contract-owner policy")
        if core["ashkalResolvesTo"] != "sun":
            raise ValueError("only Sun resolution is supported for Ashkal")
        if "ashkal" in core["firstRoundAllowed"] and not core["ashkalEnabled"]:
            raise ValueError("first-round Ashkal action requires Ashkal to be enabled")
        if core["exposedAceHokumSunOvercall"] not in {
            "priority_recall", "dealer_right_only",
        }:
            raise ValueError("unsupported exposed-Ace Hokum-to-Sun policy")

        return cls(
            allowed_contracts,
            core["requirePreselected"],
            default_contract,
            core["rounds"],
            core["passAllowed"],
            tuple(core["firstRoundAllowed"]),
            tuple(core["secondRoundAllowed"]),
            core["firstRoundHokumUsesPurchaseSuit"],
            core["secondRoundHokumMustDiffer"],
            core["ashkalEnabled"],
            positions,
            core["ashkalTaker"],
            core["ashkalContractOwner"],
            core["ashkalResolvesTo"],
            core["exposedAceHokumSunOvercall"],
            _canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "allowedContracts": [item.value for item in self.allowed_contracts],
            "requirePreselected": self.require_preselected,
            "defaultContract": (
                self.default_contract.value if self.default_contract is not None else None
            ),
            "rounds": self.rounds,
            "passAllowed": self.pass_allowed,
            "firstRoundAllowed": list(self.first_round_allowed),
            "secondRoundAllowed": list(self.second_round_allowed),
            "firstRoundHokumUsesPurchaseSuit": self.first_round_hokum_uses_purchase_suit,
            "secondRoundHokumMustDiffer": self.second_round_hokum_must_differ,
            "ashkalEnabled": self.ashkal_enabled,
            "ashkalPriorityPositions": list(self.ashkal_priority_positions),
            "ashkalTaker": self.ashkal_taker,
            "ashkalContractOwner": self.ashkal_contract_owner,
            "ashkalResolvesTo": self.ashkal_resolves_to,
            "exposedAceHokumSunOvercall": self.exposed_ace_hokum_sun_overcall,
            "sha256": self.sha256,
        }

    def legal_actions(
        self,
        *,
        round_index: int,
        position: int,
        in_recall: bool,
        candidate_contract: Contract | None,
        prior_round_actions: tuple[str, ...],
        purchase_rank: Rank | None = None,
    ) -> tuple[str, ...]:
        if round_index not in {1, 2} or round_index > self.rounds:
            raise ValueError("auction state references an unsupported round")
        if position not in {1, 2, 3, 4}:
            raise ValueError("auction state references an invalid priority position")
        allowed = (
            self.first_round_allowed
            if round_index == 1
            else self.second_round_allowed
        )
        actions = ["pass"] if self.pass_allowed and "pass" in allowed else []
        restricted = (
            round_index == 1
            and self.exposed_ace_hokum_sun_overcall == "dealer_right_only"
            and purchase_rank == Rank.ACE
            and candidate_contract == Contract.HOKUM
            and position != 1
        )
        if in_recall:
            if "sun" in allowed:
                if not restricted:
                    actions.append("sun")
            return tuple(actions)
        if candidate_contract is not None:
            if "sun" in allowed and candidate_contract != Contract.SUN and not restricted:
                actions.append("sun")
            return tuple(actions)
        for action in ("sun", "hokum"):
            if action in allowed:
                actions.append(action)
        if (
            "ashkal" in allowed
            and self.ashkal_enabled
            and position in self.ashkal_priority_positions
            and all(action == "pass" for action in prior_round_actions)
        ):
            actions.append("ashkal")
        return tuple(actions)

    def resolve_action(
        self,
        action: str,
        *,
        round_index: int,
        actor: Seat,
        purchase_suit: Suit,
        trump: Suit | None,
    ) -> AuctionResolutionV1:
        if action == "pass":
            return AuctionResolutionV1(None, None, None, None, None, None)
        contract = Contract.parse(action)
        if contract not in self.allowed_contracts:
            raise ValueError(f"contract {contract.value} is disabled")
        caller = actor
        purchase_recipient = (
            actor.partner
            if contract == Contract.ASHKAL and self.ashkal_taker == "bidder_partner"
            else actor
        )
        buyer = (
            actor
            if contract == Contract.ASHKAL and self.ashkal_contract_owner == "bidder"
            else purchase_recipient
        )
        declarer = (
            actor
            if contract == Contract.ASHKAL and self.ashkal_contract_owner == "bidder"
            else purchase_recipient
        )
        resolved_trump = None
        if contract == Contract.HOKUM:
            if round_index == 1:
                if not self.first_round_hokum_uses_purchase_suit:
                    raise ValueError("first-round Hokum purchase-suit rule is disabled")
                resolved_trump = purchase_suit
            else:
                if trump is None:
                    raise ValueError("second-round Hokum requires a trump suit")
                if self.second_round_hokum_must_differ and trump == purchase_suit:
                    raise ValueError("second-round Hokum requires a different trump suit")
                resolved_trump = trump
        return AuctionResolutionV1(
            contract,
            resolved_trump,
            caller,
            purchase_recipient,
            declarer,
            buyer,
        )

    def preselected_bid(
        self, *, dealer: Seat, purchase_suit: Suit
    ) -> AuctionPreselectionV1:
        if not self.require_preselected or self.default_contract is None:
            raise ValueError("auction profile does not require a preselected contract")
        buyer = dealer.next()
        bidder = (
            buyer.partner
            if self.default_contract == Contract.ASHKAL
            and self.ashkal_taker == "bidder_partner"
            else buyer
        )
        first = dealer.next()
        order = (first, first.next(), first.next().next(), dealer)
        position = order.index(bidder) + 1
        if (
            self.default_contract == Contract.ASHKAL
            and position not in self.ashkal_priority_positions
        ):
            raise ValueError("preselected Ashkal buyer does not map to an eligible bidder")
        resolution = self.resolve_action(
            self.default_contract.value,
            round_index=1,
            actor=bidder,
            purchase_suit=purchase_suit,
            trump=None,
        )
        if resolution.buyer != buyer:
            raise AssertionError("preselected auction resolution changed the buyer")
        if resolution.caller != bidder or resolution.declarer != resolution.buyer:
            raise AssertionError("preselected auction resolution changed a purchased role")
        return AuctionPreselectionV1(bidder, resolution)


__all__ = [
    "AuctionDomainIRV1",
    "AuctionPreselectionV1",
    "AuctionResolutionV1",
]
