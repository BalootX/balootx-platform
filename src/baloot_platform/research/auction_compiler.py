from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .contracts import canonical_sha256
from .reference_auction import ReferenceAuctionInterpreterV1


@dataclass(frozen=True, slots=True)
class AuctionActionV1:
    ordinal: int
    round_index: int
    actor: Seat
    action: str
    trump: Suit | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "roundIndex": self.round_index,
            "actor": self.actor.name.lower(),
            "action": self.action,
            "trumpSuit": self.trump.name.lower() if self.trump is not None else None,
        }


@dataclass(frozen=True, slots=True)
class AuctionPlanV1:
    actions: tuple[AuctionActionV1, ...]
    buyer: Seat
    declarer: Seat
    contract: Contract
    trump: Suit | None
    purchase_round: int
    model_hash: str
    solver_status: str
    sha256: str
    purchase_recipient: Seat | None = None
    caller: Seat | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "actions": [item.to_dict() for item in self.actions],
            "caller": self.caller.name.lower() if self.caller is not None else None,
            "buyer": self.buyer.name.lower(),
            "declarer": self.declarer.name.lower(),
            "contract": self.contract.value,
            "trumpSuit": self.trump.name.lower() if self.trump is not None else None,
            "purchaseRound": self.purchase_round,
            "modelHash": self.model_hash,
            "solverStatus": self.solver_status,
            "sha256": self.sha256,
        }
        if (
            self.purchase_recipient is not None
            and self.purchase_recipient != self.buyer
        ):
            value["purchaseRecipient"] = self.purchase_recipient.name.lower()
        return value


@dataclass(frozen=True, slots=True)
class _LegalAuction:
    actions: tuple[AuctionActionV1, ...]
    buyer: Seat
    declarer: Seat
    contract: Contract
    trump: Suit | None
    purchase_round: int
    fingerprint: str
    purchase_recipient: Seat | None = None
    caller: Seat | None = None


class OrToolsCpSatAuctionCompilerV1:
    """Compile the complete runtime auction transition table into CP-SAT.

    The auction is small (currently 111 terminal traces for the default
    profile), so an exact legal-trace table is both simpler and safer than a
    second handwritten set of auction constraints.
    """

    descriptor = PluginDescriptor(
        id="ortools_cp_sat_auction_compiler_v1",
        kind="domain_lowering",
        version="1.0.0",
        description="Exact legal-auction table compiled to deterministic CP-SAT selection",
        capabilities=(
            "complete_auction",
            "first_second_purchase",
            "buyer_declarer_contract_trump",
            "runtime_transition_table_parity",
        ),
        deterministic=True,
        resource_class="cpu",
        timeout_seconds=30,
        maximum_memory_mb=256,
        contract_id="cp_sat_lowering",
        contract_version="1",
        safety_guards=(
            "compiled_domain_authority",
            "authoritative_legality",
            "hard_gate_preservation",
        ),
    )

    @staticmethod
    def _choices(machine: Any) -> tuple[tuple[str, Suit | None], ...]:
        choices: list[tuple[str, Suit | None]] = []
        for action in machine.legal_actions():
            if action != "hokum":
                choices.append((action, None))
            elif machine.round_index == 1:
                choices.append(("hokum", machine.purchase_suit))
            else:
                choices.extend(
                    ("hokum", suit)
                    for suit in Suit
                    if suit != machine.purchase_suit
                )
        return tuple(choices)

    @staticmethod
    def _actions_from_result(result: Any) -> tuple[AuctionActionV1, ...]:
        records = getattr(result, "records", None)
        if records is None:
            records = getattr(result, "history", ())
        return tuple(
            AuctionActionV1(
                index,
                int(record.round_index),
                record.seat,
                str(record.action),
                record.trump,
            )
            for index, record in enumerate(records, start=1)
        )

    @staticmethod
    def _terminal_plan(
        result: Any,
        *,
        actions: tuple[AuctionActionV1, ...],
        dealer: Seat,
        purchase_suit: Suit,
    ) -> _LegalAuction | None:
        if result.all_passed:
            return None
        if (
            result.caller is None
            or result.buyer is None
            or result.declarer is None
            or result.contract is None
        ):
            raise AssertionError("completed purchased auction lacks semantic result")
        resolved_actions = actions or OrToolsCpSatAuctionCompilerV1._actions_from_result(
            result
        )
        if not resolved_actions:
            raise AssertionError("purchased auction lacks an action trace")
        core = {
            "dealer": dealer.name.lower(),
            "purchaseSuit": purchase_suit.name.lower(),
            "actions": [item.to_dict() for item in resolved_actions],
            "caller": result.caller.name.lower(),
            "buyer": result.buyer.name.lower(),
            "declarer": result.declarer.name.lower(),
            "contract": result.contract.value,
            "trumpSuit": (
                result.trump.name.lower() if result.trump is not None else None
            ),
        }
        if (
            result.purchase_recipient is not None
            and result.purchase_recipient != result.buyer
        ):
            core["purchaseRecipient"] = result.purchase_recipient.name.lower()
        return _LegalAuction(
            actions=resolved_actions,
            buyer=result.buyer,
            declarer=result.declarer,
            contract=result.contract,
            trump=result.trump,
            purchase_round=resolved_actions[-1].round_index,
            fingerprint=canonical_sha256(core),
            purchase_recipient=result.purchase_recipient,
            caller=result.caller,
        )

    @classmethod
    def _enumerate_auctions(
        cls,
        root: Any,
        *,
        dealer: Seat,
        purchase_suit: Suit,
    ) -> tuple[_LegalAuction, ...]:
        terminal: list[_LegalAuction] = []

        def visit(machine: Any, actions: tuple[AuctionActionV1, ...]) -> None:
            if machine.result is not None:
                plan = cls._terminal_plan(
                    machine.result,
                    actions=actions,
                    dealer=dealer,
                    purchase_suit=purchase_suit,
                )
                if plan is not None:
                    terminal.append(plan)
                return
            actor = machine.actor
            if actor is None:
                raise AssertionError("unfinished auction has no actor")
            for action_name, trump_suit in cls._choices(machine):
                # The frozen Domain IR is shared; all mutable auction state stays isolated.
                branch = copy.deepcopy(machine, {id(machine.domain): machine.domain})
                round_index = branch.round_index
                branch.submit(action_name, trump=trump_suit)
                action = AuctionActionV1(
                    len(actions) + 1,
                    round_index,
                    actor,
                    action_name,
                    trump_suit,
                )
                visit(branch, (*actions, action))

        visit(root, ())
        terminal.sort(key=lambda item: item.fingerprint)
        if len({item.fingerprint for item in terminal}) != len(terminal):
            raise AssertionError("legal auction compiler produced duplicate traces")
        return tuple(terminal)

    @classmethod
    @lru_cache(maxsize=128)
    def _initial_auction_table(
        cls,
        machine_type: type,
        domain: AuctionDomainIRV1,
        dealer: Seat,
        purchase_suit: Suit,
        purchase_rank: Rank | None,
    ) -> tuple[_LegalAuction, ...]:
        # Only immutable initial tables are reused; mutable partial roots are not.
        return cls._enumerate_auctions(
            machine_type(dealer, purchase_suit, domain, purchase_rank=purchase_rank),
            dealer=dealer,
            purchase_suit=purchase_suit,
        )

    @classmethod
    def _legal_auctions(
        cls,
        *,
        dealer: Seat,
        purchase_suit: Suit,
        rules: RuleBook,
        purchase_rank: Rank | None = None,
    ) -> tuple[_LegalAuction, ...]:
        domain = AuctionDomainIRV1.from_rules(rules.data)
        return cls._initial_auction_table(
            ReferenceAuctionInterpreterV1, domain, dealer, purchase_suit, purchase_rank,
        )

    @classmethod
    def _runtime_legal_auctions(
        cls,
        *,
        dealer: Seat,
        purchase_suit: Suit,
        rules: RuleBook,
        purchase_rank: Rank | None = None,
    ) -> tuple[_LegalAuction, ...]:
        return cls._initial_auction_table(
            BiddingMachine, AuctionDomainIRV1.from_rules(rules.data),
            dealer, purchase_suit, purchase_rank,
        )

    @classmethod
    def reference_runtime_parity(
        cls,
        *,
        dealer: Seat,
        purchase_suit: Suit,
        rules: RuleBook,
        purchase_rank: Rank | None = None,
    ) -> dict[str, Any]:
        reference = cls._legal_auctions(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
            purchase_rank=purchase_rank,
        )
        runtime = cls._runtime_legal_auctions(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
            purchase_rank=purchase_rank,
        )
        reference_fingerprints = [item.fingerprint for item in reference]
        runtime_fingerprints = [item.fingerprint for item in runtime]
        if reference_fingerprints != runtime_fingerprints:
            raise AssertionError(
                "DomainIR reference auctions differ from runtime auctions"
            )
        core = {
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "auctionDomainIrHash": AuctionDomainIRV1.from_rules(rules.data).sha256,
            "dealer": dealer.name.lower(),
            "purchaseSuit": purchase_suit.name.lower(),
            "traceCount": len(reference_fingerprints),
            "traceSetHash": canonical_sha256(reference_fingerprints),
            "targetsObserved": ["reference", "runtime"],
        }
        if purchase_rank is not None:
            core["purchaseRank"] = purchase_rank.name.lower()
        return {**core, "proofHash": canonical_sha256(core)}

    @staticmethod
    def _replay(
        plan: _LegalAuction,
        *,
        dealer: Seat,
        purchase_suit: Suit,
        rules: RuleBook,
        purchase_rank: Rank | None = None,
    ) -> None:
        machine = BiddingMachine(
            dealer,
            purchase_suit,
            AuctionDomainIRV1.from_rules(rules.data),
            purchase_rank,
        )
        if machine.result is not None:
            replayed_actions = OrToolsCpSatAuctionCompilerV1._actions_from_result(
                machine.result
            )
            if replayed_actions != plan.actions:
                raise AssertionError("compiled preselected auction trace differs from runtime")
            actual = (
                machine.result.caller,
                machine.result.buyer,
                machine.result.declarer,
                machine.result.contract,
                machine.result.trump,
                machine.result.purchase_recipient,
            )
            expected = (
                plan.caller,
                plan.buyer,
                plan.declarer,
                plan.contract,
                plan.trump,
                plan.purchase_recipient,
            )
            if actual != expected:
                raise AssertionError("compiled preselected auction result differs from runtime")
            return
        for action in plan.actions:
            if machine.actor != action.actor or machine.round_index != action.round_index:
                raise AssertionError("compiled auction actor or round differs from runtime")
            machine.submit(action.action, trump=action.trump)
        result = machine.result
        if result is None or result.all_passed:
            raise AssertionError("compiled auction did not purchase")
        actual = (
            result.caller,
            result.buyer,
            result.declarer,
            result.contract,
            result.trump,
            result.purchase_recipient,
        )
        expected = (
            plan.caller,
            plan.buyer,
            plan.declarer,
            plan.contract,
            plan.trump,
            plan.purchase_recipient,
        )
        if actual != expected:
            raise AssertionError("compiled auction result differs from runtime")

    def compile(
        self,
        *,
        dealer: Seat,
        purchase_suit: Suit,
        purchase_rank: Rank | None = None,
        rules: RuleBook,
        required_buyer: Seat | None = None,
        required_contracts: Iterable[Contract] = (),
        required_purchase_round: int | None = None,
        required_trump: Suit | None = None,
        required_actions: Iterable[AuctionActionV1 | Mapping[str, Any]] | None = None,
        required_actions_complete: bool = True,
        maximum_conflicts: int = 250_000,
        timeout_seconds: float = 15.0,
    ) -> AuctionPlanV1 | None:
        from ortools.sat.python import cp_model

        if required_purchase_round not in {None, 1, 2}:
            raise ValueError("purchase round must be one, two, or unrestricted")
        contracts = frozenset(Contract.parse(item) for item in required_contracts)
        required_trace = tuple(
            {
                "ordinal": (
                    item.ordinal
                    if isinstance(item, AuctionActionV1)
                    else int(item["ordinal"])
                    if item.get("ordinal") is not None
                    else None
                ),
                "roundIndex": (
                    item.round_index
                    if isinstance(item, AuctionActionV1)
                    else int(item["roundIndex"])
                    if item.get("roundIndex") is not None
                    else None
                ),
                "actor": (
                    item.actor
                    if isinstance(item, AuctionActionV1)
                    else Seat.parse(item["actor"])
                    if item.get("actor") is not None
                    else None
                ),
                "action": (
                    item.action
                    if isinstance(item, AuctionActionV1)
                    else str(item["action"]).strip().lower()
                ),
                "trump": (
                    item.trump
                    if isinstance(item, AuctionActionV1)
                    else Suit.parse(item["trumpSuit"])
                    if item.get("trumpSuit") is not None
                    else None
                ),
                "trumpSpecified": (
                    True
                    if isinstance(item, AuctionActionV1)
                    else "trumpSuit" in item
                ),
            }
            for item in (required_actions or ())
        )
        if any(
            item["ordinal"] is not None and item["ordinal"] != index
            for index, item in enumerate(required_trace, start=1)
        ):
            raise ValueError("required auction actions must have contiguous ordinals")

        def trace_matches(plan: _LegalAuction) -> bool:
            if required_actions_complete and len(plan.actions) != len(required_trace):
                return False
            if len(plan.actions) < len(required_trace):
                return False
            return all(
                (expected["roundIndex"] is None or actual.round_index == expected["roundIndex"])
                and (expected["actor"] is None or actual.actor == expected["actor"])
                and actual.action == expected["action"]
                and (
                    not expected["trumpSpecified"]
                    or actual.trump == expected["trump"]
                )
                for actual, expected in zip(
                    plan.actions, required_trace, strict=False
                )
            )
        plans = self._legal_auctions(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
            purchase_rank=purchase_rank,
        )
        parity = self.reference_runtime_parity(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
            purchase_rank=purchase_rank,
        )
        model = cp_model.CpModel()
        selected = [model.new_bool_var(f"auction_{index}") for index in range(len(plans))]
        model.add_exactly_one(selected)
        eligible: list[int] = []
        for index, plan in enumerate(plans):
            matches = (
                (required_buyer is None or plan.buyer == required_buyer)
                and (not contracts or plan.contract in contracts)
                and (required_purchase_round is None or plan.purchase_round == required_purchase_round)
                and (required_trump is None or plan.trump == required_trump)
                and (
                    not required_trace
                    or trace_matches(plan)
                )
            )
            if not matches:
                model.add(selected[index] == 0)
            else:
                eligible.append(index)
        # A stable fingerprint-based objective makes one-worker results
        # reproducible when multiple legal auction traces satisfy the recipe.
        costs = [int(plan.fingerprint[:12], 16) for plan in plans]
        model.minimize(sum(costs[index] * selected[index] for index in range(len(plans))))
        model_core = {
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "dealer": dealer.name.lower(),
            "purchaseSuit": purchase_suit.name.lower(),
            "requiredBuyer": required_buyer.name.lower() if required_buyer is not None else None,
            "requiredContracts": sorted(item.value for item in contracts),
            "requiredPurchaseRound": required_purchase_round,
            "requiredTrump": required_trump.name.lower() if required_trump is not None else None,
            "requiredActions": [
                {
                    "ordinal": item["ordinal"],
                    "roundIndex": item["roundIndex"],
                    "actor": (
                        item["actor"].name.lower()
                        if item["actor"] is not None
                        else None
                    ),
                    "action": item["action"],
                    "trumpSuit": (
                        item["trump"].name.lower()
                        if item["trump"] is not None
                        else None
                    ),
                    "trumpSpecified": item["trumpSpecified"],
                }
                for item in required_trace
            ],
            "requiredActionsComplete": bool(required_actions_complete),
            "legalPlanFingerprints": [item.fingerprint for item in plans],
            "eligiblePlanIndexes": eligible,
            "referenceRuntimeParityHash": parity["proofHash"],
        }
        if purchase_rank is not None:
            model_core["purchaseRank"] = purchase_rank.name.lower()
        model_hash = canonical_sha256(model_core)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_number_of_conflicts = int(maximum_conflicts)
        solver.parameters.max_time_in_seconds = float(timeout_seconds)
        status = solver.solve(model)
        if status == cp_model.INFEASIBLE:
            return None
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "CP-SAT auction compiler did not prove an optimal deterministic result: "
                + solver.status_name(status)
            )
        chosen = [index for index, variable in enumerate(selected) if solver.boolean_value(variable)]
        if len(chosen) != 1:
            raise AssertionError("CP-SAT auction compiler did not select exactly one plan")
        plan = plans[chosen[0]]
        self._replay(
            plan,
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
            purchase_rank=purchase_rank,
        )
        core = {
            "schemaVersion": 1,
            "actions": [item.to_dict() for item in plan.actions],
            "caller": plan.caller.name.lower() if plan.caller is not None else None,
            "buyer": plan.buyer.name.lower(),
            "declarer": plan.declarer.name.lower(),
            "contract": plan.contract.value,
            "trumpSuit": plan.trump.name.lower() if plan.trump is not None else None,
            "purchaseRound": plan.purchase_round,
            "modelHash": model_hash,
            "solverStatus": "optimal",
        }
        if plan.purchase_recipient is not None and plan.purchase_recipient != plan.buyer:
            core["purchaseRecipient"] = plan.purchase_recipient.name.lower()
        return AuctionPlanV1(
            actions=plan.actions,
            buyer=plan.buyer,
            declarer=plan.declarer,
            contract=plan.contract,
            trump=plan.trump,
            purchase_round=plan.purchase_round,
            model_hash=model_hash,
            solver_status="optimal",
            sha256=canonical_sha256(core),
            purchase_recipient=plan.purchase_recipient,
            caller=plan.caller,
        )
