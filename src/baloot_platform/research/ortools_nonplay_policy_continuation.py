"""Policy-conditioned non-play continuations using only shared Domain IR."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.escalation_ir import (
    EscalationDecisionV1,
    EscalationDomainIRV1,
)
from baloot_platform.core.models import Project
from baloot_platform.core.project_ir import ProjectDeclarationActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .ortools_complete_state_graph import (
    CompleteStateGraphSeedV1,
    OrToolsCompleteStateGraphCompilerV1,
)
from .ortools_nonplay_graphs import (
    AuctionGraphTerminalV1,
    EscalationGraphActionV1,
    EscalationGraphTerminalV1,
    ProjectDeclarationTerminalV1,
)
from .ortools_play_tail import (
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
    TailPrefixActionV1,
)
from .ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
    OrToolsProjectLifecycleResultV1,
)
from .ortools_proof import (
    DeclarativeActionPolicyV1,
    PolicyActionChoiceV1,
    ProofGraphEdgeV1,
    select_declarative_policy_action_v1,
)
from .scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


def composed_team_legal_action_order_v1(
    *,
    domain: EscalationDomainIRV1,
    team: Team,
    node_id: str,
    decisions: Sequence[EscalationDecisionV1],
    policies: Mapping[Seat, DeclarativeActionPolicyV1],
) -> tuple[EscalationDecisionV1, dict[str, object]]:
    """Choose one public team escalation action without inventing a seat turn."""

    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.code))
    if not ordered_decisions:
        raise ValueError("composed escalation policy requires legal decisions")
    members = tuple(sorted((seat for seat in Seat if seat.team == team), key=int))
    if any(seat not in policies for seat in members):
        raise ValueError("composed escalation policy requires both team members")
    identities = [
        {"seat": seat.name.lower(), **policies[seat].identity_dict()}
        for seat in members
    ]
    profile_core = {
        "schemaVersion": 1,
        "strategyId": "composed_team_legal_action_order_v1",
        "team": team.name.lower(),
        "domainIrHash": domain.sha256,
        "members": identities,
    }
    profile_hash = canonical_sha256(profile_core)
    selected = min(
        ordered_decisions,
        key=lambda item: (
            canonical_sha256(
                {
                    "schemaVersion": 1,
                    "profileHash": profile_hash,
                    "decision": OrToolsNonPlayPolicyContinuationV1._decision_core(item),
                }
            ),
            item.code,
        ),
    )
    choice_core: dict[str, object] = {
        **profile_core,
        "profileHash": profile_hash,
        "nodeId": node_id,
        "legalDecisions": [
            OrToolsNonPlayPolicyContinuationV1._decision_core(item)
            for item in ordered_decisions
        ],
        "selectedDecision": OrToolsNonPlayPolicyContinuationV1._decision_core(
            selected
        ),
    }
    return selected, {
        **choice_core,
        "materializedHash": canonical_sha256(choice_core),
    }


class OrToolsNonPlayPolicyContinuationV1:
    """Score one complete deal under frozen observer-safe action policies."""

    def __init__(
        self,
        rules: RuleBook,
        prefix: ExactTracePrefixV1,
        request: ScenarioProofRequestV1,
        state: Mapping[str, object],
        *,
        legal_actions: Sequence[Mapping[str, object]] = (),
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
        native_cp_sat: bool = False,
    ) -> None:
        if type(native_cp_sat) is not bool:
            raise TypeError("native_cp_sat must be bool")
        self.native_cp_sat = native_cp_sat
        if request.mode != "policy_conditioned" or request.policies is None:
            raise ValueError("non-play continuation requires declarative policies")
        self.rules = rules
        self.prefix = prefix
        self.request = request
        self.state = dict(state)
        self.legal_actions = tuple(dict(item) for item in legal_actions)
        self.scoring = ScoringDomainIRV1.from_rules(rules.data)
        if type(maximum_seconds) not in {int, float} or maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be positive")
        self._deadline = time.monotonic() + float(maximum_seconds)
        self._cancel_check = cancel_check
        self.policies = {
            Seat.parse(seat): DeclarativeActionPolicyV1.build(seat, value)
            for seat, value in request.policies.items()
        }
        self._continuations: list[dict[str, object]] = []
        self._auction_cache: dict[
            tuple[str, str, str | None, int], dict[str, object]
        ] = {}

    def auction_utility(self, terminal: AuctionGraphTerminalV1) -> int:
        """Complete one public auction outcome under the frozen policies."""

        stage_actions = tuple(
            self._auction_code(item.action, item.trump)
            for item in terminal.actions[len(self.prefix.auction_actions) :]
        )
        if not stage_actions:
            raise ValueError("auction continuation must add at least one action")
        auction_actions = tuple(
            {
                "actor": item.actor.name.lower(),
                "action": item.action,
                **(
                    {"trumpSuit": item.trump.name.lower()}
                    if item.trump is not None
                    else {}
                ),
            }
            for item in terminal.actions
        )
        if terminal.all_passed:
            return self._all_pass_utility(
                root_action=stage_actions[0],
                stage_actions=stage_actions,
                auction_actions=auction_actions,
            )
        if (
            terminal.buyer is None
            or terminal.contract is None
            or terminal.purchase_round is None
        ):
            raise ValueError("purchased auction terminal is incomplete")
        cache_key = (
            terminal.buyer.name.lower(),
            terminal.contract.value,
            terminal.trump.name.lower() if terminal.trump is not None else None,
            terminal.purchase_round,
        )
        cached = self._auction_cache.get(cache_key)
        if cached is not None:
            return self._append_cached_auction_continuation(
                cached,
                root_action=stage_actions[0],
                stage_actions=stage_actions,
                auction_actions=auction_actions,
            )
        escalation_actions, escalation_policy = self._select_escalation_continuation(
            buyer=terminal.buyer,
            contract=terminal.contract,
        )
        state = (
            EscalationDomainIRV1.from_rules(self.rules.data).initial_state()
            if not escalation_actions
            else self._resolved_escalation_state(
                buyer=terminal.buyer,
                contract=terminal.contract,
                actions=escalation_actions,
            )
        )
        utility, stored = self._round_utility(
            root_action=stage_actions[0],
            stage_actions=stage_actions,
            buyer=terminal.buyer,
            contract=terminal.contract,
            trump=terminal.trump,
            escalation=state.level,
            locked=state.locked,
            last_escalating_team=(
                escalation_actions[-1].actor.team if escalation_actions else None
            ),
            escalation_actions=escalation_actions,
            auction_actions=auction_actions,
            escalation_policy=escalation_policy,
        )
        self._auction_cache[cache_key] = {
            key: value
            for key, value in stored.items()
            if key not in {"rootAction", "stageActions", "sha256"}
        }
        return utility

    def escalation_utility(self, terminal: EscalationGraphTerminalV1) -> int:
        state = terminal.state
        new_actions = terminal.actions[len(self.prefix.escalation_actions) :]
        stage_actions = [self._escalation_code(item) for item in new_actions]
        if terminal.declined:
            stage_actions.append("decline")
        utility, _continuation = self._round_utility(
            root_action=stage_actions[0],
            stage_actions=tuple(stage_actions),
            buyer=Seat.parse(self._required_state("buyer")),
            contract=Contract.parse(self._required_state("contract")),
            trump=self._trump(),
            escalation=state.level,
            locked=state.locked,
            last_escalating_team=(
                terminal.actions[-1].actor.team if terminal.actions else None
            ),
            escalation_actions=terminal.actions,
        )
        return utility

    def project_utility(self, terminal: ProjectDeclarationTerminalV1) -> int:
        new_actions = terminal.actions[len(self.prefix.project_actions) :]
        if not new_actions:
            raise ValueError("project proof terminal lacks a root action")
        utility, _continuation = self._round_utility(
            root_action=new_actions[0].code,
            stage_actions=tuple(action.code for action in new_actions),
            buyer=Seat.parse(self._required_state("buyer")),
            contract=Contract.parse(self._required_state("contract")),
            trump=self._trump(),
            escalation=str(self._required_state("escalation")),
            locked=bool(self._required_state("locked")),
            last_escalating_team=self._last_escalating_team(),
            project_actions=terminal.actions,
            escalation_actions=tuple(
                EscalationGraphActionV1.build(
                    item["actor"],
                    item["action"],
                    item.get("mode", item.get("requestedMode")),
                )
                for item in self.prefix.escalation_actions
            ),
        )
        return utility

    @staticmethod
    def _escalation_code(action: EscalationGraphActionV1) -> str:
        return (
            action.action
            if action.mode is None
            else f"{action.action}:{action.mode}"
        )

    @staticmethod
    def _auction_code(action: str, trump: Suit | None) -> str:
        return (
            f"hokum:{trump.name.lower()}"
            if action == "hokum" and trump is not None
            else action
        )

    def _all_pass_utility(
        self,
        *,
        root_action: str,
        stage_actions: tuple[str, ...],
        auction_actions: Sequence[Mapping[str, object]],
    ) -> int:
        full_prefix = self.prefix.to_dict()
        full_prefix.pop("sha256", None)
        full_prefix.update(
            {
                "auctionActions": [dict(item) for item in auction_actions],
                "escalationActions": [],
                "projectDeclarations": [],
                "projectAssertions": [],
                "playActions": [],
                "stateAssertions": {},
            }
        )
        exact_prefix = ExactTracePrefixV1.build(full_prefix)
        core: dict[str, object] = {
            "schemaVersion": 1,
            "kind": "neutral_all_pass_research_continuation_v1",
            "rootAction": root_action,
            "stageActions": list(stage_actions),
            "exactTracePrefix": exact_prefix.to_dict(),
            "allPassed": True,
            "playable": False,
            "completionStage": "redeal",
            "utility": 0,
            "policyIdentities": {},
        }
        self._continuations.append({**core, "sha256": canonical_sha256(core)})
        return 0

    def _append_cached_auction_continuation(
        self,
        cached: Mapping[str, object],
        *,
        root_action: str,
        stage_actions: tuple[str, ...],
        auction_actions: Sequence[Mapping[str, object]],
    ) -> int:
        exact = dict(cached["exactTracePrefix"])
        exact.pop("sha256", None)
        exact["auctionActions"] = [dict(item) for item in auction_actions]
        exact_prefix = ExactTracePrefixV1.build(exact)
        core = {
            **dict(cached),
            "rootAction": root_action,
            "stageActions": list(stage_actions),
            "exactTracePrefix": exact_prefix.to_dict(),
        }
        self._continuations.append({**core, "sha256": canonical_sha256(core)})
        return int(cached["utility"])

    def _select_escalation_continuation(
        self,
        *,
        buyer: Seat,
        contract: Contract,
    ) -> tuple[tuple[EscalationGraphActionV1, ...], Mapping[str, object]]:
        domain = EscalationDomainIRV1.from_rules(self.rules.data)
        state = domain.initial_state()
        history: list[EscalationGraphActionV1] = []
        individual_choices = {seat: [] for seat in Seat}
        team_choices: list[dict[str, object]] = []
        while True:
            decisions = domain.legal_decisions(
                state,
                declarer=buyer,
                contract=contract,
                match_score=self.prefix.match_score,
                include_closure=True,
            )
            actors = tuple(
                sorted(
                    {item.actor for item in decisions if item.actor is not None},
                    key=int,
                )
            )
            if state.level == domain.default_level:
                controlling_team = buyer.team.other
            elif actors:
                teams = {item.team for item in actors}
                if len(teams) != 1:
                    raise ValueError("escalation continuation exposed opposing teams")
                controlling_team = next(iter(teams))
            elif history:
                controlling_team = history[-1].actor.team.other
            else:
                controlling_team = buyer.team.other
            node_id = "nonplay-escalation:" + canonical_sha256(
                {
                    "schemaVersion": 1,
                    "domainIrHash": domain.sha256,
                    "buyer": buyer.name.lower(),
                    "contract": contract.value,
                    "matchScore": list(self.prefix.match_score),
                    "state": {
                        "level": state.level,
                        "locked": state.locked,
                        "doubleSeat": (
                            state.double_seat.name.lower()
                            if state.double_seat is not None
                            else None
                        ),
                    },
                    "history": [self._escalation_action_dict(item) for item in history],
                    "legalDecisions": [self._decision_core(item) for item in decisions],
                }
            )[:24]
            if len(actors) > 1 or not actors:
                selected, receipt = self._select_composed_team_decision(
                    domain=domain,
                    team=controlling_team,
                    node_id=node_id,
                    decisions=decisions,
                )
                team_choices.append(receipt)
            else:
                actor = actors[0]
                edges = tuple(
                    ProofGraphEdgeV1.build(index, item.code, node_id)
                    for index, item in enumerate(decisions)
                )
                edge = select_declarative_policy_action_v1(self.policies[actor], edges)
                individual_choices[actor].append(
                    PolicyActionChoiceV1(node_id, edge.action_id)
                )
                selected = decisions[edge.action_id]
            if selected.is_closure:
                break
            assert selected.actor is not None
            history.append(
                EscalationGraphActionV1(
                    selected.actor,
                    selected.action,
                    selected.requested_mode,
                )
            )
            state = selected.state
        individual = {
            seat.name.lower(): self.policies[seat]
            .materialize(individual_choices[seat])
            .identity_dict()
            for seat in Seat
            if individual_choices[seat]
        }
        core: dict[str, object] = {
            "schemaVersion": 1,
            "strategyId": "composed_team_legal_action_order_v1",
            "domainIrHash": domain.sha256,
            "teamChoices": team_choices,
            "individualPolicyIdentities": individual,
        }
        return tuple(history), {**core, "sha256": canonical_sha256(core)}

    def _select_composed_team_decision(
        self,
        *,
        domain: EscalationDomainIRV1,
        team: Team,
        node_id: str,
        decisions: Sequence[EscalationDecisionV1],
    ) -> tuple[EscalationDecisionV1, dict[str, object]]:
        return composed_team_legal_action_order_v1(
            domain=domain,
            team=team,
            node_id=node_id,
            decisions=decisions,
            policies=self.policies,
        )

    @staticmethod
    def _decision_core(item: EscalationDecisionV1) -> dict[str, object]:
        return {
            "code": item.code,
            "actor": item.actor.name.lower() if item.actor is not None else None,
            "action": item.action,
            "requestedMode": item.requested_mode,
            "resolvedMode": item.resolved_mode,
        }

    @staticmethod
    def _escalation_action_dict(item: EscalationGraphActionV1) -> dict[str, object]:
        return {
            "actor": item.actor.name.lower(),
            "action": item.action,
            "mode": item.mode,
        }

    def _resolved_escalation_state(
        self,
        *,
        buyer: Seat,
        contract: Contract,
        actions: Sequence[EscalationGraphActionV1],
    ) -> Any:
        domain = EscalationDomainIRV1.from_rules(self.rules.data)
        state = domain.initial_state()
        for item in actions:
            state = domain.transition(
                state,
                declarer=buyer,
                contract=contract,
                match_score=self.prefix.match_score,
                actor=item.actor,
                action=item.action,
                mode=item.mode,
            ).state
        return state

    def selected_continuation(
        self,
        principal_variation: Sequence[str],
    ) -> Mapping[str, object]:
        """Return the one immutable terminal trace selected by the proof."""

        principal = tuple(str(item) for item in principal_variation)

        def matches_stage(item: Mapping[str, object]) -> bool:
            stages = tuple(str(value) for value in item["stageActions"])
            if len(principal) < len(stages):
                return False
            return all(
                actual == expected or actual.endswith(f":{expected}")
                for actual, expected in zip(
                    principal[: len(stages)], stages, strict=True
                )
            )

        matches = [
            item
            for item in self._continuations
            if matches_stage(item)
        ]
        if len(matches) != 1:
            raise ValueError("policy proof did not select one non-play continuation")
        return dict(matches[0])

    def selected_root_continuation(self, root_action: str) -> Mapping[str, object]:
        """Return the frozen-policy witness for one independently proved root."""

        root = str(root_action)
        matches = [
            item for item in self._continuations if item["rootAction"] == root
        ]
        if len(matches) != 1:
            raise ValueError("proof root did not select one non-play continuation")
        return dict(matches[0])

    def detected_projects(self) -> tuple[Project, ...]:
        """Return the authoritative original-hand projects for graph input."""

        buyer = Seat.parse(self._required_state("buyer"))
        lifecycle = self._project_lifecycle(
            hands=self._hands(buyer),
            contract=Contract.parse(self._required_state("contract")),
            trump=self._trump(),
        )
        return tuple(
            Project(
                item.owner,
                item.kind,
                item.cards,
                item.value,
                item.high_rank,
                item.category_priority,
            )
            for item in lifecycle.projects
            if item.selected
        )

    def _remaining_seconds(self) -> float:
        if self._cancel_check is not None and self._cancel_check():
            raise RuntimeError("policy continuation was cancelled")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("policy continuation exhausted maximum_seconds")
        return remaining

    def _project_lifecycle(
        self,
        *,
        hands: Mapping[Seat, Sequence[Card]],
        contract: Contract,
        trump: Suit | None,
        project_actions: Sequence[ProjectDeclarationActionV1] = (),
    ) -> OrToolsProjectLifecycleResultV1:
        return OrToolsProjectLifecycleModelV1().solve(
            hands=hands,
            contract=contract,
            dealer=Seat.parse(self.prefix.dealer),
            trump=trump,
            domain=self.rules.project_domain,
            timeout_seconds=self._remaining_seconds(),
            cancel_check=self._cancel_check,
            required_declaration_actions=project_actions,
            required_declaration_actions_complete=bool(project_actions),
        )

    def _required_state(self, key: str) -> object:
        value = self.state[key]
        if value is None:
            raise ValueError(f"non-play continuation state lacks {key}")
        return value

    def _trump(self) -> Suit | None:
        value = self.state.get("trumpSuit")
        return Suit.parse(value) if value is not None else None

    def _last_escalating_team(self) -> Team | None:
        history = self.state.get("escalationHistory")
        if not isinstance(history, list) or not history:
            return None
        team = history[-1].get("team")
        return Team[str(team).upper()] if team is not None else None

    def _hands(self, buyer: Seat) -> dict[Seat, list[Card]]:
        layout = DealLayoutV1.build(
            self.rules.structural_domain,
            dealer=Seat.parse(self.prefix.dealer),
            buyer=buyer,
        )
        assert layout.final_hand_positions is not None
        deck = self.prefix.deck_order
        return {
            seat: [
                Card.parse(deck[position])
                for position in layout.final_hand_positions[int(seat)]
            ]
            for seat in Seat
        }

    def _select(
        self,
        actor: Seat,
        actions: Sequence[tuple[int, str]],
        *,
        node_id: str,
        choices: dict[Seat, list[PolicyActionChoiceV1]],
    ) -> ProofGraphEdgeV1:
        edges = tuple(
            ProofGraphEdgeV1.build(action_id, code, node_id)
            for action_id, code in actions
        )
        selected = select_declarative_policy_action_v1(self.policies[actor], edges)
        choices[actor].append(PolicyActionChoiceV1(node_id, selected.action_id))
        return selected

    def continuation_action(
        self,
        actor: Seat,
        node_id: str,
        actions: Sequence[tuple[int, str]],
    ) -> int:
        edges = tuple(
            ProofGraphEdgeV1.build(action_id, code, node_id)
            for action_id, code in actions
        )
        return select_declarative_policy_action_v1(
            self.policies[actor], edges
        ).action_id

    def _round_utility(
        self,
        *,
        root_action: str,
        stage_actions: tuple[str, ...],
        buyer: Seat,
        contract: Contract,
        trump: Suit | None,
        escalation: str,
        locked: bool,
        last_escalating_team: Team | None,
        projects: Sequence[Project] | None = None,
        project_actions: Sequence[ProjectDeclarationActionV1] = (),
        escalation_actions: Sequence[EscalationGraphActionV1],
        auction_actions: Sequence[Mapping[str, object]] | None = None,
        escalation_policy: Mapping[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        hands = self._hands(buyer)
        initial_hands = {seat: tuple(cards) for seat, cards in hands.items()}
        lifecycle = self._project_lifecycle(
            hands=initial_hands,
            contract=contract,
            trump=trump,
            project_actions=project_actions,
        )
        if projects is not None:
            supplied = {
                (
                    item.owner,
                    item.kind,
                    item.value,
                    item.high_rank,
                    tuple(card.id for card in item.cards),
                )
                for item in projects
                if item.declared
            }
            native = {
                (
                    item.owner,
                    item.kind,
                    item.value,
                    item.high_rank,
                    tuple(card.id for card in item.cards),
                )
                for item in lifecycle.projects
                if item.declared
            }
            if supplied != native:
                raise ValueError(
                    "project decision disagrees with native CP-SAT lifecycle"
                )
        project_points = lifecycle.combined_awarded_value
        baloot_points = lifecycle.baloot_points

        leader = self.rules.play_domain.resolve_first_leader(
            dealer=Seat.parse(self.prefix.dealer),
            buyer=buyer,
            structural_domain=self.rules.structural_domain,
        )
        base_prefix = tuple(
            TailPrefixActionV1.build(
                item["actor"],
                item["card"],
                ekka=item.get("ekka") is True,
            )
            for item in self.prefix.play_actions
        )
        if len(base_prefix) > 3:
            raise ValueError(
                "project policy continuation supports only the first-trick root"
            )
        played = {action.card for action in base_prefix}
        remaining_hands = tuple(
            tuple(card for card in initial_hands[seat] if card not in played)
            for seat in Seat
        )
        next_actor = (
            self.rules.structural_domain.offset(leader, len(base_prefix))
        )
        if base_prefix and self.legal_actions and all(
            item.get("card") is not None for item in self.legal_actions
        ):
            legal_seed = tuple(
                TailPrefixActionV1.build(
                    next_actor,
                    item["card"],
                    ekka=item.get("ekka") is True,
                )
                for item in self.legal_actions
            )
        elif base_prefix:
            ownership = {
                card: seat for seat, cards in initial_hands.items() for card in cards
            }
            boundary = OrToolsPlayTailStageV1().solve(
                cards=tuple(sorted(ownership, key=lambda card: card.id)),
                ownership=ownership,
                contract=contract,
                trump=trump,
                first_leader=leader,
                play_domain=self.rules.play_domain,
                structural_domain=self.rules.structural_domain,
                prefix=base_prefix,
                locked=locked,
                decision_prefix_length=len(base_prefix),
                maximum_time_seconds=self._remaining_seconds(),
            )
            if boundary.solver_status != "optimal":
                raise RuntimeError("project play boundary legality did not complete")
            legal_seed = boundary.legal_play_actions
        else:
            legal_seed = tuple(
                TailPrefixActionV1(next_actor, card)
                for card in sorted(
                    remaining_hands[int(next_actor)],
                    key=lambda card: card.id,
                )
            )
        seed = CompleteStateGraphSeedV1.build(
            exact_state_id="STATE-POLICY-" + canonical_sha256(
                {
                    "schemaVersion": 1,
                    "rulesHash": self.rules.sha256,
                    "buyer": buyer.name.lower(),
                    "contract": contract.value,
                    "trump": trump.name.lower() if trump is not None else None,
                    "escalation": escalation,
                    "locked": locked,
                    "hands": {
                        seat.name.lower(): [card.id for card in initial_hands[seat]]
                        for seat in Seat
                    },
                }
            )[:24].upper(),
            rules_hash=self.rules.sha256,
            remaining_hands=remaining_hands,
            current_trick=base_prefix,
            first_leader=leader,
            next_actor=next_actor,
            legal_actions=legal_seed,
            contract=contract,
            trump=trump,
            buyer=buyer,
            locked=locked,
            base_raw=(0, 0),
            base_tricks=(0, 0),
            awarded_project_points=project_points,
            baloot_points=baloot_points,
            escalation=escalation,
            last_escalating_team=last_escalating_team,
            starting_match_score=self.prefix.match_score,
        )
        compiler = OrToolsCompleteStateGraphCompilerV1(
            play_domain=self.rules.play_domain,
            scoring_domain=self.scoring,
            structural_domain=self.rules.structural_domain,
            project_domain=self.rules.project_domain,
            native_conversion=self.native_cp_sat,
        )
        start = compiler._parse_start(
            seed,
            expected_rules_hash=self.rules.sha256,
        )
        target = Team[str(self.request.target_team).upper()]
        oracle = compiler._path_oracle(
            start,
            target=target,
            utility=self.request.utility,
            utility_encoding=self.request.utility_encoding,
            defer_terminal_scoring=True,
        )
        try:
            solved = oracle.solve_policy_path(
                self.policies,
                maximum_seconds=self._remaining_seconds(),
                cancelled=self._cancel_check,
            )
        except PlayTailSolveStopped as exc:
            raise RuntimeError(
                "native policy play did not complete for "
                f"buyer={buyer.name.lower()}, contract={contract.value}, "
                f"trump={trump.name.lower() if trump is not None else 'none'}, "
                f"escalation={escalation}: {exc.solver_status}; "
                f"evidence={exc.partial_receipt.get('evidence', {})}"
            ) from exc
        solved_play = solved.play
        path = tuple(solved_play.plays)
        policy_choices = {seat: [] for seat in Seat}
        for step, action in enumerate(path[len(base_prefix) :], start=len(base_prefix)):
            decision = solved_play.at_prefix(step)
            actor = decision.next_actor
            if actor is None:
                raise AssertionError("native policy trace ended before its final play")
            variants = tuple(decision.legal_play_actions)
            chosen = self._select(
                actor,
                [
                    (
                        item.card.id + 32 * int(item.ekka),
                        item.card.code + ("+ekka" if item.ekka else ""),
                    )
                    for item in variants
                ],
                node_id="nonplay-play:" + canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "actor": actor.name.lower(),
                        "hand": [
                            card.id
                            for card in decision.remaining_hands[int(actor)]
                        ],
                        "trick": [
                            {
                                "actor": item.actor.name.lower(),
                                "card": item.card.id,
                                "ekka": item.ekka,
                            }
                            for item in decision.current_trick
                        ],
                        "legalActions": [
                            {
                                "actionId": item.card.id + 32 * int(item.ekka),
                                "action": item.card.code
                                + ("+ekka" if item.ekka else ""),
                            }
                            for item in variants
                        ],
                        "contract": contract.value,
                        "trump": trump.name.lower() if trump is not None else None,
                        "locked": locked,
                    }
                )[:24],
                choices=policy_choices,
            )
            if chosen.action_id != action.card.id + 32 * int(action.ekka):
                raise AssertionError("native policy solve selected the wrong action")
        score = dict(solved.scoring_evidence)
        score["awardedPoints"] = list(score["gamePoints"])
        utility = solved.terminal_utility

        declarations = [
            {
                "owner": item.owner.name.lower(),
                "kind": item.kind,
                "value": item.value,
                "highRank": item.high_rank,
                "declarationOrder": item.declaration_order,
            }
            for item in sorted(
                (project for project in lifecycle.projects if project.declared),
                key=lambda item: int(item.declaration_order or 0),
            )
        ]
        play_actions = [
            {
                "actor": item.actor.name.lower(),
                "card": item.card.id,
                "ekka": item.ekka,
            }
            for item in path
        ]
        full_prefix = self.prefix.to_dict()
        full_prefix.pop("sha256", None)
        full_prefix.update(
            {
                **(
                    {"auctionActions": [dict(item) for item in auction_actions]}
                    if auction_actions is not None
                    else {}
                ),
                "escalationActions": [
                    {
                        "actor": item.actor.name.lower(),
                        "action": item.action,
                        **({"mode": item.mode} if item.mode is not None else {}),
                    }
                    for item in escalation_actions
                ],
                "projectDeclarations": declarations,
                "projectAssertions": [],
                "playActions": play_actions,
                "stateAssertions": {},
            }
        )
        resolved_project_actions = tuple(
            project_actions or lifecycle.declaration_actions
        )
        if resolved_project_actions:
            full_prefix["projectActions"] = [
                action.to_dict() for action in resolved_project_actions
            ]
        exact_prefix = ExactTracePrefixV1.build(full_prefix)
        materialized = {
            seat.name.lower(): self.policies[seat]
            .materialize(policy_choices[seat])
            .identity_dict()
            for seat in Seat
        }
        continuation_core: dict[str, object] = {
            "schemaVersion": 1,
            "kind": "policy_scored_round_continuation_v1",
            "rootAction": root_action,
            "stageActions": list(stage_actions),
            "exactTracePrefix": exact_prefix.to_dict(),
            "score": score,
            "utility": utility,
            "allPassed": False,
            "playable": True,
            "completionStage": "scored",
            "policyIdentities": materialized,
            **(
                {"escalationPolicy": dict(escalation_policy)}
                if escalation_policy is not None
                else {}
            ),
        }
        continuation = {
            **continuation_core,
            "sha256": canonical_sha256(continuation_core),
        }
        self._continuations.append(continuation)
        return utility, continuation


__all__ = [
    "OrToolsNonPlayPolicyContinuationV1",
    "composed_team_legal_action_order_v1",
]
