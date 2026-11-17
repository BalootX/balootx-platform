"""Unified, budgeted proof service for Research V4 exact states."""

from __future__ import annotations

import copy
import importlib.resources
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding import _choose_action
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import FULL_DECK, Card, mask_to_cards
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1, RoundState
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.bidding import legal_bid_choices

from .exact_state import NormalizedExactStateV1
from .scenario_contracts import ProofOutcomeV1, ScenarioProofBudgetV1, ScenarioProofRequestV1, canonical_sha256


class ProofBudgetExceeded(RuntimeError):
    pass


class ScenarioProofBudgetRegistryV1:
    def __init__(self, profiles: Mapping[str, ScenarioProofBudgetV1], sha256: str):
        self._profiles = dict(profiles)
        self.sha256 = sha256

    @classmethod
    def load(cls, path: Path) -> "ScenarioProofBudgetRegistryV1":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or set(raw) != {
            "schemaVersion",
            "id",
            "version",
            "profiles",
        }:
            raise ValueError("scenario proof budget document is invalid")
        if raw["schemaVersion"] != 1:
            raise ValueError("scenario proof budget schema is unsupported")
        profiles_raw = raw["profiles"]
        if not isinstance(profiles_raw, Mapping) or not profiles_raw:
            raise ValueError("scenario proof budgets are empty")
        profiles = {
            str(profile_id): ScenarioProofBudgetV1.build(str(profile_id), value)
            for profile_id, value in profiles_raw.items()
        }
        return cls(profiles, canonical_sha256(raw))

    def get(self, profile_id: str) -> ScenarioProofBudgetV1:
        try:
            return self._profiles[str(profile_id)]
        except KeyError as exc:
            raise ValueError(f"unknown scenario proof budget: {profile_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "sha256": self.sha256,
            "profiles": {
                key: {
                    "profileId": value.profile_id,
                    "maximumNodes": value.maximum_nodes,
                    "maximumDepth": value.maximum_depth,
                    "maximumHorizonActions": value.maximum_horizon_actions,
                    "maximumHiddenWorlds": value.maximum_hidden_worlds,
                    "maximumWallSeconds": value.maximum_wall_seconds,
                    "maximumCpuSeconds": value.maximum_cpu_seconds,
                    "maximumCacheEntries": value.maximum_cache_entries,
                    "maximumCacheBytes": value.maximum_cache_bytes,
                    "cancellationCheckIntervalNodes": value.cancellation_check_interval_nodes,
                    "maximumWorkers": value.maximum_workers,
                    "maximumLineCountNodes": value.maximum_line_count_nodes,
                    "maximumStateSpace": value.maximum_state_space,
                    **({"policySearch": value.policy_search} if value.policy_search is not None else {}),
                    **({"nativeObserved": value.native_observed.to_dict()}
                       if value.native_observed is not None else {}),
                    "sha256": value.sha256,
                }
                for key, value in sorted(self._profiles.items())
            },
        }

    @classmethod
    def load_default(cls) -> "ScenarioProofBudgetRegistryV1":
        repository_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "research"
            / "proof_budgets_v1.yaml"
        )
        if repository_path.is_file():
            return cls.load(repository_path)
        resource = importlib.resources.files("baloot_platform.resources").joinpath(
            "configs/research/proof_budgets_v1.yaml"
        )
        with importlib.resources.as_file(resource) as resource_path:
            return cls.load(resource_path)


@dataclass(slots=True)
class _Counter:
    maximum_nodes: int
    deadline_wall: float
    deadline_cpu: float
    cancellation_check_interval_nodes: int
    cancel_check: Any | None = None
    nodes: int = 0
    cancelled: bool = False
    exhaustion_reason: str | None = None
    forecast_state_space: int | None = None
    retained_cache_entries: int = 0
    retained_cache_bytes: int = 0

    def visit(self) -> None:
        if self.nodes >= self.maximum_nodes:
            self._exhaust("maximum_nodes")
        self.nodes += 1
        self.poll(
            check_cancellation=(
                self.nodes == 1
                or self.nodes % self.cancellation_check_interval_nodes == 0
            )
        )

    def poll(self, *, check_cancellation: bool = True) -> None:
        if time.monotonic() > self.deadline_wall:
            self._exhaust("maximum_wall_seconds")
        if time.thread_time() > self.deadline_cpu:
            self._exhaust("maximum_cpu_seconds")
        if (
            self.cancel_check is not None
            and check_cancellation
            and self.cancel_check()
        ):
            self.cancelled = True
            self._exhaust("cancelled")

    def finish(self) -> None:
        if self.exhaustion_reason is not None:
            return
        try:
            self.poll()
        except ProofBudgetExceeded:
            pass

    def _exhaust(self, reason: str) -> None:
        self.exhaustion_reason = reason
        raise ProofBudgetExceeded(f"scenario proof budget exhausted: {reason}")


@dataclass(slots=True)
class _ProofCache:
    maximum_entries: int
    maximum_bytes: int
    values: dict[tuple[Any, ...], "_Evaluation"] = field(default_factory=dict)
    estimated_bytes: int = 0

    def get(self, key: tuple[Any, ...]) -> "_Evaluation | None":
        return self.values.get(key)

    def put(self, key: tuple[Any, ...], value: "_Evaluation") -> None:
        if key in self.values or len(self.values) >= self.maximum_entries:
            return
        size = sys.getsizeof({key: value}) + sys.getsizeof(key) + sys.getsizeof(value)
        size += sum(sys.getsizeof(item) for item in key)
        size += sum(sys.getsizeof(item) for item in value.principal_variation)
        size += sum(
            sys.getsizeof(item)
            for item in (
                value.value,
                value.all_line_count,
                value.winning_line_count,
                value.optimal_line_count,
            )
        )
        if self.estimated_bytes + size > self.maximum_bytes:
            return
        self.values[key] = value
        self.estimated_bytes += size


@dataclass(frozen=True, slots=True)
class _Evaluation:
    value: float
    principal_variation: tuple[str, ...]
    all_line_count: int
    winning_line_count: int
    optimal_line_count: int


_ProofPlayAction = LegalPlayActionV1


def decode_proof_play_action_v1(value: str) -> tuple[Card, bool]:
    """Decode the canonical card/Ekka token retained in proof witnesses."""

    action = LegalPlayActionV1.from_code(value)
    return action.card, action.ekka


def _legal_proof_play_actions(
    engine: BalootEngine,
    state: RoundState,
) -> tuple[LegalPlayActionV1, ...]:
    return engine.legal_play_actions(state)


def _root_action_upper_bound(engine: BalootEngine, state: RoundState) -> int:
    legal = engine.legal_actions(state)
    trick = state.current_trick
    if (
        state.current_player is None
        or trick is None
        or trick.plays
        or state.trump_suit is None
        or not engine.play_domain.ekka_enabled
    ):
        return len(legal)
    ekka_suits = {
        card.suit for card in legal if card.suit != state.trump_suit
    }
    return len(legal) + len(ekka_suits)


@dataclass(slots=True)
class _InformationSetCatalog:
    maximum_entries: int
    maximum_bytes: int
    maximum_state_space: int
    state_space_offset: int
    domains: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    estimated_bytes: int = 0
    tree_nodes: int = 0
    built_states: int = 0

    def __post_init__(self) -> None:
        self.estimated_bytes = sys.getsizeof(self) + sys.getsizeof(self.domains)

    def visit_state(self, counter: _Counter) -> None:
        required = self.state_space_offset + self.built_states + 1
        counter.forecast_state_space = max(
            counter.forecast_state_space or 0,
            required,
        )
        if self.built_states >= self.maximum_state_space:
            counter._exhaust("maximum_state_space")
        self.built_states += 1
        counter.retained_cache_bytes = max(
            counter.retained_cache_bytes,
            self.estimated_bytes,
        )

    def add(
        self,
        key: tuple[str, str],
        actions: tuple[str, ...],
        counter: _Counter,
    ) -> None:
        counter.forecast_state_space = max(
            counter.forecast_state_space or 0,
            counter.nodes,
        )
        existing = self.domains.get(key)
        if existing is not None:
            if existing != actions:
                counter._exhaust("actor_information_set_action_domain_mismatch")
            return
        if len(self.domains) + self.tree_nodes >= self.maximum_entries:
            counter._exhaust("maximum_cache_entries")
        size = (
            sys.getsizeof({key: actions})
            + sys.getsizeof(key)
            + sum(sys.getsizeof(item) for item in key)
            + sys.getsizeof(actions)
            + sum(sys.getsizeof(item) for item in actions)
        )
        if self.estimated_bytes + size > self.maximum_bytes:
            counter._exhaust("maximum_cache_bytes")
        self.domains[key] = actions
        self.estimated_bytes += size
        counter.retained_cache_entries = max(
            counter.retained_cache_entries,
            self.retained_entries,
        )
        counter.retained_cache_bytes = max(
            counter.retained_cache_bytes,
            self.estimated_bytes,
        )
        counter.forecast_state_space = max(
            counter.forecast_state_space or 0,
            counter.nodes,
        )

    def retain_tree_node(self, size: int, counter: _Counter) -> None:
        counter.forecast_state_space = max(
            counter.forecast_state_space or 0,
            counter.nodes,
        )
        if len(self.domains) + self.tree_nodes >= self.maximum_entries:
            counter._exhaust("maximum_cache_entries")
        if self.estimated_bytes + size > self.maximum_bytes:
            counter._exhaust("maximum_cache_bytes")
        self.tree_nodes += 1
        self.estimated_bytes += size
        counter.retained_cache_entries = max(
            counter.retained_cache_entries,
            self.retained_entries,
        )
        counter.retained_cache_bytes = max(
            counter.retained_cache_bytes,
            self.estimated_bytes,
        )
        counter.forecast_state_space = max(
            counter.forecast_state_space or 0,
            counter.nodes,
        )

    def retain_container_bytes(self, size: int, counter: _Counter) -> None:
        """Account retained forest containers that are not tree nodes."""

        if size < 0:
            raise ValueError("retained container byte size cannot be negative")
        if self.estimated_bytes + size > self.maximum_bytes:
            counter._exhaust("maximum_cache_bytes")
        self.estimated_bytes += size
        counter.retained_cache_bytes = max(
            counter.retained_cache_bytes,
            self.estimated_bytes,
        )

    @property
    def retained_entries(self) -> int:
        return len(self.domains) + self.tree_nodes


@dataclass(frozen=True, slots=True)
class _PolicyTreeNode:
    information_key: tuple[str, str] | None
    children: tuple[tuple[str, "_PolicyTreeNode"], ...]
    value: float | None


@dataclass(frozen=True, slots=True)
class _JointRootEvaluation:
    value: float
    world_values: tuple[float, ...]
    worst_path: tuple[str, ...]
    worst_world_hash: str
    target_policy_hash: str
    opponent_policy_hash: str
    target_policy_count: int
    opponent_policy_count: int
    forecast_state_space: int
    catalog_entries: int
    catalog_bytes: int


def _team(value: str) -> Team:
    return Team.NORTH_SOUTH if value == "north_south" else Team.WEST_EAST


def _terminal_utility(state: RoundState, request: ScenarioProofRequestV1) -> float:
    target = _team(request.target_team)
    score = state.round_score
    if score is None:
        raise ValueError("terminal proof state lacks a round score")
    if request.utility == "awarded_point_margin":
        return float(score.awarded_points[int(target)] - score.awarded_points[int(target.other)])
    if request.utility == "contract_success":
        if state.declarer is None:
            raise ValueError("contract utility requires a declarer")
        succeeded = score.winner == state.declarer.team
        target_succeeded = succeeded if target == state.declarer.team else not succeeded
        return 1.0 if target_succeeded else (-1.0 if request.utility_encoding == "signed_win_loss_v1" else 0.0)
    if request.utility == "round_win":
        return 1.0 if score.winner == target else (-1.0 if score.winner == target.other else 0.0)
    if request.utility == "match_win":
        return 1.0 if state.match_winner == target else (-1.0 if state.match_winner == target.other else 0.0)
    if request.utility == "kabout":
        return 1.0 if score.kabout and score.winner == target else -1.0
    raise ValueError("unsupported scenario proof utility")


def _boundary_utility(state: RoundState, request: ScenarioProofRequestV1) -> float:
    if request.boundary_evaluator != "current_raw_point_margin_v1":
        raise ValueError("unsupported bounded proof boundary evaluator")
    target = _team(request.target_team)
    points = [0, 0]
    for trick in state.completed_tricks:
        points[int(trick.winner.team)] += int(trick.raw_points)
    for project in state.projects:
        if project.scoring:
            points[int(project.owner.team)] += int(project.value)
    return float(points[int(target)] - points[int(target.other)])


class UnifiedScenarioProofServiceV1:
    """Evaluate typed proof requests without changing the authoritative state."""

    engine_id = "unified_scenario_proof_v1"
    engine_version = "1.0.0"

    @staticmethod
    def _strategy_class(request: ScenarioProofRequestV1) -> str:
        return request.strategy_class or (
            "frozen_policy"
            if request.mode == "policy_conditioned"
            else "pure_deterministic"
        )

    @staticmethod
    def _complete_state_coverage(
        normalized: NormalizedExactStateV1,
    ) -> dict[str, Any]:
        core = {
            "method": "complete_deal_exact_input",
            "complete": True,
            "expectedWorldCount": 1,
            "worldHashes": [normalized.state_hash],
        }
        return {**core, "sha256": canonical_sha256(core)}

    def __init__(self, budgets: ScenarioProofBudgetRegistryV1):
        self.budgets = budgets

    @classmethod
    def from_repository(cls, root: Path) -> "UnifiedScenarioProofServiceV1":
        return cls(
            ScenarioProofBudgetRegistryV1.load(
                root / "configs" / "research" / "proof_budgets_v1.yaml"
            )
        )

    @classmethod
    def from_default_authority(cls) -> "UnifiedScenarioProofServiceV1":
        return cls(ScenarioProofBudgetRegistryV1.load_default())

    @staticmethod
    def _counter(
        budget: ScenarioProofBudgetV1,
        *,
        maximum_nodes: int,
        started_wall: float,
        started_cpu: float,
        cancel_check: Any | None,
    ) -> _Counter:
        return _Counter(
            maximum_nodes,
            started_wall + budget.maximum_wall_seconds,
            started_cpu + budget.maximum_cpu_seconds,
            budget.cancellation_check_interval_nodes,
            cancel_check,
        )

    @staticmethod
    def _depth(request: ScenarioProofRequestV1, state: RoundState, budget: ScenarioProofBudgetV1) -> int:
        remaining = sum(mask.bit_count() for mask in state.hands)
        if request.mode == "exact":
            if remaining > budget.maximum_depth:
                raise ProofBudgetExceeded("maximum_depth")
            return remaining
        if request.mode == "bounded":
            assert request.horizon is not None
            unit = str(request.horizon["unit"])
            value = int(request.horizon["value"])
            requested = value * 4 if unit == "tricks" else value
            if unit == "round_end":
                requested = remaining
            if requested > budget.maximum_horizon_actions:
                raise ProofBudgetExceeded("maximum_horizon_actions")
            if requested > budget.maximum_depth:
                raise ProofBudgetExceeded("maximum_depth")
            return min(requested, remaining)
        return remaining

    @staticmethod
    def _state_space_forecast(
        state: RoundState,
        *,
        depth: int,
        world_count: int,
        exact: bool,
    ) -> int:
        counts = [mask.bit_count() for mask in state.hands]
        if exact and depth >= sum(counts):
            leaves = math.prod(math.factorial(count) for count in counts)
        else:
            leaves = max(counts, default=1) ** depth
        if state.trump_suit is not None:
            # A trick leader can make one additional binary declaration choice
            # for the selected lead card. Card-order branching above already
            # accounts for which card is led.
            leaves *= 2 ** math.ceil(depth / len(Seat))
        return max(1, world_count) * (1 + depth * leaves)

    @staticmethod
    def _preflight_policy_budget(
        budget: ScenarioProofBudgetV1,
        counter: _Counter,
        *,
        horizon_actions: int,
        root_action_count: int,
        world_count: int = 1,
    ) -> int:
        if horizon_actions > budget.maximum_horizon_actions:
            counter._exhaust("maximum_horizon_actions")
        if horizon_actions > budget.maximum_depth:
            counter._exhaust("maximum_depth")
        forecast = max(1, world_count) * max(1, root_action_count) * (
            1 + horizon_actions
        )
        counter.forecast_state_space = forecast
        if forecast > budget.maximum_state_space:
            counter._exhaust("maximum_state_space")
        return forecast

    def _evaluate(
        self,
        state: RoundState,
        *,
        engine: BalootEngine,
        request: ScenarioProofRequestV1,
        depth: int,
        counter: _Counter,
        cache: _ProofCache,
    ) -> _Evaluation:
        counter.visit()
        state_key = canonical_sha256(state.to_dict(reveal_hands=True))
        key = ("complete", state_key, depth)
        cached = cache.get(key)
        if cached is not None:
            return cached
        terminal = state.current_player is None
        if terminal or depth == 0:
            value = (
                _terminal_utility(state, request)
                if terminal
                else _boundary_utility(state, request)
            )
            winning = int(value >= request.success_threshold)
            result = _Evaluation(value, (), 1, winning, winning)
            cache.put(key, result)
            return result
        target = _team(request.target_team)
        maximizing = state.current_player.team == target
        children: list[tuple[_ProofPlayAction, _Evaluation]] = []
        for action in _legal_proof_play_actions(engine, state):
            token = engine.apply(
                state,
                action.card,
                ekka=action.ekka,
                verify_identity=False,
                emit_events=False,
            )
            try:
                child = self._evaluate(
                    state,
                    engine=engine,
                    request=request,
                    depth=depth - 1,
                    counter=counter,
                    cache=cache,
                )
            finally:
                engine.undo(state, token)
            children.append((action, child))
        chooser = max if maximizing else min
        chosen_action, chosen = chooser(
            children,
            key=lambda item: (
                item[1].value,
                -item[0].action_id if maximizing else item[0].action_id,
            ),
        )
        optimal_count = sum(
            item.optimal_line_count
            for _, item in children
            if item.value == chosen.value
        )
        result = _Evaluation(
            chosen.value,
            (chosen_action.code, *chosen.principal_variation),
            sum(item.all_line_count for _, item in children),
            sum(item.winning_line_count for _, item in children),
            optimal_count,
        )
        cache.put(key, result)
        return result

    @staticmethod
    def _information_key(state: RoundState) -> tuple[str, str]:
        actor = state.current_player
        if actor is None:
            raise ValueError("terminal state has no information-set actor")
        return (
            actor.name.lower(),
            canonical_sha256(
                InformationState.from_complete(state, actor).to_dict()
            ),
        )

    def _build_information_tree(
        self,
        state: RoundState,
        *,
        engine: BalootEngine,
        request: ScenarioProofRequestV1,
        depth: int,
        counter: _Counter,
        catalog: _InformationSetCatalog,
    ) -> _PolicyTreeNode:
        catalog.visit_state(counter)
        counter.visit()
        if state.current_player is None or depth == 0:
            value = (
                _terminal_utility(state, request)
                if state.current_player is None
                else _boundary_utility(state, request)
            )
            node = _PolicyTreeNode(None, (), value)
            catalog.retain_tree_node(
                sys.getsizeof(node)
                + sys.getsizeof(node.children)
                + sys.getsizeof(value),
                counter,
            )
            return node
        legal = tuple(
            _legal_proof_play_actions(engine, state)
        )
        information_key = self._information_key(state)
        catalog.add(
            information_key,
            tuple(action.code for action in legal),
            counter,
        )
        children: list[tuple[str, _PolicyTreeNode]] = []
        for action in legal:
            token = engine.apply(
                state,
                action.card,
                ekka=action.ekka,
                verify_identity=False,
                emit_events=False,
            )
            try:
                children.append(
                    (
                        action.code,
                        self._build_information_tree(
                            state,
                            engine=engine,
                            request=request,
                            depth=depth - 1,
                            counter=counter,
                            catalog=catalog,
                        ),
                    )
                )
            finally:
                engine.undo(state, token)
        node = _PolicyTreeNode(information_key, tuple(children), None)
        size = sys.getsizeof(node) + sys.getsizeof(node.children)
        size += sys.getsizeof(information_key)
        size += sum(sys.getsizeof(item) for item in information_key)
        size += sum(
            sys.getsizeof(pair) + sys.getsizeof(pair[0])
            for pair in node.children
        )
        catalog.retain_tree_node(size, counter)
        return node

    @staticmethod
    def _policy_hash(
        keys: Sequence[tuple[str, str]],
        choices: Sequence[str],
    ) -> str:
        return canonical_sha256(
            [
                {
                    "seat": key[0],
                    "informationStateHash": key[1],
                    "action": action,
                }
                for key, action in zip(keys, choices, strict=True)
            ]
        )

    def _play_information_strategy(
        self,
        tree: _PolicyTreeNode,
        *,
        root_action: str,
        target: Team,
        target_policy: Mapping[tuple[str, str], str],
        opponent_policy: Mapping[tuple[str, str], str],
        counter: _Counter,
    ) -> tuple[float, tuple[str, ...]]:
        node = tree
        path = [root_action]
        while node.information_key is not None:
            counter.visit()
            key = node.information_key
            policy = (
                target_policy
                if Seat.parse(key[0]).team == target
                else opponent_policy
            )
            action_code = policy.get(key)
            if action_code is None:
                counter._exhaust("actor_information_set_policy_is_incomplete")
            child = next(
                (
                    candidate
                    for action, candidate in node.children
                    if action == action_code
                ),
                None,
            )
            if child is None:
                counter._exhaust("actor_information_set_policy_is_illegal")
            node = child
            path.append(action_code)
        counter.visit()
        assert node.value is not None
        return node.value, tuple(path)

    def _evaluate_information_set_root(
        self,
        worlds: Sequence[NormalizedExactStateV1],
        *,
        root_action: _ProofPlayAction,
        request: ScenarioProofRequestV1,
        depth: int,
        budget: ScenarioProofBudgetV1,
        counter: _Counter,
        remaining_state_space: int,
    ) -> _JointRootEvaluation:
        """Solve one root as a robust pure-strategy imperfect-information game."""

        worlds = tuple(sorted(worlds, key=lambda world: world.state_hash))
        catalog = _InformationSetCatalog(
            budget.maximum_cache_entries,
            budget.maximum_cache_bytes,
            max(0, remaining_state_space),
            budget.maximum_state_space - remaining_state_space,
        )
        trees: list[tuple[NormalizedExactStateV1, _PolicyTreeNode]] = []
        catalog.retain_container_bytes(sys.getsizeof(trees), counter)
        for world in worlds:
            state = world.round_state.clone()
            engine = BalootEngine(world.rules)
            engine.play_card(
                state,
                root_action.card,
                ekka=root_action.ekka,
                emit_events=False,
            )
            tree = self._build_information_tree(
                state,
                engine=engine,
                request=request,
                depth=max(depth - 1, 0),
                counter=counter,
                catalog=catalog,
            )
            pair = (world, tree)
            previous_list_bytes = sys.getsizeof(trees)
            trees.append(pair)
            catalog.retain_container_bytes(
                sys.getsizeof(pair)
                + sys.getsizeof(trees)
                - previous_list_bytes,
                counter,
            )

        target = _team(request.target_team)
        ordered_keys = tuple(sorted(catalog.domains))
        target_keys = tuple(
            key
            for key in ordered_keys
            if Seat.parse(key[0]).team == target
        )
        opponent_keys = tuple(
            key
            for key in ordered_keys
            if Seat.parse(key[0]).team != target
        )
        target_domains = tuple(catalog.domains[key] for key in target_keys)
        opponent_domains = tuple(catalog.domains[key] for key in opponent_keys)
        target_policy_count = math.prod(len(domain) for domain in target_domains)
        opponent_policy_count = math.prod(
            len(domain) for domain in opponent_domains
        )
        forecast = catalog.built_states + (
            target_policy_count
            * opponent_policy_count
            * len(worlds)
            * max(1, depth)
        )
        counter.forecast_state_space = (
            budget.maximum_state_space - remaining_state_space + forecast
        )
        if forecast > remaining_state_space:
            counter._exhaust("maximum_state_space")

        best_value = -math.inf
        best_world_values: tuple[float, ...] = ()
        best_path: tuple[str, ...] = ()
        best_world_hash = ""
        best_target_hash = ""
        best_opponent_hash = ""
        target_products = itertools.product(*target_domains)
        for target_choices in target_products:
            target_policy = dict(zip(target_keys, target_choices, strict=True))
            world_minima = [math.inf] * len(worlds)
            worst: tuple[float, str, tuple[str, ...], str] | None = None
            for opponent_choices in itertools.product(*opponent_domains):
                opponent_policy = dict(
                    zip(opponent_keys, opponent_choices, strict=True)
                )
                opponent_hash = self._policy_hash(
                    opponent_keys,
                    opponent_choices,
                )
                for index, (world, tree) in enumerate(trees):
                    value, path = self._play_information_strategy(
                        tree,
                        root_action=root_action.code,
                        target=target,
                        target_policy=target_policy,
                        opponent_policy=opponent_policy,
                        counter=counter,
                    )
                    world_minima[index] = min(world_minima[index], value)
                    candidate = (value, world.state_hash, path, opponent_hash)
                    if worst is None or candidate < worst:
                        worst = candidate
            assert worst is not None
            target_value = worst[0]
            if target_value > best_value:
                best_value = target_value
                best_world_values = tuple(world_minima)
                best_path = worst[2]
                best_world_hash = worst[1]
                best_target_hash = self._policy_hash(
                    target_keys,
                    target_choices,
                )
                best_opponent_hash = worst[3]
        if not math.isfinite(best_value):
            counter._exhaust("actor_information_set_policy_space_is_empty")
        return _JointRootEvaluation(
            best_value,
            best_world_values,
            best_path,
            best_world_hash,
            best_target_hash,
            best_opponent_hash,
            target_policy_count,
            opponent_policy_count,
            forecast,
            catalog.retained_entries,
            catalog.estimated_bytes,
        )

    @staticmethod
    def _validate_policy_request(request: ScenarioProofRequestV1) -> dict[str, Mapping[str, Any]]:
        if request.policies is None:
            raise ValueError("policy-conditioned proof requires frozen policies")
        result: dict[str, Mapping[str, Any]] = {}
        for seat, value in request.policies.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"policy {seat} must be an object")
            raw = dict(value)
            if set(raw) != {"agentId", "version", "profileHash", "seed"}:
                raise ValueError(f"policy {seat} shape is invalid")
            if raw["agentId"] != "heuristic_bot_v1" or raw["version"] != "1.0.0":
                raise ValueError(f"policy {seat} is unavailable")
            if type(raw["seed"]) is not int:
                raise TypeError(f"policy {seat} seed must be an integer")
            expected_hash = canonical_sha256(
                {"agentId": raw["agentId"], "version": raw["version"], "seed": raw["seed"]}
            )
            if raw["profileHash"] != expected_hash:
                raise ValueError(f"policy {seat} profile hash mismatch")
            result[str(seat)] = copy.deepcopy(raw)
        return result

    @staticmethod
    def _claim_result(
        request: ScenarioProofRequestV1,
        root_rows: Sequence[Mapping[str, Any]],
        selected_code: str | None,
    ) -> bool:
        winning_count = sum(bool(item["satisfies"]) for item in root_rows)
        selected = next(
            (
                item
                for item in root_rows
                if str(item.get("action")) == str(selected_code)
            ),
            None,
        )
        if request.claim == "outcome_reachable":
            return winning_count >= 1
        if request.claim == "outcome_forced":
            return bool(root_rows) and winning_count == len(root_rows)
        if request.claim == "selected_action_is_unique_winner":
            return (
                winning_count == 1
                and selected is not None
                and bool(selected["satisfies"])
            )
        if request.claim == "target_team_has_no_salvage":
            return winning_count == 0
        return True

    @staticmethod
    def _bidding_machine(normalized: NormalizedExactStateV1) -> BiddingMachine:
        state = normalized.round_state
        if state.purchase_card is None:
            raise ValueError("bid proof exact state lacks its purchase card")
        machine = BiddingMachine(
            state.dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(normalized.rules.data),
            state.purchase_card.rank,
        )
        bidding = normalized.state.get("bidding")
        if not isinstance(bidding, Mapping):
            raise ValueError("bid proof exact state lacks its auction projection")
        history = bidding.get("history")
        if not isinstance(history, list):
            raise ValueError("bid proof auction history is invalid")
        for index, row in enumerate(history):
            if not isinstance(row, Mapping):
                raise ValueError("bid proof auction history row is invalid")
            if machine.actor is None or str(row.get("seat")) != machine.actor.name.lower():
                raise ValueError(f"bid proof auction history actor mismatch at {index}")
            trump = row.get("trumpSuit")
            machine.submit(
                str(row.get("action") or ""),
                trump=Suit.parse(str(trump)) if trump is not None else None,
            )
        if machine.finished or machine.actor is None:
            raise ValueError("bid proof requires an unfinished auction decision")
        return machine

    def _finalize_policy_decision(
        self,
        *,
        normalized: NormalizedExactStateV1,
        request: ScenarioProofRequestV1,
        bundle_hash: str,
        selected_action: str | None,
        cancel_check: Any | None,
        actor_team: Team,
        policies: Mapping[str, Mapping[str, Any]],
        budget: ScenarioProofBudgetV1,
        counter: _Counter,
        started: float,
        started_cpu: float,
        root_rows: list[dict[str, Any]],
        witnesses: Mapping[str, Sequence[str]],
        depth: int,
        forecast_state_space: int | None,
        expected_root_action_count: int,
    ) -> ProofOutcomeV1:
        counter.finish()
        cancelled = counter.cancelled
        requested_selected = selected_action or request.selected_action
        selected = next(
            (
                item
                for item in root_rows
                if str(item["action"]) == str(requested_selected)
            ),
            None,
        )
        if selected is None and root_rows and requested_selected is None:
            selected = (max if actor_team == _team(request.target_team) else min)(
                root_rows,
                key=lambda item: (float(item["value"]), str(item["action"])),
            )
        selected_code = (
            str(selected["action"])
            if selected is not None
            else str(requested_selected)
            if requested_selected is not None
            else None
        )
        complete = (
            bool(root_rows)
            and not cancelled
            and counter.exhaustion_reason is None
        )
        claim_satisfied = complete and self._claim_result(
            request,
            root_rows,
            selected_code,
        )
        completion = (
            "policy_complete"
            if claim_satisfied
            else "proof_failed"
            if complete
            else "proof_incomplete"
        )
        identities = {
            "rulesHash": normalized.rules_hash,
            "bundleHash": str(bundle_hash),
            "stateHash": normalized.state_hash,
            "exactStateId": normalized.exact_state_id,
            "proofProfileHash": budget.sha256,
            "proofBudgetRegistryHash": self.budgets.sha256,
            "proofRequestHash": request.sha256,
            "strategyClass": self._strategy_class(request),
            "engine": {"id": self.engine_id, "version": self.engine_version},
            "policyHashes": {
                seat: str(value["profileHash"])
                for seat, value in policies.items()
            },
        }
        return ProofOutcomeV1.build(
            {
                "schemaVersion": 1,
                "requestedMode": request.mode,
                "achievedMode": "policy_conditioned" if complete else "none",
                "requestedVisibility": request.visibility,
                "achievedVisibility": "complete_state" if complete else "none",
                "strategyClass": self._strategy_class(request),
                "claim": request.claim,
                "targetTeam": request.target_team,
                "utility": request.utility,
                "successThreshold": request.success_threshold,
                "horizon": copy.deepcopy(request.horizon),
                "boundaryEvaluator": request.boundary_evaluator,
                "rootActions": root_rows,
                "selectedAction": selected_code,
                "winningRootActionCount": sum(
                    bool(item["satisfies"]) for item in root_rows
                ),
                "continuationLines": (
                    {
                        "complete": True,
                        "exactCount": (
                            "1" if selected and selected["satisfies"] else "0"
                        ),
                        "lineUniverse": "policy_conditioned",
                    }
                    if complete
                    else {
                        "complete": False,
                        "lowerBound": "0",
                        "upperBound": None,
                        "lineUniverse": "policy_conditioned",
                    }
                ),
                "hiddenWorlds": {
                    "count": 1,
                    "complete": True,
                    "hashes": [normalized.state_hash],
                    "coverage": self._complete_state_coverage(normalized),
                },
                "resources": {
                    "nodes": counter.nodes,
                    "depth": depth,
                    "cacheEntries": 0,
                    "cacheBytes": 0,
                    "elapsedWallSeconds": time.monotonic() - started,
                    "elapsedCpuSeconds": time.thread_time() - started_cpu,
                    "nodeLimit": budget.maximum_nodes,
                    "depthLimit": budget.maximum_depth,
                    "wallTimeLimitSeconds": budget.maximum_wall_seconds,
                    "cacheEntryLimit": budget.maximum_cache_entries,
                    "cacheByteLimit": budget.maximum_cache_bytes,
                    "hiddenWorldLimit": budget.maximum_hidden_worlds,
                    "horizonActionLimit": budget.maximum_horizon_actions,
                    "cpuTimeLimitSeconds": budget.maximum_cpu_seconds,
                    "cancellationCheckIntervalNodes": budget.cancellation_check_interval_nodes,
                    "workerLimit": budget.maximum_workers,
                    "workers": 1,
                    "strategyClass": self._strategy_class(request),
                    "lineCountNodeLimit": budget.maximum_line_count_nodes,
                    "evaluatedRootActionCount": len(root_rows),
                    "expectedRootActionCount": expected_root_action_count,
                    "rootActionCoverageComplete": (
                        len(root_rows) == expected_root_action_count
                    ),
                    "winningRootActionLowerBound": sum(
                        bool(item["satisfies"]) for item in root_rows
                    ),
                    "winningRootActionUpperBound": (
                        sum(bool(item["satisfies"]) for item in root_rows)
                        + max(0, expected_root_action_count - len(root_rows))
                    ),
                    "forecastStateSpace": (
                        counter.forecast_state_space
                        if counter.forecast_state_space is not None
                        else forecast_state_space
                    ),
                    "stateSpaceLimit": budget.maximum_state_space,
                    "exhaustionReason": counter.exhaustion_reason,
                    "cancelled": cancelled,
                },
                "completionStatus": completion,
                "claimSatisfied": claim_satisfied,
                "principalVariation": list(witnesses.get(selected_code or "", ())),
                "identities": identities,
            }
        )

    def _prove_policy_bid(
        self,
        normalized: NormalizedExactStateV1,
        request: ScenarioProofRequestV1,
        *,
        bundle_hash: str,
        selected_action: str | None,
        cancel_check: Any | None,
    ) -> ProofOutcomeV1:
        if request.mode != "policy_conditioned":
            raise ValueError("bid proof currently requires policy_conditioned mode")
        if request.visibility != "complete_state":
            raise ValueError("bid proof currently requires complete_state visibility")
        policies = self._validate_policy_request(request)
        if set(policies) != {"north", "west", "south", "east"}:
            raise ValueError("bid proof requires one frozen policy for every seat")
        budget = self.budgets.get(request.budget_profile)
        started = time.monotonic()
        started_cpu = time.thread_time()
        counter = self._counter(
            budget,
            maximum_nodes=budget.maximum_nodes,
            started_wall=started,
            started_cpu=started_cpu,
            cancel_check=cancel_check,
        )
        machine = self._bidding_machine(normalized)
        actor = machine.actor
        assert actor is not None
        target = _team(request.target_team)
        root_rows: list[dict[str, Any]] = []
        witnesses: dict[str, list[str]] = {}
        forecast_state_space: int | None = None
        evaluated_depth = 0
        expected_root_action_count = 0
        try:
            choices = legal_bid_choices(machine)
            expected_root_action_count = len(choices)
            requested_selected = selected_action or request.selected_action
            if requested_selected is not None and str(requested_selected) not in {
                choice.id for choice in choices
            }:
                raise ValueError("proof selectedAction is not a legal root action")
            maximum_bid_horizon = len(FULL_DECK) + (2 * len(Seat))
            forecast_state_space = self._preflight_policy_budget(
                budget,
                counter,
                horizon_actions=maximum_bid_horizon,
                root_action_count=len(choices),
            )
            for choice in choices:
                counter.visit()
                branch = normalized.round_state.clone()
                branch_machine = copy.deepcopy(machine)
                result = branch_machine.submit(
                    choice.action,
                    trump=choice.trump_suit,
                )
                witness = [f"bid:{actor.name.lower()}:{choice.id}"]
                five_cards = {
                    seat: mask_to_cards(branch.initial_five[int(seat)])
                    for seat in Seat
                }
                while result is None:
                    counter.visit()
                    next_actor = branch_machine.actor
                    assert next_actor is not None
                    action, trump = _choose_action(
                        branch_machine,
                        five_cards,
                        branch.purchase_card,
                        normalized.rules,
                    )
                    witness.append(
                        f"bid:{next_actor.name.lower()}:{action}"
                        + (f":{trump.name.lower()}" if trump is not None else "")
                    )
                    result = branch_machine.submit(action, trump=trump)
                value = 0.0
                if not result.all_passed:
                    engine = BalootEngine(normalized.rules)
                    engine.complete_deal_from_bid_result(branch, result)
                    engine.prepare_projects(branch)
                    bots = {
                        Seat.parse(seat): HeuristicBot(
                            engine,
                            BotProfile(int(policy["seed"])),
                        )
                        for seat, policy in policies.items()
                    }
                    while branch.current_player is not None:
                        counter.visit()
                        seat = branch.current_player
                        card = bots[seat].choose(branch)
                        witness.append(f"play:{seat.name.lower()}:{card.code}")
                        engine.play_card(branch, card, emit_events=False)
                    value = _terminal_utility(branch, request)
                evaluated_depth = max(evaluated_depth, len(witness))
                root_rows.append(
                    {
                        "action": choice.id,
                        "value": value,
                        "satisfies": value >= request.success_threshold,
                        "worldValues": [value],
                        "completionStatus": "policy_complete",
                    }
                )
                witnesses[choice.id] = witness
        except ProofBudgetExceeded:
            pass
        counter.finish()
        cancelled = counter.cancelled
        requested_selected = selected_action or request.selected_action
        selected = next(
            (
                item
                for item in root_rows
                if str(item["action"]) == str(requested_selected)
            ),
            None,
        )
        if selected is None and root_rows and requested_selected is None:
            selected = (max if actor.team == target else min)(
                root_rows,
                key=lambda item: (float(item["value"]), str(item["action"])),
            )
        selected_code = (
            str(selected["action"])
            if selected is not None
            else str(requested_selected)
            if requested_selected is not None
            else None
        )
        complete = (
            bool(root_rows)
            and not cancelled
            and counter.exhaustion_reason is None
        )
        claim_satisfied = complete and self._claim_result(
            request,
            root_rows,
            selected_code,
        )
        completion = (
            "policy_complete"
            if claim_satisfied
            else "proof_failed"
            if complete
            else "proof_incomplete"
        )
        identities = {
            "rulesHash": normalized.rules_hash,
            "bundleHash": str(bundle_hash),
            "stateHash": normalized.state_hash,
            "exactStateId": normalized.exact_state_id,
            "proofProfileHash": budget.sha256,
            "proofBudgetRegistryHash": self.budgets.sha256,
            "proofRequestHash": request.sha256,
            "strategyClass": self._strategy_class(request),
            "engine": {"id": self.engine_id, "version": self.engine_version},
            "policyHashes": {
                seat: str(value["profileHash"])
                for seat, value in policies.items()
            },
        }
        payload = {
            "schemaVersion": 1,
            "requestedMode": request.mode,
            "achievedMode": "policy_conditioned" if complete else "none",
            "requestedVisibility": request.visibility,
            "achievedVisibility": "complete_state" if complete else "none",
            "strategyClass": self._strategy_class(request),
            "claim": request.claim,
            "targetTeam": request.target_team,
            "utility": request.utility,
            "successThreshold": request.success_threshold,
            "horizon": copy.deepcopy(request.horizon),
            "boundaryEvaluator": request.boundary_evaluator,
            "rootActions": root_rows,
            "selectedAction": selected_code,
            "winningRootActionCount": sum(
                bool(item["satisfies"]) for item in root_rows
            ),
            "continuationLines": (
                {
                    "complete": True,
                    "exactCount": "1" if selected and selected["satisfies"] else "0",
                    "lineUniverse": "policy_conditioned",
                }
                if complete
                else {
                    "complete": False,
                    "lowerBound": "0",
                    "upperBound": None,
                    "lineUniverse": "policy_conditioned",
                }
            ),
            "hiddenWorlds": {
                "count": 1,
                "complete": True,
                "hashes": [normalized.state_hash],
                "coverage": self._complete_state_coverage(normalized),
            },
            "resources": {
                "nodes": counter.nodes,
                "depth": evaluated_depth,
                "cacheEntries": 0,
                "cacheBytes": 0,
                "elapsedWallSeconds": time.monotonic() - started,
                "elapsedCpuSeconds": time.thread_time() - started_cpu,
                "nodeLimit": budget.maximum_nodes,
                "depthLimit": budget.maximum_depth,
                "wallTimeLimitSeconds": budget.maximum_wall_seconds,
                "cacheEntryLimit": budget.maximum_cache_entries,
                "cacheByteLimit": budget.maximum_cache_bytes,
                "hiddenWorldLimit": budget.maximum_hidden_worlds,
                "horizonActionLimit": budget.maximum_horizon_actions,
                "cpuTimeLimitSeconds": budget.maximum_cpu_seconds,
                "cancellationCheckIntervalNodes": budget.cancellation_check_interval_nodes,
                "workerLimit": budget.maximum_workers,
                "workers": 1,
                "strategyClass": self._strategy_class(request),
                "lineCountNodeLimit": budget.maximum_line_count_nodes,
                "evaluatedRootActionCount": len(root_rows),
                "expectedRootActionCount": expected_root_action_count,
                "rootActionCoverageComplete": (
                    len(root_rows) == expected_root_action_count
                ),
                "winningRootActionLowerBound": sum(
                    bool(item["satisfies"]) for item in root_rows
                ),
                "winningRootActionUpperBound": (
                    sum(bool(item["satisfies"]) for item in root_rows)
                    + max(0, expected_root_action_count - len(root_rows))
                ),
                "forecastStateSpace": (
                    counter.forecast_state_space
                    if counter.forecast_state_space is not None
                    else forecast_state_space
                ),
                "stateSpaceLimit": budget.maximum_state_space,
                "exhaustionReason": counter.exhaustion_reason,
                "cancelled": cancelled,
            },
            "completionStatus": completion,
            "claimSatisfied": claim_satisfied,
            "principalVariation": witnesses.get(selected_code or "", []),
            "identities": identities,
        }
        return ProofOutcomeV1.build(payload)

    def _prove_policy_escalation(
        self,
        normalized: NormalizedExactStateV1,
        request: ScenarioProofRequestV1,
        *,
        bundle_hash: str,
        selected_action: str | None,
        cancel_check: Any | None,
    ) -> ProofOutcomeV1:
        if request.mode != "policy_conditioned":
            raise ValueError(
                "escalation proof currently requires policy_conditioned mode"
            )
        if request.visibility != "complete_state":
            raise ValueError(
                "escalation proof currently requires complete_state visibility"
            )
        policies = self._validate_policy_request(request)
        if set(policies) != {"north", "west", "south", "east"}:
            raise ValueError(
                "escalation proof requires one frozen policy for every seat"
            )
        if normalized.next_actor is None:
            raise ValueError("escalation proof exact state lacks its acting seat")
        actor = Seat.parse(normalized.next_actor)
        budget = self.budgets.get(request.budget_profile)
        started = time.monotonic()
        started_cpu = time.thread_time()
        counter = self._counter(
            budget,
            maximum_nodes=budget.maximum_nodes,
            started_wall=started,
            started_cpu=started_cpu,
            cancel_check=cancel_check,
        )
        root_rows: list[dict[str, Any]] = []
        witnesses: dict[str, list[str]] = {}
        forecast_state_space: int | None = None
        expected_root_action_count = len(normalized.legal_actions)
        requested_selected = selected_action or request.selected_action
        legal_action_codes = {
            str(raw.get("code") or "") for raw in normalized.legal_actions
        }
        if requested_selected is not None and str(requested_selected) not in legal_action_codes:
            raise ValueError("proof selectedAction is not a legal root action")
        try:
            horizon_actions = 1 + sum(
                mask.bit_count() for mask in normalized.round_state.hands
            )
            forecast_state_space = self._preflight_policy_budget(
                budget,
                counter,
                horizon_actions=horizon_actions,
                root_action_count=len(normalized.legal_actions),
            )
            for raw in normalized.legal_actions:
                counter.visit()
                action = str(raw.get("action") or "")
                code = str(raw.get("code") or "")
                if not action or not code:
                    raise ValueError("escalation proof legal action is invalid")
                branch = normalized.round_state.clone()
                engine = BalootEngine(normalized.rules)
                witness = [f"escalation:{actor.name.lower()}:{code}"]
                if action != "decline":
                    engine.apply_escalation(
                        branch,
                        seat=actor,
                        action=action,
                        mode=(
                            str(raw["mode"])
                            if raw.get("mode") is not None
                            else None
                        ),
                    )
                engine.prepare_projects(branch)
                bots = {
                    Seat.parse(seat): HeuristicBot(
                        engine,
                        BotProfile(int(policy["seed"])),
                    )
                    for seat, policy in policies.items()
                }
                while branch.current_player is not None:
                    counter.visit()
                    seat = branch.current_player
                    card = bots[seat].choose(branch)
                    witness.append(f"play:{seat.name.lower()}:{card.code}")
                    engine.play_card(branch, card, emit_events=False)
                value = _terminal_utility(branch, request)
                root_rows.append(
                    {
                        "action": code,
                        "value": value,
                        "satisfies": value >= request.success_threshold,
                        "worldValues": [value],
                        "completionStatus": "policy_complete",
                    }
                )
                witnesses[code] = witness
        except ProofBudgetExceeded:
            pass
        return self._finalize_policy_decision(
            normalized=normalized,
            request=request,
            bundle_hash=bundle_hash,
            selected_action=selected_action,
            cancel_check=cancel_check,
            actor_team=actor.team,
            policies=policies,
            budget=budget,
            counter=counter,
            started=started,
            started_cpu=started_cpu,
            root_rows=root_rows,
            witnesses=witnesses,
            depth=horizon_actions,
            forecast_state_space=forecast_state_space,
            expected_root_action_count=expected_root_action_count,
        )

    def _prove_policy_project(
        self,
        normalized: NormalizedExactStateV1,
        request: ScenarioProofRequestV1,
        *,
        bundle_hash: str,
        selected_action: str | None,
        cancel_check: Any | None,
    ) -> ProofOutcomeV1:
        if request.mode != "policy_conditioned":
            raise ValueError(
                "project proof currently requires policy_conditioned mode"
            )
        if request.visibility != "complete_state":
            raise ValueError(
                "project proof currently requires complete_state visibility"
            )
        policies = self._validate_policy_request(request)
        if set(policies) != {"north", "west", "south", "east"}:
            raise ValueError("project proof requires one frozen policy for every seat")
        if normalized.next_actor is None or len(normalized.legal_actions) != 1:
            raise ValueError(
                "project proof requires one authoritative declaration action"
            )
        actor = Seat.parse(normalized.next_actor)
        action = str(normalized.legal_actions[0].get("code") or "")
        if not action.startswith("declare:"):
            raise ValueError("project proof declaration action is invalid")
        requested_selected = selected_action or request.selected_action
        if requested_selected is not None and str(requested_selected) != action:
            raise ValueError("proof selectedAction is not a legal root action")
        budget = self.budgets.get(request.budget_profile)
        started = time.monotonic()
        started_cpu = time.thread_time()
        counter = self._counter(
            budget,
            maximum_nodes=budget.maximum_nodes,
            started_wall=started,
            started_cpu=started_cpu,
            cancel_check=cancel_check,
        )
        root_rows: list[dict[str, Any]] = []
        witnesses: dict[str, list[str]] = {}
        forecast_state_space: int | None = None
        try:
            horizon_actions = 1 + sum(
                mask.bit_count() for mask in normalized.round_state.hands
            )
            forecast_state_space = self._preflight_policy_budget(
                budget,
                counter,
                horizon_actions=horizon_actions,
                root_action_count=1,
            )
            branch = normalized.round_state.clone()
            engine = BalootEngine(normalized.rules)
            bots = {
                Seat.parse(seat): HeuristicBot(
                    engine,
                    BotProfile(int(policy["seed"])),
                )
                for seat, policy in policies.items()
            }
            witness = [f"project:{actor.name.lower()}:{action}"]
            while branch.current_player is not None:
                counter.visit()
                seat = branch.current_player
                card = bots[seat].choose(branch)
                witness.append(f"play:{seat.name.lower()}:{card.code}")
                engine.play_card(branch, card, emit_events=False)
            value = _terminal_utility(branch, request)
            root_rows.append(
                {
                    "action": action,
                    "value": value,
                    "satisfies": value >= request.success_threshold,
                    "worldValues": [value],
                    "completionStatus": "policy_complete",
                }
            )
            witnesses[action] = witness
        except ProofBudgetExceeded:
            pass
        return self._finalize_policy_decision(
            normalized=normalized,
            request=request,
            bundle_hash=bundle_hash,
            selected_action=selected_action,
            cancel_check=cancel_check,
            actor_team=actor.team,
            policies=policies,
            budget=budget,
            counter=counter,
            started=started,
            started_cpu=started_cpu,
            root_rows=root_rows,
            witnesses=witnesses,
            depth=horizon_actions,
            forecast_state_space=forecast_state_space,
            expected_root_action_count=1,
        )

    @staticmethod
    def _normalized_hidden_world(
        source: NormalizedExactStateV1,
        state: RoundState,
    ) -> NormalizedExactStateV1:
        payload = state.to_dict(reveal_hands=True)
        state_hash = canonical_sha256(payload)
        legal_actions = tuple(
            action.to_dict()
            for action in BalootEngine(source.rules).legal_play_actions(state)
        )
        next_actor = (
            state.current_player.name.lower()
            if state.current_player is not None
            else None
        )
        event_chain_hash = canonical_sha256(payload.get("events") or [])
        core = {
            "schemaVersion": 1,
            "prefixHash": source.prefix_hash,
            "rulesHash": source.rules_hash,
            "stateHash": state_hash,
            "phase": state.phase.value,
            "decisionKind": source.decision_kind,
            "nextActor": next_actor,
            "legalActions": [dict(item) for item in legal_actions],
            "eventChainHash": event_chain_hash,
        }
        digest = canonical_sha256(core)
        return NormalizedExactStateV1(
            f"STATE-{digest[:20].upper()}",
            source.prefix_hash,
            source.rules_hash,
            state_hash,
            state.phase.value,
            source.decision_kind,
            next_actor,
            legal_actions,
            payload,
            source.project_declarations_verified,
            source.project_assertions_verified,
            event_chain_hash,
            digest,
            state.clone(),
            source.rules,
        )

    def _policy_root_actions(
        self,
        state: RoundState,
        request: ScenarioProofRequestV1,
        engine: BalootEngine,
        counter: _Counter,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...], int]:
        policies = self._validate_policy_request(request)
        rows: list[dict[str, Any]] = []
        best_line: tuple[str, ...] = ()
        target = _team(request.target_team)
        root_maximizes = state.current_player is not None and state.current_player.team == target
        for action in _legal_proof_play_actions(engine, state):
            clone = state.clone()
            engine.play_card(
                clone,
                action.card,
                ekka=action.ekka,
                emit_events=False,
            )
            line = [action.code]
            bots = {
                seat: HeuristicBot(engine, BotProfile(int(policy["seed"])))
                for seat, policy in policies.items()
            }
            while clone.current_player is not None:
                counter.visit()
                selected = bots[clone.current_player.name.lower()].choose(clone)
                line.append(selected.code)
                engine.play_card(clone, selected, emit_events=False)
            value = _terminal_utility(clone, request)
            rows.append(
                {
                    "action": action.code,
                    "value": value,
                    "satisfies": value >= request.success_threshold,
                    "completionStatus": "policy_complete",
                    "witness": line,
                }
            )
        selected = (max if root_maximizes else min)(
            rows,
            key=lambda item: (float(item["value"]), str(item["action"])),
        )
        best_line = tuple(str(item) for item in selected["witness"])
        return rows, best_line, sum(bool(item["satisfies"]) for item in rows)

    def prove(
        self,
        normalized: NormalizedExactStateV1,
        request: ScenarioProofRequestV1,
        *,
        bundle_hash: str,
        hidden_worlds: Sequence[NormalizedExactStateV1] | None = None,
        hidden_world_coverage: Mapping[str, Any] | None = None,
        selected_action: str | None = None,
        cancel_check: Any | None = None,
    ) -> ProofOutcomeV1:
        if (
            selected_action is not None
            and request.selected_action is not None
            and selected_action != request.selected_action
        ):
            raise ValueError("proof selected action conflicts with the request")
        requested_selected_action = selected_action or request.selected_action
        legal_root_actions = (
            {
                action.code
                for action in _legal_proof_play_actions(
                    BalootEngine(normalized.rules),
                    normalized.round_state,
                )
            }
            if normalized.phase == "playing"
            else {
                str(item.get("code") or "")
                for item in normalized.legal_actions
            }
        )
        if (
            requested_selected_action is not None
            and requested_selected_action not in legal_root_actions
        ):
            raise ValueError("proof selectedAction is not a legal root action")
        if (
            request.mode == "exact"
            and request.visibility == "actor_information_set"
            and hidden_worlds is not None
        ):
            raise ValueError(
                "exact actor-information-set worlds must be generated internally"
            )
        if normalized.phase == "bidding":
            return self._prove_policy_bid(
                normalized,
                request,
                bundle_hash=bundle_hash,
                selected_action=selected_action,
                cancel_check=cancel_check,
            )
        if normalized.phase == "final_dealt":
            return self._prove_policy_escalation(
                normalized,
                request,
                bundle_hash=bundle_hash,
                selected_action=selected_action,
                cancel_check=cancel_check,
            )
        if normalized.decision_kind == "project":
            return self._prove_policy_project(
                normalized,
                request,
                bundle_hash=bundle_hash,
                selected_action=selected_action,
                cancel_check=cancel_check,
            )
        if normalized.phase != "playing" or normalized.round_state.current_player is None:
            raise ValueError("scenario proof requires a playable exact decision state")
        budget = self.budgets.get(request.budget_profile)
        if request.visibility == "actor_information_set" and hidden_worlds is not None:
            supplied_world_count = len(hidden_worlds)
            if supplied_world_count == 0:
                raise ValueError(
                    "actor-information-set proof cannot use an empty world cohort"
                )
            if supplied_world_count > budget.maximum_hidden_worlds:
                raise ValueError(
                    "actor-information-set world cohort exceeds maximumHiddenWorlds"
                )
        started = time.monotonic()
        started_cpu = time.thread_time()
        node_limit = (
            min(budget.maximum_nodes, budget.maximum_line_count_nodes)
            if request.claim == "winning_continuation_line_count"
            or request.line_universe is not None
            else budget.maximum_nodes
        )
        counter = self._counter(
            budget,
            maximum_nodes=node_limit,
            started_wall=started,
            started_cpu=started_cpu,
            cancel_check=cancel_check,
        )
        forecast_state_space: int | None = None
        generated_coverage_incomplete = False
        generated_internally = False
        if request.visibility == "actor_information_set" and hidden_worlds is None:
            generated_internally = True
            actor = normalized.round_state.current_player
            assert actor is not None
            information = InformationState.from_complete(
                normalized.round_state,
                actor,
            )
            observation_hash = canonical_sha256(information.to_dict())
            generation_seed = int(normalized.state_hash[:16], 16)
            generation_input_hash = canonical_sha256(
                {
                    "schemaVersion": 1,
                    "observationHash": observation_hash,
                    "rulesHash": normalized.rules_hash,
                    "requestedSamples": budget.maximum_hidden_worlds,
                    "exactAllocationLimit": budget.maximum_hidden_worlds,
                    "maximumAttempts": budget.maximum_hidden_worlds,
                    "seed": generation_seed,
                    "strategyClass": self._strategy_class(request),
                }
            )

            def generation_cancelled() -> bool:
                try:
                    counter.poll()
                except ProofBudgetExceeded:
                    return True
                return False

            try:
                if request.mode == "policy_conditioned":
                    forecast_state_space = self._preflight_policy_budget(
                        budget,
                        counter,
                        horizon_actions=sum(
                            mask.bit_count()
                            for mask in normalized.round_state.hands
                        ),
                        root_action_count=_root_action_upper_bound(
                            BalootEngine(normalized.rules),
                            normalized.round_state,
                        ),
                        world_count=budget.maximum_hidden_worlds,
                    )
                else:
                    preflight_depth = self._depth(
                        request,
                        normalized.round_state,
                        budget,
                    )
                    forecast_state_space = self._state_space_forecast(
                        normalized.round_state,
                        depth=preflight_depth,
                        world_count=budget.maximum_hidden_worlds,
                        exact=request.mode == "exact",
                    )
                    counter.forecast_state_space = forecast_state_space
                    if forecast_state_space > budget.maximum_state_space:
                        counter._exhaust("maximum_state_space")
                generated = generate_hidden_deals(
                    information,
                    BalootEngine(normalized.rules),
                    requested_samples=budget.maximum_hidden_worlds,
                    exact_allocation_limit=budget.maximum_hidden_worlds,
                    maximum_attempts=budget.maximum_hidden_worlds,
                    seed=generation_seed,
                    cancel_check=generation_cancelled,
                )
                hidden_worlds = tuple(
                    self._normalized_hidden_world(normalized, state)
                    for state in generated.states
                )
                coverage_core = {
                    "method": generated.method,
                    "complete": generated.coverage_complete,
                    "expectedWorldCount": len(hidden_worlds),
                    "observationHash": observation_hash,
                    "generatorInputHash": generation_input_hash,
                    "worldHashes": sorted(
                        world.state_hash for world in hidden_worlds
                    ),
                }
                hidden_world_coverage = {
                    **coverage_core,
                    "sha256": canonical_sha256(coverage_core),
                }
                generated_coverage_incomplete = not generated.coverage_complete
            except (InterruptedError, ProofBudgetExceeded) as exc:
                hidden_worlds = (normalized,)
                coverage_core = {
                    "method": "constructive_uniform_hidden_deals",
                    "complete": False,
                    "expectedWorldCount": 1,
                    "observationHash": observation_hash,
                    "generatorInputHash": generation_input_hash,
                    "worldHashes": [normalized.state_hash],
                }
                hidden_world_coverage = {
                    **coverage_core,
                    "sha256": canonical_sha256(coverage_core),
                }
                generated_coverage_incomplete = True
                if isinstance(exc, ProofBudgetExceeded):
                    if counter.exhaustion_reason is None:
                        counter.exhaustion_reason = str(exc)
                elif counter.exhaustion_reason is None:
                    counter.cancelled = True
                    counter.exhaustion_reason = "cancelled"
        worlds = tuple(hidden_worlds) if hidden_worlds is not None else (normalized,)
        if request.visibility == "complete_state" and len(worlds) != 1:
            raise ValueError("complete-state proof accepts exactly one world")
        if request.visibility == "actor_information_set":
            if hidden_worlds is None:
                raise ValueError("actor-information-set proof requires enumerated worlds")
            if len({world.state_hash for world in worlds}) != len(worlds):
                raise ValueError("actor-information-set proof worlds must be unique")
            worlds = tuple(sorted(worlds, key=lambda world: world.state_hash))
            legal_domains = {
                tuple(item["code"] for item in world.to_dict()["legalActions"])
                for world in worlds
            }
            if len(legal_domains) != 1:
                raise ValueError("information-set worlds expose different visible actions")
            if hidden_world_coverage is None:
                raise ValueError("actor-information-set proof requires a coverage receipt")
            coverage = dict(hidden_world_coverage)
            expected_coverage_fields = {
                "method",
                "complete",
                "expectedWorldCount",
                "observationHash",
            }
            if generated_internally:
                expected_coverage_fields.update(
                    {"generatorInputHash", "worldHashes", "sha256"}
                )
            if set(coverage) != expected_coverage_fields:
                raise ValueError("hidden-world coverage receipt shape is invalid")
            if coverage["method"] not in {
                "exhaustive_constrained_hidden_deals",
                "exhaustive_project_conditioned_hidden_deals",
                "exhaustive_reference_fixture",
                "constructive_uniform_hidden_deals",
                "constructive_uniform_project_conditioned_hidden_deals",
            }:
                raise ValueError("actor-information-set coverage method is invalid")
            if coverage["complete"] is not True and not generated_coverage_incomplete:
                raise ValueError("actor-information-set coverage is not exhaustive")
            if type(coverage["expectedWorldCount"]) is not int or int(
                coverage["expectedWorldCount"]
            ) != len(worlds):
                raise ValueError("hidden-world coverage count mismatch")
            actor = normalized.round_state.current_player
            assert actor is not None
            observation_hashes = {
                canonical_sha256(
                    InformationState.from_complete(
                        world.round_state,
                        actor,
                    ).to_dict()
                )
                for world in worlds
            }
            if observation_hashes != {str(coverage["observationHash"])}:
                raise ValueError("hidden worlds are not actor-indistinguishable")
            if generated_internally:
                if coverage["worldHashes"] != [
                    world.state_hash for world in worlds
                ]:
                    raise ValueError("hidden-world coverage hashes mismatch")
                coverage_core = dict(coverage)
                declared_coverage_hash = str(coverage_core.pop("sha256"))
                if canonical_sha256(coverage_core) != declared_coverage_hash:
                    raise ValueError("hidden-world coverage checksum mismatch")
        completion_status = "proof_incomplete"
        achieved_mode = "none"
        achieved_visibility = "none"
        root_rows: list[dict[str, Any]] = []
        principal: tuple[str, ...] = ()
        winning_count = 0
        continuation: dict[str, Any] = {
            "complete": False,
            "lowerBound": "0",
            "upperBound": None,
            "lineUniverse": request.line_universe or "optimal_strategy",
        }
        cache = _ProofCache(
            budget.maximum_cache_entries,
            budget.maximum_cache_bytes,
        )
        information_cache_entries = 0
        information_cache_bytes = 0
        evaluated_depth = 0
        requested_selected_action = selected_action or request.selected_action
        if (
            selected_action is not None
            and request.selected_action is not None
            and selected_action != request.selected_action
        ):
            raise ValueError("proof selected action conflicts with the request")
        try:
            if counter.exhaustion_reason is not None:
                raise ProofBudgetExceeded(counter.exhaustion_reason)
            if generated_coverage_incomplete:
                raise ProofBudgetExceeded(
                    "actor-information-set world coverage exceeds configured budget"
                )
            if request.mode == "policy_conditioned":
                policy_horizon = sum(
                    mask.bit_count() for mask in normalized.round_state.hands
                )
                evaluated_depth = policy_horizon
                forecast_state_space = self._preflight_policy_budget(
                    budget,
                    counter,
                    horizon_actions=policy_horizon,
                    root_action_count=_root_action_upper_bound(
                        BalootEngine(normalized.rules),
                        normalized.round_state,
                    ),
                    world_count=len(worlds),
                )
                policy_worlds = []
                for world in worlds:
                    rows, witness, _count = self._policy_root_actions(
                        world.round_state,
                        request,
                        BalootEngine(world.rules),
                        counter,
                    )
                    policy_worlds.append((rows, witness))
                action_sets = [
                    {str(item["action"]): item for item in rows}
                    for rows, _witness in policy_worlds
                ]
                root_action_domain = tuple(sorted(action_sets[0]))
                if any(
                    tuple(sorted(rows)) != root_action_domain
                    for rows in action_sets[1:]
                ):
                    counter._exhaust(
                        "actor_information_set_action_domain_mismatch"
                    )
                root_rows = [
                    {
                        "action": action,
                        "value": min(float(rows[action]["value"]) for rows in action_sets),
                        "satisfies": all(
                            bool(rows[action]["satisfies"])
                            for rows in action_sets
                        ),
                        "worldValues": [
                            float(rows[action]["value"])
                            for rows in action_sets
                        ],
                        "completionStatus": "policy_complete",
                    }
                    for action in root_action_domain
                ]
                policy_witnesses: dict[str, tuple[str, ...]] = {}
                for row in root_rows:
                    action = str(row["action"])
                    worst_world, worst_rows = min(
                        zip(worlds, action_sets, strict=True),
                        key=lambda item: (
                            float(item[1][action]["value"]),
                            item[0].state_hash,
                            tuple(item[1][action]["witness"]),
                        ),
                    )
                    witness = tuple(
                        str(item) for item in worst_rows[action]["witness"]
                    )
                    policy_witnesses[action] = witness
                    row["worstWorldHash"] = worst_world.state_hash
                    row["worstWorldWitnessHash"] = canonical_sha256(
                        {
                            "worldHash": worst_world.state_hash,
                            "actions": list(witness),
                        }
                    )
                winning_count = sum(bool(item["satisfies"]) for item in root_rows)
                target = _team(request.target_team)
                maximizing = normalized.round_state.current_player.team == target
                selected = next(
                    (
                        item
                        for item in root_rows
                        if str(item["action"]) == requested_selected_action
                    ),
                    None,
                )
                if selected is None and requested_selected_action is not None:
                    raise ValueError("proof selectedAction is not a legal root action")
                if selected is None:
                    selected = (max if maximizing else min)(
                        root_rows,
                        key=lambda item: (
                            float(item["value"]),
                            str(item["action"]),
                        ),
                    )
                principal = policy_witnesses[str(selected["action"])]
                completion_status = "policy_complete"
                achieved_mode = "policy_conditioned"
                achieved_visibility = request.visibility
                selected_policy_line_count = sum(
                    bool(rows[str(selected["action"])]["satisfies"])
                    for rows in action_sets
                )
                continuation = {
                    "complete": True,
                    "exactCount": str(selected_policy_line_count),
                    "lineUniverse": "policy_conditioned",
                }
            else:
                depth = self._depth(request, normalized.round_state, budget)
                evaluated_depth = depth
                shared_information_strategy = (
                    request.visibility == "actor_information_set" and len(worlds) > 1
                )
                if (
                    shared_information_strategy
                    and request.claim == "winning_continuation_line_count"
                ):
                    counter._exhaust(
                        "actor_information_set_continuation_count_not_supported"
                    )
                root_evaluations: dict[str, _Evaluation] = {}
                joint_evaluations: dict[str, _JointRootEvaluation] = {}
                if shared_information_strategy:
                    forecast_state_space = 0
                    root_domains = [
                        _legal_proof_play_actions(
                            BalootEngine(world.rules),
                            world.round_state,
                        )
                        for world in worlds
                    ]
                    root_codes = tuple(
                        action.code for action in root_domains[0]
                    )
                    if any(
                        tuple(action.code for action in domain) != root_codes
                        for domain in root_domains[1:]
                    ):
                        counter._exhaust(
                            "actor_information_set_action_domain_mismatch"
                        )
                    root_actions = root_domains[0]
                    for action in root_actions:
                        evaluation = self._evaluate_information_set_root(
                            worlds,
                            root_action=action,
                            request=request,
                            depth=depth,
                            budget=budget,
                            counter=counter,
                            remaining_state_space=(
                                budget.maximum_state_space - forecast_state_space
                            ),
                        )
                        forecast_state_space += evaluation.forecast_state_space
                        information_cache_entries = max(
                            information_cache_entries,
                            evaluation.catalog_entries,
                        )
                        information_cache_bytes = max(
                            information_cache_bytes,
                            evaluation.catalog_bytes,
                        )
                        joint_evaluations[action.code] = evaluation
                        root_rows.append(
                            {
                                "action": action.code,
                                "value": evaluation.value,
                                "satisfies": (
                                    evaluation.value >= request.success_threshold
                                ),
                                "worldValues": list(evaluation.world_values),
                                "strategyScope": "pure_nonanticipative_maximin",
                                "targetPolicyCount": evaluation.target_policy_count,
                                "opponentPolicyCount": evaluation.opponent_policy_count,
                                "targetPolicyHash": evaluation.target_policy_hash,
                                "worstOpponentPolicyHash": (
                                    evaluation.opponent_policy_hash
                                ),
                                "worstWorldHash": evaluation.worst_world_hash,
                                "worstWorldWitnessHash": canonical_sha256(
                                    {
                                        "worldHash": evaluation.worst_world_hash,
                                        "actions": list(evaluation.worst_path),
                                    }
                                ),
                                "worstWorldWitness": {
                                    "visibility": "audit_only",
                                    "worldHash": evaluation.worst_world_hash,
                                    "actions": list(evaluation.worst_path),
                                },
                                "completionStatus": (
                                    "exact"
                                    if request.mode == "exact"
                                    else "bounded"
                                ),
                            }
                        )
                else:
                    forecast_state_space = self._state_space_forecast(
                        normalized.round_state,
                        depth=depth,
                        world_count=len(worlds),
                        exact=request.mode == "exact",
                    )
                    counter.forecast_state_space = forecast_state_space
                    if forecast_state_space > budget.maximum_state_space:
                        counter._exhaust("maximum_state_space")
                    per_world_rows: list[dict[str, _Evaluation]] = []
                    for world in worlds:
                        world_engine = BalootEngine(world.rules)
                        evaluations: dict[str, _Evaluation] = {}
                        for action in _legal_proof_play_actions(
                            world_engine,
                            world.round_state,
                        ):
                            state = world.round_state.clone()
                            token = world_engine.apply(
                                state,
                                action.card,
                                ekka=action.ekka,
                                verify_identity=False,
                                emit_events=False,
                            )
                            try:
                                evaluations[action.code] = self._evaluate(
                                    state,
                                    engine=world_engine,
                                    request=request,
                                    depth=max(depth - 1, 0),
                                    counter=counter,
                                    cache=cache,
                                )
                            finally:
                                world_engine.undo(state, token)
                        per_world_rows.append(evaluations)
                    root_evaluations = {
                        action: min(
                            (rows[action] for rows in per_world_rows),
                            key=lambda item: (
                                item.value,
                                item.principal_variation,
                            ),
                        )
                        for action in sorted(per_world_rows[0])
                    }
                    for action, evaluation in sorted(root_evaluations.items()):
                        value = evaluation.value
                        root_rows.append(
                            {
                                "action": action,
                                "value": value,
                                "satisfies": value >= request.success_threshold,
                                "worldValues": [
                                    rows[action].value for rows in per_world_rows
                                ],
                                "strategyScope": "complete_state",
                                "completionStatus": (
                                    "exact"
                                    if request.mode == "exact"
                                    else "bounded"
                                ),
                            }
                        )
                winning_count = sum(bool(item["satisfies"]) for item in root_rows)
                target = _team(request.target_team)
                maximizing = normalized.round_state.current_player.team == target
                selected = next(
                    (
                        item
                        for item in root_rows
                        if str(item["action"]) == requested_selected_action
                    ),
                    None,
                )
                if selected is None and requested_selected_action is not None:
                    raise ValueError("proof selectedAction is not a legal root action")
                if selected is None:
                    selected = (max if maximizing else min)(
                        root_rows,
                        key=lambda item: (
                            float(item["value"]),
                            str(item["action"]),
                        ),
                    )
                selected_eval = root_evaluations.get(str(selected["action"]))
                principal = (
                    (str(selected["action"]),)
                    if shared_information_strategy
                    else (
                        str(selected["action"]),
                        # The complete-state evaluator retains an exact line.
                        # Multi-world lines remain behind their witness receipt.
                        # This avoids presenting one hidden world as universal.
                        *(
                            selected_eval.principal_variation
                            if selected_eval is not None
                            else ()
                        ),
                    )
                )
                line_universe = request.line_universe or "optimal_strategy"
                if shared_information_strategy:
                    continuation = {
                        "complete": False,
                        "lowerBound": "0",
                        "upperBound": None,
                        "lineUniverse": line_universe,
                    }
                else:
                    assert selected_eval is not None
                    line_count = (
                        selected_eval.winning_line_count
                        if line_universe == "all_legal"
                        else selected_eval.optimal_line_count
                    )
                    continuation = {
                        "complete": True,
                        "exactCount": str(line_count),
                        "lineUniverse": line_universe,
                    }
                achieved_mode = request.mode
                achieved_visibility = request.visibility
                completion_status = (
                    "exact_complete" if request.mode == "exact" else "bounded_complete"
                )
        except ProofBudgetExceeded as exc:
            if counter.exhaustion_reason is None:
                counter.exhaustion_reason = str(exc)
            completion_status = "proof_incomplete"
        # Rows are retained as partial evidence. Their satisfying count is an
        # honest lower bound even when a later root exhausts the proof budget.
        winning_count = sum(bool(item.get("satisfies")) for item in root_rows)
        if counter.forecast_state_space is not None:
            forecast_state_space = counter.forecast_state_space
        counter.finish()
        cancelled = counter.cancelled
        if cancelled:
            completion_status = "proof_incomplete"
            achieved_mode = "none"
            achieved_visibility = "none"
        elif counter.exhaustion_reason is not None:
            completion_status = "proof_incomplete"
            achieved_mode = "none"
            achieved_visibility = "none"
        selected_code = requested_selected_action or (
            principal[0] if principal else None
        )
        selected_row = next(
            (
                item
                for item in root_rows
                if str(item.get("action")) == str(selected_code)
            ),
            None,
        )
        claim_satisfied = False
        if completion_status != "proof_incomplete":
            if request.claim == "outcome_reachable":
                claim_satisfied = winning_count >= 1
            elif request.claim == "outcome_forced":
                claim_satisfied = bool(root_rows) and winning_count == len(root_rows)
            elif request.claim == "selected_action_is_unique_winner":
                claim_satisfied = (
                    winning_count == 1
                    and selected_row is not None
                    and bool(selected_row["satisfies"])
                )
            elif request.claim == "target_team_has_no_salvage":
                claim_satisfied = winning_count == 0
            else:
                claim_satisfied = True
            if not claim_satisfied:
                completion_status = "proof_failed"
        identities = {
            "rulesHash": normalized.rules_hash,
            "bundleHash": str(bundle_hash),
            "stateHash": normalized.state_hash,
            "exactStateId": normalized.exact_state_id,
            "proofProfileHash": budget.sha256,
            "proofBudgetRegistryHash": self.budgets.sha256,
            "proofRequestHash": request.sha256,
            "strategyClass": self._strategy_class(request),
            "engine": {"id": self.engine_id, "version": self.engine_version},
            "policyHashes": (
                {
                    seat: str(value["profileHash"])
                    for seat, value in dict(request.policies or {}).items()
                }
            ),
        }
        payload = {
            "schemaVersion": 1,
            "requestedMode": request.mode,
            "achievedMode": achieved_mode,
            "requestedVisibility": request.visibility,
            "achievedVisibility": achieved_visibility,
            "strategyClass": self._strategy_class(request),
            "claim": request.claim,
            "targetTeam": request.target_team,
            "utility": request.utility,
            "successThreshold": request.success_threshold,
            "horizon": copy.deepcopy(request.horizon),
            "boundaryEvaluator": request.boundary_evaluator,
            "rootActions": root_rows,
            "selectedAction": selected_code,
            "winningRootActionCount": winning_count,
            "continuationLines": continuation,
            "hiddenWorlds": {
                "count": len(worlds),
                "complete": (
                    bool(dict(hidden_world_coverage or {}).get("complete"))
                    if request.visibility == "actor_information_set"
                    else True
                ),
                "hashes": [world.state_hash for world in worlds],
                "coverage": (
                    copy.deepcopy(dict(hidden_world_coverage))
                    if hidden_world_coverage is not None
                    else self._complete_state_coverage(normalized)
                ),
            },
            "resources": {
                "nodes": counter.nodes,
                "depth": evaluated_depth,
                "cacheEntries": max(
                    len(cache.values),
                    information_cache_entries,
                    counter.retained_cache_entries,
                ),
                "cacheBytes": max(
                    cache.estimated_bytes,
                    information_cache_bytes,
                    counter.retained_cache_bytes,
                ),
                "elapsedWallSeconds": time.monotonic() - started,
                "elapsedCpuSeconds": time.thread_time() - started_cpu,
                "nodeLimit": node_limit,
                "depthLimit": budget.maximum_depth,
                "wallTimeLimitSeconds": budget.maximum_wall_seconds,
                "cacheEntryLimit": budget.maximum_cache_entries,
                "cacheByteLimit": budget.maximum_cache_bytes,
                "hiddenWorldLimit": budget.maximum_hidden_worlds,
                "horizonActionLimit": budget.maximum_horizon_actions,
                "cpuTimeLimitSeconds": budget.maximum_cpu_seconds,
                "cancellationCheckIntervalNodes": budget.cancellation_check_interval_nodes,
                "workerLimit": budget.maximum_workers,
                "workers": 1,
                "strategyClass": self._strategy_class(request),
                "lineCountNodeLimit": budget.maximum_line_count_nodes,
                "evaluatedRootActionCount": len(root_rows),
                "expectedRootActionCount": len(legal_root_actions),
                "rootActionCoverageComplete": (
                    len(root_rows) == len(legal_root_actions)
                ),
                "winningRootActionLowerBound": winning_count,
                "winningRootActionUpperBound": (
                    winning_count
                    + max(0, len(legal_root_actions) - len(root_rows))
                ),
                "forecastStateSpace": (
                    counter.forecast_state_space
                    if counter.forecast_state_space is not None
                    else forecast_state_space
                ),
                "stateSpaceLimit": budget.maximum_state_space,
                "exhaustionReason": counter.exhaustion_reason,
                "cancelled": cancelled,
            },
            "completionStatus": completion_status,
            "claimSatisfied": claim_satisfied,
            "principalVariation": list(principal),
            "identities": identities,
        }
        return ProofOutcomeV1.build(payload)
__all__ = [
    "decode_proof_play_action_v1",
    "ProofBudgetExceeded",
    "ScenarioProofBudgetRegistryV1",
    "ScenarioProofBudgetV1",
    "UnifiedScenarioProofServiceV1",
]
