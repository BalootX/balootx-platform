from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from baloot_platform.agents.profiles import AdaptiveBiddingStrategy, AdaptivePlayStrategy, AgentProfileRegistry
from baloot_platform.config import ProfileBundle, build_bundle
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.version import __version__


@dataclass(slots=True)
class GameSession:
    id: str
    mode: str
    engine: BalootEngine
    state: RoundState
    bidding: BiddingMachine | None
    version: int = 1
    bundle_hash: str = ""
    parent_game_hash: str | None = None
    game_hash: str | None = None
    serial: str | None = None
    agent_profiles: dict[str, str] = field(default_factory=dict)
    shared_environment: dict[str, Any] | None = None
    deal_schedule_index: int = 0

    def to_dict(self, *, reveal_hands: bool = False) -> dict[str, Any]:
        data = self.state.to_dict(reveal_hands=reveal_hands)
        data.update(
            {
                "gameId": self.id,
                "version": self.version,
                "mode": self.mode,
                "bundleHash": self.bundle_hash,
                "parentGameHash": self.parent_game_hash,
                "gameHash": self.game_hash,
                "serial": self.serial,
                "agentProfiles": dict(self.agent_profiles),
                "sharedEnvironment": ({
                    "scope": self.shared_environment.get("scope"),
                    "scopeLabelAr": self.shared_environment.get("scopeLabelAr"),
                    "environmentHash": self.shared_environment.get("environmentHash"),
                    "dealScheduleHash": self.shared_environment.get("dealScheduleHash"),
                    "agentSnapshotHash": self.shared_environment.get("agentSnapshotHash"),
                    "dealScheduleIndex": self.deal_schedule_index,
                } if self.shared_environment else None),
            }
        )
        if self.bidding and not self.bidding.finished:
            data["bidding"] = {
                "round": self.bidding.round_index,
                "actor": self.bidding.actor.name.lower(),
                "legalActions": list(self.bidding.legal_actions()),
                "history": [record.to_dict() for record in self.bidding.records],
            }
        else:
            data["bidding"] = None
        return data


class GameManager:
    def __init__(self, root, store=None, *, learning_store=None, agent_registry: AgentProfileRegistry | None = None,
                 canonical_history=None, bundle_provider: Callable[[str], ProfileBundle] | None = None):
        self.root = root
        self.store = store
        self.learning_store = learning_store
        self.canonical_history = canonical_history
        self.bundle_provider = bundle_provider
        self.agent_registry = (
            agent_registry
            or AgentProfileRegistry.from_bundle(self.root)
        )
        self.learning_service: Any | None = None
        self._games: dict[str, GameSession] = {}
        self._lock = threading.RLock()

    def _bundle(self, mode: str) -> ProfileBundle:
        if self.bundle_provider is not None:
            return self.bundle_provider(mode)
        return build_bundle(self.root, mode=mode)

    def bundle(self, mode: str) -> ProfileBundle:
        """Return the exact governed bundle used by this manager."""
        return self._bundle(mode)

    def _persist(self, game: GameSession) -> None:
        if self.store is not None:
            self.store.save(game)
        if self.canonical_history is not None:
            stream_id = f"{game.id}:round:{game.state.round_number}"
            self.canonical_history.append(
                game_id=stream_id, events=tuple(game.state.events),
                profile_bundle_hash=game.bundle_hash, engine_version=__version__,
            )
            self.canonical_history.snapshot(
                game_id=stream_id, state=game.state.to_dict(reveal_hands=True),
                engine_version=__version__, profile_bundle_hash=game.bundle_hash,
            )

    def _archive(self, game: GameSession, *, status: str, label: str | None = None) -> dict[str, Any] | None:
        if self.learning_store is None:
            return None
        result = self.learning_store.archive_game(
            game,
            bundle_hash=game.bundle_hash,
            engine_version=__version__,
            parent_game_hash=game.parent_game_hash,
            status=status,
            label=label,
        )
        game.game_hash = result["gameHash"]
        game.serial = result["serial"]
        return result

    def create(
        self,
        *,
        seed: int,
        dealer: Seat,
        mode: str = "full",
        match_score=(0, 0),
        agent_profiles: Mapping[str, str] | None = None,
    ) -> GameSession:
        bundle = self._bundle(mode)
        engine = BalootEngine(bundle.rules)
        state = engine.create_round(seed=seed, dealer=dealer, match_score=match_score)
        engine.shuffle_and_initial_deal(state)
        bidding = BiddingMachine(
            state.dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(bundle.rules.data),
            state.purchase_card.rank,
        )
        if bidding.finished and bidding.result is not None:
            completed = engine.complete_deal_from_bid_result(state, bidding.result)
            if completed:
                engine.prepare_projects(state)
        lineup = dict(self.agent_registry.default_lineup)
        if agent_profiles:
            for seat, profile in agent_profiles.items():
                self.agent_registry.get(profile)
                lineup[str(seat).lower()] = profile
        game = GameSession(
            uuid.uuid4().hex[:16],
            mode,
            engine,
            state,
            bidding,
            1,
            bundle.sha256,
            agent_profiles=lineup,
        )
        with self._lock:
            self._games[game.id] = game
        self._persist(game)
        return game

    def import_session(self, session: GameSession, *, replace_existing: bool = False) -> GameSession:
        with self._lock:
            if session.id in self._games and not replace_existing:
                raise ValueError(f"game {session.id} already exists")
            self._games[session.id] = session
            self._persist(session)
        return session

    def list(self, *, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        if self.store is not None and hasattr(self.store, "list_games"):
            for item in self.store.list_games(query=query, limit=limit):
                results[item["gameId"]] = item
        with self._lock:
            for game in self._games.values():
                if query and query.lower() not in game.id.lower() and query.lower() not in (game.serial or "").lower():
                    continue
                results[game.id] = game.to_dict(reveal_hands=False)
        return list(results.values())[: max(1, min(5000, int(limit)))]

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            if game_id not in self._games:
                loaded = self.store.load(game_id) if self.store is not None else None
                if loaded is None:
                    raise KeyError(game_id)
                self._games[game_id] = loaded
            return self._games[game_id]

    def versions(self, game_id: str) -> list[dict[str, Any]]:
        self.get(game_id)
        return self.store.list_versions(game_id) if self.store is not None and hasattr(self.store, "list_versions") else []

    def branch_from_version(self, game_id: str, version: int) -> GameSession:
        if self.store is None or not hasattr(self.store, "load_version"):
            raise RuntimeError("versioned game storage is unavailable")
        original = self.store.load_version(game_id, int(version))
        if original is None:
            raise KeyError(f"{game_id}@{version}")
        parent_hash = original.game_hash
        if not parent_hash and self.learning_store is not None:
            archived = self._archive(original, status="branch_source")
            parent_hash = archived["gameHash"] if archived else None
        original.id = uuid.uuid4().hex[:16]
        original.version = 1
        original.parent_game_hash = parent_hash
        original.game_hash = None
        original.serial = None
        return self.import_session(original)

    def bid(
        self,
        game_id: str,
        *,
        seat: Seat,
        action: str,
        trump: Suit | None = None,
        expected_version: int | None = None,
        decision_metadata: Mapping[str, Any] | None = None,
    ) -> GameSession:
        with self._lock:
            game = self.get(game_id)
            self._version(game, expected_version)
            if game.bidding is None or game.bidding.finished:
                raise ValueError("bidding is not active")
            if game.bidding.actor != seat:
                raise ValueError(f"it is {game.bidding.actor.name.lower()}'s turn")
            if self.learning_store is not None:
                from baloot_platform.analysis.bidding import legal_bid_choices
                review_legal = tuple(choice.to_dict() for choice in legal_bid_choices(game.bidding))
                self.learning_store.record_decision(
                    game_id=game.id,
                    game_version=game.version,
                    mode=game.mode,
                    state=game.state,
                    actor=seat,
                    decision_kind="bid",
                    legal_actions=review_legal,
                    selected_action={"action": action, "trumpSuit": trump.name.lower() if trump else None},
                    bidding=copy.deepcopy(game.bidding),
                    decision_metadata=dict(decision_metadata or {
                        "schemaVersion": 1,
                        "source": "human",
                        "decisionSource": "external_input",
                        "observerSafe": True,
                    }),
                )
            result = game.bidding.submit(action, trump=trump)
            if result is not None:
                completed = game.engine.complete_deal_from_bid_result(game.state, result)
                if completed:
                    game.engine.prepare_projects(game.state)
            game.version += 1
            self._persist(game)
            if result is not None and result.all_passed:
                self._archive(game, status="all_passed")
                game = self._redeal_after_all_pass(game)
            return game

    def _redeal_after_all_pass(self, game: GameSession) -> GameSession:
        bundle = self._bundle(game.mode)
        engine = BalootEngine(bundle.rules)
        cut_position = None
        if game.shared_environment is not None:
            from baloot_platform.shared import next_deal_entry, verify_shared_environment
            check = verify_shared_environment(game.shared_environment)
            if not check["valid"]:
                raise RuntimeError(f"shared environment is invalid: {check['issues']}")
            entry = next_deal_entry(game.shared_environment, game.deal_schedule_index)
            game.deal_schedule_index += 1
            dealer = Seat.parse(entry["dealer"])
            seed = int(entry["seed"])
            cut_position = entry.get("cutPosition")
        else:
            policy = str(bundle.component("game").get("all_pass", {}).get("dealer_policy", "counter_clockwise_next"))
            game.engine.structural_domain.validate_legacy_all_pass_dealer_policy(
                policy
            )
            dealer = game.engine.next_round_dealer(game.state)
            seed = game.state.seed + 1
        state = engine.create_round(
            seed=seed,
            dealer=dealer,
            round_number=game.state.round_number + 1,
            match_score=tuple(game.state.match_score),
        )
        engine.shuffle_and_initial_deal(state, cut_position=cut_position)
        game.engine = engine
        game.state = state
        game.bidding = BiddingMachine(
            state.dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(bundle.rules.data),
            state.purchase_card.rank,
        )
        game.bundle_hash = bundle.sha256
        game.version += 1
        game.game_hash = None
        game.serial = None
        self._persist(game)
        return game

    def play(
        self,
        game_id: str,
        *,
        seat: Seat,
        card,
        ekka: bool = False,
        expected_version: int | None = None,
        decision_metadata: Mapping[str, Any] | None = None,
    ) -> GameSession:
        with self._lock:
            game = self.get(game_id)
            self._version(game, expected_version)
            if game.state.current_player != seat:
                current = game.state.current_player.name.lower() if game.state.current_player is not None else "nobody"
                raise ValueError(f"it is {current}'s turn")
            legal = tuple(game.engine.legal_moves(game.state))
            parsed = __import__("baloot_platform.core.cards", fromlist=["Card"]).Card.parse(card)
            if self.learning_store is not None:
                self.learning_store.record_decision(
                    game_id=game.id,
                    game_version=game.version,
                    mode=game.mode,
                    state=game.state,
                    actor=seat,
                    decision_kind="play",
                    legal_actions=legal,
                    selected_action=parsed,
                    decision_metadata=dict(decision_metadata or {
                        "schemaVersion": 1,
                        "source": "human",
                        "decisionSource": "external_input",
                        "observerSafe": True,
                    }),
                )
            game.engine.play_card(game.state, parsed, ekka=ekka)
            game.version += 1
            if game.state.current_player is None:
                self._archive(game, status="completed")
            self._persist(game)
            return game

    def advance_bots(
        self,
        game_id: str,
        *,
        stop_before: Seat | None = Seat.SOUTH,
        maximum_actions: int = 64,
    ) -> GameSession:
        """Advance observer-safe bidding and play agents through one game.

        A promoted model is advisory. Legal masks and authoritative state
        transitions remain owned by the rules engine. All-pass auctions are
        archived and redealt using the configured dealer policy.
        """
        with self._lock:
            game = self.get(game_id)
            count = 0
            bundle = self._bundle(game.mode)
            model = None
            if self.learning_service is not None:
                try:
                    model = self.learning_service.production_model()
                except Exception:
                    model = None

            while count < maximum_actions:
                if game.bidding is not None and not game.bidding.finished:
                    actor = game.bidding.actor
                    if actor is None or (stop_before is not None and actor == stop_before):
                        break
                    profile_id = game.agent_profiles.get(actor.name.lower(), self.agent_registry.default_profile)
                    profile = self.agent_registry.get(profile_id)
                    strategy = AdaptiveBiddingStrategy(profile, model=model)
                    action, trump, trace = strategy.choose_with_trace(
                        game.state, game.bidding, bundle.rules,
                        seed=game.state.seed + game.version * 65537 + int(actor),
                    )
                    previous_round = game.state.round_number
                    game = self.bid(
                        game.id, seat=actor, action=action, trump=trump, expected_version=game.version,
                        decision_metadata=trace,
                    )
                    count += 1
                    if game.state.round_number != previous_round:
                        bundle = self._bundle(game.mode)
                    continue

                if game.state.current_player is None:
                    break
                actor = game.state.current_player
                if stop_before is not None and actor == stop_before:
                    break
                profile_id = game.agent_profiles.get(actor.name.lower(), self.agent_registry.default_profile)
                profile = self.agent_registry.get(profile_id)
                strategy = AdaptivePlayStrategy(profile, bundle.convention, model=model)
                card, trace = strategy.choose_with_trace(
                    game.engine, game.state,
                    seed=game.state.seed + game.version * 104729 + int(actor),
                    timeout_fallback=True,
                )
                game = self.play(
                    game.id, seat=actor, card=card, expected_version=game.version,
                    decision_metadata=trace,
                )
                count += 1
            self._persist(game)
            return game

    def next_round(self, game_id: str, *, seed: int | None = None) -> GameSession:
        with self._lock:
            old = self.get(game_id)
            if old.state.phase not in {GamePhase.SCORED, GamePhase.COMPLETE}:
                raise ValueError("current round is not complete")
            score = tuple(old.state.match_score)
            bundle = self._bundle(old.mode)
            engine = BalootEngine(bundle.rules)
            cut_position = None
            if old.shared_environment is not None:
                from baloot_platform.shared import next_deal_entry, verify_shared_environment
                check = verify_shared_environment(old.shared_environment)
                if not check["valid"]:
                    raise RuntimeError(f"shared environment is invalid: {check['issues']}")
                entry = next_deal_entry(old.shared_environment, old.deal_schedule_index)
                old.deal_schedule_index += 1
                dealer = Seat.parse(entry["dealer"])
                new_seed = int(entry["seed"])
                cut_position = entry.get("cutPosition")
                if seed is not None and int(seed) != new_seed:
                    raise ValueError("a shared لفة/صكّة cannot override its precommitted deal seed")
            else:
                dealer = old.engine.next_round_dealer(old.state)
                new_seed = seed if seed is not None else old.state.seed + 1
            state = engine.create_round(
                seed=new_seed,
                dealer=dealer,
                round_number=old.state.round_number + 1,
                match_score=score,
            )
            engine.shuffle_and_initial_deal(state, cut_position=cut_position)
            old.engine = engine
            old.state = state
            old.bidding = BiddingMachine(
                state.dealer,
                state.purchase_card.suit,
                AuctionDomainIRV1.from_rules(bundle.rules.data),
                state.purchase_card.rank,
            )
            old.bundle_hash = bundle.sha256
            old.version += 1
            old.game_hash = None
            old.serial = None
            self._persist(old)
            return old

    @staticmethod
    def _version(game: GameSession, expected: int | None) -> None:
        if expected is not None and expected != game.version:
            raise ValueError(f"version conflict: expected {expected}, current {game.version}")
