from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baloot_platform.agents.profiles import AdaptiveBiddingStrategy, AgentProfileRegistry
from baloot_platform.core.bidding import _choose_action
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import cards_to_mask
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.engine import BalootEngine
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.ml.actions import ACTION_COUNT, ActionCodec
from baloot_platform.ml.features import FEATURE_COUNT, encode_bid_state
from baloot_platform.ml.models import ModelOutput
from baloot_platform.review import DecisionReviewService
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


class _PassModel:
    id = "test_pass_model"
    version = "1"

    def infer(self, observation, legal_mask):
        assert observation.shape == (FEATURE_COUNT,)
        logits = np.full(ACTION_COUNT, -1e9, dtype=np.float32)
        for index, legal in enumerate(legal_mask):
            if legal:
                logits[index] = 0.0
        logits[ActionCodec.bid("pass")] = 9.0
        return ModelOutput(logits, 0.0)


def _initial_round(root: Path, seed: int = 404):
    from baloot_platform.config import build_bundle

    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    state = engine.create_round(seed=seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(bundle.rules.data),
    )
    return bundle, state, machine


def test_bid_feature_encoder_is_observer_safe(root: Path):
    bundle, state, machine = _initial_round(root)
    actor = machine.actor
    assert actor is not None
    original = encode_bid_state(state, machine, actor)
    changed = state.clone()
    hidden = [seat for seat in Seat if seat != actor]
    changed.hands[int(hidden[0])], changed.hands[int(hidden[1])] = (
        changed.hands[int(hidden[1])],
        changed.hands[int(hidden[0])],
    )
    assert np.array_equal(original, encode_bid_state(changed, machine, actor))


def test_ashkal_heuristic_does_not_read_partner_hand(root: Path):
    bundle, state, machine = _initial_round(root, 405)
    machine.submit("pass")
    machine.submit("pass")
    actor = machine.actor
    assert actor is not None and "ashkal" in machine.legal_actions()
    actor_cards = tuple(state.hand(actor))
    seats = {seat: tuple(state.hand(seat)) for seat in Seat}
    first = _choose_action(machine, seats, state.purchase_card, bundle.rules)
    partner = actor.partner
    opponent = actor.next()
    seats[partner], seats[opponent] = seats[opponent], seats[partner]
    second = _choose_action(machine, seats, state.purchase_card, bundle.rules)
    assert first == second
    assert actor_cards == tuple(state.hand(actor))


def test_promoted_model_can_advise_buying_without_bypassing_legality(root: Path):
    bundle, state, machine = _initial_round(root, 406)
    registry = AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml")
    profile = registry.get("expert")
    action, trump = AdaptiveBiddingStrategy(profile, model=_PassModel()).choose(
        state, machine, bundle.rules, seed=1
    )
    assert action == "pass" and trump is None
    assert action in machine.legal_actions()


def test_reviewed_bid_enters_unified_learning_dataset(root: Path, tmp_path: Path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    manager = GameManager(
        root,
        SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=store,
        agent_registry=AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml"),
    )
    learning = LearningService(root, tmp_path / "learning", store=store, game_manager=manager)
    manager.learning_service = learning
    game = manager.create(seed=407, dealer=Seat.EAST)
    actor = game.bidding.actor
    manager.bid(game.id, seat=actor, action="pass", expected_version=game.version)
    decision = store.list_decisions(game.id)[0]
    learning.add_feedback(
        decision["decisionId"], preferred_action={"action": "pass", "trumpSuit": None},
        confidence=0.99, rationale="Legacy evidence is not a training label",
        source="legacy_feedback", reviewer="legacy-reviewer",
    )
    with pytest.raises(ValueError, match="no reviewed legal"):
        learning.build_dataset(minimum_confidence=0.5)
    reviews = DecisionReviewService(store)
    for reviewer, role in (("expert-bid", "expert"), ("peer-bid", "player")):
        reviews.submit(
            decision["decisionId"], reviewer_id=reviewer, reviewer_role=role,
            preferred_action={"action": "pass", "trumpSuit": None},
            confidence=0.95, reason_codes=["risk_management"],
            rationale="Observer-safe reviewed buy/pass decision",
            view_mode="observer",
        )
    dataset = learning.build_dataset(minimum_confidence=0.5)
    row = json.loads(Path(dataset["examples"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["decisionKind"] == "bid"
    assert len(row["features"]) == FEATURE_COUNT
    assert row["targetAction"] == ActionCodec.bid("pass")
    assert row["legalMask"][row["targetAction"]] == 1
    assert dataset["manifestData"]["metadata"]["decisionKindCounts"]["bid"] == 1


def test_professional_renderer_is_default_replay_audit_capable(root: Path):
    renderer = RemotionProfessionalRenderer.from_yaml(root / "configs/video/premium.yaml")
    assert renderer.descriptor.id == "premium_video"
    assert "full_baloot_lifecycle" in renderer.descriptor.capabilities
    assert "historical_actual" in renderer.descriptor.capabilities
    assert "replay_audit" in renderer.descriptor.capabilities
    assert "render_bundle_v1" in renderer.descriptor.capabilities
