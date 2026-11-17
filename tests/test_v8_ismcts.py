from __future__ import annotations

from baloot_platform.analysis.ismcts_v8 import analyze_information_set_mcts, information_set_key
from baloot_platform.analysis.models import InformationState
from baloot_platform.config import build_bundle
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.engine import BalootEngine


def _episode(root):
    bundle = build_bundle(root)
    request = ScenarioRequest.governed(
        seed=7351, viewer=Seat.SOUTH, contract=Contract.HOKUM,
        requested_samples=8, exact_allocation_limit=5000,
        pause_trick_min=5, pause_trick_max=6,
        require_project=False, max_seed_attempts=80,
        minimum_hidden_deals=1,
    )
    return bundle, generate_episode(bundle.rules, bundle.convention, bundle_hash=bundle.sha256, request=request)


def test_information_set_key_excludes_opponent_hands(root):
    bundle, episode = _episode(root)
    info = InformationState.from_complete(episode.state_before_pause, Seat.SOUTH)
    engine = BalootEngine(bundle.rules)
    from baloot_platform.analysis.belief_v7 import build_particle_belief
    belief = build_particle_belief(info, engine, particles=12, exact_allocation_limit=5000, seed=9)
    keys = {information_set_key(particle.state) for particle in belief.particles}
    assert len(keys) == 1


def test_information_set_mcts_is_legal_deterministic_and_strategy_shared(root):
    bundle, episode = _episode(root)
    info = InformationState.from_complete(episode.state_before_pause, Seat.SOUTH)
    engine = BalootEngine(bundle.rules)
    first = analyze_information_set_mcts(
        info, engine, particles=16, exact_allocation_limit=5000,
        iterations=180, maximum_tree_depth=12, seed=77,
    )
    second = analyze_information_set_mcts(
        info, engine, particles=16, exact_allocation_limit=5000,
        iterations=180, maximum_tree_depth=12, seed=77,
    )
    legal = {card.id for card in engine.legal_moves(episode.state_before_pause)}
    assert first.recommended_card.id in legal
    assert first.to_dict() == second.to_dict()
    assert first.method == 'information_set_mcts_v1'
    assert first.guarantee == 'shared_strategy_by_information_set_not_equilibrium'
    assert first.iterations == 180
    assert sum(item.visits for item in first.actions) == first.iterations
    assert first.common_cohort_size == min(
        len(first.belief.particles),
        first.iterations // len(first.actions),
    )
    assert all(
        item.visits >= first.common_cohort_size
        for item in first.actions
    )


def test_information_set_mcts_is_independent_of_hidden_hand_assignment(root):
    bundle, episode = _episode(root)
    changed = episode.state_before_pause.clone()
    hidden = [seat for seat in Seat if seat != Seat.SOUTH]
    first, second = next(
        (left, right)
        for index, left in enumerate(hidden)
        for right in hidden[index + 1:]
        if changed.hands[int(left)].bit_count()
        == changed.hands[int(right)].bit_count()
    )
    changed.hands[int(first)], changed.hands[int(second)] = (
        changed.hands[int(second)],
        changed.hands[int(first)],
    )
    original_info = InformationState.from_complete(
        episode.state_before_pause,
        Seat.SOUTH,
    )
    changed_info = InformationState.from_complete(changed, Seat.SOUTH)
    assert original_info.to_dict() == changed_info.to_dict()
    engine = BalootEngine(bundle.rules)
    settings = {
        "particles": 8,
        "exact_allocation_limit": 5000,
        "iterations": 80,
        "maximum_tree_depth": 8,
        "seed": 91,
    }
    original = analyze_information_set_mcts(
        original_info,
        engine,
        **settings,
    )
    altered = analyze_information_set_mcts(
        changed_info,
        engine,
        **settings,
    )
    assert original.to_dict() == altered.to_dict()
