"""Independent agent reanalysis under seat, team, and suit symmetries."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from baloot_platform.agents.profiles import AdaptivePlayStrategy, AgentProfile
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.benchmark import canonical_sha256, reconstruct_state
from baloot_platform.core.cards import Card, cards_to_mask, mask_to_cards
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Suit, Team
from baloot_platform.core.models import BidRecord, PlayedCard, Project, Trick


def _rotate_seat(seat: Seat | None, turns: int) -> Seat | None:
    return None if seat is None else Seat((int(seat) + turns) % 4)


def _rotate_team(team: Team | None, turns: int) -> Team | None:
    return None if team is None or turns % 2 == 0 else team.other


def _map_mask(mask: int, card_map: Callable[[Card], Card]) -> int:
    return cards_to_mask(card_map(card) for card in mask_to_cards(mask))


def _map_trick(
    trick: Trick | None,
    seat_map: Callable[[Seat | None], Seat | None],
    card_map: Callable[[Card], Card],
) -> Trick | None:
    if trick is None:
        return None
    mapped = Trick(
        index=trick.index,
        leader=seat_map(trick.leader),  # type: ignore[arg-type]
        plays=[
            PlayedCard(
                seat=seat_map(play.seat),  # type: ignore[arg-type]
                card=card_map(play.card),
                trick_index=play.trick_index,
                position=play.position,
                legal_cards=tuple(card_map(card) for card in play.legal_cards),
                ekka=play.ekka,
                baloot=play.baloot,
            )
            for play in trick.plays
        ],
        winner=seat_map(trick.winner),
        raw_points=trick.raw_points,
    )
    return mapped


def transform_state(
    state,
    *,
    quarter_turns: int = 0,
    suit_mapping: Mapping[Suit, Suit] | None = None,
):
    """Return a complete, conservation-valid transformed state."""

    if quarter_turns not in range(4):
        raise ValueError("quarter turns must be between zero and three")
    suits = dict(suit_mapping or {suit: suit for suit in Suit})
    if set(suits) != set(Suit) or set(suits.values()) != set(Suit):
        raise ValueError("suit mapping must be a bijection")
    seat_map = lambda seat: _rotate_seat(seat, quarter_turns)
    card_map = lambda card: Card.of(suits[card.suit], card.rank)
    result = state.clone()
    result.dealer = seat_map(state.dealer)  # type: ignore[assignment]
    result.match_score = (
        list(reversed(state.match_score))
        if quarter_turns % 2
        else list(state.match_score)
    )
    for field in ("initial_five", "initial_hands", "hands"):
        source = list(getattr(state, field))
        target = [0, 0, 0, 0]
        for seat in Seat:
            mapped = seat_map(seat)
            target[int(mapped)] = _map_mask(source[int(seat)], card_map)
        setattr(result, field, target)
    result.purchase_card = card_map(state.purchase_card) if state.purchase_card else None
    result.undealt = tuple(card_map(card) for card in state.undealt)
    result.bid_history = [
        BidRecord(
            item.round_index,
            item.turn_index,
            seat_map(item.seat),  # type: ignore[arg-type]
            item.action,
            item.contract,
            suits[item.trump] if item.trump is not None else None,
            seat_map(item.bidder),
            seat_map(item.buyer),
        )
        for item in state.bid_history
    ]
    result.buyer = seat_map(state.buyer)
    result.declarer = seat_map(state.declarer)
    result.trump_suit = suits[state.trump_suit] if state.trump_suit is not None else None
    result.double_seat = seat_map(state.double_seat)
    result.first_leader = seat_map(state.first_leader)
    result.current_player = seat_map(state.current_player)
    result.current_trick = _map_trick(state.current_trick, seat_map, card_map)
    result.completed_tricks = [
        _map_trick(trick, seat_map, card_map) for trick in state.completed_tricks
    ]
    result.projects = [
        Project(
            owner=seat_map(item.owner),  # type: ignore[arg-type]
            kind=item.kind,
            cards=tuple(card_map(card) for card in item.cards),
            value=item.value,
            high_rank=item.high_rank,
            category_priority=item.category_priority,
            declared=item.declared,
            declaration_order=item.declaration_order,
            scoring=item.scoring,
            revealed=item.revealed,
        )
        for item in state.projects
    ]
    result.project_winning_team = _rotate_team(state.project_winning_team, quarter_turns)
    result.baloot_teams = {
        _rotate_team(team, quarter_turns) for team in state.baloot_teams
    }
    result.annulled_by = seat_map(state.annulled_by)
    # Shuffle/deal/event provenance belongs to the source trace, not the derived
    # qualification state. The transformed state remains complete and auditable
    # through its source state hash and transformation record.
    result.shuffle = None
    result.deal_trace = []
    result.events = []
    result.validate_conservation()
    return result


def _ranked_codes(trace: Mapping[str, Any]) -> tuple[str, ...]:
    require_uncensored_play_trace(trace)
    return tuple(
        str(dict(item.get("action") or {}).get("code") or "")
        for item in trace.get("rankedActions") or ()
    )


def evaluate_agent_symmetry(
    *,
    engine: BalootEngine,
    profile: AgentProfile,
    convention: Mapping[str, Any],
    candidates: Mapping[str, Any],
    trusted: Mapping[str, Any],
    exact: Mapping[str, Any],
    maximum_cases: int,
    tolerance: float,
) -> dict[str, Any]:
    """Re-run exact-supported states under every governed transformation."""

    candidate_by_id = {
        str(item["candidateId"]): dict(item)
        for item in candidates.get("positions") or ()
    }
    recipes = dict(trusted.get("recipes") or {})
    labels = [
        dict(item) for item in exact.get("positions") or ()
        if str(dict(recipes.get(str(item.get("candidateId"))) or {}).get("stage"))
        == "card_play"
    ][:maximum_cases]
    if not labels:
        raise ValueError("symmetry qualification has no exact-supported cases")
    strategy = AdaptivePlayStrategy(profile, dict(convention))
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    maximum_delta = 0.0
    suit_cycle = {suit: Suit((int(suit) + 1) % 4) for suit in Suit}
    for label in labels:
        candidate_id = str(label["candidateId"])
        source = reconstruct_state(
            engine, candidate_by_id[candidate_id], dict(recipes[candidate_id])
        )
        seed = int(canonical_sha256({
            "candidateId": candidate_id,
            "profileId": profile.id,
            "purpose": "symmetry-qualification",
        })[:16], 16)
        selected, source_trace = strategy.choose_with_trace(engine, source, seed=seed)
        source_ranked = _ranked_codes(source_trace)
        variants = [
            ("seat_rotation", turns, transform_state(source, quarter_turns=turns), lambda card: card)
            for turns in (1, 2, 3)
        ]
        variants.append((
            "suit_relabel", 0,
            transform_state(source, suit_mapping=suit_cycle),
            lambda card: Card.of(suit_cycle[card.suit], card.rank),
        ))
        for transform_name, turns, transformed, map_card in variants:
            transformed_selected, transformed_trace = strategy.choose_with_trace(
                engine, transformed, seed=seed
            )
            expected = map_card(selected)
            expected_ranked = tuple(map_card(Card.parse(code)).code for code in source_ranked)
            actual_ranked = _ranked_codes(transformed_trace)
            confidence_delta = abs(
                float(source_trace.get("confidence", 0.0))
                - float(transformed_trace.get("confidence", 0.0))
            )
            maximum_delta = max(maximum_delta, confidence_delta)
            case_failures: list[str] = []
            if transformed_selected != expected:
                case_failures.append("selected_action")
            if actual_ranked != expected_ranked:
                case_failures.append("ranking")
            if confidence_delta > tolerance:
                case_failures.append("confidence")
            if transformed_trace.get("observerSafe") is not True:
                case_failures.append("observer_safety")
            failures.extend(
                f"{candidate_id}:{transform_name}:{turns}:{item}"
                for item in case_failures
            )
            records.append({
                "candidateId": candidate_id,
                "sourceStateHash": engine.state_hash(source),
                "transformedStateHash": engine.state_hash(transformed),
                "transform": transform_name,
                "quarterTurns": turns,
                "sourceAction": selected.code,
                "expectedAction": expected.code,
                "actualAction": transformed_selected.code,
                "confidenceDelta": confidence_delta,
                "passed": not case_failures,
            })
    transforms = ["seat_rotation", "team_swap", "suit_relabel"]
    value: dict[str, Any] = {
        "schemaVersion": 1,
        "profileId": profile.id,
        "profileHash": canonical_sha256(profile.to_dict()),
        "transforms": transforms,
        "sourceCaseCount": len(labels),
        "comparedDecisions": len(records),
        "maximumDelta": maximum_delta,
        "tolerance": tolerance,
        "passed": not failures,
        "failures": sorted(set(failures)),
        "recordsHash": canonical_sha256(records),
        "proofBoundary": (
            "Independent observer-safe reanalysis of exact-supported card-play "
            "states under cyclic seat rotation, team swap, and suit relabeling."
        ),
    }
    value["resultHash"] = canonical_sha256(value)
    return value
