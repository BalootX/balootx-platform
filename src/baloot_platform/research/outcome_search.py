"""Deal-only neighbourhoods and target-distance search; never an admission gate."""
from __future__ import annotations

import copy
import math
import random
from collections import deque

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Rank, Team


SEARCH_PROFILE = dict(id='outcome_deal_search_v2', neighboursPerParent=8,
    stepsPerProposal=16, initialTemperature=8, minimumDeckDistance=4)


def deck_distance(first, second):
    return sum(a != b for a, b in zip(first, second, strict=True))


def critical_cards(compiled, state):
    named = compiled.source.get('outcome', {}).get('decidingTrick', {})
    critical = {Card.parse(named[key]).id for key in ('winningCard', 'beatenCard') if key in named}
    index = named.get('index', compiled.source.get('outcome', {}).get('swing', {}).get('trickIndex', 8))
    critical.update(play.card.id for play in state.completed_tricks[index-1].plays)
    critical.update(card.id for project in state.projects for card in project.cards)
    if state.trump_suit is not None:
        critical.update(card.id for card in (Card.of(state.trump_suit, Rank.KING),
            Card.of(state.trump_suit, Rank.QUEEN)))
    # The public purchase card fixes the auction's first-round trump options.
    critical.add(state.purchase_card.id)
    return frozenset(critical)


def target_distance(compiled, state, record, accepted):
    if accepted:
        return 0
    outcome = compiled.source.get('outcome', {})
    named = outcome.get('decidingTrick', {})
    target = named.get('index', 8)-1
    distance = 0
    for key in ('winningCard', 'beatenCard'):
        if key not in named:
            continue
        card = Card.parse(named[key])
        trick_index, play = next((i, p) for i, trick in enumerate(state.completed_tricks)
            for p in trick.plays if p.card == card)
        distance += 8*abs(trick_index-target)
        wanted_team = named.get('winningTeam', compiled.source.get('team', outcome.get('winner', 'north_south')))
        side = int(Team[wanted_team.upper()])
        distance += 40*int(int(play.seat.team) != (side if key == 'winningCard' else 1-side))
        if key == 'winningCard':
            distance += 30*int(state.completed_tricks[target].winner != play.seat)
    distance += 20*len(record['failedPredicates'])
    for row in record['trajectory']:
        distance += max(0, row['minimum']-row['value'], row['value']-row['maximum'])
    for side, score in outcome.get('score', {}).items():
        distance += abs(state.round_score.awarded_points[int(Team[side.upper()])]-score)
    bounds = outcome.get('awardedMargin', {})
    team = int(Team[compiled.source.get('team', outcome.get('winner', 'north_south')).upper()])
    margin = state.round_score.awarded_points[team]-state.round_score.awarded_points[1-team]
    distance += max(0, bounds.get('minimum', margin)-margin, margin-bounds.get('maximum', margin))
    return distance


def perturb(context, rng, protected=(), *, preferred=()):
    """Two cross-hand swaps give four changed slots without moving protected cards."""
    owners, deck = context['cardOwners'], context['deckOrder']
    mutable = [card for card in deck if card not in protected and card != context['purchaseCard']]
    pairs = [(a, b) for i, a in enumerate(mutable) for b in mutable[i+1:] if owners[a] != owners[b]]
    rng.shuffle(pairs)
    # Same-suit swaps are a cheap first preference, not an equivalence proof.
    pairs.sort(key=lambda pair: Card(pair[0]).suit != Card(pair[1]).suit)
    preferred = [pair for pair in preferred if all(c in mutable for c in pair) and owners[pair[0]] != owners[pair[1]]]
    choices = [*preferred, *pairs]
    for first in choices:
        second = next((pair for pair in pairs if not set(pair) & set(first)), None)
        if second is None:
            continue
        result = copy.deepcopy(context)
        swaps = [first, second]
        for a, b in swaps:
            ia, ib = result['deckOrder'].index(a), result['deckOrder'].index(b)
            result['deckOrder'][ia], result['deckOrder'][ib] = b, a
            result['cardOwners'][a], result['cardOwners'][b] = result['cardOwners'][b], result['cardOwners'][a]
        return result, [[Card(a).code, Card(b).code] for a, b in swaps]
    return None


class OutcomeSearch:
    def __init__(self, compiled, seed):
        self.compiled, self.rng = compiled, random.Random(seed)
        self.parents, self.neighbourhoods = deque(), {}
        self.base, self.distance, self.steps, self.preferred, self.protected = None, math.inf, 0, (), ()

    def next(self, fresh):
        while self.parents:
            parent = self.parents.popleft()
            stats = self.neighbourhoods[parent['id']]
            if stats['tried'] >= SEARCH_PROFILE['neighboursPerParent']:
                continue
            changed = perturb(parent['context'], self.rng, parent['critical'])
            if changed is None:
                stats['unavailable'] = 'critical_cards_leave_no_two_cross_hand_swaps'
                continue
            stats['tried'] += 1
            self.parents.append(parent)
            context, swaps = changed
            return context, dict(kind='neighbour', parent=parent['id'], swaps=swaps,
                criticalCards=[Card(c).code for c in sorted(parent['critical'])])
        if self.base is not None and self.steps < SEARCH_PROFILE['stepsPerProposal']:
            self.steps += 1
            preferred = list(self.preferred)
            self.rng.shuffle(preferred)
            changed = perturb(self.base, self.rng, self.protected, preferred=preferred)
            if changed is not None:
                context, swaps = changed
                return context, dict(kind='guided', step=self.steps, swaps=swaps, parentDistance=self.distance)
        self.base, self.distance, self.steps, self.preferred, self.protected = None, math.inf, 0, (), ()
        return fresh(), dict(kind='cp_sat')

    def observe(self, context, state, record, accepted, provenance):
        distance = target_distance(self.compiled, state, record, accepted)
        provenance['targetDistance'] = distance
        if provenance['kind'] == 'neighbour':
            self.neighbourhoods[provenance['parent']]['accepted'] += int(accepted)
            return
        temperature = SEARCH_PROFILE['initialTemperature']/(1+self.steps)
        if self.base is None or distance <= self.distance or self.rng.random() < math.exp(min(0, (self.distance-distance)/temperature)):
            self.base, self.distance = context, distance
            named = self.compiled.source.get('outcome', {}).get('decidingTrick', {})
            target = state.completed_tricks[named.get('index', 8)-1]
            preferred, protected = [], []
            wanted_team = named.get('winningTeam', self.compiled.source.get('team',
                self.compiled.source.get('outcome', {}).get('winner', 'north_south')))
            side = int(Team[wanted_team.upper()])
            for key in ('winningCard', 'beatenCard'):
                if key in named:
                    card = Card.parse(named[key]).id
                    desired = side if key == 'winningCard' else 1-side
                    matching = next((p for p in target.plays if p.card.id == card and int(p.seat.team) == desired), None)
                    if matching is not None:
                        protected.append(card)
                    else:
                        preferred.extend((card, p.card.id) for p in target.plays
                            if p.card.id != card and int(p.seat.team) == desired)
            if 'winningCard' in named:
                card = Card.parse(named['winningCard'])
                winner = next(p for p in target.plays if p.seat == target.winner)
                if winner.card != card:
                    preferred.extend((winner.card.id, p.card.id) for trick in state.completed_tricks
                        if trick is not target for p in trick.plays if p.card.id not in protected)
            self.preferred = preferred
            self.protected = protected

    def admit(self, context, state, identity, provenance, *, seed=None):
        if provenance['kind'] == 'neighbour':
            self.neighbourhoods[provenance['parent']]['distinct'] += 1
        self.neighbourhoods[identity] = dict(tried=0, accepted=0, distinct=0,
            seed=seed, seedDeck=context['deckOrder'])
        self.parents.append(dict(id=identity, context=context, critical=critical_cards(self.compiled, state)))
        self.base, self.distance, self.steps = None, math.inf, 0
