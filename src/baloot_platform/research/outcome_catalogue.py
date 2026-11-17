"""Deterministic game-fact catalogue, CSV and summaries; no LLM or rendering."""
import csv
import json
from pathlib import Path

from baloot_platform.core.enums import Team
from baloot_platform.core.cards import Card
from .scenario_contracts import canonical_sha256


def game_facts(state, last_bonus, buyer_tie_outcome=None):
    swings, running, leads = [], 0, []
    for i, trick in enumerate(state.completed_tricks):
        points = trick.raw_points+(last_bonus if i == 7 else 0)
        swings.append(points*(1 if int(trick.winner.team) == 0 else -1))
        running += swings[-1]
        leads.append(1 if running > 0 else -1 if running < 0 else 0)
    previous = next((v for v in reversed(leads[:-1]) if v), 0)
    last_reversal = leads[-1]*previous < 0
    wins = [sum(int(t.winner.team) == team for t in state.completed_tricks) for team in (0, 1)]
    projects = [dict(owner=p.owner.name.lower(), kind=p.kind, value=p.value,
        declared=p.declared, scoring=p.scoring, revealed=p.revealed) for p in state.projects]
    largest = max(range(8), key=lambda i: (abs(swings[i]), -i))
    return dict(mode=state.contract.value, contract=state.contract.value,
        trumpSuit=state.trump_suit.name.lower() if state.trump_suit is not None else None,
        buyer=state.buyer.name.lower(), winner=state.round_score.winner.name.lower(),
        sunTieRule=buyer_tie_outcome or 'legacy_raw_units',
        finalScore=list(state.round_score.awarded_points), rawCardPoints=list(state.round_score.raw_card_points),
        balootTeams=[team.name.lower() for team in sorted(state.baloot_teams, key=int)], swingProfile=swings,
        biggestScoreSwing=abs(swings[largest]), biggestSwingTrick=largest+1,
        lastTrickReversal=last_reversal, nearKaboot=max(wins) == 7, tricksWon=wins,
        projects=projects, declarations=[p for p in projects if p['declared']],
        tricks=[dict(index=i+1, winner=t.winner.name.lower(), team=t.winner.team.name.lower(),
            rawPoints=t.raw_points, plays=[dict(actor=p.seat.name.lower(), card=p.card.code) for p in t.plays])
            for i, t in enumerate(state.completed_tricks)])


def story_granularity(value=None):
    settings = dict(scoreBucket=10, swingBucket=10, openingTricks=3)
    if value is not None:
        if not isinstance(value, dict) or set(value)-set(settings):
            raise ValueError('unsupported story granularity')
        settings.update(value)
    if any(type(v) is not int for v in settings.values()) or settings['scoreBucket'] < 1 or settings['swingBucket'] < 1 or not 0 <= settings['openingTricks'] <= 3:
        raise ValueError('story buckets must be positive integers; openingTricks must be 0..3')
    return settings


def story_facts(facts, compiled, granularity=None):
    granularity = story_granularity(granularity)
    source = compiled.source
    outcome = source.get('outcome', {})
    named = outcome.get('decidingTrick', {})
    key = compiled.source.get('outcome', {}).get('decidingTrick', {}).get('index', facts['biggestSwingTrick'])
    trick = facts['tricks'][key-1]
    winning = next(p['card'] for p in trick['plays'] if p['actor'] == trick['winner'])
    opponents = sorted(p['card'] for p in trick['plays'] if
        (trick['team'] == 'north_south') != (p['actor'] in {'north', 'south'}))
    signature = dict(policy='outcome_story_v2', granularity=granularity,
        mode=facts['mode'], contract=dict(mode=facts['contract'], trump=facts['trumpSuit']),
        buyer=facts['buyer'], winner=facts['winner'],
        scoreBucket={team: facts['finalScore'][i]//granularity['scoreBucket']*granularity['scoreBucket']
            for i, team in enumerate(('north_south', 'west_east')) if team not in outcome.get('score', {})},
        keyTrick=key, winningCard=winning, beatenCards=opponents,
        swingBucket=facts['biggestScoreSwing']//granularity['swingBucket']*granularity['swingBucket'],
        earlierSwingTrick=(max(range(key-1), key=lambda i: (abs(facts['swingProfile'][i]), -i))+1 if key > 1 else None),
        declarationKinds=sorted({p['kind'] for p in facts['declarations']}),
        baloot=bool(facts.get('balootTeams')), trick7Winner=facts['tricks'][6]['winner'],
        openingLedSuits=[Card.parse(t['plays'][0]['card']).suit.name.lower()
            for t in facts['tricks'][:granularity['openingTricks']]])
    if 'mode' in source or 'trumpSuit' in source:
        signature.pop('mode')
        signature['contract'].pop('mode')
    if 'trumpSuit' in source or facts['trumpSuit'] is None:
        signature['contract'].pop('trump')
    if 'buyer' in source or 'auction' in source:
        signature.pop('buyer')
    if 'auction' in source:
        signature.pop('mode', None)
        signature.pop('contract')
    if 'winner' in outcome or 'comeback' in outcome or source.get('category') in {'comeback', 'last_trick_reversal', 'kaboot'}:
        signature.pop('winner')
    if 'index' in named:
        signature.pop('keyTrick')
    if 'winningCard' in named:
        signature.pop('winningCard')
    if 'beatenCard' in named:
        signature['beatenCards'].remove(Card.parse(named['beatenCard']).code)
    if named.get('index') == 7 and 'winningSeat' in named:
        signature.pop('trick7Winner')
    if any(p['category'] == 'baloot' for p in source.get('declarations', [])):
        signature.pop('baloot')
    upset = max([0]+[int(Card.parse(c).rank)-int(Card.parse(winning).rank) for c in opponents])
    components = dict(swing=facts['biggestScoreSwing'],
        closeness=max(0, 40-abs(facts['finalScore'][0]-facts['finalScore'][1])),
        lateness=key, rankUpset=upset, declarations=len(facts['declarations']),
        baloot=len(facts.get('balootTeams', [])))
    score = (components['swing']+components['closeness']*2+key*3+upset*10
        +components['declarations']*10+components['baloot']*20)
    return dict(storySignature=signature, storyHash=canonical_sha256(signature),
        interestScore=score, interestComponents=components)


def catalogue_rows(accepted, compiled, granularity=None):
    rows = []
    for item in accepted:
        record = json.loads(Path(item['path']).read_text(encoding='utf-8'))
        facts = record['facts']
        key = compiled.source.get('outcome', {}).get('decidingTrick', {}).get('index', facts['biggestSwingTrick'])
        trick = facts['tricks'][key-1]
        winning = next(p['card'] for p in trick['plays'] if p['actor'] == trick['winner'])
        opponents = ', '.join(p['card'] for p in trick['plays'] if
            (int(Team[trick['team'].upper()]) == 0) != (p['actor'] in {'north', 'south'}))
        policies = dict(compiled.seat_policies)
        winner = facts['winner'].replace('_', '-').title()
        tie = ''
        if facts['mode'] in {'sun', 'ashkal'} and facts['finalScore'][0] == facts['finalScore'][1]:
            tie = f"; Sun tie rule {facts.get('sunTieRule') or 'legacy_raw_units'} (buyer {facts['buyer']})"
        bonus = ' (including the trick-8 bonus)' if facts['biggestSwingTrick'] == 8 else ' (no last-trick bonus)'
        gaining_team = facts['tricks'][facts['biggestSwingTrick']-1]['team'].replace('_', '-').title()
        units = 'trick points' if facts['biggestSwingTrick'] == 8 else 'raw card points'
        summary = (f"{facts['mode'].title()}: round winner {winner}, score {facts['finalScore'][0]}-{facts['finalScore'][1]} "
            f"(NS-WE){tie}, under the recorded {', '.join(sorted(set(policies.values())))} card policies. "
            f"Trick {key}: {winning} wins over the opponents' {opponents}. Largest gain: "
            f"{gaining_team} gains {facts['biggestScoreSwing']} {units} on trick {facts['biggestSwingTrick']}{bonus}.")
        ranked_facts = {**facts, **story_facts(facts, compiled, granularity)}
        rows.append(dict(id=item['id'], seed=item['seed'], specHash=compiled.sha256,
            spec=dict(compiled.source), policies=policies, **ranked_facts, keyMomentTrickIndex=key,
            keyTrick=trick, distinctnessHash=item['diversityKey'], receipt=item['path'],
            receiptSha256=item['sha256'], summary=summary))
    rows.sort(key=lambda row: (-row['interestScore'], row['distinctnessHash'], row['id']))
    return [dict(row, rank=i+1) for i, row in enumerate(rows)]


def write_exports(output, rows):
    fields = ['rank', 'id', 'seed', 'specHash', 'spec', 'mode', 'contract', 'policies', 'finalScore',
        'interestScore', 'interestComponents', 'keyMomentTrickIndex', 'keyTrick',
        'biggestScoreSwing', 'biggestSwingTrick', 'declarations', 'lastTrickReversal',
        'nearKaboot', 'storySignature', 'storyHash', 'distinctnessHash', 'receipt', 'receiptSha256', 'summary']
    with (output/'catalogue.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], (dict, list))
                else row[key] for key in fields})
    text = '# Verified configured outcomes\n\nNot independently audited lessons; informed policies are sampled, not exact.\n\n'
    text += '\n\n'.join(f"{r['rank']}. {r['id']}: {r['summary']}" for r in rows[:50])+'\n'
    (output/'summaries.md').write_text(text, encoding='utf-8')


def write_neighbourhoods(output, workers, retained=None):
    from collections import Counter
    retained_children = Counter(row.get('parent') for row in retained or [])
    totals = dict(tried=0, accepted=0, distinct=0, shardDistinct=0)
    with (output/'neighbourhoods.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=['parent', 'seed', 'variantsTried',
            'outcomeVerified', 'shardDistinct', 'distinctAdmitted', 'unavailable'])
        writer.writeheader()
        for worker in sorted(workers, key=lambda w: w['worker']):
            for parent, stats in sorted(worker['neighbourhoods'].items()):
                distinct = stats['distinct'] if retained is None else retained_children[parent]
                totals['tried'] += stats['tried']
                totals['accepted'] += stats['accepted']
                totals['shardDistinct'] += stats['distinct']
                totals['distinct'] += distinct
                writer.writerow(dict(parent=parent, seed=stats['seed'], variantsTried=stats['tried'],
                    outcomeVerified=stats['accepted'], shardDistinct=stats['distinct'], distinctAdmitted=distinct,
                    unavailable=stats.get('unavailable', '')))
    return totals
