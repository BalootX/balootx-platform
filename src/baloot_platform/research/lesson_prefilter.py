"""Cheap public-card lesson hypotheses, never qualification evidence."""
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit
from .contracts import canonical_sha256


CLASSIFIED_LESSON_FAMILIES_V1 = frozenset({"stopper_forced_return", "stranded_master", "overtake_unblock",
    "cash_into_partner", "ace_vs_ten", "partner_entry"})
LESSON_FAMILIES = CLASSIFIED_LESSON_FAMILIES_V1 | {"kaboot_stop", "negative_project_inference"}


def classify_public_lesson_v1(*, rules, actor, own_cards, completed_tricks, current_trick,
                             contract, trump, action, baseline):
    """Classify only directly witnessed patterns; unknown deductions stay unknown."""
    actor, contract = Seat.parse(actor), Contract.parse(contract)
    trump = Suit.parse(trump) if trump is not None else None
    own = set(map(Card.parse, own_cards))
    chosen, alternative = Card.parse(action), Card.parse(baseline)
    rows = current_trick.get("plays", [])
    history = [*completed_tricks, current_trick]
    played = {Card.parse(row["card"]) for trick in history for row in trick["plays"]}
    outstanding = set(map(Card, range(32))) - played - own
    family, deductions = None, []
    domain = rules.play_domain
    if len(own) in (2, 3):
        led = Card.parse(rows[0]["card"]).suit if rows else None
        winning_suit = trump if trump is not None and any(Card.parse(row['card']).suit == trump for row in rows) else led
        winner = max((row for row in rows if Card.parse(row["card"]).suit == winning_suit),
            key=lambda row: domain.strength(Card.parse(row["card"]), contract, trump), default=None)
        winner_card = Card.parse(winner["card"]) if winner else None
        partner = winner is not None and Seat.parse(winner.get("actor") or winner.get("seat")).team == actor.team
        strength = lambda card: domain.strength(card, contract, trump)
        master = lambda card: not any(other.suit == card.suit and strength(other) > strength(card)
                                      for other in outstanding)
        partner_seat = next(seat for seat in Seat if seat != actor and seat.team == actor.team)
        partner_void = {Card.parse(trick["plays"][0]["card"]).suit for trick in history
            if trick["plays"] and any(Seat.parse(row.get("actor") or row.get("seat")) == partner_seat
                and Card.parse(row["card"]).suit != Card.parse(trick["plays"][0]["card"]).suit
                for row in trick["plays"])}
        voids = {seat: set() for seat in Seat}
        counts = {seat: 8 for seat in Seat}
        for trick in history:
            if not trick['plays']:
                continue
            suit = Card.parse(trick['plays'][0]['card']).suit
            for row in trick['plays']:
                seat = Seat.parse(row.get('actor') or row.get('seat'))
                counts[seat] -= 1
                if Card.parse(row['card']).suit != suit:
                    voids[seat].add(suit)
        domains = {card: {seat for seat in Seat if seat != actor and counts[seat] > 0
                         and card.suit not in voids[seat]} for card in outstanding}
        changed = True
        while changed:
            before = {card: frozenset(seats) for card, seats in domains.items()}
            for seat in Seat:
                if seat == actor:
                    continue
                possible = [card for card, seats in domains.items() if seat in seats]
                fixed = [card for card in possible if domains[card] == {seat}]
                if len(possible) == counts[seat]:
                    for card in possible:
                        domains[card] = {seat}
                elif len(fixed) == counts[seat]:
                    for card in possible:
                        if card not in fixed:
                            domains[card].discard(seat)
            changed = before != {card: frozenset(seats) for card, seats in domains.items()}
        returns = {card for card, seats in domains.items() if partner_seat in seats}
        if (partner and chosen.suit == led == winning_suit and strength(chosen) > strength(winner_card)
                and alternative.suit == led and strength(alternative) < strength(winner_card)
                and (master(alternative) or any(master(card) for card in own - {chosen}))):
            family, deductions = "overtake_unblock", ["partner_current_winner", "remaining_own_card_established"]
        # A partner's long suit and an own loser distinguish entry timing from cashing winners.
        if (partner and chosen.suit == alternative.suit == led == winning_suit
                and strength(chosen) > strength(winner_card) > strength(alternative)
                and len(own) == 3
                and len([card for card, seats in domains.items() if card.suit == led and seats == {partner_seat}]) >= 2
                and any(card.suit != led and not master(card) for card in own)):
            family, deductions = "overtake_unblock", ["public_constraints_locate_remaining_led_suit",
                "partner_holds_remaining_long_suit", "overtake_preserves_return_entry",
                "partner_long_card_can_dispose_own_loser"]
        elif (partner and chosen.suit != led and chosen.suit == alternative.suit
                and (chosen.suit in partner_void or (contract.effective == Contract.SUN
                     and master(chosen) and master(winner_card)))
                and domain.points(chosen, contract, trump)
                > domain.points(alternative, contract, trump)):
            family, deductions = "cash_into_partner", ["partner_current_winner",
                "partner_cannot_return_own_suit" if chosen.suit in partner_void else "bank_master_in_secure_partner_trick"]
        elif (chosen.suit == alternative.suit == led == winning_suit and chosen.code.endswith('A')
                and any(card.suit == led and strength(alternative) < strength(card) < strength(chosen)
                        for card in outstanding)):
            family, deductions = "ace_vs_ten", ["outstanding_higher_honor", "ace_retains_control"]
        elif (chosen.suit != led and alternative.suit != led and master(alternative)
                and returns and not any(card.suit == chosen.suit for card in returns)
                and any(card.suit == alternative.suit for card in returns)):
            family, deductions = "stranded_master", ["public_voids_exclude_discarded_suit_return", "retained_master_has_possible_return"]
        elif (not rows and len(own) == 2 and trump is not None
                and alternative.suit == trump and chosen.suit != trump):
            family, deductions = "stopper_forced_return", ["lead_side_suit_before_last_trump", "retain_trump_for_final_trick"]
        elif (rows and chosen.suit == alternative.suit == led == winning_suit
                and strength(chosen) < strength(winner_card) < strength(alternative)
                and master(alternative) and len(own) == 2):
            family, deductions = "stopper_forced_return", ["duck_current_trick", "retain_master_for_final_return"]
        elif (rows and chosen.suit != led and alternative.suit != led
                and chosen.suit == alternative.suit and strength(chosen) < strength(alternative)):
            family, deductions = "stranded_master", ["discard_lower_card_of_same_suit", "retain_higher_card"]
        elif (rows and chosen.suit != led and alternative.suit != led
                and not any(card.suit == chosen.suit for card in outstanding)
                and any(card.suit == alternative.suit for card in outstanding)):
            family, deductions = "stranded_master", ["discard_exhausted_suit", "retain_live_suit_control"]
        elif (rows and chosen.suit != led and alternative.suit != led
                and not master(chosen) and master(alternative)):
            family, deductions = "stranded_master", ["discard_nonmaster", "retain_established_winner"]
        elif not rows and trump is None:
            higher = [card for card in outstanding if card.suit == chosen.suit and strength(card) > strength(chosen)]
            if higher and all(domains[card] == {partner_seat} for card in higher):
                family, deductions = "partner_entry", ["public_voids_locate_partner_control", "lead_to_partner_control"]
    # Fingerprint the witnessed inference/plan, never concrete cards or hidden allocations.
    if family is not None and family not in CLASSIFIED_LESSON_FAMILIES_V1:
        raise ValueError("public lesson classifier has no registered producer capability")
    causal = {"schemaVersion": 1, "family": family, "deductions": deductions} if family else None
    return {"schemaVersion": 1, "family": family, "deductions": deductions,
            "causalFingerprint": canonical_sha256(causal) if causal else None,
            "classification": "heuristic" if family else "unknown", "qualificationEvidence": False}


def lesson_is_repeated_v1(hypothesis, accepted_library):
    """Compare reviewed causal hashes, retaining explicit legacy count-library behavior."""
    if not isinstance(accepted_library, dict) or any(
            key not in LESSON_FAMILIES or not (
                type(value) is int and value >= 0 or
                isinstance(value, list) and all(
                    isinstance(item, str) and len(item) == 64
                    and all(char in "0123456789abcdef" for char in item) for item in value)
                and len(set(value)) == len(value))
            for key, value in accepted_library.items()):
        raise ValueError("accepted lesson library requires known families and causal fingerprints or legacy counts")
    family = hypothesis["family"]
    reviewed = accepted_library.get(family)
    if type(reviewed) is int:
        return len(hypothesis["deductions"]) <= reviewed
    fingerprint = hypothesis.get("causalFingerprint")
    return isinstance(reviewed, list) and fingerprint is not None and fingerprint in reviewed
