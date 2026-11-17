from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from .cards import Card, FULL_DECK
from .enums import Seat
from .structural_ir import StructuralDomainIRV1

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = int(seed) & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def randbelow(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            value = self.next()
            if value < limit:
                return value % n


@dataclass(frozen=True, slots=True)
class ShuffleRecord:
    algorithm: str
    seed: int
    dealer: Seat
    cutter: Seat | None
    cut_position: int | None
    deck_commitment: str
    deck_order: tuple[int, ...]

    def to_dict(self, *, reveal_order: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "dealer": self.dealer.name.lower(),
            "cutter": self.cutter.name.lower() if self.cutter is not None else None,
            "cutPosition": self.cut_position,
            "deckCommitment": self.deck_commitment,
        }
        if reveal_order:
            result["deckOrder"] = list(self.deck_order)
        return result


@dataclass(frozen=True, slots=True)
class ShuffledDeck:
    cards: tuple[Card, ...]
    record: ShuffleRecord


def shuffle_and_cut(
    seed: int,
    cut_position: int | None,
    *,
    dealer: Seat,
    structural_domain: StructuralDomainIRV1 | None = None,
    skip_cut: bool = False,
    cut_enabled: bool | None = None,
) -> ShuffledDeck:
    """Shuffle and cut through the compiled structural domain.

    ``cut_enabled`` remains only for the fixed release-manifest verifier. The
    game engine must pass ``structural_domain``; providing both authorities is
    rejected so a legacy flag cannot override a published rule bundle.
    """
    if structural_domain is not None and not isinstance(
        structural_domain, StructuralDomainIRV1
    ):
        raise TypeError("structural_domain must be StructuralDomainIRV1")
    if structural_domain is not None and cut_enabled is not None:
        raise TypeError("cut_enabled cannot override StructuralDomainIRV1")
    if structural_domain is not None:
        structural_domain.require_integrity()
        if structural_domain.shuffle_algorithm != "splitmix64_fisher_yates_v1":
            raise ValueError(
                f"unsupported shuffle algorithm: {structural_domain.shuffle_algorithm}"
            )
        if skip_cut and not structural_domain.allow_no_cut:
            raise ValueError("the selected structural domain requires a cut")
        apply_cut = structural_domain.cut_enabled and not skip_cut
        governed_ids = structural_domain.card_ids
        algorithm = structural_domain.shuffle_algorithm
    else:
        if skip_cut:
            raise TypeError("skip_cut requires StructuralDomainIRV1")
        apply_cut = True if cut_enabled is None else bool(cut_enabled)
        governed_ids = tuple(card.id for card in FULL_DECK)
        algorithm = "splitmix64_fisher_yates_v1"

    rng = SplitMix64(seed)
    cards = list(FULL_DECK)
    if tuple(card.id for card in cards) != governed_ids:
        raise ValueError("runtime deck does not match the structural domain")
    for i in range(len(cards) - 1, 0, -1):
        j = rng.randbelow(i + 1)
        cards[i], cards[j] = cards[j], cards[i]
    cutter = (
        structural_domain.cutter_for(dealer)
        if structural_domain is not None and apply_cut
        else dealer.previous() if apply_cut else None
    )
    if apply_cut:
        if cut_position is None:
            cut_position = 5 + rng.randbelow(23)
        cut_position = (
            structural_domain.validate_cut_position(cut_position)
            if structural_domain is not None
            else int(cut_position)
        )
        if structural_domain is None and not 1 <= cut_position < len(cards):
            raise ValueError(
                f"cut position must be between 1 and {len(cards) - 1}"
            )
        cards = cards[cut_position:] + cards[:cut_position]
    else:
        cut_position = None
    encoded = b"".join(struct.pack("B", c.id) for c in cards)
    commitment = hashlib.sha256(encoded).hexdigest()
    record = ShuffleRecord(
        algorithm,
        int(seed), dealer, cutter, cut_position, commitment, tuple(c.id for c in cards)
    )
    return ShuffledDeck(tuple(cards), record)


def governed_preordered_deck(
    deck_order: tuple[int, ...] | list[int],
    *,
    seed: int,
    dealer: Seat,
    structural_domain: StructuralDomainIRV1,
) -> ShuffledDeck:
    """Validate a complete post-cut deck proposed by a governed generator.

    This boundary records constructed-deal provenance without granting the
    generator any access to bidding, legality, play, or scoring behavior.
    """
    if not isinstance(structural_domain, StructuralDomainIRV1):
        raise TypeError("governed deck construction requires StructuralDomainIRV1")
    structural_domain.require_integrity()
    ids = structural_domain.validate_deck_order(deck_order)
    cards = tuple(Card(card_id) for card_id in ids)
    encoded = b"".join(struct.pack("B", card.id) for card in cards)
    record = ShuffleRecord(
        "governed_preordered_v1",
        int(seed),
        dealer,
        None,
        None,
        hashlib.sha256(encoded).hexdigest(),
        ids,
    )
    return ShuffledDeck(cards, record)
