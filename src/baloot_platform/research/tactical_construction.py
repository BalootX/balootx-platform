from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class TacticalPlayConstructionProfileV1:
    """Governed structural precursor used only to guide CP-SAT construction.

    The resulting witness is never tactical truth. Normal actor-safe replay,
    causal verification, and qualification remain mandatory.
    """

    profile_id: str
    revision: int
    enabled: bool
    maximum_payoff_tricks: int
    minimum_later_raw_points: int
    require_precursor_for_gated_tactics: bool
    sha256: str

    @classmethod
    def build(
        cls,
        payload: Mapping[str, Any],
    ) -> "TacticalPlayConstructionProfileV1":
        raw = dict(payload)
        authored = {
            "schemaVersion",
            "id",
            "revision",
            "enabled",
            "maximumPayoffTricks",
            "minimumLaterRawPoints",
            "requirePrecursorForGatedTactics",
        }
        derived = {"sha256"}
        if (
            not authored.issubset(raw)
            or set(raw) - authored - derived
            or int(raw.get("schemaVersion", 0)) != 1
        ):
            raise ValueError(
                "tactical play construction profile is incomplete or contains unknown keys"
            )
        profile_id = str(raw["id"]).strip()
        revision = int(raw["revision"])
        enabled = bool(raw["enabled"])
        maximum_payoff_tricks = int(raw["maximumPayoffTricks"])
        minimum_later_raw_points = int(raw["minimumLaterRawPoints"])
        require_precursor = bool(raw["requirePrecursorForGatedTactics"])
        if profile_id != "tactical_play_precursor_v1" or revision < 1:
            raise ValueError("unsupported tactical play construction profile")
        if not 1 <= maximum_payoff_tricks <= 4:
            raise ValueError("tactical payoff depth must be in [1, 4]")
        if not 1 <= minimum_later_raw_points <= 40:
            raise ValueError("tactical later raw points must be in [1, 40]")
        core = {
            "schemaVersion": 1,
            "id": profile_id,
            "revision": revision,
            "enabled": enabled,
            "maximumPayoffTricks": maximum_payoff_tricks,
            "minimumLaterRawPoints": minimum_later_raw_points,
            "requirePrecursorForGatedTactics": require_precursor,
        }
        digest = canonical_sha256(core)
        if raw.get("sha256") not in {None, "", digest}:
            raise ValueError("tactical play construction profile checksum mismatch")
        return cls(
            profile_id,
            revision,
            enabled,
            maximum_payoff_tricks,
            minimum_later_raw_points,
            require_precursor,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "enabled": self.enabled,
            "maximumPayoffTricks": self.maximum_payoff_tricks,
            "minimumLaterRawPoints": self.minimum_later_raw_points,
            "requirePrecursorForGatedTactics": (
                self.require_precursor_for_gated_tactics
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TacticalDealBalanceProfileV1:
    """Governed deal-shape constraints for unresolved tactical positions.

    This profile changes only CP-SAT construction. It is never tactical proof
    and cannot bypass replay, observer-safe analysis, or qualification.
    """

    profile_id: str
    revision: int
    enabled: bool
    maximum_team_high_card_difference: int
    minimum_high_cards_per_seat: int
    maximum_high_cards_per_seat: int
    minimum_target_low_cards: int
    sha256: str

    @classmethod
    def build(
        cls,
        payload: Mapping[str, Any],
    ) -> "TacticalDealBalanceProfileV1":
        raw = dict(payload)
        authored = {
            "schemaVersion",
            "id",
            "revision",
            "enabled",
            "maximumTeamHighCardDifference",
            "minimumHighCardsPerSeat",
            "maximumHighCardsPerSeat",
            "minimumTargetLowCards",
        }
        if (
            set(raw) - authored - {"sha256"}
            or not authored.issubset(raw)
            or int(raw.get("schemaVersion", 0)) != 1
        ):
            raise ValueError(
                "tactical deal balance profile is incomplete or contains unknown keys"
            )
        core = {
            "schemaVersion": 1,
            "id": str(raw["id"]).strip(),
            "revision": int(raw["revision"]),
            "enabled": bool(raw["enabled"]),
            "maximumTeamHighCardDifference": int(
                raw["maximumTeamHighCardDifference"]
            ),
            "minimumHighCardsPerSeat": int(raw["minimumHighCardsPerSeat"]),
            "maximumHighCardsPerSeat": int(raw["maximumHighCardsPerSeat"]),
            "minimumTargetLowCards": int(raw["minimumTargetLowCards"]),
        }
        if core["id"] != "balanced_tactical_deal_v1" or core["revision"] < 1:
            raise ValueError("unsupported tactical deal balance profile")
        if not 0 <= core["maximumTeamHighCardDifference"] <= 12:
            raise ValueError("team high-card difference must be in [0, 12]")
        if not 0 <= core["minimumHighCardsPerSeat"] <= 3:
            raise ValueError("minimum high cards per seat must be in [0, 3]")
        if not 3 <= core["maximumHighCardsPerSeat"] <= 8:
            raise ValueError("maximum high cards per seat must be in [3, 8]")
        if core["minimumHighCardsPerSeat"] > core["maximumHighCardsPerSeat"]:
            raise ValueError("high-card seat bounds are inverted")
        if not 0 <= core["minimumTargetLowCards"] <= 3:
            raise ValueError("minimum target low cards must be in [0, 3]")
        digest = canonical_sha256(core)
        if raw.get("sha256") not in {None, "", digest}:
            raise ValueError("tactical deal balance profile checksum mismatch")
        return cls(
            core["id"],
            core["revision"],
            core["enabled"],
            core["maximumTeamHighCardDifference"],
            core["minimumHighCardsPerSeat"],
            core["maximumHighCardsPerSeat"],
            core["minimumTargetLowCards"],
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "enabled": self.enabled,
            "maximumTeamHighCardDifference": (
                self.maximum_team_high_card_difference
            ),
            "minimumHighCardsPerSeat": self.minimum_high_cards_per_seat,
            "maximumHighCardsPerSeat": self.maximum_high_cards_per_seat,
            "minimumTargetLowCards": self.minimum_target_low_cards,
            "sha256": self.sha256,
        }
