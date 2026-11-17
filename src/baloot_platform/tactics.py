"""Stable tactical identities and frozen profiles shared by Research and Challenge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class TacticalKind(str, Enum):
    ONLY_WINNING_MOVE = "only_winning_move"
    SMALL_CARD_BIG_GAIN = "small_card_big_gain"
    LARGE_CARD_SACRIFICE = "large_card_sacrifice"
    KABOUT_CREATION = "kabout_creation"
    KABOUT_PRESERVATION = "kabout_preservation"
    KABOUT_PREVENTION = "kabout_prevention"
    ENTRY_UNBLOCKING = "entry_unblocking"
    TRUMP_CONTROL = "trump_control"
    CONTRACT_RESCUE = "contract_rescue"
    PARTNER_ENABLEMENT = "partner_enablement"
    PROJECT_BIDDING = "project_bidding"
    DEBATE = "debate"


DELAYED_PAYOFF_TACTIC_IDS = frozenset({
    TacticalKind.SMALL_CARD_BIG_GAIN.value, TacticalKind.LARGE_CARD_SACRIFICE.value,
})
NATIVE_PROFILE_TACTIC_IDS = DELAYED_PAYOFF_TACTIC_IDS | {
    TacticalKind.PROJECT_BIDDING.value, TacticalKind.TRUMP_CONTROL.value,
    TacticalKind.PARTNER_ENABLEMENT.value, TacticalKind.ENTRY_UNBLOCKING.value,
}


@dataclass(frozen=True, slots=True)
class TacticalDetectorProfileV1:
    profile_id: str
    revision: str
    low_card_max_strength: float
    high_card_min_strength: float
    minimum_delayed_gain: float
    minimum_causal_gain: float
    kabout_probability_delta: float
    contract_probability_delta: float
    project_consequence_delta: float
    debate_margin_gap: float
    debate_disagreement: float
    sha256: str
    native_delayed_gain_metric: str | None = None
    native_delayed_minimum_gain: float | None = None

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "TacticalDetectorProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "lowCardMaxStrength",
            "highCardMinStrength", "minimumDelayedGain", "minimumCausalGain",
            "kaboutProbabilityDelta", "contractProbabilityDelta",
            "projectConsequenceDelta", "debateMarginGap", "debateDisagreement",
            "nativeDelayedPayoff",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown tactical detector profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported tactical detector profile schema")
        core = {
            "schemaVersion": 1,
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "lowCardMaxStrength": float(value.get("lowCardMaxStrength", -1)),
            "highCardMinStrength": float(value.get("highCardMinStrength", -1)),
            "minimumDelayedGain": float(value.get("minimumDelayedGain", -1)),
            "minimumCausalGain": float(value.get("minimumCausalGain", -1)),
            "kaboutProbabilityDelta": float(value.get("kaboutProbabilityDelta", -1)),
            "contractProbabilityDelta": float(value.get("contractProbabilityDelta", -1)),
            "projectConsequenceDelta": float(value.get("projectConsequenceDelta", -1)),
            "debateMarginGap": float(value.get("debateMarginGap", -1)),
            "debateDisagreement": float(value.get("debateDisagreement", -1)),
        }
        if not core["id"] or not core["revision"]:
            raise ValueError("tactical detector profile identity is required")
        if not 0 <= core["lowCardMaxStrength"] < core["highCardMinStrength"] <= 100:
            raise ValueError("tactical detector card-strength boundaries are invalid")
        for field in ("minimumDelayedGain", "minimumCausalGain", "projectConsequenceDelta", "debateMarginGap"):
            if not 0 <= core[field] <= 100:
                raise ValueError(f"tactical detector {field} is out of range")
        for field in ("kaboutProbabilityDelta", "contractProbabilityDelta", "debateDisagreement"):
            if not 0 <= core[field] <= 1:
                raise ValueError(f"tactical detector {field} is out of range")
        native_metric = native_gain = None
        if "nativeDelayedPayoff" in value:
            native = value["nativeDelayedPayoff"]
            if not isinstance(native, Mapping) or set(native) != {"gainMetric", "minimumGain"}:
                raise ValueError("native delayed payoff requires only gainMetric and minimumGain")
            native_metric = native["gainMetric"]
            native_gain = native["minimumGain"]
            if type(native_metric) is not str or native_metric not in {"raw_card_points", "awarded_point_margin"}:
                raise ValueError("native delayed payoff gainMetric is unsupported")
            if type(native_gain) not in (int, float) or not 0 <= native_gain <= 100:
                raise ValueError("native delayed payoff minimumGain must be a finite number in [0, 100]")
            native_gain = float(native_gain)
            core["nativeDelayedPayoff"] = {"gainMetric": native_metric, "minimumGain": native_gain}
        return cls(
            core["id"], core["revision"], core["lowCardMaxStrength"],
            core["highCardMinStrength"], core["minimumDelayedGain"],
            core["minimumCausalGain"], core["kaboutProbabilityDelta"],
            core["contractProbabilityDelta"], core["projectConsequenceDelta"],
            core["debateMarginGap"], core["debateDisagreement"],
            hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            native_metric, native_gain,
        )
    @classmethod
    def defaults(cls) -> "TacticalDetectorProfileV1":
        return cls.build({
            "schemaVersion": 1,
            "id": "causal_tactical_detector_default_v1",
            "revision": "1",
            "lowCardMaxStrength": 35.0,
            "highCardMinStrength": 70.0,
            "minimumDelayedGain": 8.0,
            "minimumCausalGain": 0.01,
            "kaboutProbabilityDelta": 0.20,
            "contractProbabilityDelta": 0.20,
            "projectConsequenceDelta": 5.0,
            "debateMarginGap": 2.0,
            "debateDisagreement": 0.25,
        })

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "lowCardMaxStrength": self.low_card_max_strength,
            "highCardMinStrength": self.high_card_min_strength,
            "minimumDelayedGain": self.minimum_delayed_gain,
            "minimumCausalGain": self.minimum_causal_gain,
            "kaboutProbabilityDelta": self.kabout_probability_delta,
            "contractProbabilityDelta": self.contract_probability_delta,
            "projectConsequenceDelta": self.project_consequence_delta,
            "debateMarginGap": self.debate_margin_gap,
            "debateDisagreement": self.debate_disagreement,
            "sha256": self.sha256,
        }
        if self.native_delayed_gain_metric is not None or self.native_delayed_minimum_gain is not None:
            result["nativeDelayedPayoff"] = {
                "gainMetric": self.native_delayed_gain_metric,
                "minimumGain": self.native_delayed_minimum_gain,
            }
        return result

    def validate(self) -> None:
        raw = self.to_dict()
        raw.pop("sha256")
        if self != self.build(raw):
            raise ValueError("tactical detector profile content or hash mismatch")


TACTICAL_LOCALIZATION_KEYS = {
    TacticalKind.ONLY_WINNING_MOVE.value: "research_tactic_only_move",
    TacticalKind.SMALL_CARD_BIG_GAIN.value: "research_tactic_small_gain",
    TacticalKind.LARGE_CARD_SACRIFICE.value: "research_tactic_sacrifice",
    TacticalKind.KABOUT_CREATION.value: "research_tactic_kabout",
    TacticalKind.KABOUT_PRESERVATION.value: "research_tactic_kabout_preservation",
    TacticalKind.KABOUT_PREVENTION.value: "research_tactic_kabout_prevention",
    TacticalKind.ENTRY_UNBLOCKING.value: "research_tactic_entry",
    TacticalKind.TRUMP_CONTROL.value: "research_tactic_trump",
    TacticalKind.CONTRACT_RESCUE.value: "research_tactic_contract_rescue",
    TacticalKind.PARTNER_ENABLEMENT.value: "research_tactic_partner",
    TacticalKind.PROJECT_BIDDING.value: "research_tactic_project",
    TacticalKind.DEBATE.value: "research_tactic_debate",
}
