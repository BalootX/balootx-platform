from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .cards import Card
from .enums import Contract, Suit
from .play_ir import PlayDomainIRV1
from .project_ir import ProjectDomainIRV1
from .structural_ir import StructuralDomainIRV1


def resolved_disputed_options(rules: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expose disputed choices without creating a second selection authority."""

    metadata = rules["disputed_options"]
    selected = {
        "first_leader": rules["play"]["first_leader"],
        "double_unspecified_mode": rules["escalation"]["default_mode"],
        "equal_hokum_sequence_trump_priority": metadata[
            "equal_hokum_sequence_trump_priority"
        ]["selected"],
        # The supported convention model treats hints as probabilistic evidence,
        # never universally binding facts.
        "hints_have_universal_meaning": False,
        "kawesh_saneen": bool(rules["dealing"]["annulling"]["enabled"]),
    }
    return {
        option_id: {
            "selected": value,
            "alternatives": list(metadata[option_id]["alternatives"]),
        }
        for option_id, value in selected.items()
    }


def _merge(base: Any, overlay: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Merge a mode overlay without silently accepting misspelled rule keys.

    Lists and scalar values replace their base value. Mapping keys must already
    exist in the authoritative rule profile; new capabilities require an
    explicit base-profile and schema change in the same release.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        unknown = sorted(set(overlay) - set(base))
        if unknown:
            location = ".".join(path) or "<root>"
            raise ValueError(f"unknown rule overlay keys at {location}: {unknown}")
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value, path=(*path, str(key)))
        return result
    return copy.deepcopy(overlay)


class RuleBook:
    def __init__(self, data: dict[str, Any]):
        self.data = copy.deepcopy(data)
        self._validate()
        self.play_domain = PlayDomainIRV1.from_rules(
            self.data,
            structural_domain=self.structural_domain,
        ).require_integrity(self.structural_domain)
        self.project_domain = ProjectDomainIRV1.from_rules(self.data)
        canonical = json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path, *overlays: str | Path) -> "RuleBook":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for overlay in overlays:
            with Path(overlay).open("r", encoding="utf-8") as fh:
                data = _merge(data, yaml.safe_load(fh))
        return cls(data)

    def _validate(self) -> None:
        required = {
            "schema_version", "profile", "contracts", "seating", "deck",
            "card_rules", "dealing", "bidding", "escalation", "play",
            "projects", "scoring", "disputed_options",
        }
        missing = required - set(self.data)
        if missing:
            raise ValueError(f"missing rule sections: {sorted(missing)}")
        unknown = set(self.data) - required
        if unknown:
            raise ValueError(f"unknown rule sections: {sorted(unknown)}")

        def expect(path: str, actual: Any, expected: Any) -> None:
            if actual != expected:
                raise ValueError(
                    f"unsupported {path}={actual!r}; supported value is {expected!r}"
                )

        expect("schema_version", self.data["schema_version"], 1)
        allowed = set(self.data["contracts"]["allowed"])
        if not allowed or not allowed <= {"sun", "hokum", "ashkal"}:
            raise ValueError("contracts.allowed is invalid")
        contract_cfg = self.data["contracts"]
        require_preselected = bool(contract_cfg["require_preselected"])
        default_contract = contract_cfg["default"]
        if require_preselected:
            if len(allowed) != 1 or default_contract not in allowed:
                raise ValueError(
                    "preselected contract profiles require one allowed default contract"
                )
        elif default_contract is not None:
            raise ValueError(
                "contracts.default requires contracts.require_preselected=true"
            )

        # StructuralDomainIRV1 is the sole deck/seating/dealing admission
        # authority. Runtime, reference, and CP-SAT consumers receive this
        # exact immutable instance instead of re-validating parallel fields.
        self.structural_domain = StructuralDomainIRV1.from_rules(self.data)

        for key in ("sequence_strength", "sun_strength", "hokum_non_trump_strength", "hokum_trump_strength", "sun_points", "hokum_non_trump_points", "hokum_trump_points"):
            if len(self.data["card_rules"][key]) != 8:
                raise ValueError(f"{key} must contain eight values")

        bidding = self.data["bidding"]
        expect("bidding.rounds", bidding["rounds"], 2)
        expect("bidding.pass_allowed", bidding["pass_allowed"], True)
        expected_first = {"pass", *allowed}
        expected_second = {"pass", *(allowed - {"ashkal"})}
        if set(bidding["first_round"]["allowed"]) != expected_first:
            raise ValueError("bidding.first_round.allowed must match enabled contracts")
        if set(bidding["second_round"]["allowed"]) != expected_second:
            raise ValueError("bidding.second_round.allowed must match enabled non-Ashkal contracts")
        expect(
            "bidding.first_round.hokum_uses_purchase_suit",
            bidding["first_round"]["hokum_uses_purchase_suit"],
            "hokum" in allowed,
        )
        expect(
            "bidding.second_round.hokum_must_differ_from_purchase_suit",
            bidding["second_round"]["hokum_must_differ_from_purchase_suit"],
            "hokum" in allowed,
        )
        ashkal = bidding["ashkal"]
        expect("bidding.ashkal.eligible_priority_positions", ashkal["eligible_priority_positions"], [3, 4])
        expect("bidding.ashkal.taker", ashkal["taker"], "bidder_partner")
        expect("bidding.ashkal.resolves_to", ashkal["resolves_to"], "sun")
        if "ashkal" in allowed and not bool(ashkal["enabled"]):
            raise ValueError("Ashkal cannot be allowed while bidding.ashkal.enabled=false")
        heuristics = bidding["heuristics"]
        for key in ("sun_rank_weights", "hokum_trump_rank_weights"):
            if len(heuristics[key]) != 8 or any(float(value) < 0 for value in heuristics[key]):
                raise ValueError(f"bidding.heuristics.{key} must contain eight non-negative values")
        for key in (
            "sun_extra_suit_card_weight", "hokum_extra_trump_card_weight",
            "hokum_off_trump_ace_ten_weight",
        ):
            if float(heuristics[key]) < 0:
                raise ValueError(f"bidding.heuristics.{key} must be non-negative")
        for key in ("sun_extra_suit_card_after", "hokum_extra_trump_card_after"):
            if int(heuristics[key]) < 0:
                raise ValueError(f"bidding.heuristics.{key} must be non-negative")

        escalation = self.data["escalation"]
        expect("escalation.default_level", escalation["default_level"], "normal")
        if escalation["default_mode"] not in {"open", "locked"}:
            raise ValueError("escalation.default_mode must be open or locked")
        expect(
            "escalation.opening_actor_policy",
            escalation["opening_actor_policy"],
            "opponent_team_any_seat",
        )
        expect(
            "escalation.closure_policy",
            escalation["closure_policy"],
            "advance_to_projects_without_announcement",
        )
        if escalation["response_actor_policy"] not in {
            "declarer_and_original_doubler", "team",
        }:
            raise ValueError("unsupported escalation.response_actor_policy")
        full_hokum_chain = ["double", "triple", "four", "gahwa"]
        full_sun_chain = ["double"]
        hokum_chain = escalation["hokum_chain"]
        sun_chain = escalation["sun_chain"]
        if hokum_chain != full_hokum_chain[: len(hokum_chain)]:
            raise ValueError("escalation.hokum_chain must be an ordered supported prefix")
        if sun_chain != full_sun_chain[: len(sun_chain)]:
            raise ValueError("escalation.sun_chain must be an ordered supported prefix")

        play = self.data["play"]
        if play["first_leader"] not in {"dealer_right", "buyer", "configured"}:
            raise ValueError("unsupported play.first_leader")
        for key in (
            "must_follow_suit", "must_trump_when_void",
            "must_overtrump_when_possible", "locked_hokum_blocks_trump_lead",
            "ekka_enabled",
        ):
            expect(f"play.{key}", play[key], True)
        expect(
            "play.partner_winning_exception",
            play["partner_winning_exception"],
            "position_aware_ace_or_ekka",
        )

        disputed = self.data["disputed_options"]

        def validate_disputed_alternatives(
            name: str, expected: tuple[object, ...]
        ) -> None:
            choice = disputed[name]
            alternatives = list(choice["alternatives"])
            if alternatives != list(expected):
                raise ValueError(
                    f"disputed_options.{name}.alternatives must match supported metadata"
                )

        validate_disputed_alternatives("first_leader", ("buyer",))
        validate_disputed_alternatives("double_unspecified_mode", ("locked",))
        validate_disputed_alternatives("hints_have_universal_meaning", ())
        validate_disputed_alternatives("kawesh_saneen", (True,))
        project_priority = disputed["equal_hokum_sequence_trump_priority"]
        if not isinstance(project_priority["selected"], bool):
            raise ValueError(
                "disputed_options.equal_hokum_sequence_trump_priority.selected must be boolean"
            )
        if list(project_priority["alternatives"]) != [
            not bool(project_priority["selected"])
        ]:
            raise ValueError(
                "disputed_options.equal_hokum_sequence_trump_priority.alternatives must list the other supported value"
            )

        projects = self.data["projects"]
        if projects.get("top_rank_visibility", "declared") not in {"declared", "comparison_only"}:
            raise ValueError("unsupported projects.top_rank_visibility")
        expect("projects.declaration_timing", projects["declaration_timing"], "first_trick_turn")
        if projects.get("declaration_mode", "mandatory") not in {
            "mandatory",
            "optional",
        }:
            raise ValueError("unsupported projects.declaration_mode")
        expect("projects.comparison_timing", projects["comparison_timing"], "after_first_trick")
        expect("projects.reveal_timing", projects["reveal_timing"], "during_second_trick")
        if int(projects["max_scoring_projects_per_player"]) < 1:
            raise ValueError("projects.max_scoring_projects_per_player must be positive")
        expect("projects.only_winning_team_scores", projects["only_winning_team_scores"], True)
        minimum_length = int(projects["sequences"]["minimum_length"])
        if minimum_length not in {3, 4, 5}:
            raise ValueError("projects.sequences.minimum_length must be 3, 4, or 5")
        baloot = projects["baloot"]
        expect("projects.baloot.only_hokum", baloot["only_hokum"], True)
        expect("projects.baloot.declare_on_second_card", baloot["declare_on_second_card"], True)
        expect("projects.baloot.siri_or_fifty_inclusion", baloot["siri_or_fifty_inclusion"], "automatic")
        if baloot["hundred_inclusion"] not in {"automatic", "blocked"}:
            raise ValueError("unsupported projects.baloot.hundred_inclusion")
        expect("projects.baloot.single_card_in_other_project", baloot["single_card_in_other_project"], "blocked")
        comparison = projects.get("comparison")
        if comparison is not None:
            priorities = comparison.get("category_priorities")
            if not isinstance(priorities, dict) or set(priorities) != {
                "siri", "fifty", "hundred_sequence", "hundred_four_kind", "four_hundred",
            } or any(type(value) is not int or value < 0 for value in priorities.values()):
                raise ValueError("projects.comparison.category_priorities is invalid")

        scoring = self.data["scoring"]
        expected_sun_raw = 4 * sum(self.data["card_rules"]["sun_points"]) + int(scoring["last_trick_bonus_raw"])
        expected_hokum_raw = (
            sum(self.data["card_rules"]["hokum_trump_points"])
            + 3 * sum(self.data["card_rules"]["hokum_non_trump_points"])
            + int(scoring["last_trick_bonus_raw"])
        )
        expect("scoring.raw_card_totals.sun", int(scoring["raw_card_totals"]["sun"]), expected_sun_raw)
        expect("scoring.raw_card_totals.hokum", int(scoring["raw_card_totals"]["hokum"]), expected_hokum_raw)
        expect("scoring.conversion.sun", scoring["conversion"]["sun"], "nearest_ten_except_exact_five_then_divide_by_five")
        expect("scoring.conversion.hokum", scoring["conversion"]["hokum"], "nearest_ten_half_down_then_divide_by_ten")
        expect("scoring.contract.buyer_team_must_win", scoring["contract"]["buyer_team_must_win"], True)
        expect("scoring.contract.loss_transfers_all", scoring["contract"]["loss_transfers_all"], True)
        expect("scoring.escalation.winner_takes_all_when_multiplied", scoring["escalation"]["winner_takes_all_when_multiplied"], True)
        if "winner" in scoring["kabout"]:
            expect("scoring.kabout.winner", scoring["kabout"]["winner"], "trick_sweeper")
        if int(scoring["escalation"]["project_multiplier_cap"]) < 1:
            raise ValueError("scoring.escalation.project_multiplier_cap must be positive")
        if int(scoring["escalation"]["baloot_multiplier_cap"]) < 1:
            raise ValueError("scoring.escalation.baloot_multiplier_cap must be positive")
        expect("scoring.match.tie_above_target_continues", scoring["match"]["tie_above_target_continues"], True)
        expect("scoring.match.dealer_rotates_right", scoring["match"]["dealer_rotates_right"], True)
        if self.data["scoring"]["match"]["target"] <= 0:
            raise ValueError("match target must be positive")

    @property
    def id(self) -> str:
        return str(self.data["profile"]["id"])

    @property
    def version(self) -> str:
        return str(self.data["profile"]["version"])

    def strength(self, card: Card, contract: Contract, trump: Suit | None) -> int:
        return self.play_domain.strength(card, contract, trump)

    def sequence_strength(self, card: Card) -> int:
        return self.project_domain.sequence_strength_for(card)

    def points(self, card: Card, contract: Contract, trump: Suit | None) -> int:
        return self.play_domain.points(card, contract, trump)
