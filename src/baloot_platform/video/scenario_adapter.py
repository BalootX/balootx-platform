"""Observer-safe scene reduction for verified Research V4 scenario results."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.exact_state_view import build_observer_safe_exact_state_v1
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.serialization import replay_trusted_context

from .scene import build_scene_timeline, reduce_scene_document
from .scene_adapter import DecisionScenePackageV1


def _public_play(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seat": value["seat"],
        "card": copy.deepcopy(dict(value["card"])),
        "trickIndex": int(value.get("trickIndex", 0)),
        "position": int(value.get("position", 0)),
        "ekka": bool(value.get("ekka", False)),
        "baloot": bool(value.get("baloot", False)),
    }


def _public_trick(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": int(value.get("index", 0)),
        "leader": value.get("leader"),
        "winner": value.get("winner"),
        "plays": [_public_play(item) for item in value.get("plays") or ()],
    }


def _proof_scope_position_values(receipt: Mapping[str, Any], focal: int) -> dict[str, int] | None:
    group = dict(receipt.get("postAuction") or {}).get("projectPlay") or {}
    if not group:
        return None
    start = group.get("coveredFromPlayIndex")
    execution = receipt.get("qualityExecution") or {}
    if execution.get("exactTailProof") is True:
        exact_trick = execution.get("exactPlayFromTrick")
        if type(exact_trick) is not int or not 1 <= exact_trick <= 8:
            raise ValueError("Research V4 exact-tail proof-start is invalid")
        start = max(start, 4 * (exact_trick - 1)) if type(start) is int else None
    if type(start) is not int or type(focal) is not int or not 0 <= start <= focal < 32:
        raise ValueError("Research V4 proof-start and focal-play boundary is invalid")
    return {"proofStartTrick": start // 4 + 1, "proofStartPlay": start % 4 + 1,
            "featuredTrick": focal // 4 + 1, "featuredPlay": focal % 4 + 1}


def _proof_reason_key(proof: Mapping[str, Any], localizations: Mapping[str, Any]) -> str:
    completion = str(proof.get("completionStatus") or "")
    mode = str(proof.get("achievedMode") or "")
    if completion == "proof_incomplete":
        return "video.analysis.proof_incomplete"
    if completion == "proof_failed":
        return "video.analysis.proof_failed"
    if mode == "bounded":
        return "video.analysis.bounded_proof"
    if mode == "policy_conditioned":
        return "video.analysis.policy_conditioned_proof"
    if (proof.get("utility") == "awarded_point_margin" and completion == "exact_complete"
            and proof.get("achievedVisibility") == "actor_information_set"
            and "video.analysis.worst_case_awarded_margin" in localizations):
        return "video.analysis.worst_case_awarded_margin"
    winning = int(proof.get("winningRootActionCount") or 0)
    if winning == 0:
        return "video.analysis.no_salvage"
    if winning == 1:
        roots = proof.get("rootActions")
        if (proof.get("utility") == "awarded_point_margin"
                and completion == "exact_complete"
                # Hidden-world minimax margins are guarantees, not this deal's score.
                and proof.get("achievedVisibility") != "actor_information_set"
                and proof.get("requestedVisibility") != "actor_information_set"
                and isinstance(roots, list)
                and len(roots) > 1
                and all(type(root.get("value")) is int for root in roots)
                and "video.analysis.unique_awarded_margin" in localizations):
            return "video.analysis.unique_awarded_margin"
        return "video.analysis.one_winning_action"
    return "video.analysis.multiple_winning_actions"


def build_scenario_decision_scene_package(
    *,
    exact_state: Mapping[str, Any],
    proof_outcome: Mapping[str, Any],
    scenario_projection: Mapping[str, Any],
    profile: Mapping[str, Any],
    effective_config_sha256: str,
    observer_seat: str,
    rules: RuleBook,
    trusted_principal_variation: Sequence[str] | None = None,
    trusted_complete_trace: Mapping[str, Any] | None = None,
) -> DecisionScenePackageV1:
    """Build a play-decision package without exposing trusted hidden state."""

    if exact_state.get("rulesHash") != rules.sha256:
        raise ValueError("Research V4 render rules differ from the verified exact state")
    if proof_outcome.get("identities", {}).get("rulesHash") != rules.sha256:
        raise ValueError("Research V4 render rules differ from the verified proof")
    if any(proof_outcome.get("identities", {}).get(key) != exact_state.get(key)
           for key in ("exactStateId", "stateHash")):
        raise ValueError("Research V4 render proof references a different exact state")

    # Keep the video package importable while Research loads its render service.
    from baloot_platform.research.scenario_result import (
        actor_public_proof_outcome_v1,
    )

    decision_kind = str(scenario_projection.get("decisionKind") or "")
    if decision_kind not in {"bid", "escalation", "project", "play"}:
        raise ValueError("Research V4 rendering requires a supported decision kind")
    acting = str(exact_state.get("nextActor") or "")
    if acting not in {seat.name.lower() for seat in Seat}:
        raise ValueError("Research V4 exact state lacks an acting seat")
    if observer_seat != acting:
        raise ValueError("Research V4 actor render requires the acting observer")
    projected_exact_state_id = scenario_projection.get("exactStateId")
    if (
        projected_exact_state_id is not None
        and exact_state.get("exactStateId") != projected_exact_state_id
    ):
        raise ValueError(
            "Research V4 render evidence references different exact states"
        )
    if str(exact_state.get("decisionKind") or "") != decision_kind:
        raise ValueError(
            "Research V4 render evidence references a different decision kind"
        )
    private_proof = copy.deepcopy(dict(proof_outcome))
    private_continuation = (
        list(trusted_principal_variation)
        if trusted_principal_variation is not None
        else private_proof.get("principalVariation")
    )
    proof_outcome = actor_public_proof_outcome_v1(private_proof)
    playable = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=scenario_projection,
        observer_seat=observer_seat,
    )
    state = playable["publicState"]
    legal_actions = playable.get("legalActions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("Research V4 play render requires legal actions")
    root_actions = proof_outcome.get("rootActions")
    if not isinstance(root_actions, list):
        raise ValueError("Research V4 proof root actions are invalid")
    if {str(item.get("action")) for item in root_actions} != {
        str(item.get("code")) for item in legal_actions
    }:
        raise ValueError("Research V4 proof and exact-state legal actions differ")

    observer_view = {
        "observer": observer_seat,
        **state,
    }
    public_events = tuple(state.get("events") or ())
    decision_tricks = [
        _public_trick(trick) for trick in state.get("completedTricks") or ()
    ]
    decision_plays = [play for trick in decision_tricks for play in trick["plays"]]
    if state.get("currentTrick"):
        decision_plays.extend(
            _public_play(item) for item in state["currentTrick"].get("plays") or ()
        )
    public_lifecycle: dict[str, Any] = {
        "dealer": state.get("dealer"),
        "buyer": state.get("buyer"),
        "contract": state.get("contract"),
        "trumpSuit": state.get("trumpSuit"),
        "firstLeader": state.get("firstLeader"),
        "bidHistory": copy.deepcopy(state.get("biddingHistory") or []),
        "projects": copy.deepcopy(state.get("projects") or []),
        "preDecisionMatchScore": copy.deepcopy(state.get("matchScore") or [0, 0]),
        "matchScore": copy.deepcopy(state.get("matchScore") or [0, 0]),
        "roundScore": copy.deepcopy(state.get("roundScore")),
        "completedTricks": decision_tricks,
        "plays": copy.deepcopy(decision_plays),
        "pauseBeforePlayIndex": len(decision_plays),
    }
    observer_lifecycle: dict[str, Any] = {
        "initialHandCards": copy.deepcopy(
            dict(state.get("hands") or {}).get(observer_seat) or []
        )
    }
    if decision_kind != "play" and isinstance(trusted_complete_trace, Mapping):
        completed = trusted_complete_trace.get("completeExactNormalization")
        completed_state = (
            completed.get("state") if isinstance(completed, Mapping) else None
        )
        if isinstance(completed_state, Mapping):
            if completed.get("rulesHash") != exact_state.get("rulesHash"):
                raise ValueError("Research V4 continuation rules authority mismatch")
            final_tricks = [
                _public_trick(trick)
                for trick in completed_state.get("completedTricks") or ()
            ]
            final_plays = [
                play for trick in final_tricks for play in trick["plays"]
            ]
            public_lifecycle = {
                "dealer": completed_state.get("dealer"),
                "buyer": completed_state.get("buyer"),
                "contract": completed_state.get("contract"),
                "trumpSuit": completed_state.get("trumpSuit"),
                "firstLeader": completed_state.get("firstLeader"),
                "bidHistory": copy.deepcopy(
                    completed_state.get("biddingHistory") or []
                ),
                "projects": copy.deepcopy(completed_state.get("projects") or []),
                "preDecisionMatchScore": copy.deepcopy(
                    state.get("matchScore") or [0, 0]
                ),
                "matchScore": copy.deepcopy(
                    completed_state.get("matchScore") or [0, 0]
                ),
                "roundScore": copy.deepcopy(completed_state.get("roundScore")),
                "completedTricks": final_tricks,
                "plays": copy.deepcopy(final_plays),
                "pauseBeforePlayIndex": 0,
            }
            initial_hand_by_code = {
                str(card["code"]): copy.deepcopy(card)
                for card in dict(state.get("hands") or {}).get(observer_seat) or []
            }
            for play in final_plays:
                if play["seat"] == observer_seat:
                    initial_hand_by_code[str(play["card"]["code"])] = copy.deepcopy(
                        play["card"]
                    )
            observer_lifecycle = {
                "initialHandCards": sorted(
                    initial_hand_by_code.values(), key=lambda card: int(card["id"])
                )
            }
    if decision_kind == "play":
        trusted_state = replay_trusted_context(dict(exact_state["state"]), rules)
        pause_before_play_index = len(trusted_state.played_cards)
        retained_plays = None
        if isinstance(trusted_complete_trace, Mapping):
            completed = trusted_complete_trace.get("completeExactNormalization")
            if not isinstance(completed, Mapping) or completed.get("rulesHash") != exact_state.get("rulesHash"):
                raise ValueError("Research V4 continuation rules authority mismatch")
            completed_state = completed.get("state")
            if not isinstance(completed_state, Mapping):
                raise ValueError("Research V4 continuation state is missing")
            retained_plays = [
                _public_play(play)
                for trick in completed_state.get("completedTricks") or ()
                for play in trick.get("plays") or ()
            ]
            if len(retained_plays) != 32 or retained_plays[:pause_before_play_index] != decision_plays:
                raise ValueError("Research V4 continuation decision prefix mismatch")
            private_continuation = [
                str(play["card"]["code"]) + ("+ekka" if play["ekka"] else "")
                for play in retained_plays[pause_before_play_index:]
            ]
        if not isinstance(private_continuation, list) or not private_continuation:
            raise ValueError("Research V4 play render requires a proof continuation")
        engine = BalootEngine(rules)
        for raw_action in private_continuation:
            action = LegalPlayActionV1.from_code(str(raw_action))
            if action not in engine.legal_play_actions(trusted_state):
                raise ValueError("Research V4 proof continuation is illegal")
            engine.play_card(
                trusted_state,
                action.card,
                ekka=action.ekka,
            )
        final_state = trusted_state.to_dict(reveal_hands=False)
        final_tricks = [
            _public_trick(trick) for trick in final_state.get("completedTricks") or ()
        ]
        final_plays = [play for trick in final_tricks for play in trick["plays"]]
        if final_state.get("currentTrick"):
            final_plays.extend(
                _public_play(item)
                for item in final_state["currentTrick"].get("plays") or ()
            )
        if retained_plays is not None and final_plays != retained_plays:
            raise ValueError("Research V4 continuation replay mismatch")
        current_state = state
        public_lifecycle = {
            "dealer": current_state.get("dealer"),
            "buyer": current_state.get("buyer"),
            "contract": current_state.get("contract"),
            "trumpSuit": current_state.get("trumpSuit"),
            "firstLeader": current_state.get("firstLeader"),
            "bidHistory": copy.deepcopy(current_state.get("biddingHistory") or []),
            "projects": copy.deepcopy(current_state.get("projects") or []),
            "preDecisionMatchScore": copy.deepcopy(
                current_state.get("matchScore") or [0, 0]
            ),
            "matchScore": copy.deepcopy(final_state.get("matchScore") or [0, 0]),
            "roundScore": copy.deepcopy(final_state.get("roundScore") or {}),
            "completedTricks": final_tricks,
            "plays": copy.deepcopy(final_plays),
            "pauseBeforePlayIndex": pause_before_play_index,
        }
        initial_hand = copy.deepcopy(
            dict(current_state.get("hands") or {}).get(observer_seat) or []
        )
        initial_hand.extend(
            copy.deepcopy(play["card"])
            for play in final_plays[:pause_before_play_index]
            if play["seat"] == observer_seat
        )
        initial_hand.sort(key=lambda card: int(card["id"]))
        observer_lifecycle = {"initialHandCards": initial_hand}
        complete_prefix = (
            trusted_complete_trace.get("completeExactTracePrefix")
            if isinstance(trusted_complete_trace, Mapping)
            else None
        )
        if isinstance(complete_prefix, Mapping):
            deck = complete_prefix.get("deckOrder")
            if not isinstance(deck, list) or len(deck) != 32 or sorted(deck) != list(range(32)):
                raise ValueError("Research V4 complete video deck is invalid")
            purchase = next(
                (event for event in public_events if event.get("type") == "contract_purchased"),
                None,
            )
            if purchase is None or purchase.get("purchaseRecipient") is None:
                raise ValueError("Research V4 complete video purchase event is missing")
            layout = DealLayoutV1.build(
                rules.structural_domain,
                dealer=Seat.parse(public_lifecycle["dealer"]),
                buyer=Seat.parse(purchase["purchaseRecipient"]),
            )
            observer = Seat.parse(observer_seat)
            final_cards = {deck[position] for position in layout.positions_for(observer, final=True)}
            if final_cards != {card["id"] for card in initial_hand}:
                raise ValueError("Research V4 complete video observer hand disagrees with the deal")
            observer_lifecycle["initialFiveCards"] = [
                Card(deck[position]).to_dict()
                for position in layout.positions_for(observer, final=False)
            ]
            public_lifecycle["purchaseRecipient"] = purchase["purchaseRecipient"]
            public_lifecycle["purchaseCard"] = copy.deepcopy(purchase["purchaseCard"])
            public_lifecycle["escalationHistory"] = copy.deepcopy(
                state.get("escalationHistory") or []
            )
    reason_key = _proof_reason_key(proof_outcome, profile["localizations"])
    achieved_visibility = proof_outcome.get("achievedVisibility")
    proof_scope = (
        achieved_visibility
        if proof_outcome.get("completionStatus")
        in {"exact_complete", "bounded_complete", "policy_complete", "proof_failed"}
        and achieved_visibility in {"complete_state", "actor_information_set"}
        else "unproven"
    )
    scope_key = f"video.analysis.proof_scope.{proof_scope}"
    scope_copy = {
        f"video.analysis.proof_scope.{scope}"
        for scope in ("complete_state", "actor_information_set", "unproven")
    }
    localizations = profile["localizations"]
    if scope_copy.intersection(localizations) and any(
        not isinstance(localizations.get(key), str) or not localizations[key].strip()
        for key in scope_copy
    ):
        raise ValueError("Research V4 video proof-scope copy is incomplete")
    scope_values = None
    if decision_kind == "play" and isinstance(trusted_complete_trace, Mapping):
        quality_receipt = dict(trusted_complete_trace.get("playingQualityEvidence") or {}).get("receipt") or {}
        tail_key = f"video.analysis.proof_scope.{proof_scope}_tail"
        if tail_key in localizations:
            scope_values = _proof_scope_position_values(quality_receipt, len(decision_plays))
            if scope_values is not None:
                scope_key = tail_key
    reason_values = None
    optimal_count = None
    if proof_outcome.get("completionStatus") == "exact_complete" and root_actions:
        values = [row.get("value") for row in root_actions]
        if all(type(value) in {int, float} for value in values):
            optimal_count = sum(value == max(values) for value in values)
    if reason_key == "video.analysis.worst_case_awarded_margin":
        selected_roots = [row for row in root_actions if row.get("action") == proof_outcome.get("selectedAction")]
        alternatives = [row.get("value") for row in root_actions
                        if row.get("action") != proof_outcome.get("selectedAction")]
        if len(selected_roots) != 1 or not alternatives or optimal_count is None:
            raise ValueError("worst-case explanation requires exact root margins")
        reason_values = {"worstCaseValue": selected_roots[0]["value"],
                         "nextBestWorstCaseValue": max(alternatives), "optimalActionCount": optimal_count}
    if reason_key == "video.analysis.unique_awarded_margin":
        selected = proof_outcome.get("selectedAction")
        selected_roots = [row for row in root_actions if row.get("action") == selected]
        alternatives = [row.get("value") for row in root_actions if row.get("action") != selected]
        if (len(selected_roots) != 1 or not alternatives
                or type(selected_roots[0].get("value")) is not int
                or any(type(value) is not int for value in alternatives)):
            raise ValueError("score-swing explanation requires exact root margins")
        reason_values = {"selectedMargin": selected_roots[0]["value"],
                         "nextBestMargin": max(alternatives)}
    analysis = {
        "reasonKey": reason_key,
        **({"reasonValues": reason_values} if reason_values is not None else {}),
        "decisionKind": decision_kind,
        "truthLabel": (
            "exact_optimal"
            if proof_outcome.get("achievedMode") == "exact"
            and proof_outcome.get("completionStatus") == "exact_complete"
            and proof_outcome.get("claimSatisfied") is True
            else "robust_best_known"
            if proof_outcome.get("completionStatus")
            in {"bounded_complete", "policy_complete"}
            else "unresolved"
        ),
        "requestedMode": proof_outcome.get("requestedMode"),
        "achievedMode": proof_outcome.get("achievedMode"),
        "completionStatus": proof_outcome.get("completionStatus"),
        "requestedVisibility": proof_outcome.get("requestedVisibility"),
        "achievedVisibility": achieved_visibility,
        **({"proofScopeKey": scope_key} if scope_key in localizations else {}),
        **({"proofScopeValues": scope_values} if scope_values is not None else {}),
        **({"optimalActionCount": optimal_count} if optimal_count is not None else {}),
        "sunTieOutcome": ScoringDomainIRV1.from_rules(rules.data).buyer_tie_outcome or "legacy_raw_units",
        "rulesProfileSha256": rules.sha256,
        "claim": proof_outcome.get("claim"),
        "utility": proof_outcome.get("utility"),
        "claimSatisfied": proof_outcome.get("claimSatisfied"),
        "selectedAction": proof_outcome.get("selectedAction"),
        "winningRootActionCount": proof_outcome.get("winningRootActionCount"),
        "rootActions": copy.deepcopy(root_actions),
        "horizon": copy.deepcopy(proof_outcome.get("horizon")),
        "boundaryEvaluator": proof_outcome.get("boundaryEvaluator"),
        "continuationComplete": bool(public_lifecycle.get("roundScore")),
        "publicProjectionHash": playable["publicProjectionHash"],
        "decisionReference": copy.deepcopy(playable.get("decisionReference")),
    }
    render = dict(profile["render"])
    common = {
        "observer_view": observer_view,
        "public_events": public_events,
        "effective_config_sha256": effective_config_sha256,
        "locale": str(render["locale"]),
        "direction": str(render["direction"]),
    }
    prompt = {
        "prompt_key": (
            "video.bid.challenge_question"
            if decision_kind == "bid"
            else "video.escalation.challenge_question"
            if decision_kind == "escalation"
            else "video.project.challenge_question"
            if decision_kind == "project"
            else "video.decision.choose_best_move"
        ),
        "legal_choices": tuple(copy.deepcopy(legal_actions)),
        "acting_seat": acting,
        "prompt_contract": "choose_acting_action",
        "rendition_type": "actor_challenge",
    }
    context = reduce_scene_document(
        stage="context",
        public_lifecycle_facts=public_lifecycle,
        observer_lifecycle_facts=observer_lifecycle,
        **common,
    )
    question = reduce_scene_document(stage="question", **prompt, **common)
    reveal = reduce_scene_document(
        stage="reveal",
        analysis_facts=analysis,
        **prompt,
        **common,
    )
    continuation = reduce_scene_document(
        stage="continuation", public_lifecycle_facts=public_lifecycle,
        observer_lifecycle_facts=observer_lifecycle, **common,
    )
    outcome = reduce_scene_document(
        stage="outcome",
        analysis_facts=analysis,
        **common,
    )
    timeline_profile = dict(profile["timeline"])
    timeline = build_scene_timeline(
        fps=int(render["fps"]),
        width=int(render["width"]),
        height=int(render["height"]),
        segments=(
            ("question", question.scene_id, timeline_profile["pause_seconds"]),
            ("reveal", reveal.scene_id, timeline_profile["reveal_seconds"]),
            (
                "continuation",
                continuation.scene_id,
                timeline_profile["continuation_seconds"],
            ),
        ),
    )
    return DecisionScenePackageV1(
        scenes=(context, question, reveal, continuation, outcome),
        timeline=timeline,
    )


__all__ = ["build_scenario_decision_scene_package"]
