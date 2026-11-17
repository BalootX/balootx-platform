from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from baloot_platform.agents.profiles import AdaptivePlayStrategy, AgentProfileRegistry
from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.models import InformationState
from baloot_platform.challenge.contracts_v7 import (
    ChallengeContract,
    ContractProfile,
    ContractVerification,
    normalize_bid_actions,
    normalize_play_actions,
    safest_matching_contract,
    verify_contract,
)
from baloot_platform.challenge.interestingness_v4 import (
    ActionOutcome,
    ContextFeatures,
    InterestingnessVector,
    score_interestingness,
)
from baloot_platform.config import ProfileBundle, build_bundle, platform_root
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract
from baloot_platform.core.scenario import EpisodeSpec, ScenarioRequest, generate_episode
from baloot_platform.core.episode_io import episode_spec_from_dict
from baloot_platform.core.bid_scenario import BidEpisodeSpec, generate_bid_episode
from baloot_platform.history.provenance_v4 import game_serial
from baloot_platform.learning.challenge_publication import publication_policy_sha256
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.plugins.base import PluginDescriptor


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MiningRequest:
    source: str
    decision_kind: str
    seed_start: int
    seeds: int
    contract: str
    buyer_relationship: str
    archetype: str
    difficulty_min: int
    difficulty_max: int
    minimum_publishability: float
    maximum_results: int
    require_project: bool
    pause_trick_min: int
    pause_trick_max: int
    maximum_seconds: float
    player_profiles: tuple[tuple[str, str], ...]
    shallow_samples: int | None
    deep_samples: int | None
    exact_allocation_limit: int | None
    exact_search_remaining: int | None
    content_contract: str


@dataclass(frozen=True, slots=True)
class MinedChallenge:
    challenge_id: str
    fingerprint: str
    archetype: str
    vector: InterestingnessVector
    episode: EpisodeSpec
    source: str
    contract_verification: ContractVerification
    stability: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        actions = sorted(
            self.episode.analysis.actions,
            key=lambda item: (
                item.match_win_probability,
                item.expected_round_margin,
                item.best_share,
            ),
            reverse=True,
        )
        best = actions[0]
        worst = actions[-1]
        second = actions[1] if len(actions) > 1 else None
        return {
            "challengeId": self.challenge_id,
            "fingerprint": self.fingerprint,
            "archetype": self.archetype,
            "source": self.source,
            "observerSafe": True,
            "contentContract": self.contract_verification.to_dict(),
            "stability": dict(self.stability or {}),
            "publishability": self.vector.publishability_score,
            "difficulty": self.vector.difficulty_prior,
            "interestingness": {
                "regretGap": self.vector.regret_gap,
                "winProbabilityGap": self.vector.win_probability_gap,
                "disasterGap": self.vector.disaster_gap,
                "cliffScore": self.vector.cliff_score,
                "debateScore": self.vector.debate_score,
                "comebackScore": self.vector.comeback_score,
                "surpriseScore": self.vector.surprise_score,
                "hiddenInformationScore": self.vector.hidden_information_score,
                "educationalScore": self.vector.educational_score,
                "robustnessScore": self.vector.robustness_score,
                "archetypeScores": dict(self.vector.archetype_scores),
                "reasons": list(self.vector.reasons),
            },
            "episodeId": self.episode.episode_id,
            "gameSerial": self.episode.game_serial,
            "gameRecordHash": self.episode.game_record_hash,
            "decision": {
                "kind": "play",
                "recommendedCard": best.card.to_dict(),
                "legalActions": [item.card.to_dict() for item in actions],
                "best": best.to_dict(),
                "second": second.to_dict() if second else None,
                "worst": worst.to_dict(),
                "matchWinProbabilitySwing": best.match_win_probability - worst.match_win_probability,
                "expectedRoundMarginSwing": best.expected_round_margin - worst.expected_round_margin,
                "analysis": self.episode.analysis.to_dict(),
            },
        }


class ProgressiveChallengeMiner:
    """Resource-efficient staged discovery of publishable Baloot decisions.

    The miner intentionally avoids deeply solving every random position. It
    screens many candidates cheaply, reuses one hidden-deal cohort across all
    legal actions, confirms only survivors with larger budgets, and applies
    diversity plus robustness gates before persistence.
    """

    descriptor = PluginDescriptor(
        "progressive_challenge_miner_v2",
        "challenge_miner",
        "2.0.0",
        "Progressive simulation/history challenge discovery with paired counterfactual confirmation",
        (
            "simulation",
            "history",
            "paired_counterfactuals",
            "difficulty",
            "diversity",
            "provenance",
        ),
        resource_class="cpu_high",
    )

    def __init__(
        self,
        root: str | Path | None,
        store: LearningStore,
        *,
        learning: LearningService | None = None,
        bundle: ProfileBundle | None = None,
        agent_registry: AgentProfileRegistry | None = None,
    ) -> None:
        self.root = platform_root(root)
        self.store = store
        self.learning = learning
        self.bundle = bundle or build_bundle(self.root)
        self.agent_registry = (
            agent_registry
            or AgentProfileRegistry.from_bundle(self.root)
        )
        self.profile = self.bundle.component("challenge")
        self.contract_profile = ContractProfile.from_mapping(
            self.bundle.component("challenge_contracts")
        )
        self.last_run: dict[str, Any] = {}

    def _pin_publication_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["observerSafe"] = True
        payload["configurationBundleSha256"] = self.bundle.sha256
        payload["publicationPolicySha256"] = publication_policy_sha256(
            self.profile["publication"]
        )
        return payload

    @staticmethod
    def _stability_metrics(
        first,
        second,
        policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = (
            "minimum_regret_gap",
            "minimum_best_share",
            "maximum_expected_margin_drift",
            "minimum_hidden_deals",
            "require_same_recommended_action",
            "publish_uncertain",
        )
        missing = [key for key in required if key not in policy]
        if missing:
            raise ValueError(
                "challenge stability policy is missing "
                + ", ".join(sorted(missing))
            )
        if len(first.actions) < 2 or len(second.actions) < 2:
            raise ValueError(
                "challenge stability requires at least two legal actions"
            )

        def metrics(result) -> tuple[str, float, float, float]:
            best, runner_up = result.actions[:2]
            return (
                best.card.code,
                float(best.expected_match_margin)
                - float(runner_up.expected_match_margin),
                float(best.best_share),
                min(float(item.exact_share) for item in result.actions),
            )

        first_action, first_gap, first_share, first_exact = metrics(first)
        second_action, second_gap, second_share, second_exact = metrics(
            second
        )
        same_action = first_action == second_action
        minimum_gap = min(first_gap, second_gap)
        minimum_share = min(first_share, second_share)
        margin_drift = abs(first_gap - second_gap)
        hidden_deals = min(first.hidden_deals, second.hidden_deals)
        reasons: list[str] = []
        if (
            bool(policy["require_same_recommended_action"])
            and not same_action
        ):
            reasons.append("recommended_action_changed")
        if minimum_gap < float(policy["minimum_regret_gap"]):
            reasons.append("regret_gap_below_policy")
        if minimum_share < float(policy["minimum_best_share"]):
            reasons.append("best_share_below_policy")
        if margin_drift > float(
            policy["maximum_expected_margin_drift"]
        ):
            reasons.append("margin_drift_above_policy")
        if hidden_deals < int(policy["minimum_hidden_deals"]):
            reasons.append("hidden_deal_support_below_policy")
        accepted = not reasons
        return {
            "status": "stable" if accepted else "uncertain",
            "accepted": accepted,
            "publishUncertain": bool(policy["publish_uncertain"]),
            "firstRecommendedAction": first_action,
            "reanalysisRecommendedAction": second_action,
            "recommendedActionStable": same_action,
            "minimumRegretGap": minimum_gap,
            "minimumBestShare": minimum_share,
            "expectedMarginGapDrift": margin_drift,
            "minimumHiddenDeals": hidden_deals,
            "minimumExactShare": min(first_exact, second_exact),
            "reasons": reasons,
            "policySha256": _hash(dict(policy)),
        }

    def _verify_play_contract(self, episode: EpisodeSpec, vector: InterestingnessVector, requested: str) -> ContractVerification:
        actions = normalize_play_actions(episode.analysis.actions)
        opponent_bought = episode.state_before_pause.buyer.team != episode.viewer.team
        disagreement = vector.hidden_information_score
        if requested in {"", "any", "auto"}:
            return safest_matching_contract(
                actions, self.contract_profile, opponent_bought=opponent_bought,
                hidden_deal_disagreement=disagreement, decision_kind="play",
            )
        return verify_contract(
            requested, actions, self.contract_profile, opponent_bought=opponent_bought,
            hidden_deal_disagreement=disagreement,
        )

    def _verify_bid_contract(self, episode: BidEpisodeSpec, vector: InterestingnessVector, requested: str) -> ContractVerification:
        actions = normalize_bid_actions(episode.analysis.choices)
        contract = ChallengeContract.BUY_OR_PASS if requested in {"", "any", "auto"} else ChallengeContract.parse(requested)
        return verify_contract(
            contract, actions, self.contract_profile,
            opponent_bought=episode.final_state.buyer is not None and episode.final_state.buyer.team != episode.viewer.team,
            hidden_deal_disagreement=vector.hidden_information_score,
        )

    @staticmethod
    def _contract(value: str) -> Contract | None:
        return None if value in {"any", "auto", ""} else Contract.parse(value)

    @staticmethod
    def _relationship(episode: EpisodeSpec) -> str:
        buyer = episode.state_before_pause.buyer
        viewer = episode.viewer
        if buyer == viewer:
            return "self"
        if buyer == viewer.partner:
            return "partner"
        return "opponent"

    @classmethod
    def _relationship_matches(cls, episode: EpisodeSpec, requested: str) -> bool:
        requested = str(requested or "any").lower()
        return requested in {"any", ""} or cls._relationship(episode) == requested

    def _fingerprint(self, episode: EpisodeSpec) -> str:
        state = episode.state_before_pause
        identity = {
            "fingerprintVersion": int(self.profile.get("diversity", {}).get("fingerprint_version", 1)),
            "algorithmVersion": 3,
            "decisionKind": "play",
            "contract": state.contract.value,
            "trump": state.trump_suit.name.lower() if state.trump_suit else None,
            "viewer": episode.viewer.name.lower(),
            "viewerHand": state.hands[int(episode.viewer)],
            "played": [(item.seat.name.lower(), item.card.id) for item in state.played_cards],
            "trick": [(item.seat.name.lower(), item.card.id) for item in state.current_trick.plays],
            "legal": sorted(item.card.id for item in episode.analysis.actions),
            "matchScore": list(state.match_score),
            "buyerRelationship": ProgressiveChallengeMiner._relationship(episode),
            "projectKinds": sorted(project.kind for project in state.projects if project.declared or project.revealed),
            "escalation": state.escalation,
        }
        return _hash(identity)

    @staticmethod
    def _vector(episode: EpisodeSpec) -> InterestingnessVector:
        state = episode.state_before_pause
        actions = episode.analysis.actions
        outcomes = [
            ActionOutcome(
                action_id=item.card.code,
                expected_match_utility=(2.0 * item.match_win_probability) - 1.0,
                match_win_probability=item.match_win_probability,
                expected_round_margin=item.expected_round_margin,
                downside_cvar=math.tanh(item.downside_cvar / 55.0),
                best_deal_share=item.best_share,
                hidden_deal_values=tuple(math.tanh(value / 55.0) for value in item.deal_margins),
                is_obvious_heuristic_choice=False,
                explanation_features=("card_counting", "trump_control")
                if state.contract == Contract.HOKUM
                else ("card_counting",),
            )
            for item in actions
        ]
        context = ContextFeatures(
            decision_kind="play",
            contract=state.contract.value,
            buyer_relationship=ProgressiveChallengeMiner._relationship(episode),
            opponent_bought=state.buyer.team != episode.viewer.team,
            score_pressure=min(1.0, abs(state.match_score[0] - state.match_score[1]) / 152.0),
            tricks_remaining=8 - len(state.completed_tricks),
            depends_on_signal=bool(episode.analysis.signals),
            counterintuitive=episode.analysis.debate_score < 55 and len(actions) > 1,
            project_relevant=bool(state.projects),
            escalation_relevant=state.escalation != "normal",
            tags=(state.contract.value,),
        )
        return score_interestingness(outcomes, context)

    @staticmethod
    def _archetype(vector: InterestingnessVector, requested: str) -> str:
        if requested not in {"any", ""}:
            return requested
        return max(vector.archetype_scores.items(), key=lambda item: item[1])[0]

    def _passes_requested_archetype(self, vector: InterestingnessVector, requested: str) -> bool:
        if requested in {"any", ""}:
            return True
        minimum = float(self.profile.get("publication", {}).get("minimum_archetype_score", 0.50))
        return float(vector.archetype_scores.get(requested, 0.0)) >= minimum

    def _deep_confirm(
        self,
        episode: EpisodeSpec,
        seed: int,
        *,
        request: MiningRequest | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> EpisodeSpec:
        info = InformationState.from_complete(episode.state_before_pause, episode.viewer)
        confirmation = self.profile.get("confirmation", {})
        result = analyze(
            info,
            self.bundle.rules,
            self.bundle.convention,
            requested_samples=int(request.deep_samples if request and request.deep_samples is not None else confirmation.get("deep_samples", 96)),
            exact_allocation_limit=int(request.exact_allocation_limit if request and request.exact_allocation_limit is not None else confirmation.get("exact_allocation_limit", 20_000)),
            exact_search_max_remaining=int(request.exact_search_remaining if request and request.exact_search_remaining is not None else confirmation.get("exact_search_remaining", 10)),
            maximum_sampling_attempts=int(confirmation.get("maximum_sampling_attempts", max(12_000, int(request.deep_samples if request and request.deep_samples is not None else confirmation.get("deep_samples", 96)) * 500))),
            exact_search_max_nodes=int(confirmation.get("exact_search_max_nodes", 40_000)),
            exact_search_max_seconds=float(confirmation.get("exact_search_max_seconds", 0.15)),
            seed=seed,
            cancel_check=cancel_check,
        )
        final = episode.state_before_pause.clone()
        engine = BalootEngine(self.bundle.rules)
        engine.play_card(final, result.recommended_card)
        profile_map = dict(request.player_profiles) if request is not None else {}
        while final.current_player is not None:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("challenge confirmation cancelled")
            seat_key = final.current_player.name.lower()
            profile_id = profile_map.get(
                seat_key,
                self.agent_registry.default_lineup.get(seat_key, self.agent_registry.default_profile),
            )
            strategy = AdaptivePlayStrategy(self.agent_registry.get(profile_id), self.bundle.convention)
            engine.play_card(
                final,
                strategy.choose_card(
                    engine,
                    final,
                    seed=seed + 991 + len(final.played_cards) * 65537 + int(final.current_player),
                    cancel_check=cancel_check,
                ),
            )
        record_hash = _hash(
            {
                "bundleHash": self.bundle.sha256,
                "seed": final.seed,
                "events": [event.to_dict() for event in final.events],
                "score": final.round_score.to_dict(),
            }
        )
        return replace(
            episode,
            schema_version=max(5, episode.schema_version),
            bundle_hash=self.bundle.sha256,
            final_state=final,
            analysis=result,
            game_record_hash=record_hash,
            game_serial=game_serial(record_hash),
        )

    def _require_legal_replay(self, episode: EpisodeSpec) -> None:
        if not bool(self.profile.get("confirmation", {}).get("require_legal_replay", False)):
            return
        restored = episode_spec_from_dict(episode.to_dict(), self.bundle.rules)
        engine = BalootEngine(self.bundle.rules)
        if engine.state_hash(restored.final_state) != engine.state_hash(episode.final_state):
            raise ValueError("confirmed challenge differs from authoritative legal replay")

    def _confirm_stability(
        self,
        episode: EpisodeSpec,
        seed: int,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        policy = dict(self.profile.get("stability") or {})
        required_budgets = (
            "reanalysis_samples",
            "maximum_sampling_attempts",
            "exact_allocation_limit",
            "exact_search_remaining",
            "exact_search_max_nodes",
            "exact_search_max_seconds",
            "seed_offset",
        )
        missing = [
            key for key in required_budgets if key not in policy
        ]
        if missing:
            raise ValueError(
                "challenge stability budget is missing "
                + ", ".join(sorted(missing))
            )
        info = InformationState.from_complete(
            episode.state_before_pause,
            episode.viewer,
        )
        confirmation = analyze(
            info,
            self.bundle.rules,
            self.bundle.convention,
            requested_samples=int(policy["reanalysis_samples"]),
            exact_allocation_limit=int(
                policy["exact_allocation_limit"]
            ),
            exact_search_max_remaining=int(
                policy["exact_search_remaining"]
            ),
            maximum_sampling_attempts=int(
                policy["maximum_sampling_attempts"]
            ),
            exact_search_max_nodes=int(
                policy["exact_search_max_nodes"]
            ),
            exact_search_max_seconds=float(
                policy["exact_search_max_seconds"]
            ),
            seed=seed + int(policy["seed_offset"]),
            cancel_check=cancel_check,
        )
        result = self._stability_metrics(
            episode.analysis,
            confirmation,
            policy,
        )
        result.update(
            {
                "firstAnalysisSha256": _hash(
                    episode.analysis.to_dict()
                ),
                "reanalysisSha256": _hash(
                    confirmation.to_dict()
                ),
                "reanalysisSeed": seed + int(policy["seed_offset"]),
            }
        )
        return result

    def _bid_fingerprint(self, episode: BidEpisodeSpec) -> str:
        identity = {
            "fingerprintVersion": int(self.profile.get("diversity", {}).get("fingerprint_version", 1)),
            "algorithmVersion": 1,
            "decisionKind": "bid",
            "viewer": episode.viewer.name.lower(),
            "dealer": episode.state_at_bid.dealer.name.lower(),
            "purchase": episode.state_at_bid.purchase_card.id if episode.state_at_bid.purchase_card else None,
            "hand": episode.state_at_bid.initial_five[int(episode.viewer)],
            "history": [item.to_dict() for item in episode.bidding_history_before],
            "legal": [item.choice.id for item in episode.analysis.choices],
            "matchScore": list(episode.state_at_bid.match_score),
        }
        return _hash(identity)

    @staticmethod
    def _bid_vector(episode: BidEpisodeSpec) -> InterestingnessVector:
        outcomes = [
            ActionOutcome(
                action_id=item.choice.id,
                expected_match_utility=math.tanh(item.expected_match_margin / 55.0),
                # Bidding analysis currently estimates round outcomes only.
                # Keep match probability neutral until true match-to-152
                # evaluation is available instead of mislabeling round lead.
                match_win_probability=0.5,
                expected_round_margin=item.expected_match_margin,
                downside_cvar=math.tanh(item.downside_cvar10 / 55.0),
                best_deal_share=item.best_share,
                hidden_deal_values=(),
                is_obvious_heuristic_choice=item.choice.action == "pass",
                explanation_features=("buy_threshold", "contract_selection", "score_pressure"),
            )
            for item in episode.analysis.choices
        ]
        final = episode.final_state
        context = ContextFeatures(
            decision_kind="bid",
            contract=final.contract.value if final.contract else None,
            buyer_relationship=(
                "self" if final.buyer == episode.viewer
                else "partner" if final.buyer == episode.viewer.partner
                else "opponent"
            ),
            opponent_bought=final.buyer is not None and final.buyer.team != episode.viewer.team,
            score_pressure=min(1.0, abs(episode.state_at_bid.match_score[0] - episode.state_at_bid.match_score[1]) / 152.0),
            tricks_remaining=8,
            counterintuitive=episode.analysis.recommended.action == "pass",
            project_relevant=bool(final.projects),
            escalation_relevant=final.escalation != "normal",
            tags=("bid", final.contract.value if final.contract else "all_pass"),
        )
        return score_interestingness(outcomes, context)

    def _mine_bid(self, request: MiningRequest) -> list[dict[str, Any]]:
        start = monotonic()
        accepted: list[dict[str, Any]] = []
        seen = {item["fingerprint"] for item in self.store.list_challenges(limit=5000)}
        stats: Counter[str] = Counter()
        for offset in range(max(1, int(request.seeds))):
            if monotonic() - start > request.maximum_seconds:
                break
            stats["seedsAttempted"] += 1
            seed = request.seed_start + offset
            try:
                episode = generate_bid_episode(
                    self.bundle.rules,
                    bundle_hash=self.bundle.sha256,
                    copy=self.bundle.video["localizations"],
                    seed=seed,
                    shallow_samples=max(6, int(request.shallow_samples if request.shallow_samples is not None else self.profile.get("screening", {}).get("shallow_samples", 12))),
                    deep_samples=max(24, int(request.deep_samples if request.deep_samples is not None else self.profile.get("confirmation", {}).get("deep_samples", 96))),
                    maximum_seed_attempts=1,
                    minimum_margin_swing=0.0,
                    minimum_win_swing=0.0,
                )
            except Exception:
                stats["generationRejected"] += 1
                continue
            vector = self._bid_vector(episode)
            verification = self._verify_bid_contract(episode, vector, request.content_contract)
            if not verification.accepted:
                stats[f"contractRejected:{verification.contract.value}"] += 1
                continue
            if not (request.difficulty_min <= vector.difficulty_prior <= request.difficulty_max):
                stats["difficultyRejected"] += 1
                continue
            if vector.publishability_score < request.minimum_publishability:
                stats["publishabilityRejected"] += 1
                continue
            if request.archetype not in {"any", "", "buy_or_pass"}:
                stats["archetypeRejected"] += 1
                continue
            fingerprint = self._bid_fingerprint(episode)
            if fingerprint in seen:
                stats["duplicateRejected"] += 1
                continue
            seen.add(fingerprint)
            challenge_id = f"CHL-{fingerprint[:20].upper()}"
            episode = replace(episode, challenge_id=challenge_id)
            choices = list(episode.analysis.choices)
            best, worst = choices[0], choices[-1]
            payload = self._pin_publication_policy({
                "challengeId": challenge_id,
                "fingerprint": fingerprint,
                "archetype": "buy_or_pass",
                "source": "simulation",
                "contentContract": verification.to_dict(),
                "publishability": vector.publishability_score,
                "difficulty": vector.difficulty_prior,
                "interestingness": {
                    "regretGap": vector.regret_gap,
                    "winProbabilityGap": vector.win_probability_gap,
                    "disasterGap": vector.disaster_gap,
                    "cliffScore": vector.cliff_score,
                    "debateScore": vector.debate_score,
                    "comebackScore": vector.comeback_score,
                    "surpriseScore": vector.surprise_score,
                    "educationalScore": vector.educational_score,
                    "robustnessScore": vector.robustness_score,
                    "archetypeScores": dict(vector.archetype_scores),
                    "reasons": list(vector.reasons),
                },
                "episodeId": episode.episode_id,
                "gameSerial": episode.game_serial,
                "gameRecordHash": episode.game_record_hash,
                "decision": {
                    "kind": "bid",
                    "recommendedAction": best.choice.to_dict(),
                    "legalActions": [item.choice.to_dict() for item in choices],
                    "best": best.to_dict(),
                    "worst": worst.to_dict(),
                    "roundLeadProbabilitySwing": best.team_win_probability - worst.team_win_probability,
                    "expectedRoundMarginSwing": best.expected_match_margin - worst.expected_match_margin,
                    "analysis": episode.analysis.to_dict(),
                },
            })
            self.store.put_challenge(
                challenge_id=challenge_id,
                decision_id=episode.decision_id,
                fingerprint=fingerprint,
                archetype="buy_or_pass",
                publishability=vector.publishability_score,
                payload=payload,
                episode=episode,
                status="candidate",
            )
            accepted.append(payload)
            stats["accepted"] += 1
            if len(accepted) >= request.maximum_results:
                break
        self.last_run = {
            "source": "simulation",
            "decisionKind": "bid",
            "durationSeconds": round(monotonic() - start, 3),
            "stats": dict(stats),
            "acceptedIds": [item["challengeId"] for item in accepted],
        }
        return accepted

    def mine(self, request: MiningRequest) -> list[dict[str, Any]]:
        decision_kind = str(request.decision_kind or "play").lower()
        if decision_kind == "bid" or request.archetype == "buy_or_pass":
            if request.source == "history":
                return self._mine_history(request)
            return self._mine_bid(request)
        if decision_kind not in {"play", "any"}:
            raise ValueError(f"unsupported decision kind {decision_kind}")
        if request.source == "history":
            return self._mine_history(request)
        if request.source != "simulation":
            raise ValueError(f"unsupported challenge source {request.source}")

        start = monotonic()
        screening = self.profile.get("screening", {})
        limits = self.profile.get("resource_limits", {})
        requested_seeds = min(max(1, int(request.seeds)), int(limits.get("maximum_seeds", 5000)))
        maximum_seconds = min(float(request.maximum_seconds), float(limits.get("maximum_seconds", 7200)))
        deadline = start + maximum_seconds

        def cancel_check() -> bool:
            return monotonic() >= deadline

        maximum_results = max(1, int(request.maximum_results))
        contract = self._contract(request.contract)
        candidates: list[tuple[float, EpisodeSpec]] = []
        stats: Counter[str] = Counter()
        stability_rejections: list[dict[str, Any]] = []

        for offset in range(requested_seeds):
            if monotonic() - start > maximum_seconds:
                stats["deadlineStopped"] = 1
                break
            stats["seedsAttempted"] += 1
            seed = request.seed_start + offset
            try:
                episode = generate_episode(
                    self.bundle.rules,
                    self.bundle.convention,
                    bundle_hash=self.bundle.sha256,
                    copy=self.bundle.video["localizations"],
                    request=ScenarioRequest.governed(
                        seed=seed,
                        contract=contract,
                        require_project=request.require_project,
                        pause_trick_min=request.pause_trick_min,
                        pause_trick_max=request.pause_trick_max,
                        requested_samples=int(request.shallow_samples if request.shallow_samples is not None else screening.get("shallow_samples", 12)),
                        exact_allocation_limit=int(request.exact_allocation_limit if request.exact_allocation_limit is not None else screening.get("shallow_exact_allocation_limit", 800)),
                        exact_search_max_remaining=int(request.exact_search_remaining if request.exact_search_remaining is not None else screening.get("shallow_exact_search_remaining", 5)),
                        maximum_sampling_attempts=int(screening.get("maximum_sampling_attempts", 8_000)),
                        exact_search_max_nodes=int(screening.get("exact_search_max_nodes", 10_000)),
                        exact_search_max_seconds=float(screening.get("exact_search_max_seconds", 0.03)),
                        max_seed_attempts=1,
                        max_candidates_per_seed=int(screening.get("maximum_candidates_per_seed", 4)),
                        minimum_hidden_deals=1,
                        minimum_expected_margin_gap=0.0,
                        minimum_win_probability_gap=0.0,
                        target_debate_score=65,
                        agent_profiles=request.player_profiles,
                    ),
                    agent_registry=self.agent_registry,
                    cancel_check=cancel_check,
                )
            except InterruptedError:
                stats["deadlineStopped"] = 1
                break
            except TimeoutError:
                raise
            except Exception:
                stats["generationRejected"] += 1
                continue
            stats["generated"] += 1
            if len(episode.analysis.actions) < int(screening.get("minimum_legal_actions", 2)):
                stats["legalActionCountRejected"] += 1
                continue
            if not self._relationship_matches(episode, request.buyer_relationship):
                stats["buyerFilterRejected"] += 1
                continue
            vector = self._vector(episode)
            cheap = max(
                vector.cliff_score,
                vector.debate_score,
                vector.comeback_score,
                vector.surprise_score,
                vector.publishability_score,
            )
            candidates.append((cheap, episode))

        retain_fraction = max(0.01, min(1.0, float(screening.get("retain_fraction", 0.15))))
        retain = min(
            len(candidates),
            max(maximum_results * 2, math.ceil(len(candidates) * retain_fraction)),
        )
        candidates.sort(key=lambda item: item[0], reverse=True)
        stats["screenedCandidates"] = len(candidates)
        stats["deepRetained"] = retain

        accepted: list[MinedChallenge] = []
        seen = {item["fingerprint"] for item in self.store.list_challenges(limit=5000)}
        publication = self.profile.get("publication", {})
        minimum_robustness = float(publication.get("minimum_robustness", 0.35))
        max_per_archetype = int(self.profile.get("diversity", {}).get("maximum_per_archetype_per_batch", 20))
        archetype_counts: Counter[str] = Counter()

        for index, (_cheap, episode) in enumerate(candidates[:retain]):
            if monotonic() - start > maximum_seconds:
                stats["deadlineStopped"] = 1
                break
            stats["deepAnalyses"] += 1
            try:
                deep = self._deep_confirm(
                    episode,
                    request.seed_start * 1009 + index,
                    request=request,
                    cancel_check=cancel_check,
                )
                self._require_legal_replay(deep)
                stability = self._confirm_stability(
                    deep,
                    request.seed_start * 1009 + index,
                    cancel_check=cancel_check,
                )
            except InterruptedError:
                stats["deadlineStopped"] = 1
                break
            except TimeoutError:
                raise
            except Exception:
                stats["deepRejected"] += 1
                continue
            if not stability["accepted"] and not stability[
                "publishUncertain"
            ]:
                stats["stabilityRejected"] += 1
                for reason in stability["reasons"]:
                    stats[f"stabilityRejected:{reason}"] += 1
                if len(stability_rejections) < 100:
                    stability_rejections.append(
                        {
                            "episodeId": deep.episode_id,
                            "stateSha256": BalootEngine(
                                self.bundle.rules
                            ).state_hash(deep.state_before_pause),
                            "stability": stability,
                        }
                    )
                continue
            vector = self._vector(deep)
            verification = self._verify_play_contract(deep, vector, request.content_contract)
            if not verification.accepted:
                stats[f"contractRejected:{verification.contract.value}"] += 1
                continue
            archetype = self._archetype(vector, request.archetype)
            if not (request.difficulty_min <= vector.difficulty_prior <= request.difficulty_max):
                stats["difficultyRejected"] += 1
                continue
            if vector.publishability_score < request.minimum_publishability:
                stats["publishabilityRejected"] += 1
                continue
            if vector.robustness_score < minimum_robustness:
                stats["robustnessRejected"] += 1
                continue
            if not self._passes_requested_archetype(vector, request.archetype):
                stats["archetypeRejected"] += 1
                continue
            if archetype_counts[archetype] >= max_per_archetype:
                stats["diversityRejected"] += 1
                continue
            fingerprint = self._fingerprint(deep)
            if fingerprint in seen:
                stats["duplicateRejected"] += 1
                continue
            seen.add(fingerprint)
            archetype_counts[archetype] += 1
            challenge_id = f"CHL-{fingerprint[:20].upper()}"
            deep = replace(deep, challenge_id=challenge_id)
            item = MinedChallenge(
                challenge_id,
                fingerprint,
                archetype,
                vector,
                deep,
                "simulation",
                verification,
                stability,
            )
            payload = self._pin_publication_policy(item.to_dict())
            self.store.put_challenge(
                challenge_id=challenge_id,
                decision_id=deep.decision_id,
                fingerprint=fingerprint,
                archetype=archetype,
                publishability=vector.publishability_score,
                payload=payload,
                episode=deep,
                status="candidate",
            )
            accepted.append(item)
            stats["accepted"] += 1
            if len(accepted) >= maximum_results:
                break

        self.last_run = {
            "source": "simulation",
            "durationSeconds": round(monotonic() - start, 3),
            "request": request.__dict__ if hasattr(request, "__dict__") else {
                field: getattr(request, field) for field in request.__dataclass_fields__
            },
            "stats": dict(stats),
            "acceptedIds": [item.challenge_id for item in accepted],
            "stabilityRejections": stability_rejections,
        }
        return [item.to_dict() for item in accepted]

    def _mine_history(self, request: MiningRequest) -> list[dict[str, Any]]:
        if self.learning is None:
            raise RuntimeError("history mining requires LearningService")
        start = monotonic()
        accepted: list[dict[str, Any]] = []
        seen = {item["fingerprint"] for item in self.store.list_challenges(limit=5000)}
        stats: Counter[str] = Counter()
        requested_kind = str(request.decision_kind or "play").lower()
        for queue_item in self.store.learning_queue(limit=max(100, request.seeds)):
            if monotonic() - start > request.maximum_seconds:
                break
            decision = queue_item["decision"]
            stats["decisionsExamined"] += 1
            kind = str(decision.get("decisionKind", "play"))
            if kind != requested_kind:
                stats["kindSkipped"] += 1
                continue
            try:
                analysis = self.learning.reanalyze_decision(
                    decision["decisionId"],
                    samples=int(request.deep_samples or 96),
                    exact_allocation_limit=int(request.exact_allocation_limit or 20_000),
                    exact_search_remaining=int(request.exact_search_remaining or 10),
                )
                if kind == "bid":
                    episode = self.learning.create_bid_episode_from_decision(
                        decision["decisionId"], analysis_id=analysis["analysisId"]
                    )
                    vector = self._bid_vector(episode)
                    verification = self._verify_bid_contract(episode, vector, request.content_contract)
                    if not verification.accepted:
                        stats[f"contractRejected:{verification.contract.value}"] += 1
                        continue
                    if request.archetype not in {"any", "", "buy_or_pass"}:
                        stats["archetypeRejected"] += 1
                        continue
                    fingerprint = self._bid_fingerprint(episode)
                    archetype = "buy_or_pass"
                else:
                    episode = self.learning.create_episode_from_decision(
                        decision["decisionId"], analysis_id=analysis["analysisId"]
                    )
                    stability = self._confirm_stability(
                        episode,
                        request.seed_start + stats["decisionsExamined"],
                    )
                    if not stability["accepted"] and not stability[
                        "publishUncertain"
                    ]:
                        stats["stabilityRejected"] += 1
                        continue
                    if not self._relationship_matches(episode, request.buyer_relationship):
                        stats["buyerFilterRejected"] += 1
                        continue
                    vector = self._vector(episode)
                    verification = self._verify_play_contract(episode, vector, request.content_contract)
                    if not verification.accepted:
                        stats[f"contractRejected:{verification.contract.value}"] += 1
                        continue
                    if not self._passes_requested_archetype(vector, request.archetype):
                        stats["archetypeRejected"] += 1
                        continue
                    fingerprint = self._fingerprint(episode)
                    archetype = self._archetype(vector, request.archetype)
            except TimeoutError:
                raise
            except Exception:
                stats["analysisRejected"] += 1
                continue
            if not (request.difficulty_min <= vector.difficulty_prior <= request.difficulty_max):
                stats["difficultyRejected"] += 1
                continue
            if vector.publishability_score < request.minimum_publishability:
                stats["publishabilityRejected"] += 1
                continue
            if fingerprint in seen:
                stats["duplicateRejected"] += 1
                continue
            seen.add(fingerprint)
            challenge_id = f"CHL-{fingerprint[:20].upper()}"
            episode = replace(episode, challenge_id=challenge_id)
            if kind == "bid":
                choices = list(episode.analysis.choices)
                best, worst = choices[0], choices[-1]
                payload = self._pin_publication_policy({
                    "challengeId": challenge_id,
                    "fingerprint": fingerprint,
                    "archetype": archetype,
                    "source": "history",
                    "contentContract": verification.to_dict(),
                    "publishability": vector.publishability_score,
                    "difficulty": vector.difficulty_prior,
                    "interestingness": {
                        "regretGap": vector.regret_gap,
                        "winProbabilityGap": vector.win_probability_gap,
                        "disasterGap": vector.disaster_gap,
                        "cliffScore": vector.cliff_score,
                        "debateScore": vector.debate_score,
                        "comebackScore": vector.comeback_score,
                        "surpriseScore": vector.surprise_score,
                        "educationalScore": vector.educational_score,
                        "robustnessScore": vector.robustness_score,
                        "archetypeScores": dict(vector.archetype_scores),
                        "reasons": list(vector.reasons),
                    },
                    "episodeId": episode.episode_id,
                    "gameSerial": episode.game_serial,
                    "gameRecordHash": episode.game_record_hash,
                    "decision": {
                        "kind": "bid",
                        "recommendedAction": best.choice.to_dict(),
                        "legalActions": [item.choice.to_dict() for item in choices],
                        "best": best.to_dict(),
                        "worst": worst.to_dict(),
                        "roundLeadProbabilitySwing": best.team_win_probability - worst.team_win_probability,
                        "expectedRoundMarginSwing": best.expected_match_margin - worst.expected_match_margin,
                        "analysis": episode.analysis.to_dict(),
                    },
                })
            else:
                item = MinedChallenge(
                    challenge_id,
                    fingerprint,
                    archetype,
                    vector,
                    episode,
                    "history",
                    verification,
                    stability,
                )
                payload = self._pin_publication_policy(item.to_dict())
            self.store.put_challenge(
                challenge_id=challenge_id,
                decision_id=decision["decisionId"],
                fingerprint=fingerprint,
                archetype=archetype,
                publishability=vector.publishability_score,
                payload=payload,
                episode=episode,
                status="candidate",
            )
            accepted.append(payload)
            stats["accepted"] += 1
            if len(accepted) >= request.maximum_results:
                break
        self.last_run = {
            "source": "history",
            "decisionKind": requested_kind,
            "durationSeconds": round(monotonic() - start, 3),
            "stats": dict(stats),
            "acceptedIds": [item["challengeId"] for item in accepted],
        }
        return accepted
