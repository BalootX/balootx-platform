"""Bounded actor-safe CP-SAT selection for one public auction decision.

This is intentionally a decision-local Research accelerator, not a claim of
globally optimal bidding.  Every legal public action is evaluated on the same
observer-safe hidden-deal cohort, and CP-SAT chooses the best fixed-point
objective with a deterministic tie-break.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from baloot_platform.analysis.bidding import analyze_bid, legal_bid_choices
from baloot_platform.analysis.decision_scores import (
    BID_DECISION_OBJECTIVE_V1,
    BID_DECISION_TIE_BREAK_V1,
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
    bid_score_rows_v1,
    legal_action_set_hash_v1,
    primary_decision_score_evidence_v1,
)
from baloot_platform.challenge.analysis_pipeline import TraceAnalysisProfileV1
from baloot_platform.challenge.decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.challenge.simulation import observer_safe_bid_state_v1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.enums import Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .contracts import ActorSafeCpSatAuctionProfileV1, canonical_sha256


_PRIMARY_COHORT_INDEX = 0


class ActorSafeCpSatAuctionSelectorV1:
    descriptor = PluginDescriptor(
        id="actor_safe_cp_sat_auction_selector_v1",
        kind="research_auction_selector",
        version="1.1.0",
        description=(
            "Decision-local CP-SAT auction selection over all legal actions "
            "using one paired observer-safe hidden-deal cohort"
        ),
        capabilities=(
            "observer_safe",
            "all_legal_bid_actions",
            "paired_hidden_deal_cohort",
            "canonical_decision_score_evidence",
            "primary_actor_observer_cohort",
            "fixed_point_cp_sat_objective",
            "deterministic",
        ),
        resource_class="analysis-cpu",
        deterministic=True,
        timeout_seconds=30,
        maximum_memory_mb=1024,
    )

    def __init__(
        self,
        profile: ActorSafeCpSatAuctionProfileV1,
        *,
        exact_search_backend: Any | None = None,
        trace_analysis_profile: TraceAnalysisProfileV1 | None = None,
        analysis_identity_hash: str | None = None,
    ) -> None:
        if not isinstance(profile, ActorSafeCpSatAuctionProfileV1):
            raise TypeError(
                "actor-safe auction profile must be ActorSafeCpSatAuctionProfileV1"
            )
        if ActorSafeCpSatAuctionProfileV1.build(profile.to_dict()) != profile:
            raise ValueError("actor-safe auction profile integrity mismatch")
        self.profile = profile
        self.exact_search_backend = exact_search_backend
        self.trace_analysis_profile = trace_analysis_profile
        self.analysis_identity_hash = analysis_identity_hash
        self._validate_analysis_binding()

    def _validate_analysis_binding(self) -> None:
        profile = self.trace_analysis_profile
        identity_hash = self.analysis_identity_hash
        if (profile is None) != (identity_hash is None):
            raise ValueError(
                "trace analysis profile and analysis identity hash must be injected together"
            )
        if profile is None:
            return
        if not isinstance(profile, TraceAnalysisProfileV1):
            raise TypeError("trace analysis profile must be TraceAnalysisProfileV1")
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity_hash)):
            raise ValueError("analysis identity hash must be a canonical SHA-256")
        profile_payload = profile.to_dict()
        profile_hash = str(profile_payload.pop("sha256", ""))
        if (
            profile_hash != profile.sha256
            or canonical_sha256(profile_payload) != profile_hash
        ):
            raise ValueError("trace analysis profile checksum mismatch")
        descriptor = getattr(self.exact_search_backend, "descriptor", None)
        backend_id = str(getattr(descriptor, "id", ""))
        if not backend_id:
            raise ValueError(
                "trace analysis binding requires an explicit exact-search backend"
            )
        if backend_id != profile.search_backend_id:
            raise ValueError(
                "trace analysis profile search backend does not match the injected backend"
            )
        if profile.bid_samples != self.profile.requested_samples:
            raise ValueError(
                "strategic auction requestedSamples must mirror the trace analysis bidSamples"
            )

    @property
    def analysis_requested_samples(self) -> int:
        if self.trace_analysis_profile is not None:
            return self.trace_analysis_profile.bid_samples
        return self.profile.requested_samples

    def choose_with_trace(
        self,
        state: RoundState,
        machine: BiddingMachine,
        rules: RuleBook,
        *,
        seed: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[str, Suit | None, dict[str, Any]]:
        from ortools.sat.python import cp_model

        actor = machine.actor
        if actor is None:
            raise ValueError("auction has no current actor")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("auction decision seed must be an integer")
        if self.trace_analysis_profile is None or self.analysis_identity_hash is None:
            raise ValueError(
                "canonical auction decision-score evidence requires frozen deep "
                "analysis authority"
            )
        if state.rules_hash != rules.sha256:
            raise ValueError(
                "auction state rules do not match the decision-score authority"
            )
        public_prefix_hash = canonical_sha256(
            observer_safe_bid_state_v1(state, machine, actor)
        )
        expected_seed = derive_actor_observer_cohort_seed_v1(
            public_prefix_hash=public_prefix_hash,
            decision_kind="bid",
            acting_seat=actor,
            cohort_authority_hash=self.analysis_identity_hash,
            cohort_index=_PRIMARY_COHORT_INDEX,
            governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
        )
        if seed != expected_seed:
            raise ValueError(
                "auction decision seed does not match the primary actor-observer cohort"
            )
        analysis = analyze_bid(
            state,
            machine,
            actor,
            rules,
            requested_samples=self.analysis_requested_samples,
            seed=seed,
            cancel_check=cancel_check,
            exact_search_backend=self.exact_search_backend,
        )
        evaluations = tuple(analysis.choices)
        if len(evaluations) < 2:
            raise ValueError("actor-safe auction selection requires alternatives")
        if analysis.actor != actor:
            raise ValueError("bid analysis actor does not match the auction actor")
        if analysis.hidden_deals != self.analysis_requested_samples:
            raise ValueError(
                "bid analysis hidden-deal support does not match the deep profile"
            )
        if any(item.samples != analysis.hidden_deals for item in evaluations):
            raise ValueError(
                "bid analysis row samples do not match its hidden-deal support"
            )

        canonical_rows = bid_score_rows_v1(evaluations)
        legal_choices = legal_bid_choices(machine)
        legal_actions = tuple(choice.to_dict() for choice in legal_choices)
        legal_action_set_hash = legal_action_set_hash_v1(legal_actions)
        score_evidence = primary_decision_score_evidence_v1(
            decision_kind="bid",
            actor=actor,
            public_prefix_hash=public_prefix_hash,
            rules_hash=rules.sha256,
            analysis_identity_hash=self.analysis_identity_hash,
            trace_analysis_profile_hash=self.trace_analysis_profile.sha256,
            cohort_seed_policy=PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
            cohort_index=_PRIMARY_COHORT_INDEX,
            cohort_seed=expected_seed,
            objective=BID_DECISION_OBJECTIVE_V1,
            tie_break=BID_DECISION_TIE_BREAK_V1,
            backend=self.trace_analysis_profile.search_backend_id,
            method=analysis.method,
            hidden_deal_support=analysis.hidden_deals,
            effective_sample_size=float(analysis.hidden_deals),
            recommended_action=analysis.recommended.to_dict(),
            rows=canonical_rows,
            legal_action_set_hash=legal_action_set_hash,
        )

        choices_by_id = {item.choice.id: item.choice for item in evaluations}
        if len(choices_by_id) != len(evaluations):
            raise ValueError("bid analysis contains duplicate actions")
        action_ids = tuple(choice.id for choice in legal_choices)
        stable_preference = {
            action_id: len(action_ids) - index
            for index, action_id in enumerate(action_ids)
        }
        action_span = len(action_ids)
        best_share_weight = action_span + 1
        round_lead_weight = 1_000 * best_share_weight + action_span + 1
        margin_weight = (
            1_000 * round_lead_weight + 1_000 * best_share_weight + action_span + 1
        )
        lexicographic_weights = {
            "fixedPointExpectedRoundMargin": margin_weight,
            "fixedPointRoundLeadProbability": round_lead_weight,
            "fixedPointBestShare": best_share_weight,
            "deterministicActionPreference": 1,
        }
        analysis_rows: list[dict[str, Any]] = []
        cp_sat_objective_rows: list[dict[str, Any]] = []
        objective_values: list[int] = []
        for item, row in zip(evaluations, score_evidence.rows, strict=True):
            if item.choice.id != row["actionId"]:
                raise ValueError(
                    "canonical bid-score row order does not match analysis"
                )
            analysis_rows.append(item.to_dict())
            margin = round(
                float(row["expectedRoundMargin"]) * self.profile.integer_scale
            )
            win = round(float(row["roundLeadProbability"]) * 1_000)
            best_share = round(float(row["bestShare"]) * 1_000)
            combined = (
                margin * margin_weight
                + win * round_lead_weight
                + best_share * best_share_weight
                + stable_preference[str(row["actionId"])]
            )
            objective_values.append(combined)
            cp_sat_objective_rows.append(
                {
                    "rank": int(row["rank"]),
                    "actionId": str(row["actionId"]),
                    "action": dict(row["action"]),
                    "fixedPointExpectedRoundMargin": margin,
                    "fixedPointRoundLeadProbability": win,
                    "fixedPointBestShare": best_share,
                    "deterministicActionPreference": stable_preference[
                        str(row["actionId"])
                    ],
                    "fixedPointSelectionObjective": combined,
                }
            )

        cp_sat_input_core = {
            "schemaVersion": 1,
            "authority": "primary_decision_score_evidence_v1",
            "decisionScoreEvidenceHash": score_evidence.sha256,
            "decisionScoreRowsHash": score_evidence.rows_hash,
            "legalActionSetHash": score_evidence.legal_action_set_hash,
            "integerScale": self.profile.integer_scale,
            "objective": BID_DECISION_OBJECTIVE_V1,
            "tieBreak": BID_DECISION_TIE_BREAK_V1,
            "lexicographicWeights": lexicographic_weights,
            "orderedFixedPointRows": cp_sat_objective_rows,
        }
        cp_sat_input_hash = canonical_sha256(cp_sat_input_core)

        model = cp_model.CpModel()
        selected = [
            model.new_bool_var(f"bid_{index}") for index in range(len(evaluations))
        ]
        model.add_exactly_one(selected)
        model.maximize(
            sum(
                objective_values[index] * selected[index]
                for index in range(len(evaluations))
            )
        )
        proof_core = {
            "schemaVersion": 1,
            "profileHash": self.profile.sha256,
            "traceAnalysisProfileHash": (
                self.trace_analysis_profile.sha256
                if self.trace_analysis_profile is not None
                else None
            ),
            "analysisIdentityHash": self.analysis_identity_hash,
            "bidObjective": BID_DECISION_OBJECTIVE_V1,
            "playObjective": (
                self.trace_analysis_profile.objective
                if self.trace_analysis_profile is not None
                else None
            ),
            "analysisRequestedSamples": self.analysis_requested_samples,
            "publicPrefixHash": public_prefix_hash,
            "primaryDecisionScoreEvidenceHash": score_evidence.sha256,
            "decisionScoreRowsHash": score_evidence.rows_hash,
            "legalActionSetHash": score_evidence.legal_action_set_hash,
            "cpSatCanonicalInputHash": cp_sat_input_hash,
            "auctionDomainHash": machine.domain.sha256,
            "actor": actor.name.lower(),
            "roundIndex": machine.round_index,
            "purchaseSuit": machine.purchase_suit.name.lower(),
            "publicActions": [
                {
                    "roundIndex": item.round_index,
                    "actor": item.seat.name.lower(),
                    "action": item.action,
                    "trumpSuit": (
                        item.trump.name.lower() if item.trump is not None else None
                    ),
                }
                for item in machine.records
            ],
            "actorHand": sorted(card.id for card in state.hand(actor)),
            "purchaseCard": state.purchase_card.id if state.purchase_card else None,
            "matchScore": list(state.match_score),
            "analysisMethod": analysis.method,
            "hiddenDealCount": analysis.hidden_deals,
            "canonicalDecisionScoreRows": [dict(row) for row in score_evidence.rows],
            "cpSatCanonicalInput": cp_sat_input_core,
        }
        model_hash = canonical_sha256(proof_core)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_number_of_conflicts = self.profile.maximum_conflicts
        solver.parameters.max_time_in_seconds = self.profile.maximum_selection_seconds
        status = solver.solve(model)
        status_name = {
            cp_model.OPTIMAL: "optimal",
            cp_model.FEASIBLE: "feasible",
            cp_model.INFEASIBLE: "infeasible",
            cp_model.MODEL_INVALID: "model_invalid",
            cp_model.UNKNOWN: "unknown",
        }.get(status, "unknown")
        optimal = status == cp_model.OPTIMAL
        selected_index = (
            next(
                index
                for index, variable in enumerate(selected)
                if solver.boolean_value(variable)
            )
            if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
            else 0
        )
        selected_row = score_evidence.rows[selected_index]
        chosen = choices_by_id[str(selected_row["actionId"])]
        if chosen.to_dict() != score_evidence.recommended_action:
            raise ValueError(
                "CP-SAT fixed-point argmax does not match the canonical bid recommendation"
            )
        qualification_blocked = (
            self.profile.require_optimal_for_qualification and not optimal
        )
        trace = {
            "schemaVersion": 1,
            "source": "research_optimizer",
            "strategyId": self.descriptor.id,
            "strategyVersion": self.descriptor.version,
            "decisionSource": "paired_actor_safe_cp_sat_bid_objective_v1",
            "selectionReason": "highest_paired_expected_round_margin",
            "observerSafe": True,
            "seed": seed,
            "publicPrefixHash": public_prefix_hash,
            "cohortSeedPolicy": PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
            "cohortIndex": _PRIMARY_COHORT_INDEX,
            "cohortSeed": expected_seed,
            "legalActionCount": len(evaluations),
            "allLegalActionsEvaluated": True,
            "hiddenDealCount": analysis.hidden_deals,
            "analysisMethod": analysis.method,
            "analysisRequestedSamples": self.analysis_requested_samples,
            "traceAnalysisProfileHash": (
                self.trace_analysis_profile.sha256
                if self.trace_analysis_profile is not None
                else None
            ),
            "analysisIdentityHash": self.analysis_identity_hash,
            "bidObjective": BID_DECISION_OBJECTIVE_V1,
            "playObjective": (
                self.trace_analysis_profile.objective
                if self.trace_analysis_profile is not None
                else None
            ),
            "selectedAction": chosen.to_dict(),
            "recommendedAction": analysis.recommended.to_dict(),
            "rankedActions": analysis_rows,
            "primaryDecisionScoreEvidence": score_evidence.to_dict(),
            "primaryDecisionScoreEvidenceHash": score_evidence.sha256,
            "decisionScoreRowsHash": score_evidence.rows_hash,
            "legalActionSetHash": score_evidence.legal_action_set_hash,
            "cpSatObjectiveRows": cp_sat_objective_rows,
            "cpSatCanonicalInputHash": cp_sat_input_hash,
            "solverStatus": status_name,
            "cpSatStructuralObjectiveValue": (
                int(round(solver.objective_value))
                if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
                else None
            ),
            "cpSatStructuralBestObjectiveBound": (
                int(round(solver.best_objective_bound))
                if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
                else None
            ),
            "cpSatObjectiveScope": (
                "deterministic_fixed_point_argmax_only_not_baloot_score_or_"
                "decision_estimate"
            ),
            "modelHash": model_hash,
            "profile": self.profile.to_dict(),
            "qualificationBlocked": qualification_blocked,
            "globalOptimalityClaimed": False,
            "proofScope": "single_actor_safe_auction_decision",
        }
        return chosen.action, chosen.trump_suit, trace
