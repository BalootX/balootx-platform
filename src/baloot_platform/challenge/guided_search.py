"""Deterministic guided legal search over complete equal-agent games."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from baloot_platform.core.enums import Seat
from baloot_platform.plugins.base import PluginDescriptor

from .constraint_deal_search import DealConstraintSpecV1, OrToolsCpSatDealGeneratorV1
from .evidence_v3 import canonical_sha256
from .oracle import ActualDealOracleVerifierV1
from .simulation import NaturalLaffaSimulationResultV1, NaturalLaffaSimulatorV1


_COMPONENTS = frozenset({
    "contractMatch",
    "projectPresence",
    "escalationMatch",
    "decisionOpportunity",
    "scoreCloseness",
    "actualDealOutcomeGap",
})


@dataclass(frozen=True, slots=True)
class GuidedSearchProfileV1:
    profile_id: str
    revision: str
    agent_profile_id: str
    candidates_per_seed: int
    target_contracts: tuple[str, ...]
    target_escalations: tuple[str, ...]
    target_project_presence: bool
    target_mode: str
    require_projects: bool
    minimum_non_forced_decisions: int
    require_actual_deal_opportunity: bool
    minimum_actual_deal_margin_gap: float
    maximum_actual_deal_decisions: int
    minimum_score: float
    weights: tuple[tuple[str, float], ...]
    deal_constraint_spec: DealConstraintSpecV1 | None
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "GuidedSearchProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "candidatesPerSeed",
            "agentProfileId",
            "targetContracts", "targetEscalations", "requireProjects",
            "targetProjectPresence", "targetMode",
            "minimumNonForcedDecisions", "weights",
            "requireActualDealOpportunity", "minimumActualDealMarginGap",
            "maximumActualDealDecisions",
            "minimumScore",
            "dealConstraint",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown guided search profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported guided search profile schema version")
        candidates = int(value.get("candidatesPerSeed", 0))
        if not 1 <= candidates <= 1_000:
            raise ValueError("guided candidates per seed must be in [1, 1000]")
        contracts = tuple(sorted({str(item) for item in value.get("targetContracts") or ()}))
        if set(contracts) - {"sun", "hokum", "ashkal"}:
            raise ValueError("guided search contains an unsupported contract target")
        escalations = tuple(sorted({str(item) for item in value.get("targetEscalations") or ()}))
        if set(escalations) - {"normal", "double", "triple", "quadruple", "gahwa"}:
            raise ValueError("guided search contains an unsupported escalation target")
        target_mode = str(value.get("targetMode") or "any")
        if target_mode not in {"any", "all"}:
            raise ValueError("guided search target mode must be any or all")
        minimum_decisions = int(value.get("minimumNonForcedDecisions", 0))
        minimum_actual_gap = float(value.get("minimumActualDealMarginGap", 0.0))
        maximum_actual_decisions = int(value.get("maximumActualDealDecisions", 4))
        minimum_score = float(value.get("minimumScore", 0.0))
        if (
            minimum_decisions < 0
            or minimum_actual_gap < 0.0
            or not 1 <= maximum_actual_decisions <= 128
            or not 0.0 <= minimum_score <= 100.0
        ):
            raise ValueError("guided search decision and score gates are out of range")
        weights = tuple(sorted(
            (str(key), float(weight))
            for key, weight in dict(value.get("weights") or {}).items()
        ))
        if not weights or any(weight <= 0.0 for _, weight in weights):
            raise ValueError("guided search weights must be positive and non-empty")
        unsupported = {key for key, _ in weights} - _COMPONENTS
        if unsupported:
            raise ValueError(f"unsupported guided search components: {sorted(unsupported)}")
        core = {
            "schemaVersion": 1,
            "id": str(value["id"]),
            "revision": str(value["revision"]),
            "agentProfileId": str(value.get("agentProfileId") or "").strip(),
            "candidatesPerSeed": candidates,
            "targetContracts": list(contracts),
            "targetEscalations": list(escalations),
            "targetProjectPresence": bool(
                value.get("targetProjectPresence", False)
            ),
            "targetMode": target_mode,
            "requireProjects": bool(value.get("requireProjects", False)),
            "minimumNonForcedDecisions": minimum_decisions,
            "requireActualDealOpportunity": bool(
                value.get("requireActualDealOpportunity", False)
            ),
            "minimumActualDealMarginGap": minimum_actual_gap,
            "maximumActualDealDecisions": maximum_actual_decisions,
            "minimumScore": minimum_score,
            "weights": dict(weights),
        }
        constraint_spec = (
            DealConstraintSpecV1.build(value["dealConstraint"])
            if "dealConstraint" in value
            else None
        )
        if constraint_spec is not None:
            core["dealConstraint"] = {
                key: item
                for key, item in constraint_spec.to_dict().items()
                if key != "sha256"
            }
        if not core["agentProfileId"]:
            raise ValueError("guided search agent profile identity is required")
        return cls(
            core["id"], core["revision"], core["agentProfileId"],
            candidates, contracts, escalations,
            core["targetProjectPresence"], target_mode,
            core["requireProjects"], minimum_decisions,
            core["requireActualDealOpportunity"],
            minimum_actual_gap,
            maximum_actual_decisions,
            minimum_score,
            weights,
            constraint_spec,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "agentProfileId": self.agent_profile_id,
            "candidatesPerSeed": self.candidates_per_seed,
            "targetContracts": list(self.target_contracts),
            "targetEscalations": list(self.target_escalations),
            "targetProjectPresence": self.target_project_presence,
            "targetMode": self.target_mode,
            "requireProjects": self.require_projects,
            "minimumNonForcedDecisions": self.minimum_non_forced_decisions,
            "requireActualDealOpportunity": self.require_actual_deal_opportunity,
            "minimumActualDealMarginGap": self.minimum_actual_deal_margin_gap,
            "maximumActualDealDecisions": self.maximum_actual_deal_decisions,
            "minimumScore": self.minimum_score,
            "weights": dict(self.weights),
            "sha256": self.sha256,
        }
        if self.deal_constraint_spec is not None:
            result["dealConstraint"] = self.deal_constraint_spec.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class GuidedSearchResultV1:
    root_seed: int
    profile_hash: str
    evaluated: tuple[Mapping[str, Any], ...]
    selected_seed: int
    selected_trace_hash: str
    selected_score: float
    target_satisfied: bool
    provenance: str
    sha256: str
    selected_simulation: NaturalLaffaSimulationResultV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "rootSeed": self.root_seed,
            "profileHash": self.profile_hash,
            "evaluated": [dict(item) for item in self.evaluated],
            "selectedSeed": self.selected_seed,
            "selectedTraceHash": self.selected_trace_hash,
            "selectedScore": self.selected_score,
            "targetSatisfied": self.target_satisfied,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


class GuidedLegalSearchV1:
    descriptor = PluginDescriptor(
        id="guided_legal_search_v1",
        kind="generator",
        version="1.2.0",
        description="Bounded deterministic legal-game search with exact tactical opportunity screening",
        capabilities=(
            "complete_legal_games",
            "equal_frozen_agents",
            "bounded_constraint_search",
            "actual_deal_discovery_only",
            "observer_safe_confirmation_required",
            "candidate_provenance",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        simulator: NaturalLaffaSimulatorV1,
        profile: GuidedSearchProfileV1,
        opportunity_scanner: ActualDealOracleVerifierV1 | None = None,
        deal_generator: OrToolsCpSatDealGeneratorV1 | None = None,
    ):
        if simulator.agent_profile.id != profile.agent_profile_id:
            raise ValueError(
                "guided search simulator does not match its governed agent profile"
            )
        if profile.require_actual_deal_opportunity and opportunity_scanner is None:
            raise ValueError(
                "guided tactical opportunity screening requires an exact scanner"
            )
        if (
            profile.deal_constraint_spec is not None
            and profile.deal_constraint_spec.enabled
            and deal_generator is None
        ):
            raise ValueError("active guided deal constraints require a generator plugin")
        self.simulator = simulator
        self.profile = profile
        self.opportunity_scanner = opportunity_scanner
        self.deal_generator = deal_generator

    @staticmethod
    def _candidate_seed(root_seed: int, profile_hash: str, ordinal: int) -> int:
        digest = canonical_sha256({
            "schemaVersion": 1,
            "kind": "guided_candidate_seed_v1",
            "rootSeed": int(root_seed),
            "profileHash": profile_hash,
            "ordinal": int(ordinal),
        })
        return int(digest[:15], 16)

    def _evaluate(
        self,
        simulation: NaturalLaffaSimulationResultV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        trace = simulation.final_trace
        state = trace.trusted_final_state
        contract = str(state.get("contract") or "unknown")
        escalation = str(state.get("escalation") or "normal")
        project_count = len(state.get("projects") or ())
        non_forced = sum(1 for item in trace.decision_ledger.records if not item.forced)
        contract_match = (
            not self.profile.target_contracts
            or contract in self.profile.target_contracts
        )
        escalation_match = (
            not self.profile.target_escalations
            or escalation in self.profile.target_escalations
        )
        target_checks: list[bool] = []
        if self.profile.target_contracts:
            target_checks.append(contract_match)
        if self.profile.target_escalations:
            target_checks.append(escalation_match)
        if self.profile.target_project_presence:
            target_checks.append(project_count > 0)
        target_pass = (
            True
            if not target_checks
            else (
                all(target_checks)
                if self.profile.target_mode == "all"
                else any(target_checks)
            )
        )
        round_score = state.get("roundScore")
        if not simulation.publishable_laffa or not isinstance(round_score, Mapping):
            return {
                "candidateSeed": int(simulation.root_seed),
                "traceHash": trace.sha256,
                "score": 0.0,
                "hardConstraintsPassed": False,
                "components": {
                    "contractMatch": 0.0,
                    "projectPresence": 0.0,
                    "escalationMatch": 0.0,
                    "decisionOpportunity": 0.0,
                    "scoreCloseness": 0.0,
                    "actualDealOutcomeGap": 0.0,
                },
                "contract": contract,
                "escalation": escalation,
                "projectCount": project_count,
                "nonForcedDecisionCount": non_forced,
                "targetPassed": False,
                "matchedTargets": [],
                "actualDealOpportunities": [],
                "analysisStatus": "ineligible_all_pass",
            }
        awarded = tuple(int(item) for item in round_score["awardedPoints"])
        opportunities = (
            ()
            if self.opportunity_scanner is None
            else self.opportunity_scanner.discover_opportunities(
                trace,
                maximum_decisions=self.profile.maximum_actual_deal_decisions,
                cancel_check=cancel_check,
            )
        )
        peak_actual_gap = max(
            (item.outcome_gap for item in opportunities if item.complete),
            default=0.0,
        )
        actual_opportunity_match = (
            peak_actual_gap >= self.profile.minimum_actual_deal_margin_gap
        )
        components = {
            "contractMatch": (
                100.0 if contract_match else 0.0
            ),
            "projectPresence": (
                100.0 if project_count > 0 else 0.0
            ),
            "escalationMatch": (
                100.0 if escalation_match else 0.0
            ),
            "decisionOpportunity": min(
                100.0,
                100.0 * non_forced / max(1, self.profile.minimum_non_forced_decisions),
            ) if self.profile.minimum_non_forced_decisions else 100.0,
            "scoreCloseness": max(
                0.0,
                100.0 - 100.0 * abs(awarded[0] - awarded[1]) / max(1, sum(awarded)),
            ),
            "actualDealOutcomeGap": min(100.0, 5.0 * peak_actual_gap),
        }
        hard_pass = (
            (not self.profile.require_projects or project_count > 0)
            and non_forced >= self.profile.minimum_non_forced_decisions
            and target_pass
            and (
                not self.profile.require_actual_deal_opportunity
                or actual_opportunity_match
            )
        )
        weighted = sum(components[key] * weight for key, weight in self.profile.weights) / sum(
            weight for _, weight in self.profile.weights
        )
        return {
            "candidateSeed": int(simulation.root_seed),
            "traceHash": trace.sha256,
            "score": round(weighted if hard_pass else 0.0, 9),
            "hardConstraintsPassed": hard_pass,
            "components": components,
            "contract": contract,
            "escalation": escalation,
            "projectCount": project_count,
            "nonForcedDecisionCount": non_forced,
            "targetPassed": target_pass,
            "actualDealOpportunityPassed": actual_opportunity_match,
            "actualDealOpportunities": [
                item.to_dict() for item in opportunities
            ],
            "matchedTargets": [
                name
                for name, matched, enabled in (
                    ("contract", contract_match, bool(self.profile.target_contracts)),
                    ("escalation", escalation_match, bool(self.profile.target_escalations)),
                    ("project", project_count > 0, self.profile.target_project_presence),
                    (
                        "actual_deal_opportunity",
                        actual_opportunity_match,
                        self.profile.require_actual_deal_opportunity,
                    ),
                )
                if enabled and matched
            ],
            "analysisStatus": "completed",
        }

    @staticmethod
    def _constraint_buyer_matches(
        spec: DealConstraintSpecV1,
        simulation: NaturalLaffaSimulationResultV1,
    ) -> bool:
        return (
            spec.assumed_buyer is None
            or (
                simulation.publishable_laffa
                and str(simulation.final_trace.trusted_final_state.get("buyer"))
                == spec.assumed_buyer.name.lower()
            )
        )

    def search(
        self,
        root_seed: int,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> GuidedSearchResultV1:
        simulations: list[NaturalLaffaSimulationResultV1] = []
        rows: list[dict[str, Any]] = []
        constraint_spec = self.profile.deal_constraint_spec
        if constraint_spec is not None and constraint_spec.enabled:
            assert self.deal_generator is not None
            candidates = self.deal_generator.generate(
                constraint_spec,
                root_seed=int(root_seed),
                dealer=Seat.EAST,
                rules=self.simulator.rules,
            )
            if not candidates:
                raise RuntimeError("guided constraint search produced no legal deck candidates")
            for ordinal, candidate in enumerate(candidates[: self.profile.candidates_per_seed]):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("guided legal search cancelled")
                candidate_seed = self._candidate_seed(root_seed, self.profile.sha256, ordinal)
                simulation = self.simulator.simulate_preordered_deck(
                    candidate_seed,
                    candidate.deck_order,
                    deal_generator_provenance=candidate.provenance_dict(),
                    cancel_check=cancel_check,
                )
                row = self._evaluate(simulation, cancel_check=cancel_check)
                buyer_match = self._constraint_buyer_matches(
                    constraint_spec,
                    simulation,
                )
                row["constraintProfileHash"] = constraint_spec.sha256
                row["constraintCandidateHash"] = candidate.candidate_hash
                row["constraintBuyerMatched"] = buyer_match
                row["constraintTags"] = list(constraint_spec.tags)
                if not buyer_match:
                    row["hardConstraintsPassed"] = False
                    row["score"] = 0.0
                    row["analysisStatus"] = "constructed_buyer_mismatch"
                simulations.append(simulation)
                rows.append(row)
        else:
            for ordinal in range(self.profile.candidates_per_seed):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("guided legal search cancelled")
                candidate_seed = self._candidate_seed(root_seed, self.profile.sha256, ordinal)
                simulation = self.simulator.simulate(candidate_seed, cancel_check=cancel_check)
                simulations.append(simulation)
                rows.append(self._evaluate(simulation, cancel_check=cancel_check))
        ranked = sorted(
            zip(rows, simulations),
            key=lambda pair: (-float(pair[0]["score"]), str(pair[0]["traceHash"])),
        )
        selected_row, selected = ranked[0]
        evaluated = tuple(sorted(rows, key=lambda item: (int(item["candidateSeed"]), str(item["traceHash"]))))
        core = {
            "schemaVersion": 1,
            "rootSeed": int(root_seed),
            "profileHash": self.profile.sha256,
            "evaluated": [dict(item) for item in evaluated],
            "selectedSeed": int(selected_row["candidateSeed"]),
            "selectedTraceHash": str(selected_row["traceHash"]),
            "selectedScore": float(selected_row["score"]),
            "targetSatisfied": (
                bool(selected_row["hardConstraintsPassed"])
                and float(selected_row["score"]) >= self.profile.minimum_score
            ),
            "provenance": (
                "guided_constraint_legal_search"
                if constraint_spec is not None and constraint_spec.enabled
                else "guided_legal_search"
            ),
        }
        return GuidedSearchResultV1(
            int(root_seed),
            self.profile.sha256,
            evaluated,
            int(selected_row["candidateSeed"]),
            str(selected_row["traceHash"]),
            float(selected_row["score"]),
            bool(core["targetSatisfied"]),
            str(core["provenance"]),
            canonical_sha256(core),
            selected,
        )
