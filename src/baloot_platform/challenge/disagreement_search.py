"""Deterministic discovery of consequential shallow/deep policy disagreement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from baloot_platform.plugins.base import PluginDescriptor

from .analysis_pipeline import StagedTraceAnalyzerV1, TraceDecisionAnalysisV3
from .evidence_v3 import canonical_sha256
from .simulation import NaturalLaffaSimulationResultV1, NaturalLaffaSimulatorV1


_COMPONENTS = frozenset({
    "disagreementCount",
    "peakMarginGap",
    "peakProbabilityGap",
    "deepConfidence",
    "methodDiversity",
})


def _action_key(value: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(value))


@dataclass(frozen=True, slots=True)
class DisagreementSearchProfileV1:
    profile_id: str
    revision: str
    candidates_per_seed: int
    maximum_deep_decisions_per_candidate: int
    minimum_deep_confidence: float
    minimum_expected_margin_gap: float
    minimum_probability_gap: float
    minimum_score: float
    weights: tuple[tuple[str, float], ...]
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "DisagreementSearchProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "candidatesPerSeed",
            "maximumDeepDecisionsPerCandidate",
            "minimumDeepConfidence", "minimumExpectedMarginGap",
            "minimumProbabilityGap", "minimumScore", "weights",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown disagreement search profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported disagreement search profile schema version")
        candidates = int(value.get("candidatesPerSeed", 0))
        maximum_deep_decisions = int(
            value.get("maximumDeepDecisionsPerCandidate", 8)
        )
        confidence = float(value.get("minimumDeepConfidence", -1.0))
        margin = float(value.get("minimumExpectedMarginGap", -1.0))
        probability = float(value.get("minimumProbabilityGap", -1.0))
        minimum_score = float(value.get("minimumScore", -1.0))
        if not 1 <= candidates <= 100:
            raise ValueError("disagreement candidates per seed must be in [1, 100]")
        if not 1 <= maximum_deep_decisions <= 64:
            raise ValueError(
                "disagreement deep decisions per candidate must be in [1, 64]"
            )
        if not 0.0 <= confidence <= 1.0 or margin < 0.0:
            raise ValueError("disagreement confidence and margin gates are out of range")
        if not 0.0 <= probability <= 1.0 or not 0.0 <= minimum_score <= 100.0:
            raise ValueError("disagreement probability and score gates are out of range")
        weights = tuple(sorted(
            (str(key), float(weight))
            for key, weight in dict(value.get("weights") or {}).items()
        ))
        if not weights or any(weight <= 0.0 for _, weight in weights):
            raise ValueError("disagreement weights must be positive and non-empty")
        unsupported = {key for key, _ in weights} - _COMPONENTS
        if unsupported:
            raise ValueError(f"unsupported disagreement components: {sorted(unsupported)}")
        core = {
            "schemaVersion": 1,
            "id": str(value["id"]),
            "revision": str(value["revision"]),
            "candidatesPerSeed": candidates,
            "maximumDeepDecisionsPerCandidate": maximum_deep_decisions,
            "minimumDeepConfidence": confidence,
            "minimumExpectedMarginGap": margin,
            "minimumProbabilityGap": probability,
            "minimumScore": minimum_score,
            "weights": dict(weights),
        }
        return cls(
            core["id"], core["revision"], candidates, maximum_deep_decisions,
            confidence, margin,
            probability, minimum_score, weights, canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "candidatesPerSeed": self.candidates_per_seed,
            "maximumDeepDecisionsPerCandidate": (
                self.maximum_deep_decisions_per_candidate
            ),
            "minimumDeepConfidence": self.minimum_deep_confidence,
            "minimumExpectedMarginGap": self.minimum_expected_margin_gap,
            "minimumProbabilityGap": self.minimum_probability_gap,
            "minimumScore": self.minimum_score,
            "weights": dict(self.weights),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DisagreementSearchResultV1:
    root_seed: int
    profile_hash: str
    shallow_profile_hash: str
    deep_profile_hash: str
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
            "shallowProfileHash": self.shallow_profile_hash,
            "deepProfileHash": self.deep_profile_hash,
            "evaluated": [dict(item) for item in self.evaluated],
            "selectedSeed": self.selected_seed,
            "selectedTraceHash": self.selected_trace_hash,
            "selectedScore": self.selected_score,
            "targetSatisfied": self.target_satisfied,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


class DisagreementSearchV1:
    descriptor = PluginDescriptor(
        id="observer_safe_disagreement_search_v1",
        kind="generator",
        version="1.1.0",
        description="Consequential shallow/deep actor-observer disagreement discovery",
        capabilities=(
            "complete_legal_games",
            "paired_actor_observer_analysis",
            "shallow_deep_disagreement",
            "confidence_gated",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        simulator: NaturalLaffaSimulatorV1,
        shallow_analyzer: StagedTraceAnalyzerV1,
        deep_analyzer: StagedTraceAnalyzerV1,
        profile: DisagreementSearchProfileV1,
    ):
        if shallow_analyzer.profile.sha256 == deep_analyzer.profile.sha256:
            raise ValueError("disagreement search requires distinct shallow and deep profiles")
        if shallow_analyzer.rules.sha256 != deep_analyzer.rules.sha256:
            raise ValueError("disagreement analyzers must use the same rules")
        if shallow_analyzer.rules.sha256 != simulator.rules.sha256:
            raise ValueError("disagreement simulator and analyzers must use the same rules")
        self.simulator = simulator
        self.shallow_analyzer = shallow_analyzer
        self.deep_analyzer = deep_analyzer
        self.profile = profile

    @staticmethod
    def _candidate_seed(root_seed: int, profile_hash: str, ordinal: int) -> int:
        return int(canonical_sha256({
            "schemaVersion": 1,
            "kind": "disagreement_candidate_seed_v1",
            "rootSeed": int(root_seed),
            "profileHash": profile_hash,
            "ordinal": int(ordinal),
        })[:15], 16)

    @staticmethod
    def _alternative(
        decision: TraceDecisionAnalysisV3,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        key = _action_key(action)
        return next(
            (item for item in decision.alternatives if _action_key(item["action"]) == key),
            None,
        )

    @staticmethod
    def _deep_target_priority(
        decision: TraceDecisionAnalysisV3,
    ) -> tuple[float, int, str]:
        alternatives = tuple(decision.alternatives)
        consequence = 0.0
        if len(alternatives) >= 2:
            best, second = alternatives[:2]
            consequence = max(
                abs(
                    float(best.get("expectedMargin", 0.0))
                    - float(second.get("expectedMargin", 0.0))
                ) * 5.0,
                abs(
                    float(best.get("winProbability", 0.5))
                    - float(second.get("winProbability", 0.5))
                ) * 100.0,
            )
        priority = max(
            float(decision.difficulty),
            float(decision.debate_score),
            consequence,
        )
        return (-priority, int(decision.ordinal), str(decision.decision_id))

    def _evaluate(self, simulation: NaturalLaffaSimulationResultV1, *, cancel_check) -> dict[str, Any]:
        trace = simulation.final_trace
        shallow = self.shallow_analyzer.analyze_trace(trace, cancel_check=cancel_check)
        deep_target_ids = tuple(
            item.decision_id
            for item in sorted(
                (item for item in shallow.decisions if not item.forced),
                key=self._deep_target_priority,
            )[: self.profile.maximum_deep_decisions_per_candidate]
        )
        deep_decisions = self.deep_analyzer.analyze_decisions(
            trace,
            deep_target_ids,
            cancel_check=cancel_check,
        )
        deep_hash = canonical_sha256({
            "schemaVersion": 1,
            "traceHash": trace.sha256,
            "profileHash": self.deep_analyzer.profile.sha256,
            "decisionAnalysisHashes": [item.sha256 for item in deep_decisions],
        })
        shallow_by_id = {item.decision_id: item for item in shallow.decisions}
        disagreements: list[dict[str, Any]] = []
        for deep_item in deep_decisions:
            shallow_item = shallow_by_id.get(deep_item.decision_id)
            if shallow_item is None or deep_item.forced:
                continue
            if _action_key(shallow_item.best_action) == _action_key(deep_item.best_action):
                continue
            shallow_in_deep = self._alternative(deep_item, shallow_item.best_action)
            if shallow_in_deep is None:
                raise ValueError("shallow recommendation is absent from deep legal alternatives")
            deep_best = deep_item.alternatives[0]
            margin_gap = max(
                0.0,
                float(deep_best["expectedMargin"]) - float(shallow_in_deep["expectedMargin"]),
            )
            probability_gap = max(
                0.0,
                float(deep_best["winProbability"]) - float(shallow_in_deep["winProbability"]),
            )
            qualifies = (
                deep_item.confidence >= self.profile.minimum_deep_confidence
                and margin_gap >= self.profile.minimum_expected_margin_gap
                and probability_gap >= self.profile.minimum_probability_gap
            )
            disagreements.append({
                "decisionId": deep_item.decision_id,
                "actingSeat": deep_item.acting_seat,
                "decisionKind": deep_item.decision_kind,
                "shallowAction": dict(shallow_item.best_action),
                "deepAction": dict(deep_item.best_action),
                "deepConfidence": deep_item.confidence,
                "expectedMarginGap": margin_gap,
                "probabilityGap": probability_gap,
                "shallowMethod": shallow_item.method,
                "deepMethod": deep_item.method,
                "qualifies": qualifies,
            })
        qualifying = [item for item in disagreements if item["qualifies"]]
        peak_margin = max((float(item["expectedMarginGap"]) for item in qualifying), default=0.0)
        peak_probability = max((float(item["probabilityGap"]) for item in qualifying), default=0.0)
        peak_confidence = max((float(item["deepConfidence"]) for item in qualifying), default=0.0)
        method_diversity = any(
            item["shallowMethod"] != item["deepMethod"] for item in qualifying
        )
        components = {
            "disagreementCount": min(100.0, 25.0 * len(qualifying)),
            "peakMarginGap": min(100.0, 5.0 * peak_margin),
            "peakProbabilityGap": min(100.0, 100.0 * peak_probability),
            "deepConfidence": 100.0 * peak_confidence,
            "methodDiversity": 100.0 if method_diversity else 0.0,
        }
        score = sum(components[key] * weight for key, weight in self.profile.weights) / sum(
            weight for _, weight in self.profile.weights
        )
        return {
            "candidateSeed": simulation.root_seed,
            "traceHash": trace.sha256,
            "score": round(score if qualifying else 0.0, 9),
            "targetSatisfied": bool(qualifying) and score >= self.profile.minimum_score,
            "components": components,
            "disagreements": sorted(disagreements, key=lambda item: item["decisionId"]),
            "shallowAnalysisHash": shallow.sha256,
            "deepAnalysisHash": deep_hash,
            "deepTargetDecisionIds": list(deep_target_ids),
            "analysisStatus": "completed",
            "rejectionReason": None,
        }

    @staticmethod
    def _unresolved_hidden_deals(
        simulation: NaturalLaffaSimulationResultV1,
    ) -> dict[str, Any]:
        """Retain an honest zero-evidence row when conditioning is unsupported.

        A public project declaration can make constructive hidden-deal sampling
        too sparse for the governed budget. That is an unresolved candidate,
        not a retryable infrastructure failure and never positive evidence.
        """
        return {
            "candidateSeed": simulation.root_seed,
            "traceHash": simulation.final_trace.sha256,
            "score": 0.0,
            "targetSatisfied": False,
            "components": {
                "disagreementCount": 0.0,
                "peakMarginGap": 0.0,
                "peakProbabilityGap": 0.0,
                "deepConfidence": 0.0,
                "methodDiversity": 0.0,
            },
            "disagreements": [],
            "shallowAnalysisHash": None,
            "deepAnalysisHash": None,
            "deepTargetDecisionIds": [],
            "analysisStatus": "unresolved_hidden_deals",
            "rejectionReason": "public_history_conditioning_exhausted",
        }

    def search(
        self,
        root_seed: int,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> DisagreementSearchResultV1:
        pairs: list[tuple[dict[str, Any], NaturalLaffaSimulationResultV1]] = []
        for ordinal in range(self.profile.candidates_per_seed):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("disagreement search cancelled")
            seed = self._candidate_seed(root_seed, self.profile.sha256, ordinal)
            simulation = self.simulator.simulate(seed, cancel_check=cancel_check)
            try:
                evaluated = self._evaluate(
                    simulation, cancel_check=cancel_check
                )
            except RuntimeError as exc:
                if str(exc) != "no legal hidden deals could reproduce the public history":
                    raise
                evaluated = self._unresolved_hidden_deals(simulation)
            pairs.append((evaluated, simulation))
        ranked = sorted(
            pairs,
            key=lambda pair: (-float(pair[0]["score"]), str(pair[0]["traceHash"])),
        )
        selected_row, selected = ranked[0]
        evaluated = tuple(sorted(
            (item for item, _ in pairs),
            key=lambda item: (int(item["candidateSeed"]), str(item["traceHash"])),
        ))
        core = {
            "schemaVersion": 1,
            "rootSeed": int(root_seed),
            "profileHash": self.profile.sha256,
            "shallowProfileHash": self.shallow_analyzer.profile.sha256,
            "deepProfileHash": self.deep_analyzer.profile.sha256,
            "evaluated": [dict(item) for item in evaluated],
            "selectedSeed": int(selected_row["candidateSeed"]),
            "selectedTraceHash": str(selected_row["traceHash"]),
            "selectedScore": float(selected_row["score"]),
            "targetSatisfied": bool(selected_row["targetSatisfied"]),
            "provenance": "observer_safe_disagreement_search",
        }
        return DisagreementSearchResultV1(
            int(root_seed), self.profile.sha256,
            self.shallow_analyzer.profile.sha256,
            self.deep_analyzer.profile.sha256,
            evaluated, int(selected_row["candidateSeed"]),
            str(selected_row["traceHash"]), float(selected_row["score"]),
            bool(selected_row["targetSatisfied"]),
            "observer_safe_disagreement_search", canonical_sha256(core), selected,
        )
