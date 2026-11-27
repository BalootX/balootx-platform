from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from baloot_platform.config import build_bundle
from baloot_platform.version import __version__
from baloot_platform.plugins.base import PluginDescriptor

from .models import AuditedAlternative, DecisionAudit, GameAuditReport, action_key, canonical_action, content_hash
from .reasoning import actual_intent_summary, build_reasons, quality_label


class ReplayAuditService:
    descriptor = PluginDescriptor(
        "observer_safe_replay_audit_v1",
        "replay_audit",
        "1.0.0",
        "Observer-safe per-decision and full-game counterfactual audit with provenance-separated intent",
        ("decision_audit", "game_audit", "counterfactuals", "blunder_detection", "version_comparison"),
        "analysis-cpu",
        deterministic=True,
    )

    """Observer-safe replay audit and counterfactual explanation service.

    Effects are computed from legal counterfactuals. Intent is sourced only
    from the action-time policy trace or explicit human feedback; it is never
    reverse-engineered and presented as fact.
    """

    def __init__(self, root: str | Path, store: Any, learning_service: Any):
        self.root = Path(root)
        self.store = store
        self.learning = learning_service

    def _config(self) -> dict[str, Any]:
        return copy.deepcopy(build_bundle(self.root).component("audit"))

    @staticmethod
    def _candidate_action(candidate: dict[str, Any], kind: str) -> Any:
        if kind == "play":
            return candidate.get("card")
        return {"action": candidate.get("action"), "trumpSuit": candidate.get("trumpSuit")}

    @staticmethod
    def _metrics(candidate: dict[str, Any], kind: str) -> dict[str, float]:
        confidence = candidate.get("confidence95") or [0.0, 1.0]
        distribution = candidate.get("distribution") or {}
        return {
            "margin": float(candidate.get("expectedMatchMargin", 0.0)),
            "win": float(candidate.get("winProbability" if kind == "play" else "teamWinProbability", 0.5)),
            "low": float(confidence[0]) if len(confidence) > 0 else 0.0,
            "high": float(confidence[1]) if len(confidence) > 1 else 1.0,
            "cvar": float(distribution.get("downsideCvar10", candidate.get("downsideCvar10", 0.0))),
            "volatility": float(distribution.get("volatility", candidate.get("volatility", 0.0))),
            "bestShare": float(candidate.get("bestShare", 0.0)),
            "exactShare": float(candidate.get("exactShare", 0.0)),
        }

    def _analysis_for(
        self,
        decision_id: str,
        *,
        deep: bool,
        force: bool,
        analysis_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        config = self._config()
        analysis_cfg = copy.deepcopy(config.get("analysis", {}))
        if overrides:
            analysis_cfg.update({key: value for key, value in overrides.items() if value is not None})
        requested = int(analysis_cfg.get("requested_samples", 96))
        exact_limit = int(analysis_cfg.get("exact_allocation_limit", 20_000))
        exact_remaining = int(analysis_cfg.get("exact_search_remaining", 10))
        seed = int(analysis_cfg.get("seed", 1))
        use_conventions = bool(analysis_cfg.get("use_conventions", False))
        if not deep:
            requested = max(2, min(16, requested))
            exact_limit = max(100, min(3000, exact_limit))
            exact_remaining = max(4, min(7, exact_remaining))

        analyses = decision.get("analyses") or self.store.list_analyses(decision_id)
        if analysis_id:
            selected = next((item for item in analyses if item.get("analysisId") == analysis_id), None)
            if selected is None:
                raise ValueError(f"analysis {analysis_id} does not belong to decision {decision_id}")
            return selected
        if analyses and not force:
            for latest in analyses:
                latest_cfg = latest.get("config") or {}
                if (
                    int(latest_cfg.get("samples", 0)) >= requested
                    and int(latest_cfg.get("exactAllocationLimit", 0)) >= exact_limit
                    and int(latest_cfg.get("exactSearchRemaining", 0)) >= exact_remaining
                    and bool(latest_cfg.get("useConventions", False)) == use_conventions
                ):
                    return latest
        return self.learning.reanalyze_decision(
            decision_id,
            samples=requested,
            exact_allocation_limit=exact_limit,
            exact_search_remaining=exact_remaining,
            use_conventions=use_conventions,
            seed=seed,
        )

    def audit_decision(
        self,
        decision_id: str,
        *,
        deep: bool = True,
        force: bool = False,
        analysis_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        if decision.get("audits") and not force and deep and analysis_id is None and not overrides:
            return decision["audits"][0]
        legal_actions = list(decision.get("legalActions") or [])
        if len(legal_actions) <= 1:
            actual = decision.get("selectedAction")
            reasons = build_reasons(
                decision=decision,
                analysis={},
                actual_action=actual,
                best_action=actual,
                forced=True,
                regret=0.0,
                win_drop=0.0,
                catastrophic=False,
                maximum_reasons=int(self._config().get("reasoning", {}).get("maximum_reasons", 5)),
            )
            bundle_hash = str((decision.get("decisionMetadata") or {}).get("bundleHash") or build_bundle(self.root).sha256)
            audit = DecisionAudit(
                decision_id=decision_id,
                decision_hash=str(decision["decisionHash"]),
                game_id=str(decision["gameId"]),
                game_hash=decision.get("gameHash"),
                ordinal=int(decision["ordinal"]),
                decision_kind=str(decision["decisionKind"]),
                actor=str(decision["actor"]),
                phase=str(decision["phase"]),
                actual_action=canonical_action(actual),
                best_action=canonical_action(actual),
                selected_rank=1,
                quality_label="forced",
                quality_score=100.0,
                forced=True,
                expected_regret=0.0,
                win_probability_drop=0.0,
                catastrophic_crossing=False,
                alternatives=(AuditedAlternative(
                    action=canonical_action(actual), rank=1, expected_margin=0.0,
                    win_probability=0.5, confidence_low=0.5, confidence_high=0.5,
                    downside_cvar10=0.0, volatility=0.0, best_share=1.0,
                    regret=0.0, win_probability_drop=0.0, exact_share=1.0,
                    method_notes=("Only legal action; counterfactual search skipped.",),
                ),),
                hard_facts=({
                    "kind": "only_legal_action",
                    "certainty": "fact",
                    "descriptionAr": "هذا هو الخيار القانوني الوحيد في لحظة القرار.",
                    "descriptionEn": "This was the only legal action at the decision point.",
                },),
                signal_hypotheses=(),
                reasons=reasons,
                actual_intent=actual_intent_summary(decision),
                engine_method="forced_legal_action",
                engine_version=__version__,
                bundle_hash=bundle_hash,
                analysis_id=None,
                model_id=(decision.get("decisionMetadata") or {}).get("modelId"),
                conventions_applied=False,
                hidden_deals=0,
                effective_sample_size=1.0,
                predicted_difficulty=0.0,
                debate_score=0.0,
                audit_confidence=1.0,
            )
            payload = audit.to_dict()
            payload["auditHash"] = content_hash(payload)
            payload["auditId"] = f"DAU-{payload['auditHash'][:20].upper()}"
            return self.store.save_decision_audit(payload)
        analysis_row = self._analysis_for(
            decision_id,
            deep=deep,
            force=force,
            analysis_id=analysis_id,
            overrides=overrides,
        )
        result = analysis_row.get("result") or {}
        kind = str(decision["decisionKind"])
        candidates = list(result.get("actions" if kind == "play" else "candidates", []))
        if not candidates:
            raise ValueError("analysis produced no candidate actions")
        candidates.sort(
            key=lambda item: (
                float(item.get("expectedMatchMargin", 0.0)),
                float(item.get("winProbability" if kind == "play" else "teamWinProbability", 0.0)),
                float(item.get("bestShare", 0.0)),
            ),
            reverse=True,
        )
        selected_key = action_key(decision.get("selectedAction"))
        selected_index = next(
            (
                index
                for index, item in enumerate(candidates)
                if action_key(self._candidate_action(item, kind)) == selected_key
            ),
            None,
        )
        if selected_index is None:
            raise ValueError("recorded selected action is absent from the current legal candidate set")

        best_metrics = self._metrics(candidates[0], kind)
        selected_metrics = self._metrics(candidates[selected_index], kind)
        regret = max(0.0, best_metrics["margin"] - selected_metrics["margin"])
        win_drop = max(0.0, best_metrics["win"] - selected_metrics["win"])
        catastrophic = (best_metrics["margin"] > 0 > selected_metrics["margin"]) or (
            best_metrics["win"] >= 0.5 > selected_metrics["win"]
        )
        forced = len(decision.get("legalActions") or []) <= 1
        config = self._config()
        label, quality = quality_label(
            forced=forced,
            regret=regret,
            win_drop=win_drop,
            catastrophic=catastrophic,
            thresholds=config.get("quality", {}),
        )
        alternatives: list[AuditedAlternative] = []
        for index, candidate in enumerate(candidates):
            metrics = self._metrics(candidate, kind)
            alternatives.append(
                AuditedAlternative(
                    canonical_action(self._candidate_action(candidate, kind)),
                    index + 1,
                    metrics["margin"],
                    metrics["win"],
                    metrics["low"],
                    metrics["high"],
                    metrics["cvar"],
                    metrics["volatility"],
                    metrics["bestShare"],
                    max(0.0, best_metrics["margin"] - metrics["margin"]),
                    max(0.0, best_metrics["win"] - metrics["win"]),
                    metrics["exactShare"],
                    principal_variation=tuple(candidate.get("principalVariation", [])),
                    method_notes=tuple(candidate.get("methodNotes", [])),
                )
            )
        reasons = build_reasons(
            decision=decision,
            analysis=result,
            actual_action=decision.get("selectedAction"),
            best_action=self._candidate_action(candidates[0], kind),
            forced=forced,
            regret=regret,
            win_drop=win_drop,
            catastrophic=catastrophic,
            maximum_reasons=int(config.get("reasoning", {}).get("maximum_reasons", 5)),
        )
        hidden_deals = int(result.get("hiddenDeals", 0))
        effective = float(result.get("effectiveSampleSize", hidden_deals or 1))
        confidence = min(1.0, math.sqrt(max(1.0, effective)) / (8.0 if kind == "bid" else 10.0))
        audit = DecisionAudit(
            decision_id=decision_id,
            decision_hash=str(decision["decisionHash"]),
            game_id=str(decision["gameId"]),
            game_hash=decision.get("gameHash"),
            ordinal=int(decision["ordinal"]),
            decision_kind=kind,
            actor=str(decision["actor"]),
            phase=str(decision["phase"]),
            actual_action=canonical_action(decision.get("selectedAction")),
            best_action=canonical_action(self._candidate_action(candidates[0], kind)),
            selected_rank=selected_index + 1,
            quality_label=label,
            quality_score=quality,
            forced=forced,
            expected_regret=round(regret, 6),
            win_probability_drop=round(win_drop, 6),
            catastrophic_crossing=catastrophic,
            alternatives=tuple(alternatives),
            hard_facts=tuple(copy.deepcopy(result.get("deductions", []))),
            signal_hypotheses=tuple(copy.deepcopy(result.get("signals", []))),
            reasons=reasons,
            actual_intent=actual_intent_summary(decision),
            engine_method=str(result.get("method", "unknown")),
            engine_version=__version__,
            bundle_hash=str(analysis_row.get("bundleHash", build_bundle(self.root).sha256)),
            analysis_id=analysis_row.get("analysisId"),
            model_id=(decision.get("decisionMetadata") or {}).get("modelId"),
            conventions_applied=bool(result.get("conventionsApplied", False)),
            hidden_deals=hidden_deals,
            effective_sample_size=effective,
            predicted_difficulty=float(result.get("difficulty", 0.0)),
            debate_score=float(result.get("debateScore", 0.0)),
            audit_confidence=round(confidence, 6),
        )
        payload = audit.to_dict()
        payload["auditHash"] = content_hash(payload)
        payload["auditId"] = f"DAU-{payload['auditHash'][:20].upper()}"
        return self.store.save_decision_audit(payload)

    def audit_game(
        self,
        identifier: str,
        *,
        mode: str = "progressive",
        force: bool = False,
        maximum_decisions: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        game = self.store.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        decisions = list(
            game.get("decisions")
            or self.store.list_decisions(game.get("gameHash") or game["gameId"])
        )
        if maximum_decisions is not None:
            decisions = decisions[: max(1, int(maximum_decisions))]
        screen = [
            self.audit_decision(
                item["decisionId"],
                deep=False,
                force=force,
                overrides=overrides,
            )
            for item in decisions
        ]
        bundle = build_bundle(self.root)
        config = bundle.component("audit")
        if mode in {"deep", "progressive", "observer_safe", "consensus", "trusted_omniscient"}:
            limit = int(config.get("batch", {}).get("maximum_decisions_per_sync_request", 12))
            if maximum_decisions is not None:
                limit = min(limit, max(1, int(maximum_decisions)))
            critical = sorted(
                (item for item in screen if not item.get("forced")),
                key=lambda item: (
                    bool(item.get("catastrophicCrossing")),
                    float(item.get("winProbabilityDrop", 0.0)),
                    float(item.get("expectedRegret", 0.0)),
                    float(item.get("debateScore", 0.0)),
                ),
                reverse=True,
            )[:limit]
            deep_by_id = {
                item["decisionId"]: self.audit_decision(
                    item["decisionId"],
                    deep=True,
                    force=force,
                    overrides=overrides,
                )
                for item in critical
            }
            screen = [deep_by_id.get(item["decisionId"], item) for item in screen]
        typed = [self._from_dict(item) for item in screen]
        report = GameAuditReport.build(
            game_id=str(game["gameId"]),
            game_hash=game.get("gameHash"),
            game_serial=game.get("serial"),
            mode=mode,
            bundle_hash=str(game.get("bundleHash", bundle.sha256)),
            engine_version=__version__,
            config={**config, "requestOverrides": copy.deepcopy(overrides or {})},
            decisions=typed,
        ).to_dict()
        return self.store.save_game_audit(report)

    @staticmethod
    def _from_dict(item: dict[str, Any]) -> DecisionAudit:
        alternatives = tuple(
            AuditedAlternative(
                action=alt["action"],
                rank=int(alt["rank"]),
                expected_margin=float(alt["expectedMargin"]),
                win_probability=float(alt["winProbability"]),
                confidence_low=float((alt.get("confidence95") or [0, 1])[0]),
                confidence_high=float((alt.get("confidence95") or [0, 1])[1]),
                downside_cvar10=float(alt.get("downsideCvar10", 0)),
                volatility=float(alt.get("volatility", 0)),
                best_share=float(alt.get("bestShare", 0)),
                regret=float(alt.get("regret", 0)),
                win_probability_drop=float(alt.get("winProbabilityDrop", 0)),
                exact_share=float(alt.get("exactShare", 0)),
                principal_variation=tuple(alt.get("principalVariation", [])),
                method_notes=tuple(alt.get("methodNotes", [])),
            )
            for alt in item.get("alternatives", [])
        )
        from .models import ReasonEvidence

        reasons = tuple(
            ReasonEvidence(
                code=reason["code"],
                category=reason["category"],
                certainty=reason["certainty"],
                title_ar=reason["titleAr"],
                detail_ar=reason["detailAr"],
                title_en=reason["titleEn"],
                detail_en=reason["detailEn"],
                weight=float(reason.get("weight", 1)),
                source_ids=tuple(reason.get("sourceIds", [])),
            )
            for reason in item.get("reasons", [])
        )
        return DecisionAudit(
            decision_id=item["decisionId"],
            decision_hash=item["decisionHash"],
            game_id=item["gameId"],
            game_hash=item.get("gameHash"),
            ordinal=int(item["ordinal"]),
            decision_kind=item["decisionKind"],
            actor=item["actor"],
            phase=item["phase"],
            actual_action=item["actualAction"],
            best_action=item["bestAction"],
            selected_rank=int(item["selectedRank"]),
            quality_label=item["qualityLabel"],
            quality_score=float(item["qualityScore"]),
            forced=bool(item["forced"]),
            expected_regret=float(item["expectedRegret"]),
            win_probability_drop=float(item["winProbabilityDrop"]),
            catastrophic_crossing=bool(item["catastrophicCrossing"]),
            alternatives=alternatives,
            hard_facts=tuple(item.get("hardFacts", [])),
            signal_hypotheses=tuple(item.get("signalHypotheses", [])),
            reasons=reasons,
            actual_intent=item.get("actualIntent", {}),
            engine_method=item.get("engineMethod", "unknown"),
            engine_version=item.get("engineVersion", ""),
            bundle_hash=item.get("bundleHash", ""),
            analysis_id=item.get("analysisId"),
            model_id=item.get("modelId"),
            conventions_applied=bool(item.get("conventionsApplied", False)),
            hidden_deals=int(item.get("hiddenDeals", 0)),
            effective_sample_size=float(item.get("effectiveSampleSize", 0)),
            predicted_difficulty=float(item.get("predictedDifficulty", 0)),
            debate_score=float(item.get("debateScore", 0)),
            audit_confidence=float(item.get("auditConfidence", 0)),
            created_at=item.get("createdAt", ""),
        )

    def compare_decision(self, decision_id: str) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        audits = decision.get("audits") or self.store.list_decision_audits(decision_id)
        if not audits:
            audits = [self.audit_decision(decision_id)]
        newest = audits[0]
        previous = audits[1] if len(audits) > 1 else None
        return {
            "decisionId": decision_id,
            "current": newest,
            "previous": previous,
            "changed": None
            if previous is None
            else {
                "bestAction": action_key(newest.get("bestAction"))
                != action_key(previous.get("bestAction")),
                "qualityLabel": newest.get("qualityLabel") != previous.get("qualityLabel"),
                "regretDelta": float(newest.get("expectedRegret", 0))
                - float(previous.get("expectedRegret", 0)),
                "difficultyDelta": float(newest.get("predictedDifficulty", 0))
                - float(previous.get("predictedDifficulty", 0)),
                "engineVersionChanged": newest.get("engineVersion")
                != previous.get("engineVersion"),
                "bundleHashChanged": newest.get("bundleHash")
                != previous.get("bundleHash"),
            },
        }

    # Compatibility names for integrations written during the V6 transition.
    def explain_decision(self, decision_id: str, *, analysis_id: str | None = None, rerun: bool = False, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.audit_decision(
            decision_id,
            deep=True,
            force=rerun,
            analysis_id=analysis_id,
            overrides=overrides,
        )

    compare_decision_versions = compare_decision
