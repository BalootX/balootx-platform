from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.belief_v7 import hidden_deal_attempt_budget
from baloot_platform.analysis.models import InformationState
from baloot_platform.config import ProfileBundle, build_bundle, platform_root
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit
from baloot_platform.core.episode_copy import governed_episode_text
from baloot_platform.core.scenario import EpisodeSpec
from baloot_platform.ml.actions import ActionCodec
from baloot_platform.ml.dataset import DatasetWriter
from baloot_platform.ml.features import encode_bid_state, encode_information_state
from baloot_platform.ml.models import model_runtime_identity
from baloot_platform.version import __version__
from baloot_platform.storage.artifact_store import build_artifact_store

from .store import LearningStore
from .training import (
    NumpyLinearPolicyValueModel,
    TorchScriptPolicyValueModel,
    evaluate_rows,
    train_numpy_linear,
    train_torch_policy_value,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _split_for_group(group: str, fractions: tuple[float, float, float]) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    train, validation, _test = fractions
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LearningService:
    """Application service for immutable replay learning and model lineage.

    The service never lets feedback modify the authoritative rules engine.
    Feedback creates new analyses, datasets, and model candidates; every
    artifact is linked to a game hash, decision hash, engine version, bundle
    hash, feature schema, and action schema.
    """

    def __init__(
        self,
        root: str | Path | None,
        data_dir: str | Path,
        *,
        store: LearningStore | None = None,
        game_manager: Any | None = None,
        artifact_store: Any | None = None,
    ) -> None:
        self.root = platform_root(root)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or LearningStore(self.data_dir / "learning.sqlite3")
        self.game_manager = game_manager
        bundle = build_bundle(self.root)
        self.presentation_copy = dict(bundle.video["localizations"])
        self.artifact_store = artifact_store or build_artifact_store(bundle, writable_root=self.data_dir)
        self.artifact_cache = self.data_dir / "artifact-cache"
        self.artifact_cache.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = self.data_dir / "datasets"
        self.models_dir = self.data_dir / "models"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._production_model_cache: tuple[str, str, Any] | None = None

    def _materialize(self, reference: str | Path, *, suffix: str = "") -> Path:
        value = str(reference)
        if "://" not in value:
            return Path(value)
        return self.artifact_store.materialize(value, suffix=suffix, cache_dir=self.artifact_cache)

    def _model_files(self, model: dict[str, Any]) -> tuple[Path, Path]:
        plugin_id = str(model.get("pluginId", ""))
        suffix = ".pt" if plugin_id == TorchScriptPolicyValueModel.id else ".npz"
        artifact = self._materialize(str(model["artifactPath"]), suffix=suffix)
        manifest = artifact.with_suffix(".manifest.json")
        manifest.write_text(json.dumps(model.get("manifest", {}), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return artifact, manifest

    def _governed_lineage_status(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve current consent without mutating an immutable dataset artifact."""
        manifest = dict(dataset.get("manifest") or dataset.get("manifestData") or {})
        metadata = dict(manifest.get("metadata") or {})
        lineage = list(metadata.get("governedHitlLineage") or [])
        governance_store = getattr(self, "hitl_store", None)
        invalid: list[dict[str, str]] = []
        if lineage and governance_store is None:
            invalid.append({"candidateId": "", "reason": "governance_store_unavailable"})
        elif governance_store is not None:
            for item in lineage:
                candidate_id = str(item.get("candidateId") or "")
                candidate = governance_store.get_candidate(candidate_id) if candidate_id else None
                if not candidate or candidate.get("status") != "accepted":
                    invalid.append({"candidateId": candidate_id, "reason": "candidate_not_accepted"})
                    continue
                for subject_id in item.get("consentSubjects") or ():
                    consent = governance_store.get_consent(str(subject_id))
                    if not consent or not consent.get("allowTraining") or consent.get("revokedAt"):
                        invalid.append({"candidateId": candidate_id, "reason": "consent_withdrawn"})
                        break
        return {
            "schemaVersion": 1,
            "governedLineagePresent": bool(lineage),
            "eligible": not invalid,
            "invalid": invalid,
            "lineageSha256": _hash(lineage),
        }

    def _require_governed_lineage(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        status = self._governed_lineage_status(dataset)
        if not status["eligible"]:
            raise ValueError("dataset contains withdrawn or ineligible governed HITL lineage")
        return status

    def production_model(self) -> Any | None:
        """Load the currently promoted model with checksum/schema validation.

        The cache key includes model identity and update timestamp, so promotion
        or replacement becomes visible without restarting game workers.
        """
        model = next((item for item in self.store.list_models() if item.get("status") == "production"), None)
        if model is None:
            self._production_model_cache = None
            return None
        cache_key = (str(model["modelId"]), str(model.get("updatedAt", "")))
        if self._production_model_cache is not None and self._production_model_cache[:2] == cache_key:
            return self._production_model_cache[2]
        artifact, manifest = self._model_files(model)
        plugin_id = str(model.get("pluginId", ""))
        if plugin_id == TorchScriptPolicyValueModel.id:
            loaded = TorchScriptPolicyValueModel(artifact, manifest)
        elif plugin_id == NumpyLinearPolicyValueModel.id:
            loaded = NumpyLinearPolicyValueModel(artifact, manifest)
        else:
            raise ValueError(f"unsupported production model plugin {plugin_id}")
        identity = model_runtime_identity(loaded)
        if identity is None or identity["modelId"] != str(model["modelId"]):
            raise ValueError("production model identity does not match its registry record")
        if identity["pluginId"] != plugin_id:
            raise ValueError("production model plugin does not match its registry record")
        self._production_model_cache = (cache_key[0], cache_key[1], loaded)
        return loaded

    def archive_session(
        self,
        session: Any,
        *,
        status: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        bundle = build_bundle(self.root, mode=session.mode)
        inferred = status or ("completed" if session.state.current_player is None else "snapshot")
        parent = getattr(session, "parent_game_hash", None)
        result = self.store.archive_game(
            session,
            bundle_hash=getattr(session, "bundle_hash", bundle.sha256),
            engine_version=__version__,
            parent_game_hash=parent,
            status=inferred,
            label=label,
        )
        try:
            session.game_hash = result["gameHash"]
            session.serial = result["serial"]
        except Exception:
            pass
        return result

    def reopen_game(self, identifier: str, *, branch: bool = False, label: str | None = None) -> dict[str, Any]:
        session = self.store.reopen_session(identifier)
        original_hash = self.store.get_game(identifier)["gameHash"]
        if branch:
            session.id = uuid.uuid4().hex[:16]
            session.version = 1
            session.parent_game_hash = original_hash
            session.game_hash = None
            session.serial = None
        if self.game_manager is not None:
            self.game_manager.import_session(session, replace_existing=not branch)
        return {
            "game": session.to_dict(reveal_hands=False),
            "branched": branch,
            "parentGameHash": original_hash if branch else None,
            "label": label,
        }

    def reanalyze_decision(
        self,
        decision_id: str,
        *,
        samples: int = 96,
        exact_allocation_limit: int = 20_000,
        exact_search_remaining: int = 10,
        use_conventions: bool = False,
        seed: int = 1,
    ) -> dict[str, Any]:
        payload = self.store.load_decision_payload(decision_id)
        mode = str(payload.get("mode", "full"))
        bundle = build_bundle(self.root, mode=mode)
        state = payload["state"]
        actor = Seat.parse(payload["actor"])
        kind = str(payload["decisionKind"])
        config = {
            "samples": int(samples),
            "exactAllocationLimit": int(exact_allocation_limit),
            "exactSearchRemaining": int(exact_search_remaining),
            "useConventions": bool(use_conventions),
            "seed": int(seed),
        }
        if kind == "play":
            if state.current_player != actor:
                raise ValueError("archived play decision actor is not current player")
            info = InformationState.from_complete(state, actor)
            convention = dict(bundle.convention)
            convention["apply_to_probability"] = bool(use_conventions)
            result = analyze(
                info,
                bundle.rules,
                convention,
                requested_samples=max(2, int(samples)),
                exact_allocation_limit=max(1, int(exact_allocation_limit)),
                exact_search_max_remaining=max(0, int(exact_search_remaining)),
                maximum_sampling_attempts=hidden_deal_attempt_budget(
                    int(samples),
                    multiplier=600,
                    minimum=10_000,
                ),
                seed=int(seed),
            ).to_dict()
        elif kind == "bid":
            result = self._reanalyze_bid(payload, bundle, samples=max(2, int(samples)), seed=int(seed))
        else:
            raise ValueError(f"decision kind {kind} is not yet trainable")
        prior = self.store.list_analyses(decision_id)
        return self.store.add_analysis(
            decision_id,
            engine_version=__version__,
            bundle_hash=bundle.sha256,
            config=config,
            result=result,
            supersedes_analysis_id=prior[0]["analysisId"] if prior else None,
        )

    @staticmethod
    def _reanalyze_bid(
        payload: dict[str, Any],
        bundle: ProfileBundle,
        *,
        samples: int = 48,
        seed: int = 1,
    ) -> dict[str, Any]:
        from baloot_platform.analysis.bidding import analyze_bid

        state = payload["state"]
        actor = Seat.parse(payload["actor"])
        machine = payload.get("bidding")
        if machine is None:
            raise ValueError("archived bid decision does not contain its auction machine")
        return analyze_bid(
            state,
            machine,
            actor,
            bundle.rules,
            requested_samples=samples,
            seed=seed,
        ).to_dict()

    def add_feedback(
        self,
        decision_id: str,
        *,
        preferred_action: Any,
        confidence: float,
        rationale: str,
        source: str = "expert",
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        analyses = decision.get("analyses", [])
        engine_action: Any | None = None
        if analyses:
            result = analyses[0].get("result", {})
            engine_action = result.get("recommendedCard") or result.get("recommendedAction")
        disagreement = 0.0 if engine_action == preferred_action else 1.0
        interestingness = max((item.get("result", {}).get("difficulty", 0) / 100 for item in analyses), default=0.0)
        return self.store.add_feedback(
            decision_id,
            preferred_action=preferred_action,
            engine_action=engine_action,
            confidence=confidence,
            rationale=rationale,
            source=source,
            reviewer=reviewer,
            interestingness=interestingness,
            model_error=disagreement,
        )

    def build_dataset(
        self,
        *,
        purpose: str = "human_corrected_decisions_v3",
        minimum_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Build an observer-safe, audit-enriched policy/value dataset.

        Every example is grouped by immutable game hash to prevent replay
        leakage across train, validation and test. Human preference, the
        action actually taken, the engine recommendation, live policy trace,
        counterfactual audit and calibrated difficulty are preserved as
        separate fields. The policy target remains a legal reviewed action;
        rules and legal masks stay authoritative.
        """
        bundle = build_bundle(self.root)
        rows: list[dict[str, Any]] = []
        queue = self.store.learning_queue(limit=5000)
        learning = bundle.component("learning")
        dataset_cfg = learning.get("dataset", {})
        fractions = (
            float(dataset_cfg.get("train_fraction", 0.80)),
            float(dataset_cfg.get("validation_fraction", 0.10)),
            float(dataset_cfg.get("test_fraction", 0.10)),
        )
        threshold = float(
            learning.get("feedback", {}).get("minimum_confidence", 0.55)
            if minimum_confidence is None
            else minimum_confidence
        )

        def bid_parts(value: Any) -> tuple[str, Suit | None]:
            if isinstance(value, dict):
                inner = value.get("value") if value.get("kind") == "object" else value
                if isinstance(inner, dict):
                    value = inner
                action = str(value.get("action", "")).strip().lower()
                suit_value = value.get("trumpSuit")
            else:
                text = str(value).strip().lower()
                action, _, suit_value = text.partition(":")
            trump = Suit.parse(suit_value) if suit_value not in {None, "", "null"} else None
            return action, trump

        def action_index(kind: str, value: Any) -> int | None:
            if value is None:
                return None
            try:
                if kind == "play":
                    if isinstance(value, dict) and value.get("kind") == "card":
                        value = {key: value.get(key) for key in ("code", "suit", "rank", "id") if value.get(key) is not None}
                    return ActionCodec.play(Card.parse(value))
                action, trump = bid_parts(value)
                return ActionCodec.bid(action, trump)
            except Exception:
                return None

        def safe_policy_trace(metadata: dict[str, Any]) -> dict[str, Any]:
            allowed = {
                "schemaVersion", "source", "decisionSource", "profileId",
                "modelId", "modelVersion", "searchMethod", "analysisMethod",
                "confidence", "controlledMistake", "controlledDeception",
                "reasonTags", "rankedActions", "observerSafe", "riskProfile",
                "analysisSolverCompletion", "analysisCompletionStatuses",
            }
            result = {key: metadata[key] for key in allowed if key in metadata}
            if isinstance(result.get("rankedActions"), list):
                result["rankedActions"] = result["rankedActions"][:12]
            return result

        for item in queue:
            decision = item["decision"]
            structured = decision.get("structuredReviews") or []
            consensus = decision.get("reviewConsensus") or {}
            if structured:
                # Decision-Workbench feedback is evidence, not an automatic label.
                # It enters training only after independent reviewed consensus.
                if not bool(consensus.get("trainingEligible", False)):
                    continue
                from baloot_platform.audit.models import action_key
                preferred_key = action_key(consensus.get("preferredAction"))
                supporting = [
                    review for review in structured
                    if action_key(review.get("preferredAction")) == preferred_key
                ]
                if not supporting:
                    continue
                source_review = max(
                    supporting, key=lambda value: (float(value.get("confidence", 0.0)), value.get("createdAt", ""))
                )
                feedback = {
                    "preferredAction": consensus.get("preferredAction"),
                    "engineAction": None,
                    "confidence": min(1.0, float(consensus.get("supportShare", 0.0))),
                    "rationale": source_review.get("rationale", ""),
                    "source": "structured_review_consensus",
                    "createdAt": source_review.get("createdAt", ""),
                    "reviewConsensus": consensus,
                }
            else:
                # Legacy feedback remains useful for audits and the active-
                # learning queue, but it is not a governed training label.
                # Only structured consensus or accepted HITL candidates can
                # cross the dataset boundary.
                continue
            payload = self.store.load_decision_payload(decision["decisionId"])
            state = payload["state"]
            actor = Seat.parse(payload["actor"])
            feedback_confidence = float(feedback["confidence"])
            if feedback_confidence < threshold:
                continue
            analyses = decision.get("analyses", [])
            latest_audit = (decision.get("audits") or [None])[0]
            difficulty = decision.get("difficulty") or {}
            value_target = 0.0
            rule_bundle = build_bundle(self.root, mode=str(payload.get("mode", "full")))

            if decision["decisionKind"] == "play":
                if state.current_player != actor:
                    continue
                preferred = Card.parse(feedback["preferredAction"])
                from baloot_platform.core.engine import BalootEngine
                legal_cards = BalootEngine(rule_bundle.rules).legal_moves(state)
                if preferred not in legal_cards:
                    continue
                features = encode_information_state(InformationState.from_complete(state, actor))
                legal_mask = ActionCodec.legal_play_mask(legal_cards)
                target = ActionCodec.play(preferred)
                if analyses:
                    actions = analyses[0].get("result", {}).get("actions", [])
                    match = next((candidate for candidate in actions if Card.parse(candidate["card"]) == preferred), None)
                    if match is not None:
                        value_target = math.tanh(float(match.get("expectedMatchMargin", match.get("expectedMargin", 0.0))) / 50.0)
            elif decision["decisionKind"] == "bid":
                machine = payload.get("bidding")
                if machine is None or machine.finished or machine.actor != actor:
                    continue
                from baloot_platform.analysis.bidding import legal_bid_choices
                choices = legal_bid_choices(machine)
                action, trump = bid_parts(feedback["preferredAction"])
                matched = next((choice for choice in choices if choice.action == action and choice.trump_suit == trump), None)
                if matched is None:
                    continue
                features = encode_bid_state(state, machine, actor)
                mask = [0] * len(ActionCodec.legal_bid_mask(machine.legal_actions(), purchase_suit=machine.purchase_suit, second_round=machine.round_index == 2))
                for choice in choices:
                    mask[ActionCodec.bid(choice.action, choice.trump_suit)] = 1
                legal_mask = tuple(mask)
                target = ActionCodec.bid(matched.action, matched.trump_suit)
                if analyses:
                    candidates = analyses[0].get("result", {}).get("candidates", [])
                    match = next((
                        candidate for candidate in candidates
                        if str(candidate.get("action", "")).lower() == action
                        and (candidate.get("trumpSuit") or None) == (trump.name.lower() if trump is not None else None)
                    ), None)
                    if match is not None:
                        value_target = math.tanh(float(match.get("expectedMatchMargin", match.get("expectedMargin", 0.0))) / 50.0)
            else:
                continue

            actual_index = action_index(decision["decisionKind"], decision.get("selectedAction"))
            engine_index = action_index(decision["decisionKind"], feedback.get("engineAction"))
            if engine_index is None and latest_audit:
                engine_index = action_index(decision["decisionKind"], latest_audit.get("bestAction"))

            regret = float((latest_audit or {}).get("expectedRegret", 0.0))
            win_drop = float((latest_audit or {}).get("winProbabilityDrop", 0.0))
            catastrophic = bool((latest_audit or {}).get("catastrophicCrossing", False))
            audit_confidence = float((latest_audit or {}).get("auditConfidence", 0.0))
            severity = max(
                1.0 if catastrophic else 0.0,
                min(1.0, regret / 25.0),
                min(1.0, win_drop / 0.35),
            )
            disagreement = 1.0 if actual_index is not None and actual_index != target else 0.0
            evidence_factor = 0.65 + 0.35 * max(audit_confidence, feedback_confidence)
            sample_weight = max(0.10, min(3.0, feedback_confidence * evidence_factor * (1.0 + 0.75 * severity + 0.25 * disagreement)))

            accepted_alternatives: list[int] = []
            if latest_audit:
                for alternative in latest_audit.get("alternatives", []):
                    if float(alternative.get("regret", 1e9)) <= 1.5 and float(alternative.get("winProbabilityDrop", 1e9)) <= 0.02:
                        encoded = action_index(decision["decisionKind"], alternative.get("action"))
                        if encoded is not None and encoded not in accepted_alternatives:
                            accepted_alternatives.append(encoded)
            if target not in accepted_alternatives:
                accepted_alternatives.insert(0, target)

            group = decision.get("gameHash") or decision["gameId"]
            row = {
                "schemaVersion": 3,
                "decisionId": decision["decisionId"],
                "decisionHash": decision.get("decisionHash"),
                "decisionKind": decision["decisionKind"],
                "gameHash": group,
                "split": _split_for_group(str(group), fractions),
                "features": features.tolist(),
                "legalMask": list(legal_mask),
                "targetAction": target,
                "acceptedActions": accepted_alternatives,
                "actualAction": actual_index,
                "engineAction": engine_index,
                "valueTarget": value_target,
                "sampleWeight": sample_weight,
                "source": feedback["source"],
                "reviewConsensus": copy.deepcopy(feedback.get("reviewConsensus")),
                "preferencePair": {
                    "preferredAction": target,
                    "actualAction": actual_index,
                    "engineAction": engine_index,
                    "preferredOverActual": actual_index is not None and target != actual_index,
                    "preferredOverEngine": engine_index is not None and target != engine_index,
                    "confidence": feedback_confidence,
                    "rationale": feedback.get("rationale", ""),
                },
                "policyTrace": safe_policy_trace(dict(decision.get("decisionMetadata") or {})),
                "audit": None if latest_audit is None else {
                    "auditId": latest_audit.get("auditId") or latest_audit.get("decisionAuditId"),
                    "qualityLabel": latest_audit.get("qualityLabel"),
                    "qualityScore": latest_audit.get("qualityScore"),
                    "expectedRegret": regret,
                    "winProbabilityDrop": win_drop,
                    "catastrophicCrossing": catastrophic,
                    "auditConfidence": audit_confidence,
                    "engineMethod": latest_audit.get("engineMethod"),
                    "selectedRank": latest_audit.get("selectedRank"),
                },
                "difficulty": {
                    "engineDifficulty": (difficulty or {}).get("engineDifficulty", (latest_audit or {}).get("predictedDifficulty")),
                    "calibratedDifficulty": (difficulty or {}).get("calibratedDifficulty", (difficulty or {}).get("calibratedScore")),
                    "responseCount": int((difficulty or {}).get("responseCount", 0)),
                    "standardError": (difficulty or {}).get("standardError"),
                },
            }
            rows.append(row)

        # V18 governed Human-in-the-Loop candidates remain separate labels until
        # a candidate has passed independent challenger evaluation and explicit
        # acceptance. They never mutate the production engine directly.
        governance_store = getattr(self, "hitl_store", None)
        governed_lineage: list[dict[str, Any]] = []
        if governance_store is not None:
            for candidate in governance_store.list_candidates(status="accepted"):
                consent_subjects = sorted(
                    str(value) for value in candidate.get("consentSubjects") or ()
                )
                if any(
                    not (consent := governance_store.get_consent(subject_id))
                    or not consent.get("allowTraining")
                    or consent.get("revokedAt")
                    for subject_id in consent_subjects
                ):
                    continue
                case = governance_store.get_case(str(candidate.get("caseId")))
                if not case:
                    continue
                decision = self.store.get_decision(str(case.get("decisionId")))
                if not decision:
                    continue
                try:
                    payload = self.store.load_decision_payload(decision["decisionId"])
                    state = payload["state"]
                    actor = Seat.parse(payload["actor"])
                    preferred_value = candidate.get("preferredAction")
                    rule_bundle = build_bundle(self.root, mode=str(payload.get("mode", "full")))
                    if decision["decisionKind"] == "play":
                        if state.current_player != actor:
                            continue
                        preferred = Card.parse(preferred_value)
                        from baloot_platform.core.engine import BalootEngine
                        legal_cards = BalootEngine(rule_bundle.rules).legal_moves(state)
                        if preferred not in legal_cards:
                            continue
                        features = encode_information_state(InformationState.from_complete(state, actor))
                        legal_mask = ActionCodec.legal_play_mask(legal_cards)
                        target = ActionCodec.play(preferred)
                    elif decision["decisionKind"] == "bid":
                        machine = payload.get("bidding")
                        if machine is None or machine.finished or machine.actor != actor:
                            continue
                        from baloot_platform.analysis.bidding import legal_bid_choices
                        choices = legal_bid_choices(machine)
                        action, trump = bid_parts(preferred_value)
                        matched = next((choice for choice in choices if choice.action == action and choice.trump_suit == trump), None)
                        if matched is None:
                            continue
                        features = encode_bid_state(state, machine, actor)
                        mask = [0] * len(ActionCodec.legal_bid_mask(machine.legal_actions(), purchase_suit=machine.purchase_suit, second_round=machine.round_index == 2))
                        for choice in choices:
                            mask[ActionCodec.bid(choice.action, choice.trump_suit)] = 1
                        legal_mask = tuple(mask)
                        target = ActionCodec.bid(matched.action, matched.trump_suit)
                    else:
                        continue
                except Exception:
                    continue
                actual_index = action_index(decision["decisionKind"], decision.get("selectedAction"))
                engine_index = action_index(decision["decisionKind"], case.get("productionAction"))
                labels = copy.deepcopy(candidate.get("labelProvenance") or {})
                group = decision.get("gameHash") or decision["gameId"]
                rows.append({
                    "schemaVersion": 3,
                    "decisionId": f"{decision['decisionId']}#{candidate['candidateId']}",
                    "decisionHash": decision.get("decisionHash"),
                    "decisionKind": decision["decisionKind"],
                    "gameHash": group,
                    "split": _split_for_group(str(group), fractions),
                    "features": features.tolist(),
                    "legalMask": list(legal_mask),
                    "targetAction": target,
                    "acceptedActions": [target],
                    "actualAction": actual_index,
                    "engineAction": engine_index,
                    "valueTarget": 0.0,
                    "sampleWeight": 1.5,
                    "source": "governed_hitl_candidate",
                    "reviewConsensus": labels.get("review_consensus"),
                    "labelProvenance": labels,
                    "preferencePair": {
                        "preferredAction": target, "actualAction": actual_index, "engineAction": engine_index,
                        "preferredOverActual": actual_index is not None and target != actual_index,
                        "preferredOverEngine": engine_index is not None and target != engine_index,
                        "confidence": float((labels.get("review_consensus") or {}).get("weightedSupport", 1.0)),
                        "rationale": "governed Human-in-the-Loop candidate",
                    },
                    "policyTrace": safe_policy_trace(dict(decision.get("decisionMetadata") or {})),
                    "audit": None,
                    "difficulty": {"engineDifficulty": None, "calibratedDifficulty": None, "responseCount": 0, "standardError": None},
                })
                governed_lineage.append({
                    "candidateId": str(candidate["candidateId"]),
                    "caseId": str(candidate["caseId"]),
                    "decisionId": str(case["decisionId"]),
                    "consentSubjects": consent_subjects,
                    "caseHash": str(case.get("caseHash") or ""),
                })

        if not rows:
            raise ValueError("no reviewed legal play or bid decisions are available")
        time_censored_examples = sum(
            row["policyTrace"].get("analysisSolverCompletion") == "time_censored"
            for row in rows
        )
        artifact = DatasetWriter(self.datasets_dir).write(
            rows,
            rules_hash=bundle.rules.sha256,
            bundle_hash=bundle.sha256,
            purpose=purpose,
            metadata={
                "schemaVersion": 3,
                **({"timeCensoredExamples": time_censored_examples} if time_censored_examples else {}),
                "groupSplitKey": "gameHash",
                "splitCounts": {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")},
                "decisionKindCounts": {kind: sum(row["decisionKind"] == kind for row in rows) for kind in ("play", "bid")},
                "auditEnrichedExamples": sum(row["audit"] is not None for row in rows),
                "preferencePairs": sum(bool(row["preferencePair"].get("preferredOverActual") or row["preferencePair"].get("preferredOverEngine")) for row in rows),
                "catastrophicExamples": sum(bool((row["audit"] or {}).get("catastrophicCrossing")) for row in rows),
                "averageSampleWeight": sum(float(row["sampleWeight"]) for row in rows) / len(rows),
                "governedHitlExamples": sum(row.get("source") == "governed_hitl_candidate" for row in rows),
                "governedHitlLineage": governed_lineage,
                "governedHitlLineageSha256": _hash(governed_lineage),
                "engineVersion": __version__,
            },
        )
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        examples_ref = self.artifact_store.put_file(
            artifact.examples_path,
            media_type="application/x-ndjson",
            name=f"{artifact.id}-examples.jsonl",
        )
        manifest["artifactUri"] = examples_ref["uri"]
        manifest["artifactSha256"] = examples_ref["sha256"]
        artifact.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        manifest_ref = self.artifact_store.put_file(
            artifact.manifest_path,
            media_type="application/json",
            name=f"{artifact.id}-manifest.json",
        )
        manifest["manifestArtifactUri"] = manifest_ref["uri"]
        registered = self.store.register_dataset(
            artifact,
            purpose=purpose,
            manifest=manifest,
            artifact_path=examples_ref["uri"],
        )
        self.store.link_artifact(owner_kind="dataset", owner_id=artifact.id, kind="examples", uri=examples_ref["uri"], sha256=examples_ref["sha256"], metadata=examples_ref)
        self.store.link_artifact(owner_kind="dataset", owner_id=artifact.id, kind="manifest", uri=manifest_ref["uri"], sha256=manifest_ref["sha256"], metadata=manifest_ref)
        return registered

    def train_model(
        self,
        dataset_id: str,
        *,
        overrides: dict[str, Any] | None = None,
        backend: str = "auto",
    ) -> dict[str, Any]:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        self._require_governed_lineage(dataset)
        config = dict(build_bundle(self.root).component("learning").get("training", {}))
        config.update(overrides or {})
        selected = str(backend or "auto").lower()
        if selected == "auto":
            try:
                import torch  # noqa: F401
                selected = "torch" if int(dataset.get("exampleCount", 0)) >= 16 else "numpy"
            except ImportError:
                selected = "numpy"
        trainer = train_torch_policy_value if selected == "torch" else train_numpy_linear
        dataset_path = self._materialize(str(dataset["artifactPath"]), suffix=".jsonl")
        result = trainer(
            dataset_path,
            self.models_dir,
            dataset_id=dataset_id,
            epochs=int(config.get("epochs", 40)),
            learning_rate=float(config.get("learning_rate", 0.001 if selected == "torch" else 0.03)),
            l2=float(config.get("l2", 1e-5 if selected == "torch" else 1e-4)),
            batch_size=int(config.get("batch_size", 64)),
            seed=int(config.get("seed", 1)),
            preference_loss_weight=float(config.get("preference_loss_weight", 0.25)),
        )
        media_type = "application/vnd.pytorch.torchscript" if result.plugin_id == TorchScriptPolicyValueModel.id else "application/x-npz"
        model_ref = self.artifact_store.put_file(result.artifact_path, media_type=media_type, name=result.artifact_path.name)
        manifest = dict(result.manifest)
        manifest["artifactUri"] = model_ref["uri"]
        manifest_path = result.artifact_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        manifest_ref = self.artifact_store.put_file(manifest_path, media_type="application/json", name=manifest_path.name)
        manifest["manifestArtifactUri"] = manifest_ref["uri"]
        registered = self.store.register_model(
            model_id=result.model_id,
            plugin_id=result.plugin_id,
            version=result.manifest["version"],
            artifact_path=model_ref["uri"],
            manifest=manifest,
            dataset_id=dataset_id,
            status="candidate",
            metrics=result.metrics,
        )
        self.store.link_artifact(owner_kind="model", owner_id=result.model_id, kind="model", uri=model_ref["uri"], sha256=model_ref["sha256"], metadata=model_ref)
        self.store.link_artifact(owner_kind="model", owner_id=result.model_id, kind="manifest", uri=manifest_ref["uri"], sha256=manifest_ref["sha256"], metadata=manifest_ref)
        return registered

    def evaluate_model(self, model_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        model = self.store.get_model(model_id)
        if model is None:
            raise KeyError(model_id)
        selected_dataset = dataset_id or model.get("datasetId")
        dataset = self.store.get_dataset(selected_dataset) if selected_dataset else None
        if dataset is None:
            raise KeyError(str(selected_dataset))
        artifact_path, manifest_path = self._model_files(model)
        plugin_id = str(model.get("pluginId", ""))
        if plugin_id == TorchScriptPolicyValueModel.id:
            loaded = TorchScriptPolicyValueModel(artifact_path, manifest_path)
        else:
            loaded = NumpyLinearPolicyValueModel(artifact_path, manifest_path)
        rows = _read_jsonl(self._materialize(str(dataset["artifactPath"]), suffix=".jsonl"))
        metrics = {
            split: evaluate_rows(loaded, [row for row in rows if row.get("split", "train") == split])
            for split in ("train", "validation", "test")
        }
        return self.store.update_model(model_id, metrics=metrics)

    def promote_model(
        self,
        model_id: str,
        *,
        shadow_evidence: dict[str, Any] | None = None,
        rollback_model_id: str | None = None,
    ) -> dict[str, Any]:
        model = self.store.get_model(model_id)
        if model is None:
            raise KeyError(model_id)
        if model.get("status") not in {"candidate", "validated"}:
            raise ValueError("only candidate or validated models can be promoted")
        dataset_id = model.get("datasetId")
        dataset = self.store.get_dataset(str(dataset_id)) if dataset_id else None
        if getattr(self, "hitl_store", None) is not None:
            if dataset is None:
                raise ValueError("model dataset lineage is unavailable")
            lineage_status = self._require_governed_lineage(dataset)
        else:
            lineage_status = {
                "schemaVersion": 1,
                "governedLineagePresent": False,
                "eligible": True,
                "invalid": [],
                "lineageSha256": _hash([]),
            }
        bundle = build_bundle(self.root)
        learning_profile = bundle.component("learning")
        ml_profile = bundle.component("ml")
        promotion = dict(learning_profile.get("promotion", {}))
        release_gate = dict(ml_profile.get("promotion", {}))
        metrics = model.get("metrics", {})
        holdout = metrics.get("test") or metrics.get("validation") or {}
        if int(holdout.get("exampleCount", 0)) < int(promotion.get("minimum_examples", 1)):
            raise ValueError("model has insufficient holdout examples")
        if float(holdout.get("top3Accuracy", 0.0)) < float(promotion.get("minimum_top3_accuracy", 0.0)):
            raise ValueError("model does not meet top-3 accuracy gate")
        if float(holdout.get("illegalActionRate", 1.0)) > float(promotion.get("maximum_illegal_action_rate", 0.0)):
            raise ValueError("model violates illegal-action gate")
        evidence = dict(shadow_evidence or {})
        required = {
            "decisionCount",
            "boardSets",
            "scoreRate",
            "scoreRateLowerBound",
            "calibrationError",
            "p95LatencyMs",
            "oodRecall",
            "legalViolationRate",
        }
        missing = sorted(required - set(evidence))
        if missing:
            raise ValueError(
                "shadow promotion evidence is incomplete: "
                + ", ".join(missing)
            )
        numeric = {
            key: float(evidence[key])
            for key in required
        }
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("shadow evidence values must be finite")
        if any(
            not numeric[key].is_integer()
            for key in ("decisionCount", "boardSets")
        ):
            raise ValueError("shadow evidence counts must be integers")
        if int(numeric["decisionCount"]) < int(
            release_gate.get("minimum_decisions", 0)
        ):
            raise ValueError("shadow evidence has insufficient decisions")
        if int(numeric["boardSets"]) < int(
            release_gate.get("minimum_board_sets", 0)
        ):
            raise ValueError("shadow evidence has insufficient board sets")
        lower_bounds = {
            "scoreRate": float(release_gate.get("minimum_score_rate", 0.0)),
            "scoreRateLowerBound": float(
                release_gate.get("minimum_lower_bound", 0.0)
            ),
            "oodRecall": float(release_gate.get("minimum_ood_recall", 0.0)),
        }
        for key, minimum in lower_bounds.items():
            if not 0.0 <= numeric[key] <= 1.0 or numeric[key] < minimum:
                raise ValueError(f"shadow evidence fails {key} gate")
        upper_bounds = {
            "calibrationError": float(
                release_gate.get("maximum_calibration_error", 1.0)
            ),
            "p95LatencyMs": float(
                release_gate.get("maximum_p95_latency_ms", math.inf)
            ),
            "legalViolationRate": float(
                release_gate.get("maximum_legal_violation_rate", 0.0)
            ),
        }
        for key, maximum in upper_bounds.items():
            if numeric[key] < 0.0 or numeric[key] > maximum:
                raise ValueError(f"shadow evidence fails {key} gate")
        current = next(
            (
                item
                for item in self.store.list_models()
                if item.get("status") == "production"
            ),
            None,
        )
        expected_rollback = (
            str(current["modelId"])
            if current is not None
            else "deterministic_heuristic_fallback_v1"
        )
        if bool(release_gate.get("rollback_required", True)):
            if str(rollback_model_id or "") != expected_rollback:
                raise ValueError(
                    "rollback model must match the active production model "
                    "or deterministic fallback"
                )
        governed_evidence = {
            "schemaVersion": 1,
            "candidateModelId": model_id,
            "rollbackModelId": expected_rollback,
            "datasetLineage": lineage_status,
            "shadow": {
                key: (
                    int(numeric[key])
                    if key in {"decisionCount", "boardSets"}
                    else numeric[key]
                )
                for key in sorted(required)
            },
            "learningProfile": {
                "id": str(learning_profile["profile"]["id"]),
                "version": str(learning_profile["profile"]["version"]),
            },
            "mlProfile": {
                "id": str(ml_profile["profile"]["id"]),
                "version": str(ml_profile["profile"]["version"]),
            },
        }
        governed_evidence["sha256"] = _hash(governed_evidence)
        updated_metrics = copy.deepcopy(metrics)
        updated_metrics["promotionEvidence"] = governed_evidence
        return self.store.update_model(
            model_id,
            status="production",
            metrics=updated_metrics,
        )


    def fork_decision(
        self, decision_id: str, *, action: Any | None = None,
        continue_to_end: bool = False, maximum_actions: int = 128,
    ) -> dict[str, Any]:
        if self.game_manager is None:
            raise RuntimeError("game manager is unavailable")
        payload = self.store.load_decision_payload(decision_id)
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        from baloot_platform.core.engine import BalootEngine
        from baloot_platform.service.game import GameSession

        mode = str(payload.get("mode", "full"))
        bundle = build_bundle(self.root, mode=mode)
        state = payload["state"].clone()
        game = GameSession(
            uuid.uuid4().hex[:16],
            mode,
            BalootEngine(bundle.rules),
            state,
            payload.get("bidding"),
            1,
            bundle.sha256,
            parent_game_hash=decision.get("gameHash"),
            agent_profiles=dict(self.game_manager.agent_registry.default_lineup),
        )
        self.game_manager.import_session(game)
        selected = action if action is not None else payload.get("selectedAction")
        if payload["decisionKind"] == "play" and selected is not None:
            game = self.game_manager.play(
                game.id, seat=Seat.parse(payload["actor"]), card=selected, expected_version=game.version
            )
        elif payload["decisionKind"] == "bid" and selected is not None:
            value = selected if isinstance(selected, dict) else {"action": str(selected)}
            trump = value.get("trumpSuit")
            game = self.game_manager.bid(
                game.id,
                seat=Seat.parse(payload["actor"]),
                action=str(value.get("action", "pass")),
                trump=None if not trump else __import__("baloot_platform.core.enums", fromlist=["Suit"]).Suit.parse(trump),
                expected_version=game.version,
            )
        if continue_to_end and (
            (game.bidding is not None and not game.bidding.finished) or game.state.current_player is not None
        ):
            game = self.game_manager.advance_bots(
                game.id, stop_before=None, maximum_actions=max(1, min(10_000, int(maximum_actions)))
            )
        result = game.to_dict(reveal_hands=False)
        result["branchGameId"] = game.id
        result["parentGameHash"] = decision.get("gameHash")
        result["sourceDecisionId"] = decision_id
        result["continuedToEnd"] = bool(continue_to_end)
        source_game = self.store.get_game(decision.get("gameHash") or decision.get("gameId", "")) or {}
        source_state = source_game.get("state") or {}
        source_score = source_state.get("roundScore")
        branch_score = game.state.round_score.to_dict() if game.state.round_score is not None else None
        result["counterfactualComparison"] = {
            "sourceRoundScore": source_score,
            "branchRoundScore": branch_score,
            "sourceMatchScore": source_state.get("matchScore"),
            "branchMatchScore": list(game.state.match_score),
            "completed": game.state.current_player is None and game.state.round_score is not None,
        }
        return result


    def create_bid_episode_from_decision(
        self,
        decision_id: str,
        *,
        analysis_id: str | None = None,
        challenge_id: str | None = None,
    ) -> Any:
        """Reconstruct one archived buy/pass decision as a full bid episode.

        The analysis is rerun from the observer-safe auction snapshot. The
        recommended choice is then applied to a cloned trusted state solely to
        produce the counterfactual continuation and video; the archived source
        game remains immutable.
        """
        from baloot_platform.analysis.bidding import analyze_bid
        from baloot_platform.core.bid_scenario import BidEpisodeSpec, _finish_from_choice
        from baloot_platform.core.engine import BalootEngine
        from baloot_platform.history.provenance_v4 import game_serial

        payload = self.store.load_decision_payload(decision_id)
        if payload["decisionKind"] != "bid":
            raise ValueError("bid episode requires an archived bid decision")
        state = payload["state"].clone()
        actor = Seat.parse(payload["actor"])
        machine = payload.get("bidding")
        if machine is None:
            raise ValueError("archived bid decision has no auction machine")
        bundle = build_bundle(self.root, mode=str(payload.get("mode", "full")))
        analyses = self.store.list_analyses(decision_id)
        selected = next((item for item in analyses if item["analysisId"] == analysis_id), analyses[0] if analyses else None)
        if selected is None:
            selected = self.reanalyze_decision(decision_id, samples=96, seed=1)
        config = selected.get("config", {})
        typed = analyze_bid(
            state,
            __import__("copy").deepcopy(machine),
            actor,
            bundle.rules,
            requested_samples=int(config.get("samples", 96)),
            seed=int(config.get("seed", 1)),
        )
        final = _finish_from_choice(
            state,
            __import__("copy").deepcopy(machine),
            typed.recommended,
            BalootEngine(bundle.rules),
            seed=state.seed,
        )
        if final is None or final.round_score is None:
            raise ValueError("recommended bid continuation ended in all-pass; no scored episode is available")
        decision = self.store.get_decision(decision_id) or {}
        game = self.store.get_game(decision.get("gameHash") or decision.get("gameId", ""))
        record_hash = _hash(
            {
                "bundleHash": bundle.sha256,
                "decisionId": decision_id,
                "events": [event.to_dict() for event in final.events],
                "score": final.round_score.to_dict(),
            }
        )
        return BidEpisodeSpec(
            schema_version=2,
            episode_id=f"bid-episode-{decision_id.lower()}",
            bundle_hash=bundle.sha256,
            viewer=actor,
            state_at_bid=state,
            final_state=final,
            bidding_history_before=tuple(__import__("copy").deepcopy(machine.records)),
            bid_pause_index=len(machine.records),
            analysis=typed,
            title_ar=governed_episode_text(
                "video.review.bid_title",
                copy=self.presentation_copy,
            ),
            question_ar=governed_episode_text(
                "video.review.bid_question",
                copy=self.presentation_copy,
            ),
            game_serial=game.get("serial") if game else game_serial(record_hash),
            game_record_hash=game.get("gameHash") if game else record_hash,
            challenge_id=challenge_id,
            analysis_id=selected.get("analysisId"),
            model_id=None,
            decision_id=decision_id,
        )

    def create_episode_from_decision(
        self,
        decision_id: str,
        *,
        analysis_id: str | None = None,
        challenge_id: str | None = None,
    ) -> EpisodeSpec:
        payload = self.store.load_decision_payload(decision_id)
        if payload["decisionKind"] != "play":
            raise ValueError("video episodes currently require a play decision")
        state = payload["state"].clone()
        actor = Seat.parse(payload["actor"])
        bundle = build_bundle(self.root, mode=str(payload.get("mode", "full")))
        analyses = self.store.list_analyses(decision_id)
        chosen = next((item for item in analyses if item["analysisId"] == analysis_id), analyses[0] if analyses else None)
        if chosen is None:
            chosen = self.reanalyze_decision(decision_id)
        # Re-run to obtain the typed object used by the renderer. This is
        # deterministic for the stored config and preserves an auditable record.
        config = chosen.get("config", {})
        typed = analyze(
            InformationState.from_complete(state, actor),
            bundle.rules,
            bundle.convention,
            requested_samples=int(config.get("samples", 96)),
            exact_allocation_limit=int(config.get("exactAllocationLimit", 20_000)),
            exact_search_max_remaining=int(config.get("exactSearchRemaining", 10)),
            seed=int(config.get("seed", 1)),
        )
        final = state.clone()
        pause_index = len(final.played_cards)
        final_engine = __import__("baloot_platform.core.engine", fromlist=["BalootEngine"]).BalootEngine(bundle.rules)
        final_engine.play_card(final, typed.recommended_card)
        from baloot_platform.agents.profiles import AdaptivePlayStrategy, AgentProfileRegistry
        registry = AgentProfileRegistry.from_bundle(self.root)
        while final.current_player is not None:
            profile = registry.get(registry.default_lineup.get(final.current_player.name.lower(), registry.default_profile))
            strategy = AdaptivePlayStrategy(profile, bundle.convention)
            card = strategy.choose_card(final_engine, final, seed=final.seed + len(final.played_cards) * 7919)
            final_engine.play_card(final, card)
        decision = self.store.get_decision(decision_id) or {}
        game_identifier = decision.get("gameHash") or decision.get("gameId", "")
        game = self.store.get_game(game_identifier)
        source_decisions = self.store.list_decisions(game_identifier)
        decision_index: list[tuple[str, str]] = []
        bid_index = 0
        play_index = 0
        for source_decision in source_decisions:
            if source_decision.get("decisionKind") == "bid":
                bid_index += 1
                decision_index.append((f"bid:{bid_index}", source_decision["decisionId"]))
            elif source_decision.get("decisionKind") == "play":
                if play_index <= pause_index:
                    seat_value = int(Seat.parse(source_decision.get("actor", "south")))
                    decision_index.append((f"play:{play_index}:{seat_value}", source_decision["decisionId"]))
                play_index += 1
        decision_index.extend((
            ("question", decision_id),
            ("reveal", decision_id),
            ("continuation", decision_id),
        ))
        audits = decision.get("audits") or []
        return EpisodeSpec(
            schema_version=5,
            episode_id=f"episode-{decision_id.lower()}",
            bundle_hash=bundle.sha256,
            viewer=actor,
            state_before_pause=state,
            final_state=final,
            pause_before_play_index=pause_index,
            analysis=typed,
            title_ar=governed_episode_text(
                "video.review.play_title",
                copy=self.presentation_copy,
            ),
            question_ar=governed_episode_text(
                "video.review.play_question",
                copy=self.presentation_copy,
            ),
            game_serial=game.get("serial") if game else None,
            game_record_hash=game.get("gameHash") if game else None,
            challenge_id=challenge_id,
            analysis_id=chosen.get("analysisId"),
            model_id=None,
            audit_id=audits[0].get("auditId") if audits else None,
            decision_id=decision_id,
            decision_index=tuple(decision_index),
            audit_summary=copy.deepcopy(audits[0]) if audits else None,
        )


    def create_historical_replay_episode(
        self,
        identifier: str,
        *,
        decision_id: str,
        analysis_id: str | None = None,
    ) -> EpisodeSpec:
        """Build a full-round audit video from the immutable archived game.

        Unlike a challenge branch, the continuation is the exact historical
        sequence. The reveal may recommend a different card, but the following
        frames explicitly continue the recorded game. Every archived bid and
        card play is mapped to its exact renderer label and therefore to a
        stable video timecode.
        """
        game = self.store.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        if str(decision.get("gameHash") or decision.get("gameId")) not in {
            str(game.get("gameHash")), str(game.get("gameId"))
        }:
            raise ValueError("decision does not belong to the requested archived game")
        if decision.get("decisionKind") != "play":
            raise ValueError("full-round audit videos currently pause on a card-play decision")

        branch = self.create_episode_from_decision(
            decision_id, analysis_id=analysis_id, challenge_id=None
        )
        session = self.store.reopen_session(identifier)
        final_state = session.state.clone()
        if final_state.current_player is not None or len(final_state.played_cards) != 32:
            raise ValueError("historical replay video requires a completed 32-card round")

        source_decisions = self.store.list_decisions(game.get("gameHash") or game["gameId"], limit=10_000)
        decision_index: list[tuple[str, str]] = []
        bid_index = 0
        play_index = 0
        for source in source_decisions:
            kind = str(source.get("decisionKind"))
            if kind == "bid":
                bid_index += 1
                decision_index.append((f"bid:{bid_index}", str(source["decisionId"])))
            elif kind == "play":
                actor = Seat.parse(str(source.get("actor", "south")))
                decision_index.append((f"play:{play_index}:{int(actor)}", str(source["decisionId"])))
                play_index += 1
        decision_index.extend((("question", decision_id), ("reveal", decision_id), ("continuation", decision_id)))

        audits = decision.get("audits") or self.store.list_decision_audits(decision_id)
        return replace(
            branch,
            schema_version=max(6, int(branch.schema_version)),
            episode_id=f"historical-{str(game.get('serial') or game['gameId']).lower()}-{decision_id.lower()}",
            final_state=final_state,
            game_serial=game.get("serial"),
            game_record_hash=game.get("gameHash"),
            challenge_id=None,
            audit_id=audits[0].get("auditId") if audits else branch.audit_id,
            decision_id=decision_id,
            decision_index=tuple(decision_index),
            audit_summary=copy.deepcopy(audits[0]) if audits else branch.audit_summary,
            playback_mode="historical_actual",
            title_ar=governed_episode_text(
                "video.review.historical_title",
                copy=self.presentation_copy,
            ),
            question_ar=governed_episode_text(
                "video.review.historical_question",
                copy=self.presentation_copy,
            ),
        )
