"""On-demand observer-safe render preparation for replay-valid Research results."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.exact_state_view import build_observer_safe_exact_state_v1
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.challenge.analysis_pipeline import TraceAnalysisResultV1
from baloot_platform.challenge.evidence_v3 import RenderVariantV1, TruthMode
from baloot_platform.challenge.finalist_adjudication import (
    frozen_adaptive_finalist_profile_hash_v1,
)
from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1
from baloot_platform.challenge.perspective_pipeline import (
    build_decision_perspective_evidence,
    build_observer_projection,
)
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.challenge.video_production_batch import (
    HookedLaffaBatchProfileV1,
    evaluate_video_production_candidate_v1,
)
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.research.audit_trace import (
    COMPLETE_TRACE_MEDIA_TYPE,
    ResearchAuditTraceRecordV1,
)
from baloot_platform.research.contracts import ResearchConfigurationAuthorityV1
from baloot_platform.research.decision_audit import (
    load_decision_audit_v2,
    validate_governed_decision_audit_v2,
    validate_selected_decision_audit_v2,
)
from baloot_platform.research.decision_selection import evaluate_round_interestingness_v1, interest_admission_summary_v1
from baloot_platform.research.scenario_result import (
    SCENARIO_ARTIFACT_MEDIA_TYPES,
    validate_scenario_result_v4,
)
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.ortools_exact_state import _effective_prefix_hash
from baloot_platform.research.result_authority import (
    CP_SAT_QUALIFICATION_AUTHORITY,
    ORTOOLS_RESEARCH_BACKEND,
    ORTOOLS_RESULT_IDENTITY_KIND,
    ORTOOLS_RESULT_IDENTITY_V3_KIND,
    ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES,
    native_full_round_quality_issue_v1,
    native_featured_play_proof_supported_v1,
    validate_ortools_result_sections_v1,
    validate_trace_quality_evidence_v1,
)
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1
from baloot_platform.research.result_diversity import select_research_results
from baloot_platform.research.runtime_compatibility import (
    require_runtime_compatibility_passed_v1,
    runtime_compatibility_admission_v1,
)
from baloot_platform.video.batch_service import RENDER_BUNDLE_MEDIA_TYPE


TRACE_ANALYSIS_MEDIA_TYPE = "application/vnd.balootx.trace-analysis-v1+json"
_RESEARCH_ARTIFACT_REFERENCE_KEYS = frozenset(
    {"uri", "sha256", "bytes", "mediaType", "name", "ownerKind", "ownerHash"}
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _declared_hash_matches(value: Mapping[str, Any]) -> bool:
    core = copy.deepcopy(dict(value))
    declared = str(core.pop("sha256", ""))
    return bool(declared) and declared == _canonical_sha256(core)


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _recorded_scenario_action_v3(complete_trace: Mapping[str, Any], exact: Mapping[str, Any]) -> str | None:
    """Bind a V3 video's proof choice to the next card/action in its frozen round."""

    source_raw = complete_trace.get("exactTracePrefix")
    complete_raw = complete_trace.get("completeExactTracePrefix")
    kind = exact.get("decisionKind")
    field = {"bid": "auction_actions", "escalation": "escalation_actions",
             "project": "project_actions", "play": "play_actions"}.get(kind)
    if not isinstance(source_raw, Mapping) or not isinstance(complete_raw, Mapping) or field is None:
        return None
    source = ExactTracePrefixV1.build(source_raw)
    complete = ExactTracePrefixV1.build(complete_raw)
    if _effective_prefix_hash(source) != exact.get("prefixHash"):
        return None
    before, after = getattr(source, field), getattr(complete, field)
    if len(after) <= len(before):
        return None
    recorded = after[len(before)]
    if recorded.get("actor", exact.get("nextActor")) != exact.get("nextActor"):
        return None
    if kind == "play":
        code = LegalPlayActionV1(Card.parse(recorded["card"]), recorded.get("ekka", False)).code
        return code if code in {row.get("code") for row in exact.get("legalActions", [])} else None
    matches = [row["code"] for row in exact.get("legalActions", [])
               if row.get("action") == recorded.get("action")
               and all(row.get(key) == recorded[key] for key in
                       ("trumpSuit", "mode", "requestedMode", "projectId") if key in recorded)]
    return matches[0] if len(matches) == 1 else None


def _validate_optional_semantic_owner(
    reference: Mapping[str, Any],
    *,
    owner_kind: str,
    owner_hash: str,
    field: str,
) -> None:
    """Accept ownerless retained references, but never partial or false ownership."""

    declared_kind = reference.get("ownerKind")
    declared_hash = reference.get("ownerHash")
    if declared_kind is None and declared_hash is None:
        return
    if declared_kind != owner_kind or declared_hash != owner_hash:
        raise ValueError(f"Research {field} semantic owner identity mismatch")


def _validated_research_result_hash(
    result: Mapping[str, Any],
    *,
    result_code: str,
    trace_hash: str,
    analysis_hash: str,
    audit_analysis_hash: str,
    audit_hash: str,
) -> str:
    """Rebuild the exact immutable identity used by Research result admission."""

    provenance = dict(result.get("provenance") or {})
    shard_index = provenance.get("researchShardIndex")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValueError("Research result shard identity is invalid")
    identity: dict[str, Any] = {
        "schemaVersion": 1,
        "runCode": str(result.get("runCode") or ""),
        "shardId": str(provenance.get("researchShardId") or ""),
        "shardIndex": shard_index,
        "recipeHash": _sha256(
            provenance.get("recipeHash"), field="Research result recipe hash"
        ),
        "traceHash": _sha256(trace_hash, field="Research result trace hash"),
        "analysisHash": _sha256(analysis_hash, field="Research result analysis hash"),
        "auditAnalysisHash": _sha256(
            audit_analysis_hash,
            field="Research result audit-analysis hash",
        ),
        "auditHash": _sha256(audit_hash, field="Research result audit hash"),
        "auctionPlanHash": provenance.get("auctionCompilerPlanHash"),
    }
    if not identity["runCode"] or not identity["shardId"]:
        raise ValueError("Research result run or shard identity is missing")
    auction_hash = identity["auctionPlanHash"]
    if auction_hash is not None:
        identity["auctionPlanHash"] = _sha256(
            auction_hash, field="Research result auction plan hash"
        )
    parity_status = str(provenance.get("decisionScoreParityStatus") or "")
    if provenance.get("auditAnalysisHash") != identity["auditAnalysisHash"]:
        raise ValueError("Research result audit-analysis identity is inconsistent")
    primary_hash = provenance.get("primaryDecisionScoreAnalysisHash")
    parity_hash = provenance.get("decisionScoreParityHash")
    if parity_status == "matched":
        if primary_hash != identity["auditAnalysisHash"]:
            raise ValueError(
                "Research primary decision-score analysis is not the audit authority"
            )
        identity.update(
            {
                "primaryDecisionScoreAnalysisHash": _sha256(
                    primary_hash,
                    field="Research primary decision-score analysis hash",
                ),
                "decisionScoreParityHash": _sha256(
                    parity_hash, field="Research decision-score parity hash"
                ),
            }
        )
    elif parity_status == "not_applicable":
        if primary_hash is not None or parity_hash is not None:
            raise ValueError("Research decision-score parity identity is inconsistent")
    else:
        raise ValueError("Research decision-score parity status is invalid")

    declared = _sha256(result.get("resultHash"), field="Research result identity hash")
    if _canonical_sha256(identity) != declared:
        raise ValueError("Research result identity checksum mismatch")
    expected_code = f"LAF-{declared[:10].upper()}"
    if (
        str(result.get("resultCode") or "") != expected_code
        or str(result_code).upper() != expected_code
    ):
        raise ValueError("Research result code does not match its identity")
    return declared


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_items(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(str(item) for item in value)


def _audit_publication_policy(
    *, decision_id: str, blockers: list[str]
) -> dict[str, Any]:
    """Build checksum-bound evidence that an audit render cannot be published."""

    normalized = list(dict.fromkeys(str(item) for item in blockers if str(item)))
    decision_core = {
        "schemaVersion": 1,
        "profileHash": None,
        "evidenceHash": None,
        "decisionId": decision_id,
        "gatePassed": False,
        "qualifiedDecision": False,
        "blockers": normalized,
    }
    decision_evidence = {
        **decision_core,
        "sha256": _canonical_sha256(decision_core),
    }
    video_core = {
        "schemaVersion": 1,
        "profileHash": None,
        "candidate": {"decisionId": decision_id, "auditOnly": True},
        "gatePassed": False,
        "blockers": normalized,
    }
    video_evidence = {**video_core, "sha256": _canonical_sha256(video_core)}
    policy_core = {
        "schemaVersion": 1,
        "decisionSelection": decision_evidence,
        "videoProduction": video_evidence,
        "blockers": normalized,
    }
    return {**policy_core, "sha256": _canonical_sha256(policy_core)}


_RETIRED_NON_SEMANTIC_RULE_SECTIONS = frozenset({"analysis_defaults", "policies"})
_REPLAY_ONLY_BIDDING_HEURISTIC_KEYS = frozenset(
    {
        "sun_rank_weights",
        "sun_extra_suit_card_weight",
        "sun_extra_suit_card_after",
        "hokum_trump_rank_weights",
        "hokum_extra_trump_card_weight",
        "hokum_extra_trump_card_after",
        "hokum_off_trump_ace_ten_weight",
    }
)


def _frozen_rulebook(
    payload: Mapping[str, Any],
    expected_hash: str,
    *,
    replay_only_defaults: Mapping[str, Any] | None = None,
) -> RuleBook:
    """Load exact historical rules while tolerating only retired non-gameplay metadata."""

    frozen = copy.deepcopy(dict(payload))
    if hashlib.sha256(_canonical_bytes(frozen)).hexdigest() != expected_hash:
        raise ValueError("frozen trace rules payload does not match its rules hash")
    try:
        return RuleBook(frozen)
    except ValueError as original_error:
        retired = set(frozen).intersection(_RETIRED_NON_SEMANTIC_RULE_SECTIONS)
        if not retired:
            raise
        for section in retired:
            frozen.pop(section)
        if replay_only_defaults is not None:
            historical = dict(dict(frozen.get("bidding") or {}).get("heuristics") or {})
            current = dict(
                dict(replay_only_defaults.get("bidding") or {}).get("heuristics") or {}
            )
            for key in _REPLAY_ONLY_BIDDING_HEURISTIC_KEYS:
                if key not in historical and key in current:
                    historical[key] = copy.deepcopy(current[key])
            frozen["bidding"]["heuristics"] = historical
        try:
            rules = RuleBook(frozen)
        except ValueError:
            raise original_error
        # Replay identity remains the checksum of the complete immutable historical
        # payload. The removed sections never governed engine transitions.
        rules.sha256 = expected_hash
        return rules


class ResearchRenderPreparationServiceV1:
    """Create semantic previews and immutable render bundles only on request."""

    descriptor = PluginDescriptor(
        id="research_render_adapter_v1",
        kind="research_render_adapter",
        version="1.0.0",
        description="Prepare replay-valid Research results for explicit Remotion rendering",
        capabilities=(
            "qualified_and_rejected_results",
            "audit_render_labeling",
            "decision_clip",
            "hooked_flagship",
            "observer_safe_preview",
            "on_demand_only",
        ),
        resource_class="io",
        deterministic=True,
        timeout_seconds=60,
        maximum_memory_mb=512,
        contract_id="research_render_adapter",
        contract_version="1",
        safety_guards=(
            "authoritative_legality",
            "observer_safe",
            "frozen_provenance",
            "on_demand_only",
        ),
        side_effects=("artifact_write",),
    )

    def __init__(
        self,
        *,
        research: Any,
        artifact_store: Any,
        publication_service: Any,
        renderer: Any,
        rules: Any,
        learning_store: Any | None = None,
        locator_service: Any | None = None,
        configuration_store: Any | None = None,
    ) -> None:
        self.research = research
        self.artifact_store = artifact_store
        self.publication_service = publication_service
        self.renderer = renderer
        self.rules = rules
        self.learning_store = learning_store
        self.locator_service = locator_service
        self.configuration_store = configuration_store

    def _load(
        self,
        reference: Mapping[str, Any],
        *,
        expected_media_type: str,
    ) -> Mapping[str, Any]:
        unknown = sorted(set(reference) - _RESEARCH_ARTIFACT_REFERENCE_KEYS)
        missing = sorted({"uri", "sha256", "bytes", "mediaType"} - set(reference))
        if unknown or missing:
            raise ValueError(
                "Research artifact reference is not closed: "
                f"missing={missing}, unknown={unknown}"
            )
        if reference.get("mediaType") != expected_media_type:
            raise ValueError("Research artifact media type mismatch")
        expected_bytes = reference.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 1
        ):
            raise ValueError("Research artifact byte count is invalid")
        payload = self.artifact_store.get(str(reference.get("uri") or ""))
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != str(
            reference.get("sha256") or ""
        ):
            raise ValueError("Research artifact integrity check failed")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Research artifact is not valid UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("Research artifact must contain one JSON object")
        if _canonical_bytes(value) != payload:
            raise ValueError("Research artifact is not canonical JSON")
        return value

    @staticmethod
    def _is_current_adaptive_analysis(
        document: Mapping[str, Any],
    ) -> bool:
        return (
            str(document.get("analyzerId") or "")
            == "adaptive_finalist_trace_analyzer_v1"
            and str(document.get("analyzerVersion") or "") == "1.3.0"
        )

    def _frozen_expected_analysis_profile_hash(
        self,
        result: Mapping[str, Any],
        trace: CompleteGameTraceV1,
        analysis_document: Mapping[str, Any],
        *,
        audit_analysis: bool = False,
    ) -> str | None:
        current_adaptive = self._is_current_adaptive_analysis(analysis_document)
        if not audit_analysis and not current_adaptive:
            return None
        store = getattr(self.research, "store", None)
        build_context = getattr(self.research, "_build_recipe_and_context", None)
        validate_context = getattr(
            self.research,
            "_assert_run_execution_context",
            None,
        )
        if (
            store is None
            or not callable(getattr(store, "get_recipe", None))
            or not callable(getattr(store, "get_run", None))
            or not callable(build_context)
            or not callable(validate_context)
        ):
            raise ValueError(
                "current adaptive Research analysis lacks frozen execution authority"
            )
        recipe_value = store.get_recipe(str(result.get("recipeCode") or ""))
        run = store.get_run(str(result.get("runCode") or ""))
        if not isinstance(recipe_value, Mapping) or not isinstance(run, Mapping):
            raise ValueError(
                "current adaptive Research recipe or run authority is unavailable"
            )
        recipe, context = build_context(recipe_value)
        validate_context(run, recipe, context)
        provenance = dict(result.get("provenance") or {})
        if (
            str(run.get("recipeCode") or "") != recipe.code
            or str(run.get("recipeHash") or "") != recipe.sha256
            or str(result.get("recipeCode") or "") != recipe.code
            or str(provenance.get("recipeHash") or "") != recipe.sha256
        ):
            raise ValueError("Research result recipe authority mismatch")
        uses_finalist_policy = getattr(
            self.research,
            "_uses_finalist_primary_policy",
            None,
        )
        analysis_mode = (
            "deep"
            if audit_analysis
            and callable(uses_finalist_policy)
            and uses_finalist_policy(recipe)
            else (
                str(
                    ResearchConfigurationAuthorityV1.build(
                        recipe.configuration
                    ).identities["analyzer"]["id"]
                )
                if isinstance(recipe, ResearchRecipeV4)
                else str(recipe.resources.get("analysisMode") or "")
            )
        )
        analyzers = (
            context.analyzers
            if context is not None
            else getattr(self.research, "analyzers", {})
        )
        analyzer = analyzers.get(analysis_mode)
        if analyzer is None:
            raise ValueError("Research result analysis mode is unavailable")
        descriptor = analyzer.descriptor
        if audit_analysis and (
            str(analysis_document.get("analyzerId") or "")
            != str(getattr(descriptor, "id", ""))
            or str(analysis_document.get("analyzerVersion") or "")
            != str(getattr(descriptor, "version", ""))
            or str(analysis_document.get("profileHash") or "")
            != str(analyzer.profile.sha256)
        ):
            raise ValueError("Research result audit-analysis authority mismatch")
        if current_adaptive and (
            str(getattr(descriptor, "id", "")) != "adaptive_finalist_trace_analyzer_v1"
            or str(getattr(descriptor, "version", "")) != "1.3.0"
        ):
            raise ValueError(
                "Research result analyzer does not match the adaptive artifact"
            )
        expected_hash = str(analyzer.profile.sha256)
        if (
            current_adaptive
            and expected_hash != frozen_adaptive_finalist_profile_hash_v1(trace)
        ):
            raise ValueError(
                "Research adaptive profile does not match the frozen trace authority"
            )
        return expected_hash if current_adaptive else None

    def _load_audit_trace(
        self,
        reference: Mapping[str, Any],
        *,
        expected_trace_hash: str,
    ) -> CompleteGameTraceV1:
        """Load canonical CAS bytes and verify both artifact and trace identity."""

        value = self._load(
            reference,
            expected_media_type=COMPLETE_TRACE_MEDIA_TYPE,
        )
        trace = CompleteGameTraceV1.from_dict(value)
        if trace.sha256 != expected_trace_hash:
            raise ValueError("Research audit complete-trace identity mismatch")
        return trace

    def _rules_for_trace(self, trace: CompleteGameTraceV1) -> RuleBook:
        """Resolve the immutable rules snapshot that produced a stored trace."""

        if self.rules.sha256 == trace.rules_hash:
            return self.rules
        snapshot = dict(trace.trusted_final_state.get("configurationSnapshot") or {})
        rules_component = dict(
            dict(snapshot.get("components") or {}).get("rules") or {}
        )
        payload = rules_component.get("payload")
        if not isinstance(payload, Mapping) and self.configuration_store is not None:
            matches = [
                item
                for item in self.configuration_store.list_profiles(
                    family="rules", limit=1_000
                )
                if str(item.payload_sha256) == trace.rules_hash
            ]
            if matches:
                payload = matches[0].payload
        if not isinstance(payload, Mapping):
            raise ValueError(
                "trace rules payload is unavailable from both its snapshot and versioned configuration"
            )
        return _frozen_rulebook(
            payload,
            trace.rules_hash,
            replay_only_defaults=getattr(self.rules, "data", None),
        )

    @staticmethod
    def _select_audit(
        result: Mapping[str, Any], audits: list[Mapping[str, Any]], ordinal: int | None
    ) -> Mapping[str, Any]:
        if ordinal is not None:
            matches = [item for item in audits if int(item["ordinal"]) == int(ordinal)]
            if len(matches) != 1:
                raise KeyError(ordinal)
            chosen = matches[0]
        else:
            critical = set(result.get("criticalDecisionIds") or ())
            eligible = [
                item
                for item in audits
                if item.get("decisionId") in critical
                and item.get("decisionKind") == "play"
                and not bool(item.get("forced"))
            ]
            if not eligible:
                eligible = [
                    item
                    for item in audits
                    if item.get("decisionKind") == "play"
                    and not bool(item.get("forced"))
                ]
            if not eligible:
                raise ValueError(
                    "Research result has no non-forced play for a video hook"
                )
            chosen = max(
                eligible,
                key=lambda item: (
                    len(item.get("tacticalEvidence") or ()),
                    float(item.get("confidence", 0.0)),
                    -int(item["ordinal"]),
                ),
            )
        if chosen.get("decisionKind") != "play":
            raise ValueError(
                "the current interactive video supports card-play decisions"
            )
        if bool(chosen.get("forced")):
            raise ValueError("a forced play cannot be prepared as a decision challenge")
        return chosen

    @staticmethod
    def _publication_eligibility(
        result: Mapping[str, Any],
    ) -> tuple[bool, str, list[str]]:
        """Retain the legacy status projection without fixed quality thresholds.

        The active render path uses ``_publication_governance`` below, which
        verifies the frozen decision-selection and video-production evidence.
        """

        disposition = str(result.get("disposition") or "")
        if disposition not in {"qualified", "rejected"}:
            raise ValueError(
                "only a completed replay-valid Research result can be prepared for video"
            )
        qualification = dict(result.get("qualification") or {})
        failures = list(_string_items(qualification.get("failures")))
        eligible = disposition == "qualified" and not failures
        return eligible, disposition, failures

    @staticmethod
    def _select_perspective(
        analysis: TraceAnalysisResultV1,
        decision: Any,
        observer_seat: Seat | str | int | None,
    ) -> Any:
        observer = (
            decision.acting_seat if observer_seat is None else Seat.parse(observer_seat)
        )
        truth_mode = (
            TruthMode.ACTOR_OBSERVER
            if observer == decision.acting_seat
            else TruthMode.SEAT_EXPERIENCE
        )
        matches = [
            item
            for item in analysis.laffa_evidence.perspectives
            if item.decision_id == decision.decision_id
            and item.observer_seat == observer
            and item.truth_mode == truth_mode
        ]
        if len(matches) != 1:
            raise ValueError(
                "Research decision lacks exactly one stored perspective for the requested observer"
            )
        return matches[0]

    @staticmethod
    def _select_generated_perspective(
        perspectives: Any,
        decision: Any,
        observer_seat: Seat | str | int | None,
    ) -> Any:
        observer = (
            decision.acting_seat if observer_seat is None else Seat.parse(observer_seat)
        )
        truth_mode = (
            TruthMode.ACTOR_OBSERVER
            if observer == decision.acting_seat
            else TruthMode.SEAT_EXPERIENCE
        )
        matches = [
            item
            for item in perspectives
            if item.observer_seat == observer and item.truth_mode == truth_mode
        ]
        if len(matches) != 1:
            raise ValueError(
                "Research trace lacks exactly one perspective for the requested observer"
            )
        return matches[0]

    @staticmethod
    def _rendition_type(evidence: Any) -> str:
        if evidence.truth_mode == TruthMode.ACTOR_OBSERVER:
            return "actor_challenge"
        if evidence.observer_seat == evidence.acting_seat.partner:
            return "partner_inference"
        return "opponent_prediction"

    @staticmethod
    def _frozen_publication_profiles(
        trace: CompleteGameTraceV1,
    ) -> tuple[
        SelectionProfileV1 | None,
        HookedLaffaBatchProfileV1 | None,
        list[str],
    ]:
        blockers: list[str] = []
        snapshot = dict(trace.trusted_final_state.get("configurationSnapshot") or {})
        if not snapshot:
            return None, None, ["frozen_configuration_snapshot_missing"]
        if not _declared_hash_matches(snapshot):
            return None, None, ["frozen_configuration_snapshot_hash_mismatch"]
        if str(snapshot.get("bundleHash") or "") != trace.bundle_hash:
            return None, None, ["frozen_configuration_bundle_hash_mismatch"]

        component = dict(dict(snapshot.get("components") or {}).get("challenge") or {})
        payload = component.get("payload")
        if not isinstance(payload, Mapping):
            return None, None, ["frozen_challenge_profile_missing"]
        if str(component.get("sha256") or "") != _canonical_sha256(payload):
            return None, None, ["frozen_challenge_profile_hash_mismatch"]

        selection_profile = None
        raw_selection = payload.get("selection_policy")
        if not isinstance(raw_selection, Mapping):
            blockers.append("frozen_decision_selection_profile_missing")
        else:
            try:
                selection_profile = SelectionProfileV1.build(raw_selection)
            except (KeyError, TypeError, ValueError):
                blockers.append("frozen_decision_selection_profile_invalid")

        video_profile = None
        raw_video = payload.get("video_production_batch")
        if not isinstance(raw_video, Mapping):
            blockers.append("frozen_video_production_profile_missing")
        else:
            try:
                video_profile = HookedLaffaBatchProfileV1.build(raw_video)
            except (KeyError, TypeError, ValueError):
                blockers.append("frozen_video_production_profile_invalid")
        return selection_profile, video_profile, blockers

    @staticmethod
    def _decision_selection_evidence(
        result: Mapping[str, Any],
        *,
        analysis: TraceAnalysisResultV1,
        decision_id: str,
        profile: SelectionProfileV1 | None,
    ) -> tuple[dict[str, Any], Mapping[str, Any] | None, list[str]]:
        blockers: list[str] = []
        score_evidence = result.get("scoreEvidence")
        qualification = dict(result.get("qualification") or {})
        qualified_summary = qualification.get("decisionSelection")
        if not isinstance(score_evidence, Mapping) or not isinstance(
            qualified_summary, Mapping
        ):
            core = {
                "schemaVersion": 1,
                "profileHash": profile.sha256 if profile is not None else None,
                "evidenceHash": None,
                "decisionId": decision_id,
                "gatePassed": False,
                "qualifiedDecision": False,
                "blockers": ["decision_selection_evidence_missing"],
            }
            return (
                {**core, "sha256": _canonical_sha256(core)},
                None,
                list(core["blockers"]),
            )

        score_document = copy.deepcopy(dict(score_evidence))
        if not _declared_hash_matches(score_document):
            blockers.append("decision_selection_score_evidence_hash_mismatch")
        selection_summary = score_document.get("decisionSelectionQualification")
        rows = score_document.get("decisionChallengeScores")
        if not isinstance(selection_summary, Mapping) or not isinstance(rows, list):
            blockers.append("decision_selection_evidence_missing")
            selection_summary = {}
            rows = []

        selection_document = copy.deepcopy(dict(selection_summary))
        declared_evidence_hash = str(selection_document.pop("sha256", ""))
        reconstructed = {**selection_document, "decisions": copy.deepcopy(rows)}
        if not declared_evidence_hash or declared_evidence_hash != _canonical_sha256(
            reconstructed
        ):
            blockers.append("decision_selection_evidence_hash_mismatch")

        declared_profile_hash = str(selection_document.get("profileHash") or "")
        linked_profile_hash = str(qualified_summary.get("profileHash") or "")
        score_profile_hash = str(score_document.get("decisionProfileHash") or "")
        if (
            profile is None
            or not declared_profile_hash
            or {
                declared_profile_hash,
                linked_profile_hash,
                score_profile_hash,
                profile.sha256 if profile is not None else "",
            }
            != {profile.sha256 if profile is not None else ""}
        ):
            blockers.append("decision_selection_profile_hash_mismatch")
        if declared_evidence_hash != str(qualified_summary.get("evidenceHash") or ""):
            blockers.append("decision_selection_qualification_hash_mismatch")
        if str(selection_document.get("analysisHash") or "") != analysis.sha256:
            blockers.append("decision_selection_analysis_hash_mismatch")

        matching_rows = [
            dict(item)
            for item in rows
            if isinstance(item, Mapping)
            and str(item.get("decisionId") or "") == decision_id
        ]
        row = matching_rows[0] if len(matching_rows) == 1 else None
        if row is None:
            blockers.append("selected_decision_evidence_missing")
        gate_passed = bool(row and row.get("gatePassed") is True)
        if not gate_passed:
            blockers.append("selected_decision_gate_failed")
        qualified_ids = {
            str(item)
            for item in _string_items(selection_document.get("qualifiedDecisionIds"))
        }
        linked_qualified_ids = {
            str(item)
            for item in _string_items(qualified_summary.get("qualifiedDecisionIds"))
        }
        if qualified_ids != linked_qualified_ids:
            blockers.append("decision_selection_qualified_ids_mismatch")
        qualified_decision = (
            decision_id in qualified_ids and decision_id in linked_qualified_ids
        )
        if not qualified_decision:
            blockers.append("selected_decision_not_qualified")
        if (
            selection_document.get("passed") is not True
            or qualified_summary.get("passed") is not True
        ):
            blockers.append("decision_selection_not_passed")

        blockers = list(dict.fromkeys(blockers))
        core = {
            "schemaVersion": 1,
            "profileHash": declared_profile_hash or None,
            "evidenceHash": declared_evidence_hash or None,
            "decisionId": decision_id,
            "gatePassed": gate_passed,
            "qualifiedDecision": qualified_decision,
            "blockers": blockers,
        }
        return {**core, "sha256": _canonical_sha256(core)}, row, blockers

    @classmethod
    def _publication_governance(
        cls,
        result: Mapping[str, Any],
        *,
        audit: Mapping[str, Any],
        analysis: TraceAnalysisResultV1,
        trace: CompleteGameTraceV1,
        evidence: Any,
        authority_blockers: tuple[str, ...] = (),
    ) -> tuple[bool, str, list[str], list[str], dict[str, Any]]:
        _, disposition, failures = cls._publication_eligibility(result)
        qualification = dict(result.get("qualification") or {})
        blockers: list[str] = []
        blockers.extend(str(item) for item in authority_blockers)
        if disposition != "qualified" or qualification.get("qualified") is not True:
            blockers.append("result_not_qualified")
        if not isinstance(qualification.get("failures"), (list, tuple)):
            blockers.append("qualification_failures_invalid")
        if failures:
            blockers.append("qualification_failures_present")
        is_actor = evidence.truth_mode == TruthMode.ACTOR_OBSERVER
        if not is_actor:
            blockers.append("non_actor_observer")

        selection_profile, video_profile, profile_blockers = (
            cls._frozen_publication_profiles(trace)
        )
        blockers.extend(profile_blockers)
        decision_evidence, decision_row, decision_blockers = (
            cls._decision_selection_evidence(
                result,
                analysis=analysis,
                decision_id=str(audit["decisionId"]),
                profile=selection_profile,
            )
        )
        blockers.extend(decision_blockers)

        raw_score = (decision_row or {}).get("score")
        score = dict(raw_score) if isinstance(raw_score, Mapping) else {}
        decision_id = str(audit.get("decisionId") or "")
        result_critical = decision_id in {
            str(item) for item in _string_items(result.get("criticalDecisionIds"))
        }
        analysis_critical = decision_id in {
            str(item) for item in analysis.laffa_evidence.critical_decision_ids
        }
        if result_critical != analysis_critical:
            blockers.append("critical_decision_identity_mismatch")
        candidate = {
            "decisionKind": str(audit.get("decisionKind") or ""),
            "critical": result_critical and analysis_critical,
            "truthLabel": str(
                (decision_row or {}).get("truthLabel") or evidence.truth_label.value
            ),
            "selectionScore": _safe_float(score.get("score")),
            "confidence": _safe_float(
                (decision_row or {}).get("confidence", evidence.confidence),
                default=_safe_float(evidence.confidence),
            ),
            "compatibilityBlockers": [
                str(item) for item in _string_items(audit.get("compatibilityBlockers"))
            ],
        }
        video_blockers = (
            list(evaluate_video_production_candidate_v1(candidate, video_profile))
            if video_profile is not None
            else ["profile_unavailable"]
        )
        blockers.extend(f"video_production_gate:{item}" for item in video_blockers)
        video_core = {
            "schemaVersion": 1,
            "profileHash": video_profile.sha256 if video_profile is not None else None,
            "candidate": candidate,
            "gatePassed": video_profile is not None and not video_blockers,
            "blockers": video_blockers,
        }
        video_evidence = {**video_core, "sha256": _canonical_sha256(video_core)}
        blockers = list(dict.fromkeys(blockers))
        policy_core = {
            "schemaVersion": 1,
            "decisionSelection": decision_evidence,
            "videoProduction": video_evidence,
            "blockers": blockers,
        }
        policy = {**policy_core, "sha256": _canonical_sha256(policy_core)}
        return not blockers, disposition, failures, blockers, policy

    def _build_and_store(
        self,
        *,
        trace: CompleteGameTraceV1,
        decision: Any,
        evidence: Any,
        analysis_hash: str,
        locator: Any,
        reveal_policy: str,
        preset: str,
        reason_key: str,
        rules: RuleBook,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build one renderer-neutral bundle for every governed source path."""

        projection = build_observer_projection(
            TraceDecisionProjectionStoreV1(trace, rules),
            decision.decision_id,
            evidence.observer_seat,
        )
        public_lifecycle = None
        observer_lifecycle = None
        if preset == "hooked_flagship":
            # Keep this import lazy: analysis_catalog imports the video package,
            # whose scenario adapter imports Research result contracts.
            from baloot_platform.challenge.analysis_catalog import (
                complete_observer_safe_lifecycle,
            )

            public_lifecycle, observer_lifecycle = complete_observer_safe_lifecycle(
                trace,
                observer=evidence.observer_seat,
                pause_before_play_index=len(
                    dict(projection.information_state).get("plays") or ()
                ),
            )
        rendition_type = self._rendition_type(evidence)
        prompt_contract = str(
            evidence.prompt_contract
            or (
                "choose_acting_action"
                if evidence.truth_mode == TruthMode.ACTOR_OBSERVER
                else ""
            )
        )
        variant = RenderVariantV1(
            analysis_hash,
            decision.decision_id,
            evidence.observer_seat,
            rendition_type,
            prompt_contract,
            reveal_policy,
            locator.universal_link,
        )
        bundle = self.renderer.build_perspective_render_request(
            projection=projection,
            evidence=evidence,
            variant=variant,
            effective_config_sha256=trace.bundle_hash,
            game_serial=locator.code,
            legal_actions=(
                tuple(dict(item) for item in decision.legal_actions)
                if evidence.truth_mode == TruthMode.ACTOR_OBSERVER
                else ()
            ),
            reveal_reason_key=reason_key,
            export_preset=preset,
            complete_public_lifecycle=public_lifecycle,
            complete_observer_lifecycle=observer_lifecycle,
            source_metadata=source_metadata,
        )
        payload = _canonical_bytes(bundle.to_dict())
        stored = self.artifact_store.put(
            payload,
            media_type=RENDER_BUNDLE_MEDIA_TYPE,
            name=f"{bundle.bundle_sha256}.render-bundle.json",
        )
        reference = {
            "ownerKind": "render_bundle_v1",
            "ownerHash": bundle.bundle_sha256,
            **dict(stored),
        }
        preview = {
            "bundleHash": bundle.bundle_sha256,
            "sceneCount": len(bundle.value["scenes"]),
            "timeline": bundle.value["timeline"],
            "scenes": bundle.value["scenes"],
        }
        return reference, preview

    def _scenario_render_authority(
        self,
        result: Mapping[str, Any],
        *,
        exact_state: Mapping[str, Any],
    ) -> tuple[RuleBook, dict[str, Any], str, str, dict[str, Any]]:
        if self.configuration_store is None:
            raise RuntimeError("Research V4 configuration store is unavailable")
        recipe = self.research.get_recipe(str(result["recipeCode"]))
        if not isinstance(recipe, Mapping):
            raise ValueError("Research V4 recipe authority is unavailable")
        provenance = dict(result.get("provenance") or {})
        identity = dict(provenance.get("resultIdentity") or {})
        recipe_hash = str(recipe.get("sha256") or "")
        recipe_code = str(recipe.get("code") or "")
        if (
            recipe_code != str(result.get("recipeCode") or "")
            or recipe_hash != str(provenance.get("recipeHash") or "")
            or recipe_hash != str(identity.get("recipeHash") or "")
            or recipe_code != str(identity.get("recipeCode") or "")
        ):
            raise ValueError("Research V4 result recipe authority mismatch")
        configuration = dict((recipe or {}).get("configuration") or {})
        revision_id = str(configuration.get("bundleRevisionId") or "")
        if not revision_id:
            raise ValueError("Research V4 recipe lacks its bundle revision")
        configuration_hash = _canonical_sha256(configuration)
        if (
            str(provenance.get("configurationAuthorityHash") or "")
            != configuration_hash
            or str(identity.get("configurationAuthorityHash") or "")
            != configuration_hash
        ):
            raise ValueError("Research V4 result configuration authority mismatch")
        bundle = self.configuration_store.get_bundle(revision_id)
        if bundle is None:
            raise ValueError("Research V4 bundle revision is unavailable")
        if str(getattr(bundle, "revision_id", revision_id)) != revision_id:
            raise ValueError("Research V4 bundle revision identity mismatch")
        component_revisions = dict(bundle.component_revisions)

        authority = ResearchConfigurationAuthorityV1.build(configuration)
        expected_revisions = {
            family: str(value["revisionId"])
            for family, value in authority.component_revisions.items()
        }
        if component_revisions != expected_revisions:
            raise ValueError("Research V4 bundle component revision mismatch")
        if (
            str(getattr(bundle, "bundle_id", "")) != authority.bundle_id
            or str(getattr(bundle, "version", "")) != authority.bundle_version
        ):
            raise ValueError("Research V4 bundle identity drift")
        bundle_content_hash = _canonical_sha256(
            {
                "mode": str(getattr(bundle, "mode", "")),
                "components": component_revisions,
            }
        )
        if bundle_content_hash != str(getattr(bundle, "content_sha256", "")):
            raise ValueError("Research V4 bundle content hash mismatch")

        payloads: dict[str, dict[str, Any]] = {}
        payload_hashes: dict[str, str] = {}
        required_families = set(component_revisions)
        for family in sorted(required_families):
            component_revision_id = str(component_revisions.get(family) or "")
            if not component_revision_id:
                raise ValueError(
                    f"Research V4 {family} profile revision is unavailable"
                )
            component = self.configuration_store.get_profile(component_revision_id)
            if component is None:
                raise ValueError(
                    f"Research V4 {family} profile revision is unavailable"
                )
            if (
                str(getattr(component, "revision_id", component_revision_id))
                != component_revision_id
                or str(getattr(component, "family", family)) != family
            ):
                raise ValueError(
                    f"Research V4 {family} profile revision identity mismatch"
                )
            payload = copy.deepcopy(dict(component.payload))
            payload_hash = _canonical_sha256(payload)
            if payload_hash != str(component.payload_sha256):
                raise ValueError(
                    f"Research V4 {family} profile payload hash mismatch"
                )
            if payload_hash != str(
                authority.component_revisions[family]["payloadSha256"]
            ):
                raise ValueError(
                    f"Research V4 {family} frozen profile substitution detected"
                )
            payloads[family] = payload
            payload_hashes[family] = payload_hash

        rules = _frozen_rulebook(payloads["rules"], payload_hashes["rules"])
        proof_identities = dict(
            dict(result.get("proofOutcome") or {}).get("identities") or {}
        )
        exact_rules_hash = str(exact_state.get("rulesHash") or "")
        expected_bundle_hash = str(proof_identities.get("bundleHash") or "")
        if expected_bundle_hash != authority.semantic_bundle_hash:
            raise ValueError("Research V4 proof bundle authority mismatch")
        semantic_hash = _canonical_sha256(
            {
                "bundleProfile": {
                    "id": authority.bundle_id,
                    "version": authority.bundle_version,
                    "status": "published",
                },
                "mode": str(bundle.mode),
                "components": payload_hashes,
            }
        )
        if semantic_hash != authority.semantic_bundle_hash:
            raise ValueError("Research V4 semantic bundle payload drift")
        if (
            not expected_bundle_hash
            or str(proof_identities.get("rulesHash") or "") != rules.sha256
            or exact_rules_hash != rules.sha256
        ):
            raise ValueError("Research V4 frozen rules authority mismatch")
        return (rules, payloads["video"], payload_hashes["video"], expected_bundle_hash,
                dict(payloads["challenge"].get("scenarioResearchV4") or {}))

    @staticmethod
    def _scenario_publication_governance(
        result: Mapping[str, Any],
        *,
        proof: Mapping[str, Any],
        exact_state_id: str,
        video_profile_hash: str,
        video_profile: Mapping[str, Any] | None = None,
        full_round: bool = False,
        recorded_action: str | None = None,
        recorded_terminal_score: Mapping[str, Any] | None = None,
        proof_request: Mapping[str, Any] | None = None,
        quality_trace: Mapping[str, Any] | None = None,
        quality_policy: ScenarioPlayingQualityPolicyV1 | None = None,
        rules_hash: str | None = None,
        rules: RuleBook | None = None,
        quality_settings: Mapping[str, Any] | None = None,
        _validated_quality_audit: Mapping[str, Any] | None = None,
    ) -> tuple[bool, str, list[str], list[str], dict[str, Any]]:
        disposition = str(result.get("disposition") or "rejected")
        qualification = dict(result.get("qualification") or {})
        failures = [str(item) for item in qualification.get("failures") or ()]
        blockers: list[str] = []
        quality_hash = None
        interest_verdict = None
        interestingness = dict(dict(quality_settings or {}).get("qualityExecution") or {}).get("videoInterestingness")
        if (interestingness or {}).get("schemaVersion") == 4:
            blockers.append("interestingness_verified_evidence_unavailable")
        if full_round or (interestingness or {}).get("schemaVersion") == 4:
            quality_issue = native_full_round_quality_issue_v1(
                str(dict(result.get("researchAuthority") or {}).get("backendId") or ""),
                quality_policy.mode if quality_policy is not None else None,
            ) if full_round else None
            if quality_trace is not None and quality_policy is not None and rules_hash is not None:
                evidence = quality_trace.get("playingQualityEvidence")
                if isinstance(evidence, Mapping) and evidence.get("receipt", {}).get("status") == "complete":
                    if _validated_quality_audit is None:
                        validate_trace_quality_evidence_v1(
                            quality_trace, rules_hash=rules_hash, required_policy=quality_policy, rules=rules,
                            require_family_opening=True,
                            required_proof_profiles=dict(quality_settings or {}).get("proofBudgets", {}).get("profiles"),
                            required_search_profiles=dict(quality_settings or {}).get("searchBudgets", {}).get("profiles"),
                            required_execution=(ScenarioQualityExecutionBudgetV1.build(quality_settings["qualityExecution"])
                                                if quality_settings and quality_settings.get("qualityExecution") else None),
                        )
                    else:
                        # Only prepare supplies this freshly verified service projection;
                        # it is not an API input or a caller-authored proof receipt.
                        audit = _validated_quality_audit
                        if (
                            audit.get("status") != "complete"
                            or audit.get("scope") != "all_decisions"
                            or audit.get("resultCode") != result.get("resultCode")
                            or audit.get("resultHash") != result.get("resultHash")
                            or audit.get("completeTraceHash") != _canonical_sha256(quality_trace)
                            or audit.get("qualityReceiptHash") != evidence["receipt"].get("sha256")
                            or audit.get("sha256") != _canonical_sha256(
                                {key: value for key, value in audit.items() if key != "sha256"}
                            )
                        ):
                            raise ValueError("Research render verified quality audit binding mismatch")
                    quality_hash = evidence["receipt"]["sha256"]
                    interestingness = dict(dict(quality_settings or {}).get("qualityExecution") or {}).get("videoInterestingness")
                    if interestingness is not None:
                        screen = evaluate_round_interestingness_v1(
                            evidence["receipt"], recorded_terminal_score or {}, interestingness,
                            featured_state_id=exact_state_id,
                        )
                        interest_verdict = screen
                        if screen["policy"]["schemaVersion"] == 4:
                            blockers.remove("interestingness_verified_evidence_unavailable")
                            if qualification.get("interestAdmission") != interest_admission_summary_v1(screen):
                                blockers.append("interestingness_result_binding_mismatch")
                        blockers.extend(screen["blockers"])
            if quality_issue is not None:
                blockers.append(quality_issue["code"])
            elif quality_policy is not None and quality_hash is None:
                blockers.append("all_decision_quality_evidence_unavailable")
            if quality_hash is not None and (
                qualification.get("allDecisionPolicyAuditComplete") is not True
                or qualification.get("playingQualityEvidenceHash") != quality_hash
            ):
                blockers.append("all_decision_quality_result_binding_unavailable")
        if disposition != "qualified" or qualification.get("qualified") is not True:
            blockers.append("result_not_qualified")
        if failures:
            blockers.append("qualification_failures_present")
        if proof.get("completionStatus") not in {
            "exact_complete",
            "bounded_complete",
            "policy_complete",
            # A complete counterproof is terminal evidence that the requested
            # claim is false; it is not proof-budget incompleteness.
            "proof_failed",
        }:
            blockers.append("scenario_proof_incomplete")
        if proof.get("claimSatisfied") is not True:
            blockers.append("scenario_claim_not_satisfied")
        native_v3 = str(dict(result.get("researchAuthority") or {}).get("backendId") or "") == "ortools_cp_sat_v3"
        if native_v3 and proof_request is not None and (
            any(proof.get(key) != proof_request.get(key) for key in
                ("utility", "targetTeam", "successThreshold"))
            or dict(proof.get("identities") or {}).get("proofRequestHash")
            != proof_request.get("sha256")
        ):
            raise ValueError("V3 render proof differs from frozen recipe")
        roots = proof.get("rootActions")
        selected = proof.get("selectedAction")
        # Frozen historical render admission keeps its original meaning;
        # only new native V3 flagships claim proof-backed criticality.
        critical = not native_v3
        if native_v3 and (
            proof.get("completionStatus") == "exact_complete"
            and proof.get("claimSatisfied") is True
            and isinstance(roots, list)
            and len(roots) >= 2
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("action"), str)
                and type(row.get("value")) is int
                and type(row.get("satisfies")) is bool
                for row in roots
            )
            and len({row["action"] for row in roots}) == len(roots)
        ):
            winners = [row for row in roots if row["satisfies"]]
            if (len(winners) == 1 and winners[0]["action"] == selected
                    and (not full_round or selected == recorded_action)):
                critical = winners[0]["value"] > max(
                    row["value"] for row in roots if row["action"] != selected
                )
        if native_v3 and full_round and not critical:
            blockers.append("critical_decision_not_proven")
        if native_v3 and full_round and critical:
            winner = (recorded_terminal_score or {}).get("winner")
            target = proof.get("targetTeam")
            threshold = proof.get("successThreshold")
            utility = proof.get("utility")
            terminal_verified = False
            if (target in {"north_south", "west_east"}
                    and type(threshold) in {int, float}
                    and math.isfinite(threshold)):
                if utility in {"round_win", "contract_success"}:
                    terminal_verified = (
                        winner in {"north_south", "west_east"}
                        and (1 if winner == target else -1) >= threshold
                    )
                elif utility == "match_win" and "matchWinner" in (recorded_terminal_score or {}):
                    match_winner = recorded_terminal_score["matchWinner"]
                    if match_winner in (None, "north_south", "west_east"):
                        # No winner at this round's end is utility zero, not
                        # a prediction about a later deal or the round winner.
                        value = 0 if match_winner is None else (1 if match_winner == target else -1)
                        terminal_verified = value >= threshold and any(
                            row["action"] == selected and row["value"] == value
                            for row in roots
                        )
                elif utility == "awarded_point_margin":
                    points = (recorded_terminal_score or {}).get("awardedPoints")
                    if (isinstance(points, (list, tuple)) and len(points) == 2
                            and all(type(value) is int for value in points)):
                        margin = (points[0] - points[1] if target == "north_south"
                                  else points[1] - points[0])
                        terminal_verified = (
                            margin >= threshold
                            and any(row["action"] == selected and row["value"] == margin
                                    for row in roots)
                        )
            if not terminal_verified:
                blockers.append("recorded_terminal_utility_unverified")
            if (utility == "awarded_point_margin"
                    and not str(dict((video_profile or {}).get("localizations") or {}).get(
                        "video.analysis.unique_awarded_margin") or "").strip()):
                blockers.append("score_swing_explanation_unavailable")
            if utility == "awarded_point_margin":
                featured_native = False
                if quality_hash is not None and quality_trace is not None:
                    evidence = quality_trace["playingQualityEvidence"]
                    receipt = evidence["receipt"]
                    play = receipt["postAuction"]["projectPlay"]
                    rows = [row for row in play["decisions"]
                            if row.get("exactStateId") == exact_state_id
                            and row.get("action") == recorded_action]
                    if len(rows) == 1:
                        artifact = play["artifacts"].get(rows[0]["proofArtifactHash"], {})
                        prefix = quality_trace.get("exactTracePrefix")
                        featured_native = (
                            isinstance(prefix, Mapping)
                            and isinstance(prefix.get("playActions"), (list, tuple))
                            and rows[0].get("decisionKind") == "play"
                            and type(rows[0].get("playIndex")) is int
                            and rows[0]["playIndex"] == len(prefix["playActions"])
                            and receipt.get("qualityExecution", {}).get("featuredPlayNativeProof") is True
                            and native_featured_play_proof_supported_v1(artifact)
                        )
                if not featured_native:
                    blockers.append("featured_play_native_proof_unavailable")
        compatibility = result.get("runtimeCompatibility")
        if isinstance(compatibility, Mapping):
            blockers.extend(
                runtime_compatibility_admission_v1(compatibility)["blockers"]
            )
        proof_hash = str(
            proof.get("semanticSha256") or proof.get("sha256") or ""
        )
        proof_profile_hash = str(
            dict(proof.get("identities") or {}).get("proofProfileHash") or ""
        )
        decision_core = {
            "schemaVersion": 1,
            "profileHash": proof_profile_hash or None,
            "evidenceHash": proof_hash or None,
            "decisionId": exact_state_id,
            "gatePassed": not blockers,
            "qualifiedDecision": not blockers,
            "blockers": list(blockers),
        }
        decision_evidence = {
            **decision_core,
            "sha256": _canonical_sha256(decision_core),
        }
        candidate = {
            "decisionKind": str(
                dict(result.get("scenarioProjection") or {}).get("decisionKind") or ""
            ),
            "critical": critical,
            "truthLabel": str(result.get("truthLabel") or "rejected"),
            "selectionScore": float(result.get("score", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
            "compatibilityBlockers": list(blockers),
        }
        video_core = {
            "schemaVersion": 1,
            "profileHash": video_profile_hash,
            "candidate": candidate,
            "gatePassed": not blockers,
            "blockers": list(blockers),
        }
        video_evidence = {
            **video_core,
            "sha256": _canonical_sha256(video_core),
        }
        policy_core = {
            "schemaVersion": 1,
            "decisionSelection": decision_evidence,
            "videoProduction": video_evidence,
            "blockers": list(blockers),
            **({"qualityEvidenceHash": quality_hash} if quality_hash is not None else {}),
            **({"interestVerdict": interest_verdict} if interest_verdict is not None
               and interest_verdict["policy"]["schemaVersion"] == 4 else {}),
        }
        policy = {**policy_core, "sha256": _canonical_sha256(policy_core)}
        return not blockers, disposition, failures, blockers, policy

    @staticmethod
    def _require_interest_admission(interest_policy, disposition, blockers):
        failures = [code for code in blockers if code.startswith("interestingness_")]
        if failures and (disposition == "qualified" or interest_policy.get("schemaVersion") == 4):
            raise ValueError("governed pre-render interestingness screen rejected this round: " + ", ".join(failures))

    @staticmethod
    def _public_scenario_publication_policy(
        policy: Mapping[str, Any], *, result_code: str
    ) -> dict[str, Any]:
        public_policy = copy.deepcopy(dict(policy))
        interest = public_policy.pop("interestVerdict", None)
        if interest is not None:
            public_policy["interestAdmission"] = {"passed": interest["passed"], "evidenceHash": interest["sha256"]}
        decision = dict(public_policy["decisionSelection"])
        decision["decisionId"] = result_code
        decision["evidenceHash"] = None
        decision_core = dict(decision)
        decision_core.pop("sha256", None)
        decision["sha256"] = _canonical_sha256(decision_core)
        public_policy["decisionSelection"] = decision
        policy_core = dict(public_policy)
        policy_core.pop("sha256", None)
        public_policy["sha256"] = _canonical_sha256(policy_core)
        return public_policy

    def _prepare_scenario(
        self,
        result_code: str,
        result: Mapping[str, Any],
        *,
        decision_ordinal: int | None,
        observer_seat: Seat | str | int | None,
        reveal_policy: str,
        export_preset: str | None,
    ) -> dict[str, Any]:
        if decision_ordinal not in {None, 1}:
            raise ValueError("Research V4 scenario has exactly one decision")
        preset = export_preset or (
            "decision_clip" if decision_ordinal is not None else "hooked_flagship"
        )
        if preset not in {"decision_clip", "hooked_flagship"}:
            raise ValueError("unsupported Research V4 scenario render preset")
        if str(result.get("resultCode") or "") != result_code:
            raise ValueError("Research V4 result code mismatch")
        identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
        authority = dict(result.get("researchAuthority") or {})
        ortools_result = (
            identity.get("kind")
            in {ORTOOLS_RESULT_IDENTITY_KIND, ORTOOLS_RESULT_IDENTITY_V3_KIND}
            or authority.get("backendId") == ORTOOLS_RESEARCH_BACKEND
            or authority.get("qualificationAuthority")
            == CP_SAT_QUALIFICATION_AUTHORITY
        )
        if ortools_result:
            recipe = self.research.get_recipe(str(result.get("recipeCode") or ""))
            if not isinstance(recipe, Mapping):
                raise ValueError("Research V4 recipe authority is unavailable")
            validated = validate_ortools_result_sections_v1(result, recipe=recipe)
        else:
            validated = validate_scenario_result_v4(
                result,
                artifact_store=self.artifact_store,
            )
        if ortools_result:
            require_runtime_compatibility_passed_v1(
                dict(validated.get("runtimeCompatibility") or {})
            )
        if dict(validated.get("scenarioProjection") or {}).get(
            "exactStateAvailable"
        ) is not True:
            raise ValueError(
                "research result has no replay-valid trace; audit rendering is unavailable"
            )
        artifacts = dict(validated["artifacts"])
        proof_artifact_owned = copy.deepcopy(dict(artifacts["proofOutcome"]))
        complete_trace: dict[str, Any] | None = None
        if ortools_result:
            complete_trace = dict(
                self._load(
                    dict(artifacts["completeTrace"]),
                    expected_media_type=ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[
                        "completeTrace"
                    ],
                )
            )
            exact_value = complete_trace.get("exactNormalization")
            if not isinstance(exact_value, Mapping):
                raise ValueError("OR-Tools Research exact-state document is unavailable")
            exact = copy.deepcopy(dict(exact_value))
        else:
            exact = dict(
                self._load(
                    dict(artifacts["exactState"]),
                    expected_media_type=SCENARIO_ARTIFACT_MEDIA_TYPES["exactState"],
                )
            )
        proof = dict(validated["proofOutcome"])
        private_proof = dict(
            self._load(
                proof_artifact_owned,
                expected_media_type=(
                    ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES["proofOutcome"]
                    if ortools_result
                    else SCENARIO_ARTIFACT_MEDIA_TYPES["proofOutcome"]
                ),
            )
        )
        render_proof = copy.deepcopy(proof)
        if ortools_result:
            if (exact.get("exactStateId") != identity.get("exactStateId")
                    or exact.get("stateHash") != identity.get("exactStateHash")):
                raise ValueError("Research V4 render exact state differs from its result identity")
            for key, value in (("exactStateId", identity["exactStateId"]),
                               ("stateHash", identity["exactStateHash"])):
                if any(dict(document.get("identities") or {}).get(key) not in (None, value)
                       for document in (proof, private_proof)):
                    raise ValueError("Research V4 render proof references a different exact state")
            # Public native receipts omit state identities; retain their verified private binding.
            render_proof["identities"] = {
                **dict(proof["identities"]),
                "exactStateId": identity["exactStateId"],
                "stateHash": identity["exactStateHash"],
            }
        else:
            private_identities = dict(private_proof.get("identities") or {})
            if any(private_identities.get(key) != exact.get(key)
                   for key in ("exactStateId", "stateHash")):
                raise ValueError("Research V4 render proof references a different exact state")
            render_proof["identities"] = {
                **dict(proof["identities"]),
                "exactStateId": private_identities["exactStateId"],
                "stateHash": private_identities["stateHash"],
            }
        proof_artifact = {
            key: value
            for key, value in proof_artifact_owned.items()
            if key in {"uri", "sha256", "bytes", "mediaType", "name"}
        }
        projection = dict(validated["scenarioProjection"])
        decision_kind = str(projection.get("decisionKind") or "")
        if decision_kind not in {"bid", "escalation", "project", "play"}:
            raise ValueError(
                "Research V4 render preparation requires a supported decision"
            )
        acting = str(exact.get("nextActor") or "")
        observer = (
            acting if observer_seat is None else Seat.parse(observer_seat).name.lower()
        )
        if observer != acting:
            raise ValueError(
                "Research V4 render preparation requires the acting observer"
            )
        complete_normalization = (complete_trace or {}).get("completeExactNormalization")
        terminal_state = (complete_normalization or {}).get("state")
        recorded_terminal_score = (terminal_state or {}).get("roundScore")
        (
            scenario_rules,
            video_profile,
            video_profile_hash,
            semantic_bundle_hash,
            quality_settings,
        ) = self._scenario_render_authority(validated, exact_state=exact)
        interest_policy = dict(dict(quality_settings or {}).get("qualityExecution") or {}).get("videoInterestingness") or {}
        (
            publication_eligible,
            disposition,
            failures,
            blockers,
            policy,
        ) = self._scenario_publication_governance(
            validated,
            proof=proof,
            exact_state_id=str(exact["exactStateId"]),
            video_profile_hash=video_profile_hash,
            video_profile=video_profile,
            full_round=preset == "hooked_flagship",
            recorded_action=(_recorded_scenario_action_v3(complete_trace, exact)
                             if complete_trace is not None and authority.get("backendId") == "ortools_cp_sat_v3"
                             else None),
            recorded_terminal_score=(recorded_terminal_score
                                     if isinstance(recorded_terminal_score, Mapping) else None),
            proof_request=(dict(recipe.get("proof") or {})
                           if authority.get("backendId") == "ortools_cp_sat_v3" else None),
            quality_trace=complete_trace,
            quality_policy=(ScenarioPlayingQualityPolicyV1.build(recipe["playingQuality"])
                            if ortools_result and recipe.get("playingQuality") is not None else None),
            rules_hash=scenario_rules.sha256,
            rules=scenario_rules,
            quality_settings=quality_settings,
            _validated_quality_audit=(
                self.research.get_playing_quality_audit(result_code)
                if (preset == "hooked_flagship" or interest_policy.get("schemaVersion") == 4)
                and identity.get("kind") == ORTOOLS_RESULT_IDENTITY_V3_KIND
                and isinstance(complete_trace, Mapping)
                and dict(dict(complete_trace.get("playingQualityEvidence") or {}).get("receipt") or {}).get("status") == "complete"
                else None
            ),
        )
        render_purpose = (
            "content_candidate" if publication_eligible else "research_audit"
        )
        self._require_interest_admission(interest_policy, disposition, blockers)
        decision_reference = {
            "resultCode": result_code,
            "decisionOrdinal": 1,
        }
        public_exact = build_observer_safe_exact_state_v1(
            exact_state=exact,
            scenario_projection=projection,
            observer_seat=observer,
            decision_reference=decision_reference,
        )
        projection_for_render = {
            **projection,
            "decisionReference": decision_reference,
            "publicProjectionHash": str(public_exact["publicProjectionHash"]),
        }
        public_policy = self._public_scenario_publication_policy(
            policy, result_code=result_code
        )
        enrollment_uri = f"balootx://research/{result_code}/decisions/1"
        source = {
            "sourceSchemaVersion": 3,
            "kind": "research_scenario_result",
            "code": result_code,
            "decisionOrdinal": 1,
            "decisionReference": decision_reference,
            "publicProjectionHash": str(public_exact["publicProjectionHash"]),
            "qualificationStatus": disposition,
            "renderPurpose": render_purpose,
            "qualificationFailures": failures,
            "publicationBlockers": blockers,
            "observerSeat": observer,
            "auditOnly": not publication_eligible,
            "publicationEligible": publication_eligible,
            "qrActionable": False,
            "publicationPolicy": public_policy,
            "rulesHash": str(exact["rulesHash"]),
            "proofReference": result_code,
            "proofCompletionStatus": str(proof["completionStatus"]),
            "proofClaimSatisfied": bool(proof["claimSatisfied"]),
            "proofAchievedMode": (
                "incomplete"
                if proof["completionStatus"] == "proof_incomplete"
                else str(proof["achievedMode"])
            ),
        }
        authority_core = {
            "schemaVersion": 1,
            "audience": "operator_audit",
            "resultCode": result_code,
            "resultHash": str(validated["resultHash"]),
            "decisionOrdinal": 1,
            "exactStateId": str(exact["exactStateId"]),
            "stateHash": str(exact["stateHash"]),
            "exactStateHash": str(exact["sha256"]),
            "rulesHash": str(exact["rulesHash"]),
            "configurationBundleHash": semantic_bundle_hash,
            "scenarioHash": str(dict(validated["provenance"])["scenarioHash"]),
            "proofSemanticSha256": str(
                proof_artifact["sha256"]
                if ortools_result
                else private_proof.get("semanticSha256") or private_proof["sha256"]
            ),
            "proofArtifactSha256": str(proof_artifact["sha256"]),
            "constraintEvidenceHash": str(
                dict(
                    validated[
                        "scenarioSatisfaction"
                        if ortools_result
                        else "constraintEvidence"
                    ]
                )["sha256"]
            ),
        }
        authority_receipt = {
            **authority_core,
            "sha256": _canonical_sha256(authority_core),
        }
        bundle = self.renderer.build_scenario_render_request(
            exact_state=exact,
            proof_outcome=render_proof,
            scenario_projection=projection_for_render,
            effective_config_sha256=semantic_bundle_hash,
            game_serial=result_code,
            observer_seat=observer,
            enrollment_uri=enrollment_uri,
            export_preset=preset,
            source_metadata=source,
            rules=scenario_rules,
            video_profile=video_profile,
            trusted_principal_variation=private_proof.get("principalVariation"),
            trusted_complete_trace=(
                complete_trace
                if ortools_result
                else None
            ),
        )
        payload = _canonical_bytes(bundle.to_dict())
        stored = self.artifact_store.put(
            payload,
            media_type=RENDER_BUNDLE_MEDIA_TYPE,
            name=f"{bundle.bundle_sha256}.render-bundle.json",
        )
        reference = {
            "ownerKind": "render_bundle_v1",
            "ownerHash": bundle.bundle_sha256,
            **dict(stored),
        }
        preview = {
            "bundleHash": bundle.bundle_sha256,
            "sceneCount": len(bundle.value["scenes"]),
            "timeline": bundle.value["timeline"],
            "scenes": bundle.value["scenes"],
        }
        suffix = "decision-01" if preset == "decision_clip" else "full"
        basename = f"balootx-{result_code.lower()}-scenario-{decision_kind}-{suffix}"
        if not publication_eligible:
            basename += "-audit"
        return {
            "schemaVersion": 1,
            "resultCode": result_code,
            "decisionOrdinal": 1,
            "decisionId": result_code,
            "observerSeat": observer,
            "exportPreset": preset,
            "qualificationStatus": disposition,
            "publicationEligible": publication_eligible,
            "renderPurpose": render_purpose,
            "qualificationFailures": failures,
            "publicationBlockers": blockers,
            "publicationPolicy": public_policy,
            "locator": (
                {
                    "schemaVersion": 1,
                    "code": result_code,
                    "universalLink": f"/c/{result_code}",
                    "observerSeat": observer,
                    "actingSeat": acting,
                    "promptContract": "choose_acting_action",
                    "revealPolicy": reveal_policy,
                    "expiresAt": None,
                }
                if publication_eligible
                else None
            ),
            "bundleArtifact": reference,
            "proofArtifact": proof_artifact,
            "proofArtifactSha256": str(proof_artifact["sha256"]),
            "authorityReceipt": authority_receipt,
            "basename": basename,
            "preview": preview,
            "source": copy.deepcopy(source),
        }

    def prepare(
        self,
        result_code: str,
        *,
        decision_ordinal: int | None = None,
        observer_seat: Seat | str | int | None = None,
        reveal_policy: str = "after_commit",
        export_preset: str | None = None,
    ) -> dict[str, Any]:
        result = self.research.get_result(result_code)
        if result is None:
            raise KeyError(result_code)
        if result.get("classification") == "scenario_v4":
            return self._prepare_scenario(
                result_code,
                result,
                decision_ordinal=decision_ordinal,
                observer_seat=observer_seat,
                reveal_policy=reveal_policy,
                export_preset=export_preset,
            )
        canonical_decisions = self.research.get_decisions(result_code)
        preset = export_preset or (
            "decision_clip" if decision_ordinal is not None else "hooked_flagship"
        )
        if preset not in {"decision_clip", "hooked_flagship"}:
            raise ValueError("unsupported Research render preset")

        artifacts = dict(result["artifacts"])
        trace_reference = dict(artifacts["completeTrace"])
        trace = CompleteGameTraceV1.from_dict(
            self._load(
                trace_reference,
                expected_media_type=COMPLETE_TRACE_MEDIA_TYPE,
            )
        )
        _validate_optional_semantic_owner(
            trace_reference,
            owner_kind="complete_trace_v1",
            owner_hash=trace.sha256,
            field="complete trace",
        )
        replay_rules = self._rules_for_trace(trace)
        content_analysis_reference = dict(artifacts["analysis"])
        content_analysis_document = self._load(
            content_analysis_reference,
            expected_media_type=TRACE_ANALYSIS_MEDIA_TYPE,
        )
        if (
            str(result.get("traceHash") or "") != trace.sha256
            or str(content_analysis_document.get("traceHash") or "") != trace.sha256
        ):
            raise ValueError("Research result trace and analysis membership mismatch")
        expected_content_analysis_profile_hash = (
            self._frozen_expected_analysis_profile_hash(
                result,
                trace,
                content_analysis_document,
            )
        )
        content_analysis = TraceAnalysisResultV1.from_dict(
            content_analysis_document,
            ledger=trace.decision_ledger,
            expected_adaptive_profile_hash=expected_content_analysis_profile_hash,
        )
        if content_analysis.trace_hash != trace.sha256:
            raise ValueError("Research parsed analysis belongs to another trace")
        _validate_optional_semantic_owner(
            content_analysis_reference,
            owner_kind="trace_analysis_v1",
            owner_hash=content_analysis.sha256,
            field="analysis",
        )
        selection = dict(
            dict(result.get("scoreEvidence") or {}).get(
                "decisionSelectionQualification"
            )
            or {}
        )
        if str(selection.get("analysisHash") or "") != content_analysis.sha256:
            raise ValueError(
                "Research result does not contain the parsed content-analysis identity"
            )

        audit_analysis_reference = dict(artifacts.get("auditAnalysis") or {})
        if set(audit_analysis_reference) != _RESEARCH_ARTIFACT_REFERENCE_KEYS:
            raise ValueError(
                "current Research audit-analysis reference is not closed or owned"
            )
        audit_analysis_document = self._load(
            audit_analysis_reference,
            expected_media_type=TRACE_ANALYSIS_MEDIA_TYPE,
        )
        if str(audit_analysis_document.get("traceHash") or "") != trace.sha256:
            raise ValueError("Research audit analysis belongs to another trace")
        expected_audit_analysis_profile_hash = (
            self._frozen_expected_analysis_profile_hash(
                result,
                trace,
                audit_analysis_document,
                audit_analysis=True,
            )
        )
        audit_analysis = TraceAnalysisResultV1.from_dict(
            audit_analysis_document,
            ledger=trace.decision_ledger,
            expected_adaptive_profile_hash=expected_audit_analysis_profile_hash,
        )
        if audit_analysis.trace_hash != trace.sha256:
            raise ValueError("Research parsed audit analysis belongs to another trace")
        if (
            audit_analysis_reference.get("ownerKind") != "trace_analysis_v1"
            or audit_analysis_reference.get("ownerHash") != audit_analysis.sha256
        ):
            raise ValueError("Research audit analysis semantic owner identity mismatch")
        expected_analysis_profile_hash = expected_audit_analysis_profile_hash
        analysis = audit_analysis
        analysis_reference = audit_analysis_reference
        historical_adaptive = (
            analysis.analyzer_id == "adaptive_finalist_trace_analyzer_v1"
            and analysis.analyzer_version != "1.3.0"
        )
        audit_document = load_decision_audit_v2(
            self.artifact_store,
            dict(artifacts["decisionAudit"]),
            expected_trace_hash=trace.sha256,
            ledger=trace.decision_ledger,
            expected_engine_version=trace.engine_version,
            expected_bundle_hash=trace.bundle_hash,
            expected_rules_hash=trace.rules_hash,
        )
        validate_governed_decision_audit_v2(
            audit_document,
            trace=trace,
            rules=replay_rules,
        )
        result_hash = _validated_research_result_hash(
            result,
            result_code=result_code,
            trace_hash=trace.sha256,
            analysis_hash=content_analysis.sha256,
            audit_analysis_hash=analysis.sha256,
            audit_hash=str(audit_document["sha256"]),
        )
        audits = [dict(item) for item in canonical_decisions]
        if audits != [dict(item) for item in audit_document["decisions"]]:
            raise ValueError(
                "Research render decisions are not the canonical retained audit"
            )
        audit = self._select_audit(result, audits, decision_ordinal)
        decision_id = str(audit["decisionId"])
        decision = next(
            item
            for item in trace.decision_ledger.records
            if item.decision_id == decision_id
        )
        validate_selected_decision_audit_v2(audit, analysis=analysis)
        perspective = self._select_perspective(
            analysis,
            decision,
            observer_seat,
        )
        (
            publication_eligible,
            disposition,
            failures,
            publication_blockers,
            publication_policy,
        ) = self._publication_governance(
            result,
            audit=audit,
            analysis=content_analysis,
            trace=trace,
            evidence=perspective,
            authority_blockers=(
                ("historical_adaptive_analysis_unbound",) if historical_adaptive else ()
            ),
        )
        observer_name = perspective.observer_seat.name.lower()
        is_actor = perspective.truth_mode == TruthMode.ACTOR_OBSERVER
        render_purpose = (
            "content_candidate" if publication_eligible else "research_audit"
        )
        source = {
            "sourceSchemaVersion": 2,
            "kind": "research_result",
            "code": result_code,
            "resultHash": result_hash,
            "decisionOrdinal": int(audit["ordinal"]),
            "decisionId": decision_id,
            "exactStateId": decision_id,
            "qualificationStatus": disposition,
            "renderPurpose": render_purpose,
            "qualificationFailures": list(dict.fromkeys(failures)),
            "publicationBlockers": publication_blockers,
            "observerSeat": observer_name,
            "auditOnly": not publication_eligible,
            "publicationEligible": publication_eligible,
            "qrActionable": bool(is_actor and publication_eligible),
            "publicationPolicy": publication_policy,
            "traceHash": trace.sha256,
            "analysisHash": analysis.sha256,
            "auditHash": str(audit_document["sha256"]),
            "analysisProfileHash": analysis.profile_hash,
            "analyzerId": analysis.analyzer_id,
            "analyzerVersion": analysis.analyzer_version,
            "domainHash": dict(result.get("provenance") or {}).get("domainHash"),
            "rulesHash": trace.rules_hash,
        }
        analysis_reference = {
            **analysis_reference,
            "ownerKind": "trace_analysis_v1",
            "ownerHash": analysis.sha256,
            "mediaType": TRACE_ANALYSIS_MEDIA_TYPE,
        }
        reason_key = (
            "video.analysis.expert_reason" if is_actor else "video.observer.insight"
        )
        if is_actor and publication_eligible:
            published = self.publication_service.publish_actor_challenge(
                trace,
                analysis_reference=analysis_reference,
                decision_id=decision_id,
                reveal_policy=reveal_policy,
                answer_explanation=self.renderer.localized_reveal_reason(
                    reason_key, {}
                ),
                rules=replay_rules,
                expected_adaptive_profile_hash=expected_analysis_profile_hash,
            )
            locator = published.locator
        else:
            locator_service = self.locator_service or getattr(
                self.publication_service, "locator_service", None
            )
            if locator_service is None:
                raise RuntimeError(
                    "observer-safe Research locator service is unavailable"
                )
            locator = locator_service.publish(
                trace,
                decision_id=decision_id,
                observer_seat=perspective.observer_seat,
                prompt_contract=str(
                    perspective.prompt_contract
                    or ("choose_acting_action" if is_actor else "")
                ),
                reveal_policy=reveal_policy,
                rules=replay_rules,
            )
        reference, preview = self._build_and_store(
            trace=trace,
            decision=decision,
            evidence=perspective,
            analysis_hash=analysis.laffa_evidence.sha256,
            locator=locator,
            reveal_policy=reveal_policy,
            preset=preset,
            reason_key=reason_key,
            rules=replay_rules,
            source_metadata=source,
        )
        basename = (
            f"balootx-{result_code.lower()}-decision-{int(audit['ordinal']):02d}"
            if preset == "decision_clip"
            else f"balootx-{result_code.lower()}-laffa"
        )
        if not publication_eligible:
            basename = f"{basename}-audit"
        return {
            "schemaVersion": 1,
            "resultCode": result_code,
            "decisionOrdinal": int(audit["ordinal"]),
            "decisionId": decision_id,
            "observerSeat": observer_name,
            "exportPreset": preset,
            "qualificationStatus": disposition,
            "publicationEligible": publication_eligible,
            "renderPurpose": render_purpose,
            "qualificationFailures": failures,
            "publicationBlockers": publication_blockers,
            "publicationPolicy": publication_policy,
            "locator": locator.to_dict(),
            "bundleArtifact": reference,
            "basename": basename,
            "preview": preview,
            "source": copy.deepcopy(source),
        }

    def prepare_selected_run(
        self,
        run_code: str,
        *,
        observer_seat: Seat | str | int | None = None,
        reveal_policy: str = "after_commit",
        export_preset: str | None = None,
    ) -> dict[str, Any]:
        """Prepare the frozen diversity-selected V4 video target for one run."""

        run = self.research.get_run(run_code)
        if run is None:
            raise KeyError(run_code)
        if run.get("status") != "completed":
            raise ValueError("Research render selection requires a completed run")
        raw_recipe = self.research.get_recipe(str(run.get("recipeCode") or ""))
        if raw_recipe is None or raw_recipe.get("schemaVersion") != 4:
            raise ValueError("Research render selection requires a V4 recipe")
        recipe_payload = copy.deepcopy(dict(raw_recipe))
        recipe_payload.pop("lifecycleStatus", None)
        recipe = ResearchRecipeV4.build(recipe_payload)
        if (
            str(run.get("recipeCode") or "") != recipe.code
            or str(run.get("recipeHash") or "") != recipe.sha256
        ):
            raise ValueError("Research render selection recipe authority mismatch")
        if recipe.render_policy["mode"] != "on_demand":
            raise ValueError("Research rendering is disabled by the frozen recipe")
        video_count = int(recipe.render_policy["videoCount"])
        if video_count < 1:
            raise ValueError("Research recipe requests no videos")

        output_receipt = run.get("outputReceipt")
        if not isinstance(output_receipt, Mapping):
            raise ValueError("Research run output receipt is unavailable")
        output_core = copy.deepcopy(dict(output_receipt))
        output_hash = str(output_core.pop("sha256", ""))
        if (
            output_hash != _canonical_sha256(output_core)
            or output_core.get("recipeHash") != recipe.sha256
            or output_core.get("output") != dict(recipe.output)
            or output_core.get("renderPolicy") != dict(recipe.render_policy)
            or output_core.get("requestedResultCount")
            != int(recipe.output["resultCount"])
            or output_core.get("requestedVideoCount") != video_count
            or output_core.get("retainedResultCount") != run.get("resultCount")
            or output_core.get("qualifiedResultCount") != run.get("qualifiedCount")
            or output_core.get("selectedResultCount") != run.get("selectedCount")
            or output_core.get("resultTargetSatisfied")
            is not (
                int(run.get("selectedCount") or 0)
                >= int(recipe.output["resultCount"])
            )
            or run.get("requestedResultCount") != int(recipe.output["resultCount"])
            or run.get("requestedVideoCount") != video_count
        ):
            raise ValueError("Research run output receipt authority mismatch")

        results: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.research.list_results(
                run_code=run_code,
                sort="result_code_asc",
                offset=offset,
                limit=500,
            )
            results.extend(dict(item) for item in page["items"])
            if not page.get("hasMore"):
                break
            offset = int(page["nextOffset"])
        diversity_payload = dict(recipe.diversity)
        diversity_payload.pop("sha256", None)
        selection = select_research_results(
            results,
            profile=QualityDiversityProfileV1.build(diversity_payload),
            requested=int(recipe.output["resultCount"]),
        )
        if selection.to_dict() != run.get("diversitySelection"):
            raise ValueError("Research run diversity selection authority mismatch")
        selected_codes = list(selection.selected_result_codes[:video_count])
        if len(selected_codes) != video_count:
            raise ValueError(
                "Research run has fewer diversity-selected results than requested videos"
            )

        prepared = [
            self.prepare(
                code,
                observer_seat=observer_seat,
                reveal_policy=reveal_policy,
                export_preset=export_preset or "hooked_flagship",
            )
            for code in selected_codes
        ]
        items = [
            {
                "bundleArtifact": copy.deepcopy(dict(item["bundleArtifact"])),
                "proofArtifact": copy.deepcopy(dict(item["proofArtifact"])),
                "authorityReceipt": copy.deepcopy(dict(item["authorityReceipt"])),
                "basename": str(item["basename"]),
                "source": copy.deepcopy(dict(item["source"])),
                **(
                    {
                        "locator": {
                            "schemaVersion": 1,
                            "code": str(item["resultCode"]),
                            "universalLink": str(
                                dict(item["locator"])["universalLink"]
                            ),
                        }
                    }
                    if item.get("publicationEligible") is True
                    and item.get("renderPurpose") == "content_candidate"
                    and dict(item["source"]).get("auditOnly") is False
                    else {}
                ),
            }
            for item in prepared
        ]
        core = {
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": str(run.get("runHash") or ""),
            "recipeCode": recipe.code,
            "recipeHash": recipe.sha256,
            "outputReceiptHash": output_hash,
            "diversitySelectionHash": selection.sha256,
            "requestedVideoCount": video_count,
            "selectedResultCodes": selected_codes,
            "items": items,
        }
        return {**core, "sha256": _canonical_sha256(core)}

    def prepare_laffa(
        self,
        identifier: str,
        *,
        decision_ordinal: int | None = None,
        observer_seat: Seat | str | int | None = None,
        reveal_policy: str = "after_commit",
        export_preset: str | None = None,
    ) -> dict[str, Any]:
        """Prepare any complete replay-valid generated Laffa as audit media."""

        if self.learning_store is None or self.locator_service is None:
            raise RuntimeError("legacy Laffa render preparation is unavailable")
        game = self.learning_store.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        trace = self.learning_store.complete_game_trace(identifier)
        # Round-trip parsing is the fail-closed replay-validity gate.
        trace = CompleteGameTraceV1.from_dict(trace.to_dict(trusted=True))
        replay_rules = self._rules_for_trace(trace)
        decision_rows = [
            dict(item)
            for item in self.learning_store.list_decisions(identifier, limit=10_000)
        ]
        reports = self.learning_store.list_game_audits(identifier, limit=1)
        report = dict(reports[0]) if reports else {}
        audit_by_id = {
            str(item.get("decisionId") or ""): dict(item)
            for item in report.get("decisions") or ()
            if item.get("decisionId")
        }
        candidates: list[dict[str, Any]] = []
        for item in decision_rows:
            if item.get("decisionKind") != "play":
                continue
            audit = audit_by_id.get(str(item["decisionId"]))
            if audit is None:
                legal = tuple(dict(value) for value in item.get("legalActions") or ())
                audit = {
                    "decisionId": item["decisionId"],
                    "ordinal": item["ordinal"],
                    "decisionKind": "play",
                    "actor": item["actor"],
                    "forced": len(legal) <= 1,
                    "bestAction": dict(item["selectedAction"]),
                    "alternatives": [],
                    "auditConfidence": 0.0,
                    "qualityScore": 0.0,
                    "predictedDifficulty": 0.0,
                    "debateScore": 0.0,
                    "engineMethod": "replay_only",
                    "hiddenDeals": 0,
                    "effectiveSampleSize": 0.0,
                    "reasons": [],
                }
            candidates.append(audit)
        selected = self._select_audit(
            {"criticalDecisionIds": report.get("criticalDecisions") or ()},
            candidates,
            decision_ordinal,
        )
        decision_id = str(selected["decisionId"])
        decision_row = next(
            item for item in decision_rows if str(item["decisionId"]) == decision_id
        )
        decision = next(
            item
            for item in trace.decision_ledger.records
            if item.decision_id == decision_id
        )
        perspective = self._select_generated_perspective(
            build_decision_perspective_evidence(
                TraceDecisionProjectionStoreV1(trace, replay_rules),
                decision_row,
                selected,
            ),
            decision,
            observer_seat,
        )
        is_actor = perspective.truth_mode == TruthMode.ACTOR_OBSERVER
        locator = self.locator_service.publish(
            trace,
            decision_id=decision_id,
            observer_seat=perspective.observer_seat,
            prompt_contract=str(
                getattr(perspective, "prompt_contract", "")
                or ("choose_acting_action" if is_actor else "")
            ),
            reveal_policy=reveal_policy,
            rules=replay_rules,
        )
        preset = export_preset or (
            "decision_clip" if decision_ordinal is not None else "hooked_flagship"
        )
        if preset not in {"decision_clip", "hooked_flagship"}:
            raise ValueError("unsupported Laffa render preset")
        analysis_available = bool(audit_by_id.get(decision_id))
        failures = (
            ["legacy_content_qualification_not_established"]
            if analysis_available
            else [
                "analysis_unavailable",
                "legacy_content_qualification_not_established",
            ]
        )
        publication_blockers = ["legacy_content_qualification_not_established"]
        observer_name = perspective.observer_seat.name.lower()
        code = str(game.get("serial") or game.get("gameHash") or identifier)
        publication_policy = _audit_publication_policy(
            decision_id=decision_id,
            blockers=publication_blockers,
        )
        source = {
            "sourceSchemaVersion": 2,
            "kind": "laffa",
            "code": code,
            "decisionOrdinal": int(selected["ordinal"]),
            "decisionId": decision_id,
            "exactStateId": decision_id,
            "qrActionable": False,
            "traceHash": trace.sha256,
            "rulesHash": trace.rules_hash,
            "qualificationStatus": "legacy_replay",
            "renderPurpose": "research_audit",
            "qualificationFailures": failures,
            "publicationBlockers": publication_blockers,
            "observerSeat": observer_name,
            "auditOnly": True,
            "publicationEligible": False,
            "publicationPolicy": publication_policy,
            "originKind": "legacy_laffa",
        }
        analysis_hash = hashlib.sha256(
            _canonical_bytes(
                {
                    "traceHash": trace.sha256,
                    "decisionId": decision_id,
                    "audit": selected,
                }
            )
        ).hexdigest()
        reference, preview = self._build_and_store(
            trace=trace,
            decision=decision,
            evidence=perspective,
            analysis_hash=analysis_hash,
            locator=locator,
            reveal_policy=reveal_policy,
            preset=preset,
            reason_key=(
                "video.analysis.expert_reason"
                if analysis_available
                else "video.analysis.replay_only"
            ),
            rules=replay_rules,
            source_metadata=source,
        )
        basename = (
            f"balootx-{code.lower()}-decision-{int(selected['ordinal']):02d}-audit"
            if preset == "decision_clip"
            else f"balootx-{code.lower()}-laffa-audit"
        )
        return {
            "schemaVersion": 1,
            "laffaCode": code,
            "decisionOrdinal": int(selected["ordinal"]),
            "decisionId": decision_id,
            "observerSeat": observer_name,
            "exportPreset": preset,
            "qualificationStatus": "legacy_replay",
            "publicationEligible": False,
            "renderPurpose": "research_audit",
            "qualificationFailures": failures,
            "locator": locator.to_dict(),
            "bundleArtifact": reference,
            "basename": basename,
            "preview": preview,
            "source": copy.deepcopy(source),
        }

    def _prepare_trace(
        self,
        trace: CompleteGameTraceV1,
        *,
        code: str,
        decision_ordinal: int | None = None,
        observer_seat: Seat | str | int | None = None,
        reveal_policy: str = "after_commit",
        export_preset: str | None = None,
        source_kind: str = "sakka_round",
        qualification_status: str = "legacy_replay",
        qualification_failures: tuple[str, ...] = (
            "analysis_unavailable",
            "content_qualification_not_established",
        ),
        publication_blockers: tuple[str, ...] = (
            "content_qualification_not_established",
        ),
        source_identity: Mapping[str, Any] | None = None,
        expose_audit_governance: bool = False,
    ) -> dict[str, Any]:
        """Prepare a checksum-verified stored trace that has no archive row."""

        if self.locator_service is None:
            raise RuntimeError("Laffa trace render preparation is unavailable")
        replay_rules = self._rules_for_trace(trace)
        decisions: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for record in trace.decision_ledger.records:
            decision = {
                "decisionId": record.decision_id,
                "ordinal": record.ordinal,
                "decisionKind": record.decision_kind,
                "actor": record.acting_seat.name.lower(),
                "legalActions": [dict(item) for item in record.legal_actions],
                "selectedAction": dict(record.chosen_action),
            }
            decisions.append(decision)
            audits.append(
                {
                    **decision,
                    "forced": record.forced,
                    "bestAction": dict(record.chosen_action),
                    "alternatives": [],
                    "auditConfidence": 0.0,
                    "qualityScore": 0.0,
                    "predictedDifficulty": 0.0,
                    "debateScore": 0.0,
                    "engineMethod": "replay_only",
                    "hiddenDeals": 0,
                    "effectiveSampleSize": 0.0,
                    "reasons": [],
                }
            )
        selected = self._select_audit(
            {"criticalDecisionIds": ()}, audits, decision_ordinal
        )
        decision_id = str(selected["decisionId"])
        decision_row = next(
            item for item in decisions if str(item["decisionId"]) == decision_id
        )
        decision = next(
            item
            for item in trace.decision_ledger.records
            if item.decision_id == decision_id
        )
        perspective = self._select_generated_perspective(
            build_decision_perspective_evidence(
                TraceDecisionProjectionStoreV1(trace, replay_rules),
                decision_row,
                selected,
            ),
            decision,
            observer_seat,
        )
        is_actor = perspective.truth_mode == TruthMode.ACTOR_OBSERVER
        locator = self.locator_service.publish(
            trace,
            decision_id=decision_id,
            observer_seat=perspective.observer_seat,
            prompt_contract=str(
                getattr(perspective, "prompt_contract", "")
                or ("choose_acting_action" if is_actor else "")
            ),
            reveal_policy=reveal_policy,
            rules=replay_rules,
        )
        preset = export_preset or (
            "decision_clip" if decision_ordinal is not None else "hooked_flagship"
        )
        if preset not in {"decision_clip", "hooked_flagship"}:
            raise ValueError("unsupported Laffa render preset")
        failures = list(dict.fromkeys(str(item) for item in qualification_failures))
        blockers = list(dict.fromkeys(str(item) for item in publication_blockers))
        if not failures or any(not item for item in failures):
            raise ValueError("audit render qualification failures are required")
        if not blockers or any(not item for item in blockers):
            raise ValueError("audit render publication blockers are required")
        observer_name = perspective.observer_seat.name.lower()
        publication_policy = _audit_publication_policy(
            decision_id=decision_id,
            blockers=blockers,
        )
        source = {
            "sourceSchemaVersion": 2,
            "kind": "laffa",
            "code": code,
            "decisionOrdinal": int(selected["ordinal"]),
            "decisionId": decision_id,
            "exactStateId": decision_id,
            "qrActionable": False,
            "traceHash": trace.sha256,
            "rulesHash": trace.rules_hash,
            "qualificationStatus": qualification_status,
            "renderPurpose": "research_audit",
            "qualificationFailures": failures,
            "publicationBlockers": blockers,
            "observerSeat": observer_name,
            "auditOnly": True,
            "publicationEligible": False,
            "publicationPolicy": publication_policy,
            "originKind": source_kind,
        }
        extra_source = copy.deepcopy(dict(source_identity or {}))
        collisions = sorted(set(source).intersection(extra_source))
        if collisions:
            raise ValueError(
                f"audit render source identity overrides governed fields: {collisions}"
            )
        source.update(extra_source)
        analysis_hash = hashlib.sha256(
            _canonical_bytes(
                {
                    "traceHash": trace.sha256,
                    "decisionId": decision_id,
                    "audit": selected,
                    "sourceKind": source_kind,
                    "observerSeat": observer_name,
                    "sourceIdentity": extra_source,
                }
            )
        ).hexdigest()
        bundle_reference, preview = self._build_and_store(
            trace=trace,
            decision=decision,
            evidence=perspective,
            analysis_hash=analysis_hash,
            locator=locator,
            reveal_policy=reveal_policy,
            preset=preset,
            reason_key="video.analysis.replay_only",
            rules=replay_rules,
            source_metadata=source,
        )
        basename = (
            f"balootx-{code.lower()}-decision-{int(selected['ordinal']):02d}-audit"
            if preset == "decision_clip"
            else f"balootx-{code.lower()}-laffa-audit"
        )
        response = {
            "schemaVersion": 1,
            "laffaCode": code,
            "decisionOrdinal": int(selected["ordinal"]),
            "decisionId": decision_id,
            "observerSeat": observer_name,
            "exportPreset": preset,
            "qualificationStatus": qualification_status,
            "publicationEligible": False,
            "renderPurpose": "research_audit",
            "qualificationFailures": failures,
            "locator": locator.to_dict(),
            "bundleArtifact": bundle_reference,
            "basename": basename,
            "preview": preview,
            "source": copy.deepcopy(source),
        }
        if expose_audit_governance:
            response.update(
                {
                    "auditOnly": True,
                    "publicationBlockers": blockers,
                    "publicationPolicy": publication_policy,
                }
            )
        return response

    def prepare_trace_reference(
        self,
        reference: Mapping[str, Any],
        *,
        code: str,
        decision_ordinal: int | None = None,
        observer_seat: Seat | str | int | None = None,
        reveal_policy: str = "after_commit",
        export_preset: str | None = None,
        source_kind: str = "sakka_round",
    ) -> dict[str, Any]:
        """Prepare a checksum-verified stored trace that has no archive row."""

        trace = CompleteGameTraceV1.from_dict(
            self._load(
                reference,
                expected_media_type=COMPLETE_TRACE_MEDIA_TYPE,
            )
        )
        return self._prepare_trace(
            trace,
            code=code,
            decision_ordinal=decision_ordinal,
            observer_seat=observer_seat,
            reveal_policy=reveal_policy,
            export_preset=export_preset,
            source_kind=source_kind,
        )

    def prepare_audit_trace(
        self,
        record: ResearchAuditTraceRecordV1 | Mapping[str, Any],
        *,
        decision_ordinal: int | None = None,
        observer_seat: Seat | str | int | None = None,
        reveal_policy: str = "after_commit",
        export_preset: str | None = None,
    ) -> dict[str, Any]:
        """Prepare one retained replay-valid trace as non-publishable audit media."""

        raw_record = (
            record.to_dict()
            if isinstance(record, ResearchAuditTraceRecordV1)
            else dict(record)
        )
        audit_record = ResearchAuditTraceRecordV1.build(raw_record)
        trace = self._load_audit_trace(
            audit_record.complete_trace,
            expected_trace_hash=audit_record.trace_hash,
        )
        blocker = "audit_trace_publication_not_authorized"
        failures = tuple(dict.fromkeys((*audit_record.failure_categories, blocker)))
        prepared = self._prepare_trace(
            trace,
            code=audit_record.audit_trace_code,
            decision_ordinal=decision_ordinal,
            observer_seat=observer_seat,
            reveal_policy=reveal_policy,
            export_preset=export_preset,
            source_kind="research_audit_trace",
            qualification_status=audit_record.disposition,
            qualification_failures=failures,
            publication_blockers=(blocker,),
            source_identity={
                "auditTraceCode": audit_record.audit_trace_code,
                "recordHash": audit_record.record_hash,
                "attemptStatus": audit_record.attempt_status,
            },
            expose_audit_governance=True,
        )
        prepared.update(
            {
                "auditTraceCode": audit_record.audit_trace_code,
                "recordHash": audit_record.record_hash,
                "traceHash": audit_record.trace_hash,
            }
        )
        return prepared


__all__ = ["ResearchRenderPreparationServiceV1"]
