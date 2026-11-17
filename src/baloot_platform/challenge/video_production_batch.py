"""Durable orchestration for complete, governed Foundry video batches.

The service intentionally composes the existing campaign, analysis, finalist,
candidate-preparation, and render authorities.  It does not duplicate any of
their domain decisions.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from baloot_platform.challenge.analysis_catalog import NoPlayableFinalistCandidateError
from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.deep_analysis import NoRetainedCandidateTracesError


FOUNDRY_VIDEO_BATCH_JOB_KIND = "foundry_video_batch_v1"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class HookedLaffaBatchProfileV1:
    profile_id: str
    revision: str
    default_target: int
    maximum_target: int
    lane_order: tuple[str, ...]
    wave_seed_count: int
    maximum_waves: int
    seeds_per_shard: int
    simulation_worker_count: int
    simulation_timeout_seconds: int
    finalist_concurrency: int
    use_candidate_resimulation: bool
    render_concurrency: int
    render_policy: str
    allowed_truth_labels: tuple[str, ...]
    minimum_selection_score: float
    minimum_confidence: float
    require_play_decision: bool
    require_critical: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "HookedLaffaBatchProfileV1":
        allowed = {
            "schemaVersion",
            "id",
            "revision",
            "defaultTarget",
            "maximumTarget",
            "laneOrder",
            "waveSeedCount",
            "maximumWaves",
            "seedsPerShard",
            "simulationWorkerCount",
            "simulationTimeoutSeconds",
            "finalistConcurrency",
            "useCandidateResimulation",
            "renderConcurrency",
            "renderPolicy",
            "allowedTruthLabels",
            "minimumSelectionScore",
            "minimumConfidence",
            "requirePlayDecision",
            "requireCritical",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown video production batch profile keys: {sorted(unknown)}"
            )
        core = {
            "schemaVersion": int(value.get("schemaVersion", 0)),
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "defaultTarget": int(value.get("defaultTarget", 0)),
            "maximumTarget": int(value.get("maximumTarget", 0)),
            "laneOrder": [str(item) for item in value.get("laneOrder") or ()],
            "waveSeedCount": int(value.get("waveSeedCount", 0)),
            "maximumWaves": int(value.get("maximumWaves", 0)),
            "seedsPerShard": int(value.get("seedsPerShard", 0)),
            "simulationWorkerCount": int(value.get("simulationWorkerCount", 0)),
            "simulationTimeoutSeconds": int(value.get("simulationTimeoutSeconds", 0)),
            "finalistConcurrency": int(
                value.get("finalistConcurrency", value.get("renderConcurrency", 0))
            ),
            "useCandidateResimulation": bool(
                value.get("useCandidateResimulation", False)
            ),
            "renderConcurrency": int(value.get("renderConcurrency", 0)),
            "renderPolicy": str(value.get("renderPolicy") or ""),
            "allowedTruthLabels": [
                str(item) for item in value.get("allowedTruthLabels") or ()
            ],
            "minimumSelectionScore": float(value.get("minimumSelectionScore", -1.0)),
            "minimumConfidence": float(value.get("minimumConfidence", -1.0)),
            "requirePlayDecision": value.get("requirePlayDecision"),
            "requireCritical": value.get("requireCritical"),
        }
        if core["schemaVersion"] != 1 or not core["id"] or not core["revision"]:
            raise ValueError("video production batch profile identity is invalid")
        if not 1 <= core["defaultTarget"] <= core["maximumTarget"] <= 100:
            raise ValueError("video production batch target bounds are invalid")
        if (
            not core["laneOrder"]
            or len(set(core["laneOrder"])) != len(core["laneOrder"])
            or set(core["laneOrder"]) - {"organic", "guided", "disagreement"}
        ):
            raise ValueError("video production batch lanes are invalid")
        if not 1 <= core["waveSeedCount"] <= 10_000:
            raise ValueError("video production batch wave size is invalid")
        if not 1 <= core["maximumWaves"] <= 10_000:
            raise ValueError("video production batch wave limit is invalid")
        if not 1 <= core["seedsPerShard"] <= core["waveSeedCount"]:
            raise ValueError("video production batch shard size is invalid")
        if not 1 <= core["simulationWorkerCount"] <= 16:
            raise ValueError("video production batch worker count is invalid")
        if not 60 <= core["simulationTimeoutSeconds"] <= 86_400:
            raise ValueError("video production batch simulation timeout is invalid")
        if not 1 <= core["finalistConcurrency"] <= 4:
            raise ValueError("video production batch finalist concurrency is invalid")
        if not 1 <= core["renderConcurrency"] <= 2:
            raise ValueError("video production batch render concurrency is invalid")
        if core["renderPolicy"] not in {"on_demand", "automatic"}:
            raise ValueError("video production batch render policy is invalid")
        if (
            not core["allowedTruthLabels"]
            or len(set(core["allowedTruthLabels"])) != len(core["allowedTruthLabels"])
            or set(core["allowedTruthLabels"])
            - {"exact_optimal", "robust_best_known", "debatable"}
        ):
            raise ValueError("video production batch truth labels are invalid")
        if not 0.0 <= core["minimumSelectionScore"] <= 100.0:
            raise ValueError("video production batch selection threshold is invalid")
        if not 0.0 <= core["minimumConfidence"] <= 1.0:
            raise ValueError("video production batch confidence threshold is invalid")
        if not isinstance(core["requirePlayDecision"], bool) or not isinstance(
            core["requireCritical"], bool
        ):
            raise ValueError("video production batch hard gates must be boolean")
        return cls(
            core["id"],
            core["revision"],
            core["defaultTarget"],
            core["maximumTarget"],
            tuple(core["laneOrder"]),
            core["waveSeedCount"],
            core["maximumWaves"],
            core["seedsPerShard"],
            core["simulationWorkerCount"],
            core["simulationTimeoutSeconds"],
            core["finalistConcurrency"],
            core["useCandidateResimulation"],
            core["renderConcurrency"],
            core["renderPolicy"],
            tuple(core["allowedTruthLabels"]),
            core["minimumSelectionScore"],
            core["minimumConfidence"],
            core["requirePlayDecision"],
            core["requireCritical"],
            _canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "defaultTarget": self.default_target,
            "maximumTarget": self.maximum_target,
            "laneOrder": list(self.lane_order),
            "waveSeedCount": self.wave_seed_count,
            "maximumWaves": self.maximum_waves,
            "seedsPerShard": self.seeds_per_shard,
            "simulationWorkerCount": self.simulation_worker_count,
            "simulationTimeoutSeconds": self.simulation_timeout_seconds,
            "finalistConcurrency": self.finalist_concurrency,
            "useCandidateResimulation": self.use_candidate_resimulation,
            "renderConcurrency": self.render_concurrency,
            "renderPolicy": self.render_policy,
            "allowedTruthLabels": list(self.allowed_truth_labels),
            "minimumSelectionScore": self.minimum_selection_score,
            "minimumConfidence": self.minimum_confidence,
            "requirePlayDecision": self.require_play_decision,
            "requireCritical": self.require_critical,
        }
        return {**core, "sha256": self.sha256}


def evaluate_video_production_candidate_v1(
    candidate: Mapping[str, Any],
    policy: HookedLaffaBatchProfileV1 | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every governed video-production blocker in stable gate order.

    Both the durable batch lane and on-demand Research rendering use this pure
    evaluator so publication decisions cannot drift from the frozen batch
    profile.
    """

    if isinstance(policy, HookedLaffaBatchProfileV1):
        require_play = policy.require_play_decision
        require_critical = policy.require_critical
        allowed_truth_labels = policy.allowed_truth_labels
        minimum_selection_score = policy.minimum_selection_score
        minimum_confidence = policy.minimum_confidence
    else:
        require_play = bool(policy["requirePlayDecision"])
        require_critical = bool(policy["requireCritical"])
        allowed_truth_labels = tuple(
            str(item) for item in policy["allowedTruthLabels"]
        )
        minimum_selection_score = float(policy["minimumSelectionScore"])
        minimum_confidence = float(policy["minimumConfidence"])

    blockers: list[str] = []
    if require_play and candidate.get("decisionKind") != "play":
        blockers.append("not_play_decision")
    if require_critical and not bool(candidate.get("critical")):
        blockers.append("not_critical")
    if str(candidate.get("truthLabel")) not in allowed_truth_labels:
        blockers.append("disallowed_truth_label")
    if float(candidate.get("selectionScore", 0.0)) < minimum_selection_score:
        blockers.append("below_selection_score")
    if float(candidate.get("confidence", 0.0)) < minimum_confidence:
        blockers.append("below_confidence")
    if candidate.get("compatibilityBlockers"):
        blockers.append("compatibility_blocker")
    return tuple(blockers)


class FoundryVideoBatchServiceV1:
    """Search and qualify a bounded Laffa target; render only by governed policy."""

    def __init__(
        self,
        *,
        jobs: Any,
        background_executor: Any,
        campaign_store: Any,
        deep_analysis: Any,
        candidates: Any,
        finalist_validations: Any,
        render_batches: Any,
        runtime_profile: Mapping[str, Any],
        schedule_campaign: Callable[[str], None],
        schedule_analysis: Callable[[], None],
        qualified_library: Any | None = None,
        poll_seconds: float = 1.0,
        maximum_attempts: int = 3,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("video batch poll interval must be positive")
        if maximum_attempts < 1:
            raise ValueError("video batch maximum attempts must be positive")
        self.jobs = jobs
        self.background_executor = background_executor
        self.campaign_store = campaign_store
        self.deep_analysis = deep_analysis
        self.candidates = candidates
        self.finalist_validations = finalist_validations
        self.render_batches = render_batches
        self.runtime_profile = dict(runtime_profile)
        self.schedule_campaign = schedule_campaign
        self.schedule_analysis = schedule_analysis
        self.qualified_library = qualified_library
        self.poll_seconds = float(poll_seconds)
        self.maximum_attempts = int(maximum_attempts)
        self._active_lock = threading.Lock()
        self._active_jobs: set[str] = set()
        self._qualified_backfill_lock = threading.Lock()
        self._qualified_backfill_done = False

    @staticmethod
    def runtime_snapshot(profile: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "rulesHash": str(profile["rulesHash"]),
            "bundleHash": str(profile["bundleHash"]),
            "engineVersion": str(profile["engineVersion"]),
            "analysisProfileHash": str(profile["analysisProfileHash"]),
            "searchBackendId": str(profile["searchBackend"]["id"]),
            "matchValueProfileHash": str(profile["matchValueProfile"]["sha256"]),
            "perspectiveProfileHash": str(profile["perspectiveProfileHash"]),
            "selectionProfileHash": str(profile["selectionProfileHash"]),
            "agentRelease": str(profile["agentRelease"]),
            "seatAgents": dict(profile["seatAgents"]),
            "staticLaneProfileHashes": dict(profile["staticLaneProfileHashes"]),
            "laneAgentReleases": dict(profile["laneAgentReleases"]),
            "laneSeatAgents": {
                str(lane): dict(values)
                for lane, values in dict(profile["laneSeatAgents"]).items()
            },
            "candidateResimulationProfileHash": str(
                profile["candidateResimulationProfile"]["sha256"]
            ),
            "candidateResimulationAgentRelease": str(
                profile["candidateResimulationAgentRelease"]
            ),
            "decisionAnalysisExecutorProfileHashes": {
                str(stage): str(dict(value)["sha256"])
                for stage, value in dict(
                    profile["decisionAnalysisExecutorProfile"]
                ).items()
                if stage in {"screening", "confirmation", "finalistReanalysis"}
            },
        }

    def submit(
        self,
        request: Mapping[str, Any],
        *,
        idempotency_key: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("video batch requires a bounded idempotency key")
        normalized = self._validate_request(request)
        job_id = uuid.uuid4().hex
        actual_id = self.jobs.create(
            job_id,
            normalized,
            kind=FOUNDRY_VIDEO_BATCH_JOB_KIND,
            tenant_id=tenant_id,
            max_attempts=self.maximum_attempts,
            idempotency_key=idempotency_key,
        )
        persisted = self.get(actual_id)
        if persisted is None:
            raise RuntimeError("video batch job was not persisted")
        if dict(persisted.get("request") or {}) != normalized:
            raise ValueError("Idempotency-Key was already used for another video batch")
        if actual_id == job_id:
            self.background_executor.submit(self._execute, actual_id)
        return persisted

    def resume(self, job_id: str) -> dict[str, Any]:
        item = self.get(job_id)
        if item is None:
            raise KeyError(job_id)
        if item["status"] in {"completed", "failed", "cancelled", "exhausted"}:
            return item
        if item.get("cancelRequested"):
            self._terminalize_cancelled(
                job_id, item, "cancelled before video batch resumed"
            )
            return self.get(job_id) or item
        self.background_executor.submit(self._execute, job_id)
        return self.get(job_id) or item

    def recover_pending(self) -> tuple[str, ...]:
        """Resume local jobs after restart; callers must gate this to single-node mode."""
        recovered: set[str] = set()
        for status in ("queued", "running"):
            for item in self.list(status=status, limit=1_000):
                job_id = str(item["jobId"])
                if item.get("cancelRequested"):
                    self._terminalize_cancelled(
                        job_id, item, "cancelled during video batch recovery"
                    )
                else:
                    self.background_executor.submit(self._execute, job_id)
                recovered.add(job_id)
        return tuple(sorted(recovered))

    def get(self, job_id: str) -> dict[str, Any] | None:
        item = self.jobs.get(str(job_id))
        if item is None or item.get("kind") != FOUNDRY_VIDEO_BATCH_JOB_KIND:
            return None
        return item

    def list(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.jobs.list_jobs(
            kinds=(FOUNDRY_VIDEO_BATCH_JOB_KIND,), status=status, limit=limit
        )

    def list_qualified_laffat(
        self,
        *,
        search: str = "",
        truth_label: str | None = None,
        classification: str | None = None,
        profile: str | None = None,
        sort: str = "laffa_score",
        direction: str = "desc",
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Backfill and return the durable render-ready Laffa library."""
        if self.qualified_library is None:
            values: dict[str, dict[str, Any]] = {}
            for job in self.list(limit=500):
                for item in dict(job.get("result") or {}).get("preparedItems") or ():
                    prepared = self._prepared_laffa_projection(job, item)
                    if not prepared.get("laffaSelectionScore"):
                        continue
                    values[str(prepared.get("bundleHash") or "")] = prepared
            return {"items": list(values.values())[:limit], "nextCursor": None}
        if not self._qualified_backfill_done:
            with self._qualified_backfill_lock:
                if not self._qualified_backfill_done:
                    for job in self.list(limit=500):
                        for item in dict(job.get("result") or {}).get("preparedItems") or ():
                            try:
                                self.qualified_library.put_qualified_laffa(
                                    self._prepared_laffa_projection(job, item)
                                )
                            except ValueError:
                                # Preserve legacy job evidence without promoting an item
                                # that predates the current qualification contract.
                                continue
                    self._qualified_backfill_done = True
        return self.qualified_library.list_qualified_laffat(
            search=search,
            truth_label=truth_label,
            classification=classification,
            profile=profile,
            sort=sort,
            direction=direction,
            cursor=cursor,
            limit=limit,
        )

    def _prepared_laffa_projection(
        self,
        job: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Derive the current library projection from immutable finalist evidence."""
        prepared = dict(item)
        declared_score = dict(prepared.get("laffaSelectionScore") or {})
        if (
            declared_score.get("scoreFamily") == "laffa_content_selection"
            and declared_score.get("passed") is True
        ):
            return prepared
        candidate_hash = str(prepared.get("candidateHash") or "")
        validation_items = list(
            dict(job.get("result") or {}).get("validationItems") or ()
        )
        matching = next(
            (
                dict(value)
                for value in validation_items
                if str(dict(value).get("candidateHash") or "") == candidate_hash
                and str(dict(value).get("status") or "") == "accepted"
            ),
            None,
        )
        if matching is None:
            return prepared
        finalist_job_id = str(matching.get("finalistJobId") or "")
        finalist = (
            self.finalist_validations.get(finalist_job_id)
            if finalist_job_id
            else None
        )
        finalist_items = list(
            dict((finalist or {}).get("result") or {}).get("items") or ()
        )
        if not finalist_items:
            return prepared
        score = dict(
            dict(dict(finalist_items[0]).get("analysisEvidence") or {}).get(
                "laffaSelectionScore"
            )
            or {}
        )
        if (
            score.get("scoreFamily") != "laffa_content_selection"
            or score.get("passed") is not True
        ):
            return prepared
        prepared["laffaSelectionScore"] = score
        prepared["finalistJobId"] = finalist_job_id
        return prepared

    def _qualified_trace_hashes(self) -> set[str]:
        if self.qualified_library is None:
            return set()
        values: set[str] = set()
        cursor: str | None = None
        while True:
            page = self.qualified_library.list_qualified_laffat(
                cursor=cursor,
                sort="laffa_score",
                direction="desc",
                limit=500,
            )
            values.update(
                str(item.get("traceHash") or "")
                for item in page.get("items") or ()
                if str(item.get("traceHash") or "")
            )
            cursor = str(page.get("nextCursor") or "") or None
            if cursor is None:
                return values

    def cancel(self, job_id: str) -> bool:
        item = self.get(job_id)
        if item is None:
            return False
        requested = bool(self.jobs.request_cancel(job_id))
        if requested:
            with self._active_lock:
                active = job_id in self._active_jobs
            if active:
                self._cancel_children(dict(item.get("result") or {}))
            else:
                self._terminalize_cancelled(
                    job_id, item, "cancelled before video batch resumed"
                )
        return requested

    def _terminalize_cancelled(
        self, job_id: str, item: Mapping[str, Any], error: str
    ) -> None:
        checkpoint = dict(item.get("result") or {})
        self._cancel_children(checkpoint)
        checkpoint["stage"] = "cancelled"
        checkpoint["outcome"] = "cancelled"
        self.jobs.update(
            job_id,
            status="cancelled",
            progress=1.0,
            result=checkpoint,
            error=error,
        )

    def _cancel_children(self, checkpoint: Mapping[str, Any]) -> None:
        for campaign_hash in checkpoint.get("campaigns") or ():
            try:
                self.campaign_store.request_cancel(str(campaign_hash))
            except (AttributeError, KeyError, RuntimeError, ValueError):
                pass
        for service, key in (
            (self.deep_analysis, "analysisJobs"),
            (self.finalist_validations, "finalistJobs"),
            (self.render_batches, "renderJobs"),
        ):
            cancel = getattr(service, "cancel", None)
            if not callable(cancel):
                continue
            for child_job_id in checkpoint.get(key) or ():
                try:
                    cancel(str(child_job_id))
                except (KeyError, RuntimeError, ValueError):
                    pass

    def events(self, job_id: str) -> list[dict[str, Any]] | None:
        if self.get(job_id) is None:
            return None
        return list(self.jobs.events(job_id))

    def _validate_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "schemaVersion",
            "targetVideos",
            "laneOrder",
            "seedStart",
            "roundsPerWave",
            "maximumRounds",
            "seedsPerShard",
            "finalistConcurrency",
            "useCandidateResimulation",
            "renderConcurrency",
            "exportPreset",
            "renderPolicy",
            "revealPolicy",
            "runtimeSnapshot",
            "allowedTruthLabels",
            "minimumSelectionScore",
            "minimumConfidence",
            "requirePlayDecision",
            "requireCritical",
        }
        unknown = set(request) - allowed
        if unknown:
            raise ValueError(f"unknown video batch keys: {sorted(unknown)}")
        target = int(request.get("targetVideos", 0))
        wave = int(request.get("roundsPerWave", 0))
        maximum = int(request.get("maximumRounds", 0))
        shards = int(request.get("seedsPerShard", 0))
        concurrency = int(request.get("renderConcurrency", 0))
        render_policy = str(request.get("renderPolicy") or "")
        finalist_concurrency = int(request.get("finalistConcurrency", concurrency))
        use_candidate_resimulation = request.get("useCandidateResimulation", False)
        lane_order = tuple(str(item) for item in request.get("laneOrder") or ())
        truth_labels = tuple(
            str(item) for item in request.get("allowedTruthLabels") or ()
        )
        if int(request.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported video batch schema")
        if not 1 <= target <= 100:
            raise ValueError("video batch target must be between one and one hundred")
        if (
            not lane_order
            or len(set(lane_order)) != len(lane_order)
            or set(lane_order) - {"organic", "guided", "disagreement"}
        ):
            raise ValueError("video batch lanes are not locally executable")
        if not 1 <= wave <= 10_000 or not wave <= maximum <= 1_000_000:
            raise ValueError("video batch round bounds are invalid")
        if not 1 <= shards <= wave:
            raise ValueError("video batch seeds per shard are invalid")
        if not 1 <= concurrency <= 2:
            raise ValueError("video batch render concurrency must be one or two")
        if render_policy not in {"on_demand", "automatic"}:
            raise ValueError("video batch render policy must be on_demand or automatic")
        if not 1 <= finalist_concurrency <= 4:
            raise ValueError(
                "video batch finalist concurrency must be between one and four"
            )
        if not isinstance(use_candidate_resimulation, bool):
            raise ValueError(
                "video batch candidate-resimulation policy must be boolean"
            )
        if str(request.get("exportPreset")) != "hooked_flagship":
            raise ValueError("complete Laffa batches require hooked_flagship")
        if str(request.get("revealPolicy")) != "after_commit":
            raise ValueError("complete Laffa batches require after_commit reveal")
        if not truth_labels or set(truth_labels) - {
            "exact_optimal",
            "robust_best_known",
            "debatable",
        }:
            raise ValueError("video batch truth labels are invalid")
        minimum_score = float(request.get("minimumSelectionScore", -1.0))
        minimum_confidence = float(request.get("minimumConfidence", -1.0))
        if not 0.0 <= minimum_score <= 100.0 or not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("video batch quality thresholds are invalid")
        require_play = request.get("requirePlayDecision")
        require_critical = request.get("requireCritical")
        if not isinstance(require_play, bool) or not isinstance(require_critical, bool):
            raise ValueError("video batch hard gates are invalid")
        snapshot = dict(request.get("runtimeSnapshot") or {})
        current = self.runtime_snapshot(self.runtime_profile)
        if snapshot != current:
            raise ValueError(
                "video batch runtime snapshot does not match the active bundle"
            )
        return {
            "schemaVersion": 1,
            "targetVideos": target,
            "laneOrder": list(lane_order),
            "seedStart": int(request.get("seedStart", 0)),
            "roundsPerWave": wave,
            "maximumRounds": maximum,
            "seedsPerShard": shards,
            "finalistConcurrency": finalist_concurrency,
            "useCandidateResimulation": use_candidate_resimulation,
            "renderConcurrency": concurrency,
            "renderPolicy": render_policy,
            "exportPreset": "hooked_flagship",
            "revealPolicy": "after_commit",
            "allowedTruthLabels": list(truth_labels),
            "minimumSelectionScore": minimum_score,
            "minimumConfidence": minimum_confidence,
            "requirePlayDecision": require_play,
            "requireCritical": require_critical,
            "runtimeSnapshot": current,
        }

    @staticmethod
    def _initial_checkpoint(request: Mapping[str, Any]) -> dict[str, Any]:
        lane_metrics = {
            str(lane): {
                "roundsGenerated": 0,
                "campaignsCompleted": 0,
                "rawCandidates": 0,
                "candidatesAfterBatchGates": 0,
                "finalistsSubmitted": 0,
                "finalistsAccepted": 0,
                "finalistsRejected": 0,
            }
            for lane in request["laneOrder"]
        }
        return {
            "schemaVersion": 1,
            "stage": "searching",
            "outcome": "running",
            "targetVideos": int(request["targetVideos"]),
            "renderPolicy": str(request["renderPolicy"]),
            "useCandidateResimulation": bool(request["useCandidateResimulation"]),
            "roundsGenerated": 0,
            "campaignsCompleted": 0,
            "retainedCandidates": 0,
            "finalistsAccepted": 0,
            "finalistsRejected": 0,
            "preparedVideoCount": 0,
            "renderReadyCount": 0,
            "completedVideoCount": 0,
            "failedVideoCount": 0,
            "renderProgress": 0.0,
            "stageProgress": 0.0,
            "activeWorkerCount": 0,
            "currentWave": 0,
            "campaigns": [],
            "analysisJobs": [],
            "finalistJobs": [],
            "validatedTraces": [],
            "rejectedTraces": [],
            "validationItems": [],
            "preparedItems": [],
            "renderJobs": [],
            "renderItems": [],
            "rejections": [],
            "operationalMetrics": {
                "schemaVersion": 1,
                "stageWallSeconds": {
                    "simulation": 0.0,
                    "analysis": 0.0,
                    "finalist": 0.0,
                },
                "laneMetrics": lane_metrics,
                "gateDispositions": {},
                "acceptedPer100Rounds": 0.0,
                "retainedPer100Rounds": 0.0,
                "acceptedPerWallHour": 0.0,
            },
        }

    @staticmethod
    def _operational_metrics(
        checkpoint: dict[str, Any],
        request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = checkpoint.setdefault("operationalMetrics", {})
        metrics.setdefault("schemaVersion", 1)
        stages = metrics.setdefault("stageWallSeconds", {})
        for stage in ("simulation", "analysis", "finalist"):
            stages.setdefault(stage, 0.0)
        lanes = metrics.setdefault("laneMetrics", {})
        lane_order = tuple((request or {}).get("laneOrder") or lanes)
        for lane in lane_order:
            value = lanes.setdefault(str(lane), {})
            for name in (
                "roundsGenerated",
                "campaignsCompleted",
                "rawCandidates",
                "candidatesAfterBatchGates",
                "finalistsSubmitted",
                "finalistsAccepted",
                "finalistsRejected",
            ):
                value.setdefault(name, 0)
        metrics.setdefault("gateDispositions", {})
        metrics.setdefault("acceptedPer100Rounds", 0.0)
        metrics.setdefault("retainedPer100Rounds", 0.0)
        metrics.setdefault("acceptedPerWallHour", 0.0)
        return metrics

    @classmethod
    def _add_stage_wall_seconds(
        cls,
        checkpoint: dict[str, Any],
        stage: str,
        elapsed: float,
    ) -> None:
        stages = cls._operational_metrics(checkpoint)["stageWallSeconds"]
        stages[stage] = round(float(stages.get(stage, 0.0)) + max(0.0, elapsed), 6)

    @classmethod
    def _lane_metrics(
        cls,
        checkpoint: dict[str, Any],
        lane: str,
    ) -> dict[str, int]:
        lanes = cls._operational_metrics(checkpoint)["laneMetrics"]
        value = lanes.setdefault(lane, {})
        for name in (
            "roundsGenerated",
            "campaignsCompleted",
            "rawCandidates",
            "candidatesAfterBatchGates",
            "finalistsSubmitted",
            "finalistsAccepted",
            "finalistsRejected",
        ):
            value.setdefault(name, 0)
        return value

    @classmethod
    def _record_gate_disposition(
        cls,
        checkpoint: dict[str, Any],
        code: str,
        count: int = 1,
    ) -> None:
        dispositions = cls._operational_metrics(checkpoint)["gateDispositions"]
        dispositions[code] = int(dispositions.get(code, 0)) + int(count)

    @staticmethod
    def _candidate_gate_rejection_code(
        candidate: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> str | None:
        blockers = evaluate_video_production_candidate_v1(candidate, request)
        return blockers[0] if blockers else None

    @staticmethod
    def _finalist_rejection_code(
        finalist: Mapping[str, Any],
        reason: str,
    ) -> str:
        if "candidate resimulation" in reason.lower():
            return "candidate_resimulation_failed"
        items = list(dict(finalist.get("result") or {}).get("items") or ())
        if items:
            blockers = {str(item) for item in dict(items[0]).get("blockers") or ()}
            if "no_verified_tactical_play_opportunity" in blockers:
                return "no_verified_tactical_play_opportunity"
            analysis = dict(dict(items[0]).get("analysisEvidence") or {})
            if int(analysis.get("unresolvedDecisionCount") or 0) > 0:
                return "unresolved_surrounding_decisions"
        return "finalist_quality_failed"

    def _checkpoint(self, job_id: str, value: Mapping[str, Any]) -> None:
        mutable = dict(value)
        metrics = self._operational_metrics(mutable)
        target = max(1, int(value["targetVideos"]))
        prepared = int(value.get("preparedVideoCount", 0))
        completed = int(value.get("completedVideoCount", 0))
        stage = str(value.get("stage") or "searching")
        if stage in {"rendering", "draining"}:
            progress = 0.7 + 0.3 * float(
                value.get("renderProgress", completed / target)
            )
        else:
            progress = min(0.69, 0.69 * prepared / target)
        rounds = int(value.get("roundsGenerated", 0))
        retained = int(value.get("retainedCandidates", 0))
        accepted = int(value.get("finalistsAccepted", 0))
        total_stage_seconds = sum(
            float(item) for item in dict(metrics["stageWallSeconds"]).values()
        )
        metrics["acceptedPer100Rounds"] = (
            round(100.0 * accepted / rounds, 6) if rounds else 0.0
        )
        metrics["retainedPer100Rounds"] = (
            round(100.0 * retained / rounds, 6) if rounds else 0.0
        )
        metrics["acceptedPerWallHour"] = (
            round(3600.0 * accepted / total_stage_seconds, 6)
            if total_stage_seconds
            else 0.0
        )
        if isinstance(value, dict):
            value["operationalMetrics"] = metrics
            result = value
        else:
            mutable["operationalMetrics"] = metrics
            result = mutable
        self.jobs.update(job_id, status="running", progress=progress, result=result)

    def _wait(
        self,
        job_id: str,
        getter: Callable[[], Mapping[str, Any] | None],
        on_poll: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, Any]:
        while True:
            if self.jobs.cancelled(job_id):
                raise InterruptedError("video batch cancelled")
            item = getter()
            if item is None:
                raise RuntimeError("video batch child job disappeared")
            if on_poll is not None:
                on_poll(item)
            if item.get("status") in {"completed", "failed", "cancelled", "exhausted"}:
                return item
            time.sleep(self.poll_seconds)

    @staticmethod
    def _candidate_sort_key(
        item: Mapping[str, Any],
    ) -> tuple[int, int, float, float, str]:
        truth = {"exact_optimal": 3, "robust_best_known": 2, "debatable": 1}
        return (
            -(1 if item.get("decisionKind") == "play" else 0),
            -truth.get(str(item.get("truthLabel")), 0),
            -float(item.get("selectionScore", 0.0)),
            -float(item.get("confidence", 0.0)),
            str(item.get("candidateHash") or ""),
        )

    @classmethod
    def _distinct_trace_representatives(
        cls,
        values: Sequence[Mapping[str, Any]],
        *,
        excluded_traces: set[str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Choose one stable candidate for each complete Laffa trace."""

        representatives: list[dict[str, Any]] = []
        seen = set(excluded_traces)
        duplicates = 0
        for value in sorted(
            (dict(item) for item in values), key=cls._candidate_sort_key
        ):
            trace_hash = str(dict(value.get("technical") or {}).get("traceHash") or "")
            if not trace_hash:
                duplicates += 1
                continue
            if trace_hash in seen:
                duplicates += 1
                continue
            seen.add(trace_hash)
            representatives.append(value)
        return representatives, duplicates

    def _refresh_render_state(
        self,
        checkpoint: dict[str, Any],
        *,
        denominator: int,
    ) -> tuple[int, int, int]:
        completed = 0
        failed = 0
        active = 0
        active_progress = 0.0
        for record in checkpoint.setdefault("renderItems", []):
            child = self.render_batches.get(str(record["jobId"]))
            if child is None:
                raise RuntimeError("video batch render child disappeared")
            status = str(child.get("status") or "")
            record["status"] = status
            if status in {"queued", "running"}:
                active += 1
                active_progress += float(child.get("progress", 0.0))
                continue
            execution = dict(dict(child.get("result") or {}).get("execution") or {})
            result_items = list(execution.get("items") or ())
            if (
                status == "completed"
                and len(result_items) == 1
                and result_items[0].get("status") == "completed"
            ):
                completed += 1
            else:
                failed += 1
        checkpoint["completedVideoCount"] = completed
        checkpoint["failedVideoCount"] = failed
        checkpoint["renderProgress"] = min(
            1.0,
            (completed + failed + active_progress) / max(1, denominator),
        )
        if str(checkpoint.get("stage") or "") in {"rendering", "draining"}:
            checkpoint["stageProgress"] = checkpoint["renderProgress"]
            checkpoint["activeWorkerCount"] = active
        return completed, failed, active

    def _submit_render_slots(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
        request: Mapping[str, Any],
    ) -> tuple[int, int, int]:
        target = int(request["targetVideos"])
        completed, failed, active = self._refresh_render_state(
            checkpoint,
            denominator=target,
        )
        if failed:
            return completed, failed, active
        submitted = {
            str(value.get("bundleHash") or "")
            for value in checkpoint.setdefault("renderItems", [])
        }
        pending = [
            value
            for value in checkpoint.get("preparedItems", [])[:target]
            if str(value.get("bundleHash") or "") not in submitted
        ]
        available = max(0, int(request["renderConcurrency"]) - active)
        for value in pending[:available]:
            bundle_hash = str(value["bundleHash"])
            render = self.render_batches.submit(
                [
                    {
                        "basename": value["basename"],
                        "bundleArtifact": value["bundleArtifact"],
                    }
                ],
                idempotency_key=f"video-batch-{job_id}-render-{bundle_hash}",
            )
            render_id = str(render["jobId"])
            if render_id not in checkpoint["renderJobs"]:
                checkpoint["renderJobs"].append(render_id)
            checkpoint["renderItems"].append(
                {
                    "jobId": render_id,
                    "traceHash": str(value["traceHash"]),
                    "bundleHash": bundle_hash,
                    "status": str(render.get("status") or "queued"),
                }
            )
            self._checkpoint(job_id, checkpoint)
            resume_render = getattr(self.render_batches, "resume", None)
            if callable(resume_render):
                resume_render(render_id)
            active += 1
        return self._refresh_render_state(checkpoint, denominator=target)

    def _qualify_wave(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
        request: Mapping[str, Any],
        *,
        campaign_hash: str,
        analysis_id: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        target = int(request["targetVideos"])
        prepared = {
            str(value.get("traceHash") or "")
            for value in checkpoint.get("preparedItems", [])
        }
        prepared.update(
            str(value)
            for value in checkpoint.get("libraryExcludedTraces", [])
            if str(value)
        )
        rejected = set(
            str(value) for value in checkpoint.setdefault("rejectedTraces", [])
        )
        pending = {
            str(value.get("traceHash") or "")
            for value in checkpoint.setdefault("validationItems", [])
            if value.get("status") not in {"accepted", "rejected", "superseded"}
        }
        representatives, duplicate_count = self._distinct_trace_representatives(
            candidates,
            excluded_traces=prepared | rejected | pending,
        )
        if duplicate_count:
            self._record_gate_disposition(
                checkpoint, "duplicate_laffa", duplicate_count
            )
            checkpoint["rejections"].append(
                {
                    "stage": "trace_deduplication",
                    "reasonCode": "duplicate_laffa",
                    "reason": f"{duplicate_count} candidates shared an already represented complete Laffa",
                    "count": duplicate_count,
                    "analysisJobId": analysis_id,
                }
            )
        concurrency = int(request["finalistConcurrency"])
        for offset in range(0, len(representatives), concurrency):
            if int(checkpoint["preparedVideoCount"]) >= target:
                break
            submitted: list[dict[str, Any]] = []
            for candidate in representatives[offset : offset + concurrency]:
                trace_hash = str(dict(candidate["technical"])["traceHash"])
                candidate_hash = str(candidate["candidateHash"])
                record = {
                    "ordinal": len(checkpoint["validationItems"]),
                    "traceHash": trace_hash,
                    "candidateHash": candidate_hash,
                    "analysisJobId": analysis_id,
                    "campaignHash": campaign_hash,
                    "candidate": dict(candidate),
                    "status": "planned",
                }
                checkpoint["validationItems"].append(record)
                submitted.append(record)
                self._checkpoint(job_id, checkpoint)
            self._run_validation_records(
                job_id,
                checkpoint,
                request,
                submitted,
            )

    def _run_validation_records(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
        request: Mapping[str, Any],
        records: Sequence[dict[str, Any]],
    ) -> None:
        target = int(request["targetVideos"])
        ordered = sorted(records, key=lambda value: int(value["ordinal"]))
        for record in ordered:
            if record.get("status") == "planned":
                finalist = self.finalist_validations.submit(
                    analysis_job_id=str(record["analysisJobId"]),
                    candidate_hashes=(str(record["candidateHash"]),),
                    idempotency_key=f"video-batch-{job_id}-trace-{record['traceHash']}",
                    candidate_resimulation=bool(request["useCandidateResimulation"]),
                    required_decision_kind=(
                        "play" if request["requirePlayDecision"] else None
                    ),
                )
                finalist_id = str(finalist["jobId"])
                record["finalistJobId"] = finalist_id
                record["status"] = "submitted"
                if record["traceHash"] not in checkpoint["validatedTraces"]:
                    checkpoint["validatedTraces"].append(record["traceHash"])
                if finalist_id not in checkpoint["finalistJobs"]:
                    checkpoint["finalistJobs"].append(finalist_id)
                candidate = dict(record.get("candidate") or {})
                lane = str(
                    dict(candidate.get("dimensions") or {}).get("source") or "unknown"
                )
                lane_metrics = self._lane_metrics(checkpoint, lane)
                lane_metrics["finalistsSubmitted"] += 1
                self._checkpoint(job_id, checkpoint)
                resume_finalist = getattr(self.finalist_validations, "resume", None)
                if callable(resume_finalist):
                    resume_finalist(finalist_id)
        checkpoint["activeWorkerCount"] = min(
            int(request["finalistConcurrency"]),
            sum(
                1
                for record in ordered
                if record.get("status") not in {"accepted", "rejected", "superseded"}
            ),
        )
        checkpoint["stageProgress"] = 0.0
        self._checkpoint(job_id, checkpoint)

        # Child jobs may finish in any order. Reduction always follows the
        # stable ordinal captured before any child was submitted.
        for reduction_index, record in enumerate(ordered):
            if record.get("status") in {"accepted", "rejected", "superseded"}:
                continue
            finalist_id = str(record["finalistJobId"])
            last_reported_percent = -1

            def report_finalist_progress(
                child: Mapping[str, Any],
                *,
                base: int = reduction_index,
            ) -> None:
                nonlocal last_reported_percent
                child_progress = max(
                    0.0,
                    min(1.0, float(child.get("progress", 0.0))),
                )
                percent = int(child_progress * 100.0)
                if percent == last_reported_percent:
                    return
                last_reported_percent = percent
                checkpoint["stageProgress"] = min(
                    1.0,
                    (base + child_progress) / max(1, len(ordered)),
                )
                checkpoint["activeWorkerCount"] = min(
                    int(request["finalistConcurrency"]),
                    sum(
                        1
                        for item in ordered
                        if item.get("status")
                        not in {
                            "accepted",
                            "rejected",
                            "superseded",
                        }
                    ),
                )
                self._checkpoint(job_id, checkpoint)

            finalist_started = time.perf_counter()
            finalist = self._wait(
                job_id,
                lambda value=finalist_id: self.finalist_validations.get(value),
                on_poll=report_finalist_progress,
            )
            self._add_stage_wall_seconds(
                checkpoint,
                "finalist",
                time.perf_counter() - finalist_started,
            )
            candidate_hash = str(record["candidateHash"])
            candidate = dict(record["candidate"])
            lane = str(
                dict(candidate.get("dimensions") or {}).get("source") or "unknown"
            )
            lane_metrics = self._lane_metrics(checkpoint, lane)
            if finalist["status"] != "completed" or not bool(
                dict(finalist.get("result") or {}).get("localPreviewReady")
            ):
                record["status"] = "rejected"
                checkpoint["finalistsRejected"] += 1
                if record["traceHash"] not in checkpoint["rejectedTraces"]:
                    checkpoint["rejectedTraces"].append(record["traceHash"])
                reason = str(
                    finalist.get("error") or "quality gates rejected the Laffa"
                )
                reason_code = self._finalist_rejection_code(finalist, reason)
                record["rejectionCode"] = reason_code
                lane_metrics["finalistsRejected"] += 1
                self._record_gate_disposition(checkpoint, reason_code)
                checkpoint["rejections"].append(
                    {
                        "stage": (
                            "candidate_resimulation"
                            if "candidate resimulation" in reason.lower()
                            else "finalist"
                        ),
                        "reasonCode": reason_code,
                        "reason": reason,
                        "candidateHash": candidate_hash,
                    }
                )
                self._checkpoint(job_id, checkpoint)
                continue
            finalist_items = list(dict(finalist.get("result") or {}).get("items") or ())
            finalist_item: dict[str, Any] = {}
            if finalist_items:
                finalist_item = dict(finalist_items[0])
                record["candidateResimulation"] = dict(
                    finalist_item.get("candidateResimulation") or {}
                )
                record["candidateResimulationTraceHash"] = finalist_item.get(
                    "candidateResimulationTraceHash"
                )
                record["finalistDescendantTraceHash"] = finalist_item.get(
                    "descendantTraceHash"
                )
            if int(checkpoint["preparedVideoCount"]) >= target:
                record["status"] = "superseded"
                self._checkpoint(job_id, checkpoint)
                continue
            try:
                prepared_item = self.candidates.prepare_best_finalist_candidate(
                    finalist,
                    reveal_policy=str(request["revealPolicy"]),
                    export_preset=str(request["exportPreset"]),
                )
            except NoPlayableFinalistCandidateError as exc:
                record["status"] = "rejected"
                record["rejectionCode"] = "no_playable_finalist"
                checkpoint["finalistsRejected"] += 1
                lane_metrics["finalistsRejected"] += 1
                self._record_gate_disposition(checkpoint, "no_playable_finalist")
                if record["traceHash"] not in checkpoint["rejectedTraces"]:
                    checkpoint["rejectedTraces"].append(record["traceHash"])
                checkpoint["rejections"].append(
                    {
                        "stage": "finalist",
                        "reasonCode": "no_playable_finalist",
                        "reason": str(exc),
                        "candidateHash": candidate_hash,
                    }
                )
                self._checkpoint(job_id, checkpoint)
                continue
            artifact = dict(prepared_item["bundleArtifact"])
            accepted_item = {
                "traceHash": str(record["traceHash"]),
                "candidateHash": candidate_hash,
                "shortCode": str(
                    dict(prepared_item.get("locator") or {}).get("code")
                    or candidate.get("shortCode")
                    or ""
                ),
                "basename": str(prepared_item["basename"]),
                "bundleArtifact": artifact,
                "bundleHash": str(artifact.get("ownerHash") or ""),
                "selectionScore": float(candidate.get("selectionScore") or 0.0),
                "laffaSelectionScore": dict(
                    dict(finalist_item.get("analysisEvidence") or {}).get(
                        "laffaSelectionScore"
                    )
                    or {}
                ),
                "confidence": float(candidate.get("confidence") or 0.0),
                "truthLabel": str(candidate.get("truthLabel") or ""),
                "decisionKind": str(candidate.get("decisionKind") or ""),
                "laffaClassification": str(candidate.get("laffaClassification") or ""),
                "observerSeat": str(candidate.get("observerSeat") or ""),
                "dimensions": dict(candidate.get("dimensions") or {}),
                "bestAction": dict(candidate.get("bestAction") or {}),
                "sourceCampaignHash": str(record.get("campaignHash") or ""),
                "analysisJobId": str(record.get("analysisJobId") or ""),
                "interestingProfiles": list(
                    dict.fromkeys(
                        value
                        for value in (
                            str(
                                dict(candidate.get("dimensions") or {}).get(
                                    "tactical_detector"
                                )
                                or ""
                            ).split("+")
                            + [str(candidate.get("laffaClassification") or "")]
                        )
                        if value
                    )
                ),
                "scoreProfiles": {
                    "decisionSelectionProfileHash": str(
                        dict(request["runtimeSnapshot"])["selectionProfileHash"]
                    ),
                    "laffaSelectionProfileHash": str(
                        dict(
                            dict(finalist_item.get("analysisEvidence") or {}).get(
                                "laffaSelectionScore"
                            )
                            or {}
                        ).get("profileHash")
                        or ""
                    ),
                },
                "qualification": dict(prepared_item.get("qualification") or {}),
                "candidateResimulation": dict(
                    record.get("candidateResimulation") or {}
                ),
                "candidateResimulationTraceHash": record.get(
                    "candidateResimulationTraceHash"
                ),
                "finalistDescendantTraceHash": record.get(
                    "finalistDescendantTraceHash"
                ),
                "finalistJobId": finalist_id,
            }
            if self.qualified_library is not None:
                self.qualified_library.put_qualified_laffa(accepted_item)
            checkpoint["preparedItems"].append(accepted_item)
            record["status"] = "accepted"
            checkpoint["finalistsAccepted"] += 1
            lane_metrics["finalistsAccepted"] += 1
            checkpoint["preparedVideoCount"] = len(checkpoint["preparedItems"])
            checkpoint["renderReadyCount"] = len(checkpoint["preparedItems"])
            self._checkpoint(job_id, checkpoint)
        checkpoint["stageProgress"] = 1.0
        checkpoint["activeWorkerCount"] = 0
        self._checkpoint(job_id, checkpoint)

    def _campaign_spec(
        self, request: Mapping[str, Any], *, lane: str, seed_start: int, count: int
    ) -> SimulationCampaignSpecV1:
        runtime = dict(request["runtimeSnapshot"])
        seats = dict(dict(runtime["laneSeatAgents"])[lane])
        agent_release = str(dict(runtime["laneAgentReleases"])[lane])
        return SimulationCampaignSpecV1.build(
            rules_hash=str(runtime["rulesHash"]),
            bundle_hash=str(runtime["bundleHash"]),
            engine_version=str(runtime["engineVersion"]),
            analysis_profile_hash=str(runtime["analysisProfileHash"]),
            perspective_profile_hash=str(runtime["perspectiveProfileHash"]),
            selection_profile_hash=str(runtime["selectionProfileHash"]),
            agent_release=agent_release,
            seat_agents=tuple(
                seats[seat] for seat in ("north", "west", "south", "east")
            ),
            lanes=(lane,),
            lane_profile_hashes={lane: str(runtime["staticLaneProfileHashes"][lane])},
            seed_start=seed_start,
            seed_count=count,
            seeds_per_shard=min(int(request["seedsPerShard"]), count),
        )

    def _execute(self, job_id: str) -> None:
        with self._active_lock:
            if job_id in self._active_jobs:
                return
            self._active_jobs.add(job_id)
        try:
            self._execute_once(job_id)
        finally:
            with self._active_lock:
                self._active_jobs.discard(job_id)

    def _execute_once(self, job_id: str) -> None:
        item = self.get(job_id)
        if item is None or item["status"] in {
            "completed",
            "failed",
            "cancelled",
            "exhausted",
        }:
            return
        request = dict(item["request"])
        if dict(request.get("runtimeSnapshot") or {}) != self.runtime_snapshot(
            self.runtime_profile
        ):
            self.jobs.update(
                job_id,
                status="failed",
                progress=1.0,
                error="active runtime changed after video batch submission",
            )
            return
        checkpoint = dict(item.get("result") or self._initial_checkpoint(request))
        checkpoint.setdefault("validatedTraces", [])
        checkpoint.setdefault("rejectedTraces", [])
        checkpoint.setdefault("validationItems", [])
        checkpoint.setdefault("renderItems", [])
        if "libraryExcludedTraces" not in checkpoint:
            checkpoint["libraryExcludedTraces"] = sorted(self._qualified_trace_hashes())
        try:
            checkpoint["stage"] = "searching"
            self._checkpoint(job_id, checkpoint)
            pending_validations = [
                value
                for value in checkpoint["validationItems"]
                if value.get("status") not in {"accepted", "rejected", "superseded"}
            ]
            if pending_validations:
                checkpoint["stage"] = "qualifying"
                self._checkpoint(job_id, checkpoint)
                self._run_validation_records(
                    job_id,
                    checkpoint,
                    request,
                    pending_validations,
                )
                if request["renderPolicy"] == "automatic":
                    checkpoint["stage"] = "rendering"
                    _, render_failures, _ = self._submit_render_slots(
                        job_id,
                        checkpoint,
                        request,
                    )
                    if render_failures:
                        raise RuntimeError(
                            "one or more verified videos failed rendering"
                        )
            while int(checkpoint["preparedVideoCount"]) < int(
                request["targetVideos"]
            ) and int(checkpoint["roundsGenerated"]) < int(request["maximumRounds"]):
                checkpoint["stage"] = "searching"
                remaining = int(request["maximumRounds"]) - int(
                    checkpoint["roundsGenerated"]
                )
                count = min(int(request["roundsPerWave"]), remaining)
                seed_start = int(request["seedStart"]) + int(
                    checkpoint["roundsGenerated"]
                )
                lane_order = tuple(str(value) for value in request["laneOrder"])
                lane = lane_order[len(checkpoint["campaigns"]) % len(lane_order)]
                campaign = self.campaign_store.create_campaign(
                    self._campaign_spec(
                        request, lane=lane, seed_start=seed_start, count=count
                    )
                )
                campaign_hash = str(campaign["campaignHash"])
                if campaign_hash not in checkpoint["campaigns"]:
                    checkpoint["campaigns"].append(campaign_hash)
                checkpoint["currentWave"] = len(checkpoint["campaigns"])
                checkpoint["stageProgress"] = 0.0
                checkpoint["activeWorkerCount"] = 0
                self._checkpoint(job_id, checkpoint)
                self.schedule_campaign(campaign_hash)
                last_campaign_signature: tuple[tuple[str, int], ...] | None = None

                def report_campaign_progress(child: Mapping[str, Any]) -> None:
                    nonlocal last_campaign_signature
                    counts = {
                        str(key): int(value)
                        for key, value in dict(child.get("shardCounts") or {}).items()
                    }
                    signature = tuple(sorted(counts.items()))
                    if signature == last_campaign_signature:
                        return
                    last_campaign_signature = signature
                    total = max(1, sum(counts.values()))
                    terminal = sum(
                        counts.get(status, 0)
                        for status in ("completed", "failed", "cancelled")
                    )
                    checkpoint["stageProgress"] = min(1.0, terminal / total)
                    checkpoint["activeWorkerCount"] = counts.get("running", 0)
                    self._checkpoint(job_id, checkpoint)

                campaign_started = time.perf_counter()
                campaign = self._wait(
                    job_id,
                    lambda value=campaign_hash: self.campaign_store.get_campaign(value),
                    on_poll=report_campaign_progress,
                )
                self._add_stage_wall_seconds(
                    checkpoint,
                    "simulation",
                    time.perf_counter() - campaign_started,
                )
                checkpoint["roundsGenerated"] += count
                checkpoint["stageProgress"] = 1.0
                checkpoint["activeWorkerCount"] = 0
                if campaign["status"] == "cancelled":
                    raise InterruptedError(
                        str(campaign.get("error") or "video batch campaign cancelled")
                    )
                if campaign["status"] != "completed":
                    raise RuntimeError(
                        str(
                            campaign.get("error")
                            or f"video batch campaign ended as {campaign['status']}"
                        )
                    )
                checkpoint["campaignsCompleted"] += 1
                lane_metrics = self._lane_metrics(checkpoint, lane)
                lane_metrics["roundsGenerated"] += count
                lane_metrics["campaignsCompleted"] += 1
                checkpoint["stage"] = "analyzing"
                try:
                    analysis = self.deep_analysis.submit(campaign_hash)
                except NoRetainedCandidateTracesError:
                    self._record_gate_disposition(checkpoint, "no_retained_trace")
                    checkpoint["rejections"].append(
                        {
                            "stage": "screening",
                            "reasonCode": "no_retained_trace",
                            "reason": "search wave produced no retained candidate trace",
                            "campaignHash": campaign_hash,
                        }
                    )
                    checkpoint["stageProgress"] = 1.0
                    checkpoint["activeWorkerCount"] = 0
                    self._checkpoint(job_id, checkpoint)
                    continue
                analysis_id = str(analysis["jobId"])
                if analysis_id not in checkpoint["analysisJobs"]:
                    checkpoint["analysisJobs"].append(analysis_id)
                self._checkpoint(job_id, checkpoint)
                self.schedule_analysis()
                last_analysis_percent = -1

                def report_analysis_progress(child: Mapping[str, Any]) -> None:
                    nonlocal last_analysis_percent
                    child_progress = max(
                        0.0,
                        min(1.0, float(child.get("progress", 0.0))),
                    )
                    percent = int(child_progress * 100.0)
                    if percent == last_analysis_percent:
                        return
                    last_analysis_percent = percent
                    checkpoint["stageProgress"] = child_progress
                    checkpoint["activeWorkerCount"] = (
                        1 if child.get("status") in {"queued", "running"} else 0
                    )
                    self._checkpoint(job_id, checkpoint)

                analysis_started = time.perf_counter()
                analysis = self._wait(
                    job_id,
                    lambda value=analysis_id: self.deep_analysis.get(value),
                    on_poll=report_analysis_progress,
                )
                self._add_stage_wall_seconds(
                    checkpoint,
                    "analysis",
                    time.perf_counter() - analysis_started,
                )
                checkpoint["stageProgress"] = 1.0
                checkpoint["activeWorkerCount"] = 0
                if analysis["status"] == "cancelled":
                    raise InterruptedError(
                        str(analysis.get("error") or "video batch analysis cancelled")
                    )
                if analysis["status"] != "completed":
                    raise RuntimeError(
                        str(
                            analysis.get("error")
                            or f"video batch analysis ended as {analysis['status']}"
                        )
                    )
                checkpoint["stage"] = "qualifying"
                self._checkpoint(job_id, checkpoint)
                page = self.candidates.list_candidates(analysis_id)
                raw_values = [dict(value) for value in page.get("items") or ()]
                lane_metrics["rawCandidates"] += len(raw_values)
                gate_counts: dict[str, int] = {}
                accepted_values: list[dict[str, Any]] = []
                for value in raw_values:
                    reason_code = self._candidate_gate_rejection_code(value, request)
                    if reason_code is None:
                        accepted_values.append(value)
                    else:
                        gate_counts[reason_code] = gate_counts.get(reason_code, 0) + 1
                values = sorted(
                    accepted_values,
                    key=self._candidate_sort_key,
                )
                checkpoint["retainedCandidates"] += len(values)
                lane_metrics["candidatesAfterBatchGates"] += len(values)
                filtered_count = len(raw_values) - len(values)
                if filtered_count:
                    dispositions = [
                        {"code": code, "count": count}
                        for code, count in sorted(gate_counts.items())
                    ]
                    for code, disposition_count in gate_counts.items():
                        self._record_gate_disposition(
                            checkpoint, code, disposition_count
                        )
                    checkpoint["rejections"].append(
                        {
                            "stage": "quality_filter",
                            "reasonCode": "batch_gate_filtered",
                            "reason": f"{filtered_count} candidates did not pass the governed batch gates",
                            "count": filtered_count,
                            "dispositions": dispositions,
                            "campaignHash": campaign_hash,
                        }
                    )
                self._qualify_wave(
                    job_id,
                    checkpoint,
                    request,
                    campaign_hash=campaign_hash,
                    analysis_id=analysis_id,
                    candidates=values,
                )

                # Automatic rendering is an explicit governed opt-in. The
                # default Foundry path stops at immutable render-ready bundles.
                if request["renderPolicy"] == "automatic":
                    checkpoint["stage"] = "rendering"
                    _, render_failures, _ = self._submit_render_slots(
                        job_id,
                        checkpoint,
                        request,
                    )
                    if render_failures:
                        raise RuntimeError(
                            "one or more verified videos failed rendering"
                        )
                self._checkpoint(job_id, checkpoint)

            target_reached = int(checkpoint["preparedVideoCount"]) >= int(
                request["targetVideos"]
            )
            if int(checkpoint["preparedVideoCount"]) == 0:
                checkpoint["stage"] = "exhausted"
                checkpoint["outcome"] = "maximum_rounds_reached"
                self.jobs.update(
                    job_id, status="exhausted", progress=1.0, result=checkpoint
                )
                return

            checkpoint["searchTargetReached"] = target_reached
            if request["renderPolicy"] == "on_demand":
                checkpoint["stage"] = (
                    "ready_for_selection" if target_reached else "exhausted"
                )
                checkpoint["outcome"] = (
                    "target_reached" if target_reached else "maximum_rounds_reached"
                )
                self.jobs.update(
                    job_id,
                    status="completed" if target_reached else "exhausted",
                    progress=1.0,
                    result=checkpoint,
                )
                return

            checkpoint["stage"] = "draining"
            self._checkpoint(job_id, checkpoint)
            selected_count = min(
                len(checkpoint["preparedItems"]),
                int(request["targetVideos"]),
            )
            while True:
                if self.jobs.cancelled(job_id):
                    raise InterruptedError("video batch cancelled")
                completed, failures, active = self._submit_render_slots(
                    job_id,
                    checkpoint,
                    request,
                )
                self._checkpoint(job_id, checkpoint)
                if failures or (completed >= selected_count and active == 0):
                    break
                time.sleep(self.poll_seconds)
            if failures:
                checkpoint["stage"] = "failed"
                checkpoint["outcome"] = "render_failures"
                status = "failed"
                error = "one or more verified videos failed rendering"
            elif target_reached and completed >= int(request["targetVideos"]):
                checkpoint["stage"] = "completed"
                checkpoint["outcome"] = "target_reached"
                status = "completed"
                error = None
            else:
                checkpoint["stage"] = "exhausted"
                checkpoint["outcome"] = "maximum_rounds_reached"
                status = "exhausted"
                error = None
            self.jobs.update(
                job_id,
                status=status,
                progress=1.0,
                result=checkpoint,
                error=error,
            )
        except InterruptedError as exc:
            self._cancel_children(checkpoint)
            checkpoint["stage"] = "cancelled"
            checkpoint["outcome"] = "cancelled"
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                result=checkpoint,
                error=str(exc),
            )
        except Exception as exc:
            checkpoint["stage"] = "failed"
            checkpoint["outcome"] = (
                "render_failures" if "failed rendering" in str(exc) else "failed"
            )
            self.jobs.update(
                job_id,
                status="failed",
                progress=1.0,
                result=checkpoint,
                error=f"{type(exc).__name__}: {exc}",
            )
