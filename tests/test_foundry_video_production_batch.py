from __future__ import annotations

import hashlib
import json
from pathlib import Path

from baloot_platform.challenge.analysis_catalog import NoPlayableFinalistCandidateError
from baloot_platform.challenge.video_production_batch import (
    FoundryVideoBatchServiceV1,
    HookedLaffaBatchProfileV1,
)
from baloot_platform.challenge.deep_analysis import NoRetainedCandidateTracesError
from baloot_platform.storage.jobs import JobStore


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _InlineExecutor:
    def submit(self, callback, *args):
        callback(*args)


class _DeferredExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, callback, *args):
        self.tasks.append((callback, args))


class _QualifiedLibrary:
    def __init__(self, items=()):
        self.items = [dict(item) for item in items]

    def list_qualified_laffat(self, *, cursor=None, limit=500, **_filters):
        assert cursor is None
        return {"items": self.items[:limit], "nextCursor": None}

    def put_qualified_laffa(self, item):
        self.items.append(dict(item))
        return dict(item)


class _RecordingJobs(JobStore):
    def __init__(self, path):
        super().__init__(path)
        self.stages = []

    def update(self, job_id, **values):
        result = values.get("result")
        if result and result.get("stage"):
            self.stages.append(result["stage"])
        return super().update(job_id, **values)


class _Campaigns:
    def __init__(self):
        self.values = {}
        self.cancelled = []

    def create_campaign(self, spec):
        item = {
            "campaignHash": spec.sha256,
            "status": "completed",
            "spec": spec.to_dict(),
        }
        self.values[spec.sha256] = item
        return item

    def get_campaign(self, campaign_hash):
        return self.values.get(campaign_hash)

    def request_cancel(self, campaign_hash):
        self.cancelled.append(campaign_hash)
        return True


class _FailedCampaigns(_Campaigns):
    def create_campaign(self, spec):
        item = {
            "campaignHash": spec.sha256,
            "status": "failed",
            "error": "claimed campaign does not match this worker's frozen runtime",
            "spec": spec.to_dict(),
        }
        self.values[spec.sha256] = item
        return item


class _Analysis:
    def __init__(self):
        self.values = {}
        self.cancelled = []

    def submit(self, campaign_hash):
        job_id = f"analysis-{len(self.values) + 1}"
        item = {"jobId": job_id, "status": "completed", "campaignHash": campaign_hash}
        self.values[job_id] = item
        return item

    def get(self, job_id):
        return self.values.get(job_id)

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return True


class _FailedAnalysis(_Analysis):
    def submit(self, campaign_hash):
        job_id = f"analysis-{len(self.values) + 1}"
        item = {
            "jobId": job_id,
            "status": "failed",
            "campaignHash": campaign_hash,
            "error": "analysis runtime identity mismatch",
        }
        self.values[job_id] = item
        return item


class _EmptyAnalysis(_Analysis):
    def submit(self, campaign_hash):
        raise NoRetainedCandidateTracesError(
            "deep analysis requires at least one retained candidate trace"
        )


class _Candidates:
    def __init__(self, *, yield_candidates=True):
        self.yield_candidates = yield_candidates

    def list_candidates(self, analysis_id):
        if not self.yield_candidates:
            return {"items": []}
        ordinal = int(analysis_id.rsplit("-", 1)[1])
        candidate_hash = _sha(f"candidate-{ordinal}")
        return {
            "items": [
                {
                    "candidateHash": candidate_hash,
                    "shortCode": candidate_hash[:8].upper(),
                    "decisionKind": "play",
                    "truthLabel": "exact_optimal",
                    "selectionScore": 80.0,
                    "confidence": 1.0,
                    "critical": True,
                    "compatibilityBlockers": [],
                    "dimensions": {"source": "guided"},
                    "technical": {"traceHash": _sha(f"trace-{ordinal}")},
                }
            ]
        }

    def prepare_best_finalist_candidate(
        self, finalist, *, reveal_policy, export_preset
    ):
        candidate_hash = finalist["request"]["candidateHashes"][0]
        owner_hash = _sha(f"bundle-{candidate_hash}")
        return {
            "basename": f"balootx-{candidate_hash[:8]}-laffa",
            "locator": {"code": candidate_hash[:8].upper()},
            "bundleArtifact": {
                "ownerKind": "render_bundle_v1",
                "ownerHash": owner_hash,
                "uri": f"memory://{owner_hash}",
                "sha256": owner_hash,
                "bytes": 1,
                "mediaType": "application/vnd.balootx.render-bundle-v1+json",
            },
            "qualification": {
                "schemaVersion": 1,
                "localPreviewReady": True,
                "publicationReady": False,
                "publicationBlockers": ["professional_agent_qualification_missing"],
                "finalistManifestHash": _sha("finalist-manifest"),
                "replayEvidenceHash": _sha("replay-evidence"),
                "analysisEvidenceHash": _sha("analysis-evidence"),
            },
        }


class _SameTraceCandidates(_Candidates):
    def list_candidates(self, analysis_id):
        page = super().list_candidates(analysis_id)
        page["items"][0]["technical"]["traceHash"] = _sha("same-trace")
        return page


class _FirstPrepareRejectCandidates(_Candidates):
    def __init__(self):
        super().__init__()
        self.prepare_calls = 0

    def prepare_best_finalist_candidate(
        self, finalist, *, reveal_policy, export_preset
    ):
        self.prepare_calls += 1
        if self.prepare_calls == 1:
            raise NoPlayableFinalistCandidateError(
                "finalist validation has no surviving playable candidate"
            )
        return super().prepare_best_finalist_candidate(
            finalist,
            reveal_policy=reveal_policy,
            export_preset=export_preset,
        )


class _DuplicateTracePageCandidates(_Candidates):
    def list_candidates(self, analysis_id):
        best = super().list_candidates(analysis_id)["items"][0]
        trace_hash = _sha("single-complete-laffa")
        best["technical"]["traceHash"] = trace_hash
        alternatives = []
        for index, score in enumerate((79.0, 78.0), start=1):
            alternatives.append(
                {
                    **best,
                    "candidateHash": _sha(f"same-laffa-alternative-{index}"),
                    "selectionScore": score,
                    "technical": {"traceHash": trace_hash},
                }
            )
        return {"items": [*alternatives, best]}


class _ManyTraceCandidates(_Candidates):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def list_candidates(self, analysis_id):
        values = []
        for index in range(self.count):
            candidate_hash = _sha(f"many-candidate-{index}")
            values.append(
                {
                    "candidateHash": candidate_hash,
                    "shortCode": candidate_hash[:8].upper(),
                    "decisionKind": "play",
                    "truthLabel": "exact_optimal",
                    "selectionScore": 90.0 - index,
                    "confidence": 1.0,
                    "critical": True,
                    "compatibilityBlockers": [],
                    "technical": {"traceHash": _sha(f"many-trace-{index}")},
                }
            )
        return {"items": list(reversed(values))}


class _MixedQualityCandidates(_Candidates):
    def list_candidates(self, analysis_id):
        good = super().list_candidates(analysis_id)["items"][0]
        rejected = []
        for index, update in enumerate(
            (
                {"decisionKind": "bid"},
                {"critical": False},
                {"truthLabel": "unresolved"},
                {"selectionScore": 59.9},
                {"confidence": 0.74},
                {"compatibilityBlockers": ["profile mismatch"]},
            )
        ):
            item = {**good, "candidateHash": _sha(f"rejected-{index}"), **update}
            rejected.append(item)
        return {"items": rejected + [good]}


class _Finalists:
    def __init__(self):
        self.values = {}
        self.cancelled = []

    def submit(
        self,
        *,
        analysis_job_id,
        candidate_hashes,
        idempotency_key,
        candidate_resimulation=False,
        required_decision_kind=None,
    ):
        job_id = f"finalist-{candidate_hashes[0][:12]}"
        resimulation_hash = _sha(f"resimulation-{candidate_hashes[0]}")
        finalist_hash = _sha(f"finalist-descendant-{candidate_hashes[0]}")
        item = {
            "jobId": job_id,
            "status": "completed",
            "request": {
                "analysisJobId": analysis_job_id,
                "candidateHashes": list(candidate_hashes),
                "requiredDecisionKind": required_decision_kind,
            },
            "result": {
                "localPreviewReady": True,
                "items": [
                    {
                        "candidateResimulationTraceHash": (
                            resimulation_hash if candidate_resimulation else None
                        ),
                        "descendantTraceHash": finalist_hash,
                        "candidateResimulation": {
                            "schemaVersion": 1,
                            "stage": "candidate_resimulation",
                            "status": (
                                "completed"
                                if candidate_resimulation
                                else "not_requested"
                            ),
                            **(
                                {
                                    "profileHash": _sha(
                                        "candidate-resimulation-profile"
                                    ),
                                    "exactDeckCommitmentPreserved": True,
                                    "parentTraceLinkPreserved": True,
                                }
                                if candidate_resimulation
                                else {}
                            ),
                        },
                        "analysisEvidence": {
                            "laffaSelectionScore": {
                                "schemaVersion": 1,
                                "scoreFamily": "laffa_content_selection",
                                "profileHash": _sha("thrill-profile"),
                                "evidenceHash": _sha("verified-laffa-evidence"),
                                "score": 90.0,
                                "passed": True,
                                "rejectionReasons": [],
                                "sha256": _sha("laffa-selection-score"),
                            }
                        },
                    }
                ],
            },
        }
        self.values[job_id] = item
        return item

    def get(self, job_id):
        return self.values.get(job_id)

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return True


class _BoundedReverseFinalists(_Finalists):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.peak_active = 0
        self.release_order = []

    def submit(
        self,
        *,
        analysis_job_id,
        candidate_hashes,
        idempotency_key,
        candidate_resimulation=False,
        required_decision_kind=None,
    ):
        item = super().submit(
            analysis_job_id=analysis_job_id,
            candidate_hashes=candidate_hashes,
            idempotency_key=idempotency_key,
            candidate_resimulation=candidate_resimulation,
            required_decision_kind=required_decision_kind,
        )
        item["status"] = "queued"
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        return item

    def get(self, job_id):
        item = self.values.get(job_id)
        if item is not None and item["status"] == "queued":
            queued = [
                value for value in self.values.values() if value["status"] == "queued"
            ]
            for value in reversed(queued):
                value["status"] = "completed"
                self.release_order.append(value["request"]["candidateHashes"][0])
                self.active -= 1
        return item


class _FailedCandidateResimulationFinalists(_Finalists):
    def submit(self, **values):
        item = super().submit(**values)
        item["status"] = "failed"
        item["result"] = None
        item["error"] = (
            "RuntimeError: candidate resimulation failed: exact deal changed"
        )
        return item


class _FirstNoTacticalPlayFinalists(_Finalists):
    def __init__(self):
        super().__init__()
        self.submissions = 0

    def submit(self, **values):
        item = super().submit(**values)
        self.submissions += 1
        if self.submissions == 1:
            item["result"]["localPreviewReady"] = False
            item["result"]["items"][0]["blockers"] = [
                "no_verified_tactical_play_opportunity"
            ]
            item["result"]["items"][0]["analysisEvidence"] = None
        return item


class _Renders:
    def __init__(self):
        self.values = {}
        self.cancelled = []

    def submit(self, items, *, idempotency_key):
        job_id = f"render-{len(self.values) + 1}"
        item = {
            "jobId": job_id,
            "status": "completed",
            "result": {
                "execution": {
                    "items": [
                        {"basename": value["basename"], "status": "completed"}
                        for value in items
                    ]
                }
            },
        }
        self.values[job_id] = item
        return item

    def get(self, job_id):
        return self.values.get(job_id)

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return True


class _FailedRenders(_Renders):
    def submit(self, items, *, idempotency_key):
        item = super().submit(items, idempotency_key=idempotency_key)
        item["status"] = "failed"
        item["error"] = "encoder failed"
        item["result"]["execution"]["items"][0]["status"] = "failed"
        return item


class _StreamingCampaigns(_Campaigns):
    def __init__(self, renders):
        super().__init__()
        self.renders = renders
        self.created = 0

    def create_campaign(self, spec):
        self.created += 1
        if self.created == 2:
            assert len(self.renders.values) == 1
        return super().create_campaign(spec)


def _runtime():
    release = _sha("agent")
    return {
        "rulesHash": _sha("rules"),
        "bundleHash": _sha("bundle"),
        "engineVersion": "test",
        "analysisProfileHash": _sha("analysis"),
        "searchBackend": {"id": "test-search-backend"},
        "matchValueProfile": {"sha256": _sha("match-value")},
        "perspectiveProfileHash": _sha("perspective"),
        "selectionProfileHash": _sha("selection"),
        "agentRelease": release,
        "seatAgents": {seat: release for seat in ("north", "west", "south", "east")},
        "staticLaneProfileHashes": {
            lane: _sha(lane) for lane in ("organic", "guided", "disagreement")
        },
        "laneAgentReleases": {
            lane: release for lane in ("organic", "guided", "disagreement")
        },
        "laneSeatAgents": {
            lane: {seat: release for seat in ("north", "west", "south", "east")}
            for lane in ("organic", "guided", "disagreement")
        },
        "candidateResimulationProfile": {
            "sha256": _sha("candidate-resimulation-profile")
        },
        "candidateResimulationAgentRelease": _sha("candidate-resimulation-agent"),
        "decisionAnalysisExecutorProfile": {
            stage: {"sha256": _sha(f"decision-executor-{stage}")}
            for stage in ("screening", "confirmation", "finalistReanalysis")
        },
    }


def _request(
    runtime,
    *,
    target=2,
    maximum=2,
    render_policy="on_demand",
    candidate_resimulation=False,
):
    return {
        "schemaVersion": 1,
        "targetVideos": target,
        "laneOrder": ["guided", "organic"],
        "seedStart": 100,
        "roundsPerWave": 1,
        "maximumRounds": maximum,
        "seedsPerShard": 1,
        "finalistConcurrency": 2,
        "useCandidateResimulation": candidate_resimulation,
        "renderConcurrency": 1,
        "renderPolicy": render_policy,
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumSelectionScore": 60.0,
        "minimumConfidence": 0.75,
        "requirePlayDecision": True,
        "requireCritical": True,
        "exportPreset": "hooked_flagship",
        "revealPolicy": "after_commit",
        "runtimeSnapshot": FoundryVideoBatchServiceV1.runtime_snapshot(runtime),
    }


def _service(
    tmp_path: Path,
    *,
    yield_candidates=True,
    candidates=None,
    campaigns=None,
    analysis=None,
    finalists=None,
    renders=None,
    jobs=None,
    qualified_library=None,
    executor=None,
):
    runtime = _runtime()
    return runtime, FoundryVideoBatchServiceV1(
        jobs=jobs or JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=executor or _InlineExecutor(),
        campaign_store=campaigns or _Campaigns(),
        deep_analysis=analysis or _Analysis(),
        candidates=candidates or _Candidates(yield_candidates=yield_candidates),
        finalist_validations=finalists or _Finalists(),
        render_batches=renders or _Renders(),
        runtime_profile=runtime,
        schedule_campaign=lambda _: None,
        schedule_analysis=lambda: None,
        poll_seconds=0.001,
        qualified_library=qualified_library,
    )


def test_profile_is_closed_and_hashable():
    profile = HookedLaffaBatchProfileV1.build(
        {
            "schemaVersion": 1,
            "id": "hooked_laffa_batch_v1",
            "revision": "1",
            "defaultTarget": 10,
            "maximumTarget": 100,
            "laneOrder": ["guided", "disagreement", "organic"],
            "waveSeedCount": 20,
            "maximumWaves": 20,
            "seedsPerShard": 5,
            "simulationWorkerCount": 2,
            "simulationTimeoutSeconds": 7200,
            "finalistConcurrency": 2,
            "useCandidateResimulation": False,
            "renderConcurrency": 1,
            "renderPolicy": "on_demand",
            "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
            "minimumSelectionScore": 60.0,
            "minimumConfidence": 0.75,
            "requirePlayDecision": True,
            "requireCritical": True,
        }
    )
    assert profile.to_dict()["sha256"] == profile.sha256
    assert profile.default_target == 10
    assert profile.use_candidate_resimulation is False


def test_production_lane_allocation_is_governed_by_retained_evidence(root: Path):
    evidence = json.loads(
        (root / "release" / "foundry-lane-allocation-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["mixedAllocation"]["roundsGenerated"] == 390
    assert evidence["lanes"]["guided"]["finalistsAccepted"] == 5
    assert evidence["lanes"]["organic"]["finalistsAccepted"] == 0
    assert evidence["lanes"]["disagreement"]["finalistsAccepted"] == 0
    assert evidence["decision"] == {
        "profileId": "hooked_laffa_batch_v1",
        "revision": "7",
        "laneOrder": ["guided"],
        "truthGatesChanged": False,
        "renderPolicyChanged": False,
        "coverageLanesRemovedFromPlatform": False,
    }


def test_candidate_ranking_is_quality_first_and_hash_stable():
    candidates = [
        {
            "decisionKind": "play",
            "truthLabel": "exact_optimal",
            "selectionScore": 80,
            "confidence": 0.8,
            "candidateHash": "b",
        },
        {
            "decisionKind": "play",
            "truthLabel": "exact_optimal",
            "selectionScore": 80,
            "confidence": 0.9,
            "candidateHash": "z",
        },
        {
            "decisionKind": "play",
            "truthLabel": "exact_optimal",
            "selectionScore": 80,
            "confidence": 0.9,
            "candidateHash": "a",
        },
        {
            "decisionKind": "play",
            "truthLabel": "robust_best_known",
            "selectionScore": 100,
            "confidence": 1.0,
            "candidateHash": "x",
        },
    ]
    ranked = sorted(candidates, key=FoundryVideoBatchServiceV1._candidate_sort_key)
    assert [item["candidateHash"] for item in ranked] == ["a", "z", "b", "x"]


def test_batch_reaches_exact_distinct_target(tmp_path: Path):
    finalists = _Finalists()
    runtime, service = _service(tmp_path, finalists=finalists)
    job = service.submit(_request(runtime), idempotency_key="exact-two")
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    assert completed["result"]["completedVideoCount"] == 0
    assert completed["result"]["preparedVideoCount"] == 2
    assert completed["result"]["renderReadyCount"] == 2
    assert completed["result"]["stage"] == "ready_for_selection"
    assert completed["result"]["renderJobs"] == []
    assert (
        len({item["traceHash"] for item in completed["result"]["preparedItems"]}) == 2
    )
    assert all(
        item["selectionScore"] == 80.0
        and item["laffaSelectionScore"]["score"] == 90.0
        and item["laffaSelectionScore"]["passed"] is True
        and item["confidence"] == 1.0
        and item["truthLabel"] == "exact_optimal"
        and item["decisionKind"] == "play"
        for item in completed["result"]["preparedItems"]
    )
    assert all(
        item["request"]["requiredDecisionKind"] == "play"
        for item in finalists.values.values()
    )
    assert all(
        item["qualification"]["localPreviewReady"] is True
        and item["qualification"]["publicationReady"] is False
        and "professional_agent_qualification_missing"
        in item["qualification"]["publicationBlockers"]
        for item in completed["result"]["preparedItems"]
    )
    assert all(
        item["candidateResimulation"]["status"] == "not_requested"
        and item["candidateResimulationTraceHash"] is None
        and item["finalistDescendantTraceHash"]
        and item["finalistJobId"]
        for item in completed["result"]["preparedItems"]
    )
    historical_projection = dict(completed["result"]["preparedItems"][0])
    historical_projection.pop("laffaSelectionScore")
    enriched = service._prepared_laffa_projection(completed, historical_projection)
    assert enriched["laffaSelectionScore"]["score"] == 90.0
    assert enriched["finalistJobId"]


def test_batch_excludes_laffat_already_in_the_durable_library(tmp_path: Path):
    existing_trace = _sha("trace-1")
    library = _QualifiedLibrary(({"traceHash": existing_trace},))
    runtime, service = _service(tmp_path, qualified_library=library)

    job = service.submit(
        _request(runtime, target=1, maximum=2),
        idempotency_key="exclude-library-trace",
    )
    completed = service.get(job["jobId"])

    assert completed["status"] == "completed"
    assert completed["result"]["campaignsCompleted"] == 2
    assert completed["result"]["preparedItems"][0]["traceHash"] == _sha("trace-2")
    assert (
        completed["result"]["operationalMetrics"]["gateDispositions"]["duplicate_laffa"]
        == 1
    )


def test_batch_stops_after_first_failed_campaign(tmp_path: Path):
    campaigns = _FailedCampaigns()
    analysis = _Analysis()
    runtime, service = _service(
        tmp_path,
        campaigns=campaigns,
        analysis=analysis,
    )
    job = service.submit(
        _request(runtime, target=2, maximum=10),
        idempotency_key="failed-campaign",
    )
    failed = service.get(job["jobId"])
    assert failed["status"] == "failed"
    assert "frozen runtime" in failed["error"]
    assert len(campaigns.values) == 1
    assert analysis.values == {}


def test_batch_stops_after_first_failed_analysis(tmp_path: Path):
    campaigns = _Campaigns()
    analysis = _FailedAnalysis()
    runtime, service = _service(
        tmp_path,
        campaigns=campaigns,
        analysis=analysis,
    )
    job = service.submit(
        _request(runtime, target=2, maximum=10),
        idempotency_key="failed-analysis",
    )
    failed = service.get(job["jobId"])
    assert failed["status"] == "failed"
    assert "runtime identity mismatch" in failed["error"]
    assert len(campaigns.values) == 1
    assert len(analysis.values) == 1


def test_batch_exhaustion_is_not_reported_as_success(tmp_path: Path):
    runtime, service = _service(tmp_path, yield_candidates=False)
    job = service.submit(
        _request(runtime, target=1, maximum=2), idempotency_key="zero-yield"
    )
    exhausted = service.get(job["jobId"])
    assert exhausted["status"] == "exhausted"
    assert exhausted["result"]["outcome"] == "maximum_rounds_reached"
    assert exhausted["result"]["completedVideoCount"] == 0


def test_batch_continues_when_a_wave_has_no_retained_trace(tmp_path: Path):
    runtime, service = _service(tmp_path, analysis=_EmptyAnalysis())
    job = service.submit(
        _request(runtime, target=1, maximum=2),
        idempotency_key="empty-screening-waves",
    )
    exhausted = service.get(job["jobId"])
    assert exhausted["status"] == "exhausted"
    assert exhausted["result"]["roundsGenerated"] == 2
    assert len(exhausted["result"]["campaigns"]) == 2
    assert all(
        item["stage"] == "screening" for item in exhausted["result"]["rejections"]
    )


def test_batch_continues_when_finalist_has_no_playable_decision(tmp_path: Path):
    candidates = _FirstPrepareRejectCandidates()
    runtime, service = _service(tmp_path, candidates=candidates)
    job = service.submit(
        _request(runtime, target=1, maximum=2),
        idempotency_key="no-playable-finalist",
    )
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    assert completed["result"]["completedVideoCount"] == 0
    assert completed["result"]["finalistsRejected"] == 1
    assert any(
        item["stage"] == "finalist"
        and item["reasonCode"] == "no_playable_finalist"
        and "no surviving playable candidate" in item["reason"]
        for item in completed["result"]["rejections"]
    )


def test_batch_continues_after_exact_tactical_impossibility(tmp_path: Path):
    finalists = _FirstNoTacticalPlayFinalists()
    runtime, service = _service(tmp_path, finalists=finalists)
    job = service.submit(
        _request(runtime, target=1, maximum=2),
        idempotency_key="no-verified-tactical-play",
    )
    completed = service.get(job["jobId"])

    assert completed["status"] == "completed"
    assert completed["result"]["finalistsAccepted"] == 1
    assert completed["result"]["finalistsRejected"] == 1
    assert (
        completed["result"]["operationalMetrics"]["gateDispositions"][
            "no_verified_tactical_play_opportunity"
        ]
        == 1
    )
    assert any(
        item["reasonCode"] == "no_verified_tactical_play_opportunity"
        for item in completed["result"]["rejections"]
    )


def test_batch_reports_candidate_resimulation_rejection_as_its_own_stage(
    tmp_path: Path,
):
    runtime, service = _service(
        tmp_path,
        finalists=_FailedCandidateResimulationFinalists(),
    )
    job = service.submit(
        _request(
            runtime,
            target=1,
            maximum=1,
            candidate_resimulation=True,
        ),
        idempotency_key="candidate-resimulation-rejected",
    )
    exhausted = service.get(job["jobId"])

    assert exhausted["status"] == "exhausted"
    assert exhausted["result"]["finalistsRejected"] == 1
    assert any(
        item["stage"] == "candidate_resimulation"
        and "exact deal changed" in item["reason"]
        for item in exhausted["result"]["rejections"]
    )


def test_batch_rejects_duplicate_laffat_across_waves(tmp_path: Path):
    runtime, service = _service(tmp_path, candidates=_SameTraceCandidates())
    job = service.submit(
        _request(runtime, target=2, maximum=2), idempotency_key="duplicate-trace"
    )
    exhausted = service.get(job["jobId"])
    assert exhausted["status"] == "exhausted"
    assert exhausted["result"]["preparedVideoCount"] == 1
    assert exhausted["result"]["completedVideoCount"] == 0
    assert len(exhausted["result"]["renderJobs"]) == 0
    assert (
        len({item["traceHash"] for item in exhausted["result"]["preparedItems"]}) == 1
    )


def test_batch_validates_one_representative_per_complete_trace(tmp_path: Path):
    finalists = _Finalists()
    runtime, service = _service(
        tmp_path,
        candidates=_DuplicateTracePageCandidates(),
        finalists=finalists,
    )
    job = service.submit(
        _request(runtime, target=1, maximum=1),
        idempotency_key="one-finalist-per-trace",
    )
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    assert len(finalists.values) == 1
    assert completed["result"]["preparedItems"][0]["candidateHash"] == _sha(
        "candidate-1"
    )
    assert any(
        item["stage"] == "trace_deduplication"
        for item in completed["result"]["rejections"]
    )


def test_parallel_finalists_reduce_stably_when_children_finish_in_reverse(
    tmp_path: Path,
):
    finalists = _BoundedReverseFinalists()
    runtime, service = _service(
        tmp_path,
        candidates=_ManyTraceCandidates(3),
        finalists=finalists,
    )
    request = _request(runtime, target=3, maximum=1)
    request["finalistConcurrency"] = 3
    job = service.submit(request, idempotency_key="reverse-finalists")
    completed = service.get(job["jobId"])
    expected = [_sha(f"many-candidate-{index}") for index in range(3)]
    assert completed["status"] == "completed"
    assert finalists.release_order == list(reversed(expected))
    assert [
        item["candidateHash"] for item in completed["result"]["preparedItems"]
    ] == expected


def test_finalist_concurrency_never_exceeds_governed_bound(tmp_path: Path):
    finalists = _BoundedReverseFinalists()
    runtime, service = _service(
        tmp_path,
        candidates=_ManyTraceCandidates(5),
        finalists=finalists,
    )
    request = _request(runtime, target=5, maximum=1)
    request["finalistConcurrency"] = 2
    job = service.submit(request, idempotency_key="bounded-finalists")
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    assert finalists.peak_active == 2


def test_render_starts_before_the_next_search_wave(tmp_path: Path):
    renders = _Renders()
    campaigns = _StreamingCampaigns(renders)
    runtime, service = _service(
        tmp_path,
        campaigns=campaigns,
        renders=renders,
    )
    job = service.submit(
        _request(runtime, target=2, maximum=2, render_policy="automatic"),
        idempotency_key="stream-render",
    )
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    assert campaigns.created == 2
    assert len(renders.values) == 2


def test_checkpoint_reports_real_pipeline_stages(tmp_path: Path):
    jobs = _RecordingJobs(tmp_path / "stages.sqlite3")
    runtime, service = _service(tmp_path, jobs=jobs)
    job = service.submit(
        _request(runtime, target=1, maximum=1),
        idempotency_key="stage-transitions",
    )
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    for stage in ("searching", "analyzing", "qualifying", "ready_for_selection"):
        assert stage in jobs.stages
    assert completed["result"]["stageProgress"] == 1.0
    assert completed["result"]["activeWorkerCount"] == 0
    assert completed["result"]["currentWave"] == 1


def test_streamed_render_failure_fails_closed(tmp_path: Path):
    renders = _FailedRenders()
    runtime, service = _service(tmp_path, renders=renders)
    job = service.submit(
        _request(runtime, target=1, maximum=1, render_policy="automatic"),
        idempotency_key="failed-stream-render",
    )
    failed = service.get(job["jobId"])
    assert failed["status"] == "failed"
    assert failed["result"]["stage"] == "failed"
    assert failed["result"]["outcome"] == "render_failures"
    assert failed["result"]["completedVideoCount"] == 0
    assert failed["result"]["failedVideoCount"] == 1


def test_resume_finishes_a_checkpointed_planned_validation(tmp_path: Path):
    finalists = _Finalists()
    runtime, service = _service(tmp_path, finalists=finalists)
    request = service._validate_request(_request(runtime, target=1, maximum=1))
    candidate = _Candidates().list_candidates("analysis-1")["items"][0]
    checkpoint = FoundryVideoBatchServiceV1._initial_checkpoint(request)
    checkpoint["roundsGenerated"] = 1
    checkpoint["validationItems"].append(
        {
            "ordinal": 0,
            "traceHash": candidate["technical"]["traceHash"],
            "candidateHash": candidate["candidateHash"],
            "analysisJobId": "analysis-1",
            "candidate": candidate,
            "status": "planned",
        }
    )
    job_id = service.jobs.create(
        "resume-planned",
        request,
        kind="foundry_video_batch_v1",
        idempotency_key="resume-planned",
    )
    service.jobs.update(job_id, status="running", result=checkpoint)

    service.resume(job_id)

    completed = service.get(job_id)
    assert completed["status"] == "completed"
    assert completed["result"]["validationItems"][0]["status"] == "accepted"
    assert completed["result"]["completedVideoCount"] == 0
    assert len(finalists.values) == 1


def test_video_batch_recovers_after_restart_and_terminalizes_inactive_cancel(
    tmp_path: Path,
):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    runtime, original = _service(
        tmp_path, jobs=jobs, executor=_DeferredExecutor()
    )
    interrupted = original.submit(
        _request(runtime, target=1, maximum=1),
        idempotency_key="restart-video-batch",
    )
    jobs.update(interrupted["jobId"], status="running", progress=0.2)

    _, restarted = _service(tmp_path, jobs=jobs)
    assert restarted.recover_pending() == (interrupted["jobId"],)
    assert restarted.get(interrupted["jobId"])["status"] == "completed"

    stale = original.submit(
        _request(runtime, target=1, maximum=1),
        idempotency_key="cancel-video-batch",
    )
    jobs.update(stale["jobId"], status="running", progress=0.2)
    assert restarted.cancel(stale["jobId"])
    terminal = restarted.get(stale["jobId"])
    assert terminal["status"] == "cancelled"
    assert terminal["progress"] == 1.0
    assert terminal["result"]["stage"] == "cancelled"


def test_batch_applies_every_governed_quality_gate(tmp_path: Path):
    runtime, service = _service(tmp_path, candidates=_MixedQualityCandidates())
    job = service.submit(
        _request(runtime, target=1, maximum=1), idempotency_key="quality-gates"
    )
    completed = service.get(job["jobId"])
    assert completed["status"] == "completed"
    assert completed["result"]["preparedVideoCount"] == 1
    assert completed["result"]["retainedCandidates"] == 1
    rejection = next(
        item
        for item in completed["result"]["rejections"]
        if item["stage"] == "quality_filter"
    )
    assert rejection["reasonCode"] == "batch_gate_filtered"
    assert rejection["count"] == 6
    assert rejection["dispositions"] == [
        {"code": "below_confidence", "count": 1},
        {"code": "below_selection_score", "count": 1},
        {"code": "compatibility_blocker", "count": 1},
        {"code": "disallowed_truth_label", "count": 1},
        {"code": "not_critical", "count": 1},
        {"code": "not_play_decision", "count": 1},
    ]
    metrics = completed["result"]["operationalMetrics"]
    assert metrics["gateDispositions"] == {
        "below_confidence": 1,
        "below_selection_score": 1,
        "compatibility_blocker": 1,
        "disallowed_truth_label": 1,
        "not_critical": 1,
        "not_play_decision": 1,
    }
    assert metrics["laneMetrics"]["guided"]["rawCandidates"] == 7
    assert metrics["laneMetrics"]["guided"]["candidatesAfterBatchGates"] == 1
    assert metrics["laneMetrics"]["guided"]["finalistsAccepted"] == 1
    assert metrics["acceptedPer100Rounds"] == 100.0


def test_profile_mismatch_fails_closed(tmp_path: Path):
    runtime, service = _service(tmp_path)
    request = _request(runtime)
    request["runtimeSnapshot"] = {
        **request["runtimeSnapshot"],
        "bundleHash": _sha("other"),
    }
    try:
        service.submit(request, idempotency_key="mismatch")
    except ValueError as exc:
        assert "runtime snapshot" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("runtime mismatch was accepted")


def test_cancel_propagates_to_every_persisted_child(tmp_path: Path):
    runtime, service = _service(tmp_path)
    job_id = service.jobs.create(
        "cancel-parent",
        _request(runtime, target=1, maximum=1),
        kind="foundry_video_batch_v1",
        idempotency_key="cancel-parent",
    )
    service.jobs.update(
        job_id,
        status="running",
        result={
            **FoundryVideoBatchServiceV1._initial_checkpoint(
                _request(runtime, target=1, maximum=1)
            ),
            "campaigns": ["campaign-one"],
            "analysisJobs": ["analysis-one"],
            "finalistJobs": ["finalist-one", "finalist-two"],
            "renderJobs": ["render-one", "render-two"],
        },
    )

    assert service.cancel(job_id) is True
    assert service.campaign_store.cancelled == ["campaign-one"]
    assert service.deep_analysis.cancelled == ["analysis-one"]
    assert service.finalist_validations.cancelled == ["finalist-one", "finalist-two"]
    assert service.render_batches.cancelled == ["render-one", "render-two"]
