from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import baloot_platform.challenge.finalist_validation_service as finalist_service_module
from baloot_platform.challenge.finalist_validation import FinalistValidationProfileV1
from baloot_platform.challenge.finalist_validation_service import (
    FINALIST_VALIDATION_JOB_KIND,
    FinalistValidationJobServiceV1,
)
from baloot_platform.challenge.evidence_v3 import (
    DecisionLedgerV1,
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    TruthLabel,
    TruthMode,
)
from baloot_platform.challenge.profiles import ThrillProfileV1
from baloot_platform.core.enums import Seat
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
from baloot_platform.storage.jobs import JobStore


def _profile(
    *,
    maximum: int = 4,
    maximum_trace_seconds: int = 300,
    tactical_precheck: bool = False,
) -> FinalistValidationProfileV1:
    return FinalistValidationProfileV1.build(
        {
            "schemaVersion": 1,
            "id": "test_finalist_replay",
            "revision": "1",
            "agentProfileId": "test_equal_seat_candidate",
            "maximumTracesPerJob": maximum,
            "maximumTraceSeconds": maximum_trace_seconds,
            "requireQualifiedAgentReleaseForPublication": True,
            "allowLocalPreview": True,
            "enableExactTacticalViabilityPrecheck": tactical_precheck,
        }
    )


def _thrill_profile(*, minimum_score: float = 0.0) -> ThrillProfileV1:
    return ThrillProfileV1.build({
        "schemaVersion": 1,
        "id": "test_finalist_laffa_thrill",
        "revision": "1",
        "weights": {
            "overallPeak": 1.0,
            "outcomeImportance": 1.0,
        },
        "minimumCriticalPoints": 1,
        "minimumScore": minimum_score,
    })


class _Trace:
    def __init__(self, digest: str, *, parent: str | None = None):
        self.sha256 = digest
        self.parent = parent
        self.decision_ledger = DecisionLedgerV1.build(
            "game-test",
            tuple(
                {
                    "decisionId": f"decision-{index}",
                    "ordinal": index,
                    "actingSeat": seat.name.lower(),
                    "decisionKind": "play",
                    "phase": "playing",
                    "legalActions": [
                        {"kind": "play", "card": "AS"},
                        {"kind": "play", "card": "KS"},
                    ],
                    "chosenAction": {"kind": "play", "card": "AS"},
                }
                for index, seat in enumerate(Seat)
            ),
        )

    def to_dict(self, *, trusted: bool = False):
        value = {
            "schemaVersion": 1,
            "traceHash": self.sha256,
            "parentTraceHash": self.parent,
            "decisionLedger": self.decision_ledger.to_dict(),
        }
        if trusted:
            value["trustedFinalState"] = {"phase": "scored"}
        return value


class _Evidence:
    def __init__(self, digest: str, parent: str, descendant: str):
        self.sha256 = digest
        self.parent = parent
        self.descendant = descendant
        self.blockers = ("professional_agent_qualification_missing",)
        self.local_preview_allowed = True
        self.publication_ready = False

    def to_dict(self):
        return {
            "schemaVersion": 1,
            "parentTraceHash": self.parent,
            "descendantTraceHash": self.descendant,
            "replayComplete": True,
            "localPreviewAllowed": True,
            "publicationReady": False,
            "blockers": ["professional_agent_qualification_missing"],
            "sha256": self.sha256,
        }


class _Replay:
    descriptor = SimpleNamespace(id="finalist_replay_test", version="1.0.0")

    def __init__(self):
        self.calls = 0
        self.parents = []

    def replay(self, parent, *, cancel_check=None):
        assert not cancel_check()
        self.calls += 1
        self.parents.append(parent.sha256)
        descendant = _Trace("d" * 64, parent=parent.sha256)
        return descendant, _Evidence("e" * 64, parent.sha256, descendant.sha256)


class _TimeoutReplay(_Replay):
    def replay(self, parent, *, cancel_check=None):
        if cancel_check():
            raise InterruptedError("finalist replay cancelled")
        return super().replay(parent, cancel_check=cancel_check)


class _Candidates:
    def __init__(self, parent):
        self.parent = parent
        self.requests = []

    def candidate_trace(self, job_id, candidate_hash):
        self.requests.append((job_id, candidate_hash))
        return self.parent


class _ResimulationEvidence:
    sha256 = "9" * 64

    def to_dict(self):
        return {
            "schemaVersion": 1,
            "stage": "candidate_resimulation",
            "status": "completed",
            "sourceTraceHash": "c" * 64,
            "descendantTraceHash": "8" * 64,
            "exactDeckCommitmentPreserved": True,
            "parentTraceLinkPreserved": True,
            "sha256": self.sha256,
        }


class _CandidateResimulation:
    descriptor = SimpleNamespace(id="candidate_resimulation_test", version="1.0.0")
    profile = SimpleNamespace(sha256="7" * 64)
    simulator = SimpleNamespace(agent_release="foundry_balanced@test")

    def __init__(self):
        self.calls = 0

    def resimulate(self, source, *, cancel_check=None):
        assert not cancel_check()
        self.calls += 1
        return _Trace("8" * 64, parent=source.sha256), _ResimulationEvidence()


class _TacticalViabilityProbe:
    def __init__(self, *, can_reject: bool):
        self.can_reject_value = can_reject
        self.calls = 0

    def probe_verified_tactical_play(self, trace, *, cancel_check=None):
        assert not cancel_check()
        self.calls += 1
        status = (
            "no_verified_tactical_play_opportunity"
            if self.can_reject_value
            else "viable"
        )
        return SimpleNamespace(
            sha256="4" * 64,
            can_reject=self.can_reject_value,
            action_oracles_by_decision={},
            to_dict=lambda: {
                "schemaVersion": 1,
                "traceHash": trace.sha256,
                "eligibleDecisionCount": 4,
                "scannedDecisions": [],
                "scanComplete": self.can_reject_value,
                "viable": not self.can_reject_value,
                "status": status,
                "sha256": "4" * 64,
            },
        )


def _put(store, owner_kind, owner_hash, document):
    import json

    stored = store.put(
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        media_type=f"application/vnd.balootx.{owner_kind.replace('_', '-')}+json",
    )
    return {"ownerKind": owner_kind, "ownerHash": owner_hash, **stored}


class _Funnel:
    def __init__(
        self,
        trace,
        store,
        *,
        retain_candidate=True,
        result_profile_hash: str | None = None,
        result_trace_hashes: tuple[str, ...] | None = None,
        result_analysis_hashes: tuple[str, ...] | None = None,
    ):
        self.trace = trace
        self.store = store
        self.retain_candidate = retain_candidate
        self.analyzer = SimpleNamespace(
            profile=SimpleNamespace(sha256="3" * 64)
        )
        self.result_profile_hash = result_profile_hash
        self.result_trace_hashes = result_trace_hashes
        self.result_analysis_hashes = result_analysis_hashes
        self.calls = 0

    def process(
        self,
        references,
        *,
        cancel_check=None,
        source_by_trace=None,
        progress_callback=None,
        oracle_action_cache_by_trace=None,
    ):
        assert not cancel_check()
        trace_reference = tuple(references)[0]
        assert trace_reference["ownerHash"] == self.trace.sha256
        import json

        stored_trace = json.loads(self.store.get(trace_reference["uri"]))
        assert stored_trace["trustedFinalState"] == {"phase": "scored"}
        assert source_by_trace == {
            self.trace.sha256: "finalist_validation_descendant"
        }
        if oracle_action_cache_by_trace is not None:
            assert oracle_action_cache_by_trace == {self.trace.sha256: {}}
        self.calls += 1
        if progress_callback:
            progress_callback(1.0)
        perspectives = []
        for record in self.trace.decision_ledger.records:
            for observer in Seat:
                actor = observer == record.acting_seat
                perspectives.append(
                    DecisionPerspectiveEvidenceV3(
                        decision_id=record.decision_id,
                        acting_seat=record.acting_seat,
                        observer_seat=observer,
                        truth_mode=(
                            TruthMode.ACTOR_OBSERVER
                            if actor
                            else TruthMode.SEAT_EXPERIENCE
                        ),
                        projection_hash=(record.decision_id + observer.name).encode().hex().ljust(64, "0")[:64],
                        forced=False,
                        truth_label=(
                            TruthLabel.ROBUST_BEST_KNOWN
                            if actor
                            else TruthLabel.UNRESOLVED
                        ),
                        confidence=0.9 if actor else 0.0,
                        evidence=(
                            {"outcomeImpact": 90.0, "confidence": 90.0}
                            if actor
                            else {"threatVisibility": 40.0}
                        ),
                        recommended_action=(
                            {"kind": "play", "card": "AS"} if actor else None
                        ),
                        prompt_contract=(
                            "choose_acting_action" if actor else "identify_threat"
                        ),
                    )
                )
            if record.ordinal == 0:
                perspectives.append(
                    DecisionPerspectiveEvidenceV3(
                        decision_id=record.decision_id,
                        acting_seat=record.acting_seat,
                        observer_seat=record.acting_seat,
                        truth_mode=TruthMode.ACTUAL_DEAL_ORACLE,
                        projection_hash=(record.decision_id + "oracle").encode().hex().ljust(64, "0")[:64],
                        forced=False,
                        truth_label=TruthLabel.EXACT_OPTIMAL,
                        confidence=1.0,
                        evidence={"outcomeImpact": 100.0},
                        recommended_action=None,
                        prompt_contract="post_reveal_analysis",
                    )
                )
        laffa = LaffaEvidenceV2.build(
            self.trace.decision_ledger,
            perspectives,
            critical_threshold=70.0,
            minimum_critical_confidence=0.75,
        )
        analysis = {
            "schemaVersion": 2,
            "traceHash": self.trace.sha256,
            "decisions": [
                {
                    "decisionId": item.decision_id,
                    "actor": item.acting_seat.name.lower(),
                }
                for item in self.trace.decision_ledger.records
            ],
            "sha256": "1" * 64,
        }
        analysis_reference = _put(
            self.store, "trace_analysis_v1", analysis["sha256"], analysis
        )
        laffa_reference = _put(
            self.store, "verified_laffa_evidence_v2", laffa.sha256, laffa.to_dict()
        )
        selected = SimpleNamespace(trace_hash=self.trace.sha256, sha256="2" * 64)
        selected_items = (selected,) if self.retain_candidate else ()
        return SimpleNamespace(
            sha256="f" * 64,
            analysis_profile_hash=(
                self.result_profile_hash or self.analyzer.profile.sha256
            ),
            selection_profile_hash="4" * 64,
            diversity_profile_hash="5" * 64,
            trace_hashes=(
                self.result_trace_hashes
                if self.result_trace_hashes is not None
                else (self.trace.sha256,)
            ),
            analysis_hashes=(
                self.result_analysis_hashes
                if self.result_analysis_hashes is not None
                else (analysis["sha256"],)
            ),
            selection=SimpleNamespace(selected=selected_items),
            candidate_hashes=tuple(item.sha256 for item in selected_items),
            surrounding_competence_evidence_hashes=("6" * 64,),
            artifact_references=(analysis_reference, laffa_reference),
            to_dict=lambda: {
                "schemaVersion": 2,
                "traceHashes": [self.trace.sha256],
                "candidateHashes": [item.sha256 for item in selected_items],
                "sha256": "f" * 64,
            },
        )


class _AuthorityOnlyFunnel:
    analyzer = SimpleNamespace(profile=SimpleNamespace(sha256="3" * 64))

    def process(self, *_args, **_kwargs):
        raise AssertionError("authority-only funnel must not execute")


def _wait(service, job_id: str):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        item = service.get(job_id)
        if item and item["status"] in {"completed", "failed", "cancelled"}:
            return item
        time.sleep(0.01)
    raise AssertionError("finalist validation did not finish")


class _DeferredExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, callback, *args):
        self.tasks.append((callback, args))


def test_finalist_recovers_after_restart_and_terminalizes_inactive_cancel(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    replay = _Replay()
    parent = _Trace("c" * 64)
    candidates = _Candidates(parent)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    replay.descendant = descendant
    funnel = _Funnel(descendant, artifacts)
    common = {
        "jobs": jobs,
        "replay": replay,
        "profile": _profile(),
        "candidates": candidates,
        "funnel": funnel,
        "artifact_store": artifacts,
        "maximum_attempts": 2,
        "thrill_profile": _thrill_profile(),
    }
    original = FinalistValidationJobServiceV1(
        background_executor=_DeferredExecutor(), **common
    )
    interrupted = original.submit(
        analysis_job_id="analysis-restart",
        candidate_hashes=("a" * 64,),
        idempotency_key="restart-finalist",
    )
    jobs.update(interrupted["jobId"], status="running", progress=0.2)

    executor = ThreadPoolExecutor(max_workers=1)
    restarted = FinalistValidationJobServiceV1(
        background_executor=executor, **common
    )
    try:
        assert restarted.recover_pending() == (interrupted["jobId"],)
        assert _wait(restarted, interrupted["jobId"])["status"] == "completed"

        stale = original.submit(
            analysis_job_id="analysis-restart",
            candidate_hashes=("b" * 64,),
            idempotency_key="cancel-finalist",
        )
        jobs.update(stale["jobId"], status="running", progress=0.2)
        assert restarted.cancel(stale["jobId"])
        terminal = restarted.get(stale["jobId"])
        assert terminal["status"] == "cancelled"
        assert terminal["progress"] == 1.0
    finally:
        executor.shutdown(wait=True)


def test_finalist_job_is_idempotent_deduplicates_laffa_and_stays_publication_locked(
    tmp_path,
):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    replay = _Replay()
    parent = _Trace("c" * 64)
    candidates = _Candidates(parent)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    replay.descendant = descendant
    funnel = _Funnel(descendant, artifacts)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=replay,
        profile=_profile(),
        candidates=candidates,
        funnel=funnel,
        artifact_store=artifacts,
        maximum_attempts=2,
        thrill_profile=_thrill_profile(),
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-1",
            candidate_hashes=("a" * 64, "b" * 64),
            idempotency_key="finalist-proof-1",
        )
        completed = _wait(service, submitted["jobId"])
        repeated = service.submit(
            analysis_job_id="analysis-1",
            candidate_hashes=("a" * 64, "b" * 64),
            idempotency_key="finalist-proof-1",
        )

        assert repeated["jobId"] == submitted["jobId"]
        assert completed["kind"] == FINALIST_VALIDATION_JOB_KIND
        assert completed["status"] == "completed"
        assert replay.calls == 1
        assert funnel.calls == 1
        assert completed["result"]["validatedLaffaCount"] == 1
        assert completed["result"]["candidateCount"] == 2
        assert completed["result"]["localPreviewReady"] is True
        assert completed["result"]["publicationReady"] is False
        assert completed["request"]["finalistAnalysisProfileHash"] == "3" * 64
        assert [item["jobId"] for item in service.list(limit=10)] == [
            submitted["jobId"]
        ]
        assert service.list(status="failed", limit=10) == []
        assert len(completed["result"]["items"]) == 2
        analysis = completed["result"]["items"][0]["analysisEvidence"]
        assert analysis["allDecisionsAnalyzed"] is True
        assert analysis["fourObserverProjections"] is True
        assert analysis["allSeatsAnalyzed"] is True
        assert analysis["decisionCount"] == 4
        assert analysis["perspectiveCount"] == 17
        assert analysis["laffaSelectionScore"]["scoreFamily"] == (
            "laffa_content_selection"
        )
        assert analysis["laffaSelectionScore"]["passed"] is True
        assert completed["result"]["items"][0][
            "laffaSelectionScoreArtifact"
        ]["ownerKind"] == "laffa_content_selection_v1"
        manifest = completed["result"]["manifestArtifact"]
        assert artifacts.get(manifest["uri"])
        manifest_document = json.loads(artifacts.get(manifest["uri"]))
        assert manifest_document["finalistAnalysisProfileHash"] == "3" * 64

        with pytest.raises(ValueError, match="different finalist validation"):
            service.submit(
                analysis_job_id="analysis-1",
                candidate_hashes=("a" * 64,),
                idempotency_key="finalist-proof-1",
            )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("funnel_options", "message"),
    (
        (
            {"result_profile_hash": "0" * 64},
            "analyzer profile changed after submission",
        ),
        (
            {"result_trace_hashes": ("0" * 64,)},
            "trace membership is inconsistent",
        ),
        (
            {"result_analysis_hashes": ("0" * 64,)},
            "analysis membership is inconsistent",
        ),
    ),
)
def test_finalist_execution_rejects_profile_or_membership_substitution(
    tmp_path,
    funnel_options,
    message,
):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_Replay(),
        profile=_profile(),
        candidates=_Candidates(parent),
        funnel=_Funnel(descendant, artifacts, **funnel_options),
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-authority",
            candidate_hashes=("a" * 64,),
            idempotency_key="authority-substitution",
        )

        failed = _wait(service, submitted["jobId"])

        assert failed["status"] == "failed"
        assert message in failed["error"]
    finally:
        executor.shutdown(wait=True)


def test_finalist_request_profile_authority_is_pinned_on_resume_and_recovery(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    executor = _DeferredExecutor()
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_Replay(),
        profile=_profile(),
        candidates=_Candidates(parent),
        funnel=_Funnel(descendant, artifacts),
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
    )

    resume_job = service.submit(
        analysis_job_id="analysis-resume-authority",
        candidate_hashes=("a" * 64,),
        idempotency_key="resume-authority",
    )
    resume_request = dict(resume_job["request"])
    assert resume_request["finalistAnalysisProfileHash"] == "3" * 64
    resume_request["finalistAnalysisProfileHash"] = "0" * 64
    with jobs._connect() as connection:
        connection.execute(
            "UPDATE jobs SET request_json=? WHERE id=?",
            (json.dumps(resume_request), resume_job["jobId"]),
        )
    with pytest.raises(ValueError, match="frozen runtime authority"):
        service.resume(resume_job["jobId"])
    jobs.update(resume_job["jobId"], status="failed", progress=1.0)

    recovery_job = service.submit(
        analysis_job_id="analysis-recovery-authority",
        candidate_hashes=("b" * 64,),
        idempotency_key="recovery-authority",
    )
    recovery_request = dict(recovery_job["request"])
    recovery_request["finalistAnalysisProfileHash"] = "0" * 64
    with jobs._connect() as connection:
        connection.execute(
            "UPDATE jobs SET request_json=? WHERE id=?",
            (json.dumps(recovery_request), recovery_job["jobId"]),
        )

    assert service.recover_pending() == (recovery_job["jobId"],)
    recovered = service.get(recovery_job["jobId"])
    assert recovered["status"] == "failed"
    assert "frozen runtime authority" in recovered["error"]


def test_finalist_laffa_score_blocks_preview_without_changing_decision_selection(
    tmp_path,
):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_Replay(),
        profile=_profile(),
        candidates=_Candidates(parent),
        funnel=_Funnel(descendant, artifacts),
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(minimum_score=95.0),
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-laffa-gate",
            candidate_hashes=("a" * 64,),
            idempotency_key="laffa-thrill-gate",
        )
        completed = _wait(service, submitted["jobId"])
        item = completed["result"]["items"][0]

        assert completed["status"] == "completed"
        assert item["analysisEvidence"]["qualifiedCandidateCount"] == 1
        assert item["analysisEvidence"]["selectedCandidateHashes"] == ["2" * 64]
        assert item["analysisEvidence"]["laffaSelectionScore"]["passed"] is False
        assert item["analysisEvidence"]["challengeReady"] is False
        assert "laffa_selection_failed" in item["blockers"]
        assert item["localPreviewReady"] is False
        assert artifacts.get(item["laffaSelectionScoreArtifact"]["uri"])
    finally:
        executor.shutdown(wait=True)


def test_finalist_job_selectively_resimulates_before_expensive_replay(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    replay = _Replay()
    resimulation = _CandidateResimulation()
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent="8" * 64)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=replay,
        profile=_profile(),
        candidates=_Candidates(parent),
        funnel=_Funnel(descendant, artifacts),
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
        candidate_resimulation=resimulation,
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-resimulation",
            candidate_hashes=("a" * 64,),
            idempotency_key="candidate-resimulation-proof",
            candidate_resimulation=True,
        )
        completed = _wait(service, submitted["jobId"])
        item = completed["result"]["items"][0]

        assert completed["status"] == "completed"
        assert resimulation.calls == 1
        assert replay.parents == ["8" * 64]
        assert item["sourceTraceHash"] == "c" * 64
        assert item["parentTraceHash"] == "8" * 64
        assert item["candidateResimulationTraceHash"] == "8" * 64
        assert item["descendantTraceHash"] == "d" * 64
        assert item["candidateResimulation"]["status"] == "completed"
        assert item["candidateResimulationTraceArtifact"]["ownerHash"] == "8" * 64
        assert (
            item["candidateResimulationEvidenceArtifact"]["ownerHash"]
            == "9" * 64
        )
    finally:
        executor.shutdown(wait=True)


def test_finalist_job_rejects_unbounded_or_invalid_candidate_selection(tmp_path):
    service = FinalistValidationJobServiceV1(
        jobs=JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=SimpleNamespace(submit=lambda *args: None),
        replay=_Replay(),
        profile=_profile(maximum=1),
        candidates=_Candidates(_Trace("c" * 64)),
        funnel=None,
        artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
    )

    with pytest.raises(ValueError, match="count is out of range"):
        service.submit(
            analysis_job_id="analysis-1",
            candidate_hashes=("a" * 64, "b" * 64),
            idempotency_key="too-many",
        )
    with pytest.raises(ValueError, match="unique SHA-256"):
        service.submit(
            analysis_job_id="analysis-1",
            candidate_hashes=("not-a-hash",),
            idempotency_key="invalid",
        )


def test_finalist_job_blocks_preview_when_no_candidate_survives_reanalysis(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_Replay(),
        profile=_profile(),
        candidates=_Candidates(parent),
        funnel=_Funnel(descendant, artifacts, retain_candidate=False),
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-1",
            candidate_hashes=("a" * 64,),
            idempotency_key="no-survivor",
        )
        completed = _wait(service, submitted["jobId"])
        item = completed["result"]["items"][0]

        assert completed["status"] == "completed"
        assert completed["result"]["localPreviewReady"] is False
        assert item["analysisEvidence"]["challengeReady"] is False
        assert "no_finalist_candidate_survived" in item["blockers"]
    finally:
        executor.shutdown(wait=True)


def test_finalist_job_short_circuits_only_after_exact_tactical_impossibility(
    tmp_path,
):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    funnel = _Funnel(descendant, artifacts)
    probe = _TacticalViabilityProbe(can_reject=True)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_Replay(),
        profile=_profile(tactical_precheck=True),
        candidates=_Candidates(parent),
        funnel=funnel,
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
        tactical_viability_probe=probe,
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-no-tactical-play",
            candidate_hashes=("a" * 64,),
            idempotency_key="exact-tactical-impossibility",
            required_decision_kind="play",
        )
        completed = _wait(service, submitted["jobId"])
        item = completed["result"]["items"][0]

        assert completed["status"] == "completed"
        assert completed["result"]["localPreviewReady"] is False
        assert probe.calls == 1
        assert funnel.calls == 0
        assert item["analysisEvidence"] is None
        assert "no_verified_tactical_play_opportunity" in item["blockers"]
        assert item["tacticalViabilityProbe"]["scanComplete"] is True
        assert artifacts.get(item["tacticalViabilityArtifact"]["uri"])
        metrics = completed["result"]["operationalMetrics"]
        assert metrics["stageWallSeconds"]["tacticalViabilityProbe"] >= 0.0
        assert "finalistFunnel" not in metrics["stageWallSeconds"]
    finally:
        executor.shutdown(wait=True)


def test_finalist_job_keeps_full_analysis_for_positive_tactical_probe(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    executor = ThreadPoolExecutor(max_workers=1)
    parent = _Trace("c" * 64)
    descendant = _Trace("d" * 64, parent=parent.sha256)
    funnel = _Funnel(descendant, artifacts)
    probe = _TacticalViabilityProbe(can_reject=False)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_Replay(),
        profile=_profile(tactical_precheck=True),
        candidates=_Candidates(parent),
        funnel=funnel,
        artifact_store=artifacts,
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
        tactical_viability_probe=probe,
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-tactical-play",
            candidate_hashes=("a" * 64,),
            idempotency_key="positive-tactical-viability",
            required_decision_kind="play",
        )
        completed = _wait(service, submitted["jobId"])

        assert completed["status"] == "completed"
        assert completed["result"]["localPreviewReady"] is True
        assert probe.calls == 1
        assert funnel.calls == 1
        assert completed["result"]["items"][0]["analysisEvidence"] is not None
        assert (
            completed["result"]["operationalMetrics"]
            ["stageWallSeconds"]["finalistFunnel"]
            >= 0.0
        )
    finally:
        executor.shutdown(wait=True)


def test_finalist_job_can_be_cancelled_before_execution(tmp_path):
    service = FinalistValidationJobServiceV1(
        jobs=JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=SimpleNamespace(submit=lambda *args: None),
        replay=_Replay(),
        profile=_profile(),
        candidates=_Candidates(_Trace("c" * 64)),
        funnel=_AuthorityOnlyFunnel(),
        artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
    )
    queued = service.submit(
        analysis_job_id="analysis-1",
        candidate_hashes=("a" * 64,),
        idempotency_key="cancel-me",
    )

    assert service.cancel(queued["jobId"])
    assert service.get(queued["jobId"])["status"] == "cancelled"


def test_finalist_job_enforces_governed_trace_timeout(tmp_path, monkeypatch):
    ticks = iter((0.0, 31.0))
    monkeypatch.setattr(
        finalist_service_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    executor = ThreadPoolExecutor(max_workers=1)
    service = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=_TimeoutReplay(),
        profile=_profile(maximum_trace_seconds=30),
        candidates=_Candidates(_Trace("c" * 64)),
        funnel=_AuthorityOnlyFunnel(),
        artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        maximum_attempts=1,
        thrill_profile=_thrill_profile(),
    )
    try:
        submitted = service.submit(
            analysis_job_id="analysis-timeout",
            candidate_hashes=("a" * 64,),
            idempotency_key="timeout-proof",
        )
        completed = _wait(service, submitted["jobId"])
        assert completed["status"] == "failed"
        assert completed["error"] == (
            "finalist validation exceeded the governed 30-second trace limit"
        )
    finally:
        executor.shutdown(wait=True)
