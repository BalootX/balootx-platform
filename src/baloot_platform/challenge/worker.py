"""Fenced Foundry worker for organic deterministic simulation shards."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .campaign import SimulationCampaignSpecV1
from .campaign_control import ClaimedSimulationShardV1
from .evidence_v3 import canonical_sha256
from .executor import DeterministicBatchExecutorV1
from .disagreement_search import DisagreementSearchV1
from .guided_search import GuidedLegalSearchV1
from .screening import (
    OrganicScreeningProfileV1,
    OrganicShardScreeningManifestV1,
    screen_organic_trace_v1,
)
from .simulation import NaturalLaffaSimulatorV1
from .trace_import import TraceImportIngestorV1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class FoundryWorkerRunV1:
    worker_id: str
    status: str
    campaign_hash: str | None
    shard_hash: str | None
    result_hash: str | None
    trace_count: int
    artifact_count: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "workerId": self.worker_id,
            "status": self.status,
            "campaignHash": self.campaign_hash,
            "shardHash": self.shard_hash,
            "resultHash": self.result_hash,
            "traceCount": self.trace_count,
            "artifactCount": self.artifact_count,
            "error": self.error,
        }


class OrganicFoundryWorkerV1:
    """Claim, execute, persist, and fenced-commit one organic shard."""

    lane = "organic"

    def _lane_profile_hash(self) -> str:
        return (
            self.screening_profile.sha256
            if self.screening_profile is not None
            else self.simulator.bundle_hash
        )

    def __init__(
        self,
        campaign_store: Any,
        artifact_store: Any,
        simulator: NaturalLaffaSimulatorV1,
        *,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
        executor: DeterministicBatchExecutorV1 | None = None,
        screening_profile: OrganicScreeningProfileV1 | None = None,
        audit_trace_repository: Any | None = None,
    ):
        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("Foundry worker ID cannot be empty")
        if int(lease_seconds) < 3 or int(heartbeat_seconds) < 1:
            raise ValueError("Foundry worker lease and heartbeat must be positive")
        if int(heartbeat_seconds) * 3 > int(lease_seconds):
            raise ValueError("Foundry worker heartbeat must be no slower than one-third of its lease")
        self.campaign_store = campaign_store
        self.artifact_store = artifact_store
        self.simulator = simulator
        self.worker_id = worker
        self.lease_seconds = int(lease_seconds)
        self.heartbeat_seconds = int(heartbeat_seconds)
        self.executor = executor or DeterministicBatchExecutorV1()
        self.screening_profile = screening_profile
        if audit_trace_repository is None:
            # Keep the Research package import lazy: challenge.__init__ exports
            # this worker while the Research package also consumes challenge
            # contracts.
            from baloot_platform.research.audit_trace import (
                ResearchAuditTraceRepositoryV1,
            )

            audit_trace_repository = ResearchAuditTraceRepositoryV1(
                store=campaign_store,
                artifact_store=artifact_store,
            )
        self.audit_traces = audit_trace_repository

    def _campaign(self, claim: ClaimedSimulationShardV1) -> SimulationCampaignSpecV1:
        row = self.campaign_store.get_campaign(claim.spec.campaign_hash)
        if row is None:
            raise KeyError(claim.spec.campaign_hash)
        spec = SimulationCampaignSpecV1.from_dict(dict(row["spec"]))
        expected = {
            "rulesHash": self.simulator.rules.sha256,
            "bundleHash": self.simulator.bundle_hash,
            "engineVersion": self.simulator.engine_version,
            "agentRelease": self.simulator.agent_release,
            "seatAgents": list(self.simulator.equal_seat_agents),
        }
        actual = {
            "rulesHash": spec.rules_hash,
            "bundleHash": spec.bundle_hash,
            "engineVersion": spec.engine_version,
            "agentRelease": spec.agent_release,
            "seatAgents": list(spec.seat_agents),
        }
        if actual != expected:
            raise ValueError("claimed campaign does not match this worker's frozen runtime")
        if claim.spec.lane != self.lane or claim.spec.lane not in spec.lanes:
            raise ValueError(f"{self.lane} worker received an incompatible shard")
        if (
            claim.spec.lane_profile_hash != self._lane_profile_hash()
            or dict(spec.lane_profile_hashes).get(self.lane) != self._lane_profile_hash()
        ):
            raise ValueError(f"{self.lane} worker profile does not match its immutable shard")
        return spec

    def _persist_trace(self, trace) -> dict[str, Any]:
        reference = self.artifact_store.put(
            _canonical_bytes(trace.to_dict(trusted=True)),
            media_type="application/vnd.balootx.complete-trace-v1+json",
            name=f"{trace.sha256}.complete-trace.json",
        )
        return {
            "ownerKind": "complete_trace_v1",
            "ownerHash": trace.sha256,
            "uri": str(reference["uri"]),
            "sha256": str(reference["sha256"]),
            "bytes": int(reference["bytes"]),
            "mediaType": str(reference["mediaType"]),
            "name": reference.get("name"),
        }

    def _persist_simulation(self, simulation) -> dict[str, Any]:
        artifacts = [self._persist_trace(trace) for trace in simulation.attempt_traces]
        return {
            **simulation.to_batch_output(),
            "artifactReferences": artifacts,
        }

    @staticmethod
    def _audit_candidate_hash(
        claim: ClaimedSimulationShardV1,
        *,
        seed: int,
        candidate_ordinal: int,
    ) -> str:
        """Identify one logical campaign occurrence without binding trace bytes."""

        return canonical_sha256({
            "schemaVersion": 1,
            "kind": "organic_foundry_candidate_occurrence_v1",
            "campaignHash": claim.spec.campaign_hash,
            "shardHash": claim.spec.input_hash,
            "seed": int(seed),
            "candidateOrdinal": int(candidate_ordinal),
        })

    def _retain_audit_traces(
        self,
        claim: ClaimedSimulationShardV1,
        campaign: SimulationCampaignSpecV1,
        simulation: Any,
        *,
        seed: int,
    ) -> dict[str, Any]:
        profile = self.screening_profile
        research_configuration = {
            "schemaVersion": 1,
            "kind": "organic_foundry_campaign_v1",
            "campaign": campaign.to_dict(),
            "screeningProfile": None if profile is None else profile.to_dict(),
        }
        retained: dict[str, Any] = {}
        final_hash = simulation.final_trace.sha256
        for candidate_ordinal, trace in enumerate(simulation.attempt_traces):
            if trace.sha256 in retained:
                continue
            trace_role = (
                "scored_final"
                if simulation.publishable_laffa and trace.sha256 == final_hash
                else "all_pass_attempt"
            )
            source = {
                "kind": "organic_foundry_trace",
                "runCode": claim.spec.campaign_hash,
                "runHash": claim.spec.campaign_hash,
                "recipeCode": (
                    "organic-unscreened-v1"
                    if profile is None
                    else "organic-screening-v1"
                ),
                "recipeHash": claim.spec.lane_profile_hash,
                "shardId": claim.spec.input_hash,
                "shardHash": claim.spec.input_hash,
                "shardIndex": claim.spec.shard_index,
                "seed": int(seed),
                "candidateHash": self._audit_candidate_hash(
                    claim,
                    seed=seed,
                    candidate_ordinal=candidate_ordinal,
                ),
                "candidateOrdinal": candidate_ordinal,
                "generationLane": claim.spec.lane,
                "attemptCount": claim.attempt,
                "fencingToken": claim.fencing_token,
                "workerId": claim.worker_id,
                "traceRole": trace_role,
            }
            record = self.audit_traces.retain(
                trace,
                rules=self.simulator.rules,
                source=source,
                research_configuration=research_configuration,
            )
            if trace_role == "all_pass_attempt":
                record = self.audit_traces.finalize(
                    record.record_hash,
                    disposition="rejected",
                    failure_categories=(
                        "stage_organic_simulation",
                        "organic_all_pass_complete_trace",
                    ),
                )
            retained[trace.sha256] = record
        return retained

    def _execute_screened_shard(
        self,
        claim: ClaimedSimulationShardV1,
        campaign: SimulationCampaignSpecV1,
        *,
        cancel_check,
    ):
        profile = self.screening_profile
        if profile is None:
            raise RuntimeError("screened shard execution requires a screening profile")
        processed: list[int] = []
        traces: list[str] = []
        candidates: list[str] = []
        coverage: list[str] = []
        artifacts: list[dict[str, Any]] = []
        screens = []
        status = "completed"
        for seed in range(claim.spec.seed_start, claim.spec.seed_end_exclusive):
            if cancel_check():
                status = "cancelled"
                break
            simulation = self.simulator.simulate(seed, cancel_check=cancel_check)
            audit_records = self._retain_audit_traces(
                claim,
                campaign,
                simulation,
                seed=seed,
            )
            screen = screen_organic_trace_v1(
                root_seed=seed,
                attempt_trace_hashes=(item.sha256 for item in simulation.attempt_traces),
                final_trace=simulation.final_trace,
                publishable_laffa=simulation.publishable_laffa,
                profile=profile,
            )
            processed.append(seed)
            traces.extend(screen.attempt_trace_hashes)
            coverage.extend(simulation.coverage_cells)
            screens.append(screen)
            if screen.retained:
                candidates.append(screen.final_trace_hash)
                artifacts.append(self._persist_trace(simulation.final_trace))
            elif simulation.publishable_laffa:
                record = audit_records[simulation.final_trace.sha256]
                self.audit_traces.finalize(
                    record.record_hash,
                    disposition="rejected",
                    failure_categories=(
                        "stage_organic_screening",
                        "organic_screening_nonselected",
                        f"organic_screening_{screen.retention_reason}",
                    ),
                )
        if screens:
            manifest = OrganicShardScreeningManifestV1.build(
                claim.spec.input_hash, profile, screens
            )
            stored = self.artifact_store.put(
                manifest.compressed_bytes(level=profile.compression_level),
                media_type=(
                    "application/vnd.balootx.organic-screening-manifest-v1+json+zlib"
                ),
                name=f"{manifest.sha256}.organic-screening.json.zlib",
            )
            artifacts.append({
                "ownerKind": "organic_screening_manifest_v1",
                "ownerHash": manifest.sha256,
                "uri": str(stored["uri"]),
                "sha256": str(stored["sha256"]),
                "bytes": int(stored["bytes"]),
                "mediaType": str(stored["mediaType"]),
                "name": stored.get("name"),
            })
        from .campaign import SimulationShardResultV1

        return SimulationShardResultV1.build(
            claim.spec,
            processed_seeds=processed,
            trace_hashes=traces,
            candidate_hashes=candidates,
            coverage_cells=coverage,
            artifact_references=artifacts,
            completion_status=status,
        )

    def _execute_seed(self, seed: int, *, cancel_check) -> dict[str, Any]:
        simulation = self.simulator.simulate(seed, cancel_check=cancel_check)
        return self._persist_simulation(simulation)

    def _execute_unscreened_organic_seed(
        self,
        claim: ClaimedSimulationShardV1,
        campaign: SimulationCampaignSpecV1,
        seed: int,
        *,
        cancel_check,
    ) -> dict[str, Any]:
        simulation = self.simulator.simulate(seed, cancel_check=cancel_check)
        self._retain_audit_traces(claim, campaign, simulation, seed=seed)
        return self._persist_simulation(simulation)

    def run_once(self, *, campaign_hash: str | None = None) -> FoundryWorkerRunV1:
        claim = self.campaign_store.claim_shard(
            self.worker_id,
            lease_seconds=self.lease_seconds,
            campaign_hash=campaign_hash,
            supported_lanes=(self.lane,),
        )
        if claim is None:
            return FoundryWorkerRunV1(
                self.worker_id, "idle", campaign_hash, None, None, 0, 0, None
            )
        return self.run_claim(claim)

    def run_claim(
        self,
        claim: ClaimedSimulationShardV1,
    ) -> FoundryWorkerRunV1:
        """Execute one already-fenced claim for controlled schedule proofs."""
        if claim.worker_id != self.worker_id:
            raise ValueError("Foundry worker cannot execute another worker's claim")
        next_heartbeat = time.time() + self.heartbeat_seconds
        artifact_count = 0

        def maintain_lease() -> bool:
            nonlocal next_heartbeat
            campaign = self.campaign_store.get_campaign(claim.spec.campaign_hash)
            if campaign is None or bool(campaign.get("cancelRequested")):
                return True
            current = time.time()
            if current >= next_heartbeat:
                accepted = self.campaign_store.heartbeat(
                    claim.spec.input_hash,
                    worker_id=self.worker_id,
                    fencing_token=claim.fencing_token,
                    lease_seconds=self.lease_seconds,
                    now=current,
                )
                if not accepted:
                    raise PermissionError("Foundry worker lost its fenced shard lease")
                next_heartbeat = current + self.heartbeat_seconds
            return False

        try:
            campaign = self._campaign(claim)
            if self.screening_profile is not None:
                result = self._execute_screened_shard(
                    claim,
                    campaign,
                    cancel_check=maintain_lease,
                )
                artifact_count = len(result.artifact_references)
            else:
                def simulate_seed(seed: int) -> dict[str, Any]:
                    nonlocal artifact_count
                    if self.lane == "organic":
                        output = self._execute_unscreened_organic_seed(
                            claim,
                            campaign,
                            seed,
                            cancel_check=maintain_lease,
                        )
                    else:
                        output = self._execute_seed(seed, cancel_check=maintain_lease)
                    artifact_count += len(output["artifactReferences"])
                    return output

                result = self.executor.execute_batch(
                    claim.spec,
                    simulate_seed=simulate_seed,
                    cancel_check=maintain_lease,
                )
            if result.completion_status != "completed":
                return FoundryWorkerRunV1(
                    self.worker_id,
                    result.completion_status,
                    claim.spec.campaign_hash,
                    claim.spec.input_hash,
                    result.sha256,
                    len(result.trace_hashes),
                    len(result.artifact_references),
                    None,
                )
            self.campaign_store.commit_result(
                result,
                worker_id=self.worker_id,
                fencing_token=claim.fencing_token,
            )
            return FoundryWorkerRunV1(
                self.worker_id,
                "completed",
                claim.spec.campaign_hash,
                claim.spec.input_hash,
                result.sha256,
                len(result.trace_hashes),
                len(result.artifact_references),
                None,
            )
        except Exception as exc:
            retryable = not isinstance(exc, (ValueError, KeyError, PermissionError))
            try:
                self.campaign_store.fail_shard(
                    claim.spec.input_hash,
                    worker_id=self.worker_id,
                    fencing_token=claim.fencing_token,
                    error=f"{exc.__class__.__name__}: {exc}",
                    retryable=retryable,
                )
            except PermissionError:
                pass
            return FoundryWorkerRunV1(
                self.worker_id,
                "failed",
                claim.spec.campaign_hash,
                claim.spec.input_hash,
                None,
                0,
                artifact_count,
                f"{exc.__class__.__name__}: {exc}",
            )


class GuidedFoundryWorkerV1(OrganicFoundryWorkerV1):
    """Fenced worker for deterministic guided legal-search shards."""

    lane = "guided"

    def _lane_profile_hash(self) -> str:
        return self.searcher.profile.sha256

    def __init__(
        self,
        campaign_store: Any,
        artifact_store: Any,
        searcher: GuidedLegalSearchV1,
        *,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
        executor: DeterministicBatchExecutorV1 | None = None,
    ):
        super().__init__(
            campaign_store,
            artifact_store,
            searcher.simulator,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            executor=executor,
        )
        self.searcher = searcher

    def _execute_seed(self, seed: int, *, cancel_check) -> dict[str, Any]:
        result = self.searcher.search(seed, cancel_check=cancel_check)
        output = self._persist_simulation(result.selected_simulation)
        manifest = self.artifact_store.put(
            _canonical_bytes(result.to_dict()),
            media_type="application/vnd.balootx.guided-search-v1+json",
            name=f"{result.sha256}.guided-search.json",
        )
        reference = {
            "ownerKind": "guided_search_v1",
            "ownerHash": result.sha256,
            "uri": str(manifest["uri"]),
            "sha256": str(manifest["sha256"]),
            "bytes": int(manifest["bytes"]),
            "mediaType": str(manifest["mediaType"]),
            "name": manifest.get("name"),
        }
        return {
            **output,
            "candidateHashes": (
                (result.selected_trace_hash,)
                if result.target_satisfied
                and result.selected_simulation.publishable_laffa
                else ()
            ),
            "coverageCells": (
                "source:guided",
                f"guided-profile:{result.profile_hash}",
                f"guided-target:{'satisfied' if result.target_satisfied else 'unsatisfied'}",
            ),
            "artifactReferences": [*output["artifactReferences"], reference],
        }


class DisagreementFoundryWorkerV1(OrganicFoundryWorkerV1):
    """Fenced worker for actor-safe shallow/deep disagreement shards."""

    lane = "disagreement"

    def _lane_profile_hash(self) -> str:
        return self.searcher.profile.sha256

    def __init__(
        self,
        campaign_store: Any,
        artifact_store: Any,
        searcher: DisagreementSearchV1,
        *,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
        executor: DeterministicBatchExecutorV1 | None = None,
    ):
        super().__init__(
            campaign_store,
            artifact_store,
            searcher.simulator,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            executor=executor,
        )
        self.searcher = searcher

    def _execute_seed(self, seed: int, *, cancel_check) -> dict[str, Any]:
        result = self.searcher.search(seed, cancel_check=cancel_check)
        output = self._persist_simulation(result.selected_simulation)
        manifest = self.artifact_store.put(
            _canonical_bytes(result.to_dict()),
            media_type="application/vnd.balootx.disagreement-search-v1+json",
            name=f"{result.sha256}.disagreement-search.json",
        )
        reference = {
            "ownerKind": "disagreement_search_v1",
            "ownerHash": result.sha256,
            "uri": str(manifest["uri"]),
            "sha256": str(manifest["sha256"]),
            "bytes": int(manifest["bytes"]),
            "mediaType": str(manifest["mediaType"]),
            "name": manifest.get("name"),
        }
        return {
            **output,
            "candidateHashes": (
                (result.selected_trace_hash,)
                if result.target_satisfied
                and result.selected_simulation.publishable_laffa
                else ()
            ),
            "coverageCells": (
                "source:disagreement",
                f"disagreement-profile:{result.profile_hash}",
                f"disagreement-target:{'satisfied' if result.target_satisfied else 'unsatisfied'}",
            ),
            "artifactReferences": [*output["artifactReferences"], reference],
        }


class ImportedTraceFoundryWorkerV1(OrganicFoundryWorkerV1):
    """Fenced base worker for a frozen external complete-trace catalog."""

    def __init__(
        self,
        campaign_store: Any,
        artifact_store: Any,
        ingestor: TraceImportIngestorV1,
        *,
        expected_agent_release: str,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
        executor: DeterministicBatchExecutorV1 | None = None,
    ):
        if ingestor.catalog.lane != self.lane:
            raise ValueError(f"{self.lane} worker requires a matching import catalog")
        release = str(expected_agent_release).strip()
        if not release:
            raise ValueError("import worker requires the campaign analysis agent release")
        super().__init__(
            campaign_store,
            artifact_store,
            ingestor,  # parent lifecycle is reused; execution is overridden below
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            executor=executor,
        )
        self.ingestor = ingestor
        self.expected_agent_release = release

    def _lane_profile_hash(self) -> str:
        return self.ingestor.catalog.sha256

    def _campaign(self, claim: ClaimedSimulationShardV1) -> SimulationCampaignSpecV1:
        row = self.campaign_store.get_campaign(claim.spec.campaign_hash)
        if row is None:
            raise KeyError(claim.spec.campaign_hash)
        spec = SimulationCampaignSpecV1.from_dict(dict(row["spec"]))
        if (
            spec.rules_hash != self.ingestor.rules.sha256
            or spec.bundle_hash != self.ingestor.bundle_hash
            or spec.engine_version != self.ingestor.engine_version
        ):
            raise ValueError("import campaign does not match the frozen trace runtime")
        if spec.agent_release != self.expected_agent_release:
            raise ValueError("import campaign analysis agent release is incompatible")
        if claim.spec.lane != self.lane or claim.spec.lane not in spec.lanes:
            raise ValueError(f"{self.lane} worker received an incompatible shard")
        if (
            claim.spec.lane_profile_hash != self._lane_profile_hash()
            or dict(spec.lane_profile_hashes).get(self.lane) != self._lane_profile_hash()
        ):
            raise ValueError(f"{self.lane} import catalog does not match its immutable shard")
        assigned = set(range(claim.spec.seed_start, claim.spec.seed_end_exclusive))
        available = set(range(len(self.ingestor.catalog.entries)))
        if not assigned <= available:
            raise ValueError("import shard references catalog ordinals that do not exist")
        return spec

    def _execute_seed(self, seed: int, *, cancel_check) -> dict[str, Any]:
        if cancel_check():
            raise InterruptedError("trace import cancelled")
        result = self.ingestor.ingest(seed)
        manifest = self.artifact_store.put(
            _canonical_bytes(result.to_dict()),
            media_type="application/vnd.balootx.trace-import-result-v1+json",
            name=f"{result.sha256}.trace-import.json",
        )
        manifest_reference = {
            "ownerKind": "trace_import_result_v1",
            "ownerHash": result.sha256,
            "uri": str(manifest["uri"]),
            "sha256": str(manifest["sha256"]),
            "bytes": int(manifest["bytes"]),
            "mediaType": str(manifest["mediaType"]),
            "name": manifest.get("name"),
        }
        return {
            "traceHash": result.trace_hash,
            "traceHashes": (result.trace_hash,),
            "candidateHashes": (),
            "coverageCells": result.coverage_cells,
            "artifactReferences": [dict(result.trace_reference), manifest_reference],
        }


class HistoricalFoundryWorkerV1(ImportedTraceFoundryWorkerV1):
    lane = "historical"


class ExpertFoundryWorkerV1(ImportedTraceFoundryWorkerV1):
    lane = "expert"
