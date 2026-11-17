"""Selective same-deal resimulation for retained Foundry candidates."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import canonical_sha256
from .simulation import NaturalLaffaSimulatorV1
from .trace import CompleteGameTraceV1


class CandidateResimulationTimeoutError(TimeoutError):
    """The bounded candidate-resimulation stage exhausted its own budget."""


@dataclass(frozen=True, slots=True)
class CandidateResimulationProfileV1:
    profile_id: str
    revision: str
    enabled: bool
    agent_profile_id: str
    maximum_trace_seconds: int
    require_exact_deck_commitment: bool
    require_parent_trace_link: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "CandidateResimulationProfileV1":
        allowed = {
            "schemaVersion",
            "id",
            "revision",
            "enabled",
            "agentProfileId",
            "maximumTraceSeconds",
            "requireExactDeckCommitment",
            "requireParentTraceLink",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown candidate resimulation profile keys: {sorted(unknown)}"
            )
        core = {
            "schemaVersion": int(value.get("schemaVersion", 0)),
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "enabled": value.get("enabled"),
            "agentProfileId": str(value.get("agentProfileId") or ""),
            "maximumTraceSeconds": int(value.get("maximumTraceSeconds", 0)),
            "requireExactDeckCommitment": value.get(
                "requireExactDeckCommitment"
            ),
            "requireParentTraceLink": value.get("requireParentTraceLink"),
        }
        if core["schemaVersion"] != 1 or not core["id"] or not core["revision"]:
            raise ValueError("candidate resimulation profile identity is invalid")
        if core["enabled"] is not True:
            raise ValueError("candidate resimulation must be explicitly enabled")
        if not core["agentProfileId"]:
            raise ValueError("candidate resimulation agent profile is required")
        if not 1 <= core["maximumTraceSeconds"] <= 300:
            raise ValueError("candidate resimulation timeout is out of range")
        if core["requireExactDeckCommitment"] is not True:
            raise ValueError("candidate resimulation must preserve the exact deal")
        if core["requireParentTraceLink"] is not True:
            raise ValueError("candidate resimulation must preserve parent lineage")
        return cls(
            core["id"],
            core["revision"],
            True,
            core["agentProfileId"],
            core["maximumTraceSeconds"],
            True,
            True,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "enabled": self.enabled,
            "agentProfileId": self.agent_profile_id,
            "maximumTraceSeconds": self.maximum_trace_seconds,
            "requireExactDeckCommitment": self.require_exact_deck_commitment,
            "requireParentTraceLink": self.require_parent_trace_link,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateResimulationEvidenceV1:
    profile_hash: str
    source_trace_hash: str
    descendant_trace_hash: str
    source_agent_release: str
    descendant_agent_release: str
    exact_deck_commitment_preserved: bool
    parent_trace_link_preserved: bool
    lineage_kind: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "stage": "candidate_resimulation",
            "status": "completed",
            "profileHash": self.profile_hash,
            "sourceTraceHash": self.source_trace_hash,
            "descendantTraceHash": self.descendant_trace_hash,
            "sourceAgentRelease": self.source_agent_release,
            "descendantAgentRelease": self.descendant_agent_release,
            "exactDeckCommitmentPreserved": (
                self.exact_deck_commitment_preserved
            ),
            "parentTraceLinkPreserved": self.parent_trace_link_preserved,
            "lineageKind": self.lineage_kind,
            "sha256": self.sha256,
        }


class CandidateResimulationV1:
    """Resimulate only selected traces with a stronger equal-seat policy."""

    descriptor = PluginDescriptor(
        id="candidate_same_deal_resimulation_v1",
        kind="generator",
        version="1.0.0",
        description=(
            "Deterministic same-deal resimulation of retained candidates before "
            "finalist validation"
        ),
        capabilities=(
            "retained_candidate_only",
            "exact_deal_commitment",
            "parent_descendant_lineage",
            "equal_agent_release",
            "observer_safe_policy",
            "deterministic",
            "cancellable",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=300,
        maximum_memory_mb=2048,
    )

    def __init__(
        self,
        simulator: NaturalLaffaSimulatorV1,
        profile: CandidateResimulationProfileV1,
    ) -> None:
        if simulator.agent_profile.id != profile.agent_profile_id:
            raise ValueError(
                "candidate resimulation profile does not match its agent"
            )
        if simulator.generation_tier != "balanced_discovery":
            raise ValueError(
                "candidate resimulation requires the governed balanced-discovery tier"
            )
        self.simulator = simulator
        self.profile = profile

    @staticmethod
    def _agent_release(trace: CompleteGameTraceV1) -> str:
        values = sorted({value for _, value in trace.agent_profiles})
        return values[0] if len(values) == 1 else "mixed"

    def resimulate(
        self,
        source: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[CompleteGameTraceV1, CandidateResimulationEvidenceV1]:
        deadline = time.monotonic() + self.profile.maximum_trace_seconds
        timed_out = False

        def bounded_cancel() -> bool:
            nonlocal timed_out
            if cancel_check is not None and cancel_check():
                return True
            if time.monotonic() >= deadline:
                timed_out = True
                return True
            return False

        try:
            descendant = self.simulator.resimulate_exact_deal_as_candidate_descendant(
                source,
                cancel_check=bounded_cancel,
            )
        except InterruptedError as exc:
            if timed_out:
                raise CandidateResimulationTimeoutError(
                    "candidate resimulation exceeded the governed "
                    f"{self.profile.maximum_trace_seconds}-second trace limit"
                ) from exc
            raise

        source_commitment = str(source.seed_provenance.get("deckCommitment") or "")
        descendant_commitment = str(
            descendant.seed_provenance.get("deckCommitment") or ""
        )
        generation = dict(descendant.trusted_final_state.get("generation") or {})
        exact_deal = bool(
            source_commitment
            and source_commitment == descendant_commitment
        )
        parent_link = str(generation.get("parentTraceHash") or "") == source.sha256
        lineage_kind = str(generation.get("kind") or "")
        if self.profile.require_exact_deck_commitment and not exact_deal:
            raise ValueError("candidate resimulation changed the exact deal")
        if self.profile.require_parent_trace_link and not parent_link:
            raise ValueError("candidate resimulation lost parent trace lineage")
        if lineage_kind != "candidate_resimulation_descendant":
            raise ValueError("candidate resimulation lineage kind is invalid")
        core = {
            "schemaVersion": 1,
            "profileHash": self.profile.sha256,
            "sourceTraceHash": source.sha256,
            "descendantTraceHash": descendant.sha256,
            "sourceAgentRelease": self._agent_release(source),
            "descendantAgentRelease": self.simulator.agent_release,
            "exactDeckCommitmentPreserved": exact_deal,
            "parentTraceLinkPreserved": parent_link,
            "lineageKind": lineage_kind,
        }
        evidence = CandidateResimulationEvidenceV1(
            self.profile.sha256,
            source.sha256,
            descendant.sha256,
            core["sourceAgentRelease"],
            self.simulator.agent_release,
            exact_deal,
            parent_link,
            lineage_kind,
            canonical_sha256(core),
        )
        return descendant, evidence
