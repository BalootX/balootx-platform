"""Downstream engine compatibility quarantine for CP-qualified Research.

This module is deliberately outside the OR-Tools Research backend.  It replays
an immutable exact-state document through the authoritative runtime engine only
to decide whether the state can be admitted to interactive/player or normal
video preparation.  Its receipt and evidence never participate in, or mutate,
the CP-SAT qualification identity.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from baloot_platform.core.rules import RuleBook
from baloot_platform.core.serialization import replay_trusted_exact_state_context

from .result_authority import RuntimeCompatibilityV1
from .scenario_contracts import canonical_sha256


RUNTIME_COMPATIBILITY_EVIDENCE_KIND = "runtime_compatibility_evidence_v1"
_HASH = re.compile(r"[0-9a-f]{64}")

# These values describe how the original round was produced, but
# ``replay_trusted_context`` cannot derive them from the resulting hands and
# legal play trace.  They remain covered by the immutable input-document hash;
# all replay-derived game fields continue to be compared below.
_NON_REPLAYABLE_PROVENANCE_FIELDS = frozenset(
    {
        "biddingHistory",
        "dealTrace",
        "events",
        "purchaseCard",
        "shuffle",
        "winningBidder",
    }
)


class RuntimeCompatibilityAdmissionError(ValueError):
    """Raised when a non-passed receipt is used for playable/render admission."""


def _require_hash(value: Any, label: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _state_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical portion that authoritative replay can derive."""

    return {
        key: copy.deepcopy(value)
        for key, value in sorted(state.items())
        if key not in _NON_REPLAYABLE_PROVENANCE_FIELDS
    }


def _trace_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical legal play/trick trace derived by both authorities."""

    return {
        "schemaVersion": 1,
        "completedTricks": copy.deepcopy(list(state.get("completedTricks") or [])),
        "currentTrick": copy.deepcopy(state.get("currentTrick")),
    }


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityEvidenceV1:
    """Immutable evidence behind one downstream compatibility receipt."""

    _canonical_json: str

    @classmethod
    def build(cls, core: Mapping[str, Any]) -> "RuntimeCompatibilityEvidenceV1":
        payload = copy.deepcopy(dict(core))
        payload["sha256"] = canonical_sha256(payload)
        return cls(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    @property
    def sha256(self) -> str:
        return str(json.loads(self._canonical_json)["sha256"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityCheckV1:
    """One immutable receipt plus its owned evidence artifact."""

    receipt: RuntimeCompatibilityV1
    evidence: RuntimeCompatibilityEvidenceV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtimeCompatibility": self.receipt.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


def quarantine_runtime_compatibility_failure_v1(
    *,
    semantic_result_hash: str,
    bundle_hash: str,
    trace_hash: str,
    failure_code: str,
    failure: Exception,
    engine_version: str = "compatibility-boundary-v1",
) -> RuntimeCompatibilityCheckV1:
    """Turn a downstream compatibility failure into owned quarantine evidence."""

    semantic_hash = _require_hash(semantic_result_hash, "semantic result hash")
    bundle_digest = _require_hash(bundle_hash, "bundle hash")
    trace_digest = _require_hash(trace_hash, "trace hash")
    runtime_version = _require_text(engine_version, "engine version")
    code = _require_text(failure_code, "runtime compatibility failure code")
    failure_hash = canonical_sha256(
        {"exceptionType": type(failure).__name__, "failureCode": code}
    )
    evidence = RuntimeCompatibilityEvidenceV1.build(
        {
            "schemaVersion": 1,
            "kind": RUNTIME_COMPATIBILITY_EVIDENCE_KIND,
            "semanticResultHash": semantic_hash,
            "bundleHash": bundle_digest,
            "traceHash": trace_digest,
            "engineVersion": runtime_version,
            "replayStatus": "quarantined",
            "replayFailureHash": failure_hash,
            "failureCodes": [code],
        }
    )
    receipt = RuntimeCompatibilityV1.build(
        status="failed",
        bundle_hash=bundle_digest,
        trace_hash=trace_digest,
        engine_version=runtime_version,
        evidence_hash=evidence.sha256,
        failure_codes=(code,),
    )
    return RuntimeCompatibilityCheckV1(receipt, evidence)


def check_runtime_compatibility_v1(
    *,
    exact_state: Mapping[str, Any],
    semantic_result_hash: str,
    bundle_hash: str,
    trace_hash: str,
    rules: RuleBook,
    engine_version: str,
    match_score_before: Sequence[int] | None = None,
    project_actions: Sequence[Mapping[str, Any]] | None = None,
) -> RuntimeCompatibilityCheckV1:
    """Replay one CP-qualified exact state and emit quarantine evidence.

    ``semantic_result_hash`` is accepted only as immutable provenance.  The
    function has no access to the qualification or proof objects and therefore
    cannot rewrite or strengthen them.
    """

    semantic_hash = _require_hash(semantic_result_hash, "semantic result hash")
    bundle_digest = _require_hash(bundle_hash, "bundle hash")
    trace_digest = _require_hash(trace_hash, "trace hash")
    runtime_version = _require_text(engine_version, "engine version")
    try:
        if not isinstance(exact_state, Mapping):
            raise TypeError("exact state must be an immutable mapping document")
        state = exact_state.get("state")
        if not isinstance(state, Mapping):
            raise TypeError("exact state document must contain a state object")
        document = copy.deepcopy(dict(exact_state))
        qualified_state = copy.deepcopy(dict(state))
        document_hash_before = canonical_sha256(document)
        expected_state_hash = _require_hash(
            document.get("stateHash"), "exact state stateHash"
        )
        expected_rules_hash = _require_hash(
            document.get("rulesHash"), "exact state rulesHash"
        )
        expected_event_chain_hash = _require_hash(
            document.get("eventChainHash"), "exact state eventChainHash"
        )
    except Exception as exc:
        return quarantine_runtime_compatibility_failure_v1(
            semantic_result_hash=semantic_hash,
            bundle_hash=bundle_digest,
            trace_hash=trace_digest,
            engine_version=runtime_version,
            failure_code="exact_state_document_malformed",
            failure=exc,
        )
    expected_projection_hash = canonical_sha256(_state_projection(qualified_state))
    expected_trace_projection_hash = canonical_sha256(
        _trace_projection(qualified_state)
    )

    failure_codes: list[str] = []
    if canonical_sha256(qualified_state) != expected_state_hash:
        failure_codes.append("document_state_hash_mismatch")
    if (
        canonical_sha256(qualified_state.get("events") or [])
        != expected_event_chain_hash
    ):
        failure_codes.append("event_chain_hash_mismatch")
    if expected_rules_hash != rules.sha256:
        failure_codes.append("rules_hash_mismatch")

    replay_status = "skipped_invalid_input"
    derived_state_hash: str | None = None
    derived_projection_hash: str | None = None
    derived_trace_projection_hash: str | None = None
    replay_failure_hash: str | None = None
    if not failure_codes:
        try:
            replay_input = copy.deepcopy(qualified_state)
            if match_score_before is not None:
                if (
                    not isinstance(match_score_before, (list, tuple))
                    or len(match_score_before) != 2
                    or any(type(item) is not int for item in match_score_before)
                ):
                    raise TypeError("match score before must contain two integers")
                replay_input["matchScoreBefore"] = list(match_score_before)
            if project_actions is not None:
                if not isinstance(project_actions, (list, tuple)) or any(
                    not isinstance(item, Mapping) for item in project_actions
                ):
                    raise TypeError("project actions must contain mapping objects")
                replay_input["projectActions"] = [
                    copy.deepcopy(dict(item)) for item in project_actions
                ]
            replayed = replay_trusted_exact_state_context(replay_input, rules)
            derived_state = replayed.to_dict(reveal_hands=True)
            derived_state_hash = canonical_sha256(derived_state)
            derived_projection_hash = canonical_sha256(_state_projection(derived_state))
            derived_trace_projection_hash = canonical_sha256(
                _trace_projection(derived_state)
            )
            replay_status = "completed"
            if derived_projection_hash != expected_projection_hash:
                failure_codes.append("derived_state_mismatch")
            if derived_trace_projection_hash != expected_trace_projection_hash:
                failure_codes.append("derived_trace_mismatch")
        except Exception as exc:  # replay failures are quarantine evidence
            replay_status = "failed"
            failure_codes.append("engine_replay_failed")
            replay_failure_hash = canonical_sha256(
                {"exceptionType": type(exc).__name__, "message": str(exc)}
            )

    canonical_codes = sorted(set(failure_codes))
    # A downstream check must be observational: even a caller-provided mutable
    # mapping cannot be altered by replay.
    document_hash_after = canonical_sha256(exact_state)
    if document_hash_after != document_hash_before:
        canonical_codes = sorted(set(canonical_codes) | {"input_document_mutated"})

    evidence = RuntimeCompatibilityEvidenceV1.build(
        {
            "schemaVersion": 1,
            "kind": RUNTIME_COMPATIBILITY_EVIDENCE_KIND,
            "semanticResultHash": semantic_hash,
            "bundleHash": bundle_digest,
            "traceHash": trace_digest,
            "engineVersion": runtime_version,
            "rulesHash": rules.sha256,
            "exactStateId": document.get("exactStateId"),
            "exactStateDocumentHash": document_hash_before,
            "declaredStateHash": expected_state_hash,
            "derivedStateHash": derived_state_hash,
            "expectedStateProjectionHash": expected_projection_hash,
            "derivedStateProjectionHash": derived_projection_hash,
            "expectedTraceProjectionHash": expected_trace_projection_hash,
            "derivedTraceProjectionHash": derived_trace_projection_hash,
            "replayStatus": replay_status,
            "replayFailureHash": replay_failure_hash,
            "failureCodes": canonical_codes,
        }
    )
    status = "passed" if not canonical_codes else "failed"
    receipt = RuntimeCompatibilityV1.build(
        status=status,
        bundle_hash=bundle_digest,
        trace_hash=trace_digest,
        engine_version=runtime_version,
        evidence_hash=evidence.sha256,
        failure_codes=tuple(canonical_codes),
    )
    return RuntimeCompatibilityCheckV1(receipt, evidence)


def require_runtime_compatibility_passed_v1(
    receipt: RuntimeCompatibilityV1 | Mapping[str, Any],
) -> RuntimeCompatibilityV1:
    """Single admission guard shared by playable and all video preparation."""

    validated = (
        receipt
        if isinstance(receipt, RuntimeCompatibilityV1)
        else RuntimeCompatibilityV1.from_dict(receipt)
    )
    admission = runtime_compatibility_admission_v1(validated)
    if admission["playableEligible"] is not True:
        raise RuntimeCompatibilityAdmissionError(
            "runtime compatibility must pass before playable or video admission"
        )
    return validated


def runtime_compatibility_admission_v1(
    receipt: RuntimeCompatibilityV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the single player/video/audit admission decision for a receipt."""

    validated = (
        receipt
        if isinstance(receipt, RuntimeCompatibilityV1)
        else RuntimeCompatibilityV1.from_dict(receipt)
    ).to_dict()
    passed = validated["status"] == "passed"
    blocker = [] if passed else [f"runtime_compatibility_{validated['status']}"]
    core = {
        "schemaVersion": 1,
        "runtimeCompatibilityHash": validated["sha256"],
        "playableEligible": passed,
        "publishableVideoEligible": passed,
        "auditMediaEligible": passed,
        "auditLabelRequired": not passed,
        "blockers": blocker,
    }
    return {**core, "sha256": canonical_sha256(core)}


__all__ = [
    "RUNTIME_COMPATIBILITY_EVIDENCE_KIND",
    "RuntimeCompatibilityAdmissionError",
    "RuntimeCompatibilityCheckV1",
    "RuntimeCompatibilityEvidenceV1",
    "check_runtime_compatibility_v1",
    "quarantine_runtime_compatibility_failure_v1",
    "require_runtime_compatibility_passed_v1",
    "runtime_compatibility_admission_v1",
]
