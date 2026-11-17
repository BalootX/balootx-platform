"""Deterministic, team-neutral structural screening for organic simulations.

The screen is deliberately not a challenge recommendation. It only decides
which complete traces are worth retaining for the expensive observer-safe
analysis funnel. Every seed still contributes immutable trace identities and
a compact screening record to campaign evidence.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .evidence_v3 import canonical_sha256
from .trace import CompleteGameTraceV1


@dataclass(frozen=True, slots=True)
class OrganicScreeningProfileV1:
    profile_id: str
    revision: str
    agent_profile_id: str
    minimum_legal_actions: int
    base_retain_basis_points: int
    maximum_retain_basis_points: int
    elite_score: float
    elite_retain_basis_points: int
    rejected_audit_modulus: int
    retain_kabout: bool
    retain_escalated: bool
    compression_level: int
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "OrganicScreeningProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "agentProfileId", "minimumLegalActions",
            "baseRetainBasisPoints", "maximumRetainBasisPoints", "eliteScore",
            "eliteRetainBasisPoints", "rejectedAuditModulus", "retainKabout",
            "retainEscalated", "compressionLevel",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown organic screening profile keys: {sorted(unknown)}")
        core = {
            "schemaVersion": int(value.get("schemaVersion", 0)),
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "agentProfileId": str(value.get("agentProfileId") or ""),
            "minimumLegalActions": int(value.get("minimumLegalActions", 0)),
            "baseRetainBasisPoints": int(value.get("baseRetainBasisPoints", -1)),
            "maximumRetainBasisPoints": int(value.get("maximumRetainBasisPoints", -1)),
            "eliteScore": float(value.get("eliteScore", -1.0)),
            "eliteRetainBasisPoints": int(value.get("eliteRetainBasisPoints", -1)),
            "rejectedAuditModulus": int(value.get("rejectedAuditModulus", 0)),
            "retainKabout": bool(value.get("retainKabout", False)),
            "retainEscalated": bool(value.get("retainEscalated", False)),
            "compressionLevel": int(value.get("compressionLevel", 0)),
        }
        if core["schemaVersion"] != 1:
            raise ValueError("unsupported organic screening profile schema")
        if not core["id"] or not core["revision"] or not core["agentProfileId"]:
            raise ValueError("organic screening profile identity is required")
        if core["minimumLegalActions"] < 2:
            raise ValueError("organic screening requires at least two legal actions")
        rates = (
            core["baseRetainBasisPoints"],
            core["maximumRetainBasisPoints"],
            core["eliteRetainBasisPoints"],
        )
        if any(not 0 <= item <= 10_000 for item in rates):
            raise ValueError("organic screening retention basis points are invalid")
        if not rates[0] <= rates[1] <= rates[2]:
            raise ValueError("organic screening retention rates must be monotonic")
        if not 0.0 <= core["eliteScore"] <= 100.0:
            raise ValueError("organic screening elite score is invalid")
        if not 1 <= core["rejectedAuditModulus"] <= 1_000_000:
            raise ValueError("organic screening audit modulus is invalid")
        if not 1 <= core["compressionLevel"] <= 9:
            raise ValueError("organic screening compression level is invalid")
        return cls(
            core["id"], core["revision"], core["agentProfileId"],
            core["minimumLegalActions"],
            core["baseRetainBasisPoints"], core["maximumRetainBasisPoints"],
            core["eliteScore"], core["eliteRetainBasisPoints"],
            core["rejectedAuditModulus"], core["retainKabout"],
            core["retainEscalated"], core["compressionLevel"],
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "agentProfileId": self.agent_profile_id,
            "minimumLegalActions": self.minimum_legal_actions,
            "baseRetainBasisPoints": self.base_retain_basis_points,
            "maximumRetainBasisPoints": self.maximum_retain_basis_points,
            "eliteScore": self.elite_score,
            "eliteRetainBasisPoints": self.elite_retain_basis_points,
            "rejectedAuditModulus": self.rejected_audit_modulus,
            "retainKabout": self.retain_kabout,
            "retainEscalated": self.retain_escalated,
            "compressionLevel": self.compression_level,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class OrganicTraceScreenV1:
    root_seed: int
    attempt_trace_hashes: tuple[str, ...]
    final_trace_hash: str
    publishable_laffa: bool
    structural_score: float
    features: Mapping[str, Any]
    retain_basis_points: int
    retention_reason: str
    retained: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "rootSeed": self.root_seed,
            "attemptTraceHashes": list(self.attempt_trace_hashes),
            "finalTraceHash": self.final_trace_hash,
            "publishableLaffa": self.publishable_laffa,
            "structuralScore": self.structural_score,
            "features": dict(self.features),
            "retainBasisPoints": self.retain_basis_points,
            "retentionReason": self.retention_reason,
            "retained": self.retained,
            "sha256": self.sha256,
        }


def _selection_draw(profile_hash: str, trace_hash: str, purpose: str) -> int:
    return int(canonical_sha256({
        "schemaVersion": 1,
        "policy": "organic_screen_draw_v1",
        "profileHash": profile_hash,
        "traceHash": trace_hash,
        "purpose": purpose,
    })[:16], 16)


def screen_organic_trace_v1(
    *,
    root_seed: int,
    attempt_trace_hashes: Iterable[str],
    final_trace: CompleteGameTraceV1,
    publishable_laffa: bool,
    profile: OrganicScreeningProfileV1,
) -> OrganicTraceScreenV1:
    attempts = tuple(str(item) for item in attempt_trace_hashes)
    records = tuple(final_trace.decision_ledger.records)
    eligible = tuple(
        item for item in records
        if not item.forced and len(item.legal_actions) >= profile.minimum_legal_actions
    )
    play_eligible = tuple(item for item in eligible if item.decision_kind == "play")
    latter_half = play_eligible[len(play_eligible) // 2:]
    seats = sorted({item.acting_seat.name.lower() for item in play_eligible})
    maximum_branching = max((len(item.legal_actions) for item in eligible), default=0)
    state = dict(final_trace.trusted_final_state)
    projects = tuple(dict(item) for item in state.get("projects") or ())
    project_kinds = sorted({str(item.get("kind") or "unknown") for item in projects})
    round_score = dict(state.get("roundScore") or {})
    escalation = str(state.get("escalation") or "normal")
    kabout = bool(round_score.get("kabout", False))
    awarded = tuple(int(item) for item in round_score.get("awardedPoints") or (0, 0))
    shutout = len(awarded) == 2 and min(awarded) == 0
    score = min(100.0, round(
        min(25.0, len(play_eligible) * 1.25)
        + min(10.0, maximum_branching * 1.25)
        + (len(seats) / 4.0) * 10.0
        + min(15.0, len(latter_half) * 1.25)
        + min(10.0, len(project_kinds) * 5.0)
        + (20.0 if kabout else 0.0)
        + (10.0 if escalation != "normal" else 0.0)
        + (5.0 if shutout else 0.0),
        6,
    ))
    features = {
        "contract": str(state.get("contract") or "not_established"),
        "trumpSuit": str(state.get("trumpSuit") or "none"),
        "escalation": escalation,
        "kabout": kabout,
        "projectKinds": project_kinds,
        "eligibleDecisionCount": len(eligible),
        "eligiblePlayCount": len(play_eligible),
        "lateEligiblePlayCount": len(latter_half),
        "maximumBranching": maximum_branching,
        "actingSeatBreadth": len(seats),
        "awardedShutout": shutout,
    }
    if not publishable_laffa:
        basis_points = 0
        retained = False
        reason = "not_publishable"
    elif profile.retain_kabout and kabout:
        basis_points = 10_000
        retained = True
        reason = "kabout"
    elif profile.retain_escalated and escalation != "normal":
        basis_points = 10_000
        retained = True
        reason = "escalated"
    else:
        basis_points = round(
            profile.base_retain_basis_points
            + (profile.maximum_retain_basis_points - profile.base_retain_basis_points)
            * score / 100.0
        )
        if score >= profile.elite_score:
            basis_points = max(basis_points, profile.elite_retain_basis_points)
        retained = (
            _selection_draw(profile.sha256, final_trace.sha256, "priority") % 10_000
            < basis_points
        )
        reason = "structural_sample" if retained else "summary_only"
        if not retained and (
            _selection_draw(profile.sha256, final_trace.sha256, "audit")
            % profile.rejected_audit_modulus == 0
        ):
            retained = True
            reason = "rejected_audit_sample"
    core = {
        "schemaVersion": 1,
        "profileHash": profile.sha256,
        "rootSeed": int(root_seed),
        "attemptTraceHashes": list(attempts),
        "finalTraceHash": final_trace.sha256,
        "publishableLaffa": bool(publishable_laffa),
        "structuralScore": score,
        "features": features,
        "retainBasisPoints": basis_points,
        "retentionReason": reason,
        "retained": retained,
    }
    return OrganicTraceScreenV1(
        int(root_seed), attempts,
        final_trace.sha256, bool(publishable_laffa), score, features,
        int(basis_points), reason, retained, canonical_sha256(core),
    )


@dataclass(frozen=True, slots=True)
class OrganicShardScreeningManifestV1:
    shard_spec_hash: str
    profile_hash: str
    records: tuple[OrganicTraceScreenV1, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        shard_spec_hash: str,
        profile: OrganicScreeningProfileV1,
        records: Iterable[OrganicTraceScreenV1],
    ) -> "OrganicShardScreeningManifestV1":
        values = tuple(sorted(records, key=lambda item: item.root_seed))
        if not values:
            raise ValueError("screening manifest requires at least one seed record")
        if len({item.root_seed for item in values}) != len(values):
            raise ValueError("screening manifest contains duplicate root seeds")
        core = {
            "schemaVersion": 1,
            "shardSpecHash": str(shard_spec_hash),
            "profileHash": profile.sha256,
            "records": [item.to_dict() for item in values],
        }
        return cls(
            str(shard_spec_hash), profile.sha256, values, canonical_sha256(core)
        )

    def to_dict(self) -> dict[str, Any]:
        retained = tuple(item for item in self.records if item.retained)
        reasons: dict[str, int] = {}
        for item in self.records:
            reasons[item.retention_reason] = reasons.get(item.retention_reason, 0) + 1
        return {
            "schemaVersion": 1,
            "shardSpecHash": self.shard_spec_hash,
            "profileHash": self.profile_hash,
            "recordCount": len(self.records),
            "retainedCount": len(retained),
            "retentionReasons": dict(sorted(reasons.items())),
            "records": [item.to_dict() for item in self.records],
            "sha256": self.sha256,
        }

    def compressed_bytes(self, *, level: int) -> bytes:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return zlib.compress(encoded, level=int(level))

    @staticmethod
    def decode_compressed(data: bytes) -> Mapping[str, Any]:
        value = json.loads(zlib.decompress(data))
        if not isinstance(value, Mapping):
            raise ValueError("screening manifest payload must be an object")
        return value
