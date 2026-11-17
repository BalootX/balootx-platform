"""Order-independent quality-diversity archive for verified candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import TruthLabel, canonical_sha256
from .profiles import ContentSelectionScoreV1


_CELL_DIMENSIONS = frozenset({
    "decision_kind",
    "proof_mode",
    "contract",
    "purchase_round",
    "trump_suit",
    "project",
    "escalation",
    "tactical_detector",
    "trick_number",
    "acting_seat",
    "buyer_relationship",
    "score_pressure",
    "difficulty_band",
    "perspective_breadth",
    "source",
    "video_duration",
    "surrounding_competence",
    "candidate_scope",
})


@dataclass(frozen=True, slots=True)
class QualityDiversityProfileV1:
    cell_dimensions: tuple[str, ...]
    top_k_per_cell: int
    rejected_audit_modulus: int
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "QualityDiversityProfileV1":
        allowed = {
            "schemaVersion", "cellDimensions", "topKPerCell",
            "rejectedAuditModulus",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown quality-diversity profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported quality-diversity profile schema")
        dimensions = tuple(str(item) for item in value.get("cellDimensions") or ())
        if not dimensions or len(set(dimensions)) != len(dimensions):
            raise ValueError("quality-diversity cell dimensions must be unique and non-empty")
        unsupported = set(dimensions) - _CELL_DIMENSIONS
        if unsupported:
            raise ValueError(f"unsupported quality-diversity dimensions: {sorted(unsupported)}")
        top_k = int(value.get("topKPerCell", 0))
        modulus = int(value.get("rejectedAuditModulus", 0))
        if not 1 <= top_k <= 1_000 or not 1 <= modulus <= 1_000_000:
            raise ValueError("quality-diversity bounds exceed governed limits")
        core = {
            "schemaVersion": 1,
            "cellDimensions": list(dimensions),
            "topKPerCell": top_k,
            "rejectedAuditModulus": modulus,
        }
        return cls(dimensions, top_k, modulus, canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "cellDimensions": list(self.cell_dimensions),
            "topKPerCell": self.top_k_per_cell,
            "rejectedAuditModulus": self.rejected_audit_modulus,
            "sha256": self.sha256,
        }

@dataclass(frozen=True, slots=True)
class QualityDiversityCandidateV1:
    trace_hash: str
    decision_id: str
    observer_seat: str
    truth_label: TruthLabel
    confidence: float
    structural_fingerprint: str
    dimensions: tuple[tuple[str, str], ...]
    selection_score: ContentSelectionScoreV1
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        trace_hash: str,
        decision_id: str,
        observer_seat: str,
        truth_label: TruthLabel | str,
        confidence: float,
        structural_fingerprint: str,
        dimensions: Mapping[str, Any],
        selection_score: ContentSelectionScoreV1,
    ) -> "QualityDiversityCandidateV1":
        if len(str(trace_hash)) != 64 or len(str(structural_fingerprint)) != 64:
            raise ValueError("quality-diversity candidates require content-addressed trace and fingerprint hashes")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("quality-diversity confidence must be between zero and one")
        label = truth_label if isinstance(truth_label, TruthLabel) else TruthLabel(str(truth_label))
        if label in {TruthLabel.UNRESOLVED, TruthLabel.REJECTED}:
            raise ValueError("unresolved or rejected evidence cannot enter the quality-diversity archive")
        if not selection_score.passed:
            raise ValueError("failed truth or quality gates cannot enter the quality-diversity archive")
        normalized = tuple(sorted((str(key), str(value)) for key, value in dimensions.items()))
        if not normalized or set(key for key, _ in normalized) - _CELL_DIMENSIONS:
            raise ValueError("candidate dimensions are empty or unsupported")
        core = {
            "schemaVersion": 1,
            "traceHash": str(trace_hash),
            "decisionId": str(decision_id),
            "observerSeat": str(observer_seat),
            "truthLabel": label.value,
            "confidence": float(confidence),
            "structuralFingerprint": str(structural_fingerprint),
            "dimensions": dict(normalized),
            "selectionScoreHash": selection_score.sha256,
        }
        return cls(
            str(trace_hash),
            str(decision_id),
            str(observer_seat),
            label,
            float(confidence),
            str(structural_fingerprint),
            normalized,
            selection_score,
            canonical_sha256(core),
        )

    def cell(self, profile: QualityDiversityProfileV1) -> tuple[tuple[str, str], ...]:
        values = dict(self.dimensions)
        missing = [key for key in profile.cell_dimensions if key not in values]
        if missing:
            raise ValueError(f"candidate is missing archive cell dimensions: {missing}")
        return tuple((key, values[key]) for key in profile.cell_dimensions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "traceHash": self.trace_hash,
            "decisionId": self.decision_id,
            "observerSeat": self.observer_seat,
            "truthLabel": self.truth_label.value,
            "confidence": self.confidence,
            "structuralFingerprint": self.structural_fingerprint,
            "dimensions": dict(self.dimensions),
            "selectionScore": self.selection_score.to_dict(),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualityDiversityCandidateV1":
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported quality-diversity candidate schema")
        selection = ContentSelectionScoreV1.from_dict(
            dict(value.get("selectionScore") or {})
        )
        candidate = cls.build(
            trace_hash=str(value.get("traceHash") or ""),
            decision_id=str(value.get("decisionId") or ""),
            observer_seat=str(value.get("observerSeat") or ""),
            truth_label=str(value.get("truthLabel") or ""),
            confidence=float(value.get("confidence", -1.0)),
            structural_fingerprint=str(value.get("structuralFingerprint") or ""),
            dimensions=dict(value.get("dimensions") or {}),
            selection_score=selection,
        )
        if str(value.get("sha256") or "") != candidate.sha256:
            raise ValueError("quality-diversity candidate checksum mismatch")
        return candidate


@dataclass(frozen=True, slots=True)
class QualityDiversitySelectionV1:
    profile_hash: str
    selected: tuple[QualityDiversityCandidateV1, ...]
    rejected: tuple[Mapping[str, str], ...]
    rejected_audit_sample: tuple[str, ...]
    cell_counts: tuple[tuple[str, int], ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileHash": self.profile_hash,
            "selected": [item.to_dict() for item in self.selected],
            "rejected": [dict(item) for item in self.rejected],
            "rejectedAuditSample": list(self.rejected_audit_sample),
            "cellCounts": dict(self.cell_counts),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualityDiversitySelectionV1":
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported quality-diversity selection schema")
        selected = tuple(
            QualityDiversityCandidateV1.from_dict(dict(item))
            for item in value.get("selected") or ()
        )
        if len({item.sha256 for item in selected}) != len(selected):
            raise ValueError("quality-diversity selection contains duplicate candidates")
        rejected = tuple(
            {
                "candidateHash": str(dict(item).get("candidateHash") or ""),
                "reason": str(dict(item).get("reason") or ""),
            }
            for item in value.get("rejected") or ()
        )
        audit_sample = tuple(
            str(item) for item in value.get("rejectedAuditSample") or ()
        )
        cell_counts = tuple(
            (str(key), int(count))
            for key, count in dict(value.get("cellCounts") or {}).items()
        )
        profile_hash = str(value.get("profileHash") or "")
        core = {
            "schemaVersion": 1,
            "profileHash": profile_hash,
            "selectedHashes": [item.sha256 for item in selected],
            "rejected": [dict(item) for item in rejected],
            "rejectedAuditSample": list(audit_sample),
            "cellCounts": dict(cell_counts),
        }
        digest = canonical_sha256(core)
        if str(value.get("sha256") or "") != digest:
            raise ValueError("quality-diversity selection checksum mismatch")
        return cls(
            profile_hash,
            selected,
            rejected,
            audit_sample,
            cell_counts,
            digest,
        )


class QualityDiversitySelectorV1:
    descriptor = PluginDescriptor(
        id="quality_diversity_selector_v1",
        kind="selection_policy",
        version="1.0.0",
        description="Deterministic global deduplication and bounded top-K selection per coverage cell",
        capabilities=(
            "order_independent",
            "global_fingerprint_deduplication",
            "top_k_per_cell",
            "rejected_audit_sample",
        ),
        deterministic=True,
        contract_id="diversity_policy",
        contract_version="1",
        safety_guards=(
            "observer_safe",
            "hard_gate_preservation",
            "frozen_provenance",
        ),
    )

    @staticmethod
    def _rank(candidate: QualityDiversityCandidateV1) -> tuple[float, float, str]:
        return (
            -candidate.selection_score.score,
            -candidate.confidence,
            candidate.sha256,
        )

    def select(
        self,
        candidates: Iterable[QualityDiversityCandidateV1],
        profile: QualityDiversityProfileV1,
        **_: Any,
    ) -> QualityDiversitySelectionV1:
        values = tuple(candidates)
        if len({item.sha256 for item in values}) != len(values):
            values = tuple({item.sha256: item for item in values}.values())
        by_fingerprint: dict[str, list[QualityDiversityCandidateV1]] = {}
        for item in values:
            item.cell(profile)
            by_fingerprint.setdefault(item.structural_fingerprint, []).append(item)

        unique: list[QualityDiversityCandidateV1] = []
        rejected: list[dict[str, str]] = []
        for fingerprint in sorted(by_fingerprint):
            ranked = sorted(by_fingerprint[fingerprint], key=self._rank)
            unique.append(ranked[0])
            rejected.extend(
                {"candidateHash": item.sha256, "reason": "duplicate_fingerprint"}
                for item in ranked[1:]
            )

        cells: dict[tuple[tuple[str, str], ...], list[QualityDiversityCandidateV1]] = {}
        for item in unique:
            cells.setdefault(item.cell(profile), []).append(item)
        selected: list[QualityDiversityCandidateV1] = []
        cell_counts: list[tuple[str, int]] = []
        for cell in sorted(cells):
            ranked = sorted(cells[cell], key=self._rank)
            kept = ranked[: profile.top_k_per_cell]
            selected.extend(kept)
            rejected.extend(
                {"candidateHash": item.sha256, "reason": "cell_top_k"}
                for item in ranked[profile.top_k_per_cell :]
            )
            cell_counts.append((canonical_sha256(dict(cell)), len(kept)))

        selected.sort(key=lambda item: (item.cell(profile), *self._rank(item)))
        rejected.sort(key=lambda item: (item["reason"], item["candidateHash"]))
        audit_sample = tuple(
            item["candidateHash"]
            for item in rejected
            if int(item["candidateHash"][:16], 16) % profile.rejected_audit_modulus == 0
        )
        core = {
            "schemaVersion": 1,
            "profileHash": profile.sha256,
            "selectedHashes": [item.sha256 for item in selected],
            "rejected": rejected,
            "rejectedAuditSample": list(audit_sample),
            "cellCounts": dict(cell_counts),
        }
        return QualityDiversitySelectionV1(
            profile.sha256,
            tuple(selected),
            tuple(rejected),
            audit_sample,
            tuple(cell_counts),
            canonical_sha256(core),
        )
