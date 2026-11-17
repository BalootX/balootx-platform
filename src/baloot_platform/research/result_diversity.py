"""Deterministic diversity selection for already-qualified Research results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1


def _first(values: Any, fallback: str = "none") -> str:
    normalized = sorted(
        {str(item).strip().lower() for item in values or () if str(item).strip()}
    )
    return normalized[0] if normalized else fallback


def _band(value: float, boundaries: tuple[float, float]) -> str:
    if value < boundaries[0]:
        return "low"
    if value < boundaries[1]:
        return "medium"
    return "high"


def _dimensions(result: Mapping[str, Any]) -> dict[str, str]:
    projection = dict(result.get("libraryProjection") or {})
    scenario = dict(result.get("scenarioProjection") or {})
    qualification = dict(result.get("qualification") or {})
    provenance = dict(result.get("provenance") or {})
    score = float(result.get("score", 0.0))
    acting_seats = projection.get("actingSeats") or result.get("interestingSeats") or ()
    observer_seats = projection.get("observerSeats") or ()
    escalation = str(projection.get("escalation") or "normal").lower()
    return {
        "decision_kind": str(scenario.get("decisionKind") or "laffa").lower(),
        "proof_mode": str(
            scenario.get("achievedProofMode")
            or scenario.get("requestedProofMode")
            or "unknown"
        ).lower(),
        "contract": str(projection.get("contract") or "unknown").lower(),
        "purchase_round": str(projection.get("purchaseRound") or "unknown").lower(),
        "trump_suit": str(projection.get("trumpSuit") or "none").lower(),
        "project": _first(projection.get("projectKinds")),
        "escalation": escalation,
        "tactical_detector": _first(result.get("verifiedTacticalTags")),
        "trick_number": "complete_laffa",
        "acting_seat": _first(acting_seats, "multiple"),
        "buyer_relationship": "multiple",
        "score_pressure": str(projection.get("scorePressure") or "unknown").lower(),
        "difficulty_band": _band(score, (40.0, 70.0)),
        "perspective_breadth": str(len(set(str(item) for item in observer_seats))),
        "source": str(
            projection.get("generationLane")
            or provenance.get("generationLane")
            or "unknown"
        ).lower(),
        "video_duration": "not_rendered",
        "surrounding_competence": _band(
            float(qualification.get("minimumSurroundingQuality", 0.0)),
            (50.0, 70.0),
        ),
        "candidate_scope": str(result.get("classification") or "laffa").lower(),
    }


def _rank(result: Mapping[str, Any]) -> tuple[float, float, str]:
    admission = dict(result.get("qualification") or {}).get("interestAdmission")
    return (
        -float(admission["score"] if isinstance(admission, Mapping) else result.get("score", 0.0)),
        -float(result.get("confidence", 0.0)),
        str(result.get("resultCode") or ""),
    )


@dataclass(frozen=True, slots=True)
class ResearchResultDiversitySelectionV1:
    profile_hash: str
    requested: int
    qualified_count: int
    selected_result_codes: tuple[str, ...]
    rejected: tuple[Mapping[str, str], ...]
    rejected_audit_sample: tuple[str, ...]
    cell_counts: tuple[tuple[str, int], ...]
    duplicate_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileHash": self.profile_hash,
            "requested": self.requested,
            "qualifiedCount": self.qualified_count,
            "selectedCount": len(self.selected_result_codes),
            "selectedResultCodes": list(self.selected_result_codes),
            "rejected": [dict(item) for item in self.rejected],
            "rejectedAuditSample": list(self.rejected_audit_sample),
            "cellCounts": dict(self.cell_counts),
            "duplicateCount": self.duplicate_count,
            "sha256": self.sha256,
        }


def select_research_results(
    results: Iterable[Mapping[str, Any]],
    *,
    profile: QualityDiversityProfileV1,
    requested: int,
) -> ResearchResultDiversitySelectionV1:
    if int(requested) < 1:
        raise ValueError(
            "research diversity selection requires a positive result count"
        )
    game_qualified = tuple(
        dict(item)
        for item in results
        if str(item.get("disposition")) == "qualified"
        and bool(dict(item.get("qualification") or {}).get("qualified"))
    )
    interest_rejected = [
        {"resultCode": str(item["resultCode"]), "reason": "interest_gate_rejected"}
        for item in game_qualified
        if "interestAdmission" in item["qualification"]
        and dict(item["qualification"].get("interestAdmission") or {}).get("passed") is not True
    ]
    rejected_codes = {item["resultCode"] for item in interest_rejected}
    qualified = tuple(item for item in game_qualified if item["resultCode"] not in rejected_codes)
    by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for item in qualified:
        fingerprint = str(
            item.get("gameHash")
            or item.get("traceHash")
            or item.get("resultHash")
            or ""
        )
        if len(fingerprint) != 64:
            raise ValueError(
                "qualified Research result is missing a content fingerprint"
            )
        by_fingerprint.setdefault(fingerprint, []).append(item)

    unique: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = interest_rejected
    for fingerprint in sorted(by_fingerprint):
        ranked = sorted(by_fingerprint[fingerprint], key=_rank)
        unique.append(ranked[0])
        rejected.extend(
            {
                "resultCode": str(item["resultCode"]),
                "reason": "duplicate_fingerprint",
            }
            for item in ranked[1:]
        )

    # Unknown lessons stay independent; retain every certified record in storage.
    seen_motifs: set[str] = set()
    motif_unique = []
    for item in sorted(unique, key=_rank):
        admission = dict(item.get("qualification") or {}).get("interestAdmission") or {}
        motif = admission.get("structuralMotifHash")
        if motif is not None:
            if not isinstance(motif, str) or len(motif) != 64 or any(c not in "0123456789abcdef" for c in motif):
                raise ValueError("qualified Research result has invalid structural motif hash")
            if motif in seen_motifs:
                rejected.append({"resultCode": str(item["resultCode"]), "reason": "duplicate_motif"})
                continue
            seen_motifs.add(motif)
        motif_unique.append(item)
    unique = motif_unique

    cells: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for item in unique:
        dimensions = _dimensions(item)
        cell = tuple((name, dimensions[name]) for name in profile.cell_dimensions)
        cells.setdefault(cell, []).append(item)
    for cell in cells:
        cells[cell].sort(key=_rank)

    selected: list[str] = []
    cell_counts: dict[str, int] = {}
    active_cells = sorted(cells)
    while active_cells and len(selected) < int(requested):
        next_cells: list[tuple[tuple[str, str], ...]] = []
        for cell in active_cells:
            values = cells[cell]
            cell_hash = canonical_sha256(dict(cell))
            kept = cell_counts.get(cell_hash, 0)
            if (
                values
                and kept < profile.top_k_per_cell
                and len(selected) < int(requested)
            ):
                selected.append(str(values.pop(0)["resultCode"]))
                cell_counts[cell_hash] = kept + 1
            if values and cell_counts.get(cell_hash, 0) < profile.top_k_per_cell:
                next_cells.append(cell)
        active_cells = next_cells

    selected_set = set(selected)
    already_rejected = {item["resultCode"] for item in rejected}
    for item in unique:
        code = str(item["resultCode"])
        if code not in selected_set and code not in already_rejected:
            rejected.append({"resultCode": code, "reason": "diversity_or_result_limit"})
    rejected.sort(key=lambda item: (item["reason"], item["resultCode"]))
    rejected_audit_sample = tuple(
        item["resultCode"]
        for item in rejected
        if int(canonical_sha256(item["resultCode"])[:16], 16)
        % profile.rejected_audit_modulus
        == 0
    )
    core = {
        "schemaVersion": 1,
        "profileHash": profile.sha256,
        "requested": int(requested),
        "qualifiedCount": len(qualified),
        "selectedResultCodes": selected,
        "rejected": rejected,
        "rejectedAuditSample": list(rejected_audit_sample),
        "cellCounts": cell_counts,
        "duplicateCount": len(qualified) - len(unique),
    }
    return ResearchResultDiversitySelectionV1(
        profile.sha256,
        int(requested),
        len(qualified),
        tuple(selected),
        tuple(rejected),
        rejected_audit_sample,
        tuple(sorted(cell_counts.items())),
        len(qualified) - len(unique),
        canonical_sha256(core),
    )
