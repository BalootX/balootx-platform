"""Fail-closed status policy for governed challenge publication."""
from __future__ import annotations

import re
import hashlib
import json
from typing import Any, Mapping, Sequence

_MANUAL_TRANSITIONS = {
    "candidate": frozenset({"review", "rejected"}),
    "review": frozenset({"rejected"}),
    "approved": frozenset(),
    "published": frozenset(),
    "rejected": frozenset(),
    "retired": frozenset(),
}
_GOVERNED_ONLY = frozenset({"approved", "published", "retired"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_truth_contract(payload: Mapping[str, Any]) -> None:
    contract = payload.get("contentContract")
    if not isinstance(contract, Mapping):
        raise ValueError("challenge lacks a typed content contract")
    if not str(contract.get("contract", "")).strip():
        raise ValueError("challenge content contract type is required")
    if contract.get("accepted") is not True:
        raise ValueError("challenge content contract is not accepted")

    provenance = {
        "gameRecordHash": payload.get("gameRecordHash"),
        "gameSerial": payload.get("gameSerial"),
        "decisionId": payload.get("decisionId"),
    }
    if not any(str(value or "").strip() for value in provenance.values()):
        raise ValueError("challenge lacks source provenance")

    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("challenge lacks decision evidence")
    best = decision.get("best")
    if not isinstance(best, Mapping):
        raise ValueError("challenge lacks best-action evidence")
    confidence = best.get("matchWinConfidence95") or best.get("roundLeadConfidence95")
    if not (
        isinstance(confidence, Sequence)
        and not isinstance(confidence, (str, bytes))
        and len(confidence) == 2
        and 0.0 <= float(confidence[0]) <= float(confidence[1]) <= 1.0
    ):
        raise ValueError("challenge lacks a valid confidence interval")


def publication_policy_sha256(policy: Mapping[str, Any]) -> str:
    raw = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_manual_status_transition(current: str, target: str) -> None:
    """Reject transitions that bypass independent review and publication gates."""

    if current not in _MANUAL_TRANSITIONS or target not in _MANUAL_TRANSITIONS:
        raise ValueError("unsupported challenge status")
    if current == target:
        return
    if target in _GOVERNED_ONLY:
        raise ValueError(
            "approved, published, and retired require the governed independent-review "
            "workflow; the generic status endpoint is fail-closed"
        )
    if target not in _MANUAL_TRANSITIONS[current]:
        raise ValueError(
            f"challenge status transition {current!r} -> {target!r} is not allowed"
        )


def validate_evidence_sha256(value: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError("publication evidenceSha256 must be 64 lowercase hexadecimal characters")
    return digest


def validate_reviewer_role(role: str, policy: Mapping[str, Any]) -> str:
    normalized = str(role).strip().lower()
    allowed = {str(item).strip().lower() for item in policy["reviewer_roles"]}
    if normalized not in allowed:
        raise PermissionError("identity role is not allowed to review challenges")
    return normalized


def validate_governed_transition(
    challenge: Mapping[str, Any],
    reviews: Sequence[Mapping[str, Any]],
    *,
    target: str,
    actor_role: str,
    policy: Mapping[str, Any],
) -> None:
    """Validate an approval/publication transition from persisted evidence."""

    current = str(challenge["status"])
    if not str(challenge.get("actorId", "")).strip():
        raise ValueError("challenge governance actor identity is required")
    governors = {str(item).strip().lower() for item in policy["publisher_roles"]}
    if str(actor_role).strip().lower() not in governors:
        raise PermissionError("identity role is not allowed to approve or publish challenges")
    if target == "approved":
        if current != "review":
            raise ValueError("only a challenge in review can be approved")
        payload = challenge.get("payload") or {}
        _require_truth_contract(payload)
        if payload.get("publicationPolicySha256") != publication_policy_sha256(policy):
            raise ValueError("challenge publication policy identity does not match")
        validate_evidence_sha256(str(payload.get("configurationBundleSha256", "")))
        interestingness = payload.get("interestingness") or {}
        stability = payload.get("stability") or {}
        if float(challenge["publishability"]) < float(policy["minimum_publishability"]):
            raise ValueError("challenge publishability is below policy")
        if float(interestingness.get("robustnessScore", 0.0)) < float(policy["minimum_robustness"]):
            raise ValueError("challenge robustness is below policy")
        archetype = str(challenge.get("archetype", ""))
        archetype_score = float((interestingness.get("archetypeScores") or {}).get(archetype, 0.0))
        if archetype_score < float(policy["minimum_archetype_score"]):
            raise ValueError("challenge archetype score is below policy")
        if payload.get("observerSafe") is not True:
            raise ValueError("challenge lacks affirmative observer-safety evidence")
        if stability.get("accepted") is not True or stability.get("status") != "stable":
            raise ValueError("challenge lacks accepted stable reanalysis evidence")
        unique = {str(item["reviewerId"]) for item in reviews}
        approvals = {str(item["reviewerId"]) for item in reviews if item["decision"] == "approve"}
        if any(item["decision"] == "reject" for item in reviews):
            raise ValueError("challenge has an independent rejection")
        quorum = int(policy["reviewer_quorum"])
        if len(unique) < quorum or len(approvals) < quorum:
            raise ValueError("challenge independent-review quorum is not satisfied")
        if policy.get("require_governor_independence", True) and str(challenge.get("actorId", "")) in unique:
            raise ValueError("challenge governor must be independent from its reviewers")
        return
    if target == "published":
        if current != "approved":
            raise ValueError("only an approved challenge can be published")
        return
    if target == "retired":
        if current not in {"approved", "published"}:
            raise ValueError("only an approved or published challenge can be retired")
        return
    raise ValueError("governed challenge target must be approved, published, or retired")
