from __future__ import annotations

from typing import Any

from .models import content_hash


def build_video_audit_map(store: Any, video_id: str) -> dict[str, Any]:
    """Return one immutable video timeline enriched with replay intelligence.

    This is an application-level projection shared by REST, gRPC, MCP and SDKs.
    It deliberately contains no rendering or Baloot rules.  Decisions and
    audits are read from the authoritative history store and joined to the
    immutable renderer timecodes.
    """
    value = store.get_video_record(video_id)
    if value is None:
        raise KeyError(video_id)
    timeline = value.get("timeline") or {"segments": []}
    segments = list(timeline.get("segments") or [])
    decision_ids = sorted({str(item["decisionId"]) for item in segments if item.get("decisionId")})
    decisions: dict[str, Any] = {}
    for decision_id in decision_ids:
        decision = store.get_decision(decision_id)
        if decision is None:
            continue
        audits = decision.get("audits") or store.list_decision_audits(decision_id)
        audit = audits[0] if audits else None
        mapped_segments = [
            {
                "label": item.get("label"),
                "phase": item.get("phase"),
                "startSeconds": item.get("startSeconds"),
                "endSeconds": item.get("endSeconds"),
            }
            for item in segments
            if str(item.get("decisionId") or "") == decision_id
        ]
        feedback = decision.get("feedback") or store.list_feedback(decision_id)
        difficulty = decision.get("difficulty") or store.get_difficulty_item(decision_id)
        metadata = decision.get("decisionMetadata") or {}
        intent = (audit or {}).get("actualIntent")
        if intent is None and metadata.get("source") == "bot":
            intent = {
                "status": "recorded_policy_trace",
                "source": "bot",
                "profileId": metadata.get("profileId"),
                "strategyId": metadata.get("strategyId"),
                "trace": metadata,
            }
        decisions[decision_id] = {
            "decisionId": decision_id,
            "decisionHash": decision.get("decisionHash"),
            "gameHash": decision.get("gameHash"),
            "ordinal": decision.get("ordinal"),
            "decisionKind": decision.get("decisionKind"),
            "actor": decision.get("actor"),
            "phase": decision.get("phase"),
            "selectedAction": decision.get("selectedAction"),
            "legalActions": decision.get("legalActions") or [],
            "legalActionCount": len(decision.get("legalActions") or []),
            "decisionMetadata": metadata,
            "actualIntent": intent or {"status": "unknown"},
            "audit": audit,
            "auditHash": (audit or {}).get("auditHash"),
            "difficulty": difficulty,
            "feedbackCount": len(feedback),
            "reviewed": bool(feedback),
            "timecodes": mapped_segments,
            "links": {
                "decision": f"/api/v1/history/decisions/{decision_id}",
                "explain": f"/api/v1/history/decisions/{decision_id}:explain",
                "comparison": f"/api/v1/history/decisions/{decision_id}/comparison",
                "difficulty": f"/api/v1/history/decisions/{decision_id}/difficulty",
                "attempts": f"/api/v1/history/decisions/{decision_id}/attempts",
                "feedback": f"/api/v1/history/decisions/{decision_id}/feedback",
                "fork": f"/api/v1/history/decisions/{decision_id}:fork",
            },
        }
    projection = {
        "videoId": video_id,
        "ownerKind": value.get("ownerKind"),
        "ownerId": value.get("ownerId"),
        "gameHash": value.get("gameHash"),
        "rendererId": value.get("rendererId"),
        "createdAt": value.get("createdAt"),
        "manifest": value.get("manifest") or {},
        "timeline": timeline,
        "timelineHash": content_hash(timeline),
        "mappedDecisionCount": len(decisions),
        "decisions": decisions,
    }
    projection["auditMapHash"] = content_hash({
        "videoId": video_id,
        "gameHash": projection["gameHash"],
        "timelineHash": projection["timelineHash"],
        "decisionHashes": {key: value.get("decisionHash") for key, value in decisions.items()},
        "auditHashes": {key: value.get("auditHash") for key, value in decisions.items()},
    })
    return projection
