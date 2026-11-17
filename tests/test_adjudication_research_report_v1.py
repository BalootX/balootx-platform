from __future__ import annotations

import json
from pathlib import Path

import pytest

from baloot_platform.adjudication import (
    AdjudicationResearchAuthorityV1,
    AdjudicationResearchReportV1,
    AuthoredAdjudicationEventV1,
)


ROOT = Path(__file__).resolve().parents[1]


def _authority() -> AdjudicationResearchAuthorityV1:
    return AdjudicationResearchAuthorityV1.from_yaml(
        ROOT / "configs/adjudication/research_events_v1.yaml"
    )


def _event(*, remedy: str | None = "continue_play") -> AuthoredAdjudicationEventV1:
    return AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 6,
            "actorSeat": "west",
            "expectedActorSeat": "south",
            "attemptedAction": {"card": "SA"},
            "reasonCode": "play_before_turn",
            "evidenceRefs": ["PRIVATE-EVIDENCE-REF"],
            "selectedRemedy": remedy,
            "rulingEvidenceRef": "PRIVATE-RULING-REF" if remedy else None,
        }
    )


def test_report_is_authority_bound_deterministic_and_sanitized() -> None:
    authority = _authority()
    event = _event()
    report = AdjudicationResearchReportV1.build(
        authority,
        prefix_sha256="1" * 64,
        events=(event,),
        state_sha256="2" * 64,
        exact_state_id="STATE-1234567890ABCDEF1234",
    )

    wire = report.to_dict()
    assert set(wire) == {
        "schemaVersion",
        "kind",
        "authorityProfileId",
        "authoritySha256",
        "sourceId",
        "sourceSha256",
        "currentAuthorityBoundary",
        "prefixSha256",
        "stateSha256",
        "exactStateId",
        "events",
        "sha256",
    }
    assert wire["authorityProfileId"] == authority.profile_id
    assert wire["authoritySha256"] == authority.sha256
    assert wire["sourceId"] == authority.source_id
    assert wire["sourceSha256"] == authority.source_sha256
    assert wire["currentAuthorityBoundary"] == authority.current_authority_boundary
    assert wire["prefixSha256"] == "1" * 64
    assert wire["stateSha256"] == "2" * 64
    assert wire["exactStateId"] == "STATE-1234567890ABCDEF1234"

    event_wire = wire["events"][0]
    capability = authority.capability("out_of_turn")
    assert event_wire == {
        "schemaVersion": 1,
        "eventSha256": event.sha256,
        "capabilitySha256": capability.sha256,
        "kind": "out_of_turn",
        "stage": "play",
        "actionSequence": 6,
        "reasonCode": "play_before_turn",
        "sourceClause": capability.source_clause,
        "observerSafety": "referee_restricted",
        "remedyStatus": "selected",
        "selectedRemedy": "continue_play",
    }
    encoded = json.dumps(wire, sort_keys=True)
    for private_value in (
        "SA",
        "west",
        "south",
        "PRIVATE-EVIDENCE-REF",
        "PRIVATE-RULING-REF",
    ):
        assert private_value not in encoded

    rebuilt = AdjudicationResearchReportV1.build(
        authority,
        prefix_sha256="1" * 64,
        events=(event,),
        state_sha256="2" * 64,
        exact_state_id="STATE-1234567890ABCDEF1234",
    )
    assert rebuilt.sha256 == report.sha256
    with pytest.raises(TypeError):
        report.events[0]["kind"] = "changed"  # type: ignore[index]
    wire["events"][0]["kind"] = "changed"
    assert report.to_dict()["events"][0]["kind"] == "out_of_turn"


def test_report_validates_hashes_uniqueness_and_authority_remedies() -> None:
    authority = _authority()
    event = _event(remedy=None)
    report = AdjudicationResearchReportV1.build(
        authority,
        prefix_sha256="a" * 64,
        events=(event,),
        state_sha256="b" * 64,
        exact_state_id="STATE-AAAAAAAAAAAAAAAAAAAA",
    )
    assert report.to_dict()["events"][0]["remedyStatus"] == "unresolved"
    assert report.to_dict()["events"][0]["selectedRemedy"] is None

    with pytest.raises(ValueError, match="prefix_sha256"):
        AdjudicationResearchReportV1.build(
            authority,
            prefix_sha256="not-a-hash",
            events=(event,),
            state_sha256="b" * 64,
            exact_state_id="STATE-AAAAAAAAAAAAAAAAAAAA",
        )
    with pytest.raises(ValueError, match="events must be unique"):
        AdjudicationResearchReportV1.build(
            authority,
            prefix_sha256="a" * 64,
            events=(event, event),
            state_sha256="b" * 64,
            exact_state_id="STATE-AAAAAAAAAAAAAAAAAAAA",
        )

    wrong_remedy = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "invalid_bid",
            "actionSequence": 1,
            "actorSeat": "east",
            "expectedActorSeat": "north",
            "attemptedAction": {"token": "sun"},
            "reasonCode": "purchase_before_turn",
            "selectedRemedy": "opponent_register_with_projects",
            "rulingEvidenceRef": "PRIVATE-RULING-REF",
        }
    )
    with pytest.raises(ValueError, match="is not a remedy"):
        AdjudicationResearchReportV1.build(
            authority,
            prefix_sha256="a" * 64,
            events=(wrong_remedy,),
            state_sha256="b" * 64,
            exact_state_id="STATE-AAAAAAAAAAAAAAAAAAAA",
        )
