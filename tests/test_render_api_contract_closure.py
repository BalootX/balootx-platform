from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from service_test_client import OperatorClient as TestClient
from pydantic import ValidationError

from baloot_platform.service import api
from baloot_platform.service.grpc_service import validate_render_batch_request_payload
from baloot_platform.service.routes.foundry_admin_v2 import UnifiedLaffaRowV1
from baloot_platform.video.batch_service import (
    RENDER_AUDIT_VIDEO_CANDIDATE_EVIDENCE_KEYS,
    RENDER_BATCH_ITEM_KEYS,
    RENDER_BUNDLE_ARTIFACT_KEYS,
    RENDER_DECISION_SELECTION_EVIDENCE_KEYS,
    RENDER_PUBLICATION_POLICY_KEYS,
    RENDER_PLAYER_LOCATOR_KEYS,
    RENDER_SOURCE_KEYS,
    RENDER_VIDEO_CANDIDATE_EVIDENCE_KEYS,
    RENDER_VIDEO_PRODUCTION_EVIDENCE_KEYS,
    RenderBatchServiceV1,
)
from baloot_platform.video.render_bundle import RENDER_INTEREST_ADMISSION_KEYS


def _artifact() -> dict[str, Any]:
    return {
        "ownerKind": "render_bundle_v1",
        "ownerHash": "a" * 64,
        "uri": "cas://sha256/" + "b" * 64,
        "sha256": "b" * 64,
        "bytes": 12,
        "mediaType": "application/vnd.balootx.render-bundle-v1+json",
        "name": "bundle.json",
    }


def _publication_policy() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "decisionSelection": {
            "schemaVersion": 1,
            "profileHash": "c" * 64,
            "evidenceHash": "d" * 64,
            "decisionId": "DEC-1",
            "gatePassed": True,
            "qualifiedDecision": True,
            "blockers": [],
            "sha256": "e" * 64,
        },
        "videoProduction": {
            "schemaVersion": 1,
            "profileHash": "f" * 64,
            "candidate": {
                "decisionKind": "play",
                "critical": True,
                "truthLabel": "robust_best_known",
                "selectionScore": 91.0,
                "confidence": 0.9,
                "compatibilityBlockers": [],
            },
            "gatePassed": True,
            "blockers": [],
            "sha256": "1" * 64,
        },
        "blockers": [],
        "qualityEvidenceHash": "7" * 64,
        "sha256": "2" * 64,
    }


def _audit_publication_policy() -> dict[str, Any]:
    blockers = ["audit_trace_publication_not_authorized"]
    return {
        "schemaVersion": 1,
        "decisionSelection": {
            "schemaVersion": 1,
            "profileHash": None,
            "evidenceHash": None,
            "decisionId": "DEC-2",
            "gatePassed": False,
            "qualifiedDecision": False,
            "blockers": blockers,
            "sha256": "a" * 64,
        },
        "videoProduction": {
            "schemaVersion": 1,
            "profileHash": None,
            "candidate": {"decisionId": "DEC-2", "auditOnly": True},
            "gatePassed": False,
            "blockers": blockers,
            "sha256": "b" * 64,
        },
        "blockers": blockers,
        "sha256": "c" * 64,
    }


def _research_prepare_response(
    *, observer_seat: str = "south", decision_ordinal: int = 1
) -> dict[str, Any]:
    policy = _publication_policy()
    source = {
        "kind": "research_result",
        "code": "LAF-ABC123",
        "decisionOrdinal": decision_ordinal,
        "decisionId": "DEC-1",
        "observerSeat": observer_seat,
        "traceHash": "3" * 64,
        "analysisHash": "4" * 64,
        "domainHash": "5" * 64,
        "rulesHash": "6" * 64,
        "qualificationStatus": "qualified",
        "renderPurpose": "content_candidate",
        "qualificationFailures": [],
        "publicationBlockers": [],
        "publicationEligible": True,
        "publicationPolicy": policy,
    }
    return {
        "schemaVersion": 1,
        "resultCode": "LAF-ABC123",
        "decisionOrdinal": decision_ordinal,
        "decisionId": "DEC-1",
        "observerSeat": observer_seat,
        "exportPreset": "decision_clip",
        "qualificationStatus": "qualified",
        "publicationEligible": True,
        "renderPurpose": "content_candidate",
        "qualificationFailures": [],
        "publicationBlockers": [],
        "publicationPolicy": policy,
        "locator": {},
        "bundleArtifact": _artifact(),
        "basename": "research-decision",
        "preview": {},
        "source": source,
    }


def _audit_prepare_response(observer_seat: str) -> dict[str, Any]:
    blocker = "audit_trace_publication_not_authorized"
    failures = ["content_qualification_not_established", blocker]
    code = "LAF-AUD-ABCDEF1234"
    policy = _audit_publication_policy()
    return {
        "schemaVersion": 1,
        "laffaCode": code,
        "auditTraceCode": code,
        "recordHash": "8" * 64,
        "traceHash": "7" * 64,
        "decisionOrdinal": 2,
        "decisionId": "DEC-2",
        "observerSeat": observer_seat,
        "exportPreset": "hooked_flagship",
        "qualificationStatus": "quarantined",
        "auditOnly": True,
        "publicationEligible": False,
        "renderPurpose": "research_audit",
        "qualificationFailures": failures,
        "publicationBlockers": [blocker],
        "publicationPolicy": policy,
        "locator": {},
        "bundleArtifact": _artifact(),
        "basename": "research-audit",
        "preview": {},
        "source": {
            "kind": "laffa",
            "code": code,
            "decisionOrdinal": 2,
            "decisionId": "DEC-2",
            "auditTraceCode": code,
            "recordHash": "8" * 64,
            "attemptStatus": "completed",
            "observerSeat": observer_seat,
            "traceHash": "7" * 64,
            "rulesHash": "6" * 64,
            "renderPurpose": "research_audit",
            "originKind": "research_audit_trace",
            "qualificationStatus": "quarantined",
            "auditOnly": True,
            "publicationEligible": False,
            "qualificationFailures": failures,
            "publicationBlockers": [blocker],
            "publicationPolicy": policy,
        },
    }


def _legacy_prepare_response(*, code: str, observer_seat: str) -> dict[str, Any]:
    failures = ["content_qualification_not_established"]
    return {
        "schemaVersion": 1,
        "laffaCode": code,
        "decisionOrdinal": 1,
        "decisionId": "DEC-1",
        "observerSeat": observer_seat,
        "exportPreset": "hooked_flagship",
        "qualificationStatus": "legacy_replay",
        "publicationEligible": False,
        "renderPurpose": "research_audit",
        "qualificationFailures": failures,
        "locator": {},
        "bundleArtifact": _artifact(),
        "basename": "legacy-audit",
        "preview": {},
        "source": {
            "kind": "laffa",
            "code": code,
            "decisionOrdinal": 1,
            "traceHash": "9" * 64,
            "renderPurpose": "research_audit",
        },
    }


def _batch_body() -> dict[str, Any]:
    return {
        "items": [
            {
                "bundleArtifact": _artifact(),
                "basename": "decision",
                "source": {
                    "kind": "research_result",
                    "code": "LAF-ABC123",
                    "observerSeat": "west",
                    "renderPurpose": "research_audit",
                    "publicationPolicy": _publication_policy(),
                },
            }
        ]
    }


def test_render_models_are_closed_and_provenance_is_typed() -> None:
    request = api.ResearchRenderPrepareRequest.model_validate(
        {"observerSeat": "west"}
    )
    assert request.observerSeat == "west"
    with pytest.raises(ValidationError):
        api.ResearchRenderPrepareRequest.model_validate({"observerSeat": "up"})
    with pytest.raises(ValidationError):
        api.ResearchRenderPrepareRequest.model_validate({"unexpected": True})

    response = api.ResearchRenderPrepareResponse.model_validate(
        _research_prepare_response()
    )
    assert response.source.analysisHash == "4" * 64
    assert response.source.domainHash == "5" * 64
    assert response.source.rulesHash == "6" * 64
    assert response.publicationPolicy.videoProduction.candidate is not None

    invalid = _research_prepare_response()
    invalid["source"]["unexpected"] = True
    with pytest.raises(ValidationError):
        api.ResearchRenderPrepareResponse.model_validate(invalid)

    audit = api.ResearchAuditTraceRenderPrepareResponse.model_validate(
        _audit_prepare_response("south")
    )
    assert audit.source.auditTraceCode == "LAF-AUD-ABCDEF1234"
    assert audit.source.attemptStatus == "completed"
    assert audit.publicationPolicy.videoProduction.candidate is not None
    batch = api.FoundryRenderBatchCreateRequest.model_validate(
        {
            "items": [
                {
                    "bundleArtifact": _artifact(),
                    "basename": "audit-decision",
                    "source": audit.source.model_dump(exclude_none=True),
                }
            ]
        }
    )
    assert batch.items[0].source is not None
    assert batch.items[0].source.recordHash == "8" * 64


def test_batch_model_fields_match_direct_service_allowlists() -> None:
    assert set(api.FoundryRenderBatchItemRequest.model_fields) == RENDER_BATCH_ITEM_KEYS
    assert (
        set(api.FoundryRenderBundleArtifactReference.model_fields)
        == RENDER_BUNDLE_ARTIFACT_KEYS
    )
    assert set(api.FoundryRenderPlayerLocator.model_fields) == RENDER_PLAYER_LOCATOR_KEYS
    assert set(api.FoundryRenderSourceReference.model_fields) == RENDER_SOURCE_KEYS
    assert (
        set(api.ResearchRenderPublicationPolicyResponse.model_fields)
        == RENDER_PUBLICATION_POLICY_KEYS
    )
    assert set(api.ResearchRenderInterestAdmissionResponse.model_fields) == RENDER_INTEREST_ADMISSION_KEYS
    assert (
        set(api.ResearchRenderDecisionSelectionEvidenceResponse.model_fields)
        == RENDER_DECISION_SELECTION_EVIDENCE_KEYS
    )
    assert (
        set(api.ResearchRenderVideoProductionEvidenceResponse.model_fields)
        == RENDER_VIDEO_PRODUCTION_EVIDENCE_KEYS
    )
    assert (
        set(api.ResearchRenderVideoCandidateEvidenceResponse.model_fields)
        == RENDER_VIDEO_CANDIDATE_EVIDENCE_KEYS
    )
    assert (
        set(api.ResearchAuditRenderVideoCandidateEvidenceResponse.model_fields)
        == RENDER_AUDIT_VIDEO_CANDIDATE_EVIDENCE_KEYS
    )


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ((), "unexpectedOuter"),
        (("items", 0), "unexpectedItem"),
        (("items", 0, "bundleArtifact"), "unexpectedArtifact"),
        (("items", 0, "source"), "unexpectedSource"),
        (("items", 0, "source", "publicationPolicy"), "unexpectedPolicy"),
        (
            ("items", 0, "source", "publicationPolicy", "decisionSelection"),
            "unexpectedDecisionEvidence",
        ),
        (
            ("items", 0, "source", "publicationPolicy", "videoProduction"),
            "unexpectedVideoEvidence",
        ),
        (
            (
                "items",
                0,
                "source",
                "publicationPolicy",
                "videoProduction",
                "candidate",
            ),
            "unexpectedCandidateEvidence",
        ),
    ],
)
def test_nested_render_batch_models_reject_unknown_fields(
    path: tuple[str | int, ...], field: str
) -> None:
    payload = _batch_body()
    cursor: Any = payload
    for part in path:
        cursor = cursor[part]
    cursor[field] = True
    with pytest.raises(ValidationError):
        api.FoundryRenderBatchCreateRequest.model_validate(payload)


@pytest.mark.parametrize("admission", [
    {"passed": True, "evidenceHash": "a" * 64},
    {"passed": False, "evidenceHash": "b" * 64},
    None, {}, {"passed": "true", "evidenceHash": "a" * 64},
    {"passed": 1, "evidenceHash": "a" * 64},
    {"passed": True, "evidenceHash": "A" * 64},
    {"passed": True, "evidenceHash": "a" * 64, "baselineAction": "SA"},
])
def test_render_interest_admission_api_and_direct_batch_agree(admission):
    from baloot_platform.video.batch_service import _check_source_keys
    payload = _batch_body()
    source = payload["items"][0]["source"]
    source["publicationPolicy"]["interestAdmission"] = admission
    valid = isinstance(admission, dict) and type(admission.get("passed")) is bool and len(admission) == 2
    valid = valid and len(admission["evidenceHash"]) == 64 and admission["evidenceHash"].islower()
    if valid:
        api.FoundryRenderBatchCreateRequest.model_validate(payload)
        _check_source_keys(source)
    else:
        with pytest.raises(ValidationError):
            api.FoundryRenderBatchCreateRequest.model_validate(payload)
        with pytest.raises(ValueError, match="interest admission"):
            _check_source_keys(source)


def test_legacy_publication_policy_serializes_without_new_null_field():
    model = api.ResearchRenderPublicationPolicyResponse.model_validate(_publication_policy())
    serialized = model.model_dump(mode="json")
    assert "interestAdmission" not in serialized
    api.ResearchRenderPublicationPolicyResponse.model_validate(serialized)
    payload = _batch_body()
    payload["items"][0]["source"]["publicationPolicy"] = serialized
    request = api.FoundryRenderBatchCreateRequest.model_validate(payload)
    api.FoundryRenderBatchCreateRequest.model_validate(request.model_dump(mode="json"))


@pytest.mark.parametrize("mode", ("validation", "serialization"))
def test_interest_admission_schema_is_optional_without_null(mode):
    schema = api.ResearchRenderPublicationPolicyResponse.model_json_schema(mode=mode)
    admission = schema["properties"]["interestAdmission"]
    assert admission["$ref"] == "#/$defs/ResearchRenderInterestAdmissionResponse"
    assert "anyOf" not in admission and admission.get("type") != "null"
    assert "interestAdmission" not in schema["required"]


def _qualified_scenario_batch_item() -> dict[str, Any]:
    code = "LAF-A1B2C3D4E5"
    source = {
        "sourceSchemaVersion": 3,
        "kind": "research_scenario_result",
        "code": code,
        "decisionOrdinal": 1,
        "decisionReference": {"resultCode": code, "decisionOrdinal": 1},
        "publicProjectionHash": "1" * 64,
        "qrActionable": False,
        "rulesHash": "2" * 64,
        "proofReference": code,
        "proofCompletionStatus": "exact_complete",
        "proofClaimSatisfied": True,
        "proofAchievedMode": "exact",
        "qualificationStatus": "qualified",
        "renderPurpose": "content_candidate",
        "qualificationFailures": [],
        "publicationBlockers": [],
        "observerSeat": "south",
        "auditOnly": False,
        "publicationEligible": True,
        "publicationPolicy": _publication_policy(),
    }
    authority = {
        "schemaVersion": 1,
        "audience": "operator_audit",
        "resultCode": code,
        "resultHash": "3" * 64,
        "decisionOrdinal": 1,
        "exactStateId": "STATE-" + "A" * 20,
        "stateHash": "4" * 64,
        "exactStateHash": "5" * 64,
        "rulesHash": "2" * 64,
        "configurationBundleHash": "6" * 64,
        "scenarioHash": "7" * 64,
        "proofSemanticSha256": "8" * 64,
        "proofArtifactSha256": "9" * 64,
        "constraintEvidenceHash": "a" * 64,
        "sha256": "b" * 64,
    }
    return {
        "bundleArtifact": _artifact(),
        "basename": "qualified-scenario",
        "source": source,
        "authorityReceipt": authority,
        "locator": {
            "schemaVersion": 1,
            "code": code,
            "universalLink": f"/c/{code}",
        },
    }


def test_render_player_locator_is_closed_same_origin_and_authority_bound() -> None:
    item = _qualified_scenario_batch_item()
    request = api.FoundryRenderBatchCreateRequest.model_validate({"items": [item]})
    assert request.items[0].locator is not None
    assert request.items[0].locator.universalLink == "/c/LAF-A1B2C3D4E5"

    for invalid_link in (
        "balootx://research/LAF-A1B2C3D4E5/decisions/1",
        "https://example.test/c/LAF-A1B2C3D4E5",
        "/c/LAF-A1B2C3D4E5?observer=south",
        "/c/LAF-FFFFFFFFFF",
    ):
        invalid = copy.deepcopy(item)
        invalid["locator"]["universalLink"] = invalid_link
        with pytest.raises(ValidationError):
            api.FoundryRenderBatchCreateRequest.model_validate({"items": [invalid]})

    audit_only = copy.deepcopy(item)
    audit_only["source"].update(
        {
            "auditOnly": True,
            "publicationEligible": False,
            "renderPurpose": "research_audit",
        }
    )
    with pytest.raises(ValidationError, match="qualified prepared result authority"):
        api.FoundryRenderBatchCreateRequest.model_validate({"items": [audit_only]})

    legacy = copy.deepcopy(item)
    legacy.pop("locator")
    assert api.FoundryRenderBatchCreateRequest.model_validate({"items": [legacy]})


class _FakeResearchRenders:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def prepare(self, code: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((code, kwargs))
        return _research_prepare_response(
            observer_seat=str(kwargs.get("observer_seat") or "south"),
            decision_ordinal=int(kwargs.get("decision_ordinal") or 1),
        )


def test_http_result_and_decision_prepare_forward_observer(monkeypatch) -> None:
    renders = _FakeResearchRenders()
    monkeypatch.setattr(api, "RESEARCH_RENDERS", renders)
    client = TestClient(api.app)

    result = client.post(
        "/api/v2/research-results/LAF-ABC123:prepare-render",
        json={"observerSeat": "west", "exportPreset": "decision_clip"},
    )
    assert result.status_code == 200, result.text
    assert result.json()["source"]["analysisHash"] == "4" * 64
    assert result.json()["publicationPolicy"]["videoProduction"]["gatePassed"] is True
    assert result.json()["publicationPolicy"]["qualityEvidenceHash"] == "7" * 64
    assert result.json()["source"]["publicationPolicy"]["qualityEvidenceHash"] == "7" * 64

    decision = client.post(
        "/api/v2/research-results/LAF-ABC123/decisions/7:prepare-render",
        json={"observerSeat": "east"},
    )
    assert decision.status_code == 200, decision.text
    assert decision.json()["observerSeat"] == "east"
    assert renders.calls == [
        (
            "LAF-ABC123",
            {
                "observer_seat": "west",
                "reveal_policy": "after_commit",
                "export_preset": "decision_clip",
            },
        ),
        (
            "LAF-ABC123",
            {
                "decision_ordinal": 7,
                "observer_seat": "east",
                "reveal_policy": "after_commit",
                "export_preset": "decision_clip",
            },
        ),
    ]

    rejected = client.post(
        "/api/v2/research-results/LAF-ABC123:prepare-render",
        json={"observerSeat": "west", "unexpected": True},
    )
    assert rejected.status_code == 422
    assert len(renders.calls) == 2


def test_http_universal_audit_prepare_forwards_observer(monkeypatch) -> None:
    record = {"auditTraceCode": "LAF-AUD-ABCDEF1234"}

    class Research:
        @staticmethod
        def get_result(_identifier: str) -> None:
            return None

    class Library:
        @staticmethod
        def resolve_audit_trace(identifier: str) -> dict[str, Any] | None:
            return record if identifier == "LAF-AUD-ABCDEF1234" else None

    class Renders:
        calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

        @classmethod
        def prepare_audit_trace(
            cls, source: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            cls.calls.append((source, kwargs))
            return _audit_prepare_response(str(kwargs["observer_seat"]))

    class Batches:
        calls: list[tuple[str, str]] = []

        @classmethod
        def list_for_source(
            cls, *, source_kind: str, source_code: str
        ) -> list[dict[str, Any]]:
            cls.calls.append((source_kind, source_code))
            return []

    monkeypatch.setattr(api, "RESEARCH", Research())
    monkeypatch.setattr(api, "LAFFA_LIBRARY", Library())
    monkeypatch.setattr(api, "RESEARCH_RENDERS", Renders())
    monkeypatch.setattr(api, "RENDER_BATCHES", Batches())
    response = TestClient(api.app).post(
        "/api/v2/laffat/LAF-AUD-ABCDEF1234:prepare-render",
        json={"observerSeat": "north"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["auditOnly"] is True
    assert response.json()["source"]["originKind"] == "research_audit_trace"
    assert Renders.calls == [
        (
            record,
            {
                "observer_seat": "north",
                "reveal_policy": "after_commit",
                "export_preset": None,
            },
        )
    ]
    renders = TestClient(api.app).get(
        "/api/v2/laffat/LAF-AUD-ABCDEF1234/renders"
    )
    assert renders.status_code == 200, renders.text
    assert renders.json() == {"laffaCode": "LAF-AUD-ABCDEF1234", "items": []}
    assert Batches.calls == [("laffa", "LAF-AUD-ABCDEF1234")]


def test_http_universal_prepare_forwards_observer_for_existing_sources(
    monkeypatch,
) -> None:
    class Research:
        @staticmethod
        def get_result(identifier: str) -> dict[str, Any] | None:
            return {} if identifier == "LAF-ABC123" else None

    class Library:
        @staticmethod
        def resolve_audit_trace(_identifier: str) -> None:
            return None

        @staticmethod
        def resolve_sakka_round(identifier: str) -> dict[str, Any] | None:
            if identifier != "LAF-SAKKA01":
                return None
            return {"traceReference": {}, "code": "LAF-SAKKA01"}

    class Renders:
        calls: list[tuple[str, str, dict[str, Any]]] = []

        @classmethod
        def prepare(cls, code: str, **kwargs: Any) -> dict[str, Any]:
            cls.calls.append(("research", code, kwargs))
            return _research_prepare_response(
                observer_seat=str(kwargs["observer_seat"])
            )

        @classmethod
        def prepare_trace_reference(
            cls, _reference: dict[str, Any], *, code: str, **kwargs: Any
        ) -> dict[str, Any]:
            cls.calls.append(("sakka", code, kwargs))
            return _legacy_prepare_response(
                code=code, observer_seat=str(kwargs["observer_seat"])
            )

        @classmethod
        def prepare_laffa(
            cls, identifier: str, **kwargs: Any
        ) -> dict[str, Any]:
            cls.calls.append(("archive", identifier, kwargs))
            return _legacy_prepare_response(
                code=identifier, observer_seat=str(kwargs["observer_seat"])
            )

    monkeypatch.setattr(api, "RESEARCH", Research())
    monkeypatch.setattr(api, "LAFFA_LIBRARY", Library())
    monkeypatch.setattr(api, "RESEARCH_RENDERS", Renders())
    client = TestClient(api.app)
    calls = (
        ("LAF-ABC123", "west"),
        ("LAF-SAKKA01", "east"),
        ("GAME-ABC123", "north"),
    )
    for identifier, observer_seat in calls:
        response = client.post(
            f"/api/v2/laffat/{identifier}:prepare-render",
            json={"observerSeat": observer_seat},
        )
        assert response.status_code == 200, response.text
        assert response.json()["observerSeat"] == observer_seat

    assert [call[2]["observer_seat"] for call in Renders.calls] == [
        "west",
        "east",
        "north",
    ]
    assert [call[2]["export_preset"] for call in Renders.calls] == [None] * 3


@pytest.mark.parametrize("source_kind", ["research_result", "research_scenario_result"])
def test_http_universal_research_renders_match_both_kinds_by_code(
    monkeypatch, source_kind
) -> None:
    code = "LAF-ABC123"
    jobs = [
        {
            "jobId": job_id,
            "status": "completed",
            "request": {"items": [{"source": {"kind": kind, "code": source_code}}]},
        }
        for job_id, kind, source_code in (
            ("RENDER-MATCH", source_kind, code),
            ("RENDER-FOREIGN", source_kind, "LAF-FOREIGN"),
            ("RENDER-NONRESEARCH", "laffa", code),
        )
    ]
    shared = {
        "jobId": "RENDER-SHARED",
        "status": "completed",
        "request": {"items": [
            {"source": {"kind": kind, "code": code}}
            for kind in ("research_result", "research_scenario_result")
        ]},
    }
    jobs.append(shared)
    original_jobs = copy.deepcopy(jobs)

    class Research:
        @staticmethod
        def get_result(identifier: str) -> dict[str, Any] | None:
            return {"resultCode": code} if identifier == code else None

    class Batches:
        list_for_source = RenderBatchServiceV1.list_for_source

        @staticmethod
        def list(*, limit: int) -> list[dict[str, Any]]:
            assert limit == 1_000
            return jobs

    monkeypatch.setattr(api, "RESEARCH", Research())
    monkeypatch.setattr(api, "RENDER_BATCHES", Batches())
    response = TestClient(api.app).get(f"/api/v2/laffat/{code}/renders")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["laffaCode"] == code
    assert len(payload["items"]) == 2
    assert {job["jobId"]: job for job in payload["items"]} == {
        job["jobId"]: job for job in (jobs[0], shared)
    }
    assert jobs == original_jobs


@pytest.mark.parametrize(
    "item",
    [
        {"bundleArtifact": _artifact(), "basename": "decision", "unexpected": True},
        {
            "bundleArtifact": {**_artifact(), "unexpected": True},
            "basename": "decision",
        },
        {
            "bundleArtifact": _artifact(),
            "basename": "decision",
            "source": {
                "kind": "research_result",
                "code": "LAF-ABC123",
                "unexpected": True,
            },
        },
        {
            "bundleArtifact": _artifact(),
            "basename": "decision",
            "source": {
                "kind": "research_result",
                "code": "LAF-ABC123",
                "publicationPolicy": {
                    **_publication_policy(),
                    "unexpected": True,
                },
            },
        },
    ],
)
def test_direct_batch_service_rejects_unknown_keys_before_side_effects(
    item: dict[str, Any]
) -> None:
    service = object.__new__(RenderBatchServiceV1)
    with pytest.raises(ValueError, match="unknown fields"):
        service.submit((item,), idempotency_key="closed-contract")


def test_direct_batch_service_rejects_unknown_persisted_outer_keys() -> None:
    class Jobs:
        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        @staticmethod
        def cancelled(_job_id: str) -> bool:
            return False

        def update(self, _job_id: str, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    service = object.__new__(RenderBatchServiceV1)
    service.jobs = Jobs()
    service._execute_once(
        "render-1",
        {"schemaVersion": 1, "items": [_batch_body()["items"][0]], "unknown": 1},
    )
    assert service.jobs.updates[-1]["status"] == "failed"
    assert "unknown fields" in service.jobs.updates[-1]["error"]


def test_grpc_render_batch_validator_checks_complete_envelope() -> None:
    items = validate_render_batch_request_payload(
        json.dumps(_batch_body()).encode("utf-8")
    )
    assert items[0]["source"]["observerSeat"] == "west"
    assert "source" in items[0]

    invalid = _batch_body()
    invalid["unexpectedOuter"] = True
    with pytest.raises(ValueError):
        validate_render_batch_request_payload(json.dumps(invalid).encode("utf-8"))
    with pytest.raises(ValueError, match="must contain an object"):
        validate_render_batch_request_payload(b"[]")


def test_http_render_batch_rejects_nested_unknown_fields(monkeypatch) -> None:
    class Batches:
        calls: list[tuple[tuple[dict[str, Any], ...], str, bool]] = []

        @classmethod
        def submit(
            cls,
            items: tuple[dict[str, Any], ...],
            *,
            idempotency_key: str,
            playback_evidence: bool = False,
        ) -> dict[str, Any]:
            cls.calls.append((items, idempotency_key, playback_evidence))
            return {
                "jobId": "render-1",
                "tenantId": "default",
                "kind": "render_batch_v1",
                "status": "queued",
                "priority": 0,
                "progress": 0.0,
                "request": {"schemaVersion": 1, "items": list(items)},
                "attempts": 0,
                "maxAttempts": 3,
                "cancelRequested": False,
                "createdAt": "2026-08-09T00:00:00Z",
                "updatedAt": "2026-08-09T00:00:00Z",
            }

    monkeypatch.setattr(api, "RENDER_BATCHES", Batches())
    client = TestClient(api.app)
    assert client.post(
        "/api/v2/render-batches", json=_batch_body()
    ).status_code == 428
    assert client.post(
        "/api/v2/render-batches",
        json=_batch_body(),
        headers={"Idempotency-Key": "   "},
    ).status_code == 428
    oversized = client.post(
        "/api/v2/render-batches",
        json=_batch_body(),
        headers={"Idempotency-Key": "x" * 201},
    )
    assert oversized.status_code == 422
    assert isinstance(oversized.json()["detail"], str)
    assert Batches.calls == []

    invalid = _batch_body()
    invalid["items"][0]["source"]["unexpected"] = True
    response = client.post(
        "/api/v2/render-batches",
        json=invalid,
        headers={"Idempotency-Key": "render-closed"},
    )
    assert response.status_code == 422
    assert Batches.calls == []

    coercive = _batch_body()
    coercive["playbackEvidence"] = "true"
    assert client.post(
        "/api/v2/render-batches",
        json=coercive,
        headers={"Idempotency-Key": "render-coercive"},
    ).status_code == 422
    assert Batches.calls == []

    accepted = client.post(
        "/api/v2/render-batches",
        json=_batch_body(),
        headers={"Idempotency-Key": "render-closed"},
    )
    assert accepted.status_code == 202, accepted.text
    assert Batches.calls[0][0][0]["source"]["observerSeat"] == "west"


def test_render_source_v2_api_delta_is_strict_but_keeps_bundle_conditionality() -> None:
    source = {
        "sourceSchemaVersion": 2,
        "kind": "research_result",
        "code": "LAF-ABC123",
        "decisionOrdinal": 1,
        "decisionId": "DEC-1",
        "exactStateId": "DEC-1",
        "qrActionable": True,
        "resultHash": "1" * 64,
        "auditHash": "2" * 64,
        "analysisProfileHash": "3" * 64,
        "analyzerId": "adaptive_finalist_trace_analyzer_v1",
        "analyzerVersion": "1.3.0",
    }
    model = api.ResearchRenderSourceProvenanceResponse.model_validate(source)
    assert model.model_dump(mode="json", exclude_none=True) == source

    historical = {"kind": "laffa", "code": "LAF-HISTORICAL"}
    assert api.ResearchRenderSourceProvenanceResponse.model_validate(
        historical
    ).model_dump(mode="json", exclude_none=True) == historical

    for field, aliases in {
        "sourceSchemaVersion": (True, 2.0, "2"),
        "decisionOrdinal": (True, 1.0, "1"),
        "qrActionable": (1, 1.0, "true"),
    }.items():
        for alias in aliases:
            invalid = copy.deepcopy(source)
            invalid[field] = alias
            with pytest.raises(ValidationError):
                api.ResearchRenderSourceProvenanceResponse.model_validate(invalid)
    for field in ("exactStateId", "analyzerId"):
        invalid = copy.deepcopy(source)
        invalid[field] = ""
        with pytest.raises(ValidationError):
            api.ResearchRenderSourceProvenanceResponse.model_validate(invalid)
    for field in ("resultHash", "auditHash", "analysisProfileHash"):
        invalid = copy.deepcopy(source)
        invalid[field] = "A" * 64
        with pytest.raises(ValidationError):
            api.ResearchRenderSourceProvenanceResponse.model_validate(invalid)
    for version in ("1.3", "01.3.0", "1.3.0-beta", 1.3):
        invalid = copy.deepcopy(source)
        invalid["analyzerVersion"] = version
        with pytest.raises(ValidationError):
            api.ResearchRenderSourceProvenanceResponse.model_validate(invalid)


def test_live_openapi_closes_render_and_v3_tactic_schemas() -> None:
    schema = api.app.openapi()
    components = schema["components"]["schemas"]
    for name in (
        "ResearchRenderPrepareRequest",
        "ResearchRenderPrepareResponse",
        "ResearchRenderSourceProvenanceResponse",
        "FoundryRenderBatchCreateRequest",
        "FoundryRenderBatchItemRequest",
        "FoundryRenderBundleArtifactReference",
        "FoundryRenderProofArtifactReference",
        "FoundryRenderSourceReference",
    ):
        assert components[name]["additionalProperties"] is False

    request_properties = components["ResearchRenderPrepareRequest"]["properties"]
    observer_enum = next(
        item["enum"]
        for item in request_properties["observerSeat"]["anyOf"]
        if "enum" in item
    )
    assert set(observer_enum) == {"north", "west", "south", "east"}
    assert set(
        components["FoundryRenderProofArtifactReference"]["properties"]["mediaType"]
        ["enum"]
    ) == {
        "application/vnd.baloot.research-proof-outcome-v1+json",
        "application/vnd.balootx.ortools-proof-outcome-v1+json",
    }

    source_properties = components["ResearchRenderSourceProvenanceResponse"][
        "properties"
    ]
    assert {
        "observerSeat",
        "analysisHash",
        "domainHash",
        "rulesHash",
        "originKind",
        "sourceStatus",
        "renderPurpose",
        "qualificationFailures",
        "publicationEligible",
        "publicationPolicy",
        "auditTraceCode",
        "recordHash",
        "attemptStatus",
        "sourceSchemaVersion",
        "exactStateId",
        "qrActionable",
        "resultHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    } <= set(source_properties)
    for field_name in (
        "analysisHash",
        "domainHash",
        "rulesHash",
        "publicationPolicy",
    ):
        assert {item.get("type") for item in source_properties[field_name]["anyOf"]} >= {
            "null"
        }

    render_operation = schema["paths"]["/api/v2/render-batches"]["post"]
    idempotency_header = next(
        item
        for item in render_operation["parameters"]
        if item["in"] == "header" and item["name"] == "Idempotency-Key"
    )
    assert idempotency_header["required"] is True
    assert idempotency_header["schema"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": 200,
        "pattern": r".*\S.*",
    }
    assert {"202", "422", "428"}.issubset(render_operation["responses"])
    assert render_operation["responses"]["422"]["content"][
        "application/json"
    ]["schema"] == {
        "$ref": "#/components/schemas/ResearchMutationValidationErrorResponse"
    }
    assert render_operation["responses"]["428"]["content"][
        "application/json"
    ]["schema"] == {
        "$ref": "#/components/schemas/ResearchMutationErrorResponse"
    }

    universal_response = schema["paths"][
        "/api/v2/laffat/{identifier}:prepare-render"
    ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert {
        item["$ref"].rsplit("/", 1)[-1] for item in universal_response["anyOf"]
    } == {
        "ResearchRenderPrepareResponse",
        "LaffaRenderPrepareResponse",
        "ResearchAuditTraceRenderPrepareResponse",
    }
    audit_response_properties = components[
        "ResearchAuditTraceRenderPrepareResponse"
    ]["properties"]
    assert {
        "auditTraceCode",
        "recordHash",
        "traceHash",
        "auditOnly",
        "publicationBlockers",
        "publicationPolicy",
    } <= set(audit_response_properties)

    assert components["ResearchTacticV3Request"]["properties"]["mode"]["enum"] == [
        "preference",
        "qualification_gate",
    ]
    assert "hard" in components["ResearchTacticRequest"]["properties"]["mode"][
        "enum"
    ]
    assert "research_audit_trace" in components["UnifiedLaffaRowV1"]["properties"][
        "sourceKind"
    ]["enum"]
    assert "research_audit_trace" in components["LaffaSourceReferenceV1"][
        "properties"
    ]["kind"]["enum"]
    assert "research_audit_trace" in components["LaffaSourceFacetV1"]["properties"][
        "sourceKind"
    ]["enum"]
    assert {
        "auditOnly", "publicationEligible", "qualificationFailures"
    } <= set(components["UnifiedLaffaRowV1"]["required"])
    source_parameter = next(
        item
        for item in schema["paths"]["/api/v2/laffat"]["get"]["parameters"]
        if item["name"] == "sourceKind"
    )
    source_enum = next(
        item["enum"] for item in source_parameter["schema"]["anyOf"] if "enum" in item
    )
    assert "research_audit_trace" in source_enum


def test_v3_tactic_hard_is_rejected_while_legacy_model_preserves_it() -> None:
    with pytest.raises(ValidationError):
        api.ResearchTacticV3Request.model_validate(
            {"kind": "only_winning_move", "mode": "hard"}
        )
    assert api.ResearchTacticV3Request.model_validate(
        {"kind": "only_winning_move", "mode": "qualification_gate"}
    ).mode == "qualification_gate"
    assert api.ResearchTacticRequest.model_validate(
        {"kind": "only_winning_move", "mode": "hard"}
    ).mode == "hard"

    request = {
        "schemaVersion": 3,
        "titleAr": "اختبار وصفة",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": "CBR-TEST-V3",
        },
        "tactics": [{"kind": "only_winning_move", "mode": "hard"}],
    }
    with pytest.raises(ValidationError):
        api.ResearchRecipeV3CreateRequest.model_validate(request)
    response = TestClient(api.app).post(
        "/api/v2/research-recipes:validate", json=request
    )
    assert response.status_code == 422


def test_unified_audit_row_requires_governance_fields() -> None:
    row = {
        "humanCode": "LAF-AUD-ABCDEF1234",
        "sourceKind": "research_audit_trace",
        "sourceKinds": ["research_audit_trace"],
        "replayValid": True,
        "auditOnly": True,
        "publicationEligible": False,
        "qualificationFailures": ["content_qualification_not_established"],
        "qualificationStatus": "quarantined",
        "tacticalTags": [],
        "projects": [],
        "interestingSeats": [],
        "relatedSakkas": [],
        "renderCount": 0,
        "renderStatus": "not_prepared",
        "createdAt": "2026-08-09T00:00:00Z",
        "sourceLocator": {
            "kind": "research_audit_trace",
            "code": "LAF-AUD-ABCDEF1234",
            "identity": "8" * 64,
            "uri": "/api/v2/laffat/LAF-AUD-ABCDEF1234",
        },
        "sourceReferences": [],
    }
    assert UnifiedLaffaRowV1.model_validate(row).auditOnly is True
    missing = copy.deepcopy(row)
    del missing["publicationEligible"]
    with pytest.raises(ValidationError):
        UnifiedLaffaRowV1.model_validate(missing)
