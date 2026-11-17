from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from baloot_platform.research.result_authority import ORTOOLS_RESULT_IDENTITY_V3_KIND
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.service import ResearchFoundryServiceV1


def _rows(kind: str, count: int) -> list[dict]:
    offset = {"bid": 0, "project": 100, "play": 200}[kind]
    return [
        {
            "decisionKind": kind,
            "actor": ("north", "east", "south", "west")[index % 4],
            "action": f"{kind}-{index}",
            "actualActionOptimal": True,
            "regret": 0,
            "selectionReason": "native_actor_information_set_best_value",
            "proofArtifactHash": f"{offset + index + 1:064x}",
            "actionQualityHash": f"{offset + index + 501:064x}",
        }
        for index in range(count)
    ]


@pytest.mark.parametrize("mutation", (None, "snapshot_owner", "snapshot_content", "rules_hash"))
@pytest.mark.parametrize("complete_state", [False, True])
@pytest.mark.parametrize("paired_quality", [False, True])
def test_playing_quality_audit_projects_all_decisions_without_private_state(
    monkeypatch: pytest.MonkeyPatch,
    rules,
    mutation,
    complete_state,
    paired_quality,
) -> None:
    import baloot_platform.research.service as module

    escalation_artifact_hash = "e" * 64
    escalation_quality_hash = "f" * 64
    auction_rows = _rows("bid", 6)
    project_play_rows = _rows("project", 4) + _rows("play", 32)
    mode = "exact" if complete_state or paired_quality else "bounded"
    visibility = "complete_state" if complete_state or paired_quality else "actor_information_set"
    completion = "exact_complete" if mode == "exact" else "bounded_complete"
    scope = "exact_fixed_deal_terminal_tail" if mode == "exact" else "supplied_worlds_only"
    def artifact_for(row):
        return {"request": {"mode": mode, "visibility": visibility},
            "proof": {"completionStatus": completion, "completionScope": scope,
                "rootActions": [{"action": row["action"]},
                    *([] if row["action"] == "play-0" else [{"action": "other"}])]},
            "proofModel": {}, "graph": {}}
    receipt = {
        "scope": "all_decisions",
        "status": "complete",
        "operation": "generate",
        "rulesHash": rules.sha256,
        "sha256": "b" * 64,
        "auction": {
            "decisions": auction_rows,
            "artifacts": {
                row["proofArtifactHash"]: artifact_for(row)
                for row in auction_rows
            },
        },
        "postAuction": {
            "escalation": {
                "operation": "generate",
                "recordedEvents": None,
                "decisions": [
                    {"actor": "west", "artifactHash": escalation_artifact_hash}
                ],
                "artifacts": {
                    escalation_artifact_hash: {
                        "request": {"mode": mode, "visibility": visibility},
                        "proof": {"completionStatus": completion, "completionScope": scope,
                            "selectedAction": "decline", "rootActions": [
                            {"action": "decline"}, {"action": "accept"}
                        ]},
                        "proofModel": {
                            "actionQuality": {
                                "actualActionOptimal": True,
                                "regret": 0,
                                "sha256": escalation_quality_hash,
                            }
                        },
                        "graph": {},
                    }
                },
            },
            "projectPlay": {
                "decisions": project_play_rows,
                "artifacts": {
                    row["proofArtifactHash"]: artifact_for(row)
                    for row in project_play_rows
                },
            },
        },
    }
    if paired_quality:
        for row in project_play_rows:
            row["optimalActionCount"] = None
            artifact = receipt["postAuction"]["projectPlay"]["artifacts"][row["proofArtifactHash"]]
            for action in artifact["proof"]["rootActions"]:
                action.update(value=None, satisfies=None)
            artifact["proof"]["completionScope"] = "selected_play_fixed_state_terminal_continuation"
            artifact["proofModel"] = {"kind": "native_complete_state_policy_pair_quality_v1",
                "actionQuality": {"optimalActionsComplete": False}}
    trace = {"playingQualityEvidence": {"receipt": receipt}}
    trace_hash = canonical_sha256(trace)
    owner_hash = "c" * 64
    snapshot = {"bundleHash": "e" * 64,
        "components": {"rules": {"payload": rules.data, "sha256": rules.sha256}}}
    snapshot["sha256"] = canonical_sha256(snapshot)
    result = {
        "resultCode": "LAF-QUALITY001",
        "resultHash": "d" * 64,
        "provenance": {
            "resultIdentity": {
                "kind": ORTOOLS_RESULT_IDENTITY_V3_KIND,
                "executorSemanticHash": owner_hash,
                "configurationSnapshotHash": snapshot["sha256"],
            }
        },
        "researchAuthority": {"traceHash": trace_hash, "bundleHash": snapshot["bundleHash"]},
        "authorityArtifacts": {"configurationSnapshot": {"uri": "cas://snapshot"}},
        "artifacts": {"completeTrace": {"uri": "cas://quality"}},
    }
    validated: list[str] = []

    def get_result(code):
        validated.append(code)
        return result if code == result["resultCode"] else None

    service = SimpleNamespace(
        get_result=get_result,
        artifact_store=object(),
    )
    monkeypatch.setattr(
        module,
        "_load_owned_result_json_artifact_v1",
        lambda store, reference, **kwargs: ((snapshot, "wrong-owner" if mutation == "snapshot_owner" else result["resultHash"])
            if reference.get("uri") == "cas://snapshot" else (trace, owner_hash)),
    )

    if mutation == "snapshot_content":
        snapshot["unexpected"] = True
    elif mutation == "rules_hash":
        snapshot["components"]["rules"]["sha256"] = "0" * 64
        snapshot["sha256"] = canonical_sha256({key: value for key, value in snapshot.items() if key != "sha256"})
        result["provenance"]["resultIdentity"]["configurationSnapshotHash"] = snapshot["sha256"]
    if mutation is not None:
        with pytest.raises(ValueError, match="frozen (configuration|rules) mismatch"):
            ResearchFoundryServiceV1.get_playing_quality_audit(service, result["resultCode"])
        assert validated == [result["resultCode"]]
        return
    audit = ResearchFoundryServiceV1.get_playing_quality_audit(
        service, result["resultCode"]
    )

    assert audit["decisionCount"] == audit["optimalDecisionCount"] == 43
    assert audit["wholeRoundCertified"] is False
    assert audit["zeroRegretDecisionCount"] == 43
    assert audit["decisions"][0]["proofMode"] == mode
    assert audit["decisions"][0]["proofVisibility"] == visibility
    assert audit["decisions"][0]["proofCompletionStatus"] == completion
    assert audit["decisions"][0]["proofScope"] == scope
    assert audit["decisions"][11]["legalActionCount"] == 1
    assert audit["decisions"][12]["legalActionCount"] == 2
    assert audit["countsByKind"] == {
        "bid": 6,
        "escalation": 1,
        "project": 4,
        "play": 32,
    }
    assert validated == [result["resultCode"]]
    assert audit["sha256"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "sha256"}
    )
    assert not ({"legalActions", "hands", "artifacts", "proofModel"} & audit.keys())
    assert all(
        not ({"legalActions", "hands", "alternatives", "proofModel"} & row.keys())
        for row in audit["decisions"]
    )

    policy_artifact = receipt["postAuction"]["projectPlay"]["artifacts"][
        project_play_rows[5]["proofArtifactHash"]
    ]
    policy_artifact["proof"]["completionScope"] = "action_optimality_only"
    policy_artifact["proofModel"]["actionQuality"] = {
        "scope": "observer_safe_policy_conditioned"
    }
    result["researchAuthority"]["traceHash"] = canonical_sha256(trace)
    policy_audit = ResearchFoundryServiceV1.get_playing_quality_audit(
        service, result["resultCode"]
    )
    assert policy_audit["decisions"][12]["proofScope"] == "observer_safe_policy_conditioned"
    policy_artifact["proof"]["completionScope"] = scope

    featured_row = project_play_rows[4]
    featured_row.update({"playIndex": 0, "exactStateId": "STATE-" + "A" * 20})
    featured_artifact = receipt["postAuction"]["projectPlay"]["artifacts"][featured_row["proofArtifactHash"]]
    featured_artifact.update({
        "request": {"visibility": "actor_information_set", "mode": "bounded"},
        "proofModel": {"kind": "native_observed_root_outcome_v1", "modelProto": "private"},
        "proof": {"rootActions": [{"action": "play-0"}, {"action": "other"}],
                  "completionStatus": "bounded_complete",
                  "completionScope": "supplied_worlds_only",
                  "hiddenWorlds": {"method": "supplied_worlds_only", "count": 6,
                                   "complete": False, "worldState": "private"}},
    })
    if complete_state:
        featured_artifact["request"] = {"visibility": "complete_state", "mode": "exact"}
        featured_artifact["proofModel"].update(kind="native_complete_state_root_outcome_v1",
            rootActionCountScope="exact_fixed_deal_terminal_tail")
        featured_artifact["proof"]["completionStatus"] = "exact_complete"
        featured_artifact["proof"]["completionScope"] = "exact_fixed_deal_terminal_tail"
        featured_artifact["proof"].pop("hiddenWorlds")
    receipt["qualityExecution"] = {"featuredPlayNativeProof": True}
    trace["exactTracePrefix"] = {"test": "prefix"}
    trace["exactNormalization"] = {"exactStateId": featured_row["exactStateId"]}
    monkeypatch.setattr(module.ExactTracePrefixV1, "build",
                        lambda raw: SimpleNamespace(play_actions=()))
    result["researchAuthority"]["traceHash"] = canonical_sha256(trace)
    featured_audit = ResearchFoundryServiceV1.get_playing_quality_audit(
        service, result["resultCode"]
    )
    if complete_state:
        assert "hiddenWorlds" not in featured_audit["featuredPlayProof"]
        assert featured_audit["featuredPlayProof"]["completionStatus"] == "exact_complete"
    else:
        assert featured_audit["featuredPlayProof"]["hiddenWorlds"] == {
            "method": "supplied_worlds_only", "count": 6, "complete": False,
        }
    assert "private" not in str(featured_audit)
    from baloot_platform.service.api import ResearchPlayingQualityAuditV1Response
    ResearchPlayingQualityAuditV1Response.model_validate(featured_audit)

    if not complete_state:
        original = copy.deepcopy(featured_artifact)
        featured_artifact["request"]["mode"] = "exact"
        featured_artifact["proofModel"]["kind"] = "actor_information_set_cegis_model_v1"
        featured_artifact["allocationModel"] = {
            "reachability": "deal_auction_escalation_project_play_cp_sat_v1"}
        featured_artifact["proof"].update(completionStatus="exact_complete",
            completionScope="exhaustive_actor_information_set",
            hiddenWorlds={"count": 6, "complete": True, "coverage": {
                "method": "exhaustive_cp_sat_hidden_allocations_v1",
                "observerInformationHash": "private-observation", "scopeHash": "private-scope"}})
        result["researchAuthority"]["traceHash"] = canonical_sha256(trace)
        native_audit = ResearchFoundryServiceV1.get_playing_quality_audit(service, result["resultCode"])
        assert native_audit["featuredPlayProof"]["hiddenWorlds"] == {
            "method": "exhaustive_cp_sat_hidden_allocations_v1", "count": 6, "complete": True}
        assert "private" not in str(native_audit)
        ResearchPlayingQualityAuditV1Response.model_validate(native_audit)
        featured_artifact.clear()
        featured_artifact.update(original)
        result["researchAuthority"]["traceHash"] = canonical_sha256(trace)

    first_play_hash = project_play_rows[4]["proofArtifactHash"]
    first_proof = receipt["postAuction"]["projectPlay"]["artifacts"][first_play_hash]["proof"]
    featured_mode = featured_artifact["request"]["mode"]
    first_proof["completionStatus"] = (
        "bounded_complete" if featured_mode == "exact" else "exact_complete"
    )
    result["researchAuthority"]["traceHash"] = canonical_sha256(trace)
    with pytest.raises(ValueError, match="proof boundary is inconsistent"):
        ResearchFoundryServiceV1.get_playing_quality_audit(service, result["resultCode"])
    first_proof["completionStatus"] = (
        "exact_complete" if featured_mode == "exact" else "bounded_complete"
    )
    receipt["postAuction"]["projectPlay"]["artifacts"][first_play_hash]["proof"]["rootActions"] = []
    result["researchAuthority"]["traceHash"] = canonical_sha256(trace)
    with pytest.raises(ValueError, match="proof roots are unavailable"):
        ResearchFoundryServiceV1.get_playing_quality_audit(
            service, result["resultCode"]
        )


def test_playing_quality_audit_rejects_substituted_trace_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import baloot_platform.research.service as module

    trace = {"playingQualityEvidence": {"receipt": {}}}
    result = {
        "resultCode": "LAF-QUALITY002",
        "resultHash": "d" * 64,
        "provenance": {
            "resultIdentity": {
                "kind": ORTOOLS_RESULT_IDENTITY_V3_KIND,
                "executorSemanticHash": "a" * 64,
            }
        },
        "researchAuthority": {"traceHash": canonical_sha256(trace)},
        "artifacts": {"completeTrace": {"uri": "cas://quality"}},
    }
    service = SimpleNamespace(get_result=lambda code: result, artifact_store=object())
    monkeypatch.setattr(
        module,
        "_load_owned_result_json_artifact_v1",
        lambda *args, **kwargs: (trace, "b" * 64),
    )

    with pytest.raises(ValueError, match="complete-trace authority mismatch"):
        ResearchFoundryServiceV1.get_playing_quality_audit(
            service, result["resultCode"]
        )
