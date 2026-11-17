from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .family_catalog import TAG_AXES, TagQueryV1
from .scenario_contracts import canonical_sha256
from .result_authority import (
    ORTOOLS_RESULT_IDENTITY_KIND,
    ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION,
    ORTOOLS_RESULT_IDENTITY_V3_KIND,
    ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION,
    validate_ortools_result_sections_v1,
)


RESULT_PROJECTION_VERSION = 5


FACET_FILTERS = {
    "tactical_tag": "tactical_tag",
    "acting_seat": "acting_seat",
    "observer_seat": "observer_seat",
    "contract": "contract",
    "project": "project",
    "generation_lane": "generation_lane",
    "scenario_template": "scenario_template",
    "scenario_origin": "scenario_origin",
    "scenario_status": "scenario_status",
    "requested_proof_mode": "requested_proof_mode",
    "achieved_proof_mode": "achieved_proof_mode",
    "proof_completion": "proof_completion",
    "proof_visibility": "proof_visibility",
    "target_team": "target_team",
    "contract_outcome": "contract_outcome",
    "project_kind": "project_kind",
    "project_value": "project_value",
    "project_state": "project_state",
    "winning_root_action_band": "winning_root_action_band",
    "exact_state_available": "exact_state_available",
    "decision_kind": "decision_kind",
    "research_backend": "research_backend",
    "qualification_authority": "qualification_authority",
    "runtime_compatibility_status": "runtime_compatibility_status",
    "configuration_bundle": "configuration_bundle",
    "exact_state_id": "exact_state_id",
    "scenario_category": "scenario_category",
    "variant_collection": "variant_collection",
}

GROUP_FIELDS = frozenset(
    {
        *FACET_FILTERS,
        "truth_label",
        "disposition",
    }
)

SORT_ORDERS = {
    "score_desc": "{alias}.score DESC,{alias}.result_code ASC",
    "score_asc": "{alias}.score ASC,{alias}.result_code ASC",
    "confidence_desc": "{alias}.confidence DESC,{alias}.result_code ASC",
    "confidence_asc": "{alias}.confidence ASC,{alias}.result_code ASC",
    "newest": "{alias}.created_at DESC,{alias}.result_code ASC",
    "oldest": "{alias}.created_at ASC,{alias}.result_code ASC",
    "result_code_asc": "{alias}.result_code ASC",
    "result_code_desc": "{alias}.result_code DESC",
}


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        sorted(
            {
                str(item).strip().lower()
                for item in value
                if str(item).strip()
            }
        )
    )


def normalize_timestamp(value: Any, *, fallback: str | None = None) -> str | None:
    if value is None or str(value).strip() == "":
        return fallback
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def validate_stored_result_identity_v1(
    payload: Mapping[str, Any],
    *,
    result_code: Any,
    result_hash: Any,
    run_code: Any,
) -> dict[str, Any]:
    """Fail closed when a stored result payload is moved to another CAS row."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("stored research result payload is invalid")
    value = dict(payload)
    if (
        value.get("resultCode") != str(result_code)
        or value.get("resultHash") != str(result_hash)
        or value.get("runCode") != str(run_code)
    ):
        raise RuntimeError("stored research result identity mismatch")
    return value


def stored_result_payload_hash_v1(payload: Mapping[str, Any]) -> str:
    """Hash the complete immutable result record, including non-semantic evidence."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("stored research result payload is invalid")
    return canonical_sha256(dict(payload))


def stored_result_library_projection_hash_v1(
    projection: Mapping[str, Any], *, payload_hash: Any
) -> str:
    """Bind one compact library projection to its immutable result payload."""

    if not isinstance(projection, Mapping) or type(payload_hash) is not str:
        raise RuntimeError("stored research result library projection is invalid")
    return canonical_sha256(
        {"payloadHash": payload_hash, "projection": dict(projection)}
    )


def validate_stored_result_library_projection_hash_v1(
    projection: Mapping[str, Any],
    *,
    result_code: Any,
    result_hash: Any,
    run_code: Any,
    payload_hash: Any,
    projection_hash: Any,
) -> dict[str, Any]:
    """Validate a compact projection without loading its large source payload."""

    value = validate_stored_result_identity_v1(
        projection,
        result_code=result_code,
        result_hash=result_hash,
        run_code=run_code,
    )
    if (
        type(projection_hash) is not str
        or stored_result_library_projection_hash_v1(
            value, payload_hash=payload_hash
        )
        != projection_hash
    ):
        raise RuntimeError("stored research result library projection mismatch")
    return value


def validate_stored_result_record_v1(
    payload: Mapping[str, Any],
    *,
    result_code: Any,
    result_hash: Any,
    run_code: Any,
    payload_hash: Any,
) -> dict[str, Any]:
    """Bind the full immutable payload while preserving its semantic result hash."""

    value = validate_stored_result_identity_v1(
        payload,
        result_code=result_code,
        result_hash=result_hash,
        run_code=run_code,
    )
    if (
        type(payload_hash) is not str
        or stored_result_payload_hash_v1(value) != payload_hash
    ):
        raise RuntimeError("stored research result payload hash mismatch")
    return value


def _recipe_scenario_projection_v1(
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the recipe-owned subset of the V4 scenario projection."""

    predicates: list[Mapping[str, Any]] = []

    def collect(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        if "predicateId" in value or "kind" in value and "arguments" in value:
            predicates.append(value)
            return
        children = value.get("children")
        if isinstance(children, list):
            for child in children:
                collect(child)

    collect(scenario.get("constraints"))
    collect(scenario.get("outcome"))
    project_kinds: set[str] = set()
    project_values: set[str] = set()
    project_states: set[str] = set()
    for predicate in predicates:
        if not str(predicate.get("kind") or "").startswith("project."):
            continue
        arguments = predicate.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        if arguments.get("kind"):
            project_kinds.add(str(arguments["kind"]))
        if arguments.get("value") is not None:
            project_values.add(str(arguments["value"]))
        if arguments.get("state"):
            project_states.add(str(arguments["state"]))
    scenario_hash = str(scenario.get("sha256") or "")
    origin = scenario.get("origin")
    return {
        "scenarioTemplateId": f"SCN-{scenario_hash[:12].upper()}",
        "origin": (
            str(origin.get("kind") or "") if isinstance(origin, Mapping) else ""
        ),
        "projectKinds": sorted(project_kinds),
        "projectValues": sorted(project_values),
        "projectStates": sorted(project_states),
    }


def validate_scenario_result_authority_v1(
    payload: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one internally valid V4 result to its stored run and recipe."""

    value = dict(payload)
    provenance = dict(value.get("provenance") or {})
    identity = dict(provenance.get("resultIdentity") or {})
    run_value = dict(run)
    recipe_value = dict(recipe)
    configuration = dict(recipe_value.get("configuration") or {})
    scenario = dict(recipe_value.get("scenario") or {})
    proof_request = dict(recipe_value.get("proof") or {})
    satisfaction = dict(value.get("scenarioSatisfaction") or {})
    constraint = dict(value.get("constraintEvidence") or {})
    proof = dict(value.get("proofOutcome") or {})
    artifacts = dict(value.get("artifacts") or {})
    proof_reference = dict(artifacts.get("proofOutcome") or {})
    proof_artifact_hash = str(
        proof.get("artifactSha256") or proof_reference.get("sha256") or ""
    )
    proof_semantic_hash = proof.get("semanticSha256")
    if proof_semantic_hash is None and "artifactSha256" not in proof:
        proof_semantic_hash = proof.get("sha256")
    proof_identities = dict(proof.get("identities") or {})
    public_proof = "artifactSha256" in proof
    projection = dict(value.get("scenarioProjection") or {})
    qualification = dict(value.get("qualification") or {})
    score_evidence = dict(value.get("scoreEvidence") or {})

    if public_proof:
        public_proof_hash = str(proof.get("sha256") or "")
        if (
            len(proof_artifact_hash) != 64
            or any(character not in "0123456789abcdef" for character in proof_artifact_hash)
            or canonical_sha256(
                {key: item for key, item in proof.items() if key != "sha256"}
            )
            != public_proof_hash
        ):
            raise RuntimeError("stored Research V4 public proof integrity mismatch")
        if (
            proof_reference.get("sha256") != proof_artifact_hash
            or dict(identity.get("artifactHashes") or {}).get("proofOutcome")
            != proof_artifact_hash
        ):
            raise RuntimeError("stored Research V4 proof artifact authority mismatch")
    current_identity = all(
        identity.get(key) is not None
        for key in ("scenarioProjectionHash", "scoreEvidenceHash", "verdictHash")
    )
    if current_identity and proof_semantic_hash is None:
        raise RuntimeError("stored Research V4 proof semantic identity is missing")

    result_hash = str(value.get("resultHash") or "")
    expected_result_code = f"LAF-{result_hash[:10].upper()}"
    legacy_identity = (
        identity.get("schemaVersion") == 1
        and identity.get("kind") == "research_scenario_result_identity_v1"
    )
    ortools_identity = (
        (
            identity.get("schemaVersion") == ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION
            and identity.get("kind") == ORTOOLS_RESULT_IDENTITY_KIND
        )
        or (
            identity.get("schemaVersion") == ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION
            and identity.get("kind") == ORTOOLS_RESULT_IDENTITY_V3_KIND
        )
    )
    if (
        type(value.get("schemaVersion")) is not int
        or value.get("schemaVersion") != 2
        or not (legacy_identity or ortools_identity)
        or value.get("resultCode") != expected_result_code
        or canonical_sha256(identity) != result_hash
        or value.get("runCode") != identity.get("runCode")
        or identity.get("runCode") != run_value.get("runCode")
        or identity.get("runHash") != run_value.get("runHash")
        or value.get("recipeCode")
        != identity.get("recipeCode")
        or identity.get("recipeCode") != run_value.get("recipeCode")
        or run_value.get("recipeCode") != recipe_value.get("code")
        or identity.get("recipeHash") != run_value.get("recipeHash")
        or run_value.get("recipeHash") != provenance.get("recipeHash")
        or provenance.get("recipeHash") != recipe_value.get("sha256")
        or value.get("createdAt") != run_value.get("createdAt")
    ):
        raise RuntimeError("stored Research V4 result owner identity mismatch")
    if run_value.get("configurationAuthority") != configuration:
        raise RuntimeError("stored Research V4 bundle authority mismatch")
    if ortools_identity:
        validate_ortools_result_sections_v1(value, recipe=recipe_value)
    if run_value.get("status") == "completed":
        result_codes = (
            run_value.get("resultCodes")
            if bool(qualification.get("qualified"))
            else run_value.get("auditResultCodes")
        )
        expected_count = (
            run_value.get("resultCount")
            if bool(qualification.get("qualified"))
            else run_value.get("auditResultCount")
        )
        legacy_membership_unrecorded = legacy_identity and all(
            run_value.get(key) is None
            for key in ("resultCodes", "auditResultCodes")
        )
        if not legacy_membership_unrecorded and (
            not isinstance(result_codes, list)
            or value.get("resultCode") not in result_codes
            or expected_count != len(result_codes)
        ):
            raise RuntimeError("stored Research V4 completed-run membership mismatch")

    # The new backend is a closed, engine-free qualification contract.  Its
    # authority validator above owns all semantic checks; continuing into the
    # historical runtime-proof verifier would create a second authority.
    if ortools_identity:
        return value

    authority_hash = canonical_sha256(configuration)
    bundle_hash = str(configuration.get("semanticBundleHash") or "")
    rules_hash = str(
        dict(dict(configuration.get("identities") or {}).get("rules") or {}).get(
            "sha256"
        )
        or ""
    )
    if (
        identity.get("configurationAuthorityHash") != authority_hash
        or provenance.get("configurationAuthorityHash") != authority_hash
        or proof_identities.get("bundleHash") != bundle_hash
        or proof_identities.get("rulesHash") != rules_hash
    ):
        raise RuntimeError("stored Research V4 bundle authority mismatch")

    scenario_hash = str(scenario.get("sha256") or "")
    exact_state_id = projection.get("exactStateId")
    if (
        identity.get("scenarioHash") != scenario_hash
        or provenance.get("scenarioHash") != scenario_hash
        or satisfaction.get("scenarioHash") != scenario_hash
        or constraint.get("scenarioHash") != scenario_hash
        or provenance.get("exactStateId") != exact_state_id
        or satisfaction.get("exactStateId") != exact_state_id
        or constraint.get("exactStateId") != exact_state_id
        or identity.get("exactStateHash") != constraint.get("exactStateHash")
    ):
        raise RuntimeError("stored Research V4 scenario authority mismatch")
    expected_recipe_projection = _recipe_scenario_projection_v1(scenario)
    if any(
        projection.get(key) != expected
        for key, expected in expected_recipe_projection.items()
    ):
        raise RuntimeError("stored Research V4 scenario projection authority mismatch")

    expected_strategy_class = proof_request.get("strategyClass") or (
        "frozen_policy"
        if proof_request.get("mode") == "policy_conditioned"
        else "pure_deterministic"
    )
    expected_proof = {
        "requestedMode": proof_request.get("mode"),
        "requestedVisibility": proof_request.get("visibility"),
        "strategyClass": expected_strategy_class,
        "claim": proof_request.get("claim"),
        "targetTeam": proof_request.get("targetTeam"),
        "utility": proof_request.get("utility"),
        "successThreshold": proof_request.get("successThreshold"),
        "horizon": proof_request.get("horizon"),
        "boundaryEvaluator": proof_request.get("boundaryEvaluator"),
    }
    legacy_strategy_class_implicit = (
        legacy_identity
        and proof.get("strategyClass") is None
        and proof_identities.get("strategyClass") is None
    )
    requested_policies = proof_request.get("policies")
    expected_policy_hashes = (
        {
            str(seat): str(dict(policy).get("profileHash") or "")
            for seat, policy in dict(requested_policies).items()
            if isinstance(policy, Mapping)
        }
        if isinstance(requested_policies, Mapping)
        else {}
    )
    admission_proof = dict(
        dict(dict(run_value.get("launchAdmission") or {}).get("estimate") or {}).get(
            "proof"
        )
        or {}
    )
    if (
        any(
            proof.get(key) != expected
            for key, expected in expected_proof.items()
            if key != "strategyClass" or not legacy_strategy_class_implicit
        )
        or (
            proof_request.get("selectedAction") is not None
            and proof.get("selectedAction") != proof_request.get("selectedAction")
        )
        or (
            proof_request.get("lineUniverse") is not None
            and dict(proof.get("continuationLines") or {}).get("lineUniverse")
            != proof_request.get("lineUniverse")
        )
        or (
            not public_proof
            and proof_identities.get("stateHash") != value.get("traceHash")
        )
        or (
            not public_proof
            and proof_identities.get("exactStateId") != exact_state_id
        )
        or dict(proof_identities.get("policyHashes") or {})
        != expected_policy_hashes
        or (
            not public_proof
            and proof_identities.get("strategyClass") != expected_strategy_class
            and not legacy_strategy_class_implicit
        )
        or (
            not public_proof
            and proof_identities.get("proofRequestHash")
            != proof_request.get("sha256")
        )
        or (
            admission_proof
            and (
                admission_proof.get("budgetProfile")
                != proof_request.get("budgetProfile")
                or proof_identities.get("proofProfileHash")
                != admission_proof.get("budgetProfileHash")
                or proof_identities.get("proofBudgetRegistryHash")
                != admission_proof.get("proofBudgetRegistryHash")
            )
        )
        or (
            proof_semantic_hash is not None
            and identity.get("proofOutcomeHash") != proof_semantic_hash
        )
        or identity.get("scenarioSatisfactionHash") != satisfaction.get("sha256")
        or identity.get("constraintEvidenceHash") != constraint.get("sha256")
        or score_evidence.get("scenarioSatisfactionHash")
        not in {None, satisfaction.get("sha256")}
        or (
            proof_semantic_hash is not None
            and score_evidence.get("proofOutcomeHash")
            not in {None, proof_semantic_hash}
        )
    ):
        raise RuntimeError("stored Research V4 proof authority mismatch")

    completion = str(proof.get("completionStatus") or "")
    proof_complete = completion in {
        "exact_complete",
        "bounded_complete",
        "policy_complete",
    }
    legacy_observer_safety = (
        qualification.get("legacyObserverSafetyUnverified") is True
    )
    observer_evidence = qualification.get("observerSafetyEvidence")
    observer_safe = (
        isinstance(observer_evidence, Mapping)
        and observer_evidence.get("passed") is True
        and not legacy_observer_safety
    )
    scenario_satisfied = satisfaction.get("status") == "satisfied"
    qualified = scenario_satisfied and proof_complete and observer_safe
    historical_qualified = scenario_satisfied and proof_complete
    expected_score = 100.0 if (
        historical_qualified if legacy_observer_safety else qualified
    ) else 0.0
    expected_failures: list[str] = []
    if not scenario_satisfied:
        expected_failures.append(str(satisfaction.get("status") or ""))
    if not proof_complete:
        expected_failures.append(completion)
    if not observer_safe:
        expected_failures.append("observer_safety_unverified")
    expected_truth = (
        "exact_optimal"
        if completion == "exact_complete"
        else "robust_best_known"
        if completion in {"bounded_complete", "policy_complete"}
        else "rejected"
    )
    if (
        value.get("classification") != "scenario_v4"
        or value.get("truthLabel") != expected_truth
        or value.get("confidence") != (1.0 if proof_complete else 0.0)
        or value.get("score") != expected_score
        or score_evidence.get("score") != expected_score
        or value.get("disposition") != ("qualified" if qualified else "rejected")
        or qualification.get("qualified") is not qualified
        or qualification.get("observerSafe") is not observer_safe
        or qualification.get("decisionPointAuditComplete") is not proof_complete
        or qualification.get("scenarioSatisfactionHash") != satisfaction.get("sha256")
        or (
            proof_semantic_hash is not None
            and qualification.get("proofOutcomeHash") != proof_semantic_hash
        )
        or list(qualification.get("failures") or ()) != expected_failures
        or value.get("decisionCount")
        != (1 if list(proof.get("rootActions") or ()) else 0)
        or list(value.get("criticalDecisionIds") or ()) != [exact_state_id]
    ):
        raise RuntimeError("stored Research V4 verdict authority mismatch")
    return value


def result_library_safe_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fail-closed V4 observer-safety view to derived library data."""

    value = dict(payload)
    if value.get("schemaVersion") != 2:
        return value
    qualification = dict(value.get("qualification") or {})
    evidence = qualification.get("observerSafetyEvidence")
    identity = dict(dict(value.get("provenance") or {}).get("resultIdentity") or {})
    if (
        (
            identity.get("schemaVersion") == ORTOOLS_RESULT_IDENTITY_SCHEMA_VERSION
            and identity.get("kind") == ORTOOLS_RESULT_IDENTITY_KIND
        )
        or (
            identity.get("schemaVersion") == ORTOOLS_RESULT_IDENTITY_V3_SCHEMA_VERSION
            and identity.get("kind") == ORTOOLS_RESULT_IDENTITY_V3_KIND
        )
    ):
        return value
    score_evidence = dict(value.get("scoreEvidence") or {})
    evidence_hash = evidence.get("sha256") if isinstance(evidence, Mapping) else None
    evidence_valid = (
        isinstance(evidence, Mapping)
        and canonical_sha256(
            {key: item for key, item in evidence.items() if key != "sha256"}
        )
        == evidence_hash
        and identity.get("observerSafetyEvidenceHash") == evidence_hash
        and score_evidence.get("observerSafetyEvidenceHash") == evidence_hash
    )
    if evidence_valid:
        return value

    failures = list(qualification.get("failures") or ())
    if "observer_safety_unverified" not in failures:
        failures.append("observer_safety_unverified")
    qualification.update(
        {
            "qualified": False,
            "failures": failures,
            "observerSafe": False,
            "observerSafetyEvidence": None,
            "legacyObserverSafetyUnverified": evidence is None,
        }
    )
    value.update(
        {
            "disposition": "rejected",
            "publicationEligible": False,
            "qualificationFailures": failures,
            "qualification": qualification,
        }
    )
    return value


def result_library_projection(
    payload: Mapping[str, Any], *, created_at: str
) -> dict[str, Any]:
    """Materialize the compact immutable fields needed by the Foundry library."""

    value = result_library_safe_payload(payload)
    artifacts = dict(value.get("artifacts") or {})
    complete_trace = artifacts.get("completeTrace")
    exact_state = artifacts.get("exactState")
    return {
        "schemaVersion": value.get("schemaVersion"),
        "resultCode": value.get("resultCode"),
        "resultHash": value.get("resultHash"),
        "runCode": value.get("runCode"),
        "traceHash": value.get("traceHash"),
        "gameHash": value.get("gameHash"),
        "truthLabel": value.get("truthLabel"),
        "confidence": float(value.get("confidence", 0.0)),
        "score": float(value.get("score", 0.0)),
        "disposition": value.get("disposition"),
        "publicationEligible": value.get("publicationEligible") is True,
        "qualificationFailures": value.get("qualificationFailures") or [],
        "qualification": value.get("qualification") or {},
        "classification": value.get("classification"),
        "tacticalTags": value.get("tacticalTags") or [],
        "verifiedTacticalTags": value.get("verifiedTacticalTags") or [],
        "interestingSeats": value.get("interestingSeats") or [],
        "libraryProjection": value.get("libraryProjection") or {},
        "scenarioProjection": value.get("scenarioProjection") or {},
        "artifacts": {
            **({"completeTrace": complete_trace} if complete_trace else {}),
            **({"exactState": exact_state} if exact_state else {}),
        },
        "createdAt": created_at,
    }


def validate_stored_result_library_projection_v1(
    projection: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    result_code: Any,
    result_hash: Any,
    run_code: Any,
    payload_hash: Any,
    created_at: Any,
) -> dict[str, Any]:
    """Bind a compact projection to the immutable payload it summarizes."""

    stored = validate_stored_result_record_v1(
        payload,
        result_code=result_code,
        result_hash=result_hash,
        run_code=run_code,
        payload_hash=payload_hash,
    )
    projected = validate_stored_result_identity_v1(
        projection,
        result_code=result_code,
        result_hash=result_hash,
        run_code=run_code,
    )
    try:
        fallback_created_at = normalize_timestamp(created_at) or str(created_at)
    except ValueError:
        fallback_created_at = str(created_at)
    try:
        normalized_created_at = normalize_timestamp(
            stored.get("createdAt"), fallback=fallback_created_at
        )
    except ValueError:
        normalized_created_at = fallback_created_at
    expected = result_library_projection(
        result_library_safe_payload(stored),
        created_at=str(normalized_created_at),
    )
    if projected != expected:
        raise RuntimeError("stored research result library projection mismatch")
    return projected


def validate_result_lesson_tags_v1(document: Any) -> TagQueryV1:
    """Validate stored editorial tags without duplicating their proof authority."""
    required = {
        "schemaVersion", "tags", "sources", "playIndex",
        "qualificationEvidence", "sha256",
    }
    if (
        not isinstance(document, Mapping)
        or required - set(document)
        or set(document) - required - {"classifier", "qualityEvidenceHash"}
        or type(document.get("schemaVersion")) is not int
        or document["schemaVersion"] != 1
        or document.get("qualificationEvidence") is not False
        or not isinstance(document.get("sources"), Mapping)
        or type(document.get("playIndex")) is not int
        or not 16 <= document["playIndex"] <= 27
        or not isinstance(document.get("tags"), Mapping)
        or (
            document.get("classifier") is not None
            and not isinstance(document["classifier"], Mapping)
        )
        or (
            "qualityEvidenceHash" in document
            and (
                type(document["qualityEvidenceHash"]) is not str
                or len(document["qualityEvidenceHash"]) != 64
                or any(character not in "0123456789abcdef" for character in document["qualityEvidenceHash"])
            )
        )
        or canonical_sha256({key: value for key, value in document.items() if key != "sha256"})
        != document.get("sha256")
    ):
        raise ValueError("research result lesson tags document is invalid")
    tags = document["tags"]
    if any(
        axis not in TAG_AXES
        or not isinstance(values, list)
        or any(type(value) is not str for value in values)
        or len(values) != len(set(values))
        for axis, values in tags.items()
    ):
        raise ValueError("research result lesson tags axes are invalid")
    # Derived documents retain unproven empty axes; queries select only proven values.
    return TagQueryV1.build({axis: values for axis, values in tags.items() if values})


def result_facets(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Build query-only projections from immutable result evidence."""

    value = dict(payload)
    projection = dict(value.get("libraryProjection") or {})
    provenance = dict(value.get("provenance") or {})
    tactical_tags = _strings(
        [
            *(value.get("tacticalTags") or ()),
            *(value.get("verifiedTacticalTags") or ()),
        ]
    )
    contract = str(projection.get("contract") or "").strip().lower()
    lane = str(
        projection.get("generationLane")
        or provenance.get("generationLane")
        or ""
    ).strip().lower()
    scenario = dict(value.get("scenarioProjection") or {})
    proof = dict(value.get("proofOutcome") or {})
    authority = dict(value.get("researchAuthority") or {})
    compatibility = dict(value.get("runtimeCompatibility") or {})
    winning_count = scenario.get(
        "winningRootActionCount",
        proof.get("winningRootActionCount"),
    )
    if type(winning_count) is int and winning_count >= 0:
        winning_band = (
            "0"
            if winning_count == 0
            else "1"
            if winning_count == 1
            else "2_3"
            if winning_count <= 3
            else "4_plus"
        )
    else:
        winning_band = ""

    def one(key: str, fallback: Any = None) -> tuple[str, ...]:
        text = str(scenario.get(key, fallback) or "").strip().lower()
        return (text,) if text else ()

    result = {
        "tactical_tag": tactical_tags,
        "acting_seat": _strings(
            projection.get("actingSeats") or value.get("interestingSeats") or ()
        ),
        "observer_seat": _strings(projection.get("observerSeats") or ()),
        "contract": (contract,) if contract else (),
        "project": _strings(projection.get("projectKinds") or ()),
        "generation_lane": (lane,) if lane else (),
    }
    if authority:
        result.update(
            {
                "research_backend": _strings([authority.get("backendId")]),
                "qualification_authority": _strings(
                    [authority.get("qualificationAuthority")]
                ),
                "configuration_bundle": _strings(
                    [authority.get("bundleRevisionId")]
                ),
            }
        )
    if compatibility:
        result["runtime_compatibility_status"] = _strings(
            [compatibility.get("status")]
        )
    run_code = str(value.get("runCode") or "").strip().lower()
    if run_code:
        result["variant_collection"] = (run_code,)
    if scenario or proof:
        result.update(
            {
                "scenario_template": one("scenarioTemplateId"),
                "scenario_origin": one("origin"),
                "scenario_status": one("status"),
                "requested_proof_mode": one(
                    "requestedProofMode", proof.get("requestedMode")
                ),
                "achieved_proof_mode": one(
                    "achievedProofMode", proof.get("achievedMode")
                ),
                "proof_completion": one(
                    "proofCompletion", proof.get("completionStatus")
                ),
                "proof_visibility": one(
                    "proofVisibility", proof.get("achievedVisibility")
                ),
                "target_team": one("targetTeam", proof.get("targetTeam")),
                "contract_outcome": one("contractOutcome"),
                "project_kind": _strings(scenario.get("projectKinds") or ()),
                "project_value": _strings(scenario.get("projectValues") or ()),
                "project_state": _strings(scenario.get("projectStates") or ()),
                "winning_root_action_band": (winning_band,) if winning_band else (),
                "exact_state_available": (
                    ("true",)
                    if scenario.get("exactStateAvailable") is True
                    or (
                        "exactStateAvailable" not in scenario
                        and bool(scenario.get("exactStateId"))
                    )
                    else (("false",) if scenario else ())
                ),
                "decision_kind": one("decisionKind"),
                "exact_state_id": one("exactStateId"),
                "scenario_category": _strings(
                    scenario.get("scenarioCategories") or ()
                ),
            }
        )
    if "lessonTags" in scenario:
        tags = validate_result_lesson_tags_v1(scenario["lessonTags"])
        result.update({f"lesson_tag_{axis}": values for axis, values in tags.selections})
    return result


@dataclass(frozen=True, slots=True)
class ResearchResultQueryV1:
    run_code: str | None = None
    truth_label: str | None = None
    disposition: str | None = None
    minimum_score: float | None = None
    search: str | None = None
    tactical_tag: str | None = None
    acting_seat: str | None = None
    observer_seat: str | None = None
    contract: str | None = None
    project: str | None = None
    generation_lane: str | None = None
    scenario_template: str | None = None
    scenario_origin: str | None = None
    scenario_status: str | None = None
    requested_proof_mode: str | None = None
    achieved_proof_mode: str | None = None
    proof_completion: str | None = None
    proof_visibility: str | None = None
    target_team: str | None = None
    contract_outcome: str | None = None
    project_kind: str | None = None
    project_value: str | None = None
    project_state: str | None = None
    winning_root_action_band: str | None = None
    exact_state_available: str | None = None
    decision_kind: str | None = None
    research_backend: str | None = None
    qualification_authority: str | None = None
    runtime_compatibility_status: str | None = None
    configuration_bundle: str | None = None
    exact_state_id: str | None = None
    scenario_category: str | None = None
    variant_collection: str | None = None
    tag_query: TagQueryV1 | Mapping[str, Any] | None = None
    created_from: str | None = None
    created_to: str | None = None
    group_by: str | None = None
    sort: str = "score_desc"
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        if self.sort not in SORT_ORDERS:
            raise ValueError("unsupported research result sort")
        if self.group_by is not None and self.group_by not in GROUP_FIELDS:
            raise ValueError("unsupported research result grouping")
        if int(self.offset) < 0:
            raise ValueError("research result offset must be non-negative")
        if not 1 <= int(self.limit) <= 500:
            raise ValueError("research result limit must be between 1 and 500")
        start = normalize_timestamp(self.created_from)
        end = normalize_timestamp(self.created_to)
        if start is not None and end is not None and start > end:
            raise ValueError("research result created_from must not exceed created_to")
        object.__setattr__(self, "created_from", start)
        object.__setattr__(self, "created_to", end)
        if self.tag_query is not None:
            object.__setattr__(self, "tag_query", TagQueryV1.build(
                self.tag_query.to_dict()
                if isinstance(self.tag_query, TagQueryV1) else self.tag_query
            ))

    @property
    def facet_filters(self) -> tuple[tuple[str, str], ...]:
        values = {
            name: getattr(self, name)
            for name in FACET_FILTERS
        }
        return tuple(
            (FACET_FILTERS[name], str(value).strip().lower())
            for name, value in values.items()
            if value is not None and str(value).strip()
        )

    def order_sql(self, alias: str) -> str:
        if not alias.replace("_", "").isalnum():
            raise ValueError("research result SQL alias is invalid")
        return SORT_ORDERS[self.sort].format(alias=alias)

    @property
    def tag_facet_filters(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (f"lesson_tag_{axis}", values)
            for axis, values in (self.tag_query.selections if self.tag_query else ())
        )
