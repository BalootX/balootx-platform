from __future__ import annotations


from research_audit_trace_helpers import (
    _Harness,
    _payload,
    _store_validated_legacy_recipe,
)
from research_audit_trace_helpers import service_harness as service_harness


def test_ortools_submission_freezes_canonical_generator_identity(
    service_harness: _Harness,
) -> None:
    research = service_harness.research
    payload = _payload(
        research,
        label="submission-generator-identity",
        lane="ortools",
    )
    payload["schemaVersion"] = 3
    payload.pop("profiles")
    payload["tactics"] = []
    payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": service_harness.container.configuration.resolve()[
            "bundle"
        ]["revisionId"],
    }
    recipe = _store_validated_legacy_recipe(research, payload)

    run = research._submit_recipe_run(
        recipe["code"],
        idempotency_key="submission-generator-identity",
        schedule=False,
    )

    identity = research._deal_generator_identity()
    assert run["generationLane"] == "ortools"
    assert run["dealGeneratorIdentity"] == identity
    assert run["dealGeneratorDescriptorHash"] == identity["descriptorHash"]
