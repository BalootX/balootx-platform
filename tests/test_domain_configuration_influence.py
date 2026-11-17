from __future__ import annotations

from baloot_platform.research import BalootDomainCompilerV1
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.service.container import build_container


def _alternative(value):
    if value is None:
        return "__influence_probe__"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "__influence_probe__"
    if isinstance(value, list):
        if len(value) > 1:
            return [*reversed(value)]
        if value and isinstance(value[0], str):
            return [*value, "__influence_probe__"]
        return [*value, None]
    raise TypeError(f"unsupported governed leaf type: {type(value).__name__}")


def test_value_hash_propagation_is_identity_evidence_not_behavioral_proof():
    container = build_container()
    domain = container.research.domain
    compiled = BalootDomainCompilerV1().compile(
        rules=domain.rules,
        conventions=domain.conventions,
        localization=domain.localization,
        bundle_hash=domain.bundle_hash,
        plugins=container.registry.list(),
    ).compiled
    payloads = compiled.to_dict(include_payloads=True)["targetPayloads"]
    target_names = {
        "runtime": "runtimePlan",
        "reference": "referencePlan",
        "cpSat": "cpSatBlueprint",
        "policy": "policyPlan",
        "documentation": "documentationModel",
    }
    indexes = {
        target: {
            str(item["primitiveId"]): item
            for item in payloads[name]["entries" if name != "documentationModel" else "items"]
        }
        for target, name in target_names.items()
    }
    ui_index = {
        str(item["id"]): item for item in payloads["uiSchema"]["fields"]
    }

    for row in compiled.domain_ir.primitive_instances:
        assert canonical_sha256(_alternative(row.value)) != row.value_hash
        assert ui_index[row.primitive_id]["valueHash"] == row.value_hash
        for target in row.covered_targets:
            if target in {"tests", "influence"}:
                continue
            if target == "documentation":
                assert row.primitive_id in indexes[target]
                continue
            entry = indexes[target][row.primitive_id]
            assert entry["valueHash"] == row.value_hash

        assert row.executable_binding_status == "covered"
        assert row.declared_behavioral_evidence == "tests/test_domain_configuration_influence.py"
        if row.behavioral_or_invariant_status == "unproven":
            assert row.verified_behavioral_evidence is None
            assert row.qualification_eligible is False
        elif row.behavioral_or_invariant_status == "partial_behavioral":
            assert row.primitive_id == "rules.contracts.allowed"
            assert row.verified_behavioral_evidence is not None
            assert row.qualification_eligible is False
        else:
            assert row.primitive_id.startswith(
                (
                    "rules.",
                    "localization.",
                    "conventions.",
                )
            )
            assert row.verified_behavioral_evidence is not None
            assert row.qualification_eligible is True


def test_declared_influence_file_is_not_used_as_executable_receipt():
    container = build_container()
    definitions = container.research.compiled_domain.compiled.definitions
    assert definitions
    assert all(
        item.influence_proof == "tests/test_domain_configuration_influence.py"
        for item in definitions
    )
    coverage = container.research.compiled_domain.compiled.coverage
    assert coverage.executable_binding_complete is True
    assert coverage.behavioral_or_invariant_complete is True
    assert coverage.behavioral_or_invariant_covered == coverage.total
    assert all(
        row.verified_behavioral_evidence
        and row.verified_behavioral_evidence != row.declared_behavioral_evidence
        for row in coverage.rows
    )
    assert coverage.complete is True
    assert container.research.compiled_domain.compiled.publishable is True
