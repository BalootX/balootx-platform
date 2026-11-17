from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from baloot_platform.challenge.constraint_deal_search import (
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.configuration.service import VersionedConfigurationService
from baloot_platform.core.rule_source_catalog import RuleSourceCatalogV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.auction_compiler import OrToolsCpSatAuctionCompilerV1
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.research.domain_compiler import BalootDomainCompilerV1
from baloot_platform.research.escalation_compiler import (
    OrToolsCpSatEscalationCompilerV1,
)
from baloot_platform.research.play_compiler import OrToolsCpSatPlayCompilerV1
from baloot_platform.research.project_compiler import OrToolsCpSatProjectCompilerV1


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        return {
            path
            for key, child in value.items()
            for path in _leaf_paths(child, f"{prefix}.{key}" if prefix else str(key))
        }
    return {prefix}


def _profile(root: Path, relative_path: str) -> dict[str, Any]:
    return yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))


def _native_v3_bundle_paths() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[1]
    selected: list[str] = []
    for path in sorted((root / "configs").glob("bundle*.yaml")):
        bundle = yaml.safe_load(path.read_text(encoding="utf-8"))
        components = bundle.get("components") or {}
        challenge_relative = components.get("challenge")
        if not isinstance(challenge_relative, str):
            continue
        challenge_path = root / "configs" / challenge_relative
        challenge = yaml.safe_load(challenge_path.read_text(encoding="utf-8"))
        if challenge.get("scenarioResearchV4", {}).get("backend") == "ortools_cp_sat_v3":
            selected.append(path.relative_to(root).as_posix())
    if not selected:
        raise AssertionError("no active native V3 Research bundles were discovered")
    return tuple(selected)


_NATIVE_V3_BUNDLES = _native_v3_bundle_paths()


def _alternative(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}__mutation_probe__"
    if isinstance(value, list):
        return [*value, "__mutation_probe__"]
    raise TypeError(f"unsupported rule identity leaf: {type(value).__name__}")


def _replace_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _plugin_descriptors() -> tuple[dict[str, Any], ...]:
    return tuple(
        owner.descriptor.to_dict()
        for owner in (
            OrToolsCpSatAuctionCompilerV1,
            OrToolsCpSatDealGeneratorV1,
            OrToolsCpSatEscalationCompilerV1,
            OrToolsCpSatProjectCompilerV1,
            OrToolsCpSatPlayCompilerV1,
        )
    )


def test_new_rule_leaf_receipts_do_not_rewrite_historical_profiles(root: Path) -> None:
    rules = _profile(root, "configs/rules/saudi_common_v1.yaml")
    before = canonical_sha256(rules)
    plugins = {item["id"]: item for item in _plugin_descriptors()}

    proofs = BalootDomainCompilerV1._behavioral_proofs(
        rules=rules,
        plugins=plugins,
    )

    assert "multiplier_cap" not in rules["scoring"]["kabout"]
    assert "rules.scoring.kabout.multiplier_cap" not in proofs
    assert canonical_sha256(rules) == before


@pytest.mark.parametrize(
    "config_root",
    (
        pytest.param("configs", id="source"),
        pytest.param(
            "src/baloot_platform/resources/configs",
            id="package",
        ),
    ),
)
def test_optional_project_profile_has_only_its_declared_behavior_delta(
    root: Path,
    config_root: str,
) -> None:
    official = _profile(
        root,
        f"{config_root}/rules/saudi_federation_2023_v2.yaml",
    )
    variant = _profile(
        root,
        f"{config_root}/rules/saudi_project_optional_unratified_v1.yaml",
    )
    changes = VersionedConfigurationService.compare_payloads(official, variant)

    assert {item["path"] for item in changes} == {
        "profile.description",
        "profile.id",
        "profile.status",
        "profile.version",
        "projects.declaration_mode",
    }
    behavior_changes = [
        item for item in changes if not item["path"].startswith("profile.")
    ]
    assert behavior_changes == [
        {
            "path": "projects.declaration_mode",
            "change": "changed",
            "left": "mandatory",
            "right": "optional",
        }
    ]

    conventions = _profile(root, f"{config_root}/conventions/conservative_v1.yaml")
    localization = _profile(root, f"{config_root}/product/studio_ortools.yaml")
    plugins = _plugin_descriptors()
    compiled = {
        mode: BalootDomainCompilerV1().compile(
            rules=rules,
            conventions=conventions,
            localization=localization,
            bundle_hash=canonical_sha256(
                {
                    "rules": rules,
                    "conventions": conventions,
                    "localization": localization,
                    "plugins": plugins,
                }
            ),
            plugins=plugins,
        ).compiled
        for mode, rules in (("mandatory", official), ("optional", variant))
    }

    for target in ("runtimePlan", "referencePlan", "cpSatBlueprint"):
        plans = {
            mode: copy.deepcopy(result.target_payloads[target])
            for mode, result in compiled.items()
        }
        for plan in plans.values():
            plan["entries"] = {
                item["primitiveId"]: item for item in plan["entries"]
            }
        assert {
            item["path"]
            for item in VersionedConfigurationService.compare_payloads(
                plans["mandatory"],
                plans["optional"],
            )
        } == {
            "compiledProjects.declarationMode",
            "compiledProjects.sha256",
            "entries.rules.projects.declaration_mode.valueHash",
        }

    for mode, result in compiled.items():
        proof = result.target_payloads["behavioralProofs"][
            "rules.projects.declaration_mode"
        ]
        assert proof["status"] == "proven_behavioral"
        assert proof["proofKind"] == (
            "project_declaration_mode_reference_runtime_native_cp_sat_counterfactual"
        )
        assert set(proof["targetsObserved"]) == {"runtime", "reference", "cpSat"}
        assert result.target_payloads["runtimePlan"]["compiledProjects"][
            "declarationMode"
        ] == mode


@pytest.mark.parametrize("bundle_relative", _NATIVE_V3_BUNDLES)
def test_every_active_rule_leaf_has_executable_mutation_or_rejection_evidence(
    root: Path, bundle_relative: str,
) -> None:
    bundle = _profile(root, bundle_relative)
    components = bundle["components"]
    certification = _profile(root, f"configs/{components['certification']}")
    rules_data = _profile(root, f"configs/{components['rules']}")
    conventions = _profile(root, f"configs/{components['conventions']}")
    localization = _profile(root, f"configs/{components['product']}")
    rules = RuleBook(rules_data)
    catalog = RuleSourceCatalogV1.load(
        root / "configs" / certification["rule_source_catalog"]["path"], rules
    )
    catalog.verify_retained_sources(root)
    compiler = BalootDomainCompilerV1()
    plugins = _plugin_descriptors()
    compiled = compiler.compile(
        rules=rules_data,
        conventions=conventions,
        localization=localization,
        bundle_hash=canonical_sha256(
            {
                "bundle": bundle,
                "rules": rules_data,
                "conventions": conventions,
                "localization": localization,
                "plugins": plugins,
            }
        ),
        plugins=plugins,
    ).compiled

    rows = {
        row.primitive_id: row
        for row in compiled.coverage.rows
        if row.primitive_id.startswith("rules.")
    }
    expected = {
        f"rules.{section}.{path}"
        for section, value in rules_data.items()
        if section not in {"schema_version", "profile"}
        for path in _leaf_paths(value)
    }
    identity_expected = {
        "rules.schema_version",
        *{
            f"rules.profile.{path}"
            for path in _leaf_paths(rules_data["profile"])
        },
    }
    proofs = dict(compiled.target_payloads["behavioralProofs"])
    definitions = {item.kind: item for item in compiled.definitions}
    docs = {
        item["primitiveId"]: item
        for item in compiled.target_payloads["documentationModel"]["items"]
    }
    target_entries = {
        target: {
            item["primitiveId"]: item
            for item in compiled.target_payloads[plan]["entries"]
        }
        for target, plan in (
            ("runtime", "runtimePlan"),
            ("reference", "referencePlan"),
            ("cpSat", "cpSatBlueprint"),
            ("policy", "policyPlan"),
        )
    }

    failures: list[str] = []

    # Protocol/profile leaves are identity authorities rather than executable
    # gameplay leaves. Mutate only ephemeral copies and prove that either the
    # RuleBook or its exact source-catalog binding rejects the substitution.
    for primitive_id in sorted(identity_expected):
        profile_path = primitive_id.removeprefix("rules.")
        try:
            binding = catalog.binding_for(profile_path)
        except KeyError:
            failures.append(f"{primitive_id}: no source-catalog binding")
            continue
        expected_authority = (
            "protocol_invariant"
            if profile_path == "schema_version"
            else "provenance_metadata"
        )
        if binding.get("authority_class") != expected_authority:
            failures.append(
                f"{primitive_id}: expected {expected_authority} classification"
            )
        for owner_key in ("runtime_owner", "cp_sat_owner"):
            owner_path = root / str(binding.get(owner_key) or "")
            if not owner_path.is_file():
                failures.append(
                    f"{primitive_id}: missing {owner_key} {binding.get(owner_key)!r}"
                )
        if not binding.get("source_ids") or not binding.get("clause"):
            failures.append(f"{primitive_id}: incomplete source/clause authority")
        if not binding.get("examples") or not binding.get("counterexamples"):
            failures.append(f"{primitive_id}: missing examples or counterexamples")
        for example in binding.get("examples") or ():
            if str(example).endswith(".py") and not (root / str(example)).is_file():
                failures.append(f"{primitive_id}: missing source-catalog test {example}")

        mutated = copy.deepcopy(rules_data)
        cursor: Any = rules_data
        for part in profile_path.split("."):
            cursor = cursor[part]
        _replace_path(mutated, profile_path, _alternative(cursor))
        try:
            mutated_rules = RuleBook(mutated)
        except ValueError:
            continue
        try:
            RuleSourceCatalogV1(catalog.data, mutated_rules)
        except ValueError:
            continue
        failures.append(f"{primitive_id}: identity mutation did not fail closed")

    for missing in sorted(expected - rows.keys()):
        failures.append(f"{missing}: absent from compiled rule-leaf inventory")
    for unexpected in sorted(rows.keys() - expected):
        failures.append(f"{unexpected}: absent from selected rules profile")

    for primitive_id in sorted(expected & rows.keys()):
        row = rows[primitive_id]
        profile_path = primitive_id.removeprefix("rules.")
        try:
            binding = catalog.binding_for(profile_path)
        except KeyError:
            failures.append(f"{primitive_id}: no source-catalog binding")
            continue

        for owner_key in ("runtime_owner", "cp_sat_owner"):
            owner_path = root / str(binding.get(owner_key) or "")
            if not owner_path.is_file():
                failures.append(
                    f"{primitive_id}: missing {owner_key} {binding.get(owner_key)!r}"
                )
        if not binding.get("source_ids") or not binding.get("clause"):
            failures.append(f"{primitive_id}: incomplete source/clause authority")
        if not binding.get("examples") or not binding.get("counterexamples"):
            failures.append(f"{primitive_id}: missing examples or counterexamples")
        for example in binding.get("examples") or ():
            if str(example).endswith(".py") and not (root / str(example)).is_file():
                failures.append(f"{primitive_id}: missing source-catalog test {example}")

        definition = definitions[row.definition_kind]
        cursor: Any = rules_data
        for part in profile_path.split("."):
            cursor = cursor[part]
        if row.value_hash != canonical_sha256(cursor):
            failures.append(f"{primitive_id}: compiled value identity mismatch")
        if not definition.documentation or any(
            not (root / path).is_file() for path in definition.documentation
        ):
            failures.append(f"{primitive_id}: missing governed documentation owner")
        if not definition.tests or any(
            not (root / path).is_file() for path in definition.tests
        ):
            failures.append(f"{primitive_id}: missing executable owning tests")
        if primitive_id not in docs:
            failures.append(f"{primitive_id}: missing compiled documentation binding")
        elif docs[primitive_id].get("valueHash") != row.value_hash:
            failures.append(f"{primitive_id}: stale compiled documentation value")

        executable_targets = set(row.required_targets) & set(target_entries)
        for target in sorted(executable_targets):
            entry = target_entries[target].get(primitive_id)
            if entry is None or not entry.get("executableReceipt", {}).get("verified"):
                failures.append(f"{primitive_id}: unverified {target} lowering")
            elif entry.get("valueHash") != row.value_hash:
                failures.append(f"{primitive_id}: stale {target} value binding")

        proof = proofs.get(primitive_id)
        if proof is None:
            failures.append(f"{primitive_id}: no mutation/invariant proof")
            continue
        proof_core = dict(proof)
        proof_hash = proof_core.pop("proofHash", None)
        if proof_hash != canonical_sha256(proof_core):
            failures.append(f"{primitive_id}: mutation receipt hash mismatch")
        if proof.get("primitiveId") != primitive_id:
            failures.append(f"{primitive_id}: receipt identity mismatch")
        if proof.get("rulesHash") != rules.sha256:
            failures.append(f"{primitive_id}: receipt rules identity mismatch")

        status = proof.get("status")
        if status == "proven_behavioral":
            if proof.get("counterfactualRulesHash") in {None, rules.sha256}:
                failures.append(f"{primitive_id}: no distinct ephemeral mutation")
            evidence_hashes = {
                key: value
                for key, value in proof.items()
                if key.lower().endswith("observationhash")
                or key.lower().endswith("observationhashes")
                or key in {
                    "cpSatPlanHash",
                    "referenceRuntimeParityHash",
                    "referenceRuntimeParityHashes",
                }
            }
            if not any(evidence_hashes.values()):
                failures.append(f"{primitive_id}: no observed mutation evidence")
            required_observations = set(row.required_targets) & {
                "runtime",
                "reference",
                "cpSat",
                "policy",
            }
            observed = set(proof.get("targetsObserved") or ())
            if not required_observations.issubset(observed):
                failures.append(
                    f"{primitive_id}: mutation did not observe required targets "
                    f"{sorted(required_observations - observed)}"
                )
        elif status == "proven_invariant":
            if "rejection" not in str(proof.get("proofKind") or ""):
                failures.append(f"{primitive_id}: invariant did not fail closed")
            if not proof.get("counterexampleHash") or not proof.get("rejectionType"):
                failures.append(f"{primitive_id}: invariant rejection evidence incomplete")
        else:
            failures.append(f"{primitive_id}: unsupported proof status {status!r}")

        if row.behavioral_or_invariant_status != status or not row.qualification_eligible:
            failures.append(f"{primitive_id}: proof is not qualification-bound")

    assert not failures, (
        f"governed rule-leaf mutation gate failed for {bundle_relative}:\n"
        + "\n".join(failures)
    )
