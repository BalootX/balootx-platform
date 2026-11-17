from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from baloot_platform.config import (
    PROFILE_FAMILIES,
    build_bundle,
    validate_profile_payload,
)
from baloot_platform.configuration import (
    SQLiteConfigurationStore,
    VersionedConfigurationService,
)
from baloot_platform.configuration.models import digest
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.rule_source_catalog import load_bound_rule_source_catalog
from baloot_platform.guide import build_engine_guide


_ACTIVE_BUNDLE_FILE_SHA256 = (
    "964917900749379f0cb84722db0463e411a13538e8dc3486eb6c00fb6de67881"
)
_ALTERNATE_BUNDLE_ID = "baloot_ortools_research_v4"
_NATIVE_V3_BUNDLE_ID = "baloot_ortools_research_native_v3"
_RATIFIED_PROJECTS_BUNDLE = "configs/bundle_ortools_research_ratified_projects.yaml"
_OPTIONAL_PROJECT_BUNDLE_ID = (
    "baloot_ortools_research_native_v3_optional_project"
)


def _load(root: Path, relative: str) -> dict[str, Any]:
    return yaml.safe_load((root / relative).read_text(encoding="utf-8"))


def test_time_budgeted_policy_profile_is_closed_hash_bound_and_optional(root):
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    from baloot_platform.research.proof_service import ScenarioProofBudgetRegistryV1
    from baloot_platform.config import _validate_scenario_research_v4
    raw = _load(root, "configs/challenge/intelligent_research.yaml")
    profile = raw["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    old = ScenarioProofBudgetV1.build("test", profile)
    assert old.policy_search is None and "policySearch" not in profile
    chosen = {**profile, "policySearch": "time_budgeted_symbolic_v1"}
    revised = ScenarioProofBudgetV1.build("test", chosen)
    assert revised.sha256 != old.sha256
    registry = ScenarioProofBudgetRegistryV1({"test": revised}, "0" * 64).to_dict()
    assert registry["profiles"]["test"]["policySearch"] == chosen["policySearch"]
    raw["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"] = chosen
    validate_profile_payload("challenge", raw)
    legacy = _load(root, "configs/challenge/progressive_mining_ortools.yaml")
    legacy["scenarioResearchV4"]["proofBudgets"]["version"] = "2.1.0"
    for budget in legacy["scenarioResearchV4"]["proofBudgets"]["profiles"].values():
        budget["nativeObserved"] = copy.deepcopy(chosen["nativeObserved"])
    _validate_scenario_research_v4(legacy, legacy["profile"]["version"])
    legacy["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]["policySearch"] = chosen["policySearch"]
    with pytest.raises(ValueError, match="shape"):
        _validate_scenario_research_v4(legacy, legacy["profile"]["version"])
    for invalid in (None, True, "other", {"mode": "time_budgeted_symbolic_v1"}):
        with pytest.raises(ValueError):
            ScenarioProofBudgetV1.build("test", {**profile, "policySearch": invalid})
    with pytest.raises(ValueError, match="shape"):
        ScenarioProofBudgetV1.build("test", {**chosen, "policyTypo": True})


def _component_hashes(root: Path, bundle: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for family, relative in bundle["components"].items():
        path = root / "configs" / str(relative)
        hashes[family] = (
            RuleBook.load(path).sha256
            if family == "rules"
            else digest(yaml.safe_load(path.read_text(encoding="utf-8")))
        )
    return hashes


def _bundle_hash(bundle: dict[str, Any], hashes: dict[str, str]) -> str:
    return digest(
        {
            "bundleProfile": bundle["profile"],
            "mode": "full",
            "components": {family: hashes[family] for family in PROFILE_FAMILIES},
        }
    )


def _service(root: Path, tmp_path: Path) -> VersionedConfigurationService:
    governance = _load(root, "configs/configuration/control-plane.yaml")
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=root,
        profile=governance,
    )
    service.bootstrap_packaged_bundle(actor_id="TEST-ACTIVE")
    return service


def _seed_published_authored_profile(
    service: VersionedConfigurationService,
    *,
    family: str,
    payload: dict[str, Any],
    tenant_id: str,
) -> str:
    """Seed a source-qualified revision before generated schemas are refreshed."""

    identity = dict(payload["profile"])
    revision = service.store.create_profile(
        family=family,
        profile_id=str(identity["id"]),
        version=str(identity["version"]),
        scope="tenant",
        scope_key=tenant_id,
        tenant_id=tenant_id,
        payload=copy.deepcopy(payload),
        status="published",
        created_by="TEST-AUTHORED-SOURCE",
    )
    return revision.revision_id


def test_alternate_bundle_is_exactly_two_component_changes(root: Path) -> None:
    active_path = root / "configs/bundle.yaml"
    active = _load(root, "configs/bundle.yaml")
    alternate = _load(root, "configs/bundle_ortools_research_v4.yaml")

    assert hashlib.sha256(active_path.read_bytes()).hexdigest() == (
        _ACTIVE_BUNDLE_FILE_SHA256
    )
    assert alternate["profile"] == {
        "id": _ALTERNATE_BUNDLE_ID,
        "version": "22.0.1",
        "status": "governed_candidate",
    }
    assert set(active["components"]) == set(PROFILE_FAMILIES)
    assert set(alternate["components"]) == set(PROFILE_FAMILIES)
    assert {
        family
        for family in PROFILE_FAMILIES
        if active["components"][family] != alternate["components"][family]
    } == {"challenge", "product"}
    assert alternate["components"]["challenge"] == (
        "challenge/progressive_mining_ortools.yaml"
    )
    assert alternate["components"]["product"] == "product/studio_ortools.yaml"


def test_alternate_bundle_component_and_identity_hashes_are_distinct(
    root: Path,
) -> None:
    active = _load(root, "configs/bundle.yaml")
    alternate = _load(root, "configs/bundle_ortools_research_v4.yaml")
    active_hashes = _component_hashes(root, active)
    alternate_hashes = _component_hashes(root, alternate)

    assert {
        family
        for family in PROFILE_FAMILIES
        if active_hashes[family] != alternate_hashes[family]
    } == {"challenge", "product"}
    assert _bundle_hash(active, active_hashes) != _bundle_hash(
        alternate, alternate_hashes
    )

    packaged_active = build_bundle(root)
    assert packaged_active.bundle_profile["profile"]["id"] == "baloot_default"
    assert packaged_active.components["challenge"].path.endswith(
        "progressive_mining_v2.yaml"
    )
    assert packaged_active.components["product"].path.endswith("studio_v1.yaml")


def test_alternate_bundle_publication_requires_explicit_selection(
    root: Path,
    tmp_path: Path,
) -> None:
    service = _service(root, tmp_path)
    active = service.resolve(bundle_id="baloot_default")
    alternate_definition = _load(root, "configs/bundle_ortools_research_v4.yaml")
    tenant_id = "ORTOOLS-V4-AUTHORING"
    component_revisions = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    for family in ("challenge", "product"):
        payload = _load(
            root,
            f"configs/{alternate_definition['components'][family]}",
        )
        validate_profile_payload(family, payload, root=root)
        component_revisions[family] = _seed_published_authored_profile(
            service,
            family=family,
            payload=payload,
            tenant_id=tenant_id,
        )

    published = service.create_bundle(
        bundle_id=_ALTERNATE_BUNDLE_ID,
        version="22.0.1",
        mode="full",
        component_revisions=component_revisions,
        created_by="TEST-PUBLISHER",
        scope="tenant",
        tenant_id=tenant_id,
        publish=True,
    )

    assert published["status"] == "published"
    with pytest.raises(KeyError, match="published configuration bundle"):
        service.resolve(bundle_id=_ALTERNATE_BUNDLE_ID)

    selected = service.resolve(
        bundle_id=_ALTERNATE_BUNDLE_ID,
        tenant_id=tenant_id,
    )
    exact = service.resolve(revision_id=str(published["revisionId"]))
    default = service.resolve(bundle_id="baloot_default")

    assert selected["bundle"]["revisionId"] == published["revisionId"]
    assert exact["effectiveHash"] == selected["effectiveHash"]
    assert selected["effectiveHash"] != active["effectiveHash"]
    assert selected["profiles"]["challenge"]["scenarioResearchV4"]["backend"] == (
        "ortools_cp_sat_v2"
    )
    assert selected["profiles"]["product"]["profile"]["version"] == _load(
        root,
        f"configs/{alternate_definition['components']['product']}",
    )["profile"]["version"]
    assert default["bundle"]["revisionId"] == active["bundle"]["revisionId"]
    assert default["effectiveHash"] == active["effectiveHash"]


def test_native_v3_bundle_bootstraps_published_without_replacing_default(
    root: Path,
    tmp_path: Path,
) -> None:
    service = _service(root, tmp_path)
    default_before = service.resolve(bundle_id="baloot_default")

    published = service.bootstrap_packaged_bundle(
        actor_id="TEST-NATIVE-V3",
        bundle_path="configs/bundle_ortools_research_native_v3.yaml",
        scope="environment",
        environment="research-native-v3",
    )

    assert published["bundle"]["bundleId"] == _NATIVE_V3_BUNDLE_ID
    assert published["bundle"]["status"] == "published"
    assert published["bundle"]["scope"] == "environment"
    assert published["profiles"]["challenge"]["scenarioResearchV4"]["backend"] == (
        "ortools_cp_sat_v3"
    )
    assert published["profiles"]["rules"]["profile"]["id"] == ("saudi_federation_2023")
    default_after = service.resolve(revision_id=default_before["bundle"]["revisionId"])
    assert default_after["effectiveHash"] == default_before["effectiveHash"]


def test_native_v3_1_bundle_creates_new_revision_and_preserves_v3_0(
    root: Path,
    tmp_path: Path,
) -> None:
    service = _service(root, tmp_path)
    historical = service.bootstrap_packaged_bundle(
        actor_id="TEST-NATIVE-V3-HISTORICAL",
        bundle_path="configs/bundle_ortools_research_native_v3.yaml",
        scope="environment",
        environment="research-native-v3",
    )
    current = service.bootstrap_packaged_bundle(
        actor_id="TEST-NATIVE-V3-1",
        bundle_path="configs/bundle_ortools_research_native_v3_1.yaml",
        scope="environment",
        environment="research-native-v3",
    )

    assert historical["bundle"]["version"] == "23.0.1"
    assert current["bundle"]["version"] == "24.0.1"
    # Packaged bootstrap publishes by immutable content identity and supersedes
    # the prior active row, but its contract does not infer optional lineage.
    assert current["bundle"]["supersedesRevisionId"] is None
    assert historical["profiles"]["challenge"]["profile"]["version"] == "3.0.0"
    assert "authoringDefaults" not in historical["profiles"]["challenge"][
        "scenarioResearchV4"
    ]
    assert current["profiles"]["challenge"]["profile"]["version"] == "3.1.0"
    assert current["profiles"]["challenge"]["scenarioResearchV4"][
        "authoringDefaults"
    ]["resources"]["seedsPerShard"] == 1
    exact_historical = service.resolve(
        revision_id=historical["bundle"]["revisionId"]
    )
    assert exact_historical["bundle"]["status"] == "superseded"
    assert current["bundle"]["status"] == "published"
    assert exact_historical["effectiveHash"] == historical["effectiveHash"]


def test_native_v3_successor_and_optional_project_bundles_are_explicit(
    root: Path,
) -> None:
    historical = _load(root, "configs/bundle_ortools_research_native_v3.yaml")
    successor = _load(root, "configs/bundle_ortools_research_native_v3_1.yaml")
    optional = _load(
        root,
        "configs/bundle_ortools_research_native_v3_optional_project.yaml",
    )

    assert historical["components"]["rules"] == (
        "rules/saudi_federation_2023.yaml"
    )
    assert successor["components"]["rules"] == (
        "rules/saudi_federation_2023_v2.yaml"
    )
    assert successor["components"]["adjudication"] == (
        "adjudication/research_events_v2.yaml"
    )
    assert successor["components"]["certification"] == (
        "certification/rules_saudi_federation_2023_v2.yaml"
    )
    assert optional["profile"] == {
        "id": _OPTIONAL_PROJECT_BUNDLE_ID,
        "version": "1.0.1",
        "status": "published",
    }
    assert optional["components"]["rules"] == (
        "rules/saudi_project_optional_unratified_v1.yaml"
    )
    assert optional["components"]["challenge"] == (
        "challenge/progressive_mining_ortools_v3_1.yaml"
    )
    assert optional["components"]["certification"] == (
        "certification/rules_saudi_project_optional_unratified_v1.yaml"
    )


def test_native_v3_1_challenge_schema_is_generated_and_packaged(
    root: Path,
) -> None:
    challenge = _load(
        root,
        "configs/challenge/progressive_mining_ortools_v3_1.yaml",
    )

    validate_profile_payload("challenge", challenge, root=root)
    assert challenge["scenarioResearchV4"]["authoringDefaults"] == {
        "schemaVersion": 1,
        "resources": {"seedsPerShard": 1},
        "playingQuality": {
            "budgetProfile": "bounded_v1",
            "boundaryEvaluatorByDecision": {
                "bid": "auction_hand_margin_v1",
                "escalation": "current_raw_point_margin_v1",
                "project": "current_raw_point_margin_v1",
                "play": "current_raw_point_margin_v1",
            },
        },
    }
    assert (
        root / "schemas/profiles/challenge.schema.json"
    ).read_bytes() == (
        root
        / "src/baloot_platform/resources/schemas/profiles/challenge.schema.json"
    ).read_bytes()


def test_ratified_projects_bundle_binds_exact_v8_rules_catalog_and_guide(root):
    definition = _load(root, _RATIFIED_PROJECTS_BUNDLE)
    historical = _load(root, "configs/bundle_ortools_research_native_v3_intelligent.yaml")
    assert definition["profile"] == {
        "id": "baloot_ortools_research_native_v3_ratified_projects",
        "version": "1.0.0",
        "status": "governed_candidate",
    }
    assert {family for family in PROFILE_FAMILIES
            if definition["components"][family] != historical["components"][family]} == {
        "rules", "certification",
    }
    bundle = build_bundle(root, bundle_path=_RATIFIED_PROJECTS_BUNDLE)
    certification = bundle.component("certification")
    assert certification["candidate_profile_identity"] == {
        "id": "saudi_federation_2023_v8", "version": "8.0.0",
        "sha256": "001bd15ec683cee0b1c88a505f88d7c62025612e34f056e006082576e480094d",
    }
    assert bundle.rules.sha256 == certification["candidate_profile_identity"]["sha256"]
    assert bundle.components["certification"].sha256 == (
        "99a6c4fcce761cf4cfc931d8d5e52bad4cd367b30656fe6862934a4723448fd9"
    )
    assert certification["profile"]["status"] == "candidate_external_review_required"
    assert certification["certification_eligible"] is True
    assert certification["trusted_signers"] == []
    assert certification["automatic_activation"] is False
    assert certification["software_conformance"]["external_certification_substitute"] is False
    assert bundle.rules.data["profile"]["expert_ratified"] is False
    catalog = load_bound_rule_source_catalog(certification, bundle.rules, root)
    assert catalog.sha256 == certification["rule_source_catalog"]["sha256"] == (
        "1cd2bdf5d2f83bac0bf45f7b14bb85462678c221a43a01c8346a049e9f57a88f"
    )
    guide = build_engine_guide(bundle)
    assert guide["profile"]["id"] == bundle.rules.id
    assert guide["audit"]["rulesHash"] == bundle.rules.sha256
    assert guide["ruleSourceCatalog"]["sha256"] == catalog.sha256
    entries = {item["path"]: item for item in guide["capabilityReport"]["entries"]}
    for path, value in {
        "projects.declaration_mode": "mandatory",
        "projects.top_rank_visibility": "comparison_only",
        "projects.reveal_scope": "scoring_team",
        "projects.tie_break_order": "second_trick_reveal",
    }.items():
        assert entries[path]["value"] == value
        assert entries[path]["ratificationStatus"] == "expert_ratified"
    corpus = json.loads((root / certification["golden_corpus"]).read_text(encoding="utf-8"))
    from scripts.generate_rule_fixtures import build
    assert corpus == build(root / "configs" / certification["candidate_profile"],
                           root / "configs" / certification["adjudication_profile"])
    assert (corpus["rulesProfile"], corpus["rulesVersion"], corpus["rulesSha256"]) == (
        bundle.rules.id, bundle.rules.version, bundle.rules.sha256,
    )
    assert corpus["caseCount"] >= certification["software_conformance"]["minimum_executable_fixtures"]
    assert "AUTH-PROJECT-DECLARATION-MODE" in {case["caseId"] for case in corpus["cases"]}
    assert {case["provenance"]["source"] for case in corpus["cases"]} == {
        "configs/rules/saudi_federation_2023_v8.yaml",
    }
    for relative in (_RATIFIED_PROJECTS_BUNDLE,
                     "configs/certification/rules_saudi_federation_2023_v8.yaml"):
        assert (root / relative).read_bytes() == (
            root / "src/baloot_platform/resources" / relative
        ).read_bytes()


def test_ratified_projects_bundle_requires_explicit_private_store_selection(root, tmp_path):
    service = _service(root, tmp_path)
    default = service.resolve(bundle_id="baloot_default")
    frozen = service.bootstrap_packaged_bundle(
        actor_id="TEST-FROZEN-V2", bundle_path="configs/bundle_ortools_research_native_v3_intelligent.yaml",
        scope="environment", environment="private-ratified-projects-test",
    )
    selected = service.bootstrap_packaged_bundle(
        actor_id="TEST-RATIFIED-PROJECTS", bundle_path=_RATIFIED_PROJECTS_BUNDLE,
        scope="environment", environment="private-ratified-projects-test",
    )
    assert selected["profiles"]["challenge"]["scenarioResearchV4"]["backend"] == "ortools_cp_sat_v3"
    assert selected["profiles"]["rules"]["profile"]["id"] == "saudi_federation_2023_v8"
    with pytest.raises(KeyError, match="published configuration bundle"):
        service.resolve(bundle_id=selected["bundle"]["bundleId"])
    explicit = service.resolve(revision_id=selected["bundle"]["revisionId"])
    assert explicit["effectiveHash"] == selected["effectiveHash"]
    assert service.resolve(bundle_id="baloot_default")["effectiveHash"] == default["effectiveHash"]
    historical = service.resolve(revision_id=frozen["bundle"]["revisionId"])
    assert historical["effectiveHash"] == frozen["effectiveHash"]
    assert historical["profiles"]["rules"]["profile"]["id"] == "saudi_federation_2023_v2"
