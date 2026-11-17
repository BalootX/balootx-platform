from __future__ import annotations

import json
from pathlib import Path

import yaml

from baloot_platform.config import build_bundle
from baloot_platform.hitl import PostgresReviewGovernanceStore


def test_v18_bundle_contains_governance_certification_benchmark_and_operations(root: Path):
    bundle = build_bundle(root)
    expected_versions = {
        "hitl": "18.0.0",
        "certification": "21.0.0",
        "benchmark": "21.2.0",
        "operations": "18.0.0",
    }
    for family, version in expected_versions.items():
        assert bundle.component(family)["profile"]["version"] == version


def test_hitl_postgres_migration_has_rls_and_complete_tables(root: Path):
    sql = (root / "migrations/010_hitl_governance_v18.sql").read_text(encoding="utf-8")
    for table in ("review_cases", "blind_assignments", "review_submissions", "consent", "training_candidates", "improvement_proposals", "events"):
        assert f"hitl.{table}" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "current_setting('baloot.subject_id'" in sql
    # The production adapter must expose the complete local-store contract.
    methods = set(dir(PostgresReviewGovernanceStore))
    assert {"put_case", "get_case", "put_assignment", "put_submission", "put_candidate", "put_proposal", "verify_events"} <= methods
    assert "assign" not in methods


def test_public_card_pack_policy_is_redistributable_and_configurable(root: Path):
    config = yaml.safe_load((root / "configs/assets/card_packs.yaml").read_text(encoding="utf-8"))
    assert config["default_pack"] == "goodall_modern_baloot_32"
    assert config["packs"]["goodall_modern_baloot_32"]["license_status"] == "CC0-1.0-public-domain"
    assert "poker_double_backmarked_baloot_32" in config["packs"]
    assert config["packs"]["poker_double_backmarked_baloot_32"]["license_status"] == "user_supplied_review_required"


def test_unified_arabic_application_links_hitl_and_core_workflows(root: Path):
    app = (root / "src/baloot_platform/service/static/app/index.html").read_text(encoding="utf-8")
    assert "hitl-studio" in app
    assert "لفة" in app and "صكّة" in app
    studio = (root / "src/baloot_platform/service/static/hitl_studio.html").read_text(encoding="utf-8")
    assert "موافقة" in studio and "التدريب" in studio


def test_openapi_and_packaged_resources_are_regenerable(root: Path):
    openapi = json.loads((root / "api/openapi.json").read_text(encoding="utf-8"))
    assert "/api/v1/hitl/cases" in openapi["paths"]
    assert "/api/v1/hitl/improvements" in openapi["paths"]
    packaged = root / "src/baloot_platform/resources/configs/hitl/governance_v18.yaml"
    assert packaged.is_file()
    assert packaged.read_bytes() == (root / "configs/hitl/governance_v18.yaml").read_bytes()
