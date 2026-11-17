from __future__ import annotations

import hashlib
import json
import re

from fastapi.testclient import TestClient

from baloot_platform.config import build_bundle
from baloot_platform.guide import build_engine_guide
from baloot_platform.service import api
from scripts.generate_engine_guide_docs import render_markdown


ACTIVE_GUIDE_BUNDLE = "configs/bundle_ortools_research_native_v3_1.yaml"


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def test_engine_guide_is_derived_from_active_rules_and_conventions(root):
    bundle = build_bundle(root, bundle_path=ACTIVE_GUIDE_BUNDLE)
    guide = build_engine_guide(bundle)
    rules = bundle.components["rules"].data

    assert guide["profile"]["id"] == rules["profile"]["id"]
    assert guide["profile"]["version"] == rules["profile"]["version"]
    assert guide["profile"]["expertRatified"] is False
    assert guide["audit"]["rulesHash"] == bundle.rules.sha256
    assert guide["audit"]["bundleHash"] == bundle.sha256
    assert guide["guideHash"] == _hash(
        {key: value for key, value in guide.items() if key != "guideHash"}
    )

    sections = {section["id"]: section for section in guide["sections"]}
    assert set(sections) == {
        "buying",
        "card_order",
        "escalation",
        "legal_play",
        "projects",
        "scoring",
        "reading",
        "authority",
        "disputed",
    }
    hokum = sections["card_order"]["tables"][1]["cards"]
    assert [card["rank"] for card in hokum[:3]] == ["ولد", "9", "آص"]
    assert [card["points"] for card in hokum[:3]] == [20, 14, 11]
    assert sections["reading"]["facts"][0]["value"] == "لا"
    project_facts = {
        fact["label"]: fact["value"] for fact in sections["projects"]["facts"]
    }
    assert project_facts["سرا / خمسين / مية في الصن"] == "4 / 10 / 20"
    assert project_facts["سرا / خمسين / مية في الحكم"] == "2 / 5 / 10"
    assert all(len(section["examples"]) >= 2 for section in sections.values())
    capability = guide["capabilityReport"]
    assert capability["rulesHash"] == bundle.rules.sha256
    assert capability["totalSettings"] == len(capability["entries"])
    assert capability["totalSettings"] >= 100
    entries = {entry["path"]: entry for entry in capability["entries"]}
    assert entries["play.must_follow_suit"]["enforcement"] == "runtime_or_fail_closed"
    assert (
        entries["bidding.heuristics.sun_rank_weights"]["enforcement"]
        == "policy_only_not_qualification"
    )
    assert entries["profile.id"]["enforcement"] == "identity_only"
    assert {entry["path"] for entry in capability["entries"]} >= {
        "bidding.heuristics.sun_rank_weights",
        "play.must_follow_suit",
        "scoring.match.target",
    }
    assert not any(
        entry["path"]
        in {
            "disputed_options.first_leader.selected",
            "disputed_options.double_unspecified_mode.selected",
            "disputed_options.hints_have_universal_meaning.selected",
            "disputed_options.kawesh_saneen.selected",
        }
        for entry in capability["entries"]
    )
    disputed_values = {
        fact["label"]: fact["value"] for fact in sections["disputed"]["facts"]
    }
    assert len(disputed_values) == 5


def test_native_v3_guide_includes_governed_escalation_section(root):
    bundle = build_bundle(
        root,
        bundle_path="configs/bundle_ortools_research_native_v3.yaml",
    )
    guide = build_engine_guide(bundle)
    sections = {section["id"]: section for section in guide["sections"]}

    assert "escalation" in sections
    assert {fact["label"] for fact in sections["escalation"]["facts"]} == {
        "التدبيل مفعّل",
        "سلسلة تدبيل الحكم",
        "سلسلة تدبيل الصن",
        "وضع التدبيل الافتراضي",
        "حد تدبيل الصن",
    }
    source_catalog = guide["ruleSourceCatalog"]
    assert source_catalog["id"] == "saudi_federation_2023_sources"
    assert source_catalog["sha256"] == guide["audit"]["ruleSourceCatalogHash"]
    assert source_catalog["certificationStatus"] == (
        "public_regulation_plus_unratified_configured_gaps"
    )
    entries = {entry["path"]: entry for entry in guide["capabilityReport"]["entries"]}
    assert entries["projects.comparison.category_priorities.hundred_sequence"][
        "authorityClass"
    ] == "official_regulation"
    assert entries["card_rules.sun_points"]["ratificationStatus"] == (
        "external_referee_required"
    )
    assert all(
        entry["sourceIds"] and entry["runtimeOwner"] and entry["cpSatOwner"]
        for entry in entries.values()
    )


def test_native_v3_1_guide_binds_successor_rules_and_adjudication_authority(root):
    bundle = build_bundle(
        root,
        bundle_path="configs/bundle_ortools_research_native_v3_1.yaml",
    )
    guide = build_engine_guide(bundle)

    assert guide["profile"]["id"] == "saudi_federation_2023_v2"
    assert guide["ruleSourceCatalog"]["id"] == (
        "saudi_federation_2023_v2_sources"
    )
    authority = guide["adjudicationAuthority"]
    assert authority["authoritySha256"] == (
        "a347741d373c9de8761aaa08d2ba593b0e9301ce97bcb928ac2a60df7d4695a8"
    )
    assert authority["authoringCatalogSha256"] == (
        "8389760a4a247c50c214b11a2596860ec72895cb73147b2a456c56a22002ee2e"
    )
    assert authority["profile"] == {
        "id": "gea_baloot_2023_adjudication_research_v2",
        "version": "2.0.0",
        "status": "source_verified_uncertified",
    }
    assert authority["sourceAuthority"]["sha256"] == (
        "9a7c67046549c5013f10e31069e6feba70e20eb1d47b27c7a2ba8abaa3e848c2"
    )
    assert authority["sourceAuthority"]["verificationReceiptPath"] == (
        "release/saudi-baloot-official-source-verification-20260831-v2.json"
    )
    events = {item["id"]: item for item in authority["events"]}
    assert "incorrect_highest_card_naming" in events
    assert "false_ekka" not in events
    assert all(
        item["hardPredicate"] == "unsupported"
        and item["uiExposure"] == "hidden"
        for item in events.values()
    )
    assert guide["audit"]["adjudicationHash"] == bundle.components[
        "adjudication"
    ].sha256




def test_public_guide_shell_contains_no_embedded_arabic_product_copy(root):
    static = root / "src/baloot_platform/service/static/guide"
    for name in ("index.html", "app.js", "app.css"):
        value = (static / name).read_text(encoding="utf-8")
        assert re.search(r"[\u0600-\u06ff]", value) is None, name


def test_public_guide_api_page_and_markdown_share_one_document(root):
    guide = build_engine_guide(
        build_bundle(root, bundle_path=ACTIVE_GUIDE_BUNDLE)
    )

    with TestClient(api.app) as client:
        response = client.get("/api/v1/engine-guide")
        assert response.status_code == 200
        assert response.json() == api.ENGINE_GUIDE
        page = client.get("/guide")
        assert page.status_code == 200
        assert api.ENGINE_GUIDE["pageTitle"] in page.text
        assert api.ENGINE_GUIDE["guideHash"] in page.text
        assert "{{engine_guide_json}}" not in page.text
        assert "{{guide_asset_version}}" not in page.text
