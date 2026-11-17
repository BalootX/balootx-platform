from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import json

import pytest
import yaml
from jsonschema import Draft202012Validator

from baloot_platform.config import validate_certification_adjudication_identity
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.models import Project
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.rule_source_catalog import (
    RuleSourceCatalogV1,
    load_bound_rule_source_catalog,
    load_exact_rule_source_catalog,
    validate_certification_rules_identity,
)
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1
from baloot_platform.research.reference_auction import ReferenceAuctionInterpreterV1


def _profiles(root: Path) -> tuple[RuleBook, RuleBook]:
    return (
        RuleBook.load(root / "configs/rules/saudi_common_v1.yaml"),
        RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml"),
    )


def test_draft_source_successor_preserves_gameplay_and_historical_identity(root: Path):
    old = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    draft = RuleBook.load(root / "configs/rules/saudi_federation_2023_v9.yaml")
    assert old.sha256 == "001bd15ec683cee0b1c88a505f88d7c62025612e34f056e006082576e480094d"
    assert load_exact_rule_source_catalog(old, root).sha256 == (
        "1cd2bdf5d2f83bac0bf45f7b14bb85462678c221a43a01c8346a049e9f57a88f")
    assert (draft.id, draft.version) == ("saudi_federation_2023_v9", "9.0.0")
    assert draft.data["profile"]["status"] == "draft_unratified_mixed_authority"
    assert draft.data["profile"]["expert_ratified"] is False
    assert draft.sha256 != old.sha256
    assert {key: value for key, value in draft.data.items() if key != "profile"} == {
        key: value for key, value in old.data.items() if key != "profile"}
    for attribute in ("structural_domain", "play_domain", "project_domain"):
        assert getattr(draft, attribute).sha256 == getattr(old, attribute).sha256
    assert ScoringDomainIRV1.from_rules(draft.data).sha256 == ScoringDomainIRV1.from_rules(old.data).sha256
    schema = json.loads((root / "schemas/profiles/rules.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(draft.data)
    for name in ("saudi_federation_2023_v9.yaml", "source_catalog_saudi_federation_2023_v9.yaml"):
        relative = Path("configs/rules") / name
        payload = (root / relative).read_bytes()
        assert payload == (root / "src/baloot_platform/resources" / relative).read_bytes()
        assert payload.endswith(b"\n") and b"\r\n" not in payload


def test_draft_source_successor_assigns_precise_authority_and_implementation_owners(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v9.yaml")
    catalog = load_exact_rule_source_catalog(rules, root)
    assert catalog.data["catalog"]["certification_status"] == "draft_provenance_correction_not_certified"
    tie = catalog.binding_for("scoring.match.tie_above_target_continues")
    assert tie["path_prefix"] == "scoring.match.tie_above_target_continues"
    assert tie["authority_class"] == "configured_variant"
    assert tie["source_ids"] == ["balootx_saudi_federation_2023_v9_config"]
    assert tie["ratification_status"] == "external_referee_required"
    assert tie["cp_sat_owner"] == "src/baloot_platform/research/ortools_scoring_stage.py"
    assert catalog.binding_for("scoring.match.target")["authority_class"] == "official_regulation"
    sequence = catalog.binding_for("card_rules.sequence_strength")
    assert sequence["runtime_owner"] == "src/baloot_platform/core/project_ir.py"
    assert sequence["cp_sat_owner"] == "src/baloot_platform/research/ortools_project_fragment.py"
    assert catalog.binding_for("card_rules.sun_strength")["runtime_owner"] == "src/baloot_platform/core/play_ir.py"
    heuristic = catalog.binding_for("bidding.heuristics.sun_threshold")
    assert heuristic["authority_class"] == "heuristic"
    assert heuristic["cp_sat_owner"] == "src/baloot_platform/research/ortools_scenario_model.py"
    assert "policy" in heuristic["applicability"] and "construction" in heuristic["applicability"]
    assert "not a game rule or exact strategic proof" in heuristic["clause"]
    assert all((root / binding[owner]).is_file() for binding in catalog.coverage.values()
               for owner in ("runtime_owner", "cp_sat_owner"))
    receipts = catalog.verify_retained_sources(root)
    retained = [row for row in receipts if row["status"] == "verified"]
    assert {row["path"] for row in retained} == {
        "configs/rules/saudi_federation_2023_v9.yaml",
    }
    assert all(row["status"] == "not_retained" for row in receipts if row not in retained)


@pytest.mark.parametrize("source_id,filename,url,expected,magic", [
    ("gea_baloot_regulation_2023", "gea_baloot_regulation_2023.pdf",
     "https://enjoy.sa/media/j1ofreng/baloot.pdf",
     "9a7c67046549c5013f10e31069e6feba70e20eb1d47b27c7a2ba8abaa3e848c2", b"%PDF-"),
    ("saudi_federation_gameplay_2020_archive", "saudi_federation_gameplay_2020_archive.html.bin",
     "https://baloot.safeis.sa/gameplay",
     "1b27febda31acb4b2c1490ac5762aea19360c269c241b111cc4eae84ebceb9d6", b"<!DOCTYPE html>"),
])
def test_draft_official_source_snapshot_is_verified_and_fails_closed(
    root: Path, tmp_path: Path, source_id: str, filename: str, url: str, expected: str, magic: bytes,
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v9.yaml")
    catalog = load_exact_rule_source_catalog(rules, root)
    source = next(row for row in catalog.sources if row["id"] == source_id)
    relative = f"configs/rules/sources/{filename}"
    assert source["url"] == url
    assert source["snapshot_url"].startswith("https://")
    assert source["content_sha256"] == expected
    assert source["status"] == "external_snapshot_hash_recorded_not_retained"
    for base in (root, root / "src/baloot_platform/resources"):
        checked = load_exact_rule_source_catalog(rules, base)
        receipt = next(row for row in checked.verify_retained_sources(base) if row["sourceId"] == source_id)
        assert checked.sha256 == catalog.sha256
        assert receipt == {"sourceId": source_id, "status": "not_retained", "expectedSha256": expected}
        assert not (base / relative).exists()
    # Retained-byte validation still rejects missing/corrupt owned synthetic data.
    owned = catalog.data
    owned_source = next(row for row in owned["sources"] if row["id"] == source_id)
    payload = b"owned synthetic source fixture"
    owned_source["snapshot_url"] = "repo://" + relative
    owned_source["content_sha256"] = hashlib.sha256(payload).hexdigest()
    synthetic = RuleSourceCatalogV1(owned, rules)
    for receipt in catalog.verify_retained_sources(root):
        if receipt["status"] == "verified":
            target = tmp_path / receipt["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / receipt["path"]).read_bytes())
    with pytest.raises(ValueError, match="retained rule source is missing"):
        synthetic.verify_retained_sources(tmp_path)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="retained rule source hash mismatch"):
        synthetic.verify_retained_sources(tmp_path)
    target.write_bytes(payload)
    assert next(row for row in synthetic.verify_retained_sources(tmp_path) if row["sourceId"] == source_id)["status"] == "verified"


def test_draft_source_successor_cannot_reuse_historical_certification_or_invent_a_tie_variant(root: Path):
    old = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    draft = RuleBook.load(root / "configs/rules/saudi_federation_2023_v9.yaml")
    for catalog, other in ((load_exact_rule_source_catalog(old, root), draft),
                           (load_exact_rule_source_catalog(draft, root), old)):
        with pytest.raises(ValueError, match="exact rules profile"):
            RuleSourceCatalogV1(catalog.data, other)
    certification = yaml.safe_load((root / "configs/certification/rules_saudi_federation_2023_v8.yaml").read_text(
        encoding="utf-8"))
    with pytest.raises(ValueError, match="profile identity is stale"):
        validate_certification_rules_identity(certification, draft)
    with pytest.raises(ValueError, match="exact rules profile"):
        load_bound_rule_source_catalog(certification, draft, root)
    unsupported = copy.deepcopy(draft.data)
    unsupported["scoring"]["match"]["tie_above_target_continues"] = False
    with pytest.raises(ValueError, match="tie_above_target_continues"):
        RuleBook(unsupported)


def test_official_profile_schema_and_source_catalog_cover_every_leaf(root: Path):
    profile_path = root / "configs/rules/saudi_federation_2023.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (root / "schemas/profiles/rules.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(profile)

    rules = RuleBook(profile)
    catalog = RuleSourceCatalogV1.load(
        root / "configs/rules/source_catalog_v1.yaml", rules,
    )
    assert catalog.coverage
    assert set(profile["profile"]["source_ids"]) <= {
        source["id"] for source in catalog.sources
    }
    assert all(binding["runtime_owner"] and binding["cp_sat_owner"] for binding in catalog.coverage.values())
    assert all(
        (root / binding[owner]).is_file()
        for binding in catalog.coverage.values()
        for owner in ("runtime_owner", "cp_sat_owner")
    )
    source_receipts = catalog.verify_retained_sources(root)
    assert [item["sourceId"] for item in source_receipts if item["status"] == "verified"] == [
        "balootx_configured_gap_2026"
    ]
    assert all(
        item["status"] == "not_retained"
        for item in source_receipts
        if item["sourceId"] != "balootx_configured_gap_2026"
    )

    changed = copy.deepcopy(profile)
    changed["profile"]["version"] = "1.0.1"
    with pytest.raises(ValueError, match="exact rules profile"):
        RuleSourceCatalogV1.load(
            root / "configs/rules/source_catalog_v1.yaml", RuleBook(changed),
        )


@pytest.mark.parametrize("surface", ("input", "data", "binding", "bindings", "coverage", "sources"))
def test_rule_source_catalog_snapshots_cannot_mutate_frozen_identity(root: Path, surface: str):
    document = yaml.safe_load(
        (root / "configs/rules/source_catalog_v1.yaml").read_text(encoding="utf-8")
    )
    catalog = RuleSourceCatalogV1(
        document, RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    )
    expected_data = copy.deepcopy(catalog.data)
    expected_coverage = copy.deepcopy(catalog.coverage)
    expected_sources = copy.deepcopy(catalog.sources)
    leaf = next(iter(expected_coverage))
    targets = {
        "input": document["bindings"][0],
        "data": catalog.data["bindings"][0],
        "binding": catalog.binding_for(leaf),
        "bindings": catalog.bindings[0],
        "coverage": catalog.coverage[leaf],
        "sources": catalog.sources[0],
    }
    if surface == "sources":
        targets[surface]["url"] = "https://invalid.example/changed"
    else:
        targets[surface]["source_ids"].append("unvalidated-source")

    assert catalog.data == expected_data
    assert catalog.coverage == expected_coverage
    assert catalog.sources == expected_sources
    assert catalog.sha256 == hashlib.sha256(json.dumps(
        catalog.data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("digest", ("g" * 64, 10 ** 63, True, None, "a" * 63))
def test_rule_source_catalog_rejects_invalid_source_digest(root: Path, digest):
    document = yaml.safe_load(
        (root / "configs/rules/source_catalog_v1.yaml").read_text(encoding="utf-8")
    )
    document["sources"][0]["content_sha256"] = digest
    with pytest.raises(ValueError, match="rule source metadata"):
        RuleSourceCatalogV1(
            document, RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
        )


def test_rule_source_catalog_preserves_uppercase_external_digest(root: Path):
    document = yaml.safe_load(
        (root / "configs/rules/source_catalog_v1.yaml").read_text(encoding="utf-8")
    )
    digest = document["sources"][0]["content_sha256"].upper()
    document["sources"][0]["content_sha256"] = digest
    catalog = RuleSourceCatalogV1(
        document, RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    )
    assert catalog.sources[0]["content_sha256"] == digest
    assert catalog.verify_retained_sources(root)[0]["status"] == "not_retained"


@pytest.mark.parametrize("uppercase", [False, True])
def test_retained_rule_source_accepts_digest_case(root: Path, uppercase: bool):
    document = yaml.safe_load(
        (root / "configs/rules/source_catalog_v1.yaml").read_text(encoding="utf-8")
    )
    retained = next(
        source for source in document["sources"]
        if source["snapshot_url"].startswith("repo://")
    )
    if uppercase:
        retained["content_sha256"] = retained["content_sha256"].upper()
    catalog = RuleSourceCatalogV1(
        document, RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    )
    receipt = next(
        item for item in catalog.verify_retained_sources(root)
        if item["sourceId"] == retained["id"]
    )
    assert receipt["status"] == "verified"
    assert receipt["expectedSha256"] == retained["content_sha256"]
    assert receipt["actualSha256"] == retained["content_sha256"].lower()


def test_retained_rule_source_hash_mismatch_fails_closed(root: Path, tmp_path: Path):
    profile = yaml.safe_load(
        (root / "configs/rules/saudi_federation_2023.yaml").read_text(
            encoding="utf-8"
        )
    )
    catalog_data = yaml.safe_load(
        (root / "configs/rules/source_catalog_v1.yaml").read_text(encoding="utf-8")
    )
    copied_root = tmp_path / "repo"
    copied_profile = copied_root / "configs/rules/saudi_federation_2023.yaml"
    copied_profile.parent.mkdir(parents=True)
    copied_profile.write_text("tampered\n", encoding="utf-8")
    catalog = RuleSourceCatalogV1(catalog_data, RuleBook(profile))

    with pytest.raises(ValueError, match="retained rule source hash mismatch"):
        catalog.verify_retained_sources(copied_root)


def test_certification_catalog_and_profile_substitution_fail_closed(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    certification = yaml.safe_load(
        (
            root
            / "configs/certification/rules_saudi_federation_2023.yaml"
        ).read_text(encoding="utf-8")
    )
    catalog = load_bound_rule_source_catalog(certification, rules, root)
    assert catalog is not None

    stale_catalog = copy.deepcopy(certification)
    stale_catalog["rule_source_catalog"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="catalog identity is stale"):
        load_bound_rule_source_catalog(stale_catalog, rules, root)

    stale_profile = copy.deepcopy(certification)
    stale_profile["candidate_profile_identity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="profile identity is stale"):
        validate_certification_rules_identity(stale_profile, rules)


@pytest.mark.parametrize("escape", ("traversal", "absolute", "symlink"))
def test_certification_catalog_path_is_confined_before_loading(root, tmp_path, monkeypatch, escape):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    certification = yaml.safe_load(
        (root / "configs/certification/rules_saudi_federation_2023.yaml").read_text(encoding="utf-8"))
    repository = tmp_path / "repository"
    configs = repository / "configs"
    configs.mkdir(parents=True)
    outside = repository / "outside.yaml"
    outside.write_bytes((root / "configs" / certification["rule_source_catalog"]["path"]).read_bytes())
    if escape == "symlink":
        link = configs / "linked.yaml"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"host does not permit symlinks: {exc}")
        target = "linked.yaml"
    else:
        target = "../outside.yaml" if escape == "traversal" else str(outside)
    certification["rule_source_catalog"]["path"] = target
    def forbidden(*args, **kwargs):
        pytest.fail("escaped catalog was opened before confinement")
    monkeypatch.setattr(RuleSourceCatalogV1, "load", forbidden)
    with pytest.raises(ValueError, match="catalog escapes configuration root"):
        load_bound_rule_source_catalog(certification, rules, repository)


@pytest.mark.parametrize(
    (
        "rules_name",
        "catalog_name",
        "certification_name",
        "adjudication_profile",
        "eligible",
    ),
    (
        (
            "saudi_federation_2023_v2.yaml",
            "source_catalog_saudi_federation_2023_v2.yaml",
            "rules_saudi_federation_2023_v2.yaml",
            "adjudication/research_events_v2.yaml",
            True,
        ),
        (
            "saudi_project_optional_unratified_v1.yaml",
            "source_catalog_saudi_project_optional_unratified_v1.yaml",
            "rules_saudi_project_optional_unratified_v1.yaml",
            "adjudication/research_events_false_ekka_unratified_v1.yaml",
            False,
        ),
    ),
)
def test_successor_and_optional_profiles_bind_exact_source_and_certification_authority(
    root: Path,
    rules_name: str,
    catalog_name: str,
    certification_name: str,
    adjudication_profile: str,
    eligible: bool,
) -> None:
    rules = RuleBook.load(root / "configs/rules" / rules_name)
    certification = yaml.safe_load(
        (root / "configs/certification" / certification_name).read_text(
            encoding="utf-8"
        )
    )
    catalog = load_bound_rule_source_catalog(certification, rules, root)

    assert catalog is not None
    validate_certification_adjudication_identity(certification, root)
    assert certification["adjudication_profile"] == adjudication_profile
    adjudication = yaml.safe_load(
        (
            root
            / "configs"
            / certification["adjudication_profile"]
        ).read_text(encoding="utf-8")
    )
    receipt_path = root / adjudication["sourceAuthority"][
        "verificationReceiptPath"
    ]
    receipt_bytes = receipt_path.read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == adjudication[
        "sourceAuthority"
    ]["verificationReceiptSha256"]
    receipt = json.loads(receipt_bytes)
    assert receipt["supersedesEvidenceId"] == (
        "saudi-baloot-official-source-verification-20260831"
    )
    claims = {item["claim"]: item for item in receipt["verifiedClauses"]}
    assert "false_ekka_has_official_opponent_choice_remedy" not in claims
    assert claims[
        "incorrect_highest_card_naming_has_official_opponent_choice_remedy"
    ]["status"] == "confirmed_text_only_interpretation_not_false_ekka"
    assert certification["certification_eligible"] is eligible
    assert certification["candidate_profile_identity"] == {
        "id": rules.id,
        "version": rules.version,
        "sha256": rules.sha256,
    }
    assert all(binding["clause_locator"] for binding in catalog.bindings)
    assert all(
        binding["clause_locator"].startswith("section_")
        for binding in catalog.bindings
        if binding["authority_class"] == "official_regulation"
    )
    retained = [
        item
        for item in catalog.verify_retained_sources(root)
        if item["status"] == "verified"
    ]
    assert len(retained) == 1
    assert retained[0]["path"] == f"configs/rules/{rules_name}"
    assert catalog.binding_for("projects.declaration_mode")[
        "ratification_status"
    ] == "external_referee_required"

    stale_adjudication = copy.deepcopy(certification)
    stale_adjudication["adjudication_authority_identity"][
        "authority_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="adjudication authority identity is stale"):
        validate_certification_adjudication_identity(stale_adjudication, root)


@pytest.mark.parametrize(
    ("rules_name", "catalog_name"),
    (
        ("saudi_federation_2023.yaml", "source_catalog_v1.yaml"),
        (
            "saudi_federation_2023_v2.yaml",
            "source_catalog_saudi_federation_2023_v2.yaml",
        ),
        (
            "saudi_project_optional_unratified_v1.yaml",
            "source_catalog_saudi_project_optional_unratified_v1.yaml",
        ),
        (
            "saudi_federation_2023_v9.yaml",
            "source_catalog_saudi_federation_2023_v9.yaml",
        ),
    ),
)
def test_native_v3_executor_binds_the_exact_rule_source_catalog(
    root: Path,
    rules_name: str,
    catalog_name: str,
) -> None:
    rules = RuleBook.load(root / "configs/rules" / rules_name)
    expected = RuleSourceCatalogV1.load(
        root / "configs/rules" / catalog_name,
        rules,
    )

    assert load_exact_rule_source_catalog(rules, root).sha256 == expected.sha256
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-NATIVE-V3",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    )
    assert executor.rule_source_catalog is not None
    assert executor.rule_source_catalog.sha256 == expected.sha256


def test_native_v3_executor_rejects_a_profile_without_an_exact_catalog(
    root: Path,
) -> None:
    legacy = RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    with pytest.raises(ValueError, match="no rule source catalog"):
        OrToolsUnifiedResearchExecutorV1(
            legacy,
            bundle_revision_id="CBR-TEST-NATIVE-V3",
            bundle_hash="a" * 64,
            backend_id="ortools_cp_sat_v3",
        )


def test_native_v3_executor_preserves_selected_catalog_among_revisions(root, tmp_path, monkeypatch):
    from baloot_platform.research import ortools_research_executor as module

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    selected = RuleSourceCatalogV1.load(root / "configs/rules/source_catalog_saudi_federation_2023_v2.yaml", rules)
    revised = selected.data
    revised["catalog"]["version"] = "1.0.1"
    directory = tmp_path / "configs/rules"
    directory.mkdir(parents=True)
    for name, data in (("selected", selected.data), ("revised", revised)):
        (directory / f"source_catalog_{name}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(module, "platform_root", lambda: tmp_path)
    parameters = dict(bundle_revision_id="CBR-TEST-NATIVE-V3", bundle_hash="a" * 64, backend_id="ortools_cp_sat_v3")
    with pytest.raises(ValueError, match="ambiguous rule source catalogs"):
        OrToolsUnifiedResearchExecutorV1(rules, **parameters)
    snapshot = {"data": selected.data, "sha256": selected.sha256}
    executor = OrToolsUnifiedResearchExecutorV1(rules, rule_source_catalog=snapshot, **parameters)
    assert executor.rule_source_catalog.sha256 == selected.sha256
    snapshot["data"]["catalog"]["version"] = "mutated"
    assert executor.rule_source_catalog.sha256 == selected.sha256
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        OrToolsUnifiedResearchExecutorV1(rules, rule_source_catalog=snapshot, **parameters)
    with pytest.raises(ValueError, match="exact rules profile"):
        OrToolsUnifiedResearchExecutorV1(RuleBook.load(root / "configs/rules/saudi_common_v1.yaml"),
            rule_source_catalog={"data": selected.data, "sha256": selected.sha256}, **parameters)


def _hundreds(rules: RuleBook) -> tuple[Project, Project]:
    domain = rules.project_domain
    sequence = Project(
        Seat.NORTH, "hundred_sequence",
        tuple(Card.of(Suit.HEARTS, rank) for rank in (Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING)),
        10, int(Rank.KING), domain.category_priorities["hundred_sequence"],
        declared=True, declaration_order=2, scoring=True,
    )
    quartet = Project(
        Seat.WEST, "hundred_four_kind",
        tuple(Card.of(suit, Rank.TEN) for suit in Suit),
        10, int(Rank.TEN), domain.category_priorities["hundred_four_kind"],
        declared=True, declaration_order=1, scoring=True,
    )
    return sequence, quartet


def test_official_project_priority_and_baloot_in_hundred_preserve_legacy(root: Path):
    legacy, official = _profiles(root)
    legacy_sequence, legacy_quartet = _hundreds(legacy)
    official_sequence, official_quartet = _hundreds(official)

    assert legacy.project_domain.comparison_winner((legacy_sequence, legacy_quartet)) == legacy_quartet
    assert official.project_domain.comparison_winner((official_sequence, official_quartet)) == official_sequence
    assert legacy.project_domain.baloot_project_state(
        (legacy_sequence,), owner=Seat.NORTH, trump_suit=Suit.HEARTS,
    ) == (False, True)
    assert official.project_domain.baloot_project_state(
        (official_sequence,), owner=Seat.NORTH, trump_suit=Suit.HEARTS,
    ) == (True, False)


def test_official_kabout_and_sun_double_boundary_preserve_legacy(root: Path):
    legacy, official = _profiles(root)
    legacy_scoring = ScoringDomainIRV1.from_rules(legacy.data)
    official_scoring = ScoringDomainIRV1.from_rules(official.data)
    inputs = dict(
        contract=Contract.SUN, winner=Team.NORTH_SOUTH,
        declarer_team=Team.WEST_EAST, escalation="double",
        total_before_award=(0, 0), project_points=(0, 0),
        baloot_points=(0, 0), kabout=True,
    )
    assert legacy_scoring.round_award(**inputs) == (88, 0)
    assert official_scoring.round_award(**inputs) == (44, 0)

    with pytest.raises(ValueError, match="split-score"):
        EscalationDomainIRV1.from_rules(legacy.data).transition(
            EscalationDomainIRV1.from_rules(legacy.data).initial_state(),
            declarer=Seat.SOUTH, contract=Contract.SUN, match_score=(101, 100),
            actor=Seat.EAST, action="double",
        )
    domain = EscalationDomainIRV1.from_rules(official.data)
    assert domain.transition(
        domain.initial_state(), declarer=Seat.SOUTH, contract=Contract.SUN,
        match_score=(101, 100), actor=Seat.EAST, action="double",
    ).state.level == "double"


def test_official_exposed_ace_and_ashkal_roles_preserve_legacy(root: Path):
    legacy, official = _profiles(root)
    legacy_domain = AuctionDomainIRV1.from_rules(legacy.data)
    official_domain = AuctionDomainIRV1.from_rules(official.data)

    for domain, expected_second_recall in (
        (legacy_domain, ("pass", "sun")),
        (official_domain, ("pass",)),
    ):
        machine = BiddingMachine(Seat.EAST, Suit.HEARTS, domain, Rank.ACE)
        machine.submit("pass")
        machine.submit("pass")
        machine.submit("hokum")
        machine.submit("pass")
        assert machine.legal_actions() == expected_second_recall
        machine.submit("pass")
        assert machine.legal_actions() == ("pass", "sun")

    legacy_ashkal = legacy_domain.resolve_action(
        "ashkal", round_index=1, actor=Seat.SOUTH,
        purchase_suit=Suit.HEARTS, trump=None,
    )
    official_ashkal = official_domain.resolve_action(
        "ashkal", round_index=1, actor=Seat.SOUTH,
        purchase_suit=Suit.HEARTS, trump=None,
    )
    assert legacy_ashkal.buyer == legacy_ashkal.purchase_recipient == Seat.NORTH
    assert official_ashkal.buyer == Seat.SOUTH
    assert official_ashkal.purchase_recipient == Seat.NORTH

    engine = BalootEngine(official)
    state = engine.create_round(seed=7, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    purchase_card = state.purchase_card
    assert purchase_card is not None
    engine.bid_and_complete_deal(
        state, forced_contract=Contract.ASHKAL, forced_buyer=Seat.EAST,
    )
    assert state.buyer == state.declarer == Seat.EAST
    assert state.purchase_recipient == Seat.WEST
    assert purchase_card in state.hand(Seat.WEST)
    assert all(len(state.hand(seat)) == 8 for seat in Seat)


@pytest.mark.parametrize(
    ("rank", "legal_actions"),
    ((Rank.ACE, ("pass",)), (Rank.KING, ("pass", "sun"))),
)
def test_official_cp_sat_binds_exposed_rank_to_recall_legality(
    root: Path,
    rank: Rank,
    legal_actions: tuple[str, ...],
):
    _, official = _profiles(root)
    model = OrToolsScenarioModelV1(rules=official, dealer=Seat.EAST)
    model.model.add(model.purchase_rank == int(rank))
    model.add_auction_fragment(
        required_actions=(
            {"action": "pass"},
            {"action": "pass"},
            {"action": "hokum"},
            {"action": "pass"},
        )
    )

    result = model.solve(maximum_seconds=5)

    assert result.status == "satisfied"
    assert result.purchase_card is not None and result.purchase_card.rank == rank
    assert result.auction is not None
    assert result.auction.legal_next_actions == legal_actions


@pytest.mark.parametrize("rank", (Rank.ACE, Rank.KING))
@pytest.mark.parametrize("round_index,in_recall", ((1, False), (1, True), (2, False)))
def test_exposed_ace_restriction_covers_first_round_windows_only(root: Path, rank, round_index, in_recall):
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.research.exact_state import ExactStateNormalizerV1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.proof_service import UnifiedScenarioProofServiceV1
    from baloot_platform.research.scenario_result import _canonical_bid_authority_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    official = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    domain = AuctionDomainIRV1.from_rules(official.data)
    machine = BiddingMachine(Seat.EAST, Suit.SPADES, domain, rank)
    actions = [{"action": "pass"}] * (4 if round_index == 2 else 0)
    if in_recall:
        actions = [*actions, {"action": "pass"}, {"action": "pass"},
            {"action": "hokum", "trumpSuit": "hearts" if round_index == 2 else "spades"},
            {"action": "pass"}]
    else:
        actions = [*actions, {"action": "hokum", "trumpSuit": "hearts" if round_index == 2 else "spades"}]
    for action in actions:
        machine.submit(action["action"], trump=Suit.parse(action["trumpSuit"]) if "trumpSuit" in action else None)
    assert machine.actor == Seat.WEST
    deck = list(range(32))
    purchase_position = DealLayoutV1.build(official.structural_domain, dealer=Seat.EAST).purchase_position
    card = Card.of(Suit.SPADES, rank).id
    other = deck.index(card)
    deck[purchase_position], deck[other] = deck[other], deck[purchase_position]
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, dealer="east", deckOrder=deck,
        auctionActions=actions, escalationActions=[], projectDeclarations=[], playActions=[]))
    normalized = OrToolsExactStateNormalizerV1(official, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    expected = ("pass",) if round_index == 1 and rank == Rank.ACE else ("pass", "sun")
    actual_native = tuple(sorted(row["code"] for row in normalized.legal_actions))
    assert (machine.legal_actions(), actual_native) == (expected, expected)
    reference = ExactStateNormalizerV1(official).normalize(prefix, decision_point={"kind": "bid"})
    _, result_actions = _canonical_bid_authority_v1(reference)
    proof_machine = UnifiedScenarioProofServiceV1._bidding_machine(reference)
    assert tuple(sorted(row["code"] for row in result_actions)) == expected
    assert proof_machine.legal_actions() == expected


def test_official_cp_sat_keeps_ashkal_buyer_and_card_recipient_distinct(root: Path):
    _, official = _profiles(root)
    model = OrToolsScenarioModelV1(rules=official, dealer=Seat.EAST)
    model.add_auction_fragment(
        required_actions=(
            {"action": "pass"},
            {"action": "pass"},
            {"action": "ashkal"},
            {"action": "pass"},
            {"action": "pass"},
        ),
        required_actions_complete=True,
        allowed_contracts=(Contract.ASHKAL,),
    )

    result = model.solve(maximum_seconds=5)

    assert result.status == "satisfied"
    assert result.buyer == Seat.SOUTH
    assert result.purchase_recipient == Seat.NORTH
    assert result.purchase_card is not None and result.owner_of_card is not None
    assert result.owner_of_card[result.purchase_card.id] == Seat.NORTH
    assert result.auction is not None
    assert result.auction.completion_buyer == Seat.SOUTH
    assert result.auction.completion_purchase_recipient == Seat.NORTH
    assert result.to_dict()["purchaseRecipient"] == "north"


@pytest.mark.parametrize(
    ("rules_name", "expected_roles"),
    (
        (
            "saudi_federation_2023_v2.yaml",
            (Seat.SOUTH, Seat.SOUTH, Seat.NORTH, Seat.SOUTH),
        ),
        (
            "saudi_common_v1.yaml",
            (Seat.SOUTH, Seat.NORTH, Seat.NORTH, Seat.NORTH),
        ),
    ),
)
def test_ashkal_roles_match_runtime_reference_and_native_search(
    root: Path,
    rules_name: str,
    expected_roles: tuple[Seat, Seat, Seat, Seat],
) -> None:
    rules = RuleBook.load(root / "configs/rules" / rules_name)
    domain = AuctionDomainIRV1.from_rules(rules.data)
    runtime = BiddingMachine(Seat.EAST, Suit.HEARTS, domain, Rank.SEVEN)
    reference = ReferenceAuctionInterpreterV1(
        Seat.EAST,
        Suit.HEARTS,
        domain,
        Rank.SEVEN,
    )
    actions = ("pass", "pass", "ashkal", "pass", "pass")
    for action in actions:
        runtime.submit(action)
        reference.submit(action)

    assert runtime.result is not None
    assert reference.result is not None
    runtime_roles = (
        runtime.result.caller,
        runtime.result.buyer,
        runtime.result.purchase_recipient,
        runtime.result.declarer,
    )
    reference_roles = (
        reference.result.caller,
        reference.result.buyer,
        reference.result.purchase_recipient,
        reference.result.declarer,
    )

    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.model.add(model.purchase_rank == int(Rank.SEVEN))
    model.add_auction_fragment(
        required_actions=tuple({"action": action} for action in actions),
        required_actions_complete=True,
        allowed_contracts=(Contract.ASHKAL,),
    )
    solved = model.solve(maximum_seconds=5)
    assert solved.auction is not None
    native_roles = (
        solved.auction.caller,
        solved.auction.buyer,
        solved.auction.prefix_purchase_recipient,
        solved.auction.declarer,
    )

    assert runtime_roles == reference_roles == native_roles == expected_roles
