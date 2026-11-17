"""Observer-safe public guide derived from the active rules and convention profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.adjudication.research import AdjudicationResearchAuthorityV1
from baloot_platform.config import platform_root
from baloot_platform.core.rule_source_catalog import (
    RuleSourceCatalogV1,
    load_bound_rule_source_catalog,
)
from baloot_platform.core.rules import resolved_disputed_options


# Every admissible dynamic value is registered here so an inactive vocabulary
# entry is still executable evidence, not an unverified spare translation.
PUBLIC_GUIDE_DYNAMIC_VOCABULARIES: Mapping[str, tuple[str, ...]] = {
    "booleans": ("false", "true"),
    "contracts": ("ashkal", "hokum", "pass", "sun"),
    "disputed": (
        "double_unspecified_mode",
        "equal_hokum_sequence_trump_priority",
        "first_leader",
        "hints_have_universal_meaning",
        "kawesh_saneen",
    ),
    "escalation": ("double", "four", "gahwa", "normal", "triple"),
    "general": (
        "bidder_partner",
        "buyer",
        "dealer_left",
        "dealer_right",
        "locked",
        "open",
    ),
    "options": (
        "buyer",
        "dealer_right",
        "locked",
        "open",
        "position_aware_ace_or_ekka",
    ),
    "positions": ("buyer", "dealer_right"),
    "ranks": ("ace", "eight", "jack", "king", "nine", "queen", "seven", "ten"),
    "signals": (
        "high_expenditure",
        "high_then_low",
        "low_then_high",
        "minimum_winner",
        "partner_return",
    ),
    "timing": ("after_first_trick", "during_second_trick", "first_trick_turn"),
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _localized(value: Any, terms: Mapping[str, Any]) -> str:
    key = _text(value)
    if key not in terms:
        raise ValueError(f"public guide localization is missing: {key}")
    return str(terms[key])


def _section(
    section_id: str,
    presentation: Mapping[str, Any],
    *,
    facts: list[dict[str, str]],
    tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "title": str(presentation["title"]),
        "summary": str(presentation["summary"]),
        "facts": facts,
        "examples": [str(item) for item in presentation.get("examples") or ()],
        "tables": list(tables or ()),
    }


def _fact(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def _rule_settings(
    value: Any,
    path: tuple[str, ...] = (),
    *,
    source_catalog: RuleSourceCatalogV1 | None = None,
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(value):
            result.extend(
                _rule_settings(
                    value[key],
                    (*path, str(key)),
                    source_catalog=source_catalog,
                )
            )
        return result
    rule_path = ".".join(path)
    if rule_path == "schema_version":
        authority_class, enforcement = "protocol_invariant", "schema_validation"
    elif rule_path.startswith("profile."):
        authority_class, enforcement = "provenance_metadata", "identity_only"
    elif rule_path.startswith("bidding.heuristics."):
        authority_class, enforcement = "heuristic", "policy_only_not_qualification"
    else:
        authority_class, enforcement = "configured_rule", "runtime_or_fail_closed"
    entry = {
        "path": rule_path,
        "value": value,
        "authorityClass": authority_class,
        "enforcement": enforcement,
    }
    if source_catalog is not None:
        binding = source_catalog.binding_for(rule_path)
        entry.update(
            {
                "authorityClass": str(binding["authority_class"]),
                "sourceIds": [str(item) for item in binding["source_ids"]],
                "clauseLocator": str(binding["clause_locator"]),
                "clause": str(binding["clause"]),
                "applicability": str(binding["applicability"]),
                "examples": [str(item) for item in binding["examples"]],
                "counterexamples": [
                    str(item) for item in binding["counterexamples"]
                ],
                "runtimeOwner": str(binding["runtime_owner"]),
                "cpSatOwner": str(binding["cp_sat_owner"]),
                "ratificationStatus": str(binding["ratification_status"]),
            }
        )
    return [entry]


def _card_table(
    *,
    title: str,
    ranks: list[str],
    strengths: list[int],
    points: list[int],
    rank_terms: Mapping[str, Any],
    suit: str,
) -> dict[str, Any]:
    asset_rank = {
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "jack": "J",
        "queen": "Q",
        "king": "K",
        "ace": "A",
    }
    order = sorted(range(len(ranks)), key=lambda index: strengths[index], reverse=True)
    return {
        "title": title,
        "cards": [
            {
                "rank": str(rank_terms[ranks[index]]),
                "points": int(points[index]),
                "asset": f"/card-assets/{asset_rank[ranks[index]]}_{suit}.svg",
            }
            for index in order
        ],
    }


def build_engine_guide(
    bundle: Any, *, root: str | Path | None = None
) -> dict[str, Any]:
    """Build one deterministic API/UI guide from the effective runtime bundle."""

    rules = dict(bundle.components["rules"].data)
    conventions = dict(bundle.component("conventions"))
    product = dict(bundle.component("product"))
    adjudication = dict(bundle.component("adjudication"))
    repository_root = platform_root(root)
    adjudication_authority = dict(adjudication.get("sourceAuthority") or {})
    adjudication_profile = (
        AdjudicationResearchAuthorityV1.from_mapping(adjudication)
        if adjudication_authority
        else None
    )
    adjudication_authoring_catalog = (
        adjudication_profile.authoring_catalog()
        if adjudication_profile is not None
        else None
    )
    if adjudication_authority:
        receipt_path = (
            repository_root
            / str(adjudication_authority["verificationReceiptPath"])
        ).resolve()
        if not receipt_path.is_relative_to(repository_root.resolve()):
            raise ValueError("adjudication authority receipt escapes repository root")
        if not receipt_path.is_file():
            raise ValueError("adjudication authority receipt is missing")
        receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        if receipt_hash != str(
            adjudication_authority["verificationReceiptSha256"]
        ):
            raise ValueError("adjudication authority receipt hash is stale")
    source_catalog = load_bound_rule_source_catalog(
        bundle.component("certification"),
        bundle.rules,
        repository_root,
    )
    guide = dict(product["public_guide"])
    labels = dict(guide["labels"])
    terms = dict(guide["terms"])
    configured_vocabularies = {
        str(group): tuple(sorted(str(key) for key in dict(values)))
        for group, values in terms.items()
    }
    expected_vocabularies = {
        group: tuple(sorted(values))
        for group, values in PUBLIC_GUIDE_DYNAMIC_VOCABULARIES.items()
    }
    legacy_vocabularies = {
        group: values
        for group, values in expected_vocabularies.items()
        if group != "escalation"
    }
    if configured_vocabularies not in (expected_vocabularies, legacy_vocabularies):
        raise ValueError(
            "public guide dynamic vocabulary does not match its executable registry"
        )
    has_escalation_guide = configured_vocabularies == expected_vocabularies
    sections = dict(guide["sections"])
    markdown = dict(guide["markdown"])
    ranks = list(rules["deck"]["ranks"])
    card_rules = dict(rules["card_rules"])
    bidding = dict(rules["bidding"])
    escalation = dict(rules["escalation"])
    play = dict(rules["play"])
    projects = dict(rules["projects"])
    scoring = dict(rules["scoring"])

    contract_terms = dict(terms["contracts"])
    general_terms = dict(terms["general"])
    position_terms = dict(terms["positions"])
    timing_terms = dict(terms["timing"])
    option_terms = dict(terms["options"])
    boolean_terms = dict(terms["booleans"])
    rank_terms = dict(terms["ranks"])
    signal_terms = dict(terms["signals"])
    escalation_terms = dict(terms.get("escalation", {}))

    def local_list(values: Any, mapping: Mapping[str, Any]) -> str:
        return " · ".join(_localized(item, mapping) for item in values or ())

    def yes_no(value: Any) -> str:
        return str(boolean_terms["true" if bool(value) else "false"])

    first_round = dict(bidding["first_round"])
    second_round = dict(bidding["second_round"])
    ashkal = dict(bidding["ashkal"])
    sequence_values = dict(projects["sequences"]["values_game_points"])
    four_values = dict(projects["four_of_kind"]["values_game_points"])
    baloot = dict(projects["baloot"])
    kabout = dict(scoring["kabout"])
    match = dict(scoring["match"])
    capability_copy = dict(guide["capability"])
    capability_entries = _rule_settings(rules, source_catalog=source_catalog)
    disputed_selected = {
        key: value["selected"]
        for key, value in resolved_disputed_options(rules).items()
    }

    output_sections = [
        _section(
            "buying",
            sections["buying"],
            facts=[
                _fact(
                    labels["contracts"],
                    local_list(rules["contracts"]["allowed"], contract_terms),
                ),
                _fact(
                    labels["first_round"],
                    local_list(first_round["allowed"], contract_terms),
                ),
                _fact(
                    labels["second_round"],
                    local_list(second_round["allowed"], contract_terms),
                ),
                _fact(
                    labels["ashkal_owner"], _localized(ashkal["taker"], general_terms)
                ),
                _fact(
                    labels["second_hokum_suit"],
                    yes_no(second_round["hokum_must_differ_from_purchase_suit"]),
                ),
            ],
        ),
        _section(
            "card_order",
            sections["card_order"],
            facts=[
                _fact(
                    labels["deck_size"], _text(len(rules["deck"]["suits"]) * len(ranks))
                ),
                _fact(
                    labels["cards_per_player"], _text(rules["deck"]["cards_per_player"])
                ),
            ],
            tables=[
                _card_table(
                    title=labels["sun_order"],
                    ranks=ranks,
                    strengths=list(card_rules["sun_strength"]),
                    points=list(card_rules["sun_points"]),
                    rank_terms=rank_terms,
                    suit="hearts",
                ),
                _card_table(
                    title=labels["hokum_order"],
                    ranks=ranks,
                    strengths=list(card_rules["hokum_trump_strength"]),
                    points=list(card_rules["hokum_trump_points"]),
                    rank_terms=rank_terms,
                    suit="spades",
                ),
            ],
        ),
        *(
            [
                _section(
                    "escalation",
                    sections["escalation"],
                    facts=[
                        _fact(
                            labels["escalation_enabled"], yes_no(escalation["enabled"])
                        ),
                        _fact(
                            labels["hokum_escalation"],
                            local_list(escalation["hokum_chain"], escalation_terms),
                        ),
                        _fact(
                            labels["sun_escalation"],
                            local_list(escalation["sun_chain"], escalation_terms),
                        ),
                        _fact(
                            labels["escalation_mode"],
                            _localized(escalation["default_mode"], option_terms),
                        ),
                        _fact(
                            labels["sun_double_threshold"],
                            _text(escalation["sun_score_threshold"]),
                        ),
                    ],
                )
            ]
            if has_escalation_guide
            else []
        ),
        _section(
            "legal_play",
            sections["legal_play"],
            facts=[
                _fact(
                    labels["first_leader"],
                    _localized(play["first_leader"], position_terms),
                ),
                _fact(labels["follow_suit"], yes_no(play["must_follow_suit"])),
                _fact(labels["must_trump"], yes_no(play["must_trump_when_void"])),
                _fact(
                    labels["must_overtrump"],
                    yes_no(play["must_overtrump_when_possible"]),
                ),
                _fact(
                    labels["partner_exception"],
                    _localized(play["partner_winning_exception"], option_terms),
                ),
                _fact(
                    labels["locked_hokum"],
                    yes_no(play["locked_hokum_blocks_trump_lead"]),
                ),
            ],
        ),
        _section(
            "projects",
            sections["projects"],
            facts=[
                _fact(
                    labels["project_mention"],
                    _localized(projects["declaration_timing"], timing_terms),
                ),
                _fact(
                    labels["project_compare"],
                    _localized(projects["comparison_timing"], timing_terms),
                ),
                _fact(
                    labels["project_reveal"],
                    _localized(projects["reveal_timing"], timing_terms),
                ),
                _fact(
                    labels["sun_sequences"],
                    " / ".join(
                        _text(sequence_values["sun"][key])
                        for key in ("three", "four", "five_plus")
                    ),
                ),
                _fact(
                    labels["hokum_sequences"],
                    " / ".join(
                        _text(sequence_values["hokum"][key])
                        for key in ("three", "four", "five_plus")
                    ),
                ),
                _fact(labels["four_hundred"], _text(four_values["sun"]["ace"])),
                _fact(labels["baloot_value"], _text(baloot["value_game_points"])),
            ],
        ),
        _section(
            "scoring",
            sections["scoring"],
            facts=[
                _fact(labels["match_target"], _text(match["target"])),
                _fact(
                    labels["sun_raw_total"], _text(scoring["raw_card_totals"]["sun"])
                ),
                _fact(
                    labels["hokum_raw_total"],
                    _text(scoring["raw_card_totals"]["hokum"]),
                ),
                _fact(labels["last_trick"], _text(scoring["last_trick_bonus_raw"])),
                _fact(labels["sun_kabout"], _text(kabout["base_game_points"]["sun"])),
                _fact(
                    labels["hokum_kabout"], _text(kabout["base_game_points"]["hokum"])
                ),
                _fact(
                    labels["tie_continues"], yes_no(match["tie_above_target_continues"])
                ),
            ],
        ),
        _section(
            "reading",
            sections["reading"],
            facts=[
                _fact(
                    labels["probability_application"],
                    yes_no(conventions["apply_to_probability"]),
                ),
                *[
                    _fact(
                        str(signal_terms[signal_id]),
                        str(
                            labels["relative_hint"]
                            if signal["enabled"]
                            else boolean_terms["false"]
                        ),
                    )
                    for signal_id, signal in conventions["signals"].items()
                ],
            ],
        ),
        _section(
            "authority",
            sections["authority"],
            facts=[
                _fact(labels["capability_settings"], _text(len(capability_entries))),
                _fact(
                    labels["capability_enforcement"],
                    str(capability_copy["enforcement"]),
                ),
                _fact(labels["capability_evidence"], str(capability_copy["evidence"])),
            ],
        ),
        _section(
            "disputed",
            sections["disputed"],
            facts=[
                _fact(
                    str(dict(terms["disputed"])[option_id]),
                    _localized(
                        disputed_selected[option_id],
                        {**general_terms, **option_terms, **boolean_terms},
                    ),
                )
                for option_id in rules["disputed_options"]
            ],
        ),
    ]

    core = {
        "schemaVersion": 1,
        "locale": str(guide["locale"]),
        "direction": str(guide["direction"]),
        "pageTitle": str(guide["page_title"]),
        "brand": str(guide["brand"]),
        "subtitle": str(guide["subtitle"]),
        "interface": {
            "searchPlaceholder": str(guide["search_placeholder"]),
            "noResults": str(guide["no_results"]),
            "examplesLabel": str(guide["examples_label"]),
            "pointsLabel": str(guide["points_label"]),
            "auditLabel": str(guide["audit_label"]),
            "homeLabel": str(guide["home_label"]),
            "foundryLabel": str(guide["foundry_label"]),
            "versionPrefix": str(guide["version_prefix"]),
            "markdownCardColumn": str(markdown["card_column"]),
            "markdownPointsColumn": str(markdown["points_column"]),
            "markdownAuditHeading": str(markdown["audit_heading"]),
            "markdownGeneratedNotice": str(markdown["generated_notice"]),
        },
        "profile": {
            "label": str(labels["active_profile"]),
            "id": str(rules["profile"]["id"]),
            "version": str(rules["profile"]["version"]),
            "status": _localized(
                rules["profile"]["status"], dict(guide["status_labels"])
            ),
            "expertRatified": bool(rules["profile"]["expert_ratified"]),
        },
        "proofBoundary": {
            "title": str(guide["proof_boundary"]["title"]),
            "body": str(guide["proof_boundary"]["body"]),
        },
        "sections": output_sections,
        "capabilityReport": {
            "schemaVersion": 1,
            "rulesHash": bundle.rules.sha256,
            "totalSettings": len(capability_entries),
            "enforcement": "classified_per_setting",
            "entries": capability_entries,
        },
        "ruleSourceCatalog": (
            {
                "schemaVersion": 1,
                "id": str(source_catalog.data["catalog"]["id"]),
                "version": str(source_catalog.data["catalog"]["version"]),
                "sha256": source_catalog.sha256,
                "certificationStatus": str(
                    source_catalog.data["catalog"]["certification_status"]
                ),
                "sources": [
                    {
                        "id": str(source["id"]),
                        "authorityClass": str(source["authority_class"]),
                        "url": str(source["url"]),
                        "snapshotUrl": str(source["snapshot_url"]),
                        "contentSha256": str(source["content_sha256"]),
                        "status": str(source["status"]),
                    }
                    for source in source_catalog.sources
                ],
                "retainedSourceReceipts": [
                    dict(item)
                    for item in source_catalog.verify_retained_sources(repository_root)
                ],
            }
            if source_catalog is not None
            else None
        ),
        "adjudicationAuthority": (
            {
                "schemaVersion": 1,
                "authoritySha256": adjudication_profile.sha256,
                "authoringCatalogSha256": adjudication_authoring_catalog[
                    "sha256"
                ],
                "profile": {
                    "id": adjudication_profile.profile_id,
                    "version": adjudication_profile.version,
                    "status": adjudication_profile.status,
                },
                "sourceAuthority": adjudication_authority,
                "events": [
                    {
                        "id": str(event_id),
                        "authorityClass": str(event["authorityClass"]),
                        "sourceClause": str(event["sourceClause"]),
                        "generatedSearch": str(event["generatedSearch"]),
                        "exactInput": str(event["exactInput"]),
                        "hardPredicate": str(event["hardPredicate"]),
                        "uiExposure": str(event["uiExposure"]),
                    }
                    for event_id, event in sorted(adjudication["events"].items())
                ],
            }
            if adjudication_profile is not None
            else None
        ),
        "audit": {
            "rulesHash": bundle.rules.sha256,
            "conventionHash": bundle.components["conventions"].sha256,
            "productHash": bundle.components["product"].sha256,
            "adjudicationHash": bundle.components["adjudication"].sha256,
            "bundleHash": bundle.sha256,
            "rulesStatus": str(rules["profile"]["status"]),
            "conventionProfile": str(conventions["profile"]["id"]),
            **(
                {"ruleSourceCatalogHash": source_catalog.sha256}
                if source_catalog is not None
                else {}
            ),
        },
    }
    return {**core, "guideHash": _canonical_hash(core)}
