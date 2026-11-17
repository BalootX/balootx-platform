from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from baloot_platform.adjudication import (
    AdjudicationEventEvidenceV1,
    AdjudicationProfile,
    AdjudicationResearchAuthorityV1,
    AdjudicationService,
    AuthoredAdjudicationEventV1,
    SQLiteAdjudicationStore,
)
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import iter_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.research.scenario_registry_v2 import (
    scenario_predicate_definitions_v2,
)
from baloot_platform.research.scenario_adjudication_predicates_v3 import (
    adjudication_predicate_definitions_v3,
)
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
from baloot_platform.research.scenario_execution_capabilities import (
    build_scenario_execution_capabilities_v1,
)
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.service.game import GameManager
from baloot_platform.service import api as service_api
from baloot_platform.service.api import ExactAdjudicationEventV1Request


def _authority(root: Path) -> AdjudicationResearchAuthorityV1:
    return AdjudicationResearchAuthorityV1.from_yaml(
        root / "configs/adjudication/research_events_v1.yaml"
    )


def _false_ekka_variant(root: Path) -> AdjudicationResearchAuthorityV1:
    return AdjudicationResearchAuthorityV1.from_yaml(
        root
        / "configs/adjudication/research_events_false_ekka_unratified_v1.yaml"
    )


def _active_round(root: Path):
    engine = BalootEngine(build_bundle(root).rules)
    state = engine.create_round(seed=31_001, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.HEARTS,
    )
    engine.prepare_projects(state)
    return engine, state


def test_authority_is_complete_localized_documented_and_packaged(root: Path) -> None:
    canonical = root / "configs/adjudication/research_events_v1.yaml"
    packaged = root / "src/baloot_platform/resources/configs/adjudication/research_events_v1.yaml"
    assert canonical.read_bytes() == packaged.read_bytes()
    authority = _authority(root)
    assert set(authority.capabilities) == {
        "incorrect_highest_card_naming",
        "out_of_turn",
        "invalid_bid",
        "exposed_card",
        "signalling",
        "verbal_baloot_inside_sequential_100",
    }
    assert authority.status == "source_verified_uncertified"
    assert authority.source_id == "gea_baloot_regulation_2023"
    assert authority.source_sha256 == (
        "9a7c67046549c5013f10e31069e6feba70e20eb1d47b27c7a2ba8abaa3e848c2"
    )
    assert authority.source_bytes == 3_035_758
    assert authority.current_authority_boundary.endswith(
        "requires_external_ratification"
    )
    receipt = root / authority.verification_receipt_path
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == (
        authority.verification_receipt_sha256
    )
    for capability in authority.capabilities.values():
        row = capability.to_dict()
        assert row["generatedSearch"] == "unsupported"
        assert row["hardPredicate"] == "unsupported"
        assert row["uiExposure"] == "hidden"
        assert row["requiresTableReferee"] is True
        assert row["availableRemedies"]
        assert row["sourceClause"]
        assert row["messageAr"] and row["messageEn"]
        document, anchor = row["documentation"].split("#", 1)
        text = (root / document).read_text(encoding="utf-8").lower().replace(" ", "-")
        assert anchor in text
    verbal_baloot = authority.capability("verbal_baloot_inside_sequential_100")
    assert verbal_baloot.domain_primitive == "rules.projects.baloot_verbal_declaration"
    assert verbal_baloot.ordinary_trace == "not_an_action"
    assert verbal_baloot.event_scope == "external_evidence"
    assert "opponent_register_with_projects" in verbal_baloot.available_remedies


def test_unsupported_adjudication_cases_are_not_scenario_controls(root: Path) -> None:
    assert not any(
        definition.kind.startswith("adjudication.")
        for definition in scenario_predicate_definitions_v2()
    )


def test_false_ekka_is_variant_only_and_external_judgments_stay_hidden(
    root: Path,
) -> None:
    official = _authority(root)
    variant = _false_ekka_variant(root)
    assert "false_ekka_attempt" not in official.capabilities
    assert variant.variant_status == "unratified_referee_variant"
    false_ekka = variant.capability("false_ekka_attempt")
    assert false_ekka.hard_predicate == "native_exact"
    assert false_ekka.ui_exposure == "variant_only"
    assert false_ekka.requires_table_referee is True

    catalog = {row["kind"]: row for row in variant.authoring_catalog()["events"]}
    for kind in ("incorrect_highest_card_naming", "exposed_card", "signalling"):
        assert catalog[kind]["allowedHardness"] == ["report"]
        assert catalog[kind]["generatedSearchHardSupport"] == "unsupported"
        assert catalog[kind]["uiExposure"] == "hidden"

    workflow = yaml.safe_load(
        (root / "configs/product/studio_ortools.yaml").read_text(encoding="utf-8")
    )["foundry_copy"]["groups"]["workflow"]
    for key in (
        "research_scenario_adjudication_false_ekka_attempt",
        "research_scenario_adjudication_false_ekka_attempt_en",
        "research_adjudication_kind_false_ekka_attempt",
        "research_adjudication_kind_false_ekka_attempt_en",
    ):
        assert workflow[key]

    event = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "false_ekka_attempt",
            "actionSequence": 4,
            "actorSeat": "south",
            "expectedActorSeat": "south",
            "attemptedAction": {"cardId": 12, "ekka": True},
            "reasonCode": "illegal_ekka_declaration",
        }
    )
    assert variant.validate_authored_event(event) == false_ekka
    with pytest.raises(KeyError):
        official.validate_authored_event(event)
    evidence = AdjudicationEventEvidenceV1.build(
        variant,
        kind="false_ekka_attempt",
        game_id="GAME-FALSE-EKKA",
        round_number=1,
        event_sequence=4,
        actor_seat="south",
        expected_actor_seat="south",
        attempted_action={"cardId": 12, "ekka": True},
        reason_code="illegal_ekka_declaration",
        state_sha256="a" * 64,
    )
    assert evidence.to_dict()["attemptedAction"] == {"cardId": 12, "ekka": True}
    authority = _authority(root)
    assert all(
        capability.exact_input
        in {
            "legal_trace_rejects_attempt_audit_sidecar_supported",
            "audit_sidecar_only",
        }
        for capability in authority.capabilities.values()
    )


def test_adjudication_event_evidence_is_deterministic_and_observer_safe(
    root: Path,
    tmp_path: Path,
) -> None:
    authority = _authority(root)
    values = dict(
        kind="out_of_turn",
        game_id="GAME-31",
        round_number=2,
        event_sequence=17,
        actor_seat="west",
        expected_actor_seat="south",
        attempted_action={"card": "S7"},
        reason_code="play_before_turn",
        state_sha256="a" * 64,
    )
    first = AdjudicationEventEvidenceV1.build(authority, **values)
    second = AdjudicationEventEvidenceV1.build(authority, **values)
    assert first == second
    assert first.sha256 == second.sha256
    payload = first.to_dict()
    encoded = str(payload).lower()
    assert "legalactions" not in encoded
    assert "hands" not in encoded
    assert "observer_safe" not in encoded
    assert payload["observerSafety"] == "referee_restricted"
    assert payload["availableRemedies"] == [
        "opponent_register_with_projects",
        "continue_play",
    ]
    assert payload["requiresTableReferee"] is True
    item = first.to_evidence_item(submitted_by="REF-31", description="محاولة مرفوضة")
    assert item.sha256 == first.sha256
    assert item.metadata["evidenceId"] == first.evidence_id
    service = AdjudicationService(
        SQLiteAdjudicationStore(tmp_path / "event.sqlite3"),
        AdjudicationProfile.from_yaml(
            root / "configs/adjudication/saudi_researched_v1.yaml"
        ),
    )
    case = service.open_event_case(
        first,
        reported_by="REF-31",
        description=authority.capability("out_of_turn").message_ar,
    )
    assert service.verify(case.case_id)["valid"] is True
    recommendation = service.recommended_ruling(case.case_id)
    assert recommendation["recommendedOutcome"] == "referee_choice_required"
    assert recommendation["profileId"] == authority.profile_id
    with pytest.raises(ValueError, match="supported referee remedy"):
        service.rule(
            case.case_id,
            referee_id="REF-31",
            outcome="warning",
            reason_ar="لا يعتمد الحكم العام",
        )
    with pytest.raises(ValueError, match="unsupported adjudication remedy"):
        service.rule_event(
            case.case_id,
            referee_id="REF-31",
            selected_remedy="automatic_replay",
            reason_ar="غير مدعوم",
        )
    ruled = service.rule_event(
        case.case_id,
        referee_id="REF-31",
        selected_remedy="continue_play",
        reason_ar="اختار الخصم الاستمرار",
    )
    assert ruled.rulings[-1].profile_id == authority.profile_id
    assert ruled.rulings[-1].profile_hash == authority.sha256
    directive = service.directive(case.case_id)
    assert directive["selectedRemedy"] == "continue_play"
    assert directive["adjudicationEventId"] == first.evidence_id
    with pytest.raises(ValueError, match="external referee evidence"):
        AdjudicationEventEvidenceV1.build(
            authority,
            **{**values, "kind": "incorrect_highest_card_naming"},
        )
    with pytest.raises(ValueError, match="only covers purchase_before_turn"):
        AdjudicationEventEvidenceV1.build(
            authority,
            **{
                **values,
                "kind": "invalid_bid",
                "reason_code": "generic_invalid_bid",
            },
        )
    with pytest.raises(ValueError, match="must differ from expected actor"):
        AdjudicationEventEvidenceV1.build(
            authority,
            **{**values, "actor_seat": "south"},
        )


def test_exact_trace_sidecars_are_hash_bound_and_legacy_empty_hash_is_stable(
    root: Path,
) -> None:
    raw = {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
    }
    legacy = ExactTracePrefixV1.build(raw)
    assert "adjudicationEvents" not in legacy.to_dict()
    assert ExactTracePrefixV1.build(legacy.to_dict()).sha256 == legacy.sha256

    event = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 0,
            "actorSeat": "west",
            "expectedActorSeat": "south",
            "attemptedAction": {"card": "S7"},
            "reasonCode": "play_before_turn",
        }
    )
    exact = ExactTracePrefixV1.build({**raw, "adjudicationEvents": [event.to_dict()]})
    assert exact.sha256 != legacy.sha256
    assert exact.adjudication_events == (event,)
    assert ExactTracePrefixV1.build(exact.to_dict()).sha256 == exact.sha256
    assert _authority(root).validate_authored_event(event).kind.value == "out_of_turn"


def test_exact_adjudication_authority_rejects_false_or_unowned_evidence(
    root: Path,
) -> None:
    authority = _authority(root)
    same_actor = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 0,
            "actorSeat": "south",
            "expectedActorSeat": "south",
            "attemptedAction": {"card": "S7"},
            "reasonCode": "play_before_turn",
        }
    )
    with pytest.raises(ValueError, match="must differ from expectedActorSeat"):
        authority.validate_authored_event(same_actor)

    external_without_receipt = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "incorrect_highest_card_naming",
            "actionSequence": 0,
            "actorSeat": "south",
            "reasonCode": "incorrect_highest_card_naming",
        }
    )
    with pytest.raises(ValueError, match="requires external evidenceRefs"):
        authority.validate_authored_event(external_without_receipt)

    wrong_remedy = AuthoredAdjudicationEventV1.build(
        {
            "schemaVersion": 1,
            "kind": "out_of_turn",
            "actionSequence": 0,
            "actorSeat": "west",
            "expectedActorSeat": "south",
            "attemptedAction": {"card": "S7"},
            "reasonCode": "play_before_turn",
            "selectedRemedy": "opponent_takweesh",
            "rulingEvidenceRef": "EVIDENCE-RULING-1",
        }
    )
    with pytest.raises(ValueError, match="is not a remedy"):
        authority.validate_authored_event(wrong_remedy)


def test_adjudication_authoring_catalog_is_closed_localized_and_exact_only(
    root: Path,
) -> None:
    catalog = _authority(root).authoring_catalog()
    assert catalog["fieldPath"] == (
        "scenario.origin.exactTracePrefix.adjudicationEvents"
    )
    item = catalog["exactInputItemSchema"]
    assert item["additionalProperties"] is False
    properties = item["properties"]
    assert properties["kind"]["enum"] == [
        "exposed_card",
        "incorrect_highest_card_naming",
        "invalid_bid",
        "out_of_turn",
        "signalling",
        "verbal_baloot_inside_sequential_100",
    ]
    for schema in properties.values():
        assert schema["x-messageArKey"]
        assert schema["x-messageEnKey"]
    remedies = properties["selectedRemedy"]["x-remediesByKind"]
    assert remedies["out_of_turn"] == [
        "opponent_register_with_projects",
        "continue_play",
    ]
    assert remedies["invalid_bid"] == ["opponent_takweesh", "continue_play"]
    assert all(row["allowedHardness"] == ["report"] for row in catalog["events"])


def test_selected_bundle_authority_projects_into_exact_foundry_catalog(
    root: Path,
) -> None:
    challenge = yaml.safe_load(
        (
            root
            / "configs/challenge/progressive_mining_ortools_v3_1.yaml"
        ).read_text(encoding="utf-8")
    )
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=challenge,
        bundle_identity={
            "bundleRevisionId": "CBR-ADJUDICATION-CATALOG",
            "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64,
        },
    )
    authority = _authority(root)
    execution = build_scenario_execution_capabilities_v1(
        backend,
        adjudication_authority=authority,
    )
    schema = execution["authoringCatalog"]["exactTrace"]["adjudicationEvents"]
    assert schema["type"] == "array"
    assert schema["default"] == []
    assert schema["x-authorityProfileId"] == authority.profile_id
    assert schema["x-authoritySha256"] == authority.sha256
    assert schema["items"] == authority.authoring_catalog()[
        "exactInputItemSchema"
    ]


def test_exact_adjudication_api_contract_is_strict_and_uses_domain_contract() -> None:
    payload = {
        "schemaVersion": 1,
        "kind": "out_of_turn",
        "actionSequence": 0,
        "actorSeat": "west",
        "expectedActorSeat": "south",
        "attemptedAction": {"card": "S7"},
        "reasonCode": "play_before_turn",
    }
    typed = ExactAdjudicationEventV1Request.model_validate(payload)
    assert typed.model_dump(mode="json")["kind"] == "out_of_turn"
    with pytest.raises(ValueError):
        ExactAdjudicationEventV1Request.model_validate(
            {**payload, "kind": "tournament_fee"}
        )
    with pytest.raises(ValueError):
        ExactAdjudicationEventV1Request.model_validate(
            {**payload, "attemptedAction": {"nested": {"not": "scalar"}}}
        )
    with pytest.raises(ValueError):
        ExactAdjudicationEventV1Request.model_validate(
            {**payload, "unknown": True}
        )


def test_exact_adjudication_sidecar_does_not_mutate_solver_state(root: Path) -> None:
    raw = {
        "schemaVersion": 1,
        "deckOrder": list(range(32)),
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [],
        "escalationActions": [],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [],
        "stateAssertions": {},
    }
    event = {
        "schemaVersion": 1,
        "kind": "out_of_turn",
        "actionSequence": 0,
        "actorSeat": "west",
        "expectedActorSeat": "north",
        "attemptedAction": {"card": "S7"},
        "reasonCode": "play_before_turn",
    }
    authority = _authority(root)
    authority.validate_authored_event(AuthoredAdjudicationEventV1.build(event))
    normalizer = OrToolsExactStateNormalizerV1(build_bundle(root).rules)
    baseline = normalizer.normalize(ExactTracePrefixV1.build(raw))
    audited = normalizer.normalize(
        ExactTracePrefixV1.build({**raw, "adjudicationEvents": [event]})
    )
    assert audited.state_hash == baseline.state_hash
    assert audited.exact_state_id == baseline.exact_state_id
    assert audited.legal_actions == baseline.legal_actions


def test_v3_1_service_catalog_resolves_only_bundle_bound_adjudication() -> None:
    resolved = service_api.CONTAINER.configuration.resolve(
        bundle_id="baloot_ortools_research_native_v3",
        environment="research-native-v3",
    )
    revision_id = str(resolved["bundle"]["revisionId"])
    _effective, context = (
        service_api.RESEARCH.execution_context_factory.resolve_bundle_revision(
            revision_id
        )
    )
    authority = service_api.RESEARCH._adjudication_research_authority_v1(context)
    assert authority is not None
    assert authority.sha256 == (
        "a347741d373c9de8761aaa08d2ba593b0e9301ce97bcb928ac2a60df7d4695a8"
    )
    capabilities = service_api.RESEARCH.capabilities(
        bundle_revision_id=revision_id
    )
    schema = capabilities["research"]["scenarioResearchV4"][
        "executionCapabilities"
    ]["authoringCatalog"]["exactTrace"]["adjudicationEvents"]
    assert schema["x-authoritySha256"] == authority.sha256


def test_selected_bundle_localizes_every_adjudication_catalog_key() -> None:
    resolved = service_api.CONTAINER.configuration.resolve(
        bundle_id="baloot_ortools_research_native_v3",
        environment="research-native-v3",
    )
    revision_id = str(resolved["bundle"]["revisionId"])
    _effective, context = (
        service_api.RESEARCH.execution_context_factory.resolve_bundle_revision(
            revision_id
        )
    )
    workflow = dict(
        context.bundle.component("product")["foundry_copy"]["groups"]["workflow"]
    )
    required: set[str] = set()

    def collect(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"x-messageArKey", "x-messageEnKey"}:
                    required.add(str(item))
                elif key in {"x-enumMessageArKeys", "x-enumMessageEnKeys"}:
                    required.update(str(message) for message in item.values())
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    authority = service_api.RESEARCH._adjudication_research_authority_v1(context)
    assert authority is not None
    collect(authority.authoring_catalog()["exactInputItemSchema"])
    for definition in adjudication_predicate_definitions_v3():
        required.update((definition.message_ar_key, definition.message_en_key))
        collect(dict(definition.argument_schema))
    assert sorted(required - set(workflow)) == []


def test_exact_preview_binds_adjudication_attempt_to_selected_bundle_and_boundary() -> None:
    resolved = service_api.CONTAINER.configuration.resolve(
        bundle_id="baloot_ortools_research_native_v3",
        environment="research-native-v3",
    )
    revision_id = str(resolved["bundle"]["revisionId"])
    event = {
        "schemaVersion": 1,
        "kind": "invalid_bid",
        "actionSequence": 0,
        "actorSeat": "west",
        "expectedActorSeat": "north",
        "attemptedAction": {"token": "sun"},
        "reasonCode": "purchase_before_turn",
    }

    def preview(authored_event: dict) -> dict:
        return service_api.RESEARCH.preview_exact_trace_legal_actions(
            {
                "schemaVersion": 1,
                "configuration": {
                    "schemaVersion": 1,
                    "bundleRevisionId": revision_id,
                },
                "exactTracePrefix": {
                    "schemaVersion": 1,
                    "deckOrder": list(range(32)),
                    "dealer": "east",
                    "adjudicationEvents": [authored_event],
                },
            }
        )

    initial = preview(event)
    assert initial["valid"] is True, initial["diagnostics"]
    assert preview({**event, "actionSequence": 999})["valid"] is False
    assert preview(
        {
            **event,
            "selectedRemedy": "opponent_register_with_projects",
            "rulingEvidenceRef": "RULING-INVALID-FOR-PURCHASE",
        }
    )["valid"] is False
    assert preview(
        {**event, "expectedActorSeat": "south"}
    )["valid"] is False


def test_rejected_ekka_and_out_of_turn_play_leave_legal_trace_unchanged(root: Path) -> None:
    engine, state = _active_round(root)
    invalid_ekka_card = next(
        card
        for card in engine.legal_actions(state)
        if not engine._can_declare_ekka(state, state.current_player, card)
    )
    before = engine.canonical_state(state)
    with pytest.raises(ValueError, match="Ekka may only"):
        engine.apply(state, invalid_ekka_card, ekka=True)
    assert engine.canonical_state(state) == before

    manager = GameManager(root)
    game = manager.create(seed=31_002, dealer=Seat.EAST)
    bought = False
    while game.bidding is not None and not game.bidding.finished:
        actor = game.bidding.actor
        legal = game.bidding.legal_actions()
        action = "hokum" if not bought and "hokum" in legal else "pass"
        bought = bought or action == "hokum"
        game = manager.bid(game.id, seat=actor, action=action)
    actor = game.state.current_player
    assert actor is not None
    wrong_actor = actor.next()
    card = next(iter_mask(game.state.hands[int(wrong_actor)]))
    state_before = game.engine.canonical_state(game.state)
    with pytest.raises(ValueError, match="turn"):
        manager.play(game.id, seat=wrong_actor, card=card)
    assert game.engine.canonical_state(game.state) == state_before


def test_invalid_bid_is_rejected_before_history_and_external_cases_remain_referee_evidence(
    root: Path,
    tmp_path: Path,
) -> None:
    manager = GameManager(root)
    game = manager.create(seed=31_003, dealer=Seat.EAST)
    assert game.bidding is not None and game.bidding.actor is not None
    history = tuple(game.bidding.records)
    state = game.engine.canonical_state(game.state)
    with pytest.raises(ValueError, match="illegal bid"):
        manager.bid(game.id, seat=game.bidding.actor, action="invented")
    assert tuple(game.bidding.records) == history
    assert game.engine.canonical_state(game.state) == state

    authority = _authority(root)
    service = AdjudicationService(
        SQLiteAdjudicationStore(tmp_path / "adjudication.sqlite3"),
        AdjudicationProfile.from_yaml(
            root / "configs/adjudication/saudi_researched_v1.yaml"
        ),
    )
    for kind in ("exposed_card", "signalling"):
        capability = authority.capability(kind)
        case = service.open_case(
            game_id=game.id,
            violation=capability.violation,
            reported_by="REF-31",
            description=capability.message_ar,
            metadata={
                "authorityProfileId": authority.profile_id,
                "authoritySha256": authority.sha256,
                "capabilitySha256": capability.sha256,
                "availableRemedies": list(capability.available_remedies),
                "requiresTableReferee": capability.requires_table_referee,
            },
        )
        assert case.violation == capability.violation
        assert case.metadata["capabilitySha256"] == capability.sha256
        assert case.metadata["requiresTableReferee"] is True


def test_authority_rejects_unknown_and_missing_cases(root: Path, tmp_path: Path) -> None:
    source = root / "configs/adjudication/research_events_v1.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["events"]["tournament_fee"] = document["events"]["signalling"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="must be complete"):
        AdjudicationResearchAuthorityV1.from_yaml(bad)

    del document["events"]["tournament_fee"]
    del document["events"]["incorrect_highest_card_naming"]
    bad.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="must be complete"):
        AdjudicationResearchAuthorityV1.from_yaml(bad)
