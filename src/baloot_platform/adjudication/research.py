from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .models import EvidenceItem, ViolationCode


_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVENT_KINDS = frozenset(
    {
        "incorrect_highest_card_naming",
        "out_of_turn",
        "invalid_bid",
        "exposed_card",
        "signalling",
        "verbal_baloot_inside_sequential_100",
    }
)
_VARIANT_EVENT_KINDS = frozenset({"false_ekka_attempt"})
_STAGES = frozenset({"auction", "play", "projects"})
_SEATS = frozenset({"north", "west", "south", "east"})
_AUTHORITY_CLASSES = frozenset(
    {
        "official_regulation_adjudication",
        "official_regulation_referee_adjudication",
        "configured_unratified_referee_variant",
    }
)
_EVENT_SCOPES = frozenset({"adjudicated_attempt", "external_evidence"})
_EXACT_INPUT = frozenset(
    {"legal_trace_rejects_attempt_audit_sidecar_supported", "audit_sidecar_only"}
)
_REMEDIES = frozenset(
    {
        "continue_play",
        "opponent_register_with_projects",
        "opponent_takweesh",
        "register_against_with_projects",
        "warning",
    }
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _closed_mapping(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result = copy.deepcopy(dict(value))
    missing = sorted(required - set(result))
    unknown = sorted(set(result) - required - optional)
    if missing or unknown:
        raise ValueError(f"{name} shape mismatch; missing={missing}, unknown={unknown}")
    return result


def _string(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


class AdjudicationResearchEventKindV1(StrEnum):
    INCORRECT_HIGHEST_CARD_NAMING = "incorrect_highest_card_naming"
    OUT_OF_TURN = "out_of_turn"
    INVALID_BID = "invalid_bid"
    EXPOSED_CARD = "exposed_card"
    SIGNALLING = "signalling"
    VERBAL_BALOOT_INSIDE_SEQUENTIAL_100 = "verbal_baloot_inside_sequential_100"
    FALSE_EKKA_ATTEMPT = "false_ekka_attempt"


@dataclass(frozen=True, slots=True)
class AuthoredAdjudicationEventV1:
    kind: AdjudicationResearchEventKindV1
    action_sequence: int
    actor_seat: str
    expected_actor_seat: str | None
    attempted_action: tuple[tuple[str, str | int | bool | None], ...]
    reason_code: str
    evidence_refs: tuple[str, ...]
    selected_remedy: str | None
    ruling_evidence_ref: str | None
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> AuthoredAdjudicationEventV1:
        raw = _closed_mapping(
            value,
            name="authored adjudication event",
            required=frozenset(
                {"schemaVersion", "kind", "actionSequence", "actorSeat", "reasonCode"}
            ),
            optional=frozenset(
                {
                    "expectedActorSeat",
                    "attemptedAction",
                    "evidenceRefs",
                    "selectedRemedy",
                    "rulingEvidenceRef",
                    "sha256",
                }
            ),
        )
        if type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1:
            raise ValueError("authored adjudication event schemaVersion must equal 1")
        kind = AdjudicationResearchEventKindV1(
            _string(raw["kind"], "authored adjudication event kind")
        )
        action_sequence = raw["actionSequence"]
        if type(action_sequence) is not int or action_sequence < 0:
            raise ValueError("authored adjudication event actionSequence must be non-negative")
        actor_seat = _string(raw["actorSeat"], "authored adjudication event actorSeat")
        if actor_seat not in _SEATS:
            raise ValueError("authored adjudication event actorSeat is invalid")
        expected_actor_seat = (
            _string(
                raw["expectedActorSeat"],
                "authored adjudication event expectedActorSeat",
            )
            if raw.get("expectedActorSeat") is not None
            else None
        )
        if expected_actor_seat is not None and expected_actor_seat not in _SEATS:
            raise ValueError(
                "authored adjudication event expectedActorSeat is invalid"
            )
        action = raw.get("attemptedAction", {})
        if not isinstance(action, Mapping):
            raise TypeError("authored adjudication event attemptedAction must be an object")
        action_row = dict(action)
        if any(type(key) is not str or not key for key in action_row):
            raise ValueError("authored adjudication attemptedAction keys must be named")
        if any(
            type(item) not in {str, int, bool, type(None)}
            for item in action_row.values()
        ):
            raise TypeError("authored adjudication attemptedAction values must be JSON scalars")
        references = raw.get("evidenceRefs", [])
        if type(references) is not list or any(type(item) is not str for item in references):
            raise TypeError("authored adjudication evidenceRefs must be an array of strings")
        evidence_refs = tuple(_string(item, "authored adjudication evidenceRef") for item in references)
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("authored adjudication evidenceRefs must be unique")
        selected_remedy = (
            _string(raw["selectedRemedy"], "authored adjudication selectedRemedy")
            if raw.get("selectedRemedy") is not None
            else None
        )
        ruling_ref = (
            _string(raw["rulingEvidenceRef"], "authored adjudication rulingEvidenceRef")
            if raw.get("rulingEvidenceRef") is not None
            else None
        )
        if (selected_remedy is None) != (ruling_ref is None):
            raise ValueError(
                "selectedRemedy and rulingEvidenceRef must be supplied together"
            )
        if selected_remedy is not None and selected_remedy not in _REMEDIES:
            raise ValueError("authored adjudication selectedRemedy is unsupported")
        core = {
            "schemaVersion": 1,
            "kind": kind.value,
            "actionSequence": action_sequence,
            "actorSeat": actor_seat,
            "expectedActorSeat": expected_actor_seat,
            "attemptedAction": {key: action_row[key] for key in sorted(action_row)},
            "reasonCode": _string(raw["reasonCode"], "authored adjudication reasonCode"),
            "evidenceRefs": list(evidence_refs),
            "selectedRemedy": selected_remedy,
            "rulingEvidenceRef": ruling_ref,
        }
        digest = _canonical_hash(core)
        declared = raw.get("sha256")
        if declared is not None and declared != digest:
            raise ValueError("authored adjudication event checksum mismatch")
        return cls(
            kind,
            action_sequence,
            actor_seat,
            expected_actor_seat,
            tuple(core["attemptedAction"].items()),
            core["reasonCode"],
            evidence_refs,
            selected_remedy,
            ruling_ref,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": self.kind.value,
            "actionSequence": self.action_sequence,
            "actorSeat": self.actor_seat,
            "expectedActorSeat": self.expected_actor_seat,
            "attemptedAction": dict(self.attempted_action),
            "reasonCode": self.reason_code,
            "evidenceRefs": list(self.evidence_refs),
            "selectedRemedy": self.selected_remedy,
            "rulingEvidenceRef": self.ruling_evidence_ref,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationResearchCapabilityV1:
    kind: AdjudicationResearchEventKindV1
    label_ar: str
    label_en: str
    violation: ViolationCode
    stage: str
    authority_class: str
    domain_primitive: str
    source_clause: str
    source_text_ar: str | None
    event_scope: str
    ordinary_trace: str
    covered_subtype: str | None
    available_remedies: tuple[str, ...]
    requires_table_referee: bool
    exact_input: str
    generated_search: str
    hard_predicate: str
    ui_exposure: str
    observer_safety: str
    message_ar: str
    message_en: str
    documentation: str
    sha256: str

    @classmethod
    def build(
        cls,
        kind: str,
        value: Mapping[str, Any],
    ) -> AdjudicationResearchCapabilityV1:
        raw = _closed_mapping(
            value,
            name=f"events.{kind}",
            required=frozenset(
                {
                    "violationCode",
                    "labelAr",
                    "labelEn",
                    "stage",
                    "authorityClass",
                    "domainPrimitive",
                    "sourceClause",
                    "eventScope",
                    "ordinaryTrace",
                    "availableRemedies",
                    "requiresTableReferee",
                    "generatedSearch",
                    "exactInput",
                    "hardPredicate",
                    "uiExposure",
                    "observerSafety",
                    "messageAr",
                    "messageEn",
                    "documentation",
                }
            ),
            optional=frozenset({"coveredSubtype", "sourceTextAr"}),
        )
        event_kind = AdjudicationResearchEventKindV1(kind)
        violation = ViolationCode(_string(raw["violationCode"], f"events.{kind}.violationCode"))
        stage = _string(raw["stage"], f"events.{kind}.stage")
        authority_class = _string(raw["authorityClass"], f"events.{kind}.authorityClass")
        event_scope = _string(raw["eventScope"], f"events.{kind}.eventScope")
        exact_input = _string(raw["exactInput"], f"events.{kind}.exactInput")
        ordinary_trace = _string(raw["ordinaryTrace"], f"events.{kind}.ordinaryTrace")
        if ordinary_trace not in {"rejected", "not_an_action"}:
            raise ValueError(f"events.{kind}.ordinaryTrace is unsupported")
        remedies = raw["availableRemedies"]
        if type(remedies) is not list or not remedies:
            raise ValueError(f"events.{kind}.availableRemedies must be a non-empty array")
        if any(type(item) is not str for item in remedies):
            raise TypeError(f"events.{kind}.availableRemedies must contain strings")
        available_remedies = tuple(remedies)
        if len(set(available_remedies)) != len(available_remedies):
            raise ValueError(f"events.{kind}.availableRemedies must be unique")
        if not set(available_remedies).issubset(_REMEDIES):
            raise ValueError(f"events.{kind}.availableRemedies is unsupported")
        if raw["requiresTableReferee"] is not True:
            raise ValueError(f"events.{kind}.requiresTableReferee must equal true")
        if stage not in _STAGES:
            raise ValueError(f"events.{kind}.stage is unsupported")
        if authority_class not in _AUTHORITY_CLASSES:
            raise ValueError(f"events.{kind}.authorityClass is unsupported")
        if event_scope not in _EVENT_SCOPES:
            raise ValueError(f"events.{kind}.eventScope is unsupported")
        if exact_input not in _EXACT_INPUT:
            raise ValueError(f"events.{kind}.exactInput is unsupported")
        for field, expected in (("observerSafety", "referee_restricted"),):
            if _string(raw[field], f"events.{kind}.{field}") != expected:
                raise ValueError(f"events.{kind}.{field} must equal {expected}")
        generated_search = _string(raw["generatedSearch"], f"events.{kind}.generatedSearch")
        hard_predicate = _string(raw["hardPredicate"], f"events.{kind}.hardPredicate")
        ui_exposure = _string(raw["uiExposure"], f"events.{kind}.uiExposure")
        native_variant = kind == "false_ekka_attempt"
        expected_support = "native_exact" if native_variant else "unsupported"
        expected_exposure = "variant_only" if native_variant else "hidden"
        if generated_search != expected_support or hard_predicate != expected_support:
            raise ValueError(
                f"events.{kind} search support must equal {expected_support}"
            )
        if ui_exposure != expected_exposure:
            raise ValueError(f"events.{kind}.uiExposure must equal {expected_exposure}")
        if (
            event_scope == "adjudicated_attempt"
            and exact_input != "legal_trace_rejects_attempt_audit_sidecar_supported"
        ):
            raise ValueError(f"events.{kind} attempts must stay outside exact input")
        if event_scope == "external_evidence" and exact_input != "audit_sidecar_only":
            raise ValueError(f"events.{kind} external evidence cannot enter exact input")
        covered_subtype = (
            _string(raw["coveredSubtype"], f"events.{kind}.coveredSubtype")
            if "coveredSubtype" in raw
            else None
        )
        if (kind in {"out_of_turn", "invalid_bid", "false_ekka_attempt"}) != (
            covered_subtype is not None
        ):
            raise ValueError(
                "only finite attempted-action events must declare an exact subtype"
            )

        core = {"kind": kind, **raw}
        return cls(
            event_kind,
            _string(raw["labelAr"], f"events.{kind}.labelAr"),
            _string(raw["labelEn"], f"events.{kind}.labelEn"),
            violation,
            stage,
            authority_class,
            _string(raw["domainPrimitive"], f"events.{kind}.domainPrimitive"),
            _string(raw["sourceClause"], f"events.{kind}.sourceClause"),
            (
                _string(raw["sourceTextAr"], f"events.{kind}.sourceTextAr")
                if "sourceTextAr" in raw
                else None
            ),
            event_scope,
            ordinary_trace,
            covered_subtype,
            available_remedies,
            True,
            exact_input,
            generated_search,
            hard_predicate,
            ui_exposure,
            "referee_restricted",
            _string(raw["messageAr"], f"events.{kind}.messageAr"),
            _string(raw["messageEn"], f"events.{kind}.messageEn"),
            _string(raw["documentation"], f"events.{kind}.documentation"),
            _canonical_hash(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": self.kind.value,
            "labelAr": self.label_ar,
            "labelEn": self.label_en,
            "violationCode": self.violation.value,
            "stage": self.stage,
            "authorityClass": self.authority_class,
            "domainPrimitive": self.domain_primitive,
            "sourceClause": self.source_clause,
            **(
                {"sourceTextAr": self.source_text_ar}
                if self.source_text_ar is not None
                else {}
            ),
            "eventScope": self.event_scope,
            "ordinaryTrace": self.ordinary_trace,
            **(
                {"coveredSubtype": self.covered_subtype}
                if self.covered_subtype is not None
                else {}
            ),
            "availableRemedies": list(self.available_remedies),
            "requiresTableReferee": self.requires_table_referee,
            "generatedSearch": self.generated_search,
            "exactInput": self.exact_input,
            "hardPredicate": self.hard_predicate,
            "uiExposure": self.ui_exposure,
            "observerSafety": self.observer_safety,
            "messageAr": self.message_ar,
            "messageEn": self.message_en,
            "documentation": self.documentation,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationResearchAuthorityV1:
    profile_id: str
    version: str
    status: str
    source_id: str
    source_url: str
    source_archive_url: str
    source_sha256: str
    source_bytes: int
    verification_receipt_path: str
    verification_receipt_sha256: str
    current_authority_boundary: str
    variant_id: str | None
    variant_status: str | None
    capabilities: Mapping[str, AdjudicationResearchCapabilityV1]
    sha256: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> AdjudicationResearchAuthorityV1:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(document)

    @classmethod
    def from_mapping(
        cls,
        document: Mapping[str, Any],
    ) -> AdjudicationResearchAuthorityV1:
        raw = _closed_mapping(
            document,
            name="adjudication research authority",
            required=frozenset({"schemaVersion", "profile", "sourceAuthority", "events"}),
            optional=frozenset({"variant"}),
        )
        if type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1:
            raise ValueError("adjudication research authority schemaVersion must equal 1")
        profile = _closed_mapping(
            raw["profile"],
            name="profile",
            required=frozenset({"id", "version", "status"}),
        )
        source = _closed_mapping(
            raw["sourceAuthority"],
            name="sourceAuthority",
            required=frozenset(
                {
                    "class",
                    "sourceId",
                    "url",
                    "archiveUrl",
                    "sha256",
                    "bytes",
                    "verificationReceiptPath",
                    "verificationReceiptSha256",
                    "currentAuthorityBoundary",
                }
            ),
        )
        if _string(source["class"], "sourceAuthority.class") != "official_regulation":
            raise ValueError("adjudication authority source class must be official_regulation")
        source_sha256 = _string(source["sha256"], "sourceAuthority.sha256")
        if _SHA256.fullmatch(source_sha256) is None:
            raise ValueError("sourceAuthority.sha256 must be a lowercase SHA-256")
        source_bytes = source["bytes"]
        if type(source_bytes) is not int or source_bytes <= 0:
            raise ValueError("sourceAuthority.bytes must be a positive integer")
        receipt_sha256 = _string(
            source["verificationReceiptSha256"],
            "sourceAuthority.verificationReceiptSha256",
        )
        if _SHA256.fullmatch(receipt_sha256) is None:
            raise ValueError(
                "sourceAuthority.verificationReceiptSha256 must be a lowercase SHA-256"
            )
        boundary = _string(
            source["currentAuthorityBoundary"],
            "sourceAuthority.currentAuthorityBoundary",
        )
        if boundary != "verified_2023_not_proven_current_2026_requires_external_ratification":
            raise ValueError("adjudication authority must preserve the current-version caveat")
        variant = None
        if "variant" in raw:
            variant = _closed_mapping(
                raw["variant"],
                name="variant",
                required=frozenset({"id", "status", "certificationEligible"}),
            )
            if _string(variant["status"], "variant.status") != "unratified_referee_variant":
                raise ValueError("adjudication variant must remain unratified")
            if variant["certificationEligible"] is not False:
                raise ValueError("adjudication variant cannot be certification eligible")
        if not isinstance(raw["events"], Mapping):
            raise TypeError("events must be an object")
        event_rows = dict(raw["events"])
        expected_events = _EVENT_KINDS | (_VARIANT_EVENT_KINDS if variant else frozenset())
        if set(event_rows) != expected_events:
            raise ValueError(
                "adjudication research events must be complete; "
                f"missing={sorted(expected_events - set(event_rows))}, "
                f"unknown={sorted(set(event_rows) - expected_events)}"
            )
        capabilities = {
            kind: AdjudicationResearchCapabilityV1.build(kind, event_rows[kind])
            for kind in sorted(event_rows)
        }
        core = {
            "schemaVersion": 1,
            "profile": profile,
            "sourceAuthority": source,
            **({"variant": variant} if variant is not None else {}),
            "events": {kind: capability.to_dict() for kind, capability in capabilities.items()},
        }
        return cls(
            _string(profile["id"], "profile.id"),
            _string(profile["version"], "profile.version"),
            _string(profile["status"], "profile.status"),
            _string(source["sourceId"], "sourceAuthority.sourceId"),
            _string(source["url"], "sourceAuthority.url"),
            _string(source["archiveUrl"], "sourceAuthority.archiveUrl"),
            source_sha256,
            source_bytes,
            _string(
                source["verificationReceiptPath"],
                "sourceAuthority.verificationReceiptPath",
            ),
            receipt_sha256,
            boundary,
            _string(variant["id"], "variant.id") if variant is not None else None,
            _string(variant["status"], "variant.status") if variant is not None else None,
            MappingProxyType(capabilities),
            _canonical_hash(core),
        )

    def capability(
        self,
        kind: AdjudicationResearchEventKindV1 | str,
    ) -> AdjudicationResearchCapabilityV1:
        return self.capabilities[AdjudicationResearchEventKindV1(kind).value]

    def validate_authored_event(
        self,
        event: AuthoredAdjudicationEventV1,
    ) -> AdjudicationResearchCapabilityV1:
        capability = self.capability(event.kind)
        if capability.event_scope == "adjudicated_attempt" and not event.attempted_action:
            raise ValueError(f"{event.kind.value} requires attemptedAction evidence")
        if capability.event_scope == "adjudicated_attempt":
            if event.expected_actor_seat is None:
                raise ValueError(
                    f"{event.kind.value} requires expectedActorSeat evidence"
                )
            false_ekka = event.kind == AdjudicationResearchEventKindV1.FALSE_EKKA_ATTEMPT
            if false_ekka and event.actor_seat != event.expected_actor_seat:
                raise ValueError(
                    "false_ekka_attempt actorSeat must equal expectedActorSeat"
                )
            if false_ekka:
                action = dict(event.attempted_action)
                if (
                    set(action) != {"cardId", "ekka"}
                    or type(action["cardId"]) is not int
                    or not 0 <= action["cardId"] < 32
                    or action["ekka"] is not True
                ):
                    raise ValueError(
                        "false_ekka_attempt requires cardId 0..31 and ekka=true"
                    )
            if not false_ekka and event.actor_seat == event.expected_actor_seat:
                raise ValueError(
                    f"{event.kind.value} actorSeat must differ from expectedActorSeat"
                )
        if capability.event_scope == "external_evidence" and not event.evidence_refs:
            raise ValueError(f"{event.kind.value} requires external evidenceRefs")
        if (
            capability.covered_subtype is not None
            and event.reason_code != capability.covered_subtype
        ):
            raise ValueError(
                f"{event.kind.value} only covers {capability.covered_subtype}"
            )
        if (
            event.selected_remedy is not None
            and event.selected_remedy not in capability.available_remedies
        ):
            raise ValueError(
                f"{event.selected_remedy} is not a remedy for {event.kind.value}"
            )
        return capability

    def authoring_catalog(self) -> dict[str, Any]:
        kind_ar_keys = {
            kind: f"research_adjudication_kind_{kind}"
            for kind in self.capabilities
        }
        kind_en_keys = {kind: f"{key}_en" for kind, key in kind_ar_keys.items()}
        remedy_ar_keys = {
            remedy: f"research_adjudication_remedy_{remedy}"
            for remedy in sorted(_REMEDIES)
        }
        remedy_en_keys = {
            remedy: f"{key}_en" for remedy, key in remedy_ar_keys.items()
        }
        seat_ar_keys = {
            seat: f"research_scenario_option_seat_{seat}" for seat in sorted(_SEATS)
        }
        seat_en_keys = {seat: f"{key}_en" for seat, key in seat_ar_keys.items()}
        exact_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schemaVersion",
                "kind",
                "actionSequence",
                "actorSeat",
                "reasonCode",
            ],
            "properties": {
                "schemaVersion": {
                    "type": "integer",
                    "const": 1,
                    "default": 1,
                    "x-messageArKey": "research_adjudication_schema_version",
                    "x-messageEnKey": "research_adjudication_schema_version_en",
                },
                "kind": {
                    "type": "string",
                    "enum": list(self.capabilities),
                    "default": "incorrect_highest_card_naming",
                    "x-messageArKey": "research_adjudication_kind",
                    "x-messageEnKey": "research_adjudication_kind_en",
                    "x-enumMessageArKeys": kind_ar_keys,
                    "x-enumMessageEnKeys": kind_en_keys,
                },
                "actionSequence": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "x-messageArKey": "research_adjudication_action_sequence",
                    "x-messageEnKey": "research_adjudication_action_sequence_en",
                },
                "actorSeat": {
                    "type": "string",
                    "enum": sorted(_SEATS),
                    "default": "south",
                    "x-messageArKey": "research_scenario_argument_actor",
                    "x-messageEnKey": "research_scenario_argument_actor_en",
                    "x-enumMessageArKeys": seat_ar_keys,
                    "x-enumMessageEnKeys": seat_en_keys,
                },
                "expectedActorSeat": {
                    "type": ["string", "null"],
                    "enum": [*sorted(_SEATS), None],
                    "default": None,
                    "x-messageArKey": "research_adjudication_expected_actor",
                    "x-messageEnKey": "research_adjudication_expected_actor_en",
                    "x-enumMessageArKeys": {
                        **seat_ar_keys,
                        "null": "research_adjudication_none",
                    },
                    "x-enumMessageEnKeys": {
                        **seat_en_keys,
                        "null": "research_adjudication_none_en",
                    },
                },
                "attemptedAction": {
                    "type": "object",
                    "default": {},
                    "additionalProperties": {
                        "type": ["string", "integer", "boolean", "null"]
                    },
                    "x-messageArKey": "research_adjudication_attempted_action",
                    "x-messageEnKey": "research_adjudication_attempted_action_en",
                },
                "reasonCode": {
                    "type": "string",
                    "minLength": 1,
                    "x-messageArKey": "research_adjudication_reason_code",
                    "x-messageEnKey": "research_adjudication_reason_code_en",
                },
                "evidenceRefs": {
                    "type": "array",
                    "default": [],
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                    "x-messageArKey": "research_adjudication_evidence_refs",
                    "x-messageEnKey": "research_adjudication_evidence_refs_en",
                },
                "selectedRemedy": {
                    "type": ["string", "null"],
                    "enum": [*sorted(_REMEDIES), None],
                    "default": None,
                    "x-messageArKey": "research_adjudication_selected_remedy",
                    "x-messageEnKey": "research_adjudication_selected_remedy_en",
                    "x-enumMessageArKeys": {
                        **remedy_ar_keys,
                        "null": "research_adjudication_none",
                    },
                    "x-enumMessageEnKeys": {
                        **remedy_en_keys,
                        "null": "research_adjudication_none_en",
                    },
                    "x-remediesByKind": {
                        kind: list(capability.available_remedies)
                        for kind, capability in self.capabilities.items()
                    },
                },
                "rulingEvidenceRef": {
                    "type": ["string", "null"],
                    "default": None,
                    "x-messageArKey": "research_adjudication_ruling_evidence_ref",
                    "x-messageEnKey": "research_adjudication_ruling_evidence_ref_en",
                },
            },
        }
        rows = []
        for capability in self.capabilities.values():
            finite_attempt = capability.event_scope == "adjudicated_attempt"
            rows.append(
                {
                    **capability.to_dict(),
                    "allowedHardness": (
                        ["hard", "report"]
                        if capability.hard_predicate == "native_exact"
                        else ["report"]
                    ),
                    "exactInputAuthoring": "report_evidence",
                    "generatedSearchHardSupport": capability.generated_search,
                    "searchBlockingGap": (
                        "none"
                        if capability.generated_search == "native_exact"
                        else "native_adjudication_branch_not_lowered"
                        if finite_attempt
                        else "external_evidence_not_derivable"
                    ),
                }
            )
        core = {
            "schemaVersion": 1,
            "authorityProfileId": self.profile_id,
            "authoritySha256": self.sha256,
            "fieldPath": "scenario.origin.exactTracePrefix.adjudicationEvents",
            "qualificationSource": "report_only",
            "exactInputItemSchema": exact_schema,
            "events": rows,
        }
        return {**core, "sha256": _canonical_hash(core)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profile": {
                "id": self.profile_id,
                "version": self.version,
                "status": self.status,
            },
            "sourceAuthority": {
                "class": "official_regulation",
                "sourceId": self.source_id,
                "url": self.source_url,
                "archiveUrl": self.source_archive_url,
                "sha256": self.source_sha256,
                "bytes": self.source_bytes,
                "verificationReceiptPath": self.verification_receipt_path,
                "verificationReceiptSha256": self.verification_receipt_sha256,
                "currentAuthorityBoundary": self.current_authority_boundary,
            },
            **(
                {
                    "variant": {
                        "id": self.variant_id,
                        "status": self.variant_status,
                        "certificationEligible": False,
                    }
                }
                if self.variant_id is not None
                else {}
            ),
            "events": {
                kind: capability.to_dict()
                for kind, capability in self.capabilities.items()
            },
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationResearchReportV1:
    """Authority-bound observer-safe summary of authored adjudication events."""

    authority_profile_id: str
    authority_sha256: str
    source_id: str
    source_sha256: str
    current_authority_boundary: str
    prefix_sha256: str
    state_sha256: str
    exact_state_id: str
    events: tuple[Mapping[str, Any], ...]
    sha256: str

    @classmethod
    def build(
        cls,
        authority: AdjudicationResearchAuthorityV1,
        *,
        prefix_sha256: str,
        events: tuple[AuthoredAdjudicationEventV1, ...],
        state_sha256: str,
        exact_state_id: str,
    ) -> AdjudicationResearchReportV1:
        if not isinstance(authority, AdjudicationResearchAuthorityV1):
            raise TypeError("adjudication report authority is invalid")
        for digest, name in (
            (prefix_sha256, "prefix_sha256"),
            (state_sha256, "state_sha256"),
        ):
            if type(digest) is not str or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        exact_id = _string(exact_state_id, "exact_state_id")
        if type(events) is not tuple or any(
            not isinstance(event, AuthoredAdjudicationEventV1) for event in events
        ):
            raise TypeError("adjudication report events must be authored event objects")
        if len({event.sha256 for event in events}) != len(events):
            raise ValueError("adjudication report events must be unique")

        event_rows: list[dict[str, Any]] = []
        for event in events:
            capability = authority.validate_authored_event(event)
            event_rows.append(
                {
                    "schemaVersion": 1,
                    "eventSha256": event.sha256,
                    "capabilitySha256": capability.sha256,
                    "kind": event.kind.value,
                    "stage": capability.stage,
                    "actionSequence": event.action_sequence,
                    "reasonCode": event.reason_code,
                    "sourceClause": capability.source_clause,
                    "observerSafety": capability.observer_safety,
                    "remedyStatus": (
                        "selected" if event.selected_remedy is not None else "unresolved"
                    ),
                    "selectedRemedy": event.selected_remedy,
                }
            )
        core = {
            "schemaVersion": 1,
            "kind": "adjudication_research_report_v1",
            "authorityProfileId": authority.profile_id,
            "authoritySha256": authority.sha256,
            "sourceId": authority.source_id,
            "sourceSha256": authority.source_sha256,
            "currentAuthorityBoundary": authority.current_authority_boundary,
            "prefixSha256": prefix_sha256,
            "stateSha256": state_sha256,
            "exactStateId": exact_id,
            "events": event_rows,
        }
        digest = _canonical_hash(core)
        return cls(
            authority.profile_id,
            authority.sha256,
            authority.source_id,
            authority.source_sha256,
            authority.current_authority_boundary,
            prefix_sha256,
            state_sha256,
            exact_id,
            tuple(MappingProxyType(copy.deepcopy(row)) for row in event_rows),
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "adjudication_research_report_v1",
            "authorityProfileId": self.authority_profile_id,
            "authoritySha256": self.authority_sha256,
            "sourceId": self.source_id,
            "sourceSha256": self.source_sha256,
            "currentAuthorityBoundary": self.current_authority_boundary,
            "prefixSha256": self.prefix_sha256,
            "stateSha256": self.state_sha256,
            "exactStateId": self.exact_state_id,
            "events": [copy.deepcopy(dict(event)) for event in self.events],
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationEventEvidenceV1:
    evidence_id: str
    kind: AdjudicationResearchEventKindV1
    violation: ViolationCode
    stage: str
    game_id: str
    round_number: int
    event_sequence: int
    actor_seat: str
    expected_actor_seat: str
    attempted_action: tuple[tuple[str, str | int | bool | None], ...]
    reason_code: str
    state_sha256: str
    authority_profile_id: str
    authority_sha256: str
    source_clause: str
    available_remedies: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        authority: AdjudicationResearchAuthorityV1,
        *,
        kind: AdjudicationResearchEventKindV1 | str,
        game_id: str,
        round_number: int,
        event_sequence: int,
        actor_seat: str,
        expected_actor_seat: str,
        attempted_action: Mapping[str, str | int | bool | None],
        reason_code: str,
        state_sha256: str,
    ) -> AdjudicationEventEvidenceV1:
        capability = authority.capability(kind)
        if capability.event_scope != "adjudicated_attempt":
            raise ValueError(f"{capability.kind.value} requires external referee evidence")
        if (
            capability.covered_subtype is not None
            and reason_code != capability.covered_subtype
        ):
            raise ValueError(
                f"{capability.kind.value} only covers {capability.covered_subtype}"
            )
        if type(round_number) is not int or round_number < 0:
            raise ValueError("round_number must be a non-negative integer")
        if type(event_sequence) is not int or event_sequence < 0:
            raise ValueError("event_sequence must be a non-negative integer")
        actor = _string(actor_seat, "actor_seat")
        expected_actor = _string(expected_actor_seat, "expected_actor_seat")
        if actor not in _SEATS or expected_actor not in _SEATS:
            raise ValueError("adjudication event seats are invalid")
        false_ekka = capability.kind == AdjudicationResearchEventKindV1.FALSE_EKKA_ATTEMPT
        if false_ekka and actor != expected_actor:
            raise ValueError(
                "false_ekka_attempt actor seat must equal expected actor"
            )
        if not false_ekka and actor == expected_actor:
            raise ValueError(
                f"{capability.kind.value} actor seat must differ from expected actor"
            )
        action = dict(attempted_action)
        if not action or any(type(key) is not str or not key for key in action):
            raise ValueError("attempted_action must contain named scalar fields")
        if any(type(value) not in {str, int, bool, type(None)} for value in action.values()):
            raise TypeError("attempted_action values must be JSON scalars")
        if false_ekka and (
            set(action) != {"cardId", "ekka"}
            or type(action["cardId"]) is not int
            or not 0 <= action["cardId"] < 32
            or action["ekka"] is not True
        ):
            raise ValueError(
                "false_ekka_attempt requires cardId 0..31 and ekka=true"
            )
        state_hash = _string(state_sha256, "state_sha256")
        if _SHA256.fullmatch(state_hash) is None:
            raise ValueError("state_sha256 must be a lowercase SHA-256")
        core = {
            "schemaVersion": 1,
            "kind": capability.kind.value,
            "violationCode": capability.violation.value,
            "stage": capability.stage,
            "gameId": _string(game_id, "game_id"),
            "roundNumber": round_number,
            "eventSequence": event_sequence,
            "actorSeat": actor,
            "expectedActorSeat": expected_actor,
            "attemptedAction": {key: action[key] for key in sorted(action)},
            "reasonCode": _string(reason_code, "reason_code"),
            "stateSha256": state_hash,
            "authorityProfileId": authority.profile_id,
            "authoritySha256": authority.sha256,
            "sourceClause": capability.source_clause,
            "availableRemedies": list(capability.available_remedies),
            "requiresTableReferee": capability.requires_table_referee,
            "observerSafety": capability.observer_safety,
        }
        digest = _canonical_hash(core)
        return cls(
            f"ADJEV-{digest[:20].upper()}",
            capability.kind,
            capability.violation,
            capability.stage,
            core["gameId"],
            round_number,
            event_sequence,
            core["actorSeat"],
            core["expectedActorSeat"],
            tuple(core["attemptedAction"].items()),
            core["reasonCode"],
            state_hash,
            authority.profile_id,
            authority.sha256,
            capability.source_clause,
            capability.available_remedies,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "evidenceId": self.evidence_id,
            "kind": self.kind.value,
            "violationCode": self.violation.value,
            "stage": self.stage,
            "gameId": self.game_id,
            "roundNumber": self.round_number,
            "eventSequence": self.event_sequence,
            "actorSeat": self.actor_seat,
            "expectedActorSeat": self.expected_actor_seat,
            "attemptedAction": dict(self.attempted_action),
            "reasonCode": self.reason_code,
            "stateSha256": self.state_sha256,
            "authorityProfileId": self.authority_profile_id,
            "authoritySha256": self.authority_sha256,
            "sourceClause": self.source_clause,
            "availableRemedies": list(self.available_remedies),
            "requiresTableReferee": True,
            "observerSafety": "referee_restricted",
            "sha256": self.sha256,
        }

    def to_evidence_item(
        self,
        *,
        submitted_by: str,
        description: str,
    ) -> EvidenceItem:
        return EvidenceItem.create(
            kind="adjudication_event_v1",
            submitted_by=submitted_by,
            description=description,
            sha256=self.sha256,
            metadata=self.to_dict(),
        )


__all__ = [
    "AuthoredAdjudicationEventV1",
    "AdjudicationResearchAuthorityV1",
    "AdjudicationResearchCapabilityV1",
    "AdjudicationResearchEventKindV1",
    "AdjudicationResearchReportV1",
    "AdjudicationEventEvidenceV1",
]
