from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.core.enums import Contract, Seat

from .contracts import (
    HISTORICAL_RECORD_REQUIRES_READJUDICATION,
    canonical_json,
    canonical_sha256,
)
from .reference_escalation import (
    ReferenceEscalationInterpreterV1,
    ReferenceEscalationStateV1,
)


DECISION_AUDIT_V2_MEDIA_TYPE = "application/vnd.balootx.decision-audit-v2+json"
_DECISION_AUDIT_V2_KEYS = frozenset(
    {
        "schemaVersion",
        "traceHash",
        "decisionCount",
        "choiceDecisionCount",
        "governedActionCount",
        "incompleteGovernedActionCount",
        "complete",
        "informationSetComplete",
        "sequentialResponseComplete",
        "configurationProvenance",
        "decisions",
        "sha256",
    }
)
_DECISION_AUDIT_REFERENCE_KEYS = frozenset(
    {"uri", "sha256", "bytes", "mediaType", "name", "ownerKind", "ownerHash"}
)
_DECISION_AUDIT_ROW_COMMON_KEYS = frozenset(
    {
        "schemaVersion",
        "decisionId",
        "decisionOrdinal",
        "actingSeat",
        "actingTeam",
        "decisionKind",
        "phase",
        "publicStateHash",
        "informationStateHash",
        "informationSetEvidence",
        "sequentialResponseEvidence",
        "legalActions",
        "rulePaths",
        "actualAction",
        "actionSource",
        "forced",
        "recommendedAction",
        "actualMatchesRecommendation",
        "alternatives",
        "expectedRegret",
        "probabilityDrop",
        "catastrophicCrossing",
        "hiddenDealSupport",
        "effectiveSampleSize",
        "confidence",
        "convergence",
        "principalContinuation",
        "supportedMetrics",
        "unsupportedMetrics",
        "selectionExplanation",
        "analyticalExplanation",
        "forcedExplanation",
        "tacticalEvidence",
        "actualDealOracle",
        "truthLabel",
        "provenance",
        "ordinal",
    }
)
_DECISION_AUDIT_CHOICE_ROW_KEYS = frozenset(
    {"primaryDecisionScoreEvidence", "forcedLegalityProof"}
)
_DECISION_AUDIT_GOVERNED_ROW_KEYS = frozenset(
    {"recommendationAvailable", "eventSequence", "eventHash"}
)
_DECISION_AUDIT_ROW_KEYS = (
    _DECISION_AUDIT_ROW_COMMON_KEYS
    | _DECISION_AUDIT_CHOICE_ROW_KEYS
    | _DECISION_AUDIT_GOVERNED_ROW_KEYS
)
_GOVERNED_EVENT_KINDS = {
    "project_declared": "project",
    "projects_revealed": "project",
    "baloot_declared": "project",
    "baloot_included_in_project": "project",
    "escalation_announced": "escalation",
}
_HASH_CHARACTERS = frozenset("0123456789abcdef")


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    text = value
    if len(text) != 64 or any(character not in _HASH_CHARACTERS for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return text


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _same_canonical_value(left: Any, right: Any) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def _validate_ledger_decision_row(
    row: Mapping[str, Any],
    record: Any,
) -> None:
    expected_scalars = {
        "decisionOrdinal": record.ordinal,
        "actingSeat": record.acting_seat.name.lower(),
        "actingTeam": record.acting_seat.team.name.lower(),
        "decisionKind": record.decision_kind,
        "phase": record.phase,
        "publicStateHash": record.public_prefix_hash,
        "forced": record.forced,
    }
    drifted = sorted(
        key for key, expected in expected_scalars.items() if row.get(key) != expected
    )
    if drifted:
        raise ValueError(
            f"DecisionAuditV2 row does not match its trace-ledger decision: {drifted}"
        )
    if not _same_canonical_value(
        row.get("legalActions"),
        [dict(item) for item in record.legal_actions],
    ):
        raise ValueError("DecisionAuditV2 legal actions do not match the trace ledger")
    if not _same_canonical_value(row.get("actualAction"), dict(record.chosen_action)):
        raise ValueError(
            "DecisionAuditV2 chosen action does not match the trace ledger"
        )


def _validate_decision_row_shape(
    row: Mapping[str, Any],
    *,
    choice: bool,
) -> None:
    unknown = sorted(set(row) - _DECISION_AUDIT_ROW_KEYS)
    required = _DECISION_AUDIT_ROW_COMMON_KEYS | (
        _DECISION_AUDIT_CHOICE_ROW_KEYS if choice else _DECISION_AUDIT_GOVERNED_ROW_KEYS
    )
    missing = sorted(required - set(row))
    forbidden = sorted(
        set(row)
        & (
            _DECISION_AUDIT_GOVERNED_ROW_KEYS
            if choice
            else _DECISION_AUDIT_CHOICE_ROW_KEYS
        )
    )
    if unknown or missing or forbidden:
        raise ValueError(
            "DecisionAuditV2 row schema is not closed: "
            f"missing={missing}, unknown={unknown}, forbidden={forbidden}"
        )
    if (
        isinstance(row.get("schemaVersion"), bool)
        or not isinstance(row.get("schemaVersion"), int)
        or row["schemaVersion"] != 1
    ):
        raise ValueError("DecisionAuditV2 row schema version is invalid")
    for key in (
        "decisionId",
        "actingSeat",
        "actingTeam",
        "decisionKind",
        "phase",
        "publicStateHash",
        "informationStateHash",
        "actionSource",
        "convergence",
        "truthLabel",
    ):
        if not isinstance(row.get(key), str) or not str(row[key]).strip():
            raise ValueError(f"DecisionAuditV2 row {key} is invalid")
    for key in (
        "informationSetEvidence",
        "sequentialResponseEvidence",
        "actualAction",
        "recommendedAction",
        "selectionExplanation",
        "analyticalExplanation",
        "provenance",
    ):
        if not isinstance(row.get(key), Mapping):
            raise ValueError(f"DecisionAuditV2 row {key} must be an object")
    for key in (
        "legalActions",
        "rulePaths",
        "alternatives",
        "principalContinuation",
        "supportedMetrics",
        "unsupportedMetrics",
        "tacticalEvidence",
    ):
        if not isinstance(row.get(key), list):
            raise ValueError(f"DecisionAuditV2 row {key} must be an array")
    for key in ("forced", "actualMatchesRecommendation", "catastrophicCrossing"):
        if not isinstance(row.get(key), bool):
            raise ValueError(f"DecisionAuditV2 row {key} must be boolean")
    if not choice:
        if not isinstance(row.get("recommendationAvailable"), bool):
            raise ValueError(
                "DecisionAuditV2 governed recommendation availability is invalid"
            )
        event_sequence = row.get("eventSequence")
        if (
            isinstance(event_sequence, bool)
            or not isinstance(event_sequence, int)
            or event_sequence < 0
        ):
            raise ValueError("DecisionAuditV2 governed event sequence is invalid")
        _hash(row.get("eventHash"), "DecisionAuditV2 governed event hash")


def _validate_trace_provenance(
    rows: Iterable[Mapping[str, Any]],
    *,
    configuration_provenance: Mapping[str, Any] | None,
    expected_engine_version: str | None,
    expected_bundle_hash: str | None,
    expected_rules_hash: str | None,
) -> None:
    expected = {
        key: value
        for key, value in {
            "engineVersion": expected_engine_version,
            "bundleHash": expected_bundle_hash,
            "rulesHash": expected_rules_hash,
        }.items()
        if value is not None
    }
    for row in rows:
        provenance = dict(row["provenance"])
        drifted = sorted(
            key for key, value in expected.items() if provenance.get(key) != value
        )
        if configuration_provenance is not None:
            drifted.extend(
                key
                for key, value in configuration_provenance.items()
                if not _same_canonical_value(provenance.get(key), value)
            )
        if drifted:
            raise ValueError(
                f"DecisionAuditV2 row trace provenance mismatch: {sorted(set(drifted))}"
            )


def decode_decision_audit_v2(
    payload: bytes,
    *,
    expected_trace_hash: str,
    ledger: DecisionLedgerV1,
    expected_engine_version: str | None = None,
    expected_bundle_hash: str | None = None,
    expected_rules_hash: str | None = None,
) -> dict[str, Any]:
    """Decode one canonical DecisionAuditV2 and bind it to one exact ledger.

    Historical audit schemas remain readable through their historical owners,
    but current Research rendering requires the closed V2 document because its
    rows participate in video selection and publication governance.
    """

    trace_hash = _hash(expected_trace_hash, "expected DecisionAuditV2 trace hash")
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("DecisionAuditV2 artifact is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("DecisionAuditV2 artifact must contain one JSON object")
    document = copy.deepcopy(dict(value))
    try:
        canonical_payload = _canonical_bytes(document)
    except (TypeError, ValueError) as exc:
        raise ValueError("DecisionAuditV2 artifact is not canonical JSON") from exc
    if canonical_payload != payload:
        raise ValueError("DecisionAuditV2 artifact is not canonical JSON")
    unknown = sorted(set(document) - _DECISION_AUDIT_V2_KEYS)
    missing = sorted(_DECISION_AUDIT_V2_KEYS - set(document))
    if unknown or missing:
        raise ValueError(
            "DecisionAuditV2 document schema is not closed: "
            f"missing={missing}, unknown={unknown}"
        )
    if (
        isinstance(document.get("schemaVersion"), bool)
        or not isinstance(document.get("schemaVersion"), int)
        or document["schemaVersion"] != 2
    ):
        raise ValueError(
            "historical DecisionAudit requires readjudication before rendering"
        )
    if _hash(document.get("traceHash"), "DecisionAuditV2 trace hash") != trace_hash:
        raise ValueError("DecisionAuditV2 belongs to another complete trace")
    declared_hash = _hash(document.get("sha256"), "DecisionAuditV2 document hash")
    core = copy.deepcopy(document)
    core.pop("sha256")
    if canonical_sha256(core) != declared_hash:
        raise ValueError("DecisionAuditV2 document checksum mismatch")

    decisions = document.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("DecisionAuditV2 decisions must be a non-empty array")
    if any(not isinstance(item, Mapping) for item in decisions):
        raise ValueError("DecisionAuditV2 decision rows must be objects")
    rows = [dict(item) for item in decisions]
    identifiers = [str(item.get("decisionId") or "") for item in rows]
    ordinals = [item.get("ordinal") for item in rows]
    if (
        any(not identifier for identifier in identifiers)
        or len(identifiers) != len(set(identifiers))
        or any(
            isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1
            for ordinal in ordinals
        )
        or ordinals != list(range(1, len(rows) + 1))
    ):
        raise ValueError("DecisionAuditV2 row identities and ordinals must be exact")

    ledger_by_id = {record.decision_id: record for record in ledger.records}
    choice_rows = [row for row in rows if str(row.get("decisionId")) in ledger_by_id]
    if {str(row["decisionId"]) for row in choice_rows} != set(ledger_by_id):
        raise ValueError(
            "DecisionAuditV2 does not cover the exact trace decision ledger"
        )
    for row in choice_rows:
        _validate_decision_row_shape(row, choice=True)
        _validate_ledger_decision_row(row, ledger_by_id[str(row["decisionId"])])
    governed_rows = [
        row for row in rows if str(row.get("decisionId")) not in ledger_by_id
    ]
    for row in governed_rows:
        _validate_decision_row_shape(row, choice=False)
    if any(
        row.get("decisionKind") not in {"project", "escalation"}
        or row.get("decisionOrdinal") is not None
        for row in governed_rows
    ):
        raise ValueError("DecisionAuditV2 contains a foreign decision row")

    exact_counts = {
        "decisionCount": len(rows),
        "choiceDecisionCount": len(ledger.records),
        "governedActionCount": len(governed_rows),
    }
    if any(
        isinstance(document.get(key), bool)
        or not isinstance(document.get(key), int)
        or document.get(key) != expected
        for key, expected in exact_counts.items()
    ):
        raise ValueError("DecisionAuditV2 counts do not match its trace ledger")
    for key in (
        "incompleteGovernedActionCount",
        "complete",
        "informationSetComplete",
        "sequentialResponseComplete",
    ):
        if key == "incompleteGovernedActionCount":
            if isinstance(document[key], bool) or not isinstance(document[key], int):
                raise ValueError("DecisionAuditV2 incomplete action count is invalid")
        elif not isinstance(document[key], bool):
            raise ValueError(f"DecisionAuditV2 {key} flag is invalid")
    incomplete = int(document["incompleteGovernedActionCount"])
    if incomplete < 0 or incomplete > len(governed_rows):
        raise ValueError("DecisionAuditV2 incomplete action count is invalid")
    if bool(document["complete"]) != (incomplete == 0):
        raise ValueError("DecisionAuditV2 completion flag is inconsistent")
    configuration_provenance = document.get("configurationProvenance")
    if configuration_provenance is not None and not isinstance(
        configuration_provenance, Mapping
    ):
        raise ValueError("DecisionAuditV2 configuration provenance is invalid")
    _validate_trace_provenance(
        rows,
        configuration_provenance=(
            dict(configuration_provenance)
            if isinstance(configuration_provenance, Mapping)
            else None
        ),
        expected_engine_version=expected_engine_version,
        expected_bundle_hash=expected_bundle_hash,
        expected_rules_hash=expected_rules_hash,
    )
    information_set_complete = all(
        dict(row["informationSetEvidence"]).get("status") != "incomplete"
        and dict(row["informationSetEvidence"]).get("observerSafe") is True
        and dict(row["informationSetEvidence"]).get("actionCoverageComplete") is True
        and dict(row["informationSetEvidence"]).get("actorChoiceBoundOnce") is True
        for row in choice_rows
    )
    sequential_response_complete = all(
        dict(row["sequentialResponseEvidence"]).get("status") != "incomplete"
        and dict(row["sequentialResponseEvidence"]).get("allAlternativesEvaluated")
        is True
        and dict(row["sequentialResponseEvidence"]).get("pairedActorSafeCohort") is True
        for row in choice_rows
    )
    if document["informationSetComplete"] is not information_set_complete:
        raise ValueError("DecisionAuditV2 information-set completion is inconsistent")
    if document["sequentialResponseComplete"] is not sequential_response_complete:
        raise ValueError(
            "DecisionAuditV2 sequential-response completion is inconsistent"
        )
    return document


def load_decision_audit_v2(
    artifact_store: Any,
    reference: Mapping[str, Any],
    *,
    expected_trace_hash: str,
    ledger: DecisionLedgerV1,
    expected_engine_version: str | None = None,
    expected_bundle_hash: str | None = None,
    expected_rules_hash: str | None = None,
) -> dict[str, Any]:
    """Load and verify one content-addressed DecisionAuditV2 artifact."""

    raw_reference = dict(reference)
    unknown = sorted(set(raw_reference) - _DECISION_AUDIT_REFERENCE_KEYS)
    required = {"uri", "sha256", "bytes", "mediaType"}
    missing = sorted(required - set(raw_reference))
    if unknown or missing:
        raise ValueError(
            "DecisionAuditV2 artifact reference is not closed: "
            f"missing={missing}, unknown={unknown}"
        )
    if raw_reference.get("mediaType") != DECISION_AUDIT_V2_MEDIA_TYPE:
        raise ValueError("DecisionAuditV2 artifact media type is invalid")
    expected_bytes = raw_reference.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
    ):
        raise ValueError("DecisionAuditV2 artifact byte count is invalid")
    artifact_hash = _hash(raw_reference.get("sha256"), "DecisionAuditV2 artifact hash")
    owner_kind = raw_reference.get("ownerKind")
    owner_hash = raw_reference.get("ownerHash")
    if (owner_kind is None) != (owner_hash is None):
        raise ValueError("DecisionAuditV2 semantic owner fields must appear together")
    if owner_kind is not None and owner_kind != "decision_audit_v2":
        raise ValueError("DecisionAuditV2 semantic owner kind is invalid")
    payload = artifact_store.get(str(raw_reference.get("uri") or ""))
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != artifact_hash
    ):
        raise ValueError("DecisionAuditV2 artifact integrity check failed")
    document = decode_decision_audit_v2(
        payload,
        expected_trace_hash=expected_trace_hash,
        ledger=ledger,
        expected_engine_version=expected_engine_version,
        expected_bundle_hash=expected_bundle_hash,
        expected_rules_hash=expected_rules_hash,
    )
    if (
        owner_hash is not None
        and _hash(owner_hash, "DecisionAuditV2 semantic owner hash")
        != document["sha256"]
    ):
        raise ValueError("DecisionAuditV2 semantic owner identity mismatch")
    return document


def validate_selected_decision_audit_v2(
    row: Mapping[str, Any],
    *,
    analysis: Any,
) -> None:
    """Bind one already ledger-validated audit row to its parsed analysis row."""

    decision_id = str(row.get("decisionId") or "")
    matches = [
        item
        for item in getattr(analysis, "decisions", ())
        if str(getattr(item, "decision_id", "")) == decision_id
    ]
    if len(matches) != 1:
        raise ValueError("selected DecisionAuditV2 analysis identity is not unique")
    decision_analysis = matches[0]
    expected_values = {
        "decisionId": decision_analysis.decision_id,
        "decisionOrdinal": decision_analysis.ordinal,
        "actingSeat": decision_analysis.acting_seat,
        "decisionKind": decision_analysis.decision_kind,
        "forced": decision_analysis.forced,
        "actualAction": dict(decision_analysis.actual_action),
        "recommendedAction": dict(decision_analysis.best_action),
        "alternatives": [dict(item) for item in decision_analysis.alternatives],
        "expectedRegret": decision_analysis.expected_regret,
        "probabilityDrop": decision_analysis.probability_drop,
        "catastrophicCrossing": decision_analysis.catastrophic_crossing,
        "hiddenDealSupport": decision_analysis.hidden_deals,
        "effectiveSampleSize": decision_analysis.effective_sample_size,
        "confidence": decision_analysis.confidence,
        "convergence": decision_analysis.method,
        "supportedMetrics": list(decision_analysis.supported_metrics),
        "unsupportedMetrics": list(decision_analysis.unsupported_metrics),
        "primaryDecisionScoreEvidence": (
            decision_analysis.primary_decision_score_evidence.to_dict()
            if decision_analysis.primary_decision_score_evidence is not None
            else None
        ),
        "forcedLegalityProof": (
            dict(decision_analysis.forced_legality_proof)
            if decision_analysis.forced_legality_proof is not None
            else None
        ),
    }
    drifted = sorted(
        key
        for key, expected in expected_values.items()
        if not _same_canonical_value(row.get(key), expected)
    )
    if drifted:
        raise ValueError(
            "selected DecisionAuditV2 row does not match its parsed analysis: "
            f"{drifted}"
        )
    if row.get("actualMatchesRecommendation") is not _same_canonical_value(
        decision_analysis.actual_action,
        decision_analysis.best_action,
    ):
        raise ValueError(
            "selected DecisionAuditV2 recommendation result does not match analysis"
        )

    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selected DecisionAuditV2 analysis provenance is missing")
    expected_provenance = {
        "analyzerId": analysis.analyzer_id,
        "analyzerVersion": analysis.analyzer_version,
        "analysisProfileHash": analysis.profile_hash,
        "analysisHash": decision_analysis.sha256,
    }
    drifted_provenance = sorted(
        key
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    )
    if drifted_provenance:
        raise ValueError(
            "selected DecisionAuditV2 analysis provenance mismatch: "
            f"{drifted_provenance}"
        )


def validate_governed_decision_audit_v2(
    document: Mapping[str, Any],
    *,
    trace: Any,
    rules: Any,
) -> None:
    """Bind every non-choice audit row to one exact canonical trace event."""

    decisions = document.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("DecisionAuditV2 decisions must be an array")
    ledger_ids = {
        str(record.decision_id)
        for record in getattr(getattr(trace, "decision_ledger", None), "records", ())
    }
    governed_rows = [
        dict(item)
        for item in decisions
        if isinstance(item, Mapping)
        and str(item.get("decisionId") or "") not in ledger_ids
    ]
    governed_events = [
        event
        for event in getattr(trace, "events", ())
        if str(getattr(event, "event_type", "")) in _GOVERNED_EVENT_KINDS
    ]
    complete_events: list[tuple[Any, Seat]] = []
    incomplete_count = 0
    for event in governed_events:
        event_type = str(event.event_type)
        payload = dict(event.payload)
        actor_key = "seat" if event_type in {
            "baloot_declared",
            "escalation_announced",
        } else "owner"
        raw_actor = payload.get(actor_key)
        if raw_actor is None:
            incomplete_count += 1
            continue
        try:
            actor = Seat.parse(str(raw_actor))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "DecisionAuditV2 governed trace event actor is invalid"
            ) from exc
        complete_events.append((event, actor))

    if (
        document.get("governedActionCount") != len(complete_events)
        or document.get("incompleteGovernedActionCount") != incomplete_count
        or len(governed_rows) != len(complete_events)
    ):
        raise ValueError(
            "DecisionAuditV2 governed rows do not cover the exact trace events"
        )

    rows_by_event = {
        (row.get("eventSequence"), row.get("eventHash")): row
        for row in governed_rows
    }
    if len(rows_by_event) != len(governed_rows):
        raise ValueError("DecisionAuditV2 governed trace-event bindings are not unique")

    expected_bindings = {
        (int(event.sequence), str(event.event_hash))
        for event, _actor in complete_events
    }
    if set(rows_by_event) != expected_bindings:
        raise ValueError(
            "DecisionAuditV2 governed rows do not match the exact trace events"
        )

    trace_hash = _hash(
        getattr(trace, "sha256", None),
        "DecisionAuditV2 governed trace hash",
    )
    event_sequence_by_hash = {
        str(event.event_hash): int(event.sequence)
        for event in getattr(trace, "events", ())
    }
    play_snapshots = [
        (
            event_sequence_by_hash.get(
                str(snapshot.decision.resulting_event_hash),
                10**9,
            ),
            snapshot,
        )
        for snapshot in TraceDecisionProjectionStoreV1(
            trace,
            rules,
        ).replay.decisions
        if snapshot.decision.decision_kind == "play"
    ]
    prior_escalations: list[tuple[Seat, str, str | None]] = []
    for event, actor in complete_events:
        event_type = str(event.event_type)
        payload = copy.deepcopy(dict(event.payload))
        decision_kind = _GOVERNED_EVENT_KINDS[event_type]
        row = rows_by_event[(int(event.sequence), str(event.event_hash))]
        expected_action = (
            {
                "action": str(payload.get("action") or ""),
                "mode": str(payload.get("mode") or ""),
            }
            if decision_kind == "escalation"
            else {"eventType": event_type, **payload}
        )
        if decision_kind == "escalation" and (
            not expected_action["action"] or not expected_action["mode"]
        ):
            raise ValueError(
                "DecisionAuditV2 governed escalation trace event is incomplete"
            )
        if event_type in {
            "baloot_declared",
            "baloot_included_in_project",
            "escalation_announced",
        } and str(payload.get("team") or "") != actor.team.name.lower():
            raise ValueError(
                "DecisionAuditV2 governed trace event team is invalid"
            )
        escalation_legal_actions: list[dict[str, Any]] | None = None
        if decision_kind == "escalation":
            trigger = next(
                (
                    snapshot
                    for sequence, snapshot in play_snapshots
                    if sequence > int(event.sequence)
                    and snapshot.decision.acting_seat == actor
                ),
                None,
            )
            if (
                trigger is None
                or trigger.state.declarer is None
                or trigger.state.contract is None
            ):
                raise ValueError(
                    "DecisionAuditV2 governed escalation lacks its replay state"
                )
            interpreter = ReferenceEscalationInterpreterV1.from_rules(rules.data)
            prior_state = ReferenceEscalationStateV1(
                interpreter.domain.default_level
            )
            for prior_actor, prior_action, prior_mode in prior_escalations:
                prior_state, _ = interpreter.apply(
                    prior_state,
                    declarer=trigger.state.declarer,
                    contract=trigger.state.contract,
                    match_score=tuple(trigger.state.match_score),
                    actor=prior_actor,
                    action=prior_action,
                    mode=prior_mode,
                )
            escalation_legal_actions = [{"action": "decline", "mode": None}]
            for candidate_action in ("double", "triple", "four", "gahwa"):
                candidate_modes: tuple[str | None, ...] = (
                    ("open", "locked")
                    if trigger.state.contract == Contract.HOKUM
                    and candidate_action in {"double", "four"}
                    else (None,)
                )
                for candidate_mode in candidate_modes:
                    try:
                        _next_state, resolved = interpreter.apply(
                            prior_state,
                            declarer=trigger.state.declarer,
                            contract=trigger.state.contract,
                            match_score=tuple(trigger.state.match_score),
                            actor=actor,
                            action=candidate_action,
                            mode=candidate_mode,
                        )
                    except ValueError:
                        continue
                    candidate = {
                        "action": resolved.action,
                        "mode": resolved.resolved_mode,
                    }
                    if candidate not in escalation_legal_actions:
                        escalation_legal_actions.append(candidate)
            prior_escalations.append(
                (
                    actor,
                    expected_action["action"],
                    (
                        expected_action["mode"]
                        if expected_action["action"] in {"double", "four"}
                        else None
                    ),
                )
            )
        expected_id = (
            "ACT-"
            + canonical_sha256(
                {
                    "traceHash": trace_hash,
                    "eventHash": str(event.event_hash),
                    "decisionKind": decision_kind,
                }
            )[:20].upper()
        )
        expected_values = {
            "decisionId": expected_id,
            "decisionOrdinal": None,
            "actingSeat": actor.name.lower(),
            "actingTeam": actor.team.name.lower(),
            "decisionKind": decision_kind,
            "phase": "final_dealt" if decision_kind == "escalation" else "playing",
            "publicStateHash": (
                str(event.previous_event_hash)
                if event.previous_event_hash is not None
                else canonical_sha256([])
            ),
            "actualAction": expected_action,
            "forced": decision_kind == "project",
            "recommendationAvailable": decision_kind == "project",
        }
        if decision_kind == "project":
            expected_values.update(
                {
                    "legalActions": [expected_action],
                    "recommendedAction": expected_action,
                    "actualMatchesRecommendation": True,
                }
            )
        else:
            expected_values.update(
                {
                    "legalActions": escalation_legal_actions,
                    "recommendedAction": {"status": "unavailable"},
                    "actualMatchesRecommendation": False,
                }
            )
        drifted = sorted(
            key
            for key, expected in expected_values.items()
            if not _same_canonical_value(row.get(key), expected)
        )
        if drifted:
            raise ValueError(
                "DecisionAuditV2 governed row does not match its trace event: "
                f"{drifted}"
            )
        selection = row.get("selectionExplanation")
        selection_evidence = (
            dict(selection.get("evidence") or {})
            if isinstance(selection, Mapping)
            else {}
        )
        if (
            selection_evidence.get("eventSequence") != int(event.sequence)
            or selection_evidence.get("eventHash") != str(event.event_hash)
        ):
            raise ValueError(
                "DecisionAuditV2 governed selection evidence does not match its trace event"
            )
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("transitionAuthority") != "baloot_engine"
        ):
            raise ValueError(
                "DecisionAuditV2 governed transition authority is invalid"
            )


def _action_hashes(actions: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(canonical_sha256(dict(action)) for action in actions))


DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE = "decision_consistency_evidence_unavailable"
DECISION_MATERIAL_COUNTEREXAMPLE = "decision_material_counterexample"
DECISION_RECOMMENDATION_MISMATCH = "decision_recommendation_mismatch"


def decision_consistency_qualification_failures(
    audit: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return fail-closed qualification reasons from DecisionAuditV2 evidence.

    Numerical regret is only one projection of the configured decision
    objective.  The explicit recommendation comparison and sequential-response
    counterexample therefore remain independent hard qualification gates.
    """

    schema_version = audit.get("schemaVersion")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 2
    ):
        return (HISTORICAL_RECORD_REQUIRES_READJUDICATION,)
    if schema_version != 2:
        return (DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE,)

    decisions = audit.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return (DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE,)

    evidence_unavailable = False
    material_counterexample = False
    recommendation_mismatch = False
    for value in decisions:
        if not isinstance(value, Mapping):
            evidence_unavailable = True
            continue
        forced = value.get("forced")
        if not isinstance(forced, bool):
            evidence_unavailable = True
            continue
        recommendation_available = value.get("recommendationAvailable", True)
        if not isinstance(recommendation_available, bool):
            evidence_unavailable = True
            continue
        if not recommendation_available:
            evidence_unavailable = True
            continue
        actual_action = value.get("actualAction")
        recommended_action = value.get("recommendedAction")
        matches = value.get("actualMatchesRecommendation")
        response = value.get("sequentialResponseEvidence")
        if (
            not isinstance(actual_action, Mapping)
            or not isinstance(recommended_action, Mapping)
            or not isinstance(matches, bool)
            or not isinstance(response, Mapping)
        ):
            evidence_unavailable = True
            continue
        counterexample = response.get("materialCounterexampleFound")
        actual_hash = canonical_sha256(dict(actual_action))
        recommended_hash = canonical_sha256(dict(recommended_action))
        if (
            not isinstance(counterexample, bool)
            or response.get("chosenActionHash") != actual_hash
            or response.get("recommendedActionHash") != recommended_hash
            or matches != (actual_hash == recommended_hash)
        ):
            evidence_unavailable = True
            continue
        material_counterexample = material_counterexample or counterexample
        recommendation_mismatch = recommendation_mismatch or not matches

    failures = []
    if evidence_unavailable:
        failures.append(DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE)
    if material_counterexample:
        failures.append(DECISION_MATERIAL_COUNTEREXAMPLE)
    if recommendation_mismatch:
        failures.append(DECISION_RECOMMENDATION_MISMATCH)
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class InformationSetEvidenceV1:
    observer: Seat
    information_state_hash: str
    legal_action_set_hash: str
    analysis_evidence_hash: str
    paired_evidence_hash: str
    hidden_deal_support: int
    effective_sample_size: float
    action_coverage_complete: bool
    observer_safe: bool
    actor_choice_bound_once: bool
    status: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        observer: Seat | str | int,
        information_state: Mapping[str, Any],
        legal_actions: Iterable[Mapping[str, Any]],
        alternatives: Iterable[Mapping[str, Any]],
        forced: bool,
        analysis_hash: str,
        method: str,
        hidden_deal_support: int,
        effective_sample_size: float,
    ) -> "InformationSetEvidenceV1":
        seat = Seat.parse(observer)
        info = copy.deepcopy(dict(information_state))
        legal = tuple(copy.deepcopy(dict(item)) for item in legal_actions)
        ranked = tuple(copy.deepcopy(dict(item)) for item in alternatives)
        evaluated = tuple(dict(item.get("action") or {}) for item in ranked)
        legal_hashes = _action_hashes(legal)
        evaluated_hashes = _action_hashes(evaluated)
        coverage = legal_hashes == evaluated_hashes
        observer_matches = str(info.get("observer") or "") == seat.name.lower()
        normalized_method = str(method)
        observer_safe = observer_matches and not any(
            marker in normalized_method.lower()
            for marker in ("actual_deal", "oracle", "omniscient")
        )
        paired = bool(forced) or (
            int(hidden_deal_support) > 0 and coverage and observer_safe
        )
        exact = bool(forced) or (
            normalized_method.startswith("exhaustive_")
            and ranked
            and all(
                str(item.get("searchCompletionStatus") or "") == "exact"
                and float(item.get("exactShare", 0.0)) >= 1.0 - 1e-12
                for item in ranked
            )
        )
        status = (
            "exact_information_set"
            if paired and exact
            else "paired_sampled_information_set"
            if paired
            else "incomplete"
        )
        information_state_hash = canonical_sha256(info)
        legal_action_set_hash = canonical_sha256(list(legal_hashes))
        evidence_core = {
            "analysisHash": str(analysis_hash),
            "method": normalized_method,
            "hiddenDealSupport": int(hidden_deal_support),
            "effectiveSampleSize": float(effective_sample_size),
            "legalActionSetHash": legal_action_set_hash,
            "evaluatedActionHashes": list(evaluated_hashes),
        }
        paired_evidence_hash = canonical_sha256(evidence_core)
        core = {
            "schemaVersion": 1,
            "truthMode": "actor_observer",
            "observer": seat.name.lower(),
            "informationStateHash": information_state_hash,
            "legalActionSetHash": legal_action_set_hash,
            "analysisEvidenceHash": str(analysis_hash),
            "pairedEvidenceHash": paired_evidence_hash,
            "hiddenDealSupport": int(hidden_deal_support),
            "effectiveSampleSize": float(effective_sample_size),
            "actionCoverageComplete": coverage,
            "observerSafe": observer_safe,
            "actorChoiceBoundOnce": paired,
            "hiddenStateIdentifiersExposed": False,
            "status": status,
        }
        return cls(
            seat,
            information_state_hash,
            legal_action_set_hash,
            str(analysis_hash),
            paired_evidence_hash,
            int(hidden_deal_support),
            float(effective_sample_size),
            coverage,
            observer_safe,
            paired,
            status,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "truthMode": "actor_observer",
            "observer": self.observer.name.lower(),
            "informationStateHash": self.information_state_hash,
            "legalActionSetHash": self.legal_action_set_hash,
            "analysisEvidenceHash": self.analysis_evidence_hash,
            "pairedEvidenceHash": self.paired_evidence_hash,
            "hiddenDealSupport": self.hidden_deal_support,
            "effectiveSampleSize": self.effective_sample_size,
            "actionCoverageComplete": self.action_coverage_complete,
            "observerSafe": self.observer_safe,
            "actorChoiceBoundOnce": self.actor_choice_bound_once,
            "hiddenStateIdentifiersExposed": False,
            "status": self.status,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SequentialResponseEvidenceV1:
    chosen_action_hash: str
    recommended_action_hash: str
    legal_action_count: int
    evaluated_action_count: int
    response_model: str
    continuation_statuses: tuple[str, ...]
    minimum_exact_share: float
    expected_regret: float
    all_alternatives_evaluated: bool
    paired_actor_safe_cohort: bool
    exact_continuation_coverage: bool
    material_counterexample_found: bool
    status: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        actual_action: Mapping[str, Any],
        recommended_action: Mapping[str, Any],
        legal_actions: Iterable[Mapping[str, Any]],
        alternatives: Iterable[Mapping[str, Any]],
        information_set: InformationSetEvidenceV1,
        expected_regret: float,
        forced: bool,
    ) -> "SequentialResponseEvidenceV1":
        legal = tuple(copy.deepcopy(dict(item)) for item in legal_actions)
        ranked = tuple(copy.deepcopy(dict(item)) for item in alternatives)
        evaluated = tuple(dict(item.get("action") or {}) for item in ranked)
        all_evaluated = _action_hashes(legal) == _action_hashes(evaluated)
        continuation_statuses = tuple(
            sorted(
                {
                    str(item.get("searchCompletionStatus") or "not_applicable")
                    for item in ranked
                }
            )
        )
        exact_shares = tuple(float(item.get("exactShare", 0.0)) for item in ranked)
        minimum_exact_share = min(exact_shares, default=1.0 if forced else 0.0)
        exact_continuations = bool(forced) or bool(
            ranked
            and all(status == "exact" for status in continuation_statuses)
            and minimum_exact_share >= 1.0 - 1e-12
        )
        paired = bool(
            information_set.observer_safe
            and information_set.action_coverage_complete
            and information_set.actor_choice_bound_once
        )
        regret = max(0.0, float(expected_regret))
        same_action = canonical_sha256(dict(actual_action)) == canonical_sha256(
            dict(recommended_action)
        )
        counterexample = (
            not forced
            and paired
            and all_evaluated
            and (regret > 1e-12 or not same_action)
        )
        exact = bool(
            forced
            or (
                information_set.status == "exact_information_set"
                and all_evaluated
                and exact_continuations
            )
        )
        robust = bool(not exact and paired and all_evaluated and ranked)
        status = (
            "exact_sequential_response"
            if exact
            else "robust_policy_conditioned_response"
            if robust
            else "incomplete"
        )
        response_model = (
            "authoritative_forced_rule"
            if forced
            else "adversarial_exact_continuation"
            if exact_continuations
            else "policy_conditioned_continuation"
        )
        core = {
            "schemaVersion": 1,
            "chosenActionHash": canonical_sha256(dict(actual_action)),
            "recommendedActionHash": canonical_sha256(dict(recommended_action)),
            "legalActionCount": len(legal),
            "evaluatedActionCount": len(evaluated),
            "responseModel": response_model,
            "continuationStatuses": list(continuation_statuses),
            "minimumExactShare": minimum_exact_share,
            "expectedRegret": regret,
            "allAlternativesEvaluated": all_evaluated,
            "pairedActorSafeCohort": paired,
            "exactContinuationCoverage": exact_continuations,
            "materialCounterexampleFound": counterexample,
            "status": status,
        }
        return cls(
            core["chosenActionHash"],
            core["recommendedActionHash"],
            len(legal),
            len(evaluated),
            response_model,
            continuation_statuses,
            minimum_exact_share,
            regret,
            all_evaluated,
            paired,
            exact_continuations,
            counterexample,
            status,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "chosenActionHash": self.chosen_action_hash,
            "recommendedActionHash": self.recommended_action_hash,
            "legalActionCount": self.legal_action_count,
            "evaluatedActionCount": self.evaluated_action_count,
            "responseModel": self.response_model,
            "continuationStatuses": list(self.continuation_statuses),
            "minimumExactShare": self.minimum_exact_share,
            "expectedRegret": self.expected_regret,
            "allAlternativesEvaluated": self.all_alternatives_evaluated,
            "pairedActorSafeCohort": self.paired_actor_safe_cohort,
            "exactContinuationCoverage": self.exact_continuation_coverage,
            "materialCounterexampleFound": self.material_counterexample_found,
            "status": self.status,
            "sha256": self.sha256,
        }
