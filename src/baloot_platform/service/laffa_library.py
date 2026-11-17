"""One read-only Laffa library projection over authoritative source records."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.research.audit_trace import ResearchAuditTraceRecordV1


_SOURCE_PRIORITY = {
    "archived_game": 1,
    "sakka_round": 2,
    "research_audit_trace": 3,
    "research_result": 4,
    "qualified_laffa": 5,
}
_GOVERNANCE_FIELDS = (
    "qualificationStatus",
    "auditOnly",
    "publicationEligible",
    "qualificationFailures",
    "truthLabel",
)


class _AuditTraceUnavailableError(Exception):
    pass


class _AuditTraceIntegrityError(Exception):
    pass
_RENDER_PRIORITY = {
    "not_prepared": 0,
    "prepared": 1,
    "cancelled": 2,
    "failed": 3,
    "completed": 4,
    "queued": 5,
    "running": 6,
}


def _timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if text:
        return text
    return "1970-01-01T00:00:00+00:00"


def _strings(values: Any) -> list[str]:
    return sorted({str(item).strip().lower() for item in values or () if str(item).strip()})


def _qualification_failures(value: Mapping[str, Any]) -> list[str]:
    qualification = value.get("qualification")
    nested = (
        dict(qualification).get("failures")
        if isinstance(qualification, Mapping)
        else ()
    )
    return _strings([*(value.get("qualificationFailures") or ()), *(nested or ())])


def _source(kind: str, code: str, identity: str, uri: str) -> dict[str, str]:
    return {"kind": kind, "code": code, "identity": identity, "uri": uri}


class LaffaLibraryServiceV1:
    """Federate query metadata while every semantic object stays authoritative."""

    def __init__(
        self,
        *,
        learning_store: Any,
        research: Any | None,
        foundry_catalogs: Any | None,
        render_batches: Any | None,
        active_bundle_hash: str | None = None,
    ) -> None:
        self.learning_store = learning_store
        self.research = research
        self.foundry_catalogs = foundry_catalogs
        self.render_batches = render_batches
        self.active_bundle_hash = str(active_bundle_hash or "")
        self._sakka_cache_key: tuple[str, ...] = ()
        self._sakka_cache: dict[str, list[dict[str, Any]]] = {}
        self._research_cache_key: tuple[Any, ...] = ()
        self._research_cache: list[dict[str, Any]] = []

    def _research_results(self) -> list[dict[str, Any]]:
        if self.research is None:
            return []
        values: list[dict[str, Any]] = []
        offset = 0
        # The store validates every compact projection against the immutable
        # payload and its hashes before returning it.  Re-reading every V4
        # detail here made the first library page O(total results).
        reader = getattr(self.research, "store", self.research)
        list_page = getattr(reader, "list_result_library_projections", None)
        first_page = (
            list_page(offset=0, limit=500)
            if list_page is not None
            else reader.list_results(sort="newest", offset=0, limit=500)
        )
        first_items = [dict(item) for item in first_page.get("items") or ()]
        cache_key = (
            first_page.get("total"),
            *(
                (
                    first_items[0].get("resultCode"),
                    first_items[0].get("resultHash"),
                )
                if first_items
                else (None, None)
            ),
        )
        if list_page is not None and cache_key == self._research_cache_key:
            return self._research_cache
        page = first_page
        while True:
            items = first_items if offset == 0 else [
                dict(item) for item in page.get("items") or ()
            ]
            for item in items:
                projected = item
                artifacts = dict(projected.get("artifacts") or {})
                values.append({
                    key: projected.get(key)
                    for key in (
                        "schemaVersion", "resultCode", "resultHash", "runCode",
                        "traceHash", "gameHash", "truthLabel", "confidence",
                        "score", "disposition", "publicationEligible",
                        "qualificationFailures", "qualification", "classification",
                        "tacticalTags", "verifiedTacticalTags", "interestingSeats",
                        "libraryProjection", "scenarioProjection", "createdAt",
                    )
                } | {
                    "artifacts": artifacts,
                    "_libraryReplayValid": bool(
                        isinstance(artifacts.get("exactState"), Mapping)
                        or (
                            projected.get("schemaVersion") == 2
                            and bool(projected.get("traceHash"))
                            and dict(projected.get("scenarioProjection") or {}).get(
                                "exactStateAvailable"
                            )
                            is True
                        )
                    ),
                })
            next_offset = page.get("nextOffset")
            if next_offset is None or not items:
                if list_page is not None:
                    self._research_cache_key = cache_key
                    self._research_cache = values
                return values
            offset = int(next_offset)
            page = (
                list_page(offset=offset, limit=500)
                if list_page is not None
                else reader.list_results(sort="newest", offset=offset, limit=500)
            )

    def _research_audit_traces(self) -> list[dict[str, Any]]:
        if self.research is None or not hasattr(
            self.research, "list_audit_traces"
        ):
            return []
        values: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.research.list_audit_traces(offset=offset, limit=500)
            items = [dict(item) for item in page.get("items") or ()]
            values.extend(items)
            next_offset = page.get("nextOffset")
            if next_offset is None or not items:
                return values
            offset = int(next_offset)

    def _load_research_audit_trace(
        self, record: ResearchAuditTraceRecordV1
    ) -> CompleteGameTraceV1:
        if self.research is None or not hasattr(
            self.research, "load_audit_trace"
        ):
            raise _AuditTraceUnavailableError(
                "Research audit trace loader is unavailable"
            )
        try:
            value = self.research.load_audit_trace(record.audit_trace_code)
        except (FileNotFoundError, KeyError) as exc:
            raise _AuditTraceUnavailableError(
                "Research audit trace artifact is unavailable"
            ) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _AuditTraceIntegrityError(
                "Research audit trace artifact failed integrity validation"
            ) from exc
        try:
            trace = (
                value
                if isinstance(value, CompleteGameTraceV1)
                else CompleteGameTraceV1.from_dict(value)
            )
        except (TypeError, ValueError) as exc:
            raise _AuditTraceIntegrityError(
                "Research audit trace payload is invalid"
            ) from exc
        if trace.sha256 != record.trace_hash:
            raise _AuditTraceIntegrityError(
                "Research audit trace identity mismatch"
            )
        return trace

    def resolve_audit_trace(self, identifier: str) -> dict[str, Any] | None:
        """Resolve one checksum-verified audit trace for an internal render path."""

        selected = str(identifier or "").strip()
        if not selected or self.research is None:
            return None
        raw: Mapping[str, Any] | None = None
        if hasattr(self.research, "get_audit_trace"):
            raw = self.research.get_audit_trace(selected)
        else:
            for item in self._research_audit_traces():
                aliases = {
                    str(item.get("auditTraceCode") or ""),
                    str(item.get("recordHash") or ""),
                    str(item.get("traceHash") or ""),
                }
                if selected in aliases:
                    raw = item
                    break
        if raw is None:
            return None
        record = ResearchAuditTraceRecordV1.build(raw)
        try:
            self._load_research_audit_trace(record)
        except (_AuditTraceIntegrityError, _AuditTraceUnavailableError) as exc:
            raise ValueError(
                "Research audit trace failed checksum-bound replay loading"
            ) from exc
        return record.to_dict()

    def _archived_games(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.learning_store.list_games(query="", offset=offset, limit=1_000)
            items = [dict(item) for item in page]
            values.extend(items)
            if len(items) < 1_000:
                return values
            offset += len(items)

    def _qualified_entries(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if self.foundry_catalogs is None or not hasattr(self.foundry_catalogs, "list"):
            return []
        result: list[tuple[dict[str, Any], dict[str, Any]]] = []
        after: str | None = None
        while True:
            page = self.foundry_catalogs.list(
                kind="qualified_laffa_v2", after=after, limit=500
            )
            records = [dict(item) for item in page.get("items") or ()]
            for record in records:
                catalog_hash = str(record.get("catalogHash") or "")
                if not catalog_hash:
                    continue
                document = self.foundry_catalogs.get(catalog_hash)
                prepared = dict(dict(document).get("document", {}).get("preparedItem") or {})
                if prepared:
                    result.append((record, prepared))
            after = page.get("nextCursor")
            if not after or not records:
                return result

    def _sakka_rounds(self) -> dict[str, list[dict[str, Any]]]:
        if (
            self.foundry_catalogs is None
            or not hasattr(self.foundry_catalogs, "list_sakkas")
            or not hasattr(self.foundry_catalogs, "load_sakka")
        ):
            return {}
        records: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            page = self.foundry_catalogs.list_sakkas(after=after, limit=500)
            selected = [dict(item) for item in page.get("items") or ()]
            records.extend(selected)
            after = page.get("nextCursor")
            if not after or not selected:
                break
        cache_key = tuple(sorted(str(item.get("catalogHash") or "") for item in records))
        if cache_key == self._sakka_cache_key:
            return self._sakka_cache
        result: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            catalog_hash = str(record.get("catalogHash") or "")
            if not catalog_hash:
                continue
            sakka = self.foundry_catalogs.load_sakka(catalog_hash)
            stored = dict(self.foundry_catalogs.get(catalog_hash))
            document = dict(stored.get("document") or {})
            reference_key = (
                "traceReferences"
                if str(record.get("kind") or "") == "natural_sakka_v1"
                else "descendantTraceReferences"
            )
            references = [dict(item) for item in document.get(reference_key) or ()]
            if len(references) != len(sakka.rounds):
                raise ValueError("Sakka round trace references do not match its rounds")
            locator = {
                "code": f"SAK-{catalog_hash[:10].upper()}",
                "identity": catalog_hash,
                "uri": f"/api/v2/sakkas/{catalog_hash}",
            }
            for round_value, reference in zip(sakka.rounds, references):
                trace_hash = str(
                    round_value.get("traceHash")
                    or round_value.get("descendantTraceHash")
                    or ""
                )
                if not trace_hash:
                    continue
                if str(reference.get("ownerHash") or "") != trace_hash:
                    raise ValueError("Sakka round trace reference identity mismatch")
                result.setdefault(trace_hash, []).append({
                    "traceHash": trace_hash,
                    "traceReference": reference,
                    "round": dict(round_value),
                    "sakka": locator,
                    "createdAt": _timestamp(record.get("createdAt")),
                })
        self._sakka_cache_key = cache_key
        self._sakka_cache = result
        return result

    def _sakka_index(self) -> dict[str, list[dict[str, str]]]:
        return {
            trace_hash: [dict(item["sakka"]) for item in entries]
            for trace_hash, entries in self._sakka_rounds().items()
        }

    def resolve_sakka_round(self, identifier: str) -> dict[str, Any] | None:
        """Resolve one replay-valid Sakka round without exposing its CAS reference."""

        selected = str(identifier or "").strip()
        for trace_hash, entries in sorted(self._sakka_rounds().items()):
            code = f"LAF-{trace_hash[:10].upper()}"
            if selected not in {trace_hash, code}:
                continue
            entry = min(
                entries,
                key=lambda item: (
                    str(dict(item["sakka"])["identity"]),
                    int(dict(item["round"]).get("ordinal") or 0),
                ),
            )
            return {**entry, "code": code}
        return None

    def _research_state(self, result: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        if self.research is None:
            return {}, False
        projected_replay = result.get("_libraryReplayValid")
        if type(projected_replay) is bool:
            return {}, projected_replay
        artifacts = dict(result.get("artifacts") or {})
        state_key = "trustedFinalState"
        reference = dict(artifacts.get("completeTrace") or {})
        if not reference:
            return {}, False
        try:
            payload = self.research.artifact_store.get(str(reference.get("uri") or ""))
        except (FileNotFoundError, KeyError):
            return {}, False
        if len(payload) != int(reference.get("bytes", -1)):
            return {}, False
        if hashlib.sha256(payload).hexdigest() != str(reference.get("sha256") or ""):
            return {}, False
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}, False
        if not isinstance(value, Mapping):
            return {}, False
        if result.get("schemaVersion") == 2 and isinstance(value.get("exactNormalization"), Mapping):
            value = value["exactNormalization"]
            state_key = "state"
        if not isinstance(value.get(state_key), Mapping):
            return {}, False
        return dict(value[state_key]), True

    def _render_indexes(self) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]], dict[str, str]]:
        by_source: dict[tuple[str, str], set[str]] = {}
        by_identity: dict[str, set[str]] = {}
        statuses: dict[str, str] = {}
        if self.render_batches is not None:
            for job in self.render_batches.list(limit=5_000):
                job_id = str(job.get("jobId") or job.get("id") or "")
                if not job_id:
                    continue
                statuses[job_id] = str(job.get("status") or "queued")
                for item in dict(job.get("request") or {}).get("items") or ():
                    source = dict(item.get("source") or {})
                    kind, code = str(source.get("kind") or ""), str(source.get("code") or "")
                    if kind == "research_scenario_result":
                        kind = "research_result"
                    if kind and code:
                        by_source.setdefault((kind, code), set()).add(job_id)
                    for key in ("traceHash", "gameHash"):
                        identity = str(source.get(key) or "")
                        if identity:
                            by_identity.setdefault(identity, set()).add(job_id)
        try:
            video_records = self.learning_store.list_video_records(limit=5_000)
        except (AttributeError, TypeError):
            video_records = []
        for record in video_records:
            game_hash = str(record.get("gameHash") or "")
            video_id = f"video:{record.get('videoId')}"
            if game_hash:
                by_identity.setdefault(game_hash, set()).add(video_id)
                statuses[video_id] = "completed"
        return by_source, by_identity, statuses

    @staticmethod
    def _merge(base: dict[str, Any], incoming: dict[str, Any]) -> None:
        base["identityKeys"].update(incoming["identityKeys"])
        existing = {(item["kind"], item["identity"]) for item in base["sourceReferences"]}
        base["sourceReferences"].extend(
            item for item in incoming["sourceReferences"]
            if (item["kind"], item["identity"]) not in existing
        )
        base["sourceKinds"].update(incoming["sourceKinds"])
        base["tacticalTags"].update(incoming["tacticalTags"])
        base["projects"].update(incoming["projects"])
        base["interestingSeats"].update(incoming["interestingSeats"])
        base["replayValid"] = bool(base["replayValid"] or incoming["replayValid"])
        for field in ("traceHash", "gameHash", "contract", "buyer", "dealer"):
            if not base.get(field) and incoming.get(field):
                base[field] = incoming[field]
        for field in ("confidence", "challengeScore"):
            candidates = [item for item in (base.get(field), incoming.get(field)) if item is not None]
            base[field] = max(candidates) if candidates else None
        if incoming["createdAt"] < base["createdAt"]:
            base["createdAt"] = incoming["createdAt"]
        if _SOURCE_PRIORITY[incoming["sourceKind"]] > _SOURCE_PRIORITY[base["sourceKind"]]:
            base["sourceKind"] = incoming["sourceKind"]
            base["humanCode"] = incoming["humanCode"]
            base["sourceLocator"] = incoming["sourceLocator"]
            for field in _GOVERNANCE_FIELDS:
                base[field] = incoming[field]

    def _candidates(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for game in self._archived_games():
            game_hash = str(game.get("gameHash") or "")
            if not game_hash:
                continue
            if (
                self.active_bundle_hash
                and str(game.get("bundleHash") or "") != self.active_bundle_hash
            ):
                continue
            complete = self.learning_store.get_game(game_hash) or {}
            state = dict(complete.get("state") or {})
            verification = dict(complete.get("verification") or self.learning_store.verify_game(game_hash))
            if not bool(verification.get("valid")):
                continue
            code = str(game.get("serial") or game_hash[:12].upper())
            locator = _source("archived_game", code, game_hash, f"/api/v1/history/games/{game_hash}")
            values.append({
                "identityKeys": {game_hash}, "traceHash": None, "gameHash": game_hash,
                "humanCode": code, "sourceKind": "archived_game", "sourceKinds": {"archived_game"},
                "sourceLocator": locator, "sourceReferences": [locator],
                "replayValid": bool(verification.get("valid")), "qualificationStatus": "unqualified",
                "auditOnly": True, "publicationEligible": False,
                "qualificationFailures": ["content_qualification_not_established"],
                "truthLabel": None, "confidence": None, "challengeScore": None,
                "tacticalTags": set(), "contract": game.get("contract") or state.get("contract"),
                "projects": {str(item.get("kind")) for item in state.get("projects") or () if item.get("kind")},
                "buyer": game.get("buyer") or state.get("buyer"), "dealer": state.get("dealer"),
                "interestingSeats": set(), "createdAt": _timestamp(game.get("createdAt")),
                "prepared": False,
            })

        for trace_hash, entries in sorted(self._sakka_rounds().items()):
            entry = min(
                entries,
                key=lambda item: (
                    str(dict(item["sakka"])["identity"]),
                    int(dict(item["round"]).get("ordinal") or 0),
                ),
            )
            round_value = dict(entry["round"])
            sakka = dict(entry["sakka"])
            code = f"LAF-{trace_hash[:10].upper()}"
            locator = _source(
                "sakka_round", code, trace_hash, str(sakka["uri"])
            )
            values.append({
                "identityKeys": {trace_hash}, "traceHash": trace_hash, "gameHash": None,
                "humanCode": code, "sourceKind": "sakka_round", "sourceKinds": {"sakka_round"},
                "sourceLocator": locator, "sourceReferences": [locator], "replayValid": True,
                "qualificationStatus": "unqualified", "truthLabel": None,
                "auditOnly": True, "publicationEligible": False,
                "qualificationFailures": ["content_qualification_not_established"],
                "confidence": None, "challengeScore": None, "tacticalTags": set(),
                "contract": None, "projects": set(), "buyer": None,
                "dealer": round_value.get("dealer"), "interestingSeats": set(),
                "createdAt": str(entry["createdAt"]), "prepared": False,
            })

        for raw_record in self._research_audit_traces():
            try:
                record = ResearchAuditTraceRecordV1.build(raw_record)
            except (TypeError, ValueError):
                continue
            trace: CompleteGameTraceV1 | None = None
            failures = set(record.failure_categories)
            failures.add("audit_trace_publication_not_authorized")
            try:
                trace = self._load_research_audit_trace(record)
            except _AuditTraceUnavailableError:
                failures.add("audit_trace_artifact_unavailable")
            except _AuditTraceIntegrityError:
                failures.add("audit_trace_integrity_check_failed")
            state = {} if trace is None else dict(trace.trusted_final_state)
            game_hash = None if trace is None else trace.game_hash
            code = record.audit_trace_code
            locator = _source(
                "research_audit_trace",
                code,
                record.record_hash,
                f"/api/v2/laffat/{code}",
            )
            values.append({
                "identityKeys": {
                    item for item in (record.trace_hash, game_hash) if item
                },
                "traceHash": record.trace_hash,
                "gameHash": game_hash,
                "humanCode": code,
                "sourceKind": "research_audit_trace",
                "sourceKinds": {"research_audit_trace"},
                "sourceLocator": locator,
                "sourceReferences": [locator],
                "replayValid": trace is not None,
                "qualificationStatus": record.disposition,
                "auditOnly": True,
                "publicationEligible": False,
                "qualificationFailures": sorted(failures),
                "truthLabel": None,
                "confidence": None,
                "challengeScore": None,
                "tacticalTags": set(),
                "contract": state.get("contract"),
                "projects": {
                    str(item.get("kind"))
                    for item in state.get("projects") or ()
                    if isinstance(item, Mapping) and item.get("kind")
                },
                "buyer": state.get("buyer"),
                "dealer": state.get("dealer"),
                "interestingSeats": {
                    item.acting_seat.name.lower()
                    for item in (() if trace is None else trace.decision_ledger.records)
                },
                "createdAt": _timestamp(record.created_at),
                "prepared": False,
            })

        for result in self._research_results():
            trace_hash, game_hash = str(result.get("traceHash") or ""), str(result.get("gameHash") or "")
            if not trace_hash and not game_hash:
                continue
            code = str(result.get("resultCode") or trace_hash[:12].upper())
            state, replay_valid = self._research_state(result)
            projection = dict(result.get("libraryProjection") or {})
            disposition = str(result.get("disposition") or "unqualified")
            publication_eligible = bool(
                disposition == "qualified"
                and result.get("publicationEligible") is True
            )
            failures = set(_qualification_failures(result))
            if not publication_eligible:
                failures.add(
                    "publication_eligibility_not_established"
                    if disposition == "qualified"
                    else "content_qualification_not_established"
                )
            locator = _source("research_result", code, str(result.get("resultHash") or code), f"/api/v2/research-results/{code}")
            scenario_v4 = result.get("classification") == "scenario_v4"
            values.append({
                "identityKeys": {item for item in (trace_hash, game_hash) if item},
                "traceHash": trace_hash or None, "gameHash": game_hash or None,
                "humanCode": code, "sourceKind": "research_result", "sourceKinds": {"research_result"},
                "sourceLocator": locator, "sourceReferences": [locator],
                "replayValid": replay_valid,
                "qualificationStatus": disposition,
                "auditOnly": not publication_eligible,
                "publicationEligible": publication_eligible,
                "qualificationFailures": sorted(failures),
                "truthLabel": result.get("truthLabel"),
                "confidence": None if scenario_v4 else result.get("confidence"),
                "challengeScore": None if scenario_v4 else result.get("score"),
                "tacticalTags": set(_strings([*(result.get("tacticalTags") or ()), *(result.get("verifiedTacticalTags") or ())])),
                "contract": projection.get("contract") or state.get("contract"),
                "projects": set(_strings(projection.get("projectKinds") or ())),
                "buyer": state.get("buyer"), "dealer": state.get("dealer"),
                "interestingSeats": set(_strings([*(result.get("interestingSeats") or ()), *(projection.get("actingSeats") or ())])),
                "createdAt": _timestamp(result.get("createdAt")), "prepared": False,
            })

        for record, prepared in self._qualified_entries():
            trace_hash = str(prepared.get("traceHash") or "")
            if not trace_hash:
                continue
            catalog_hash = str(record.get("catalogHash") or "")
            code = str(prepared.get("shortCode") or f"LAF-{trace_hash[:10].upper()}")
            publication_eligible = prepared.get("publicationEligible") is True
            failures = set(_qualification_failures(prepared))
            if not publication_eligible:
                failures.add("publication_eligibility_not_established")
            locator = _source("qualified_laffa", code, catalog_hash, f"/api/v2/foundry-catalogs/{catalog_hash}")
            values.append({
                "identityKeys": {trace_hash}, "traceHash": trace_hash, "gameHash": None,
                "humanCode": code, "sourceKind": "qualified_laffa", "sourceKinds": {"qualified_laffa"},
                "sourceLocator": locator, "sourceReferences": [locator], "replayValid": True,
                "qualificationStatus": "qualified", "truthLabel": prepared.get("truthLabel"),
                "auditOnly": not publication_eligible,
                "publicationEligible": publication_eligible,
                "qualificationFailures": sorted(failures),
                "confidence": prepared.get("confidence"),
                "challengeScore": dict(prepared.get("laffaSelectionScore") or {}).get("score"),
                "tacticalTags": set(_strings(prepared.get("interestingProfiles") or ())),
                "contract": None, "projects": set(), "buyer": None, "dealer": None,
                "interestingSeats": set(), "createdAt": _timestamp(record.get("createdAt")),
                "prepared": bool(prepared.get("bundleArtifact")),
            })
        return values

    def list(
        self,
        *,
        search: str = "",
        source_kind: str | None = None,
        replay_valid: bool | None = None,
        qualification_status: str | None = None,
        truth_label: str | None = None,
        contract: str | None = None,
        project: str | None = None,
        tactic: str | None = None,
        seat: str | None = None,
        render_status: str | None = None,
        sort: str = "newest",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        aliases: dict[str, dict[str, Any]] = {}
        for candidate in self._candidates():
            matches = {id(aliases[key]): aliases[key] for key in candidate["identityKeys"] if key in aliases}
            if matches:
                target = next(iter(matches.values()))
                self._merge(target, candidate)
                for other in list(matches.values())[1:]:
                    if other is target:
                        continue
                    self._merge(target, other)
                    if other in groups:
                        groups.remove(other)
            else:
                target = candidate
                groups.append(target)
            for key in target["identityKeys"]:
                aliases[key] = target

        sakka_index = self._sakka_index()
        render_sources, render_identities, render_states = self._render_indexes()
        rows: list[dict[str, Any]] = []
        for value in groups:
            related = {
                item["identity"]: item
                for identity in value["identityKeys"]
                for item in sakka_index.get(identity, ())
            }
            job_ids: set[str] = set()
            for reference in value["sourceReferences"]:
                job_ids.update(render_sources.get((reference["kind"], reference["code"]), ()))
            for identity in value["identityKeys"]:
                job_ids.update(render_identities.get(identity, ()))
            statuses = [render_states[item] for item in job_ids if item in render_states]
            status = max(statuses, key=lambda item: _RENDER_PRIORITY.get(item, -1)) if statuses else (
                "prepared" if value["prepared"] else "not_prepared"
            )
            row = {
                "humanCode": value["humanCode"], "sourceKind": value["sourceKind"],
                "sourceKinds": sorted(value["sourceKinds"]), "replayValid": value["replayValid"],
                "auditOnly": value["auditOnly"],
                "publicationEligible": value["publicationEligible"],
                "qualificationFailures": list(value["qualificationFailures"]),
                "qualificationStatus": value["qualificationStatus"], "truthLabel": value["truthLabel"],
                "confidence": value["confidence"], "challengeScore": value["challengeScore"],
                "tacticalTags": sorted(value["tacticalTags"]), "contract": value["contract"],
                "projects": sorted(value["projects"]), "buyer": value["buyer"], "dealer": value["dealer"],
                "interestingSeats": sorted(value["interestingSeats"]),
                "relatedSakkas": [related[key] for key in sorted(related)],
                "renderCount": len(job_ids), "renderStatus": status, "createdAt": value["createdAt"],
                "traceHash": value["traceHash"], "gameHash": value["gameHash"],
                "sourceLocator": value["sourceLocator"],
                "sourceReferences": sorted(value["sourceReferences"], key=lambda item: (item["kind"], item["identity"])),
            }
            row["project"] = row["projects"][0] if row["projects"] else None
            row["seat"] = row["interestingSeats"][0] if row["interestingSeats"] else None
            rows.append(row)

        source_facets = {
            kind: sum(kind in row["sourceKinds"] for row in rows)
            for kind in sorted(_SOURCE_PRIORITY)
        }
        needle = search.strip().lower()
        def accepted(row: Mapping[str, Any]) -> bool:
            haystack = " ".join([
                str(row.get("humanCode") or ""), str(row.get("contract") or ""),
                *row.get("projects", ()), *row.get("tacticalTags", ()),
            ]).lower()
            return (
                (not needle or needle in haystack)
                and (source_kind is None or source_kind in row["sourceKinds"])
                and (replay_valid is None or row["replayValid"] is replay_valid)
                and (qualification_status is None or row["qualificationStatus"] == qualification_status)
                and (truth_label is None or row["truthLabel"] == truth_label)
                and (contract is None or row["contract"] == contract)
                and (project is None or project in row["projects"])
                and (tactic is None or tactic in row["tacticalTags"])
                and (
                    seat is None
                    or seat == row["seat"]
                    or seat in row["interestingSeats"]
                )
                and (render_status is None or row["renderStatus"] == render_status)
            )
        filtered = [row for row in rows if accepted(row)]
        sorters = {
            "newest": lambda row: (str(row["createdAt"]), row["humanCode"]),
            "oldest": lambda row: (str(row["createdAt"]), row["humanCode"]),
            "score_desc": lambda row: (float(row["challengeScore"] or -1.0), row["humanCode"]),
            "confidence_desc": lambda row: (float(row["confidence"] or -1.0), row["humanCode"]),
            "code_asc": lambda row: row["humanCode"],
        }
        if sort not in sorters:
            raise ValueError("unsupported Laffa library sort")
        reverse = sort in {"newest", "score_desc", "confidence_desc"}
        filtered.sort(key=sorters[sort], reverse=reverse)
        start, size = max(0, int(offset)), max(1, min(500, int(limit)))
        selected = filtered[start : start + size]
        return {
            "schemaVersion": 1,
            "resourceKind": "unified_laffa_library_v1",
            "items": selected,
            "total": len(filtered),
            "offset": start,
            "limit": size,
            "nextOffset": start + len(selected) if start + len(selected) < len(filtered) else None,
            "sourceFacets": [{"sourceKind": key, "count": value} for key, value in source_facets.items()],
        }

    def get(self, identifier: str) -> dict[str, Any] | None:
        """Resolve one canonical row by its human code or authoritative identity."""

        selected = str(identifier or "").strip()
        if not selected:
            return None
        offset = 0
        while True:
            page = self.list(sort="code_asc", offset=offset, limit=500)
            for row in page["items"]:
                aliases = {
                    str(row.get("humanCode") or ""),
                    str(row.get("traceHash") or ""),
                    str(row.get("gameHash") or ""),
                }
                for reference in row.get("sourceReferences") or ():
                    aliases.add(str(reference.get("code") or ""))
                    aliases.add(str(reference.get("identity") or ""))
                if selected in aliases:
                    detail = dict(row)
                    getter = getattr(self.research, "get_result", None)
                    # Keep list queries compact; validate only the opened result.
                    if getter is not None and any(not detail.get(key) for key in ("buyer", "dealer")):
                        reference = next((item for item in detail["sourceReferences"]
                                          if item["kind"] == "research_result"), None)
                        if reference is not None:
                            result = getter(reference["code"])
                            if result is not None:
                                if result.get("resultHash") != reference["identity"]:
                                    raise RuntimeError("research library detail identity mismatch")
                                state, valid = self._research_state(result)
                                if valid:
                                    for key in ("buyer", "dealer"):
                                        if isinstance(state.get(key), str) and state[key] in {"north", "west", "south", "east"}:
                                            detail[key] = state[key]
                    return detail
            next_offset = page.get("nextOffset")
            if next_offset is None:
                return None
            offset = int(next_offset)
