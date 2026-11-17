"""Trusted complete-game trace contract for Foundry derivations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.analysis.search_errors import require_uncensored_play_trace

from .evidence_v3 import DecisionLedgerV1, canonical_sha256


@dataclass(frozen=True, slots=True)
class TraceEventV1:
    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    previous_event_hash: str | None
    event_hash: str

    @classmethod
    def chain(cls, events: Iterable[Mapping[str, Any]]) -> tuple["TraceEventV1", ...]:
        result: list[TraceEventV1] = []
        previous: str | None = None
        for expected, event in enumerate(events):
            sequence = int(event.get("sequence", -1))
            if sequence != expected:
                raise ValueError("trace events must have contiguous zero-based sequence numbers")
            event_type = str(event.get("type") or "").strip()
            if not event_type:
                raise ValueError("trace event type cannot be empty")
            payload = {
                str(key): value
                for key, value in event.items()
                if key not in {"sequence", "type"}
            }
            identity = {
                "sequence": sequence,
                "eventType": event_type,
                "payload": payload,
                "previousEventHash": previous,
            }
            event_hash = canonical_sha256(identity)
            result.append(cls(sequence, event_type, payload, previous, event_hash))
            previous = event_hash
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "eventType": self.event_type,
            "payload": dict(self.payload),
            "previousEventHash": self.previous_event_hash,
            "eventHash": self.event_hash,
        }


@dataclass(frozen=True, slots=True)
class CompleteGameTraceV1:
    game_hash: str
    game_id: str
    state_hash: str
    rules_hash: str
    bundle_hash: str
    engine_version: str
    seed_provenance: Mapping[str, Any]
    agent_profiles: tuple[tuple[str, str], ...]
    model_releases: tuple[str, ...]
    decision_ledger: DecisionLedgerV1
    events: tuple[TraceEventV1, ...]
    trusted_final_state: Mapping[str, Any]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        game_record: Mapping[str, Any],
        trusted_final_state: Mapping[str, Any],
        decision_ledger: DecisionLedgerV1,
        model_releases: Iterable[str] = (),
    ) -> "CompleteGameTraceV1":
        game_hash = str(game_record.get("gameHash") or "").strip()
        game_id = str(game_record.get("gameId") or "").strip()
        state_hash = str(game_record.get("stateHash") or "").strip()
        bundle_hash = str(game_record.get("bundleHash") or "").strip()
        engine_version = str(game_record.get("engineVersion") or "").strip()
        if not all((game_hash, game_id, state_hash, bundle_hash, engine_version)):
            raise ValueError("complete trace requires immutable game and version provenance")
        for policy in trusted_final_state.get("decisionPolicyTraces") or ():
            require_uncensored_play_trace(policy.get("trace") or {})
        calculated_state_hash = canonical_sha256(trusted_final_state)
        if calculated_state_hash != state_hash:
            raise ValueError("trusted final state hash does not match the archived game")
        if decision_ledger.game_id != game_hash:
            raise ValueError("decision ledger does not belong to the archived game hash")
        rules_hash = str(trusted_final_state.get("rulesHash") or "").strip()
        if not rules_hash:
            raise ValueError("complete trace requires the exact rules hash")
        generation = trusted_final_state.get("generation")
        if isinstance(generation, Mapping) and generation.get("configurationSnapshotHash"):
            snapshot_value = trusted_final_state.get("configurationSnapshot")
            if not isinstance(snapshot_value, Mapping):
                raise ValueError("generated complete trace requires its frozen configuration snapshot")
            snapshot = dict(snapshot_value)
            declared_snapshot_hash = str(snapshot.pop("sha256", ""))
            if (
                int(snapshot.get("schemaVersion", 0)) != 1
                or str(snapshot.get("bundleHash") or "") != bundle_hash
                or canonical_sha256(snapshot) != declared_snapshot_hash
                or str(generation.get("configurationSnapshotHash") or "")
                != declared_snapshot_hash
            ):
                raise ValueError("generated complete trace configuration snapshot is invalid")
            rules_component = dict(
                dict(snapshot.get("components") or {}).get("rules") or {}
            )
            if (
                str(rules_component.get("sha256") or "") != rules_hash
                or canonical_sha256(rules_component.get("payload")) != rules_hash
            ):
                raise ValueError("generated complete trace configuration rules do not match")
        shuffle = dict(trusted_final_state.get("shuffle") or {})
        if not shuffle.get("algorithm") or not shuffle.get("deckCommitment"):
            raise ValueError("complete trace requires deterministic seed and deck commitment provenance")
        seed_provenance = {
            "rngAlgorithm": str(shuffle["algorithm"]),
            "seed": int(shuffle.get("seed", trusted_final_state.get("seed", 0))),
            "dealer": str(shuffle.get("dealer", trusted_final_state.get("dealer", ""))),
            "cutter": shuffle.get("cutter"),
            "cutPosition": shuffle.get("cutPosition"),
            "deckCommitment": str(shuffle["deckCommitment"]),
        }
        agent_profiles = tuple(
            sorted(
                (str(seat), str(profile))
                for seat, profile in dict(trusted_final_state.get("agentProfiles") or {}).items()
            )
        )
        models = tuple(sorted({str(item) for item in model_releases if str(item).strip()}))
        events = TraceEventV1.chain(trusted_final_state.get("events") or ())
        core = {
            "schemaVersion": 1,
            "gameHash": game_hash,
            "gameId": game_id,
            "stateHash": state_hash,
            "rulesHash": rules_hash,
            "bundleHash": bundle_hash,
            "engineVersion": engine_version,
            "seedProvenance": seed_provenance,
            "agentProfiles": dict(agent_profiles),
            "modelReleases": list(models),
            "decisionLedger": decision_ledger.to_dict(),
            "events": [event.to_dict() for event in events],
            "trustedFinalState": dict(trusted_final_state),
        }
        return cls(
            game_hash,
            game_id,
            state_hash,
            rules_hash,
            bundle_hash,
            engine_version,
            seed_provenance,
            agent_profiles,
            models,
            decision_ledger,
            events,
            dict(trusted_final_state),
            canonical_sha256(core),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompleteGameTraceV1":
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported complete trace schema version")
        trusted = value.get("trustedFinalState")
        if not isinstance(trusted, Mapping):
            raise ValueError("trusted complete trace loading requires its frozen final state")
        ledger_value = value.get("decisionLedger")
        if not isinstance(ledger_value, Mapping):
            raise ValueError("complete trace decision ledger is required")
        ledger = DecisionLedgerV1.build(
            str(ledger_value.get("gameId") or value.get("gameHash") or ""),
            tuple(ledger_value.get("decisions") or ()),
        )
        if ledger_value.get("sha256") and str(ledger_value["sha256"]) != ledger.sha256:
            raise ValueError("complete trace decision ledger checksum mismatch")
        trace = cls.build(
            game_record={
                "gameHash": value.get("gameHash"),
                "gameId": value.get("gameId"),
                "stateHash": value.get("stateHash"),
                "bundleHash": value.get("bundleHash"),
                "engineVersion": value.get("engineVersion"),
            },
            trusted_final_state=trusted,
            decision_ledger=ledger,
            model_releases=tuple(value.get("modelReleases") or ()),
        )
        declared = {
            "rulesHash": value.get("rulesHash"),
            "seedProvenance": value.get("seedProvenance"),
            "agentProfiles": value.get("agentProfiles"),
            "events": value.get("events"),
        }
        actual = {
            "rulesHash": trace.rules_hash,
            "seedProvenance": dict(trace.seed_provenance),
            "agentProfiles": dict(trace.agent_profiles),
            "events": [item.to_dict() for item in trace.events],
        }
        if declared != actual:
            raise ValueError("complete trace derived provenance does not match its trusted state")
        if value.get("sha256") and str(value["sha256"]) != trace.sha256:
            raise ValueError("complete trace checksum mismatch")
        return trace

    def to_dict(self, *, trusted: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "gameHash": self.game_hash,
            "gameId": self.game_id,
            "stateHash": self.state_hash,
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "seedProvenance": dict(self.seed_provenance),
            "agentProfiles": dict(self.agent_profiles),
            "modelReleases": list(self.model_releases),
            "decisionLedger": self.decision_ledger.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "sha256": self.sha256,
        }
        if trusted:
            value["trustedFinalState"] = dict(self.trusted_final_state)
        return value
