from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def bundle_payload_fingerprint(mode: str, profile_hashes: Mapping[str, str]) -> str:
    return digest({"mode": mode, "profiles": dict(profile_hashes)})


def configuration_tenant_visible(
    scope: str, tenant_id: str | None, visible_to_tenant: str | None,
) -> bool:
    return visible_to_tenant is None or scope == "global" or tenant_id == visible_to_tenant


class RevisionStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    REJECTED = "rejected"
    RETIRED = "retired"
    SUPERSEDED = "superseded"


class ConfigScope(StrEnum):
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    TENANT = "tenant"
    USER_EXPERIMENT = "user_experiment"
    ROOM = "room"
    SHARED_GAME = "shared_game"
    VIDEO_PROFILE = "video_profile"


@dataclass(frozen=True, slots=True)
class ResolutionContext:
    tenant_id: str | None = None
    environment: str | None = None
    user_experiment_key: str | None = None
    room_id: str | None = None
    shared_game_id: str | None = None
    video_profile_id: str | None = None
    subject_id: str | None = None
    at_time: str | None = None

    def cache_key(self) -> tuple[str, ...]:
        return tuple(str(value or "") for value in (
            self.tenant_id, self.environment, self.user_experiment_key,
            self.room_id, self.shared_game_id, self.video_profile_id,
            self.subject_id, self.at_time,
        ))


@dataclass(frozen=True, slots=True)
class ProfileRevision:
    revision_id: str
    family: str
    profile_id: str
    version: str
    scope: str
    scope_key: str
    tenant_id: str | None
    environment: str | None
    status: str
    payload: Mapping[str, Any]
    payload_sha256: str
    sequence: int
    created_by: str
    created_at: str
    updated_at: str
    effective_from: str | None = None
    effective_until: str | None = None
    rollout_percentage: float = 100.0
    rollout_seed: str = "default"
    published_by: str | None = None
    published_at: str | None = None
    supersedes_revision_id: str | None = None
    etag: int = 1
    local_schedule: Mapping[str, str] | None = None

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "revisionId": self.revision_id,
            "family": self.family,
            "profileId": self.profile_id,
            "version": self.version,
            "scope": self.scope,
            "scopeType": self.scope,
            "scopeKey": self.scope_key,
            "tenantId": self.tenant_id,
            "environment": self.environment,
            "status": self.status,
            "payloadSha256": self.payload_sha256,
            "sequence": self.sequence,
            "effectiveFrom": self.effective_from,
            "effectiveUntil": self.effective_until,
            "rolloutPercentage": self.rollout_percentage,
            "rolloutSeed": self.rollout_seed,
            "createdBy": self.created_by,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "publishedBy": self.published_by,
            "publishedAt": self.published_at,
            "supersedesRevisionId": self.supersedes_revision_id,
            "etag": self.etag,
        }
        if include_payload:
            result["payload"] = dict(self.payload)
        if self.local_schedule is not None:
            result["localSchedule"] = dict(self.local_schedule)
        return result


@dataclass(frozen=True, slots=True)
class BundleRevision:
    revision_id: str
    bundle_id: str
    version: str
    mode: str
    scope: str
    scope_key: str
    tenant_id: str | None
    environment: str | None
    status: str
    component_revisions: Mapping[str, str]
    content_sha256: str
    sequence: int
    created_by: str
    created_at: str
    updated_at: str
    effective_from: str | None = None
    effective_until: str | None = None
    rollout_percentage: float = 100.0
    rollout_seed: str = "default"
    published_by: str | None = None
    published_at: str | None = None
    supersedes_revision_id: str | None = None
    etag: int = 1
    local_schedule: Mapping[str, str] | None = None
    scheduled_predecessor_id: str | None = None
    scheduled_successor_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "revisionId": self.revision_id,
            "bundleId": self.bundle_id,
            "version": self.version,
            "mode": self.mode,
            "scope": self.scope,
            "scopeType": self.scope,
            "scopeKey": self.scope_key,
            "tenantId": self.tenant_id,
            "environment": self.environment,
            "status": self.status,
            "componentRevisions": dict(self.component_revisions),
            "contentSha256": self.content_sha256,
            "sequence": self.sequence,
            "effectiveFrom": self.effective_from,
            "effectiveUntil": self.effective_until,
            "rolloutPercentage": self.rollout_percentage,
            "rolloutSeed": self.rollout_seed,
            "createdBy": self.created_by,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "publishedBy": self.published_by,
            "publishedAt": self.published_at,
            "supersedesRevisionId": self.supersedes_revision_id,
            "etag": self.etag,
        }
        if self.local_schedule is not None:
            result["localSchedule"] = dict(self.local_schedule)
        return result


@dataclass(frozen=True, slots=True)
class EffectiveRuntimeConfigV2:
    """Immutable V2 projection of the existing authoritative resolver output."""

    payload_json: str
    content_sha256: str

    @staticmethod
    def _identities(values: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
        allowed = {
            "id",
            "version",
            "kind",
            "sha256",
            "artifactSha256",
            "configSha256",
        }
        normalized = [copy.deepcopy(dict(value)) for value in values]
        for value in normalized:
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(
                    "runtime identity contains unknown keys: "
                    + ", ".join(sorted(unknown))
                )
            if not str(value.get("id") or "").strip():
                raise ValueError("runtime identity requires id")
            if not str(value.get("version") or "").strip():
                raise ValueError("runtime identity requires version")
            for name in ("sha256", "artifactSha256", "configSha256"):
                digest_value = value.get(name)
                if digest_value is not None and (
                    len(str(digest_value)) != 64
                    or any(character not in "0123456789abcdef" for character in str(digest_value))
                ):
                    raise ValueError(f"runtime identity {name} must be lowercase SHA-256")
        return sorted(
            normalized,
            key=lambda value: (
                str(value["id"]),
                str(value["version"]),
                canonical(value),
            ),
        )

    @classmethod
    def from_resolved(
        cls,
        resolved: Mapping[str, Any],
        *,
        plugin_identities: tuple[Mapping[str, Any], ...] = (),
        model_identities: tuple[Mapping[str, Any], ...] = (),
    ) -> "EffectiveRuntimeConfigV2":
        bundle = copy.deepcopy(dict(resolved["bundle"]))
        profiles = copy.deepcopy(dict(resolved["profiles"]))
        revisions = copy.deepcopy(dict(resolved["profileRevisions"]))
        component_revisions = {
            family: {
                "revisionId": revision["revisionId"],
                "profileId": revision["profileId"],
                "version": revision["version"],
                "payloadSha256": revision["payloadSha256"],
            }
            for family, revision in sorted(revisions.items())
        }
        payload = {
            "schemaVersion": 2,
            "contractVersion": "2.0.0",
            "bundle": {
                "revisionId": bundle["revisionId"],
                "bundleId": bundle["bundleId"],
                "version": bundle["version"],
                "mode": bundle["mode"],
                "contentSha256": bundle["contentSha256"],
            },
            "normalizedValues": profiles,
            "componentRevisions": component_revisions,
            "pluginIdentities": cls._identities(plugin_identities),
            "modelIdentities": cls._identities(model_identities),
            "resolutionTrace": copy.deepcopy(list(resolved["resolutionTrace"])),
            "compatibility": {
                "status": "compatible",
                "readAdapters": ["EffectiveRuntimeConfigV1"],
                "writeContract": "EffectiveRuntimeConfigV2",
            },
            "sourceEffectiveHash": str(resolved["effectiveHash"]),
            "invalidationEpoch": int(resolved.get("invalidationEpoch", 0)),
        }
        payload_json = canonical(payload)
        return cls(payload_json=payload_json, content_sha256=digest(payload))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffectiveRuntimeConfigV2":
        """Verify one externally stored V2 snapshot without resolver access."""

        snapshot = copy.deepcopy(dict(value))
        allowed = {
            "schemaVersion",
            "contractVersion",
            "bundle",
            "normalizedValues",
            "componentRevisions",
            "pluginIdentities",
            "modelIdentities",
            "resolutionTrace",
            "compatibility",
            "sourceEffectiveHash",
            "invalidationEpoch",
            "contentSha256",
        }
        if set(snapshot) != allowed:
            raise ValueError("EffectiveRuntimeConfigV2 contains missing or unknown keys")
        if int(snapshot["schemaVersion"]) != 2:
            raise ValueError("unsupported EffectiveRuntimeConfigV2 schema version")
        if snapshot["contractVersion"] != "2.0.0":
            raise ValueError("unsupported EffectiveRuntimeConfigV2 contract version")
        declared = str(snapshot.pop("contentSha256") or "")
        if len(declared) != 64 or any(
            character not in "0123456789abcdef" for character in declared
        ):
            raise ValueError("EffectiveRuntimeConfigV2 content hash is invalid")
        bundle = dict(snapshot["bundle"])
        if set(bundle) != {
            "revisionId", "bundleId", "version", "mode", "contentSha256"
        }:
            raise ValueError("EffectiveRuntimeConfigV2 bundle identity is invalid")
        for key in ("revisionId", "bundleId", "version", "mode"):
            if not str(bundle.get(key) or "").strip():
                raise ValueError("EffectiveRuntimeConfigV2 bundle identity is incomplete")
        bundle_hash = str(bundle.get("contentSha256") or "")
        if len(bundle_hash) != 64 or any(
            character not in "0123456789abcdef" for character in bundle_hash
        ):
            raise ValueError("EffectiveRuntimeConfigV2 bundle hash is invalid")
        profiles = dict(snapshot["normalizedValues"])
        revisions = dict(snapshot["componentRevisions"])
        if not profiles or set(profiles) != set(revisions):
            raise ValueError("EffectiveRuntimeConfigV2 component set is invalid")
        for family, raw_revision in revisions.items():
            revision = dict(raw_revision)
            if set(revision) != {
                "revisionId", "profileId", "version", "payloadSha256"
            }:
                raise ValueError(
                    f"EffectiveRuntimeConfigV2 {family} revision identity is invalid"
                )
            if not all(
                str(revision.get(key) or "").strip()
                for key in ("revisionId", "profileId", "version")
            ):
                raise ValueError(
                    f"EffectiveRuntimeConfigV2 {family} revision identity is incomplete"
                )
            if str(revision["payloadSha256"]) != digest(profiles[family]):
                raise ValueError(
                    f"EffectiveRuntimeConfigV2 {family} payload checksum mismatch"
                )
        plugins = tuple(dict(item) for item in snapshot["pluginIdentities"])
        models = tuple(dict(item) for item in snapshot["modelIdentities"])
        if cls._identities(plugins) != list(plugins):
            raise ValueError("EffectiveRuntimeConfigV2 plugin identities are not canonical")
        if cls._identities(models) != list(models):
            raise ValueError("EffectiveRuntimeConfigV2 model identities are not canonical")
        if not isinstance(snapshot["resolutionTrace"], list):
            raise ValueError("EffectiveRuntimeConfigV2 resolution trace must be a list")
        compatibility = dict(snapshot["compatibility"])
        if set(compatibility) != {"status", "readAdapters", "writeContract"}:
            raise ValueError("EffectiveRuntimeConfigV2 compatibility block is invalid")
        if (
            compatibility["status"] != "compatible"
            or compatibility["writeContract"] != "EffectiveRuntimeConfigV2"
            or compatibility["readAdapters"] != ["EffectiveRuntimeConfigV1"]
        ):
            raise ValueError("EffectiveRuntimeConfigV2 compatibility declaration is invalid")
        source_hash = str(snapshot["sourceEffectiveHash"] or "")
        if len(source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in source_hash
        ):
            raise ValueError("EffectiveRuntimeConfigV2 source hash is invalid")
        if int(snapshot["invalidationEpoch"]) < 0:
            raise ValueError("EffectiveRuntimeConfigV2 invalidation epoch is invalid")
        result = cls(payload_json=canonical(snapshot), content_sha256=declared)
        if not result.verify():
            raise ValueError("EffectiveRuntimeConfigV2 content checksum mismatch")
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "EffectiveRuntimeConfigV2":
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("EffectiveRuntimeConfigV2 bytes are invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("EffectiveRuntimeConfigV2 snapshot must be an object")
        if canonical(value).encode("utf-8") != data:
            raise ValueError("EffectiveRuntimeConfigV2 bytes are not canonical")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(self.payload_json)
        payload["contentSha256"] = self.content_sha256
        return payload

    def verify(self) -> bool:
        return digest(json.loads(self.payload_json)) == self.content_sha256


def migrate_effective_runtime_config_v1(
    frozen_v1: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    plugin_identities: tuple[Mapping[str, Any], ...] = (),
    model_identities: tuple[Mapping[str, Any], ...] = (),
) -> EffectiveRuntimeConfigV2:
    """Read a frozen V1 reference and deterministically project its exact V2 value."""

    if int(frozen_v1.get("schemaVersion", 0)) != 1:
        raise ValueError("EffectiveRuntimeConfigV1 adapter requires schemaVersion 1")
    expected_components = {
        family: revision["revisionId"]
        for family, revision in resolved["profileRevisions"].items()
    }
    checks = {
        "bundleRevisionId": (
            frozen_v1.get("bundleRevisionId"),
            resolved["bundle"]["revisionId"],
        ),
        "componentRevisionIds": (
            dict(frozen_v1.get("componentRevisionIds") or {}),
            expected_components,
        ),
        "effectiveHash": (
            frozen_v1.get("effectiveHash"),
            resolved["effectiveHash"],
        ),
    }
    mismatched = [
        name for name, (actual, expected) in checks.items() if actual != expected
    ]
    if mismatched:
        raise ValueError(
            "EffectiveRuntimeConfigV1 does not identify the supplied resolution: "
            + ", ".join(sorted(mismatched))
        )
    return EffectiveRuntimeConfigV2.from_resolved(
        resolved,
        plugin_identities=plugin_identities,
        model_identities=model_identities,
    )
