from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from baloot_platform.config import (
    PROFILE_FAMILIES,
    build_bundle,
    validate_research_profile_bundle,
    validate_profile_payload,
)

from .models import (
    BundleRevision,
    ConfigScope,
    EffectiveRuntimeConfigV2,
    ProfileRevision,
    ResolutionContext,
    RevisionStatus,
    digest,
)
from .schedule import ConfigurationIntegrityError, parse_instant, verify_revision_schedule

_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_PRECEDENCE_KEYS = {
    "request_override",
    "shared_game_published",
    "room_published",
    "user_experiment_published",
    "video_profile_published",
    "tenant_published",
    "environment_published",
    "global_published",
    "file_bundle_fallback",
}


class VersionedConfigurationService:
    """Validated, immutable, multi-scope V20 configuration control plane."""

    def __init__(self, store: Any, *, root: str | Path, profile: dict[str, Any]):
        self.store = store
        self.root = Path(root).resolve()
        self.profile = copy.deepcopy(profile)
        governance = dict(self.profile.get("governance") or {})
        resolution = dict(self.profile.get("resolution") or {})
        required = {
            "governance.maximum_payload_bytes": governance.get("maximum_payload_bytes"),
            "resolution.cache_entries": resolution.get("cache_entries"),
            "resolution.cache_ttl_seconds": resolution.get("cache_ttl_seconds"),
            "resolution.precedence": resolution.get("precedence"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "configuration control-plane profile is missing governed values: "
                + ", ".join(sorted(missing))
            )
        self.maximum_payload_bytes = int(governance["maximum_payload_bytes"])
        self._cache_entries = int(resolution["cache_entries"])
        self._cache_ttl_seconds = float(resolution["cache_ttl_seconds"])
        self._precedence = tuple(str(value) for value in resolution["precedence"])
        if not 4_096 <= self.maximum_payload_bytes <= 16_777_216:
            raise ValueError(
                "governance.maximum_payload_bytes is outside the safe range"
            )
        if not 0 <= self._cache_entries <= 10_000:
            raise ValueError("resolution.cache_entries is outside the safe range")
        if not 0 <= self._cache_ttl_seconds <= 3_600:
            raise ValueError("resolution.cache_ttl_seconds is outside the safe range")
        unknown_precedence = set(self._precedence) - _PRECEDENCE_KEYS
        if (
            not self._precedence
            or len(set(self._precedence)) != len(self._precedence)
            or unknown_precedence
        ):
            raise ValueError(
                "resolution.precedence must contain unique supported values"
            )
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, dict[str, Any]]] = (
            OrderedDict()
        )
        self._cache_lock = threading.RLock()
        self._seen_epoch = -1
        self._bundle_compiler: Callable[[Mapping[str, Any]], Any] | None = None

    def set_bundle_compiler(self, compiler: Callable[[Mapping[str, Any]], Any]) -> None:
        """Attach the compiler used for draft previews and publication gates."""

        self._bundle_compiler = compiler

    def _invalidate_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
        self._seen_epoch = (
            self.store.get_invalidation_epoch()
            if hasattr(self.store, "get_invalidation_epoch")
            else self._seen_epoch + 1
        )

    def _epoch(self) -> int:
        epoch = (
            self.store.get_invalidation_epoch()
            if hasattr(self.store, "get_invalidation_epoch")
            else 0
        )
        if epoch != self._seen_epoch:
            with self._cache_lock:
                self._cache.clear()
            self._seen_epoch = epoch
        return epoch

    def _cache_get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        if self._cache_entries <= 0 or self._cache_ttl_seconds <= 0:
            return None
        with self._cache_lock:
            value = self._cache.get(key)
            if value is None:
                return None
            created, payload = value
            if time.monotonic() - created > self._cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(payload)

    def _cache_put(self, key: tuple[Any, ...], payload: dict[str, Any]) -> None:
        if self._cache_entries <= 0 or self._cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), copy.deepcopy(payload))
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)

    @staticmethod
    def _profile_identity(payload: dict[str, Any]) -> tuple[str, str]:
        meta = dict(payload.get("profile") or {})
        profile_id = str(meta.get("id") or "").strip()
        version = str(meta.get("version") or "").strip()
        if not profile_id:
            raise ValueError("configuration profile payload requires profile.id")
        if not _SEMVER.match(version):
            raise ValueError("configuration profile version must be semantic")
        return profile_id, version

    @staticmethod
    def _scope(
        scope: str,
        *,
        scope_key: str | None,
        tenant_id: str | None,
        environment: str | None,
    ) -> tuple[str, str, str | None, str | None]:
        normalized_tenant = None if tenant_id is None else str(tenant_id).strip()
        normalized_environment = (
            None if environment is None else str(environment).strip()
        )
        if tenant_id is not None and not normalized_tenant:
            raise ValueError("tenant_id must be nonblank when provided")
        if environment is not None and not normalized_environment:
            raise ValueError("environment must be nonblank when provided")
        value = ConfigScope(scope).value
        if value == ConfigScope.GLOBAL.value:
            if normalized_tenant is not None:
                raise ValueError("global scope does not accept tenant_id")
            if scope_key is not None and str(scope_key).strip() != "*":
                raise ValueError("global scope_key must be '*'")
            return value, "*", None, normalized_environment
        if value == ConfigScope.ENVIRONMENT.value:
            if normalized_tenant is not None:
                raise ValueError("environment scope does not accept tenant_id")
            key = str(scope_key or normalized_environment or "").strip()
            if not key:
                raise ValueError("environment scope requires environment or scope_key")
            if normalized_environment is not None and normalized_environment != key:
                raise ValueError("environment and scope_key must match")
            return value, key, None, key
        if value == ConfigScope.TENANT.value:
            key = str(scope_key or normalized_tenant or "").strip()
            if not key:
                raise ValueError("tenant scope requires tenant_id or scope_key")
            if normalized_tenant is not None and normalized_tenant != key:
                raise ValueError("tenant_id and scope_key must match")
            return value, key, key, normalized_environment
        key = str(scope_key or "").strip()
        if not key:
            raise ValueError(f"{value} scope requires scope_key")
        return value, key, normalized_tenant, normalized_environment

    @staticmethod
    def _resolution_context(
        *,
        tenant_id: str | None,
        environment: str | None,
        user_experiment_key: str | None,
        room_id: str | None,
        shared_game_id: str | None,
        video_profile_id: str | None,
        subject_id: str | None,
        at_time: str | None,
    ) -> ResolutionContext:
        def optional(name: str, value: str | None) -> str | None:
            if value is None:
                return None
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"{name} must be nonblank when provided")
            return normalized

        instant = parse_instant(at_time, "atTime")
        return ResolutionContext(
            optional("tenant_id", tenant_id),
            optional("environment", environment),
            optional("user_experiment_key", user_experiment_key),
            optional("room_id", room_id),
            optional("shared_game_id", shared_game_id),
            optional("video_profile_id", video_profile_id),
            optional("subject_id", subject_id),
            instant.isoformat() if instant is not None else None,
        )

    def validate_profile(self, family: str, payload: dict[str, Any]) -> dict[str, Any]:
        if len(str(payload).encode("utf-8")) > self.maximum_payload_bytes:
            raise ValueError("configuration profile payload exceeds maximum size")
        return validate_profile_payload(family, payload, root=self.root)

    def profile_schema(self, family: str) -> dict[str, Any]:
        """Return the same closed schema used by row validation."""

        family = str(family)
        if family not in PROFILE_FAMILIES:
            raise ValueError(f"unknown profile family: {family}")
        candidate = self.root / "schemas" / "profiles" / f"{family}.schema.json"
        schema_path = (
            candidate
            if candidate.exists()
            else self.root / "schemas" / "profile.schema.json"
        )
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def create_profile(
        self,
        *,
        family: str,
        payload: dict[str, Any],
        created_by: str,
        scope: str = ConfigScope.GLOBAL.value,
        tenant_id: str | None = None,
        scope_key: str | None = None,
        environment: str | None = None,
        effective_from: str | None = None,
        effective_until: str | None = None,
        local_schedule: Mapping[str, str] | None = None,
        rollout_percentage: float = 100.0,
        rollout_seed: str = "default",
        supersedes_revision_id: str | None = None,
        publish: bool = False,
        require_matching_schedule: bool = True,
    ) -> dict[str, Any]:
        validated = self.validate_profile(str(family), payload)
        profile_id, version = self._profile_identity(validated)
        scope, key, tenant_id, environment = self._scope(
            scope, scope_key=scope_key, tenant_id=tenant_id, environment=environment
        )
        percentage = float(rollout_percentage)
        if not 0 < percentage <= 100:
            raise ValueError(
                "rollout_percentage must be greater than 0 and at most 100"
            )
        revision = self.store.create_profile(
            family=str(family),
            profile_id=profile_id,
            version=version,
            scope=scope,
            scope_key=key,
            tenant_id=tenant_id,
            environment=environment,
            payload=validated,
            status=RevisionStatus.PUBLISHED.value
            if publish
            else RevisionStatus.DRAFT.value,
            created_by=created_by,
            effective_from=effective_from,
            effective_until=effective_until,
            local_schedule=local_schedule,
            rollout_percentage=percentage,
            rollout_seed=str(rollout_seed or "default"),
            supersedes_revision_id=supersedes_revision_id,
            require_matching_schedule=require_matching_schedule,
        )
        self._invalidate_cache()
        return revision.to_dict()

    def update_profile(
        self,
        revision_id: str,
        *,
        payload: dict[str, Any],
        expected_etag: int,
        actor_id: str,
    ) -> dict[str, Any]:
        current = self.store.get_profile(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if current.status != RevisionStatus.DRAFT.value:
            raise ValueError("only draft profile revisions can be edited")
        validated = self.validate_profile(current.family, payload)
        profile_id, version = self._profile_identity(validated)
        result = self.store.update_profile_draft(
            revision_id,
            expected_etag=expected_etag,
            profile_id=profile_id,
            version=version,
            payload=validated,
            actor_id=actor_id,
        )
        self._invalidate_cache()
        return result.to_dict()

    def validate_profile_revision(self, revision_id: str) -> dict[str, Any]:
        current = self.store.get_profile(revision_id)
        if current is None:
            raise KeyError(revision_id)
        validated = self.validate_profile(current.family, dict(current.payload))
        profile_id, version = self._profile_identity(validated)
        return {
            "revisionId": revision_id,
            "family": current.family,
            "profileId": profile_id,
            "version": version,
            "payloadSha256": digest(validated),
            "etag": current.etag,
            "status": current.status,
            "valid": True,
            "automaticProductionApply": False,
        }

    def publish_profile(
        self, revision_id: str, *, expected_etag: int, actor_id: str
    ) -> dict[str, Any]:
        current = self.store.get_profile(revision_id)
        if current is None:
            raise KeyError(revision_id)
        self.validate_profile(current.family, dict(current.payload))
        result = self.store.publish_profile(
            revision_id,
            expected_etag=expected_etag,
            actor_id=actor_id,
        )
        self._invalidate_cache()
        return result.to_dict()

    def reject_profile(
        self, revision_id: str, *, expected_etag: int, actor_id: str, reason: str
    ) -> dict[str, Any]:
        current = self.store.get_profile(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if current.status != RevisionStatus.DRAFT.value:
            raise ValueError("only draft profile revisions can be rejected")
        result = self.store.set_profile_status(
            revision_id,
            expected_etag=expected_etag,
            status=RevisionStatus.REJECTED.value,
            actor_id=actor_id,
        )
        self.store.append_event(
            "profile",
            revision_id,
            "rejection_reason",
            actor_id,
            {"reason": str(reason)[:2000]},
        )
        self._invalidate_cache()
        return result.to_dict()

    def retire_profile(
        self, revision_id: str, *, expected_etag: int, actor_id: str, reason: str
    ) -> dict[str, Any]:
        current = self.store.get_profile(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if current.status != RevisionStatus.PUBLISHED.value:
            raise ValueError("only published profile revisions can be retired")
        result = self.store.set_profile_status(
            revision_id,
            expected_etag=expected_etag,
            status=RevisionStatus.RETIRED.value,
            actor_id=actor_id,
        )
        self.store.append_event(
            "profile",
            revision_id,
            "retirement_reason",
            actor_id,
            {"reason": str(reason)[:2000]},
        )
        self._invalidate_cache()
        return result.to_dict()

    @staticmethod
    def _inside_window(revision: BundleRevision | ProfileRevision, at_time: datetime) -> bool:
        start = parse_instant(revision.effective_from, f"{revision.revision_id}.effectiveFrom")
        end = parse_instant(revision.effective_until, f"{revision.revision_id}.effectiveUntil")
        return (start is None or start <= at_time) and (end is None or at_time < end)

    @staticmethod
    def _in_rollout(revision: BundleRevision | ProfileRevision, context: ResolutionContext) -> bool:
        percentage = float(revision.rollout_percentage)
        if percentage >= 100:
            return True
        if not context.subject_id:
            return False
        raw = f"{revision.rollout_seed}|{revision.revision_id}|{context.subject_id}".encode(
            "utf-8"
        )
        bucket = (
            int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 10_000 / 100.0
        )
        return bucket < percentage

    def _scheduled_bundle_history(self, anchor: BundleRevision) -> Iterator[BundleRevision]:
        dimensions = (anchor.bundle_id, anchor.scope, anchor.scope_key, anchor.tenant_id, anchor.environment)
        seen: set[str] = set()
        revision_id = anchor.scheduled_successor_id or anchor.revision_id
        while revision_id != anchor.revision_id:
            revision = self.store.get_bundle(revision_id)
            if (
                revision_id in seen
                or revision is None
                or revision.status not in {RevisionStatus.PUBLISHED.value, RevisionStatus.SUPERSEDED.value}
                or revision.scheduled_predecessor_id is None
                or (revision.bundle_id, revision.scope, revision.scope_key,
                    revision.tenant_id, revision.environment) != dimensions
            ):
                raise ConfigurationIntegrityError(
                    anchor.revision_id,
                    "The scheduled publication could not be verified. Publish a current replacement or roll back.",
                )
            seen.add(revision_id)
            yield revision
            revision_id = revision.scheduled_predecessor_id
        yield anchor

    def _effective_scheduled_bundle(
        self, head: BundleRevision, context: ResolutionContext, now: datetime,
        trace: list[dict[str, Any]],
    ) -> BundleRevision | None:
        for revision in self._scheduled_bundle_history(head):
            verify_revision_schedule(revision)
            in_window = self._inside_window(revision, now)
            eligible = in_window and self._in_rollout(revision, context)
            trace.append({
                "scope": revision.scope, "scopeKey": revision.scope_key,
                "revisionId": revision.revision_id, "eligible": eligible,
            })
            if in_window:
                return revision if eligible else None
        return None

    def _scope_candidates(
        self, context: ResolutionContext
    ) -> list[tuple[str, str, str | None, str | None]]:
        values: list[tuple[str, str, str | None, str | None]] = []
        for item in self._precedence:
            if item in {"request_override", "file_bundle_fallback"}:
                continue
            if item == "shared_game_published" and context.shared_game_id:
                values.append(
                    (
                        ConfigScope.SHARED_GAME.value,
                        context.shared_game_id,
                        context.tenant_id,
                        context.environment,
                    )
                )
            elif item == "room_published" and context.room_id:
                values.append(
                    (
                        ConfigScope.ROOM.value,
                        context.room_id,
                        context.tenant_id,
                        context.environment,
                    )
                )
            elif item == "user_experiment_published" and context.user_experiment_key:
                values.append(
                    (
                        ConfigScope.USER_EXPERIMENT.value,
                        context.user_experiment_key,
                        context.tenant_id,
                        context.environment,
                    )
                )
            elif item == "video_profile_published" and context.video_profile_id:
                values.append(
                    (
                        ConfigScope.VIDEO_PROFILE.value,
                        context.video_profile_id,
                        context.tenant_id,
                        context.environment,
                    )
                )
            elif item == "tenant_published" and context.tenant_id:
                values.append(
                    (
                        ConfigScope.TENANT.value,
                        context.tenant_id,
                        context.tenant_id,
                        context.environment,
                    )
                )
            elif item == "environment_published" and context.environment:
                values.append(
                    (
                        ConfigScope.ENVIRONMENT.value,
                        context.environment,
                        None,
                        context.environment,
                    )
                )
            elif item == "global_published":
                values.append(
                    (ConfigScope.GLOBAL.value, "*", None, context.environment)
                )
        return values

    def resolve_profile(
        self,
        *,
        family: str,
        profile_id: str,
        tenant_id: str | None = None,
        environment: str | None = None,
        room_id: str | None = None,
        shared_game_id: str | None = None,
        video_profile_id: str | None = None,
        user_experiment_key: str | None = None,
        subject_id: str | None = None,
        at_time: str | None = None,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        context = self._resolution_context(
            tenant_id=tenant_id,
            environment=environment,
            user_experiment_key=user_experiment_key,
            room_id=room_id,
            shared_game_id=shared_game_id,
            video_profile_id=video_profile_id,
            subject_id=subject_id,
            at_time=at_time,
        )
        epoch = self._epoch()
        key = (
            "profile",
            family,
            profile_id,
            revision_id or "",
            epoch,
            *context.cache_key(),
        )
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if revision_id:
            revision = self.store.get_profile(revision_id)
            if (
                revision is None
                or revision.family != family
                or revision.profile_id != profile_id
            ):
                raise KeyError(revision_id)
            if revision.status != RevisionStatus.PUBLISHED.value:
                raise PermissionError(
                    "runtime resolution requires a published revision"
                )
            verify_revision_schedule(revision)
            result = revision.to_dict()
            result["resolutionTrace"] = [
                {"scope": "exact_revision", "revisionId": revision_id}
            ]
            self._cache_put(key, result)
            return result
        trace = []
        now = parse_instant(context.at_time, "atTime") or datetime.now(timezone.utc)
        for (
            scope,
            scope_key,
            scoped_tenant,
            scoped_environment,
        ) in self._scope_candidates(context):
            rows = self.store.list_profiles(
                family=family,
                profile_id=profile_id,
                scope=scope,
                scope_key=scope_key,
                tenant_id=scoped_tenant,
                environment=scoped_environment,
                status=RevisionStatus.PUBLISHED.value,
                limit=100,
                exact_dimensions=True,
            )
            for revision in rows:
                verify_revision_schedule(revision)
                eligible = self._inside_window(revision, now) and self._in_rollout(revision, context)
                trace.append(
                    {
                        "scope": scope,
                        "scopeKey": scope_key,
                        "revisionId": revision.revision_id,
                        "eligible": eligible,
                    }
                )
                if eligible:
                    result = revision.to_dict()
                    result["resolutionTrace"] = trace
                    result["invalidationEpoch"] = epoch
                    self._cache_put(key, result)
                    return result
        raise KeyError(
            f"published configuration profile not found: {family}/{profile_id}"
        )

    @staticmethod
    def compare_payloads(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
        if isinstance(left, dict) and isinstance(right, dict):
            result = []
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else str(key)
                if key not in left:
                    result.append(
                        {"path": child, "change": "added", "right": right[key]}
                    )
                elif key not in right:
                    result.append(
                        {"path": child, "change": "removed", "left": left[key]}
                    )
                else:
                    result.extend(
                        VersionedConfigurationService.compare_payloads(
                            left[key], right[key], child
                        )
                    )
            return result
        if isinstance(left, list) and isinstance(right, list):
            return (
                []
                if left == right
                else [{"path": path, "change": "changed", "left": left, "right": right}]
            )
        return (
            []
            if left == right
            else [{"path": path, "change": "changed", "left": left, "right": right}]
        )

    def create_bundle(
        self,
        *,
        bundle_id: str,
        version: str,
        mode: str,
        component_revisions: dict[str, str],
        created_by: str,
        scope: str = ConfigScope.GLOBAL.value,
        tenant_id: str | None = None,
        scope_key: str | None = None,
        environment: str | None = None,
        effective_from: str | None = None,
        effective_until: str | None = None,
        local_schedule: Mapping[str, str] | None = None,
        rollout_percentage: float = 100.0,
        rollout_seed: str = "default",
        supersedes_revision_id: str | None = None,
        publish: bool = False,
        allow_draft_components: bool = False,
        allow_historical_components: bool = False,
        force_new_revision: bool = False,
        rollback_target_revision_id: str | None = None,
        require_matching_schedule: bool = True,
    ) -> dict[str, Any]:
        if not _SEMVER.match(str(version)):
            raise ValueError("configuration bundle version must be semantic")
        missing = set(PROFILE_FAMILIES) - set(component_revisions)
        extra = set(component_revisions) - set(PROFILE_FAMILIES)
        if missing or extra:
            raise ValueError(
                f"configuration bundle components mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        allowed_component_statuses = {RevisionStatus.PUBLISHED.value}
        if allow_draft_components and not publish:
            allowed_component_statuses.add(RevisionStatus.DRAFT.value)
        if allow_historical_components:
            allowed_component_statuses.add(RevisionStatus.SUPERSEDED.value)
        component_payloads: dict[str, dict[str, Any]] = {}
        for family, rid in component_revisions.items():
            rev = self.store.get_profile(rid)
            if rev is None or rev.family != family:
                raise ValueError(f"invalid component revision for {family}: {rid}")
            if rev.status not in allowed_component_statuses:
                raise ValueError(f"component revision is not eligible: {rid}")
            component_payloads[family] = dict(rev.payload)
        validate_research_profile_bundle(component_payloads)
        scope, key, tenant_id, environment = self._scope(
            scope, scope_key=scope_key, tenant_id=tenant_id, environment=environment
        )
        result = self.store.create_bundle(
            bundle_id=bundle_id,
            version=version,
            mode=mode,
            scope=scope,
            scope_key=key,
            tenant_id=tenant_id,
            environment=environment,
            component_revisions=component_revisions,
            status=RevisionStatus.PUBLISHED.value
            if publish
            else RevisionStatus.DRAFT.value,
            created_by=created_by,
            effective_from=effective_from,
            effective_until=effective_until,
            local_schedule=local_schedule,
            rollout_percentage=float(rollout_percentage),
            rollout_seed=str(rollout_seed or "default"),
            supersedes_revision_id=supersedes_revision_id,
            force_new_revision=force_new_revision,
            rollback_target_revision_id=rollback_target_revision_id,
            require_matching_schedule=require_matching_schedule,
            allowed_component_statuses=tuple(sorted(allowed_component_statuses)),
        )
        self._invalidate_cache()
        return result.to_dict()

    def compile_bundle(self, revision_id: str) -> dict[str, Any]:
        """Compile an exact bundle revision without applying it to runtime."""

        current = self.store.get_bundle(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if self._bundle_compiler is None:
            raise RuntimeError("configuration domain compiler is unavailable")
        resolved = self.resolve(revision_id=revision_id, allow_draft=True)
        validate_research_profile_bundle(dict(resolved["profiles"]))
        compiled = self._bundle_compiler(resolved)
        if hasattr(compiled, "to_dict"):
            value = compiled.to_dict(include_payloads=False)
        elif isinstance(compiled, Mapping):
            value = copy.deepcopy(dict(compiled))
        else:
            raise TypeError(
                "configuration domain compiler returned an unsupported result"
            )
        compiled_domain = dict(value.get("compiledDomain") or value)
        coverage = dict(compiled_domain.get("coverage") or {})
        publishable = bool(compiled_domain.get("publishable", False))
        return {
            "bundle": current.to_dict(),
            "domainSpecHash": value.get("sha256"),
            "compilationHash": compiled_domain.get("sha256"),
            "coverage": coverage,
            "publishable": publishable,
            "compilerVersion": compiled_domain.get("compilerVersion"),
            "automaticProductionApply": False,
            "previewOnly": current.status != RevisionStatus.PUBLISHED.value,
        }

    def publish_bundle(
        self, revision_id: str, *, expected_etag: int, actor_id: str
    ) -> dict[str, Any]:
        current = self.store.get_bundle(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if current.etag != int(expected_etag):
            raise RuntimeError("configuration revision etag conflict")
        if current.status != RevisionStatus.DRAFT.value:
            raise ValueError("only draft bundle revisions can be published")
        for family, rid in current.component_revisions.items():
            profile = self.store.get_profile(rid)
            if (
                profile is None
                or profile.family != family
                or profile.status != RevisionStatus.PUBLISHED.value
            ):
                raise ValueError(
                    f"bundle component is not a published {family} revision"
                )
        compilation = self.compile_bundle(revision_id)
        if not compilation["publishable"]:
            coverage = compilation.get("coverage") or {}
            raise ValueError(
                "domain compiler qualification coverage is incomplete "
                f"({coverage.get('covered', 0)}/{coverage.get('total', 0)})"
            )
        result = self.store.publish_bundle(
            revision_id,
            expected_etag=expected_etag,
            actor_id=actor_id,
        )
        self._invalidate_cache()
        return result.to_dict()

    def reject_bundle(
        self, revision_id: str, *, expected_etag: int, actor_id: str, reason: str
    ) -> dict[str, Any]:
        current = self.store.get_bundle(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if current.status != RevisionStatus.DRAFT.value:
            raise ValueError("only draft bundle revisions can be rejected")
        result = self.store.set_bundle_status(
            revision_id,
            expected_etag=expected_etag,
            status=RevisionStatus.REJECTED.value,
            actor_id=actor_id,
        )
        self.store.append_event(
            "bundle",
            revision_id,
            "rejection_reason",
            actor_id,
            {"reason": str(reason)[:2000]},
        )
        self._invalidate_cache()
        return result.to_dict()

    def retire_bundle(
        self, revision_id: str, *, expected_etag: int, actor_id: str, reason: str
    ) -> dict[str, Any]:
        current = self.store.get_bundle(revision_id)
        if current is None:
            raise KeyError(revision_id)
        if current.status != RevisionStatus.PUBLISHED.value:
            raise ValueError("only published bundle revisions can be retired")
        result = self.store.set_bundle_status(
            revision_id,
            expected_etag=expected_etag,
            status=RevisionStatus.RETIRED.value,
            actor_id=actor_id,
        )
        self.store.append_event(
            "bundle",
            revision_id,
            "retirement_reason",
            actor_id,
            {"reason": str(reason)[:2000]},
        )
        self._invalidate_cache()
        return result.to_dict()

    def rollback_bundle(
        self,
        target_revision_id: str,
        *,
        new_version: str,
        actor_id: str,
    ) -> dict[str, Any]:
        target = self.store.get_bundle(target_revision_id)
        if target is None:
            raise KeyError(target_revision_id)
        if target.status not in {
            RevisionStatus.PUBLISHED.value,
            RevisionStatus.SUPERSEDED.value,
        }:
            raise ValueError(
                "configuration rollback target must be published or superseded"
            )
        return self.create_bundle(
            bundle_id=target.bundle_id,
            version=new_version,
            mode=target.mode,
            component_revisions=dict(target.component_revisions),
            created_by=actor_id,
            scope=target.scope,
            scope_key=target.scope_key,
            tenant_id=target.tenant_id,
            environment=target.environment,
            publish=True,
            allow_historical_components=True,
            rollback_target_revision_id=target_revision_id,
        )

    def resolve(
        self,
        *,
        revision_id: str | None = None,
        bundle_id: str | None = None,
        tenant_id: str | None = None,
        environment: str | None = None,
        room_id: str | None = None,
        shared_game_id: str | None = None,
        video_profile_id: str | None = None,
        user_experiment_key: str | None = None,
        subject_id: str | None = None,
        at_time: str | None = None,
        allow_draft: bool = False,
        allow_retired: bool = False,
        allow_scheduled_predecessor: bool = False,
    ) -> dict[str, Any]:
        if allow_retired and revision_id is None:
            raise ValueError("retired configuration requires an exact bundle revision")
        context = self._resolution_context(
            tenant_id=tenant_id,
            environment=environment,
            user_experiment_key=user_experiment_key,
            room_id=room_id,
            shared_game_id=shared_game_id,
            video_profile_id=video_profile_id,
            subject_id=subject_id,
            at_time=at_time,
        )
        epoch = self._epoch()
        key = (
            "bundle",
            revision_id or "",
            bundle_id or "",
            bool(allow_draft),
            bool(allow_retired),
            bool(allow_scheduled_predecessor),
            epoch,
            *context.cache_key(),
        )
        cacheable = not allow_draft
        if cacheable:
            cached = self._cache_get(key)
            if cached is not None:
                return cached
        bundle = self.store.get_bundle(revision_id) if revision_id else None
        if revision_id is not None and bundle is None:
            raise KeyError(revision_id)
        if (
            bundle is not None
            and bundle_id is not None
            and bundle.bundle_id != bundle_id
        ):
            raise KeyError(revision_id or bundle_id)
        trace = []
        selected_predecessor = False
        if bundle is None:
            now = parse_instant(context.at_time, "atTime") or datetime.now(timezone.utc)
            for (
                scope,
                scope_key,
                scoped_tenant,
                scoped_environment,
            ) in self._scope_candidates(context):
                rows = self.store.list_bundles(
                    bundle_id=bundle_id,
                    scope=scope,
                    scope_key=scope_key,
                    tenant_id=scoped_tenant,
                    environment=scoped_environment,
                    status=None if allow_draft else RevisionStatus.PUBLISHED.value,
                    limit=100,
                    exact_dimensions=True,
                )
                for head in rows:
                    if head.scheduled_predecessor_id is not None:
                        continue
                    # Clock changes do not bump the registry epoch.
                    if context.at_time is None and (head.effective_from or head.effective_until or head.scheduled_successor_id):
                        cacheable = False
                    bundle = self._effective_scheduled_bundle(head, context, now, trace)
                    if bundle is not None:
                        selected_predecessor = bundle.revision_id != head.revision_id
                        break
                if bundle is not None:
                    break
        if bundle is None:
            raise KeyError("published configuration bundle not found")
        if (
            allow_scheduled_predecessor and revision_id
            and bundle.status == RevisionStatus.SUPERSEDED.value
        ):
            cacheable = False
            heads = self.store.list_bundles(
                bundle_id=bundle.bundle_id, scope=bundle.scope, scope_key=bundle.scope_key,
                tenant_id=bundle.tenant_id, environment=bundle.environment,
                status=RevisionStatus.PUBLISHED.value, limit=1, exact_dimensions=True,
            )
            if heads and heads[0].scheduled_successor_id is not None:
                now = parse_instant(context.at_time, "atTime") or datetime.now(timezone.utc)
                selected = self._effective_scheduled_bundle(heads[0], context, now, trace)
                selected_predecessor = selected is not None and selected.revision_id == revision_id
        exact_historical = (
            revision_id is not None and (
                bundle.status == RevisionStatus.SUPERSEDED.value
                or allow_retired and bundle.status == RevisionStatus.RETIRED.value
            )
        )
        if (
            bundle.status != RevisionStatus.PUBLISHED.value
            and not exact_historical
            and not selected_predecessor
            and not allow_draft
        ):
            raise PermissionError(
                "unpublished configuration bundle requires explicit preview access"
            )
        if revision_id is not None:
            verify_revision_schedule(bundle)
        result = self._bundle_snapshot(
            bundle, trace=trace, epoch=epoch, allow_draft=allow_draft, allow_retired=allow_retired,
            runtime=revision_id is None, effective=selected_predecessor,
        )
        if cacheable:
            self._cache_put(key, result)
        return result

    def _bundle_snapshot(
        self, bundle: BundleRevision, *, trace: list[dict[str, Any]], epoch: int, runtime: bool,
        allow_draft: bool = False, allow_retired: bool = False, effective: bool = False,
    ) -> dict[str, Any]:
        profiles = {}
        identities = {}
        for family, rid in bundle.component_revisions.items():
            revision = self.store.get_profile(rid)
            if revision is None:
                if runtime:
                    raise ConfigurationIntegrityError(rid, "The effective bundle references a missing component. Publish a replacement or roll back.")
                raise RuntimeError(
                    f"configuration bundle references missing revision {rid}"
                )
            verify_revision_schedule(revision)
            if (
                revision.status
                not in {
                    RevisionStatus.PUBLISHED.value,
                    RevisionStatus.SUPERSEDED.value,
                }
                and not (allow_retired and revision.status == RevisionStatus.RETIRED.value)
                and not allow_draft
            ):
                if runtime:
                    raise ConfigurationIntegrityError(rid, "The effective bundle references an unavailable component. Publish a replacement or roll back.")
                raise PermissionError(
                    f"configuration bundle references unavailable revision {rid}"
                )
            profiles[family] = copy.deepcopy(dict(revision.payload))
            identities[family] = revision.to_dict(include_payload=False)
        bundle_document = bundle.to_dict()
        if effective:
            bundle_document["status"] = RevisionStatus.PUBLISHED.value
        effective_hash = digest(
            {
                "bundleAuthority": {
                    key: bundle_document[key]
                    for key in (
                        "revisionId",
                        "bundleId",
                        "version",
                        "mode",
                        "scope",
                        "scopeKey",
                        "tenantId",
                        "environment",
                        "componentRevisions",
                        "contentSha256",
                        "effectiveFrom",
                        "effectiveUntil",
                        "rolloutPercentage",
                        "rolloutSeed",
                    )
                },
                "profileHashes": {
                    family: value["payloadSha256"]
                    for family, value in identities.items()
                },
            }
        )
        return {
            "bundle": bundle_document,
            "profiles": profiles,
            "profileRevisions": identities,
            "effectiveHash": effective_hash,
            "audit": self.store.verify_events("bundle", bundle.revision_id),
            "resolutionTrace": trace
            or [{"scope": "exact_revision", "revisionId": bundle.revision_id}],
            "invalidationEpoch": epoch,
        }

    def freeze(self, **context: Any) -> dict[str, Any]:
        resolved = self.resolve(**context)
        return {
            "schemaVersion": 1,
            "bundleRevisionId": resolved["bundle"]["revisionId"],
            "componentRevisionIds": {
                k: v["revisionId"] for k, v in resolved["profileRevisions"].items()
            },
            "effectiveHash": resolved["effectiveHash"],
            "resolutionTrace": resolved["resolutionTrace"],
        }

    def resolve_effective_v2(
        self,
        *,
        plugin_identities: tuple[dict[str, Any], ...] = (),
        model_identities: tuple[dict[str, Any], ...] = (),
        **context: Any,
    ) -> EffectiveRuntimeConfigV2:
        return EffectiveRuntimeConfigV2.from_resolved(
            self.resolve(**context),
            plugin_identities=plugin_identities,
            model_identities=model_identities,
        )

    def freeze_v2(
        self,
        *,
        plugin_identities: tuple[dict[str, Any], ...] = (),
        model_identities: tuple[dict[str, Any], ...] = (),
        **context: Any,
    ) -> dict[str, Any]:
        return self.resolve_effective_v2(
            plugin_identities=plugin_identities,
            model_identities=model_identities,
            **context,
        ).to_dict()

    def bootstrap_packaged_bundle(
        self,
        *,
        actor_id: str = "SYSTEM-PACKAGED-BOOTSTRAP",
        mode: str = "full",
        bundle_path: str | Path | None = None,
        scope: str = ConfigScope.GLOBAL.value,
        scope_key: str | None = None,
        tenant_id: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        packaged = build_bundle(self.root, mode=mode, bundle_path=bundle_path)
        scope, scope_key, tenant_id, environment = self._scope(
            scope,
            scope_key=scope_key,
            tenant_id=tenant_id,
            environment=environment,
        )
        revision_ids = {}
        for family in PROFILE_FAMILIES:
            if family == "rules":
                payload = copy.deepcopy(packaged.rules.data)
            else:
                payload = copy.deepcopy(packaged.components[family].data)
            payload_hash = digest(payload)
            existing = self.store.find_profile_by_content(
                family=family,
                scope=scope,
                tenant_id=tenant_id,
                scope_key=scope_key,
                environment=environment,
                payload_sha256=payload_hash,
            )
            if existing is None:
                value = self.create_profile(
                    family=family,
                    payload=payload,
                    created_by=actor_id,
                    scope=scope,
                    scope_key=scope_key,
                    tenant_id=tenant_id,
                    environment=environment,
                    publish=True,
                )
                revision_ids[family] = value["revisionId"]
            else:
                if existing.status == RevisionStatus.DRAFT.value:
                    existing = self.store.publish_profile(
                        existing.revision_id,
                        expected_etag=existing.etag,
                        actor_id=actor_id,
                    )
                revision_ids[family] = existing.revision_id
        profile = dict(packaged.bundle_profile.get("profile") or {})
        bundle_id = str(profile.get("id") or "packaged_bundle")
        version = str(profile.get("version") or "0.0.0")
        content_hash = digest({"mode": mode, "components": revision_ids})
        existing_bundle = self.store.find_bundle_by_content(
            bundle_id=bundle_id,
            scope=scope,
            tenant_id=tenant_id,
            scope_key=scope_key,
            environment=environment,
            content_sha256=content_hash,
        )
        if existing_bundle is None:
            value = self.create_bundle(
                bundle_id=bundle_id,
                version=version,
                mode=mode,
                component_revisions=revision_ids,
                created_by=actor_id,
                scope=scope,
                scope_key=scope_key,
                tenant_id=tenant_id,
                environment=environment,
                publish=True,
                allow_historical_components=True,
            )
            rid = value["revisionId"]
        else:
            if existing_bundle.status == RevisionStatus.DRAFT.value:
                existing_bundle = self.store.publish_bundle(
                    existing_bundle.revision_id,
                    expected_etag=existing_bundle.etag,
                    actor_id=actor_id,
                )
            rid = existing_bundle.revision_id
        rows = self.store.list_bundles(
            bundle_id=bundle_id, scope=scope, scope_key=scope_key, tenant_id=tenant_id,
            environment=environment, status=RevisionStatus.PUBLISHED.value, limit=1, exact_dimensions=True,
        )
        anchor = rows[0] if rows else None
        selected = anchor
        if anchor is not None:
            if anchor.scheduled_successor_id or anchor.effective_from or anchor.effective_until:
                context = ResolutionContext(tenant_id=tenant_id, environment=environment)
                selected = self._effective_scheduled_bundle(anchor, context, datetime.now(timezone.utc), [])
            else:
                verify_revision_schedule(anchor)
        if selected is None:
            raise ConfigurationIntegrityError(
                anchor.revision_id if anchor is not None else rid,
                "No configuration bundle is effective. Publish a current replacement or roll back.",
            )
        return self._bundle_snapshot(selected, trace=[], epoch=self._epoch(), runtime=True, effective=True)
