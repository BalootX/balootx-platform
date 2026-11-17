from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Mapping

from .base import PluginDescriptor
from .contracts import (
    PLUGIN_CONTRACTS_V1,
    PluginCancelledError,
    PluginExecutionError,
    validate_plugin_contract,
)
from .host import PluginHostPolicy, SubprocessPluginHost, WasmtimeComponentHost


MAXIMUM_PLUGIN_ROLLBACK_DEPTH = 16


def _major(value: str) -> int:
    match = re.match(r"^\s*(\d+)", str(value))
    if not match:
        raise ValueError(f"invalid API version: {value}")
    return int(match.group(1))


def _range_allows(expression: str, version: str) -> bool:
    current = _major(version)
    for term in [item.strip() for item in str(expression or "").split(",") if item.strip()]:
        match = re.match(r"(>=|<=|>|<|==|=)\s*(\d+)", term)
        if not match:
            raise ValueError(f"unsupported compatibility range: {expression}")
        operator, raw = match.groups()
        expected = int(raw)
        if operator in {"=", "=="} and current != expected:
            return False
        if operator == ">=" and current < expected:
            return False
        if operator == "<=" and current > expected:
            return False
        if operator == ">" and current <= expected:
            return False
        if operator == "<" and current >= expected:
            return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class PluginPolicy:
    host_api_version: str
    profile_schema_version: str
    allowed_ids: frozenset[str] | None
    allowed_isolation: frozenset[str]
    allowed_permissions: frozenset[str]
    require_signed_external: bool
    require_content_addressed_external: bool
    maximum_timeout_seconds: int
    maximum_response_bytes: int
    maximum_memory_mb: int
    maximum_cpu_seconds: int
    maximum_stderr_bytes: int
    trusted_signers: tuple[tuple[str, str], ...]

    @classmethod
    def from_profile(cls, value: dict[str, Any]) -> "PluginPolicy":
        raw = dict(value)
        allowed_keys = {
            "host_api_version", "profile_schema_version", "allowed_ids",
            "allowed_isolation", "allowed_permissions",
            "require_signed_external", "require_content_addressed_external",
            "maximum_timeout_seconds", "maximum_response_bytes",
            "maximum_memory_mb", "maximum_cpu_seconds",
            "maximum_stderr_bytes", "trusted_signers", "wit_contract",
        }
        unknown = set(raw) - allowed_keys
        if unknown:
            raise ValueError(
                f"unknown plugin policy keys: {sorted(unknown)}"
            )
        allowed = raw.get("allowed_ids")
        signers = tuple(sorted((str(key), str(val)) for key, val in dict(raw.get("trusted_signers") or {}).items()))
        policy = cls(
            host_api_version=str(raw["host_api_version"]),
            profile_schema_version=str(raw["profile_schema_version"]),
            allowed_ids=frozenset(str(x) for x in allowed) if allowed else None,
            allowed_isolation=frozenset(str(x) for x in raw["allowed_isolation"]),
            allowed_permissions=frozenset(str(x) for x in raw["allowed_permissions"]),
            require_signed_external=bool(raw["require_signed_external"]),
            require_content_addressed_external=bool(raw["require_content_addressed_external"]),
            maximum_timeout_seconds=int(raw["maximum_timeout_seconds"]),
            maximum_response_bytes=int(raw["maximum_response_bytes"]),
            maximum_memory_mb=int(raw["maximum_memory_mb"]),
            maximum_cpu_seconds=int(raw["maximum_cpu_seconds"]),
            maximum_stderr_bytes=int(raw["maximum_stderr_bytes"]),
            trusted_signers=signers,
        )
        if policy.maximum_timeout_seconds < 1:
            raise ValueError("plugin maximum timeout must be positive")
        if policy.maximum_response_bytes < 1024:
            raise ValueError("plugin maximum response must be at least 1024 bytes")
        if policy.maximum_memory_mb < 64:
            raise ValueError("plugin maximum memory must be at least 64 MiB")
        if policy.maximum_cpu_seconds < 1:
            raise ValueError("plugin maximum CPU time must be positive")
        if policy.maximum_stderr_bytes < 1024:
            raise ValueError("plugin maximum stderr must be at least 1024 bytes")
        return policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostApiVersion": self.host_api_version, "profileSchemaVersion": self.profile_schema_version,
            "allowedIds": sorted(self.allowed_ids) if self.allowed_ids is not None else None,
            "allowedIsolation": sorted(self.allowed_isolation), "allowedPermissions": sorted(self.allowed_permissions),
            "requireSignedExternal": self.require_signed_external,
            "requireContentAddressedExternal": self.require_content_addressed_external,
            "maximumTimeoutSeconds": self.maximum_timeout_seconds, "maximumMemoryMb": self.maximum_memory_mb,
            "maximumResponseBytes": self.maximum_response_bytes,
            "maximumCpuSeconds": self.maximum_cpu_seconds,
            "maximumStderrBytes": self.maximum_stderr_bytes,
            "trustedSigners": sorted(key for key, _ in self.trusted_signers),
            "manifestFirstDiscovery": True, "externalInProcessAllowed": False,
        }


@dataclass(frozen=True, slots=True)
class ExternalPluginManifest:
    descriptor: PluginDescriptor
    artifact: Path
    command: tuple[str, ...]
    signer_id: str | None
    signature: str | None
    manifest_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "ExternalPluginManifest":
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        raw = dict(data.get("descriptor") or {})
        mapping = {
            "apiVersion": "api_version", "hostApiRange": "host_api_range",
            "profileSchemaRange": "profile_schema_range", "resourceClass": "resource_class",
            "sideEffects": "side_effects", "artifactSha256": "artifact_sha256",
            "timeoutSeconds": "timeout_seconds", "maximumMemoryMb": "maximum_memory_mb",
            "trustClass": "trust_class", "compatibleProfileSchemas": "compatible_profile_schemas",
        }
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            normalized[mapping.get(key, key)] = value
        for key in ("capabilities", "permissions", "side_effects", "compatible_profile_schemas"):
            if key in normalized:
                normalized[key] = tuple(str(item) for item in normalized[key])
        contract = normalized.pop("contract", None)
        if contract is not None:
            if not isinstance(contract, Mapping):
                raise ValueError("plugin contract metadata must be an object")
            contract_value = dict(contract)
            contract_keys = {
                "id", "version", "safetyGuards", "cancellationMode",
                "failureMode", "rollbackSupported",
            }
            unknown_contract_keys = set(contract_value) - contract_keys
            missing_contract_keys = contract_keys - set(contract_value)
            if unknown_contract_keys:
                raise ValueError(
                    f"unknown plugin contract keys: {sorted(unknown_contract_keys)}"
                )
            if missing_contract_keys:
                raise ValueError(
                    f"missing plugin contract keys: {sorted(missing_contract_keys)}"
                )
            if not isinstance(contract_value["id"], str) or not isinstance(
                contract_value["version"], str
            ):
                raise ValueError("plugin contract id and version must be strings")
            if not isinstance(contract_value["safetyGuards"], list) or not all(
                isinstance(item, str) for item in contract_value["safetyGuards"]
            ):
                raise ValueError("plugin contract safety guards must be a string list")
            if not isinstance(contract_value["cancellationMode"], str) or not isinstance(
                contract_value["failureMode"], str
            ):
                raise ValueError("plugin contract lifecycle modes must be strings")
            if not isinstance(contract_value["rollbackSupported"], bool):
                raise ValueError("plugin contract rollback support must be boolean")
            normalized.update({
                "contract_id": contract_value.get("id"),
                "contract_version": contract_value.get("version"),
                "safety_guards": tuple(
                    str(item) for item in contract_value.get("safetyGuards") or ()
                ),
                "cancellation_mode": str(
                    contract_value.get("cancellationMode", "bounded")
                ),
                "failure_mode": str(
                    contract_value.get("failureMode", "fail_closed")
                ),
                "rollback_supported": contract_value["rollbackSupported"],
            })
        descriptor = PluginDescriptor(**normalized)
        artifact = Path(str(data["artifact"]))
        if not artifact.is_absolute():
            artifact = (source.parent / artifact).resolve()
        command = tuple(str(item) for item in data.get("command") or ())
        return cls(descriptor, artifact, command, data.get("signerId"), data.get("signature"), source)

    def signed_payload(self) -> bytes:
        value = {
            "descriptor": self.descriptor.to_dict(), "artifactSha256": self.descriptor.artifact_sha256,
            "command": list(self.command), "signerId": self.signer_id,
        }
        value["descriptor"].pop("signature", None)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ExternalPluginProxy:
    def __init__(
        self, manifest: ExternalPluginManifest, registry_policy: PluginPolicy
    ):
        self.manifest = manifest
        self.descriptor = manifest.descriptor
        self.registry_policy = registry_policy

    def execute(self, request: Any, **kwargs: Any) -> Any:
        if not self.manifest.artifact.is_file() or _sha256(
            self.manifest.artifact
        ) != self.descriptor.artifact_sha256:
            raise PermissionError("external plugin artifact changed after admission")
        params = {"request": request, "kwargs": kwargs}
        policy = PluginHostPolicy(
            timeout_seconds=self.descriptor.timeout_seconds,
            maximum_response_bytes=self.registry_policy.maximum_response_bytes,
            maximum_memory_mb=self.descriptor.maximum_memory_mb,
            maximum_cpu_seconds=self.registry_policy.maximum_cpu_seconds,
            allow_network="network_egress" in self.descriptor.permissions,
            allow_filesystem="filesystem_read" in self.descriptor.permissions,
            maximum_stderr_bytes=self.registry_policy.maximum_stderr_bytes,
        )
        if self.descriptor.isolation == "wasm_component":
            with WasmtimeComponentHost(self.manifest.artifact, policy=policy) as host:
                return host.call("execute", params)
        command = list(self.manifest.command) or [str(self.manifest.artifact)]
        with SubprocessPluginHost(command, policy=policy) as host:
            return host.call("execute", params)


@dataclass(slots=True)
class RegisteredPlugin:
    plugin: Any
    external: bool
    source: str = "builtin"


class PluginRegistry:
    """Versioned registry with manifest-first external discovery.

    Entry-point discovery never calls ``entry_point.load()``. External code is
    admitted only after static manifest, actual artifact hash and signature
    verification, and runs through an isolated host.
    """

    def __init__(self, *, policy: PluginPolicy):
        self.policy = policy
        self._plugins: dict[tuple[str, str], RegisteredPlugin] = {}
        self._history: dict[tuple[str, str], list[RegisteredPlugin]] = {}
        self._lock = threading.RLock()

    def _validate(self, descriptor: Any, *, external: bool) -> None:
        if not _range_allows(str(getattr(descriptor, "host_api_range", ">=1,<2")), self.policy.host_api_version):
            raise ValueError(f"plugin {descriptor.id} is incompatible with host API {self.policy.host_api_version}")
        if not _range_allows(str(getattr(descriptor, "profile_schema_range", ">=1,<2")), self.policy.profile_schema_version):
            raise ValueError(f"plugin {descriptor.id} is incompatible with profile schema {self.policy.profile_schema_version}")
        if str(getattr(descriptor, "api_version", "1")) != self.policy.host_api_version:
            raise ValueError(f"unsupported plugin API version {descriptor.api_version}")
        if self.policy.allowed_ids is not None and descriptor.id not in self.policy.allowed_ids:
            raise PermissionError(f"plugin {descriptor.id} is not allow-listed")
        isolation = str(getattr(descriptor, "isolation", "in_process"))
        if isolation not in self.policy.allowed_isolation:
            raise PermissionError(f"plugin isolation is not allowed: {isolation}")
        permissions = set(str(x) for x in getattr(descriptor, "permissions", ()))
        if permissions - self.policy.allowed_permissions:
            raise PermissionError(f"plugin requests disallowed permissions: {sorted(permissions - self.policy.allowed_permissions)}")
        if int(getattr(descriptor, "timeout_seconds", 30)) > self.policy.maximum_timeout_seconds:
            raise PermissionError("plugin timeout exceeds policy")
        if int(getattr(descriptor, "maximum_memory_mb", 512)) > self.policy.maximum_memory_mb:
            raise PermissionError("plugin memory exceeds policy")
        if external:
            if isolation not in {"process", "wasm_component"}:
                raise PermissionError("external plugins must use process or wasm_component isolation")
            if self.policy.require_signed_external and not bool(getattr(descriptor, "signed", False)):
                raise PermissionError("external plugin artifact must be signed")
            artifact_hash = str(getattr(descriptor, "artifact_sha256", "") or "")
            if self.policy.require_content_addressed_external and not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
                raise PermissionError("external plugin requires a SHA-256 artifact identity")

    def _verify_manifest(self, manifest: ExternalPluginManifest) -> None:
        self._validate(manifest.descriptor, external=True)
        if not manifest.artifact.is_file():
            raise FileNotFoundError(f"external plugin artifact not found: {manifest.artifact}")
        actual = _sha256(manifest.artifact)
        if actual != manifest.descriptor.artifact_sha256:
            raise PermissionError("external plugin artifact hash mismatch")
        if self.policy.require_signed_external:
            trusted = dict(self.policy.trusted_signers)
            if not manifest.signer_id or manifest.signer_id not in trusted or not manifest.signature:
                raise PermissionError("external plugin signer is not trusted")
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                key = Ed25519PublicKey.from_public_bytes(base64.b64decode(trusted[manifest.signer_id]))
                key.verify(base64.b64decode(manifest.signature), manifest.signed_payload())
            except Exception as exc:
                raise PermissionError("external plugin signature verification failed") from exc

    def register(self, plugin: Any, *, external: bool = False, source: str = "builtin") -> None:
        descriptor = getattr(plugin, "descriptor", None)
        if descriptor is None:
            raise TypeError("plugin has no descriptor")
        self._validate(descriptor, external=external)
        validate_plugin_contract(plugin, descriptor, external=external)
        key = (descriptor.kind, descriptor.id)
        with self._lock:
            if key in self._plugins:
                raise ValueError(f"duplicate plugin {key}")
            self._plugins[key] = RegisteredPlugin(plugin=plugin, external=external, source=source)

    def activate(self, plugin: Any, *, external: bool = False, source: str = "builtin") -> None:
        """Atomically activate a compatible version while retaining rollback state."""

        descriptor = getattr(plugin, "descriptor", None)
        if descriptor is None:
            raise TypeError("plugin has no descriptor")
        self._validate(descriptor, external=external)
        validate_plugin_contract(plugin, descriptor, external=external)
        key = (descriptor.kind, descriptor.id)
        replacement = RegisteredPlugin(plugin=plugin, external=external, source=source)
        with self._lock:
            current = self._plugins.get(key)
            if current is None:
                self._plugins[key] = replacement
                return
            if current.plugin.descriptor.version == descriptor.version:
                raise ValueError(f"plugin {key} version is already active")
            history = self._history.setdefault(key, [])
            history.append(current)
            if len(history) > MAXIMUM_PLUGIN_ROLLBACK_DEPTH:
                del history[:-MAXIMUM_PLUGIN_ROLLBACK_DEPTH]
            self._plugins[key] = replacement

    def rollback(self, kind: str, plugin_id: str) -> Any:
        """Restore the exact previously active plugin revision."""

        key = (str(kind), str(plugin_id))
        with self._lock:
            history = self._history.get(key)
            if not history:
                raise KeyError(f"no rollback revision for {kind} plugin {plugin_id}")
            restored = history.pop()
            self._plugins[key] = restored
            if not history:
                self._history.pop(key, None)
            return restored.plugin

    def register_manifest(self, path: str | Path, *, source: str = "manifest") -> str:
        manifest = ExternalPluginManifest.load(path)
        self._verify_manifest(manifest)
        proxy = ExternalPluginProxy(manifest, self.policy)
        self.register(proxy, external=True, source=f"{source}:{manifest.manifest_path}")
        return manifest.descriptor.id

    def activate_manifest(self, path: str | Path, *, source: str = "manifest") -> str:
        """Verify and atomically activate an external manifest revision."""

        manifest = ExternalPluginManifest.load(path)
        self._verify_manifest(manifest)
        proxy = ExternalPluginProxy(manifest, self.policy)
        self.activate(proxy, external=True, source=f"{source}:{manifest.manifest_path}")
        return manifest.descriptor.id

    @staticmethod
    def _manifest_path(entry_point: Any) -> Path:
        explicit = getattr(entry_point, "manifest_path", None)
        if explicit:
            return Path(explicit)
        dist = getattr(entry_point, "dist", None)
        if dist is None:
            raise FileNotFoundError(f"entry point {entry_point.name} has no distribution manifest")
        candidates = ["baloot-plugin.json", f"baloot-plugin-{entry_point.name}.json"]
        for name in candidates:
            try:
                candidate = Path(dist.locate_file(name))
            except Exception:
                continue
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"static plugin manifest not found for {entry_point.name}")

    def discover(self, group: str = "baloot_platform.plugins") -> list[str]:
        loaded: list[str] = []
        for entry_point in entry_points(group=group):
            # Deliberately do not call entry_point.load().
            loaded.append(self.register_manifest(self._manifest_path(entry_point), source=f"entrypoint:{entry_point.name}"))
        return loaded

    def get(self, kind: str, plugin_id: str) -> Any:
        with self._lock:
            try:
                return self._plugins[(kind, plugin_id)].plugin
            except KeyError as exc:
                raise KeyError(f"unknown {kind} plugin {plugin_id}") from exc

    def all(self, kind: str | None = None) -> tuple[Any, ...]:
        with self._lock:
            return tuple(item.plugin for (plugin_kind, _), item in sorted(self._plugins.items()) if kind is None or plugin_kind == kind)

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        with self._lock:
            items = tuple(sorted(self._plugins.items()))
        for (plugin_kind, _), registered in items:
            if kind is not None and plugin_kind != kind:
                continue
            value = registered.plugin.descriptor.to_dict()
            value.update({"external": registered.external, "source": registered.source})
            values.append(value)
        return values

    def invoke(
        self,
        kind: str,
        plugin_id: str,
        *args: Any,
        method: str | None = None,
        cancel_check: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a plugin through one fail-closed cancellation boundary."""

        plugin = self.get(kind, plugin_id)
        descriptor = plugin.descriptor
        contract = (
            PLUGIN_CONTRACTS_V1.get(descriptor.contract_id)
            if descriptor.contract_id is not None else None
        )
        if contract is not None and method is not None and method != contract.entrypoint:
            raise PluginExecutionError(
                f"formal plugin {plugin_id} may only invoke {contract.entrypoint}"
            )
        entrypoint_name = str(method or (contract.entrypoint if contract else "execute"))
        if cancel_check is not None and bool(cancel_check()):
            raise PluginCancelledError(f"plugin {plugin_id} invocation was cancelled")
        external = isinstance(plugin, ExternalPluginProxy)
        entrypoint = getattr(plugin, "execute" if external else entrypoint_name, None)
        if not callable(entrypoint):
            raise PluginExecutionError(
                f"plugin {plugin_id} does not implement {entrypoint_name}"
            )
        call_kwargs = dict(kwargs)
        if descriptor.cancellation_mode == "cooperative" and cancel_check is not None:
            call_kwargs["cancel_check"] = cancel_check
        try:
            if external:
                result = entrypoint({
                    "contractId": descriptor.contract_id,
                    "method": entrypoint_name,
                    "args": list(args),
                    "kwargs": call_kwargs,
                })
            else:
                result = entrypoint(*args, **call_kwargs)
        except PluginCancelledError:
            raise
        except Exception as exc:
            raise PluginExecutionError(
                f"plugin {plugin_id} failed in its {descriptor.failure_mode} boundary"
            ) from exc
        if cancel_check is not None and bool(cancel_check()):
            raise PluginCancelledError(f"plugin {plugin_id} invocation was cancelled")
        return result
