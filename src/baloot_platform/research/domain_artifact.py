from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from urllib.parse import urlparse

from .contracts import canonical_json, canonical_sha256
from .domain_compiler import (
    CompiledDomainBundleV1,
    CompilerCoverageV1,
    DomainIRV1,
    DomainPrimitiveDefinitionV1,
    DomainPrimitiveInstanceV1,
)


DOMAIN_COMPILATION_MEDIA_TYPE = (
    "application/vnd.balootx.domain-compilation-v3+json"
)


def _closed(
    value: Mapping[str, Any] | Any,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    raw = copy.deepcopy(dict(value))
    if set(raw) != expected:
        raise ValueError(f"{label} fields are incomplete or contain unknown keys")
    return raw


def _sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 identity")
    return text


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string or null")
    return value


def _definition(value: Mapping[str, Any]) -> DomainPrimitiveDefinitionV1:
    raw = _closed(
        value,
        {
            "schemaVersion",
            "kind",
            "version",
            "category",
            "requiredTargets",
            "lowerings",
            "documentation",
            "tests",
            "influenceProof",
        },
        "compiled-domain definition",
    )
    if _integer(raw["schemaVersion"], "definition schemaVersion") != 1:
        raise ValueError("unsupported compiled-domain definition version")
    lowerings = _closed(
        raw["lowerings"],
        {"runtime", "reference", "cpSat", "policy"},
        "compiled-domain definition lowerings",
    )
    definition = DomainPrimitiveDefinitionV1(
        kind=str(raw["kind"]),
        version=str(raw["version"]),
        category=str(raw["category"]),
        required_targets=_string_tuple(
            raw["requiredTargets"], "definition requiredTargets"
        ),
        runtime_lowering=_optional_string(lowerings["runtime"], "runtime lowering"),
        reference_lowering=_optional_string(
            lowerings["reference"], "reference lowering"
        ),
        cp_sat_lowering=_optional_string(lowerings["cpSat"], "CP-SAT lowering"),
        policy_lowering=_optional_string(lowerings["policy"], "policy lowering"),
        documentation=_string_tuple(raw["documentation"], "definition documentation"),
        tests=_string_tuple(raw["tests"], "definition tests"),
        influence_proof=_optional_string(raw["influenceProof"], "influence proof"),
    )
    if definition.to_dict() != raw:
        raise ValueError("compiled-domain definition canonical form mismatch")
    return definition


def _primitive(value: Mapping[str, Any]) -> DomainPrimitiveInstanceV1:
    raw = _closed(
        value,
        {
            "schemaVersion",
            "primitiveId",
            "definitionKind",
            "category",
            "valueHash",
            "requiredTargets",
            "coveredTargets",
            "missingTargets",
            "coverageStatus",
            "executableBindingStatus",
            "behavioralOrInvariantStatus",
            "declaredBehavioralEvidence",
            "verifiedBehavioralEvidence",
            "qualificationEligible",
            "value",
        },
        "compiled-domain primitive",
    )
    if _integer(raw["schemaVersion"], "primitive schemaVersion") != 1:
        raise ValueError("unsupported compiled-domain primitive version")
    instance = DomainPrimitiveInstanceV1(
        primitive_id=str(raw["primitiveId"]),
        definition_kind=str(raw["definitionKind"]),
        category=str(raw["category"]),
        value=copy.deepcopy(raw["value"]),
        value_hash=_sha256(raw["valueHash"], "primitive valueHash"),
        required_targets=_string_tuple(
            raw["requiredTargets"], "primitive requiredTargets"
        ),
        covered_targets=_string_tuple(
            raw["coveredTargets"], "primitive coveredTargets"
        ),
        missing_targets=_string_tuple(
            raw["missingTargets"], "primitive missingTargets"
        ),
        coverage_status=str(raw["coverageStatus"]),
        executable_binding_status=str(raw["executableBindingStatus"]),
        behavioral_or_invariant_status=str(raw["behavioralOrInvariantStatus"]),
        declared_behavioral_evidence=_optional_string(
            raw["declaredBehavioralEvidence"], "declared behavioral evidence"
        ),
        verified_behavioral_evidence=_optional_string(
            raw["verifiedBehavioralEvidence"], "verified behavioral evidence"
        ),
        qualification_eligible=_boolean(
            raw["qualificationEligible"], "primitive qualificationEligible"
        ),
    )
    if instance.value_hash != canonical_sha256(instance.value):
        raise ValueError("compiled-domain primitive value hash mismatch")
    expected_missing = tuple(
        target
        for target in instance.required_targets
        if target not in instance.covered_targets
    )
    if instance.missing_targets != expected_missing:
        raise ValueError("compiled-domain primitive target coverage mismatch")
    if instance.to_dict(include_value=True) != raw:
        raise ValueError("compiled-domain primitive canonical form mismatch")
    return instance


def _percentage(covered: int, total: int) -> float:
    return 100.0 if not total else round(covered * 100.0 / total, 6)


def hydrate_compiled_domain_bundle_v1(
    value: Mapping[str, Any],
) -> CompiledDomainBundleV1:
    """Hydrate and verify one full canonical compiled-domain payload."""

    raw = _closed(
        value,
        {
            "schemaVersion",
            "compilerVersion",
            "domainIr",
            "coverage",
            "definitions",
            "targets",
            "publishable",
            "sha256",
            "targetPayloads",
        },
        "compiled-domain artifact",
    )
    if _integer(raw["schemaVersion"], "compiled-domain schemaVersion") != 1:
        raise ValueError("unsupported compiled-domain artifact schema version")
    if raw["compilerVersion"] != "domain_compiler_v1":
        raise ValueError("unsupported compiled-domain compiler version")

    domain_raw = _closed(
        raw["domainIr"],
        {
            "schemaVersion",
            "primitiveInstances",
            "sha256",
            "rules",
            "conventions",
            "localization",
        },
        "compiled-domain IR",
    )
    if _integer(domain_raw["schemaVersion"], "domain IR schemaVersion") != 1:
        raise ValueError("unsupported compiled-domain IR version")
    if not isinstance(domain_raw["primitiveInstances"], list):
        raise ValueError("compiled-domain primitiveInstances must be an array")
    instances = tuple(_primitive(item) for item in domain_raw["primitiveInstances"])
    for key in ("rules", "conventions", "localization"):
        if not isinstance(domain_raw[key], Mapping):
            raise ValueError(f"compiled-domain {key} payload must be an object")
    rules = copy.deepcopy(dict(domain_raw["rules"]))
    conventions = copy.deepcopy(dict(domain_raw["conventions"]))
    localization = copy.deepcopy(dict(domain_raw["localization"]))
    domain_core = {
        "schemaVersion": 1,
        "rulesHash": canonical_sha256(rules),
        "conventionHash": canonical_sha256(conventions),
        "localizationHash": canonical_sha256(localization),
        "primitiveInstances": [
            instance.to_dict(include_value=True) for instance in instances
        ],
    }
    domain_hash = _sha256(domain_raw["sha256"], "compiled-domain IR sha256")
    if domain_hash != canonical_sha256(domain_core):
        raise ValueError("compiled-domain IR identity mismatch")
    domain_ir = DomainIRV1(
        rules,
        conventions,
        localization,
        instances,
        domain_hash,
    )

    coverage_raw = _closed(
        raw["coverage"],
        {
            "schemaVersion",
            "covered",
            "total",
            "percentage",
            "complete",
            "executableBindingCovered",
            "executableBindingTotal",
            "executableBindingPercentage",
            "executableBindingComplete",
            "behavioralOrInvariantCovered",
            "behavioralOrInvariantTotal",
            "behavioralOrInvariantPercentage",
            "behavioralOrInvariantComplete",
            "rows",
            "sha256",
        },
        "compiled-domain coverage",
    )
    if _integer(coverage_raw["schemaVersion"], "coverage schemaVersion") != 1:
        raise ValueError("unsupported compiled-domain coverage version")
    if not isinstance(coverage_raw["rows"], list):
        raise ValueError("compiled-domain coverage rows must be an array")
    coverage_rows = tuple(_primitive(item) for item in coverage_raw["rows"])
    if coverage_rows != instances:
        raise ValueError("compiled-domain coverage rows do not match the domain IR")
    total = len(instances)
    covered = sum(item.qualification_eligible for item in instances)
    executable_covered = sum(
        item.executable_binding_status == "covered" for item in instances
    )
    behavioral_covered = sum(
        item.behavioral_or_invariant_status
        in {"proven_behavioral", "proven_invariant"}
        for item in instances
    )
    expected_coverage = {
        "covered": covered,
        "total": total,
        "percentage": _percentage(covered, total),
        "complete": covered == total,
        "executableBindingCovered": executable_covered,
        "executableBindingTotal": total,
        "executableBindingPercentage": _percentage(executable_covered, total),
        "executableBindingComplete": executable_covered == total,
        "behavioralOrInvariantCovered": behavioral_covered,
        "behavioralOrInvariantTotal": total,
        "behavioralOrInvariantPercentage": _percentage(behavioral_covered, total),
        "behavioralOrInvariantComplete": behavioral_covered == total,
    }
    for key, expected in expected_coverage.items():
        if coverage_raw[key] != expected:
            raise ValueError(f"compiled-domain coverage {key} mismatch")
    coverage_core = {
        "schemaVersion": 1,
        "rows": [item.to_dict() for item in instances],
        "covered": covered,
        "total": total,
        "executableBindingCovered": executable_covered,
        "behavioralOrInvariantCovered": behavioral_covered,
    }
    coverage_hash = _sha256(
        coverage_raw["sha256"], "compiled-domain coverage sha256"
    )
    if coverage_hash != canonical_sha256(coverage_core):
        raise ValueError("compiled-domain coverage identity mismatch")
    coverage = CompilerCoverageV1(
        rows=instances,
        covered=covered,
        total=total,
        percentage=expected_coverage["percentage"],
        complete=expected_coverage["complete"],
        executable_binding_covered=executable_covered,
        executable_binding_total=total,
        executable_binding_percentage=expected_coverage[
            "executableBindingPercentage"
        ],
        executable_binding_complete=expected_coverage[
            "executableBindingComplete"
        ],
        behavioral_or_invariant_covered=behavioral_covered,
        behavioral_or_invariant_total=total,
        behavioral_or_invariant_percentage=expected_coverage[
            "behavioralOrInvariantPercentage"
        ],
        behavioral_or_invariant_complete=expected_coverage[
            "behavioralOrInvariantComplete"
        ],
        sha256=coverage_hash,
    )

    if not isinstance(raw["definitions"], list):
        raise ValueError("compiled-domain definitions must be an array")
    definitions = tuple(_definition(item) for item in raw["definitions"])
    targets = _closed(
        raw["targets"],
        {
            "runtimePlanHash",
            "referencePlanHash",
            "cpSatBlueprintHash",
            "policyPlanHash",
            "uiSchemaHash",
            "documentationModelHash",
        },
        "compiled-domain targets",
    )
    target_payloads = _closed(
        raw["targetPayloads"],
        {
            "runtimePlan",
            "referencePlan",
            "cpSatBlueprint",
            "policyPlan",
            "uiSchema",
            "documentationModel",
            "executableLowerings",
            "behavioralProofs",
        },
        "compiled-domain target payloads",
    )
    behavioral_proofs = target_payloads["behavioralProofs"]
    if not isinstance(behavioral_proofs, Mapping):
        raise ValueError("compiled-domain behavioral proofs must be an object")
    expected_proofs = {
        row.primitive_id: row
        for row in coverage.rows
        if row.verified_behavioral_evidence is not None
    }
    if set(behavioral_proofs) != set(expected_proofs):
        raise ValueError("compiled-domain behavioral proof coverage mismatch")
    rules_hash = canonical_sha256(domain_ir.rules)
    convention_hash = canonical_sha256(domain_ir.conventions)
    for primitive_id, row in expected_proofs.items():
        proof = behavioral_proofs[primitive_id]
        if not isinstance(proof, Mapping):
            raise ValueError("compiled-domain behavioral proof must be an object")
        receipt = copy.deepcopy(dict(proof))
        proof_hash = _sha256(
            receipt.get("proofHash"),
            "compiled-domain behavioral proofHash",
        )
        core = dict(receipt)
        core.pop("proofHash", None)
        if (
            receipt.get("primitiveId") != primitive_id
            or receipt.get("status") != row.behavioral_or_invariant_status
            or proof_hash != row.verified_behavioral_evidence
            or proof_hash != canonical_sha256(core)
        ):
            raise ValueError("compiled-domain behavioral proof identity mismatch")
        if primitive_id.startswith(("rules.", "conventions.")) and (
            receipt.get("rulesHash") != rules_hash
        ):
            raise ValueError("compiled-domain behavioral proof rules mismatch")
        if primitive_id.startswith("conventions.") and (
            receipt.get("profileHash") != convention_hash
        ):
            raise ValueError("compiled-domain behavioral proof profile mismatch")
    target_pairs = (
        ("runtimePlanHash", "runtimePlan"),
        ("referencePlanHash", "referencePlan"),
        ("cpSatBlueprintHash", "cpSatBlueprint"),
        ("policyPlanHash", "policyPlan"),
        ("uiSchemaHash", "uiSchema"),
        ("documentationModelHash", "documentationModel"),
    )
    target_hashes: list[str] = []
    for target_name, payload_name in target_pairs:
        declared = _sha256(targets[target_name], f"compiled-domain {target_name}")
        if declared != canonical_sha256(target_payloads[payload_name]):
            raise ValueError(f"compiled-domain {payload_name} identity mismatch")
        target_hashes.append(declared)

    compiled_core = {
        "schemaVersion": 1,
        "domainIrHash": domain_ir.sha256,
        "coverageHash": coverage.sha256,
        "targets": target_hashes,
    }
    compilation_hash = _sha256(raw["sha256"], "compiled-domain sha256")
    if compilation_hash != canonical_sha256(compiled_core):
        raise ValueError("compiled-domain internal identity mismatch")
    publishable = _boolean(raw["publishable"], "compiled-domain publishable")
    if publishable != coverage.complete:
        raise ValueError("compiled-domain publishability mismatch")
    compiled = CompiledDomainBundleV1(
        domain_ir=domain_ir,
        coverage=coverage,
        definitions=definitions,
        runtime_plan_hash=target_hashes[0],
        reference_plan_hash=target_hashes[1],
        cp_sat_blueprint_hash=target_hashes[2],
        policy_plan_hash=target_hashes[3],
        ui_schema_hash=target_hashes[4],
        documentation_model_hash=target_hashes[5],
        target_payloads=target_payloads,
        publishable=publishable,
        sha256=compilation_hash,
    )
    if compiled.to_dict(include_payloads=True) != raw:
        raise ValueError("compiled-domain canonical payload mismatch")
    return compiled


@dataclass(frozen=True, slots=True)
class DomainCompilationArtifactV1:
    compilation_hash: str
    artifact: Mapping[str, Any]
    bundle_hash: str
    effective_runtime_config_hash: str
    context_hash: str
    domain_hash: str
    created_at: str | None = None

    @staticmethod
    def _artifact(value: Mapping[str, Any]) -> dict[str, Any]:
        artifact = _closed(
            value,
            {"uri", "sha256", "bytes", "mediaType", "name"},
            "domain-compilation artifact reference",
        )
        digest = _sha256(artifact["sha256"], "artifact sha256")
        uri = str(artifact["uri"] or "").strip()
        parsed = urlparse(uri)
        local = (
            parsed.scheme == "cas"
            and parsed.netloc == "sha256"
            and parsed.path == f"/{digest}"
        )
        remote = (
            parsed.scheme == "s3"
            and bool(parsed.netloc)
            and parsed.path not in {"", "/"}
            and parsed.path.count("/") >= 2
            and parsed.path.rsplit("/", 1)[-1] == digest
        )
        if parsed.params or parsed.query or parsed.fragment or not (local or remote):
            raise ValueError("domain-compilation artifact URI does not match its hash")
        byte_count = _integer(artifact["bytes"], "artifact bytes")
        if byte_count < 1:
            raise ValueError("domain-compilation artifact must contain bytes")
        if artifact["mediaType"] != DOMAIN_COMPILATION_MEDIA_TYPE:
            raise ValueError("domain-compilation artifact media type is invalid")
        name = artifact["name"]
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("domain-compilation artifact name is invalid")
        return {
            "uri": uri,
            "sha256": digest,
            "bytes": byte_count,
            "mediaType": DOMAIN_COMPILATION_MEDIA_TYPE,
            "name": None if name is None else name.strip(),
        }

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "DomainCompilationArtifactV1":
        raw = _closed(
            value,
            {
                "schemaVersion",
                "compilationHash",
                "artifact",
                "bundleHash",
                "effectiveRuntimeConfigHash",
                "contextHash",
                "domainHash",
                "createdAt",
            },
            "domain-compilation artifact index",
        )
        if _integer(raw["schemaVersion"], "artifact index schemaVersion") != 1:
            raise ValueError("unsupported domain-compilation artifact index version")
        created_at = raw["createdAt"]
        if created_at is not None:
            if not isinstance(created_at, str):
                raise ValueError("domain-compilation createdAt must be a timestamp")
            try:
                datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    "domain-compilation createdAt must be a timestamp"
                ) from exc
        return cls(
            compilation_hash=_sha256(
                raw["compilationHash"], "compilationHash"
            ),
            artifact=cls._artifact(raw["artifact"]),
            bundle_hash=_sha256(raw["bundleHash"], "bundleHash"),
            effective_runtime_config_hash=_sha256(
                raw["effectiveRuntimeConfigHash"],
                "effectiveRuntimeConfigHash",
            ),
            context_hash=_sha256(raw["contextHash"], "contextHash"),
            domain_hash=_sha256(raw["domainHash"], "domainHash"),
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "compilationHash": self.compilation_hash,
            "artifact": copy.deepcopy(dict(self.artifact)),
            "bundleHash": self.bundle_hash,
            "effectiveRuntimeConfigHash": self.effective_runtime_config_hash,
            "contextHash": self.context_hash,
            "domainHash": self.domain_hash,
            "createdAt": self.created_at,
        }


class DomainCompilationArtifactRepositoryV1:
    """Bounded cache over an immutable Research-store/CAS compilation index."""

    def __init__(
        self,
        *,
        store: Any,
        artifact_store: Any,
        cache_size: int = 8,
    ) -> None:
        if not 1 <= int(cache_size) <= 64:
            raise ValueError("domain-compilation artifact cache size must be in [1, 64]")
        self.store = store
        self.artifact_store = artifact_store
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, CompiledDomainBundleV1] = OrderedDict()
        self._lock = threading.RLock()

    def _remember(self, compiled: CompiledDomainBundleV1) -> None:
        with self._lock:
            self._cache[compiled.sha256] = copy.deepcopy(compiled)
            self._cache.move_to_end(compiled.sha256)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def register(
        self,
        compiled: CompiledDomainBundleV1,
        *,
        bundle_hash: str,
        effective_runtime_config_hash: str,
        context_hash: str,
        domain_hash: str,
    ) -> DomainCompilationArtifactV1:
        with self._lock:
            return self._register_locked(
                compiled,
                bundle_hash=bundle_hash,
                effective_runtime_config_hash=effective_runtime_config_hash,
                context_hash=context_hash,
                domain_hash=domain_hash,
            )

    def _register_locked(
        self,
        compiled: CompiledDomainBundleV1,
        *,
        bundle_hash: str,
        effective_runtime_config_hash: str,
        context_hash: str,
        domain_hash: str,
    ) -> DomainCompilationArtifactV1:
        payload = canonical_json(
            compiled.to_dict(include_payloads=True)
        ).encode("utf-8")
        artifact = self.artifact_store.put(
            payload,
            media_type=DOMAIN_COMPILATION_MEDIA_TYPE,
            name=f"{compiled.sha256}.domain-compilation-v3.json",
        )
        if artifact.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("domain-compilation artifact digest mismatch")
        record = DomainCompilationArtifactV1.build(
            {
                "schemaVersion": 1,
                "compilationHash": compiled.sha256,
                "artifact": artifact,
                "bundleHash": bundle_hash,
                "effectiveRuntimeConfigHash": effective_runtime_config_hash,
                "contextHash": context_hash,
                "domainHash": domain_hash,
                "createdAt": None,
            }
        )
        stored = DomainCompilationArtifactV1.build(
            self.store.put_domain_compilation_artifact(record.to_dict())
        )
        if (
            stored.compilation_hash != compiled.sha256
            or stored.artifact["sha256"] != artifact["sha256"]
        ):
            raise ValueError("domain-compilation artifact index identity mismatch")
        self._remember(compiled)
        return stored

    def get(self, compilation_hash: str) -> CompiledDomainBundleV1 | None:
        expected = _sha256(compilation_hash, "compilationHash")
        with self._lock:
            cached = self._cache.get(expected)
            if cached is not None:
                self._cache.move_to_end(expected)
                return copy.deepcopy(cached)
        stored = self.store.get_domain_compilation_artifact(expected)
        if stored is None:
            return None
        record = DomainCompilationArtifactV1.build(stored)
        if record.compilation_hash != expected:
            raise ValueError("domain-compilation artifact index key mismatch")
        reference = record.artifact
        data = self.artifact_store.get(str(reference["uri"]))
        if len(data) != int(reference["bytes"]):
            raise ValueError("domain-compilation artifact byte count mismatch")
        if hashlib.sha256(data).hexdigest() != str(reference["sha256"]):
            raise ValueError("domain-compilation artifact checksum mismatch")
        try:
            decoded = data.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("domain-compilation artifact is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("domain-compilation artifact must contain a JSON object")
        if canonical_json(value).encode("utf-8") != data:
            raise ValueError("domain-compilation artifact is not canonically serialized")
        compiled = hydrate_compiled_domain_bundle_v1(value)
        if compiled.sha256 != expected:
            raise ValueError("domain-compilation artifact internal identity mismatch")
        self._remember(compiled)
        return copy.deepcopy(compiled)

    @property
    def cached_count(self) -> int:
        with self._lock:
            return len(self._cache)
