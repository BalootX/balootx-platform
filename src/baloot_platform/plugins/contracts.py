from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .base import PluginDescriptor


AUTHORITATIVE_LEGALITY = "authoritative_legality"
COMPILED_DOMAIN_AUTHORITY = "compiled_domain_authority"
OBSERVER_SAFE = "observer_safe"
HARD_GATE_PRESERVATION = "hard_gate_preservation"
FROZEN_PROVENANCE = "frozen_provenance"
ON_DEMAND_ONLY = "on_demand_only"
KNOWN_SAFETY_GUARDS = frozenset({
    AUTHORITATIVE_LEGALITY,
    COMPILED_DOMAIN_AUTHORITY,
    OBSERVER_SAFE,
    HARD_GATE_PRESERVATION,
    FROZEN_PROVENANCE,
    ON_DEMAND_ONLY,
})


class PluginContractError(ValueError):
    """A plugin does not satisfy its declared versioned contract."""


class PluginCancelledError(RuntimeError):
    """A plugin call was cancelled before it could produce accepted output."""


class PluginExecutionError(RuntimeError):
    """A plugin failed inside its fail-closed execution boundary."""


@dataclass(frozen=True, slots=True)
class PluginContractDescriptorV1:
    id: str
    version: str
    kinds: tuple[str, ...]
    entrypoint: str
    required_safety_guards: tuple[str, ...]
    cancellation_modes: tuple[str, ...]
    allowed_resource_classes: tuple[str, ...] = (
        "cpu", "analysis-cpu", "io", "storage", "gpu",
    )
    deterministic_required: bool = True
    failure_modes: tuple[str, ...] = ("fail_closed", "isolated")
    rollback_required: bool = True
    allowed_side_effects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.id,
            "version": self.version,
            "kinds": list(self.kinds),
            "entrypoint": self.entrypoint,
            "requiredSafetyGuards": list(self.required_safety_guards),
            "cancellationModes": list(self.cancellation_modes),
            "allowedResourceClasses": list(self.allowed_resource_classes),
            "deterministicRequired": self.deterministic_required,
            "failureModes": list(self.failure_modes),
            "rollbackRequired": self.rollback_required,
            "allowedSideEffects": list(self.allowed_side_effects),
        }


DOMAIN_PRIMITIVE_CONTRACT_V1 = PluginContractDescriptorV1(
    "domain_primitive", "1", ("domain_primitive",), "definition",
    (COMPILED_DOMAIN_AUTHORITY, HARD_GATE_PRESERVATION), ("bounded", "cooperative"),
)
REFERENCE_LOWERING_CONTRACT_V1 = PluginContractDescriptorV1(
    "reference_lowering", "1", ("reference_lowering",), "lower_reference",
    (COMPILED_DOMAIN_AUTHORITY, AUTHORITATIVE_LEGALITY), ("bounded", "cooperative"),
)
RUNTIME_LOWERING_CONTRACT_V1 = PluginContractDescriptorV1(
    "runtime_lowering", "1", ("runtime_lowering",), "lower_runtime",
    (COMPILED_DOMAIN_AUTHORITY, AUTHORITATIVE_LEGALITY), ("bounded", "cooperative"),
)
CP_SAT_LOWERING_CONTRACT_V1 = PluginContractDescriptorV1(
    "cp_sat_lowering", "1", ("cp_sat_lowering", "domain_lowering"), "compile",
    (COMPILED_DOMAIN_AUTHORITY, AUTHORITATIVE_LEGALITY, HARD_GATE_PRESERVATION),
    ("bounded", "cooperative", "host"),
)
INFORMATION_SET_CONTRACT_V1 = PluginContractDescriptorV1(
    "information_set", "1", ("information_set",), "partition",
    (AUTHORITATIVE_LEGALITY, OBSERVER_SAFE, FROZEN_PROVENANCE), ("bounded", "cooperative"),
)
BELIEF_PROFILE_CONTRACT_V1 = PluginContractDescriptorV1(
    "belief_profile", "1", ("belief_profile",), "build",
    (OBSERVER_SAFE, FROZEN_PROVENANCE), ("bounded", "cooperative", "host"),
)
RESEARCH_OBJECTIVE_CONTRACT_V1 = PluginContractDescriptorV1(
    "research_objective", "1", ("research_objective", "challenge_scorer"), "score",
    (OBSERVER_SAFE, HARD_GATE_PRESERVATION, FROZEN_PROVENANCE),
    ("bounded", "cooperative", "host"),
)
BEST_RESPONSE_CONTRACT_V1 = PluginContractDescriptorV1(
    "best_response", "1", ("best_response",), "prove",
    (AUTHORITATIVE_LEGALITY, OBSERVER_SAFE, HARD_GATE_PRESERVATION, FROZEN_PROVENANCE),
    ("bounded", "cooperative", "host"),
)
TACTICAL_PROOF_CONTRACT_V1 = PluginContractDescriptorV1(
    "tactical_proof", "1", ("tactical_proof", "tactical_detector"), "detect",
    (OBSERVER_SAFE, HARD_GATE_PRESERVATION, FROZEN_PROVENANCE),
    ("bounded", "cooperative", "host"),
)
DIVERSITY_POLICY_CONTRACT_V1 = PluginContractDescriptorV1(
    "diversity_policy", "1", ("diversity_policy", "selection_policy"), "select",
    (OBSERVER_SAFE, HARD_GATE_PRESERVATION, FROZEN_PROVENANCE),
    ("bounded", "cooperative", "host"),
)
RESEARCH_EXECUTOR_CONTRACT_V1 = PluginContractDescriptorV1(
    "research_executor", "1", ("research_executor", "research_foundry"), "submit",
    (AUTHORITATIVE_LEGALITY, OBSERVER_SAFE, HARD_GATE_PRESERVATION, FROZEN_PROVENANCE),
    ("lifecycle", "host"), allowed_side_effects=("storage_write", "artifact_write"),
)
RESEARCH_RENDER_ADAPTER_CONTRACT_V1 = PluginContractDescriptorV1(
    "research_render_adapter", "1", ("research_render_adapter",), "prepare",
    (AUTHORITATIVE_LEGALITY, OBSERVER_SAFE, FROZEN_PROVENANCE, ON_DEMAND_ONLY),
    ("bounded", "cooperative", "host"), allowed_resource_classes=("cpu", "io"),
    allowed_side_effects=("artifact_write",),
)


PLUGIN_CONTRACTS_V1: Mapping[str, PluginContractDescriptorV1] = MappingProxyType({
    item.id: item
    for item in (
        DOMAIN_PRIMITIVE_CONTRACT_V1,
        REFERENCE_LOWERING_CONTRACT_V1,
        RUNTIME_LOWERING_CONTRACT_V1,
        CP_SAT_LOWERING_CONTRACT_V1,
        INFORMATION_SET_CONTRACT_V1,
        BELIEF_PROFILE_CONTRACT_V1,
        RESEARCH_OBJECTIVE_CONTRACT_V1,
        BEST_RESPONSE_CONTRACT_V1,
        TACTICAL_PROOF_CONTRACT_V1,
        DIVERSITY_POLICY_CONTRACT_V1,
        RESEARCH_EXECUTOR_CONTRACT_V1,
        RESEARCH_RENDER_ADAPTER_CONTRACT_V1,
    )
})

CONTRACT_REQUIRED_BY_KIND: Mapping[str, str] = MappingProxyType({
    kind: contract.id
    for contract in PLUGIN_CONTRACTS_V1.values()
    for kind in contract.kinds
})


def contract_catalog_v1() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "contracts": [
            PLUGIN_CONTRACTS_V1[key].to_dict()
            for key in sorted(PLUGIN_CONTRACTS_V1)
        ],
    }


def validate_plugin_contract(
    plugin: Any,
    descriptor: PluginDescriptor,
    *,
    external: bool,
) -> PluginContractDescriptorV1 | None:
    required_contract_id = CONTRACT_REQUIRED_BY_KIND.get(descriptor.kind)
    if descriptor.contract_id is None:
        if required_contract_id is not None:
            raise PluginContractError(
                f"plugin kind {descriptor.kind} requires contract {required_contract_id}"
            )
        if descriptor.contract_version is not None or descriptor.safety_guards:
            raise PluginContractError("contract metadata requires a contract id")
        return None
    contract = PLUGIN_CONTRACTS_V1.get(descriptor.contract_id)
    if contract is None:
        raise PluginContractError(f"unknown plugin contract {descriptor.contract_id}")
    if required_contract_id is not None and contract.id != required_contract_id:
        raise PluginContractError(
            f"plugin kind {descriptor.kind} requires contract {required_contract_id}"
        )
    if descriptor.kind not in contract.kinds:
        raise PluginContractError(
            f"contract {contract.id} does not allow plugin kind {descriptor.kind}"
        )
    if descriptor.contract_version != contract.version:
        raise PluginContractError(
            f"plugin contract {contract.id} requires version {contract.version}"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", descriptor.id):
        raise PluginContractError("formal plugin ids must be stable lowercase identifiers")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", descriptor.version):
        raise PluginContractError("formal plugin versions must use semantic versioning")
    if (
        not isinstance(descriptor.safety_guards, tuple)
        or not all(isinstance(item, str) for item in descriptor.safety_guards)
        or len(set(descriptor.safety_guards)) != len(descriptor.safety_guards)
    ):
        raise PluginContractError("plugin safety guards must be a unique tuple")
    unknown_guards = set(descriptor.safety_guards) - KNOWN_SAFETY_GUARDS
    if unknown_guards:
        raise PluginContractError(
            f"plugin {descriptor.id} declares unknown safety guards: {sorted(unknown_guards)}"
        )
    missing_guards = set(contract.required_safety_guards) - set(descriptor.safety_guards)
    if missing_guards:
        raise PluginContractError(
            f"plugin {descriptor.id} is missing safety guards: {sorted(missing_guards)}"
        )
    if contract.deterministic_required and descriptor.deterministic is not True:
        raise PluginContractError(f"plugin {descriptor.id} must declare deterministic behavior")
    if descriptor.resource_class not in contract.allowed_resource_classes:
        raise PluginContractError(
            f"plugin {descriptor.id} has unsupported resource class {descriptor.resource_class}"
        )
    if (
        not isinstance(descriptor.timeout_seconds, int)
        or isinstance(descriptor.timeout_seconds, bool)
        or not isinstance(descriptor.maximum_memory_mb, int)
        or isinstance(descriptor.maximum_memory_mb, bool)
        or descriptor.timeout_seconds < 1
        or descriptor.maximum_memory_mb < 1
    ):
        raise PluginContractError("plugin resource limits must be positive")
    if descriptor.cancellation_mode not in contract.cancellation_modes:
        raise PluginContractError(
            f"plugin {descriptor.id} has unsupported cancellation mode {descriptor.cancellation_mode}"
        )
    if descriptor.failure_mode not in contract.failure_modes:
        raise PluginContractError(
            f"plugin {descriptor.id} has unsupported failure mode {descriptor.failure_mode}"
        )
    unsupported_side_effects = set(descriptor.side_effects) - set(
        contract.allowed_side_effects
    )
    if unsupported_side_effects:
        raise PluginContractError(
            f"plugin {descriptor.id} declares unsupported side effects: "
            f"{sorted(unsupported_side_effects)}"
        )
    if external and descriptor.failure_mode != "isolated":
        raise PluginContractError("external formal plugins must declare isolated failure mode")
    if (
        not external
        and descriptor.isolation == "in_process"
        and descriptor.trust_class != "builtin"
    ):
        raise PluginContractError(
            "only reviewed builtin formal plugins may run in process"
        )
    if not external and descriptor.failure_mode == "isolated" and descriptor.isolation == "in_process":
        raise PluginContractError("an in-process plugin cannot claim isolated failure mode")
    if contract.rollback_required and descriptor.rollback_supported is not True:
        raise PluginContractError(f"plugin {descriptor.id} must support registry rollback")
    entrypoint = getattr(plugin, "execute" if external else contract.entrypoint, None)
    if not callable(entrypoint):
        raise PluginContractError(
            f"plugin {descriptor.id} must implement {contract.entrypoint}"
        )
    if descriptor.cancellation_mode == "cooperative" and not external:
        signature = inspect.signature(entrypoint)
        parameters = signature.parameters
        accepts_cancel = "cancel_check" in parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        )
        if not accepts_cancel:
            raise PluginContractError(
                f"plugin {descriptor.id} must accept a cooperative cancel_check"
            )
    if descriptor.cancellation_mode == "lifecycle" and not callable(getattr(plugin, "cancel", None)):
        raise PluginContractError(f"plugin {descriptor.id} must implement cancel")
    return contract
