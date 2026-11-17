from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from baloot_platform.challenge.quality_diversity import QualityDiversitySelectorV1
from baloot_platform.challenge.scoring import ProfileChallengeScorerV1
from baloot_platform.challenge.tactical_evidence import CausalTacticalDetectorV1
from baloot_platform.plugins import (
    BeliefProfilePlugin,
    BestResponsePlugin,
    CpSatLoweringPlugin,
    DiversityPolicyPlugin,
    DomainPrimitivePlugin,
    InformationSetPlugin,
    ReferenceLoweringPlugin,
    ResearchObjectivePlugin,
    ResearchExecutorPlugin,
    ResearchRenderAdapterPlugin,
    RuntimeLoweringPlugin,
    TacticalProofPlugin,
    contract_catalog_v1,
)
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.plugins.contracts import (
    PLUGIN_CONTRACTS_V1,
    PluginCancelledError,
    PluginContractError,
    PluginExecutionError,
)
from baloot_platform.plugins.registry import (
    ExternalPluginManifest,
    PluginPolicy,
    PluginRegistry,
)
from baloot_platform.plugins.host import PluginHostPolicy, SubprocessPluginHost
from baloot_platform.research.auction_compiler import OrToolsCpSatAuctionCompilerV1
from baloot_platform.research.best_response_proof import OrToolsBoundedBestResponseProofV1
from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
from baloot_platform.research.service import ResearchFoundryServiceV1


def _policy(
    *,
    maximum_timeout_seconds: int = 60,
    maximum_memory_mb: int = 1024,
) -> PluginPolicy:
    return PluginPolicy.from_profile({
        "host_api_version": "1",
        "profile_schema_version": "1",
        "allowed_isolation": ["in_process", "process", "wasm_component"],
        "allowed_permissions": ["none"],
        "require_signed_external": False,
        "require_content_addressed_external": True,
        "maximum_timeout_seconds": maximum_timeout_seconds,
        "maximum_response_bytes": 1_048_576,
        "maximum_memory_mb": maximum_memory_mb,
        "maximum_cpu_seconds": 10,
        "maximum_stderr_bytes": 65_536,
    })


def _descriptor(
    contract_id: str,
    *,
    plugin_id: str = "test-plugin",
    version: str = "1.0.0",
    kind: str | None = None,
    deterministic: bool = True,
    guards: tuple[str, ...] | None = None,
    timeout_seconds: int = 10,
    cancellation_mode: str = "bounded",
    failure_mode: str = "fail_closed",
    isolation: str = "in_process",
    artifact_sha256: str | None = None,
) -> PluginDescriptor:
    contract = PLUGIN_CONTRACTS_V1[contract_id]
    return PluginDescriptor(
        id=plugin_id,
        kind=kind or contract.kinds[0],
        version=version,
        description="contract test plugin",
        deterministic=deterministic,
        timeout_seconds=timeout_seconds,
        maximum_memory_mb=128,
        isolation=isolation,
        artifact_sha256=artifact_sha256,
        contract_id=contract_id,
        contract_version="1",
        safety_guards=guards if guards is not None else contract.required_safety_guards,
        cancellation_mode=cancellation_mode,
        failure_mode=failure_mode,
    )


def test_contract_catalog_is_complete_versioned_and_stable() -> None:
    expected = {
        "domain_primitive",
        "reference_lowering",
        "runtime_lowering",
        "cp_sat_lowering",
        "information_set",
        "belief_profile",
        "research_objective",
        "best_response",
        "tactical_proof",
        "diversity_policy",
        "research_executor",
        "research_render_adapter",
    }
    assert set(PLUGIN_CONTRACTS_V1) == expected
    first = json.dumps(contract_catalog_v1(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(contract_catalog_v1(), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert all(item["version"] == "1" for item in contract_catalog_v1()["contracts"])


def test_protocols_match_current_implementations_and_minimal_lowerings() -> None:
    class Primitive:
        descriptor = _descriptor("domain_primitive")

        def definition(self) -> dict[str, Any]:
            return {}

    class Reference:
        descriptor = _descriptor("reference_lowering")

        def lower_reference(self, primitive: Any, **kwargs: Any) -> Any:
            return primitive

    class Runtime:
        descriptor = _descriptor("runtime_lowering")

        def lower_runtime(self, primitive: Any, **kwargs: Any) -> Any:
            return primitive

    class Information:
        descriptor = _descriptor("information_set")

        def partition(self, state: Any, actor: Any, **kwargs: Any) -> Any:
            return state, actor

    class Belief:
        descriptor = _descriptor("belief_profile")

        def build(self, profile: Any, **kwargs: Any) -> Any:
            return profile

    assert isinstance(Primitive(), DomainPrimitivePlugin)
    assert isinstance(Reference(), ReferenceLoweringPlugin)
    assert isinstance(Runtime(), RuntimeLoweringPlugin)
    assert isinstance(Information(), InformationSetPlugin)
    assert isinstance(Belief(), BeliefProfilePlugin)
    assert isinstance(OrToolsCpSatAuctionCompilerV1(), CpSatLoweringPlugin)
    assert isinstance(ProfileChallengeScorerV1(), ResearchObjectivePlugin)
    assert isinstance(OrToolsBoundedBestResponseProofV1(), BestResponsePlugin)
    assert isinstance(CausalTacticalDetectorV1(), TacticalProofPlugin)
    assert isinstance(QualityDiversitySelectorV1(), DiversityPolicyPlugin)
    assert isinstance(
        object.__new__(ResearchFoundryServiceV1), ResearchExecutorPlugin,
    )
    assert isinstance(
        object.__new__(ResearchRenderPreparationServiceV1),
        ResearchRenderAdapterPlugin,
    )


def test_registry_rejects_uncontracted_or_unsafe_formal_plugins() -> None:
    class MissingContract:
        descriptor = PluginDescriptor(
            "missing", "research_objective", "1.0.0", "missing contract",
        )

        def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any:
            return evidence

    registry = PluginRegistry(policy=_policy())
    with pytest.raises(PluginContractError, match="requires contract"):
        registry.register(MissingContract())

    class MissingObserverGuard:
        descriptor = _descriptor(
            "research_objective",
            plugin_id="unsafe",
            guards=("hard_gate_preservation", "frozen_provenance"),
        )

        def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any:
            return evidence

    with pytest.raises(PluginContractError, match="observer_safe"):
        registry.register(MissingObserverGuard())

    class NonDeterministic:
        descriptor = _descriptor(
            "research_objective", plugin_id="random", deterministic=False,
        )

        def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any:
            return evidence

    with pytest.raises(PluginContractError, match="deterministic"):
        registry.register(NonDeterministic())


def test_registry_enforces_contract_compatibility_and_resource_limits() -> None:
    class WrongVersion:
        descriptor = replace(
            _descriptor("research_objective", plugin_id="wrong"),
            contract_version="2",
        )

        def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any:
            return evidence

    with pytest.raises(PluginContractError, match="requires version 1"):
        PluginRegistry(policy=_policy()).register(WrongVersion())

    class TooSlow:
        descriptor = _descriptor(
            "research_objective", plugin_id="slow", timeout_seconds=11,
        )

        def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any:
            return evidence

    with pytest.raises(PermissionError, match="timeout"):
        PluginRegistry(
            policy=_policy(maximum_timeout_seconds=10)
        ).register(TooSlow())


def test_cooperative_cancellation_and_failure_are_fail_closed() -> None:
    calls: list[str] = []

    class Belief:
        descriptor = _descriptor(
            "belief_profile",
            cancellation_mode="cooperative",
        )

        def build(self, profile: Any, *, cancel_check: Any = None) -> Any:
            calls.append("called")
            if profile == "fail":
                raise ValueError("private implementation detail")
            return {"profile": profile}

    registry = PluginRegistry(policy=_policy())
    registry.register(Belief())
    with pytest.raises(PluginCancelledError):
        registry.invoke("belief_profile", "test-plugin", {}, cancel_check=lambda: True)
    assert calls == []
    first = registry.invoke(
        "belief_profile", "test-plugin", "stable", cancel_check=lambda: False,
    )
    second = registry.invoke(
        "belief_profile", "test-plugin", "stable", cancel_check=lambda: False,
    )
    assert first == second == {"profile": "stable"}
    with pytest.raises(PluginExecutionError, match="may only invoke build"):
        registry.invoke(
            "belief_profile", "test-plugin", "stable", method="unsafe_override",
        )
    with pytest.raises(PluginExecutionError) as error:
        registry.invoke("belief_profile", "test-plugin", "fail")
    assert isinstance(error.value.__cause__, ValueError)
    assert registry.get("belief_profile", "test-plugin").descriptor.version == "1.0.0"


def test_activation_failure_preserves_current_and_rollback_restores_exact_revision() -> None:
    class Objective:
        def __init__(self, version: str, *, safe: bool = True):
            guards = None if safe else ("hard_gate_preservation",)
            self.descriptor = _descriptor(
                "research_objective",
                plugin_id="objective",
                version=version,
                guards=guards,
            )

        def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any:
            return self.descriptor.version, evidence

    registry = PluginRegistry(policy=_policy())
    first = Objective("1.0.0")
    second = Objective("1.1.0")
    registry.register(first)
    with pytest.raises(PluginContractError):
        registry.activate(Objective("2.0.0", safe=False))
    assert registry.get("research_objective", "objective") is first
    registry.activate(second)
    assert registry.invoke("research_objective", "objective", "evidence", None)[0] == "1.1.0"
    assert registry.rollback("research_objective", "objective") is first
    assert registry.get("research_objective", "objective") is first


def test_manifest_first_discovery_preserves_formal_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "plugin.py"
    artifact.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " req=json.loads(line)\n"
        " print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
        "'result':{'version':'1.0.0'}}),flush=True)\n",
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "belief_profile",
        plugin_id="external-belief",
        cancellation_mode="host",
        failure_mode="isolated",
        isolation="process",
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    manifest = tmp_path / "baloot-plugin.json"
    manifest_payload = {
        "descriptor": descriptor.to_dict(),
        "artifact": artifact.name,
        "command": [sys.executable, str(artifact)],
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    invalid_manifest = tmp_path / "invalid-plugin.json"
    invalid_payload = json.loads(json.dumps(manifest_payload))
    invalid_payload["descriptor"]["contract"]["rollbackSupported"] = "true"
    invalid_manifest.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="rollback support must be boolean"):
        ExternalPluginManifest.load(invalid_manifest)

    class EntryPoint:
        name = "external-belief"
        manifest_path = manifest

        @staticmethod
        def load() -> Any:
            raise AssertionError("discovery must not import external code")

    import baloot_platform.plugins.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: (EntryPoint(),),
    )
    registry = PluginRegistry(policy=_policy())
    assert registry.discover() == ["external-belief"]
    listed = registry.list("belief_profile")
    assert listed[0]["contract"]["id"] == "belief_profile"
    assert listed[0]["external"] is True

    replacement_artifact = tmp_path / "plugin-v1-1.py"
    replacement_artifact.write_text(
        artifact.read_text(encoding="utf-8").replace("1.0.0", "1.1.0"),
        encoding="utf-8",
    )
    replacement_descriptor = replace(
        descriptor,
        version="1.1.0",
        artifact_sha256=hashlib.sha256(
            replacement_artifact.read_bytes()
        ).hexdigest(),
    )
    replacement_manifest = tmp_path / "baloot-plugin-v1-1.json"
    replacement_manifest.write_text(json.dumps({
        "descriptor": replacement_descriptor.to_dict(),
        "artifact": replacement_artifact.name,
        "command": [sys.executable, str(replacement_artifact)],
    }), encoding="utf-8")
    registry.activate_manifest(replacement_manifest)
    assert registry.get(
        "belief_profile", "external-belief",
    ).descriptor.version == "1.1.0"
    registry.rollback("belief_profile", "external-belief")
    assert registry.get(
        "belief_profile", "external-belief",
    ).descriptor.version == "1.0.0"
    assert registry.invoke(
        "belief_profile", "external-belief", {"profile": "frozen"},
    ) == {"version": "1.0.0"}
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(PluginExecutionError) as error:
        registry.invoke(
            "belief_profile", "external-belief", {"profile": "frozen"},
        )
    assert isinstance(error.value.__cause__, PermissionError)


@pytest.mark.real_clock
def test_external_host_timeout_is_fail_closed(tmp_path: Path) -> None:
    worker = tmp_path / "slow_plugin.py"
    worker.write_text(
        "import json,sys,time\n"
        "for line in sys.stdin:\n"
        " time.sleep(1)\n"
        " req=json.loads(line)\n"
        " print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{}}),flush=True)\n",
        encoding="utf-8",
    )
    host = SubprocessPluginHost(
        [sys.executable, str(worker)],
        policy=PluginHostPolicy(
            timeout_seconds=0.05,
            maximum_response_bytes=65_536,
            maximum_memory_mb=128,
            maximum_cpu_seconds=2,
            allow_network=False,
            allow_filesystem=False,
            maximum_stderr_bytes=8_192,
        ),
    )
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            host.call("build", {})
    finally:
        host.close()
