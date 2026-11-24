from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from baloot_platform.research.scenario_contracts import canonical_sha256
from scripts import benchmark_ortools_research_v4 as benchmark


def test_default_output_targets_native_v3_performance_receipt() -> None:
    assert benchmark.OUTPUT == (
        benchmark.ROOT / "release/ortools-research-v3-performance.json"
    )


def test_real_benchmark_refuses_stale_v3_contradiction_profile(root: Path) -> None:
    with pytest.raises(ValueError, match="unsound V3 flagship contradiction"):
        benchmark.run_benchmark(root)


@pytest.mark.real_clock
@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("complete_round", [False, True])
def test_quality_stage_diagnostic_accounts_and_restores_solver(monkeypatch, tmp_path, exit_code, complete_round):
    from ortools.sat.python import cp_model
    original = cp_model.CpSolver.solve
    def run(arguments):
        assert "test_native_quality_generates_all_first_trick_decisions and " + (
            "True" if complete_round else "False"
        ) in arguments
        model = cp_model.CpModel()
        value = model.new_int_var(0, 1, "value")
        model.minimize(value)
        assert cp_model.CpSolver().solve(model) == cp_model.OPTIMAL
        return exit_code
    monkeypatch.setattr(pytest, "main", run)
    monkeypatch.setattr(benchmark, "git_identity", lambda root: {"revision": "test"})
    result = benchmark.profile_quality_fixture(tmp_path, complete_round=complete_round)
    assert cp_model.CpSolver.solve is original
    assert result["testExitCode"] == exit_code
    assert result["status"] == ("passed" if exit_code == 0 else "failed")
    assert result["timingAccountingValid"] is True
    assert result["normalizationInputCounts"]["calls"] == result["stages"].get("normalization", {"calls": 0})["calls"]
    row = result["stages"]["cpSatSolve"]
    assert row["calls"] == 1
    assert sum(site["calls"] for site in result["solverCallSites"].values()) == row["calls"]
    assert sum(site["seconds"] for site in result["solverCallSites"].values()) == pytest.approx(row["inclusiveSeconds"])
    assert 0 <= row["exclusiveSeconds"] <= row["inclusiveSeconds"] <= result["elapsedSeconds"]
    assert result["evidenceSha256"] == canonical_sha256({key: value for key, value in result.items() if key != "evidenceSha256"})


@pytest.mark.parametrize("output", [None, "release/diagnostic.json", "existing.json"])
def test_quality_diagnostic_cli_preserves_evidence(monkeypatch, tmp_path, output):
    import sys
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    arguments = ["benchmark", "--quality-diagnostic"]
    if output is not None:
        target = tmp_path / output
        arguments.extend(["--output", str(target)])
        if output == "existing.json":
            target.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as stopped:
        benchmark.main()
    assert stopped.value.code == 2
    if output == "existing.json":
        assert target.read_text(encoding="utf-8") == "preserved"


def test_full_round_diagnostic_requires_diagnostic_mode(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["benchmark", "--quality-diagnostic-full-round"])
    with pytest.raises(SystemExit) as stopped:
        benchmark.main()
    assert stopped.value.code == 2


def _operation(
    name: str,
    *,
    measurement_ms: float,
    completed: int,
    extra_resources: dict | None = None,
):
    semantic_hash = canonical_sha256({"operation": name})
    context_hash = canonical_sha256({"context": "published-ortools"})

    def factory():
        return lambda: {
            "semanticHash": semantic_hash,
            "contextHash": context_hash,
            "measurementMs": measurement_ms,
            "stageTimings": {
                stage: measurement_ms / 1000.0
                for stage in benchmark._REQUIRED_STAGE_TIMINGS[name]
            },
            "resources": {
                "conflicts": 1,
                "branches": 2,
                "cacheBytes": 3,
                "completedVariants": completed,
                **(
                    {
                        "normalizationBackendId": "ortools_cp_sat_v3",
                        "normalizationReceiptHash": "1" * 64,
                        "normalizationModelHash": "2" * 64,
                    }
                    if name == "exactNormalization"
                    else {}
                ),
                **(extra_resources or {}),
                **(
                    {"cancellationStatus": "cancelled"}
                    if name == "cancellationAcknowledgement"
                    else {}
                ),
            },
        }

    return benchmark.BenchmarkOperationV1(factory)


def test_real_exact_operation_constructs_the_native_normalizer(monkeypatch) -> None:
    import types

    captured = []

    class FakeNormalizer:
        def __init__(self, rules, *, native_cp_sat=False):
            captured.append((rules, native_cp_sat))

    owner = object.__new__(benchmark._RealOperationsV1)
    owner.context = types.SimpleNamespace(
        bundle=types.SimpleNamespace(rules="RULES")
    )
    owner.exact_normalization = lambda normalizer: normalizer
    monkeypatch.setattr(benchmark, "OrToolsExactStateNormalizerV1", FakeNormalizer)

    owner.operations()["exactNormalization"].factory()()

    assert captured == [("RULES", True)]


@pytest.mark.real_clock
def test_first_result_benchmark_forwards_governed_candidate_deadline(
    monkeypatch,
) -> None:
    captured = []

    class FakeExecutor:
        def execute(self, *args, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace()

    owner = object.__new__(benchmark._RealOperationsV1)
    owner.search_one = {
        "scenario": {},
        "proof": {},
        "resources": {"seedCount": 7, "solverTimeoutSeconds": 12.5},
    }
    owner.search_five = owner.search_one
    owner.proof_budget = {
        "maximumNodes": 101,
        "maximumDepth": 9,
        "maximumHiddenWorlds": 11,
    }
    owner.search_budget = {"maximumPrefixLiftAttempts": 13}
    owner.maximum_line_count_limbs = 3
    owner.container = SimpleNamespace(
        research=SimpleNamespace(
            validate_recipe=lambda payload: {"valid": True},
            estimate=lambda payload: {"launchAdmissible": True},
        )
    )
    owner._is_qualified_alternate = lambda result: True
    owner._compatibility_seconds = lambda result: 0.0
    owner._executor_result = lambda result, **kwargs: {"resources": {}}
    owner._search_stage_timings = lambda *args, **kwargs: {}
    monkeypatch.setattr(benchmark.ResearchScenarioV1, "build", lambda value: "SCENARIO")
    monkeypatch.setattr(benchmark.ScenarioProofRequestV1, "build", lambda value: "PROOF")

    owner._search(count=1, executor=FakeExecutor())

    assert len(captured) == 1
    assert captured[0]["maximum_seconds"] == 12.5
    assert captured[0]["maximum_candidates"] == 7
    assert captured[0]["maximum_prefix_lift_attempts"] == 13
    assert captured[0]["maximum_proof_nodes"] == 101


def test_exact_operation_rejects_a_normalizer_from_another_backend() -> None:
    import types

    class Receipt:
        @staticmethod
        def to_dict():
            return {
                "backendId": "ortools_cp_sat_v2",
                "sha256": "1" * 64,
                "modelHash": "2" * 64,
            }

    class Normalizer:
        @staticmethod
        def normalize(*_args, **_kwargs):
            return types.SimpleNamespace(
                sha256="3" * 64,
                normalization_receipt=Receipt(),
            )

    owner = object.__new__(benchmark._RealOperationsV1)
    owner.context = types.SimpleNamespace(
        bundle=types.SimpleNamespace(
            component=lambda _family: {
                "scenarioResearchV4": {"backend": "ortools_cp_sat_v3"}
            }
        )
    )
    owner.context_hash = "4" * 64
    owner.exact_scenario = types.SimpleNamespace(
        origin=types.SimpleNamespace(exact_trace_prefix={}),
        decision_point={"kind": "play"},
    )

    import pytest

    with pytest.raises(RuntimeError, match="does not match the bundle"):
        owner.exact_normalization(Normalizer())


def test_benchmark_rejects_exact_sample_authority_drift(
    root: Path,
    tmp_path: Path,
) -> None:
    import pytest
    import yaml

    profile = benchmark._load_profile(
        root / "configs/research/ortools_performance_benchmark_v1.yaml"
    )
    profile["measurement"].update(
        {"coldSamples": 2, "warmSamples": 2, "warmupIterations": 0}
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    operations = {
        name: _operation(
            name,
            measurement_ms=1.0,
            completed=benchmark._REQUIRED_COMPLETED_VARIANTS[name],
        )
        for name in benchmark._OPERATIONS
    }
    operations["exactNormalization"] = _operation(
        "exactNormalization",
        measurement_ms=1.0,
        completed=1,
        extra_resources={"normalizationBackendId": "ortools_cp_sat_v2"},
    )

    with pytest.raises(RuntimeError, match="authority drifted"):
        benchmark.run_benchmark(
            root,
            operations=operations,
            source_identity={"binding": "test"},
            profile_path=profile_path,
            benchmark_recipe_identities=_recipe_identities(),
        )


def _recipe_identities():
    return {
        "original": {
            "source": "examples/research-scenario-v4-flagship-exact-infeasible.json",
            "sourceSha256": "a" * 64,
            "recipeCode": "RCP-FLAGSHIPINF",
            "scenarioHash": "b" * 64,
            "proofHash": "c" * 64,
            "classification": "original_exact_structural_contradiction",
        },
        "alternate": {
            "source": (
                "examples/research-scenario-v4-flagship-"
                "alternate-target-wins-six-search-five.json"
            ),
            "sourceSha256": "d" * 64,
            "recipeCode": "RCP-FLAGSHIPALT",
            "scenarioHash": "e" * 64,
            "proofHash": "f" * 64,
            "classification": "alternate_target_team_wins_trick_six",
        },
    }


def test_benchmark_profiles_preserve_v1_and_bind_v2_to_the_new_bundle(
    root: Path,
    tmp_path: Path,
) -> None:
    legacy_path = root / "configs/research/ortools_performance_benchmark_v1.yaml"
    current_path = root / "configs/research/ortools_performance_benchmark_v2.yaml"
    packaged_path = (
        root
        / "src/baloot_platform/resources/configs/research/ortools_performance_benchmark_v2.yaml"
    )

    legacy = benchmark._load_profile(legacy_path)
    current = benchmark._load_profile(current_path)

    assert benchmark.PROFILE == current_path
    assert legacy["profile"]["version"] == "1.1.0"
    assert legacy["bundle"]["source"] == (
        "configs/bundle_ortools_research_native_v3.yaml"
    )
    assert current["profile"]["version"] == "2.0.0"
    assert current["bundle"]["source"] == (
        "configs/bundle_intelligent_research.yaml"
    )
    assert current_path.read_bytes() == packaged_path.read_bytes()

    current["bundle"] = legacy["bundle"]
    cross_wired_path = tmp_path / "cross-wired.yaml"
    cross_wired_path.write_text(
        __import__("yaml").safe_dump(current, sort_keys=False),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="must select the published V4 bundle"):
        benchmark._load_profile(cross_wired_path)


@pytest.mark.parametrize(
    "mismatch", [None, "id", "version", "component", "backend"]
)
def test_publish_explicit_bundle_uses_configuration_ssot_and_checks_identity(
    root: Path,
    mismatch: str | None,
) -> None:
    profile = benchmark._load_profile(
        root / "configs/research/ortools_performance_benchmark_v2.yaml"
    )
    source = str(profile["bundle"]["source"])
    packaged = benchmark.build_bundle(root, bundle_path=source)
    frozen_profile = dict(packaged.bundle_profile["profile"])
    if mismatch in {"id", "version"}:
        frozen_profile[mismatch] = "mismatch"
    frozen_components = {
        family: SimpleNamespace(sha256=component.sha256)
        for family, component in packaged.components.items()
    }
    if mismatch == "component":
        family = next(iter(frozen_components))
        frozen_components[family] = SimpleNamespace(sha256="0" * 64)
    challenge = dict(packaged.component("challenge"))
    if mismatch == "backend":
        challenge = {
            **challenge,
            "scenarioResearchV4": {
                **challenge["scenarioResearchV4"],
                "backend": "mismatch",
            },
        }
    bundle = SimpleNamespace(
        bundle_profile={"profile": frozen_profile},
        sha256="0" * 64,
        mode=packaged.mode,
        rules=packaged.rules,
        components=frozen_components,
        component=lambda family: (
            challenge if family == "challenge" else packaged.component(family)
        ),
    )
    bootstrap_calls = []
    freeze_calls = []

    def bootstrap(**kwargs):
        bootstrap_calls.append(kwargs)
        return {"bundle": {"revisionId": "CBR-TEST"}}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("manual profile or bundle publication is forbidden")

    authority = object()
    context = SimpleNamespace(bundle=bundle)

    def freeze(**kwargs):
        freeze_calls.append(kwargs)
        return authority, context

    container = SimpleNamespace(
        configuration=SimpleNamespace(
            bootstrap_packaged_bundle=bootstrap,
            create_profile=forbidden,
            create_bundle=forbidden,
        ),
        research=SimpleNamespace(
            execution_context_factory=SimpleNamespace(freeze_selector=freeze)
        ),
    )

    if mismatch is None:
        assert benchmark._publish_explicit_bundle(container, root, profile) == (
            authority,
            context,
        )
    else:
        with pytest.raises(RuntimeError, match="bundle (identity|backend) mismatch"):
            benchmark._publish_explicit_bundle(container, root, profile)

    assert bootstrap_calls == [
        {
            "actor_id": "ORTOOLS-V4-PERFORMANCE",
            "bundle_path": source,
            "scope": "environment",
            "environment": "ortools-v4-performance",
        }
    ]
    assert freeze_calls == [
        {"bundle_revision_id": "CBR-TEST", "analysis_mode": "cascaded"}
    ]


def test_benchmark_emits_hash_bound_cold_warm_p50_p95(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    peak_memory_bytes = 64 * 1024 * 1024
    monkeypatch.setattr(
        benchmark, "process_lifetime_peak_rss_bytes", lambda: peak_memory_bytes
    )
    profile = benchmark._load_profile(
        root / "configs/research/ortools_performance_benchmark_v1.yaml"
    )
    profile["measurement"].update(
        {"coldSamples": 2, "warmSamples": 2, "warmupIterations": 0}
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        __import__("yaml").safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    operations = {
        name: _operation(
            name,
            measurement_ms=1.0,
            completed=benchmark._REQUIRED_COMPLETED_VARIANTS[name],
        )
        for name in benchmark._OPERATIONS
    }
    operations["alternateFiveVariants"] = _operation(
        "alternateFiveVariants",
        measurement_ms=1.0,
        completed=5,
        extra_resources={
            "campaignReceipt": {
                "coordinatorReceipt": {
                    "workerDeclaration": {
                        "shardWorkerCount": 5,
                        "cpSatWorkersPerSolve": 1,
                    }
                }
            }
        },
    )
    evidence = benchmark.run_benchmark(
        root,
        operations=operations,
        source_identity={"binding": "test"},
        profile_path=profile_path,
        benchmark_recipe_identities=_recipe_identities(),
    )

    assert evidence["status"] == "passed"
    assert evidence["researchAuthority"] == {
        "backendId": "ortools_cp_sat_v3",
        "qualificationAuthority": "cp_sat",
        "bundleId": "baloot_ortools_research_native_v3",
        "sourceBundleHash": evidence["researchAuthority"]["sourceBundleHash"],
        "runtimeBundleHash": None,
        "rulesHash": evidence["researchAuthority"]["rulesHash"],
    }
    assert evidence["evidenceSha256"] == canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidenceSha256"}
    )
    assert set(evidence["operations"]) == set(benchmark._OPERATIONS)
    for name, operation in evidence["operations"].items():
        assert operation["passed"] is True
        assert operation["cold"]["latencyMs"]["sampleCount"] == 2
        assert operation["warm"]["latencyMs"]["sampleCount"] == 2
        assert operation["cold"]["latencyMs"]["p50"] == 1.0
        assert operation["warm"]["latencyMs"]["p95"] == 1.0
        assert len(operation["semanticHash"]) == 64
        assert len(operation["contextHash"]) == 64
        assert len(operation["cold"]["executionReceiptHashes"]) == 2
        assert len(operation["cold"]["samples"]) == 2
        assert len(operation["warm"]["samples"]) == 2
        assert {
            sample["executionReceipt"]["sha256"]
            for sample in operation["cold"]["samples"]
        } == set(operation["cold"]["executionReceiptHashes"])
        assert set(operation["cold"]["stageTimingsSeconds"]) == set(
            benchmark._REQUIRED_STAGE_TIMINGS[name]
        )
        resource_gate = operation["resourceGate"]
        assert resource_gate["passed"] is True
        assert resource_gate["observed"]["peakMemoryBytes"] == peak_memory_bytes
        assert operation["sha256"] == canonical_sha256(
            {key: value for key, value in operation.items() if key != "sha256"}
        )
    assert evidence["operations"]["alternateFiveVariants"]["resources"][
        "completedVariants"
    ] == 5
    assert evidence["operations"]["alternateFiveVariants"]["cold"]["samples"][0][
        "resources"
    ]["campaignReceipt"]["coordinatorReceipt"]["workerDeclaration"] == {
        "shardWorkerCount": 5,
        "cpSatWorkersPerSolve": 1,
    }
    assert evidence["environment"]["pythonImplementation"]
    assert "processor" in evidence["environment"]
    assert "not evidence that the original request has a solution" in evidence[
        "proofBoundary"
    ]
    assert evidence["resourceMeasurementBoundary"] == {
        "conflictsAndBranches": "candidate_stage_receipts_only",
        "tailConstructionCounters": "unavailable",
        "maximumConflicts": "observed_gate_not_executor_enforcement_proof",
        "peakMemory": "process_lifetime_peak_rss",
        "processLifetimeMemoryProfileId": "late_game_exact_v1",
        "cache": "executor_receipt_cache_bytes",
        "cancellation": "cooperative_acknowledgement",
    }


def test_benchmark_fails_closed_on_identity_drift(root: Path, tmp_path: Path):
    profile = benchmark._load_profile(
        root / "configs/research/ortools_performance_benchmark_v1.yaml"
    )
    profile["measurement"].update(
        {"coldSamples": 2, "warmSamples": 2, "warmupIterations": 0}
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        __import__("yaml").safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    calls = 0

    def drifting_factory():
        def run():
            nonlocal calls
            calls += 1
            return {
                "semanticHash": canonical_sha256({"call": calls}),
                "contextHash": canonical_sha256({"context": "fixed"}),
                "stageTimings": {
                    stage: 0.001
                    for stage in benchmark._REQUIRED_STAGE_TIMINGS[
                        "validationEstimate"
                    ]
                },
                "resources": {"completedVariants": 0},
            }

        return run

    operations = {
        name: _operation(
            name,
            measurement_ms=1.0,
            completed=benchmark._REQUIRED_COMPLETED_VARIANTS[name],
        )
        for name in benchmark._OPERATIONS
    }
    operations["validationEstimate"] = benchmark.BenchmarkOperationV1(
        drifting_factory
    )

    import pytest

    with pytest.raises(RuntimeError, match="identity drifted") as error:
        benchmark.run_benchmark(
            root,
            operations=operations,
            source_identity={"binding": "test"},
            profile_path=profile_path,
            benchmark_recipe_identities=_recipe_identities(),
        )
    assert '"phase":"cold"' in str(error.value)
    assert '"semanticHash"' in str(error.value)
    assert '"coordinatorAttempts":[]' in str(error.value)




def test_real_benchmark_fails_closed_without_canonical_recipe(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="canonical flagship benchmark"):
        benchmark._load_benchmark_recipe_source(
            tmp_path,
            "missing.json",
            alternate=False,
        )


def test_benchmark_rejects_incomplete_stage_evidence(root: Path, tmp_path: Path):
    profile = benchmark._load_profile(
        root / "configs/research/ortools_performance_benchmark_v1.yaml"
    )
    profile["measurement"].update(
        {"coldSamples": 2, "warmSamples": 2, "warmupIterations": 0}
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        __import__("yaml").safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    operations = {
        name: _operation(
            name,
            measurement_ms=1.0,
            completed=benchmark._REQUIRED_COMPLETED_VARIANTS[name],
        )
        for name in benchmark._OPERATIONS
    }
    original = operations["alternateFirstQualifiedResult"]

    def incomplete_factory():
        operation = original.factory()

        def run():
            result = dict(operation())
            result["stageTimings"] = {"total": 0.001}
            return result

        return run

    operations["alternateFirstQualifiedResult"] = benchmark.BenchmarkOperationV1(
        incomplete_factory
    )

    import pytest

    with pytest.raises(ValueError, match="stage timing set is incomplete"):
        benchmark.run_benchmark(
            root,
            operations=operations,
            source_identity={"binding": "test"},
            profile_path=profile_path,
            benchmark_recipe_identities=_recipe_identities(),
        )


def test_benchmark_fails_governed_resource_gate(root: Path, tmp_path: Path):
    profile = benchmark._load_profile(
        root / "configs/research/ortools_performance_benchmark_v1.yaml"
    )
    profile["measurement"].update(
        {"coldSamples": 2, "warmSamples": 2, "warmupIterations": 0}
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        __import__("yaml").safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    operations = {
        name: _operation(
            name,
            measurement_ms=1.0,
            completed=benchmark._REQUIRED_COMPLETED_VARIANTS[name],
        )
        for name in benchmark._OPERATIONS
    }
    original = operations["alternateFirstQualifiedResult"]

    def excessive_conflicts_factory():
        operation = original.factory()

        def run():
            result = dict(operation())
            result["resources"] = {
                **result["resources"],
                "conflicts": 10_000_001,
            }
            return result

        return run

    operations["alternateFirstQualifiedResult"] = benchmark.BenchmarkOperationV1(
        excessive_conflicts_factory
    )

    evidence = benchmark.run_benchmark(
        root,
        operations=operations,
        source_identity={"binding": "test"},
        profile_path=profile_path,
        benchmark_recipe_identities=_recipe_identities(),
    )

    assert evidence["status"] == "failed"
    gate = evidence["operations"]["alternateFirstQualifiedResult"]["resourceGate"]
    assert gate["passed"] is False
    assert gate["limits"]["maximumConflicts"] == 10_000_000
    assert gate["observed"]["conflicts"] == 10_000_001
