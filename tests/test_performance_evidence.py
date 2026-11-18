from __future__ import annotations

import copy
import functools
import hashlib
import subprocess

import pytest

from scripts import benchmark_research_scenario_v4 as research_benchmark
from scripts import benchmark_ortools_research_v4 as ortools_benchmark
from scripts import generate_performance_evidence as evidence


_build_bundle = functools.lru_cache(maxsize=None)(evidence.build_bundle)


@functools.lru_cache(maxsize=1)
def _current_research_document():
    return research_benchmark.run_benchmark(evidence.ROOT)


def _synthetic_ortools_document(source):
    """Exercise the pinned legacy aggregator contract, not current performance."""
    def operation(name: str):
        semantic_hash = evidence.canonical_sha256({"operation": name})
        context_hash = evidence.canonical_sha256({"context": "ortools"})
        return ortools_benchmark.BenchmarkOperationV1(
            lambda: lambda: {
                "semanticHash": semantic_hash,
                "contextHash": context_hash,
                "measurementMs": 1.0,
                "stageTimings": {
                    stage: 0.001
                    for stage in ortools_benchmark._REQUIRED_STAGE_TIMINGS[name]
                },
                "resources": {
                    "completedVariants": (
                        ortools_benchmark._REQUIRED_COMPLETED_VARIANTS[name]
                    ),
                    **(
                        {
                            "normalizationBackendId": "ortools_cp_sat_v3",
                            "normalizationReceiptHash": "1" * 64,
                            "normalizationModelHash": "2" * 64,
                        }
                        if name == "exactNormalization"
                        else {}
                    ),
                    **(
                        {
                            "campaignReceipt": {
                                "coordinatorReceipt": {
                                    "workerDeclaration": {
                                        "shardWorkerCount": 5,
                                        "cpSatWorkersPerSolve": 1,
                                    }
                                }
                            }
                        }
                        if name == "alternateFiveVariants"
                        else {}
                    ),
                    **(
                        {"cancellationStatus": "cancelled"}
                        if name == "cancellationAcknowledgement"
                        else {}
                    ),
                },
            }
        )

    bundle = evidence.build_bundle(
        evidence.ROOT,
        bundle_path=evidence.ORTOOLS_BENCHMARK_BUNDLE_SOURCE,
    )
    resource_limits = ortools_benchmark._governed_resource_limits(bundle)
    operations = {
        name: ortools_benchmark._measure_operation(
            name, operation(name), cold_samples=2, warm_samples=2, warmups=0,
            budget_ms=2.0, resource_limits=resource_limits,
        )
        for name in ortools_benchmark._OPERATIONS
    }
    profile = {"kind": "synthetic_legacy_aggregator_contract_fixture"}
    core = {
        "schemaVersion": 1,
        "kind": "ortools_research_v4_performance_evidence",
        "status": (
            "passed" if all(item["passed"] for item in operations.values()) else "failed"
        ),
        "source": copy.deepcopy(source),
        "benchmarkProfile": {**profile, "sha256": evidence.canonical_sha256(profile)},
        "researchAuthority": {
            "backendId": "ortools_cp_sat_v3",
            "qualificationAuthority": "cp_sat",
            "bundleId": "baloot_ortools_research_native_v3",
            "sourceBundleHash": bundle.sha256,
            "rulesHash": bundle.rules.sha256,
        },
        "environment": {"measurement": "synthetic_unit_fixture"},
        "operations": operations,
        "resourceMeasurementBoundary": {
            "conflictsAndBranches": "candidate_stage_receipts_only",
            "tailConstructionCounters": "unavailable",
            "maximumConflicts": "observed_gate_not_executor_enforcement_proof",
            "peakMemory": "process_lifetime_peak_rss",
            "processLifetimeMemoryProfileId": resource_limits["profileId"],
            "cache": "executor_receipt_cache_bytes",
            "cancellation": "cooperative_acknowledgement",
        },
        "proofBoundary": "Synthetic contract fixture; no solver or performance qualification.",
        "scenarioRequests": {
            "original": {
                "source": (
                    "examples/research-scenario-v4-flagship-exact-infeasible.json"
                ),
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
        },
    }
    return {**core, "evidenceSha256": evidence.canonical_sha256(core)}


def _fresh_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    research_document=None,
):
    monkeypatch.setattr(evidence, "build_bundle", lambda *args, **kwargs: copy.deepcopy(_build_bundle(*args, **kwargs)))
    research_document = research_document or _current_research_document()
    source = copy.deepcopy(research_document["source"])
    # These test inputs exercise aggregation, never historical performance claims.
    documents = {
        evidence.SEARCH.name: {
            "status": "passed", "exactSearchObservation": {"completionStatus": "exact"},
            "hardware": {"measurement": "synthetic_unit_fixture"},
            "measurements": {}, "throughput": {},
        },
        evidence.BASELINE.name: {
            "status": "passed", "bundleHash": "a" * 64,
            "environment": {"measurement": "synthetic_unit_fixture"}, "measurements": {},
        },
        evidence.MEDIA.name: {
            "status": "passed",
            "media": [{"path": "synthetic.mp4", "width": 1080, "height": 1920}],
        },
    }
    documents[evidence.RESEARCH_V4.name] = copy.deepcopy(research_document)
    documents[evidence.ORTOOLS_RESEARCH_V4.name] = copy.deepcopy(
        _synthetic_ortools_document(source)
    )
    search = documents[evidence.SEARCH.name]
    search["source"] = dict(source)
    ortools_bundle = evidence.build_bundle(
        evidence.ROOT,
        bundle_path=evidence.SEARCH_ORTOOLS_BUNDLE_SOURCE,
    )
    search["engine"] = {
        "bundleSha256": ortools_bundle.sha256,
        "rulesSha256": ortools_bundle.rules.sha256,
    }
    search["researchAuthority"] = {
        "backendId": "ortools_cp_sat_v3",
        "qualificationAuthority": "cp_sat",
        "bundleSource": evidence.SEARCH_ORTOOLS_BUNDLE_SOURCE,
    }
    search["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in search.items() if key != "evidenceSha256"}
    )
    monkeypatch.setattr(
        evidence,
        "load",
        lambda path: copy.deepcopy(documents[path.name]),
    )
    monkeypatch.setattr(
        evidence,
        "git_identity",
        lambda _root: copy.deepcopy(source),
    )
    return documents


def test_synthetic_ortools_fixture_preserves_legacy_receipt_contract() -> None:
    value = _synthetic_ortools_document({"revision": "a" * 40})

    assert value["environment"]["measurement"] == "synthetic_unit_fixture"
    assert value["benchmarkProfile"]["kind"] == (
        "synthetic_legacy_aggregator_contract_fixture"
    )
    assert value["status"] == "passed"
    assert set(value["operations"]) == set(evidence.ORTOOLS_STAGE_TIMINGS)
    for name, operation in value["operations"].items():
        assert evidence._ortools_raw_sample_receipts_are_valid(name, operation)
    tampered = copy.deepcopy(value["operations"]["exactNormalization"])
    tampered["cold"]["samples"][0]["resources"][
        "normalizationBackendId"
    ] = "ortools_cp_sat_v2"
    assert not evidence._ortools_raw_sample_receipts_are_valid("exactNormalization", tampered)


def test_v21_performance_evidence_is_bounded_and_honest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_inputs(monkeypatch)
    value = evidence.build()

    assert value["status"] == "passed"
    assert value["ortoolsResearchV4"]["scenarioRequests"]["original"][
        "scenarioHash"
    ] == "b" * 64
    assert value["ortoolsResearchV4"]["resourceMeasurementBoundary"][
        "tailConstructionCounters"
    ] == "unavailable"
    assert value["ortoolsResearchV4"]["resourceMeasurementBoundary"][
        "processLifetimeMemoryProfileId"
    ] == "late_game_exact_v1"
    assert value["search"]["exactSearchObservation"]["completionStatus"] == "exact"
    assert (
        value["researchV4"]["summary"]["qualifiedDistinctGeneratedScenarioCount"] == 1
    )
    generated = value["researchV4"]["workloads"]["generatedSunHundredNoSalvage"]
    assert value["researchV4"]["summary"][
        "qualifiedDistinctGeneratedScenarioHashes"
    ] == [generated["scenarioIdentityHash"]]
    assert value["researchV4"]["summary"]["qualifiedDistinctGeneratedResultHashes"] == [
        generated["result"]["resultHash"]
    ]
    assert value["researchV4"]["workloads"]["exactNoSalvage"]["winningRootActions"] == 0
    assert (
        value["researchV4"]["workloads"]["exactUniqueSavingMove"]["winningRootActions"]
        == 1
    )
    assert value["media"]["aspectRatio"] == "9:16"
    assert value["promotionEvaluation"]["automaticPromotion"] is False
    assert value["promotionEvaluation"]["videoThirtyPercentTarget"] == (
        "not_yet_measured_on_locked_same_hardware"
    )


def test_current_research_v4_producer_contract_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    produced = _current_research_document()
    _fresh_inputs(monkeypatch, research_document=produced)

    value = evidence.build()

    assert value["researchV4"]["evidenceSha256"] == produced["evidenceSha256"]
    assert value["researchV4"]["summary"] == produced["summary"]


def test_v21_performance_evidence_rejects_failed_ortools_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    ortools = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    ortools["operations"]["alternateFiveVariants"]["passed"] = False
    operation = ortools["operations"]["alternateFiveVariants"]
    operation["sha256"] = evidence.canonical_sha256(
        {key: value for key, value in operation.items() if key != "sha256"}
    )
    ortools["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in ortools.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="OR-Tools Research V4"):
        evidence.build()


@pytest.mark.parametrize("tamper", ("compatibility_stage", "resource_gate"))
def test_v21_performance_evidence_rejects_incomplete_flagship_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    ortools = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    operation = ortools["operations"]["alternateFirstQualifiedResult"]
    if tamper == "compatibility_stage":
        del operation["cold"]["stageTimingsSeconds"]["compatibility"]
    else:
        operation["resourceGate"]["passed"] = False
    operation["sha256"] = evidence.canonical_sha256(
        {key: value for key, value in operation.items() if key != "sha256"}
    )
    ortools["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in ortools.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="OR-Tools Research V4"):
        evidence.build()


def test_v21_performance_evidence_rejects_non_vertical_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_inputs(monkeypatch)
    original = evidence.load

    def load_with_landscape(path):
        value = copy.deepcopy(original(path))
        if path.name == "media-probe.json":
            value["media"][0]["width"] = 1920
            value["media"][0]["height"] = 1080
        return value

    monkeypatch.setattr(evidence, "load", load_with_landscape)
    with pytest.raises(ValueError, match="non-vertical"):
        evidence.build()


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("source", "revision"),
        ("source", "tree"),
        ("engine", "bundleSha256"),
        ("engine", "rulesSha256"),
    ),
)
def test_v21_performance_evidence_rejects_stale_search_authority(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    search = documents[evidence.SEARCH.name]
    search[section][field] = "f" * 64
    search["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in search.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="search performance"):
        evidence.build()


def test_v21_performance_evidence_rejects_tampered_worktree_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    source = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "scope": evidence.SOURCE_SCOPE,
        "excludedPaths": list(evidence.SOURCE_EXCLUDED_PATHS),
        "binding": "git_head_plus_worktree_snapshot",
        "worktreeSnapshot": {
            "algorithm": "sha256",
            "scope": evidence.SOURCE_SCOPE,
            "excludedPaths": list(evidence.SOURCE_EXCLUDED_PATHS),
            "fileCount": 2,
            "statusSha256": "c" * 64,
            "sha256": "d" * 64,
        },
    }
    search = documents[evidence.SEARCH.name]
    search["source"] = copy.deepcopy(source)
    search["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in search.items() if key != "evidenceSha256"}
    )
    research = documents[evidence.RESEARCH_V4.name]
    research["source"] = copy.deepcopy(source)
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )
    ortools = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    ortools["source"] = copy.deepcopy(source)
    ortools["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in ortools.items() if key != "evidenceSha256"}
    )
    monkeypatch.setattr(
        evidence,
        "git_identity",
        lambda _root: copy.deepcopy(source),
    )
    assert evidence.build()["source"] == source

    search["source"]["worktreeSnapshot"]["sha256"] = "e" * 64
    search["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in search.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="source identity is stale"):
        evidence.build()


@pytest.mark.parametrize(
    "path",
    (
        ("source", "tree"),
        ("bundle", "sha256"),
        ("bundle", "semanticRuntimeSha256"),
        ("bundle", "rulesSha256"),
        ("proofBudgetProfileSha256",),
        ("status",),
    ),
)
def test_v21_performance_evidence_rejects_stale_research_v4_authority(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
) -> None:
    documents = _fresh_inputs(monkeypatch)
    research = documents[evidence.RESEARCH_V4.name]
    node = research
    for field in path[:-1]:
        node = node[field]
    node[path[-1]] = "failed" if path == ("status",) else "f" * 64
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="authority is stale"):
        evidence.build()


def test_v21_performance_evidence_rejects_stale_research_authority_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    research = documents[evidence.RESEARCH_V4.name]
    research["configurationSnapshot"]["contentSha256"] = "f" * 64
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="configuration snapshot is stale"):
        evidence.build()

    documents = _fresh_inputs(monkeypatch)
    research = documents[evidence.RESEARCH_V4.name]
    research["proofBudgetAuthority"]["registrySha256"] = "f" * 64
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="workloads are incomplete"):
        evidence.build()


def test_v21_performance_evidence_rejects_rehashed_snapshot_authority_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    research = documents[evidence.RESEARCH_V4.name]
    snapshot = research["configurationSnapshot"]
    snapshot["sourceEffectiveHash"] = "f" * 64
    snapshot["contentSha256"] = evidence.canonical_sha256(
        {key: value for key, value in snapshot.items() if key != "contentSha256"}
    )
    snapshot_bytes = evidence.canonical_json_bytes(snapshot)
    authority = research["configurationAuthority"]
    artifact_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    authority["snapshotArtifact"].update(
        {
            "uri": f"cas://sha256/{artifact_hash}",
            "sha256": artifact_hash,
            "bytes": len(snapshot_bytes),
        }
    )
    authority["sha256"] = evidence.canonical_sha256(
        {key: value for key, value in authority.items() if key != "sha256"}
    )
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="snapshot authority mismatch"):
        evidence.build()


def test_v21_performance_evidence_rejects_incomplete_research_v4_workloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    research = documents[evidence.RESEARCH_V4.name]
    research["workloads"]["generatedSunHundredNoSalvage"]["funnel"][
        "proofCompleted"
    ] = 0
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="workloads are incomplete"):
        evidence.build()


def test_v21_performance_evidence_rejects_tampered_scenario_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    research = documents[evidence.RESEARCH_V4.name]
    generated = research["workloads"]["generatedSunHundredNoSalvage"]
    generated["scenarioIdentityHash"] = "f" * 64
    research["summary"]["qualifiedDistinctGeneratedScenarioHashes"] = [
        generated["scenarioIdentityHash"]
    ]
    research["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in research.items() if key != "evidenceSha256"}
    )

    with pytest.raises(ValueError, match="workloads are incomplete"):
        evidence.build()


def test_v21_performance_evidence_rejects_missing_resource_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    benchmark = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    benchmark.pop("resourceMeasurementBoundary")
    benchmark["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in benchmark.items() if key != "evidenceSha256"}
    )

    with pytest.raises(
        ValueError, match="OR-Tools Research V4 performance authority is stale"
    ):
        evidence.build()


@pytest.mark.parametrize("tamper", ("samples", "worker_declaration"))
def test_v21_performance_evidence_rejects_incomplete_raw_ortools_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    benchmark = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    operation = benchmark["operations"]["alternateFiveVariants"]
    if tamper == "samples":
        operation["cold"].pop("samples")
    else:
        operation["cold"]["samples"][0]["resources"]["campaignReceipt"][
            "coordinatorReceipt"
        ]["workerDeclaration"]["shardWorkerCount"] = 1
    operation["sha256"] = evidence.canonical_sha256(
        {key: value for key, value in operation.items() if key != "sha256"}
    )
    benchmark["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in benchmark.items() if key != "evidenceSha256"}
    )

    with pytest.raises(
        ValueError, match="OR-Tools Research V4 performance authority is stale"
    ):
        evidence.build()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("normalizationBackendId", "ortools_cp_sat_v2"),
        ("normalizationReceiptHash", "f" * 63),
        ("normalizationModelHash", "f" * 63),
    ),
)
def test_v21_performance_evidence_rejects_exact_normalization_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    benchmark = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    operation = benchmark["operations"]["exactNormalization"]
    operation["cold"]["samples"][0]["resources"][field] = value
    operation["sha256"] = evidence.canonical_sha256(
        {key: item for key, item in operation.items() if key != "sha256"}
    )
    benchmark["evidenceSha256"] = evidence.canonical_sha256(
        {key: item for key, item in benchmark.items() if key != "evidenceSha256"}
    )

    with pytest.raises(
        ValueError, match="OR-Tools Research V4 performance authority is stale"
    ):
        evidence.build()


def test_v21_performance_evidence_rejects_unknown_memory_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _fresh_inputs(monkeypatch)
    benchmark = documents[evidence.ORTOOLS_RESEARCH_V4.name]
    benchmark["resourceMeasurementBoundary"][
        "processLifetimeMemoryProfileId"
    ] = "unknown"
    benchmark["evidenceSha256"] = evidence.canonical_sha256(
        {key: value for key, value in benchmark.items() if key != "evidenceSha256"}
    )

    with pytest.raises(
        ValueError, match="OR-Tools Research V4 performance authority is stale"
    ):
        evidence.build()


def test_git_identity_preserves_exact_head_only_for_a_clean_worktree(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def clean_run(command, **kwargs):
        calls.append((command, kwargs))
        arguments = tuple(command[1:])
        pathspecs = tuple(
            f":(exclude){path}**" for path in evidence.SOURCE_EXCLUDED_PATHS
        )
        outputs = {
            ("rev-parse", "HEAD"): ("1" * 40 + "\n").encode(),
            ("rev-parse", "HEAD^{tree}"): ("2" * 40 + "\n").encode(),
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                ".",
                *pathspecs,
            ): b"",
        }
        return subprocess.CompletedProcess(command, 0, stdout=outputs[arguments])

    monkeypatch.setattr(evidence.subprocess, "run", clean_run)

    assert evidence.git_identity(tmp_path) == {
        "revision": "1" * 40,
        "tree": "2" * 40,
        "scope": evidence.SOURCE_SCOPE,
        "excludedPaths": list(evidence.SOURCE_EXCLUDED_PATHS),
        "binding": "exact_git_head",
    }
    assert len(calls) == 3
    assert all(call[1]["cwd"] == tmp_path for call in calls)


def test_git_identity_binds_dirty_worktree_content_without_exact_head_claim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked = tmp_path / "tracked.txt"
    untracked = tmp_path / "new.txt"
    generated = tmp_path / "generated" / "contract.json"
    downstream = tmp_path / "release" / "sbom.cdx.json"
    generated.parent.mkdir()
    downstream.parent.mkdir()
    tracked.write_text("first\n", encoding="utf-8", newline="\n")
    untracked.write_text("new\n", encoding="utf-8", newline="\n")
    generated.write_text("first\n", encoding="utf-8", newline="\n")
    downstream.write_text("first\n", encoding="utf-8", newline="\n")
    index_hash = "3" * 40

    def dirty_run(command, **_kwargs):
        arguments = tuple(command[1:])
        if arguments == ("rev-parse", "HEAD"):
            output = ("1" * 40 + "\n").encode()
        elif arguments == ("rev-parse", "HEAD^{tree}"):
            output = ("2" * 40 + "\n").encode()
        elif arguments[0] == "status":
            output = b" M generated/contract.json\0 M tracked.txt\0?? new.txt\0"
        elif arguments[:2] == ("ls-files", "-z"):
            output = b"generated/contract.json\0new.txt\0tracked.txt\0"
        elif arguments[:3] == ("ls-files", "-s", "-z"):
            output = (
                f"100644 {index_hash} 0\tgenerated/contract.json\0"
                f"100644 {index_hash} 0\ttracked.txt\0"
            ).encode()
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(evidence.subprocess, "run", dirty_run)

    first = evidence.git_identity(tmp_path)
    repeated = evidence.git_identity(tmp_path)
    assert first == repeated
    assert first["binding"] == "git_head_plus_worktree_snapshot"
    assert first["scope"] == evidence.SOURCE_SCOPE
    assert first["excludedPaths"] == list(evidence.SOURCE_EXCLUDED_PATHS)
    assert first["worktreeSnapshot"]["scope"] == evidence.SOURCE_SCOPE
    assert first["worktreeSnapshot"]["excludedPaths"] == list(
        evidence.SOURCE_EXCLUDED_PATHS
    )
    assert first["worktreeSnapshot"]["fileCount"] == 3

    downstream.write_text("changed\n", encoding="utf-8", newline="\n")
    downstream_changed = evidence.git_identity(tmp_path)
    assert downstream_changed == first

    generated.write_text("changed\n", encoding="utf-8", newline="\n")
    generated_changed = evidence.git_identity(tmp_path)
    assert (
        generated_changed["worktreeSnapshot"]["sha256"]
        != first["worktreeSnapshot"]["sha256"]
    )

    tracked.write_text("changed\n", encoding="utf-8", newline="\n")
    changed = evidence.git_identity(tmp_path)
    assert changed["worktreeSnapshot"]["sha256"] != first["worktreeSnapshot"]["sha256"]
