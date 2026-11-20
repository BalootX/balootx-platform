from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import tempfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from baloot_platform.challenge.evidence_v3 import (
    DecisionLedgerRecordV1,
    DecisionLedgerV1,
)
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.finalist_validation import (
    FinalistValidationProfileV1,
)
from baloot_platform.challenge.constraint_deal_search import (
    ConstraintDealCandidateV1,
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.service.container import build_container
import scripts.run_combined_strategy_pilot as combined_pilot
from scripts.run_combined_strategy_pilot import (
    FINALIST_CAMPAIGN_ANALYSIS_MODE,
    FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION,
    FINALIST_CAMPAIGN_EXPERIMENT_ID,
    FINALIST_CAMPAIGN_PROFILE_ID,
    FINALIST_CAMPAIGN_PROFILE_REVISION,
    FINALIST_CAMPAIGN_WORKER_KIND,
    LEGACY_HYBRID_FINALIST_PROFILE_ID,
    _campaign_exit_code,
    _campaign_invalid_reason,
    _candidate_exception_is_fatal,
    _context_finalist_simulator,
    _decision_score_parity_failures,
    _direct_deep_decision_score_parity,
    _exact_search_backend_runtime_identity,
    _evaluate_finalist_candidate,
    _fatal_policy_defects,
    _finalist_campaign_recipe,
    _finalist_policy_gate_evidence,
    _observer_safety_passed,
    _validated_finalist_recipe_and_context,
    _policy_evidence,
    _require_qualification_profile,
    _require_current_strategic_auction_mirror,
    _run_finalist_campaign_supervised,
    _safe_campaign_output,
    _run_product_sequential_candidate_search,
    _source_fingerprint,
    _supervise_finalist_worker,
    _terminate_owned_process_tree,
    _validated_finalist_worker_evidence,
    _write_supervision_failure_evidence,
)


def test_finalist_campaign_is_exact_v3_and_context_bound(root, tmp_path) -> None:
    assert FINALIST_CAMPAIGN_PROFILE_REVISION == "2"
    assert FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION == 2
    assert FINALIST_CAMPAIGN_EXPERIMENT_ID.endswith("_campaign_v2")
    assert FINALIST_CAMPAIGN_WORKER_KIND.endswith("_worker_v2")
    container = build_container(
        root=root,
        writable_root=tmp_path / "runtime",
        database_url="",
    )
    try:
        bundle = container.configuration.resolve()["bundle"]
        payload = _finalist_campaign_recipe(
            container.research,
            bundle_revision_id=bundle["revisionId"],
            seed_start=502,
            seed_count=5,
            candidates_per_seed=4,
            analysis_mode="deep",
        )
        assert payload["schemaVersion"] == 3
        assert "profiles" not in payload
        assert payload["configuration"] == {
            "schemaVersion": 1,
            "bundleRevisionId": bundle["revisionId"],
        }
        assert payload["situation"]["contracts"] == ["sun"]
        assert payload["tactics"] == [{
            "kind": "only_winning_move",
            "mode": "qualification_gate",
            "minimumScore": 1,
            "weight": 1,
            "perspectives": [],
        }]
        assert payload["qualification"] == {
            "minimumTruth": "robust_best_known",
            "minimumConfidence": 0.75,
            "minimumSurroundingQuality": 70,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0,
        }
        assert payload["resources"]["auctionConstructionMode"] == (
            "actor_safe_agent_verified"
        )
        assert payload["resources"]["playConstructionMode"] == (
            "actor_safe_agent_verified"
        )
        assert "strategicAuction" not in payload["resources"]
        assert payload["resources"]["seedStart"] == 502
        assert payload["resources"]["seedCount"] == 5
        assert payload["resources"]["candidatesPerSeed"] == 4
        assert payload["resources"]["analysisMode"] == (
            FINALIST_CAMPAIGN_ANALYSIS_MODE
        )
        assert payload["output"]["resultCount"] == 20
        _, selected_context = (
            container.research.execution_context_factory.resolve_bundle_revision(
                bundle["revisionId"]
            )
        )
        assert payload["titleAr"] == selected_context.workflow_copy["research_title"]

        recipes_before = container.research.store.list_recipes(
            status=None,
            limit=10_000,
        )
        recipe, context = _validated_finalist_recipe_and_context(
            container.research,
            payload,
        )
        assert container.research.store.list_recipes(
            status=None,
            limit=10_000,
        ) == recipes_before
        assert recipe.resources["analysisMode"] == (
            FINALIST_CAMPAIGN_ANALYSIS_MODE
        )
        simulator = _context_finalist_simulator(context)
        assert recipe.resources["analysisMode"] == FINALIST_CAMPAIGN_ANALYSIS_MODE
        assert recipe.configuration["bundleRevisionId"] == bundle["revisionId"]
        assert context.bundle_revision_id == bundle["revisionId"]
        assert (
            context.analyzers[recipe.resources["analysisMode"]].descriptor.id
            == "staged_trace_analyzer_v1"
        )
        assert simulator.agent_profile.id == "finalist_candidate"
        assert simulator.analysis_policy_authority["mode"] == "explicit_governed"
        assert simulator.agent_qualification is not None
        assert simulator.agent_qualification.status == "configured_candidate"
        assert simulator.agent_qualification.expert_generation_allowed is False
        assert simulator.decision_agreement is not None
        agreement = simulator.decision_agreement
        assert agreement.descriptor.version == "1.3.0"
        assert agreement.profile.to_dict()["schemaVersion"] == 2
        assert agreement.profile.base_agent_profile_hash == simulator.profile_hash
        assert agreement.exact_primary_policy is True
        deep_profile = context.analyzers["deep"].profile
        reanalysis_profile = agreement.reanalysis_trace_analysis_profile
        assert reanalysis_profile is not None
        deep_identity_hash = context.analysis_identity_hashes["deep"]
        reanalysis_identity_hash = context.analysis_identity_hashes[
            "finalist_reanalysis"
        ]
        assert simulator.trace_analysis_profile is deep_profile
        assert simulator.trace_analysis_profile_hash == deep_profile.sha256
        assert simulator.analysis_identity_hash == deep_identity_hash
        assert agreement.primary_trace_analysis_profile is deep_profile
        assert agreement.primary_trace_analysis_profile_hash == deep_profile.sha256
        assert agreement.primary_analysis_identity_hash == deep_identity_hash
        assert agreement.reanalysis_trace_analysis_profile_hash == (
            reanalysis_profile.sha256
        )
        assert agreement.reanalysis_analysis_identity_hash == (
            reanalysis_identity_hash
        )
        assert reanalysis_profile.sha256 != deep_profile.sha256
        assert reanalysis_identity_hash != deep_identity_hash
        assert deep_profile.play_samples == 192
        assert reanalysis_profile.play_samples == 256
        assert simulator.primary_analysis_budget == {
            "requested_samples": deep_profile.play_samples,
            "exact_allocation_limit": deep_profile.exact_allocation_limit,
            "exact_search_max_remaining": deep_profile.exact_search_max_remaining,
            "maximum_sampling_attempts": deep_profile.maximum_sampling_attempts,
            "exact_search_max_nodes": deep_profile.exact_search_max_nodes,
            "exact_search_max_seconds": deep_profile.exact_search_max_seconds,
            "exact_search_max_depth": deep_profile.exact_search_max_depth,
            "exact_search_max_cache_entries": (
                deep_profile.exact_search_max_cache_entries
            ),
        }
        assert agreement.primary_analysis_budget == simulator.primary_analysis_budget
        assert agreement.independent_analysis_budget == {
            "requested_samples": 256,
            "exact_allocation_limit": 256,
            "exact_search_max_remaining": 9,
            "maximum_sampling_attempts": 6000,
            "exact_search_max_nodes": 120000,
            "exact_search_max_seconds": 0.8,
            "exact_search_max_depth": 32,
            "exact_search_max_cache_entries": 100000,
        }
        cohort_seeds = [
            agreement.actor_observer_seed(
                public_prefix_hash="1" * 64,
                decision_kind="play",
                acting_seat=Seat.NORTH,
                cohort_index=index,
            )
            for index in range(3)
        ]
        assert len(set(cohort_seeds)) == 3
        finalist_validation = FinalistValidationProfileV1.build(
            context.bundle.component("challenge")["finalist_validation"]
        )
        expected_agent_identity = canonical_sha256({
            "schemaVersion": 3,
            "authority": "research_primary_policy_v4",
            "descriptor": simulator.descriptor.to_dict(),
            "agentRelease": simulator.agent_release,
            "agentProfile": simulator.agent_profile.to_dict(),
            "generationTier": simulator.generation_tier,
            "finalistValidationProfile": finalist_validation.to_dict(),
            "decisionAgreement": {
                "descriptor": agreement.descriptor.to_dict(),
                "profile": agreement.profile.to_dict(),
                "profileHash": agreement.profile.sha256,
                "exactPrimaryPolicy": True,
            },
            "deepAnalysis": {
                "profile": deep_profile.to_dict(),
                "profileHash": deep_profile.sha256,
                "analysisIdentityHash": deep_identity_hash,
                "playBudget": dict(agreement.primary_analysis_budget),
            },
            "finalistReanalysis": {
                "profile": reanalysis_profile.to_dict(),
                "profileHash": reanalysis_profile.sha256,
                "analysisIdentityHash": reanalysis_identity_hash,
                "playBudget": dict(agreement.independent_analysis_budget),
            },
        })
        assert context.agent_identity_hash == expected_agent_identity
        expected_context_identity = canonical_sha256({
            "schemaVersion": 3,
            "authority": "research_primary_policy_v4",
            "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
            "semanticBundleHash": context.bundle.sha256,
            "domainHash": context.compiled_domain.sha256,
            "legacyDomainV2Hash": context.legacy_domain_v2.sha256,
            "compiledDomainHash": context.compiled_domain.sha256,
            "agentIdentityHash": context.agent_identity_hash,
            "analysisIdentityHashes": dict(context.analysis_identity_hashes),
            "scoreProfileHash": context.score_profile_hash,
            "decisionScoreProfileHash": context.decision_score_profile_hash,
            "comparisonProfileHash": context.comparison_profile_hash,
            "plugins": [dict(item) for item in context.plugin_identities],
        })
        assert context.context_hash == expected_context_identity
        assert simulator.play_strategy.convention == {}
        assert agreement.independent_play.convention == {}
        backend_identity = _exact_search_backend_runtime_identity(
            context.exact_search_backend
        )
        assert backend_identity["backendId"] == (
            context.exact_search_backend.descriptor.id
        )
        native = getattr(context.exact_search_backend, "native", None)
        if native is None:
            assert backend_identity["nativeSelected"] is False
            assert backend_identity["nativeLibrary"] is None
        else:
            loaded_library = Path(os.fsdecode(native.lib._name))
            native_identity = backend_identity["nativeLibrary"]
            assert backend_identity["nativeSelected"] is True
            assert native_identity["basename"] == loaded_library.name
            assert native_identity["sha256"] == hashlib.sha256(
                loaded_library.read_bytes()
            ).hexdigest()
            assert native_identity["abiVersion"] >= 3
            assert native_identity["capabilitiesMask"] == int(native.capabilities)
            assert str(loaded_library.parent) not in json.dumps(backend_identity)
    finally:
        container.shutdown()


def test_historical_hybrid_profile_is_explicitly_non_qualifying() -> None:
    assert FINALIST_CAMPAIGN_PROFILE_ID != LEGACY_HYBRID_FINALIST_PROFILE_ID
    _require_qualification_profile(FINALIST_CAMPAIGN_PROFILE_ID)
    with pytest.raises(
        ValueError,
        match="historical, non-qualifying evidence",
    ):
        _require_qualification_profile(LEGACY_HYBRID_FINALIST_PROFILE_ID)


@pytest.mark.real_clock
def test_finalist_evaluator_injects_no_strategic_bid_selector() -> None:
    observed: dict[str, object] = {}

    class Simulator:
        agent_release = "finalist-release"

        def simulate_preordered_deck(self, seed, deck_order, **kwargs):
            observed.update(seed=seed, deck_order=deck_order, **kwargs)
            raise InterruptedError("controlled bounded stop")

    candidate = SimpleNamespace(
        candidate_hash="a" * 64,
        model_hash="b" * 64,
        solver_status="optimal",
        deck_order=tuple(range(32)),
        provenance_dict=lambda: {"profileHash": "c" * 64},
    )
    result = _evaluate_finalist_candidate(
        research=object(),
        context=object(),
        recipe=SimpleNamespace(
            situation={"dealer": "east", "matchScore": [0, 0]}
        ),
        simulator=Simulator(),
        candidate=candidate,
        seed=505,
        candidate_index=1,
        cancel_check=lambda: False,
    )

    assert result["status"] == "bounded_timeout"
    assert "bid_selector" not in observed
    assert "scripted_bids" not in observed
    provenance = observed["deal_generator_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["auctionConstructionMode"] == "actor_safe_agent_verified"
    assert provenance["playConstructionMode"] == "actor_safe_agent_verified"
    assert provenance["auctionCpSatRole"] == "exact_required_action_rebind_only"
    assert provenance["playCpSatRole"] == "exact_required_action_rebind_only"


@pytest.mark.parametrize("analysis_mode", ("screening", "cascaded"))
def test_finalist_campaign_recipe_rejects_non_deep_analysis(
    root,
    tmp_path,
    analysis_mode,
) -> None:
    container = build_container(
        root=root,
        writable_root=tmp_path / "runtime",
        database_url="",
    )
    try:
        bundle = container.configuration.resolve()["bundle"]
        with pytest.raises(
            ValueError,
            match=(
                rf"{FINALIST_CAMPAIGN_PROFILE_ID} requires "
                rf"analysisMode={FINALIST_CAMPAIGN_ANALYSIS_MODE}"
            ),
        ):
            _finalist_campaign_recipe(
                container.research,
                bundle_revision_id=bundle["revisionId"],
                seed_start=502,
                seed_count=5,
                candidates_per_seed=4,
                analysis_mode=analysis_mode,
            )
    finally:
        container.shutdown()


def test_finalist_campaign_rejects_selected_deep_sample_mirror_drift() -> None:
    context = SimpleNamespace(
        analyzers={
            "deep": SimpleNamespace(profile=SimpleNamespace(bid_samples=191))
        }
    )

    with pytest.raises(
        ValueError,
        match="selected published bundle deep bidSamples",
    ):
        _require_current_strategic_auction_mirror(context)


@pytest.mark.parametrize("analysis_mode", ("screening", "cascaded"))
def test_finalist_campaign_rejects_non_deep_before_fingerprint_or_container(
    tmp_path,
    monkeypatch,
    analysis_mode,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / f"{analysis_mode}-rejected.json"

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("preflight guard ran too late")

    monkeypatch.setattr(combined_pilot, "_source_fingerprint", unexpected_call)
    monkeypatch.setattr(combined_pilot, "build_container", unexpected_call)
    args = SimpleNamespace(
        root=root,
        output=output,
        attempts=5,
        candidates_per_seed=4,
        maximum_seed_seconds=120.0,
        maximum_wall_seconds=600.0,
        seed_start=502,
        bundle_version="21.1.22",
        analysis_mode=analysis_mode,
    )

    assert combined_pilot._run_finalist_campaign(args) == 1
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "invalid"
    assert evidence["stopReason"] == "campaign_prelaunch_failure"
    assert evidence["fatalFailure"] == "campaign_prelaunch_failure"
    assert evidence["failure"] == {
        "stage": "argument_validation",
        "errorType": "ValueError",
        "message": (
            f"{FINALIST_CAMPAIGN_PROFILE_ID} requires "
            f"analysisMode={FINALIST_CAMPAIGN_ANALYSIS_MODE}"
        ),
    }
    assert evidence["sourceFingerprintStart"] is None
    assert evidence["sourceFingerprintEnd"] is None
    assert evidence["sourceFrozen"] is None
    assert evidence["temporaryRuntimeCleaned"] is True
    assert evidence["counts"]["seedsEvaluated"] == 0
    assert evidence["counts"]["candidatesEvaluated"] == 0


def test_transient_finalist_recipe_rejects_invalid_payload() -> None:
    research = SimpleNamespace(
        validate_recipe=lambda _payload: {
            "valid": False,
            "issues": [{"code": "invalid_selector"}],
        }
    )

    with pytest.raises(ValueError, match="invalid_selector"):
        _validated_finalist_recipe_and_context(research, {})


def test_legacy_v2_comparison_cli_is_explicitly_retired(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_combined_strategy_pilot.py",
            "--output",
            str(tmp_path / "retired.json"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        combined_pilot.main()

    assert raised.value.code == 2
    assert "benchmark_research_scenario_v4.py" in capsys.readouterr().err


def test_finalist_campaign_output_stays_outside_source_areas(tmp_path) -> None:
    root = tmp_path / "repository"
    for directory in (
        "src",
        "configs",
        "schemas",
        "scripts",
        "native",
        "release",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    for unsafe in (
        root / "src" / "evidence.json",
        root / "configs" / "evidence.json",
        root / "schemas" / "evidence.json",
        root / "scripts" / "evidence.json",
        root / "native" / "evidence.json",
        root / "pyproject.toml",
    ):
        with pytest.raises(ValueError, match="outside source"):
            _safe_campaign_output(root, unsafe)

    expected = (root / "release" / "evidence.json").resolve()
    assert _safe_campaign_output(root, expected) == expected


def test_source_fingerprint_includes_imported_runner_but_not_output(
    tmp_path,
) -> None:
    root = tmp_path / "repository"
    for directory in (
        "src/baloot_platform",
        "configs",
        "schemas",
        "scripts",
        "native/src",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src/baloot_platform/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs/example.yaml").write_text("value: 1\n", encoding="utf-8")
    (root / "schemas/example.json").write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "scripts/run_combined_strategy_pilot.py").write_text(
        "from scripts.run_strategic_play_pilot import helper\n",
        encoding="utf-8",
    )
    imported_runner = root / "scripts/run_strategic_play_pilot.py"
    imported_runner.write_text("helper = 1\n", encoding="utf-8")
    native_source = root / "native/src/baloot_native.cpp"
    native_source.write_text("constexpr int VERSION = 1;\n", encoding="utf-8")

    initial = _source_fingerprint(root)
    imported_runner.write_text("helper = 2\n", encoding="utf-8")
    changed = _source_fingerprint(root)
    assert changed["sha256"] != initial["sha256"]

    native_source.write_text("constexpr int VERSION = 2;\n", encoding="utf-8")
    native_changed = _source_fingerprint(root)
    assert native_changed["sha256"] != changed["sha256"]

    output = root / "release" / "evidence.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    assert _source_fingerprint(root) == native_changed


def test_native_runtime_identity_hashes_loaded_library_without_machine_path(
    tmp_path,
) -> None:
    library = tmp_path / "machine" / "baloot_native.dll"
    library.parent.mkdir()
    library.write_bytes(b"loaded-native-library")
    native = SimpleNamespace(
        lib=SimpleNamespace(_name=str(library)),
        capabilities=255,
        supports_v2=True,
        supports_v3=True,
        supports_compact_rollout=True,
        CAP_LEGAL_MOVES=1,
        CAP_BATCH_LEGAL_MOVES=2,
        CAP_CARD_POINTS=4,
        CAP_TRICK_EVALUATION=8,
        CAP_STATE_HASH=16,
        CAP_APPLY_PLAY=32,
        CAP_APPLY_UNDO=64,
        CAP_EXACT_SEARCH=128,
    )
    backend = SimpleNamespace(
        descriptor=SimpleNamespace(
            id="native_exact_v3_python_fallback_v1",
            version="1.0.0",
            capabilities=("exact_search", "native_bitboards"),
            deterministic=True,
        ),
        native=native,
    )

    identity = _exact_search_backend_runtime_identity(backend)

    assert identity["nativeSelected"] is True
    assert identity["nativeLibrary"] == {
        "basename": library.name,
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "abiVersion": 3,
        "abiVersions": [1, 2, 3],
        "capabilitiesMask": 255,
        "capabilities": [
            "apply_play",
            "apply_undo",
            "batch_legal_moves",
            "card_points",
            "exact_search",
            "legal_moves",
            "state_hash",
            "trick_evaluation",
            "compact_rollout",
        ],
    }
    assert str(library.parent) not in json.dumps(identity)


def test_native_runtime_identity_fails_closed_when_library_cannot_be_hashed(
    tmp_path,
) -> None:
    backend = SimpleNamespace(
        descriptor=SimpleNamespace(
            id="native_exact_v3_python_fallback_v1",
            version="1.0.0",
            capabilities=("exact_search", "native_bitboards"),
            deterministic=True,
        ),
        native=SimpleNamespace(
            lib=SimpleNamespace(_name=str(tmp_path / "missing.dll")),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="selected native exact-search library identity cannot be hashed",
    ):
        _exact_search_backend_runtime_identity(backend)


@pytest.mark.parametrize(
    ("source_frozen", "wall_passed", "seed_passed", "reason"),
    (
        (False, True, True, "source_fingerprint_changed"),
        (True, False, True, "campaign_time_bound_exceeded"),
        (True, True, False, "campaign_time_bound_exceeded"),
        (True, True, True, None),
    ),
)
def test_campaign_source_and_time_bounds_are_fatal(
    source_frozen,
    wall_passed,
    seed_passed,
    reason,
) -> None:
    assert (
        _campaign_invalid_reason(
            source_frozen=source_frozen,
            strict_wall_bound_passed=wall_passed,
            all_seed_bounds_passed=seed_passed,
        )
        == reason
    )
    if reason is not None:
        assert _campaign_exit_code("invalid") == 1
    else:
        assert _campaign_exit_code("negative") == 0


def _sleeping_test_process(
    descendant_pid_path: Path | None = None,
) -> subprocess.Popen[str]:
    options = {"text": True}
    if os.name == "nt":
        options["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        ) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        options["start_new_session"] = True
    command = [sys.executable, "-B", "-c", "import time; time.sleep(60)"]
    if descendant_pid_path is not None:
        command = [
            sys.executable,
            "-B",
            "-c",
            (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-B', '-c', "
                "'import time; time.sleep(60)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), "
                "encoding='utf-8'); time.sleep(60)"
            ),
            str(descendant_pid_path),
        ]
    return subprocess.Popen(command, **options)


def _pid_is_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            return stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            pass
    return True


@pytest.mark.real_clock
def test_hard_supervisor_preempts_a_real_noncooperative_worker_only(
    tmp_path,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    worker = _sleeping_test_process(descendant_pid_path)
    unrelated = _sleeping_test_process()
    try:
        wait_deadline = time.monotonic() + 5.0
        while not descendant_pid_path.is_file() and time.monotonic() < wait_deadline:
            time.sleep(0.02)
        assert descendant_pid_path.is_file()
        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        assert _pid_is_running(descendant_pid)
        progress = tmp_path / "progress.json"
        progress.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "phase": "candidate_evaluation",
                    "seed": 505,
                    "candidateIndex": 1,
                    "seedDeadlineMonotonic": time.monotonic() + 0.15,
                    "completedCandidates": [],
                    "completedSeeds": [],
                }
            ),
            encoding="utf-8",
        )
        started = time.monotonic()
        outcome = _supervise_finalist_worker(
            worker,
            progress_path=progress,
            global_deadline_monotonic=time.monotonic() + 60.0,
        )
        assert outcome["timedOut"] is True
        assert outcome["boundary"] == "seed_wall_bound"
        assert outcome["lastProgress"]["phase"] == "candidate_evaluation"
        assert outcome["termination"]["workerTerminated"] is True
        assert worker.poll() is not None
        assert unrelated.poll() is None
        descendant_wait_deadline = time.monotonic() + 5.0
        while (
            _pid_is_running(descendant_pid)
            and time.monotonic() < descendant_wait_deadline
        ):
            time.sleep(0.02)
        assert _pid_is_running(descendant_pid) is False
        assert time.monotonic() - started < 5.0
    finally:
        if worker.poll() is None:
            _terminate_owned_process_tree(worker)
        if unrelated.poll() is None:
            _terminate_owned_process_tree(unrelated)


@pytest.mark.real_clock
@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill fallback")
@pytest.mark.parametrize("taskkill_failure", ("timeout", "nonzero"))
def test_taskkill_failure_still_kills_and_waits_for_the_owned_root(
    monkeypatch,
    taskkill_failure,
) -> None:
    worker = _sleeping_test_process()

    def controlled_taskkill(*_args, **_kwargs):
        if taskkill_failure == "timeout":
            raise subprocess.TimeoutExpired("taskkill", timeout=0.01)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(combined_pilot.subprocess, "run", controlled_taskkill)
    try:
        outcome = _terminate_owned_process_tree(worker, wait_seconds=0.2)
        assert outcome["requested"] is True
        assert outcome["workerTerminated"] is True
        assert outcome["rootFallbackKillRequested"] is True
        assert outcome["processTreeTerminationFailure"] is not None
        assert worker.poll() is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5.0)


def test_hidden_finalist_worker_cannot_bypass_authenticated_supervision(
    tmp_path,
) -> None:
    args = SimpleNamespace(
        root=tmp_path,
        output=tmp_path / "evidence.json",
        _finalist_worker=True,
        _supervised_runtime=None,
        _supervision_progress=None,
        _supervision_token=None,
        _supervision_launch_id=None,
    )

    with pytest.raises(
        ValueError,
        match="incomplete finalist campaign supervision authority",
    ):
        _run_finalist_campaign_supervised(args)


@pytest.mark.real_clock
def test_deadline_wins_over_a_just_late_worker_exit(tmp_path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps({
            "schemaVersion": 1,
            "phase": "candidate_evaluation",
            "seedDeadlineMonotonic": 100.0,
        }),
        encoding="utf-8",
    )
    poll_calls = []

    class JustLateWorker:
        @staticmethod
        def poll():
            poll_calls.append(True)
            return 0

    outcome = _supervise_finalist_worker(
        JustLateWorker(),
        progress_path=progress,
        global_deadline_monotonic=200.0,
        monotonic=lambda: 101.0,
        sleep=lambda _seconds: pytest.fail("late exit must not sleep"),
        terminate=lambda _process: {
            "requested": False,
            "workerExitCode": 0,
            "workerTerminated": True,
        },
    )

    assert outcome["timedOut"] is True
    assert outcome["boundary"] == "seed_wall_bound"
    assert poll_calls == []


@pytest.mark.real_clock
def test_seed_result_deadline_releases_only_before_the_atomic_commit() -> None:
    timely = combined_pilot._seed_result_deadline_update(
        seed_deadline_monotonic=100.0,
        result_recorded_monotonic=99.999,
    )
    at_deadline = combined_pilot._seed_result_deadline_update(
        seed_deadline_monotonic=100.0,
        result_recorded_monotonic=100.0,
    )

    assert timely == {
        "seedDeadlineMonotonic": None,
        "seedResultRecordedMonotonic": 99.999,
        "seedDeadlineReleased": True,
    }
    assert at_deadline == {
        "seedDeadlineMonotonic": 100.0,
        "seedResultRecordedMonotonic": 100.0,
        "seedDeadlineReleased": False,
    }


@pytest.mark.real_clock
def test_late_seed_result_commit_remains_armed_and_invalid(tmp_path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps({
            "schemaVersion": 1,
            "phase": "seed_result_recorded",
            **combined_pilot._seed_result_deadline_update(
                seed_deadline_monotonic=100.0,
                result_recorded_monotonic=100.001,
            ),
            "completedCandidates": [],
            "completedSeeds": [{"seed": 517}],
        }),
        encoding="utf-8",
    )
    poll_calls = []

    class LateWorker:
        @staticmethod
        def poll():
            poll_calls.append(True)
            return 0

    outcome = _supervise_finalist_worker(
        LateWorker(),
        progress_path=progress,
        global_deadline_monotonic=200.0,
        monotonic=lambda: 100.001,
        sleep=lambda _seconds: pytest.fail("late seed result must not sleep"),
        terminate=lambda _process: {
            "requested": False,
            "workerExitCode": 0,
            "workerTerminated": True,
        },
    )

    assert outcome["timedOut"] is True
    assert outcome["boundary"] == "seed_wall_bound"
    assert outcome["lastProgress"]["seedDeadlineReleased"] is False
    assert poll_calls == []


def test_global_reserve_covers_termination_and_evidence_finalization() -> None:
    assert combined_pilot._FINALIST_GLOBAL_CLEANUP_RESERVE_SECONDS >= (
        3 * combined_pilot._FINALIST_PROCESS_TERMINATION_WAIT_SECONDS
        + combined_pilot._FINALIST_EVIDENCE_FINALIZATION_RESERVE_SECONDS
    )


@pytest.mark.real_clock
@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_atomic_checkpoint_retries_while_a_real_reader_denies_delete(
    tmp_path,
) -> None:
    import ctypes
    from ctypes import wintypes

    target = tmp_path / "supervision-progress.json"
    target.write_text('{"generation": 0}\n', encoding="utf-8")
    reader_opened = threading.Event()
    release_reader = threading.Event()
    reader_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    invalid_handle = wintypes.HANDLE(-1).value

    def hold_reader_without_share_delete() -> None:
        handle = create_file(
            str(target),
            0x80000000,
            0x00000001 | 0x00000002,
            None,
            3,
            0x00000080,
            None,
        )
        if handle == invalid_handle:
            reader_errors.append(ctypes.WinError(ctypes.get_last_error()))
            reader_opened.set()
            return
        try:
            reader_opened.set()
            release_reader.wait(timeout=5.0)
        finally:
            kernel32.CloseHandle(handle)

    def publish_checkpoint() -> None:
        try:
            combined_pilot._atomic_write_json(target, {"generation": 1})
        except BaseException as exc:
            writer_errors.append(exc)

    reader = threading.Thread(target=hold_reader_without_share_delete)
    writer = threading.Thread(target=publish_checkpoint)
    reader.start()
    assert reader_opened.wait(timeout=5.0)
    assert reader_errors == []
    writer.start()
    time.sleep(0.08)
    assert writer.is_alive()
    release_reader.set()
    reader.join(timeout=5.0)
    writer.join(timeout=5.0)

    assert reader.is_alive() is False
    assert writer.is_alive() is False
    assert writer_errors == []
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}
    assert list(tmp_path.glob(".supervision-progress.json.*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_exhausted_checkpoint_retry_removes_the_owned_temporary_file(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "supervision-progress.json"

    def deny_replace(*_args, **_kwargs):
        import ctypes

        raise ctypes.WinError(5)

    monkeypatch.setattr(combined_pilot.os, "replace", deny_replace)
    monkeypatch.setattr(combined_pilot, "_WINDOWS_ATOMIC_REPLACE_ATTEMPTS", 2)
    monkeypatch.setattr(combined_pilot.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError):
        combined_pilot._atomic_write_json(target, {"generation": 1})
    assert target.exists() is False
    assert list(tmp_path.glob(".supervision-progress.json.*.tmp")) == []


@pytest.mark.real_clock
def test_hard_timeout_receipt_is_invalid_hashed_and_has_no_policy_outcome(
    tmp_path,
) -> None:
    output = tmp_path / "timeout.json"
    source = {"schemaVersion": 1, "fileCount": 3, "sha256": "a" * 64}
    progress = {
        "schemaVersion": 1,
        "phase": "candidate_evaluation",
        "seed": 505,
        "candidateIndex": 2,
        "candidateHash": "b" * 64,
        "completedCandidates": [{
            "seed": 505,
            "candidateIndex": 1,
            "candidateHash": "c" * 64,
            "status": "rejected",
            "qualified": False,
            "tacticRetained": False,
            "fatalDefect": False,
            "semanticDefect": False,
            "observerLeakDefect": False,
        }],
        "completedSeeds": [],
    }
    args = SimpleNamespace(
        seed_start=505,
        attempts=1,
        candidates_per_seed=4,
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
    )
    result = _write_supervision_failure_evidence(
        args,
        output=output,
        source_start=source,
        source_end=source,
        progress=progress,
        elapsed_seconds=120.2,
        failure={
            "stage": "candidate_evaluation",
            "errorType": "CampaignHardTimeout",
            "message": "controlled hard timeout",
        },
        stop_reason="campaign_hard_timeout",
        temporary_runtime_cleaned=True,
        cleanup_failure=None,
        supervision={
            "schemaVersion": 1,
            "kind": "owned_finalist_campaign_process_v1",
            "hardBoundaryEnforced": True,
            "timedOut": True,
            "boundary": "seed_wall_bound",
            "originalFailurePrimary": True,
        },
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == result
    assert persisted["schemaVersion"] == FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION
    assert persisted["experimentId"] == FINALIST_CAMPAIGN_EXPERIMENT_ID
    assert persisted["qualificationProfile"]["revision"] == (
        FINALIST_CAMPAIGN_PROFILE_REVISION
    )
    assert persisted["status"] == "invalid"
    assert persisted["stopReason"] == "campaign_hard_timeout"
    assert persisted["policyOutcome"] is None
    assert persisted["policyOutcomeAssigned"] is False
    assert persisted["counts"]["qualified"] == 0
    assert persisted["candidateResults"] == []
    assert persisted["partialCandidateReceipts"] == progress["completedCandidates"]
    assert persisted["supervision"]["originalFailurePrimary"] is True
    assert persisted["temporaryRuntimeCleaned"] is True
    assert persisted["sha256"] == canonical_sha256({
        key: value for key, value in persisted.items() if key != "sha256"
    })


@pytest.mark.real_clock
def test_top_level_finalist_supervisor_writes_timeout_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "hard-timeout.json"
    commands: list[list[str]] = []

    class ExitedWorker:
        pid = 81001
        returncode = -9

        @staticmethod
        def poll():
            return -9

    def fake_popen(command, **_kwargs):
        commands.append(list(command))
        runtime = Path(command[command.index("--_supervised-runtime") + 1])
        progress = Path(command[command.index("--_supervision-progress") + 1])
        launch_id = command[command.index("--_supervision-launch-id") + 1]
        token = _kwargs["env"][combined_pilot._FINALIST_SUPERVISION_ENV]
        marker = (runtime / combined_pilot._FINALIST_SUPERVISION_MARKER).read_text(encoding="utf-8")
        assert token not in marker
        assert token not in command
        worker = SimpleNamespace(
            _finalist_worker=True,
            _supervised_runtime=runtime,
            _supervision_progress=progress,
            _supervision_token=token,
            _supervision_launch_id=launch_id,
        )
        assert combined_pilot._validated_supervised_runtime(worker) == (runtime, progress, launch_id)
        worker._supervision_token = "different"
        with pytest.raises(ValueError, match="invalid finalist campaign supervision token"):
            combined_pilot._validated_supervised_runtime(worker)
        return ExitedWorker()

    def fake_supervise(*_args, **_kwargs):
        return {
            "timedOut": True,
            "boundary": "seed_wall_bound",
            "lastProgress": {
                "schemaVersion": 1,
                "phase": "candidate_evaluation",
                "seed": 505,
                "candidateIndex": 1,
                "candidateHash": "d" * 64,
                "completedCandidates": [],
                "completedSeeds": [],
            },
            "termination": {
                "requested": True,
                "workerExitCode": -9,
                "workerTerminated": True,
            },
        }

    monkeypatch.setattr(combined_pilot.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        combined_pilot,
        "_supervise_finalist_worker",
        fake_supervise,
    )
    args = SimpleNamespace(
        root=root,
        output=output,
        qualification_profile=FINALIST_CAMPAIGN_PROFILE_ID,
        analysis_mode="deep",
        attempts=1,
        seed_start=505,
        candidates_per_seed=4,
        bundle_version="21.1.22",
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
        _finalist_worker=False,
    )

    assert _run_finalist_campaign_supervised(args) == 1
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "invalid"
    assert evidence["failure"]["stage"] == "candidate_evaluation"
    assert evidence["supervision"]["boundary"] == "seed_wall_bound"
    assert evidence["supervision"]["lateWorkerResultAccepted"] is False
    assert evidence["policyOutcomeAssigned"] is False
    assert evidence["temporaryRuntimeCleaned"] is True
    assert commands and "--_finalist-worker" in commands[0]
    assert commands[0][commands[0].index("--qualification-profile") + 1] == (
        FINALIST_CAMPAIGN_PROFILE_ID
    )


@lru_cache(maxsize=8)
def _worker_control_plane(
    authority_root: str,
    bundle_version: str,
    seed_start: int,
    attempts: int,
    candidates_per_seed: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="baloot-worker-fixture-") as runtime:
        container = build_container(
            root=Path(authority_root),
            writable_root=Path(runtime),
            database_url="",
        )
        try:
            bundle = container.configuration.resolve()["bundle"]
            assert bundle["version"] == bundle_version
            recipe, context = _validated_finalist_recipe_and_context(
                container.research,
                _finalist_campaign_recipe(
                    container.research,
                    bundle_revision_id=bundle["revisionId"],
                    seed_start=seed_start,
                    seed_count=attempts,
                    candidates_per_seed=candidates_per_seed,
                    analysis_mode=FINALIST_CAMPAIGN_ANALYSIS_MODE,
                )
            )
            simulator = _context_finalist_simulator(context)
            backend = _exact_search_backend_runtime_identity(
                context.exact_search_backend
            )
            return combined_pilot._finalist_control_plane(
                context=context,
                recipe=recipe,
                simulator=simulator,
                backend_runtime_identity=backend,
                bundle_version=bundle_version,
            )
        finally:
            container.shutdown()


@lru_cache(maxsize=1)
def _sequential_fixture_rules():
    return build_bundle().rules


def _sequential_fixture_spec(maximum_solutions: int) -> DealConstraintSpecV1:
    return DealConstraintSpecV1.build({
        "schemaVersion": 1,
        "id": "combined_finalist_runner_fixture_v1",
        "revision": "1",
        "enabled": True,
        "backend": "ortools_cp_sat_v1",
        "assumedBuyer": None,
        "purchaseSuits": [],
        "initialRequiredCards": {},
        "finalRequiredCards": {},
        "finalForbiddenCards": {},
        "initialSuitCountRanges": {},
        "finalSuitCountRanges": {},
        "projectPattern": None,
        "maximumSolutions": maximum_solutions,
        "maximumConflicts": 1_000,
        "safetyTimeoutSeconds": 1.0,
        "solverWorkers": 1,
        "seedOffset": 0,
        "tags": ["engine_validation_required"],
    })


def _fixture_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _SequentialFixtureGenerator:
    descriptor = OrToolsCpSatDealGeneratorV1.descriptor

    def __init__(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        dealer: Seat,
        candidate_count: int,
        continuation_exhausted: bool,
    ) -> None:
        rules = _sequential_fixture_rules()
        model_hash = _fixture_hash("combined-finalist-runner-model")
        backend_version = combined_pilot.ortools.__version__
        search_hash = OrToolsCpSatDealGeneratorV1.search_identity_document(
            spec=spec,
            model_hash=model_hash,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            backend_version=backend_version,
            no_good_cut_type="complete_deck_assignment_v1",
        )["sha256"]
        cuts: list[str] = []
        candidates: list[ConstraintDealCandidateV1] = []
        for ordinal in range(candidate_count + 1):
            deck = tuple(range(ordinal, 32)) + tuple(range(ordinal))
            candidate = OrToolsCpSatDealGeneratorV1._build_candidate(
                ordinal=ordinal,
                deck_order=deck,
                deck_commitment=hashlib.sha256(bytes(deck)).hexdigest(),
                ordering_score=900 + ordinal,
                structural_objective_score=42,
                profile_hash=spec.sha256,
                model_hash=model_hash,
                search_identity_hash=search_hash,
                backend_version=backend_version,
                no_good_cut_type="complete_deck_assignment_v1",
                preceding_no_good_cut_hashes=tuple(cuts),
            )
            candidates.append(candidate)
            if ordinal < candidate_count:
                cuts.append(
                    OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
                        candidate,
                        layout=None,
                    )["cutSha256"]
                )
        self.candidates = tuple(candidates[:candidate_count])
        self.witness = candidates[-1]
        self.cuts = tuple(cuts)
        self.continuation_exhausted = continuation_exhausted
        self.generate_calls: list[tuple[str, ...]] = []
        self.probe_calls: list[tuple[str, ...]] = []

    def generate_next(self, _spec, *, rejected_candidates, **_kwargs):
        self.generate_calls.append(tuple(
            candidate.candidate_hash for candidate in rejected_candidates
        ))
        ordinal = len(rejected_candidates)
        return self.candidates[ordinal] if ordinal < len(self.candidates) else None

    def probe_continuation(self, _spec, *, rejected_candidates, **_kwargs):
        self.probe_calls.append(tuple(
            candidate.candidate_hash for candidate in rejected_candidates
        ))
        authority = rejected_candidates[0] if rejected_candidates else self.witness
        exhausted = self.continuation_exhausted
        return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
            profile_hash=authority.profile_hash,
            model_hash=authority.model_hash,
            backend_version=authority.backend_version,
            solver_status="infeasible" if exhausted else "optimal",
            solution_ordinal=len(rejected_candidates),
            search_identity_hash=authority.search_identity_hash,
            preceding_no_good_cut_hashes=self.cuts[:len(rejected_candidates)],
            no_good_cut_type=authority.no_good_cut_type,
            exhausted=exhausted,
            witness=None if exhausted else self.witness,
            structural_objective_score=authority.structural_objective_score,
        )


@lru_cache(maxsize=32)
def _empty_sequential_search_receipt(
    seed: int,
    maximum_candidates: int,
) -> dict:
    spec = _sequential_fixture_spec(maximum_candidates)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        candidate_count=0,
        continuation_exhausted=True,
    )
    result = combined_pilot._run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=maximum_candidates,
        deadline_monotonic=time.monotonic() + 10.0,
        evaluate_candidate=lambda *_args: (_ for _ in ()).throw(
            AssertionError("an exhausted fixture cannot evaluate a candidate")
        ),
    )
    assert result["rows"] == []
    assert result["stopReason"] == "candidates_exhausted"
    return copy.deepcopy(result["sequentialSearchReceipt"])


def _complete_worker_evidence(
    args,
    *,
    launch_id: str,
    authority_root: Path | None = None,
) -> dict:
    authority = Path(authority_root or args.root).resolve()
    control_plane = copy.deepcopy(_worker_control_plane(
        str(authority),
        str(args.bundle_version),
        int(args.seed_start),
        int(args.attempts),
        int(args.candidates_per_seed),
    ))
    source = (
        _source_fingerprint(Path(args.root))
        if authority_root is not None
        else {"schemaVersion": 1, "fileCount": 1, "sha256": "a" * 64}
    )
    backend_identity = copy.deepcopy(control_plane["searchBackendRuntime"])
    seed_results = [
        {
            "schemaVersion": 1,
            "kind": "combined_finalist_seed_search_result_v1",
            "seed": seed,
            "generatedCandidateCount": 0,
            "evaluatedCandidateCount": 0,
            "tacticRetentionCount": 0,
            "distinctCandidateHashes": 0,
            "noPostHocRepair": True,
            "sequentialSearchReceipt": _empty_sequential_search_receipt(
                seed,
                int(args.candidates_per_seed),
            ),
            "continuationProbeWallSeconds": 0.0,
            "stopReason": "candidates_exhausted",
            "wallSeconds": 0.0,
            "cpuSeconds": 0.0,
        }
        for seed in range(
            int(args.seed_start),
            int(args.seed_start) + int(args.attempts),
        )
    ]
    stable = {
        "schemaVersion": FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION,
        "experimentId": FINALIST_CAMPAIGN_EXPERIMENT_ID,
        "qualificationProfile": {
            "id": FINALIST_CAMPAIGN_PROFILE_ID,
            "revision": FINALIST_CAMPAIGN_PROFILE_REVISION,
            "globallyEnabled": False,
        },
        "status": "negative",
        "stopReason": "seed_bound_exhausted",
        "fatalFailure": None,
        "failure": None,
        "cleanupFailure": None,
        "temporaryRuntimeCleaned": False,
        "bounds": {
            "seedStart": int(args.seed_start),
            "seedCount": int(args.attempts),
            "maximumCandidatesPerSeed": int(args.candidates_per_seed),
            "maximumWallSeconds": float(args.maximum_wall_seconds),
            "maximumSeedSeconds": float(args.maximum_seed_seconds),
        },
        "target": {"contract": "sun", "tactic": "only_winning_move"},
        "controlPlane": control_plane,
        "sourceFingerprintStart": source,
        "sourceFingerprintEnd": source,
        "sourceFrozen": True,
        "seedResults": seed_results,
        "candidateResults": [],
        "qualifiedCandidateStableHash": None,
        "terminalRepeat": None,
        "counts": {
            "seedsEvaluated": len(seed_results),
            "candidatesEvaluated": 0,
            "tacticRetained": 0,
            "qualified": 0,
            "semanticDefects": 0,
            "observerLeakDefects": 0,
            "distinctCandidateHashes": 0,
            "distinctTraceHashes": 0,
        },
        "cost": {
            "wallSeconds": 0.0,
            "cpuSeconds": 0.0,
            "strictWallBoundPassed": True,
            "allSeedBoundsPassed": True,
        },
        "runtime": {
            "platform": combined_pilot.platform.platform(),
            "python": combined_pilot.platform.python_version(),
            "ortools": combined_pilot.ortools.__version__,
            "logicalCpuCount": combined_pilot.os.cpu_count(),
            "exactSearchBackend": backend_identity,
        },
        "supervision": {
            "schemaVersion": 2,
            "kind": FINALIST_CAMPAIGN_WORKER_KIND,
            "launchId": launch_id,
            "runtimeOwner": "supervisor_process",
            "parentRuntimeFinalized": False,
        },
        "productSourceChangedByRunner": False,
        "gatesWeakened": False,
        "provisionalTruthRelabeled": False,
    }
    return {**stable, "sha256": canonical_sha256(stable)}


def _validate_worker_fixture(
    output: Path,
    *,
    args,
    launch_id: str,
    expected_control_plane: dict,
    authoritative_control_plane: dict | None = None,
    worker_exit_code: int | None = None,
    parent_observation: dict | None = None,
    parent_source_start: dict | None = None,
    parent_source_end: dict | None = None,
) -> dict:
    payload = json.loads(output.read_text(encoding="utf-8"))
    return _validated_finalist_worker_evidence(
        output,
        args=args,
        launch_id=launch_id,
        expected_control_plane=expected_control_plane,
        authoritative_control_plane=(
            authoritative_control_plane or expected_control_plane
        ),
        parent_source_start=(
            parent_source_start or payload["sourceFingerprintStart"]
        ),
        parent_source_end=parent_source_end or payload["sourceFingerprintEnd"],
        worker_exit_code=(
            _campaign_exit_code(payload["status"])
            if worker_exit_code is None
            else worker_exit_code
        ),
        parent_observation=parent_observation,
    )


@pytest.mark.real_clock
def test_worker_evidence_must_be_checksum_valid_complete_and_launch_bound(
    tmp_path,
    root,
) -> None:
    output = tmp_path / "worker-evidence.json"
    args = SimpleNamespace(
        root=root,
        bundle_version="21.1.22",
        seed_start=505,
        attempts=1,
        candidates_per_seed=4,
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
    )
    launch_id = "fresh-launch"
    payload = _complete_worker_evidence(args, launch_id=launch_id)
    combined_pilot._atomic_write_json(output, payload)

    assert _validate_worker_fixture(
        output,
        args=args,
        launch_id=launch_id,
        expected_control_plane=payload["controlPlane"],
    ) == payload
    assert payload["schemaVersion"] == 2
    assert payload["experimentId"] == FINALIST_CAMPAIGN_EXPERIMENT_ID
    assert payload["qualificationProfile"]["revision"] == "2"
    assert payload["supervision"]["schemaVersion"] == 2
    assert payload["supervision"]["kind"] == FINALIST_CAMPAIGN_WORKER_KIND
    authorities = payload["controlPlane"]["analysisAuthorities"]
    assert authorities["primary"]["analysisBudget"]["requested_samples"] == 192
    assert (
        authorities["reanalysis"]["analysisBudget"]["requested_samples"]
        == 256
    )
    assert authorities["primary"]["traceAnalysisProfileHash"] != (
        authorities["reanalysis"]["traceAnalysisProfileHash"]
    )
    assert authorities["primary"]["analysisIdentityHash"] != (
        authorities["reanalysis"]["analysisIdentityHash"]
    )
    assert payload["controlPlane"]["finalistAgreementDescriptor"] == (
        FinalistAgentAgreementV1.descriptor.to_dict()
    )
    with pytest.raises(RuntimeError, match="stale or launch-unbound"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id="different-launch",
            expected_control_plane=payload["controlPlane"],
        )

    stale = copy.deepcopy(payload)
    stale["supervision"]["launchId"] = "stale-launch"
    stale["sha256"] = canonical_sha256({
        key: value for key, value in stale.items() if key != "sha256"
    })
    combined_pilot._atomic_write_json(output, stale)
    with pytest.raises(RuntimeError, match="stale or launch-unbound"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id=launch_id,
            expected_control_plane=payload["controlPlane"],
        )

    drifted_authority = copy.deepcopy(payload)
    drifted_authority["runtime"]["exactSearchBackend"]["backendVersion"] = (
        "drifted"
    )
    drifted_authority["sha256"] = canonical_sha256({
        key: value
        for key, value in drifted_authority.items()
        if key != "sha256"
    })
    combined_pilot._atomic_write_json(output, drifted_authority)
    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id=launch_id,
            expected_control_plane=payload["controlPlane"],
        )

    stale_authority = copy.deepcopy(payload)
    stale_authority["controlPlane"]["finalistAgreementDescriptor"][
        "version"
    ] = "1.2.0"
    stale_authority["sha256"] = canonical_sha256({
        key: value
        for key, value in stale_authority.items()
        if key != "sha256"
    })
    combined_pilot._atomic_write_json(output, stale_authority)
    with pytest.raises(RuntimeError, match="control-plane authority is stale"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id=launch_id,
            expected_control_plane=stale_authority["controlPlane"],
        )

    stale_candidate = copy.deepcopy(payload)
    descriptor = payload["controlPlane"]["finalistAgreementDescriptor"]
    stale_candidate["candidateResults"] = [{
        "selectionAndVerification": {
            "qualificationProfileRevision": "1",
        },
        "policyEvidence": {"agreementDescriptor": descriptor},
        "decisionAgreement": {"descriptor": descriptor},
    }]
    stale_candidate["sha256"] = canonical_sha256({
        key: value
        for key, value in stale_candidate.items()
        if key != "sha256"
    })
    combined_pilot._atomic_write_json(output, stale_candidate)
    with pytest.raises(RuntimeError, match="candidate (receipt|authority)"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id=launch_id,
            expected_control_plane=payload["controlPlane"],
        )

    corrupt = copy.deepcopy(payload)
    corrupt["sha256"] = "0" * 64
    combined_pilot._atomic_write_json(output, corrupt)
    with pytest.raises(RuntimeError, match="evidence hash is invalid"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id=launch_id,
            expected_control_plane=payload["controlPlane"],
        )


def test_parent_rebuilds_exact_control_plane_from_transient_selector_and_cas(
    tmp_path,
    root,
    monkeypatch,
) -> None:
    runtime = tmp_path / "runtime"
    container = build_container(
        root=root,
        writable_root=runtime,
        database_url="",
    )
    try:
        bundle = container.configuration.resolve()["bundle"]
        recipe, context = _validated_finalist_recipe_and_context(
            container.research,
            _finalist_campaign_recipe(
                container.research,
                bundle_revision_id=bundle["revisionId"],
                seed_start=505,
                seed_count=1,
                candidates_per_seed=4,
                analysis_mode=FINALIST_CAMPAIGN_ANALYSIS_MODE,
            )
        )
        simulator = _context_finalist_simulator(context)
        backend = _exact_search_backend_runtime_identity(
            context.exact_search_backend
        )
        control = combined_pilot._finalist_control_plane(
            context=context,
            recipe=recipe,
            simulator=simulator,
            backend_runtime_identity=backend,
            bundle_version=bundle["version"],
        )
    finally:
        container.shutdown()

    source = {"schemaVersion": 1, "fileCount": 1, "sha256": "a" * 64}
    monkeypatch.setattr(
        combined_pilot,
        "_source_fingerprint",
        lambda _root: copy.deepcopy(source),
    )
    args = SimpleNamespace(
        root=root,
        bundle_version="21.1.22",
        seed_start=505,
        attempts=1,
        candidates_per_seed=4,
    )
    assert combined_pilot._validated_runtime_control_plane(
        root=root,
        runtime=runtime,
        args=args,
        control_plane=control,
        parent_source_start=source,
    ) == control

    forged = copy.deepcopy(control)
    identities = forged["configurationAuthority"]["identities"]
    identity = identities[next(iter(sorted(identities)))]
    identity["sha256"] = "f" * 64
    configuration = forged["configurationAuthority"]
    configuration["sha256"] = canonical_sha256({
        key: value for key, value in configuration.items() if key != "sha256"
    })
    forged["configurationAuthorityHash"] = configuration["sha256"]
    with pytest.raises((RuntimeError, ValueError)):
        combined_pilot._validated_runtime_control_plane(
            root=root,
            runtime=runtime,
            args=args,
            control_plane=forged,
            parent_source_start=source,
        )


def _rehash_receipt(payload: dict) -> dict:
    payload["sha256"] = canonical_sha256({
        key: value for key, value in payload.items() if key != "sha256"
    })
    return payload


def _qualified_worker_candidate(control_plane: dict, *, seed: int) -> dict:
    authorities = control_plane["analysisAuthorities"]
    agreement_descriptor = control_plane["finalistAgreementDescriptor"]
    agreement_profile_hash = control_plane["finalistAgreementProfileHash"]
    rules_hash = control_plane["configurationAuthority"]["identities"][
        "rules"
    ]["sha256"]
    primary = authorities["primary"]
    ledger_records = (
        DecisionLedgerRecordV1(
            "PLAY-1",
            1,
            Seat.EAST,
            "play",
            "playing",
            "e" * 64,
            "1" * 64,
            ({"code": "S7"}, {"code": "S8"}),
            {"code": "S8"},
            False,
            "2" * 64,
        ),
        DecisionLedgerRecordV1(
            "PLAY-2",
            2,
            Seat.SOUTH,
            "play",
            "playing",
            "f" * 64,
            "3" * 64,
            ({"code": "H7"}, {"code": "H8"}),
            {"code": "H8"},
            False,
            "4" * 64,
        ),
        DecisionLedgerRecordV1(
            "PLAY-3",
            3,
            Seat.WEST,
            "play",
            "playing",
            "b" * 64,
            "5" * 64,
            ({"code": "D7"},),
            {"code": "D7"},
            True,
            "6" * 64,
        ),
    )
    ledger = DecisionLedgerV1.build("4" * 64, ledger_records)
    non_forced_records = tuple(
        record for record in ledger_records if not record.forced
    )
    policy_rows = []
    parity_rows = []
    for record in non_forced_records:
        recommendation_hash = canonical_sha256(dict(record.chosen_action))
        evidence_hash = canonical_sha256({
            "authority": "canonical_primary_decision_score_evidence_v1",
            "decisionId": record.decision_id,
        })
        rows_hash = canonical_sha256({
            "authority": "canonical_decision_score_rows_v1",
            "decisionId": record.decision_id,
        })
        cohorts = []
        for index in range(3):
            authority = authorities[
                "primary" if index == 0 else "reanalysis"
            ]
            cohorts.append({
                "cohortIndex": index,
                "analysisAuthorityRole": authority["role"],
                "traceAnalysisProfileHash": authority[
                    "traceAnalysisProfileHash"
                ],
                "analysisIdentityHash": authority["analysisIdentityHash"],
                "analysisBudget": copy.deepcopy(authority["analysisBudget"]),
                "seed": combined_pilot.derive_actor_observer_cohort_seed_v1(
                    public_prefix_hash=record.public_prefix_hash,
                    decision_kind=record.decision_kind,
                    acting_seat=record.acting_seat.name.lower(),
                    cohort_authority_hash=authority["analysisIdentityHash"],
                    cohort_index=index,
                    governed_salt=(
                        combined_pilot.PRIMARY_DECISION_COHORT_SALT_V1
                    ),
                ),
                "recommendedActionHash": recommendation_hash,
            })
        policy_rows.append({
            "decisionId": record.decision_id,
            "ordinal": record.ordinal,
            "decisionKind": record.decision_kind,
            "forced": False,
            "actingSeat": record.acting_seat.name.lower(),
            "publicPrefixHash": record.public_prefix_hash,
            "legalActionSetHash": combined_pilot.legal_action_set_hash_v1(
                record.legal_actions
            ),
            "rulesHash": rules_hash,
            "selectedAction": dict(record.chosen_action),
            "selectedActionMatchesTrace": True,
            "primaryDecisionScoreEvidenceHash": evidence_hash,
            "analysisAuthorityRole": primary["role"],
            "traceAnalysisProfileHash": primary["traceAnalysisProfileHash"],
            "analysisIdentityHash": primary["analysisIdentityHash"],
            "analysisBudget": copy.deepcopy(primary["analysisBudget"]),
            "cohortCount": 3,
            "cohortSupport": cohorts,
            "agreementStatus": "strict_confirmed",
            "qualificationBlocked": False,
            "featuredTruthEligible": True,
            "agreementCount": 3,
            "majorityActionHash": recommendation_hash,
            "failures": [],
        })
        parity_rows.append({
            "decisionId": record.decision_id,
            "ordinal": record.ordinal,
            "decisionKind": record.decision_kind,
            "forced": False,
            "generationEvidenceHash": evidence_hash,
            "auditEvidenceHash": evidence_hash,
            "generationRowsHash": rows_hash,
            "auditRowsHash": rows_hash,
            "matched": True,
        })
    forced_record = next(record for record in ledger_records if record.forced)
    forced_proof_core = {
        "schemaVersion": 1,
        "proofType": "authoritative_forced_legal_action_v1",
        "decisionKind": forced_record.decision_kind,
        "actor": forced_record.acting_seat.name.lower(),
        "publicPrefixHash": forced_record.public_prefix_hash,
        "rulesHash": rules_hash,
        "legalActionCount": 1,
        "legalActionSetHash": combined_pilot.legal_action_set_hash_v1(
            forced_record.legal_actions
        ),
        "selectedAction": dict(forced_record.chosen_action),
        "selectedActionHash": canonical_sha256(
            dict(forced_record.chosen_action)
        ),
    }
    forced_proof = {
        **forced_proof_core,
        "sha256": canonical_sha256(forced_proof_core),
    }
    forced_policy_row = {
        "decisionId": forced_record.decision_id,
        "ordinal": forced_record.ordinal,
        "decisionKind": forced_record.decision_kind,
        "forced": True,
        "actingSeat": forced_record.acting_seat.name.lower(),
        "publicPrefixHash": forced_record.public_prefix_hash,
        "legalActionSetHash": forced_proof["legalActionSetHash"],
        "rulesHash": rules_hash,
        "selectedAction": dict(forced_record.chosen_action),
        "selectedActionMatchesTrace": True,
        "forcedLegalityProof": forced_proof,
        "forcedLegalityProofHash": forced_proof["sha256"],
    }
    parity_rows.append({
        "decisionId": forced_record.decision_id,
        "ordinal": forced_record.ordinal,
        "decisionKind": forced_record.decision_kind,
        "forced": True,
        "generationEvidenceHash": forced_proof["sha256"],
        "auditEvidenceHash": forced_proof["sha256"],
        "generationRowsHash": None,
        "auditRowsHash": None,
        "matched": True,
    })
    policy_core = {
        "agreementDescriptor": copy.deepcopy(agreement_descriptor),
        "agreementProfileHash": agreement_profile_hash,
        "liveTraceAgreementDescriptor": copy.deepcopy(agreement_descriptor),
        "liveTraceAgreementProfileHash": agreement_profile_hash,
        "analysisAuthorities": {
            role: {
                "role": authority["role"],
                "profileHash": authority["traceAnalysisProfileHash"],
                "identityHash": authority["analysisIdentityHash"],
                "budget": copy.deepcopy(authority["analysisBudget"]),
            }
            for role, authority in authorities.items()
        },
        "decisionLedgerHash": ledger.sha256,
        "rulesHash": rules_hash,
        "bids": [],
        "plays": policy_rows,
        "nonForcedBidCount": 0,
        "nonForcedPlayCount": len(policy_rows),
        "forcedDecisionCount": 1,
        "forcedDecisions": [forced_policy_row],
        "competentDecisionIds": [],
        "unresolvedDecisionIds": [],
        "featuredTruthBlockedDecisionIds": [],
        "stabilityResolved": True,
        "zeroObserverLeaks": True,
        "observerSafetyPassed": True,
        "oneSharedActionAcrossIndistinguishableWorlds": True,
        "primaryPolicyValid": True,
    }
    policy = {**policy_core, "sha256": canonical_sha256(policy_core)}
    parity_core = {
        "schemaVersion": 1,
        "authority": "finalist_primary_decision_score_parity_v1",
        "traceHash": "3" * 64,
        "deepAnalysisHash": "a" * 64,
        "deepAnalysisProfileHash": primary["traceAnalysisProfileHash"],
        "deepAnalysisIdentityHash": primary["analysisIdentityHash"],
        "analysisBackend": control_plane["searchBackendRuntime"]["backendId"],
        "decisionCount": len(ledger_records),
        "nonForcedDecisionCount": len(non_forced_records),
        "forcedDecisionCount": 1,
        "matchedDecisionCount": len(ledger_records),
        "matched": True,
        "mismatches": [],
        "decisions": parity_rows,
        "crossLaneComparison": False,
        "structuralObjectiveCompared": False,
        "officialBalootScoreCompared": False,
        "researchContextHash": control_plane["contextHash"],
    }
    parity = {**parity_core, "sha256": canonical_sha256(parity_core)}
    gates = {
        field: True
        for field in (
            "semanticReplay",
            "zeroSemanticDefects",
            "auctionCpSatExactTraceRebind",
            "playCpSatExactTraceRebind",
            "fullCpSatReplayParity",
            "decisionScoreParity",
            "exactContract",
            "tacticRetained",
            "zeroObserverLeaks",
            "observerSafety",
            "sharedActionAcrossWorlds",
            "finalistPrimaryAuctionAndPlay",
            "finalistAgreementResolved",
            "auditComplete",
            "informationSetComplete",
            "sequentialResponseComplete",
            "qualityAtLeast70",
            "zeroRecommendationMismatches",
            "zeroMaterialCounterexamples",
            "regretGate",
            "decisionSelection",
            "scoreProfile",
            "ordinaryQualification",
        )
    }
    stable = {
        "seed": seed,
        "candidateIndex": 1,
        "candidateHash": "1" * 64,
        "dealModelHash": "2" * 64,
        "dealSolverStatus": "optimal",
        "traceHash": "3" * 64,
        "rulesHash": rules_hash,
        "decisionLedger": ledger.to_dict(),
        "gameHash": "4" * 64,
        "deckCommitment": "5" * 64,
        "auctionPlanHash": "6" * 64,
        "projectPlanHash": "7" * 64,
        "playPlanHash": "8" * 64,
        "playModelHash": "9" * 64,
        "analysisHash": "a" * 64,
        "decisionScoreParityHash": parity["sha256"],
        "auditHash": "c" * 64,
        "decisionSelectionHash": "d" * 64,
        "qualified": True,
        "qualification": {
            "qualified": True,
            "truthLabel": "robust_best_known",
            "minimumSurroundingQuality": 70.0,
            "failures": [],
        },
        "score": {"passed": True},
        "gates": gates,
        "tacticRetained": True,
        "maximumBidRegret": 0.0,
        "maximumPlayRegret": 0.0,
        "auditProblems": [],
        "recipeTargetMismatches": [],
        "semanticMismatches": [],
        "decisionScoreParity": parity,
        "policyEvidence": policy,
        "selectionAndVerification": {
            "qualificationProfileId": FINALIST_CAMPAIGN_PROFILE_ID,
            "qualificationProfileRevision": FINALIST_CAMPAIGN_PROFILE_REVISION,
            "auctionSelectionAuthority": (
                "finalist_multi_cohort_agent_primary_policy"
            ),
            "playSelectionAuthority": (
                "finalist_multi_cohort_agent_primary_policy"
            ),
            "auctionCpSatVerification": {
                "role": "exact_required_action_rebind_only",
                "planHash": "6" * 64,
                "solverStatus": "optimal",
            },
            "playCpSatVerification": {
                "role": "exact_required_action_rebind_only",
                "projectPlanHash": "7" * 64,
                "planHash": "8" * 64,
                "solverStatus": "optimal",
            },
            "postHocActionRepair": False,
        },
        "fatalPolicyDefects": [],
        "decisionAgreement": {
            "descriptor": copy.deepcopy(agreement_descriptor),
            "profileHash": agreement_profile_hash,
            "decisionCount": len(non_forced_records),
            "unresolvedCount": 0,
        },
    }
    return {
        **stable,
        "stableHash": canonical_sha256(stable),
        "wallSeconds": 0.01,
        "cpuSeconds": 0.01,
        "fatalDefect": False,
    }


def _qualified_worker_evidence(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    seed = int(result["bounds"]["seedStart"])
    maximum_candidates = int(result["bounds"]["maximumCandidatesPerSeed"])
    spec = _sequential_fixture_spec(maximum_candidates)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    generated_candidate = generator.candidates[0]
    candidate = _qualified_worker_candidate(
        result["controlPlane"],
        seed=seed,
    )
    candidate.update({
        "candidateHash": generated_candidate.candidate_hash,
        "dealModelHash": generated_candidate.model_hash,
        "dealSolverStatus": generated_candidate.solver_status,
        "deckCommitment": generated_candidate.deck_commitment,
    })
    candidate["stableHash"] = canonical_sha256({
        key: value
        for key, value in candidate.items()
        if key not in {"stableHash", "wallSeconds", "cpuSeconds", "fatalDefect"}
    })
    sequential = combined_pilot._run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=maximum_candidates,
        deadline_monotonic=time.monotonic() + 10.0,
        evaluate_candidate=lambda *_args: copy.deepcopy(candidate),
    )
    assert sequential["qualifiedRow"] == candidate
    assert sequential["terminalRepeat"]["deterministic"] is True
    result.update({
        "status": "qualified",
        "stopReason": "qualified_candidate",
        "candidateResults": [candidate],
        "qualifiedCandidateStableHash": candidate["stableHash"],
        "terminalRepeat": {
            "performed": True,
            "stableHash": candidate["stableHash"],
            "traceHash": candidate["traceHash"],
            "auditHash": candidate["auditHash"],
            "qualified": True,
            "wallSeconds": 0.01,
            "cpuSeconds": 0.01,
            "deterministic": True,
        },
    })
    result["seedResults"] = [{
        **result["seedResults"][0],
        "generatedCandidateCount": 1,
        "evaluatedCandidateCount": 1,
        "tacticRetentionCount": 1,
        "distinctCandidateHashes": 1,
        "sequentialSearchReceipt": copy.deepcopy(
            sequential["sequentialSearchReceipt"]
        ),
        "stopReason": "qualified_candidate",
    }]
    result["counts"].update({
        "seedsEvaluated": 1,
        "candidatesEvaluated": 1,
        "tacticRetained": 1,
        "qualified": 1,
        "distinctCandidateHashes": 1,
        "distinctTraceHashes": 1,
    })
    return _rehash_receipt(result)


def _rehash_candidate(payload: dict) -> None:
    candidate = payload["candidateResults"][0]
    candidate["stableHash"] = canonical_sha256({
        key: value
        for key, value in candidate.items()
        if key not in {"stableHash", "wallSeconds", "cpuSeconds", "fatalDefect"}
    })
    payload["qualifiedCandidateStableHash"] = candidate["stableHash"]
    payload["terminalRepeat"]["stableHash"] = candidate["stableHash"]
    _rehash_receipt(payload)


@pytest.mark.real_clock
def test_rehashed_worker_state_authority_source_and_exit_forgeries_fail_closed(
    tmp_path,
    root,
) -> None:
    output = tmp_path / "worker-tamper.json"
    args = SimpleNamespace(
        root=root,
        bundle_version="21.1.22",
        seed_start=505,
        attempts=1,
        candidates_per_seed=4,
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
    )
    launch_id = "tamper-launch"
    baseline_receipt = _complete_worker_evidence(args, launch_id=launch_id)
    trusted_control = copy.deepcopy(baseline_receipt["controlPlane"])
    trusted_source = copy.deepcopy(baseline_receipt["sourceFingerprintStart"])

    cases: list[dict] = []
    qualified_without_candidate = copy.deepcopy(baseline_receipt)
    qualified_without_candidate.update({
        "status": "qualified",
        "stopReason": "qualified_candidate",
    })
    cases.append(_rehash_receipt(qualified_without_candidate))

    source_rewrite = copy.deepcopy(baseline_receipt)
    source_rewrite["sourceFingerprintEnd"]["sha256"] = "b" * 64
    source_rewrite["sourceFrozen"] = False
    cases.append(_rehash_receipt(source_rewrite))

    identity_rewrite = copy.deepcopy(baseline_receipt)
    identity_rewrite["controlPlane"]["analysisAuthorities"]["primary"][
        "analysisIdentityHash"
    ] = "f" * 64
    cases.append(_rehash_receipt(identity_rewrite))

    configuration_rewrite = copy.deepcopy(baseline_receipt)
    configuration = configuration_rewrite["controlPlane"][
        "configurationAuthority"
    ]
    configuration["bundleRevisionId"] = "CBR-" + "F" * 24
    configuration["sha256"] = canonical_sha256({
        key: value for key, value in configuration.items() if key != "sha256"
    })
    configuration_rewrite["controlPlane"]["bundleRevisionId"] = configuration[
        "bundleRevisionId"
    ]
    configuration_rewrite["controlPlane"]["configurationAuthorityHash"] = (
        configuration["sha256"]
    )
    cases.append(_rehash_receipt(configuration_rewrite))

    for forged in cases:
        combined_pilot._atomic_write_json(output, forged)
        with pytest.raises(RuntimeError):
            _validate_worker_fixture(
                output,
                args=args,
                launch_id=launch_id,
                expected_control_plane=forged["controlPlane"],
                authoritative_control_plane=trusted_control,
                parent_source_start=trusted_source,
                parent_source_end=trusted_source,
            )

    combined_pilot._atomic_write_json(output, baseline_receipt)
    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        _validate_worker_fixture(
            output,
            args=args,
            launch_id=launch_id,
            expected_control_plane=trusted_control,
            authoritative_control_plane=trusted_control,
            worker_exit_code=1,
        )


def test_publish_json_once_has_exactly_one_winner_under_a_direct_race(
    tmp_path,
) -> None:
    output = tmp_path / "immutable-receipt.json"
    barrier = threading.Barrier(16)
    winners: list[int] = []
    collisions: list[int] = []
    lock = threading.Lock()

    def publish(index: int) -> None:
        barrier.wait()
        try:
            combined_pilot._publish_json_once(output, {"winner": index})
        except FileExistsError:
            with lock:
                collisions.append(index)
        else:
            with lock:
                winners.append(index)

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(winners) == 1
    assert len(collisions) == 15
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "winner": winners[0]
    }
    assert not list(tmp_path.glob("*.receipt.tmp"))


def test_current_receipt_authority_is_rebuilt_instead_of_cached(
    root,
    monkeypatch,
) -> None:
    calls = 0
    real_build_bundle = combined_pilot.build_bundle

    def counted_build_bundle(selected_root):
        nonlocal calls
        calls += 1
        return real_build_bundle(selected_root)

    monkeypatch.setattr(combined_pilot, "build_bundle", counted_build_bundle)
    first = combined_pilot._current_finalist_receipt_authority(
        root,
        "python_reference_exact_search_v1",
    )
    second = combined_pilot._current_finalist_receipt_authority(
        root,
        "python_reference_exact_search_v1",
    )

    assert first == second
    assert calls == 2


@pytest.mark.real_clock
def test_rehashed_candidate_outcome_cohort_budget_role_and_seed_tampering_fails(
    tmp_path,
    root,
) -> None:
    output = tmp_path / "qualified-worker.json"
    args = SimpleNamespace(
        root=root,
        bundle_version="21.1.22",
        seed_start=505,
        attempts=1,
        candidates_per_seed=4,
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
    )
    launch_id = "qualified-tamper-launch"
    baseline_receipt = _qualified_worker_evidence(
        _complete_worker_evidence(args, launch_id=launch_id)
    )
    combined_pilot._validated_candidate_receipt(
        {
            "seed": 505,
            "candidateIndex": 1,
            "candidateHash": "1" * 64,
            "dealModelHash": "2" * 64,
            "dealSolverStatus": "optimal",
            "status": "rejected_before_audit",
            "qualified": False,
            "fatalDefect": False,
            "failures": ["constructed_auction_all_pass"],
            "errorType": "ValueError",
            "wallSeconds": 0.01,
            "cpuSeconds": 0.01,
        },
        agreement_descriptor=baseline_receipt["controlPlane"][
            "finalistAgreementDescriptor"
        ],
        agreement_profile_hash=baseline_receipt["controlPlane"][
            "finalistAgreementProfileHash"
        ],
        authorities=baseline_receipt["controlPlane"]["analysisAuthorities"],
        binding_authority={
            "researchContextHash": baseline_receipt["controlPlane"]["contextHash"],
            "rulesHash": baseline_receipt["controlPlane"]["configurationAuthority"][
                "identities"
            ]["rules"]["sha256"],
            "analysisBackend": baseline_receipt["controlPlane"][
                "searchBackendRuntime"
            ]["backendId"],
        },
    )
    combined_pilot._atomic_write_json(output, baseline_receipt)
    assert _validate_worker_fixture(
        output,
        args=args,
        launch_id=launch_id,
        expected_control_plane=baseline_receipt["controlPlane"],
    ) == baseline_receipt

    cases = []
    gate_tamper = copy.deepcopy(baseline_receipt)
    gate_tamper["candidateResults"][0]["gates"]["exactContract"] = False
    _rehash_candidate(gate_tamper)
    cases.append(gate_tamper)

    cohort_tamper = copy.deepcopy(baseline_receipt)
    decision = cohort_tamper["candidateResults"][0]["policyEvidence"][
        "plays"
    ][0]
    decision["cohortSupport"] = decision["cohortSupport"][:1]
    decision["cohortCount"] = 1
    decision["agreementCount"] = 1
    policy = cohort_tamper["candidateResults"][0]["policyEvidence"]
    policy["sha256"] = canonical_sha256({
        key: value for key, value in policy.items() if key != "sha256"
    })
    _rehash_candidate(cohort_tamper)
    cases.append(cohort_tamper)

    budget_role_tamper = copy.deepcopy(baseline_receipt)
    cohort = budget_role_tamper["candidateResults"][0]["policyEvidence"][
        "plays"
    ][0]["cohortSupport"][1]
    cohort["analysisAuthorityRole"] = "deep_confirmation"
    cohort["analysisBudget"]["requested_samples"] = 192
    policy = budget_role_tamper["candidateResults"][0]["policyEvidence"]
    policy["sha256"] = canonical_sha256({
        key: value for key, value in policy.items() if key != "sha256"
    })
    _rehash_candidate(budget_role_tamper)
    cases.append(budget_role_tamper)

    seed_tamper = copy.deepcopy(baseline_receipt)
    seed_tamper["candidateResults"][0]["policyEvidence"]["plays"][0][
        "cohortSupport"
    ][2]["seed"] += 1
    policy = seed_tamper["candidateResults"][0]["policyEvidence"]
    policy["sha256"] = canonical_sha256({
        key: value for key, value in policy.items() if key != "sha256"
    })
    _rehash_candidate(seed_tamper)
    cases.append(seed_tamper)

    for forged in cases:
        combined_pilot._atomic_write_json(output, forged)
        with pytest.raises(RuntimeError):
            _validate_worker_fixture(
                output,
                args=args,
                launch_id=launch_id,
                expected_control_plane=baseline_receipt["controlPlane"],
                authoritative_control_plane=baseline_receipt["controlPlane"],
            )


@pytest.mark.real_clock
def test_fully_rehashed_candidate_decision_binding_forgeries_fail_closed(
    tmp_path,
    root,
) -> None:
    output = tmp_path / "decision-binding-tamper.json"
    args = SimpleNamespace(
        root=root,
        bundle_version="21.1.22",
        seed_start=505,
        attempts=1,
        candidates_per_seed=4,
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
    )
    launch_id = "decision-binding-tamper-launch"
    baseline_receipt = _qualified_worker_evidence(
        _complete_worker_evidence(args, launch_id=launch_id)
    )
    control = baseline_receipt["controlPlane"]
    combined_pilot._atomic_write_json(output, baseline_receipt)
    assert _validate_worker_fixture(
        output,
        args=args,
        launch_id=launch_id,
        expected_control_plane=control,
        authoritative_control_plane=control,
    ) == baseline_receipt

    def reseal(payload: dict) -> dict:
        candidate = payload["candidateResults"][0]
        policy = candidate["policyEvidence"]
        policy["sha256"] = canonical_sha256({
            key: value for key, value in policy.items() if key != "sha256"
        })
        parity = candidate["decisionScoreParity"]
        parity["sha256"] = canonical_sha256({
            key: value for key, value in parity.items() if key != "sha256"
        })
        candidate["decisionScoreParityHash"] = parity["sha256"]
        _rehash_candidate(payload)
        return payload

    cases: list[dict] = []

    prefix_actor = copy.deepcopy(baseline_receipt)
    row = prefix_actor["candidateResults"][0]["policyEvidence"]["plays"][0]
    row["publicPrefixHash"] = "0" * 64
    row["actingSeat"] = "north"
    for index, cohort in enumerate(row["cohortSupport"]):
        authority = control["analysisAuthorities"][
            "primary" if index == 0 else "reanalysis"
        ]
        cohort["seed"] = combined_pilot.derive_actor_observer_cohort_seed_v1(
            public_prefix_hash=row["publicPrefixHash"],
            decision_kind=row["decisionKind"],
            acting_seat=row["actingSeat"],
            cohort_authority_hash=authority["analysisIdentityHash"],
            cohort_index=index,
            governed_salt=combined_pilot.PRIMARY_DECISION_COHORT_SALT_V1,
        )
    cases.append(reseal(prefix_actor))

    selected_action = copy.deepcopy(baseline_receipt)
    candidate = selected_action["candidateResults"][0]
    row = candidate["policyEvidence"]["plays"][0]
    row["selectedAction"] = {"code": "S7"}
    row["selectedActionMatchesTrace"] = True
    action_hash = canonical_sha256(row["selectedAction"])
    for cohort in row["cohortSupport"]:
        cohort["recommendedActionHash"] = action_hash
    row["majorityActionHash"] = action_hash
    evidence_hash = canonical_sha256({
        "forgedAction": row["selectedAction"],
        "forgedDecisionId": row["decisionId"],
    })
    rows_hash = canonical_sha256({
        "forgedRows": row["selectedAction"],
        "forgedDecisionId": row["decisionId"],
    })
    row["primaryDecisionScoreEvidenceHash"] = evidence_hash
    parity_row = candidate["decisionScoreParity"]["decisions"][0]
    parity_row["generationEvidenceHash"] = evidence_hash
    parity_row["auditEvidenceHash"] = evidence_hash
    parity_row["generationRowsHash"] = rows_hash
    parity_row["auditRowsHash"] = rows_hash
    cases.append(reseal(selected_action))

    rules = copy.deepcopy(baseline_receipt)
    candidate = rules["candidateResults"][0]
    candidate["rulesHash"] = "0" * 64
    policy = candidate["policyEvidence"]
    policy["rulesHash"] = candidate["rulesHash"]
    for row in policy["plays"]:
        row["rulesHash"] = candidate["rulesHash"]
    cases.append(reseal(rules))

    legal = copy.deepcopy(baseline_receipt)
    legal["candidateResults"][0]["policyEvidence"]["plays"][0][
        "legalActionSetHash"
    ] = "0" * 64
    cases.append(reseal(legal))

    order = copy.deepcopy(baseline_receipt)
    order["candidateResults"][0]["policyEvidence"]["plays"].reverse()
    cases.append(reseal(order))

    kind = copy.deepcopy(baseline_receipt)
    candidate = kind["candidateResults"][0]
    candidate["policyEvidence"]["plays"][0]["decisionKind"] = "bid"
    candidate["decisionScoreParity"]["decisions"][0][
        "decisionKind"
    ] = "bid"
    cases.append(reseal(kind))

    ordinal = copy.deepcopy(baseline_receipt)
    candidate = ordinal["candidateResults"][0]
    candidate["policyEvidence"]["plays"][0]["ordinal"] = 3
    candidate["decisionScoreParity"]["decisions"][0]["ordinal"] = 3
    cases.append(reseal(ordinal))

    forced = copy.deepcopy(baseline_receipt)
    candidate = forced["candidateResults"][0]
    forced_row = candidate["policyEvidence"]["forcedDecisions"][0]
    forced_row["selectedAction"] = {"code": "D8"}
    forced_row["selectedActionMatchesTrace"] = True
    proof = forced_row["forcedLegalityProof"]
    proof["selectedAction"] = copy.deepcopy(forced_row["selectedAction"])
    proof["selectedActionHash"] = canonical_sha256(proof["selectedAction"])
    proof["legalActionSetHash"] = combined_pilot.legal_action_set_hash_v1(
        (proof["selectedAction"],)
    )
    proof["sha256"] = canonical_sha256({
        key: value for key, value in proof.items() if key != "sha256"
    })
    forced_row["legalActionSetHash"] = proof["legalActionSetHash"]
    forced_row["forcedLegalityProofHash"] = proof["sha256"]
    parity_row = candidate["decisionScoreParity"]["decisions"][-1]
    parity_row["generationEvidenceHash"] = proof["sha256"]
    parity_row["auditEvidenceHash"] = proof["sha256"]
    cases.append(reseal(forced))

    for forged in cases:
        combined_pilot._atomic_write_json(output, forged)
        with pytest.raises(RuntimeError):
            _validate_worker_fixture(
                output,
                args=args,
                launch_id=launch_id,
                expected_control_plane=control,
                authoritative_control_plane=control,
            )


@pytest.mark.real_clock
def test_parent_validates_then_cleans_and_finalizes_worker_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "complete.json"
    args = SimpleNamespace(
        root=root,
        output=output,
        qualification_profile=FINALIST_CAMPAIGN_PROFILE_ID,
        analysis_mode="deep",
        attempts=1,
        seed_start=505,
        candidates_per_seed=4,
        bundle_version="21.1.22",
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
        _finalist_worker=False,
    )
    runtime_paths: list[Path] = []
    worker_control_planes: list[dict] = []
    validation_runtime_states: list[bool] = []
    real_popen = combined_pilot.subprocess.Popen
    real_validate = combined_pilot._validated_finalist_worker_evidence
    real_current_authority = combined_pilot._current_finalist_receipt_authority
    authority_root = Path(__file__).resolve().parents[1]

    # Resolve host setup and runtime metadata before intercepting process launches.
    assert platform_root(authority_root) == authority_root
    combined_pilot.platform.platform()

    class ExitedWorker:
        pid = 81004
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def fake_popen(command, **_kwargs):
        # Receipt setup can invoke other subprocesses.
        if "--_supervised-runtime" not in command:
            assert Path(command[0]).stem == "git", command
            return real_popen(command, **_kwargs)
        runtime = Path(command[command.index("--_supervised-runtime") + 1])
        launch_id = command[command.index("--_supervision-launch-id") + 1]
        runtime_paths.append(runtime)
        worker_payload = _complete_worker_evidence(
            args,
            launch_id=launch_id,
            authority_root=authority_root,
        )
        worker_control_planes.append(worker_payload["controlPlane"])
        worker_output = Path(command[command.index("--output") + 1])
        combined_pilot._publish_json_once(worker_output, worker_payload)
        return ExitedWorker()

    with pytest.raises(AssertionError, match="unexpected_clock_guard_worker"):
        fake_popen([str(tmp_path / "unexpected_clock_guard_worker.exe")])

    def validate_with_runtime_state(*validate_args, **validate_kwargs):
        validation_runtime_states.append(runtime_paths[-1].exists())
        return real_validate(*validate_args, **validate_kwargs)

    monkeypatch.setattr(combined_pilot.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        combined_pilot,
        "_supervise_finalist_worker",
        lambda *_args, **_kwargs: {
            "timedOut": False,
            "workerExitCode": 0,
            "lastProgress": {
                "controlPlane": worker_control_planes[-1],
            },
        },
    )
    monkeypatch.setattr(
        combined_pilot,
        "_validated_finalist_worker_evidence",
        validate_with_runtime_state,
    )
    monkeypatch.setattr(
        combined_pilot,
        "_current_finalist_receipt_authority",
        lambda _root, backend_id: real_current_authority(
            authority_root,
            backend_id,
        ),
    )
    monkeypatch.setattr(
        combined_pilot,
        "_validated_runtime_control_plane",
        lambda **kwargs: copy.deepcopy(kwargs["control_plane"]),
    )

    assert _run_finalist_campaign_supervised(args) == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert validation_runtime_states == [True, False]
    assert runtime_paths and runtime_paths[0].exists() is False
    assert evidence["temporaryRuntimeCleaned"] is True
    assert evidence["supervision"]["runtimeOwner"] == "supervisor_process"
    assert evidence["supervision"]["parentRuntimeFinalized"] is True
    assert evidence["sha256"] == canonical_sha256({
        key: value for key, value in evidence.items() if key != "sha256"
    })


@pytest.mark.real_clock
def test_supervisor_rejects_preexisting_output_without_overwriting_or_launching(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "stale.json"
    args = SimpleNamespace(
        root=root,
        output=output,
        qualification_profile=FINALIST_CAMPAIGN_PROFILE_ID,
        analysis_mode="deep",
        attempts=1,
        seed_start=505,
        candidates_per_seed=4,
        bundle_version="21.1.22",
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
        _finalist_worker=False,
    )
    stale = _complete_worker_evidence(
        args,
        launch_id="stale-launch",
        authority_root=Path(__file__).resolve().parents[1],
    )
    combined_pilot._atomic_write_json(output, stale)
    original = output.read_bytes()
    launched = False

    def forbidden_popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("preexisting output must block worker launch")

    monkeypatch.setattr(combined_pilot.subprocess, "Popen", forbidden_popen)

    with pytest.raises(
        ValueError,
        match="output is immutable and already exists",
    ):
        _run_finalist_campaign_supervised(args)

    assert launched is False
    assert output.read_bytes() == original
    assert json.loads(output.read_text(encoding="utf-8")) == stale


@pytest.mark.real_clock
def test_finally_termination_failure_cannot_override_timeout_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "timeout.json"

    class LingeringWorker:
        pid = 81003
        returncode = None
        killed = False
        waited = False

        @classmethod
        def poll(cls):
            return -9 if cls.killed else None

        @classmethod
        def kill(cls):
            cls.killed = True
            cls.returncode = -9

        @classmethod
        def wait(cls, **_kwargs):
            cls.waited = True
            return cls.returncode

    monkeypatch.setattr(
        combined_pilot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: LingeringWorker(),
    )
    monkeypatch.setattr(
        combined_pilot,
        "_supervise_finalist_worker",
        lambda *_args, **_kwargs: {
            "timedOut": True,
            "boundary": "seed_wall_bound",
            "lastProgress": {
                "phase": "candidate_evaluation",
                "completedCandidates": [],
                "completedSeeds": [],
            },
            "termination": {
                "requested": True,
                "workerTerminated": False,
            },
        },
    )
    monkeypatch.setattr(
        combined_pilot,
        "_terminate_owned_process_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("controlled final termination failure")
        ),
    )
    args = SimpleNamespace(
        root=root,
        output=output,
        qualification_profile=FINALIST_CAMPAIGN_PROFILE_ID,
        analysis_mode="deep",
        attempts=1,
        seed_start=505,
        candidates_per_seed=4,
        bundle_version="21.1.22",
        maximum_wall_seconds=180.0,
        maximum_seed_seconds=120.0,
        _finalist_worker=False,
    )

    assert _run_finalist_campaign_supervised(args) == 1
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["stopReason"] == "campaign_hard_timeout"
    assert evidence["failure"]["errorType"] == "CampaignHardTimeout"
    assert LingeringWorker.killed is True
    assert LingeringWorker.waited is True


@pytest.mark.real_clock
def test_finalist_generation_passes_the_exact_campaign_deadline(monkeypatch):
    spec = _sequential_fixture_spec(1)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=0,
        continuation_exhausted=True,
    )
    receipt = combined_pilot.SequentialCandidateSearchReceiptV1.from_dict(
        _empty_sequential_search_receipt(502, 1)
    )
    observed = {}

    def product_call(selected_generator, selected_spec, **kwargs):
        observed.update(
            generator=selected_generator,
            spec=selected_spec,
            **kwargs,
        )
        return SimpleNamespace(receipt=receipt, qualified_candidate=None)

    monkeypatch.setattr(
        combined_pilot,
        "run_sequential_candidate_search_v1",
        product_call,
    )
    deadline = time.monotonic() + 123.5
    result = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=1,
        deadline_monotonic=deadline,
        evaluate_candidate=lambda *_args: {},
    )

    assert observed["spec"] is spec
    assert observed["generator"].delegate is generator
    assert observed["root_seed"] == 502
    assert observed["dealer"] is Seat.EAST
    assert observed["rules"] is _sequential_fixture_rules()
    assert observed["deadline_monotonic"] == deadline
    assert result["sequentialSearchReceipt"] == receipt.to_dict()


@pytest.mark.real_clock
def test_finalist_generation_deadline_is_a_structured_bounded_stop():
    spec = _sequential_fixture_spec(1)

    class Generator(_SequentialFixtureGenerator):
        def generate_next(self, *_args, **_kwargs):
            raise InterruptedError("CP-SAT deal generation deadline exceeded")

    generator = Generator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=0,
        continuation_exhausted=True,
    )
    result = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=1,
        deadline_monotonic=time.monotonic() + 10.0,
        evaluate_candidate=lambda *_args: {},
    )

    assert result["generationTimedOut"] is True
    assert result["stopReason"] == "candidate_generation_deadline"
    assert result["sequentialSearchReceipt"] is None


@pytest.mark.real_clock
def test_finalist_generation_does_not_hide_non_deadline_failures():
    spec = _sequential_fixture_spec(1)

    class Generator(_SequentialFixtureGenerator):
        def generate_next(self, *_args, **_kwargs):
            raise RuntimeError("solver defect")

    generator = Generator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=0,
        continuation_exhausted=True,
    )
    with pytest.raises(RuntimeError, match="solver defect"):
        _run_product_sequential_candidate_search(
            generator,
            spec,
            root_seed=502,
            dealer=Seat.EAST,
            rules=_sequential_fixture_rules(),
            maximum_candidates=1,
            deadline_monotonic=time.monotonic() + 10.0,
            evaluate_candidate=lambda *_args: {},
        )


def _real_deal_spec(*, compact: bool) -> DealConstraintSpecV1:
    value = {
        "schemaVersion": 1,
        "id": "terminal_no_good_probe_test_v1",
        "revision": "1",
        "enabled": True,
        "backend": "ortools_cp_sat_v1",
        "assumedBuyer": "south",
        "purchaseSuits": ["spades"],
        "initialRequiredCards": {"south": ["SJ", "S9"]},
        "finalRequiredCards": {},
        "finalForbiddenCards": {},
        "initialSuitCountRanges": {
            "south": {"spades": {"minimum": 2, "maximum": 5}},
        },
        "finalSuitCountRanges": {
            "south": {"spades": {"minimum": 3, "maximum": 8}},
        },
        "projectPattern": None,
        "maximumSolutions": 2,
        "maximumConflicts": 1_000_000,
        "safetyTimeoutSeconds": 30.0,
        "solverWorkers": 1,
        "seedOffset": 0,
        "tags": ["engine_validation_required", "guided_structural"],
    }
    if compact:
        value.update({
            "purchaseSuits": [],
            "initialRequiredCards": {},
            "initialSuitCountRanges": {},
            "finalSuitCountRanges": {},
            "tacticalObjectiveWeights": {"only_winning_move": 100},
            "objectiveTargetSeat": "west",
            "tacticalContract": "sun",
        })
    return DealConstraintSpecV1.build(value)


def _candidate_result(
    candidate: ConstraintDealCandidateV1,
    *,
    qualified: bool = False,
    fatal: bool = False,
    status: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "candidateHash": candidate.candidate_hash,
        "qualified": qualified,
        "fatalDefect": fatal,
        "stableHash": "4" * 64,
        "traceHash": "5" * 64,
        "auditHash": "6" * 64,
        "wallSeconds": 0.01,
        "cpuSeconds": 0.01,
    }
    if status is not None:
        result["status"] = status
    return result


def _parity_evidence(
    *,
    trace_hash: str,
    analysis_hash: str,
    context_hash: str,
    matched: bool,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schemaVersion": 1,
        "authority": "finalist_primary_decision_score_parity_v1",
        "traceHash": trace_hash,
        "deepAnalysisHash": analysis_hash,
        "researchContextHash": context_hash,
        "matched": matched,
        "mismatches": (
            []
            if matched
            else [{
                "decisionId": "PLAY-1",
                "ordinal": 1,
                "decisionKind": "play",
                "reason": "primary_decision_score_evidence_bytes_mismatch",
            }]
        ),
        "decisions": [],
        "crossLaneComparison": False,
        "structuralObjectiveCompared": False,
        "officialBalootScoreCompared": False,
    }
    return {**core, "sha256": canonical_sha256(core)}


@pytest.mark.parametrize("matched", (True, False))
def test_direct_deep_parity_uses_frozen_production_wrapper_exactly(matched):
    trace = SimpleNamespace(sha256="1" * 64)
    analysis = SimpleNamespace(sha256="2" * 64)
    context_hash = "3" * 64
    expected = _parity_evidence(
        trace_hash=trace.sha256,
        analysis_hash=analysis.sha256,
        context_hash=context_hash,
        matched=matched,
    )
    calls = []

    class Analyzer:
        def analyze_trace(self, value, *, cancel_check):
            calls.append(("analyze", value, cancel_check))
            return analysis

    class Research:
        @staticmethod
        def _decision_score_parity_evidence(value, result, *, context):
            calls.append(("parity", value, result, context))
            return expected

    context = SimpleNamespace(
        analyzers={"deep": Analyzer()},
        context_hash=context_hash,
    )

    def cancel_check():
        return False

    actual_analysis, actual_evidence = _direct_deep_decision_score_parity(
        research=Research(),
        context=context,
        trace=trace,
        cancel_check=cancel_check,
    )

    assert actual_analysis is analysis
    assert actual_evidence == expected
    assert calls == [
        ("analyze", trace, cancel_check),
        ("parity", trace, analysis, context),
    ]
    assert _decision_score_parity_failures(actual_evidence) == (
        []
        if matched
        else [
            "decision_score_parity:"
            "primary_decision_score_evidence_bytes_mismatch"
        ]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("crossLaneComparison", True),
        ("structuralObjectiveCompared", True),
        ("officialBalootScoreCompared", True),
        ("researchContextHash", "4" * 64),
    ),
)
def test_direct_deep_parity_rejects_proof_boundary_or_provenance_drift(
    field,
    value,
):
    trace = SimpleNamespace(sha256="1" * 64)
    analysis = SimpleNamespace(sha256="2" * 64)
    context = SimpleNamespace(
        analyzers={
            "deep": SimpleNamespace(
                analyze_trace=lambda *_args, **_kwargs: analysis
            )
        },
        context_hash="3" * 64,
    )
    evidence = _parity_evidence(
        trace_hash=trace.sha256,
        analysis_hash=analysis.sha256,
        context_hash=context.context_hash,
        matched=True,
    )
    evidence[field] = value
    evidence["sha256"] = canonical_sha256({
        key: item for key, item in evidence.items() if key != "sha256"
    })
    research = SimpleNamespace(
        _decision_score_parity_evidence=lambda *_args, **_kwargs: evidence
    )

    with pytest.raises(RuntimeError, match="parity evidence provenance"):
        _direct_deep_decision_score_parity(
            research=research,
            context=context,
            trace=trace,
            cancel_check=lambda: False,
        )


def test_decision_score_parity_integrity_drift_fails_closed():
    trace = SimpleNamespace(sha256="1" * 64)
    analysis = SimpleNamespace(sha256="2" * 64)
    context = SimpleNamespace(
        analyzers={
            "deep": SimpleNamespace(
                analyze_trace=lambda *_args, **_kwargs: analysis
            )
        },
        context_hash="3" * 64,
    )
    evidence = _parity_evidence(
        trace_hash=trace.sha256,
        analysis_hash=analysis.sha256,
        context_hash=context.context_hash,
        matched=True,
    )
    evidence["sha256"] = "0" * 64
    research = SimpleNamespace(
        _decision_score_parity_evidence=lambda *_args, **_kwargs: evidence
    )

    with pytest.raises(RuntimeError, match="parity evidence integrity"):
        _direct_deep_decision_score_parity(
            research=research,
            context=context,
            trace=trace,
            cancel_check=lambda: False,
        )


def _run_fixture_adapter_search(
    *,
    seed: int = 502,
    maximum_candidates: int = 2,
    candidate_count: int = 1,
    continuation_exhausted: bool = False,
    evaluate_candidate=None,
    generator=None,
    deadline_monotonic: float | None = None,
):
    spec = _sequential_fixture_spec(maximum_candidates)
    selected_generator = generator or _SequentialFixtureGenerator(
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        candidate_count=candidate_count,
        continuation_exhausted=continuation_exhausted,
    )
    evaluator = evaluate_candidate or (
        lambda candidate, _index: {
            **_candidate_result(candidate),
            "failures": ["content_qualification_rejection"],
        }
    )
    result = _run_product_sequential_candidate_search(
        selected_generator,
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=maximum_candidates,
        deadline_monotonic=(
            time.monotonic() + 10.0
            if deadline_monotonic is None
            else deadline_monotonic
        ),
        evaluate_candidate=evaluator,
    )
    return spec, selected_generator, result


def _canonical_product_receipt(result: dict) -> dict:
    document = copy.deepcopy(result["sequentialSearchReceipt"])
    assert combined_pilot.SequentialCandidateSearchReceiptV1.from_dict(
        document
    ).verify().to_dict() == document
    assert canonical_sha256({
        key: value for key, value in document.items() if key != "sha256"
    }) == document["sha256"]
    return document


def _rehash_product_receipt(document: dict) -> None:
    document["sha256"] = canonical_sha256({
        key: value for key, value in document.items() if key != "sha256"
    })


def _ordinary_terminal_search_result(*, exhausted: bool = True) -> dict:
    _, _, result = _run_fixture_adapter_search(
        maximum_candidates=1,
        candidate_count=1,
        continuation_exhausted=exhausted,
    )
    return result


@pytest.mark.real_clock
def test_decision_score_parity_mismatch_is_fatal_before_no_good_feedback():
    evidence = _parity_evidence(
        trace_hash="1" * 64,
        analysis_hash="2" * 64,
        context_hash="3" * 64,
        matched=False,
    )
    _, generator, result = _run_fixture_adapter_search(
        candidate_count=1,
        evaluate_candidate=lambda value, _index: {
            **_candidate_result(value, fatal=True),
            "decisionScoreParityHash": evidence["sha256"],
            "decisionScoreParity": evidence,
            "failures": _decision_score_parity_failures(evidence),
        },
    )

    receipt = _canonical_product_receipt(result)
    assert generator.generate_calls == [()]
    assert result["fatalFailure"] == "fatal_semantic_or_leak_defect"
    assert result["rows"][0]["decisionScoreParity"] == evidence
    assert result["rejectedCandidates"] == ()
    assert receipt["stopReason"] == "fatal_quarantine"
    assert receipt["rejectionFeedbackRows"] == []


@pytest.mark.real_clock
def test_sequential_rejections_ask_for_exact_next_candidate_without_repair(
    monkeypatch,
):
    spec = _sequential_fixture_spec(2)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=2,
        continuation_exhausted=False,
    )
    evaluated: list[str] = []
    shared_receipts: list[dict] = []
    product_call = combined_pilot.run_sequential_candidate_search_v1

    def tracked_product_call(*args, **kwargs):
        product_result = product_call(*args, **kwargs)
        shared_receipts.append(product_result.receipt.verify().to_dict())
        return product_result

    monkeypatch.setattr(
        combined_pilot,
        "run_sequential_candidate_search_v1",
        tracked_product_call,
    )
    result = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=2,
        deadline_monotonic=time.monotonic() + 10.0,
        evaluate_candidate=lambda candidate, _index: (
            evaluated.append(candidate.candidate_hash)
            or {
                **_candidate_result(candidate),
                "failures": ["content_qualification_rejection"],
            }
        ),
    )

    receipt = _canonical_product_receipt(result)
    assert receipt == shared_receipts[0]
    assert receipt["sha256"] == shared_receipts[0]["sha256"]
    assert evaluated == [candidate.candidate_hash for candidate in generator.candidates]
    assert generator.generate_calls == [(), (generator.candidates[0].candidate_hash,)]
    assert generator.probe_calls == [tuple(evaluated)]
    assert result["stopReason"] == "candidate_bound_exhausted"
    assert result["rejectedCandidates"] == generator.candidates
    feedback = receipt["rejectionFeedbackRows"]
    assert [item["candidateHash"] for item in feedback] == evaluated
    assert all(item["askedForNextSolution"] for item in feedback)
    assert all(item["appliedToSubsequentSolve"] for item in feedback)
    assert feedback[0]["applicationReceipt"]["nextCandidateHash"] == evaluated[1]
    assert feedback[1]["applicationReceipt"]["solveEvidenceHash"] == receipt[
        "terminalContinuationReceipt"
    ]["sha256"]


@pytest.mark.real_clock
@pytest.mark.skipif(
    not OrToolsCpSatDealGeneratorV1.available(),
    reason="OR-Tools optimization extra is not installed",
)
@pytest.mark.parametrize("compact", (False, True), ids=("full-deck", "compact-hands"))
def test_real_generator_runs_post_final_cut_solver_without_evaluating_witness(
    root,
    monkeypatch,
    compact,
):
    from ortools.sat.python import cp_model

    bundle = build_bundle(root)
    spec = _real_deal_spec(compact=compact)
    generator = OrToolsCpSatDealGeneratorV1()
    solve_cut_literal_sets: list[tuple[frozenset[str], ...]] = []
    original_solve = cp_model.CpSolver.solve

    def tracked_solve(solver, model, solution_callback=None):
        status = original_solve(solver, model, solution_callback)
        proto = copy.deepcopy(model.proto)
        names = {
            index: variable.name
            for index, variable in enumerate(proto.variables)
        }
        cut_literal_sets = []
        for constraint in proto.constraints:
            linear = constraint.linear
            if (
                len(linear.vars) == 32
                and list(linear.coeffs) == [1] * 32
                and list(linear.domain)[-1:] == [31]
            ):
                cut_literal_sets.append(frozenset(
                    names[index] for index in linear.vars
                ))
        solve_cut_literal_sets.append(tuple(cut_literal_sets))
        return status

    monkeypatch.setattr(cp_model.CpSolver, "solve", tracked_solve)
    evaluations: list[str] = []

    def evaluate(candidate, _candidate_index):
        evaluations.append(candidate.candidate_hash)
        return {
            **_candidate_result(candidate),
            "failures": ["content_qualification_rejection"],
        }

    result = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=20260809,
        dealer=Seat.EAST,
        rules=bundle.rules,
        maximum_candidates=2,
        deadline_monotonic=time.monotonic() + 120.0,
        evaluate_candidate=evaluate,
    )

    receipt_bundle = _canonical_product_receipt(result)
    receipt = receipt_bundle["terminalContinuationReceipt"]
    feedback = receipt_bundle["rejectionFeedbackRows"]
    assert len(evaluations) == 2
    assert len(result["rows"]) == 2
    assert receipt["solverInvokedAfterFinalCut"] is True
    assert receipt["solverStatus"] == "optimal"
    assert receipt["solutionOrdinal"] == 2
    assert receipt["candidateEmitted"] is False
    assert receipt["candidateEvaluated"] is False
    assert receipt["candidatePublished"] is False
    assert receipt["witnessCandidateHash"] not in evaluations
    assert receipt["witnessDeckCommitment"] not in {
        candidate.deck_commitment for candidate in result["rejectedCandidates"]
    }
    assert "deckOrder" not in json.dumps(receipt, sort_keys=True)
    assert receipt["precedingNoGoodCutHashes"] == [
        item["noGoodCutHash"] for item in feedback
    ]

    layout = DealLayoutV1.build(
        bundle.rules.structural_domain,
        dealer=Seat.EAST,
        buyer=spec.assumed_buyer,
    )
    expected_cut_sets = []
    for candidate in result["rejectedCandidates"]:
        if compact:
            hands = generator._hands_from_deck(candidate.deck_order, layout=layout)
            expected_cut_sets.append(frozenset(
                f"c{card}_s{int(seat)}"
                for seat, cards in hands.items()
                for card in cards
            ))
        else:
            expected_cut_sets.append(frozenset(
                f"c{card}_p{position}"
                for position, card in enumerate(candidate.deck_order)
            ))
    assert solve_cut_literal_sets[-1] == tuple(expected_cut_sets)

    rerun = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=20260809,
        dealer=Seat.EAST,
        rules=bundle.rules,
        maximum_candidates=2,
        deadline_monotonic=time.monotonic() + 120.0,
        evaluate_candidate=evaluate,
    )
    assert rerun["sequentialSearchReceipt"] == receipt_bundle


@pytest.mark.real_clock
def test_terminal_no_good_probe_can_prove_real_exhaustion_without_extra_evaluation():
    evaluations = 0

    def evaluate(candidate, _candidate_index):
        nonlocal evaluations
        evaluations += 1
        return {
            **_candidate_result(candidate),
            "failures": ["content_qualification_rejection"],
        }

    _, generator, result = _run_fixture_adapter_search(
        maximum_candidates=1,
        candidate_count=1,
        continuation_exhausted=True,
        evaluate_candidate=evaluate,
    )
    receipt = _canonical_product_receipt(result)
    continuation = receipt["terminalContinuationReceipt"]
    assert evaluations == 1
    assert generator.probe_calls == [
        (generator.candidates[0].candidate_hash,)
    ]
    assert continuation["solverStatus"] == "infeasible"
    assert continuation["witnessCandidateHash"] is None
    assert receipt["rejectionFeedbackRows"][0][
        "appliedToSubsequentSolve"
    ] is True


@pytest.mark.real_clock
def test_terminal_no_good_probe_timeout_preserves_unapplied_feedback():
    spec = _sequential_fixture_spec(1)

    class Generator(_SequentialFixtureGenerator):
        def probe_continuation(self, *_args, **_kwargs):
            raise InterruptedError("bounded continuation probe")

    generator = Generator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    _, _, result = _run_fixture_adapter_search(
        maximum_candidates=1,
        generator=generator,
    )
    assert result["stopReason"] == "candidate_generation_deadline"
    assert result["generationTimedOut"] is True
    assert result["sequentialSearchReceipt"] is None
    assert len(result["rejectedCandidates"]) == 1


@pytest.mark.real_clock
def test_terminal_continuation_receipt_tamper_fails_closed():
    spec = _sequential_fixture_spec(1)

    class Generator(_SequentialFixtureGenerator):
        def probe_continuation(self, *args, **kwargs):
            receipt = super().probe_continuation(*args, **kwargs)
            return replace(receipt, receipt_hash="0" * 64)

    generator = Generator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    with pytest.raises(ValueError, match="continuation receipt integrity"):
        _run_fixture_adapter_search(maximum_candidates=1, generator=generator)


@pytest.mark.real_clock
def test_sequential_candidate_rejects_same_deck_under_new_history_hash():
    spec = _sequential_fixture_spec(2)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    first = generator.candidates[0]
    repeated = OrToolsCpSatDealGeneratorV1._build_candidate(
        ordinal=1,
        deck_order=first.deck_order,
        deck_commitment=first.deck_commitment,
        ordering_score=first.ordering_score,
        structural_objective_score=first.structural_objective_score,
        profile_hash=first.profile_hash,
        model_hash=first.model_hash,
        search_identity_hash=first.search_identity_hash,
        backend_version=first.backend_version,
        no_good_cut_type=first.no_good_cut_type,
        preceding_no_good_cut_hashes=(generator.cuts[0],),
    )

    class RepeatingGenerator:
        descriptor = OrToolsCpSatDealGeneratorV1.descriptor

        def generate_next(self, _spec, *, rejected_candidates, **_kwargs):
            return first if not rejected_candidates else repeated

    with pytest.raises(ValueError, match="repeated a rejected candidate or deck"):
        _run_fixture_adapter_search(
            maximum_candidates=2,
            generator=RepeatingGenerator(),
        )


@pytest.mark.real_clock
@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("witness_deck_commitment", "repeated a rejected candidate"),
        ("witness_candidate_hash", "repeated a rejected candidate"),
    ),
)
def test_terminal_receipt_rejects_reused_candidate_identity(field, message):
    spec = _sequential_fixture_spec(1)

    class Generator(_SequentialFixtureGenerator):
        def probe_continuation(self, *args, **kwargs):
            rejected = tuple(kwargs["rejected_candidates"])
            receipt = super().probe_continuation(*args, **kwargs)
            forged = replace(
                receipt,
                **{
                    field: (
                        rejected[0].deck_commitment
                        if field == "witness_deck_commitment"
                        else rejected[0].candidate_hash
                    )
                },
                receipt_hash="",
            )
            return replace(
                forged,
                receipt_hash=canonical_sha256(forged.integrity_core()),
            )

    generator = Generator(
        spec,
        root_seed=502,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    with pytest.raises(ValueError, match=message):
        _run_fixture_adapter_search(maximum_candidates=1, generator=generator)


@pytest.mark.real_clock
def test_finalized_no_good_feedback_detects_mutated_application_flags():
    document = _canonical_product_receipt(_ordinary_terminal_search_result())
    feedback = document["rejectionFeedbackRows"][0]
    feedback["askedForNextSolution"] = False
    feedback["stateHash"] = canonical_sha256({
        key: value for key, value in feedback.items() if key != "stateHash"
    })
    _rehash_product_receipt(document)

    with pytest.raises(ValueError, match="feedback"):
        combined_pilot.SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.real_clock
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("noGoodCutHash", "a" * 64),
        ("askedForNextSolution", False),
        ("applicationReceipt", {"forged": True}),
    ),
)
def test_unapplied_no_good_feedback_cannot_claim_solve_cross_fields(field, value):
    document = _canonical_product_receipt(_ordinary_terminal_search_result())
    feedback = document["rejectionFeedbackRows"][0]
    feedback.update({
        "appliedToSubsequentSolve": False,
        "noGoodCutHash": None,
        "applicationReceipt": None,
    })
    feedback[field] = copy.deepcopy(value)
    feedback["stateHash"] = canonical_sha256({
        key: item for key, item in feedback.items() if key != "stateHash"
    })
    _rehash_product_receipt(document)

    with pytest.raises(ValueError):
        combined_pilot.SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.real_clock
def test_finalized_no_good_feedback_validates_terminal_receipt_cross_hash():
    document = _canonical_product_receipt(_ordinary_terminal_search_result())
    feedback = document["rejectionFeedbackRows"][0]
    application = feedback["applicationReceipt"]
    application["solveEvidenceHash"] = "8" * 64
    application["sha256"] = canonical_sha256({
        key: value for key, value in application.items() if key != "sha256"
    })
    feedback["stateHash"] = canonical_sha256({
        key: value for key, value in feedback.items() if key != "stateHash"
    })
    _rehash_product_receipt(document)

    with pytest.raises(ValueError, match="continuation application evidence"):
        combined_pilot.SequentialCandidateSearchReceiptV1.from_dict(document)


@pytest.mark.real_clock
def test_seed_validator_rechecks_rehashed_no_good_and_continuation_receipts():
    seed = 502
    spec = _sequential_fixture_spec(1)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=True,
    )
    candidate = generator.candidates[0]
    candidate_row = {
        "seed": seed,
        "candidateIndex": 1,
        "candidateHash": candidate.candidate_hash,
        "dealModelHash": candidate.model_hash,
        "dealSolverStatus": candidate.solver_status,
        "qualified": False,
        "fatalDefect": False,
        "stableHash": "4" * 64,
        "traceHash": "5" * 64,
        "auditHash": "6" * 64,
        "failures": ["content_qualification_rejection"],
        "tacticRetained": False,
        "wallSeconds": 0.01,
        "cpuSeconds": 0.01,
    }
    sequential = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=seed,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=1,
        deadline_monotonic=time.monotonic() + 10.0,
        evaluate_candidate=lambda *_args: copy.deepcopy(candidate_row),
    )
    seed_receipt = {
        "schemaVersion": 1,
        "kind": "combined_finalist_seed_search_result_v1",
        "seed": seed,
        "generatedCandidateCount": 1,
        "evaluatedCandidateCount": 1,
        "tacticRetentionCount": 0,
        "distinctCandidateHashes": 1,
        "noPostHocRepair": True,
        "sequentialSearchReceipt": copy.deepcopy(
            sequential["sequentialSearchReceipt"]
        ),
        "continuationProbeWallSeconds": 0.01,
        "stopReason": "candidates_exhausted",
        "wallSeconds": 0.02,
        "cpuSeconds": 0.01,
    }
    combined_pilot._validated_seed_receipt(
        seed_receipt,
        candidate_rows=[candidate_row],
        maximum_candidates=1,
    )

    seed_forgery = copy.deepcopy(seed_receipt)
    seed_forgery["seed"] = seed + 1
    with pytest.raises(RuntimeError, match="seed-unbound"):
        combined_pilot._validated_seed_receipt(
            seed_forgery,
            candidate_rows=[candidate_row],
            maximum_candidates=1,
        )

    outcome_forgery = copy.deepcopy(candidate_row)
    outcome_forgery["failures"] = ["different_qualification_rejection"]
    with pytest.raises(RuntimeError, match="outcome is unbound"):
        combined_pilot._validated_seed_receipt(
            seed_receipt,
            candidate_rows=[outcome_forgery],
            maximum_candidates=1,
        )


@pytest.mark.real_clock
def test_runner_rejects_candidate_bound_drift_from_governed_spec():
    spec = _sequential_fixture_spec(2)
    with pytest.raises(RuntimeError, match="candidate bound differs"):
        _run_product_sequential_candidate_search(
            object(),
            spec,
            root_seed=502,
            dealer=Seat.EAST,
            rules=_sequential_fixture_rules(),
            maximum_candidates=1,
            deadline_monotonic=time.monotonic() + 10.0,
            evaluate_candidate=lambda *_args: {},
        )


@pytest.mark.real_clock
@pytest.mark.parametrize("ordinary_failure", ("tactic", "content", "audit", "decision"))
def test_ordinary_qualification_rejections_continue_with_exact_no_good(
    ordinary_failure,
):
    spec = _sequential_fixture_spec(2)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=503,
        dealer=Seat.EAST,
        candidate_count=2,
        continuation_exhausted=False,
    )

    def evaluate(candidate, _candidate_index):
        if candidate.ordinal == 0:
            return {
                **_candidate_result(candidate),
                "failures": [f"{ordinary_failure}_qualification_rejection"],
            }
        return _candidate_result(candidate, qualified=True)

    result = _run_product_sequential_candidate_search(
        generator,
        spec,
        root_seed=503,
        dealer=Seat.EAST,
        rules=_sequential_fixture_rules(),
        maximum_candidates=2,
        deadline_monotonic=time.monotonic() + 10.0,
        evaluate_candidate=evaluate,
    )
    receipt = _canonical_product_receipt(result)
    assert generator.generate_calls == [(), (generator.candidates[0].candidate_hash,)]
    assert result["stopReason"] == "qualified_candidate"
    assert result["qualifiedRow"]["candidateHash"] == generator.candidates[1].candidate_hash
    assert result["terminalRepeat"]["deterministic"] is True
    assert [item["candidateHash"] for item in receipt["rejectionFeedbackRows"]] == [
        generator.candidates[0].candidate_hash
    ]


@pytest.mark.real_clock
def test_fatal_semantic_or_observer_defect_stops_without_no_good_feedback():
    _, generator, result = _run_fixture_adapter_search(
        candidate_count=1,
        evaluate_candidate=lambda candidate, _index: _candidate_result(
            candidate,
            fatal=True,
            status="semantic_defect",
        ),
    )
    receipt = _canonical_product_receipt(result)
    assert generator.generate_calls == [()]
    assert result["stopReason"] == "fatal_semantic_or_leak_defect"
    assert result["fatalFailure"] == "fatal_semantic_or_leak_defect"
    assert result["rejectedCandidates"] == ()
    assert receipt["rejectionFeedbackRows"] == []


@pytest.mark.real_clock
def test_bounded_incomplete_evaluation_is_not_rejected_or_repaired():
    _, _, result = _run_fixture_adapter_search(
        candidate_count=1,
        evaluate_candidate=lambda candidate, _index: _candidate_result(
            candidate,
            status="bounded_timeout",
        ),
    )
    receipt = _canonical_product_receipt(result)
    assert result["stopReason"] == "candidate_evaluation_deadline"
    assert result["rejectedCandidates"] == ()
    assert receipt["rejectionFeedbackRows"] == []


@pytest.mark.real_clock
def test_post_evaluation_deadline_overrun_cannot_reject_or_repeat(monkeypatch):
    evaluation_calls = 0
    clock = iter((0.0, 2.0))

    def evaluate(candidate, _candidate_index):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return _candidate_result(candidate, qualified=True)

    monkeypatch.setattr(combined_pilot.time, "monotonic", lambda: next(clock))
    _, generator, result = _run_fixture_adapter_search(
        candidate_count=1,
        evaluate_candidate=evaluate,
        deadline_monotonic=1.0,
    )
    receipt = _canonical_product_receipt(result)
    assert generator.generate_calls == [()]
    assert evaluation_calls == 1
    assert result["stopReason"] == "candidate_evaluation_deadline"
    assert result["rows"][0]["status"] == "bounded_timeout"
    assert result["qualifiedRow"] is None
    assert result["terminalRepeat"] is None
    assert receipt["rejectionFeedbackRows"] == []


@pytest.mark.real_clock
@pytest.mark.parametrize(
    "message",
    (
        "complete-play and scoring CP-SAT models disagree",
        "reference and CP-SAT scoring models disagree",
        "a complete played Laffa must be authoritatively scored",
        "primary decision score evidence mismatch",
    ),
)
def test_authoritative_score_or_decision_disagreement_is_fatal_without_no_good(
    message,
):
    assert _candidate_exception_is_fatal(message) is True
    _, generator, result = _run_fixture_adapter_search(
        candidate_count=1,
        evaluate_candidate=lambda candidate, _index: {
            **_candidate_result(candidate, fatal=True),
            "status": "semantic_defect",
            "failures": [message],
        },
    )
    receipt = _canonical_product_receipt(result)
    assert generator.generate_calls == [()]
    assert result["fatalFailure"] == "fatal_semantic_or_leak_defect"
    assert result["rejectedCandidates"] == ()
    assert receipt["rejectionFeedbackRows"] == []


def test_only_explicit_all_pass_content_miss_is_an_ordinary_exception():
    assert _candidate_exception_is_fatal("constructed_auction_all_pass") is False
    assert _candidate_exception_is_fatal("auction_cp_sat_rebind_failed") is True
    assert _candidate_exception_is_fatal("escalation_cp_sat_rebind_failed") is True


@pytest.mark.real_clock
def test_qualified_repeat_mismatch_is_a_fatal_determinism_failure():
    evaluation_count = 0

    def evaluate(candidate, _candidate_index):
        nonlocal evaluation_count
        evaluation_count += 1
        row = _candidate_result(candidate, qualified=True)
        if evaluation_count == 2:
            row["auditHash"] = "7" * 64
        return row

    _, _, result = _run_fixture_adapter_search(
        candidate_count=1,
        evaluate_candidate=evaluate,
    )
    receipt = _canonical_product_receipt(result)
    assert evaluation_count == 2
    assert result["stopReason"] == "qualified_repeat_determinism_failure"
    assert result["fatalFailure"] == "qualified_repeat_determinism_failure"
    assert result["qualifiedRow"] is None
    assert result["terminalRepeat"]["deterministic"] is False
    assert receipt["stopReason"] == "fatal_quarantine"
    assert receipt["rejectionFeedbackRows"] == []


@pytest.mark.real_clock
def test_next_solve_deadline_preserves_unapplied_feedback_evidence():
    spec = _sequential_fixture_spec(2)

    class Generator(_SequentialFixtureGenerator):
        def generate_next(self, *args, rejected_candidates, **kwargs):
            if rejected_candidates:
                raise InterruptedError("bounded next solve")
            return super().generate_next(
                *args,
                rejected_candidates=rejected_candidates,
                **kwargs,
            )

    generator = Generator(
        spec,
        root_seed=506,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    _, _, result = _run_fixture_adapter_search(
        seed=506,
        maximum_candidates=2,
        generator=generator,
    )
    assert len(generator.generate_calls) == 1
    assert result["stopReason"] == "candidate_generation_deadline"
    assert result["generationTimedOut"] is True
    assert result["sequentialSearchReceipt"] is None
    assert len(result["rejectedCandidates"]) == 1


@pytest.mark.real_clock
def test_sequential_search_rejects_structural_optimum_drift():
    spec = _sequential_fixture_spec(2)
    generator = _SequentialFixtureGenerator(
        spec,
        root_seed=507,
        dealer=Seat.EAST,
        candidate_count=1,
        continuation_exhausted=False,
    )
    first = generator.candidates[0]
    witness = generator.witness
    drifted = OrToolsCpSatDealGeneratorV1._build_candidate(
        ordinal=1,
        deck_order=witness.deck_order,
        deck_commitment=witness.deck_commitment,
        ordering_score=witness.ordering_score,
        structural_objective_score=witness.structural_objective_score - 1,
        profile_hash=witness.profile_hash,
        model_hash=witness.model_hash,
        search_identity_hash=witness.search_identity_hash,
        backend_version=witness.backend_version,
        no_good_cut_type=witness.no_good_cut_type,
        preceding_no_good_cut_hashes=(generator.cuts[0],),
    )

    class DriftGenerator:
        descriptor = OrToolsCpSatDealGeneratorV1.descriptor

        def generate_next(self, _spec, *, rejected_candidates, **_kwargs):
            return first if not rejected_candidates else drifted

    with pytest.raises(ValueError, match="frozen search authority"):
        _run_fixture_adapter_search(
            seed=507,
            maximum_candidates=2,
            generator=DriftGenerator(),
        )

def test_prelaunch_failure_is_structured_and_runtime_is_cleaned(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "prelaunch-failure.json"
    runtime_parent = tmp_path / "temporary-runtimes"
    runtime_parent.mkdir()
    runtime_paths: list[Path] = []
    real_temporary_directory = combined_pilot.tempfile.TemporaryDirectory

    def tracked_temporary_directory(*args, **kwargs):
        temporary_directory = real_temporary_directory(
            *args,
            dir=runtime_parent,
            **kwargs,
        )
        runtime_paths.append(Path(temporary_directory.name))
        return temporary_directory

    def fail_bootstrap(**_kwargs):
        raise RuntimeError("controlled bootstrap failure")

    monkeypatch.setattr(
        combined_pilot.tempfile,
        "TemporaryDirectory",
        tracked_temporary_directory,
    )
    monkeypatch.setattr(combined_pilot, "build_container", fail_bootstrap)
    args = SimpleNamespace(
        root=root,
        output=output,
        attempts=5,
        candidates_per_seed=4,
        maximum_seed_seconds=120.0,
        maximum_wall_seconds=600.0,
        seed_start=502,
        bundle_version="21.1.22",
        analysis_mode="deep",
    )

    assert combined_pilot._run_finalist_campaign(args) == 1
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "invalid"
    assert evidence["stopReason"] == "campaign_prelaunch_failure"
    assert evidence["fatalFailure"] == "campaign_prelaunch_failure"
    assert evidence["failure"] == {
        "stage": "container_bootstrap",
        "errorType": "RuntimeError",
        "message": "controlled bootstrap failure",
    }
    assert evidence["sourceFrozen"] is True
    assert evidence["temporaryRuntimeCleaned"] is True
    assert evidence["counts"]["seedsEvaluated"] == 0
    assert evidence["counts"]["candidatesEvaluated"] == 0
    assert evidence["sha256"] == canonical_sha256({
        key: value for key, value in evidence.items() if key != "sha256"
    })
    assert runtime_paths
    assert all(not path.exists() for path in runtime_paths)


@pytest.mark.real_clock
def test_completed_seed_disarms_deadline_before_final_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "completed-seed.json"
    runtime_owner = combined_pilot.tempfile.TemporaryDirectory(
        prefix=combined_pilot._FINALIST_RUNTIME_PREFIX
    )
    runtime = Path(runtime_owner.name).resolve()
    progress = runtime / "supervision-progress.json"
    token = "controlled-supervision-token"
    launch_id = "controlled-launch-id"
    combined_pilot._atomic_write_json(
        runtime / combined_pilot._FINALIST_SUPERVISION_MARKER,
        {
            "schemaVersion": 2,
            "tokenDigest": combined_pilot._supervision_token_digest(token, launch_id),
            "launchId": launch_id,
        },
    )
    real_atomic_write = combined_pilot._atomic_write_json
    real_publish_once = combined_pilot._publish_json_once
    progress_updates: list[dict] = []
    evidence_write_progress: list[dict] = []
    source_fingerprint_progress: list[dict] = []

    def recording_atomic_write(path, payload):
        resolved = Path(path).resolve()
        if resolved == progress:
            progress_updates.append(copy.deepcopy(dict(payload)))
        real_atomic_write(path, payload)

    def recording_publish_once(path, payload):
        if Path(path).resolve() == output.resolve():
            evidence_write_progress.append(copy.deepcopy(progress_updates[-1]))
        real_publish_once(path, payload)

    source = {"schemaVersion": 1, "fileCount": 1, "sha256": "a" * 64}
    fingerprint_calls = 0

    def recording_source_fingerprint(_root):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        if fingerprint_calls == 2:
            source_fingerprint_progress.append(copy.deepcopy(progress_updates[-1]))
        return copy.deepcopy(source)

    descriptor = SimpleNamespace(to_dict=lambda: {"id": "controlled-backend"})
    backend = SimpleNamespace(descriptor=descriptor)
    research = SimpleNamespace(
        create_recipe=lambda _payload: {"code": "RCP-CONTROLLED"},
        _compile_deal_spec=lambda *_args, **_kwargs: object(),
        deal_generator=object(),
    )
    container = SimpleNamespace(
        configuration=SimpleNamespace(resolve=lambda: {
            "bundle": {"version": "21.1.22", "revisionId": "bundle-revision"}
        }),
        research=research,
        shutdown=lambda: None,
    )
    snapshot_artifact = {"schemaVersion": 1, "sha256": "1" * 64}
    configuration_core = {
        "schemaVersion": 3,
        "snapshotArtifact": snapshot_artifact,
    }
    configuration_authority = {
        **configuration_core,
        "sha256": canonical_sha256(configuration_core),
    }
    recipe = SimpleNamespace(
        sha256="b" * 64,
        situation={"dealer": "east"},
        configuration=configuration_authority,
    )
    finalist_validation = {
        "schemaVersion": 1,
        "id": "finalist_validation_v1",
        "revision": "9",
        "agentProfileId": "finalist_candidate",
        "maximumTracesPerJob": 20,
        "maximumConfirmationDecisionsPerTrace": 64,
        "maximumTraceSeconds": 300,
        "requireQualifiedAgentReleaseForPublication": True,
        "allowLocalPreview": True,
        "enableExactTacticalViabilityPrecheck": True,
    }
    context = SimpleNamespace(
        bundle_revision_id="bundle-revision",
        bundle=SimpleNamespace(
            sha256="c" * 64,
            component=lambda family: (
                {"finalist_validation": finalist_validation}
                if family == "challenge"
                else {}
            ),
        ),
        effective_runtime_config_hash="d" * 64,
        context_hash="e" * 64,
        exact_search_backend=backend,
        plugin_identities=(),
    )
    agreement = _PolicyTestAgreement()
    agreement.profile = SimpleNamespace(sha256="6" * 64)
    simulator = SimpleNamespace(
        profile_hash="f" * 64,
        agent_release="7" * 64,
        agent_qualification=None,
        decision_agreement=agreement,
        analysis_policy_authority={"mode": "controlled"},
        rules=object(),
    )
    empty_search_receipt = _empty_sequential_search_receipt(517, 2)

    monkeypatch.setattr(combined_pilot, "_atomic_write_json", recording_atomic_write)
    monkeypatch.setattr(
        combined_pilot,
        "_publish_json_once",
        recording_publish_once,
    )
    monkeypatch.setattr(
        combined_pilot,
        "_source_fingerprint",
        recording_source_fingerprint,
    )
    monkeypatch.setattr(combined_pilot, "build_container", lambda **_kwargs: container)
    monkeypatch.setattr(
        combined_pilot,
        "_finalist_campaign_recipe",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        combined_pilot,
        "_validated_finalist_recipe_and_context",
        lambda *_args, **_kwargs: (recipe, context),
    )
    monkeypatch.setattr(
        combined_pilot,
        "_context_finalist_simulator",
        lambda _context: simulator,
    )
    monkeypatch.setattr(
        combined_pilot,
        "_exact_search_backend_runtime_identity",
        lambda _backend: {
            "schemaVersion": 1,
            "backendId": "controlled-backend",
            "backendVersion": "1",
            "backendCapabilities": ["cancellation"],
            "deterministic": True,
            "nativeSelected": False,
            "nativeLibrary": None,
        },
    )
    monkeypatch.setattr(
        combined_pilot,
        "_finalist_control_plane",
        lambda **_kwargs: {
            "finalistAgreementDescriptor": (
                FinalistAgentAgreementV1.descriptor.to_dict()
            ),
            "analysisAuthorities": {
                "primary": {
                    "analysisBudget": {"requested_samples": 192},
                },
                "reanalysis": {
                    "analysisBudget": {"requested_samples": 256},
                },
            },
        },
    )
    monkeypatch.setattr(
        combined_pilot,
        "_run_product_sequential_candidate_search",
        lambda *_args, **_kwargs: {
            "rows": [],
            "qualifiedRow": None,
            "terminalRepeat": None,
            "fatalFailure": None,
            "stopReason": "candidates_exhausted",
            "sequentialSearchReceipt": copy.deepcopy(empty_search_receipt),
            "continuationProbeWallSeconds": 0.0,
        },
    )
    args = SimpleNamespace(
        root=root,
        output=output,
        attempts=1,
        candidates_per_seed=2,
        maximum_seed_seconds=120.0,
        maximum_wall_seconds=180.0,
        seed_start=517,
        bundle_version="21.1.22",
        analysis_mode="deep",
        _finalist_worker=True,
        _supervised_runtime=runtime,
        _supervision_progress=progress,
        _supervision_token=token,
        _supervision_launch_id=launch_id,
    )

    try:
        assert combined_pilot._run_finalist_campaign(args) == 0
        seed_result = next(
            item for item in progress_updates
            if item.get("phase") == "seed_result_recorded"
        )
        assert len(seed_result["completedSeeds"]) == 1
        assert seed_result["completedSeeds"][0]["stopReason"] == (
            "candidates_exhausted"
        )
        assert seed_result["seedDeadlineMonotonic"] is None
        assert seed_result["seedDeadlineReleased"] is True
        assert len(source_fingerprint_progress) == 1
        assert source_fingerprint_progress[0]["seedDeadlineMonotonic"] is None
        assert source_fingerprint_progress[0]["seedDeadlineReleased"] is True
        assert len(evidence_write_progress) == 1
        assert evidence_write_progress[0]["seedDeadlineMonotonic"] is None
        assert evidence_write_progress[0]["seedDeadlineReleased"] is True
        assert evidence_write_progress[0]["phase"] == "supervised_runtime_release"
        evidence = json.loads(output.read_text(encoding="utf-8"))
        assert evidence["schemaVersion"] == 2
        assert evidence["qualificationProfile"]["revision"] == "2"
        assert evidence["supervision"]["schemaVersion"] == 2
        assert evidence["supervision"]["kind"] == FINALIST_CAMPAIGN_WORKER_KIND
        assert evidence["controlPlane"]["finalistAgreementDescriptor"] == (
            FinalistAgentAgreementV1.descriptor.to_dict()
        )
        assert evidence["controlPlane"]["analysisAuthorities"]["primary"][
            "analysisBudget"
        ]["requested_samples"] == 192
        assert evidence["controlPlane"]["analysisAuthorities"]["reanalysis"][
            "analysisBudget"
        ]["requested_samples"] == 256
        assert evidence["status"] == "negative"
        assert evidence["counts"]["seedsEvaluated"] == 1
    finally:
        runtime_owner.cleanup()


def test_supervised_worker_leaves_parent_owned_runtime_while_progress_reader_open(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "release" / "parent-owned-runtime.json"
    runtime_owner = combined_pilot.tempfile.TemporaryDirectory(
        prefix=combined_pilot._FINALIST_RUNTIME_PREFIX
    )
    runtime = Path(runtime_owner.name).resolve()
    progress = runtime / "supervision-progress.json"
    token = "controlled-supervision-token"
    launch_id = "controlled-launch-id"
    combined_pilot._atomic_write_json(
        runtime / combined_pilot._FINALIST_SUPERVISION_MARKER,
        {
            "schemaVersion": 2,
            "tokenDigest": combined_pilot._supervision_token_digest(token, launch_id),
            "launchId": launch_id,
        },
    )
    real_atomic_write = combined_pilot._atomic_write_json
    held_reader: list[object] = []

    def open_reader_without_share_delete() -> object:
        if os.name != "nt":
            return progress.open("rb")
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(progress),
            0x80000000,
            0x00000001 | 0x00000002,
            None,
            3,
            0x00000080,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return (kernel32, handle)

    def publish_and_hold_reader(path, payload):
        real_atomic_write(path, payload)
        if (
            Path(path) == progress
            and payload.get("phase") == "supervised_runtime_release"
        ):
            held_reader.append(open_reader_without_share_delete())

    def fail_bootstrap(**_kwargs):
        raise RuntimeError("controlled bootstrap failure")

    monkeypatch.setattr(
        combined_pilot,
        "_atomic_write_json",
        publish_and_hold_reader,
    )
    monkeypatch.setattr(combined_pilot, "build_container", fail_bootstrap)
    args = SimpleNamespace(
        root=root,
        output=output,
        attempts=1,
        candidates_per_seed=4,
        maximum_seed_seconds=120.0,
        maximum_wall_seconds=180.0,
        seed_start=505,
        bundle_version="21.1.22",
        analysis_mode="deep",
        _finalist_worker=True,
        _supervised_runtime=runtime,
        _supervision_progress=progress,
        _supervision_token=token,
        _supervision_launch_id=launch_id,
    )

    try:
        assert combined_pilot._run_finalist_campaign(args) == 1
        evidence = json.loads(output.read_text(encoding="utf-8"))
        assert evidence["status"] == "invalid"
        assert evidence["stopReason"] == "campaign_prelaunch_failure"
        assert evidence["failure"]["message"] == "controlled bootstrap failure"
        assert evidence["cleanupFailure"] is None
        assert evidence["temporaryRuntimeCleaned"] is False
        assert evidence["supervision"]["runtimeOwner"] == "supervisor_process"
        assert evidence["supervision"]["parentRuntimeFinalized"] is False
        assert held_reader
        assert runtime.exists() is True
        assert progress.exists() is True
    finally:
        for reader in held_reader:
            if isinstance(reader, tuple):
                reader[0].CloseHandle(reader[1])
            else:
                reader.close()
        runtime_owner.cleanup()
    assert runtime.exists() is False


def _policy_test_analysis_budget(*, reanalysis: bool) -> dict[str, object]:
    if reanalysis:
        return {
            "requested_samples": 256,
            "exact_allocation_limit": 256,
            "exact_search_max_remaining": 9,
            "maximum_sampling_attempts": 6000,
            "exact_search_max_nodes": 120000,
            "exact_search_max_seconds": 0.8,
            "exact_search_max_depth": 32,
            "exact_search_max_cache_entries": 100000,
        }
    return {
        "requested_samples": 192,
        "exact_allocation_limit": 128,
        "exact_search_max_remaining": 9,
        "maximum_sampling_attempts": 4000,
        "exact_search_max_nodes": 80000,
        "exact_search_max_seconds": 0.4,
        "exact_search_max_depth": 32,
        "exact_search_max_cache_entries": 100000,
    }


class _PolicyTestAgreement:
    descriptor = FinalistAgentAgreementV1.descriptor
    exact_primary_policy = True
    primary_trace_analysis_profile_hash = "2" * 64
    primary_analysis_identity_hash = "3" * 64
    reanalysis_trace_analysis_profile_hash = "4" * 64
    reanalysis_analysis_identity_hash = "5" * 64
    primary_analysis_budget = _policy_test_analysis_budget(reanalysis=False)
    independent_analysis_budget = _policy_test_analysis_budget(reanalysis=True)
    profile = SimpleNamespace(sha256="6" * 64)

    @staticmethod
    def actor_observer_seed(
        *,
        public_prefix_hash: str,
        decision_kind: str,
        acting_seat: Seat,
        cohort_index: int,
    ) -> int:
        identity = (
            f"{public_prefix_hash}|{decision_kind}|"
            f"{acting_seat.name}|{cohort_index}"
        )
        return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)


def _policy_test_simulator() -> SimpleNamespace:
    return SimpleNamespace(
        analysis_policy_authority={"mode": "explicit_governed"},
        decision_agreement=_PolicyTestAgreement(),
    )


def _trace_with_policies(*, hidden_feature: bool = False):
    bid = DecisionLedgerRecordV1(
        "BID-1",
        1,
        Seat.NORTH,
        "bid",
        "bidding",
        "a" * 64,
        "b" * 64,
        ({"action": "pass", "trumpSuit": None}, {"action": "sun", "trumpSuit": None}),
        {"action": "sun", "trumpSuit": None},
        False,
        "c" * 64,
    )
    play = DecisionLedgerRecordV1(
        "PLAY-2",
        2,
        Seat.WEST,
        "play",
        "playing",
        "d" * 64,
        "e" * 64,
        ({"code": "S7"}, {"code": "S8"}),
        {"code": "S8"},
        False,
        "f" * 64,
    )
    simulator = _policy_test_simulator()
    agreement = simulator.decision_agreement

    def cohort(
        base: dict[str, object],
        record: DecisionLedgerRecordV1,
        cohort_index: int,
    ) -> dict[str, object]:
        primary = cohort_index == 0
        return {
            **base,
            "cohortIndex": cohort_index,
            "intendedBestAction": dict(record.chosen_action),
            "seed": agreement.actor_observer_seed(
                public_prefix_hash=record.public_prefix_hash,
                decision_kind=record.decision_kind,
                acting_seat=record.acting_seat,
                cohort_index=cohort_index,
            ),
            "traceAnalysisProfileHash": (
                agreement.primary_trace_analysis_profile_hash
                if primary
                else agreement.reanalysis_trace_analysis_profile_hash
            ),
            "analysisIdentityHash": (
                agreement.primary_analysis_identity_hash
                if primary
                else agreement.reanalysis_analysis_identity_hash
            ),
            "analysisBudget": dict(
                agreement.primary_analysis_budget
                if primary
                else agreement.independent_analysis_budget
            ),
            "analysisAuthorityRole": (
                "deep_confirmation" if primary else "finalist_reanalysis"
            ),
            "confidence": 0.9,
            "analysisExhaustiveSupport": True,
            "analysisPopulationCoverageComplete": True,
            "analysisSolverCompletion": "exact",
            "analysisTruthClass": "exact_optimal",
        }

    bid_base = {
        "strategyId": "adaptive_observer_safe_bidding_v2",
        "profileId": "finalist_candidate",
        "observerSafe": True,
        "selectedAction": dict(bid.chosen_action),
        "hiddenDealCount": 16,
    }
    bid_cohorts = [cohort(bid_base, bid, index) for index in range(3)]
    bid_primary = bid_cohorts[0]
    bid_policy = {
        "source": "finalist_multi_cohort_agent",
        "decisionKind": "bid",
        "selectionMode": "primary_policy_only",
        "primaryActionPreserved": True,
        "observerSafe": True,
        "qualificationBlocked": False,
        "featuredTruthEligible": True,
        "selectedAction": dict(bid.chosen_action),
        "hiddenDealCount": 16,
        "traceAnalysisProfileHash": (
            agreement.primary_trace_analysis_profile_hash
        ),
        "analysisIdentityHash": agreement.primary_analysis_identity_hash,
        "analysisBudget": dict(agreement.primary_analysis_budget),
        "analysisAuthorityRole": "deep_confirmation",
        "status": "strict_confirmed",
        "cohortCount": 3,
        "agreementCount": 3,
        "majorityActionHash": canonical_sha256(dict(bid.chosen_action)),
        "primarySeed": bid_cohorts[0]["seed"],
        "independentSeed": bid_cohorts[1]["seed"],
        "thirdSeed": bid_cohorts[2]["seed"],
        "primaryTrace": dict(bid_primary),
        "cohorts": [dict(item) for item in bid_cohorts],
    }
    if hidden_feature:
        bid_policy["deckOrder"] = list(range(32))
    play_base = {
        "strategyId": "adaptive_observer_safe_v2",
        "profileId": "finalist_candidate",
        "observerSafe": True,
        "selectedAction": dict(play.chosen_action),
        "analysisHiddenDeals": 32,
        "analysisMethod": "paired_actor_safe",
        "analysisPolicyAuthority": {"mode": "explicit_governed"},
    }
    play_cohorts = [cohort(play_base, play, index) for index in range(3)]
    play_primary = play_cohorts[0]
    play_policy = {
        "source": "finalist_multi_cohort_agent",
        "decisionKind": "play",
        "selectionMode": "primary_policy_only",
        "primaryActionPreserved": True,
        "observerSafe": True,
        "qualificationBlocked": False,
        "featuredTruthEligible": True,
        "selectedAction": dict(play.chosen_action),
        "analysisHiddenDeals": 32,
        "traceAnalysisProfileHash": (
            agreement.primary_trace_analysis_profile_hash
        ),
        "analysisIdentityHash": agreement.primary_analysis_identity_hash,
        "analysisBudget": dict(agreement.primary_analysis_budget),
        "analysisAuthorityRole": "deep_confirmation",
        "status": "strict_confirmed",
        "cohortCount": 3,
        "agreementCount": 3,
        "majorityActionHash": canonical_sha256(dict(play.chosen_action)),
        "primarySeed": play_cohorts[0]["seed"],
        "independentSeed": play_cohorts[1]["seed"],
        "thirdSeed": play_cohorts[2]["seed"],
        "primaryTrace": dict(play_primary),
        "cohorts": [dict(item) for item in play_cohorts],
    }
    return SimpleNamespace(
        decision_ledger=DecisionLedgerV1.build("GAME", (bid, play)),
        rules_hash="7" * 64,
        trusted_final_state={
            "generation": {
                "decisionAgreementDescriptor": agreement.descriptor.to_dict(),
                "decisionAgreementProfileHash": agreement.profile.sha256,
            },
            "decisionPolicyTraces": [
                {"ordinal": 1, "trace": bid_policy},
                {"ordinal": 2, "trace": play_policy},
            ]
        },
    )


def test_combined_policy_evidence_requires_shared_actor_safe_actions() -> None:
    simulator = _policy_test_simulator()
    trace = _trace_with_policies()
    evidence = _policy_evidence(trace, simulator)
    assert evidence["schemaVersion"] == 2
    assert evidence["profileId"] == FINALIST_CAMPAIGN_PROFILE_ID
    assert evidence["agreementDescriptor"] == (
        FinalistAgentAgreementV1.descriptor.to_dict()
    )
    assert evidence["analysisAuthorities"]["primary"] == {
        "role": "deep_confirmation",
        "profileHash": "2" * 64,
        "identityHash": "3" * 64,
        "budget": _policy_test_analysis_budget(reanalysis=False),
    }
    assert evidence["analysisAuthorities"]["reanalysis"] == {
        "role": "finalist_reanalysis",
        "profileHash": "4" * 64,
        "identityHash": "5" * 64,
        "budget": _policy_test_analysis_budget(reanalysis=True),
    }
    assert evidence["nonForcedBidCount"] == 1
    assert evidence["nonForcedPlayCount"] == 1
    assert evidence["oneSharedActionAcrossIndistinguishableWorlds"] is True
    assert evidence["zeroObserverLeaks"] is True
    assert evidence["observerSafetyPassed"] is True
    assert evidence["primaryPolicyValid"] is True
    assert evidence["stabilityResolved"] is True
    assert evidence["stabilityFailures"] == []
    assert evidence["primaryPolicyFailures"] == []
    assert evidence["failures"] == []
    assert evidence["competentDecisionIds"] == []
    assert evidence["unresolvedDecisionIds"] == []
    assert evidence["featuredTruthBlockedDecisionIds"] == []
    assert evidence["bids"][0]["hiddenWorldCount"] == 16
    assert evidence["bids"][0]["cohortSupport"][0][
        "hiddenWorldCount"
    ] == 16
    assert evidence["plays"][0]["hiddenWorldCount"] == 32
    assert evidence["plays"][0]["cohortSupport"][0][
        "hiddenWorldCount"
    ] == 32
    assert evidence["auctionSelectionAuthority"] == (
        "finalist_multi_cohort_agent_primary_policy"
    )
    assert evidence["playSelectionAuthority"] == (
        "finalist_multi_cohort_agent_primary_policy"
    )
    assert evidence["cpSatAuctionRole"] == "exact_required_action_rebind_only"
    assert evidence["cpSatPlayRole"] == "exact_required_action_rebind_only"
    for item in trace.trusted_final_state["decisionPolicyTraces"]:
        cohorts = item["trace"]["cohorts"]
        assert [row["analysisBudget"]["requested_samples"] for row in cohorts] == [
            192,
            256,
            256,
        ]
        assert [row["analysisAuthorityRole"] for row in cohorts] == [
            "deep_confirmation",
            "finalist_reanalysis",
            "finalist_reanalysis",
        ]
        assert len({row["seed"] for row in cohorts}) == 3
        assert cohorts[0]["traceAnalysisProfileHash"] != (
            cohorts[1]["traceAnalysisProfileHash"]
        )
        assert cohorts[0]["analysisIdentityHash"] != (
            cohorts[1]["analysisIdentityHash"]
        )

    leaking = _policy_evidence(
        _trace_with_policies(hidden_feature=True), simulator
    )
    assert leaking["zeroObserverLeaks"] is False
    assert leaking["observerSafetyPassed"] is False
    assert leaking["oneSharedActionAcrossIndistinguishableWorlds"] is False
    assert leaking["primaryPolicyValid"] is False
    assert "bid_policy_hidden_feature" in leaking["failures"]
    assert _fatal_policy_defects(leaking) == ["bid_policy_hidden_feature"]


def test_safe_unresolved_agreement_is_not_mislabeled_as_observer_failure():
    simulator = _policy_test_simulator()
    trace = _trace_with_policies()
    policy_rows = copy.deepcopy(
        trace.trusted_final_state["decisionPolicyTraces"]
    )
    for item in policy_rows:
        policy = item["trace"]
        policy.update({
            "qualificationBlocked": True,
            "featuredTruthEligible": False,
            "status": "unresolved",
            "agreementCount": 0,
            "majorityActionHash": None,
            "reasons": [
                "cohort_reversal",
                "insufficient_qualified_agreement",
                "primary_action_not_confirmed",
            ],
            "initialReasons": [
                "cohort_reversal",
                "insufficient_qualified_agreement",
            ],
        })
    unresolved = SimpleNamespace(
        decision_ledger=trace.decision_ledger,
        rules_hash=trace.rules_hash,
        trusted_final_state={
            "generation": copy.deepcopy(trace.trusted_final_state["generation"]),
            "decisionPolicyTraces": policy_rows,
        },
    )

    evidence = _policy_evidence(unresolved, simulator)

    assert evidence["failures"] == [
        "bid_policy_blocked",
        "play_policy_blocked",
    ]
    assert evidence["stabilityFailures"] == evidence["failures"]
    assert evidence["stabilityResolved"] is False
    assert evidence["primaryPolicyFailures"] == []
    assert evidence["primaryPolicyValid"] is True
    assert evidence["zeroObserverLeaks"] is True
    assert evidence["observerSafetyPassed"] is True
    assert evidence["oneSharedActionAcrossIndistinguishableWorlds"] is True
    assert evidence["competentDecisionIds"] == []
    assert evidence["unresolvedDecisionIds"] == ["BID-1", "PLAY-2"]
    assert evidence["featuredTruthBlockedDecisionIds"] == [
        "BID-1",
        "PLAY-2",
    ]
    assert _fatal_policy_defects(evidence) == []
    assert _observer_safety_passed(
        {
            "informationSetComplete": True,
            "sequentialResponseComplete": True,
        },
        evidence,
    ) is True
    gates = _finalist_policy_gate_evidence(
        audit={
            "informationSetComplete": True,
            "sequentialResponseComplete": True,
        },
        policy=evidence,
        agreement_unresolved=2,
    )
    assert gates == {
        "zeroObserverLeaks": True,
        "observerSafety": True,
        "sharedActionAcrossWorlds": True,
        "finalistPrimaryAuctionAndPlay": True,
        "finalistAgreementResolved": False,
    }
    assert all(gates.values()) is False
    assert evidence["bids"][0]["agreementReasons"] == [
        "cohort_reversal",
        "insufficient_qualified_agreement",
        "primary_action_not_confirmed",
    ]
    assert evidence["bids"][0]["initialAgreementReasons"] == [
        "cohort_reversal",
        "insufficient_qualified_agreement",
    ]


def test_robust_competent_surrounding_decisions_remain_featured_blocked():
    simulator = _policy_test_simulator()
    trace = _trace_with_policies()
    policy_rows = copy.deepcopy(
        trace.trusted_final_state["decisionPolicyTraces"]
    )
    for item in policy_rows:
        item["trace"].update({
            "qualificationBlocked": False,
            "featuredTruthEligible": False,
            "status": "primary_robust_competent",
            "reasons": ["primary_action_not_confirmed"],
            "initialReasons": ["primary_action_not_confirmed"],
        })
    competent = SimpleNamespace(
        decision_ledger=trace.decision_ledger,
        rules_hash=trace.rules_hash,
        trusted_final_state={
            "generation": copy.deepcopy(trace.trusted_final_state["generation"]),
            "decisionPolicyTraces": policy_rows,
        },
    )

    evidence = _policy_evidence(competent, simulator)

    assert evidence["failures"] == []
    assert evidence["stabilityResolved"] is True
    assert evidence["primaryPolicyValid"] is True
    assert evidence["competentDecisionIds"] == ["BID-1", "PLAY-2"]
    assert evidence["unresolvedDecisionIds"] == []
    assert evidence["featuredTruthBlockedDecisionIds"] == [
        "BID-1",
        "PLAY-2",
    ]
    assert all(
        row["featuredTruthEligible"] is False
        for row in [*evidence["bids"], *evidence["plays"]]
    )


def test_policy_evidence_persists_recursively_immutable_action_as_plain_json(
) -> None:
    selected_action = MappingProxyType({
        "action": "play",
        "card": MappingProxyType({"code": "S8"}),
    })
    record = SimpleNamespace(
        decision_id="PLAY-1",
        ordinal=1,
        acting_seat=Seat.WEST,
        decision_kind="play",
        forced=False,
        chosen_action=selected_action,
        public_prefix_hash="a" * 64,
        legal_actions=({"code": "S7"}, selected_action),
    )
    simulator = _policy_test_simulator()
    agreement = simulator.decision_agreement
    cohorts = []
    for index in range(3):
        primary_authority = index == 0
        cohorts.append({
            "cohortIndex": index,
            "strategyId": "adaptive_observer_safe_v2",
            "profileId": "finalist_candidate",
            "analysisMethod": "paired_actor_safe",
            "analysisPolicyAuthority": {"mode": "explicit_governed"},
            "analysisHiddenDeals": 32,
            "intendedBestAction": selected_action,
            "traceAnalysisProfileHash": (
                agreement.primary_trace_analysis_profile_hash
                if primary_authority
                else agreement.reanalysis_trace_analysis_profile_hash
            ),
            "analysisIdentityHash": (
                agreement.primary_analysis_identity_hash
                if primary_authority
                else agreement.reanalysis_analysis_identity_hash
            ),
            "analysisBudget": dict(
                agreement.primary_analysis_budget
                if primary_authority
                else agreement.independent_analysis_budget
            ),
            "analysisAuthorityRole": (
                "deep_confirmation"
                if primary_authority
                else "finalist_reanalysis"
            ),
            "seed": agreement.actor_observer_seed(
                public_prefix_hash=record.public_prefix_hash,
                decision_kind=record.decision_kind,
                acting_seat=record.acting_seat,
                cohort_index=index,
            ),
        })
    policy = {
        "source": "finalist_multi_cohort_agent",
        "decisionKind": "play",
        "selectionMode": "primary_policy_only",
        "primaryActionPreserved": True,
        "observerSafe": True,
        "qualificationBlocked": False,
        "featuredTruthEligible": True,
        "selectedAction": selected_action,
        "analysisHiddenDeals": 32,
        "traceAnalysisProfileHash": agreement.primary_trace_analysis_profile_hash,
        "analysisIdentityHash": agreement.primary_analysis_identity_hash,
        "analysisBudget": dict(agreement.primary_analysis_budget),
        "analysisAuthorityRole": "deep_confirmation",
        "status": "strict_confirmed",
        "cohortCount": 3,
        "agreementCount": 3,
        "majorityActionHash": canonical_sha256({
            "action": "play",
            "card": {"code": "S8"},
        }),
        "primarySeed": cohorts[0]["seed"],
        "independentSeed": cohorts[1]["seed"],
        "thirdSeed": cohorts[2]["seed"],
        "primaryTrace": cohorts[0],
        "cohorts": cohorts,
    }
    trace = SimpleNamespace(
        decision_ledger=SimpleNamespace(
            records=(record,),
            sha256="b" * 64,
        ),
        rules_hash="7" * 64,
        trusted_final_state={
            "generation": {
                "decisionAgreementDescriptor": agreement.descriptor.to_dict(),
                "decisionAgreementProfileHash": agreement.profile.sha256,
            },
            "decisionPolicyTraces": ({"ordinal": 1, "trace": policy},)
        },
    )
    evidence = _policy_evidence(trace, simulator)

    assert evidence["failures"] == []
    assert evidence["plays"][0]["selectedAction"] == {
        "action": "play",
        "card": {"code": "S8"},
    }
    json.dumps(evidence, allow_nan=False)


def test_explicit_observer_unsafe_policy_is_a_fatal_defect_without_leak_keys():
    simulator = _policy_test_simulator()
    trace = _trace_with_policies()
    policy_rows = copy.deepcopy(
        trace.trusted_final_state["decisionPolicyTraces"]
    )
    policy_rows[0]["trace"]["observerSafe"] = False
    unsafe = SimpleNamespace(
        decision_ledger=trace.decision_ledger,
        rules_hash=trace.rules_hash,
        trusted_final_state={
            "generation": copy.deepcopy(trace.trusted_final_state["generation"]),
            "decisionPolicyTraces": policy_rows,
        },
    )

    evidence = _policy_evidence(unsafe, simulator)

    assert evidence["zeroObserverLeaks"] is True
    assert "bid_not_observer_safe" in evidence["failures"]
    assert _fatal_policy_defects(evidence) == ["bid_not_observer_safe"]


def test_combined_policy_evidence_rejects_profile_or_authority_drift() -> None:
    simulator = _policy_test_simulator()
    trace = _trace_with_policies()
    policy_rows = copy.deepcopy(
        trace.trusted_final_state["decisionPolicyTraces"]
    )
    policy_rows[0]["trace"]["selectionMode"] = "majority_policy"
    policy_rows[0]["trace"]["analysisIdentityHash"] = "4" * 64
    policy_rows[0]["trace"]["cohorts"][1]["seed"] += 1
    policy_rows[1]["trace"]["primaryTrace"]["profileId"] = "foundry_balanced"
    policy_rows[1]["trace"]["traceAnalysisProfileHash"] = "5" * 64
    policy_rows[1]["trace"]["cohorts"][2]["analysisIdentityHash"] = "6" * 64
    drifted = SimpleNamespace(
        decision_ledger=trace.decision_ledger,
        rules_hash=trace.rules_hash,
        trusted_final_state={
            "generation": copy.deepcopy(trace.trusted_final_state["generation"]),
            "decisionPolicyTraces": policy_rows,
        },
    )
    evidence = _policy_evidence(drifted, simulator)
    assert "bid_selection_mode_drift" in evidence["failures"]
    assert "bid_primary_analysis_authority_drift" in evidence["failures"]
    assert "bid_cohort_1_seed_drift" in evidence["failures"]
    assert "wrong_play_profile" in evidence["failures"]
    assert "play_primary_analysis_authority_drift" in evidence["failures"]
    assert "play_cohort_2_analysis_authority_drift" in evidence["failures"]
    assert evidence["oneSharedActionAcrossIndistinguishableWorlds"] is False
    fatal = _fatal_policy_defects(evidence)
    assert "bid_selection_mode_drift" in fatal
    assert "bid_primary_analysis_authority_drift" in fatal
    assert "bid_cohort_1_seed_drift" in fatal
    assert "wrong_play_profile" in fatal
    assert "play_primary_analysis_authority_drift" in fatal
    assert "play_cohort_2_analysis_authority_drift" in fatal
