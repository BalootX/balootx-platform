from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import platform
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import ortools

from baloot_platform.agents.profiles import canonical_plain_profile
from baloot_platform.analysis.decision_scores import (
    canonical_json_bytes_v1,
    legal_action_set_hash_v1,
)
from baloot_platform.challenge.replay import CompleteTraceReplayerV1
from baloot_platform.challenge.analysis_pipeline import (
    TraceAnalysisProfileV1,
    _validate_forced_legality_proof_v1,
    build_finalist_reanalysis_trace_profile_v1,
)
from baloot_platform.challenge.decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.finalist_competence import (
    canonical_score_support_classification_v1,
)
from baloot_platform.challenge.finalist_validation import (
    FinalistValidationProfileV1,
)
from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.analysis.search_backend import (
    NativeExactSearchBackendV3,
    PythonReferenceSearchBackend,
)
from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Seat
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.contracts import (
    RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
    RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
    ResearchConfigurationAuthorityV1,
    ResearchRecipeV3,
    canonical_sha256,
)
from baloot_platform.research.decision_selection import (
    apply_decision_selection_qualification_v1,
    evaluate_decision_selection_v1,
)
from baloot_platform.research.sequential_candidate_search import (
    SequentialCandidateEvaluationV1,
    SequentialCandidateOutcomeKindV1,
    SequentialCandidateSearchReceiptV1,
    run_sequential_candidate_search_v1,
)
from baloot_platform.service.container import build_container

from scripts.run_strategic_play_pilot import (
    _strategic_profile,
)


def _recipe(
    research: Any,
    *,
    seed_start: int,
    attempts: int,
    auction_mode: str,
    play_mode: str,
    analysis_mode: str,
    candidates_per_seed: int,
    tactic: str,
    tactic_mode: str,
    tactical_viability_pre_screen: bool,
    tactical_viability_maximum_decisions: int,
    tactical_deal_balance: bool,
    strategic_minimum_played_cards: int,
) -> dict[str, Any]:
    capabilities = research.capabilities()
    active = dict(capabilities["research"])
    resources: dict[str, Any] = {
        "generationLane": "ortools",
        "seedStart": seed_start,
        "seedCount": attempts,
        "seedsPerShard": min(attempts, 5),
        "candidatesPerSeed": candidates_per_seed,
        "solverTimeoutSeconds": 15,
        "maximumConflicts": 250_000,
        "analysisMode": analysis_mode,
        "deepVerificationPolicy": "required_gates",
        "auctionConstructionMode": auction_mode,
        "playConstructionMode": play_mode,
    }
    if auction_mode == "cp_sat_actor_safe_decision_v1":
        resources["strategicAuction"] = {
            "schemaVersion": 1,
            "requestedSamples": RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
            "integerScale": 1_000,
            "maximumConflicts": 50_000,
            "maximumSelectionSeconds": 5.0,
            "objective": "expected_round_margin",
            "tieBreak": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
            "requireOptimalForQualification": True,
        }
    if play_mode == "cp_sat_actor_safe_late_game_v1":
        strategic_play = _strategic_profile()
        strategic_play["minimumPlayedCards"] = strategic_minimum_played_cards
        strategic_play["maximumPlies"] = 32 - strategic_minimum_played_cards
        resources["strategicPlay"] = strategic_play
    if tactical_viability_pre_screen:
        resources["tacticalViabilityPreScreen"] = {
            "schemaVersion": 1,
            "id": "exact_tactical_viability_screen_v1",
            "revision": "1",
            "enabled": True,
            "maximumDecisions": tactical_viability_maximum_decisions,
            "requiredTacticsOnly": True,
            "rejectWhenExactAbsent": True,
        }
    if tactical_deal_balance:
        resources["tacticalDealBalance"] = {
            "schemaVersion": 1,
            "id": "balanced_tactical_deal_v1",
            "revision": 1,
            "enabled": True,
            "maximumTeamHighCardDifference": 2,
            "minimumHighCardsPerSeat": 2,
            "maximumHighCardsPerSeat": 4,
            "minimumTargetLowCards": 1,
        }
    core = {
        "schemaVersion": 2,
        "titleAr": "اختبار مؤهل لاستراتيجية OR-Tools",
        "output": {
            "kind": "laffa_collection",
            "resultCount": attempts * candidates_per_seed,
        },
        "profiles": {
            "domainHash": capabilities["sha256"],
            "bundleHash": capabilities["bundleHash"],
            "rulesHash": capabilities["rulesHash"],
            "conventionHash": capabilities["conventionHash"],
            "agentRelease": active["defaultAgentRelease"],
            "analysisProfileHash": active["analysisProfiles"][analysis_mode],
            "scoreProfileHash": active["defaultScoreProfileHash"],
        },
        "situation": {
            "dealer": "east",
            "buyer": None,
            "purchaseRound": None,
            "trumpSuit": None,
            "contracts": [],
            "matchScore": [0, 0],
            "requiredCards": {},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [{
            "kind": tactic,
            "mode": tactic_mode,
            "minimumScore": 1,
            "weight": 1,
            "perspectives": [],
        }],
        "qualification": {
            "minimumTruth": "robust_best_known",
            "minimumConfidence": 0.75,
            "minimumSurroundingQuality": 70,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0,
        },
        "resources": resources,
        "diversity": {
            "schemaVersion": 1,
            "cellDimensions": [
                "contract", "tactical_detector", "acting_seat", "source"
            ],
            "topKPerCell": 3,
            "rejectedAuditModulus": 100,
        },
        "renderPolicy": {"mode": "on_demand"},
        "status": "published",
    }
    return core


LEGACY_HYBRID_FINALIST_PROFILE_ID = (
    "actor_safe_cp_sat_auction_plus_finalist_play_v1"
)
FINALIST_CAMPAIGN_PROFILE_ID = "finalist_agent_auction_and_play_v1"
FINALIST_CAMPAIGN_PROFILE_REVISION = "2"
FINALIST_CAMPAIGN_ANALYSIS_MODE = "deep"
FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION = 2
FINALIST_CAMPAIGN_EXPERIMENT_ID = (
    "finalist_agent_auction_and_play_qualification_campaign_v2"
)
FINALIST_CAMPAIGN_WORKER_KIND = "owned_finalist_campaign_worker_v2"


def _require_finalist_campaign_analysis_mode(analysis_mode: str) -> None:
    if str(analysis_mode) != FINALIST_CAMPAIGN_ANALYSIS_MODE:
        raise ValueError(
            f"{FINALIST_CAMPAIGN_PROFILE_ID} requires "
            f"analysisMode={FINALIST_CAMPAIGN_ANALYSIS_MODE}"
        )


def _require_qualification_profile(profile_id: str) -> None:
    if str(profile_id) == LEGACY_HYBRID_FINALIST_PROFILE_ID:
        raise ValueError(
            f"{LEGACY_HYBRID_FINALIST_PROFILE_ID} is historical, "
            "non-qualifying evidence; use the cp_sat_auction diagnostic lane "
            f"or {FINALIST_CAMPAIGN_PROFILE_ID}"
        )
    if str(profile_id) != FINALIST_CAMPAIGN_PROFILE_ID:
        raise ValueError(f"unsupported qualification profile: {profile_id}")


def _require_current_strategic_auction_mirror(context: Any) -> int:
    """Fail before publication if the selected bundle outgrows this V3 catalog."""

    selected = int(context.analyzers["deep"].profile.bid_samples)
    if selected != RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3:
        raise ValueError(
            "the selected published bundle deep bidSamples do not match the "
            "current V3 strategic-auction compatibility mirror"
        )
    return selected

_FINGERPRINT_SOURCE_DIRECTORIES = (
    "src/baloot_platform",
    "configs",
    "schemas",
    "native",
)
_FINGERPRINT_SOURCE_FILES = (
    "pyproject.toml",
    "scripts/run_combined_strategy_pilot.py",
    "scripts/run_strategic_play_pilot.py",
)
_OUTPUT_FORBIDDEN_SOURCE_DIRECTORIES = (
    "src",
    "configs",
    "schemas",
    "scripts",
    "native",
)
_FINALIST_RUNTIME_PREFIX = "baloot-finalist-campaign-"
_FINALIST_SUPERVISION_MARKER = ".balootx-finalist-supervision"
_FINALIST_SUPERVISION_ENV = "BALOOT_FINALIST_SUPERVISION_TOKEN"
_FINALIST_SUPERVISION_POLL_SECONDS = 0.05
_FINALIST_PROCESS_TERMINATION_WAIT_SECONDS = 5.0
_FINALIST_EVIDENCE_FINALIZATION_RESERVE_SECONDS = 5.0
_FINALIST_GLOBAL_CLEANUP_RESERVE_SECONDS = (
    3 * _FINALIST_PROCESS_TERMINATION_WAIT_SECONDS
    + _FINALIST_EVIDENCE_FINALIZATION_RESERVE_SECONDS
)
_WINDOWS_ATOMIC_REPLACE_ATTEMPTS = 40
_WINDOWS_ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS = 0.002
_WINDOWS_ATOMIC_REPLACE_MAXIMUM_BACKOFF_SECONDS = 0.025


def _windows_replace_contention(exc: OSError) -> bool:
    return os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}


def _replace_with_bounded_windows_retry(source: Path, destination: Path) -> None:
    backoff = _WINDOWS_ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS
    for attempt in range(_WINDOWS_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if (
                not _windows_replace_contention(exc)
                or attempt + 1 >= _WINDOWS_ATOMIC_REPLACE_ATTEMPTS
            ):
                raise
            time.sleep(backoff)
            backoff = min(
                _WINDOWS_ATOMIC_REPLACE_MAXIMUM_BACKOFF_SECONDS,
                backoff * 2,
            )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _replace_with_bounded_windows_retry(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _publish_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish one immutable receipt without an overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.receipt.tmp"
    )
    document = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_supervision_progress(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _seed_result_deadline_update(
    *,
    seed_deadline_monotonic: float,
    result_recorded_monotonic: float,
) -> dict[str, float | bool | None]:
    """Release a seed deadline only when its complete result commits in time."""

    released = result_recorded_monotonic < seed_deadline_monotonic
    return {
        "seedDeadlineMonotonic": None if released else seed_deadline_monotonic,
        "seedResultRecordedMonotonic": result_recorded_monotonic,
        "seedDeadlineReleased": released,
    }


def _supervision_token_digest(token: str, launch_id: str) -> str:
    return hmac.digest(token.encode(), launch_id.encode(), "sha256").hex()


def _validated_supervised_runtime(
    args: argparse.Namespace,
) -> tuple[Path, Path, str] | None:
    worker_requested = bool(getattr(args, "_finalist_worker", False))
    raw_runtime = getattr(args, "_supervised_runtime", None)
    raw_progress = getattr(args, "_supervision_progress", None)
    token = str(getattr(args, "_supervision_token", "") or "")
    launch_id = str(getattr(args, "_supervision_launch_id", "") or "")
    if (
        not worker_requested
        and raw_runtime is None
        and raw_progress is None
        and not token
        and not launch_id
    ):
        return None
    if not worker_requested:
        raise ValueError("finalist supervision arguments require worker mode")
    if (
        raw_runtime is None
        or raw_progress is None
        or not token
        or not launch_id
    ):
        raise ValueError("incomplete finalist campaign supervision authority")
    runtime = Path(raw_runtime).resolve()
    progress = Path(raw_progress).resolve()
    system_temporary = Path(tempfile.gettempdir()).resolve()
    if (
        runtime.parent != system_temporary
        or not runtime.name.startswith(_FINALIST_RUNTIME_PREFIX)
        or progress.parent != runtime
    ):
        raise ValueError("invalid finalist campaign supervision runtime")
    marker = runtime / _FINALIST_SUPERVISION_MARKER
    try:
        declared = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("missing finalist campaign supervision marker") from exc
    if (
        not isinstance(declared, dict)
        or declared.get("schemaVersion") != 2
        or declared.get("launchId") != launch_id
        or not secrets.compare_digest(
            str(declared.get("tokenDigest") or "").encode(),
            _supervision_token_digest(token, launch_id).encode(),
        )
    ):
        raise ValueError("invalid finalist campaign supervision token")
    return runtime, progress, launch_id


def _finalist_worker_command(
    args: argparse.Namespace,
    *,
    runtime: Path,
    progress: Path,
    launch_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--root",
        str(Path(args.root).resolve()),
        "--qualification-profile",
        FINALIST_CAMPAIGN_PROFILE_ID,
        "--analysis-mode",
        str(args.analysis_mode),
        "--attempts",
        str(int(args.attempts)),
        "--seed-start",
        str(int(args.seed_start)),
        "--candidates-per-seed",
        str(int(args.candidates_per_seed)),
        "--bundle-version",
        str(args.bundle_version),
        "--maximum-wall-seconds",
        str(float(args.maximum_wall_seconds)),
        "--maximum-seed-seconds",
        str(float(args.maximum_seed_seconds)),
        "--output",
        str((runtime / "worker-evidence.json").resolve()),
        "--_finalist-worker",
        "--_supervised-runtime",
        str(runtime),
        "--_supervision-progress",
        str(progress),
        "--_supervision-launch-id",
        launch_id,
    ]


def _terminate_owned_process_tree(
    process: subprocess.Popen[Any],
    *,
    wait_seconds: float = _FINALIST_PROCESS_TERMINATION_WAIT_SECONDS,
) -> dict[str, Any]:
    requested = process.poll() is None
    tree_command_exit_code: int | None = None
    tree_termination_failure: dict[str, str] | None = None
    root_fallback_kill_requested = False
    if requested:
        if os.name == "nt":
            try:
                completed = subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=wait_seconds,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                tree_command_exit_code = int(completed.returncode)
                if completed.returncode != 0:
                    tree_termination_failure = {
                        "errorType": "TaskkillExitFailure",
                        "message": (
                            "taskkill returned non-zero while terminating the "
                            "owned finalist campaign process tree"
                        ),
                    }
            except (OSError, subprocess.TimeoutExpired) as exc:
                tree_termination_failure = {
                    "errorType": type(exc).__name__,
                    "message": str(exc).strip() or type(exc).__name__,
                }
            if tree_termination_failure is not None and process.poll() is None:
                root_fallback_kill_requested = True
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            root_fallback_kill_requested = True
            process.kill()
        process.wait(timeout=wait_seconds)
    return {
        "requested": requested,
        "processTreeCommandExitCode": tree_command_exit_code,
        "processTreeTerminationFailure": tree_termination_failure,
        "rootFallbackKillRequested": root_fallback_kill_requested,
        "workerExitCode": process.returncode,
        "workerTerminated": process.poll() is not None,
    }


def _supervise_finalist_worker(
    process: subprocess.Popen[Any],
    *,
    progress_path: Path,
    global_deadline_monotonic: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    terminate: Callable[[subprocess.Popen[Any]], dict[str, Any]] = (
        _terminate_owned_process_tree
    ),
) -> dict[str, Any]:
    last_progress: dict[str, Any] = {}
    global_enforcement_deadline = max(
        monotonic(),
        global_deadline_monotonic - _FINALIST_GLOBAL_CLEANUP_RESERVE_SECONDS,
    )
    while True:
        progress = _read_supervision_progress(progress_path)
        if progress is not None:
            last_progress = progress
        boundaries = [("global_wall_bound", global_enforcement_deadline)]
        raw_seed_deadline = last_progress.get("seedDeadlineMonotonic")
        if isinstance(raw_seed_deadline, (int, float)):
            boundaries.append(("seed_wall_bound", float(raw_seed_deadline)))
        boundary, deadline = min(boundaries, key=lambda item: item[1])
        now = monotonic()
        if now >= deadline:
            termination = terminate(process)
            return {
                "timedOut": True,
                "boundary": boundary,
                "deadlineMonotonic": deadline,
                "observedMonotonic": now,
                "lastProgress": last_progress,
                "termination": termination,
            }
        exit_code = process.poll()
        if exit_code is not None:
            return {
                "timedOut": False,
                "workerExitCode": int(exit_code),
                "lastProgress": last_progress,
            }
        sleep(min(_FINALIST_SUPERVISION_POLL_SECONDS, max(0.0, deadline - now)))


def _safe_campaign_output(
    root: Path,
    output: Path,
    *,
    require_absent: bool = False,
) -> Path:
    resolved_root = root.resolve()
    resolved_output = output.resolve()
    forbidden_directories = tuple(
        (resolved_root / owner).resolve()
        for owner in _OUTPUT_FORBIDDEN_SOURCE_DIRECTORIES
    )
    forbidden_files = tuple(
        (resolved_root / owner).resolve()
        for owner in _FINGERPRINT_SOURCE_FILES
    )
    if resolved_output.exists() and resolved_output.is_dir():
        raise ValueError("the finalist campaign output must be a file")
    if require_absent and resolved_output.exists():
        raise ValueError(
            "the finalist campaign output is immutable and already exists"
        )
    if any(
        resolved_output == directory
        or directory in resolved_output.parents
        for directory in forbidden_directories
    ) or resolved_output in forbidden_files:
        raise ValueError(
            "the finalist campaign output must stay outside source and fingerprinted paths"
        )
    return resolved_output


def _validated_finalist_recipe_and_context(
    research: Any,
    payload: Mapping[str, Any],
) -> tuple[ResearchRecipeV3, Any]:
    """Validate the frozen V3 compatibility selector without persisting it."""

    validated = research.validate_recipe(payload)
    if not validated["valid"]:
        raise ValueError(json.dumps(validated["issues"], ensure_ascii=False))
    recipe, context = research._build_recipe_and_context(validated["recipe"])
    if not isinstance(recipe, ResearchRecipeV3) or context is None:
        raise RuntimeError("finalist compatibility selector is not exact V3")
    return recipe, context


def _campaign_invalid_reason(
    *,
    source_frozen: bool,
    strict_wall_bound_passed: bool,
    all_seed_bounds_passed: bool,
) -> str | None:
    if not source_frozen:
        return "source_fingerprint_changed"
    if not strict_wall_bound_passed or not all_seed_bounds_passed:
        return "campaign_time_bound_exceeded"
    return None


def _campaign_exit_code(status: str) -> int:
    return 0 if status in {"qualified", "negative"} else 1


def _runner_evaluation_failures(row: Mapping[str, Any]) -> tuple[str, ...]:
    qualification = dict(row.get("qualification") or {})
    failures = [
        str(item)
        for item in (
            row.get("failures")
            or qualification.get("failures")
            or ()
        )
        if str(item)
    ]
    return tuple(sorted(set(failures)))


def _runner_evaluation_evidence_hashes(
    row: Mapping[str, Any],
) -> tuple[str, ...]:
    declared = tuple(
        str(row[field])
        for field in ("stableHash", "traceHash", "auditHash")
        if _canonical_sha256_value(row.get(field))
    )
    if declared:
        return tuple(sorted(set(declared)))
    stable_core = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"wallSeconds", "cpuSeconds"}
    }
    return (canonical_sha256(stable_core),)


def _runner_candidate_evaluation(
    row: Mapping[str, Any],
) -> SequentialCandidateEvaluationV1:
    value = dict(row)
    failures = _runner_evaluation_failures(value)
    if value.get("status") == "bounded_timeout":
        outcome = SequentialCandidateOutcomeKindV1.INCOMPLETE
        failures = failures or ("bounded_timeout",)
    elif bool(value.get("fatalDefect")):
        outcome = SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE
        failures = failures or ("fatal_semantic_or_leak_defect",)
    elif bool(value.get("qualified")):
        outcome = SequentialCandidateOutcomeKindV1.QUALIFIED
        failures = ()
    else:
        outcome = SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION
        failures = failures or ("ordinary_qualification_rejection",)
    return SequentialCandidateEvaluationV1.build(
        outcome=outcome,
        candidate_hash=str(value.get("candidateHash") or ""),
        evaluation_evidence_hashes=_runner_evaluation_evidence_hashes(value),
        failure_codes=failures,
    )


class _RunnerSequentialGeneratorAdapter:
    def __init__(
        self,
        delegate: Any,
        *,
        on_phase: Callable[[str, Mapping[str, Any]], None] | None,
    ) -> None:
        self.delegate = delegate
        self.descriptor = getattr(delegate, "descriptor", None)
        self.on_phase = on_phase
        self.continuation_probe_wall_seconds = 0.0

    def generate_next(self, spec: Any, **kwargs: Any) -> Any:
        rejected = tuple(kwargs.get("rejected_candidates") or ())
        if self.on_phase is not None:
            self.on_phase("candidate_generation", {
                "candidateIndex": len(rejected) + 1,
                "rejectedCandidateCount": len(rejected),
            })
        return self.delegate.generate_next(spec, **kwargs)

    def probe_continuation(self, spec: Any, **kwargs: Any) -> Any:
        rejected = tuple(kwargs.get("rejected_candidates") or ())
        if self.on_phase is not None:
            self.on_phase(
                (
                    "candidate_continuation_probe"
                    if len(rejected) >= int(spec.maximum_solutions)
                    else "candidate_exhaustion_probe"
                ),
                {
                    "candidateIndex": len(rejected) + 1,
                    "rejectedCandidateCount": len(rejected),
                },
            )
        started = time.perf_counter()
        try:
            return self.delegate.probe_continuation(spec, **kwargs)
        finally:
            self.continuation_probe_wall_seconds += (
                time.perf_counter() - started
            )


def _runner_stop_reason(product_stop_reason: str) -> str:
    return {
        "qualified": "qualified_candidate",
        "fatal_quarantine": "fatal_semantic_or_leak_defect",
        "incomplete": "candidate_evaluation_deadline",
        "cancelled": "candidate_evaluation_deadline",
        "maximum_solutions_reached": "candidate_bound_exhausted",
        "candidates_exhausted": "candidates_exhausted",
    }.get(product_stop_reason, product_stop_reason)


def _run_product_sequential_candidate_search(
    deal_generator: Any,
    spec: Any,
    *,
    root_seed: int,
    dealer: Seat,
    rules: Any,
    maximum_candidates: int,
    deadline_monotonic: float,
    evaluate_candidate: Callable[[Any, int], dict[str, Any]],
    on_result: Callable[[dict[str, Any]], None] | None = None,
    on_phase: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Adapt runner progress and candidate rows to the product search SSOT."""

    if type(maximum_candidates) is not int or maximum_candidates < 1:
        raise RuntimeError("runner candidate bound is invalid")
    if type(getattr(spec, "maximum_solutions", None)) is not int or (
        spec.maximum_solutions != maximum_candidates
    ):
        raise RuntimeError(
            "runner candidate bound differs from the governed deal spec"
        )
    if time.monotonic() >= deadline_monotonic:
        return {
            "rows": [],
            "qualifiedRow": None,
            "terminalRepeat": None,
            "fatalFailure": None,
            "stopReason": "candidate_search_deadline",
            "generationTimedOut": True,
            "rejectedCandidates": (),
            "sequentialSearchReceipt": None,
            "continuationProbeWallSeconds": 0.0,
        }

    rows: list[dict[str, Any]] = []
    rejected: list[Any] = []
    primary_by_hash: dict[str, dict[str, Any]] = {}
    repeat: dict[str, Any] | None = None
    fatal_failure: str | None = None
    generator = _RunnerSequentialGeneratorAdapter(
        deal_generator,
        on_phase=on_phase,
    )

    def evaluate(candidate: Any, candidate_index: int) -> SequentialCandidateEvaluationV1:
        nonlocal fatal_failure, repeat
        candidate_hash = str(candidate.candidate_hash)
        primary = primary_by_hash.get(candidate_hash)
        if on_phase is not None:
            on_phase(
                (
                    "qualified_candidate_repeat"
                    if primary is not None
                    else "candidate_evaluation"
                ),
                {
                    "candidateIndex": candidate_index,
                    "candidateHash": candidate_hash,
                    "rejectedCandidateCount": len(rejected),
                },
            )
        try:
            row = copy.deepcopy(dict(evaluate_candidate(candidate, candidate_index)))
        except InterruptedError:
            row = {
                "seed": int(root_seed),
                "candidateIndex": int(candidate_index),
                "candidateHash": candidate_hash,
                "dealModelHash": str(candidate.model_hash),
                "dealSolverStatus": str(candidate.solver_status),
                "status": "bounded_timeout",
                "qualified": False,
                "fatalDefect": False,
                "failures": ["candidate_evaluation_cancelled"],
                "wallSeconds": 0.0,
                "cpuSeconds": 0.0,
            }
        if str(row.get("candidateHash") or "") != candidate_hash:
            raise RuntimeError("candidate evaluation changed the candidate identity")
        if (
            time.monotonic() >= deadline_monotonic
            and not bool(row.get("fatalDefect"))
        ):
            row.update({
                "status": "bounded_timeout",
                "qualified": False,
                "fatalDefect": False,
                "failures": list(dict.fromkeys([
                    *row.get("failures", ()),
                    "bounded_timeout_after_evaluation",
                ])),
            })
        if primary is None:
            primary_by_hash[candidate_hash] = row
            rows.append(row)
            if on_result is not None:
                on_result(row)
            evaluation = _runner_candidate_evaluation(row)
            if evaluation.outcome == SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION:
                rejected.append(candidate)
            elif evaluation.outcome == SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE:
                fatal_failure = "fatal_semantic_or_leak_defect"
            return evaluation

        repeat = {
            "performed": True,
            "stableHash": row.get("stableHash"),
            "traceHash": row.get("traceHash"),
            "auditHash": row.get("auditHash"),
            "qualified": bool(row.get("qualified")),
            "wallSeconds": row.get("wallSeconds"),
            "cpuSeconds": row.get("cpuSeconds"),
            "deterministic": (
                row.get("stableHash") == primary.get("stableHash")
                and row.get("traceHash") == primary.get("traceHash")
                and row.get("auditHash") == primary.get("auditHash")
                and bool(row.get("qualified"))
                and not bool(row.get("fatalDefect"))
            ),
        }
        if not repeat["deterministic"]:
            fatal_failure = "qualified_repeat_determinism_failure"
        return _runner_candidate_evaluation(row)

    try:
        product_result = run_sequential_candidate_search_v1(
            generator,
            spec,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            evaluate_candidate=evaluate,
            deadline_monotonic=deadline_monotonic,
        )
    except InterruptedError:
        return {
            "rows": rows,
            "qualifiedRow": None,
            "terminalRepeat": repeat,
            "fatalFailure": fatal_failure,
            "stopReason": "candidate_generation_deadline",
            "generationTimedOut": True,
            "rejectedCandidates": tuple(rejected),
            "sequentialSearchReceipt": None,
            "continuationProbeWallSeconds": (
                generator.continuation_probe_wall_seconds
            ),
        }

    receipt = product_result.receipt.verify().to_dict()
    product_stop = str(receipt["stopReason"])
    if fatal_failure == "qualified_repeat_determinism_failure":
        stop_reason = fatal_failure
    else:
        stop_reason = _runner_stop_reason(product_stop)
    qualified_row = (
        primary_by_hash.get(product_result.qualified_candidate.candidate_hash)
        if product_result.qualified_candidate is not None
        and fatal_failure is None
        else None
    )
    return {
        "rows": rows,
        "qualifiedRow": qualified_row,
        "terminalRepeat": repeat,
        "fatalFailure": fatal_failure,
        "stopReason": stop_reason,
        "generationTimedOut": False,
        "rejectedCandidates": tuple(rejected),
        "sequentialSearchReceipt": receipt,
        "continuationProbeWallSeconds": (
            generator.continuation_probe_wall_seconds
        ),
    }


def _source_fingerprint(root: Path) -> dict[str, Any]:
    """Hash the authored behavior and this opt-in runner, never local evidence."""

    resolved = root.resolve()
    files: list[Path] = []
    for owner in _FINGERPRINT_SOURCE_DIRECTORIES:
        base = resolved / owner
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file())
    files.extend(
        path
        for path in (resolved / owner for owner in _FINGERPRINT_SOURCE_FILES)
        if path.is_file()
    )
    rows = {
        path.relative_to(resolved).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(set(files))
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    core = {"schemaVersion": 1, "files": rows}
    return {
        "schemaVersion": 1,
        "fileCount": len(rows),
        "sha256": canonical_sha256(core),
    }


def _exact_search_backend_runtime_identity(backend: Any) -> dict[str, Any]:
    descriptor = getattr(backend, "descriptor", None)
    if descriptor is None:
        raise RuntimeError("selected exact-search backend identity is unavailable")
    backend_capabilities = tuple(
        str(item) for item in getattr(descriptor, "capabilities", ())
    )
    backend_id = str(getattr(descriptor, "id", "") or "")
    native = getattr(backend, "native", None)
    native_selected = bool(
        native is not None
        or backend_id.startswith("native_")
        or "native_bitboards" in backend_capabilities
    )
    identity: dict[str, Any] = {
        "schemaVersion": 1,
        "backendId": backend_id,
        "backendVersion": str(getattr(descriptor, "version", "") or ""),
        "backendCapabilities": list(backend_capabilities),
        "deterministic": bool(getattr(descriptor, "deterministic", False)),
        "nativeSelected": native_selected,
        "nativeLibrary": None,
    }
    if not native_selected:
        return identity
    lib = getattr(native, "lib", None) if native is not None else None
    raw_library_name = getattr(lib, "_name", None)
    try:
        library_name = os.fsdecode(raw_library_name)
        library = Path(library_name)
        if not library.is_file():
            raise OSError("native library is not a file")
        library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "selected native exact-search library identity cannot be hashed"
        ) from exc
    capabilities_mask = int(getattr(native, "capabilities", 0))
    native_capabilities = sorted({
        name[4:].lower()
        for name in dir(native)
        if name.startswith("CAP_")
        and isinstance(getattr(native, name, None), int)
        and capabilities_mask & int(getattr(native, name))
    })
    if bool(getattr(native, "supports_compact_rollout", False)):
        native_capabilities.append("compact_rollout")
    supports_v2 = bool(getattr(native, "supports_v2", False))
    supports_v3 = bool(getattr(native, "supports_v3", False))
    abi_versions = [1, *([2] if supports_v2 else ()), *([3] if supports_v3 else ())]
    identity["nativeLibrary"] = {
        "basename": library.name,
        "sha256": library_sha256,
        "abiVersion": max(abi_versions),
        "abiVersions": abi_versions,
        "capabilitiesMask": capabilities_mask,
        "capabilities": native_capabilities,
    }
    return identity


def _write_supervision_failure_evidence(
    args: argparse.Namespace,
    *,
    output: Path,
    source_start: Mapping[str, Any] | None,
    source_end: Mapping[str, Any] | None,
    progress: Mapping[str, Any],
    elapsed_seconds: float,
    failure: Mapping[str, Any],
    stop_reason: str,
    temporary_runtime_cleaned: bool,
    cleanup_failure: Mapping[str, Any] | None,
    supervision: Mapping[str, Any],
) -> dict[str, Any]:
    completed_candidates = [
        dict(item) for item in progress.get("completedCandidates") or ()
    ]
    completed_seeds = [
        dict(item) for item in progress.get("completedSeeds") or ()
    ]
    source_frozen = (
        dict(source_start) == dict(source_end)
        if source_start is not None and source_end is not None
        else None
    )
    maximum_wall_seconds = float(args.maximum_wall_seconds)
    maximum_seed_seconds = float(args.maximum_seed_seconds)
    stable = {
        "schemaVersion": FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION,
        "experimentId": FINALIST_CAMPAIGN_EXPERIMENT_ID,
        "qualificationProfile": {
            "id": FINALIST_CAMPAIGN_PROFILE_ID,
            "revision": FINALIST_CAMPAIGN_PROFILE_REVISION,
            "globallyEnabled": False,
        },
        "status": "invalid",
        "stopReason": stop_reason,
        "fatalFailure": stop_reason,
        "failure": dict(failure),
        "cleanupFailure": (
            dict(cleanup_failure) if cleanup_failure is not None else None
        ),
        "temporaryRuntimeCleaned": bool(temporary_runtime_cleaned),
        "bounds": {
            "seedStart": int(args.seed_start),
            "seedCount": int(args.attempts),
            "maximumCandidatesPerSeed": int(args.candidates_per_seed),
            "maximumWallSeconds": maximum_wall_seconds,
            "maximumSeedSeconds": maximum_seed_seconds,
        },
        "target": {"contract": "sun", "tactic": "only_winning_move"},
        "controlPlane": progress.get("controlPlane"),
        "sourceFingerprintStart": (
            dict(source_start) if source_start is not None else None
        ),
        "sourceFingerprintEnd": (
            dict(source_end) if source_end is not None else None
        ),
        "sourceFrozen": source_frozen,
        "seedResults": completed_seeds,
        "candidateResults": [],
        "partialCandidateReceipts": completed_candidates,
        "qualifiedCandidateStableHash": None,
        "terminalRepeat": None,
        "policyOutcome": None,
        "policyOutcomeAssigned": False,
        "counts": {
            "seedsEvaluated": len(completed_seeds),
            "candidatesEvaluated": len(completed_candidates),
            "tacticRetained": sum(
                bool(item.get("tacticRetained"))
                for item in completed_candidates
            ),
            "qualified": 0,
            "semanticDefects": sum(
                bool(item.get("semanticDefect"))
                for item in completed_candidates
            ),
            "observerLeakDefects": sum(
                bool(item.get("observerLeakDefect"))
                for item in completed_candidates
            ),
            "distinctCandidateHashes": len({
                str(item["candidateHash"])
                for item in completed_candidates
                if item.get("candidateHash")
            }),
            "distinctTraceHashes": 0,
        },
        "cost": {
            "wallSeconds": float(elapsed_seconds),
            "cpuSeconds": None,
            "strictWallBoundPassed": elapsed_seconds <= maximum_wall_seconds,
            "allSeedBoundsPassed": False,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ortools": ortools.__version__,
            "logicalCpuCount": os.cpu_count(),
            "exactSearchBackend": copy.deepcopy(
                dict(progress.get("controlPlane") or {}).get(
                    "searchBackendRuntime"
                )
            ),
        },
        "supervision": copy.deepcopy(dict(supervision)),
        "productSourceChangedByRunner": False,
        "gatesWeakened": False,
        "provisionalTruthRelabeled": False,
    }
    result = {**stable, "sha256": canonical_sha256(stable)}
    _publish_json_once(output, result)
    print(json.dumps({
        "status": result["status"],
        "stopReason": result["stopReason"],
        "counts": result["counts"],
        "cost": result["cost"],
        "sha256": result["sha256"],
        "output": str(output),
    }, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return result


def _canonical_sha256_value(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trace_analysis_budget(profile: TraceAnalysisProfileV1) -> dict[str, Any]:
    return {
        "requested_samples": profile.play_samples,
        "exact_allocation_limit": profile.exact_allocation_limit,
        "exact_search_max_remaining": profile.exact_search_max_remaining,
        "maximum_sampling_attempts": profile.maximum_sampling_attempts,
        "exact_search_max_nodes": profile.exact_search_max_nodes,
        "exact_search_max_seconds": profile.exact_search_max_seconds,
        "exact_search_max_depth": profile.exact_search_max_depth,
        "exact_search_max_cache_entries": profile.exact_search_max_cache_entries,
    }


def _current_finalist_receipt_authority(
    root: Path,
    backend_id: str,
) -> dict[str, Any]:
    """Rebuild receipt-critical authority from the published source bundle."""

    backend_descriptors = {
        item.id: item.to_dict()
        for item in (
            PythonReferenceSearchBackend.descriptor,
            NativeExactSearchBackendV3.descriptor,
        )
    }
    if backend_id not in backend_descriptors:
        raise RuntimeError("the finalist campaign search backend is not current")
    bundle = build_bundle(Path(root).resolve())
    challenge = dict(bundle.component("challenge"))
    confirmation = dict(challenge.get("confirmation") or {})
    stability = dict(challenge.get("stability") or {})
    primary_profile = TraceAnalysisProfileV1.build(
        bid_samples=int(confirmation.get("bid_samples", 0)),
        play_samples=int(confirmation.get("deep_samples", 0)),
        exact_allocation_limit=int(
            confirmation.get("exact_allocation_limit", 0)
        ),
        exact_search_max_remaining=int(
            confirmation.get("exact_search_remaining", -1)
        ),
        maximum_sampling_attempts=int(
            confirmation.get("maximum_sampling_attempts", 0)
        ),
        exact_search_max_nodes=int(
            confirmation.get("exact_search_max_nodes", 0)
        ),
        exact_search_max_seconds=float(
            confirmation.get("exact_search_max_seconds", 0.0)
        ),
        exact_search_max_depth=int(
            confirmation.get("exact_search_max_depth", 0)
        ),
        exact_search_max_cache_entries=int(
            confirmation.get("exact_search_max_cache_entries", -1)
        ),
        search_backend_id=backend_id,
        objective=str(confirmation.get("objective") or ""),
        use_conventions=False,
    )
    reanalysis_profile = build_finalist_reanalysis_trace_profile_v1(
        stability,
        confirmation,
        search_backend_id=backend_id,
    )
    validation_profile = FinalistValidationProfileV1.build(
        challenge.get("finalist_validation") or {}
    )
    return {
        "bundleVersion": str(bundle.bundle_profile["profile"]["version"]),
        "semanticBundleHash": bundle.sha256,
        "agreementDescriptor": FinalistAgentAgreementV1.descriptor.to_dict(),
        "finalistValidationProfile": validation_profile.to_dict(),
        "searchBackend": backend_descriptors[backend_id],
        "analysisAuthorities": {
            "primary": {
                "role": "deep_confirmation",
                "traceAnalysisProfileHash": primary_profile.sha256,
                "analysisBudget": _trace_analysis_budget(primary_profile),
            },
            "reanalysis": {
                "role": "finalist_reanalysis",
                "traceAnalysisProfileHash": reanalysis_profile.sha256,
                "analysisBudget": _trace_analysis_budget(reanalysis_profile),
            },
        },
    }


def _finalist_control_plane(
    *,
    context: Any,
    recipe: ResearchRecipeV3,
    simulator: NaturalLaffaSimulatorV1,
    backend_runtime_identity: Mapping[str, Any],
    bundle_version: str,
) -> dict[str, Any]:
    """Project one fully resolved finalist context into receipt authority."""

    agreement = simulator.decision_agreement
    if agreement is None:
        raise ValueError("finalist campaign lacks decision-agreement authority")
    configuration = ResearchConfigurationAuthorityV1.build(
        recipe.configuration
    ).to_dict()
    validation_profile = FinalistValidationProfileV1.build(
        context.bundle.component("challenge")["finalist_validation"]
    )
    return {
        "bundleRevisionId": context.bundle_revision_id,
        "bundleVersion": str(bundle_version),
        "semanticBundleHash": context.bundle.sha256,
        "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
        "contextHash": context.context_hash,
        "recipeHash": recipe.sha256,
        "configurationAuthorityHash": configuration["sha256"],
        "configurationAuthority": configuration,
        "snapshotArtifact": copy.deepcopy(
            dict(configuration["snapshotArtifact"])
        ),
        "pluginIdentities": [
            copy.deepcopy(dict(item)) for item in context.plugin_identities
        ],
        "finalistProfileHash": simulator.profile_hash,
        "finalistReleaseHash": simulator.agent_release,
        "finalistReleaseStatus": (
            simulator.agent_qualification.status
            if simulator.agent_qualification is not None
            else None
        ),
        "finalistAgreementProfileHash": agreement.profile.sha256,
        "finalistAgreementDescriptor": agreement.descriptor.to_dict(),
        "finalistValidationProfile": validation_profile.to_dict(),
        "analysisAuthorities": {
            "primary": {
                "role": "deep_confirmation",
                "traceAnalysisProfileHash": (
                    agreement.primary_trace_analysis_profile_hash
                ),
                "analysisIdentityHash": agreement.primary_analysis_identity_hash,
                "analysisBudget": dict(agreement.primary_analysis_budget),
            },
            "reanalysis": {
                "role": "finalist_reanalysis",
                "traceAnalysisProfileHash": (
                    agreement.reanalysis_trace_analysis_profile_hash
                ),
                "analysisIdentityHash": (
                    agreement.reanalysis_analysis_identity_hash
                ),
                "analysisBudget": dict(agreement.independent_analysis_budget),
            },
        },
        "analysisPolicyAuthority": copy.deepcopy(
            dict(simulator.analysis_policy_authority)
        ),
        "auctionSelectionAuthority": (
            "finalist_multi_cohort_agent_primary_policy"
        ),
        "playSelectionAuthority": (
            "finalist_multi_cohort_agent_primary_policy"
        ),
        "auctionCpSatRole": "exact_required_action_rebind_only",
        "playCpSatRole": "exact_required_action_rebind_only",
        "searchBackend": context.exact_search_backend.descriptor.to_dict(),
        "searchBackendRuntime": copy.deepcopy(dict(backend_runtime_identity)),
    }


def _validated_finalist_recipe_selector(
    recipe: ResearchRecipeV3,
    *,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
) -> None:
    resources = dict(recipe.resources)
    expected_qualification = {
        "minimumTruth": "robust_best_known",
        "minimumConfidence": 0.75,
        "minimumSurroundingQuality": 70,
        "requireObserverSafety": True,
        "requireAllDecisionAudit": True,
        "maximumRegret": 0,
    }
    if (
        dict(recipe.configuration) != dict(configuration)
        or resources.get("generationLane") != "ortools"
        or resources.get("analysisMode") != FINALIST_CAMPAIGN_ANALYSIS_MODE
        or resources.get("auctionConstructionMode")
        != "actor_safe_agent_verified"
        or resources.get("playConstructionMode")
        != "actor_safe_agent_verified"
        or int(resources.get("seedStart", -1)) != int(args.seed_start)
        or int(resources.get("seedCount", -1)) != int(args.attempts)
        or int(resources.get("candidatesPerSeed", -1))
        != int(args.candidates_per_seed)
        or int(recipe.output.get("resultCount", -1))
        != int(args.attempts) * int(args.candidates_per_seed)
        or list(recipe.situation.get("contracts") or ()) != ["sun"]
        or dict(recipe.qualification) != expected_qualification
        or list(recipe.tactics) != [{
            "kind": "only_winning_move",
            "mode": "qualification_gate",
            "minimumScore": 1,
            "weight": 1,
            "perspectives": [],
        }]
        or recipe.status != "published"
    ):
        raise RuntimeError("the finalist campaign compatibility selector is stale")


def _validated_runtime_control_plane(
    *,
    root: Path,
    runtime: Path,
    args: argparse.Namespace,
    control_plane: Mapping[str, Any],
    parent_source_start: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the worker CAS snapshot through the canonical context factory."""

    resolved_root = Path(root).resolve()
    runtime = Path(runtime).resolve()
    if not runtime.is_dir():
        raise RuntimeError("the finalist campaign validation runtime is unavailable")
    source_before = _source_fingerprint(resolved_root)
    if source_before != dict(parent_source_start):
        raise RuntimeError(
            "the finalist campaign source changed before authority validation"
        )
    control = copy.deepcopy(dict(control_plane))
    try:
        configuration = ResearchConfigurationAuthorityV1.build(
            control.get("configurationAuthority") or {}
        ).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "the finalist campaign configuration authority is invalid"
        ) from exc

    validation_container = build_container(root=resolved_root, writable_root=runtime)
    expected: dict[str, Any] | None = None
    try:
        factory = validation_container.research.execution_context_factory
        if factory is None:
            raise RuntimeError(
                "the finalist campaign validation context authority is unavailable"
            )
        resolved_context = factory.resolve_authority(
            configuration,
            analysis_mode=FINALIST_CAMPAIGN_ANALYSIS_MODE,
        )
        recipe, recipe_context = _validated_finalist_recipe_and_context(
            validation_container.research,
            _finalist_campaign_recipe(
                validation_container.research,
                bundle_revision_id=str(configuration["bundleRevisionId"]),
                seed_start=int(args.seed_start),
                seed_count=int(args.attempts),
                candidates_per_seed=int(args.candidates_per_seed),
                analysis_mode=FINALIST_CAMPAIGN_ANALYSIS_MODE,
            ),
        )
        recipe_hash = str(control.get("recipeHash") or "")
        if (
            recipe.sha256 != recipe_hash
            or recipe_context.context_hash != resolved_context.context_hash
            or recipe_context.effective_runtime_config_hash
            != resolved_context.effective_runtime_config_hash
            or recipe_context.bundle.sha256 != resolved_context.bundle.sha256
        ):
            raise RuntimeError(
                "the finalist campaign compatibility selector context is stale"
            )
        _validated_finalist_recipe_selector(
            recipe,
            args=args,
            configuration=configuration,
        )
        simulator = _context_finalist_simulator(recipe_context)
        backend_runtime = _exact_search_backend_runtime_identity(
            recipe_context.exact_search_backend
        )
        expected = _finalist_control_plane(
            context=recipe_context,
            recipe=recipe,
            simulator=simulator,
            backend_runtime_identity=backend_runtime,
            bundle_version=str(configuration["bundleVersion"]),
        )
        if expected != control:
            raise RuntimeError(
                "the finalist campaign worker control-plane authority is stale"
            )
    finally:
        validation_container.shutdown()
    source_after = _source_fingerprint(resolved_root)
    if source_after != source_before:
        raise RuntimeError(
            "the finalist campaign source changed during authority validation"
        )
    assert expected is not None
    return expected


def _validated_control_plane_authority(
    control_plane: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    runtime_identity: Mapping[str, Any],
    expected_control_plane: Mapping[str, Any],
    authoritative_control_plane: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    str,
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    control = dict(control_plane)
    expected_control = dict(expected_control_plane)
    authoritative_control = dict(authoritative_control_plane)
    backend_id = str(runtime_identity.get("backendId") or "")
    current = _current_finalist_receipt_authority(
        Path(args.root),
        backend_id,
    )
    agreement_descriptor = dict(
        control.get("finalistAgreementDescriptor") or {}
    )
    agreement_profile_hash = str(
        control.get("finalistAgreementProfileHash") or ""
    )
    configuration_authority = dict(
        control.get("configurationAuthority") or {}
    )
    try:
        normalized_configuration = ResearchConfigurationAuthorityV1.build(
            configuration_authority
        ).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "the finalist campaign worker control-plane authority is stale"
        ) from exc
    authorities = dict(control.get("analysisAuthorities") or {})
    primary_authority = dict(authorities.get("primary") or {})
    reanalysis_authority = dict(authorities.get("reanalysis") or {})
    required_control_keys = {
        "bundleRevisionId",
        "bundleVersion",
        "semanticBundleHash",
        "effectiveRuntimeConfigHash",
        "contextHash",
        "recipeHash",
        "configurationAuthorityHash",
        "configurationAuthority",
        "snapshotArtifact",
        "pluginIdentities",
        "finalistProfileHash",
        "finalistReleaseHash",
        "finalistReleaseStatus",
        "finalistAgreementProfileHash",
        "finalistAgreementDescriptor",
        "finalistValidationProfile",
        "analysisAuthorities",
        "analysisPolicyAuthority",
        "auctionSelectionAuthority",
        "playSelectionAuthority",
        "auctionCpSatRole",
        "playCpSatRole",
        "searchBackend",
        "searchBackendRuntime",
    }
    hashes = (
        control.get("semanticBundleHash"),
        control.get("effectiveRuntimeConfigHash"),
        control.get("contextHash"),
        control.get("recipeHash"),
        control.get("configurationAuthorityHash"),
        control.get("finalistProfileHash"),
        control.get("finalistReleaseHash"),
        agreement_profile_hash,
        primary_authority.get("traceAnalysisProfileHash"),
        primary_authority.get("analysisIdentityHash"),
        reanalysis_authority.get("traceAnalysisProfileHash"),
        reanalysis_authority.get("analysisIdentityHash"),
    )
    if (
        not expected_control
        or not authoritative_control
        or control != expected_control
        or control != authoritative_control
        or set(control) != required_control_keys
        or control.get("bundleVersion") != str(args.bundle_version)
        or agreement_descriptor != current["agreementDescriptor"]
        or control.get("searchBackend") != current["searchBackend"]
        or control.get("searchBackendRuntime") != dict(runtime_identity)
        or configuration_authority != normalized_configuration
        or configuration_authority.get("sha256")
        != control.get("configurationAuthorityHash")
        or configuration_authority.get("bundleRevisionId")
        != control.get("bundleRevisionId")
        or configuration_authority.get("bundleVersion")
        != control.get("bundleVersion")
        or configuration_authority.get("semanticBundleHash")
        != control.get("semanticBundleHash")
        or configuration_authority.get("effectiveRuntimeConfigHash")
        != control.get("effectiveRuntimeConfigHash")
        or control.get("snapshotArtifact")
        != configuration_authority.get("snapshotArtifact")
        or not isinstance(control.get("pluginIdentities"), list)
        or not control.get("pluginIdentities")
        or control.get("finalistReleaseStatus") != "configured_candidate"
        or control.get("auctionSelectionAuthority")
        != "finalist_multi_cohort_agent_primary_policy"
        or control.get("playSelectionAuthority")
        != "finalist_multi_cohort_agent_primary_policy"
        or control.get("auctionCpSatRole")
        != "exact_required_action_rebind_only"
        or control.get("playCpSatRole")
        != "exact_required_action_rebind_only"
        or any(not _canonical_sha256_value(value) for value in hashes)
        or primary_authority.get("role") != "deep_confirmation"
        or reanalysis_authority.get("role") != "finalist_reanalysis"
        or int(primary_authority.get("analysisBudget", {}).get(
            "requested_samples", -1
        )) != 192
        or int(reanalysis_authority.get("analysisBudget", {}).get(
            "requested_samples", -1
        )) != 256
        or primary_authority.get("analysisIdentityHash")
        == reanalysis_authority.get("analysisIdentityHash")
    ):
        raise RuntimeError(
            "the finalist campaign worker control-plane authority is stale"
        )
    return (
        agreement_descriptor,
        agreement_profile_hash,
        {
            "primary": primary_authority,
            "reanalysis": reanalysis_authority,
        },
        {
            "researchContextHash": str(control["contextHash"]),
            "rulesHash": str(
                configuration_authority["identities"]["rules"]["sha256"]
            ),
            "analysisBackend": backend_id,
        },
    )


def _validated_candidate_authority(
    candidate: Mapping[str, Any],
    *,
    agreement_descriptor: Mapping[str, Any],
    agreement_profile_hash: str,
    authorities: Mapping[str, Mapping[str, Any]],
) -> None:
    value = dict(candidate)
    selection = dict(value.get("selectionAndVerification") or {})
    policy = dict(value.get("policyEvidence") or {})
    decision_agreement = dict(value.get("decisionAgreement") or {})
    policy_core = {
        key: item for key, item in policy.items() if key != "sha256"
    }
    expected_policy_authorities = {
        role: {
            "role": authority["role"],
            "profileHash": authority["traceAnalysisProfileHash"],
            "identityHash": authority["analysisIdentityHash"],
            "budget": dict(authority["analysisBudget"]),
        }
        for role, authority in authorities.items()
    }
    if (
        set(authorities) != {"primary", "reanalysis"}
        or selection.get("qualificationProfileId")
        != FINALIST_CAMPAIGN_PROFILE_ID
        or selection.get("qualificationProfileRevision")
        != FINALIST_CAMPAIGN_PROFILE_REVISION
        or policy.get("agreementDescriptor") != dict(agreement_descriptor)
        or policy.get("agreementProfileHash") != agreement_profile_hash
        or policy.get("liveTraceAgreementDescriptor")
        != dict(agreement_descriptor)
        or policy.get("liveTraceAgreementProfileHash")
        != agreement_profile_hash
        or policy.get("analysisAuthorities") != expected_policy_authorities
        or policy.get("sha256") != canonical_sha256(policy_core)
        or decision_agreement.get("descriptor") != dict(agreement_descriptor)
        or decision_agreement.get("profileHash") != agreement_profile_hash
    ):
        raise RuntimeError("the finalist campaign candidate authority is stale")

    rows: list[tuple[str, dict[str, Any]]] = [
        (kind, dict(row))
        for kind, values in (
            ("bid", policy.get("bids") or ()),
            ("play", policy.get("plays") or ()),
        )
        for row in values
    ]
    decision_ids = [str(row.get("decisionId") or "") for _, row in rows]
    if not decision_ids or any(not item for item in decision_ids) or len(
        set(decision_ids)
    ) != len(decision_ids):
        raise RuntimeError("the finalist campaign candidate authority is stale")
    competent_ids = sorted({
        decision_id
        for decision_id, (_, row) in zip(decision_ids, rows, strict=True)
        if row.get("agreementStatus") == "primary_robust_competent"
    })
    unresolved_ids = sorted({
        decision_id
        for decision_id, (_, row) in zip(decision_ids, rows, strict=True)
        if row.get("agreementStatus") == "unresolved"
    })
    if (
        policy.get("competentDecisionIds") != competent_ids
        or policy.get("unresolvedDecisionIds") != unresolved_ids
        or policy.get("featuredTruthBlockedDecisionIds")
        != sorted({*competent_ids, *unresolved_ids})
        or bool(policy.get("stabilityResolved")) != (not unresolved_ids)
        or int(decision_agreement.get("decisionCount", -1)) != len(rows)
        or int(decision_agreement.get("unresolvedCount", -1))
        != len(unresolved_ids)
    ):
        raise RuntimeError("the finalist campaign candidate authority is stale")

    for decision_kind, row in rows:
        status = str(row.get("agreementStatus") or "")
        cohorts = [dict(item) for item in row.get("cohortSupport") or ()]
        cohort_count = int(row.get("cohortCount", -1))
        if (
            row.get("decisionKind") != decision_kind
            or not _canonical_sha256_value(row.get("publicPrefixHash"))
            or row.get("actingSeat") not in {
                "east",
                "south",
                "west",
                "north",
            }
            or row.get("analysisAuthorityRole") != "deep_confirmation"
            or row.get("traceAnalysisProfileHash")
            != authorities["primary"]["traceAnalysisProfileHash"]
            or row.get("analysisIdentityHash")
            != authorities["primary"]["analysisIdentityHash"]
            or row.get("analysisBudget")
            != authorities["primary"]["analysisBudget"]
            or cohort_count != len(cohorts)
            or cohort_count not in ({1, 2, 3} if status == "unresolved" else {2, 3})
            or status not in {
                "strict_confirmed",
                "primary_robust_competent",
                "unresolved",
            }
            or bool(row.get("qualificationBlocked")) != (status == "unresolved")
            or bool(row.get("featuredTruthEligible"))
            != (status == "strict_confirmed")
            or type(row.get("agreementCount")) is not int
            or not 0 <= int(row["agreementCount"]) <= cohort_count
        ):
            raise RuntimeError("the finalist campaign candidate authority is stale")
        row_failures = [str(item) for item in row.get("failures") or ()]
        if (
            status in {"strict_confirmed", "primary_robust_competent"}
            and row_failures
        ) or (
            status == "unresolved"
            and f"{decision_kind}_policy_blocked" not in row_failures
        ):
            raise RuntimeError("the finalist campaign candidate authority is stale")
        observed_seeds: list[int] = []
        recommendation_hashes: list[str] = []
        for index, cohort in enumerate(cohorts):
            authority = authorities["primary" if index == 0 else "reanalysis"]
            expected_seed = derive_actor_observer_cohort_seed_v1(
                public_prefix_hash=str(row["publicPrefixHash"]),
                decision_kind=decision_kind,
                acting_seat=str(row.get("actingSeat") or ""),
                cohort_authority_hash=str(authority["analysisIdentityHash"]),
                cohort_index=index,
                governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
            )
            if (
                cohort.get("cohortIndex") != index
                or cohort.get("analysisAuthorityRole") != authority["role"]
                or cohort.get("traceAnalysisProfileHash")
                != authority["traceAnalysisProfileHash"]
                or cohort.get("analysisIdentityHash")
                != authority["analysisIdentityHash"]
                or cohort.get("analysisBudget") != authority["analysisBudget"]
                or int(authority["analysisBudget"].get("requested_samples", -1))
                != (192 if index == 0 else 256)
                or cohort.get("seed") != expected_seed
                or not _canonical_sha256_value(
                    cohort.get("recommendedActionHash")
                )
            ):
                raise RuntimeError(
                    "the finalist campaign candidate authority is stale"
                )
            observed_seeds.append(expected_seed)
            recommendation_hashes.append(
                str(cohort["recommendedActionHash"])
            )
        if len(set(observed_seeds)) != len(observed_seeds):
            raise RuntimeError("the finalist campaign candidate authority is stale")
        if (
            status == "strict_confirmed"
            and (
                int(row.get("agreementCount", 0)) < 2
                or not _canonical_sha256_value(row.get("majorityActionHash"))
                or recommendation_hashes.count(
                    str(row.get("majorityActionHash"))
                )
                < int(row["agreementCount"])
            )
        ):
            raise RuntimeError("the finalist campaign candidate authority is stale")
        if status != "strict_confirmed" and (
            row.get("majorityActionHash") is not None
            and (
                not _canonical_sha256_value(row.get("majorityActionHash"))
                or str(row["majorityActionHash"]) not in recommendation_hashes
            )
        ):
            raise RuntimeError("the finalist campaign candidate authority is stale")


def _validated_candidate_decision_bindings(
    candidate: Mapping[str, Any],
    *,
    authorities: Mapping[str, Mapping[str, Any]],
    binding_authority: Mapping[str, str],
) -> None:
    value = copy.deepcopy(dict(candidate))
    policy = dict(value.get("policyEvidence") or {})
    parity = dict(value.get("decisionScoreParity") or {})
    raw_ledger = value.get("decisionLedger")
    expected_rules_hash = str(binding_authority.get("rulesHash") or "")
    expected_context_hash = str(
        binding_authority.get("researchContextHash") or ""
    )
    expected_backend = str(binding_authority.get("analysisBackend") or "")
    if not isinstance(raw_ledger, Mapping):
        raise RuntimeError("the finalist campaign decision ledger is missing")
    raw_ledger = copy.deepcopy(dict(raw_ledger))
    raw_decisions = raw_ledger.get("decisions")
    if (
        set(raw_ledger) != {"schemaVersion", "gameId", "decisions", "sha256"}
        or type(raw_ledger.get("schemaVersion")) is not int
        or raw_ledger["schemaVersion"] != 1
        or not isinstance(raw_ledger.get("gameId"), str)
        or not raw_ledger["gameId"]
        or not isinstance(raw_decisions, list)
        or not raw_decisions
        or any(not isinstance(item, Mapping) for item in raw_decisions)
        or not _canonical_sha256_value(raw_ledger.get("sha256"))
    ):
        raise RuntimeError("the finalist campaign decision ledger is invalid")
    try:
        ledger = DecisionLedgerV1.build(
            raw_ledger["gameId"],
            [dict(item) for item in raw_decisions],
        )
        normalized_ledger = ledger.to_dict()
        ledger_bytes_match = canonical_json_bytes_v1(
            raw_ledger
        ) == canonical_json_bytes_v1(normalized_ledger)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "the finalist campaign decision ledger is invalid"
        ) from exc
    if not ledger_bytes_match or ledger.game_id != value.get("gameHash"):
        raise RuntimeError("the finalist campaign decision ledger is stale")

    records = tuple(ledger.records)
    for record in records:
        legal = tuple(dict(item) for item in record.legal_actions)
        if (
            record.decision_kind not in {"bid", "play"}
            or not _canonical_sha256_value(record.public_prefix_hash)
            or not legal
            or record.forced != (len(legal) == 1)
            or canonical_json_bytes_v1(record.chosen_action)
            not in {canonical_json_bytes_v1(item) for item in legal}
        ):
            raise RuntimeError(
                "the finalist campaign decision ledger choice is invalid"
            )

    parity_keys = {
        "schemaVersion",
        "authority",
        "traceHash",
        "deepAnalysisHash",
        "deepAnalysisProfileHash",
        "deepAnalysisIdentityHash",
        "analysisBackend",
        "decisionCount",
        "nonForcedDecisionCount",
        "forcedDecisionCount",
        "matchedDecisionCount",
        "matched",
        "mismatches",
        "decisions",
        "crossLaneComparison",
        "structuralObjectiveCompared",
        "officialBalootScoreCompared",
        "researchContextHash",
        "sha256",
    }
    parity_rows_raw = parity.get("decisions")
    mismatches = parity.get("mismatches")
    count_fields = (
        "decisionCount",
        "nonForcedDecisionCount",
        "forcedDecisionCount",
        "matchedDecisionCount",
    )
    if (
        set(parity) != parity_keys
        or type(parity.get("schemaVersion")) is not int
        or parity["schemaVersion"] != 1
        or parity.get("authority")
        != "finalist_primary_decision_score_parity_v1"
        or parity.get("traceHash") != value.get("traceHash")
        or parity.get("deepAnalysisHash") != value.get("analysisHash")
        or parity.get("deepAnalysisProfileHash")
        != authorities["primary"]["traceAnalysisProfileHash"]
        or parity.get("deepAnalysisIdentityHash")
        != authorities["primary"]["analysisIdentityHash"]
        or parity.get("analysisBackend") != expected_backend
        or parity.get("researchContextHash") != expected_context_hash
        or any(type(parity.get(field)) is not int for field in count_fields)
        or type(parity.get("matched")) is not bool
        or not isinstance(mismatches, list)
        or not isinstance(parity_rows_raw, list)
        or parity.get("crossLaneComparison") is not False
        or parity.get("structuralObjectiveCompared") is not False
        or parity.get("officialBalootScoreCompared") is not False
    ):
        raise RuntimeError(
            "the finalist campaign decision-score parity authority is invalid"
        )
    mismatch_keys = {"decisionId", "ordinal", "decisionKind", "reason"}
    if any(
        not isinstance(item, Mapping)
        or set(item) != mismatch_keys
        or item.get("decisionId") is not None
        and (not isinstance(item.get("decisionId"), str) or not item["decisionId"])
        or item.get("ordinal") is not None
        and type(item.get("ordinal")) is not int
        or item.get("decisionKind") is not None
        and item.get("decisionKind") not in {"bid", "play"}
        or not isinstance(item.get("reason"), str)
        or not item["reason"]
        for item in mismatches
    ):
        raise RuntimeError(
            "the finalist campaign decision-score parity mismatches are invalid"
        )
    parity_rows = [dict(item) for item in parity_rows_raw]
    parity_row_keys = {
        "decisionId",
        "ordinal",
        "decisionKind",
        "forced",
        "generationEvidenceHash",
        "auditEvidenceHash",
        "generationRowsHash",
        "auditRowsHash",
        "matched",
    }
    if (
        len(parity_rows) != len(records)
        or parity["decisionCount"] != len(records)
        or parity["nonForcedDecisionCount"]
        != sum(not record.forced for record in records)
        or parity["forcedDecisionCount"]
        != sum(record.forced for record in records)
        or parity["matchedDecisionCount"]
        != sum(row.get("matched") is True for row in parity_rows)
        or parity["matched"] != (not mismatches)
        or (parity["matched"] and parity["matchedDecisionCount"] != len(records))
    ):
        raise RuntimeError(
            "the finalist campaign decision-score parity counts are invalid"
        )
    parity_by_id: dict[str, dict[str, Any]] = {}
    for record, row in zip(records, parity_rows, strict=True):
        evidence_hashes = (
            row.get("generationEvidenceHash"),
            row.get("auditEvidenceHash"),
        )
        rows_hashes = (
            row.get("generationRowsHash"),
            row.get("auditRowsHash"),
        )
        if (
            set(row) != parity_row_keys
            or row.get("decisionId") != record.decision_id
            or type(row.get("ordinal")) is not int
            or row["ordinal"] != record.ordinal
            or row.get("decisionKind") != record.decision_kind
            or type(row.get("forced")) is not bool
            or row["forced"] != record.forced
            or type(row.get("matched")) is not bool
            or any(
                item is not None and not _canonical_sha256_value(item)
                for item in (*evidence_hashes, *rows_hashes)
            )
            or (
                row["matched"]
                and (
                    not all(_canonical_sha256_value(item) for item in evidence_hashes)
                    or evidence_hashes[0] != evidence_hashes[1]
                    or (
                        record.forced
                        and rows_hashes != (None, None)
                    )
                    or (
                        not record.forced
                        and (
                            not all(
                                _canonical_sha256_value(item)
                                for item in rows_hashes
                            )
                            or rows_hashes[0] != rows_hashes[1]
                        )
                    )
                )
            )
        ):
            raise RuntimeError(
                "the finalist campaign decision-score parity row is invalid"
            )
        parity_by_id[record.decision_id] = row

    if (
        value.get("rulesHash") != expected_rules_hash
        or policy.get("rulesHash") != expected_rules_hash
        or policy.get("decisionLedgerHash") != ledger.sha256
        or not _canonical_sha256_value(expected_rules_hash)
        or not _canonical_sha256_value(expected_context_hash)
        or not expected_backend
    ):
        raise RuntimeError(
            "the finalist campaign decision binding authority is stale"
        )

    expected_non_forced = {
        "bids": [
            record
            for record in records
            if not record.forced and record.decision_kind == "bid"
        ],
        "plays": [
            record
            for record in records
            if not record.forced and record.decision_kind == "play"
        ],
    }
    if (
        policy.get("nonForcedBidCount") != len(expected_non_forced["bids"])
        or policy.get("nonForcedPlayCount") != len(expected_non_forced["plays"])
    ):
        raise RuntimeError(
            "the finalist campaign non-forced decision counts are stale"
        )
    for policy_key, expected_records in expected_non_forced.items():
        raw_rows = policy.get(policy_key)
        if not isinstance(raw_rows, list) or len(raw_rows) != len(expected_records):
            raise RuntimeError(
                "the finalist campaign non-forced decision ledger is incomplete"
            )
        for record, row_raw in zip(expected_records, raw_rows, strict=True):
            row = dict(row_raw)
            parity_row = parity_by_id[record.decision_id]
            expected_legal_hash = legal_action_set_hash_v1(
                record.legal_actions
            )
            expected_action_hash = canonical_sha256(
                canonical_plain_profile(record.chosen_action)
            )
            cohorts = row.get("cohortSupport")
            if (
                row.get("decisionId") != record.decision_id
                or type(row.get("ordinal")) is not int
                or row["ordinal"] != record.ordinal
                or row.get("decisionKind") != record.decision_kind
                or row.get("forced") is not False
                or row.get("actingSeat")
                != record.acting_seat.name.lower()
                or row.get("publicPrefixHash") != record.public_prefix_hash
                or row.get("legalActionSetHash") != expected_legal_hash
                or row.get("rulesHash") != expected_rules_hash
                or canonical_json_bytes_v1(row.get("selectedAction"))
                != canonical_json_bytes_v1(record.chosen_action)
                or row.get("selectedActionMatchesTrace") is not True
                or row.get("primaryDecisionScoreEvidenceHash")
                != parity_row.get("generationEvidenceHash")
                or row.get("primaryDecisionScoreEvidenceHash")
                != parity_row.get("auditEvidenceHash")
                or not _canonical_sha256_value(
                    row.get("primaryDecisionScoreEvidenceHash")
                )
                or not isinstance(cohorts, list)
                or not cohorts
                or dict(cohorts[0]).get("recommendedActionHash")
                != expected_action_hash
            ):
                raise RuntimeError(
                    "the finalist campaign non-forced decision binding is stale"
                )

    expected_forced = [record for record in records if record.forced]
    forced_rows_raw = policy.get("forcedDecisions")
    if (
        type(policy.get("forcedDecisionCount")) is not int
        or policy["forcedDecisionCount"] != len(expected_forced)
        or not isinstance(forced_rows_raw, list)
        or len(forced_rows_raw) != len(expected_forced)
    ):
        raise RuntimeError(
            "the finalist campaign forced decision ledger is incomplete"
        )
    for record, row_raw in zip(expected_forced, forced_rows_raw, strict=True):
        row = dict(row_raw)
        proof = row.get("forcedLegalityProof")
        parity_row = parity_by_id[record.decision_id]
        if not isinstance(proof, Mapping):
            raise RuntimeError(
                "the finalist campaign forced decision proof is missing"
            )
        try:
            validated_proof = _validate_forced_legality_proof_v1(
                proof,
                record=record,
                rules_hash=expected_rules_hash,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "the finalist campaign forced decision proof is invalid"
            ) from exc
        if (
            row.get("decisionId") != record.decision_id
            or type(row.get("ordinal")) is not int
            or row["ordinal"] != record.ordinal
            or row.get("decisionKind") != record.decision_kind
            or row.get("forced") is not True
            or row.get("actingSeat") != record.acting_seat.name.lower()
            or row.get("publicPrefixHash") != record.public_prefix_hash
            or row.get("legalActionSetHash")
            != legal_action_set_hash_v1(record.legal_actions)
            or row.get("rulesHash") != expected_rules_hash
            or canonical_json_bytes_v1(row.get("selectedAction"))
            != canonical_json_bytes_v1(record.chosen_action)
            or row.get("selectedActionMatchesTrace") is not True
            or row.get("forcedLegalityProofHash")
            != validated_proof["sha256"]
            or row.get("forcedLegalityProofHash")
            != parity_row.get("generationEvidenceHash")
            or row.get("forcedLegalityProofHash")
            != parity_row.get("auditEvidenceHash")
        ):
            raise RuntimeError(
                "the finalist campaign forced decision binding is stale"
            )


def _validated_candidate_receipt(
    candidate: Mapping[str, Any],
    *,
    agreement_descriptor: Mapping[str, Any],
    agreement_profile_hash: str,
    authorities: Mapping[str, Mapping[str, Any]],
    binding_authority: Mapping[str, str],
) -> None:
    value = copy.deepcopy(dict(candidate))
    base_fields = {
        "seed",
        "candidateIndex",
        "candidateHash",
        "dealModelHash",
        "dealSolverStatus",
    }
    audited_stable_fields = {
        *base_fields,
        "traceHash",
        "rulesHash",
        "decisionLedger",
        "gameHash",
        "deckCommitment",
        "auctionPlanHash",
        "projectPlanHash",
        "playPlanHash",
        "playModelHash",
        "analysisHash",
        "decisionScoreParityHash",
        "auditHash",
        "decisionSelectionHash",
        "qualified",
        "qualification",
        "score",
        "gates",
        "tacticRetained",
        "maximumBidRegret",
        "maximumPlayRegret",
        "auditProblems",
        "recipeTargetMismatches",
        "semanticMismatches",
        "decisionScoreParity",
        "policyEvidence",
        "selectionAndVerification",
        "fatalPolicyDefects",
        "decisionAgreement",
    }
    audited_receipt_fields = {
        *audited_stable_fields,
        "stableHash",
        "wallSeconds",
        "cpuSeconds",
        "fatalDefect",
    }
    exception_fields = {
        *base_fields,
        "status",
        "qualified",
        "fatalDefect",
        "failures",
        "errorType",
        "wallSeconds",
        "cpuSeconds",
    }
    bounded_fields = exception_fields - {"errorType"}
    for field in ("candidateHash", "dealModelHash"):
        if not _canonical_sha256_value(value.get(field)):
            raise RuntimeError("the finalist campaign candidate receipt is invalid")
    if (
        type(value.get("seed")) is not int
        or type(value.get("candidateIndex")) is not int
        or int(value["candidateIndex"]) < 1
        or not str(value.get("dealSolverStatus") or "")
        or not isinstance(value.get("wallSeconds"), (int, float))
        or not isinstance(value.get("cpuSeconds"), (int, float))
        or float(value["wallSeconds"]) < 0.0
        or float(value["cpuSeconds"]) < 0.0
        or type(value.get("qualified")) is not bool
        or type(value.get("fatalDefect")) is not bool
    ):
        raise RuntimeError("the finalist campaign candidate receipt is invalid")

    status = value.get("status")
    bounded_audited = False
    if status is not None:
        failures = [str(item) for item in value.get("failures") or ()]
        if status == "rejected_before_audit":
            if (
                set(value) != exception_fields
                or value["qualified"]
                or value["fatalDefect"]
                or not failures
                or any(_candidate_exception_is_fatal(item) for item in failures)
            ):
                raise RuntimeError(
                    "the finalist campaign rejected candidate receipt is invalid"
                )
        elif status == "semantic_defect":
            if (
                set(value) != exception_fields
                or value["qualified"]
                or not value["fatalDefect"]
                or not failures
                or not all(_candidate_exception_is_fatal(item) for item in failures)
            ):
                raise RuntimeError(
                    "the finalist campaign fatal candidate receipt is invalid"
                )
        elif status == "bounded_timeout":
            bounded_audited = set(value) == {
                *audited_receipt_fields,
                "status",
                "failures",
            }
            if (
                (set(value) != bounded_fields and not bounded_audited)
                or value.get("errorType") is not None
                or value["qualified"]
                or value["fatalDefect"]
                or not failures
                or not any(item.startswith("bounded_timeout") for item in failures)
            ):
                raise RuntimeError(
                    "the finalist campaign bounded candidate receipt is invalid"
                )
        else:
            raise RuntimeError("the finalist campaign candidate status is invalid")
        if not bounded_audited:
            return
        if value.get("policyEvidence"):
            _validated_candidate_authority(
                value,
                agreement_descriptor=agreement_descriptor,
                agreement_profile_hash=agreement_profile_hash,
                authorities=authorities,
            )
    elif set(value) != audited_receipt_fields:
        raise RuntimeError("the finalist campaign audited candidate schema is invalid")

    stable_hash = value.get("stableHash")
    stable_core = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key in audited_stable_fields
    }
    if bounded_audited:
        stable_core["qualified"] = bool(
            dict(value.get("qualification") or {}).get("qualified")
        )
    if (
        not _canonical_sha256_value(stable_hash)
        or canonical_sha256(stable_core) != stable_hash
        or any(
            not _canonical_sha256_value(value.get(field))
            for field in (
                "traceHash",
                "gameHash",
                "deckCommitment",
                "auctionPlanHash",
                "projectPlanHash",
                "playPlanHash",
                "playModelHash",
                "analysisHash",
                "decisionScoreParityHash",
                "auditHash",
                "decisionSelectionHash",
            )
        )
    ):
        raise RuntimeError("the finalist campaign candidate stable hash is invalid")
    _validated_candidate_authority(
        value,
        agreement_descriptor=agreement_descriptor,
        agreement_profile_hash=agreement_profile_hash,
        authorities=authorities,
    )
    qualification = dict(value.get("qualification") or {})
    gates = dict(value.get("gates") or {})
    policy = dict(value.get("policyEvidence") or {})
    score_parity = dict(value.get("decisionScoreParity") or {})
    score = dict(value.get("score") or {})
    audit_problems = [
        dict(item) for item in value.get("auditProblems") or ()
    ]
    expected_gate_keys = {
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
    }
    parity_core = {
        key: copy.deepcopy(item)
        for key, item in score_parity.items()
        if key != "sha256"
    }
    if (
        set(gates) != expected_gate_keys
        or any(type(item) is not bool for item in gates.values())
        or score_parity.get("sha256") != canonical_sha256(parity_core)
        or value.get("decisionScoreParityHash") != score_parity.get("sha256")
        or gates["semanticReplay"] is not True
        or gates["zeroSemanticDefects"]
        != (not bool(value.get("semanticMismatches")))
        or gates["fullCpSatReplayParity"]
        != (not bool(value.get("semanticMismatches")))
        or gates["decisionScoreParity"] != bool(score_parity.get("matched"))
        or gates["tacticRetained"] != bool(value.get("tacticRetained"))
        or gates["zeroObserverLeaks"] != bool(policy.get("zeroObserverLeaks"))
        or gates["sharedActionAcrossWorlds"]
        != bool(policy.get("oneSharedActionAcrossIndistinguishableWorlds"))
        or gates["finalistPrimaryAuctionAndPlay"]
        != bool(policy.get("primaryPolicyValid"))
        or gates["finalistAgreementResolved"]
        != bool(policy.get("stabilityResolved"))
        or gates["observerSafety"]
        != _observer_safety_passed(
            {
                "informationSetComplete": gates["informationSetComplete"],
                "sequentialResponseComplete": gates[
                    "sequentialResponseComplete"
                ],
            },
            policy,
        )
        or gates["qualityAtLeast70"]
        != (float(qualification.get("minimumSurroundingQuality", 0.0)) >= 70.0)
        or gates["zeroRecommendationMismatches"]
        != (not any(bool(item.get("recommendationMismatch")) for item in audit_problems))
        or gates["zeroMaterialCounterexamples"]
        != (not any(bool(item.get("materialCounterexample")) for item in audit_problems))
        or gates["regretGate"]
        != (
            max(
                float(value.get("maximumBidRegret", 0.0)),
                float(value.get("maximumPlayRegret", 0.0)),
            )
            <= 0.0
        )
        or gates["scoreProfile"] != bool(score.get("passed"))
        or gates["ordinaryQualification"]
        != bool(qualification.get("qualified"))
    ):
        raise RuntimeError("the finalist campaign candidate gates are invalid")
    _validated_candidate_decision_bindings(
        value,
        authorities=authorities,
        binding_authority=binding_authority,
    )
    fatal_expected = bool(
        value.get("semanticMismatches")
        or score_parity.get("matched") is not True
        or policy.get("zeroObserverLeaks") is not True
        or value.get("fatalPolicyDefects")
    )
    qualified_expected = all(gates.values())
    receipt_qualified = bool(value["qualified"])
    audited_qualified = bool(qualification.get("qualified"))
    if (
        type(qualification.get("qualified")) is not bool
        or audited_qualified != qualified_expected
        or (not bounded_audited and receipt_qualified != audited_qualified)
        or (bounded_audited and receipt_qualified)
        or bool(value["fatalDefect"]) != fatal_expected
        or (bounded_audited and fatal_expected)
        or (value["qualified"] and value["fatalDefect"])
        or bool(value.get("tacticRetained"))
        != bool(gates.get("tacticRetained"))
    ):
        raise RuntimeError("the finalist campaign candidate outcome is invalid")
    selection = dict(value.get("selectionAndVerification") or {})
    auction_verification = dict(selection.get("auctionCpSatVerification") or {})
    play_verification = dict(selection.get("playCpSatVerification") or {})
    if (
        set(selection)
        != {
            "qualificationProfileId",
            "qualificationProfileRevision",
            "auctionSelectionAuthority",
            "playSelectionAuthority",
            "auctionCpSatVerification",
            "playCpSatVerification",
            "postHocActionRepair",
        }
        or set(auction_verification) != {"role", "planHash", "solverStatus"}
        or set(play_verification)
        != {"role", "projectPlanHash", "planHash", "solverStatus"}
        or selection.get("auctionSelectionAuthority")
        != "finalist_multi_cohort_agent_primary_policy"
        or selection.get("playSelectionAuthority")
        != "finalist_multi_cohort_agent_primary_policy"
        or selection.get("postHocActionRepair") is not False
        or auction_verification.get("role")
        != "exact_required_action_rebind_only"
        or auction_verification.get("solverStatus") != "optimal"
        or auction_verification.get("planHash") != value.get("auctionPlanHash")
        or play_verification.get("role") != "exact_required_action_rebind_only"
        or play_verification.get("solverStatus") != "optimal"
        or play_verification.get("projectPlanHash")
        != value.get("projectPlanHash")
        or play_verification.get("planHash") != value.get("playPlanHash")
    ):
        raise RuntimeError(
            "the finalist campaign candidate verification authority is invalid"
        )


def _validated_seed_receipt(
    seed_receipt: Mapping[str, Any],
    *,
    candidate_rows: list[dict[str, Any]],
    maximum_candidates: int,
) -> None:
    """Bind the runner seed summary to the product-owned search receipt."""

    value = copy.deepcopy(dict(seed_receipt))
    required_seed_fields = {
        "schemaVersion",
        "kind",
        "seed",
        "generatedCandidateCount",
        "evaluatedCandidateCount",
        "tacticRetentionCount",
        "distinctCandidateHashes",
        "noPostHocRepair",
        "sequentialSearchReceipt",
        "continuationProbeWallSeconds",
        "stopReason",
        "wallSeconds",
        "cpuSeconds",
    }
    clock_fields = (
        "continuationProbeWallSeconds",
        "wallSeconds",
        "cpuSeconds",
    )
    if (
        set(value) != required_seed_fields
        or value.get("schemaVersion") != 1
        or value.get("kind") != "combined_finalist_seed_search_result_v1"
        or type(value.get("seed")) is not int
        or type(value.get("generatedCandidateCount")) is not int
        or type(value.get("evaluatedCandidateCount")) is not int
        or type(value.get("tacticRetentionCount")) is not int
        or type(value.get("distinctCandidateHashes")) is not int
        or value.get("noPostHocRepair") is not True
        or type(maximum_candidates) is not int
        or not 1 <= maximum_candidates <= 100
        or any(type(value.get(field)) not in {int, float} for field in clock_fields)
        or any(
            not math.isfinite(float(value[field])) or float(value[field]) < 0.0
            for field in clock_fields
        )
        or len(candidate_rows) > maximum_candidates
        or value["generatedCandidateCount"] != len(candidate_rows)
        or value["evaluatedCandidateCount"] != len(candidate_rows)
        or value["tacticRetentionCount"]
        != sum(bool(item.get("tacticRetained")) for item in candidate_rows)
        or value["distinctCandidateHashes"]
        != len({str(item.get("candidateHash") or "") for item in candidate_rows})
    ):
        raise RuntimeError("the finalist campaign seed receipt is invalid")

    receipt_document = value.get("sequentialSearchReceipt")
    if receipt_document is None:
        if value.get("stopReason") not in {"seed_wall_bound", "global_wall_bound"}:
            raise RuntimeError(
                "the finalist campaign seed receipt lost its product search receipt"
            )
        return
    if not isinstance(receipt_document, Mapping):
        raise RuntimeError("the finalist campaign product search receipt is invalid")
    try:
        canonical_receipt = SequentialCandidateSearchReceiptV1.from_dict(
            receipt_document
        ).verify().to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "the finalist campaign product search receipt is invalid"
        ) from exc
    if canonical_receipt != dict(receipt_document):
        raise RuntimeError(
            "the finalist campaign product search receipt is not canonical"
        )

    evaluations = [
        copy.deepcopy(dict(item))
        for item in canonical_receipt.get("evaluationReceipts") or ()
    ]
    primary = [item for item in evaluations if item.get("attempt") == "primary"]
    repeats = [
        item for item in evaluations if item.get("attempt") == "qualified_repeat"
    ]
    if (
        canonical_receipt.get("rootSeed") != value["seed"]
        or canonical_receipt.get("maximumSolutions") != maximum_candidates
        or canonical_receipt.get("emittedCandidateCount") != len(candidate_rows)
        or canonical_receipt.get("evaluatedCandidateCount") != len(candidate_rows)
        or len(primary) != len(candidate_rows)
        or len(repeats) > 1
    ):
        raise RuntimeError(
            "the finalist campaign product search receipt is seed-unbound"
        )

    expected_rejections: list[str] = []
    qualified_primary = False
    for ordinal, (evaluation_receipt, candidate_row) in enumerate(
        zip(primary, candidate_rows, strict=True)
    ):
        candidate_core = dict(
            evaluation_receipt.get("candidateIntegrityCore") or {}
        )
        provenance = dict(evaluation_receipt.get("candidateProvenance") or {})
        try:
            expected_evaluation = _runner_candidate_evaluation(
                candidate_row
            ).to_dict()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "the finalist campaign candidate search outcome is invalid"
            ) from exc
        actual_evaluation = dict(evaluation_receipt.get("evaluation") or {})
        candidate_hash = str(candidate_row.get("candidateHash") or "")
        deck_commitment = candidate_row.get("deckCommitment")
        if (
            evaluation_receipt.get("candidateOrdinal") != ordinal
            or candidate_core.get("ordinal") != ordinal
            or provenance.get("solutionOrdinal") != ordinal
            or provenance.get("candidateHash") != candidate_hash
            or actual_evaluation != expected_evaluation
            or provenance.get("modelHash") != candidate_row.get("dealModelHash")
            or provenance.get("solverStatus")
            != candidate_row.get("dealSolverStatus")
            or (
                deck_commitment is not None
                and provenance.get("deckCommitment") != deck_commitment
            )
        ):
            raise RuntimeError(
                "the finalist campaign candidate search outcome is unbound"
            )
        outcome = str(actual_evaluation.get("outcome") or "")
        if outcome == SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION.value:
            expected_rejections.append(candidate_hash)
        if outcome == SequentialCandidateOutcomeKindV1.QUALIFIED.value:
            qualified_primary = True

    feedback_hashes = [
        str(item.get("candidateHash") or "")
        for item in canonical_receipt.get("rejectionFeedbackRows") or ()
    ]
    if feedback_hashes != expected_rejections:
        raise RuntimeError(
            "the finalist campaign product search rejection history is unbound"
        )
    if repeats:
        if not qualified_primary or repeats[0].get("candidateOrdinal") != (
            len(primary) - 1
        ):
            raise RuntimeError(
                "the finalist campaign qualified repeat receipt is unbound"
            )
    elif qualified_primary:
        raise RuntimeError(
            "the finalist campaign qualified repeat receipt is missing"
        )

    expected_stop = _runner_stop_reason(str(canonical_receipt.get("stopReason") or ""))
    if (
        canonical_receipt.get("stopReason") == "fatal_quarantine"
        and qualified_primary
    ):
        expected_stop = "qualified_repeat_determinism_failure"
    seed_stop = str(value.get("stopReason") or "")
    if seed_stop not in {"seed_wall_bound", "global_wall_bound"} and (
        seed_stop != expected_stop
    ):
        raise RuntimeError(
            "the finalist campaign product search stop reason is unbound"
        )


def _expected_worker_counts(
    rows: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    *,
    qualified: bool,
) -> dict[str, int]:
    return {
        "seedsEvaluated": len(seeds),
        "candidatesEvaluated": len(rows),
        "tacticRetained": sum(bool(item.get("tacticRetained")) for item in rows),
        "qualified": int(qualified),
        "semanticDefects": sum(
            bool(item.get("semanticMismatches"))
            or item.get("status") == "semantic_defect"
            for item in rows
        ),
        "observerLeakDefects": sum(
            not bool(
                dict(item.get("policyEvidence") or {}).get(
                    "zeroObserverLeaks", True
                )
            )
            for item in rows
        ),
        "distinctCandidateHashes": len({
            str(item["candidateHash"]) for item in rows
        }),
        "distinctTraceHashes": len({
            str(item["traceHash"]) for item in rows if item.get("traceHash")
        }),
    }


def _validated_finalist_worker_evidence(
    output: Path,
    *,
    args: argparse.Namespace,
    launch_id: str,
    expected_control_plane: Mapping[str, Any] | None,
    authoritative_control_plane: Mapping[str, Any] | None,
    parent_source_start: Mapping[str, Any],
    parent_source_end: Mapping[str, Any],
    worker_exit_code: int,
    parent_observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        raw = json.loads(output.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            "the finalist campaign worker did not publish readable evidence"
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeError("the finalist campaign worker evidence is not an object")
    payload = dict(raw)
    declared_hash = str(payload.pop("sha256", ""))
    if len(declared_hash) != 64 or declared_hash != canonical_sha256(payload):
        raise RuntimeError("the finalist campaign worker evidence hash is invalid")
    supervision = dict(payload.get("supervision") or {})
    if (
        supervision.get("kind") != FINALIST_CAMPAIGN_WORKER_KIND
        or not secrets.compare_digest(
            str(supervision.get("launchId") or ""), launch_id
        )
    ):
        raise RuntimeError(
            "the finalist campaign worker evidence is stale or launch-unbound"
        )
    parent_runtime_finalized = supervision.get("parentRuntimeFinalized")
    qualification_profile = dict(payload.get("qualificationProfile") or {})
    bounds = dict(payload.get("bounds") or {})
    runtime_identity = dict(payload.get("runtime") or {})
    exact_backend_identity = runtime_identity.get("exactSearchBackend")
    control_plane = dict(payload.get("controlPlane") or {})
    authority_required = bool(
        payload.get("status") in {"qualified", "negative"}
        or control_plane
        or payload.get("candidateResults")
    )
    try:
        worker_maximum_wall_seconds = float(bounds["maximumWallSeconds"])
        worker_maximum_seed_seconds = float(bounds["maximumSeedSeconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "the finalist campaign worker evidence is incomplete"
        ) from exc
    required_keys = {
        "schemaVersion",
        "experimentId",
        "qualificationProfile",
        "status",
        "stopReason",
        "fatalFailure",
        "failure",
        "cleanupFailure",
        "temporaryRuntimeCleaned",
        "bounds",
        "target",
        "controlPlane",
        "sourceFingerprintStart",
        "sourceFingerprintEnd",
        "sourceFrozen",
        "seedResults",
        "candidateResults",
        "qualifiedCandidateStableHash",
        "terminalRepeat",
        "counts",
        "cost",
        "runtime",
        "supervision",
        "productSourceChangedByRunner",
        "gatesWeakened",
        "provisionalTruthRelabeled",
    }
    if (
        payload.get("schemaVersion") != FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION
        or payload.get("experimentId")
        != FINALIST_CAMPAIGN_EXPERIMENT_ID
        or payload.get("status") not in {"qualified", "negative", "invalid"}
        or qualification_profile.get("id") != FINALIST_CAMPAIGN_PROFILE_ID
        or qualification_profile.get("revision")
        != FINALIST_CAMPAIGN_PROFILE_REVISION
        or qualification_profile.get("globallyEnabled") is not False
        or set(payload) != required_keys
        or bounds.get("seedStart") != int(args.seed_start)
        or bounds.get("seedCount") != int(args.attempts)
        or bounds.get("maximumCandidatesPerSeed")
        != int(args.candidates_per_seed)
        or worker_maximum_wall_seconds != float(args.maximum_wall_seconds)
        or worker_maximum_seed_seconds != float(args.maximum_seed_seconds)
        or type(payload.get("temporaryRuntimeCleaned")) is not bool
        or supervision.get("schemaVersion") != 2
        or supervision.get("runtimeOwner") != "supervisor_process"
        or type(parent_runtime_finalized) is not bool
        or bool(payload.get("temporaryRuntimeCleaned"))
        != bool(parent_runtime_finalized)
        or not isinstance(payload.get("seedResults"), list)
        or not isinstance(payload.get("candidateResults"), list)
        or not isinstance(payload.get("counts"), dict)
        or not isinstance(payload.get("cost"), dict)
        or "exactSearchBackend" not in runtime_identity
        or runtime_identity
        != {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ortools": ortools.__version__,
            "logicalCpuCount": os.cpu_count(),
            "exactSearchBackend": exact_backend_identity,
        }
        or (
            authority_required
            and not isinstance(exact_backend_identity, dict)
        )
        or (
            exact_backend_identity is not None
            and control_plane.get("searchBackendRuntime")
            != exact_backend_identity
        )
        or payload.get("target")
        != {"contract": "sun", "tactic": "only_winning_move"}
        or payload.get("productSourceChangedByRunner") is not False
        or payload.get("gatesWeakened") is not False
        or payload.get("provisionalTruthRelabeled") is not False
        or int(worker_exit_code)
        != _campaign_exit_code(str(payload.get("status")))
    ):
        raise RuntimeError("the finalist campaign worker evidence is incomplete")

    base_supervision = {
        "schemaVersion": 2,
        "kind": FINALIST_CAMPAIGN_WORKER_KIND,
        "launchId": launch_id,
        "runtimeOwner": "supervisor_process",
        "parentRuntimeFinalized": bool(parent_runtime_finalized),
    }
    expected_supervision = dict(base_supervision)
    if parent_runtime_finalized:
        if not parent_observation:
            raise RuntimeError("the finalist campaign parent observation is missing")
        expected_supervision.update(copy.deepcopy(dict(parent_observation)))
    elif parent_observation is not None:
        raise RuntimeError("the finalist campaign parent observation is premature")
    if supervision != expected_supervision:
        raise RuntimeError("the finalist campaign supervision evidence is stale")

    source_start = payload.get("sourceFingerprintStart")
    source_end = payload.get("sourceFingerprintEnd")
    source_frozen = source_start == source_end
    if (
        source_start != dict(parent_source_start)
        or source_end != dict(parent_source_end)
        or type(payload.get("sourceFrozen")) is not bool
        or bool(payload.get("sourceFrozen")) != source_frozen
        or (
            payload.get("status") in {"qualified", "negative"}
            and not source_frozen
        )
    ):
        raise RuntimeError("the finalist campaign source fingerprint is stale")

    rows = [copy.deepcopy(dict(item)) for item in payload["candidateResults"]]
    seed_rows = [copy.deepcopy(dict(item)) for item in payload["seedResults"]]
    if authority_required:
        if (
            not control_plane
            or not isinstance(exact_backend_identity, dict)
            or not authoritative_control_plane
        ):
            raise RuntimeError("the finalist campaign worker evidence is incomplete")
        (
            agreement_descriptor,
            agreement_profile_hash,
            authorities,
            binding_authority,
        ) = _validated_control_plane_authority(
            control_plane,
            args=args,
            runtime_identity=exact_backend_identity,
            expected_control_plane=dict(expected_control_plane or {}),
            authoritative_control_plane=dict(authoritative_control_plane),
        )
        for candidate in rows:
            _validated_candidate_receipt(
                candidate,
                agreement_descriptor=agreement_descriptor,
                agreement_profile_hash=agreement_profile_hash,
                authorities=authorities,
                binding_authority=binding_authority,
            )
    elif control_plane or rows or authoritative_control_plane:
        raise RuntimeError("the finalist campaign worker authority is unexpected")

    seed_start = int(args.seed_start)
    candidate_seeds = {int(item["seed"]) for item in rows}
    observed_seeds = [int(item.get("seed", -1)) for item in seed_rows]
    if (
        observed_seeds
        != list(range(seed_start, seed_start + len(observed_seeds)))
        or len(set(observed_seeds)) != len(observed_seeds)
        or candidate_seeds - set(observed_seeds)
    ):
        raise RuntimeError("the finalist campaign seed sequence is invalid")
    for seed_row in seed_rows:
        seed = int(seed_row["seed"])
        candidates = [item for item in rows if int(item["seed"]) == seed]
        if [int(item["candidateIndex"]) for item in candidates] != list(
            range(1, len(candidates) + 1)
        ):
            raise RuntimeError("the finalist campaign candidate sequence is invalid")
        _validated_seed_receipt(
            seed_row,
            candidate_rows=candidates,
            maximum_candidates=int(args.candidates_per_seed),
        )

    status = str(payload["status"])
    qualified_rows = [item for item in rows if bool(item.get("qualified"))]
    terminal_repeat = payload.get("terminalRepeat")
    if status == "qualified":
        if (
            payload.get("stopReason") != "qualified_candidate"
            or len(qualified_rows) != 1
            or payload.get("fatalFailure") is not None
            or payload.get("failure") is not None
            or payload.get("cleanupFailure") is not None
            or not isinstance(terminal_repeat, Mapping)
            or terminal_repeat.get("performed") is not True
            or terminal_repeat.get("deterministic") is not True
            or terminal_repeat.get("qualified") is not True
            or terminal_repeat.get("stableHash")
            != qualified_rows[0].get("stableHash")
            or terminal_repeat.get("traceHash") != qualified_rows[0].get("traceHash")
            or terminal_repeat.get("auditHash") != qualified_rows[0].get("auditHash")
            or payload.get("qualifiedCandidateStableHash")
            != qualified_rows[0].get("stableHash")
        ):
            raise RuntimeError("the finalist qualified receipt is invalid")
    elif status == "negative":
        if (
            payload.get("stopReason") != "seed_bound_exhausted"
            or qualified_rows
            or len(seed_rows) != int(args.attempts)
            or payload.get("fatalFailure") is not None
            or payload.get("failure") is not None
            or payload.get("cleanupFailure") is not None
            or payload.get("qualifiedCandidateStableHash") is not None
            or terminal_repeat is not None
        ):
            raise RuntimeError("the finalist negative receipt is invalid")
    elif (
        qualified_rows
        or payload.get("stopReason")
        in {"qualified_candidate", "seed_bound_exhausted"}
        or not any(
            (
                payload.get("fatalFailure") is not None,
                payload.get("failure") is not None,
                payload.get("cleanupFailure") is not None,
                payload.get("sourceFrozen") is False,
            )
        )
        or payload.get("qualifiedCandidateStableHash") is not None
    ):
        raise RuntimeError("the finalist invalid receipt is inconsistent")

    expected_counts = _expected_worker_counts(
        rows,
        seed_rows,
        qualified=status == "qualified",
    )
    cost = dict(payload["cost"])
    try:
        wall_seconds = float(cost["wallSeconds"])
        cpu_seconds = float(cost["cpuSeconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("the finalist campaign cost receipt is invalid") from exc
    expected_strict_wall = wall_seconds <= worker_maximum_wall_seconds
    expected_all_seed_bounds = all(
        float(item["wallSeconds"]) <= worker_maximum_seed_seconds
        for item in seed_rows
    )
    if (
        dict(payload["counts"]) != expected_counts
        or set(cost)
        != {
            "wallSeconds",
            "cpuSeconds",
            "strictWallBoundPassed",
            "allSeedBoundsPassed",
        }
        or wall_seconds < 0.0
        or cpu_seconds < 0.0
        or cost.get("strictWallBoundPassed") is not expected_strict_wall
        or cost.get("allSeedBoundsPassed") is not expected_all_seed_bounds
        or (
            status in {"qualified", "negative"}
            and (not expected_strict_wall or not expected_all_seed_bounds)
        )
    ):
        raise RuntimeError("the finalist campaign derived receipt is invalid")
    return dict(raw)


def _finalize_parent_owned_worker_evidence(
    output: Path,
    evidence: Mapping[str, Any],
    *,
    launch_id: str,
    parent_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the worker receipt only after its parent-owned runtime is gone."""

    stable = copy.deepcopy(dict(evidence))
    stable.pop("sha256", None)
    supervision = dict(stable.get("supervision") or {})
    if (
        supervision.get("schemaVersion") != 2
        or supervision.get("kind") != FINALIST_CAMPAIGN_WORKER_KIND
        or not secrets.compare_digest(
            str(supervision.get("launchId") or ""), launch_id
        )
        or supervision.get("runtimeOwner") != "supervisor_process"
        or supervision.get("parentRuntimeFinalized") is not False
        or stable.get("temporaryRuntimeCleaned") is not False
    ):
        raise RuntimeError(
            "the finalist campaign worker did not preserve its parent-owned runtime"
        )
    supervision.update({
        "runtimeOwner": "supervisor_process",
        "parentRuntimeFinalized": True,
        **copy.deepcopy(dict(parent_observation)),
    })
    stable["supervision"] = supervision
    stable["temporaryRuntimeCleaned"] = True
    result = {**stable, "sha256": canonical_sha256(stable)}
    _publish_json_once(output, result)
    return result


def _run_finalist_campaign_supervised(args: argparse.Namespace) -> int:
    """Run the complete finalist lane behind one owned hard process boundary."""

    if bool(getattr(args, "_finalist_worker", False)):
        _validated_supervised_runtime(args)
        return _run_finalist_campaign(args)
    try:
        maximum_wall_seconds = float(args.maximum_wall_seconds)
    except (TypeError, ValueError):
        return _run_finalist_campaign(args)
    if not 30 <= maximum_wall_seconds <= 600:
        return _run_finalist_campaign(args)

    root = Path(args.root).resolve()
    output = _safe_campaign_output(
        root,
        Path(args.output),
        require_absent=True,
    )
    wall_started = time.perf_counter()
    global_deadline = time.monotonic() + maximum_wall_seconds
    source_start = _source_fingerprint(root)
    source_end: dict[str, Any] | None = None
    process: subprocess.Popen[Any] | None = None
    temporary_runtime = tempfile.TemporaryDirectory(
        prefix=_FINALIST_RUNTIME_PREFIX
    )
    runtime = Path(temporary_runtime.name).resolve()
    progress_path = runtime / "supervision-progress.json"
    worker_output = runtime / "worker-evidence.json"
    token = secrets.token_hex(32)
    launch_id = secrets.token_hex(16)
    cleanup_failure: dict[str, Any] | None = None
    try:
        _atomic_write_json(runtime / _FINALIST_SUPERVISION_MARKER, {
            "schemaVersion": 2,
            "tokenDigest": _supervision_token_digest(token, launch_id),
            "launchId": launch_id,
        })
        command = _finalist_worker_command(
            args,
            runtime=runtime,
            progress=progress_path,
            launch_id=launch_id,
        )
        popen_options: dict[str, Any] = {
            "cwd": str(root),
            "env": {**os.environ, _FINALIST_SUPERVISION_ENV: token},
        }
        if os.name == "nt":
            popen_options["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(command, **popen_options)
        outcome = _supervise_finalist_worker(
            process,
            progress_path=progress_path,
            global_deadline_monotonic=global_deadline,
        )
        if not bool(outcome.get("timedOut")):
            exit_code = int(outcome["workerExitCode"])
            worker_evidence: dict[str, Any] | None = None
            validation_failure: Exception | None = None
            authoritative_control_plane: dict[str, Any] | None = None
            expected_control_plane = dict(
                dict(outcome.get("lastProgress") or {}).get("controlPlane")
                or {}
            )
            try:
                if expected_control_plane:
                    authoritative_control_plane = _validated_runtime_control_plane(
                        root=root,
                        runtime=runtime,
                        args=args,
                        control_plane=expected_control_plane,
                        parent_source_start=source_start,
                    )
                source_end = _source_fingerprint(root)
                worker_evidence = _validated_finalist_worker_evidence(
                    worker_output,
                    args=args,
                    launch_id=launch_id,
                    expected_control_plane=expected_control_plane,
                    authoritative_control_plane=authoritative_control_plane,
                    parent_source_start=source_start,
                    parent_source_end=source_end,
                    worker_exit_code=exit_code,
                    parent_observation=None,
                )
                if (
                    worker_evidence.get("temporaryRuntimeCleaned") is not False
                    or not runtime.is_dir()
                ):
                    raise RuntimeError(
                        "the finalist campaign worker did not preserve its "
                        "parent-owned runtime"
                    )
            except Exception as exc:
                validation_failure = exc
            try:
                temporary_runtime.cleanup()
            except Exception as exc:
                cleanup_failure = {
                    "stage": "supervisor_runtime_cleanup",
                    "errorType": type(exc).__name__,
                    "message": str(exc).strip() or type(exc).__name__,
                }
            if validation_failure is None and cleanup_failure is None:
                try:
                    assert worker_evidence is not None
                    source_end = _source_fingerprint(root)
                    if source_end != source_start:
                        raise RuntimeError(
                            "the finalist campaign source changed before publication"
                        )
                    parent_observation = {
                        "hardBoundaryEnforced": True,
                        "timedOut": False,
                        "workerExitCode": exit_code,
                        "workerTerminated": process.poll() is not None,
                        "sourceFingerprintStart": copy.deepcopy(source_start),
                        "sourceFingerprintEnd": copy.deepcopy(source_end),
                        "sourceFrozen": source_start == source_end,
                    }
                    _finalize_parent_owned_worker_evidence(
                        output,
                        worker_evidence,
                        launch_id=launch_id,
                        parent_observation=parent_observation,
                    )
                    _validated_finalist_worker_evidence(
                        output,
                        args=args,
                        launch_id=launch_id,
                        expected_control_plane=expected_control_plane,
                        authoritative_control_plane=authoritative_control_plane,
                        parent_source_start=source_start,
                        parent_source_end=source_end,
                        worker_exit_code=exit_code,
                        parent_observation=parent_observation,
                    )
                except Exception as exc:
                    validation_failure = exc
                else:
                    return exit_code
            source_end = _source_fingerprint(root)
            worker_failure = (
                dict(worker_evidence.get("failure") or {})
                if worker_evidence is not None
                else {}
            )
            if cleanup_failure is not None:
                failure = worker_failure or dict(cleanup_failure)
                failure_stop_reason = "campaign_cleanup_failure"
            else:
                failure = {
                    "stage": "campaign_worker_evidence_validation",
                    "errorType": (
                        type(validation_failure).__name__
                        if validation_failure is not None
                        else "CampaignWorkerExit"
                    ),
                    "message": (
                        str(validation_failure).strip()
                        if validation_failure is not None
                        else (
                        "the supervised finalist campaign worker exited without "
                        "complete evidence"
                        )
                    ),
                }
                failure_stop_reason = "campaign_worker_failure"
            _write_supervision_failure_evidence(
                args,
                output=output,
                source_start=source_start,
                source_end=source_end,
                progress=dict(outcome.get("lastProgress") or {}),
                elapsed_seconds=time.perf_counter() - wall_started,
                failure=failure,
                stop_reason=failure_stop_reason,
                temporary_runtime_cleaned=not runtime.exists(),
                cleanup_failure=cleanup_failure,
                supervision={
                    "schemaVersion": 1,
                    "kind": "owned_finalist_campaign_process_v1",
                    "launchId": launch_id,
                    "hardBoundaryEnforced": True,
                    "timedOut": False,
                    "workerExitCode": exit_code,
                    "workerTerminated": process.poll() is not None,
                    "lastProgress": dict(outcome.get("lastProgress") or {}),
                    "originalFailurePrimary": bool(worker_failure),
                },
            )
            return 1

        try:
            temporary_runtime.cleanup()
        except Exception as exc:
            cleanup_failure = {
                "stage": "supervisor_runtime_cleanup",
                "errorType": type(exc).__name__,
                "message": str(exc).strip() or type(exc).__name__,
            }
        source_end = _source_fingerprint(root)
        boundary = str(outcome["boundary"])
        failure = {
            "stage": str(
                dict(outcome.get("lastProgress") or {}).get("phase")
                or "campaign_execution"
            ),
            "errorType": "CampaignHardTimeout",
            "message": (
                "the supervised finalist campaign worker exceeded the "
                f"{boundary} hard deadline"
            ),
        }
        _write_supervision_failure_evidence(
            args,
            output=output,
            source_start=source_start,
            source_end=source_end,
            progress=dict(outcome.get("lastProgress") or {}),
            elapsed_seconds=time.perf_counter() - wall_started,
            failure=failure,
            stop_reason="campaign_hard_timeout",
            temporary_runtime_cleaned=not runtime.exists(),
            cleanup_failure=cleanup_failure,
            supervision={
                "schemaVersion": 1,
                "kind": "owned_finalist_campaign_process_v1",
                "launchId": launch_id,
                "hardBoundaryEnforced": True,
                "timedOut": True,
                "boundary": boundary,
                "cleanupReserveSeconds": (
                    _FINALIST_GLOBAL_CLEANUP_RESERVE_SECONDS
                ),
                "termination": dict(outcome.get("termination") or {}),
                "lastProgress": dict(outcome.get("lastProgress") or {}),
                "originalFailurePrimary": True,
                "lateWorkerResultAccepted": False,
            },
        )
        return 1
    except Exception as exc:
        termination: dict[str, Any] = {}
        progress = _read_supervision_progress(progress_path) or {}
        if process is not None and process.poll() is None:
            try:
                termination = _terminate_owned_process_tree(process)
            except Exception as termination_exc:
                termination = {
                    "requested": True,
                    "workerTerminated": process.poll() is not None,
                    "terminationErrorType": type(termination_exc).__name__,
                    "terminationError": (
                        str(termination_exc).strip()
                        or type(termination_exc).__name__
                    ),
                }
        try:
            temporary_runtime.cleanup()
        except Exception as cleanup_exc:
            cleanup_failure = {
                "stage": "supervisor_runtime_cleanup",
                "errorType": type(cleanup_exc).__name__,
                "message": (
                    str(cleanup_exc).strip() or type(cleanup_exc).__name__
                ),
            }
        try:
            source_end = _source_fingerprint(root)
        except Exception:
            source_end = None
        message = str(exc).strip() or type(exc).__name__
        message = message.replace(str(runtime), "<temporary-runtime>")
        message = message.replace(str(root), "<repository-root>")
        _write_supervision_failure_evidence(
            args,
            output=output,
            source_start=source_start,
            source_end=source_end,
            progress=progress,
            elapsed_seconds=time.perf_counter() - wall_started,
            failure={
                "stage": "campaign_supervision",
                "errorType": type(exc).__name__,
                "message": message,
            },
            stop_reason="campaign_supervision_failure",
            temporary_runtime_cleaned=not runtime.exists(),
            cleanup_failure=cleanup_failure,
            supervision={
                "schemaVersion": 1,
                "kind": "owned_finalist_campaign_process_v1",
                "launchId": launch_id,
                "hardBoundaryEnforced": True,
                "timedOut": False,
                "termination": termination,
                "originalFailurePrimary": True,
            },
        )
        return 1
    finally:
        if process is not None and process.poll() is None:
            try:
                _terminate_owned_process_tree(process)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(
                        timeout=_FINALIST_PROCESS_TERMINATION_WAIT_SECONDS
                    )
                except Exception:
                    pass
        try:
            temporary_runtime.cleanup()
        except Exception:
            pass


def _finalist_campaign_recipe(
    research: Any,
    *,
    bundle_revision_id: str,
    seed_start: int,
    seed_count: int,
    candidates_per_seed: int,
    analysis_mode: str,
) -> dict[str, Any]:
    """Build the explicit V3 selector for the isolated combined-policy lane."""

    _require_finalist_campaign_analysis_mode(analysis_mode)
    if research.execution_context_factory is None:
        raise RuntimeError("the finalist campaign requires Research context authority")
    _, selected_context = research.execution_context_factory.resolve_bundle_revision(
        str(bundle_revision_id)
    )
    payload = _recipe(
        research,
        seed_start=seed_start,
        attempts=seed_count,
        auction_mode="actor_safe_agent_verified",
        play_mode="actor_safe_agent_verified",
        analysis_mode=analysis_mode,
        candidates_per_seed=candidates_per_seed,
        tactic="only_winning_move",
        tactic_mode="qualification_gate",
        tactical_viability_pre_screen=False,
        tactical_viability_maximum_decisions=8,
        tactical_deal_balance=True,
        strategic_minimum_played_cards=24,
    )
    payload["schemaVersion"] = 3
    payload.pop("profiles", None)
    payload["titleAr"] = str(selected_context.workflow_copy["research_title"])
    payload["configuration"] = {
        "schemaVersion": 1,
        "bundleRevisionId": str(bundle_revision_id),
    }
    payload["situation"]["contracts"] = ["sun"]
    payload["qualification"] = {
        "minimumTruth": "robust_best_known",
        "minimumConfidence": 0.75,
        "minimumSurroundingQuality": 70,
        "requireObserverSafety": True,
        "requireAllDecisionAudit": True,
        "maximumRegret": 0,
    }
    return payload


def _context_finalist_simulator(context: Any) -> NaturalLaffaSimulatorV1:
    """Return the simulator already compiled by the frozen Research authority."""

    simulator = context.simulator
    agreement = simulator.decision_agreement
    if simulator.agent_profile.id != "finalist_candidate" or agreement is None:
        raise ValueError("the selected context does not govern finalist_candidate")
    expected = {
        "primaryProfileHash": context.analyzers["deep"].profile.sha256,
        "primaryIdentityHash": str(context.analysis_identity_hashes["deep"]),
        "reanalysisIdentityHash": str(
            context.analysis_identity_hashes["finalist_reanalysis"]
        ),
    }
    actual = {
        "primaryProfileHash": agreement.primary_trace_analysis_profile_hash,
        "primaryIdentityHash": agreement.primary_analysis_identity_hash,
        "reanalysisIdentityHash": agreement.reanalysis_analysis_identity_hash,
    }
    if actual != expected or agreement.reanalysis_trace_analysis_profile is None:
        raise ValueError("the selected context contains drifted finalist authorities")
    if (
        float(agreement.profile.minimum_confidence)
        != float(context.decision_score_profile.minimum_surrounding_confidence)
    ):
        raise ValueError("the selected context contains drifted finalist confidence")
    return simulator


def _canonical_action(value: Mapping[str, Any]) -> str:
    return json.dumps(
        canonical_plain_profile(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_ORDINARY_CANDIDATE_EXCEPTION_MESSAGES = frozenset({
    "constructed_auction_all_pass",
})
_ORDINARY_POLICY_REJECTIONS = frozenset({
    "bid_solver_not_optimal",
    "bid_action_coverage_incomplete",
    "bid_policy_blocked",
    "play_policy_blocked",
})


def _candidate_exception_is_fatal(message: str) -> bool:
    """Only an explicit content miss may become sequential no-good feedback."""

    return str(message) not in _ORDINARY_CANDIDATE_EXCEPTION_MESSAGES


def _fatal_policy_defects(policy: Mapping[str, Any]) -> list[str]:
    """Separate authority/observer defects from ordinary policy rejection."""

    return sorted({
        str(item)
        for item in policy.get("failures") or ()
        if str(item) not in _ORDINARY_POLICY_REJECTIONS
    })


def _direct_deep_decision_score_parity(
    *,
    research: Any,
    context: Any,
    trace: Any,
    cancel_check: Callable[[], bool],
) -> tuple[Any, dict[str, Any]]:
    """Run the production same-game decision parity wrapper unchanged."""

    deep_analysis = context.analyzers["deep"].analyze_trace(
        trace,
        cancel_check=cancel_check,
    )
    evidence = copy.deepcopy(dict(
        research._decision_score_parity_evidence(
            trace,
            deep_analysis,
            context=context,
        )
    ))
    core = copy.deepcopy(evidence)
    declared_hash = str(core.pop("sha256", ""))
    if declared_hash != canonical_sha256(core):
        raise RuntimeError("decision score parity evidence integrity defect")
    if (
        evidence.get("traceHash") != trace.sha256
        or evidence.get("deepAnalysisHash") != deep_analysis.sha256
        or evidence.get("researchContextHash") != context.context_hash
        or type(evidence.get("matched")) is not bool
        or evidence.get("crossLaneComparison") is not False
        or evidence.get("structuralObjectiveCompared") is not False
        or evidence.get("officialBalootScoreCompared") is not False
    ):
        raise RuntimeError("decision score parity evidence provenance defect")
    return deep_analysis, evidence


def _decision_score_parity_failures(
    evidence: Mapping[str, Any],
) -> list[str]:
    if evidence.get("matched") is True:
        return []
    reasons = sorted({
        str(item.get("reason") or "unknown_mismatch")
        for item in evidence.get("mismatches") or ()
        if isinstance(item, Mapping)
    })
    return [f"decision_score_parity:{reason}" for reason in reasons] or [
        "decision_score_parity:unexplained_mismatch"
    ]


def _policy_evidence(trace: Any, simulator: NaturalLaffaSimulatorV1) -> dict[str, Any]:
    records = {item.ordinal: item for item in trace.decision_ledger.records}
    decision_ledger_hash = str(trace.decision_ledger.sha256)
    rules_hash = str(trace.rules_hash)
    agreement = simulator.decision_agreement
    if agreement is None or not agreement.exact_primary_policy:
        raise RuntimeError("finalist campaign requires frozen dual analysis authority")
    role_authorities = (
        {
            "role": "deep_confirmation",
            "profileHash": agreement.primary_trace_analysis_profile_hash,
            "identityHash": agreement.primary_analysis_identity_hash,
            "budget": dict(agreement.primary_analysis_budget),
        },
        {
            "role": "finalist_reanalysis",
            "profileHash": agreement.reanalysis_trace_analysis_profile_hash,
            "identityHash": agreement.reanalysis_analysis_identity_hash,
            "budget": dict(agreement.independent_analysis_budget),
        },
    )
    failures: list[str] = []
    agreement_generation = dict(
        trace.trusted_final_state.get("generation") or {}
    )
    live_trace_agreement_descriptor = agreement_generation.get(
        "decisionAgreementDescriptor"
    )
    live_trace_agreement_profile_hash = agreement_generation.get(
        "decisionAgreementProfileHash"
    )
    if live_trace_agreement_descriptor != agreement.descriptor.to_dict():
        failures.append("live_trace_agreement_descriptor_drift")
    if live_trace_agreement_profile_hash != agreement.profile.sha256:
        failures.append("live_trace_agreement_profile_hash_drift")
    bid_rows: list[dict[str, Any]] = []
    play_rows: list[dict[str, Any]] = []
    forced_rows: list[dict[str, Any]] = []
    forbidden_keys = {
        "completestate",
        "deckorder",
        "hiddenhands",
        "opponenthands",
        "privatehands",
    }

    def hidden_keys(value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).replace("_", "").lower() in forbidden_keys:
                    found.append(str(key))
                found.extend(hidden_keys(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.extend(hidden_keys(item))
        return found

    def hidden_world_count(value: Mapping[str, Any]) -> int:
        raw = value.get("hiddenDealCount")
        if raw is None:
            raw = value.get("analysisHiddenDeals")
        return int(raw) if raw is not None else 0

    def cohort_support(
        values: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            score_evidence = value.get("primaryDecisionScoreEvidence")
            classification = (
                canonical_score_support_classification_v1(score_evidence)
                if isinstance(score_evidence, Mapping)
                else {
                    "completionStatuses": [],
                    "exhaustiveExactSupport": False,
                    "populationCoverageComplete": False,
                    "solverCompletion": "missing",
                    "truthClass": "unresolved",
                }
            )
            result.append({
                "cohortIndex": int(value.get("cohortIndex", index)),
                "analysisAuthorityRole": value.get("analysisAuthorityRole"),
                "traceAnalysisProfileHash": value.get(
                    "traceAnalysisProfileHash"
                ),
                "analysisIdentityHash": value.get("analysisIdentityHash"),
                "analysisBudget": copy.deepcopy(
                    dict(value.get("analysisBudget") or {})
                ),
                "seed": (
                    int(value["seed"])
                    if value.get("seed") is not None
                    else None
                ),
                "hiddenWorldCount": (
                    int(score_evidence.get("hiddenDealSupport", 0))
                    if isinstance(score_evidence, Mapping)
                    else hidden_world_count(value)
                ),
                "confidence": (
                    float(value["confidence"])
                    if value.get("confidence") is not None
                    else None
                ),
                "analysisMethod": value.get("analysisMethod"),
                "completionStatuses": list(
                    classification["completionStatuses"]
                ),
                "exhaustiveSupport": bool(
                    classification["exhaustiveExactSupport"]
                ),
                "populationCoverageComplete": bool(
                    classification["populationCoverageComplete"]
                ),
                "solverCompletion": str(classification["solverCompletion"]),
                "truthClass": str(classification["truthClass"]),
                "recommendedActionHash": canonical_sha256(
                    canonical_plain_profile(
                        value.get("intendedBestAction") or {}
                    )
                ),
            })
        return result

    def authority_failures(
        record: Any,
        policy: Mapping[str, Any],
        values: list[dict[str, Any]],
        *,
        prefix: str,
    ) -> list[str]:
        row_failures: list[str] = []
        if (
            policy.get("traceAnalysisProfileHash")
            != role_authorities[0]["profileHash"]
            or policy.get("analysisIdentityHash")
            != role_authorities[0]["identityHash"]
            or dict(policy.get("analysisBudget") or {})
            != role_authorities[0]["budget"]
            or policy.get("analysisAuthorityRole")
            != role_authorities[0]["role"]
        ):
            row_failures.append(f"{prefix}_primary_analysis_authority_drift")
        declared_seeds = (
            policy.get("primarySeed"),
            policy.get("independentSeed"),
            policy.get("thirdSeed"),
        )
        observed_seeds: list[int] = []
        for index, value in enumerate(values):
            expected_role = role_authorities[0 if index == 0 else 1]
            if (
                value.get("traceAnalysisProfileHash")
                != expected_role["profileHash"]
                or value.get("analysisIdentityHash")
                != expected_role["identityHash"]
                or dict(value.get("analysisBudget") or {})
                != expected_role["budget"]
                or value.get("analysisAuthorityRole") != expected_role["role"]
            ):
                row_failures.append(
                    f"{prefix}_cohort_{index}_analysis_authority_drift"
                )
            expected_seed = agreement.actor_observer_seed(
                public_prefix_hash=record.public_prefix_hash,
                decision_kind=record.decision_kind,
                acting_seat=record.acting_seat,
                cohort_index=index,
            )
            trace_seed = value.get("seed")
            declared_seed = declared_seeds[index]
            if declared_seed is None or int(declared_seed) != expected_seed or (
                trace_seed is not None and int(trace_seed) != expected_seed
            ):
                row_failures.append(f"{prefix}_cohort_{index}_seed_drift")
            observed_seeds.append(expected_seed)
        if len(set(observed_seeds)) != len(observed_seeds):
            row_failures.append(f"{prefix}_cohort_seed_collision")
        status = str(policy.get("status") or "")
        if status not in {
            "strict_confirmed",
            "primary_robust_competent",
            "unresolved",
        }:
            row_failures.append(f"{prefix}_agreement_status_drift")
        if bool(policy.get("qualificationBlocked")) != (status == "unresolved"):
            row_failures.append(f"{prefix}_qualification_status_drift")
        if bool(policy.get("featuredTruthEligible")) != (
            status == "strict_confirmed"
        ):
            row_failures.append(f"{prefix}_featured_truth_status_drift")
        return row_failures

    for item in trace.trusted_final_state.get("decisionPolicyTraces") or ():
        ordinal = int(item["ordinal"])
        record = records.get(ordinal)
        if record is None:
            continue
        policy = dict(item.get("trace") or {})
        selected_matches = _canonical_action(
            dict(policy.get("selectedAction") or {})
        ) == _canonical_action(record.chosen_action)
        if record.forced:
            proof = policy.get("forcedLegalityProof")
            proof_hash = (
                str(proof.get("sha256") or "")
                if isinstance(proof, Mapping)
                else None
            )
            forced_rows.append({
                "decisionId": record.decision_id,
                "decisionKind": record.decision_kind,
                "ordinal": ordinal,
                "forced": True,
                "actingSeat": record.acting_seat.name.lower(),
                "publicPrefixHash": record.public_prefix_hash,
                "legalActionSetHash": legal_action_set_hash_v1(
                    canonical_plain_profile(action)
                    for action in record.legal_actions
                ),
                "rulesHash": rules_hash,
                "selectedAction": canonical_plain_profile(
                    record.chosen_action
                ),
                "selectedActionMatchesTrace": selected_matches,
                "forcedLegalityProof": (
                    canonical_plain_profile(proof)
                    if isinstance(proof, Mapping)
                    else None
                ),
                "forcedLegalityProofHash": proof_hash,
            })
            if not selected_matches:
                failures.append("forced_selected_action_drift")
            if not _canonical_sha256_value(proof_hash):
                failures.append("forced_legality_proof_missing")
            continue
        primary = dict(policy.get("primaryTrace") or policy)
        cohort_traces = [
            dict(value)
            for value in policy.get("cohorts") or (primary,)
        ]
        leaks = sorted(set(hidden_keys(policy)))
        common = {
            "decisionId": record.decision_id,
            "decisionKind": record.decision_kind,
            "ordinal": ordinal,
            "forced": False,
            "actingSeat": record.acting_seat.name.lower(),
            "publicPrefixHash": record.public_prefix_hash,
            "legalActionSetHash": legal_action_set_hash_v1(
                canonical_plain_profile(action)
                for action in record.legal_actions
            ),
            "rulesHash": rules_hash,
            "selectedAction": canonical_plain_profile(record.chosen_action),
            "observerSafe": policy.get("observerSafe") is True,
            "selectedActionMatchesTrace": selected_matches,
            "primaryDecisionScoreEvidenceHash": policy.get(
                "primaryDecisionScoreEvidenceHash"
            ),
            "hiddenWorldCount": hidden_world_count(primary),
            "cohortSupport": cohort_support(cohort_traces),
            "agreementReasons": sorted({
                str(reason) for reason in policy.get("reasons") or ()
            }),
            "initialAgreementReasons": sorted({
                str(reason) for reason in policy.get("initialReasons") or ()
            }),
            "agreementCount": int(policy.get("agreementCount", 0)),
            "majorityActionHash": policy.get("majorityActionHash"),
            "leakKeys": leaks,
            "traceAnalysisProfileHash": policy.get("traceAnalysisProfileHash"),
            "analysisIdentityHash": policy.get("analysisIdentityHash"),
            "analysisAuthorityRole": policy.get("analysisAuthorityRole"),
            "analysisBudget": copy.deepcopy(
                dict(policy.get("analysisBudget") or {})
            ),
            "agreementStatus": policy.get("status"),
            "qualificationBlocked": policy.get("qualificationBlocked") is True,
            "featuredTruthEligible": policy.get("featuredTruthEligible") is True,
        }
        if record.decision_kind == "bid":
            row_failures = authority_failures(
                record, policy, cohort_traces, prefix="bid"
            )
            if policy.get("source") != "finalist_multi_cohort_agent":
                row_failures.append("wrong_bid_policy")
            if policy.get("selectionMode") != "primary_policy_only":
                row_failures.append("bid_selection_mode_drift")
            if policy.get("primaryActionPreserved") is not True:
                row_failures.append("bid_primary_action_not_preserved")
            if primary.get("strategyId") != "adaptive_observer_safe_bidding_v2":
                row_failures.append("wrong_primary_bid_policy")
            if primary.get("profileId") != "finalist_candidate":
                row_failures.append("wrong_bid_profile")
            if policy.get("observerSafe") is not True:
                row_failures.append("bid_not_observer_safe")
            if not selected_matches:
                row_failures.append("bid_selected_action_drift")
            if policy.get("qualificationBlocked") is True:
                row_failures.append("bid_policy_blocked")
            if leaks:
                row_failures.append("bid_policy_hidden_feature")
            bid_rows.append({
                **common,
                "analysisMethod": primary.get("analysisMethod"),
                "profileId": primary.get("profileId"),
                "cohortCount": int(policy.get("cohortCount", 1)),
                "failures": row_failures,
            })
            failures.extend(row_failures)
        elif record.decision_kind == "play":
            row_failures = authority_failures(
                record, policy, cohort_traces, prefix="play"
            )
            if primary.get("strategyId") != "adaptive_observer_safe_v2":
                row_failures.append("wrong_play_policy")
            if primary.get("profileId") != "finalist_candidate":
                row_failures.append("wrong_play_profile")
            if policy.get("source") != "finalist_multi_cohort_agent":
                row_failures.append("wrong_play_agreement_policy")
            if policy.get("selectionMode") != "primary_policy_only":
                row_failures.append("play_selection_mode_drift")
            if policy.get("primaryActionPreserved") is not True:
                row_failures.append("play_primary_action_not_preserved")
            if policy.get("observerSafe") is not True:
                row_failures.append("play_not_observer_safe")
            if any(
                dict(value.get("analysisPolicyAuthority") or {})
                != dict(simulator.analysis_policy_authority)
                for value in cohort_traces
            ):
                row_failures.append("play_analysis_authority_drift")
            if not selected_matches:
                row_failures.append("play_selected_action_drift")
            if policy.get("qualificationBlocked") is True:
                row_failures.append("play_policy_blocked")
            if leaks:
                row_failures.append("play_policy_hidden_feature")
            play_rows.append({
                **common,
                "analysisMethod": primary.get("analysisMethod"),
                "profileId": primary.get("profileId"),
                "cohortCount": int(policy.get("cohortCount", 1)),
                "failures": row_failures,
            })
            failures.extend(row_failures)
    expected_non_forced_bids = sum(
        item.decision_kind == "bid" and not item.forced
        for item in trace.decision_ledger.records
    )
    expected_non_forced_plays = sum(
        item.decision_kind == "play" and not item.forced
        for item in trace.decision_ledger.records
    )
    expected_forced = sum(item.forced for item in trace.decision_ledger.records)
    if len(bid_rows) != expected_non_forced_bids:
        failures.append("bid_policy_trace_incomplete")
    if len(play_rows) != expected_non_forced_plays:
        failures.append("play_policy_trace_incomplete")
    if len(forced_rows) != expected_forced:
        failures.append("forced_policy_trace_incomplete")
    failures = list(dict.fromkeys(failures))
    stability_failures = [
        item
        for item in failures
        if item in {"bid_policy_blocked", "play_policy_blocked"}
    ]
    primary_policy_failures = [
        item for item in failures if item not in stability_failures
    ]
    rows = [*bid_rows, *play_rows]
    competent_decision_ids = sorted({
        str(row["decisionId"])
        for row in rows
        if row.get("agreementStatus") == "primary_robust_competent"
    })
    unresolved_decision_ids = sorted({
        str(row["decisionId"])
        for row in rows
        if row.get("agreementStatus") == "unresolved"
    })
    featured_truth_blocked_ids = sorted({
        *competent_decision_ids,
        *unresolved_decision_ids,
    })
    observer_safety_failures = [
        item
        for item in primary_policy_failures
        if item in {
            "bid_not_observer_safe",
            "play_not_observer_safe",
            "bid_policy_hidden_feature",
            "play_policy_hidden_feature",
        }
    ]
    shared_action_failures = [
        item
        for item in primary_policy_failures
        if item in {
            "bid_selection_mode_drift",
            "play_selection_mode_drift",
            "bid_primary_action_not_preserved",
            "play_primary_action_not_preserved",
            "bid_selected_action_drift",
            "play_selected_action_drift",
            "bid_not_observer_safe",
            "play_not_observer_safe",
            "bid_policy_hidden_feature",
            "play_policy_hidden_feature",
            "bid_policy_trace_incomplete",
            "play_policy_trace_incomplete",
        }
    ]
    core = {
        "schemaVersion": 2,
        "profileId": FINALIST_CAMPAIGN_PROFILE_ID,
        "auctionProfileId": "finalist_candidate",
        "playProfileId": "finalist_candidate",
        "auctionSelectionAuthority": (
            "finalist_multi_cohort_agent_primary_policy"
        ),
        "playSelectionAuthority": "finalist_multi_cohort_agent_primary_policy",
        "cpSatAuctionRole": "exact_required_action_rebind_only",
        "cpSatPlayRole": "exact_required_action_rebind_only",
        "analysisPolicyAuthority": simulator.analysis_policy_authority,
        "agreementDescriptor": agreement.descriptor.to_dict(),
        "agreementProfileHash": agreement.profile.sha256,
        "liveTraceAgreementDescriptor": copy.deepcopy(
            live_trace_agreement_descriptor
        ),
        "liveTraceAgreementProfileHash": live_trace_agreement_profile_hash,
        "analysisAuthorities": {
            "primary": copy.deepcopy(role_authorities[0]),
            "reanalysis": copy.deepcopy(role_authorities[1]),
        },
        "decisionLedgerHash": decision_ledger_hash,
        "rulesHash": rules_hash,
        "nonForcedBidCount": expected_non_forced_bids,
        "nonForcedPlayCount": expected_non_forced_plays,
        "forcedDecisionCount": len(forced_rows),
        "forcedDecisions": forced_rows,
        "bids": bid_rows,
        "plays": play_rows,
        "oneSharedActionAcrossIndistinguishableWorlds": not shared_action_failures,
        "zeroObserverLeaks": not any(row["leakKeys"] for row in rows),
        "observerSafetyPassed": not observer_safety_failures,
        "primaryPolicyValid": not primary_policy_failures,
        "stabilityResolved": not stability_failures,
        "stabilityFailures": stability_failures,
        "competentDecisionIds": competent_decision_ids,
        "unresolvedDecisionIds": unresolved_decision_ids,
        "featuredTruthBlockedDecisionIds": featured_truth_blocked_ids,
        "primaryPolicyFailures": primary_policy_failures,
        "failures": failures,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _tactic_retained(
    qualification: Mapping[str, Any],
    *,
    featured_truth_blocked_ids: Iterable[str] = (),
) -> bool:
    blocked = {str(item) for item in featured_truth_blocked_ids}
    required = [
        dict(item)
        for item in qualification.get("tacticEvaluations") or ()
        if item.get("kind") == "only_winning_move"
        and item.get("mode") in {"hard", "qualification_gate"}
    ]
    if not required:
        return False
    for item in required:
        eligible_evidence = [
            dict(evidence)
            for evidence in item.get("matchingEvidence") or ()
            if str(evidence.get("decisionId")) not in blocked
        ]
        if (
            not bool(item.get("eligible"))
            or not eligible_evidence
            or max(float(evidence.get("magnitude", 0.0)) for evidence in eligible_evidence)
            < float(item.get("minimumScore", 0.0))
        ):
            return False
    return True


def _observer_safety_passed(
    audit: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    return bool(
        audit.get("informationSetComplete")
        and audit.get("sequentialResponseComplete")
        and policy.get("zeroObserverLeaks") is True
        and policy.get("observerSafetyPassed") is True
    )


def _finalist_policy_gate_evidence(
    *,
    audit: Mapping[str, Any],
    policy: Mapping[str, Any],
    agreement_unresolved: int,
) -> dict[str, bool]:
    return {
        "zeroObserverLeaks": bool(policy["zeroObserverLeaks"]),
        "observerSafety": _observer_safety_passed(audit, policy),
        "sharedActionAcrossWorlds": bool(
            policy["oneSharedActionAcrossIndistinguishableWorlds"]
        ),
        "finalistPrimaryAuctionAndPlay": bool(policy["primaryPolicyValid"]),
        "finalistAgreementResolved": bool(
            int(agreement_unresolved) == 0 and policy["stabilityResolved"]
        ),
    }


def _actor_safe_mismatch_partition(
    *,
    research: Any,
    recipe: ResearchRecipeV3,
    simulation: Any,
    project_plan: Any,
    play_plan: Any,
    candidate: Any,
    rules: Any,
    context: Any,
) -> tuple[list[str], list[str]]:
    """Keep policy target misses separate from semantic integrity defects."""

    target_mismatches = list(
        research._policy_target_mismatches(recipe, simulation)
    )
    semantic_mismatches = sorted(set([
        *research._semantic_mismatches(
            recipe,
            simulation,
            context=context,
            include_policy_target_mismatches=False,
        ),
        *research._compiled_play_mismatches(
            play_plan,
            simulation,
            project_plan=project_plan,
            project_domain=rules.project_domain,
            scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
            structural_domain=rules.structural_domain,
            expected_deck_order=tuple(int(item) for item in candidate.deck_order),
            expected_deck_commitment=hashlib.sha256(
                bytes(candidate.deck_order)
            ).hexdigest(),
            expected_candidate_hash=candidate.candidate_hash,
            expected_dealer=recipe.situation["dealer"],
            expected_round_number=1,
            expected_next_dealer=rules.structural_domain.next_dealer(
                Seat.parse(recipe.situation["dealer"])
            ),
            starting_match_score=tuple(
                int(item) for item in recipe.situation["matchScore"]
            ),
        ),
    ]))
    return sorted(set(target_mismatches)), semantic_mismatches


def _evaluate_finalist_candidate(
    *,
    research: Any,
    context: Any,
    recipe: ResearchRecipeV3,
    simulator: NaturalLaffaSimulatorV1,
    candidate: Any,
    seed: int,
    candidate_index: int,
    cancel_check: Callable[[], bool],
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    base = {
        "seed": int(seed),
        "candidateIndex": int(candidate_index),
        "candidateHash": str(candidate.candidate_hash),
        "dealModelHash": str(candidate.model_hash),
        "dealSolverStatus": str(candidate.solver_status),
    }
    try:
        simulation = simulator.simulate_preordered_deck(
            seed,
            candidate.deck_order,
            deal_generator_provenance={
                **candidate.provenance_dict(),
                "qualificationProfileId": FINALIST_CAMPAIGN_PROFILE_ID,
                "qualificationProfileRevision": FINALIST_CAMPAIGN_PROFILE_REVISION,
                "auctionConstructionMode": "actor_safe_agent_verified",
                "playConstructionMode": "actor_safe_agent_verified",
                "auctionSelectionAuthority": (
                    "finalist_multi_cohort_agent_primary_policy"
                ),
                "playSelectionAuthority": (
                    "finalist_multi_cohort_agent_primary_policy"
                ),
                "auctionCpSatRole": "exact_required_action_rebind_only",
                "playCpSatRole": "exact_required_action_rebind_only",
                "playPolicyRelease": simulator.agent_release,
            },
            dealer=Seat.parse(recipe.situation["dealer"]),
            match_score=tuple(
                int(item) for item in recipe.situation["matchScore"]
            ),
            cancel_check=cancel_check,
        )
        if not simulation.publishable_laffa:
            raise ValueError("constructed_auction_all_pass")
        trace = simulation.final_trace
        replay = CompleteTraceReplayerV1(simulator.rules).replay(trace)
        auction = research._compile_auction(
            recipe,
            candidate.deck_order,
            required_actions=research._auction_actions_from_trace(trace),
            enforce_recipe=False,
            context=context,
        )
        if auction is None:
            raise ValueError("auction_cp_sat_rebind_failed")
        scripted_escalations = tuple(
            dict(item)
            for item in dict(trace.trusted_final_state.get("generation") or {}).get(
                "scriptedEscalations"
            )
            or ()
        )
        escalation = research.escalation_compiler.compile(
            dealer=Seat.parse(recipe.situation["dealer"]),
            declarer=auction.declarer,
            contract=auction.contract,
            match_score=tuple(
                int(item) for item in recipe.situation["matchScore"]
            ),
            rules=simulator.rules,
            required_level=(
                tuple(recipe.situation.get("escalations") or ())[0]
                if recipe.situation.get("escalations")
                else "normal"
            ),
            required_actions=tuple(
                (
                    str(item["actor"]),
                    str(item["action"]),
                    (
                        str(item["requestedMode"])
                        if item.get("requestedMode") is not None else None
                    ),
                )
                for item in scripted_escalations
            ),
            maximum_conflicts=int(recipe.resources["maximumConflicts"]),
            timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
        )
        if escalation is None:
            raise ValueError("escalation_cp_sat_rebind_failed")
        cards, ekka = research._played_line(simulation)
        project_plan, play_plan, _ = research._compile_complete_play(
            recipe,
            candidate.deck_order,
            auction,
            escalation,
            required_cards=cards,
            required_ekka_leads=ekka,
            context=context,
        )
        target_mismatches, semantic_mismatches = _actor_safe_mismatch_partition(
            research=research,
            recipe=recipe,
            simulation=simulation,
            project_plan=project_plan,
            play_plan=play_plan,
            candidate=candidate,
            rules=simulator.rules,
            context=context,
        )
        policy = _policy_evidence(trace, simulator)
        fatal_policy_defects = _fatal_policy_defects(policy)
        agreement_generation = dict(
            trace.trusted_final_state.get("generation") or {}
        )
        agreement_unresolved = int(
            agreement_generation.get("decisionAgreementUnresolvedCount", 0)
        )
        analysis, decision_score_parity = _direct_deep_decision_score_parity(
            research=research,
            context=context,
            trace=trace,
            cancel_check=cancel_check,
        )
        decision_score_parity_failures = _decision_score_parity_failures(
            decision_score_parity
        )
        tactical = research._verify_tactics(
            trace,
            analysis,
            recipe=recipe,
            cancel_check=cancel_check,
            context=context,
            featured_truth_blocked_ids=policy[
                "featuredTruthBlockedDecisionIds"
            ],
        )
        audit = research._decision_audit(
            trace,
            analysis,
            tactical,
            context=context,
            configuration=recipe.configuration,
        )
        qualification = research._qualification(
            recipe, audit, analysis, context=context
        )
        decision_selection = evaluate_decision_selection_v1(
            context.decision_score_profile,
            perspectives=analysis.laffa_evidence.perspectives,
            analyses=analysis.decisions,
            analysis_hash=analysis.sha256,
            decision_kinds={
                str(item["decisionId"]): str(item["decisionKind"])
                for item in audit["decisions"]
            },
            tactical_verifications=tactical,
            competent_decision_ids=policy["competentDecisionIds"],
            unresolved_decision_ids=policy["unresolvedDecisionIds"],
            authority_evidence_hash=policy["sha256"],
        )
        qualification = apply_decision_selection_qualification_v1(
            qualification, decision_selection
        )
        governed_score = context.score_profile.score(analysis.laffa_evidence)
        extra_failures: list[str] = []
        if not governed_score.passed:
            extra_failures.extend(
                f"score_profile_gate:{item}"
                for item in governed_score.rejection_reasons
            )
        if semantic_mismatches:
            extra_failures.extend(
                f"semantic_parity:{item}" for item in semantic_mismatches
            )
        extra_failures.extend(decision_score_parity_failures)
        extra_failures.extend(
            f"construction_target_miss:{item}" for item in target_mismatches
        )
        if str(auction.solver_status) != "optimal":
            extra_failures.append("auction_rebind_solver_incomplete")
        if str(play_plan.solver_status) != "optimal":
            extra_failures.append("play_rebind_solver_incomplete")
        extra_failures.extend(
            f"combined_policy:{item}" for item in policy["failures"]
        )
        if agreement_unresolved:
            extra_failures.append("combined_policy:finalist_agreement_unresolved")
        decisions = list(audit["decisions"])
        non_forced = [item for item in decisions if not bool(item["forced"])]
        audit_problems = [
            {
                "ordinal": int(item["ordinal"]),
                "decisionId": str(item["decisionId"]),
                "kind": str(item["decisionKind"]),
                "actualAction": copy.deepcopy(dict(item["actualAction"])),
                "recommendedAction": copy.deepcopy(
                    dict(item["recommendedAction"])
                ),
                "expectedRegret": float(item["expectedRegret"]),
                "recommendationMismatch": not bool(
                    item["actualMatchesRecommendation"]
                ),
                "materialCounterexample": bool(
                    dict(item.get("sequentialResponseEvidence") or {}).get(
                        "materialCounterexampleFound"
                    )
                ),
            }
            for item in non_forced
            if (
                not bool(item["actualMatchesRecommendation"])
                or bool(
                    dict(item.get("sequentialResponseEvidence") or {}).get(
                        "materialCounterexampleFound"
                    )
                )
            )
        ]
        bid_regrets = [
            float(item["expectedRegret"])
            for item in non_forced
            if item["decisionKind"] == "bid"
        ]
        play_regrets = [
            float(item["expectedRegret"])
            for item in non_forced
            if item["decisionKind"] == "play"
        ]
        tactic_retained = _tactic_retained(
            qualification,
            featured_truth_blocked_ids=policy[
                "featuredTruthBlockedDecisionIds"
            ],
        )
        policy_gates = _finalist_policy_gate_evidence(
            audit=audit,
            policy=policy,
            agreement_unresolved=agreement_unresolved,
        )
        observer_safe = policy_gates["observerSafety"]
        if not tactic_retained:
            extra_failures.append("combined_policy:tactic_not_retained")
        if not observer_safe:
            extra_failures.append("combined_policy:observer_safety")
        if str(trace.trusted_final_state.get("contract")) != "sun":
            extra_failures.append("combined_policy:exact_contract")
        if audit_problems:
            extra_failures.append("combined_policy:decision_counterexample")
        failures = list(dict.fromkeys([
            *qualification.get("failures", ()), *extra_failures
        ]))
        qualified = bool(qualification.get("qualified")) and not failures
        qualification = {
            **copy.deepcopy(dict(qualification)),
            "qualified": qualified,
            "truthLabel": (
                str(qualification["truthLabel"]) if qualified else "rejected"
            ),
            "failures": failures,
        }
        gates = {
            "semanticReplay": bool(replay.final_state_hash),
            "zeroSemanticDefects": not semantic_mismatches,
            "auctionCpSatExactTraceRebind": (
                str(auction.solver_status) == "optimal"
            ),
            "playCpSatExactTraceRebind": (
                str(play_plan.solver_status) == "optimal"
            ),
            "fullCpSatReplayParity": not semantic_mismatches,
            "decisionScoreParity": bool(decision_score_parity["matched"]),
            "exactContract": str(trace.trusted_final_state.get("contract")) == "sun",
            "tacticRetained": tactic_retained,
            **policy_gates,
            "auditComplete": bool(audit.get("complete")),
            "informationSetComplete": bool(audit.get("informationSetComplete")),
            "sequentialResponseComplete": bool(audit.get("sequentialResponseComplete")),
            "qualityAtLeast70": float(
                qualification.get("minimumSurroundingQuality", 0.0)
            ) >= 70.0,
            "zeroRecommendationMismatches": not any(
                item["recommendationMismatch"] for item in audit_problems
            ),
            "zeroMaterialCounterexamples": not any(
                item["materialCounterexample"] for item in audit_problems
            ),
            "regretGate": max([*bid_regrets, *play_regrets], default=0.0) <= 0.0,
            "decisionSelection": bool(decision_selection["passed"]),
            "scoreProfile": bool(governed_score.passed),
            "ordinaryQualification": bool(qualification["qualified"]),
        }
        qualified = qualified and all(gates.values())
        qualification["qualified"] = qualified
        if not qualified:
            qualification["truthLabel"] = "rejected"
        stable = {
            **base,
            "traceHash": trace.sha256,
            "rulesHash": trace.rules_hash,
            "decisionLedger": trace.decision_ledger.to_dict(),
            "gameHash": trace.game_hash,
            "deckCommitment": trace.seed_provenance["deckCommitment"],
            "auctionPlanHash": auction.sha256,
            "projectPlanHash": project_plan.sha256,
            "playPlanHash": play_plan.sha256,
            "playModelHash": play_plan.model_hash,
            "analysisHash": analysis.sha256,
            "decisionScoreParityHash": decision_score_parity["sha256"],
            "auditHash": audit["sha256"],
            "decisionSelectionHash": decision_selection["sha256"],
            "qualified": qualified,
            "qualification": qualification,
            "score": governed_score.to_dict(),
            "gates": gates,
            "tacticRetained": tactic_retained,
            "maximumBidRegret": max(bid_regrets, default=0.0),
            "maximumPlayRegret": max(play_regrets, default=0.0),
            "auditProblems": audit_problems,
            "recipeTargetMismatches": target_mismatches,
            "semanticMismatches": semantic_mismatches,
            "decisionScoreParity": decision_score_parity,
            "policyEvidence": policy,
            "selectionAndVerification": {
                "qualificationProfileId": FINALIST_CAMPAIGN_PROFILE_ID,
                "qualificationProfileRevision": (
                    FINALIST_CAMPAIGN_PROFILE_REVISION
                ),
                "auctionSelectionAuthority": (
                    "finalist_multi_cohort_agent_primary_policy"
                ),
                "playSelectionAuthority": (
                    "finalist_multi_cohort_agent_primary_policy"
                ),
                "auctionCpSatVerification": {
                    "role": "exact_required_action_rebind_only",
                    "planHash": auction.sha256,
                    "solverStatus": str(auction.solver_status),
                },
                "playCpSatVerification": {
                    "role": "exact_required_action_rebind_only",
                    "projectPlanHash": project_plan.sha256,
                    "planHash": play_plan.sha256,
                    "solverStatus": str(play_plan.solver_status),
                },
                "postHocActionRepair": False,
            },
            "fatalPolicyDefects": fatal_policy_defects,
            "decisionAgreement": {
                "descriptor": simulator.decision_agreement.descriptor.to_dict(),
                "profileHash": agreement_generation.get(
                    "decisionAgreementProfileHash"
                ),
                "decisionCount": int(
                    agreement_generation.get("decisionAgreementDecisionCount", 0)
                ),
                "unresolvedCount": agreement_unresolved,
            },
        }
        stable_hash = canonical_sha256(stable)
        return {
            **stable,
            "stableHash": stable_hash,
            "wallSeconds": time.perf_counter() - wall_started,
            "cpuSeconds": time.process_time() - cpu_started,
            "fatalDefect": bool(
                semantic_mismatches
                or not bool(decision_score_parity["matched"])
                or not policy["zeroObserverLeaks"]
                or fatal_policy_defects
            ),
        }
    except InterruptedError:
        return {
            **base,
            "status": "bounded_timeout",
            "qualified": False,
            "fatalDefect": False,
            "failures": ["bounded_timeout"],
            "wallSeconds": time.perf_counter() - wall_started,
            "cpuSeconds": time.process_time() - cpu_started,
        }
    except (AssertionError, RuntimeError, ValueError) as exc:
        message = str(exc).strip() or type(exc).__name__
        semantic = _candidate_exception_is_fatal(message)
        return {
            **base,
            "status": "semantic_defect" if semantic else "rejected_before_audit",
            "qualified": False,
            "fatalDefect": semantic,
            "failures": [message],
            "errorType": type(exc).__name__,
            "wallSeconds": time.perf_counter() - wall_started,
            "cpuSeconds": time.process_time() - cpu_started,
        }


def _run_finalist_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    output = _safe_campaign_output(
        root,
        Path(args.output),
        require_absent=True,
    )
    wall_started = time.perf_counter()
    monotonic_started = time.monotonic()
    cpu_started = time.process_time()
    attempts: int | None = None
    candidates_per_seed: int | None = None
    maximum_seed_seconds: float | None = None
    maximum_wall_seconds: float | None = None
    source_start: dict[str, Any] | None = None
    source_end: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    repeat: dict[str, Any] | None = None
    stop_reason = "seed_bound_exhausted"
    fatal_failure: str | None = None
    failure: dict[str, str] | None = None
    cleanup_failure: dict[str, str] | None = None
    context_identity: dict[str, Any] | None = None
    backend_runtime_identity: dict[str, Any] | None = None
    qualified_row: dict[str, Any] | None = None
    temporary_runtime: tempfile.TemporaryDirectory[str] | None = None
    runtime: Path | None = None
    container: Any | None = None
    supervised_runtime: tuple[Path, Path, str] | None = None
    progress_path: Path | None = None
    progress_state: dict[str, Any] = {
        "schemaVersion": 1,
        "phase": "argument_validation",
        "completedCandidates": [],
        "completedSeeds": [],
    }
    stage = "argument_validation"

    def update_progress(**updates: Any) -> None:
        if progress_path is None:
            return
        progress_state.update(copy.deepcopy(updates))
        _atomic_write_json(progress_path, progress_state)

    def failure_evidence(
        failure_stage: str,
        exc: Exception,
    ) -> dict[str, str]:
        message = str(exc).strip() or type(exc).__name__
        replacements = ((runtime, "<temporary-runtime>"), (root, "<repository-root>"))
        for path, replacement in replacements:
            if path is not None:
                message = message.replace(str(path), replacement)
        return {
            "stage": failure_stage,
            "errorType": type(exc).__name__,
            "message": message,
        }

    try:
        attempts = int(args.attempts)
        candidates_per_seed = int(args.candidates_per_seed)
        maximum_seed_seconds = float(args.maximum_seed_seconds)
        maximum_wall_seconds = float(args.maximum_wall_seconds)
        if not 1 <= attempts <= 5:
            raise ValueError(
                "the finalist campaign seed count must be between 1 and 5"
            )
        if not 1 <= candidates_per_seed <= 4:
            raise ValueError(
                "the finalist campaign candidate count must be between 1 and 4"
            )
        if not 30 <= maximum_seed_seconds <= 120:
            raise ValueError("maximum seed seconds must be between 30 and 120")
        if not 30 <= maximum_wall_seconds <= 600:
            raise ValueError("maximum wall seconds must be between 30 and 600")
        _require_finalist_campaign_analysis_mode(str(args.analysis_mode))

        stage = "supervision_authority"
        supervised_runtime = _validated_supervised_runtime(args)

        stage = "source_fingerprint_start"
        if supervised_runtime is not None:
            progress_path = supervised_runtime[1]
            update_progress(
                phase=stage,
                launchId=supervised_runtime[2],
            )
        source_start = _source_fingerprint(root)
        global_deadline = monotonic_started + maximum_wall_seconds

        stage = "temporary_runtime_creation"
        update_progress(phase=stage)
        if supervised_runtime is None:
            temporary_runtime = tempfile.TemporaryDirectory(
                prefix=_FINALIST_RUNTIME_PREFIX
            )
            runtime = Path(temporary_runtime.name)
        else:
            runtime = supervised_runtime[0]

        stage = "container_bootstrap"
        update_progress(phase=stage)
        container = build_container(root=root, writable_root=runtime)

        stage = "configuration_resolution"
        update_progress(phase=stage)
        resolved = container.configuration.resolve()
        bundle = dict(resolved["bundle"])
        if str(bundle["version"]) != str(args.bundle_version):
            raise ValueError(
                f"expected bundle {args.bundle_version}, got {bundle['version']}"
            )
        stage = "recipe_validation"
        update_progress(phase=stage)
        recipe_payload = _finalist_campaign_recipe(
            container.research,
            bundle_revision_id=str(bundle["revisionId"]),
            seed_start=int(args.seed_start),
            seed_count=attempts,
            candidates_per_seed=candidates_per_seed,
            analysis_mode=str(args.analysis_mode),
        )
        recipe, context = _validated_finalist_recipe_and_context(
            container.research,
            recipe_payload,
        )
        stage = "campaign_execution"
        update_progress(phase=stage)
        simulator = _context_finalist_simulator(context)
        backend_runtime_identity = _exact_search_backend_runtime_identity(
            context.exact_search_backend
        )
        context_identity = _finalist_control_plane(
            context=context,
            recipe=recipe,
            simulator=simulator,
            backend_runtime_identity=backend_runtime_identity,
            bundle_version=str(bundle["version"]),
        )
        update_progress(controlPlane=context_identity)
        for seed in range(
            int(args.seed_start), int(args.seed_start) + attempts
        ):
            if time.monotonic() >= global_deadline:
                stop_reason = "global_wall_bound"
                break
            seed_started = time.perf_counter()
            seed_cpu_started = time.process_time()
            seed_deadline = min(
                global_deadline,
                time.monotonic() + maximum_seed_seconds,
            )
            update_progress(
                phase="seed_execution",
                seed=int(seed),
                seedDeadlineMonotonic=seed_deadline,
                seedResultRecordedMonotonic=None,
                seedDeadlineReleased=False,
                candidateIndex=None,
                candidateHash=None,
            )
            spec = container.research._compile_deal_spec(
                recipe, seed=seed, context=context
            )

            def evaluate_candidate(
                candidate: Any,
                candidate_index: int,
            ) -> dict[str, Any]:
                return _evaluate_finalist_candidate(
                    research=container.research,
                    context=context,
                    recipe=recipe,
                    simulator=simulator,
                    candidate=candidate,
                    seed=seed,
                    candidate_index=candidate_index,
                    cancel_check=lambda deadline=seed_deadline: (
                        time.monotonic() >= deadline
                    ),
                )

            def report_candidate(row: dict[str, Any]) -> None:
                completed_candidates = [
                    *list(progress_state.get("completedCandidates") or ()),
                    {
                        "seed": int(seed),
                        "candidateIndex": int(row["candidateIndex"]),
                        "candidateHash": str(row["candidateHash"]),
                        "status": str(row.get("status") or "unknown"),
                        "qualified": bool(row.get("qualified")),
                        "tacticRetained": bool(row.get("tacticRetained")),
                        "fatalDefect": bool(row.get("fatalDefect")),
                        "semanticDefect": bool(row.get("semanticMismatches"))
                        or row.get("status") == "semantic_defect",
                        "observerLeakDefect": not bool(
                            dict(row.get("policyEvidence") or {}).get(
                                "zeroObserverLeaks", True
                            )
                        ),
                    },
                ]
                update_progress(
                    phase="candidate_result_recorded",
                    candidateIndex=int(row["candidateIndex"]),
                    candidateHash=str(row["candidateHash"]),
                    completedCandidates=completed_candidates,
                )
                print(json.dumps({
                    "seed": seed,
                    "candidate": int(row["candidateIndex"]),
                    "qualified": bool(row.get("qualified")),
                    "tacticRetained": bool(row.get("tacticRetained")),
                    "quality": dict(row.get("qualification") or {}).get(
                        "minimumSurroundingQuality"
                    ),
                    "wallSeconds": round(float(row["wallSeconds"]), 3),
                    "failures": list(
                        dict(row.get("qualification") or {}).get("failures")
                        or row.get("failures") or ()
                    ),
                }, ensure_ascii=False, sort_keys=True), flush=True)

            def report_phase(
                phase: str,
                details: Mapping[str, Any],
            ) -> None:
                update_progress(
                    phase=phase,
                    seed=int(seed),
                    seedDeadlineMonotonic=seed_deadline,
                    **dict(details),
                )

            sequential = _run_product_sequential_candidate_search(
                container.research.deal_generator,
                spec,
                root_seed=seed,
                dealer=Seat.parse(recipe.situation["dealer"]),
                rules=simulator.rules,
                maximum_candidates=candidates_per_seed,
                deadline_monotonic=seed_deadline,
                evaluate_candidate=evaluate_candidate,
                on_result=report_candidate,
                on_phase=report_phase,
            )
            seed_candidate_rows = list(sequential["rows"])
            rows.extend(seed_candidate_rows)
            evaluated = len(seed_candidate_rows)
            retained = sum(
                bool(row.get("tacticRetained")) for row in seed_candidate_rows
            )
            seed_stop = str(sequential["stopReason"])
            if seed_stop in {
                "candidate_search_deadline",
                "candidate_generation_deadline",
                "candidate_evaluation_deadline",
            }:
                seed_stop = (
                    "global_wall_bound"
                    if time.monotonic() >= global_deadline
                    else "seed_wall_bound"
                )
            qualified_row = sequential["qualifiedRow"]
            repeat = sequential["terminalRepeat"]
            fatal_failure = sequential["fatalFailure"]
            if fatal_failure is not None:
                stop_reason = fatal_failure
            elif qualified_row is not None:
                stop_reason = "qualified_candidate"
            seed_summary = {
                "schemaVersion": 1,
                "kind": "combined_finalist_seed_search_result_v1",
                "seed": seed,
                "generatedCandidateCount": evaluated,
                "evaluatedCandidateCount": evaluated,
                "tacticRetentionCount": retained,
                "distinctCandidateHashes": len({
                    str(item["candidateHash"])
                    for item in seed_candidate_rows
                }),
                "noPostHocRepair": True,
                "sequentialSearchReceipt": sequential.get(
                    "sequentialSearchReceipt"
                ),
                "continuationProbeWallSeconds": float(
                    sequential.get("continuationProbeWallSeconds") or 0.0
                ),
                "stopReason": seed_stop,
                "wallSeconds": time.perf_counter() - seed_started,
                "cpuSeconds": time.process_time() - seed_cpu_started,
            }
            seed_result_recorded_monotonic = time.monotonic()
            seed_deadline_update = _seed_result_deadline_update(
                seed_deadline_monotonic=seed_deadline,
                result_recorded_monotonic=seed_result_recorded_monotonic,
            )
            if not bool(seed_deadline_update["seedDeadlineReleased"]):
                seed_stop = (
                    "global_wall_bound"
                    if seed_result_recorded_monotonic >= global_deadline
                    else "seed_wall_bound"
                )
                seed_summary["stopReason"] = seed_stop
            seed_rows.append(seed_summary)
            update_progress(
                phase="seed_result_recorded",
                completedSeeds=[
                    *list(progress_state.get("completedSeeds") or ()),
                    copy.deepcopy(seed_summary),
                ],
                **seed_deadline_update,
            )
            if seed_stop in {"global_wall_bound", "seed_wall_bound"}:
                stop_reason = seed_stop
                fatal_failure = "campaign_time_bound_exceeded"
                break
            if qualified_row is not None or fatal_failure is not None:
                break
    except Exception as exc:
        failure = failure_evidence(stage, exc)
        stop_reason = (
            "campaign_execution_failure"
            if stage == "campaign_execution"
            else "campaign_prelaunch_failure"
        )
        fatal_failure = stop_reason
    finally:
        if container is not None:
            try:
                update_progress(phase="container_shutdown")
            except OSError:
                pass
            try:
                container.shutdown()
            except Exception as exc:
                cleanup_failure = failure_evidence("container_shutdown", exc)
        if temporary_runtime is not None:
            try:
                update_progress(phase="temporary_runtime_cleanup")
            except OSError:
                pass
            try:
                temporary_runtime.cleanup()
            except Exception as exc:
                cleanup_failure = failure_evidence("temporary_runtime_cleanup", exc)
        elif supervised_runtime is not None:
            try:
                update_progress(phase="supervised_runtime_release")
            except OSError:
                pass

    temporary_runtime_cleaned = runtime is None or not runtime.exists()
    if (
        not temporary_runtime_cleaned
        and cleanup_failure is None
        and supervised_runtime is None
    ):
        cleanup_failure = {
            "stage": "temporary_runtime_cleanup",
            "errorType": "RuntimeError",
            "message": "temporary Research runtime was not removed",
        }
    if cleanup_failure is not None:
        stop_reason = "campaign_cleanup_failure"
        fatal_failure = stop_reason
        if failure is None:
            failure = cleanup_failure

    if source_start is not None:
        try:
            source_end = _source_fingerprint(root)
        except Exception as exc:
            source_failure = failure_evidence("source_fingerprint_end", exc)
            if failure is None:
                failure = source_failure
            stop_reason = "source_fingerprint_failure"
            fatal_failure = stop_reason

    source_frozen: bool | None = (
        source_start == source_end
        if source_start is not None and source_end is not None
        else None
    )
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    strict_wall_bound_passed = bool(
        maximum_wall_seconds is not None
        and wall_seconds <= maximum_wall_seconds
    )
    all_seed_bounds_passed = bool(
        maximum_seed_seconds is not None
        and all(
            float(item["wallSeconds"]) <= maximum_seed_seconds
            for item in seed_rows
        )
    )
    boundary_failure = _campaign_invalid_reason(
        source_frozen=source_frozen is not False,
        strict_wall_bound_passed=(
            strict_wall_bound_passed if maximum_wall_seconds is not None else True
        ),
        all_seed_bounds_passed=(
            all_seed_bounds_passed if maximum_seed_seconds is not None else True
        ),
    )
    if boundary_failure is not None and fatal_failure is None:
        stop_reason = boundary_failure
        fatal_failure = boundary_failure

    genuinely_qualified = bool(
        qualified_row is not None
        and repeat is not None
        and repeat.get("deterministic")
        and source_frozen is True
        and fatal_failure is None
        and failure is None
        and cleanup_failure is None
        and boundary_failure is None
    )
    invalid = bool(
        fatal_failure is not None
        or failure is not None
        or cleanup_failure is not None
        or boundary_failure is not None
    )
    status = "invalid" if invalid else (
        "qualified" if genuinely_qualified else "negative"
    )
    stable = {
        "schemaVersion": FINALIST_CAMPAIGN_EVIDENCE_SCHEMA_VERSION,
        "experimentId": FINALIST_CAMPAIGN_EXPERIMENT_ID,
        "qualificationProfile": {
            "id": FINALIST_CAMPAIGN_PROFILE_ID,
            "revision": FINALIST_CAMPAIGN_PROFILE_REVISION,
            "globallyEnabled": False,
        },
        "status": status,
        "stopReason": stop_reason,
        "fatalFailure": fatal_failure,
        "failure": failure,
        "cleanupFailure": cleanup_failure,
        "temporaryRuntimeCleaned": temporary_runtime_cleaned,
        "bounds": {
            "seedStart": int(args.seed_start),
            "seedCount": attempts,
            "maximumCandidatesPerSeed": candidates_per_seed,
            "maximumWallSeconds": maximum_wall_seconds,
            "maximumSeedSeconds": maximum_seed_seconds,
        },
        "target": {"contract": "sun", "tactic": "only_winning_move"},
        "controlPlane": context_identity,
        "sourceFingerprintStart": source_start,
        "sourceFingerprintEnd": source_end,
        "sourceFrozen": source_frozen,
        "seedResults": seed_rows,
        "candidateResults": rows,
        "qualifiedCandidateStableHash": (
            qualified_row.get("stableHash") if qualified_row is not None else None
        ),
        "terminalRepeat": repeat,
        "counts": {
            "seedsEvaluated": len(seed_rows),
            "candidatesEvaluated": len(rows),
            "tacticRetained": sum(bool(item.get("tacticRetained")) for item in rows),
            "qualified": int(genuinely_qualified),
            "semanticDefects": sum(
                bool(item.get("semanticMismatches"))
                or item.get("status") == "semantic_defect"
                for item in rows
            ),
            "observerLeakDefects": sum(
                not bool(
                    dict(item.get("policyEvidence") or {}).get(
                        "zeroObserverLeaks", True
                    )
                )
                for item in rows
            ),
            "distinctCandidateHashes": len({
                str(item["candidateHash"]) for item in rows
            }),
            "distinctTraceHashes": len({
                str(item["traceHash"])
                for item in rows if item.get("traceHash")
            }),
        },
        "cost": {
            "wallSeconds": wall_seconds,
            "cpuSeconds": cpu_seconds,
            "strictWallBoundPassed": strict_wall_bound_passed,
            "allSeedBoundsPassed": all_seed_bounds_passed,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ortools": ortools.__version__,
            "logicalCpuCount": os.cpu_count(),
            "exactSearchBackend": backend_runtime_identity,
        },
        "supervision": (
            {
                "schemaVersion": 2,
                "kind": FINALIST_CAMPAIGN_WORKER_KIND,
                "launchId": supervised_runtime[2],
                "runtimeOwner": "supervisor_process",
                "parentRuntimeFinalized": False,
            }
            if supervised_runtime is not None
            else None
        ),
        "productSourceChangedByRunner": False,
        "gatesWeakened": False,
        "provisionalTruthRelabeled": False,
    }
    result = {**stable, "sha256": canonical_sha256(stable)}
    _publish_json_once(output, result)
    print(json.dumps({
        "status": result["status"],
        "stopReason": result["stopReason"],
        "counts": result["counts"],
        "cost": result["cost"],
        "sha256": result["sha256"],
        "output": str(output),
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return _campaign_exit_code(str(result["status"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=202608080001)
    parser.add_argument(
        "--analysis-mode",
        choices=("screening", "cascaded", "deep"),
        default="cascaded",
    )
    parser.add_argument("--candidates-per-seed", type=int, default=1)
    parser.add_argument(
        "--experimental-oracle-remaining-cards",
        type=int,
        default=12,
        help=(
            "Verification-only tactical horizon override for a bounded pilot; "
            "it does not change the governed production profile."
        ),
    )
    parser.add_argument(
        "--tactic",
        choices=(
            "only_winning_move",
            "small_card_big_gain",
            "large_card_sacrifice",
            "kabout_creation",
            "kabout_prevention",
        ),
        default="small_card_big_gain",
    )
    parser.add_argument(
        "--tactic-mode",
        choices=("preference", "hard", "qualification_gate"),
        default="preference",
    )
    parser.add_argument(
        "--tactical-viability-pre-screen",
        action="store_true",
        help=(
            "Reject only when an exact bounded scan proves the required "
            "tactic absent; never qualifies a result."
        ),
    )
    parser.add_argument(
        "--tactical-viability-maximum-decisions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tactical-deal-balance",
        action="store_true",
        help=(
            "Apply governed high/low-card balance constraints for an "
            "only-winning-move construction pilot."
        ),
    )
    parser.add_argument(
        "--strategic-minimum-played-cards",
        type=int,
        default=24,
        choices=(20, 21, 22, 23, 24, 25, 26, 27, 28),
    )
    parser.add_argument(
        "--only-variant",
        choices=(
            "actor_safe_baseline",
            "cp_sat_auction",
            "cp_sat_auction_and_late_play",
        ),
    )
    parser.add_argument(
        "--qualification-profile",
        choices=(
            FINALIST_CAMPAIGN_PROFILE_ID,
            LEGACY_HYBRID_FINALIST_PROFILE_ID,
        ),
        help=(
            "Run the isolated V3 finalist-agent auction-and-play qualification "
            "campaign; the historical hybrid profile is rejected explicitly."
        ),
    )
    parser.add_argument("--bundle-version", default="21.1.22")
    parser.add_argument("--maximum-wall-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-seed-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--_finalist-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_supervised-runtime",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_supervision-progress",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_supervision-launch-id",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args._finalist_worker:
        args._supervision_token = os.environ.pop(_FINALIST_SUPERVISION_ENV, None)
    if args.qualification_profile is not None:
        _require_qualification_profile(str(args.qualification_profile))
        return _run_finalist_campaign_supervised(args)
    if not 1 <= args.attempts <= 100:
        raise ValueError("attempts must be between 1 and 100")
    if not 1 <= args.candidates_per_seed <= 20:
        raise ValueError("candidates per seed must be between 1 and 20")
    if not 1 <= args.experimental_oracle_remaining_cards <= 20:
        raise ValueError("experimental oracle horizon must be between 1 and 20 cards")
    if not 1 <= args.tactical_viability_maximum_decisions <= 16:
        raise ValueError("tactical viability decision bound must be between 1 and 16")
    if args.tactical_viability_pre_screen and args.tactic_mode == "preference":
        raise ValueError(
            "tactical viability pre-screen requires a hard or qualification-gate tactic"
        )
    if args.tactical_deal_balance and not (
        args.tactic == "only_winning_move"
        and args.tactic_mode in {"hard", "qualification_gate"}
    ):
        raise ValueError(
            "tactical deal balance requires a hard or gated only-winning-move tactic"
        )

    parser.error(
        "the legacy V2 comparison pilot is retired; use "
        "scripts/benchmark_research_scenario_v4.py for new Research authoring"
    )


if __name__ == "__main__":
    raise SystemExit(main())
