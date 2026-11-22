from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.scenario_backend import (
    LEGACY_SCENARIO_RESEARCH_BACKEND,
    ORTOOLS_SCENARIO_RESEARCH_BACKEND,
    ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
    OrToolsScenarioResearchBackendV2,
    ScenarioResearchAuthorityV1,
)
from baloot_platform.research.service import ResearchFoundryServiceV1


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
RESEARCH = SOURCE / "baloot_platform" / "research"

_NEW_RESEARCH_DISPATCHER_ENTRYPOINTS = (
    "baloot_platform.research.scenario_backend",
    "baloot_platform.research.campaign_coordinator",
)

_NEW_RESEARCH_ENTRYPOINTS = (
    *(
        f"baloot_platform.research.{path.stem}"
        for path in sorted(RESEARCH.glob("ortools_*.py"))
    ),
    "baloot_platform.research.scenario_predicate_lowering_v2",
    *_NEW_RESEARCH_DISPATCHER_ENTRYPOINTS,
)

_FORBIDDEN_NEW_RESEARCH_IMPORTS = frozenset(
    {
        "baloot_platform.core.engine",
        "baloot_platform.core.bots",
        "baloot_platform.research.best_response_proof",
        "baloot_platform.research.exact_state",
        "baloot_platform.research.guided_construction_pilot",
        "baloot_platform.research.policy_proof",
        "baloot_platform.research.policy_proof_campaign",
        "baloot_platform.research.proof_service",
        "baloot_platform.research.scenario_evidence",
        "baloot_platform.research.scenario_search",
        "baloot_platform.research.sequential_candidate_search",
        "baloot_platform.research.strategic_auction",
        "baloot_platform.research.strategic_play",
    }
)

# Runtime compatibility is a post-qualification admission gate.  This lazy edge
# is deliberately outside the native Research dependency closure.
_DOWNSTREAM_ONLY_IMPORT_EDGES = frozenset(
    {
        (
            "baloot_platform.research.result_authority",
            "baloot_platform.research.runtime_compatibility",
        )
    }
)


_LEGACY_SUPPORT_OWNERS = (
    (
        "exact_manually_defined_legal_scenario",
        "tests/test_research_scenario_v4_proof.py::test_exact_trace_prefix_replays_and_proves_every_root_action",
    ),
    (
        "deterministically_generated_scenario",
        "tests/test_research_scenario_v4.py::test_search_v4_constructs_replays_and_proves_one_actor_safe_candidate",
    ),
    (
        "exact_complete_state_proof",
        "tests/test_research_scenario_v4_proof.py::test_golden_sun_hundred_project_has_one_saving_move",
    ),
    (
        "exact_actor_information_set_proof",
        "tests/test_research_proof_governance_v4.py::test_actor_information_set_exact_uses_one_strategy_across_worlds",
    ),
    (
        "bounded_proof",
        "tests/test_research_scenario_v4_proof.py::test_bounded_and_policy_conditioned_proofs_keep_their_labels",
    ),
    (
        "policy_conditioned_proof",
        "tests/test_research_scenario_v4_proof.py::test_bounded_and_policy_conditioned_proofs_keep_their_labels",
    ),
    (
        "zero_winning_root_actions",
        "tests/test_research_scenario_v4_proof.py::test_golden_sun_hundred_project_has_exactly_no_salvage",
    ),
    (
        "one_winning_root_action",
        "tests/test_research_scenario_v4_proof.py::test_golden_sun_hundred_project_has_one_saving_move",
    ),
    (
        "multiple_winning_root_actions",
        "tests/test_research_scenario_v4_proof.py::test_policy_continuation_count_is_for_selected_root_not_all_roots",
    ),
    (
        "exact_continuation_line_count",
        "tests/test_research_scenario_v4_proof.py::test_golden_ace_sacrifice_is_only_win_and_final_nine_takes_two_aces",
    ),
    (
        "incomplete_continuation_line_bounds",
        "tests/test_research_scenario_v4_proof.py::test_exact_proof_budget_exhaustion_returns_only_incomplete_bounds",
    ),
    (
        "contradictory_scenario_rejection",
        "tests/test_research_scenario_v4.py::test_v4_compiler_reports_card_ownership_contradiction_by_predicate_id",
    ),
    (
        "unsupported_predicate_rejection",
        "tests/test_research_scenario_v4.py::test_v4_compiler_fails_unsupported_non_play_proof_before_execution",
    ),
    (
        "exact_structural_infeasibility",
        "tests/test_research_scenario_v4_proof.py::test_structural_solver_proves_feasible_deal_and_exact_infeasible_core",
    ),
    (
        "not_found_within_budget",
        "tests/test_research_scenario_v4.py::test_v4_search_run_preserves_misses_no_goods_and_stage_yields",
    ),
    (
        "semantic_postcondition_miss",
        "tests/test_research_scenario_v4_proof.py::test_search_candidate_skips_illegal_selected_action_without_aborting_budget",
    ),
    (
        "proof_incomplete",
        "tests/test_research_scenario_v4_proof.py::test_exact_proof_budget_exhaustion_returns_only_incomplete_bounds",
    ),
    (
        "proof_false",
        "tests/test_research_scenario_v4_proof.py::test_bounded_and_policy_conditioned_proofs_keep_their_labels",
    ),
    (
        "qualified_scenario_through_foundry",
        "tests/test_research_scenario_v4_api_exact.py::test_v4_exact_public_api_journey_retains_and_queries_proof",
    ),
    (
        "qualified_result_to_video_preparation",
        "tests/test_research_scenario_v4.py::test_exact_v4_play_result_prepares_observer_safe_governed_render",
    ),
    (
        "rejected_or_incomplete_result_is_labeled_audit_media",
        "tests/test_two_ace_flagship_video.py::test_rejected_v4_result_prepares_as_badged_non_qr_audit_media",
    ),
    (
        "no_hidden_information_leak",
        "tests/test_research_v4_observer_safety.py::test_actor_public_result_masks_all_full_state_identifiers",
    ),
    (
        "no_qualification_gate_relaxation",
        "tests/test_research_qualification_consistency.py::test_preference_miss_does_not_reject_but_qualification_gate_miss_does",
    ),
    (
        "no_false_optimality_claim",
        "tests/test_research_scenario_v4_proof.py::test_bounded_and_policy_conditioned_proofs_keep_their_labels",
    ),
)

_NEW_ACCEPTANCE_OWNERS = {
    "V4-ACC-01": [
        "tests/test_ortools_exact_state.py::test_native_cp_exact_derives_flagship_without_python_semantic_replay"
    ],
    "V4-ACC-02": [
        "tests/test_research_ortools_backend_service.py::test_search_v4_run_lifts_and_persists_one_qualified_exact_state"
    ],
    "V4-ACC-03": [
        "tests/test_research_ortools_backend_service.py::test_exact_v4_run_persists_one_owned_qualified_result"
    ],
    "V4-ACC-04": [
        "tests/test_research_ortools_backend_service.py::test_exact_actor_information_set_run_persists_exhaustive_safe_proof"
    ],
    "V4-ACC-05": [
        "tests/test_research_ortools_complete_state_graph.py::test_bounded_frontier_uses_governed_raw_margin",
        "tests/test_research_ortools_backend_service.py::test_exact_bounded_play_run_persists_honest_boundary",
    ],
    "V4-ACC-06": [
        "tests/test_research_ortools_backend_service.py::test_policy_conditioned_v4_run_materializes_governed_profiles"
    ],
    "V4-ACC-07": [
        "tests/test_research_ortools_complete_state_graph.py::test_exact_graph_drives_zero_one_and_multiple_root_counts",
        "tests/test_research_ortools_backend_service.py::test_native_root_count_persists_zero_one_and_multiple_actions",
    ],
    "V4-ACC-08": [
        "tests/test_research_ortools_complete_state_graph.py::test_exact_graph_drives_zero_one_and_multiple_root_counts",
        "tests/test_research_ortools_backend_service.py::test_native_root_count_persists_zero_one_and_multiple_actions",
    ],
    "V4-ACC-09": [
        "tests/test_research_ortools_complete_state_graph.py::test_exact_graph_drives_zero_one_and_multiple_root_counts",
        "tests/test_research_ortools_backend_service.py::test_native_root_count_persists_zero_one_and_multiple_actions",
    ],
    "V4-ACC-10": [
        "tests/test_research_ortools_backend_service.py::test_exact_v4_run_persists_one_owned_qualified_result"
    ],
    "V4-ACC-11": [
        "tests/test_research_ortools_backend_service.py::test_exact_v4_line_count_exhaustion_retains_safe_audit_bounds",
        "tests/test_research_ortools_proof.py::test_limb_tier_refusal_preserves_safe_line_bounds",
    ],
    "V4-ACC-12": [
        "tests/test_research_ortools_unified_executor.py::test_search_construction_reports_contradictory_constraints_infeasible",
        "tests/test_research_ortools_backend_service.py::test_native_unsatisfied_search_persists_honest_terminal_outcome",
    ],
    "V4-ACC-13": [
        "tests/test_research_ortools_backend_service.py::test_causal_api_admission_refuses_before_normalization_or_search"
    ],
    "V4-ACC-14": [
        "tests/test_ortools_scenario_model.py::test_exact_nonterminal_auction_prefix_is_infeasible"
    ],
    "V4-ACC-15": [
        "tests/test_research_ortools_backend_service.py::test_native_unsatisfied_search_persists_honest_terminal_outcome",
        "tests/test_research_ortools_backend_service.py::test_search_v4_terminal_outcomes_persist_without_qualified_results",
        "tests/test_research_ortools_unified_executor.py::test_bid_bounded_proof_uses_the_declared_frontier",
    ],
    "V4-ACC-16": [
        "tests/test_research_ortools_backend_service.py::test_search_v4_terminal_outcomes_persist_without_qualified_results",
        "tests/test_research_ortools_backend_service.py::test_native_nonplay_state_rejects_false_fact",
        "tests/test_research_ortools_unified_executor.py::test_bid_bounded_proof_uses_the_declared_frontier",
        "tests/test_research_ortools_unified_executor.py::test_optional_project_exact_bounded_proof_uses_pass_and_declare_root",
    ],
    "V4-ACC-17": [
        "tests/test_research_ortools_backend_service.py::test_policy_conditioned_incomplete_proof_preserves_boundary",
        "tests/test_research_ortools_unified_executor.py::test_optional_project_continuation_count_reports_depth_exhaustion",
        "tests/test_research_ortools_unified_executor.py::test_native_complete_state_solver_failures_preserve_outcome_boundary",
    ],
    "V4-ACC-18": [
        "tests/test_research_ortools_unified_executor.py::test_native_exact_proof_cannot_bypass_hard_qualification"
    ],
    "V4-ACC-19": [
        "src/baloot_platform/service/foundry_dashboard/e2e/foundry_ortools_operator.spec.ts::the disclosed target-wins-trick-6 alternative finds five exact variants",
        "tests/test_research_ortools_backend_service.py::test_search_v4_full_ui_selector_campaign_persists_five_distinct_qualified_results",
    ],
    "V4-ACC-20": [
        "tests/test_research_ortools_backend_service.py::test_exact_v4_run_persists_one_owned_qualified_result"
    ],
    "V4-ACC-21": [
        "tests/test_research_ortools_backend_service.py::test_native_proof_failed_result_prepares_badged_non_publishable_audit_media"
    ],
    "V4-ACC-22": [
        "tests/test_research_ortools_backend_service.py::test_exact_actor_api_uses_frozen_profile_for_admission",
        "tests/test_research_ortools_information_set_proof.py::test_hidden_world_nodes_use_real_canonical_observer_documents"
    ],
    "V4-ACC-23": [
        "tests/test_research_ortools_unified_executor.py::test_native_exact_proof_cannot_bypass_hard_qualification",
        "tests/test_research_ortools_backend_service.py::test_native_false_root_count_persists_rejected_exact_proof",
    ],
    "V4-ACC-24": [
        "tests/test_research_ortools_proof.py::test_infeasible_is_model_error_and_feasible_is_not_exact",
        "tests/test_research_ortools_unified_executor.py::test_native_complete_state_solver_failures_preserve_outcome_boundary",
    ],
}

_PENDING_ACCEPTANCE_BLOCKERS: dict[str, str] = {}

MANDATORY_RESEARCH_V4_ACCEPTANCE_V1 = {
    "schemaVersion": 1,
    "entries": [
        {
            "id": f"V4-ACC-{index:02d}",
            "scenario": scenario,
            "backendAuthority": ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
            "evidenceKind": "cp_sat_acceptance",
            "status": (
                "covered"
                if f"V4-ACC-{index:02d}" in _NEW_ACCEPTANCE_OWNERS
                else "pending"
            ),
            "owners": _NEW_ACCEPTANCE_OWNERS.get(f"V4-ACC-{index:02d}", []),
            "blocker": _PENDING_ACCEPTANCE_BLOCKERS.get(f"V4-ACC-{index:02d}"),
        }
        for index, (scenario, _legacy_owner) in enumerate(
            _LEGACY_SUPPORT_OWNERS, start=1
        )
    ],
    "legacyCompatibilityEvidence": [
        {
            "acceptanceId": f"V4-ACC-{index:02d}",
            "backendAuthority": LEGACY_SCENARIO_RESEARCH_BACKEND,
            "evidenceKind": "legacy_compatibility",
            "owners": [owner],
        }
        for index, (_scenario, owner) in enumerate(_LEGACY_SUPPORT_OWNERS, start=1)
    ],
}


def _module_path(module: str) -> Path | None:
    candidate = SOURCE.joinpath(*module.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = SOURCE.joinpath(*module.split("."), "__init__.py")
    return package if package.is_file() else None


def _is_type_checking_guard(node: ast.If) -> bool:
    return (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
        isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
    )


def _local_imports(module: str, *, include_nested: bool) -> set[str]:
    path = _module_path(module)
    assert path is not None, module
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    importlib_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "importlib"
    }
    import_module_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "importlib"
        for alias in node.names
        if alias.name == "import_module"
    }
    for node in ast.walk(tree):
        ancestors: list[ast.AST] = []
        parent = parents.get(node)
        while parent is not None:
            ancestors.append(parent)
            parent = parents.get(parent)
        if any(
            isinstance(item, ast.If) and _is_type_checking_guard(item)
            for item in ancestors
        ):
            continue
        if not include_nested and any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for item in ancestors
        ):
            continue
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if node.level:
                imported = importlib.util.resolve_name(
                    "." * node.level + imported, package
                )
            if imported:
                imports.add(imported)
            for alias in node.names:
                candidate = f"{imported}.{alias.name}" if imported else alias.name
                if _module_path(candidate) is not None:
                    imports.add(candidate)
        elif isinstance(node, ast.Call) and node.args:
            target = node.args[0]
            if not isinstance(target, ast.Constant) or not isinstance(
                target.value, str
            ):
                continue
            lazy_module = target.value
            is_lazy_import = (
                isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", *import_module_aliases}
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
            )
            if is_lazy_import:
                imports.add(
                    importlib.util.resolve_name(lazy_module, package)
                    if lazy_module.startswith(".")
                    else lazy_module
                )
    return imports


def _is_forbidden(module: str) -> bool:
    return (
        module in _FORBIDDEN_NEW_RESEARCH_IMPORTS
        or module.startswith("baloot_platform.agents.")
        or module == "baloot_platform.agents"
        or any(
            part.endswith("bot") or part.endswith("bots") for part in module.split(".")
        )
    )


def _forbidden_transitive_paths(
    entrypoints: tuple[str, ...] = _NEW_RESEARCH_ENTRYPOINTS,
) -> list[str]:
    queue = [(module, (module,)) for module in entrypoints]
    visited: set[str] = set()
    failures: list[str] = []
    while queue:
        module, chain = queue.pop(0)
        if module in visited:
            continue
        visited.add(module)
        for imported in sorted(
            _local_imports(
                module,
                include_nested=module.startswith("baloot_platform.research."),
            )
        ):
            imported_chain = (*chain, imported)
            if _is_forbidden(imported):
                failures.append(" -> ".join(imported_chain))
                continue
            if (module, imported) in _DOWNSTREAM_ONLY_IMPORT_EDGES:
                continue
            if imported.startswith("baloot_platform.") and _module_path(imported):
                queue.append((imported, imported_chain))
    return failures


def _forbidden_modules_loaded_by_new_entrypoints() -> list[str]:
    script = f"""
import importlib
import json
import sys

for module in {tuple(_NEW_RESEARCH_ENTRYPOINTS)!r}:
    importlib.import_module(module)
forbidden = {tuple(sorted(_FORBIDDEN_NEW_RESEARCH_IMPORTS))!r}
loaded = sorted(
    module
    for module in sys.modules
    if module in forbidden
    or module == "baloot_platform.agents"
    or module.startswith("baloot_platform.agents.")
    or any(part.endswith("bot") or part.endswith("bots") for part in module.split("."))
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def _ortools_profile() -> dict:
    return {
        "scenarioResearchV4": {
            "schemaVersion": 1,
            "backend": ORTOOLS_SCENARIO_RESEARCH_BACKEND,
            "planner": "tail_first_v1",
            "deterministicExecution": True,
            "policyCatalog": {
                "schemaVersion": 1,
                "defaultPolicyId": "fixed_action_map_v1",
                "profiles": {
                    "fixed_action_map_v1": {
                        "version": "1.0.0",
                        "materialization": "seeded_legal_action_order_v1",
                    }
                },
            },
            "proofBudgets": {"schemaVersion": 1},
            "searchBudgets": {"schemaVersion": 1},
            "diversityDefaults": {"schemaVersion": 1},
            "cacheLimits": {"schemaVersion": 1},
            "refusalThresholds": {"schemaVersion": 1},
        }
    }


def _context(challenge: dict, rules: RuleBook) -> SimpleNamespace:
    class Bundle:
        sha256 = "a" * 64

        def __init__(self) -> None:
            self.rules = rules

        @staticmethod
        def component(name: str) -> dict:
            assert name == "challenge"
            return challenge

    return SimpleNamespace(
        bundle=Bundle(),
        bundle_revision_id="CBR-AUTHORITY-TEST",
        effective_runtime_config_hash="b" * 64,
    )


def _test_functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_decorator_name(node.value)}.{node.attr}".strip(".")
    return node.id if isinstance(node, ast.Name) else ""


def _owner_node(node_id: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    relative, separator, test_name = node_id.partition("::")
    if not separator or not test_name.startswith("test_"):
        return None
    path = ROOT / relative
    return _test_functions(path).get(test_name) if path.is_file() else None


def _playwright_owner_source(node_id: str) -> str | None:
    relative, separator, test_name = node_id.partition("::")
    path = ROOT / relative
    if not separator or not path.name.endswith(".spec.ts") or not path.is_file():
        return None
    source = path.read_text(encoding="utf-8")
    if not any(
        marker in source for marker in (f"test('{test_name}'", f'test("{test_name}"')
    ):
        return None
    return source


def _owner_imports_ortools_module(node_id: str) -> bool:
    relative, separator, _test_name = node_id.partition("::")
    path = ROOT / relative
    if not separator or not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
    return any(
        isinstance(item, ast.ImportFrom)
        and item.module is not None
        and item.module.startswith("baloot_platform.research.ortools_")
        for item in tree.body
    )


def _asserted_string_literals(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    return {
        item.value
        for assertion in (
            item for item in ast.walk(node) if isinstance(item, ast.Assert)
        )
        for item in ast.walk(assertion.test)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _acceptance_audit_report() -> dict:
    """Report owner coverage, not execution or release acceptance."""
    matrix = json.loads(json.dumps(MANDATORY_RESEARCH_V4_ACCEPTANCE_V1, sort_keys=True))
    pending = [
        entry["id"] for entry in matrix["entries"] if entry["status"] == "pending"
    ]
    core = {
        "schemaVersion": 1,
        "backendAuthority": ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
        "evidenceKind": "mandatory_acceptance_owner_inventory",
        "acceptanceExecutionVerified": False,
        "status": "passed" if not pending else "expected_incomplete",
        "coveredCount": 24 - len(pending),
        "requiredCount": 24,
        "pendingAcceptanceIds": pending,
        "pendingBlockers": {
            entry["id"]: entry["blocker"]
            for entry in matrix["entries"]
            if entry["status"] == "pending"
        },
    }
    canonical = json.dumps(
        core, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return {**core, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def test_new_ortools_research_authority_has_no_forbidden_transitive_imports() -> None:
    assert len(_NEW_RESEARCH_ENTRYPOINTS) >= 4
    assert set(_NEW_RESEARCH_DISPATCHER_ENTRYPOINTS).issubset(_NEW_RESEARCH_ENTRYPOINTS)
    assert _forbidden_transitive_paths() == []
    assert _forbidden_modules_loaded_by_new_entrypoints() == []


def test_transitive_guard_rejects_nested_and_lazy_forbidden_imports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases = {
        "nested_engine": (
            "def hidden():\n    from baloot_platform.core.engine import BalootEngine\n",
            "baloot_platform.core.engine",
        ),
        "nested_bot": (
            "def hidden():\n    from baloot_platform.core.bots import HeuristicBot\n",
            "baloot_platform.core.bots",
        ),
        "lazy_proof": (
            "def hidden():\n"
            "    import importlib as loader\n"
            "    loader.import_module('baloot_platform.research.proof_service')\n",
            "baloot_platform.research.proof_service",
        ),
        "lazy_verifier": (
            "def hidden():\n"
            "    from importlib import import_module as load\n"
            "    load('baloot_platform.research.scenario_evidence')\n",
            "baloot_platform.research.scenario_evidence",
        ),
    }
    for case, (hidden_source, forbidden) in cases.items():
        source_root = tmp_path / case
        package = source_root / "baloot_platform" / "research"
        package.mkdir(parents=True)
        (package / "entrypoint.py").write_text(
            "from . import nested\n", encoding="utf-8"
        )
        (package / "nested.py").write_text(hidden_source, encoding="utf-8")
        monkeypatch.setattr(sys.modules[__name__], "SOURCE", source_root)

        failures = _forbidden_transitive_paths(("baloot_platform.research.entrypoint",))

        assert any(forbidden in failure for failure in failures), (case, failures)


def test_lazy_research_package_preserves_legacy_public_exports() -> None:
    script = """
from baloot_platform.research import (
    ScenarioRuntimeVerifierV1,
    ScenarioSearchExecutorV1,
    UnifiedScenarioProofServiceV1,
)
assert ScenarioRuntimeVerifierV1.__name__ == "ScenarioRuntimeVerifierV1"
assert ScenarioSearchExecutorV1.__name__ == "ScenarioSearchExecutorV1"
assert UnifiedScenarioProofServiceV1.__name__ == "UnifiedScenarioProofServiceV1"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )


def test_bundle_dispatch_keeps_legacy_and_ortools_authorities_disjoint(
    rules: RuleBook,
) -> None:
    legacy_authority = ScenarioResearchAuthorityV1.from_challenge_profile({})
    ortools_authority = ScenarioResearchAuthorityV1.from_challenge_profile(
        _ortools_profile()
    )
    assert legacy_authority.backend == LEGACY_SCENARIO_RESEARCH_BACKEND
    assert ortools_authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND
    assert (
        ResearchFoundryServiceV1._ortools_scenario_backend_v2(_context({}, rules))
        is None
    )
    assert isinstance(
        ResearchFoundryServiceV1._ortools_scenario_backend_v2(
            _context(_ortools_profile(), rules)
        ),
        OrToolsScenarioResearchBackendV2,
    )
    for module in (
        "baloot_platform.research.exact_state",
        "baloot_platform.research.proof_service",
        "baloot_platform.research.scenario_search",
        "baloot_platform.research.scenario_evidence",
    ):
        assert importlib.import_module(module) is not None


def test_v3_bundle_without_frozen_component_inventory_fails_closed(
    rules: RuleBook,
) -> None:
    profile = _ortools_profile()
    profile["scenarioResearchV4"]["backend"] = ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3

    with pytest.raises(ValueError, match="explicit frozen adjudication component"):
        ResearchFoundryServiceV1._ortools_scenario_backend_v2(
            _context(profile, rules)
        )


def test_mandatory_v4_acceptance_matrix_separates_new_and_legacy_evidence() -> None:
    matrix = json.loads(json.dumps(MANDATORY_RESEARCH_V4_ACCEPTANCE_V1, sort_keys=True))
    entries = matrix["entries"]
    assert matrix["schemaVersion"] == 1
    assert [entry["id"] for entry in entries] == [
        f"V4-ACC-{index:02d}" for index in range(1, 25)
    ]
    assert len({entry["scenario"] for entry in entries}) == 24
    assert len(matrix["legacyCompatibilityEvidence"]) == 24

    missing: list[str] = []
    skipped: list[str] = []
    wrong_authority: list[str] = []
    for entry in entries:
        assert entry["backendAuthority"] == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
        assert entry["evidenceKind"] == "cp_sat_acceptance"
        assert entry["status"] in {"covered", "pending"}
        assert bool(entry["owners"]) == (entry["status"] == "covered")
        assert bool(entry["blocker"]) == (entry["status"] == "pending")
        for node_id in entry["owners"]:
            node = _owner_node(node_id)
            if node is None:
                playwright_source = _playwright_owner_source(node_id)
                if playwright_source is None:
                    missing.append(node_id)
                elif "test.skip" in playwright_source:
                    skipped.append(node_id)
                elif not any(
                    marker in playwright_source
                    for marker in ("OR-Tools", "ortools_cp_sat_v3", "cp_sat")
                ):
                    wrong_authority.append(node_id)
                continue
            decorator_names = {_decorator_name(item) for item in node.decorator_list}
            if any(name.endswith((".skip", ".skipif")) for name in decorator_names):
                skipped.append(node_id)
            asserted = _asserted_string_literals(node)
            if not (
                {ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3, "cp_sat"} & asserted
                or _owner_imports_ortools_module(node_id)
            ):
                wrong_authority.append(node_id)

    for support in matrix["legacyCompatibilityEvidence"]:
        assert support["backendAuthority"] == LEGACY_SCENARIO_RESEARCH_BACKEND
        assert support["evidenceKind"] == "legacy_compatibility"
        assert support["acceptanceId"] in {entry["id"] for entry in entries}
        for node_id in support["owners"]:
            if _owner_node(node_id) is None:
                missing.append(node_id)

    assert missing == []
    assert skipped == []
    assert wrong_authority == []


def test_mandatory_v4_acceptance_audit_is_complete() -> None:
    report = _acceptance_audit_report()
    assert report["evidenceKind"] == "mandatory_acceptance_owner_inventory"
    assert report["acceptanceExecutionVerified"] is False
    assert report["status"] == "passed", json.dumps(report, sort_keys=True)
    assert report["coveredCount"] == report["requiredCount"] == 24
    assert report["pendingAcceptanceIds"] == []
    assert report["pendingBlockers"] == {}
