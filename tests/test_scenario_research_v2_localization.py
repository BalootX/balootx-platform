from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from baloot_platform.config import validate_profile_payload

from baloot_platform.research.scenario_registry_v2 import (
    CardSelectorV1,
    play_action_matches_definition_v2,
    play_trick_card_count_definition_v2,
    play_trick_winner_matches_definition_v2,
    proof_winning_root_action_matches_definition_v2,
)


_AUTHORITY_COPY = frozenset(
    {
        "research_scenario_authority_title",
        "research_scenario_authority_help",
        "research_scenario_backend",
        "research_scenario_backend_help",
        "research_scenario_backend_ortools_cp_sat_v2",
        "research_scenario_planner",
        "research_scenario_planner_help",
        "research_scenario_planner_tail_first_v1",
        "research_scenario_deterministic_execution",
        "research_scenario_deterministic_execution_help",
        "research_scenario_proof_budgets",
        "research_scenario_proof_budgets_help",
        "research_scenario_search_budgets",
        "research_scenario_search_budgets_help",
    }
)
_PROOF_COPY = frozenset(
    {
        "research_scenario_proof_exact",
        "research_scenario_proof_exact_help",
        "research_scenario_proof_bounded",
        "research_scenario_proof_bounded_help",
        "research_scenario_proof_policy_conditioned",
        "research_scenario_proof_policy_conditioned_help",
        "research_scenario_issue_continuation_count_requires_selected_action",
        "research_scenario_issue_continuation_count_line_universe_mismatch",
    }
)
_COMPATIBILITY_COPY = frozenset(
    {
        "research_scenario_runtime_compatibility",
        "research_scenario_runtime_compatibility_help",
        "research_scenario_runtime_compatibility_not_checked",
        "research_scenario_runtime_compatibility_passed",
        "research_scenario_runtime_compatibility_failed",
    }
)
_TIMING_COPY = frozenset(
    {
        "research_scenario_timing_title",
        "research_scenario_timing_help",
        "research_scenario_timing_validation",
        "research_scenario_timing_tail_construction",
        "research_scenario_timing_tail_model_build",
        "research_scenario_timing_tail_model_configure",
        "research_scenario_timing_tail_solver",
        "research_scenario_timing_prefix_lift",
        "research_scenario_timing_construction",
        "research_scenario_timing_proof",
        "research_scenario_timing_qualification",
        "research_scenario_timing_compatibility",
        "research_scenario_timing_first_result",
        "research_scenario_timing_variant",
        "research_scenario_timing_total",
    }
)
_PREDICATE_HELP_COPY = frozenset(
    {
        "research_scenario_argument_card_selector_help",
        "research_scenario_predicate_play_action_matches_help",
        "research_scenario_predicate_play_trick_winner_matches_help",
        "research_scenario_predicate_play_trick_card_count_help",
        "research_scenario_predicate_proof_winning_root_action_matches_help",
    }
)
_ORTOOLS_AUTHORITY_OVERRIDES = frozenset(
    {
        "laffa_library_audit_only",
        "laffa_library_qualification_failures",
        "research_description",
        "research_scenario_help",
        "research_scenario_exact_builder_help",
        "research_scenario_action_timeline_help",
        "research_scenario_exact_trace_advanced_help",
        "research_scenario_catalog_help",
        "research_scenario_exact_state_derived",
        "research_issue_tactics_replay_warning",
        "research_truth_exact",
        "research_truth_robust",
        "research_truth_debatable",
        "research_estimate_tactical_verification",
    }
)


def _load(root: Path, name: str) -> dict[str, Any]:
    return yaml.safe_load(
        (root / "configs/product" / name).read_text(encoding="utf-8")
    )


def _workflow(profile: dict[str, Any]) -> dict[str, str]:
    return dict(profile["foundry_copy"]["groups"]["workflow"])


def _hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _catalog_localization_keys() -> set[str]:
    keys: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                if name in {"x-messageArKey", "x-messageEnKey"}:
                    keys.add(str(child))
                elif name in {"x-enumMessageArKeys", "x-enumMessageEnKeys"}:
                    keys.update(str(item) for item in dict(child).values())
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(CardSelectorV1.argument_schema())
    for factory in (
        play_action_matches_definition_v2,
        play_trick_winner_matches_definition_v2,
        play_trick_card_count_definition_v2,
        proof_winning_root_action_matches_definition_v2,
    ):
        definition = factory()
        keys.update((definition.message_ar_key, definition.message_en_key))
        visit(definition.argument_schema)
    return keys


def test_ortools_product_profile_preserves_legacy_source_without_stale_copy(
    root: Path,
) -> None:
    legacy = _load(root, "studio_v1.yaml")
    ortools = _load(root, "studio_ortools.yaml")
    legacy_copy = _workflow(legacy)
    ortools_copy = _workflow(ortools)

    assert legacy["profile"] == {
        "id": "studio_v1",
        "version": "1.1.21",
        "status": "production_candidate",
    }
    assert ortools["profile"] == {
        "id": "studio_v1",
        "version": "1.1.37",
        "status": "production_candidate",
    }
    shared = set(legacy_copy) & set(ortools_copy)
    assert all(
        key in _ORTOOLS_AUTHORITY_OVERRIDES
        or ortools_copy[key] == legacy_copy[key]
        for key in shared
    )
    missing_legacy_copy = set(legacy_copy) - set(ortools_copy)
    assert all(
        key.startswith(("research_", "compare_"))
        or key in {"nav_compare", "nav_compare_help", "batch_yield_per_cpu_hour"}
        for key in missing_legacy_copy
    )
    assert _ORTOOLS_AUTHORITY_OVERRIDES <= set(ortools_copy)
    assert all(
        ortools_copy[key] != legacy_copy[key] for key in _ORTOOLS_AUTHORITY_OVERRIDES
    )
    assert _hash(legacy) != _hash(ortools)

    normalized = copy.deepcopy(ortools)
    normalized["profile"] = copy.deepcopy(legacy["profile"])
    normalized["public_guide"] = copy.deepcopy(legacy["public_guide"])
    normalized["foundry_copy"]["groups"]["workflow"] = legacy_copy
    assert normalized["foundry_copy"]["groups"]["configuration_fields"].pop("label_bid_alternative_first") == "فحص بدائل الشراء قبل الخيار المختار — عند كشف كل الأوراق"
    native_only_controls = {
        "label_exact_tail_proof", "label_video_interestingness",
        "label_maximum_final_margin", "label_minimum_point_swing",
        "label_maximum_featured_trick", "label_minimum_meaningful_decisions",
    }
    fields = normalized["foundry_copy"]["groups"]["configuration_fields"]
    assert set(fields) - set(legacy["foundry_copy"]["groups"]["configuration_fields"]) == native_only_controls
    for key in native_only_controls:
        assert fields.pop(key).strip()
    # The frozen legacy profile predates the browser request deadline; the dashboard falls back for it.
    assert normalized["foundry_copy"]["groups"]["helpers"].pop("request_timeout") == "انتهت مهلة الطلب. حدّث الصفحة وتحقّق من النتيجة قبل إعادة المحاولة."
    assert normalized == legacy


@pytest.mark.parametrize("key,legacy_text,ortools_text", [
    (
        "laffa_library_audit_only",
        "هذه اللفة للبحث والتدقيق فقط، وفشل التأهيل يبقى محفوظًا معها.",
        "للمراجعة البحثية فقط؛ أهلية النشر غير مثبتة.",
    ),
    (
        "laffa_library_qualification_failures",
        "أسباب عدم التأهيل",
        "ملاحظات التأهيل والنشر",
    ),
])
def test_publication_copy_preserves_frozen_legacy_qualification_wording(
    root: Path, key: str, legacy_text: str, ortools_text: str,
) -> None:
    assert _workflow(_load(root, "studio_v1.yaml"))[key] == legacy_text
    assert _workflow(_load(root, "studio_ortools.yaml"))[key] == ortools_text


def test_product_copy_without_worker_control_remains_schema_valid(root: Path) -> None:
    product = copy.deepcopy(_load(root, "studio_ortools.yaml"))
    workflow = product["foundry_copy"]["groups"]["workflow"]
    for key in (
        "research_shard_worker_count",
        "research_shard_worker_count_en",
        "research_scenario_help_shard_worker_count",
        "research_scenario_help_shard_worker_count_en",
    ):
        del workflow[key]
    validate_profile_payload("product", product, root=root)


def test_manual_auction_opening_copy_is_bilingual_and_optional_for_frozen_profiles(root: Path) -> None:
    product = copy.deepcopy(_load(root, "studio_ortools.yaml"))
    workflow = product["foundry_copy"]["groups"]["workflow"]
    for key in ("add", "invalid"):
        for suffix in ("", "_en"):
            assert workflow.pop(f"research_auction_seeds_{key}{suffix}").strip()
    validate_profile_payload("product", product, root=root)


def test_native_estimate_copy_distinguishes_analysis_and_tactical_work(root: Path) -> None:
    workflow = _workflow(_load(root, "studio_ortools.yaml"))
    assert "التحليل" in workflow["research_estimate_analysis"]
    assert "التكتيك" in workflow["research_estimate_tactical_verification"]
    assert "الإثبات العميق" not in workflow["research_estimate_tactical_verification"]
    timeout_ar = workflow["research_scenario_help_solver_timeout_seconds"]
    timeout_en = workflow["research_scenario_help_solver_timeout_seconds_en"]
    assert "محاولة البناء كاملة" in timeout_ar
    assert "وقت ومحاولات" in timeout_ar
    assert "complete construction attempt" in timeout_en
    assert "both time and allowed attempts" in timeout_en
    assert "per solve" not in timeout_en


def test_published_product_copy_without_new_tail_timing_remains_valid(root: Path) -> None:
    product = copy.deepcopy(_load(root, "studio_ortools.yaml"))
    workflow = product["foundry_copy"]["groups"]["workflow"]
    for stage in ("tail_model_build", "tail_model_configure", "tail_solver"):
        del workflow[f"research_scenario_timing_{stage}"]
        del workflow[f"research_scenario_timing_{stage}_en"]
    validate_profile_payload("product", product, root=root)


def test_ortools_research_copy_is_bilingual_and_complete(root: Path) -> None:
    legacy_copy = _workflow(_load(root, "studio_v1.yaml"))
    copy = _workflow(_load(root, "studio_ortools.yaml"))
    added = set(copy) - set(legacy_copy)
    required = (
        _AUTHORITY_COPY
        | _PROOF_COPY
        | _COMPATIBILITY_COPY
        | _TIMING_COPY
        | _PREDICATE_HELP_COPY
        | _catalog_localization_keys()
    )

    assert required <= set(copy)
    bilingual = required | {
        f"{key}_en" for key in required if not key.endswith("_en")
    }
    assert bilingual <= set(copy)
    assert added
    for key in bilingual:
        assert copy[key].strip()


@pytest.mark.parametrize("owners,core_owners", [
    ([], None), (["PRED-CAUSAL"], None), (["PRED-CAUSAL"], ["PRED-CORE"]),
    (["PRED-CAUSAL"], []),
])
def test_preflight_diagnostic_uses_only_selected_bundle_copy(root: Path, owners, core_owners) -> None:
    from baloot_platform.research.service import ResearchFoundryServiceV1

    issue = {"code": "continuation_count_requires_selected_action", "path": "proof.selectedAction"}
    if owners:
        issue["predicateIds"] = owners
    if core_owners is not None:
        issue["structuralInfeasibility"] = {"namedCorePredicateIds": core_owners}
    old_copy = _workflow(_load(root, "studio_v1.yaml"))
    new_copy = _workflow(_load(root, "studio_ortools.yaml"))
    old = ResearchFoundryServiceV1._ortools_preflight_issue_v2(
        issue, recipe_hash="recipe", workflow_copy=old_copy
    )
    new = ResearchFoundryServiceV1._ortools_preflight_issue_v2(
        issue, recipe_hash="recipe", workflow_copy=new_copy
    )

    assert old["fieldPaths"] == new["fieldPaths"] == ["proof.selectedAction"]
    assert old["predicateIds"] == new["predicateIds"] == (
        owners if core_owners is None else core_owners)
    assert old["messageArKey"] == "research_validation_failed"
    assert new["messageArKey"] == "research_scenario_issue_continuation_count_requires_selected_action"
    assert old_copy[old["messageArKey"]]
    assert new_copy[new["messageArKey"]]


def test_proof_and_compatibility_help_preserve_claim_boundaries(root: Path) -> None:
    copy = _workflow(_load(root, "studio_ortools.yaml"))

    assert "الأفق المحدد فقط" in copy["research_scenario_proof_bounded_help"]
    assert "السياسات المجمدة" in copy[
        "research_scenario_proof_policy_conditioned_help"
    ]
    assert "ما يؤهلها ولا يرفض ادعاء البحث" in copy[
        "research_scenario_runtime_compatibility_help"
    ]
    assert "cannot qualify it, reject the Research claim" in copy[
        "research_scenario_runtime_compatibility_help_en"
    ]
    assert "OR-Tools CP-SAT يبني المسار القانوني" in copy[
        "research_scenario_help"
    ]
    assert "OR-Tools CP-SAT" in copy["research_scenario_exact_state_derived"]
    assert "المحرك وحده" not in copy["research_scenario_help"]
    assert "إعادة اللعب والتحليل" not in copy[
        "research_issue_tactics_replay_warning"
    ]
    assert "excluded from the semantic result hash" in copy[
        "research_scenario_timing_help_en"
    ]
