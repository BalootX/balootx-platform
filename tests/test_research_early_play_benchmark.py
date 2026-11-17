from pathlib import Path

import pytest

from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Contract

from scripts.benchmark_research_early_play import run_pair


def _measured_quality_profile(bundle):
    profile = bundle.component("challenge")["scenarioResearchV4"]
    seconds = 900.0  # Sulaiman p99 280.911s, 3x margin rounded to 60s.
    profile["qualityExecution"]["maximumWallSeconds"] = max(
        float(profile["qualityExecution"]["maximumWallSeconds"]), seconds,
    )
    profile["proofBudgets"]["profiles"]["bounded_v1"].update(
        maximumWallSeconds=seconds,
        maximumCpuSeconds=seconds,
    )
    assert profile["proofBudgets"]["profiles"]["bounded_v1"]["maximumWallSeconds"] >= seconds
    assert profile["proofBudgets"]["profiles"]["bounded_v1"]["maximumCpuSeconds"] >= seconds
    return profile


@pytest.mark.parametrize("seed,contract", [(41, Contract.SUN), (43, Contract.SUN),
                                            (51, Contract.HOKUM)])
def test_matched_early_play_benchmark_replays_and_scores_both_arms(seed, contract):
    rules = build_bundle(Path(__file__).resolve().parents[1]).rules
    result = run_pair(seed=seed, contract=contract, rules=rules)

    assert result["sourcePrefixHash"]
    assert result["candidatePolicy"] == "observer_safe_trick_control_v1"
    assert {arm["candidateTeam"] for arm in result["arms"]} == {
        "north_south", "west_east"}
    assert all(len(arm["playActions"]) == 32 for arm in result["arms"])
    assert result["pairedMargin"] == sum(arm["margin"] for arm in result["arms"]) / 2


def test_current_policy_benchmark_uses_explicit_source_candidate_and_complete_round():
    root = Path(__file__).resolve().parents[1]
    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    result = run_pair(
        seed=41, contract=Contract.SUN, rules=bundle.rules,
        candidate_policy="source_candidate_ordinary_nonfeatured",
        current_quality_profile=_measured_quality_profile(bundle),
    )

    assert result["profileScope"] == "source_candidate_ordinary_nonfeatured"
    assert result["rulesHash"] == bundle.rules.sha256
    assert all(arm["policyPlayCount"] == 16 for arm in result["arms"])
    assert all(len(arm["playActions"]) == 32 for arm in result["arms"])
    assert result["qualityExecutionHash"] and result["proofBudgetHash"]
    assert result["candidateAdvantage"] == sum(arm["margin"] for arm in result["arms"])
    assert "earlyTricks" not in result


def test_current_policy_forced_opening_keeps_other_actions_and_native_parity():
    root = Path(__file__).resolve().parents[1]
    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    result = run_pair(
        seed=41, contract=Contract.SUN, rules=bundle.rules,
        candidate_policy="source_candidate_ordinary_nonfeatured",
        current_quality_profile=_measured_quality_profile(bundle),
        opening_action="D8",
    )

    assert result["scope"] == "single_deal_causal_screen"
    assert result["openingAction"] == "D8"
    assert len(result["arms"]) == 1
    assert result["arms"][0]["playActions"][0]["card"] == "D8"
    assert result["arms"][0]["policyPlayCount"] == 16
    assert len(result["arms"][0]["playActions"]) == 32
    assert "candidateAdvantage" not in result


def test_full_round_actor_tail_diagnostic_preserves_proof_boundary(monkeypatch):
    from scripts.pilot_research_quality_yield import run_full_round
    from baloot_platform.research import ortools_research_executor

    def reject_current_deal_bounds(*args, **kwargs):
        pytest.fail("exact actor-visible tail cannot use current-deal policy bounds")

    monkeypatch.setattr(ortools_research_executor, "_observed_play_quality_artifact_v1",
                        reject_current_deal_bounds)

    root = Path(__file__).resolve().parents[1]
    result = run_full_round(root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json",
        root / "configs/rules/saudi_federation_2023_v2.yaml", seconds=300, exact_play_from_trick=7)
    assert result["summary"]["status"] == "complete"
    assert result["summary"]["tailCertified"] is True
    assert result["summary"]["runtimeCompatibility"] == "passed"
    assert result["summary"]["exactTailNonForcedDecisions"] >= 1
    assert result["readbackError"] is None
    assert result["sourcePrefix"]["auctionActions"] == []
    assert result["sourcePrefix"]["projectDeclarations"] == []
    receipt = result["receipt"]
    assert receipt["scope"] == "all_decisions"
    assert len(receipt["finalPrefix"]["playActions"]) == 32
    for stage in (receipt["auction"], receipt["postAuction"]["projectPlay"]):
        for row in stage["decisions"]:
            artifact = stage["artifacts"][row["proofArtifactHash"]]
            assert artifact["request"]["visibility"] == "actor_information_set"
            is_tail = row["decisionKind"] == "play" and row["playIndex"] >= 24
            assert artifact["request"]["mode"] == ("exact" if is_tail else "bounded")
            assert artifact["proof"]["completionStatus"] == (
                "exact_complete" if is_tail else "bounded_complete")
            if is_tail:
                assert artifact["proofModel"]["kind"] in {
                    "actor_information_set_cegis_model_v1", "forced_action_optimality_model_v1"}
                if artifact["proofModel"]["kind"] == "actor_information_set_cegis_model_v1":
                    assert artifact["proof"]["hiddenWorlds"]["complete"] is True
