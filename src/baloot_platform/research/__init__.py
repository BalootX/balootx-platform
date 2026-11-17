"""Governed Research Foundry public API.

Public exports are loaded on first access so importing an OR-Tools Research
module cannot initialize the frozen legacy engine-backed Research stack.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_GROUPS = {
    "contracts": (
        "BalootDomainSpecV2",
        "ResearchConfigurationAuthorityV1",
        "ResearchEstimateV1",
        "ResearchRecipeV1",
        "ResearchRecipeV2",
        "ResearchRecipeV3",
        "SemanticDisagreementV1",
        "SemanticDisagreementV2",
        "build_legacy_research_recipe_for_read",
        "build_research_recipe",
        "research_recipe_configuration_v1",
    ),
    "domain_compiler": (
        "BalootDomainCompilerV1",
        "BalootDomainSpecV3",
        "CompiledDomainBundleV1",
        "CompilerCoverageV1",
        "DomainIRV1",
        "DomainPrimitiveDefinitionV1",
        "DomainPrimitiveInstanceV1",
    ),
    "domain_artifact": (
        "DomainCompilationArtifactRepositoryV1",
        "DomainCompilationArtifactV1",
    ),
    "auction_compiler": (
        "AuctionActionV1",
        "AuctionPlanV1",
        "OrToolsCpSatAuctionCompilerV1",
    ),
    "escalation_compiler": (
        "EscalationActionV1",
        "EscalationPlanV1",
        "OrToolsCpSatEscalationCompilerV1",
    ),
    "baloot_platform.core.escalation_ir": ("EscalationDomainIRV1",),
    "reference_escalation": (
        "ReferenceEscalationActionV1",
        "ReferenceEscalationInterpreterV1",
        "ReferenceEscalationPlanV1",
        "ReferenceEscalationStateV1",
    ),
    "project_compiler": (
        "CompiledProjectV1",
        "OrToolsCpSatProjectCompilerV1",
        "ProjectPlanV1",
        "ProjectRevealV1",
    ),
    "play_compiler": (
        "CompiledTrickV1",
        "OrToolsCpSatPlayCompilerV1",
        "PlayPlanV1",
    ),
    "tactical_construction": (
        "TacticalDealBalanceProfileV1",
        "TacticalPlayConstructionProfileV1",
    ),
    "reference_play": (
        "PlayDomainIRV1",
        "ReferencePlayInterpreterV1",
        "ReferencePlayV1",
    ),
    "baloot_platform.core.project_ir": ("ProjectDomainIRV1",),
    "reference_project": ("ReferenceProjectInterpreterV1", "ReferenceProjectV1"),
    "reference_scoring": (
        "ReferenceMatchOutcomeV1",
        "ReferenceRoundScoreV1",
        "ReferenceScoringInterpreterV1",
        "ScoringDomainIRV1",
    ),
    "reference_structural": (
        "ReferenceStructuralInterpreterV1",
        "StructuralDealLayoutV1",
        "StructuralDomainIRV1",
    ),
    "structural_compiler": (
        "OrToolsCpSatStructuralCompilerV1",
        "StructuralPlanV1",
    ),
    "service": ("ResearchFoundryServiceV1",),
    "execution_context": (
        "ResearchExecutionContextFactoryV1",
        "ResearchExecutionContextV1",
        "ResearchHostSafetyCeilingV1",
    ),
    "render_service": ("ResearchRenderPreparationServiceV1",),
    "audit_trace": (
        "CompositeResearchAuditTraceRepositoryV1",
        "ResearchAuditTraceRecordV1",
        "ResearchAuditTraceRepositoryV1",
    ),
    "decision_audit": ("InformationSetEvidenceV1", "SequentialResponseEvidenceV1"),
    "store": ("PostgresResearchStoreV1", "SQLiteResearchStoreV1"),
    "policy_proof": (
        "ActorVisibleActionCutV1",
        "ForcedLateGameProblemBuilderV1",
        "OnlyWinningMoveWitnessV1",
        "OrToolsReducedPolicyProofV1",
        "ReducedActorSafePolicyProblemV1",
        "ReducedPolicyProfileV1",
        "ReducedPolicyProofV1",
        "ReducedWorldOutcomeV1",
    ),
    "best_response_proof": (
        "BestResponseEvidenceV1",
        "EngineReducedBestResponseProblemBuilderV1",
        "OrToolsBoundedBestResponseProofV1",
        "ReducedBestResponseProblemV1",
        "ReducedBestResponseProfileV1",
        "ReducedBestResponseProofV1",
        "ReducedGameNodeV1",
        "ReducedGameWorldV1",
        "ReducedRootActionV1",
        "RegretEvidenceV1",
        "SmallCardBigGainWitnessV1",
    ),
    "decision_proof_adapter": (
        "DecisionAuditV2SolverEvidenceV1",
        "DecisionProofResourcesV1",
        "TraceDecisionPolicyProofAdapterV1",
    ),
    "policy_proof_benchmark": (
        "PolicyProofBenchmarkCaseV1",
        "authoritative_late_game_fixture",
        "benchmark_reduced_policy_proof",
    ),
    "policy_proof_campaign": (
        "PolicyProofCampaignProfileV1",
        "run_policy_proof_qualification_campaign",
    ),
    "strategic_play": (
        "ActorSafeCpSatPlayProfileV1",
        "ActorSafeCpSatPlaySelectorV1",
    ),
    "strategic_auction": (
        "ActorSafeCpSatAuctionProfileV1",
        "ActorSafeCpSatAuctionSelectorV1",
    ),
    "guided_construction_pilot": (
        "GuidedConstructionPilotProfileV1",
        "build_guided_pilot_recipe_v1",
        "run_guided_construction_pilot_v1",
    ),
    "exact_state": ("ExactStateNormalizerV1", "NormalizedExactStateV1"),
    "recipe_conversion": ("preview_v3_to_v4_conversion",),
    "recipe_v4": ("ResearchRecipeV4",),
    "proof_service": (
        "ProofBudgetExceeded",
        "ScenarioProofBudgetRegistryV1",
        "ScenarioProofBudgetV1",
        "UnifiedScenarioProofServiceV1",
    ),
    "scenario_contracts": (
        "ExactTracePrefixV1",
        "ProofOutcomeV1",
        "ResearchScenarioV1",
        "ScenarioDiagnosticV1",
        "ScenarioExpressionV1",
        "ScenarioOriginV1",
        "ScenarioPredicateV1",
        "ScenarioProofRequestV1",
    ),
    "scenario_registry": (
        "ScenarioCapabilityPlanV1",
        "ScenarioCompilerV1",
        "ScenarioPredicateDefinitionV1",
        "ScenarioPredicateRegistryV1",
    ),
    "scenario_evidence": (
        "ScenarioPredicateEvidenceV1",
        "ScenarioRuntimeVerifierV1",
        "ScenarioSatisfactionV1",
    ),
    "scenario_construction": (
        "ScenarioConstructionPlanV1",
        "ScenarioStructuralCompilerV1",
        "ScenarioStructuralResultV1",
    ),
    "scenario_search": (
        "ScenarioGuidedActorSafePolicyV1",
        "ScenarioSearchCandidateV1",
        "ScenarioSearchExecutorV1",
    ),
}

_EXPORT_MODULES = {
    name: module if module.startswith("baloot_platform.") else f"{__name__}.{module}"
    for module, names in _EXPORT_GROUPS.items()
    for name in names
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
