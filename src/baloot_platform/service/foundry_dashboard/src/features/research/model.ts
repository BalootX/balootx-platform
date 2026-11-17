import type {JsonObject, JsonValue, SeatName} from '../../shared/json';
import type {ResearchLessonTagsV1Response, ResearchScenarioAuthorityReceiptResponse} from '../../api/generated/types.gen';

export type ResearchLessonTagQuery = Partial<Record<keyof ResearchLessonTagsV1Response['tags'], string[]>>;
export type ResearchLessonTagCatalog = {
  schemaVersion: 1;
  axes: Array<{
    id: keyof ResearchLessonTagsV1Response['tags'];
    queryParameter: string;
    labelKey: string;
    options: Array<{value: string; labelKey: string}>;
  }>;
  sha256: string;
};

export type ResearchAuctionSeed = {seed: number; dealer: SeatName};

export type ResearchGenerationRequest = {
  schemaVersion: 1;
  catalogHash: string;
  tagQuery: Record<string, string[]>;
  auctionSeeds?: ResearchAuctionSeed[];
  sha256?: string;
};

export type ResearchBackendId = 'ortools_cp_sat_v2' | 'ortools_cp_sat_v3';

export function isOrToolsResearchBackend(value: unknown): value is ResearchBackendId {
  return value === 'ortools_cp_sat_v2' || value === 'ortools_cp_sat_v3';
}

export function isNativeResearchAuthoringBackend(value: unknown): value is 'ortools_cp_sat_v3' {
  return value === 'ortools_cp_sat_v3';
}

export type ResearchComponentRevisionIdentity = {
  revisionId: string;
  profileId: string;
  version: string;
  payloadSha256: string;
};

export type ResearchConfigurationAuthority = {
  schemaVersion: 1;
  bundleRevisionId: string;
  bundleId: string;
  bundleVersion: string;
  semanticBundleHash: string;
  effectiveConfigHash: string;
  effectiveRuntimeConfigHash: string;
  snapshotContract: 'EffectiveRuntimeConfigV2';
  componentRevisions: Record<string, ResearchComponentRevisionIdentity>;
  identities: Record<string, JsonObject>;
  sha256: string;
};

export type ResearchConfigurationSelection = {
  schemaVersion: 1;
  bundleRevisionId: string;
  bundleId: string;
  bundleVersion: string;
  semanticBundleHash: string;
  effectiveRuntimeConfigHash: string;
  contextHash: string;
  domainHash: string;
  domainIrHash: string;
  legacyDomainV2Hash: string;
  ready: boolean;
};

export type ResearchCatalogField = JsonObject & {
  control: string;
  valueType: string;
  mode: string;
  default?: JsonValue;
  required?: true;
  defaultKey?: string;
  helpKey: string;
  example: JsonValue;
};

export type ResearchCatalogRelationship = JsonObject & {
  code: string;
  labelKey: string;
  helpKey: string;
  exampleKey: string;
  example: {valid: JsonValue; invalid: JsonValue};
};

export type ResearchCatalogPreset = {
  code: string;
  labelKey: string;
  helpKey: string;
  patch: JsonObject;
};

export type ResearchRecipeConfigurationCatalog = {
  schemaVersion: number;
  recipeSchemaVersion: number;
  supportedRecipeSchemaVersions: number[];
  legacyReadSchemaVersions: number[];
  fields: Record<string, ResearchCatalogField>;
  relationships: ResearchCatalogRelationship[];
  recipePresets: ResearchCatalogPreset[];
  resourcePresets: ResearchCatalogPreset[];
  projectOptions: Array<{
    contract: string;
    kind: string;
    value: number;
    highRank: string;
    highRankId: number;
  }>;
  projectDomainHash: string;
  unsupportedBehaviorRequiresPlugin: boolean;
  sha256: string;
};

export type ScenarioConstructionCapability =
  | 'cp_sat_exact'
  | 'incremental_actor_safe'
  | 'guided_search'
  | 'postcondition_only'
  | 'exact_input_only'
  | 'unsupported';

export type ScenarioPredicateDefinitionV1 = {
  schemaVersion: 1;
  kind: string;
  version: 1;
  category: string;
  stage: string;
  argumentSchema: JsonObject;
  messageArKey: string;
  messageEnKey: string;
  requiredDomainPrimitives: string[];
  allowedHardness: Array<'hard' | 'prefer' | 'report'>;
  constructionCapability: ScenarioConstructionCapability;
  runtimeVerifier: string;
  supportedProofModes: Array<'exact' | 'bounded' | 'policy_conditioned'>;
  hiddenInformation:
    | 'public'
    | 'actor_visible'
    | 'complete_state_only'
    | 'information_set_safe';
  costClass:
    | 'constant'
    | 'linear'
    | 'solver'
    | 'tree_search'
    | 'hidden_world_search';
  safeComplement: boolean;
  examples: JsonObject[];
  counterexamples: JsonObject[];
  documentation: string[];
  tests: string[];
  sha256: string;
};

export type ScenarioCpSatSupportV1 = {
  schemaVersion: 1;
  search: {
    status: 'exact' | 'derived' | 'unsupported';
    stages: string[];
  };
  exactInput: {
    status: 'exact' | 'unsupported';
    stages: string[];
  };
  qualificationSource: 'construction' | 'derived_state' | 'proof' | 'report_only';
  actorInformationSet: 'public' | 'observation' | 'reference_world_only' | 'nonanticipative';
  proof: {
    exact: 'exact' | 'unsupported';
    bounded: 'exact' | 'bounds' | 'unsupported';
    policyConditioned: 'exact' | 'unsupported';
  };
  evidenceExtractor: string;
};

export type ScenarioPredicateDefinitionV2 = {
  schemaVersion: 2;
  kind: string;
  version: number;
  category: string;
  stage: string;
  argumentSchema: JsonObject;
  messageArKey: string;
  messageEnKey: string;
  requiredDomainPrimitives: string[];
  allowedHardness: Array<'hard' | 'prefer' | 'report'>;
  cpSatSupport: ScenarioCpSatSupportV1;
  costClass: ScenarioPredicateDefinitionV1['costClass'];
  safeComplement: boolean;
  examples: JsonObject[];
  counterexamples: JsonObject[];
  documentation: string[];
  tests: string[];
  lifecycleStatus: 'active' | 'deprecated' | 'retired';
  sha256: string;
};

export type ScenarioStudioPredicateDefinition = Omit<
  ScenarioPredicateDefinitionV1,
  'schemaVersion'
> & {
  schemaVersion: 1 | 2;
  cpSatSupport?: ScenarioCpSatSupportV1;
  lifecycleStatus?: 'active' | 'deprecated' | 'retired';
  v3ExecutionCapability?: ScenarioPredicateCapabilityV3 | null;
};

export type ScenarioPredicateExecutionContextV3 = {
  origin: 'search' | 'exact_trace_prefix';
  decisionKind: 'bid' | 'escalation' | 'project' | 'play';
  proofMode: 'exact' | 'bounded' | 'policy_conditioned';
  visibility: 'complete_state' | 'actor_information_set';
  lineUniverses: ResearchScenarioConfigurationV1['proofLineUniverses'];
};

export type ScenarioPredicateArgumentContextV3 = {
  argumentName: string;
  argumentValue: string;
  decisionKinds: ScenarioPredicateExecutionContextV3['decisionKind'][];
  requiredUtility: ResearchScenarioConfigurationV1['proofUtilities'][number] | null;
  requiredProofBindings: Array<{
    binding: string;
    minimumCount: number;
  }>;
  executionContexts?: ScenarioPredicateExecutionContextV3[];
  causalVerificationAvailable?: boolean;
};

export type ScenarioPredicateCapabilityV3 = {
  schemaVersion: 3;
  kind: string;
  definitionSha256: string;
  registrationSha256: string;
  lowererOwner: string;
  evidenceExtractor: string;
  owningTest: string;
  qualificationBinding: 'cp_sat_literal' | 'proof_variable';
  requiredRulePrimitives: string[];
  nativeStatus: 'native_exact' | 'unsupported';
  origins: ScenarioPredicateExecutionContextV3['origin'][];
  decisionKinds: ScenarioPredicateExecutionContextV3['decisionKind'][];
  proofModes: ScenarioPredicateExecutionContextV3['proofMode'][];
  visibilityClasses: ScenarioPredicateExecutionContextV3['visibility'][];
  executionContexts: ScenarioPredicateExecutionContextV3[];
  argumentContexts?: ScenarioPredicateArgumentContextV3[];
  sha256: string;
};

export type ScenarioPredicateCapabilityManifestV3 = {
  schemaVersion: 3;
  registryVersion: 'scenario_predicate_capability_registry_v3';
  backendId: 'ortools_cp_sat_v3';
  baseManifestHash: string;
  capabilities: ScenarioPredicateCapabilityV3[];
  extensions?: JsonObject[];
  sha256: string;
};

export type ScenarioResearchLocalizationV1 = {
  schemaVersion: 1;
  bundleRevisionId: string;
  productRevisionId: string;
  productProfileHash: string;
  locale: 'ar-SA';
  direction: 'rtl';
  groups: {workflow: Record<string, string>};
  sha256: string;
};

export type ScenarioResearchV4CapabilityV1 = {
  schemaVersion: 1;
  backendIdentity: {
    schemaVersion: 1;
    backendId: ResearchBackendId;
    backendVersion: string;
    qualificationAuthority: 'cp_sat';
    solverVersion: string;
    planner: 'tail_first_v1';
    deterministicExecution: true;
    bundleRevisionId: string;
    bundleIdentityHash: string;
    authorityHash: string;
    predicateCatalogHash: string;
    loweringManifestHash?: string;
    sha256: string;
  };
  executionReady: boolean;
  launchAdmissible: boolean;
  supportedOrigins: Array<'search' | 'exact_trace_prefix'>;
  supportedProofVisibility: Array<{
    mode: 'exact' | 'bounded' | 'policy_conditioned';
    visibility: 'complete_state' | 'actor_information_set';
  }>;
  executionCapabilities: {
    schemaVersion: 1;
    combinations: Array<{
      origin: 'search' | 'exact_trace_prefix';
      decisionKind: 'bid' | 'escalation' | 'project' | 'play';
      mode: 'exact' | 'bounded' | 'policy_conditioned';
      visibility: 'complete_state' | 'actor_information_set';
      claims: ResearchScenarioConfigurationV1['proofClaims'];
      lineUniverses: ResearchScenarioConfigurationV1['proofLineUniverses'];
      strategyClasses: ResearchScenarioConfigurationV1['proofStrategyClasses'];
      contractValues: Array<'sun' | 'hokum' | 'ashkal'> | null;
    }>;
    phase4AcceptanceGaps: Array<{
      acceptanceId: 'phase4_strategic_proof_matrix_v1';
      origin: 'search' | 'exact_trace_prefix';
      decisionKind: 'bid' | 'escalation' | 'project' | 'play';
      mode: 'exact' | 'bounded' | 'policy_conditioned';
      visibility: 'complete_state' | 'actor_information_set';
      reason: 'non_vacuous_full_service_fixture_missing';
      requiredEvidence: string[];
    }>;
    decisionSelectorSchemas: Partial<Record<
      'bid' | 'escalation' | 'project' | 'play',
      JsonObject
    >>;
    authoringCatalog: ScenarioAuthoringCatalogV1;
    sha256: string;
  };
  hardPredicateCatalog: {
    schemaVersion: 2;
    backendIdentityHash: string;
    loweringManifestHash?: string;
    origin: 'search' | 'exact_trace_prefix';
    proofMode: 'exact' | 'bounded' | 'policy_conditioned';
    visibility: 'complete_state' | 'actor_information_set';
    hardness: 'hard' | 'prefer' | 'report';
    definitions: ScenarioPredicateDefinitionV2[];
    registrations?: JsonObject[];
    sha256: string;
  };
  localization: ScenarioResearchLocalizationV1;
  sha256: string;
};

export type ScenarioAuthoringCatalogV1 = {
  familySearch?: {
    schemaVersion: 1;
    playIndices?: number[];
    liveTechniquesByFamily?: Record<string, string[]>;
    catalog: {
      schemaVersion: 1; id: string; revision: number; sha256: string;
      families: Array<{
        id: string; name: string; enabled: boolean; contracts: string[];
        capabilityTags: Record<string, string[]>;
        parameters: {seats: string[]; playIndex: {minimum: number; maximum: number}};
      }>;
    };
    requestSchema: JsonObject;
    supportedAxes: string[];
  };
  familyGeneration?: {
    schemaVersion: 1; familyId: string; name: string; targetHash: string;
    tagQuery: ResearchLessonTagQuery; constraints: JsonObject;
    decisionPoint: {kind: 'play'; selector: {trickIndex: number; position: number;
      remainingCards: number; actingSeat: SeatName}};
    sha256: string;
  };
  lessonTags?: ResearchLessonTagCatalog;
  playingQuality?: {
    schemaVersion: 1;
    policySchemaVersion?: 2;
    visibilities?: string[];
    legacyVisibility?: string;
    modes: string[];
    modesByOrigin?: Record<string, string[]>;
    decisionKinds: string[];
    defaultBudgetProfile: string;
    defaultBudgetProfileByMode: Record<string, string>;
    defaultBudgetProfileByModeAndVisibility?: Record<string, Record<string, string>>;
    defaultBoundaryEvaluatorByDecision: Record<string, string>;
    requiredDecisionFields: string[];
    fieldSources: Record<string, string>;
    execution: JsonObject;
    boundaryEvaluators: string[];
    boundaryEvaluatorDefinitions: Record<string, {
      operatorVisible: boolean; decisionKinds: string[]; kind: string;
      horizonUnits: string[]; horizonValue?: number; utilities?: string[];
    }>;
    horizonBounds: Record<string, Record<string, {minimum:number; step:number; maximum?:number}>>;
    horizonUnitsByDecision: Record<string, string[]>;
    recordedEscalationOpportunities: JsonObject;
  };
  schemaVersion: 1;
  recipeSchemaVersion: 4;
  sections: string[];
  logicalOperators: Array<'all' | 'any' | 'not' | 'count'>;
  tacticKinds: string[];
  proof: {
    modes: ResearchScenarioConfigurationV1['proofModes'];
    visibilities: ResearchScenarioConfigurationV1['proofVisibilities'];
    strategyClasses: ResearchScenarioConfigurationV1['proofStrategyClasses'];
    claims: ResearchScenarioConfigurationV1['proofClaims'];
    utilities: ResearchScenarioConfigurationV1['proofUtilities'];
    lineUniverses: ResearchScenarioConfigurationV1['proofLineUniverses'];
    horizonUnits: ResearchScenarioConfigurationV1['proofHorizonUnits'];
    boundaryEvaluators: ResearchScenarioConfigurationV1['proofBoundaryEvaluators'];
    budgetProfiles: Record<string, ScenarioProofBudgetV1>;
  };
  fields: Record<string, JsonObject>;
  relationships: JsonObject[];
  expressionLimits: {
    maximumDepth: number;
    maximumNodes: number;
    maximumChildren: number;
    maximumSerializedBytes: number;
    maximumCountRange: number;
  };
  exactTrace: {
    schemaVersion: 1;
    roundNumber: JsonObject;
    matchScore: JsonObject;
    stateAssertions: JsonObject;
  };
  executionCombinations: ScenarioResearchV4CapabilityV1['executionCapabilities']['combinations'];
  decisionSelectorSchemas: ScenarioResearchV4CapabilityV1['executionCapabilities']['decisionSelectorSchemas'];
  predicateCapabilityManifest?: ScenarioPredicateCapabilityManifestV3;
  policyCatalog: {
    schemaVersion: 1;
    defaultPolicyId: string;
    profiles: Record<string, {version: string; materialization: string}>;
    seed: {type: 'integer'; minimum: number; maximum: number};
    defaultProfiles: Record<'north' | 'west' | 'south' | 'east', JsonObject>;
  };
  localizationKeys: {
    values: Record<string, string>;
    fields: Record<string, string>;
    exactArguments: Record<string, string>;
    predicateHelp: Record<string, string>;
  };
  sha256: string;
};

export type ScenarioProofBudgetV1 = {
  profileId: string;
  maximumNodes: number;
  maximumDepth: number;
  maximumHorizonActions: number;
  maximumHiddenWorlds: number;
  maximumWallSeconds: number;
  maximumCpuSeconds: number;
  maximumCacheEntries: number;
  maximumCacheBytes: number;
  cancellationCheckIntervalNodes: number;
  maximumWorkers: number;
  maximumLineCountNodes: number;
  maximumStateSpace: number;
  sha256: string;
};

export type ResearchScenarioConfigurationV1 = {
  schemaVersion: 1;
  recipeSchemaVersion: 4;
  supportedLegacyReadVersions: [1, 2, 3];
  origins: Array<'search' | 'exact_trace_prefix'>;
  outputPolicy: {
    kinds: Array<'laffa' | 'laffa_collection'>;
    resultCount: {minimum: number; maximum: number; default: number};
    renderModes: Array<'on_demand' | 'disabled'>;
    videoCount: {
      minimum: number;
      maximum: number;
      default: number;
      maximumRelation: 'output.resultCount';
    };
  };
  decisionCapabilities: Array<{
    kind: 'bid' | 'escalation' | 'project' | 'play';
    authoringSupported: boolean;
    exactTraceNormalizationSupported: boolean;
    searchGenerationSupported: boolean;
    proofModes: Array<'exact' | 'bounded' | 'policy_conditioned'>;
    renderPreparationSupported: boolean;
    unavailableReason: string | null;
  }>;
  sections: string[];
  logicalOperators: Array<'all' | 'any' | 'not' | 'count'>;
  hardness: Array<'hard' | 'prefer' | 'report'>;
  proofModes: Array<'exact' | 'bounded' | 'policy_conditioned'>;
  proofStrategyClasses: Array<'pure_deterministic' | 'frozen_policy'>;
  proofVisibilities: Array<'complete_state' | 'actor_information_set'>;
  proofClaims: Array<'outcome_reachable' | 'outcome_forced' | 'winning_root_action_count' | 'winning_continuation_line_count' | 'selected_action_is_unique_winner' | 'target_team_has_no_salvage'>;
  proofUtilities: Array<'awarded_point_margin' | 'contract_success' | 'round_win' | 'match_win' | 'kabout'>;
  proofBoundaryEvaluators: Array<'current_raw_point_margin_v1'>;
  proofLineUniverses: Array<'all_legal' | 'optimal_strategy' | 'policy_conditioned'>;
  proofHorizonUnits: Array<'plies' | 'decisions' | 'tricks' | 'round_end'>;
  predicateRegistry: {
    schemaVersion: 1;
    registryVersion: 'scenario_predicate_registry_v1';
    definitions: ScenarioPredicateDefinitionV1[];
    sha256: string;
  };
  proofBudgets: {
    schemaVersion: 1;
    profiles: Record<string, ScenarioProofBudgetV1>;
    sha256: string;
  };
  defaultPolicyProfiles: Record<
    'north' | 'west' | 'south' | 'east',
    {
      agentId: 'heuristic_bot_v1';
      version: '1.0.0';
      seed: number;
      profileHash: string;
    }
    | {
      schemaVersion: 1;
      policyId: 'fixed_action_map_v1';
      version: '1.0.0';
      seed: number;
      choices: Array<{nodeId: string; actionId: string}>;
      materialization: 'seeded_legal_action_order_v1';
      profileHash: string;
    }
  >;
  cardOptions: Array<{
    id: number;
    code: string;
    suit: string;
    rank: string;
    symbol: string;
    label: string;
  }>;
  tacticOptions: string[];
  projectOptions: Array<{
    contract: 'sun' | 'hokum' | 'ashkal';
    kind: string;
    value: number;
    highRank: string;
    highRankId: number;
  }>;
  ruleOptions: {
    contracts: Array<'sun' | 'hokum' | 'ashkal'>;
    auctionActionsByRound: Array<{
      round: 1 | 2;
      actions: Array<'pass' | 'sun' | 'hokum' | 'ashkal'>;
    }>;
    escalationLevelsByContract: Array<{
      contract: 'sun' | 'hokum' | 'ashkal';
      levels: Array<'normal' | 'double' | 'triple' | 'four' | 'gahwa'>;
      actions: Array<'decline' | 'double' | 'triple' | 'four' | 'gahwa'>;
    }>;
    ashkalBuyerRelations: Array<'bidder' | 'bidder_partner'>;
    trumpContracts: Array<'sun' | 'hokum' | 'ashkal'>;
    balootContracts: Array<'sun' | 'hokum' | 'ashkal'>;
    playActionOptions: Array<{
      actionId: number;
      code: string;
      card: {
        id: number;
        code: string;
        suit: string;
        rank: string;
        symbol: string;
        label: string;
      };
      ekka: boolean;
    }>;
    projectsEnabled: boolean;
    kaboutEnabled: boolean;
  };
  projectDomainHash: string;
  sha256: string;
};

export type ResearchCapabilities = {
  schemaVersion: 2;
  sha256: string;
  rulesHash: string;
  conventionHash: string;
  bundleHash: string;
  research: {
    solverAvailable: boolean;
    generationLanes: Array<'ortools' | 'organic_runtime'>;
    auctionConstructionModes: Array<
      | 'actor_safe_agent_verified'
      | 'cp_sat_actor_safe_decision_v1'
      | 'cp_sat_feasibility_line'
    >;
    playConstructionModes: Array<
      | 'actor_safe_agent_verified'
      | 'cp_sat_actor_safe_late_game_v1'
      | 'cp_sat_feasibility_line'
    >;
    recommendedPlayConstructionMode:
      | 'actor_safe_agent_verified'
      | 'cp_sat_actor_safe_late_game_v1'
      | 'cp_sat_feasibility_line';
    defaultAgentRelease: string;
    analysisModes: string[];
    deepVerificationPolicies: Array<'required_gates' | 'all_candidates' | 'off'>;
    analysisProfiles: Record<string, string>;
    defaultScoreProfileHash: string;
    recipeConfiguration: ResearchRecipeConfigurationCatalog;
    scenarioConfiguration: ResearchScenarioConfigurationV1;
    scenarioResearchV4?: ScenarioResearchV4CapabilityV1;
    configurationSelection?: ResearchConfigurationSelection;
  };
};

export type ResearchRecipeLifecycleStatus = 'draft' | 'published' | 'retired';

export type ResearchResourcesV4Authoring = JsonObject & {
  generationLane: 'ortools';
  generationPolicy: 'ortools_cp_sat_v3';
  seedStart: number;
  seedCount: number;
  seedsPerShard: number;
  shardWorkerCount?: number;
  solverTimeoutSeconds: number;
  proofBudgetProfile: string;
};

export type ResearchRecipe = JsonObject & {
  code: string;
  titleAr: string;
  sha256: string;
  status: string;
  lifecycleStatus: ResearchRecipeLifecycleStatus;
  configuration?: ResearchConfigurationAuthority;
};

export type ResearchRecipeV4Document = JsonObject & {
  schemaVersion: 4;
  titleAr: string;
  output: {kind: 'laffa' | 'laffa_collection'; resultCount: number};
  configuration: ResearchConfigurationAuthority | {
    schemaVersion: 1;
    bundleRevisionId: string;
  };
  scenario: JsonObject;
  proof: JsonObject;
  generation?: ResearchGenerationRequest;
  qualification: JsonObject;
  resources: ResearchResourcesV4Authoring;
  diversity: JsonObject;
  renderPolicy: JsonObject;
  status: ResearchRecipeLifecycleStatus;
};

export type ResearchRecipeV4ConversionPreview = JsonObject & {
  schemaVersion: 1;
  source: {schemaVersion: 3; recipeCode: string; recipeHash: string};
  proposedRecipe: JsonObject;
  targetRecipeCode: string | null;
  targetRecipeHash: string | null;
  mappedFields: Array<{
    sourcePath: string;
    targetPaths: string[];
    mappingKind: string;
  }>;
  unmappedFields: Array<{
    fieldPath: string;
    reasonCode: string;
    requiresOperatorReview: boolean;
    sourceValue: JsonValue;
  }>;
  requiredOperatorFields: Array<'proof' | 'scenario.decisionPoint' | 'scenario.outcome'>;
  buildDiagnostics: Array<{
    code: string;
    fieldPath: string;
    message: string;
    details?: JsonObject | null;
  }>;
  exactlyRepresentable: boolean;
  buildable: boolean;
  sha256: string;
};

export type ScenarioDiagnosticV1 = JsonObject & {
  issueId?: string;
  class?: string;
  severity: 'error' | 'warning' | 'info';
  code: string;
  field?: string | null;
  predicateIds: string[];
  fieldPaths: string[];
  messageAr?: string | null;
  messageEn?: string | null;
  messageArKey?: string | null;
  messageEnKey?: string | null;
  details: JsonObject;
  sha256?: string | null;
};

export type ResearchRecipeValidationV2 = JsonObject & {
  schemaVersion: 1 | 2;
  valid: boolean;
  issues: ScenarioDiagnosticV1[];
  scenarioPlanHash?: string;
  configurationAuthority?: ResearchConfigurationAuthority;
  capabilityPlan?: JsonObject;
  normalizedExactState?: {
    exactStateId: string;
    phase: string;
    decisionKind: 'bid' | 'escalation' | 'project' | 'play';
    nextActor: string | null;
    legalActions: JsonObject[];
    stateHash: string;
  } | null;
};

export type ExactTraceLegalActionV1 = JsonObject & {
  stage: 'auction' | 'escalation' | 'project' | 'play';
  code: string;
  actor: 'north' | 'west' | 'south' | 'east';
  action?: string | null;
  trumpSuit?: 'spades' | 'hearts' | 'diamonds' | 'clubs' | null;
  mode?: 'open' | 'locked' | null;
  actionId?: number | null;
  projectId?: string | null;
  card?: ResearchScenarioConfigurationV1['cardOptions'][number] | null;
  ekka?: boolean | null;
  projectDeclarations?: Array<{
    owner: 'north' | 'west' | 'south' | 'east';
    kind: string;
    value: number;
    highRank: number;
    declarationOrder: number;
  }>;
};

export type ExactTraceLegalActionsPreviewV1 = JsonObject & {
  schemaVersion: 1;
  configurationAuthority: {
    bundleRevisionId: string;
    semanticBundleHash: string;
    effectiveRuntimeConfigHash: string;
    backendIdentityHash: string;
  };
  valid: boolean;
  currentStage: 'auction' | 'escalation' | 'project' | 'play' | 'terminal' | null;
  nextActor: 'north' | 'west' | 'south' | 'east' | null;
  terminal: boolean;
  legalActions: ExactTraceLegalActionV1[];
  diagnostics: Array<{code: 'exact_trace_prefix_invalid' | 'exact_normalization_incomplete'; fieldPath: 'exactTracePrefix'}>;
  prefixHash: string | null;
  previewStateHash: string | null;
  previewReceiptHash: string | null;
  sha256: string;
};

export type ResearchEstimateProofV2 = JsonObject & {
  requestedMode: 'exact' | 'bounded' | 'policy_conditioned';
  visibility: 'complete_state' | 'actor_information_set';
  strategyClass: 'pure_deterministic' | 'frozen_policy';
  horizon: JsonObject | null;
  admissible: boolean;
  reason: string | null;
  recommendedMode: 'bounded' | 'policy_conditioned' | null;
  recommendedModes: Array<'bounded' | 'policy_conditioned'>;
  estimatedRootActionUpperBound: number;
  actionDomainUpperBound: number;
  estimatedEkkaActionVariantUpperBound: number;
  ekkaDeclarationSequenceUpperBound: string;
  targetInformationSetUpperBound?: number | null;
  opponentInformationSetUpperBound?: number | null;
  targetPolicyCountUpperBound?: string | null;
  opponentPolicyCountUpperBound?: string | null;
  policyProductUpperBound?: string | null;
  estimatedTreeStateUpperBound?: number | null;
  policyProductExceedsBudget?: boolean | null;
  informationTreeExceedsBudget?: boolean | null;
  estimatedStateSpaceClass: string;
  hiddenWorldClass: string;
  budgetProfile: string;
  resourceBudgetProfile: string;
  remainingCardCount?: number | null;
  requestedDepth?: number | null;
  budgetMaximumDepth?: number | null;
  budgetMaximumHorizonActions?: number | null;
  estimatedStateSpace?: number | null;
  estimatedHiddenWorldUpperBound?: number | null;
  budgetMaximumHiddenWorlds?: number | null;
  budgetStateSpaceLimit?: number | null;
  budgetProfileHash?: string | null;
  proofBudgetRegistryHash: string;
};

export type ResearchTacticKind =
  | 'only_winning_move'
  | 'small_card_big_gain'
  | 'large_card_sacrifice'
  | 'kabout_creation'
  | 'kabout_preservation'
  | 'kabout_prevention'
  | 'contract_rescue'
  | 'entry_unblocking'
  | 'trump_control'
  | 'partner_enablement'
  | 'project_bidding'
  | 'debate';

export type ResearchTacticDraft = {
  kind: ResearchTacticKind;
  mode: 'hard' | 'preference' | 'qualification_gate';
  minimumScore: number;
  weight: number;
  perspectives: string[];
};

export type ResearchRunPerformanceReceipt = JsonObject & {
  schemaVersion: 1;
  measurementScope: 'research_v4_completed_run_v1';
  runCode: string;
  executionKind: 'exact_trace_prefix' | 'scenario_search';
  generationLane: string;
  attemptedSeeds: number;
  attemptedCandidates: number;
  structuralCandidates: number;
  auctionFeasible: number;
  solverOptimal: number;
  solverIncomplete: number;
  runtimeParityPassed: number;
  semanticReplayPassed: number;
  proofCompleted: number;
  resultCount: number;
  qualifiedCount: number;
  rejectedCount: number;
  quarantineCount: number;
  funnel: {
    attempted: number;
    structurallyFeasible: number;
    legalReplay: number;
    semanticSatisfied: number;
    proofCompleted: number;
    qualified: number;
  };
  constructionSeconds: number | null;
  simulationSeconds: number | null;
  semanticParitySeconds: number | null;
  analysisSeconds: number | null;
  tacticalVerificationSeconds: number | null;
  candidatePipelineSeconds: number | null;
  firstFeasibleCandidateSeconds: number | null;
  firstOptimalCandidateSeconds: number | null;
  firstQualifiedResultSeconds: number | null;
  shardWallWorkSeconds: number;
  cpuSeconds: number;
  cpuMeasurementBoundary: string;
  peakProcessRssBytes: number | null;
  peakRssMeasurementBoundary: string | null;
  stageTimingBoundary: Record<string, string>;
  environment: JsonObject;
  cacheScopes: JsonObject;
  proofResources: {
    nodes: number;
    cacheEntries: number;
    cacheBytes: number;
    hiddenWorlds: number;
    aggregation: string;
  };
  cancellationObserved: boolean;
  terminalStatus: string;
  sha256: string;
};

export type ResearchCandidateProgressClassification =
  | 'qualified'
  | 'rejected'
  | 'duplicate'
  | 'incomplete';

export type ResearchCandidateProgressReceipt = Readonly<JsonObject & {
  schemaVersion: 1;
  receiptId: string;
  classification: ResearchCandidateProgressClassification;
  status: string;
  stage: string;
  candidateHash: string | null;
  variantSemanticHash?: string;
  shardIndex?: number;
  shardId?: string;
  attempt?: number;
  executionOrdinal?: number;
  retryAttempt?: number;
  completedCandidates?: number;
  qualifiedCandidates?: number;
  proofCompletionStatus?: string;
  nearestAcceptedDistance?: number;
  candidateSummary?: JsonObject;
  incompleteCompilation?: JsonObject;
}>;

export type ResearchCampaignProgress = JsonObject & {
  schemaVersion: 1;
  state: 'running' | 'completed';
  qualitySolver?: {
    phase: 'preparation' | 'master' | 'counterexample' | 'binding';
    masterCount: number;
    queryCount: number;
    bindingCount: number;
    decisionKind?: ResearchScenarioConfigurationV1['decisionCapabilities'][number]['kind'];
    proofPhase?: 'best' | 'selected';
  } | null;
  currentSolverCounters?: {
    tailCandidates?: number;
    proofBoundaryAttempts?: number;
    prefixLiftAttempts?: number;
  } | null;
  candidateReceiptCount?: number;
  candidateReceiptCounts?: Record<ResearchCandidateProgressClassification, number>;
  candidateStatusCounts?: Record<string, number>;
  candidateReceiptHistoryLimit?: number;
  candidateReceiptHistoryTruncated?: boolean;
  candidateReceipts?: ResearchCandidateProgressReceipt[];
};

export type ResearchRunProgressEvent = JsonObject & {
  type: string;
  sequence?: number;
  candidateReceipt?: ResearchCandidateProgressReceipt;
};

export type ResearchRun = JsonObject & {
  runCode: string;
  recipeCode: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  resultCount: number;
  resultCodes?: string[];
  auditResultCount?: number;
  auditResultCodes?: string[];
  qualifiedCount: number;
  selectedCount?: number;
  duplicateQualifiedCount?: number;
  quarantineCount: number;
  events?: ResearchRunProgressEvent[];
  auctionOpenings?: ResearchAuctionSeed[];
  error?: JsonObject | string | null;
  createdAt?: string;
  performanceReceipt?: ResearchRunPerformanceReceipt;
  campaignProgress?: ResearchCampaignProgress;
  campaignAccounting?: ResearchCampaignProgress;
};

export function researchRunDisplayStatus(run: ResearchRun): string {
  if (run.status !== 'completed') return run.status;
  // A terminal worker is not necessarily a completed search or proof.
  const campaign = run.campaignAccounting ?? run.campaignProgress;
  switch (campaign?.stopReason) {
    case 'proof_incomplete':
    case 'resource_incomplete':
    case 'global_campaign_timeout': return 'incomplete';
    case 'exact_structural_infeasibility': return 'structurally_infeasible';
    case 'candidate_range_exhausted': return 'exhausted';
    case 'qualified_target_satisfied': return 'qualified';
    case 'cancelled': return 'cancelled';
    default: return run.status;
  }
}

export type ResearchAuthorityV1 = {
  schemaVersion: 1;
  backendId: ResearchBackendId;
  backendVersion: string;
  qualificationAuthority: 'cp_sat';
  solverVersion: string;
  bundleRevisionId: string;
  bundleHash: string;
  domainIrHashes: Record<string, string>;
  loweringManifestHash: string;
  constructionModelHash: string;
  proofModelHash: string;
  graphHash: string;
  traceHash: string;
  proofOutcomeHash: string;
  sha256: string;
};

export type ResearchRuntimeCompatibilityV1 = {
  schemaVersion: 1;
  status: 'not_checked' | 'passed' | 'failed';
  engineVersion: string | null;
  bundleHash: string;
  traceHash: string;
  evidenceHash: string | null;
  failureCodes: string[];
  sha256: string;
};

export type ResearchExecutionReceiptV1 = {
  schemaVersion: 1;
  semanticResultHash: string;
  stageTimings: Record<string, number>;
  cpuSeconds: number;
  peakMemoryBytes: number;
  conflicts: number;
  branches: number;
  cacheBytes: number;
  completedVariants: number;
  machine: JsonObject;
  sha256: string;
};

export type ResearchResultOperatorAuditV1 = {
  schemaVersion: 1;
  resultCode: string;
  researchAuthority: ResearchAuthorityV1;
  runtimeCompatibility: ResearchRuntimeCompatibilityV1;
  executionReceipt: ResearchExecutionReceiptV1;
};

export type ResearchPlayingQualityDecisionV1 = {
  ordinal: number;
  kind: 'bid' | 'escalation' | 'project' | 'play';
  actor: SeatName;
  action: string;
  legalActionCount: number;
  actualActionOptimal: boolean;
  regret: number;
  selectionReason: string;
  proofMode: 'exact' | 'bounded';
  proofVisibility: 'complete_state' | 'actor_information_set';
  proofCompletionStatus: 'exact_complete' | 'bounded_complete';
  proofScope: string;
  proofArtifactHash: string;
  actionQualityHash: string;
  sha256: string;
};

export type ResearchPlayingQualityAuditV1 = {
  schemaVersion: 1;
  resultCode: string;
  resultHash: string;
  scope: 'all_decisions';
  status: 'complete';
  wholeRoundCertified: false;
  operation: 'generate' | 'audit_recorded' | 'audit_then_generate';
  qualityReceiptHash: string;
  completeTraceHash: string;
  decisionCount: number;
  optimalDecisionCount: number;
  zeroRegretDecisionCount: number;
  countsByKind: Record<'bid' | 'escalation' | 'project' | 'play', number>;
  decisions: ResearchPlayingQualityDecisionV1[];
  featuredPlayProof?: {
    exactStateId: string;
    playIndex: number;
    actor: SeatName;
    action: string;
    proofArtifactHash: string;
  } & ({
    proofModelKind: 'native_observed_root_outcome_v1';
    completionStatus: 'bounded_complete';
    hiddenWorlds: { method: 'supplied_worlds_only'; count: number; complete: false };
  } | {
    proofModelKind: 'native_complete_state_root_outcome_v1';
    completionStatus: 'exact_complete';
    hiddenWorlds?: never;
  } | {
    proofModelKind: 'actor_information_set_cegis_model_v1';
    completionStatus: 'exact_complete';
    hiddenWorlds: { method: 'exhaustive_cp_sat_hidden_allocations_v1'; count: number; complete: true };
  }) | null;
  sha256: string;
};

export type ResearchResult = JsonObject & {
  resultCode: string;
  runCode: string;
  score: number;
  truthLabel: string;
  confidence: number;
  disposition: string;
  decisionCount: number;
  tacticalTags: string[];
  classification?: string;
  publicationEligible?: boolean;
  publicationBlockers?: string[];
  publicationPolicy?: ResearchRenderPublicationPolicy;
  researchBackend?: ResearchBackendId;
  qualificationAuthority?: 'cp_sat';
  runtimeCompatibilityStatus?: 'not_checked' | 'passed' | 'failed';
  researchAuthority?: ResearchAuthorityV1;
  runtimeCompatibility?: ResearchRuntimeCompatibilityV1;
  executionReceipt?: ResearchExecutionReceiptV1;
};

export type ResearchResultGroup = {value: string; count: number};

export type ResearchV4ResultFacetName =
  | 'scenario_template'
  | 'scenario_origin'
  | 'scenario_status'
  | 'requested_proof_mode'
  | 'achieved_proof_mode'
  | 'proof_completion'
  | 'proof_visibility'
  | 'target_team'
  | 'contract_outcome'
  | 'project_kind'
  | 'project_value'
  | 'project_state'
  | 'winning_root_action_band'
  | 'exact_state_available'
  | 'decision_kind'
  | 'research_backend'
  | 'qualification_authority'
  | 'runtime_compatibility_status'
  | 'configuration_bundle'
  | 'exact_state_id'
  | 'scenario_category'
  | 'variant_collection';

export type ResearchV4ResultFilters = Partial<Record<
  | 'scenarioTemplate'
  | 'scenarioOrigin'
  | 'scenarioStatus'
  | 'requestedProofMode'
  | 'achievedProofMode'
  | 'proofCompletion'
  | 'proofVisibility'
  | 'targetTeam'
  | 'contractOutcome'
  | 'projectKind'
  | 'projectValue'
  | 'projectState'
  | 'winningRootActionBand'
  | 'exactStateAvailable'
  | 'decisionKind'
  | 'researchBackend'
  | 'qualificationAuthority'
  | 'runtimeCompatibilityStatus'
  | 'configurationBundle'
  | 'exactStateId'
  | 'scenarioCategory'
  | 'variantCollection',
  string
>>;

export type ResearchResultPage = {
  items: ResearchResult[];
  total: number;
  offset: number;
  limit: number;
  nextOffset: number | null;
  hasMore: boolean;
  groupBy: string | null;
  groups: ResearchResultGroup[];
};

export type SemanticDisagreement = JsonObject & {
  disagreementId: string;
  runCode: string;
  category: string;
  effectiveCategory?: string;
  semanticDefect?: boolean;
  status: string;
};

export type ResearchEstimate = JsonObject & {
  schemaVersion: 1 | 2;
  valid: boolean;
  seedCount: number;
  candidateCount: number;
  shardCount: number;
  sharding?: {
    configuredSeedsPerShard: number;
    requestedShardCount: number;
    actualShardCount: number;
    maximumShardWorkers: number;
    requestedShardWorkers?: number;
    effectiveShardWorkers: number;
    maximumShardWallSeconds?: number;
    parallelismLimitedByShardPlan: boolean;
    recommendedSeedsPerShard: number;
  };
  maximumSolverSeconds: number;
  maximumCampaignSeconds?: number;
  analysisDecisionUpperBound: number;
  warnings: string[];
  tacticalVerificationDecisionUpperBound: number;
  issues: ScenarioDiagnosticV1[];
  scenarioPlanHash?: string;
  capabilityPlanHash?: string;
  nativeCapabilityPlanHash?: string;
  predicatesByConstructionCapability?: Record<string, string[]>;
  postconditionOnlyPredicates?: string[];
  exactInputOnlyPredicates?: string[];
  unsupportedPredicates?: string[];
  proof?: ResearchEstimateProofV2;
  output?: {
    kind: 'laffa' | 'laffa_collection';
    requestedResultCount: number;
    requestedVideoCount: number;
    maximumCandidateCount: number;
    admissible: boolean;
    reason?: string | null;
    recommendedResultCount: number;
    requiresResourceExpansion: boolean;
  };
  expectedCostCategory?: string;
  backendIdentity?: ScenarioResearchV4CapabilityV1['backendIdentity'];
  compiledStages?: string[];
  constructionPlan?: {
    schemaVersion: 1;
    planner: 'generic' | 'tail_first';
    fallbackReason?: string | null;
    prefixLength?: number | null;
    tailOffset?: number | null;
    tailLength?: number | null;
    sha256: string;
  };
  resourceTier?: string | null;
  refusalReason?: string | null;
  estimatedModelVariables?: number | null;
  estimatedModelConstraints?: number | null;
  estimatedGraphNodes?: number | null;
  estimatedGraphEdges?: number | null;
  estimatedHiddenWorlds?: number | string | null;
  estimatedCountLimbs?: number | null;
  firstResultCostClass?: 'within_selected_tier' | 'exceeds_selected_tier' | 'not_estimable' | null;
  requestedResultsCostClass?: 'within_selected_tier' | 'exceeds_selected_tier' | 'not_estimable' | null;
  alternativeProofModes?: Array<'exact' | 'bounded' | 'policy_conditioned'>;
  requestedCombinationSupported?: boolean;
  exhaustiveSearchInfeasibilityEstablishable?: boolean;
  launchAdmissible?: boolean;
  configurationAuthority?: ResearchConfigurationAuthority;
  estimateHash?: string;
};

export type ResearchStats = JsonObject & {
  funnel: Record<string, number>;
  measuredRunCount: number;
  elapsedWallSeconds: number;
  cpuSeconds: number | null;
  qualifiedPerWallHour: number;
  qualifiedPerCpuHour: number | null;
  qualifiedPer100Attempts: number;
  stageSeconds: Record<string, number>;
  semanticDefects: Record<string, number>;
  historicalRecords: Record<string, number>;
  byGenerationLane: Record<
    string,
    {
      runCount: number;
      attemptedSeeds: number;
      generatedCandidates: number;
      solverIncomplete: number;
      semanticReplayPassed: number;
      playCompilerAttempted: number;
      playCompilerOptimal: number;
      playCompilerFeasible: number;
      playCompilerIncomplete: number;
      qualifiedCount: number;
      rejectedCount: number;
      quarantineCount: number;
      elapsedWallSeconds: number;
      cpuSeconds: number;
      cpuMeasuredRunCount: number;
      qualifiedPerWallHour: number;
      qualifiedPerCpuHour: number | null;
      qualifiedPer100Candidates: number;
      averageSecondsPerAttempt: number;
      tacticalVerificationCandidates: number;
      tacticalVerificationDecisions: number;
      tacticalVerificationSeconds: number;
      constructionSeconds: number;
      simulationSeconds: number;
      semanticParitySeconds: number;
      analysisSeconds: number;
    }
  >;
  performance: ResearchPerformanceRollup;
  performanceByGenerationLane: Record<string, ResearchPerformanceRollup>;
  sequentialSearch: ResearchSequentialSearchSummary;
};

export type ResearchSequentialSearchContinuationStatusDistribution = Readonly<{
  optimal: number;
  infeasible: number;
}>;

export type ResearchSequentialSearchStopReasonDistribution = Readonly<{
  candidates_exhausted: number;
  maximum_solutions_reached: number;
  qualified: number;
  incomplete: number;
  cancelled: number;
  fatal_quarantine: number;
}>;

export type ResearchSequentialSearchSummary = Readonly<{
  admittedSeedCount: number;
  exactRejectionCount: number;
  noGoodAppliedCount: number;
  noGoodUnappliedCount: number;
  continuationSolveCount: number;
  continuationStatusDistribution: ResearchSequentialSearchContinuationStatusDistribution;
  stopReasonDistribution: ResearchSequentialSearchStopReasonDistribution;
}>;

export type ResearchSequentialSearchReceiptItem = Readonly<{
  schemaVersion: 1;
  occurrenceHash: string;
  receiptHash: string;
  runCode: string;
  shardId: string;
  attemptCount: number;
  fencingToken: number;
  seed: number;
  admitted: true;
  createdAt: string;
  summary: ResearchSequentialSearchSummary;
  receiptBundle: Readonly<JsonObject>;
}>;

export type ResearchSequentialSearchReceiptPage = Readonly<{
  items: readonly ResearchSequentialSearchReceiptItem[];
  total: number;
  offset: number;
  limit: number;
  nextOffset: number | null;
  hasMore: boolean;
}>;

export type ResearchTimingDistribution = {
  sampleCount: number;
  p50Seconds: number | null;
  p95Seconds: number | null;
  totalSeconds: number | null;
};

export type ResearchPerformanceRollup = {
  shardCount: number;
  completedShardCount: number;
  attemptedSeeds: number;
  resultCount: number;
  qualifiedCount: number;
  qualifiedDistinctScenarioCount: number | null;
  qualifiedDistinctScenariosPerWallHour: number | null;
  qualifiedDistinctScenariosPerCpuHour: number | null;
  qualifiedPer100Attempts: number;
  resultPer100Attempts: number;
  duplicateCount: number | null;
  duplicateRate: number | null;
  duplicateEvidence: string;
  timing: Record<string, ResearchTimingDistribution>;
  cpu: ResearchTimingDistribution & {
    availability: 'measured' | 'unavailable';
    note: string;
  };
  costProxy: {
    wallSecondsPerQualified: number | null;
    workerThreadCpuSecondsPerQualified: number | null;
    monetaryCost: null;
    note: string;
  };
  operations: {
    currentQueuedShardCount: number;
    currentRunningShardCount: number;
    currentCancelledShardCount: number;
    currentFailedShardCount: number;
    shardExecutionCount: number;
    repeatedExecutionCount: number;
    retryCount: number | null;
    retryAvailability: string;
    reclaimCount: number | null;
    reclaimAvailability: string;
  };
  solverStatusDistribution: {
    construction: Record<string, number>;
    playCompiler: Record<string, number>;
    coverage: string;
  };
  measurementAvailability: Record<string, {status: string; reason: string}>;
};

export type ResearchRenderPublicationPolicy = JsonObject & {
  schemaVersion: 1;
  blockers: string[];
  sha256: string;
  interestAdmission?: {passed: boolean; evidenceHash: string};
};

export type ResearchRenderSourceProvenance = Readonly<
  JsonObject & {
    kind:
      | 'research_result'
      | 'research_scenario_result'
      | 'analysis_candidate'
      | 'finalist'
      | 'laffa'
      | 'research_audit_trace';
    code: string;
    decisionOrdinal?: number;
    decisionId?: string;
    auditTraceCode?: string;
    recordHash?: string;
    attemptStatus?:
      | 'running'
      | 'completed'
      | 'retry_scheduled'
      | 'failed'
      | 'reclaimed'
      | 'cancelled';
    observerSeat?: SeatName;
    traceHash?: string;
    analysisHash?: string;
    domainHash?: string;
    rulesHash?: string;
    originKind?: string;
    sourceStatus?: string;
    qualificationStatus?:
      | 'qualified'
      | 'provisional'
      | 'rejected'
      | 'quarantined'
      | 'legacy_replay';
    auditOnly?: boolean;
    renderPurpose?: 'content_candidate' | 'research_audit';
    qualificationFailures?: string[];
    publicationBlockers?: string[];
    publicationEligible?: boolean;
    publicationPolicy?: ResearchRenderPublicationPolicy;
  }
>;

export type ResearchPreparedRender = JsonObject & {
  resultCode?: string;
  laffaCode?: string;
  decisionOrdinal: number;
  decisionId: string;
  observerSeat: SeatName;
  exportPreset: 'decision_clip' | 'hooked_flagship';
  qualificationStatus:
    | 'qualified'
    | 'provisional'
    | 'rejected'
    | 'quarantined'
    | 'legacy_replay';
  auditOnly?: boolean;
  publicationEligible: boolean;
  renderPurpose: 'content_candidate' | 'research_audit';
  qualificationFailures: string[];
  publicationBlockers?: string[];
  publicationPolicy?: ResearchRenderPublicationPolicy;
  bundleArtifact: JsonObject;
  proofArtifact?: JsonObject;
  proofArtifactSha256?: string;
  authorityReceipt?: ResearchScenarioAuthorityReceiptResponse;
  basename: string;
  preview: JsonObject & {sceneCount?: number};
  source: ResearchRenderSourceProvenance;
};
