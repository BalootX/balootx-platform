import type {JsonObject} from '../../shared/json';
import type {
  ExactTraceLegalActionV1,
  ResearchRecipe,
  ResearchResult,
  ScenarioPredicateArgumentContextV3,
  ScenarioPredicateDefinitionV1,
  ScenarioPredicateDefinitionV2,
  ScenarioResearchV4CapabilityV1,
  ScenarioStudioPredicateDefinition,
} from './model';
import {isOrToolsResearchBackend} from './model';
import type {ConfigBundle} from '../configuration/model';

export function preferredPlayingQualityBudget(
  mode: string, previousMode: unknown, previousProfile: unknown,
  availableProfiles: readonly string[], defaultProfile: string,
): string {
  return previousMode === mode && typeof previousProfile === 'string'
    && availableProfiles.includes(previousProfile) ? previousProfile : defaultProfile;
}

export function playingQualityBudgetDefaultMode(mode: string, kind: string, exactTailProof: unknown): string {
  return mode === 'budgeted' && kind === 'play' && exactTailProof === true ? 'exact' : mode;
}

export function scenarioLaunchBlockMessageKey(reason: string): string {
  return reason === 'predicate_cp_sat_unsupported'
    ? 'workflow.research_scenario_diagnostic_unsupported_predicate'
    : 'workflow.research_scenario_estimate_refused';
}

export function continuationLineBoundaryStatus(value: JsonObject, lines: JsonObject): string | null {
  if (lines.complete === true) return String(value.completionStatus || 'exact_complete');
  if (value.claim !== 'winning_continuation_line_count'
    && String(lines.lowerBound ?? '0') === '0'
    && (lines.upperBound === null || lines.upperBound === undefined)) return null;
  return 'proof_incomplete';
}

export function preferredNativeResearchBundle(bundles: readonly ConfigBundle[]): ConfigBundle | undefined {
  return bundles.filter((item)=>item.bundleId.startsWith('baloot_ortools_research_native_v3'))
    .sort((left,right)=>right.version.localeCompare(left.version,undefined,{numeric:true}))[0];
}

export function preferredScenarioRecipeCode(
  recipes: readonly ResearchRecipe[], current: string, bundleRevisionId: string,
): string {
  // Preserve an explicit selection for comparison or cloning across bundles.
  if (recipes.some((item)=>item.code===current)) return current;
  return recipes.find((item)=>Number(item.schemaVersion)===4
    && item.configuration?.bundleRevisionId===bundleRevisionId)?.code ?? '';
}

export type ScenarioProjectOption = Readonly<{
  contract: string;
  kind: string;
  value: number;
}>;

function governedEnumSchema(schema: JsonObject, values: readonly (string | number)[]): JsonObject {
  const result: JsonObject = {...schema, enum: [...values]};
  for (const field of ['x-enumMessageArKeys', 'x-enumMessageEnKeys']) {
    const labels = result[field];
    if (!labels || typeof labels !== 'object' || Array.isArray(labels)) continue;
    const filtered = Object.fromEntries(values
      .filter((value) => String(value) in labels)
      .map((value) => [String(value), (labels as JsonObject)[String(value)]]));
    if (Object.keys(filtered).length === values.length) result[field] = filtered;
    else delete result[field];
  }
  return result;
}

export function projectExistsControlSchema(
  argumentSchema: JsonObject,
  projectOptions: readonly ScenarioProjectOption[],
  contract: string | undefined,
  selectedKind: unknown,
): JsonObject {
  const applicable = projectOptions.filter((item) => !contract || item.contract === contract);
  const kinds = [...new Set(applicable.map((item) => item.kind))];
  if (!kinds.length) throw new Error('scenario_project_options_missing');
  const kind = selectedKind === undefined
    ? kinds[0]
    : typeof selectedKind === 'string' && kinds.includes(selectedKind)
      ? selectedKind
      : undefined;
  const values = kind === undefined
    ? []
    : [...new Set(applicable.filter((item) => item.kind === kind).map((item) => item.value))];
  const properties = argumentSchema.properties && typeof argumentSchema.properties === 'object'
    && !Array.isArray(argumentSchema.properties) ? argumentSchema.properties as JsonObject : {};
  return {
    ...argumentSchema,
    properties: {
      ...properties,
      kind: governedEnumSchema(properties.kind as JsonObject, kinds),
      value: governedEnumSchema(properties.value as JsonObject, values),
    },
  };
}

export type ScenarioOrdinalField = 'trickIndex' | 'position';

export function scenarioOrdinalDisplayValue(
  _field: ScenarioOrdinalField,
  wireValue: number,
): number {
  return wireValue + 1;
}

export function scenarioOrdinalWireValue(
  _field: ScenarioOrdinalField,
  displayValue: number,
): number {
  return displayValue - 1;
}

export function undoRecordedEscalationOpportunity(
  prefix: JsonObject,
  history: readonly JsonObject[],
): {prefix: JsonObject; history: JsonObject[]} {
  const event = history.at(-1);
  if (!event) throw new Error('exact_opportunity_history_empty');
  if (['projectActions', 'projectDeclarations', 'playActions'].some(
    (field) => Array.isArray(prefix[field]) && prefix[field].length > 0,
  )) throw new Error('exact_opportunity_has_continuation');
  const next = {...prefix};
  delete next.sha256;
  if (event.actionCode !== 'decline') {
    const actions = Array.isArray(prefix.escalationActions) ? prefix.escalationActions : [];
    const last = actions.at(-1);
    if (!last || typeof last !== 'object' || Array.isArray(last)
      || last.actor !== event.actor
      || [last.action, last.mode].filter(Boolean).join(':') !== event.actionCode) {
      throw new Error('exact_opportunity_public_history_mismatch');
    }
    next.escalationActions = actions.slice(0, -1);
  }
  return {prefix: next, history: history.slice(0, -1)};
}

export function appendExactTraceLegalAction(
  prefix: JsonObject,
  legalAction: ExactTraceLegalActionV1,
): JsonObject {
  const next = {...prefix};
  delete next.sha256;
  const append = (field: string, value: JsonObject) => {
    const current = Array.isArray(next[field]) ? next[field] : [];
    next[field] = [...current, value];
  };
  if (legalAction.stage === 'auction') {
    if (!legalAction.action) throw new Error('exact_preview_auction_action_invalid');
    append('auctionActions', {
      actor: legalAction.actor,
      action: legalAction.action,
      ...(legalAction.trumpSuit ? {trumpSuit: legalAction.trumpSuit} : {}),
    });
    return next;
  }
  if (legalAction.stage === 'escalation') {
    if (!legalAction.action) throw new Error('exact_preview_escalation_action_invalid');
    if (legalAction.action === 'decline') throw new Error('exact_private_opportunity_requires_history');
    append('escalationActions', {
      actor: legalAction.actor,
      action: legalAction.action,
      ...(legalAction.mode ? {mode: legalAction.mode} : {}),
    });
    return next;
  }
  if (legalAction.stage === 'project') {
    if (!legalAction.action || !['pass','declare'].includes(legalAction.action)) {
      throw new Error('exact_preview_project_action_invalid');
    }
    if (legalAction.action === 'declare' && !legalAction.projectId) {
      throw new Error('exact_preview_project_id_invalid');
    }
    append('projectActions', {
      actor: legalAction.actor,
      action: legalAction.action,
      ...(legalAction.projectId ? {projectId: legalAction.projectId} : {}),
    });
    return next;
  }
  if (!legalAction.card || typeof legalAction.ekka !== 'boolean') {
    throw new Error('exact_preview_play_action_invalid');
  }
  append('playActions', {
    actor: legalAction.actor,
    card: legalAction.card.code,
    ekka: legalAction.ekka,
  });
  for (const declaration of legalAction.projectDeclarations ?? []) {
    append('projectDeclarations', {...declaration});
  }
  return next;
}

export type ScenarioSelection = Readonly<{
  origin: 'search' | 'exact_trace_prefix';
  proofMode: 'exact' | 'bounded' | 'policy_conditioned';
  visibility: 'complete_state' | 'actor_information_set';
}>;

export type ScenarioExecutionSelection = ScenarioSelection & Readonly<{
  decisionKind: 'bid' | 'escalation' | 'project' | 'play';
}>;

export type ScenarioPredicateSelection = ScenarioSelection & Readonly<{
  decisionKind?: ScenarioExecutionSelection['decisionKind'];
  lineUniverse?: 'all_legal' | 'optimal_strategy' | 'policy_conditioned';
  utility?: 'awarded_point_margin' | 'contract_success' | 'round_win' | 'match_win' | 'kabout';
}>;

function predicateArgumentContextSupported(
  context: ScenarioPredicateArgumentContextV3,
  selection: ScenarioPredicateSelection,
): boolean {
  return selection.decisionKind !== undefined
    && context.decisionKinds.includes(selection.decisionKind)
    && (context.requiredUtility === null
      || context.requiredUtility === selection.utility)
    && (!context.executionContexts?.length || context.executionContexts.some((item) =>
      item.origin === selection.origin
      && item.decisionKind === selection.decisionKind
      && item.proofMode === selection.proofMode
      && item.visibility === selection.visibility
      && (!item.lineUniverses.length || (selection.lineUniverse !== undefined
        && item.lineUniverses.includes(selection.lineUniverse)))))
    && context.requiredProofBindings.length > 0
    && context.requiredProofBindings.every((binding) =>
      binding.binding.length > 0
      && Number.isInteger(binding.minimumCount)
      && binding.minimumCount > 0);
}

export function scenarioExecutionCombinations(
  capability: ScenarioResearchV4CapabilityV1 | undefined,
  selection: Partial<ScenarioExecutionSelection>,
) {
  if (!isOrToolsScenarioCapability(capability)) return [];
  return capability.executionCapabilities.combinations.filter((item) =>
    (!selection.origin || item.origin === selection.origin)
    && (!selection.decisionKind || item.decisionKind === selection.decisionKind)
    && (!selection.proofMode || item.mode === selection.proofMode)
    && (!selection.visibility || item.visibility === selection.visibility));
}

export function preferredScenarioExecutionCombination(
  capability: ScenarioResearchV4CapabilityV1 | undefined,
) {
  const combinations = scenarioExecutionCombinations(capability, {});
  return combinations.find((item) =>
    item.origin === 'search'
    && item.decisionKind === 'play'
    && item.mode === 'exact'
    && item.visibility === 'complete_state') ?? combinations[0];
}

export function scenarioExecutionSupported(
  capability: ScenarioResearchV4CapabilityV1 | undefined,
  selection: ScenarioExecutionSelection,
  claim: string,
  lineUniverse?: string,
): boolean {
  return scenarioExecutionCombinations(capability, selection).some((item) =>
    item.claims.includes(claim as typeof item.claims[number])
    && (claim !== 'winning_continuation_line_count'
      || item.lineUniverses.includes(lineUniverse as typeof item.lineUniverses[number])));
}

export function scenarioDecisionSelectorSchema(
  capability: ScenarioResearchV4CapabilityV1 | undefined,
  decisionKind: ScenarioExecutionSelection['decisionKind'],
): JsonObject {
  if (!isOrToolsScenarioCapability(capability)) return {};
  return capability.executionCapabilities.decisionSelectorSchemas[decisionKind] ?? {};
}

export function scenarioExecutionContractValues(
  capability: ScenarioResearchV4CapabilityV1 | undefined,
  selection: ScenarioExecutionSelection,
): Array<'sun' | 'hokum' | 'ashkal'> | null {
  const constrained = scenarioExecutionCombinations(capability, selection)
    .map((item)=>item.contractValues)
    .filter((values): values is Array<'sun' | 'hokum' | 'ashkal'> => values !== null);
  if (!constrained.length) return null;
  return constrained.slice(1).reduce(
    (allowed, values)=>allowed.filter((value)=>values.includes(value)),
    [...constrained[0]],
  );
}

function v2AsStudioDefinition(
  definition: ScenarioPredicateDefinitionV2,
): ScenarioStudioPredicateDefinition {
  const proofModes: ScenarioPredicateDefinitionV1['supportedProofModes'] = [];
  if (definition.cpSatSupport.proof.exact !== 'unsupported') proofModes.push('exact');
  if (definition.cpSatSupport.proof.bounded !== 'unsupported') proofModes.push('bounded');
  if (definition.cpSatSupport.proof.policyConditioned !== 'unsupported') {
    proofModes.push('policy_conditioned');
  }
  const hiddenInformation = {
    public: 'public',
    observation: 'actor_visible',
    reference_world_only: 'complete_state_only',
    nonanticipative: 'information_set_safe',
  }[definition.cpSatSupport.actorInformationSet] as ScenarioPredicateDefinitionV1['hiddenInformation'];
  return {
    ...definition,
    schemaVersion: 2,
    version: 1,
    constructionCapability: definition.cpSatSupport.search.status === 'exact'
      ? 'cp_sat_exact'
      : 'unsupported',
    runtimeVerifier: definition.cpSatSupport.evidenceExtractor,
    supportedProofModes: proofModes,
    hiddenInformation,
  };
}

export function isOrToolsScenarioCapability(
  value: ScenarioResearchV4CapabilityV1 | undefined,
): value is ScenarioResearchV4CapabilityV1 {
  return value?.schemaVersion === 1
    && isOrToolsResearchBackend(value.backendIdentity.backendId)
    && value.backendIdentity.qualificationAuthority === 'cp_sat'
    && value.backendIdentity.planner === 'tail_first_v1'
    && value.backendIdentity.deterministicExecution === true
    && value.executionCapabilities?.schemaVersion === 1
    && Array.isArray(value.executionCapabilities.combinations);
}

export function mergeScenarioPredicateDefinitions(
  legacy: readonly ScenarioPredicateDefinitionV1[],
  capability: ScenarioResearchV4CapabilityV1 | undefined,
): ScenarioStudioPredicateDefinition[] {
  if (!isOrToolsScenarioCapability(capability)) return [...legacy];
  const v3Capabilities = capability.backendIdentity.backendId === 'ortools_cp_sat_v3'
    ? new Map(
      capability.executionCapabilities.authoringCatalog.predicateCapabilityManifest
        ?.capabilities.map((item) => [item.kind, item]) ?? [],
    )
    : null;
  const bindV3Capability = (
    definition: ScenarioStudioPredicateDefinition,
  ): ScenarioStudioPredicateDefinition => v3Capabilities === null
    ? definition
    : {
      ...definition,
      v3ExecutionCapability: v3Capabilities.get(definition.kind) ?? null,
    };
  const v2ByKind = new Map(
    capability.hardPredicateCatalog.definitions.map((definition) => [
      definition.kind,
      definition,
    ]),
  );
  const merged = legacy.map((definition): ScenarioStudioPredicateDefinition => {
    const v2 = v2ByKind.get(definition.kind);
    if (!v2) return bindV3Capability(definition);
    v2ByKind.delete(definition.kind);
    return bindV3Capability({...definition, ...v2AsStudioDefinition(v2)});
  });
  return [
    ...merged,
    ...[...v2ByKind.values()].map((definition) =>
      bindV3Capability(v2AsStudioDefinition(definition))),
  ];
}

export function scenarioProofVisibilitySupported(
  capability: ScenarioResearchV4CapabilityV1 | undefined,
  proofMode: ScenarioSelection['proofMode'],
  visibility: ScenarioSelection['visibility'],
): boolean {
  if (!isOrToolsScenarioCapability(capability)) return true;
  return capability.supportedProofVisibility.some(
    (item) => item.mode === proofMode && item.visibility === visibility,
  );
}

export function scenarioPredicateCausalVerificationAvailable(
  definition: ScenarioStudioPredicateDefinition,
  argumentsValue: JsonObject,
): boolean {
  return definition.v3ExecutionCapability?.argumentContexts?.find((context) =>
    context.argumentName === 'kind' && context.argumentValue === argumentsValue.kind,
  )?.causalVerificationAvailable !== false;
}

export function scenarioPredicateSupported(
  definition: ScenarioStudioPredicateDefinition,
  selection: ScenarioPredicateSelection,
  hardness: 'hard' | 'prefer' | 'report',
  argumentsValue?: JsonObject,
): boolean {
  const support = definition.cpSatSupport;
  if (!support || !definition.allowedHardness.includes(hardness)) return false;
  if (definition.kind === 'tactic.verified' && hardness === 'hard'
    && argumentsValue?.causalRequired === true
    && !scenarioPredicateCausalVerificationAvailable(definition, argumentsValue)) return false;
  if (Object.prototype.hasOwnProperty.call(definition, 'v3ExecutionCapability')) {
    const capability = definition.v3ExecutionCapability;
    if (!capability || capability.nativeStatus !== 'native_exact' || !selection.decisionKind) {
      return false;
    }
    const contextSupported = capability.executionContexts.some((context) =>
      context.origin === selection.origin
      && context.decisionKind === selection.decisionKind
      && context.proofMode === selection.proofMode
      && context.visibility === selection.visibility
      && (context.lineUniverses.length === 0
        || (selection.lineUniverse !== undefined
          && context.lineUniverses.includes(selection.lineUniverse))));
    if (!contextSupported) return false;
    if (capability.argumentContexts?.length) {
      const argumentContexts = capability.argumentContexts.filter((context) =>
        predicateArgumentContextSupported(context, selection));
      if (argumentsValue) {
        if (definition.kind.startsWith('tactic.')
          && typeof argumentsValue.kind !== 'string') return false;
        const constrained = capability.argumentContexts.some(
          (context) => context.argumentName in argumentsValue,
        );
        if (constrained && !argumentContexts.some((context) =>
          argumentsValue[context.argumentName] === context.argumentValue)) {
          return false;
        }
      } else if (definition.kind.startsWith('tactic.') && !argumentContexts.length) {
        return false;
      }
    } else if (definition.kind.startsWith('tactic.')) {
      return false;
    }
  }
  const originSupported = selection.origin === 'search'
    ? support.search.status === 'exact' || support.search.status === 'derived'
    : support.exactInput.status === 'exact';
  const proofSupported = {
    exact: support.proof.exact,
    bounded: support.proof.bounded,
    policy_conditioned: support.proof.policyConditioned,
  }[selection.proofMode] !== 'unsupported';
  const visibilitySupported = selection.visibility === 'complete_state'
    || ['public', 'observation', 'nonanticipative'].includes(
      support.actorInformationSet,
    );
  return originSupported && proofSupported && visibilitySupported;
}

export function unsupportedScenarioPredicateIds(
  definitions: readonly ScenarioStudioPredicateDefinition[],
  predicates: readonly Readonly<{
    predicateId: string;
    kind: string;
    hardness: 'hard' | 'prefer' | 'report';
    arguments?: JsonObject;
  }>[],
  selection: ScenarioPredicateSelection,
  contractValues: readonly string[] | null = null,
): string[] {
  const byKind = new Map(definitions.map((definition) => [definition.kind, definition]));
  return predicates
    .filter((predicate) => {
      const definition = byKind.get(predicate.kind);
      return !definition
        || !scenarioPredicateSupported(
          definition,
          selection,
          predicate.hardness,
          predicate.arguments,
        )
        || (predicate.kind === 'auction.contract'
          && contractValues !== null
          && !contractValues.includes(String(predicate.arguments?.contract ?? '')));
    })
    .map((predicate) => predicate.predicateId);
}

export function scenarioPredicateArgumentValues(
  definition: ScenarioStudioPredicateDefinition,
  selection: ScenarioPredicateSelection,
  argumentName: string,
): string[] | null {
  if (!Object.prototype.hasOwnProperty.call(definition, 'v3ExecutionCapability')) {
    return null;
  }
  const capability = definition.v3ExecutionCapability;
  if (!capability || !selection.decisionKind) return [];
  const contexts = capability.argumentContexts?.filter((context) =>
    context.argumentName === argumentName
    && predicateArgumentContextSupported(context, selection)) ?? [];
  return [...new Set(contexts.map((context) => context.argumentValue))].sort();
}

export function scenarioCompiledStages(
  definitions: readonly ScenarioStudioPredicateDefinition[],
  predicates: readonly Readonly<{kind: string; hardness: string}>[],
  origin: ScenarioSelection['origin'],
): string[] {
  const byKind = new Map(definitions.map((definition) => [definition.kind, definition]));
  const stages = new Set<string>();
  predicates.forEach((predicate) => {
    if (predicate.hardness !== 'hard') return;
    const support = byKind.get(predicate.kind)?.cpSatSupport;
    const active = origin === 'search' ? support?.search : support?.exactInput;
    active?.stages.forEach((stage) => stages.add(stage));
  });
  return [...stages].sort();
}

export function resultRuntimeCompatibilityStatus(
  result: ResearchResult,
): 'legacy' | 'not_checked' | 'passed' | 'failed' {
  const authority = result.researchAuthority as JsonObject | undefined;
  if (!isOrToolsResearchBackend(authority?.backendId ?? result.researchBackend)) return 'legacy';
  const status = result.runtimeCompatibility?.status ?? result.runtimeCompatibilityStatus;
  return status === 'passed' || status === 'failed' ? status : 'not_checked';
}

export function resultAllowsNormalMedia(result: ResearchResult): boolean {
  const status = resultRuntimeCompatibilityStatus(result);
  return status === 'legacy' || status === 'passed';
}

export function resultHasReplayTrace(result: ResearchResult): boolean {
  const authority = result.researchAuthority as JsonObject | undefined;
  if (!isOrToolsResearchBackend(authority?.backendId ?? result.researchBackend)) return true;
  const projection = result.scenarioProjection as JsonObject | undefined;
  return projection?.exactStateAvailable === true;
}
