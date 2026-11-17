import {copyJson, type JsonObject, type JsonValue} from '../../shared/json';
import type {
  ResearchRecipeConfigurationCatalog,
  ScenarioAuthoringCatalogV1,
} from './model';

export type ResearchFieldCatalog = Pick<
  ResearchRecipeConfigurationCatalog | ScenarioAuthoringCatalogV1,
  'fields'
>;

function researchConfigurationValues(value: JsonValue, path: string): JsonValue[] {
  const segments = path.split('.').filter(Boolean);
  const visit = (current: JsonValue, index: number): JsonValue[] => {
    if (index >= segments.length) {
      return Array.isArray(current)
        ? current.flatMap((item) => visit(item, index))
        : [current];
    }
    if (Array.isArray(current)) {
      return current.flatMap((item) => visit(item, index));
    }
    if (!current || typeof current !== 'object') return [];
    const segment = segments[index];
    const repeated = segment.endsWith('[]');
    const key = repeated ? segment.slice(0, -2) : segment;
    const next = (current as JsonObject)[key];
    if (next === undefined) return [];
    return repeated && Array.isArray(next)
      ? next.flatMap((item) => visit(item, index + 1))
      : visit(next, index + 1);
  };
  return visit(value, 0);
}

function researchConfigurationPresent(value: JsonValue): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (value && typeof value === 'object') return Object.keys(value).length > 0;
  return Boolean(value);
}

export function researchRelationshipActive(
  relationship: JsonObject,
  payload: JsonObject,
): boolean {
  const condition = relationship.if;
  if (typeof condition === 'string') {
    return researchConfigurationValues(payload, condition).some(
      researchConfigurationPresent,
    );
  }
  if (!condition || typeof condition !== 'object' || Array.isArray(condition)) {
    return false;
  }
  return Object.entries(condition as JsonObject).every(([path, expected]) => {
    const actual = researchConfigurationValues(payload, path);
    const expectedValues = Array.isArray(expected) ? expected : [expected];
    return actual.some((value) => expectedValues.some((item) => value === item));
  });
}

export function researchCatalogFieldApplicable(
  catalog: ResearchFieldCatalog,
  path: string,
  payload: JsonObject,
): boolean {
  const field = catalog.fields[path];
  if (!field) return false;
  const condition = field.applicableWhen;
  if (condition === undefined) return true;
  if (!condition || typeof condition !== 'object' || Array.isArray(condition)) {
    return false;
  }
  const conditionValue = condition as JsonObject;
  if (
    typeof conditionValue.path !== 'string'
    || typeof conditionValue.operator !== 'string'
  ) {
    return false;
  }
  const actual = researchConfigurationValues(payload, conditionValue.path);
  if (conditionValue.operator === 'equals') {
    return actual.some((value) => value === conditionValue.value);
  }
  if (conditionValue.operator === 'present') {
    return actual.some(researchConfigurationPresent);
  }
  return false;
}

export function researchValueAtPath(
  value: JsonValue,
  path: string,
): JsonValue | undefined {
  let current: JsonValue | undefined = value;
  for (const segment of path.split('.').filter(Boolean)) {
    if (!current || typeof current !== 'object' || Array.isArray(current)) {
      return undefined;
    }
    current = (current as JsonObject)[segment];
  }
  return current;
}

export function mergeResearchPatch(
  base: JsonObject,
  patch: JsonObject,
): JsonObject {
  const merged = copyJson(base);
  for (const [key, value] of Object.entries(patch)) {
    const previous = merged[key];
    merged[key] =
      value
      && typeof value === 'object'
      && !Array.isArray(value)
      && previous
      && typeof previous === 'object'
      && !Array.isArray(previous)
        ? mergeResearchPatch(previous as JsonObject, value as JsonObject)
        : copyJson(value);
  }
  return merged;
}

export function researchCatalogInitial<T>(
  catalog: ResearchFieldCatalog,
  path: string,
  patch: JsonObject,
): T {
  const patched = researchValueAtPath(patch, path);
  if (patched !== undefined) return copyJson(patched) as T;
  const field = catalog.fields[path];
  if (!field || field.default === undefined) {
    throw new Error(`Research catalog default is missing: ${path}`);
  }
  return copyJson(field.default) as T;
}

export function researchCatalogObjectInitial(
  catalog: ResearchFieldCatalog,
  path: string,
  patch: JsonObject,
): JsonObject {
  const base = catalog.fields[path]?.default;
  if (!base || typeof base !== 'object' || Array.isArray(base)) {
    throw new Error(`Research catalog profile default is missing: ${path}`);
  }
  const patched = researchValueAtPath(patch, path);
  if (patched === undefined) return copyJson(base as JsonObject);
  if (!patched || typeof patched !== 'object' || Array.isArray(patched)) {
    throw new Error(`Research catalog profile patch is invalid: ${path}`);
  }
  return mergeResearchPatch(base as JsonObject, patched as JsonObject);
}

const researchCreationDefaultPaths = [
  'output.kind',
  'output.resultCount',
  'situation.dealer',
  'situation.buyer',
  'situation.objectiveSeat',
  'situation.constructionBuyerRelation',
  'situation.purchaseRound',
  'situation.purchaseSuits',
  'situation.trumpSuit',
  'situation.contracts',
  'situation.matchScore',
  'situation.requiredCards',
  'situation.forbiddenCards',
  'situation.finalSuitCountRanges',
  'situation.projects',
  'situation.escalations',
  'tactics',
  'tactics[].mode',
  'tactics[].minimumScore',
  'tactics[].weight',
  'tactics[].perspectives',
  'qualification.minimumTruth',
  'qualification.minimumConfidence',
  'qualification.minimumSurroundingQuality',
  'qualification.maximumRegret',
  'resources.generationLane',
  'resources.seedStart',
  'resources.seedCount',
  'resources.seedsPerShard',
  'resources.candidatesPerSeed',
  'resources.solverTimeoutSeconds',
  'resources.maximumConflicts',
  'resources.analysisMode',
  'resources.deepVerificationPolicy',
  'resources.tacticalConstruction',
  'resources.tacticalConstruction.enabled',
  'resources.tacticalConstruction.maximumPayoffTricks',
  'resources.tacticalConstruction.minimumLaterRawPoints',
  'resources.tacticalConstruction.requirePrecursorForGatedTactics',
  'resources.constructionPreScreen',
  'resources.constructionPreScreen.enabled',
  'resources.constructionPreScreen.maximumPlayDecisions',
  'resources.constructionPreScreen.maximumBidRegret',
  'resources.constructionPreScreen.maximumPlayRegret',
  'resources.constructionPreScreen.maximumProbabilityDrop',
  'resources.constructionPreScreen.minimumConfidence',
  'resources.constructionPreScreen.rejectCatastrophicCrossing',
  'resources.tacticalViabilityPreScreen',
  'resources.tacticalViabilityPreScreen.enabled',
  'resources.tacticalViabilityPreScreen.maximumDecisions',
  'resources.tacticalViabilityPreScreen.requiredTacticsOnly',
  'resources.tacticalViabilityPreScreen.rejectWhenExactAbsent',
  'resources.tacticalDealBalance',
  'resources.tacticalDealBalance.enabled',
  'resources.tacticalDealBalance.maximumTeamHighCardDifference',
  'resources.tacticalDealBalance.minimumHighCardsPerSeat',
  'resources.tacticalDealBalance.maximumHighCardsPerSeat',
  'resources.tacticalDealBalance.minimumTargetLowCards',
  'resources.auctionConstructionMode',
  'resources.strategicAuction',
  'resources.strategicAuction.requestedSamples',
  'resources.strategicAuction.integerScale',
  'resources.strategicAuction.maximumConflicts',
  'resources.strategicAuction.maximumSelectionSeconds',
  'resources.strategicAuction.requireOptimalForQualification',
  'resources.playConstructionMode',
  'resources.strategicPlay',
  'resources.strategicPlay.minimumPlayedCards',
  'resources.strategicPlay.requestedSamples',
  'resources.strategicPlay.exactAllocationLimit',
  'resources.strategicPlay.maximumSamplingAttempts',
  'resources.strategicPlay.maximumWorlds',
  'resources.strategicPlay.maximumContinuationSteps',
  'resources.strategicPlay.maximumPlies',
  'resources.strategicPlay.maximumSelectionSeconds',
  'resources.strategicPlay.requireExactForQualification',
  'resources.strategicPlay.outcomeBoundary',
  'diversity',
  'diversity.cellDimensions',
  'diversity.topKPerCell',
  'diversity.rejectedAuditModulus',
  'status',
] as const;

const researchRequiredNestedOperatorPaths = [
  'configuration.bundleRevisionId',
  'situation.projects[].seat',
  'situation.projects[].contract',
  'situation.projects[].kind',
  'situation.projects[].value',
  'situation.projects[].highRank',
  'situation.projects[].mode',
  'tactics[].kind',
] as const;

export function researchCatalogReady(
  catalog: ResearchRecipeConfigurationCatalog | undefined,
  translateCatalog: (key: string) => string,
): boolean {
  if (
    !catalog
    || catalog.schemaVersion !== 1
    || catalog.recipeSchemaVersion !== 3
    || !catalog.supportedRecipeSchemaVersions.includes(3)
  ) {
    return false;
  }
  if (!catalog.recipePresets.length || !catalog.resourcePresets.length) return false;
  const fields = Object.values(catalog.fields);
  if (
    !fields.length
    || fields.some(
      (field) =>
        !field.control
        || !field.valueType
        || !field.mode
        || !field.helpKey
        || field.example === undefined
        || (field.default === undefined) === (field.required === undefined),
    )
  ) {
    return false;
  }
  if (
    fields.some(
      (field) =>
        field.applicableWhen !== undefined
        && (
          !field.applicableWhen
          || typeof field.applicableWhen !== 'object'
          || Array.isArray(field.applicableWhen)
          || !['equals', 'present'].includes(
            String((field.applicableWhen as JsonObject).operator || ''),
          )
          || typeof (field.applicableWhen as JsonObject).path !== 'string'
          || field.nonApplicableBehavior !== 'omitted'
        ),
    )
  ) {
    return false;
  }
  if (
    researchCreationDefaultPaths.some(
      (path) => catalog.fields[path]?.default === undefined,
    )
  ) {
    return false;
  }
  if (
    researchRequiredNestedOperatorPaths.some(
      (path) => !catalog.fields[path] || catalog.fields[path].mode !== 'operator',
    )
  ) {
    return false;
  }
  if (!catalog.fields.titleAr?.defaultKey) return false;
  const tactics = catalog.fields.tactics?.default;
  if (!Array.isArray(tactics) || !tactics.length) return false;
  const contracts = catalog.fields['situation.contracts']?.default;
  if (!Array.isArray(contracts) || contracts.length !== 1) return false;
  if (
    typeof catalog.fields['resources.auctionConstructionMode']?.default !== 'string'
    || typeof catalog.fields['resources.playConstructionMode']?.default !== 'string'
  ) {
    return false;
  }
  for (const path of [
    'resources.tacticalConstruction',
    'resources.constructionPreScreen',
    'resources.tacticalViabilityPreScreen',
    'resources.tacticalDealBalance',
    'resources.strategicAuction',
    'resources.strategicPlay',
  ]) {
    const value = catalog.fields[path]?.default;
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  }
  const presets = [...catalog.recipePresets, ...catalog.resourcePresets];
  if (
    presets.some(
      (preset) =>
        !preset.code || !preset.labelKey || !preset.helpKey || !preset.patch,
    )
  ) {
    return false;
  }
  if (
    catalog.relationships.some(
      (relationship) =>
        !relationship.code
        || !relationship.labelKey
        || !relationship.helpKey
        || !relationship.exampleKey
        || !relationship.example,
    )
  ) {
    return false;
  }
  try {
    fields.forEach((field) => {
      translateCatalog(field.helpKey);
      if (field.defaultKey) translateCatalog(field.defaultKey);
    });
    presets.forEach((preset) => {
      translateCatalog(preset.labelKey);
      translateCatalog(preset.helpKey);
    });
    catalog.relationships.forEach((relationship) => {
      translateCatalog(relationship.labelKey);
      translateCatalog(relationship.helpKey);
      translateCatalog(relationship.exampleKey);
    });
  } catch {
    return false;
  }
  return true;
}
