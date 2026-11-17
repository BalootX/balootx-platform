import type {ResearchAuctionSeed, ResearchGenerationRequest, ResearchLessonTagCatalog, ResearchLessonTagQuery, ResearchRun, ScenarioAuthoringCatalogV1} from './model';
import type {JsonObject, JsonValue} from '../../shared/json';
import {isSeatName} from '../../shared/json';
import {appendLessonTagQuery, lessonTagCatalogReady} from './lessonTags';

function object(value: JsonValue | undefined): JsonObject {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

export function frozenFamilyGeneration(catalog: ScenarioAuthoringCatalogV1): ScenarioAuthoringCatalogV1['familyGeneration'] | null {
  const target = catalog.familyGeneration;
  const selector = target?.decisionPoint?.selector;
  const properties = object(catalog.decisionSelectorSchemas?.play?.properties);
  if (!selector || !catalog.lessonTags || !lessonTagCatalogReady(catalog.lessonTags, () => true)) return null;
  const seats = object(properties.actingSeat).enum;
  if (!Array.isArray(seats) || !seats.includes(selector.actingSeat)) return null;
  if (!(['trickIndex', 'position', 'remainingCards'] as const).every((key) => {
    const field = object(properties[key]);
    return Number.isInteger(selector[key]) && typeof field.minimum === 'number' && typeof field.maximum === 'number'
      && selector[key] >= field.minimum && selector[key] <= field.maximum;
  })) return null;
  try { appendLessonTagQuery(new URLSearchParams(), catalog.lessonTags, target!.tagQuery); } catch { return null; }
  return target?.schemaVersion === 1 && typeof target.name === 'string'
    && typeof target.familyId === 'string' && /^[0-9a-f]{64}$/.test(target.targetHash)
    && /^[0-9a-f]{64}$/.test(target.sha256) && target.decisionPoint?.kind === 'play'
    && target.constraints && typeof target.constraints === 'object' && !Array.isArray(target.constraints)
    && /^[0-9a-f]{64}$/.test(String(target.constraints.sha256))
    && (typeof target.constraints.predicateId === 'string'
      || Array.isArray(target.constraints.children) && target.constraints.children.length > 0)
    && target.tagQuery && typeof target.tagQuery === 'object' && !Array.isArray(target.tagQuery)
    ? target : null;
}

export function frozenFamilySearch(catalog: ScenarioAuthoringCatalogV1): ScenarioAuthoringCatalogV1['familySearch'] | null {
  const search = catalog.familySearch;
  if (search?.schemaVersion !== 1 || !lessonTagCatalogReady(catalog.lessonTags, () => true)
    || search.catalog?.schemaVersion !== 1 || !/^[0-9a-f]{64}$/.test(search.catalog.sha256)
    || typeof search.catalog.id !== 'string' || !Number.isInteger(search.catalog.revision) || search.catalog.revision < 1
    || !Array.isArray(search.catalog.families) || !search.catalog.families.length
    || new Set(search.catalog.families.map((family) => family?.id)).size !== search.catalog.families.length
    || !Array.isArray(search.supportedAxes) || !search.supportedAxes.length
    || new Set(search.supportedAxes).size !== search.supportedAxes.length
    || search.supportedAxes.some((axis) => axis === 'level' || !catalog.lessonTags!.axes.some((item) => item.id === axis))
    || search.requestSchema?.type !== 'object' || search.requestSchema.additionalProperties !== false
    || !Array.isArray(search.requestSchema.required)
    || !['schemaVersion', 'catalogHash', 'tagQuery'].every((key) => (search.requestSchema.required as JsonValue[]).includes(key))) return null;
  try {
    if (!search.catalog.families.every((family) => {
      if (!family || typeof family.id !== 'string' || typeof family.name !== 'string'
        || typeof family.enabled !== 'boolean' || !Array.isArray(family.contracts)) return false;
      appendLessonTagQuery(new URLSearchParams(), catalog.lessonTags!, family.capabilityTags);
      return !family.enabled || family.contracts.length === 1 && family.contracts[0] === 'sun'
        && family.capabilityTags.mode?.length === 1 && family.capabilityTags.mode[0] === 'sun'
        && catalog.lessonTags!.axes.every((axis) => Array.isArray(family.capabilityTags[axis.id]) && family.capabilityTags[axis.id].length > 0)
        && Array.isArray(family.parameters?.seats) && family.parameters.seats.length > 0
        && Number.isInteger(family.parameters.playIndex?.minimum) && Number.isInteger(family.parameters.playIndex?.maximum)
        && family.parameters.playIndex.minimum >= 16 && family.parameters.playIndex.maximum <= 27
        && family.parameters.playIndex.minimum <= family.parameters.playIndex.maximum;
    }) || !search.catalog.families.some((family) => family.enabled)) return null;
    if (search.playIndices !== undefined && (!Array.isArray(search.playIndices) || !search.playIndices.length
      || new Set(search.playIndices).size !== search.playIndices.length
      || search.playIndices.some((index) => !Number.isSafeInteger(index) || index < 0))) return null;
    if (search.liveTechniquesByFamily !== undefined && (
      !search.liveTechniquesByFamily || typeof search.liveTechniquesByFamily !== 'object'
      || Array.isArray(search.liveTechniquesByFamily)
      || Object.keys(search.liveTechniquesByFamily).length !== search.catalog.families.length
      || search.catalog.families.some((family) => {
        const values = search.liveTechniquesByFamily![family.id];
        return !Array.isArray(values) || new Set(values).size !== values.length
          || values.some((value) => !family.capabilityTags.technique.includes(value));
      }))) return null;
    if (generationTagValues(search, 'technique').includes('project-inference')
      && search.liveTechniquesByFamily === undefined) return null;
  } catch { return null; }
  return search;
}

export function familySearchTagCatalog(catalog: ScenarioAuthoringCatalogV1,
  selected: ResearchLessonTagQuery = {}): ResearchLessonTagCatalog | null {
  const search = frozenFamilySearch(catalog);
  if (!search) return null;
  const families = search.catalog.families.filter((family) => family.enabled);
  return {...catalog.lessonTags!, axes: catalog.lessonTags!.axes.filter((axis) => search.supportedAxes.includes(axis.id))
    .map((axis) => ({...axis, options: axis.options.filter((option) =>
      generationTagValues(search, axis.id).includes(option.value) && families.some((family) =>
        familyGenerationValues(search, family, axis.id)?.includes(option.value)) || selected[axis.id]?.includes(option.value))}))
    .filter((axis) => axis.options.length > 0)};
}

function generationTagValues(search: NonNullable<ScenarioAuthoringCatalogV1['familySearch']>, axis: string): JsonValue[] {
  const field = object(object(object(search.requestSchema.properties).tagQuery).properties)[axis];
  const schema = object(field);
  const values = object(schema.contains ?? schema.items).enum;
  return Array.isArray(values) ? values : [];
}

function familyGenerationValues(search: NonNullable<ScenarioAuthoringCatalogV1['familySearch']>,
  family: NonNullable<ScenarioAuthoringCatalogV1['familySearch']>['catalog']['families'][number], axis: string): string[] | undefined {
  return axis === 'technique' && search.liveTechniquesByFamily
    ? search.liveTechniquesByFamily[family.id] : family.capabilityTags[axis];
}

function familyMatchesQuery(family: NonNullable<ScenarioAuthoringCatalogV1['familySearch']>['catalog']['families'][number],
  query: ResearchLessonTagQuery, search?: NonNullable<ScenarioAuthoringCatalogV1['familySearch']>): boolean {
  return family.enabled && Object.entries(query).every(([axis, values]) =>
    values?.some((value) => (search ? familyGenerationValues(search, family, axis) : family.capabilityTags[axis])?.includes(value)));
}

export function matchingSearchFamilies(catalog: ScenarioAuthoringCatalogV1, query: ResearchLessonTagQuery,
  selector?: JsonObject): string[] {
  const search = frozenFamilySearch(catalog);
  if (!search) return [];
  const supported = Object.fromEntries(Object.entries(query).map(([axis, values]) =>
    [axis, values?.filter((value) => generationTagValues(search, axis).includes(value))]));
  const families = search.catalog.families.filter((family) => familyMatchesQuery(family, supported, search));
  if (selector === undefined) return families.map((family) => family.id);
  const {actingSeat, trickIndex, position} = selector;
  if (!isSeatName(actingSeat) || typeof trickIndex !== 'number' || !Number.isSafeInteger(trickIndex) || trickIndex < 0
    || typeof position !== 'number' || !Number.isSafeInteger(position) || position < 0 || position >= 4
    || 'remainingCards' in selector && selector.remainingCards !== 8 - trickIndex) return [];
  const index = trickIndex * 4 + position;
  if (!search.playIndices?.includes(index)
    || query.trickBand && !query.trickBand.includes(`trick-${trickIndex + 1}`)) return [];
  return families.filter((family) => family.parameters.seats.includes(actingSeat)
    && index >= family.parameters.playIndex.minimum && index <= family.parameters.playIndex.maximum)
    .map((family) => family.id);
}

export function familySearchAuctionSeedLimit(catalog: ScenarioAuthoringCatalogV1): number | null {
  const schema = object(object(frozenFamilySearch(catalog)?.requestSchema.properties).auctionSeeds);
  const maximum = schema.maxItems;
  return schema.type === 'array' && Number.isSafeInteger(maximum) && typeof maximum === 'number' && maximum > 0
    ? maximum : null;
}

function auctionSeed(catalog: ScenarioAuthoringCatalogV1, raw: unknown): ResearchAuctionSeed | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const row = raw as Record<string, unknown>;
  const schema = object(object(catalog.familySearch?.requestSchema.properties).auctionSeeds);
  const dealers = object(object(object(schema.items).properties).dealer).enum;
  return Object.keys(row).length === 2 && Number.isSafeInteger(row.seed) && typeof row.seed === 'number' && row.seed >= 0
    && isSeatName(row.dealer) && Array.isArray(dealers) && dealers.includes(row.dealer)
    ? {seed: row.seed, dealer: row.dealer} : null;
}

export function familySearchAuctionOpenings(catalog: ScenarioAuthoringCatalogV1, runs: readonly ResearchRun[],
  bundleRevisionId: string, selected: readonly ResearchAuctionSeed[] = []): ResearchAuctionSeed[] {
  if (familySearchAuctionSeedLimit(catalog) === null) return [];
  const choices = [...selected, ...runs.filter((run) => run.status === 'completed'
    && object(run.configurationAuthority).bundleRevisionId === bundleRevisionId).flatMap((run) => run.auctionOpenings ?? [])];
  const seen = new Set<string>();
  return choices.flatMap((raw) => {
    const row = auctionSeed(catalog, raw);
    if (!row) return [];
    const key = `${row.seed}:${row.dealer}`;
    if (seen.has(key)) return [];
    seen.add(key);
    return [row];
  });
}

export function readFamilySearchRequest(catalog: ScenarioAuthoringCatalogV1, raw: unknown): ResearchGenerationRequest | null {
  if (raw === undefined) return null;
  const search = frozenFamilySearch(catalog);
  if (!search || !raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('workflow.research_lesson_tag_search_invalid');
  }
  const value = raw as Record<string, unknown>;
  const query = value.tagQuery;
  if (Object.keys(value).some((key) => !['schemaVersion', 'catalogHash', 'tagQuery', 'auctionSeeds', 'sha256'].includes(key))
    || value.schemaVersion !== 1 || value.catalogHash !== search.catalog.sha256
    || !query || typeof query !== 'object' || Array.isArray(query)
    || Object.values(query).some((values) => !Array.isArray(values) || !values.length)
    || 'sha256' in value && (typeof value.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(value.sha256))) {
    throw new Error('workflow.research_lesson_tag_search_invalid');
  }
  const tags = query as Record<string, string[]>;
  try { appendLessonTagQuery(new URLSearchParams(), familySearchTagCatalog(catalog, tags)!, tags); }
  catch { throw new Error('workflow.research_lesson_tag_search_invalid'); }
  // Saved requests remain readable for correction even when a producer is unavailable.
  if (!search.catalog.families.some((family) => familyMatchesQuery(family, tags))) {
    throw new Error('workflow.research_lesson_tag_search_no_match');
  }
  let auctionSeeds: ResearchAuctionSeed[] | undefined;
  if ('auctionSeeds' in value) {
    const maximum = familySearchAuctionSeedLimit(catalog);
    if (maximum === null || !Array.isArray(value.auctionSeeds) || !value.auctionSeeds.length || value.auctionSeeds.length > maximum) {
      throw new Error('workflow.research_lesson_tag_search_invalid');
    }
    auctionSeeds = value.auctionSeeds.map((row) => {
      const hint = auctionSeed(catalog, row);
      if (!hint) throw new Error('workflow.research_lesson_tag_search_invalid');
      return hint;
    });
    if (new Set(auctionSeeds.map((row) => `${row.seed}:${row.dealer}`)).size !== auctionSeeds.length) {
      throw new Error('workflow.research_lesson_tag_search_invalid');
    }
  }
  return {schemaVersion: 1, catalogHash: search.catalog.sha256,
    tagQuery: Object.fromEntries(Object.entries(tags).map(([axis, values]) => [axis, [...values]])),
    ...(auctionSeeds ? {auctionSeeds} : {}),
    ...(typeof value.sha256 === 'string' ? {sha256: value.sha256} : {})};
}

export function familySearchPayload(catalog: ScenarioAuthoringCatalogV1, generation: ResearchGenerationRequest | null,
  scope: {origin: string; decisionKind: string; playingQuality: boolean}): JsonObject {
  if (generation === null) return {};
  if (scope.origin !== 'search' || scope.decisionKind !== 'play' || !scope.playingQuality) {
    throw new Error('workflow.research_lesson_tag_generation_requires_play');
  }
  return {generation: readFamilySearchRequest(catalog, generation)!};
}

export function appendFamilySearchAuctionSeed(catalog: ScenarioAuthoringCatalogV1,
  generation: ResearchGenerationRequest, opening: unknown): ResearchGenerationRequest {
  const {sha256: _hash, ...request} = generation;
  return readFamilySearchRequest(catalog, {...request, auctionSeeds: [...(request.auctionSeeds ?? []), opening]})!;
}
