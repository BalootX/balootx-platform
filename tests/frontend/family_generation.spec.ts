import {expect, test} from 'vitest';
import {appendFamilySearchAuctionSeed, familySearchAuctionOpenings, familySearchAuctionSeedLimit, familySearchPayload, familySearchTagCatalog, frozenFamilyGeneration, frozenFamilySearch, matchingSearchFamilies, readFamilySearchRequest} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/familyGeneration';
import type {ResearchRun, ScenarioAuthoringCatalogV1} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model';

test('frozen generation target remains structural, detached from result evidence', () => {
  const target = {schemaVersion: 1, familyId: 'guard', name: 'Guard', targetHash: 'a'.repeat(64), sha256: 'b'.repeat(64),
    tagQuery: {technique: ['guard/protect']}, constraints: {predicateId: 'PRED-GUARD', sha256: 'c'.repeat(64)},
    decisionPoint: {kind: 'play', selector: {trickIndex: 6, position: 3, remainingCards: 2, actingSeat: 'west'}}};
  const metadata = {decisionSelectorSchemas: {play: {properties: {trickIndex: {minimum: 0, maximum: 7},
    position: {minimum: 0, maximum: 3}, remainingCards: {minimum: 0, maximum: 8}, actingSeat: {enum: ['west']}}}},
    lessonTags: {schemaVersion: 1, sha256: 'd'.repeat(64), axes: [{id: 'technique', queryParameter: 'lessonTechnique',
      labelKey: 'technique', options: [{value: 'guard/protect', labelKey: 'guard'}]}]}};
  const catalog = {...metadata, familyGeneration: target} as unknown as ScenarioAuthoringCatalogV1;
  expect(frozenFamilyGeneration(catalog)).toBe(target);
  expect(frozenFamilyGeneration({} as ScenarioAuthoringCatalogV1)).toBeNull();
  for (const invalid of [{...target, decisionPoint: null}, {...target, constraints: []},
    {...target, targetHash: 'forged'}, {...target, tagQuery: null}, {...target, constraints: {}},
    {...target, tagQuery: {invented: ['x']}},
    {...target, decisionPoint: {kind: 'play', selector: {...target.decisionPoint.selector, position: 99}}},
    {...target, decisionPoint: {kind: 'play', selector: {...target.decisionPoint.selector, actingSeat: 'unknown'}}}]) {
    expect(frozenFamilyGeneration({...metadata, familyGeneration: invalid} as unknown as ScenarioAuthoringCatalogV1)).toBeNull();
  }
});

function searchCatalog(): ScenarioAuthoringCatalogV1 {
  const selections = {impact: ['round-flip', 'points-only'], technique: ['guard/protect', 'endplay'],
    level: ['intermediate'], mode: ['sun', 'hokum'], trickBand: ['trick-5', 'trick-7']};
  const family = {id: 'guard', name: 'Guard', enabled: true, contracts: ['sun'],
    parameters: {seats: ['west'], playIndex: {minimum: 16, maximum: 27}},
    capabilityTags: {impact: ['round-flip'], technique: ['guard/protect'], level: ['intermediate'],
      mode: ['sun'], trickBand: ['trick-5', 'trick-7']}};
  return {lessonTags: {schemaVersion: 1, sha256: 'a'.repeat(64), axes: Object.entries(selections).map(([id, values]) => ({
    id, labelKey: id, queryParameter: `lesson${id[0].toUpperCase()}${id.slice(1)}`,
    options: values.map((value) => ({value, labelKey: value})),
  }))}, familySearch: {schemaVersion: 1, playIndices: [16, 17, 24, 27],
    supportedAxes: ['impact', 'technique', 'mode', 'trickBand'],
    requestSchema: {type: 'object', additionalProperties: false, required: ['schemaVersion', 'catalogHash', 'tagQuery'],
      properties: {tagQuery: {type: 'object', additionalProperties: false, properties: {
        impact: {items: {enum: selections.impact}, contains: {enum: ['round-flip', 'points-only']}},
        technique: {items: {enum: selections.technique}, contains: {enum: ['guard/protect']}},
        mode: {items: {enum: selections.mode}, contains: {enum: ['sun']}},
        trickBand: {items: {enum: selections.trickBand}},
      }}, auctionSeeds: {type: 'array', minItems: 1, maxItems: 2, uniqueItems: true,
        items: {type: 'object', additionalProperties: false, required: ['seed', 'dealer'],
          properties: {seed: {type: 'integer', minimum: 0}, dealer: {type: 'string', enum: ['north', 'west', 'south', 'east']}}}}}},
    catalog: {schemaVersion: 1, id: 'families', revision: 1, sha256: 'b'.repeat(64), families: [family,
      {...family, id: 'disabled', enabled: false, contracts: ['hokum'], capabilityTags: {...family.capabilityTags,
        impact: ['points-only'], technique: ['endplay'], mode: ['hokum']}}]}}} as unknown as ScenarioAuthoringCatalogV1;
}

test('search tags expose only enabled capabilities and never audit-owned level', () => {
  const catalog = searchCatalog();
  expect(frozenFamilySearch(catalog)).toBe(catalog.familySearch);
  const visible = familySearchTagCatalog(catalog)!;
  expect(visible.axes.map((axis) => axis.id)).toEqual(['impact', 'technique', 'mode', 'trickBand']);
  expect(visible.axes.find((axis) => axis.id === 'mode')!.options.map((item) => item.value)).toEqual(['sun']);
  expect(visible.axes.find((axis) => axis.id === 'technique')!.options.map((item) => item.value)).toEqual(['guard/protect']);
  expect(matchingSearchFamilies(catalog, {technique: ['guard/protect']})).toEqual(['guard']);
  expect(matchingSearchFamilies(catalog, {impact: ['points-only']})).toEqual([]);
  expect(catalog.lessonTags!.axes.find((axis) => axis.id === 'mode')!.options).toHaveLength(2);
});

test('search metadata fails closed for unknown catalogs and unsupported axes', () => {
  expect(frozenFamilySearch({} as ScenarioAuthoringCatalogV1)).toBeNull();
  for (const mutate of [
    (catalog: ScenarioAuthoringCatalogV1) => {catalog.familySearch!.catalog.sha256 = 'forged';},
    (catalog: ScenarioAuthoringCatalogV1) => {catalog.familySearch!.supportedAxes.push('level');},
    (catalog: ScenarioAuthoringCatalogV1) => {catalog.familySearch!.supportedAxes.push('invented');},
    (catalog: ScenarioAuthoringCatalogV1) => {catalog.familySearch!.catalog.families[0].contracts = ['hokum'];},
    (catalog: ScenarioAuthoringCatalogV1) => {catalog.familySearch!.requestSchema.additionalProperties = true;},
  ]) {
    const catalog = searchCatalog();
    mutate(catalog);
    expect(frozenFamilySearch(catalog)).toBeNull();
  }
});

test('new choices require a live producer, while unsupported saved choices remain readable for correction', () => {
  const catalog = searchCatalog();
  catalog.lessonTags!.axes.find((axis) => axis.id === 'impact')!.options.push({value: 'project-swing', labelKey: 'project'});
  catalog.lessonTags!.axes.find((axis) => axis.id === 'technique')!.options.push({value: 'project-inference', labelKey: 'inference'});
  const family = catalog.familySearch!.catalog.families[0];
  family.capabilityTags.impact.push('project-swing');
  family.capabilityTags.technique.push('project-inference');
  const visible = familySearchTagCatalog(catalog)!;
  expect(visible.axes.find((axis) => axis.id === 'impact')!.options.map((option) => option.value)).toEqual(['round-flip']);
  expect(visible.axes.find((axis) => axis.id === 'technique')!.options.map((option) => option.value)).toEqual(['guard/protect']);
  expect(matchingSearchFamilies(catalog, {impact: ['project-swing']})).toEqual([]);
  expect(matchingSearchFamilies(catalog, {impact: ['project-swing', 'points-only']})).toEqual([]);
  expect(matchingSearchFamilies(catalog, {impact: ['project-swing', 'round-flip'],
    technique: ['project-inference', 'guard/protect']})).toEqual(['guard']);
  const saved = {schemaVersion: 1, catalogHash: catalog.familySearch!.catalog.sha256,
    tagQuery: {impact: ['project-swing'], technique: ['project-inference']}, sha256: 'c'.repeat(64)};
  expect(readFamilySearchRequest(catalog, saved)).toEqual(saved);
  expect(familySearchPayload(catalog, readFamilySearchRequest(catalog, saved),
    {origin: 'search', decisionKind: 'play', playingQuality: true})).toEqual({generation: saved});
  expect(familySearchTagCatalog(catalog, saved.tagQuery)!.axes.find((axis) => axis.id === 'impact')!.options
    .map((option) => option.value)).toEqual(['round-flip', 'project-swing']);
  expect(catalog.lessonTags!.axes.find((axis) => axis.id === 'impact')!.options.map((option) => option.value))
    .toContain('project-swing');
});

test('missing producer metadata cannot offer new tag choices or match an authored query', () => {
  const catalog = searchCatalog();
  delete catalog.familySearch!.requestSchema.properties;
  expect(familySearchTagCatalog(catalog)!.axes).toEqual([]);
  expect(matchingSearchFamilies(catalog, {impact: ['round-flip']})).toEqual([]);
});

test('matching family count respects the actual actor, decision and frozen proposal indices', () => {
  const catalog = searchCatalog();
  catalog.familySearch!.catalog.families[0].parameters.playIndex = {minimum: 17, maximum: 24};
  const query = {technique: ['guard/protect']};
  const before = structuredClone(catalog);
  for (const selector of [
    {actingSeat: 'west', trickIndex: 4, position: 1, remainingCards: 4},
    {actingSeat: 'west', trickIndex: 6, position: 0, remainingCards: 2},
  ]) expect(matchingSearchFamilies(catalog, query, selector)).toEqual(['guard']);
  for (const selector of [
    {actingSeat: 'north', trickIndex: 4, position: 1},
    {actingSeat: 'west', trickIndex: 4, position: 0},
    {actingSeat: 'west', trickIndex: 4, position: 2},
    {actingSeat: 'west', trickIndex: 6, position: 3},
    {actingSeat: 'west', trickIndex: 4, position: 1, remainingCards: 2},
    {actingSeat: 'west', trickIndex: 4, position: 4},
    {actingSeat: 'west', trickIndex: 4.25, position: 0},
    {actingSeat: 'west', trickIndex: 4},
    {},
  ]) expect(matchingSearchFamilies(catalog, query, selector)).toEqual([]);
  const late = {actingSeat: 'west', trickIndex: 6, position: 0, remainingCards: 2};
  expect(matchingSearchFamilies(catalog, {...query, trickBand: ['trick-5']}, late)).toEqual([]);
  expect(matchingSearchFamilies(catalog, {...query, trickBand: ['trick-5', 'trick-7']}, late)).toEqual(['guard']);
  expect(catalog).toEqual(before);
  const historical = structuredClone(catalog);
  delete historical.familySearch!.playIndices;
  expect(frozenFamilySearch(historical)).not.toBeNull();
  expect(matchingSearchFamilies(historical, query, {actingSeat: 'west', trickIndex: 4, position: 1})).toEqual([]);
  expect(matchingSearchFamilies(historical, query)).toEqual(['guard']);
  for (const indices of [[], [17, 17], [-1], [17.5]]) {
    expect(frozenFamilySearch({...catalog, familySearch: {...catalog.familySearch!, playIndices: indices}})).toBeNull();
  }
});

test('derived family techniques exclude impossible clues without changing the frozen catalog or saved query', () => {
  const catalog = searchCatalog();
  const search = catalog.familySearch!;
  const historical = search.catalog.families[0];
  historical.capabilityTags.technique.push('project-inference');
  search.catalog.families.push({...historical, id: 'rank', capabilityTags: {
    ...historical.capabilityTags, technique: ['project-inference']}});
  catalog.lessonTags!.axes.find((axis) => axis.id === 'technique')!.options
    .push({value: 'project-inference', labelKey: 'inference'});
  const properties = (search.requestSchema.properties as {
    tagQuery: {properties: {technique: {items: {enum: string[]}; contains: {enum: string[]}}}}}).tagQuery.properties;
  properties.technique.items.enum.push('project-inference');
  properties.technique.contains.enum.push('project-inference');
  search.liveTechniquesByFamily = {guard: ['guard/protect'], disabled: ['endplay'], rank: ['project-inference']};
  const frozen = structuredClone(search.catalog);
  const query = {technique: ['project-inference']};
  expect(matchingSearchFamilies(catalog, query)).toEqual(['rank']);
  expect(matchingSearchFamilies(catalog, {technique: ['project-inference', 'guard/protect']})).toEqual(['guard', 'rank']);
  expect(familySearchTagCatalog(catalog)!.axes.find((axis) => axis.id === 'technique')!.options.map((row) => row.value))
    .toEqual(['guard/protect', 'project-inference']);
  expect(search.catalog).toEqual(frozen);
  expect(query).toEqual({technique: ['project-inference']});
  search.liveTechniquesByFamily.rank = [];
  const saved = {schemaVersion: 1, catalogHash: search.catalog.sha256, tagQuery: query, sha256: 'c'.repeat(64)};
  expect(readFamilySearchRequest(catalog, saved)).toEqual(saved);
  expect(matchingSearchFamilies(catalog, query)).toEqual([]);
  expect(familySearchTagCatalog(catalog)!.axes.find((axis) => axis.id === 'technique')!.options.map((row) => row.value))
    .toEqual(['guard/protect']);
  expect(frozenFamilySearch({...catalog, familySearch: {...search, liveTechniquesByFamily: undefined}})).toBeNull();
  for (const invalid of [{guard: ['guard/protect']}, {...search.liveTechniquesByFamily, rank: ['invented']},
    {...search.liveTechniquesByFamily, rank: ['project-inference', 'project-inference']}]) {
    expect(frozenFamilySearch({...catalog, familySearch: {...search, liveTechniquesByFamily: invalid}})).toBeNull();
  }
});

test('generation payload is optional and preserves a loaded request without rewriting its authority', () => {
  const catalog = searchCatalog();
  const scope = {origin: 'search', decisionKind: 'play', playingQuality: true};
  const saved = {schemaVersion: 1, catalogHash: 'b'.repeat(64),
    tagQuery: {technique: ['guard/protect']}, sha256: 'c'.repeat(64)};
  expect(readFamilySearchRequest(catalog, undefined)).toBeNull();
  expect(familySearchPayload(catalog, null, scope)).toEqual({});
  const loaded = readFamilySearchRequest(catalog, saved)!;
  expect(familySearchPayload(catalog, loaded, scope)).toEqual({generation: saved});
  loaded.tagQuery.technique.push('endplay');
  expect(saved.tagQuery.technique).toEqual(['guard/protect']);
  for (const invalid of [{...saved, catalogHash: 'd'.repeat(64)}, {...saved, unknown: true},
    {...saved, tagQuery: {level: ['intermediate']}}, {...saved, tagQuery: {invented: ['x']}},
    {...saved, tagQuery: {mode: ['hokum']}}, {...saved, tagQuery: {technique: []}}, null]) {
    expect(() => readFamilySearchRequest(catalog, invalid)).toThrow();
  }
  for (const invalid of [{...scope, origin: 'exact_trace_prefix'}, {...scope, decisionKind: 'bid'}, {...scope, playingQuality: false}]) {
    expect(() => familySearchPayload(catalog, readFamilySearchRequest(catalog, saved), invalid)).toThrow();
  }
});

test('saved OR queries retain known alternatives outside enabled capabilities without widening generation', () => {
  const catalog = searchCatalog();
  const saved = {schemaVersion: 1, catalogHash: 'b'.repeat(64), sha256: 'c'.repeat(64),
    tagQuery: {mode: ['sun', 'hokum'], impact: ['round-flip', 'points-only']}};
  const loaded = readFamilySearchRequest(catalog, saved)!;
  expect(loaded).toEqual(saved);
  expect(familySearchPayload(catalog, loaded, {origin: 'search', decisionKind: 'play', playingQuality: true}))
    .toEqual({generation: saved});
  const visible = familySearchTagCatalog(catalog, loaded.tagQuery)!;
  expect(visible.axes.find((axis) => axis.id === 'mode')!.options.map((option) => option.value)).toEqual(['sun', 'hokum']);
  expect(visible.axes.find((axis) => axis.id === 'impact')!.options.map((option) => option.value)).toEqual(['round-flip', 'points-only']);
  expect(matchingSearchFamilies(catalog, loaded.tagQuery)).toEqual(['guard']);
  expect(() => readFamilySearchRequest(catalog, {...saved, tagQuery: {mode: ['hokum']}})).toThrow();
});

test('saved opening hints are strict, detached, optional, and bounded by frozen metadata', () => {
  const catalog = searchCatalog();
  const saved = {schemaVersion: 1, catalogHash: 'b'.repeat(64), tagQuery: {technique: ['guard/protect']},
    auctionSeeds: [{seed: 0, dealer: 'west'}, {seed: 7, dealer: 'north'}], sha256: 'c'.repeat(64)};
  expect(familySearchAuctionSeedLimit(catalog)).toBe(2);
  const loaded = readFamilySearchRequest(catalog, saved)!;
  expect(loaded).toEqual(saved);
  expect(familySearchPayload(catalog, loaded, {origin: 'search', decisionKind: 'play', playingQuality: true}))
    .toEqual({generation: saved});
  loaded.auctionSeeds![0].seed = 99;
  expect(saved.auctionSeeds[0].seed).toBe(0);
  const {auctionSeeds: _hints, ...withoutHints} = saved;
  expect(readFamilySearchRequest(catalog, withoutHints)).toEqual(withoutHints);
  for (const invalid of [null, [], [{seed: 0, dealer: 'west'}, {seed: 0, dealer: 'west'}],
    [{seed: 0, dealer: 'west'}, {seed: 1, dealer: 'west'}, {seed: 2, dealer: 'west'}],
    ...['0', true, -1, 0.5, Number.MAX_SAFE_INTEGER + 1].map((seed) => [{seed, dealer: 'west'}]),
    [{seed: 0, dealer: 'invented'}], [{seed: 0, dealer: 'west', qualified: true}], [{seed: 0}]]) {
    expect(() => readFamilySearchRequest(catalog, {...saved, auctionSeeds: invalid})).toThrow();
  }
  const sameSeedDifferentDealer = {...saved, auctionSeeds: [{seed: 7, dealer: 'north'}, {seed: 7, dealer: 'south'}]};
  expect(readFamilySearchRequest(catalog, sameSeedDifferentDealer)).toEqual(sameSeedDifferentDealer);
  delete catalog.familySearch!.requestSchema.properties;
  expect(familySearchAuctionSeedLimit(catalog)).toBeNull();
  expect(() => readFamilySearchRequest(catalog, saved)).toThrow();
  expect(readFamilySearchRequest(catalog, withoutHints)).toEqual(withoutHints);
});

test('opening picker uses only terminal same-bundle public hints and keeps loaded selections', () => {
  const catalog = searchCatalog();
  const run = {status: 'completed', configurationAuthority: {bundleRevisionId: 'CBR-A'},
    auctionOpenings: [{seed: 1, dealer: 'west'}, {seed: 7, dealer: 'north'}]} as unknown as ResearchRun;
  const selected = [{seed: 9, dealer: 'east'}] as const;
  const choices = familySearchAuctionOpenings(catalog, [run, run,
    {...run, status: 'running', auctionOpenings: [{seed: 2, dealer: 'south'}]},
    {...run, status: 'failed', auctionOpenings: [{seed: 3, dealer: 'south'}]},
    {...run, configurationAuthority: {bundleRevisionId: 'CBR-B'}, auctionOpenings: [{seed: 4, dealer: 'south'}]},
    {...run, configurationAuthority: null, auctionOpenings: [{seed: 5, dealer: 'south'}]},
    {...run, auctionOpenings: [{seed: -1, dealer: 'north'}]},
  ], 'CBR-A', selected);
  expect(choices).toEqual([{seed: 9, dealer: 'east'}, {seed: 1, dealer: 'west'}, {seed: 7, dealer: 'north'}]);
  choices[0].seed = 100;
  expect(selected[0].seed).toBe(9);
  expect(familySearchAuctionOpenings(catalog, [], 'CBR-A', selected)).toEqual(selected);
  expect(familySearchAuctionOpenings(catalog, [run], 'CBR-B')).toEqual([]);
  delete catalog.familySearch!.requestSchema.properties;
  expect(familySearchAuctionOpenings(catalog, [run], 'CBR-A', selected)).toEqual([]);
});

test('manual opening hints bootstrap an empty bundle without changing qualification or trusting a saved hash', () => {
  const catalog = searchCatalog();
  const saved = {schemaVersion: 1 as const, catalogHash: 'b'.repeat(64),
    tagQuery: {technique: ['guard/protect']}, sha256: 'c'.repeat(64)};
  const added = appendFamilySearchAuctionSeed(catalog, saved, {seed: 0, dealer: 'west'});
  expect(added).toEqual({schemaVersion: 1, catalogHash: saved.catalogHash, tagQuery: saved.tagQuery,
    auctionSeeds: [{seed: 0, dealer: 'west'}]});
  expect(familySearchAuctionOpenings(catalog, [], 'CBR-NEW', added.auctionSeeds)).toEqual(added.auctionSeeds);
  expect(saved).toHaveProperty('sha256');
  for (const invalid of [{seed: 0, dealer: 'west'}, {seed: '', dealer: 'north'},
    {seed: 1, dealer: ''}, {seed: 1, dealer: 'north', qualified: true},
    ...[-1, NaN, Infinity, 0.5, Number.MAX_SAFE_INTEGER + 1].map((seed) => ({seed, dealer: 'north'}))]) {
    expect(() => appendFamilySearchAuctionSeed(catalog, added, invalid)).toThrow();
  }
  const full = appendFamilySearchAuctionSeed(catalog, added, {seed: 0, dealer: 'north'});
  expect(full.auctionSeeds).toHaveLength(2);
  expect(() => appendFamilySearchAuctionSeed(catalog, full, {seed: 2, dealer: 'east'})).toThrow();
  expect(familySearchPayload(catalog, full, {origin: 'search', decisionKind: 'play', playingQuality: true}))
    .toEqual({generation: full});
  expect(added.auctionSeeds).toHaveLength(1);
});
