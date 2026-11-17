import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

import {describe, expect, test, vi} from 'vitest';

const dashboardRequire = createRequire(new URL(
  '../../src/baloot_platform/service/foundry_dashboard/package.json', import.meta.url,
));

test('research qualification is not replaced by missing publication eligibility', async () => {
  const source = await readFile(new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/main.tsx', import.meta.url,
  ), 'utf8');
  const start = source.indexOf('function unifiedLaffaQualificationLabel(');
  const end = source.indexOf('\nfunction UnifiedLaffaLibrary(', start);
  expect(start).toBeGreaterThan(0);
  expect(end).toBeGreaterThan(start);
  const {transpileModule} = dashboardRequire('typescript');
  const {outputText} = transpileModule(source.slice(start, end), {compilerOptions: {}});
  const label = new Function('t', 'laffaQualificationLabel', `${outputText}\nreturn unifiedLaffaQualificationLabel;`)(
    (key: string) => key,
    {qualified: 'content candidate', rejected: 'rejected'},
  );
  for (const [status, eligible, expected] of [
    ['qualified', false, 'workflow.research_status_qualified'],
    ['qualified', true, 'content candidate'],
    ['rejected', false, 'rejected'],
    ['legacy_replay', false, 'workflow.laffa_library_qualification_audit'],
  ] as const) {
    const item = Object.freeze({qualificationStatus: status, publicationEligible: eligible});
    expect(label(item)).toBe(expected);
    expect(item.publicationEligible).toBe(eligible);
  }
});

const runRequestFactory = readFile(new URL(
  '../../src/baloot_platform/service/foundry_dashboard/src/features/research/scenarioStudio.tsx', import.meta.url,
), 'utf8').then((source) => {
  const start = source.indexOf('  const runRequest = async');
  const end = source.indexOf('  const loadRecipe = async', start);
  if (start < 0 || end < 0) throw new Error('scenario_request_handler_missing');
  const {transpileModule, ScriptTarget} = dashboardRequire('typescript');
  return transpileModule(source.slice(start, end), {compilerOptions: {target: ScriptTarget.ES2022}}).outputText;
});

describe('Scenario Studio request lifecycle', () => {
  async function setup() {
    let resolve!: (value: object) => void;
    const pending = new Promise<object>((done) => {resolve = done;});
    const context = {
      studioRef: {current: {querySelectorAll: () => []} as object | null},
      fingerprintRef: {current: 'bundle-A-payload'},
      launchBlockReason: null, launchStatusBlocked: false,
      recipePayload: () => ({configuration: {schemaVersion: 1, bundleRevisionId: 'bundle-A'}}),
      validation: {valid: true}, validatedFingerprint: 'bundle-A-payload',
      estimate: {valid: true, launchAdmissible: true}, estimatedFingerprint: 'bundle-A-payload',
      request: vi.fn().mockReturnValueOnce(pending).mockResolvedValue({runCode: 'run-A'}),
      createRecipeAndRun: vi.fn().mockResolvedValue({run: {runCode: 'run-A'}}),
      setValidation: vi.fn(), setValidatedFingerprint: vi.fn(),
      setEstimate: vi.fn(), setEstimatedFingerprint: vi.fn(),
      setActiveRunCode: vi.fn(), setMessage: vi.fn(), setBusy: vi.fn(),
      refresh: vi.fn().mockResolvedValue(undefined), onOpenRun: vi.fn(),
      t: (key: string) => key,
    };
    const handler = new Function(...Object.keys(context), `${await runRequestFactory}\nreturn runRequest;`)(
      ...Object.values(context),
    ) as (action: string) => Promise<void>;
    return {context, handler, resolve};
  }

  for (const action of ['validate', 'estimate', 'run'] as const) {
    test.each(['current', 'changed', 'unmounted'] as const)(`${action} response for %s studio`, async (state) => {
      const {context, handler, resolve} = await setup();
      const completion = handler(action);
      expect(context.request).toHaveBeenCalledTimes(1);
      if (state === 'changed') context.fingerprintRef.current = 'bundle-B-payload';
      if (state === 'unmounted') context.studioRef.current = null;
      resolve({valid: true, launchAdmissible: true});
      await completion;
      const accepted = state === 'current';
      expect(context.setValidation).toHaveBeenCalledTimes(Number(accepted && action === 'validate'));
      expect(context.setEstimate).toHaveBeenCalledTimes(Number(accepted && action !== 'validate'));
      expect(context.createRecipeAndRun).toHaveBeenCalledTimes(Number(accepted && action === 'run'));
      expect(context.onOpenRun).toHaveBeenCalledTimes(Number(accepted && action === 'run'));
      expect(context.setBusy.mock.calls).toEqual([[true], [false]]);
      if (!accepted) expect(context.setMessage).not.toHaveBeenCalled();
    });
  }

  test.each(['creation', 'refresh', 'read'] as const)('does not open an abandoned run after %s', async (stage) => {
    const {context, handler, resolve} = await setup();
    const unmount = () => {context.studioRef.current = null;};
    if (stage === 'creation') context.createRecipeAndRun.mockImplementationOnce(async () => {
      unmount();
      return {run: {runCode: 'run-A'}};
    });
    if (stage === 'refresh') context.refresh.mockImplementationOnce(async () => {unmount();});
    if (stage === 'read') context.request.mockImplementation(async () => {
      unmount();
      return {runCode: 'run-A'};
    });
    const completion = handler('run');
    resolve({valid: true, launchAdmissible: true});
    await completion;
    expect(context.createRecipeAndRun).toHaveBeenCalledTimes(1);
    expect(context.onOpenRun).not.toHaveBeenCalled();
    expect(context.setMessage).not.toHaveBeenCalled();
    expect(context.setBusy.mock.calls).toEqual([[true], [false]]);
  });

  test('does not publish an abandoned request error into the current studio', async () => {
    const {context, handler, resolve} = await setup();
    const completion = handler('run');
    context.studioRef.current = null;
    resolve(Promise.reject(new Error('old-bundle-estimate-failed')));
    await completion;
    expect(context.createRecipeAndRun).not.toHaveBeenCalled();
    expect(context.setMessage).not.toHaveBeenCalled();
    expect(context.setBusy.mock.calls).toEqual([[true], [false]]);
  });
});

test('quality execution is initialized before the required-policy state initializer', async () => {
  const source = await readFile(new URL('../../src/baloot_platform/service/foundry_dashboard/src/features/research/scenarioStudio.tsx', import.meta.url), 'utf8');
  expect(source.indexOf('const qualityExecution =')).toBeGreaterThan(0);
  expect(source.indexOf('const qualityExecution =')).toBeLessThan(source.indexOf('const [playingQuality, setPlayingQuality]'));
});

test('optional opening-entry drafts keep their own validation outside recipe admission', async () => {
  const source = await readFile(new URL('../../src/baloot_platform/service/foundry_dashboard/src/features/research/scenarioStudio.tsx', import.meta.url), 'utf8');
  expect(source).toContain('<form className="scenario-grid" data-research-opening-draft');
  expect(source).toContain(".find((control) => !control.closest('[data-research-opening-draft]'))");
  expect(source).toContain('name="auctionSeed" type="number" required step={1}');
  expect(source).toContain('name="auctionDealer" required defaultValue=""');
});

import type {
  ExactTraceLegalActionV1,
  ResearchRecipe,
  ResearchResult,
  ScenarioPredicateDefinitionV1,
  ScenarioPredicateDefinitionV2,
  ScenarioAuthoringCatalogV1,
  ScenarioResearchV4CapabilityV1,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model.js';
import {
  isNativeResearchAuthoringBackend,
  isOrToolsResearchBackend,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model.js';
import {
  continuationLineBoundaryStatus,
  preferredPlayingQualityBudget,
  playingQualityBudgetDefaultMode,
  scenarioLaunchBlockMessageKey,
  appendExactTraceLegalAction,
  undoRecordedEscalationOpportunity,
  isOrToolsScenarioCapability,
  mergeScenarioPredicateDefinitions,
  preferredScenarioExecutionCombination,
  preferredScenarioRecipeCode,
  preferredNativeResearchBundle,
  projectExistsControlSchema,
  resultAllowsNormalMedia,
  resultHasReplayTrace,
  resultRuntimeCompatibilityStatus,
  scenarioCompiledStages,
  scenarioDecisionSelectorSchema,
  scenarioExecutionCombinations,
  scenarioExecutionContractValues,
  scenarioExecutionSupported,
  scenarioOrdinalDisplayValue,
  scenarioOrdinalWireValue,
  scenarioPredicateArgumentValues,
  scenarioPredicateSupported,
  scenarioProofVisibilitySupported,
  unsupportedScenarioPredicateIds,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/ortoolsScenarioUi.js';

test('scope changes preserve existing governed budgets but mode changes use their defaults', () => {
  const available = ['actor_information_set_exact_v1', 'offline_heavyweight_v1'];
  expect(preferredPlayingQualityBudget('exact', 'exact', 'offline_heavyweight_v1', available,
    'actor_information_set_exact_v1')).toBe('offline_heavyweight_v1');
  expect(preferredPlayingQualityBudget('exact', 'budgeted', 'offline_heavyweight_v1', available,
    'actor_information_set_exact_v1')).toBe('actor_information_set_exact_v1');
  expect(preferredPlayingQualityBudget('exact', 'exact', 'missing', available,
    'actor_information_set_exact_v1')).toBe('actor_information_set_exact_v1');
  expect(preferredPlayingQualityBudget('exact', 'exact', null, available,
    'actor_information_set_exact_v1')).toBe('actor_information_set_exact_v1');
});

test('exact-tail defaults select exact play resources without changing explicit choices', () => {
  expect(playingQualityBudgetDefaultMode('budgeted', 'play', true)).toBe('exact');
  for (const kind of ['bid', 'escalation', 'project']) {
    expect(playingQualityBudgetDefaultMode('budgeted', kind, true)).toBe('budgeted');
  }
  expect(playingQualityBudgetDefaultMode('budgeted', 'play', false)).toBe('budgeted');
  const available = ['bounded_v1', 'offline_heavyweight_v1'];
  expect(preferredPlayingQualityBudget('budgeted', undefined, undefined, available,
    'offline_heavyweight_v1')).toBe('offline_heavyweight_v1');
  expect(preferredPlayingQualityBudget('budgeted', 'budgeted', 'bounded_v1', available,
    'offline_heavyweight_v1')).toBe('bounded_v1');
});

test('unsupported predicates are not presented as exhausted resource budgets', () => {
  expect(scenarioLaunchBlockMessageKey('predicate_cp_sat_unsupported'))
    .toBe('workflow.research_scenario_diagnostic_unsupported_predicate');
  expect(scenarioLaunchBlockMessageKey('bounded_depth_exceeds_proof_budget'))
    .toBe('workflow.research_scenario_estimate_refused');
});

test('continuation counts never inherit exactness from an unrelated completed proof', () => {
  const proof = {claim: 'winning_root_action_count', completionStatus: 'exact_complete'};
  const absent = {complete: false, exactCount: null, lowerBound: '0', upperBound: null};
  expect(continuationLineBoundaryStatus(proof, absent)).toBeNull();
  expect(continuationLineBoundaryStatus({...proof, claim: 'winning_continuation_line_count'}, absent))
    .toBe('proof_incomplete');
  expect(continuationLineBoundaryStatus(proof, {...absent, lowerBound: '2'})).toBe('proof_incomplete');
  expect(continuationLineBoundaryStatus(proof, {...absent, upperBound: '9'})).toBe('proof_incomplete');
  expect(continuationLineBoundaryStatus(proof, {...absent, complete: true, exactCount: '0'}))
    .toBe('exact_complete');
  expect(continuationLineBoundaryStatus({...proof, completionStatus: 'bounded_complete'},
    {...absent, complete: true, exactCount: '4'})).toBe('bounded_complete');
});
import {
  rulesAuthorityIdentity,
  rulesCertificationBoundary,
  semanticRulesDelta,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/ruleAuthority.js';
import type {ConfigBundle, ConfigProfile} from '../../src/baloot_platform/service/foundry_dashboard/src/features/configuration/model.js';
import {configurationEditorArrayOptions, configurationEditorChange, configurationEditorEntries, configurationEditorEnum, configurationEditorEnumTitle, configurationEditorObjectValue, configurationEditorSchema, configurationPreviewComponents} from '../../src/baloot_platform/service/foundry_dashboard/src/features/configuration/model.js';

test('exposes schema-owned tag choices without adding defaults to a frozen profile', () => {
  const schema = {type: 'object', properties: {impact: {type: 'array', uniqueItems: true,
    minItems: 1, items: {type: 'string', enum: ['round-flip', 'points-only']}}}};
  const frozen = {};
  expect(configurationEditorEntries(frozen, schema.properties, schema)).toEqual([['impact', null]]);
  expect(frozen).toEqual({});
  expect(configurationEditorArrayOptions(schema.properties.impact)).toEqual(['round-flip', 'points-only']);
  expect(configurationEditorArrayOptions({type: 'array', items: {enum: ['x']}})).toBeUndefined();
  expect(configurationEditorArrayOptions({type: 'array', uniqueItems: true, items: {enum: [{}]}})).toBeUndefined();
});

test('configuration preview changes only the selected family of an explicit published bundle', () => {
  const base = {status:'published', componentRevisions:{rules:'frozen-rules',challenge:'old-challenge',product:'frozen-copy'}} as ConfigBundle;
  const profile = {status:'published',family:'challenge',revisionId:'new-challenge'} as ConfigProfile;
  expect(configurationPreviewComponents(base,profile)).toEqual({rules:'frozen-rules',challenge:'new-challenge',product:'frozen-copy'});
  expect(base.componentRevisions.challenge).toBe('old-challenge');
  expect(configurationPreviewComponents(undefined,profile)).toBeNull();
  expect(configurationPreviewComponents({...base,status:'draft'},profile)).toBeNull();
  expect(configurationPreviewComponents(base,{...profile,family:'unknown'})).toBeNull();
  expect(configurationPreviewComponents(base,{...profile,status:'retired'})).toBeNull();
  expect(configurationPreviewComponents(base,{...profile,status:'draft'})).toEqual({...base.componentRevisions,challenge:'new-challenge'});
});
import {
  challengePlayerHref,
  renderBatchMediaEntries,
  renderBatchPlayerEntries,
  renderJobReferencesResearch,
  renderJobReferencesSource,
  renderRetryKey,
  type RenderJob,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/videos/model.js';

const hash = 'a'.repeat(64);

test('recipe defaults match the frozen bundle without replacing explicit selections', () => {
  const recipes = [
    {code:'other',schemaVersion:4,configuration:{bundleRevisionId:'CBR-OTHER'}},
    {code:'legacy',schemaVersion:3,configuration:{bundleRevisionId:'CBR-CURRENT'}},
    {code:'matching',schemaVersion:4,configuration:{bundleRevisionId:'CBR-CURRENT'}},
  ] as ResearchRecipe[];
  expect(preferredScenarioRecipeCode(recipes,'','CBR-CURRENT')).toBe('matching');
  expect(preferredScenarioRecipeCode(recipes,'missing','CBR-CURRENT')).toBe('matching');
  expect(preferredScenarioRecipeCode(recipes,'other','CBR-CURRENT')).toBe('other');
  expect(preferredScenarioRecipeCode(recipes,'legacy','CBR-CURRENT')).toBe('legacy');
  expect(preferredScenarioRecipeCode(recipes,'','CBR-NONE')).toBe('');
  expect(preferredScenarioRecipeCode([],'matching','CBR-CURRENT')).toBe('');
});

test('render retry identity changes only after a terminal failure or cancellation', () => {
  for (const status of ['queued', 'running', 'completed']) {
    expect(renderRetryKey('run-one', {jobId:'job-one',status})).toBe('run-one');
  }
  for (const status of ['failed', 'cancelled']) {
    expect(renderRetryKey('run-one', {jobId:'job-one',status})).toBe('run-one-retry-job-one');
    expect(renderRetryKey('run-one', {jobId:'job-two',status})).toBe('run-one-retry-job-two');
  }
});

test('explicit proof version edits round-trip required branch fields without mutating legacy profiles', async () => {
  const schema = JSON.parse(await readFile(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
  const native = schema.properties.scenarioResearchV4.properties.proofBudgets.properties.profiles.properties.bounded_v1.properties.nativeObserved;
  const legacy = {schemaVersion: 1, maximumIterations: 1000, maximumQueries: 5000,
    maximumObservations: 1000, maximumBoundPrefixes: 25000, maximumCachedPrefixesPerBranch: 5000};
  expect(configurationEditorEntries(legacy, native.properties)).toEqual(Object.entries(legacy));
  const upgraded = configurationEditorChange(legacy, native, 'schemaVersion', 2);
  expect(upgraded).toEqual({...legacy, schemaVersion: 2, counterexampleObjective: 'minimum_utility'});
  expect(configurationEditorChange(upgraded, native, 'maximumQueries', 99)).toEqual({...upgraded, maximumQueries: 99});
  expect(configurationEditorChange(upgraded, native, 'schemaVersion', 1)).toEqual(legacy);
  const batched = configurationEditorChange(upgraded, native, 'schemaVersion', 3);
  expect(batched).toEqual({...upgraded, schemaVersion: 3, playAllocationMethod: 'deterministic_cp_sat_batched_prefix_v1'});
  expect(configurationEditorChange(batched, native, 'schemaVersion', 2)).toEqual(upgraded);
  expect(configurationEditorChange(batched, native, 'schemaVersion', 1)).toEqual(legacy);
  expect(legacy).not.toHaveProperty('counterexampleObjective');
  const ambiguous = {...native, oneOf: [native.oneOf[1], native.oneOf[1]]};
  expect(configurationEditorChange(legacy, ambiguous, 'schemaVersion', 2)).not.toHaveProperty('counterexampleObjective');
  expect(configurationEditorChange(legacy, {oneOf: [{properties: {schemaVersion: {const: 2}}, required: ['unknown'], not: {required: ['maximumQueries', 'maximumIterations']}}]}, 'schemaVersion', 2))
    .toEqual({...legacy, schemaVersion: 2});
});

test('shows optional enum controls without inserting them into frozen profiles', () => {
  const value = {maximumWallSeconds: 1800};
  const schema = {properties: {policySearch: {type: 'string', enum: ['time_budgeted_symbolic_v1']}}, required: []};
  expect(configurationEditorEntries(value, schema.properties, schema)).toEqual([
    ['maximumWallSeconds', 1800], ['policySearch', null],
  ]);
  expect(value).toEqual({maximumWallSeconds: 1800});
  expect(configurationEditorChange(value, schema, 'policySearch', 'time_budgeted_symbolic_v1'))
    .toEqual({...value, policySearch: 'time_budgeted_symbolic_v1'});
});

test('overlapping challenge schemas do not offer native controls to a frozen legacy profile', async () => {
  const schema = JSON.parse(await readFile(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
  expect(schema.anyOf.length).toBeGreaterThan(1);
  const frozen = {schema_version: 1, profile: {id: 'progressive_challenge_mining_v2', version: '2.8.4'}};
  expect(configurationEditorEntries(frozen, schema.properties, schema)).toEqual(Object.entries(frozen));
  expect(frozen).not.toHaveProperty('scenarioResearchV4');
});

test.each(['oneOf', 'anyOf'])('only an unambiguous %s branch offers missing controls', (keyword) => {
  const branches = [
    {properties: {version: {const: 1}, legacy: {type: 'object', properties: {}}}},
    {properties: {version: {const: 2}, native: {type: 'object', properties: {}}}},
  ];
  const schema = {properties: {...branches[0].properties, ...branches[1].properties}, [keyword]: branches};
  expect(configurationEditorEntries({version: 1}, schema.properties, schema)).toEqual([
    ['version', 1], ['legacy', null],
  ]);
  expect(configurationEditorEntries({version: 1}, schema.properties, {...schema, [keyword]: [branches[0], branches[0]]}))
    .toEqual([['version', 1]]);
});

test('Sun tie control uses schema choices and preserves omitted frozen values until an explicit edit', async () => {
  const source = await readFile(new URL('../../schemas/profiles/rules.schema.json', import.meta.url), 'utf8');
  expect(await readFile(new URL('../../src/baloot_platform/resources/schemas/profiles/rules.schema.json', import.meta.url), 'utf8')).toBe(source);
  const schema = JSON.parse(source);
  for (const profile of [schema, ...schema.oneOf]) {
    const contract = profile.properties.scoring.properties.contract;
    const field = contract.properties.buyer_tie_outcome;
    const frozen = {buyer_team_must_win: true, loss_transfers_all: true};
    const before = JSON.stringify(frozen);
    expect(contract.required).not.toContain('buyer_tie_outcome');
    expect(field.default).toBe('buyer_makes');
    expect(field.title).toBe('تعادل نقاط الصن');
    expect(configurationEditorEnum(field)).toEqual(['buyer_makes', 'buyer_loses']);
    expect(configurationEditorEnumTitle('buyer_makes', field)).toBe('المشتري ينجح عند التعادل');
    expect(configurationEditorEnumTitle('buyer_loses', field)).toBe('المشتري يخسر عند التعادل');
    expect(configurationEditorEntries(frozen, contract.properties, contract))
      .toContainEqual(['buyer_tie_outcome', null]);
    expect(JSON.stringify(frozen)).toBe(before);
    for (const choice of configurationEditorEnum(field)!) {
      const edited = configurationEditorChange(frozen, contract, 'buyer_tie_outcome', choice);
      expect(edited).toEqual({...frozen, buyer_tie_outcome: choice});
      expect(JSON.parse(JSON.stringify(edited))).toEqual(edited);
    }
    expect(JSON.stringify(frozen)).toBe(before);
  }
  expect(configurationEditorEnumTitle('unknown', {anyOf: [{const: 'buyer_makes', title: 'المشتري ينجح عند التعادل'}]})).toBeUndefined();
});

test('shows omitted declared boolean defaults without changing the frozen configuration', () => {
  const value = {maximumWallSeconds: 600};
  const properties = {
    completeStatePlayPolicyPair: {type: 'boolean', default: false},
    existing: {type: 'boolean', default: true},
    noDefault: {type: 'boolean'},
    invalidDefault: {type: 'boolean', default: 'false'},
  };
  expect(configurationEditorEntries(value, properties)).toEqual([
    ['maximumWallSeconds', 600], ['completeStatePlayPolicyPair', false], ['existing', true],
  ]);
  expect(configurationEditorEntries({...value, completeStatePlayPolicyPair: true, existing: false}, properties)).toEqual([
    ['maximumWallSeconds', 600], ['completeStatePlayPolicyPair', true], ['existing', false],
  ]);
  expect(value).toEqual({maximumWallSeconds: 600});
  expect(configurationEditorEntries(value)).toEqual([['maximumWallSeconds', 600]]);
  const required = {properties, required: ['noDefault']};
  expect(configurationEditorEntries(value, properties, required)).toContainEqual(['noDefault', null]);
  expect(configurationEditorEnum(properties.noDefault)).toEqual([false, true]);
  expect(configurationEditorEnum({type: 'boolean', anyOf: [{const: false}]})).toEqual([false]);
  for (const choice of [false, true]) {
    expect(configurationEditorChange(value, required, 'noDefault', choice))
      .toEqual({...value, noDefault: choice});
  }
  expect(value).not.toHaveProperty('noDefault');
});

test('alternative proof version edits expose missing budgets without inventing defaults', async () => {
  const schema = JSON.parse(await readFile(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
  const alternative = schema.properties.scenarioResearchV4.properties.qualityExecution.properties.bidAlternativeProof;
  const legacy = {schemaVersion: 1, upperBound: 46, maximumSeconds: 60, maximumQueries: 1000,
    maximumWorlds: 3, maximumWorldSeconds: 20, maximumWorldQueries: 300, worldRankingId: 'opponent_auction_value_v1'};
  const shared = configurationEditorChange(legacy, alternative, 'schemaVersion', 2);
  expect(shared).toEqual({schemaVersion: 2, upperBound: 46, maximumSeconds: 60, maximumQueries: 1000,
    method: 'information_set_refutation_v1'});
  expect(configurationEditorEntries(shared, alternative.properties, alternative)).toEqual(Object.entries(shared));
  const restored = configurationEditorChange(shared, alternative, 'schemaVersion', 1);
  expect(restored).toEqual({schemaVersion: 1, upperBound: 46, maximumSeconds: 60, maximumQueries: 1000});
  const entries = configurationEditorEntries(restored, alternative.properties, alternative);
  for (const key of ['maximumWorlds', 'maximumWorldSeconds', 'maximumWorldQueries']) {
    expect(entries).toContainEqual([key, null]);
    expect(restored).not.toHaveProperty(key);
  }
  expect(entries).not.toContainEqual(['worldRankingId', null]);
  expect(legacy.maximumWorlds).toBe(3);
});

test('absent optional proof groups are visually creatable without default budgets or snapshot changes', async () => {
  const schema = JSON.parse(await readFile(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
  const quality = schema.properties.scenarioResearchV4.properties.qualityExecution;
  const frozen = {maximumWallSeconds: 600};
  expect(configurationEditorEntries(frozen, quality.properties, quality)).toContainEqual(['bidAlternativeProof', null]);
  const alternative = quality.properties.bidAlternativeProof;
  const created = configurationEditorObjectValue(alternative);
  expect(created).toEqual({});
  expect(configurationEditorEntries(created, alternative.properties, alternative)).toEqual([
    ['maximumQueries', null], ['maximumSeconds', null], ['schemaVersion', null], ['upperBound', null],
  ]);
  expect(configurationEditorEnum(alternative.properties.schemaVersion)).toEqual([1, 2]);
  const selected = configurationEditorChange(created, alternative, 'schemaVersion', 2);
  expect(selected).toEqual({schemaVersion: 2, method: 'information_set_refutation_v1'});
  expect(configurationEditorEntries(selected, alternative.properties, alternative)).toContainEqual(['maximumQueries', null]);
  expect(configurationEditorEntries(selected, alternative.properties, alternative)).not.toContainEqual(['maximumWorlds', null]);
  expect(configurationEditorObjectValue(alternative.oneOf[1])).toEqual({schemaVersion: 2, method: 'information_set_refutation_v1'});
  expect(configurationEditorEnum({anyOf:[{const:1}, {type:'integer'}]})).toBeUndefined();
  expect(configurationEditorObjectValue({required:['unknown'],properties:{unknown:{type:'integer',default:99}}})).toEqual({});
  expect(frozen).toEqual({maximumWallSeconds: 600});
});

test('native bid proof order is editable from its schema default without rewriting the snapshot', async () => {
  const schema = JSON.parse(await readFile(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
  const incumbent = schema.properties.scenarioResearchV4.properties.qualityExecution.properties.bidIncumbentProof;
  const frozen = {schemaVersion: 2, maximumSeconds: 20, maximumQueries: 64, nativePlayPolicyId: 'observer_safe_trick_control_v1'};
  expect(configurationEditorEntries(frozen, incumbent.properties, incumbent)).toContainEqual(['completeStateAlternativeFirst', true]);
  const legacy = {schemaVersion: 1, maximumSeconds: 20, maximumQueries: 64, playPolicies: {}};
  expect(configurationEditorEntries(legacy, incumbent.properties, incumbent)).toEqual(Object.entries(legacy));
  expect(configurationEditorEntries(frozen, incumbent.properties, {...incumbent, oneOf: [incumbent.oneOf[1], incumbent.oneOf[1]]})).toEqual(Object.entries(frozen));
  expect(configurationEditorChange(frozen, incumbent, 'completeStateAlternativeFirst', false)).toEqual({...frozen, completeStateAlternativeFirst: false});
  expect(frozen).not.toHaveProperty('completeStateAlternativeFirst');
});

test('editor numeric bounds and required controls follow the selected frozen policy version', async () => {
  const schema = JSON.parse(await readFile(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
  const policy = schema.properties.scenarioResearchV4.properties.qualityExecution.properties.videoInterestingness;
  const frozen = {schemaVersion: 1, minimumPointSwing: 20, maximumFeaturedTrick: 5, minimumMeaningfulDecisions: 2, maximumFinalMargin: 20};
  const original = JSON.stringify(frozen);
  const fields = configurationEditorSchema(frozen, policy)!;
  expect(fields.properties.minimumPointSwing.minimum).toBe(20);
  expect(fields.properties.maximumFeaturedTrick.maximum).toBe(5);
  expect(fields.properties.minimumMeaningfulDecisions.minimum).toBe(2);
  expect(fields.required).toContain('minimumPointSwing');
  const relaxed = configurationEditorSchema({...frozen, schemaVersion: 2}, policy)!;
  expect(relaxed.properties.minimumPointSwing.minimum).toBe(15);
  expect(relaxed.properties.maximumFeaturedTrick.maximum).toBe(6);
  expect(relaxed.properties.minimumMeaningfulDecisions.minimum).toBe(1);
  const late = configurationEditorSchema({...frozen, schemaVersion: 3}, policy)!;
  expect(late.properties.maximumFeaturedTrick.maximum).toBe(7);
  expect(late.properties.minimumPointSwing.minimum).toBe(20);
  expect(late.properties.minimumMeaningfulDecisions.minimum).toBe(1);
  expect(JSON.stringify(frozen)).toBe(original);
  expect(configurationEditorSchema(frozen, {...policy, oneOf: [policy.oneOf[0], policy.oneOf[0]]})).toEqual({...policy, oneOf: [policy.oneOf[0], policy.oneOf[0]]});
});

test('defaults to the newest published native Research bundle without changing bundle order', () => {
  const bundles = [
    {bundleId: 'baloot_default', version: '21.1.22'},
    {bundleId: 'baloot_ortools_research_native_v3', version: '24.0.1'},
    {bundleId: 'baloot_ortools_research_native_v3_intelligent', version: '24.2.1'},
    {bundleId: 'baloot_ortools_research_native_v3_optional_project', version: '1.1.1'},
  ] as ConfigBundle[];
  expect(preferredNativeResearchBundle(bundles)?.version).toBe('24.2.1');
  expect(bundles.map((bundle)=>bundle.version)).toEqual(['21.1.22', '24.0.1', '24.2.1', '1.1.1']);
});

type RealScenarioCatalog = {
  authoringCatalog: ScenarioAuthoringCatalogV1;
  definitions: ScenarioPredicateDefinitionV2[];
  manifestHash: string;
};

function loadRealScenarioCatalog(): RealScenarioCatalog {
  const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
  const source = [
    'import json, yaml',
    'from pathlib import Path',
    'from baloot_platform.adjudication.research import AdjudicationResearchAuthorityV1',
    'from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2',
    'from baloot_platform.research.scenario_execution_capabilities import build_scenario_execution_capabilities_v1',
    "root = Path('.')",
    "profile = yaml.safe_load((root / 'configs/challenge/progressive_mining_ortools_v3_1.yaml').read_text(encoding='utf-8'))",
    "adjudication_document = yaml.safe_load((root / 'configs/adjudication/research_events_v2.yaml').read_text(encoding='utf-8'))",
    'adjudication = AdjudicationResearchAuthorityV1.from_mapping(adjudication_document)',
    "backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity={'bundleRevisionId':'CBR-FRONTEND-REAL-CATALOG','semanticBundleHash':'a'*64,'effectiveRuntimeConfigHash':'b'*64}, adjudication_authority=adjudication_document)",
    'execution = build_scenario_execution_capabilities_v1(backend, adjudication_authority=adjudication)',
    'manifest = backend.lowering_manifest',
    "print(json.dumps({'authoringCatalog':execution['authoringCatalog'],'definitions':[item.to_dict() for item in manifest.exposed_definitions],'manifestHash':manifest.manifest_hash}, ensure_ascii=False))",
  ].join(';');
  return JSON.parse(execFileSync('python', ['-c', source], {
    cwd: repositoryRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      PYTHONPATH: `${repositoryRoot}src`,
    },
  })) as RealScenarioCatalog;
}

const realScenarioCatalog = loadRealScenarioCatalog();

test('binds the selected rules identity, source boundary, and semantic-only variant delta', () => {
  const profile = (changes: Partial<ConfigProfile> = {}): ConfigProfile => ({
    revisionId: 'CPR-RULES-VARIANT',
    family: 'rules',
    profileId: 'saudi_project_optional_unratified_v1',
    version: '1.0.0',
    scope: 'global',
    scopeKey: '*',
    status: 'published',
    payloadSha256: hash,
    sequence: 1,
    etag: 1,
    updatedAt: '2026-09-02T00:00:00Z',
    payload: {profile: {
      id: 'saudi_project_optional_unratified_v1',
      version: '1.0.0',
      status: 'published_unratified_configured_variant',
      expert_ratified: false,
    }},
    ...changes,
  });
  const identity = rulesAuthorityIdentity(profile(),'CPR-RULES-VARIANT',hash);
  expect(identity).toEqual({
    profileId: 'saudi_project_optional_unratified_v1',
    version: '1.0.0',
    profileStatus: 'published_unratified_configured_variant',
    expertRatified: false,
    payloadSha256: hash,
  });
  expect(rulesCertificationBoundary(profile({
    revisionId: 'CPR-CERT-VARIANT',
    family: 'certification',
    profileId: 'rules_certification_saudi_project_optional_unratified_v1',
    payload: {
      profile: {status: 'configured_variant_not_certification_eligible'},
      certification_eligible: false,
      candidate_profile_identity: {
        id: identity.profileId,
        version: identity.version,
        sha256: identity.payloadSha256,
      },
      rule_source_catalog: {id: 'variant_sources',version: '1.0.0',sha256: hash},
    },
  }),identity)).toEqual({
    status: 'configured_variant_not_certification_eligible',
    certificationEligible: false,
    sourceCatalog: {id: 'variant_sources',version: '1.0.0',sha256: hash},
  });
  expect(semanticRulesDelta({
    left: profile({revisionId:'CPR-RULES-OFFICIAL'}),
    right: profile(),
    changes: [
      {path:'profile.id',change:'changed',left:'official',right:'variant'},
      {path:'projects.declaration_mode',change:'changed',left:'mandatory',right:'optional'},
    ],
  })).toEqual([
    {path:'projects.declaration_mode',change:'changed',left:'mandatory',right:'optional'},
  ]);
});

test('declares the complete incremental campaign receipt and live refresh path', async () => {
  const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
  const source = await readFile(
    `${repositoryRoot}src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx`,
    'utf8',
  );
  const studioSource = await readFile(
    `${repositoryRoot}src/baloot_platform/service/foundry_dashboard/src/features/research/scenarioStudio.tsx`,
    'utf8',
  );
  expect(source).toContain('run.campaignAccounting || run.campaignProgress');
  for (const field of [
    'requestedCandidates',
    'candidateAttempts',
    'constructedCandidates',
    'structurallyInfeasible',
    'semanticPostconditionMiss',
    'proofIncomplete',
    'proofFalse',
    'deepProofRetries',
    'provisionalQualified',
    'provisionalResultHashes',
    'provisionalResults',
    'qualified',
    'elapsedSeconds',
    'remainingCampaignSeconds',
    'latestIncompleteCompilation',
    'stopReason',
  ]) expect(source).toContain(`value.${field}`);
  expect(source).toContain('value.provisionalResultHashes.map');
  expect(source).toContain('value.provisionalResults.map');
  expect(source).toContain('workflow.research_scenario_exact_state_id');
  expect(source).toContain('incomplete.exhaustionReason');
  expect(studioSource).toContain('estimate.maximumCampaignSeconds ?? estimate.maximumSolverSeconds');
  expect(studioSource).toContain("t('workflow.research_scenario_search_budgets')");
  expect(studioSource).toContain("t('workflow.research_estimate_candidates')} · {t('workflow.research_estimate_shards')}");
  expect(studioSource).toContain('Number(timing.total??Object.values(timing)');
  expect(studioSource).toContain("playingQuality?.mode==='budgeted' && qualityBoundaryCopy ?");
  expect(source).toContain('const stopReasonLabel = researchRunStatusLabel(run)');
  expect(source).toContain('workflow.research_sequential_stop_candidates_exhausted');
  expect(source).toContain("t('settings.compiler_coverage_remaining')");
  expect(source).toContain("['queued','running'].includes(run.status) && <div><span>{t('workflow.research_estimate_solver')} · <code dir=\"ltr\">campaign</code> · {t('settings.compiler_coverage_remaining')}");
  expect(studioSource).toContain("['queued','running'].includes(activeRunStatus) ? <div><dt>{t('workflow.research_estimate_solver')} · <code dir=\"ltr\">campaign</code> · {t('settings.compiler_coverage_remaining')}");
  expect(source).not.toContain("t('workflow.compiler_coverage_remaining')");
  expect(source).toContain('aria-live="polite"');
  expect(source).toContain('new EventSource(`/api/v2/research-runs/${encodeURIComponent(run.runCode)}/events`)');
  expect(source).toContain("['completed', 'failed', 'cancelled']");
  expect(source).toContain('if (!document.hidden) refresh()');
  expect(source).toContain("if (withStats || (nextRun.status !== 'queued' && nextRun.status !== 'running'))");
  expect(source).toContain('const nextStats = await loadStats().catch(() => undefined)');
  expect(source).not.toContain('Promise.all([loadRun(), ...(withStats ? [loadStats()] : [])])');
  expect(source).toContain("scenarioStatusLabel('structurally_infeasible')");
  expect(source).toContain("scenarioStatusLabel('semantic_postcondition_miss')");
  expect(studioSource).toContain("scenarioStatusLabel(String(run.scenarioStatus");
  expect(source).not.toContain('<code dir="ltr">structurally_infeasible</code>');
});

test('active research polling does not repeatedly read terminal history or overlap requests', async () => {
  const source = await readFile(new URL('../../src/baloot_platform/service/foundry_dashboard/src/main.tsx', import.meta.url), 'utf8');
  const start = source.indexOf('    if (!activeResearchRunCodes) return;');
  const end = source.indexOf('  }, [activeResearchRunCodes, refresh]);', start);
  expect(start).toBeGreaterThan(-1);
  expect(end).toBeGreaterThan(start);
  const poller = source.slice(start, end);
  expect(poller).not.toContain('/api/v2/research-runs?limit=100');
  expect(poller).toContain('Promise.all(runCodes.map');
  expect(poller).toContain('api<ResearchRun>(`/api/v2/research-runs/${encodeURIComponent(code)}`)');
  expect(poller).toContain('if (inFlight) return;');
  expect(poller).toContain('if (disposed) return;');
  expect(poller).toContain('setResearchRuns((current) => current.map');
  expect(poller).toContain('byCode.get(item.runCode) || item');
  expect(poller).toContain('updates.every');
  expect(poller).toContain('disposed = true; window.clearInterval(timer)');
});

test('library browsing uses projections without eagerly reproving unrelated results', async () => {
  const source = await readFile(new URL('../../src/baloot_platform/service/foundry_dashboard/src/main.tsx', import.meta.url), 'utf8');
  expect(source).not.toContain('/api/v2/research-results?limit=20&sort=score_desc');
  expect(source).not.toContain('setResearchResults');
  expect(source).toContain('api<UnifiedLaffaPage>(`/api/v2/laffat?${query.toString()}`)');
  expect(source).toContain('api<ResearchResult>(`/api/v2/research-results/${encodeURIComponent(code)}`)');
});

test('streams every active render and keeps the open receipt live', async () => {
  const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
  const source = await readFile(
    `${repositoryRoot}src/baloot_platform/service/foundry_dashboard/src/main.tsx`,
    'utf8',
  );
  expect(source).toContain("renderHistory.some((item) => ['queued', 'running'].includes(item.status))");
  expect(source).toContain("api<{items: RenderJob[]}>('/api/v2/render-batches?limit=20')");
  expect(source).toContain('setRenderHistory(page.items)');
  expect(source).toContain("renderHistory.filter((item) => ['queued', 'running'].includes(item.status))");
  expect(source).toContain("setDrawer((current) => current?.kind === 'render'");
  expect(source).toContain('activeRenders.length > 0 ? activeRenders');
});

test('maps every completed render-batch item and accepts only native player locators', async () => {
  const outputBaseUrl = '/outputs/render-batches/batch-abcdef0123456789';
  const basenames = Array.from({length:5},(_,index)=>`variant-${index+1}`);
  const job: RenderJob = {
    jobId: 'RB-FIVE',
    status: 'completed',
    request: {
      items: basenames.map((basename,index)=>({
        basename,
        source:{code:`LAF-${index+1}`,...(index===1?{auditOnly:true}:{})},
        ...(index===0?{locator:{universalLink:'/c/LAF-A1B2C3D4E5'}}:{}),
      })),
    },
    result: {
      outputBaseUrl,
      batch:{items:basenames.map((basename,ordinal)=>({basename,ordinal}))},
      manifests:basenames.map((basename)=>`${outputBaseUrl}/${basename}-manifest.json`),
    },
  };

  expect(renderBatchMediaEntries(job)).toEqual(basenames.map((basename,index)=>({
    ...(index===1?{auditOnly:true}:{}),
    basename,
    manifestUrl:`${outputBaseUrl}/${basename}-manifest.json`,
    playerHref:index===0?'/c/LAF-A1B2C3D4E5':'',
    posterUrl:`${outputBaseUrl}/${basename}.jpg`,
    sourceCode:`LAF-${index+1}`,
    videoUrl:`${outputBaseUrl}/${basename}.mp4`,
  })));
  expect(challengePlayerHref('/c/BX_SAFE-2')).toBe('/c/BX_SAFE-2');
  expect(challengePlayerHref('/foundry')).toBe('');
  expect(challengePlayerHref('/c/BX?admin=true')).toBe('');
  expect(challengePlayerHref('https://balootx.test/c/BX_SAFE-2')).toBe('');
  expect(challengePlayerHref('balootx://research/LAF-A1B2C3D4E5')).toBe('');
  expect(renderBatchMediaEntries({...job,status:'running'})).toEqual([]);
  expect(renderBatchPlayerEntries({...job,status:'running'})).toEqual([{
    ordinal:0,
    playerHref:'/c/LAF-A1B2C3D4E5',
    sourceCode:'LAF-1',
  }]);
  expect(renderBatchMediaEntries({
    ...job,
    status:'running',
    result:{
      outputBaseUrl,
      readyItems:[{
        ordinal:1,
        basename:'variant-2',
        manifestName:'variant-2-manifest.json',
        manifestSha256:'a'.repeat(64),
      }],
    },
  })).toEqual([{
    auditOnly:true,
    basename:'variant-2',
    manifestUrl:`${outputBaseUrl}/variant-2-manifest.json`,
    playerHref:'',
    posterUrl:`${outputBaseUrl}/variant-2.jpg`,
    sourceCode:'LAF-2',
    videoUrl:`${outputBaseUrl}/variant-2.mp4`,
  }]);
  const studioSource = await readFile(
    `${fileURLToPath(new URL('../../', import.meta.url))}src/baloot_platform/service/foundry_dashboard/src/features/research/scenarioStudio.tsx`,
    'utf8',
  );
  expect(studioSource).toContain("media.auditOnly?'workflow.research_render_research_audit':'workflow.batch_item_ready'");
});

test.each(['research_scenario_result', 'research_result'])(
  'associates completed renders with %s without changing stored source identity', (kind) => {
  const job: RenderJob = {
    jobId: 'RB-V4',
    status: 'completed',
    request: {items: [{source: {kind, code: 'LAF-V4'}}]},
  };
  expect(renderJobReferencesResearch(job, 'LAF-V4')).toBe(true);
  expect(renderJobReferencesResearch(job, 'LAF-OTHER')).toBe(false);
  expect(renderJobReferencesSource(job, kind, 'LAF-V4')).toBe(true);
  expect(renderJobReferencesSource(job, kind === 'research_result' ? 'research_scenario_result' : 'research_result', 'LAF-V4')).toBe(false);
  expect(job.request?.items).toEqual([{source: {kind, code: 'LAF-V4'}}]);
});

test.each(['laffa', 'qualified_laffa', 'unknown'])(
  'does not associate a %s render with a Research result sharing its code', (kind) => {
  const job: RenderJob = {jobId:'RB-OTHER', status:'completed',
    request:{items:[{source:{kind, code:'LAF-V4'}}]}};
  expect(renderJobReferencesResearch(job, 'LAF-V4')).toBe(false);
});

test('shows an existing Research video instead of offering duplicate preparation', async () => {
  const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
  const source = await readFile(
    `${repositoryRoot}src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx`,
    'utf8',
  );
  expect(source).toContain("existingVideoUrl?<a className=\"button-link primary\" href={existingVideoUrl}>{t('workflow.label_open_video')}</a>");
  expect(source).toContain(":<button type=\"button\" className=\"primary\"");
  // Native selected-action certificates can compose universal bounds. Do not
  // label every quality audit as merely a policy-conditioned joint rollout.
  expect(source).toContain("<small>{isNativeV3 && operatorLocalization\n        ? operatorCopy('research_scenario_proof_exact_help')\n        : t('workflow.research_strategic_not_global')}</small>");
  expect(source).toContain('const proofScopeCopy = operatorLocalization && proofScopeKey');
  expect(source).toContain('? operatorCopy(proofScopeKey) : null');
  expect(source).not.toContain("t('workflow.research_scenario_proof_exact_help')");
});

test('wires Research and unified Laffa drawers to non-fatal render association reads', async () => {
  const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
  const source = await readFile(
    `${repositoryRoot}src/baloot_platform/service/foundry_dashboard/src/main.tsx`,
    'utf8',
  );
  expect(source).toContain('/api/v2/research-results/${item.resultCode}/renders`');
  expect(source).toContain('/api/v2/laffat/${encodeURIComponent(item.sourceLocator.code)}/renders`');
  expect(source).toContain('api<JsonObject>(item.sourceLocator.uri), relatedRequest, detailRequest,');
  expect(source).toContain('api<UnifiedLaffa>(`/api/v2/laffat/${encodeURIComponent(item.sourceLocator.code)}`)');
  expect(source).toContain('.catch(() => item)');
  expect(source).toContain('<UnifiedLaffaDetails item={detail} exactSource={exactSource}');
  expect(source).toContain('<UnifiedLaffaDetails item={detail} renders={related}');
  expect(source).toContain('.catch(() => cachedRelated)');
  expect(source).toContain('.catch(() => ({items: []}))');
  expect(source).toContain('existingVideoUrl={existingVideoUrl}');
});

function v1(kind = 'auction.contract'): ScenarioPredicateDefinitionV1 {
  return {
    schemaVersion: 1,
    kind,
    version: 1,
    category: 'auction',
    stage: 'auction',
    argumentSchema: {type: 'object', properties: {contract: {type: 'string'}}},
    messageArKey: `research_scenario_predicate_${kind.replace('.', '_')}`,
    messageEnKey: `research_scenario_predicate_${kind.replace('.', '_')}_en`,
    requiredDomainPrimitives: ['auction.contract'],
    allowedHardness: ['hard', 'prefer'],
    constructionCapability: 'cp_sat_exact',
    runtimeVerifier: 'legacy',
    supportedProofModes: ['exact'],
    hiddenInformation: 'public',
    costClass: 'constant',
    safeComplement: true,
    examples: [],
    counterexamples: [],
    documentation: ['doc'],
    tests: ['test'],
    sha256: hash,
  };
}

function v2(
  kind: string,
  search: 'exact' | 'derived' = 'exact',
  actorInformationSet: 'public' | 'observation' | 'reference_world_only' | 'nonanticipative' = 'nonanticipative',
): ScenarioPredicateDefinitionV2 {
  return {
    schemaVersion: 2,
    kind,
    version: 1,
    category: 'playPrefix',
    stage: 'playPrefix',
    argumentSchema: {type: 'object', properties: {selector: {type: 'string'}}},
    messageArKey: `research_scenario_predicate_${kind.replace('.', '_')}`,
    messageEnKey: `research_scenario_predicate_${kind.replace('.', '_')}_en`,
    requiredDomainPrimitives: ['play.action'],
    allowedHardness: ['hard', 'prefer'],
    cpSatSupport: {
      schemaVersion: 1,
      search: {status: search, stages: ['play', 'proof']},
      exactInput: {status: 'exact', stages: ['play']},
      qualificationSource: search === 'exact' ? 'construction' : 'derived_state',
      actorInformationSet,
      proof: {exact: 'exact', bounded: 'bounds', policyConditioned: 'unsupported'},
      evidenceExtractor: 'evidence.v1',
    },
    costClass: 'solver',
    safeComplement: false,
    examples: [],
    counterexamples: [],
    documentation: ['doc'],
    tests: ['test'],
    lifecycleStatus: 'active',
    sha256: hash,
  };
}

function capability(): ScenarioResearchV4CapabilityV1 {
  const definitions = [v2('auction.contract'), v2('play.action_matches', 'derived')];
  return {
    schemaVersion: 1,
    backendIdentity: {
      schemaVersion: 1,
      backendId: 'ortools_cp_sat_v2',
      backendVersion: '2.0.0',
      qualificationAuthority: 'cp_sat',
      solverVersion: '9.15',
      planner: 'tail_first_v1',
      deterministicExecution: true,
      bundleRevisionId: 'CBR-ORTOOLS',
      bundleIdentityHash: hash,
      authorityHash: hash,
      predicateCatalogHash: realScenarioCatalog.manifestHash,
      loweringManifestHash: realScenarioCatalog.manifestHash,
      sha256: hash,
    },
    executionReady: false,
    launchAdmissible: false,
    supportedOrigins: ['search', 'exact_trace_prefix'],
    supportedProofVisibility: [
      {mode: 'exact', visibility: 'complete_state'},
      {mode: 'exact', visibility: 'actor_information_set'},
      {mode: 'bounded', visibility: 'complete_state'},
      {mode: 'policy_conditioned', visibility: 'complete_state'},
    ],
    executionCapabilities: {
      schemaVersion: 1,
      combinations: [
        ...(['search', 'exact_trace_prefix'] as const).map((origin) => ({
          origin,
          decisionKind: 'bid' as const,
          mode: 'policy_conditioned' as const,
          visibility: 'complete_state' as const,
          claims: [
            'outcome_reachable',
            'outcome_forced',
            'winning_root_action_count',
            'winning_continuation_line_count',
            'selected_action_is_unique_winner',
            'target_team_has_no_salvage',
          ],
          lineUniverses: ['policy_conditioned'],
          strategyClasses: ['frozen_policy'],
          contractValues: null,
        })),
        {
          origin: 'search',
          decisionKind: 'play',
          mode: 'exact',
          visibility: 'complete_state',
          claims: [
            'outcome_forced',
            'winning_root_action_count',
            'winning_continuation_line_count',
          ],
          lineUniverses: ['all_legal', 'optimal_strategy'],
          strategyClasses: ['pure_deterministic'],
          contractValues: null,
        },
        {
          origin: 'exact_trace_prefix',
          decisionKind: 'escalation',
          mode: 'policy_conditioned',
          visibility: 'complete_state',
          claims: ['outcome_forced', 'winning_continuation_line_count'],
          lineUniverses: ['policy_conditioned'],
          strategyClasses: ['frozen_policy'],
          contractValues: null,
        },
        {
          origin: 'exact_trace_prefix',
          decisionKind: 'project',
          mode: 'policy_conditioned',
          visibility: 'complete_state',
          claims: ['outcome_forced', 'winning_continuation_line_count'],
          lineUniverses: ['policy_conditioned'],
          strategyClasses: ['frozen_policy'],
          contractValues: null,
        },
        ...(['escalation', 'project'] as const).flatMap((decisionKind) => (
          (['exact', 'bounded'] as const).map((mode) => ({
            origin: 'exact_trace_prefix' as const,
            decisionKind,
            mode,
            visibility: 'complete_state' as const,
            claims: [
              'outcome_reachable',
              'outcome_forced',
              'winning_root_action_count',
              'selected_action_is_unique_winner',
              'target_team_has_no_salvage',
            ],
            lineUniverses: [],
            strategyClasses: ['pure_deterministic'],
            contractValues: null,
          }))
        )),
        {
          origin: 'exact_trace_prefix',
          decisionKind: 'play',
          mode: 'policy_conditioned',
          visibility: 'complete_state',
          claims: [
            'outcome_reachable',
            'outcome_forced',
            'winning_root_action_count',
            'winning_continuation_line_count',
            'selected_action_is_unique_winner',
            'target_team_has_no_salvage',
          ],
          lineUniverses: ['policy_conditioned'],
          strategyClasses: ['frozen_policy'],
          contractValues: null,
        },
        {
          origin: 'exact_trace_prefix',
          decisionKind: 'play',
          mode: 'exact',
          visibility: 'actor_information_set',
          claims: ['outcome_forced', 'winning_root_action_count'],
          lineUniverses: [],
          strategyClasses: ['pure_deterministic'],
          contractValues: ['sun'],
        },
      ],
      decisionSelectorSchemas: {
        bid: {
          type: 'object',
          additionalProperties: false,
          required: ['actionIndex'],
          properties: {
            actionIndex: {type: 'integer', minimum: 0, maximum: 10_000, default: 0},
          },
        },
        escalation: {
          type: 'object',
          properties: {
            actionIndex: {type: 'integer', minimum: 0, maximum: 10_000, default: 0},
            actingSeat: {type: 'string', enum: ['north', 'west', 'south', 'east'], default: 'south'},
          },
        },
        project: {
          type: 'object',
          properties: {
            owner: {type: 'string', enum: ['north', 'west', 'south', 'east'], default: 'south'},
            projectIndex: {type: 'integer', minimum: 0, maximum: 10_000, default: 0},
          },
        },
        play: {
          type: 'object',
          additionalProperties: false,
          required: ['trickIndex', 'position'],
          properties: {
            trickIndex: {type: 'integer', minimum: 0, maximum: 7, default: 5},
            position: {type: 'integer', minimum: 0, maximum: 3, default: 0},
            remainingCards: {type: 'integer', minimum: 0, maximum: 8, default: 3},
            actingSeat: {type: 'string', enum: ['north', 'west', 'south', 'east']},
          },
        },
      },
      authoringCatalog: structuredClone(realScenarioCatalog.authoringCatalog),
      sha256: hash,
    },
    hardPredicateCatalog: {
      schemaVersion: 2,
      backendIdentityHash: hash,
      loweringManifestHash: realScenarioCatalog.manifestHash,
      origin: 'search',
      proofMode: 'exact',
      visibility: 'complete_state',
      hardness: 'hard',
      definitions,
      sha256: realScenarioCatalog.manifestHash,
    },
    localization: {
      schemaVersion: 1,
      bundleRevisionId: 'CBR-ORTOOLS',
      productRevisionId: 'CPR-PRODUCT',
      productProfileHash: hash,
      locale: 'ar-SA',
      direction: 'rtl',
      groups: {workflow: {research_scenario_authority_title: 'سلطة البحث'}},
      sha256: hash,
    },
    sha256: hash,
  };
}

function result(status?: 'not_checked' | 'passed' | 'failed'): ResearchResult {
  return {
    resultCode: 'RES-1',
    runCode: 'RUN-1',
    score: 1,
    truthLabel: 'exact',
    confidence: 1,
    disposition: 'qualified',
    decisionCount: 1,
    tacticalTags: [],
    ...(status ? {
      researchAuthority: {
        schemaVersion: 1,
        backendId: 'ortools_cp_sat_v2',
        backendVersion: '2.0.0',
        qualificationAuthority: 'cp_sat',
        solverVersion: '9.15',
        bundleRevisionId: 'CBR-ORTOOLS',
        bundleHash: hash,
        domainIrHashes: {rules: hash},
        loweringManifestHash: hash,
        constructionModelHash: hash,
        proofModelHash: hash,
        graphHash: hash,
        traceHash: hash,
        proofOutcomeHash: hash,
        sha256: hash,
      },
      runtimeCompatibility: {
        schemaVersion: 1,
        status,
        engineVersion: status === 'not_checked' ? null : '1.0.0',
        bundleHash: hash,
        traceHash: hash,
        evidenceHash: status === 'not_checked' ? null : hash,
        failureCodes: status === 'failed' ? ['parity_failed'] : [],
        sha256: hash,
      },
    } : {}),
  };
}

describe('OR-Tools Scenario Studio capability integration', () => {
  test('undo preserves public/private escalation alignment and refuses continuation loss', () => {
    const prefix = {sha256: 'stale', escalationActions: [{actor: 'east', action: 'double', mode: 'open'}]};
    const history = [{actor: 'west', actionCode: 'decline'}, {actor: 'east', actionCode: 'double:open'}];
    const undone = undoRecordedEscalationOpportunity(prefix, history);
    expect(undone.prefix).toEqual({escalationActions: []});
    expect(undone.history).toEqual(history.slice(0, 1));
    expect(undoRecordedEscalationOpportunity(undone.prefix, undone.history)).toEqual({prefix: {escalationActions: []}, history: []});
    expect(prefix.escalationActions).toHaveLength(1);
    expect(history).toHaveLength(2);
    for (const field of ['projectActions', 'projectDeclarations', 'playActions']) {
      expect(() => undoRecordedEscalationOpportunity({...prefix, [field]: [{}]}, history)).toThrow('exact_opportunity_has_continuation');
    }
    expect(() => undoRecordedEscalationOpportunity(prefix, [])).toThrow('exact_opportunity_history_empty');
    expect(() => undoRecordedEscalationOpportunity(prefix, [{actor: 'west', actionCode: 'double:open'}])).toThrow('exact_opportunity_public_history_mismatch');
  });
  test('recognizes both immutable OR-Tools Research authorities', () => {
    expect(isOrToolsResearchBackend('ortools_cp_sat_v2')).toBe(true);
    expect(isOrToolsResearchBackend('ortools_cp_sat_v3')).toBe(true);
    expect(isOrToolsResearchBackend('legacy_v4')).toBe(false);
    expect(isNativeResearchAuthoringBackend('ortools_cp_sat_v2')).toBe(false);
    expect(isNativeResearchAuthoringBackend('ortools_cp_sat_v3')).toBe(true);
  });
  test('appends only server-authorized exact trace actions and their project effect', () => {
    const prefix = {
      schemaVersion: 1,
      auctionActions: [],
      escalationActions: [],
      projectDeclarations: [],
      playActions: [],
      sha256: hash,
    };
    const auction = appendExactTraceLegalAction(prefix, {
      stage: 'auction',
      code: 'south:sun',
      actor: 'south',
      action: 'sun',
    });
    expect(auction.auctionActions).toEqual([{actor: 'south', action: 'sun'}]);
    expect(auction).not.toHaveProperty('sha256');
    expect(() => appendExactTraceLegalAction(auction, {
      stage: 'escalation', code: 'decline', actor: 'west', action: 'decline',
    })).toThrow('exact_private_opportunity_requires_history');
    expect(auction.escalationActions).toEqual([]);

    const escalation = appendExactTraceLegalAction(auction, {
      stage: 'escalation',
      code: 'north:double:open',
      actor: 'north',
      action: 'double',
      mode: 'open',
    });
    expect(escalation.escalationActions).toEqual([
      {actor: 'north', action: 'double', mode: 'open'},
    ]);

    const play = appendExactTraceLegalAction(escalation, {
      stage: 'play',
      code: 'south:SA',
      actor: 'south',
      actionId: 31,
      card: {id: 31, code: 'SA', suit: 'spades', rank: 'ace', symbol: 'A♠', label: 'A♠'},
      ekka: false,
      projectDeclarations: [{
        owner: 'south',
        kind: 'sequence_5',
        value: 100,
        highRank: 6,
        declarationOrder: 1,
      }],
    } as ExactTraceLegalActionV1);
    expect(play.playActions).toEqual([{actor: 'south', card: 'SA', ekka: false}]);
    expect(play.projectDeclarations).toEqual([{
      owner: 'south',
      kind: 'sequence_5',
      value: 100,
      highRank: 6,
      declarationOrder: 1,
    }]);
    expect(() => appendExactTraceLegalAction(play, {
      stage: 'play', code: 'bad', actor: 'south',
    })).toThrow('exact_preview_play_action_invalid');
  });

  test('merges V2 support without changing the legacy lane', () => {
    const legacy = [v1()];
    expect(mergeScenarioPredicateDefinitions(legacy, undefined)).toEqual(legacy);
    const selected = capability();
    expect(isOrToolsScenarioCapability(selected)).toBe(true);
    const merged = mergeScenarioPredicateDefinitions(legacy, selected);
    expect(merged.map((item)=>item.kind)).toEqual([
      'auction.contract',
      'play.action_matches',
    ]);
    expect(merged[0]?.cpSatSupport?.search.status).toBe('exact');
    expect(merged[1]?.argumentSchema).toEqual(
      selected.hardPredicateCatalog.definitions[1]?.argumentSchema,
    );
  });

  test('closes project kind and value controls over the selected bundle', () => {
    const definition = realScenarioCatalog.definitions.find(
      (item)=>item.kind==='project.exists',
    )!;
    const schema = projectExistsControlSchema(definition.argumentSchema, [
      {contract:'sun',kind:'siri',value:4},
      {contract:'sun',kind:'hundred_sequence',value:20},
      {contract:'hokum',kind:'fifty',value:10},
    ], 'sun', 'hundred_sequence');
    const properties = schema.properties as Record<string, Record<string, unknown>>;
    expect(properties.kind?.enum).toEqual(['siri','hundred_sequence']);
    expect(properties.value?.enum).toEqual([20]);
    expect(properties.kind?.['x-enumMessageArKeys']).toMatchObject({
      siri:'research_scenario_option_project_siri',
      hundred_sequence:'research_scenario_option_project_hundred_sequence',
    });
    expect(properties.value?.['x-enumMessageArKeys']).toMatchObject({
      '20':'research_scenario_option_project_value_20',
    });
    expect((projectExistsControlSchema(
      definition.argumentSchema,
      [{contract:'sun',kind:'siri',value:4}],
      'sun',
      'hundred_sequence',
    ).properties as Record<string, Record<string, unknown>>).value?.enum).toEqual([]);
  });

  test('maps human trick and play-position ordinals to zero-based wire values', () => {
    expect(scenarioOrdinalDisplayValue('trickIndex',0)).toBe(1);
    expect(scenarioOrdinalDisplayValue('trickIndex',7)).toBe(8);
    expect(scenarioOrdinalWireValue('trickIndex',1)).toBe(0);
    expect(scenarioOrdinalWireValue('trickIndex',8)).toBe(7);
    expect(scenarioOrdinalDisplayValue('position',0)).toBe(1);
    expect(scenarioOrdinalWireValue('position',4)).toBe(3);
  });

  test('drives every definition from the real current generated predicate catalog', () => {
    const selected = capability();
    selected.backendIdentity.predicateCatalogHash = realScenarioCatalog.manifestHash;
    selected.hardPredicateCatalog.sha256 = realScenarioCatalog.manifestHash;
    selected.hardPredicateCatalog.definitions = realScenarioCatalog.definitions;

    const merged = mergeScenarioPredicateDefinitions([], selected);
    expect(selected.backendIdentity.predicateCatalogHash).toBe(realScenarioCatalog.manifestHash);
    expect(merged).toHaveLength(realScenarioCatalog.definitions.length);
    expect(new Set(merged.map((definition) => definition.kind)).size).toBe(
      realScenarioCatalog.definitions.length,
    );
    expect(merged.map((definition) => definition.kind).sort()).toEqual(
      realScenarioCatalog.definitions.map((definition) => definition.kind).sort(),
    );
    expect(merged.every((definition) => definition.messageArKey && definition.messageEnKey)).toBe(true);
    expect(merged.every((definition) => definition.argumentSchema.additionalProperties===false)).toBe(true);
    expect(merged.some((definition) => definition.kind==='play.action_matches')).toBe(true);
    expect(merged.some((definition) => definition.kind==='proof.winning_root_action_matches')).toBe(true);
    for (const kind of [
      'adjudication.out_of_turn_attempt',
      'adjudication.purchase_before_turn_attempt',
    ]) {
      const definition = merged.find((item) => item.kind===kind);
      expect(definition, kind).toBeDefined();
      expect(definition?.allowedHardness).toEqual(['hard']);
      expect(definition?.cpSatSupport?.qualificationSource).toBe('construction');
    }
    const flagship = merged.filter((definition) => [
      'play.action_matches',
      'play.trick_winner_matches',
      'play.trick_card_count',
      'proof.winning_root_action_matches',
    ].includes(definition.kind));
    expect(flagship).toHaveLength(4);
    for (const definition of flagship) {
      expect(definition.allowedHardness).toContain('hard');
      expect(scenarioPredicateSupported(definition, {
        origin:'search', proofMode:'exact', visibility:'complete_state',
      }, 'hard')).toBe(true);
      expect(scenarioPredicateSupported(definition, {
        origin:'exact_trace_prefix', proofMode:'exact', visibility:'complete_state',
      }, 'hard')).toBe(true);
    }
  });

  test('offers native actor tactics only in their real catalog execution and utility contexts', () => {
    const selected = capability();
    selected.backendIdentity.backendId = 'ortools_cp_sat_v3';
    selected.hardPredicateCatalog.definitions = realScenarioCatalog.definitions;
    const definitions = mergeScenarioPredicateDefinitions([], selected);
    for (const predicateKind of ['tactic.applicable', 'tactic.verified']) {
      const definition = definitions.find((item) => item.kind === predicateKind)!;
      expect(definition, predicateKind).toBeDefined();
      for (const origin of ['search', 'exact_trace_prefix'] as const) {
        const selection = {
          origin, decisionKind: 'play', proofMode: 'exact',
          visibility: 'actor_information_set', utility: 'awarded_point_margin',
        } as const;
        const options = scenarioPredicateArgumentValues(definition, selection, 'kind');
        expect(options).toEqual(['debate', 'only_winning_move', 'small_card_big_gain']);
        for (const kind of options!) {
          expect(scenarioPredicateSupported(definition, selection, 'hard', {kind})).toBe(true);
          for (const unsupported of [
            {...selection, decisionKind: 'bid' as const},
            {...selection, proofMode: 'bounded' as const},
            {...selection, proofMode: 'policy_conditioned' as const},
          ]) {
            expect(scenarioPredicateSupported(definition, unsupported, 'hard', {kind})).toBe(false);
            expect(scenarioPredicateArgumentValues(definition, unsupported, 'kind')).toEqual([]);
          }
        }
        for (const [utility, kinds] of [
          ['kabout', ['kabout_creation', 'kabout_preservation', 'kabout_prevention']],
          ['contract_success', ['contract_rescue']],
        ] as const) {
          const supported = {...selection, utility};
          for (const kind of kinds) {
            expect(scenarioPredicateArgumentValues(definition, supported, 'kind')).toContain(kind);
            expect(scenarioPredicateSupported(definition, supported, 'hard', {kind})).toBe(true);
            expect(scenarioPredicateSupported(definition, selection, 'hard', {kind})).toBe(false);
          }
        }
        for (const kind of [
          'entry_unblocking', 'partner_enablement', 'trump_control',
          'large_card_sacrifice', 'project_bidding',
        ]) {
          expect(scenarioPredicateSupported(definition, selection, 'hard', {kind})).toBe(false);
        }
      }
    }
  });

  test('filters proof, visibility, origin, and predicate hardness truthfully', () => {
    const selected = capability();
    const merged = mergeScenarioPredicateDefinitions([v1()], selected);
    const exactActor = {
      origin: 'search',
      proofMode: 'exact',
      visibility: 'actor_information_set',
    } as const;
    expect(scenarioProofVisibilitySupported(selected, 'exact', 'actor_information_set')).toBe(true);
    expect(scenarioProofVisibilitySupported(selected, 'bounded', 'actor_information_set')).toBe(false);
    expect(scenarioPredicateSupported(merged[0]!, exactActor, 'hard')).toBe(true);
    expect(scenarioPredicateSupported(merged[0]!, {...exactActor, proofMode: 'policy_conditioned'}, 'hard')).toBe(false);
    expect(unsupportedScenarioPredicateIds(merged, [
      {predicateId: 'PRED-1', kind: 'auction.contract', hardness: 'hard'},
      {predicateId: 'PRED-2', kind: 'missing.kind', hardness: 'hard'},
    ], exactActor)).toEqual(['PRED-2']);
    expect(scenarioCompiledStages(merged, [
      {kind: 'auction.contract', hardness: 'hard'},
      {kind: 'play.action_matches', hardness: 'hard'},
    ], 'search')).toEqual(['play', 'proof']);

    const observerSafe = v2('state.observer_visibility', 'derived', 'nonanticipative');
    const referenceWorldOnly = v2('state.hidden_world_reference', 'derived', 'reference_world_only');
    expect(scenarioPredicateSupported(observerSafe, exactActor, 'hard')).toBe(true);
    expect(scenarioPredicateSupported(referenceWorldOnly, exactActor, 'hard')).toBe(false);
    expect(scenarioPredicateSupported(referenceWorldOnly, {
      ...exactActor,
      visibility: 'complete_state',
    }, 'hard')).toBe(true);
  });

  test('exposes only executable decision/proof/count combinations and governed selector bounds', () => {
    const selected = capability();
    expect(scenarioExecutionCombinations(selected, {origin: 'search'})).toHaveLength(2);
    expect(scenarioExecutionCombinations(selected, {
      origin: 'search',
      decisionKind: 'bid',
    })).toHaveLength(1);
    expect(scenarioExecutionSupported(selected, {
      origin: 'search',
      decisionKind: 'bid',
      proofMode: 'policy_conditioned',
      visibility: 'complete_state',
    }, 'winning_root_action_count')).toBe(true);
    expect(scenarioExecutionCombinations(selected, {
      origin: 'exact_trace_prefix',
      decisionKind: 'escalation',
      proofMode: 'policy_conditioned',
      visibility: 'complete_state',
    })).toHaveLength(1);
    expect(scenarioExecutionCombinations(selected, {
      origin: 'exact_trace_prefix',
      decisionKind: 'project',
      proofMode: 'policy_conditioned',
      visibility: 'complete_state',
    })).toHaveLength(1);
    expect(scenarioExecutionSupported(selected, {
      origin: 'exact_trace_prefix',
      decisionKind: 'project',
      proofMode: 'policy_conditioned',
      visibility: 'complete_state',
    }, 'winning_continuation_line_count', 'policy_conditioned')).toBe(true);
    for (const decisionKind of ['escalation', 'project'] as const) {
      for (const proofMode of ['exact', 'bounded'] as const) {
        const selection = {
          origin: 'exact_trace_prefix',
          decisionKind,
          proofMode,
          visibility: 'complete_state',
        } as const;
        expect(scenarioExecutionSupported(
          selected,
          selection,
          'winning_root_action_count',
        )).toBe(true);
        expect(scenarioExecutionSupported(
          selected,
          selection,
          'winning_continuation_line_count',
          'all_legal',
        )).toBe(false);
      }
    }
    expect(scenarioExecutionSupported(selected, {
      origin: 'search',
      decisionKind: 'play',
      proofMode: 'exact',
      visibility: 'complete_state',
    }, 'winning_continuation_line_count', 'all_legal')).toBe(true);
    expect(scenarioExecutionSupported(selected, {
      origin: 'search',
      decisionKind: 'play',
      proofMode: 'exact',
      visibility: 'complete_state',
    }, 'winning_continuation_line_count', 'policy_conditioned')).toBe(false);
    expect(scenarioExecutionSupported(selected, {
      origin: 'search',
      decisionKind: 'play',
      proofMode: 'exact',
      visibility: 'complete_state',
    }, 'winning_root_action_count')).toBe(true);
    expect(scenarioExecutionSupported(selected, {
      origin: 'exact_trace_prefix',
      decisionKind: 'play',
      proofMode: 'policy_conditioned',
      visibility: 'complete_state',
    }, 'winning_root_action_count')).toBe(true);
    expect(scenarioExecutionSupported(selected, {
      origin: 'exact_trace_prefix',
      decisionKind: 'play',
      proofMode: 'policy_conditioned',
      visibility: 'complete_state',
    }, 'target_team_has_no_salvage')).toBe(true);
    expect(scenarioDecisionSelectorSchema(selected, 'play')).toMatchObject({
      required: ['trickIndex', 'position'],
      properties: {
        trickIndex: {minimum: 0, maximum: 7},
        position: {minimum: 0, maximum: 3},
        remainingCards: {minimum: 0, maximum: 8},
      },
    });
    expect(scenarioDecisionSelectorSchema(selected, 'project')).toMatchObject({
      properties: {projectIndex: {minimum: 0, maximum: 10_000}},
    });
    expect(scenarioDecisionSelectorSchema(selected, 'bid')).toMatchObject({
      required: ['actionIndex'],
      properties: {actionIndex: {minimum: 0, maximum: 10_000}},
    });
    const actorSelection = {
      origin: 'exact_trace_prefix',
      decisionKind: 'play',
      proofMode: 'exact',
      visibility: 'actor_information_set',
    } as const;
    expect(scenarioExecutionContractValues(selected, actorSelection)).toEqual(['sun']);
    expect(scenarioExecutionSupported(
      selected,
      actorSelection,
      'winning_root_action_count',
    )).toBe(true);
    expect(scenarioExecutionSupported(
      selected,
      actorSelection,
      'winning_continuation_line_count',
      'all_legal',
    )).toBe(false);
    expect(scenarioExecutionCombinations(selected, {
      ...actorSelection,
      origin: 'search',
    })).toEqual([]);
    expect(scenarioExecutionCombinations(selected, {
      ...actorSelection,
      decisionKind: 'bid',
    })).toEqual([]);
    expect(unsupportedScenarioPredicateIds([v2('auction.contract')], [{
      predicateId: 'PRED-HOKUM',
      kind: 'auction.contract',
      hardness: 'hard',
      arguments: {contract: 'hokum'},
    }], actorSelection, ['sun'])).toEqual(['PRED-HOKUM']);
  });

  test('keeps the established play search as the operator default when nonplay search is added', () => {
    const selected = capability();
    const play = selected.executionCapabilities.combinations.find((item)=>(
      item.origin==='search'
      && item.decisionKind==='play'
      && item.mode==='exact'
      && item.visibility==='complete_state'
    ))!;
    selected.executionCapabilities.combinations = [
      {
        ...play,
        decisionKind: 'bid',
        mode: 'policy_conditioned',
        lineUniverses: ['policy_conditioned'],
        strategyClasses: ['frozen_policy'],
      },
      {
        ...play,
        decisionKind: 'escalation',
        mode: 'policy_conditioned',
        lineUniverses: ['policy_conditioned'],
        strategyClasses: ['frozen_policy'],
      },
      play,
    ];

    expect(preferredScenarioExecutionCombination(selected)).toMatchObject({
      origin: 'search',
      decisionKind: 'play',
      mode: 'exact',
      visibility: 'complete_state',
    });
  });

  test('permits normal media only after downstream compatibility passes', () => {
    expect(resultRuntimeCompatibilityStatus(result())).toBe('legacy');
    expect(resultAllowsNormalMedia(result())).toBe(true);
    expect(resultAllowsNormalMedia(result('not_checked'))).toBe(false);
    expect(resultAllowsNormalMedia(result('failed'))).toBe(false);
    expect(resultAllowsNormalMedia(result('passed'))).toBe(true);
    const projected = result();
    projected.researchBackend = 'ortools_cp_sat_v3';
    projected.runtimeCompatibilityStatus = 'passed';
    projected.scenarioProjection = {exactStateAvailable: false};
    expect(resultRuntimeCompatibilityStatus(projected)).toBe('passed');
    expect(resultAllowsNormalMedia(projected)).toBe(true);
    expect(resultHasReplayTrace(projected)).toBe(false);
    projected.runtimeCompatibilityStatus = 'failed';
    expect(resultAllowsNormalMedia(projected)).toBe(false);
  });

  test('reports a retained replay trace independently from compatibility', () => {
    const replayable = result('failed');
    replayable.scenarioProjection = {exactStateAvailable: true};
    const noTrace = result('not_checked');
    noTrace.scenarioProjection = {exactStateAvailable: false};
    expect(resultHasReplayTrace(result())).toBe(true);
    expect(resultHasReplayTrace(replayable)).toBe(true);
    expect(resultHasReplayTrace(noTrace)).toBe(false);
  });

  test('keeps OR-Tools controls and audit-media gating explicit in source', async () => {
    const sourceRoot = new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/',
      import.meta.url,
    );
    const [studio, workspace, audit, researchApi, localization, styles, html, main] = await Promise.all([
      readFile(new URL('features/research/scenarioStudio.tsx', sourceRoot), 'utf8'),
      readFile(new URL('features/research/workspace.tsx', sourceRoot), 'utf8'),
      readFile(new URL('features/research/audit.tsx', sourceRoot), 'utf8'),
      readFile(new URL('api/research.ts', sourceRoot), 'utf8'),
      readFile(new URL('shared/localization.ts', sourceRoot), 'utf8'),
      readFile(new URL('styles.css', sourceRoot), 'utf8'),
      readFile(new URL('../index.html', sourceRoot), 'utf8'),
      readFile(new URL('main.tsx', sourceRoot), 'utf8'),
    ]);
    expect(studio).toContain("generationPolicy: 'ortools_cp_sat_v3'");
    expect(studio).toContain("!ortools && origin === 'search'");
    expect(studio).toContain("generationLane: 'ortools'");
    expect(studio).toContain('unsupportedScenarioPredicateIds');
    expect(studio).toContain('estimatedModelVariables');
    expect(studio).toContain('research_scenario_estimate_unavailable');
    expect(studio).toContain("warning==='playing_quality_full_round_cost_not_estimated'");
    expect(studio).toContain("hasFoundryCopy('workflow.research_scenario_estimate_full_round_unknown')");
    expect(studio).toContain("t('workflow.research_scenario_estimate_full_round_unknown')");
    expect(studio).toContain("value === 'not_estimable'");
    expect(studio).toContain('estimate.backendIdentity?.backendId');
    expect(studio).toContain('requestedSolverSeconds > estimate.maximumSolverSeconds');
    expect(studio).toContain('estimate.alternativeProofModes?.length');
    expect(studio).not.toContain('estimate.firstResultCostClass ?? estimate.expectedCostCategory');
    expect(studio).not.toContain('estimate.compiledStages ?? compiledStages');
    expect(studio).toContain('launchBlockReason');
    expect(studio).toContain("const launchStatusBlocked = recipeStatus !== 'published'");
    expect(studio).toContain("disabled={busy||!validationFresh||!estimateFresh||Boolean(launchBlockReason)||launchStatusBlocked}");
    expect(studio).toContain('trickIndex: decisionTrickIndex');
    expect(studio).toContain('playRemainingBounds.maximum');
    expect(studio).toContain('sameMembers(supportedDecisions.map((item)=>item.kind),executableDecisionKinds)');
    expect(studio).toContain('availableClaims.map');
    expect(studio).toContain('availableLineUniverses.map');
    expect(studio).toContain('if (strategyClass)');
    expect(studio).toContain('executableDecisionKinds');
    expect(studio).toContain("pattern={pattern}");
    expect(studio).toContain("items.length<=schema.minItems");
    expect(studio).toContain("minimum={typeof schema.minItems==='number'?schema.minItems:0}");
    expect(studio).toContain('scenario_selector_integer_schema_missing');
    expect(studio).toContain('ortools ? undefined : {minimum: 0, maximum: 7, default: 0}');
    expect(studio).toContain("'x-enumMessageArKeys': schema['x-enumMessageArKeys']");
    expect(studio).toContain('projectExistsControlSchema(');
    expect(studio).not.toContain("!ortools && draft.kind === 'project.exists'");
    expect(studio).toContain("scenarioOrdinalDisplayValue('trickIndex'");
    expect(studio).toContain("scenarioOrdinalDisplayValue('position'");
    expect(studio).toContain('decisionPosition === undefined ? {} : {position: decisionPosition}');
    expect(studio).toContain('scenario_catalog_enum_label_missing');
    expect(studio).toContain('<PredicateCatalogMetadata definition={definition} catalog={authoringCatalog}/>');
    expect(studio).toContain('definition.cpSatSupport?.actorInformationSet??definition.hiddenInformation');
    expect(studio).toContain('catalogValueLabel(authoringCatalog,definition.cpSatSupport.actorInformationSet)');
    expect(studio).toContain('definition.examples');
    expect(studio).toContain('definition.counterexamples');
    expect(studio).toContain('definition.costClass');
    expect(studio).toContain('definition.hiddenInformation');
    expect(studio).toContain('definition.documentation');
    expect(studio).toContain("!ortools && draft.kind === 'escalation.level'");
    expect(studio).not.toContain("seatLabel[text]");
    expect(studio).not.toContain("contractLabel[text]");
    expect(studio).not.toContain('tCatalogWithFallback');
    expect(studio).not.toContain('humanKey');
    expect(studio).toContain('executionCapabilities.authoringCatalog');
    expect(studio).toContain('authoringCatalog.expressionLimits.maximumSerializedBytes');
    expect(studio).toContain('catalog.expressionLimits.maximumCountRange');
    expect(studio).toContain('predicateArgumentErrors(validation,draft.predicateId,name)');
    expect(studio).toContain('fieldPath={fieldPath}');
    expect(studio).toContain('authoringCatalog.policyCatalog.profiles');
    expect(studio).toContain('authoringCatalog.tacticKinds');
    expect(studio).toContain('updatePolicyProfile');
    expect(studio).toContain('stateAssertions.${name}');
    expect(studio).toContain('{allDefinitions.length}');
    expect(studio).toContain("!ortools && decisionKind !== 'play'");
    expect(workspace).toContain('scenarioResearchV4={scenarioResearchV4}');
    expect(workspace).toContain('isNativeResearchAuthoringBackend(scenarioResearchV4?.backendIdentity.backendId)');
    expect(workspace).toContain('selectResearchWorkflowCopy(value.research.scenarioResearchV4?.localization??null,selectedBundleRevisionId)');
    expect(localization).toContain('useSyncExternalStore');
    expect(localization).toContain('researchWorkflowListeners.forEach((listener) => listener())');
    expect(main).toContain('useResearchWorkflowCopyRevision();');
    expect(main).toContain("researchDescription.includes('OR-Tools')");
    expect(audit).toContain('disabled={!observerSelectionReady||!normalMediaAllowed||(isScenarioV4&&!exactStateAvailable)}');
    expect(audit).toContain("disabled={!observerSelectionReady} onClick={() => onPrepare(observerSeat, Number(item.ordinal))}");
    expect(audit).toContain("t('workflow.batch_rejection_no_retained_trace')");
    expect(audit).not.toContain("t('workflow.research_render_audit')");
    expect(audit).not.toContain('isOrToolsResult&&!normalMediaAllowed?<button');
    expect(workspace).toContain('replayTraceAvailable&&normalMediaAllowed');
    expect(audit).toContain('executionReceipt?.stageTimings');
    expect(audit).toContain("stage.replace(/([A-Z])/g, '_$1').toLowerCase()");
    expect(audit).toContain('<code dir="ltr">{stage}</code>');
    expect(studio).toContain("exactStateAvailable: scenario.exactStateAvailable === true ? 'true' : 'false'");
    expect(studio).toContain("researchBackend:'research_backend'");
    expect(studio).toContain("variantCollection:'variant_collection'");
    expect(studio).toContain('setStudioResults(page.items)');
    expect(studio).toContain('Number(activeRun?.resultCount??0)');
    expect(studio).toContain('studioResults.slice(0,Math.max(desiredResultCount,Number(activeRun?.resultCount??0)))');
    expect(studio).toContain("groupFields.filter((field)=>field!=='variantCollection')");
    expect(studio).not.toContain('getResearchResultOperatorAudit(item.resultCode)');
    expect(studio).toContain('activeRunProgress.provisionalResults');
    expect(studio).toContain("t('workflow.research_truth_provisional')");
    expect(studio).toContain('scenario-proof-mode-help');
    expect(studio).toContain('workflow.research_quality_native_from_first');
    expect(studio).toContain('qualityExecution.exactTailProof === true');
    expect(studio).toContain('workflow.research_quality_exact_tail_boundary');
    expect(studio).toContain('qualityExecution.videoInterestingness');
    expect(studio).toContain('workflow.research_quality_video_interest');
    expect(studio).toContain('workflow.research_scenario_proof_exact_help');
    expect(studio).toContain('workflow.research_scenario_proof_bounded_help');
    expect(studio).toContain('workflow.research_scenario_proof_policy_conditioned_help');
    expect(studio).toContain('/api/v2/research-exact-traces:preview-legal-actions');
    expect(studio).toContain('data-exact-stage={action.stage}');
    expect(studio).toContain('result.configurationAuthority.bundleRevisionId !== selectedBundle.revisionId');
    expect(studio).toContain('return version === 3 || (version === 4 && Boolean(recipeBundleRevisionId));');
    expect(studio).toContain("mode === 'load' && String(authority.bundleRevisionId ?? '') !== selectedBundle.revisionId");
    expect(studio).toContain("setRecipeStatus(suppliedDocument || mode !== 'load'");
    expect(studio).toContain('selectedRecipeVersion!==4||!selectedRecipeSameBundle||loadingRecipe');
    expect(studio).toContain('preferredScenarioRecipeCode(items, current, selectedBundle.revisionId)');
    expect(studio).toContain('appendExactTraceLegalAction(value, action)');
    expect(studio).not.toContain('researchAuthority:audit.researchAuthority');
    expect(studio).not.toContain('return {...item,...audit}');
    expect(studio).toContain('/api/v2/render-batches/${encodeURIComponent(renderBatchJobId)}');
    expect(main).toContain("prepared.exportPreset === 'hooked_flagship'");
    expect(main).toContain('playbackEvidence}),');
    expect(studio).not.toContain('renderBatchPollLimit');
    expect(studio).toContain('Number(activeRun?.selectedCount??0) >= requestedVideoCount');
    expect(studio).toContain('controller.abort()');
    expect(studio).toContain("/api/v2/render-batches/${encodeURIComponent(renderBatchJobId)}:cancel");
    expect(studio).toContain("if (selectedAction && !selectedActionOptions.includes(selectedAction))");
    expect(studio).toContain('renderedMedia.map');
    expect(studio).toContain('renderBatchPlayerEntries(renderJob)');
    expect(studio).toContain('data-testid="scenario-prepared-player-link"');
    expect(studio).toContain('data-testid="scenario-player-link"');
    expect(studio).toContain('role="status" aria-live="polite"');
    expect(studio).toContain('type="checkbox" value=');
    expect(audit).toContain('getResearchResultOperatorAudit(result.resultCode)');
    expect(audit).toContain('getResearchResultQualityAudit(result.resultCode)');
    expect(audit).toContain("t('workflow.research_all_decisions_required')");
    expect(audit).toContain('qualityAudit.optimalDecisionCount');
    expect(audit).toContain("!isScenarioV4 && <Stat label={t('workflow.research_column_score')}");
    expect(audit).toContain("!isScenarioV4 && <Stat label={t('workflow.research_column_confidence')}");
    expect(main).toContain("item.classification==='scenario_v4'?'':` · ${number.format(Math.round(item.score))}`");
    expect(main).toContain("item.qualificationStatus === 'qualified' && !item.publicationEligible");
    expect(main).toContain("{unifiedLaffaQualificationLabel(item)}");
    expect(audit).toContain("prepared.qualificationStatus === 'qualified' && !prepared.publicationEligible ? t('workflow.research_status_qualified')");
    expect(audit).toContain("publication_eligibility_not_established: t('workflow.laffa_library_qualification_audit')");
    expect(audit).toContain("t(prepared.qualificationStatus === 'qualified' ? 'workflow.laffa_library_qualification_audit' : 'workflow.laffa_library_audit_only')");
    expect(main).toContain('t(item.qualificationStatus===\'qualified\'?"workflow.laffa_library_qualification_audit":"workflow.laffa_library_audit_only")');
    expect(main).toMatch(/function unifiedLaffaQualificationLabel[\s\S]*?!item\.publicationEligible\)\s*\{\s*return t\('workflow.research_status_qualified'\)/);
    expect(main).toContain("value==='qualified'?t('workflow.research_status_qualified')");
    expect(workspace).toContain("item.classification==='scenario_v4'?'—':<ScorePill value={item.score}/>");
    expect(workspace).toContain("item.classification==='scenario_v4'?'—':`${number.format(Math.round(item.confidence*100))}");
    expect(audit).toContain('maximumQualityRegret');
    expect(audit).toContain("runtimeCompatibilityStatus === 'passed'");
    expect(audit).toContain('isOrToolsResearchBackend(result.researchBackend)');
    expect(audit).toContain("result.qualificationAuthority === 'cp_sat'");
    expect(audit).not.toContain('(result.researchAuthority || {})');
    expect(researchApi).toContain('/operator-audit`');
    expect(researchApi).toContain('decodeOperatorAudit');
    expect(researchApi).toContain('/quality-audit`');
    expect(researchApi).toContain('decodeQualityAudit');
    expect(researchApi).toContain('kindTotal !== count');
    expect(localization).toContain("if (group === 'workflow' && selectedResearchWorkflow)");
    expect(localization).toContain('value.bundleRevisionId !== expectedBundleRevisionId');
    expect(styles).toContain('.scenario-ortools-authority');
    expect(styles).toContain('.research-scenario-authority');
    expect(styles).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)|@media\(prefers-reduced-motion:reduce\)/);
    expect(styles).toContain(':focus-visible');
    expect(html).toContain('<html lang="ar" dir="rtl">');
  });

  test('publishes complete V4 field, relationship, policy, localization, and five-output controls', () => {
    const catalog = realScenarioCatalog.authoringCatalog;
    expect(catalog.recipeSchemaVersion).toBe(4);
    expect(catalog.relationships.map((item)=>item.code)).toEqual(expect.arrayContaining([
      'video_count_fits_results',
      'proof_resource_budget_identity',
      'shards_divide_seed_count',
    ]));
    expect(catalog.fields['output.resultCount']).toMatchObject({minimum:1,maximum:1000});
    expect(catalog.fields['renderPolicy.videoCount']).toMatchObject({minimum:0,maximum:100});
    for (const gate of ['requireObserverSafety','requireAllDecisionAudit']) {
      expect(catalog.fields[`qualification.${gate}`]).toMatchObject({
        control:'boolean', value:true, default:true, mode:'invariant',
      });
    }
    expect(catalog.fields.titleAr).toMatchObject({
      control:'text',
      minimumLength:3,
      maximumLength:160,
      required:true,
      defaultKey:'research_scenario_default_title',
      labelKey:'research_scenario_recipe_title',
      helpKey:'research_scenario_help_title',
    });
    expect(Object.keys(catalog.fields).filter((path)=>path.startsWith('resources.')).sort()).toEqual([
      'resources.proofBudgetProfile',
      'resources.seedCount',
      'resources.seedStart',
      'resources.seedsPerShard',
      'resources.shardWorkerCount',
      'resources.solverTimeoutSeconds',
    ]);
    expect(catalog.fields).not.toHaveProperty('resources.candidatesPerSeed');
    expect(catalog.fields).not.toHaveProperty('resources.maximumConflicts');
    expect(catalog.fields).not.toHaveProperty('resources.analysisMode');
    expect(catalog.policyCatalog.defaultPolicyId).toBe('fixed_action_map_v1');
    expect(catalog.policyCatalog.seed).toEqual({type:'integer',minimum:0,maximum:2147483647});
    expect(Object.keys(catalog.policyCatalog.defaultProfiles)).toEqual([
      'north', 'west', 'south', 'east',
    ]);
    expect(catalog.tacticKinds).toEqual(realScenarioCatalog.definitions
      .find((item)=>item.kind === 'tactic.verified').argumentSchema.properties.kind.enum);
    expect(catalog.tacticKinds).toContain('project_bidding');
    for (const value of catalog.tacticKinds) {
      expect(catalog.localizationKeys.values[value], `tactic:${value}`).toBeTruthy();
    }
    for (const [path, field] of Object.entries(catalog.fields)) {
      expect(field.labelKey, `${path} label`).toBeTruthy();
      expect(field.helpKey, `${path} help`).toBeTruthy();
      for (const value of Array.isArray(field.values) ? field.values : []) {
        expect(catalog.localizationKeys.values[String(value)], `${path}:${String(value)}`).toBeTruthy();
      }
    }
    expect(catalog.expressionLimits).toEqual({
      maximumDepth:8,
      maximumNodes:256,
      maximumChildren:64,
      maximumSerializedBytes:131072,
      maximumCountRange:64,
    });
  });

  test('uses the real CardSelectorV1 schema with governed bounds and unique arrays', () => {
    const action = realScenarioCatalog.definitions.find(
      (definition)=>definition.kind==='play.action_matches',
    );
    expect(action).toBeDefined();
    const selector = (action!.argumentSchema.properties as Record<string, Record<string, unknown>>).selector;
    expect(selector.additionalProperties).toBe(false);
    const properties = selector.properties as Record<string, Record<string, unknown>>;
    for (const name of ['cards','ranks','suits']) {
      expect(properties[name]).toMatchObject({type:'array',uniqueItems:true});
      expect(properties[name]?.minItems).toBeTypeOf('number');
      expect(properties[name]?.maxItems).toBeTypeOf('number');
    }
  });
});
