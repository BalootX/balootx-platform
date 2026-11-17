import {afterAll, describe, expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

import {decodeQualityAudit, ResearchApiError} from '../../src/baloot_platform/service/foundry_dashboard/src/api/research.js';
import type {ResearchResult, ResearchRun, ResearchStats} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model';

vi.mock('../../src/baloot_platform/service/foundry_dashboard/src/shared/localization', () => ({
  number: new Intl.NumberFormat('en-US'),
  t: (key: string) => key,
  tVariant: (key: string) => key,
  hasFoundryCopy: () => true,
}));
const dashboardRequire = createRequire(new URL('../../src/baloot_platform/service/foundry_dashboard/package.json', import.meta.url));
const {createElement} = dashboardRequire('react') as typeof import('react');
const {renderToStaticMarkup} = dashboardRequire('react-dom/server') as typeof import('react-dom/server');
const effect = vi.spyOn(dashboardRequire('react') as typeof import('react'), 'useEffect')
  .mockImplementation(() => undefined);
afterAll(() => effect.mockRestore());

test.each([
  ['curatedSelection', undefined, 'workflow.research_run_rejected',
    'interestingness_next_best_regret_insufficient, interestingness_baseline_regret_insufficient'],
  ['playingQuality', 'maximum_seconds', 'workflow.research_scenario_support_incomplete', undefined],
  ['graph', 'cancelled', 'workflow.research_scenario_support_incomplete', undefined],
] as const)('labels the retained %s partial receipt by its actual disposition', async (stage, exhaustionReason, label, qualityFailureReason) => {
  const {ResearchRunDetails} = await import('../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit');
  const run: ResearchRun = {runCode: 'RUN-EDITORIAL', recipeCode: 'RCP-EDITORIAL', status: 'completed',
    progress: 1, resultCount: 0, qualifiedCount: 0, quarantineCount: 0,
    scenarioResearchStatus: 'not_found_within_budget', scenarioYieldCounts: {qualified: 0},
    campaignProgress: {schemaVersion: 1, state: 'completed', proofIncomplete: exhaustionReason ? 1 : 0,
      stopReason: exhaustionReason ? 'resource_incomplete' : 'candidate_range_exhausted',
      latestIncompleteCompilation: {stage, ...(exhaustionReason ? {exhaustionReason} : {}),
        ...(qualityFailureReason ? {qualityFailureReason} : {})}}};
  const html = renderToStaticMarkup(createElement(ResearchRunDetails, {run, stats: {} as ResearchStats, busy: false,
    onCancel: async () => undefined, onRetry: async () => undefined, onResults: () => undefined,
    loadRun: async () => run, loadStats: async () => ({} as ResearchStats),
    loadSequentialSearchReceipts: async () => { throw new Error('not needed'); }, PerformanceSummary: () => null}));
  const notice = html.match(/<p class="scenario-launch-block">(.*?)<\/p>/)?.[1];
  expect(notice).toContain(`<strong>${label}</strong>`);
  expect(notice).toContain(stage);
  if (exhaustionReason) expect(notice).toContain(exhaustionReason);
  if (qualityFailureReason) expect(notice).toContain(qualityFailureReason);
});

test('shows localized construction conflict groups without exposing unknown model data', async () => {
  const {ResearchRunDetails} = await import('../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit');
  const run: ResearchRun = {runCode: 'RUN-CONFLICT', recipeCode: 'RCP-CONFLICT', status: 'completed',
    progress: 1, resultCount: 0, qualifiedCount: 0, quarantineCount: 0,
    scenarioResearchStatus: 'not_found_within_budget', scenarioYieldCounts: {qualified: 0},
    events: [{type: 'progress', currentStage: 'family_proposal', constructionStatus: 'seed_miss',
      constructionReason: 'history_expression_infeasible',
      constructionInfeasibleCore: ['decision_actor', 'scenario_expression', 'SA', {privateHand: ['SA']}]}]};
  const html = renderToStaticMarkup(createElement(ResearchRunDetails, {run, stats: {} as ResearchStats, busy: false,
    onCancel: async () => undefined, onRetry: async () => undefined, onResults: () => undefined,
    loadRun: async () => run, loadStats: async () => ({} as ResearchStats),
    loadSequentialSearchReceipts: async () => { throw new Error('not needed'); }, PerformanceSummary: () => null}));
  const events = html.match(/<section class="research-run-events">(.*?)<\/section>/)?.[1];
  expect(events).toContain('workflow.research_scenario_decision_actor');
  expect(events).toContain('workflow.research_scenario_constraints');
  expect(events).toContain('workflow.research_scenario_outcome');
  expect(events).not.toContain('SA');
  expect(events).not.toContain('privateHand');
});

describe('Research all-decision quality audit', () => {
  test.each([20, 0, null, undefined])('shows the stored proof threshold beside its action count (%s)', async (successThreshold) => {
    const {ResearchDecisionAudit} = await import('../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit');
    const result = {schemaVersion: 2, researchBackend: 'ortools_cp_sat_v3', resultCode: 'LAF-THRESHOLD',
      score: 0, confidence: 0, qualificationAuthority: 'cp_sat',
      scenarioProjection: {schemaVersion: 1, status: 'satisfied', exactStateAvailable: true},
      proofOutcome: {completionStatus: 'exact_complete', utility: 'awarded_point_margin',
        successThreshold, winningRootActionCount: 1,
        rootActions: [{action: 'C9', value: 20, satisfies: true}, {action: 'DQ', value: 0, satisfies: false}]},
      libraryProjection: {observerSeats: ['south']}} as ResearchResult;
    const html = renderToStaticMarkup(createElement(ResearchDecisionAudit, {result, decisions: [],
      onPrepare: () => undefined, onVerify: () => undefined, onOpenExactState: () => undefined,
      loadPerspectives: async () => ({items: []})}));
    const summary = html.match(/<section class="research-scenario-result [^"]*">(.*?)<details/s)?.[1];
    expect(summary).toContain('<span>workflow.research_scenario_utility</span><b>workflow.research_scenario_option_utility_awarded_point_margin</b>');
    expect(summary).toContain(`<span>workflow.research_scenario_threshold</span><b>${successThreshold ?? 'workflow.research_metric_unavailable'}</b>`);
    expect(summary).toContain('<span>workflow.research_scenario_winning_actions</span><b>1</b>');
  });

  test.each([
    [1, undefined, true],
    [2, undefined, false],
    [2, 'ortools_cp_sat_v2', false],
    [2, 'ortools_cp_sat_v3', false],
  ] as const)('shows the legacy perspective selector only for supported results (%s, %s)', async (schemaVersion, researchBackend, supported) => {
    const {ResearchDecisionAudit} = await import('../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit');
    effect.mockClear();
    const loadPerspectives = vi.fn(async () => ({items: []}));
    const result = {schemaVersion, researchBackend, resultCode: 'LAF-PERSPECTIVE', score: 0,
      confidence: 0, qualificationAuthority: 'cp_sat',
      scenarioProjection: {schemaVersion: 1, status: 'satisfied', exactStateAvailable: true},
      proofOutcome: {completionStatus: 'exact_complete', winningRootActionCount: 1},
      libraryProjection: {observerSeats: ['west']}} as ResearchResult;
    const html = renderToStaticMarkup(createElement(ResearchDecisionAudit, {result, decisions: [],
      onPrepare: () => undefined, onVerify: () => undefined, onOpenExactState: () => undefined,
      loadPerspectives}));
    const perspectiveEffect = effect.mock.calls.find(([, dependencies]) => dependencies?.includes(loadPerspectives));
    expect(perspectiveEffect).toBeDefined();
    perspectiveEffect![0]();
    expect(loadPerspectives).toHaveBeenCalledTimes(supported ? 1 : 0);
    if (supported) expect(loadPerspectives).toHaveBeenCalledWith(result.resultCode, 'west');
    expect(html.includes('research-perspective-switch')).toBe(supported);
    expect(html.includes('workflow.research_filter_observer_seat')).toBe(supported);
    if (schemaVersion === 2) {
      expect(html).toContain('workflow.research_scenario_proof_boundary');
      expect(html).toContain('exact_complete');
      expect(html).toContain('workflow.research_scenario_open_exact_state');
      expect(html).not.toContain('workflow.research_perspective_empty');
    }
  });

  test('explains an unproven flagship decision without exposing its blocker code', () => {
    const source = readFileSync(fileURLToPath(new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx',
      import.meta.url,
    )), 'utf8');
    expect(source).toContain("critical_decision_not_proven: t('workflow.research_failure_no_critical_point')");
    expect(source).toContain("featured_play_native_proof_unavailable: t('workflow.research_failure_observer_information')");
    expect(source).toContain("failure === 'native_round_win_counterstrategy'");
    expect(source).toContain("operatorCopy('research_failure_counterstrategy')");
  });

  test('keeps the focal audit available when optional quality evidence is absent', () => {
    const source = readFileSync(fileURLToPath(new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx',
      import.meta.url,
    )), 'utf8');
    const parent = readFileSync(fileURLToPath(new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/main.tsx',
      import.meta.url,
    )), 'utf8');
    expect(source).toContain('getResearchResultQualityAudit(result.resultCode).catch(() => null)');
    expect(source).toContain('getResearchResultOperatorAudit(result.resultCode)');
    expect(parent).toContain('recipe={recipe}');
    expect(source).toContain('recipe && recipe.sha256 === recipeHash');
    expect(source).toContain("t('workflow.research_strategic_not_global')");
    expect(source).toContain("<summary>{operatorCopy('research_decision_proof_boundaries')}</summary>");
    expect(source).toContain('getScenarioResearchLocalization(');
    expect(source).toContain('value.researchAuthority.bundleRevisionId');
    expect(source).toContain('proofBoundaryCopyReady ? <details>');
    expect(source).toContain("operatorLocalization?.groups.workflow.research_scenario_support_incomplete");
    expect(source).toContain("proofScopeLabel(decision.proofScope, (key) => operatorLocalization?.groups.workflow[key] || '')");
    expect(source).toContain('decision.proofScope');
    expect(source).toContain("<summary>{t('workflow.research_scenario_requested_proof')}</summary>");
    expect(source).toContain("t('workflow.research_scenario_option_horizon_round_end')");
    expect(source).toContain('analysis.boundaryEvaluator');
    expect(source).toContain("decision.kind === 'play' && decision.legalActionCount === 1");
    expect(source).toContain("operatorCopy('research_scenario_proof_bounded_help')");
    expect(source).toContain('!isNativeV3 && <em>');
    expect(source).toContain("operatorAuditState === 'ready' || operatorAuditState === 'failed'");
    expect(source).toContain("decisionOnlyProof ? 'workflow.research_status_qualified' : 'workflow.research_qualification_passed'");
    expect(source).toContain('decisionOnlyProof && proofScopeCopy && <small>{proofScopeCopy}</small>');
  });

  test('rejects inconsistent counts and hashes', () => {
    const payload = {
      schemaVersion: 1,
      resultCode: 'LAF-QUALITY001',
      resultHash: '1'.repeat(64),
      scope: 'all_decisions',
      status: 'complete',
      wholeRoundCertified: false,
      operation: 'generate',
      qualityReceiptHash: '2'.repeat(64),
      completeTraceHash: '3'.repeat(64),
      decisionCount: 1,
      optimalDecisionCount: 1,
      zeroRegretDecisionCount: 1,
      countsByKind: {bid: 0, escalation: 0, project: 0, play: 1},
      decisions: [{
        ordinal: 1,
        kind: 'play',
        actor: 'north',
        action: 'H7',
        legalActionCount: 1,
        actualActionOptimal: true,
        regret: 0,
        selectionReason: 'native_actor_information_set_best_value',
        proofMode: 'bounded',
        proofVisibility: 'actor_information_set',
        proofCompletionStatus: 'bounded_complete',
        proofScope: 'supplied_worlds_only',
        proofArtifactHash: '4'.repeat(64),
        actionQualityHash: '5'.repeat(64),
        sha256: '6'.repeat(64),
      }],
      sha256: '7'.repeat(64),
    };
    expect(decodeQualityAudit(payload, payload.resultCode)).toEqual(payload);
    const featured = {
      ...payload,
      featuredPlayProof: {
        exactStateId: `STATE-${'A'.repeat(20)}`,
        playIndex: 22,
        actor: 'north',
        action: 'H7',
        proofArtifactHash: '4'.repeat(64),
        proofModelKind: 'native_observed_root_outcome_v1',
        completionStatus: 'bounded_complete',
        hiddenWorlds: {method: 'supplied_worlds_only', count: 6, complete: false},
      },
    };
    expect(decodeQualityAudit(featured, payload.resultCode)).toEqual(featured);
    const {hiddenWorlds, ...featuredIdentity} = featured.featuredPlayProof;
    const complete = {...payload, featuredPlayProof: {...featuredIdentity,
      proofModelKind: 'native_complete_state_root_outcome_v1', completionStatus: 'exact_complete'}};
    expect(decodeQualityAudit(complete, payload.resultCode)).toEqual(complete);
    const exactActor = {...featured, featuredPlayProof: {...featured.featuredPlayProof,
      proofModelKind: 'actor_information_set_cegis_model_v1', completionStatus: 'exact_complete',
      hiddenWorlds: {method: 'exhaustive_cp_sat_hidden_allocations_v1', count: 6, complete: true}}};
    expect(decodeQualityAudit(exactActor, payload.resultCode)).toEqual(exactActor);
    for (const changed of [{hiddenWorlds}, {completionStatus: 'bounded_complete'},
      {hiddenWorlds: null}, {hiddenWorlds: {...exactActor.featuredPlayProof.hiddenWorlds, complete: false}},
      {hiddenWorlds: {...exactActor.featuredPlayProof.hiddenWorlds, worldState: 'private'}}]) {
      expect(() => decodeQualityAudit({...exactActor,
        featuredPlayProof: {...exactActor.featuredPlayProof, ...changed}}, payload.resultCode)).toThrow(ResearchApiError);
    }
    for (const changed of [{hiddenWorlds}, {completionStatus: 'bounded_complete'},
      {proofModelKind: 'native_observed_root_outcome_v1'}]) {
      expect(() => decodeQualityAudit({...complete,
        featuredPlayProof: {...complete.featuredPlayProof, ...changed}}, payload.resultCode)).toThrow(ResearchApiError);
    }
    expect(() => decodeQualityAudit({
      ...featured,
      featuredPlayProof: {...featured.featuredPlayProof, hiddenWorlds: {
        ...featured.featuredPlayProof.hiddenWorlds,
        worldState: 'private',
      }},
    }, payload.resultCode)).toThrow(ResearchApiError);
    for (const invalid of [{...payload, decisionCount: 2}, {...payload, sha256: 'bad'}, {
      ...payload, decisions: [{...payload.decisions[0], legalActionCount: 0}],
    }, {...payload, wholeRoundCertified: true}, {
      ...payload, decisions: [{...payload.decisions[0], proofMode: 'exact'}],
    }]) {
      try {
        decodeQualityAudit(invalid, payload.resultCode);
        throw new Error('decoder accepted an inconsistent audit');
      } catch (error) {
        expect(error).toBeInstanceOf(ResearchApiError);
        expect((error as ResearchApiError).detail).toBe('invalid_quality_audit_response');
      }
    }
  });
});
