import {expect, type Page} from '@playwright/test';

// Synthetic result documents exercise rendering and navigation of solver receipts.
// The browser still authors, validates, estimates and publishes a recipe on the test server.
export async function useFlagshipCampaign(
  page: Page, {resultCount, active = false}: {resultCount: number; active?: boolean},
) {
  const bundleRevisionId = await page.locator('#research-bundle-select').inputValue();
  const statsResponse = await page.request.get('/api/v2/research-stats');
  await expect(statsResponse).toBeOK();
  const stats = await statsResponse.json();
  const runCode = 'RUN-E2E-FLAGSHIP';
  const runsPath = '/api/v2/research-runs';
  const runPath = `${runsPath}/${runCode}`;
  const resultsPath = '/api/v2/research-results';
  const batchPath = '/api/v2/render-batches/RENDER-E2E-FLAGSHIP';
  const digest = (ordinal: number) => ordinal.toString(16).padStart(64, '0');
  const results = Array.from({length: resultCount}, (_, index) => ({
    resultCode: `LAF-${String(index + 1).padStart(10, '0')}`,
    runCode,
    schemaVersion: 2,
    classification: 'scenario_v4',
    researchBackend: 'ortools_cp_sat_v3',
    qualificationAuthority: 'cp_sat',
    disposition: 'qualified',
    qualification: {failures: []},
    scenarioProjection: {
      schemaVersion: 1, status: 'satisfied', origin: 'search',
      decisionKind: 'play', predicateEvidenceCount: 11, exactStateAvailable: true,
    },
    proofOutcome: {
      requestedMode: 'exact', achievedMode: 'exact', achievedVisibility: 'complete_state',
      claim: 'winning_root_action_count', completionStatus: 'exact_complete',
      winningRootActionCount: 1, claimSatisfied: true,
    },
    runtimeCompatibilityStatus: 'passed',
  }));
  let run: Record<string, unknown> | undefined;
  await page.route((url) => url.pathname === runsPath, async (route) => {
    if (route.request().method() === 'GET') return route.fulfill({json: {items: run ? [run] : []}});

    const {recipeCode} = route.request().postDataJSON();
    expect(recipeCode).toMatch(/^RCP-/);
    const recipeResponse = await page.request.get(`/api/v2/research-recipes/${recipeCode}`);
    await expect(recipeResponse).toBeOK();
    const recipe = await recipeResponse.json();
    expect(recipe.output.resultCount).toBe(active ? 1 : resultCount);
    run = {
      runCode, recipeCode, recipeHash: recipe.sha256, status: active ? 'running' : 'completed',
      progress: active ? 0 : 1, resultCount, selectedCount: resultCount,
      qualifiedCount: resultCount, quarantineCount: 0,
    };
    return route.fulfill({status: 202, json: run});
  });
  await page.route((url) => url.pathname === runPath, (route) => route.fulfill({json: run}));
  await page.route((url) => url.pathname === `${runPath}/stats`, (route) => route.fulfill({json: stats}));
  await page.route((url) => url.pathname === `${runPath}/events`, (route) =>
    route.fulfill({status: 200, contentType: 'text/event-stream', body: ''}));
  await page.route((url) => url.pathname === `${runPath}:cancel`, (route) => {
    expect(route.request().method()).toBe('POST');
    expect(run?.status).toBe('running');
    run = {...run, status: 'cancelled', campaignProgress: {state: 'cancelled', stopReason: 'cancelled'}};
    return route.fulfill({json: run});
  });
  await page.route((url) => url.pathname === resultsPath, (route) => {
    const items = run ? results : [];
    return route.fulfill({json: {items, total: items.length, limit: resultCount, offset: 0, groups: []}});
  });
  for (const [index, result] of results.entries()) {
    const path = `${resultsPath}/${result.resultCode}`;
    await page.route((url) => url.pathname === path, (route) =>
      route.fulfill({json: {...result, recipeCode: run?.recipeCode}}));
    for (const relation of ['decisions', 'renders']) {
      await page.route((url) => url.pathname === `${path}/${relation}`, (route) =>
        route.fulfill({json: {items: []}}));
    }
    await page.route((url) => url.pathname === `${path}/quality-audit`, (route) =>
      route.fulfill({status: 404, json: {detail: 'No quality audit for the synthetic result'}}));
    await page.route((url) => url.pathname === `${path}/operator-audit`, (route) =>
      route.fulfill({json: {
        schemaVersion: 1, resultCode: result.resultCode,
        researchAuthority: {
          schemaVersion: 1, backendId: result.researchBackend, qualificationAuthority: 'cp_sat',
          bundleRevisionId, traceHash: digest(index + 1),
        },
        runtimeCompatibility: {schemaVersion: 1, status: 'passed', failureCodes: []},
        executionReceipt: {
          schemaVersion: 1, semanticResultHash: digest(index + 101),
          stageTimings: {finalNormalization: 0.01},
        },
      }}));
  }

  let batch = {
    jobId: 'RENDER-E2E-FLAGSHIP', status: 'queued',
    request: {items: results.map(({resultCode}) => ({locator: {universalLink: `/c/${resultCode}`}}))},
  };
  await page.route((url) => url.pathname === `${runPath}:render-selected`, (route) => {
    expect(route.request().method()).toBe('POST');
    return route.fulfill({status: 202, json: batch});
  });
  await page.route((url) => url.pathname === batchPath, (route) => route.fulfill({json: batch}));
  await page.route((url) => url.pathname === `${batchPath}:cancel`, (route) => {
    expect(route.request().method()).toBe('POST');
    batch = {...batch, status: 'cancelled'};
    return route.fulfill({json: batch});
  });
}
