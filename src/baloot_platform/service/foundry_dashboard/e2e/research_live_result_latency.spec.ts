import {expect, test, type Locator, type Page, type Request} from '@playwright/test';
import {mkdir, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
import {fileURLToPath} from 'node:url';
import {measureFoundryPage} from './foundryPagePerformance';
import {nativeResearchBundleRevision} from './nativeResearchBundle';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../..');
const sampleOutput = process.env.BALOOT_LIVE_RESULT_LATENCY_SAMPLE_OUTPUT
  || resolve(root, 'output/playwright/research-live-result-latency-samples.json');

test('shows safe live quality work without claiming proof', async ({page}) => {
  const runCode = 'RUN-LIVE-QUALITY-WORK';
  const recipeCode = 'RCP-LIVE-QUALITY-WORK';
  const run = {
    runCode, recipeCode, status: 'running', progress: 0,
    resultCount: 0, qualifiedCount: 0, quarantineCount: 0,
    campaignProgress: {
      currentStage: 'playingQuality',
      candidateReceiptCounts: {qualified: 0, rejected: 0, duplicate: 0, incomplete: 0},
      qualitySolver: {phase: 'counterexample', masterCount: 2, queryCount: 3,
        bindingCount: 1, decisionKind: 'bid', proofPhase: 'best'},
    },
  };
  await page.route((url) => url.pathname === '/api/v2/research-runs',
    async (route) => route.fulfill({json: {items: [run]}}));
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}`,
    async (route) => route.fulfill({json: run}));
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/stats`,
    async (route) => route.fulfill({json: {
      performance: {
        shardCount: 0, attemptedSeeds: 0, resultCount: 0, qualifiedCount: 0,
        qualifiedDistinctScenarioCount: null, qualifiedDistinctScenariosPerWallHour: null,
        qualifiedDistinctScenariosPerCpuHour: null, qualifiedPer100Attempts: 0,
        duplicateCount: null, costProxy: {wallSecondsPerQualified: null}, timing: {},
        solverStatusDistribution: {construction: {}, playCompiler: {}},
        operations: {repeatedExecutionCount: 0, currentCancelledShardCount: 0, retryCount: null},
      },
      sequentialSearch: {
        admittedSeedCount: 0, exactRejectionCount: 0, noGoodAppliedCount: 0,
        noGoodUnappliedCount: 0, continuationSolveCount: 0,
        continuationStatusDistribution: {optimal: 0, infeasible: 0},
        stopReasonDistribution: {candidates_exhausted: 0, maximum_solutions_reached: 0,
          qualified: 0, incomplete: 0, cancelled: 0, fatal_quarantine: 0},
      },
    }}));
  await page.route((url) => url.pathname === `/api/v2/research-recipes/${recipeCode}`,
    async (route) => route.fulfill({json: {schemaVersion: 4, code: recipeCode,
      sha256: 'a'.repeat(64), lifecycleStatus: 'published', status: 'published',
      configuration: {bundleVersion: '24.0.0'}}}));
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/events`,
    async (route) => route.fulfill({status: 200, contentType: 'text/event-stream', body: ''}));
  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await page.locator('button.list-item').filter({hasText: runCode}).click();
  for (const viewport of [{width: 1280, height: 900}, {width: 390, height: 844}]) {
    await page.setViewportSize(viewport);
    const panel = page.locator('section.scenario-live-candidates');
    await expect(panel.getByText('استعلامات نُفذت')).toBeVisible();
    await expect(panel.getByText('البحث عن مسار معاكس')).toBeVisible();
    await expect(panel.getByText('تقدّم البحث فقط؛ النتيجة لم تُثبت بعد')).toBeVisible();
    await expect(panel.locator('progress')).toHaveAttribute('value', '0');
    await test.info().attach(`quality-work-${viewport.width}.png`, {
      body: await panel.screenshot(), contentType: 'image/png',
    });
  }
});

test('retains governed live-result latency and Foundry page performance samples', async ({context, baseURL, locale}) => {
  const latenciesMilliseconds: number[] = [];
  const browser = context.browser();
  if (!browser) throw new Error('browser_identity_unavailable');

  for (let sample = 0; sample < 5; sample += 1) {
    const page = await context.newPage();
    const runCode = `RUN-LIVE-LATENCY-${sample + 1}`;
    const recipeCode = `RCP-LIVE-LATENCY-${sample + 1}`;
    const candidateHash = String(sample + 1).repeat(64);
    let eventEmittedAt: number | null = null;
    let detailReads = 0;
    const run = (visible: boolean) => ({
      runCode,
      recipeCode,
      status: 'running',
      progress: 0.5,
      resultCount: visible ? 1 : 0,
      qualifiedCount: visible ? 1 : 0,
      quarantineCount: 0,
      scenarioResearchStatus: 'running',
      scenarioYieldCounts: {attempted: 1, structurallyFeasible: 1, legalReplay: 1, qualified: 0},
      campaignProgress: {
        currentStage: 'proof',
        proofIncomplete: 0,
        candidateReceiptCounts: {qualified: 0, rejected: 0, duplicate: 0, incomplete: 2},
        provisionalQualified: visible ? 1 : 0,
        provisionalResultHashes: visible ? [candidateHash] : [],
        provisionalResults: visible ? [{
          schemaVersion: 1,
          candidateHash,
          exactStateId: 'STATE-00000000000000000000',
          decisionKind: 'play',
          contract: 'sun',
          actingSeat: 'west',
          proofCompletionStatus: 'exact_complete',
          winningRootActionCount: 1,
          selectedAction: 'play:spades:ace',
        }] : [],
      },
    });

    await page.route((url) => url.pathname === '/api/v2/research-runs', async (route) => {
      await route.fulfill({json: {items: [run(false)]}});
    });
    await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}`, async (route) => {
      detailReads += 1;
      await route.fulfill({json: run(eventEmittedAt !== null)});
    });
    await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/stats`, async (route) => {
      await route.fulfill({json: {}});
    });
    await page.route((url) => url.pathname === `/api/v2/research-recipes/${recipeCode}`, async (route) => {
      await route.fulfill({json: {
        schemaVersion: 4,
        code: recipeCode,
        sha256: 'a'.repeat(64),
        lifecycleStatus: 'published',
        status: 'published',
        configuration: {bundleVersion: '24.0.0'},
      }});
    });
    await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/events`, async (route) => {
      eventEmittedAt = performance.now();
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        body: 'id: 1\nevent: progress\ndata: {"status":"running"}\n\n',
      });
    });

    await page.goto('/foundry?section=research');
    const bundle = page.locator('#research-bundle-select');
    await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
    await expect(page.locator('#scenario-studio-title')).toBeVisible();
    await page.locator('button.list-item').filter({hasText: runCode}).click();
    const runDialog = page.getByRole('dialog', {name: `تفاصيل عملية البحث · ${runCode}`});
    await expect(runDialog).toBeVisible();
    await expect(runDialog.locator('.research-sequential-summary .detail-tags')
      .getByText(candidateHash, {exact: true})).toBeVisible();
    const incomplete = page.locator('section.scenario-live-candidates dl > div').filter({
      has: page.getByText('غير مكتمل', {exact: true}),
    });
    await expect(incomplete.locator('dd')).toHaveText(/^[2٢]$/);
    if (eventEmittedAt === null) throw new Error('sse_progress_event_not_observed');
    latenciesMilliseconds.push(Number((performance.now() - eventEmittedAt).toFixed(6)));
    expect(detailReads).toBeGreaterThan(1);
    await page.close();
  }

  const measurement = {
    schemaVersion: 1,
    measurementKind: 'browser_sse_live_result_visibility_v1',
    capturedAt: new Date().toISOString(),
    transport: 'server_sent_events_v1',
    uiPath: '/foundry?section=research',
    eventPathTemplate: '/api/v2/research-runs/{runCode}/events',
    visibleSignal: 'campaign_progress.provisional_result',
    browser: {name: 'chrome', version: browser.version()},
    latenciesMilliseconds,
  };
  let pagePerformance: Awaited<ReturnType<typeof measureFoundryPage>> | undefined;
  try {
    if (!baseURL || !locale) throw new Error('Foundry performance requires the project base URL and locale');
    pagePerformance = await test.step('Measure Foundry overview load and research interaction',
      () => measureFoundryPage(browser, baseURL, locale));
  } finally {
    const receipt = `${JSON.stringify({...measurement, pagePerformance}, null, 2)}\n`;
    await mkdir(dirname(sampleOutput), {recursive: true});
    await writeFile(sampleOutput, receipt, 'utf8');
    await test.info().attach('research-live-result-latency-samples.json', {
      path: sampleOutput,
      contentType: 'application/json',
    });
  }
  const {lcp, inp} = pagePerformance;
  expect.soft(lcp.p75Milliseconds, 'Foundry p75 LCP (ms)').toBeLessThan(lcp.budgetMilliseconds);
  expect.soft(inp.p75Milliseconds, 'Foundry p75 INP (ms)').toBeLessThan(inp.budgetMilliseconds);
});

test('refreshes terminal run stats when a progress event observes completion', async ({page}) => {
  const runCode = 'RUN-TERMINAL-STATS';
  const recipeCode = 'RCP-TERMINAL-STATS';
  const baseStats = await (await page.request.get('/api/v2/research-stats')).json();
  let terminal = false;
  let statsReads = 0;
  const run = () => ({
    runCode,
    recipeCode,
    status: terminal ? 'completed' : 'running',
    progress: terminal ? 1 : 0.5,
    resultCount: terminal ? 1 : 0,
    qualifiedCount: 0,
    quarantineCount: 0,
    events: terminal ? [{type: 'progress', currentStage: 'proofBoundary',
      proofBoundaryStatus: 'infeasible', proofBoundaryReason: 'tail_boundary_infeasible'}] : [],
  });

  await page.route((url) => url.pathname === '/api/v2/research-runs', async (route) => {
    await route.fulfill({json: {items: [run()]}});
  });
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}`, async (route) => {
    await route.fulfill({json: run()});
  });
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/stats`, async (route) => {
    statsReads += 1;
    await route.fulfill({json: {
      ...baseStats,
      performance: {...baseStats.performance, shardCount: 1, attemptedSeeds: terminal ? 4 : 0},
    }});
  });
  await page.route((url) => url.pathname === `/api/v2/research-recipes/${recipeCode}`, async (route) => {
    await route.fulfill({json: {
      schemaVersion: 4,
      code: recipeCode,
      sha256: 'a'.repeat(64),
      lifecycleStatus: 'published',
      status: 'published',
      configuration: {bundleVersion: '24.0.0'},
    }});
  });
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/events`, async (route) => {
    terminal = true;
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'id: 1\nevent: progress\ndata: {"status":"running"}\n\n',
    });
  });

  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await page.locator('button.list-item').filter({hasText: runCode}).click();
  const runDialog = page.getByRole('dialog', {name: `تفاصيل عملية البحث · ${runCode}`});
  const attemptedSeeds = runDialog.getByRole('button', {name: /^عدد البذور/});
  await expect(attemptedSeeds.locator('b')).toHaveText('4');
  await expect(runDialog.locator('section.research-run-events')).toContainText('tail_boundary_infeasible');
  expect(statsReads).toBeGreaterThan(1);
});

test('reconciles a completed run when the event stream stays open but silent', async ({page}) => {
  const runCode = 'RUN-SILENT-TERMINAL';
  const recipeCode = 'RCP-SILENT-TERMINAL';
  const baseStats = await (await page.request.get('/api/v2/research-stats')).json();
  let terminal = false;
  let releaseTerminalStats: (() => void) | undefined;
  const terminalStatsGate = new Promise<void>((resolve) => { releaseTerminalStats = resolve; });
  await page.addInitScript(() => {
    Object.defineProperty(window, 'EventSource', {
      configurable: true,
      value: class {
        addEventListener() {}
        close() {}
      },
    });
  });
  const run = () => ({
    runCode,
    recipeCode,
    status: terminal ? 'completed' : 'running',
    progress: terminal ? 1 : 0.5,
    resultCount: 0,
    qualifiedCount: 0,
    quarantineCount: 0,
  });
  await page.route((url) => url.pathname === '/api/v2/research-runs', async (route) => {
    await route.fulfill({json: {items: [run()]}});
  });
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}`, async (route) => {
    await route.fulfill({json: run()});
  });
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/stats`, async (route) => {
    if (terminal) await terminalStatsGate;
    await route.fulfill({json: baseStats});
  });
  await page.route((url) => url.pathname === `/api/v2/research-recipes/${recipeCode}`, async (route) => {
    await route.fulfill({json: {
      schemaVersion: 4,
      code: recipeCode,
      sha256: 'a'.repeat(64),
      lifecycleStatus: 'published',
      status: 'published',
      configuration: {bundleVersion: '24.0.0'},
    }});
  });

  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await page.locator('button.list-item').filter({hasText: runCode}).click();
  const runDialog = page.getByRole('dialog', {name: `تفاصيل عملية البحث · ${runCode}`});
  await expect(runDialog).toBeVisible();
  await expect(runDialog.locator('section.research-run-events')).toBeVisible();
  terminal = true;
  await expect(runDialog.locator('.drawer > header').getByText('مكتمل', {exact: true})).toBeVisible({timeout: 10_000});
  releaseTerminalStats?.();
  await expect(runDialog.getByRole('button', {name: 'أوقف البحث'})).toHaveCount(0);
});

const staleRunCode = 'RUN-STALE-DRAWER-A';
const currentRunCode = 'RUN-STALE-DRAWER-B';

// The first run's details stay unanswered until released, so each test decides what happens before they arrive.
async function holdFirstRunDetails(page: Page) {
  const recipeCode = 'RCP-STALE-DRAWER';
  const baseStats = await (await page.request.get('/api/v2/research-stats')).json();
  let currentRunTerminal = false;
  let releaseStaleDetails!: () => void;
  let releaseEvents!: () => void;
  const staleDetailsGate = new Promise<void>((resolve) => { releaseStaleDetails = resolve; });
  const eventsGate = new Promise<void>((resolve) => { releaseEvents = resolve; });
  const run = (runCode: string) => ({
    runCode,
    recipeCode,
    status: runCode === currentRunCode && currentRunTerminal ? 'completed' : 'running',
    progress: 0.5,
    resultCount: 0,
    qualifiedCount: 0,
    quarantineCount: 0,
  });
  const staleStatsPath = `/api/v2/research-runs/${staleRunCode}/stats`;

  await page.route((url) => url.pathname === '/api/v2/research-runs', async (route) => {
    await route.fulfill({json: {items: [run(staleRunCode), run(currentRunCode)]}});
  });
  for (const runCode of [staleRunCode, currentRunCode]) {
    await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}`, async (route) => {
      await route.fulfill({json: run(runCode)});
    });
    await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/stats`, async (route) => {
      if (runCode === staleRunCode) await staleDetailsGate;
      await route.fulfill({json: baseStats});
    });
    await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}/events`, async (route) => {
      await eventsGate;
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'id: 1\nevent: progress\ndata: {"status":"running"}\n\n',
      });
    });
  }
  await page.route((url) => url.pathname === `/api/v2/research-recipes/${recipeCode}`, async (route) => {
    await route.fulfill({json: {
      schemaVersion: 4,
      code: recipeCode,
      sha256: 'a'.repeat(64),
      lifecycleStatus: 'published',
      status: 'published',
      configuration: {bundleVersion: '24.0.0'},
    }});
  });

  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await expect(page.locator('#scenario-studio-title')).toBeVisible();

  const drawers = page.locator('dialog.drawer-layer');
  const currentDialog = page.getByRole('dialog', {name: `تفاصيل عملية البحث · ${currentRunCode}`});
  return {
    drawers,
    currentDialog,
    async openStaleRunAndDismiss() {
      const requested = page.waitForRequest((request) => new URL(request.url()).pathname === staleStatsPath);
      await page.locator('button.list-item').filter({hasText: staleRunCode}).click();
      await requested;
      const staleDialog = page.getByRole('dialog', {name: `تفاصيل عملية البحث · ${staleRunCode}`});
      await expect(staleDialog).toBeVisible();
      await staleDialog.locator('header button').click();
      await expect(drawers).toHaveCount(0);
    },
    async openCurrentRun() {
      await page.locator('button.list-item').filter({hasText: currentRunCode}).click();
      await expect(currentDialog.locator('section.research-run-events')).toBeVisible();
    },
    async deliverStaleDetails() {
      const focused = await page.evaluateHandle(() => document.activeElement);
      const answered = page.waitForResponse((response) => new URL(response.url()).pathname === staleStatsPath);
      releaseStaleDetails();
      await (await answered).finished();
      // A discarded late response must leave keyboard focus where the user put it.
      expect(await focused.evaluate((element) => element === document.activeElement)).toBe(true);
    },
    completeCurrentRun() {
      currentRunTerminal = true;
      releaseEvents();
    },
    release() {
      releaseStaleDetails();
      releaseEvents();
    },
  };
}

test('keeps the run drawer closed when its details arrive after dismissal', async ({page}) => {
  const held = await holdFirstRunDetails(page);
  try {
    await held.openStaleRunAndDismiss();
    await held.deliverStaleDetails();
    await expect(held.drawers).toHaveCount(0);
    await held.openCurrentRun();
    await expect(held.drawers).toHaveCount(1);
  } finally {
    held.release();
  }
});

test('keeps the selected run drawer when an earlier run answers last', async ({page}) => {
  const held = await holdFirstRunDetails(page);
  try {
    await held.openStaleRunAndDismiss();
    await held.openCurrentRun();
    await held.deliverStaleDetails();
    held.completeCurrentRun();
    await expect(held.currentDialog.locator('.drawer > header').getByText('مكتمل', {exact: true})).toBeVisible();
    await expect(held.currentDialog.getByRole('button', {name: 'أوقف البحث'})).toHaveCount(0);
    await expect(held.drawers).toHaveCount(1);
  } finally {
    held.release();
  }
});

// The other drawer loaders. Each case names the one request that stays unanswered until the test releases it.
type DrawerLoad = {
  name: string;
  stalePath: string;
  staleAnswer: unknown;
  showsLoadingDrawer: boolean;
  arrange(page: Page): Promise<unknown>;
  startStale(page: Page): Promise<unknown>;
  openCurrent(page: Page): Promise<unknown>;
  expectCurrent(drawers: Locator): Promise<unknown>;
};

const answer = (page: Page, path: string, json: unknown) =>
  page.route((url) => url.pathname === path, (route) => route.fulfill({json}));

const sakka = (catalogHash: string, finalScore: number[]) => ({
  catalogHash,
  kind: 'sakka',
  createdAt: '2026-01-01T00:00:00Z',
  metadata: {provenance: 'natural_sakka', roundCount: 0, winner: 'north_south', finalScore},
});
type Sakka = ReturnType<typeof sakka>;
const sakkaScore = (record: Sakka) => record.metadata.finalScore.join(' — ');
const sakkaDetail = (record: Sakka) => ({
  record,
  document: {sakka: {target: 152, finalScore: record.metadata.finalScore, rounds: []}},
});
const sakkaRow = (page: Page, record: Sakka) =>
  page.locator('table.sakka-table tbody tr').filter({hasText: sakkaScore(record)}).locator('td').first();
const staleSakka = sakka('a'.repeat(64), [152, 96]);
const currentSakka = sakka('b'.repeat(64), [160, 88]);

const sakkaDrawerLoad: DrawerLoad = {
  name: 'sakka',
  stalePath: `/api/v2/sakkas/${staleSakka.catalogHash}`,
  staleAnswer: sakkaDetail(staleSakka),
  showsLoadingDrawer: true,
  async arrange(page) {
    await answer(page, '/api/v2/sakkas', {items: [staleSakka, currentSakka]});
    await answer(page, `/api/v2/sakkas/${currentSakka.catalogHash}`, sakkaDetail(currentSakka));
  },
  async startStale(page) {
    await page.goto('/foundry?section=sakkas');
    await sakkaRow(page, staleSakka).click();
  },
  openCurrent: (page) => sakkaRow(page, currentSakka).click(),
  expectCurrent: (drawers) => expect(drawers.locator('.drawer-stats')).toContainText(sakkaScore(currentSakka)),
};

const laffa = (humanCode: string, gameHash: string) => {
  const source = {
    kind: 'archived_game',
    code: humanCode,
    identity: gameHash,
    uri: `/api/v1/history/games/${gameHash}`,
  };
  return {
    humanCode,
    sourceKind: 'archived_game',
    sourceKinds: ['archived_game'],
    sourceLocator: source,
    sourceReferences: [source],
    replayValid: false,
    auditOnly: true,
    publicationEligible: false,
    qualificationStatus: 'unqualified',
    qualificationFailures: [],
    truthLabel: null,
    confidence: null,
    challengeScore: null,
    tacticalTags: [],
    contract: 'sun',
    projects: [],
    buyer: 'north',
    dealer: 'east',
    interestingSeats: [],
    relatedSakkas: [],
    renderCount: 0,
    renderStatus: 'not_prepared',
    createdAt: '2026-01-01T00:00:00Z',
    traceHash: null,
    gameHash,
  };
};
const laffaRow = (page: Page, item: ReturnType<typeof laffa>) =>
  page.locator('table.unified-laffa-table tbody tr').filter({hasText: item.humanCode}).locator('td').first();
const staleLaffa = laffa('LAF-STALE-DRAWER-A', 'c'.repeat(64));
const currentLaffa = laffa('LAF-STALE-DRAWER-B', 'd'.repeat(64));

// The laffa list, with every answer except the stale laffa's exact source.
async function arrangeLaffat(page: Page) {
  const items = [staleLaffa, currentLaffa];
  await answer(page, '/api/v2/laffat', {
    schemaVersion: 1,
    resourceKind: 'unified_laffa_library_v1',
    items,
    total: items.length,
    offset: 0,
    limit: 50,
    nextOffset: null,
    sourceFacets: [{sourceKind: 'archived_game', count: items.length}],
  });
  for (const item of items) await answer(page, `/api/v2/laffat/${item.humanCode}/renders`, {items: []});
  await answer(page, currentLaffa.sourceLocator.uri, {gameHash: currentLaffa.gameHash});
}

const laffaDrawerLoad: DrawerLoad = {
  name: 'laffa',
  stalePath: staleLaffa.sourceLocator.uri,
  staleAnswer: {gameHash: staleLaffa.gameHash},
  showsLoadingDrawer: true,
  arrange: arrangeLaffat,
  async startStale(page) {
    await page.goto('/foundry?section=laffat');
    await laffaRow(page, staleLaffa).click();
  },
  openCurrent: (page) => laffaRow(page, currentLaffa).click(),
  async expectCurrent(drawers) {
    await expect(drawers.locator('.drawer > header')).toContainText(currentLaffa.humanCode);
    await expect(drawers.locator('.drawer-stats')).toBeVisible();
  },
};

const staleResultCode = 'LAF-STALE-RESULT';
const staleResultPath = `/api/v2/research-results/${staleResultCode}`;
const staleResultRecipeCode = 'RCP-STALE-RESULT';

// The result link starts its load with the page, and the laffa opened meanwhile is the drawer to keep.
const researchResultDrawerLoad: DrawerLoad = {
  name: 'research result',
  stalePath: `${staleResultPath}/decisions`,
  staleAnswer: {items: []},
  showsLoadingDrawer: false,
  async arrange(page) {
    await arrangeLaffat(page);
    await answer(page, staleResultPath, {
      resultCode: staleResultCode,
      recipeCode: staleResultRecipeCode,
      schemaVersion: 2,
      classification: 'scenario_v4',
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
    });
    await answer(page, `${staleResultPath}/renders`, {items: []});
    await answer(page, `/api/v2/research-recipes/${staleResultRecipeCode}`, {
      schemaVersion: 4,
      code: staleResultRecipeCode,
      sha256: 'a'.repeat(64),
      lifecycleStatus: 'published',
      status: 'published',
      configuration: {bundleVersion: '24.0.0'},
    });
  },
  startStale: (page) => page.goto(`/foundry?section=laffat&result=${staleResultCode}`),
  openCurrent: laffaDrawerLoad.openCurrent,
  expectCurrent: laffaDrawerLoad.expectCurrent,
};

const campaign = (campaignHash: string, createdAt: string, shardCount: number) => ({
  campaignHash,
  spec: {},
  status: 'completed',
  cancelRequested: false,
  shardCounts: {completed: shardCount},
  createdAt,
  updatedAt: createdAt,
});
type Campaign = ReturnType<typeof campaign>;
const campaignShards = ({campaignHash, shardCounts}: Campaign) => ({
  shards: Array.from({length: shardCounts.completed}, (_, shardIndex) => ({
    specHash: `${campaignHash}-${shardIndex}`,
    shardIndex,
    status: 'completed',
    attempts: 1,
  })),
});
const staleCampaign = campaign('e'.repeat(64), '2026-01-01T00:00:00Z', 2);
const currentCampaign = campaign('f'.repeat(64), '2026-01-02T00:00:00Z', 1);
// The list shows the newest campaign first.
const campaignButton = (page: Page, item: Campaign) =>
  page.locator('details.advanced-workflow .activity-panel button.list-item').nth(item === currentCampaign ? 0 : 1);

const campaignDrawerLoad: DrawerLoad = {
  name: 'campaign',
  stalePath: `/api/v2/campaigns/${staleCampaign.campaignHash}/shards`,
  staleAnswer: campaignShards(staleCampaign),
  showsLoadingDrawer: false,
  async arrange(page) {
    await answer(page, '/api/v2/campaigns', {items: [staleCampaign, currentCampaign]});
    await answer(page, `/api/v2/campaigns/${currentCampaign.campaignHash}/shards`, campaignShards(currentCampaign));
    for (const item of [staleCampaign, currentCampaign]) {
      await answer(page, `/api/v2/campaigns/${item.campaignHash}/attempts`, {attempts: []});
    }
  },
  async startStale(page) {
    await page.goto('/foundry?section=generation');
    await page.locator('details.advanced-workflow > summary').click();
    await campaignButton(page, staleCampaign).click();
  },
  openCurrent: (page) => campaignButton(page, currentCampaign).click(),
  // A late answer for the two-shard campaign must not reach the one-shard campaign's drawer either.
  expectCurrent: (drawers) =>
    expect(drawers.locator('.shard-track i')).toHaveCount(currentCampaign.shardCounts.completed),
};

async function holdStaleDrawerLoad(page: Page, load: DrawerLoad) {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const isStale = (request: Request) => new URL(request.url()).pathname === load.stalePath;
  const staleRequests: Request[] = [];
  page.on('request', (request) => { if (isStale(request)) staleRequests.push(request); });
  await load.arrange(page);
  await page.route((url) => url.pathname === load.stalePath, async (route) => {
    await gate;
    await route.fulfill({json: load.staleAnswer});
  });

  const drawers = page.locator('dialog.drawer-layer');
  const openCurrent = async () => {
    await load.openCurrent(page);
    await load.expectCurrent(drawers);
  };
  return {
    drawers,
    openCurrent,
    expectCurrent: () => load.expectCurrent(drawers),
    async startStaleAndDismiss() {
      const requested = page.waitForRequest(isStale);
      await load.startStale(page);
      await requested;
      // A loader that shows nothing while loading can only be dismissed through another drawer.
      if (!load.showsLoadingDrawer) await openCurrent();
      await drawers.locator('header button').click();
      await expect(drawers).toHaveCount(0);
    },
    async deliverStale() {
      const focused = await page.evaluateHandle(() => document.activeElement);
      release();
      await Promise.all(staleRequests.map(async (request) => (await request.response())?.finished()));
      // Give the page a rendered frame to act on the late answer before checking that it changed nothing.
      await page.evaluate(() => new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }));
      // A discarded late response must leave keyboard focus where the user put it.
      expect(await focused.evaluate((element) => element === document.activeElement)).toBe(true);
    },
    release,
  };
}

for (const load of [sakkaDrawerLoad, laffaDrawerLoad, researchResultDrawerLoad, campaignDrawerLoad]) {
  test(`keeps the drawer closed when the ${load.name} answers after dismissal`, async ({page}) => {
    const held = await holdStaleDrawerLoad(page, load);
    try {
      await held.startStaleAndDismiss();
      await held.deliverStale();
      await expect(held.drawers).toHaveCount(0);
      await held.openCurrent();
      await expect(held.drawers).toHaveCount(1);
    } finally {
      held.release();
    }
  });

  test(`keeps the drawer opened afterwards when the earlier ${load.name} answers last`, async ({page}) => {
    const held = await holdStaleDrawerLoad(page, load);
    try {
      await held.startStaleAndDismiss();
      await held.openCurrent();
      await held.deliverStale();
      await held.expectCurrent();
      await expect(held.drawers).toHaveCount(1);
    } finally {
      held.release();
    }
  });
}
