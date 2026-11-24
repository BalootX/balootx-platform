import {expect, test, type Locator, type Page} from '@playwright/test';
import {nativeResearchBundleRevision} from './nativeResearchBundle';

test('library filters reflow without page overflow on narrow and desktop viewports', async ({page}, testInfo) => {
  await page.goto('/foundry?section=laffat');
  const toolbar = page.locator('.unified-laffa-toolbar');
  await expect(toolbar).toBeVisible();
  for (const width of [320, 390, 768, 901, 1024, 1087, 1100, 1101, 1180, 1440]) {
    await page.setViewportSize({width, height: 900});
    const layout = await toolbar.evaluate((element) => ({
      columns: getComputedStyle(element).gridTemplateColumns.split(' ').length,
      pageWidth: document.documentElement.scrollWidth,
      controls: Array.from(element.querySelectorAll('input, select, button'), (control) => {
        const {left, right} = control.getBoundingClientRect();
        return {left, right};
      }),
    }));
    expect.soft(layout.columns, `columns at ${width}px`).toBe(width <= 560 ? 1 : width <= 1100 ? 2 : 5);
    expect.soft(layout.pageWidth, `page width at ${width}px`).toBeLessThanOrEqual(width);
    for (const control of layout.controls) {
      expect.soft(control.left).toBeGreaterThanOrEqual(0);
      expect.soft(control.right).toBeLessThanOrEqual(width);
    }
    await page.screenshot({path: testInfo.outputPath(`library-${width}.png`)});
  }
});

test('optional proof group can be enabled, configured and removed visually without profile writes', async ({page}) => {
  const revision = 'CPR-VISUAL-OPTIONAL-PROOF';
  let fixture: Record<string, unknown>;
  await page.route('**/api/v1/configuration-registry/profiles?limit=1000', async (route) => {
    const response = await route.fetch();
    const payload = await response.json();
    fixture = {...payload.items.find((item: {family: string}) => item.family === 'challenge'),
      revisionId: revision, status: 'draft', payload: {scenarioResearchV4: {qualityExecution: {maximumWallSeconds: 600}}}};
    await route.fulfill({json: {...payload, items: [...payload.items.filter((item: {family: string}) => item.family !== 'challenge'), fixture]}});
  });
  await page.route(`**/api/v1/configuration-registry/profiles/${revision}`, (route) => route.fulfill({json: fixture}));
  const writes: string[] = [];
  page.on('request', (request) => {
    if (request.method() !== 'GET' && /\/configuration\/(profiles|bundles)/.test(request.url())) writes.push(request.url());
  });
  await page.goto('/foundry?section=settings');
  await page.getByRole('button', {name: /^اختيار التحديات/}).click();
  const research = page.locator('[data-config-path="scenarioResearchV4"]');
  if (await research.getAttribute('open') === null) await research.locator(':scope > summary').click();
  const quality = page.locator('[data-config-path="scenarioResearchV4.qualityExecution"]');
  await quality.locator(':scope > summary').click();
  const add = page.locator('#config-scenarioResearchV4-qualityExecution-bidAlternativeProof');
  await expect(add).toHaveAttribute('aria-checked', 'false');
  await add.focus();
  await page.keyboard.press('Space');
  const group = page.locator('[data-config-path="scenarioResearchV4.qualityExecution.bidAlternativeProof"]');
  await group.locator(':scope > summary').click();
  const version = group.getByRole('combobox').first();
  await expect(version).toHaveValue('');
  await version.selectOption('2');
  const field = (key: string) => group.locator(`#config-scenarioResearchV4-qualityExecution-bidAlternativeProof-${key}`);
  await expect(field('schemaVersion')).toBeDisabled();
  await expect(field('schemaVersion')).toHaveValue('2');
  await expect(field('method')).toHaveValue('information_set_refutation_v1');
  await expect(group.locator('input[type="number"]:enabled')).toHaveCount(3);
  for (const [name, value] of [['maximumQueries', '512'], ['maximumSeconds', '120'], ['upperBound', '46']]) {
    await expect(field(name)).toHaveValue('');
    await field(name).fill(value);
  }
  await group.screenshot({path:test.info().outputPath('configured-proof-group.png')});
  await group.locator(':scope > summary > button').click();
  await expect(group).toHaveCount(0);
  await expect(add).toHaveAttribute('aria-checked', 'false');
  expect(writes).toEqual([]);
});

test('intelligent quality controls preserve per-decision settings in the recipe', async ({page}) => {
  await openStudio(page);
  await page.locator('[data-field-path="proof.utility"] select').selectOption('round_win');
  const quality = page.locator('[data-field-path="playingQuality"]');
  await expect(quality).toBeVisible();
  await expect(quality.getByRole('checkbox')).toBeChecked();
  await quality.getByRole('checkbox').check();
  await expect(quality.locator('fieldset')).toHaveCount(5);
  const bid = quality.getByRole('group', {name: 'قرار مزايدة', exact: true});
  const boundary = bid.locator('select').filter({has: page.locator('option[value="auction_hand_margin_v1"]')});
  const horizon = bid.getByRole('spinbutton');
  await expect(boundary).toHaveValue('auction_hand_margin_v1');
  await expect(horizon).toHaveValue('1');
  await expect(horizon).toHaveAttribute('max', '1');
  await boundary.selectOption('current_raw_point_margin_v1');
  await bid.getByRole('combobox', {name: 'ميزانية الإثبات', exact: true})
    .selectOption('interactive_validation_v1');
  await expect(horizon).toHaveAttribute('min', '1');
  await expect(horizon).toHaveAttribute('max', '8');
  await horizon.fill('2');
  const catalog = page.locator('.scenario-catalog');
  await catalog.locator('select').nth(0).selectOption('auction.contract');
  await catalog.locator('button.primary').click();
  await catalog.locator('select').nth(0).selectOption('outcome.team_round');
  await catalog.locator('select').nth(1).selectOption('EXPR-OUTCOME');
  await catalog.locator('button.primary').click();
  const preview = page.locator('details').filter({has: page.getByText('معاينة الوصفة المرسلة', {exact: true})});
  await preview.locator('summary').click();
  const payload = JSON.parse(await preview.locator('pre').innerText());
  expect(payload.playingQuality.mode).toBe('budgeted');
  expect(payload.playingQuality.decisions.bid.utility).toBe('awarded_point_margin');
  expect(Object.keys(payload.playingQuality.decisions).sort()).toEqual(['bid', 'escalation', 'play', 'project']);
  expect(payload.playingQuality.decisions.bid.budgetProfile).toBe('interactive_validation_v1');
  expect(payload.playingQuality.decisions.bid.horizon).toEqual({unit: 'plies', value: 2});
  expect(payload.playingQuality.decisions.play.horizon.value).toBe(3);
  await page.getByRole('button', {name: 'افحص الوصفة', exact: true}).click();
  await expect(page.getByRole('button', {name: 'احسب التكلفة والوقت', exact: true})).toBeEnabled();
  await boundary.selectOption('auction_hand_margin_v1');
  await expect(horizon).toHaveValue('1');
  await expect(horizon).toHaveAttribute('max', '1');
  expect(await quality.locator('option[value="auction_hand_margin_v1"]').count()).toBe(1);
  await expect(page.locator('#scenario-proof option[value="auction_hand_margin_v1"]')).toHaveCount(0);
  const heuristicPayload = JSON.parse(await preview.locator('pre').innerText());
  expect(heuristicPayload.playingQuality.decisions.bid.boundaryEvaluator).toBe('auction_hand_margin_v1');
  expect(heuristicPayload.playingQuality.decisions.bid.horizon).toEqual({unit:'plies',value:1});
  expect(heuristicPayload.playingQuality.decisions.play.horizon.value).toBe(3);
  await page.getByRole('button', {name: 'افحص الوصفة', exact: true}).click();
  await expect(page.getByRole('button', {name: 'احسب التكلفة والوقت', exact: true})).toBeEnabled();
  await quality.locator('select[name="playing-quality-visibility"]').selectOption('actor_information_set');
  await quality.locator('input[name="playing-quality-mode"][value="exact"]').check();
  const exactBudget = bid.getByRole('combobox', {name: 'ميزانية الإثبات', exact: true});
  await expect(exactBudget).toHaveValue('offline_heavyweight_v1');
  await exactBudget.selectOption('actor_information_set_exact_v1');
  const exactPayload = JSON.parse(await preview.locator('pre').innerText());
  expect(exactPayload.playingQuality.mode).toBe('exact');
  expect(exactPayload.playingQuality.visibility).toBe('actor_information_set');
  expect(exactPayload.playingQuality.decisions.bid).toEqual({
    utility: 'awarded_point_margin',
    budgetProfile: 'actor_information_set_exact_v1',
  });
  await expect(quality.getByRole('spinbutton')).toHaveCount(0);
  await quality.getByRole('checkbox').uncheck();
  expect(JSON.parse(await preview.locator('pre').innerText())).not.toHaveProperty('playingQuality');
  await page.getByRole('button', {name: 'افحص الوصفة', exact: true}).click();
  await expect(page.locator('.scenario-diagnostic-list b')).toContainText('يلزم تدقيق كل القرارات');
});

test('exact quality history can undo a private decline and a public escalation', async ({page}) => {
  await openStudio(page);
  await page.locator('[data-field-path="playingQuality"]').getByRole('checkbox').check();
  await page.locator('[data-field-path="scenario.origin.kind"] select').selectOption('exact_trace_prefix');
  await page.locator('button[data-exact-stage="auction"]').filter({hasText: 'حكم'}).first().click();
  const decline = page.getByRole('button', {name: 'بدون تصعيد', exact: true});
  const pass = page.locator('button[data-exact-stage="auction"]').filter({hasText: /^بس$/});
  for (let response = 0; response < 3; response += 1) {
    await expect(decline.or(pass).first()).toBeVisible();
    if (await decline.isVisible()) break;
    await pass.click();
  }
  await expect(decline).toBeVisible();
  await decline.click();
  const history = page.locator('[data-field-path="scenario.origin.recordedEscalationOpportunities"]');
  await expect(history.locator('li')).toHaveCount(1);
  await history.getByRole('button', {name: 'احذف', exact: true}).click();
  await expect(history).toHaveCount(0);
  await expect(decline).toBeVisible();
  await page.getByRole('button', {name: 'دبل', exact: true}).first().click();
  await expect(history.locator('li')).toHaveCount(1);
  await history.getByRole('button', {name: 'احذف', exact: true}).click();
  await expect(history).toHaveCount(0);
  await expect(decline).toBeVisible();
});

async function tabTo(page: Page, target: Locator, key: 'Tab' | 'Shift+Tab' = 'Tab') {
  await expect(target).toBeVisible();
  await expect(target).toBeEnabled();
  if (await target.evaluate((element) => element === document.activeElement)) return;
  for (let step = 0; step < 400; step += 1) {
    await page.keyboard.press(key);
    if (await target.evaluate((element) => element === document.activeElement)) return;
  }
  throw new Error('keyboard_focus_target_unreachable');
}

async function choose(
  page: Page,
  select: Locator,
  value: string,
  key: 'Tab' | 'Shift+Tab' = 'Tab',
) {
  await tabTo(page, select, key);
  const options = await select.locator('option').evaluateAll((items) => items.map((item) => ({
    value: (item as HTMLOptionElement).value,
    label: item.textContent?.trim() ?? '',
  })));
  const index = options.findIndex((option) => option.value === value || option.label === value);
  expect(index, `${value} must be available`).toBeGreaterThanOrEqual(0);
  await page.keyboard.press('Home');
  for (let offset = 0; offset < index; offset += 1) await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(select).toHaveValue(options[index]!.value);
}

async function replace(page: Page, input: Locator, value: string) {
  await tabTo(page, input);
  await page.keyboard.press('Control+A');
  await page.keyboard.insertText(value);
  await expect(input).toHaveValue(value);
}

async function check(page: Page, input: Locator) {
  await tabTo(page, input);
  await page.keyboard.press('Space');
  await expect(input).toBeChecked();
}

async function openStudio(page: Page, keyboard = false) {
  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  const revision = await nativeResearchBundleRevision(page, bundle);
  if (keyboard) await choose(page, bundle, revision);
  else await bundle.selectOption(revision);
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
}

async function addPredicate(
  page: Page,
  kind: string,
  target: 'EXPR-CONSTRAINTS' | 'EXPR-OUTCOME',
  id: string,
) {
  const catalog = page.locator('.scenario-catalog');
  const direction = await page.locator('.scenario-predicate').count() ? 'Shift+Tab' : 'Tab';
  await choose(page, catalog.locator('select').nth(0), kind, direction);
  await choose(page, catalog.locator('select').nth(1), target);
  const add = catalog.locator('button.primary');
  await tabTo(page, add);
  await page.keyboard.press('Enter');
  await expect(page.locator(`#scenario-${id}`)).toBeVisible();
}

async function setPredicateSelect(page: Page, id: string, argument: string, value: string) {
  const selects = page.locator(`[data-field-path="${id}.arguments.${argument}"] select`);
  let matching: Locator | null = null;
  for (let index = 0; index < await selects.count(); index += 1) {
    const candidate = selects.nth(index);
    if (await candidate.locator(`option[value="${value}"]`).count()) {
      matching = candidate;
      break;
    }
  }
  expect(matching, `${id}.${argument} must expose ${value}`).not.toBeNull();
  await choose(page, matching!, value);
}

async function setPredicateNumber(page: Page, id: string, argument: string, value: number) {
  await replace(page, page.locator(`[data-field-path="${id}.arguments.${argument}"] input`), String(value));
}

async function setSelectorRanks(page: Page, id: string, ranks: string[]) {
  for (const rank of ranks) {
    await check(page, page.locator(
      `[data-field-path="${id}.arguments.selector.ranks"] input[value="${rank}"]`,
    ));
  }
}

const readinessLabel = 'إعدادات منشورة ومجمّدة';

function readinessTrigger(page: Page) {
  return page.locator('.readiness-grid').getByRole('button').filter({
    has: page.getByText(readinessLabel, {exact: true}),
  });
}

async function openReadinessDrawer(page: Page) {
  await page.goto('/foundry?section=overview');
  const trigger = readinessTrigger(page);
  await expect(trigger).toBeVisible();
  await expect(trigger).toBeEnabled();
  await trigger.focus();
  await page.keyboard.press('Enter');
  const dialog = page.locator('dialog.drawer-layer');
  await expect(dialog).toBeVisible();
  return {dialog, trigger};
}

test('native drawer confines focus, exposes its accessibility tree, and restores focus', async ({page}) => {
  const {dialog, trigger} = await openReadinessDrawer(page);
  const close = dialog.locator('header button');
  await expect(dialog).toHaveJSProperty('open', true);
  await expect(close).toBeFocused();
  expect(await page.locator('body').evaluate((element) => getComputedStyle(element).overflow)).toBe('hidden');

  const title = (await dialog.locator('#drawer-title').textContent())?.trim() ?? '';
  const accessibilityTree = await dialog.ariaSnapshot();
  expect(accessibilityTree).toContain('dialog');
  expect(accessibilityTree).toContain(title);

  await page.keyboard.press('Tab');
  await expect(page.locator('.guide-link')).not.toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(close).toBeFocused();
  await page.locator('.guide-link').focus();
  await expect(page.locator('.guide-link')).not.toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();
});

test('distinct performance metrics follow the selected governed copy capability', async ({page}) => {
  await page.route((url) => url.pathname === '/api/v2/research-runs', (route) => route.fulfill({json: {
    items: [{runCode: 'RUN-PERFORMANCE-COPY', recipeCode: 'RCP-PERFORMANCE-COPY', status: 'completed'}],
  }}));
  await page.goto('/foundry?section=overview');
  const distinctMetrics = page.locator('[data-performance-metric^="distinct-"]');
  await expect(page.locator('.research-performance-summary')).toBeVisible();
  await expect(distinctMetrics).toHaveCount(0);

  await page.locator('.workspace-nav button').nth(1).click();
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  await page.locator('.workspace-nav button').first().click();
  await expect(distinctMetrics).toHaveCount(3);
});

test('readiness details open after a failed startup load', async ({page}) => {
  await page.route('**/api/v2/foundry-profile', (route) => route.fulfill({status: 503, json: {detail: 'profile unavailable'}}));
  await page.goto('/foundry?section=overview');
  const trigger = readinessTrigger(page);
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const dialog = page.locator('dialog.drawer-layer');
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAccessibleName(readinessLabel);
});

test('@mobile touch opens and closes the drawer with a full-size control', async ({page}) => {
  expect(await page.evaluate(() => navigator.maxTouchPoints)).toBeGreaterThan(0);
  let releaseProfile!: () => void;
  const profileGate = new Promise<void>((resolve) => { releaseProfile = resolve; });
  await page.route('**/api/v2/foundry-profile', async (route) => {
    await profileGate;
    await route.continue();
  });
  const trigger = readinessTrigger(page);
  try {
    await page.goto('/foundry?section=overview');
    await expect(trigger).toBeVisible();
    await expect(trigger).toBeDisabled();
  } finally {
    releaseProfile();
  }
  await expect(trigger).toBeEnabled();
  await trigger.tap();
  const dialog = page.locator('dialog.drawer-layer');
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAccessibleName(readinessLabel);
  const close = dialog.locator('header button');
  const size = await close.evaluate((element) => {
    const style = getComputedStyle(element);
    return {width: parseFloat(style.width), height: parseFloat(style.height)};
  });
  expect(size.width).toBeGreaterThanOrEqual(44);
  expect(size.height).toBeGreaterThanOrEqual(44);
  await close.tap();
  await expect(dialog).toBeHidden();
});

test('@reduced-motion drawer keeps its state change without animation', async ({page}) => {
  await page.emulateMedia({reducedMotion: 'reduce'});
  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
  const {dialog} = await openReadinessDrawer(page);
  const motion = await dialog.locator('.drawer').evaluate((element) => {
    const style = getComputedStyle(element);
    return {animationName: style.animationName, transitionDuration: style.transitionDuration};
  });
  expect(motion).toEqual({animationName: 'none', transitionDuration: '0s'});
});

test('@zoom drawer remains operable and contained at 200 percent zoom', async ({page}) => {
  await page.goto('/foundry?section=overview');
  await page.evaluate(() => { document.documentElement.style.zoom = '2'; });
  const trigger = readinessTrigger(page);
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const dialog = page.locator('dialog.drawer-layer');
  await expect(dialog).toBeVisible();
  const close = dialog.locator('header button');
  await expect(close).toBeVisible();
  const bounds = await dialog.locator('.drawer').evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {left: rect.left, right: rect.right, viewportWidth: innerWidth};
  });
  expect(bounds.left).toBeGreaterThanOrEqual(0);
  expect(bounds.right).toBeLessThanOrEqual(bounds.viewportWidth + 1);
  await close.click();
  await expect(dialog).toBeHidden();
});

test('shows provisional solver variants while the selected research run is still running', async ({page}) => {
  const runCode = 'RUN-PROGRESSIVE-RESULT';
  const candidateHash = 'A'.repeat(64);
  const run = {
    runCode, recipeCode: 'RCP-PROGRESSIVE-RESULT', status: 'running',
    progress: 0.5, resultCount: 1, qualifiedCount: 1, quarantineCount: 0,
  };
  await page.route((url) => url.pathname === '/api/v2/research-runs', (route) =>
    route.fulfill({json: {items: [run]}}));
  let detailReads = 0;
  await page.route((url) => url.pathname === `/api/v2/research-runs/${runCode}`, (route) => {
    detailReads += 1;
    return route.fulfill({json: {...run, campaignProgress: {
      state: 'running',
      provisionalResults: [{
        schemaVersion: 1, candidateHash, exactStateId: 'STATE-00000000000000000000',
        decisionKind: 'play', contract: 'sun', actingSeat: 'west',
        proofCompletionStatus: 'exact_complete', winningRootActionCount: 1,
        selectedAction: 'play:spades:ace',
      }],
    }}});
  });
  await page.route((url) => url.pathname === '/api/v2/research-results', async (route) => {
    await route.fulfill({json: {
      items: [],
      total: 0,
      offset: 0,
      limit: 5,
      groups: [],
    }});
  });

  await openStudio(page);
  await page.locator('.scenario-result-facets select').first().selectOption(runCode);
  const provisional = page.getByRole('article').filter({hasText: candidateHash});
  await expect(provisional).toBeVisible();
  await expect(provisional.getByText('أولي للبحث فقط')).toBeVisible();
  await expect(provisional.getByText('play:spades:ace')).toBeVisible();
  expect(detailReads).toBeGreaterThan(0);
  await expect(page.getByText(`${runCode}`, {exact: true}).last()).toBeVisible();
});

test('shows every active render and keeps an open receipt current', async ({page}) => {
  let reads = 0;
  await page.route((url) => url.pathname === '/api/v2/render-batches', async (route) => {
    reads += 1;
    const status = reads < 5 ? 'running' : 'completed';
    await route.fulfill({json: {items: ['A1', 'B2'].map((suffix, index) => ({
      jobId: `RENDER-${suffix}`,
      status,
      progress: status === 'completed' ? 1 : 0.25 + index * 0.25,
      request: {items: [{basename: `variant-${suffix.toLowerCase()}`, source: {code: `LAF-${suffix}`}}]},
      result: status === 'completed' ? {status: 'ready'} : null,
      error: null,
    }))}});
  });

  await page.goto('/foundry?section=renders');
  const first = page.locator('#selected-render .job-progress').filter({hasText: 'LAF-A1'});
  const second = page.locator('#selected-render .job-progress').filter({hasText: 'LAF-B2'});
  await expect(first).toBeVisible();
  await expect(second).toBeVisible();
  await first.click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await dialog.locator('summary').click();
  await expect(dialog.locator('pre')).toContainText('"status": "completed"', {timeout: 8_000});
  expect(reads).toBeGreaterThanOrEqual(5);
});

test('non-play proof controls show their governed supported boundaries', async ({page}) => {
  await openStudio(page);
  await page.locator('[data-field-path="scenario.decisionPoint.kind"] select').selectOption('bid');
  const mode = page.locator('[data-field-path="proof.mode"] select');
  const help = page.locator('#scenario-proof .scenario-proof-mode-help');

  await expect(mode.locator('option[value="exact"]')).toHaveCount(0);

  await mode.selectOption('bounded');
  await expect(help).toContainText('الأفق المحدد فقط');

  await mode.selectOption('policy_conditioned');
  await expect(help).toContainText('السياسات المجمدة');
});

test('exact trace prefix is authored visually with a unique deck and legal timeline actions', async ({page}) => {
  let previewReads = 0;
  await page.route(
    (url) => url.pathname === '/api/v2/research-exact-traces:preview-legal-actions',
    async (route) => {
      previewReads += 1;
      const request = route.request().postDataJSON() as {
        configuration: {bundleRevisionId: string};
      };
      await route.fulfill({json: {
        schemaVersion: 1,
        configurationAuthority: {
          bundleRevisionId: request.configuration.bundleRevisionId,
          semanticBundleHash: 'A'.repeat(64),
          effectiveRuntimeConfigHash: 'B'.repeat(64),
          backendIdentityHash: 'C'.repeat(64),
        },
        valid: previewReads > 1,
        currentStage: previewReads > 1 ? 'auction' : null,
        nextActor: previewReads > 1 ? 'west' : null,
        terminal: false,
        legalActions: previewReads > 1 ? [{
          stage: 'auction',
          code: 'pass',
          actor: 'west',
          action: 'pass',
        }] : [],
        diagnostics: previewReads > 1 ? [] : [{code: 'exact_normalization_incomplete', fieldPath: 'exactTracePrefix'}],
        prefixHash: previewReads > 1 ? 'D'.repeat(64) : null,
        previewStateHash: previewReads > 1 ? 'E'.repeat(64) : null,
        previewReceiptHash: previewReads > 1 ? 'F'.repeat(64) : null,
        sha256: '0'.repeat(64),
      }});
    },
  );

  await openStudio(page);
  await page.locator('[data-field-path="scenario.origin.kind"] select')
    .selectOption('exact_trace_prefix');
  const builder = page.locator('.scenario-exact-builder');
  await expect(builder).toBeVisible();
  await expect(builder.getByRole('alert')).toContainText('غير مكتمل');
  await expect(builder.getByRole('alert')).not.toContainText('JSON');
  await builder.getByRole('button', {name: 'حاول مرة ثانية', exact: true}).click();
  await expect(builder.getByRole('button', {name: 'بس', exact: true})).toBeVisible();
  const deck = builder.locator('.scenario-deck-editor select');
  await expect(deck).toHaveCount(32);
  const initialValues = await deck.evaluateAll((items) => items.map(
    (item) => (item as HTMLSelectElement).value,
  ));
  expect(new Set(initialValues).size).toBe(32);

  const secondCard = await deck.nth(1).inputValue();
  await deck.nth(0).selectOption(secondCard);
  const swappedValues = await deck.evaluateAll((items) => items.map(
    (item) => (item as HTMLSelectElement).value,
  ));
  expect(new Set(swappedValues).size).toBe(32);
  expect(swappedValues[0]).toBe(initialValues[1]);
  expect(swappedValues[1]).toBe(initialValues[0]);

  const legalPass = builder.locator(
    '.scenario-exact-next-actions button[data-exact-stage="auction"]',
  );
  await expect(legalPass).toBeVisible();
  await legalPass.click();
  const advanced = builder.locator('details.audit-details');
  await advanced.locator('summary').click();
  await expect(advanced.locator('pre')).toContainText('"action": "pass"');
  await expect(advanced.locator('pre')).toContainText('"actor": "west"');
  expect(previewReads).toBeGreaterThan(1);
});

test('keyboard-only operator authors the complete five-variant flagship request', async ({page}) => {
  await openStudio(page, true);

  await replace(page, page.locator('[data-field-path="titleAr"] input'), 'بحث تضحية الآص بخمس لفات');
  await replace(page, page.locator('[data-field-path="scenario.decisionPoint.selector.trickIndex"] input'), '6');
  await replace(page, page.locator('[data-field-path="scenario.decisionPoint.selector.position"] input'), '4');
  await replace(page, page.locator('[data-field-path="scenario.decisionPoint.selector.remainingCards"] input'), '3');
  await choose(page, page.locator('[data-field-path="scenario.decisionPoint.selector.actingSeat"] select'), 'west');

  await addPredicate(page, 'auction.contract', 'EXPR-CONSTRAINTS', 'PRED-0001');
  await addPredicate(page, 'auction.buyer', 'EXPR-CONSTRAINTS', 'PRED-0002');
  await addPredicate(page, 'project.exists', 'EXPR-CONSTRAINTS', 'PRED-0003');
  await addPredicate(page, 'play.action_matches', 'EXPR-CONSTRAINTS', 'PRED-0004');
  await addPredicate(page, 'play.trick_winner_matches', 'EXPR-CONSTRAINTS', 'PRED-0005');
  await addPredicate(page, 'play.trick_winner_matches', 'EXPR-CONSTRAINTS', 'PRED-0006');
  await addPredicate(page, 'play.trick_card_count', 'EXPR-CONSTRAINTS', 'PRED-0007');
  await addPredicate(page, 'proof.winning_root_action_matches', 'EXPR-CONSTRAINTS', 'PRED-0008');
  await addPredicate(page, 'proof.winning_root_action_count', 'EXPR-CONSTRAINTS', 'PRED-0009');
  await addPredicate(page, 'outcome.team_round', 'EXPR-OUTCOME', 'PRED-0010');

  await setPredicateSelect(page, 'PRED-0001', 'contract', 'sun');
  await setPredicateSelect(page, 'PRED-0002', 'seat', 'north');
  await setPredicateSelect(page, 'PRED-0003', 'team', 'north_south');
  await setPredicateSelect(page, 'PRED-0003', 'owner', 'north');
  await setPredicateSelect(page, 'PRED-0003', 'kind', 'hundred_sequence');
  await setPredicateSelect(page, 'PRED-0003', 'value', '20');
  await setPredicateSelect(page, 'PRED-0003', 'state', 'awarded');
  await setPredicateNumber(page, 'PRED-0004', 'trickIndex', 6);
  await setPredicateNumber(page, 'PRED-0004', 'position', 4);
  await setSelectorRanks(page, 'PRED-0004', ['ace']);
  await setPredicateSelect(page, 'PRED-0004', 'selector.owner', 'actor');
  await setPredicateNumber(page, 'PRED-0005', 'trickIndex', 7);
  await setPredicateSelect(page, 'PRED-0005', 'team', 'west_east');
  await setSelectorRanks(page, 'PRED-0005', ['seven', 'eight', 'nine']);
  await setPredicateNumber(page, 'PRED-0006', 'trickIndex', 8);
  await setPredicateSelect(page, 'PRED-0006', 'team', 'west_east');
  await setSelectorRanks(page, 'PRED-0006', ['seven', 'eight', 'nine']);
  await setPredicateNumber(page, 'PRED-0007', 'trickIndex', 8);
  await setSelectorRanks(page, 'PRED-0007', ['ace']);
  await setPredicateSelect(page, 'PRED-0007', 'selector.owner', 'team');
  await setPredicateSelect(page, 'PRED-0007', 'selector.owner', 'north_south');
  await setPredicateNumber(page, 'PRED-0007', 'minimum', 2);
  await setPredicateNumber(page, 'PRED-0007', 'maximum', 2);
  await setSelectorRanks(page, 'PRED-0008', ['ace']);
  await setPredicateSelect(page, 'PRED-0008', 'selector.owner', 'actor');
  await setPredicateNumber(page, 'PRED-0009', 'minimum', 1);
  await setPredicateNumber(page, 'PRED-0009', 'maximum', 1);
  await setPredicateSelect(page, 'PRED-0010', 'team', 'west_east');
  await setPredicateSelect(page, 'PRED-0010', 'status', 'won');

  await choose(page, page.locator('[data-field-path="proof.claim"] select'), 'winning_root_action_count');
  await choose(page, page.locator('[data-field-path="proof.targetTeam"] select'), 'west_east');
  await choose(page, page.locator('[data-field-path="proof.utility"] select'), 'round_win');
  await replace(page, page.locator('[data-field-path="proof.successThreshold"] input'), '1');
  await choose(page, page.locator('[data-field-path="renderPolicy.mode"] select'), 'on_demand');
  await choose(page, page.locator('[data-field-path="output.kind"] select'), 'laffa_collection');
  await replace(page, page.locator('[data-field-path="output.resultCount"] input'), '5');
  await replace(page, page.locator('[data-field-path="renderPolicy.videoCount"] input'), '5');

  const preview = page.locator('.scenario-actions .audit-details');
  await tabTo(page, preview.locator('summary'));
  await page.keyboard.press('Enter');
  const payload = JSON.parse(await preview.locator('pre').textContent() ?? '{}');
  expect(payload.scenario.decisionPoint.selector).toMatchObject({
    trickIndex: 5,
    position: 3,
    remainingCards: 3,
    actingSeat: 'west',
  });
  expect(payload.output).toEqual({kind: 'laffa_collection', resultCount: 5});
  expect(payload.renderPolicy).toEqual({mode: 'on_demand', videoCount: 5});
  expect(payload.scenario.constraints.children.map((item: {kind: string}) => item.kind)).toEqual([
    'auction.contract',
    'auction.buyer',
    'project.exists',
    'play.action_matches',
    'play.trick_winner_matches',
    'play.trick_winner_matches',
    'play.trick_card_count',
    'proof.winning_root_action_matches',
    'proof.winning_root_action_count',
  ]);
  expect(payload.scenario.outcome.children[0]).toMatchObject({
    kind: 'outcome.team_round',
    hardness: 'hard',
    arguments: {team: 'west_east', status: 'won'},
  });
});

test('unknown loaded predicate remains visible, announced, removable, and unlaunchable', async ({page}) => {
  await openStudio(page);
  const bundleRevisionId = await page.locator('#research-bundle-select').inputValue();
  const code = 'RCP-UNSUPPORTED-E2E';
  const document = {
    schemaVersion: 4,
    code,
    sha256: 'a'.repeat(64),
    titleAr: 'وصفة قديمة فيها قيد غير مدعوم',
    output: {kind: 'laffa', resultCount: 1},
    configuration: {schemaVersion: 1, bundleRevisionId},
    scenario: {
      schemaVersion: 1,
      origin: {kind: 'search'},
      constraints: {nodeId: 'EXPR-CONSTRAINTS', op: 'all', children: [{
        predicateId: 'PRED-UNSUPPORTED',
        kind: 'missing.kind',
        version: 1,
        hardness: 'hard',
        arguments: {legacyValue: 7},
      }]},
      decisionPoint: {kind: 'play', selector: {trickIndex: 5, position: 3, remainingCards: 3, actingSeat: 'west'}},
      outcome: {nodeId: 'EXPR-OUTCOME', op: 'all', children: [{
        predicateId: 'PRED-OUTCOME',
        kind: 'outcome.team_round',
        version: 1,
        hardness: 'hard',
        arguments: {team: 'west_east', status: 'won'},
      }]},
    },
    proof: {
      schemaVersion: 1,
      mode: 'exact',
      visibility: 'complete_state',
      claim: 'winning_root_action_count',
      targetTeam: 'west_east',
      utility: 'round_win',
      successThreshold: 1,
      lineUniverse: 'all_legal',
      budgetProfile: 'late_game_exact_v1',
    },
    qualification: {},
    resources: {},
    diversity: {},
    renderPolicy: {mode: 'disabled', videoCount: 0},
    status: 'draft',
  };
  await page.route((url) => url.pathname === '/api/v2/research-recipes' && url.searchParams.has('status'), async (route) => {
    const url = new URL(route.request().url());
    await route.fulfill({json: {
      items: url.searchParams.get('status') === 'draft'
        ? [
            {code, schemaVersion: 4, titleAr: document.titleAr, status: 'draft', configuration: document.configuration},
            {code: 'RCP-LEGACY-V1', schemaVersion: 1, titleAr: 'قديم V1', status: 'draft'},
            {code: 'RCP-LEGACY-V2', schemaVersion: 2, titleAr: 'قديم V2', status: 'draft'},
            {
              code: 'RCP-WRONG-BUNDLE',
              schemaVersion: 4,
              titleAr: 'V4 من حزمة ثانية',
              status: 'draft',
              configuration: {schemaVersion: 1, bundleRevisionId: 'CBR-WRONG-BUNDLE'},
            },
          ]
        : [],
      nextCursor: null,
    }});
  });
  await page.route(`**/api/v2/research-recipes/${code}`, async (route) => route.fulfill({json: document}));
  await page.reload();
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  const library = page.locator('.scenario-recipe-library');
  const sourceRecipe = library.locator('select');
  await expect(sourceRecipe.locator('option')).toHaveCount(3);
  await expect(sourceRecipe).not.toContainText('قديم V1');
  await expect(sourceRecipe).not.toContainText('قديم V2');
  await sourceRecipe.selectOption('RCP-WRONG-BUNDLE');
  await expect(library.locator('#scenario-recipe-bundle-mismatch')).toContainText('CBR-WRONG-BUNDLE');
  await expect(library.getByRole('button', {name: 'حمّل'})).toBeDisabled();
  await sourceRecipe.selectOption(code);
  await library.locator('button').nth(0).click();

  const unsupported = page.locator('#scenario-PRED-UNSUPPORTED');
  await expect(unsupported).toBeVisible();
  await expect(unsupported).toHaveAttribute('aria-invalid', 'true');
  await expect(unsupported).toContainText('PRED-UNSUPPORTED · missing.kind');
  await expect(unsupported.locator('pre')).toContainText('legacyValue');
  await expect(unsupported.getByRole('status')).toContainText('predicate_cp_sat_unsupported');
  await expect(page.locator('.scenario-ortools-authority').getByRole('status')).toContainText('PRED-UNSUPPORTED');
  await expect(page.locator('.scenario-actions button.primary')).toBeDisabled();
  await expect(unsupported.locator('button')).toBeEnabled();
});

test('saved recipe explains a frozen-bundle mismatch before loading', async ({page}) => {
  await openStudio(page);
  const recipeBundleRevisionId = 'CBR-WRONG-BUNDLE';
  await page.route((url) => url.pathname === '/api/v2/research-recipes' && url.searchParams.has('status'), async (route) => {
    const url = new URL(route.request().url());
    await route.fulfill({json: {
      items: url.searchParams.get('status') === 'draft' ? [{
        code: 'RCP-WRONG-BUNDLE', schemaVersion: 4, titleAr: 'وصفة من نسخة ثانية',
        status: 'draft', configuration: {schemaVersion: 1, bundleRevisionId: recipeBundleRevisionId},
      }] : [],
      nextCursor: null,
    }});
  });
  await page.reload();
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  const library = page.locator('.scenario-recipe-library');
  await library.locator('select').selectOption('RCP-WRONG-BUNDLE');
  await expect(library.locator('#scenario-recipe-bundle-mismatch')).toContainText(recipeBundleRevisionId);
  await expect(library.locator('#scenario-recipe-bundle-mismatch')).toContainText('ما نبدأ بحثًا بإعدادات مخفية أو قديمة');
  await expect(library.getByRole('button', {name: 'حمّل'})).toBeDisabled();
  await expect(bundle.locator(`option[value="${recipeBundleRevisionId}"]`)).toHaveCount(0);
});

test('validation diagnostic moves focus to its failing control', async ({page}) => {
  await openStudio(page);
  const catalog = page.locator('.scenario-catalog');
  await catalog.locator('select').nth(0).selectOption('auction.contract');
  await catalog.locator('button.primary').click();
  await catalog.locator('select').nth(1).selectOption('EXPR-OUTCOME');
  await catalog.locator('select').nth(0).selectOption('outcome.team_round');
  await catalog.locator('button.primary').click();
  await page.route('**/api/v2/research-recipes:validate', async (route) => route.fulfill({json: {
    schemaVersion: 2,
    valid: false,
    issues: [{
      issueId: 'ISSUE-TITLE-E2E',
      severity: 'error',
      code: 'title_required',
      field: 'titleAr',
      fieldPaths: ['titleAr'],
      predicateIds: [],
      messageAr: 'عنوان السيناريو مطلوب',
      details: {},
    }],
  }}));

  await page.locator('.scenario-actions button').nth(0).click();
  const diagnostic = page.locator('.scenario-diagnostic-list button');
  await expect(diagnostic).toContainText('عنوان السيناريو مطلوب');
  await tabTo(page, diagnostic);
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-field-path="titleAr"] input')).toBeFocused();
});
