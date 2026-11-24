import {expect, test, type Locator, type Page} from '@playwright/test';

type PredicateCapability = {
  kind: string;
  safeComplement: boolean;
  stage: string;
  allowedHardness: string[];
  cpSatSupport?: {qualificationSource?: string};
};

type ExecutionCombination = {
  origin: 'search' | 'exact_trace_prefix';
  decisionKind: 'bid' | 'escalation' | 'project' | 'play';
  mode: 'exact' | 'bounded' | 'policy_conditioned';
  visibility: 'complete_state' | 'actor_information_set';
  claims: string[];
  lineUniverses: string[];
};

type ScenarioCapability = {
  backendIdentity: {backendId: string};
  hardPredicateCatalog: {definitions: PredicateCapability[]};
  executionCapabilities: {
    combinations: ExecutionCombination[];
    authoringCatalog: {
      expressionLimits: {maximumChildren: number};
      playingQuality?: {defaultBudgetProfileByModeAndVisibility?: Record<string, Record<string, string>>};
    };
  };
};

type JsonObject = {[key: string]: unknown};

async function capability(page: Page, bundleRevisionId: string): Promise<ScenarioCapability | null> {
  return page.evaluate(async (revision) => {
    const response = await fetch(
      `/api/v2/domain-capabilities?bundleRevisionId=${encodeURIComponent(revision)}`,
    );
    if (!response.ok) return null;
    const payload = await response.json() as {
      research?: {scenarioResearchV4?: ScenarioCapability};
    };
    return payload.research?.scenarioResearchV4 ?? null;
  }, bundleRevisionId);
}

async function openNativeStudio(page: Page): Promise<ScenarioCapability> {
  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  await expect(bundle).toBeVisible();
  await expect.poll(async () => (await optionValues(bundle)).filter(Boolean).length).toBeGreaterThan(0);
  const revisions = await optionValues(bundle);
  const observed: string[] = [];
  for (const revision of revisions.filter(Boolean)) {
    const selected = await capability(page, revision);
    observed.push(`${revision}:${selected?.backendIdentity.backendId ?? 'unavailable'}`);
    if (selected?.backendIdentity.backendId !== 'ortools_cp_sat_v3') continue;
    await bundle.selectOption(revision);
    await expect(page.locator('#scenario-studio-title')).toBeVisible();
    await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
    return selected;
  }
  throw new Error(`native_ortools_bundle_not_published:${observed.join(',')}`);
}

async function optionValues(select: Locator): Promise<string[]> {
  return select.locator('option').evaluateAll((options) => options.map(
    (option) => (option as HTMLOptionElement).value,
  ));
}

test('Research opens on a published native V3 bundle and keeps explicit bundle choices', async ({page}) => {
  const [catalogResponse] = await Promise.all([
    page.waitForResponse((response) => {
      const url = new URL(response.url());
      return response.request().method() === 'GET'
        && url.pathname === '/api/v2/domain-capabilities'
        && Boolean(url.searchParams.get('bundleRevisionId'));
    }),
    page.goto('/foundry?section=research'),
  ]);
  expect(catalogResponse.ok()).toBeTruthy();
  const bundle = page.locator('#research-bundle-select');
  await expect(bundle).toBeVisible();
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  const initial = await bundle.inputValue();
  expect(new URL(catalogResponse.url()).searchParams.get('bundleRevisionId')).toBe(initial);
  const labels = await bundle.locator('option').evaluateAll((options) => options.map((option) => ({
    value: (option as HTMLOptionElement).value, label: option.textContent ?? '',
  })));
  expect(new Set(labels.map((option) => option.label)).size).toBe(labels.length);
  for (const option of labels) expect(option.label).toContain(option.value);
  await expect(page.locator('.research-bundle-meta')).toContainText(initial);
  expect((await capability(page, initial))?.backendIdentity.backendId).toBe('ortools_cp_sat_v3');

  const revisions = (await optionValues(bundle)).filter((value) => value && value !== initial);
  const historical = await (async () => {
    for (const revision of revisions) {
      if ((await capability(page, revision))?.backendIdentity.backendId !== 'ortools_cp_sat_v3') return revision;
    }
    return null;
  })();
  expect(historical).not.toBeNull();
  await bundle.selectOption(historical!);
  await expect(bundle).toHaveValue(historical!);
  await expect(page.locator('#scenario-studio-title')).toBeHidden();
  await bundle.selectOption(initial);
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  await expect(page.locator('.research-bundle-meta')).toContainText(initial);
  await page.locator('.research-bundle-status button').click();
  await expect(page.locator('dialog .research-bundle-details > small')).toContainText(initial);
});

test('@mobile published bundle identity and expanded recipe stay within the viewport', async ({page}) => {
  await openNativeStudio(page);
  const size = await page.locator('.research-bundle-selector').evaluate((element) => ({
    width: element.clientWidth, scrollWidth: element.scrollWidth,
    left: element.getBoundingClientRect().left, right: element.getBoundingClientRect().right,
    viewport: document.documentElement.clientWidth,
  }));
  expect(size.scrollWidth).toBeLessThanOrEqual(size.width + 1);
  expect(size.left).toBeGreaterThanOrEqual(0);
  expect(size.right).toBeLessThanOrEqual(size.viewport);

  await addPredicate(page, 'auction.contract', 'EXPR-CONSTRAINTS', 1);
  await page.locator('#scenario-PRED-0001 select').last().selectOption('sun');
  await addPredicate(page, 'outcome.buyer_contract', 'EXPR-OUTCOME', 2);
  await page.locator('#scenario-PRED-0002 select').last().selectOption('won');
  const preview = page.locator('.scenario-actions .audit-details').first();
  await preview.locator('summary').click();
  await expect(preview.locator('pre')).toContainText('bundleRevisionId');
  const layout = await page.locator('.scenario-actions').evaluate((element) => ({
    width: element.clientWidth, scrollWidth: element.scrollWidth,
    pageWidth: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width + 1);
  expect(layout.pageWidth).toBeLessThanOrEqual(layout.viewport + 1);
});

test('optional search play actor does not silently restrict generation', async ({page}) => {
  await openNativeStudio(page);
  const seat = page.locator('#research-decision-seat');
  await expect(seat).toHaveValue('');
  await seat.selectOption('south');
  const decisionKind = page.locator('[data-field-path="scenario.decisionPoint.kind"] select');
  await decisionKind.selectOption('escalation');
  await decisionKind.selectOption('play');
  await expect(seat).toHaveValue('');
});

async function chooseOptionWithKeyboard(select: Locator, value: string) {
  await expect.poll(() => optionValues(select)).toContain(value);
  const values = await optionValues(select);
  const index = values.indexOf(value);
  if (index < 0) throw new Error(`bundle_option_missing:${value}`);
  await select.focus();
  await select.page().keyboard.press('Home');
  for (let offset = 0; offset < index; offset += 1) {
    await select.page().keyboard.press('ArrowDown');
  }
  await select.page().keyboard.press('Enter');
  await expect(select).toHaveValue(value);
}

async function addPredicate(
  page: Page,
  kind: string,
  target: string,
  ordinal: number,
) {
  const controls = page.locator('.scenario-catalog > .scenario-catalog-controls');
  await controls.locator('select').nth(0).selectOption(kind);
  await controls.locator('select').nth(1).selectOption(target);
  await controls.locator('button.primary').click();
  await expect(page.locator(`#scenario-PRED-${String(ordinal).padStart(4, '0')}`)).toBeVisible();
}

function expressionPredicates(value: unknown): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  const node = value as JsonObject;
  if (typeof node.kind === 'string') return [node.kind];
  return Array.isArray(node.children) ? node.children.flatMap(expressionPredicates) : [];
}

function expressionGroups(value: unknown): JsonObject[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  const node = value as JsonObject;
  if (typeof node.kind === 'string') return [];
  return [node, ...(Array.isArray(node.children) ? node.children.flatMap(expressionGroups) : [])];
}

function contrastRatio(foreground: string, background: string): number {
  const channels = (value: string) => {
    const hex = value.trim().match(/^#([0-9a-f]{6})$/i)?.[1];
    if (hex) return [0, 2, 4].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
    const rgb = value.match(/rgba?\((\d+)[, ]+(\d+)[, ]+(\d+)/i);
    if (!rgb) throw new Error(`unsupported_color:${value}`);
    return rgb.slice(1, 4).map(Number);
  };
  const luminance = (value: string) => {
    const [red, green, blue] = channels(value).map((channel) => {
      const normalized = channel / 255;
      return normalized <= 0.04045
        ? normalized / 12.92
        : ((normalized + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  };
  const values = [luminance(foreground), luminance(background)].sort((left, right) => right - left);
  return (values[0]! + 0.05) / (values[1]! + 0.05);
}

test('requires explicit project value correction after changing contract', async ({page}) => {
  await openNativeStudio(page);
  await addPredicate(page, 'auction.contract', 'EXPR-CONSTRAINTS', 1);
  const contract = page.locator('[data-field-path="PRED-0001.arguments.contract"] select');
  await contract.selectOption('sun');
  await addPredicate(page, 'project.exists', 'EXPR-CONSTRAINTS', 2);
  await page.locator('[data-field-path="PRED-0002.arguments.kind"] select').selectOption('hundred_sequence');
  const value = page.locator('[data-field-path="PRED-0002.arguments.value"] select');
  await value.selectOption('20');
  await contract.selectOption('hokum');
  await expect(value).toHaveValue('');
  await expect(value).toHaveAttribute('aria-invalid', 'true');
  await expect.poll(() => value.evaluate((element: HTMLSelectElement) => element.validity.valueMissing)).toBe(true);
  let validationRequests = 0;
  page.on('request', (request) => {
    if (request.url().endsWith('/api/v2/research-recipes:validate')) validationRequests += 1;
  });
  await page.locator('.scenario-actions button').nth(0).click();
  expect(validationRequests).toBe(0);
  await value.selectOption('10');
  await expect(value).toHaveValue('10');
  await expect(value).not.toHaveAttribute('aria-invalid', 'true');
  await expect.poll(() => value.evaluate((element: HTMLSelectElement) => element.checkValidity())).toBe(true);
});

for (const kind of ['large_card_sacrifice', 'small_card_big_gain']) {
  test(`requires explicit causal correction after changing tactic to ${kind}`, async ({page}) => {
    await openNativeStudio(page);
    await addPredicate(page, 'tactic.verified', 'EXPR-CONSTRAINTS', 1);
    const tactic = page.locator('[data-field-path="PRED-0001.arguments.kind"] select');
    const causal = page.locator('[data-field-path="PRED-0001.arguments.causalRequired"] select');
    await tactic.selectOption('entry_unblocking');
    await causal.selectOption('true');
    await tactic.selectOption(kind);
    await expect(causal).toHaveValue('');
    await expect(causal).toHaveAttribute('aria-invalid', 'true');
    await expect.poll(() => causal.evaluate((element: HTMLSelectElement) => element.validity.valueMissing)).toBe(true);
    const messageId = await causal.getAttribute('aria-describedby');
    expect(messageId).toBeTruthy();
    await expect(page.locator('[data-field-path="PRED-0001.arguments.causalRequired"] small')).toHaveAttribute('id', messageId!);
    await expect(page.locator('[data-field-path="PRED-0001.arguments.causalRequired"] small')).toContainText('غير مدعوم');
    expect(await optionValues(causal)).not.toContain('true');
    let validationRequests = 0;
    page.on('request', (request) => {
      if (request.url().endsWith('/api/v2/research-recipes:validate')) validationRequests += 1;
    });
    await page.locator('.scenario-actions button').nth(0).click();
    expect(validationRequests).toBe(0);
    await expect(causal).toBeFocused();
    await tactic.selectOption('entry_unblocking');
    await expect(causal).toHaveValue('true');
    await expect(causal).not.toHaveAttribute('aria-invalid', 'true');
    await expect(causal).not.toHaveAttribute('aria-describedby');
    await tactic.selectOption(kind);
    await causal.selectOption('false');
    await expect(causal).toHaveValue('false');
    await expect(causal).not.toHaveAttribute('aria-invalid', 'true');
    await expect(causal).not.toHaveAttribute('aria-describedby');
  });
}

test('whole-round information scope stays independent of the featured proof', async ({page}) => {
  const catalog = await openNativeStudio(page);
  const exactDefaults = catalog.executionCapabilities.authoringCatalog.playingQuality?.defaultBudgetProfileByModeAndVisibility?.exact;
  expect(exactDefaults?.complete_state).toBeTruthy();
  expect(exactDefaults?.actor_information_set).toBeTruthy();
  await page.locator('[data-field-path="playingQuality"] input[type="checkbox"]').check();
  const scope = page.locator('[data-field-path="playingQuality.visibility"] select');
  const featured = page.locator('[data-field-path="proof.visibility"] select');
  const originalFeatured = await featured.inputValue();
  await expect(scope).toHaveValue('actor_information_set');
  await scope.selectOption('complete_state');
  await expect(scope).toHaveValue('complete_state');
  await expect(featured).toHaveValue(originalFeatured);
  await page.locator('input[name="playing-quality-mode"][value="exact"]').check();
  await expect(scope).toHaveValue('complete_state');
  const budgets = page.locator('fieldset[data-field-path^="playingQuality.decisions."] > .scenario-grid > label:nth-child(2) select');
  await expect(budgets).toHaveCount(4);
  for (const budget of await budgets.all()) await expect(budget).toHaveValue(exactDefaults!.complete_state);
  await scope.selectOption('actor_information_set');
  for (const budget of await budgets.all()) await expect(budget).toHaveValue(exactDefaults!.actor_information_set);
  await expect(featured).toHaveValue(originalFeatured);
});

test('estimate explains the full-play fallback and clears it for an explicit position', async ({page}) => {
  await openNativeStudio(page);
  await addPredicate(page, 'state.legal_action_count', 'EXPR-CONSTRAINTS', 1);
  await addPredicate(page, 'outcome.team_round', 'EXPR-OUTCOME', 2);
  await page.locator('[data-field-path="playingQuality"] input[type="checkbox"]').check();
  await page.locator('#research-play-trick-index').fill('7');
  await page.locator('#research-play-remaining-cards').fill('2');
  await page.locator('#research-decision-seat').selectOption('south');
  await page.locator('#research-play-position').fill('');

  const actions = page.locator('.scenario-action-buttons button');
  await actions.nth(0).click();
  await expect(actions.nth(1)).toBeEnabled();
  await actions.nth(1).click();
  await expect(page.getByTestId('scenario-generic-play-search-warning')).toBeVisible();
  await expect(page.locator('.scenario-ortools-estimate')).toContainText('ما حددت ترتيب اللاعب');
  await expect(page.locator('.scenario-estimate > dl > div').filter({has: page.getByText('فئة التكلفة المتوقعة', {exact: true})}).locator('dd'))
    .toHaveText('غير قابل للتقدير بدقة قبل التشغيل');

  await page.locator('#research-play-position').fill('1');
  await actions.nth(0).click();
  await expect(actions.nth(1)).toBeEnabled();
  await actions.nth(1).click();
  await expect(page.getByTestId('scenario-generic-play-search-warning')).toHaveCount(0);
  await expect(page.locator('.scenario-ortools-estimate')).toContainText('generic');
  await expect(page.locator('.scenario-ortools-estimate')).not.toContainText('ما حددت ترتيب اللاعب');
});

test('new tactic predicates start with a context-supported argument', async ({page}) => {
  await openNativeStudio(page);
  let ordinal = 1;
  for (const origin of ['search', 'exact_trace_prefix']) {
    await page.locator('[data-field-path="scenario.origin.kind"] select').selectOption(origin);
    for (const visibility of ['complete_state', 'actor_information_set']) {
      await page.locator('[data-field-path="proof.visibility"] select').selectOption(visibility);
      for (const kind of ['tactic.applicable', 'tactic.verified']) {
        await addPredicate(page, kind, 'EXPR-CONSTRAINTS', ordinal);
        const row = page.locator(`#scenario-PRED-${String(ordinal++).padStart(4, '0')}`);
        const argument = row.locator('[data-field-path$=".arguments.kind"] select');
        const supported = (await optionValues(argument)).filter(Boolean);
        expect(supported.length).toBeGreaterThan(0);
        await expect(argument).toHaveValue(supported[0]!);
        await expect(row.locator('[aria-invalid="true"]')).toHaveCount(0);
        await row.locator('header button').click();
      }
    }
  }
});

test('authors every exposed predicate and governed expression operator from the live catalog', async ({page}) => {
  const selected = await openNativeStudio(page);
  const definitions = selected.hardPredicateCatalog.definitions;
  for (const kind of [
    'adjudication.out_of_turn_attempt',
    'adjudication.purchase_before_turn_attempt',
  ]) {
    const definition = definitions.find((item) => item.kind === kind);
    expect(definition, `missing merged V3.1 predicate ${kind}`).toBeDefined();
    expect(definition?.allowedHardness).toEqual(['hard']);
    expect(definition?.cpSatSupport?.qualificationSource).toBe('construction');
  }
  const catalog = page.locator('.scenario-catalog');
  expect((await optionValues(catalog.locator('select').nth(0))).sort()).toEqual(
    definitions.map(({kind}) => kind).sort(),
  );
  const countTokens = (await catalog.locator('header output').textContent() ?? '')
    .match(/\d+/g)?.map(Number) ?? [];
  expect(countTokens.length).toBeGreaterThan(0);
  expect(new Set(countTokens)).toEqual(new Set([definitions.length]));

  const constraints = page.locator('#scenario-EXPR-CONSTRAINTS');
  const addGroup = constraints.locator(
    ':scope > .scenario-expression-toolbar > .scenario-expression-buttons button',
  ).nth(1);
  for (const ordinal of [1, 2, 3]) {
    await addGroup.click();
    await expect(page.locator(`#scenario-EXPR-${String(ordinal).padStart(4, '0')}`)).toBeVisible();
  }

  const safe = definitions.find(({safeComplement}) => safeComplement);
  expect(safe, 'the catalog must expose at least one safe logical complement').toBeTruthy();
  const remaining = definitions.filter(({kind}) => kind !== safe!.kind);
  const anyDefinitions = remaining.splice(0, 2);
  const countDefinitions = remaining.splice(0, 2);
  let ordinal = 1;

  await addPredicate(page, safe!.kind, 'EXPR-0001', ordinal++);
  const notGroup = page.locator('#scenario-EXPR-0001');
  await notGroup.locator(':scope > .scenario-expression-toolbar select').selectOption('not');

  const anyGroup = page.locator('#scenario-EXPR-0002');
  await anyGroup.locator(':scope > .scenario-expression-toolbar select').selectOption('any');
  for (const definition of anyDefinitions) {
    await addPredicate(page, definition.kind, 'EXPR-0002', ordinal++);
  }

  const countGroup = page.locator('#scenario-EXPR-0003');
  await countGroup.locator(':scope > .scenario-expression-toolbar select').selectOption('count');
  for (const definition of countDefinitions) {
    await addPredicate(page, definition.kind, 'EXPR-0003', ordinal++);
  }
  const countInputs = countGroup.locator(
    ':scope > .scenario-expression-toolbar input[type="number"]',
  );
  await countInputs.nth(0).fill('1');
  await countInputs.nth(1).fill('2');

  const maximumChildren =
    selected.executionCapabilities.authoringCatalog.expressionLimits.maximumChildren;
  const directConstraintCapacity = Math.min(maximumChildren - 3, remaining.length - 1);
  expect(remaining.length).toBeLessThanOrEqual(directConstraintCapacity + maximumChildren);
  for (const [index, definition] of remaining.entries()) {
    await addPredicate(
      page,
      definition.kind,
      index < directConstraintCapacity ? 'EXPR-CONSTRAINTS' : 'EXPR-OUTCOME',
      ordinal++,
    );
  }

  await expect(page.locator('.scenario-predicate')).toHaveCount(definitions.length);
  const preview = page.locator('.scenario-actions .audit-details').first();
  await preview.locator('summary').click();
  const payload = JSON.parse(await preview.locator('pre').textContent() ?? '{}') as JsonObject;
  const scenario = payload.scenario as JsonObject;
  const authoredKinds = [
    ...expressionPredicates(scenario.constraints),
    ...expressionPredicates(scenario.outcome),
  ];
  expect(authoredKinds.sort()).toEqual(definitions.map(({kind}) => kind).sort());

  const constraintsPayload = scenario.constraints as JsonObject;
  expect(constraintsPayload.op).toBe('all');
  const groups = expressionGroups(constraintsPayload);
  const notGroupPayload = groups.find((group) => group.op === 'not');
  expect(notGroupPayload).toMatchObject({op: 'not'});
  expect(groups.find((group) => group.op === 'any')).toMatchObject({op: 'any'});
  expect(groups.find((group) => group.op === 'count')).toMatchObject({
    op: 'count', minimum: 1, maximum: 2,
  });
  expect(expressionPredicates(notGroupPayload)).toEqual([safe!.kind]);
});

test('applies catalog title constraints and checks core Foundry contrast', async ({page}) => {
  await openNativeStudio(page);
  const title = page.locator('[data-field-path="titleAr"] input');
  await expect(title).toHaveAttribute('required', '');
  await expect(title).toHaveAttribute('minlength', '3');
  await expect(title).toHaveAttribute('maxlength', '160');
  const helpId = await title.getAttribute('aria-describedby');
  expect(helpId).toBeTruthy();
  await expect(page.locator(`#${helpId}`)).toBeVisible();

  const samples = await page.evaluate(() => {
    const root = getComputedStyle(document.documentElement);
    const input = getComputedStyle(document.querySelector<HTMLElement>('[data-field-path="titleAr"] input')!);
    const muted = getComputedStyle(document.querySelector<HTMLElement>('.scenario-studio-heading p')!);
    const action = getComputedStyle(document.querySelector<HTMLElement>('.scenario-action-buttons button.secondary')!);
    return [
      {label:'body text', foreground:root.color, background:root.backgroundColor, minimum:4.5},
      {label:'muted panel text', foreground:muted.color, background:root.getPropertyValue('--panel2'), minimum:4.5},
      {label:'input text', foreground:input.color, background:input.backgroundColor, minimum:4.5},
      {label:'secondary action text', foreground:action.color, background:action.backgroundColor, minimum:4.5},
      {label:'focus indicator', foreground:root.getPropertyValue('--gold'), background:input.backgroundColor, minimum:3},
    ];
  });
  for (const sample of samples) {
    expect(
      contrastRatio(sample.foreground, sample.background),
      `${sample.label}: ${sample.foreground} on ${sample.background}`,
    ).toBeGreaterThanOrEqual(sample.minimum);
  }
});

test('@zoom shows exact official and optional rule authority with an accessible semantic delta', async ({page}) => {
  await page.goto('/foundry?section=research');
  await page.evaluate(() => { document.documentElement.style.zoom = '2'; });
  await expect(page.locator('html')).toHaveAttribute('dir','rtl');
  const bundle = page.locator('#research-bundle-select');
  const published = await page.evaluate(async () => {
    const response = await fetch('/api/v1/configuration-registry/bundles?status=published&limit=100');
    if (!response.ok) throw new Error(`bundle_list_failed:${response.status}`);
    return (await response.json() as {items:Array<{
      revisionId:string; bundleId:string; version:string;
    }>}).items;
  });
  const official = published
    .filter((item) => item.bundleId === 'baloot_ortools_research_native_v3')
    .sort((left, right) => right.version.localeCompare(left.version, undefined, {numeric: true}))[0];
  const optional = published.find((item)=>
    item.bundleId === 'baloot_ortools_research_native_v3_optional_project');
  expect(official).toBeTruthy();
  expect(optional).toBeTruthy();

  await chooseOptionWithKeyboard(bundle,official!.revisionId);
  await expect(page.getByTestId('rule-profile-id')).toHaveText('saudi_federation_2023_v2');
  await expect(page.getByTestId('rule-profile-version')).toHaveText('2.0.0');
  await expect(page.getByTestId('rule-profile-hash')).toHaveText(
    'bbc97b44e0705cad859bb512dde5f6bf08e9cb63d0e2cc42eaa167a09b1984e5',
  );
  await expect(page.getByTestId('rule-source-catalog')).toContainText(
    'saudi_federation_2023_v2_sources · 1.0.0',
  );
  await expect(page.getByTestId('rule-semantic-delta-empty')).toBeVisible();

  await chooseOptionWithKeyboard(bundle,optional!.revisionId);
  await expect(page.getByTestId('rule-profile-id')).toHaveText(
    'saudi_project_optional_unratified_v1',
  );
  await expect(page.getByTestId('rule-profile-version')).toHaveText('1.0.0');
  await expect(page.getByTestId('rule-profile-hash')).toHaveText(
    '80e0bc270074e9f8c1abd7b62f5e3f9826d171ef2a71f4fba3ab0b8669424ebd',
  );
  await expect(page.getByTestId('rule-source-catalog')).toContainText(
    'saudi_project_optional_unratified_v1_sources · 1.0.0',
  );
  const delta = page.locator('[data-delta-path="projects.declaration_mode"]');
  await expect(delta).toBeVisible();
  await expect(delta.locator('dd > span').nth(0)).toContainText('إلزامي');
  await expect(delta.locator('dd > span').nth(1)).toContainText('اختياري');
  expect(await page.getByTestId('rule-semantic-delta').ariaSnapshot()).toContain(
    'إلزام إعلان المشاريع',
  );

  const layout = await page.locator('.scenario-rule-authority').evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {left:rect.left,right:rect.right,viewportWidth:innerWidth};
  });
  expect(layout.left).toBeGreaterThanOrEqual(0);
  expect(layout.right).toBeLessThanOrEqual(layout.viewportWidth + 1);
  const samples = await page.evaluate(() => {
    const panel = document.querySelector<HTMLElement>('.scenario-rule-authority')!;
    const help = document.querySelector<HTMLElement>('.scenario-rule-authority > header p')!;
    const value = document.querySelector<HTMLElement>('.scenario-rule-authority-delta dd b')!;
    return [
      {label:'authority help',foreground:getComputedStyle(help).color,background:getComputedStyle(panel).backgroundColor},
      {label:'semantic value',foreground:getComputedStyle(value).color,background:'#06130f'},
    ];
  });
  for (const sample of samples) {
    expect(
      contrastRatio(sample.foreground,sample.background),
      `${sample.label}: ${sample.foreground} on ${sample.background}`,
    ).toBeGreaterThanOrEqual(4.5);
  }
});

test('authors every execution tuple advertised by the selected bundle', async ({page}) => {
  const selected = await openNativeStudio(page);
  const combinations = selected.executionCapabilities.combinations;
  expect(combinations.length).toBeGreaterThan(0);

  const origin = page.locator('[data-field-path="scenario.origin.kind"] select');
  const decision = page.locator('[data-field-path="scenario.decisionPoint.kind"] select');
  const mode = page.locator('[data-field-path="proof.mode"] select');
  const visibility = page.locator('[data-field-path="proof.visibility"] select');
  const claim = page.locator('[data-field-path="proof.claim"] select');

  for (const combination of combinations) {
    const name = [
      combination.origin,
      combination.decisionKind,
      combination.mode,
      combination.visibility,
    ].join('/');
    await test.step(name, async () => {
      await origin.selectOption(combination.origin);
      await expect(origin).toHaveValue(combination.origin);
      await expect(decision.locator(`option[value="${combination.decisionKind}"]`)).toHaveCount(1);
      await decision.selectOption(combination.decisionKind);
      await expect(decision).toHaveValue(combination.decisionKind);
      await expect(mode.locator(`option[value="${combination.mode}"]`)).toHaveCount(1);
      await mode.selectOption(combination.mode);
      await expect(mode).toHaveValue(combination.mode);
      await expect(visibility.locator(`option[value="${combination.visibility}"]`)).toHaveCount(1);
      await visibility.selectOption(combination.visibility);
      await expect(visibility).toHaveValue(combination.visibility);
      await expect(claim.locator('option')).toHaveCount(combination.claims.length);
      expect(await optionValues(claim)).toEqual(combination.claims);

      if (combination.lineUniverses.length) {
        expect(combination.claims).toContain('winning_continuation_line_count');
        await claim.selectOption('winning_continuation_line_count');
        const lineUniverse = page.locator('[data-field-path="proof.lineUniverse"] select');
        await expect(lineUniverse).toBeVisible();
        expect(await optionValues(lineUniverse)).toEqual(combination.lineUniverses);
        await expect(page.locator('[data-field-path="proof.selectedAction"] select')).toBeVisible();
      }
      if (combination.mode === 'bounded') {
        await expect(page.locator('[data-field-path="proof.horizon.value"] input')).toBeVisible();
      } else if (combination.mode === 'policy_conditioned') {
        await expect(page.locator('.scenario-policy-identities')).toBeVisible();
      }
    });
  }

  await origin.selectOption('exact_trace_prefix');
  await expect(origin).toHaveValue('exact_trace_prefix');
  const exposedPredicates = await optionValues(
    page.locator('.scenario-catalog > .scenario-catalog-controls select').nth(0),
  );
  const constraintDefinition = selected.hardPredicateCatalog.definitions.find(
    ({kind, stage}) => stage !== 'outcome' && !kind.startsWith('adjudication.'),
  );
  const outcomeDefinition = selected.hardPredicateCatalog.definitions.find(
    ({stage}) => stage === 'outcome',
  );
  expect(constraintDefinition).toBeTruthy();
  expect(outcomeDefinition).toBeTruthy();
  expect(exposedPredicates).toEqual(expect.arrayContaining([
    constraintDefinition!.kind,
    outcomeDefinition!.kind,
  ]));
  await addPredicate(page, constraintDefinition!.kind, 'EXPR-CONSTRAINTS', 1);
  await addPredicate(page, outcomeDefinition!.kind, 'EXPR-OUTCOME', 2);

  const adjudicationRoot = page.locator(
    '[data-field-path="scenario.origin.exactTracePrefix.adjudicationEvents"]',
  );
  await expect(adjudicationRoot).toBeVisible();
  await adjudicationRoot.locator(
    ':scope > .scenario-structured-list > button.secondary',
  ).click();

  const adjudicationKind = page.locator(
    '[data-field-path="scenario.origin.exactTracePrefix.adjudicationEvents.0.kind"] select',
  );
  const adjudicationRemedy = page.locator(
    '[data-field-path="scenario.origin.exactTracePrefix.adjudicationEvents.0.selectedRemedy"] select',
  );
  await expect(adjudicationKind).toBeVisible();
  await expect(adjudicationRemedy).toBeVisible();

  const eventKinds = (await optionValues(adjudicationKind)).filter(Boolean);
  expect(eventKinds.length).toBeGreaterThan(0);
  let selectedKind = '';
  let selectedRemedy = '';
  for (const eventKind of eventKinds) {
    await adjudicationKind.selectOption(eventKind);
    const remedies = (await optionValues(adjudicationRemedy)).filter(Boolean);
    if (!remedies.length) continue;
    selectedKind = eventKind;
    selectedRemedy = remedies[0];
    await adjudicationRemedy.selectOption(selectedRemedy);
    break;
  }
  expect(selectedKind, 'at least one governed adjudication event must offer a remedy').not.toBe('');
  await expect(adjudicationKind).toHaveValue(selectedKind);
  await expect(adjudicationRemedy).toHaveValue(selectedRemedy);

  const adjudicationReason = page.locator(
    '[data-field-path="scenario.origin.exactTracePrefix.adjudicationEvents.0.reasonCode"] input',
  );
  const adjudicationRuling = page.locator(
    '[data-field-path="scenario.origin.exactTracePrefix.adjudicationEvents.0.rulingEvidenceRef"] input',
  );
  await adjudicationReason.fill('catalog_browser_authoring');
  await expect(adjudicationRuling).toBeEnabled();
  await adjudicationRuling.fill('EVIDENCE-CATALOG-BROWSER');

  const preview = page.locator('.scenario-actions .audit-details').first();
  await preview.locator('summary').click();
  await expect.poll(async () => {
    const payload = JSON.parse(await preview.locator('pre').textContent() ?? '{}') as JsonObject;
    const scenario = payload.scenario as JsonObject;
    const authoredOrigin = scenario.origin as JsonObject;
    const exactTracePrefix = authoredOrigin.exactTracePrefix as JsonObject;
    const adjudicationEvents = exactTracePrefix.adjudicationEvents as JsonObject[];
    return adjudicationEvents[0] ?? {};
  }).toMatchObject({
    kind: selectedKind,
    selectedRemedy,
    reasonCode: 'catalog_browser_authoring',
    rulingEvidenceRef: 'EVIDENCE-CATALOG-BROWSER',
  });
});

test('hard continuation counts expose their required proof controls', async ({page}) => {
  await openNativeStudio(page);
  await addPredicate(page, 'auction.contract', 'EXPR-CONSTRAINTS', 1);
  await addPredicate(page, 'proof.winning_continuation_line_count', 'EXPR-OUTCOME', 2);
  await expect(page.locator('[data-field-path="proof.lineUniverse"] select')).toBeVisible();
  const selectedAction = page.locator('[data-field-path="proof.selectedAction"] select');
  await expect(selectedAction).toBeVisible();
  await expect(selectedAction).toHaveAttribute('aria-required', 'true');
  await expect(selectedAction).toHaveValue('');
  const options = (await optionValues(selectedAction)).filter(Boolean);
  expect(options.length).toBeGreaterThan(0);
  await selectedAction.selectOption(options[0]!);
  const preview = page.locator('.scenario-actions .audit-details').first();
  await preview.locator('summary').click();
  await expect.poll(async () => {
    const payload = JSON.parse(await preview.locator('pre').textContent() ?? '{}') as JsonObject;
    return (payload.proof as JsonObject)?.selectedAction;
  }).toBe(options[0]);
});
