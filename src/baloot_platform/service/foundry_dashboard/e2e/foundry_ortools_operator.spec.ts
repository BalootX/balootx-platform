import {expect, test, type Locator, type Page} from '@playwright/test';
import {nativeResearchBundleRevision} from './nativeResearchBundle';

type JsonObject = Record<string, unknown>;

function object(value: unknown): JsonObject {
  expect(value).not.toBeNull();
  expect(Array.isArray(value)).toBe(false);
  expect(typeof value).toBe('object');
  return value as JsonObject;
}

async function fillPath(page: Page, path: string, value: string) {
  const input = page.locator(`[data-field-path="${path}"] input`).first();
  await expect(input).toBeVisible();
  await input.fill(value);
}

async function selectPath(page: Page, path: string, value: string) {
  const select = page.locator(`[data-field-path="${path}"] select`).first();
  await expect(select).toBeVisible();
  await select.selectOption(value);
}

async function addPredicate(
  page: Page,
  ordinal: number,
  kind: string,
  target = 'EXPR-CONSTRAINTS',
) {
  const controls = page.locator('.scenario-catalog > .scenario-catalog-controls');
  await controls.locator('select').nth(0).selectOption(kind);
  await controls.locator('select').nth(1).selectOption(target);
  await controls.getByRole('button').click();
  const id = `PRED-${String(ordinal).padStart(4, '0')}`;
  const predicate = page.locator(`#scenario-${id}`);
  await expect(predicate).toBeVisible();
  await predicate.locator(`[data-field-path="${id}.hardness"] select`).selectOption('hard');
  return {id, predicate};
}

async function checkSelectorRanks(
  predicate: Locator,
  id: string,
  ranks: readonly string[],
) {
  for (const rank of ranks) {
    await predicate
      .locator(`[data-field-path="${id}.arguments.selector.ranks"] input[value="${rank}"]`)
      .check();
  }
}

async function selectSelectorOwner(
  predicate: Locator,
  id: string,
  relation: string,
  team?: string,
) {
  const owner = predicate.locator(`[data-field-path="${id}.arguments.selector.owner"]`);
  await owner.locator('select').first().selectOption(relation);
  if (team) await owner.locator('select').nth(1).selectOption(team);
}

async function configureWinner(
  page: Page,
  ordinal: number,
  trick: string,
  winner: {seat?: string; team?: string},
) {
  const {id, predicate} = await addPredicate(page, ordinal, 'play.trick_winner_matches');
  await fillPath(page, `${id}.arguments.trickIndex`, trick);
  if (winner.seat) await selectPath(page, `${id}.arguments.seat`, winner.seat);
  if (winner.team) await selectPath(page, `${id}.arguments.team`, winner.team);
  await checkSelectorRanks(predicate, id, ['seven', 'eight', 'nine']);
}

async function openOperatorStudio(page: Page) {
  await page.goto('/foundry?section=research');
  const bundle = page.locator('#research-bundle-select');
  await bundle.selectOption(await nativeResearchBundleRevision(page, bundle));
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
}

async function authorFlagshipRequest(
  page: Page,
  title: string,
  trickSixWinner: 'north' | 'east',
  seedCount = '5',
  seedsPerShard = '1',
  resultCount = '5',
  videoCount = '5',
  budgetProfile = 'late_game_exact_v1',
  solverTimeoutSeconds = '60',
) {
  await fillPath(page, 'titleAr', title);
  await selectPath(page, 'scenario.origin.kind', 'search');
  await selectPath(page, 'scenario.decisionPoint.kind', 'play');
  await fillPath(page, 'scenario.decisionPoint.selector.trickIndex', '6');
  await fillPath(page, 'scenario.decisionPoint.selector.position', '4');
  await fillPath(page, 'scenario.decisionPoint.selector.remainingCards', '3');
  await selectPath(page, 'scenario.decisionPoint.selector.actingSeat', 'west');

  let added = await addPredicate(page, 1, 'auction.contract');
  await selectPath(page, `${added.id}.arguments.contract`, 'sun');

  added = await addPredicate(page, 2, 'auction.buyer');
  await selectPath(page, `${added.id}.arguments.seat`, 'north');

  added = await addPredicate(page, 3, 'project.exists');
  await selectPath(page, `${added.id}.arguments.team`, 'north_south');
  await selectPath(page, `${added.id}.arguments.owner`, 'north');
  await selectPath(page, `${added.id}.arguments.kind`, 'hundred_sequence');
  await selectPath(page, `${added.id}.arguments.value`, '20');
  await selectPath(page, `${added.id}.arguments.state`, 'awarded');

  added = await addPredicate(page, 4, 'play.action_matches');
  await fillPath(page, `${added.id}.arguments.trickIndex`, '6');
  await fillPath(page, `${added.id}.arguments.position`, '4');
  await selectPath(page, `${added.id}.arguments.actingSeat`, 'west');
  await checkSelectorRanks(added.predicate, added.id, ['ace']);
  await selectSelectorOwner(added.predicate, added.id, 'actor');

  await configureWinner(page, 5, '6', {seat: trickSixWinner});
  await configureWinner(page, 6, '7', {team: 'west_east'});
  await configureWinner(page, 7, '8', {team: 'west_east'});

  added = await addPredicate(page, 8, 'play.trick_card_count');
  await fillPath(page, `${added.id}.arguments.trickIndex`, '8');
  await checkSelectorRanks(added.predicate, added.id, ['ace']);
  await selectSelectorOwner(added.predicate, added.id, 'team', 'north_south');
  await fillPath(page, `${added.id}.arguments.minimum`, '2');
  await fillPath(page, `${added.id}.arguments.maximum`, '2');

  added = await addPredicate(page, 9, 'proof.winning_root_action_matches');
  await checkSelectorRanks(added.predicate, added.id, ['ace']);
  await selectSelectorOwner(added.predicate, added.id, 'actor');

  added = await addPredicate(page, 10, 'proof.winning_root_action_count');
  await fillPath(page, `${added.id}.arguments.minimum`, '1');
  await fillPath(page, `${added.id}.arguments.maximum`, '1');

  added = await addPredicate(page, 11, 'outcome.team_round', 'EXPR-OUTCOME');
  await selectPath(page, `${added.id}.arguments.team`, 'west_east');
  await selectPath(page, `${added.id}.arguments.status`, 'won');

  await selectPath(page, 'proof.mode', 'exact');
  await selectPath(page, 'proof.visibility', 'complete_state');
  await selectPath(page, 'proof.claim', 'winning_root_action_count');
  await selectPath(page, 'proof.targetTeam', 'west_east');
  await selectPath(page, 'proof.utility', 'round_win');
  await fillPath(page, 'proof.successThreshold', '1');
  await selectPath(page, 'proof.budgetProfile', budgetProfile);
  await page.locator('[data-field-path="playingQuality"] input[type="checkbox"]').check();

  await fillPath(page, 'resources.seedCount', seedCount);
  await fillPath(page, 'resources.seedsPerShard', seedsPerShard);
  await fillPath(page, 'resources.solverTimeoutSeconds', solverTimeoutSeconds);
  await fillPath(page, 'diversity.topKPerCell', '5');
  await selectPath(page, 'output.kind', 'laffa_collection');
  await fillPath(page, 'output.resultCount', resultCount);
  await selectPath(page, 'renderPolicy.mode', 'on_demand');
  await fillPath(page, 'renderPolicy.videoCount', videoCount);

  return page.locator('.scenario-actions button');
}

test.describe.serial('ACC-19 exact request and disclosed alternative', () => {
  test('historical optional bundles fail validation at the bundle selector before Save', async ({page}) => {
    await page.goto('/foundry?section=research');
    const bundle = page.locator('#research-bundle-select');
    const optional = bundle.locator('option').filter({hasText: 'optional_project'}).first();
    await expect(optional).toHaveCount(1);
    await bundle.selectOption(await optional.getAttribute('value') ?? '');
    await expect(page.locator('#scenario-studio-title')).toBeVisible();
    await addPredicate(page, 1, 'auction.contract');
    await addPredicate(page, 2, 'outcome.team_round', 'EXPR-OUTCOME');
    const response = page.waitForResponse(value =>
      new URL(value.url()).pathname === '/api/v2/research-recipes:validate',
    );
    const actions = page.locator('.scenario-actions button');
    await actions.nth(0).click();
    const validation = object(await (await response).json());
    expect(validation.valid).toBe(false);
    const diagnostic = page.locator('.scenario-actions .scenario-diagnostics');
    await expect(diagnostic).toBeVisible();
    await expect(actions.nth(1)).toBeDisabled();
    await expect(actions.nth(2)).toBeDisabled();
    await diagnostic.getByRole('button').click();
    await expect(bundle).toBeFocused();
  });

  test('exact estimate explains a budget refusal without exposing solver codes as guidance', async ({page}) => {
    await openOperatorStudio(page);
    await addPredicate(page, 1, 'auction.contract');
    await addPredicate(page, 2, 'outcome.team_round', 'EXPR-OUTCOME');
    await selectPath(page, 'proof.mode', 'exact');
    await selectPath(page, 'proof.visibility', 'complete_state');
    const actions = page.locator('.scenario-actions button');
    await actions.nth(0).click();
    await expect(actions.nth(1)).toBeEnabled();
    await actions.nth(1).click();
    const estimate = page.locator('.scenario-estimate');
    await expect(estimate).toContainText('العمق المطلوب');
    await expect(estimate).toContainText('أقصى عمق في الميزانية');
    await expect(estimate.locator('.scenario-launch-block').first()).toContainText('32');
    await expect(estimate.locator('.scenario-launch-block').first()).toContainText('16');
    await expect(estimate).toContainText('إثبات محدود');
    await expect(estimate.locator('.scenario-launch-block details code')).not.toBeVisible();
    await expect(estimate).toContainText('إثبات كامل لنهاية الجولة');
    expect(await estimate.innerText()).not.toMatch(/requested_depth_exceeds_proof_budget|not_compiled|exceeds_selected_tier|late_game_exact_v1/);
  });

  test('empty predicate groups explain recovery without sending a validation request', async ({page}) => {
    await openOperatorStudio(page);
    let requests = 0;
    page.on('request', request => {
      if (request.url().includes('/research-recipes:validate')) requests += 1;
    });
    const validate = page.locator('.scenario-actions button').first();
    const message = page.locator('.notice[role="status"]');
    await validate.click();
    await expect(message).toContainText('أضف شرطًا واحدًا على الأقل لوصف الحالة المطلوبة.');
    await addPredicate(page, 1, 'auction.contract');
    await validate.click();
    await expect(message).toContainText('أضف شرط نتيجة واحدًا على الأقل.');
    await expect(page.locator('#scenario-PRED-0001')).toBeVisible();
    expect(requests).toBe(0);
  });

  test('exact trace prefix advances through the real legal-action timeline', async ({page}) => {
    test.setTimeout(120_000);
    await openOperatorStudio(page);
    await selectPath(page, 'scenario.origin.kind', 'exact_trace_prefix');

    const builder = page.locator('.scenario-exact-builder');
    await expect(builder).toBeVisible();
    const legalPass = builder.locator(
      '.scenario-exact-next-actions button[data-exact-stage="auction"]',
    ).filter({hasText: 'بس'});
    await expect(legalPass).toBeVisible();
    await legalPass.click();

    const advanced = builder.locator('details.audit-details');
    await advanced.locator('summary').click();
    await expect(advanced.locator('pre')).toContainText('"action": "pass"');
    await expect(advanced.locator('pre')).toContainText('"actor": "east"');
    await legalPass.click();
    await expect(advanced.locator('pre')).toContainText('"actor": "north"');
  });

  test('changing to actor-visible proof clears unsupported continuation counting', async ({page}) => {
    await openOperatorStudio(page);
    await selectPath(page, 'proof.visibility', 'complete_state');
    await selectPath(page, 'proof.claim', 'winning_continuation_line_count');
    const claim = page.locator('[data-field-path="proof.claim"] select');
    const lineUniverse = page.locator('[data-field-path="proof.lineUniverse"] select');
    await expect(lineUniverse).toBeVisible();

    await selectPath(page, 'proof.visibility', 'actor_information_set');
    await expect(claim).not.toHaveValue('winning_continuation_line_count');
    await expect(claim.locator('option[value="winning_continuation_line_count"]')).toHaveCount(0);
    await expect(lineUniverse).not.toBeVisible();
    await expect(page.locator('.scenario-catalog > .scenario-catalog-controls option[value="proof.winning_continuation_line_count"]'))
      .toHaveCount(0);
  });

  test('the opponent-wins-trick-6 request is not falsely refused by the V3 preflight', async ({page}) => {
    test.setTimeout(180_000);
    const pageErrors: string[] = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    await openOperatorStudio(page);
    const actions = await authorFlagshipRequest(
      page,
      'الطلب الأصلي غير المعدل: الخصم يفوز بالسادسة ويحتفظ بإكتين للثامنة',
      'north',
    );

    await actions.nth(0).click();
    const validationDiagnostics = page.locator('.scenario-actions .scenario-diagnostics');
    await expect(page.locator('.scenario-actions .context-proof')).toBeVisible();
    await expect(validationDiagnostics).toHaveCount(0);
    await expect(actions.nth(1)).toBeEnabled();
    const estimateResponse = page.waitForResponse(response =>
      new URL(response.url()).pathname === '/api/v2/research-recipes:estimate' &&
      response.request().method() === 'POST',
    );
    await actions.nth(1).click();
    const estimated = object(await (await estimateResponse).json());
    const estimate = page.locator('.scenario-estimate');
    await expect(estimate).toBeVisible();
    await expect(estimate).not.toContainText('exact_structural_infeasibility');
    await expect(estimate.locator('dl > div').filter({hasText: 'قرارات التحليل، حد أعلى'}).first().locator('dd'))
      .toHaveText(String(estimated.analysisDecisionUpperBound));
    await expect(estimate.locator('dl > div').filter({hasText: 'قرارات التحقق التكتيكي، حد أعلى'}).first().locator('dd'))
      .toHaveText(String(estimated.tacticalVerificationDecisionUpperBound));
    await expect(estimate).not.toContainText('قرارات الإثبات العميق، حد أعلى');
    expect(typeof estimated.nativeCapabilityPlanHash).toBe('string');
    await expect(estimate.locator('.scenario-estimate-identities').last().locator('code').nth(1))
      .toHaveText(String(estimated.nativeCapabilityPlanHash));
    await expect(actions.nth(2)).toBeEnabled();
    expect(pageErrors).toEqual([]);
    await estimate.screenshot({path: test.info().outputPath('original-request-estimate.png')});
    await page.screenshot({
      path: test.info().outputPath('original-request-admissible.png'),
      fullPage: true,
    });
  });

  test('operator estimate discloses the shared campaign deadline', async ({page}) => {
    await openOperatorStudio(page);
    const actions = await authorFlagshipRequest(
      page,
      'فحص مدة الحزمة قبل إطلاق البحث',
      'east',
      '50',
      '1',
      '1',
      '0',
      'offline_heavyweight_v1',
      '100',
    );
    await actions.nth(0).click();
    await actions.nth(1).click();
    const estimate = page.locator('.scenario-estimate');
    await expect(estimate).toBeVisible();
    const sharding = estimate.locator('.scenario-sharding-estimate');
    await expect(sharding.locator('dl > div').filter({hasText: 'الوقت الأقصى لكل حزمة (ثوانٍ)'}).locator('dd'))
      .toHaveText('600');
    await expect(sharding).not.toContainText('قد تتوقف الحزمة');
  });

  test('the disclosed target-wins-trick-6 alternative diagnoses its first result', async ({page}) => {
    test.setTimeout(750_000);
    const pageErrors: string[] = [];
    const statisticsStatuses: number[] = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('response', (response) => {
      if (new URL(response.url()).pathname === '/api/v2/research-stats') {
        statisticsStatuses.push(response.status());
      }
    });

    await openOperatorStudio(page);
    const actions = await authorFlagshipRequest(
      page,
      'بديل تشخيصي معلن: فريق غرب وشرق يفوز بالسادسة ونتيجة واحدة بلا فيديو',
      'east',
      '50',
      '1',
      '1',
      '0',
      'offline_heavyweight_v1',
      '100',
    );

    const normalizedPreview = page.locator('.scenario-actions > details.audit-details').last();
    await normalizedPreview.locator('summary').click();
    const normalizedRecipeText = await normalizedPreview.locator('pre').innerText();
    const normalizedRecipe = object(JSON.parse(normalizedRecipeText));
    expect(object(normalizedRecipe.output).resultCount).toBe(1);
    expect(object(normalizedRecipe.renderPolicy).videoCount).toBe(0);
    expect(object(normalizedRecipe.resources).seedCount).toBe(50);
    expect(object(normalizedRecipe.resources).seedsPerShard).toBe(1);
    await test.info().attach('alternate-first-result-normalized-recipe.json', {
      body: normalizedRecipeText,
      contentType: 'application/json',
    });
    await normalizedPreview.locator('summary').click();

    await actions.nth(0).click();
    await expect(page.locator('.scenario-actions .context-proof')).toBeVisible();
    await actions.nth(1).click();
    const estimate = page.locator('.scenario-estimate');
    await expect(estimate).toBeVisible();
    await expect(actions.nth(2)).toBeEnabled();
    await test.info().attach('alternate-first-result-estimate.txt', {
      body: await estimate.innerText(),
      contentType: 'text/plain',
    });
    await page.screenshot({
      path: test.info().outputPath('alternate-first-result-estimate.png'),
      fullPage: true,
    });

    await actions.nth(2).click();
    const runDialog = page.getByRole('dialog');
    await expect(runDialog).toBeVisible();
    await expect(runDialog.locator('.research-run-actions button.primary')).toBeVisible({
      timeout: 650_000,
    });
    const liveRunEvidence = runDialog.locator('.drawer-body > details.audit-details').last();
    await liveRunEvidence.locator('summary').click();
    const liveRunDocument = object(JSON.parse(await liveRunEvidence.locator('pre').innerText()));
    expect(object(liveRunDocument.run).status).toBe('completed');
    expect(object(liveRunDocument.run).progress).toBe(1);
    await runDialog.locator('aside > header > button').click();
    const runListItem = page.locator('button.list-item').filter({hasText: /^RUN-/}).first();
    await expect(runListItem).toBeVisible({timeout: 30_000});
    await runListItem.click();
    const terminalRunDialog = page.getByRole('dialog');
    await expect(terminalRunDialog).toBeVisible();
    const runEvidence = terminalRunDialog.locator('.drawer-body > details.audit-details').last();
    await runEvidence.locator('summary').click();
    const runEvidenceText = await runEvidence.locator('pre').innerText();
    const runEvidenceDocument = object(JSON.parse(runEvidenceText));
    const run = object(runEvidenceDocument.run ?? runEvidenceDocument);
    await test.info().attach('alternate-first-result-run-receipt.json', {
      body: runEvidenceText,
      contentType: 'application/json',
    });
    await page.screenshot({
      path: test.info().outputPath('alternate-first-result-terminal-run.png'),
      fullPage: true,
    });
    await terminalRunDialog.locator('aside > header > button').click();

    const results = page.locator('button.list-item').filter({hasText: /^LAF-/});
    await expect(results).toHaveCount(1, {timeout: 30_000});
    const operatorAuditResponse = page.waitForResponse((response) =>
      response.url().endsWith('/operator-audit') && response.status() === 200,
    );
    await results.first().click();
    const operatorAudit = object(await (await operatorAuditResponse).json());
    const resultDialog = page.getByRole('dialog');
    await expect(resultDialog.locator('.research-scenario-authority')).toBeVisible();
    const resultEvidence = resultDialog.locator('.drawer-body > details.audit-details').last();
    await resultEvidence.locator('summary').click();
    const resultEvidenceText = await resultEvidence.locator('pre').innerText();
    const detail = object(JSON.parse(resultEvidenceText));
    await test.info().attach('alternate-first-result-evidence.json', {
      body: resultEvidenceText,
      contentType: 'application/json',
    });
    await page.screenshot({
      path: test.info().outputPath('alternate-first-result-terminal-result.png'),
      fullPage: true,
    });

    const scenario = object(detail.scenarioProjection);
    const proof = object(detail.proofOutcome);
    const authority = object(operatorAudit.researchAuthority);
    const compatibility = object(operatorAudit.runtimeCompatibility);
    const qualification = object(detail.qualification);
    expect(run.status).toBe('completed');
    expect(detail.researchBackend).toBe('ortools_cp_sat_v3');
    expect(detail.qualificationAuthority).toBe('cp_sat');
    expect(detail.disposition).toBe('qualified');
    expect(qualification.failures).toEqual([]);
    expect(scenario.status).toBe('satisfied');
    expect(scenario.predicateEvidenceCount).toBe(11);
    expect(proof.achievedMode).toBe('exact');
    expect(proof.completionStatus).toBe('exact_complete');
    expect(proof.winningRootActionCount).toBe(1);
    expect(proof.claimSatisfied).toBe(true);
    expect(compatibility.status).toBe('passed');
    expect(String(run.recipeHash ?? '')).toMatch(/^[a-f0-9]{64}$/);
    expect(String(object(operatorAudit.executionReceipt).semanticResultHash ?? '')).toMatch(/^[a-f0-9]{64}$/);
    expect(scenario.exactStateAvailable).toBe(true);
    expect(String(authority.traceHash ?? '')).toMatch(/^[a-f0-9]{64}$/);
    const stageTimings = object(object(operatorAudit.executionReceipt).stageTimings);
    expect(Number(stageTimings.finalNormalization)).toBeGreaterThanOrEqual(0);
    await expect(
      resultDialog.locator('.research-scenario-stage-timings code', {
        hasText: 'finalNormalization',
      }),
    ).toBeVisible();
    expect(statisticsStatuses.length).toBeGreaterThan(0);
    expect(new Set(statisticsStatuses)).toEqual(new Set([200]));
    expect(pageErrors).toEqual([]);
  });

  test('cancelling a live campaign is visible and immediately terminal', async ({page}) => {
    test.setTimeout(180_000);
    await openOperatorStudio(page);
    const actions = await authorFlagshipRequest(
      page,
      'إثبات إلغاء حملة البحث من واجهة المشغّل',
      'east',
      '50',
      '1',
      '1',
      '0',
      'late_game_exact_v1',
      '15',
    );

    await actions.nth(0).click();
    await expect(page.locator('.scenario-actions .context-proof')).toBeVisible();
    await actions.nth(1).click();
    await expect(actions.nth(2)).toBeEnabled();
    await actions.nth(2).click();

    const liveDialog = page.getByRole('dialog');
    const cancel = liveDialog.locator('.research-run-actions button.danger');
    await expect(cancel).toBeVisible();
    await cancel.click();

    const runListItem = page.locator('button.list-item').filter({hasText: /^RUN-/}).first();
    await expect(runListItem).toBeVisible({timeout: 5_000});
    await runListItem.click();
    const terminalDialog = page.getByRole('dialog');
    const audit = terminalDialog.locator('.drawer-body > details.audit-details').last();
    await audit.locator('summary').click();
    const document = object(JSON.parse(await audit.locator('pre').innerText()));
    const run = object(document.run ?? document);
    const progress = object(run.campaignProgress);
    expect(run.status).toBe('cancelled');
    expect(progress.state).toBe('cancelled');
    expect(progress.stopReason).toBe('cancelled');
  });

  test('the disclosed target-wins-trick-6 alternative finds five exact variants', async ({page}) => {
    test.setTimeout(1_800_000);
    const pageErrors: string[] = [];
    const statisticsStatuses: number[] = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('response', (response) => {
      if (new URL(response.url()).pathname === '/api/v2/research-stats') {
        statisticsStatuses.push(response.status());
      }
    });
    await openOperatorStudio(page);
    const actions = await authorFlagshipRequest(
      page,
      'بديل معلن: فريق غرب وشرق يفوز بالسادسة قبل طلب خمس تنويعات',
      'east',
      '50',
      '1',
      '5',
      '5',
      'offline_heavyweight_v1',
      '60',
    );

    const normalizedPreview = page.locator('.scenario-actions > details.audit-details').last();
    await normalizedPreview.locator('summary').click();
    const normalizedRecipe = object(JSON.parse(await normalizedPreview.locator('pre').innerText()));
    expect(object(normalizedRecipe.resources).seedCount).toBe(50);
    expect(object(normalizedRecipe.resources).seedsPerShard).toBe(1);
    expect(object(normalizedRecipe.output).resultCount).toBe(5);
    expect(object(normalizedRecipe.renderPolicy).videoCount).toBe(5);
    await normalizedPreview.locator('summary').click();

    await actions.nth(0).click();
    await expect(page.locator('.scenario-actions .context-proof')).toBeVisible();
    await actions.nth(1).click();
    await expect(page.locator('.scenario-estimate')).toBeVisible();
    await expect(actions.nth(2)).toBeEnabled();
    await actions.nth(2).click();

  const runDialog = page.getByRole('dialog');
  await expect(runDialog).toBeVisible();
  await expect(runDialog.locator('.research-run-actions button.primary')).toBeVisible({timeout: 660_000});
  await runDialog.locator('aside > header > button').click();

  const results = page.locator('button.list-item').filter({hasText: /^LAF-/});
  await expect(results).toHaveCount(5);
  await expect(results).toContainText([
    'exact_complete',
    'exact_complete',
    'exact_complete',
    'exact_complete',
    'exact_complete',
  ]);

  const resultHashes = new Set<string>();
  const traceHashes = new Set<string>();
  for (let index = 0; index < 5; index += 1) {
    const operatorAuditResponse = page.waitForResponse(
      (response) => response.url().endsWith('/operator-audit') && response.status() === 200,
    );
    await results.nth(index).click();
    const operatorAudit = object(await (await operatorAuditResponse).json());
    const resultDialog = page.getByRole('dialog');
    await expect(resultDialog.locator('.research-scenario-authority')).toBeVisible();
    const evidence = resultDialog.locator('.drawer-body > details.audit-details').last();
    await evidence.locator('summary').click();
    const detail = object(JSON.parse(await evidence.locator('pre').innerText()));
    const scenario = object(detail.scenarioProjection);
    const proof = object(detail.proofOutcome);
    const authority = object(operatorAudit.researchAuthority);
    const compatibility = object(operatorAudit.runtimeCompatibility);
    const qualification = object(detail.qualification);

    expect(detail.researchBackend).toBe('ortools_cp_sat_v3');
    expect(detail.qualificationAuthority).toBe('cp_sat');
    expect(detail.disposition).toBe('qualified');
    expect(qualification.failures).toEqual([]);
    expect(scenario.status).toBe('satisfied');
    expect(scenario.predicateEvidenceCount).toBe(11);
    expect(proof.achievedMode).toBe('exact');
    expect(proof.completionStatus).toBe('exact_complete');
    expect(proof.winningRootActionCount).toBe(1);
    expect(proof.claimSatisfied).toBe(true);
    expect(compatibility.status).toBe('passed');

    const resultHash = String(object(operatorAudit.executionReceipt).semanticResultHash ?? '');
    const traceHash = String(authority.traceHash ?? '');
    expect(resultHash).toMatch(/^[a-f0-9]{64}$/);
    expect(scenario.exactStateAvailable).toBe(true);
    expect(traceHash).toMatch(/^[a-f0-9]{64}$/);
    resultHashes.add(resultHash);
    traceHashes.add(traceHash);
    await resultDialog.locator('aside > header > button').click();
  }
  expect(resultHashes.size).toBe(5);
  expect(traceHashes.size).toBe(5);
  expect(statisticsStatuses.length).toBeGreaterThan(0);
  expect(new Set(statisticsStatuses)).toEqual(new Set([200]));
  expect(pageErrors).toEqual([]);

  const prepare = page.locator('.scenario-result-actions button.primary');
  await expect(prepare).toBeEnabled();
  await prepare.click();
  const batchStatus = page.locator('[data-render-status]');
  await expect(batchStatus).toHaveAttribute('data-render-status', /queued|running/);
  const players = page.getByTestId('scenario-prepared-player-link');
  await expect(players).toHaveCount(5);
  const playerLinks = await players.evaluateAll((links) => links.map((link) => link.getAttribute('href')));
  expect(new Set(playerLinks).size).toBe(5);
  const playerOpened = page.context().waitForEvent('page');
  await players.first().click();
  const player = await playerOpened;
  await player.waitForLoadState('domcontentloaded');
  await expect(player).toHaveURL(/\/c\/LAF-[A-F0-9]{10}$/);
  await player.close();

  if (process.env.BALOOT_OPERATOR_E2E_RENDER_FULL === '1') {
    await expect(batchStatus).toHaveAttribute('data-render-status', 'completed', {
      timeout: 1_500_000,
    });
  } else {
    // Full encoding is qualified by the separate five-video evidence command.
    await page.locator('.scenario-result-actions button.quiet-danger').click();
    await expect(batchStatus).toHaveAttribute('data-render-status', 'cancelled', {
      timeout: 30_000,
    });
  }
  });
});
