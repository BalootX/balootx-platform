import {expect, test, type Page} from '@playwright/test';

const target = 'scenarioResearchV4.qualityExecution.familyTarget';
const family = `${target}.catalog.families.0`;
const expression = `${family}.template.expression`;

async function open(page: Page, path: string) {
  const section = page.locator(`[data-config-path="${path}"]`);
  await expect(section).toBeVisible();
  if (await section.getAttribute('open') === null) await section.locator(':scope > summary').click();
  return section;
}

async function openTemplate(page: Page) {
  for (const path of ['scenarioResearchV4', 'scenarioResearchV4.qualityExecution',
    target, `${target}.catalog`, `${target}.catalog.families`, family,
    `${family}.template`, expression, `${expression}.children`, `${expression}.children.1`,
    `${expression}.children.1.arguments`]) await open(page,path);
}

async function selectChallenge(page: Page) {
  const picker = page.locator('.family-select select');
  if (await picker.isVisible()) await picker.selectOption('challenge');
  else {
    const name = await picker.locator('option[value="challenge"]').textContent();
    await page.locator('.family-nav').getByRole('button',{name:new RegExp(name!.trim())}).click();
  }
}

async function openNativeSettings(page: Page) {
  // New visual authoring uses the explicitly selected native bundle's copy;
  // a historical process bundle retains its own advanced controls on reload.
  await page.goto('/foundry?section=research');
  await expect(page.locator('#scenario-studio-title')).toBeVisible();
  await page.getByRole('button', {name:/الإعدادات والقواعد/}).click();
  await selectChallenge(page);
  await page.getByPlaceholder('أدخل مفتاح الإدارة وقت الحفظ').fill(process.env.BALOOT_OPERATOR_E2E_API_KEY!);
}

async function createDraft(page: Page, version: string): Promise<string> {
  const headers = {'x-api-key': process.env.BALOOT_OPERATOR_E2E_API_KEY!, 'x-baloot-role': 'admin'};
  const profiles = '/api/v1/configuration-registry/profiles';
  const drafts = await page.request.get(profiles, {
    headers, params: {family: 'challenge', status: 'draft'},
  });
  await expect(drafts).toBeOK();
  const seeds = (await drafts.json()).items.filter((item: {createdBy: string}) => item.createdBy === 'OPERATOR-E2E');
  expect(seeds).toHaveLength(1);
  const source = await page.request.get(`${profiles}/${seeds[0].revisionId}`, {headers});
  await expect(source).toBeOK();
  const {payload} = await source.json();
  payload.profile.version = version;
  const response = await page.request.post('/api/v2/configuration/profiles', {
    headers, data: {family: 'challenge', payload, publish: false},
  });
  await expect(response).toBeOK();
  return (await response.json()).revisionId;
}

test('private family template edits save and reload through typed UI without JSON', async ({page}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error=>errors.push(error.message));
  const revisionId = await createDraft(page, '99.0.1');
  await openNativeSettings(page);
  const revisions = page.locator('.revision-picker select');
  await revisions.selectOption(revisionId);
  await expect(revisions).toHaveValue(revisionId);
  await openTemplate(page);
  const kindPicker = page.locator(`[data-config-path="${expression}.children.1"]`)
    .locator(':scope > .json-editor > .field-label > select');
  for (const name of ['فعل لعب يطابق محدد الورق','فائز الأكلة يطابق محدد الورق','عدد أوراق الأكلة المطابقة']) {
    await expect(kindPicker.getByRole('option',{name,exact:true})).toBeEnabled();
  }
  const minimum = page.locator(`#config-${expression.replaceAll('.','-')}-children-1-arguments-minimum`);
  await expect(minimum).toHaveValue('2');
  await minimum.fill('1');
  // Changing a frozen template requires explicit removal of its enclosing hashes.
  for (const path of [`${target}.catalog`,target]) {
    const section = page.locator(`[data-config-path="${path}"]`);
    const hash = section.locator(`#config-${path.replaceAll('.','-')}-sha256`);
    await hash.locator('..').getByRole('button').last().click();
  }
  await page.screenshot({path:testInfo.outputPath('typed-family-draft.png'),fullPage:true});
  const saved = page.waitForResponse(response =>
    response.request().method() === 'PATCH' && response.url().includes('/api/v2/configuration/profiles/'));
  await page.getByRole('button',{name:'حفظ كمسودة',exact:true}).click();
  expect((await saved).status()).toBe(200);
  await expect(page.getByRole('button',{name:'افحص المسودة',exact:true})).toBeEnabled();
  await expect(page.getByRole('button',{name:'حفظ كمسودة',exact:true})).toBeDisabled();
  await page.reload();
  await expect(page.locator('.settings-layout')).toBeVisible();
  // Reopen the saved draft rather than relying on the pre-save React state.
  await openNativeSettings(page);
  await revisions.selectOption(revisionId);
  await expect(revisions).toHaveValue(revisionId);
  await openTemplate(page);
  await expect(minimum).toHaveValue('1');
  await page.getByRole('button',{name:'افحص المسودة',exact:true}).click();
  await expect(page.locator('.draft-proof-flow .status')).toHaveText('المسودة سليمة');
  expect(errors).toEqual([]);
  await page.screenshot({path:testInfo.outputPath('typed-family-saved.png'),fullPage:true});
  await minimum.scrollIntoViewIfNeeded();
  await page.screenshot({path:testInfo.outputPath('typed-family-controls.png')});
});

for (const [mode, version] of [['desktop', '99.0.2'], ['@mobile', '99.0.3']]) test(`delayed payoff controls enable save reload validate and remove ${mode}`, async ({page}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error=>errors.push(error.message));
  const revisionId = await createDraft(page, version);
  await openNativeSettings(page);
  const revisions = page.locator('.revision-picker select');
  await revisions.selectOption(revisionId);
  await expect(revisions).toHaveValue(revisionId);
  await open(page, 'tactical_detectors');
  const label = 'إثبات فائدة الورقة المتأخرة';
  const sectionPath = 'tactical_detectors.nativeDelayedPayoff';
  const section = page.locator(`[data-config-path="${sectionPath}"]`);
  await page.getByRole('switch',{name:label,exact:true}).click();
  await open(page, sectionPath);
  const metric = section.getByLabel('وحدة قياس المكسب',{exact:true});
  const minimum = section.getByLabel('أقل مكسب مثبت',{exact:true});
  await expect(metric).toHaveValue('');
  await expect(minimum).toHaveValue('');
  await expect(minimum).toHaveAttribute('min','0');
  await expect(minimum).toHaveAttribute('max','100');
  await expect(minimum).toHaveAttribute('step','any');
  await expect(metric).toHaveAccessibleDescription(/بدون مكافأة الأكلة الأخيرة/);
  await expect(minimum).toHaveAccessibleDescription(/لا نختار قيمة نيابةً عنك/);
  await section.locator(':scope > summary .info-badge').click();
  await expect(section.locator(':scope > summary .info-popover')).toContainText('معرفة كل الأوراق فقط');
  for (const unit of ['raw_card_points','awarded_point_margin']) {
    await metric.selectOption(unit);
    await expect(metric.locator('option:checked')).not.toHaveText(unit);
  }
  await metric.selectOption('raw_card_points');
  await minimum.fill('20.5');
  const saved = page.waitForResponse(response=>response.request().method()==='PATCH'
    && response.url().includes('/api/v2/configuration/profiles/'));
  await page.getByRole('button',{name:'حفظ كمسودة',exact:true}).click();
  const response = await saved;
  expect(response.status()).toBe(200);
  expect((await response.json()).payload.tactical_detectors.nativeDelayedPayoff).toEqual({gainMetric:'raw_card_points',minimumGain:20.5});
  await page.reload();
  await openNativeSettings(page);
  await revisions.selectOption(revisionId);
  await expect(revisions).toHaveValue(revisionId);
  await open(page, 'tactical_detectors');
  await open(page, sectionPath);
  await expect(metric).toHaveValue('raw_card_points');
  await expect(minimum).toHaveValue('20.5');
  await page.getByRole('button',{name:'افحص المسودة',exact:true}).click();
  await expect(page.locator('.draft-proof-flow .status')).toHaveText('المسودة سليمة');
  await minimum.scrollIntoViewIfNeeded();
  await page.screenshot({path:testInfo.outputPath('delayed-payoff-controls.png')});
  await section.locator(':scope > summary').getByRole('button',{name:new RegExp(` · ${label}$`)}).click();
  await expect(page.getByRole('switch',{name:label,exact:true})).toHaveAttribute('aria-checked','false');
  const removed = page.waitForResponse(response=>response.request().method()==='PATCH'
    && response.url().includes('/api/v2/configuration/profiles/'));
  await page.getByRole('button',{name:'حفظ كمسودة',exact:true}).click();
  expect((await (await removed).json()).payload.tactical_detectors).not.toHaveProperty('nativeDelayedPayoff');
  expect(errors).toEqual([]);
});
