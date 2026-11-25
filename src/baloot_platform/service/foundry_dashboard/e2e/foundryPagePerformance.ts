import {expect, type Browser, type Page} from '@playwright/test';
import {readFile} from 'node:fs/promises';

const sampleCount = 12;
const viewport = {width: 1280, height: 900};
const eventDurationThresholdMilliseconds = 16;

async function samplePage(page: Page) {
  await page.route((url) => url.pathname === '/api/v2/research-runs',
    (route) => route.fulfill({json: {items: []}}));
  await page.goto('/foundry');
  await expect(page.locator('.readiness-grid')).toHaveAttribute('aria-busy', 'false');
  await page.evaluate(async () => { await document.fonts.ready; });

  const readEntries = await page.evaluateHandle((durationThreshold) => {
    const types = ['largest-contentful-paint', 'event', 'first-input'];
    if (!types.every((type) => PerformanceObserver.supportedEntryTypes.includes(type))) {
      throw new Error('Chromium must support LCP, Event Timing and first-input entries');
    }
    const entries: PerformanceEntry[] = [];
    const observer = new PerformanceObserver((list) => entries.push(...list.getEntries()));
    observer.observe({type: 'largest-contentful-paint', buffered: true});
    observer.observe({type: 'event', durationThreshold});
    // First-input entries also cover the first input below the event observer's threshold.
    observer.observe({type: 'first-input', buffered: true});
    return () => {
      entries.push(...observer.takeRecords());
      return entries;
    };
  }, eventDurationThresholdMilliseconds);

  await page.locator('.workspace-intro button.primary').click();
  await expect(page.locator('#research-bundle-select')).toBeEnabled();
  await page.waitForFunction((read) => performance.eventCounts.get('click') === 1
    && read().some((entry) => entry.entryType === 'first-input'), readEntries);

  const sample = await readEntries.evaluate((read) => {
    const entries = read();
    const lcp = entries.filter((entry) => entry.entryType === 'largest-contentful-paint').at(-1);
    const input = entries.find((entry) => entry.entryType === 'first-input') as PerformanceEventTiming;
    const interaction = entries.filter((entry) => entry.entryType === 'event'
      && (entry as PerformanceEventTiming).interactionId === input.interactionId);
    return {
      lcpMilliseconds: lcp?.startTime,
      inpMilliseconds: Math.max(input.duration, ...interaction.map((entry) => entry.duration)),
      interactionId: input.interactionId,
    };
  });
  expect(sample.lcpMilliseconds, 'Chromium LCP entry').toBeGreaterThan(0);
  expect(sample.interactionId, 'Chromium interaction id').toBeGreaterThan(0);
  expect(Number.isFinite(sample.inpMilliseconds), 'finite Chromium interaction duration').toBe(true);
  return {lcpMilliseconds: sample.lcpMilliseconds!, inpMilliseconds: sample.inpMilliseconds};
}

function summarize(samples: number[], budgetMilliseconds: number) {
  const sorted = [...samples].sort((left, right) => left - right);
  const p75Milliseconds = sorted[Math.ceil(sorted.length * 0.75) - 1]!;
  return {
    p75Milliseconds,
    budgetMilliseconds,
    underBudget: p75Milliseconds < budgetMilliseconds,
    headroomPercent: 100 * (budgetMilliseconds - p75Milliseconds) / budgetMilliseconds,
  };
}

export async function measureFoundryPage(browser: Browser, baseURL: string, locale: string) {
  const design = await readFile(new URL('../../../../../DESIGN.md', import.meta.url), 'utf8');
  const budgets = /^- The page budgets are p75 Largest Contentful Paint under (\d+(?:\.\d+)?) seconds and p75 Interaction to Next Paint under (\d+(?:\.\d+)?) ms\.$/m.exec(design);
  const lcpBudget = Number(budgets?.[1]) * 1000;
  const inpBudget = Number(budgets?.[2]);
  if (![lcpBudget, inpBudget].every((budget) => Number.isFinite(budget) && budget > 0)) {
    throw new Error('Foundry p75 budgets are missing or invalid in DESIGN.md');
  }

  const samples = [];
  for (let index = 0; index < sampleCount; index += 1) {
    const page = await browser.newPage({baseURL, locale, viewport});
    try {
      samples.push(await samplePage(page));
    } finally {
      await page.close();
    }
  }

  return {
    pagePath: '/foundry',
    interaction: 'open-research-workspace',
    viewport,
    locale,
    eventDurationThresholdMilliseconds,
    percentileMethod: 'nearest-rank',
    budgetSource: 'DESIGN.md#foundry-dashboard',
    budgetMode: 'record-only',
    sampleCount,
    samples,
    lcp: summarize(samples.map((sample) => sample.lcpMilliseconds), lcpBudget),
    inp: summarize(samples.map((sample) => sample.inpMilliseconds), inpBudget),
  };
}
