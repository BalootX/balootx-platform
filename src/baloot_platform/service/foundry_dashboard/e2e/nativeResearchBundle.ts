import {expect, type Locator, type Page} from '@playwright/test';

type ScenarioCapability = {
  backendIdentity?: {backendId?: string};
};

async function optionValues(select: Locator) {
  return select.locator('option').evaluateAll((options) => options.map(
    (option) => (option as HTMLOptionElement).value,
  ));
}

export async function nativeResearchBundleRevision(page: Page, bundle: Locator) {
  await expect(bundle).toBeVisible();
  await expect.poll(async () => (await optionValues(bundle)).filter(Boolean).length)
    .toBeGreaterThan(0);
  const observed: string[] = [];
  for (const revision of (await optionValues(bundle)).filter(Boolean)) {
    const backendId = await page.evaluate(async (bundleRevisionId) => {
      const response = await fetch(
        `/api/v2/domain-capabilities?bundleRevisionId=${encodeURIComponent(bundleRevisionId)}`,
      );
      if (!response.ok) return 'unavailable';
      const payload = await response.json() as {
        research?: {scenarioResearchV4?: ScenarioCapability};
      };
      return payload.research?.scenarioResearchV4?.backendIdentity?.backendId ?? 'unavailable';
    }, revision);
    observed.push(`${revision}:${backendId}`);
    if (backendId === 'ortools_cp_sat_v3') return revision;
  }
  throw new Error(`native_ortools_bundle_not_published:${observed.join(',')}`);
}
