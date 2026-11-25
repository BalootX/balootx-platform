import {expect, test} from '@playwright/test';

// Temporary: proves a hosted failure retains its trace and screenshot (#191). Removed before merge.
test('temporary probe for retained failure diagnostics', async ({page}) => {
  await page.goto('/foundry?section=overview');
  expect(true).toBe(false);
});
