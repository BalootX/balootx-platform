import {type Page} from '@playwright/test';

// Narrow layouts replace the family buttons with a select, so pick whichever control is shown.
export async function selectChallenge(page: Page) {
  const picker = page.locator('.family-select select');
  if (await picker.isVisible()) await picker.selectOption('challenge');
  else {
    const name = await picker.locator('option[value="challenge"]').textContent();
    await page.locator('.family-nav').getByRole('button',{name:new RegExp(name!.trim())}).click();
  }
}
