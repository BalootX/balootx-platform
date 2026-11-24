import {readFileSync} from 'node:fs';
import {expect, test} from '@playwright/test';

const cards = `
  <main class="app-shell decision-audit-list">
    <details open>
      <summary><span>١</span><b>الجنوب</b><i>لعب</i><em>مؤكد</em></summary>
      <div class="decision-audit-grid">
        <div><span>اللعب الفعلي</span><b dir="ltr">C9</b></div>
        <div><span>اللعب الموصى به</span><b dir="ltr">C9</b></div>
        <div class="wide research-score-breakdown"><span>تفاصيل النتيجة</span><b>٨٠</b></div>
        <div class="wide decision-context"><span>سياق القرار</span><p>معلومات اللاعب المتاحة</p></div>
        <div class="wide forced-explanation"><span>لعب إجباري</span><p>اختيار واحد متاح</p></div>
        <div class="wide decision-alternatives"><span>الخيارات البديلة</span><small>لا توجد خيارات أخرى</small></div>
        <details class="wide decision-provenance"><summary>مصدر التحليل</summary></details>
      </div>
    </details>
  </main>`;

test('audit card variants retain their appearance', async ({page}, testInfo) => {
  const baseline = process.env.BALOOT_FOUNDRY_AUDIT_BASE_CSS;
  const candidate = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');
  await page.goto('/healthz');
  await page.setContent(`<html lang="ar" dir="rtl"><body>${cards}</body></html>`);
  const panel = page.locator('.decision-audit-list');

  const capture = async (css: string, file: string) => {
    const style = await page.addStyleTag({content: css});
    const png = await panel.screenshot({path: testInfo.outputPath(file), animations: 'disabled'});
    await style.evaluate(element => element.remove());
    return png;
  };

  const before = baseline ? await capture(readFileSync(baseline, 'utf8'), 'audit-before.png') : undefined;
  const after = await capture(candidate, 'audit-after.png');
  if (before) expect(after.equals(before)).toBe(true);
  expect(candidate.replace(/^@media\(prefers-reduced-motion:reduce\).*$/gm, '')).not.toContain('!important');
});
