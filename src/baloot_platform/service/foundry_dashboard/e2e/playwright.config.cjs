const {defineConfig} = require('@playwright/test');
const {resolve} = require('node:path');
const {randomUUID} = require('node:crypto');

const root = resolve(__dirname, '../../../../..');
const port = Number(process.env.BALOOT_OPERATOR_E2E_PORT || 8174);
process.env.BALOOT_OPERATOR_E2E_API_KEY ??= randomUUID();

module.exports = defineConfig({
  testDir: __dirname,
  fullyParallel: false,
  workers: process.env.CI ? 2 : 1,
  timeout: 120_000,
  expect: {timeout: 30_000},
  use: {
    actionTimeout: 30_000,
    baseURL: `http://127.0.0.1:${port}`,
    browserName: 'chromium',
    headless: true,
    locale: 'ar-SA',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {name: 'desktop-chrome', grepInvert: /@(mobile|reduced-motion|zoom)/},
    {
      name: 'mobile-touch-chrome',
      grep: /@mobile/,
      use: {viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true},
    },
    {
      name: 'reduced-motion-chrome',
      grep: /@reduced-motion/,
      use: {reducedMotion: 'reduce'},
    },
    {
      name: 'zoom-chrome',
      grep: /@zoom/,
      use: {viewport: {width: 1280, height: 900}},
    },
  ],
  webServer: {
    command: 'python -B tests/browser/serve_foundry_operator_e2e.py',
    cwd: root,
    env: {
      PYTHONPATH: resolve(root, 'src'),
      BALOOT_OPERATOR_E2E_FAMILY_FIXTURE: '1',
      BALOOT_API_KEY: process.env.BALOOT_OPERATOR_E2E_API_KEY,
    },
    url: `http://127.0.0.1:${port}/healthz`,
    reuseExistingServer: process.env.BALOOT_REUSE_OPERATOR_E2E_SERVER === '1',
    timeout: 120_000,
  },
  outputDir: resolve(root, process.env.BALOOT_OPERATOR_E2E_OUTPUT_DIR || 'output/playwright/acc19'),
  reporter: [['list']],
});
