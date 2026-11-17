import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFileSync, writeFileSync, mkdirSync, symlinkSync} from 'node:fs';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import {localConfig, assertDue, assertActivated, assertClaim, ownedPath, validatePacket, load} from '../../scripts/publisher/runtime.mjs';
import {releaseAuthority, historyAdmission, sha} from '../../scripts/publisher/policy.mjs';
import {currentRow} from '../../scripts/publisher/ledger.mjs';
import {approvedCaption} from '../../scripts/publisher/captions.mjs';
import {schedulePlan} from '../../scripts/publisher/operator.mjs';
import {runtimeFixture} from './runtime-fixture.mjs';

test('local config fails closed before reading files in CI or without explicit config', () => {
  assert.throws(() => localConfig(undefined, {}), /EXPLICIT_LOCAL_CONFIG/);
  assert.throws(() => localConfig('nonexistent.json', {GITHUB_ACTIONS: 'true'}), /FORBIDDEN_IN_CI/);
  const f = runtimeFixture();
  try {
    assert.equal(localConfig(f.configPath, {}).binding.account, f.b.account);
    for (const controlUrl of ['https://example.invalid/', 'http://127.0.0.1.evil.invalid:12345/', 'http://127.0.0.1:12345/?forward=1']) {
      f.put('config.json', {...f.config, controlUrl});
      assert.throws(() => localConfig(f.configPath, {}), /LOOPBACK/);
    }
    f.put('config.json', {...f.config, applicationAckTexts: {}});
    assert.throws(() => localConfig(f.configPath, {}), /APPLICATION_ACK_CONFIG/);
  } finally { f.dispose(); }
});

test('live helper CLI cannot start from a fork or CI environment', () => {
  assert.throws(() => execFileSync(process.execPath, [fileURLToPath(new URL('../../scripts/publisher/operator.mjs', import.meta.url)),
    '--config', 'nonexistent.json', '--action', 'helper'], {env: {...process.env, CI: 'true'}, windowsHide: true, stdio: 'pipe'}),
  error => error.status === 1 && String(error.stderr).includes('LIVE_BROWSER_FORBIDDEN_IN_CI'));
});

test('due guard rejects early, invalid and expired slots', () => {
  const binding = {slotUtc: '2099-01-01T12:00:00Z'}, due = Date.parse(binding.slotUtc);
  assert.throws(() => assertDue(binding, due - 1), /BEFORE_RELEASED_SLOT/);
  assertDue(binding, due);
  assert.throws(() => assertDue(binding, Date.parse('2099-01-02T00:00:00Z')), /EXPIRED/);
  assert.throws(() => assertDue({slotUtc: 'invalid'}, due), /BEFORE_RELEASED_SLOT/);
});

test('activation binds every source, config, current authority and helper', () => {
  const f = runtimeFixture();
  try {
    assertActivated(f.runtime);
    const plan = schedulePlan(f.runtime);
    assert.deepEqual(plan.map(job => job.maxAttempts), [1, 1]);
    assert(plan.every(job => job.dueUtc === f.b.slotUtc && job.requiresExclusiveLocalClaim));
    f.put('execution.json', {...f.pin, sourceHashes: {}});
    assert.throws(() => assertActivated(f.runtime), /COMPLETE_SOURCE_FREEZE/);
    f.put('execution.json', f.pin);
    writeFileSync(f.config.helperPath, '// changed fixture');
    assert.throws(() => assertActivated(f.runtime), /HELPER_CHANGED/);
  } finally { f.dispose(); }
});

test('claim must match the exact local lock bytes, owner and live holder', () => {
  const f = runtimeFixture();
  try {
    assert.equal(assertClaim(f.runtime, f.env).holderPid, process.pid);
    for (const changes of [{CHROME_LOCK_CLAIM_SHA256: sha('stale')}, {CHROME_LOCK_HOLDER_PID: '0'}, {CI: 'true'}]) {
      assert.throws(() => assertClaim(f.runtime, {...f.env, ...changes}));
    }
    assert.throws(() => assertClaim(f.runtime, {}), /LOCAL_LOCK/);
  } finally { f.dispose(); }
});

test('owned evidence cannot escape via a sibling prefix or symlink', () => {
  const f = runtimeFixture();
  try {
    assert(ownedPath(f.b.evidenceRoot, f.b.admissionProofPaths.TikTok));
    const sibling = join(f.root, 'evidence-other'); mkdirSync(sibling);
    const external = join(sibling, 'external.json'); writeFileSync(external, '{}');
    assert.throws(() => ownedPath(f.b.evidenceRoot, external), /OWNED_LOCAL_PATH/);
    const link = join(f.b.evidenceRoot, 'outside');
    symlinkSync(sibling, link, process.platform === 'win32' ? 'junction' : 'dir');
    assert.throws(() => ownedPath(f.b.evidenceRoot, join(link, 'external.json')), /OWNED_LOCAL_PATH/);
  } finally { f.dispose(); }
});

test('offline packet fixture checks all bindings and fresh evidence without browser or transport', () => {
  const f = runtimeFixture();
  try {
    const context = {assert, guard: () => load(f.b.admissionProofPaths.TikTok).session, readFileSync, sha, releaseAuthority,
      historyAdmission: (b, ledger, platform, options) => historyAdmission(b, ledger, platform, {...options, nowMs: Date.parse(b.slotUtc) + 1000}), approvedCaption,
      currentRow, ownedPath, load, resolve: (...parts) => join(...parts), JSON,
      Date: class extends Date {static now() {return Date.parse(f.b.slotUtc) + 1000;}}};
    const validate = runInNewContext('(' + validatePacket.toString() + ')', context);
    assert.equal(validate(f.runtime, 'TikTok').caption, f.caption);
    const proof = load(f.b.admissionProofPaths.TikTok);
    writeFileSync(f.b.admissionProofPaths.TikTok, JSON.stringify({...proof, session: {...proof.session, claimSha256: sha('stale-claim')}}));
    assert.throws(() => validate(f.runtime, 'TikTok'), /ACCOUNT_SESSION/);
    writeFileSync(f.b.admissionProofPaths.TikTok, JSON.stringify(proof));
    writeFileSync(f.b.mediaPath, 'changed fixture media');
    assert.throws(() => validate(f.runtime, 'TikTok'), /MEDIA_APPROVAL/);
  } finally { f.dispose(); }
});
