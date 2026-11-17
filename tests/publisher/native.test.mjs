import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {nativeRequest, extend, load as loaderHook} from '../../scripts/publisher/native-loader.mjs';
import {localConfig} from '../../scripts/publisher/runtime.mjs';
import {sha} from '../../scripts/publisher/policy.mjs';
import {runtimeFixture} from './runtime-fixture.mjs';

test('using the loader without local configuration cannot load an unguarded helper', async () => {
  const hook = runInNewContext('(' + loaderHook.toString() + ')', {localConfig, process: {env: {}}});
  let loads = 0;
  await assert.rejects(() => hook('file:///synthetic-helper.mjs', {}, async () => { loads++; }), /EXPLICIT_LOCAL_CONFIG/);
  assert.equal(loads, 0);
});

test('native allowlist forbids credential APIs and seals caption, file, bytes and digest', () => {
  const f = runtimeFixture();
  try {
    for (const method of ['Network.getAllCookies', 'Storage.getStorageKeyForFrame', 'Target.attachToTarget', 'Runtime.evaluate', 'Page.captureScreenshot']) {
      assert.throws(() => nativeRequest({method}, f.b), /METHOD_NOT_ALLOWED/);
    }
    assert.equal(nativeRequest({method: 'Input.insertText', params: {text: f.caption}}, f.b).params.text, f.caption);
    assert.throws(() => nativeRequest({method: 'Input.insertText', params: {text: f.caption + ' '}}, f.b), /SEALED_CAPTION/);
    assert.throws(() => nativeRequest({method: 'DOM.setFileInputFiles', params: {files: ['another.bin']}}, f.b), /EXACT_FILE/);
    nativeRequest({method: 'DOM.setFileInputFiles', params: {files: [f.b.mediaPath]}}, f.b);
    writeFileSync(f.b.mediaPath, 'changed fixture');
    assert.throws(() => nativeRequest({method: 'DOM.setFileInputFiles', params: {files: [f.b.mediaPath]}}, f.b), /UPLOAD_HASH/);
  } finally { f.dispose(); }
});

test('in-memory helper extension retains one connection and guards every local request', async () => {
  const f = runtimeFixture();
  try {
    const source = `const ws = new WebSocket('wss://example.invalid');
export async function request(req, body, targetId, session, send) {
let out;
const { op, arg } = JSON.parse(body || '{}');
if (op === 'navigate') out = 'navigated';
else if (op === 'eval') out = 'synthetic-eval-not-executed';
return out;
}`;
    f.config.helperSha256 = sha(source);
    const derived = extend(source, f.runtime);
    assert.equal((derived.match(/new WebSocket/g) || []).length, 1);
    assert.throws(() => extend(source + ' ', f.runtime), /SOURCE_CHANGED/);
    let guards = 0, sends = 0;
    const context = {URL, publisherRuntime: f.runtime, publisherNative: nativeRequest,
      publisherGuard: () => { guards++; return {claimSha256: 'synthetic-claim-hash'}; }};
    const route = derived.slice(derived.indexOf('export async function request')).replace('export async function', 'async function');
    const request = runInNewContext(route + '\nrequest;', context);
    const headers = {'x-publisher-claim': 'synthetic-claim-hash'};
    const send = async (method, params, session) => { sends++; return {method, params, session}; };
    const call = (op, arg, req = {method: 'POST', headers}, target = 'synthetic-owned-target') =>
      request(req, JSON.stringify({op, arg}), target, 'synthetic-owned-session', send);
    const result = JSON.parse(await call('native', {method: 'Input.insertText', params: {text: f.caption}}));
    assert.equal(result.session, 'synthetic-owned-session');
    assert.equal(sends, 1);
    await assert.rejects(() => call('native', {method: 'Page.bringToFront'}, undefined, f.config.protectedTargetIds[0]), /PROTECTED_TARGET/);
    await assert.rejects(() => call('eval', 'not executed', {method: 'POST', headers: {...headers, origin: 'https://example.invalid'}}), /OWN_LOCAL_REQUEST/);
    await assert.rejects(() => call('eval', 'not executed', {method: 'POST', headers: {}}), /OWN_LOCAL_REQUEST/);
    await assert.rejects(() => call('navigate', 'https://example.invalid.evil.invalid/'), /NAVIGATION_ORIGIN/);
    await assert.rejects(() => call('navigate', 'data:text/plain,synthetic'), /NAVIGATION_ORIGIN/);
    assert.equal(await call('navigate', f.config.profileCheckUrl), 'navigated');
    assert.equal(guards, 7);
    assert.equal(sends, 1);
  } finally { f.dispose(); }
});
