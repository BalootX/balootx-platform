import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {runInNewContext} from 'node:vm';
import {createClient} from '../../scripts/publisher/client.mjs';
import {load} from '../../scripts/publisher/runtime.mjs';
import {sha} from '../../scripts/publisher/policy.mjs';
import {writeExclusive} from '../../scripts/publisher/ledger.mjs';
import {pageState, pointerTarget, composerDom, qualify} from '../../scripts/publisher/dom.mjs';
import {waitThenJudge} from '../../scripts/publisher/readiness.mjs';
import {runtimeFixture} from './runtime-fixture.mjs';

test('client uses native insertText, freshest pointer and the same account target before upload', async () => {
  const f = runtimeFixture();
  try {
    const account = load(resolve(f.b.evidenceRoot, 'tiktok-account.json'));
    let targetId = 'synthetic-changed-target', pointerReads = 0;
    const operations = [];
    const context = {assert, resolve, load, sha, writeExclusive, pageState, pointerTarget, composerDom, qualify,
      waitThenJudge, AbortSignal, guard: () => ({claimSha256: 'synthetic-current-claim'}),
      validatePacket: () => ({caption: f.caption, sha256: f.b.mediaSha256, bytes: f.b.mediaBytes}),
      fetch: async (url, request) => {
        assert.equal(url, f.config.controlUrl);
        assert.equal(request.headers['X-Publisher-Claim'], 'synthetic-current-claim');
        const body = JSON.parse(request.body); operations.push(body);
        let value = {};
        if (body.op === 'binding') value = {...account.binding, targetId};
        if (body.op === 'eval') value = body.arg.includes('pointerTarget') ? {x: ++pointerReads, y: 10, sameTarget: true} : true;
        if (body.op === 'native' && body.arg.method === 'DOM.getDocument') value = {root: {nodeId: 1}};
        if (body.op === 'native' && body.arg.method === 'DOM.querySelector') value = {nodeId: 2};
        return {ok: true, text: async () => JSON.stringify(value)};
      }};
    const factory = runInNewContext('(' + createClient.toString() + ')', context);
    const client = factory(f.runtime, 'TikTok');
    await assert.rejects(() => client.attach('#file'), /ACCOUNT_SESSION_CHANGED/);
    assert(!operations.some(op => op.op === 'native'));
    targetId = account.binding.targetId;
    await client.attach('#file');
    const upload = operations.find(op => op.arg?.method === 'DOM.setFileInputFiles');
    assert.deepEqual(upload.arg.params.files, [f.b.mediaPath]);
    await client.fill('#caption', f.caption);
    assert.equal(operations.find(op => op.arg?.method === 'Input.insertText').arg.params.text, f.caption);
    assert.equal(operations.filter(op => op.arg?.method === 'Input.dispatchKeyEvent').length, 2);
    await client.click('#submit');
    const pointerEvents = operations.filter(op => op.arg?.method === 'Input.dispatchMouseEvent');
    assert.equal(pointerReads, 2);
    assert(pointerEvents.every(op => op.arg.params.x === 2));
    assert.deepEqual(pointerEvents.map(op => op.arg.params.type), ['mousePressed', 'mouseReleased']);
  } finally { f.dispose(); }
});
