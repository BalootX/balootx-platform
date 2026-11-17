import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {runAction} from '../../scripts/publisher/operator.mjs';
import {readPost} from '../../scripts/publisher/post-readback.mjs';
import {confirmPost, writeExclusive} from '../../scripts/publisher/ledger.mjs';
import {sha} from '../../scripts/publisher/policy.mjs';
import {load} from '../../scripts/publisher/runtime.mjs';
import {runtimeFixture} from './runtime-fixture.mjs';

test('exclusive intent and ledger update refuse retries, stale ledgers and unconfirmed captions', () => {
  const f = runtimeFixture();
  try {
    const path = join(f.b.evidenceRoot, 'intent.json');
    writeExclusive(path, {status: 'INTENT_RESERVED'});
    assert.throws(() => writeExclusive(path, {}), /EEXIST/);
    const digest = sha(readFileSync(f.b.ledgerPath));
    const post = {id: '123', url: 'https://example.invalid/@synthetic_creator/video/123',
      postedAtUtc: '2099-01-01T12:01:00Z', actualCaption: f.caption, actualCaptionSha256: f.b.captionSha256,
      applicationAcknowledged: true, exact: true, within120s: true};
    for (const changes of [{applicationAcknowledged: false}, {exact: false}, {within120s: false}, {actualCaption: 'different'}]) {
      assert.throws(() => confirmPost(f.b.ledgerPath, digest, f.b, 'TikTok', {...post, ...changes}), /ACK_AND_EXACT_CAPTION/);
    }
    assert.throws(() => confirmPost(f.b.ledgerPath, sha('stale'), f.b, 'TikTok', post), /LEDGER_CHANGED/);
    const row = confirmPost(f.b.ledgerPath, digest, f.b, 'TikTok', {...post, artifactSha256: sha('untrusted extra field')});
    assert.equal(row.status, 'POSTED_ACTUAL_CAPTION_CONFIRMED');
    assert.equal(row.artifactSha256, f.b.mediaSha256);
    assert.throws(() => confirmPost(f.b.ledgerPath, sha(readFileSync(f.b.ledgerPath)), f.b, 'TikTok', post), /DUE_TASK_ROW/);
  } finally { f.dispose(); }
});

test('share reserves before one click, separates acknowledgments and never retries an unknown result', async () => {
  const f = runtimeFixture();
  try {
    const session = {targetId: 'synthetic-owned-target', claimSha256: sha('synthetic-claim')};
    const identity = {caption: f.caption, ledgerSha256: sha(readFileSync(f.b.ledgerPath)), sha256: f.b.mediaSha256, bytes: f.b.mediaBytes};
    for (const suffix of ['session', 'account', 'upload', 'input']) {
      writeFileSync(join(f.b.evidenceRoot, 'tiktok-' + suffix + '.json'), JSON.stringify({binding: session, ...identity}));
    }
    const context = {assert, guard: () => {}, existsSync, resolve, load, validatePacket: () => identity, Date,
      confirmPost: () => assert.fail('protocol or application ack cannot confirm ledger')};
    const run = runInNewContext('(' + runAction.toString() + ')', context);
    const state = Object.fromEntries(['scope', 'editor', 'counter', 'submit'].map(key => [key, {selector: '#' + key}]));
    let clicks = 0, waits = 0;
    const client = {runtime: f.runtime, platform: 'TikTok', composer: async () => state, op: async () => session,
      save: (name, value) => writeExclusive(join(f.b.evidenceRoot, name), value),
      click: async () => { assert(existsSync(join(f.b.evidenceRoot, 'tiktok-share-intent.json'))); clicks++; return {sameTarget: true}; },
      inspect: async () => ({blocker: false, text: 'Synthetic application acknowledged'}),
      waitProof: async (name, read, judge) => { waits++; const state = await read(); judge(state); return {state}; }};
    await run(client, 'share');
    assert.equal(clicks, 1); assert.equal(waits, 1);
    assert(load(join(f.b.evidenceRoot, 'tiktok-native-ack.json')).claim.includes('NOT_APPLICATION_SUCCESS'));
    assert(existsSync(join(f.b.evidenceRoot, 'tiktok-application-ack.json')));
    for (const action of ['share', 'fill', 'attach']) await assert.rejects(() => run(client, action), /NO_RETRY/);
    assert.equal(clicks, 1);
  } finally { f.dispose(); }
});

test('failed final click retains intent and produces no application success or second attempt', async () => {
  const f = runtimeFixture();
  try {
    const session = {targetId: 'synthetic-owned-target', claimSha256: sha('synthetic-claim')};
    const identity = {caption: f.caption, ledgerSha256: sha(readFileSync(f.b.ledgerPath)), sha256: f.b.mediaSha256, bytes: f.b.mediaBytes};
    for (const suffix of ['session', 'account', 'upload', 'input']) writeFileSync(join(f.b.evidenceRoot, 'tiktok-' + suffix + '.json'), JSON.stringify({binding: session, ...identity}));
    const run = runInNewContext('(' + runAction.toString() + ')', {assert, guard: () => {}, existsSync, resolve, load, validatePacket: () => identity, Date});
    const state = Object.fromEntries(['scope', 'editor', 'counter', 'submit'].map(key => [key, {selector: '#' + key}]));
    let clicks = 0;
    const client = {runtime: f.runtime, platform: 'TikTok', composer: async () => state, op: async () => session,
      save: (name, value) => writeExclusive(join(f.b.evidenceRoot, name), value),
      click: async () => { clicks++; throw Error('SYNTHETIC_UNKNOWN_NATIVE_RESULT'); }};
    await assert.rejects(() => run(client, 'share'), /UNKNOWN_NATIVE_RESULT/);
    await assert.rejects(() => run(client, 'share'), /NO_RETRY/);
    assert.equal(clicks, 1);
    assert(!existsSync(join(f.b.evidenceRoot, 'tiktok-application-ack.json')));
  } finally { f.dispose(); }
});

test('post readback distinguishes exact caption from changed, wrong author, wrong URL and challenge', async () => {
  const f = runtimeFixture();
  try {
    const fixed = Date.parse(f.b.slotUtc) + 1000;
    const read = runInNewContext('(' + readPost.toString() + ')', {assert, sha,
      Date: class extends Date {static now() {return fixed;}}, tiktokPostDom: () => {}});
    const url = 'https://example.invalid/@synthetic_creator/video/123';
    let state = {url, editing: false, captions: [{author: f.b.account, text: f.caption}]}, blocker = false, reads = 0;
    const client = {runtime: f.runtime, inspect: async () => ({blocker}), dom: async () => { reads++; return state; },
      waitProof: async (name, read, judge) => { const state = await read(); judge(state); return {state}; }};
    const options = {url, caption: f.caption, deadline: fixed + 1000};
    assert((await read(client, options)).exact);
    state.captions[0].text = 'different'; assert.equal((await read(client, options)).exact, false);
    state.captions[0].author = 'another_creator'; await assert.rejects(() => read(client, options), /AUTHOR_BOUND/);
    state.url += '/different'; await assert.rejects(() => read(client, options), /EXACT_NEW_URL/);
    blocker = true; await assert.rejects(() => read(client, options), /OWNER_ONLY_PROMPT/);
    const before = reads;
    await assert.rejects(() => read(client, {...options, deadline: fixed - 1}), /120S_DEADLINE/);
    assert.equal(reads, before);
  } finally { f.dispose(); }
});
