import assert from 'node:assert/strict';
import {readFileSync, openSync, closeSync, writeFileSync, fsyncSync, renameSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {sha, visibleFingerprint} from './policy.mjs';

export function validateLedger(ledger, account) {
  assert.equal(ledger.schemaVersion, 1, 'LEDGER_SCHEMA');
  assert.equal(ledger.account, account, 'LEDGER_ACCOUNT');
  assert(Array.isArray(ledger.records), 'LEDGER_RECORDS');
  const keys = new Set();
  for (const row of ledger.records) {
    assert(['Instagram', 'TikTok'].includes(row.platform), 'LEDGER_PLATFORM');
    assert(row.deal && Number.isFinite(Date.parse(row.slotUtc)), 'LEDGER_RELEASE_SCOPE');
    assert(/^[a-f0-9]{64}$/.test(row.artifactSha256) && /^[a-f0-9]{64}$/.test(row.captionSha256), 'LEDGER_DIGEST');
    assert(row.scenarioFamily && row.visibleOpenerSha256 === visibleFingerprint(row.visibleOpener), 'LEDGER_FINGERPRINT');
    const key = row.platform + ':' + (row.id || row.deal + ':' + row.artifactSha256 + ':' + row.slotUtc);
    assert(!keys.has(key), 'DUPLICATE_LEDGER_IDENTITY');
    keys.add(key);
    if (row.id || row.url) assert(row.id && row.url, 'INCOMPLETE_POST_IDENTITY');
    if (row.status === 'POSTED_ACTUAL_CAPTION_CONFIRMED') {
      const posted = row.postedAtUtc || row.publishedAtUtc || row.postedAtUtcRange?.at(-1) ||
        row.publishedAtUtcCandidates?.at(-1) || row.postedAtUtcCandidates?.at(-1);
      assert(row.id && row.url && Number.isFinite(Date.parse(posted)), 'ACTUAL_POST_REQUIRED');
      if (row.applicationAcknowledged !== undefined) assert(row.applicationAcknowledged === true, 'ACTUAL_ACK_AND_CAPTION_REQUIRED');
      if (row.actualCaptionSha256 !== undefined) assert.equal(row.actualCaptionSha256, row.captionSha256, 'ACTUAL_ACK_AND_CAPTION_REQUIRED');
    }
  }
  return ledger;
}

export function currentRow(binding, ledger, platform) {
  validateLedger(ledger, binding.account);
  const rows = ledger.records.filter(row => row.platform === platform && row.artifactSha256 === binding.mediaSha256);
  assert.equal(rows.length, 1, 'EXACT_CURRENT_LEDGER_ROW_REQUIRED');
  const row = rows[0];
  assert(row.deal === binding.deal && row.slotUtc === binding.slotUtc && row.releaseReceipt === binding.releasePath &&
    row.captionSha256 === binding.captionSha256 && row.scenarioFamily === binding.opener.family, 'CURRENT_RELEASE_ROW_BINDING');
  assert(row.status === 'RELEASED_SLOT_BOUND_CAPTION_PATH_CONFIRMED' && row.maxPosts === 1 &&
    row.scheduledTaskId === binding.scheduledTaskId, 'CURRENT_DUE_TASK_ROW');
  assert(!row.id && !row.url && !row.shareAttemptedAtUtc, 'NO_OWN_PLATFORM_REPEAT');
  assert(row.postingHold?.result !== 'HOLD' && row.postingHold?.portfolioWordRequired !== true, 'POSTING_HOLD');
  return row;
}

export function writeExclusive(path, value) {
  const fd = openSync(path, 'wx', 0o600);
  try { writeFileSync(fd, JSON.stringify(value, null, 2) + '\n'); fsyncSync(fd); }
  finally { closeSync(fd); }
  return value;
}

// The caller holds the local browser lock; the hash also detects outside ledger edits.
export function confirmPost(path, expectedSha256, binding, platform, post) {
  const bytes = readFileSync(path);
  assert.equal(sha(bytes), expectedSha256, 'LEDGER_CHANGED_RECONCILE_FIRST');
  const ledger = JSON.parse(bytes);
  const row = currentRow(binding, ledger, platform);
  assert(post.applicationAcknowledged === true && post.exact === true && post.within120s === true &&
    post.actualCaptionSha256 === binding.captionSha256 && sha(post.actualCaption) === binding.captionSha256 &&
    post.id && post.url && Number.isFinite(Date.parse(post.postedAtUtc)), 'ACTUAL_ACK_AND_EXACT_CAPTION_REQUIRED');
  Object.assign(row, {id: post.id, url: post.url, postedAtUtc: post.postedAtUtc, applicationAcknowledged: true,
    actualCaptionSha256: post.actualCaptionSha256, status: 'POSTED_ACTUAL_CAPTION_CONFIRMED'});
  validateLedger(ledger, binding.account);
  const temporary = join(dirname(path), '.' + randomUUID() + '.json');
  writeExclusive(temporary, ledger);
  assert.equal(sha(readFileSync(path)), expectedSha256, 'LEDGER_CHANGED_RECONCILE_FIRST');
  renameSync(temporary, path);
  return row;
}
