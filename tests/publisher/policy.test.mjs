import test from 'node:test';
import assert from 'node:assert/strict';
import {sha, visibleFingerprint, historyAdmission, pairedCounterpart, releaseAuthority} from '../../scripts/publisher/policy.mjs';
import {approvedCaption} from '../../scripts/publisher/captions.mjs';
import {validateLedger} from '../../scripts/publisher/ledger.mjs';

import {fixture} from './fixture.mjs';

test('caption pool preserves exact text, UTF-8 bytes, platform coverage and unique indices', () => {
  const {b, caption} = fixture();
  const pool = {status: 'APPROVED_CAPTION_POOL', captions: [{index: 1, caption, platforms: b.platforms}]};
  assert.equal(approvedCaption(pool, b), caption);
  for (const changes of [{caption: caption + ' '}, {platforms: ['TikTok']}, {index: 2}]) {
    assert.throws(() => approvedCaption({...pool, captions: [{...pool.captions[0], ...changes}]}, b));
  }
  assert.throws(() => approvedCaption({...pool, captions: [...pool.captions, ...pool.captions]}, b), /DUPLICATE/);
});

test('current release authority and all-ever-posted history include removed records', () => {
  const {b, ledger, release, options} = fixture();
  assert(releaseAuthority(release, b));
  assert(historyAdmission(b, ledger, 'TikTok', options).includesRemoved);
  for (const changes of [{status: 'HISTORICAL'}, {platforms: {Instagram: 2, TikTok: 1}}, {slotUtc: '2098-01-01T12:00:00Z'}]) {
    assert.throws(() => releaseAuthority({...release, ...changes}, b));
  }
  const removed = structuredClone(ledger);
  Object.assign(removed.records[0], {status: 'REMOVED_ON_OWNER_WORD', visibleOpener: b.opener, visibleOpenerSha256: b.opener.sha256});
  assert.throws(() => historyAdmission(b, removed, 'TikTok', options), /OPENER_REPEAT/);
  assert.throws(() => historyAdmission(b, {...ledger, records: ledger.records.slice(1)}, 'TikTok', options), /HISTORY_ROW_REMOVED/);
});

test('only the other platform from the exact current release may share its fingerprint', () => {
  const {b, ledger, release} = fixture();
  const row = {...ledger.records[1], id: 'synthetic-current-pair', url: 'https://example.invalid/current', status: 'POSTED_ACTUAL_CAPTION_CONFIRMED'};
  assert(pairedCounterpart(b, row, 'TikTok', release));
  for (const changes of [{platform: 'TikTok'}, {releaseReceipt: 'historical.json'}, {slotUtc: '2098-01-01T12:00:00Z'},
    {captionSha256: sha('wrong')}, {status: 'REMOVED_ON_OWNER_WORD'}]) assert(!pairedCounterpart(b, {...row, ...changes}, 'TikTok', release));
  assert(!pairedCounterpart(b, row, 'TikTok', null));
});

test('runtime predecessor proof is due, current-ledger-bound and from an owned session', () => {
  const {b, ledger, proof, options} = fixture();
  for (const changes of [{ledgerSha256: sha('stale')}, {actualPredecessorId: 'planned'}, {session: {}},
    {observedAtUtc: '2098-01-01T12:00:00Z'}, {evidence: []}, {account: 'another_creator'}]) {
    assert.throws(() => historyAdmission(b, ledger, 'TikTok', {...options, actualPredecessor: {...proof, ...changes}}));
  }
  assert.throws(() => historyAdmission(b, ledger, 'TikTok', {...options, actualPredecessor: null}), /PREDECESSOR_PROOF_REQUIRED/);
});

test('same-mechanism predecessor needs a freshly bound substantive reassessment', () => {
  const {b, ledger, proof, options} = fixture();
  b.familyCases[0].mechanism = b.opener.mechanism;
  assert.throws(() => historyAdmission(b, ledger, 'TikTok', options), /REASSESS_OR_SKIP/);
  proof.familyReassessment = {status: 'FRESH_ACTUAL_DIFFERENT_FAMILIES_CONFIRMED', differentFamilies: true,
    predecessorId: proof.actualPredecessorId, predecessorFingerprint: b.familyCases[0].visibleFingerprintSha256,
    predecessorFamily: 'previous-family', candidateDeal: b.deal, candidateFingerprint: b.opener.sha256,
    candidateFamily: b.opener.family, ledgerSha256: options.ledgerSha256, rationale: 'Synthetic specific tactical comparison',
    evidence: [{path: 'synthetic-review.json', sha256: sha('review')}]};
  assert(historyAdmission(b, ledger, 'TikTok', options).family.sameMechanism);
  proof.familyReassessment.predecessorId = 'stale';
  assert.throws(() => historyAdmission(b, ledger, 'TikTok', options), /REASSESSMENT_BINDING/);
});

test('ledger rejects foreign account, duplicate artifacts and unacknowledged success', () => {
  const {b, ledger} = fixture();
  assert(validateLedger(ledger, b.account));
  assert.throws(() => validateLedger(ledger, 'another_creator'), /ACCOUNT/);
  assert.throws(() => validateLedger({...ledger, records: [...ledger.records, ledger.records[0]]}, b.account), /DUPLICATE/);
  const bad = structuredClone(ledger); bad.records[0].applicationAcknowledged = false;
  assert.throws(() => validateLedger(bad, b.account), /ACK_AND_CAPTION/);
});

test('existing removed and confirmed history retains original time evidence and fields', () => {
  const {b, ledger} = fixture();
  const historical = ledger.records[0];
  historical.postedAtUtcRange = [historical.postedAtUtc, historical.postedAtUtc];
  delete historical.postedAtUtc;
  delete historical.applicationAcknowledged;
  delete historical.actualCaptionSha256;
  assert(validateLedger(ledger, b.account));
  historical.status = 'REMOVED_ON_OWNER_WORD';
  assert(validateLedger(ledger, b.account));
});
