import test from 'node:test';
import assert from 'node:assert/strict';
import {sha, visibleFingerprint, historyAdmission, pairedCounterpart, releaseAuthority} from '../../scripts/publisher/policy.mjs';
import {approvedCaption} from '../../scripts/publisher/captions.mjs';
import {validateLedger} from '../../scripts/publisher/ledger.mjs';

export function fixture() {
  const caption = 'SYNTHETIC caption فقط';
  const opener = {hand: ['SA', 'D9'], table: ['HK'], family: 'candidate-family', mechanism: 'capture'};
  opener.sha256 = visibleFingerprint(opener);
  const previous = {hand: ['C8', 'H9'], table: ['SK']};
  const b = {account: 'synthetic_creator', deal: 'synthetic-candidate', dealNumber: 'candidate',
    mediaSha256: sha('synthetic-media'), slotUtc: '2099-01-01T12:00:00Z', platforms: ['Instagram', 'TikTok'],
    releasePath: 'synthetic-release.json', releaseSha256: sha('synthetic-release'), captionSha256: sha(caption),
    captionCharacters: caption.length, captionUtf8Bytes: Buffer.byteLength(caption), captionPoolIndex: 1,
    contentRequestSha256: sha('synthetic-request'), contentHandoffSha256: sha('synthetic-handoff'),
    scheduledTaskId: 'synthetic-task', knownMinimumIds: ['synthetic-previous-instagram', 'synthetic-previous-tiktok'], opener,
    familyCases: [{deal: 'synthetic-previous', visibleFingerprintSha256: visibleFingerprint(previous), family: 'previous-family', mechanism: 'discard'}]};
  const release = {status: 'RELEASED_CONDITIONAL_CURRENT_SLOT', deal: b.dealNumber, slotUtc: b.slotUtc,
    candidate: {deal: b.deal, artifactSha256: b.mediaSha256, slotUtc: b.slotUtc, captionSha256: b.captionSha256, visibleFingerprintSha256: opener.sha256},
    platforms: {Instagram: 1, TikTok: 1}, requestSha256: b.contentRequestSha256, handoffSha256: b.contentHandoffSha256,
    caption: {text: caption, sha256: b.captionSha256, characters: b.captionCharacters, utf8Bytes: b.captionUtf8Bytes}};
  const ledger = {schemaVersion: 1, account: b.account, records: b.platforms.flatMap(platform => [{platform,
    id: 'synthetic-previous-' + platform.toLowerCase(), url: 'https://example.invalid/' + platform + '/previous',
    deal: 'synthetic-previous', slotUtc: '2098-12-31T12:00:00Z', artifactSha256: sha('previous-media'), captionSha256: sha('previous-caption'),
    scenarioFamily: 'previous-family', visibleOpener: previous, visibleOpenerSha256: visibleFingerprint(previous),
    status: 'POSTED_ACTUAL_CAPTION_CONFIRMED', postedAtUtc: '2098-12-31T12:01:00Z', applicationAcknowledged: true, actualCaptionSha256: sha('previous-caption')},
  {platform, deal: b.deal, slotUtc: b.slotUtc, artifactSha256: b.mediaSha256, captionSha256: b.captionSha256,
    scenarioFamily: opener.family, visibleOpener: opener, visibleOpenerSha256: opener.sha256,
    status: 'RELEASED_SLOT_BOUND_CAPTION_PATH_CONFIRMED', maxPosts: 1, scheduledTaskId: b.scheduledTaskId, releaseReceipt: b.releasePath}])};
  const ledgerSha256 = sha(JSON.stringify(ledger));
  const nowMs = Date.parse(b.slotUtc) + 1000;
  const proof = {status: 'FRESH_ACTUAL_PREDECESSOR_CONFIRMED', platform: 'TikTok', account: b.account, deal: b.deal,
    slotUtc: b.slotUtc, releaseSha256: b.releaseSha256, ledgerSha256, actualPredecessorId: b.knownMinimumIds[1],
    actualPredecessorUrl: ledger.records[2].url, observedAtUtc: new Date(nowMs).toISOString(),
    session: {targetId: 'synthetic-target', claimSha256: sha('synthetic-claim')},
    evidence: [{kind: 'COMPLETE_UNFILTERED_STUDIO', path: 'synthetic-evidence.json', sha256: sha('synthetic-evidence')}]};
  return {b, caption, ledger, release, proof, options: {authority: release, ledgerSha256, actualPredecessor: proof, nowMs}};
}

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
