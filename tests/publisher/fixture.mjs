import {sha, visibleFingerprint} from '../../scripts/publisher/policy.mjs';

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
