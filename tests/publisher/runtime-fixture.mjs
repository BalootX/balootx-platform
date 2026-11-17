import {mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fixture} from './fixture.mjs';
import {sha} from '../../scripts/publisher/policy.mjs';

export function runtimeFixture() {
  const {b, caption, release, ledger, proof} = fixture();
  const root = mkdtempSync(join(tmpdir(), 'publisher-unit-'));
  const put = (name, value) => {
    const path = join(root, name);
    writeFileSync(path, typeof value === 'string' ? value : JSON.stringify(value));
    return path;
  };
  b.mediaPath = put('synthetic-media.bin', 'synthetic-media');
  b.mediaBytes = readFileSync(b.mediaPath).length;
  b.mediaName = 'synthetic-media.bin';
  b.originalCaptionSha256 = sha('synthetic-renderer-caption');
  b.approvalPath = put('approval.json', {caption: 'synthetic-renderer-caption', files: {[b.mediaName]: b.mediaSha256}});
  b.manifestPath = put('manifest.json', {files: {video: {sha256: b.mediaSha256, bytes: b.mediaBytes}}});
  b.captionPath = put('caption-pool.json', {status: 'APPROVED_CAPTION_POOL', captions: [{index: 1, caption, platforms: b.platforms}]});
  b.contentRequestPath = put('request.json', {fullyQualifiedAdmission: true, releaseRequestReadyForPortfolio: true,
    sourceAdmission: {status: 'PASS_RETAINED_SOURCE_CONTENT_AND_CUSTODY'}, candidate: release.candidate});
  b.contentRequestSha256 = sha(readFileSync(b.contentRequestPath));
  b.contentHandoffPath = put('handoff.json', {});
  b.contentHandoffSha256 = sha(readFileSync(b.contentHandoffPath));
  b.receivingPath = put('receiving.json', {status: 'PASS_ROOT_RECEIVING_CORRECTED_OFFLINE_SUCCESSORS',
    results: [{deal: b.dealNumber, requestSha256: b.contentRequestSha256, handoffSha256: b.contentHandoffSha256}]});
  release.requestSha256 = b.contentRequestSha256;
  release.handoffSha256 = b.contentHandoffSha256;
  b.releasePath = put('release.json', release);
  b.releaseSha256 = sha(readFileSync(b.releasePath));
  for (const row of ledger.records.filter(row => !row.id)) row.releaseReceipt = b.releasePath;
  b.ledgerPath = put('ledger.json', ledger);
  b.evidenceRoot = join(root, 'evidence'); mkdirSync(b.evidenceRoot);
  proof.ledgerSha256 = sha(readFileSync(b.ledgerPath));
  proof.releaseSha256 = b.releaseSha256;
  const evidencePath = join(b.evidenceRoot, 'synthetic-evidence.json'); writeFileSync(evidencePath, '{}');
  proof.evidence[0] = {...proof.evidence[0], path: evidencePath, sha256: sha(readFileSync(evidencePath))};
  b.admissionProofPaths = {TikTok: join(b.evidenceRoot, 'synthetic-admission.json')};
  writeFileSync(b.admissionProofPaths.TikTok, JSON.stringify(proof));
  writeFileSync(join(b.evidenceRoot, 'tiktok-account.json'), JSON.stringify({binding: proof.session}));
  b.executionPinPath = join(root, 'execution.json');
  b.inputPins = Object.fromEntries([b.approvalPath, b.manifestPath, b.captionPath, b.contentRequestPath, b.contentHandoffPath, b.receivingPath]
    .map(path => [path, sha(readFileSync(path))]));
  const config = {localOnly: true, bindingPath: put('binding.json', b), helperPath: put('helper.mjs', '// Synthetic helper fixture only'),
    lockModulePath: put('lock.py', '# Synthetic lock fixture only'), lockPath: join(root, 'claim.json'), lockOwner: 'synthetic-owner',
    controlUrl: 'http://127.0.0.1:12345/', protectedTargetIds: ['synthetic-protected-target'], allowedOrigins: ['https://example.invalid'],
    profileCheckUrl: 'chrome://version/', profileSuffix: 'SyntheticProfile', startUrl: 'https://example.invalid/',
    profileUrls: {TikTok: 'https://example.invalid/@synthetic_creator', Instagram: 'https://example.invalid/synthetic_creator/'},
    uploadUrls: {TikTok: 'https://example.invalid/tiktokstudio/upload', Instagram: 'https://example.invalid/'},
    applicationAckTexts: {TikTok: ['Synthetic application acknowledged'], Instagram: ['Synthetic application acknowledged']}};
  config.operatorAdapterPath = put('operator-adapter.mjs', 'export default async function syntheticAdapter() {}');
  config.operatorAdapterSha256 = sha(readFileSync(config.operatorAdapterPath));
  config.helperSha256 = sha(readFileSync(config.helperPath)); config.lockModuleSha256 = sha(readFileSync(config.lockModulePath));
  const configPath = put('config.json', config);
  const sourceHashes = Object.fromEntries(['runtime.mjs', 'policy.mjs', 'captions.mjs', 'ledger.mjs', 'dom.mjs', 'readiness.mjs',
    'client.mjs', 'operator.mjs', 'native-loader.mjs', 'post-readback.mjs'].map(name =>
    ['./' + name, sha(readFileSync(new URL('../../scripts/publisher/' + name, import.meta.url)))]));
  const pin = {status: 'REVIEWED_EXECUTION_FREEZE', liveEnabled: true, bindingSha256: sha(readFileSync(config.bindingPath)),
    configSha256: sha(readFileSync(configPath)), scheduledTaskId: b.scheduledTaskId, slotUtc: b.slotUtc,
    releaseSha256: b.releaseSha256, platforms: b.platforms, sourceHashes};
  put('execution.json', pin);
  const claim = {owner: config.lockOwner, claim: 'synthetic-claim', pid: process.pid, started_utc: new Date().toISOString(), schema: 3, os_lock: true};
  put('claim.json', claim);
  const env = {CHROME_LOCK_PATH: config.lockPath, CHROME_LOCK_CLAIM_SHA256: sha(readFileSync(config.lockPath)), CHROME_LOCK_HOLDER_PID: String(process.pid)};
  return {runtime: {binding: b, config, configPath}, b, config, configPath, pin, claim, env, put, root, caption,
    dispose: () => rmSync(root, {recursive: true, force: true})};
}
