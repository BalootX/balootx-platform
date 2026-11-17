import assert from 'node:assert/strict';
import {readFileSync, realpathSync} from 'node:fs';
import {isAbsolute, relative, resolve, sep} from 'node:path';
import {sha, releaseAuthority, historyAdmission} from './policy.mjs';
import {approvedCaption} from './captions.mjs';
import {currentRow} from './ledger.mjs';

export const load = path => JSON.parse(readFileSync(path, 'utf8'));
export function ownedPath(root, path) {
  const rel = relative(realpathSync(root), realpathSync(path));
  assert(rel && rel !== '..' && !rel.startsWith('..' + sep) && !isAbsolute(rel), 'OWNED_LOCAL_PATH_REQUIRED');
  return path;
}

export function localConfig(path, env = process.env) {
  assert(!env.CI && !env.GITHUB_ACTIONS, 'LIVE_BROWSER_FORBIDDEN_IN_CI');
  assert(path && isAbsolute(path), 'EXPLICIT_LOCAL_CONFIG_REQUIRED');
  const config = load(path);
  assert(config.localOnly === true && config.bindingPath && config.helperPath && config.lockModulePath, 'LOCAL_CONFIG_REQUIRED');
  for (const key of ['bindingPath', 'helperPath', 'lockModulePath', 'lockPath']) assert(isAbsolute(config[key]), 'ABSOLUTE_LOCAL_PATH_REQUIRED');
  const url = new URL(config.controlUrl);
  assert(url.protocol === 'http:' && url.hostname === '127.0.0.1' && url.port && url.pathname === '/' &&
    !url.username && !url.password && !url.search && !url.hash, 'LOOPBACK_CONTROL_REQUIRED');
  const binding = load(config.bindingPath);
  assert(Array.isArray(binding.platforms) && binding.platforms.length > 0 &&
    binding.platforms.every(platform => ['Instagram', 'TikTok'].includes(platform)), 'LOCAL_PLATFORMS_REQUIRED');
  assert(Array.isArray(config.protectedTargetIds) && Array.isArray(config.allowedOrigins), 'LOCAL_TARGET_AND_ORIGIN_SCOPE_REQUIRED');
  assert(config.allowedOrigins.length > 0 && config.allowedOrigins.every(origin => {
    const parsed = new URL(origin);
    return parsed.origin !== 'null' && parsed.origin === origin && parsed.protocol === 'https:';
  }), 'EXACT_HTTPS_ORIGINS_REQUIRED');
  assert(typeof config.profileSuffix === 'string' && config.profileSuffix.length > 0 &&
    /^chrome:\/\/version\/?$/.test(config.profileCheckUrl), 'LOCAL_PROFILE_CONFIG_REQUIRED');
  for (const platform of binding.platforms) {
    for (const group of ['profileUrls', 'uploadUrls']) {
      const parsed = new URL(config[group]?.[platform]);
      assert(config.allowedOrigins.includes(parsed.origin) && !parsed.username && !parsed.password &&
        !parsed.search && !parsed.hash, 'EXACT_PLATFORM_URL_REQUIRED');
    }
    assert(Array.isArray(config.applicationAckTexts?.[platform]) && config.applicationAckTexts[platform].length > 0 &&
      config.applicationAckTexts[platform].every(text => typeof text === 'string' && text.trim()), 'APPLICATION_ACK_CONFIG_REQUIRED');
  }
  assert(config.startUrl === config.profileCheckUrl || config.allowedOrigins.includes(new URL(config.startUrl).origin), 'LOCAL_START_URL_REQUIRED');
  return {config, configPath: path, binding};
}

export function assertDue(binding, nowMs = Date.now()) {
  const due = Date.parse(binding.slotUtc);
  assert(Number.isFinite(due) && nowMs >= due, 'BEFORE_RELEASED_SLOT');
  assert(nowMs < Date.parse(binding.slotUtc.slice(0, 10) + 'T00:00:00Z') + 86400000, 'RELEASE_DATE_EXPIRED_NO_REPLAY');
}

export function assertActivated(runtime) {
  const {binding: b, config: c, configPath} = runtime;
  const pin = load(b.executionPinPath);
  assert(pin.status === 'REVIEWED_EXECUTION_FREEZE' && pin.liveEnabled === true, 'EXECUTION_FREEZE_NOT_BOUND');
  assert(pin.bindingSha256 === sha(readFileSync(c.bindingPath)) && pin.configSha256 === sha(readFileSync(configPath)) &&
    pin.scheduledTaskId === b.scheduledTaskId && pin.slotUtc === b.slotUtc && pin.releaseSha256 === b.releaseSha256, 'EXECUTION_PIN_BINDING');
  assert.deepEqual(pin.platforms, b.platforms, 'EXECUTION_PLATFORMS');
  for (const [name, digest] of Object.entries(pin.sourceHashes)) {
    assert.equal(sha(readFileSync(new URL(name, import.meta.url))), digest, 'FROZEN_SOURCE_CHANGED');
  }
  for (const name of ['runtime.mjs', 'policy.mjs', 'captions.mjs', 'ledger.mjs', 'dom.mjs', 'readiness.mjs', 'client.mjs', 'operator.mjs', 'native-loader.mjs', 'post-readback.mjs']) {
    assert(Object.hasOwn(pin.sourceHashes, './' + name), 'COMPLETE_SOURCE_FREEZE_REQUIRED');
  }
  assert.equal(sha(readFileSync(c.helperPath)), c.helperSha256, 'LOCAL_HELPER_CHANGED');
  assert.equal(sha(readFileSync(c.lockModulePath)), c.lockModuleSha256, 'LOCAL_LOCK_MODULE_CHANGED');
  const releaseBytes = readFileSync(b.releasePath);
  assert.equal(sha(releaseBytes), b.releaseSha256, 'CURRENT_RELEASE_BYTES_CHANGED');
  releaseAuthority(JSON.parse(releaseBytes), b);
}

export function assertClaim(runtime, env = process.env) {
  const {config: c} = runtime;
  assert(!env.CI && !env.GITHUB_ACTIONS, 'LIVE_BROWSER_FORBIDDEN_IN_CI');
  assert(env.CHROME_LOCK_PATH && resolve(env.CHROME_LOCK_PATH) === resolve(c.lockPath), 'LOCAL_LOCK_REQUIRED');
  const bytes = readFileSync(c.lockPath), claim = JSON.parse(bytes);
  assert(sha(bytes) === env.CHROME_LOCK_CLAIM_SHA256 && String(claim.pid) === env.CHROME_LOCK_HOLDER_PID &&
    claim.schema === 3 && claim.os_lock === true && claim.owner === c.lockOwner, 'OWN_EXCLUSIVE_CLAIM_REQUIRED');
  assert(Number.isInteger(claim.pid) && claim.pid > 0, 'LOCK_HOLDER_REQUIRED');
  process.kill(claim.pid, 0);
  return {claimSha256: sha(bytes), holderPid: claim.pid, lockPath: c.lockPath};
}

export function guard(runtime) {
  assertDue(runtime.binding);
  assertActivated(runtime);
  return assertClaim(runtime);
}

export function validatePacket(runtime, platform, {admission = true} = {}) {
  const claim = guard(runtime);
  const b = runtime.binding;
  assert(b.platforms.includes(platform), 'EXPLICIT_PLATFORM_REQUIRED');
  for (const [path, digest] of Object.entries(b.inputPins)) assert.equal(sha(readFileSync(path)), digest, 'STATIC_PACKET_BINDING');
  const release = load(b.releasePath);
  releaseAuthority(release, b);
  const caption = approvedCaption(load(b.captionPath), b);
  assert.equal(release.caption.text, caption, 'CURRENT_RELEASE_EXACT_CAPTION');
  const media = readFileSync(b.mediaPath), approval = load(b.approvalPath), manifest = load(b.manifestPath);
  assert(media.length === b.mediaBytes && sha(media) === b.mediaSha256 &&
    approval.files[b.mediaName] === b.mediaSha256 && manifest.files.video.sha256 === b.mediaSha256 &&
    manifest.files.video.bytes === b.mediaBytes, 'EXACT_MEDIA_APPROVAL_MANIFEST');
  assert.equal(sha(approval.caption), b.originalCaptionSha256, 'ORIGINAL_RENDERER_CAPTION_PRESERVED');
  const request = load(b.contentRequestPath);
  assert(request.fullyQualifiedAdmission && request.releaseRequestReadyForPortfolio &&
    request.sourceAdmission.status === 'PASS_RETAINED_SOURCE_CONTENT_AND_CUSTODY', 'QUALIFIED_RETAINED_CONTENT_REQUIRED');
  assert(request.candidate.deal === b.deal && request.candidate.artifactSha256 === b.mediaSha256 &&
    request.candidate.slotUtc === b.slotUtc && request.candidate.visibleFingerprintSha256 === b.opener.sha256, 'CONTENT_CANDIDATE');
  assert.equal(sha(readFileSync(b.contentRequestPath)), b.contentRequestSha256, 'CONTENT_REQUEST_BINDING');
  assert.equal(sha(readFileSync(b.contentHandoffPath)), b.contentHandoffSha256, 'CONTENT_HANDOFF_BINDING');
  const receiving = load(b.receivingPath);
  assert.equal(receiving.status, 'PASS_ROOT_RECEIVING_CORRECTED_OFFLINE_SUCCESSORS', 'QUALIFIED_CONTENT_RECEIVING');
  assert(receiving.results.some(row => row.deal === b.dealNumber && row.requestSha256 === b.contentRequestSha256 &&
    row.handoffSha256 === b.contentHandoffSha256), 'RECEIVING_CONTENT_BINDING');
  const ledgerBytes = readFileSync(b.ledgerPath), ledgerSha256 = sha(ledgerBytes), ledger = JSON.parse(ledgerBytes);
  currentRow(b, ledger, platform);
  let history;
  if (admission) {
    const proof = load(ownedPath(b.evidenceRoot, b.admissionProofPaths[platform]));
    const account = load(ownedPath(b.evidenceRoot, resolve(b.evidenceRoot, platform.toLowerCase() + '-account.json')));
    assert(proof.session?.targetId === account.binding.targetId &&
      proof.session?.claimSha256 === account.binding.claimSha256 &&
      proof.session?.claimSha256 === claim.claimSha256, 'ACTUAL_PREDECESSOR_ACCOUNT_SESSION');
    history = historyAdmission(b, ledger, platform, {authority: release, ledgerSha256, actualPredecessor: proof});
    for (const pin of [...proof.evidence, ...(proof.familyReassessment?.evidence || [])]) {
      assert.equal(sha(readFileSync(ownedPath(b.evidenceRoot, pin.path))), pin.sha256, 'FRESH_ADMISSION_EVIDENCE_CHANGED');
    }
  }
  return {caption, ledgerSha256, history, sha256: b.mediaSha256, bytes: b.mediaBytes};
}
