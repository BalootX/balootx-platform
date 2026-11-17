import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {spawn} from 'node:child_process';
import {localConfig, assertActivated, guard, validatePacket, load} from './runtime.mjs';
import {sha} from './policy.mjs';
import {createClient} from './client.mjs';
import {confirmPost} from './ledger.mjs';
import {readPost, postCaptionDom, tiktokPostDom} from './post-readback.mjs';

export function schedulePlan(runtime) {
  assertActivated(runtime);
  const b = runtime.binding;
  return b.platforms.map(platform => ({taskId: b.scheduledTaskId + '-' + platform.toLowerCase(),
    dueUtc: b.slotUtc, maxAttempts: 1, command: [process.execPath, fileURLToPath(import.meta.url),
      '--config', runtime.configPath, '--platform', platform, '--action', 'dispatch'],
    requiresExclusiveLocalClaim: true}));
}

export async function operatorAdapter(runtime, name, argument) {
  const path = runtime.config[name + 'Path'];
  assert(path && runtime.config[name + 'Sha256'] === sha(readFileSync(path)), 'PINNED_LOCAL_ADAPTER_REQUIRED');
  const adapter = await import(pathToFileURL(path).href);
  assert(typeof adapter.default === 'function', 'LOCAL_ADAPTER_FUNCTION_REQUIRED');
  return adapter.default(argument);
}

export async function runAction(client, action, {url} = {}) {
  const {runtime, platform} = client, b = runtime.binding, c = runtime.config;
  guard(runtime);
  const key = platform.toLowerCase();
  const intentPath = resolve(b.evidenceRoot, key + '-share-intent.json');
  if (['attach', 'fill', 'share'].includes(action)) assert(!existsSync(intentPath), 'PRIOR_INTENT_RECONCILE_FIRST_NO_RETRY');
  if (action === 'observe') return client.inspect();
  if (action === 'init') {
    validatePacket(runtime, platform, {admission: false});
    await client.op('navigate', c.profileCheckUrl);
    await client.waitProof('local-profile', () => client.dom("document.querySelector('#profile_path')?.innerText||null"),
      path => assert(typeof path === 'string' && path.endsWith(c.profileSuffix), 'LOCAL_PROFILE_REQUIRED'));
    await client.focus();
    return client.save(key + '-session.json', {binding: await client.op('binding')});
  }
  if (action === 'account') {
    validatePacket(runtime, platform, {admission: false});
    await client.op('navigate', c.profileUrls[platform]);
    const result = await client.waitProof('own-account', client.inspect, state => {
      assert(!state.blocker, 'OWNER_ONLY_PROMPT');
      assert.equal(state.url, c.profileUrls[platform], 'OWN_PROFILE_URL');
      assert(state.text.split('\n').includes(b.account), 'ACTUAL_HANDLE_REQUIRED');
      assert(state.controls.some(control => control.text === 'Edit profile' && control.enabled), 'EDIT_PROFILE_REQUIRED');
    });
    return client.save(key + '-account.json', {binding: await client.op('binding'),
      baselinePosts: result.state.links.filter(link => /\/(?:video|reel|p)\//.test(link.url))});
  }
  if (action === 'attach') {
    const result = await client.waitProof('active-upload-file', client.inspect, state => {
      assert(!state.blocker, 'OWNER_ONLY_PROMPT');
      assert.equal(state.url, c.uploadUrls[platform], 'ACTIVE_UPLOAD_URL');
      assert.equal(state.files.filter(file => file.accept.includes('video/')).length, 1, 'UNIQUE_UPLOAD_FILE_INPUT');
    });
    const file = result.state.files.find(file => file.accept.includes('video/'));
    return client.save(key + '-upload.json', await client.attach(file.selector));
  }
  if (action === 'fill') {
    const identity = validatePacket(runtime, platform), state = await client.composer();
    await client.focus();
    await client.click(state.editor.selector);
    const input = await client.fill(state.editor.selector, identity.caption);
    return client.save(key + '-input.json', {input, state: await client.composer(true), binding: await client.op('binding')});
  }
  if (action === 'share') {
    const identity = validatePacket(runtime, platform), session = await client.op('binding');
    for (const suffix of ['session', 'account', 'upload', 'input']) {
      const proof = load(resolve(b.evidenceRoot, key + '-' + suffix + '.json'));
      assert(proof.binding.targetId === session.targetId && proof.binding.claimSha256 === session.claimSha256, 'OWN_SESSION_CHANGED');
      if (suffix === 'upload') assert(proof.sha256 === identity.sha256 && proof.bytes === identity.bytes, 'UPLOAD_BINDING');
    }
    const first = await client.composer(true);
    const intent = {platform, session, identity, reservedAtUtc: new Date().toISOString(), status: 'INTENT_RESERVED'};
    client.save(key + '-share-intent.json', intent);
    const fresh = await client.composer(true);
    for (const part of ['scope', 'editor', 'counter', 'submit']) assert.equal(fresh[part].selector, first[part].selector, 'COMPOSER_CHANGED');
    const pointer = await client.click(fresh.submit.selector);
    client.save(key + '-native-ack.json', {pointer, claim: 'PROTOCOL_ONLY_NOT_APPLICATION_SUCCESS'});
    const acknowledgment = await client.waitProof('application-acknowledgment', client.inspect, state => {
      assert(!state.blocker, 'OWNER_ONLY_PROMPT');
      assert(c.applicationAckTexts[platform].some(text => text && state.text.includes(text)), 'APPLICATION_ACK_REQUIRED');
    });
    return client.save(key + '-application-ack.json', {binding: session, observedAtUtc: new Date().toISOString(), acknowledgment});
  }
  if (action === 'readback') {
    const intent = load(intentPath), ack = load(resolve(b.evidenceRoot, key + '-application-ack.json'));
    const session = await client.op('binding');
    assert(ack.binding.targetId === session.targetId && ack.binding.claimSha256 === session.claimSha256 &&
      intent.session.targetId === session.targetId && intent.session.claimSha256 === session.claimSha256, 'READBACK_SESSION_CHANGED');
    const parsed = new URL(url);
    assert(parsed.origin !== 'null' && parsed.origin === new URL(c.profileUrls[platform]).origin &&
      !parsed.search && !parsed.hash && !parsed.username && !parsed.password, 'EXACT_PUBLIC_POST_URL_REQUIRED');
    const match = platform === 'TikTok' ? parsed.pathname.match(/^\/@([^/]+)\/video\/([0-9]+)$/) :
      parsed.pathname.match(/^\/(?:[A-Za-z0-9._]+\/)?(?:reel|p)\/([A-Za-z0-9_-]+)\/$/);
    assert(match && (platform !== 'TikTok' || match[1] === b.account), 'ACTUAL_PERMALINK_REQUIRED');
    const account = load(resolve(b.evidenceRoot, key + '-account.json'));
    assert(!account.baselinePosts.some(post => post.url === url), 'NEW_POST_URL_REQUIRED');
    const deadline = Date.parse(intent.reservedAtUtc) + 120000;
    assert(Date.now() < deadline, 'CAPTION_120S_DEADLINE');
    await client.op('navigate', url);
    const post = await readPost(client, {url, caption: intent.identity.caption, deadline,
      postDom: platform === 'Instagram' ? postCaptionDom : tiktokPostDom});
    client.save(key + '-actual-caption.json', post);
    assert(post.exact && post.within120s, 'CAPTION_NOT_CONFIRMED_RECONCILE_NO_RETRY');
    return confirmPost(b.ledgerPath, intent.identity.ledgerSha256, b, platform, {...post,
      id: match.at(-1), postedAtUtc: new Date().toISOString(), applicationAcknowledged: true});
  }
  throw Error('UNKNOWN_ACTION');
}

async function main() {
  const {values} = parseArgs({options: {config: {type: 'string'}, platform: {type: 'string'},
    action: {type: 'string'}, url: {type: 'string'}}});
  const runtime = localConfig(values.config);
  if (values.action === 'schedule') {
    console.log(JSON.stringify(schedulePlan(runtime), null, 2));
    return;
  }
  if (values.action === 'register') {
    return operatorAdapter(runtime, 'schedulerAdapter', schedulePlan(runtime));
  }
  guard(runtime);
  if (values.action === 'helper') {
    const child = spawn(process.execPath, ['--loader', fileURLToPath(new URL('./native-loader.mjs', import.meta.url)),
      runtime.config.helperPath, runtime.config.startUrl], {stdio: 'inherit', windowsHide: true,
      env: {...process.env, PUBLISHER_CONFIG: values.config}});
    child.on('error', error => { console.error(error.message); process.exitCode = 1; });
    child.on('exit', code => { process.exitCode = code ?? 1; });
    return;
  }
  const client = createClient(runtime, values.platform);
  if (values.action === 'dispatch') return operatorAdapter(runtime, 'operatorAdapter', {client, runAction});
  const result = await runAction(client, values.action, {url: values.url});
  console.log(JSON.stringify({status: values.action === 'readback' ? result.status : 'STEP_COMPLETED', action: values.action}));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
