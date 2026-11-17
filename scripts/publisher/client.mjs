import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {sha} from './policy.mjs';
import {guard, validatePacket} from './runtime.mjs';
import {pageState, pointerTarget, composerDom, qualify} from './dom.mjs';
import {waitThenJudge} from './readiness.mjs';
import {writeExclusive} from './ledger.mjs';

export function createClient(runtime, platform) {
  const {binding: b, config: c} = runtime;
  assert(b.platforms.includes(platform), 'EXPLICIT_PLATFORM_REQUIRED');
  const save = (name, value) => writeExclusive(resolve(b.evidenceRoot, name), value);
  async function op(op, arg) {
    const claim = guard(runtime);
    const response = await fetch(c.controlUrl, {method: 'POST',
      headers: {'X-Publisher-Claim': claim.claimSha256, 'Content-Type': 'application/json'},
      body: JSON.stringify({op, arg}), signal: AbortSignal.timeout(30000)});
    assert(response.ok, 'LOCAL_HELPER_HTTP_FAILURE');
    const text = await response.text();
    if (text.startsWith('error:')) throw Error(text);
    try { return JSON.parse(text); } catch { return text; }
  }
  const dom = expression => op('eval', expression);
  const native = (method, params = {}) => op('native', {method, params});
  const inspect = () => dom('(' + pageState.toString() + ')()');
  const waitProof = (name, read, judge, options = {}) => waitThenJudge(name, read, judge, {
    ...options, record: proof => save('readiness-' + name + '-' + Date.now() + '-' + Math.random().toString(16).slice(2) + '.json', proof)});
  async function focus() {
    await op('front');
    await native('Page.bringToFront');
    await native('Emulation.setFocusEmulationEnabled', {enabled: true});
  }
  async function click(selector) {
    let pointer = await dom('(' + pointerTarget.toString() + ')(' + JSON.stringify(selector) + ')');
    save('pointer-' + Date.now() + '-' + Math.random().toString(16).slice(2) + '.json', pointer);
    pointer = await dom('(' + pointerTarget.toString() + ')(' + JSON.stringify(selector) + ')');
    try { await native('Input.dispatchMouseEvent', {type: 'mousePressed', x: pointer.x, y: pointer.y, button: 'left', buttons: 1, clickCount: 1}); }
    finally { await native('Input.dispatchMouseEvent', {type: 'mouseReleased', x: pointer.x, y: pointer.y, button: 'left', buttons: 0, clickCount: 1}); }
    return pointer;
  }
  async function composer(exact = false) {
    const identity = validatePacket(runtime, platform);
    const result = await waitProof('active-composer', async () => {
      const state = await inspect();
      assert(!state.blocker, 'OWNER_ONLY_PROMPT');
      return {...await dom('(' + composerDom.toString() + ')(' + JSON.stringify(platform) + ')'), blocker: state.blocker};
    }, state => { if (exact) qualify(state, platform, identity.caption, b); });
    return result.state;
  }
  async function fill(selector, caption) {
    const focused = await dom(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('EDITOR_MISSING');e.focus();return document.activeElement===e;})()`);
    assert(focused, 'EDITOR_FOCUS_REQUIRED');
    await native('Input.dispatchKeyEvent', {type: 'rawKeyDown', key: 'a', code: 'KeyA', windowsVirtualKeyCode: 65, modifiers: 2});
    await native('Input.dispatchKeyEvent', {type: 'keyUp', key: 'a', code: 'KeyA', windowsVirtualKeyCode: 65, modifiers: 0});
    await native('Input.insertText', {text: caption});
    return {method: 'Input.insertText', captionSha256: sha(caption)};
  }
  async function attach(selector) {
    const proof = validatePacket(runtime, platform);
    const doc = await native('DOM.getDocument');
    const node = await native('DOM.querySelector', {nodeId: doc.root.nodeId, selector});
    assert(node.nodeId, 'FILE_INPUT_MISSING');
    await native('DOM.setFileInputFiles', {nodeId: node.nodeId, files: [b.mediaPath]});
    return {sha256: proof.sha256, bytes: proof.bytes, binding: await op('binding')};
  }
  return {runtime, platform, save, op, dom, native, inspect, waitProof, focus, click, composer, fill, attach};
}
