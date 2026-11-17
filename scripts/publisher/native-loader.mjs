import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {sha} from './policy.mjs';
import {localConfig, guard} from './runtime.mjs';

export function nativeRequest(arg, binding) {
  const allowed = ['Page.bringToFront', 'Emulation.setFocusEmulationEnabled', 'Input.insertText',
    'Input.dispatchKeyEvent', 'Input.dispatchMouseEvent', 'DOM.getDocument', 'DOM.querySelector', 'DOM.setFileInputFiles'];
  assert(allowed.includes(arg.method), 'METHOD_NOT_ALLOWED');
  if (arg.method === 'Input.insertText') {
    assert(typeof arg.params?.text === 'string' && sha(arg.params.text) === binding.captionSha256, 'SEALED_CAPTION_REQUIRED');
  }
  if (arg.method === 'DOM.setFileInputFiles') {
    assert.deepEqual(arg.params?.files, [binding.mediaPath], 'EXACT_FILE_REQUIRED');
    const bytes = readFileSync(binding.mediaPath);
    assert(bytes.length === binding.mediaBytes && sha(bytes) === binding.mediaSha256, 'UPLOAD_HASH_MISMATCH');
  }
  return arg;
}

export function extend(source, runtime) {
  assert.equal(sha(source), runtime.config.helperSha256, 'SHARED_HELPER_SOURCE_CHANGED');
  const needle = "else if (op === 'eval')";
  assert.equal(source.split(needle).length, 2, 'UNIQUE_SHARED_NATIVE_INSERTION_REQUIRED');
  const prefix = `import {localConfig as publisherConfig, guard as publisherGuard} from ${JSON.stringify(new URL('./runtime.mjs', import.meta.url).href)};
import {nativeRequest as publisherNative} from ${JSON.stringify(import.meta.url)};
const publisherRuntime = publisherConfig(process.env.PUBLISHER_CONFIG);
publisherGuard(publisherRuntime);
process.env.CDP_CTL_PORT = new URL(publisherRuntime.config.controlUrl).port;
`;
  const authorization = `const publisherClaim = publisherGuard(publisherRuntime);
        if(req.method !== 'POST' || req.headers.origin || req.headers['x-publisher-claim'] !== publisherClaim.claimSha256) throw Error('OWN_LOCAL_REQUEST_REQUIRED');
        if(publisherRuntime.config.protectedTargetIds.includes(targetId)) throw Error('PROTECTED_TARGET');
        const { op, arg } = JSON.parse(body || '{}');
        if(op === 'navigate') {
          const url = new URL(arg);
          if(arg !== publisherRuntime.config.profileCheckUrl && (url.origin === 'null' || !publisherRuntime.config.allowedOrigins.includes(url.origin))) throw Error('NAVIGATION_ORIGIN_REQUIRED');
        }
        if(!['navigate','url','title','eval','text','front','quit','binding','native'].includes(op)) throw Error('OPERATION_NOT_ALLOWED');`;
  const route = `else if(op === 'binding') out = JSON.stringify({targetId, ...publisherClaim});
        else if(op === 'native') {
          publisherNative(arg, publisherRuntime.binding);
          out = JSON.stringify(await send(arg.method, arg.params || {}, session));
        }
        `;
  assert.equal(source.split("const { op, arg } = JSON.parse(body || '{}');").length, 2, 'SHARED_REQUEST_ROUTE_CHANGED');
  return prefix + source.replace("const { op, arg } = JSON.parse(body || '{}');", authorization).replace(needle, route + needle);
}

export async function load(url, context, nextLoad) {
  const runtime = localConfig(process.env.PUBLISHER_CONFIG);
  if (url !== pathToFileURL(runtime.config.helperPath).href) return nextLoad(url, context);
  guard(runtime);
  const result = await nextLoad(url, context);
  return {...result, source: extend(String(result.source), runtime)};
}
