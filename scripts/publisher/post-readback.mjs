import assert from 'node:assert/strict';
import {sha} from './policy.mjs';

export function tiktokPostDom({author}) {
  const visible=e=>e.getClientRects().length>0&&getComputedStyle(e).visibility!=='hidden';
  const authors=[...document.querySelectorAll('a[href]')].filter(visible).filter(a=>new URL(a.href,location.href).pathname==='/@'+author);
  const descriptions=[...document.querySelectorAll('h1,h2,[data-e2e]')].filter(visible).filter(e=>!e.closest('[data-e2e*="comment"]')).filter(e=>e.matches('h1,h2')||/desc/i.test(e.getAttribute('data-e2e')||''));
  const bound=[];for(const e of descriptions){for(let r=e.parentElement,n=0;r&&n<8;r=r.parentElement,n++){if(r.querySelectorAll('video').length===1&&authors.some(a=>r.contains(a))){bound.push({text:e.innerText,author});break;}}}
  return {url:location.origin+location.pathname,captions:bound,editing:[...document.querySelectorAll('[role="dialog"] [contenteditable="true"]')].some(visible)};
}

export async function readPost(client, {url, caption, deadline, postDom = tiktokPostDom}) {
  assert(Date.now() < deadline, 'CAPTION_120S_DEADLINE');
  const author = client.runtime.binding.account;
  const result = await client.waitProof('actual-post-caption', async () => {
    const state = await client.inspect();
    assert(!state.blocker, 'OWNER_ONLY_PROMPT');
    return {...await client.dom('(' + postDom.toString() + ')(' + JSON.stringify({author, caption}) + ')'), blocker: state.blocker};
  }, state => {
    assert.equal(state.url, url, 'EXACT_NEW_URL');
    assert(!state.editing, 'NO_CAPTION_EDITOR');
    assert(state.captions.some(row => row.author === author), 'ACTUAL_AUTHOR_BOUND_CAPTION_REQUIRED');
  }, {timeoutMs: Math.min(20000, deadline - Date.now())});
  const rows = result.state.captions.filter(row => row.author === author);
  const exact = rows.find(row => row.text === caption);
  return {url, actualCaption: exact?.text ?? rows[0].text, actualCaptionSha256: sha(exact?.text ?? rows[0].text),
    exact: !!exact, within120s: Date.now() <= deadline, proof: result};
}

export function postCaptionDom({ author, caption }) {
  const visible = e => e.getClientRects().length > 0 && getComputedStyle(e).visibility !== 'hidden';
  if (/\/accounts\/|\/challenge\/|\/checkpoint\/|captcha|two_factor/i.test(location.pathname)
    || [...document.querySelectorAll('input[type="password"],input[autocomplete="one-time-code"],iframe[src*="captcha"]')].some(visible)) throw Error('AUTH_OR_CHALLENGE');
  if (/503 Service Unavailable|HTTP ERROR 503/i.test(document.body.innerText)) throw Error('HTTP_503');
  const editing = [...document.querySelectorAll('[role="dialog"],dialog')].filter(visible).some(d =>
    [...d.querySelectorAll('textarea,[contenteditable="true"]')].some(visible));
  const main = document.querySelector('main,[role="main"]');
  // The observed full-permalink layout has the header author link, then the caption author link.
  const captionOwner = main ? [...main.querySelectorAll('a[href]')].filter(visible).filter(a =>
    new URL(a.href, location.href).pathname === '/' + author + '/' && a.innerText.trim() === author)[1] : null;
  const captions = main ? [...main.querySelectorAll('h1,span')].filter(visible).filter(e => e.innerText === caption).flatMap(e => {
    for (let row = e.parentElement, depth = 0; row && row !== main && depth < 8; row = row.parentElement, depth++) {
      if (row.querySelector('video') || row.innerText.length > 2500) break;
      const links = [...row.querySelectorAll('a[href]')].filter(visible).filter(a =>
        /^\/[A-Za-z0-9._]+\/$/.test(new URL(a.href, location.href).pathname));
      if (links.length === 1 && links[0] === captionOwner) return [{ text: e.innerText, author, binding: 'first_caption_owner_row' }];
      if (links.length) break;
    }
    return [];
  }) : [];
  return { url: location.origin + location.pathname, editing, captions };
}
