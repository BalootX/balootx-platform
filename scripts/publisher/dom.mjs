import assert from 'node:assert/strict';
import {sha} from './policy.mjs';
export function pageState(){
 const vis=e=>e.getClientRects().length>0&&getComputedStyle(e).visibility!=='hidden';
 const css=e=>{const a=[];for(;e&&e.nodeType===1;e=e.parentElement){const s=e.parentElement?[...e.parentElement.children].filter(n=>n.tagName===e.tagName):[e];a.unshift(e.tagName.toLowerCase()+':nth-of-type('+(s.indexOf(e)+1)+')')}return a.join('>')};
 const text=document.body?.innerText||'';
 const blocker=/\/(?:login|challenge|checkpoint|two_factor)(?:\/|$)/i.test(location.pathname)||[...document.querySelectorAll('input[type=password],input[autocomplete=one-time-code],iframe[src*=captcha]')].some(vis)||/Confirm it.s you|Verify your identity|Enter (the )?security code|Suspicious login|Help us confirm|Accept (?:the )?(?:terms|Terms)|Agree to (?:the )?(?:terms|Terms)|Sign in to continue|Log in to TikTok|Verify (?:your|you are)|CAPTCHA|Enter (?:your )?(?:password|OTP)|Payment required/.test(text);
 return {url:location.origin+location.pathname,blocker,text:text.slice(0,10000),controls:[...document.querySelectorAll('button,[role=button],[role=menuitem],svg[aria-label="New post"]')].filter(vis).map(e=>({selector:css(e),tag:e.tagName,text:(e.innerText||'').trim().slice(0,150),label:e.getAttribute('aria-label'),enabled:!e.disabled&&e.getAttribute('aria-disabled')!=='true'})),editors:[...document.querySelectorAll('textarea,[contenteditable=true]')].filter(vis).map(e=>({selector:css(e),tag:e.tagName,label:e.getAttribute('aria-label'),placeholder:e.getAttribute('placeholder'),text:e.tagName==='TEXTAREA'?e.value:e.textContent,active:document.activeElement===e})),counters:[...document.querySelectorAll('div,span')].filter(vis).filter(e=>/^\s*[\d,]+\s*\/\s*[\d,]+\s*$/.test(e.innerText)).map(e=>({selector:css(e),text:e.innerText})),files:[...document.querySelectorAll('input[type=file]')].map(e=>({selector:css(e),accept:e.accept,files:[...e.files].map(f=>({name:f.name,size:f.size}))})),links:[...document.querySelectorAll('a[href]')].filter(vis).map(e=>({text:e.innerText.slice(0,200),url:e.href.split('?')[0].split('#')[0]}))};
}
export function pointerTarget(selector){const raw=document.querySelector(selector);if(!raw)throw Error('CONTROL_MISSING');const e=raw.closest('button,[role="button"],a,[contenteditable="true"],input,textarea,[role="menuitem"]');if(!e)throw Error('INTERACTIVE_TARGET_REQUIRED');e.scrollIntoView({block:'center',inline:'center',behavior:'instant'});const r=e.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2,hit=document.elementFromPoint(x,y);if(!(r.width>0&&r.height>0&&x>=0&&y>=0&&x<innerWidth&&y<innerHeight&&hit&&(hit===e||e.contains(hit))))throw Error('POINTER_TARGET_NOT_IN_VIEW_OR_OCCLUDED');if(e.disabled||e.getAttribute('aria-disabled')==='true'||e.closest('[inert],[aria-disabled="true"]')||getComputedStyle(e).pointerEvents==='none')throw Error('CONTROL_DISABLED');return{x,y,selector,interactiveTag:e.tagName,interactiveRole:e.getAttribute('role'),viewport:{width:innerWidth,height:innerHeight},rect:{x:r.x,y:r.y,width:r.width,height:r.height},sameTarget:true,containedLabel:hit!==e,enabled:true,label:e.innerText||e.getAttribute('aria-label')||'',url:location.origin+location.pathname};}
export function composerDom(platform){
 const vis=e=>e.getClientRects().length>0&&getComputedStyle(e).visibility!=='hidden';
 const css=e=>{const a=[];for(;e&&e.nodeType===1;e=e.parentElement){const siblings=e.parentElement?[...e.parentElement.children].filter(n=>n.tagName===e.tagName):[e];a.unshift(e.tagName.toLowerCase()+':nth-of-type('+(siblings.indexOf(e)+1)+')')}return a.join('>')};
 const editors=r=>[...r.querySelectorAll('textarea,[contenteditable="true"]')].filter(vis);
 const counters=(r,e,limit)=>[...r.querySelectorAll('div,span')].filter(vis).filter(n=>!e.contains(n)&&!n.contains(e)).flatMap(n=>{const text=n.innerText.trim(),m=text.match(/^([\d,]+)\s*\/\s*([\d,]+)$/);return m&&Number(m[2].replaceAll(',',''))===limit&&![...n.children].some(c=>vis(c)&&c.innerText.trim()===text)?[{selector:css(n),text,count:Number(m[1].replaceAll(',','')),limit}]:[]});
 const buttons=(r,label)=>[...r.querySelectorAll('button,[role="button"]')].filter(vis).filter(e=>e.innerText.trim()===label);
 const scopes=[];
 if(platform==='Instagram'){
  for(const r of [...document.querySelectorAll('[role="dialog"],dialog')].filter(vis)){
   const h=[...r.querySelectorAll('h1,h2,h3,[role="heading"]')].filter(vis).filter(e=>/^(New reel|Create new post|New post)$/.test(e.innerText.trim()));
   const e=editors(r).filter(e=>e.closest('[role="dialog"],dialog')===r&&/caption/i.test(e.getAttribute('aria-label')||e.getAttribute('placeholder')||''));
   if(h.length!==1||e.length!==1)continue;
   const c=counters(r,e[0],2200),s=buttons(r,'Share').filter(e=>e.closest('[role="dialog"],dialog')===r);
   if(c.length===1&&s.length===1)scopes.push({r,e:e[0],c:c[0],s:s[0],heading:h[0].innerText.trim(),kind:'ACTIVE_INSTAGRAM_CAPTION_DIALOG'});
  }
 }else if(platform==='TikTok'){
  if(!/^\/tiktokstudio\/upload\/?$/.test(location.pathname))throw Error('TIKTOK_UPLOAD_COMPOSER_URL');
  for(const label of [...document.querySelectorAll('label,h1,h2,h3,span,div')].filter(vis).filter(e=>e.innerText.trim()==='Description')){
   for(let r=label.parentElement,n=0;r&&n<8&&!['BODY','HTML'].includes(r.tagName);r=r.parentElement,n++){
    const e=editors(r).filter(e=>e.getAttribute('role')==='combobox');if(e.length!==1)continue;
    const c=counters(r,e[0],4000);if(c.length!==1)continue;
    for(let form=r,k=0;form&&k<8&&!['BODY','HTML'].includes(form.tagName);form=form.parentElement,k++){
     if(editors(form).filter(e=>e.getAttribute('role')==='combobox').length!==1)break;
     const s=buttons(form,'Post');if(s.length===1){scopes.push({r,e:e[0],c:c[0],s:s[0],submitRoot:form,heading:'Description',kind:'ACTIVE_TIKTOK_DESCRIPTION_COMPOSER'});break;}
    }
    break;
   }
  }
 }else throw Error('EXPLICIT_PLATFORM_REQUIRED');
 const unique=[...new Map(scopes.map(s=>[css(s.r),s])).values()];if(unique.length!==1)throw Error('UNIQUE_ACTIVE_COMPOSER_SCOPE_REQUIRED:'+unique.length);
 const {r,e,c,s,heading,kind,submitRoot}=unique[0];
 return {url:location.origin+location.pathname,scope:{selector:css(r),submitScope:css(submitRoot||r),heading,kind},editor:{selector:css(e),label:e.getAttribute('aria-label')||heading,role:e.getAttribute('role'),text:e.tagName==='TEXTAREA'?e.value:e.textContent,active:document.activeElement===e},counter:c,submit:{selector:css(s),text:s.innerText.trim(),enabled:!s.disabled&&s.getAttribute('aria-disabled')!=='true'&&!s.closest('[inert],[aria-disabled="true"]')}};
}
export function qualify(state,platform,caption,binding){
 const label=platform==='Instagram'?'Share':'Post',limit=platform==='Instagram'?2200:4000;
 assert(!state.blocker,'OWNER_ONLY_PROMPT');
 assert.equal(state.scope?.kind,platform==='Instagram'?'ACTIVE_INSTAGRAM_CAPTION_DIALOG':'ACTIVE_TIKTOK_DESCRIPTION_COMPOSER','BOUND_COMPOSER_REQUIRED');
 assert.equal(state.editor.text,caption,'FIELD_CAPTION_MISMATCH');
 const actual=state.counter.text.trim().match(/^([\d,]+)\s*\/\s*([\d,]+)$/);
 assert(actual&&Number(actual[1].replaceAll(',',''))===binding.captionCharacters&&Number(actual[2].replaceAll(',',''))===limit,'ACTUAL_PAGE_COUNTER_REQUIRED');
 assert.equal(state.submit.text,label,'BOUND_SUBMIT_CONTROL');assert(state.submit.enabled,'SUBMIT_DISABLED');
 return{...state,captionSha256:sha(state.editor.text),expectedCharacters:binding.captionCharacters};
}
export function studioPostsDom(){
 if(location.origin!=='https://www.tiktok.com'||!/^\/tiktokstudio\//.test(location.pathname))throw Error('ACTUAL_STUDIO_URL_REQUIRED');
 const visible=e=>e.getClientRects().length>0&&getComputedStyle(e).visibility!=='hidden';
 const name=e=>(e.getAttribute('aria-label')||(e.getAttribute('aria-labelledby')||'').split(/\s+/).filter(Boolean).map(id=>document.getElementById(id)?.innerText||'').join(' ')||e.innerText||'').replace(/\s+/g,' ').trim();
 const interactive=e=>e.tagName==='BUTTON'||(e.tagName==='A'&&e.hasAttribute('href')&&(()=>{const u=new URL(e.href,location.href);return u.origin===location.origin&&u.pathname.startsWith('/tiktokstudio/');})());
 const controls=r=>[...r.querySelectorAll('button,a[href]')].filter(visible).filter(interactive);
 const css=e=>{const a=[];for(;e&&e.nodeType===1;e=e.parentElement){const s=e.parentElement?[...e.parentElement.children].filter(n=>n.tagName===e.tagName):[e];a.unshift(e.tagName.toLowerCase()+':nth-of-type('+(s.indexOf(e)+1)+')')}return a.join('>')};
 const matches=[];
 for(const candidate of controls(document).filter(e=>name(e)==='Posts')){
  for(let scope=candidate.parentElement,n=0;scope&&n<8&&!['BODY','HTML'].includes(scope.tagName);scope=scope.parentElement,n++){
   const items=controls(scope),anchors=['Home','Posts','Analytics','Comments'];
   if(!anchors.every(label=>items.filter(e=>name(e)===label).length===1))continue;
   matches.push({scope,candidate,anchors});break;
  }
 }
 if(matches.length!==1)throw Error('UNIQUE_STUDIO_POSTS_CONTROL_REQUIRED:'+matches.length);
 const {scope,candidate,anchors}=matches[0];
 const enabled=!candidate.disabled&&candidate.getAttribute('aria-disabled')!=='true'&&!candidate.closest('[inert],[aria-disabled="true"]')&&getComputedStyle(candidate).pointerEvents!=='none';
 if(!enabled)throw Error('STUDIO_POSTS_CONTROL_DISABLED');
 return {url:location.origin+location.pathname,scope:{kind:'STUDIO_MANAGE_NAVIGATION',selector:css(scope),anchors},control:{selector:css(candidate),name:name(candidate),tag:candidate.tagName,href:candidate.tagName==='A'?candidate.href:null,enabled}};
}
