import assert from 'node:assert/strict';
import {studioPostsDom} from '../../scripts/publisher/dom.mjs';
import {waitThenJudge} from '../../scripts/publisher/readiness.mjs';
class Element{
 constructor(tag,text='',attrs={},children=[]){this.tagName=tag.toUpperCase();this.nodeType=1;this.ownText=text;this.attrs=attrs;this.children=children;for(const c of children)c.parentElement=this;}
 get innerText(){return [this.ownText,...this.children.map(c=>c.innerText)].filter(Boolean).join('\n')}
 get href(){return this.attrs.href}
 get disabled(){return !!this.attrs.disabled}
 getClientRects(){return this.attrs.hidden?[]:[{}]}
 getAttribute(k){return this.attrs[k]??null}
 hasAttribute(k){return k in this.attrs}
 querySelectorAll(selector){const tags=selector.split(',').map(s=>s.split('[')[0].toUpperCase());return this.children.flatMap(c=>[...(tags.includes(c.tagName)&&(c.tagName!=='A'||c.hasAttribute('href'))?[c]:[]),...c.querySelectorAll(selector)])}
 closest(){return this.attrs.inert?this:null}
}
const el=(...a)=>new Element(...a),button=(name,attrs={})=>el('button',name,attrs);
let current;
globalThis.location={origin:'https://www.tiktok.com',pathname:'/tiktokstudio/upload',href:'https://www.tiktok.com/tiktokstudio/upload'};
globalThis.getComputedStyle=e=>({visibility:'visible',pointerEvents:e.attrs.pointerEvents||'auto'});
globalThis.document={querySelectorAll:s=>current.querySelectorAll(s),getElementById:()=>null};
const nav=(post)=>el('nav','',{},[button('Home'),post,button('Analytics'),button('Comments')]);
const fixture=(...children)=>current=el('body','',{},children);
for(const tag of ['button','a']){
 const post=tag==='button'?button('Posts'):el('a','Posts',{href:'https://www.tiktok.com/tiktokstudio/content'});
 let clock=0,reads=0,proof;
 await waitThenJudge('delayed-'+tag,async()=>{fixture(++reads<3?nav(button('Loading')):nav(post));return studioPostsDom();},s=>assert.equal(s.control.tag,tag.toUpperCase()),{clock:()=>clock,sleep:async ms=>{clock+=ms},record:p=>proof=p});
 assert.equal(reads,3);assert.equal(proof.polls.filter(p=>p.ready===false).length,2);assert.equal(proof.state.scope.kind,'STUDIO_MANAGE_NAVIGATION');
}
fixture(nav(button('Other')));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(nav(button('Posts',{disabled:true})));assert.throws(studioPostsDom,/DISABLED/);
fixture(nav(button('Posts',{'aria-disabled':'true'})));assert.throws(studioPostsDom,/DISABLED/);
fixture(nav(button('Posts',{pointerEvents:'none'})));assert.throws(studioPostsDom,/DISABLED/);
fixture(nav(button('Other')),button('Posts'));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(nav(button('Posts')),nav(button('Posts')));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(el('nav','',{},[button('Home'),button('Posts'),button('Posts'),button('Analytics'),button('Comments')]));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(nav(el('div','Posts')));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(nav(el('a','Posts',{href:'https://evil.invalid/tiktokstudio/content'})));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(nav(button('Posts elsewhere')));assert.throws(studioPostsDom,/UNIQUE_STUDIO/);
fixture(nav(button('not-name',{'aria-label':'Posts'})));assert.equal(studioPostsDom().control.name,'Posts');
location.pathname='/@synthetic_creator';assert.throws(studioPostsDom,/ACTUAL_STUDIO_URL/);location.pathname='/tiktokstudio/upload';
let clock=0,reads=0,proof;
await assert.rejects(()=>waitThenJudge('boundary',async()=>({}),()=>{throw Error('SUBSTANTIVE_SELECTOR_FAILURE')},{timeoutMs:20000,clock:()=>clock,sleep:async ms=>{clock+=ms},readWithin:async read=>{if(++reads===2){clock=20000;throw Error('PAGE_READ_DEADLINE')}return read()},record:p=>proof=p}),/SUBSTANTIVE_SELECTOR_FAILURE; deadline=PAGE_READ_DEADLINE/);
assert.equal(proof.lastCompletedFailure,'SUBSTANTIVE_SELECTOR_FAILURE');assert.equal(proof.deadlineCause,'PAGE_READ_DEADLINE');assert.equal(proof.polls.length,2);assert.equal(proof.polls[0].completed,true);assert.equal(proof.polls[1].completed,false);assert.equal(proof.elapsedMs,20000);
reads=0;await assert.rejects(()=>waitThenJudge('challenge',async()=>{reads++;return {blocker:true}},()=>assert.fail('must not judge'),{record:p=>proof=p}),/OWNER_ONLY_PROMPT/);assert.equal(reads,1);assert.equal(proof.status,'STOP_OWNER_ONLY_PROMPT');
console.log('PASS: delayed BUTTON and valid interactive LINK in actual Studio scope/exact accessible name; reject missing/ambiguous/disabled/wrong scope/global text/foreign URL/challenge; substantive completed failure preserved when final read races20s deadline');
