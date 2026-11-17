import assert from 'node:assert/strict';
import {qualify,composerDom} from '../../scripts/publisher/dom.mjs';
import {fixture as publisherFixture} from './fixture.mjs';
const {b:binding,caption}=publisherFixture();
const check=(state,platform)=>qualify(state,platform,caption,binding);
for(const platform of ['Instagram','TikTok']){
 const s={blocker:false,scope:{selector:'#composer',kind:platform==='Instagram'?'ACTIVE_INSTAGRAM_CAPTION_DIALOG':'ACTIVE_TIKTOK_DESCRIPTION_COMPOSER'},editor:{selector:'#caption',label:'Add a caption...',text:caption},counter:{text:platform==='Instagram'?''+caption.length+'/2,200':''+caption.length+'/4000'},submit:{selector:'#submit',text:platform==='Instagram'?'Share':'Post',enabled:true}};
 assert.equal(check(s,platform).editor.text,caption);
 assert.throws(()=>check({...s,counter:{text:'106/4000'}},platform),/PAGE_COUNTER/);
 assert.throws(()=>check({...s,counter:{text:''}},platform),/PAGE_COUNTER/);
 assert.throws(()=>check({...s,editor:{...s.editor,text:''}},platform),/CAPTION_MISMATCH/);
 assert.throws(()=>check({...s,submit:{...s.submit,enabled:false}},platform),/DISABLED/);
 assert.throws(()=>check({...s,scope:null},platform),/BOUND_COMPOSER/);
 assert.throws(()=>check({...s,blocker:true},platform),/OWNER_ONLY/);
}
// A document-wide editable or counter cannot substitute for a missing active composer.
globalThis.document={querySelectorAll(){return []}};
globalThis.location={origin:'https://example.invalid',pathname:'/synthetic_creator/'};
assert.throws(()=>composerDom('Instagram'),/UNIQUE_ACTIVE_COMPOSER/);
location.origin='https://example.invalid';location.pathname='/@synthetic_creator';
assert.throws(()=>composerDom('TikTok'),/UPLOAD_COMPOSER_URL/);
// Minimal DOM fixtures include unrelated document editors/counters to verify containment.
class Element{
 constructor(tag,text='',attrs={},children=[]){this.tagName=tag.toUpperCase();this.nodeType=1;this.ownText=text;this.attrs=attrs;this.children=children;for(const c of children)c.parentElement=this;}
 get innerText(){return [this.ownText,...this.children.map(c=>c.innerText)].filter(Boolean).join('\n')}
 get textContent(){return this.ownText+this.children.map(c=>c.textContent).join('')}
 getClientRects(){return [{}]}
 getAttribute(k){return this.attrs[k]??null}
 matches(selector){return selector.split(',').some(part=>{const tag=part.match(/^[a-z][a-z0-9]*/i)?.[0];if(tag&&tag.toUpperCase()!==this.tagName)return false;return [...part.matchAll(/\[([^\]]+)\]/g)].every(([,attr])=>{const [k,v]=attr.split('=');return v===undefined?this.getAttribute(k)!==null:this.getAttribute(k)===v.replaceAll('"','');});})}
 querySelectorAll(selector){return this.children.flatMap(c=>[...(c.matches(selector)?[c]:[]),...c.querySelectorAll(selector)])}
 closest(selector){return this.matches(selector)?this:this.parentElement?.closest(selector)||null}
 contains(node){return node===this||this.children.some(c=>c.contains(node))}
}
const n=(...args)=>new Element(...args);
const editable=(text,attrs={})=>n('div',text,{contenteditable:'true',...attrs});
globalThis.getComputedStyle=()=>({visibility:'visible'});
function fixture(children){const body=n('body','',{},children);globalThis.document={querySelectorAll:s=>body.querySelectorAll(s),activeElement:null};return body;}
const ig=()=>n('div','',{role:'dialog'},[n('h1','New reel'),editable(caption,{'aria-label':'Add a caption...'}),n('span',''+caption.length+'/2,200'),n('button','Share')]);
fixture([editable('unrelated'),n('span','106/2,200'),ig()]);location.origin='https://example.invalid';location.pathname='/synthetic_creator/';
assert.equal(check(composerDom('Instagram'),'Instagram').counter.text,''+caption.length+'/2,200');
fixture([ig(),ig()]);assert.throws(()=>composerDom('Instagram'),/UNIQUE_ACTIVE_COMPOSER_SCOPE_REQUIRED:2/);
const tt=n('form','',{},[n('div','',{},[n('label','Description'),editable(caption,{role:'combobox'}),n('span',''+caption.length+'/4000')]),n('button','Post')]);
fixture([editable('outside',{role:'combobox'}),n('span','106/4000'),tt]);location.origin='https://example.invalid';location.pathname='/tiktokstudio/upload';
assert.equal(check(composerDom('TikTok'),'TikTok').counter.text,''+caption.length+'/4000');
tt.children[0].children[2].ownText='0/4000';assert.throws(()=>check(composerDom('TikTok'),'TikTok'),/ACTUAL_PAGE_COUNTER/);
console.log('PASS:current-packet IG/TT composer exact bound caption/page counters; wrong/missing counter,caption,disabled submit,challenge,ambiguous scopes and global editor fallback refused offline');
