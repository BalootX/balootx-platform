import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
export const sha=x=>createHash('sha256').update(x).digest('hex');
export function visibleCards(v){
 const card=x=>{
  const s=String(x).trim().toUpperCase().replaceAll('10','T');
  let m=s.match(/^([SHDC])([AKQJT789])$/);
  if(!m){m=s.match(/^([AKQJT789])([SHDC])$/);assert(m,'CARD_NOTATION_REQUIRED');m=[m[0],m[2],m[1]];}
  return m[1]+(m[2]==='T'?'10':m[2]);
 };
 assert(v&&Array.isArray(v.hand)&&v.hand.length>0&&Array.isArray(v.table),'OBSERVED_CARDS_REQUIRED');
 return {hand:v.hand.map(card).sort(),table:v.table.map(card).sort()};
}
export const visibleFingerprint=v=>sha(JSON.stringify(visibleCards(v)));

const norm=x=>String(x).replaceAll('\\','/').toLowerCase();

export function releaseAuthority(r,b){
 assert.equal(r.status,'RELEASED_CONDITIONAL_CURRENT_SLOT','CURRENT_SLOT_RELEASE_REQUIRED');
 assert.equal(r.deal,b.dealNumber,'RELEASE_NUMBER');
 assert.deepEqual(r.candidate,{deal:b.deal,artifactSha256:b.mediaSha256,slotUtc:b.slotUtc,captionSha256:b.captionSha256,visibleFingerprintSha256:b.opener.sha256},'EXACT_RELEASE_CANDIDATE');
 assert.equal(r.slotUtc,b.slotUtc,'RELEASE_SLOT');
 assert.deepEqual(r.platforms,{Instagram:1,TikTok:1},'ONE_POST_EACH_EXACT_RELEASE');
 assert.equal(r.requestSha256,b.contentRequestSha256,'RELEASE_REQUEST_BINDING');
 assert.equal(r.handoffSha256,b.contentHandoffSha256,'RELEASE_CONTENT_HANDOFF');
 assert.equal(r.caption.sha256,b.captionSha256,'RELEASE_CAPTION');
 assert.equal(r.caption.characters,b.captionCharacters,'RELEASE_CAPTION_CHARACTERS');
 assert.equal(r.caption.utf8Bytes,b.captionUtf8Bytes,'RELEASE_CAPTION_BYTES');
 return true;
}

export function pairedCounterpart(b,row,platform,authority){
 try{releaseAuthority(authority,b);}catch{return false;}
 return b.platforms.includes(platform)&&b.platforms.includes(row.platform)&&row.platform!==platform&&
  !!row.id&&!!row.url&&row.artifactSha256===b.mediaSha256&&row.deal===b.deal&&row.slotUtc===b.slotUtc&&
  norm(row.releaseReceipt)===norm(b.releasePath)&&row.captionSha256===b.captionSha256&&
  row.status==='POSTED_ACTUAL_CAPTION_CONFIRMED';
}

export function familyAdmission(b,latest,proof,{ledgerSha256,runtime=true}={}){
 assert(latest.scenarioFamily,'LATEST_FAMILY_UNKNOWN');
 assert(latest.scenarioFamily!==b.opener.family,'CONSECUTIVE_FAMILY_REPEAT');
 const known=b.familyCases.find(x=>x.deal===latest.deal&&x.visibleFingerprintSha256===latest.visibleOpenerSha256);
 assert(known,'ACTUAL_PREDECESSOR_TACTIC_EVIDENCE_REQUIRED');
 assert.equal(latest.scenarioFamily,known.family,'LATEST_FAMILY_EVIDENCE_BINDING');
 const sameMechanism=known.mechanism===b.opener.mechanism;
 if(sameMechanism){
  const review=proof?.familyReassessment;
  assert(runtime&&review?.status==='FRESH_ACTUAL_DIFFERENT_FAMILIES_CONFIRMED'&&review.differentFamilies===true,'SAME_MECHANISM_REASSESS_OR_SKIP');
  assert(review.predecessorId===latest.id&&review.predecessorFingerprint===latest.visibleOpenerSha256&&review.predecessorFamily===latest.scenarioFamily&&
   review.candidateDeal===b.deal&&review.candidateFingerprint===b.opener.sha256&&review.candidateFamily===b.opener.family&&review.ledgerSha256===ledgerSha256,'FRESH_FAMILY_REASSESSMENT_BINDING');
  assert(typeof review.rationale==='string'&&review.rationale.trim().length>0&&review.evidence?.length>0,'SUBSTANTIVE_FAMILY_REASSESSMENT_REQUIRED');
 }
 return {actualPredecessorId:latest.id,actualPredecessorFamily:latest.scenarioFamily,actualMechanism:known.mechanism,
  candidateFamily:b.opener.family,candidateMechanism:b.opener.mechanism,sameMechanism,
  decision:sameMechanism?'FRESH_SPECIFIC_DIFFERENT_FAMILIES_REVIEW':'DIFFERENT_EVIDENCED_MECHANISMS',plannedSequenceUsedAsPass:false};
}

export function actualPredecessorGuard(b,latest,proof,platform,ledgerSha256,nowMs){
 assert(proof?.status==='FRESH_ACTUAL_PREDECESSOR_CONFIRMED','FRESH_ACTUAL_PREDECESSOR_PROOF_REQUIRED');
 assert(proof.platform===platform&&proof.account===b.account&&proof.deal===b.deal&&proof.slotUtc===b.slotUtc&&proof.releaseSha256===b.releaseSha256,'ACTUAL_PREDECESSOR_SCOPE');
 assert(proof.ledgerSha256===ledgerSha256&&proof.actualPredecessorId===latest.id&&proof.actualPredecessorUrl===latest.url,'ACTUAL_PREDECESSOR_LEDGER_CHANGED');
 const at=Date.parse(proof.observedAtUtc);assert(at>=Date.parse(b.slotUtc)&&at<=nowMs,'DUE_TIME_PREDECESSOR_OBSERVATION_REQUIRED');
 assert(proof.session?.targetId&&proof.session.claimSha256,'ACTUAL_PREDECESSOR_OWN_SESSION_REQUIRED');
 const kind=platform==='TikTok'?'COMPLETE_UNFILTERED_STUDIO':'FRESH_PROFILE_GRID';
 assert(proof.evidence?.some(x=>x.kind===kind),'FRESH_GRID_OR_COMPLETE_STUDIO_REQUIRED');
 assert(proof.evidence.every(x=>x.path&&/^[a-f0-9]{64}$/.test(x.sha256)),'ACTUAL_PREDECESSOR_EVIDENCE_PINS');
}

export function historyAdmission(b,ledger,platform,{authority,ledgerSha256,actualPredecessor,runtime=true,nowMs=Date.now()}={}){
 releaseAuthority(authority,b);
 const candidate=visibleFingerprint(b.opener);assert.equal(candidate,b.opener.sha256,'CANDIDATE_FINGERPRINT');
 const rows=ledger.records.filter(r=>r.id||r.url),compared=[];
 for(const id of b.knownMinimumIds)assert(rows.some(r=>r.id===id),'KNOWN_HISTORY_ROW_REMOVED '+id);
 assert.equal(new Set(rows.map(r=>r.platform+':'+r.id)).size,rows.length,'AMBIGUOUS_KNOWN_HISTORY_ID');
 for(const row of rows){
  assert(row.deal&&row.visibleOpener&&row.visibleOpenerSha256,'MISSING_POSTED_FINGERPRINT '+row.id);
  const fingerprint=visibleFingerprint(row.visibleOpener),paired=pairedCounterpart(b,row,platform,authority);
  assert.equal(fingerprint,row.visibleOpenerSha256,'LEDGER_FINGERPRINT_BINDING '+row.id);
  assert(fingerprint!==candidate||paired,'KNOWN_POSTED_OPENER_REPEAT '+row.id);
  if(paired)assert.equal(fingerprint,candidate,'PAIRED_FINGERPRINT_MISMATCH');
  compared.push({id:row.id,platform:row.platform,status:row.status,sha256:fingerprint,pairedCurrentReleaseCounterpart:paired});
 }
 const time=r=>Date.parse(r.postedAtUtc||r.publishedAtUtc||r.postedAtUtcRange?.at(-1)||r.publishedAtUtcCandidates?.at(-1)||r.postedAtUtcCandidates?.at(-1)||'')||0;
 const own=rows.filter(r=>r.platform===platform).sort((a,z)=>time(z)-time(a));
 const latest=own[0];assert(latest,'LATEST_PLATFORM_POST_REQUIRED');
 assert(!own[1]||time(latest)!==time(own[1]),'AMBIGUOUS_ACTUAL_PREDECESSOR_TIME');
 assert(latest.captionSha256,'MISSING_LATEST_CAPTION');assert(latest.captionSha256!==b.captionSha256,'CONSECUTIVE_CAPTION_REPEAT');
 if(runtime)actualPredecessorGuard(b,latest,actualPredecessor,platform,ledgerSha256,nowMs);
 const family=familyAdmission(b,latest,actualPredecessor,{ledgerSha256,runtime});
 return {candidateSha256:candidate,compared,knownEverPostedIds:rows.length,includesRemoved:true,latestPlatformPost:latest.id,
  family,captionRule:'DIFFERENT_FROM_ACTUAL_LATEST',runtimeProof:runtime?'FRESH_ACTUAL_PREDECESSOR_BOUND':'NOT_RUN_OFFLINE_LEDGER_SNAPSHOT_ONLY'};
}
