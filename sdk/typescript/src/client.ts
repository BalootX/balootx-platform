export type Json = null | boolean | number | string | Json[] | {[key:string]:Json};
export type ObjectJson = {[key:string]:Json};
export interface IdentityPageOptions {
  cursor?: string;
  limit?: number;
}
export interface IdentityItem {
  subjectId: string;
  kind: string;
  displayName: string;
  username: string|null;
  authenticated: boolean;
  guest: boolean;
  createdAt: string|null;
}
export interface PendingRequestItem {
  requestId: string;
  senderId: string;
  recipientId: string;
  status: 'pending';
  createdAt: string;
  senderDisplayName: string;
  senderUsername: string|null;
}
export interface IdentityPage<Item=IdentityItem> {
  items: Item[];
  nextCursor: string|null;
}
export interface ResearchRenderPrepareRequest {
  revealPolicy?: 'after_commit'|'after_completion'|'post_reveal_only';
  exportPreset?: 'decision_clip'|'hooked_flagship'|null;
  observerSeat?: 'north'|'west'|'south'|'east'|null;
}
export interface ResearchRecipeListOptions {
  status?: 'draft'|'published'|'retired';
  cursor?: string;
  limit?: number;
}
export interface ResearchRecipeListResponse {
  items: ObjectJson[];
  nextCursor: string|null;
}
export interface ResearchAuthorityV1 {
  schemaVersion: 1;
  backendId: 'ortools_cp_sat_v2'|'ortools_cp_sat_v3';
  backendVersion: string;
  qualificationAuthority: 'cp_sat';
  solverVersion: string;
  bundleRevisionId: string;
  bundleHash: string;
  domainIrHashes: Record<string,string>;
  loweringManifestHash: string;
  constructionModelHash: string;
  proofModelHash: string;
  graphHash: string;
  traceHash: string;
  proofOutcomeHash: string;
  sha256: string;
}
export interface RuntimeCompatibilityV1 {
  schemaVersion: 1;
  status: 'not_checked'|'passed'|'failed';
  engineVersion: string|null;
  bundleHash: string;
  traceHash: string;
  evidenceHash: string|null;
  failureCodes: string[];
  sha256: string;
}
export interface ExecutionReceiptV1 {
  schemaVersion: 1;
  semanticResultHash: string;
  stageTimings: Record<string,number>;
  cpuSeconds: number;
  peakMemoryBytes: number;
  conflicts: number;
  branches: number;
  cacheBytes: number;
  completedVariants: number;
  machine: ObjectJson;
  sha256: string;
}
export interface ResearchOperatorAuditV1Response {
  schemaVersion: 1;
  resultCode: string;
  researchAuthority: ResearchAuthorityV1;
  runtimeCompatibility: RuntimeCompatibilityV1;
  executionReceipt: ExecutionReceiptV1;
}
export interface ClientOptions { baseUrl?: string; apiKey?: string; trustedKey?: string; subjectId?: string; subjectName?: string; accessToken?: string; fetchImpl?: typeof fetch; credentials?: RequestCredentials }

export class BalootClient {
  private readonly baseUrl:string;
  private readonly apiKey?:string;
  private readonly trustedKey?:string;
  private readonly subjectId?:string;
  private readonly subjectName?:string;
  private accessToken?:string;
  private readonly fetchImpl:typeof fetch;
  private readonly credentials:RequestCredentials;
  constructor(options:ClientOptions={}) {
    this.baseUrl=(options.baseUrl??'http://127.0.0.1:8080').replace(/\/$/,'');
    this.apiKey=options.apiKey;this.trustedKey=options.trustedKey;this.subjectId=options.subjectId;this.subjectName=options.subjectName;this.accessToken=options.accessToken;this.fetchImpl=options.fetchImpl??fetch;this.credentials=options.credentials??'include';
  }
  bundle(mode='full'){return this.request(`/api/v1/bundles/current?mode=${encodeURIComponent(mode)}`)}
  health(){return this.request('/api/v1/health')}
  plugins(kind?:string){return this.request(`/api/v1/plugins${kind?`?kind=${encodeURIComponent(kind)}`:''}`)}
  listPlugins(){return this.plugins()}
  me(){return this.request('/api/v1/auth/me')}
  async guest(displayName?:string){const value=await this.post('/api/v1/auth/guest',{displayName:displayName??null});if(value.accessToken)this.accessToken=value.accessToken;return value}
  async register(username:string,password:string,displayName?:string){const value=await this.post('/api/v1/auth/register',{username,password,displayName:displayName??null});if(value.accessToken)this.accessToken=value.accessToken;return value}
  async login(username:string,password:string){const value=await this.post('/api/v1/auth/login',{username,password});if(value.accessToken)this.accessToken=value.accessToken;return value}
  async logout(){await this.request('/api/v1/auth/logout',{method:'POST'});this.accessToken=undefined}
  friends(){return this.request('/api/v1/friends')}
  friendsPage(options:IdentityPageOptions={}):Promise<IdentityPage>{
    return this.identityPage('/api/v2/friends',options);
  }
  friendRequests(options:IdentityPageOptions={}):Promise<IdentityPage<PendingRequestItem>>{
    return this.identityPage('/api/v2/friends/requests',options);
  }
  searchUsers(query:string,options:IdentityPageOptions={}):Promise<IdentityPage>{
    return this.identityPage('/api/v1/users:search',options,new URLSearchParams({query}));
  }
  private identityPage<Item=IdentityItem>(path:string,{cursor,limit}:IdentityPageOptions,params=new URLSearchParams()):Promise<IdentityPage<Item>>{
    if(cursor!==undefined)params.set('cursor',cursor);
    if(limit!==undefined)params.set('limit',String(limit));
    const query=params.toString();
    return this.request(query?`${path}?${query}`:path);
  }
  requestFriend(recipientId:string){return this.post('/api/v1/friends/requests',{recipientId})}
  validateConfiguration(request:ObjectJson){return this.post('/api/v1/configuration:validate',{request})}
  createGame(payload:ObjectJson){return this.post('/api/v1/games',payload)}
  game(id:string,revealHands=false){return this.request(`/api/v1/games/${encodeURIComponent(id)}?revealHands=${revealHands}`)}
  getGame(id:string,revealHands=false){return this.game(id,revealHands)}
  bid(id:string,payload:ObjectJson){return this.post(`/api/v1/games/${encodeURIComponent(id)}/bids`,payload)}
  play(id:string,payload:ObjectJson){return this.post(`/api/v1/games/${encodeURIComponent(id)}/actions`,payload)}
  playCard(id:string,payload:ObjectJson){return this.play(id,payload)}
  createLiveRoom(payload:ObjectJson){return this.post('/api/v1/live/rooms',payload)}
  websocketTicket(payload:ObjectJson){return this.post('/api/v1/security/websocket-tickets',payload)}
  advanceBots(id:string,stopBefore:string|null='south',maximumActions=64){const q=new URLSearchParams({maximumActions:String(maximumActions)});if(stopBefore!==null)q.set('stopBefore',stopBefore);return this.post(`/api/v1/games/${encodeURIComponent(id)}/bots:advance?${q}`,null)}
  nextRound(id:string,seed?:number){return this.post(`/api/v1/games/${encodeURIComponent(id)}/rounds:next${seed===undefined?'':`?seed=${seed}`}`,null)}
  validateContext(state:ObjectJson,rulesProfile='full'){return this.post('/api/v1/contexts:validate',{state,rulesProfile})}
  continueContext(state:ObjectJson,rulesProfile='full'){return this.post('/api/v1/contexts:continue',{state,rulesProfile})}
  analyze(payload:ObjectJson){return this.post('/api/v1/analyses',payload)}
  createVideo(payload:ObjectJson,idempotencyKey?:string){return this.request('/api/v1/video-jobs',{method:'POST',headers:{'Idempotency-Key':this.researchIdempotencyKey(idempotencyKey)},body:JSON.stringify(payload)})}
  videoJob(id:string){return this.request(`/api/v1/video-jobs/${encodeURIComponent(id)}`)}
  cancelVideo(id:string){return this.post(`/api/v1/video-jobs/${encodeURIComponent(id)}:cancel`,null)}
  createFoundryAnalysisBatch(campaignHash:string){return this.post('/api/v2/analysis-batches',{campaignHash})}
  foundryAnalysisBatch(id:string){return this.request(`/api/v2/analysis-batches/${encodeURIComponent(id)}`)}
  cancelFoundryAnalysisBatch(id:string){return this.post(`/api/v2/analysis-batches/${encodeURIComponent(id)}:cancel`,null)}
  createRenderBatch(items:ObjectJson[],idempotencyKey:string){return this.request('/api/v2/render-batches',{method:'POST',headers:{'Idempotency-Key':idempotencyKey},body:JSON.stringify({items})})}
  renderBatch(id:string){return this.request(`/api/v2/render-batches/${encodeURIComponent(id)}`)}
  cancelRenderBatch(id:string){return this.post(`/api/v2/render-batches/${encodeURIComponent(id)}:cancel`,null)}
  createContextSearch(parentTraceReference:ObjectJson,decisionId:string,contextProofHashes:string[],idempotencyKey:string){return this.request('/api/v2/context-searches',{method:'POST',headers:{'Idempotency-Key':idempotencyKey},body:JSON.stringify({parentTraceReference,decisionId,contextProofHashes})})}
  contextSearch(id:string){return this.request(`/api/v2/context-searches/${encodeURIComponent(id)}`)}
  cancelContextSearch(id:string){return this.post(`/api/v2/context-searches/${encodeURIComponent(id)}:cancel`,null)}
  createNaturalSakkaGeneration(rootSeed:number,idempotencyKey:string,initialDealer:'north'|'east'|'south'|'west'='east'){return this.request('/api/v2/sakka-generations',{method:'POST',headers:{'Idempotency-Key':idempotencyKey},body:JSON.stringify({rootSeed,initialDealer})})}
  naturalSakkaGeneration(id:string){return this.request(`/api/v2/sakka-generations/${encodeURIComponent(id)}`)}
  cancelNaturalSakkaGeneration(id:string){return this.post(`/api/v2/sakka-generations/${encodeURIComponent(id)}:cancel`,null)}
  foundrySakkas(options:{provenance?:'natural'|'curated';after?:string;limit?:number}={}){const q=new URLSearchParams({limit:String(options.limit??100)});if(options.provenance)q.set('provenance',options.provenance);if(options.after)q.set('after',options.after);return this.request(`/api/v2/sakkas?${q}`)}
  createNaturalSakka(traceReferences:ObjectJson[],target=152){return this.post('/api/v2/sakkas/natural',{traceReferences,target})}
  createCuratedSakka(selections:ObjectJson[],target=152){return this.post('/api/v2/sakkas/curated',{selections,target})}
  foundrySakka(sakkaHash:string){return this.request(`/api/v2/sakkas/${encodeURIComponent(sakkaHash)}`)}
  createCampaignEquivalenceProof(executions:ObjectJson[]){return this.post('/api/v2/campaign-equivalence-proofs',{executions})}
  foundryCampaignAttempts(campaignHash:string){return this.request(`/api/v2/campaigns/${encodeURIComponent(campaignHash)}/attempts`)}
  foundryProfile(){return this.request('/api/v2/foundry-profile')}
  foundryCampaigns(options:{after?:string;limit?:number}={}){const q=new URLSearchParams({limit:String(options.limit??100)});if(options.after)q.set('after',options.after);return this.request(`/api/v2/campaigns?${q}`)}
  createTraceImportCatalog(lane:string,entries:ObjectJson[]){return this.post('/api/v2/trace-import-catalogs',{lane,entries})}
  createScoreReachabilityCatalog(traceReferences:ObjectJson[]){return this.post('/api/v2/score-reachability-catalogs',{traceReferences})}
  proveScoreReachability(catalogHash:string,targetScore:[number,number],targetDealer:string,initialDealer='east'){return this.post(`/api/v2/score-reachability-catalogs/${encodeURIComponent(catalogHash)}:prove`,{targetScore,targetDealer,initialDealer})}
  foundryCatalogs(options:{kind?:string;after?:string;limit?:number}={}){const q=new URLSearchParams({limit:String(options.limit??100)});if(options.kind)q.set('kind',options.kind);if(options.after)q.set('after',options.after);return this.request(`/api/v2/foundry-catalogs?${q}`)}
  foundryCatalog(catalogHash:string){return this.request(`/api/v2/foundry-catalogs/${encodeURIComponent(catalogHash)}`)}
  createChallengeLocator(payload:ObjectJson){return this.post('/api/v2/challenge-locators',payload)}
  challengeLocator(code:string){return this.request(`/api/v2/challenge-locators/${encodeURIComponent(code)}`)}
  commitChallengeLocatorAttempt(code:string,payload:ObjectJson){return this.post(`/api/v2/challenge-locators/${encodeURIComponent(code)}/attempts`,payload)}
  revokeChallengeLocator(code:string){return this.post(`/api/v2/challenge-locators/${encodeURIComponent(code)}:revoke`,null)}
  // Unified domain compiler and Research Foundry V2.
  domainCapabilities(includePayloads=false,bundleRevisionId?:string){const q=new URLSearchParams({includePayloads:String(includePayloads)});if(bundleRevisionId!==undefined)q.set('bundleRevisionId',bundleRevisionId);return this.request(`/api/v2/domain-capabilities?${q}`)}
  domainCompilation(compilationHash:string,includePayloads=false){const q=new URLSearchParams({includePayloads:String(includePayloads)});return this.request(`/api/v2/domain-compilations/${encodeURIComponent(compilationHash)}?${q}`)}
  domainCompilationCoverage(compilationHash:string){return this.request(`/api/v2/domain-compilations/${encodeURIComponent(compilationHash)}/coverage`)}
  rulePrimitives(bundleRevisionId?:string){return this.request(`/api/v2/rule-primitives${bundleRevisionId===undefined?'':`?bundleRevisionId=${encodeURIComponent(bundleRevisionId)}`}`)}
  validateResearchRecipe(payload:ObjectJson){return this.post('/api/v2/research-recipes:validate',payload)}
  estimateResearchRecipe(payload:ObjectJson){return this.post('/api/v2/research-recipes:estimate',payload)}
  createResearchRecipe(payload:ObjectJson,idempotencyKey?:string){return this.request('/api/v2/research-recipes',{method:'POST',headers:{'Idempotency-Key':this.researchIdempotencyKey(idempotencyKey)},body:JSON.stringify(payload)})}
  researchRecipes(options:ResearchRecipeListOptions={}):Promise<ResearchRecipeListResponse>{const q=new URLSearchParams({limit:String(options.limit??100)});if(options.status)q.set('status',options.status);if(options.cursor!==undefined)q.set('cursor',options.cursor);return this.request(`/api/v2/research-recipes?${q}`)}
  researchRecipe(code:string){return this.request(`/api/v2/research-recipes/${encodeURIComponent(code)}`)}
  retireResearchRecipe(code:string,idempotencyKey?:string){return this.request(`/api/v2/research-recipes/${encodeURIComponent(code)}:retire`,{method:'POST',headers:{'Idempotency-Key':this.researchIdempotencyKey(idempotencyKey)}})}
  createResearchRun(recipeCode:string,idempotencyKey:string){return this.request('/api/v2/research-runs',{method:'POST',headers:{'Idempotency-Key':idempotencyKey},body:JSON.stringify({recipeCode})})}
  researchRuns(options:{status?:'queued'|'running'|'completed'|'failed'|'cancelled';limit?:number}={}){const q=new URLSearchParams({limit:String(options.limit??100)});if(options.status)q.set('status',options.status);return this.request(`/api/v2/research-runs?${q}`)}
  researchRun(runCode:string){return this.request(`/api/v2/research-runs/${encodeURIComponent(runCode)}`)}
  researchRunStats(runCode:string){return this.request(`/api/v2/research-runs/${encodeURIComponent(runCode)}/stats`)}
  researchStats(){return this.request('/api/v2/research-stats')}
  cancelResearchRun(runCode:string){return this.post(`/api/v2/research-runs/${encodeURIComponent(runCode)}:cancel`,null)}
  researchRunEvents(runCode:string){return this.events(`/api/v2/research-runs/${encodeURIComponent(runCode)}/events`)}
  createComparisonCampaign(recipeCode:string,idempotencyKey:string){return this.request('/api/v2/comparison-campaigns',{method:'POST',headers:{'Idempotency-Key':idempotencyKey},body:JSON.stringify({recipeCode})})}
  comparisonCampaigns(limit=100){return this.request(`/api/v2/comparison-campaigns?limit=${limit}`)}
  comparisonCampaign(campaignCode:string){return this.request(`/api/v2/comparison-campaigns/${encodeURIComponent(campaignCode)}`)}
  comparisonCampaignEvents(campaignCode:string){return this.events(`/api/v2/comparison-campaigns/${encodeURIComponent(campaignCode)}/events`)}
  researchResults(options:{runCode?:string;truthLabel?:string;disposition?:string;minimumScore?:number;search?:string;tacticalTag?:string;actingSeat?:string;observerSeat?:string;contract?:string;project?:string;generationLane?:string;scenarioTemplate?:string;scenarioOrigin?:string;scenarioStatus?:string;requestedProofMode?:string;achievedProofMode?:string;proofCompletion?:string;proofVisibility?:string;targetTeam?:string;contractOutcome?:string;projectKind?:string;projectValue?:string;projectState?:string;winningRootActionBand?:string;exactStateAvailable?:boolean;decisionKind?:string;researchBackend?:string;qualificationAuthority?:string;runtimeCompatibilityStatus?:string;configurationBundle?:string;exactStateId?:string;scenarioCategory?:string;variantCollection?:string;createdFrom?:string;createdTo?:string;groupBy?:string;sort?:string;offset?:number;limit?:number}={}){const q=new URLSearchParams({sort:options.sort??'score_desc',offset:String(options.offset??0),limit:String(options.limit??100)});Object.entries(options).forEach(([key,value])=>{if(value!==undefined)q.set(key,String(value))});return this.request(`/api/v2/research-results?${q}`)}
  researchResult(code:string){return this.request(`/api/v2/research-results/${encodeURIComponent(code)}`)}
  researchResultOperatorAudit(code:string):Promise<ResearchOperatorAuditV1Response>{return this.request(`/api/v2/research-results/${encodeURIComponent(code)}/operator-audit`)}
  researchResultDecisions(code:string){return this.request(`/api/v2/research-results/${encodeURIComponent(code)}/decisions`)}
  researchResultPerspectives(code:string,observerSeat:'north'|'west'|'south'|'east'){const q=new URLSearchParams({observerSeat});return this.request(`/api/v2/research-results/${encodeURIComponent(code)}/perspectives?${q}`)}
  researchResultDecision(code:string,ordinal:number){return this.request(`/api/v2/research-results/${encodeURIComponent(code)}/decisions/${ordinal}`)}
  prepareResearchResultRender(code:string,payload:ResearchRenderPrepareRequest|ObjectJson={}){return this.post(`/api/v2/research-results/${encodeURIComponent(code)}:prepare-render`,payload as ObjectJson)}
  prepareResearchDecisionRender(code:string,ordinal:number,payload:ResearchRenderPrepareRequest|ObjectJson={}){return this.post(`/api/v2/research-results/${encodeURIComponent(code)}/decisions/${ordinal}:prepare-render`,payload as ObjectJson)}
  researchResultRenders(code:string){return this.request(`/api/v2/research-results/${encodeURIComponent(code)}/renders`)}
  laffat(options:{search?:string;sourceKind?:string;replayValid?:boolean;qualificationStatus?:string;truthLabel?:string;contract?:string;project?:string;tactic?:string;seat?:string;renderStatus?:string;sort?:string;offset?:number;limit?:number}={}){const q=new URLSearchParams({sort:options.sort??'newest',offset:String(options.offset??0),limit:String(options.limit??100)});Object.entries(options).forEach(([key,value])=>{if(value!==undefined)q.set(key,String(value))});return this.request(`/api/v2/laffat?${q}`)}
  laffa(identifier:string){return this.request(`/api/v2/laffat/${encodeURIComponent(identifier)}`)}
  prepareLaffaRender(identifier:string,payload:ResearchRenderPrepareRequest|ObjectJson={}){return this.post(`/api/v2/laffat/${encodeURIComponent(identifier)}:prepare-render`,payload as ObjectJson)}
  laffaRenders(identifier:string){return this.request(`/api/v2/laffat/${encodeURIComponent(identifier)}/renders`)}
  archivedGame(id:string){return this.request(`/api/v1/history/games/${encodeURIComponent(id)}`)}
  replayPackage(id:string){return this.request(`/api/v1/history/games/${encodeURIComponent(id)}/replay`)}
  auditGame(id:string,payload:ObjectJson={}){return this.post(`/api/v1/history/games/${encodeURIComponent(id)}:audit`,payload)}
  explainDecision(id:string,payload:ObjectJson={}){return this.post(`/api/v1/history/decisions/${encodeURIComponent(id)}:explain`,payload)}
  compareDecision(id:string){return this.request(`/api/v1/history/decisions/${encodeURIComponent(id)}/comparison`)}
  annotateIntent(id:string,payload:ObjectJson){return this.post(`/api/v1/history/decisions/${encodeURIComponent(id)}/intent`,payload)}
  recordDecisionAttempt(id:string,payload:ObjectJson){return this.post(`/api/v1/history/decisions/${encodeURIComponent(id)}/attempts`,payload)}
  decisionDifficulty(id:string){return this.request(`/api/v1/history/decisions/${encodeURIComponent(id)}/difficulty`)}
  recordChallengeAttempt(id:string,payload:ObjectJson){return this.post(`/api/v1/challenges/${encodeURIComponent(id)}/attempts`,payload)}
  challengeDifficulty(id:string){return this.request(`/api/v1/challenges/${encodeURIComponent(id)}/difficulty`)}
  adaptiveChallenge(playerId?:string,recent:string[]=[]){const q=new URLSearchParams({recent:recent.join(',')});if(playerId)q.set('playerId',playerId);return this.request(`/api/v1/adaptive/challenges/next?${q}`)}
  videoRecord(id:string){return this.request(`/api/v1/videos/${encodeURIComponent(id)}`)}
  videoTimeline(id:string){return this.request(`/api/v1/videos/${encodeURIComponent(id)}/timeline`)}
  videoAuditMap(id:string){return this.request(`/api/v1/videos/${encodeURIComponent(id)}/audit-map`)}
  // Canonical standardized shared play: لفة (round) or صكّة (match).
  sharedBoards(limit=100){return this.request(`/api/v1/shared/boards?limit=${limit}`)}
  createSharedBoard(payload:ObjectJson){return this.post('/api/v1/shared/boards',payload)}
  sharedBoard(id:string){return this.request(`/api/v1/shared/boards/${encodeURIComponent(id)}`)}
  verifySharedBoard(id:string){return this.request(`/api/v1/shared/boards/${encodeURIComponent(id)}/verification`)}
  createSharedRoom(payload:ObjectJson){return this.post('/api/v1/shared/rooms',payload)}
  sharedRoom(id:string){return this.request(`/api/v1/shared/rooms/${encodeURIComponent(id)}`)}
  joinSharedRoom(id:string){return this.post(`/api/v1/shared/rooms/${encodeURIComponent(id)}:join`,{})}
  startSharedRun(id:string){return this.post(`/api/v1/shared/rooms/${encodeURIComponent(id)}/runs`,{})}
  sharedRun(id:string){return this.request(`/api/v1/shared/runs/${encodeURIComponent(id)}`)}
  verifySharedRun(id:string){return this.request(`/api/v1/shared/runs/${encodeURIComponent(id)}/verification`)}
  playSharedAction(id:string,action:ObjectJson){return this.post(`/api/v1/shared/runs/${encodeURIComponent(id)}/actions`,action)}
  sharedRunReview(id:string,ensureAudit=false){return this.request(`/api/v1/shared/runs/${encodeURIComponent(id)}/review?ensureAudit=${ensureAudit}`)}
  explainSharedDecision(runId:string,decisionId:string,options:{level?:string;surface?:string;challengeRevealed?:boolean;ensureAudit?:boolean;viewMode?:string}={}){const q=new URLSearchParams({level:options.level??'basic',surface:options.surface??'replay',challengeRevealed:String(options.challengeRevealed??false),ensureAudit:String(options.ensureAudit??false),viewMode:options.viewMode??'observer'});return this.request(`/api/v1/shared/runs/${encodeURIComponent(runId)}/decisions/${encodeURIComponent(decisionId)}/explanation?${q}`)}
  sharedDecisionReviews(runId:string,decisionId:string){return this.request(`/api/v1/shared/runs/${encodeURIComponent(runId)}/decisions/${encodeURIComponent(decisionId)}/reviews`)}
  reviewSharedDecision(runId:string,decisionId:string,payload:ObjectJson){return this.post(`/api/v1/shared/runs/${encodeURIComponent(runId)}/decisions/${encodeURIComponent(decisionId)}/reviews`,payload)}
  shareSharedDecision(runId:string,sequence:number,payload:ObjectJson){return this.post(`/api/v1/shared/runs/${encodeURIComponent(runId)}/actions/${sequence}:share`,payload)}
  sharedLeaderboard(id:string){return this.request(`/api/v1/shared/rooms/${encodeURIComponent(id)}/leaderboard`)}
  compareSharedRuns(runA:string,runB:string){return this.post('/api/v1/shared/comparisons',{runA,runB})}
  sakkaBoards(limit=100){return this.request(`/api/v1/sakka/boards?limit=${limit}`)}
  createSakkaBoard(payload:ObjectJson){return this.post('/api/v1/sakka/boards',payload)}
  sakkaBoard(id:string){return this.request(`/api/v1/sakka/boards/${encodeURIComponent(id)}`)}
  verifySakkaBoard(id:string){return this.request(`/api/v1/sakka/boards/${encodeURIComponent(id)}/verification`)}
  createSakkaRoom(payload:ObjectJson){return this.post('/api/v1/sakka/rooms',payload)}
  sakkaRoom(id:string){return this.request(`/api/v1/sakka/rooms/${encodeURIComponent(id)}`)}
  joinSakkaRoom(id:string){return this.post(`/api/v1/sakka/rooms/${encodeURIComponent(id)}:join`,{})}
  startSakkaRun(id:string){return this.post(`/api/v1/sakka/rooms/${encodeURIComponent(id)}/runs`,{})}
  sakkaRun(id:string){return this.request(`/api/v1/sakka/runs/${encodeURIComponent(id)}`)}
  verifySakkaRun(id:string){return this.request(`/api/v1/sakka/runs/${encodeURIComponent(id)}/verification`)}
  playSakkaAction(id:string,action:ObjectJson){return this.post(`/api/v1/sakka/runs/${encodeURIComponent(id)}/actions`,action)}
  auditSakkaRun(id:string,payload:ObjectJson={}){return this.post(`/api/v1/sakka/runs/${encodeURIComponent(id)}:audit`,payload)}
  shareSakkaDecision(id:string,sequence:number,payload:ObjectJson){return this.post(`/api/v1/sakka/runs/${encodeURIComponent(id)}/actions/${sequence}:share`,payload)}
  sakkaFeedback(id:string,sequence:number,payload:ObjectJson){return this.post(`/api/v1/sakka/runs/${encodeURIComponent(id)}/actions/${sequence}/feedback`,payload)}
  sakkaLeaderboard(id:string){return this.request(`/api/v1/sakka/rooms/${encodeURIComponent(id)}/leaderboard`)}
  compareSakkaRuns(runA:string,runB:string){return this.post('/api/v1/sakka/comparisons',{runA,runB})}
  hitlProfile(){return this.request('/api/v1/hitl/profile')}
  hitlCases(limit=100){return this.request(`/api/v1/hitl/cases?limit=${limit}`)}
  createHitlCase(decisionId:string,highImpact=false,correctionType='analysis_defect'){return this.post('/api/v1/hitl/cases',{decisionId,correctionType,highImpact})}
  hitlCase(id:string,blind=false){return this.request(`/api/v1/hitl/cases/${encodeURIComponent(id)}?blind=${blind}`)}
  assignHitlReview(id:string,payload:ObjectJson={}){return this.post(`/api/v1/hitl/cases/${encodeURIComponent(id)}/assignments`,payload)}
  submitHitlReview(id:string,payload:ObjectJson){return this.post(`/api/v1/hitl/cases/${encodeURIComponent(id)}/submissions`,payload)}
  computeHitlConsensus(id:string){return this.post(`/api/v1/hitl/cases/${encodeURIComponent(id)}/consensus`,{})}
  hitlConsent(){return this.request('/api/v1/hitl/consent')}
  setHitlConsent(payload:ObjectJson){return this.request('/api/v1/hitl/consent',{method:'PUT',body:JSON.stringify(payload)})}
  createHitlTrainingCandidate(id:string){return this.post(`/api/v1/hitl/cases/${encodeURIComponent(id)}/training-candidates`,{})}
  evaluateHitlCandidate(id:string,payload:ObjectJson){return this.post(`/api/v1/hitl/training-candidates/${encodeURIComponent(id)}/evaluations`,payload)}
  renderHitlReviewClip(id:string,payload:ObjectJson={}){return this.post(`/api/v1/hitl/cases/${encodeURIComponent(id)}/review-clip`,payload)}
  hitlImprovements(){return this.request('/api/v1/hitl/improvements')}
  createHitlImprovement(payload:ObjectJson){return this.post('/api/v1/hitl/improvements',payload)}
  evaluateHitlImprovement(id:string,payload:ObjectJson){return this.post(`/api/v1/hitl/improvements/${encodeURIComponent(id)}/evaluations`,payload)}
  // Versioned row-based configuration control plane V19.
  configurationCapabilities(){return this.request('/api/v1/configuration-registry/profile')}
  configurationProfiles(params:{family?:string;profileId?:string;status?:string;tenantId?:string;scope?:string;limit?:number}={}){const q=new URLSearchParams();Object.entries(params).forEach(([k,v])=>{if(v!==undefined)q.set(k,String(v))});return this.request(`/api/v1/configuration-registry/profiles?${q}`)}
  createConfigurationProfile(family:string,payload:ObjectJson,options:{scope?:string;tenantId?:string;supersedesRevisionId?:string;publish?:boolean}={}){return this.post('/api/v1/configuration-registry/profiles',{family,payload,scope:options.scope??'global',tenantId:options.tenantId??null,supersedesRevisionId:options.supersedesRevisionId??null,publish:options.publish??false})}
  publishConfigurationProfile(revisionId:string,expectedEtag:number){return this.post(`/api/v1/configuration-registry/profiles/${encodeURIComponent(revisionId)}:publish`,{expectedEtag})}
  rejectConfigurationProfile(revisionId:string,expectedEtag:number,reason:string){return this.post(`/api/v1/configuration-registry/profiles/${encodeURIComponent(revisionId)}:reject`,{expectedEtag,reason})}
  retireConfigurationProfile(revisionId:string,expectedEtag:number,reason:string){return this.post(`/api/v1/configuration-registry/profiles/${encodeURIComponent(revisionId)}:retire`,{expectedEtag,reason})}
  resolveConfigurationProfile(family:string,profileId:string,tenantId?:string){const q=new URLSearchParams({family,profileId});if(tenantId)q.set('tenantId',tenantId);return this.request(`/api/v1/configuration-registry/profiles:resolve?${q}`)}
  compareConfigurationProfiles(left:string,right:string){const q=new URLSearchParams({left,right});return this.request(`/api/v1/configuration-registry/profiles:compare?${q}`)}
  configurationBundles(params:{bundleId?:string;status?:string;tenantId?:string;scope?:string;limit?:number}={}){const q=new URLSearchParams();Object.entries(params).forEach(([k,v])=>{if(v!==undefined)q.set(k,String(v))});return this.request(`/api/v1/configuration-registry/bundles?${q}`)}
  createConfigurationBundle(bundleId:string,version:string,componentRevisions:{[key:string]:string},options:{mode?:string;scope?:string;tenantId?:string;supersedesRevisionId?:string;publish?:boolean;allowDraftComponents?:boolean}={}){return this.post('/api/v1/configuration-registry/bundles',{bundleId,version,componentRevisions,mode:options.mode??'full',scope:options.scope??'global',tenantId:options.tenantId??null,supersedesRevisionId:options.supersedesRevisionId??null,publish:options.publish??false,allowDraftComponents:options.allowDraftComponents??false})}
  publishConfigurationBundle(revisionId:string,expectedEtag:number){return this.post(`/api/v1/configuration-registry/bundles/${encodeURIComponent(revisionId)}:publish`,{expectedEtag})}
  publishConfiguration(revisionId:string,expectedEtag:number){return this.publishConfigurationBundle(revisionId,expectedEtag)}
  rejectConfigurationBundle(revisionId:string,expectedEtag:number,reason:string){return this.post(`/api/v1/configuration-registry/bundles/${encodeURIComponent(revisionId)}:reject`,{expectedEtag,reason})}
  retireConfigurationBundle(revisionId:string,expectedEtag:number,reason:string){return this.post(`/api/v1/configuration-registry/bundles/${encodeURIComponent(revisionId)}:retire`,{expectedEtag,reason})}
  resolveConfigurationBundle(options:{bundleId?:string;revisionId?:string;tenantId?:string;allowDraft?:boolean}={}){if(options.revisionId){const q=new URLSearchParams({allowDraft:String(options.allowDraft??false)});return this.request(`/api/v1/configuration-registry/bundles/${encodeURIComponent(options.revisionId)}:resolve?${q}`)}const q=new URLSearchParams();if(options.bundleId)q.set('bundleId',options.bundleId);if(options.tenantId)q.set('tenantId',options.tenantId);return this.request(`/api/v1/configuration-registry/resolve?${q}`)}
  resolveConfiguration(options:{bundleId?:string;revisionId?:string;tenantId?:string;allowDraft?:boolean}={}){return this.resolveConfigurationBundle(options)}
  platformCapabilities(){return this.request('/api/v1/platform/capabilities')}
  performanceProfile(){return this.request('/api/v1/performance/profile')}
  runPerformanceQuick(){return this.post('/api/v1/performance/quick',{})}
  private researchIdempotencyKey(value?:string){return value??crypto.randomUUID()}
  private post(path:string,payload:ObjectJson|null){return this.request(path,{method:'POST',body:payload===null?undefined:JSON.stringify(payload)})}
  private async request(path:string,init:RequestInit={}):Promise<any>{
    const headers=new Headers(init.headers);headers.set('accept','application/json');
    if(init.body)headers.set('content-type','application/json');if(this.apiKey)headers.set('x-api-key',this.apiKey);if(this.trustedKey)headers.set('x-baloot-trusted-key',this.trustedKey);if(this.subjectId)headers.set('x-baloot-subject-id',this.subjectId);if(this.subjectName)headers.set('x-baloot-subject-name',this.subjectName);if(this.accessToken)headers.set('authorization',`Bearer ${this.accessToken}`);
    const response=await this.fetchImpl(this.baseUrl+path,{...init,headers,credentials:this.credentials});if(!response.ok)throw new Error(`${response.status}: ${await response.text()}`);if(response.status===204)return null;return response.json();
  }
  private async *events(path:string):AsyncGenerator<ObjectJson>{
    const headers=new Headers();headers.set('accept','text/event-stream');if(this.apiKey)headers.set('x-api-key',this.apiKey);if(this.trustedKey)headers.set('x-baloot-trusted-key',this.trustedKey);if(this.subjectId)headers.set('x-baloot-subject-id',this.subjectId);if(this.subjectName)headers.set('x-baloot-subject-name',this.subjectName);if(this.accessToken)headers.set('authorization',`Bearer ${this.accessToken}`);
    const response=await this.fetchImpl(this.baseUrl+path,{headers,credentials:this.credentials});if(!response.ok)throw new Error(`${response.status}: ${await response.text()}`);if(!response.body)return;
    const reader=response.body.getReader();const decoder=new TextDecoder();let buffered='';
    while(true){const {done,value}=await reader.read();buffered+=decoder.decode(value,{stream:!done});const lines=buffered.split(/\r?\n/);buffered=lines.pop()??'';for(const line of lines){if(line.startsWith('data: '))yield JSON.parse(line.slice(6)) as ObjectJson}if(done)break}
    if(buffered.startsWith('data: '))yield JSON.parse(buffered.slice(6)) as ObjectJson;
  }
}
