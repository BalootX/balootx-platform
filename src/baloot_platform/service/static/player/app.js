(() => {
  const state = { identity:null, csrf:null, room:null, board:null, run:null, game:null, authMode:'login', socket:null, pollTimer:null, eventCursor:0, leaderboard:[], selectedPlay:null, drag:null, busy:false, suppressClickUntil:0 };
  const $ = (id) => document.getElementById(id);
  const governedCopy = JSON.parse($('player-copy')?.textContent || '{}');
  function copy(key, values={}){
    const template=governedCopy[key];
    if(typeof template!=='string'||!template.trim())throw new Error(`Missing governed player copy: ${key}`);
    return template.replace(/\{([a-zA-Z0-9_]+)\}/g,(match,name)=>Object.hasOwn(values,name)?String(values[name]):match);
  }
  const views = [...document.querySelectorAll('.view')];
  const seatAr = {north:copy('player_seat_top'),south:copy('player_seat_bottom'),west:copy('player_seat_left'),east:copy('player_seat_right')};
  const seatOrder = ['south','west','north','east'];
  const canonicalSeats = ['north','west','south','east'];
  const contractAr = {sun:copy('contract_sun'),hokum:copy('contract_hokum'),ashkal:copy('contract_ashkal')};
  const suitAr = {spades:'♠',hearts:'♥',diamonds:'♦',clubs:'♣'};
  const rankCodes = {ace:'A',king:'K',queen:'Q',jack:'J',ten:'10',nine:'9',eight:'8',seven:'7'};
  const internalSuitNames = {S:'spades',H:'hearts',D:'diamonds',C:'clubs'};
  const internalRankNames = {A:'ace',K:'king',Q:'queen',J:'jack','10':'ten','9':'nine','8':'eight','7':'seven'};

  function show(name){ views.forEach(v=>v.classList.toggle('active',v.id===`${name}-view`)); document.body.classList.toggle('game-active',name==='game'); window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'}); }
  function toast(msg){ const el=$('toast'); el.textContent=msg; el.classList.add('show'); clearTimeout(el._t); el._t=setTimeout(()=>el.classList.remove('show'),2200); }
  const requestModule = import('/shared-assets/request.js');
  async function api(path, options={}){
    const headers = {'content-type':'application/json',...(options.headers||{})};
    if(state.csrf && !['GET','HEAD'].includes((options.method||'GET').toUpperCase())) headers['x-csrf-token']=state.csrf;
    const {fetchWithDeadline,isDeadlineError}=await requestModule;
    let res,text; try{res=await fetchWithDeadline(path,{credentials:'include',...options,headers}); text=await res.text()}catch(error){throw isDeadlineError(error)?new Error(copy('request_timeout')):error}
    let data=null; try{data=text?JSON.parse(text):null}catch{data={detail:text}}
    if(!res.ok) throw new Error(copy('request_error',{status:res.status})); return data;
  }
  async function ensureIdentity(){
    let me=await api('/api/v1/auth/me');
    if(!me.identity){ me=await api('/api/v1/auth/guest',{method:'POST',body:'{}'}); }
    state.identity=me.identity; state.csrf=me.csrfToken; $('identity-name').textContent=state.identity.displayName;
    $('auth-button').textContent=state.identity.guest?copy('player_optional_login'):state.identity.displayName;
  }
  function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
  function rank(card){return rankCodes[card?.rank]||card?.rank||''}
  function red(card){return ['hearts','diamonds'].includes(card?.suit)}
  function cardLabel(card){return card?.label||`${rank(card)}${suitAr[card?.suit]||''}`}
  function cardAsset(card){const r=rankCodes[card?.rank],s=Object.hasOwn(suitAr,card?.suit)?card.suit:null;return r&&s?`/card-assets/${encodeURIComponent(r)}_${encodeURIComponent(s)}.svg`:''}
  function cardFace(card){const source=cardAsset(card);return `<span class="card-face ${red(card)?'red':''}"><span class="card-fallback" aria-hidden="true"><b>${escapeHtml(rank(card))}</b><i>${escapeHtml(suitAr[card?.suit]||'')}</i></span>${source?`<img src="${source}" alt="${escapeHtml(cardLabel(card))}" draggable="false">`:''}</span>`}
  function cardView(card,legalCodes=[],index=0,count=1){const legal=legalCodes.includes(card.code),distance=index-(count-1)/2,angle=distance*2.2,lift=Math.abs(distance)*2.1;return `<button class="playing-card ${legal?'legal':''}" style="--fan-angle:${angle.toFixed(2)}deg;--fan-y:${lift.toFixed(2)}px;--card-z:${index+1}" data-card="${escapeHtml(card.code)}" ${legal?'':'disabled'} aria-label="${escapeHtml(cardLabel(card))}" aria-pressed="false">${cardFace(card)}</button>`}
  function relativePositions(observer){const i=Math.max(0,seatOrder.indexOf(observer));return {bottom:observer,left:seatOrder[(i+1)%4],top:seatOrder[(i+2)%4],right:seatOrder[(i+3)%4]}}
  function relativeName(seat,positions){return Object.entries(positions).find(([,value])=>value===seat)?.[0]||'top'}
  function handCount(hand){return Array.isArray(hand)?hand.length:Number(hand?.count??0)}
  function backStack(count){const visible=Math.min(5,Math.max(0,Number(count)||0));return `<span class="back-stack" aria-hidden="true">${Array.from({length:visible},(_,index)=>{const offset=index-(visible-1)/2;return `<img src="/challenge-video-assets/card-back.svg" alt="" draggable="false" style="--offset:${offset};--layer:${index+1}">`}).join('')}</span>`}
  function purchaseRound(game){const history=Array.isArray(game.biddingHistory)?game.biddingHistory:[];const bid=[...history].reverse().find(item=>item&&(item.action===game.contract||item.contract===game.contract)&&item.round);return Number(bid?.round||1)}
  function purchaseLabel(game){if(!game.contract)return copy('player_purchase_stage');const contract=contractAr[game.contract]||game.contract;const round=copy(purchaseRound(game)===2?'purchase_second':'purchase_first');const trump=game.contract==='hokum'?(suitAr[game.trumpSuit]||''):'';return copy(game.contract==='hokum'?'purchase_hokum_summary':'purchase_plain_summary',{contract,round,trump})}
  function renderMembers(items=[]){const player=copy('role_player');$('member-count').textContent=items.length;$('members').innerHTML=items.map(m=>`<div class="member"><span class="avatar">${escapeHtml((m.display_name||m.displayName||player).slice(0,1))}</span><span>${escapeHtml(m.display_name||m.displayName||player)}</span></div>`).join('')||`<p class="empty">${escapeHtml(copy('player_waiting_participants'))}</p>`;}
  function renderLeaderboard(items=[]){
    state.leaderboard=items;
    const el=$('leaderboard'), compare=$('compare-top'), panel=$('comparison-panel');
    compare.hidden=items.length<2; if(items.length<2)panel.hidden=true;
    if(!items.length){el.className='leaderboard empty';el.textContent=copy('player_no_results');return}
    el.className='leaderboard';
    el.innerHTML=items.map(i=>`<div class="leader-row" data-run="${i.runId}"><b class="rank">#${i.rank}</b><span>${escapeHtml(i.displayName||copy('role_player'))}</span><strong>${i.result?.ourMatchPoints??0} — ${i.result?.opponentMatchPoints??0}</strong></div>`).join('');
  }
  function renderComparison(value){
    const panel=$('comparison-panel'), a=value.runA, b=value.runB, winner=value.winnerRunId;
    const player=(item,label)=>`<article class="comparison-player ${winner===item.runId?'winner':''}"><small>${escapeHtml(label)}</small><b>${escapeHtml(item.subjectId===state.identity?.subjectId?copy('player_role_you'):copy('role_player'))}</b><strong>${item.result.ourMatchPoints??0} — ${item.result.opponentMatchPoints??0}</strong><span>${escapeHtml(item.result.matchWon?copy('player_won_match'):copy('player_did_not_win'))}</span></article>`;
    const first=value.firstDivergence; const choiceA=first?.runA?.action?.card||first?.runA?.action?.action||'—', choiceB=first?.runB?.action?.card||first?.runB?.action?.action||'—';
    panel.innerHTML=`<h4>${escapeHtml(copy('player_compare_title'))}</h4><div class="comparison-grid">${player(a,copy('player_path_first',{choice:choiceA}))}<div class="comparison-vs">${escapeHtml(copy('player_versus'))}</div>${player(b,copy('player_path_second',{choice:choiceB}))}</div><p class="comparison-note">${escapeHtml(copy('player_compare_verified'))}</p>`;
    panel.hidden=false;
  }
  async function compareTopRuns(){
    if(state.leaderboard.length<2)return;
    try{const value=await api('/api/v1/sakka/comparisons',{method:'POST',body:JSON.stringify({runA:state.leaderboard[0].runId,runB:state.leaderboard[1].runId})});renderComparison(value)}catch(err){toast(err.message)}
  }
  function connectRoomStream(code){
    if(state.socket){state.socket.close();state.socket=null}clearTimeout(state.pollTimer);state.pollTimer=null;
    if(state.room?.mode!=='live'){
      const poll=async()=>{
        if(state.room?.roomCode!==code||document.body.classList.contains('game-active'))return;
        try{const payload=await api(`/api/v1/sakka/rooms/${encodeURIComponent(code)}`);renderMembers(payload.members);renderLeaderboard(payload.leaderboard)}catch{}
        if(state.room?.roomCode===code&&!document.body.classList.contains('game-active'))state.pollTimer=setTimeout(poll,5000);
      };
      state.pollTimer=setTimeout(poll,5000);return;
    }
    const protocol=location.protocol==='https:'?'wss':'ws';
    const ws=new WebSocket(`${protocol}://${location.host}/api/v1/sakka/rooms/${encodeURIComponent(code)}/stream?afterSequence=${state.eventCursor}`);
    state.socket=ws;
    ws.onmessage=async event=>{try{const message=JSON.parse(event.data);if(message.cursor)state.eventCursor=message.cursor;if(message.type==='events'&&state.room){const important=message.events.some(i=>['member_joined','run_completed','run_audited'].includes(i.eventType));if(important){const payload=await api(`/api/v1/sakka/rooms/${encodeURIComponent(state.room.roomCode)}`);renderMembers(payload.members);renderLeaderboard(payload.leaderboard)}}}catch{}};
    ws.onclose=()=>{if(state.room?.roomCode===code&&state.room?.mode==='live')setTimeout(()=>connectRoomStream(code),1500)};
  }
  async function renderLobby(payload){state.room=payload.room;state.board=payload.board;$('share-code').textContent=state.room.roomCode;$('lobby-title').textContent=state.board.label||copy('player_default_sakka_title');$('board-meta').innerHTML=`<span>${escapeHtml(copy(state.board.scope==='match'?'player_scope_match':'player_scope_round'))}</span><span>${escapeHtml(copy('player_seat_summary',{seat:seatAr[state.board.observerSeat]}))}</span><span>${escapeHtml(copy('player_start_score',{score:state.board.startingMatchScore.join(' — ')}))}</span>`;renderMembers(payload.members);renderLeaderboard(payload.leaderboard);show('lobby');history.replaceState({},'',`/play?code=${state.room.roomCode}`);if(!state.socket||state.socket.readyState>1)connectRoomStream(state.room.roomCode);const line=$('board-verification');try{const check=await api(`/api/v1/sakka/boards/${encodeURIComponent(state.board.boardId)}/verification`);line.className=`verification-line ${check.valid?'valid':'invalid'}`;line.textContent=copy(check.valid?'player_board_verified':'player_board_invalid')}catch{line.className='verification-line invalid';line.textContent=copy('player_board_invalid')}}
  function playerName(payload,seat){const names=payload.playerNames&&typeof payload.playerNames==='object'?payload.playerNames:{};return String(names[seat]||'').trim()}
  function renderSeats(payload){
    const game=payload.state,positions=relativePositions(payload.board.observerSeat);
    const counts=Object.fromEntries(canonicalSeats.map(seat=>[seat,handCount(game.hands?.[seat])]));
    $('match-seats').innerHTML=Object.entries(positions).map(([position,seat])=>{
      const name=playerName(payload,seat),tags=[];
      if(seat===game.dealer)tags.push(`<span class="dealer-tag"><b>◆</b> ${escapeHtml(copy('role_dealer'))}</span>`);
      if(seat===game.buyer)tags.push(`<span class="buyer-tag"><b>${escapeHtml(copy('role_buyer'))}</b><em>${escapeHtml(purchaseLabel(game))}</em></span>`);
      return `<section class="seat seat-${position} ${seat===(game.currentPlayer||game.bidding?.actor)?'active':''}" data-seat="${seat}" aria-label="${escapeHtml(name||copy('role_player'))}">
        ${position==='bottom'?'':backStack(counts[seat])}
        <span class="seat-medallion"><b>${counts[seat]}</b></span>
        ${name?`<span class="seat-name">${escapeHtml(name)}</span>`:''}
        <span class="seat-tags">${tags.join('')}</span>
      </section>`;
    }).join('');
  }
  function tableCard(play,positions,entering=false){const card=play.card||play,position=relativeName(play.seat,positions);return `<span class="table-card ${position} ${entering?'entering':''}" data-play-seat="${escapeHtml(play.seat)}" data-card-code="${escapeHtml(card.code||'')}">${cardFace(card)}</span>`}
  function selectPlay(code){
    const action=(state.currentLegalPlays||[]).find(item=>item.card===code)||null;
    state.selectedPlay=action;
    document.querySelectorAll('#hand [data-card]').forEach(button=>{const selected=button.dataset.card===code;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',selected?'true':'false')});
    $('play-selected').disabled=!action||state.busy;
  }
  function clearDrag(button){
    if(button){button.style.removeProperty('--drag-x');button.style.removeProperty('--drag-y');button.classList.remove('dragging','drag-ready')}
    $('match-table').classList.remove('drop-ready');$('match-drag-hint').textContent=copy('drag_hint');state.drag=null;
  }
  function onCardPointerDown(event){if(state.busy||event.button>0)return;const button=event.currentTarget,table=$('match-table').getBoundingClientRect(),threshold=Math.min(88,Math.max(54,event.clientY-table.bottom+18));state.drag={button,pointerId:event.pointerId,startX:event.clientX,startY:event.clientY,x:0,y:0,threshold};button.setPointerCapture?.(event.pointerId);button.classList.add('dragging')}
  function onCardPointerMove(event){const drag=state.drag;if(!drag||drag.pointerId!==event.pointerId)return;event.preventDefault();drag.x=Math.max(-72,Math.min(72,event.clientX-drag.startX));drag.y=Math.max(-190,Math.min(12,event.clientY-drag.startY));drag.button.style.setProperty('--drag-x',`${drag.x*.3}px`);drag.button.style.setProperty('--drag-y',`${drag.y}px`);const table=$('match-table').getBoundingClientRect(),insideTable=event.clientY>=table.top&&event.clientY<=table.bottom,ready=drag.y<=-drag.threshold||insideTable;drag.button.classList.toggle('drag-ready',ready);$('match-table').classList.toggle('drop-ready',ready);$('match-drag-hint').textContent=copy(ready?'drag_release':'drag_up')}
  function onCardPointerCancel(event){if(state.drag?.pointerId===event.pointerId)clearDrag(state.drag.button)}
  function onCardPointerUp(event){const drag=state.drag;if(!drag||drag.pointerId!==event.pointerId)return;const table=$('match-table').getBoundingClientRect(),insideTable=event.clientY>=table.top&&event.clientY<=table.bottom,shouldPlay=drag.y<=-drag.threshold||insideTable,button=drag.button,code=button.dataset.card,sourceRect=shouldPlay?button.getBoundingClientRect():null;state.suppressClickUntil=performance.now()+350;if(shouldPlay)button.style.opacity='0';clearDrag(button);selectPlay(code);if(shouldPlay){navigator.vibrate?.(12);act({kind:'play',card:code},{sourceButton:button,sourceRect,swiped:true})}}
  function bindCard(button){button.addEventListener('pointerdown',onCardPointerDown);button.addEventListener('pointermove',onCardPointerMove);button.addEventListener('pointerup',onCardPointerUp);button.addEventListener('pointercancel',onCardPointerCancel);button.addEventListener('click',()=>{if(performance.now()<state.suppressClickUntil)return;selectPlay(button.dataset.card)})}
  function renderGame(payload){
    clearTimeout(state.pollTimer);state.pollTimer=null;
    state.run=payload.run;state.game=payload.state;state.room=payload.room;state.board=payload.board;state.selectedPlay=null;
    const game=state.game,observer=state.board.observerSeat,ours=['north','south'].includes(observer)?0:1,score=game.matchScore||[0,0];
    $('round-number').textContent=game.roundNumber||1;$('match-us-score').textContent=score[ours]??0;$('match-them-score').textContent=score[1-ours]??0;$('match-room-code').textContent=state.room.roomCode||'—';
    $('game-contract').textContent=purchaseLabel(game);const buyerName=game.buyer?playerName(payload,game.buyer):'';$('buyer-label').textContent=game.buyer?(buyerName?copy('buyer_with_name',{buyer:buyerName}):copy('role_buyer')):copy('player_waiting_purchase');
    $('purchase-label').textContent=game.purchaseCard?copy('player_purchase_card',{card:game.purchaseCard.label}):'';const actor=game.currentPlayer||game.bidding?.actor;$('turn-label').textContent=actor===observer?copy('player_your_turn'):copy('player_turn_named',{player:playerName(payload,actor)||seatAr[actor]||copy('role_player')});
    renderSeats(payload);const positions=relativePositions(observer),trick=game.currentTrick?.plays||[];$('trick-cards').innerHTML=trick.map(play=>tableCard(play,positions)).join('');
    const hand=Array.isArray(game.hands?.[observer])?game.hands[observer]:[];state.currentLegalPlays=(payload.legalActions||[]).filter(action=>action.kind==='play');const legalCodes=state.currentLegalPlays.map(action=>action.card);$('game-view').classList.toggle('play-phase',legalCodes.length>0);$('hand').dataset.count=String(hand.length);$('hand').innerHTML=hand.map((card,index)=>cardView(card,legalCodes,index,hand.length)).join('');document.querySelectorAll('#hand .playing-card.legal').forEach(bindCard);
    const bids=(payload.legalActions||[]).filter(action=>action.kind==='bid');$('legal-actions').innerHTML=bids.map(action=>`<button class="bid-choice" data-bid='${JSON.stringify(action)}'>${escapeHtml(action.action==='pass'?copy('player_bid_pass'):action.action==='sun'?copy('player_bid_sun'):action.action==='ashkal'?copy('player_bid_ashkal'):copy('player_bid_hokum',{suit:suitAr[action.trumpSuit]||''}))}</button>`).join('');document.querySelectorAll('.bid-choice').forEach(button=>button.addEventListener('click',()=>act(JSON.parse(button.dataset.bid))));
    $('decision-title').textContent=copy(bids.length?'player_decision_buy':legalCodes.length?'player_decision_play':'player_playing');$('match-drag-hint').hidden=!legalCodes.length;$('play-selected').disabled=true;$('play-selected').classList.remove('busy');show('game');
  }
  const wait=milliseconds=>new Promise(resolve=>setTimeout(resolve,milliseconds));
  async function animateCardToTable(sourceButton,sourceRect=null){
    if(!sourceButton)return;const source=sourceRect||sourceButton.getBoundingClientRect(),target=$('match-play-target').getBoundingClientRect(),clone=sourceButton.cloneNode(true);clone.classList.add('card-flight');clone.removeAttribute('data-card');clone.style.cssText=`left:${source.left}px;top:${source.top}px;width:${source.width}px;height:${source.height}px`;document.body.appendChild(clone);sourceButton.style.opacity='0';const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches,dx=target.left+target.width/2-(source.left+source.width/2),dy=target.top+target.height/2-(source.top+source.height/2);
    try{await clone.animate(reduced?[{opacity:1},{opacity:0}]:[{transform:'translate3d(0,0,0) scale(1)',opacity:1},{transform:`translate3d(${dx}px,${dy}px,0) scale(.76) rotate(-2deg)`,opacity:.96}],{duration:reduced?80:360,easing:'cubic-bezier(.2,.82,.24,1)',fill:'forwards'}).finished}catch{}clone.remove();
  }
  async function collectTrick(winner){
    const cards=[...$('trick-cards').querySelectorAll('.table-card')],seat=document.querySelector(`#match-seats .seat[data-seat="${winner}"]`);if(!cards.length||!seat)return;seat.classList.add('winner');const target=seat.getBoundingClientRect(),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches,flights=cards.map(card=>{const rect=card.getBoundingClientRect(),clone=card.cloneNode(true);clone.className='table-card-flight';clone.style.cssText=`position:fixed;z-index:90;left:${rect.left}px;top:${rect.top}px;width:${rect.width}px;height:${rect.height}px`;document.body.appendChild(clone);card.style.opacity='0';const dx=target.left+target.width/2-(rect.left+rect.width/2),dy=target.top+target.height/2-(rect.top+rect.height/2);return clone.animate(reduced?[{opacity:1},{opacity:0}]:[{transform:'translate3d(0,0,0) scale(1)',opacity:1},{transform:`translate3d(${dx}px,${dy}px,0) scale(.38)`,opacity:.2}],{duration:reduced?80:400,easing:'cubic-bezier(.36,.02,.22,1)',fill:'forwards'}).finished.catch(()=>{}).finally(()=>clone.remove())});await Promise.all(flights);$('trick-cards').innerHTML='';await wait(reduced?40:170);seat.classList.remove('winner');
  }
  async function animateNewEvents(payload,startSequence,skipCode){
    const positions=relativePositions(payload.board.observerSeat),events=(payload.state.events||[]).filter(event=>Number(event.sequence)>=startSequence),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;let skipped=false;
    for(const event of events){
      if(event.type==='card_played'&&event.card){const skip=!skipped&&skipCode&&event.card.code===skipCode;if(skip)skipped=true;$('trick-cards').insertAdjacentHTML('beforeend',tableCard(event,positions,!skip));await wait(reduced?35:(skip?40:230))}
      if(event.type==='trick_completed')await collectTrick(event.winner);
    }
  }
  async function act(action,{sourceButton=null,sourceRect=null,swiped=false}={}){
    if(state.busy)return;state.busy=true;$('play-selected').disabled=true;$('play-selected').classList.add('busy');const startSequence=(state.game?.events||[]).length;
    try{if(swiped)await animateCardToTable(sourceButton,sourceRect);const payload=await api(`/api/v1/sakka/runs/${state.run.runId}/actions`,{method:'POST',body:JSON.stringify(action)});await animateNewEvents(payload,startSequence,swiped?action.card:null);if(payload.run.status==='completed')renderResult(payload);else renderGame(payload)}catch(error){if(sourceButton)sourceButton.style.opacity='';toast(error.message)}finally{state.busy=false;$('play-selected').classList.remove('busy')}
  }
  function renderResult(payload){state.run=payload.run;const r=payload.run.result||{};$('result-score').textContent=`${r.ourMatchPoints??0} — ${r.opponentMatchPoints??0}`;$('result-title').textContent=copy(r.matchWon?'player_result_excellent':'player_result_compared');$('result-icon').textContent=r.matchWon?'★':'✓';$('result-copy').textContent=copy('player_result_body',{rounds:r.roundsCompleted||1});$('result-audit').hidden=true;$('run-verification').hidden=true;$('analyze-run').disabled=false;show('result')}
  function cardFromInternalCode(value){const match=/^([SHDC])(10|[AKQJ987])$/.exec(String(value||'').toUpperCase());if(!match)return null;const suit=internalSuitNames[match[1]],rank=internalRankNames[match[2]];return suit&&rank?{code:String(value).toUpperCase(),suit,rank,label:`${match[2]}${suitAr[suit]}`} : null}
  function actionCard(action){if(!action)return null;const value=typeof action==='string'?action:action.card||action.code;return cardFromInternalCode(value)}
  function actionText(action){if(!action)return '—';const card=actionCard(action);if(card)return card.label;if(typeof action==='string')return action;const name=action.action;if(name==='pass')return copy('player_bid_pass');if(name==='sun')return copy('player_bid_sun');if(name==='ashkal')return copy('player_bid_ashkal');if(name==='hokum')return copy('player_bid_hokum',{suit:suitAr[action.trumpSuit]||''});return name||JSON.stringify(action)}
  function actionVisual(action){const card=actionCard(action);return card?`<span class="audit-card-face" aria-label="${escapeHtml(card.label)}">${cardFace(card)}</span>`:`<span class="audit-action-text">${escapeHtml(actionText(action))}</span>`}
  function primaryDecisionReason(reasons){return (reasons||[]).find(reason=>!['HUMAN_INTENT_UNKNOWN','AGENT_POLICY_TRACE'].includes(reason.code))||(reasons||[])[0]}
  function renderAudit(report){
    $('result-audit').hidden=false;
    $('audit-summary').innerHTML=`<div><span>${escapeHtml(copy('player_audit_quality'))}</span><strong>${Math.round(report.averageDecisionQuality??100)}%</strong></div><div><span>${escapeHtml(copy('player_audit_regret'))}</span><strong>${Number(report.totalExpectedRegret||0).toFixed(1)}</strong></div><div><span>${escapeHtml(copy('player_audit_critical'))}</span><strong>${report.catastrophicCount||0}</strong></div>`;
    $('audit-decisions').innerHTML=(report.decisions||[]).map((d,index)=>{const quality=d.qualityLabel==='excellent'?copy('player_quality_excellent'):d.qualityLabel==='good'?copy('player_quality_good'):d.qualityLabel==='mistake'?copy('player_quality_mistake'):d.qualityLabel==='blunder'?copy('player_quality_blunder'):d.forced?copy('player_quality_forced'):copy('player_quality_analysis'),reason=primaryDecisionReason(d.reasons);return `<article class="audit-decision"><div class="audit-head"><span>#${index+1} · ${escapeHtml(copy(d.decisionKind==='bid'?'player_decision_bid':'player_decision_card'))}</span><b class="quality ${escapeHtml(d.qualityLabel||'')}">${escapeHtml(quality)}</b></div><div class="audit-actions"><div><small>${escapeHtml(copy('player_your_play'))}</small><strong>${actionVisual(d.actualAction)}</strong></div><div><small>${escapeHtml(copy('player_best_play'))}</small><strong>${actionVisual(d.bestAction)}</strong></div><div><small>${escapeHtml(copy('player_win_difference'))}</small><strong>${Math.round((d.winProbabilityDrop||0)*100)}%</strong></div></div><p>${escapeHtml(reason?.detailAr||copy('player_analysis_fallback'))}</p><div class="audit-buttons"><button class="button ghost small share-point" data-sequence="${d.sakkaSequence??index}">${escapeHtml(copy('player_share_here'))}</button><button class="button ghost small feedback-point" data-sequence="${d.sakkaSequence??index}" data-action='${escapeHtml(JSON.stringify(d.bestAction||null))}'>${escapeHtml(copy('player_feedback'))}</button></div></article>`}).join('')||`<p class="empty">${escapeHtml(copy('player_no_comparable'))}</p>`;
    document.querySelectorAll('.share-point').forEach(btn=>btn.onclick=()=>shareDecision(+btn.dataset.sequence));
    document.querySelectorAll('.feedback-point').forEach(btn=>btn.onclick=()=>feedbackDecision(+btn.dataset.sequence,JSON.parse(btn.dataset.action)));
  }
  async function shareDecision(sequence){try{const board=await api(`/api/v1/sakka/runs/${state.run.runId}/actions/${sequence}:share`,{method:'POST',body:JSON.stringify({scope:'round',maximumRounds:1,label:copy('player_shared_sakka_label'),visibility:'unlisted'})});const room=await api('/api/v1/sakka/rooms',{method:'POST',body:JSON.stringify({boardIdentifier:board.boardId,mode:'async',maxPlayers:16,allowGuests:true})});await navigator.clipboard?.writeText(room.room.roomCode);toast(copy('player_share_created',{code:room.room.roomCode}))}catch(e){toast(e.message)}}
  async function feedbackDecision(sequence,bestAction){const rationale=prompt(copy('player_feedback_prompt'));if(!rationale)return;try{await api(`/api/v1/sakka/runs/${state.run.runId}/actions/${sequence}/feedback`,{method:'POST',body:JSON.stringify({preferredAction:bestAction,confidence:.8,rationale,interestingness:.5})});toast(copy('player_feedback_saved'))}catch(e){toast(e.message)}}
  async function loadRoom(code){ const payload=await api(`/api/v1/sakka/rooms/${encodeURIComponent(code)}`); await api(`/api/v1/sakka/rooms/${encodeURIComponent(code)}:join`,{method:'POST',body:'{}'}); renderLobby(await api(`/api/v1/sakka/rooms/${encodeURIComponent(code)}`)); }

  $('open-create').onclick=()=>show('create'); $('open-join').onclick=()=>show('join'); document.querySelectorAll('[data-back]').forEach(b=>b.onclick=()=>show('home'));
  $('create-form').addEventListener('submit',async e=>{e.preventDefault(); const f=new FormData(e.currentTarget); try{const board=await api('/api/v1/sakka/boards',{method:'POST',body:JSON.stringify({source:'new',label:f.get('label')||null,scope:f.get('scope'),rulesProfile:f.get('rulesProfile'),observerSeat:f.get('observerSeat'),usScore:+f.get('usScore'),themScore:+f.get('themScore'),seed:+f.get('seed'),visibility:f.get('visibility'),maximumRounds:16})}); const room=await api('/api/v1/sakka/rooms',{method:'POST',body:JSON.stringify({boardIdentifier:board.boardId,mode:'async',maxPlayers:16,allowGuests:true})}); await loadRoom(room.room.roomCode);}catch(err){toast(err.message)}});
  $('join-form').addEventListener('submit',async e=>{e.preventDefault(); const code=$('room-code').value.trim().toUpperCase(); if(!code)return; try{await loadRoom(code)}catch(err){toast(err.message)}});
  $('copy-code').onclick=async()=>{const text=state.room?.roomCode||'';try{await navigator.clipboard.writeText(text);toast(copy('player_code_copied'))}catch{toast(text)}};
  $('refresh-lobby').onclick=async()=>{try{if(state.room)renderLobby(await api(`/api/v1/sakka/rooms/${state.room.roomCode}`))}catch(err){toast(err.message)}};
  $('compare-top').onclick=compareTopRuns;
  $('start-run').onclick=async()=>{try{renderGame(await api(`/api/v1/sakka/rooms/${state.room.roomCode}/runs`,{method:'POST',body:'{}'}))}catch(err){toast(err.message)}};
  $('play-selected').onclick=()=>{if(state.selectedPlay)act(state.selectedPlay,{sourceButton:document.querySelector('#hand .playing-card.selected'),swiped:true})};
  $('leave-game').onclick=()=>{if(state.room)loadRoom(state.room.roomCode).catch(err=>toast(err.message))};
  $('show-leaderboard').onclick=()=>loadRoom(state.room.roomCode).catch(err=>toast(err.message)); $('home-button').onclick=()=>show('home');
  $('verify-run').onclick=async()=>{try{$('verify-run').disabled=true;const check=await api(`/api/v1/sakka/runs/${state.run.runId}/verification`);const line=$('run-verification');line.hidden=false;line.className=`verification-line result-verification ${check.valid?'valid':'invalid'}`;line.textContent=check.valid?copy('player_replay_verified',{actions:check.actionCount}):copy('player_replay_failed',{issues:(check.issues||[]).map(i=>i.code).join(' · ')})}catch(e){toast(e.message)}finally{$('verify-run').disabled=false}};
  $('analyze-run').onclick=async()=>{try{$('analyze-run').disabled=true;$('analyze-run').textContent=copy('player_analyzing');const report=await api(`/api/v1/sakka/runs/${state.run.runId}:audit`,{method:'POST',body:JSON.stringify({deep:false,maximumDecisions:32})});renderAudit(report)}catch(e){toast(e.message)}finally{$('analyze-run').disabled=false;$('analyze-run').textContent=copy('player_analyze_decisions')}};

  const dialog=$('auth-dialog');$('auth-button').onclick=()=>dialog.showModal();document.querySelectorAll('[data-auth-tab]').forEach(tab=>tab.onclick=()=>{state.authMode=tab.dataset.authTab;document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t===tab));$('display-name-field').hidden=state.authMode!=='register';$('auth-submit').textContent=copy(state.authMode==='register'?'player_register':'player_login');$('auth-error').textContent=''});
  $('auth-submit').onclick=async()=>{const username=$('auth-username').value.trim(),password=$('auth-password').value;try{const path=state.authMode==='register'?'/api/v1/auth/register':'/api/v1/auth/login';const body=state.authMode==='register'?{username,password,displayName:$('auth-display-name').value.trim()||username}:{username,password};const result=await api(path,{method:'POST',body:JSON.stringify(body)});state.identity=result.identity;state.csrf=result.csrfToken;$('identity-name').textContent=state.identity.displayName;$('auth-button').textContent=state.identity.displayName;dialog.close();toast(copy('player_login_success'))}catch(err){$('auth-error').textContent=err.message}};

  (async()=>{try{await ensureIdentity();const code=new URLSearchParams(location.search).get('code');if(code)await loadRoom(code.toUpperCase())}catch(err){toast(err.message)}})();
})();
