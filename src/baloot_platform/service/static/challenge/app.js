(() => {
  const state = {
    challenge: null,
    identity: null,
    csrf: null,
    selected: null,
    confidence: 0.7,
    started: 0,
    drag: null,
    committing: false,
    selectionChanges: 0,
    suppressClickUntil: 0,
    copy: null,
  };
  const $ = (id) => document.getElementById(id);
  const seatOrder = ['south', 'west', 'north', 'east'];
  const canonicalSeats = ['north', 'west', 'south', 'east'];
  const rankCodes = {
    ace: 'A', king: 'K', queen: 'Q', jack: 'J', ten: '10',
    nine: '9', eight: '8', seven: '7',
  };
  const suitSymbols = {spades: '♠', hearts: '♥', diamonds: '♦', clubs: '♣'};
  const requiredFrozenCopy = [
    'action_accessibility', 'action_bid_ashkal', 'action_bid_hokum',
    'action_bid_pass', 'action_bid_sun', 'action_escalation_decline',
    'action_escalation_double', 'action_escalation_four',
    'action_escalation_gahwa', 'action_escalation_triple',
    'action_mode_locked', 'action_mode_open', 'action_project_declare',
    'action_project_fifty', 'action_project_four_hundred',
    'action_project_hundred_four_kind', 'action_project_hundred_sequence',
    'action_project_siri', 'decision_options_aria', 'proof_boundary_bounded',
    'proof_boundary_exact', 'proof_boundary_policy_conditioned',
    'prompt_bid', 'prompt_escalation', 'prompt_project',
  ];

  function copy(key, values = {}) {
    const template = state.copy?.[key];
    if (typeof template !== 'string' || !template.trim()) {
      throw new Error(`Missing governed challenge copy: ${key}`);
    }
    return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) =>
      Object.hasOwn(values, name) ? String(values[name]) : match
    );
  }
  function applyStaticCopy() {
    document.title = copy('page_title');
    $('brand').setAttribute('aria-label', copy('brand_name'));
    $('brand-mark').textContent = copy('brand_mark');
    $('brand-name').textContent = copy('brand_name');
    $('code-label').textContent = copy('code_label');
    $('loading-copy').textContent = copy('loading');
    $('error-title').textContent = copy('error_title');
    $('scorebar').setAttribute('aria-label', copy('score_aria'));
    $('us-label').textContent = copy('us_score');
    $('them-label').textContent = copy('them_score');
    $('table').setAttribute('aria-label', copy('table_aria'));
    $('play-target-copy').textContent = copy('play_target');
    $('decision-eyebrow').textContent = copy('decision_eyebrow');
    $('confidence').setAttribute('aria-label', copy('confidence_aria'));
    $('confidence-unsure').textContent = copy('confidence_unsure');
    $('confidence-confident').textContent = copy('confidence_confident');
    $('confidence-certain').textContent = copy('confidence_certain');
    $('drag-hint').textContent = copy('drag_hint');
    $('hand').setAttribute('aria-label', copy('hand_aria'));
    $('commit').textContent = copy('commit');
    $('truth-label').textContent = copy('result_default_label');
    $('reason-label').textContent = copy('reason_label');
    $('participants-label').textContent = copy('participants');
    $('recommended-rate-label').textContent = copy('recommended_rate');
    $('difficulty-label').textContent = copy('audience_difficulty');
    $('player-choices-label').textContent = copy('player_choices');
    $('friend-choices-label').textContent = copy('friend_choices');
    $('next-challenge').textContent = copy('next_challenge');
  }

  function bindFrozenCopy(value) {
    const profile = value?.copyProfile;
    const governed = value?.copy;
    const researchChallenge = Boolean(value?.decisionKind);
    if (
      (researchChallenge && (
        !profile || profile.schemaVersion !== 1
        || !profile.bundleRevisionId || !profile.profileId || !profile.profileVersion
        || !['ar-SA', 'en'].includes(profile.locale)
        || !['rtl', 'ltr'].includes(profile.direction)
        || !/^[0-9a-f]{64}$/.test(String(profile.profileSha256 || ''))
        || !/^[0-9a-f]{64}$/.test(String(profile.copySha256 || ''))
      ))
      || !governed || typeof governed !== 'object' || Array.isArray(governed)
      || requiredFrozenCopy.some((key) => typeof governed[key] !== 'string' || !governed[key].trim())
    ) {
      throw new Error('frozen_challenge_copy_unavailable');
    }
    state.copy = governed;
    if (profile) {
      document.documentElement.lang = profile.locale;
      document.documentElement.dir = profile.direction;
    }
    applyStaticCopy();
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
    })[character]);
  }
  function toast(message) {
    const element = $('toast');
    element.textContent = message;
    element.classList.add('show');
    clearTimeout(element._timer);
    element._timer = setTimeout(() => element.classList.remove('show'), 2200);
  }
  async function api(path, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    const headers = {'content-type': 'application/json', ...(options.headers || {})};
    if (state.csrf && !['GET', 'HEAD'].includes(method)) headers['x-csrf-token'] = state.csrf;
    const response = await fetch(path, {credentials: 'include', ...options, headers});
    const text = await response.text();
    let value = {};
    try { value = text ? JSON.parse(text) : {}; } catch { value = {detail: text}; }
    if (!response.ok) {
      const error = new Error(
        response.status === 404
          ? copy('missing_code')
          : copy('request_error', {status: response.status})
      );
      error.status = response.status;
      throw error;
    }
    return value;
  }
  async function frozenChallenge(code) {
    const response = await fetch(
      `/api/v2/challenge-locators/${encodeURIComponent(code)}`,
      {credentials: 'include', headers: {'content-type': 'application/json'}},
    );
    if (!response.ok) throw new Error(`challenge_unavailable:${response.status}`);
    const value = await response.json();
    bindFrozenCopy(value);
    return value;
  }
  async function ensureIdentity() {
    let value = await api('/api/v1/auth/me');
    if (!value.identity) {
      value = await api('/api/v1/auth/guest', {
        method: 'POST', body: JSON.stringify({displayName: copy('guest_name')}),
      });
    }
    state.identity = value.identity;
    state.csrf = value.csrfToken || null;
  }
  function rank(card) { return rankCodes[card?.rank] || card?.rank || ''; }
  function red(card) { return ['hearts', 'diamonds'].includes(card?.suit); }
  function cardLabel(card) { return card?.label || `${rank(card)}${suitSymbols[card?.suit] || ''}`; }
  function cardAsset(card) {
    const rankCode = rankCodes[card?.rank];
    const suit = Object.hasOwn(suitSymbols, card?.suit) ? card.suit : null;
    return rankCode && suit
      ? `/card-assets/${encodeURIComponent(rankCode)}_${encodeURIComponent(suit)}.svg`
      : '';
  }
  function cardFace(card, className = '') {
    const source = cardAsset(card);
    const label = escapeHtml(cardLabel(card));
    return `<span class="card-face ${className} ${red(card) ? 'red' : ''}">
      <span class="card-fallback" aria-hidden="true"><b>${escapeHtml(rank(card))}</b><i>${escapeHtml(suitSymbols[card?.suit] || '')}</i></span>
      ${source ? `<img src="${source}" alt="${label}" draggable="false">` : ''}
    </span>`;
  }
  function relativePositions(observer) {
    const index = seatOrder.indexOf(observer);
    return {
      bottom: observer,
      left: seatOrder[(index + 1) % 4],
      top: seatOrder[(index + 2) % 4],
      right: seatOrder[(index + 3) % 4],
    };
  }
  function relativeName(seat, positions) {
    return Object.entries(positions).find(([, value]) => value === seat)?.[0] || 'top';
  }
  function lastTrick(plays = []) {
    const remainder = plays.length % 4;
    return remainder ? plays.slice(-remainder) : [];
  }
  function purchaseRound(game) {
    const history = Array.isArray(game.biddingHistory) ? game.biddingHistory : [];
    const bid = [...history].reverse().find((item) =>
      item && (item.action === game.contract || item.contract === game.contract) && item.round
    );
    return Number(bid?.round || 1);
  }
  function purchaseLabel(game) {
    const contract = copy(`contract_${game.contract}`);
    const round = copy(purchaseRound(game) === 2 ? 'purchase_second' : 'purchase_first');
    const trump = game.contract === 'hokum' ? suitSymbols[game.trumpSuit] || '' : '';
    return copy(game.contract === 'hokum' ? 'purchase_hokum_summary' : 'purchase_plain_summary', {
      contract,
      round,
      trump,
    });
  }
  function playerName(value, seat) {
    const names = value.playerNames && typeof value.playerNames === 'object'
      ? value.playerNames : {};
    return String(names[seat] || '').trim();
  }
  function backStack(count) {
    const visible = Math.min(5, Math.max(0, Number(count) || 0));
    return `<span class="back-stack" aria-hidden="true">${Array.from({length: visible}, (_, index) => {
      const middle = (visible - 1) / 2;
      const offset = index - middle;
      return `<img src="/challenge-video-assets/card-back.svg" alt="" draggable="false" style="--offset:${offset};--layer:${index + 1}">`;
    }).join('')}</span>`;
  }
  function renderSeats(value) {
    const game = value.state;
    const positions = relativePositions(value.observerSeat);
    const sizes = game.handSizes || [];
    const counts = Object.fromEntries(canonicalSeats.map((seat, index) => [seat, sizes[index] ?? 0]));
    $('seats').innerHTML = Object.entries(positions).map(([position, seat]) => {
      const name = playerName(value, seat);
      const tags = [];
      if (seat === game.dealer) tags.push(`<span class="dealer-tag"><b>◆</b> ${escapeHtml(copy('role_dealer'))}</span>`);
      if (seat === game.buyer) tags.push(`<span class="buyer-tag"><b>♛</b> ${escapeHtml(copy('role_buyer'))}</span>`);
      const hiddenCards = position === 'bottom' ? '' : backStack(counts[seat]);
      return `<section class="seat seat-${position} ${seat === game.currentPlayer ? 'active' : ''}" aria-label="${escapeHtml(name || copy('role_player'))}">
        ${hiddenCards}
        <span class="seat-medallion"><b>${counts[seat]}</b></span>
        ${name ? `<span class="seat-name">${escapeHtml(name)}</span>` : ''}
        <span class="seat-tags">${tags.join('')}</span>
      </section>`;
    }).join('');
    const trick = lastTrick(game.plays || []);
    $('trick').innerHTML = trick.map((play) => {
      const card = play.card || play;
      const position = relativeName(play.seat, positions);
      return `<span class="table-card ${position}" data-play-seat="${escapeHtml(play.seat)}">${cardFace(card)}</span>`;
    }).join('');
  }
  function renderHeader(value) {
    const game = value.state;
    const score = game.matchScore || [0, 0];
    const ours = ['north', 'south'].includes(value.observerSeat) ? 0 : 1;
    $('us-score').textContent = score[ours] ?? 0;
    $('them-score').textContent = score[1 - ours] ?? 0;
    $('contract').textContent = purchaseLabel(game);
    const buyerName = game.buyer ? playerName(value, game.buyer) : '';
    $('buyer').textContent = game.buyer
      ? (buyerName ? copy('buyer_with_name', {buyer: buyerName}) : copy('role_buyer'))
      : '—';
    $('code').textContent = value.code;
    $('prompt').textContent = value.decisionKind && value.decisionKind !== 'play'
      ? copy(`prompt_${value.decisionKind}`)
      : copy(`prompt_${value.promptContract}`);
  }
  function clearDrag(button) {
    if (button) {
      button.style.removeProperty('--drag-x');
      button.style.removeProperty('--drag-y');
      button.classList.remove('dragging', 'drag-ready');
    }
    $('table').classList.remove('drop-ready');
    $('drag-hint').textContent = copy('drag_hint');
    state.drag = null;
  }
  function selectAction(code) {
    const next = state.challenge.legalActions.find((action) => action.code === code) || null;
    if (state.selected && next && state.selected.code !== next.code) state.selectionChanges += 1;
    state.selected = next;
    document.querySelectorAll('[data-action-code]').forEach((button) => {
      const selected = button.dataset.actionCode === code;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    $('commit').disabled = !state.selected || state.committing;
  }
  function selectCard(code) { selectAction(code); }
  function onPointerDown(event) {
    if (state.committing || event.button > 0) return;
    const button = event.currentTarget;
    state.drag = {
      button, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY,
      x: 0, y: 0,
    };
    button.setPointerCapture?.(event.pointerId);
    button.classList.add('dragging');
  }
  function onPointerMove(event) {
    const drag = state.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    drag.x = Math.max(-55, Math.min(55, event.clientX - drag.startX));
    drag.y = Math.max(-145, Math.min(12, event.clientY - drag.startY));
    drag.button.style.setProperty('--drag-x', `${drag.x * 0.28}px`);
    drag.button.style.setProperty('--drag-y', `${drag.y}px`);
    const ready = drag.y <= -72;
    drag.button.classList.toggle('drag-ready', ready);
    $('table').classList.toggle('drop-ready', ready);
    $('drag-hint').textContent = copy(ready ? 'drag_release' : 'drag_up');
  }
  function onPointerCancel(event) {
    if (state.drag?.pointerId === event.pointerId) clearDrag(state.drag.button);
  }
  function onPointerUp(event) {
    const drag = state.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const shouldPlay = drag.y <= -72;
    const button = drag.button;
    const code = button.dataset.card;
    state.suppressClickUntil = performance.now() + 350;
    clearDrag(button);
    selectCard(code);
    if (shouldPlay) {
      navigator.vibrate?.(12);
      commit({fromSwipe: true, sourceButton: button});
    }
  }
  function bindCard(button) {
    button.addEventListener('pointerdown', onPointerDown);
    button.addEventListener('pointermove', onPointerMove);
    button.addEventListener('pointerup', onPointerUp);
    button.addEventListener('pointercancel', onPointerCancel);
    button.addEventListener('click', () => {
      if (performance.now() < state.suppressClickUntil) return;
      selectCard(button.dataset.actionCode);
    });
  }
  function renderHand(value) {
    const cardDecision = value.decisionKind === 'play' || (
      !value.decisionKind
      && (value.legalActions || []).every((action) => action?.suit && action?.rank)
    );
    if (!cardDecision) {
      $('hand').classList.add('typed-actions');
      $('hand').dataset.count = String(value.legalActions.length);
      $('hand').setAttribute('aria-label', copy('decision_options_aria'));
      $('hand').innerHTML = value.legalActions.map((action) => (
        `<button type="button" class="decision-action" data-action-code="${escapeHtml(action.code)}" aria-label="${escapeHtml(copy('action_accessibility', {action: actionLabel(action)}))}" aria-pressed="false">${escapeHtml(actionLabel(action))}</button>`
      )).join('');
      document.querySelectorAll('.decision-action').forEach((button) => {
        button.addEventListener('click', () => selectAction(button.dataset.actionCode));
      });
      return;
    }
    $('hand').classList.remove('typed-actions');
    $('hand').setAttribute('aria-label', copy('hand_aria'));
    const legal = new Map((value.legalActions || []).map((card) => [card.code, card]));
    const observerHand = value.state.observerHand || [];
    $('hand').dataset.count = String(observerHand.length);
    $('hand').innerHTML = observerHand.map((card) => {
      const isLegal = legal.has(card.code);
      return `<button type="button" class="playing-card ${isLegal ? 'legal' : ''}" data-card="${escapeHtml(card.code)}" data-action-code="${escapeHtml(card.code)}" ${isLegal ? '' : 'disabled'} aria-label="${escapeHtml(cardLabel(card))}" aria-pressed="false">
        ${cardFace(card)}
      </button>`;
    }).join('');
    document.querySelectorAll('.playing-card.legal').forEach(bindCard);
  }
  function miniCard(card) {
    return card ? `<span class="answer-card">${cardFace(card)}</span>` : '';
  }
  function actionLabel(action) {
    if (action?.suit && action?.rank) return cardLabel(action);
    if (action?.kind && Number.isInteger(action?.value)) {
      const project = copy(`action_project_${action.kind}`);
      return copy('action_project_declare', {project});
    }
    const decisionKind = state.challenge?.decisionKind;
    if (decisionKind === 'bid') {
      return copy(`action_bid_${action?.action}`);
    }
    if (decisionKind === 'escalation') {
      const label = copy(`action_escalation_${action?.action}`);
      return action?.mode ? `${label} · ${copy(`action_mode_${action.mode}`)}` : label;
    }
    throw new Error('unsupported_governed_action_label');
  }
  function miniAction(action) {
    if (action?.suit && action?.rank) return miniCard(action);
    return `<span class="answer-action" dir="auto">${escapeHtml(actionLabel(action))}</span>`;
  }
  function actionKey(action) { return action?.code || JSON.stringify(action || {}); }
  function metricText(row) {
    if (!row) return '';
    const parts = [];
    if (Number.isFinite(row.expectedRoundMargin)) {
      parts.push(copy('metric_round_margin', {value: Number(row.expectedRoundMargin).toFixed(1)}));
    }
    if (Number.isFinite(row.contractSuccessProbability)) {
      parts.push(copy('metric_contract_success', {value: (100 * Number(row.contractSuccessProbability)).toFixed(0)}));
    } else if (Number.isFinite(row.roundLeadProbability)) {
      parts.push(copy('metric_round_lead', {value: (100 * Number(row.roundLeadProbability)).toFixed(0)}));
    }
    return parts.slice(0, 2).join(' · ');
  }
  function expertReason(reveal) {
    if (typeof reveal.explanation === 'string' && reveal.explanation.trim()) {
      return reveal.explanation.trim();
    }
    const rows = [...(reveal.actionEvidence || [])].sort((a, b) => Number(a.rank || 99) - Number(b.rank || 99));
    const best = rows[0];
    const next = rows[1];
    if (!best) return copy('reason_verified');
    if (Number.isFinite(best.expectedRoundMargin)) {
      return copy('reason_margin');
    }
    if (Number.isFinite(best.contractSuccessProbability) && next) {
      return copy('reason_contract');
    }
    return copy('reason_ranked');
  }
  function choiceRows(items = [], evidence = []) {
    const statsByAction = new Map(items.map((item) => [actionKey(item.action), item]));
    const rows = evidence.length
      ? evidence.map((row) => ({...statsByAction.get(actionKey(row.action)), action: row.action, evidence: row}))
      : items.map((item) => ({...item, evidence: null}));
    return rows.map((item) => {
      const detail = metricText(item.evidence);
      return `<div class="choice-row">
        <span class="choice-card">${miniAction(item.action)}</span>
        <span class="bar"><i style="width:${Math.max(0, Math.min(100, item.percentage || 0))}%"></i></span>
        <b>${Number(item.percentage || 0).toFixed(1)}%</b>
        ${detail ? `<small>${escapeHtml(detail)}</small>` : ''}
      </div>`;
    }).join('') || `<p class="result-note">${escapeHtml(copy('no_statistics'))}</p>`;
  }
  function renderResult(value) {
    $('challenge').hidden = true;
    $('result').hidden = false;
    const reveal = value.reveal || {};
    const stats = value.statistics || {};
    const governed = reveal.status === 'governed_answer';
    const accepted = governed && reveal.chosenAccepted;
    $('result-mark').textContent = governed ? (accepted ? '✓' : '×') : '•';
    $('result-mark').classList.toggle('neutral', !governed);
    $('truth-label').textContent = reveal.truthLabel
      ? copy(`truth_${reveal.truthLabel}`)
      : copy('result_default_label');
    $('result-title').textContent = governed
      ? (accepted ? copy('result_supported') : copy('result_different')) : copy('result_recorded');
    const answer = [`<span class="answer-block"><small>${escapeHtml(copy('your_choice'))}</small>${miniAction(value.chosenAction)}</span>`];
    if (governed) answer.push(`<span class="answer-block"><small>${escapeHtml(copy('recommendation'))}</small>${miniAction(reveal.recommendedAction)}</span>`);
    $('answer-cards').innerHTML = answer.join('');
    const boundary = reveal.proofBoundary || {};
    $('truth-label').dataset.proofMode = boundary.achievedMode || '';
    if (boundary.achievedMode) {
      const boundaryKey = boundary.achievedMode === 'policy_conditioned'
        ? 'proof_boundary_policy_conditioned'
        : boundary.achievedMode === 'bounded'
          ? 'proof_boundary_bounded'
          : boundary.achievedMode === 'exact'
            ? 'proof_boundary_exact'
            : null;
      if (!boundaryKey) throw new Error('unsupported_governed_proof_boundary');
      $('truth-label').insertAdjacentHTML(
        'beforeend',
        ` <small class="proof-boundary">${escapeHtml(copy(boundaryKey))}</small>`,
      );
    }
    $('result-note').textContent = governed
      ? (reveal.answerMode === 'multiple_supported'
        ? copy('note_multiple')
        : copy('note_single'))
      : copy('note_unavailable');
    $('expert-reason').hidden = !governed;
    if (governed) $('expert-reason-copy').textContent = expertReason(reveal);
    $('attempt-count').textContent = stats.totalAttempts ?? 0;
    $('recommended-rate').textContent = stats.recommendedActionPercentage == null
      ? '—' : `${Number(stats.recommendedActionPercentage).toFixed(1)}%`;
    const audience = stats.audienceDifficulty || {};
    $('difficulty').textContent = audience.status === 'calibrated'
      ? `${Math.round(audience.audienceCalibratedDifficulty)}/100` : copy('calibrating');
    $('choices').innerHTML = choiceRows(stats.choices || [], reveal.actionEvidence || []);
    const friends = stats.friends || {};
    $('friends-panel').hidden = friends.status !== 'available' || !friends.totalAttempts;
    $('friends').innerHTML = choiceRows(friends.choices || []);
    window.scrollTo({top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
  }
  async function animateCardToTable(sourceButton) {
    if (!sourceButton) return;
    const source = sourceButton.getBoundingClientRect();
    const target = $('play-target').getBoundingClientRect();
    const clone = sourceButton.cloneNode(true);
    clone.classList.add('card-flight');
    clone.removeAttribute('data-card');
    clone.style.cssText = `left:${source.left}px;top:${source.top}px;width:${source.width}px;height:${source.height}px`;
    document.body.appendChild(clone);
    sourceButton.style.opacity = '0';
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const dx = target.left + target.width / 2 - (source.left + source.width / 2);
    const dy = target.top + target.height / 2 - (source.top + source.height / 2);
    try {
      await clone.animate(
        reduced
          ? [{opacity: 1}, {opacity: 0}]
          : [
              {transform: 'translate3d(0,0,0) scale(1) rotate(0deg)', opacity: 1},
              {transform: `translate3d(${dx}px,${dy}px,0) scale(.72) rotate(-2deg)`, opacity: 0.92},
            ],
        {duration: reduced ? 90 : 360, easing: 'cubic-bezier(.2,.82,.24,1)', fill: 'forwards'},
      ).finished;
    } catch { /* The API commit still proceeds if animation is interrupted. */ }
    clone.remove();
  }
  async function commit({fromSwipe = false, sourceButton = null} = {}) {
    if (!state.selected || state.committing) return;
    state.committing = true;
    const button = $('commit');
    button.disabled = true;
    button.textContent = copy('committing');
    try {
      if (fromSwipe) await animateCardToTable(sourceButton);
      const elapsed = Math.min(3600, (performance.now() - state.started) / 1000);
      const value = await api(`/api/v2/challenge-locators/${encodeURIComponent(state.challenge.code)}/attempts`, {
        method: 'POST',
        body: JSON.stringify({
          chosenAction: state.selected,
          responseSeconds: elapsed,
          confidence: state.confidence,
          changedAnswer: state.selectionChanges > 0,
        }),
      });
      renderResult(value);
    } catch (error) {
      if (sourceButton) sourceButton.style.opacity = '';
      state.committing = false;
      button.disabled = false;
      button.textContent = copy('commit');
      toast(error.message);
    }
  }
  async function start() {
    const code = decodeURIComponent(location.pathname.split('/').filter(Boolean).pop() || '').trim().toUpperCase();
    if (!code) throw new Error(copy('missing_code'));
    const value = await frozenChallenge(code);
    await ensureIdentity();
    if (value.promptContract !== 'choose_acting_action') {
      throw new Error(copy('unsupported_prompt'));
    }
    state.challenge = value;
    state.started = performance.now();
    renderHeader(value);
    renderSeats(value);
    renderHand(value);
    const restored = await api(
      `/api/v2/challenge-locators/${encodeURIComponent(code)}/attempt`
    );
    $('loading').hidden = true;
    if (restored.committed) {
      renderResult(restored);
    } else {
      $('challenge').hidden = false;
    }
  }

  document.querySelectorAll('[data-confidence]').forEach((button) => {
    button.addEventListener('click', () => {
      state.confidence = Number(button.dataset.confidence);
      document.querySelectorAll('[data-confidence]').forEach((item) => item.classList.toggle('active', item === button));
    });
  });
  $('commit').addEventListener('click', () => commit());
  (async () => {
    await start();
  })().catch((error) => {
    $('loading').hidden = true;
    if (state.copy) {
      $('error').hidden = false;
      $('error-copy').textContent = error.message;
    } else {
      document.documentElement.dataset.challengeStatus = 'copy-unavailable';
    }
  });
})();
