// Server-renders the Majlis frame (no browser) and checks the selectors that
// playback DOM capture and media checks depend on, plus the palette contract.
import assert from "node:assert/strict";
import {copyFileSync, mkdirSync, readFileSync, writeFileSync} from "node:fs";
import {test} from "node:test";
import ts from "typescript";
import {checkLayoutGeometry, loadLayoutTable, sha256} from "../scripts/layout-gate.mjs";

const out = new URL("../node_modules/.cache/majlis-dom-test/", import.meta.url);
mkdirSync(out, {recursive: true});
const transpile = (name, rewrite = (text) => text) => {
  const source = readFileSync(new URL(`../src/${name}`, import.meta.url), "utf8");
  const {outputText} = ts.transpileModule(rewrite(source), {
    compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX},
  });
  writeFileSync(new URL(name.replace(/\.tsx?$/, ".mjs"), out), outputText);
};
// Remotion's <Img>/<AbsoluteFill> need a live composition; plain elements keep the tree identical.
writeFileSync(new URL("remotion-stub.mjs", out), `
import React from "react";
export {Easing, interpolate} from "remotion";
export const AbsoluteFill = ({children, ...props}) => React.createElement("div", props, children);
export const Img = (props) => React.createElement("img", props);
export const Sequence = ({children}) => children;
export const Audio = () => null;
export const Artifact = () => null;
export const staticFile = (path) => "/" + path;
export const delayRender = () => 0;
export const continueRender = () => undefined;
export const cancelRender = (error) => { throw error; };
export const useCurrentFrame = () => 0;
`);
transpile("playback-state.ts");
copyFileSync(new URL("../src/playback-timing.mjs", import.meta.url), new URL("playback-timing.mjs", out));
transpile("professional-prototype.tsx", (text) => text
  .replace('import "./style.css";', "")
  .replace('} from "remotion";', '} from "./remotion-stub.mjs";')
  .replace('from "./playback-state";', 'from "./playback-state.mjs";'));
const React = (await import("react")).default;
const {renderToStaticMarkup} = await import("react-dom/server");
const {ProfessionalPrototypeFrame, majlisPaletteTokens} = await import(new URL("professional-prototype.mjs", out));

const card = (id, rank, suit) => ({id, rank, suit});
const palette = Object.fromEntries(majlisPaletteTokens.map((token, index) => [token, `#${(0x100000 + index).toString(16)}`]));
const localizations = new Proxy({}, {get: (_, key) => (typeof key === "string" ? `L(${key})` : undefined)});
const bundle = (overrides = {}) => ({
  exportPreset: "decision_clip",
  render: {fps: 30},
  identity: {gameSerial: "BLT-9646-44A0-AA14-ACC8", sceneId: "q", enrollmentUri: "balootx://g/BLT", players: {north: "N", west: "W", east: "E", south: "S"}},
  localizations,
  audioCues: [],
  timeline: {segments: []},
  scenes: [{stage: "context", observer: "south", layers: [
    {id: "public-lifecycle", content: {dealer: "east", buyer: "west", contract: "hokum", trumpSuit: "diamonds", purchaseCard: card(9, "nine", "diamonds")}},
  ]}, {stage: "question", observer: "south", layers: [
    {id: "table", content: {currentPlayer: "south", dealer: "east", buyer: "west", contract: "hokum", trumpSuit: "diamonds", matchScore: [140, 140],
      currentTrick: {plays: [{seat: "north", card: card(1, "jack", "hearts")}, {seat: "west", card: card(2, "eight", "hearts")}, {seat: "east", card: card(3, "ten", "hearts")}]}}},
    {id: "observer-hand", content: {cards: [card(4, "ace", "clubs"), card(5, "nine", "clubs"), card(6, "jack", "spades"), card(7, "seven", "hearts"), card(8, "ace", "hearts")]}},
    {id: "seats", content: {handCounts: {north: 4, west: 4, east: 4, south: 5}}},
  ]}],
  presentation: {
    timeline: {pause_seconds: 7}, layout: {}, motion: {}, callouts: {fade_in_ms: 110, fade_out_ms: 130},
    presentation: {show_identity_pill: true, show_enrollment_qr: true, qr_size_px: 80, qr_quiet_zone_modules: 4},
    majlis: {theme: "red", hud: "plaque", id_box: "qr", score_target: 152, social: {tiktok: "t", instagram: "i"}, palette: {...palette}},
  },
  ...overrides,
});
const render = (value, stage = "hook") =>
  renderToStaticMarkup(React.createElement(ProfessionalPrototypeFrame, {bundle: value, stage, compositionFrame: 0}));

test("preserved selectors and data attributes render", () => {
  const html = render(bundle());
  assert.match(html, /<header class="broadcast-hud[^"]*"[^>]*data-score-us="140"[^>]*data-score-them="140"/);
  assert.match(html, /<main class="play-table"/);
  assert.match(html, /class="trick-zone"[^>]*data-completed-play-count="0"[^>]*data-current-trick-count="3"[^>]*data-table-play-count="3"/);
  assert.equal((html.match(/class="card played /g) ?? []).length, 3);
  for (const seat of ["north", "west", "east"]) {
    assert.match(html, new RegExp(`class="seat-dock dock-${seat}[^"]*" data-seat="${seat}" data-hand-count="4"`));
  }
  assert.match(html, /class="observer-hand[^"]*"[^>]*data-seat="south" data-hand-count="5"/);
});

test("hidden hands carry no face card ids; small cards never carry ids", () => {
  const html = render(bundle());
  for (const dock of html.match(/<section class="seat-dock[\s\S]*?<\/section>/g) ?? []) {
    assert.doesNotMatch(dock, /data-card-id=/);
  }
  assert.match(html, /class="tag buyer"[^>]*><span>[^<]*<\/span><img class="card card-small pcard" data-small-card-code="D9"/);
  assert.doesNotMatch(html, /card-small[^>]*data-card-id=/);
});

test("palette drives CSS variables and bad colours fail closed", () => {
  const html = render(bundle());
  assert.match(html, new RegExp(`--mj-rug:${palette.rug}`));
  assert.match(html, new RegExp(`--mj-rug-line:${palette.rug_line}`));
  const bad = bundle();
  bad.presentation.majlis.palette = {...palette, accent: "red"};
  assert.throws(() => render(bad), /invalid governed majlis colour: accent/);
  const missing = bundle();
  delete missing.presentation.majlis.palette.rug;
  assert.throws(() => render(missing), /governed majlis colour: rug/);
  const hud = bundle();
  hud.presentation.majlis.hud = "banner";
  assert.throws(() => render(hud), /governed majlis hud/);
});

test("hud styles, id box and research-only mode", () => {
  for (const style of ["plaque", "slim", "cushions"]) {
    const value = bundle();
    value.presentation.majlis.hud = style;
    assert.match(render(value), new RegExp(`class="broadcast-hud hud hud-${style}"`));
  }
  assert.match(render(bundle()), /class="qr qr-shell"/);
  const number = bundle();
  number.presentation.majlis.id_box = "number";
  // Public rule: the session code box is red even with the research notice off.
  assert.match(render(number), /class="idnum game-number is-notice"[\s\S]*?ACC8/);
  const research = bundle({source: {qualificationStatus: "rejected", renderPurpose: "research_audit", auditOnly: true, publicationEligible: false,
    qualificationFailures: ["x"], publicationBlockers: ["x"], qrActionable: false}});
  const html = render(research);
  assert.match(html, /class="audit-media-badge rstamp"/);
  assert.match(html, /class="rhead"/);
  assert.equal((html.match(/class="tape tape-/g) ?? []).length, 4);
  assert.match(html, /class="watermark"/);
  assert.doesNotMatch(html, /qr-shell/);
});

test("show_justification_text=false hides reason, analysis and proof text; absent keeps it", () => {
  const withReveal = (flag) => {
    const value = bundle();
    const [, question] = value.scenes;
    value.scenes.push({...question, stage: "reveal", layers: [...question.layers, {id: "analysis", content: {
      selectedAction: "CA", reasonKey: "video.analysis.expert_reason", proofScopeKey: "video.analysis.proof_scope.complete_state",
      sunTieOutcome: "buyer_makes", winningRootActionCount: 1,
    }}]});
    if (flag !== undefined) value.presentation.presentation.show_justification_text = flag;
    value.presentation.layout = {};
    return render(value, "reveal");
  };
  for (const flag of [false]) {
    const html = withReveal(flag);
    assert.match(html, /class="reveal-reason"[^>]*data-selected-action="CA"/);
    assert.match(html, /class="reason is-answer-only"/);
    assert.match(html, /data-small-card-code="CA"/);
    assert.doesNotMatch(html, /reason-text|class="scope"|L\(video\.analysis\.|L\(video\.scoring\./);
  }
  for (const flag of [undefined, true]) {
    assert.match(withReveal(flag), /class="reason-text"[\s\S]*?L\(video\.analysis\.expert_reason\)/);
  }
});

const approvedNotice = "هذا الفيديو لأغراض البحث فقط، ويُمنع تداوله أو نشره أو مشاركته";
const researchNoticeBundle = (noticeStrings = {}) => {
  const value = bundle({
    source: {qualificationStatus: "qualified", renderPurpose: "content_candidate", auditOnly: false,
      publicationEligible: true, qualificationFailures: [], publicationBlockers: [], qrActionable: false},
    localizations: new Proxy({"video.notice.sentence": approvedNotice, ...noticeStrings}, {
      get: (strings, key) => strings[key] ?? localizations[key],
    }),
  });
  value.presentation.majlis.research_notice = true;
  return value;
};

test("research notice: one approved line in the header band with a red code box and no audit marks", () => {
  const value = researchNoticeBundle();
  const html = render(value);
  assert.match(html, /class="rhead research-notice" role="note"><span class="rhead-icon" aria-hidden="true">!<\/span>/);
  const lines = [...html.matchAll(/<span class="rhead-text"><span class="notice-sentence" dir="rtl">([^<]*)<\/span><\/span>/g)];
  assert.deepEqual(lines.map((line) => line[1]), [approvedNotice]);
  assert.match(html, /class="idnum game-number is-notice"[\s\S]*?ACC8/);
  assert.doesNotMatch(html, /tape tape-|class="watermark"|rstamp|L\(video\.governance\.not_publishable\)/);
  value.presentation.majlis.research_notice = false;
  const off = render(value);
  assert.doesNotMatch(off, /class="rhead/);
  assert.match(off, /class="idnum game-number is-notice"[\s\S]*?ACC8/);
});

test("research notice omits the old title even if a legacy localization is supplied", () => {
  const oldTitle = "ملاحظة بحثية";
  const html = render(researchNoticeBundle({"video.notice.title": oldTitle}));
  assert.match(html, /class="rhead research-notice"/);
  assert.ok(!html.includes(oldTitle));
});

test("rendered face cards equal hand plus table, including the bidding deck", () => {
  const faces = (html) => (html.match(/data-card-code="/g) ?? []).length;
  assert.equal(faces(render(bundle())), 5 + 3);
  const bidding = bundle({timeline: {segments: [{stage: "setup", startFrame: 0, durationFrames: 30}]}, exportPreset: "hooked_flagship"});
  const context = bidding.scenes[0].layers[0].content;
  Object.assign(context, {buyer: null, bidHistory: [{seat: "north", action: "pass", round: 1}, {seat: "west", action: "hokum", round: 1}], purchaseRecipient: "west"});
  bidding.scenes.push({stage: "continuation", observer: "south", layers: [
    {id: "public-lifecycle", content: {...context, plays: [], completedTricks: []}},
    {id: "observer-lifecycle", content: {initialFiveCards: bidding.scenes[1].layers[1].content.cards, initialHandCards: bidding.scenes[1].layers[1].content.cards}},
  ]});
  bidding.scenes[0].layers.push({id: "observer-lifecycle", content: {initialFiveCards: bidding.scenes[1].layers[1].content.cards, initialHandCards: bidding.scenes[1].layers[1].content.cards}});
  const html = renderToStaticMarkup(React.createElement(ProfessionalPrototypeFrame, {bundle: bidding, stage: "flagship", compositionFrame: 0}));
  assert.match(html, /class="setup-deck"/);
  assert.match(html, /class="card card-small purchase-card" data-small-card-code="D9"/);
  assert.equal(faces(html), 5);
});

const replayClip = (bidHistory, extraSegments = []) => {
  const value = bundle({timeline: {segments: [
    {stage: "setup", startFrame: 0, durationFrames: 90}, {stage: "replay", startFrame: 90, durationFrames: 30}, ...extraSegments,
  ]}});
  const five = value.scenes[1].layers[1].content.cards;
  const context = value.scenes[0].layers[0].content;
  Object.assign(context, {buyer: "south", bidHistory, purchaseRecipient: "south", plays: [], completedTricks: []});
  value.scenes[0].layers.push({id: "observer-lifecycle", content: {initialFiveCards: five, initialHandCards: five}});
  value.scenes.push({stage: "continuation", observer: "south", layers: value.scenes[0].layers});
  return value;
};
const frameAt = (value, compositionFrame) =>
  renderToStaticMarkup(React.createElement(ProfessionalPrototypeFrame, {bundle: value, stage: "flagship", compositionFrame}));
const setupBeat = (html) => html.match(/class="call-layer" data-setup-action-index="(\d+)" data-setup-action-kind="(\w+)" data-setup-actor="(\w+)"/)?.slice(1);

test("decision-clip bidding: the dealer calls أول, the first round, then ثاني before a recorded second round", () => {
  const passes = ["south", "west", "north", "east"].map((seat) => ({seat, action: "pass", round: 1}));
  const value = replayClip([...passes, {seat: "south", action: "hokum", round: 2, trumpSuit: "diamonds"}]);
  // 8 equal beats over 90 frames: أول, four round-1 calls, ثاني, the round-2 bid, purchase.
  const beats = [0, 12, 23, 34, 45, 57, 68, 80].map((frame) => setupBeat(frameAt(value, frame)));
  assert.deepEqual(beats, [
    ["0", "round", "east"], ["1", "bid", "south"], ["2", "bid", "west"], ["3", "bid", "north"], ["4", "bid", "east"],
    ["5", "round", "east"], ["6", "bid", "south"], ["7", "purchase", "south"],
  ]);
  assert.match(frameAt(value, 0), /class="call call-hot call-at-east call-setup"[^>]*><b>L\(video\.bid\.round\.1\)<\/b><\/div>/);
  assert.match(frameAt(value, 57), /class="call call-hot call-at-east call-setup"[^>]*><b>L\(video\.bid\.round\.2\)<\/b><\/div>/);
});

test("decision-clip bidding with a single recorded round never calls ثاني", () => {
  const value = replayClip([{seat: "north", action: "sun", round: 1}]);
  const frames = [0, 45, 89].map((frame) => frameAt(value, frame));
  assert.deepEqual(frames.map(setupBeat), [["0", "round", "east"], ["1", "bid", "north"], ["2", "purchase", "south"]]);
  for (const html of frames) assert.doesNotMatch(html, /L\(video\.bid\.round\.2\)/);
});

test("decision clip played to the round end holds on the engine's final score, with no reason text", () => {
  const value = replayClip([{seat: "north", action: "sun", round: 1}], [
    {stage: "continuation", startFrame: 120, durationFrames: 30}, {stage: "outcome", startFrame: 150, durationFrames: 75},
  ]);
  Object.assign(value.scenes[0].layers[0].content, {matchScore: [0, 46], roundScore: {
    cardGamePoints: [2, 24], projectPoints: [20, 0], balootPoints: [0, 0], awardedPoints: [0, 46],
  }});
  const html = frameAt(value, 180);
  assert.match(html, /class="outcome-card ledger/);
  assert.match(html, /class="total-us">0<\/b><b dir="ltr" class="total-them">46<\/b>/);
  assert.match(html, /data-score-us="0"[^>]*data-score-them="46"/);
  assert.doesNotMatch(html, /reason-text|class="scope"|L\(video\.analysis\.|call-layer/);
});

test("north and south side items sit in a lane around the real name width", () => {
  const value = bundle();
  value.identity.players.north = "لاعب تجريبي شمالي";
  value.identity.players.south = "لاعب تجريبي جنوبي";
  value.scenes[0].layers[0].content.buyer = "north";
  value.scenes[1].layers[0].content.buyer = "north";
  const html = render(value);
  const lane = (side) => {
    const state = side === "south" ? " is-play-prompt" : "";
    const tag = `<div class="seat-lane lane-${side}${state}" dir="ltr">`;
    const start = html.indexOf(tag);
    assert.ok(start >= 0, side);
    const end = html.indexOf('<div class="seat-lane', start + 1);
    return html.slice(start + tag.length, end > 0 ? end : undefined);
  };
  const north = lane("north");
  assert.match(north, /^<div class="lane-side lane-start" dir="rtl"><div class="seat-tags tags-north"><span class="tag buyer"/);
  assert.match(north, /<div class="lane-spacer" aria-hidden="true"><div class="seat "><span class="count">4<\/span><span class="name">لاعب تجريبي شمالي<\/span>/);
  assert.match(lane("south"), /<div class="lane-spacer" aria-hidden="true"><div class="seat "><span class="count">5<\/span><span class="name">لاعب تجريبي جنوبي<\/span>/);
  assert.equal((html.match(/tags-north/g) ?? []).length, 1);
});

test("empty social handle hides its pill; placeholders block publishable clips", () => {
  const hidden = bundle();
  hidden.presentation.majlis.social = {tiktok: "", instagram: "balootx"};
  const html = render(hidden);
  assert.doesNotMatch(html, /social-tiktok/);
  assert.match(html, /social-instagram[\s\S]*?@balootx/);
  const publishable = bundle({source: {qualificationStatus: "qualified", renderPurpose: "content_candidate", auditOnly: false,
    publicationEligible: true, qualificationFailures: [], publicationBlockers: [], qrActionable: true}});
  for (const handle of ["[tiktok]", "@[tiktok]", " @[tiktok] "]) {
    publishable.presentation.majlis.social = {tiktok: handle, instagram: "x"};
    assert.throws(() => render(publishable), /placeholder social handles/, handle);
  }
});

const css = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
const rule = (selector) => {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`(?:^|\\n)${escaped} \\{([^}]*)\\}`));
  assert.ok(!new RegExp(`(?:^|\\n)${escaped} \\{`).test(css.replaceAll(selector, selector.replaceAll(".", "x"))), `${selector} must match literally`);
  assert.ok(match, selector);
  return match[1];
};
const px = (selector, property) => Number(rule(selector).match(new RegExp(`(?:^|[;\\s])${property}:\\s*(-?\\d+)px`))[1]);

test("reveal: choice badge, name pills and the raised card keep a 12 px gap at every seat", () => {
  const MIN_GAP = 12;
  const W = 1080;
  // Separation of two boxes: positive when apart on either axis, negative when they intersect.
  const gap = (a, b) => Math.max(b.x0 - a.x1, a.x0 - b.x1, b.y0 - a.y1, a.y0 - b.y1);
  const box = (x0, y0, w, h) => ({x0, y0, x1: x0 + w, y1: y0 + h});
  // Global border-box sizing: the pill is exactly .seat height tall.
  assert.match(css, /^[\s\S]*?\* \{\s*box-sizing: border-box;/);
  const seatHeight = px(".seat", "height");
  const [, padRight, , padLeft] = rule(".seat").match(/padding:\s*(\d+)px (\d+)px (\d+)px (\d+)px/).slice(1).map(Number);
  const seatChrome = 2 * px(".seat", "border") + padLeft + padRight + px(".count", "width") + px(".seat", "gap");
  // Badge: answer-only panel = border + band + body padding + pick box.
  const border = px(".reason", "border");
  const [bodyTop, bodySide, bodyBottom] = rule(".reason-body").match(/padding:\s*(\d+)px (\d+)px (\d+)px/).slice(1).map(Number);
  const badgeW = 2 * border + 2 * bodySide + px(".pick", "width");
  const badgeH = 2 * border + px(".reason-band", "height") + bodyTop + bodyBottom + px(".pick", "height");
  const laneBottom = px(".lane-south", "top") + Number(rule(".lane-south").match(/grid-template-rows:\s*(\d+)px/)[1]);
  const laneGap = px(".seat-lane", "column-gap");
  assert.match(rule(".lane-south.is-reveal .lane-end"), /align-self:\s*end/);
  assert.match(rule(".lane-end"), /justify-self:\s*start/);
  assert.match(rule(".seat-lane .reveal-reason"), /position:\s*relative/);
  assert.ok(laneGap >= MIN_GAP, `lane column gap ${laneGap}`);
  // The raised hand card (pre-rotation size and raise come from CSS and the rendered transform).
  const handTop = px(".observer-hand", "top");
  const cardLeft = px(".observer-hand .card", "left");
  const cardW = px(".observer-hand .card", "width");
  const cardH = px(".observer-hand .card", "height");

  const value = bundle();
  // The geometry case models the production answer-only reveal layout.
  value.presentation.presentation.show_justification_text = false;
  value.identity.players.south = "South Example";
  const [, question] = value.scenes;
  // Raise each hand card in turn: outer cards carry the largest rotation and sideways shift.
  for (const code of ["CA", "C9", "SJ", "H7", "HA"]) {
    value.scenes = [value.scenes[0], question, {...question, stage: "reveal", layers: [...question.layers, {id: "analysis", content: {selectedAction: code}}]}];
    const html = render(value, "reveal");
    assert.match(html, /class="seat-lane lane-south is-reveal"[^>]*>(?:(?!seat-lane)[\s\S])*?<div class="lane-side lane-end"><section class="reveal-reason" data-selected-action/, code);
    assert.equal((html.match(/class="reveal-reason"/g) ?? []).length, 1);
    assert.match(html, /class="viewer-identity seat-row row-south is-reveal"/);
    const style = html.match(/<img class="card [^"]*is-recommended[^"]*"[^>]*style="([^"]*)"/)?.[1];
    assert.ok(style, `raised card for ${code}`);
    const [shift, angle, raise] = style.match(/translateX\((-?[\d.]+)px\) rotate\((-?[\d.]+)deg\) translateY\((-?[\d.]+)px\)/).slice(1).map(Number);
    assert.ok(raise < 0, "the card is raised");
    const theta = Math.abs(angle) * Math.PI / 180;
    const rotW = cardW * Math.cos(theta) + cardH * Math.sin(theta);
    const rotH = cardH * Math.cos(theta) + cardW * Math.sin(theta);
    const cx = cardLeft + cardW / 2 + shift + Math.sin(angle * Math.PI / 180) * -raise;
    const cy = handTop + cardH / 2 + Math.cos(theta) * raise;
    const raised = box(cx - rotW / 2, cy - rotH / 2, rotW, rotH);

    for (const nameWidth of [40, 160, 420]) {
      const pillW = seatChrome + nameWidth;
      const pill = (centerX, top) => box(centerX - pillW / 2, top, pillW, seatHeight);
      const southPill = pill(W / 2, px(".row-south.is-reveal", "top"));
      const badge = box(W / 2 + pillW / 2 + laneGap, laneBottom - badgeH, badgeW, badgeH);
      const pills = {
        south: southPill,
        north: pill(W / 2, px(".row-north", "top")),
        west: pill(px(".row-west", "left") + px(".row-west", "width") / 2, px(".row-west", "top")),
        east: pill(px(".row-east", "left") + px(".row-east", "width") / 2, px(".row-east", "top")),
      };
      for (const [seat, seatPill] of Object.entries(pills)) {
        assert.ok(gap(badge, seatPill) >= MIN_GAP, `${code} name ${nameWidth}: badge vs ${seat} pill gap ${gap(badge, seatPill)}`);
      }
      assert.ok(gap(badge, raised) >= MIN_GAP, `${code} name ${nameWidth}: badge vs raised card gap ${gap(badge, raised)}`);
      assert.ok(gap(southPill, raised) >= MIN_GAP, `${code} name ${nameWidth}: pill vs raised card gap ${gap(southPill, raised)}`);
      assert.equal(southPill.y1, laneBottom, "pill and badge share the lane bottom");
    }
  }
});

test("play prompts and east name pills have clearance from role tags", () => {
  const z = (selector) => Number(rule(selector).match(/z-index:\s*(\d+)/)[1]);
  assert.ok(z(".lane-south") < z(".decision-prompt"));
  assert.ok(z(".lane-south") < z(".reveal-reason"));
  const value = bundle();
  value.scenes[0].layers[0].content.dealer = "south";
  value.timeline.segments = [{stage: "hook", startFrame: 0, durationFrames: 60},
    {stage: "question", startFrame: 60, durationFrames: 60}, {stage: "reveal", startFrame: 120, durationFrames: 60}];
  for (const frame of [30, 90]) {
    assert.match(frameAt(value, frame), /class="seat-lane lane-south is-play-prompt"[^>]*><div[^>]*><div class="seat-tags tags-south"><span class="tag dealer"/);
  }
  assert.doesNotMatch(frameAt(value, 150), /is-play-prompt/);

  const theta = Number(rule(".pcard").match(/rotate\((-?\d+)deg\)/)[1]) * Math.PI / 180;
  const cardHalfHeight = (px(".pcard", "height") * Math.cos(theta) + px(".pcard", "width") * Math.abs(Math.sin(theta))) / 2;
  const tagHalfHeight = px(".tag", "height") / 2;
  const southBottom = px(".lane-south.is-play-prompt", "top") + tagHalfHeight + cardHalfHeight;
  assert.ok(southBottom < px(".decision-prompt", "top"), "south purchase card stays above the prompt");
  const eastTop = px(".tags-east", "top") + px(".tags-east > .tag.buyer:has(.pcard)", "margin-top") + tagHalfHeight - cardHalfHeight;
  assert.ok(eastTop > px(".row-east", "top") + px(".seat", "height"), "east purchase card stays below the name");
});

test("north tray meets the pill-row bottom", () => {
  assert.match(rule(".lane-north .ptray"), /align-self:\s*end/);
  const laneTop = Number(rule(".lane-north").match(/top:\s*(\d+)px/)[1]);
  const rowHeight = Number(rule(".lane-north").match(/grid-template-rows:\s*(\d+)px/)[1]);
  const tray = rule(".ptray");
  const [padTop, , padBottom] = tray.match(/padding:\s*(\d+)px (\d+)px (\d+)px/).slice(1).map(Number);
  const border = Number(tray.match(/border:\s*(\d+)px/)[1]);
  const cardHeight = Number(rule(".tray-card").match(/height:\s*(\d+)px/)[1]);
  const bottom = laneTop + rowHeight;
  const top = bottom - (padTop + padBottom + 2 * border + cardHeight);
  assert.ok(bottom <= 540, `tray bottom ${bottom} must stay above the rug top 540`);
  assert.ok(Math.abs(top - 420) <= 5, `tray top ${top} should sit near the design's 420`);
});

test("the calibration table qualifies exact red notice checkpoints and excludes planted contacts", async () => {
  const table = await loadLayoutTable();
  const widened = JSON.parse(table.bytes);
  widened.rows[0].observedMaxWidth = 200;
  const bytes = Buffer.from(JSON.stringify(widened));
  const file = new URL("widened-table.json", out);
  writeFileSync(file, bytes);
  const override = {file, expectedSha256: sha256(bytes)};
  await assert.rejects(loadLayoutTable(override), /require NEVER_POST measurement mode/);
  const measurementTable = await loadLayoutTable({measure: true, ...override});
  assert.equal(measurementTable.tableSha256, override.expectedSha256);
  for (const phase of ["hook", "question", "reveal"]) {
    const reveal = phase === "reveal";
    const rect = (left, top, width, height) => ({left, top, right: left + width, bottom: top + height});
    const name = rect(200, 400, 180, 40);
    const groups = {
      promptPill: reveal ? [] : [rect(400, 280, 200, 100)],
      namePills: [rect(40, 400, 100, 40), name, rect(900, 400, 100, 40), ...(reveal ? [rect(750, 400, 100, 40)] : [])],
      buyerBadge: [rect(200, 436.1748046875, 167.578125, 50)],
      dealerBadge: [rect(150, 500, 100, 40)],
      choiceBox: reveal ? [rect(400, 630, 200, 60)] : [],
      researchNotice: [rect(40, 40, 1000, 116)],
      broadcastHud: [rect(40, 175, 1000, 50)],
    };
    const roles = Object.fromEntries(Object.entries(groups).map(([role, rectangles]) => [role, {
      matchedCount: rectangles.length,
      items: rectangles.map((bounds, index) => ({id: `${role}:${index}`, visible: true, ...bounds})),
    }]));
    const binding = {mode: "qualification", tableSha256: table.tableSha256};
    const expected = {phase, frame: 30, width: 1080, height: 1920, binding, playDecision: true};
    const row = {...expected, schemaVersion: 1, measurementMethod: "DOM.getBoundingClientRect",
      coordinateSpace: "composition_pixels", rootRect: {left: 0, top: 0, width: 1080, height: 1920}, roles};
    const result = checkLayoutGeometry(row, expected, table);
    assert.equal(result.status, "passed");
    assert.equal(result.tableSha256, sha256(table.bytes));
    assert.deepEqual(result.designedAdjacencies.map(({width, height}) => [width, height]), [[167.578125, 3.8251953125]]);
    const widenedBinding = {...binding, tableSha256: measurementTable.tableSha256};
    assert.throws(() => checkLayoutGeometry(
      {...row, binding: widenedBinding}, {...expected, binding: widenedBinding}, measurementTable,
    ), /approved layout table SHA-256 mismatch/);
    const measurementBinding = {...widenedBinding, mode: "measurement"};
    assert.equal(checkLayoutGeometry(
      {...row, binding: measurementBinding}, {...expected, binding: measurementBinding}, measurementTable,
    ).status, "measured");
    for (const [role, bounds] of [
      ["broadcastHud", rect(40, 155, 1000, 50)],
      ["dealerBadge", name],
      ["buyerBadge", rect(200, 430, 167.578125, 50)],
      ["buyerBadge", rect(200, 436.1748046875, 170, 50)],
    ]) {
      const altered = structuredClone(row);
      Object.assign(altered.roles[role].items[0], bounds);
      assert.throws(() => checkLayoutGeometry(altered, expected, table), /uncalibrated intersections/);
      const measureBinding = {...binding, mode: "measurement"};
      assert.equal(checkLayoutGeometry(
        {...altered, binding: measureBinding}, {...expected, binding: measureBinding}, table,
      ).status, "measured");
      assert.throws(() => checkLayoutGeometry({...altered, binding: measureBinding}, expected, table), /binding mismatch/);
    }
    const taller = structuredClone(row);
    taller.roles.researchNotice.items[0].bottom += 1;
    assert.throws(() => checkLayoutGeometry(taller, expected, table), /notice band height/);
    const tableSha256 = "0".repeat(64);
    assert.throws(() => checkLayoutGeometry(row, {...expected, binding: {...binding, tableSha256}}, table), /binding mismatch/);
  }
});

test("measurement overrides require an exact SHA-256", async () => {
  const table = await loadLayoutTable();
  const file = new URL("reformatted-table.json", out);
  writeFileSync(file, `${table.bytes.toString().trim()}  \n`);
  await assert.rejects(loadLayoutTable({measure: true, file}), /requires its SHA-256 pin/);
  await assert.rejects(loadLayoutTable({measure: true, file, expectedSha256: table.tableSha256}), /SHA-256 mismatch/);
  await assert.rejects(loadLayoutTable({expectedSha256: table.tableSha256}), /require NEVER_POST measurement mode/);
});

test("qualification rejects a changed default calibration without environment overrides", async () => {
  const isolated = new URL("changed-default-layout/", out);
  mkdirSync(isolated, {recursive: true});
  const gate = new URL("layout-gate.mjs", isolated);
  writeFileSync(gate, readFileSync(new URL("../scripts/layout-gate.mjs", import.meta.url)));
  const approved = await loadLayoutTable();
  writeFileSync(new URL("approved-layout-adjacencies.json", isolated), Buffer.concat([approved.bytes, Buffer.from("\n")]));
  const {loadLayoutTable: loadChangedDefault} = await import(gate.href);
  await assert.rejects(loadChangedDefault(), /layout table SHA-256 mismatch/);
});
