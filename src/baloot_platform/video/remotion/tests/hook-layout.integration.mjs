import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {copyFileSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import {canonical, layoutFrames, loadLayoutTable, sha256} from "../scripts/layout-gate.mjs";

// The hosted media job supplies the generated public prototype. The cases are
// synthetic display states using its assets, Arabic copy and house typography.
const prototype = JSON.parse(readFileSync(process.argv[2], "utf8"));
const framesDir = process.argv[3];
const table = await loadLayoutTable();
const work = mkdtempSync(path.join(tmpdir(), "baloot-hook-layout-"));
const screenshotDir = process.env.BALOOT_HOOK_LAYOUT_EVIDENCE_DIR;
const seats = ["north", "west", "south", "east"];
const layer = (bundle, stage, id) => bundle.scenes.find(scene => scene.stage === stage)
  .layers.find(item => item.id === id).content;
const cards = layer(prototype, "context", "public-lifecycle").plays.map(play => play.card);

const cases = [
  {name: "viewer-dealer", observer: "west", buyer: "east", dealer: "west", contract: "sun", pause: 27},
  {name: "viewer-dealer-east-buyer", observer: "west", buyer: "south", dealer: "west", contract: "ashkal", pause: 27},
  {name: "east-dealer-and-buyer", observer: "north", buyer: "west", dealer: "west", contract: "sun", pause: 27},
  {name: "west-buyer", observer: "south", buyer: "west", dealer: "east", contract: "sun", pause: 26},
  {name: "west-buyer-without-safe-areas", observer: "south", buyer: "west", dealer: "east", contract: "sun", pause: 26, omitSafeAreas: true},
  {name: "viewer-buyer", observer: "south", buyer: "south", dealer: "east", contract: "sun", allPhases: true},
  {name: "wide-west-buyer", observer: "south", buyer: "west", dealer: "east", contract: "sun", pause: 26, wideName: true},
  {name: "uncalibrated-prompt-hud-contact", observer: "south", buyer: "west", dealer: "east", contract: "sun", pause: 26, reject: true},
];

try {
  if (framesDir) mkdirSync(framesDir);
  if (screenshotDir) mkdirSync(screenshotDir, {recursive: true});
  for (const {name, observer, buyer, dealer, contract, pause, allPhases, wideName, reject, omitSafeAreas} of cases) {
    const {bundleSha256: _, ...fixture} = structuredClone(prototype);
    fixture.observerSeat = observer;
    fixture.identity.players = {north: "Zayd", west: wideName ? "Alexandria Example" : "Khalid", south: "Omar", east: "Salem"};
    fixture.presentation.majlis.research_notice = true;
    fixture.presentation.presentation.show_justification_text = false;
    if (omitSafeAreas) delete fixture.presentation.safe_areas;
    // Lift the synthetic south stack until the rendered prompt intersects the HUD.
    if (reject) fixture.presentation.safe_areas.bottom_px = 1310;
    fixture.audioCues = [];
    fixture.scenes.forEach(scene => { scene.observer = observer; });
    const lifecycle = layer(fixture, "context", "public-lifecycle");
    Object.assign(lifecycle, {
      buyer, dealer, contract, purchaseCard: cards[4], bidHistory: [{seat: buyer, action: contract, round: 1}],
    });
    layer(fixture, "question", "decision-prompt").actingSeat = observer;
    if (allPhases) {
      // Question and reveal replay the prototype trace, so keep its hand and answer together.
      const answer = layer(fixture, "reveal", "analysis");
      answer.selectedAction = answer.recommendedAction.card.code;
    } else {
      const turnOrder = seats.map((_, index) => seats[(seats.indexOf(observer) + index + 1) % seats.length]);
      const trickPlays = turnOrder.slice(0, pause % 4).map((seat, index) => ({seat, card: cards[index]}));
      const hands = Object.fromEntries(turnOrder.map((seat, index) => [seat, 2 - Number(index < trickPlays.length)]));
      lifecycle.pauseBeforePlayIndex = pause;
      Object.assign(layer(fixture, "question", "table"), {buyer, dealer, contract, currentPlayer: observer, currentTrick: {plays: trickPlays}});
      Object.assign(layer(fixture, "question", "observer-hand"), {seat: observer, cards: cards.slice(5, 7)});
      layer(fixture, "question", "seats").handCounts = hands;
    }

    const input = path.join(work, `${name}.json`);
    const output = path.join(work, name);
    writeFileSync(input, JSON.stringify({...fixture, bundleSha256: sha256(JSON.stringify(canonical(fixture)))}));
    if (allPhases) {
      const full = spawnSync(process.execPath, ["scripts/render-decision-clip.mjs", input, output + "-all"], {
        encoding: "utf8", env: {...process.env, BALOOTX_STILLS_ONLY: "1"}, timeout: 120_000,
      });
      assert.equal(full.status, 0, full.error?.message ?? full.stderr);
      const report = JSON.parse(readFileSync(path.join(output + "-all", "layout-evidence.json"), "utf8"));
      assert.equal(report.status, "stills_passed");
      assert.deepEqual(report.items.map(({phase}) => phase), ["hook", "question", "reveal"]);
      for (const item of report.items) checkSeatLayout(item, fixture, name);
    }
    const result = spawnSync(process.execPath, ["scripts/render-decision-clip.mjs", "--check-hook", input, output], {
      encoding: "utf8", timeout: 120_000,
    });
    assert.equal(result.status, reject ? 1 : 0, result.error?.message ?? result.stderr);
    assert.deepEqual(readdirSync(output).sort(), ["hook-layout-evidence.json", "layout-hook.png", "layout-table.json"]);
    const report = JSON.parse(readFileSync(path.join(output, "hook-layout-evidence.json"), "utf8"));
    assert.equal(report.scope, "hook");
    assert.equal(report.postingAuthorized, false);
    assert.equal(report.status, reject ? "failed" : "stills_passed");
    assert.equal(report.binding.tableSha256, table.tableSha256);
    assert.equal(report.binding.bundleFileSha256, sha256(readFileSync(input)));
    assert.equal(report.items.length, 1);
    const [item] = report.items;
    assert.equal(item.phase, "hook");
    assert.equal(item.frame, layoutFrames(fixture)[0].frame);
    const frame = path.join(output, "layout-hook.png");
    assert.equal(item.pngSha256, sha256(readFileSync(frame)));
    if (reject) {
      assert.equal(item.validation.status, "failed");
      assert.ok(item.validation.disallowedIntersections.some(({first, second, width, height}) =>
        first === "promptPill:0" && second === "broadcastHud:0" && width > 0 && height > 0));
    } else {
      checkSeatLayout(item, fixture, name);
    }
    if (framesDir && !reject) copyFileSync(frame, path.join(framesDir, `${name}.png`));
    if (screenshotDir) copyFileSync(frame, path.join(screenshotDir, `${name}.png`));
    process.stdout.write(`${name}: ${item.validation.status}\n`);
  }

  const {bundleSha256: _, ...fixture} = structuredClone(prototype);
  fixture.presentation.majlis.research_notice = true;
  assert.equal(fixture.presentation.presentation.show_justification_text, false);
  // The prototype names no selected action, and the gate requires the answer box on a reveal.
  const answer = layer(fixture, "reveal", "analysis");
  answer.selectedAction = answer.recommendedAction.card.code;
  const input = path.join(work, "answer-only-reveal.json");
  const output = path.join(work, "answer-only-reveal");
  writeFileSync(input, JSON.stringify({...fixture, bundleSha256: sha256(JSON.stringify(canonical(fixture)))}));
  const result = spawnSync(process.execPath, ["scripts/render-decision-clip.mjs", input, output], {
    encoding: "utf8", timeout: 120_000, env: {...process.env, BALOOTX_STILLS_ONLY: "1"},
  });
  assert.equal(result.status, 0, result.error?.message ?? result.stderr);
  const report = JSON.parse(readFileSync(path.join(output, "layout-evidence.json"), "utf8"));
  assert.equal(report.status, "stills_passed");
  assert.equal(report.binding.tableSha256, table.tableSha256);
  const reveal = report.items.find(item => item.phase === "reveal");
  assert.equal(reveal?.validation.status, "passed");
  if (screenshotDir) copyFileSync(path.join(output, reveal.source), path.join(screenshotDir, "answer-only-reveal.png"));
  process.stdout.write(`answer-only-reveal: ${report.status}\n`);
} finally {
  rmSync(work, {recursive: true, force: true});
}

function checkSeatLayout(item, fixture, name) {
  const label = `${name} ${item.phase}`;
  const {left_px, right_px, top_px, bottom_px} = fixture.presentation.safe_areas ?? prototype.presentation.safe_areas;
  const controls = ["promptPill", "namePills", "buyerBadge", "dealerBadge", "choiceBox"]
    .flatMap(role => item.roles[role].items.filter(box => box.visible));
  const hand = item.observerCards.filter(box => box.visible);
  assert.equal(item.validation.status, "passed", label);
  assert.deepEqual(item.validation.intersections, [], label);
  assert.equal(item.buyerCardContained, true, label);
  assert.ok(hand.length > 0, `${label}: viewer hand not measured`);
  for (const box of [...controls, ...hand]) {
    assert.ok(box.left >= left_px && box.top >= top_px && box.right <= fixture.render.width - right_px
      && box.bottom <= fixture.render.height - bottom_px, `${label}: out of safe area ${JSON.stringify(box)}`);
  }
  for (const box of controls) {
    assert.ok(hand.every(card => box.right <= card.left || box.left >= card.right || box.bottom <= card.top || box.top >= card.bottom),
      `${label}: ${box.id} overlaps a hand card`);
  }
}
