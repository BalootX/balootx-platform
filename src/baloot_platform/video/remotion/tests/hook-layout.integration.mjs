import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import {canonical, layoutFrames, loadLayoutTable, sha256} from "../scripts/layout-gate.mjs";

// The hosted media job supplies the generated public prototype. The cases are
// synthetic display states using its assets, Arabic copy and house typography.
const prototype = JSON.parse(readFileSync(process.argv[2], "utf8"));
const table = await loadLayoutTable();
const work = mkdtempSync(path.join(tmpdir(), "baloot-hook-layout-"));
const seats = ["north", "west", "south", "east"];
const layer = (bundle, stage, id) => bundle.scenes.find(scene => scene.stage === stage)
  .layers.find(item => item.id === id).content;
const cards = layer(prototype, "context", "public-lifecycle").plays.map(play => play.card);

const cases = [
  {name: "viewer-dealer", observer: "west", buyer: "east", dealer: "west", contract: "sun", pause: 27},
  {name: "viewer-dealer-east-buyer", observer: "west", buyer: "south", dealer: "west", contract: "ashkal", pause: 27},
  {name: "east-dealer-and-buyer", observer: "north", buyer: "west", dealer: "west", contract: "sun", pause: 27},
  {name: "west-buyer", observer: "south", buyer: "west", dealer: "east", contract: "sun", pause: 26},
  {name: "uncalibrated-west-contact", observer: "south", buyer: "west", dealer: "east", contract: "sun", pause: 26, reject: true},
];

try {
  for (const {name, observer, buyer, dealer, contract, pause, reject} of cases) {
    const {bundleSha256: _, ...fixture} = structuredClone(prototype);
    fixture.observerSeat = observer;
    fixture.identity.players = {north: "Zayd", west: reject ? "Alexandria Example" : "Khalid", south: "Omar", east: "Salem"};
    fixture.presentation.majlis.research_notice = true;
    fixture.presentation.presentation.show_justification_text = false;
    fixture.audioCues = [];
    fixture.scenes.forEach(scene => { scene.observer = observer; });
    const turnOrder = seats.map((_, index) => seats[(seats.indexOf(observer) + index + 1) % seats.length]);
    const trickPlays = turnOrder.slice(0, pause % 4).map((seat, index) => ({seat, card: cards[index]}));
    const hands = Object.fromEntries(turnOrder.map((seat, index) => [seat, 2 - Number(index < trickPlays.length)]));
    Object.assign(layer(fixture, "context", "public-lifecycle"), {
      buyer, dealer, contract, purchaseCard: cards[4], bidHistory: [{seat: buyer, action: contract, round: 1}],
      pauseBeforePlayIndex: pause,
    });
    Object.assign(layer(fixture, "question", "table"), {buyer, dealer, contract, currentPlayer: observer, currentTrick: {plays: trickPlays}});
    Object.assign(layer(fixture, "question", "observer-hand"), {seat: observer, cards: cards.slice(5, 7)});
    layer(fixture, "question", "seats").handCounts = hands;
    layer(fixture, "question", "decision-prompt").actingSeat = observer;

    const input = path.join(work, `${name}.json`);
    const output = path.join(work, name);
    writeFileSync(input, JSON.stringify({...fixture, bundleSha256: sha256(JSON.stringify(canonical(fixture)))}));
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
    assert.equal(item.pngSha256, sha256(readFileSync(path.join(output, "layout-hook.png"))));
    assert.equal(item.validation.status, reject ? "failed" : "passed");
    const pairs = item.validation.disallowedIntersections.map(({first, second}) => [first, second]);
    assert.deepEqual(pairs, reject ? [["namePills:1", "buyerBadge:0"]] : []);
    process.stdout.write(`${name}: ${item.validation.status}\n`);
  }
} finally {
  rmSync(work, {recursive: true, force: true});
}
