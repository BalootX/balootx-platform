import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {bundle} from "@remotion/bundler";
import {openBrowser, renderStill, selectComposition} from "@remotion/renderer";
import {checkLayoutGeometry, layoutFrames, loadLayoutTable, pngHash, propsSha256, sha256} from "../scripts/layout-gate.mjs";
import {parseValidatedRenderBundle} from "../scripts/render-options.mjs";
import {createRenderOutput, writeRenderEvidence} from "../scripts/render-output.mjs";

const [inputPath, output] = process.argv.slice(2);
const repository = fileURLToPath(new URL("../../../../../", import.meta.url));
const bytes = await readFile(inputPath);
const bundleData = parseValidatedRenderBundle(bytes);
const table = await loadLayoutTable();
const {width, height, fps} = bundleData.render;
const {decisionKind, selectedAction} = bundleData.scenes.find(({stage}) => stage === "reveal")
  .layers.find(({id}) => id === "analysis").content;
assert.equal(bundleData.exportPreset, "decision_clip");
assert.equal(decisionKind, "play");
assert.ok(selectedAction, "fixture requires a proved reveal action");

const steady = layoutFrames(bundleData);
const checkpoints = new Map(steady.map(row => [row.frame, {...row, samples: ["steady"]}]));
function addFrame(phase, frame, sample, expectedCountdown) {
  if (!checkpoints.has(frame)) checkpoints.set(frame, {phase, frame, samples: []});
  const checkpoint = checkpoints.get(frame);
  checkpoint.samples.push(sample);
  if (expectedCountdown !== undefined) checkpoint.expectedCountdown = String(expectedCountdown);
}
let overlapFrame;
for (const {phase} of steady) {
  const {startFrame, durationFrames} = bundleData.timeline.segments.find(({stage}) => stage === phase);
  addFrame(phase, startFrame, "phase-entry");
  addFrame(phase, startFrame + durationFrames - 1, "phase-exit");
  if (phase !== "question") continue;
  const endFrame = startFrame + durationFrames;
  for (let frame = startFrame + (durationFrames % fps || fps); frame < endFrame; frame += fps) {
    const secondsLeft = (endFrame - frame) / fps;
    addFrame(phase, frame - 1, "before-countdown-change", secondsLeft + 1);
    addFrame(phase, frame, "countdown-change", secondsLeft);
    overlapFrame ??= frame;
  }
}
assert.ok(overlapFrame !== undefined, "fixture requires a question countdown change");
assert.ok(steady.every(({frame}) => Math.abs(frame - overlapFrame) >= fps / 2),
  "steady samples must fall outside the fixture's motion");

const layoutBinding = {
  mode: "qualification",
  bundleFileSha256: sha256(bytes),
  bundleSha256: bundleData.bundleSha256,
  tableSha256: table.tableSha256,
};
const binding = {...layoutBinding, propsSha256: propsSha256(bundleData)};
await createRenderOutput(output, "synthetic-transition");
await writeRenderEvidence(path.join(output, "bundle.json"), bytes);
await writeRenderEvidence(path.join(output, "layout-table.json"), table.bytes);
const serveUrl = await bundle({
  entryPoint: fileURLToPath(new URL("./transition-layout.fixture.tsx", import.meta.url)),
  publicDir: path.join(repository, "assets"),
  onProgress: () => undefined,
});
const chromiumOptions = {enableMultiProcessOnLinux: true};
const browser = await openBrowser("chrome", {chromiumOptions});

async function measure(name, frames, motionFrame) {
  const fixtureProps = motionFrame === undefined ? {} : {overlapFrame: motionFrame};
  const inputProps = {bundle: bundleData, reducedMotion: false, capturePlaybackDom: true, layoutBinding, ...fixtureProps};
  const composition = await selectComposition({
    serveUrl, id: "SyntheticDecisionClip", inputProps, chromiumOptions, puppeteerInstance: browser,
  });
  const report = {schemaVersion: 1, scope: "synthetic-transition", postingAuthorized: false, binding, fixtureProps, items: []};
  try {
    for (const checkpoint of frames) {
      const {phase, frame} = checkpoint;
      const source = `${name}-${phase}-${frame}.png`;
      const stillPath = path.join(output, source);
      const row = {...checkpoint, source, validation: {status: "failed"}};
      report.items.push(row);
      let geometry;
      try {
        await renderStill({
          composition, serveUrl, frame, inputProps, output: stillPath, imageFormat: "png",
          chromiumOptions, puppeteerInstance: browser,
          onArtifact: ({filename, content}) => {
            if (filename === "layout-geometry.json") geometry = JSON.parse(String(content));
            if (filename === "countdown.txt") row.countdownText = String(content);
          },
        });
        Object.assign(row, {geometry, pngSha256: pngHash(await readFile(stillPath), width, height)});
        if (checkpoint.expectedCountdown !== undefined) {
          assert.equal(row.countdownText, checkpoint.expectedCountdown, `rendered countdown at frame ${frame}`);
        }
        row.validation = checkLayoutGeometry(geometry, {...checkpoint, width, height, playDecision: decisionKind === "play", binding}, table);
      } catch (error) {
        row.validation = error.validation ?? {status: "failed", error: String(error)};
      }
      console.log(`${name} ${phase} frame ${frame} (${checkpoint.samples.join(", ")}): ${row.validation.status}`);
    }
  } finally {
    await writeRenderEvidence(path.join(output, `${name}-layout-evidence.json`), `${JSON.stringify(report, null, 2)}\n`);
  }
  return report.items;
}

try {
  const decisionFrames = [...checkpoints.values()].sort((first, second) => first.frame - second.frame);
  const decisionRows = await measure("decision-clip", decisionFrames);
  const oldFrames = steady.map(({frame}) => checkpoints.get(frame));
  const measured = await measure(
    "moving-overlap", [...oldFrames, checkpoints.get(overlapFrame)], overlapFrame,
  );
  const baseline = measured.slice(0, -1);
  const transition = measured.at(-1);
  assert.ok(baseline.every(row => row.validation.status === "passed"), JSON.stringify(baseline));
  console.log(`Moving overlap: steady sampling passed at frames ${baseline.map(({frame}) => frame).join(", ")}.`);
  assert.equal(transition.validation.status, "failed");
  assert.deepEqual(transition.validation.disallowedIntersections?.map(({first, second}) => [first, second]),
    [["promptPill:0", "researchNotice:0"]]);
  console.log(`Moving overlap: transition sampling rejected frame ${transition.frame} for prompt/notice contact.`);

  assert.deepEqual(decisionRows.filter(({validation}) => validation.status !== "passed"), [],
    "decision-clip geometry failed; see decision-clip-layout-evidence.json");
} finally {
  await browser.close({silent: true});
}
