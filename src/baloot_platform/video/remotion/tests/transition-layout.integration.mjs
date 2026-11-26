import assert from "node:assert/strict";
import {mkdir, readFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {bundle} from "@remotion/bundler";
import {openBrowser, renderStill, selectComposition} from "@remotion/renderer";
import {canonical, checkLayoutGeometry, layoutFrames, loadLayoutTable, pngHash, propsSha256, sha256} from "../scripts/layout-gate.mjs";
import {parseValidatedRenderBundle} from "../scripts/render-options.mjs";
import {createRenderOutput, writeRenderEvidence} from "../scripts/render-output.mjs";
import {questionTimer} from "../src/playback-state.ts";

const [inputPath, output] = process.argv.slice(2);
const repository = fileURLToPath(new URL("../../../../../", import.meta.url));
const {bundleSha256: _, ...fixture} = JSON.parse(await readFile(inputPath, "utf8"));
const names = JSON.parse(await readFile(path.join(repository, "examples/synthetic-clip-identity.json"), "utf8"));
fixture.identity.players = names.default_player_names;
fixture.presentation.majlis.research_notice = true;
fixture.presentation.presentation.show_justification_text = false;
fixture.audioCues = [];
const bytes = Buffer.from(JSON.stringify({...fixture, bundleSha256: sha256(JSON.stringify(canonical(fixture)))}));
const bundleData = parseValidatedRenderBundle(bytes);
const table = await loadLayoutTable();
const {width, height, fps} = bundleData.render;
const {decisionKind = "play"} = bundleData.scenes.find(({stage}) => stage === "reveal")
  .layers.find(({id}) => id === "analysis").content;
assert.equal(decisionKind, "play");

const steady = layoutFrames(bundleData);
const checkpoints = new Map(steady.map(row => [row.frame, {...row, samples: ["steady"]}]));
function addFrame(phase, frame, sample) {
  if (!checkpoints.has(frame)) checkpoints.set(frame, {phase, frame, samples: []});
  checkpoints.get(frame).samples.push(sample);
}
let overlapFrame;
for (const {phase} of steady) {
  const {startFrame, durationFrames} = bundleData.timeline.segments.find(({stage}) => stage === phase);
  addFrame(phase, startFrame, "phase-entry");
  addFrame(phase, startFrame + durationFrames - 1, "phase-exit");
  if (phase !== "question") continue;
  for (let offset = 1; offset < durationFrames; offset++) {
    if (questionTimer(durationFrames, offset, fps) === questionTimer(durationFrames, offset - 1, fps)) continue;
    const frame = startFrame + offset;
    addFrame(phase, frame - 1, "before-countdown-change");
    addFrame(phase, frame, "countdown-change");
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
await mkdir(path.dirname(output), {recursive: true});
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
          },
        });
        Object.assign(row, {geometry, pngSha256: pngHash(await readFile(stillPath), width, height)});
        row.validation = checkLayoutGeometry(geometry, {...checkpoint, width, height, playDecision: true, binding}, table);
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
  const oldFrames = steady.map(({frame}) => checkpoints.get(frame));
  const measured = await measure(
    "moving-overlap", [...oldFrames, checkpoints.get(overlapFrame)], overlapFrame,
  );
  const baseline = measured.slice(0, -1);
  const transition = measured.at(-1);
  assert.ok(baseline.every(row => row.validation.status === "passed"), JSON.stringify(baseline));
  assert.equal(transition.validation.status, "failed");
  assert.deepEqual(transition.validation.disallowedIntersections?.map(({first, second}) => [first, second]),
    [["promptPill:0", "researchNotice:0"]]);
  console.log("Moving overlap: steady sampling passed; transition sampling detected prompt/notice contact.");

  const decisionFrames = [...checkpoints.values()].sort((first, second) => first.frame - second.frame);
  const decisionRows = await measure("decision-clip", decisionFrames);
  assert.deepEqual(decisionRows.filter(({validation}) => validation.status !== "passed")
    .map(({phase, frame, validation}) => ({phase, frame, validation})), [],
  "decision-clip geometry failed; see decision-clip-layout-evidence.json");
} finally {
  await browser.close({silent: true});
}
