import {mkdir, readFile} from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {bundle} from "@remotion/bundler";
import {openBrowser, renderMedia, renderStill, selectComposition} from "@remotion/renderer";
import {bt709FfmpegOverride} from "./ffmpeg-profile.mjs";
import {writeFileWithRetry} from "./fs-safe.mjs";
import {governedH264Options} from "./render-options.mjs";
import {validateViewerBuyerAdmission} from "./render-admission.mjs";
import {isPlayDecision, layoutFrames, pngHash, propsHash, sha256, sourceBinding, validateLayoutGeometry} from "./layout-gate.mjs";

const rendererRoot = process.cwd();
const repositoryRoot = path.resolve(rendererRoot, "../../../..");
const assetsRoot = process.env.BALOOTX_ASSETS_ROOT ?? path.join(repositoryRoot, "assets");
const inputPath = process.env.BALOOTX_RENDER_BUNDLE ?? process.argv[2] ?? path.join(repositoryRoot, "outputs/professional-video-prototypes/render-bundle-decision.json");
const outputDir = process.env.BALOOTX_RENDER_OUTPUT ?? process.argv[3] ?? path.join(repositoryRoot, "outputs/professional-video-decision");
const outputBase = process.env.BALOOTX_RENDER_BASENAME ?? "balootx-decision-clip";
const outputPath = path.join(outputDir, `${outputBase}.mp4`);
const stillsOnly = process.env.BALOOTX_STILLS_ONLY === "1";
const playbackEvidence = process.env.BALOOTX_PLAYBACK_EVIDENCE === "1";
const chromiumOptions = process.platform === "linux" ? {enableMultiProcessOnLinux: true} : undefined;
const bundleData = JSON.parse(await readFile(inputPath, "utf8"));
validateViewerBuyerAdmission(bundleData);
const layoutBinding = {...await sourceBinding(repositoryRoot,
  process.env.BALOOTX_LAYOUT_SOURCE_IDENTITY ? JSON.parse(process.env.BALOOTX_LAYOUT_SOURCE_IDENTITY) : undefined),
  bundleFileSha256: sha256(await readFile(inputPath))};
const geometryFrames = layoutFrames(bundleData);
const mediaRenderOptions = governedH264Options(bundleData.render);
const inputProps = {
  bundle: bundleData,
  reducedMotion: false,
};
const timeline = bundleData.timeline;
const segment = stage => {
  const value = timeline.segments.find(item => item.stage === stage);
  if (!value) throw new Error(`missing governed timeline segment: ${stage}`);
  return value;
};
const timestamp = frames => {
  const milliseconds = Math.round((frames * 1000) / bundleData.render.fps);
  const hours = Math.floor(milliseconds / 3600000);
  const minutes = Math.floor((milliseconds % 3600000) / 60000);
  const seconds = Math.floor((milliseconds % 60000) / 1000);
  const millis = milliseconds % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
};

await mkdir(outputDir, {recursive: true});
const revealScene = bundleData.scenes.find((scene) => scene.stage === "reveal");
const analysisLayer = revealScene?.layers.find((layer) => layer.id === "analysis");
const showJustification = bundleData.presentation?.presentation?.show_justification_text !== false;
const captionSegment = segment("reveal");
const captionText = showJustification
  ? bundleData.localizations[analysisLayer?.content?.reasonKey ?? "video.analysis.expert_reason"]
  : "";
if (showJustification && !captionText) throw new Error("Missing governed reveal localization");
await writeFileWithRetry(
  path.join(outputDir, `${outputBase}.ar.vtt`),
  showJustification
    ? `WEBVTT\n\n${timestamp(captionSegment.startFrame)} --> ${timestamp(captionSegment.startFrame + Math.min(240, captionSegment.durationFrames))}\n${captionText}\n`
    : "WEBVTT\n\n",
  "utf8",
);
const serveUrl = await bundle({
  entryPoint: path.join(rendererRoot, "src/index.ts"),
  publicDir: assetsRoot,
  onProgress: () => undefined,
});
const browser = await openBrowser("chrome", {chromiumOptions});
try {
const composition = await selectComposition({serveUrl, id: "ProfessionalDecisionClip", inputProps, chromiumOptions, puppeteerInstance: browser});
const middleFrame = stage => segment(stage).startFrame + Math.floor(segment(stage).durationFrames / 2);
const continuationScene = bundleData.scenes.find(scene => scene.stage === "continuation");
const lifecycle = continuationScene?.layers.find(layer => layer.id === "public-lifecycle")?.content;
const allPlays = lifecycle?.plays ?? lifecycle?.completedTricks?.flatMap(trick => trick.plays ?? []) ?? [];
const pause = lifecycle?.pauseBeforePlayIndex;
if (playbackEvidence && (allPlays.length !== 32 || !Number.isInteger(pause) || pause < 1 || pause >= 32)) {
  throw new Error("decision-clip playback checks require a complete trace and an interior play decision");
}
const completeReplay = timeline.segments.some(row => row.stage === "replay");
const replayCheckpoints = () => {
  // Mirrors _complete_replay_checkpoints in remotion_renderer.py.
  const half = Math.floor(pause / 2);
  const sample = Math.max(1, half - (half % 4 === 0 ? 1 : 0));
  const setupActions = (lifecycle?.bidHistory?.length ?? 0) + 1 + (lifecycle?.escalationHistory?.length ?? 0);
  const [hook, setup, replay] = ["hook", "setup", "replay"].map(segment);
  const toRoundEnd = timeline.segments.some(row => row.stage === "outcome");
  const tail = toRoundEnd ? [
    // First remaining play landed (the answer), then the round's final score.
    ["06-continuation", segment("continuation").startFrame + Math.min(segment("continuation").durationFrames - 1,
      Math.ceil(segment("continuation").durationFrames / (32 - pause)) - 1), "continuation", pause + 1],
    ["07-outcome", middleFrame("outcome"), "outcome", 32],
  ] : [
    ["06-hold", segment("hold").startFrame + segment("hold").durationFrames - 1, "hold", pause + 1],
  ];
  return [
    ["01-hook", hook.startFrame + Math.min(bundleData.render.fps, hook.durationFrames - 1), "hook", pause],
    ["02-setup", setup.startFrame + Math.floor(setup.durationFrames / (2 * Math.max(1, setupActions))), "setup", 0],
    ["03-replay", replay.startFrame + Math.ceil((sample * replay.durationFrames) / pause) - 1, "replay", sample],
    ["04-decision", middleFrame("question"), "question", pause],
    ["05-reveal", middleFrame("reveal"), "reveal", pause],
    ...tail,
  ];
};
const continuation = completeReplay ? undefined : segment("continuation");
const firstContinuationFrame = completeReplay ? 0 : continuation.startFrame + Math.min(continuation.durationFrames - 1,
  Math.max(0, Math.ceil(continuation.durationFrames / Math.max(1, allPlays.length - pause)) - 1));
const checkpoints = completeReplay ? replayCheckpoints() : [
  ["01-hook", 0],
  ["02-rewind", middleFrame("rewind")],
  ["03-context", segment("context").startFrame + segment("context").durationFrames - 1, "context", pause],
  ["04-decision", middleFrame("question"), "question", pause],
  ["05-reveal", middleFrame("reveal"), "reveal", pause],
  ["06-continuation", playbackEvidence ? firstContinuationFrame : middleFrame("continuation"), "continuation", pause + 1],
  ...(playbackEvidence ? [["07-continuation-end", continuation.startFrame + continuation.durationFrames - 1, "continuation_end", 32]] : []),
  ...(playbackEvidence && timeline.segments.some(row => row.stage === "outcome")
    ? [["08-outcome", middleFrame("outcome"), "outcome", 32]] : []),
];
const evidence = [];
const layoutEvidence = {schemaVersion: 1, status: "running", items: []};
const layoutPath = path.join(outputDir, "layout-evidence.json");
const saveLayout = () => writeFileWithRetry(layoutPath, `${JSON.stringify(layoutEvidence, null, 2)}\n`, "utf8");
for (const row of geometryFrames) {
  if (!checkpoints.some(([, frame, phase]) => frame === row.frame && phase === row.phase)) {
    checkpoints.push([`layout-${row.phase}`, row.frame, row.phase]);
  }
}
for (const [name, frame, phase, completedPlayCount] of checkpoints) {
  let renderedDom;
  let geometry;
  const geometryFrame = geometryFrames.find(row => row.frame === frame && row.phase === phase);
  const captureDom = Boolean((playbackEvidence && phase && completedPlayCount !== undefined) || geometryFrame);
  const stillProps = {...inputProps, ...(captureDom ? {capturePlaybackDom: true} : {}), ...(geometryFrame ? {layoutBinding} : {})};
  await renderStill({
    composition: {...composition, props: {...composition.props, ...stillProps}},
    serveUrl,
    output: path.join(outputDir, `${name}.png`),
    inputProps: stillProps,
    onArtifact: artifact => {
      if (artifact.filename === "playback-dom.json") renderedDom = JSON.parse(String(artifact.content));
      if (artifact.filename === "layout-geometry.json") geometry = JSON.parse(String(artifact.content));
    },
    imageFormat: "png",
    frame,
    overwrite: true,
    chromiumOptions,
    puppeteerInstance: browser,
  });
  if (geometryFrame) {
    const row = {...geometry, source: `${name}.png`, pngSha256: pngHash(await readFile(path.join(outputDir, `${name}.png`)), bundleData.render.width, bundleData.render.height)};
    layoutEvidence.items.push(row);
    await saveLayout();
    try {
      row.validation = validateLayoutGeometry(geometry, {...geometryFrame, width: bundleData.render.width, height: bundleData.render.height,
        playDecision: isPlayDecision(bundleData), binding: {...layoutBinding, bundleSha256: bundleData.bundleSha256,
          propsSha256: propsHash(bundleData), reducedMotion: false, capturePlaybackDom: true}});
    } catch (error) {
      row.validation = error.validation ?? {status: "failed", error: String(error)};
      layoutEvidence.status = "failed"; layoutEvidence.error = String(error); await saveLayout(); throw error;
    }
  }
  if (playbackEvidence && phase && completedPlayCount !== undefined) {
    if (!renderedDom || renderedDom.frame !== frame || renderedDom.completedPlayCount !== completedPlayCount) {
      throw new Error(`Decision-clip DOM checkpoint mismatch: ${JSON.stringify({phase, frame, completedPlayCount, renderedDom})}`);
    }
    evidence.push({phase, completedPlayCount, frame, source: `${name}.png`, renderedDom});
  }
}
if (layoutEvidence.items.length !== 3) throw new Error("layout gate missing measured hook/question/reveal");
layoutEvidence.items.sort((a, b) => geometryFrames.findIndex(row => row.phase === a.phase) - geometryFrames.findIndex(row => row.phase === b.phase));
layoutEvidence.status = "passed";
await saveLayout();
if (playbackEvidence) await writeFileWithRetry(path.join(outputDir, "playback-evidence.json"),
  `${JSON.stringify({schemaVersion: 1, items: evidence}, null, 2)}\n`, "utf8");
if (!stillsOnly) {
  await renderMedia({
    composition,
    serveUrl,
    outputLocation: outputPath,
    inputProps,
    ...mediaRenderOptions,
    colorSpace: "bt709",
    ffmpegOverride: bt709FfmpegOverride,
    disallowParallelEncoding: true,
    audioCodec: "aac",
    audioBitrate: "192k",
    overwrite: true,
    chromiumOptions,
    puppeteerInstance: browser,
  });
  process.stdout.write(`${outputPath}\n`);
}
} finally {
  await browser.close({silent: true});
}
