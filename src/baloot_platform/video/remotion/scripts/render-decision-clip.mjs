import {mkdir, readFile} from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {bundle} from "@remotion/bundler";
import {openBrowser, renderMedia, renderStill, selectComposition} from "@remotion/renderer";
import {bt709FfmpegOverride} from "./ffmpeg-profile.mjs";
import {writeFileWithRetry} from "./fs-safe.mjs";
import {governedH264Options, parseValidatedRenderBundle} from "./render-options.mjs";
import {checkLayoutGeometry, fileSha256, layoutFrames, layoutMeasurementMarker, loadLayoutTable, pngHash, propsSha256, sha256} from "./layout-gate.mjs";

const rendererRoot = process.cwd();
const repositoryRoot = path.resolve(rendererRoot, "../../../..");
const assetsRoot = process.env.BALOOTX_ASSETS_ROOT ?? path.join(repositoryRoot, "assets");
const args = process.argv.slice(2);
const measure = args[0] === "--measure-layout";
if ((measure && args.length !== 3) || (!measure && (args.length > 2 || args.some(arg => arg.startsWith("--"))))) {
  throw new Error("usage: render-decision-clip.mjs [--measure-layout] [bundle output]");
}
const inputPath = measure ? args[1] : process.env.BALOOTX_RENDER_BUNDLE ?? args[0] ?? path.join(repositoryRoot, "outputs/professional-video-prototypes/render-bundle-decision.json");
const outputDir = measure ? args[2] : process.env.BALOOTX_RENDER_OUTPUT ?? args[1] ?? path.join(repositoryRoot, "outputs/professional-video-decision");
const outputBase = measure ? "NEVER_POST-layout-measurement" : process.env.BALOOTX_RENDER_BASENAME ?? "balootx-decision-clip";
const outputPath = path.join(outputDir, `${outputBase}.mp4`);
const stillsOnly = process.env.BALOOTX_STILLS_ONLY === "1";
const playbackEvidence = !measure && process.env.BALOOTX_PLAYBACK_EVIDENCE === "1";
const chromiumOptions = process.platform === "linux" ? {enableMultiProcessOnLinux: true} : undefined;
const inputBytes = await readFile(inputPath);
const bundleData = parseValidatedRenderBundle(inputBytes);
const qualifyLayout = bundleData.presentation?.majlis?.research_notice === true;
if (measure && (!qualifyLayout || stillsOnly)) throw new Error("layout measurement requires a red notice MP4");
const table = qualifyLayout ? await loadLayoutTable({
  measure,
  file: process.env.BALOOTX_LAYOUT_TABLE,
  expectedSha256: process.env.BALOOTX_LAYOUT_TABLE_SHA256,
}) : null;
const geometryFrames = table ? layoutFrames(bundleData) : [];
const layoutBinding = table && {
  mode: measure ? "measurement" : "qualification",
  bundleFileSha256: sha256(inputBytes),
  bundleSha256: bundleData.bundleSha256,
  tableSha256: table.tableSha256,
};
const layoutEvidence = table && {
  schemaVersion: 1, status: "running", binding: {...layoutBinding, propsSha256: propsSha256(bundleData)}, items: [],
  ...(measure ? {disposition: "NEVER_POST", postingAuthorized: false} : {}),
};
const layoutPath = path.join(outputDir, measure ? "layout-measurement.json" : "layout-evidence.json");
const saveLayout = () => writeFileWithRetry(layoutPath, `${JSON.stringify(layoutEvidence, null, 2)}\n`, "utf8");
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

await mkdir(outputDir, {recursive: !measure});
if (table) {
  await writeFileWithRetry(path.join(outputDir, "layout-table.json"), table.bytes);
  await saveLayout();
}
const revealScene = bundleData.scenes.find((scene) => scene.stage === "reveal");
const analysisLayer = revealScene?.layers.find((layer) => layer.id === "analysis");
const playDecision = (analysisLayer?.content?.decisionKind ?? "play") === "play";
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
const layoutStills = geometryFrames.map(({phase, frame}) => [`layout-${phase}`, frame, phase]);
if (!measure) {
  checkpoints.push(...layoutStills.filter(([, frame, phase]) =>
    !checkpoints.some(([, existingFrame, existingPhase]) => frame === existingFrame && phase === existingPhase)));
}
for (const [name, frame, phase, completedPlayCount] of measure ? layoutStills : checkpoints) {
  let renderedDom;
  let geometry;
  const geometryFrame = geometryFrames.find(row => row.frame === frame && row.phase === phase);
  const capturePlayback = playbackEvidence && phase && completedPlayCount !== undefined;
  const stillProps = {
    ...inputProps,
    ...(capturePlayback || geometryFrame ? {capturePlaybackDom: true} : {}),
    ...(geometryFrame ? {layoutBinding} : {}),
  };
  const stillPath = path.join(outputDir, `${name}.png`);
  await renderStill({
    composition: {...composition, props: {...composition.props, ...stillProps}},
    serveUrl,
    output: stillPath,
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
    const {width, height} = bundleData.render;
    const row = {...geometry, source: `${name}.png`, pngSha256: pngHash(await readFile(stillPath), width, height)};
    layoutEvidence.items.push(row);
    try {
      row.validation = checkLayoutGeometry(geometry, {...geometryFrame, width, height, playDecision, binding: layoutEvidence.binding}, table);
    } catch (error) {
      row.validation = error.validation ?? {status: "failed", error: String(error)};
      layoutEvidence.status = "failed";
      await saveLayout();
      throw error;
    }
  }
  if (capturePlayback) {
    if (!renderedDom || renderedDom.frame !== frame || renderedDom.completedPlayCount !== completedPlayCount) {
      throw new Error(`Decision-clip DOM checkpoint mismatch: ${JSON.stringify({phase, frame, completedPlayCount, renderedDom})}`);
    }
    evidence.push({phase, completedPlayCount, frame, source: `${name}.png`, renderedDom});
  }
}
if (table) {
  if (layoutEvidence.items.length !== geometryFrames.length) throw new Error("layout checkpoints missing");
  layoutEvidence.items = geometryFrames.map(({phase}) => layoutEvidence.items.find(row => row.phase === phase));
  layoutEvidence.status = "stills_passed";
  await saveLayout();
}
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
    ffmpegOverride: measure ? ({args}) => {
      const governed = bt709FfmpegOverride({args});
      return [...governed.slice(0, -1), "-metadata", `comment=${layoutMeasurementMarker}`, governed.at(-1)];
    } : bt709FfmpegOverride,
    disallowParallelEncoding: true,
    audioCodec: "aac",
    audioBitrate: "192k",
    overwrite: true,
    chromiumOptions,
    puppeteerInstance: browser,
  });
  if (table) {
    const videoSha256 = await fileSha256(outputPath);
    if (!measure && table.forbidden.includes(videoSha256)) throw new Error("layout measurement MP4 is permanently NEVER_POST");
    Object.assign(layoutEvidence, {status: measure ? "measured" : "passed", videoSha256});
    await saveLayout();
  }
  process.stdout.write(`${outputPath}\n`);
}
} finally {
  await browser.close({silent: true});
}
