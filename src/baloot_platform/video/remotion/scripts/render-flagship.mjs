import {mkdir} from "node:fs/promises";
import {availableParallelism, cpus} from "node:os";
import path from "node:path";
import process from "node:process";
import {setTimeout as delay} from "node:timers/promises";
import {bundle} from "@remotion/bundler";
import {openBrowser, renderMedia, renderStill, selectComposition} from "@remotion/renderer";
import {bt709FfmpegOverride} from "./ffmpeg-profile.mjs";
import {writeFileWithRetry} from "./fs-safe.mjs";
import {trickCollectionSampleFrameRange} from "./playback-timing.mjs";
import {cgroupCpuLimit, governedH264Options, hostSafeConcurrency, readValidatedRenderBundle} from "./render-options.mjs";

const rendererRoot = process.cwd();
const repositoryRoot = path.resolve(rendererRoot, "../../../..");
const assetsRoot = process.env.BALOOTX_ASSETS_ROOT ?? path.join(repositoryRoot, "assets");
const inputPath = process.env.BALOOTX_RENDER_BUNDLE ?? process.argv[2] ?? path.join(repositoryRoot, "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json");
const outputDir = process.env.BALOOTX_RENDER_OUTPUT ?? process.argv[3] ?? path.join(repositoryRoot, "outputs/professional-video-flagship");
const outputBase = process.env.BALOOTX_RENDER_BASENAME ?? "balootx-complete-laffah-audit";
const outputPath = path.join(outputDir, `${outputBase}.mp4`);
const stillsOnly = process.env.BALOOTX_STILLS_ONLY === "1";
const collectionOnly = process.env.BALOOTX_COLLECTION_ONLY === "1";
const batchRender = process.env.BALOOTX_BATCH_RENDER === "1";
const playbackEvidence = process.env.BALOOTX_PLAYBACK_EVIDENCE === "1";
const chromiumOptions = process.platform === "linux" ? {enableMultiProcessOnLinux: true} : undefined;
const bundleData = readValidatedRenderBundle(inputPath);
const collectionFrameRange = collectionOnly ? trickCollectionSampleFrameRange(bundleData) : undefined;
const mediaRenderOptions = governedH264Options(bundleData.render);
const sampleHostCpu = async () => {
  let previous = cpus();
  const busy = [];
  for (let sample = 0; sample < 3; sample += 1) {
    await delay(500);
    const current = cpus();
    let elapsed = 0;
    let idle = 0;
    for (let index = 0; index < current.length; index += 1) {
      const before = previous[index].times;
      const after = current[index].times;
      elapsed += Object.values(after).reduce((sum, value) => sum + value, 0)
        - Object.values(before).reduce((sum, value) => sum + value, 0);
      idle += after.idle - before.idle;
    }
    busy.push(elapsed > 0 ? 1 - idle / elapsed : 0);
    previous = current;
  }
  return {logicalCpus: previous.length, sustainedBusy: Math.min(...busy)};
};
const inputProps = {
  bundle: bundleData,
  stage: "flagship",
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
const captionText = showJustification
  ? bundleData.localizations[analysisLayer?.content?.reasonKey ?? "video.analysis.expert_reason"]
  : "";
if (showJustification && !captionText) throw new Error("Missing governed reveal localization");
const captionStart = segment("reveal").startFrame;
await writeFileWithRetry(
  path.join(outputDir, `${outputBase}.ar.vtt`),
  showJustification
    ? `WEBVTT\n\n${timestamp(captionStart)} --> ${timestamp(captionStart + 240)}\n${captionText}\n`
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
const composition = await selectComposition({serveUrl, id: "ProfessionalFlagship", inputProps, chromiumOptions, puppeteerInstance: browser});
const replaySegment = timeline.segments.find(item => item.stage === "replay");
const openingPlaybackSegment = replaySegment ?? segment("continuation");
const openingCheckpoint = (playOffset, nonplayProgress) => replaySegment
  ? replaySegment.startFrame + playOffset
  : openingPlaybackSegment.startFrame + Math.floor(openingPlaybackSegment.durationFrames * nonplayProgress);
const completeCheckpoints = [
  ["01-hook", 0],
  ["01b-rewind", segment("rewind").startFrame + 10],
  ["02-opening-play", openingPlaybackSegment.startFrame],
  ["02-project-mention-cleared", openingCheckpoint(392, 0.14)],
  ["02a-second-trick-start", openingCheckpoint(492, 0.18)],
  ["02b-project-reveal", openingCheckpoint(692, 0.25)],
  ["02b2-project-reveal-late", openingCheckpoint(942, 0.34)],
  ["02c-project-reveal-cleared", openingCheckpoint(1042, 0.38)],
  ["03-mid-play", openingPlaybackSegment.startFrame + Math.floor(openingPlaybackSegment.durationFrames / 2)],
  ["04-critical-pause", segment("question").startFrame + 120],
  ["05-reveal", segment("reveal").startFrame + 120],
  ["06-continuation", segment("continuation").startFrame + Math.floor(segment("continuation").durationFrames * 0.28)],
  ["06a-sacrifice", segment("continuation").startFrame + Math.floor(segment("continuation").durationFrames * 0.05)],
  ["06b-final-trick", segment("continuation").startFrame + Math.floor(segment("continuation").durationFrames * 0.92)],
  ["07-outcome", segment("outcome").startFrame + 180],
];
const checkpoints = batchRender
  ? completeCheckpoints.filter(([name]) => ["01-hook", "02-opening-play", "02b-project-reveal", "04-critical-pause", "05-reveal", "07-outcome"].includes(name))
  : completeCheckpoints;
const continuationScene = bundleData.scenes.find(scene => scene.stage === "continuation");
const publicLifecycle = continuationScene?.layers.find(layer => layer.id === "public-lifecycle")?.content;
const allPlays = Array.isArray(publicLifecycle?.plays)
  ? publicLifecycle.plays
  : Array.isArray(publicLifecycle?.completedTricks)
    ? publicLifecycle.completedTricks.flatMap(trick => trick.plays ?? [])
    : [];
const pausePlayIndex = Number(publicLifecycle?.pauseBeforePlayIndex ?? 0);
const hasContextPlayback = timeline.segments.some(item => item.stage === "context" && item.durationFrames > 0);
const playbackFrame = completedPlayCount => {
  if (completedPlayCount === 0) return segment("hook").startFrame;
  if (completedPlayCount < pausePlayIndex || completedPlayCount === pausePlayIndex && !hasContextPlayback) {
    const replay = segment("replay");
    const count = Math.max(1, pausePlayIndex - Number(hasContextPlayback));
    return replay.startFrame + Math.min(
      replay.durationFrames - 1,
      Math.max(0, Math.ceil(completedPlayCount * replay.durationFrames / count) - 1),
    );
  }
  if (completedPlayCount === pausePlayIndex) {
    const context = segment("context");
    return context.startFrame + context.durationFrames - 1;
  }
  const continuation = segment("continuation");
  const count = Math.max(1, allPlays.length - pausePlayIndex);
  return continuation.startFrame + Math.min(
    continuation.durationFrames - 1,
    Math.max(0, Math.ceil((completedPlayCount - pausePlayIndex) * continuation.durationFrames / count) - 1),
  );
};
if (!collectionOnly) for (const [name, frame] of checkpoints) {
  await renderStill({
    composition,
    serveUrl,
    output: path.join(outputDir, `${name}.png`),
    inputProps,
    imageFormat: "png",
    frame,
    overwrite: true,
    chromiumOptions,
    puppeteerInstance: browser,
  });
}
if (playbackEvidence) {
  if (allPlays.length !== 32 || !Number.isInteger(pausePlayIndex) || pausePlayIndex < 1 || pausePlayIndex >= 32) {
    throw new Error("playback evidence requires one complete 32-card play trace and an interior decision");
  }
  const items = [];
  const outcome = segment("outcome");
  const playbackCheckpoints = [
    ...Array.from({length: allPlays.length + 1}, (_, completedPlayCount) => ({
      completedPlayCount, frame: playbackFrame(completedPlayCount),
      source: `playback-${String(completedPlayCount).padStart(2, "0")}.png`,
    })),
    {phase: "outcome", completedPlayCount: allPlays.length,
      frame: outcome.startFrame + Math.floor(outcome.durationFrames / 2), source: "playback-outcome.png"},
  ];
  for (const checkpoint of playbackCheckpoints) {
    const {completedPlayCount, frame, source: name} = checkpoint;
    let renderedDom;
    await renderStill({
      composition: {...composition, props: {...composition.props, capturePlaybackDom: true}},
      serveUrl,
      output: path.join(outputDir, name),
      inputProps: {...inputProps, capturePlaybackDom: true},
      onArtifact: artifact => {
        if (artifact.filename === "playback-dom.json") renderedDom = JSON.parse(String(artifact.content));
      },
      imageFormat: "png",
      frame,
      overwrite: true,
      chromiumOptions,
      puppeteerInstance: browser,
    });
    if (!renderedDom || renderedDom.frame !== frame || renderedDom.completedPlayCount !== completedPlayCount) {
      throw new Error(`Playback DOM checkpoint identity mismatch: ${JSON.stringify({frame, completedPlayCount, renderedDom})}`);
    }
    items.push({...checkpoint, renderedDom});
  }
  await writeFileWithRetry(
    path.join(outputDir, "playback-evidence.json"),
    `${JSON.stringify({schemaVersion: 1, items}, null, 2)}\n`,
    "utf8",
  );
}
if (!stillsOnly) {
  const {logicalCpus, sustainedBusy} = await sampleHostCpu();
  mediaRenderOptions.concurrency = hostSafeConcurrency(
    mediaRenderOptions.concurrency, logicalCpus, sustainedBusy, Math.min(availableParallelism(), cgroupCpuLimit()),
  );
  // ponytail: pre-render pressure guard; chunked rendering only if mid-render pressure proves material.
  process.stderr.write(`Remotion concurrency ${mediaRenderOptions.concurrency}/${bundleData.render.concurrency} on ${logicalCpus} CPUs (sustained busy ${Math.round(sustainedBusy * 100)}%)\n`);
}
if (collectionOnly) {
  const samplePath = path.join(outputDir, "trick-collection-sample.mp4");
  await renderMedia({
    composition,
    serveUrl,
    outputLocation: samplePath,
    inputProps,
    ...mediaRenderOptions,
    colorSpace: "bt709",
    ffmpegOverride: bt709FfmpegOverride,
    disallowParallelEncoding: true,
    frameRange: collectionFrameRange,
    overwrite: true,
    chromiumOptions,
    puppeteerInstance: browser,
  });
  process.stdout.write(`${samplePath}\n`);
}
if (!stillsOnly && !collectionOnly) {
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
