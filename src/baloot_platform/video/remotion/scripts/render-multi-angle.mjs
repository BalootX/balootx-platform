import {mkdir, readFile} from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {bundle} from "@remotion/bundler";
import {renderMedia, renderStill, selectComposition} from "@remotion/renderer";
import {bt709FfmpegOverride} from "./ffmpeg-profile.mjs";
import {governedH264Options} from "./render-options.mjs";

const rendererRoot = process.cwd();
const inputPath = process.env.BALOOTX_MULTI_ANGLE_INPUT;
const outputDir = process.env.BALOOTX_RENDER_OUTPUT;
const assetsRoot = process.env.BALOOTX_MULTI_ANGLE_PUBLIC;
const outputBase = process.env.BALOOTX_RENDER_BASENAME ?? "multi-angle";
if (!inputPath || !outputDir || !assetsRoot) {
  throw new Error("Multi-angle rendering requires governed input, output, and public roots");
}
const inputProps = JSON.parse(await readFile(inputPath, "utf8"));
const mediaRenderOptions = governedH264Options(inputProps.render);
const chromiumOptions = process.platform === "linux" ? {enableMultiProcessOnLinux: true} : undefined;
await mkdir(outputDir, {recursive: true});
const serveUrl = await bundle({
  entryPoint: path.join(rendererRoot, "src/index.ts"),
  publicDir: assetsRoot,
  onProgress: () => undefined,
});
const composition = await selectComposition({
  serveUrl,
  id: "MultiAngleComposition",
  inputProps,
  chromiumOptions,
});
let start = 0;
for (const segment of inputProps.plan.segments) {
  const duration = segment.sourceEndFrame - segment.sourceStartFrame;
  const sample = start + Math.floor(duration / 2);
  await renderStill({
    composition,
    serveUrl,
    output: path.join(outputDir, `checkpoint-${String(segment.ordinal).padStart(2, "0")}.png`),
    inputProps,
    imageFormat: "png",
    frame: sample,
    overwrite: true,
    chromiumOptions,
  });
  start += duration;
}
await renderMedia({
  composition,
  serveUrl,
  outputLocation: path.join(outputDir, `${outputBase}.mp4`),
  inputProps,
  ...mediaRenderOptions,
  colorSpace: "bt709",
  ffmpegOverride: bt709FfmpegOverride,
  disallowParallelEncoding: true,
  audioCodec: "aac",
  audioBitrate: "192k",
  overwrite: true,
  chromiumOptions,
});
