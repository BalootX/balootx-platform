import {mkdir} from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {bundle} from "@remotion/bundler";
import {renderMedia, renderStill, selectComposition} from "@remotion/renderer";
import {bt709FfmpegOverride} from "./ffmpeg-profile.mjs";
import {governedH264Options, readValidatedRenderBundle} from "./render-options.mjs";

const rendererRoot = process.cwd();
const repositoryRoot = path.resolve(rendererRoot, "../../../..");
const assetsRoot = process.env.BALOOTX_ASSETS_ROOT ?? path.join(repositoryRoot, "assets");
const inputPath = process.argv[2] ?? path.join(repositoryRoot, "outputs/professional-video-prototypes/render-bundle-decision.json");
const outputDir = process.argv[3] ?? path.join(repositoryRoot, "outputs/professional-video-prototypes");
const inputProps = {bundle: readValidatedRenderBundle(inputPath), stage: "normal", reducedMotion: false};
const mediaRenderOptions = governedH264Options(inputProps.bundle.render);
await mkdir(outputDir, {recursive: true});

const serveUrl = await bundle({
  entryPoint: path.join(rendererRoot, "src/index.ts"),
  publicDir: assetsRoot,
  onProgress: () => undefined,
});

for (const stage of ["normal", "hook", "reveal"]) {
  const stageProps = {...inputProps, stage};
  const stillComposition = await selectComposition({serveUrl, id: "ProfessionalPrototype", inputProps: stageProps});
  await renderStill({
    composition: stillComposition,
    serveUrl,
    output: path.join(outputDir, `${stage}.png`),
    inputProps: stageProps,
    imageFormat: "png",
    frame: 0,
    overwrite: true,
  });
}

const motionComposition = await selectComposition({serveUrl, id: "CardMotionSample", inputProps});
await renderMedia({
  composition: motionComposition,
  serveUrl,
  outputLocation: path.join(outputDir, "card-motion.mp4"),
  inputProps: {...inputProps, stage: "normal"},
  ...mediaRenderOptions,
  colorSpace: "bt709",
  ffmpegOverride: bt709FfmpegOverride,
  disallowParallelEncoding: true,
  overwrite: true,
});
const reducedProps = {...inputProps, stage: "normal", reducedMotion: true};
const reducedComposition = await selectComposition({serveUrl, id: "CardMotionSample", inputProps: reducedProps});
await renderMedia({
  composition: reducedComposition,
  serveUrl,
  outputLocation: path.join(outputDir, "card-motion-reduced.mp4"),
  inputProps: reducedProps,
  ...mediaRenderOptions,
  colorSpace: "bt709",
  ffmpegOverride: bt709FfmpegOverride,
  disallowParallelEncoding: true,
  overwrite: true,
});
