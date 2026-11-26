import {readFileSync} from "node:fs";

const profile = new URL("./ffmpeg-runtime.json", import.meta.url);
const threads = JSON.parse(readFileSync(profile, "utf8"))?.threads;
if (!Number.isInteger(threads) || threads < 1 || threads > 2_147_483_647) {
  throw new Error(`${profile.pathname}: threads must be a positive 32-bit integer`);
}
const threadCount = String(threads);

export const bt709FfmpegOverride = ({args}) => [
  "-filter_threads", threadCount, "-filter_complex_threads", threadCount,
  ...args.slice(0, -1).flatMap(arg => arg === "-i" ? ["-threads", threadCount, arg] : [arg]),
  "-threads", threadCount,
  "-color_primaries",
  "bt709",
  "-color_trc",
  "bt709",
  "-colorspace",
  "bt709",
  "-color_range",
  "tv",
  "-x264-params",
  // x264 can choose different predictions for the same frames on different CPUs.
  "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:cpu-independent=1",
  args.at(-1),
];
