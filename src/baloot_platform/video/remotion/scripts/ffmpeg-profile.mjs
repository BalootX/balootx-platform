export const bt709FfmpegOverride = ({args}) => {
  if (process.env.CI === "true") {
    args = [
      "-filter_threads", "4", "-filter_complex_threads", "4",
      ...args.slice(0, -1).flatMap(arg => arg === "-i" ? ["-threads", "4", arg] : [arg]),
      "-threads", "4", args.at(-1),
    ];
  }
  return [
    ...args.slice(0, -1),
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-colorspace",
    "bt709",
    "-color_range",
    "tv",
    "-x264-params",
    "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
    args.at(-1),
  ];
};
