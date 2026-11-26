import assert from "node:assert/strict";
import test from "node:test";
import childProcess from "node:child_process";
import fs from "node:fs";
import {createRequire} from "node:module";
import os from "node:os";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {bt709FfmpegOverride} from "../scripts/ffmpeg-profile.mjs";

test("FFmpeg uses the configured decoder, filter and encoder cap on every host", () => {
  const previous = process.env.CI;
  const args = ["-i", "video.mp4", "-i", "audio.wav", "-c:v", "libx264", "output.mp4"];
  const config = new URL("../scripts/ffmpeg-runtime.json", import.meta.url);
  const threads = String(JSON.parse(fs.readFileSync(config, "utf8")).threads);
  try {
    delete process.env.CI;
    const normal = bt709FfmpegOverride({args});
    assert.deepEqual(normal, [
      "-filter_threads", threads, "-filter_complex_threads", threads,
      "-threads", threads, "-i", "video.mp4", "-threads", threads, "-i", "audio.wav",
      "-c:v", "libx264", "-threads", threads,
      "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
      "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:cpu-independent=1",
      "output.mp4",
    ]);
    process.env.CI = "true";
    assert.deepEqual(bt709FfmpegOverride({args}), normal);
    assert.deepEqual(args, ["-i", "video.mp4", "-i", "audio.wav", "-c:v", "libx264", "output.mp4"]);
  } finally {
    if (previous === undefined) delete process.env.CI;
    else process.env.CI = previous;
  }
});

test("FFmpeg reads a changed cap and rejects an invalid cap", async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "baloot-ffmpeg-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const script = path.join(directory, "ffmpeg-profile.mjs");
  fs.copyFileSync(new URL("../scripts/ffmpeg-profile.mjs", import.meta.url), script);
  const config = path.join(directory, "ffmpeg-runtime.json");
  fs.writeFileSync(config, JSON.stringify({threads: 3}));
  const url = pathToFileURL(script);
  const {bt709FfmpegOverride: override} = await import(url.href);
  const args = override({args: ["-i", "video.mp4", "output.mp4"]});
  assert.deepEqual(args.slice(0, 10), [
    "-filter_threads", "3", "-filter_complex_threads", "3", "-threads", "3", "-i", "video.mp4", "-threads", "3",
  ]);

  fs.writeFileSync(config, JSON.stringify({threads: 0}));
  url.search = "invalid";
  await assert.rejects(import(url.href), /ffmpeg-runtime\.json: threads must be a positive/);
});

test("pinned Remotion derives Chrome video threads from the bounded nproc result", t => {
  const require = createRequire(import.meta.url);
  const renderer = path.dirname(require.resolve("@remotion/renderer"));
  t.mock.method(childProcess, "execSync", command => {
    assert.equal(command, "nproc");
    return Buffer.from("4\n");
  });
  t.mock.method(os, "availableParallelism", () => 32);
  const memory = require(path.join(renderer, "memory/get-available-memory.js"));
  t.mock.method(memory, "getAvailableMemory", () => 16_000_000_000);
  const {getCpuCount} = require(path.join(renderer, "get-cpu-count.js"));
  const {getIdealVideoThreadsFlag} = require(path.join(renderer, "get-video-threads-flag.js"));
  assert.equal(getCpuCount(), 4);
  assert.equal(getIdealVideoThreadsFlag("error"), 3);
  assert.match(fs.readFileSync(path.join(renderer, "open-browser.js"), "utf8"), /--video-threads=\$\{.*getIdealVideoThreadsFlag/);
});
