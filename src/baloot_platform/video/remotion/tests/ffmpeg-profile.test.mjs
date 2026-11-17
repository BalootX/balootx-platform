import assert from "node:assert/strict";
import test from "node:test";
import childProcess from "node:child_process";
import fs from "node:fs";
import {createRequire} from "node:module";
import os from "node:os";
import path from "node:path";
import {bt709FfmpegOverride} from "../scripts/ffmpeg-profile.mjs";

test("CI caps every FFmpeg decoder, both filter pools and the encoder without changing non-CI profiles", () => {
  const previous = process.env.CI;
  const args = ["-i", "video.mp4", "-i", "audio.wav", "-c:v", "libx264", "output.mp4"];
  try {
    delete process.env.CI;
    const normal = bt709FfmpegOverride({args});
    assert.deepEqual(normal.slice(0, args.length - 1), args.slice(0, -1));
    assert.equal(normal.includes("-threads"), false);
    process.env.CI = "true";
    const capped = bt709FfmpegOverride({args});
    assert.deepEqual(capped.slice(0, 18), [
      "-filter_threads", "4", "-filter_complex_threads", "4",
      "-threads", "4", "-i", "video.mp4", "-threads", "4", "-i", "audio.wav",
      "-c:v", "libx264", "-threads", "4", "-color_primaries", "bt709",
    ]);
    assert.equal(capped.at(-1), "output.mp4");
    assert.equal(capped.includes("colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"), true);
    assert.deepEqual(args, ["-i", "video.mp4", "-i", "audio.wav", "-c:v", "libx264", "output.mp4"]);
  } finally {
    if (previous === undefined) delete process.env.CI;
    else process.env.CI = previous;
  }
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
