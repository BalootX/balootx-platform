import assert from "node:assert/strict";
import fs from "node:fs";
import {syncBuiltinESMExports} from "node:module";
import os from "node:os";
import test from "node:test";
import {RenderInternals} from "@remotion/renderer";
import {governedH264Options} from "../scripts/render-options.mjs";
import "../scripts/render-options.test.mjs";

test("all render entrypoints fit the cgroup CPU quota as well as narrower existing caps", t => {
  let files = {};
  t.mock.method(fs, "readFileSync", filename => {
    if (Object.hasOwn(files, filename)) return files[filename];
    throw Object.assign(new Error("missing synthetic cgroup file"), {code: "ENOENT"});
  });
  t.mock.method(os, "availableParallelism", () => 32);
  t.mock.method(RenderInternals, "getMaxConcurrency", () => 32);
  syncBuiltinESMExports();
  const render = {codec: "libx264", pixel_format: "yuv420p", preset: "fast", concurrency: 32, crf: 18, audio_rate: 48000};
  try {
    for (const [scenario, expected] of [
      [{"/sys/fs/cgroup/cpu.max": "400000 100000"}, 4],
      [{"/proc/self/cgroup": "0::/pod/renderer", "/sys/fs/cgroup/pod/renderer/cpu.max": "max 100000", "/sys/fs/cgroup/pod/cpu.max": "400000 100000"}, 4],
      [{"/proc/self/cgroup": "0::/pod/renderer", "/sys/fs/cgroup/pod/renderer/cpu.max": "800000 100000", "/sys/fs/cgroup/pod/cpu.max": "250000 100000"}, 2],
      [{"/sys/fs/cgroup/cpu.max": "50000 100000"}, 1],
      [{"/proc/self/cgroup": "3:cpu,cpuacct:/pod", "/sys/fs/cgroup/cpu,cpuacct/pod/cpu.cfs_quota_us": "400000", "/sys/fs/cgroup/cpu,cpuacct/pod/cpu.cfs_period_us": "100000"}, 4],
      [{"/sys/fs/cgroup/cpu.max": "max 100000"}, 32],
      [{"/sys/fs/cgroup/cpu.max": "400000 0"}, 32],
      [{"/sys/fs/cgroup/cpu.max": "invalid"}, 32],
      [{}, 32],
    ]) {
      files = scenario;
      assert.equal(governedH264Options(render).concurrency, expected, JSON.stringify(files));
      assert.equal(render.concurrency, 32);
    }
    files = {"/sys/fs/cgroup/cpu.max": "400000 100000"};
    assert.equal(governedH264Options({...render, concurrency: 2}).concurrency, 2);
    t.mock.method(os, "availableParallelism", () => 3);
    syncBuiltinESMExports();
    assert.equal(governedH264Options(render).concurrency, 3);
    t.mock.method(RenderInternals, "getMaxConcurrency", () => 1);
    assert.equal(governedH264Options(render).concurrency, 1);
  } finally {
    t.mock.restoreAll();
    syncBuiltinESMExports();
  }
});
