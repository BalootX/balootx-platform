import assert from "node:assert/strict";
import test from "node:test";
import {availableParallelism} from "node:os";
import {RenderInternals} from "@remotion/renderer";
import {cgroupCpuLimit, governedH264Options, hostSafeConcurrency} from "./render-options.mjs";

test("all render entrypoints respect process CPU availability", () => {
  const render = {codec: "libx264", pixel_format: "yuv420p", preset: "fast", concurrency: 32, crf: 18, audio_rate: 48000};
  assert.equal(governedH264Options(render).concurrency, Math.min(32, availableParallelism(), RenderInternals.getMaxConcurrency(), cgroupCpuLimit()));
  assert.equal(render.concurrency, 32);
});

test("renderer admission can be narrower than Node CPU availability", () => {
  const original = RenderInternals.getMaxConcurrency;
  try {
    RenderInternals.getMaxConcurrency = () => 1;
    const render = {codec: "libx264", pixel_format: "yuv420p", preset: "fast", concurrency: 4, crf: 18, audio_rate: 48000};
    assert.equal(governedH264Options(render).concurrency, 1);
    assert.equal(render.concurrency, 4);
  } finally {
    RenderInternals.getMaxConcurrency = original;
  }
});

test("renderer caps concurrency at idle and under sustained pressure", () => {
  assert.equal(hostSafeConcurrency(6, 24, 0.1), 4);
  assert.equal(hostSafeConcurrency(6, 24, 0.3), 2);
  assert.equal(hostSafeConcurrency(6, 24, 0.6), 1);
  assert.equal(hostSafeConcurrency(6, 24, 0.9), 1);
  assert.equal(hostSafeConcurrency(2, 24, 0.1), 2);
  assert.equal(hostSafeConcurrency(6, 9, 0.1), 1);
});

test("small and quota-bound renderers reserve headroom and honor pressure", () => {
  for (const [busy, expected] of [[0.1, 3], [0.5, 2], [0.9, 1]]) {
    assert.equal(hostSafeConcurrency(6, 40, busy, 4), expected);
    assert.equal(hostSafeConcurrency(6, 4, busy), expected);
  }
  assert.equal(hostSafeConcurrency(6, 8, 0.1), 6);
  assert.equal(hostSafeConcurrency(6, 40, 0.1, 8), 6);
  assert.equal(hostSafeConcurrency(6, 40, 0.9, 8), 1);
  assert.equal(hostSafeConcurrency(6, 1, 0.1), 1);
  assert.equal(hostSafeConcurrency(2, 40, 0.1, 4), 2);
});
