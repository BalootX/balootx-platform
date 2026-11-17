import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import ts from "typescript";

const source = readFileSync(new URL("../src/playback-state.ts", import.meta.url), "utf8");
const {outputText} = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext}});
const {collectionOpacity, derivePlaybackState, playbackMotionTiming} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
assert.equal(collectionOpacity(0), 1);
assert.equal(collectionOpacity(0.72), 1);
assert.ok(collectionOpacity(0.86) > 0 && collectionOpacity(0.86) < 1);
assert.equal(collectionOpacity(1), 0);
assert.equal(collectionOpacity(1.1), 0);
assert.deepEqual(playbackMotionTiming(600 / 13, 0, 0, 4), {
  flightFrames: 0, collectionFrames: 0, collectionStart: 600 / 13 - 2,
});
for (const steps of [[-1, 22, 4], [21, -1, 4], [21, 22, -1], [NaN, 22, 4]]) {
  assert.throws(() => playbackMotionTiming(600 / 13, ...steps), /nonnegative integer durations/);
}
for (const steps of [[21, 22, 4], [2, 2, 4]]) {
  for (let remaining = 1; remaining < 32; remaining += 1) {
    const interval = 600 / remaining;
    const timing = playbackMotionTiming(interval, ...steps);
    assert.ok(timing.flightFrames <= timing.collectionStart);
    for (let count = 1; count <= remaining; count += 1) {
      const age = Math.ceil(count * interval) - 1 - (count - 1) * interval;
      const progress = Math.min(1, Math.max(0, (age - timing.collectionStart) / timing.collectionFrames));
      assert.equal(collectionOpacity(progress), 0, `${steps}: ${remaining} plays, boundary ${count}`);
    }
  }
}
const plays = ["north", "west", "south", "east"].map((seat, id) => ({seat, card: {id}}));
const paused = derivePlaybackState({
  initialHand: [{id: 2}], allPlays: plays,
  allTricks: [{index: 1, plays, winner: "east"}],
  observer: "south", completedPlayCount: 4,
});
assert.deepEqual(paused.currentTrick, []);
assert.equal(paused.tablePlays.length, 4);
console.log("Playback collection opacity boundaries passed");
