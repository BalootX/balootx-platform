// @ts-check

/**
 * @param {number} interval
 * @param {Record<string, unknown>} [motion]
 */
export const playbackMotionTiming = (interval, motion = {}) => {
  const flightSteps = Number(motion.flight_steps ?? 21);
  const collectionSteps = Number(motion.collection_steps ?? 22);
  const transitionSteps = Number(motion.transition_steps ?? 4);
  if (!Number.isFinite(interval) || interval < 0 ||
    [flightSteps, collectionSteps, transitionSteps].some(steps => !Number.isInteger(steps) || steps < 0)) {
    throw new Error("Playback motion requires a finite nonnegative interval and nonnegative integer durations");
  }
  // Leave a cleared frame plus fractional-frame rounding room before the next play.
  const collectionFrames = Math.min(collectionSteps, Math.max(1, interval - flightSteps - transitionSteps - 2));
  const flightFrames = Math.min(flightSteps, Math.max(1, interval - collectionFrames - transitionSteps - 2));
  return {flightFrames, collectionFrames, collectionStart: Math.max(0, interval - collectionFrames - 2)};
};

/**
 * @param {number} playAge
 * @param {ReturnType<typeof playbackMotionTiming>} timing
 */
export const trickCollectionProgress = (playAge, {collectionFrames, collectionStart}) =>
  collectionFrames === 0
    ? Number(playAge >= collectionStart)
    : Math.min(1, Math.max(0, (playAge - collectionStart) / collectionFrames));

/**
 * @param {import("./types").RenderBundle} bundle
 * @returns {[number, number]}
 */
export const trickCollectionSampleFrameRange = ({timeline, scenes, presentation}) => {
  const replay = timeline.segments.find(segment => segment.stage === "replay");
  if (!replay) throw new Error("Trick collection sample requires a play-decision replay segment");
  const replayScene = scenes.find(scene => scene.sceneId === replay.sceneId);
  const pausePlayIndex = replayScene?.layers.find(layer => layer.id === "public-lifecycle")?.content.pauseBeforePlayIndex;
  if (typeof pausePlayIndex !== "number" || !Number.isInteger(pausePlayIndex)) {
    throw new Error("Trick collection sample requires a replay play count");
  }
  const hasContextPlayback = timeline.segments.some(segment => segment.stage === "context" && segment.durationFrames > 0);
  const replayPlayCount = pausePlayIndex - Number(hasContextPlayback);
  if (replayPlayCount < 5) {
    throw new Error("Trick collection sample requires at least five plays in replay");
  }
  const interval = replay.durationFrames / replayPlayCount;
  const timing = playbackMotionTiming(interval, presentation.motion);
  const fourthPlayStart = replay.startFrame + 3 * interval;
  const collectionBoundary = Math.ceil(fourthPlayStart + timing.collectionStart);
  // An instantaneous collection can advance on the boundary where a fade is still at zero.
  const beforeCollection = trickCollectionProgress(collectionBoundary - fourthPlayStart, timing) > 0
    ? collectionBoundary - 1
    : collectionBoundary;
  const nextPlay = replay.startFrame + Math.ceil(4 * replay.durationFrames / replayPlayCount);
  if (beforeCollection < replay.startFrame || beforeCollection + 1 >= nextPlay ||
    nextPlay >= replay.startFrame + replay.durationFrames) {
    throw new Error("Replay has too few frames to show the first trick collection and the next play");
  }
  return [beforeCollection, nextPlay];
};
