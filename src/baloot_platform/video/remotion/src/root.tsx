import React from "react";
import {Composition} from "remotion";
import {ProfessionalDecisionClip} from "./professional-decision-clip";
import {ProfessionalPrototype} from "./professional-prototype";
import {MultiAngleComposition} from "./multi-angle-composition";
import type {MultiAngleProps, PrototypeProps, RenderBundle} from "./types";

const placeholderBundle: RenderBundle = {
  schemaVersion: 1,
  bundleSha256: "0".repeat(64),
  exportPreset: "decision_clip",
  locale: "ar-SA",
  direction: "rtl",
  observerSeat: "south",
  render: {width: 1080, height: 1920, fps: 60},
  presentation: {},
  localizations: {},
  identity: {
    gameSerial: "BLT-PREVIEW",
    sceneId: "preview",
    enrollmentUri: "balootx://game/BLT-PREVIEW?scene=preview",
    players: {north: "—", west: "—", east: "—", south: "—"},
  },
  audioCues: [],
  scenes: [],
  timeline: {totalFrames: 120, segments: []},
};

const resolveComposition = ({props}: {props: {bundle: RenderBundle}}) => ({
  durationInFrames: props.bundle.timeline.totalFrames,
  fps: props.bundle.render.fps,
  width: props.bundle.render.width,
  height: props.bundle.render.height,
});

const placeholderMultiAngle: MultiAngleProps = {
  plan: {
    sha256: "0".repeat(64),
    fps: 60,
    width: 1080,
    height: 1920,
    totalFrames: 120,
    segments: [
      {
        ordinal: 0,
        sourceName: "placeholder.mp4",
        observerSeat: "south",
        perspectiveName: "—",
        sourceStartFrame: 0,
        sourceEndFrame: 120,
        transition: "cut",
        transitionFrames: 0,
      },
    ],
  },
};

const resolveMultiAngle = ({props}: {props: MultiAngleProps}) => ({
  durationInFrames: props.plan.totalFrames,
  fps: props.plan.fps,
  width: props.plan.width,
  height: props.plan.height,
});

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="ProfessionalPrototype"
      component={ProfessionalPrototype}
      durationInFrames={1}
      fps={60}
      width={1080}
      height={1920}
      defaultProps={{bundle: placeholderBundle, stage: "normal", reducedMotion: false}}
    />
    <Composition
      id="CardMotionSample"
      component={ProfessionalPrototype}
      durationInFrames={120}
      fps={60}
      width={1080}
      height={1920}
      defaultProps={{bundle: placeholderBundle, stage: "normal", reducedMotion: false}}
    />
    <Composition
      id="ProfessionalFlagship"
      component={ProfessionalPrototype}
      durationInFrames={4320}
      fps={60}
      width={1080}
      height={1920}
      defaultProps={{bundle: placeholderBundle, stage: "flagship", reducedMotion: false}}
      calculateMetadata={resolveComposition}
    />
    <Composition
      id="ProfessionalDecisionClip"
      component={ProfessionalDecisionClip}
      durationInFrames={1680}
      fps={60}
      width={1080}
      height={1920}
      defaultProps={{bundle: placeholderBundle, reducedMotion: false}}
      calculateMetadata={resolveComposition}
    />
    <Composition
      id="MultiAngleComposition"
      component={MultiAngleComposition}
      durationInFrames={120}
      fps={60}
      width={1080}
      height={1920}
      defaultProps={placeholderMultiAngle}
      calculateMetadata={resolveMultiAngle}
    />
  </>
);
