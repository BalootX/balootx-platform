import React from "react";
import {ProfessionalPrototype} from "./professional-prototype";
import type {RenderBundle} from "./types";

type DecisionClipProps = {
  bundle: RenderBundle;
  reducedMotion?: boolean;
  capturePlaybackDom?: boolean;
};

/** A governed short export driven entirely by its bundled SceneTimelineV1. */
export const ProfessionalDecisionClip: React.FC<DecisionClipProps> = ({bundle, reducedMotion = false, capturePlaybackDom = false}) => {
  return <ProfessionalPrototype bundle={bundle} stage="flagship" reducedMotion={reducedMotion} capturePlaybackDom={capturePlaybackDom} />;
};
