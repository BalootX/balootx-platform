import React from "react";
import {ProfessionalPrototype} from "./professional-prototype";
import type {RenderBundle} from "./types";
import {LayoutGeometryArtifact} from "./layout-geometry";

type DecisionClipProps = {
  bundle: RenderBundle;
  reducedMotion?: boolean;
  capturePlaybackDom?: boolean;
  layoutBinding?: Record<string, unknown>;
};

/** A governed short export driven entirely by its bundled SceneTimelineV1. */
export const ProfessionalDecisionClip: React.FC<DecisionClipProps> = ({bundle, reducedMotion = false, capturePlaybackDom = false, layoutBinding}) => {
  return <><ProfessionalPrototype bundle={bundle} stage="flagship" reducedMotion={reducedMotion} capturePlaybackDom={capturePlaybackDom} />
    {layoutBinding ? <LayoutGeometryArtifact bundle={bundle} binding={layoutBinding} reducedMotion={reducedMotion} /> : null}</>;
};
