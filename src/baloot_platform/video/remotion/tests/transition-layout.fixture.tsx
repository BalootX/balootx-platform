import React from "react";
import {Composition, getInputProps, registerRoot, useCurrentFrame} from "remotion";
import {ProfessionalDecisionClip} from "../src/professional-decision-clip";

type FixtureProps = React.ComponentProps<typeof ProfessionalDecisionClip> & {
  overlapFrame?: number;
};

const SyntheticDecisionClip = ({overlapFrame, ...props}: FixtureProps) => {
  const frame = useCurrentFrame();
  const {height, fps} = props.bundle.render;
  const pulse = overlapFrame === undefined ? 0 : Math.max(0, 1 - Math.abs(frame - overlapFrame) / (fps / 2));
  return <>
    <ProfessionalDecisionClip {...props} />
    {pulse > 0 && <style>{`.decision-prompt { translate: 0 ${-0.6 * height * pulse}px; }`}</style>}
  </>;
};

function FixtureRoot() {
  const props = getInputProps<FixtureProps>();
  const {width, height, fps} = props.bundle.render;
  return <Composition id="SyntheticDecisionClip" component={SyntheticDecisionClip} defaultProps={props}
    width={width} height={height} fps={fps} durationInFrames={props.bundle.timeline.totalFrames} />;
}

registerRoot(FixtureRoot);
