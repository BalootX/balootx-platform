import React from "react";
import {
  AbsoluteFill,
  interpolate,
  OffthreadVideo,
  Sequence,
  staticFile,
  useCurrentFrame,
} from "remotion";
import type {MultiAngleProps, MultiAngleSegment} from "./types";

const Segment: React.FC<{
  segment: MultiAngleSegment;
  incomingFrames: number;
  outgoingFrames: number;
}> = ({segment, incomingFrames, outgoingFrames}) => {
  const frame = useCurrentFrame();
  const duration = segment.sourceEndFrame - segment.sourceStartFrame;
  const incomingOpacity = incomingFrames === 0
    ? 1
    : interpolate(frame, [0, incomingFrames], [0, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
  const outgoingOpacity = outgoingFrames === 0
    ? 1
    : interpolate(frame, [duration - outgoingFrames - 1, duration - 1], [1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
  const markerOpacity = interpolate(frame, [0, 8, 72, 88], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{backgroundColor: "#041411", opacity: Math.min(incomingOpacity, outgoingOpacity)}}>
      <OffthreadVideo
        src={staticFile(segment.sourceName)}
        trimBefore={segment.sourceStartFrame}
        trimAfter={segment.sourceEndFrame}
        volume={Math.min(incomingOpacity, outgoingOpacity)}
        style={{height: "100%", width: "100%", objectFit: "cover"}}
      />
      <div
        aria-label={segment.perspectiveName}
        style={{
          position: "absolute",
          top: 190,
          left: 72,
          minWidth: 176,
          height: 56,
          padding: "0 24px",
          border: "1px solid rgba(224,182,83,0.72)",
          borderRadius: 28,
          background: "rgba(4,20,17,0.88)",
          color: "#FFF8E8",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          direction: "rtl",
          fontFamily: "Noto Sans Arabic, sans-serif",
          fontSize: 28,
          fontWeight: 650,
          lineHeight: 1,
          opacity: markerOpacity,
          boxShadow: "0 8px 24px rgba(0,0,0,0.22)",
        }}
      >
        <span style={{color: "#E0B653", fontSize: 22}}>◉</span>
        <span>{segment.perspectiveName}</span>
      </div>
    </AbsoluteFill>
  );
};

export const MultiAngleComposition: React.FC<MultiAngleProps> = ({plan}) => {
  let start = 0;
  return (
    <AbsoluteFill style={{backgroundColor: "#041411"}}>
      {plan.segments.map((segment, index) => {
        const duration = segment.sourceEndFrame - segment.sourceStartFrame;
        const incoming = segment.transition === "fade" ? Math.floor(segment.transitionFrames / 2) : 0;
        const next = plan.segments[index + 1];
        const outgoing = next?.transition === "fade" ? Math.ceil(next.transitionFrames / 2) : 0;
        const from = start;
        start += duration;
        return (
          <Sequence key={`${segment.ordinal}-${segment.sourceName}`} from={from} durationInFrames={duration}>
            <Segment segment={segment} incomingFrames={incoming} outgoingFrames={outgoing} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
