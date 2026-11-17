import React from "react";
import {Artifact, cancelRender, continueRender, delayRender, useCurrentFrame} from "remotion";
import {visibleRoleBounds} from "./layout-bounds";

export const layoutSelectors = {
  promptPill: ".decision-prompt .ask",
  namePills: ".seat-dock .seat, .viewer-identity .seat",
  buyerBadge: ".tag.buyer",
  dealerBadge: ".tag.dealer",
  choiceBox: ".reveal-reason .pick, .decision-choices",
};

const canonical = (value: any): any => Array.isArray(value) ? value.map(canonical)
  : value !== null && typeof value === "object"
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;

export const LayoutGeometryArtifact: React.FC<{bundle: any; binding: any; reducedMotion?: boolean}> = ({bundle, binding, reducedMotion = false}) => {
  const frame = useCurrentFrame();
  const [snapshot, setSnapshot] = React.useState<string | null>(null);
  const [handle] = React.useState(() => delayRender("measure actual frame pill/badge geometry"));
  React.useLayoutEffect(() => {
    const capture = async () => {
      await document.fonts.ready;
      await Promise.all(Array.from(document.images).map(image => image.decode()));
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      const root = document.querySelector<HTMLElement>(".canvas.majlis");
      if (!root) throw new Error("layout measurement root is missing");
      const bounds = root.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) throw new Error("layout measurement root is unmeasured");
      const scaleX = bundle.render.width / bounds.width;
      const scaleY = bundle.render.height / bounds.height;
      const roles = Object.fromEntries(Object.entries(layoutSelectors).map(([role, selector]) => {
        const elements = Array.from(root.querySelectorAll<HTMLElement>(selector));
        const items = elements.map((element, index) => {
          const rect = visibleRoleBounds(element, root);
          return {id: `${role}:${index}`, visible: rect.visible,
            left: (rect.left - bounds.left) * scaleX, top: (rect.top - bounds.top) * scaleY,
            right: (rect.right - bounds.left) * scaleX, bottom: (rect.bottom - bounds.top) * scaleY};
        });
        return [role, {selector, matchedCount: elements.length, items}];
      }));
      const phase = bundle.timeline.segments.find((segment: any) => frame >= segment.startFrame && frame < segment.startFrame + segment.durationFrames)?.stage;
      const props = {bundle, reducedMotion, capturePlaybackDom: true};
      const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(canonical(props))));
      const propsSha256 = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
      setSnapshot(JSON.stringify({schemaVersion: 1, measurementMethod: "DOM.getBoundingClientRect", coordinateSpace: "composition_pixels",
        frame, phase, width: bundle.render.width, height: bundle.render.height,
        rootRect: {left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height},
        binding: {...binding, bundleSha256: bundle.bundleSha256, propsSha256, reducedMotion, capturePlaybackDom: true}, roles}));
      continueRender(handle);
    };
    void capture().catch(cancelRender);
  }, [bundle, binding, frame, handle, reducedMotion]);
  return snapshot === null ? null : <Artifact filename="layout-geometry.json" content={snapshot} />;
};
