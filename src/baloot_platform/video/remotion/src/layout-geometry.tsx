import React from "react";
import {Artifact, cancelRender, continueRender, delayRender, useCurrentFrame} from "remotion";
import type {RenderBundle} from "./types";

const selectors = {
  promptPill: ".decision-prompt .ask",
  namePills: ".seat-dock .seat, .viewer-identity .seat",
  buyerBadge: ".tag.buyer",
  dealerBadge: ".tag.dealer",
  choiceBox: ".reveal-reason .pick, .decision-choices",
  researchNotice: ".rhead.research-notice",
  broadcastHud: ".broadcast-hud",
};

function visibleBounds(element: Element, root: Element) {
  const rects = [element, ...Array.from(element.querySelectorAll("*"))].filter(node => {
    const visibility = getComputedStyle(node).visibility;
    if (visibility === "hidden" || visibility === "collapse") return false;
    for (let ancestor: Element | null = node; ancestor; ancestor = ancestor.parentElement) {
      const style = getComputedStyle(ancestor);
      if (style.display === "none" || Number(style.opacity) <= 0) return false;
      if (ancestor === root) break;
    }
    return true;
  }).map(node => node.getBoundingClientRect()).filter(rect => rect.width > 0 && rect.height > 0);
  let {left, top, right, bottom} = rects[0] ?? element.getBoundingClientRect();
  // shortcut: include descendant overflow conservatively, account for clipping if a valid layout is rejected.
  for (const rect of rects.slice(1)) {
    left = Math.min(left, rect.left);
    top = Math.min(top, rect.top);
    right = Math.max(right, rect.right);
    bottom = Math.max(bottom, rect.bottom);
  }
  return {visible: rects.length > 0, left, top, right, bottom};
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value === null || typeof value !== "object") return value;
  const record = value as Record<string, unknown>;
  return Object.fromEntries(Object.keys(record).sort().map(key => [key, canonical(record[key])]));
}

export const LayoutGeometryArtifact: React.FC<{
  bundle: RenderBundle;
  binding: Record<string, string>;
  reducedMotion: boolean;
  capturePlaybackDom: boolean;
}> = ({bundle, binding, reducedMotion, capturePlaybackDom}) => {
  const frame = useCurrentFrame();
  const [snapshot, setSnapshot] = React.useState<string | null>(null);
  const [handle] = React.useState(() => delayRender("measure decision clip layout"));
  React.useLayoutEffect(() => {
    const capture = async () => {
      await document.fonts.ready;
      await Promise.all(Array.from(document.images, image => image.decode()));
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      const root = document.querySelector(".canvas.majlis");
      if (!root) throw new Error("layout measurement root is missing");
      const {left, top, width, height} = root.getBoundingClientRect();
      if (width <= 0 || height <= 0) throw new Error("layout measurement root is unmeasured");
      const scaleX = bundle.render.width / width;
      const scaleY = bundle.render.height / height;
      const bounds = (element: Element) => {
        const rect = visibleBounds(element, root);
        return {visible: rect.visible, left: (rect.left - left) * scaleX, top: (rect.top - top) * scaleY,
          right: (rect.right - left) * scaleX, bottom: (rect.bottom - top) * scaleY};
      };
      const roles = Object.fromEntries(Object.entries(selectors).map(([role, selector]) => {
        const items = Array.from(root.querySelectorAll(selector), (element, index) => ({id: `${role}:${index}`, ...bounds(element)}));
        return [role, {matchedCount: items.length, items}];
      }));
      const buyerCard = root.querySelector(".tag.buyer > .pcard");
      const card = buyerCard?.getBoundingClientRect();
      const badge = buyerCard?.parentElement?.getBoundingClientRect();
      const buyerCardContained = card && badge && card.width > 0 && card.height > 0
        && card.left >= badge.left && card.top >= badge.top && card.right <= badge.right && card.bottom <= badge.bottom;
      const observerCards = Array.from(root.querySelectorAll(".observer-hand .card"), bounds);
      const phase = bundle.timeline.segments.find(segment =>
        frame >= segment.startFrame && frame < segment.startFrame + segment.durationFrames)?.stage;
      const props = canonical({bundle, reducedMotion, capturePlaybackDom});
      const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(props)));
      const propsSha256 = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
      setSnapshot(JSON.stringify({schemaVersion: 1, measurementMethod: "DOM.getBoundingClientRect", coordinateSpace: "composition_pixels",
        phase, frame, width: bundle.render.width, height: bundle.render.height,
        rootRect: {left, top, width, height}, binding: {...binding, propsSha256}, roles, buyerCardContained, observerCards}));
    };
    void capture().catch(cancelRender);
  }, [bundle, binding, frame, reducedMotion, capturePlaybackDom]);
  React.useLayoutEffect(() => {
    if (snapshot !== null) continueRender(handle);
  }, [snapshot, handle]);
  return snapshot === null ? null : <Artifact filename="layout-geometry.json" content={snapshot} />;
};
