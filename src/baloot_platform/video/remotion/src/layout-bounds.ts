export function visibleRoleBounds(element: Element, root: Element) {
  const isVisible = (element: Element) => {
    // A descendant can override inherited visibility, but not ancestor display/opacity.
    const visibility = getComputedStyle(element).visibility;
    if (visibility === "hidden" || visibility === "collapse") return false;
    for (let node: Element | null = element; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.display === "none" || Number(style.opacity) <= 0) return false;
      if (node === root) break;
    }
    return true;
  };
  const rects = [element, ...Array.from(element.querySelectorAll("*"))]
    .filter(isVisible).map(node => node.getBoundingClientRect())
    .filter(rect => !(rect.width <= 0 || rect.height <= 0));
  let {left, top, right, bottom} = rects[0] ?? element.getBoundingClientRect();
  // shortcut: clipping is conservative; intersect ancestor clips if false failures become a problem.
  for (const rect of rects.slice(1)) {
    left = Math.min(left, rect.left);
    top = Math.min(top, rect.top);
    right = Math.max(right, rect.right);
    bottom = Math.max(bottom, rect.bottom);
  }
  return {visible: rects.length > 0, left, top, right, bottom};
}
