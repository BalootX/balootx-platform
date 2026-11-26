import {createHash} from "node:crypto";
import {createReadStream} from "node:fs";
import {readFile} from "node:fs/promises";

const phases = ["hook", "question", "reveal"];
const adjacencyRoles = ["promptPill", "namePills", "buyerBadge", "dealerBadge", "choiceBox"];
const roles = [...adjacencyRoles, "researchNotice", "broadcastHud"];
const pairKey = (phase, pair) => [phase, ...[...pair].sort()].join("|");
const positive = value => Number.isFinite(value) && value > 0;
const roleId = /^(promptPill|namePills|buyerBadge|dealerBadge|choiceBox):(0|[1-9]\d*)$/;
const hash = /^[a-f0-9]{64}$/;
const approvedLayoutTableSha256 = "9f5f305b43c0d4f15cf4c15479807ec4cd4e8840150c35afae742ffa85687d28";

export const layoutMeasurementMarker = "balootx-layout-measurement-never-post";
export const sha256 = value => createHash("sha256").update(value).digest("hex");
export const canonical = value => Array.isArray(value) ? value.map(canonical)
  : value !== null && typeof value === "object"
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
export const propsSha256 = bundle => sha256(JSON.stringify(canonical({bundle, reducedMotion: false, capturePlaybackDom: true})));

export async function fileSha256(file) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(file)) digest.update(chunk);
  return digest.digest("hex");
}

export async function loadLayoutTable({measure = false, file, expectedSha256} = {}) {
  if (!measure && (file !== undefined || expectedSha256 !== undefined)) {
    throw new Error("layout table overrides require NEVER_POST measurement mode");
  }
  if (file !== undefined && !expectedSha256) {
    throw new Error("an external layout table requires its SHA-256 pin");
  }
  const pin = expectedSha256 ?? approvedLayoutTableSha256;
  const bytes = await readFile(file ?? new URL("./approved-layout-adjacencies.json", import.meta.url));
  const tableSha256 = sha256(bytes);
  if (pin !== tableSha256) {
    throw new Error("layout table SHA-256 mismatch");
  }
  const table = JSON.parse(bytes);
  const forbidden = table?.forbiddenMeasurementSha256 ?? [];
  if (table?.schemaVersion !== 1 || table.metric !== "intersection_width_height_in_composition_pixels"
      || !positive(table.noticeHeightPx) || !Array.isArray(table.rows)
      || !Array.isArray(forbidden) || !forbidden.every(value => typeof value === "string" && hash.test(value))) {
    throw new Error("invalid layout calibration table");
  }
  const allowances = new Map();
  for (const row of table.rows) {
    if (!phases.includes(row?.phase) || !Array.isArray(row.pair) || row.pair.length !== 2
        || !row.pair.every(id => typeof id === "string" && roleId.test(id)) || row.pair[0] === row.pair[1]
        || !positive(row.observedMaxWidth) || !positive(row.observedMaxHeight)
        || ![0, 1, 2].includes(row.marginEachAxis)) throw new Error("invalid layout calibration row");
    const key = pairKey(row.phase, row.pair);
    if (allowances.has(key)) throw new Error("duplicate layout calibration row");
    allowances.set(key, {
      allowedWidth: row.observedMaxWidth + row.marginEachAxis,
      allowedHeight: row.observedMaxHeight + row.marginEachAxis,
    });
  }
  return {bytes, tableSha256, noticeHeightPx: table.noticeHeightPx, allowances, forbidden};
}

export function layoutFrames(bundle) {
  return phases.map(phase => {
    const segment = bundle.timeline.segments.find(row => row.stage === phase);
    if (!segment || !Number.isSafeInteger(segment.startFrame) || segment.startFrame < 0
        || !Number.isSafeInteger(segment.durationFrames) || segment.durationFrames < 1) {
      throw new Error(`layout gate missing or invalid ${phase} segment`);
    }
    return {phase, frame: segment.startFrame + (phase === "hook"
      ? Math.min(bundle.render.fps, segment.durationFrames - 1)
      : Math.floor(segment.durationFrames / 2))};
  });
}

export function pngHash(bytes, width, height) {
  if (bytes.length < 24 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      || bytes.toString("ascii", 12, 16) !== "IHDR" || bytes.readUInt32BE(16) !== width || bytes.readUInt32BE(20) !== height) {
    throw new Error("layout PNG signature or dimensions mismatch");
  }
  return sha256(bytes);
}

export function checkLayoutGeometry(row, expected, table) {
  const fail = reason => { throw new Error(`layout gate ${expected.phase}: ${reason}`); };
  if (!phases.includes(expected.phase) || row?.schemaVersion !== 1
      || row.measurementMethod !== "DOM.getBoundingClientRect" || row.coordinateSpace !== "composition_pixels") {
    fail("actual DOM measurement missing");
  }
  for (const key of ["frame", "phase", "width", "height"]) {
    if (row[key] !== expected[key]) fail(`${key} binding mismatch`);
  }
  if (JSON.stringify(canonical(row.binding)) !== JSON.stringify(canonical(expected.binding))
      || row.binding?.tableSha256 !== table.tableSha256) fail("render or table binding mismatch");
  const measure = expected.binding.mode === "measurement";
  if (!measure && expected.binding.mode !== "qualification") fail("invalid layout mode");
  if (!measure && table.tableSha256 !== approvedLayoutTableSha256) fail("approved layout table SHA-256 mismatch");
  if (!Number.isFinite(row.rootRect?.left) || !Number.isFinite(row.rootRect?.top)
      || !positive(row.rootRect?.width) || !positive(row.rootRect?.height)) fail("root is unmeasured");
  if (!row.roles || Object.keys(row.roles).length !== roles.length) fail("role measurements missing");
  const reveal = expected.phase === "reveal";
  const required = {promptPill: reveal ? 0 : 1, namePills: reveal ? 4 : 3, buyerBadge: 1, dealerBadge: 1,
    choiceBox: reveal || !expected.playDecision ? 1 : 0, researchNotice: 1, broadcastHud: 1};
  const visible = [];
  for (const role of roles) {
    const group = row.roles[role];
    if (!Array.isArray(group?.items) || group.matchedCount !== group.items.length) fail(`${role} is unmeasured`);
    let count = 0;
    for (const [index, item] of group.items.entries()) {
      if (item?.id !== `${role}:${index}` || typeof item.visible !== "boolean"
          || ![item.left, item.top, item.right, item.bottom].every(Number.isFinite)) fail(`${role} invalid rectangle`);
      if (!item.visible) continue;
      if (item.right <= item.left || item.bottom <= item.top || item.left < 0 || item.top < 0
          || item.right > expected.width || item.bottom > expected.height) fail(`${item.id} invalid or outside frame`);
      if (!measure && role === "researchNotice" && Math.abs(item.bottom - item.top - table.noticeHeightPx) > 0.01) {
        fail("notice band height differs from calibration");
      }
      visible.push(item);
      count++;
    }
    if (count !== required[role]) fail(`${role} expected ${required[role]} visible, measured ${count}`);
  }
  const intersections = [], designedAdjacencies = [], disallowedIntersections = [];
  for (const [index, first] of visible.entries()) {
    for (const second of visible.slice(index + 1)) {
      const width = Math.min(first.right, second.right) - Math.max(first.left, second.left);
      const height = Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
      if (width <= 0 || height <= 0) continue;
      const contact = {first: first.id, second: second.id, width, height};
      intersections.push(contact);
      const allowance = table.allowances.get(pairKey(expected.phase, [first.id, second.id]));
      if (allowance && width <= allowance.allowedWidth && height <= allowance.allowedHeight) {
        designedAdjacencies.push({...contact, ...allowance});
      } else {
        disallowedIntersections.push(contact);
      }
    }
  }
  const status = measure ? "measured" : disallowedIntersections.length ? "failed" : "passed";
  const validation = {status, tableSha256: table.tableSha256, measuredVisibleCount: visible.length,
    intersections, designedAdjacencies, disallowedIntersections};
  if (status === "failed") {
    const error = new Error(`layout gate ${expected.phase}: uncalibrated intersections ${JSON.stringify(disallowedIntersections)}`);
    error.validation = validation;
    throw error;
  }
  return validation;
}
