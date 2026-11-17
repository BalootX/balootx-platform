import {createHash} from "node:crypto";
import {readFile, lstat} from "node:fs/promises";
import {readFileSync} from "node:fs";
import {execFileSync} from "node:child_process";
import path from "node:path";
import {fileURLToPath} from "node:url";

export const canonical = value => Array.isArray(value) ? value.map(canonical)
  : value !== null && typeof value === "object"
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
export const sha256 = value => createHash("sha256").update(value).digest("hex");
export const propsHash = bundle => sha256(JSON.stringify(canonical({bundle, reducedMotion: false, capturePlaybackDom: true})));
export const isPlayDecision = bundle => (bundle.scenes.find(scene => scene.stage === "reveal")?.layers.find(layer => layer.id === "analysis")?.content?.decisionKind ?? "play") === "play";
const roles = ["promptPill", "namePills", "buyerBadge", "dealerBadge", "choiceBox"];
const phases = ["hook", "question", "reveal"];
const hex = /^[0-9a-f]{40}(?:[0-9a-f]{24})?$/;
const pairKey = (pair, phase) => `${phase}|${[...pair].sort().join("|")}`;
export const adjacencyTableFile = "src/baloot_platform/video/remotion/scripts/approved-layout-adjacencies.json";

export function validateAdjacencyTable(table) {
  const fail = reason => { throw new Error(`layout adjacency table ${reason}`); };
  if (table?.schemaVersion !== 1 || table.metric !== "intersection_width_height_in_composition_pixels"
      || !Array.isArray(table.rows) || table.rows.length !== 3) fail("missing/invalid");
  const result = new Map();
  const positive = value => Number.isFinite(value) && value > 0;
  const roleId = value => typeof value === "string" && roles.some(role => {
    const index = value.slice(role.length + 1);
    return value.startsWith(`${role}:`) && /^(0|[1-9][0-9]*)$/.test(index) && Number.isSafeInteger(Number(index));
  });
  for (const row of table.rows) {
    if (!Array.isArray(row?.pair) || row.pair.length !== 2 || !row.pair.every(roleId) || row.pair[0] === row.pair[1]
        || !phases.includes(row.phase) || !positive(row.observedMaxWidth) || !positive(row.observedMaxHeight)
        || row.observedMaxWidth !== 167.578125 || row.observedMaxHeight !== 3.8251953125
        || row.marginEachAxis !== 2 || [...row.pair].sort().join() !== "buyerBadge:0,namePills:1") fail("missing/invalid row");
    const key = pairKey(row.pair, row.phase);
    if (result.has(key)) fail("duplicate pair/phase");
    result.set(key, {allowedWidth: row.observedMaxWidth + row.marginEachAxis, allowedHeight: row.observedMaxHeight + row.marginEachAxis});
  }
  if (!phases.every(phase => table.rows.some(row => row.phase === phase))) fail("missing phase");
  return result;
}

const adjacencyBytes = readFileSync(new URL("./approved-layout-adjacencies.json", import.meta.url));
const adjacencies = validateAdjacencyTable(JSON.parse(adjacencyBytes));
const adjacencyTableSha256 = sha256(adjacencyBytes);
export const layoutSourceFiles = ["configs/video/premium.yaml", "release/approved-v4-design-inputs.json",
  ...["professional-prototype.tsx", "professional-decision-clip.tsx", "style.css", "layout-geometry.tsx", "layout-bounds.ts"].map(name => `src/baloot_platform/video/remotion/src/${name}`),
  ...["layout-gate.mjs", "render-decision-clip.mjs", "render-admission.mjs"].map(name => `src/baloot_platform/video/remotion/scripts/${name}`), adjacencyTableFile];

export async function sourceBinding(root, declaration) {
  let manifest;
  let gitHead;
  try { await lstat(path.join(root, ".git")); }
  catch (error) { if (error.code !== "ENOENT") throw error; manifest = JSON.parse(await readFile(path.join(root, "release/source-manifest.json"), "utf8")); }
  if (!manifest) gitHead = execFileSync("git", ["rev-parse", "HEAD"], {cwd: root, encoding: "utf8"}).trim();
  const sourceHead = gitHead ?? manifest.source.revision;
  const sourceTree = gitHead ? execFileSync("git", ["rev-parse", "HEAD^{tree}"], {cwd: root, encoding: "utf8"}).trim() : manifest.source.tree;
  if (!hex.test(sourceHead) || !hex.test(sourceTree) || (declaration &&
      (declaration.sourceHead !== sourceHead || declaration.sourceTree !== sourceTree))) throw new Error("layout source identity mismatch/unavailable");
  const measuredSourceInputs = {};
  for (const name of layoutSourceFiles) {
    const measured = sha256(await readFile(path.join(root, name)));
    const expected = gitHead ? sha256(execFileSync("git", ["show", `${sourceHead}:${name}`], {cwd: root, maxBuffer: 8 * 1024 * 1024})) : manifest.files?.[name]?.sha256;
    if (expected !== measured || (name === adjacencyTableFile && measured !== adjacencyTableSha256)) throw new Error(`layout source input mismatch/unavailable: ${name}`);
    measuredSourceInputs[name] = measured;
  }
  return {sourceHead, sourceTree, measuredSourceInputs,
    collectorSha256: measuredSourceInputs["src/baloot_platform/video/remotion/src/layout-geometry.tsx"]};
}

export function pngHash(bytes, width, height) {
  if (bytes.length < 24 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      || bytes.toString("ascii", 12, 16) !== "IHDR" || bytes.readUInt32BE(16) !== width || bytes.readUInt32BE(20) !== height) {
    throw new Error("layout PNG dimensions/signature mismatch");
  }
  return sha256(bytes);
}

export function validateLayoutGeometry(row, expected) {
  const fail = message => { throw new Error(`layout gate ${expected.phase}: ${message}`); };
  if (!phases.includes(expected.phase)) fail("unsupported measured phase");
  if (!row || row.schemaVersion !== 1 || row.measurementMethod !== "DOM.getBoundingClientRect" || row.coordinateSpace !== "composition_pixels") fail("actual DOM measurement missing");
  for (const key of ["frame", "phase", "width", "height"]) if (row[key] !== expected[key]) fail(`${key} binding mismatch`);
  for (const [key, value] of Object.entries(expected.binding)) {
    if (JSON.stringify(canonical(row.binding?.[key])) !== JSON.stringify(canonical(value))) fail(`${key} binding mismatch`);
  }
  if (!row.rootRect || ![row.rootRect.left, row.rootRect.top, row.rootRect.width, row.rootRect.height].every(Number.isFinite)
      || row.rootRect.width <= 0 || row.rootRect.height <= 0) fail("root is unmeasured");
  if (!row.roles || Object.keys(row.roles).sort().join() !== [...roles].sort().join()) fail("role measurements missing");
  const reveal = expected.phase === "reveal";
  const required = {promptPill: reveal ? 0 : 1, namePills: reveal ? 4 : 3, buyerBadge: 1, dealerBadge: 1, choiceBox: reveal ? 1 : expected.playDecision ? 0 : 1};
  const visible = [];
  for (const role of roles) {
    const group = row.roles[role];
    if (!group || typeof group.selector !== "string" || !group.selector || !Array.isArray(group.items)
        || group.matchedCount !== group.items.length) fail(`${role} is unmeasured`);
    const shown = group.items.filter(item => item.visible === true);
    if (shown.length !== required[role]) fail(`${role} expected ${required[role]} visible, measured ${shown.length}`);
    for (const [index, item] of group.items.entries()) {
      if (item.id !== `${role}:${index}` || typeof item.visible !== "boolean" || ![item.left, item.top, item.right, item.bottom].every(Number.isFinite)) fail(`${role} invalid rectangle`);
      if (!item.visible) continue;
      if (item.right <= item.left || item.bottom <= item.top || item.left < 0 || item.top < 0 || item.right > expected.width || item.bottom > expected.height) fail(`${item.id} invalid/outside frame`);
      visible.push(item);
    }
  }
  const intersections = [];
  for (let a = 0; a < visible.length; a++) for (let b = a + 1; b < visible.length; b++) {
    const first = visible[a], second = visible[b];
    const width = Math.min(first.right, second.right) - Math.max(first.left, second.left);
    const height = Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
    if (width > 0 && height > 0) intersections.push({first: first.id, second: second.id, width, height});
  }
  const designedAdjacencies = [], disallowedIntersections = [];
  for (const intersection of intersections) {
    const allowance = adjacencies.get(pairKey([intersection.first, intersection.second], expected.phase));
    if (allowance && intersection.width <= allowance.allowedWidth && intersection.height <= allowance.allowedHeight) {
      designedAdjacencies.push({...intersection, phase: expected.phase, ...allowance});
    } else disallowedIntersections.push(intersection);
  }
  const validation = {status: disallowedIntersections.length ? "failed" : "passed",
    checks: ["actual_dom_rectangles", "frame_props_source_binding", "required_role_presence", "visible_rectangles_in_frame", "exact_pair_phase_adjacency_within_calibrated_axes"],
    measuredVisibleCount: visible.length, adjacencyTableSha256, intersections, designedAdjacencies, disallowedIntersections};
  if (disallowedIntersections.length) {
    const error = new Error(`layout gate ${expected.phase}: intersections ${JSON.stringify(intersections)}`);
    error.validation = validation;
    throw error;
  }
  return validation;
}

export function layoutFrames(bundle) {
  return ["hook", "question", "reveal"].map(phase => {
    const segment = bundle.timeline.segments.find(row => row.stage === phase);
    if (!segment) throw new Error(`layout gate missing ${phase} segment`);
    return {phase, frame: segment.startFrame + (phase === "hook" ? Math.min(bundle.render.fps, segment.durationFrames - 1) : Math.floor(segment.durationFrames / 2))};
  });
}

export async function verifyPersistedLayout(root, bundleFile, evidenceFile, declaration) {
  const bytes = await readFile(bundleFile);
  const bundle = JSON.parse(bytes);
  const evidence = JSON.parse(await readFile(evidenceFile, "utf8"));
  const binding = {...await sourceBinding(root, declaration), bundleFileSha256: sha256(bytes), bundleSha256: bundle.bundleSha256,
    propsSha256: propsHash(bundle), reducedMotion: false, capturePlaybackDom: true};
  const frames = layoutFrames(bundle);
  if (evidence.schemaVersion !== 1 || !Array.isArray(evidence.items) || evidence.items.length !== 3) throw new Error("layout gate measurements missing");
  if (evidence.status !== "passed") throw new Error("layout gate persisted status is not passed");
  for (const [index, expected] of frames.entries()) {
    const row = evidence.items[index];
    const check = validateLayoutGeometry(row, {...expected, width: bundle.render.width, height: bundle.render.height, binding,
      playDecision: isPlayDecision(bundle)});
    if (typeof row.source !== "string" || path.basename(row.source) !== row.source
        || pngHash(await readFile(path.join(path.dirname(evidenceFile), row.source)), bundle.render.width, bundle.render.height) !== row.pngSha256) throw new Error("layout PNG binding mismatch");
    row.validation = check;
  }
  return {...evidence, status: "passed", binding};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [root, bundleFile, evidenceFile, declaration] = process.argv.slice(2);
  process.stdout.write(`${JSON.stringify(await verifyPersistedLayout(root, bundleFile, evidenceFile, declaration ? JSON.parse(declaration) : undefined))}\n`);
}
