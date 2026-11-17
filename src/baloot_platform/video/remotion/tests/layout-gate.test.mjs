import assert from "node:assert/strict";
import test from "node:test";
import {mkdtemp, mkdir, readFile, writeFile} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {adjacencyTableFile, layoutSourceFiles, pngHash, propsHash, sha256, sourceBinding, validateAdjacencyTable, validateLayoutGeometry, verifyPersistedLayout} from "../scripts/layout-gate.mjs";
import {visibleRoleBounds} from "../src/layout-bounds.ts";

// Synthetic algorithm cases only; these rectangles are never design evidence.
const example = phase => {
  let ordinal = 0;
  const required = {promptPill: phase === "reveal" ? 0 : 1, namePills: phase === "reveal" ? 4 : 3, buyerBadge: 1, dealerBadge: 1, choiceBox: phase === "reveal" ? 1 : 0};
  return {schemaVersion: 1, measurementMethod: "DOM.getBoundingClientRect", coordinateSpace: "composition_pixels",
    phase, frame: 30, width: 1080, height: 1920, rootRect: {left: 0, top: 0, width: 540, height: 960},
    binding: {sourceHead: "a".repeat(40), propsSha256: "b".repeat(64)},
    roles: Object.fromEntries(Object.entries(required).map(([role, count]) => [role, {selector: `.${role}`, matchedCount: count,
      items: Array.from({length: count}, (_, index) => { const left = ordinal++ * 100; return {id: `${role}:${index}`, visible: true, left, top: 10, right: left + 100, bottom: 80}; })}]))};
};
const expected = row => ({phase: row.phase, frame: row.frame, width: row.width, height: row.height, binding: row.binding, playDecision: true});
const archive = async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "baloot-layout-source-"));
  const manifest = {source: {revision: "a".repeat(40), tree: "b".repeat(40)}, files: {}};
  for (const name of layoutSourceFiles) {
    const data = name === adjacencyTableFile ? await readFile(new URL("../scripts/approved-layout-adjacencies.json", import.meta.url)) : `synthetic trust-boundary input: ${name}\n`;
    await mkdir(path.dirname(path.join(root, name)), {recursive: true});
    await writeFile(path.join(root, name), data);
    manifest.files[name] = {sha256: sha256(data)};
  }
  const saveManifest = () => writeFile(path.join(root, "release/source-manifest.json"), JSON.stringify(manifest));
  await saveManifest();
  return {root, manifest, saveManifest};
};

test("all visible roles measured at hook/question/reveal; touching edges allowed", () => {
  for (const phase of ["hook", "question", "reveal"]) {
    const row = example(phase);
    assert.equal(validateLayoutGeometry(row, expected(row)).status, "passed");
    assert.equal(validateLayoutGeometry(row, expected(row)).checks.at(-1), "exact_pair_phase_adjacency_within_calibrated_axes");
  }
});

test("each uncalibrated role-instance contact is rejected", () => {
  for (const role of ["promptPill", "namePills", "buyerBadge", "dealerBadge", "choiceBox"]) {
    const row = example(role === "choiceBox" ? "reveal" : "question");
    const reference = row.roles.namePills.items[role === "namePills" ? 1 : 0];
    Object.assign(row.roles[role].items[0], {left: reference.left, right: reference.right});
    assert.throws(() => validateLayoutGeometry(row, expected(row)), /intersections/);
  }
});

test("visible descendant overflow is included before checking role intersections", () => {
  const previous = globalThis.getComputedStyle;
  globalThis.getComputedStyle = element => ({display: "block", opacity: "1",
    visibility: element.style.visibility ?? (element.parentElement ? getComputedStyle(element.parentElement).visibility : "visible"), ...element.style});
  const element = (left, top, right, bottom, parentElement = null, style = {}) => ({
    parentElement, style, children: [],
    getBoundingClientRect: () => ({left, top, right, bottom, width: right - left, height: bottom - top}),
    querySelectorAll() { return this.children.flatMap(child => [child, ...child.querySelectorAll("*")]); },
  });
  try {
    const root = element(0, 0, 1080, 1920);
    const badge = element(400, 100, 500, 140, root);
    const card = element(410, 87, 477, 154, badge);
    badge.children.push(card);
    const row = example("hook");
    Object.assign(row.roles.promptPill.items[0], {left: 400, top: 150, right: 500, bottom: 242});
    Object.assign(row.roles.buyerBadge.items[0], visibleRoleBounds(badge, root));
    assert.deepEqual(visibleRoleBounds(badge, root), {visible: true, left: 400, top: 87, right: 500, bottom: 154});
    assert.throws(() => validateLayoutGeometry(row, expected(row)), /intersections.*"width":100,"height":4/);
    for (const style of [{display: "none"}, {visibility: "hidden"}, {opacity: "0"}]) {
      card.style = style;
      Object.assign(row.roles.buyerBadge.items[0], visibleRoleBounds(badge, root));
      assert.equal(validateLayoutGeometry(row, expected(row)).status, "passed");
    }
    card.style = {};
    badge.style = {opacity: "0"};
    assert.equal(visibleRoleBounds(badge, root).visible, false);
    badge.style = {visibility: "hidden"};
    card.style = {visibility: "visible"};
    assert.deepEqual(visibleRoleBounds(badge, root), {visible: true, left: 410, top: 87, right: 477, bottom: 154});
    const reveal = example("reveal");
    reveal.roles.promptPill = {selector: ".promptPill", matchedCount: 1,
      items: [{id: "promptPill:0", ...visibleRoleBounds(badge, root)}]};
    assert.throws(() => validateLayoutGeometry(reveal, expected(reveal)), /promptPill expected 0 visible, measured 1/);
    card.style = {};
    assert.equal(visibleRoleBounds(badge, root).visible, false);
    badge.style = {display: "none"};
    card.style = {visibility: "visible"};
    assert.equal(visibleRoleBounds(badge, root).visible, false);
  } finally {
    if (previous === undefined) delete globalThis.getComputedStyle;
    else globalThis.getComputedStyle = previous;
  }
});

test("missing, invalid and stale measurements fail closed", () => {
  for (const change of [
    row => { delete row.roles.buyerBadge; },
    row => { row.roles.buyerBadge.items = []; row.roles.buyerBadge.matchedCount = 0; },
    row => { row.roles.namePills.items[0].right = Number.NaN; },
    row => { row.roles.namePills.items[0].visible = false; },
    row => { row.rootRect.width = 0; },
    row => { row.frame++; },
    row => { row.phase = "reveal"; },
    row => { row.binding.propsSha256 = "c".repeat(64); },
    row => { row.binding.sourceHead = "c".repeat(40); },
  ]) {
    const row = example("question"), binding = structuredClone(expected(row));
    change(row);
    assert.throws(() => validateLayoutGeometry(row, binding), /layout gate/);
  }
  assert.throws(() => validateLayoutGeometry(undefined, expected(example("hook"))), /actual DOM measurement missing/);
});

test("props identity hashes actual inputs and is independent of key order", () => {
  assert.equal(propsHash({a: 1, b: 2}), propsHash({b: 2, a: 1}));
  assert.notEqual(propsHash({a: 1}), propsHash({a: 2}));
});

test("source admission measures bytes and rejects unavailable or stale archive bindings", async () => {
  const {root, manifest, saveManifest} = await archive();
  const actual = await sourceBinding(root);
  for (const name of layoutSourceFiles) assert.equal(actual.measuredSourceInputs[name], manifest.files[name].sha256);
  await assert.rejects(sourceBinding(root, {sourceHead: "c".repeat(40), sourceTree: manifest.source.tree}), /source identity/);
  const baseline = "release/approved-v4-design-inputs.json";
  const baselineEntry = manifest.files[baseline];
  delete manifest.files[baseline]; await saveManifest();
  await assert.rejects(sourceBinding(root), /input mismatch\/unavailable/);
  manifest.files[baseline] = baselineEntry; await saveManifest();
  await writeFile(path.join(root, baseline), "changed bytes");
  await assert.rejects(sourceBinding(root), /input mismatch\/unavailable/);
});

test("PNG identity checks signature, dimensions and actual byte hash", () => {
  const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN1cAAAAASUVORK5CYII=", "base64");
  assert.equal(pngHash(png, 1, 1), sha256(png));
  assert.throws(() => pngHash(png, 1080, 1920), /dimensions\/signature/);
  assert.throws(() => pngHash(Buffer.from("missing"), 1, 1), /dimensions\/signature/);
});

test("missing persisted three-phase measurements never pass", async () => {
  const {root} = await archive();
  const bundleFile = path.join(root, "bundle.json"), evidenceFile = path.join(root, "layout-evidence.json");
  await writeFile(bundleFile, JSON.stringify({render: {width: 1080, height: 1920, fps: 30},
    timeline: {segments: ["hook", "question", "reveal"].map((stage, index) => ({stage, startFrame: index * 60, durationFrames: 60}))}}));
  await writeFile(evidenceFile, JSON.stringify({schemaVersion: 1, items: []}));
  await assert.rejects(verifyPersistedLayout(root, bundleFile, evidenceFile), /measurements missing/);
  for (const status of [undefined, "failed", "unknown"]) {
    await writeFile(evidenceFile, JSON.stringify({schemaVersion: 1, status, items: [{}, {}, {}]}));
    await assert.rejects(verifyPersistedLayout(root, bundleFile, evidenceFile), /persisted status is not passed/);
  }
});

const tableBytes = await readFile(new URL("../scripts/approved-layout-adjacencies.json", import.meta.url));
const table = JSON.parse(tableBytes);

// Synthetic boundary geometry only; never a substitute for captured approval evidence.
const contactExample = (phase, index = 1, role = "buyerBadge", width = 169.578125, height = 5.8251953125) => {
  const row = example(phase);
  Object.assign(row.roles.namePills.items[index], {left: 100, top: 200, right: 500, bottom: 300});
  Object.assign(row.roles[role].items[0], {left: 500 - width, top: 300 - height, right: 600, bottom: 400});
  return row;
};

for (const phase of ["hook", "question", "reveal"]) test(`${phase} exact per-axis boundary passes; either axis above it fails`, () => {
  const boundary = contactExample(phase);
  assert.equal(validateLayoutGeometry(boundary, expected(boundary)).status, "passed");
  for (const [width, height] of [[169.578125 + 1 / 1024, 5.8251953125], [169.578125, 5.8251953125 + 1 / 1024]]) {
    const row = contactExample(phase, 1, "buyerBadge", width, height);
    assert.throws(() => validateLayoutGeometry(row, expected(row)), /intersections/);
  }
});

test("unseen pairs, phases and indices have no global positive-intersection tolerance", () => {
  for (const row of [contactExample("hook", 0, "buyerBadge", 0.25, 0.25), contactExample("question", 1, "dealerBadge", 0.25, 0.25)]) {
    assert.throws(() => validateLayoutGeometry(row, expected(row)), /intersections/);
  }
  const phase = contactExample("hook"); phase.phase = "outcome";
  assert.throws(() => validateLayoutGeometry(phase, expected(phase)), /unsupported measured phase/);
  const index = contactExample("hook"); index.roles.namePills.items[1].id = "namePills:7";
  assert.throws(() => validateLayoutGeometry(index, expected(index)), /invalid rectangle/);
});

test("a disallowed contact fails without erasing raw or permitted contacts", () => {
  const row = contactExample("hook");
  Object.assign(row.roles.dealerBadge.items[0], {left: 450, top: 295, right: 650, bottom: 350});
  assert.throws(() => validateLayoutGeometry(row, expected(row)), error => {
    assert.equal(error.validation.status, "failed");
    assert.equal(error.validation.intersections.length, 3);
    assert.equal(error.validation.designedAdjacencies.length, 1);
    assert.equal(error.validation.disallowedIntersections.length, 2);
    assert.equal(error.message, `layout gate hook: intersections ${JSON.stringify(error.validation.intersections)}`);
    return true;
  });
});

test("calibration table rejects malformed/nonfinite/unmeasured maxima and duplicate keys", () => {
  assert.equal(validateAdjacencyTable(table).size, 3);
  for (const change of [
    value => { value.rows = []; },
    value => { value.rows[0].observedMaxWidth = NaN; },
    value => { value.rows[0].observedMaxHeight = Infinity; },
    value => { value.rows[0].observedMaxWidth++; },
    value => { value.rows[0].marginEachAxis = 3; },
    value => { value.rows[0].pair[0] = "namePills:*"; },
    value => { value.rows[0].pair[0] = "namePills:0"; },
    value => { value.rows[0].phase = "outcome"; },
    value => { value.rows[1] = structuredClone(value.rows[0]); value.rows[1].pair.reverse(); },
  ]) {
    const changed = structuredClone(table); change(changed);
    assert.throws(() => validateAdjacencyTable(changed), /layout adjacency table/);
  }
});

test("unavailable or invalid table prevents module admission", async () => {
  for (const contents of [undefined, "{", JSON.stringify({...table, rows: []})]) {
    const directory = await mkdtemp(path.join(os.tmpdir(), "baloot-layout-table-"));
    await writeFile(path.join(directory, "layout-gate.mjs"), await readFile(new URL("../scripts/layout-gate.mjs", import.meta.url)));
    if (contents !== undefined) await writeFile(path.join(directory, "approved-layout-adjacencies.json"), contents);
    await assert.rejects(import(pathToFileURL(path.join(directory, "layout-gate.mjs"))), /ENOENT|JSON|layout adjacency table/);
  }
});

test("measured table bytes must match the table actually loaded by the validator", async () => {
  const {root, manifest, saveManifest} = await archive();
  const changedBytes = Buffer.concat([tableBytes, Buffer.from("\n")]);
  await writeFile(path.join(root, adjacencyTableFile), changedBytes);
  manifest.files[adjacencyTableFile].sha256 = sha256(changedBytes); await saveManifest();
  await assert.rejects(sourceBinding(root), /input mismatch\/unavailable/);
});

// Geometry-only regression of the measured deal04 overflow; no capture receipts.
for (const phase of ["hook", "question", "reveal"]) test(`held deal04 ${phase} remains FAIL/FAIL/PASS`, () => {
  const row = example(phase);
  if (phase === "reveal") {
    assert.equal(validateLayoutGeometry(row, expected(row)).status, "passed");
  } else {
    Object.assign(row.roles.promptPill.items[0], {left: 0, top: 100, right: 100, bottom: 200});
    Object.assign(row.roles.buyerBadge.items[0], {left: 54.03125, top: 164.1748046875, right: 150, bottom: 250});
    assert.throws(() => validateLayoutGeometry(row, expected(row)), error => {
      assert.deepEqual(error.validation.disallowedIntersections, [{first: "promptPill:0", second: "buyerBadge:0", width: 45.96875, height: 35.8251953125}]);
      return true;
    });
  }
});
