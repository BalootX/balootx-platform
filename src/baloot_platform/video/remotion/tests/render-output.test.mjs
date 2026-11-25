import assert from "node:assert/strict";
import fs from "node:fs/promises";
import {tmpdir} from "node:os";
import path from "node:path";
import test from "node:test";
import {createRenderOutput, writeRenderEvidence} from "../scripts/render-output.mjs";

test("only one render can claim an output directory and publish complete evidence", async t => {
  const root = await fs.mkdtemp(path.join(tmpdir(), "baloot-render-"));
  t.after(() => fs.rm(root, {recursive: true}));
  const output = path.join(root, "outputs", "render");
  await assert.rejects(createRenderOutput(output, "../outside"), /render basename is invalid/);
  const claims = await Promise.allSettled([
    createRenderOutput(output, "render"),
    createRenderOutput(output, "render"),
  ]);
  assert.equal(claims.filter(claim => claim.status === "fulfilled").length, 1);
  assert.equal(claims.find(claim => claim.status === "rejected").reason.code, "EEXIST");

  const file = path.join(output, "evidence.json");
  await writeRenderEvidence(file, '{"status":"running"}\n');
  const final = Buffer.from('{"status":"passed"}\n');
  await writeRenderEvidence(file, final);
  assert.deepEqual(await fs.readFile(file), final);
  assert.deepEqual(await fs.readdir(output), ["evidence.json"]);
});

test("evidence write and rename errors escape on the first attempt", async t => {
  for (const code of ["EBUSY", "EACCES", "EPERM", "UNKNOWN"]) {
    const error = Object.assign(new Error("injected write failure"), {code});
    const write = t.mock.method(fs, "writeFile", async () => { throw error; });
    const rename = t.mock.method(fs, "rename");
    await assert.rejects(writeRenderEvidence("evidence.json", "{}"), reason => reason === error);
    assert.equal(write.mock.callCount(), 1);
    assert.equal(rename.mock.callCount(), 0);
    t.mock.restoreAll();
  }

  const root = await fs.mkdtemp(path.join(tmpdir(), "baloot-render-"));
  t.after(() => fs.rm(root, {recursive: true}));
  const file = path.join(root, "evidence.json");
  await writeRenderEvidence(file, '{"status":"running"}');
  const error = Object.assign(new Error("injected rename failure"), {code: "EACCES"});
  const rename = t.mock.method(fs, "rename", async () => { throw error; });
  await assert.rejects(writeRenderEvidence(file, '{"status":"passed"}'), reason => reason === error);
  assert.equal(rename.mock.callCount(), 1);
  assert.equal(await fs.readFile(file, "utf8"), '{"status":"running"}');
});
