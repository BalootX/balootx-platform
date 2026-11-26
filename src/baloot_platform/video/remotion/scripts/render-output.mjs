import fs from "node:fs/promises";
import path from "node:path";

export async function createRenderOutput(directory, basename) {
  if (!basename || /[\\/:*?"<>|]/.test(basename)) {
    throw new Error("render basename is invalid");
  }
  const output = path.resolve(directory);
  await fs.mkdir(path.dirname(output), {recursive: true});
  await fs.mkdir(output, {mode: 0o700});
}

export async function writeRenderEvidence(file, data) {
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.tmp`);
  await fs.writeFile(temporary, data, {flag: "wx"});
  await fs.rename(temporary, file);
}
