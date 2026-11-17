import {readFile, writeFile} from "node:fs/promises";
import jpeg from "jpeg-js";
import jsQR from "jsqr";

const imagePath = process.env.BALOOTX_QR_IMAGE;
const expectedValue = process.env.BALOOTX_QR_EXPECTED;
const outputPath = process.env.BALOOTX_QR_RESULT;
const policy = process.env.BALOOTX_QR_POLICY;
const reason = process.env.BALOOTX_QR_REASON;

if (!imagePath || !outputPath || !["required", "forbidden"].includes(policy)) {
  throw new Error("QR verification requires image, result, and governed policy");
}
if (policy === "required" && !expectedValue) {
  throw new Error("required QR verification needs the governed expected value");
}

const encoded = await readFile(imagePath);
const decoded = jpeg.decode(encoded, {useTArray: true});
const pixels = new Uint8ClampedArray(decoded.data);
const candidate = jsQR(pixels, decoded.width, decoded.height, {
  inversionAttempts: "attemptBoth",
});
// Card artwork can produce a degenerate finder with no decoded payload.
const result = candidate?.data === "" && candidate.binaryData?.length === 0 && candidate.chunks?.length === 0
  ? null : candidate;

if (policy === "required" && !result) {
  throw new Error("rendered poster QR could not be decoded");
}
if (policy === "required" && result.data !== expectedValue) {
  throw new Error("rendered poster QR does not match the governed enrollment URI");
}
if (policy === "forbidden" && result) {
  throw new Error("rendered poster contains a forbidden actionable QR code");
}

await writeFile(
  outputPath,
  `${JSON.stringify({
    schemaVersion: 1,
    policy,
    reason: reason || null,
    passed: true,
    decodedValue: result?.data ?? null,
  })}\n`,
  "utf8",
);
