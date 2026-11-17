import {afterEach, beforeEach, expect, test, vi} from "vitest";

const mocks = vi.hoisted(() => ({decode: vi.fn(), write: vi.fn()}));
vi.mock("node:fs/promises", async (original) => ({
  ...await original<typeof import("node:fs/promises")>(),
  readFile: vi.fn(async () => new Uint8Array()),
  writeFile: mocks.write,
}));
vi.mock("../../src/baloot_platform/video/remotion/node_modules/jpeg-js/index.js", () => ({
  default: {decode: () => ({data: new Uint8Array(4), width: 1, height: 1})},
}));
vi.mock("../../src/baloot_platform/video/remotion/node_modules/jsqr/dist/jsQR.js", () => ({default: mocks.decode}));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.stubEnv("BALOOTX_QR_IMAGE", "test.jpg");
  vi.stubEnv("BALOOTX_QR_RESULT", "result.json");
  vi.stubEnv("BALOOTX_QR_EXPECTED", "https://example.test/challenge");
  vi.stubEnv("BALOOTX_QR_POLICY", "forbidden");
});

const run = () => import("../../src/baloot_platform/video/remotion/scripts/verify-qr.mjs");
afterEach(() => vi.unstubAllEnvs());

test("empty card-art decoder result is not an actionable QR", async () => {
  mocks.decode.mockReturnValue({data: "", binaryData: [], chunks: [], version: 1});
  await run();
  expect(JSON.parse(mocks.write.mock.calls[0][1]).decodedValue).toBeNull();
});

test.each([
  {data: "https://example.test/challenge", binaryData: [1], chunks: [{}]},
  {data: "", binaryData: [1], chunks: []},
  {data: "", binaryData: [], chunks: [{}]},
  {data: "", binaryData: undefined, chunks: []},
])("forbidden mode rejects nonempty or ambiguous payload %j", async (candidate) => {
  mocks.decode.mockReturnValue(candidate);
  await expect(run()).rejects.toThrow("forbidden actionable QR");
  expect(mocks.write).not.toHaveBeenCalled();
});

test("required mode rejects normalized empty result", async () => {
  vi.stubEnv("BALOOTX_QR_POLICY", "required");
  mocks.decode.mockReturnValue({data: "", binaryData: [], chunks: []});
  await expect(run()).rejects.toThrow("could not be decoded");
});

test("required mode accepts its exact URI and rejects a different URI", async () => {
  vi.stubEnv("BALOOTX_QR_POLICY", "required");
  mocks.decode.mockReturnValue({data: "https://example.test/challenge", binaryData: [1], chunks: [{}]});
  await run();
  expect(JSON.parse(mocks.write.mock.calls[0][1]).passed).toBe(true);
  vi.resetModules();
  vi.stubEnv("BALOOTX_QR_EXPECTED", "https://example.test/wrong");
  await expect(run()).rejects.toThrow("does not match");
});
