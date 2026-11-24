import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { expect, test, vi } from "vitest";

const studio = readFileSync(
  new URL("../../src/baloot_platform/service/static/configuration/index.html", import.meta.url),
  "utf8",
);

function loadDraftForm() {
  const controls = new Map();
  const control = (id: string) => {
    if (!controls.has(id)) {
      controls.set(id, {
        value: "",
        required: false,
        style: {},
        focus() {},
        classList: { contains: () => false },
        addEventListener(_event: string, callback: () => void) {
          Object.assign(this, { input: callback });
        },
      });
    }
    return controls.get(id);
  };
  const fetch = vi.fn((_path: string, options: { method?: string; body?: string }) =>
    options.method === "POST"
      ? Promise.resolve({ ok: true, json: async () => ({ revisionId: "CPR-TEST" }) })
      : new Promise(() => {}),
  );
  const script = studio.split("<script>")[1].split("</script>")[0];
  runInNewContext(script, {
    document: { getElementById: control, querySelectorAll: () => [] },
    fetch,
    URLSearchParams,
    setTimeout: () => {},
  });
  const submit = async () => {
    const endpoint = "/api/v1/configuration-registry/profiles";
    await control("draftForm").onsubmit({
      preventDefault: () => {},
      currentTarget: { getAttribute: () => endpoint },
    });
    const post = fetch.mock.calls.find(([path, options]) => path === endpoint && options.method === "POST");
    expect(post?.[1].body).toBeDefined();
    return JSON.parse(post![1].body!);
  };
  return { control, submit };
}

test("schedule submission sends browser wall time with the chosen IANA zone", async () => {
  const { control, submit } = loadDraftForm();
  control("effectiveFrom").value = "2026-11-01T01:30";
  control("effectiveFrom").input();
  expect(control("timeZone").required).toBe(true);
  control("timeZone").value = " America/New_York ";
  const request = await submit();
  expect(request.localSchedule).toEqual({
    timeZone: "America/New_York",
    effectiveFrom: "2026-11-01T01:30",
    effectiveUntil: null,
  });
  expect(request).not.toHaveProperty("effectiveFrom");
});

test("a draft with both time fields cleared omits the schedule", async () => {
  const { control, submit } = loadDraftForm();
  control("effectiveUntil").value = "2026-07-01T17:00";
  control("effectiveUntil").input();
  control("effectiveUntil").value = "";
  control("effectiveUntil").input();
  expect(control("timeZone").required).toBe(false);
  expect(await submit()).not.toHaveProperty("localSchedule");
});
