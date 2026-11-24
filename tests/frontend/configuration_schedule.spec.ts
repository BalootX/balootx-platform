import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { expect, test, vi } from "vitest";

const studio = readFileSync(
  new URL("../../src/baloot_platform/service/static/configuration/index.html", import.meta.url),
  "utf8",
);

function loadDraftForm(refusal?: { msg: string }[]) {
  const controls = new Map();
  const control = (id: string) => {
    if (!controls.has(id)) {
      controls.set(id, {
        value: "",
        required: false,
        style: {},
        focus: vi.fn(),
        classList: { contains: () => false },
        addEventListener(_event: string, callback: () => void) {
          Object.assign(this, { input: callback });
        },
      });
    }
    return controls.get(id);
  };
  const fetch = vi.fn((_path: string, options: { method?: string; body?: string }) => {
    if (options.method !== "POST") return new Promise(() => {});
    return Promise.resolve({
      ok: !refusal,
      statusText: refusal ? "Unprocessable Content" : "Created",
      json: async () => refusal ? { detail: refusal } : { revisionId: "CPR-TEST" },
    });
  });
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

test("a refused schedule shows and focuses the server's validation messages", async () => {
  const message = "localSchedule.effectiveFrom is ambiguous in this time zone";
  const nextStep = "Choose a different local time";
  const { control, submit } = loadDraftForm([{ msg: message }, { msg: nextStep }]);
  control("effectiveFrom").value = "2026-11-01T01:30";
  control("timeZone").value = "America/New_York";
  await submit();
  const error = control("draftError");
  expect(error.hidden).toBe(false);
  expect(error.textContent).toBe(`${message} · ${nextStep}`);
  expect(error.focus).toHaveBeenCalledTimes(1);
  expect(control("effectiveFrom").value).toBe("2026-11-01T01:30");
});
