import {availableParallelism} from "node:os";
import {existsSync, readFileSync} from "node:fs";
import {execFileSync} from "node:child_process";
import {posix} from "node:path";
import {fileURLToPath} from "node:url";
import {RenderInternals} from "@remotion/renderer";

// Direct source-checkout renders use the same cap as Python and its children.
if (process.platform === "win32" && existsSync(new URL("../../../../../pyproject.toml", import.meta.url))) {
  execFileSync("powershell.exe", [
    "-NoProfile", "-File",
    fileURLToPath(new URL("../../../../../scripts/run_local_capped.ps1", import.meta.url)),
    "-AttachRootId", String(process.pid),
  ], {windowsHide: true, timeout: 30000, stdio: "pipe"});
}

const x264Presets = new Set([
  "superfast",
  "veryfast",
  "faster",
  "fast",
  "medium",
  "slow",
  "slower",
  "veryslow",
  "placebo",
]);

export const cgroupCpuLimit = (read = readFileSync) => {
  const text = filename => {
    try { return read(filename, "utf8").trim(); }
    catch { return ""; }
  };
  const groups = [["/sys/fs/cgroup", "", false]];
  for (const line of text("/proc/self/cgroup").split("\n")) {
    const [, controllers, group] = line.split(":");
    if (controllers === "") groups.push(["/sys/fs/cgroup", group, false]);
    else if (controllers?.split(",").includes("cpu")) {
      for (const mount of ["cpu", "cpu,cpuacct"]) {
        groups.push([`/sys/fs/cgroup/${mount}`, group, true]);
      }
    }
  }
  let limit = Infinity;
  for (const [root, group, legacy] of groups) {
    let directory = posix.join(root, group);
    while (directory === root || directory.startsWith(`${root}/`)) {
      const [quota, period] = (legacy
        ? `${text(`${directory}/cpu.cfs_quota_us`)} ${text(`${directory}/cpu.cfs_period_us`)}`
        : text(`${directory}/cpu.max`)).split(/\s+/).map(Number);
      if (Number.isFinite(quota) && quota > 0 && Number.isFinite(period) && period > 0) {
        limit = Math.min(limit, Math.max(1, Math.floor(quota / period)));
      }
      if (directory === root) break;
      directory = posix.dirname(directory);
    }
  }
  return limit;
};

export const governedH264Options = render => {
  if (render?.codec !== "libx264" || render?.pixel_format !== "yuv420p") {
    throw new Error("the governed Remotion renderer supports libx264/yuv420p only");
  }
  if (!x264Presets.has(render.preset)) {
    throw new Error(`unsupported governed x264 preset: ${render.preset}`);
  }
  for (const [name, value, minimum, maximum] of [
    ["concurrency", render.concurrency, 1, 32],
    ["crf", render.crf, 0, 51],
    ["audio_rate", render.audio_rate, 8000, 192000],
  ]) {
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
      throw new Error(`invalid governed render ${name}: ${value}`);
    }
  }
  return {
    codec: "h264",
    pixelFormat: render.pixel_format,
    imageFormat: "jpeg",
    crf: render.crf,
    x264Preset: render.preset,
    // Affinity/nproc can expose the host CPUs despite a narrower cgroup quota.
    concurrency: Math.min(render.concurrency, availableParallelism(), RenderInternals.getMaxConcurrency(), cgroupCpuLimit()),
    sampleRate: render.audio_rate,
  };
};

export const hostSafeConcurrency = (requested, logicalCpus, sustainedBusy, quotaCpus = logicalCpus) => {
  const effectiveCpus = Math.min(logicalCpus, quotaCpus);
  if (effectiveCpus <= 8 || effectiveCpus < logicalCpus) {
    const pressureCap = Math.max(1, Math.floor((1 - sustainedBusy) * effectiveCpus));
    return Math.max(1, Math.min(requested, effectiveCpus - 1, pressureCap));
  }
  const baselineCap = Math.max(1, Math.floor(logicalCpus / 6));
  const pressureCap = Math.max(1, Math.floor((0.5 - sustainedBusy) * logicalCpus / 2));
  return Math.min(requested, baselineCap, pressureCap);
};
