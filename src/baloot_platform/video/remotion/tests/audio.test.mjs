import assert from "node:assert/strict";
import {existsSync, readFileSync} from "node:fs";
import {createRequire} from "node:module";
import path from "node:path";
import test from "node:test";
import {fileURLToPath} from "node:url";
import {runInThisContext} from "node:vm";
import React from "react";
import {renderToStaticMarkup} from "react-dom/server";
import ts from "typescript";

const rendererRoot = fileURLToPath(new URL("../", import.meta.url));
const require = createRequire(import.meta.url);
const loaded = new Map();
// Exercise the actual React compositions without Chromium or frame/audio I/O.
// Sequences expose their children so the assertion counts scheduled audio clips.
const remotion = {
  AbsoluteFill: ({children}) => React.createElement("div", null, children),
  Audio: ({src, volume}) => React.createElement("audio", {src, "data-volume": volume}),
  Easing: {bezier: () => value => value},
  Img: props => React.createElement("img", props),
  interpolate: () => 1,
  Sequence: ({children}) => children,
  staticFile: value => `/${value}`,
  useCurrentFrame: () => 60,
};

function loadComponent(relative) {
  let filename = path.join(rendererRoot, "src", relative);
  if (!path.extname(filename)) filename += existsSync(`${filename}.tsx`) ? ".tsx" : ".ts";
  if (loaded.has(filename)) return loaded.get(filename);
  const {outputText} = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
    fileName: filename,
  });
  const module = {exports: {}};
  const componentRequire = specifier => {
    if (specifier === "remotion") return remotion;
    if (specifier.endsWith(".css")) return {};
    if (specifier.startsWith("./")) return loadComponent(specifier.slice(2));
    return require(specifier);
  };
  runInThisContext(`(function(require, module, exports) {${outputText}\n})`, {filename})(
    componentRequire, module, module.exports,
  );
  loaded.set(filename, module.exports);
  return module.exports;
}

const {ProfessionalPrototype} = loadComponent("professional-prototype.tsx");
const {ProfessionalDecisionClip} = loadComponent("professional-decision-clip.tsx");
const {majlisPaletteTokens} = loadComponent("professional-prototype.tsx");
const majlis = {
  theme: "red", hud: "plaque", id_box: "qr", score_target: 152, social: {tiktok: "t", instagram: "i"},
  palette: Object.fromEntries(majlisPaletteTokens.map(token => [token, "#123456"])),
};

function audioBundle(exportPreset) {
  const observer = "east";
  const reasonKey = "video.analysis.expert_reason";
  const layer = (id, disclosure, content) => ({id, kind: "test", disclosure, content});
  return {
    schemaVersion: 1,
    bundleSha256: "0".repeat(64),
    exportPreset,
    locale: "ar-SA",
    direction: "rtl",
    observerSeat: observer,
    render: {width: 1080, height: 1920, fps: 60},
    // The explanation voice is justification, so show_justification_text=false silences it.
    presentation: {audio: {enabled: true}, presentation: {show_justification_text: true}, layout: {}, motion: {}, majlis},
    localizations: {
      [reasonKey]: "السبب المطابق للتسجيل",
      "video.accessibility.score_and_game": "النتيجة",
      "video.accessibility.hand": "يدك",
      "video.bid.sun": "صن",
      "video.bid.round.1": "الأول",
      "video.purchase.plain_summary": "{contract} {round}",
      "video.result.us": "لنا",
      "video.result.them": "لهم",
      "video.role.buyer": "المشتري",
      "video.role.dealer": "الموزع",
      "video.hud.trick": "الأكلة",
      "video.governance.research_short": "للبحث فقط",
      "video.governance.not_publishable": "غير مؤهل للنشر",
    },
    identity: {
      gameSerial: "AUDIO-TEST",
      sceneId: "QUESTION",
      enrollmentUri: "https://example.invalid/c/AUDIO-TEST",
      players: {north: "N", west: "W", south: "S", east: "E"},
    },
    audioCues: [{
      stage: "reveal", assetId: "voice-explanation", path: "audio/test-explanation.wav",
      volume: 0.42, transcriptKey: reasonKey, transcriptSha256: "1".repeat(64),
    }],
    scenes: [
      {
        schemaVersion: 1, sceneId: "QUESTION", stage: "question", observer,
        locale: "ar-SA", direction: "rtl",
        layers: [
          layer("table", "public", {currentPlayer: observer, contract: "sun", currentTrick: {plays: []}}),
          layer("observer-hand", "observer_private", {cards: []}),
          layer("seats", "public", {handCounts: {north: 0, west: 0, south: 0, east: 0}}),
          layer("decision-prompt", "public", {actingSeat: observer, promptContract: "choose_acting_action"}),
        ],
      },
      {
        schemaVersion: 1, sceneId: "REVEAL", stage: "reveal", observer,
        locale: "ar-SA", direction: "rtl",
        layers: [layer("analysis", "post_reveal_analysis", {reasonKey})],
      },
    ],
    timeline: {totalFrames: 120, segments: [
      {stage: "question", sceneId: "QUESTION", startFrame: 0, durationFrames: 60},
      {stage: "reveal", sceneId: "REVEAL", startFrame: 60, durationFrames: 60},
    ]},
  };
}

for (const [preset, Component, props] of [
  ["hooked_flagship", ProfessionalPrototype, {stage: "flagship"}],
  ["decision_clip", ProfessionalDecisionClip, {}],
]) {
  for (const enabled of [true, false, undefined]) {
    test(`${preset} schedules audio with audio.enabled=${enabled ?? "legacy default"}`, () => {
      const bundle = audioBundle(preset);
      if (enabled === undefined) delete bundle.presentation.audio;
      else bundle.presentation.audio = {enabled};
      const html = renderToStaticMarkup(React.createElement(Component, {bundle, ...props}));
      assert.equal((html.match(/<audio\b/g) ?? []).length, enabled === false ? 0 : 1);
      if (enabled !== false) {
        assert.match(html, /src="\/audio\/test-explanation\.wav"/);
        assert.match(html, /data-volume="0\.42"/);
      }
    });
  }

  {
    test(`${preset} never voices a reason with show_justification_text=false`, () => {
      const bundle = audioBundle(preset);
      bundle.presentation.presentation.show_justification_text = false;
      const html = renderToStaticMarkup(React.createElement(Component, {bundle, ...props}));
      assert.doesNotMatch(html, /<audio\b/);
      assert.doesNotMatch(html, /السبب المطابق للتسجيل/);
    });
  }

  for (const missingCue of [true, false]) {
    test(`${preset} suppresses audio with a ${missingCue ? "missing" : "mismatched"} governed cue`, () => {
      const bundle = audioBundle(preset);
      if (missingCue) bundle.audioCues = [];
      else bundle.audioCues[0].transcriptKey = "video.analysis.other_reason";
      const html = renderToStaticMarkup(React.createElement(Component, {bundle, ...props}));
      assert.doesNotMatch(html, /<audio\b/);
    });
  }
}

test("prototype audio suppression still overrides an enabled bundle", () => {
  const bundle = audioBundle("hooked_flagship");
  const html = renderToStaticMarkup(React.createElement(ProfessionalPrototype, {
    bundle, stage: "flagship", suppressAudio: true,
  }));
  assert.doesNotMatch(html, /<audio\b/);
});
