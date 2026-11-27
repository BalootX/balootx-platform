import {execFileSync} from "node:child_process";
import {createRequire} from "node:module";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";

import {expect, test, vi} from "vitest";

vi.mock("../../src/baloot_platform/video/remotion/node_modules/remotion/dist/esm/index.mjs", async (importOriginal) => ({
  ...await importOriginal<typeof import("remotion")>(),
  Img: "img",
}));

import {ProfessionalPrototypeFrame} from "../../src/baloot_platform/video/remotion/src/professional-prototype.js";
import {trickCollectionSampleFrameRange} from "../../src/baloot_platform/video/remotion/src/playback-timing.mjs";
import {
  derivePlaybackState,
  type PlaybackPlay,
  type PlaybackTrick,
  type Seat,
} from "../../src/baloot_platform/video/remotion/src/playback-state.js";
import type {CardValue, RenderBundle} from "../../src/baloot_platform/video/remotion/src/types.js";

const remotionRequire = createRequire(new URL(
  "../../src/baloot_platform/video/remotion/package.json",
  import.meta.url,
));
const {createElement} = remotionRequire("react") as typeof import("react");
const {renderToStaticMarkup} = remotionRequire("react-dom/server") as typeof import("react-dom/server");

const seats: Seat[] = ["north", "west", "south", "east"];
const sourcePlays: Array<[Seat, number, string]> = [
  ["north", 30, "CK"], ["west", 27, "C10"], ["south", 29, "CQ"], ["east", 28, "CJ"],
  ["west", 15, "HA"], ["south", 11, "H10"], ["east", 14, "HK"], ["north", 22, "DK"],
  ["west", 6, "SK"], ["south", 1, "S8"], ["east", 8, "H7"], ["north", 21, "DQ"],
  ["west", 5, "SQ"], ["south", 4, "SJ"], ["east", 13, "HQ"], ["north", 19, "D10"],
  ["west", 2, "S9"], ["south", 3, "S10"], ["east", 12, "HJ"], ["north", 20, "DJ"],
  ["south", 9, "H8"], ["east", 10, "H9"], ["north", 18, "D9"], ["west", 31, "CA"],
  ["east", 25, "C8"], ["north", 24, "C7"], ["west", 16, "D7"], ["south", 0, "S7"],
  ["east", 26, "C9"], ["north", 23, "DA"], ["west", 17, "D8"], ["south", 7, "SA"],
];
const suitByCode: Record<string, CardValue["suit"]> = {
  S: "spades", H: "hearts", C: "clubs", D: "diamonds",
};
const rankByCode: Record<string, string> = {
  "7": "seven", "8": "eight", "9": "nine", "10": "ten",
  J: "jack", Q: "queen", K: "king", A: "ace",
};
const plays: PlaybackPlay[] = sourcePlays.map(([seat, id, code], index) => ({
  seat,
  card: {id, suit: suitByCode[code[0]], rank: rankByCode[code.slice(1)]},
  trickIndex: Math.floor(index / 4),
  position: index % 4,
}));
const trickWinners: Seat[] = ["west", "west", "west", "west", "south", "east", "east", "east"];
const tricks: PlaybackTrick[] = Array.from({length: 8}, (_, index) => ({
  index,
  plays: plays.slice(index * 4, index * 4 + 4),
  winner: trickWinners[index],
}));
const observer: Seat = "south";
const initialHand = plays.filter((play) => play.seat === observer).map((play) => play.card);
const qualifiedOrToolsSource = {
  resultCode: "LAF-A7D8EFCC57",
  exactStateId: "STATE-4D2B3E1AC0057FFBD977",
  publicProjectionHash: "e97d562a0357b4bf02ca5fff6d9b2f2f76af3dc8ed5cafabcc128b0ffef997cb",
  rulesHash: "a2eb1f6d6ca8b1a05d43faddad0678116b247e15d17174faa0fea5e66bcad510",
  traceHash: "f2d2f02d07e7ab4236b0b9b4eede8f32a8948b6e0a96101eed3a1d9a39eb6b96",
  researchAuthority: {
    backendId: "ortools_cp_sat_v3",
    backendVersion: "3.0.0",
    qualificationAuthority: "cp_sat",
    solverVersion: "9.15.6755",
    bundleRevisionId: "CBR-9F7A0969B419493C4C6D7725",
    proofOutcomeHash: "79f86d5b5c5f3729b6a037d544f2cbe709ffe1e1f3e3984a7d770b9e3079ac35",
  },
  runtimeCompatibility: {status: "passed"},
  trace: {initialHand, plays, completedTricks: tricks},
} as const;

// Use the same governed palette and research labels as production render bundles.
const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const governedPresentation = JSON.parse(execFileSync("python", ["-c", [
  "import json, yaml",
  "from baloot_platform.video.render_bundle import resolve_majlis_presentation",
  "profile = yaml.safe_load(open('configs/video/premium.yaml', encoding='utf-8'))",
  "print(json.dumps(dict(majlis=resolve_majlis_presentation(profile['majlis']), localizations=profile['localizations'])))",
].join(";")], {
  cwd: repositoryRoot,
  encoding: "utf8",
  env: {...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: `${repositoryRoot}src`},
})) as {majlis: Record<string, unknown>; localizations: Record<string, string>};

const layer = (id: string, disclosure: "public" | "observer_private" | "post_reveal_analysis", content: Record<string, unknown>) => ({
  id,
  kind: "test",
  disclosure,
  content,
});

const bundle = {
  schemaVersion: 1,
  bundleSha256: "0".repeat(64),
  exportPreset: "hooked_flagship",
  locale: "ar-SA",
  direction: "rtl",
  observerSeat: observer,
  source: {
    sourceSchemaVersion: 3,
    kind: "research_scenario_result",
    code: qualifiedOrToolsSource.resultCode,
    decisionOrdinal: 1,
    decisionReference: {resultCode: qualifiedOrToolsSource.resultCode, decisionOrdinal: 1},
    publicProjectionHash: qualifiedOrToolsSource.publicProjectionHash,
    qualificationStatus: "qualified",
    auditOnly: false,
    renderPurpose: "content_candidate",
    qualificationFailures: [],
    publicationBlockers: [],
    publicationEligible: true,
    publicationPolicy: {},
    observerSeat: observer,
    qrActionable: false,
    rulesHash: qualifiedOrToolsSource.rulesHash,
    proofReference: qualifiedOrToolsSource.resultCode,
    proofCompletionStatus: "exact_complete",
    proofClaimSatisfied: true,
    proofAchievedMode: "exact",
  },
  render: {width: 1080, height: 1920, fps: 60},
  presentation: {
    presentation: {show_identity_pill: false, show_enrollment_qr: false},
    colors: {},
    layout: {show_projects: false, show_score_breakdown: false},
    timeline: {pause_seconds: 1},
    motion: {collection_steps: 1},
    majlis: governedPresentation.majlis,
  },
  localizations: {
    ...governedPresentation.localizations,
    "video.accessibility.decision_choices": "الخيارات القانونية",
    "video.accessibility.hand": "يدك",
    "video.accessibility.score_and_game": "النتيجة",
    "video.analysis.expert_reason": "السبب",
    "video.bid.round.1": "الأول",
    "video.bid.pass": "بس",
    "video.bid.sun": "صن",
    "video.escalation.decline": "بس",
    "video.escalation.double": "دبل",
    "video.escalation.mode.open": "مفتوح",
    "video.decision.choose_best_move": "وش تلعب؟",
    "video.decision.timer_open": "مفتوح",
    "video.purchase.plain_summary": "{contract} {round}",
    "video.project.hundred_sequence": "مية",
    "video.result.title": "النتيجة",
    "video.result.round": "اللعب",
    "video.result.projects": "المشاريع",
    "video.result.awarded": "المحتسب",
    "video.result.total": "المجموع",
    "video.result.them": "هم",
    "video.result.us": "حنا",
    "video.role.buyer": "المشتري",
    "video.role.dealer": "الموزع",
  },
  identity: {
    gameSerial: qualifiedOrToolsSource.resultCode,
    sceneId: "SCENE-TEST",
    enrollmentUri: "https://example.invalid",
    players: {north: "north", west: "west", south: "south", east: "east"},
  },
  audioCues: [],
  scenes: [
    {
      schemaVersion: 1,
      sceneId: "SCENE-CONTINUATION",
      stage: "continuation",
      observer,
      locale: "ar-SA",
      direction: "rtl",
      layers: [
        layer("public-lifecycle", "post_reveal_analysis", {
          plays: qualifiedOrToolsSource.trace.plays,
          completedTricks: qualifiedOrToolsSource.trace.completedTricks,
          firstLeader: "north",
          pauseBeforePlayIndex: 0,
          dealer: "east",
          buyer: "west",
          contract: "sun",
          bidHistory: [{action: "sun", round: 1}],
          projects: [],
          preDecisionMatchScore: [0, 0],
          matchScore: [0, 0],
          roundScore: {
            cardGamePoints: [0, 0],
            projectPoints: [0, 0],
            balootPoints: [0, 0],
            awardedPoints: [0, 0],
          },
        }),
        layer("observer-lifecycle", "post_reveal_analysis", {
          initialHandCards: qualifiedOrToolsSource.trace.initialHand,
        }),
      ],
    },
    {
      schemaVersion: 1,
      sceneId: "SCENE-QUESTION",
      stage: "question",
      observer,
      locale: "ar-SA",
      direction: "rtl",
      layers: [
        layer("table", "public", {currentPlayer: "north", currentTrick: {plays: []}}),
        layer("observer-hand", "observer_private", {cards: initialHand}),
        layer("seats", "public", {handCounts: {north: 8, west: 8, south: 8, east: 8}}),
        layer("decision-prompt", "public", {actingSeat: "north"}),
      ],
    },
    {
      schemaVersion: 1,
      sceneId: "SCENE-REVEAL",
      stage: "reveal",
      observer,
      locale: "ar-SA",
      direction: "rtl",
      layers: [layer("analysis", "post_reveal_analysis", {reasonKey: "video.analysis.expert_reason"})],
    },
  ],
  timeline: {
    totalFrames: 33,
    segments: [
      {stage: "hook", sceneId: "SCENE-QUESTION", startFrame: 0, durationFrames: 1},
      {stage: "continuation", sceneId: "SCENE-CONTINUATION", startFrame: 1, durationFrames: 32},
    ],
  },
} as RenderBundle;

bundle.scenes.push({
  schemaVersion: 1, sceneId: "SCENE-CONTEXT", stage: "context", observer,
  locale: "ar-SA", direction: "rtl",
  layers: [
    layer("public-lifecycle", "public", {
      plays: [], completedTricks: [], firstLeader: "north", pauseBeforePlayIndex: 0,
      dealer: "east", buyer: "west", contract: "sun", bidHistory: [{action: "sun", round: 1}],
      projects: [], preDecisionMatchScore: [0, 0], matchScore: [0, 0],
    }),
    layer("observer-lifecycle", "observer_private", {initialHandCards: initialHand}),
  ],
});

const cardCode = (card: CardValue): string => {
  const suit = {spades: "S", hearts: "H", clubs: "C", diamonds: "D"}[card.suit];
  const rank = {seven: "7", eight: "8", nine: "9", ten: "10", jack: "J", queen: "Q", king: "K", ace: "A"}[card.rank];
  return `${suit}${rank}`;
};

const renderedCodes = (markup: string): string[] =>
  [...markup.matchAll(/data-card-code="([^"]+)"/g)].map((match) => match[1]).sort();

test.each([false, true])("replay and collection sample follow play boundaries with context=%s", (withContext) => {
  const candidate = structuredClone(bundle);
  candidate.scenes.find((scene) => scene.stage === "continuation")!.layers
    .find((item) => item.id === "public-lifecycle")!.content.pauseBeforePlayIndex = 24;
  Object.assign(candidate.scenes.find((scene) => scene.stage === "context")!.layers
    .find((item) => item.id === "public-lifecycle")!.content,
    {pauseBeforePlayIndex: 24, plays: plays.slice(0, 24), completedTricks: tricks.slice(0, 6)});
  candidate.timeline = {totalFrames: 4320, segments: [
    {stage: "hook", sceneId: "SCENE-QUESTION", startFrame: 0, durationFrames: 900},
    {stage: "replay", sceneId: "SCENE-CONTEXT", startFrame: 900, durationFrames: withContext ? 1890 : 1980},
    ...(withContext ? [{stage: "context", sceneId: "SCENE-CONTEXT", startFrame: 2790, durationFrames: 90}] : []),
    {stage: "question", sceneId: "SCENE-QUESTION", startFrame: 2880, durationFrames: 120},
    {stage: "continuation", sceneId: "SCENE-CONTINUATION", startFrame: 3000, durationFrames: 1320},
  ]};
  const markup = (frame: number) => renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: candidate, compositionFrame: frame, stage: "flagship", reducedMotion: false,
    storyFrame: frame, suppressAudio: true,
  }));
  const lastStart = withContext ? 2790 : 900 + Math.ceil(23 * 1980 / 24);
  expect(markup(lastStart - 1)).toContain('data-completed-play-count="23"');
  expect(markup(lastStart)).toContain('data-completed-play-count="24"');
  expect(markup(lastStart)).toContain('data-collection-progress="0"');
  expect(markup(2879)).toContain('data-completed-play-count="24"');
  expect(markup(2879)).toContain('data-table-opacity="0"');
  expect(markup(2880)).toContain('data-completed-play-count="24"');

  const [beforeCollection, nextPlay] = trickCollectionSampleFrameRange(candidate);
  expect([beforeCollection, nextPlay]).toEqual(withContext ? [1225, 1229] : [1227, 1230]);
  expect(markup(beforeCollection)).toContain('data-completed-play-count="4"');
  expect(markup(beforeCollection)).toContain('data-collection-progress="0"');
  expect(Number(markup(beforeCollection + 1).match(/data-collection-progress="([^"]+)"/)?.[1])).toBeGreaterThan(0);
  expect(markup(nextPlay - 1)).toContain('data-table-opacity="0"');
  expect(markup(nextPlay)).toContain('data-completed-play-count="5"');

  candidate.presentation.motion.collection_steps = 0;
  const [beforeInstantCollection] = trickCollectionSampleFrameRange(candidate);
  expect(markup(beforeInstantCollection)).toContain('data-collection-progress="0"');
  expect(markup(beforeInstantCollection + 1)).toContain('data-collection-progress="1"');
});

test.each([false, true])("encoded checkpoint captures pre-question play in playback with context=%s", (withContext) => {
  const source = readFileSync(new URL('../../src/baloot_platform/video/remotion/scripts/render-flagship.mjs', import.meta.url), 'utf8');
  const excerpt = source.slice(source.indexOf('const pausePlayIndex ='), source.indexOf('if (!collectionOnly) for'));
  const segments = [
    {stage: 'replay', startFrame: 900, durationFrames: withContext ? 1890 : 1980},
    ...(withContext ? [{stage: 'context', startFrame: 2790, durationFrames: 90}] : []),
    {stage: 'question', startFrame: 2880, durationFrames: 120},
  ];
  const checkpoint = new Function('publicLifecycle', 'allPlays', 'timeline', 'segment',
    excerpt + '; return playbackFrame(24);');
  expect(checkpoint({pauseBeforePlayIndex: 24}, plays, {segments},
    (stage: string) => segments.find((segment) => segment.stage === stage))).toBe(2879);
});

test("qualified OR-Tools bundle renders all 33 source-state boundaries without hidden-card leaks", () => {
  expect(qualifiedOrToolsSource.researchAuthority).toMatchObject({
    backendId: "ortools_cp_sat_v3",
    qualificationAuthority: "cp_sat",
  });
  expect(qualifiedOrToolsSource.runtimeCompatibility.status).toBe("passed");
  expect(bundle.source).toMatchObject({
    code: qualifiedOrToolsSource.resultCode,
    publicProjectionHash: qualifiedOrToolsSource.publicProjectionHash,
    qualificationStatus: "qualified",
    proofCompletionStatus: "exact_complete",
  });

  const immutableSource = JSON.stringify(qualifiedOrToolsSource);
  let previousCounts = {north: 8, west: 8, south: 8, east: 8};

  for (let playCount = 0; playCount <= qualifiedOrToolsSource.trace.plays.length; playCount += 1) {
    const visibleSourcePlays = qualifiedOrToolsSource.trace.plays.slice(0, playCount);
    const expectedHand = qualifiedOrToolsSource.trace.initialHand.filter((card) =>
      !visibleSourcePlays.some((play) => play.seat === observer && play.card.id === card.id),
    );
    const expectedCounts = Object.fromEntries(seats.map((seat) => [
      seat,
      8 - visibleSourcePlays.filter((play) => play.seat === seat).length,
    ])) as Record<Seat, number>;
    expectedCounts[observer] = expectedHand.length;
    const position = playCount % 4;
    const expectedCurrentTrick = position === 0
      ? []
      : visibleSourcePlays.slice(playCount - position);
    const expectedCompletedTrick = playCount > 0 && position === 0
      ? qualifiedOrToolsSource.trace.completedTricks[playCount / 4 - 1]
      : undefined;
    const expectedTablePlays = expectedCompletedTrick?.plays ?? expectedCurrentTrick;

    const state = derivePlaybackState({
      initialHand: qualifiedOrToolsSource.trace.initialHand,
      allPlays: qualifiedOrToolsSource.trace.plays,
      allTricks: qualifiedOrToolsSource.trace.completedTricks,
      observer,
      completedPlayCount: playCount,
    });
    const markup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
      bundle,
      compositionFrame: playCount,
      stage: "flagship",
      reducedMotion: true,
      storyFrame: playCount,
      suppressAudio: true,
    }));

    expect(markup).toContain(`data-completed-play-count="${playCount}"`);
    expect(markup).toContain(`data-source-play-count="32"`);
    expect(markup).toContain(`data-current-trick-count="${expectedCurrentTrick.length}"`);
    expect(markup).toContain(`data-table-play-count="${expectedTablePlays.length}"`);
    if (expectedCompletedTrick) {
      expect(markup).toContain(`data-completed-trick-index="${expectedCompletedTrick.index}"`);
    } else {
      expect(markup).not.toContain("data-completed-trick-index=");
    }

    expect(state.hand).toEqual(expectedHand);
    expect(state.handCounts).toEqual(expectedCounts);
    expect(state.currentTrick).toEqual(expectedCurrentTrick);
    expect(state.tablePlays).toEqual(expectedTablePlays);
    expect(state.completedTrickIndex).toBe(expectedCompletedTrick?.index ?? null);
    expect(markup).toContain(`data-seat="south" data-hand-count="${expectedHand.length}"`);
    for (const seat of seats.filter((seat) => seat !== observer)) {
      expect(markup).toContain(
        `data-seat="${seat}" data-hand-count="${expectedCounts[seat]}" data-card-back-count="${Math.min(6, expectedCounts[seat])}"`,
      );
    }

    const expectedFaces = [
      ...expectedHand.map(cardCode),
      ...expectedTablePlays.map((play) => cardCode(play.card)),
    ].sort();
    expect(renderedCodes(markup)).toEqual(expectedFaces);
    expect(markup.match(/class="back-stack[^"]*" aria-hidden="true"/g)).toHaveLength(3);

    if (playCount > 0) {
      const actor = qualifiedOrToolsSource.trace.plays[playCount - 1].seat;
      for (const seat of seats) {
        expect(expectedCounts[seat]).toBe(previousCounts[seat] - (seat === actor ? 1 : 0));
      }
    }
    previousCounts = expectedCounts;

    if (playCount > 0 && position === 0) {
      expect(state.currentTrick).toEqual([]);
      expect(state.tablePlays).toEqual(qualifiedOrToolsSource.trace.completedTricks[playCount / 4 - 1].plays);
    } else {
      expect(state.currentTrick).toHaveLength(position);
      expect(state.tablePlays).toEqual(state.currentTrick);
    }

    expect(qualifiedOrToolsSource.trace.completedTricks).toHaveLength(8);
    expect(JSON.stringify(qualifiedOrToolsSource)).toBe(immutableSource);
    expect(markup).not.toContain(qualifiedOrToolsSource.exactStateId);
    expect(markup).not.toContain(qualifiedOrToolsSource.researchAuthority.proofOutcomeHash);
  }

  expect(qualifiedOrToolsSource.trace.initialHand).toHaveLength(8);
  expect(derivePlaybackState({
    initialHand: qualifiedOrToolsSource.trace.initialHand,
    allPlays: qualifiedOrToolsSource.trace.plays,
    allTricks: qualifiedOrToolsSource.trace.completedTricks,
    observer,
    completedPlayCount: 32,
  }).hand).toHaveLength(0);
});

test("complete replay shows five-card auction then the purchase and escalation before play", () => {
  const setupBundle: RenderBundle = structuredClone(bundle);
  const context = setupBundle.scenes.find((scene) => scene.stage === "context");
  const publicLayer = context?.layers.find((layer) => layer.id === "public-lifecycle");
  const observerLayer = context?.layers.find((layer) => layer.id === "observer-lifecycle");
  expect(publicLayer && observerLayer).toBeTruthy();
  Object.assign(publicLayer!.content, {
    bidHistory: [{seat: "north", action: "pass", round: 1}, {seat: "west", action: "sun", round: 1}],
    purchaseRecipient: "west",
    purchaseCard: plays[0].card,
    escalationHistory: [{seat: "south", action: "double", mode: "open"}],
  });
  Object.assign(observerLayer!.content, {initialFiveCards: initialHand.slice(0, 5)});
  setupBundle.timeline = {
    totalFrames: 42,
    segments: [
      {stage: "hook", sceneId: "SCENE-QUESTION", startFrame: 0, durationFrames: 1},
      {stage: "rewind", sceneId: "SCENE-QUESTION", startFrame: 1, durationFrames: 1},
      {stage: "setup", sceneId: "SCENE-CONTEXT", startFrame: 2, durationFrames: 4},
      {stage: "replay", sceneId: "SCENE-CONTEXT", startFrame: 6, durationFrames: 1},
      {stage: "question", sceneId: "SCENE-QUESTION", startFrame: 7, durationFrames: 1},
      {stage: "reveal", sceneId: "SCENE-REVEAL", startFrame: 8, durationFrames: 1},
      {stage: "continuation", sceneId: "SCENE-CONTINUATION", startFrame: 9, durationFrames: 32},
      {stage: "outcome", sceneId: "SCENE-CONTEXT", startFrame: 41, durationFrames: 1},
    ],
  };
  const markup = (frame: number) => renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: setupBundle,
    compositionFrame: frame,
    stage: "flagship",
    reducedMotion: true,
    storyFrame: frame,
    suppressAudio: true,
  }));
  for (const frame of [0, 2, 3]) {
    expect(markup(frame)).toContain('data-seat="south" data-hand-count="5"');
    expect(renderedCodes(markup(frame))).toEqual(initialHand.slice(0, 5).map(cardCode).sort());
  }
  expect(markup(2)).toContain('data-setup-action-index="0" data-setup-action-kind="bid" data-setup-actor="north"');
  expect(markup(3)).toContain('data-setup-action-index="1" data-setup-action-kind="bid" data-setup-actor="west"');
  expect(markup(4)).toContain('data-setup-action-index="2" data-setup-action-kind="purchase" data-setup-actor="west"');
  expect(markup(5)).toContain('data-setup-action-index="3" data-setup-action-kind="escalation" data-setup-actor="south"');
  for (const frame of [4, 5]) {
    expect(markup(frame)).toContain('data-seat="south" data-hand-count="8"');
    expect(renderedCodes(markup(frame))).toEqual(initialHand.map(cardCode).sort());
  }
  expect(markup(9)).toContain('data-completed-play-count="1"');
});

test("bid question and reveal keep the five-card pre-contract state beside a complete trace", () => {
  const bidBundle = structuredClone(bundle) as RenderBundle;
  const decisionHand = initialHand.slice(0, 5);
  const terminalProjectCards = plays
    .filter((play) => play.seat === "north" && play.card.suit === "diamonds")
    .slice(0, 5)
    .map((play) => play.card);
  const lifecycle = bidBundle.scenes
    .find((scene) => scene.stage === "continuation")!
    .layers.find((item) => item.id === "public-lifecycle")!;
  Object.assign(lifecycle.content, {
    projects: [{
      owner: "north",
      team: "north_south",
      kind: "hundred_sequence",
      value: 20,
      revealed: true,
      scoring: true,
      cards: terminalProjectCards,
    }],
    preDecisionMatchScore: [7, 9],
    matchScore: [7, 55],
    roundScore: {
      cardGamePoints: [0, 26],
      projectPoints: [20, 0],
      balootPoints: [0, 0],
      awardedPoints: [0, 46],
    },
  });
  Object.assign(bidBundle.presentation.layout, {
    show_projects: true,
    show_score_breakdown: true,
    persist_revealed_projects: true,
  });
  Object.assign(bidBundle.localizations, {
    "video.bid.challenge_question": "وش تشتري؟",
    "video.analysis.policy_conditioned_proof": "إثبات بحسب السياسات",
  });
  const question = bidBundle.scenes.find((scene) => scene.stage === "question")!;
  const questionTable = question.layers.find((item) => item.id === "table")!;
  const questionHand = question.layers.find((item) => item.id === "observer-hand")!;
  const questionSeats = question.layers.find((item) => item.id === "seats")!;
  const prompt = question.layers.find((item) => item.id === "decision-prompt")!;
  Object.assign(questionTable.content, {
    phase: "bidding",
    dealer: "east",
    buyer: null,
    contract: null,
    trumpSuit: null,
    matchScore: [7, 9],
    currentPlayer: observer,
    currentTrick: {plays: []},
    completedTricks: [],
    projects: [],
    roundScore: null,
  });
  Object.assign(questionHand.content, {cards: decisionHand});
  const context = bidBundle.scenes.find((scene) => scene.stage === "context")!;
  Object.assign(context.layers.find((item) => item.id === "public-lifecycle")!.content, {
    buyer: null, contract: null, preDecisionMatchScore: [7, 9], matchScore: [7, 9], projects: [],
  });
  Object.assign(context.layers.find((item) => item.id === "observer-lifecycle")!.content, {
    initialHandCards: decisionHand,
  });
  Object.assign(questionSeats.content, {
    handCounts: {north: 5, west: 5, south: 5, east: 5},
  });
  Object.assign(prompt.content, {
    textKey: "video.bid.challenge_question",
    actingSeat: observer,
    promptContract: "choose_acting_action",
    legalChoices: [
      {code: "pass", action: "pass", trumpSuit: null},
      {code: "sun", action: "sun", trumpSuit: null},
    ],
  });
  const reveal = bidBundle.scenes.find((scene) => scene.stage === "reveal")!;
  Object.assign(reveal.layers.find((item) => item.id === "analysis")!.content, {
    decisionKind: "bid",
    selectedAction: "pass",
    achievedMode: "policy_conditioned",
    completionStatus: "policy_complete",
    continuationComplete: true,
    claimSatisfied: true,
    winningRootActionCount: 1,
    reasonKey: "video.analysis.policy_conditioned_proof",
  });
  bidBundle.timeline = {
    totalFrames: 37,
    segments: [
      {stage: "hook", sceneId: "SCENE-QUESTION", startFrame: 0, durationFrames: 1},
      {stage: "rewind", sceneId: "SCENE-QUESTION", startFrame: 1, durationFrames: 1},
      {stage: "context", sceneId: "SCENE-CONTEXT", startFrame: 2, durationFrames: 1},
      {stage: "question", sceneId: "SCENE-QUESTION", startFrame: 3, durationFrames: 1},
      {stage: "reveal", sceneId: "SCENE-REVEAL", startFrame: 4, durationFrames: 1},
      {stage: "continuation", sceneId: "SCENE-CONTINUATION", startFrame: 5, durationFrames: 31},
      {stage: "outcome", sceneId: "SCENE-REVEAL", startFrame: 36, durationFrames: 1},
    ],
  };
  const markup = (frame: number) => renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: bidBundle,
    compositionFrame: frame,
    stage: "flagship",
    reducedMotion: true,
    suppressAudio: true,
  }));

  for (const frame of [0, 1, 2, 3, 4]) {
    const decisionMarkup = markup(frame);
    expect(renderedCodes(decisionMarkup)).toEqual(decisionHand.map(cardCode).sort());
    expect(decisionMarkup).toContain('data-source-play-count="0"');
    for (const seat of seats) {
      expect(decisionMarkup).toContain(`data-seat="${seat}" data-hand-count="5"`);
    }
    expect(decisionMarkup).not.toContain('class="tag buyer"');
    expect(decisionMarkup).not.toContain('class="call call-proj ');
    expect(decisionMarkup).not.toContain('class="ptray ');
    expect(decisionMarkup).not.toContain("project-badge");
    expect(decisionMarkup).not.toContain("outcome-card");
  }

  const continuation = markup(5);
  expect(continuation).toContain('data-source-play-count="32"');
  expect(continuation).toContain('class="tag buyer"');
  expect(continuation).toContain("صن");
  bidBundle.exportPreset = "decision_clip";
  const clipContinuation = markup(5);
  expect(renderedCodes(clipContinuation)).toEqual(decisionHand.map(cardCode).sort());
  expect(clipContinuation).toContain('data-source-play-count="0"');
  expect(clipContinuation).not.toContain('class="tag buyer"');
  expect(clipContinuation).not.toContain('class="ptray ');
  bidBundle.exportPreset = "hooked_flagship";
  const outcome = markup(36);
  expect(outcome).toContain('class="outcome-card ledger has-breakdown"');
  expect(outcome).toContain('<span>المجموع</span><b dir="ltr" class="total-us">7</b><b dir="ltr" class="total-them">55</b>');
});

test("V3 flagship reveals its project and transitions from pre-round to awarded score", () => {
  const scoredBundle = structuredClone(bundle) as RenderBundle;
  const lifecycle = scoredBundle.scenes
    .find((scene) => scene.stage === "continuation")!
    .layers.find((item) => item.id === "public-lifecycle")!;
  Object.assign(lifecycle.content, {
    pauseBeforePlayIndex: 24,
    projects: [{
      owner: "north",
      team: "north_south",
      kind: "hundred_sequence",
      value: 20,
      revealed: true,
      scoring: true,
      cards: plays
        .filter((play) => play.seat === "north" && play.card.suit === "diamonds")
        .slice(0, 5)
        .map((play) => play.card),
    }],
    preDecisionMatchScore: [4, 5],
    matchScore: [4, 51],
    roundScore: {
      cardGamePoints: [0, 26],
      projectPoints: [20, 0],
      balootPoints: [0, 0],
      awardedPoints: [0, 46],
    },
  });
  Object.assign(scoredBundle.presentation.layout, {
    show_projects: true,
    show_score_breakdown: true,
    project_announcement_trick: 1,
    project_reveal_trick: 1,
    persist_revealed_projects: true,
  });
  Object.assign(scoredBundle.scenes.find((scene) => scene.stage === "context")!
    .layers.find((item) => item.id === "public-lifecycle")!.content, {
    pauseBeforePlayIndex: 24, projects: lifecycle.content.projects,
    preDecisionMatchScore: [4, 5], matchScore: [4, 5],
  });
  Object.assign(scoredBundle.presentation.timeline, {
    project_call_seconds: 1,
    project_reveal_seconds: 1,
  });
  scoredBundle.render.fps = 30;

  scoredBundle.timeline = {
    totalFrames: 960,
    segments: [{stage: "replay", sceneId: "SCENE-CONTEXT", startFrame: 0, durationFrames: 960}],
  };
  const playbackMarkup = (frame: number) => renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: scoredBundle,
    compositionFrame: frame,
    stage: "flagship",
    reducedMotion: true,
    storyFrame: frame,
    suppressAudio: true,
  }));
  expect(playbackMarkup(0)).toMatch(/class="call call-proj [^"]*"[^>]*><small>مشروع<\/small><b>مية<\/b>/);
  expect(playbackMarkup(0)).toMatch(/class="ptray-label"><small>مشروع<\/small><b>مية<\/b>/);
  for (const code of ["D9", "D10", "DJ", "DQ", "DK"]) {
    expect(playbackMarkup(0)).toContain(`class="card card-small tray-card" data-small-card-code="${code}"`);
  }
  expect(playbackMarkup(29)).toContain('class="call call-proj ');
  expect(playbackMarkup(29)).toContain('class="ptray ');
  expect(playbackMarkup(30)).not.toContain('class="call call-proj ');
  expect(playbackMarkup(30)).not.toContain('class="ptray ');
  expect(playbackMarkup(30)).toContain('class="project-badge ');

  scoredBundle.timeline = {
    totalFrames: 2,
    segments: [
      {stage: "hook", sceneId: "SCENE-QUESTION", startFrame: 0, durationFrames: 1},
      {stage: "outcome", sceneId: "SCENE-REVEAL", startFrame: 1, durationFrames: 1},
    ],
  };
  const preRound = playbackMarkup(0);
  const outcome = playbackMarkup(1);
  expect(preRound).toContain('data-score-us="4"');
  expect(preRound).toContain('data-score-them="5"');
  expect(outcome).toContain('class="outcome-card ledger has-breakdown"');
  expect(outcome).toContain('class="lrow result-row project-points"><span>المشاريع</span><b dir="ltr" class="is-plus">+20</b><b dir="ltr">+0</b>');
  expect(outcome).toContain('class="lrow ltotal result-total"><span>المجموع</span><b dir="ltr" class="total-us">4</b><b dir="ltr" class="total-them">51</b>');
});

test("hooked flagship holds an interior decision before resuming the full trace", () => {
  const decisionBundle = structuredClone(bundle) as RenderBundle;
  const lifecycle = decisionBundle.scenes
    .find((scene) => scene.stage === "context")!
    .layers.find((item) => item.id === "public-lifecycle")!;
  lifecycle.content.pauseBeforePlayIndex = 24;
  decisionBundle.scenes.find((scene) => scene.stage === "continuation")!
    .layers.find((item) => item.id === "public-lifecycle")!.content.pauseBeforePlayIndex = 24;
  decisionBundle.timeline = {
    totalFrames: 36,
    segments: [
      {stage: "hook", sceneId: "SCENE-QUESTION", startFrame: 0, durationFrames: 1},
      {stage: "replay", sceneId: "SCENE-CONTEXT", startFrame: 1, durationFrames: 23},
      {stage: "context", sceneId: "SCENE-CONTEXT", startFrame: 24, durationFrames: 1},
      {stage: "question", sceneId: "SCENE-QUESTION", startFrame: 25, durationFrames: 1},
      {stage: "reveal", sceneId: "SCENE-REVEAL", startFrame: 26, durationFrames: 1},
      {stage: "continuation", sceneId: "SCENE-CONTINUATION", startFrame: 27, durationFrames: 8},
      {stage: "outcome", sceneId: "SCENE-REVEAL", startFrame: 35, durationFrames: 1},
    ],
  };
  const markup = (frame: number) => renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: decisionBundle,
    compositionFrame: frame,
    stage: "flagship",
    reducedMotion: true,
    suppressAudio: true,
  }));

  expect(markup(25)).toContain('data-completed-play-count="24"');
  expect(markup(26)).toContain('data-completed-play-count="24"');
  expect(markup(27)).toContain('data-completed-play-count="25"');
  expect(markup(34)).toContain('data-completed-play-count="32"');
});

test.each([
  {
    kind: "bid",
    questionKey: "video.bid.challenge_question",
    selected: "sun",
    labels: ["بس", "صن"],
    actions: [
      {code: "pass", action: "pass", trumpSuit: null},
      {code: "sun", action: "sun", trumpSuit: null},
    ],
  },
  {
    kind: "escalation",
    questionKey: "video.escalation.challenge_question",
    selected: "double:open",
    labels: ["بس", "دبل · مفتوح"],
    actions: [
      {code: "decline", action: "decline", mode: null},
      {code: "double:open", action: "double", mode: "open"},
    ],
  },
  {
    kind: "project",
    questionKey: "video.project.challenge_question",
    selected: "declare:hundred_sequence:100:14",
    labels: ["مية"],
    actions: [{
      code: "declare:hundred_sequence:100:14",
      action: "declare",
      owner: "north",
      kind: "hundred_sequence",
      value: 100,
      highRank: 14,
    }],
  },
])("renders typed $kind legal choices and selected action", ({kind, questionKey, selected, labels, actions}) => {
  const decisionBundle = structuredClone(bundle) as RenderBundle;
  expect(decisionBundle.source).toMatchObject({
    sourceSchemaVersion: 3,
    proofReference: qualifiedOrToolsSource.resultCode,
    publicProjectionHash: qualifiedOrToolsSource.publicProjectionHash,
  });
  Object.assign(decisionBundle.localizations, {
    [questionKey]: questionKey,
    "video.analysis.policy_conditioned_proof": "إثبات بحسب السياسات",
    "video.project.hundred_sequence": "مية",
  });
  const question = decisionBundle.scenes.find((scene) => scene.stage === "question");
  const reveal = decisionBundle.scenes.find((scene) => scene.stage === "reveal");
  const prompt = question?.layers.find((item) => item.id === "decision-prompt");
  const analysis = reveal?.layers.find((item) => item.id === "analysis");
  expect(prompt).toBeDefined();
  expect(analysis).toBeDefined();
  Object.assign(prompt!.content, {
    textKey: questionKey,
    actingSeat: "north",
    promptContract: "choose_acting_action",
    legalChoices: actions,
  });
  Object.assign(analysis!.content, {
    decisionKind: kind,
    selectedAction: selected,
    achievedMode: "policy_conditioned",
    completionStatus: "policy_complete",
    claimSatisfied: true,
    winningRootActionCount: 1,
    reasonKey: "video.analysis.policy_conditioned_proof",
  });

  const promptMarkup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: decisionBundle,
    compositionFrame: 0,
    stage: "hook",
    reducedMotion: true,
    suppressAudio: true,
  }));
  expect(promptMarkup).toContain('class="decision-choices"');
  expect(promptMarkup).toContain('aria-label="الخيارات القانونية"');
  for (const label of labels) expect(promptMarkup).toContain(label);

  const revealMarkup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: decisionBundle,
    compositionFrame: 0,
    stage: "reveal",
    reducedMotion: true,
    suppressAudio: true,
  }));
  expect(revealMarkup).toContain(`data-selected-action="${selected}"`);
  expect(revealMarkup).toContain("إثبات بحسب السياسات");

  Object.assign(decisionBundle.timeline, {
    totalFrames: 1,
    segments: [{stage: "outcome", sceneId: "SCENE-REVEAL", startFrame: 0, durationFrames: 1}],
  });
  const boundedMarkup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: decisionBundle,
    compositionFrame: 0,
    stage: "flagship",
    reducedMotion: true,
    storyFrame: 0,
    suppressAudio: true,
  }));
  expect(boundedMarkup).toContain("proof-boundary-only");
  expect(boundedMarkup).not.toContain("outcome-card");
});

test.each([["CJ", 28, "J♣"], ["HJ", 12, "J♥"]] as const)(
  "formats selected play %s using the shared card label", (code, id, label) => {
    const labelled = structuredClone(bundle) as RenderBundle;
    const analysis = labelled.scenes.find((scene) => scene.stage === "reveal")!.layers.find((layer) => layer.id === "analysis")!;
    Object.assign(analysis.content, {
      decisionKind: "play", selectedAction: code,
      recommendedAction: {card: plays.find((play) => play.card.id === id)!.card},
    });
    const markup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
      bundle: labelled, compositionFrame: 0, stage: "reveal", reducedMotion: true, suppressAudio: true,
    }));
    expect(markup).toContain(`data-selected-action="${code}"`);
    expect(markup).toContain(`class="winning-move-code" dir="auto" aria-label="${label}"><img class="card card-small pick-card" data-small-card-code="${code}"`);
    expect(markup).not.toContain(`class="winning-move-code" dir="auto" aria-label="${code}"`);
  },
);

test.each(["south", "west"] as const)("decision clip outcome and hold show terminal score for %s", (seat) => {
  const clip = structuredClone(bundle) as RenderBundle;
  clip.exportPreset = "decision_clip";
  clip.observerSeat = seat;
  for (const scene of clip.scenes) scene.observer = seat;
  const lifecycle = clip.scenes.find(scene => scene.stage === "continuation")!.layers.find(layer => layer.id === "public-lifecycle")!;
  lifecycle.content.matchScore = [8, 22];
  clip.timeline = {totalFrames: 60, segments: [
    {stage: "outcome", sceneId: "SCENE-CONTINUATION", startFrame: 0, durationFrames: 50},
    {stage: "hold", sceneId: "SCENE-CONTINUATION", startFrame: 50, durationFrames: 10},
  ]};
  for (const compositionFrame of [25, 55]) {
    const markup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
      bundle: clip, compositionFrame, stage: "flagship", reducedMotion: true, suppressAudio: true,
    }));
    expect(markup).toContain(`data-score-us="${seat === "south" ? 8 : 22}"`);
    expect(markup).toContain(`data-score-them="${seat === "south" ? 22 : 8}"`);
    expect(markup).toContain('class="outcome-card');
  }
});

test.each(["buyer_makes", "buyer_loses", "legacy_raw_units"])("labels Sun tie rule %s from the bound scene", (tieOutcome) => {
  const labelled = structuredClone(bundle) as RenderBundle;
  const analysis = labelled.scenes.find((scene) => scene.stage === "reveal")!.layers.find((layer) => layer.id === "analysis")!;
  analysis.content.sunTieOutcome = tieOutcome;
  const key = `video.scoring.sun_tie.${tieOutcome}`;
  labelled.localizations[key] = `Sun tie rule: ${tieOutcome}`;
  const markup = renderToStaticMarkup(createElement(ProfessionalPrototypeFrame, {
    bundle: labelled, compositionFrame: 0, stage: "reveal", reducedMotion: true, suppressAudio: true,
  }));
  expect(markup).toContain(`data-sun-tie-outcome="${tieOutcome}"`);
  expect(markup).toContain(labelled.localizations[key]);
});
