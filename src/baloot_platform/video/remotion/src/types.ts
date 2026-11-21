export type CardValue = {
  id: number;
  rank: string;
  suit: "spades" | "hearts" | "clubs" | "diamonds";
};

export type SceneLayer = {
  id: string;
  kind: string;
  disclosure: "public" | "observer_private" | "post_reveal_analysis";
  content: Record<string, unknown>;
};

export type SceneDocument = {
  schemaVersion: 1;
  sceneId: string;
  stage: string;
  observer: string;
  locale: "ar-SA";
  direction: "rtl";
  layers: SceneLayer[];
};

export type RenderSourceKind =
  | "research_result"
  | "research_scenario_result"
  | "analysis_candidate"
  | "finalist"
  | "laffa"
  | "research_audit_trace";

export type RenderQualificationStatus =
  | "qualified"
  | "provisional"
  | "rejected"
  | "quarantined"
  | "legacy_replay";

type RenderSourceGovernance = {
  kind: RenderSourceKind;
  code: string;
  decisionOrdinal?: number;
  decisionId?: string;
  auditTraceCode?: string;
  recordHash?: string;
  attemptStatus?: "running" | "completed" | "retry_scheduled" | "failed" | "reclaimed" | "cancelled";
  observerSeat: "north" | "west" | "south" | "east";
  traceHash?: string;
  analysisHash?: string;
  domainHash?: string | null;
  rulesHash?: string | null;
  originKind?: string;
  sourceStatus?: string;
  qualificationStatus: RenderQualificationStatus;
  auditOnly: boolean;
  renderPurpose: "content_candidate" | "research_audit";
  qualificationFailures: string[];
  publicationBlockers: string[];
  publicationEligible: boolean;
  publicationPolicy: Record<string, unknown>;
};

export type HistoricalRenderSourceV1 = RenderSourceGovernance & {
  sourceSchemaVersion?: never;
  resultHash?: never;
  auditHash?: never;
  analysisProfileHash?: never;
  analyzerId?: never;
  analyzerVersion?: never;
  exactStateId?: never;
  qrActionable?: never;
};

type CurrentRenderSourceV2Common = RenderSourceGovernance & {
  sourceSchemaVersion: 2;
  decisionOrdinal: number;
  decisionId: string;
  traceHash: string;
  rulesHash: string;
  exactStateId: string;
  qrActionable: boolean;
};

export type CurrentResearchResultRenderSourceV2 = CurrentRenderSourceV2Common & {
  kind: "research_result";
  resultHash: string;
  analysisHash: string;
  auditHash: string;
  analysisProfileHash: string;
  analyzerId: string;
  analyzerVersion: string;
};

export type CurrentResearchScenarioRenderSourceV2 = CurrentRenderSourceV2Common & {
  kind: "research_scenario_result";
  resultHash: string;
  scenarioHash: string;
  proofHash: string;
  constraintEvidenceHash: string;
  exactStateHash: string;
};

export type CurrentReplayRenderSourceV2 = CurrentRenderSourceV2Common & {
  kind: Exclude<RenderSourceKind, "research_result" | "research_scenario_result">;
  resultHash?: never;
  auditHash?: never;
  analysisProfileHash?: never;
  analyzerId?: never;
  analyzerVersion?: never;
};

export type CurrentRenderSourceV2 =
  | CurrentResearchResultRenderSourceV2
  | CurrentResearchScenarioRenderSourceV2
  | CurrentReplayRenderSourceV2;

export type ObserverSafeScenarioRenderSourceV3 = RenderSourceGovernance & {
  sourceSchemaVersion: 3;
  kind: "research_scenario_result";
  decisionOrdinal: 1;
  decisionReference: {resultCode: string; decisionOrdinal: 1};
  publicProjectionHash: string;
  qrActionable: boolean;
  rulesHash: string;
  proofReference: string;
  proofCompletionStatus: "exact_complete" | "bounded_complete" | "policy_complete" | "proof_incomplete" | "proof_failed";
  proofClaimSatisfied: boolean;
  proofAchievedMode: "exact" | "bounded" | "policy_conditioned" | "incomplete";
  decisionId?: never;
  exactStateId?: never;
  traceHash?: never;
  resultHash?: never;
  scenarioHash?: never;
  proofHash?: never;
  constraintEvidenceHash?: never;
  exactStateHash?: never;
};

export type RenderSource = HistoricalRenderSourceV1 | CurrentRenderSourceV2 | ObserverSafeScenarioRenderSourceV3;

export type RenderBundle = {
  schemaVersion: 1;
  bundleSha256: string;
  exportPreset: "hooked_flagship" | "decision_clip";
  locale: "ar-SA";
  direction: "rtl";
  observerSeat: "north" | "west" | "south" | "east";
  source?: RenderSource;
  render: {width: number; height: number; fps: number};
  presentation: Record<string, Record<string, unknown>> & {audio?: {enabled: boolean}};
  localizations: Record<string, string>;
  identity: {
    gameSerial: string;
    sceneId: string;
    enrollmentUri: string;
    players: Record<"north" | "west" | "east" | "south", string>;
  };
  audioCues: Array<{
    stage: string;
    assetId: string;
    path: string;
    volume: number;
    transcriptKey: string;
    transcriptSha256: string;
  }>;
  scenes: SceneDocument[];
  timeline: {
    totalFrames: number;
    segments: Array<{stage: string; sceneId: string; startFrame: number; durationFrames: number}>;
  };
};

export type PrototypeProps = {
  bundle: RenderBundle;
  stage: "normal" | "hook" | "reveal" | "flagship";
  reducedMotion?: boolean;
  storyFrame?: number;
  rewinding?: boolean;
  suppressAudio?: boolean;
  capturePlaybackDom?: boolean;
};

export type MultiAngleSegment = {
  ordinal: number;
  sourceName: string;
  observerSeat: "north" | "west" | "south" | "east";
  perspectiveName: string;
  sourceStartFrame: number;
  sourceEndFrame: number;
  transition: "cut" | "fade";
  transitionFrames: number;
};

export type MultiAngleProps = {
  plan: {
    sha256: string;
    fps: number;
    width: number;
    height: number;
    totalFrames: number;
    segments: MultiAngleSegment[];
  };
};
