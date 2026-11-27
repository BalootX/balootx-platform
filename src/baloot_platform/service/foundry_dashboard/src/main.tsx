import {CSSProperties, FormEvent, Fragment, ReactNode, SetStateAction, useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {
  createColumnHelper,
  createSortedRowModel,
  columnVisibilityFeature,
  rowSortingFeature,
  tableFeatures,
  useTable,
} from '@tanstack/react-table';
import './styles.css';
import {researchRunStatusLabel} from './shared/presentation';
import {configurationEditorArrayOptions, configurationEditorArrayItemSchema, configurationEditorChange, configurationEditorEntries, configurationEditorEnum, configurationEditorEnumTitle, configurationEditorEnumMessageKey, configurationEditorExpressionChoices, configurationEditorExpressionChoice, configurationEditorSelectExpression, configurationEditorObjectArrayAdd, configurationEditorObjectArrayRemove, configurationEditorObjectArraySchema, configurationEditorObjectValue, configurationEditorSchema, configurationPreviewComponents, openConfigurationRevision} from './features/configuration/model';
import {
  mergeResearchPatch,
  researchCatalogFieldApplicable,
  researchCatalogInitial,
  researchCatalogObjectInitial,
  researchCatalogReady,
  researchRelationshipActive,
  researchValueAtPath,
} from './features/research/catalog';
import type {
  ResearchCapabilities,
  ResearchConfigurationAuthority,
  ResearchConfigurationSelection,
  ResearchEstimate,
  ResearchPerformanceRollup,
  ResearchPreparedRender,
  ResearchRecipe,
  ResearchRecipeConfigurationCatalog,
  ResearchRenderPublicationPolicy,
  ResearchRenderSourceProvenance,
  ResearchResult,
  ResearchResultGroup,
  ResearchResultPage,
  ResearchRun,
  ResearchSequentialSearchReceiptPage,
  ResearchStats,
  ResearchTacticDraft,
  ResearchTacticKind,
  SemanticDisagreement,
} from './features/research/model';
import {
  ResearchDecisionAudit,
  ResearchDisagreementAudit,
  ResearchPlayProof,
  ResearchRenderPreview,
  ResearchRunDetails,
  researchFailureLabel,
} from './features/research/audit';
import {ResearchRecipeLedger} from './features/research/recipeLedger';
import {ResearchRecipeController} from './features/research/recipes';
import {researchRecipeErrorMessage} from './features/research/errorCopy';
import {researchRecipeApi} from './api/research';
import {ResearchPerformanceSummary, ResearchStudio} from './features/research/workspace';
import {copyJson, isSeatName, type JsonObject, type JsonValue, type SeatName} from './shared/json';
import {date, number, hasFoundryCopy, t, tCatalog, tVariant, useResearchWorkflowCopyRevision} from './shared/localization';
import {
  renderJobReferencesResearch,
  renderJobReferencesSource,
} from './features/videos/model';

type Campaign = {
  campaignHash: string;
  spec: JsonObject;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  cancelRequested: boolean;
  shardCounts: Record<string, number>;
  createdAt: string;
  updatedAt: string;
};
type Shard = {specHash: string; shardIndex: number; status: string; attempts: number; result?: JsonObject | null};
type Attempt = {specHash: string; shardIndex: number; workerId: string; attempt: number; terminalStatus?: string | null};
type FoundryProfile = {
  runtimeBundleRevisionId: string;
  runtimeBundleVersion: string;
  rulesHash: string;
  bundleHash: string;
  engineVersion: string;
  analysisProfileHash: string;
  perspectiveProfileHash: string;
  selectionProfileHash: string;
  agentRelease: string;
  seatAgents: Record<string, string>;
  staticLaneProfileHashes: Record<string, string>;
  laneAgentReleases: Record<string, string>;
  laneSeatAgents: Record<string, Record<string, string>>;
  organicScreeningProfile: JsonObject;
  guidedSearchProfile: JsonObject;
  finalistAgentQualification: {
    profileId: string;
    configuredEqual: boolean;
    zeroMistakeSearchReady: boolean;
    benchmarkEvidenceComplete: boolean;
    professionalReviewComplete: boolean;
    expertGenerationAllowed: boolean;
    status: string;
    issues: string[];
    sha256: string;
  };
  finalistAgentPilotEvidence: {
    status: string;
    passed: boolean;
    candidatePolicy: string;
    positions: number;
    policyAccuracy: number | null;
    expectedRegret: number | null;
    illegalActionRate: number | null;
    latencyP95Ms: number | null;
    qualificationEligible: false;
    issues: string[];
    sha256: string;
  };
  campaignProofProfile: {minimumOrganicRounds?: number; requiredWorkerCounts?: number[]};
  sakkaGenerationProfile: {sha256: string; target: number; maximumRounds: number};
  videoProductionBatchProfile: {
    sha256: string;
    defaultTarget: number;
    maximumTarget: number;
    laneOrder: string[];
    waveSeedCount: number;
    maximumWaves: number;
    seedsPerShard: number;
    simulationWorkerCount: number;
    finalistConcurrency?: number;
    renderConcurrency: number;
    renderPolicy: 'on_demand' | 'automatic';
  };
};
type CatalogRecord = {catalogHash: string; kind: string; metadata: JsonObject; createdAt: string};
type FoundrySakka = CatalogRecord & {metadata: JsonObject & {provenance?: string; roundCount?: number; winner?: string; finalScore?: number[]; signatureKeyId?: string}};
type FoundrySakkaRound = {
  ordinal: number;
  traceHash: string;
  startingScore: number[];
  awardedPoints: number[];
  finalScore: number[];
  dealer: string;
  roundNumber: number;
  allPassCount?: number;
};
type FoundrySakkaDetail = {
  record: FoundrySakka;
  document: JsonObject & {
    sakka?: JsonObject & {
      target?: number;
      provenance?: string;
      winner?: string;
      startingScore?: number[];
      finalScore?: number[];
      rounds?: FoundrySakkaRound[];
      leadChanges?: number;
    };
  };
};
type SakkaGenerationJob = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  result?: {sakkaHash: string; generationHash: string; roundCount: number; finalScore: number[]; winner: string} | null;
  error?: string | null;
};
type AnalysisJob = {jobId: string; status: string; progress?: number; request?: {campaignHash?: string}; result?: JsonObject | null; error?: string | null; createdAt?: string};
type FinalistValidationJob = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  request?: {analysisJobId?: string; candidateHashes?: string[]};
  result?: (JsonObject & {localPreviewReady?: boolean; publicationReady?: boolean; items?: JsonObject[]}) | null;
  error?: string | null;
  createdAt?: string;
};
type AnalysisCandidate = {
  candidateHash: string;
  shortCode: string;
  observerSeat: string;
  decisionKind: string;
  truthLabel: string;
  confidence: number;
  selectionScore: number;
  dimensions: Record<string, string>;
  bestAction: JsonObject;
  actualAction: JsonObject;
  alternativeCount: number;
  laffaClassification: string;
  critical: boolean;
  playReady: boolean;
  finalistValidationRequired: boolean;
  compatibilityBlockers?: string[];
  technical: JsonObject;
};
type CandidatePage = {schemaVersion: 1; jobId: string; campaignHash: string; count: number; items: AnalysisCandidate[]};
type PreparedCandidate = {
  schemaVersion: 1;
  exportPreset: 'decision_clip' | 'hooked_flagship';
  candidate: AnalysisCandidate;
  locator: JsonObject & {code?: string; universalLink?: string};
  bundleArtifact: JsonObject;
  basename: string;
};
type RenderJob = {jobId: string; status: string; progress?: number; request?: JsonObject; result?: JsonObject | null; error?: string | null; createdAt?: string};
type ArchivedLaffa = {
  gameHash: string;
  serial: string;
  gameId: string;
  bundleHash: string;
  status: string;
  label?: string | null;
  createdAt: string;
  contract?: string | null;
  buyer?: string | null;
  roundNumber?: number | null;
  matchScore?: number[] | null;
};
type LaffaSourceReference = {
  kind: 'archived_game' | 'sakka_round' | 'research_result' | 'qualified_laffa' | 'research_audit_trace';
  code: string;
  identity: string;
  uri: string;
};
type UnifiedLaffa = {
  humanCode: string;
  sourceKind: LaffaSourceReference['kind'];
  sourceKinds: LaffaSourceReference['kind'][];
  replayValid: boolean;
  auditOnly: boolean;
  publicationEligible: boolean;
  qualificationFailures: string[];
  qualificationStatus: string;
  truthLabel?: string | null;
  confidence?: number | null;
  challengeScore?: number | null;
  tacticalTags: string[];
  contract?: string | null;
  project?: string | null;
  projects: string[];
  buyer?: string | null;
  dealer?: string | null;
  seat?: string | null;
  interestingSeats: string[];
  relatedSakkas: {code:string; identity:string; uri:string}[];
  renderCount: number;
  renderStatus: 'not_prepared' | 'prepared' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  createdAt: string;
  traceHash?: string | null;
  gameHash?: string | null;
  sourceLocator: LaffaSourceReference;
  sourceReferences: LaffaSourceReference[];
};
type UnifiedLaffaPage = {
  schemaVersion: 1;
  resourceKind: 'unified_laffa_library_v1';
  items: UnifiedLaffa[];
  total: number;
  offset: number;
  limit: number;
  nextOffset: number | null;
  sourceFacets: {sourceKind:LaffaSourceReference['kind']; count:number}[];
};
type UnifiedLaffaFilters = {
  search: string;
  sourceKind: string;
  qualificationStatus: string;
  truthLabel: string;
  contract: string;
  project: string;
  tactic: string;
  seat: string;
  renderStatus: string;
  sort: string;
  offset: number;
};
type BatchPreparedItem = {
  traceHash: string;
  candidateHash: string;
  shortCode: string;
  basename: string;
  bundleArtifact: JsonObject;
  bundleHash: string;
  selectionScore?: number;
  laffaSelectionScore?: JsonObject & {score?: number; passed?: boolean};
  confidence?: number;
  truthLabel?: string;
  decisionKind?: string;
  laffaClassification?: string;
  qualification?: JsonObject;
  observerSeat?: string;
  dimensions?: Record<string, string>;
  bestAction?: JsonObject;
  sourceCampaignHash?: string;
  analysisJobId?: string;
  finalistJobId?: string;
  interestingProfiles?: string[];
  scoreProfiles?: JsonObject;
};
const laffaTableFeatures = tableFeatures({
  columnVisibilityFeature,
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
});
const laffaColumnHelper = createColumnHelper<typeof laffaTableFeatures, BatchPreparedItem>();
const sakkaTableFeatures = tableFeatures({
  columnVisibilityFeature,
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
});
const sakkaColumnHelper = createColumnHelper<typeof sakkaTableFeatures, FoundrySakka>();
type VideoProductionBatch = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'exhausted';
  progress: number;
  request: JsonObject & {targetVideos?: number; maximumRounds?: number; renderPolicy?: string};
  result?: (JsonObject & {
    stage?: string;
    outcome?: string;
    targetVideos?: number;
    roundsGenerated?: number;
    retainedCandidates?: number;
    finalistsAccepted?: number;
    finalistsRejected?: number;
    preparedVideoCount?: number;
    renderReadyCount?: number;
    completedVideoCount?: number;
    failedVideoCount?: number;
    campaigns?: string[];
    analysisJobs?: string[];
    finalistJobs?: string[];
    renderJobs?: string[];
    rejections?: JsonObject[];
    validationItems?: JsonObject[];
    preparedItems?: BatchPreparedItem[];
    renderProgress?: number;
    stageProgress?: number;
    activeWorkerCount?: number;
    currentWave?: number;
  }) | null;
  error?: string | null;
  createdAt?: string;
};
type ConfigProfile = {
  revisionId: string;
  family: string;
  profileId: string;
  version: string;
  scope: string;
  scopeKey: string;
  status: string;
  payloadSha256: string;
  sequence: number;
  etag: number;
  updatedAt: string;
  payload?: JsonObject;
};
type JsonSchema = JsonObject & {
  type?: string | string[];
  title?: string;
  description?: string;
  enum?: JsonValue[];
  const?: JsonValue;
  minimum?: number;
  maximum?: number;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  $ref?: string;
};
type ProfileValidation = {
  revisionId: string;
  valid: boolean;
  etag: number;
  payloadSha256: string;
  automaticProductionApply: boolean;
};
type BundleCompilation = {
  bundle: ConfigBundle;
  domainSpecHash?: string;
  compilationHash?: string;
  coverage: CompilerCoverageSummary;
  publishable: boolean;
  compilerVersion?: string;
  automaticProductionApply: boolean;
  previewOnly: boolean;
};
type ConfigBundle = {
  revisionId: string;
  bundleId: string;
  version: string;
  status: string;
  mode?: string;
  componentRevisions: Record<string, string>;
  contentSha256: string;
  sequence?: number;
  etag: number;
};
type CompilerCoverageRow = {
  primitiveId: string;
  definitionKind: string;
  category: string;
  requiredTargets: string[];
  coveredTargets: string[];
  missingTargets: string[];
  coverageStatus: string;
  executableBindingStatus: string;
  behavioralOrInvariantStatus: string;
  qualificationEligible: boolean;
  verifiedBehavioralEvidence?: string | null;
};
type CompilerCoverageSummary = {
  covered: number;
  total: number;
  percentage: number;
  complete: boolean;
  executableBindingCovered: number;
  executableBindingTotal: number;
  executableBindingPercentage: number;
  executableBindingComplete: boolean;
  behavioralOrInvariantCovered: number;
  behavioralOrInvariantTotal: number;
  behavioralOrInvariantPercentage: number;
  behavioralOrInvariantComplete: boolean;
  rows: CompilerCoverageRow[];
  sha256?: string;
};
type DomainCapabilitiesResponse = {
  compilerV3?: {compiledDomain?: {coverage?: CompilerCoverageSummary}};
};
type DrawerState = {kind: string; title: string; subtitle?: string; content: ReactNode; audit?: JsonObject} | null;

function assertPreparedObserver(prepared: ResearchPreparedRender, requested: SeatName): ResearchPreparedRender {
  const source=prepared.source;
  if (
    prepared.observerSeat!==requested
    || source.observerSeat!==requested
    || source.publicationEligible!==prepared.publicationEligible
    || source.auditOnly!==!prepared.publicationEligible
    || source.renderPurpose!==prepared.renderPurpose
    || source.qualificationStatus!==prepared.qualificationStatus
    || !Array.isArray(source.qualificationFailures)
    || !Array.isArray(source.publicationBlockers)
    || !source.publicationPolicy
  ) {
    throw new Error(t("workflow.research_render_response_mismatch"));
  }
  return prepared;
}
const statusLabel: Record<string, string> = {
  queued: t("domain_labels.m_279963953fb4"), running: t("domain_labels.m_d761119224ab"), completed: t("domain_labels.m_c2da5684d63b"), failed: t("domain_labels.m_d506fdf79c68"), cancelled: t("domain_labels.m_616d302cb016"),
  exhausted: t("workflow.batch_status_exhausted"),
  published: t("domain_labels.m_74f0d5710a99"), draft: t("domain_labels.m_552aec56f591"), superseded: t("domain_labels.m_b701c7b36d61"), approved: t("domain_labels.m_ef0a0d80aedd"), review: t("domain_labels.m_8aac5fac1498"),
};
const videoBatchStageLabel: Record<string, string> = {
  searching: t("workflow.batch_stage_searching"),
  analyzing: t("workflow.batch_stage_analyzing"),
  qualifying: t("workflow.batch_stage_qualifying"),
  candidate_resimulation: t("workflow.batch_stage_candidate_resimulation"),
  rendering: t("workflow.batch_stage_rendering"),
  draining: t("workflow.batch_stage_draining"),
  ready_for_selection: t("workflow.batch_stage_ready_for_selection"),
  completed: t("workflow.batch_stage_completed"),
  exhausted: t("workflow.batch_status_exhausted"),
  cancelled: t("workflow.batch_stage_cancelled"),
};
const videoBatchRejectionStageLabel: Record<string, string> = {
  screening: t("workflow.batch_rejection_stage_screening"),
  quality_filter: t("workflow.batch_rejection_stage_quality_filter"),
  trace_deduplication: t("workflow.batch_rejection_stage_trace_deduplication"),
  candidate_resimulation: t("workflow.batch_rejection_stage_candidate_resimulation"),
  finalist: t("workflow.batch_rejection_stage_finalist"),
};
const videoBatchRejectionReasonLabel: Record<string, string> = {
  no_retained_trace: t("workflow.batch_rejection_no_retained_trace"),
  duplicate_laffa: t("workflow.batch_rejection_duplicate_laffa"),
  batch_gate_filtered: t("workflow.batch_rejection_batch_gate_filtered"),
  not_play_decision: t("workflow.batch_rejection_not_play_decision"),
  not_critical: t("workflow.batch_rejection_not_critical"),
  disallowed_truth_label: t("workflow.batch_rejection_disallowed_truth_label"),
  below_selection_score: t("workflow.batch_rejection_below_selection_score"),
  below_confidence: t("workflow.batch_rejection_below_confidence"),
  compatibility_blocker: t("workflow.batch_rejection_compatibility_blocker"),
  no_playable_finalist: t("workflow.batch_rejection_no_playable_finalist"),
  no_verified_tactical_play_opportunity: t("workflow.batch_rejection_no_verified_tactical_play_opportunity"),
  unresolved_surrounding_decisions: t("workflow.batch_rejection_unresolved_surrounding_decisions"),
  finalist_quality_failed: t("workflow.batch_rejection_finalist_quality_failed"),
  candidate_resimulation_failed: t("workflow.batch_rejection_candidate_resimulation_failed"),
};

function videoBatchRejectionText(value: JsonObject): string {
  const dispositions = Array.isArray(value.dispositions) ? value.dispositions : [];
  const details = dispositions.flatMap((entry) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return [];
    const code = String(entry.code || '');
    const label = videoBatchRejectionReasonLabel[code];
    if (!label) return [];
    const count = Number(entry.count || 0);
    return [`${label} (${number.format(count)})`];
  });
  if (details.length) return details.join(' • ');
  const label = videoBatchRejectionReasonLabel[String(value.reasonCode || '')]
    || t("workflow.batch_rejection_generic");
  const count = Number(value.count || 0);
  return count > 0 ? `${label} (${number.format(count)})` : label;
}

function aggregateVideoBatchRejections(values: JsonObject[]): JsonObject[] {
  const grouped = new Map<string, JsonObject>();
  const add = (stage: string, reasonCode: string, count: number) => {
    const normalizedStage = stage || 'unknown';
    const normalizedReason = reasonCode || 'unknown';
    const normalizedCount = Number.isFinite(count) && count > 0 ? count : 1;
    const key = `${normalizedStage}\u0000${normalizedReason}`;
    const current = grouped.get(key);
    grouped.set(key, {
      stage: normalizedStage,
      reasonCode: normalizedReason,
      count: Number(current?.count || 0) + normalizedCount,
    });
  };
  values.forEach((value) => {
    const stage = String(value.stage || '');
    const dispositions = Array.isArray(value.dispositions) ? value.dispositions : [];
    const validDispositions = dispositions.filter(
      (entry): entry is JsonObject => Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry),
    );
    if (validDispositions.length > 0) {
      validDispositions.forEach((entry) => add(stage, String(entry.code || ''), Number(entry.count || 0)));
      return;
    }
    add(stage, String(value.reasonCode || ''), Number(value.count || 0));
  });
  return Array.from(grouped.values()).sort((left, right) => (
    Number(right.count || 0) - Number(left.count || 0)
    || String(left.stage || '').localeCompare(String(right.stage || ''))
    || String(left.reasonCode || '').localeCompare(String(right.reasonCode || ''))
  ));
}
const seatLabel: Record<string, string> = {north: t("domain_labels.m_5cbbfccea6e8"), east: t("domain_labels.m_26b73a5271c1"), south: t("domain_labels.m_390e936c0d6c"), west: t("domain_labels.m_64141a180741")};
const decisionLabel: Record<string, string> = {play: t("domain_labels.m_e974843e20f0"), bid: t("domain_labels.m_d293ea4d0fa1"), project: t("domain_labels.m_4a2b9a3f34b5"), pass: t("domain_labels.m_d9393a6726e5"), escalation: t("domain_labels.m_a451fdf03770")};
const truthLabel: Record<string, string> = {
  exact_optimal: t("domain_labels.m_2e2b14a6747d"),
  robust_best_known: t("domain_labels.m_ec905ea7bec8"),
  debatable: t("domain_labels.m_d7b4e4e4748b"),
  provisional: t("workflow.research_truth_provisional"),
  rejected: t("workflow.research_status_rejected"),
};
const disagreementLabel: Record<string, string> = {
  compiler_parity_defect: t("workflow.research_quarantine_compiler_parity"),
  runtime_transition_defect: t("workflow.research_quarantine_runtime_transition"),
  hard_constraint_violation: t("workflow.research_quarantine_hard_constraint"),
  observer_information_defect: t("workflow.research_quarantine_observer_information"),
  historical_record_requires_readjudication: t("workflow.research_quarantine_historical"),
  constraint_mismatch: t("workflow.research_quarantine_constraint_mismatch"),
  legality_mismatch: t("workflow.research_quarantine_legality_mismatch"),
  score_mismatch: t("workflow.research_quarantine_score_mismatch"),
  optimality_mismatch: t("workflow.research_quarantine_optimality_mismatch"),
  provenance_mismatch: t("workflow.research_quarantine_provenance_mismatch"),
};
const laffaLabel: Record<string, string> = {
  single_perspective_masterpiece: t("domain_labels.m_861b54ad9bc3"),
  two_seat_duel: t("domain_labels.m_5384115427c2"),
  team_coordination: t("domain_labels.m_c13802606042"),
  multi_angle_laffa: t("domain_labels.m_53fcca186659"),
  four_seat_pressure: t("domain_labels.m_e1005716e4cd"),
  critical_sakka_laffa: t("domain_labels.m_2b9352f50e94"),
};
const candidateDimensionLabel: Record<string, string> = {
  tactical_detector: t("configuration_families.m_3dd93576db27"), contract: t("configuration_families.m_f056f488e845"), project: t("configuration_families.m_7eb84d94b7ef"), trick_number: t("configuration_families.m_0df156950fca"),
  buyer_relationship: t("configuration_families.m_30ac586bdd18"), difficulty_band: t("configuration_families.m_89210938dc1d"), perspective_breadth: t("configuration_families.m_2101408c36d0"), source: t("configuration_families.m_312f1558c460"),
};
const tacticalLabel: Record<string, string> = {
  only_winning_move: t("configuration_families.m_f5f748b64995"), contract_rescue: t("configuration_families.m_cb6b14bc384c"), small_card_big_gain: t("configuration_families.m_297b27eed797"),
  large_card_sacrifice: t("configuration_families.m_f3925eb01881"), kabout_creation: t("configuration_families.m_ee9e5bd6de43"), kabout_prevention: t("configuration_families.m_c54b9f18b4c7"),
  entry_unblocking: t("configuration_families.m_76b97d33686d"), trump_control: t("configuration_families.m_1bc334063842"), partner_enablement: t("configuration_families.m_830dce9a3190"),
};
const contractLabel: Record<string, string> = {hokum:t("configuration_families.m_9d151dc785f8"), sun:t("configuration_families.m_e2aa47421204"), ashkal:t("configuration_families.m_68ba4e657a64")};
const suitName: Record<string, string> = {spades:t("configuration_fields.label_suit_spades"),hearts:t("configuration_fields.label_suit_hearts"),diamonds:t("configuration_fields.label_suit_diamonds"),clubs:t("configuration_fields.label_suit_clubs")};
const suitSymbol: Record<string, string> = {spades:'♠', hearts:'♥', diamonds:'♦', clubs:'♣'};
const projectLabel: Record<string, string> = {none:t("configuration_families.m_5f813ffa9ddc"), siri:t("configuration_families.m_0a16ea3020b1"), fifty:t("configuration_families.m_0e03c52f7552"), hundred:t("configuration_families.m_651541d79bc9"), hundred_sequence:t("configuration_families.m_651541d79bc9"), hundred_four_kind:t("configuration_families.m_651541d79bc9"), four_hundred:t("configuration_families.m_0a3f542c78c4")};
const relationshipLabel: Record<string, string> = {self:t("configuration_families.m_1ea299cb1c11"), buyer:t("configuration_families.m_1ea299cb1c11"), partner:t("configuration_families.m_ef1ea4e74f62"), opponent:t("configuration_families.m_3cd743c301fa"), not_applicable:t("configuration_families.m_5934d1123b02")};
const difficultyLabel: Record<string, string> = {elite:t("configuration_families.m_41bc866b7b5c"), expert:t("configuration_families.m_bbb098674cde"), hard:t("configuration_families.m_8954120fb333"), medium:t("configuration_families.m_42a5dadf6e45"), easy:t("configuration_families.m_073ebbcee957")};
const sourceLabel: Record<string, string> = {organic:t("configuration_families.m_7835af119dd9"), guided:t("configuration_families.m_e3fb0090b9ff"), disagreement:t("configuration_families.m_6433a6a57447"), historical:t("configuration_families.m_d99796dd37e2"), expert:t("configuration_families.m_38c3a625e037")};
const laffaSourceLabel: Record<string, string> = {
  laffa:t("workflow.laffa_library_source_archive"),
  archived_game:t("workflow.laffa_library_source_archive"),
  sakka_round:t("workflow.laffa_library_source_sakka_round"),
  research_result:t("workflow.laffa_library_source_research"),
  qualified_laffa:t("workflow.laffa_library_source_qualified"),
  research_audit_trace:t("workflow.laffa_library_source_research_audit"),
};
const laffaQualificationLabel: Record<string, string> = {
  qualified:t("workflow.laffa_library_qualification_qualified"),
  rejected:t("workflow.laffa_library_qualification_rejected"),
  provisional:t("workflow.laffa_library_qualification_provisional"),
  unqualified:t("workflow.laffa_library_qualification_unqualified"),
  quarantined:t("workflow.research_quarantine_title"),
  legacy_replay:t("workflow.laffa_library_qualification_audit"),
};
const laffaRenderLabel: Record<string, string> = {
  not_prepared:t("workflow.laffa_library_video_not_generated"),
  prepared:t("workflow.laffa_library_render_prepared"),
  queued:t("workflow.laffa_library_render_queued"),
  running:t("workflow.laffa_library_render_running"),
  completed:t("workflow.laffa_library_video_ready"),
  failed:t("workflow.laffa_library_render_failed"),
  cancelled:t("workflow.laffa_library_render_cancelled"),
};
const candidateDimensionOrder = ['tactical_detector','contract','project','trick_number','buyer_relationship','difficulty_band','perspective_breadth','source'];

function candidateDimensionValue(key: string, value: string) {
  if (key === 'tactical_detector') return value.split('+').map((item) => tacticalLabel[item] || item).join(' + ');
  if (key === 'contract') return contractLabel[value] || value;
  if (key === 'project') return projectLabel[value] || value;
  if (key === 'trick_number') return (t("configuration_families.m_cff00c06be60") + number.format(Number(value)) + "");
  if (key === 'buyer_relationship') return relationshipLabel[value] || value;
  if (key === 'difficulty_band') return difficultyLabel[value] || value;
  if (key === 'perspective_breadth') return ("" + number.format(Number(value)) + t("configuration_families.m_c8b51beda995"));
  if (key === 'source') return sourceLabel[value] || value;
  if (key === 'acting_seat') return seatLabel[value] || value;
  return value;
}
const familyMeta: Record<string, {label: string; description: string}> = {
  rules: {label: t("configuration_families.m_9df238db73a6"), description: t("configuration_families.m_e626ae33a44e")},
  agents: {label: t("configuration_families.m_34018f2ff7af"), description: t("configuration_families.m_544d25195dc4")},
  analysis: {label: t("configuration_families.m_561ad79c7de4"), description: t("configuration_families.m_411d21d48fc2")},
  challenge: {label: t("configuration_families.m_3fc01d823013"), description: t("configuration_families.m_f44dbe1bebf3")},
  challenge_contracts: {label: t("configuration_families.m_403959dd9ea1"), description: t("configuration_families.m_c0910af7e1ee")},
  video: {label: t("configuration_families.m_8a4cbedde18f"), description: t("configuration_families.m_433fbeb40e2a")},
  conventions: {label: t("configuration_tokens.m_d4892ff648ab"), description: t("configuration_tokens.m_15041d2d4627")},
  difficulty: {label: t("configuration_tokens.m_53d2c6d3ac20"), description: t("configuration_tokens.m_9797d73d199c")},
  performance: {label: t("configuration_tokens.m_1264c71a3c12"), description: t("configuration_tokens.m_828e51c24129")},
  security: {label: t("configuration_tokens.m_976adf68f49a"), description: t("configuration_tokens.m_0f146e819801")},
  runtime: {label: t("configuration_tokens.m_d1e21f47730f"), description: t("configuration_tokens.m_dfafdc2f6829")},
  storage: {label: t("configuration_tokens.m_f08625064ac0"), description: t("configuration_tokens.m_9aacc00f0357")},
  product: {label: t("configuration_tokens.m_f35ea8df438e"), description: t("configuration_tokens.m_14ccae7b3065")},
  strategy: {label: t("configuration_tokens.m_418e914a3153"), description: t("configuration_tokens.m_e6ad899ac92f")},
  belief: {label: t("configuration_tokens.m_e5c7c81df095"), description: t("configuration_tokens.m_2f5806562eba")},
  match_value: {label: t("configuration_tokens.m_10a21c82ab8e"), description: t("configuration_tokens.m_a5256e5cd53a")},
  hitl: {label: t("configuration_tokens.m_7b1173d9f446"), description: t("configuration_tokens.m_3bd74f58fcab")},
  audit: {label: t("configuration_tokens.m_3a82b23a1d3f"), description: t("configuration_tokens.m_935e84461e3c")},
  benchmark: {label: t("configuration_tokens.m_bfb34daee02d"), description: t("configuration_tokens.m_b309ae74c3bf")},
  certification: {label: t("configuration_tokens.m_774dd7ff22d8"), description: t("configuration_tokens.m_dd8a3703c2f4")},
  configuration: {label: t("configuration_tokens.m_62f57c70a7d1"), description: t("configuration_tokens.m_43ff63cda816")},
  database: {label: t("configuration_tokens.m_430e2c93e7bb"), description: t("configuration_tokens.m_60c55a97a166")},
  duplicate: {label: t("configuration_tokens.m_b4b8bbea25ab"), description: t("configuration_tokens.m_b27cf47bcc20")},
  game: {label: t("configuration_tokens.m_cc63a0ba000c"), description: t("configuration_tokens.m_2b45f6eb81f7")},
  history: {label: t("configuration_tokens.m_9dca2d96d1fb"), description: t("configuration_tokens.m_aabe13c3b665")},
  identity: {label: t("configuration_tokens.m_c447ac426926"), description: t("configuration_tokens.m_5050b4ac85b9")},
  learning: {label: t("configuration_tokens.m_4d33dfc911a6"), description: t("configuration_tokens.m_e62c55485f8a")},
  live: {label: t("configuration_tokens.m_94ebc0bc9e48"), description: t("configuration_tokens.m_172f9b73c29d")},
  ml: {label: t("configuration_tokens.m_3775f0ba89f3"), description: t("configuration_tokens.m_d0d43450dd9c")},
  operations: {label: t("configuration_tokens.m_c7a49bb5acfd"), description: t("configuration_tokens.m_3a3ccd19dca7")},
  shared: {label: t("configuration_tokens.m_80635788cc18"), description: t("configuration_tokens.m_3866b0548573")},
  adjudication: {label: t("configuration_tokens.m_31cc71b429fe"), description: t("configuration_tokens.m_2ccd11c54f4f")},
};

const tokenAr: Record<string, string> = {
  enabled:t("configuration_tokens.m_288f135c14bf"), disabled:t("configuration_tokens.m_532e66a3a862"), maximum:t("configuration_tokens.m_9651056d93a0"), minimum:t("configuration_tokens.m_a3e1e1424079"), max:t("configuration_tokens.m_c0924d94939b"), min:t("configuration_tokens.m_8b106d5592f2"),
  threshold:t("configuration_tokens.m_393ed2496726"), thresholds:t("configuration_tokens.m_cd60f65730ab"), limit:t("configuration_tokens.m_393ed2496726"), limits:t("configuration_tokens.m_cd60f65730ab"), count:t("configuration_tokens.m_92085a8f598e"), size:t("configuration_tokens.m_3ef1959dc111"), value:t("configuration_tokens.m_4c49efecd6cb"), values:t("configuration_tokens.m_b45cda764f55"),
  score:t("configuration_tokens.m_4cf48886fd0e"), points:t("configuration_tokens.m_4cf48886fd0e"), cards:t("configuration_tokens.m_f4d7a687eb09"), card:t("configuration_tokens.m_5feff2ee1d5c"), player:t("configuration_tokens.m_f61653ec4dac"), players:t("configuration_tokens.m_45573bc6e230"), team:t("configuration_tokens.m_274824e47dcc"), teams:t("configuration_tokens.m_0b5254487af9"),
  rules:t("configuration_tokens.m_0a24ebd440e5"), rule:t("configuration_tokens.m_2e6f1696e64a"), bidding:t("configuration_tokens.m_53f10e278252"), bid:t("configuration_tokens.m_53f10e278252"), play:t("configuration_tokens.m_ab9eab84ac89"), projects:t("configuration_tokens.m_5818066ebd9f"), project:t("configuration_tokens.m_7eb84d94b7ef"),
  dealing:t("configuration_tokens.m_d5315a16d31d"), dealer:t("configuration_tokens.m_4611365b72b2"), buyer:t("configuration_tokens.m_d90ebac1eb3a"), trump:t("configuration_tokens.m_d7deb71a8ab9"), sun:t("configuration_tokens.m_e2aa47421204"), hokum:t("configuration_tokens.m_9d151dc785f8"), ashkal:t("configuration_tokens.m_68ba4e657a64"), contract:t("configuration_tokens.m_f056f488e845"), contracts:t("configuration_tokens.m_9adfda86daf5"),
  first:t("configuration_tokens.m_b33da2a58a4b"), second:t("configuration_tokens.m_333c4739a8a0"), round:t("configuration_tokens.m_bd463d4c0cf6"), rounds:t("configuration_tokens.m_421e2b93641c"), trick:t("configuration_tokens.m_70b8c56a0139"), tricks:t("configuration_tokens.m_21b3629217f7"), sequence:t("configuration_tokens.m_92eab61475bb"),
  allowed:t("configuration_tokens.m_f2c0bb00ee4d"), require:t("configuration_tokens.m_11d2121e82d0"), required:t("configuration_tokens.m_c2c05049aa06"), default:t("configuration_tokens.m_b5fda2d21fb2"), mode:t("configuration_tokens.m_21f93bd50fc1"), policy:t("configuration_tokens.m_bbf1879d5799"), profile:t("configuration_tokens.m_4ec793c37034"),
  duration:t("configuration_fields.m_a7947509c350"), timeout:t("configuration_fields.m_a5efd0dff63f"), seed:t("configuration_fields.m_691f178ed6c6"), samples:t("configuration_fields.m_a50adc309336"), sample:t("configuration_fields.m_83bbd2cabb54"), search:t("configuration_fields.m_41a3c0033a42"), exact:t("configuration_fields.m_269225bb5597"),
  analysis:t("configuration_fields.m_561ad79c7de4"), difficulty:t("configuration_fields.m_13a0df4bd90c"), confidence:t("configuration_fields.m_93694193e23b"), probability:t("configuration_fields.m_82b8204ef11c"), risk:t("configuration_fields.m_f212351c9f5e"), weight:t("configuration_fields.m_99a5fe99405a"), weights:t("configuration_fields.m_50db3f0cbf18"),
  video:t("configuration_fields.m_8a4cbedde18f"), audio:t("configuration_fields.m_dfd68f27c512"), motion:t("configuration_fields.m_139f5cbccb9c"), reduced:t("configuration_fields.m_3fed07f166b1"), fps:t("configuration_fields.m_3ca8dfbd3c6c"), bitrate:t("configuration_fields.m_7f8bc5140979"), quality:t("configuration_fields.m_884222efb729"),
  width:t("configuration_fields.m_40504fc32bfc"), height:t("configuration_fields.m_c2da4a9c4f06"), color:t("configuration_fields.m_041b469dc8a9"), colors:t("configuration_fields.m_9b6f97d8666a"), font:t("configuration_fields.m_4004b5dc6406"), layout:t("configuration_fields.m_df51fc9ec6f0"), safe:t("configuration_fields.m_ec16d1b917ea"), area:t("configuration_fields.m_e76b193fd367"),
  reveal:t("configuration_fields.m_0d9a95f86ba2"), timer:t("configuration_fields.m_931a8fb333c4"), pause:t("configuration_fields.m_5f09a5bd2bea"), caption:t("configuration_fields.m_d88673aeea34"), captions:t("configuration_fields.m_d88673aeea34"), explanation:t("configuration_fields.m_e78e51fdadaf"),
  organic:t("configuration_fields.m_b44edae2d156"), guided:t("configuration_fields.m_23b1892db8ec"), retention:t("configuration_fields.m_253bf404cfbd"), selection:t("configuration_fields.m_60fb16c5bc8b"), challenge:t("configuration_fields.m_20b8a9eab31e"), challenges:t("configuration_fields.m_afeedf9fe76a"),
  worker:t("configuration_fields.m_bbf3d733f4dc"), workers:t("configuration_fields.m_2224d1fa2967"), concurrency:t("configuration_fields.m_33c7c07748cf"), retries:t("configuration_fields.m_14d5786f2e64"), retry:t("configuration_fields.m_14d5786f2e64"), memory:t("configuration_fields.m_0760bf594af6"), storage:t("configuration_fields.m_f08625064ac0"),
  security:t("configuration_fields.m_3b309351a2d4"), privacy:t("configuration_fields.m_b2a0889b3131"), audit:t("configuration_fields.m_3a82b23a1d3f"), hash:t("configuration_fields.m_a30a2ed1a937"), hashes:t("configuration_fields.m_fc7ff61b67fb"), hidden:t("configuration_fields.m_403d7ea60117"), observer:t("configuration_fields.m_c3816641af0e"),
  allow:t("configuration_fields.m_7ff3e7cb92d0"), only:t("configuration_fields.m_52a86c5c98e1"), when:t("configuration_fields.m_9146b8021a34"), before:t("configuration_fields.m_52e897d4df3a"), after:t("configuration_fields.m_3dc3397bee39"), active:t("configuration_fields.m_eebba9046d9b"), status:t("configuration_fields.m_c3a4749caed4"), version:t("configuration_fields.m_f14158b9c061"), id:t("configuration_fields.m_00e998a144f4"),
  target:t("configuration_fields.m_acd37606a532"), rollout:t("configuration_fields.m_b3e2507d4202"), percentage:t("configuration_fields.m_0b39fb8d555d"), automatic:t("configuration_fields.m_9b8722a6f13d"), publish:t("configuration_fields.m_332a8dade43c"), publishing:t("configuration_fields.m_332a8dade43c"),
  codec:t("configuration_fields.m_ba67afc25481"), preset:t("configuration_fields.m_d5882606f91e"), pixel:t("configuration_fields.m_0b137b1eb45e"), format:t("configuration_fields.m_53f36d25c52c"), locale:t("configuration_fields.m_6520b21edd4d"), direction:t("configuration_fields.m_44e0168bf8dc"),
  family:t("configuration_fields.m_5154289e59e3"), cache:t("configuration_fields.m_b52b3dbfd525"), entries:t("configuration_fields.m_437a6fc9da31"), center:t("configuration_fields.m_67876145c7b1"), radius:t("configuration_fields.m_9ed54a480928"), index:t("configuration_fields.m_44af870a3643"),
  show:t("configuration_fields.m_00c914d2c30e"), persist:t("configuration_fields.m_c02c2b99e06a"), seconds:t("configuration_fields.m_f3b3f6e5a913"), gap:t("configuration_fields.m_35f02cb65ebd"), hold:t("configuration_fields.m_2785e8a03af6"), transition:t("configuration_fields.m_1c595ed4ff81"), call:t("configuration_fields.m_2e7c14a0b94e"),
  collect:t("configuration_fields.m_4eaf0a99a86a"), continuation:t("configuration_fields.m_1796ea83af03"), easing:t("configuration_fields.m_6d3927447bc8"), shadow:t("configuration_fields.m_ae99f15992db"), strength:t("configuration_fields.m_7d359826a8d1"), scale:t("configuration_fields.m_3ef1959dc111"),
  steps:t("configuration_fields.m_8953b0330ff1"), parallax:t("configuration_fields.m_41d8aa88878f"), glow:t("configuration_fields.m_e7f9f1c0bdb3"), interpolation:t("configuration_fields.m_d44d0deca2f2"), bezier:t("configuration_fields.m_225de61c4cd0"), arc:t("configuration_fields.m_e7404ba22a6c"),
  physical:t("configuration_fields.m_6ef1a46b835d"), stagger:t("configuration_fields.m_84e86188e166"), background:t("configuration_fields.m_49ed4149e51c"), table:t("configuration_fields.m_31047305f809"), rail:t("configuration_fields.m_bb776d764b6b"), gold:t("configuration_fields.m_43b01bb9f8e9"), text:t("configuration_fields.m_60594d7568c1"),
  muted:t("configuration_fields.m_0c001b610b84"), accent:t("configuration_fields.m_2bb4ca147bcd"), red:t("configuration_fields.m_1ca40414f018"), black:t("configuration_fields.m_6444f538b0c9"), ambience:t("configuration_fields.m_a84c1ef7b118"), countdown:t("configuration_fields.m_28b2899e6e25"), voice:t("configuration_fields.m_dfd68f27c512"),
  provider:t("configuration_fields.m_3e253e9c0afb"), backend:t("configuration_fields.label_constraint_backend"), pack:t("configuration_fields.m_d46c1d1fd3ba"), directory:t("configuration_fields.m_c5109ba15984"), synthetic:t("configuration_fields.m_43e4b7cba86c"), foley:t("configuration_fields.m_79632dbf8f8b"), stereo:t("configuration_fields.m_6748d7f7b0d6"), pan:t("configuration_fields.m_e79475c620fb"),
  spoken:t("configuration_fields.m_b62b0aba8134"), themes:t("configuration_fields.m_8e5f319dd13f"), theme:t("configuration_fields.m_b4affa5656f7"), include:t("configuration_fields.m_fd0f169b19cf"), decision:t("configuration_fields.m_a881a87897ba"), timecodes:t("configuration_fields.m_75d18084cda0"), actual:t("configuration_fields.m_6ef1a46b835d"),
  intent:t("configuration_fields.m_6d8d930bff0b"), recorded:t("configuration_fields.m_4056e5279a62"), assets:t("configuration_fields.m_32052f48126a"), manifest:t("configuration_fields.m_862b89c3fb6c"), fallback:t("configuration_fields.m_b3a4412adaf8"), freeze:t("configuration_fields.m_56e7587aebbd"), license:t("configuration_fields.m_bf8370ff4bc2"),
  callouts:t("configuration_fields.m_31c766496c18"), seat:t("configuration_fields.m_c818646d89e0"), attached:t("configuration_fields.m_1f37cc755576"), lines:t("configuration_fields.m_aaa9a6fabb67"), characters:t("configuration_fields.m_d42d270a66ef"), public:t("configuration_fields.m_41d6e9a63721"), exports:t("configuration_fields.m_2bd8f330f47e"),
  opening:t("configuration_fields.m_96862da9647d"), hook:t("configuration_fields.m_adb9af6bb698"), rewind:t("configuration_fields.m_c3b07bf12723"), setup:t("configuration_fields.m_606621502c2d"), context:t("configuration_fields.m_1b917a08f5d7"), complete:t("configuration_fields.m_d99a74c074af"), replay:t("configuration_fields.m_f342a19e501d"),
  typography:t("configuration_fields.m_034a4165cce4"), interface:t("configuration_fields.m_e128dbd263bd"), display:t("configuration_fields.m_40504fc32bfc"), texture:t("configuration_fields.m_7df0ef4f0278"), kind:t("configuration_fields.m_caa3f2bb4a36"), opacity:t("configuration_fields.m_7c849be6b620"), pattern:t("configuration_fields.m_aa6561fcf6a3"),
  presentation:t("configuration_fields.m_40504fc32bfc"), dominant:t("configuration_fields.m_d11a63da9db1"), avatar:t("configuration_fields.m_a35a713b5b75"), identity:t("configuration_fields.m_078e2ffeb247"), enrollment:t("configuration_fields.m_f2b21294fad1"), qr:t("configuration_fields.m_6cb924e56b65"), quiet:t("configuration_fields.m_5f31656496cf"), zone:t("configuration_fields.m_f0a55225689d"),
  locator:t("configuration_fields.m_6d8c2b871f89"), developer:t("configuration_fields.m_f5c31f10c119"), codes:t("configuration_fields.m_6a69f92d46b9"), contact:t("configuration_fields.m_a060796e5308"), sheet:t("configuration_fields.m_71ac8e45bb11"), columns:t("configuration_fields.m_c0ca2dd2a2b1"), labels:t("configuration_fields.m_00ddce39552c"),
  pacing:t("configuration_fields.m_ba7fcee7d910"), uri:t("configuration_fields.m_d860b06666bc"), template:t("configuration_fields.m_4b8a1586b8af"), names:t("configuration_fields.m_a33b674a5f98"), name:t("configuration_fields.m_52ab09847cf8"),
};
const keyAr: Record<string, string> = {
  foundry_copy:t("helpers.label_foundry_copy"), groups:t("helpers.label_groups"), domain_labels:t("helpers.label_domain_labels"),
  configuration_families:t("helpers.label_configuration_families"), configuration_tokens:t("helpers.label_configuration_tokens"),
  configuration_fields:t("helpers.label_configuration_fields"), helpers:t("helpers.label_helpers"), workflow:t("helpers.label_workflow"),
  settings:t("helpers.label_settings"), evidence:t("helpers.label_evidence"),
  video_production_batch:t("workflow.batch_profile_title"),
  thrill_profile:t("configuration_fields.label_thrill_profile"),
  defaultTarget:t("workflow.batch_profile_default_target"), maximumTarget:t("workflow.batch_profile_maximum_target"),
  laneOrder:t("workflow.batch_profile_lanes"), waveSeedCount:t("workflow.batch_profile_wave_size"), maximumWaves:t("workflow.batch_profile_maximum_waves"),
  seedsPerShard:t("workflow.batch_profile_shard_size"), simulationWorkerCount:t("workflow.batch_capacity_search"),
  simulationTimeoutSeconds:t("workflow.batch_profile_simulation_timeout"), finalistConcurrency:t("workflow.batch_capacity_qualification"), renderConcurrency:t("workflow.batch_capacity_render"), renderPolicy:t("workflow.batch_profile_render_policy"),
  dealConstraint:t("configuration_fields.label_deal_constraint"),
  assumedBuyer:t("configuration_fields.label_assumed_buyer"), purchaseSuits:t("configuration_fields.label_purchase_suits"),
  initialRequiredCards:t("configuration_fields.label_initial_required_cards"),
  initialSuitCountRanges:t("configuration_fields.label_initial_suit_ranges"), finalSuitCountRanges:t("configuration_fields.label_final_suit_ranges"),
  projectPattern:t("configuration_fields.label_project_pattern"), maximumSolutions:t("configuration_fields.label_maximum_solutions"),
  maximumConflicts:t("configuration_fields.label_maximum_conflicts"), safetyTimeoutSeconds:t("configuration_fields.label_safety_timeout"),
  solverWorkers:t("configuration_fields.label_solver_workers"), seedOffset:t("configuration_fields.label_seed_offset"), tags:t("configuration_fields.label_constraint_tags"),
  challenge_copy:t("configuration_fields.m_d222bdeb43c8"),
  preview_copy:t("configuration_fields.label_config_preview_copy"),
  page_title:t("configuration_fields.m_35307d3d268e"),
  code_label:t("configuration_fields.m_153fead26ef3"),
  loading:t("configuration_fields.m_b9d459c66bcb"),
  error_title:t("configuration_fields.m_adf9eff6ffd6"),
  score_aria:t("configuration_fields.m_26be31c35816"),
  us_score:t("configuration_fields.m_02aa437a0858"),
  them_score:t("configuration_fields.m_0829c255a1b5"),
  table_aria:t("configuration_fields.m_d044e48c680b"),
  play_target:t("configuration_fields.m_f4d472065063"),
  decision_eyebrow:t("configuration_fields.m_334b7be9234c"),
  confidence_aria:t("configuration_fields.m_7c3699e2c593"),
  confidence_unsure:t("configuration_fields.m_9ba63e2f987d"),
  confidence_confident:t("configuration_fields.m_b3db706475b9"),
  confidence_certain:t("configuration_fields.m_bde7c409d3d6"),
  drag_hint:t("configuration_fields.m_c32f8ba2ac81"),
  drag_release:t("configuration_fields.m_70309472a95f"),
  drag_up:t("configuration_fields.m_d99f7345bf39"),
  hand_aria:t("configuration_fields.m_4bb396ffc816"),
  commit:t("configuration_fields.m_a44cd70583bf"),
  committing:t("configuration_fields.m_3bda36c0add0"),
  result_default_label:t("configuration_fields.m_19135cccdb65"),
  result_default_title:t("configuration_fields.m_1e6845429123"),
  reason_label:t("configuration_fields.m_360b9617cb5a"),
  participants:t("configuration_fields.m_42c450a9ed03"),
  recommended_rate:t("configuration_fields.m_5b39ff485ccc"),
  audience_difficulty:t("configuration_fields.m_e2e74ae164de"),
  player_choices:t("configuration_fields.m_cfbd7bae477c"),
  friend_choices:t("configuration_fields.m_a3cf7377af62"),
  next_challenge:t("configuration_fields.m_42a5f31e0a65"),
  guest_name:t("configuration_fields.m_a316950a2fda"),
  role_dealer:t("configuration_fields.m_c1e66fc91cab"),
  role_buyer:t("configuration_fields.m_1294ec29cf59"),
  role_player:t("configuration_fields.m_dad728908e15"),
  buyer_with_name:t("configuration_fields.m_f0ee0a489584"),
  contract_sun:t("configuration_fields.m_b75f57f8211a"),
  contract_hokum:t("configuration_fields.m_973ecef5a85b"),
  contract_ashkal:t("configuration_fields.m_80ceed9b7c13"),
  purchase_first:t("configuration_fields.m_9d29816c5675"),
  purchase_second:t("configuration_fields.m_4f406f0854e8"),
  purchase_hokum_summary:t("configuration_fields.video_purchase_hokum_summary"),
  purchase_plain_summary:t("configuration_fields.video_purchase_plain_summary"),
  truth_exact_optimal:t("configuration_fields.m_753e454c0c1e"),
  truth_robust_best_known:t("configuration_fields.m_38a64a4eda34"),
  truth_debatable:t("configuration_fields.m_1fe087efb618"),
  truth_unresolved:t("configuration_fields.m_984c7afeb2b8"),
  truth_rejected:t("configuration_fields.m_4dacda28f257"),
  prompt_choose_acting_action:t("configuration_fields.m_f66478c9841b"),
  prompt_predict_opponent_play:t("configuration_fields.m_d07bd7685877"),
  prompt_read_partner_signal:t("configuration_fields.m_96b291c35f4e"),
  prompt_identify_threat:t("configuration_fields.m_1a0762549a47"),
  prompt_plan_next_response:t("configuration_fields.m_6526c0cbf473"),
  prompt_infer_distribution:t("configuration_fields.m_367ea461911d"),
  prompt_post_reveal_analysis:t("configuration_fields.m_f3f84708c7a4"),
  metric_round_margin:t("configuration_fields.m_5443c4f8cac6"),
  metric_contract_success:t("configuration_fields.m_d9fcffa59056"),
  metric_round_lead:t("configuration_fields.m_7dd8b7723fe8"),
  reason_verified:t("configuration_fields.m_4e91036bfe00"),
  reason_margin:t("configuration_fields.m_94b883d972da"),
  reason_contract:t("configuration_fields.m_288eac00a9d8"),
  reason_ranked:t("configuration_fields.m_c8c3736bd675"),
  no_statistics:t("configuration_fields.m_b0d60294e43f"),
  result_supported:t("configuration_fields.m_5d29c43eb79d"),
  result_different:t("configuration_fields.m_95606ea059af"),
  result_recorded:t("configuration_fields.m_223443b6df31"),
  your_choice:t("configuration_fields.m_67823a11d2f5"),
  recommendation:t("configuration_fields.m_7b24736f5f15"),
  note_multiple:t("configuration_fields.m_0c846d3620a8"),
  note_single:t("configuration_fields.m_b6617df4a9e5"),
  note_unavailable:t("configuration_fields.m_79cb0eac4c8a"),
  calibrating:t("configuration_fields.m_5d64608c974e"),
  missing_code:t("configuration_fields.m_38330fea835b"),
  unsupported_prompt:t("configuration_fields.m_4dac85942cd9"),
  request_error:t("configuration_fields.m_78cefdb808c0"),
  brand_name:t("configuration_fields.preview_brand_name"), brand_mark:t("configuration_fields.preview_brand_mark"), brand_aria:t("configuration_fields.preview_brand_aria"),
  artifact_switcher_aria:t("configuration_fields.preview_artifact_switcher_aria"), flagship_label:t("configuration_fields.preview_flagship_label"), decision_label:t("configuration_fields.preview_decision_label"),
  viewer_aria:t("configuration_fields.preview_viewer_aria"), captions_label:t("configuration_fields.preview_captions_label"), captions_button:t("configuration_fields.preview_captions_button"), still_alt:t("configuration_fields.preview_still_alt"),
  flagship_title:t("configuration_fields.preview_flagship_title"), decision_title:t("configuration_fields.preview_decision_title"), playback_options_aria:t("configuration_fields.preview_playback_options_aria"),
  speed_label:t("configuration_fields.preview_speed_label"), speed_aria:t("configuration_fields.preview_speed_aria"), fullscreen:t("configuration_fields.preview_fullscreen"), chapters_aria:t("configuration_fields.preview_chapters_aria"),
  qa_summary:t("configuration_fields.preview_qa_summary"), scene_label:t("configuration_fields.preview_scene_label"), safe_area:t("configuration_fields.preview_safe_area"), disclosure:t("configuration_fields.preview_disclosure"),
  chapter_hook:t("configuration_fields.preview_chapter_hook"), chapter_replay:t("configuration_fields.preview_chapter_replay"), chapter_context:t("configuration_fields.preview_chapter_context"), chapter_question:t("configuration_fields.preview_chapter_question"),
  chapter_reveal:t("configuration_fields.preview_chapter_reveal"), chapter_continuation:t("configuration_fields.preview_chapter_continuation"), chapter_outcome:t("configuration_fields.preview_chapter_outcome"),
  scene_video:t("configuration_fields.preview_scene_video"), scene_poster:t("configuration_fields.preview_scene_poster"), scene_reveal:t("configuration_fields.preview_scene_reveal"), scene_outcome:t("configuration_fields.preview_scene_outcome"),
  schema_version:t("configuration_fields.m_9f500990eb06"), analysis_defaults:t("configuration_fields.m_4e2e3880c253"), conventions:t("configuration_fields.m_bc9fdd5e677b"),
  deterministic_seed:t("configuration_fields.m_bce5b362726f"), exact_allocation_limit:t("configuration_fields.m_20fd22b75a80"), exact_search_max_remaining:t("configuration_fields.m_df649afd9622"),
  maximum_sampling_attempts:t("configuration_fields.m_b73bb7fae0b1"), requested_samples:t("configuration_fields.m_d4dcf1e98af7"), bidding:t("configuration_fields.m_53f10e278252"),
  ashkal:t("configuration_fields.m_68ba4e657a64"), first_round:t("configuration_fields.m_5aadac2f43fd"), second_round:t("configuration_fields.m_8db1e7b1bb6f"), heuristics:t("configuration_fields.m_84f2d2fd4703"),
  pass_allowed:t("configuration_fields.m_d991a556561e"), rounds:t("configuration_fields.m_81ed27a3de52"), card_rules:t("configuration_fields.m_ca470e87628f"), contracts:t("configuration_fields.m_9adfda86daf5"),
  dealing:t("configuration_fields.m_e648102d9749"), deck:t("configuration_fields.m_d5612199b676"), disputed_options:t("configuration_fields.m_fac0448d241c"), escalation:t("configuration_fields.m_87c2fc51e1bb"),
  play:t("configuration_fields.m_a673481dfecc"), policies:t("configuration_fields.m_23999e49095c"), profile:t("configuration_fields.m_19148782809e"), projects:t("configuration_fields.m_5818066ebd9f"), scoring:t("configuration_fields.m_c13c1545dca9"),
  seating:t("configuration_fields.m_3ad56bd970e1"), enabled:t("configuration_fields.m_288f135c14bf"), allowed:t("configuration_fields.m_8e852a53a532"), default:t("configuration_fields.m_b512200e339c"), require_preselected:t("configuration_fields.m_15e55d83d2dc"),
  hokum_uses_purchase_suit:t("configuration_fields.m_54041098c663"), hokum_must_differ_from_purchase_suit:t("configuration_fields.m_1531ce2c4895"),
  eligible_priority_positions:t("configuration_fields.m_8c76857c923b"), resolves_to:t("configuration_fields.m_c88717b5bac6"), taker:t("configuration_fields.m_29b869b63918"),
  sun_threshold:t("configuration_fields.m_caf006c25a12"), hokum_threshold:t("configuration_fields.m_0380a887b66f"), ashkal_partner_threshold:t("configuration_fields.m_86c6daa1d511"),
  second_sun_threshold:t("configuration_fields.m_d2fb72cee41e"), second_hokum_threshold:t("configuration_fields.m_bfa0e8b08cc7"),
  must_follow_suit:t("configuration_fields.m_624ccbd973dc"), must_trump_when_void:t("configuration_fields.m_c65c1926bf74"), must_overtrump_when_possible:t("configuration_fields.m_9efe74e12107"),
  partner_winning_exception:t("configuration_fields.m_535f2566bf2e"), first_leader:t("configuration_fields.m_d98be0ba8fb8"), configured_first_leader:t("configuration_fields.m_83f0d3e830cf"),
  declaration_timing:t("configuration_fields.m_82a1afc02d79"), reveal_timing:t("configuration_fields.m_f56cc396d2b0"), comparison_timing:t("configuration_fields.m_a3f5f4e57826"),
  max_scoring_projects_per_player:t("configuration_fields.m_3346f7fe2e24"), only_winning_team_scores:t("configuration_fields.m_01e6f09a1f90"),
  cards_may_overlap:t("configuration_fields.m_c918cf121de8"), sequences:t("configuration_fields.m_261b9eee50a6"), four_of_kind:t("configuration_fields.m_34f6e6adde62"), baloot:t("configuration_fields.m_877947bc82e8"),
  match:t("configuration_fields.m_62f599878f59"), target:t("configuration_fields.m_477b5cc15272"), kabout:t("configuration_fields.m_50844e1f1cd0"), conversion:t("configuration_fields.m_0127a427f3a7"),
  identity:t("configuration_fields.m_a8288b51b658"), enrollment_uri_template:t("configuration_fields.m_ab42cadc7ccb"), default_player_names:t("configuration_fields.m_4dab2d40547f"),
  localizations:t("configuration_fields.m_3bdda3800848"), renderer_plugin:t("configuration_fields.m_a597e93c819a"), render:t("configuration_fields.m_51c2e233ba1d"),
  'video.decision.challenge_title':t("configuration_fields.video_challenge_title"), 'video.decision.challenge_question':t("configuration_fields.video_challenge_question"),
  'video.decision.generated_title':t("configuration_fields.video_generated_title"), 'video.decision.generated_question':t("configuration_fields.video_generated_question"),
  'video.card_back.label':t("configuration_fields.video_card_back_label"), 'video.card_back.subtitle':t("configuration_fields.video_card_back_subtitle"),
  'video.decision.choose_best_move':t("configuration_fields.m_fb1fad927497"), 'video.decision.best_move':t("configuration_fields.m_ae79d19105d4"), 'video.decision.timer_open':t("configuration_fields.m_896ca3b2fe86"),
  'video.role.dealer':t("configuration_fields.m_c1e66fc91cab"), 'video.role.buyer':t("configuration_fields.m_1294ec29cf59"),
  'video.accessibility.score_and_game':t("configuration_fields.m_6f2453f3498f"), 'video.accessibility.qr':t("configuration_fields.m_02cc1f6c64b3"), 'video.accessibility.hand':t("configuration_fields.m_919e2bef6910"),
  'video.analysis.expert_reason':t("configuration_fields.m_6f72d05988ee"), 'video.analysis.verified_tactical':t("configuration_fields.m_1e16f32834bc"),
  'video.analysis.only_winning_move':t("configuration_fields.m_75748719c25f"), 'video.analysis.contract_rescue':t("configuration_fields.m_89bc503ca2e7"),
  'video.analysis.only_winning_contract_rescue':t("configuration_fields.m_f85d0f5f0c0a"), 'video.analysis.small_card_big_gain':t("configuration_fields.m_b4401a38854b"),
  'video.analysis.large_card_sacrifice':t("configuration_fields.m_672d143e839d"), 'video.analysis.kabout_creation_prevention':t("configuration_fields.m_4e8d82784d7a"),
  'video.analysis.kabout_creation':t("configuration_fields.m_66599f072265"), 'video.analysis.kabout_prevention':t("configuration_fields.m_3e43ba84f639"),
  'video.analysis.entry_unblocking':t("configuration_fields.m_f1aebac9797f"), 'video.analysis.trump_control':t("configuration_fields.m_48ba7770ca10"), 'video.analysis.partner_enablement':t("configuration_fields.m_0c07bd788af8"),
  'video.audio.entry_control_explanation':t("configuration_fields.m_6bc41d43493f"),
  'video.observer.predict_opponent_play':t("configuration_fields.m_d07bd7685877"), 'video.observer.read_partner_signal':t("configuration_fields.m_96b291c35f4e"),
  'video.observer.identify_threat':t("configuration_fields.m_c679972987bb"), 'video.observer.plan_next_response':t("configuration_fields.m_1dd963864610"),
  'video.observer.infer_distribution':t("configuration_fields.m_367ea461911d"), 'video.observer.post_reveal_analysis':t("configuration_fields.m_de752edeb22d"), 'video.observer.insight':t("configuration_fields.m_103295e49c80"),
  'video.bid.hokum':t("configuration_fields.m_aeecef8b170d"), 'video.bid.sun':t("configuration_fields.m_7bd7dc8aa3f8"), 'video.bid.ashkal':t("configuration_fields.m_0fe986539aa5"),
  'video.bid.challenge_title':t("configuration_fields.video_bid_title"), 'video.bid.challenge_question':t("configuration_fields.video_bid_question"),
  'video.bid.round.1':t("configuration_fields.m_9d29816c5675"), 'video.bid.round.2':t("configuration_fields.m_4f406f0854e8"),
  'video.purchase.hokum_summary':t("configuration_fields.video_purchase_hokum_summary"), 'video.purchase.plain_summary':t("configuration_fields.video_purchase_plain_summary"),
  'video.project.siri':t("configuration_fields.m_c3ba5fb9cc6b"), 'video.project.fifty':t("configuration_fields.m_1218e6714c34"), 'video.project.hundred':t("configuration_fields.m_b179f4c804e4"), 'video.project.hundred_sequence':t("configuration_fields.m_b179f4c804e4"), 'video.project.hundred_four_kind':t("configuration_fields.m_b179f4c804e4"), 'video.project.four_hundred':t("configuration_fields.m_aff1d1ee3b03"),
  'video.result.title':t("configuration_fields.m_d8ccd4aa2f27"), 'video.result.us':t("configuration_fields.m_02aa437a0858"), 'video.result.them':t("configuration_fields.m_0829c255a1b5"),
  'video.result.round':t("configuration_fields.m_a1df15fb9f7a"), 'video.result.projects':t("configuration_fields.m_95ffda1c8a0c"), 'video.result.baloot':t("configuration_fields.m_6c57fdc8f9ff"),
  'video.result.awarded':t("configuration_fields.m_fc8cd7428b01"), 'video.result.total':t("configuration_fields.m_7480f9935f77"),
  'video.review.bid_title':t("configuration_fields.video_review_bid_title"), 'video.review.bid_question':t("configuration_fields.video_review_bid_question"),
  'video.review.play_title':t("configuration_fields.video_review_play_title"), 'video.review.play_question':t("configuration_fields.video_review_play_question"),
  'video.review.historical_title':t("configuration_fields.video_review_historical_title"), 'video.review.historical_question':t("configuration_fields.video_review_historical_question"),
  concurrency:t("configuration_fields.m_c6ad311cf8bd"), codec:t("configuration_fields.m_ba67afc25481"), preset:t("configuration_fields.m_7183c1e16146"), crf:t("configuration_fields.m_6b67e46cddf8"),
  pixel_format:t("configuration_fields.m_eda44529a1ac"), audio_rate:t("configuration_fields.m_d4f77f033924"), locale:t("configuration_fields.m_e18d4236ce18"), direction:t("configuration_fields.m_8b910547f987"),
  font_family:t("configuration_fields.m_4a93190504fb"), play_base_cache_entries:t("configuration_fields.m_831daafa4153"), layout:t("configuration_fields.m_ee04aef0624f"),
  safe_margin:t("configuration_fields.m_3eb725e963ea"), table_center_y:t("configuration_fields.m_a1c6a4778309"), table_radius_x:t("configuration_fields.m_1da66d629ea3"), table_radius_y:t("configuration_fields.m_6d15bd361fe1"),
  card_width:t("configuration_fields.m_79c8d146ecea"), card_height:t("configuration_fields.m_419284ceb848"), viewer_card_width:t("configuration_fields.m_d43e712eb9e6"), viewer_card_height:t("configuration_fields.m_b5a212a0e744"),
  minimum_card_index_size:t("configuration_fields.m_cb5eadab2a9f"), orient_side_cards_to_table:t("configuration_fields.m_5e69b15505dc"),
  keep_played_cards_upright:t("configuration_fields.m_5351f371153e"), show_hidden_hand_counts:t("configuration_fields.m_d078072202ce"), show_team_score:t("configuration_fields.m_5a144c0029f4"),
  show_trick_counter:t("configuration_fields.m_bf2305948a33"), show_dealer_badge:t("configuration_fields.m_ee1db94a250d"), show_buyer_badge:t("configuration_fields.m_99cb5cfa8217"),
  show_first_leader_badge:t("configuration_fields.m_1b038f29978c"), show_projects:t("configuration_fields.m_a7ed1e20eced"), project_announcement_trick:t("configuration_fields.m_b1acbb15d0a5"),
  project_reveal_trick:t("configuration_fields.m_80a6a071ec1a"), persist_revealed_projects:t("configuration_fields.m_f1ab6d8ebbbe"), show_score_breakdown:t("configuration_fields.m_68f0f38e8098"),
  show_shuffle_commitment:t("configuration_fields.m_b16f32ce0c81"), timeline:t("configuration_fields.m_79b9bc840940"), intro_seconds:t("configuration_fields.m_e580bceb54a6"), shuffle_seconds:t("configuration_fields.m_2b70df3ac2ab"),
  initial_deal_seconds:t("configuration_fields.m_a6fb4e43e17b"), bidding_seconds:t("configuration_fields.m_0737ef43ba46"), final_deal_seconds:t("configuration_fields.m_c28bdb78c1c2"), project_seconds:t("configuration_fields.m_9335c7d7c422"),
  escalation_seconds:t("configuration_fields.m_17272470ffd1"), card_flight_seconds:t("configuration_fields.m_704336a454b4"), between_cards_seconds:t("configuration_fields.m_874d132346a9"), trick_hold_seconds:t("configuration_fields.m_7e360e360362"),
  pause_seconds:t("configuration_fields.m_2077a8c3ccb7"), reveal_seconds:t("configuration_fields.m_7c2a18e51cd3"), shared_comparison_seconds:t("configuration_fields.m_5732f569d774"), final_seconds:t("configuration_fields.m_abf3feb093f1"),
  transition_seconds:t("configuration_fields.m_719347e809fb"), bid_call_seconds:t("configuration_fields.m_39e7f4d86a92"), bid_gap_seconds:t("configuration_fields.m_4150d07441d7"), bidding_result_seconds:t("configuration_fields.m_e5fdde898b2d"),
  project_call_seconds:t("configuration_fields.m_ef4068ab8b86"), project_reveal_seconds:t("configuration_fields.m_0eaadbe50e15"), trick_collect_seconds:t("configuration_fields.m_f9bb297cc382"), continuation_seconds:t("configuration_fields.m_e7d8873c95a4"),
  explanation_seconds:t("configuration_fields.m_55b1717af7db"), contract_hold_seconds:t("configuration_fields.m_b5f813f82a9d"), escalation_call_seconds:t("configuration_fields.m_17c19e8872c6"), public_callout_seconds:t("configuration_fields.m_cc03081c247a"),
  winner_glow_seconds:t("configuration_fields.m_e02c96ae4bbc"), next_leader_seconds:t("configuration_fields.m_08e4aa9ad011"), motion:t("configuration_fields.m_1a954ca7f6c1"),
  reduced:t("configuration_fields.m_e1da47ccb1fc"), flight_steps:t("configuration_fields.m_6fb93c3dbd24"), card_arc_height_px:t("configuration_fields.m_36e37e5bcca3"), card_scale_from:t("configuration_fields.m_ec22df739b62"),
  collect_stagger:t("configuration_fields.m_2841382c37c4"), collection_steps:t("configuration_fields.m_4e6dae4f5ab6"), winner_glow:t("configuration_fields.m_92841ecb46a1"), easing:t("configuration_fields.m_1adf255ee230"),
  shadow_strength:t("configuration_fields.m_313cedd1924f"), table_parallax:t("configuration_fields.m_efc95de68084"), transition_steps:t("configuration_fields.m_da6c31d3bdcf"), per_frame_interpolation:t("configuration_fields.m_d74de26ba853"),
  bezier_card_flight:t("configuration_fields.m_81a4c34ad0e6"), physical_trick_collection:t("configuration_fields.m_74195b91912e"), colors:t("configuration_fields.m_872d1c305c30"), background:t("configuration_fields.m_bd9f9d3c94aa"),
  table:t("configuration_fields.m_d17448fad3ac"), table_dark:t("configuration_fields.m_32db1eedf043"), rail:t("configuration_fields.m_0c4bf9b9edc7"), gold:t("configuration_fields.m_c95a1d1246e3"), text:t("configuration_fields.m_9e296c12bee0"), muted:t("configuration_fields.m_88be0af1156f"),
  accent:t("configuration_fields.m_ffbaead454bb"), red:t("configuration_fields.m_8495c04bb1af"), black:t("configuration_fields.m_d308611dc18c"), audio:t("configuration_fields.m_5463c45fb230"), ambience:t("configuration_fields.m_4c4fd1014676"),
  card_clicks:t("configuration_fields.m_a6f4c09e929c"), countdown:t("configuration_fields.m_40cdd24eb5f7"), reveal_chime:t("configuration_fields.m_f0ac40836bb3"), voice_callouts:t("configuration_fields.m_bc4512d8b476"),
  voice_provider:t("configuration_fields.m_18c55242be21"), voice_pack_directory:t("configuration_fields.m_78488946dfc3"), synthetic_callout_cues:t("configuration_fields.m_2f4d1405f5e1"), table_foley:t("configuration_fields.m_97b2b5f78e60"),
  voice_pack_manifest:t("configuration_fields.m_81afc9691e76"), explanation_voice_file:t("configuration_fields.m_0fd66b12ce5e"), explanation_voice_localization_key:t("configuration_fields.m_d1b9c5a219a6"),
  explanation_voice_mode:t("configuration_fields.m_95187a37f688"), explanation_voice_transcript:t("configuration_fields.m_7c1677be4b5b"), explanation_voice_volume:t("configuration_fields.m_9239c0445749"),
  stereo_seat_pan:t("configuration_fields.m_628dc1620c4a"), target_lufs:t("configuration_fields.m_1532b9bd574e"), spoken_voice_pack:t("configuration_fields.m_1ab9c2c2c0a8"), foley:t("configuration_fields.m_0f2112ef5a2d"),
  one_piece_explanation:t("configuration_fields.m_6d06bfd6c097"), themes:t("configuration_fields.m_0a25fef92615"), theme:t("configuration_fields.m_3f96d071166c"), audit:t("configuration_fields.m_031e5c5124c8"),
  show_quality_on_reveal:t("configuration_fields.m_4afaa07a4690"), include_decision_timecodes:t("configuration_fields.m_323532ff37f2"), show_actual_intent_only_when_recorded:t("configuration_fields.m_aeb939e19af6"),
  card_assets:t("configuration_fields.m_975788427b6a"), provider:t("configuration_fields.m_63b4776e3c46"), pack_id:t("configuration_fields.m_318ca0088d17"), manifest:t("configuration_fields.m_e67b664f5423"), back_id:t("configuration_fields.m_3018ce8fd441"),
  fallback_provider:t("configuration_fields.m_f328ffa94ca2"), freeze_in_shared_environment:t("configuration_fields.m_d1afe8368bd3"), license_status:t("configuration_fields.m_ef3013d5cdf3"),
  callouts:t("configuration_fields.m_3abf858046e6"), seat_attached:t("configuration_fields.m_6418ef28ef37"), maximum_lines:t("configuration_fields.m_bcffcd3a4509"), maximum_characters_per_line:t("configuration_fields.m_b12e48f58fcf"),
  fade_in_ms:t("configuration_fields.m_ac2c0ddad489"), fade_out_ms:t("configuration_fields.m_bf73ef250171"), public_only_before_reveal:t("configuration_fields.m_169749ebbf40"), safe_areas:t("configuration_fields.m_9e356c471460"),
  top_px:t("configuration_fields.m_08a05100c39f"), right_px:t("configuration_fields.m_b7ffb8d566e8"), bottom_px:t("configuration_fields.m_630a0043eaf6"), left_px:t("configuration_fields.m_7b0f7882d546"), show_mask_in_player:t("configuration_fields.m_6dd3f4d66b24"),
  exports:t("configuration_fields.m_a3f0cd50f401"), hooked_flagship:t("configuration_fields.m_be1a93dca875"), decision_clip:t("configuration_fields.m_3444a736796e"), minimum_seconds:t("configuration_fields.m_b409763aebe2"), target_seconds:t("configuration_fields.m_ac1e18284f98"),
  maximum_seconds:t("configuration_fields.m_7441a80283bc"), opening_hook_seconds:t("configuration_fields.m_3fdd07364742"), rewind_seconds:t("configuration_fields.m_21503dbfc1e6"), setup_end_seconds:t("configuration_fields.m_14921eba1863"), play_end_seconds:t("configuration_fields.m_a2b89ee43b38"),
  reveal_end_seconds:t("configuration_fields.m_169c514668e4"), decision_seconds:t("configuration_fields.m_2077a8c3ccb7"), context_seconds:t("configuration_fields.m_2b0321e4369d"), include_complete_replay:t("configuration_fields.m_91a29f234641"),
  typography:t("configuration_fields.m_ae88f87c0fe6"), interface_family:t("configuration_fields.m_330bf19ed064"), display_family:t("configuration_fields.m_365fcd92afff"), maximum_reason_lines:t("configuration_fields.m_873e2cfe188f"),
  use_kufi_for_short_display_only:t("configuration_fields.m_6370903a0ba5"), texture:t("configuration_fields.m_4af94ff2a605"), kind:t("configuration_fields.m_f0849cf37650"), opacity:t("configuration_fields.m_a93d08afa69d"), pattern_id:t("configuration_fields.m_aa6a11d4cbf3"),
  presentation:t("configuration_fields.m_a23265bd0d41"), keep_table_visible:t("configuration_fields.m_1b201ded10b2"), large_text_panels_before_reveal:t("configuration_fields.m_6313a6e6c6fa"),
  table_dominant_layout:t("configuration_fields.m_af3e2447c678"), show_avatar_initials:t("configuration_fields.m_8952b259d7eb"), show_public_callouts:t("configuration_fields.m_74de4d542332"),
  show_deep_analysis_only_after_reveal:t("helpers.m_c0eaf644de3a"), show_trick_winner_text:t("helpers.m_b96e0aad240f"), winner_yellow_glow:t("helpers.m_dc5da3bab0a3"),
  show_micro_seat_markers:t("helpers.m_f5d1490260e5"), show_identity_pill:t("helpers.m_3baf99ecece0"), show_enrollment_qr:t("helpers.m_a683e8aa389f"),
  qr_size_px:t("helpers.m_f93fe26f6b35"), qr_quiet_zone_modules:t("helpers.m_5e33c185c5a2"), identity_position:t("helpers.m_f0e4350afffc"), show_state_locator:t("helpers.m_8d4cb4965890"),
  state_locator_characters:t("helpers.m_9848c5eee308"), show_developer_card_codes:t("helpers.m_eb63fcbac11e"), contact_sheet_columns:t("helpers.m_ca5ebe4790f6"),
  contact_sheet_labels:t("helpers.m_deac4812bf41"), score_separator_style:t("helpers.m_81aa73d0cedb"), pacing_profile:t("helpers.m_e13a18f9c271"),
  observer:t("helpers.m_fd4ee4b4001a"), identify_threat:t("helpers.m_707ecd6eef54"), infer_distribution:t("helpers.m_3440c92bba3f"),
  insight:t("helpers.m_478e6dc14dab"), plan_next_response:t("helpers.m_bf87bc74d385"), post_reveal_analysis:t("helpers.m_6c8b38f34fb5"),
  predict_opponent_play:t("helpers.m_bc57c696a51e"), read_partner_signal:t("helpers.m_eb5fc6d2e143"),
  selection_policy:t("helpers.m_79ce304e36db"), answer_policy:t("helpers.m_1ebc6fd999a5"),
  campaign_proof:t("helpers.m_569156bc7b21"), campaign_screening:t("helpers.m_a6a3b921bd7d"),
  deep_analysis:t("helpers.m_0d19bb80fb45"), perspective_policy:t("helpers.m_3d13d5f97c6a"),
  tactical_detectors:t("helpers.m_b803b12c227e"), guided_search:t("helpers.m_930c83fda502"),
  disagreement_search:t("helpers.m_68ec49764cc6"), context_search:t("helpers.m_3900e416ab97"),
  sakka_generation:t("helpers.m_17f175cdeec4"), finalist_validation:t("helpers.m_480906faf84e"), trace_import_policy:t("helpers.m_544a57bb60f0"),
  resource_limits:t("helpers.m_5cdbe425f611"), scenario_defaults:t("helpers.m_caa39ab9c2ed"),
  request_defaults:t("helpers.m_a3cbe23847aa"), publication:t("helpers.m_e293654e0376"),
  stability:t("helpers.m_9a99c1512a80"), diversity:t("helpers.m_8bbf42367955"),
  minimumSurroundingQuality:t("helpers.m_1c64370d6997"),
  minimumSurroundingConfidence:t("helpers.m_27694c67235a"),
  minimumSurroundingMateriality:t("helpers.m_da4a7dadae7a"),
  maximumSurroundingMaterialMistakes:t("helpers.m_adcea5fa3de1"),
  maximumSurroundingUnverifiedDecisions:t("helpers.m_9684b52a42fa"),
  maximumSurroundingCatastrophicCrossings:t("helpers.m_dfc8e3421e5b"),
  requireVerifiedTacticalPlay:t("helpers.m_6f581de87917"),
  agentProfileId:t("helpers.m_38782c44b3e9"), maximumTracesPerJob:t("helpers.m_e0195498c759"),
  requireQualifiedAgentReleaseForPublication:t("helpers.m_4248d143fccb"), allowLocalPreview:t("helpers.m_fa5467a76b43"),
  review:t("configuration_fields.label_config_review"),
  validation:t("configuration_fields.label_config_validation"),
  consent_defaults:t("configuration_fields.label_config_consent"),
  training:t("configuration_fields.label_config_training"),
  improvement:t("configuration_fields.label_config_improvement"),
  review_clip:t("configuration_fields.label_config_review_clip"),
  copy:t("configuration_fields.label_config_copy"),
};
const thrillFieldAr: Record<string, string> = {
  id:t("configuration_fields.label_thrill_profile_id"), profile_id:t("configuration_fields.label_thrill_profile_id"), revision:t("configuration_fields.label_thrill_revision"),
  minimumCriticalPoints:t("configuration_fields.label_thrill_minimum_critical_points"), minimum_critical_points:t("configuration_fields.label_thrill_minimum_critical_points"),
  minimumScore:t("configuration_fields.label_thrill_minimum_score"), minimum_score:t("configuration_fields.label_thrill_minimum_score"),
  weights:t("configuration_fields.label_thrill_weights"), overallPeak:t("configuration_fields.label_thrill_overall_peak"),
  secondPeak:t("configuration_fields.label_thrill_second_peak"), topKThrillMass:t("configuration_fields.label_thrill_top_k_mass"),
  criticalDensity:t("configuration_fields.label_thrill_critical_density"), perspectiveBreadth:t("configuration_fields.label_thrill_perspective_breadth"),
  multiAngleYield:t("configuration_fields.label_thrill_multi_angle_yield"), causalChainStrength:t("configuration_fields.label_thrill_causal_chain_strength"),
  outcomeImportance:t("configuration_fields.label_thrill_outcome_importance"), visualClarity:t("configuration_fields.label_thrill_visual_clarity"),
  narrativeArc:t("configuration_fields.label_thrill_narrative_arc"),
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {'accept': 'application/json', ...(init?.body ? {'content-type': 'application/json'} : {}), ...init?.headers},
  });
  if (!response.ok) {
    let message = (t("helpers.m_dc85c66d7c4d") + response.status + ")");
    try { const body = await response.json(); message = body.detail || message; } catch { /* response is not JSON */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

const loadResearchPerspectives = (code: string, seat: SeatName) =>
  api<{items:JsonObject[]}>(`/api/v2/research-results/${encodeURIComponent(code)}/perspectives?observerSeat=${encodeURIComponent(seat)}`);

function configurationErrorCopy(error: unknown): string {
  const message=(error as Error)?.message||'';
  return message.includes('missing required scope: baloot.config.admin')
    ? t("settings.m_16f5c65b587c")
    : message;
}

function shortAudit(value?: string) { return value ? `${value.slice(0, 6)}…${value.slice(-4)}` : '—'; }
function humanKey(key: string) {
  if (keyAr[key]) return keyAr[key];
  const words = key.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase().split(/[_-]/).filter(Boolean);
  const translated = words.map((word) => tokenAr[word]).filter(Boolean);
  return translated.length === words.length ? translated.join(' ') : translated.length ? (t("helpers.m_732824c7561c") + translated.join(' ') + "") : t("helpers.m_615a69649fe0");
}
function friendlyPath(path: string) {
  const parts = path.split('.').filter(Boolean).map(humanKey);
  return parts.slice(-2).join(' ← ') || t("helpers.m_0af5b373fce3");
}
function contextualKeyName(key: string, path: (string|number)[]) {
  return path.includes('thrill_profile') && thrillFieldAr[key] ? thrillFieldAr[key] : humanKey(key);
}
function configFieldName(key: string, value: JsonValue, path: (string|number)[] = []) {
  if (key === 'completeStateAlternativeFirst' && hasFoundryCopy('configuration_fields.label_bid_alternative_first')) return t("configuration_fields.label_bid_alternative_first");
  if ((key.startsWith('m_') || key.startsWith('label_')) && typeof value === 'string') {
    const preview = value.length > 54 ? `${value.slice(0, 52)}…` : value;
    return t("helpers.label_foundry_message") + preview + '»';
  }
  if (key.startsWith('player_') && typeof value === 'string') {
    const preview = value.length > 54 ? `${value.slice(0, 52)}…` : value;
    return (t("helpers.m_0e6ea9b5eafa") + preview + "»");
  }
  if (typeof value === 'string' && (
    /^(actor|phase|stage|rank|suit)_/.test(key)
    || ['header_title', 'context_actor', 'reveal_fallback', 'reason_main', 'reason_secondary', 'footer_submitted', 'footer_blind'].includes(key)
  )) {
    const preview = value.length > 54 ? `${value.slice(0, 52)}…` : value;
    return t("helpers.label_video_message") + preview + '»';
  }
  return contextualKeyName(key, path);
}
function valueExample(value: JsonValue) {
  if (typeof value === 'boolean') return value ? t("helpers.m_0709cc0ffd6a") : t("helpers.m_12f8e04695d4");
  if (typeof value === 'number') return (t("helpers.m_2457c3deaf0d") + number.format(value) + ".");
  if (typeof value === 'string') {
    const shown = value || t("helpers.m_cf1b365fe1d2");
    return (t("helpers.m_cab655087fee") + shown + "».");
  }
  if (Array.isArray(value)) return (t("helpers.m_58aa3b06df30") + number.format(value.length) + t("helpers.m_2f2f5f99ff4e"));
  return t("helpers.m_3251e6cea94b");
}
function configValueExample(key: string, value: JsonValue, path: (string|number)[] = []) {
  const thrillField = path.includes('thrill_profile');
  if (thrillField && (key === 'id' || key === 'profile_id' || key === 'revision')) return t("helpers.thrill_identity_example");
  if (thrillField && (key === 'minimumCriticalPoints' || key === 'minimum_critical_points' || key === 'minimumScore' || key === 'minimum_score')) return t("helpers.thrill_gate_example");
  if (thrillField && key === 'weights') return t("helpers.thrill_weight_example");
  if (thrillField && (key === 'overallPeak' || key === 'secondPeak' || key === 'topKThrillMass' || key === 'criticalDensity')) return t("helpers.thrill_peak_example");
  if (thrillField && (key === 'perspectiveBreadth' || key === 'multiAngleYield')) return t("helpers.thrill_breadth_example");
  if (thrillField && (key === 'causalChainStrength' || key === 'outcomeImportance' || key === 'visualClarity' || key === 'narrativeArc')) return t("helpers.thrill_clarity_example");
  if (key === 'video.purchase.hokum_summary' || key === 'purchase_hokum_summary') return t("helpers.video_purchase_hokum_example");
  if (key === 'video.purchase.plain_summary' || key === 'purchase_plain_summary') return t("helpers.video_purchase_plain_example");
  if (key === 'explanation_voice_mode') return t("helpers.m_31641db9ef57");
  if (key === 'explanation_voice_transcript') return t("helpers.m_44cfdf5290cd");
  if (key === 'explanation_voice_localization_key') return t("helpers.m_126ac82fe280");
  if (key === 'explanation_voice_volume') return t("helpers.m_54f35588d783");
  if (key === 'minimumSurroundingQuality') return t("helpers.m_7a9598adb193");
  if (key === 'minimumSurroundingConfidence') return t("helpers.m_4f1509da766e");
  if (key === 'minimumSurroundingMateriality') return t("helpers.m_f6291f9729de");
  if (key === 'maximumSurroundingMaterialMistakes') return t("helpers.m_26413a198374");
  if (key === 'maximumSurroundingUnverifiedDecisions') return t("workflow.m_741da328f67a");
  if (key === 'maximumSurroundingCatastrophicCrossings') return t("workflow.m_2a7a0a5c1320");
  if (key === 'requireVerifiedTacticalPlay') return t("workflow.m_d0fdf8d1aa0b");
  if (key === 'simulationWorkerCount') return t("workflow.batch_profile_search_example");
  if (key === 'finalistConcurrency') return t("workflow.batch_profile_qualification_example");
  if (key === 'renderConcurrency') return t("workflow.batch_profile_render_example");
  if (key === 'renderPolicy') return t("workflow.batch_profile_render_policy_example");
  if (key === 'waveSeedCount') return t("workflow.batch_profile_wave_example");
  if (key === 'dealConstraint') return t("helpers.deal_constraint_example");
  if (key === 'backend' && value === 'ortools_cp_sat_v1') return t("helpers.deal_constraint_backend_example");
  if (key === 'assumedBuyer') return t("helpers.deal_constraint_buyer_example");
  if (key === 'purchaseSuits' || key === 'initialRequiredCards' || key === 'initialSuitCountRanges' || key === 'finalSuitCountRanges') return t("helpers.deal_constraint_shape_example");
  if (key === 'projectPattern') return t("helpers.deal_constraint_project_example");
  if (['maximumSolutions', 'maximumConflicts', 'safetyTimeoutSeconds', 'solverWorkers', 'seedOffset'].includes(key)) return t("helpers.deal_constraint_budget_example");
  if (key === 'agentProfileId') return t("workflow.m_5b8fb431fec1");
  if (key === 'maximumTracesPerJob') return t("workflow.m_aea8d1710ba8");
  if (key === 'requireQualifiedAgentReleaseForPublication') return t("workflow.m_b5c9e3f8e089");
  if (key === 'allowLocalPreview') return t("workflow.m_8a80ee882052");
  return valueExample(value);
}
function friendlyJobError(error?: string | null) {
  if (!error) return '';
  if (error.includes('cancelled')) return t("workflow.m_fc9aad51a22b");
  if (error.includes('no complete-trace artifacts')) return t("workflow.m_b072e0c76576");
  return t("workflow.m_303534acbecf");
}
function nextPatchVersion(value: string) {
  const match = /^(\d+)\.(\d+)\.(\d+)/.exec(value);
  if (!match) return '1.0.0';
  return `${match[1]}.${match[2]}.${Number(match[3]) + 1}`;
}
function actionLabel(action?: JsonObject) {
  if (!action) return '—';
  const label = typeof action.label === 'string' ? action.label : '';
  const card = typeof action.card === 'string' ? action.card : '';
  const code = typeof action.code === 'string' ? action.code : '';
  const kind = typeof action.kind === 'string' ? action.kind : '';
  const contract = typeof action.contract === 'string' ? action.contract : '';
  const actionName = typeof action.action === 'string' ? action.action : '';
  const trumpSuit = typeof action.trumpSuit === 'string' ? action.trumpSuit : '';
  const eventType = typeof action.eventType === 'string' ? action.eventType : '';
  const projectKind = typeof action.kind === 'string' ? action.kind : '';
  if (eventType === 'project_declared') {
    return `${t("workflow.research_project_declared")} ${projectLabel[projectKind] || projectKind}`.trim();
  }
  if (eventType === 'projects_revealed') {
    const projects = Array.isArray(action.projects) ? action.projects as JsonObject[] : [];
    const names = Array.from(new Set(projects.map((item) => projectLabel[String(item.kind)] || String(item.kind)).filter(Boolean)));
    return `${t("workflow.research_projects_revealed")} ${names.join(' · ')}`.trim();
  }
  if (eventType === 'baloot_declared' || eventType === 'baloot_included_in_project') {
    return t("workflow.research_baloot_event");
  }
  const bid = actionName === 'pass' ? decisionLabel.pass : contractLabel[actionName] || '';
  const bidWithSuit = actionName === 'hokum' && trumpSuit
    ? `${bid} ${suitSymbol[trumpSuit] || trumpSuit}`
    : bid;
  return label || card || code || contractLabel[contract] || contract || bidWithSuit || actionName || decisionLabel[kind] || t("workflow.m_5e67642412a2");
}
function tacticDisplayName(kind:string) {
  const labels:Record<string,string>={
    only_winning_move:t("workflow.research_tactic_only_move"),
    small_card_big_gain:t("workflow.research_tactic_small_gain"),
    large_card_sacrifice:t("workflow.research_tactic_sacrifice"),
    kabout_creation:t("workflow.research_tactic_kabout"),
    kabout_preservation:t("workflow.research_tactic_kabout_preservation"),
    kabout_prevention:t("workflow.research_tactic_kabout_prevention"),
    contract_rescue:t("workflow.research_tactic_contract_rescue"),
    entry_unblocking:t("workflow.research_tactic_entry"),
    trump_control:t("workflow.research_tactic_trump"),
    partner_enablement:t("workflow.research_tactic_partner"),
    project_bidding:t("workflow.research_tactic_project"),
    debate:t("workflow.research_tactic_debate"),
  };
  return labels[kind]||kind;
}

type FoundrySection = 'overview' | 'research' | 'generation' | 'laffat' | 'sakkas' | 'renders' | 'settings';

const foundrySectionIds: FoundrySection[] = ['overview', 'research', 'generation', 'laffat', 'sakkas', 'renders', 'settings'];
const defaultLaffaFilters: UnifiedLaffaFilters = {
  search:'', sourceKind:'all', qualificationStatus:'all', truthLabel:'all',
  contract:'all', project:'all', tactic:'all', seat:'all', renderStatus:'all', sort:'newest', offset:0,
};

function App() {
  useResearchWorkflowCopyRevision();
  const [view, setView] = useState<FoundrySection>(() => {
    const requested = new URLSearchParams(window.location.search).get('section') as FoundrySection | null;
    return requested && foundrySectionIds.includes(requested) ? requested : 'overview';
  });
  const [profile, setProfile] = useState<FoundryProfile | null>(null);
  const [readinessLoaded, setReadinessLoaded] = useState(false);
  const [researchBundles, setResearchBundles] = useState<ConfigBundle[]>([]);
  const researchRecipeController = useMemo(
    () => new ResearchRecipeController(researchRecipeApi),
    [],
  );
  const [researchRecipeSnapshot, setResearchRecipeSnapshot] = useState(researchRecipeController.getSnapshot);
  const [researchRecipeReferences, setResearchRecipeReferences] = useState<Record<string, ResearchRecipe>>({});
  const researchRecipes = useMemo(() => {
    const byCode = new Map(
      researchRecipeSnapshot.items.map((recipe) => [recipe.code, recipe]),
    );
    Object.values(researchRecipeReferences).forEach((recipe) => {
      byCode.set(recipe.code, recipe);
    });
    return [...byCode.values()];
  }, [researchRecipeReferences, researchRecipeSnapshot.items]);
  const hydrateResearchRecipe = useCallback(async (code:string) => {
    const recipe = await researchRecipeController.hydrateExact(code);
    setResearchRecipeReferences((current) => current[recipe.code] === recipe
      ? current
      : {...current, [recipe.code]:recipe});
    return recipe;
  }, [researchRecipeController]);
  const [retiringRecipeCodes, setRetiringRecipeCodes] = useState<Set<string>>(() => new Set());
  const [researchRuns, setResearchRuns] = useState<ResearchRun[]>([]);
  const [researchStats, setResearchStats] = useState<ResearchStats | null>();
  const researchStatsRevision = researchRuns.map(({runCode, status}) => `${runCode}:${status}`).join('|');
  const activeResearchRunCodes = researchRuns.filter((item) => ['queued', 'running'].includes(item.status))
    .map((item) => item.runCode).sort().join('|');
  const [semanticDisagreements, setSemanticDisagreements] = useState<SemanticDisagreement[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [catalogs, setCatalogs] = useState<CatalogRecord[]>([]);
  const [sakkas, setSakkas] = useState<FoundrySakka[]>([]);
  const [publications, setPublications] = useState<JsonObject[]>([]);
  const [selected, setSelected] = useState<Campaign | null>(null);
  const [shards, setShards] = useState<Shard[]>([]);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [generationJob, setGenerationJob] = useState<SakkaGenerationJob | null>(null);
  const [analysisJob, setAnalysisJob] = useState<AnalysisJob | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisJob[]>([]);
  const [candidatePage, setCandidatePage] = useState<CandidatePage | null>(null);
  const [finalistJob, setFinalistJob] = useState<FinalistValidationJob | null>(null);
  const [finalistHistory, setFinalistHistory] = useState<FinalistValidationJob[]>([]);
  const [prepared, setPrepared] = useState<PreparedCandidate | null>(null);
  const [renderJob, setRenderJob] = useState<RenderJob | null>(null);
  const [renderHistory, setRenderHistory] = useState<RenderJob[]>([]);
  const [laffaPage, setLaffaPage] = useState<UnifiedLaffaPage | null>(null);
  const [laffaFilters, setLaffaFilters] = useState<UnifiedLaffaFilters>(defaultLaffaFilters);
  const [laffaLoading, setLaffaLoading] = useState(false);
  const [videoBatch, setVideoBatch] = useState<VideoProductionBatch | null>(null);
  const [videoBatchHistory, setVideoBatchHistory] = useState<VideoProductionBatch[]>([]);
  const [qualifiedLaffat, setQualifiedLaffat] = useState<BatchPreparedItem[]>([]);
  const [qualifiedLaffatCursor, setQualifiedLaffatCursor] = useState<string | null>(null);
  const [qualifiedLaffatLoading, setQualifiedLaffatLoading] = useState(false);
  const [videoBatchTarget, setVideoBatchTarget] = useState<number | null>(null);
  const [videoBatchSeed, setVideoBatchSeed] = useState(0);
  const [drawer, setDrawerState] = useState<DrawerState>(null);
  const drawerIntent = useRef(0);
  // Every open or close is a new intent; a loader drops its late result once the returned check is false.
  // An updater function edits the open drawer in place and keeps the current intent.
  const setDrawer = useCallback((value: SetStateAction<DrawerState>) => {
    const intent = typeof value === 'function' ? drawerIntent.current : ++drawerIntent.current;
    setDrawerState(value);
    return () => intent === drawerIntent.current;
  }, []);
  const [openHelp, setOpenHelp] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [generationLane, setGenerationLane] = useState<'organic' | 'guided' | 'disagreement'>('guided');
  const [seedStart, setSeedStart] = useState(0);
  const [seedCount, setSeedCount] = useState(4);
  const [seedsPerShard, setSeedsPerShard] = useState(1);
  const [generationSeed, setGenerationSeed] = useState(20260804);
  const [generationDealer, setGenerationDealer] = useState<'north' | 'east' | 'south' | 'west'>('east');
  const deepLinkedResultOpened = useRef<string | null>(null);
  const deepLinkedLaffaOpened = useRef<string | null>(null);
  const unifiedLaffaSourceRevision = useMemo(() => [
    ...researchRuns.map((item) => `${item.runCode}:${item.status}:${item.resultCount}`),
    ...catalogs.map((item) => item.catalogHash),
    ...sakkas.map((item) => item.catalogHash),
    ...renderHistory.map((item) => `${item.jobId}:${item.status}`),
  ].join('|'), [catalogs, renderHistory, researchRuns, sakkas]);

  const loadUnifiedLaffat = useCallback(async (filters: UnifiedLaffaFilters) => {
    setLaffaLoading(true);
    try {
      const query = new URLSearchParams({limit:'50', offset:String(filters.offset), sort:filters.sort});
      if (filters.search.trim()) query.set('search', filters.search.trim());
      if (filters.sourceKind !== 'all') query.set('sourceKind', filters.sourceKind);
      if (filters.qualificationStatus !== 'all') query.set('qualificationStatus', filters.qualificationStatus);
      if (filters.truthLabel !== 'all') query.set('truthLabel', filters.truthLabel);
      if (filters.contract !== 'all') query.set('contract', filters.contract);
      if (filters.project !== 'all') query.set('project', filters.project);
      if (filters.tactic !== 'all') query.set('tactic', filters.tactic);
      if (filters.seat !== 'all') query.set('seat', filters.seat);
      if (filters.renderStatus !== 'all') query.set('renderStatus', filters.renderStatus);
      setLaffaPage(await api<UnifiedLaffaPage>(`/api/v2/laffat?${query.toString()}`));
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setLaffaLoading(false);
    }
  }, []);

  const refresh = useCallback(async (reportStartupError?: (error: Error) => void) => {
    await ensureConfigurationIdentity();
    const runtimeRequest=api<FoundryProfile>('/api/v2/foundry-profile').then((value)=>{setProfile(value);return value;});
    const bundleRequest=api<{items: ConfigBundle[]}>('/api/v2/configuration/bundles?status=published&limit=200').then((value)=>{setResearchBundles(value.items);return value;});
    const runsRequest = api<{items: ResearchRun[]}>('/api/v2/research-runs?limit=100')
      .then((value) => setResearchRuns(value.items));
    const overview = await Promise.all([
      runtimeRequest,
      bundleRequest,
      runsRequest,
      api<{items: SemanticDisagreement[]}>('/api/v2/semantic-disagreements?limit=100'),
      api<{items: Campaign[]}>('/api/v2/campaigns?limit=100'),
      api<{items: CatalogRecord[]}>('/api/v2/foundry-catalogs?limit=100'),
      api<{items: FoundrySakka[]}>('/api/v2/sakkas?limit=100'),
      api<{items: JsonObject[]}>('/api/v2/publications?status=review&limit=100').catch(() => ({items: []})),
      api<{items: AnalysisJob[]}>('/api/v2/analysis-batches?limit=20').catch(() => ({items: []})),
      api<{items: FinalistValidationJob[]}>('/api/v2/finalist-validations?limit=20').catch(() => ({items: []})),
      api<{items: RenderJob[]}>('/api/v2/render-batches?limit=20').catch(() => ({items: []})),
      api<{items: VideoProductionBatch[]}>('/api/v2/video-production-batches?limit=20').catch(() => ({items: []})),
      api<{items: BatchPreparedItem[]; nextCursor:string|null}>('/api/v2/qualified-laffat?limit=100&sort=laffa_score&direction=desc').catch(() => ({items: [], nextCursor:null})),
    ]).catch(async (error: Error) => {
      if (!reportStartupError) throw error;
      reportStartupError(error);
      // These requests can still update the layout after the aggregate rejects.
      await Promise.allSettled([runtimeRequest, bundleRequest, runsRequest]);
      return null;
    });
    if (!overview) return;
    const [,,, disagreementPage, campaignPage, catalogPage, sakkaPage, reviewPage, analysisPage, finalistPage, renderPage, videoBatchPage, qualifiedPage] = overview;
    const recentCampaigns = [...campaignPage.items].sort((left, right) => {
      const created = Date.parse(right.createdAt) - Date.parse(left.createdAt);
      return created || right.campaignHash.localeCompare(left.campaignHash);
    });
    setSemanticDisagreements(disagreementPage.items);
    setCampaigns(recentCampaigns); setCatalogs(catalogPage.items); setSakkas(sakkaPage.items); setPublications(reviewPage.items);
    setSelected((current) => current
      ? recentCampaigns.find((item) => item.campaignHash === current.campaignHash) || null
      : recentCampaigns[0] || null);
    setAnalysisHistory(analysisPage.items); setFinalistHistory(finalistPage.items); setRenderHistory(renderPage.items);
    setVideoBatchHistory(videoBatchPage.items);
    setQualifiedLaffat(qualifiedPage.items);
    setQualifiedLaffatCursor(qualifiedPage.nextCursor);
    setVideoBatch((current) => current
      ? videoBatchPage.items.find((item) => item.jobId === current.jobId) || null
      : videoBatchPage.items[0] || null);
    setAnalysisJob((current) => current
      ? analysisPage.items.find((item) => item.jobId === current.jobId) || null
      : null);
    setFinalistJob((current) => {
      const updated = current && finalistPage.items.find((item) => item.jobId === current.jobId);
      return updated || null;
    });
    setRenderJob((current) => current
      ? renderPage.items.find((item) => item.jobId === current.jobId) || null
      : renderPage.items.find((item) => ['queued', 'running'].includes(item.status)) || null);
  }, [view]);

  useEffect(() => researchRecipeController.subscribe(() => {
    setResearchRecipeSnapshot(researchRecipeController.getSnapshot());
  }), [researchRecipeController]);
  useEffect(() => {
    void researchRecipeController.loadInitial().catch((error:unknown) => {
      setMessage(researchRecipeErrorMessage(error));
    });
  }, [researchRecipeController]);
  const loadMoreQualifiedLaffat = async () => {
    if (!qualifiedLaffatCursor || qualifiedLaffatLoading) return;
    setQualifiedLaffatLoading(true);
    try {
      const page = await api<{items:BatchPreparedItem[]; nextCursor:string|null}>(`/api/v2/qualified-laffat?limit=100&sort=laffa_score&direction=desc&cursor=${encodeURIComponent(qualifiedLaffatCursor)}`);
      setQualifiedLaffat((current) => {
        const combined = new Map(current.map((item) => [item.bundleHash, item]));
        page.items.forEach((item) => combined.set(item.bundleHash, item));
        return [...combined.values()];
      });
      setQualifiedLaffatCursor(page.nextCursor);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setQualifiedLaffatLoading(false);
    }
  };

  const retireResearchRecipe = async (recipe:ResearchRecipe) => {
    setRetiringRecipeCodes((current) => new Set(current).add(recipe.code));
    try {
      const retired = await researchRecipeController.retireRecipe(recipe.code);
      setResearchRecipeReferences((current) => ({
        ...current,
        [retired.code]:retired,
      }));
    } catch (error) {
      setMessage(researchRecipeErrorMessage(error));
      throw error;
    } finally {
      setRetiringRecipeCodes((current) => {
        const next = new Set(current);
        next.delete(recipe.code);
        return next;
      });
    }
  };

  useEffect(() => {
    const reportStartupError = (error: Error) => setMessage(error.message);
    void refresh(reportStartupError).catch(reportStartupError).finally(() => setReadinessLoaded(true));
  }, [refresh]);
  useEffect(() => {
    if ((view !== 'overview' && view !== 'research') || !researchRuns.length) return;
    api<ResearchStats>('/api/v2/research-stats').then(setResearchStats).catch((error: Error) => {
      setResearchStats((current) => current ?? null);
      setMessage(error.message);
    });
  }, [view, researchStatsRevision]);
  useEffect(() => {
    if (view !== 'laffat') return;
    void loadUnifiedLaffat(laffaFilters);
  }, [laffaFilters, loadUnifiedLaffat, unifiedLaffaSourceRevision, view]);
  useEffect(() => {
    if (!activeResearchRunCodes) return;
    const runCodes = activeResearchRunCodes.split('|');
    let disposed = false;
    let inFlight = false;
    const timer = window.setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      void Promise.all(runCodes.map((code) => api<ResearchRun>(`/api/v2/research-runs/${encodeURIComponent(code)}`)))
        .then((updates) => {
          if (disposed) return;
          const byCode = new Map(updates.map((item) => [item.runCode, item]));
          setResearchRuns((current) => current.map((item) => byCode.get(item.runCode) || item));
          if (updates.every((item) => !['queued', 'running'].includes(item.status))) {
            void refresh().catch((error: Error) => setMessage(error.message));
          }
        })
        .catch((error: Error) => { if (!disposed) setMessage(error.message); })
        .finally(() => { inFlight = false; });
    }, 1500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [activeResearchRunCodes, refresh]);
  useEffect(() => {
    if (!campaigns.some((item) => ['queued', 'running'].includes(item.status))) return;
    const timer = window.setInterval(() => refresh().catch((error: Error) => setMessage(error.message)), 2000);
    return () => window.clearInterval(timer);
  }, [campaigns, refresh]);
  useEffect(() => {
    if (!selected) return;
    let disposed = false;
    const load = async () => {
      try {
        const [shardPage, attemptPage] = await Promise.all([
          api<{shards: Shard[]}>(`/api/v2/campaigns/${selected.campaignHash}/shards`),
          api<{attempts: Attempt[]}>(`/api/v2/campaigns/${selected.campaignHash}/attempts`),
        ]);
        if (!disposed) { setShards(shardPage.shards); setAttempts(attemptPage.attempts); }
      } catch (error) { if (!disposed) setMessage((error as Error).message); }
    };
    load();
    if (!['queued', 'running'].includes(selected.status)) return () => { disposed = true; };
    const timer = window.setInterval(load, 1500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [selected?.campaignHash, selected?.status]);
  useEffect(() => {
    if (!selected) return;
    const campaignAnalysis = analysisHistory.find(
      (item) => item.request?.campaignHash === selected.campaignHash
    ) || null;
    setAnalysisJob((current) => current?.request?.campaignHash === selected.campaignHash
      ? current
      : campaignAnalysis);
    setFinalistJob(campaignAnalysis
      ? finalistHistory.find((item) => item.request?.analysisJobId === campaignAnalysis.jobId) || null
      : null);
  }, [analysisHistory, finalistHistory, selected?.campaignHash]);
  useEffect(() => {
    if (!generationJob || !['queued', 'running'].includes(generationJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const current = await api<SakkaGenerationJob>(`/api/v2/sakka-generations/${generationJob.jobId}`);
        if (current.status === 'completed') await refresh();
        setGenerationJob(current);
      } catch (error) { setMessage((error as Error).message); }
    }, 800);
    return () => window.clearInterval(timer);
  }, [generationJob?.jobId, generationJob?.status, refresh]);
  useEffect(() => {
    if (!analysisJob || !['queued', 'running'].includes(analysisJob.status)) return;
    const timer = window.setInterval(async () => {
      try { setAnalysisJob(await api<AnalysisJob>(`/api/v2/analysis-batches/${analysisJob.jobId}`)); }
      catch (error) { setMessage((error as Error).message); }
    }, 900);
    return () => window.clearInterval(timer);
  }, [analysisJob?.jobId, analysisJob?.status]);
  useEffect(() => {
    if (analysisJob && candidatePage?.jobId !== analysisJob.jobId) setCandidatePage(null);
  }, [analysisJob?.jobId]);
  useEffect(() => {
    if (!analysisJob || analysisJob.status !== 'completed') return;
    api<CandidatePage>(`/api/v2/analysis-batches/${analysisJob.jobId}/candidates`)
      .then(setCandidatePage)
      .catch((error: Error) => setMessage(error.message));
  }, [analysisJob?.jobId, analysisJob?.status]);
  useEffect(() => {
    if (!finalistJob || !['queued', 'running'].includes(finalistJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const current = await api<FinalistValidationJob>(`/api/v2/finalist-validations/${finalistJob.jobId}`);
        setFinalistJob(current);
        if (current.status === 'completed') {
          setMessage(current.result?.localPreviewReady
            ? t("workflow.m_259bcd97e7aa")
            : t("workflow.m_aee41f8c4605"));
        }
      } catch (error) { setMessage((error as Error).message); }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [finalistJob?.jobId, finalistJob?.status]);
  const renderHistoryActive = renderHistory.some((item) => ['queued', 'running'].includes(item.status));
  useEffect(() => {
    if (!renderHistoryActive) return;
    const timer = window.setInterval(async () => {
      try {
        const page = await api<{items: RenderJob[]}>('/api/v2/render-batches?limit=20');
        setRenderHistory(page.items);
        setRenderJob((current) => current
          ? page.items.find((item) => item.jobId === current.jobId) || null
          : page.items.find((item) => ['queued', 'running'].includes(item.status)) || null);
      }
      catch (error) { setMessage((error as Error).message); }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [renderHistoryActive]);
  useEffect(() => {
    if (!videoBatch || !['queued', 'running'].includes(videoBatch.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const [current, renders] = await Promise.all([
          api<VideoProductionBatch>(`/api/v2/video-production-batches/${videoBatch.jobId}`),
          api<{items: RenderJob[]}>('/api/v2/render-batches?limit=20'),
        ]);
        setVideoBatch(current);
        setRenderHistory(renders.items);
        if (!['queued', 'running'].includes(current.status)) await refresh();
      } catch (error) { setMessage((error as Error).message); }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [videoBatch?.jobId, videoBatch?.status, refresh]);

  const campaignNames = useMemo(() => new Map(campaigns.map((item, index) => [item.campaignHash, (t("workflow.m_6e6d79aaa514") + number.format(index + 1) + "")])), [campaigns]);
  const sakkaNames = useMemo(() => new Map(sakkas.map((item, index) => [item.catalogHash, (t("workflow.m_14f5a5b443d1") + number.format(index + 1) + "")])), [sakkas]);

  useEffect(() => {
    if (!drawer || drawer.kind !== 'campaign' || !selected) return;
    setDrawer((current) => current?.kind === 'campaign' ? {
      ...current,
      title: campaignNames.get(selected.campaignHash) || t("workflow.m_7255069a0d81"),
      content: <CampaignDetails campaign={selected} shards={shards} attempts={attempts} />,
      audit: selected as unknown as JsonObject,
    } : current);
  }, [attempts, campaignNames, drawer?.kind, selected, shards]);

  useEffect(() => {
    if (!renderJob) return;
    setDrawer((current) => current?.kind === 'render' ? {
      ...current,
      subtitle: statusLabel[renderJob.status] || renderJob.status,
      content: <JsonSummary value={renderJob.result || {status:renderJob.status} as JsonObject} />,
      audit: renderJob as unknown as JsonObject,
    } : current);
  }, [renderJob]);

  const visibleCandidates = useMemo(() => [...(candidatePage?.items || [])].sort((left, right) => {
    const readiness = (candidate: AnalysisCandidate) => candidate.playReady ? 3 : candidate.finalistValidationRequired ? 2 : 1;
    return readiness(right) - readiness(left) || right.selectionScore - left.selectionScore || right.confidence - left.confidence;
  }).slice(0, 12), [candidatePage]);
  const completedRenders = useMemo(
    () => renderHistory.filter((item) => item.status === 'completed').slice(0, 6),
    [renderHistory],
  );
  const activeRenders = useMemo(
    () => renderHistory.filter((item) => ['queued', 'running'].includes(item.status)).slice(0, 6),
    [renderHistory],
  );
  const videoBatchActive = Boolean(videoBatch && ['queued', 'running'].includes(videoBatch.status));
  const videoBatchResult = videoBatch?.result || {};
  const videoBatchStage = String(videoBatchResult.stage || videoBatch?.status || 'queued');
  const videoBatchStageProgress = Math.max(0, Math.min(1, Number(videoBatchResult.stageProgress ?? videoBatch?.progress ?? 0)));
  const videoBatchActiveWorkers = videoBatchActive
    ? Math.max(0, Number(videoBatchResult.activeWorkerCount || 0))
    : 0;
  const videoBatchCurrentWave = Math.max(0, Number(videoBatchResult.currentWave || 0));
  // The durable qualified-library endpoint remains the authority. Batch
  // history only enriches the local projection with already-frozen candidate
  // labels; it cannot add or promote a library row.
  const qualifiedCandidateContext = useMemo(() => {
    const values = new Map<string, JsonObject>();
    const jobs = videoBatch
      ? [videoBatch, ...videoBatchHistory.filter((item) => item.jobId !== videoBatch.jobId)]
      : videoBatchHistory;
    for (const job of jobs) {
      const validationItems = Array.isArray(job.result?.validationItems)
        ? job.result.validationItems
        : [];
      for (const entry of validationItems) {
        const candidate = entry.candidate;
        if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) continue;
        const candidateHash = String(candidate.candidateHash || entry.candidateHash || '');
        if (candidateHash && !values.has(candidateHash)) {
          values.set(candidateHash, {
            ...candidate,
            analysisJobId: entry.analysisJobId || null,
          });
        }
      }
    }
    return values;
  }, [videoBatch, videoBatchHistory]);
  const videoBatchPreparedItems = useMemo(() => qualifiedLaffat.map((item) => {
    const context = qualifiedCandidateContext.get(item.candidateHash);
    if (!context) return item;
    return {
      ...item,
      observerSeat: String(context.observerSeat || item.observerSeat || ''),
      dimensions: context.dimensions && typeof context.dimensions === 'object' && !Array.isArray(context.dimensions)
        ? context.dimensions as Record<string, string>
        : item.dimensions,
      bestAction: context.bestAction && typeof context.bestAction === 'object' && !Array.isArray(context.bestAction)
        ? context.bestAction as JsonObject
        : item.bestAction,
      analysisJobId: String(context.analysisJobId || item.analysisJobId || ''),
    };
  }), [qualifiedCandidateContext, qualifiedLaffat]);
  const effectiveVideoBatchTarget = videoBatchTarget ?? profile?.videoProductionBatchProfile.defaultTarget ?? 0;
  const qualifiedTargetGap = Math.max(0, effectiveVideoBatchTarget - qualifiedLaffat.length);
  const completedCampaigns = campaigns.filter((item) => item.status === 'completed');
  useEffect(() => {
    if (seedStart !== 0 || !campaigns.length) return;
    const next = campaigns.reduce((maximum, item) => Math.max(
      maximum,
      Number(item.spec.seedStart || 0) + Number(item.spec.seedCount || 0),
    ), 0);
    if (next > 0) setSeedStart(next);
  }, [campaigns, seedStart]);
  useEffect(() => {
    if (videoBatchSeed !== 0) return;
    const next = campaigns.reduce((maximum, item) => Math.max(
      maximum,
      Number(item.spec.seedStart || 0) + Number(item.spec.seedCount || 0),
    ), 202608050000);
    setVideoBatchSeed(next);
  }, [campaigns, videoBatchSeed]);
  const selectedCandidateCount = selected ? shards.reduce((total, shard) => (
    total + (Array.isArray(shard.result?.candidateHashes) ? shard.result.candidateHashes.length : 0)
  ), 0) : 0;
  const selectedCompatible = Boolean(selected && profile
    && selected.spec.rulesHash === profile.rulesHash
    && selected.spec.bundleHash === profile.bundleHash
    && selected.spec.engineVersion === profile.engineVersion
    && selected.spec.analysisProfileHash === profile.analysisProfileHash
    && selected.spec.perspectiveProfileHash === profile.perspectiveProfileHash
    && selected.spec.selectionProfileHash === profile.selectionProfileHash);
  const evidenceCount = Math.max(candidatePage?.count || 0, catalogs.filter((item) => item.kind.includes('evidence') || item.kind.includes('proof')).length);
  // The first research summary can still shift the readiness controls after refresh.
  const readinessLoading = !readinessLoaded || (researchRuns.length > 0 && researchStats === undefined);
  const readiness = [
    {label: t("workflow.m_d40c6942d37e"), ready: Boolean(profile?.bundleHash), detail: t("workflow.m_406fd6baf625")},
    {label: t("workflow.m_c6ad13aaf02e"), ready: completedCampaigns.length > 0 || sakkas.length > 0, detail: t("workflow.m_0c5f3d3d663d")},
    {label: t("workflow.m_e994737d1101"), ready: evidenceCount > 0 || analysisJob?.status === 'completed', detail: t("workflow.m_9c20cea4b9ce")},
    {label: t("workflow.m_38ce38816ca5"), ready: Boolean(prepared || renderJob?.status === 'completed' || sakkas.length > 0), detail: t("workflow.m_3ec10f6767ee")},
    {label: t("workflow.m_5e527606c752"), ready: publications.length > 0, detail: t("workflow.m_493fe6369db3")},
  ];
  const researchDescription = t("workflow.research_description");
  const workspaceIntro = {
    overview: {
      kicker: t("workflow.overview_kicker"),
      title: t("workflow.overview_title"),
      description: t("workflow.overview_description"),
    },
    research: {
      kicker: t("workflow.research_kicker"),
      title: t("workflow.research_title"),
      description: researchDescription.includes('OR-Tools')
        ? researchDescription
        : t("workflow.nav_research_help"),
    },
    generation: {
      kicker: t("workflow.generation_kicker"),
      title: t("workflow.generation_title"),
      description: t("workflow.generation_description"),
    },
    laffat: {
      kicker: t("workflow.laffat_kicker"),
      title: t("workflow.laffat_title"),
      description: t("workflow.laffat_description"),
    },
    sakkas: {
      kicker: t("workflow.sakkas_kicker"),
      title: t("workflow.sakkas_title"),
      description: t("workflow.sakkas_description"),
    },
    renders: {
      kicker: t("workflow.renders_kicker"),
      title: t("workflow.renders_title"),
      description: t("workflow.renders_description"),
    },
  } as const;
  const selectView = (next: FoundrySection) => {
    setMessage('');
    setView(next);
    const url = next === 'overview' ? '/foundry' : `/foundry?section=${next}`;
    window.history.replaceState(null, '', url);
    window.scrollTo({top: 0, behavior: 'smooth'});
  };

  const ensureGuest = async () => {
    const identity = await api<{identity: JsonObject | null}>('/api/v1/auth/me');
    if (!identity.identity) await api('/api/v1/auth/guest', {method: 'POST', body: JSON.stringify({displayName: t("workflow.m_a53a7869033a")})});
  };
  const openCampaign = useCallback(async (campaign: Campaign) => {
    // No drawer shows while a campaign loads, so the request itself is the intent a later open or close replaces.
    const stillOpen = setDrawer(null);
    setSelected(campaign);
    const campaignAnalysis = analysisHistory.find((item) => item.request?.campaignHash === campaign.campaignHash) || null;
    setAnalysisJob(campaignAnalysis);
    const campaignCandidates = campaignAnalysis?.status === 'completed'
      ? await api<CandidatePage>(`/api/v2/analysis-batches/${campaignAnalysis.jobId}/candidates`)
      : null;
    setCandidatePage(campaignCandidates);
    setFinalistJob(campaignAnalysis
      ? finalistHistory.find((item) => item.request?.analysisJobId === campaignAnalysis.jobId) || null
      : null);
    setPrepared(null); setRenderJob(null);
    const [shardPage, attemptPage] = await Promise.all([
      api<{shards: Shard[]}>(`/api/v2/campaigns/${campaign.campaignHash}/shards`),
      api<{attempts: Attempt[]}>(`/api/v2/campaigns/${campaign.campaignHash}/attempts`),
    ]);
    // The selected campaign's own effect keeps its shards current; a late answer must not overwrite them.
    if (!stillOpen()) return;
    setShards(shardPage.shards); setAttempts(attemptPage.attempts);
    setDrawer({
      kind: 'campaign', title: campaignNames.get(campaign.campaignHash) || t("workflow.m_7255069a0d81"), subtitle: t("workflow.m_97bed3db6a66"),
      content: <CampaignDetails campaign={campaign} shards={shardPage.shards} attempts={attemptPage.attempts} />,
      audit: campaign as unknown as JsonObject,
    });
  }, [analysisHistory, campaignNames, finalistHistory]);
  const createCampaign = async (event: FormEvent) => {
    event.preventDefault(); if (!profile) return; setBusy(true); setMessage('');
    try {
      const campaign = await api<Campaign>('/api/v2/campaigns', {method: 'POST', body: JSON.stringify({
        rulesHash: profile.rulesHash, bundleHash: profile.bundleHash, engineVersion: profile.engineVersion,
        analysisProfileHash: profile.analysisProfileHash, perspectiveProfileHash: profile.perspectiveProfileHash,
        selectionProfileHash: profile.selectionProfileHash,
        agentRelease: profile.laneAgentReleases[generationLane],
        seatAgents: profile.laneSeatAgents[generationLane],
        lanes: [generationLane],
        laneProfileHashes: {[generationLane]: profile.staticLaneProfileHashes[generationLane]},
        seedStart, seedCount, seedsPerShard,
      })});
      await api<Campaign>(`/api/v2/campaigns/${campaign.campaignHash}:execute`, {method:'POST'});
      setMessage(t("workflow.label_campaign_execution_started")); await refresh(); await openCampaign(campaign);
      setSeedStart(seedStart + seedCount);
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const createVideoBatch = async (event: FormEvent) => {
    event.preventDefault();
    if (!profile) return;
    if (qualifiedTargetGap === 0) { setMessage(t("workflow.batch_target_reached")); return; }
    setBusy(true); setMessage('');
    try {
      const waveSeedCount = Number(profile.videoProductionBatchProfile.waveSeedCount || 1);
      const campaignFloor = campaigns.reduce((maximum, item) => Math.max(
        maximum,
        Number(item.spec.seedStart || 0) + Number(item.spec.seedCount || 0),
      ), videoBatchSeed);
      const batchHistoryFloor = videoBatchHistory.reduce((maximum, item) => Math.max(
        maximum,
        Number(item.request.seedStart || 0)
          + Math.max(waveSeedCount, Number(item.result?.roundsGenerated || 0)),
      ), videoBatchSeed);
      const requestedSeed = Math.max(campaignFloor, batchHistoryFloor);
      const job = await api<VideoProductionBatch>('/api/v2/video-production-batches', {
        method:'POST',
        headers:{'Idempotency-Key':`hooked-${profile.videoProductionBatchProfile.sha256}-${qualifiedTargetGap}-${requestedSeed}`},
        body:JSON.stringify({
          targetCount:qualifiedTargetGap,
          seedStart:requestedSeed,
          profileHash:profile.videoProductionBatchProfile.sha256,
        }),
      });
      setVideoBatchSeed(requestedSeed);
      setVideoBatch(job);
      setVideoBatchHistory((current) => [job, ...current.filter((item) => item.jobId !== job.jobId)]);
      setMessage(t("workflow.batch_started"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const cancelVideoBatch = async () => {
    if (!videoBatch || !['queued', 'running'].includes(videoBatch.status)) return;
    setBusy(true); setMessage('');
    try {
      const job = await api<VideoProductionBatch>(`/api/v2/video-production-batches/${videoBatch.jobId}:cancel`, {method:'POST'});
      setVideoBatch(job); setMessage(t("workflow.batch_cancel_requested"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const submitRender = async (ownerHash: string, basename: string, bundleArtifact: JsonObject, keyPrefix: string, source?:ResearchRenderSourceProvenance, proofArtifact?:JsonObject, authorityReceipt?:ResearchPreparedRender['authorityReceipt'], playbackEvidence = false) => {
    const latest = await api<{items: RenderJob[]}>('/api/v2/render-batches?limit=100');
    setRenderHistory(latest.items);
    const related = latest.items.filter((job) => renderJobReferencesBundle(job, ownerHash)
      && (!playbackEvidence || job.request?.playbackEvidence === true));
    const reusable = related.find((job) => job.status === 'completed')
      || related.find((job) => ['queued', 'running'].includes(job.status));
    if (reusable) return {job: reusable, started: false};
    const terminal = related
      .filter((job) => ['cancelled', 'failed'].includes(job.status))
      .sort((left, right) => Date.parse(right.createdAt || '') - Date.parse(left.createdAt || ''))[0];
    const idempotencyKey = terminal
      ? `${keyPrefix}-${ownerHash}-retry-${terminal.jobId}`
      : `${keyPrefix}-${ownerHash}`;
    const job = await api<RenderJob>('/api/v2/render-batches', {
      method:'POST',
      headers:{'Idempotency-Key':idempotencyKey},
      body:JSON.stringify({items:[{bundleArtifact, basename, ...(source?{source}: {}), ...(proofArtifact?{proofArtifact}: {}), ...(authorityReceipt?{authorityReceipt}: {})}], playbackEvidence}),
    });
    setRenderHistory((current) => [job, ...current.filter((item) => item.jobId !== job.jobId)]);
    return {job, started: true};
  };
  const renderPreparedResearch = async (prepared: ResearchPreparedRender) => {
    setBusy(true); setMessage('');
    try {
      const sourceKind = String(prepared.source.kind || 'research_result');
      const sourceCode = String(prepared.source.code || prepared.resultCode || prepared.laffaCode || '');
      const ownerHash = String(prepared.bundleArtifact.ownerHash || prepared.source.traceHash || sourceCode);
      const {job, started} = await submitRender(
        ownerHash,
        prepared.basename,
        prepared.bundleArtifact,
        sourceKind === 'laffa' ? 'laffa' : 'research',
        prepared.source,
        prepared.proofArtifact,
        prepared.authorityReceipt,
        prepared.exportPreset === 'hooked_flagship',
      );
      setRenderJob(job);
      void loadUnifiedLaffat(laffaFilters);
      setDrawer(null);
      setMessage(started ? t("workflow.batch_render_started") : job.status === 'completed'
        ? t("workflow.laffa_library_video_ready")
        : t("workflow.batch_render_started"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const prepareResearchVideo = async (result: ResearchResult, observerSeat: SeatName, ordinal?: number) => {
    setBusy(true); setMessage('');
    try {
      const endpoint = ordinal === undefined
        ? `/api/v2/research-results/${result.resultCode}:prepare-render`
        : `/api/v2/research-results/${result.resultCode}/decisions/${ordinal}:prepare-render`;
      const exportPreset = ordinal === undefined ? 'hooked_flagship' : 'decision_clip';
      const prepared = assertPreparedObserver(await api<ResearchPreparedRender>(endpoint, {
        method:'POST',
        body:JSON.stringify({revealPolicy:'after_commit', exportPreset, observerSeat}),
      }), observerSeat);
      setDrawer({
        kind:'research-render-preview',
        title:t("workflow.research_prepare_video"),
        subtitle:`${result.resultCode} · ${number.format(prepared.decisionOrdinal)}`,
        content:<ResearchRenderPreview prepared={prepared} busy={busy} onRender={renderPreparedResearch} />,
        audit:prepared as unknown as JsonObject,
      });
      setMessage(t("workflow.research_prepare_ready"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const prepareUnifiedLaffaVideo = async (item: UnifiedLaffa) => {
    if (!item.replayValid) return;
    setBusy(true); setMessage('');
    try {
      const preparedQualified = item.sourceLocator.kind === 'qualified_laffa'
        ? qualifiedLaffat.find((candidate) => candidate.shortCode === item.humanCode || candidate.traceHash === item.traceHash)
        : undefined;
      if (preparedQualified) {
        setDrawer({
          kind:'qualified-laffa',
          title:t("workflow.research_prepare_video"),
          subtitle:item.humanCode,
          content:<QualifiedLaffaDetails item={preparedQualified} renders={renderHistory.filter((job)=>renderJobReferencesUnifiedLaffa(job,item))} onRender={renderBatchPreparedItem} renderBusy={busy}/>,
          audit:preparedQualified as unknown as JsonObject,
        });
        setMessage(t("workflow.research_prepare_ready"));
        return;
      }
      const sourceCode = item.sourceLocator.code;
      const endpoint = `/api/v2/laffat/${encodeURIComponent(sourceCode)}:prepare-render`;
      const prepared = await api<ResearchPreparedRender>(endpoint, {
        method:'POST',
        body:JSON.stringify({revealPolicy:'after_commit'}),
      });
      setDrawer({
        kind:'research-render-preview',
        title:t("workflow.research_prepare_video"),
        subtitle:item.humanCode,
        content:<ResearchRenderPreview prepared={prepared} busy={busy} onRender={renderPreparedResearch} />,
        audit:prepared as unknown as JsonObject,
      });
      setMessage(t("workflow.research_prepare_ready"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const openSakkaRound = (traceHash: string) => {
    const code = `LAF-${traceHash.slice(0,10).toUpperCase()}`;
    const target = new URL(window.location.href);
    target.searchParams.set('section', 'laffat');
    target.searchParams.set('laffa', code);
    target.hash = 'laffa-library';
    window.history.replaceState({}, '', target);
    setLaffaFilters((current) => ({...current, search:code, offset:0}));
    setView('laffat');
    setDrawer(null);
  };
  const prepareSakkaRoundVideo = async (traceHash: string) => {
    const code = `LAF-${traceHash.slice(0,10).toUpperCase()}`;
    setBusy(true); setMessage('');
    try {
      const prepared = await api<ResearchPreparedRender>(`/api/v2/laffat/${encodeURIComponent(code)}:prepare-render`, {
        method:'POST',
        body:JSON.stringify({revealPolicy:'after_commit', exportPreset:'hooked_flagship'}),
      });
      setDrawer({
        kind:'research-render-preview',
        title:t("workflow.research_prepare_video"),
        subtitle:code,
        content:<ResearchRenderPreview prepared={prepared} busy={busy} onRender={renderPreparedResearch} />,
        audit:prepared as unknown as JsonObject,
      });
      setMessage(t("workflow.research_prepare_ready"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const verifyResearchPlay = async (result: ResearchResult) => {
    setBusy(true); setMessage('');
    try {
      const proof = await api<JsonObject>(`/api/v2/research-results/${result.resultCode}:verify-play-compiler`, {method:'POST'});
      setDrawer({
        kind:'research-play-proof',
        title:t("workflow.research_play_proof_title"),
        subtitle:result.resultCode,
        content:<ResearchPlayProof proof={proof} />,
        audit:proof,
      });
      setMessage(t("workflow.research_play_proof_completed"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const renderBatchPreparedItem = async (item: BatchPreparedItem) => {
    setBusy(true); setMessage('');
    try {
      const prepared = item.finalistJobId
        ? await api<PreparedCandidate>(`/api/v2/finalist-validations/${item.finalistJobId}:prepare-best`, {
            method:'POST',
            body:JSON.stringify({revealPolicy:'after_commit', exportPreset:'hooked_flagship'}),
          })
        : await api<PreparedCandidate>(`/api/v2/analysis-batches/${item.analysisJobId}/candidates/${item.candidateHash}:prepare`, {
            method:'POST',
            body:JSON.stringify({revealPolicy:'after_commit', exportPreset:'hooked_flagship'}),
          });
      const ownerHash = String(prepared.bundleArtifact.ownerHash || item.bundleHash || item.candidateHash);
      const {job, started} = await submitRender(ownerHash, prepared.basename, prepared.bundleArtifact, 'foundry-batch');
      setRenderJob(job);
      setDrawer(null);
      setMessage(started ? t("workflow.batch_render_started") : job.status === 'completed'
        ? t("workflow.laffa_library_video_ready")
        : t("workflow.batch_render_started"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const openQualifiedLaffa = (item: BatchPreparedItem) => {
    const relatedRenders = renderHistory.filter((job) => renderJobReferencesBundle(job, item.bundleHash, item.basename));
    setDrawer({
      kind:'qualified-laffa',
      title:t("workflow.laffa_library_details_title"),
      subtitle:item.shortCode,
      content:<QualifiedLaffaDetails
        item={item}
        renders={relatedRenders}
        onRender={renderBatchPreparedItem}
        renderBusy={busy || ['queued','running'].includes(renderJob?.status || '')}
      />,
      audit:item as unknown as JsonObject,
    });
  };
  const openVideoBatch = (job = videoBatch) => {
    if (!job) return;
    setDrawer({
      kind:'video-batch',
      title:t("workflow.batch_details_title"),
      subtitle:statusLabel[job.status] || job.status,
      content:<VideoBatchDetails job={job} onRender={renderBatchPreparedItem} onOpen={openQualifiedLaffa} renderBusy={busy || ['queued','running'].includes(renderJob?.status || '')} />,
      audit:job as unknown as JsonObject,
    });
  };
  const generateNaturalSakka = async (event: FormEvent) => {
    event.preventDefault(); if (!profile) return; setBusy(true); setMessage('');
    try {
      await ensureGuest();
      const job = await api<SakkaGenerationJob>('/api/v2/sakka-generations', {
        method: 'POST', headers: {'Idempotency-Key': `natural-${profile.sakkaGenerationProfile.sha256}-${generationSeed}-${generationDealer}`},
        body: JSON.stringify({rootSeed: generationSeed, initialDealer: generationDealer}),
      });
      setGenerationJob(job); setMessage(t("workflow.m_bfc33cbdc9d0"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const startAnalysis = async (campaign = selected) => {
    if (!campaign || !profile || campaign.spec.rulesHash !== profile.rulesHash || campaign.spec.bundleHash !== profile.bundleHash || campaign.spec.analysisProfileHash !== profile.analysisProfileHash || campaign.spec.perspectiveProfileHash !== profile.perspectiveProfileHash || campaign.spec.selectionProfileHash !== profile.selectionProfileHash) {
      setMessage(t("workflow.label_campaign_incompatible_help"));
      return;
    }
    setBusy(true); setMessage('');
    try {
      const job = await api<AnalysisJob>('/api/v2/analysis-batches', {method: 'POST', body: JSON.stringify({campaignHash: campaign.campaignHash})});
      await api<AnalysisJob>(`/api/v2/analysis-batches/${job.jobId}:execute`, {method:'POST'});
      setAnalysisHistory((current) => [job, ...current.filter((item) => item.jobId !== job.jobId)]);
      setAnalysisJob(job); setCandidatePage(null); setFinalistJob(null); setPrepared(null); setRenderJob(null);
      setSelected(campaign); setMessage(t("workflow.m_5ba435def089"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const openCandidate = (candidate: AnalysisCandidate) => setDrawer({
    kind: 'candidate',
    title: (t("workflow.m_1bcbdd2a5e74") + number.format(Math.max(0, visibleCandidates.findIndex((item) => item.candidateHash === candidate.candidateHash)) + 1) + ""),
    subtitle: laffaLabel[candidate.laffaClassification] || t("workflow.m_ee42de255f0f"),
    content: <CandidateDetails candidate={candidate} />,
    audit: candidate as unknown as JsonObject,
  });
  const prepareCandidate = async (candidate: AnalysisCandidate) => {
    if (!analysisJob || !candidate.playReady) return;
    setBusy(true); setMessage('');
    try {
      const value = await api<PreparedCandidate>(`/api/v2/analysis-batches/${analysisJob.jobId}/candidates/${candidate.candidateHash}:prepare`, {method:'POST', body:JSON.stringify({revealPolicy:'after_commit', exportPreset:'hooked_flagship'})});
      setPrepared(value);
      setMessage(t("workflow.m_1a1fc084dd70") + String(value.locator.code || candidate.shortCode) + t("workflow.m_e38684c0a6d3"));
      setDrawer({kind:'prepared', title:t("workflow.m_0bab9d58c31d"), subtitle:t("workflow.m_c485c6b17790") + String(value.locator.code || candidate.shortCode), content:<PreparedDetails value={value} />, audit:value as unknown as JsonObject});
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const validateFinalist = async (candidate: AnalysisCandidate) => {
    if (!analysisJob || candidate.playReady || !candidate.finalistValidationRequired) return;
    setBusy(true); setMessage('');
    try {
      const value = await api<FinalistValidationJob>('/api/v2/finalist-validations', {
        method:'POST',
        headers:{'Idempotency-Key':`foundry-finalist-${analysisJob.jobId}-${candidate.candidateHash}`},
        body:JSON.stringify({analysisJobId:analysisJob.jobId, candidateHashes:[candidate.candidateHash]}),
      });
      setFinalistJob(value);
      setMessage(t("workflow.m_e07e3bec93b5"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const prepareBestFinalist = async (exportPreset: PreparedCandidate['exportPreset']) => {
    if (!finalistJob || finalistJob.status !== 'completed' || !finalistJob.result?.localPreviewReady) return;
    setBusy(true); setMessage('');
    try {
      const value = await api<PreparedCandidate>(`/api/v2/finalist-validations/${finalistJob.jobId}:prepare-best`, {method:'POST', body:JSON.stringify({revealPolicy:'after_commit', exportPreset})});
      setPrepared(value);
      setMessage(exportPreset === 'hooked_flagship' ? t("workflow.m_e9c3c87d96d7") : t("workflow.m_4f660a44646c"));
      setDrawer({kind:'prepared', title:exportPreset === 'hooked_flagship' ? t("workflow.m_bf40007eacb8") : t("workflow.m_34ef1c737707"), subtitle:t("workflow.m_c485c6b17790") + String(value.locator.code || value.candidate.shortCode), content:<PreparedDetails value={value} />, audit:value as unknown as JsonObject});
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const renderPrepared = async () => {
    if (!prepared) return;
    setBusy(true); setMessage('');
    try {
      const ownerHash = String(prepared.bundleArtifact.ownerHash || prepared.candidate.candidateHash);
      const {job, started} = await submitRender(ownerHash, prepared.basename, prepared.bundleArtifact, 'foundry');
      setRenderJob(job);
      setMessage(started ? t("workflow.m_958a62ba5d63") : job.status === 'completed'
        ? t("workflow.laffa_library_video_ready")
        : t("workflow.m_958a62ba5d63"));
    } catch (error) { setMessage((error as Error).message); } finally { setBusy(false); }
  };
  const openSakka = async (item: FoundrySakka) => {
    const title = sakkaNames.get(item.catalogHash) || t("workflow.m_a71ed66fd0dc");
    const stillOpen = setDrawer({
      kind: 'sakka', title, subtitle: t("workflow.m_bdf45938011a"),
      content: <p className="drawer-lead">{t("workflow.sakka_library_loading")}</p>,
    });
    try {
      const detail = await api<FoundrySakkaDetail>(`/api/v2/sakkas/${item.catalogHash}`);
      if (!stillOpen()) return;
      setDrawer({
        kind: 'sakka', title, subtitle: t("workflow.m_bdf45938011a"),
        content: <SakkaDetails
          detail={detail}
          renders={renderHistory}
          onOpenRound={openSakkaRound}
          onRenderRound={prepareSakkaRoundVideo}
          renderBusy={busy}
        />,
        audit: detail as unknown as JsonObject,
      });
    } catch (error) {
      if (!stillOpen()) return;
      setMessage((error as Error).message);
      setDrawer({
        kind: 'sakka', title, subtitle: t("workflow.m_bdf45938011a"),
        content: <Empty title={t("workflow.sakka_library_load_failed")} body={(error as Error).message} />,
      });
    }
  };

  const openResearchResult = async (item: ResearchResult) => {
    // No drawer shows while a result loads, so the request itself is the intent a later open or close replaces.
    const stillOpen = setDrawer(null);
    setBusy(true); setMessage('');
    try {
      const [detail, ledger, renderPage] = await Promise.all([
        api<ResearchResult>(`/api/v2/research-results/${item.resultCode}`),
        api<{items:JsonObject[]}>(`/api/v2/research-results/${item.resultCode}/decisions`),
        api<{items:RenderJob[]}>(`/api/v2/research-results/${item.resultCode}/renders`)
          .catch(() => ({items: []})),
      ]);
      if (!stillOpen()) return;
      const recipe = await hydrateResearchRecipe(String(detail.recipeCode || ''));
      if (!stillOpen()) return;
      const existingVideoUrl = renderPage.items
        .filter((job) => job.status === 'completed')
        .map((job) => renderMedia(job)?.videoUrl)
        .find((value): value is string => Boolean(value));
      setDrawer({
        kind:'research-result',
        title:`${t("workflow.research_audit_title")} · ${item.resultCode}`,
        subtitle:`${truthLabel[item.truthLabel] || item.truthLabel}${item.classification==='scenario_v4'?'':` · ${number.format(Math.round(item.score))}`}${recipe?.configuration?` · ${t("workflow.research_review_configuration")} ${recipe.configuration.bundleVersion}`:''}`,
        content:<ResearchDecisionAudit
          key={detail.resultCode}
          result={detail}
          recipe={recipe}
          decisions={ledger.items}
          existingVideoUrl={existingVideoUrl}
          onPrepare={(observerSeat,ordinal)=>prepareResearchVideo(detail, observerSeat, ordinal)}
          onVerify={()=>verifyResearchPlay(detail)}
          onOpenExactState={async (observerSeat) => {
            try {
              const exactState = await api<JsonObject>(
                `/api/v2/research-results/${detail.resultCode}/exact-state?observerSeat=${encodeURIComponent(observerSeat)}`,
              );
              const publicState = (exactState.publicState || {}) as JsonObject;
              const legalActions = Array.isArray(exactState.legalActions) ? exactState.legalActions as JsonObject[] : [];
              const observerHand = Array.isArray((publicState.hands as JsonObject | undefined)?.[observerSeat])
                ? (publicState.hands as JsonObject)[observerSeat] as JsonObject[]
                : [];
              const trickPlays = Array.isArray((publicState.currentTrick as JsonObject | undefined)?.plays)
                ? (publicState.currentTrick as JsonObject).plays as JsonObject[]
                : [];
              setDrawer({
                kind:'research-exact-state',
                title:t('workflow.research_scenario_exact_state_derived'),
                subtitle:String(exactState.exactStateId || detail.resultCode),
                content:<><p className="drawer-lead">{t('workflow.research_scenario_actor_only')}</p><div className="drawer-stats"><Stat label={t('workflow.research_scenario_exact_state_phase')} value={String(exactState.phase || '—')}/><Stat label={t('workflow.research_scenario_next_actor')} value={seatLabel[String(exactState.nextActor)] || String(exactState.nextActor || '—')}/><Stat label={t('workflow.research_label_contract')} value={contractLabel[String(publicState.contract)] || String(publicState.contract || '—')}/><Stat label={t('workflow.research_label_buyer')} value={seatLabel[String(publicState.buyer)] || String(publicState.buyer || '—')}/></div><div className="wide decision-context"><b>{t('workflow.research_scenario_observer_hand')}</b><div className="decision-action-chips">{observerHand.map((card, index)=><span dir="ltr" key={`${actionLabel(card)}-${index}`}>{actionLabel(card)}</span>)}</div>{trickPlays.length > 0 && <><b>{t('workflow.research_scenario_current_trick')}</b><div className="decision-action-chips">{trickPlays.map((play, index)=><span dir="ltr" key={`${String(play.seat)}-${index}`}>{seatLabel[String(play.seat)] || String(play.seat)} · {actionLabel(play.card as JsonObject)}</span>)}</div></>}<b>{t('workflow.research_legal_choices')}</b><div className="decision-action-chips">{legalActions.map((action, index)=><span dir="ltr" key={`${actionLabel(action)}-${index}`}>{actionLabel(action)}</span>)}</div></div></>,
                audit:exactState,
              });
            } catch (error) {
              setMessage((error as Error).message);
            }
          }}
          loadPerspectives={loadResearchPerspectives}
        />,
        audit:detail as unknown as JsonObject,
      });
    } catch (error) {
      if (stillOpen()) setMessage(researchRecipeErrorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const openUnifiedLaffa = async (item: UnifiedLaffa) => {
    const cachedRelated = renderHistory.filter((job) => renderJobReferencesUnifiedLaffa(job, item));
    const relatedRequest = api<{items:RenderJob[]}>(`/api/v2/laffat/${encodeURIComponent(item.sourceLocator.code)}/renders`)
      .then((page) => Array.from(new Map([...cachedRelated, ...page.items].map((job) => [job.jobId, job])).values()))
      .catch(() => cachedRelated);
    const detailRequest = item.buyer && item.dealer ? Promise.resolve(item)
      : api<UnifiedLaffa>(`/api/v2/laffat/${encodeURIComponent(item.sourceLocator.code)}`)
        .catch(() => item);
    const stillOpen = setDrawer({
      kind:'unified-laffa',
      title:t("workflow.laffa_library_details_title"),
      subtitle:item.humanCode,
      content:<p className="drawer-lead">{t("workflow.sakka_library_loading")}</p>,
    });
    try {
      const [exactSource, related, detail] = await Promise.all([
        api<JsonObject>(item.sourceLocator.uri), relatedRequest, detailRequest,
      ]);
      if (!stillOpen()) return;
      setDrawer({
        kind:'unified-laffa',
        title:t("workflow.laffa_library_details_title"),
        subtitle:item.humanCode,
        content:<UnifiedLaffaDetails item={detail} exactSource={exactSource} renders={related} busy={busy} onPrepare={prepareUnifiedLaffaVideo} />,
        audit:{projection:detail as unknown as JsonValue, exactSource},
      });
    } catch (error) {
      const [related, detail] = await Promise.all([relatedRequest, detailRequest]);
      if (!stillOpen()) return;
      setDrawer({
        kind:'unified-laffa',
        title:t("workflow.laffa_library_details_title"),
        subtitle:item.humanCode,
        content:<UnifiedLaffaDetails item={detail} renders={related} busy={busy} onPrepare={prepareUnifiedLaffaVideo} />,
        audit:detail as unknown as JsonObject,
      });
      setMessage((error as Error).message);
    }
  };

  useEffect(() => {
    if (view !== 'laffat') return;
    const code = new URLSearchParams(window.location.search).get('result');
    if (!code || deepLinkedResultOpened.current === code) return;
    deepLinkedResultOpened.current = code;
    api<ResearchResult>(`/api/v2/research-results/${encodeURIComponent(code)}`)
      .then(openResearchResult)
      .catch((error: Error) => setMessage(error.message));
  }, [view]);

  useEffect(() => {
    if (view !== 'laffat') return;
    const code = new URLSearchParams(window.location.search).get('laffa');
    if (!code || deepLinkedLaffaOpened.current === code) return;
    const item = laffaPage?.items.find((candidate) => candidate.humanCode === code || candidate.sourceReferences.some((source) => source.code === code));
    if (item) {
      deepLinkedLaffaOpened.current = code;
      openUnifiedLaffa(item);
      return;
    }
    deepLinkedLaffaOpened.current = code;
    api<UnifiedLaffaPage>(`/api/v2/laffat?search=${encodeURIComponent(code)}&limit=50`).then((resolved) => {
      const match = resolved.items.find((candidate) => candidate.humanCode === code || candidate.sourceReferences.some((source) => source.code === code));
      if (!match) throw new Error(t("workflow.laffa_library_no_matches"));
      openUnifiedLaffa(match);
    }).catch((error:Error) => setMessage(error.message));
  }, [laffaPage, renderHistory, view]);

  const cancelResearchRun = async (item: ResearchRun) => {
    setBusy(true); setMessage('');
    try {
      await api<ResearchRun>(`/api/v2/research-runs/${item.runCode}:cancel`, {method:'POST'});
      setDrawer(null); setMessage(t("workflow.research_run_cancelled")); await refresh();
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const retryResearchRun = async (item: ResearchRun) => {
    setBusy(true); setMessage('');
    try {
      await api<ResearchRun>('/api/v2/research-runs', {
        method:'POST',
        headers:{'Idempotency-Key':`research-retry-${item.runCode}-${Date.now()}`},
        body:JSON.stringify({recipeCode:item.recipeCode}),
      });
      setDrawer(null); setMessage(t("workflow.research_run_retried")); await refresh();
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const openResearchRun = async (item: ResearchRun) => {
    const stillOpen = setDrawer({
      kind:'research-run',
      title:`${t("workflow.research_run_details")} · ${item.runCode}`,
      subtitle:researchRunStatusLabel(item),
      content:<><p className="drawer-lead">{t("workflow.sakka_library_loading")}</p>{(item.status === 'queued' || item.status === 'running') && <div className="research-run-actions"><button type="button" className="danger" disabled={busy} onClick={() => void cancelResearchRun(item)}>{t('workflow.research_run_cancel')}</button></div>}</>,
      audit:item as unknown as JsonObject,
    });
    try {
      const [current, stats, recipe] = await Promise.all([
        api<ResearchRun>(`/api/v2/research-runs/${item.runCode}`),
        api<ResearchStats>(`/api/v2/research-runs/${item.runCode}/stats`),
        hydrateResearchRecipe(item.recipeCode),
      ]);
      if (!stillOpen()) return;
      setDrawer({
        kind:'research-run', title:`${t("workflow.research_run_details")} · ${current.runCode}`,
        subtitle:researchRunStatusLabel(current),
        content:<ResearchRunDetails key={current.runCode} run={current} recipe={recipe} stats={stats} busy={busy} onCancel={cancelResearchRun} onRetry={retryResearchRun} onResults={()=>{setDrawer(null);selectView('research');}} loadSequentialSearchReceipts={()=>api<ResearchSequentialSearchReceiptPage>(`/api/v2/research-runs/${encodeURIComponent(current.runCode)}/sequential-search-receipts?offset=0&limit=25`)} loadRun={()=>api<ResearchRun>(`/api/v2/research-runs/${encodeURIComponent(current.runCode)}`)} loadStats={()=>api<ResearchStats>(`/api/v2/research-runs/${encodeURIComponent(current.runCode)}/stats`)} onSnapshot={(nextRun, nextStats)=>setDrawer((open)=>open?.kind === 'research-run' ? {...open, subtitle:researchRunStatusLabel(nextRun), audit:{run:nextRun as unknown as JsonValue,stats:nextStats ?? (open.audit?.stats as JsonValue)}} : open)} PerformanceSummary={ResearchPerformanceSummary} />,
        audit:{run:current as unknown as JsonValue,stats},
      });
    } catch(error) { if (stillOpen()) setMessage(researchRecipeErrorMessage(error)); }
  };

  const openResearchDisagreement = (item: SemanticDisagreement) => {
    setDrawer({
      kind:'research-disagreement',
      title:`${t("workflow.research_quarantine_details")} · ${item.disagreementId}`,
      subtitle:disagreementLabel[item.effectiveCategory||item.category] || t("workflow.research_quarantine_title"),
      content:<ResearchDisagreementAudit item={item} />,
      audit:item as unknown as JsonObject,
    });
  };

  return <div className="app-shell">
    <header className="topbar">
      <div><span className="eyebrow">BalootX</span><h1>{t("workflow.m_7c3c6832cf52")}</h1><p>{t("workflow.m_00328a59e43b")}</p></div>
      <div className="top-actions">
        <a className="guide-link" href="/guide">{t("workflow.label_public_guide")}</a>
        <div className="runtime-badge"><i />{t("workflow.m_7b8ec1caafbd")}<b dir="ltr">{profile?.engineVersion ?? '—'}</b></div>
      </div>
    </header>
    {message && <div className="notice" role="status"><span>{message}</span><button onClick={() => setMessage('')} aria-label={t("workflow.m_ca90c297b099")}>×</button></div>}

    <div className="foundry-layout">
      <aside className="foundry-sidebar">
        <div className="sidebar-heading"><span>{t("workflow.workspace_nav_kicker")}</span><b>{t("workflow.workspace_nav_title")}</b><small>{t("workflow.workspace_nav_help")}</small></div>
        <nav className="workspace-nav" aria-label={t("workflow.workspace_nav_title")}>
          <button className={view === 'overview' ? 'active' : ''} onClick={() => selectView('overview')}><i>01</i><span><b>{t("workflow.nav_overview")}</b><small>{t("workflow.nav_overview_help")}</small></span></button>
          <button className={view === 'research' ? 'active' : ''} onClick={() => selectView('research')}><i>02</i><span><b>{t("workflow.nav_research")}</b><small>{t("workflow.nav_research_help")}</small></span></button>
          <button className={view === 'generation' ? 'active' : ''} onClick={() => selectView('generation')}><i>03</i><span><b>{t("workflow.nav_generation")}</b><small>{t("workflow.nav_generation_help")}</small></span></button>
          <button className={view === 'laffat' ? 'active' : ''} onClick={() => selectView('laffat')}><i>04</i><span><b>{t("workflow.nav_laffat")}</b><small>{t("workflow.nav_laffat_help")}</small></span></button>
          <button className={view === 'sakkas' ? 'active' : ''} onClick={() => selectView('sakkas')}><i>05</i><span><b>{t("workflow.nav_sakkas")}</b><small>{t("workflow.nav_sakkas_help")}</small></span></button>
          <button className={view === 'renders' ? 'active' : ''} onClick={() => selectView('renders')}><i>06</i><span><b>{t("workflow.nav_renders")}</b><small>{t("workflow.nav_renders_help")}</small></span></button>
          <button className={view === 'settings' ? 'active' : ''} onClick={() => selectView('settings')}><i>07</i><span><b>{t("workflow.nav_settings")}</b><small>{t("workflow.nav_settings_help")}</small></span></button>
        </nav>
        <div className="sidebar-library-status"><span>{t("workflow.sidebar_library_status")}</span><b>{number.format(qualifiedLaffat.length)} / {number.format(effectiveVideoBatchTarget)}</b><small>{t("workflow.sidebar_library_status_help")}</small></div>
      </aside>
      <div className="foundry-workspace">
    {view !== 'settings' ? <main className="workspace-content">
      <section className="workspace-intro">
        <div><span>{workspaceIntro[view].kicker}</span><h2>{workspaceIntro[view].title}</h2><p>{workspaceIntro[view].description}</p></div>
        {view === 'overview' && <button className="primary" onClick={() => selectView('research')}>{t("workflow.research_primary_action")}</button>}
        {view === 'laffat' && <button className="secondary" onClick={() => selectView('generation')}>{t("workflow.laffat_primary_action")}</button>}
        {view === 'renders' && <button className="secondary" onClick={() => selectView('laffat')}>{t("workflow.renders_primary_action")}</button>}
      </section>

      {view === 'overview' && <>
      <section className="journey" aria-label={t("workflow.m_600b77a1ec15")}>
        <JourneyStep index={1} title={t("workflow.m_057c3c22709a")} description={t("workflow.m_409bdc09a276")} state={campaigns.length || sakkas.length ? 'done' : 'active'} help={t("workflow.m_77963bafc884")} example={t("workflow.m_8ddb3247947e")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <JourneyStep index={2} title={t("workflow.m_20001871e6f2")} description={t("workflow.m_8307e9c93c15")} state={analysisJob?.status === 'completed' ? 'done' : completedCampaigns.length ? 'active' : 'locked'} help={t("workflow.m_d243da1215f7")} example={t("workflow.m_02c4cf217206")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <JourneyStep index={3} title={t("workflow.m_78fa587849ba")} description={t("workflow.m_ef107cf0fdf1")} state={candidatePage?.count ? 'done' : analysisJob ? 'active' : 'locked'} help={t("workflow.m_242134f5f1b1")} example={t("workflow.m_e90ac3f278a2")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <JourneyStep index={4} title={t("workflow.m_eae889cdcd70")} description={t("workflow.m_b9ecd181d6b3")} state={prepared || renderJob ? 'done' : candidatePage?.count || sakkas.length ? 'active' : 'locked'} help={t("workflow.m_ea2e89960b11")} example={t("workflow.m_bad48361edcd")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <JourneyStep index={5} title={t("workflow.m_d94d3fee9506")} description={t("workflow.m_a510e3b8ce0a")} state={publications.length ? 'active' : 'locked'} help={t("workflow.m_5dbe2b0c5803")} example={t("workflow.m_c9548a76b9d2")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
      </section>

      {profile?.finalistAgentQualification && <button className={`panel agent-readiness ${profile.finalistAgentQualification.expertGenerationAllowed ? 'qualified' : 'local'}`} onClick={() => setDrawer({
        kind:'agent-qualification',
        title:t("workflow.label_agent_details"),
        subtitle:profile.finalistAgentQualification.expertGenerationAllowed ? t("workflow.label_agent_ready_release") : t("workflow.label_agent_ready_local"),
        content:<AgentQualificationDetails value={profile.finalistAgentQualification} pilot={profile.finalistAgentPilotEvidence} />,
        audit:{qualification:profile.finalistAgentQualification, localPilot:profile.finalistAgentPilotEvidence} as unknown as JsonObject,
      })}>
        <div><span className="eyebrow">{t("workflow.label_agent_title")}</span><h2>{profile.finalistAgentQualification.expertGenerationAllowed ? t("workflow.label_agent_ready_release") : t("workflow.label_agent_ready_local")}</h2><small>{profile.finalistAgentQualification.expertGenerationAllowed ? t("workflow.label_agent_ready_release") : t("workflow.label_agent_pending")}</small></div>
        <div className="agent-badges"><span>✓ {t("workflow.label_agent_equal")}</span><span>✓ {t("workflow.label_agent_search")}</span>{profile.finalistAgentPilotEvidence?.passed && <span>✓ {t("workflow.label_agent_local_pilot")}</span>}<InfoBadge id="agent-readiness" text={t("workflow.label_agent_help")} example={t("workflow.label_agent_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} /></div>
      </button>}

      <section className="metrics" aria-label={t("workflow.m_9a21bfcc41a4")}>
        <Metric label={t("workflow.m_deb6cb326493")} value={campaigns.length} help={t("workflow.m_d00bd4ce79ab")} example={t("workflow.m_abb954f9fbf9")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <Metric label={t("workflow.m_5047cdc834ba")} value={completedCampaigns.length} tone="good" help={t("workflow.m_6f58ff62d7d8")} example={t("workflow.m_fdea32cfd803")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <Metric label={t("workflow.m_b01f402f7a7b")} value={evidenceCount} tone="gold" help={t("workflow.m_88ae5326e2a6")} example={t("workflow.m_11678e67b93e")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <Metric label={t("workflow.m_c91806e1b555")} value={sakkas.length} tone="gold" help={t("workflow.m_e8a0f102dd93")} example={t("workflow.m_1da0e6247fda")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
      </section>
      {researchStats&&<ResearchPerformanceSummary stats={researchStats} onRuns={()=>selectView('research')} onResults={()=>selectView('laffat')} />}
      </>}

      {view === 'research' && <><ResearchStudio
        bundles={researchBundles}
        runtimeBundleRevisionId={profile?.runtimeBundleRevisionId || null}
        setDrawer={setDrawer}
        recipes={researchRecipes}
        runs={researchRuns}
        disagreements={semanticDisagreements}
        stats={researchStats ?? null}
        busy={busy}
        setBusy={setBusy}
        setMessage={setMessage}
        refresh={refresh}
        request={api}
        createRecipeAndRun={(payload)=>researchRecipeController.createRecipeAndRun(payload)}
        onOpenResult={openResearchResult}
        onOpenRun={openResearchRun}
        onOpenDisagreement={openResearchDisagreement}
      />
      <ResearchRecipeLedger
        recipes={researchRecipes}
        loadingMore={researchRecipeSnapshot.loading}
        nextCursor={researchRecipeSnapshot.nextCursor}
        error={researchRecipeSnapshot.error ? researchRecipeErrorMessage(researchRecipeSnapshot.error) : null}
        success={null}
        onLoadMore={() => researchRecipeController.loadMore().then(() => undefined)}
        onRetire={retireResearchRecipe}
        retiringCodes={retiringRecipeCodes}
      /></>}

      {view === 'generation' && <>
      <section className="panel video-batch-panel" id="video-production-batch">
        <PanelTitle kicker={t("workflow.batch_kicker")} title={t("workflow.batch_title")} help={t("workflow.batch_help")} example={t("workflow.batch_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <div className="video-batch-layout">
          <form className="video-batch-form" onSubmit={createVideoBatch}>
            <div className="target-presets" role="group" aria-label={t("workflow.batch_target_label")}>
              {[5, 10, 20, 50].map((value) => <button key={value} type="button" className={effectiveVideoBatchTarget === value ? 'active' : ''} onClick={() => setVideoBatchTarget(value)} disabled={videoBatchActive || value > (profile?.videoProductionBatchProfile.maximumTarget || 100)}>{number.format(value)}</button>)}
              <label><span>{t("workflow.batch_custom_target")}</span><input aria-label={t("workflow.batch_custom_target")} type="number" min="1" max={profile?.videoProductionBatchProfile.maximumTarget || 100} value={effectiveVideoBatchTarget} disabled={videoBatchActive} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setVideoBatchTarget(Math.max(1, Math.min(profile?.videoProductionBatchProfile.maximumTarget || 100, Math.trunc(value)))); }} /></label>
            </div>
            <div className="batch-summary">
              <span className="quality-badge">{t("workflow.batch_quality_badge")}<InfoBadge id="video-batch-quality" text={t("workflow.batch_quality_help")} example={t("workflow.batch_quality_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} /></span>
                <b>{t("workflow.batch_summary_prefix")}{number.format(effectiveVideoBatchTarget)}{t("workflow.batch_summary_suffix")}</b>
              <small>{t("workflow.batch_existing_prefix")}{number.format(qualifiedLaffat.length)}{t("workflow.batch_remaining_prefix")}{number.format(qualifiedTargetGap)}</small>
              <small>{t("workflow.batch_budget_prefix")}{number.format((profile?.videoProductionBatchProfile.waveSeedCount || 0) * (profile?.videoProductionBatchProfile.maximumWaves || 0))}{t("workflow.batch_budget_suffix")}</small>
            </div>
            <div className="batch-capacity">
              <span><small>{t("workflow.batch_capacity_search")}</small><b>{number.format(profile?.videoProductionBatchProfile.simulationWorkerCount || 0)}</b></span>
              <span><small>{t("workflow.batch_capacity_qualification")}</small><b>{number.format(profile?.videoProductionBatchProfile.finalistConcurrency || 1)}</b></span>
              <span><small>{t("workflow.batch_capacity_render")}</small><b>{profile?.videoProductionBatchProfile.renderPolicy === 'automatic' ? t("workflow.batch_render_policy_automatic") : t("workflow.batch_render_policy_on_demand")}</b></span>
              <InfoBadge id="video-batch-capacity" text={t("workflow.batch_capacity_help")} example={t("workflow.batch_capacity_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
            </div>
            <small>{t("workflow.batch_no_automatic_render")}</small>
            <button className="primary batch-primary" disabled={busy || videoBatchActive || !profile || qualifiedTargetGap === 0}>{qualifiedTargetGap === 0 ? t("workflow.batch_target_reached") : t("workflow.batch_start_prefix") + number.format(qualifiedTargetGap) + t("workflow.batch_start_suffix")}</button>
          </form>
          <div className="video-batch-progress">
            {videoBatch ? <>
              <div className="batch-stage"><span className={`status-dot ${videoBatch.status}`} /><div><small>{t("workflow.batch_current_stage")}</small><b>{videoBatchStageLabel[videoBatchStage] || statusLabel[videoBatch.status] || videoBatch.status}</b></div><button className="secondary" onClick={() => openVideoBatch()}>{t("workflow.batch_open_details")}</button></div>
              <div className="batch-metrics">
                  <Stat label={t("workflow.batch_required")} value={number.format(Number(videoBatch.request.targetVideos || effectiveVideoBatchTarget))} />
                <Stat label={t("workflow.batch_qualified")} value={number.format(Number(videoBatchResult.finalistsAccepted || 0))} />
                <Stat label={t("workflow.batch_prepared")} value={number.format(Number(videoBatchResult.preparedVideoCount || 0))} />
                <Stat label={t("workflow.batch_ready")} value={number.format(Number(videoBatchResult.completedVideoCount || 0))} />
                <Stat label={t("workflow.batch_rejected")} value={number.format(Number(videoBatchResult.finalistsRejected || 0))} />
              </div>
              <div className="batch-live-progress">
                <span>{t("workflow.batch_stage_progress")}<b>{number.format(Math.round(videoBatchStageProgress * 100))}{t("workflow.batch_percent_suffix")}</b></span>
                <span>{t("workflow.batch_active_workers")}<b>{number.format(videoBatchActiveWorkers)}</b></span>
                {videoBatchCurrentWave > 0 && <span>{t("workflow.batch_wave_prefix")}<b>{number.format(videoBatchCurrentWave)}</b></span>}
              </div>
              <div className="batch-progress-track" aria-label={t("workflow.batch_stage_progress")}><i style={{transform:`scaleX(${videoBatchStageProgress})`}} /></div>
              <small>{t("workflow.batch_rounds_prefix")}{number.format(Number(videoBatchResult.roundsGenerated || 0))}{t("workflow.batch_rounds_suffix")}</small>
              {videoBatch.status === 'exhausted' && <p className="batch-warning">{t("workflow.batch_exhausted_help")}</p>}
              {videoBatch.error && <p className="batch-warning">{friendlyJobError(videoBatch.error)}</p>}
              {videoBatchActive && <button className="quiet-danger" type="button" disabled={busy} onClick={cancelVideoBatch}>{t("workflow.batch_cancel")}</button>}
              {videoBatchHistory.length > 1 && <div className="batch-history"><small>{t("workflow.batch_previous")}</small>{videoBatchHistory.slice(0, 4).map((item, index) => <button type="button" key={item.jobId} className={item.jobId === videoBatch.jobId ? 'active' : ''} onClick={() => setVideoBatch(item)}>{t("workflow.batch_number_prefix")}{number.format(index + 1)}</button>)}</div>}
            </> : <Empty title={t("workflow.batch_empty_title")} body={t("workflow.batch_empty_help")} />}
          </div>
        </div>
      </section>
      </>}

      {view === 'laffat' && <>
      <section className="panel laffa-library-panel" id="laffa-library">
        <PanelTitle kicker={t("workflow.video_library_source_laffa")} title={t("workflow.laffa_library_all_generated_title")} help={t("workflow.laffa_library_all_generated_help")} example={t("workflow.batch_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <UnifiedLaffaLibrary
          page={laffaPage}
          filters={laffaFilters}
          loading={laffaLoading}
          busy={busy}
          onFilters={setLaffaFilters}
          onOpen={openUnifiedLaffa}
          onPrepare={prepareUnifiedLaffaVideo}
        />
      </section>
      </>}

      {view === 'renders' && <>
      {(activeRenders.length > 0 || renderJob) && <section className="panel" id="selected-render">
        <PanelTitle kicker={t("workflow.batch_capacity_render")} title={t("workflow.m_67149f34df53")} help={t("workflow.label_video_library_help")} example={t("workflow.label_video_library_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        {(activeRenders.length > 0 ? activeRenders : renderJob ? [renderJob] : []).map((job) => <JobProgress key={job.jobId} title={String(renderSource(job)?.code || t("workflow.m_67149f34df53"))} status={job.status} progress={job.progress ?? 0} error={job.error} onDetails={() => { setRenderJob(job); setDrawer({kind:'render', title:t("workflow.m_67149f34df53"), subtitle:statusLabel[job.status] || job.status, content:<JsonSummary value={job.result || {status:job.status} as JsonObject} />, audit:job as unknown as JsonObject}); }} />)}
        {activeRenders.length === 0 && renderJob?.status === 'completed' && <RenderOutputCard job={renderJob} />}
      </section>}
      </>}

      {view === 'generation' && <>
      <details className="advanced-workflow">
        <summary>{t("workflow.batch_advanced_title")}<small>{t("workflow.batch_advanced_help")}</small></summary>
      <section className="workspace-grid">
        <article className="panel action-panel">
          <PanelTitle kicker={t("workflow.m_fdeae09dd0e8")} title={t("workflow.m_ffbc2fa893c5")} help={t("workflow.m_d2dbe4573a8b")} example={t("workflow.m_4ca43b0ab53b")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
          <div className="choice-tabs"><button className="active">{t("workflow.m_a87c092f031d")}</button><button onClick={() => document.getElementById('sakka-form')?.scrollIntoView({behavior: 'smooth'})}>{t("workflow.m_cdcb98d0e114")}</button></div>
          <form onSubmit={createCampaign} className="friendly-form">
            <FieldLabel label={t("workflow.campaign_lane_label")} help={t("workflow.campaign_lane_help")} example={t("workflow.campaign_lane_example")} id="campaign-lane" openHelp={openHelp} setOpenHelp={setOpenHelp}><select id="campaign-lane" value={generationLane} onChange={(event) => setGenerationLane(event.target.value as typeof generationLane)}><option value="guided">{sourceLabel.guided}</option><option value="organic">{sourceLabel.organic}</option><option value="disagreement">{sourceLabel.disagreement}</option></select></FieldLabel>
            <FieldLabel label={t("workflow.label_seed_start")} help={t("workflow.label_seed_start_help")} example={t("workflow.label_seed_start_example")} id="seed-start" openHelp={openHelp} setOpenHelp={setOpenHelp}><input id="seed-start" type="number" min="0" max="9007199254740991" value={seedStart} onChange={(event) => setSeedStart(Number(event.target.value))} /></FieldLabel>
            <FieldLabel label={t("workflow.m_ab66bff2c50f")} help={t("workflow.m_2b0f2a0067ab")} example={t("workflow.m_9722668f1ae8")} id="seed-count" openHelp={openHelp} setOpenHelp={setOpenHelp}><input id="seed-count" type="number" min="1" max="1000000000" value={seedCount} onChange={(event) => setSeedCount(Number(event.target.value))} /></FieldLabel>
            <FieldLabel label={t("workflow.m_3791227cd807")} help={t("workflow.m_f08f5758c217")} example={t("workflow.m_527d1ed23a38")} id="shard-size" openHelp={openHelp} setOpenHelp={setOpenHelp}><input id="shard-size" type="number" min="1" max="1000000" value={seedsPerShard} onChange={(event) => setSeedsPerShard(Number(event.target.value))} /></FieldLabel>
            <div className="form-summary"><span>{t("workflow.m_5fd9563e6846")}</span><b>{sourceLabel[generationLane]}</b><small>{generationLane === 'guided' ? t("workflow.campaign_lane_guided_summary") : generationLane === 'disagreement' ? t("workflow.campaign_lane_disagreement_summary") : t("workflow.campaign_lane_organic_summary")}</small></div>
            <button className="primary" disabled={busy || !profile}>{t("workflow.m_0906a5723876")}</button>
          </form>
        </article>

        <article className="panel activity-panel">
          <PanelTitle kicker={t("workflow.m_1796ea83af03")} title={t("workflow.m_d2ff97c3223d")} help={t("workflow.m_b39f9bbcb1af")} example={t("workflow.m_bbf806b5cb3a")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
          <div className="friendly-list">
            {campaigns.slice(0, 8).map((campaign) => <button key={campaign.campaignHash} className="list-item" onClick={() => openCampaign(campaign)}>
              <span className={`status-dot ${campaign.status}`} /><span className="list-copy"><b>{campaignNames.get(campaign.campaignHash)}</b><small>{statusLabel[campaign.status]}{t("workflow.m_fcb4a40c37ab")}{number.format(Object.values(campaign.shardCounts).reduce((a, b) => a + b, 0))}</small></span><span className="chevron">‹</span>
            </button>)}
            {!campaigns.length && <Empty title={t("workflow.m_2c2aa64421ac")} body={t("workflow.m_ad439b871e47")} />}
          </div>
          {selected && <div className="next-action"><div><span>{t("workflow.m_e161f8a04285")}</span><b>{selected.status === 'completed'
            ? !selectedCompatible
              ? t("workflow.label_campaign_incompatible")
              : selectedCandidateCount > 0
              ? (t("workflow.m_719562923a1d") + campaignNames.get(selected.campaignHash) + "")
              : t("workflow.m_b072e0c76576")
            : t("workflow.m_0b32761a9a6c")}</b></div><button className="secondary" disabled={busy || selected.status !== 'completed' || selectedCandidateCount === 0 || !selectedCompatible} onClick={() => startAnalysis()}>{t("workflow.m_a41b9e954b59")}</button></div>}
          {analysisJob && <JobProgress title={t("workflow.m_932cad2d200c")} status={analysisJob.status} progress={analysisJob.progress ?? (analysisJob.status === 'completed' ? 1 : 0.25)} error={analysisJob.error} onDetails={() => setDrawer({kind:'analysis', title:t("workflow.m_24e7b7d3b1cf"), subtitle:statusLabel[analysisJob.status] || analysisJob.status, content:<JsonSummary value={analysisJob.result || {status: analysisJob.status}} />, audit:analysisJob as unknown as JsonObject})} />}
        </article>
      </section>
      </details>

      {(analysisJob || candidatePage) && <section className="panel candidate-panel" id="video-workflow">
        <PanelTitle kicker={t("workflow.m_75b402a339ea")} title={t("workflow.m_d9cd7e15e585")} help={t("workflow.m_2d5af58b9fe7")} example={t("workflow.m_45a761434e78")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        {analysisJob && analysisJob.status !== 'completed' && <JobProgress title={t("workflow.m_5a3fe600e182")} status={analysisJob.status} progress={analysisJob.progress ?? 0} error={analysisJob.error} onDetails={() => setDrawer({kind:'analysis', title:t("workflow.m_24e7b7d3b1cf"), subtitle:statusLabel[analysisJob.status] || analysisJob.status, content:<JsonSummary value={analysisJob.result || {status: analysisJob.status}} />, audit:analysisJob as unknown as JsonObject})} />}
        <div className="candidate-grid">
          {visibleCandidates.map((candidate, index) => <article className="candidate-card" data-candidate-code={candidate.shortCode} key={candidate.candidateHash}>
            <button className="candidate-open" onClick={() => openCandidate(candidate)}>
              <span className="candidate-code">{t("workflow.m_20b8a9eab31e")}{number.format(index + 1)}</span>
              <span className="candidate-score"><b>{number.format(Math.round(candidate.selectionScore))}</b><small>{t("evidence.label_decision_selection_score")}</small></span>
              <h3>{decisionLabel[candidate.decisionKind] || t("workflow.m_b587409de388")}{' '}{t("workflow.m_4d7d679f7b4c")}{' '}{seatLabel[candidate.observerSeat] || candidate.observerSeat}</h3>
              <p>{laffaLabel[candidate.laffaClassification] || t("workflow.m_10db9ea7d029")}</p>
              <div className="candidate-badges"><span>{truthLabel[candidate.truthLabel] || candidate.truthLabel}</span><span>{t("workflow.m_aaf0eb9937a4")}{number.format(Math.round(candidate.confidence * 100))}{t("workflow.m_a489ae36a167")}</span><span>{number.format(candidate.alternativeCount)}{t("workflow.m_9778d8739f4e")}</span>{Boolean(candidate.compatibilityBlockers?.length) && <span>{t("workflow.label_incompatible_candidate")}</span>}</div>
            </button>
            {candidate.compatibilityBlockers?.length
              ? <button className="secondary candidate-prepare" disabled>{t("workflow.label_incompatible_candidate")}</button>
              : candidate.playReady
              ? <button className="secondary candidate-prepare" data-testid={`prepare-${candidate.shortCode}`} disabled={busy} onClick={() => prepareCandidate(candidate)}>{t("workflow.m_2e704a90eb65")}</button>
              : candidate.finalistValidationRequired
                ? <button className="secondary candidate-prepare" data-testid={`validate-${candidate.shortCode}`} disabled={busy || ['queued','running'].includes(finalistJob?.status || '')} onClick={() => validateFinalist(candidate)}>{t("workflow.m_7006ae69289a")}</button>
                : <button className="secondary candidate-prepare" disabled>{t("workflow.m_2daedc79920d")}</button>}
          </article>)}
        </div>
        {analysisJob?.status === 'completed' && candidatePage?.count === 0 && <Empty title={t("workflow.m_acb5ecccb1e5")} body={t("workflow.m_50d6eed95216")} />}
        {prepared && <div className="prepared-strip"><div><span>{prepared.exportPreset === 'hooked_flagship' ? t("workflow.m_6bb067528380") : t("workflow.m_c21e98a53a03")}</span><b dir="ltr">{String(prepared.locator.code || prepared.candidate.shortCode)}</b><small>{t("workflow.m_f8f00b7c0856")}</small></div><button className="primary" disabled={busy || ['queued','running'].includes(renderJob?.status || '')} onClick={renderPrepared}>{t("workflow.m_c9a6a6f427bc")}</button></div>}
        {finalistJob && <JobProgress title={t("workflow.m_799b88821ede")} status={finalistJob.status} progress={finalistJob.progress} error={finalistJob.error} onDetails={() => setDrawer({kind:'finalist', title:t("workflow.m_f85419b56370"), subtitle:statusLabel[finalistJob.status] || finalistJob.status, content:<FinalistSummary job={finalistJob} />, audit:finalistJob as unknown as JsonObject})} />}
        {finalistJob?.status === 'completed' && finalistJob.result?.localPreviewReady && !prepared && !renderJob && <div className="next-action"><div><span>{t("workflow.m_e161f8a04285")}</span><b>{t("workflow.m_f5a092f78591")}</b></div><div className="action-row"><button className="primary" disabled={busy} onClick={() => prepareBestFinalist('hooked_flagship')}>{t("workflow.m_47fbde86f74b")}</button><button className="secondary" disabled={busy} onClick={() => prepareBestFinalist('decision_clip')}>{t("workflow.m_7bf27642bb21")}</button></div></div>}
      </section>}
      </>}

      {view === 'renders' && <>
      <section className="panel video-library-panel" id="video-library">
        <PanelTitle kicker={t("workflow.label_ready_videos")} title={t("workflow.label_video_library")} help={t("workflow.label_video_library_help")} example={t("workflow.label_video_library_example")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <div className="video-library-grid">
          {completedRenders.map((job) => <RecentRenderCard key={job.jobId} job={job} />)}
        </div>
        {!completedRenders.length && <Empty title={t("workflow.label_no_videos")} body={t("workflow.label_no_videos_help")} />}
      </section>
      </>}

      {view === 'sakkas' && <>
      <section className="workspace-grid" id="sakka-form">
        <article className="panel action-panel">
          <PanelTitle kicker={t("settings.m_cdcb98d0e114")} title={t("settings.m_c828e12fe80b")} help={t("settings.m_2a37e72ed876")} example={t("settings.m_b85e286485f8")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
          <form onSubmit={generateNaturalSakka} className="friendly-form">
            <FieldLabel label={t("settings.m_75e926dce2a5")} help={t("settings.m_ea115d2051b5")} example={t("settings.m_5e2ad711d29a")} id="generation-seed" openHelp={openHelp} setOpenHelp={setOpenHelp}><input id="generation-seed" aria-label={t("settings.m_75e926dce2a5")} type="number" min="0" max={Number.MAX_SAFE_INTEGER} value={generationSeed} onChange={(event) => setGenerationSeed(Number(event.target.value))} /></FieldLabel>
            <FieldLabel label={t("settings.m_3beb623be0f6")} help={t("settings.m_44dfca38ea10")} example={t("settings.m_dc0b53bea612")} id="dealer" openHelp={openHelp} setOpenHelp={setOpenHelp}><select id="dealer" aria-label={t("settings.m_3beb623be0f6")} value={generationDealer} onChange={(event) => setGenerationDealer(event.target.value as typeof generationDealer)}><option value="east">{t("settings.m_26b73a5271c1")}</option><option value="south">{t("settings.m_390e936c0d6c")}</option><option value="west">{t("settings.m_64141a180741")}</option><option value="north">{t("settings.m_5cbbfccea6e8")}</option></select></FieldLabel>
            <div className="form-summary"><span>{t("settings.m_c9704be2c671")}</span><b>{number.format(profile?.sakkaGenerationProfile.target ?? 152)}{t("settings.m_3a243313fea1")}</b><small>{t("settings.m_9651056d93a0")}{number.format(profile?.sakkaGenerationProfile.maximumRounds ?? 0)}{t("settings.m_492f28c8acc3")}</small></div>
            <button className="primary" disabled={busy || !profile || ['queued', 'running'].includes(generationJob?.status ?? '')}>{t("settings.m_69dcebee5c36")}</button>
          </form>
        </article>
        <article className="panel library-panel">
          <PanelTitle kicker={t("settings.m_9ff4670ff446")} title={t("settings.m_201dd8755384")} help={t("settings.m_b9c9ada83023")} example={t("settings.m_3396882f420a")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
          {generationJob && <JobProgress title={t("settings.m_1f4e5405fa91")} status={generationJob.status} progress={generationJob.progress} error={generationJob.error} onDetails={() => setDrawer({kind:'generation', title:t("settings.m_719d415b4598"), subtitle:statusLabel[generationJob.status], content:<JsonSummary value={(generationJob.result || {status:generationJob.status}) as unknown as JsonObject} />, audit:generationJob as unknown as JsonObject})} />}
          <SakkaLibraryTable items={sakkas} names={sakkaNames} onOpen={openSakka} />
        </article>
      </section>
      </>}

      {view === 'overview' && <>
      <section className="panel readiness-panel">
        <PanelTitle kicker={t("settings.m_4dd48409b984")} title={t("settings.m_77cbd53db867")} help={t("settings.m_e706bdf84419")} example={t("settings.m_565d95bfb57e")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
        <div className="readiness-grid" aria-busy={readinessLoading}>{readiness.map((item) => <button key={item.label} className={`readiness-item ${item.ready ? 'ready' : 'pending'}`} disabled={readinessLoading} onClick={() => setDrawer({kind:'readiness', title:item.label, subtitle:item.ready ? t("settings.m_c2da5684d63b") : t("settings.m_8398aec9e029"), content:<p className="drawer-lead">{item.detail}</p>})}><i>{item.ready ? '✓' : '○'}</i><span><b>{item.label}</b><small>{item.detail}</small></span><span className="chevron">‹</span></button>)}</div>
        <div className="publishing-lock"><span>🔒</span><div><b>{t("settings.m_2382f4a3fe95")}</b><small>{t("settings.m_12c8c831be2c")}</small></div><a href={renderJob?.status === 'completed' ? '#render-output' : '#video-workflow'}>{t("settings.m_94d1f1049135")}</a></div>
      </section>
      </>}
    </main> : <ConfigurationWorkspace runtimeProfile={profile} openHelp={openHelp} setOpenHelp={setOpenHelp} setDrawer={setDrawer} setMessage={setMessage} />}
      </div>
    </div>

    {drawer && <Drawer state={drawer} onClose={() => setDrawer(null)} />}
  </div>;
}

function measured(value:number|null|undefined, digits=2) {
  return value===null||value===undefined?t("workflow.research_metric_unavailable"):number.format(Math.round(value*(10**digits))/(10**digits));
}

function ArchivedLaffaDetails({item,renders,busy,onPrepare}:{item:ArchivedLaffa;renders:RenderJob[];busy:boolean;onPrepare:(item:ArchivedLaffa)=>void}) {
  const code=item.serial||item.gameHash.slice(0,10).toUpperCase();
  const completed=renders.find((job)=>job.status==='completed');
  const media=completed ? renderMedia(completed) : null;
  const active=renders.find((job)=>['queued','running'].includes(job.status));
  return <>
    <div className="drawer-stats">
      <Stat label={t("workflow.laffa_library_column_code")} value={code}/>
      <Stat label={t("workflow.research_label_contract")} value={item.contract ? contractLabel[item.contract]||item.contract : '—'}/>
      <Stat label={t("workflow.research_label_buyer")} value={item.buyer ? seatLabel[item.buyer]||item.buyer : '—'}/>
      <Stat label={t("workflow.sakka_library_column_rounds")} value={item.roundNumber ? number.format(item.roundNumber) : '—'}/>
    </div>
    <p className="drawer-lead">{t("workflow.laffa_library_all_generated_help")}</p>
    <div className="decision-answer"><span>{t("workflow.research_render_purpose")}</span><b>{t("workflow.research_render_research_audit")}</b><small>{t("workflow.batch_no_automatic_render")}</small></div>
    <div className="drawer-actions">
      {media ? <a className="button-link primary" href={media.videoUrl}>{t("workflow.label_open_video")}</a>
        : active ? <span className="status running">{statusLabel[active.status]||active.status}</span>
        : <button type="button" className="primary" disabled={busy} onClick={()=>onPrepare(item)}>{t("workflow.research_render_audit")}</button>}
    </div>
  </>;
}

function ConfigurationWorkspace({runtimeProfile, openHelp, setOpenHelp, setDrawer, setMessage}: {runtimeProfile:FoundryProfile|null; openHelp:string|null; setOpenHelp:(value:string|null)=>void; setDrawer:(value:DrawerState)=>void; setMessage:(value:string)=>void}) {
  const [profiles, setProfiles] = useState<ConfigProfile[]>([]);
  const [bundles, setBundles] = useState<ConfigBundle[]>([]);
  const [compilerCoverage, setCompilerCoverage] = useState<CompilerCoverageSummary | null>(null);
  const [profileSchema, setProfileSchema] = useState<JsonSchema | null>(null);
  const [profileValidation, setProfileValidation] = useState<ProfileValidation | null>(null);
  const [profileCompilation, setProfileCompilation] = useState<BundleCompilation | null>(null);
  const [profileCompilationError, setProfileCompilationError] = useState('');
  const [previewBundle, setPreviewBundle] = useState<ConfigBundle | null>(null);
  const [previewBaseRevision, setPreviewBaseRevision] = useState('');
  const [family, setFamily] = useState('rules');
  const [selected, setSelected] = useState<ConfigProfile | null>(null);
  const [draft, setDraft] = useState<JsonObject | null>(null);
  // A mutation's follow-up read keeps the intent that started the mutation.
  const profileIntent = useRef(0);
  const [saving, setSaving] = useState(false);
  const [query, setQuery] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [compareRevision, setCompareRevision] = useState('');
  const [showAllRevisions, setShowAllRevisions] = useState(false);
  const [transitionReason, setTransitionReason] = useState('');
  const [rollbackRevision, setRollbackRevision] = useState('');
  const load = useCallback(async () => {
    await ensureConfigurationIdentity();
    const [profilePage, bundlePage, capabilities] = await Promise.all([
      api<{items: ConfigProfile[]}>('/api/v1/configuration-registry/profiles?limit=1000'),
      api<{items: ConfigBundle[]}>('/api/v1/configuration-registry/bundles?limit=100'),
      api<DomainCapabilitiesResponse>('/api/v2/domain-capabilities'),
    ]);
    setProfiles(profilePage.items); setBundles(bundlePage.items);
    setCompilerCoverage(capabilities.compilerV3?.compiledDomain?.coverage || null);
  }, []);
  useEffect(() => { load().catch((error:Error) => setMessage(error.message)); }, [load, setMessage]);
  useEffect(() => {
    setProfileSchema(null);
    api<JsonSchema>(`/api/v2/configuration/profile-schemas/${encodeURIComponent(family)}`)
      .then(setProfileSchema)
      .catch((error:Error) => setMessage(error.message));
  }, [family, setMessage]);
  const families = useMemo(() => [...new Set(profiles.map((item) => item.family))].sort((a,b) => (familyMeta[a]?.label || a).localeCompare(familyMeta[b]?.label || b, 'ar')), [profiles]);
  const latest = useMemo(() => profiles.filter((item) => item.family === family && item.status === 'published').sort((a,b) => b.sequence - a.sequence)[0] || profiles.find((item) => item.family === family), [profiles, family]);
  useEffect(() => () => { ++profileIntent.current; }, []);
  const selectFamily = (nextFamily: string) => {
    if (nextFamily === family && !selected) return;
    ++profileIntent.current;
    setFamily(nextFamily); setSelected(null); setDraft(null);
  };
  const openProfile = useCallback((item: ConfigProfile, intent?: number) => openConfigurationRevision(
    profileIntent,
    () => api<ConfigProfile>(`/api/v1/configuration-registry/profiles/${item.revisionId}`),
    (full, nextDraft) => {
      setSelected(full); setDraft(nextDraft); setProfileValidation(null); setProfileCompilation(null); setProfileCompilationError(''); setPreviewBundle(null);
    },
    setMessage,
    intent,
  ), [setMessage]);
  useEffect(() => { if (latest && !selected) openProfile(latest); }, [latest, selected, openProfile]);
  const siblingRevisions = useMemo(() => profiles.filter((item) => selected && item.family === selected.family && item.profileId === selected.profileId && item.revisionId !== selected.revisionId).sort((a,b) => b.sequence - a.sequence), [profiles, selected]);
  const visibleSiblingRevisions = showAllRevisions ? siblingRevisions : siblingRevisions.slice(0, 12);
  useEffect(() => { setCompareRevision(siblingRevisions[0]?.revisionId || ''); }, [selected?.revisionId, siblingRevisions[0]?.revisionId]);
  useEffect(() => { setShowAllRevisions(false); }, [selected?.revisionId]);
  const updateAt = (path: (string | number)[], value: JsonValue) => {
    setProfileValidation(null); setProfileCompilation(null); setProfileCompilationError(''); setPreviewBundle(null);
    setDraft((current) => {
      if (path.length === 0 && value && typeof value === 'object' && !Array.isArray(value)) return copyJson(value);
      const next = copyJson(current || {}); let cursor: any = next;
      for (let index = 0; index < path.length - 1; index += 1) cursor = cursor[path[index]];
      cursor[path[path.length - 1]] = value; return next;
    });
  };
  const changed = useMemo(() => diffPaths((selected?.payload || {}) as JsonValue, (draft || {}) as JsonValue), [selected, draft]);
  const saveDraft = async () => {
    if (!selected || selected.status !== 'draft' || !draft || !changed.length) return; setSaving(true);
    const intent = ++profileIntent.current;
    try {
      const result = await api<ConfigProfile>(`/api/v2/configuration/profiles/${selected.revisionId}`, {method:'PATCH', headers:adminHeaders(), body:JSON.stringify({payload:draft, expectedEtag:selected.etag})});
      if (intent === profileIntent.current) setMessage(t("settings.m_257862e1d0ee"));
      await load(); await openProfile(result, intent);
    } catch (error) { if (intent === profileIntent.current) setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const adminHeaders = () => ({'x-baloot-role':'admin', ...(adminKey ? {'x-api-key':adminKey} : {})});
  const cloneDraft = async () => {
    if (!selected?.payload) return; setSaving(true);
    const intent = ++profileIntent.current;
    try {
      const payload=copyJson(selected.payload); const profile=payload.profile as JsonObject | undefined;
      // Revision identity changes independently; cloning cannot invent support for a new contract version.
      if(profile) profile.status='draft';
      const result=await api<ConfigProfile>('/api/v2/configuration/profiles', {method:'POST', headers:adminHeaders(), body:JSON.stringify({family:selected.family,payload,scope:selected.scope,scopeKey:selected.scopeKey==='*'?null:selected.scopeKey,rolloutPercentage:100,rolloutSeed:'default',supersedesRevisionId:selected.revisionId,publish:false})});
      if (intent === profileIntent.current) setMessage(t("settings.rules_clone_ready"));
      await load(); await openProfile(result, intent);
    } catch(error) { if (intent === profileIntent.current) setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const validateDraft = async () => {
    if (!selected || selected.status !== 'draft' || changed.length) return; setSaving(true);
    try {
      const result=await api<ProfileValidation>(`/api/v2/configuration/profiles/${selected.revisionId}:validate`, {method:'POST',headers:adminHeaders()});
      setProfileValidation(result); setMessage(t("settings.rules_validation_ready"));
    } catch(error) { setProfileValidation(null); setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const compareProfile = async () => {
    if (!selected || !compareRevision) return;
    try {
      const result = await api<JsonObject>(`/api/v1/configuration-registry/profiles:compare?left=${encodeURIComponent(compareRevision)}&right=${encodeURIComponent(selected.revisionId)}`);
      setDrawer({kind:'config-compare', title:t("settings.m_98653333fd6a"), subtitle:`${familyMeta[selected.family]?.label || selected.family}`, content:<ConfigComparison value={result} />, audit:result});
    } catch (error) { setMessage((error as Error).message); }
  };
  const transitionProfile = async (action: 'publish' | 'reject' | 'retire') => {
    if (!selected) return;
    if (action !== 'publish' && transitionReason.trim().length < 2) { setMessage(t("settings.m_5c725248e82f")); return; }
    setSaving(true);
    const intent = ++profileIntent.current;
    try {
      const body = action === 'publish' ? {expectedEtag:selected.etag} : {expectedEtag:selected.etag, reason:transitionReason.trim()};
      const result = await api<ConfigProfile>(`/api/v1/configuration-registry/profiles/${selected.revisionId}:${action}`, {method:'POST', headers:adminHeaders(), body:JSON.stringify(body)});
      if (intent === profileIntent.current) {
        setMessage(action === 'publish' ? t("settings.m_d3cf2f94e36e") : action === 'reject' ? t("settings.m_eabf58cbd764") : t("settings.m_054f2c46d23f"));
        setTransitionReason('');
      }
      await load(); await openProfile(result, intent);
    } catch (error) { if (intent === profileIntent.current) setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const filteredFamilies = families.filter((item) => `${familyMeta[item]?.label || item} ${familyMeta[item]?.description || ''}`.includes(query));
  const runtimeBundle = bundles.find((item) => item.revisionId === runtimeProfile?.runtimeBundleRevisionId);
  const publishedBundle = bundles
    .filter((item) => item.status === 'published' && (!runtimeBundle || item.bundleId === runtimeBundle.bundleId))
    .sort((left,right)=>Number(right.sequence||0)-Number(left.sequence||0))[0] || runtimeBundle;
  const rollbackBundles = useMemo(() => {
    const byVersion = new Map<string, ConfigBundle>();
    bundles
      .filter((item) => ['published','superseded'].includes(item.status) && item.revisionId !== publishedBundle?.revisionId)
      .forEach((item) => { if (!byVersion.has(item.version)) byVersion.set(item.version, item); });
    return [...byVersion.values()].sort((left,right)=>right.version.localeCompare(left.version,undefined,{numeric:true}));
  }, [bundles, publishedBundle?.revisionId]);
  useEffect(()=>setRollbackRevision(rollbackBundles[0]?.revisionId||''),[publishedBundle?.revisionId,rollbackBundles[0]?.revisionId]);
  const bundleDraft = bundles.find((item) => item.status === 'draft' && item.bundleId === publishedBundle?.bundleId && Object.entries(item.componentRevisions).every(([family, revisionId]) => profiles.some((profile) => profile.family === family && profile.revisionId === revisionId && profile.status === 'published')));
  const latestPublishedProfiles = useMemo(() => {
    const result = new Map<string, ConfigProfile>();
    if (!publishedBundle) return result;
    Object.entries(publishedBundle.componentRevisions).forEach(([family, revisionId]) => {
      const selectedProfile = profiles.find((item) => item.revisionId === revisionId);
      const latest = profiles.filter((item) => item.family === family && item.profileId === selectedProfile?.profileId && item.status === 'published').sort((a,b) => b.sequence - a.sequence)[0];
      if (latest) result.set(family, latest);
    });
    return result;
  }, [profiles, publishedBundle]);
  const pendingFamilies = publishedBundle ? Object.keys(publishedBundle.componentRevisions).filter((item) => latestPublishedProfiles.get(item)?.revisionId !== publishedBundle.componentRevisions[item]) : [];
  const restartPending = Boolean(publishedBundle && runtimeBundle && publishedBundle.revisionId !== runtimeBundle.revisionId);
  const missingCoverageTargets = useMemo(() => {
    const counts = new Map<string, number>();
    compilerCoverage?.rows.forEach((row) => row.missingTargets.forEach((target) => counts.set(target, (counts.get(target) || 0) + 1)));
    return [...counts.entries()].sort((left, right) => right[1] - left[1]);
  }, [compilerCoverage]);
  const compileDraftPreview = async () => {
    const base = bundles.find((item) => item.revisionId === previewBaseRevision && item.status === 'published');
    const componentRevisions = configurationPreviewComponents(base, selected);
    if (!selected || !base || !componentRevisions || changed.length || (selected.status === 'draft' && !profileValidation?.valid)) return; setSaving(true);
    try {
      const created=await api<ConfigBundle>('/api/v2/configuration/bundles', {method:'POST',headers:adminHeaders(),body:JSON.stringify({bundleId:base.bundleId,version:nextPatchVersion(base.version),mode:base.mode||'full',componentRevisions,scope:'global',rolloutPercentage:100,rolloutSeed:'default',supersedesRevisionId:base.revisionId,publish:false,allowDraftComponents:selected.status==='draft'})});
      const compilation=await api<BundleCompilation>(`/api/v2/configuration/bundles/${created.revisionId}:compile`, {method:'POST',headers:adminHeaders()});
      setPreviewBundle(created); setProfileCompilation(compilation); setProfileCompilationError(''); setMessage(compilation.publishable?t("settings.rules_compile_ready"):t("settings.rules_compile_blocked")); await load();
    } catch(error) { const message=configurationErrorCopy(error); setProfileCompilation(null); setProfileCompilationError(message); setPreviewBundle(null); setMessage(''); } finally { setSaving(false); }
  };
  const composeBundle = async () => {
    if (!publishedBundle || !pendingFamilies.length) return;
    setSaving(true);
    try {
      const componentRevisions = Object.fromEntries(Object.keys(publishedBundle.componentRevisions).map((item) => [item, latestPublishedProfiles.get(item)?.revisionId || publishedBundle.componentRevisions[item]]));
      const result = await api<ConfigBundle>('/api/v1/configuration-registry/bundles', {method:'POST', headers:adminHeaders(), body:JSON.stringify({
        bundleId:publishedBundle.bundleId, version:nextPatchVersion(publishedBundle.version), mode:publishedBundle.mode || 'full', componentRevisions,
        scope:'global', rolloutPercentage:100, rolloutSeed:'default', supersedesRevisionId:publishedBundle.revisionId, publish:false,
      })});
      setMessage(t("settings.m_9cc38353308a")); await load();
      setDrawer({kind:'bundle-draft', title:t("settings.m_42ce07b1fcee"), subtitle:(t("settings.m_507e5fe82beb") + result.version + ""), content:<ConfigBundleDetails bundle={result} profiles={profiles} />, audit:result as unknown as JsonObject});
    } catch (error) { setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const publishBundle = async (target: ConfigBundle | null = bundleDraft || null) => {
    if (!target || target.status !== 'draft') return;
    setSaving(true);
    try {
      await api<ConfigBundle>(`/api/v1/configuration-registry/bundles/${target.revisionId}:publish`, {method:'POST', headers:adminHeaders(), body:JSON.stringify({expectedEtag:target.etag})});
      setPreviewBundle(null); setProfileCompilation(null);
      setMessage(t("settings.m_4e5fba8d5afd")); await load();
    } catch (error) { setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const rollbackBundle = async (source:ConfigBundle) => {
    setSaving(true);
    try {
      const result=await api<ConfigBundle>(`/api/v1/configuration-registry/bundles/${source.revisionId}:rollback`, {method:'POST',headers:adminHeaders(),body:JSON.stringify({newVersion:nextPatchVersion(publishedBundle?.version||source.version)})});
      setMessage(t("settings.rules_rollback_ready")); await load();
      setDrawer({kind:'bundle-rollback',title:t("settings.rules_rollback_title"),subtitle:t("settings.rules_rollback_help"),content:<ConfigBundleDetails bundle={result} profiles={profiles}/>,audit:result as unknown as JsonObject});
    } catch(error) { setMessage((error as Error).message); } finally { setSaving(false); }
  };
  const selectedFamilyLabel = familyMeta[family]?.label || t("settings.m_5fd9563e6846");
  return <main className="settings-layout">
    <aside className="settings-sidebar panel">
      <PanelTitle kicker={t("settings.m_d817347b32c4")} title={t("settings.m_3988d2354c99")} help={t("settings.m_8f3cb700222e")} example={t("settings.m_925d63afaf43")} openHelp={openHelp} setOpenHelp={setOpenHelp} />
      <label className="search-box"><span>{t("settings.m_f7a9f4607a42")}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("settings.m_6c744321c488")} /></label>
      <label className="family-select"><span>{t("settings.label_family_picker")}</span><select value={family} onChange={(event) => selectFamily(event.target.value)}>{filteredFamilies.map((item) => <option key={item} value={item}>{familyMeta[item]?.label || t("settings.m_f224d86b5a90")}</option>)}</select></label>
      <div className="family-nav">{filteredFamilies.map((item) => <button key={item} className={family === item ? 'active' : ''} onClick={() => selectFamily(item)}><span>{familyMeta[item]?.label || t("settings.m_f224d86b5a90")}</span><small>{familyMeta[item]?.description || t("settings.m_097648632423")}</small></button>)}</div>
    </aside>
    <section className="settings-main">
      <article className="panel bundle-summary">
        <div><span className="eyebrow">{t("settings.m_ff783005f7a9")}</span><h2>{runtimeBundle ? t("settings.m_fe3770a2635e") + String(runtimeProfile?.runtimeBundleVersion || runtimeBundle.version) : t("settings.m_effe18d1a1f1")}</h2><p>{t("settings.m_83e7aa9ea7c1")}{number.format(Object.keys(runtimeBundle?.componentRevisions || {}).length)}{t("settings.m_c690c0227a79")}{restartPending ? t("settings.m_5f13c68a72d2") : pendingFamilies.length ? (t("settings.m_2dfacf650d48") + number.format(pendingFamilies.length) + t("settings.m_169c4a1d6e1b")) : t("settings.m_68b181f3c00f")}</p></div>
        <button className="secondary" onClick={() => runtimeBundle && setDrawer({kind:'bundle', title:t("settings.m_ff783005f7a9"), subtitle:(t("settings.m_507e5fe82beb") + runtimeBundle.version + ""), content:<ConfigBundleDetails bundle={runtimeBundle} profiles={profiles} />, audit:runtimeBundle as unknown as JsonObject})} disabled={!runtimeBundle}>{t("settings.m_2ea791312e84")}</button>
      </article>
      <article className={`panel compiler-coverage ${compilerCoverage?.complete ? 'complete' : 'incomplete'}`}>
        <div>
          <span className="eyebrow">{t("settings.compiler_coverage_kicker")}</span>
          <h2>{compilerCoverage?.complete ? t("settings.compiler_coverage_complete") : t("settings.compiler_coverage_incomplete")}</h2>
          <p>{compilerCoverage ? `${number.format(compilerCoverage.covered)} ${t("settings.compiler_coverage_of")} ${number.format(compilerCoverage.total)} ${t("settings.compiler_coverage_leaf")}` : t("settings.compiler_coverage_loading")}</p>
          {!compilerCoverage?.complete && <small>{t("settings.compiler_coverage_blocked")}</small>}
        </div>
        <div className="coverage-actions">
          <strong>{number.format(compilerCoverage?.percentage || 0)}{t("workflow.batch_percent_suffix")}</strong>
          <div className="coverage-targets">
            <span>{t("settings.compiler_coverage_binding")} · {number.format(compilerCoverage?.executableBindingCovered || 0)} / {number.format(compilerCoverage?.executableBindingTotal || 0)}</span>
            <span>{t("settings.compiler_coverage_behavioral")} · {number.format(compilerCoverage?.behavioralOrInvariantCovered || 0)} / {number.format(compilerCoverage?.behavioralOrInvariantTotal || 0)}</span>
            {missingCoverageTargets.slice(0, 1).map(([target, count]) => <span key={target}>{coverageTargetLabel(target)} · {number.format(count)}</span>)}
          </div>
          <button className="secondary" type="button" disabled={!compilerCoverage} onClick={() => compilerCoverage && setDrawer({kind:'compiler-coverage', title:t("settings.compiler_coverage_title"), subtitle:t("settings.compiler_coverage_subtitle"), content:<CompilerCoverageDetails coverage={compilerCoverage} />, audit:compilerCoverage as unknown as JsonObject})}>{t("settings.compiler_coverage_open")}</button>
        </div>
      </article>
      <article className="panel bundle-control">
        <div><span className="eyebrow">{t("settings.m_ea893e117b6a")}</span><h2>{restartPending ? t("settings.m_5c30d3b5e104") : bundleDraft ? t("settings.m_d5e80b688112") : pendingFamilies.length ? t("settings.m_45a3cb267c58") : t("settings.m_41d3d2d5e6c6")}</h2><p>{restartPending ? t("settings.m_f1b9d9776a61") : bundleDraft ? t("settings.m_3e7ffda00641") : pendingFamilies.length ? t("settings.m_5379b40088de") : t("settings.m_b3972e397653")}</p></div>
        <div className="governance-actions">{bundleDraft ? <button className="primary" disabled={saving} onClick={()=>publishBundle()}>{t("settings.m_f266e8ad7bb8")}</button> : <button className="primary" disabled={saving || !pendingFamilies.length || restartPending} onClick={composeBundle}>{t("settings.m_bc69672095c5")}</button>}<InfoBadge id="bundle-activation" text={t("settings.m_b55ece3e8286")} example={t("settings.m_ba2d65627ac7")} openHelp={openHelp} setOpenHelp={setOpenHelp} /></div>
        {rollbackBundles.length>0&&<div className="bundle-rollback"><label><span>{t("settings.rules_rollback_title")}</span><select value={rollbackRevision} onChange={(event)=>setRollbackRevision(event.target.value)}>{rollbackBundles.map((item)=><option key={item.revisionId} value={item.revisionId}>{t("settings.m_f14158b9c061")}{item.version}</option>)}</select></label><button className="quiet" type="button" disabled={saving||!rollbackRevision} onClick={()=>{const source=rollbackBundles.find((item)=>item.revisionId===rollbackRevision);if(source) rollbackBundle(source);}}>{t("settings.rules_rollback_action")}</button><InfoBadge id="bundle-rollback" text={t("settings.rules_rollback_help")} example={t("settings.rules_rollback_example")} openHelp={openHelp} setOpenHelp={setOpenHelp}/></div>}
      </article>
      <article className="panel config-editor">
        <div className="config-header"><div><span className="eyebrow">{familyMeta[family]?.label || t("settings.m_f224d86b5a90")}</span><h2>{selected ? (t("settings.m_e2b40627f00d") + selectedFamilyLabel + "") : t("settings.m_8acf4eccc2e7")}</h2><p>{familyMeta[family]?.description}</p></div>{selected && <div className="config-status"><span className={`status ${selected.status}`}>{statusLabel[selected.status] || selected.status}</span><small>{t("settings.m_f14158b9c061")}{selected.version}</small></div>}</div>
        <FieldLabel label={t("settings.m_654f70bad2c9")} help={t("settings.m_16f5c65b587c")} example={t("settings.m_8e3e081eae03")} id="admin-key" openHelp={openHelp} setOpenHelp={setOpenHelp}><input type="password" value={adminKey} onChange={(event)=>setAdminKey(event.target.value)} placeholder={t("settings.m_b6dabbc23e59")} autoComplete="off" /></FieldLabel>
        {selected && <div className="config-governance">
          <label className="revision-picker"><span>{t("settings.rules_revision_picker")}</span><select value={selected.revisionId} onChange={(event)=>{const item=profiles.find((profile)=>profile.revisionId===event.target.value);if(item) openProfile(item);}}>{profiles.filter((item)=>item.family===family&&item.profileId===selected.profileId).sort((left,right)=>right.sequence-left.sequence).map((item)=><option key={item.revisionId} value={item.revisionId}>{t("settings.m_f14158b9c061")}{item.version} · {statusLabel[item.status]||item.status}</option>)}</select></label>
          <div className="config-compare"><label><span>{t("settings.m_6dd224c3be6a")}</span><select value={compareRevision} onChange={(event)=>setCompareRevision(event.target.value)}><option value="">{t("settings.m_b8dbfafadb8e")}</option>{visibleSiblingRevisions.map((item) => <option key={item.revisionId} value={item.revisionId}>{t("settings.m_f14158b9c061")}{item.version} · {date.format(new Date(item.updatedAt))} · {statusLabel[item.status] || item.status}</option>)}</select></label><button className="quiet" disabled={!compareRevision} onClick={compareProfile}>{t("settings.m_f40dce86288e")}</button>{siblingRevisions.length > 12 && <button className="quiet" type="button" onClick={() => setShowAllRevisions((current) => !current)}>{showAllRevisions ? t("workflow.revision_history_recent") : t("workflow.revision_history_all")}</button>}</div>
          {selected.status !== 'draft'&&<div className="governance-actions"><button className="primary" type="button" disabled={saving} onClick={cloneDraft}>{t("settings.rules_clone_draft")}</button><InfoBadge id="clone-profile-draft" text={t("settings.rules_clone_help")} example={t("settings.rules_clone_example")} openHelp={openHelp} setOpenHelp={setOpenHelp}/></div>}
          {['draft','published'].includes(selected.status)&&<div className="config-compare"><label><span>{t("settings.rules_preview_bundle")} · {t("workflow.research_bundle_label")}</span><select value={previewBaseRevision} disabled={saving} onChange={(event)=>{setPreviewBaseRevision(event.target.value);setPreviewBundle(null);setProfileCompilation(null);setProfileCompilationError('');}}><option value="">{t("settings.m_b8dbfafadb8e")}</option>{bundles.filter((item)=>item.status==='published'&&item.componentRevisions[selected.family]).map((item)=><option key={item.revisionId} value={item.revisionId}>{item.bundleId} · {item.version}</option>)}</select></label>{selected.status==='published'&&<button className="secondary" type="button" disabled={saving||!previewBaseRevision} onClick={compileDraftPreview}>{t("settings.rules_compile_preview")}</button>}{selected.status==='published'&&previewBundle&&<><span className={`status ${profileCompilation?.publishable?'completed':'failed'}`}>{profileCompilation?.publishable?t("settings.rules_compile_ready"):t("settings.rules_compile_blocked")} · {previewBundle.version}</span><button className="secondary" type="button" onClick={()=>setDrawer({kind:'bundle-preview',title:t("settings.rules_preview_bundle"),subtitle:previewBundle.version,content:<ConfigBundleDetails bundle={previewBundle} profiles={profiles}/>,audit:previewBundle as unknown as JsonObject})}>{t("settings.m_2ea791312e84")}</button></>}{selected.status==='published'&&profileCompilationError&&<span role="alert">{profileCompilationError}</span>}</div>}
          {selected.status === 'draft' && <><div className="draft-proof-flow"><button className="secondary" type="button" disabled={saving||changed.length>0} onClick={validateDraft}>{t("settings.rules_validate_draft")}</button><button className="secondary" type="button" disabled={saving||changed.length>0||!profileValidation?.valid} onClick={compileDraftPreview}>{t("settings.rules_compile_preview")}</button><span className={`status ${profileCompilation?.publishable?'completed':profileCompilation||profileCompilationError?'failed':'neutral'}`}>{profileCompilation?.publishable?t("settings.rules_compile_ready"):profileCompilation||profileCompilationError?t("settings.rules_compile_blocked"):profileValidation?.valid?t("settings.rules_validation_ready"):t("settings.rules_proof_pending")}</span>{previewBundle&&<small>{t("settings.rules_preview_bundle")} · {previewBundle.version}</small>}</div>{profileCompilation&&<button className="quiet" type="button" onClick={()=>setDrawer({kind:'draft-compilation',title:t("settings.rules_compile_preview"),subtitle:profileCompilation.publishable?t("settings.rules_compile_ready"):t("settings.rules_compile_blocked"),content:<CompilerCoverageDetails coverage={profileCompilation.coverage}/>,audit:profileCompilation as unknown as JsonObject})}>{t("settings.compiler_coverage_open")}</button>}{profileCompilationError&&<div className="configuration-incompatibility"><b>{t("settings.rules_incompatibility")}</b><span>{profileCompilationError}</span></div>}<FieldLabel label={t("settings.m_a2be30bd7d1c")} help={t("settings.m_3dc485743d27")} example={t("settings.m_3cb10aa6d449")} id="transition-reason" openHelp={openHelp} setOpenHelp={setOpenHelp}><input value={transitionReason} onChange={(event)=>setTransitionReason(event.target.value)} placeholder={t("settings.m_6c54871e618e")} /></FieldLabel><div className="governance-actions"><button className="primary" disabled={saving || changed.length > 0 || !profileValidation?.valid || !profileCompilation?.publishable} onClick={() => transitionProfile('publish')}>{t("settings.m_43b17359752d")}</button><button className="quiet danger" disabled={saving || transitionReason.trim().length < 2} onClick={() => transitionProfile('reject')}>{t("settings.m_2f943c821c00")}</button></div>{!profileCompilation?.publishable&&<small>{t("settings.rules_publish_requires_compile")}</small>}</>}
          {selected.status==='published'&&previewBundle&&<button className="primary" type="button" disabled={saving||!profileCompilation?.publishable} onClick={()=>publishBundle(previewBundle)}>{t("settings.m_f266e8ad7bb8")} · {previewBundle.version}</button>}
          {selected.status==='published'&&profileCompilation&&<button className="quiet" type="button" onClick={()=>setDrawer({kind:'draft-compilation',title:t("settings.rules_compile_preview"),subtitle:profileCompilation.publishable?t("settings.rules_compile_ready"):t("settings.rules_compile_blocked"),content:<CompilerCoverageDetails coverage={profileCompilation.coverage}/>,audit:profileCompilation as unknown as JsonObject})}>{t("settings.compiler_coverage_open")}</button>}
          {selected.status === 'published' && <><FieldLabel label={t("settings.m_ec37ff02a796")} help={t("settings.m_bdc72ddc8eed")} example={t("settings.m_a91f20a81250")} id="retire-reason" openHelp={openHelp} setOpenHelp={setOpenHelp}><input value={transitionReason} onChange={(event)=>setTransitionReason(event.target.value)} placeholder={t("settings.m_068a30795bb2")} /></FieldLabel><button className="quiet danger" disabled={saving || transitionReason.trim().length < 2} onClick={() => transitionProfile('retire')}>{t("settings.m_9f09d363a73e")}</button></>}
        </div>}
        <div className="config-actions"><div><b>{changed.length ? ("" + number.format(changed.length) + t("settings.m_608ed1b9e8ba")) : t("settings.m_9998d2d0675a")}</b><small>{selected?.status==='draft'?(changed.length ? t("settings.m_a6d50f3a8854") : t("settings.m_fa7feb486745")):t("settings.rules_clone_help")}</small></div><button className="primary" onClick={saveDraft} disabled={saving || selected?.status!=='draft' || !changed.length}>{t("settings.m_6d148c6fb480")}</button></div>
        {draft ? <><JsonEditor value={draft} schema={profileSchema} rootSchema={profileSchema} disabled={selected?.status!=='draft'} path={[]} onChange={updateAt} openHelp={openHelp} setOpenHelp={setOpenHelp} /><RawJsonFallback value={draft} disabled={selected?.status!=='draft'} onApply={(value)=>{setDraft(value);setProfileValidation(null);setProfileCompilation(null);setProfileCompilationError('');}} /></> : <Empty title={t("settings.m_8d76bccee86a")} body={t("settings.m_4abffef67dae")} />}
        {selected && <details className="audit-details"><summary>{t("settings.m_572e856db09d")}</summary><div className="audit-grid"><span>{t("settings.m_7e5f90820719")}</span><code>{selected.revisionId}</code><span>{t("settings.m_85cc164fdd41")}</span><code>{selected.payloadSha256}</code><span>{t("settings.m_0004b39f98b4")}</span><code>{selected.sequence}</code></div></details>}
      </article>
    </section>
  </main>;
}

async function ensureConfigurationIdentity() {
  const identity = await api<{identity: JsonObject | null}>('/api/v1/auth/me');
  if (!identity.identity) await api('/api/v1/auth/guest', {method:'POST', body:JSON.stringify({displayName:t("evidence.m_ac3ced791b65")})});
}
function resolveSchemaNode(root:JsonSchema|null,node:JsonSchema|null|undefined):JsonSchema|undefined {
  if(!node?.$ref||!root) return node||undefined;
  const prefix='#/$defs/'; if(!node.$ref.startsWith(prefix)) return node;
  return (root.$defs as Record<string,JsonSchema>|undefined)?.[node.$ref.slice(prefix.length)]||node;
}
function JsonEditor({value, schema, rootSchema, disabled, path, onChange, openHelp, setOpenHelp}: {value:JsonObject;schema:JsonSchema|null|undefined;rootSchema:JsonSchema|null;disabled:boolean;path:(string|number)[];onChange:(path:(string|number)[],value:JsonValue)=>void;openHelp:string|null;setOpenHelp:(value:string|null)=>void}) {
  // A nested $id starts a schema resource; its recursive refs are not profile-root refs.
  const resource = schema?.$id && schema.$defs ? schema : rootSchema;
  const node = resolveSchemaNode(resource, schema);
  const expressionCopyAvailable = node?.['x-familyExpression']!==true || (
    hasFoundryCopy('workflow.research_scenario_expression_operator')
    && hasFoundryCopy('workflow.research_scenario_capability_unsupported')
    && configurationEditorExpressionChoices(node).every((choice) => {
      const key = choice['x-labelKey'];
      return typeof key==='string' && hasFoundryCopy(key.includes('.')?key:`workflow.${key}`);
    }));
  const resolved = expressionCopyAvailable ? node : undefined;
  const fieldSchema = configurationEditorSchema(value, resolved) as JsonSchema | undefined;
  const choices = configurationEditorExpressionChoices(resolved);
  const selected = configurationEditorExpressionChoice(value, resolved);
  const entrySchema = fieldSchema && {...fieldSchema, properties: Object.fromEntries(Object.entries(fieldSchema.properties || {}).map(([key, field]) => {
    const child = resolveSchemaNode(resource, field);
    return [key, child?.items ? {...child, items: resolveSchemaNode(resource, child.items)} : child];
  }))} as JsonSchema | undefined;
  if (entrySchema && choices.length) delete entrySchema.oneOf;
  const entries=choices.length?configurationEditorEntries(value,entrySchema?.properties,entrySchema):configurationEditorEntries(value,resolved?.properties,resolved);
  return <div className="json-editor">{choices.length > 0 && <label className="field-label"><span>{t('workflow.research_scenario_expression_operator')}</span><select disabled={disabled} value={selected < 0 ? '' : String(selected)} onChange={(event)=>{const next=configurationEditorSelectExpression(value,resolved!,Number(event.target.value));if(next)onChange(path,next);}}><option value="" disabled>—</option>{choices.map((choice,index)=><option key={index} value={index} disabled={choice['x-constructionSupported']===false&&selected!==index}>{tCatalog(String(choice['x-labelKey']))}{choice['x-constructionSupported']===false?` · ${t('workflow.research_scenario_capability_unsupported')}`:''}</option>)}</select></label>}{entries.map(([key, child], index) => {
    const nextPath = [...path, key]; const id = `config-${nextPath.join('-')}`;
    const childSchema=resolveSchemaNode(resource,fieldSchema?.properties?.[key]);
    if (choices.length && (key==='op'||key==='kind')) return null;
    const required = fieldSchema?.required;
    const onRemove = (!Array.isArray(required) || !required.includes(key))
      ? () => onChange(path, Object.fromEntries(Object.entries(value).filter(([name]) => name !== key))) : undefined;
    const arraySchema = childSchema && {...childSchema, items: resolveSchemaNode(resource, childSchema.items)} as JsonSchema;
    const labelKey=childSchema?.['x-messageArKey'];
    const name = typeof labelKey==='string'&&hasFoundryCopy(labelKey.includes('.')?labelKey:`workflow.${labelKey}`) ? tCatalog(labelKey) : childSchema?.title || configFieldName(key,child,nextPath);
    if (Array.isArray(childSchema?.['x-familyParameterOptions'])) return <ConfigParameterField key={key} name={name} value={child} schema={childSchema!} disabled={disabled} path={nextPath} onChange={onChange} onRemove={onRemove} openHelp={openHelp} setOpenHelp={setOpenHelp}/>;
    if (Array.isArray(arraySchema?.items?.['x-familyParameterOptions'])) return <ConfigParameterArraySection key={key} name={name} value={child} schema={arraySchema!} disabled={disabled} path={nextPath} onChange={onChange} onRemove={onRemove} openHelp={openHelp} setOpenHelp={setOpenHelp}/>;
    const arrayItemSchema = hasFoundryCopy('workflow.research_scenario_expression_add_group')
      ? configurationEditorObjectArraySchema(arraySchema) : undefined;
    if (arrayItemSchema && (child == null || Array.isArray(child) && child.every((item) => item && typeof item === 'object' && !Array.isArray(item)))) {
      return <ConfigObjectArraySection key={key} name={name} technicalKey={key} value={child} schema={arraySchema!} rootSchema={resource} disabled={disabled} path={nextPath} onChange={onChange} onRemove={onRemove} openHelp={openHelp} setOpenHelp={setOpenHelp} />;
    }
    if (child && typeof child === 'object' && !Array.isArray(child)) return <ConfigObjectSection key={key} technicalKey={key} name={name} value={child as JsonObject} schema={childSchema} rootSchema={resource} disabled={disabled} path={nextPath} onChange={onChange} onRemove={onRemove} defaultOpen={path.length === 0 && index < 2} openHelp={openHelp} setOpenHelp={setOpenHelp} />;
    return <ConfigField key={key} name={name} technicalKey={key} value={child} schema={childSchema} disabled={disabled||childSchema?.readOnly===true||childSchema?.const!==undefined} path={nextPath} onRemove={onRemove} onChange={(_fieldPath, nextChild) => onChange(path, configurationEditorChange(value, resolved, key, nextChild))} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp} />;
  })}</div>;
}
function ConfigParameterField({name,value,schema,disabled,path,onChange,onRemove,openHelp,setOpenHelp}: {name:string;value:JsonValue;schema:JsonSchema;disabled:boolean;path:(string|number)[];onChange:(path:(string|number)[],value:JsonValue)=>void;onRemove?:()=>void;openHelp:string|null;setOpenHelp:(value:string|null)=>void}) {
  const reference=value&&typeof value==='object'&&!Array.isArray(value)?String(value.$parameter??''):'';
  const alternatives=Array.isArray(schema.anyOf)?schema.anyOf:[];
  const literal=alternatives[0] as JsonSchema;
  const id=`config-${path.join('-')}`;
  if (!hasFoundryCopy('workflow.research_lesson_tag_template_source') || !hasFoundryCopy('workflow.research_lesson_tag_template_literal')) {
    return reference
      ? <ConfigObjectSection name={name} technicalKey={String(path.at(-1))} value={value as JsonObject} schema={alternatives[1] as JsonSchema} rootSchema={null} disabled={disabled} path={path} onChange={onChange} onRemove={onRemove} defaultOpen={true} openHelp={openHelp} setOpenHelp={setOpenHelp}/>
      : <ConfigField name={name} technicalKey={String(path.at(-1))} value={value} schema={literal} disabled={disabled} path={path} onChange={onChange} onRemove={onRemove} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}/>;
  }
  return <fieldset className="scenario-structured-field"><legend>{name}</legend><label className="field-label"><span>{t('workflow.research_lesson_tag_template_source')}</span><select disabled={disabled} value={reference} onChange={(event)=>onChange(path,event.target.value?{$parameter:event.target.value}:null)}><option value="">{t('workflow.research_lesson_tag_template_literal')}</option>{(schema['x-familyParameterOptions'] as JsonValue[]).map((parameter)=><option value={String(parameter)} key={String(parameter)}>{configFieldName(String(parameter),parameter,path)}</option>)}</select></label>{!reference&&<ConfigField name={name} technicalKey={String(path.at(-1))} value={value} schema={literal} disabled={disabled} path={path} onChange={onChange} onRemove={onRemove} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}/>} {reference&&onRemove&&<button type="button" className="quiet" disabled={disabled} onClick={onRemove}>{t('workflow.research_scenario_remove')}</button>}</fieldset>;
}
function ConfigParameterArraySection({name,value,schema,disabled,path,onChange,onRemove,openHelp,setOpenHelp}: {name:string;value:JsonValue;schema:JsonSchema;disabled:boolean;path:(string|number)[];onChange:(path:(string|number)[],value:JsonValue)=>void;onRemove?:()=>void;openHelp:string|null;setOpenHelp:(value:string|null)=>void}) {
  const items=Array.isArray(value)?value:[];
  const minimum=typeof schema.minItems==='number'?schema.minItems:0;
  const maximum=typeof schema.maxItems==='number'?schema.maxItems:Infinity;
  const itemSchema=configurationEditorArrayItemSchema(schema) as JsonSchema;
  return <fieldset className="scenario-structured-field"><legend>{name}</legend>{onRemove&&<button type="button" className="quiet" disabled={disabled} onClick={onRemove}>{t('workflow.research_scenario_remove')} · {name}</button>}{items.map((item,index)=><div key={index}><ConfigParameterField name={`${name} · ${number.format(index+1)}`} value={item} schema={itemSchema} disabled={disabled} path={[...path,index]} onChange={onChange} openHelp={openHelp} setOpenHelp={setOpenHelp}/><button type="button" className="quiet" disabled={disabled||items.length<=minimum} onClick={()=>onChange(path,items.filter((_,position)=>position!==index))}>{t('workflow.research_scenario_remove')}</button></div>)}<button type="button" className="secondary" disabled={disabled||items.length>=maximum} onClick={()=>onChange(path,[...items,null])}>{t('workflow.research_scenario_add')}</button></fieldset>;
}
function ConfigObjectArraySection({name, technicalKey, value, schema, rootSchema, disabled, path, onChange, onRemove, openHelp, setOpenHelp}: {name:string;technicalKey:string;value:JsonValue;schema:JsonSchema;rootSchema:JsonSchema|null;disabled:boolean;path:(string|number)[];onChange:(path:(string|number)[],value:JsonValue)=>void;onRemove?:()=>void;openHelp:string|null;setOpenHelp:(value:string|null)=>void}) {
  const [expanded, setExpanded] = useState(false);
  const items = Array.isArray(value) ? value as JsonObject[] : [];
  const itemSchema = configurationEditorObjectArraySchema(schema) as JsonSchema;
  const id = `config-${path.join('-')}`;
  const added = configurationEditorObjectArrayAdd(value, schema);
  const minimum = typeof schema.minItems === 'number' ? schema.minItems : 0;
  return <details className="config-section" data-config-path={path.join('.')} open={expanded} onToggle={(event)=>setExpanded(event.currentTarget.open)}>
    <summary><span>{name} · {number.format(items.length)}</span><InfoBadge id={id} text={schema.description || name} example={t("evidence.m_81ef2928b9cc")} technical={technicalKey} openHelp={openHelp} setOpenHelp={setOpenHelp}/>{onRemove&&<button type="button" className="quiet" disabled={disabled} aria-label={`${t("workflow.research_scenario_remove")} · ${name}`} onClick={(event)=>{event.preventDefault();onRemove();}}>{t("workflow.research_scenario_remove")}</button>}</summary>
    {expanded&&<div className="json-editor">{items.map((item,index)=>
      <ConfigObjectSection key={index} technicalKey={technicalKey} name={`${name} · ${number.format(index+1)}`} value={item} schema={itemSchema} rootSchema={rootSchema} disabled={disabled} path={[...path,index]} onChange={onChange} onRemove={items.length>minimum?()=>{const removed=configurationEditorObjectArrayRemove(value,schema,index);if(!disabled&&removed)onChange(path,removed);}:undefined} defaultOpen={index===0} openHelp={openHelp} setOpenHelp={setOpenHelp}/>
    )}<button type="button" className="secondary" disabled={disabled||!added} aria-label={`${t("workflow.research_scenario_expression_add_group")} · ${name}`} onClick={()=>{if(!disabled&&added)onChange(path,added);}}>{t("workflow.research_scenario_expression_add_group")}</button></div>}
  </details>;
}
function configSchemaHelp(schema: JsonSchema | undefined, fallback: string): string {
  const key = schema?.['x-helpArKey'];
  return typeof key === 'string' && hasFoundryCopy(key.includes('.') ? key : `workflow.${key}`)
    ? tCatalog(key) : schema?.description || fallback;
}
function ConfigObjectSection({technicalKey, name, value, schema, rootSchema, disabled, path, onChange, onRemove, defaultOpen, openHelp, setOpenHelp}: {technicalKey:string;name:string;value:JsonObject;schema:JsonSchema|undefined;rootSchema:JsonSchema|null;disabled:boolean;path:(string|number)[];onChange:(path:(string|number)[],value:JsonValue)=>void;onRemove?:()=>void;defaultOpen:boolean;openHelp:string|null;setOpenHelp:(value:string|null)=>void}) {
  const [expanded, setExpanded] = useState(defaultOpen);
  const id = `config-${path.join('-')}`;
  const thrillSection = path.includes('thrill_profile');
  const help = configSchemaHelp(schema, t("evidence.m_717469e6c4d8") + name + t("evidence.m_29587fe83309"));
  const example = technicalKey === 'thrill_profile'
    ? t("helpers.thrill_profile_example")
    : thrillSection && technicalKey === 'weights'
      ? t("helpers.thrill_weight_example")
      : t("evidence.m_81ef2928b9cc");
  return <details className="config-section" data-config-path={path.join('.')} open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}><summary><span>{name}</span><InfoBadge id={id} text={help} example={example} technical={technicalKey} openHelp={openHelp} setOpenHelp={setOpenHelp} />{onRemove && <button type="button" className="quiet" disabled={disabled} aria-label={`${t("workflow.research_scenario_remove")} · ${name}`} onClick={(event)=>{event.preventDefault();onRemove();}}>{t("workflow.research_scenario_remove")}</button>}</summary>{expanded && <JsonEditor value={value} schema={schema} rootSchema={rootSchema} disabled={disabled} path={path} onChange={onChange} openHelp={openHelp} setOpenHelp={setOpenHelp} />}</details>;
}
function ConfigField({name, technicalKey, value, schema, disabled, path, onChange, onRemove, id, openHelp, setOpenHelp}: {name:string;technicalKey:string;value:JsonValue;schema:JsonSchema|undefined;disabled:boolean;path:(string|number)[];onChange:(path:(string|number)[],value:JsonValue)=>void;onRemove?:()=>void;id:string;openHelp:string|null;setOpenHelp:(value:string|null)=>void}) {
  const help = configSchemaHelp(schema, t("evidence.m_1f81dffaa812") + name + t("evidence.m_24b96014dd3a"));
  const example = configValueExample(technicalKey, value, path);
  const templateDirection = ['video.purchase.hokum_summary', 'video.purchase.plain_summary', 'purchase_hokum_summary', 'purchase_plain_summary'].includes(technicalKey) ? 'ltr' : undefined;
  const arrayOptions = configurationEditorArrayOptions(schema);
  const enumTitle = (item: JsonValue, itemSchema?: JsonObject) => {
    const key=configurationEditorEnumMessageKey(item,itemSchema);
    return key&&hasFoundryCopy(key.includes('.')?key:`workflow.${key}`)?tCatalog(key):configurationEditorEnumTitle(item,itemSchema);
  };
  if (arrayOptions && (value == null || Array.isArray(value) && value.every((item) => arrayOptions.includes(item)))) {
    const selected = Array.isArray(value) ? value : [];
    const minimum = typeof schema?.minItems === 'number' ? schema.minItems : 0;
    const maximum = typeof schema?.maxItems === 'number' ? schema.maxItems : arrayOptions.length;
    const itemSchema = configurationEditorArrayItemSchema(schema!);
    return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><div className="config-chip-picker" role="group" aria-label={name}>{arrayOptions.map((option) => {
      const active = selected.includes(option);
      const canRemoveField = selected.length === 1 && onRemove;
      return <button type="button" key={`${typeof option}:${String(option)}`} aria-pressed={active} className={active ? 'active' : ''} disabled={disabled || (active ? selected.length <= minimum && !canRemoveField : selected.length >= maximum)} onClick={() => {
        if (active && canRemoveField) { onRemove?.(); return; }
        onChange(path, arrayOptions.filter((item) => item === option ? !active : selected.includes(item)));
      }}>{enumTitle(option, itemSchema) ?? configFieldName(String(option), option, path)}</button>;
    })}</div></FieldLabel>;
  }
  if (typeof value === 'boolean') return <div className="config-field"><div className="field-copy"><b>{name}</b><small>{value ? t("evidence.m_33fa78354958") : t("evidence.m_58cdbfb276b8")}</small>{onRemove&&<button type="button" className="quiet" disabled={disabled} aria-label={`${t("workflow.research_scenario_remove")} · ${name}`} onClick={()=>{if(!disabled)onRemove();}}>{t("workflow.research_scenario_remove")}</button>}</div><InfoBadge id={id} text={help} example={example} technical={technicalKey} openHelp={openHelp} setOpenHelp={setOpenHelp} /><button type="button" id={id} aria-label={name} disabled={disabled} className={`toggle ${value ? 'on' : ''}`} role="switch" aria-checked={value} onClick={() => onChange(path, !value)}><i /></button></div>;
  if (value == null && schema?.type === 'object') return <div className="config-field"><div className="field-copy"><b>{name}</b><small>{t("evidence.m_58cdbfb276b8")}</small></div><InfoBadge id={id} text={help} example={example} technical={technicalKey} openHelp={openHelp} setOpenHelp={setOpenHelp} /><button id={id} type="button" className="toggle" role="switch" aria-label={name} aria-checked={false} disabled={disabled} onClick={()=>onChange(path,configurationEditorObjectValue(schema))}><i /></button></div>;
  const options = configurationEditorEnum(schema);
  if (options?.length) {
    const clear = onRemove && options.every((item)=>String(item)!=='') ? onRemove : undefined;
    return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><select id={id} aria-label={name} aria-describedby={`${id}-help-content`} disabled={disabled} value={value == null ? '' : String(value)} onChange={(event)=>{if(event.target.value===''&&clear){clear();return;}const selected=options.find((item)=>String(item)===event.target.value);if(selected!==undefined) onChange(path,selected);}}>{(value == null || clear) && <option value="" disabled={!clear}>{clear ? t("evidence.m_58cdbfb276b8") : ''}</option>}{options.map((item)=><option key={String(item)} value={String(item)}>{enumTitle(item,schema) ?? (typeof item === 'boolean' ? t(item ? "evidence.m_33fa78354958" : "evidence.m_58cdbfb276b8") : typeof item === 'number' ? number.format(item) : configFieldName(String(item),item,path))}</option>)}</select></FieldLabel>;
  }
  if (technicalKey === 'purchaseSuits' && Array.isArray(value)) {
    const options = [
      ['spades', t("configuration_fields.label_suit_spades")],
      ['hearts', t("configuration_fields.label_suit_hearts")],
      ['diamonds', t("configuration_fields.label_suit_diamonds")],
      ['clubs', t("configuration_fields.label_suit_clubs")],
    ];
    return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><div className="config-chip-picker">{options.map(([option, label]) => {const active=value.includes(option); return <button type="button" disabled={disabled} key={option} className={active ? 'active' : ''} aria-pressed={active} onClick={() => onChange(path, active ? value.filter((item) => item !== option) : [...value, option])}>{label}</button>;})}</div></FieldLabel>;
  }
  if (Array.isArray(value)&&value.every((item)=>['string','number'].includes(typeof item))) return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><input disabled={disabled} value={value.join(', ')} onChange={(event)=>{const parts=event.target.value.split(new RegExp('[,\\u060c]')).map((item)=>item.trim()).filter(Boolean);onChange(path,typeof value[0]==='number'?parts.map(Number):parts);}} /></FieldLabel>;
  if (Array.isArray(value)) return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><textarea disabled={disabled} value={JSON.stringify(value, null, 2)} onChange={(event) => {try {const parsed=JSON.parse(event.target.value); if(Array.isArray(parsed)) onChange(path, parsed);} catch {/* keep last valid value */}}} dir="ltr" /></FieldLabel>;
  if (typeof value === 'number' || schema?.type === 'number' || schema?.type === 'integer') return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp} onRemove={onRemove} disabled={disabled}><input id={id} aria-label={name} aria-describedby={`${id}-help-content`} disabled={disabled} type="number" min={schema?.minimum} max={schema?.maximum} step={schema?.type === 'integer' ? 1 : 'any'} value={value == null ? '' : Number(value)} onChange={(event) => onChange(path, event.target.value === '' ? null : event.target.valueAsNumber)} /></FieldLabel>;
  if (technicalKey === 'assumedBuyer') return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><select value={String(value)} onChange={(event) => onChange(path, event.target.value)}>{Object.entries(seatLabel).map(([seat, label]) => <option key={seat} value={seat}>{label}</option>)}</select></FieldLabel>;
  if (technicalKey === 'backend' && value === 'ortools_cp_sat_v1') return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><select value="ortools_cp_sat_v1" onChange={() => undefined}><option value="ortools_cp_sat_v1">OR-Tools CP-SAT</option></select></FieldLabel>;
  if (technicalKey === 'renderPolicy') return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><select value={String(value)} onChange={(event) => onChange(path, event.target.value)}><option value="on_demand">{t("workflow.batch_render_policy_on_demand")}</option><option value="automatic">{t("workflow.batch_render_policy_automatic")}</option></select></FieldLabel>;
  if (technicalKey === 'explanation_voice_mode') return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp}><select value={String(value)} onChange={(event) => onChange(path, event.target.value)}><option value="exact_transcript_only">{t("evidence.m_774fdbec4e7c")}</option></select></FieldLabel>;
  return <FieldLabel label={name} help={help} example={example} technical={technicalKey} id={id} openHelp={openHelp} setOpenHelp={setOpenHelp} onRemove={onRemove} disabled={disabled}><input id={id} disabled={disabled||schema?.const!==undefined} dir={templateDirection} value={value == null ? '' : String(value)} onChange={(event) => onChange(path, event.target.value)} /></FieldLabel>;
}
function RawJsonFallback({value,disabled,onApply}:{value:JsonObject;disabled:boolean;onApply:(value:JsonObject)=>void}) {
  const [text,setText]=useState(()=>JSON.stringify(value,null,2)); const [valid,setValid]=useState(true);
  useEffect(()=>{setText(JSON.stringify(value,null,2));setValid(true);},[value]);
  return <details className="raw-json-fallback"><summary>{t("settings.rules_advanced_json")}</summary><p>{t("settings.rules_advanced_json_help")}</p><textarea dir="ltr" disabled={disabled} value={text} onChange={(event)=>{setText(event.target.value);try{const parsed=JSON.parse(event.target.value);setValid(Boolean(parsed)&&typeof parsed==='object'&&!Array.isArray(parsed));}catch{setValid(false);}}}/><button className="secondary" type="button" disabled={disabled||!valid} onClick={()=>onApply(JSON.parse(text) as JsonObject)}>{t("settings.rules_advanced_json_apply")}</button></details>;
}
function diffPaths(left: JsonValue, right: JsonValue, path = ''): string[] {
  if (JSON.stringify(left) === JSON.stringify(right)) return [];
  if (left && right && typeof left === 'object' && typeof right === 'object' && !Array.isArray(left) && !Array.isArray(right)) return [...new Set([...Object.keys(left), ...Object.keys(right)])].flatMap((key) => diffPaths((left as JsonObject)[key], (right as JsonObject)[key], path ? `${path}.${key}` : key));
  return [path || 'root'];
}

function FinalistSummary({job}: {job: FinalistValidationJob}) {
  const items = Array.isArray(job.result?.items) ? job.result.items : [];
  const blockerLabels: Record<string, string> = {
    professional_agent_qualification_missing: t("evidence.m_a7bef3c18e6c"),
    no_finalist_candidate_survived: t("evidence.m_9fa2c06f7bd8"),
    incomplete_four_observer_coverage: t("evidence.m_ae3ef1bec3fe"),
    material_preceding_mistake: t("evidence.m_438e1a1d7eaa"),
    material_later_mistake: t("evidence.m_d045cb13772c"),
    laffa_selection_failed: t("evidence.label_laffa_selection_failed"),
  };
  const blockers = [...new Set(items.flatMap((item) => Array.isArray(item.blockers) ? item.blockers.map(String) : []))];
  const laffaScore = items.flatMap((item) => {
    const analysis = item.analysisEvidence && typeof item.analysisEvidence === 'object' && !Array.isArray(item.analysisEvidence) ? item.analysisEvidence as JsonObject : null;
    const score = analysis?.laffaSelectionScore && typeof analysis.laffaSelectionScore === 'object' && !Array.isArray(analysis.laffaSelectionScore) ? analysis.laffaSelectionScore as JsonObject : null;
    const value = Number(score?.score);
    return Number.isFinite(value) ? [value] : [];
  })[0];
  return <>
    <div className="drawer-stats">
      <Stat label={t("evidence.m_37e70eac069b")} value={job.result?.localPreviewReady ? t("evidence.m_066ab2581ec3") : job.status === 'completed' ? t("evidence.m_63a37d34a922") : t("evidence.m_2813b1ec6664")} />
      <Stat label={t("evidence.m_f875c8be70b0")} value={number.format(Number(job.result?.validatedLaffaCount || 0))} />
      <Stat label={t("evidence.m_d30908a0b6c6")} value={number.format(Number(job.result?.candidateCount || 0))} />
      {Number.isFinite(laffaScore) && <Stat label={t("evidence.label_laffa_selection_score")} value={number.format(Math.round(laffaScore))} />}
    </div>
    <p className="drawer-lead">{t("evidence.m_537937ac2c6a")}</p>
    {blockers.length > 0 && <><h3>{t("evidence.m_c43610d14cc7")}</h3><div className="evidence-list">{blockers.map((blocker) => <div key={blocker}><span>{t("evidence.m_c006c6a8222c")}</span><b>{blockerLabels[blocker] || t("evidence.m_deabf5678fe4")}</b></div>)}</div></>}
    {job.result?.localPreviewReady && <div className="decision-answer"><span>{t("evidence.m_e8f886996b65")}</span><b>{t("evidence.m_d09a8cdcd375")}</b><small>{t("evidence.m_9d266ec64bf4")}</small></div>}
  </>;
}

function AgentQualificationDetails({value, pilot}: {value: FoundryProfile['finalistAgentQualification']; pilot: FoundryProfile['finalistAgentPilotEvidence']}) {
  return <>
    <div className="drawer-stats">
      <Stat label={t("workflow.label_agent_equal")} value={value.configuredEqual ? t("workflow.label_complete") : t("workflow.label_pending")} />
      <Stat label={t("workflow.label_agent_search")} value={value.zeroMistakeSearchReady ? t("workflow.label_complete") : t("workflow.label_pending")} />
      <Stat label={t("workflow.label_agent_local_pilot")} value={pilot.passed ? t("workflow.label_agent_local_pilot_passed") : t("workflow.label_agent_local_pilot_pending")} />
      <Stat label={t("workflow.label_agent_machine_evidence")} value={value.benchmarkEvidenceComplete ? t("workflow.label_complete") : t("workflow.label_pending")} />
      <Stat label={t("workflow.label_agent_professional_review")} value={value.professionalReviewComplete ? t("workflow.label_complete") : t("workflow.label_pending")} />
    </div>
    <p className="drawer-lead">{t("workflow.label_agent_local_pilot_boundary")}</p>
    <p className="drawer-lead">{value.expertGenerationAllowed ? t("workflow.label_agent_ready_release") : t("workflow.label_agent_pending")}</p>
  </>;
}

function CandidateDetails({candidate}: {candidate: AnalysisCandidate}) {
  const evidence = candidateDimensionOrder.flatMap((key) => candidate.dimensions[key] == null ? [] : [[key, candidate.dimensions[key]] as const]);
  return <>
    <div className="drawer-stats">
      <Stat label={t("evidence.label_decision_selection_score")} value={number.format(Math.round(candidate.selectionScore))} />
      <Stat label={t("evidence.m_93694193e23b")} value={("" + number.format(Math.round(candidate.confidence * 100)) + t("evidence.m_a489ae36a167"))} />
      <Stat label={t("evidence.m_a78bd91114ab")} value={number.format(candidate.alternativeCount)} />
    </div>
    <p className="drawer-lead">{laffaLabel[candidate.laffaClassification] || t("evidence.m_9e323f2380b9")}</p>
    <div className="decision-answer"><span>{t("evidence.m_27fde21d5951")}</span><b dir="ltr">{actionLabel(candidate.bestAction)}</b><small>{truthLabel[candidate.truthLabel] || candidate.truthLabel}{t("evidence.m_d3e60e60ad86")}{seatLabel[candidate.observerSeat] || candidate.observerSeat}</small></div>
    <h3>{t("evidence.m_bcee4b04800e")}</h3>
    <div className="evidence-list">
      {evidence.map(([key, value]) => <div key={key}><span>{candidateDimensionLabel[key]}</span><b>{candidateDimensionValue(key, String(value))}</b></div>)}
    </div>
    {Boolean(candidate.compatibilityBlockers?.length) && <p className="drawer-lead">{t("workflow.label_incompatible_candidate_help")}</p>}
    {!candidate.playReady && !candidate.compatibilityBlockers?.length && <p className="drawer-lead">{candidate.finalistValidationRequired
      ? t("evidence.m_944f3dac94db")
      : t("evidence.m_4669103edcdc")}</p>}
  </>;
}

function PreparedDetails({value}: {value: PreparedCandidate}) {
  const code = String(value.locator.code || value.candidate.shortCode);
  const link = typeof value.locator.universalLink === 'string' ? value.locator.universalLink : '';
  return <>
    <div className="locator-card"><span>{t("evidence.m_a49bf9b3202d")}</span><b dir="ltr">{code}</b><small>{t("evidence.m_3cd8ddb045eb")}</small></div>
    <p className="drawer-lead">{t("evidence.m_0503e082827e")}</p>
    {link && <a className="button-link secondary" href={link}>{t("evidence.m_07f47136b3e0")}</a>}
  </>;
}

function RenderOutputCard({job}: {job: RenderJob}) {
  const media = renderMedia(job);
  if (!media) return null;
  const {videoUrl, posterUrl, manifest} = media;
  return <article className="render-output" id="render-output">
    <div className="render-output-copy">
      <span>{t("workflow.m_67149f34df53")}</span>
      <b>{t("workflow.m_f8f00b7c0856")}</b>
    </div>
    <video controls playsInline preload="metadata" poster={posterUrl} src={videoUrl} />
    <div className="render-output-actions">
      <a className="button-link primary" href={videoUrl}>{t("settings.m_94d1f1049135")}</a>
      {manifest && <a className="button-link secondary" href={manifest}>{t("evidence.m_97b7815c26dc")}</a>}
    </div>
  </article>;
}

function renderMedia(job: RenderJob) {
  const result = job.result as Record<string, unknown> | null | undefined;
  const outputBaseUrl = typeof result?.outputBaseUrl === 'string' ? result.outputBaseUrl : '';
  const batch = result?.batch && typeof result.batch === 'object' ? result.batch as Record<string, unknown> : null;
  const items = Array.isArray(batch?.items) ? batch.items : [];
  const first = items[0] && typeof items[0] === 'object' ? items[0] as Record<string, unknown> : null;
  const basename = typeof first?.basename === 'string' ? first.basename : '';
  const manifests = Array.isArray(result?.manifests) ? result.manifests : [];
  const manifest = typeof manifests[0] === 'string' ? manifests[0] : '';
  const safeBase = outputBaseUrl.startsWith('/outputs/render-batches/') ? outputBaseUrl : '';
  const safeName = /^[a-z0-9._-]+$/i.test(basename) ? basename : '';
  if (!safeBase || !safeName) return null;
  return {videoUrl:`${safeBase}/${safeName}.mp4`, posterUrl:`${safeBase}/${safeName}.jpg`, manifest, basename, first};
}

function renderSource(job: RenderJob): Record<string, unknown> | null {
  const requestItems = Array.isArray(job.request?.items) ? job.request.items : [];
  const first = requestItems[0];
  if (!first || typeof first !== 'object' || Array.isArray(first)) return null;
  const source = first.source;
  return source && typeof source === 'object' && !Array.isArray(source)
    ? source as Record<string, unknown>
    : null;
}

function RecentRenderCard({job}: {job: RenderJob}) {
  const media = renderMedia(job);
  if (!media) return null;
  const preset = typeof media.first?.exportPreset === 'string' ? media.first.exportPreset : '';
  const source = renderSource(job);
  const sourceCode = typeof source?.code === 'string' ? source.code : '';
  const sourceHref = sourceCode
    ? `/foundry?section=laffat&laffa=${encodeURIComponent(sourceCode)}#laffa-library`
    : '';
  const purpose = source?.auditOnly === false && source?.publicationEligible === true && source?.renderPurpose === 'content_candidate'
    ? 'content_candidate'
    : 'research_audit';
  const publicationBlockers=Array.isArray(source?.publicationBlockers)?source.publicationBlockers.map(String):[];
  return <article className="video-library-card">
    <img src={media.posterUrl} alt={t("workflow.label_video_poster")} loading="lazy" />
    <div className="video-library-copy">
      <span>{preset === 'hooked_flagship' ? t("workflow.m_6bb067528380") : t("workflow.m_c21e98a53a03")}</span>
      <b>{t("workflow.label_generated_at")}{job.createdAt ? date.format(new Date(job.createdAt)) : t("workflow.label_ready_now")}</b>
      {sourceCode && <small>{t("workflow.video_library_source_laffa")} · <b dir="ltr">{sourceCode}</b></small>}
      {source && <span className={`status ${purpose === 'research_audit' ? 'rejected' : 'completed'}`}>{tVariant('workflow.research_render', purpose)}</span>}
      {publicationBlockers.length>0&&<small>{t("workflow.laffa_library_qualification_failures")}: {number.format(publicationBlockers.length)}</small>}
    </div>
    <div className="video-library-actions">
      {sourceHref && <a className="button-link secondary" href={sourceHref}>{t("workflow.video_library_open_source")}</a>}
      <a className="button-link primary" href={media.videoUrl}>{t("workflow.label_open_video")}</a>
    </div>
  </article>;
}

function JourneyStep({index, title, description, state, help, example, openHelp, setOpenHelp}: {index:number; title:string; description:string; state:string; help:string; example:string; openHelp:string|null; setOpenHelp:(value:string|null)=>void}) {
  return <article className={`journey-step ${state}`}><span className="step-number">{state === 'done' ? '✓' : number.format(index)}</span><div><b>{title}</b><small>{description}</small></div><InfoBadge id={`journey-${index}`} text={help} example={example} openHelp={openHelp} setOpenHelp={setOpenHelp} /></article>;
}
function Metric({label, value, tone='', help, example, openHelp, setOpenHelp}: {label:string; value:number; tone?:string; help:string; example:string; openHelp:string|null; setOpenHelp:(value:string|null)=>void}) {
  return <article className={`metric ${tone}`}><div><span>{label}</span><InfoBadge id={`metric-${label}`} text={help} example={example} openHelp={openHelp} setOpenHelp={setOpenHelp} /></div><b>{number.format(value)}</b></article>;
}
function PanelTitle({kicker, title, help, example, openHelp, setOpenHelp}: {kicker:string; title:string; help:string; example:string; openHelp:string|null; setOpenHelp:(value:string|null)=>void}) {
  return <div className="panel-heading"><div><span>{kicker}</span><h2>{title}</h2></div><InfoBadge id={`panel-${title}`} text={help} example={example} openHelp={openHelp} setOpenHelp={setOpenHelp} /></div>;
}
function FieldLabel({label, help, example, technical, id, openHelp, setOpenHelp, children, onRemove, disabled}: {label:string; help:string; example:string; technical?:string; id:string; openHelp:string|null; setOpenHelp:(value:string|null)=>void; children:ReactNode;onRemove?:()=>void;disabled?:boolean}) {
  const Container = onRemove ? 'div' : 'label';
  return <Container className="field-label"><span>{onRemove?<label htmlFor={id}><b>{label}</b></label>:<b>{label}</b>}<InfoBadge id={`${id}-help`} text={help} example={example} technical={technical} openHelp={openHelp} setOpenHelp={setOpenHelp} />{onRemove&&<button type="button" className="quiet" disabled={disabled} aria-label={`${t("workflow.research_scenario_remove")} · ${label}`} onClick={(event)=>{event.preventDefault();if(!disabled)onRemove();}}>{t("workflow.research_scenario_remove")}</button>}</span>{children}</Container>;
}
function InfoBadge({id, text, example, technical, openHelp, setOpenHelp}: {id:string; text:string; example:string; technical?:string; openHelp:string|null; setOpenHelp:(value:string|null)=>void}) {
  const open = openHelp === id;
  return <span className={`info-wrap ${open ? 'open' : ''}`} onMouseLeave={() => open && setOpenHelp(null)}><button type="button" className="info-badge" aria-label={t("evidence.m_dd322d284113")} aria-expanded={open} onClick={(event) => {event.preventDefault(); event.stopPropagation(); setOpenHelp(open ? null : id);}}>{t("evidence.m_7d06b69aad65")}</button><span id={`${id}-content`} className="info-popover" role="tooltip"><b>{t("evidence.m_3ae78c1d5d3c")}</b><span>{text}</span><em>{example}</em>{technical && <small dir="ltr">{t("evidence.m_e2ee0cec1cb8")}{technical}</small>}</span></span>;
}

function renderJobReferencesBundle(job: RenderJob, bundleHash: string, basename?: string): boolean {
  const requestItems = Array.isArray(job.request?.items) ? job.request.items : [];
  return requestItems.some((value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
    if (basename && String(value.basename || '') === basename) return true;
    const artifact = value.bundleArtifact;
    if (!artifact || typeof artifact !== 'object' || Array.isArray(artifact)) return false;
    return String(artifact.ownerHash || '') === bundleHash;
  });
}

function laffaTags(item: BatchPreparedItem): string[] {
  const dimensions = item.dimensions || {};
  const tactical = String(dimensions.tactical_detector || '')
    .split('+')
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => candidateDimensionValue('tactical_detector', value));
  const contextual = ['contract', 'project', 'acting_seat'].flatMap((key) => {
    const value = dimensions[key];
    return value && value !== 'none' && value !== 'not_applicable'
      ? [candidateDimensionValue(key, value)]
      : [];
  });
  return [...new Set([...tactical, ...contextual])];
}

function renderJobReferencesUnifiedLaffa(job: RenderJob, item: UnifiedLaffa): boolean {
  if ((item.traceHash && renderJobReferencesBundle(job, item.traceHash)) || (item.gameHash && renderJobReferencesBundle(job, item.gameHash))) return true;
  return item.sourceReferences.some((source) => source.kind === 'research_result'
    ? renderJobReferencesResearch(job, source.code) : renderJobReferencesSource(
    job,
    source.kind === 'archived_game' || source.kind === 'sakka_round' || source.kind === 'research_audit_trace' ? 'laffa' : source.kind,
    source.code,
  ));
}

function unifiedLaffaQualificationLabel(item: UnifiedLaffa): string {
  if (item.qualificationStatus === 'qualified' && !item.publicationEligible) {
    return t('workflow.research_status_qualified');
  }
  return laffaQualificationLabel[item.qualificationStatus] || t('workflow.laffa_library_qualification_audit');
}

function UnifiedLaffaLibrary({page,filters,loading,busy,onFilters,onOpen,onPrepare}:{
  page:UnifiedLaffaPage|null;
  filters:UnifiedLaffaFilters;
  loading:boolean;
  busy:boolean;
  onFilters:(value:UnifiedLaffaFilters)=>void;
  onOpen:(item:UnifiedLaffa)=>void;
  onPrepare:(item:UnifiedLaffa)=>void;
}) {
  const [search, setSearch] = useState(filters.search);
  const setFilter = (key:keyof UnifiedLaffaFilters, value:string|number) => onFilters({...filters,[key]:value,offset:key==='offset'?Number(value):0});
  const sourceCount = new Map((page?.sourceFacets || []).map((item) => [item.sourceKind,item.count]));
  const options = {
    qualification:['qualified','rejected','provisional','quarantined','unqualified','legacy_replay'],
    truth:['exact_optimal','robust_best_known','debatable','provisional','rejected'],
    contract:['sun','hokum','ashkal'],
    project:['siri','fifty','hundred','hundred_sequence','hundred_four_kind','four_hundred'],
    tactic:['only_winning_move','small_card_big_gain','large_card_sacrifice','kabout','contract_rescue','entry_unblocking','trump_control','partner_enablement','project_bidding','debate'],
    seat:['north','east','south','west'],
    render:['not_prepared','prepared','queued','running','completed','failed','cancelled'],
  };
  const pageStart = page?.total ? page.offset + 1 : 0;
  const pageEnd = page ? page.offset + page.items.length : 0;
  return <>
    <form className="library-toolbar unified-laffa-toolbar" onSubmit={(event)=>{event.preventDefault();onFilters({...filters,search:search.trim(),offset:0});}}>
      <label className="library-search"><span>{t("workflow.laffa_library_search")}</span><input value={search} onChange={(event)=>setSearch(event.target.value)} placeholder={t("workflow.laffa_library_search")} /></label>
      <button type="submit" className="secondary" disabled={loading}>{t("workflow.laffa_library_apply")}</button>
      <label><span>{t("workflow.laffa_library_filter_source")}</span><select value={filters.sourceKind} onChange={(event)=>setFilter('sourceKind',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{(['archived_game','sakka_round','research_result','qualified_laffa','research_audit_trace'] as const).map((value)=><option key={value} value={value}>{laffaSourceLabel[value]} ({number.format(sourceCount.get(value)||0)})</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_filter_qualification")}</span><select value={filters.qualificationStatus} onChange={(event)=>setFilter('qualificationStatus',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.qualification.map((value)=><option key={value} value={value}>{value==='qualified'?t('workflow.research_status_qualified'):laffaQualificationLabel[value]}</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_filter_render")}</span><select value={filters.renderStatus} onChange={(event)=>setFilter('renderStatus',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.render.map((value)=><option key={value} value={value}>{laffaRenderLabel[value]}</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_filter_truth")}</span><select value={filters.truthLabel} onChange={(event)=>setFilter('truthLabel',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.truth.map((value)=><option key={value} value={value}>{truthLabel[value]}</option>)}</select></label>
      <label><span>{t("workflow.research_filter_contract")}</span><select value={filters.contract} onChange={(event)=>setFilter('contract',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.contract.map((value)=><option key={value} value={value}>{contractLabel[value]}</option>)}</select></label>
      <label><span>{t("workflow.research_filter_project")}</span><select value={filters.project} onChange={(event)=>setFilter('project',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.project.map((value)=><option key={value} value={value}>{projectLabel[value]}</option>)}</select></label>
      <label><span>{t("workflow.research_filter_tactic")}</span><select value={filters.tactic} onChange={(event)=>setFilter('tactic',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.tactic.map((value)=><option key={value} value={value}>{tacticDisplayName(value)}</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_filter_seat")}</span><select value={filters.seat} onChange={(event)=>setFilter('seat',event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{options.seat.map((value)=><option key={value} value={value}>{seatLabel[value]}</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_sort")}</span><select value={filters.sort} onChange={(event)=>setFilter('sort',event.target.value)}><option value="newest">{t("workflow.laffa_library_sort_newest")}</option><option value="oldest">{t("workflow.laffa_library_sort_oldest")}</option><option value="score_desc">{t("workflow.laffa_library_sort_score")}</option><option value="confidence_desc">{t("workflow.laffa_library_sort_confidence")}</option><option value="code_asc">{t("workflow.laffa_library_sort_code")}</option></select></label>
      <button type="button" className="quiet" onClick={()=>{setSearch('');onFilters(defaultLaffaFilters);}}>{t("workflow.laffa_library_clear")}</button>
    </form>
    <div className="library-count">{t("workflow.laffa_library_page_prefix")}{number.format(pageStart)}{t("workflow.laffa_library_page_middle")}{number.format(pageEnd)}{t("workflow.laffa_library_page_total")}{number.format(page?.total||0)}</div>
    <div className="data-table-wrap"><table className="data-table unified-laffa-table"><thead><tr>
      <th>{t("workflow.laffa_library_column_code")}</th><th>{t("workflow.laffa_library_column_source")}</th><th>{t("workflow.laffa_library_column_qualification")}</th><th>{t("workflow.laffa_library_column_laffa_score")}</th><th>{t("workflow.laffa_library_column_truth")}</th><th>{t("workflow.laffa_library_column_confidence")}</th><th>{t("workflow.research_label_contract")}</th><th>{t("workflow.research_label_project")}</th><th>{t("workflow.laffa_library_column_roles")}</th><th>{t("workflow.laffa_library_column_tags")}</th><th>{t("workflow.laffa_library_column_sakka")}</th><th>{t("workflow.laffa_library_column_video")}</th>
    </tr></thead><tbody>{(page?.items||[]).map((item)=><tr key={`${item.sourceKind}-${item.humanCode}`} tabIndex={0} onClick={()=>onOpen(item)} onKeyDown={(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();onOpen(item);}}}>
      <td><b dir="ltr">{item.humanCode}</b></td>
      <td><span className="table-tag">{laffaSourceLabel[item.sourceKind]}</span></td>
      <td><span className={`status ${item.qualificationStatus==='qualified'&&!item.publicationEligible?'neutral':item.qualificationStatus}`}>{unifiedLaffaQualificationLabel(item)}</span></td>
      <td>{item.challengeScore===null||item.challengeScore===undefined?'—':<ScorePill value={item.challengeScore}/>}</td>
      <td>{item.truthLabel?<span className="table-tag proof">{truthLabel[item.truthLabel]||item.truthLabel}</span>:'—'}</td>
      <td>{item.confidence===null||item.confidence===undefined?'—':`${number.format(Math.round(item.confidence*100))}${t("workflow.batch_percent_suffix")}`}</td>
      <td>{item.contract?contractLabel[item.contract]||item.contract:'—'}</td>
      <td>{item.projects.length?item.projects.map((value)=>projectLabel[value]||value).join(' · '):'—'}</td>
      <td><small>{item.buyer?`${t("workflow.research_label_buyer")}: ${seatLabel[item.buyer]||item.buyer}`:''}{item.dealer?` · ${t("workflow.sakka_library_dealer")}: ${seatLabel[item.dealer]||item.dealer}`:''}{item.seat?` · ${t("workflow.laffa_library_filter_seat")}: ${seatLabel[item.seat]||item.seat}`:''}</small></td>
      <td><div className="table-tags">{item.tacticalTags.slice(0,2).map((value)=><span key={value}>{tacticDisplayName(value)}</span>)}</div></td>
      <td>{item.relatedSakkas.length?<span dir="ltr">{item.relatedSakkas[0].code}{item.relatedSakkas.length>1?` +${number.format(item.relatedSakkas.length-1)}`:''}</span>:'—'}</td>
      <td onClick={(event)=>event.stopPropagation()}><div className="table-actions"><button type="button" className="table-status" onClick={()=>onOpen(item)}>{laffaRenderLabel[item.renderStatus]}</button>{item.replayValid&&['not_prepared','failed','cancelled'].includes(item.renderStatus)&&<button type="button" className="table-render" disabled={busy} onClick={()=>onPrepare(item)}>{item.publicationEligible?t("workflow.batch_render_selected"):t("workflow.research_render_audit")}</button>}</div></td>
    </tr>)}</tbody></table></div>
    {loading&&<p className="drawer-lead">{t("workflow.laffa_library_loading_more")}</p>}
    {!loading&&!page?.items.length&&<Empty title={t("workflow.laffa_library_no_matches")} body={t("workflow.laffa_library_all_generated_help")} />}
    <div className="library-pagination"><button type="button" className="secondary" disabled={loading||filters.offset===0} onClick={()=>setFilter('offset',Math.max(0,filters.offset-50))}>{t("workflow.laffa_library_previous")}</button><button type="button" className="secondary" disabled={loading||page?.nextOffset==null} onClick={()=>setFilter('offset',page?.nextOffset||0)}>{t("workflow.laffa_library_next")}</button></div>
  </>;
}

function UnifiedLaffaDetails({item,exactSource,renders,busy,onPrepare}:{item:UnifiedLaffa;exactSource?:JsonObject;renders:RenderJob[];busy:boolean;onPrepare:(item:UnifiedLaffa)=>void}) {
  const completed=renders.filter((job)=>job.status==='completed');
  const active=renders.find((job)=>['queued','running'].includes(job.status));
  const publishable=item.publicationEligible===true;
  const sourceDocument=exactSource?.document&&typeof exactSource.document==='object'&&!Array.isArray(exactSource.document)?exactSource.document as JsonObject:{};
  const preparedItem=sourceDocument.preparedItem&&typeof sourceDocument.preparedItem==='object'&&!Array.isArray(sourceDocument.preparedItem)?sourceDocument.preparedItem as JsonObject:{};
  const qualification=(exactSource?.qualification&&typeof exactSource.qualification==='object'&&!Array.isArray(exactSource.qualification)?exactSource.qualification:preparedItem.qualification) as JsonObject|undefined;
  const failures=[...new Set([
    ...(Array.isArray(item.qualificationFailures)?item.qualificationFailures.map(String):[]),
    ...(Array.isArray(qualification?.failures)?qualification.failures.map(String):[]),
  ])];
  return <>
    <div className="drawer-stats"><Stat label={t("workflow.laffa_library_column_laffa_score")} value={item.challengeScore===null||item.challengeScore===undefined?'—':number.format(Math.round(item.challengeScore))}/><Stat label={t("workflow.laffa_library_column_confidence")} value={item.confidence===null||item.confidence===undefined?'—':`${number.format(Math.round(item.confidence*100))}${t("workflow.batch_percent_suffix")}`}/><Stat label={t("workflow.laffa_library_column_video")} value={laffaRenderLabel[item.renderStatus]}/></div>
    <p className={`drawer-lead ${publishable?'':'audit-only'}`}>{publishable?t("workflow.laffa_library_content_candidate"):t(item.qualificationStatus==='qualified'?"workflow.laffa_library_qualification_audit":"workflow.laffa_library_audit_only")}</p>
    <div className="detail-tags">{item.tacticalTags.map((value)=><span key={value}>{tacticDisplayName(value)}</span>)}</div>
    <div className="evidence-list">
      <div><span>{t("workflow.laffa_library_column_source")}</span><b>{laffaSourceLabel[item.sourceKind]}</b></div>
      <div><span>{t("workflow.laffa_library_column_qualification")}</span><b>{unifiedLaffaQualificationLabel(item)}</b></div>
      <div><span>{t("workflow.research_label_contract")}</span><b>{item.contract?contractLabel[item.contract]||item.contract:'—'}</b></div>
      <div><span>{t("workflow.research_label_buyer")}</span><b>{item.buyer?seatLabel[item.buyer]||item.buyer:'—'}</b></div>
      <div><span>{t("workflow.sakka_library_dealer")}</span><b>{item.dealer?seatLabel[item.dealer]||item.dealer:'—'}</b></div>
      <div><span>{t("workflow.laffa_library_filter_seat")}</span><b>{item.seat?seatLabel[item.seat]||item.seat:'—'}</b></div>
    </div>
    {item.relatedSakkas.length>0&&<><h3>{t("workflow.laffa_library_related_sakkas")}</h3><div className="drawer-actions">{item.relatedSakkas.map((value)=><a key={value.code} className="button-link secondary" href={`/foundry?section=sakkas&sakka=${encodeURIComponent(value.code)}`}>{value.code}</a>)}</div></>}
    <h3>{t("workflow.laffa_library_exact_sources")}</h3>
    <div className="evidence-list">{item.sourceReferences.map((source)=><div key={`${source.kind}-${source.code}`}><span>{laffaSourceLabel[source.kind]}</span><b dir="ltr">{source.code}</b></div>)}</div>
    {failures.length>0&&<><h3>{t(item.qualificationStatus==='qualified'?"workflow.laffa_library_qualification_audit":"workflow.laffa_library_qualification_failures")}</h3><ul className="audit-list">{failures.map((value)=><li key={value}>{researchFailureLabel(value)}</li>)}</ul></>}
    <div className="drawer-actions">{item.replayValid&&!completed.length&&!active&&<button type="button" className="primary" disabled={busy} onClick={()=>onPrepare(item)}>{publishable?t("workflow.batch_render_selected"):t("workflow.research_render_audit")}</button>}{active&&<span className="status running">{laffaRenderLabel[item.renderStatus]}</span>}</div>
    {completed.length>0&&<><h3>{t("workflow.laffa_library_related_video")}</h3><div className="video-library-grid compact">{completed.map((job)=><RecentRenderCard key={job.jobId} job={job}/>)}</div></>}
  </>;
}

function LaffaLibraryTable({items, renders, onOpen, onRender, renderBusy, nextCursor, onLoadMore, loadingMore}: {
  items:BatchPreparedItem[];
  renders:RenderJob[];
  onOpen:(item:BatchPreparedItem)=>void;
  onRender:(item:BatchPreparedItem)=>void;
  renderBusy:boolean;
  nextCursor:string|null;
  onLoadMore:()=>void;
  loadingMore:boolean;
}) {
  const [search, setSearch] = useState('');
  const [truth, setTruth] = useState('all');
  const [classification, setClassification] = useState('all');
  const [groupBy, setGroupBy] = useState<'none'|'truth'|'classification'>('none');
  const truthValues = useMemo(() => [...new Set(items.map((item) => item.truthLabel || '').filter(Boolean))].sort(), [items]);
  const classificationValues = useMemo(() => [...new Set(items.map((item) => item.laffaClassification || '').filter(Boolean))].sort(), [items]);
  const filtered = useMemo(() => {
    const query = search.trim().toLocaleLowerCase('ar-SA');
    return items.filter((item) => {
      if (truth !== 'all' && item.truthLabel !== truth) return false;
      if (classification !== 'all' && item.laffaClassification !== classification) return false;
      if (!query) return true;
      const haystack = [
        item.shortCode,
        item.truthLabel ? truthLabel[item.truthLabel] || item.truthLabel : '',
        item.laffaClassification ? laffaLabel[item.laffaClassification] || item.laffaClassification : '',
        ...laffaTags(item),
        ...Object.values(item.dimensions || {}),
      ].join(' ').toLocaleLowerCase('ar-SA');
      return haystack.includes(query);
    });
  }, [classification, items, search, truth]);
  const columns = useMemo(() => laffaColumnHelper.columns([
    laffaColumnHelper.accessor('shortCode', {
      header: t("workflow.laffa_library_column_code"),
      cell: ({getValue}) => <b className="laffa-code" dir="ltr">{getValue()}</b>,
    }),
    laffaColumnHelper.accessor((item) => Number(item.selectionScore || 0), {
      id:'decisionScore',
      header:t("workflow.laffa_library_column_decision_score"),
      cell:({getValue}) => <ScorePill value={getValue()} />,
      sortDescFirst:true,
    }),
    laffaColumnHelper.accessor((item) => Number(item.laffaSelectionScore?.score || 0), {
      id:'laffaScore',
      header:t("workflow.laffa_library_column_laffa_score"),
      cell:({getValue}) => <ScorePill value={getValue()} tone="gold" />,
      sortDescFirst:true,
    }),
    laffaColumnHelper.accessor((item) => item.truthLabel || '', {
      id:'truthLabel',
      header:t("workflow.laffa_library_column_truth"),
      cell:({getValue}) => <span className="table-tag proof">{truthLabel[getValue()] || getValue()}</span>,
    }),
    laffaColumnHelper.accessor((item) => item.laffaClassification || '', {
      id:'classification',
      header:t("workflow.laffa_library_column_classification"),
      cell:({getValue}) => <span>{laffaLabel[getValue()] || getValue()}</span>,
    }),
    laffaColumnHelper.accessor((item) => Math.round(Number(item.confidence || 0) * 100), {
      id:'confidence',
      header:t("workflow.laffa_library_column_confidence"),
      cell:({getValue}) => <span dir="ltr">{number.format(getValue())}{t("workflow.batch_percent_suffix")}</span>,
      sortDescFirst:true,
    }),
    laffaColumnHelper.display({
      id:'tags',
      header:t("workflow.laffa_library_column_tags"),
      cell:({row}) => <div className="table-tags">{laffaTags(row.original).slice(0, 3).map((tag) => <span key={tag}>{tag}</span>)}</div>,
      enableSorting:false,
    }),
    laffaColumnHelper.display({
      id:'video',
      header:t("workflow.laffa_library_column_video"),
      cell:({row}) => {
        const related = renders.filter((job) => renderJobReferencesBundle(job, row.original.bundleHash, row.original.basename));
        const completed = related.some((job) => job.status === 'completed');
        const active = related.find((job) => ['queued','running'].includes(job.status));
        return <div className="table-actions" onClick={(event) => event.stopPropagation()}>
          <button type="button" className="table-status" onClick={() => onOpen(row.original)}>{completed ? t("workflow.laffa_library_video_ready") : active ? statusLabel[active.status] || active.status : t("workflow.laffa_library_video_not_generated")}</button>
          {!completed && !active && <button type="button" className="table-render" disabled={renderBusy} onClick={() => onRender(row.original)}>{t("workflow.batch_render_selected")}</button>}
        </div>;
      },
      enableSorting:false,
    }),
  ]), [onOpen, onRender, renderBusy, renders]);
  const table = useTable({
    features:laffaTableFeatures,
    columns,
    data:filtered,
    getRowId:(item) => item.bundleHash,
    initialState:{sorting:[{id:'laffaScore', desc:true}]},
  });
  const rows = table.getRowModel().rows;
  const grouped = new Map<string, typeof rows>();
  for (const row of rows) {
    const key = groupBy === 'truth'
      ? String(row.original.truthLabel || '')
      : groupBy === 'classification'
        ? String(row.original.laffaClassification || '')
        : '';
    const existing = grouped.get(key) || [];
    grouped.set(key, [...existing, row]);
  }
  const groupLabel = (value:string) => groupBy === 'truth'
    ? truthLabel[value] || value
    : groupBy === 'classification'
      ? laffaLabel[value] || value
      : '';
  return <>
    <div className="library-toolbar">
      <label className="library-search"><span>{t("workflow.laffa_library_search")}</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t("workflow.laffa_library_search")} /></label>
      <label><span>{t("workflow.laffa_library_filter_truth")}</span><select value={truth} onChange={(event) => setTruth(event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{truthValues.map((value) => <option key={value} value={value}>{truthLabel[value] || value}</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_filter_classification")}</span><select value={classification} onChange={(event) => setClassification(event.target.value)}><option value="all">{t("workflow.laffa_library_filter_all")}</option>{classificationValues.map((value) => <option key={value} value={value}>{laffaLabel[value] || value}</option>)}</select></label>
      <label><span>{t("workflow.laffa_library_group")}</span><select value={groupBy} onChange={(event) => setGroupBy(event.target.value as typeof groupBy)}><option value="none">{t("workflow.laffa_library_group_none")}</option><option value="truth">{t("workflow.laffa_library_group_truth")}</option><option value="classification">{t("workflow.laffa_library_group_classification")}</option></select></label>
    </div>
    <div className="library-count">{t("workflow.laffa_library_rows_prefix")}{number.format(rows.length)}{t("workflow.laffa_library_rows_middle")}{number.format(items.length)}{t("workflow.laffa_library_rows_suffix")}</div>
    <div className="data-table-wrap">
      <table className="data-table">
        <thead>{table.getHeaderGroups().map((headerGroup) => <tr key={headerGroup.id}>{headerGroup.headers.map((header) => <th key={header.id}><button type="button" disabled={!header.column.getCanSort()} onClick={header.column.getToggleSortingHandler()}><table.FlexRender header={header} />{header.column.getIsSorted() === 'asc' ? ' ↑' : header.column.getIsSorted() === 'desc' ? ' ↓' : ''}</button></th>)}</tr>)}</thead>
        <tbody>{[...grouped.entries()].map(([group, groupRows]) => <Fragment key={group || 'all'}>{groupBy !== 'none' && <tr className="data-group"><td colSpan={table.getVisibleLeafColumns().length}><b>{groupLabel(group)}</b><span>{number.format(groupRows.length)}</span></td></tr>}{groupRows.map((row) => <tr key={row.id} tabIndex={0} onClick={() => onOpen(row.original)} onKeyDown={(event) => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); onOpen(row.original);}}}>{row.getVisibleCells().map((cell) => <td key={cell.id}><table.FlexRender cell={cell} /></td>)}</tr>)}</Fragment>)}</tbody>
      </table>
    </div>
    {!rows.length && <Empty title={t("workflow.laffa_library_no_matches")} body={t("workflow.batch_library_help")} />}
    {nextCursor && <div className="library-more"><button type="button" className="secondary" disabled={loadingMore} onClick={onLoadMore}>{loadingMore ? t("workflow.laffa_library_loading_more") : t("workflow.laffa_library_load_more")}</button></div>}
  </>;
}

function ScorePill({value, tone = ''}: {value:number; tone?:string}) {
  return <span className={`score-pill ${tone}`}>{number.format(Math.round(value))}</span>;
}

function QualifiedLaffaDetails({item, renders, onRender, renderBusy}: {
  item:BatchPreparedItem;
  renders:RenderJob[];
  onRender:(item:BatchPreparedItem)=>void;
  renderBusy:boolean;
}) {
  const completed = renders.filter((job) => job.status === 'completed');
  const active = renders.find((job) => ['queued','running'].includes(job.status));
  const tags = laffaTags(item);
  return <>
    <div className="drawer-stats">
      <Stat label={t("evidence.label_decision_selection_score")} value={number.format(Math.round(Number(item.selectionScore || 0)))} />
      <Stat label={t("evidence.label_laffa_selection_score")} value={number.format(Math.round(Number(item.laffaSelectionScore?.score || 0)))} />
      <Stat label={t("workflow.laffa_library_column_confidence")} value={`${number.format(Math.round(Number(item.confidence || 0) * 100))}${t("workflow.batch_percent_suffix")}`} />
    </div>
    <p className="drawer-lead">{laffaLabel[item.laffaClassification || ''] || item.laffaClassification || t("workflow.batch_item_ready")}</p>
    {tags.length > 0 && <div className="detail-tags">{tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}
    <h3>{t("workflow.laffa_library_provenance")}</h3>
    <div className="evidence-list">
      <div><span>{t("workflow.laffa_library_column_truth")}</span><b>{truthLabel[item.truthLabel || ''] || item.truthLabel}</b></div>
      <div><span>{t("workflow.laffa_library_column_classification")}</span><b>{laffaLabel[item.laffaClassification || ''] || item.laffaClassification}</b></div>
      {item.dimensions?.acting_seat&&<div><span>{t("workflow.research_filter_acting_seat")}</span><b>{seatLabel[item.dimensions.acting_seat]||item.dimensions.acting_seat}</b></div>}
      {item.observerSeat&&<div><span>{t("workflow.research_filter_observer_seat")}</span><b>{seatLabel[item.observerSeat]||item.observerSeat}</b></div>}
      {item.bestAction && <div><span>{t("evidence.m_27fde21d5951")}</span><b dir="ltr">{actionLabel(item.bestAction)}</b></div>}
    </div>
    <div className="drawer-actions">
      <a className="button-link secondary" href={`/c/${encodeURIComponent(item.shortCode)}`}>{t("workflow.laffa_library_play_challenge")}</a>
      {!completed.length && !active && <button type="button" className="primary" disabled={renderBusy} onClick={() => onRender(item)}>{t("workflow.batch_render_selected")}</button>}
      {active && <span className="status running">{statusLabel[active.status] || active.status}</span>}
    </div>
    {completed.length > 0 && <><h3>{t("workflow.laffa_library_related_video")}</h3><div className="video-library-grid compact">{completed.map((job) => <RecentRenderCard key={job.jobId} job={job} />)}</div></>}
  </>;
}

function VideoBatchDetails({job, onRender, onOpen, renderBusy}: {job: VideoProductionBatch; onRender:(item:BatchPreparedItem)=>void; onOpen:(item:BatchPreparedItem)=>void; renderBusy:boolean}) {
  const result = job.result || {};
  const rejections = Array.isArray(result.rejections) ? result.rejections : [];
  const rejectionSummaries = aggregateVideoBatchRejections(rejections);
  const preparedItems = Array.isArray(result.preparedItems) ? result.preparedItems : [];
  const metrics = result.operationalMetrics && typeof result.operationalMetrics === 'object' && !Array.isArray(result.operationalMetrics) ? result.operationalMetrics as JsonObject : {};
  const laneMetrics = metrics.laneMetrics && typeof metrics.laneMetrics === 'object' && !Array.isArray(metrics.laneMetrics) ? metrics.laneMetrics as JsonObject : {};
  const stage = String(result.stage || job.status);
  return <>
    <div className="drawer-stats">
      <Stat label={t("workflow.batch_required")} value={number.format(Number(job.request.targetVideos || 0))} />
      <Stat label={t("workflow.batch_rounds")} value={number.format(Number(result.roundsGenerated || 0))} />
      <Stat label={t("workflow.batch_qualified")} value={number.format(Number(result.finalistsAccepted || 0))} />
      <Stat label={t("workflow.batch_prepared")} value={number.format(Number(result.preparedVideoCount || 0))} />
      <Stat label={t("workflow.batch_ready")} value={number.format(Number(result.completedVideoCount || 0))} />
    </div>
    <p className="drawer-lead"><b>{t("workflow.batch_current_stage")}:</b> {videoBatchStageLabel[stage] || statusLabel[job.status] || job.status}</p>
    <p className="drawer-lead">{job.status === 'completed' ? t("workflow.batch_completed_help") : job.status === 'exhausted' ? t("workflow.batch_exhausted_help") : t("workflow.batch_running_help")}</p>
    <div className="evidence-list">
      <div><span>{t("workflow.batch_campaign_count")}</span><b>{number.format(Array.isArray(result.campaigns) ? result.campaigns.length : 0)}</b></div>
      <div><span>{t("workflow.batch_candidate_count")}</span><b>{number.format(Number(result.retainedCandidates || 0))}</b></div>
      <div><span>{t("workflow.batch_rejected")}</span><b>{number.format(Number(result.finalistsRejected || 0))}</b></div>
      <div><span>{t("workflow.batch_render_failures")}</span><b>{number.format(Number(result.failedVideoCount || 0))}</b></div>
      <div><span>{t("workflow.batch_yield_per_100")}</span><b>{number.format(Number(metrics.acceptedPer100Rounds || 0))}</b></div>
      <div><span>{t("workflow.batch_yield_per_hour")}</span><b>{number.format(Number(metrics.acceptedPerWallHour || 0))}</b></div>
    </div>
    {Object.keys(laneMetrics).length > 0 && <details className="audit-block"><summary>{t("workflow.batch_lane_breakdown")}</summary><ul>{Object.entries(laneMetrics).map(([lane, value]) => {const laneValue = value && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {}; return <li key={lane}><b>{sourceLabel[lane] || lane}</b><span>{t("workflow.batch_lane_rounds_prefix")}{number.format(Number(laneValue.roundsGenerated || 0))}{t("workflow.batch_lane_accepted_prefix")}{number.format(Number(laneValue.finalistsAccepted || 0))}</span></li>;})}</ul></details>}
    {preparedItems.length > 0 && <BatchPreparedList items={preparedItems} onRender={onRender} onOpen={onOpen} renderBusy={renderBusy} />}
    {rejectionSummaries.length > 0 && <details className="audit-block"><summary>{t("workflow.batch_rejection_details")}</summary><ul>{rejectionSummaries.map((item) => {const rejectionStage=String(item.stage || ''); const reasonCode=String(item.reasonCode || ''); return <li key={`${rejectionStage}-${reasonCode}`}><b>{videoBatchRejectionStageLabel[rejectionStage] || t("workflow.batch_rejection_generic")}</b><span>{videoBatchRejectionText(item)}</span></li>;})}</ul></details>}
  </>;
}
function BatchPreparedList({items, onRender, onOpen, renderBusy}: {items:BatchPreparedItem[]; onRender:(item:BatchPreparedItem)=>void; onOpen:(item:BatchPreparedItem)=>void; renderBusy:boolean}) {
  return <div className="batch-ready-list">{items.map((item, index) => <article key={item.bundleHash || item.candidateHash}>
    <div><span>{t("workflow.batch_item_prefix")}{number.format(index + 1)}</span><b dir="ltr">{item.shortCode || number.format(index + 1)}</b><small>{t("workflow.batch_item_ready")}</small></div>
    <div className="candidate-badges">
      {Number.isFinite(item.selectionScore) && <span>{t("evidence.label_decision_selection_score")}: {number.format(Math.round(Number(item.selectionScore)))}</span>}
      {Number.isFinite(item.laffaSelectionScore?.score) && <span>{t("evidence.label_laffa_selection_score")}: {number.format(Math.round(Number(item.laffaSelectionScore?.score)))}</span>}
      {item.truthLabel && <span>{truthLabel[item.truthLabel] || item.truthLabel}</span>}
      {Number.isFinite(item.confidence) && <span>{t("workflow.m_aaf0eb9937a4")}{number.format(Math.round(Number(item.confidence) * 100))}{t("workflow.m_a489ae36a167")}</span>}
    </div>
    <div className="table-actions"><button className="secondary" onClick={() => onOpen(item)}>{t("workflow.laffa_library_open")}</button><button className="primary" disabled={renderBusy} onClick={() => onRender(item)}>{t("workflow.batch_render_selected")}</button></div>
  </article>)}</div>;
}
function CampaignDetails({campaign, shards, attempts}: {campaign:Campaign; shards:Shard[]; attempts:Attempt[]}) {
  return <><div className="drawer-stats"><Stat label={t("evidence.m_c3a4749caed4")} value={statusLabel[campaign.status] || campaign.status} /><Stat label={t("evidence.m_bd51a8b5fb50")} value={number.format(shards.length)} /><Stat label={t("evidence.m_a6bf7492ecdd")} value={number.format(attempts.length)} /></div><h3>{t("evidence.m_1e2e5a8ecae5")}</h3><div className="shard-track">{shards.map((shard) => <i key={shard.specHash} className={shard.status} title={t("evidence.m_4f097b521885") + number.format(shard.shardIndex + 1) + ': ' + (statusLabel[shard.status] || shard.status)} />)}</div><p className="drawer-lead">{t("evidence.m_1e5a869eacb7")}</p></>;
}
function SakkaLibraryTable({items, names, onOpen}: {items:FoundrySakka[]; names:Map<string,string>; onOpen:(item:FoundrySakka)=>void}) {
  const [query, setQuery] = useState('');
  const [provenance, setProvenance] = useState('all');
  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase('ar-SA');
    return items.filter((item) => {
      const itemProvenance = String(item.metadata.provenance || '');
      if (provenance !== 'all' && itemProvenance !== provenance) return false;
      if (!needle) return true;
      const score = Array.isArray(item.metadata.finalScore) ? item.metadata.finalScore.join(' ') : '';
      return [names.get(item.catalogHash), item.catalogHash.slice(0, 8), item.metadata.winner, score]
        .join(' ').toLocaleLowerCase('ar-SA').includes(needle);
    });
  }, [items, names, provenance, query]);
  const table = useTable({
    features: sakkaTableFeatures,
    columns: useMemo(() => sakkaColumnHelper.columns([
      sakkaColumnHelper.accessor((item) => names.get(item.catalogHash) || item.catalogHash.slice(0, 8).toUpperCase(), {
        id: 'code', header: t("workflow.sakka_library_column_code"),
        cell: ({getValue}) => <b className="laffa-code" dir="ltr">{getValue()}</b>,
      }),
      sakkaColumnHelper.accessor((item) => Number(item.metadata.roundCount || 0), {
        id: 'rounds', header: t("workflow.sakka_library_column_rounds"),
        cell: ({getValue}) => number.format(getValue()), sortDescFirst: true,
      }),
      sakkaColumnHelper.accessor((item) => Math.max(...(item.metadata.finalScore || [0])), {
        id: 'score', header: t("workflow.sakka_library_column_score"),
        cell: ({row}) => <b dir="ltr">{row.original.metadata.finalScore?.join(' — ') || '—'}</b>,
        sortDescFirst: true,
      }),
      sakkaColumnHelper.accessor((item) => String(item.metadata.winner || ''), {
        id: 'winner', header: t("workflow.sakka_library_column_winner"),
        cell: ({getValue}) => getValue() === 'north_south' ? t("workflow.sakka_library_team_us") : t("workflow.sakka_library_team_them"),
      }),
      sakkaColumnHelper.accessor((item) => String(item.metadata.provenance || ''), {
        id: 'provenance', header: t("workflow.sakka_library_column_type"),
        cell: ({getValue}) => <span className="table-tag proof">{getValue() === 'natural_sakka' ? t("workflow.sakka_library_natural") : t("workflow.sakka_library_curated")}</span>,
      }),
    ]), [names]),
    data: filtered,
    getRowId: (item) => item.catalogHash,
    initialState: {sorting: [{id:'rounds', desc:true}]},
  });
  const rows = table.getRowModel().rows;
  return <>
    <div className="library-toolbar sakka-toolbar">
      <label className="library-search"><span>{t("workflow.sakka_library_search")}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("workflow.sakka_library_search")} /></label>
      <label><span>{t("workflow.sakka_library_filter_type")}</span><select value={provenance} onChange={(event) => setProvenance(event.target.value)}><option value="all">{t("workflow.sakka_library_filter_all")}</option><option value="natural_sakka">{t("workflow.sakka_library_natural")}</option><option value="curated_sakka">{t("workflow.sakka_library_curated")}</option></select></label>
    </div>
    <div className="library-count">{t("workflow.sakka_library_rows_prefix")}{number.format(rows.length)}{t("workflow.sakka_library_rows_middle")}{number.format(items.length)}{t("workflow.sakka_library_rows_suffix")}</div>
    <div className="data-table-wrap"><table className="data-table sakka-table"><thead>{table.getHeaderGroups().map((group) => <tr key={group.id}>{group.headers.map((header) => <th key={header.id}><button type="button" disabled={!header.column.getCanSort()} onClick={header.column.getToggleSortingHandler()}><table.FlexRender header={header} />{header.column.getIsSorted() === 'asc' ? ' ↑' : header.column.getIsSorted() === 'desc' ? ' ↓' : ''}</button></th>)}</tr>)}</thead><tbody>{rows.map((row) => <tr key={row.id} tabIndex={0} onClick={() => onOpen(row.original)} onKeyDown={(event) => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); onOpen(row.original);}}}>{row.getVisibleCells().map((cell) => <td key={cell.id}><table.FlexRender cell={cell} /></td>)}</tr>)}</tbody></table></div>
    {!rows.length && <Empty title={t("workflow.sakka_library_no_matches")} body={t("settings.m_13ea0913a562")} />}
  </>;
}
function SakkaDetails({detail, renders, onOpenRound, onRenderRound, renderBusy}: {detail:FoundrySakkaDetail; renders:RenderJob[]; onOpenRound:(traceHash:string)=>void; onRenderRound:(traceHash:string)=>void; renderBusy:boolean}) {
  const sakka = detail.document.sakka || {};
  const rounds = Array.isArray(sakka.rounds) ? sakka.rounds as FoundrySakkaRound[] : [];
  const score = Array.isArray(sakka.finalScore) ? sakka.finalScore.join(' — ') : '—';
  return <>
    <div className="drawer-stats"><Stat label={t("evidence.m_421e2b93641c")} value={number.format(rounds.length)} /><Stat label={t("evidence.m_3c64ba0afab6")} value={score} /><Stat label={t("workflow.sakka_library_target")} value={number.format(Number(sakka.target || detail.record.metadata.target || 0))} /></div>
    <div className="context-proof"><i>✓</i><div><b>{t("workflow.sakka_library_context_verified")}</b><span>{t("workflow.sakka_library_context_verified_help")}</span></div></div>
    <h3>{t("workflow.sakka_library_score_path")}</h3>
    <div className="sakka-rounds">{rounds.map((round) => {
      const code = `LAF-${round.traceHash.slice(0,10).toUpperCase()}`;
      const relatedRenders = renders.filter((job) => renderJobReferencesSource(job, 'laffa', code));
      return <article key={round.traceHash}>
        <header><div><span>{t("workflow.sakka_library_round_prefix")}{number.format(round.ordinal)}</span><b dir="ltr">{round.startingScore.join(' — ')} <i>←</i> {round.finalScore.join(' — ')}</b></div><span className="dealer-pill">{t("workflow.sakka_library_dealer")}: {seatLabel[round.dealer] || round.dealer}</span></header>
        <div className="round-flow"><span>{t("workflow.sakka_library_awarded")}</span><b dir="ltr">+{round.awardedPoints.join(' / +')}</b><small>{round.allPassCount ? t("workflow.sakka_library_all_pass_prefix") + number.format(round.allPassCount) : t("workflow.sakka_library_no_all_pass")}</small></div>
        <div className="round-actions"><button type="button" className="secondary" onClick={() => onOpenRound(round.traceHash)}>{t("workflow.laffa_library_open")}</button>{relatedRenders.some((job) => job.status === 'completed') ? <span className="status completed">{t("workflow.laffa_library_video_ready")}</span> : <button type="button" className="table-render" disabled={renderBusy} onClick={() => onRenderRound(round.traceHash)}>{t("workflow.batch_render_selected")}</button>}<span className="status neutral">{t("workflow.laffa_library_qualification_audit")}</span></div>
      </article>;
    })}</div>
  </>;
}
function coverageTargetLabel(target: string) {
  const labels: Record<string, string> = {reference:t("settings.coverage_target_reference"), runtime:t("settings.coverage_target_runtime"), cpSat:t("settings.coverage_target_cpsat"), policy:t("settings.coverage_target_policy"), documentation:t("settings.coverage_target_documentation"), tests:t("settings.coverage_target_tests"), influence:t("settings.coverage_target_influence")};
  return labels[target] || target;
}
function coverageCategoryLabel(category: string) {
  const labels: Record<string, string> = {semantic_rule:t("settings.coverage_category_semantic_rule"), structural:t("settings.coverage_category_structural"), transition:t("settings.coverage_category_transition"), legality:t("settings.coverage_category_legality"), scoring:t("settings.coverage_category_scoring"), information:t("settings.coverage_category_information"), convention:t("settings.coverage_category_convention"), tactic:t("settings.coverage_category_tactic"), presentation:t("settings.coverage_category_presentation"), policy:t("settings.coverage_category_policy")};
  return labels[category] || category;
}
function coverageBehaviorStatusLabel(status: string) {
  const labels: Record<string, string> = {
    proven_behavioral:t("settings.compiler_coverage_behavior_proven"),
    proven_invariant:t("settings.compiler_coverage_invariant_proven"),
    partial_behavioral:t("settings.compiler_coverage_behavior_partial"),
    unproven:t("settings.compiler_coverage_behavior_unproven"),
  };
  return labels[status] || status;
}
function CompilerCoverageDetails({coverage}: {coverage:CompilerCoverageSummary}) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const categories = useMemo(() => [...new Set(coverage.rows.map((row) => row.category))].sort((left, right) => coverageCategoryLabel(left).localeCompare(coverageCategoryLabel(right), 'ar')), [coverage.rows]);
  const filtered = useMemo(() => coverage.rows.filter((row) => {
    if (category !== 'all' && row.category !== category) return false;
    const text = `${row.primitiveId} ${row.definitionKind} ${row.missingTargets.join(' ')}`.toLocaleLowerCase('en');
    return !query.trim() || text.includes(query.trim().toLocaleLowerCase('en'));
  }), [coverage.rows, category, query]);
  const visible = filtered.slice(0, 200);
  return <>
    <div className="drawer-stats"><div><span>{t("settings.compiler_coverage_percent")}</span><b>{number.format(coverage.percentage)}{t("workflow.batch_percent_suffix")}</b></div><div><span>{t("settings.compiler_coverage_binding")}</span><b>{number.format(coverage.executableBindingCovered)} / {number.format(coverage.executableBindingTotal)}</b></div><div><span>{t("settings.compiler_coverage_behavioral")}</span><b>{number.format(coverage.behavioralOrInvariantCovered)} / {number.format(coverage.behavioralOrInvariantTotal)}</b></div><div><span>{t("settings.compiler_coverage_remaining")}</span><b>{number.format(Math.max(0, coverage.total - coverage.covered))}</b></div></div>
    <p className="drawer-lead">{coverage.complete ? t("settings.compiler_coverage_ready_detail") : t("settings.compiler_coverage_blocked_detail")}</p>
    <div className="coverage-filters"><label><span>{t("settings.compiler_coverage_search")}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("settings.compiler_coverage_search_example")} /></label><label><span>{t("settings.compiler_coverage_category")}</span><select value={category} onChange={(event) => setCategory(event.target.value)}><option value="all">{t("settings.compiler_coverage_all_categories")}</option>{categories.map((item) => <option key={item} value={item}>{coverageCategoryLabel(item)}</option>)}</select></label></div>
    <div className="coverage-result-count">{number.format(filtered.length)} {t("settings.compiler_coverage_matching")}{filtered.length > visible.length ? ` · ${t("settings.compiler_coverage_first")} ${number.format(visible.length)}` : ''}</div>
    <div className="coverage-list">{visible.map((row) => <article key={row.primitiveId} className={row.qualificationEligible ? 'complete' : 'incomplete'}><div><span>{coverageCategoryLabel(row.category)}</span><code>{row.primitiveId}</code></div><div className="coverage-missing"><b className={row.executableBindingStatus === 'covered' ? 'done' : ''}>{row.executableBindingStatus === 'covered' ? t("settings.compiler_coverage_binding_ready") : t("settings.compiler_coverage_binding_missing")}</b><b className={row.qualificationEligible ? 'done' : ''}>{coverageBehaviorStatusLabel(row.behavioralOrInvariantStatus)}</b>{row.missingTargets.filter((target) => target !== 'influence').map((target) => <b key={target}>{coverageTargetLabel(target)}</b>)}</div></article>)}</div>
  </>;
}
function ConfigBundleDetails({bundle, profiles}: {bundle:ConfigBundle; profiles:ConfigProfile[]}) {
  return <div className="bundle-components">{Object.entries(bundle.componentRevisions).map(([family, revision]) => {const profile=profiles.find((item)=>item.revisionId===revision); return <div key={family}><span>{familyMeta[family]?.label || t("evidence.m_f224d86b5a90")}</span><b>{t("evidence.m_26dacc70780e")}</b><small>{t("evidence.m_f14158b9c061")}{profile?.version || '—'}</small></div>;})}</div>;
}
function ConfigComparison({value}: {value:JsonObject}) {
  const changes = Array.isArray(value.changes) ? value.changes as JsonObject[] : [];
  return <>{changes.length ? <div className="comparison-list">{changes.map((item,index) => <div key={`${String(item.path)}-${index}`}><span>{friendlyPath(String(item.path || ''))}</span><b>{item.change === 'added' ? t("evidence.m_eb6c834d28a1") : item.change === 'removed' ? t("evidence.m_677f12d42deb") : t("evidence.m_4dccf6155814")}</b></div>)}</div> : <Empty title={t("evidence.m_ca3550efa66f")} body={t("evidence.m_8c064d6ceff3")} />}</>;
}
function JobProgress({title, status, progress, error, onDetails}: {title:string; status:string; progress:number; error?:string|null; onDetails:()=>void}) {
  return <button className="job-progress" onClick={onDetails}><div><span>{title}</span><b>{statusLabel[status] || status}</b></div><div className="progress"><i style={{transform:`scaleX(${Math.max(0,Math.min(1,progress))})`}} /></div>{error && <small>{friendlyJobError(error)}</small>}</button>;
}
function JsonSummary({value}: {value:JsonObject}) { return <div className="json-summary">{Object.entries(value).slice(0,20).map(([key,item]) => <div key={key}><span>{humanKey(key)}</span><b>{typeof item === 'object' ? t("evidence.m_032f46c959bf") : String(item)}</b></div>)}</div>; }
function Stat({label, value}: {label:string; value:string}) { return <div><span>{label}</span><b>{value}</b></div>; }
function Empty({title, body}: {title:string; body:string}) { return <div className="empty-state"><b>{title}</b><span>{body}</span></div>; }
function Drawer({state, onClose}: {state:NonNullable<DrawerState>; onClose:()=>void}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);
  const close = () => dialogRef.current?.close();
  return <dialog ref={dialogRef} className="drawer-layer" aria-modal="true" aria-labelledby="drawer-title" onClose={onClose} onMouseDown={(event) => event.target === event.currentTarget && close()}><aside className="drawer"><header><div><span>{state.subtitle}</span><h2 id="drawer-title">{state.title}</h2></div><button type="button" autoFocus onClick={close} aria-label={t("evidence.m_45c61be18c6c")}>×</button></header><div className="drawer-body">{state.content}{state.audit && <details className="audit-details"><summary>{t("evidence.m_97b7815c26dc")}</summary><pre dir="ltr">{JSON.stringify(state.audit,null,2)}</pre></details>}</div></aside></dialog>;
}

createRoot(document.getElementById('root')!).render(<App />);
