import {
  type ComponentType,
  type CSSProperties,
  useEffect,
  useState,
} from 'react';

import {
  getResearchResultQualityAudit,
  getResearchResultOperatorAudit,
  getScenarioResearchLocalization,
} from '../../api/research';

import {
  isSeatName,
  type JsonObject,
  type JsonValue,
  type SeatName,
} from '../../shared/json';
import {number, t, tVariant} from '../../shared/localization';
import {
  actionLabel,
  contractLabel,
  decisionLabel,
  JsonSummary,
  laffaQualificationLabel,
  laffaSourceLabel,
  seatLabel,
  scenarioStatusLabel,
  researchRunStatusLabel,
  Stat,
  statusLabel,
  tacticDisplayName,
  truthLabel,
} from '../../shared/presentation';
import type {
  ResearchCandidateProgressClassification,
  ResearchCandidateProgressReceipt,
  ResearchPreparedRender,
  ResearchRecipe,
  ResearchPlayingQualityAuditV1,
  ResearchResult,
  ResearchResultOperatorAuditV1,
  ResearchRun,
  ResearchRunPerformanceReceipt,
  ResearchSequentialSearchReceiptPage,
  ResearchSequentialSearchSummary,
  ResearchStats,
  SemanticDisagreement,
  ScenarioResearchLocalizationV1,
} from './model';
import {isOrToolsResearchBackend} from './model';
import {continuationLineBoundaryStatus} from './ortoolsScenarioUi';

type ResearchPerformanceSummaryProps = {
  stats: ResearchStats;
  onRuns: () => void;
  onResults: () => void;
};

function ResearchSequentialSearchCounters({
  value,
}: {
  value: ResearchSequentialSearchSummary;
}) {
  return <div className="evidence-list research-sequential-counters">
    <div><span>{t('workflow.research_sequential_admitted_seeds')}</span><b>{number.format(value.admittedSeedCount)}</b></div>
    <div><span>{t('workflow.research_sequential_exact_rejections')}</span><b>{number.format(value.exactRejectionCount)}</b></div>
    <div><span>{t('workflow.research_sequential_no_goods_applied')}</span><b>{number.format(value.noGoodAppliedCount)}</b></div>
    <div><span>{t('workflow.research_sequential_no_goods_unapplied')}</span><b>{number.format(value.noGoodUnappliedCount)}</b></div>
    <div><span>{t('workflow.research_sequential_continuation_solves')}</span><b>{number.format(value.continuationSolveCount)}</b></div>
    <div><span>{t('workflow.research_sequential_continuation_optimal')}</span><b>{number.format(value.continuationStatusDistribution.optimal)}</b></div>
    <div><span>{t('workflow.research_sequential_continuation_infeasible')}</span><b>{number.format(value.continuationStatusDistribution.infeasible)}</b></div>
    <div><span>{t('workflow.research_sequential_stop_candidates_exhausted')}</span><b>{number.format(value.stopReasonDistribution.candidates_exhausted)}</b></div>
    <div><span>{t('workflow.research_sequential_stop_maximum_solutions')}</span><b>{number.format(value.stopReasonDistribution.maximum_solutions_reached)}</b></div>
    <div><span>{t('workflow.research_sequential_stop_qualified')}</span><b>{number.format(value.stopReasonDistribution.qualified)}</b></div>
    <div><span>{t('workflow.research_sequential_stop_incomplete')}</span><b>{number.format(value.stopReasonDistribution.incomplete)}</b></div>
    <div><span>{t('workflow.research_sequential_stop_cancelled')}</span><b>{number.format(value.stopReasonDistribution.cancelled)}</b></div>
    <div><span>{t('workflow.research_sequential_stop_fatal')}</span><b>{number.format(value.stopReasonDistribution.fatal_quarantine)}</b></div>
  </div>;
}

export type LoadResearchPerspectives = (
  resultCode: string,
  observerSeat: SeatName,
) => Promise<{items: JsonObject[]}>;

function proofScopeLabel(scope: string, operatorCopy: (key: string) => string): string | null {
  switch (scope) {
    case 'exact': return operatorCopy('research_scope_exact');
    case 'bounded': return operatorCopy('research_scope_bounded');
    case 'exact_fixed_deal_terminal_tail': return operatorCopy('research_scope_exact_fixed_deal_terminal_tail');
    case 'selected_bid_current_deal': return operatorCopy('research_scope_selected_bid_current_deal');
    case 'selected_escalation_current_deal': return operatorCopy('research_scope_selected_escalation_current_deal');
    case 'selected_final_project_complete_state': return operatorCopy('research_scope_selected_final_project_complete_state');
    case 'supplied_worlds_only': return operatorCopy('research_scope_supplied_worlds_only');
    case 'action_optimality_only': return operatorCopy('research_scope_action_optimality_only');
    case 'selected_play_fixed_state_terminal_continuation': return operatorCopy('research_scope_selected_play_fixed_state_terminal_continuation');
    case 'observer_safe_policy_conditioned': return operatorCopy('research_scope_observer_safe_policy_conditioned');
    default: return null;
  }
}

function tailTimingLabel(stage: string, operatorCopy: (key: string) => string): string | null {
  switch (stage) {
    case 'tailModelBuild': return operatorCopy('research_scenario_timing_tail_model_build');
    case 'tailModelConfigure': return operatorCopy('research_scenario_timing_tail_model_configure');
    case 'tailSolver': return operatorCopy('research_scenario_timing_tail_solver');
    default: return null;
  }
}

export function researchFailureLabel(code: string) {
  const labels: Record<string, string> = {
    publication_eligibility_not_established: t('workflow.laffa_library_qualification_audit'),
    incomplete_decision_audit: t('workflow.research_failure_incomplete_audit'),
    truth_below_required_tier: t('workflow.research_failure_truth'),
    confidence_below_gate: t('workflow.research_failure_confidence'),
    surrounding_agent_regret: t('workflow.research_failure_regret'),
    surrounding_quality_below_gate: t('workflow.research_failure_surrounding_quality'),
    no_verified_critical_point: t('workflow.research_failure_no_critical_point'),
    critical_decision_not_proven: t('workflow.research_failure_no_critical_point'),
    featured_play_native_proof_unavailable: t('workflow.research_failure_observer_information'),
    observer_information_proof_incomplete: t('workflow.research_failure_observer_information'),
    sequential_response_proof_incomplete: t('workflow.research_failure_sequential_response'),
  };
  if (code.startsWith('tactical_gate:')) {
    return `${t('workflow.research_failure_tactic')} · ${tacticDisplayName(code.slice('tactical_gate:'.length))}`;
  }
  return labels[code] || code;
}

function researchRunDuration(run: ResearchRun) {
  const events = Array.isArray(run.events) ? run.events : [];
  const started = events.find((event) => String(event.type) === 'started')?.at
    || run.createdAt;
  const ended = [...events]
    .reverse()
    .find((event) => ['completed', 'failed', 'cancelled'].includes(String(event.type)))
    ?.at;
  const seconds = (Date.parse(String(ended || '')) - Date.parse(String(started || ''))) / 1000;
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

function performanceReceipt(run: ResearchRun): ResearchRunPerformanceReceipt | null {
  const value = run.performanceReceipt;
  if (
    !value
    || value.measurementScope !== 'research_v4_completed_run_v1'
    || value.runCode !== run.runCode
    || !/^[0-9a-f]{64}$/.test(String(value.sha256 ?? ''))
  ) return null;
  return value;
}

function receiptValue(value: unknown) {
  return value === null || value === undefined ? t('workflow.research_metric_unavailable') : String(value);
}

function ResearchContinuationLineBoundary({value}: {value: JsonObject}) {
  const lines = value.continuationLines && typeof value.continuationLines === 'object'
    && !Array.isArray(value.continuationLines)
    ? value.continuationLines as JsonObject
    : null;
  if (!lines) return null;
  const status = continuationLineBoundaryStatus(value, lines);
  if (status === null) return null;
  const complete = lines.complete === true;
  return <article aria-live="polite" className={`research-qualification-summary research-proof-boundary ${complete ? 'qualified' : 'rejected'}`}>
    <header><b>{t('workflow.research_scenario_claim_winning_continuation_line_count')}</b><code dir="ltr">{status}</code></header>
    <div className="drawer-stats">
      {complete
        ? <Stat label={t('workflow.research_scenario_claim_winning_continuation_line_count')} value={String(lines.exactCount ?? '—')}/>
        : <><Stat label={t('workflow.research_scenario_count_minimum')} value={String(lines.lowerBound ?? '0')}/><Stat label={t('workflow.research_scenario_count_maximum')} value={receiptValue(lines.upperBound)}/></>}
    </div>
    <p>{t('workflow.research_scenario_proof_boundary')} <b dir="ltr">{String(value.boundaryEvaluator || 'terminal_utility')}</b></p>
  </article>;
}

function ResearchWinningRootActionBoundary({value}: {value: JsonObject}) {
  const exactCount = value.winningRootActionCount;
  return exactCount !== null && exactCount !== undefined
    ? <Stat label={t('workflow.research_scenario_winning_actions')} value={number.format(Number(exactCount))}/>
    : <>
      <Stat label={t('workflow.research_scenario_winning_actions_minimum')} value={number.format(Number(value.winningRootActionLowerBound ?? 0))}/>
      <Stat label={t('workflow.research_scenario_winning_actions_maximum')} value={receiptValue(value.winningRootActionUpperBound)}/>
    </>;
}

function ResearchPerformanceReceipt({receipt}: {receipt: ResearchRunPerformanceReceipt}) {
  const funnel = receipt.funnel;
  const proof = receipt.proofResources;
  return <section className="research-performance-receipt">
    <header><div><h4>{t('workflow.research_run_performance_receipt')}</h4><p>{t('workflow.research_run_performance_receipt_help')}</p></div><code dir="ltr">{receipt.sha256}</code></header>
    <div className="evidence-list research-performance-receipt-grid">
      <div><span>{t('workflow.research_run_execution_kind')}</span><code dir="ltr">{receipt.executionKind}</code></div>
      <div><span>{t('workflow.research_generation_lane')}</span><code dir="ltr">{receipt.generationLane}</code></div>
      <div><span>{t('workflow.research_run_attempted_seeds')}</span><b>{String(receipt.attemptedSeeds)}</b></div>
      <div><span>{t('workflow.research_run_attempted_candidates')}</span><b>{String(receipt.attemptedCandidates)}</b></div>
      <div><span>{t('workflow.research_run_structural_candidates')}</span><b>{String(receipt.structuralCandidates)}</b></div>
      <div><span>{t('workflow.research_scenario_estimate_proof')}</span><b>{String(receipt.proofCompleted)}</b></div>
      <div><span>{t('workflow.research_run_results')}</span><b>{String(receipt.resultCount)}</b></div>
      <div><span>{t('workflow.research_run_qualified')}</span><b>{String(receipt.qualifiedCount)}</b></div>
      <div><span>{t('workflow.research_run_rejected')}</span><b>{String(receipt.rejectedCount)}</b></div>
      <div><span>{t('workflow.research_run_quarantine')}</span><b>{String(receipt.quarantineCount)}</b></div>
    </div>
    <h5>{t('workflow.research_run_performance_funnel')}</h5>
    <div className="evidence-list research-performance-receipt-grid">
      <div><span>{t('workflow.research_lane_candidates')}</span><b>{String(funnel.attempted)}</b></div>
      <div><span>{t('workflow.research_run_structural_candidates')}</span><b>{String(funnel.structurallyFeasible)}</b></div>
      <div><span>{t('workflow.research_run_semantic_replay')}</span><b>{String(funnel.legalReplay)}</b></div>
      <div><span>{t('workflow.research_run_semantic_satisfied')}</span><b>{String(funnel.semanticSatisfied)}</b></div>
      <div><span>{t('workflow.research_run_proof_completed')}</span><b>{String(funnel.proofCompleted)}</b></div>
      <div><span>{t('workflow.research_run_qualified')}</span><b>{String(funnel.qualified)}</b></div>
    </div>
    <h5>{t('workflow.research_run_performance_timings')}</h5>
    <div className="evidence-list research-performance-receipt-grid">
      <div><span>{t('workflow.research_run_duration')}</span><b>{receiptValue(receipt.shardWallWorkSeconds)}</b></div>
      <div><span>{t('workflow.research_run_candidate_pipeline_seconds')}</span><b>{receiptValue(receipt.candidatePipelineSeconds)}</b></div>
      <div><span>{t('workflow.research_stage_construction')}</span><b>{receiptValue(receipt.constructionSeconds)}</b></div>
      <div><span>{t('workflow.research_stage_simulation')}</span><b>{receiptValue(receipt.simulationSeconds)}</b></div>
      <div><span>{t('workflow.research_stage_parity')}</span><b>{receiptValue(receipt.semanticParitySeconds)}</b></div>
      <div><span>{t('workflow.research_stage_analysis')}</span><b>{receiptValue(receipt.analysisSeconds)}</b></div>
      <div><span>{t('workflow.research_lane_deep_seconds')}</span><b>{receiptValue(receipt.tacticalVerificationSeconds)}</b></div>
      <div><span>{t('workflow.research_run_cpu_seconds')}</span><b>{receiptValue(receipt.cpuSeconds)}</b></div>
      <div><span>{t('workflow.research_run_first_feasible')}</span><b>{receiptValue(receipt.firstFeasibleCandidateSeconds)}</b></div>
      <div><span>{t('workflow.research_run_first_optimal')}</span><b>{receiptValue(receipt.firstOptimalCandidateSeconds)}</b></div>
      <div><span>{t('workflow.research_run_first_qualified')}</span><b>{receiptValue(receipt.firstQualifiedResultSeconds)}</b></div>
    </div>
    <h5>{t('workflow.research_run_performance_resources')}</h5>
    <div className="evidence-list research-performance-receipt-grid">
      <div><span>{t('workflow.research_run_proof_nodes')}</span><b>{String(proof.nodes)}</b></div>
      <div><span>{t('workflow.research_run_cache_entries')}</span><b>{String(proof.cacheEntries)}</b></div>
      <div><span>{t('workflow.research_run_cache_bytes')}</span><b>{String(proof.cacheBytes)}</b></div>
      <div><span>{t('workflow.research_run_hidden_worlds')}</span><b>{String(proof.hiddenWorlds)}</b></div>
      <div><span>{t('workflow.research_run_peak_rss')}</span><b>{receiptValue(receipt.peakProcessRssBytes)}</b></div>
      <div><span>{t('workflow.research_run_cancellation_observed')}</span><b>{receipt.cancellationObserved ? '✓' : '×'}</b></div>
    </div>
    <details className="audit-details"><summary>{t('workflow.research_run_performance_provenance')}</summary><dl className="research-performance-provenance"><div><dt>{t('workflow.research_run_terminal_status')}</dt><dd dir="ltr">{receipt.terminalStatus}</dd></div><div><dt>{t('workflow.research_run_cpu_boundary')}</dt><dd dir="ltr">{receipt.cpuMeasurementBoundary}</dd></div><div><dt>{t('workflow.research_run_peak_rss_boundary')}</dt><dd dir="ltr">{receipt.peakRssMeasurementBoundary ?? t('workflow.research_metric_unavailable')}</dd></div></dl><pre dir="ltr">{JSON.stringify({stageTimingBoundary:receipt.stageTimingBoundary,environment:receipt.environment,cacheScopes:receipt.cacheScopes,proofAggregation:proof.aggregation},null,2)}</pre></details>
  </section>;
}

function researchRunError(run: ResearchRun) {
  const raw = typeof run.error === 'string'
    ? run.error
    : String(
      (run.error as JsonObject | undefined)?.message
      || (run.error as JsonObject | undefined)?.detail
      || '',
    );
  if (raw.includes('CP-SAT deal construction did not prove an optimal deterministic result')) {
    return t('workflow.research_run_solver_exhausted');
  }
  return raw || t('workflow.research_run_no_error');
}

function researchEventLabel(type: string) {
  const labels: Record<string, string> = {
    queued: t('workflow.research_event_queued'),
    started: t('workflow.research_event_started'),
    completed: t('workflow.research_event_completed'),
    failed: t('workflow.research_event_failed'),
    cancelled: t('workflow.research_event_cancelled'),
    cancel_requested: t('workflow.research_event_cancel_requested'),
    progress: t('workflow.research_event_progress'),
  };
  return labels[type] || type;
}

function researchCandidateProgressLabel(
  classification: ResearchCandidateProgressClassification,
) {
  switch (classification) {
    case 'qualified': return t('workflow.research_run_qualified');
    case 'rejected': return t('workflow.research_run_rejected');
    case 'duplicate': return t('workflow.research_run_duplicate_qualified');
    case 'incomplete': return t('workflow.research_scenario_support_incomplete');
  }
}

function ResearchCampaignProgress({run}: {run: ResearchRun}) {
  const value = run.campaignAccounting || run.campaignProgress;
  if (!value) return null;
  const seconds = (raw: JsonValue | undefined) => `${number.format(Math.max(0, Math.round(Number(raw || 0))))} ${t('workflow.research_seconds')}`;
  const stopReason = String(value.stopReason || value.currentStage || 'running');
  const incomplete = value.latestIncompleteCompilation && typeof value.latestIncompleteCompilation === 'object' && !Array.isArray(value.latestIncompleteCompilation)
    ? value.latestIncompleteCompilation as JsonObject
    : null;
  const incompleteEvidence = incomplete?.evidence && typeof incomplete.evidence === 'object' && !Array.isArray(incomplete.evidence)
    ? incomplete.evidence as JsonObject
    : null;
  const partialStatusLabel = incomplete && (incomplete.stage === 'curatedSelection'
    ? t('workflow.research_run_rejected') : t('workflow.research_scenario_support_incomplete'));
  const candidateReceipts = Array.isArray(value.candidateReceipts)
    ? value.candidateReceipts.filter((item): item is ResearchCandidateProgressReceipt => (
      Boolean(item) && typeof item === 'object' && !Array.isArray(item)
    ))
    : [];
  const candidateReceiptCounts = value.candidateReceiptCounts && typeof value.candidateReceiptCounts === 'object' && !Array.isArray(value.candidateReceiptCounts)
    ? value.candidateReceiptCounts
    : null;
  const stopReasonLabel = researchRunStatusLabel(run);
  return <section className="research-sequential-summary">
    <h4>{t('workflow.research_event_progress')}</h4>
    <div className="evidence-list research-sequential-counters">
      <div><span>{t('workflow.research_estimate_candidates')}</span><b>{number.format(Number(value.requestedCandidates || 0))}</b></div>
      <div><span>{t('workflow.research_run_attempted_candidates')}</span><b>{number.format(Number(value.candidateAttempts || 0))}</b></div>
      <div><span>{t('workflow.research_run_structural_candidates')}</span><b>{number.format(Number(value.constructedCandidates || 0))}</b></div>
      <div><span>{scenarioStatusLabel('structurally_infeasible')}</span><b>{number.format(Number(value.structurallyInfeasible || 0))}</b></div>
      <div><span>{scenarioStatusLabel('semantic_postcondition_miss')}</span><b>{number.format(Number(value.semanticPostconditionMiss || 0))}</b></div>
      <div><span>{t('workflow.research_scenario_support_incomplete')}</span><b>{number.format(Number(value.proofIncomplete || 0))}</b></div>
      <div><span>{t('workflow.research_status_failed')}</span><b>{number.format(Number(value.proofFalse || 0))}</b></div>
      <div><span>{t('workflow.research_run_retry')}</span><b>{number.format(Number(value.deepProofRetries || 0))}</b></div>
      <div aria-live="polite" aria-atomic="true"><span>{t('workflow.research_truth_provisional')} · {t('workflow.research_run_qualified')}</span><b>{number.format(Number(value.provisionalQualified || 0))}</b></div>
      <div><span>{t('workflow.research_run_qualified')}</span><b>{number.format(Number(value.qualified || 0))}</b></div>
      {candidateReceiptCounts && <>
        <div><span>{t('workflow.research_sequential_receipts_title')} · {t('workflow.research_run_qualified')}</span><b>{number.format(Number(candidateReceiptCounts.qualified || 0))}</b></div>
        <div><span>{t('workflow.research_sequential_receipts_title')} · {t('workflow.research_run_rejected')}</span><b>{number.format(Number(candidateReceiptCounts.rejected || 0))}</b></div>
        <div><span>{t('workflow.research_sequential_receipts_title')} · {t('workflow.research_run_duplicate_qualified')}</span><b>{number.format(Number(candidateReceiptCounts.duplicate || 0))}</b></div>
        <div><span>{t('workflow.research_sequential_receipts_title')} · {t('workflow.research_scenario_support_incomplete')}</span><b>{number.format(Number(candidateReceiptCounts.incomplete || 0))}</b></div>
      </>}
      <div><span>{t('workflow.research_run_duration')}</span><b>{seconds(value.elapsedSeconds)}</b></div>
      {['queued','running'].includes(run.status) && <div><span>{t('workflow.research_estimate_solver')} · <code dir="ltr">campaign</code> · {t('settings.compiler_coverage_remaining')}</span><b>{seconds(value.remainingCampaignSeconds)}</b></div>}
      <div><span>{t('workflow.research_run_terminal_status')}</span>{stopReasonLabel && <b>{stopReasonLabel}</b>}<code dir="ltr">{stopReason}</code></div>
    </div>
    {incomplete && <p className="scenario-launch-block"><strong>{partialStatusLabel}</strong><code dir="ltr">{String(incomplete.stage || '—')} · {String(incomplete.qualityFailureReason ?? incomplete.exhaustionReason ?? incomplete.reason ?? '—')}</code>{incompleteEvidence?.detail && <small>{String(incompleteEvidence.detail)}</small>}</p>}
    {Array.isArray(value.provisionalResultHashes) && value.provisionalResultHashes.length > 0 && <div className="detail-tags" aria-label={t('workflow.research_truth_provisional')}>
      {value.provisionalResultHashes.map((hash) => <code dir="ltr" key={String(hash)}>{String(hash)}</code>)}
    </div>}
    {Array.isArray(value.provisionalResults) && value.provisionalResults.length > 0 && <div className="research-sequential-receipt-list" aria-label={t('workflow.research_truth_provisional')}>
      {value.provisionalResults.map((raw) => {
        const item = raw as JsonObject;
        return <article className="research-sequential-receipt" key={String(item.candidateHash)}><div className="research-sequential-identities">
          <span>{t('workflow.research_scenario_exact_state_id')}</span><code dir="ltr">{String(item.exactStateId || '—')}</code>
          <span>{t('workflow.research_label_contract')}</span><code dir="ltr">{String(item.contract || '—')}</code>
          <span>{t('workflow.research_filter_acting_seat')}</span><code dir="ltr">{String(item.actingSeat || '—')}</code>
          <span>{t('workflow.research_scenario_selected_action')}</span><code dir="ltr">{String(item.selectedAction || '—')}</code>
          <span>{t('workflow.research_scenario_proof_winning_root_action_count')}</span><code dir="ltr">{number.format(Number(item.winningRootActionCount || 0))}</code>
        </div></article>;
      })}
    </div>}
    {candidateReceipts.length > 0 && <div className="research-sequential-receipt-list" role="log" aria-live="polite" aria-relevant="additions text" aria-label={t('workflow.research_sequential_receipts_title')}>
      {value.candidateReceiptHistoryTruncated === true && <small>{t('workflow.research_sequential_receipts_page').replace('{shown}', number.format(candidateReceipts.length)).replace('{total}', number.format(Number(value.candidateReceiptCount || candidateReceipts.length)))}</small>}
      {candidateReceipts.map((item) => {
        const summary = item.candidateSummary && typeof item.candidateSummary === 'object' && !Array.isArray(item.candidateSummary)
          ? item.candidateSummary
          : null;
        const incompleteDetail = item.incompleteCompilation && typeof item.incompleteCompilation === 'object' && !Array.isArray(item.incompleteCompilation)
          ? item.incompleteCompilation
          : null;
        return <article className="research-sequential-receipt" key={item.receiptId}>
          <div className="research-sequential-receipt-heading"><b>{researchCandidateProgressLabel(item.classification)}</b><code dir="ltr">{item.status}</code></div>
          <div className="research-sequential-identities">
            <span>{t('workflow.research_scenario_status')}</span><code dir="ltr">{item.stage}</code>
            <span>{t('workflow.research_sequential_shard')}</span><code dir="ltr">{item.shardId ?? '—'}</code>
            <span>{t('workflow.research_sequential_attempt')}</span><code dir="ltr">{item.attempt ?? '—'}</code>
            <span>{t('workflow.research_sequential_receipt_hash')}</span><code dir="ltr">{item.candidateHash ?? item.variantSemanticHash ?? '—'}</code>
            {summary && <>
              <span>{t('workflow.research_scenario_exact_state_id')}</span><code dir="ltr">{String(summary.exactStateId ?? '—')}</code>
              <span>{t('workflow.research_scenario_selected_action')}</span><code dir="ltr">{String(summary.selectedAction ?? '—')}</code>
              <span>{t('workflow.research_scenario_proof_winning_root_action_count')}</span><code dir="ltr">{String(summary.winningRootActionCount ?? '—')}</code>
            </>}
          </div>
          {incompleteDetail && <small><code dir="ltr">{String(incompleteDetail.qualityFailureStage ?? incompleteDetail.stage ?? '—')}{incompleteDetail.qualityFailureCauseStage ? ` > ${String(incompleteDetail.qualityFailureCauseStage)}` : ''} · {String(incompleteDetail.qualityFailureReason ?? incompleteDetail.exhaustionReason ?? incompleteDetail.reason ?? '—')}</code></small>}
        </article>;
      })}
    </div>}
  </section>;
}

export function ResearchRunDetails({
  run: initialRun,
  recipe,
  stats: initialStats,
  busy,
  onCancel,
  onRetry,
  onResults,
  loadSequentialSearchReceipts,
  loadRun,
  loadStats,
  onSnapshot,
  PerformanceSummary,
}: {
  run: ResearchRun;
  recipe?: ResearchRecipe;
  stats: ResearchStats;
  busy: boolean;
  onCancel: (item: ResearchRun) => Promise<void>;
  onRetry: (item: ResearchRun) => Promise<void>;
  onResults: () => void;
  loadSequentialSearchReceipts: () => Promise<ResearchSequentialSearchReceiptPage>;
  loadRun: () => Promise<ResearchRun>;
  loadStats: () => Promise<ResearchStats>;
  onSnapshot?: (run: ResearchRun, stats?: ResearchStats) => void;
  PerformanceSummary: ComponentType<ResearchPerformanceSummaryProps>;
}) {
  const [run, setRun] = useState(initialRun);
  const [stats, setStats] = useState(initialStats);
  const receipt = performanceReceipt(run);
  const duration = receipt?.shardWallWorkSeconds ?? researchRunDuration(run);
  const events = Array.isArray(run.events) ? run.events : [];
  const active = run.status === 'queued' || run.status === 'running';
  const scenarioSearchStatus = typeof run.scenarioSearchStatus === 'string' ? run.scenarioSearchStatus : null;
  const scenarioResearchStatus = typeof run.scenarioResearchStatus === 'string' ? run.scenarioResearchStatus : null;
  const scenarioResearchEvidence = run.scenarioResearchEvidence && typeof run.scenarioResearchEvidence === 'object' && !Array.isArray(run.scenarioResearchEvidence)
    ? run.scenarioResearchEvidence as JsonObject
    : null;
  const proofBoundaries = Array.isArray(scenarioResearchEvidence?.proofBoundaries)
    ? scenarioResearchEvidence.proofBoundaries.filter((value): value is JsonObject => Boolean(value) && typeof value === 'object' && !Array.isArray(value))
    : [];
  const scenarioStatus = scenarioResearchStatus || scenarioSearchStatus;
  const scenarioYield = run.scenarioYieldCounts && typeof run.scenarioYieldCounts === 'object' && !Array.isArray(run.scenarioYieldCounts)
    ? run.scenarioYieldCounts as JsonObject
    : null;
  const [receiptPage, setReceiptPage] = useState<ResearchSequentialSearchReceiptPage | null>(null);
  const [receiptState, setReceiptState] = useState<'idle' | 'loading' | 'ready' | 'failed'>('idle');
  useEffect(() => {
    if (!active) return;
    let disposed = false;
    let terminalObserved = false;
    let fallback: number | null = null;
    const refresh = (withStats = false) => {
      if (terminalObserved) return;
      void loadRun().then(async (nextRun) => {
        if (disposed) return;
        setRun(nextRun);
        onSnapshot?.(nextRun);
        if (withStats || (nextRun.status !== 'queued' && nextRun.status !== 'running')) {
          terminalObserved = true;
          const nextStats = await loadStats().catch(() => undefined);
          if (disposed) return;
          if (nextStats) setStats(nextStats);
          onSnapshot?.(nextRun, nextStats);
        }
      }).catch(() => undefined);
    };
    const startFallback = () => {
      if (fallback !== null) return;
      fallback = window.setInterval(() => {
        if (!document.hidden) refresh();
      }, 1500);
    };
    if (typeof EventSource === 'undefined') {
      startFallback();
      return () => { disposed = true; if (fallback !== null) window.clearInterval(fallback); };
    }
    const source = new EventSource(`/api/v2/research-runs/${encodeURIComponent(run.runCode)}/events`);
    const onProgress = () => refresh();
    const onTerminal = () => refresh(true);
    for (const type of ['queued', 'started', 'progress', 'cancel_requested']) source.addEventListener(type, onProgress);
    for (const type of ['completed', 'failed', 'cancelled']) source.addEventListener(type, onTerminal);
    source.onerror = () => { source.close(); startFallback(); };
    // A stalled but open event stream must not leave a completed run looking active.
    const watchdog = window.setInterval(() => {
      if (!document.hidden) refresh();
    }, 5000);
    return () => {
      disposed = true;
      source.close();
      window.clearInterval(watchdog);
      if (fallback !== null) window.clearInterval(fallback);
    };
  }, [initialRun.runCode]);
  const loadReceipts = () => {
    if (receiptState !== 'idle') return;
    setReceiptState('loading');
    void loadSequentialSearchReceipts()
      .then((page) => { setReceiptPage(page); setReceiptState('ready'); })
      .catch(() => { setReceiptPage(null); setReceiptState('failed'); });
  };
  return <div className="research-run-details">
    <div className="drawer-stats">
      {recipe?.configuration && <Stat label={t('workflow.research_review_configuration')} value={recipe.configuration.bundleVersion}/>}
      <Stat label={t('workflow.research_run_results')} value={number.format(Number(run.resultCount || 0))}/>
      {run.auditResultCount !== undefined && <Stat label={t('workflow.laffa_library_audit_only')} value={number.format(Number(run.auditResultCount || 0))}/>}
      <Stat label={t('workflow.research_run_qualified')} value={number.format(Number(run.qualifiedCount || 0))}/>
      {run.selectedCount !== undefined && <Stat label={t('workflow.research_run_selected')} value={number.format(Number(run.selectedCount || 0))}/>}
      <Stat label={t('workflow.research_run_quarantine')} value={number.format(Number(run.quarantineCount || 0))}/>
      {run.duplicateQualifiedCount !== undefined && <Stat label={t('workflow.research_run_duplicate_qualified')} value={number.format(Number(run.duplicateQualifiedCount || 0))}/>}
      {duration !== null && <Stat label={t('workflow.research_run_duration')} value={`${receipt ? String(duration) : number.format(Math.round(duration))} ${t('workflow.research_seconds')}`}/>}
    </div>
    {Array.isArray(run.auditResultCodes) && run.auditResultCodes.length > 0 && <div className="detail-tags" aria-label={t('workflow.laffa_library_audit_only')}>{run.auditResultCodes.map((code)=><code dir="ltr" key={code}>{code}</code>)}</div>}
    <ResearchCampaignProgress run={run}/>
    {receipt ? <ResearchPerformanceReceipt receipt={receipt}/> : null}
    {scenarioStatus && scenarioYield ? <section className="research-sequential-summary">
      <h4>{t('workflow.research_run_details')}</h4>
      <div className="evidence-list research-sequential-counters">
        <div><span>{t('workflow.research_scenario_status')}</span><b>{scenarioStatusLabel(scenarioStatus)}</b></div>
        <div><span>{t('workflow.research_lane_candidates')}</span><b>{number.format(Number(scenarioYield.attempted || 0))}</b></div>
        <div><span>{t('workflow.research_run_structural_candidates')}</span><b>{number.format(Number(scenarioYield.structurallyFeasible || 0))}</b></div>
        <div><span>{t('workflow.research_run_semantic_replay')}</span><b>{number.format(Number(scenarioYield.legalReplay || 0))}</b></div>
        <div><span>{t('workflow.research_run_qualified')}</span><b>{number.format(Number(scenarioYield.qualified || 0))}</b></div>
        <div><span>{t('workflow.research_sequential_no_goods_applied')}</span><b>{number.format(Array.isArray(run.noGoodReceiptHashes) ? run.noGoodReceiptHashes.length : 0)}</b></div>
      </div>
      {proofBoundaries.length > 0 && <div className="research-proof-boundaries">{proofBoundaries.map((value, index) => <ResearchContinuationLineBoundary key={String(value.sha256 || index)} value={value}/>)}</div>}
    </section> : <>
      <PerformanceSummary stats={stats} onRuns={onResults} onResults={onResults}/>
      <section className="research-sequential-summary">
        <h4>{t('workflow.research_sequential_title')}</h4>
        <p>{t('workflow.research_sequential_help')}</p>
        <ResearchSequentialSearchCounters value={stats.sequentialSearch}/>
      </section>
      <div className="evidence-list">
      <div><span>{t('workflow.research_run_attempted_seeds')}</span><b>{number.format(Number(run.attemptedSeeds || 0))}</b></div>
      <div><span>{t('workflow.research_run_structural_candidates')}</span><b>{number.format(Number(run.structuralCandidates || 0))}</b></div>
      <div><span>{t('workflow.research_run_play_compiler_attempted')}</span><b>{number.format(Number(run.playCompilerAttempted || 0))}</b></div>
      <div><span>{t('workflow.research_run_play_compiler_optimal')}</span><b>{number.format(Number(run.playCompilerOptimal || 0))}</b></div>
      <div><span>{t('workflow.research_run_play_compiler_feasible')}</span><b>{number.format(Number(run.playCompilerFeasible || 0))}</b></div>
      <div><span>{t('workflow.research_run_play_compiler_incomplete')}</span><b>{number.format(Number(run.playCompilerIncomplete || 0))}</b></div>
      {Number(run.strategicSelectionDecisions || 0) > 0 && <>
        <div><span>{t('workflow.research_strategic_decisions')}</span><b>{number.format(Number(run.strategicSelectionDecisions || 0))}</b></div>
        <div><span>{t('workflow.research_strategic_exact')}</span><b>{number.format(Number(run.strategicExactDecisions || 0))}</b></div>
        <div><span>{t('workflow.research_strategic_provisional')}</span><b>{number.format(Number(run.strategicProvisionalDecisions || 0))}</b></div>
        <div><span>{t('workflow.research_strategic_fallback')}</span><b>{number.format(Number(run.strategicFallbackDecisions || 0))}</b></div>
      </>}
      <div><span>{t('workflow.research_run_semantic_replay')}</span><b>{number.format(Number(run.semanticReplayPassed || 0))}</b></div>
      <div><span>{t('workflow.research_stage_construction')}</span><b>{number.format(Math.round(Number(run.constructionSeconds || 0) * 100) / 100)}</b></div>
      <div><span>{t('workflow.research_stage_simulation')}</span><b>{number.format(Math.round(Number(run.simulationSeconds || 0) * 100) / 100)}</b></div>
      <div><span>{t('workflow.research_stage_parity')}</span><b>{number.format(Math.round(Number(run.semanticParitySeconds || 0) * 100) / 100)}</b></div>
      <div><span>{t('workflow.research_stage_analysis')}</span><b>{number.format(Math.round(Number(run.analysisSeconds || 0) * 100) / 100)}</b></div>
      <div><span>{t('workflow.research_lane_deep_verified')}</span><b>{number.format(Number(run.tacticalVerificationDecisions || 0))}</b></div>
      <div><span>{t('workflow.research_lane_deep_seconds')}</span><b>{number.format(Math.round(Number(run.tacticalVerificationSeconds || 0) * 100) / 100)}</b></div>
      </div>
      <details className="audit-details research-sequential-receipts" onToggle={(event) => { if (event.currentTarget.open) loadReceipts(); }}>
      <summary>{t('workflow.research_sequential_receipts_title')} · {number.format(stats.sequentialSearch.admittedSeedCount)}</summary>
      <p>{t('workflow.research_sequential_receipts_help')}</p>
      {receiptState === 'loading' && <p>{t('workflow.research_sequential_receipts_loading')}</p>}
      {receiptState === 'failed' && <p role="alert">{t('workflow.research_sequential_receipts_failed')}</p>}
      {receiptState === 'ready' && receiptPage?.items.length === 0 && <p>{t('workflow.research_sequential_receipts_empty')}</p>}
      {receiptPage && receiptPage.items.length > 0 && <>
        <small>{t('workflow.research_sequential_receipts_page').replace('{shown}', number.format(receiptPage.items.length)).replace('{total}', number.format(receiptPage.total))}</small>
        <div className="research-sequential-receipt-list">{receiptPage.items.map((item) => <article className="research-sequential-receipt" key={item.occurrenceHash}>
          <div className="research-sequential-receipt-heading"><b>{t('workflow.research_sequential_seed')} · <span dir="ltr">{item.seed}</span></b><small dir="ltr">{item.createdAt}</small></div>
          <div className="research-sequential-identities">
            <span>{t('workflow.research_sequential_attempt')}</span><code>{item.attemptCount}</code>
            <span>{t('workflow.research_sequential_fencing_token')}</span><code>{item.fencingToken}</code>
            <span>{t('workflow.research_sequential_shard')}</span><code>{item.shardId}</code>
            <span>{t('workflow.research_sequential_receipt_hash')}</span><code>{item.receiptHash}</code>
            <span>{t('workflow.research_sequential_occurrence_hash')}</span><code>{item.occurrenceHash}</code>
          </div>
          <details className="audit-details research-sequential-item-summary"><summary>{t('workflow.research_sequential_receipt_summary')}</summary><ResearchSequentialSearchCounters value={item.summary}/></details>
          <details className="audit-details"><summary>{t('workflow.research_sequential_exact_bundle')}</summary><pre dir="ltr">{JSON.stringify(item.receiptBundle, null, 2)}</pre></details>
        </article>)}</div>
      </>}
      </details>
    </>}
    {Number(run.solverIncomplete || 0) > 0 && <section className="research-run-error"><b>{t('workflow.research_run_solver_exhausted')}</b><p>{t('workflow.research_run_solver_exhausted')}</p></section>}
    {(run.status === 'failed' || run.error) && <section className="research-run-error"><b>{t('workflow.research_run_error')}</b><p>{researchRunError(run)}</p></section>}
    <section className="research-run-events"><h4>{t('workflow.research_run_events')}</h4>{events.length ? <ol>{events.map((event, index) => <li key={`${String(event.at || '')}-${index}`}><b>{researchEventLabel(String(event.type || ''))}</b>{event.currentStage ? <code dir="ltr">{String(event.currentStage)}</code> : null}{event.constructionStatus ? <code dir="ltr">{String(event.constructionStatus)}</code> : null}{event.constructionReason ? <code dir="ltr">{String(event.constructionReason)}</code> : null}{Array.isArray(event.constructionInfeasibleCore) ? event.constructionInfeasibleCore.map((field) => field === 'decision_actor' ? <span key={field}>{t('workflow.research_scenario_decision_actor')}</span> : field === 'scenario_expression' ? <span key={field}>{t('workflow.research_scenario_constraints')} · {t('workflow.research_scenario_outcome')}</span> : null) : null}{event.candidateStatus ? <code dir="ltr">{String(event.candidateStatus)}</code> : null}{event.proofBoundaryStatus === 'infeasible' ? <span>{t('workflow.research_run_rejected')}</span> : null}{event.proofBoundaryStatus === 'incomplete' ? <span>{t('workflow.research_scenario_support_incomplete')}</span> : null}{event.proofBoundaryReason ? <code dir="ltr">{String(event.proofBoundaryReason)}</code> : null}<time dir="ltr">{event.at ? new Date(String(event.at)).toLocaleString('ar-SA') : '—'}</time></li>)}</ol> : <p>{t('workflow.research_run_no_events')}</p>}</section>
    <div className="research-run-actions">
      {active && <button type="button" className="danger" disabled={busy} onClick={() => void onCancel(run)}>{t('workflow.research_run_cancel')}</button>}
      {!active && <button type="button" className="primary" disabled={busy} onClick={() => void onRetry(run)}>{t('workflow.research_run_retry')}</button>}
    </div>
  </div>;
}

export function ResearchDisagreementAudit({
  item,
}: {
  item: SemanticDisagreement;
}) {
  const solver = (item.solverEvidence || {}) as JsonObject;
  const engine = (item.runtimeEvidence || item.engineEvidence || {}) as JsonObject;
  return <div className="research-disagreement-audit">
    <p className="research-quarantine-notice">{t('workflow.research_quarantine_retained')}</p>
    <div className="drawer-stats"><Stat label={t('workflow.research_quarantine_status')} value={item.status === 'open' ? t('workflow.research_quarantine_open') : String(item.status || '—')}/><Stat label={t('workflow.research_quarantine_run')} value={String(item.runCode || '—')}/></div>
    <div className="research-evidence-compare"><section><h4>{t('workflow.research_quarantine_solver_evidence')}</h4><JsonSummary value={solver}/></section><section><h4>{t('workflow.research_quarantine_engine_evidence')}</h4><JsonSummary value={engine}/></section></div>
  </div>;
}

function researchScoreDimensionLabel(key: string) {
  const labels: Record<string, string> = {
    outcomeImpact: t('workflow.research_dimension_outcome_impact'),
    uniqueness: t('workflow.research_dimension_uniqueness'),
    disasterAvoidance: t('workflow.research_dimension_disaster_avoidance'),
    kaboutImpact: t('workflow.research_dimension_kabout_impact'),
    sacrificeValue: t('workflow.research_dimension_sacrifice_value'),
    delayedGain: t('workflow.research_dimension_delayed_gain'),
    tacticalDepth: t('workflow.research_dimension_tactical_depth'),
    informationBurden: t('workflow.research_dimension_information_burden'),
    robustness: t('workflow.research_dimension_robustness'),
    confidence: t('workflow.research_dimension_confidence'),
    explanationFidelity: t('workflow.research_dimension_explanation_fidelity'),
    visualClarity: t('workflow.research_dimension_visual_clarity'),
    novelty: t('workflow.research_dimension_novelty'),
    coverage: t('workflow.research_dimension_coverage'),
  };
  return labels[key] || key;
}

export function ResearchDecisionAudit({
  result,
  recipe,
  decisions,
  existingVideoUrl,
  onPrepare,
  onVerify,
  onOpenExactState,
  loadPerspectives,
}: {
  result: ResearchResult;
  recipe?: ResearchRecipe | null;
  decisions: JsonObject[];
  existingVideoUrl?: string;
  onPrepare: (observerSeat: SeatName, ordinal?: number) => void;
  onVerify: () => void;
  onOpenExactState: (observerSeat: SeatName) => void;
  loadPerspectives: LoadResearchPerspectives;
}) {
  const percent = (value: unknown) => Number.isFinite(Number(value))
    ? `${number.format(Math.round(Number(value) * 100))}${t('workflow.batch_percent_suffix')}`
    : '—';
  const metric = (value: unknown) => Number.isFinite(Number(value))
    ? number.format(Math.round(Number(value) * 10) / 10)
    : '—';
  const qualification = (result.qualification || {}) as JsonObject;
  const failures = Array.isArray(qualification.failures)
    ? qualification.failures.map(String)
    : [];
  const scoreEvidence = (result.scoreEvidence || {}) as JsonObject;
  const laffaScore = ((scoreEvidence.laffaThrillScore || scoreEvidence.baseLaffaScore || {}) as JsonObject).score;
  const peakDecisionScore = scoreEvidence.peakDecisionChallengeScore;
  const decisionScores = Array.isArray(scoreEvidence.decisionChallengeScores)
    ? scoreEvidence.decisionChallengeScores as JsonObject[]
    : [];
  const scoreByDecision = new Map(decisionScores.map((item) => [String(item.decisionId), item]));
  const strategic = (result.strategicSelection || {}) as JsonObject;
  const strategicCount = Number(strategic.decisionCount || 0);
  const scenarioProjection = (result.scenarioProjection || {}) as JsonObject;
  const scenarioProof = (result.proofOutcome || {}) as JsonObject;
  const scenarioPredicateCount = Number(scenarioProjection.predicateEvidenceCount || 0);
  const exactStateAvailable = scenarioProjection.exactStateAvailable === true;
  const scenarioRootActions = Array.isArray(scenarioProof.rootActions)
    ? scenarioProof.rootActions as JsonObject[]
    : [];
  const isScenarioV4 = result.schemaVersion === 2
    && Number(scenarioProjection.schemaVersion) === 1;
  const isOrToolsResult = isOrToolsResearchBackend(result.researchBackend)
    && result.qualificationAuthority === 'cp_sat';
  const isNativeV3 = isScenarioV4 && result.researchBackend === 'ortools_cp_sat_v3';
  const [operatorAudit, setOperatorAudit] = useState<ResearchResultOperatorAuditV1 | null>(null);
  const [qualityAudit, setQualityAudit] = useState<ResearchPlayingQualityAuditV1 | null>(null);
  const [operatorLocalization, setOperatorLocalization] = useState<ScenarioResearchLocalizationV1 | null>(null);
  const [operatorAuditState, setOperatorAuditState] = useState<'idle' | 'loading' | 'ready' | 'failed'>('idle');
  const decisionOnlyProof = isScenarioV4 && !qualityAudit
    && (operatorAuditState === 'ready' || operatorAuditState === 'failed');
  const proofScopeKey = scenarioProof.achievedMode === 'exact'
    ? 'research_scenario_proof_exact_help'
    : scenarioProof.achievedMode === 'bounded'
      ? 'research_scenario_proof_bounded_help'
      : scenarioProof.achievedMode === 'policy_conditioned'
        ? 'research_scenario_proof_policy_conditioned_help'
        : null;
  useEffect(() => {
    if (!isOrToolsResult) {
      setOperatorAudit(null);
      setQualityAudit(null);
      setOperatorLocalization(null);
      setOperatorAuditState('idle');
      return;
    }
    let disposed = false;
    setOperatorAudit(null);
    setQualityAudit(null);
    setOperatorLocalization(null);
    setOperatorAuditState('loading');
    Promise.all([
      getResearchResultOperatorAudit(result.resultCode),
      result.researchBackend === 'ortools_cp_sat_v3'
        ? getResearchResultQualityAudit(result.resultCode).catch(() => null)
        : Promise.resolve(null),
    ])
      .then(async ([value, quality]) => {
        const localization = await getScenarioResearchLocalization(
          value.researchAuthority.bundleRevisionId,
        );
        if (!disposed) {
          setOperatorAudit(value);
          setQualityAudit(quality);
          setOperatorLocalization(localization);
          setOperatorAuditState('ready');
        }
      })
      .catch(() => {
        if (!disposed) setOperatorAuditState('failed');
      });
    return () => { disposed = true; };
  }, [isOrToolsResult, result.resultCode]);
  const researchAuthority = operatorAudit?.researchAuthority;
  const runtimeCompatibility = operatorAudit?.runtimeCompatibility;
  const executionReceipt = operatorAudit?.executionReceipt;
  const stageTimings = executionReceipt?.stageTimings ?? {};
  const runtimeCompatibilityStatus = runtimeCompatibility?.status ?? 'not_checked';
  const normalMediaAllowed = !isOrToolsResult || runtimeCompatibilityStatus === 'passed';
  const maximumQualityRegret = qualityAudit?.decisions.reduce(
    (maximum, decision) => Math.max(maximum, decision.regret),
    0,
  ) ?? 0;
  const recipeHash = (result.provenance as JsonObject | undefined)?.recipeHash;
  const qualityPolicy = recipe && recipe.sha256 === recipeHash
    ? (recipe.playingQuality as JsonObject | undefined)
    : undefined;
  const qualityPolicyDecisions = (qualityPolicy?.decisions || {}) as JsonObject;
  const operatorCopy = (key: string) => {
    const value = operatorLocalization?.groups.workflow[key];
    if (!value) throw new Error(`Scenario Research copy key is missing: ${key}`);
    return value;
  };
  const proofScopeCopy = operatorLocalization && proofScopeKey
    ? operatorCopy(proofScopeKey) : null;
  const proofBoundaryCopyReady = [
    'research_decision_proof_boundaries',
    'research_decision_proof_mode',
    'research_decision_proof_scope',
    'research_decision_proof_exact',
    'research_decision_proof_bounded',
  ].every((key) => Boolean(operatorLocalization?.groups.workflow[key]));
  const projectedObserverSeats = (result.libraryProjection as JsonObject | undefined)?.observerSeats;
  const hasLegacyPerspectives = result.schemaVersion !== 2;
  const observerSeats: SeatName[] = Array.isArray(projectedObserverSeats)
    ? projectedObserverSeats.filter(isSeatName)
    : [];
  const [observerSeat, setObserverSeat] = useState<SeatName>(observerSeats[0] || 'north');
  const observerSelectionReady = observerSeats.includes(observerSeat);
  const publicationEligibilityKnown = typeof result.publicationEligible === 'boolean';
  const publicationEligible = result.publicationEligible === true;
  const publicationPolicy = (result.publicationPolicy || {}) as JsonObject;
  const publicationBlockers = [...new Set([
    ...(Array.isArray(result.publicationBlockers) ? result.publicationBlockers.map(String) : []),
    ...(Array.isArray(publicationPolicy.blockers) ? publicationPolicy.blockers.map(String) : []),
  ])];
  const [perspectiveRows, setPerspectiveRows] = useState<JsonObject[]>([]);
  const [perspectiveState, setPerspectiveState] = useState<'loading' | 'ready' | 'empty' | 'failed'>('loading');
  useEffect(() => {
    if (!hasLegacyPerspectives || !observerSelectionReady) {
      setPerspectiveRows([]);
      setPerspectiveState('empty');
      return;
    }
    let disposed = false;
    setPerspectiveState('loading');
    loadPerspectives(result.resultCode, observerSeat)
      .then((value) => {
        if (!disposed) {
          setPerspectiveRows(value.items);
          setPerspectiveState(value.items.length ? 'ready' : 'empty');
        }
      })
      .catch(() => {
        if (!disposed) {
          setPerspectiveRows([]);
          setPerspectiveState('failed');
        }
      });
    return () => { disposed = true; };
  }, [hasLegacyPerspectives, loadPerspectives, observerSeat, observerSelectionReady, result.resultCode]);
  const promptLabels: Record<string, string> = {
    choose_acting_action: t('workflow.research_prompt_choose_action'),
    predict_opponent_play: t('workflow.research_prompt_predict_opponent'),
    read_partner_signal: t('workflow.research_prompt_partner_signal'),
    identify_threat: t('workflow.research_prompt_identify_threat'),
    plan_next_response: t('workflow.research_prompt_plan_response'),
    infer_distribution: t('workflow.research_prompt_infer_distribution'),
    post_reveal_analysis: t('workflow.research_prompt_post_reveal'),
  };
  const promptCounts = [...perspectiveRows.reduce(
    (counts, item) => counts.set(
      String(item.promptContract || ''),
      (counts.get(String(item.promptContract || '')) || 0) + 1,
    ),
    new Map<string, number>(),
  ).entries()];
  const proofModeLabel = (value: unknown) => {
    const mode = String(value || '');
    return mode === 'exact'
      ? t('workflow.research_scenario_proof_exact')
      : mode === 'bounded'
        ? t('workflow.research_scenario_proof_bounded')
        : mode === 'policy_conditioned'
          ? t('workflow.research_scenario_proof_policy_conditioned')
          : mode;
  };
  const proofClaimLabel = (value: unknown) => ({
    outcome_reachable: t('workflow.research_scenario_claim_outcome_reachable'),
    outcome_forced: t('workflow.research_scenario_claim_outcome_forced'),
    winning_root_action_count: t('workflow.research_scenario_claim_winning_root_action_count'),
    winning_continuation_line_count: t('workflow.research_scenario_claim_winning_continuation_line_count'),
    selected_action_is_unique_winner: t('workflow.research_scenario_claim_selected_action_is_unique_winner'),
    target_team_has_no_salvage: t('workflow.research_scenario_claim_target_team_has_no_salvage'),
  }[String(value || '')] || String(value || '—'));
  const proofUtilityLabel = (value: unknown) => {
    const selected = String(value || '');
    switch (selected) {
      case 'awarded_point_margin': return t('workflow.research_scenario_option_utility_awarded_point_margin');
      case 'contract_success': return t('workflow.research_scenario_option_utility_contract_success');
      case 'round_win': return t('workflow.research_scenario_option_utility_round_win');
      case 'match_win': return t('workflow.research_scenario_option_utility_match_win');
      case 'kabout': return t('workflow.research_scenario_option_utility_kabout');
      default: return selected || '—';
    }
  };
  const horizonLabel = (value: unknown) => {
    const selected = String(value || '');
    switch (selected) {
      case 'plies': return t('workflow.research_scenario_option_horizon_plies');
      case 'decisions': return t('workflow.research_scenario_option_horizon_decisions');
      case 'tricks': return t('workflow.research_scenario_option_horizon_tricks');
      case 'round_end': return t('workflow.research_scenario_option_horizon_round_end');
      default: return selected || '—';
    }
  };
  return <>
    <div className="drawer-stats">
      {!isScenarioV4 && <Stat label={t('workflow.research_column_score')} value={number.format(Math.round(result.score))}/>}
      {Number.isFinite(Number(laffaScore)) && <Stat label={t('workflow.research_laffa_thrill_score')} value={number.format(Math.round(Number(laffaScore)))}/>}
      {Number.isFinite(Number(peakDecisionScore)) && <Stat label={t('workflow.research_peak_decision_score')} value={number.format(Math.round(Number(peakDecisionScore)))}/>}
      {!isScenarioV4 && <Stat label={t('workflow.research_column_confidence')} value={`${number.format(Math.round(result.confidence * 100))}${t('workflow.batch_percent_suffix')}`}/>}
      <Stat label={t('workflow.research_column_decisions')} value={number.format(decisions.length)}/>
    </div>
    {isScenarioV4 && <section className={`research-scenario-result ${String(scenarioProjection.status) === 'satisfied' ? 'qualified' : 'rejected'}`}>
      <h4>{t('workflow.research_scenario_evidence_title')}</h4>
      <div className="drawer-stats">
      <Stat label={t('workflow.research_scenario_status')} value={scenarioStatusLabel(String(scenarioProjection.status))}/>
        <Stat label={t('workflow.research_scenario_result_origin')} value={String(scenarioProjection.origin)}/>
        <Stat label={t('workflow.research_scenario_requested_proof')} value={proofModeLabel(scenarioProof.requestedMode)}/>
        <Stat label={t('workflow.research_scenario_achieved_proof')} value={proofModeLabel(scenarioProof.achievedMode)}/>
        <Stat label={t('workflow.research_scenario_utility')} value={proofUtilityLabel(scenarioProof.utility)}/>
        <Stat label={t('workflow.research_scenario_threshold')} value={receiptValue(scenarioProof.successThreshold)}/>
        <ResearchWinningRootActionBoundary value={scenarioProof}/>
        <Stat label={t('workflow.research_scenario_predicate_evidence')} value={number.format(scenarioPredicateCount)}/>
      </div>
      <p>{t('workflow.research_scenario_proof_boundary')} <b>{String(scenarioProof.completionStatus || '—')}</b></p>
      <ResearchContinuationLineBoundary value={scenarioProof}/>
      {!exactStateAvailable && <p aria-live="polite">{t('workflow.batch_rejection_no_retained_trace')}</p>}
      <div className="drawer-actions"><button type="button" className="secondary" disabled={!exactStateAvailable||!observerSelectionReady||!normalMediaAllowed} onClick={() => onOpenExactState(observerSeat)}>{t('workflow.research_scenario_open_exact_state')}</button></div>
      <details className="audit-details"><summary>{t('workflow.research_scenario_proof')}</summary>
        <div className="drawer-stats">
          <Stat label={t('workflow.research_scenario_visibility')} value={String(scenarioProof.achievedVisibility) === 'actor_information_set' ? t('workflow.research_scenario_visibility_actor_information_set') : t('workflow.research_scenario_visibility_complete_state')}/>
          <Stat label={t('workflow.research_scenario_claim')} value={proofClaimLabel(scenarioProof.claim)}/>
          <Stat label={t('workflow.research_scenario_target_team')} value={String(scenarioProof.targetTeam) === 'west_east' ? `${seatLabel.west} + ${seatLabel.east}` : `${seatLabel.north} + ${seatLabel.south}`}/>
          <Stat label={t('workflow.research_scenario_selected_action')} value={String(scenarioProof.selectedAction || '—')}/>
        </div>
        {scenarioRootActions.length > 0 && <div className="detail-tags" aria-label={t('workflow.research_scenario_root_actions')}>
          {scenarioRootActions.map((item) => <span key={String(item.action)}><b dir="ltr">{String(item.action)}</b> · {t('workflow.research_scenario_action_value')} {number.format(Number(item.value))} · {item.satisfies === true ? t('workflow.research_scenario_action_wins') : t('workflow.research_scenario_action_loses')}</span>)}
        </div>}
      </details>
    </section>}
    {isOrToolsResult && operatorAuditState==='ready' && operatorLocalization && <section className={`research-scenario-authority ${normalMediaAllowed?'qualified':'rejected'}`}>
      <h4>{operatorCopy('research_scenario_authority_title')}</h4>
      <p>{operatorCopy('research_scenario_authority_help')}</p>
      <div className="drawer-stats">
        <Stat label={operatorCopy('research_scenario_backend')} value={String(researchAuthority?.backendId || '—')}/>
        <Stat label={operatorCopy('research_scenario_planner')} value={String(researchAuthority?.backendVersion || '—')}/>
        <Stat label={operatorCopy('research_scenario_authority_title')} value={String(researchAuthority?.qualificationAuthority || '—')}/>
        <Stat label={operatorCopy('research_scenario_runtime_compatibility')} value={operatorCopy(`research_scenario_runtime_compatibility_${runtimeCompatibilityStatus}`)}/>
      </div>
      {runtimeCompatibility?.failureCodes.length?<div className="detail-tags">{runtimeCompatibility.failureCodes.map((code)=><code dir="ltr" key={String(code)}>{String(code)}</code>)}</div>:null}
      {Object.keys(stageTimings).length?<details open><summary>{operatorCopy('research_scenario_timing_title')}</summary><p>{operatorCopy('research_scenario_timing_help')}</p><dl className="research-scenario-stage-timings">{Object.entries(stageTimings).map(([stage,value])=><div key={stage}><dt>{tailTimingLabel(stage, operatorCopy) || operatorLocalization.groups.workflow[`research_scenario_timing_${stage.replace(/([A-Z])/g, '_$1').toLowerCase()}`] || <code dir="ltr">{stage}</code>}</dt><dd>{number.format(Number(value))}s</dd></div>)}</dl></details>:null}
      <div className="scenario-estimate-identities"><code dir="ltr">{String(researchAuthority?.loweringManifestHash||'—')}</code><code dir="ltr">{String(researchAuthority?.constructionModelHash||'—')}</code><code dir="ltr">{String(researchAuthority?.proofModelHash||'—')}</code><code dir="ltr">{String(executionReceipt?.sha256||'—')}</code></div>
    </section>}
    {qualityAudit && <section className="research-qualification-summary qualified">
      <b>{t('workflow.research_all_decisions_required')}</b>
      <div>
        <span>{t('workflow.research_strategic_decisions')}: {number.format(qualityAudit.decisionCount)}</span>
        <span>{t('workflow.research_forced_play')}: {number.format(qualityAudit.decisions.filter((decision) => decision.kind === 'play' && decision.legalActionCount === 1).length)} / {number.format(qualityAudit.countsByKind.play)}</span>
        {!isNativeV3 && <span>{t('workflow.research_play_proof_optimal')}: {number.format(qualityAudit.optimalDecisionCount)} / {number.format(qualityAudit.decisionCount)}</span>}
        <span>{t('workflow.research_maximum_regret')}: {number.format(maximumQualityRegret)}</span>
      </div>
      <div className="detail-tags">
        {Object.entries(qualityAudit.countsByKind).filter(([, count]) => count > 0).map(([kind, count]) => <span key={kind}>{decisionLabel[kind] || kind}: {number.format(count)}</span>)}
      </div>
      <small>{isNativeV3 && operatorLocalization
        ? operatorCopy('research_scenario_proof_exact_help')
        : t('workflow.research_strategic_not_global')}</small>
      {proofBoundaryCopyReady ? <details><summary>{operatorCopy('research_decision_proof_boundaries')}</summary>
        <div className="data-table-wrap"><table className="data-table research-quality-boundary-table">
          <caption>{operatorCopy('research_decision_proof_boundaries')}</caption>
          <thead><tr>
            <th scope="col">#</th><th scope="col">{t('workflow.research_scenario_decision')}</th>
            <th scope="col">{t('workflow.research_scenario_decision_actor')}</th><th scope="col">{t('workflow.research_actual_action')}</th>
            <th scope="col">{operatorCopy('research_decision_proof_mode')}</th><th scope="col">{operatorCopy('research_decision_proof_scope')}</th>
          </tr></thead>
          <tbody>{qualityAudit.decisions.map((decision)=><tr key={decision.ordinal}>
            <td>{number.format(decision.ordinal)}</td><td>{decisionLabel[decision.kind] || decision.kind}</td>
            <td>{seatLabel[decision.actor] || decision.actor}</td><td dir="ltr">{decision.action}</td>
            <td>{decision.proofMode === 'exact'
              ? operatorCopy('research_decision_proof_exact')
              : operatorCopy('research_decision_proof_bounded')} · {decision.proofVisibility === 'complete_state'
                ? operatorCopy('research_scenario_visibility_complete_state')
                : operatorCopy('research_scenario_visibility_actor_information_set')}</td>
            <td>{proofScopeLabel(decision.proofScope, (key) => operatorLocalization?.groups.workflow[key] || '')
              || <code dir="ltr">{decision.proofScope}</code>}</td>
          </tr>)}</tbody>
        </table></div>
      </details> : operatorLocalization?.groups.workflow.research_scenario_support_incomplete
        && <small>{operatorCopy('research_scenario_support_incomplete')}</small>}
      <details><summary>{t('workflow.research_scenario_requested_proof')}</summary>
        <div className="detail-tags">
          {Object.entries(qualityPolicyDecisions).map(([kind, value]) => {
            const analysis = value as JsonObject;
            const horizon = (analysis.horizon || {}) as JsonObject;
            return <span key={kind}>
              {decisionLabel[kind] || kind} · {proofUtilityLabel(analysis.utility)} · {horizon.unit
                ? <>{number.format(Number(horizon.value))} {horizonLabel(horizon.unit)}</>
                : t('workflow.research_scenario_option_horizon_round_end')}
              {analysis.boundaryEvaluator && <> · {t('workflow.research_scenario_boundary_evaluator')}: <code dir="ltr">{String(analysis.boundaryEvaluator)}</code></>}
            </span>;
          })}
          {!qualityPolicy && operatorLocalization && <span>{operatorCopy('research_scenario_support_incomplete')}</span>}
        </div>
      </details>
    </section>}
    {isNativeV3 && qualityAudit?.featuredPlayProof && operatorLocalization && <section className="research-qualification-summary">
      <b>{qualityAudit.featuredPlayProof.completionStatus === 'exact_complete'
        ? operatorCopy('research_scenario_proof_exact') : operatorCopy('research_scenario_proof_bounded')}</b>
      <div>
        {qualityAudit.featuredPlayProof.hiddenWorlds ? <>
          <span>{operatorCopy('research_information_set_support')}: {number.format(qualityAudit.featuredPlayProof.hiddenWorlds.count)}</span>
          <span>{operatorCopy(qualityAudit.featuredPlayProof.hiddenWorlds.complete
            ? 'research_scenario_visibility_actor_information_set' : 'research_scenario_support_incomplete')}</span>
        </> : <span>{operatorCopy('research_scenario_visibility_complete_state')}</span>}
        <span dir="ltr">{qualityAudit.featuredPlayProof.action}</span>
        <code dir="ltr">{qualityAudit.featuredPlayProof.exactStateId}</code>
      </div>
      <small>{qualityAudit.featuredPlayProof.completionStatus === 'exact_complete'
        ? operatorCopy('research_scenario_proof_exact_help') : operatorCopy('research_scenario_proof_bounded_help')}</small>
    </section>}
    {hasLegacyPerspectives && <section className="research-perspective-switch"><div><b>{t('workflow.research_perspective_title')}</b><small>{t('workflow.research_perspective_help')}</small></div><label><span>{t('workflow.research_filter_observer_seat')}</span><select aria-label={t('workflow.research_filter_observer_seat')} value={observerSeat} onChange={(event) => { if (isSeatName(event.target.value)) setObserverSeat(event.target.value); }}>{observerSeats.map((seat) => <option key={seat} value={seat}>{seatLabel[seat] || seat}</option>)}</select></label>{perspectiveState === 'loading' ? <p>{t('workflow.research_perspective_loading')}</p> : perspectiveState === 'ready' ? <div className="detail-tags" aria-live="polite"><span>{t('workflow.research_perspective_moments')}: {number.format(perspectiveRows.length)}</span>{promptCounts.map(([prompt, count]) => <span key={prompt}>{promptLabels[prompt] || prompt}: {number.format(count)}</span>)}</div> : <p>{t('workflow.research_perspective_empty')}</p>}</section>}
    {strategicCount > 0 && <section className="research-qualification-summary qualified"><b>{t('workflow.research_strategic_evidence_title')}</b><div><span>{t('workflow.research_strategic_decisions')}: {number.format(strategicCount)}</span><span>{t('workflow.research_strategic_exact')}: {number.format(Number(strategic.exactDecisionCount || 0))}</span><span>{t('workflow.research_strategic_provisional')}: {number.format(Number(strategic.provisionalDecisionCount || 0))}</span><span>{t('workflow.research_strategic_fallback')}: {number.format(Number(strategic.fallbackDecisionCount || 0))}</span></div><small>{t('workflow.research_strategic_not_global')}</small></section>}
    <section className={`research-qualification-summary ${failures.length ? 'rejected' : 'qualified'}`}>
      <b>{failures.length ? t('workflow.research_rejection_summary') : t(decisionOnlyProof ? 'workflow.research_status_qualified' : 'workflow.research_qualification_passed')}</b>
      {failures.length > 0 && <div>{failures.map((failure) => <span key={failure}>{failure === 'native_round_win_counterstrategy' && operatorLocalization?.groups.workflow.research_failure_counterstrategy
        ? operatorCopy('research_failure_counterstrategy') : researchFailureLabel(failure)}</span>)}</div>}
      {decisionOnlyProof && proofScopeCopy && <small>{proofScopeCopy}</small>}
      {Number.isFinite(Number(qualification.minimumSurroundingQuality)) && <small>{t('workflow.research_minimum_surrounding_result')}: {metric(qualification.minimumSurroundingQuality)}</small>}
    </section>
    {publicationEligibilityKnown && <section className={`research-qualification-summary ${publicationEligible ? 'qualified' : 'rejected'}`}><b>{publicationEligible ? t('workflow.laffa_library_content_candidate') : t('workflow.laffa_library_audit_only')}</b>{publicationBlockers.length > 0 && <div>{publicationBlockers.map((blocker) => <span key={blocker}>{researchFailureLabel(blocker)}</span>)}</div>}</section>}
    <div className="drawer-actions">{existingVideoUrl?<a className="button-link primary" href={existingVideoUrl}>{t('workflow.label_open_video')}</a>:<button type="button" className="primary" disabled={!observerSelectionReady||!normalMediaAllowed||(isScenarioV4&&!exactStateAvailable)} onClick={() => onPrepare(observerSeat)}>{t('workflow.research_prepare_video')}</button>}{!isScenarioV4 && <button type="button" className="secondary" onClick={onVerify}>{t('workflow.research_play_proof_action')}</button>}</div>
    <div className="decision-audit-list">{decisions.map((item) => {
      const alternatives = Array.isArray(item.alternatives) ? item.alternatives as JsonObject[] : [];
      const legalActions = Array.isArray(item.legalActions) ? item.legalActions as JsonObject[] : [];
      const analysis = (item.analyticalExplanation || {}) as JsonObject;
      const informationSet = (item.informationSetEvidence || {}) as JsonObject;
      const informationSetStatus = String(informationSet.status || 'incomplete');
      const sequentialResponse = (item.sequentialResponseEvidence || {}) as JsonObject;
      const sequentialResponseStatus = String(sequentialResponse.status || 'incomplete');
      const challengeScore = scoreByDecision.get(String(item.decisionId));
      const scoreBreakdown = (challengeScore?.breakdown || {}) as JsonObject;
      const scoreDimensions = (scoreBreakdown.dimensions || {}) as JsonObject;
      const scoreContributions = (scoreBreakdown.contributions || {}) as JsonObject;
      const continuation = Array.isArray(analysis.causalContinuation)
        ? analysis.causalContinuation as string[]
        : [];
      const provenance = (item.provenance || {}) as JsonObject;
      const actorView = observerSeat === String(item.actingSeat);
      const perspectiveRow = perspectiveRows.find((row) => String(row.decisionId) === String(item.decisionId));
      const perspectivePrompt = String(perspectiveRow?.promptContract || '');
      const decisionConfidence = actorView ? item.confidence : perspectiveRow?.confidence;
      const decisionTruth = String(actorView ? item.truthLabel : perspectiveRow?.truthLabel || '');
      return <details key={String(item.decisionId)}><summary><span>{number.format(Number(item.ordinal))}</span><b>{seatLabel[String(item.actingSeat)] || String(item.actingSeat)}</b><i>{decisionLabel[String(item.decisionKind)] || t('workflow.m_b587409de388')}</i>{!isNativeV3 && <em>{truthLabel[decisionTruth] || decisionTruth}</em>}</summary><div className="decision-audit-grid">
        <div><span>{t('workflow.research_actual_action')}</span><b dir="ltr">{actionLabel(item.actualAction as JsonObject)}</b></div>{actorView && <div><span>{t('workflow.research_recommended_action')}</span><b dir="ltr">{actionLabel(item.recommendedAction as JsonObject)}</b></div>}
        {!isNativeV3 && <div><span>{t('workflow.research_decision_confidence')}</span><b>{percent(decisionConfidence)}</b></div>}{!isNativeV3 && <div><span>{t('workflow.research_decision_truth')}</span><b>{truthLabel[decisionTruth] || decisionTruth}</b><small>{item.forced ? t('workflow.research_forced_play') : t('workflow.research_not_forced_play')}</small></div>}
        {actorView && !isNativeV3 && challengeScore && <div className="wide research-score-breakdown"><span>{t('workflow.research_decision_score_breakdown')}</span><b>{number.format(Math.round(Number((challengeScore.score as JsonObject)?.score || 0)))}</b><div>{Object.entries(scoreDimensions).sort((left, right) => Number(scoreContributions[right[0]] || 0) - Number(scoreContributions[left[0]] || 0)).map(([key, value]) => <span key={key} title={`${t('workflow.research_score_contribution')}: ${metric(scoreContributions[key])}`}><strong>{researchScoreDimensionLabel(key)}</strong><i>{metric(value)}</i></span>)}</div></div>}
        <div className="wide decision-context"><span>{t('workflow.research_decision_context')}</span><p>{t('workflow.research_observer_safe_context')}</p>{actorView ? <>{!isNativeV3 && <><div className="information-set-proof"><b>{t('workflow.research_information_set_status')}</b><em className={informationSetStatus === 'incomplete' ? 'rejected' : 'qualified'}>{tVariant('workflow.research_information_set', informationSetStatus)}</em><small>{t('workflow.research_information_set_support')}: {number.format(Number(informationSet.hiddenDealSupport || 0))}</small></div><div className="information-set-proof"><b>{t('workflow.research_sequential_response_status')}</b><em className={sequentialResponseStatus === 'incomplete' ? 'rejected' : 'qualified'}>{tVariant('workflow.research_sequential_response', sequentialResponseStatus)}</em><small>{t('workflow.research_exact_continuation_share')}: {percent(sequentialResponse.minimumExactShare)}</small></div></>}{legalActions.length > 0 && <><b>{t('workflow.research_legal_choices')}</b><div className="decision-action-chips">{legalActions.map((action, index) => <span dir="ltr" key={`${actionLabel(action)}-${index}`}>{actionLabel(action)}</span>)}</div></>}</> : <><p className="drawer-lead audit-only">{t('workflow.laffa_library_audit_only')}</p>{perspectivePrompt && <div className="information-set-proof"><b>{t('workflow.research_perspective_title')}</b><em>{promptLabels[perspectivePrompt] || perspectivePrompt}</em></div>}</>}</div>
        {actorView && !isNativeV3 && <div className="wide"><span>{t('workflow.research_selection_reason')}</span><p>{String((item.selectionExplanation as JsonObject)?.summaryAr || t('workflow.research_no_recorded_reason'))}</p></div>}
        {actorView && !isNativeV3 && <div className="wide"><span>{t('workflow.research_analysis_reason')}</span><p>{String(analysis.textAr || '')}</p></div>}
        {actorView && item.forcedExplanation && <div className="wide forced-explanation"><span>{t('workflow.research_forced_play')}</span><p>{String((item.forcedExplanation as JsonObject)?.textAr || (item.forcedExplanation as JsonObject)?.summaryAr || '')}</p></div>}
        {actorView && !isNativeV3 && alternatives.length > 0 && <div className="wide decision-alternatives"><span>{t('workflow.research_alternative_outcomes')}</span><div className="decision-alternative-table"><div className="head"><b>{t('workflow.research_legal_choices')}</b><b>{t('workflow.research_expected_margin')}</b><b>{t('workflow.research_contract_probability')}</b><b>{t('workflow.research_match_probability')}</b></div>{alternatives.map((alternative, index) => { const action = (alternative.action || {}) as JsonObject; return <div key={`${actionLabel(action)}-${index}`}><b dir="ltr"><i>{number.format(Number(alternative.rank || index + 1))}</i>{actionLabel(action)}</b><span>{metric(alternative.expectedMargin)}</span><span>{percent(alternative.contractSuccessProbability)}</span><span>{percent(alternative.matchWinProbability)}</span></div>; })}</div><small>{t('workflow.research_sample_support')}: {number.format(Math.max(0, ...alternatives.map((entry) => Number(entry.samples || 0))))} · {t('workflow.research_search_nodes')}: {number.format(Math.max(0, ...alternatives.map((entry) => Number(entry.searchNodes || 0))))}</small></div>}
        {actorView && continuation.length > 0 && <div className="wide decision-continuation"><span>{t('workflow.research_causal_continuation')}</span><div>{continuation.slice(0, 12).map((card, index) => <i dir="ltr" key={`${card}-${index}`}>{card}</i>)}{continuation.length > 12 && <small>+{number.format(continuation.length - 12)} {t('workflow.research_more_cards')}</small>}</div></div>}
        {actorView && Array.isArray(item.tacticalEvidence) && item.tacticalEvidence.length > 0 && <div className="wide"><span>{t('workflow.research_tactical_evidence')}</span><div className="detail-tags">{(item.tacticalEvidence as JsonObject[]).map((evidence) => <span key={String(evidence.causalProofHash)}>{tacticDisplayName(String(evidence.kind))} · {number.format(Math.round(Number(evidence.magnitude || 0)))}</span>)}</div></div>}
        <details className="wide decision-provenance"><summary>{t('workflow.research_decision_provenance')}</summary><div><span>{t('workflow.research_rule_basis')}: {Array.isArray(item.rulePaths) ? (item.rulePaths as string[]).join(' · ') : '—'}</span><span dir="ltr">{String(provenance.agentRelease || '—')}</span><span dir="ltr">{String(provenance.analyzerId || '—')}@{String(provenance.analyzerVersion || '—')}</span><span dir="ltr">engine@{String(provenance.engineVersion || '—')}</span></div></details>
        {item.decisionKind === 'play' && !item.forced && <div className="wide drawer-actions"><button type="button" className="secondary" disabled={!observerSelectionReady} onClick={() => onPrepare(observerSeat, Number(item.ordinal))}>{t(actorView && publicationEligible ? 'workflow.research_render_decision' : 'workflow.research_render_decision_audit')}</button></div>}
      </div></details>;
    })}</div>
  </>;
}

export function ResearchPlayProof({proof}: {proof: JsonObject}) {
  const solver = String(proof.solverStatus || '');
  const projectSolver = String(proof.projectSolverStatus || '');
  const points = (value: JsonValue | undefined) => Array.isArray(value) && value.length === 2
    ? `${t('workflow.research_team_us')} ${number.format(Number(value[0]))} · ${t('workflow.research_team_them')} ${number.format(Number(value[1]))}`
    : '—';
  const solverLabel = (value: string) => value === 'optimal'
    ? t('workflow.research_play_proof_optimal')
    : value === 'feasible'
      ? t('workflow.research_play_proof_feasible')
      : value || '—';
  return <div className="research-play-proof"><p className="drawer-lead">{t('workflow.research_play_proof_help')}</p><div className="drawer-stats"><Stat label={t('workflow.research_play_proof_outcome')} value={String(proof.outcome) === 'matched' ? t('workflow.research_play_proof_matched') : String(proof.outcome || '—')}/><Stat label={t('workflow.research_project_proof_solver')} value={solverLabel(projectSolver)}/><Stat label={t('workflow.research_play_proof_solver')} value={solverLabel(solver)}/><Stat label={t('workflow.research_project_proof_reveals')} value={number.format(Number(proof.projectRevealCount || 0))}/></div><div className="evidence-list"><div><span>{t('workflow.research_project_proof_points')}</span><b>{points(proof.projectPoints)}</b></div><div><span>{t('workflow.research_project_proof_baloot')}</span><b>{points(proof.balootPoints)}</b></div><div><span>{t('workflow.research_project_proof_plan')}</span><b dir="ltr">{String(proof.projectPlanHash || '—').slice(0, 16)}</b></div><div><span>{t('workflow.research_play_proof_plan')}</span><b dir="ltr">{String(proof.playPlanHash || '—').slice(0, 16)}</b></div><div><span>{t('workflow.research_play_proof_evidence')}</span><b dir="ltr">{String(proof.evidenceHash || '').slice(0, 12)}</b></div><div><span>{t('workflow.research_play_proof_artifact')}</span><b>{t('workflow.research_play_proof_saved')}</b></div></div></div>;
}

const previewSuitPrefix: Record<string, string> = {
  spades: 'S',
  hearts: 'H',
  diamonds: 'D',
  clubs: 'C',
};
const previewSeatRing: SeatName[] = ['north', 'west', 'south', 'east'];

function previewActualAtVisual(visual: SeatName, observer: SeatName): SeatName {
  const observerIndex = previewSeatRing.indexOf(observer);
  return previewSeatRing[
    (previewSeatRing.indexOf(visual) + observerIndex - 2 + previewSeatRing.length)
    % previewSeatRing.length
  ];
}

function previewVisualOfActual(actual: SeatName, observer: SeatName): SeatName {
  const observerIndex = previewSeatRing.indexOf(observer);
  return previewSeatRing[
    (previewSeatRing.indexOf(actual) - observerIndex + 2 + previewSeatRing.length)
    % previewSeatRing.length
  ];
}

function previewRelativePair(values: number[], observer: SeatName): number[] {
  const observerTeamIndex = observer === 'north' || observer === 'south' ? 0 : 1;
  return observerTeamIndex === 0 ? values : [values[1] ?? 0, values[0] ?? 0];
}

function previewCardAsset(card: JsonObject) {
  const suit = String(card.suit || '');
  const prefix = previewSuitPrefix[suit];
  const code = String(card.code || '').trim().toUpperCase();
  const rank = prefix && code.startsWith(prefix) ? code.slice(prefix.length) : '';
  return ['7', '8', '9', '10', 'J', 'Q', 'K', 'A'].includes(rank)
    ? `/card-assets/${encodeURIComponent(rank)}_${encodeURIComponent(suit)}.svg`
    : '';
}

export function PreviewCardFace({card}: {card: JsonObject}) {
  const source = previewCardAsset(card);
  const label = String(card.label || card.code || '');
  return <span className="research-preview-card" aria-label={label}>
    {source ? <img src={source} alt={label} draggable={false}/> : <b dir="ltr">{label}</b>}
  </span>;
}

function ResearchSemanticPreview({
  prepared,
}: {
  prepared: ResearchPreparedRender;
}) {
  const scenes = Array.isArray(prepared.preview.scenes)
    ? prepared.preview.scenes as JsonObject[]
    : [];
  const scene = scenes.find((item) => String(item.stage) === 'decision') || scenes[0];
  const layers = Array.isArray(scene?.layers) ? scene.layers as JsonObject[] : [];
  const table = (layers.find((item) => String(item.kind) === 'table')?.content || {}) as JsonObject;
  const handLayer = (layers.find((item) => String(item.kind) === 'cards')?.content || {}) as JsonObject;
  const hand = Array.isArray(handLayer.cards) ? handLayer.cards as JsonObject[] : [];
  const trick = (table.currentTrick || {}) as JsonObject;
  const plays = Array.isArray(trick.plays) ? trick.plays as JsonObject[] : [];
  const observer = prepared.observerSeat;
  const score = previewRelativePair(
    Array.isArray(table.matchScore) ? table.matchScore.map(Number) : [],
    observer,
  );
  const contract = String(table.contract || '');
  const currentPlayer = isSeatName(table.currentPlayer) ? table.currentPlayer : null;
  return <div className="research-semantic-preview" dir="rtl">
    <div className="research-preview-score"><b>{t('workflow.research_team_us')} {number.format(Number(score[0] || 0))}</b><span>{contractLabel[contract] || contract}</span><b>{t('workflow.research_team_them')} {number.format(Number(score[1] || 0))}</b></div>
    {(['north', 'west', 'east', 'south'] as const).map((visualSeat) => { const actualSeat = previewActualAtVisual(visualSeat, observer); return <span key={visualSeat} className={`research-preview-seat ${visualSeat} ${currentPlayer && previewVisualOfActual(currentPlayer, observer) === visualSeat ? 'active' : ''}`}><i>{seatLabel[actualSeat]?.slice(0, 1) || actualSeat.slice(0, 1)}</i><b>{seatLabel[actualSeat]}</b></span>; })}
    <div className="research-preview-table">
      {plays.map((play, index) => { const card = (play.card || play) as JsonObject; const actualSeat = isSeatName(play.seat) ? play.seat : observer; const visualSeat = previewVisualOfActual(actualSeat, observer); return <span key={`${actualSeat}-${String(card.code || index)}`} className={`research-preview-play ${visualSeat}`}><PreviewCardFace card={card}/></span>; })}
    </div>
    <div className="research-preview-hand">{hand.map((card, index) => <span key={String(card.code || index)} style={{'--card-index': index, '--card-count': hand.length} as CSSProperties}><PreviewCardFace card={card}/></span>)}</div>
  </div>;
}

export function ResearchRenderPreview({
  prepared,
  onRender,
  busy,
}: {
  prepared: ResearchPreparedRender;
  busy: boolean;
  onRender: (item: ResearchPreparedRender) => Promise<void>;
}) {
  const code = prepared.resultCode || prepared.laffaCode || String(prepared.source.code || '—');
  const sourceKind = String(prepared.source.originKind || prepared.source.kind || '');
  const publicationBlockers = [...new Set([
    ...prepared.qualificationFailures.map(String),
    ...(Array.isArray(prepared.publicationBlockers) ? prepared.publicationBlockers.map(String) : []),
    ...(Array.isArray(prepared.source.qualificationFailures) ? prepared.source.qualificationFailures.map(String) : []),
    ...(Array.isArray(prepared.source.publicationBlockers) ? prepared.source.publicationBlockers.map(String) : []),
  ])];
  return <>
    <p className={`drawer-lead ${prepared.publicationEligible ? '' : 'audit-only'}`}>{prepared.publicationEligible ? t('workflow.laffa_library_content_candidate') : t(prepared.qualificationStatus === 'qualified' ? 'workflow.laffa_library_qualification_audit' : 'workflow.laffa_library_audit_only')}</p>
    <ResearchSemanticPreview prepared={prepared}/>
    <div className="drawer-stats">
      <Stat label={t('workflow.research_column_code')} value={code}/>
      <Stat label={t('workflow.research_column_decisions')} value={number.format(prepared.decisionOrdinal)}/>
      <Stat label={t('workflow.research_video_scenes')} value={number.format(Number(prepared.preview.sceneCount || 0))}/>
      <Stat label={t('workflow.research_render_purpose')} value={tVariant('workflow.research_render', prepared.renderPurpose)}/>
      <Stat label={t('workflow.laffa_library_column_qualification')} value={prepared.qualificationStatus === 'qualified' && !prepared.publicationEligible ? t('workflow.research_status_qualified') : laffaQualificationLabel[prepared.qualificationStatus] || prepared.qualificationStatus}/>
      {prepared.source.attemptStatus && <Stat label={t('workflow.research_column_status')} value={statusLabel[prepared.source.attemptStatus] || prepared.source.attemptStatus}/>}
    </div>
    <div className="evidence-list">
      <div><span>{t('workflow.research_filter_observer_seat')}</span><b>{seatLabel[prepared.observerSeat] || prepared.observerSeat}</b></div>
      <div><span>{t('workflow.research_video_source')}</span><b>{laffaSourceLabel[sourceKind] || sourceKind}</b></div>
      <div><span>{t('workflow.batch_profile_render_policy')}</span><b>{t('workflow.batch_render_policy_on_demand')}</b></div>
    </div>
    {publicationBlockers.length > 0 && <><h3>{t(prepared.qualificationStatus === 'qualified' ? 'workflow.laffa_library_qualification_audit' : 'workflow.laffa_library_qualification_failures')}</h3><ul className="audit-list">{publicationBlockers.map((blocker) => <li key={blocker}>{researchFailureLabel(blocker)}</li>)}</ul></>}
    <div className="drawer-actions"><button type="button" className="primary" disabled={busy} onClick={() => onRender(prepared)}>{t('workflow.batch_render_selected')}</button></div>
  </>;
}
