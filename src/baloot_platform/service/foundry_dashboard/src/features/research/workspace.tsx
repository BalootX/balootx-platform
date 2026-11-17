import {Fragment, type FormEvent, useEffect, useMemo, useRef, useState} from 'react';

import {
  mergeResearchPatch,
  researchCatalogFieldApplicable,
  researchCatalogInitial,
  researchCatalogObjectInitial,
  researchCatalogReady,
  researchRelationshipActive,
  researchValueAtPath,
} from './catalog';
import type {
  ResearchCapabilities,
  ResearchConfigurationSelection,
  ResearchEstimate,
  ResearchRecipe,
  ResearchResult,
  ResearchResultGroup,
  ResearchResultPage,
  ResearchRun,
  ResearchStats,
  ResearchTacticDraft,
  ResearchTacticKind,
  SemanticDisagreement,
} from './model';
import {isNativeResearchAuthoringBackend} from './model';
import type {JsonObject, JsonValue} from '../../shared/json';
import {humanKey, measured} from '../../shared/format';
import {hasFoundryCopy, number, selectResearchWorkflowCopy, t, tCatalog} from '../../shared/localization';
import {
  contractLabel,
  disagreementLabel,
  Empty,
  familyMeta,
  InfoBadge,
  laffaLabel,
  projectLabel,
  ScorePill,
  seatLabel,
  Stat,
  statusLabel,
  suitName,
  tacticDisplayName,
  truthLabel,
  type DrawerState,
} from '../../shared/presentation';
import type {ConfigBundle, ConfigProfile} from '../configuration/model';
import {
  renderJobReferencesResearch,
  renderJobReferencesSource,
  type ArchivedLaffa,
  type RenderJob,
} from '../videos/model';
import {PreviewCardFace} from './audit';
import {researchRecipeErrorMessage} from './errorCopy';
import {preferredNativeResearchBundle, resultAllowsNormalMedia, resultHasReplayTrace} from './ortoolsScenarioUi';
import {ScenarioStudio, scenarioCatalogReady} from './scenarioStudio';

export type ResearchWorkspaceRequest = <T>(
  path: string,
  init?: RequestInit,
) => Promise<T>;

const distinctPerformanceCopyKeys = [
  'workflow.research_performance_distinct_scenarios',
  'workflow.research_performance_distinct_wall_rate',
  'workflow.research_performance_distinct_cpu_rate',
] as const;

export function ResearchPerformanceSummary({stats,onRuns,onResults}:{stats:ResearchStats;onRuns:()=>void;onResults:()=>void}) {
  const value=stats.performance;
  const hasPerformanceEvidence=value.shardCount>0;
  const hasDistinctPerformanceCopy=distinctPerformanceCopyKeys.every(hasFoundryCopy);
  const solver=[...Object.entries(value.solverStatusDistribution.construction).map(([status,count])=>`${status}: ${number.format(count)}`),...Object.entries(value.solverStatusDistribution.playCompiler).map(([status,count])=>`${status}: ${number.format(count)}`)].join(' · ');
  return <section className="panel research-performance-summary">
    <header><div><h3>{t("workflow.research_performance_title")}</h3><p>{t("workflow.research_performance_help")}</p></div><button className="secondary" type="button" onClick={onRuns}>{t("workflow.research_performance_open_runs")}</button></header>
    <div className="research-performance-funnel">
      <button type="button" onClick={onRuns}><span>{t("workflow.research_label_seeds")}</span><b>{hasPerformanceEvidence?number.format(value.attemptedSeeds):t("workflow.research_metric_unavailable")}</b></button>
      <button type="button" onClick={onResults}><span>{t("workflow.research_performance_results")}</span><b>{hasPerformanceEvidence?number.format(value.resultCount):t("workflow.research_metric_unavailable")}</b></button>
      <button type="button" onClick={onResults}><span>{t("workflow.research_status_qualified")}</span><b>{hasPerformanceEvidence?number.format(value.qualifiedCount):t("workflow.research_metric_unavailable")}</b></button>
      {hasDistinctPerformanceCopy?<>
        <button type="button" data-performance-metric="distinct-scenarios" onClick={onResults}><span>{t("workflow.research_performance_distinct_scenarios")}</span><b>{value.qualifiedDistinctScenarioCount===null?t("workflow.research_metric_unavailable"):number.format(value.qualifiedDistinctScenarioCount)}</b></button>
        <button type="button" data-performance-metric="distinct-wall-rate" onClick={onResults}><span>{t("workflow.research_performance_distinct_wall_rate")}</span><b>{value.qualifiedDistinctScenariosPerWallHour===null?t("workflow.research_metric_unavailable"):String(value.qualifiedDistinctScenariosPerWallHour)}</b></button>
        <button type="button" data-performance-metric="distinct-cpu-rate" onClick={onResults}><span>{t("workflow.research_performance_distinct_cpu_rate")}</span><b>{value.qualifiedDistinctScenariosPerCpuHour===null?t("workflow.research_metric_unavailable"):String(value.qualifiedDistinctScenariosPerCpuHour)}</b></button>
      </>:null}
      <button type="button" onClick={onResults}><span>{t("workflow.batch_yield_per_100")}</span><b>{hasPerformanceEvidence?measured(value.qualifiedPer100Attempts,1):t("workflow.research_metric_unavailable")}</b></button>
      <button type="button" onClick={onResults}><span>{t("workflow.research_performance_duplicates")}</span><b>{value.duplicateCount===null?t("workflow.research_metric_unavailable"):number.format(value.duplicateCount)}</b></button>
      <button type="button" onClick={onRuns}><span>{t("workflow.research_performance_cost_proxy")}</span><b>{measured(value.costProxy.wallSecondsPerQualified)}</b></button>
    </div>
    <div className="research-performance-details">
      {Object.entries(value.timing).map(([stage,timing])=><button type="button" key={stage} onClick={onRuns}><b>{researchPerformanceStageLabel(stage)}</b><span>{t("workflow.research_performance_p50")} {measured(timing.p50Seconds)} · {t("workflow.research_performance_p95")} {measured(timing.p95Seconds)}</span></button>)}
      <button type="button" onClick={onRuns}><b>{t("workflow.research_performance_solver")}</b><span>{solver||t("workflow.research_metric_unavailable")}</span></button>
      <button type="button" onClick={onRuns}><b>{t("workflow.research_performance_operations")}</b><span>{t("workflow.research_performance_repeated")} {number.format(value.operations.repeatedExecutionCount)} · {t("workflow.research_performance_cancelled")} {number.format(value.operations.currentCancelledShardCount)} · {t("workflow.research_performance_retries")} {value.operations.retryCount===null?t("workflow.research_metric_unavailable"):number.format(value.operations.retryCount)}</span></button>
    </div>
  </section>;
}

function researchPerformanceStageLabel(stage:string) {
  const labels:Record<string,string>={construction:t("workflow.research_stage_construction"),simulation:t("workflow.research_stage_simulation"),parity:t("workflow.research_stage_parity"),analysis:t("workflow.research_stage_analysis"),tactical:t("workflow.research_lane_deep_seconds"),totalWall:t("workflow.research_performance_total_wall")};
  return labels[stage]||stage;
}

type ResearchStudioProps = {
  bundles:ConfigBundle[];
  runtimeBundleRevisionId:string|null;
  recipes:ResearchRecipe[];
  runs:ResearchRun[];
  disagreements:SemanticDisagreement[];
  stats:ResearchStats|null;
  busy:boolean;
  setBusy:(value:boolean)=>void;
  setMessage:(value:string)=>void;
  setDrawer:(value:DrawerState)=>void;
  request:ResearchWorkspaceRequest;
  createRecipeAndRun:(payload:JsonObject)=>Promise<{recipe:ResearchRecipe;run:ResearchRun}>;
  refresh:()=>Promise<void>;
  onOpenResult:(item:ResearchResult)=>void;
  onOpenRun:(item:ResearchRun)=>void;
  onOpenDisagreement:(item:SemanticDisagreement)=>void;
};

function ResearchConfigurationDetails({bundle,selection,profiles}:{bundle:ConfigBundle;selection:ResearchConfigurationSelection;profiles:ConfigProfile[]}) {
  const components=Object.entries(bundle.componentRevisions).sort(([left],[right])=>(familyMeta[left]?.label||left).localeCompare(familyMeta[right]?.label||right,'ar'));
  return <div className="research-bundle-details">
    <p>{t("workflow.research_bundle_details_help")}</p>
    <div className="drawer-stats"><Stat label={t("workflow.research_bundle_selected")} value={`${bundle.bundleId} · ${bundle.version}`}/><Stat label={t("workflow.research_bundle_components_count")} value={number.format(components.length)}/></div>
    <section className="context-proof"><i>✓</i><div><b>{t("workflow.research_bundle_identity_ready")}</b><span>{t("workflow.research_bundle_identity_help")}</span></div></section>
    <h3>{t("workflow.research_bundle_components")}</h3>
    <div className="research-bundle-components">{components.map(([family,revisionId])=>{const profile=profiles.find((item)=>item.revisionId===revisionId);return <div key={family}><span>{familyMeta[family]?.label||humanKey(family)}</span><b>{profile?.version||t("workflow.research_metric_unavailable")}</b><small>{profile?(statusLabel[profile.status]||profile.status):t("workflow.research_metric_unavailable")}</small></div>;})}</div>
    <small>{t("workflow.research_bundle_selected")} · <bdi>{selection.bundleRevisionId}</bdi></small>
  </div>;
}

export function ResearchStudio(props:ResearchStudioProps) {
  const {bundles,runtimeBundleRevisionId,request,setDrawer,setMessage}=props;
  const publishedBundles=useMemo(()=>bundles
    .filter((item)=>item.status==='published')
    .sort((left,right)=>Number(right.sequence||0)-Number(left.sequence||0)||right.version.localeCompare(left.version,undefined,{numeric:true})),[bundles]);
  const nativeBundle=publishedBundles.find((item)=>item.bundleId==='baloot_ortools_research_native_v3');
  const preferredNativeBundle=preferredNativeResearchBundle(publishedBundles);
  const [selectedBundleRevisionId,setSelectedBundleRevisionId]=useState('');
  const [capabilities,setCapabilities]=useState<ResearchCapabilities|null>(null);
  const [capabilityState,setCapabilityState]=useState<'idle'|'loading'|'ready'|'failed'>('idle');
  const [requestRevision,setRequestRevision]=useState(0);
  const [bundleHelp,setBundleHelp]=useState<string|null>(null);
  const requestSequence=useRef(0);
  const selectedBundle=publishedBundles.find((item)=>item.revisionId===selectedBundleRevisionId)||null;
  const officialRulesRevisionId=nativeBundle?.componentRevisions.rules??null;
  useEffect(()=>{
    setSelectedBundleRevisionId((current)=>{
      if(current&&publishedBundles.some((item)=>item.revisionId===current)) return current;
      const initial=preferredNativeBundle||publishedBundles.find((item)=>item.revisionId===runtimeBundleRevisionId)||publishedBundles[0];
      return initial?.revisionId||'';
    });
  },[publishedBundles,runtimeBundleRevisionId,preferredNativeBundle]);
  useEffect(()=>{
    selectResearchWorkflowCopy(null);
    if(!selectedBundleRevisionId){setCapabilities(null);setCapabilityState('idle');return;}
    const controller=new AbortController();
    const sequence=++requestSequence.current;
    setCapabilities(null);setCapabilityState('loading');
    request<ResearchCapabilities>(`/api/v2/domain-capabilities?bundleRevisionId=${encodeURIComponent(selectedBundleRevisionId)}`,{signal:controller.signal})
      .then((value)=>{if(sequence!==requestSequence.current)return;selectResearchWorkflowCopy(value.research.scenarioResearchV4?.localization??null,selectedBundleRevisionId);setCapabilities(value);setCapabilityState('ready');})
      .catch(()=>{if(controller.signal.aborted||sequence!==requestSequence.current)return;selectResearchWorkflowCopy(null);setCapabilities(null);setCapabilityState('failed');});
    return()=>controller.abort();
  },[requestRevision,selectedBundleRevisionId]);
  const selection=capabilities?.research.configurationSelection||null;
  const scenarioResearchV4=capabilities?.research.scenarioResearchV4;
  const authorityMatches=Boolean(
    selectedBundle
    && selection
    && selection.ready===true
    && selection.schemaVersion===1
    && selection.bundleRevisionId===selectedBundle.revisionId
    && selection.bundleId===selectedBundle.bundleId
    && selection.bundleVersion===selectedBundle.version
    && (!scenarioResearchV4
      || scenarioResearchV4.backendIdentity.bundleRevisionId===selectedBundle.revisionId)
  );
  const scenarioReady=isNativeResearchAuthoringBackend(scenarioResearchV4?.backendIdentity.backendId)
    && scenarioCatalogReady(
      capabilities?.research.scenarioConfiguration,
      capabilities?.research.recipeConfiguration,
      scenarioResearchV4,
    );
  const openBundleDetails=async()=>{
    if(!selectedBundle||!authorityMatches||!selection)return;
    setDrawer({
      kind:'research-configuration',
      title:t("workflow.research_bundle_details_title"),
      subtitle:`${selectedBundle.bundleId} · ${selectedBundle.version}`,
      content:<p className="drawer-lead">{t("workflow.research_bundle_loading")}</p>,
      audit:{bundle:selectedBundle as unknown as JsonValue,configurationSelection:selection as unknown as JsonValue},
    });
    try {
      const profilePage=await request<{items:ConfigProfile[]}>('/api/v2/configuration/profiles?limit=1000');
      setDrawer({
        kind:'research-configuration',
        title:t("workflow.research_bundle_details_title"),
        subtitle:`${selectedBundle.bundleId} · ${selectedBundle.version}`,
        content:<ResearchConfigurationDetails bundle={selectedBundle} selection={selection} profiles={profilePage.items}/>,
        audit:{bundle:selectedBundle as unknown as JsonValue,configurationSelection:selection as unknown as JsonValue},
      });
    } catch(error) { setMessage(researchRecipeErrorMessage(error)); }
  };
  const capabilityBlocked=capabilityState==='ready'&&(!authorityMatches||!scenarioReady);
  return <>
    <section className="panel research-bundle-selector" aria-labelledby="research-bundle-title" data-field-path="configuration.bundleRevisionId">
      <header><span className="eyebrow">{t("workflow.research_bundle_selected")}</span><h3 id="research-bundle-title">{t("workflow.research_bundle_title")}</h3><p>{t("workflow.research_bundle_help")}</p></header>
      <div className="research-bundle-control"><label htmlFor="research-bundle-select">{t("workflow.research_bundle_label")}</label><select id="research-bundle-select" aria-describedby="research-bundle-reset" value={selectedBundleRevisionId} disabled={!publishedBundles.length} onChange={(event)=>setSelectedBundleRevisionId(event.target.value)}>{publishedBundles.map((item)=><option key={item.revisionId} value={item.revisionId}>{item.bundleId} · {item.version} · {item.revisionId}{item.revisionId===runtimeBundleRevisionId?` · ${t("workflow.research_bundle_current")}`:''}</option>)}</select><small id="research-bundle-reset">{t("workflow.research_bundle_reset_notice")}</small><div className="research-bundle-meta">{selectedBundle&&<span>{t("workflow.research_bundle_selected")} · <bdi>{selectedBundle.bundleId} · {selectedBundle.version} · {selectedBundle.revisionId}</bdi></span>}{selectedBundle?.revisionId===runtimeBundleRevisionId&&<span className="current">{t("workflow.research_bundle_current")}</span>}<InfoBadge id="research-bundle-help" text={t("workflow.research_bundle_help")} example={t("workflow.research_bundle_example")} openHelp={bundleHelp} setOpenHelp={setBundleHelp}/></div></div>
      <div className="research-bundle-status">{authorityMatches&&scenarioReady?<><strong>{t("workflow.research_bundle_identity_ready")}</strong><small>{t("workflow.research_bundle_identity_help")}</small><button type="button" className="secondary" onClick={()=>void openBundleDetails()}>{t("workflow.research_bundle_details")}</button></>:capabilityState==='loading'?<small>{t("workflow.research_bundle_loading")}</small>:publishedBundles.length===0?<small>{t("workflow.research_bundle_missing")}</small>:<><strong>{t("workflow.research_bundle_load_failed")}</strong><small>{capabilityBlocked?t("workflow.research_configuration_unavailable_help"):t("workflow.research_bundle_mismatch")}</small><button type="button" className="secondary" onClick={()=>setRequestRevision((value)=>value+1)}>{t("workflow.research_bundle_retry")}</button></>}</div>
    </section>
    {authorityMatches&&scenarioReady&&selectedBundle&&capabilities
      ? <ScenarioStudio key={selectedBundle.revisionId} configuration={capabilities.research.scenarioConfiguration} recipeConfiguration={capabilities.research.recipeConfiguration} scenarioResearchV4={scenarioResearchV4} selectedBundle={selectedBundle} availableBundles={publishedBundles} officialRulesRevisionId={officialRulesRevisionId} rulesHash={capabilities.rulesHash} busy={props.busy} request={request} createRecipeAndRun={props.createRecipeAndRun} setBusy={props.setBusy} setMessage={props.setMessage} refresh={props.refresh} onOpenRun={props.onOpenRun} onOpenResult={props.onOpenResult} runs={props.runs}/>
      : <section className="panel research-bundle-loading" role={capabilityBlocked||capabilityState==='failed'?'alert':'status'}>{capabilityState==='loading'?t("workflow.research_bundle_loading"):publishedBundles.length===0?t("workflow.research_bundle_missing"):t("workflow.research_bundle_mismatch")}</section>}
    {props.disagreements.length>0&&<section className="panel" aria-labelledby="research-disagreements-title"><header><span className="eyebrow">{t("workflow.research_quarantine_status")}</span><h3 id="research-disagreements-title">{t("workflow.research_quarantine_title")}</h3><p>{t("workflow.research_quarantine_retained")}</p></header><div className="research-run-events"><ol>{props.disagreements.map((item)=><li key={item.disagreementId}><span><b dir="ltr">{item.disagreementId}</b><small>{disagreementLabel[item.effectiveCategory||item.category]||item.category}</small></span><button type="button" className="secondary" onClick={()=>props.onOpenDisagreement(item)}>{t("workflow.research_quarantine_details")}</button></li>)}</ol></div></section>}
  </>;
}

function ResearchResultTableBody({items,grouping,onOpen,onPrepare,renders=[]}:{items:ResearchResult[];grouping:'none'|'run'|'truth'|'tactic'|'classification';onOpen:(item:ResearchResult)=>void;onPrepare?:(item:ResearchResult)=>void;renders?:RenderJob[]}) {
  const groupKey=(item:ResearchResult) => {
    if(grouping==='run') return item.runCode;
    if(grouping==='truth') return item.truthLabel;
    if(grouping==='tactic') return item.tacticalTags?.[0] || 'none';
    if(grouping==='classification') return item.classification || 'none';
    return 'all';
  };
  const groups=new Map<string,ResearchResult[]>();
  items.forEach((item)=>{const key=groupKey(item);groups.set(key,[...(groups.get(key)||[]),item]);});
  const title=(key:string) => {
    if(grouping==='truth') return truthLabel[key]||key;
    if(grouping==='tactic') return key==='none'?t("workflow.research_group_none"):tacticDisplayName(key);
    if(grouping==='classification') return key==='none'?t("workflow.research_group_none"):(laffaLabel[key]||key);
    return key;
  };
  return <tbody>{Array.from(groups.entries()).map(([key,rows])=><Fragment key={key}>{grouping!=='none'&&<tr className="research-group-row"><th colSpan={onPrepare?7:6}>{title(key)} <span>{number.format(rows.length)}</span></th></tr>}{rows.map((item)=>{const related=renders.filter((job)=>renderJobReferencesResearch(job,item.resultCode));const completed=related.some((job)=>job.status==='completed');const active=related.find((job)=>['queued','running'].includes(job.status));const normalMediaAllowed=resultAllowsNormalMedia(item);const replayTraceAvailable=resultHasReplayTrace(item);return <tr key={item.resultCode} tabIndex={0} onClick={()=>onOpen(item)} onKeyDown={(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();onOpen(item);}}}><td><b dir="ltr">{item.resultCode}</b></td><td>{item.classification==='scenario_v4'?'—':<ScorePill value={item.score}/>}</td><td>{item.classification==='scenario_v4'?'—':truthLabel[item.truthLabel]||item.truthLabel}</td><td>{item.classification==='scenario_v4'?'—':`${number.format(Math.round(item.confidence*100))}${t("workflow.batch_percent_suffix")}`}</td><td>{number.format(item.decisionCount)}</td><td><span className={`status ${item.disposition}`}>{item.disposition==='qualified'?t("workflow.research_status_qualified"):t("workflow.research_status_rejected")}</span></td>{onPrepare&&<td onClick={(event)=>event.stopPropagation()}><div className="table-actions"><button type="button" className="table-status" onClick={()=>onOpen(item)}>{completed?t("workflow.laffa_library_video_ready"):active?statusLabel[active.status]||active.status:t("workflow.laffa_library_video_not_generated")}</button>{!completed&&!active&&replayTraceAvailable&&normalMediaAllowed&&<button type="button" className="table-render" onClick={()=>onPrepare(item)}>{t(item.publicationEligible===true?"workflow.research_render_complete":"workflow.research_render_audit")}</button>}</div></td>}</tr>})}</Fragment>)}</tbody>;
}

function ArchivedLaffaTable({items,renders,onOpen,onPrepare,busy}:{items:ArchivedLaffa[];renders:RenderJob[];onOpen:(item:ArchivedLaffa)=>void;onPrepare:(item:ArchivedLaffa)=>void;busy:boolean}) {
  if (!items.length) return <Empty title={t("workflow.research_empty_results")} body={t("workflow.laffa_library_all_generated_help")} />;
  return <div className="data-table-wrap"><table className="data-table"><thead><tr>
    <th>{t("workflow.laffa_library_column_code")}</th>
    <th>{t("workflow.research_label_contract")}</th>
    <th>{t("workflow.research_label_buyer")}</th>
    <th>{t("workflow.sakka_library_column_rounds")}</th>
    <th>{t("workflow.sakka_library_column_score")}</th>
    <th>{t("workflow.laffa_library_column_video")}</th>
  </tr></thead><tbody>{items.map((item) => {
    const code=item.serial||item.gameHash.slice(0,10).toUpperCase();
    const related=renders.filter((job)=>renderJobReferencesSource(job,'laffa',item.serial||item.gameHash));
    const completed=related.some((job)=>job.status==='completed');
    const active=related.find((job)=>['queued','running'].includes(job.status));
    return <tr key={item.gameHash} tabIndex={0} onClick={()=>onOpen(item)} onKeyDown={(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();onOpen(item);}}}>
      <td><b dir="ltr">{code}</b></td>
      <td>{item.contract ? contractLabel[item.contract]||item.contract : '—'}</td>
      <td>{item.buyer ? seatLabel[item.buyer]||item.buyer : '—'}</td>
      <td>{item.roundNumber ? number.format(item.roundNumber) : '—'}</td>
      <td dir="ltr">{Array.isArray(item.matchScore) ? item.matchScore.join(' — ') : '—'}</td>
      <td onClick={(event)=>event.stopPropagation()}><div className="table-actions"><button type="button" className="table-status" onClick={()=>onOpen(item)}>{completed?t("workflow.laffa_library_video_ready"):active?statusLabel[active.status]||active.status:t("workflow.laffa_library_video_not_generated")}</button>{!completed&&!active&&<button type="button" className="table-render" disabled={busy} onClick={()=>onPrepare(item)}>{t("workflow.research_render_audit")}</button>}</div></td>
    </tr>;
  })}</tbody></table></div>;
}
