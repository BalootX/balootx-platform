import {useEffect, useId, useMemo, useRef, useState, type ReactNode} from 'react';

import type {JsonObject, JsonValue} from '../../shared/json';
import {hasFoundryCopy, number, t, tCatalog} from '../../shared/localization';
import {researchRunStatusLabel, scenarioStatusLabel} from '../../shared/presentation';
import {researchRunDisplayStatus} from './model';
import type {ConfigBundle, ConfigProfile} from '../configuration/model';
import {
  renderBatchMediaEntries,
  renderBatchPlayerEntries,
  renderRetryKey,
  type RenderJob,
} from '../videos/model';
import {PreviewCardFace} from './audit';
import type {
  ExactTraceLegalActionV1,
  ExactTraceLegalActionsPreviewV1,
  ResearchRecipeConfigurationCatalog,
  ResearchEstimate,
  ResearchRecipe,
  ResearchRecipeV4Document,
  ResearchRecipeV4ConversionPreview,
  ResearchRecipeValidationV2,
  ResearchCandidateProgressReceipt,
  ResearchResult,
  ResearchResultPage,
  ResearchRun,
  ResearchScenarioConfigurationV1,
  ScenarioDiagnosticV1,
  ScenarioResearchV4CapabilityV1,
  ScenarioStudioPredicateDefinition,
  ResearchV4ResultFacetName,
  ResearchV4ResultFilters,
  ScenarioAuthoringCatalogV1,
  ResearchLessonTagQuery,
  ResearchGenerationRequest,
} from './model';
import {appendLessonTagQuery, LessonTagQueryControl, lessonTagCatalogReady, lessonTagResultLabels} from './lessonTags';
import {appendFamilySearchAuctionSeed, familySearchAuctionOpenings, familySearchAuctionSeedLimit, familySearchPayload, familySearchTagCatalog, frozenFamilyGeneration, matchingSearchFamilies, readFamilySearchRequest} from './familyGeneration';
import {
  researchCatalogInitial,
  type ResearchFieldCatalog,
} from './catalog';
import {
  CardSelectorV1Control,
  isCardSelectorSchema,
  type CardSelectorError,
} from './cardSelector';
import {
  appendExactTraceLegalAction,
  undoRecordedEscalationOpportunity,
  isOrToolsScenarioCapability,
  mergeScenarioPredicateDefinitions,
  preferredScenarioExecutionCombination,
  preferredScenarioRecipeCode,
  preferredPlayingQualityBudget,
  playingQualityBudgetDefaultMode,
  projectExistsControlSchema,
  resultRuntimeCompatibilityStatus,
  scenarioOrdinalDisplayValue,
  scenarioOrdinalWireValue,
  scenarioCompiledStages,
  scenarioDecisionSelectorSchema,
  scenarioExecutionCombinations,
  scenarioExecutionContractValues,
  scenarioExecutionSupported,
  scenarioPredicateArgumentValues,
  scenarioPredicateCausalVerificationAvailable,
  scenarioPredicateSupported,
  scenarioLaunchBlockMessageKey,
  unsupportedScenarioPredicateIds,
} from './ortoolsScenarioUi';
import {
  rulesAuthorityIdentity,
  rulesCertificationBoundary,
  semanticRulesDelta,
  type ProfileComparison,
  type ProfileComparisonChange,
  type RulesAuthorityIdentity,
  type RulesCertificationBoundary,
} from './ruleAuthority';

type PredicateDraft = {
  predicateId: string;
  kind: string;
  version: 1;
  hardness: 'hard' | 'prefer' | 'report';
  arguments: JsonObject;
};

type ExpressionGroupDraft = {
  nodeId: string;
  op: ScenarioOperator;
  minimum: number;
  maximum: number;
  children: ScenarioExpressionDraft[];
};

const resultFacetParameters: Record<keyof ResearchV4ResultFilters,ResearchV4ResultFacetName> = {
  scenarioTemplate:'scenario_template', scenarioOrigin:'scenario_origin', scenarioStatus:'scenario_status',
  requestedProofMode:'requested_proof_mode', achievedProofMode:'achieved_proof_mode', proofCompletion:'proof_completion',
  proofVisibility:'proof_visibility', targetTeam:'target_team', contractOutcome:'contract_outcome',
  projectKind:'project_kind', projectValue:'project_value', projectState:'project_state',
  winningRootActionBand:'winning_root_action_band', exactStateAvailable:'exact_state_available', decisionKind:'decision_kind',
  researchBackend:'research_backend', qualificationAuthority:'qualification_authority',
  runtimeCompatibilityStatus:'runtime_compatibility_status', configurationBundle:'configuration_bundle',
  exactStateId:'exact_state_id', scenarioCategory:'scenario_category', variantCollection:'variant_collection',
};

type ScenarioExpressionDraft = PredicateDraft | ExpressionGroupDraft;

type ScenarioStudioProps = {
  configuration: ResearchScenarioConfigurationV1;
  recipeConfiguration: ResearchRecipeConfigurationCatalog;
  scenarioResearchV4?: ScenarioResearchV4CapabilityV1;
  selectedBundle: ConfigBundle;
  availableBundles: readonly ConfigBundle[];
  officialRulesRevisionId: string | null;
  rulesHash: string;
  busy: boolean;
  request: <T>(path: string, init?: RequestInit) => Promise<T>;
  createRecipeAndRun: (
    payload: JsonObject,
  ) => Promise<{recipe: ResearchRecipe; run: ResearchRun}>;
  setBusy: (value: boolean) => void;
  setMessage: (value: string) => void;
  refresh: () => Promise<void>;
  onOpenRun: (item: ResearchRun) => void;
  onOpenResult: (item: ResearchResult) => void;
  runs: ResearchRun[];
};

type ScenarioOperator = 'all' | 'any' | 'not' | 'count';

const operatorPollIntervalMs = 1_500;

function catalogValueLabel(catalog: ScenarioAuthoringCatalogV1, value: JsonValue) {
  if (typeof value === 'number' || (typeof value === 'string' && /^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value))) return number.format(Number(value));
  const text = String(value);
  const key = catalog.localizationKeys.values[text];
  if (!key) throw new Error(`scenario_catalog_value_label_missing:${text}`);
  return tCatalog(key);
}

function catalogFieldLabel(catalog: ScenarioAuthoringCatalogV1, path: string) {
  const key = String(catalog.fields[path]?.labelKey ?? '');
  if (!key) throw new Error(`scenario_catalog_field_label_missing:${path}`);
  return tCatalog(key);
}

function catalogTextDefault(catalog: ScenarioAuthoringCatalogV1, path: string) {
  const field = catalog.fields[path];
  if (!field) throw new Error(`scenario_catalog_field_missing:${path}`);
  if (typeof field.default === 'string') return field.default;
  if (typeof field.defaultKey === 'string') return tCatalog(field.defaultKey);
  throw new Error(`scenario_catalog_field_default_missing:${path}`);
}

function catalogExactArgumentLabel(catalog: ScenarioAuthoringCatalogV1, name: string) {
  const key = catalog.localizationKeys.exactArguments[name];
  if (!key) throw new Error(`scenario_catalog_exact_argument_label_missing:${name}`);
  return tCatalog(key);
}

function canonicalJson(value: JsonValue): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value)
    .sort(([left],[right])=>left.localeCompare(right))
    .map(([key,item])=>`${JSON.stringify(key)}:${canonicalJson(item)}`)
    .join(',')}}`;
  return JSON.stringify(value);
}

async function canonicalBrowserSha256(value: JsonValue): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(canonicalJson(value)),
  );
  return [...new Uint8Array(digest)]
    .map((item)=>item.toString(16).padStart(2,'0'))
    .join('');
}

const exactActionSections = [
  ['auctionActions', () => t('workflow.research_scenario_timeline_auction')],
  ['escalationActions', () => t('workflow.research_scenario_timeline_escalation')],
  ['projectDeclarations', () => t('workflow.research_scenario_timeline_projects')],
  ['projectAssertions', () => t('workflow.research_scenario_timeline_project_assertions')],
  ['playActions', () => t('workflow.research_scenario_timeline_play')],
] as const;

function catalogValues(catalog: ResearchFieldCatalog, path: string) {
  const values = catalog.fields[path]?.values;
  return Array.isArray(values) ? values.map(String) : [];
}

function authoredResourceFieldNames(catalog: ResearchFieldCatalog) {
  return Object.keys(catalog.fields)
    .filter((path)=>path.startsWith('resources.') && path !== 'resources.proofBudgetProfile')
    .map((path)=>path.slice('resources.'.length));
}

function integerSchemaBounds(
  schema: JsonObject,
  name: string,
  fallback?: {minimum: number; maximum: number; default: number},
  optional = false,
) {
  const properties = schema.properties && typeof schema.properties === 'object'
    && !Array.isArray(schema.properties) ? schema.properties as JsonObject : {};
  const field = properties[name] && typeof properties[name] === 'object'
    && !Array.isArray(properties[name]) ? properties[name] as JsonObject : {};
  if (
    typeof field.minimum !== 'number'
    || typeof field.maximum !== 'number'
    || (!optional && typeof field.default !== 'number')
  ) {
    if (fallback) return fallback;
    throw new Error(`scenario_selector_integer_schema_missing:${name}`);
  }
  return {minimum: field.minimum as number, maximum: field.maximum as number, default: (optional ? field.minimum : field.default) as number};
}

function stringSchemaDefault(schema: JsonObject, name: string, fallback?: string) {
  const properties = schema.properties && typeof schema.properties === 'object'
    && !Array.isArray(schema.properties) ? schema.properties as JsonObject : {};
  const field = properties[name] && typeof properties[name] === 'object'
    && !Array.isArray(properties[name]) ? properties[name] as JsonObject : {};
  if (typeof field.default === 'string') return field.default;
  if (fallback !== undefined) return fallback;
  throw new Error(`scenario_selector_string_schema_missing:${name}`);
}

function stringSchemaValues(schema: JsonObject, name: string, fallback: string[] = []) {
  const properties = schema.properties && typeof schema.properties === 'object'
    && !Array.isArray(schema.properties) ? schema.properties as JsonObject : {};
  const field = properties[name] && typeof properties[name] === 'object'
    && !Array.isArray(properties[name]) ? properties[name] as JsonObject : {};
  if (Array.isArray(field.enum) && field.enum.every((item)=>typeof item === 'string')) {
    return field.enum.map(String);
  }
  return fallback;
}

function selectorSchemaComplete(schema: JsonObject) {
  const required = Array.isArray(schema.required) ? schema.required.map(String) : [];
  const properties = schema.properties && typeof schema.properties === 'object'
    && !Array.isArray(schema.properties) ? schema.properties as JsonObject : {};
  return required.length > 0 && required.every((name) => {
    const field = properties[name];
    if (!field || typeof field !== 'object' || Array.isArray(field)) return false;
    const value = field as JsonObject;
    if (value.default === undefined) return false;
    if (value.type === 'integer') {
      return typeof value.minimum === 'number' && typeof value.maximum === 'number';
    }
    if (value.type === 'string') {
      return Array.isArray(value.enum) && value.enum.includes(value.default);
    }
    return false;
  });
}

function predicateEnumSchema(
  definitions: readonly ScenarioStudioPredicateDefinition[],
  name: string,
) {
  for (const definition of definitions) {
    const properties = jsonObject(definition.argumentSchema.properties);
    const field = jsonObject(properties[name]);
    if (Array.isArray(field.enum) && field.enum.length) return field;
  }
  return {};
}

function catalogChoice<T extends string>(
  value: JsonValue | undefined,
  choices: readonly T[],
  fallback: T,
  field: string,
) {
  if (value === undefined || value === null) return fallback;
  if (typeof value === 'string' && choices.includes(value as T)) return value as T;
  throw new Error(`scenario_recipe_choice_invalid:${field}`);
}

type ScenarioAuthoringOptions = {
  cards: ResearchScenarioConfigurationV1['cardOptions'];
  seats: string[];
  defaultDealer: string;
  ranks: string[];
  suits: string[];
  contracts: string[];
  auctionActions: string[];
  escalations: string[];
  escalationsByContract: Record<string, string[]>;
  tactics: string[];
  teams: string[];
  projectStates: string[];
  projects: ResearchScenarioConfigurationV1['projectOptions'];
  playActions: ResearchScenarioConfigurationV1['ruleOptions']['playActionOptions'];
};

function defaultExactTrace(options: ScenarioAuthoringOptions): JsonObject {
  return {
    schemaVersion: 1,
    deckOrder: options.cards.map((card)=>card.id),
    dealer: options.defaultDealer,
    roundNumber: 1,
    matchScore: [0, 0],
    auctionActions: [],
    escalationActions: [],
    projectActions: [],
    projectDeclarations: [],
    projectAssertions: [],
    playActions: [],
    stateAssertions: {},
  };
}

function optionLabel(
  catalog: ScenarioAuthoringCatalogV1,
  value: JsonValue,
  contextKind = '',
  schema?: JsonObject,
) {
  if (typeof value === 'boolean') return value ? '✓' : '×';
  const text = String(value);
  const schemaKeys = schema?.['x-enumMessageArKeys'];
  if (schemaKeys && typeof schemaKeys === 'object' && !Array.isArray(schemaKeys)) {
    const key = String((schemaKeys as JsonObject)[text] ?? '');
    if (key) return tCatalog(key);
    if (Array.isArray(schema?.enum)) {
      throw new Error(`scenario_catalog_enum_label_missing:${contextKind}:${text}`);
    }
  }
  return catalogValueLabel(catalog, value);
}

function resultFacetValueLabel(catalog: ScenarioAuthoringCatalogV1, value: string) {
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value)) return number.format(Number(value));
  if (value === '2_3') return `${number.format(2)}–${number.format(3)}`;
  if (value === '4_plus') return `${number.format(4)}+`;
  if (value === 'true') return tCatalog('research_scenario_exact_state_available');
  if (value === 'false') return tCatalog('research_scenario_exact_state_unavailable');
  if (value === 'ortools_cp_sat_v2') return tCatalog('research_scenario_backend_ortools_cp_sat_v2');
  if (value === 'ortools_cp_sat_v3') return tCatalog('research_scenario_backend_ortools_cp_sat_v3');
  if (value === 'cp_sat') return tCatalog('research_scenario_authority_title');
  if (['not_checked','passed','failed'].includes(value)) return tCatalog(`research_scenario_runtime_compatibility_${value}`);
  const key = catalog.localizationKeys.values[value];
  return key ? tCatalog(key) : value;
}

function argumentLabel(schema: JsonObject) {
  const key = String(schema['x-messageArKey'] ?? '');
  if (!key) throw new Error('scenario_catalog_argument_label_missing');
  return tCatalog(key);
}

function isCardSchema(schema: JsonObject) {
  return schema.type === 'integer' && schema.minimum === 0 && schema.maximum === 31;
}

function CardPicker({
  cards,
  selected,
  multiple,
  minimum = 0,
  maximum,
  onChange,
}: {
  cards: ResearchScenarioConfigurationV1['cardOptions'];
  selected: number[];
  multiple: boolean;
  minimum?: number;
  maximum?: number;
  onChange: (value: number[]) => void;
}) {
  return <div className="scenario-card-picker" role="group" aria-label={t('workflow.research_scenario_card_table')}>
    {cards.map((card) => {
      const active = selected.includes(card.id);
      return <button
        type="button"
        key={card.id}
        aria-pressed={active}
        aria-label={card.label}
        onClick={() => {
          if (!multiple) onChange([card.id]);
          else if (active && selected.length > minimum) onChange(selected.filter((item) => item !== card.id));
          else if (maximum === undefined || selected.length < maximum) onChange([...selected, card.id]);
        }}
      ><PreviewCardFace card={card}/><span>{active ? '✓' : '+'}</span></button>;
    })}
  </div>;
}

function parseExactTrace(raw: string): JsonObject | null {
  try {
    return parseJsonObject(raw);
  } catch {
    return null;
  }
}

function projectOptionKey(option: ResearchScenarioConfigurationV1['projectOptions'][number]) {
  return `${option.contract}:${option.kind}:${option.value}:${option.highRankId}`;
}

function exactTraceContract(value: JsonObject) {
  const actions = Array.isArray(value.auctionActions) ? value.auctionActions : [];
  for (const item of [...actions].reverse()) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    const action = String((item as JsonObject).action ?? '');
    if (['sun', 'hokum', 'ashkal'].includes(action)) return action;
  }
  return '';
}

function defaultExactAction(
  field: typeof exactActionSections[number][0],
  options: ScenarioAuthoringOptions,
): JsonObject {
  const project = options.projects[0];
  if (field === 'auctionActions') return {action: options.auctionActions[0] ?? ''};
  if (field === 'escalationActions') return {actor: options.seats[0] ?? 'north', action: options.escalations[0] ?? ''};
  if (field === 'playActions') {
    const action = options.playActions[0];
    return {actor: options.seats[0] ?? 'north', card: action?.card.code ?? '', ekka: action?.ekka ?? false};
  }
  if (!project) return {owner: options.seats[0] ?? 'north'};
  if (field === 'projectDeclarations') {
    return {
      owner: options.seats[0] ?? 'north',
      kind: project.kind,
      value: project.value,
      highRank: project.highRankId,
      declarationOrder: 1,
    };
  }
  return {
    owner: options.seats[0] ?? 'north',
    kind: project.kind,
    value: project.value,
    highRank: project.highRankId,
    state: 'eligible',
  };
}

function ExactActionEditor({
  field,
  action,
  options,
  catalog,
  onChange,
}: {
  field: typeof exactActionSections[number][0];
  action: JsonObject;
  options: ScenarioAuthoringOptions;
  catalog: ScenarioAuthoringCatalogV1;
  onChange: (value: JsonObject) => void;
}) {
  const set = (name: string, value: JsonValue | undefined) => {
    const next = {...action};
    if (value === undefined || value === '') delete next[name];
    else next[name] = value;
    onChange(next);
  };
  const seatSelect = (required: boolean) => <label className="field-label"><span>{catalogExactArgumentLabel(catalog, field === 'projectDeclarations' || field === 'projectAssertions' ? 'owner' : 'actor')}</span><select value={String(action[field === 'projectDeclarations' || field === 'projectAssertions' ? 'owner' : 'actor'] ?? '')} onChange={(event) => set(field === 'projectDeclarations' || field === 'projectAssertions' ? 'owner' : 'actor', event.target.value || undefined)}>{!required&&<option value="">—</option>}{options.seats.map((seat)=><option value={seat} key={seat}>{optionLabel(catalog,seat)}</option>)}</select></label>;
  if (field === 'auctionActions') {
    const selected = String(action.action ?? options.auctionActions[0] ?? '');
    return <div className="scenario-action-fields">{seatSelect(false)}<label className="field-label"><span>{catalogExactArgumentLabel(catalog,'action')}</span><select value={selected} onChange={(event)=>{const next: JsonObject={...action,action:event.target.value};if(event.target.value!=='hokum')delete next.trumpSuit;onChange(next);}}>{options.auctionActions.map((value)=><option value={value} key={value}>{optionLabel(catalog,value)}</option>)}</select></label>{selected === 'hokum'&&<label className="field-label"><span>{catalogExactArgumentLabel(catalog,'trumpSuit')}</span><select value={String(action.trumpSuit ?? '')} onChange={(event)=>set('trumpSuit',event.target.value||undefined)}><option value="">—</option>{options.suits.map((suit)=><option value={suit} key={suit}>{optionLabel(catalog,suit)}</option>)}</select></label>}</div>;
  }
  if (field === 'escalationActions') {
    const selected = String(action.action ?? options.escalations[0] ?? '');
    return <div className="scenario-action-fields">{seatSelect(true)}<label className="field-label"><span>{catalogExactArgumentLabel(catalog,'action')}</span><select value={selected} onChange={(event)=>{const next: JsonObject={...action,action:event.target.value};if(!['double','four'].includes(event.target.value))delete next.mode;onChange(next);}}>{options.escalations.map((value)=><option value={value} key={value}>{optionLabel(catalog,value)}</option>)}</select></label>{['double','four'].includes(selected)&&<label className="field-label"><span>{catalogExactArgumentLabel(catalog,'mode')}</span><select value={String(action.mode ?? '')} onChange={(event)=>set('mode',event.target.value||undefined)}><option value="">—</option><option value="open">{optionLabel(catalog,'open')}</option><option value="locked">{optionLabel(catalog,'locked')}</option></select></label>}</div>;
  }
  if (field === 'playActions') {
    const cardCode = typeof action.card === 'number' ? options.cards.find((card)=>card.id===action.card)?.code ?? '' : String(action.card ?? '');
    const selected = options.playActions.find((item)=>item.card.code===cardCode && item.ekka===(action.ekka===true))?.code ?? '';
    return <div className="scenario-action-fields">{seatSelect(false)}<label className="field-label"><span>{catalogExactArgumentLabel(catalog,'action')}</span><select value={selected} onChange={(event)=>{const next=options.playActions.find((item)=>item.code===event.target.value);if(next)onChange({...action,card:next.card.code,ekka:next.ekka});}}>{options.playActions.map((item)=><option value={item.code} key={item.actionId}>{item.card.label}{item.ekka?` · ${t('workflow.research_scenario_action_ekka')}`:''}</option>)}</select></label></div>;
  }
  const contract = String(action.contract ?? '');
  const applicable = options.projects.filter((item)=>!contract || item.contract === contract);
  const highRankId = typeof action.highRank === 'number' ? action.highRank : options.ranks.indexOf(String(action.highRank ?? ''));
  const current = applicable.find((item)=>item.kind===action.kind&&item.value===action.value&&item.highRankId===highRankId);
  const selectProject = (key: string) => {
    const selected = applicable.find((item)=>projectOptionKey(item)===key);
    if (!selected) return;
    onChange({
      ...action,
      kind: selected.kind,
      value: selected.value,
      highRank: selected.highRankId,
    });
  };
  return <div className="scenario-action-fields">{seatSelect(field === 'projectDeclarations')}<label className="field-label"><span>{t('workflow.research_label_project')}</span><select value={current ? projectOptionKey(current) : ''} onChange={(event)=>selectProject(event.target.value)}><option value="">—</option>{applicable.map((item)=><option key={projectOptionKey(item)} value={projectOptionKey(item)}>{optionLabel(catalog,item.contract)} · {optionLabel(catalog,item.kind)} · {optionLabel(catalog,String(item.value))} · {item.highRankId} · {optionLabel(catalog,item.highRank)}</option>)}</select></label>{field === 'projectDeclarations'?<label className="field-label"><span>{catalogExactArgumentLabel(catalog,'declarationOrder')}</span><input type="number" min={1} value={Number(action.declarationOrder ?? 1)} onChange={(event)=>set('declarationOrder',Math.max(1,Number(event.target.value)))}/></label>:<><label className="field-label"><span>{t('workflow.research_scenario_target_team')}</span><select value={String(action.team ?? '')} onChange={(event)=>set('team',event.target.value||undefined)}><option value="">—</option>{options.teams.map((value)=><option value={value} key={value}>{optionLabel(catalog,value)}</option>)}</select></label><label className="field-label"><span>{catalogExactArgumentLabel(catalog,'state')}</span><select value={String(action.state ?? options.projectStates[0])} onChange={(event)=>set('state',event.target.value)}>{options.projectStates.map((value)=><option value={value} key={value}>{optionLabel(catalog,value)}</option>)}</select></label></>}</div>;
}

function ProjectActionsEditor({value,catalog,onChange}: {
  value: JsonValue | undefined;
  catalog: ScenarioAuthoringCatalogV1;
  onChange: (value: JsonObject[]) => void;
}) {
  const actions = Array.isArray(value)
    ? value.filter((item): item is JsonObject => Boolean(item) && typeof item === 'object' && !Array.isArray(item))
    : [];
  if (!actions.length) return null;
  return <details className="scenario-project-actions" open data-field-path="scenario.origin.exactTracePrefix.projectActions">
    <summary>{t('workflow.research_scenario_timeline_projects')} <span>{number.format(actions.length)}</span></summary>
    <div>{actions.map((action,index)=><article key={index} data-field-path={`scenario.origin.exactTracePrefix.projectActions.${index}`}>
      <b>{number.format(index+1)}</b>
      <div className="scenario-action-fields">
        <label className="field-label"><span>{catalogExactArgumentLabel(catalog,'actor')}</span><output>{optionLabel(catalog,action.actor)}</output></label>
        <label className="field-label"><span>{catalogExactArgumentLabel(catalog,'action')}</span><output>{optionLabel(catalog,action.action)}</output></label>
        {action.projectId ? <label className="field-label"><span>{catalogExactArgumentLabel(catalog,'projectId')}</span><output dir="ltr">{String(action.projectId)}</output></label> : null}
      </div>
      <button type="button" className="quiet" onClick={()=>onChange(actions.filter((_,actionIndex)=>actionIndex!==index))}>{t('workflow.research_scenario_remove')}</button>
    </article>)}</div>
  </details>;
}

function ExactTraceEditor({
  raw,
  options,
  catalog,
  preview,
  previewBusy,
  onChange,
  onInvalid,
  onOpportunity,
  onPreviewRetry,
}: {
  raw: string;
  options: ScenarioAuthoringOptions;
  catalog: ScenarioAuthoringCatalogV1;
  preview: ExactTraceLegalActionsPreviewV1 | null;
  previewBusy: boolean;
  onChange: (value: string) => void;
  onInvalid: () => void;
  onOpportunity?: (action: ExactTraceLegalActionV1) => void;
  onPreviewRetry: () => void;
}) {
  const value = parseExactTrace(raw);
  const update = (change: JsonObject) => {
    if (!value) {
      onInvalid();
      return;
    }
    const next = {...value, ...change};
    delete next.sha256;
    onChange(JSON.stringify(next, null, 2));
  };
  const deck = Array.isArray(value?.deckOrder)
    ? value.deckOrder.map((item) => Number(item))
    : [];
  const updateDeck = (index: number, nextCard: number) => {
    if (deck.length !== 32) {
      onInvalid();
      return;
    }
    const previousIndex = deck.indexOf(nextCard);
    const next = [...deck];
    if (previousIndex >= 0) [next[index], next[previousIndex]] = [next[previousIndex], next[index]];
    else next[index] = nextCard;
    update({deckOrder: next});
  };
  const appendAction = (field: typeof exactActionSections[number][0], actionOptions = options) => {
    const current = Array.isArray(value?.[field]) ? value[field] as JsonValue[] : [];
    update({[field]: [...current, defaultExactAction(field, actionOptions)]});
  };
  const removeAction = (field: typeof exactActionSections[number][0], index: number) => {
    const current = Array.isArray(value?.[field]) ? value[field] as JsonValue[] : [];
    if (field === 'escalationActions' && onOpportunity) return onInvalid();
    update({[field]: current.filter((_, itemIndex) => itemIndex !== index)});
  };
  const updateAction = (
    field: typeof exactActionSections[number][0],
    index: number,
    action: JsonObject,
  ) => {
    const current = Array.isArray(value?.[field]) ? value[field] as JsonValue[] : [];
    update({[field]: current.map((item, itemIndex) => itemIndex === index ? action : item)});
  };
  const appendLegalAction = (action: ExactTraceLegalActionV1) => {
    if (!value) return onInvalid();
    if (action.stage === 'escalation' && onOpportunity) return onOpportunity(action);
    onChange(JSON.stringify(appendExactTraceLegalAction(value, action), null, 2));
  };
  const legalActionLabel = (action: ExactTraceLegalActionV1) => {
    if (action.stage === 'play') {
      return `${action.card?.label ?? action.code}${action.ekka ? ` · ${t('workflow.research_scenario_action_ekka')}` : ''}`;
    }
    if (action.stage === 'project') {
      return `${optionLabel(catalog,action.action ?? '')}${action.projectId ? ` · ${action.projectId}` : ''}`;
    }
    if (action.action === 'double') return t('workflow.research_escalation_double');
    if (action.action === 'triple') return t('workflow.research_escalation_triple');
    if (action.action === 'four') return t('workflow.research_escalation_four');
    if (action.action === 'gahwa') return t('workflow.research_escalation_gahwa');
    if (action.action && action.trumpSuit) {
      return `${optionLabel(catalog,action.action)} · ${optionLabel(catalog,action.trumpSuit)}`;
    }
    if (action.action) return optionLabel(catalog,action.action);
    return optionLabel(catalog,action.code);
  };

  if (!value) {
    return <section className="scenario-exact-builder invalid" role="alert"><p>{t('workflow.research_scenario_exact_trace_invalid')}</p></section>;
  }
  const score = Array.isArray(value.matchScore) ? value.matchScore : [0, 0];
  const stateAssertions = jsonObject(value.stateAssertions);
  const assertionProperties = jsonObject(catalog.exactTrace.stateAssertions.properties);
  const adjudicationEventsSchema = jsonObject(
    (catalog.exactTrace as unknown as JsonObject).adjudicationEvents,
  );
  const projectActionsSchema = jsonObject(
    (catalog.exactTrace as unknown as JsonObject).projectActions,
  );
  const updateAssertion = (name: string, assertion: string) => {
    const next = {...stateAssertions};
    if (assertion) next[name] = assertion;
    else delete next[name];
    update({stateAssertions: next});
  };
  return <section className="scenario-exact-builder">
    <header><div><span className="eyebrow">{t('workflow.research_scenario_exact_trace')}</span><h3>{t('workflow.research_scenario_exact_builder')}</h3><p>{t('workflow.research_scenario_exact_builder_help')}</p></div></header>
    <div className="scenario-grid"><label className="field-label"><span>{t('workflow.research_scenario_dealer')}</span><select value={String(value.dealer ?? options.defaultDealer)} onChange={(event)=>update({dealer:event.target.value})}>{options.seats.map((seat)=><option value={seat} key={seat}>{optionLabel(catalog,seat)}</option>)}</select></label><label className="field-label"><span>{t('workflow.research_scenario_round_number')}</span><input type="number" min={Number(catalog.exactTrace.roundNumber.minimum)} max={Number(catalog.exactTrace.roundNumber.maximum)} value={Number(value.roundNumber ?? catalog.exactTrace.roundNumber.default)} onChange={(event)=>update({roundNumber:Math.min(Number(catalog.exactTrace.roundNumber.maximum),Math.max(Number(catalog.exactTrace.roundNumber.minimum),Number(event.target.value)))})}/></label><label className="field-label"><span>{t('workflow.research_scenario_match_score_us')}</span><input type="number" min={Number(jsonObject(catalog.exactTrace.matchScore.items).minimum)} max={Number(jsonObject(catalog.exactTrace.matchScore.items).maximum)} value={Number(score[0] ?? 0)} onChange={(event)=>update({matchScore:[Math.min(Number(jsonObject(catalog.exactTrace.matchScore.items).maximum),Math.max(0,Number(event.target.value))),Number(score[1] ?? 0)]})}/></label><label className="field-label"><span>{t('workflow.research_scenario_match_score_them')}</span><input type="number" min={Number(jsonObject(catalog.exactTrace.matchScore.items).minimum)} max={Number(jsonObject(catalog.exactTrace.matchScore.items).maximum)} value={Number(score[1] ?? 0)} onChange={(event)=>update({matchScore:[Number(score[0] ?? 0),Math.min(Number(jsonObject(catalog.exactTrace.matchScore.items).maximum),Math.max(0,Number(event.target.value)))]})}/></label></div>
    <fieldset className="scenario-structured-field"><legend>{t('workflow.research_scenario_exact_state_derived')}</legend><div className="scenario-grid">{Object.entries(assertionProperties).map(([name,rawSchema])=>{const schema=jsonObject(rawSchema);const values=Array.isArray(schema.enum)?schema.enum:[];const path=`scenario.origin.exactTracePrefix.stateAssertions.${name}`;return <label className="field-label" data-field-path={path} key={name}><span>{argumentLabel(schema)}</span><select value={String(stateAssertions[name]??'')} onChange={(event)=>updateAssertion(name,event.target.value)}><option value="">—</option>{values.map((item)=><option value={String(item)} key={String(item)}>{catalogValueLabel(catalog,item)}</option>)}</select></label>;})}</div></fieldset>
    {Object.keys(adjudicationEventsSchema).length ? <AdjudicationEventsEditor schema={adjudicationEventsSchema} value={value.adjudicationEvents} options={options} catalog={catalog} onChange={(next)=>update({adjudicationEvents:next})}/> : null}
    <details className="scenario-deck-editor" open><summary>{t('workflow.research_scenario_deck_order')}</summary><p>{t('workflow.research_scenario_deck_order_help')}</p><div>{options.cards.map((_,index)=><label key={index}><span>{index + 1}</span><select aria-label={`${t('workflow.research_scenario_deck_position')} ${index + 1}`} value={deck[index] ?? -1} onChange={(event)=>updateDeck(index,Number(event.target.value))}>{options.cards.map((card)=><option key={card.id} value={card.id}>{card.label}</option>)}</select></label>)}</div></details>
    <section className="scenario-action-timeline">
      <h3>{t('workflow.research_scenario_action_timeline')}</h3>
      <p>{t('workflow.research_scenario_action_timeline_help')}</p>
      <section className="scenario-exact-next-actions" aria-live="polite" aria-busy={previewBusy}>
        {previewBusy ? <p>{t('workflow.research_status_running')}</p> : preview?.valid ? <>
          <p>{t('workflow.research_scenario_exact_state_phase')}: {preview.currentStage ? exactPreviewStageLabel(preview.currentStage) : '—'} · {t('workflow.research_scenario_next_actor')}: {preview.nextActor ? optionLabel(catalog,preview.nextActor) : '—'}</p>
          <div>{preview.legalActions.map((action)=><button type="button" className="secondary" data-exact-stage={action.stage} key={`${action.stage}:${action.code}`} onClick={()=>appendLegalAction(action)}>{legalActionLabel(action)}</button>)}</div>
        </> : <div role="alert"><p>{!preview ? t('workflow.research_scenario_request_failed')
          : preview.diagnostics.some((issue)=>issue.code==='exact_normalization_incomplete')
            ? t('workflow.research_scenario_support_incomplete') : t('workflow.research_scenario_exact_trace_invalid')}</p>
          <button type="button" className="secondary" onClick={onPreviewRetry}>{t('workflow.research_bundle_retry')}</button></div>}
      </section>
      {Object.keys(projectActionsSchema).length ? <ProjectActionsEditor value={value.projectActions} catalog={catalog} onChange={(next)=>update({projectActions:next})}/> : null}
      {exactActionSections.filter(([field])=>field==='auctionActions'?options.auctionActions.length>0:field==='escalationActions'?options.escalations.length>0:field==='projectDeclarations'||field==='projectAssertions'?options.projects.length>0:options.playActions.length>0).map(([field,labelKey])=>{
        const items=Array.isArray(value[field]) ? value[field] as JsonValue[] : [];
        const contract=exactTraceContract(value);
        const editorOptions={...options,projects:contract?options.projects.filter((item)=>item.contract===contract):options.projects,escalations:contract?options.escalationsByContract[contract]??[]:options.escalations};
        const assertions=field==='projectAssertions';
        return <details key={field}><summary>{labelKey()} <span>{items.length}</span></summary><div>{items.map((item,index)=><article key={`${field}-${index}`}><b>{index + 1}</b><fieldset disabled={!assertions}><ExactActionEditor field={field} action={item&&typeof item==='object'&&!Array.isArray(item)?item as JsonObject:{}} options={editorOptions} catalog={catalog} onChange={(action)=>updateAction(field,index,action)}/></fieldset>{field !== 'escalationActions' || !onOpportunity ? <button type="button" className="quiet" onClick={()=>removeAction(field,index)}>{t('workflow.research_scenario_remove')}</button> : null}</article>)}{assertions?<button type="button" className="secondary" onClick={()=>appendAction(field,editorOptions)}>{t('workflow.research_scenario_add_action')}</button>:null}</div></details>;
      })}
    </section>
    <details className="audit-details"><summary>{t('workflow.research_scenario_exact_trace_advanced')}</summary><p>{t('workflow.research_scenario_exact_trace_advanced_help')}</p><pre dir="ltr">{raw}</pre></details>
  </section>;
}

function schemaProperties(definition: ScenarioStudioPredicateDefinition) {
  const schema = definition.argumentSchema as {
    properties?: Record<string, JsonObject>;
    required?: string[];
  };
  return {
    properties: schema.properties ?? {},
    required: new Set(schema.required ?? []),
  };
}

function initialValue(schema: JsonObject): JsonValue {
  const values = schema.enum;
  if (Array.isArray(values) && values.length) return values[0] ?? null;
  if (schema.type === 'boolean') return false;
  if (schema.type === 'integer' || schema.type === 'number') {
    return typeof schema.minimum === 'number' ? schema.minimum : 0;
  }
  if (schema.type === 'array') {
    return Array.from(
      {length: typeof schema.minItems === 'number' ? schema.minItems : 0},
      () => initialValue(
        schema.items && typeof schema.items === 'object' && !Array.isArray(schema.items)
          ? schema.items as JsonObject
          : {},
      ),
    );
  }
  if (schema.type === 'object') {
    const properties = schema.properties && typeof schema.properties === 'object' && !Array.isArray(schema.properties)
      ? schema.properties as JsonObject
      : {};
    const required = Array.isArray(schema.required) ? schema.required.map(String) : [];
    return Object.fromEntries(required.map((name)=>[
      name,
      initialValue(properties[name] && typeof properties[name] === 'object' && !Array.isArray(properties[name]) ? properties[name] as JsonObject : {}),
    ]));
  }
  return '';
}

function initialArguments(definition: ScenarioStudioPredicateDefinition): JsonObject {
  const {properties, required} = schemaProperties(definition);
  return Object.fromEntries(
    [...required].map((name) => [name, initialValue(properties[name] ?? {})]),
  );
}

function predicateLabel(definition: ScenarioStudioPredicateDefinition) {
  return tCatalog(definition.messageArKey);
}

function predicateHelp(
  catalog: ScenarioAuthoringCatalogV1,
  definition: ScenarioStudioPredicateDefinition,
) {
  const key = catalog.localizationKeys.predicateHelp[definition.kind];
  return key ? tCatalog(key) : null;
}

function PredicateCatalogMetadata({definition,catalog}: {
  definition: ScenarioStudioPredicateDefinition;
  catalog: ScenarioAuthoringCatalogV1;
}) {
  return <details className="scenario-predicate-metadata">
    <summary>{t('workflow.research_scenario_predicate_metadata')}</summary>
    <dl>
      <div><dt>{t('workflow.research_scenario_predicate_cost')}</dt><dd>{catalogValueLabel(catalog,definition.costClass)}</dd></div>
      <div><dt>{t('workflow.research_scenario_predicate_hidden_information')}</dt><dd>{catalogValueLabel(catalog,definition.cpSatSupport?.actorInformationSet??definition.hiddenInformation)}</dd></div>
    </dl>
    <section><h4>{t('workflow.research_scenario_predicate_examples')}</h4><pre dir="ltr">{JSON.stringify(definition.examples,null,2)}</pre></section>
    <section><h4>{t('workflow.research_scenario_predicate_counterexamples')}</h4><pre dir="ltr">{JSON.stringify(definition.counterexamples,null,2)}</pre></section>
    <section><h4>{t('workflow.research_scenario_predicate_documentation')}</h4><ul>{definition.documentation.map((item)=><li key={item}><code dir="ltr">{item}</code></li>)}</ul></section>
  </details>;
}

const scenarioCapabilityLabels: Readonly<Record<string, () => string>> = {
  cp_sat_exact: () => t('workflow.research_scenario_capability_cp_sat_exact'),
  incremental_actor_safe: () => t('workflow.research_scenario_capability_incremental_actor_safe'),
  guided_search: () => t('workflow.research_scenario_capability_guided_search'),
  postcondition_only: () => t('workflow.research_scenario_capability_postcondition_only'),
  exact_input_only: () => t('workflow.research_scenario_capability_exact_input_only'),
  unsupported: () => t('workflow.research_scenario_capability_unsupported'),
};

function capabilityLabel(capability: string) {
  const label = scenarioCapabilityLabels[capability];
  if (!label) throw new Error(`scenario_capability_label_missing:${capability}`);
  return label();
}

const scenarioStageLabels: Readonly<Record<string, () => string>> = {
  deal: () => t('workflow.research_scenario_stage_deal'),
  auction: () => t('workflow.research_scenario_stage_auction'),
  escalation: () => t('workflow.research_scenario_stage_escalation'),
  projects: () => t('workflow.research_scenario_stage_projects'),
  playPrefix: () => t('workflow.research_scenario_stage_playPrefix'),
  state: () => t('workflow.research_scenario_stage_state'),
  outcome: () => t('workflow.research_scenario_stage_outcome'),
  tactics: () => t('workflow.research_scenario_stage_tactics'),
  proof: () => t('workflow.research_scenario_stage_proof'),
};

const scenarioDiagnosticSeverityLabels: Readonly<
  Record<ScenarioDiagnosticV1['severity'], () => string>
> = {
  error: () => t('workflow.research_scenario_diagnostic_error'),
  warning: () => t('workflow.research_scenario_diagnostic_warning'),
  info: () => t('workflow.research_scenario_diagnostic_info'),
};

function stageLabel(stage: string) {
  const label = scenarioStageLabels[stage];
  if (!label) throw new Error(`scenario_stage_label_missing:${stage}`);
  return label();
}

function exactPreviewStageLabel(stage: NonNullable<ExactTraceLegalActionsPreviewV1['currentStage']>) {
  const labels = {
    auction: () => t('workflow.research_scenario_timeline_auction'),
    escalation: () => t('workflow.research_scenario_timeline_escalation'),
    project: () => t('workflow.research_scenario_timeline_projects'),
    play: () => t('workflow.research_scenario_timeline_play'),
    terminal: () => t('workflow.research_status_completed'),
  } as const;
  return labels[stage]();
}

function predicateWire(draft: PredicateDraft): JsonObject {
  return {
    predicateId: draft.predicateId,
    kind: draft.kind,
    version: draft.version,
    hardness: draft.hardness,
    arguments: draft.arguments,
  };
}

function isPredicateDraft(value: ScenarioExpressionDraft): value is PredicateDraft {
  return 'predicateId' in value;
}

export function scenarioExpressionWire(value: ScenarioExpressionDraft): JsonObject {
  if (isPredicateDraft(value)) return predicateWire(value);
  const result: JsonObject = {
    op: value.op,
    children: value.children.map(scenarioExpressionWire),
  };
  if (value.op === 'count') {
    result.minimum = Math.max(0, Math.min(value.minimum, value.children.length));
    result.maximum = Math.max(
      Number(result.minimum),
      Math.min(value.maximum, value.children.length),
    );
  }
  return result;
}

function expressionPredicates(value: ScenarioExpressionDraft): PredicateDraft[] {
  return isPredicateDraft(value)
    ? [value]
    : value.children.flatMap(expressionPredicates);
}

function expressionGroups(value: ExpressionGroupDraft): ExpressionGroupDraft[] {
  return [
    value,
    ...value.children.flatMap((child) =>
      isPredicateDraft(child) ? [] : expressionGroups(child)),
  ];
}

function expressionNodeCount(value: ScenarioExpressionDraft): number {
  return isPredicateDraft(value)
    ? 1
    : 1 + value.children.reduce((total,child)=>total+expressionNodeCount(child),0);
}

function expressionSerializedBytes(value: ScenarioExpressionDraft): number {
  return new TextEncoder().encode(JSON.stringify(scenarioExpressionWire(value))).byteLength;
}

function expressionWithinLimits(
  value: ExpressionGroupDraft,
  limits: ScenarioAuthoringCatalogV1['expressionLimits'],
): boolean {
  const visit = (node: ScenarioExpressionDraft, depth: number): boolean => {
    if (depth > limits.maximumDepth) return false;
    if (isPredicateDraft(node)) return true;
    if (node.children.length > limits.maximumChildren) return false;
    if (
      node.op === 'count'
      && (node.minimum > node.maximum
        || node.maximum - node.minimum > limits.maximumCountRange)
    ) return false;
    return node.children.every((child)=>visit(child,depth+1));
  };
  return expressionNodeCount(value) <= limits.maximumNodes
    && expressionSerializedBytes(value) <= limits.maximumSerializedBytes
    && visit(value,0);
}

function expressionGroupDepth(value: ExpressionGroupDraft, nodeId: string, depth = 0): number {
  if (value.nodeId === nodeId) return depth;
  for (const child of value.children) {
    if (isPredicateDraft(child)) continue;
    const found = expressionGroupDepth(child,nodeId,depth+1);
    if (found >= 0) return found;
  }
  return -1;
}

function updateExpression(
  value: ScenarioExpressionDraft,
  nodeId: string,
  update: (current: ScenarioExpressionDraft) => ScenarioExpressionDraft,
): ScenarioExpressionDraft {
  if ((isPredicateDraft(value) ? value.predicateId : value.nodeId) === nodeId) {
    return update(value);
  }
  if (isPredicateDraft(value)) return value;
  return {
    ...value,
    children: value.children.map((child) => updateExpression(child, nodeId, update)),
  };
}

function removeExpressionChild(
  value: ExpressionGroupDraft,
  nodeId: string,
): ExpressionGroupDraft {
  return {
    ...value,
    children: value.children
      .filter((child) =>
        (isPredicateDraft(child) ? child.predicateId : child.nodeId) !== nodeId)
      .map((child) =>
        isPredicateDraft(child) ? child : removeExpressionChild(child, nodeId)),
  };
}

function expressionCanUseNot(
  value: ExpressionGroupDraft,
  definitions: Map<string, ScenarioStudioPredicateDefinition>,
) {
  if (value.children.length !== 1 || !isPredicateDraft(value.children[0])) {
    return false;
  }
  return definitions.get(value.children[0].kind)?.safeComplement === true;
}

function normalizeExpression(
  value: ExpressionGroupDraft,
  definitions: Map<string, ScenarioStudioPredicateDefinition>,
): ExpressionGroupDraft {
  const children = value.children.map((child) =>
    isPredicateDraft(child) ? child : normalizeExpression(child, definitions));
  const candidate = {...value, children};
  const op = candidate.op === 'not' && !expressionCanUseNot(candidate, definitions)
    ? 'all'
    : candidate.op;
  return {
    ...candidate,
    op,
    minimum: Math.max(0, Math.min(candidate.minimum, children.length)),
    maximum: Math.max(
      Math.max(0, Math.min(candidate.minimum, children.length)),
      Math.min(candidate.maximum, children.length),
    ),
  };
}

function hydrateExpression(
  raw: JsonValue,
  rootId: string,
  nextGroup: {value: number},
): ExpressionGroupDraft {
  const source = raw && typeof raw === 'object' && !Array.isArray(raw)
    ? raw as JsonObject
    : {};
  const children = Array.isArray(source.children)
    ? source.children
    : typeof source.predicateId === 'string' && typeof source.kind === 'string'
      ? [source]
      : [];
  return {
    nodeId: rootId,
    op: ['all', 'any', 'not', 'count'].includes(String(source.op))
      ? source.op as ScenarioOperator
      : 'all',
    minimum: typeof source.minimum === 'number' ? source.minimum : 1,
    maximum: typeof source.maximum === 'number' ? source.maximum : 1,
    children: children.flatMap((child): ScenarioExpressionDraft[] => {
      if (!child || typeof child !== 'object' || Array.isArray(child)) return [];
      const item = child as JsonObject;
      if (typeof item.predicateId === 'string' && typeof item.kind === 'string') {
        return [{
          predicateId: item.predicateId,
          kind: item.kind,
          version: 1,
          hardness: ['hard', 'prefer', 'report'].includes(String(item.hardness))
            ? item.hardness as PredicateDraft['hardness']
            : 'hard',
          arguments: item.arguments && typeof item.arguments === 'object'
            && !Array.isArray(item.arguments)
            ? item.arguments as JsonObject
            : {},
        }];
      }
      const nodeId = `EXPR-${String(nextGroup.value++).padStart(4, '0')}`;
      return [hydrateExpression(item, nodeId, nextGroup)];
    }),
  };
}

function parseJsonObject(raw: string): JsonObject {
  const value = JSON.parse(raw) as JsonValue;
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    throw new Error('json_object_required');
  }
  return value as JsonObject;
}

function ArgumentField({
  name,
  schema,
  required,
  value,
  contextKind,
  options,
  catalog,
  fieldPath,
  errors = [],
  onChange,
}: {
  name: string;
  schema: JsonObject;
  required: boolean;
  value: JsonValue | undefined;
  contextKind: string;
  options: ScenarioAuthoringOptions;
  catalog: ScenarioAuthoringCatalogV1;
  fieldPath?: string;
  errors?: readonly CardSelectorError[];
  onChange: (value: JsonValue | undefined) => void;
}) {
  const enumValues = Array.isArray(schema.enum) ? schema.enum : [];
  const invalidChoiceId = useId();
  const label = `${argumentLabel(schema)}${required ? ' *' : ''}`;
  if (isCardSelectorSchema(schema)) {
    const messageKey = typeof schema['x-messageArKey'] === 'string'
      ? schema['x-messageArKey']
      : '';
    const helpKey = typeof schema['x-helpArKey'] === 'string'
      ? schema['x-helpArKey']
      : messageKey ? `${messageKey}_help` : '';
    return <CardSelectorV1Control
      schema={schema}
      value={value}
      cards={options.cards}
      label={label}
      help={helpKey ? tCatalog(helpKey) : undefined}
      fieldPath={fieldPath}
      labelForSchema={argumentLabel}
      labelForOption={(item, itemSchema)=>optionLabel(catalog,item, contextKind, itemSchema)}
      errors={errors}
      onChange={onChange}
    />;
  }
  if (Array.isArray(schema.enum)) {
    const invalidChoice = value !== undefined && !enumValues.includes(value);
    return <label className="field-label"><span>{label}</span><select required={required || invalidChoice} aria-invalid={invalidChoice || undefined} aria-describedby={invalidChoice ? invalidChoiceId : undefined} value={invalidChoice ? '' : String(value ?? '')} onChange={(event)=>onChange(event.target.value === '' ? undefined : enumValues.find((item)=>String(item) === event.target.value))}><option value="">—</option>{enumValues.map((item)=><option key={String(item)} value={String(item)}>{optionLabel(catalog,item,contextKind,schema)}</option>)}</select>{invalidChoice ? <small id={invalidChoiceId}>{t('workflow.research_scenario_capability_unsupported')}</small> : null}</label>;
  }
  if (schema.type === 'boolean') {
    return <label className="field-label"><span>{label}</span><select value={value === true ? 'true' : value === false ? 'false' : ''} onChange={(event)=>onChange(event.target.value === '' ? undefined : event.target.value === 'true')}><option value="">—</option><option value="true">✓</option><option value="false">×</option></select></label>;
  }
  if (isCardSchema(schema)) {
    const selected = typeof value === 'number' ? [value] : [];
    return <fieldset className="scenario-structured-field scenario-card-field"><legend>{label}</legend><CardPicker cards={options.cards} selected={selected} multiple={false} onChange={(items)=>onChange(items[0])}/></fieldset>;
  }
  if (schema.type === 'integer' || schema.type === 'number') {
    const ordinal = name === 'trickIndex' || name === 'position' ? name : null;
    const minimum = typeof schema.minimum === 'number'
      ? ordinal ? scenarioOrdinalDisplayValue(ordinal,schema.minimum) : schema.minimum
      : undefined;
    const maximum = typeof schema.maximum === 'number'
      ? ordinal ? scenarioOrdinalDisplayValue(ordinal,schema.maximum) : schema.maximum
      : undefined;
    const displayValue = typeof value === 'number'
      ? ordinal ? scenarioOrdinalDisplayValue(ordinal,value) : value
      : '';
    return <label className="field-label"><span>{label}</span><input required={required} type="number" min={minimum} max={maximum} step={schema.type === 'integer' ? 1 : 'any'} value={displayValue} onChange={(event)=>onChange(event.target.value === '' ? undefined : ordinal ? scenarioOrdinalWireValue(ordinal,Number(event.target.value)) : Number(event.target.value))}/></label>;
  }
  if (schema.type === 'array') {
    const items = Array.isArray(value) ? value : [];
    const itemSchema = schema.items && typeof schema.items === 'object' && !Array.isArray(schema.items)
      ? schema.items as JsonObject
      : {};
    const itemEnums = Array.isArray(itemSchema.enum) ? itemSchema.enum : [];
    if (itemEnums.length) {
      const minimum = typeof schema.minItems === 'number' ? schema.minItems : 0;
      const maximum = typeof schema.maxItems === 'number' ? schema.maxItems : Number.POSITIVE_INFINITY;
      const enumLabelSchema = {...itemSchema,
        'x-enumMessageArKeys': schema['x-enumMessageArKeys'] ?? itemSchema['x-enumMessageArKeys'],
      };
      return <fieldset className="scenario-structured-field"><legend>{label}</legend><div className="scenario-choice-list">{itemEnums.map((item)=>{const active=items.includes(item);return <label key={String(item)}><input type="checkbox" value={String(item)} checked={active} disabled={active?items.length<=minimum:items.length>=maximum} onChange={(event)=>onChange(event.target.checked?[...items,item]:items.filter((value)=>value!==item))}/><span>{optionLabel(catalog,item,contextKind,enumLabelSchema)}</span></label>;})}</div></fieldset>;
    }
    if (isCardSchema(itemSchema)) {
      return <fieldset className="scenario-structured-field scenario-card-field"><legend>{label}</legend><CardPicker cards={options.cards} selected={items.filter((item): item is number=>typeof item==='number')} multiple minimum={typeof schema.minItems==='number'?schema.minItems:0} maximum={typeof schema.maxItems==='number'?schema.maxItems:undefined} onChange={onChange}/></fieldset>;
    }
    if (name === 'actions' && (contextKind.startsWith('auction.') || contextKind.startsWith('escalation.') || contextKind === 'play.action_prefix' || contextKind === 'state.legal_play_actions_include')) {
      const field = contextKind.startsWith('auction.') ? 'auctionActions' : contextKind.startsWith('escalation.') ? 'escalationActions' : 'playActions';
      const unique = schema.uniqueItems === true;
      const append = defaultExactAction(field,options);
      const duplicate = unique && items.some((item)=>JSON.stringify(item)===JSON.stringify(append));
      return <fieldset className="scenario-structured-field"><legend>{label}</legend><div className="scenario-structured-list">{items.map((item,index)=><article key={index}><ExactActionEditor field={field} action={item&&typeof item==='object'&&!Array.isArray(item)?item as JsonObject:{}} options={options} catalog={catalog} onChange={(next)=>{if(unique&&items.some((value,itemIndex)=>itemIndex!==index&&JSON.stringify(value)===JSON.stringify(next)))return;onChange(items.map((value,itemIndex)=>itemIndex===index?next:value));}}/><button type="button" className="quiet" disabled={typeof schema.minItems==='number'&&items.length<=schema.minItems} onClick={()=>onChange(items.filter((_,itemIndex)=>itemIndex!==index))}>{t('workflow.research_scenario_remove')}</button></article>)}<button type="button" className="secondary" disabled={(typeof schema.maxItems==='number'&&items.length>=schema.maxItems)||duplicate} onClick={()=>onChange([...items,append])}>{t('workflow.research_scenario_add_action')}</button></div></fieldset>;
    }
    const unique = schema.uniqueItems === true;
    const append = initialValue(itemSchema);
    const duplicate = unique && items.some((item)=>JSON.stringify(item)===JSON.stringify(append));
    return <fieldset className="scenario-structured-field"><legend>{label}</legend><div className="scenario-structured-list">{items.map((item,index)=><article key={index}><ArgumentField name={String(index+1)} schema={itemSchema} required value={item} contextKind={contextKind} options={options} catalog={catalog} onChange={(next)=>{const normalized=next??initialValue(itemSchema);if(unique&&items.some((current,itemIndex)=>itemIndex!==index&&JSON.stringify(current)===JSON.stringify(normalized)))return;onChange(items.map((current,itemIndex)=>itemIndex===index?normalized:current));}}/><button type="button" className="quiet" disabled={typeof schema.minItems==='number'&&items.length<=schema.minItems} onClick={()=>onChange(items.filter((_,itemIndex)=>itemIndex!==index))}>{t('workflow.research_scenario_remove')}</button></article>)}<button type="button" className="secondary" disabled={(typeof schema.maxItems==='number'&&items.length>=schema.maxItems)||duplicate} onClick={()=>onChange([...items,append])}>{t('workflow.research_scenario_add')}</button></div></fieldset>;
  }
  if (schema.type === 'object') {
    const current = value && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {};
    const properties = schema.properties && typeof schema.properties === 'object' && !Array.isArray(schema.properties) ? schema.properties as JsonObject : {};
    const nestedRequired = new Set(Array.isArray(schema.required) ? schema.required.map(String) : []);
    return <fieldset className="scenario-structured-field"><legend>{label}</legend><div className="scenario-arguments">{Object.entries(properties).map(([childName,childSchema])=><ArgumentField key={childName} name={childName} schema={childSchema&&typeof childSchema==='object'&&!Array.isArray(childSchema)?childSchema as JsonObject:{}} required={nestedRequired.has(childName)} value={current[childName]} contextKind={contextKind} options={options} catalog={catalog} fieldPath={fieldPath ? `${fieldPath}.${childName}` : undefined} errors={errors} onChange={(next)=>{const updated={...current};if(next===undefined)delete updated[childName];else updated[childName]=next;onChange(updated);}}/>)}</div></fieldset>;
  }
  const pattern = typeof schema.pattern === 'string' ? schema.pattern : undefined;
  return <label className="field-label"><span>{label}</span><input required={required} pattern={pattern} minLength={typeof schema.minLength==='number'?schema.minLength:undefined} maxLength={typeof schema.maxLength==='number'?schema.maxLength:undefined} inputMode={pattern&&/\\d|0-9/.test(pattern)?'numeric':undefined} value={typeof value === 'string' ? value : ''} onChange={(event)=>onChange(event.target.value || undefined)}/></label>;
}

function AdjudicationEventsEditor({schema,value,options,catalog,onChange}: {
  schema: JsonObject;
  value: JsonValue | undefined;
  options: ScenarioAuthoringOptions;
  catalog: ScenarioAuthoringCatalogV1;
  onChange: (value: JsonObject[]) => void;
}) {
  const rootPath = 'scenario.origin.exactTracePrefix.adjudicationEvents';
  const itemSchema = jsonObject(schema.items);
  const properties = jsonObject(itemSchema.properties);
  const required = new Set(Array.isArray(itemSchema.required) ? itemSchema.required.map(String) : []);
  const selectedRemedySchema = jsonObject(properties.selectedRemedy);
  const remediesByKind = jsonObject(selectedRemedySchema['x-remediesByKind']);
  const events = Array.isArray(value)
    ? value.filter((item): item is JsonObject => Boolean(item) && typeof item === 'object' && !Array.isArray(item))
    : [];
  const remediesFor = (kind: JsonValue | undefined) => {
    const remedies = remediesByKind[String(kind??'')];
    return Array.isArray(remedies) ? remedies.filter((item): item is string=>typeof item==='string') : [];
  };
  const updateEvent = (index: number, next: JsonObject) => onChange(
    events.map((event,eventIndex)=>eventIndex===index?next:event),
  );
  const append = () => {
    const base = jsonObject(initialValue(itemSchema));
    const defaults = Object.fromEntries(Object.entries(properties).flatMap(([name,raw])=>{
      const field = jsonObject(raw);
      return Object.prototype.hasOwnProperty.call(field,'default') ? [[name,field.default]] : [];
    }));
    onChange([...events,{...base,...defaults}]);
  };
  return <fieldset className="scenario-structured-field scenario-adjudication-events" data-field-path={rootPath}>
    <legend>{argumentLabel(schema)}</legend>
    <div className="scenario-structured-list">
      {events.map((event,index)=><article key={index}>
        <fieldset className="scenario-structured-field"><legend>{argumentLabel(schema)} · {number.format(index+1)}</legend><div className="scenario-arguments">
          {Object.entries(properties).filter(([name,raw])=>name!=='schemaVersion'&&(name==='selectedRemedy'||name==='rulingEvidenceRef'||schemaHasVisualControl(jsonObject(raw),'adjudication.exact_trace_events',name))).map(([name,raw])=>{
            const fieldSchema = jsonObject(raw);
            const fieldPath = `${rootPath}.${index}.${name}`;
            if (name==='selectedRemedy') {
              const allowed = remediesFor(event.kind);
              return <label className="field-label" data-field-path={fieldPath} key={name}><span>{argumentLabel(fieldSchema)}</span><select value={typeof event.selectedRemedy==='string'?event.selectedRemedy:''} onChange={(input)=>{const remedy=input.target.value||null;updateEvent(index,{...event,selectedRemedy:remedy,...(!remedy?{rulingEvidenceRef:null}:{})});}}><option value="">—</option>{allowed.map((remedy)=><option value={remedy} key={remedy}>{optionLabel(catalog,remedy,'adjudication.exact_trace_events',fieldSchema)}</option>)}</select></label>;
            }
            if (name==='rulingEvidenceRef') {
              return <label className="field-label" data-field-path={fieldPath} key={name}><span>{argumentLabel(fieldSchema)}</span><input disabled={typeof event.selectedRemedy!=='string'} pattern={typeof fieldSchema.pattern==='string'?fieldSchema.pattern:undefined} minLength={typeof fieldSchema.minLength==='number'?fieldSchema.minLength:undefined} maxLength={typeof fieldSchema.maxLength==='number'?fieldSchema.maxLength:undefined} value={typeof event.rulingEvidenceRef==='string'?event.rulingEvidenceRef:''} onChange={(input)=>updateEvent(index,{...event,rulingEvidenceRef:input.target.value||null})}/></label>;
            }
            return <div data-field-path={fieldPath} key={name}><ArgumentField name={name} schema={fieldSchema} required={required.has(name)} value={event[name]} contextKind="adjudication.exact_trace_events" options={options} catalog={catalog} fieldPath={fieldPath} onChange={(next)=>{const updated={...event};if(next===undefined)delete updated[name];else updated[name]=next;if(name==='kind'){const allowed=remediesFor(next);if(typeof updated.selectedRemedy!=='string'||!allowed.includes(updated.selectedRemedy)){updated.selectedRemedy=null;updated.rulingEvidenceRef=null;}}updateEvent(index,updated);}}/></div>;
          })}
        </div></fieldset>
        <button type="button" className="quiet" onClick={()=>onChange(events.filter((_,eventIndex)=>eventIndex!==index))}>{t('workflow.research_scenario_remove')}</button>
      </article>)}
      <button type="button" className="secondary" disabled={typeof schema.maxItems==='number'&&events.length>=schema.maxItems} onClick={append}>{t('workflow.research_scenario_add')}</button>
    </div>
  </fieldset>;
}

function schemaHasVisualControl(schema: JsonObject, contextKind: string, name = ''): boolean {
  if (isCardSelectorSchema(schema)) return true;
  const type = String(schema.type ?? '');
  if (['string','number','integer','boolean'].includes(type)) return true;
  if (type === 'object') {
    const properties = schema.properties && typeof schema.properties === 'object' && !Array.isArray(schema.properties) ? schema.properties as JsonObject : {};
    return Object.keys(properties).length > 0 && Object.entries(properties).every(([childName,child])=>schemaHasVisualControl(child&&typeof child==='object'&&!Array.isArray(child)?child as JsonObject:{},contextKind,childName));
  }
  if (type === 'array') {
    if (name === 'actions' && ['auction.action_sequence','escalation.action_sequence'].includes(contextKind)) return true;
    const item = schema.items && typeof schema.items === 'object' && !Array.isArray(schema.items) ? schema.items as JsonObject : {};
    return schemaHasVisualControl(item,contextKind,name);
  }
  return false;
}

function ExpressionEditor({
  value,
  root,
  activeTarget,
  catalog,
  depth = 0,
  totalNodes,
  definitions,
  onActivate,
  onChange,
  onAddGroup,
  onRemove,
  renderPredicate,
}: {
  value: ExpressionGroupDraft;
  root: boolean;
  activeTarget: string;
  catalog: ScenarioAuthoringCatalogV1;
  depth?: number;
  totalNodes: number;
  definitions: Map<string, ScenarioStudioPredicateDefinition>;
  onActivate: (nodeId: string) => void;
  onChange: (value: ExpressionGroupDraft) => void;
  onAddGroup: (nodeId: string) => void;
  onRemove: (nodeId: string) => void;
  renderPredicate: (draft: PredicateDraft) => ReactNode;
}) {
  const canUseNot = expressionCanUseNot(value, definitions);
  const atCapacity = (value.op === 'not' && value.children.length === 1)
    || value.children.length >= catalog.expressionLimits.maximumChildren
    || totalNodes >= catalog.expressionLimits.maximumNodes;
  const groupAtDepthLimit = depth >= catalog.expressionLimits.maximumDepth;
  return <fieldset
    className={`scenario-expression-node${root ? ' root' : ''}`}
    id={`scenario-${value.nodeId}`}
    data-field-path={root ? undefined : value.nodeId}
  >
    <legend>{root ? t('workflow.research_scenario_expression_root') : t('workflow.research_scenario_expression_group')}</legend>
    <div className="scenario-expression-toolbar">
      <label className="field-label">
        <span>{t('workflow.research_scenario_expression_operator')}</span>
        <select
          value={value.op}
          onChange={(event) => onChange({...value, op: event.target.value as ScenarioOperator})}
        >
          {catalog.logicalOperators
            .filter((operator) => operator !== 'not' || canUseNot || value.op === 'not')
            .map((operator) => <option value={operator} key={operator}>{catalogValueLabel(catalog,operator)}</option>)}
        </select>
      </label>
      {value.op === 'count' ? <>
        <label className="field-label">
          <span>{t('workflow.research_scenario_count_minimum')}</span>
          <input type="number" min={0} max={value.children.length} step={1} value={value.minimum} onChange={(event)=>{const minimum=Math.min(value.children.length,Math.max(0,Number(event.target.value)));onChange({...value,minimum,maximum:Math.min(value.children.length,Math.max(minimum,value.maximum))});}}/>
        </label>
        <label className="field-label">
          <span>{t('workflow.research_scenario_count_maximum')}</span>
          <input type="number" min={value.minimum} max={Math.min(value.children.length,value.minimum+catalog.expressionLimits.maximumCountRange)} step={1} value={value.maximum} onChange={(event)=>onChange({...value,maximum:Math.min(value.children.length,value.minimum+catalog.expressionLimits.maximumCountRange,Math.max(value.minimum,Number(event.target.value)))})}/>
        </label>
      </> : null}
      <div className="scenario-expression-buttons">
        <button type="button" className="secondary" aria-pressed={activeTarget === value.nodeId} disabled={atCapacity} onClick={()=>onActivate(value.nodeId)}>{t('workflow.research_scenario_expression_target')}</button>
        <button type="button" className="secondary" disabled={atCapacity||groupAtDepthLimit} onClick={()=>onAddGroup(value.nodeId)}>{t('workflow.research_scenario_expression_add_group')}</button>
        {!root ? <button type="button" className="quiet" onClick={()=>onRemove(value.nodeId)}>{t('workflow.research_scenario_remove')}</button> : null}
      </div>
    </div>
    {value.op === 'not' ? <p className="scenario-expression-note">{t('workflow.research_scenario_not_safe_help')}</p> : null}
    <div className="scenario-expression-children">
      {value.children.map((child) => isPredicateDraft(child)
        ? renderPredicate(child)
        : <ExpressionEditor
            key={child.nodeId}
            value={child}
            root={false}
            activeTarget={activeTarget}
            catalog={catalog}
            depth={depth+1}
            totalNodes={totalNodes}
            definitions={definitions}
            onActivate={onActivate}
            onChange={(next)=>onChange(updateExpression(value,child.nodeId,()=>next) as ExpressionGroupDraft)}
            onAddGroup={onAddGroup}
            onRemove={onRemove}
            renderPredicate={renderPredicate}
          />)}
      {!value.children.length ? <p className="scenario-empty">{root
        ? value.nodeId === 'EXPR-CONSTRAINTS'
          ? t('workflow.research_scenario_constraints_empty')
          : t('workflow.research_scenario_outcome_empty')
        : t('workflow.research_scenario_expression_empty')}</p> : null}
    </div>
  </fieldset>;
}

function CatalogScalarField({
  catalog,
  authoringCatalog,
  path,
  label,
  value,
  maximum,
  disabled = false,
  onChange,
}: {
  catalog: ResearchFieldCatalog;
  authoringCatalog: ScenarioAuthoringCatalogV1;
  path: string;
  label: string;
  value: JsonValue;
  maximum?: number;
  disabled?: boolean;
  onChange: (value: JsonValue) => void;
}) {
  const field = catalog.fields[path];
  if (!field) throw new Error(`scenario_catalog_field_missing:${path}`);
  const values = Array.isArray(field.values) ? field.values : [];
  const id = `scenario-field-${path.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  const helpId = `${id}-help`;
  if (values.length) {
    return <label className="field-label" data-field-path={path} htmlFor={id}>
      <span>{label}</span>
      <select id={id} name={path} value={String(value)} disabled={disabled || field.mode === 'invariant'} aria-describedby={helpId} onChange={(event)=>onChange(values.find((item)=>String(item)===event.target.value) ?? event.target.value)}>
        {values.map((item)=><option value={String(item)} key={String(item)}>{optionLabel(authoringCatalog,item)}</option>)}
      </select>
      <small id={helpId}>{tCatalog(String(field.helpKey))}</small>
    </label>;
  }
  if (field.control === 'boolean') {
    return <label className="scenario-boolean-field" data-field-path={path} htmlFor={id}>
      <input id={id} name={path} type="checkbox" value="true" checked={value === true} disabled={disabled || field.mode === 'invariant'} aria-describedby={helpId} onChange={(event)=>onChange(event.target.checked)}/>
      <span>{label}</span>
      <small id={helpId}>{tCatalog(String(field.helpKey))}</small>
    </label>;
  }
  return <label className="field-label" data-field-path={path} htmlFor={id}>
    <span>{label}</span>
    <input
      id={id}
      name={path}
      type={field.control === 'integer' || field.control === 'number' ? 'number' : 'text'}
      min={typeof field.minimum === 'number' ? field.minimum : undefined}
      max={maximum ?? (typeof field.maximum === 'number' ? field.maximum : undefined)}
      minLength={typeof field.minimumLength === 'number' ? field.minimumLength : undefined}
      maxLength={typeof field.maximumLength === 'number' ? field.maximumLength : undefined}
      required={field.required === true}
      step={field.control === 'integer' ? 1 : field.control === 'number' ? 'any' : undefined}
      value={typeof value === 'number' || typeof value === 'string' ? value : ''}
      disabled={disabled || field.mode === 'invariant'}
      aria-describedby={helpId}
      onChange={(event)=>onChange((field.control === 'integer' || field.control === 'number') && event.target.value !== '' ? Number(event.target.value) : event.target.value === '' ? null : event.target.value)}
    />
    <small id={helpId}>{tCatalog(String(field.helpKey))}</small>
  </label>;
}

function CatalogMultiField({
  catalog,
  authoringCatalog,
  path,
  label,
  value,
  onChange,
}: {
  catalog: ResearchFieldCatalog;
  authoringCatalog: ScenarioAuthoringCatalogV1;
  path: string;
  label: string;
  value: string[];
  onChange: (value: string[]) => void;
}) {
  const field = catalog.fields[path];
  if (!field || !Array.isArray(field.values)) {
    throw new Error(`scenario_catalog_field_missing:${path}`);
  }
  return <fieldset className="scenario-catalog-multi" data-field-path={path}>
    <legend>{label}</legend>
    <small>{tCatalog(String(field.helpKey))}</small>
    <div>{field.values.map((item) => {
      const option = String(item);
      return <label key={option}><input type="checkbox" value={option} checked={value.includes(option)} onChange={(event)=>onChange(event.target.checked ? [...value,option] : value.filter((current)=>current!==option))}/><span>{optionLabel(authoringCatalog,option)}</span></label>;
    })}</div>
  </fieldset>;
}

function focusDiagnostic(issue: ScenarioDiagnosticV1) {
  const predicate = issue.predicateIds[0];
  const fieldPath = issue.fieldPaths[0] ?? issue.field ?? '';
  const argumentSuffix = fieldPath.includes('.arguments.')
    ? fieldPath.slice(fieldPath.lastIndexOf('.arguments.') + '.arguments.'.length)
    : '';
  const preferred = [
    fieldPath,
    predicate && argumentSuffix ? `${predicate}.arguments.${argumentSuffix}` : '',
    predicate ? `${predicate}.${fieldPath.split('.').at(-1) ?? ''}` : '',
  ].filter(Boolean);
  const candidates = [...document.querySelectorAll<HTMLElement>('[data-field-path]')];
  let target = candidates.find((item)=>preferred.includes(item.dataset.fieldPath ?? '')) ?? null;
  if (!target && fieldPath) target = candidates.find((item)=>{
    const candidate = item.dataset.fieldPath ?? '';
    return fieldPath.endsWith(candidate) || candidate.endsWith(fieldPath);
  }) ?? null;
  target ??= predicate ? document.getElementById(`scenario-${predicate}`) : null;
  target ??= document.getElementById('scenario-studio-title');
  target?.scrollIntoView({block: 'center'});
  (target?.matches('input,select,textarea,button')
    ? target
    : target?.querySelector<HTMLElement>('input:not([disabled]),select:not([disabled]),textarea:not([disabled]),button:not([disabled])')
      ?? target)?.focus();
}

function diagnosticMessage(issue: ScenarioDiagnosticV1) {
  if (issue.messageAr) return issue.messageAr;
  if (issue.code === 'all_decision_quality_evidence_unavailable')
    return t('workflow.research_scenario_help_require_all_decision_audit');
  if (issue.code === 'continuation_count_requires_selected_action')
    return t('workflow.research_scenario_issue_continuation_count_requires_selected_action');
  if (issue.code === 'continuation_count_line_universe_mismatch')
    return t('workflow.research_scenario_issue_continuation_count_line_universe_mismatch');
  if (issue.messageArKey) {
    try { return tCatalog(issue.messageArKey); } catch { /* governed key drift is visible below */ }
  }
  return issue.code;
}

function structuralInfeasibilityDiagnostic(issues: ScenarioDiagnosticV1[]) {
  return issues.find((issue) => {
    const status = String(issue.details.status ?? '');
    return issue.class === 'structurally_infeasible'
      || issue.code === 'exact_structural_contradiction'
      || issue.code === 'exact_structural_infeasibility'
      || status === 'exact_structural_contradiction'
      || status === 'exact_structural_infeasibility';
  });
}

function structuralTheoremId(issue: ScenarioDiagnosticV1 | undefined) {
  if (!issue) return '';
  const structuralInfeasibility = jsonObject(issue.details.structuralInfeasibility);
  const theorem = jsonObject(structuralInfeasibility.theorem ?? issue.details.theorem);
  return String(
    theorem.theoremId
      ?? structuralInfeasibility.theoremId
      ?? issue.details.theoremId
      ?? '',
  );
}

function predicateArgumentErrors(
  validation: ResearchRecipeValidationV2 | null,
  predicateId: string,
  argumentName: string,
): CardSelectorError[] {
  if (!validation) return [];
  const argumentMarker = `.arguments.${argumentName}`;
  const fields: readonly NonNullable<CardSelectorError['field']>[] = [
    'cards', 'ranks', 'suits', 'ekka', 'owner',
  ];
  return validation.issues.flatMap((issue) => {
    const matchingPaths = issue.fieldPaths.filter((path) => (
      path.includes(argumentMarker)
      && (!issue.predicateIds.length || issue.predicateIds.includes(predicateId))
    ));
    if (!matchingPaths.length && !issue.predicateIds.includes(predicateId)) return [];
    const path = matchingPaths[0] ?? '';
    const suffix = path.includes(argumentMarker)
      ? path.slice(path.indexOf(argumentMarker) + argumentMarker.length)
      : '';
    const field = fields.find((candidate) => (
      suffix === `.${candidate}` || suffix.startsWith(`.${candidate}.`)
    ));
    return [{message: diagnosticMessage(issue), ...(field ? {field} : {})}];
  });
}

function Diagnostics({issues}: {issues: ScenarioDiagnosticV1[]}) {
  if (!issues.length) return null;
  return <ol className="scenario-diagnostic-list" aria-live="polite">
    {issues.map((issue)=><li className={issue.severity} key={issue.issueId ?? issue.code}>
      <button type="button" onClick={()=>focusDiagnostic(issue)}>
        <span>{scenarioDiagnosticSeverityLabels[issue.severity]()}</span>
        <b>{diagnosticMessage(issue)}</b>
        <code dir="ltr">{issue.code}</code>
      </button>
      {Object.keys(issue.details).length ? <details><summary>{t('workflow.research_scenario_diagnostic_details')}</summary><pre dir="ltr">{JSON.stringify(issue.details,null,2)}</pre></details> : null}
    </li>)}
  </ol>;
}

function estimateRefusalDetail(reason: string, estimate: ResearchEstimate): ReactNode {
  const proof = estimate.proof;
  if (proof?.requestedDepth != null && proof.budgetMaximumDepth != null && (reason === 'requested_depth_exceeds_proof_budget' || reason === 'exact_depth_exceeds_proof_budget')) {
    return <>{t('workflow.research_scenario_estimate_requested_depth')}: {number.format(Number(proof?.requestedDepth ?? 0))} · {t('workflow.research_scenario_estimate_budget_depth')}: {number.format(Number(proof?.budgetMaximumDepth ?? 0))}</>;
  }
  if (proof?.estimatedHiddenWorldUpperBound != null && proof.budgetMaximumHiddenWorlds != null && (reason === 'hidden_worlds_exceed_refusal_threshold' || reason === 'hidden_worlds_exceed_proof_budget')) {
    return <>{t('workflow.research_scenario_estimate_hidden_world_upper')}: {number.format(Number(proof?.estimatedHiddenWorldUpperBound ?? 0))} · {t('workflow.research_scenario_estimate_hidden_world_budget')}: {number.format(Number(proof?.budgetMaximumHiddenWorlds ?? 0))}</>;
  }
  if (proof?.estimatedStateSpace != null && proof.budgetStateSpaceLimit != null && (reason === 'estimated_states_exceed_refusal_threshold' || reason === 'estimated_state_space_exceeds_exact_profile')) {
    return <>{t('workflow.research_scenario_estimate_state_space_exact')}: {number.format(Number(proof?.estimatedStateSpace ?? 0))} · {t('workflow.research_scenario_estimate_state_space_budget')}: {number.format(Number(proof?.budgetStateSpaceLimit ?? 0))}</>;
  }
  if (estimate.output && (reason.startsWith('requested_result_count_') || reason === 'requested_seed_count_exceeds_candidate_budget')) {
    return <>{t('workflow.research_label_result_count')}: {number.format(estimate.output?.requestedResultCount ?? 0)} · {t('workflow.research_estimate_candidates')}: {number.format(estimate.output?.maximumCandidateCount ?? 0)}</>;
  }
  return t('workflow.research_scenario_estimate_not_admitted');
}

function EstimatePanel({
  estimate,
  ortools,
  launchBlockReason = '',
  requestedSolverSeconds,
}: {
  estimate: ResearchEstimate;
  ortools?: ScenarioResearchV4CapabilityV1;
  launchBlockReason?: string;
  requestedSolverSeconds?: number;
}) {
  const proof = estimate.proof;
  const authoringCatalog = ortools?.executionCapabilities.authoringCatalog;
  const buckets = Object.entries(estimate.predicatesByConstructionCapability ?? {});
  const displayedWarnings = estimate.warnings.filter(
    (warning)=>warning!=='parallelism_limited_by_shard_plan'
      && warning!=='This estimate is a cost forecast, not a feasibility proof.'
      && warning!=='Model and graph sizes are conservative deterministic upper bounds.',
  ).map((warning)=>warning==='playing_quality_full_round_cost_not_estimated'
    && hasFoundryCopy('workflow.research_scenario_estimate_full_round_unknown')
      ? t('workflow.research_scenario_estimate_full_round_unknown')
      : t('workflow.research_scenario_estimate_unavailable'));
  const structuralInfeasibility = structuralInfeasibilityDiagnostic(estimate.issues);
  const genericPlaySearch = estimate.constructionPlan?.planner === 'generic'
    && estimate.constructionPlan.fallbackReason === 'decision_boundary_selector_incomplete';
  const launchAdmissible = estimate.valid
    && estimate.launchAdmissible === true
    && !structuralInfeasibility
    && !launchBlockReason;
  const exactValue = (value: JsonValue | undefined) => {
    if (value === null || value === undefined || value === 'not_estimable' || value === 'not_compiled') return t('workflow.research_scenario_estimate_unavailable');
    if (typeof value === 'boolean') return t(value ? 'workflow.research_rule_authority_value_yes' : 'workflow.research_rule_authority_value_no');
    if (typeof value === 'string' && /^\d+$/.test(value)) return number.format(BigInt(value));
    return typeof value === 'number' ? number.format(value) : String(value);
  };
  const costClass = (value: ResearchEstimate['firstResultCostClass']) => value === 'within_selected_tier'
    ? t('workflow.research_scenario_estimate_admitted')
    : value === 'exceeds_selected_tier'
      ? t('workflow.research_scenario_estimate_not_admitted')
      : t('workflow.research_scenario_estimate_unavailable');
  const refusal = launchBlockReason || estimate.refusalReason;
  return <section className="scenario-estimate" aria-live="polite">
    <header><b>{t('workflow.research_estimate_title')}</b><span className={`status ${structuralInfeasibility ? 'impossible' : launchAdmissible ? 'completed' : 'failed'}`}>{structuralInfeasibility ? diagnosticMessage(structuralInfeasibility) : launchAdmissible ? t('workflow.research_scenario_estimate_admitted') : t('workflow.research_scenario_estimate_not_admitted')}</span></header>
    {structuralInfeasibility ? <p className="scenario-launch-block" role="status" aria-live="polite" aria-atomic="true"><strong>{diagnosticMessage(structuralInfeasibility)}</strong><code dir="ltr">{structuralInfeasibility.code}</code>{structuralTheoremId(structuralInfeasibility)?<code dir="ltr">{structuralTheoremId(structuralInfeasibility)}</code>:null}</p> : null}
    <dl>
      <div><dt>{t('workflow.research_estimate_candidates')}</dt><dd>{estimate.candidateCount}</dd></div>
      <div><dt>{t('workflow.research_estimate_shards')}</dt><dd>{estimate.shardCount}</dd></div>
      <div><dt>{t('workflow.research_estimate_candidates')} · {t('workflow.research_estimate_shards')}</dt><dd>{Math.ceil(estimate.candidateCount / Math.max(1, estimate.shardCount))}</dd></div>
      <div><dt>{t('workflow.research_estimate_solver')}</dt><dd>{number.format(estimate.maximumSolverSeconds)} {t('workflow.research_seconds')}</dd></div>
      <div><dt>{t('workflow.research_scenario_search_budgets')}</dt><dd>{number.format(estimate.maximumCampaignSeconds ?? estimate.maximumSolverSeconds)} {t('workflow.research_seconds')}</dd></div>
      <div><dt>{t('workflow.research_estimate_analysis')}</dt><dd>{estimate.analysisDecisionUpperBound}</dd></div>
      <div><dt>{t('workflow.research_estimate_tactical_verification')}</dt><dd>{estimate.tacticalVerificationDecisionUpperBound}</dd></div>
      <div><dt>{t('workflow.research_scenario_estimate_cost')}</dt><dd>{estimate.firstResultCostClass === 'not_estimable' ? t('workflow.research_scenario_estimate_unavailable') : estimate.expectedCostCategory === 'bounded' ? t('workflow.research_scenario_proof_bounded') : estimate.expectedCostCategory === 'policy_conditioned' ? t('workflow.research_scenario_proof_policy_conditioned') : estimate.expectedCostCategory === 'late_game_exact' ? t('workflow.research_scenario_proof_exact') : estimate.expectedCostCategory === 'offline_heavyweight' ? t('workflow.research_scenario_option_budget_offline_heavyweight_v1') : t('workflow.research_scenario_estimate_not_admitted')}</dd></div>
      <div><dt>{t('workflow.research_scenario_estimate_bundle')}</dt><dd dir="ltr">{estimate.configurationAuthority?.bundleRevisionId ?? '—'}</dd></div>
      <div><dt>{t('workflow.research_scenario_estimate_hash')}</dt><dd dir="ltr">{estimate.estimateHash ?? '—'}</dd></div>
    </dl>
    {requestedSolverSeconds != null && requestedSolverSeconds > estimate.maximumSolverSeconds ? <p className="scenario-estimate-warnings" role="status">{t('workflow.research_scenario_proof_budgets_help')} <code dir="ltr">{requestedSolverSeconds} → {estimate.maximumSolverSeconds}</code></p> : null}
    {estimate.sharding ? <section className="scenario-sharding-estimate">
      <h4>{t('workflow.research_estimate_shards')}</h4>
      <dl>
        <div><dt>{t('workflow.research_scenario_estimate_configured_seeds_per_shard')}</dt><dd>{number.format(estimate.sharding.configuredSeedsPerShard)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_actual_shards')}</dt><dd>{number.format(estimate.sharding.actualShardCount)} / {number.format(estimate.sharding.requestedShardCount)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_maximum_shard_workers')}</dt><dd>{number.format(estimate.sharding.maximumShardWorkers)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_effective_shard_workers')}</dt><dd>{number.format(estimate.sharding.effectiveShardWorkers)}</dd></div>
        {estimate.sharding.maximumShardWallSeconds != null ? <div><dt>{t('workflow.research_scenario_estimate_shard_wall_seconds')}</dt><dd>{number.format(Math.round(estimate.sharding.maximumShardWallSeconds * 10) / 10)}</dd></div> : null}
      </dl>
      {estimate.sharding.maximumShardWallSeconds != null && estimate.sharding.maximumShardWallSeconds < estimate.maximumSolverSeconds ? <p className="scenario-estimate-warnings" role="status">{t('workflow.research_scenario_estimate_shard_wall_limited')}</p> : null}
      {estimate.sharding.parallelismLimitedByShardPlan ? <p className="scenario-estimate-warnings" role="status"><strong>{t('workflow.research_scenario_estimate_parallelism_limited')}</strong> {t('workflow.research_scenario_estimate_recommended_seeds_per_shard')}: <b>{number.format(estimate.sharding.recommendedSeedsPerShard)}</b></p> : null}
    </section> : null}
    {ortools ? <section className="scenario-ortools-estimate">
      <h4>{t('workflow.research_scenario_estimate_construction')}</h4>
      <div className="detail-tags" aria-label={t('workflow.research_scenario_estimate_construction')}>
        {(estimate.compiledStages ?? []).map((stage)=><code dir="ltr" key={stage}>{stage}</code>)}
      </div>
      <dl>
        <div><dt>{t('workflow.research_scenario_estimate_backend')}</dt><dd dir="ltr">{exactValue(estimate.backendIdentity?.backendId)}</dd></div>
        {estimate.constructionPlan ? <div><dt>{t('workflow.research_scenario_planner')}</dt><dd>{estimate.constructionPlan.planner === 'tail_first' ? tCatalog('research_scenario_planner_tail_first_v1') : genericPlaySearch ? t('workflow.research_scenario_estimate_generic_play_search') : <code dir="ltr">{estimate.constructionPlan.planner}</code>}</dd></div> : null}
        <div><dt>{t('workflow.research_scenario_estimate_resource_tier')}</dt><dd>{estimate.resourceTier && authoringCatalog ? catalogValueLabel(authoringCatalog,estimate.resourceTier) : exactValue(estimate.resourceTier)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_model_variables')}</dt><dd>{exactValue(estimate.estimatedModelVariables)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_model_constraints')}</dt><dd>{exactValue(estimate.estimatedModelConstraints)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_graph_nodes')}</dt><dd>{exactValue(estimate.estimatedGraphNodes)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_graph_edges')}</dt><dd>{exactValue(estimate.estimatedGraphEdges)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_hidden_world_count')}</dt><dd>{exactValue(estimate.estimatedHiddenWorlds)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_count_limbs')}</dt><dd>{exactValue(estimate.estimatedCountLimbs)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_first_result_cost')}</dt><dd>{costClass(estimate.firstResultCostClass)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_requested_results_cost')}</dt><dd>{costClass(estimate.requestedResultsCostClass)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_combination_supported')}</dt><dd>{exactValue(estimate.requestedCombinationSupported)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_exhaustive_infeasibility')}</dt><dd>{exactValue(estimate.exhaustiveSearchInfeasibilityEstablishable)}</dd></div>
      </dl>
      {genericPlaySearch ? <p className="scenario-estimate-warnings" data-testid="scenario-generic-play-search-warning" role="status" aria-live="polite" aria-atomic="true">{t('workflow.research_scenario_estimate_generic_play_search')}</p> : null}
      {estimate.alternativeProofModes?.length && authoringCatalog ? <p><strong>{t('workflow.research_scenario_estimate_alternative')}</strong> {estimate.alternativeProofModes.map((mode)=>catalogValueLabel(authoringCatalog,mode)).join(' · ')}</p> : null}
      {refusal ? <div className="scenario-launch-block"><strong>{tCatalog(scenarioLaunchBlockMessageKey(refusal))}</strong> {estimateRefusalDetail(refusal,estimate)} <details><summary>{t('workflow.research_scenario_diagnostic_details')}</summary><code dir="ltr">{refusal}</code></details></div> : null}
    </section> : null}
    {estimate.output ? <section className={`scenario-proof-forecast ${estimate.output.admissible?'admissible':'refused'}`}>
      <h4>{t('workflow.research_scenario_output_title')}</h4>
      <dl>
        <div><dt>{t('workflow.research_label_result_count')}</dt><dd>{estimate.output.requestedResultCount}</dd></div>
        <div><dt>{t('workflow.research_scenario_desired_videos')}</dt><dd>{estimate.output.requestedVideoCount}</dd></div>
        <div><dt>{t('workflow.research_estimate_candidates')}</dt><dd>{estimate.output.maximumCandidateCount}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_alternative')}</dt><dd>{estimate.output.recommendedResultCount}</dd></div>
        <div><dt>{t('workflow.research_scenario_resources_title')}</dt><dd>{exactValue(estimate.output.requiresResourceExpansion)}</dd></div>
      </dl>
      {!estimate.output.admissible&&estimate.output.reason?<p><strong>{tCatalog(scenarioLaunchBlockMessageKey(estimate.output.reason))}</strong> {estimateRefusalDetail(estimate.output.reason,estimate)}</p>:null}
    </section>:null}
    {proof ? <section className={`scenario-proof-forecast ${proof.admissible ? 'admissible' : 'refused'}`}>
      <h4>{t('workflow.research_scenario_estimate_proof')}</h4>
      <dl>
        <div><dt>{t('workflow.research_scenario_proof_mode')}</dt><dd>{authoringCatalog ? catalogValueLabel(authoringCatalog,proof.requestedMode) : proof.requestedMode}</dd></div>
        <div><dt>{t('workflow.research_scenario_visibility')}</dt><dd>{authoringCatalog ? catalogValueLabel(authoringCatalog,proof.visibility) : proof.visibility}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_root_bound')}</dt><dd>{proof.estimatedRootActionUpperBound}</dd></div>
        <div><dt>{t('workflow.research_scenario_strategy_class')}</dt><dd>{authoringCatalog ? catalogValueLabel(authoringCatalog,proof.strategyClass) : proof.strategyClass}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_action_domain')}</dt><dd>{exactValue(proof.actionDomainUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_ekka_variants')}</dt><dd>{exactValue(proof.estimatedEkkaActionVariantUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_ekka_sequences')}</dt><dd>{exactValue(proof.ekkaDeclarationSequenceUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_target_information_sets')}</dt><dd>{exactValue(proof.targetInformationSetUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_opponent_information_sets')}</dt><dd>{exactValue(proof.opponentInformationSetUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_target_policies')}</dt><dd>{exactValue(proof.targetPolicyCountUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_opponent_policies')}</dt><dd>{exactValue(proof.opponentPolicyCountUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_policy_product')}</dt><dd>{exactValue(proof.policyProductUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_tree_states')}</dt><dd>{exactValue(proof.estimatedTreeStateUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_policy_budget')}</dt><dd>{exactValue(proof.policyProductExceedsBudget)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_information_budget')}</dt><dd>{exactValue(proof.informationTreeExceedsBudget)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_state_space')}</dt><dd>{proof.estimatedStateSpaceClass === 'refused' ? t('workflow.research_scenario_estimate_not_admitted') : t('workflow.research_scenario_estimate_admitted')}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_hidden_worlds')}</dt><dd>{proof.hiddenWorldClass === 'complete_state' ? t('workflow.research_scenario_visibility_complete_state') : t('workflow.research_scenario_visibility_actor_information_set')}</dd></div>
        <div><dt>{t('workflow.research_scenario_budget')}</dt><dd>{authoringCatalog ? catalogValueLabel(authoringCatalog,proof.budgetProfile) : proof.budgetProfile}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_resource_budget')}</dt><dd>{authoringCatalog ? catalogValueLabel(authoringCatalog,proof.resourceBudgetProfile) : proof.resourceBudgetProfile}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_remaining_cards')}</dt><dd>{exactValue(proof.remainingCardCount)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_requested_depth')}</dt><dd>{exactValue(proof.requestedDepth)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_budget_depth')}</dt><dd>{exactValue(proof.budgetMaximumDepth)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_budget_horizon')}</dt><dd>{exactValue(proof.budgetMaximumHorizonActions)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_hidden_world_upper')}</dt><dd>{exactValue(proof.estimatedHiddenWorldUpperBound)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_hidden_world_budget')}</dt><dd>{exactValue(proof.budgetMaximumHiddenWorlds)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_state_space_exact')}</dt><dd>{exactValue(proof.estimatedStateSpace)}</dd></div>
        <div><dt>{t('workflow.research_scenario_estimate_state_space_budget')}</dt><dd>{exactValue(proof.budgetStateSpaceLimit)}</dd></div>
      </dl>
      {!proof.admissible && proof.reason ? <p><strong>{tCatalog(scenarioLaunchBlockMessageKey(proof.reason))}</strong> {estimateRefusalDetail(proof.reason,estimate)}</p> : null}
      {proof.recommendedModes.length && authoringCatalog ? <p><strong>{t('workflow.research_scenario_estimate_alternative')}</strong> {proof.recommendedModes.map((mode)=>catalogValueLabel(authoringCatalog,mode)).join(' · ')}</p> : null}
      <div className="scenario-estimate-identities"><code dir="ltr">{proof.budgetProfileHash ?? '—'}</code><code dir="ltr">{proof.proofBudgetRegistryHash}</code></div>
    </section> : null}
    {displayedWarnings.length ? <section className="scenario-estimate-warnings"><h4>{t('workflow.research_scenario_estimate_warnings')}</h4><ul>{displayedWarnings.map((warning)=><li key={warning}>{warning}</li>)}</ul></section> : null}
    {buckets.length ? <details><summary>{t('workflow.research_scenario_estimate_construction')}</summary><dl>{buckets.map(([capability,predicates])=><div key={capability}><dt>{capabilityLabel(capability)}</dt><dd dir="ltr">{predicates.join(' · ') || '—'}</dd></div>)}</dl></details> : null}
    <Diagnostics issues={estimate.issues}/>
    <div className="scenario-estimate-identities"><code dir="ltr">{estimate.scenarioPlanHash ?? '—'}</code><code dir="ltr">{estimate.backendIdentity?.backendId === 'ortools_cp_sat_v3' ? estimate.nativeCapabilityPlanHash ?? '—' : estimate.capabilityPlanHash ?? '—'}</code><code dir="ltr">{estimate.configurationAuthority?.sha256 ?? '—'}</code></div>
  </section>;
}

function OrToolsAuthorityPanel({
  capability,
  launchBlockReason,
  unsupportedPredicateIds,
}: {
  capability: ScenarioResearchV4CapabilityV1;
  launchBlockReason: string;
  unsupportedPredicateIds: string[];
}) {
  const identity = capability.backendIdentity;
  return <section className="panel scenario-ortools-authority" aria-labelledby="scenario-ortools-authority-title">
    <header>
      <div><h3 id="scenario-ortools-authority-title">{tCatalog('research_scenario_authority_title')}</h3><p>{tCatalog('research_scenario_authority_help')}</p></div>
      <span className={`status ${launchBlockReason?'failed':'completed'}`}>{launchBlockReason?t('workflow.research_scenario_estimate_not_admitted'):t('workflow.research_scenario_estimate_admitted')}</span>
    </header>
    <dl>
      <div><dt>{tCatalog('research_scenario_backend')}</dt><dd>{tCatalog(`research_scenario_backend_${identity.backendId}`)}</dd><small>{tCatalog('research_scenario_backend_help')}</small></div>
      <div><dt>{tCatalog('research_scenario_planner')}</dt><dd>{tCatalog(`research_scenario_planner_${identity.planner}`)}</dd><small>{tCatalog('research_scenario_planner_help')}</small></div>
      <div><dt>{tCatalog('research_scenario_deterministic_execution')}</dt><dd>{identity.deterministicExecution?'✓':'×'}</dd><small>{tCatalog('research_scenario_deterministic_execution_help')}</small></div>
      <div><dt><code dir="ltr">qualificationAuthority</code></dt><dd dir="ltr">{identity.qualificationAuthority}</dd><small>{identity.solverVersion}</small></div>
      <div><dt>{t('workflow.research_scenario_predicate')}</dt><dd>{capability.hardPredicateCatalog.definitions.length}</dd><small dir="ltr">{capability.hardPredicateCatalog.sha256.slice(0,16)}</small></div>
    </dl>
    {launchBlockReason ? <p className="scenario-launch-block" role="status"><strong>{tCatalog(scenarioLaunchBlockMessageKey(launchBlockReason))}</strong> <code dir="ltr">{launchBlockReason}</code>{unsupportedPredicateIds.length?<span dir="ltr"> {unsupportedPredicateIds.join(' · ')}</span>:null}</p> : null}
    <div className="scenario-estimate-identities"><code dir="ltr">{identity.sha256}</code><code dir="ltr">{identity.loweringManifestHash ?? capability.hardPredicateCatalog.loweringManifestHash ?? '—'}</code></div>
  </section>;
}

type RuleAuthorityPanelState = {
  identity: RulesAuthorityIdentity;
  boundary: RulesCertificationBoundary;
  comparison: ProfileComparison | null;
  delta: ProfileComparisonChange[];
};

function ruleAuthorityValue(value: JsonValue | undefined): ReactNode {
  if (value === undefined || value === null) return t('workflow.research_rule_authority_value_missing');
  if (value === true) return t('workflow.research_rule_authority_value_yes');
  if (value === false) return t('workflow.research_rule_authority_value_no');
  if (typeof value === 'number') return number.format(value);
  if (typeof value === 'string') {
    if (value === 'mandatory') return t('workflow.research_rule_authority_value_mandatory');
    if (value === 'optional') return t('workflow.research_rule_authority_value_optional');
    if (value === 'published_unratified_mixed_authority') return t('workflow.research_rule_authority_status_mixed');
    if (value === 'published_unratified_configured_variant') return t('workflow.research_rule_authority_status_variant');
    if (value === 'candidate_external_review_required') return t('workflow.research_rule_authority_boundary_external_review');
    if (value === 'configured_variant_not_certification_eligible') return t('workflow.research_rule_authority_boundary_ineligible');
    return <code dir="ltr">{value}</code>;
  }
  if (Array.isArray(value)) return <span className="rule-authority-value-list">{value.map((item,index)=><span key={`${String(item)}-${index}`}>{ruleAuthorityValue(item)}</span>)}</span>;
  return t('workflow.research_rule_authority_value_unavailable');
}

function ruleAuthorityPath(path: string) {
  if (path === 'projects.declaration_mode') {
    return t('workflow.research_rule_authority_path_project_declaration');
  }
  return <code dir="ltr">{path}</code>;
}

function RuleAuthorityPanel({
  selectedBundle,
  officialRulesRevisionId,
  rulesHash,
  request,
}: {
  selectedBundle: ConfigBundle;
  officialRulesRevisionId: string | null;
  rulesHash: string;
  request: ScenarioStudioProps['request'];
}) {
  const [state,setState] = useState<RuleAuthorityPanelState|null>(null);
  const [failed,setFailed] = useState(false);
  const rulesRevisionId = selectedBundle.componentRevisions.rules ?? '';
  const certificationRevisionId = selectedBundle.componentRevisions.certification ?? '';
  useEffect(()=>{
    const controller = new AbortController();
    setState(null);
    setFailed(false);
    if (!rulesRevisionId || !certificationRevisionId) {
      setFailed(true);
      return ()=>controller.abort();
    }
    const profile = (revisionId:string) => request<ConfigProfile>(
      `/api/v1/configuration-registry/profiles/${encodeURIComponent(revisionId)}`,
      {signal:controller.signal},
    );
    Promise.all([
      profile(rulesRevisionId),
      profile(certificationRevisionId),
      officialRulesRevisionId && officialRulesRevisionId !== rulesRevisionId
        ? request<ProfileComparison>(
          `/api/v1/configuration-registry/profiles:compare?left=${encodeURIComponent(officialRulesRevisionId)}&right=${encodeURIComponent(rulesRevisionId)}`,
          {signal:controller.signal},
        )
        : Promise.resolve(null),
    ]).then(([rulesProfile,certificationProfile,comparison])=>{
      if (controller.signal.aborted) return;
      const identity = rulesAuthorityIdentity(rulesProfile,rulesRevisionId,rulesHash);
      const boundary = rulesCertificationBoundary(certificationProfile,identity);
      if (comparison && (
        comparison.left.revisionId !== officialRulesRevisionId
        || comparison.right.revisionId !== rulesRevisionId
      )) throw new Error('research_rule_authority_comparison_identity_mismatch');
      setState({
        identity,
        boundary,
        comparison,
        delta: comparison ? semanticRulesDelta(comparison) : [],
      });
    }).catch(()=>{
      if (!controller.signal.aborted) setFailed(true);
    });
    return ()=>controller.abort();
  },[certificationRevisionId,officialRulesRevisionId,request,rulesHash,rulesRevisionId]);

  return <section className="panel scenario-rule-authority" aria-labelledby="scenario-rule-authority-title">
    <header>
      <div><h3 id="scenario-rule-authority-title">{t('workflow.research_rule_authority_title')}</h3><p>{t('workflow.research_rule_authority_help')}</p></div>
      <span className={`status ${failed?'failed':state?'completed':'running'}`} role="status">{failed?t('workflow.research_rule_authority_failed'):state?t('workflow.research_rule_authority_ready'):t('workflow.research_rule_authority_loading')}</span>
    </header>
    {state ? <>
      <dl className="scenario-rule-authority-identities">
        <div><dt>{t('workflow.research_rule_authority_profile')}</dt><dd><code dir="ltr" data-testid="rule-profile-id">{state.identity.profileId}</code></dd></div>
        <div><dt>{t('workflow.research_rule_authority_version')}</dt><dd><code dir="ltr" data-testid="rule-profile-version">{state.identity.version}</code></dd></div>
        <div><dt>{t('workflow.research_rule_authority_content_hash')}</dt><dd><code dir="ltr" data-testid="rule-profile-hash">{state.identity.payloadSha256}</code></dd></div>
        <div><dt>{t('workflow.research_rule_authority_source_catalog')}</dt><dd><code dir="ltr" data-testid="rule-source-catalog">{state.boundary.sourceCatalog.id} · {state.boundary.sourceCatalog.version}</code><small><code dir="ltr">{state.boundary.sourceCatalog.sha256}</code></small></dd></div>
        <div><dt>{t('workflow.research_rule_authority_catalog_status')}</dt><dd data-testid="rule-profile-status">{ruleAuthorityValue(state.identity.profileStatus)}</dd></div>
        <div><dt>{t('workflow.research_rule_authority_ratification')}</dt><dd data-testid="rule-ratification-boundary">{ruleAuthorityValue(state.boundary.status)}</dd><small>{state.boundary.certificationEligible?t('workflow.research_rule_authority_certification_eligible'):t('workflow.research_rule_authority_certification_ineligible')}</small></div>
      </dl>
      <section className="scenario-rule-authority-delta" aria-labelledby="scenario-rule-authority-delta-title">
        <div><h4 id="scenario-rule-authority-delta-title">{t('workflow.research_rule_authority_delta_title')}</h4><p>{state.comparison?t('workflow.research_rule_authority_delta_help'):t('workflow.research_rule_authority_official_selected')}</p></div>
        {state.delta.length ? <dl data-testid="rule-semantic-delta">{state.delta.map((item)=><div key={item.path} data-delta-path={item.path}><dt>{ruleAuthorityPath(item.path)}</dt><dd><span><small>{t('workflow.research_rule_authority_official_value')}</small><b>{ruleAuthorityValue(item.left)}</b></span><span><small>{t('workflow.research_rule_authority_selected_value')}</small><b>{ruleAuthorityValue(item.right)}</b></span></dd></div>)}</dl>:<p className="context-proof" data-testid="rule-semantic-delta-empty"><strong>{t('workflow.research_rule_authority_no_delta')}</strong></p>}
      </section>
    </>:failed?<p className="scenario-launch-block" role="alert">{t('workflow.research_rule_authority_failed_help')}</p>:<p className="empty-state">{t('workflow.research_rule_authority_loading_help')}</p>}
  </section>;
}

function jsonObject(value: JsonValue | undefined): JsonObject {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonObject
    : {};
}

function resultFacetValues(result: ResearchResult, field: keyof ResearchV4ResultFilters) {
  const scenario = jsonObject(result.scenarioProjection);
  const proof = jsonObject(result.proofOutcome);
  const value = {
    scenarioTemplate: scenario.scenarioTemplateId,
    scenarioOrigin: scenario.origin,
    scenarioStatus: scenario.status,
    requestedProofMode: scenario.requestedProofMode ?? proof.requestedMode,
    achievedProofMode: scenario.achievedProofMode ?? proof.achievedMode,
    proofCompletion: scenario.proofCompletion ?? proof.completionStatus,
    proofVisibility: scenario.proofVisibility ?? proof.achievedVisibility,
    targetTeam: scenario.targetTeam ?? proof.targetTeam,
    contractOutcome: scenario.contractOutcome,
    projectKind: scenario.projectKinds,
    projectValue: scenario.projectValues,
    projectState: scenario.projectStates,
    winningRootActionBand: (() => {
      const count = Number(scenario.winningRootActionCount ?? proof.winningRootActionCount);
      return !Number.isFinite(count) ? undefined : count === 0 ? '0' : count === 1 ? '1' : count <= 3 ? '2_3' : '4_plus';
    })(),
    exactStateAvailable: scenario.exactStateAvailable === true ? 'true' : 'false',
    decisionKind: scenario.decisionKind,
    researchBackend: result.researchBackend ?? result.researchAuthority?.backendId,
    qualificationAuthority: result.qualificationAuthority ?? result.researchAuthority?.qualificationAuthority,
    runtimeCompatibilityStatus: result.runtimeCompatibilityStatus ?? result.runtimeCompatibility?.status,
    configurationBundle: result.researchAuthority?.bundleRevisionId,
    exactStateId: scenario.exactStateId,
    scenarioCategory: scenario.scenarioCategories,
    variantCollection: result.runCode,
  }[field];
  return (Array.isArray(value) ? value : [value])
    .filter((item): item is string | number | boolean => ['string','number','boolean'].includes(typeof item))
    .map(String)
    .filter(Boolean);
}

export function scenarioCatalogReady(
  configuration: ResearchScenarioConfigurationV1 | undefined,
  _recipeConfiguration: ResearchRecipeConfigurationCatalog | undefined,
  scenarioResearchV4?: ScenarioResearchV4CapabilityV1,
) {
  if (!configuration || configuration.recipeSchemaVersion !== 4) return false;
  const sameMembers = (left:string[],right:string[]) => left.length===right.length
    && left.every((value)=>right.includes(value));
  const ortools = isOrToolsScenarioCapability(scenarioResearchV4);
  if (ortools) {
    const authoringCatalog = scenarioResearchV4.executionCapabilities.authoringCatalog;
    if (
      authoringCatalog?.schemaVersion !== 1
      || authoringCatalog.recipeSchemaVersion !== 4
      || !/^[0-9a-f]{64}$/.test(authoringCatalog.sha256)
      || !Object.keys(authoringCatalog.fields).length
      || !Object.keys(authoringCatalog.localizationKeys.values).length
    ) return false;
    const limits = authoringCatalog.expressionLimits;
    const matchScoreItems = jsonObject(authoringCatalog.exactTrace.matchScore.items);
    const assertionSchemas = Object.values(jsonObject(
      authoringCatalog.exactTrace.stateAssertions.properties,
    )).map(jsonObject);
    const fields = Object.values(authoringCatalog.fields);
    if (
      fields.some((field)=>!field.labelKey||!field.helpKey)
      || fields.some((field)=>Array.isArray(field.values)&&field.values.some(
        (value)=>!authoringCatalog.localizationKeys.values[String(value)],
      ))
      || !Number.isInteger(limits.maximumDepth) || limits.maximumDepth < 1
      || !Number.isInteger(limits.maximumNodes) || limits.maximumNodes < 1
      || !Number.isInteger(limits.maximumChildren) || limits.maximumChildren < 1
      || !Number.isInteger(limits.maximumSerializedBytes) || limits.maximumSerializedBytes < 1
      || !Number.isInteger(limits.maximumCountRange) || limits.maximumCountRange < 0
      || typeof matchScoreItems.minimum !== 'number'
      || typeof matchScoreItems.maximum !== 'number'
      || matchScoreItems.minimum < 0
      || matchScoreItems.maximum < matchScoreItems.minimum
      || !assertionSchemas.length
      || assertionSchemas.some((schema)=>!schema['x-messageArKey'])
      || assertionSchemas.some((schema)=>Array.isArray(schema.enum)&&schema.enum.some(
        (value)=>!authoringCatalog.localizationKeys.values[String(value)],
      ))
      || !authoringCatalog.policyCatalog.profiles[authoringCatalog.policyCatalog.defaultPolicyId]
      || !(['north','west','south','east'] as const).every((seat)=>authoringCatalog.policyCatalog.defaultProfiles[seat])
      || !['action','actor','card','declarationOrder','ekka','mode','owner','state','trumpSuit'].every(
        (name)=>authoringCatalog.localizationKeys.exactArguments[name],
      )
    ) return false;
    if (!configuration.origins.length
      || !sameMembers(configuration.origins,scenarioResearchV4.supportedOrigins)) return false;
  } else if (!['search','exact_trace_prefix'].every((value)=>configuration.origins.includes(value as 'search' | 'exact_trace_prefix'))) return false;
  if (ortools) {
    const outputKind = scenarioResearchV4.executionCapabilities.authoringCatalog.fields['output.kind'];
    const resultCount = scenarioResearchV4.executionCapabilities.authoringCatalog.fields['output.resultCount'];
    const renderMode = scenarioResearchV4.executionCapabilities.authoringCatalog.fields['renderPolicy.mode'];
    const videoCount = scenarioResearchV4.executionCapabilities.authoringCatalog.fields['renderPolicy.videoCount'];
    if (!Array.isArray(outputKind.values) || !outputKind.values.length
      || !Array.isArray(renderMode.values) || !renderMode.values.length
      || Number(resultCount.minimum) < 1
      || Number(resultCount.default) < Number(resultCount.minimum)
      || Number(resultCount.default) > Number(resultCount.maximum)
      || Number(videoCount.minimum) < 0
      || Number(videoCount.default) < Number(videoCount.minimum)
      || Number(videoCount.default) > Number(videoCount.maximum)) return false;
  } else {
    if (!configuration.outputPolicy.kinds.length || !configuration.outputPolicy.renderModes.length) return false;
    if (configuration.outputPolicy.resultCount.minimum < 1
      || configuration.outputPolicy.resultCount.default < configuration.outputPolicy.resultCount.minimum
      || configuration.outputPolicy.resultCount.default > configuration.outputPolicy.resultCount.maximum
      || configuration.outputPolicy.videoCount.minimum < 0
      || configuration.outputPolicy.videoCount.default < configuration.outputPolicy.videoCount.minimum
      || configuration.outputPolicy.videoCount.default > configuration.outputPolicy.videoCount.maximum) return false;
  }
  if (ortools) {
    const executableStrategies = [...new Set(
      scenarioResearchV4.executionCapabilities.combinations.flatMap(
        (item)=>item.strategyClasses,
      ),
    )];
    if (!executableStrategies.length
      || !sameMembers(configuration.proofStrategyClasses,executableStrategies)) return false;
  } else if (!['pure_deterministic','frozen_policy'].every((value)=>configuration.proofStrategyClasses.includes(value as 'pure_deterministic' | 'frozen_policy'))) return false;
  if (ortools) {
    const proof = scenarioResearchV4.executionCapabilities.authoringCatalog.proof;
    if (!proof.modes.length || !proof.claims.length || !proof.utilities.length
      || !proof.visibilities.length || !proof.boundaryEvaluators.length) return false;
  } else if (!configuration.proofModes.length || !configuration.proofClaims.length
    || !configuration.proofUtilities.length || !configuration.proofVisibilities.length
    || !configuration.proofBoundaryEvaluators.length) return false;
  if (!configuration.ruleOptions.contracts.length || !configuration.ruleOptions.auctionActionsByRound.length) return false;
  const playActionIds = configuration.ruleOptions.playActionOptions.map((action)=>action.actionId);
  const playActionCodes = configuration.ruleOptions.playActionOptions.map((action)=>action.code);
  if (!playActionCodes.length || new Set(playActionIds).size!==playActionIds.length || new Set(playActionCodes).size!==playActionCodes.length) return false;
  if (configuration.ruleOptions.playActionOptions.some((action)=>!action.card.code||!action.card.label||typeof action.ekka!=='boolean')) return false;
  if (!ortools && !configuration.predicateRegistry.definitions.length) return false;
  if (ortools
    ? !scenarioResearchV4.executionCapabilities.authoringCatalog.tacticKinds.length
    : !configuration.tacticOptions.length) return false;
  if (!ortools && configuration.predicateRegistry.definitions.some((definition)=>!schemaHasVisualControl(definition.argumentSchema,definition.kind))) return false;
  const supportedDecisions = configuration.decisionCapabilities.filter(
    (item) => item.authoringSupported,
  );
  if (ortools) {
    const executableDecisionKinds = [...new Set(
      scenarioResearchV4.executionCapabilities.combinations.map(
        (item)=>item.decisionKind,
      ),
    )];
    if (!executableDecisionKinds.length
      || !sameMembers(supportedDecisions.map((item)=>item.kind),executableDecisionKinds)) return false;
  } else {
    if (
      supportedDecisions.length !== 4
      || !['bid', 'escalation', 'project', 'play'].every((kind) =>
        supportedDecisions.some((item) =>
          item.kind === kind
          && item.exactTraceNormalizationSupported
          && item.renderPreparationSupported,
        ),
      )
    ) return false;
    const play = supportedDecisions.find((item) => item.kind === 'play');
    if (!play?.searchGenerationSupported || play.proofModes.length !== 3) return false;
    if (supportedDecisions.some((item) => item.kind !== 'play' && (!item.searchGenerationSupported || item.proofModes.join(',') !== 'policy_conditioned'))) return false;
  }
  if (ortools && (
    scenarioResearchV4.hardPredicateCatalog.schemaVersion !== 2
    || !scenarioResearchV4.hardPredicateCatalog.definitions.length
    || !scenarioResearchV4.supportedOrigins.length
    || !scenarioResearchV4.supportedProofVisibility.length
    || scenarioResearchV4.executionCapabilities.schemaVersion !== 1
    || !scenarioResearchV4.executionCapabilities.combinations.length
    || [...new Set(scenarioResearchV4.executionCapabilities.combinations.map(
      (item)=>item.decisionKind,
    ))].some((kind)=>!selectorSchemaComplete(
      scenarioResearchV4.executionCapabilities.decisionSelectorSchemas[kind] ?? {},
    ))
  )) return false;
  if (!configuration.projectOptions.length
    || configuration.projectOptions.some((item)=>!Number.isInteger(item.highRankId)||item.highRankId<0||item.highRankId>7)
    || !/^[0-9a-f]{64}$/.test(configuration.projectDomainHash)) return false;
  if (!ortools && !['north', 'west', 'south', 'east'].every((seat) => {
    const profile = configuration.defaultPolicyProfiles[seat as keyof typeof configuration.defaultPolicyProfiles];
    return profile && 'agentId' in profile && profile.agentId === 'heuristic_bot_v1'
      && /^[0-9a-f]{64}$/.test(profile.profileHash);
  })) return false;
  try {
    const definitions = mergeScenarioPredicateDefinitions(
      configuration.predicateRegistry.definitions,
      scenarioResearchV4,
    );
    if (ortools && ['team','state'].some(
      (name)=>!Array.isArray(predicateEnumSchema(definitions,name).enum),
    )) {
      return false;
    }
    definitions.forEach((definition) => {
      if (!schemaHasVisualControl(definition.argumentSchema, definition.kind)) {
        throw new Error(`Missing visual control for ${definition.kind}.`);
      }
      predicateLabel(definition);
      tCatalog(definition.messageEnKey);
      if (ortools && definition.cpSatSupport) {
        const authoringCatalog = scenarioResearchV4.executionCapabilities.authoringCatalog;
        predicateHelp(authoringCatalog,definition);
        catalogValueLabel(authoringCatalog,definition.costClass);
        catalogValueLabel(authoringCatalog,definition.cpSatSupport.actorInformationSet);
      }
      else capabilityLabel(definition.constructionCapability);
      stageLabel(definition.stage);
    });
    if (ortools) {
      const authoringCatalog = scenarioResearchV4.executionCapabilities.authoringCatalog;
      Object.values(authoringCatalog.fields).forEach((field) => {
        tCatalog(String(field.labelKey));
        tCatalog(String(field.helpKey));
        if (typeof field.defaultKey === 'string') tCatalog(field.defaultKey);
        if (Array.isArray(field.values)) {
          field.values.forEach((value)=>catalogValueLabel(authoringCatalog,value));
        }
      });
      authoringCatalog.logicalOperators.forEach((value)=>catalogValueLabel(authoringCatalog,value));
      Object.keys(authoringCatalog.proof.budgetProfiles).forEach((value)=>catalogValueLabel(authoringCatalog,value));
      Object.entries(authoringCatalog.policyCatalog.profiles).forEach(([policyId,definition])=>{
        catalogValueLabel(authoringCatalog,policyId);
        catalogValueLabel(authoringCatalog,definition.materialization);
      });
      tCatalog('research_scenario_authority_title');
      tCatalog('research_scenario_backend_help');
      tCatalog('research_scenario_planner_help');
    }
  } catch {
    return false;
  }
  return true;
}

export function ScenarioStudio({
  configuration,
  scenarioResearchV4,
  selectedBundle,
  availableBundles,
  officialRulesRevisionId,
  rulesHash,
  busy,
  request,
  createRecipeAndRun,
  setBusy,
  setMessage,
  refresh,
  onOpenRun,
  onOpenResult,
  runs,
}: ScenarioStudioProps) {
  const ortools = isOrToolsScenarioCapability(scenarioResearchV4);
  const authoringCatalog = scenarioResearchV4?.executionCapabilities.authoringCatalog;
  if (!ortools || !authoringCatalog || authoringCatalog.recipeSchemaVersion !== 4) {
    throw new Error('scenario_v4_authoring_catalog_unavailable');
  }
  const scenarioCatalog = authoringCatalog;
  const fieldCatalog: ResearchFieldCatalog = authoringCatalog;
  const executionCombinations = ortools
    ? scenarioResearchV4.executionCapabilities.combinations
    : [];
  const defaultExecutionCombination = preferredScenarioExecutionCombination(
    scenarioResearchV4,
  );
  const supportedOrigins = ortools
    ? [...new Set(executionCombinations.map((item)=>item.origin))]
    : configuration.origins;
  const playSelectorSchema = scenarioDecisionSelectorSchema(scenarioResearchV4, 'play');
  const playTrickBounds = integerSchemaBounds(
    playSelectorSchema,
    'trickIndex',
    ortools ? undefined : {minimum: 0, maximum: 7, default: 0},
  );
  const playPositionBounds = integerSchemaBounds(
    playSelectorSchema,
    'position',
    ortools ? undefined : {minimum: 0, maximum: 3, default: 0},
    true,
  );
  const playRemainingBounds = integerSchemaBounds(
    playSelectorSchema,
    'remainingCards',
    ortools ? undefined : {minimum: 1, maximum: 32, default: 8},
  );
  const allDefinitions = useMemo(() => {
    const definitions = mergeScenarioPredicateDefinitions(
      configuration.predicateRegistry.definitions,
      scenarioResearchV4,
    );
    if (ortools) return definitions;
    return definitions.filter((definition) => {
    if (!configuration.ruleOptions.projectsEnabled && definition.kind.startsWith('project.')) return false;
    if (!configuration.ruleOptions.balootContracts.length && definition.kind === 'project.baloot') return false;
    if (!configuration.ruleOptions.kaboutEnabled && definition.kind === 'outcome.kabout') return false;
    if (!configuration.ruleOptions.contracts.includes('ashkal') && definition.kind === 'auction.ashkal_buyer_relation') return false;
    if (!configuration.ruleOptions.trumpContracts.length && definition.kind === 'auction.trump') return false;
    if (!configuration.ruleOptions.escalationLevelsByContract.some((item)=>item.levels.includes('gahwa')) && definition.kind === 'outcome.gahwa') return false;
      return true;
    });
  }, [configuration.predicateRegistry.definitions, configuration.ruleOptions, ortools, scenarioResearchV4]);
  const authoringOptions = useMemo<ScenarioAuthoringOptions>(()=>({
    cards: configuration.cardOptions,
    seats: stringSchemaValues(playSelectorSchema,'actingSeat'),
    defaultDealer: stringSchemaDefault(playSelectorSchema,'actingSeat'),
    ranks: [...new Set([...configuration.cardOptions].sort((left,right)=>left.id-right.id).map((card)=>card.rank))],
    suits: [...new Set(configuration.cardOptions.map((card)=>card.suit))],
    contracts: configuration.ruleOptions.contracts,
    auctionActions: [...new Set(configuration.ruleOptions.auctionActionsByRound.flatMap((item)=>item.actions))],
    escalations: [...new Set(configuration.ruleOptions.escalationLevelsByContract.flatMap((item)=>item.actions))],
    escalationsByContract: Object.fromEntries(configuration.ruleOptions.escalationLevelsByContract.map((item)=>[item.contract,item.actions])),
    tactics: ortools ? authoringCatalog.tacticKinds : configuration.tacticOptions,
    teams: (predicateEnumSchema(allDefinitions,'team').enum as JsonValue[]).map(String),
    projectStates: (predicateEnumSchema(allDefinitions,'state').enum as JsonValue[]).map(String),
    projects: configuration.projectOptions,
    playActions: configuration.ruleOptions.playActionOptions,
  }),[allDefinitions,authoringCatalog.tacticKinds,configuration.cardOptions,configuration.projectOptions,configuration.ruleOptions,configuration.tacticOptions,ortools,playSelectorSchema]);
  const [titleAr, setTitleAr] = useState(
    catalogTextDefault(authoringCatalog, 'titleAr'),
  );
  const [origin, setOrigin] = useState<'search' | 'exact_trace_prefix'>(
    defaultExecutionCombination?.origin ?? configuration.origins[0]!,
  );
  const [generationPolicy, setGenerationPolicy] = useState<'natural_policy' | 'scenario_guided_actor_safe'>('scenario_guided_actor_safe');
  const [constraintExpression, setConstraintExpression] = useState<ExpressionGroupDraft>({
    nodeId: 'EXPR-CONSTRAINTS', op: 'all', minimum: 1, maximum: 1, children: [],
  });
  const [outcomeExpression, setOutcomeExpression] = useState<ExpressionGroupDraft>({
    nodeId: 'EXPR-OUTCOME', op: 'all', minimum: 1, maximum: 1, children: [],
  });
  const [selectedKind, setSelectedKind] = useState(allDefinitions[0]?.kind ?? '');
  const [catalogQuery, setCatalogQuery] = useState('');
  const [targetSection, setTargetSection] = useState<'constraints' | 'outcome'>('constraints');
  const [activeExpressionTarget, setActiveExpressionTarget] = useState('EXPR-CONSTRAINTS');
  const [nextPredicate, setNextPredicate] = useState(1);
  const [nextExpression, setNextExpression] = useState(1);
  const [decisionKind, setDecisionKind] = useState<'bid' | 'escalation' | 'project' | 'play'>(
    defaultExecutionCombination?.decisionKind
      ?? configuration.decisionCapabilities.find((item)=>item.authoringSupported)!.kind,
  );
  const activeSelectorSchema = scenarioDecisionSelectorSchema(scenarioResearchV4, decisionKind);
  const supportedDecisionKinds = useMemo(
    () => configuration.decisionCapabilities.filter((item) =>
      item.authoringSupported
      && (origin === 'search' ? item.searchGenerationSupported : item.exactTraceNormalizationSupported)
      && (!ortools || scenarioExecutionCombinations(scenarioResearchV4, {
        origin,
        decisionKind: item.kind,
      }).length > 0),
    ),
    [configuration.decisionCapabilities, origin, ortools, scenarioResearchV4],
  );
  const selectedDecisionCapability = supportedDecisionKinds.find((item) => item.kind === decisionKind) ?? null;
  const [decisionSeat, setDecisionSeat] = useState(
    ortools ? '' : stringSchemaDefault(
      playSelectorSchema,
      'actingSeat',
      authoringOptions.seats[0] ?? 'south',
    ),
  );
  const [decisionActionIndex, setDecisionActionIndex] = useState(0);
  const [decisionTrickIndex, setDecisionTrickIndex] = useState(playTrickBounds.default);
  const [decisionPosition, setDecisionPosition] = useState<number | undefined>();
  const [decisionRemainingCards, setDecisionRemainingCards] = useState(playRemainingBounds.default);
  const [projectIndex, setProjectIndex] = useState(0);
  const [proofMode, setProofMode] = useState<'exact' | 'bounded' | 'policy_conditioned'>(
    defaultExecutionCombination?.mode ?? authoringCatalog.proof.modes[0]!,
  );
  const [visibility, setVisibility] = useState<'complete_state' | 'actor_information_set'>(
    defaultExecutionCombination?.visibility ?? authoringCatalog.proof.visibilities[0]!,
  );
  const [lineUniverse, setLineUniverse] = useState(
    defaultExecutionCombination?.lineUniverses[0]
      ?? authoringCatalog.proof.lineUniverses[0]!,
  );
  const [utility, setUtility] = useState(authoringCatalog.proof.utilities[0]!);
  const scenarioSelection = {origin, decisionKind, proofMode, visibility, lineUniverse, utility} as const;
  const executionSelection = {origin, decisionKind, proofMode, visibility} as const;
  const executionContractValues = ortools
    ? scenarioExecutionContractValues(scenarioResearchV4, executionSelection)
    : null;
  const scopedDefinitions = useMemo(() => allDefinitions.map((definition) => {
    if (definition.kind !== 'auction.contract' || executionContractValues === null) {
      return definition;
    }
    const properties = jsonObject(definition.argumentSchema.properties);
    const contract = jsonObject(properties.contract);
    const restrictKeys = (raw: JsonValue) => Object.fromEntries(
      Object.entries(jsonObject(raw)).filter(([key])=>
        executionContractValues.includes(key as 'sun'|'hokum'|'ashkal')),
    );
    return {
      ...definition,
      argumentSchema: {
        ...definition.argumentSchema,
        properties: {
          ...properties,
          contract: {
            ...contract,
            enum: executionContractValues,
            'x-enumMessageArKeys': restrictKeys(contract['x-enumMessageArKeys']),
            'x-enumMessageEnKeys': restrictKeys(contract['x-enumMessageEnKeys']),
          },
        },
      },
    };
  }), [allDefinitions, executionContractValues]);
  const availableDefinitions = scopedDefinitions.filter((definition) => !ortools
    || definition.allowedHardness.some((hardness) =>
      scenarioPredicateSupported(definition, scenarioSelection, hardness)));
  const definitionMap = useMemo(
    () => new Map(scopedDefinitions.map((definition) => [definition.kind, definition])),
    [scopedDefinitions],
  );
  const targetTeamSchema = useMemo(
    () => predicateEnumSchema(scopedDefinitions,'team'),
    [scopedDefinitions],
  );
  const targetTeamOptions = Array.isArray(targetTeamSchema.enum)
    ? targetTeamSchema.enum.map(String)
    : [];
  const normalizedCatalogQuery = catalogQuery.trim().toLocaleLowerCase();
  const visibleDefinitions = availableDefinitions.filter((definition) => !normalizedCatalogQuery || [
    definition.kind,
    definition.stage,
    predicateLabel(definition),
  ].some((value)=>value.toLocaleLowerCase().includes(normalizedCatalogQuery)));
  const grouped = useMemo(
    () => Object.fromEntries(authoringCatalog.sections.map((stage) => [
      stage,
      visibleDefinitions.filter((definition) => definition.stage === stage),
    ])),
    [authoringCatalog.sections, visibleDefinitions],
  );
  const availableProofModes = useMemo(
    () => ortools
      ? [...new Set(scenarioExecutionCombinations(scenarioResearchV4, {
        origin,
        decisionKind,
      }).map((item)=>item.mode))]
      : selectedDecisionCapability?.proofModes ?? [],
    [decisionKind, origin, ortools, scenarioResearchV4, selectedDecisionCapability],
  );
  const availableVisibilities = useMemo(
    () => ortools
      ? [...new Set(scenarioExecutionCombinations(scenarioResearchV4, {
        origin,
        decisionKind,
        proofMode,
      }).map((item)=>item.visibility))]
      : configuration.proofVisibilities,
    [configuration.proofVisibilities, decisionKind, origin, ortools, proofMode, scenarioResearchV4],
  );
  const availableClaims = useMemo(
    () => ortools
      ? [...new Set(scenarioExecutionCombinations(scenarioResearchV4, {
        origin,
        decisionKind,
        proofMode,
        visibility,
      }).flatMap((item)=>item.claims))]
      : configuration.proofClaims,
    [configuration.proofClaims, decisionKind, origin, ortools, proofMode, scenarioResearchV4, visibility],
  );
  const availableLineUniverses = useMemo(
    () => ortools
      ? [...new Set(scenarioExecutionCombinations(scenarioResearchV4, {
        origin,
        decisionKind,
        proofMode,
        visibility,
      }).flatMap((item)=>item.lineUniverses))]
      : configuration.proofLineUniverses,
    [configuration.proofLineUniverses, decisionKind, origin, ortools, proofMode, scenarioResearchV4, visibility],
  );
  const [claim, setClaim] = useState(defaultExecutionCombination?.claims[0] ?? authoringCatalog.proof.claims[0]!);
  const [targetTeam, setTargetTeam] = useState(targetTeamOptions[0]!);
  const [threshold, setThreshold] = useState(1);
  const [selectedAction, setSelectedAction] = useState('');
  const [budgetProfile, setBudgetProfile] = useState(String(fieldCatalog.fields['resources.proofBudgetProfile'].default));
  const [horizonUnit, setHorizonUnit] = useState(authoringCatalog.proof.horizonUnits[0]!);
  const [horizonValue, setHorizonValue] = useState(3);
  const [boundaryEvaluator, setBoundaryEvaluator] = useState(authoringCatalog.proof.boundaryEvaluators[0]!);
  const [policyProfiles, setPolicyProfiles] = useState<JsonObject>(authoringCatalog.policyCatalog.defaultProfiles);
  const [exactTraceRaw, setExactTraceRaw] = useState(()=>JSON.stringify(defaultExactTrace(authoringOptions), null, 2));
  const [exactTracePreview, setExactTracePreview] = useState<ExactTraceLegalActionsPreviewV1 | null>(null);
  const [exactTracePreviewAttempt, setExactTracePreviewAttempt] = useState(0);
  const [exactTracePreviewBusy, setExactTracePreviewBusy] = useState(false);
  const qualityExecution = jsonObject(authoringCatalog.playingQuality?.execution);
  const qualityModes = authoringCatalog.playingQuality?.modesByOrigin?.[origin]
    ?? (authoringCatalog.playingQuality?.modes ?? []).filter((mode)=>origin==='search'||mode==='budgeted');
  const [playingQuality, setPlayingQuality] = useState<JsonObject | null>(() =>
    authoringCatalog.playingQuality
      && researchCatalogInitial(fieldCatalog,'qualification.requireAllDecisionAudit',{}) === true
      && qualityModes.length
      ? buildPlayingQuality(qualityModes[0]!) : null);
  const [recordedEscalationOpportunities, setRecordedEscalationOpportunities] = useState<JsonObject[]>([]);
  const [qualification, setQualification] = useState<JsonObject>(() => ({
    minimumTruth: researchCatalogInitial(fieldCatalog,'qualification.minimumTruth',{}),
    minimumConfidence: researchCatalogInitial(fieldCatalog,'qualification.minimumConfidence',{}),
    minimumSurroundingQuality: researchCatalogInitial(fieldCatalog,'qualification.minimumSurroundingQuality',{}),
    requireObserverSafety: researchCatalogInitial(fieldCatalog,'qualification.requireObserverSafety',{}),
    requireAllDecisionAudit: researchCatalogInitial(fieldCatalog,'qualification.requireAllDecisionAudit',{}),
    maximumRegret: researchCatalogInitial(fieldCatalog,'qualification.maximumRegret',{}),
  }));
  const resourceFieldNames = authoredResourceFieldNames(fieldCatalog);
  const [resources, setResources] = useState<JsonObject>(() => Object.fromEntries(
    resourceFieldNames.map((field)=>[
      field,
      researchCatalogInitial(fieldCatalog,`resources.${field}`,{}),
    ]),
  ));
  const [diversity, setDiversity] = useState<JsonObject>(() => {
    return {schemaVersion: 1,
      cellDimensions: researchCatalogInitial(fieldCatalog,'diversity.cellDimensions',{}),
      topKPerCell: researchCatalogInitial(fieldCatalog,'diversity.topKPerCell',{}),
      rejectedAuditModulus: researchCatalogInitial(fieldCatalog,'diversity.rejectedAuditModulus',{}),
    };
  });
  const [outputKind, setOutputKind] = useState<'laffa' | 'laffa_collection'>(
    () => researchCatalogInitial(fieldCatalog,'output.kind',{}),
  );
  const [renderPolicy, setRenderPolicy] = useState<JsonObject>(() => ({
    mode: researchCatalogInitial(fieldCatalog,'renderPolicy.mode',{}),
    videoCount: researchCatalogInitial(fieldCatalog,'renderPolicy.videoCount',{}),
  }));
  const [recipeStatus, setRecipeStatus] = useState<'draft' | 'published' | 'retired'>(
    () => researchCatalogInitial(fieldCatalog,'status',{}),
  );
  const [desiredResultCount, setDesiredResultCount] = useState<number>(
    () => researchCatalogInitial(fieldCatalog,'output.resultCount',{}),
  );
  const [validation, setValidation] = useState<ResearchRecipeValidationV2 | null>(null);
  const [estimate, setEstimate] = useState<ResearchEstimate | null>(null);
  const [validatedFingerprint, setValidatedFingerprint] = useState('');
  const [estimatedFingerprint, setEstimatedFingerprint] = useState('');
  const [availableRecipes, setAvailableRecipes] = useState<ResearchRecipe[]>([]);
  const [selectedRecipeCode, setSelectedRecipeCode] = useState('');
  const [loadedSource, setLoadedSource] = useState('');
  const [loadingRecipe, setLoadingRecipe] = useState(false);
  const [conversionPreview, setConversionPreview] = useState<ResearchRecipeV4ConversionPreview | null>(null);
  const [renderBatch, setRenderBatch] = useState<JsonObject | null>(null);
  const renderIntentKeys = useRef(new Map<string, string>());
  const renderBatchRun = useRef('');
  const [activeRunCode, setActiveRunCode] = useState('');
  const initialRunSelected = useRef(false);
  const [resultFilters, setResultFilters] = useState<ResearchV4ResultFilters>({});
  const [lessonTagFilters, setLessonTagFilters] = useState<ResearchLessonTagQuery>({});
  const [generation, setGeneration] = useState<ResearchGenerationRequest | null>(null);
  const [auctionSeedError, setAuctionSeedError] = useState(false);
  const [resultGroupBy, setResultGroupBy] = useState<ResearchV4ResultFacetName|''>('');
  const [resultGroups, setResultGroups] = useState<ResearchResultPage['groups']>([]);
  const [studioResults, setStudioResults] = useState<ResearchResult[]>([]);
  const [studioResultTotal, setStudioResultTotal] = useState(0);
  const [resultBusy, setResultBusy] = useState(false);
  const fingerprintRef = useRef('');
  const studioRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (initialRunSelected.current || !runs.length) return;
    initialRunSelected.current = true;
    setActiveRunCode((current) => current || runs[0].runCode);
  }, [runs]);

  useEffect(() => {
    if (!selectedDecisionCapability && supportedDecisionKinds[0]) {
      setDecisionKind(supportedDecisionKinds[0].kind);
      if (supportedDecisionKinds[0].kind !== 'play') setGeneration(null);
    }
  }, [selectedDecisionCapability, supportedDecisionKinds]);
  useEffect(() => {
    if (!ortools) return;
    if (decisionKind === 'bid' || decisionKind === 'escalation') {
      setDecisionActionIndex(integerSchemaBounds(activeSelectorSchema,'actionIndex').default);
    }
    if (decisionKind === 'project') {
      setProjectIndex(integerSchemaBounds(activeSelectorSchema,'projectIndex').default);
    }
    if (decisionKind !== 'bid') {
      setDecisionSeat(ortools && decisionKind === 'play'
        && Array.isArray(activeSelectorSchema.required)
        && !activeSelectorSchema.required.includes('actingSeat') ? '' : stringSchemaDefault(
        activeSelectorSchema,
        decisionKind === 'project' ? 'owner' : 'actingSeat',
      ));
    }
    if (decisionKind === 'play') {
      setDecisionTrickIndex(integerSchemaBounds(activeSelectorSchema,'trickIndex').default);
      setDecisionPosition(undefined);
      setDecisionRemainingCards(integerSchemaBounds(activeSelectorSchema,'remainingCards').default);
    }
  }, [activeSelectorSchema, decisionKind, ortools]);
  useEffect(() => {
    const modes = ortools
      ? availableProofModes
      : selectedDecisionCapability?.proofModes ?? [];
    if (modes.length && !modes.includes(proofMode)) setProofMode(modes[0]);
    if (!ortools && decisionKind !== 'play' && visibility !== 'complete_state') {
      setVisibility('complete_state');
    }
    if (ortools && !availableVisibilities.includes(visibility)) {
      setVisibility(availableVisibilities[0] ?? 'complete_state');
    }
  }, [availableProofModes, availableVisibilities, decisionKind, ortools, proofMode,
    scenarioResearchV4, selectedDecisionCapability, visibility]);
  useEffect(() => {
    if (availableClaims.length && !availableClaims.includes(claim)) {
      setClaim(availableClaims[0]);
    }
    if (claim === 'winning_continuation_line_count'
      && availableLineUniverses.length
      && !availableLineUniverses.includes(lineUniverse)) {
      setLineUniverse(availableLineUniverses[0]);
    }
  }, [availableClaims, availableLineUniverses, claim, lineUniverse]);
  useEffect(() => {
    if (origin === 'exact_trace_prefix' || outputKind === 'laffa') {
      setOutputKind('laffa');
      setDesiredResultCount(Number(fieldCatalog.fields['output.resultCount'].minimum));
      setRenderPolicy((current)=>({
        ...current,
        videoCount: Math.min(Number(current.videoCount??0),Number(fieldCatalog.fields['output.resultCount'].minimum)),
      }));
    }
  }, [fieldCatalog.fields,origin,outputKind]);
  useEffect(() => {
    setRenderPolicy((current)=>{
      const maximum = Math.min(Number(fieldCatalog.fields['renderPolicy.videoCount'].maximum),desiredResultCount);
      const currentCount = Number(current.videoCount??0);
      return currentCount <= maximum ? current : {...current,videoCount:maximum};
    });
  },[desiredResultCount,fieldCatalog.fields]);

  useEffect(() => {
    let disposed = false;
    const prefix = parseExactTrace(exactTraceRaw);
    if (!ortools || origin !== 'exact_trace_prefix' || !prefix) {
      setExactTracePreview(null);
      setExactTracePreviewBusy(false);
      return () => { disposed = true; };
    }
    setExactTracePreview(null);
    setExactTracePreviewBusy(true);
    const timer = window.setTimeout(() => {
      request<ExactTraceLegalActionsPreviewV1>(
        '/api/v2/research-exact-traces:preview-legal-actions',
        {
          method: 'POST',
          body: JSON.stringify({
            schemaVersion: 1,
            configuration: {
              schemaVersion: 1,
              bundleRevisionId: selectedBundle.revisionId,
            },
            exactTracePrefix: prefix,
            ...(playingQuality ? {recordedEscalationOpportunities} : {}),
          }),
        },
      ).then((result) => {
        if (disposed) return;
        if (result.configurationAuthority.bundleRevisionId !== selectedBundle.revisionId) {
          setExactTracePreview(null);
          return;
        }
        setExactTracePreview(result);
      }).catch(() => {
        if (!disposed) setExactTracePreview(null);
      }).finally(() => {
        if (!disposed) setExactTracePreviewBusy(false);
      });
    }, 120);
    return () => {
      disposed = true;
      window.clearTimeout(timer);
    };
  }, [exactTraceRaw, exactTracePreviewAttempt, origin, ortools, playingQuality, recordedEscalationOpportunities, request, selectedBundle.revisionId]);

  useEffect(() => {
    let disposed = false;
    Promise.all(['draft','published','retired'].map((status) =>
      request<{items: ResearchRecipe[]; nextCursor: string | null}>(
        `/api/v2/research-recipes?status=${status}&limit=500`,
      ),
    )).then((pages) => {
      if (disposed) return;
      const unique = new Map(pages.flatMap((page) => page.items).map((item) => [item.code,item]));
      const items = [...unique.values()].filter((item) => {
        const version = Number(item.schemaVersion);
        const recipeBundleRevisionId = String(
          jsonObject(item.configuration).bundleRevisionId ?? '',
        );
        return version === 3 || (version === 4 && Boolean(recipeBundleRevisionId));
      }).sort((left,right) => left.code.localeCompare(right.code));
      setAvailableRecipes(items);
      setSelectedRecipeCode((current) => preferredScenarioRecipeCode(items, current, selectedBundle.revisionId));
    }).catch(() => {
      if (!disposed) setMessage(t('workflow.research_scenario_recipe_load_failed'));
    });
    return () => { disposed = true; };
  }, [request, selectedBundle.revisionId, setMessage]);

  const selectedDefinition = visibleDefinitions.find((definition) => definition.kind === selectedKind)
    ?? visibleDefinitions[0]
    ?? null;
  const authoredPredicates = [
    ...expressionPredicates(constraintExpression),
    ...expressionPredicates(outcomeExpression),
  ];
  const hasHardContinuationCount = authoredPredicates.some((item)=>
    item.kind === 'proof.winning_continuation_line_count' && item.hardness === 'hard');
  const selectedScenarioContract = authoredPredicates.find(
    (item)=>item.kind==='auction.contract',
  )?.arguments.contract;
  const scenarioContract = typeof selectedScenarioContract === 'string'
    ? selectedScenarioContract
    : undefined;
  const unsupportedPredicateIds = ortools
    ? unsupportedScenarioPredicateIds(
      allDefinitions,
      authoredPredicates,
      scenarioSelection,
      executionContractValues,
    )
    : [];
  const proofVisibilitySupported = !ortools || scenarioExecutionCombinations(
    scenarioResearchV4,
    {origin, decisionKind, proofMode, visibility},
  ).length > 0;
  const executionSupported = !ortools || scenarioExecutionSupported(
    scenarioResearchV4,
    {origin, decisionKind, proofMode, visibility},
    claim,
    lineUniverse,
  );
  const compiledStages = ortools
    ? scenarioCompiledStages(allDefinitions, authoredPredicates, origin)
    : [];
  const structuralInfeasibility = structuralInfeasibilityDiagnostic([
    ...(validation?.issues ?? []),
    ...(estimate?.issues ?? []),
  ]);
  const launchBlockReason = !ortools
    ? ''
    : structuralInfeasibility
      ? structuralInfeasibility.code
      : !authoringCatalog.playingQuality
      ? 'all_decision_quality_evidence_unavailable'
      : !scenarioResearchV4.executionReady
      ? 'ortools_stage_coverage_incomplete'
      : !scenarioResearchV4.launchAdmissible
        ? 'research_launch_not_admissible'
        : !proofVisibilitySupported
          ? 'proof_visibility_unsupported'
          : !executionSupported
            ? 'research_launch_not_admissible'
          : unsupportedPredicateIds.length
            ? 'predicate_cp_sat_unsupported'
            : '';
  const launchStatusBlocked = recipeStatus !== 'published';
  const updateDraft = (predicateId: string, change: Partial<PredicateDraft>) => {
    const update = (root: ExpressionGroupDraft) => {
      const candidate = normalizeExpression(
        updateExpression(root,predicateId,(item)=>isPredicateDraft(item)?{...item,...change}:item) as ExpressionGroupDraft,
        definitionMap,
      );
      return expressionWithinLimits(candidate,authoringCatalog.expressionLimits)
        ? candidate : root;
    };
    setConstraintExpression(update);
    setOutcomeExpression(update);
  };
  const groupAtTarget = expressionGroups(targetSection === 'constraints' ? constraintExpression : outcomeExpression)
    .find((group)=>group.nodeId===activeExpressionTarget);
  const targetRoot = targetSection === 'constraints' ? constraintExpression : outcomeExpression;
  const targetNodeCount = expressionNodeCount(targetRoot);
  const targetDepth = expressionGroupDepth(targetRoot,activeExpressionTarget);
  const targetAtCapacity = !groupAtTarget
    || (groupAtTarget.op === 'not' && groupAtTarget.children.length >= 1)
    || groupAtTarget.children.length >= authoringCatalog.expressionLimits.maximumChildren
    || targetNodeCount >= authoringCatalog.expressionLimits.maximumNodes;
  const addPredicate = () => {
    if (!selectedDefinition || targetAtCapacity) return;
    const argumentDefinition = selectedDefinition.kind === 'project.exists'
      ? {
          ...selectedDefinition,
          argumentSchema: projectExistsControlSchema(
            selectedDefinition.argumentSchema,
            configuration.projectOptions,
            scenarioContract,
            undefined,
          ),
        }
      : selectedDefinition;
    const draft: PredicateDraft = {
      predicateId: `PRED-${String(nextPredicate).padStart(4, '0')}`,
      kind: selectedDefinition.kind,
      version: 1,
      hardness: selectedDefinition.allowedHardness[0] ?? 'hard',
      arguments: initialArguments(argumentDefinition),
    };
    if (selectedDefinition.kind.startsWith('tactic.')) {
      draft.arguments.kind = (scenarioPredicateArgumentValues(
        selectedDefinition,scenarioSelection,'kind',
      ) ?? authoringCatalog.tacticKinds)[0] ?? null;
    }
    const append = (root: ExpressionGroupDraft) => {
      const candidate = normalizeExpression(updateExpression(root,activeExpressionTarget,(current) => isPredicateDraft(current)
        ? current
        : {...current,children:[...current.children,draft]}) as ExpressionGroupDraft,
      definitionMap,
      );
      return expressionSerializedBytes(candidate) <= authoringCatalog.expressionLimits.maximumSerializedBytes
        ? candidate : root;
    };
    if (targetSection === 'outcome') setOutcomeExpression(append);
    else setConstraintExpression(append);
    setNextPredicate((value) => value + 1);
  };
  const addExpressionGroup = (targetId: string) => {
    const root = expressionGroups(constraintExpression).some((group)=>group.nodeId===targetId)
      ? constraintExpression : outcomeExpression;
    const target = expressionGroups(root).find((group)=>group.nodeId===targetId);
    if (
      !target
      || target.children.length >= authoringCatalog.expressionLimits.maximumChildren
      || expressionNodeCount(root) >= authoringCatalog.expressionLimits.maximumNodes
      || expressionGroupDepth(root,targetId) >= authoringCatalog.expressionLimits.maximumDepth
    ) return;
    const child: ExpressionGroupDraft = {
      nodeId: `EXPR-${String(nextExpression).padStart(4,'0')}`,
      op: 'all', minimum: 1, maximum: 1, children: [],
    };
    const append = (currentRoot: ExpressionGroupDraft) => {
      const candidate = normalizeExpression(updateExpression(currentRoot,targetId,(current)=>isPredicateDraft(current)
        ? current
        : {...current,children:[...current.children,child]}) as ExpressionGroupDraft,
      definitionMap,
      );
      return expressionSerializedBytes(candidate) <= authoringCatalog.expressionLimits.maximumSerializedBytes
        ? candidate : currentRoot;
    };
    if (expressionGroups(constraintExpression).some((group)=>group.nodeId===targetId)) {
      setConstraintExpression(append);
      setTargetSection('constraints');
    } else {
      setOutcomeExpression(append);
      setTargetSection('outcome');
    }
    setActiveExpressionTarget(child.nodeId);
    setNextExpression((value)=>value+1);
  };
  const removeExpressionNode = (nodeId: string) => {
    setConstraintExpression((root)=>normalizeExpression(removeExpressionChild(root,nodeId),definitionMap));
    setOutcomeExpression((root)=>normalizeExpression(removeExpressionChild(root,nodeId),definitionMap));
    if (activeExpressionTarget === nodeId) {
      setActiveExpressionTarget(targetSection === 'constraints' ? 'EXPR-CONSTRAINTS' : 'EXPR-OUTCOME');
    }
  };
  const activateExpressionTarget = (nodeId: string, section: 'constraints' | 'outcome') => {
    setActiveExpressionTarget(nodeId);
    setTargetSection(section);
  };
  const executionStrategyClasses = ortools
    ? [...new Set(scenarioExecutionCombinations(scenarioResearchV4, {
      origin,
      decisionKind,
      proofMode,
      visibility,
    }).flatMap((item)=>item.strategyClasses))]
    : configuration.proofStrategyClasses;
  const preferredStrategyClass = proofMode === 'policy_conditioned'
    ? 'frozen_policy'
    : 'pure_deterministic';
  const strategyClass = executionStrategyClasses.includes(preferredStrategyClass)
    ? preferredStrategyClass
    : executionStrategyClasses[0];
  const proofModeHelp = proofMode === 'bounded'
    ? t('workflow.research_scenario_proof_bounded_help')
    : proofMode === 'policy_conditioned'
      ? t('workflow.research_scenario_proof_policy_conditioned_help')
      : t('workflow.research_scenario_proof_exact_help');
  const proofPayload = (): JsonObject => {
    const proof: JsonObject = {
      schemaVersion: 1,
      mode: proofMode,
      visibility,
      claim,
      targetTeam,
      utility,
      successThreshold: threshold,
      budgetProfile,
    };
    if (strategyClass) {
      proof.strategyClass = strategyClass;
    }
    if (claim === 'winning_continuation_line_count' || hasHardContinuationCount) proof.lineUniverse = lineUniverse;
    if (claim === 'selected_action_is_unique_winner'
      || ((claim === 'winning_continuation_line_count' || hasHardContinuationCount) && selectedAction)) {
      proof.selectedAction = selectedAction;
    }
    if (proofMode === 'bounded') {
      proof.horizon = {unit: horizonUnit, value: horizonValue};
      proof.boundaryEvaluator = boundaryEvaluator;
    }
    if (proofMode === 'policy_conditioned') proof.policies = policyProfiles;
    return proof;
  };
  const decisionPointPayload = (): JsonObject => {
    const selector: JsonObject = decisionKind === 'bid'
      ? {actionIndex: decisionActionIndex}
      : decisionKind === 'escalation'
        ? {actionIndex: decisionActionIndex, actingSeat: decisionSeat}
        : decisionKind === 'project'
          ? {owner: decisionSeat, projectIndex}
          : {
              trickIndex: decisionTrickIndex,
              ...(decisionPosition === undefined ? {} : {position: decisionPosition}),
              remainingCards: decisionRemainingCards,
              ...(decisionSeat ? {actingSeat: decisionSeat} : {}),
            };
    return {kind: decisionKind, selector};
  };
  const recipePayload = (): JsonObject => {
    if (!expressionPredicates(constraintExpression).length) {
      throw new Error('workflow.research_scenario_constraints_empty');
    }
    if (!expressionPredicates(outcomeExpression).length) {
      throw new Error('workflow.research_scenario_outcome_empty');
    }
    let scenarioOrigin: JsonObject = {kind: 'search'};
    if (origin === 'exact_trace_prefix') scenarioOrigin = {kind: origin, exactTracePrefix: parseJsonObject(exactTraceRaw),
      ...(playingQuality ? {recordedEscalationOpportunities} : {})};
    return {
      schemaVersion: 4,
      titleAr,
      output: {kind: outputKind, resultCount: desiredResultCount},
      configuration: {schemaVersion: 1, bundleRevisionId: selectedBundle.revisionId},
      scenario: {
        schemaVersion: 1,
        origin: scenarioOrigin,
        constraints: scenarioExpressionWire(constraintExpression),
        decisionPoint: decisionPointPayload(),
        outcome: scenarioExpressionWire(outcomeExpression),
      },
      proof: proofPayload(),
      ...(playingQuality ? {playingQuality} : {}),
      ...familySearchPayload(authoringCatalog, generation, {origin, decisionKind, playingQuality: playingQuality !== null}),
      qualification,
      resources: {
        ...Object.fromEntries(Object.entries(resources).filter(([,value])=>value !== null)),
        ...(ortools ? {
          generationLane: 'ortools',
          generationPolicy: 'ortools_cp_sat_v3',
        } : {
          generationPolicy: origin === 'search' ? generationPolicy : 'natural_policy',
        }),
        proofBudgetProfile: budgetProfile,
      },
      diversity,
      renderPolicy,
      status: recipeStatus,
    };
  };

  let payloadFingerprint = '';
  try {
    payloadFingerprint = JSON.stringify(recipePayload());
  } catch {
    payloadFingerprint = JSON.stringify({
      titleAr, origin, exactTraceRaw, playingQuality, ...(generation ? {generation} : {}), recordedEscalationOpportunities, constraintExpression, outcomeExpression,
      decisionKind, decisionSeat, decisionActionIndex, decisionTrickIndex,
      decisionPosition,
      decisionRemainingCards,
      projectIndex, proofMode, visibility, claim, targetTeam, utility, threshold,
      lineUniverse, selectedAction, budgetProfile, horizonUnit, horizonValue,
      boundaryEvaluator, policyProfiles, outputKind, desiredResultCount,
      qualification, resources, diversity,
      renderPolicy, recipeStatus, bundleRevisionId:selectedBundle.revisionId,
    });
  }
  fingerprintRef.current = payloadFingerprint;
  useEffect(() => {
    setValidation(null);
    setEstimate(null);
    setValidatedFingerprint('');
    setEstimatedFingerprint('');
  }, [payloadFingerprint]);

  const runRequest = async (action: 'validate' | 'estimate' | 'run') => {
    const invalidControl = [...(studioRef.current?.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(
      'input:invalid, select:invalid, textarea:invalid',
    ) ?? [])].find((control) => !control.closest('[data-research-opening-draft]'));
    if (invalidControl) {
      invalidControl.reportValidity();
      return;
    }
    if (action === 'run' && (launchBlockReason || launchStatusBlocked)) {
      setMessage(launchStatusBlocked
        ? `${catalogFieldLabel(authoringCatalog,'status')} → ${t('workflow.research_scenario_status_published')}`
        : t('workflow.research_scenario_estimate_refused'));
      return;
    }
    setBusy(true);
    const submittedFingerprint = fingerprintRef.current;
    const requestIsCurrent = () => Boolean(studioRef.current)
      && fingerprintRef.current === submittedFingerprint;
    try {
      const payload = recipePayload();
      if (action === 'validate') {
        const result = await request<ResearchRecipeValidationV2>('/api/v2/research-recipes:validate', {method: 'POST', body: JSON.stringify(payload)});
        if (!requestIsCurrent()) return;
        setValidation(result);
        setValidatedFingerprint(submittedFingerprint);
        setMessage(result.valid ? t('workflow.research_validation_ok') : t('workflow.research_validation_failed'));
      } else if (action === 'estimate') {
        if (!validation?.valid || validatedFingerprint !== submittedFingerprint) {
          throw new Error('scenario_validation_required');
        }
        const result = await request<ResearchEstimate>('/api/v2/research-recipes:estimate', {method: 'POST', body: JSON.stringify(payload)});
        if (!requestIsCurrent()) return;
        setEstimate(result);
        setEstimatedFingerprint(submittedFingerprint);
        setMessage(result.valid ? t('workflow.research_estimate_title') : t('workflow.research_validation_failed'));
      } else {
        if (!validation?.valid || validatedFingerprint !== submittedFingerprint
          || !estimate?.valid || estimate.launchAdmissible !== true
          || estimatedFingerprint !== submittedFingerprint) {
          throw new Error('scenario_estimate_required');
        }
        const launchEstimate = await request<ResearchEstimate>('/api/v2/research-recipes:estimate', {method: 'POST', body: JSON.stringify(payload)});
        if (!requestIsCurrent()) return;
        setEstimate(launchEstimate);
        setEstimatedFingerprint(submittedFingerprint);
        if (!launchEstimate.valid || launchEstimate.launchAdmissible !== true) {
          setMessage(launchEstimate.output?.reason ?? launchEstimate.proof?.reason ?? t('workflow.research_scenario_estimate_refused'));
          return;
        }
        const {run} = await createRecipeAndRun(payload);
        if (!requestIsCurrent()) return;
        setActiveRunCode(run.runCode);
        await refresh();
        if (!requestIsCurrent()) return;
        const openedRun = await request<ResearchRun>(`/api/v2/research-runs/${encodeURIComponent(run.runCode)}`);
        if (!requestIsCurrent()) return;
        onOpenRun(openedRun);
      }
    } catch (error) {
      if (!requestIsCurrent()) return;
      const key = error instanceof Error ? error.message : '';
      if (key === 'workflow.research_lesson_tag_search_invalid' && hasFoundryCopy('workflow.research_lesson_tag_search_invalid')) {
        setMessage(t('workflow.research_lesson_tag_search_invalid'));
      } else setMessage(t(key === 'workflow.research_scenario_constraints_empty'
        || key === 'workflow.research_scenario_outcome_empty'
        || key.startsWith('workflow.research_lesson_tag_search_') && hasFoundryCopy(key)
        || key === 'workflow.research_lesson_tag_generation_requires_play'
        ? key : 'workflow.research_scenario_request_failed'));
    } finally {
      setBusy(false);
    }
  };

  const loadRecipe = async (
    mode: 'load' | 'clone' | 'variation',
    suppliedDocument?: ResearchRecipeV4Document,
  ) => {
    if (!selectedRecipeCode && !suppliedDocument) return;
    setLoadingRecipe(true);
    try {
      const document = suppliedDocument ?? await request<ResearchRecipeV4Document & ResearchRecipe>(
        `/api/v2/research-recipes/${encodeURIComponent(selectedRecipeCode)}`,
      );
      if (Number(document.schemaVersion) !== 4) {
        setMessage(t('workflow.research_scenario_recipe_v4_only'));
        return;
      }
      const authority = jsonObject(document.configuration);
      if (mode === 'load' && String(authority.bundleRevisionId ?? '') !== selectedBundle.revisionId) {
        setMessage(t('workflow.research_scenario_recipe_bundle_mismatch'));
        return;
      }
      const scenario = jsonObject(document.scenario);
      const scenarioOrigin = jsonObject(scenario.origin);
      const decision = jsonObject(scenario.decisionPoint);
      const selector = jsonObject(decision.selector);
      const proof = jsonObject(document.proof);
      const nextGroup = {value: 1};
      const constraints = hydrateExpression(scenario.constraints,'EXPR-CONSTRAINTS',nextGroup);
      const outcome = hydrateExpression(scenario.outcome,'EXPR-OUTCOME',nextGroup);
      const loadedOrigin = catalogChoice(
        scenarioOrigin.kind,
        supportedOrigins,
        supportedOrigins[0]!,
        'scenario.origin.kind',
      );
      const loadedDecisionCapabilities = configuration.decisionCapabilities.filter((item)=>
        item.authoringSupported
        && (loadedOrigin === 'search' ? item.searchGenerationSupported : item.exactTraceNormalizationSupported)
        && (!ortools || scenarioExecutionCombinations(scenarioResearchV4, {
          origin: loadedOrigin,
          decisionKind: item.kind,
        }).length > 0));
      const loadedDecision = catalogChoice(
        decision.kind,
        loadedDecisionCapabilities.map((item)=>item.kind),
        loadedDecisionCapabilities[0]!.kind,
        'scenario.decisionPoint.kind',
      );
      const loadedDecisionCapability = loadedDecisionCapabilities.find((item)=>item.kind===loadedDecision);
      const loadedProofMode = catalogChoice(
        proof.mode,
        loadedDecisionCapability?.proofModes ?? authoringCatalog.proof.modes,
        (loadedDecisionCapability?.proofModes ?? authoringCatalog.proof.modes)[0]!,
        'proof.mode',
      );
      const loadedExecutionCombinations = ortools
        ? scenarioExecutionCombinations(scenarioResearchV4, {
            origin: loadedOrigin,
            decisionKind: loadedDecision,
            proofMode: loadedProofMode,
          })
        : [];
      const loadedVisibilities = ortools
        ? [...new Set(loadedExecutionCombinations.map((item)=>item.visibility))]
        : authoringCatalog.proof.visibilities;
      const loadedClaims = ortools
        ? [...new Set(loadedExecutionCombinations.flatMap((item)=>item.claims))]
        : authoringCatalog.proof.claims;
      const loadedLineUniverses = ortools
        ? [...new Set(loadedExecutionCombinations.flatMap((item)=>item.lineUniverses))]
        : authoringCatalog.proof.lineUniverses;
      const loadedGeneration = readFamilySearchRequest(authoringCatalog, document.generation);
      familySearchPayload(authoringCatalog, loadedGeneration, {origin: loadedOrigin,
        decisionKind: loadedDecision, playingQuality: Boolean(document.playingQuality)});
      setTitleAr(mode === 'clone'
        ? `${document.titleAr} ${t('workflow.research_scenario_recipe_clone_suffix')}`
        : mode === 'variation'
          ? `${document.titleAr} ${t('workflow.research_scenario_recipe_variation_suffix')}`
          : document.titleAr);
      const loadedOutput = jsonObject(document.output);
      setOutputKind(catalogChoice(loadedOutput.kind,catalogValues(fieldCatalog,'output.kind') as Array<typeof outputKind>,'laffa','output.kind'));
      setDesiredResultCount(Number(loadedOutput.resultCount??fieldCatalog.fields['output.resultCount'].default));
      setOrigin(loadedOrigin);
      setExactTraceRaw(JSON.stringify(scenarioOrigin.exactTracePrefix ?? defaultExactTrace(authoringOptions),null,2));
      setRecordedEscalationOpportunities(Array.isArray(scenarioOrigin.recordedEscalationOpportunities)
        ? scenarioOrigin.recordedEscalationOpportunities.map(jsonObject) : []);
      setConstraintExpression(constraints);
      setOutcomeExpression(outcome);
      setNextExpression(nextGroup.value);
      const predicates = [...expressionPredicates(constraints),...expressionPredicates(outcome)];
      setNextPredicate(Math.max(1,...predicates.map((item)=>Number(item.predicateId.match(/\d+$/)?.[0] ?? 0)))+1);
      setActiveExpressionTarget('EXPR-CONSTRAINTS');
      setTargetSection('constraints');
      setDecisionKind(loadedDecision);
      const loadedSelectorSchema = scenarioDecisionSelectorSchema(scenarioResearchV4,loadedDecision);
      const loadedSeatField = loadedDecision === 'project' ? 'owner' : 'actingSeat';
      const loadedSeatOptions = stringSchemaValues(
        loadedSelectorSchema,
        loadedSeatField,
        ortools ? [] : authoringOptions.seats,
      );
      if (loadedDecision !== 'bid') {
        setDecisionSeat(loadedDecision === 'play' && selector[loadedSeatField] === undefined ? '' : catalogChoice(
          selector[loadedSeatField],
          loadedSeatOptions,
          stringSchemaDefault(
            loadedSelectorSchema,
            loadedSeatField,
            ortools ? undefined : loadedSeatOptions[0],
          ),
          `scenario.decisionPoint.selector.${loadedSeatField}`,
        ));
      }
      setDecisionActionIndex(Number(selector.actionIndex ?? (
        loadedDecision === 'bid' || loadedDecision === 'escalation'
          ? integerSchemaBounds(
              loadedSelectorSchema,
              'actionIndex',
              ortools ? undefined : {minimum:0,maximum:10_000,default:0},
            ).default
          : 0
      )));
      setDecisionTrickIndex(Number(selector.trickIndex ?? playTrickBounds.default));
      setDecisionPosition(selector.position === undefined ? undefined : Number(selector.position));
      setDecisionRemainingCards(Number(selector.remainingCards ?? playRemainingBounds.default));
      setProjectIndex(Number(selector.projectIndex ?? (loadedDecision === 'project'
        ? integerSchemaBounds(
            loadedSelectorSchema,
            'projectIndex',
            ortools ? undefined : {minimum:0,maximum:10_000,default:0},
          ).default
        : 0)));
      setProofMode(loadedProofMode);
      setVisibility(catalogChoice(proof.visibility,loadedVisibilities,loadedVisibilities[0]!,'proof.visibility'));
      setClaim(catalogChoice(proof.claim,loadedClaims,loadedClaims[0]!,'proof.claim'));
      setTargetTeam(String(proof.targetTeam ?? targetTeam));
      setUtility(catalogChoice(proof.utility,authoringCatalog.proof.utilities,authoringCatalog.proof.utilities[0]!,'proof.utility'));
      setThreshold(Number(proof.successThreshold ?? 1));
      if (loadedLineUniverses.length) {
        setLineUniverse(catalogChoice(
          proof.lineUniverse,
          loadedLineUniverses,
          loadedLineUniverses[0]!,
          'proof.lineUniverse',
        ));
      }
      setSelectedAction(String(proof.selectedAction ?? ''));
      setBudgetProfile(catalogChoice(proof.budgetProfile,Object.keys(authoringCatalog.proof.budgetProfiles),budgetProfile,'proof.budgetProfile'));
      const horizon = jsonObject(proof.horizon);
      setHorizonUnit(catalogChoice(horizon.unit,authoringCatalog.proof.horizonUnits,authoringCatalog.proof.horizonUnits[0]!,'proof.horizon.unit'));
      setHorizonValue(Number(horizon.value ?? 3));
      setBoundaryEvaluator(catalogChoice(proof.boundaryEvaluator,authoringCatalog.proof.boundaryEvaluators,authoringCatalog.proof.boundaryEvaluators[0]!,'proof.boundaryEvaluator'));
      const loadedPolicies = jsonObject(proof.policies);
      setPolicyProfiles(Object.keys(loadedPolicies).length ? loadedPolicies : authoringCatalog.policyCatalog.defaultProfiles);
      setQualification(jsonObject(document.qualification));
      setPlayingQuality(document.playingQuality ? jsonObject(document.playingQuality) : null);
      setGeneration(loadedGeneration);
      const loadedResources = jsonObject(document.resources);
      setGenerationPolicy(String(loadedResources.generationPolicy) === 'scenario_guided_actor_safe' ? 'scenario_guided_actor_safe' : 'natural_policy');
      setResources(Object.fromEntries(resourceFieldNames.map((field)=>[
        field,
        loadedResources[field] ?? researchCatalogInitial(
          fieldCatalog,
          `resources.${field}`,
          {},
        ),
      ])));
      setDiversity(jsonObject(document.diversity));
      setRenderPolicy(jsonObject(document.renderPolicy));
      setRecipeStatus(suppliedDocument || mode !== 'load'
        ? 'draft'
        : catalogChoice(document.status,catalogValues(fieldCatalog,'status') as Array<typeof recipeStatus>,'draft','status'));
      const identity = suppliedDocument
        ? `${conversionPreview?.targetRecipeCode ?? ''} · ${conversionPreview?.targetRecipeHash ?? conversionPreview?.sha256 ?? ''}`
        : `${document.code} · ${document.sha256}`;
      setLoadedSource(identity);
      setMessage(t('workflow.research_scenario_recipe_loaded'));
    } catch (error) {
      const key = error instanceof Error ? error.message : '';
      if (key === 'workflow.research_lesson_tag_search_invalid' && hasFoundryCopy('workflow.research_lesson_tag_search_invalid')) {
        setMessage(t('workflow.research_lesson_tag_search_invalid'));
      } else setMessage(t(key.startsWith('workflow.research_lesson_tag_search_') && hasFoundryCopy(key)
        || key === 'workflow.research_lesson_tag_generation_requires_play'
        ? key : 'workflow.research_scenario_recipe_load_failed'));
    } finally {
      setLoadingRecipe(false);
    }
  };
  const previewLegacyRecipe = async (includeOperatorInputs = false) => {
    if (!selectedRecipeCode) return;
    setLoadingRecipe(true);
    try {
      const preview = await request<ResearchRecipeV4ConversionPreview>(
        `/api/v2/research-recipes/${encodeURIComponent(selectedRecipeCode)}:preview-v4-conversion`,
        {
          method:'POST',
          body: JSON.stringify(includeOperatorInputs ? {
            decisionPoint: decisionPointPayload(),
            outcome: scenarioExpressionWire(outcomeExpression),
            proof: proofPayload(),
          } : {}),
        },
      );
      setConversionPreview(preview);
      setMessage(t('workflow.research_scenario_conversion_ready'));
    } catch {
      setMessage(t('workflow.research_scenario_conversion_failed'));
    } finally {
      setLoadingRecipe(false);
    }
  };
  const prepareSelectedRun = async () => {
    if (!activeRunCode || Number(renderPolicy.videoCount??0) < 1) return;
    setBusy(true);
    try {
      const baseKey = `scenario-${activeRunCode}-render-selected-v1`;
      const intentKey = renderIntentKeys.current.get(activeRunCode) ?? baseKey;
      const job = await request<RenderJob>(
        `/api/v2/research-runs/${encodeURIComponent(activeRunCode)}:render-selected`,
        {
          method:'POST',
          headers:{'Idempotency-Key':intentKey},
          body:'{}',
        },
      );
      renderIntentKeys.current.set(activeRunCode, ['failed','cancelled'].includes(job.status)
        ? renderRetryKey(baseKey, job) : intentKey);
      renderBatchRun.current = activeRunCode;
      setRenderBatch(job as unknown as JsonObject);
      setMessage(t(job.status === 'completed' ? 'workflow.batch_completed_help'
        : ['failed','cancelled'].includes(job.status) ? 'workflow.research_scenario_render_batch_failed'
          : 'workflow.research_scenario_render_batch_started'));
      await refresh();
    } catch {
      setMessage(t('workflow.research_scenario_render_batch_failed'));
    } finally {
      setBusy(false);
    }
  };
  const cancelRenderBatch = async () => {
    if (!renderBatchJobId || !renderBatchActive) return;
    setBusy(true);
    try {
      const job = await request<RenderJob>(
        `/api/v2/render-batches/${encodeURIComponent(renderBatchJobId)}:cancel`,
        {method:'POST'},
      );
      if (renderBatchRun.current && ['failed','cancelled'].includes(job.status)) {
        const runCode = renderBatchRun.current;
        renderIntentKeys.current.set(runCode, renderRetryKey(`scenario-${runCode}-render-selected-v1`, job));
      }
      setRenderBatch(job as unknown as JsonObject);
      setMessage(t('workflow.batch_cancel_requested'));
    } catch {
      setMessage(t('workflow.research_scenario_render_batch_failed'));
    } finally {
      setBusy(false);
    }
  };

  const renderPredicate = (draft: PredicateDraft) => {
    const definition = definitionMap.get(draft.kind);
    if (!definition) {
      const titleId = `scenario-${draft.predicateId}-title`;
      const statusId = `scenario-${draft.predicateId}-status`;
      return <article
        className="scenario-predicate unsupported"
        key={draft.predicateId}
        id={`scenario-${draft.predicateId}`}
        data-field-path={draft.predicateId}
        aria-invalid="true"
        aria-labelledby={titleId}
        aria-describedby={statusId}
      >
        <header><div><h4 id={titleId}>{t('workflow.research_scenario_capability_unsupported')}</h4><code dir="ltr">{draft.predicateId} · {draft.kind}</code></div><span className="scenario-capability unsupported">{t('workflow.research_scenario_capability_unsupported')}</span><button type="button" className="quiet" onClick={()=>removeExpressionNode(draft.predicateId)}>{t('workflow.research_scenario_remove')}</button></header>
        <p id={statusId} className="scenario-launch-block" role="status" aria-live="polite" aria-atomic="true"><strong>{tCatalog(scenarioLaunchBlockMessageKey('predicate_cp_sat_unsupported'))}</strong> <code dir="ltr">predicate_cp_sat_unsupported</code></p>
        <div className="scenario-cp-sat-support"><span>{t('workflow.research_scenario_hardness')}: <code dir="ltr">{draft.hardness}</code></span><span>{t('workflow.research_scenario_predicate_metadata')}</span></div>
        <pre dir="ltr">{JSON.stringify(draft.arguments,null,2)}</pre>
      </article>;
    }
    const argumentSchema = draft.kind === 'project.exists'
      ? projectExistsControlSchema(
          definition.argumentSchema,
          configuration.projectOptions,
          scenarioContract,
          draft.arguments.kind ?? null,
        )
      : definition.argumentSchema;
    const schema = schemaProperties({...definition, argumentSchema});
    const properties = Object.fromEntries(Object.entries(schema.properties).map(([name,value])=>[name,{...value}]));
    const required = schema.required;
    if (!ortools && Array.isArray(properties.contract?.enum)) {
      properties.contract = {...properties.contract,enum:properties.contract.enum.filter((value)=>configuration.ruleOptions.contracts.includes(String(value) as 'sun'|'hokum'|'ashkal'))};
    }
    if (!ortools && draft.kind === 'auction.ashkal_buyer_relation' && Array.isArray(properties.relation?.enum)) {
      properties.relation = {...properties.relation,enum:properties.relation.enum.filter((value)=>configuration.ruleOptions.ashkalBuyerRelations.includes(String(value) as 'bidder'|'bidder_partner'))};
    }
    if (!ortools && draft.kind === 'escalation.level' && Array.isArray(properties.level?.enum)) {
      const activeLevels = new Set(configuration.ruleOptions.escalationLevelsByContract.flatMap((item)=>item.levels));
      properties.level = {...properties.level,enum:properties.level.enum.filter((value)=>activeLevels.has(String(value) as 'normal'|'double'|'triple'|'four'|'gahwa'))};
    }
    if (!ortools && !configuration.ruleOptions.playActionOptions.some((action)=>action.ekka) && draft.kind === 'play.current_trick_card') delete properties.ekka;
    if (draft.kind.startsWith('tactic.')) properties.kind = {
      ...properties.kind,
      enum: scenarioPredicateArgumentValues(definition,scenarioSelection,'kind')
        ?? authoringCatalog.tacticKinds,
    };
    if (draft.kind === 'tactic.verified' && draft.hardness === 'hard'
      && !scenarioPredicateCausalVerificationAvailable(definition,draft.arguments)) {
      properties.causalRequired = {...properties.causalRequired,enum:[false]};
    }
    const supported = !ortools || scenarioPredicateSupported(
      definition,
      scenarioSelection,
      draft.hardness,
      draft.arguments,
    );
    const hardnessOptions = ortools
      ? definition.allowedHardness.filter((hardness)=>scenarioPredicateSupported(
        definition,
        scenarioSelection,
        hardness,
        draft.arguments,
      ))
      : definition.allowedHardness;
    const capability = definition.cpSatSupport;
    const searchStatus = origin==='search'?capability?.search.status:capability?.exactInput.status;
    const proofStatus = proofMode==='policy_conditioned'?capability?.proof.policyConditioned:capability?.proof[proofMode];
    const help = ortools ? predicateHelp(authoringCatalog,definition) : null;
    return <article className={`scenario-predicate${supported?'':' unsupported'}`} key={draft.predicateId} id={`scenario-${draft.predicateId}`} data-field-path={draft.predicateId}><header><div><h4>{predicateLabel(definition)}</h4><code dir="ltr">{draft.predicateId} · {draft.kind}</code>{help?<small>{help}</small>:null}</div>{ortools?<span className={`scenario-capability ${supported?'cp_sat_exact':'unsupported'}`}>{catalogValueLabel(authoringCatalog,supported?definition.cpSatSupport?.qualificationSource??'unsupported':'unsupported')}</span>:<span className={`scenario-capability ${definition.constructionCapability}`}>{capabilityLabel(definition.constructionCapability)}</span>}<button type="button" className="quiet" onClick={()=>removeExpressionNode(draft.predicateId)}>{t('workflow.research_scenario_remove')}</button></header>{ortools&&capability&&searchStatus&&proofStatus?<div className="scenario-cp-sat-support"><span>{catalogValueLabel(authoringCatalog,origin)}: {catalogValueLabel(authoringCatalog,searchStatus)}</span><span>{catalogValueLabel(authoringCatalog,proofMode)}: {catalogValueLabel(authoringCatalog,proofStatus)}</span></div>:null}{ortools?<PredicateCatalogMetadata definition={definition} catalog={authoringCatalog}/>:null}<label className="field-label" data-field-path={`${draft.predicateId}.hardness`}><span>{t('workflow.research_scenario_hardness')}</span><select value={draft.hardness} aria-invalid={!supported||undefined} onChange={(event)=>updateDraft(draft.predicateId,{hardness:event.target.value as PredicateDraft['hardness']})}>{(hardnessOptions.includes(draft.hardness)?hardnessOptions:[draft.hardness,...hardnessOptions]).map((value)=><option value={value} key={value} disabled={ortools&&!scenarioPredicateSupported(definition,scenarioSelection,value,draft.arguments)}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label><div className="scenario-arguments">{Object.entries(properties).map(([name, schema])=>{const fieldPath=`${draft.predicateId}.arguments.${name}`;return <div data-field-path={fieldPath} key={name}><ArgumentField name={name} schema={schema} required={required.has(name)} value={draft.arguments[name]} contextKind={draft.kind} options={authoringOptions} catalog={authoringCatalog} fieldPath={fieldPath} errors={predicateArgumentErrors(validation,draft.predicateId,name)} onChange={(value)=>{const argumentsValue={...draft.arguments};if(value === undefined) delete argumentsValue[name];else argumentsValue[name]=value;if(draft.kind==='project.exists'&&name==='kind'){if(value===undefined)delete argumentsValue.value;else{const projectSchema=projectExistsControlSchema(definition.argumentSchema,configuration.projectOptions,scenarioContract,value);const allowedValues=jsonObject(jsonObject(projectSchema.properties).value).enum;if(Array.isArray(allowedValues)&&!allowedValues.includes(argumentsValue.value))argumentsValue.value=allowedValues[0];}}updateDraft(draft.predicateId,{arguments:argumentsValue});}}/></div>;})}</div></article>;
  };
  const normalizedLegalActions = validation?.normalizedExactState?.legalActions ?? [];
  const derivedActions = normalizedLegalActions.map((action)=>String(action.code ?? '')).filter(Boolean);
  const fallbackActions = decisionKind === 'play'
    ? configuration.ruleOptions.playActionOptions.map((action)=>action.code)
    : decisionKind === 'bid'
      ? authoringOptions.auctionActions.flatMap((action)=>action==='hokum'
        ? authoringOptions.suits.map((suit)=>`hokum:${suit}`)
        : [action])
      : decisionKind === 'escalation'
        ? ['decline',...authoringOptions.escalations.filter((value)=>value!=='normal')]
        : authoringOptions.projects.map((item)=>`declare:${item.kind}:${item.value}:${item.highRankId}`);
  const selectedActionOptions = [...new Set(derivedActions.length?derivedActions:fallbackActions)];
  const selectedActionChoices = selectedActionOptions.join('\u0000');
  const selectedActionDiagnostic = validation?.issues.find((issue)=>issue.fieldPaths.includes('proof.selectedAction'));
  useEffect(() => {
    if (selectedAction && !selectedActionOptions.includes(selectedAction)) {
      setSelectedAction('');
    }
  }, [selectedAction, selectedActionChoices]);
  const playActionLabels = new Map<string,string>();
  configuration.ruleOptions.playActionOptions.forEach((action)=>{
    playActionLabels.set(action.code,`${action.card.label}${action.ekka?` · ${t('workflow.research_scenario_action_ekka')}`:''}`);
  });
  normalizedLegalActions.forEach((action)=>{
    const code=String(action.code??'');
    const card=action.card&&typeof action.card==='object'&&!Array.isArray(action.card)?action.card as JsonObject:null;
    if(code&&card) playActionLabels.set(code,`${String(card.label??code)}${action.ekka===true?` · ${t('workflow.research_scenario_action_ekka')}`:''}`);
  });
  const filterFingerprint = JSON.stringify({resultFilters, lessonTagFilters});
  const lessonTagCatalog = authoringCatalog.lessonTags;
  const familyGeneration = frozenFamilyGeneration(authoringCatalog);
  const tagControlsReady = lessonTagCatalogReady(lessonTagCatalog, (key) => hasFoundryCopy(`workflow.${key}`))
    && hasFoundryCopy('workflow.research_lesson_tags_title')
    && hasFoundryCopy('workflow.research_lesson_tags_help');
  const familyControlsReady = familyGeneration && tagControlsReady
    && ['title', 'help', 'apply', 'requires_play'].every((key) => hasFoundryCopy(`workflow.research_lesson_tag_generation_${key}`));
  const generationTagCatalog = familySearchTagCatalog(authoringCatalog, generation?.tagQuery);
  const generationControlsReady = generationTagCatalog && tagControlsReady
    && ['title', 'help', 'enable', 'invalid', 'no_match', 'matching', 'scope_cleared']
      .every((key) => hasFoundryCopy(`workflow.research_lesson_tag_search_${key}`));
  const generationScopeReady = origin === 'search' && decisionKind === 'play' && playingQuality !== null;
  const generationMatches = generation && generationScopeReady
    ? matchingSearchFamilies(authoringCatalog, generation.tagQuery, jsonObject(decisionPointPayload().selector)) : [];
  const auctionSeedLimit = familySearchAuctionSeedLimit(authoringCatalog);
  const auctionSeedControlsReady = auctionSeedLimit !== null && ['title', 'help', 'empty', 'limit']
    .every((key) => hasFoundryCopy(`workflow.research_auction_seeds_${key}`));
  const auctionOpenings = familySearchAuctionOpenings(authoringCatalog, runs, selectedBundle.revisionId, generation?.auctionSeeds);
  const auctionSeedProperties = jsonObject(jsonObject(jsonObject(
    jsonObject(authoringCatalog.familySearch?.requestSchema.properties).auctionSeeds).items).properties);
  const auctionSeedSchema = jsonObject(auctionSeedProperties.seed);
  const auctionDealerValues = jsonObject(auctionSeedProperties.dealer).enum;
  const manualAuctionSeedReady = auctionSeedControlsReady && Array.isArray(auctionDealerValues)
    && ['add', 'invalid'].every((key) => hasFoundryCopy(`workflow.research_auction_seeds_${key}`));
  const clearGeneration = () => {
    if (generation) {
      setGeneration(null);
      setMessage(t('workflow.research_lesson_tag_search_scope_cleared'));
    }
  };
  const activeRun = runs.find((run)=>run.runCode===activeRunCode);
  const activeRunStatus = activeRun?.status ?? '';
  const activeRunProgress = jsonObject(activeRun?.campaignAccounting ?? activeRun?.campaignProgress);
  const currentSolverCounters = jsonObject(activeRunProgress.currentSolverCounters);
  const qualitySolver = jsonObject(activeRunProgress.qualitySolver);
  const qualitySolverPhase = String(qualitySolver.phase ?? '');
  const qualitySolverPhaseKeys: Record<string, string> = {
    preparation: 'workflow.research_solver_progress_preparation',
    master: 'workflow.research_solver_progress_master',
    counterexample: 'workflow.research_solver_progress_counterexample',
    binding: 'workflow.research_solver_progress_binding',
  };
  const qualitySolverPhaseKey = qualitySolverPhaseKeys[qualitySolverPhase];
  const qualitySolverCopyReady = Boolean(qualitySolverPhaseKey && [
    qualitySolverPhaseKey,
    'workflow.research_solver_progress_masters',
    'workflow.research_solver_progress_queries',
    'workflow.research_solver_progress_bindings',
    'workflow.research_solver_progress_unproved',
  ].every(hasFoundryCopy));
  const qualitySolverPhaseLabel = !qualitySolverCopyReady ? null
    : qualitySolverPhase === 'preparation' ? t('workflow.research_solver_progress_preparation')
    : qualitySolverPhase === 'master' ? t('workflow.research_solver_progress_master')
    : qualitySolverPhase === 'counterexample' ? t('workflow.research_solver_progress_counterexample')
    : t('workflow.research_solver_progress_binding');
  const provisionalResults = Array.isArray(activeRunProgress.provisionalResults)
    ? activeRunProgress.provisionalResults.filter((item): item is JsonObject => (
        Boolean(item) && typeof item === 'object' && !Array.isArray(item)
      ))
    : [];
  const candidateReceipts = Array.isArray(activeRunProgress.candidateReceipts)
    ? activeRunProgress.candidateReceipts.filter((item): item is ResearchCandidateProgressReceipt => (
        Boolean(item) && typeof item === 'object' && !Array.isArray(item)
      )).slice(-10)
    : [];
  const candidateReceiptCounts = jsonObject(activeRunProgress.candidateReceiptCounts);
  const terminalRunRefresh = ['completed','failed','cancelled'].includes(activeRunStatus)
    ? `${activeRunCode}:${activeRunStatus}`
    : '';
  const renderBatchJobId = String(renderBatch?.jobId??'');
  const renderBatchStatus = String(renderBatch?.status??'');
  const renderBatchActive = ['queued','running'].includes(renderBatchStatus);
  const renderBatchTerminal = ['completed','failed','cancelled'].includes(renderBatchStatus);
  const renderJob = (renderBatch??{}) as unknown as RenderJob;
  const renderedMedia = renderBatchMediaEntries(renderJob);
  const preparedPlayers = renderBatchPlayerEntries(renderJob);
  const renderProgress = jsonObject(jsonObject(renderBatch?.result).itemProgress);
  useEffect(() => {
    if (!renderBatchJobId || !renderBatchActive) return;
    const runCode = renderBatchRun.current;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const current = await request<RenderJob>(
          `/api/v2/render-batches/${encodeURIComponent(renderBatchJobId)}`,
          {signal: controller.signal},
        );
        if (controller.signal.aborted) return;
        setRenderBatch(current as unknown as JsonObject);
        if (runCode && ['failed','cancelled'].includes(current.status)) {
          renderIntentKeys.current.set(runCode, renderRetryKey(`scenario-${runCode}-render-selected-v1`, current));
        }
        if (!['queued','running'].includes(current.status)) {
          setMessage(t(current.status==='completed'
            ? 'workflow.batch_completed_help'
            : current.status==='cancelled'
              ? 'workflow.laffa_library_render_cancelled'
              : 'workflow.research_scenario_render_batch_failed'));
          await refresh();
          return;
        }
      } catch {
        if (controller.signal.aborted) return;
      }
      timer = window.setTimeout(poll,operatorPollIntervalMs);
    };
    timer = window.setTimeout(poll,operatorPollIntervalMs);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [refresh, renderBatchActive, renderBatchJobId, request, setMessage]);
  useEffect(() => {
    let disposed = false;
    const timer = window.setTimeout(() => {
      const query = new URLSearchParams({
        limit: String(Math.min(500,Math.max(1,desiredResultCount,Number(activeRun?.resultCount??0)))),
        offset: '0',
        sort: 'score_desc',
      });
      if (activeRunCode) query.set('runCode',activeRunCode);
      Object.entries(resultFilters).forEach(([field,value])=>{
        if (field==='exactStateId' && !/^STATE-[A-F0-9]{20}$/.test(String(value))) return;
        if (value) query.set(field,value);
      });
      if (resultGroupBy) query.set('groupBy',resultGroupBy);
      if (lessonTagCatalog && tagControlsReady) appendLessonTagQuery(query,lessonTagCatalog,lessonTagFilters);
      setResultBusy(true);
      request<ResearchResultPage>(`/api/v2/research-results?${query.toString()}`)
        .then((page)=>{
          if (disposed) return;
          setStudioResults(page.items);
          setStudioResultTotal(page.total);
          setResultGroups(page.groups);
        })
        .catch(()=>{ if (!disposed) setMessage(t('workflow.research_scenario_results_load_failed')); })
        .finally(()=>{ if (!disposed) setResultBusy(false); });
    },250);
    return ()=>{ disposed=true; window.clearTimeout(timer); };
  },[activeRun?.resultCount,activeRunCode,desiredResultCount,filterFingerprint,ortools,resultGroupBy,request,setMessage,terminalRunRefresh,lessonTagCatalog,tagControlsReady]);

  const v4Recipes = availableRecipes.filter((item)=>Number(item.schemaVersion)===4);
  const legacyRecipeCount = availableRecipes.length-v4Recipes.length;
  const selectedRecipe = availableRecipes.find((item)=>item.code===selectedRecipeCode);
  const selectedRecipeVersion = Number(selectedRecipe?.schemaVersion??0);
  const selectedRecipeBundleRevisionId = String(jsonObject(selectedRecipe?.configuration).bundleRevisionId??'');
  const selectedRecipeSameBundle = selectedRecipeBundleRevisionId === selectedBundle.revisionId;
  const selectedRecipeBundle = availableBundles.find((item)=>item.revisionId===selectedRecipeBundleRevisionId);
  const groupFields = Object.keys(resultFacetParameters) as Array<keyof ResearchV4ResultFilters>;
  const facetFields = groupFields.filter((field)=>field!=='variantCollection');
  const facetSeedOptions: Partial<Record<keyof ResearchV4ResultFilters,string[]>> = {
    scenarioOrigin: configuration.origins,
    requestedProofMode: authoringCatalog.proof.modes,
    achievedProofMode: [...authoringCatalog.proof.modes,'none'],
    proofVisibility: [...authoringCatalog.proof.visibilities,'none'],
    targetTeam: ['north_south','west_east'],
    projectKind: [...new Set(configuration.projectOptions.map((item)=>item.kind))],
    projectValue: [...new Set(configuration.projectOptions.map((item)=>String(item.value)))],
    projectState: ['eligible','declared','comparison_winner','awarded'],
    winningRootActionBand: ['0','1','2_3','4_plus'],
    exactStateAvailable: ['true','false'],
    decisionKind: configuration.decisionCapabilities.map((item)=>item.kind),
    researchBackend: ['ortools_cp_sat_v2', 'ortools_cp_sat_v3'],
    qualificationAuthority: ['cp_sat'],
    runtimeCompatibilityStatus: ['not_checked','passed','failed'],
    configurationBundle: [selectedBundle.revisionId],
    scenarioCategory: authoringCatalog.sections,
  };
  const facetOptions = Object.fromEntries(facetFields.map((field)=>[
    field,
    [...new Set([
      ...(facetSeedOptions[field] ?? []),
      ...studioResults.flatMap((result)=>resultFacetValues(result,field)),
    ])].sort(),
  ])) as Record<keyof ResearchV4ResultFilters,string[]>;
  const facetLabels: Record<keyof ResearchV4ResultFilters,string> = {
    scenarioTemplate:t('workflow.research_scenario_facet_template'),
    scenarioOrigin:t('workflow.research_scenario_facet_origin'),
    scenarioStatus:t('workflow.research_scenario_facet_status'),
    requestedProofMode:t('workflow.research_scenario_facet_requested_proof'),
    achievedProofMode:t('workflow.research_scenario_facet_achieved_proof'),
    proofCompletion:t('workflow.research_scenario_facet_completion'),
    proofVisibility:t('workflow.research_scenario_facet_visibility'),
    targetTeam:t('workflow.research_scenario_facet_team'),
    contractOutcome:t('workflow.research_scenario_facet_contract_outcome'),
    projectKind:t('workflow.research_scenario_facet_project_kind'),
    projectValue:t('workflow.research_scenario_facet_project_value'),
    projectState:t('workflow.research_scenario_facet_project_state'),
    winningRootActionBand:t('workflow.research_scenario_facet_root_actions'),
    exactStateAvailable:t('workflow.research_scenario_facet_exact_state'),
    decisionKind:t('workflow.research_scenario_facet_decision'),
    researchBackend:tCatalog('research_scenario_backend'),
    qualificationAuthority:tCatalog('research_scenario_authority_title'),
    runtimeCompatibilityStatus:tCatalog('research_scenario_runtime_compatibility'),
    configurationBundle:t('workflow.research_bundle_selected'),
    exactStateId:t('workflow.research_scenario_exact_state_id'),
    scenarioCategory:t('workflow.research_scenario_target_section'),
    variantCollection:t('workflow.research_scenario_facet_run'),
  };
  const validationFresh = validation?.valid === true && validatedFingerprint === payloadFingerprint;
  const estimateFresh = estimate?.valid === true
    && estimate.launchAdmissible === true
    && estimatedFingerprint === payloadFingerprint;
  const actionGuidance = busy
    ? t('workflow.research_status_running')
    : launchStatusBlocked
      ? `${catalogFieldLabel(authoringCatalog,'status')} → ${t('workflow.research_scenario_status_published')}`
    : !validationFresh
      ? t('workflow.research_validate')
      : !estimateFresh
        ? t('workflow.research_estimate')
        : launchBlockReason
          ? t('workflow.research_scenario_request_failed')
          : t('workflow.research_validation_ok');
  const setQualificationField = (field:string,value:JsonValue) => setQualification((current)=>({...current,[field]:value}));
  const setResourceField = (field:string,value:JsonValue) => setResources((current)=>({...current,[field]:value}));
  const setDiversityField = (field:string,value:JsonValue) => setDiversity((current)=>({...current,[field]:value}));
  const updatePolicyProfile = async (
    seat: string,
    policyId: string,
    seed: number,
    materialization: string,
  ) => {
    const definition = authoringCatalog.policyCatalog.profiles[policyId];
    if (!definition || definition.materialization !== materialization) {
      throw new Error(`scenario_policy_catalog_mismatch:${policyId}`);
    }
    const normalizedSeed = Math.min(
      authoringCatalog.policyCatalog.seed.maximum,
      Math.max(authoringCatalog.policyCatalog.seed.minimum,Math.trunc(seed)),
    );
    const core: JsonObject = {
      schemaVersion: 1, seat, policyId, version: definition.version,
      seed: normalizedSeed, choices: [], materialization,
    };
    const profileHash = await canonicalBrowserSha256(core);
    const profile = {...core};
    delete profile.seat;
    profile.profileHash = profileHash;
    setPolicyProfiles((current)=>({...current,[seat]:profile}));
  };
  const policyRows = Object.entries(policyProfiles);
  const activeActionBounds = decisionKind === 'bid' || decisionKind === 'escalation'
    ? integerSchemaBounds(
        activeSelectorSchema,
        'actionIndex',
        ortools ? undefined : {minimum: 0,maximum: 10_000,default: 0},
      )
    : null;
  const activeProjectBounds = decisionKind === 'project'
    ? integerSchemaBounds(
        activeSelectorSchema,
        'projectIndex',
        ortools ? undefined : {minimum: 0,maximum: 10_000,default: 0},
      )
    : null;
  const decisionSeatField = decisionKind === 'project' ? 'owner' : 'actingSeat';
  const decisionSeatOptions = stringSchemaValues(
    activeSelectorSchema,
    decisionSeatField,
    ortools ? [] : authoringOptions.seats,
  );
  const decisionSeatOptional = decisionKind === 'play'
    && Array.isArray(activeSelectorSchema.required)
    && !activeSelectorSchema.required.includes(decisionSeatField);
  const ortoolsResultStatuses = ortools
    ? studioResults.map(resultRuntimeCompatibilityStatus)
    : [];
  const requestedVideoCount = Number(renderPolicy.videoCount??0);
  const qualifiedReceiptCount = Number(
    candidateReceiptCounts.qualified ?? activeRunProgress.qualified ?? 0,
  );
  const liveProgressTarget = Math.max(1, desiredResultCount);
  const batchSelectionSufficient = !ortools
    || Number(activeRun?.selectedCount??0) >= requestedVideoCount;
  const normalBatchMediaAllowed = !ortools || (
    ortoolsResultStatuses.length > 0
    && ortoolsResultStatuses.every((status)=>status==='passed')
    && batchSelectionSufficient
  );
  const auditBatchMediaAvailable = ortools
    && batchSelectionSufficient
    && ortoolsResultStatuses.some((status)=>status!=='passed');
  const scenarioAuthoringPanels = <>
    <section className="panel scenario-recipe-library" id="scenario-authoring">
      <header><div><h3>{t('workflow.research_scenario_recipe_library')}</h3><p>{t('workflow.research_scenario_recipe_library_help')}</p></div></header>
      <div className="scenario-catalog-controls">
        <label className="field-label"><span>{t('workflow.research_scenario_recipe_select')}</span><select value={selectedRecipeCode} aria-describedby={selectedRecipeVersion===4&&!selectedRecipeSameBundle?'scenario-recipe-bundle-mismatch':undefined} onChange={(event)=>{setSelectedRecipeCode(event.target.value);setConversionPreview(null);}}><option value="">—</option>{availableRecipes.map((item)=><option value={item.code} key={item.code}>{item.titleAr} · V{String(item.schemaVersion)} · {item.code}</option>)}</select></label>
        <button type="button" className="secondary" disabled={selectedRecipeVersion!==4||!selectedRecipeSameBundle||loadingRecipe} onClick={()=>void loadRecipe('load')}>{t('workflow.research_scenario_recipe_load')}</button>
        <button type="button" className="secondary" disabled={selectedRecipeVersion!==4||loadingRecipe} onClick={()=>void loadRecipe('clone')}>{t('workflow.research_scenario_recipe_clone')}</button>
        <button type="button" className="secondary" disabled={selectedRecipeVersion!==4||loadingRecipe} onClick={()=>void loadRecipe('variation')}>{t('workflow.research_scenario_recipe_variation')}</button>
        <button type="button" className="secondary" disabled={selectedRecipeVersion!==3||loadingRecipe} onClick={()=>void previewLegacyRecipe()}>{t('workflow.research_scenario_conversion_preview')}</button>
      </div>
      {selectedRecipeVersion===4&&!selectedRecipeSameBundle ? <p id="scenario-recipe-bundle-mismatch" className="scenario-backend-required" role="status">{t(selectedRecipeBundle?'workflow.research_scenario_recipe_bundle_mismatch':'workflow.research_configuration_unavailable_help')} {selectedRecipeBundle ? <b dir="ltr">{selectedRecipeBundle.version}</b> : <code dir="ltr">{selectedRecipeBundleRevisionId}</code>}</p> : null}
      {loadedSource ? <code className="scenario-loaded-source" dir="ltr">{loadedSource}</code> : null}
      {legacyRecipeCount ? <p className="scenario-backend-required">{t('workflow.research_scenario_conversion_backend_required')}</p> : null}
      {conversionPreview ? <section className="scenario-conversion-preview" aria-live="polite">
        <header><h4>{t('workflow.research_scenario_conversion_title')}</h4><span className={`status ${conversionPreview.buildable?'completed':'failed'}`}>{conversionPreview.buildable?t('workflow.research_scenario_conversion_buildable'):t('workflow.research_scenario_conversion_blocked')}</span></header>
        <dl><div><dt>{t('workflow.research_scenario_conversion_exact')}</dt><dd>{conversionPreview.exactlyRepresentable?'✓':'×'}</dd></div><div><dt>{t('workflow.research_scenario_conversion_mapped')}</dt><dd>{conversionPreview.mappedFields.length}</dd></div><div><dt>{t('workflow.research_scenario_conversion_unmapped')}</dt><dd>{conversionPreview.unmappedFields.length}</dd></div></dl>
        {conversionPreview.requiredOperatorFields.length ? <section><h5>{t('workflow.research_scenario_conversion_required')}</h5><ul>{conversionPreview.requiredOperatorFields.map((field)=><li dir="ltr" key={field}>{field}</li>)}</ul></section> : null}
        {conversionPreview.unmappedFields.length ? <details><summary>{t('workflow.research_scenario_conversion_unmapped')}</summary><ul>{conversionPreview.unmappedFields.map((field)=><li key={field.fieldPath}><code dir="ltr">{field.fieldPath}</code><span>{field.reasonCode}</span></li>)}</ul></details> : null}
        {conversionPreview.buildDiagnostics.length ? <details><summary>{t('workflow.research_scenario_conversion_diagnostics')}</summary><ul>{conversionPreview.buildDiagnostics.map((item)=><li key={`${item.fieldPath}-${item.code}`}><code dir="ltr">{item.fieldPath}</code><span>{item.message}</span></li>)}</ul></details> : null}
        <details className="audit-details"><summary>{t('workflow.research_scenario_conversion_proposed')}</summary><pre dir="ltr">{JSON.stringify(conversionPreview.proposedRecipe,null,2)}</pre></details>
        <code dir="ltr">{conversionPreview.sha256}</code>
        <div className="scenario-conversion-actions">
          {!conversionPreview.buildable ? <button type="button" className="secondary" disabled={loadingRecipe} onClick={()=>void previewLegacyRecipe(true)}>{t('workflow.research_scenario_conversion_apply_inputs')}</button> : null}
          {conversionPreview.buildable ? <button type="button" className="primary" disabled={loadingRecipe} onClick={()=>void loadRecipe('load',conversionPreview.proposedRecipe as ResearchRecipeV4Document)}>{t('workflow.research_scenario_conversion_open_draft')}</button> : null}
        </div>
      </section> : null}
    </section>

    {familyControlsReady && familyGeneration && lessonTagCatalog ? <section className="panel">
      <h3>{t('workflow.research_lesson_tag_generation_title')}</h3>
      <p>{familyGeneration.name}</p><p>{t('workflow.research_lesson_tag_generation_help')}</p>
      <p>{lessonTagResultLabels(lessonTagCatalog, familyGeneration.tagQuery).map(tCatalog).join(' · ')}</p>
      <small>{t('workflow.research_lesson_tag_generation_requires_play')}</small>
      <button type="button" className="secondary" disabled={busy || origin !== 'search' || decisionKind !== 'play'} onClick={() => {
        const nextGroup = {value: nextExpression};
        setConstraintExpression(hydrateExpression(familyGeneration.constraints, 'EXPR-CONSTRAINTS', nextGroup));
        setNextExpression(nextGroup.value);
        const selector = familyGeneration.decisionPoint.selector;
        setDecisionTrickIndex(selector.trickIndex); setDecisionPosition(selector.position);
        setDecisionRemainingCards(selector.remainingCards); setDecisionSeat(selector.actingSeat);
        setActiveExpressionTarget('EXPR-CONSTRAINTS');
      }}>{t('workflow.research_lesson_tag_generation_apply')}</button>
    </section> : null}

    {generationControlsReady && generationTagCatalog ? <section className="panel" data-field-path="generation">
      <h3>{t('workflow.research_lesson_tag_search_title')}</h3>
      <p id="research-generation-help">{t('workflow.research_lesson_tag_search_help')}</p>
      <label className="field-label"><input type="checkbox" name="research-generation-enabled"
        checked={generation !== null} disabled={busy || !generationScopeReady}
        aria-describedby="research-generation-help research-generation-scope" onChange={(event) => {
          setGeneration(event.target.checked ? readFamilySearchRequest(authoringCatalog, {
            schemaVersion: 1, catalogHash: authoringCatalog.familySearch!.catalog.sha256, tagQuery: {},
          }) : null);
        }}/><span>{t('workflow.research_lesson_tag_search_enable')}</span></label>
      <small id="research-generation-scope">{t('workflow.research_lesson_tag_generation_requires_play')}</small>
      {generation ? <fieldset disabled={busy}>
        <legend>{t('workflow.research_lesson_tag_search_title')}</legend>
        <LessonTagQueryControl catalog={generationTagCatalog} query={generation.tagQuery}
          onChange={(query) => setGeneration({schemaVersion: 1, catalogHash: generation.catalogHash,
            tagQuery: Object.fromEntries(Object.entries(query).filter((entry): entry is [string, string[]] => Array.isArray(entry[1]))),
            ...(generation.auctionSeeds ? {auctionSeeds: generation.auctionSeeds} : {})})}
          message={tCatalog} label={t('workflow.research_lesson_tag_search_title')}
          help={t('workflow.research_lesson_tag_search_help')}/>
        <p role="status" aria-live="polite">{generationMatches.length
          ? t('workflow.research_lesson_tag_search_matching').replace('{count}', number.format(generationMatches.length))
          : t('workflow.research_lesson_tag_search_no_match')}</p>
        {auctionSeedControlsReady ? <fieldset data-field-path="generation.auctionSeeds">
          <legend>{t('workflow.research_auction_seeds_title')}</legend>
          <p id="research-auction-seeds-help">{t('workflow.research_auction_seeds_help')}</p>
          <p id="research-auction-seeds-limit" role="status" aria-live="polite">{t('workflow.research_auction_seeds_limit')
            .replace('{count}', number.format(generation.auctionSeeds?.length ?? 0))
            .replace('{maximum}', number.format(auctionSeedLimit!))}</p>
          {manualAuctionSeedReady ? <form className="scenario-grid" data-research-opening-draft onInput={() => setAuctionSeedError(false)}
            onSubmit={(event) => {
              event.preventDefault();
              const data = new FormData(event.currentTarget);
              const seed = data.get('auctionSeed');
              try {
                setGeneration(appendFamilySearchAuctionSeed(authoringCatalog, generation,
                  {seed: typeof seed === 'string' && seed.trim() ? Number(seed) : NaN,
                    dealer: data.get('auctionDealer')}));
                setAuctionSeedError(false);
                event.currentTarget.reset();
              } catch { setAuctionSeedError(true); }
            }}>
            <label className="field-label" htmlFor="research-auction-seed-input"><span>{t('workflow.research_sequential_seed')}</span>
              <input id="research-auction-seed-input" name="auctionSeed" type="number" required step={1}
                min={typeof auctionSeedSchema.minimum === 'number' ? auctionSeedSchema.minimum : undefined}
                max={typeof auctionSeedSchema.maximum === 'number' ? auctionSeedSchema.maximum : undefined}
                aria-describedby="research-auction-seeds-help research-auction-seeds-error"/></label>
            <label className="field-label" htmlFor="research-auction-dealer-input"><span>{t('workflow.research_scenario_dealer')}</span>
              <select id="research-auction-dealer-input" name="auctionDealer" required defaultValue=""
                aria-describedby="research-auction-seeds-help research-auction-seeds-error">
                <option value="">—</option>{auctionDealerValues.map((dealer) =>
                  <option key={String(dealer)} value={String(dealer)}>{optionLabel(authoringCatalog, dealer)}</option>)}
              </select></label>
            <button type="submit" className="secondary">{t('workflow.research_auction_seeds_add')}</button>
            <p id="research-auction-seeds-error" role="alert">{auctionSeedError ? t('workflow.research_auction_seeds_invalid') : ''}</p>
          </form> : null}
          {auctionOpenings.length ? auctionOpenings.map((opening) => {
            const key = `${opening.seed}:${opening.dealer}`;
            const id = `research-auction-seed-${opening.seed}-${opening.dealer}`;
            const checked = generation.auctionSeeds?.some((row) => row.seed === opening.seed && row.dealer === opening.dealer) ?? false;
            return <label className="field-label" htmlFor={id} key={key}>
              <input id={id} type="checkbox" name="research-auction-seeds" value={key} checked={checked}
                disabled={!checked && (generation.auctionSeeds?.length ?? 0) >= auctionSeedLimit!}
                aria-describedby="research-auction-seeds-help research-auction-seeds-limit" onChange={(event) => {
                  const selected = event.target.checked ? [...(generation.auctionSeeds ?? []), opening]
                    : (generation.auctionSeeds ?? []).filter((row) => row.seed !== opening.seed || row.dealer !== opening.dealer);
                  setGeneration({schemaVersion: 1, catalogHash: generation.catalogHash, tagQuery: generation.tagQuery,
                    ...(selected.length ? {auctionSeeds: selected} : {})});
                }}/>
              <span><bdi>{number.format(opening.seed)}</bdi>{' · '}{optionLabel(authoringCatalog, opening.dealer)}</span>
            </label>;
          }) : <p>{t('workflow.research_auction_seeds_empty')}</p>}
        </fieldset> : null}
      </fieldset> : null}
    </section> : null}

    <section className="panel">
      <div className="scenario-grid">
        <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="titleAr" label={catalogFieldLabel(authoringCatalog,'titleAr')} value={titleAr} onChange={(value)=>setTitleAr(String(value))}/>
        <label className="field-label" data-field-path="scenario.origin.kind"><span>{t('workflow.research_scenario_origin')}</span><select value={origin} onChange={(event)=>{const next=event.target.value as typeof origin;setOrigin(next);if(next!=='search')clearGeneration();if(next==='exact_trace_prefix'&&!authoringCatalog.playingQuality?.modesByOrigin?.[next]?.includes('exact'))setPlayingQuality((current)=>current?.mode==='exact'?null:current);}}>{supportedOrigins.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        {!ortools && origin === 'search' ? <label className="field-label" data-field-path="resources.generationPolicy"><span>{t('workflow.research_scenario_generation_policy')}</span><select value={generationPolicy} onChange={(event)=>setGenerationPolicy(event.target.value as typeof generationPolicy)}><option value="natural_policy">{t('workflow.research_scenario_policy_natural')}</option><option value="scenario_guided_actor_safe">{t('workflow.research_scenario_policy_guided')}</option></select></label> : null}
        <label className="field-label" data-field-path="scenario.decisionPoint.kind"><span>{t('workflow.research_scenario_decision')}</span><select value={decisionKind} onChange={(event)=>{const next=event.target.value as typeof decisionKind;setDecisionKind(next);if(next!=='play')clearGeneration();}}>{supportedDecisionKinds.map((item)=><option value={item.kind} key={item.kind}>{catalogValueLabel(authoringCatalog,item.kind)}</option>)}</select></label>
        {activeActionBounds ? <label className="field-label" data-field-path="scenario.decisionPoint.selector.actionIndex"><span>{t('workflow.research_scenario_decision_action_index')}</span><input type="number" min={activeActionBounds.minimum} max={activeActionBounds.maximum} step={1} value={decisionActionIndex} onChange={(event)=>setDecisionActionIndex(Math.min(activeActionBounds.maximum,Math.max(activeActionBounds.minimum,Number(event.target.value))))}/></label> : null}
        {activeProjectBounds ? <label className="field-label" data-field-path="scenario.decisionPoint.selector.projectIndex"><span>{t('workflow.research_scenario_project_index')}</span><input type="number" min={activeProjectBounds.minimum} max={activeProjectBounds.maximum} step={1} value={projectIndex} onChange={(event)=>setProjectIndex(Math.min(activeProjectBounds.maximum,Math.max(activeProjectBounds.minimum,Number(event.target.value))))}/></label> : null}
        {decisionKind==='play' ? <><label className="field-label" htmlFor="research-play-trick-index" data-field-path="scenario.decisionPoint.selector.trickIndex"><span>{tCatalog('research_scenario_play_trick_index')}</span><input id="research-play-trick-index" type="number" min={scenarioOrdinalDisplayValue('trickIndex',playTrickBounds.minimum)} max={scenarioOrdinalDisplayValue('trickIndex',playTrickBounds.maximum)} step={1} value={scenarioOrdinalDisplayValue('trickIndex',decisionTrickIndex)} onChange={(event)=>setDecisionTrickIndex(Math.min(playTrickBounds.maximum,Math.max(playTrickBounds.minimum,scenarioOrdinalWireValue('trickIndex',Number(event.target.value)))))}/></label><label className="field-label" htmlFor="research-play-position" data-field-path="scenario.decisionPoint.selector.position"><span>{tCatalog('research_scenario_argument_position')}{generation === null ? ` · ${t('workflow.research_rule_authority_value_optional')}` : ''}</span><input id="research-play-position" type="number" required={generation !== null} min={scenarioOrdinalDisplayValue('position',playPositionBounds.minimum)} max={scenarioOrdinalDisplayValue('position',playPositionBounds.maximum)} step={1} value={decisionPosition===undefined?'':scenarioOrdinalDisplayValue('position',decisionPosition)} onChange={(event)=>setDecisionPosition(event.target.value===''?undefined:Math.min(playPositionBounds.maximum,Math.max(playPositionBounds.minimum,scenarioOrdinalWireValue('position',Number(event.target.value)))))}/></label><label className="field-label" htmlFor="research-play-remaining-cards" data-field-path="scenario.decisionPoint.selector.remainingCards"><span>{t('workflow.research_scenario_decision_remaining_cards')}</span><input id="research-play-remaining-cards" type="number" min={playRemainingBounds.minimum} max={playRemainingBounds.maximum} step={1} value={decisionRemainingCards} onChange={(event)=>setDecisionRemainingCards(Math.min(playRemainingBounds.maximum,Math.max(playRemainingBounds.minimum,Number(event.target.value))))}/></label></> : null}
        {decisionKind!=='bid' ? <label className="field-label" htmlFor="research-decision-seat" data-field-path={`scenario.decisionPoint.selector.${decisionSeatField}`}><span>{decisionKind==='project' ? t('workflow.research_scenario_project_owner') : t('workflow.research_scenario_decision_actor')}{decisionSeatOptional && generation === null ? ` · ${t('workflow.research_rule_authority_value_optional')}` : ''}</span><select id="research-decision-seat" required={generation !== null} value={decisionSeat} onChange={(event)=>setDecisionSeat(event.target.value)}>{decisionSeatOptional ? <option value="">{generation === null ? t('workflow.research_rule_authority_value_optional') : t('workflow.research_scenario_decision_actor')}</option> : null}{decisionSeatOptions.map((seat)=><option value={seat} key={seat}>{optionLabel(authoringCatalog,seat)}</option>)}</select></label> : null}
      </div>
      {origin==='exact_trace_prefix' ? <ExactTraceEditor raw={exactTraceRaw} options={authoringOptions} catalog={authoringCatalog} preview={exactTracePreview} previewBusy={exactTracePreviewBusy} onChange={setExactTraceRaw} onInvalid={()=>setMessage(t('workflow.research_scenario_json_invalid'))} onPreviewRetry={()=>setExactTracePreviewAttempt((attempt)=>attempt+1)}
        onOpportunity={playingQuality && authoringCatalog.playingQuality ? (action)=>{
          const prefix = parseExactTrace(exactTraceRaw);
          const limit = Number(authoringCatalog.playingQuality!.recordedEscalationOpportunities.maxItems);
          if (!prefix || !action.actor || recordedEscalationOpportunities.length >= limit) {
            setMessage(t('workflow.research_scenario_json_invalid')); return;
          }
          if (action.action !== 'decline') setExactTraceRaw(JSON.stringify(appendExactTraceLegalAction(prefix,action),null,2));
          setRecordedEscalationOpportunities((current)=>[...current,{actor:action.actor,actionCode:action.code}]);
        } : undefined}/> : null}
      {origin === 'exact_trace_prefix' && playingQuality && recordedEscalationOpportunities.length ?
        <section data-field-path="scenario.origin.recordedEscalationOpportunities">
          <h4>{t('workflow.research_scenario_timeline_escalation')}</h4>
          <ol>{recordedEscalationOpportunities.map((event,index)=><li key={index}>
            {catalogValueLabel(authoringCatalog,event.actor)}{' · '}
            {String(event.actionCode).split(':').map((part,partIndex)=><span key={partIndex}>
              {partIndex ? ' · ' : ''}{authoringCatalog.localizationKeys.values[part]
                ? catalogValueLabel(authoringCatalog,part) : <code dir="ltr">{part}</code>}
            </span>)}
            {index === recordedEscalationOpportunities.length - 1 ? <button type="button" className="quiet"
              disabled={['projectActions','projectDeclarations','playActions'].some((field)=>{
                const actions = parseExactTrace(exactTraceRaw)?.[field];
                return Array.isArray(actions) && actions.length > 0;
              })}
              onClick={()=>{
                const prefix = parseExactTrace(exactTraceRaw);
                if (!prefix) return setMessage(t('workflow.research_scenario_json_invalid'));
                try {
                  const next = undoRecordedEscalationOpportunity(prefix,recordedEscalationOpportunities);
                  setExactTraceRaw(JSON.stringify(next.prefix,null,2));
                  setRecordedEscalationOpportunities(next.history);
                } catch { setMessage(t('workflow.research_scenario_json_invalid')); }
              }}>{t('workflow.research_scenario_remove')}</button> : null}
          </li>)}</ol>
        </section> : null}
    </section>

    <section className="panel scenario-catalog">
      <header><div><h3>{t('workflow.research_scenario_catalog')}</h3><p>{t('workflow.research_scenario_catalog_help')}</p></div><output aria-live="polite" data-catalog-total={allDefinitions.length}>{visibleDefinitions.length}</output></header>
      <div className="scenario-catalog-controls">
        <label className="field-label"><span>{t('workflow.research_scenario_predicate')}</span><input type="search" value={catalogQuery} aria-describedby="scenario-catalog-search-help" onChange={(event)=>setCatalogQuery(event.target.value)}/><small id="scenario-catalog-search-help">{t('workflow.research_scenario_catalog_help')}</small></label>
        <label className="field-label"><span>{t('workflow.research_scenario_predicate')}</span><select value={selectedDefinition?.kind??''} onChange={(event)=>setSelectedKind(event.target.value)}>{authoringCatalog.sections.map((stage)=><optgroup label={stageLabel(stage)} key={stage}>{(grouped[stage]??[]).map((definition)=><option value={definition.kind} key={definition.kind}>{predicateLabel(definition)}</option>)}</optgroup>)}</select></label>
        <label className="field-label"><span>{t('workflow.research_scenario_target_section')}</span><select value={activeExpressionTarget} onChange={(event)=>{const value=event.target.value;activateExpressionTarget(value,expressionGroups(constraintExpression).some((group)=>group.nodeId===value)?'constraints':'outcome');}}>{expressionGroups(constraintExpression).map((group)=><option value={group.nodeId} key={group.nodeId}>{t('workflow.research_scenario_constraints')} · {group.nodeId}</option>)}{expressionGroups(outcomeExpression).map((group)=><option value={group.nodeId} key={group.nodeId}>{t('workflow.research_scenario_outcome')} · {group.nodeId}</option>)}</select></label>
        <button type="button" className="primary" onClick={addPredicate} disabled={!selectedDefinition||targetAtCapacity}>{t('workflow.research_scenario_add')}</button>
      </div>
      {selectedDefinition ? <div className="scenario-catalog-proof"><span>{ortools?catalogValueLabel(authoringCatalog,selectedDefinition.cpSatSupport?.qualificationSource??'unsupported'):capabilityLabel(selectedDefinition.constructionCapability)}</span><code dir="ltr">{selectedDefinition.kind}</code><small>{ortools?predicateHelp(authoringCatalog,selectedDefinition):selectedDefinition.supportedProofModes.join(' · ')}</small></div> : null}
    </section>

    <section className="scenario-expression-grid">
      <section className="panel"><h3>{t('workflow.research_scenario_constraints')}</h3><ExpressionEditor value={constraintExpression} root activeTarget={activeExpressionTarget} catalog={authoringCatalog} totalNodes={expressionNodeCount(constraintExpression)} definitions={definitionMap} onActivate={(nodeId)=>activateExpressionTarget(nodeId,'constraints')} onChange={(value)=>setConstraintExpression((current)=>{const candidate=normalizeExpression(value,definitionMap);return expressionWithinLimits(candidate,authoringCatalog.expressionLimits)?candidate:current;})} onAddGroup={addExpressionGroup} onRemove={removeExpressionNode} renderPredicate={renderPredicate}/></section>
      <section className="panel"><h3>{t('workflow.research_scenario_outcome')}</h3><ExpressionEditor value={outcomeExpression} root activeTarget={activeExpressionTarget} catalog={authoringCatalog} totalNodes={expressionNodeCount(outcomeExpression)} definitions={definitionMap} onActivate={(nodeId)=>activateExpressionTarget(nodeId,'outcome')} onChange={(value)=>setOutcomeExpression((current)=>{const candidate=normalizeExpression(value,definitionMap);return expressionWithinLimits(candidate,authoringCatalog.expressionLimits)?candidate:current;})} onAddGroup={addExpressionGroup} onRemove={removeExpressionNode} renderPredicate={renderPredicate}/></section>
    </section>
  </>;
  const qualityExactFrom = Number(qualityExecution.exactPlayFromTrick ?? 0);
  const qualityVideoInterest = jsonObject(qualityExecution.videoInterestingness);
  const qualityBoundaryCopy = qualityExecution.earlyPlayPolicyId && qualityExactFrom > 1
    ? t(qualityExecution.exactTailProof === true
      ? 'workflow.research_quality_exact_tail_boundary'
      : 'workflow.research_quality_policy_boundary')
      .replace('{policyEnd}', number.format(qualityExactFrom - 1))
      .replace('{exactFrom}', number.format(qualityExactFrom))
    : qualityExactFrom === 1 ? t('workflow.research_quality_native_from_first') : null;
  function buildPlayingQuality(mode:string, current:JsonObject|null=null,
    qualityVisibility=String(current?.visibility??scenarioCatalog.playingQuality?.legacyVisibility??'')):JsonObject {
    const catalog = scenarioCatalog.playingQuality!;
    return {
    schemaVersion: catalog.policySchemaVersion??catalog.schemaVersion,
    ...(catalog.policySchemaVersion ? {visibility:qualityVisibility} : {}),
    mode,
    decisions: Object.fromEntries(catalog.decisionKinds.map((kind)=>{
      const previous = jsonObject(jsonObject(current?.decisions)[kind]);
      const budgetMode = playingQualityBudgetDefaultMode(mode, kind, qualityExecution.exactTailProof);
      const base = {
        utility: String(previous.utility??utility),
        budgetProfile: preferredPlayingQualityBudget(mode, current?.mode, previous.budgetProfile,
          Object.keys(scenarioCatalog.proof.budgetProfiles),
          catalog.defaultBudgetProfileByModeAndVisibility?.[budgetMode]?.[qualityVisibility]
          ?? catalog.defaultBudgetProfileByMode[budgetMode]
          ?? catalog.defaultBudgetProfile),
      };
      if(mode==='exact') return [kind,base];
      if(current?.mode===mode && previous.horizon && previous.boundaryEvaluator) return [kind,{...previous,...base}];
      const boundaryEvaluator = catalog.defaultBoundaryEvaluatorByDecision[kind];
      const evaluator = catalog.boundaryEvaluatorDefinitions[boundaryEvaluator];
      const allowedUnits = catalog.horizonUnitsByDecision[kind] ?? [];
      const unit = allowedUnits.includes(horizonUnit) && evaluator?.horizonUnits.includes(horizonUnit)
        ? horizonUnit : evaluator?.horizonUnits.find((value)=>allowedUnits.includes(value)) ?? allowedUnits[0]!;
      return [kind,{...base,
        utility:evaluator?.utilities?.includes(base.utility)?base.utility:evaluator?.utilities?.[0]??base.utility,
        horizon:{unit,value:evaluator?.horizonValue??horizonValue},boundaryEvaluator}];
    })),
    };
  }
  const proofPanels = <>
    {authoringCatalog.playingQuality ? <section className="panel" data-field-path="playingQuality">
      <h3>{t('workflow.research_all_decisions_required')}</h3>
      <label className="field-label"><input type="checkbox" checked={playingQuality !== null} onChange={(event)=>{
        setPlayingQuality(event.target.checked ? buildPlayingQuality(qualityModes[0]!) : null);
        if (!event.target.checked) clearGeneration();
      }}/><span>{t('workflow.research_scenario_help_require_all_decision_audit')}</span></label>
      {playingQuality ? <fieldset data-field-path="playingQuality.mode"><legend>{t('workflow.research_scenario_proof_mode')}</legend>
        {qualityModes.map((mode)=><label className="field-label" key={mode}><input type="radio" name="playing-quality-mode" value={mode} checked={playingQuality.mode===mode} onChange={()=>setPlayingQuality((current)=>buildPlayingQuality(mode,current))}/><span>{catalogValueLabel(authoringCatalog,mode)}</span></label>)}
      </fieldset> : null}
      {playingQuality && authoringCatalog.playingQuality.visibilities ? <label className="field-label" data-field-path="playingQuality.visibility">
        <span>{t('workflow.research_scenario_visibility')}</span>
        <select name="playing-quality-visibility" value={String(playingQuality.visibility??authoringCatalog.playingQuality.legacyVisibility)}
          onChange={(event)=>{const scope=event.target.value;setPlayingQuality((current)=>current?buildPlayingQuality(String(current.mode),current,scope):null);}}>
          {authoringCatalog.playingQuality.visibilities.map((scope)=><option key={scope} value={scope}>{catalogValueLabel(authoringCatalog,scope)}</option>)}
        </select>
      </label> : null}
      {playingQuality?.mode==='budgeted' && qualityBoundaryCopy ? <p className="scenario-proof-mode-help" role="note">{qualityBoundaryCopy}</p> : null}
      {playingQuality?.mode==='exact' ? <p className="scenario-proof-mode-help" role="note">{t('workflow.research_scenario_proof_exact_help')}</p> : null}
      {playingQuality?.visibility==='complete_state' && qualityVideoInterest.schemaVersion===1 ? <p className="scenario-proof-mode-help" role="note">{t('workflow.research_quality_video_interest')
        .replace('{margin}',number.format(Number(qualityVideoInterest.maximumFinalMargin)))
        .replace('{lastTrick}',number.format(Number(qualityVideoInterest.maximumFeaturedTrick)))
        .replace('{swing}',number.format(Number(qualityVideoInterest.minimumPointSwing)))
        .replace('{count}',number.format(Number(qualityVideoInterest.minimumMeaningfulDecisions)))}</p> : null}
      {playingQuality ? authoringCatalog.playingQuality.decisionKinds.map((kind)=>{
        const decisions = jsonObject(playingQuality.decisions);
        const analysis = jsonObject(decisions[kind]);
        const horizon = jsonObject(analysis.horizon);
        const evaluator = authoringCatalog.playingQuality!.boundaryEvaluatorDefinitions[String(analysis.boundaryEvaluator)];
        const allowedHorizonUnits = authoringCatalog.playingQuality!.horizonUnitsByDecision[kind] ?? [];
        const horizonBounds = authoringCatalog.playingQuality!.horizonBounds[String(analysis.budgetProfile)]?.[String(horizon.unit)];
        const update = (change:JsonObject) => setPlayingQuality((current)=>{
          if (!current) return null;
          const next: JsonObject = {...current, decisions: {...jsonObject(current.decisions), [kind]: {...analysis,...change}}};
          delete next.sha256;
          return next;
        });
        return <fieldset key={kind} data-field-path={`playingQuality.decisions.${kind}`}>
          <legend>{catalogValueLabel(authoringCatalog,kind)}</legend><div className="scenario-grid">
          {([
            ['utility', 'research_scenario_utility', authoringCatalog.proof.utilities.filter(
              (name)=>playingQuality.mode==='exact'||evaluator&&(!evaluator.utilities||evaluator.utilities.includes(name)),
            )],
            ['budgetProfile', 'research_scenario_budget', Object.keys(authoringCatalog.proof.budgetProfiles)],
            ...(playingQuality.mode==='budgeted' ? [['boundaryEvaluator', 'research_scenario_boundary_evaluator', authoringCatalog.playingQuality!.boundaryEvaluators.filter(
              (name)=>authoringCatalog.playingQuality!.boundaryEvaluatorDefinitions[name]?.decisionKinds.includes(kind),
            )] as const] : []),
          ] as const).map(([field,label,choices])=><label className="field-label" key={field}>
            <span>{tCatalog(label)}</span><select value={String(analysis[field]??'')} onChange={(event)=>{
              const value = event.target.value;
              const selected = field === 'boundaryEvaluator' ? authoringCatalog.playingQuality!.boundaryEvaluatorDefinitions[value] : undefined;
              const selectedUnits = selected?.horizonUnits.filter((unit)=>allowedHorizonUnits.includes(unit)) ?? [];
              if (selected && !selectedUnits.length) return;
              update(selected ? {[field]:value,
                utility: selected.utilities?.includes(String(analysis.utility)) !== false ? analysis.utility : selected.utilities[0],
                horizon: {...horizon,
                  unit: selectedUnits.includes(String(horizon.unit)) ? String(horizon.unit) : selectedUnits[0],
                  value: selected.horizonValue ?? horizon.value},
              } : {[field]:value});
            }}>
              {choices.map((choice)=><option key={choice} value={choice}>{catalogValueLabel(authoringCatalog,choice)}</option>)}
            </select></label>)}
          {playingQuality.mode==='budgeted' ? <><label className="field-label"><span>{t('workflow.research_scenario_horizon_unit')}</span><select value={String(horizon.unit??'')} onChange={(event)=>update({horizon:{...horizon,unit:event.target.value}})}>{authoringCatalog.proof.horizonUnits.filter((unit)=>evaluator?.horizonUnits.includes(unit) && allowedHorizonUnits.includes(unit)).map((unit)=><option key={unit} value={unit}>{catalogValueLabel(authoringCatalog,unit)}</option>)}</select></label>
          <label className="field-label" data-field-path={`playingQuality.decisions.${kind}.horizon`}><span>{t('workflow.research_scenario_horizon_value')}</span><input type="number" required min={Math.max(evaluator?.horizonValue ?? 1,horizonBounds?.minimum ?? 1)} max={evaluator?.horizonValue === undefined ? horizonBounds?.maximum : Math.min(evaluator.horizonValue,horizonBounds?.maximum ?? evaluator.horizonValue)} step={horizonBounds?.step} value={Number(horizon.value??0)} onChange={(event)=>update({horizon:{...horizon,value:Number(event.target.value)}})}/></label></> : null}
          </div></fieldset>;
      }) : null}
    </section> : null}
    <section className="panel" id="scenario-proof">
      <h3>{t('workflow.research_scenario_proof')}</h3>
      <p className="scenario-proof-mode-help">{proofModeHelp}</p>
      <div className="scenario-grid">
        <label className="field-label" data-field-path="proof.mode"><span>{t('workflow.research_scenario_proof_mode')}</span><select value={proofMode} onChange={(event)=>setProofMode(event.target.value as typeof proofMode)}>{availableProofModes.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        <label className="field-label" data-field-path="proof.visibility"><span>{t('workflow.research_scenario_visibility')}</span><select value={visibility} onChange={(event)=>setVisibility(event.target.value as typeof visibility)}>{availableVisibilities.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        {strategyClass ? <label className="field-label" data-field-path="proof.strategyClass"><span>{t('workflow.research_scenario_strategy_class')}</span><output>{catalogValueLabel(authoringCatalog,strategyClass)}</output><small>{t('workflow.research_scenario_strategy_class_help')}</small></label> : null}
        <label className="field-label" data-field-path="proof.claim"><span>{t('workflow.research_scenario_claim')}</span><select value={claim} onChange={(event)=>setClaim(event.target.value as typeof claim)}>{availableClaims.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        <label className="field-label" data-field-path="proof.targetTeam"><span>{t('workflow.research_scenario_target_team')}</span><select value={targetTeam} onChange={(event)=>setTargetTeam(event.target.value)}>{targetTeamOptions.map((team)=><option value={team} key={team}>{optionLabel(authoringCatalog,team,'proof.targetTeam',targetTeamSchema)}</option>)}</select></label>
        <label className="field-label" data-field-path="proof.utility"><span>{t('workflow.research_scenario_utility')}</span><select value={utility} onChange={(event)=>setUtility(event.target.value as typeof utility)}>{authoringCatalog.proof.utilities.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        <label className="field-label" data-field-path="proof.successThreshold"><span>{t('workflow.research_scenario_threshold')}</span><input type="number" value={threshold} onChange={(event)=>setThreshold(Number(event.target.value))}/></label>
        <label className="field-label" data-field-path="proof.budgetProfile"><span>{t('workflow.research_scenario_budget')}</span><select value={budgetProfile} onChange={(event)=>setBudgetProfile(event.target.value)}>{Object.keys(authoringCatalog.proof.budgetProfiles).map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        {claim==='winning_continuation_line_count'||hasHardContinuationCount ? <label className="field-label" data-field-path="proof.lineUniverse"><span>{t('workflow.research_scenario_line_universe')}</span><select value={lineUniverse} onChange={(event)=>setLineUniverse(event.target.value as typeof lineUniverse)}>{availableLineUniverses.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label> : null}
        {claim==='selected_action_is_unique_winner'||claim==='winning_continuation_line_count'||hasHardContinuationCount ? <label className="field-label" data-field-path="proof.selectedAction"><span>{t('workflow.research_scenario_selected_action')}</span><select value={selectedAction} aria-required={claim==='selected_action_is_unique_winner'||hasHardContinuationCount} aria-invalid={Boolean(selectedActionDiagnostic)} aria-describedby={selectedActionDiagnostic?'scenario-selected-action-error':undefined} onChange={(event)=>setSelectedAction(event.target.value)}><option value="">—</option>{selectedActionOptions.map((value)=><option value={value} key={value}>{decisionKind==='play'?playActionLabels.get(value)??value:value}</option>)}</select>{selectedActionDiagnostic?<small id="scenario-selected-action-error">{diagnosticMessage(selectedActionDiagnostic)}</small>:null}</label> : null}
        {proofMode==='bounded' ? <>
          <label className="field-label" data-field-path="proof.horizon.unit"><span>{t('workflow.research_scenario_horizon_unit')}</span><select value={horizonUnit} onChange={(event)=>setHorizonUnit(event.target.value as typeof horizonUnit)}>{authoringCatalog.proof.horizonUnits.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
          <label className="field-label" data-field-path="proof.horizon.value"><span>{t('workflow.research_scenario_horizon_value')}</span><input type="number" min={1} step={1} value={horizonValue} onChange={(event)=>setHorizonValue(Math.max(1,Number(event.target.value)))}/></label>
          <label className="field-label" data-field-path="proof.boundaryEvaluator"><span>{t('workflow.research_scenario_boundary_evaluator')}</span><select value={boundaryEvaluator} onChange={(event)=>setBoundaryEvaluator(event.target.value as typeof boundaryEvaluator)}>{authoringCatalog.proof.boundaryEvaluators.map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label>
        </> : null}
      </div>
      {proofMode==='policy_conditioned' ? <section className="scenario-policy-identities"><h4>{t('workflow.research_scenario_policy_identities')}</h4>{policyRows.map(([seat,raw])=>{const policy=jsonObject(raw);const policyId=String(policy.policyId??authoringCatalog.policyCatalog.defaultPolicyId);const definition=authoringCatalog.policyCatalog.profiles[policyId];const materialization=String(policy.materialization??definition?.materialization??'');return <article key={seat}><h5>{optionLabel(authoringCatalog,seat)}</h5><label className="field-label" data-field-path={`proof.policies.${seat}.policyId`}><span>{t('workflow.research_scenario_policy_profile')}</span><select value={policyId} onChange={(event)=>{const selected=authoringCatalog.policyCatalog.profiles[event.target.value];if(selected)void updatePolicyProfile(seat,event.target.value,Number(policy.seed??0),selected.materialization);}}>{Object.keys(authoringCatalog.policyCatalog.profiles).map((value)=><option value={value} key={value}>{catalogValueLabel(authoringCatalog,value)}</option>)}</select></label><label className="field-label" data-field-path={`proof.policies.${seat}.seed`}><span>{t('workflow.research_scenario_policy_seed')}</span><input type="number" min={authoringCatalog.policyCatalog.seed.minimum} max={authoringCatalog.policyCatalog.seed.maximum} step={1} value={Number(policy.seed??0)} onChange={(event)=>void updatePolicyProfile(seat,policyId,Number(event.target.value),materialization)}/></label><label className="field-label" data-field-path={`proof.policies.${seat}.materialization`}><span>{t('workflow.research_scenario_policy_action_order')}</span><select value={materialization} onChange={(event)=>void updatePolicyProfile(seat,policyId,Number(policy.seed??0),event.target.value)}><option value={definition?.materialization}>{catalogValueLabel(authoringCatalog,definition?.materialization??'')}</option></select></label><code dir="ltr">{String(policy.profileHash??'—')}</code></article>;})}</section> : null}
    </section>

    <div className="scenario-governance-grid">
    <section className="panel"><h3>{t('workflow.research_scenario_qualification_title')}</h3><div className="scenario-grid">{(['minimumTruth','minimumConfidence','minimumSurroundingQuality','maximumRegret','requireObserverSafety','requireAllDecisionAudit'] as const).map((field)=>{const path=`qualification.${field}`;return <CatalogScalarField key={field} catalog={fieldCatalog} authoringCatalog={authoringCatalog} path={path} label={catalogFieldLabel(authoringCatalog,path)} value={qualification[field]} onChange={(value)=>setQualificationField(field,value)}/>;})}</div></section>

    <section className="panel"><h3>{t('workflow.research_scenario_resources_title')}</h3><div className="scenario-grid">{resourceFieldNames.map((field)=>{const path=`resources.${field}`;const limits=fieldCatalog.fields[path]?.maximumByBudgetProfile;const limit=limits && typeof limits==='object' && !Array.isArray(limits) ? (limits as JsonObject)[budgetProfile] : undefined;return <CatalogScalarField key={field} catalog={fieldCatalog} authoringCatalog={authoringCatalog} path={path} label={catalogFieldLabel(authoringCatalog,path)} value={resources[field]} maximum={field==='shardWorkerCount' && typeof limit==='number' ? limit : undefined} onChange={(value)=>setResourceField(field,value)}/>;})}</div></section>

    <section className="panel"><h3>{t('workflow.research_scenario_diversity_title')}</h3><div className="scenario-grid">
      <CatalogMultiField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="diversity.cellDimensions" label={catalogFieldLabel(authoringCatalog,'diversity.cellDimensions')} value={Array.isArray(diversity.cellDimensions)?diversity.cellDimensions.map(String):[]} onChange={(value)=>setDiversityField('cellDimensions',value)}/>
      <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="diversity.topKPerCell" label={catalogFieldLabel(authoringCatalog,'diversity.topKPerCell')} value={diversity.topKPerCell} onChange={(value)=>setDiversityField('topKPerCell',value)}/>
      <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="diversity.rejectedAuditModulus" label={catalogFieldLabel(authoringCatalog,'diversity.rejectedAuditModulus')} value={diversity.rejectedAuditModulus} onChange={(value)=>setDiversityField('rejectedAuditModulus',value)}/>
    </div></section>

    <section className="panel"><h3>{t('workflow.research_scenario_publication_title')}</h3><div className="scenario-grid">
      <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="renderPolicy.mode" label={catalogFieldLabel(authoringCatalog,'renderPolicy.mode')} value={String(renderPolicy.mode)} onChange={(value)=>{const mode=String(value);setRenderPolicy((current)=>({mode,videoCount:mode==='disabled'?0:Math.max(Number(fieldCatalog.fields['renderPolicy.videoCount'].default),Number(current.videoCount??0))}));}}/>
      <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="status" label={catalogFieldLabel(authoringCatalog,'status')} value={recipeStatus} onChange={(value)=>setRecipeStatus(value as typeof recipeStatus)}/>
    </div></section>

    <section className="panel scenario-output-controls"><h3>{t('workflow.research_scenario_output_title')}</h3><div className="scenario-grid">
      <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="output.kind" label={catalogFieldLabel(authoringCatalog,'output.kind')} value={outputKind} disabled={origin==='exact_trace_prefix'} onChange={(value)=>setOutputKind(value as typeof outputKind)}/>
      <CatalogScalarField catalog={fieldCatalog} authoringCatalog={authoringCatalog} path="output.resultCount" label={catalogFieldLabel(authoringCatalog,'output.resultCount')} value={desiredResultCount} disabled={origin==='exact_trace_prefix'||outputKind==='laffa'} onChange={(value)=>setDesiredResultCount(Math.min(Number(fieldCatalog.fields['output.resultCount'].maximum),Math.max(Number(fieldCatalog.fields['output.resultCount'].minimum),Number(value))))}/>
      <label className="field-label" data-field-path="renderPolicy.videoCount"><span>{catalogFieldLabel(authoringCatalog,'renderPolicy.videoCount')}</span><input type="number" min={Number(fieldCatalog.fields['renderPolicy.videoCount'].minimum)} max={Math.min(Number(fieldCatalog.fields['renderPolicy.videoCount'].maximum),desiredResultCount)} step={1} disabled={renderPolicy.mode==='disabled'} value={Number(renderPolicy.videoCount??0)} aria-describedby="scenario-video-count-help" onChange={(event)=>setRenderPolicy((current)=>({...current,videoCount:Math.min(Number(fieldCatalog.fields['renderPolicy.videoCount'].maximum),desiredResultCount,Math.max(Number(fieldCatalog.fields['renderPolicy.videoCount'].minimum),Number(event.target.value)))}))}/><small id="scenario-video-count-help">{tCatalog(String(fieldCatalog.fields['renderPolicy.videoCount'].helpKey))}</small></label>
    </div></section>
    </div>
  </>;
  const executionPanels = <>
    <section className="panel scenario-actions" id="scenario-execution">
      <div className="scenario-action-buttons">
        <button type="button" className="secondary" aria-describedby="scenario-action-guidance" disabled={busy} onClick={()=>void runRequest('validate')}>{t('workflow.research_validate')}</button>
        <button type="button" className="secondary" aria-describedby="scenario-action-guidance" disabled={busy||!validationFresh} onClick={()=>void runRequest('estimate')}>{t('workflow.research_estimate')}</button>
        <button type="button" className="primary" aria-describedby="scenario-action-guidance" disabled={busy||!validationFresh||!estimateFresh||Boolean(launchBlockReason)||launchStatusBlocked} onClick={()=>void runRequest('run')}>{t('workflow.research_scenario_publish_run')}</button>
      </div>
      <p className="scenario-action-guidance" id="scenario-action-guidance" role="status" aria-live="polite">{actionGuidance}</p>
      {validation ? <section className={structuralInfeasibility?'scenario-diagnostics impossible':validation.valid?'context-proof':'scenario-diagnostics'} role="status" aria-live="polite" aria-atomic="true"><strong>{structuralInfeasibility?diagnosticMessage(structuralInfeasibility):validation.valid?t('workflow.research_validation_ok'):t('workflow.research_validation_failed')}</strong>{structuralInfeasibility?<span><code dir="ltr">{structuralInfeasibility.code}</code>{structuralTheoremId(structuralInfeasibility)?<code dir="ltr">{structuralTheoremId(structuralInfeasibility)}</code>:null}</span>:null}<Diagnostics issues={validation.issues}/></section> : null}
      {validation?.normalizedExactState ? <section className="scenario-exact-state" aria-live="polite"><h3>{t('workflow.research_scenario_exact_state_derived')}</h3><dl><div><dt>{t('workflow.research_scenario_exact_state_id')}</dt><dd dir="ltr">{validation.normalizedExactState.exactStateId}</dd></div><div><dt>{t('workflow.research_scenario_exact_state_phase')}</dt><dd>{validation.normalizedExactState.phase}</dd></div><div><dt>{t('workflow.research_scenario_next_actor')}</dt><dd>{validation.normalizedExactState.nextActor??'—'}</dd></div><div><dt>{t('workflow.research_scenario_legal_actions')}</dt><dd>{validation.normalizedExactState.legalActions.length}</dd></div></dl><code dir="ltr">{validation.normalizedExactState.stateHash}</code></section> : null}
      {estimate ? <EstimatePanel estimate={estimate} ortools={ortools?scenarioResearchV4:undefined} launchBlockReason={launchBlockReason} requestedSolverSeconds={Number(resources.solverTimeoutSeconds)}/> : null}
      <details className="audit-details"><summary>{t('workflow.research_scenario_normalized_preview')}</summary><pre dir="ltr">{(()=>{try{return JSON.stringify(recipePayload(),null,2);}catch{return t('workflow.research_scenario_preview_incomplete');}})()}</pre></details>
    </section>

    {activeRun ? <section className="panel scenario-live-candidates" aria-labelledby="scenario-live-candidates-title">
       <header><div><h3 id="scenario-live-candidates-title">{t('workflow.research_event_progress')}</h3><code dir="ltr">{activeRun.runCode}</code></div><span className={`status ${researchRunDisplayStatus(activeRun)}`}>{researchRunStatusLabel(activeRun)}</span></header>
      <progress aria-label={t('workflow.research_event_progress')} max={liveProgressTarget} value={Math.min(liveProgressTarget,qualifiedReceiptCount)}/>
      <dl>
        <div><dt>{t('workflow.research_run_qualified')}</dt><dd>{number.format(qualifiedReceiptCount)}</dd></div>
        <div><dt>{t('workflow.research_run_rejected')}</dt><dd>{number.format(Number(candidateReceiptCounts.rejected??0))}</dd></div>
        <div><dt>{t('workflow.research_run_duplicate_qualified')}</dt><dd>{number.format(Number(candidateReceiptCounts.duplicate??0))}</dd></div>
        <div><dt>{t('workflow.research_scenario_support_incomplete')}</dt><dd>{number.format(Number(candidateReceiptCounts.incomplete??activeRunProgress.proofIncomplete??0))}</dd></div>
        {activeRunStatus==='running'&&typeof currentSolverCounters.tailCandidates==='number' ? <div><dt>{t('workflow.research_scenario_timing_tail_construction')} · {t('workflow.research_lane_candidates')}</dt><dd>{number.format(currentSolverCounters.tailCandidates)}</dd></div> : null}
        {activeRunStatus==='running'&&typeof currentSolverCounters.proofBoundaryAttempts==='number' ? <div><dt>{t('workflow.research_scenario_timing_proof')} · {t('workflow.research_sequential_attempt')}</dt><dd>{number.format(currentSolverCounters.proofBoundaryAttempts)}</dd></div> : null}
        {activeRunStatus==='running'&&qualitySolverPhaseLabel ? <>
          <div><dt>{t('workflow.research_scenario_timing_proof')}</dt><dd>{qualitySolverPhaseLabel}</dd></div>
          <div><dt>{t('workflow.research_solver_progress_masters')}</dt><dd>{number.format(Number(qualitySolver.masterCount))}</dd></div>
          <div><dt>{t('workflow.research_solver_progress_queries')}</dt><dd>{number.format(Number(qualitySolver.queryCount))}</dd></div>
          <div><dt>{t('workflow.research_solver_progress_bindings')}</dt><dd>{number.format(Number(qualitySolver.bindingCount))}</dd></div>
        </> : null}
        <div><dt>{t('workflow.research_run_duration')}</dt><dd>{number.format(Number(activeRunProgress.elapsedSeconds??0))} {t('workflow.research_seconds')}</dd></div>
        {['queued','running'].includes(activeRunStatus) ? <div><dt>{t('workflow.research_estimate_solver')} · <code dir="ltr">campaign</code> · {t('settings.compiler_coverage_remaining')}</dt><dd>{number.format(Number(activeRunProgress.remainingCampaignSeconds??0))} {t('workflow.research_seconds')}</dd></div> : null}
      </dl>
      {activeRunStatus==='running'&&qualitySolverPhaseLabel ? <p>{t('workflow.research_solver_progress_unproved')}</p> : null}
      {candidateReceipts.length ? <div className="scenario-live-candidate-log" role="log" aria-live="polite" aria-relevant="additions text" aria-label={t('workflow.research_sequential_receipts_title')}>
        {activeRunProgress.candidateReceiptHistoryTruncated===true ? <small>{t('workflow.research_sequential_receipts_page').replace('{shown}',number.format(candidateReceipts.length)).replace('{total}',number.format(Number(activeRunProgress.candidateReceiptCount??candidateReceipts.length)))}</small> : null}
        {candidateReceipts.map((item)=>{const summary=jsonObject(item.candidateSummary);const failure=jsonObject(item.incompleteCompilation);const label={qualified:t('workflow.research_run_qualified'),rejected:t('workflow.research_run_rejected'),duplicate:t('workflow.research_run_duplicate_qualified'),incomplete:t('workflow.research_scenario_support_incomplete')}[item.classification];return <article data-classification={item.classification} key={item.receiptId}><div><b>{label}</b><code dir="ltr">{item.status}</code></div><small>{t('workflow.research_scenario_status')} · <code dir="ltr">{item.stage}</code> · {t('workflow.research_sequential_attempt')} · <code dir="ltr">{item.attempt??'—'}</code></small><small><code dir="ltr">{item.candidateHash??item.variantSemanticHash??'—'}</code>{summary.exactStateId?<> · <code dir="ltr">{String(summary.exactStateId)}</code></>:null}</small>{failure.qualityFailureReason||failure.reason?<small><code dir="ltr">{String(failure.qualityFailureStage??failure.stage??'—')}{failure.qualityFailureCauseStage?` > ${String(failure.qualityFailureCauseStage)}`:''} · {String(failure.qualityFailureReason??failure.reason)}</code></small>:null}</article>;})}
      </div> : null}
    </section> : null}

    <section className="panel scenario-result-facets" id="scenario-results">
      <header><div><h3>{t('workflow.research_scenario_facets_title')}</h3><p>{t('workflow.research_scenario_facets_help')}</p></div><span aria-live="polite">{resultBusy?t('workflow.research_scenario_results_loading'):`${studioResultTotal}`}</span></header>
      <div className="scenario-grid">
        <label className="field-label"><span>{t('workflow.research_scenario_facet_run')}</span><select value={activeRunCode} onChange={(event)=>setActiveRunCode(event.target.value)}><option value="">{t('workflow.research_scenario_facet_all')}</option>{runs.map((run)=><option value={run.runCode} key={run.runCode}>{run.runCode}</option>)}</select></label>
        <label className="field-label"><span>{t('workflow.research_scenario_group_by')}</span><select value={resultGroupBy} onChange={(event)=>setResultGroupBy(event.target.value as ResearchV4ResultFacetName|'')}><option value="">{t('workflow.research_group_none')}</option>{groupFields.map((field)=><option value={resultFacetParameters[field]} key={field}>{facetLabels[field]}</option>)}</select></label>
        {facetFields.map((field)=><label className="field-label" key={field}><span>{facetLabels[field]}</span>{field==='exactStateId'?<input dir="ltr" pattern="STATE-[A-F0-9]{20}" maxLength={26} value={resultFilters.exactStateId??''} onChange={(event)=>setResultFilters((current)=>({...current,exactStateId:event.target.value.toUpperCase()||undefined}))}/>:<select value={resultFilters[field]??''} onChange={(event)=>setResultFilters((current)=>({...current,[field]:event.target.value||undefined}))}><option value="">{t('workflow.research_scenario_facet_all')}</option>{facetOptions[field].map((value)=><option value={value} key={value}>{resultFacetValueLabel(authoringCatalog,value)}</option>)}</select>}</label>)}
      </div>
      {resultGroups.length>0&&<div className="research-server-groups">{resultGroups.map((group)=><span key={group.value}>{resultFacetValueLabel(authoringCatalog,group.value)} <i>{group.count}</i></span>)}</div>}
      {tagControlsReady && lessonTagCatalog ? <LessonTagQueryControl catalog={lessonTagCatalog}
        query={lessonTagFilters} onChange={setLessonTagFilters} message={tCatalog}
        label={t('workflow.research_lesson_tags_title')} help={t('workflow.research_lesson_tags_help')}/> : null}
      <div className="scenario-result-actions"><button type="button" className="quiet" onClick={()=>{setResultFilters({});setLessonTagFilters({});setResultGroupBy('');setActiveRunCode('');setRenderBatch(null);}}>{t('workflow.research_filters_clear')}</button><button type="button" className="primary" disabled={busy||!activeRunCode||Number(renderPolicy.videoCount??0)<1||!normalBatchMediaAllowed} onClick={()=>void prepareSelectedRun()}>{t('workflow.research_scenario_render_selected')}</button>{auditBatchMediaAvailable?<button type="button" className="secondary" disabled={busy||!activeRunCode||Number(renderPolicy.videoCount??0)<1} onClick={()=>void prepareSelectedRun()}>{t('workflow.research_render_audit')}</button>:null}{renderBatchActive?<button type="button" className="quiet-danger" disabled={busy} onClick={()=>void cancelRenderBatch()}>{t('workflow.batch_cancel')}</button>:null}</div>
      {renderBatch ? <p className="context-proof" data-render-status={renderBatchStatus} role="status" aria-live="polite"><strong>{t(renderBatchStatus==='completed'?'workflow.batch_completed_help':renderBatchTerminal?'workflow.research_scenario_render_batch_failed':'workflow.research_scenario_render_batch_started')}</strong><code dir="ltr">{renderBatchJobId||'—'} · {renderBatchStatus||'—'} · {number.format(Number(renderProgress.finished??0))}/{number.format(Number(renderProgress.total??preparedPlayers.length))}</code></p> : null}
      {preparedPlayers.length ? <div className="scenario-result-actions scenario-prepared-players" aria-label={t('workflow.laffa_library_play_challenge')}>{preparedPlayers.map((player)=><a className="button-link secondary" data-testid="scenario-prepared-player-link" href={player.playerHref} target="_blank" rel="noreferrer" key={`${player.ordinal}-${player.playerHref}`}>{t('workflow.batch_item_prefix')}{number.format(player.ordinal+1)} · {t('workflow.laffa_library_play_challenge')}</a>)}</div> : null}
      {renderedMedia.length ? <div className="video-library-grid scenario-render-media">{renderedMedia.map((media,index)=><article className="video-library-card" key={media.basename}><img src={media.posterUrl} alt={t('workflow.label_video_poster')} loading="lazy"/><div className="video-library-copy"><span>{t('workflow.batch_item_prefix')}{number.format(index+1)}</span><b dir="ltr">{media.sourceCode||media.basename}</b><small>{t(media.auditOnly?'workflow.research_render_research_audit':'workflow.batch_item_ready')}</small></div><div className="video-library-actions"><a className="button-link primary" href={media.videoUrl}>{t('workflow.label_open_video')}</a>{media.manifestUrl?<a className="button-link secondary" href={media.manifestUrl}>{t('evidence.m_97b7815c26dc')}</a>:null}{media.playerHref?<a className="button-link secondary" data-testid="scenario-player-link" href={media.playerHref}>{t('workflow.laffa_library_play_challenge')}</a>:null}</div></article>)}</div> : null}
    </section>

    <section className="scenario-expression-grid">
      <section className="panel"><h3>{t('workflow.research_runs_title')}</h3>{runs.length ? <div className="friendly-list">{runs.slice(0,8).map((run)=><button type="button" className="list-item" key={run.runCode} onClick={()=>{setActiveRunCode(run.runCode);onOpenRun(run);}}><i className={`status-dot ${researchRunDisplayStatus(run)}`}/><span className="list-copy"><b dir="ltr">{run.runCode}</b><small>{researchRunStatusLabel(run)} · {scenarioStatusLabel(String(run.scenarioStatus??''))}</small></span><span className="chevron">‹</span></button>)}</div> : <p className="empty-state">{t('workflow.research_empty_runs')}</p>}</section>
      <section className="panel"><h3>{t('workflow.research_results_title')}</h3>{provisionalResults.length?<div className="friendly-list" role="status" aria-live="polite" aria-atomic="true">{provisionalResults.map((result)=>{const candidateHash=String(result.candidateHash??'');return <article className="list-item" key={candidateHash}><i className="status-dot running"/><span className="list-copy"><b>{t('workflow.research_truth_provisional')}</b><code dir="ltr">{candidateHash}</code><small>{String(result.decisionKind??'')} · {String(result.contract??'')} · {String(result.actingSeat??'')}</small><small><span>{t('workflow.research_scenario_exact_state_id')}</span> <code dir="ltr">{String(result.exactStateId??'—')}</code> · <span>{t('workflow.research_scenario_winning_actions')}</span> {String(result.winningRootActionCount??'—')} · <span>{t('workflow.research_scenario_selected_action')}</span> <code dir="ltr">{String(result.selectedAction??'—')}</code></small></span></article>;})}</div>:null}{studioResults.length ? <div className="friendly-list">{studioResults.slice(0,Math.max(desiredResultCount,Number(activeRun?.resultCount??0))).map((result)=>{const projection=jsonObject(result.scenarioProjection);const proof=jsonObject(result.proofOutcome);const compatibility=resultRuntimeCompatibilityStatus(result);const timing=result.executionReceipt?.stageTimings;const tags=tagControlsReady&&lessonTagCatalog?lessonTagResultLabels(lessonTagCatalog,jsonObject(jsonObject(projection.lessonTags).tags)):[];return <button type="button" className="list-item" key={result.resultCode} onClick={()=>onOpenResult(result)}><i className={`status-dot ${String(projection.status??result.disposition)}`}/><span className="list-copy"><b dir="ltr">{result.resultCode}</b><small>{String(projection.decisionKind??'')} · {String(proof.achievedMode??'none')} · {String(proof.completionStatus??'')}</small>{tags.length?<small>{tags.map(tCatalog).join(' · ')}</small>:null}{compatibility!=='legacy'?<small>{tCatalog('research_scenario_runtime_compatibility')} · {tCatalog(`research_scenario_runtime_compatibility_${compatibility}`)}{timing?` · ${Number(timing.total??Object.values(timing).reduce((sum,value)=>sum+Number(value),0)).toFixed(3)}s`:''}</small>:null}</span><span className="chevron">‹</span></button>;})}</div> : provisionalResults.length ? null : <p className="empty-state">{t('workflow.research_empty_results')}</p>}</section>
    </section>
  </>;
  return <section ref={studioRef} className="scenario-studio" aria-labelledby="scenario-studio-title">
    <section className="panel scenario-studio-heading"><div><h2 id="scenario-studio-title">{t('workflow.research_scenario_title')}</h2><p>{t('workflow.research_scenario_help')}</p></div><code dir="ltr">{configuration.sha256.slice(0,16)}</code></section>
    <RuleAuthorityPanel selectedBundle={selectedBundle} officialRulesRevisionId={officialRulesRevisionId} rulesHash={rulesHash} request={request}/>
    {ortools?<OrToolsAuthorityPanel capability={scenarioResearchV4} launchBlockReason={launchBlockReason} unsupportedPredicateIds={unsupportedPredicateIds}/>:null}
    <nav className="journey scenario-journey" aria-label={t('workflow.research_scenario_title')}>
      <a className={`journey-step ${authoredPredicates.length?'done':'active'}`} href="#scenario-authoring" aria-current={!authoredPredicates.length?'step':undefined}><span className="step-number">{authoredPredicates.length?'✓':number.format(1)}</span><span><b>{t('workflow.research_scenario_catalog')}</b><small>{t('workflow.research_scenario_catalog_help')}</small></span></a>
      <a className={`journey-step ${validationFresh?'done':authoredPredicates.length?'active':'locked'}`} href="#scenario-proof" aria-current={authoredPredicates.length&&!validationFresh?'step':undefined}><span className="step-number">{validationFresh?'✓':number.format(2)}</span><span><b>{t('workflow.research_scenario_proof')}</b><small>{proofModeHelp}</small></span></a>
      <a className={`journey-step ${activeRunCode?'done':validationFresh?'active':'locked'}`} href="#scenario-execution" aria-current={validationFresh&&!activeRunCode?'step':undefined}><span className="step-number">{activeRunCode?'✓':number.format(3)}</span><span><b>{t('workflow.research_validate')}</b><small>{actionGuidance}</small></span></a>
      <a className={`journey-step ${activeRunCode?'active':'locked'}`} href="#scenario-results" aria-current={activeRunCode?'step':undefined}><span className="step-number">{number.format(4)}</span><span><b>{t('workflow.research_results_title')}</b><small>{t('workflow.research_scenario_facets_help')}</small></span></a>
    </nav>
    {scenarioAuthoringPanels}
    {proofPanels}
    {executionPanels}
  </section>;
}
