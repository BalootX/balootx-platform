import type {ReactNode} from 'react';

import {humanKey} from './format';
import type {JsonObject} from './json';
import {hasFoundryCopy, number, t} from './localization';
import {researchRunDisplayStatus, type ResearchRun} from '../features/research/model';

export type DrawerState = {
  kind: string;
  title: string;
  subtitle?: string;
  content: ReactNode;
  audit?: JsonObject;
} | null;

export const statusLabel: Record<string, string> = {
  queued: t('domain_labels.m_279963953fb4'),
  running: t('domain_labels.m_d761119224ab'),
  completed: t('domain_labels.m_c2da5684d63b'),
  failed: t('domain_labels.m_d506fdf79c68'),
  cancelled: t('domain_labels.m_616d302cb016'),
  exhausted: t('workflow.batch_status_exhausted'),
  published: t('domain_labels.m_74f0d5710a99'),
  draft: t('domain_labels.m_552aec56f591'),
  superseded: t('domain_labels.m_b701c7b36d61'),
  approved: t('domain_labels.m_ef0a0d80aedd'),
  review: t('domain_labels.m_8aac5fac1498'),
};

export function researchRunStatusLabel(run: ResearchRun): string {
  const status = researchRunDisplayStatus(run);
  switch (status) {
    case 'incomplete': return t('workflow.research_sequential_stop_incomplete');
    case 'structurally_infeasible': return t('workflow.research_sequential_continuation_infeasible');
    case 'exhausted': return t('workflow.research_sequential_stop_candidates_exhausted');
    case 'qualified': return t('workflow.research_sequential_stop_qualified');
    default: return statusLabel[status] || status;
  }
}

export function scenarioStatusLabel(status: string): string {
  // Frozen older bundles retain their own copy and may predate these labels.
  if (!hasFoundryCopy(`workflow.research_scenario_diagnostic_${status}`)) return status;
  switch (status) {
    case 'schema_invalid': return t('workflow.research_scenario_diagnostic_schema_invalid');
    case 'rules_inapplicable': return t('workflow.research_scenario_diagnostic_rules_inapplicable');
    case 'unsupported_predicate': return t('workflow.research_scenario_diagnostic_unsupported_predicate');
    case 'contradictory_constraints': return t('workflow.research_scenario_diagnostic_contradictory_constraints');
    case 'structurally_infeasible': return t('workflow.research_scenario_diagnostic_structurally_infeasible');
    case 'not_found_within_budget': return t('workflow.research_scenario_diagnostic_not_found_within_budget');
    case 'construction_target_miss': return t('workflow.research_scenario_diagnostic_construction_target_miss');
    case 'semantic_postcondition_miss': return t('workflow.research_scenario_diagnostic_semantic_postcondition_miss');
    case 'proof_incomplete': return t('workflow.research_scenario_diagnostic_proof_incomplete');
    case 'proof_failed': return t('workflow.research_scenario_diagnostic_proof_failed');
    case 'satisfied': return t('workflow.research_scenario_diagnostic_satisfied');
    default: return status;
  }
}

export const seatLabel: Record<string, string> = {
  north: t('domain_labels.m_5cbbfccea6e8'),
  east: t('domain_labels.m_26b73a5271c1'),
  south: t('domain_labels.m_390e936c0d6c'),
  west: t('domain_labels.m_64141a180741'),
};

export const decisionLabel: Record<string, string> = {
  play: t('domain_labels.m_e974843e20f0'),
  bid: t('domain_labels.m_d293ea4d0fa1'),
  project: t('domain_labels.m_4a2b9a3f34b5'),
  pass: t('domain_labels.m_d9393a6726e5'),
  escalation: t('domain_labels.m_a451fdf03770'),
};

export const truthLabel: Record<string, string> = {
  exact_optimal: t('domain_labels.m_2e2b14a6747d'),
  robust_best_known: t('domain_labels.m_ec905ea7bec8'),
  debatable: t('domain_labels.m_d7b4e4e4748b'),
  provisional: t('workflow.research_truth_provisional'),
  rejected: t('workflow.research_status_rejected'),
};

export const disagreementLabel: Record<string, string> = {
  compiler_parity_defect: t('workflow.research_quarantine_compiler_parity'),
  runtime_transition_defect: t('workflow.research_quarantine_runtime_transition'),
  hard_constraint_violation: t('workflow.research_quarantine_hard_constraint'),
  observer_information_defect: t('workflow.research_quarantine_observer_information'),
  historical_record_requires_readjudication: t('workflow.research_quarantine_historical'),
  constraint_mismatch: t('workflow.research_quarantine_constraint_mismatch'),
  legality_mismatch: t('workflow.research_quarantine_legality_mismatch'),
  score_mismatch: t('workflow.research_quarantine_score_mismatch'),
  optimality_mismatch: t('workflow.research_quarantine_optimality_mismatch'),
  provenance_mismatch: t('workflow.research_quarantine_provenance_mismatch'),
};

export const laffaLabel: Record<string, string> = {
  single_perspective_masterpiece: t('domain_labels.m_861b54ad9bc3'),
  two_seat_duel: t('domain_labels.m_5384115427c2'),
  team_coordination: t('domain_labels.m_c13802606042'),
  multi_angle_laffa: t('domain_labels.m_53fcca186659'),
  four_seat_pressure: t('domain_labels.m_e1005716e4cd'),
  critical_sakka_laffa: t('domain_labels.m_2b9352f50e94'),
};

export const laffaSourceLabel: Record<string, string> = {
  laffa: t('workflow.laffa_library_source_archive'),
  archived_game: t('workflow.laffa_library_source_archive'),
  sakka_round: t('workflow.laffa_library_source_sakka_round'),
  research_result: t('workflow.laffa_library_source_research'),
  qualified_laffa: t('workflow.laffa_library_source_qualified'),
  research_audit_trace: t('workflow.laffa_library_source_research_audit'),
};

export const laffaQualificationLabel: Record<string, string> = {
  qualified: t('workflow.laffa_library_qualification_qualified'),
  rejected: t('workflow.laffa_library_qualification_rejected'),
  provisional: t('workflow.laffa_library_qualification_provisional'),
  unqualified: t('workflow.laffa_library_qualification_unqualified'),
  quarantined: t('workflow.research_quarantine_title'),
  legacy_replay: t('workflow.laffa_library_qualification_audit'),
};

export const contractLabel: Record<string, string> = {
  hokum: t('configuration_families.m_9d151dc785f8'),
  sun: t('configuration_families.m_e2aa47421204'),
  ashkal: t('configuration_families.m_68ba4e657a64'),
};

export const suitName: Record<string, string> = {
  spades: t('configuration_fields.label_suit_spades'),
  hearts: t('configuration_fields.label_suit_hearts'),
  diamonds: t('configuration_fields.label_suit_diamonds'),
  clubs: t('configuration_fields.label_suit_clubs'),
};

const suitSymbol: Record<string, string> = {
  spades: '\u2660',
  hearts: '\u2665',
  diamonds: '\u2666',
  clubs: '\u2663',
};

export const projectLabel: Record<string, string> = {
  none: t('configuration_families.m_5f813ffa9ddc'),
  siri: t('configuration_families.m_0a16ea3020b1'),
  fifty: t('configuration_families.m_0e03c52f7552'),
  hundred: t('configuration_families.m_651541d79bc9'),
  hundred_sequence: t('configuration_families.m_651541d79bc9'),
  hundred_four_kind: t('configuration_families.m_651541d79bc9'),
  four_hundred: t('configuration_families.m_0a3f542c78c4'),
};

export const familyMeta: Record<string, {label: string; description: string}> = {
  rules: {label: t('configuration_families.m_9df238db73a6'), description: t('configuration_families.m_e626ae33a44e')},
  agents: {label: t('configuration_families.m_34018f2ff7af'), description: t('configuration_families.m_544d25195dc4')},
  analysis: {label: t('configuration_families.m_561ad79c7de4'), description: t('configuration_families.m_411d21d48fc2')},
  challenge: {label: t('configuration_families.m_3fc01d823013'), description: t('configuration_families.m_f44dbe1bebf3')},
  challenge_contracts: {label: t('configuration_families.m_403959dd9ea1'), description: t('configuration_families.m_c0910af7e1ee')},
  video: {label: t('configuration_families.m_8a4cbedde18f'), description: t('configuration_families.m_433fbeb40e2a')},
  conventions: {label: t('configuration_tokens.m_d4892ff648ab'), description: t('configuration_tokens.m_15041d2d4627')},
  difficulty: {label: t('configuration_tokens.m_53d2c6d3ac20'), description: t('configuration_tokens.m_9797d73d199c')},
  performance: {label: t('configuration_tokens.m_1264c71a3c12'), description: t('configuration_tokens.m_828e51c24129')},
  security: {label: t('configuration_tokens.m_976adf68f49a'), description: t('configuration_tokens.m_0f146e819801')},
  runtime: {label: t('configuration_tokens.m_d1e21f47730f'), description: t('configuration_tokens.m_dfafdc2f6829')},
  storage: {label: t('configuration_tokens.m_f08625064ac0'), description: t('configuration_tokens.m_9aacc00f0357')},
  product: {label: t('configuration_tokens.m_f35ea8df438e'), description: t('configuration_tokens.m_14ccae7b3065')},
  strategy: {label: t('configuration_tokens.m_418e914a3153'), description: t('configuration_tokens.m_e6ad899ac92f')},
  belief: {label: t('configuration_tokens.m_e5c7c81df095'), description: t('configuration_tokens.m_2f5806562eba')},
  match_value: {label: t('configuration_tokens.m_10a21c82ab8e'), description: t('configuration_tokens.m_a5256e5cd53a')},
  hitl: {label: t('configuration_tokens.m_7b1173d9f446'), description: t('configuration_tokens.m_3bd74f58fcab')},
  audit: {label: t('configuration_tokens.m_3a82b23a1d3f'), description: t('configuration_tokens.m_935e84461e3c')},
  benchmark: {label: t('configuration_tokens.m_bfb34daee02d'), description: t('configuration_tokens.m_b309ae74c3bf')},
  certification: {label: t('configuration_tokens.m_774dd7ff22d8'), description: t('configuration_tokens.m_dd8a3703c2f4')},
  configuration: {label: t('configuration_tokens.m_62f57c70a7d1'), description: t('configuration_tokens.m_43ff63cda816')},
  database: {label: t('configuration_tokens.m_430e2c93e7bb'), description: t('configuration_tokens.m_60c55a97a166')},
  duplicate: {label: t('configuration_tokens.m_b4b8bbea25ab'), description: t('configuration_tokens.m_b27cf47bcc20')},
  game: {label: t('configuration_tokens.m_cc63a0ba000c'), description: t('configuration_tokens.m_2b45f6eb81f7')},
  history: {label: t('configuration_tokens.m_9dca2d96d1fb'), description: t('configuration_tokens.m_aabe13c3b665')},
  identity: {label: t('configuration_tokens.m_c447ac426926'), description: t('configuration_tokens.m_5050b4ac85b9')},
  learning: {label: t('configuration_tokens.m_4d33dfc911a6'), description: t('configuration_tokens.m_e62c55485f8a')},
  live: {label: t('configuration_tokens.m_94ebc0bc9e48'), description: t('configuration_tokens.m_172f9b73c29d')},
  ml: {label: t('configuration_tokens.m_3775f0ba89f3'), description: t('configuration_tokens.m_d0d43450dd9c')},
  operations: {label: t('configuration_tokens.m_c7a49bb5acfd'), description: t('configuration_tokens.m_3a3ccd19dca7')},
  shared: {label: t('configuration_tokens.m_80635788cc18'), description: t('configuration_tokens.m_3866b0548573')},
  adjudication: {label: t('configuration_tokens.m_31cc71b429fe'), description: t('configuration_tokens.m_2ccd11c54f4f')},
};

export function tacticDisplayName(kind: string): string {
  const labels: Record<string, string> = {
    only_winning_move: t('workflow.research_tactic_only_move'),
    small_card_big_gain: t('workflow.research_tactic_small_gain'),
    large_card_sacrifice: t('workflow.research_tactic_sacrifice'),
    kabout_creation: t('workflow.research_tactic_kabout'),
    kabout_preservation: t('workflow.research_tactic_kabout_preservation'),
    kabout_prevention: t('workflow.research_tactic_kabout_prevention'),
    contract_rescue: t('workflow.research_tactic_contract_rescue'),
    entry_unblocking: t('workflow.research_tactic_entry'),
    trump_control: t('workflow.research_tactic_trump'),
    partner_enablement: t('workflow.research_tactic_partner'),
    project_bidding: t('workflow.research_tactic_project'),
    debate: t('workflow.research_tactic_debate'),
  };
  return labels[kind] || kind;
}

export function actionLabel(action?: JsonObject): string {
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
    return `${t('workflow.research_project_declared')} ${projectLabel[projectKind] || projectKind}`.trim();
  }
  if (eventType === 'projects_revealed') {
    const projects = Array.isArray(action.projects)
      ? action.projects as JsonObject[]
      : [];
    const names = Array.from(new Set(projects
      .map((item) => projectLabel[String(item.kind)] || String(item.kind))
      .filter(Boolean)));
    return `${t('workflow.research_projects_revealed')} ${names.join(' · ')}`.trim();
  }
  if (eventType === 'baloot_declared' || eventType === 'baloot_included_in_project') {
    return t('workflow.research_baloot_event');
  }
  const bid = actionName === 'pass'
    ? decisionLabel.pass
    : contractLabel[actionName] || '';
  const bidWithSuit = actionName === 'hokum' && trumpSuit
    ? `${bid} ${suitSymbol[trumpSuit] || trumpSuit}`
    : bid;
  return label
    || card
    || code
    || contractLabel[contract]
    || contract
    || bidWithSuit
    || actionName
    || decisionLabel[kind]
    || t('workflow.m_5e67642412a2');
}

export function Stat({label, value}: {label: string; value: string}) {
  return <div><span>{label}</span><b>{value}</b></div>;
}

export function Empty({title, body}: {title: string; body: string}) {
  return <div className="empty-state"><b>{title}</b><span>{body}</span></div>;
}

export function InfoBadge({
  id,
  text,
  example,
  technical,
  openHelp,
  setOpenHelp,
}: {
  id: string;
  text: string;
  example: string;
  technical?: string;
  openHelp: string | null;
  setOpenHelp: (value: string | null) => void;
}) {
  const open = openHelp === id;
  return <span className={`info-wrap ${open ? 'open' : ''}`} onMouseLeave={() => open && setOpenHelp(null)}><button type="button" className="info-badge" aria-label={t('evidence.m_dd322d284113')} aria-expanded={open} onClick={(event) => {event.preventDefault(); event.stopPropagation(); setOpenHelp(open ? null : id);}}>{t('evidence.m_7d06b69aad65')}</button><span className="info-popover" role="tooltip"><b>{t('evidence.m_3ae78c1d5d3c')}</b><span>{text}</span><em>{example}</em>{technical && <small dir="ltr">{t('evidence.m_e2ee0cec1cb8')}{technical}</small>}</span></span>;
}

export function ScorePill({value, tone = ''}: {value: number; tone?: string}) {
  return <span className={`score-pill ${tone}`}>{number.format(Math.round(value))}</span>;
}

export function JsonSummary({value}: {value: JsonObject}) {
  return <div className="json-summary">{Object.entries(value).slice(0, 20).map(([key, item]) => <div key={key}><span>{humanKey(key)}</span><b>{typeof item === 'object' ? t('evidence.m_032f46c959bf') : String(item)}</b></div>)}</div>;
}
