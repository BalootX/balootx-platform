import {useSyncExternalStore} from 'react';

type FoundryCopyProfile = {
  locale: string;
  direction: 'rtl';
  page_title: string;
  groups: Record<string, Record<string, string>>;
};

function readFoundryCopy(): FoundryCopyProfile {
  const node = document.getElementById('foundry-copy');
  if (!node?.textContent) throw new Error('Foundry copy profile is unavailable');
  const value = JSON.parse(node.textContent) as FoundryCopyProfile;
  if (value.locale !== 'ar-SA' || value.direction !== 'rtl' || !value.groups) {
    throw new Error('Foundry copy profile is invalid');
  }
  return value;
}

const foundryCopy = readFoundryCopy();
let selectedResearchWorkflow: Record<string, string> | null = null;
let selectedResearchWorkflowHash: string | null = null;
let researchWorkflowRevision = 0;
const researchWorkflowListeners = new Set<() => void>();

function updateSelectedResearchWorkflow(
  workflow: Record<string, string> | null,
  hash: string | null,
): void {
  if (selectedResearchWorkflowHash === hash) return;
  selectedResearchWorkflow = workflow;
  selectedResearchWorkflowHash = hash;
  researchWorkflowRevision += 1;
  researchWorkflowListeners.forEach((listener) => listener());
}

export function useResearchWorkflowCopyRevision(): number {
  return useSyncExternalStore(
    (listener) => {
      researchWorkflowListeners.add(listener);
      return () => researchWorkflowListeners.delete(listener);
    },
    () => researchWorkflowRevision,
    () => researchWorkflowRevision,
  );
}

export function selectResearchWorkflowCopy(
  value: {
    schemaVersion: number;
    bundleRevisionId: string;
    productRevisionId: string;
    productProfileHash: string;
    locale: string;
    direction: string;
    groups: {workflow: Record<string, string>};
    sha256: string;
  } | null,
  expectedBundleRevisionId?: string,
): void {
  if (value === null) {
    updateSelectedResearchWorkflow(null, null);
    return;
  }
  const workflow = value.groups?.workflow;
  if (
    value.schemaVersion !== 1
    || value.locale !== 'ar-SA'
    || value.direction !== 'rtl'
    || value.bundleRevisionId !== expectedBundleRevisionId
    || !value.productRevisionId
    || !/^[0-9a-f]{64}$/.test(value.productProfileHash)
    || !/^[0-9a-f]{64}$/.test(value.sha256)
    || !workflow
    || Object.keys(workflow).length === 0
    || Object.entries(workflow).some(([key, text]) => !key || !text)
  ) throw new Error('Selected Research copy profile is invalid');
  updateSelectedResearchWorkflow(workflow, value.sha256);
}

function lookupMessage(group: string, key: string): string | undefined {
  if (group === 'workflow' && selectedResearchWorkflow) {
    return selectedResearchWorkflow[key];
  }
  return foundryCopy.groups[group]?.[key];
}

export function hasFoundryCopy(key: string): boolean {
  const separator = key.indexOf('.');
  if (separator < 1) return false;
  return Boolean(lookupMessage(key.slice(0, separator), key.slice(separator + 1)));
}

export function t(key: string): string {
  const separator = key.indexOf('.');
  const group = separator > 0 ? key.slice(0, separator) : '';
  const message = separator > 0 ? key.slice(separator + 1) : '';
  const value = lookupMessage(group, message);
  if (!value) throw new Error(`Foundry copy key is missing: ${key}`);
  return value;
}

/** Profiles published before the timeout key keep the generic request failure. */
export function requestTimeoutMessage(): string {
  return hasFoundryCopy('helpers.request_timeout')
    ? t('helpers.request_timeout')
    : `${t('helpers.m_dc85c66d7c4d')}408)`;
}

export function tCatalog(key: string): string {
  const normalized = key.includes('.') ? key : `workflow.${key}`;
  const separator = normalized.indexOf('.');
  const group = normalized.slice(0, separator);
  const message = normalized.slice(separator + 1);
  const value = lookupMessage(group, message);
  if (!value) throw new Error(`Foundry catalog copy key is missing: ${key}`);
  return value;
}

export function tCatalogWithFallback(key: string, fallbackKey: string): string {
  const normalized = key.includes('.') ? key : `workflow.${key}`;
  const separator = normalized.indexOf('.');
  const group = normalized.slice(0, separator);
  const message = normalized.slice(separator + 1);
  return lookupMessage(group, message) ?? tCatalog(fallbackKey);
}

const foundryCopyVariants = {
  'workflow.research_information_set': [
    'exact_information_set',
    'paired_sampled_information_set',
    'incomplete',
  ],
  'workflow.research_sequential_response': [
    'exact_sequential_response',
    'robust_policy_conditioned_response',
    'incomplete',
  ],
  'workflow.research_render': ['content_candidate', 'research_audit'],
} as const;

export function tVariant(
  family: keyof typeof foundryCopyVariants,
  variant: string,
): string {
  const allowed = foundryCopyVariants[family] as readonly string[];
  if (!allowed.includes(variant)) {
    throw new Error(`Foundry copy variant is invalid: ${family}.${variant}`);
  }
  return t(`${family}_${variant}`);
}

export const number = new Intl.NumberFormat('ar-SA-u-nu-latn');
export const date = new Intl.DateTimeFormat('ar-SA-u-nu-latn', {
  dateStyle: 'medium',
  timeStyle: 'short',
});
