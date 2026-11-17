import type {ResearchRecipeApi} from '../features/research/recipes';
import type {
  ResearchRecipe,
  ResearchPlayingQualityAuditV1,
  ResearchResultOperatorAuditV1,
  ResearchRun,
  ScenarioResearchLocalizationV1,
} from '../features/research/model';
import {isOrToolsResearchBackend} from '../features/research/model';
import type {JsonObject} from '../shared/json';
import type {Client} from './generated/client';
import {client as defaultClient} from './generated/client.gen';
import {
  createResearchRecipeApiV2ResearchRecipesPost,
  createResearchRunApiV2ResearchRunsPost,
  getResearchRecipeApiV2ResearchRecipesCodeGet,
  listResearchRecipesApiV2ResearchRecipesGet,
  retireResearchRecipeApiV2ResearchRecipesCodeRetirePost,
} from './generated/sdk.gen';
import type {
  ResearchRecipeListResponse,
  ResearchRecipeV4CreateRequest,
  ResearchRunResponse,
  StoredResearchRecipeResponse,
  StoredResearchRecipeV2Response,
  StoredResearchRecipeV3Response,
  StoredResearchRecipeV4Response,
} from './generated/types.gen';

type StoredRecipe =
  | StoredResearchRecipeResponse
  | StoredResearchRecipeV2Response
  | StoredResearchRecipeV3Response
  | StoredResearchRecipeV4Response;

type GeneratedResult<T> = {
  data: T | undefined;
  error?: unknown;
  response?: Response;
};

export class ResearchApiError extends Error {
  readonly operation: string;
  readonly status: number | null;
  readonly detail: unknown;

  constructor(operation: string, status: number | null, detail: unknown) {
    super(errorMessage(operation, status, detail));
    this.name = 'ResearchApiError';
    this.operation = operation;
    this.status = status;
    this.detail = detail;
  }
}

export function createResearchRecipeApi(
  apiClient: Client = defaultClient,
): ResearchRecipeApi {
  return {
    async listRecipes({status, cursor, limit, signal}) {
      const response = await listResearchRecipesApiV2ResearchRecipesGet({
        client: apiClient,
        query: {status, ...(cursor === undefined ? {} : {cursor}), limit},
        signal,
      });
      const page = unwrap<ResearchRecipeListResponse>(
        response,
        'list_research_recipes',
      );
      return {
        items: page.items.map(normalizeRecipe),
        nextCursor: page.nextCursor,
      };
    },

    async getRecipe({code, signal}) {
      const response = await getResearchRecipeApiV2ResearchRecipesCodeGet({
        client: apiClient,
        path: {code},
        signal,
      });
      return normalizeRecipe(unwrap<StoredRecipe>(response, 'get_research_recipe'));
    },

    async createRecipe({payload, idempotencyKey}) {
      const response = await createResearchRecipeApiV2ResearchRecipesPost({
        client: apiClient,
        body: researchCreateRequest(payload),
        headers: {'Idempotency-Key': idempotencyKey},
      });
      return normalizeRecipe(
        unwrap<StoredRecipe>(
          response,
          'create_research_recipe',
        ),
      );
    },

    async createRun({recipeCode, idempotencyKey}) {
      const response = await createResearchRunApiV2ResearchRunsPost({
        client: apiClient,
        body: {recipeCode},
        headers: {'Idempotency-Key': idempotencyKey},
      });
      return normalizeRun(
        unwrap<ResearchRunResponse>(response, 'create_research_run'),
      );
    },

    async retireRecipe({code, idempotencyKey}) {
      const response = await retireResearchRecipeApiV2ResearchRecipesCodeRetirePost({
        client: apiClient,
        headers: {'Idempotency-Key': idempotencyKey},
        path: {code},
      });
      return normalizeRecipe(
        unwrap<StoredRecipe>(response, 'retire_research_recipe'),
      );
    },
  };
}

export const researchRecipeApi = createResearchRecipeApi();

export async function getResearchResultOperatorAudit(
  resultCode: string,
  apiClient: Client = defaultClient,
): Promise<ResearchResultOperatorAuditV1> {
  const response = await apiClient.get<{200: unknown}>({
    url: `/api/v2/research-results/${encodeURIComponent(resultCode)}/operator-audit`,
  });
  return decodeOperatorAudit(
    unwrap<unknown>(response, 'get_research_result_operator_audit'),
    resultCode,
  );
}

export async function getResearchResultQualityAudit(
  resultCode: string,
  apiClient: Client = defaultClient,
): Promise<ResearchPlayingQualityAuditV1> {
  const response = await apiClient.get<{200: unknown}>({
    url: `/api/v2/research-results/${encodeURIComponent(resultCode)}/quality-audit`,
  });
  return decodeQualityAudit(
    unwrap<unknown>(response, 'get_research_result_quality_audit'),
    resultCode,
  );
}

export async function getScenarioResearchLocalization(
  bundleRevisionId: string,
  apiClient: Client = defaultClient,
): Promise<ScenarioResearchLocalizationV1> {
  const response = await apiClient.get<{200: unknown}>({
    url: `/api/v2/domain-capabilities?bundleRevisionId=${encodeURIComponent(bundleRevisionId)}`,
  });
  const value = unwrap<unknown>(response, 'get_scenario_research_localization');
  const research = value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>).research
    : undefined;
  const capability = research && typeof research === 'object' && !Array.isArray(research)
    ? (research as Record<string, unknown>).scenarioResearchV4
    : undefined;
  const localization = capability && typeof capability === 'object' && !Array.isArray(capability)
    ? (capability as Record<string, unknown>).localization
    : undefined;
  if (
    !localization
    || typeof localization !== 'object'
    || Array.isArray(localization)
    || (localization as Record<string, unknown>).bundleRevisionId !== bundleRevisionId
  ) {
    throw new ResearchApiError(
      'decode_scenario_research_localization',
      null,
      'invalid_scenario_research_localization',
    );
  }
  return localization as ScenarioResearchLocalizationV1;
}

function unwrap<T>(result: GeneratedResult<T>, operation: string): T {
  if (result.data !== undefined) return result.data;
  throw new ResearchApiError(
    operation,
    result.response?.status ?? null,
    result.error,
  );
}

function researchCreateRequest(
  payload: JsonObject,
): ResearchRecipeV4CreateRequest {
  if (payload.schemaVersion !== 4) {
    throw new ResearchApiError(
      'create_research_recipe',
      null,
      'schema_version_must_be_4',
    );
  }
  return payload as unknown as ResearchRecipeV4CreateRequest;
}

function normalizeRecipe(value: StoredRecipe): ResearchRecipe {
  if (
    typeof value.code !== 'string' ||
    typeof value.sha256 !== 'string' ||
    !['draft', 'published', 'retired'].includes(value.lifecycleStatus)
  ) {
    throw new ResearchApiError(
      'decode_research_recipe',
      null,
      'invalid_recipe_response',
    );
  }
  return {
    ...value,
    status: value.status ?? value.lifecycleStatus,
  } as unknown as ResearchRecipe;
}

function normalizeRun(value: ResearchRunResponse): ResearchRun {
  if (
    typeof value.runCode !== 'string' ||
    typeof value.recipeCode !== 'string' ||
    typeof value.status !== 'string'
  ) {
    throw new ResearchApiError(
      'decode_research_run',
      null,
      'invalid_run_response',
    );
  }
  return value as ResearchRun;
}

function decodeOperatorAudit(
  value: unknown,
  resultCode: string,
): ResearchResultOperatorAuditV1 {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ResearchApiError(
      'decode_research_result_operator_audit',
      null,
      'invalid_operator_audit_response',
    );
  }
  const audit = value as Record<string, unknown>;
  const authority = audit.researchAuthority as Record<string, unknown> | undefined;
  const compatibility = audit.runtimeCompatibility as Record<string, unknown> | undefined;
  const receipt = audit.executionReceipt as Record<string, unknown> | undefined;
  if (
    audit.schemaVersion !== 1
    || audit.resultCode !== resultCode
    || authority?.schemaVersion !== 1
    || !isOrToolsResearchBackend(authority.backendId)
    || authority.qualificationAuthority !== 'cp_sat'
    || compatibility?.schemaVersion !== 1
    || !['not_checked', 'passed', 'failed'].includes(String(compatibility.status))
    || receipt?.schemaVersion !== 1
  ) {
    throw new ResearchApiError(
      'decode_research_result_operator_audit',
      null,
      'invalid_operator_audit_response',
    );
  }
  return audit as ResearchResultOperatorAuditV1;
}

export function decodeQualityAudit(
  value: unknown,
  resultCode: string,
): ResearchPlayingQualityAuditV1 {
  const fail = () => {
    throw new ResearchApiError(
      'decode_research_result_quality_audit',
      null,
      'invalid_quality_audit_response',
    );
  };
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail();
  const audit = value as Record<string, unknown>;
  const hashes = ['resultHash', 'qualityReceiptHash', 'completeTraceHash', 'sha256'];
  const kinds = ['bid', 'escalation', 'project', 'play'] as const;
  const counts = audit.countsByKind as Record<string, unknown> | undefined;
  const rows = Array.isArray(audit.decisions) ? audit.decisions : [];
  const featured = audit.featuredPlayProof as Record<string, unknown> | undefined;
  if (featured !== undefined && featured !== null) {
    if (typeof featured !== 'object' || Array.isArray(featured)) fail();
    const hidden = featured.hiddenWorlds as Record<string, unknown> | undefined;
    const completeState = featured.proofModelKind === 'native_complete_state_root_outcome_v1';
    const exactActor = featured.proofModelKind === 'actor_information_set_cegis_model_v1';
    if (
      Object.keys(featured).sort().join(',') !== ['exactStateId', 'playIndex', 'actor', 'action', 'proofArtifactHash', 'proofModelKind', 'completionStatus', ...(completeState ? [] : ['hiddenWorlds'])].sort().join(',')
      || !/^STATE-[A-F0-9]{20}$/.test(String(featured.exactStateId || ''))
      || !Number.isInteger(featured.playIndex) || Number(featured.playIndex) < 0 || Number(featured.playIndex) > 31
      || !['north', 'east', 'south', 'west'].includes(String(featured.actor))
      || typeof featured.action !== 'string' || !featured.action
      || !/^[0-9a-f]{64}$/.test(String(featured.proofArtifactHash || ''))
      || (completeState ? featured.completionStatus !== 'exact_complete' : (
      (!exactActor && featured.proofModelKind !== 'native_observed_root_outcome_v1')
      || featured.completionStatus !== (exactActor ? 'exact_complete' : 'bounded_complete')
      || !hidden || typeof hidden !== 'object' || Array.isArray(hidden)
      || Object.keys(hidden).sort().join(',') !== ['method', 'count', 'complete'].sort().join(',')
      || hidden.method !== (exactActor ? 'exhaustive_cp_sat_hidden_allocations_v1' : 'supplied_worlds_only')
      || !Number.isInteger(hidden.count) || Number(hidden.count) < 1
      || hidden.complete !== exactActor))
    ) fail();
  }
  const count = Number(audit.decisionCount);
  if (
    audit.schemaVersion !== 1
    || audit.resultCode !== resultCode
    || audit.scope !== 'all_decisions'
    || audit.status !== 'complete'
    || audit.wholeRoundCertified !== false
    || !['generate', 'audit_recorded', 'audit_then_generate'].includes(String(audit.operation))
    || hashes.some((key) => !/^[0-9a-f]{64}$/.test(String(audit[key] || '')))
    || !counts
    || Object.keys(counts).sort().join(',') !== [...kinds].sort().join(',')
    || !Number.isInteger(count)
    || count < 0
    || rows.length !== count
  ) fail();
  const validRows = rows.every((raw, index) => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return false;
    const row = raw as Record<string, unknown>;
    return row.ordinal === index + 1
      && kinds.includes(row.kind as typeof kinds[number])
      && ['north', 'east', 'south', 'west'].includes(String(row.actor))
      && typeof row.action === 'string' && row.action.length > 0
      && Number.isInteger(row.legalActionCount) && Number(row.legalActionCount) >= 1
      && typeof row.actualActionOptimal === 'boolean'
      && Number.isFinite(row.regret) && Number(row.regret) >= 0
      && typeof row.selectionReason === 'string' && row.selectionReason.length > 0
      && ['exact', 'bounded'].includes(String(row.proofMode))
      && ['complete_state', 'actor_information_set'].includes(String(row.proofVisibility))
      && row.proofCompletionStatus === (row.proofMode === 'exact' ? 'exact_complete' : 'bounded_complete')
      && typeof row.proofScope === 'string' && /^[a-z0-9_]{1,100}$/.test(row.proofScope)
      && ['proofArtifactHash', 'actionQualityHash', 'sha256']
        .every((key) => /^[0-9a-f]{64}$/.test(String(row[key] || '')));
  });
  const kindTotal = kinds.reduce((total, kind) => {
    const value = Number(counts?.[kind]);
    return Number.isInteger(value) && value >= 0 ? total + value : Number.NaN;
  }, 0);
  const optimal = rows.filter((row) => (row as Record<string, unknown>).actualActionOptimal === true).length;
  const zeroRegret = rows.filter((row) => Number((row as Record<string, unknown>).regret) === 0).length;
  if (
    !validRows
    || kindTotal !== count
    || audit.optimalDecisionCount !== optimal
    || audit.zeroRegretDecisionCount !== zeroRegret
  ) fail();
  return audit as ResearchPlayingQualityAuditV1;
}

function errorMessage(
  operation: string,
  status: number | null,
  detail: unknown,
): string {
  if (
    detail &&
    typeof detail === 'object' &&
    'detail' in detail
  ) {
    const responseDetail = detail.detail;
    if (typeof responseDetail === 'string' && responseDetail) {
      return responseDetail;
    }
    if (Array.isArray(responseDetail)) {
      const issue = responseDetail.find(
        (item) => item && typeof item === 'object' && 'msg' in item,
      );
      if (issue && typeof issue === 'object' && typeof issue.msg === 'string') {
        return issue.msg;
      }
    }
  }
  return `${operation}:${status ?? 'network'}`;
}
