import type {JsonObject, JsonValue} from '../../shared/json';
import type {
  ResearchRecipe,
  ResearchRecipeLifecycleStatus,
  ResearchRun,
} from './model';

export type ResearchRecipePage = {
  items: ResearchRecipe[];
  nextCursor: string | null;
};

export type ResearchRecipeListRequest = {
  status: ResearchRecipeLifecycleStatus;
  cursor?: string;
  limit: number;
  signal?: AbortSignal;
};

export interface ResearchRecipeApi {
  listRecipes(request: ResearchRecipeListRequest): Promise<ResearchRecipePage>;
  getRecipe(request: {
    code: string;
    signal?: AbortSignal;
  }): Promise<ResearchRecipe>;
  createRecipe(request: {
    payload: JsonObject;
    idempotencyKey: string;
  }): Promise<ResearchRecipe>;
  createRun(request: {
    recipeCode: string;
    idempotencyKey: string;
  }): Promise<ResearchRun>;
  retireRecipe(request: {
    code: string;
    idempotencyKey: string;
  }): Promise<ResearchRecipe>;
}

export type ResearchRecipeSnapshot = {
  status: ResearchRecipeLifecycleStatus;
  items: readonly ResearchRecipe[];
  nextCursor: string | null;
  loaded: boolean;
  loading: boolean;
  error: unknown | null;
};

export type ResearchRecipeControllerOptions = {
  pageSize?: number;
  sessionStorage?: ResearchRecipeSessionStorage | null;
  crypto?: ResearchRecipeUuidSource | null;
};

export type CreateRecipeAndRunResult = {
  recipe: ResearchRecipe;
  run: ResearchRun;
};

export type ResearchRecipeSessionStorage = Pick<
  Storage,
  'getItem' | 'setItem' | 'removeItem'
>;
export type ResearchRecipeUuidSource = {randomUUID(): string};
type Listener = () => void;

type PageFlight = {
  scope: number;
  promise: Promise<ResearchRecipeSnapshot>;
};

type StoredCreateRunOperation = {
  schemaVersion: 1;
  canonicalPayload: string;
  createIdempotencyKey: string;
  runIdempotencyKey: string;
};

type StoredRetireOperation = {
  schemaVersion: 1;
  code: string;
  idempotencyKey: string;
};

const CREATE_RUN_STORAGE_KEY = 'balootx.research.recipe-create-run.v1';
const RETIRE_STORAGE_PREFIX = 'balootx.research.recipe-retire.v1:';
const DEFAULT_PAGE_SIZE = 100;
const MAX_PAGE_SIZE = 500;

export type ResearchRecipeErrorCode =
  | 'stale_scope'
  | 'persistence_unavailable'
  | 'contract_violation';

export type ResearchRecipeErrorCopyKey =
  | 'workflow.research_idempotency_unavailable'
  | 'workflow.m_303534acbecf';

export class ResearchRecipeControllerError extends Error {
  readonly code: ResearchRecipeErrorCode;

  protected constructor(
    code: ResearchRecipeErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.code = code;
  }
}

export class StaleResearchRecipeScopeError extends ResearchRecipeControllerError {
  constructor() {
    super(
      'stale_scope',
      'The Research recipe page response belongs to a stale status scope.',
    );
    this.name = 'StaleResearchRecipeScopeError';
  }
}

export class ResearchRecipePersistenceError extends ResearchRecipeControllerError {
  constructor(message: string, options?: ErrorOptions) {
    super('persistence_unavailable', message, options);
    this.name = 'ResearchRecipePersistenceError';
  }
}

export class ResearchRecipeContractError extends ResearchRecipeControllerError {
  constructor(message: string) {
    super('contract_violation', message);
    this.name = 'ResearchRecipeContractError';
  }
}

export function researchRecipeErrorCopyKey(
  error: unknown,
): ResearchRecipeErrorCopyKey {
  return error instanceof ResearchRecipeControllerError
    && error.code === 'persistence_unavailable'
    ? 'workflow.research_idempotency_unavailable'
    : 'workflow.m_303534acbecf';
}

export function isResearchRecipeEligible(
  recipe: Pick<ResearchRecipe, 'lifecycleStatus'>,
): boolean {
  return recipe.lifecycleStatus === 'published';
}

export class ResearchRecipeController {
  readonly #api: ResearchRecipeApi;
  readonly #pageSize: number;
  readonly #storageOption: ResearchRecipeSessionStorage | null | undefined;
  readonly #cryptoOption: ResearchRecipeUuidSource | null | undefined;
  readonly #listeners = new Set<Listener>();
  readonly #exactCache = new Map<string, ResearchRecipe>();
  readonly #exactFlights = new Map<string, Promise<ResearchRecipe>>();
  readonly #retireFlights = new Map<string, Promise<ResearchRecipe>>();
  readonly #codeMutationEpoch = new Map<string, number>();

  #snapshot: ResearchRecipeSnapshot;
  #scope = 0;
  #mutationEpoch = 0;
  #consumedCursors = new Set<string | null>();
  #pageFlight: PageFlight | null = null;
  #createFlight: {
    canonicalPayload: string;
    promise: Promise<CreateRecipeAndRunResult>;
  } | null = null;

  constructor(
    api: ResearchRecipeApi,
    options: ResearchRecipeControllerOptions = {},
  ) {
    const pageSize = options.pageSize ?? DEFAULT_PAGE_SIZE;
    if (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > MAX_PAGE_SIZE) {
      throw new ResearchRecipeContractError(
        `Research recipe pageSize must be an integer from 1 to ${MAX_PAGE_SIZE}.`,
      );
    }
    this.#api = api;
    this.#pageSize = pageSize;
    this.#storageOption = Object.hasOwn(options, 'sessionStorage')
      ? options.sessionStorage
      : undefined;
    this.#cryptoOption = Object.hasOwn(options, 'crypto')
      ? options.crypto
      : undefined;
    this.#snapshot = freezeSnapshot({
      status: 'published',
      items: [],
      nextCursor: null,
      loaded: false,
      loading: false,
      error: null,
    });
  }

  getSnapshot = (): ResearchRecipeSnapshot => this.#snapshot;

  subscribe = (listener: Listener): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  setStatus(status: ResearchRecipeLifecycleStatus): void {
    if (status === this.#snapshot.status) return;
    this.#scope += 1;
    this.#consumedCursors = new Set();
    this.#pageFlight = null;
    this.#setSnapshot({
      status,
      items: [],
      nextCursor: null,
      loaded: false,
      loading: false,
      error: null,
    });
  }

  loadInitial(): Promise<ResearchRecipeSnapshot> {
    if (this.#snapshot.loaded) return Promise.resolve(this.#snapshot);
    return this.#loadPage(null, true);
  }

  reload(): Promise<ResearchRecipeSnapshot> {
    this.#scope += 1;
    this.#consumedCursors = new Set();
    this.#pageFlight = null;
    this.#setSnapshot({
      ...this.#snapshot,
      items: [],
      nextCursor: null,
      loaded: false,
      loading: false,
      error: null,
    });
    return this.#loadPage(null, true);
  }

  loadMore(): Promise<ResearchRecipeSnapshot> {
    if (!this.#snapshot.loaded) return this.loadInitial();
    if (this.#snapshot.nextCursor === null) return Promise.resolve(this.#snapshot);
    return this.#loadPage(this.#snapshot.nextCursor, false);
  }

  getCached(code: string): ResearchRecipe | undefined {
    return this.#exactCache.get(validateCode(code));
  }

  hydrateExact(code: string, refresh = false): Promise<ResearchRecipe> {
    const exactCode = validateCode(code);
    if (!refresh) {
      const cached = this.#exactCache.get(exactCode);
      if (cached) return Promise.resolve(cached);
    }
    const existingFlight = this.#exactFlights.get(exactCode);
    if (existingFlight) return existingFlight;

    const mutationEpochAtStart = this.#codeMutationEpoch.get(exactCode) ?? 0;
    const flight = Promise.resolve()
      .then(() => this.#api.getRecipe({code: exactCode}))
      .then((recipe) => {
        assertExactRecipe(exactCode, recipe);
        if ((this.#codeMutationEpoch.get(exactCode) ?? 0) > mutationEpochAtStart) {
          const current = this.#exactCache.get(exactCode);
          if (current) return current;
          throw new ResearchRecipeContractError(
            `Exact Research recipe ${exactCode} became stale during a local mutation.`,
          );
        }
        this.#cacheServerRecipe(recipe, false);
        return recipe;
      })
      .finally(() => {
        if (this.#exactFlights.get(exactCode) === flight) {
          this.#exactFlights.delete(exactCode);
        }
      });
    this.#exactFlights.set(exactCode, flight);
    return flight;
  }

  createRecipeAndRun(payload: JsonObject): Promise<CreateRecipeAndRunResult> {
    const canonicalPayload = canonicalJson(payload);
    const requestPayload = JSON.parse(canonicalPayload) as JsonObject;
    if (this.#createFlight) {
      if (this.#createFlight.canonicalPayload === canonicalPayload) {
        return this.#createFlight.promise;
      }
      return Promise.reject(
        new ResearchRecipeContractError(
          'A different Research recipe creation is already in progress.',
        ),
      );
    }

    const storage = this.#requireStorage();
    const uuidSource = this.#requireUuidSource();
    const operation = this.#getOrCreateCreateRunOperation(
      storage,
      uuidSource,
      canonicalPayload,
    );
    const flight = Promise.resolve()
      .then(() =>
        this.#api.createRecipe({
          payload: requestPayload,
          idempotencyKey: operation.createIdempotencyKey,
        }),
      )
      .then(async (recipe) => {
        validateRecipe(recipe);
        this.#applyLocalMutation(recipe);
        const run = await this.#api.createRun({
          recipeCode: recipe.code,
          idempotencyKey: operation.runIdempotencyKey,
        });
        validateRun(recipe.code, run);
        this.#clearCreateRunOperation(storage, operation);
        return {recipe, run};
      })
      .finally(() => {
        if (this.#createFlight?.promise === flight) this.#createFlight = null;
      });
    this.#createFlight = {canonicalPayload, promise: flight};
    return flight;
  }

  retireRecipe(code: string): Promise<ResearchRecipe> {
    const exactCode = validateCode(code);
    const existingFlight = this.#retireFlights.get(exactCode);
    if (existingFlight) return existingFlight;

    const storage = this.#requireStorage();
    const uuidSource = this.#requireUuidSource();
    const operation = this.#getOrCreateRetireOperation(
      storage,
      uuidSource,
      exactCode,
    );
    const flight = Promise.resolve()
      .then(() =>
        this.#api.retireRecipe({
          code: exactCode,
          idempotencyKey: operation.idempotencyKey,
        }),
      )
      .then((recipe) => {
        assertExactRecipe(exactCode, recipe);
        if (recipe.lifecycleStatus !== 'retired') {
          throw new ResearchRecipeContractError(
            `Research recipe ${exactCode} did not enter the retired lifecycle.`,
          );
        }
        this.#applyLocalMutation(recipe);
        this.#clearRetireOperation(storage, operation);
        return recipe;
      })
      .finally(() => {
        if (this.#retireFlights.get(exactCode) === flight) {
          this.#retireFlights.delete(exactCode);
        }
      });
    this.#retireFlights.set(exactCode, flight);
    return flight;
  }

  #loadPage(
    cursor: string | null,
    replace: boolean,
  ): Promise<ResearchRecipeSnapshot> {
    const scope = this.#scope;
    if (this.#pageFlight?.scope === scope) return this.#pageFlight.promise;

    if (this.#consumedCursors.has(cursor)) {
      return Promise.reject(
        new ResearchRecipeContractError(
          'The same opaque Research recipe cursor cannot be requested twice.',
        ),
      );
    }
    const status = this.#snapshot.status;
    const mutationEpochAtStart = this.#mutationEpoch;

    let flight: Promise<ResearchRecipeSnapshot>;
    flight = Promise.resolve()
      .then(() =>
        this.#api.listRecipes({
          status,
          ...(cursor === null ? {} : {cursor}),
          limit: this.#pageSize,
        }),
      )
      .then((page) => {
        if (scope !== this.#scope || status !== this.#snapshot.status) {
          throw new StaleResearchRecipeScopeError();
        }
        validatePage(page);
        for (const recipe of page.items) {
          if (recipe.lifecycleStatus !== status) {
            throw new ResearchRecipeContractError(
              `Research recipe ${recipe.code} does not belong to the requested ${status} scope.`,
            );
          }
        }
        this.#consumedCursors.add(cursor);
        if (page.nextCursor !== null && this.#consumedCursors.has(page.nextCursor)) {
          throw new ResearchRecipeContractError(
            'The Research recipe API returned a cursor that was already consumed.',
          );
        }
        const existing = replace
          ? this.#snapshot.items.filter(
              (recipe) =>
                (this.#codeMutationEpoch.get(recipe.code) ?? 0) >
                mutationEpochAtStart,
            )
          : this.#snapshot.items;
        const items = mergeRecipes(
          existing,
          page.items,
          (code) =>
            (this.#codeMutationEpoch.get(code) ?? 0) > mutationEpochAtStart,
        );
        if (this.#pageFlight?.promise === flight) this.#pageFlight = null;
        return this.#setSnapshot({
          status,
          items,
          nextCursor: page.nextCursor,
          loaded: true,
          loading: false,
          error: null,
        });
      })
      .catch((error: unknown) => {
        if (this.#pageFlight?.promise === flight) this.#pageFlight = null;
        if (
          !(error instanceof StaleResearchRecipeScopeError) &&
          scope === this.#scope &&
          status === this.#snapshot.status
        ) {
          this.#setSnapshot({...this.#snapshot, loading: false, error});
        }
        throw error;
      })
      .finally(() => {
        if (this.#pageFlight?.promise === flight) this.#pageFlight = null;
      });
    this.#pageFlight = {scope, promise: flight};
    this.#setSnapshot({...this.#snapshot, loading: true, error: null});
    return flight;
  }

  #cacheServerRecipe(recipe: ResearchRecipe, appendIfMissing: boolean): void {
    validateRecipe(recipe);
    this.#exactCache.set(recipe.code, recipe);
    const existingIndex = this.#snapshot.items.findIndex(
      (item) => item.code === recipe.code,
    );
    if (recipe.lifecycleStatus !== this.#snapshot.status) {
      if (existingIndex >= 0) {
        this.#setSnapshot({
          ...this.#snapshot,
          items: this.#snapshot.items.filter((item) => item.code !== recipe.code),
        });
      }
      return;
    }
    if (existingIndex < 0 && !appendIfMissing) return;
    const items = [...this.#snapshot.items];
    if (existingIndex >= 0) items[existingIndex] = recipe;
    else items.push(recipe);
    this.#setSnapshot({...this.#snapshot, items});
  }

  #markMutationStarted(code: string): void {
    this.#mutationEpoch += 1;
    this.#codeMutationEpoch.set(code, this.#mutationEpoch);
  }

  #applyLocalMutation(recipe: ResearchRecipe): void {
    validateRecipe(recipe);
    this.#markMutationStarted(recipe.code);
    this.#exactCache.set(recipe.code, recipe);
    const items = [...this.#snapshot.items];
    const existingIndex = items.findIndex((item) => item.code === recipe.code);
    if (recipe.lifecycleStatus === this.#snapshot.status) {
      if (existingIndex >= 0) items[existingIndex] = recipe;
      else items.unshift(recipe);
    } else if (existingIndex >= 0) {
      items.splice(existingIndex, 1);
    }
    this.#setSnapshot({...this.#snapshot, items});
  }

  #getOrCreateCreateRunOperation(
    storage: ResearchRecipeSessionStorage,
    uuidSource: ResearchRecipeUuidSource,
    canonicalPayload: string,
  ): StoredCreateRunOperation {
    const stored = readStoredJson(storage, CREATE_RUN_STORAGE_KEY);
    if (stored !== null) {
      const operation = parseCreateRunOperation(stored);
      if (operation.canonicalPayload === canonicalPayload) return operation;
    }
    const operation: StoredCreateRunOperation = {
      schemaVersion: 1,
      canonicalPayload,
      createIdempotencyKey: newIdempotencyKey(uuidSource, 'recipe'),
      runIdempotencyKey: newIdempotencyKey(uuidSource, 'run'),
    };
    writeStoredJson(storage, CREATE_RUN_STORAGE_KEY, operation);
    return operation;
  }

  #clearCreateRunOperation(
    storage: ResearchRecipeSessionStorage,
    completed: StoredCreateRunOperation,
  ): void {
    const stored = readStoredJson(storage, CREATE_RUN_STORAGE_KEY);
    if (stored === null) return;
    const current = parseCreateRunOperation(stored);
    if (
      current.canonicalPayload === completed.canonicalPayload &&
      current.createIdempotencyKey === completed.createIdempotencyKey &&
      current.runIdempotencyKey === completed.runIdempotencyKey
    ) {
      removeStored(storage, CREATE_RUN_STORAGE_KEY);
    }
  }

  #getOrCreateRetireOperation(
    storage: ResearchRecipeSessionStorage,
    uuidSource: ResearchRecipeUuidSource,
    code: string,
  ): StoredRetireOperation {
    const storageKey = retireStorageKey(code);
    const stored = readStoredJson(storage, storageKey);
    if (stored !== null) return parseRetireOperation(stored, code);
    const operation: StoredRetireOperation = {
      schemaVersion: 1,
      code,
      idempotencyKey: newIdempotencyKey(uuidSource, 'retire'),
    };
    writeStoredJson(storage, storageKey, operation);
    return operation;
  }

  #clearRetireOperation(
    storage: ResearchRecipeSessionStorage,
    completed: StoredRetireOperation,
  ): void {
    const storageKey = retireStorageKey(completed.code);
    const stored = readStoredJson(storage, storageKey);
    if (stored === null) return;
    const current = parseRetireOperation(stored, completed.code);
    if (current.idempotencyKey === completed.idempotencyKey) {
      removeStored(storage, storageKey);
    }
  }

  #requireStorage(): ResearchRecipeSessionStorage {
    if (this.#storageOption === null) {
      throw new ResearchRecipePersistenceError(
        'sessionStorage is required for idempotent Research recipe mutations.',
      );
    }
    if (this.#storageOption !== undefined) return this.#storageOption;
    try {
      if (typeof globalThis.sessionStorage === 'undefined') throw new Error();
      return globalThis.sessionStorage;
    } catch (error) {
      throw new ResearchRecipePersistenceError(
        'sessionStorage is unavailable; the Research mutation was not sent.',
        {cause: error},
      );
    }
  }

  #requireUuidSource(): ResearchRecipeUuidSource {
    if (this.#cryptoOption === null) {
      throw new ResearchRecipePersistenceError(
        'crypto.randomUUID is required for idempotent Research recipe mutations.',
      );
    }
    const source =
      this.#cryptoOption === undefined
        ? (globalThis.crypto as ResearchRecipeUuidSource | undefined)
        : this.#cryptoOption;
    if (!source || typeof source.randomUUID !== 'function') {
      throw new ResearchRecipePersistenceError(
        'crypto.randomUUID is unavailable; the Research mutation was not sent.',
      );
    }
    return source;
  }

  #setSnapshot(next: ResearchRecipeSnapshot): ResearchRecipeSnapshot {
    const published = freezeSnapshot(next);
    this.#snapshot = published;
    for (const listener of this.#listeners) listener();
    return published;
  }
}

function mergeRecipes(
  existing: readonly ResearchRecipe[],
  incoming: readonly ResearchRecipe[],
  preserveLocalMutation: (code: string) => boolean,
): ResearchRecipe[] {
  const merged = [...existing];
  const indexByCode = new Map<string, number>();
  merged.forEach((recipe, index) => indexByCode.set(recipe.code, index));
  for (const recipe of incoming) {
    validateRecipe(recipe);
    if (preserveLocalMutation(recipe.code)) continue;
    const existingIndex = indexByCode.get(recipe.code);
    if (existingIndex !== undefined) {
      merged[existingIndex] = recipe;
    } else {
      indexByCode.set(recipe.code, merged.length);
      merged.push(recipe);
    }
  }
  return merged;
}

function validatePage(page: ResearchRecipePage): void {
  if (!page || !Array.isArray(page.items)) {
    throw new ResearchRecipeContractError(
      'The Research recipe API returned an invalid page.',
    );
  }
  if (
    page.nextCursor !== null &&
    (typeof page.nextCursor !== 'string' || page.nextCursor.length === 0)
  ) {
    throw new ResearchRecipeContractError(
      'The Research recipe API returned an invalid opaque cursor.',
    );
  }
}

function validateRecipe(recipe: ResearchRecipe): void {
  validateCode(recipe?.code);
  if (!['draft', 'published', 'retired'].includes(recipe.lifecycleStatus)) {
    throw new ResearchRecipeContractError(
      `Research recipe ${recipe.code} has an invalid lifecycleStatus.`,
    );
  }
}

function assertExactRecipe(code: string, recipe: ResearchRecipe): void {
  validateRecipe(recipe);
  if (recipe.code !== code) {
    throw new ResearchRecipeContractError(
      `Exact Research recipe ${code} returned mismatched code ${recipe.code}.`,
    );
  }
}

function validateRun(recipeCode: string, run: ResearchRun): void {
  if (!run || !isNonEmptyString(run.runCode) || run.recipeCode !== recipeCode) {
    throw new ResearchRecipeContractError(
      `Research recipe ${recipeCode} returned an invalid run.`,
    );
  }
}

function validateCode(code: string): string {
  if (
    typeof code !== 'string' ||
    code.trim() === '' ||
    code.trim() !== code
  ) {
    throw new ResearchRecipeContractError(
      'A non-empty Research recipe code is required.',
    );
  }
  return code;
}

function freezeSnapshot(snapshot: ResearchRecipeSnapshot): ResearchRecipeSnapshot {
  const items = Object.freeze([...snapshot.items]);
  return Object.freeze({...snapshot, items});
}

function canonicalJson(value: JsonValue): string {
  return canonicalJsonValue(value, new Set<object>());
}

function canonicalJsonValue(value: JsonValue, stack: Set<object>): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new ResearchRecipeContractError(
        'Research recipe payload numbers must be finite.',
      );
    }
    return JSON.stringify(value);
  }
  if (typeof value !== 'object') {
    throw new ResearchRecipeContractError(
      'Research recipe payloads must contain JSON values only.',
    );
  }
  if (stack.has(value)) {
    throw new ResearchRecipeContractError(
      'Research recipe payloads cannot contain circular references.',
    );
  }
  stack.add(value);
  try {
    if (Array.isArray(value)) {
      return `[${value.map((item) => canonicalJsonValue(item, stack)).join(',')}]`;
    }
    return `{${Object.keys(value)
      .sort()
      .map(
        (key) =>
          `${JSON.stringify(key)}:${canonicalJsonValue(value[key], stack)}`,
      )
      .join(',')}}`;
  } finally {
    stack.delete(value);
  }
}

function readStoredJson(
  storage: ResearchRecipeSessionStorage,
  key: string,
): unknown | null {
  let raw: string | null;
  try {
    raw = storage.getItem(key);
  } catch (error) {
    throw new ResearchRecipePersistenceError(
      'Unable to read the retained Research mutation.',
      {cause: error},
    );
  }
  if (raw === null) return null;
  try {
    return JSON.parse(raw) as unknown;
  } catch (error) {
    throw new ResearchRecipePersistenceError(
      'The retained Research mutation is corrupt and cannot be retried safely.',
      {cause: error},
    );
  }
}

function writeStoredJson(
  storage: ResearchRecipeSessionStorage,
  key: string,
  value: StoredCreateRunOperation | StoredRetireOperation,
): void {
  try {
    storage.setItem(key, JSON.stringify(value));
  } catch (error) {
    throw new ResearchRecipePersistenceError(
      'Unable to retain the Research mutation; no request was sent.',
      {cause: error},
    );
  }
}

function removeStored(storage: ResearchRecipeSessionStorage, key: string): void {
  try {
    storage.removeItem(key);
  } catch (error) {
    throw new ResearchRecipePersistenceError(
      'The Research mutation completed but its retained retry key could not be cleared.',
      {cause: error},
    );
  }
}

function parseCreateRunOperation(value: unknown): StoredCreateRunOperation {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    typeof value.canonicalPayload !== 'string' ||
    !isNonEmptyString(value.createIdempotencyKey) ||
    !isNonEmptyString(value.runIdempotencyKey)
  ) {
    throw new ResearchRecipePersistenceError(
      'The retained Research create-and-run operation is invalid.',
    );
  }
  return value as StoredCreateRunOperation;
}

function parseRetireOperation(
  value: unknown,
  expectedCode: string,
): StoredRetireOperation {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.code !== expectedCode ||
    !isNonEmptyString(value.idempotencyKey)
  ) {
    throw new ResearchRecipePersistenceError(
      `The retained retirement for Research recipe ${expectedCode} is invalid.`,
    );
  }
  return value as StoredRetireOperation;
}

function newIdempotencyKey(
  source: ResearchRecipeUuidSource,
  operation: string,
): string {
  const value = source.randomUUID();
  if (!isNonEmptyString(value)) {
    throw new ResearchRecipePersistenceError(
      'crypto.randomUUID returned an invalid idempotency value.',
    );
  }
  return `research-${operation}:${value}`;
}

function retireStorageKey(code: string): string {
  return `${RETIRE_STORAGE_PREFIX}${encodeURIComponent(code)}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}
