import {readFile} from 'node:fs/promises';

import type {JsonObject} from '../../src/baloot_platform/service/foundry_dashboard/src/shared/json.js';
import {test} from 'vitest';
import {researchRunDisplayStatus} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model.js';
import type {
  ResearchRecipe,
  ResearchRun,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model.js';
import {
  ResearchRecipeController,
  ResearchRecipeContractError,
  ResearchRecipePersistenceError,
  StaleResearchRecipeScopeError,
  isResearchRecipeEligible,
  researchRecipeErrorCopyKey,
  type ResearchRecipeApi,
  type ResearchRecipePage,
  type ResearchRecipeSessionStorage,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/recipes.js';

test('terminal Research labels distinguish outcomes from worker completion', () => {
  const completed = {...run('status-fixture'), status: 'completed' as const};
  const cases = {
    proof_incomplete: 'incomplete', resource_incomplete: 'incomplete',
    global_campaign_timeout: 'incomplete', exact_structural_infeasibility: 'structurally_infeasible',
    candidate_range_exhausted: 'exhausted', qualified_target_satisfied: 'qualified',
    cancelled: 'cancelled', unknown_future_reason: 'completed',
  };
  for (const [stopReason, expected] of Object.entries(cases)) {
    const campaignAccounting = {schemaVersion: 1 as const, state: 'completed' as const, stopReason};
    assert(researchRunDisplayStatus({...completed, campaignAccounting}) === expected, stopReason);
    assert(researchRunDisplayStatus({...completed, campaignProgress: campaignAccounting}) === expected, `progress:${stopReason}`);
    for (const status of ['queued', 'running', 'failed', 'cancelled'] as const) {
      assert(researchRunDisplayStatus({...completed, status, campaignAccounting}) === status, `lifecycle:${status}`);
    }
  }
  assert(researchRunDisplayStatus(completed) === 'completed', 'legacy runs retain their status');
  assert(researchRunDisplayStatus({...completed,
    campaignProgress: {schemaVersion: 1, state: 'running', stopReason: 'global_campaign_timeout'},
    campaignAccounting: {schemaVersion: 1, state: 'completed', stopReason: 'qualified_target_satisfied'},
  }) === 'qualified', 'final accounting supersedes stale live progress');
});

type Deferred<T> = {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
};

class MemoryStorage implements ResearchRecipeSessionStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return {promise, resolve, reject};
}

function recipe(
  code: string,
  lifecycleStatus: ResearchRecipe['lifecycleStatus'] = 'published',
  sha256 = `sha-${code}`,
): ResearchRecipe {
  return {
    code,
    titleAr: code,
    sha256,
    status: 'published',
    lifecycleStatus,
  };
}

function run(recipeCode: string): ResearchRun {
  return {
    runCode: `run-${recipeCode}`,
    recipeCode,
    status: 'queued',
    progress: 0,
    resultCount: 0,
    qualifiedCount: 0,
    quarantineCount: 0,
  };
}

function fakeApi(overrides: Partial<ResearchRecipeApi>): ResearchRecipeApi {
  const unexpected = (name: string): never => {
    throw new Error(`Unexpected API call: ${name}`);
  };
  return {
    listRecipes: () => unexpected('listRecipes'),
    getRecipe: () => unexpected('getRecipe'),
    createRecipe: () => unexpected('createRecipe'),
    createRun: () => unexpected('createRun'),
    retireRecipe: () => unexpected('retireRecipe'),
    ...overrides,
  };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function expectReject<T>(
  promise: Promise<T>,
  predicate: (error: unknown) => boolean,
  message: string,
): Promise<void> {
  try {
    await promise;
  } catch (error) {
    assert(predicate(error), message);
    return;
  }
  throw new Error(message);
}

async function testCursorPagingAndScopeIsolation(): Promise<void> {
  const secondPage = deferred<ResearchRecipePage>();
  const stalePage = deferred<ResearchRecipePage>();
  let listCalls = 0;
  const controller = new ResearchRecipeController(
    fakeApi({
      listRecipes: ({status, cursor, limit}) => {
        listCalls += 1;
        assert(limit === 25, 'the configured page size must reach the API');
        if (status === 'retired') return stalePage.promise;
        if (cursor === undefined) {
          return Promise.resolve({
            items: [
              {...recipe('Z'), titleAr: 'shared'},
              {...recipe('A'), titleAr: 'shared'},
              recipe('Y'),
            ],
            nextCursor: '<first-page>',
          });
        }
        assert(
          cursor === '<first-page>',
          'every opaque cursor value must pass through unchanged',
        );
        return secondPage.promise;
      },
    }),
    {pageSize: 25},
  );

  await controller.loadInitial();
  const firstMore = controller.loadMore();
  const duplicateMore = controller.loadMore();
  assert(firstMore === duplicateMore, 'concurrent loadMore calls must coalesce');
  await Promise.resolve();
  assert(listCalls === 2, 'coalesced loadMore must issue one request');
  secondPage.resolve({
    items: [recipe('A', 'published', 'sha-A-new'), recipe('M')],
    nextCursor: null,
  });
  await firstMore;
  assert(
    controller.getSnapshot().items.map((item) => item.code).join(',') ===
      'Z,A,Y,M',
    'code dedupe must preserve unsorted server order and repeated titles',
  );
  assert(
    controller.getSnapshot().items[1]?.sha256 === 'sha-A-new',
    'later page evidence must refresh a duplicate in place',
  );
  await controller.loadMore();
  assert(listCalls === 2, 'a terminal null cursor must not issue another request');

  controller.setStatus('retired');
  const retiredLoad = controller.loadInitial();
  controller.setStatus('draft');
  stalePage.resolve({items: [recipe('R', 'retired')], nextCursor: null});
  await expectReject(
    retiredLoad,
    (error) => error instanceof StaleResearchRecipeScopeError,
    'a stale status response must be rejected',
  );
  assert(
    controller.getSnapshot().status === 'draft' &&
      controller.getSnapshot().items.length === 0,
    'a stale response must not overwrite the active scope',
  );
}

async function testReentrantSubscribersCannotDuplicateOrLosePages(): Promise<void> {
  const firstPage = deferred<ResearchRecipePage>();
  const secondPage = deferred<ResearchRecipePage>();
  let listCalls = 0;
  let loadingReentry: Promise<unknown> | undefined;
  let nextPageReentry: Promise<unknown> | undefined;
  const controller = new ResearchRecipeController(
    fakeApi({
      listRecipes: ({cursor}) => {
        listCalls += 1;
        return cursor === undefined ? firstPage.promise : secondPage.promise;
      },
    }),
  );
  controller.subscribe(() => {
    const snapshot = controller.getSnapshot();
    if (snapshot.loading && !snapshot.loaded && !loadingReentry) {
      loadingReentry = controller.loadInitial();
    }
    if (
      snapshot.loaded &&
      !snapshot.loading &&
      snapshot.nextCursor !== null &&
      !nextPageReentry
    ) {
      nextPageReentry = controller.loadMore();
    }
  });

  const initial = controller.loadInitial();
  await Promise.resolve();
  assert(listCalls === 1, 'loading notification reentry must coalesce');
  firstPage.resolve({items: [recipe('A')], nextCursor: 'next'});
  await initial;
  await Promise.resolve();
  assert(
    Number(listCalls) === 2 && nextPageReentry !== undefined,
    'completion notification reentry must start the next opaque page',
  );
  assert(
    loadingReentry === initial,
    'loading notification must observe the installed page flight',
  );
  secondPage.resolve({items: [recipe('B')], nextCursor: null});
  await nextPageReentry;
  assert(
    controller.getSnapshot().items.map((item) => item.code).join(',') === 'A,B',
    'completion notification must not lose the next page',
  );
}

async function testExactHydrationCannotOverwriteMutation(): Promise<void> {
  const exact = deferred<ResearchRecipe>();
  let exactCalls = 0;
  const controller = new ResearchRecipeController(
    fakeApi({
      listRecipes: () =>
        Promise.resolve({items: [recipe('A')], nextCursor: null}),
      getRecipe: ({code}) => {
        assert(code === 'A', 'exact hydration must request the exact code');
        exactCalls += 1;
        return exact.promise;
      },
      retireRecipe: () => Promise.resolve(recipe('A', 'retired')),
    }),
    {
      sessionStorage: new MemoryStorage(),
      crypto: {randomUUID: () => 'retire-uuid'},
    },
  );
  await controller.loadInitial();
  const hydration = controller.hydrateExact('A', true);
  const coalesced = controller.hydrateExact('A', true);
  assert(hydration === coalesced, 'exact hydration must coalesce by code');
  await controller.retireRecipe('A');
  exact.resolve(recipe('A'));
  const hydrated = await hydration;
  assert(exactCalls === 1, 'coalesced hydration must issue one exact request');
  assert(
    hydrated.lifecycleStatus === 'retired' &&
      controller.getCached('A')?.lifecycleStatus === 'retired',
    'stale hydration must return and preserve the newer local mutation',
  );
  assert(
    controller.getSnapshot().items.length === 0,
    'retirement must remove the row from the published scope',
  );
  await controller.hydrateExact('A');
  assert(exactCalls === 1, 'non-forced exact hydration must reuse exact cache');
}

async function testExactFailureHasNoPageFallbackAndCanRetry(): Promise<void> {
  let exactCalls = 0;
  const controller = new ResearchRecipeController(
    fakeApi({
      listRecipes: () =>
        Promise.resolve({items: [recipe('A')], nextCursor: null}),
      getRecipe: ({code}) => {
        exactCalls += 1;
        return exactCalls === 1
          ? Promise.reject(new Error('exact unavailable'))
          : Promise.resolve(recipe(code));
      },
    }),
  );
  await controller.loadInitial();
  await expectReject(
    controller.hydrateExact('A'),
    (error) => error instanceof Error && error.message === 'exact unavailable',
    'an exact failure must not fall back to the page row',
  );
  const hydrated = await controller.hydrateExact('A');
  assert(
    hydrated.code === 'A' && exactCalls === 2,
    'a rejected exact request must clear its coalescing entry for retry',
  );
}

async function testCreateAndRunRetainsKeysAcrossFailureAndReload(): Promise<void> {
  const storage = new MemoryStorage();
  const payloadOne: JsonObject = {
    beta: 2,
    alpha: {zulu: true, alpha: false},
    sequence: [2, 1],
  };
  const payloadSame: JsonObject = {
    sequence: [2, 1],
    alpha: {alpha: false, zulu: true},
    beta: 2,
  };
  const firstKeys: string[] = [];
  let uuidIndex = 0;
  const uuid = {randomUUID: () => `uuid-${++uuidIndex}`};
  const first = new ResearchRecipeController(
    fakeApi({
      createRecipe: ({idempotencyKey}) => {
        firstKeys.push(idempotencyKey);
        return Promise.resolve(recipe('NEW'));
      },
      createRun: ({idempotencyKey}) => {
        firstKeys.push(idempotencyKey);
        return Promise.reject(new Error('network failure'));
      },
    }),
    {sessionStorage: storage, crypto: uuid},
  );
  await expectReject(
    first.createRecipeAndRun(payloadOne),
    (error) => error instanceof Error && error.message === 'network failure',
    'run failure must propagate',
  );
  assert(storage.values.size === 1, 'failed operations must retain both keys');
  assert(
    first.getCached('NEW')?.code === 'NEW',
    'a created recipe must remain available when run creation fails',
  );

  const retryKeys: string[] = [];
  const reloaded = new ResearchRecipeController(
    fakeApi({
      createRecipe: ({idempotencyKey}) => {
        retryKeys.push(idempotencyKey);
        return Promise.resolve(recipe('NEW'));
      },
      createRun: ({recipeCode, idempotencyKey}) => {
        retryKeys.push(idempotencyKey);
        return Promise.resolve(run(recipeCode));
      },
    }),
    {sessionStorage: storage, crypto: uuid},
  );
  const completed = await reloaded.createRecipeAndRun(payloadSame);
  assert(
    retryKeys.join(',') === firstKeys.join(','),
    'canonical-equivalent payloads must reuse both retained keys after reload',
  );
  assert(
    [...storage.values].length === 0,
    'keys must clear only after full success',
  );
  assert(uuidIndex === 2, 'retrying the same operation must not create new keys');
  assert(
    completed.run.recipeCode === 'NEW',
    'the run must be created for the exact created recipe',
  );
}

async function testChangedPayloadRotatesCreateAndRunKeys(): Promise<void> {
  const storage = new MemoryStorage();
  const keys: string[] = [];
  let uuidIndex = 0;
  const failCreate = () =>
    new ResearchRecipeController(
      fakeApi({
        createRecipe: ({idempotencyKey}) => {
          keys.push(idempotencyKey);
          return Promise.reject(new Error('keep operation pending'));
        },
      }),
      {
        sessionStorage: storage,
        crypto: {randomUUID: () => `changed-${++uuidIndex}`},
      },
    );
  await expectReject(
    failCreate().createRecipeAndRun({target: 'first'}),
    (error) => error instanceof Error,
    'the first operation must remain retryable',
  );
  const firstOperation = JSON.parse([...storage.values.values()][0] ?? '{}') as {
    createIdempotencyKey?: string;
    runIdempotencyKey?: string;
  };
  await expectReject(
    failCreate().createRecipeAndRun({target: 'second'}),
    (error) => error instanceof Error,
    'the changed operation must remain retryable',
  );
  const secondOperation = JSON.parse([...storage.values.values()][0] ?? '{}') as {
    createIdempotencyKey?: string;
    runIdempotencyKey?: string;
  };
  assert(keys[0] !== keys[1], 'a changed canonical payload must rotate its keys');
  assert(
    firstOperation.runIdempotencyKey !== secondOperation.runIdempotencyKey &&
      firstOperation.createIdempotencyKey !==
        secondOperation.createIdempotencyKey,
    'both create and run keys must rotate for a changed canonical payload',
  );
  assert(uuidIndex === 4, 'each changed operation must own create and run keys');
}

async function testRetireCoalescesAndPreservesFailureState(): Promise<void> {
  const storage = new MemoryStorage();
  const pending = deferred<ResearchRecipe>();
  const retireKeys: string[] = [];
  const first = new ResearchRecipeController(
    fakeApi({
      listRecipes: () =>
        Promise.resolve({items: [recipe('A')], nextCursor: null}),
      retireRecipe: ({idempotencyKey}) => {
        retireKeys.push(idempotencyKey);
        return pending.promise;
      },
    }),
    {
      sessionStorage: storage,
      crypto: {randomUUID: () => 'same-retire-uuid'},
    },
  );
  await first.loadInitial();
  const retirement = first.retireRecipe('A');
  assert(
    retirement === first.retireRecipe('A'),
    'retirement must have one in-flight request per code',
  );
  await Promise.resolve();
  pending.reject(new Error('retire failed'));
  await expectReject(
    retirement,
    (error) => error instanceof Error && error.message === 'retire failed',
    'retire failure must propagate',
  );
  assert(retireKeys.length === 1, 'coalesced retirement must call the API once');
  assert(storage.values.size === 1, 'failed retirement must retain its key');
  assert(
    first.getSnapshot().items[0]?.lifecycleStatus === 'published',
    'failed retirement must preserve the visible row',
  );
  await expectReject(
    first.retireRecipe('A'),
    (error) => error instanceof Error && error.message === 'retire failed',
    'a failed in-flight retirement must clear for same-controller retry',
  );
  assert(
    retireKeys[1] !== undefined && retireKeys[0] === retireKeys[1],
    'same-controller retirement retry must reuse the retained key',
  );

  const retry = new ResearchRecipeController(
    fakeApi({
      retireRecipe: ({idempotencyKey}) => {
        retireKeys.push(idempotencyKey);
        return Promise.resolve(recipe('A', 'retired'));
      },
    }),
    {
      sessionStorage: storage,
      crypto: {randomUUID: () => 'must-not-be-used'},
    },
  );
  await retry.retireRecipe('A');
  assert(retireKeys[0] === retireKeys[2], 'reload retirement must reuse its key');
  assert(
    [...storage.values].length === 0,
    'successful retirement must clear its key',
  );
  assert(
    retry.getCached('A')?.lifecycleStatus === 'retired',
    'successful retirement must upsert exact evidence',
  );
}

async function testRetireRequiresRetiredLifecycle(): Promise<void> {
  const storage = new MemoryStorage();
  const controller = new ResearchRecipeController(
    fakeApi({
      listRecipes: () =>
        Promise.resolve({items: [recipe('A')], nextCursor: null}),
      retireRecipe: () => Promise.resolve(recipe('A', 'published')),
    }),
    {
      sessionStorage: storage,
      crypto: {randomUUID: () => 'wrong-lifecycle'},
    },
  );
  await controller.loadInitial();
  await expectReject(
    controller.retireRecipe('A'),
    (error) => error instanceof Error,
    'retirement must fail when the server does not return retired lifecycle',
  );
  assert(storage.values.size === 1, 'invalid retirement must retain its key');
  assert(
    controller.getSnapshot().items[0]?.lifecycleStatus === 'published',
    'invalid retirement must not change the row',
  );
}

async function testFailedRetireDoesNotSuppressPendingPage(): Promise<void> {
  const page = deferred<ResearchRecipePage>();
  const controller = new ResearchRecipeController(
    fakeApi({
      listRecipes: () => page.promise,
      retireRecipe: () => Promise.reject(new Error('retire failed')),
    }),
    {
      sessionStorage: new MemoryStorage(),
      crypto: {randomUUID: () => 'failed-retire-race'},
    },
  );
  const pendingPage = controller.loadInitial();
  await expectReject(
    controller.retireRecipe('A'),
    (error) => error instanceof Error && error.message === 'retire failed',
    'failed retirement must propagate during an older page request',
  );
  page.resolve({items: [recipe('A')], nextCursor: null});
  await pendingPage;
  assert(
    controller.getSnapshot().items[0]?.code === 'A',
    'failed retirement must not suppress a valid older page response',
  );
}

async function testMutationSafetyAndEligibility(): Promise<void> {
  let mutations = 0;
  const controller = new ResearchRecipeController(
    fakeApi({
      createRecipe: () => {
        mutations += 1;
        return Promise.resolve(recipe('unsafe'));
      },
    }),
    {sessionStorage: new MemoryStorage(), crypto: null},
  );
  let error: unknown;
  try {
    controller.createRecipeAndRun({value: true});
  } catch (caught) {
    error = caught;
  }
  assert(
    error instanceof ResearchRecipePersistenceError && mutations === 0,
    'missing crypto.randomUUID must fail before the API is called',
  );
  assert(
    error.code === 'persistence_unavailable'
      && researchRecipeErrorCopyKey(error)
        === 'workflow.research_idempotency_unavailable',
    'persistence failures must expose only the governed idempotency copy key',
  );
  const contractError = new ResearchRecipeContractError('internal diagnostic');
  assert(
    contractError.code === 'contract_violation'
      && researchRecipeErrorCopyKey(contractError) === 'workflow.m_303534acbecf'
      && researchRecipeErrorCopyKey(new Error('network diagnostic'))
        === 'workflow.m_303534acbecf',
    'contract and API failures must map to governed generic request copy',
  );
  assert(isResearchRecipeEligible(recipe('P')), 'published recipes are eligible');
  assert(
    !isResearchRecipeEligible(recipe('D', 'draft')),
    'draft recipes must fail eligibility closed',
  );
  assert(
    !isResearchRecipeEligible(recipe('R', 'retired')),
    'retired recipes must fail eligibility closed',
  );
}

async function runResearchRecipeControllerTests(): Promise<void> {
  await testCursorPagingAndScopeIsolation();
  await testReentrantSubscribersCannotDuplicateOrLosePages();
  await testExactHydrationCannotOverwriteMutation();
  await testExactFailureHasNoPageFallbackAndCanRetry();
  await testCreateAndRunRetainsKeysAcrossFailureAndReload();
  await testChangedPayloadRotatesCreateAndRunKeys();
  await testRetireCoalescesAndPreservesFailureState();
  await testRetireRequiresRetiredLifecycle();
  await testFailedRetireDoesNotSuppressPendingPage();
  await testMutationSafetyAndEligibility();
}

test('Research recipe controller keeps paging and mutations deterministic', async () => {
  await runResearchRecipeControllerTests();
});

test('Research run receipts use admitted API summaries without client re-derivation', async () => {
  const sourceRoot = new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/',
    import.meta.url,
  );
  const [auditSource, mainSource, modelSource, styles] = await Promise.all([
    readFile(new URL('features/research/audit.tsx', sourceRoot), 'utf8'),
    readFile(new URL('main.tsx', sourceRoot), 'utf8'),
    readFile(new URL('features/research/model.ts', sourceRoot), 'utf8'),
    readFile(new URL('styles.css', sourceRoot), 'utf8'),
  ]);
  const summaryFields = [
    'admittedSeedCount',
    'exactRejectionCount',
    'noGoodAppliedCount',
    'noGoodUnappliedCount',
    'continuationSolveCount',
    'continuationStatusDistribution',
    'stopReasonDistribution',
  ];
  for (const field of summaryFields) {
    assert(modelSource.includes(field), `the receipt model must type ${field}`);
    assert(auditSource.includes(field), `the run drawer must render ${field}`);
  }
  assert(
    modelSource.includes('ResearchSequentialSearchSummary = Readonly<{')
      && modelSource.includes('admitted: true;')
      && modelSource.includes('receiptBundle: Readonly<JsonObject>;'),
    'the admitted receipt contract must remain immutable and literal-true',
  );
  assert(
    mainSource.includes(
      '/sequential-search-receipts?offset=0&limit=25',
    ),
    'the run drawer must request the bounded admitted receipt page',
  );
  assert(
    auditSource.includes('receiptPage.items.map')
      && !auditSource.includes('receiptPage.items.sort'),
    'receipt rows must retain deterministic API order',
  );
  assert(
    auditSource.includes("if (receiptState !== 'idle') return;")
      && auditSource.includes('if (event.currentTarget.open) loadReceipts();'),
    'the admitted receipt page must load once and only after local disclosure',
  );
  assert(
    auditSource.includes('JSON.stringify(item.receiptBundle, null, 2)')
      && !auditSource.includes('rejectionFeedbackRows'),
    'the drawer must show exact bundles without deriving receipt counters',
  );
  assert(
    styles.includes('.research-sequential-summary')
      && styles.includes('.research-sequential-receipt-list')
      && styles.includes('.research-sequential-identities'),
    'the admitted receipt summary and local drilldown must retain scoped styling',
  );
});

test('Research campaign progress separates aggregate counts from bounded candidate receipts', async () => {
  const sourceRoot = new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/features/research/',
    import.meta.url,
  );
  const [modelSource, auditSource, studioSource] = await Promise.all([
    readFile(new URL('model.ts', sourceRoot), 'utf8'),
    readFile(new URL('audit.tsx', sourceRoot), 'utf8'),
    readFile(new URL('scenarioStudio.tsx', sourceRoot), 'utf8'),
  ]);
  assert(
    modelSource.includes('ResearchCandidateProgressReceipt')
      && modelSource.includes('candidateReceiptCounts?: Record<ResearchCandidateProgressClassification, number>;')
      && modelSource.includes('candidateReceiptHistoryTruncated?: boolean;')
      && modelSource.includes('candidateReceipts?: ResearchCandidateProgressReceipt[];')
      && modelSource.includes('candidateReceipt?: ResearchCandidateProgressReceipt;'),
    'the run model must type aggregate counts and the bounded detailed history separately',
  );
  assert(
    auditSource.includes('value.candidateReceiptCounts')
      && auditSource.includes('value.candidateReceipts')
      && auditSource.includes('candidateReceipts.map')
      && auditSource.includes('role="log"')
      && auditSource.includes('aria-relevant="additions text"')
      && !auditSource.includes('candidateReceipts.sort'),
    'the run drawer must render every server-ordered candidate receipt as a live log',
  );
  for (const status of ['qualified', 'rejected', 'duplicate', 'incomplete']) {
    assert(
      auditSource.includes(`candidateReceiptCounts.${status}`),
      `the run drawer must expose the ${status} receipt count`,
    );
  }
  assert(
    studioSource.includes('candidateReceiptCounts.incomplete??activeRunProgress.proofIncomplete??0'),
    'live incomplete count must prefer candidate receipts even when proof-incomplete is zero',
  );
});

test('Scenario results refresh once when the selected run becomes terminal', async () => {
  const source = await readFile(new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/features/research/scenarioStudio.tsx',
    import.meta.url,
  ), 'utf8');
  assert(
    source.includes("const activeRun = runs.find((run)=>run.runCode===activeRunCode);")
      && source.includes("const activeRunStatus = activeRun?.status ?? '';")
      && source.includes("['completed','failed','cancelled'].includes(activeRunStatus)")
      && /\},\s*\[[^\]]*\bterminalRunRefresh\b[^\]]*\]\);/.test(source),
    'the existing run-status refresh must invalidate the selected result query at terminal state',
  );
  assert(
    !source.includes('setInterval(() => request<ResearchResultPage>'),
    'result refresh must reuse the governed run poller instead of adding a second poller',
  );
});

test('Scenario result predicate count uses the actor-public projection', async () => {
  const source = await readFile(new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx',
    import.meta.url,
  ), 'utf8');
  assert(
    source.includes('scenarioProjection.predicateEvidenceCount')
      && !source.includes('scenarioSatisfaction.predicates'),
    'the drawer must use the safe public count without reading hidden predicate observations',
  );
});

test('Scenario proof counts distinguish exact totals from incomplete safe bounds', async () => {
  const source = await readFile(new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx',
    import.meta.url,
  ), 'utf8');
  assert(
    source.includes('function ResearchContinuationLineBoundary')
      && source.includes("const complete = lines.complete === true;")
      && source.includes("lines.exactCount ?? '—'")
      && source.includes("lines.lowerBound ?? '0'")
      && source.includes('receiptValue(lines.upperBound)')
      && source.includes('continuationLineBoundaryStatus(value, lines)')
      && source.includes('if (status === null) return null;')
      && source.includes('<code dir="ltr">{status}</code>'),
    'the operator must see exact counts or explicit incomplete lower/upper bounds',
  );
  assert(
    source.includes('function ResearchWinningRootActionBoundary')
      && source.includes('value.winningRootActionCount')
      && source.includes('value.winningRootActionLowerBound ?? 0')
      && source.includes('receiptValue(value.winningRootActionUpperBound)')
      && source.includes('<ResearchWinningRootActionBoundary value={scenarioProof}/>'),
    'incomplete root-action proofs must show certified bounds instead of coercing an unknown count to zero',
  );
  assert(
    source.includes('run.scenarioResearchStatus')
      && source.includes('scenarioResearchEvidence?.proofBoundaries')
      && source.includes('value.boundaryEvaluator'),
    'the run drawer must consume the server-owned Research status, evidence, and boundary',
  );
});

test('Research V4 render handoff preserves both immutable authorities', async () => {
  const sourceRoot = new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/',
    import.meta.url,
  );
  const [mainSource, modelSource] = await Promise.all([
    readFile(new URL('main.tsx', sourceRoot), 'utf8'),
    readFile(new URL('features/research/model.ts', sourceRoot), 'utf8'),
  ]);
  assert(
    modelSource.includes("| 'research_scenario_result'")
      && modelSource.includes('authorityReceipt?: ResearchScenarioAuthorityReceiptResponse;'),
    'the prepared render model must retain the canonical V4 source and authority receipt',
  );
  assert(
    mainSource.includes('...(authorityReceipt?{authorityReceipt}: {})')
      && mainSource.includes('prepared.proofArtifact,\n        prepared.authorityReceipt,'),
    'the visible Render video action must submit proof and operator authority together',
  );
});

test('Research result deep links fetch results outside the initial page', async () => {
  const source = await readFile(
    new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/main.tsx',
      import.meta.url,
    ),
    'utf8',
  );
  assert(
    source.includes(
      'api<ResearchResult>(`/api/v2/research-results/${encodeURIComponent(code)}`)',
    ) && source.includes('.then(openResearchResult)'),
    'a valid result deep link must not depend on the initial 20-row cache',
  );
});

test('Research audit resolves only the selected proof utility copy', async () => {
  const source = await readFile(
    new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/features/research/audit.tsx',
      import.meta.url,
    ),
    'utf8',
  );
  assert(
    source.includes('switch (selected)')
      && !source.includes("kabout: t('workflow.research_scenario_option_utility_kabout')")
      && source.includes("case 'kabout': return t('workflow.research_scenario_option_utility_kabout');")
      && source.includes('switch (classification)')
      && !source.includes("incomplete: t('workflow.research_scenario_support_incomplete')")
      && source.includes("operatorCopy('research_scenario_support_incomplete')"),
    'an inactive option must not require localization copy from another bundle',
  );
});

test('Research terminal audit results remain separate from qualified result counters', async () => {
  const sourceRoot = new URL(
    '../../src/baloot_platform/service/foundry_dashboard/src/features/research/',
    import.meta.url,
  );
  const [modelSource, auditSource] = await Promise.all([
    readFile(new URL('model.ts', sourceRoot), 'utf8'),
    readFile(new URL('audit.tsx', sourceRoot), 'utf8'),
  ]);
  assert(
    modelSource.includes('auditResultCount?: number;')
      && modelSource.includes('auditResultCodes?: string[];'),
    'terminal audit identities must not be folded into the qualified result contract',
  );
  assert(
    auditSource.includes('run.auditResultCount')
      && auditSource.includes('run.auditResultCodes')
      && auditSource.includes("t('workflow.laffa_library_audit_only')"),
    'the run drawer must expose terminal audit evidence with governed copy',
  );
});
