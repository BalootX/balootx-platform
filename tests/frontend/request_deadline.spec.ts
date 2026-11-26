import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {afterEach, test, vi} from 'vitest';
import {
  LONG_CALL_DEADLINE_MS,
  REQUEST_DEADLINE_MS,
  fetchWithDeadline,
  fetchWithLongDeadline,
  isDeadlineError,
} from '../../src/baloot_platform/service/static/shared/request.js';
import {researchRecipeErrorCopyKey} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/recipes.js';

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

// A request that never answers: it settles only when its signal aborts.
function stallFetch(): AbortSignal[] {
  const signals: AbortSignal[] = [];
  vi.stubGlobal('fetch', (_input: unknown, init: RequestInit) => {
    const signal = init.signal as AbortSignal;
    signals.push(signal);
    return new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
    });
  });
  return signals;
}

// The deadline fires when the test says so, so no test waits on a real clock.
function controlDeadline(): {requested: number[]; expire: () => void} {
  const deadline = new AbortController();
  const requested: number[] = [];
  vi.spyOn(AbortSignal, 'timeout').mockImplementation((milliseconds: number) => {
    requested.push(milliseconds);
    return deadline.signal;
  });
  return {
    requested,
    expire: () => deadline.abort(new DOMException('deadline', 'TimeoutError')),
  };
}

const failure = (pending: Promise<unknown>): Promise<unknown> =>
  pending.then(() => null, (reason: unknown) => reason);

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test('a stalled request ends once at the shared deadline', async () => {
  const signals = stallFetch();
  const deadline = controlDeadline();
  const pending = failure(fetchWithDeadline('/api/v1/auth/me', {credentials: 'include'}));
  deadline.expire();
  const error = await pending;
  assert(
    deadline.requested.length === 1 && deadline.requested[0] === REQUEST_DEADLINE_MS,
    'the request must use the one shared deadline value',
  );
  assert(isDeadlineError(error), 'an expired deadline must be reported as a deadline error');
  assert(signals.length === 1, 'a timed-out request must not be retried');
  assert(
    researchRecipeErrorCopyKey(error) === 'helpers.request_timeout'
      && researchRecipeErrorCopyKey({detail: error}) === 'helpers.request_timeout',
    'Foundry must map a deadline, direct or reported by the generated client, to the timeout copy',
  );
});

test('a caller abort still ends the request and is not a deadline', async () => {
  stallFetch();
  controlDeadline();
  const caller = new AbortController();
  const pending = failure(fetchWithDeadline('/api/v1/auth/me', {signal: caller.signal}));
  caller.abort();
  const error = await pending;
  assert((error as Error | null)?.name === 'AbortError', 'the caller abort must end the request');
  assert(!isDeadlineError(error), 'a caller abort must not be reported as a deadline');
});

test('a Request keeps its own signal and gains the deadline', async () => {
  stallFetch();
  const deadline = controlDeadline();
  const caller = new AbortController();
  const aborted = failure(fetchWithDeadline(
    new Request('http://localhost/api/v2/research-recipes', {signal: caller.signal}),
  ));
  caller.abort();
  assert(!isDeadlineError(await aborted), 'the signal a Request carries must still abort it');
  const expired = failure(fetchWithDeadline(
    new Request('http://localhost/api/v2/research-recipes'),
  ));
  deadline.expire();
  assert(isDeadlineError(await expired), 'a Request must end at the shared deadline');
});

test('a long call ends at the long-call deadline, not the interactive one', async () => {
  stallFetch();
  const deadline = controlDeadline();
  const pending = failure(fetchWithLongDeadline(
    '/api/v2/research-results/RES-1:verify-play-compiler',
    {method: 'POST'},
  ));
  deadline.expire();
  assert(isDeadlineError(await pending), 'an expired long call must be reported as a deadline error');
  assert(
    deadline.requested.length === 1 && deadline.requested[0] === LONG_CALL_DEADLINE_MS,
    'a long call must use the long-call deadline value',
  );
  assert(
    LONG_CALL_DEADLINE_MS > REQUEST_DEADLINE_MS,
    'the long-call deadline must be longer than the interactive one',
  );
});

test('a failed helper load is not kept: the next request loads it again', async () => {
  type Loader = () => Promise<unknown>;
  const page: {createRequestHelperLoader?: (load: (specifier: string) => Promise<unknown>) => Loader} = {};
  // The pages run this file as a classic script, so the test evaluates the shipped text.
  runInNewContext(readFileSync(new URL(
    '../../src/baloot_platform/service/static/shared/request-loader.js',
    import.meta.url,
  ), 'utf8'), page);
  assert(page.createRequestHelperLoader, 'the loader script must define its factory');
  const helper = {fetchWithDeadline, isDeadlineError};
  const offline = new TypeError('Failed to fetch dynamically imported module');
  const requested: string[] = [];
  let reachable = false;
  const loadRequestHelper = page.createRequestHelperLoader((specifier) => {
    requested.push(specifier);
    return reachable ? Promise.resolve(helper) : Promise.reject(offline);
  });

  const failed = await Promise.all([failure(loadRequestHelper()), failure(loadRequestHelper())]);
  assert(failed[0] === offline && failed[1] === offline, 'a failed load must reject the request');
  assert(requested.length === 1, 'requests made while a load is pending must share it');

  reachable = true;
  assert(await loadRequestHelper() === helper, 'the request after a failed load must load again');
  assert(await loadRequestHelper() === helper, 'a loaded helper must be kept');
  assert(requested.length === 2, 'a loaded helper must not be fetched again');
  assert(
    requested[0] === '/shared-assets/request.js'
      && requested[1] !== requested[0]
      && requested[1].startsWith('/shared-assets/request.js?'),
    'the retry must use a new URL, because the module map keeps a failed fetch',
  );
});
