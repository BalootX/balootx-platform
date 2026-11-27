// The one owner of the browser API deadlines. Page helpers load this module through
// /shared-assets/request-loader.js and the Foundry bundle imports it at build time.
// tests/test_video_motion_release.py reads the server budgets from their owners and
// fails when a deadline here stops covering them.

// Interactive calls. The server answers them inside its interactive validation
// budget, which the shipped profiles cap at a few seconds.
export const REQUEST_DEADLINE_MS = 60_000;

// Synchronous Research calls that run a solver or prepare a candidate before they
// answer. The longest, :verify-play-compiler, runs the project compiler and then the
// play compiler, so the deadline exceeds the sum of both solver budgets by the margin
// below. The margin covers evidence storage and transport.
export const LONG_CALL_MARGIN_MS = 30_000;
export const LONG_CALL_DEADLINE_MS = 120_000;

export function isDeadlineError(error) {
  return error?.name === 'TimeoutError';
}

function fetchUntil(deadline, input, init = {}) {
  // A Request carries its own signal; passing init.signal would replace it.
  const caller = init.signal ?? input?.signal;
  return fetch(input, {...init, signal: caller ? AbortSignal.any([caller, deadline]) : deadline});
}

export function fetchWithDeadline(input, init) {
  return fetchUntil(AbortSignal.timeout(REQUEST_DEADLINE_MS), input, init);
}

export function fetchWithLongDeadline(input, init) {
  return fetchUntil(AbortSignal.timeout(LONG_CALL_DEADLINE_MS), input, init);
}
