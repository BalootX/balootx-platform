// The one owner of the browser API deadline. Page helpers load this module from
// /shared-assets/request.js and the Foundry bundle imports it at build time.
// 60 s matches the read timeout common to reverse proxies in front of the service.
export const REQUEST_DEADLINE_MS = 60_000;

/** True when a request or its body read was ended by the shared deadline. */
export function isDeadlineError(error) {
  return error?.name === 'TimeoutError';
}

/**
 * fetch() with the shared deadline. A caller signal keeps working: whichever
 * fires first ends the request, and only the deadline reports TimeoutError.
 */
export function fetchWithDeadline(input, init = {}) {
  const deadline = AbortSignal.timeout(REQUEST_DEADLINE_MS);
  // A Request carries its own signal; passing init.signal would replace it.
  const caller = init.signal ?? input?.signal;
  return fetch(input, {
    ...init,
    signal: caller ? AbortSignal.any([caller, deadline]) : deadline,
  });
}
