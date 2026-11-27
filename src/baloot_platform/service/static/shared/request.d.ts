export declare const REQUEST_DEADLINE_MS: number;
export declare const LONG_CALL_MARGIN_MS: number;
export declare const LONG_CALL_DEADLINE_MS: number;
export declare function isDeadlineError(error: unknown): boolean;
export declare const fetchWithDeadline: typeof fetch;
export declare const fetchWithLongDeadline: typeof fetch;
