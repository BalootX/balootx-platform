import type {JsonObject} from '../../shared/json';

export type RenderJob = {
  jobId: string;
  status: string;
  progress?: number;
  request?: JsonObject;
  result?: JsonObject | null;
  error?: string | null;
  createdAt?: string;
};

export type RenderMediaEntry = {
  auditOnly?: true;
  basename: string;
  manifestUrl: string;
  playerHref: string;
  posterUrl: string;
  sourceCode: string;
  videoUrl: string;
};

export type RenderPlayerEntry = {
  ordinal: number;
  playerHref: string;
  sourceCode: string;
};

export function renderRetryKey(baseKey: string, job: Pick<RenderJob, 'jobId' | 'status'>): string {
  return ['failed', 'cancelled'].includes(job.status)
    ? `${baseKey}-retry-${job.jobId}`
    : baseKey;
}

export type ArchivedLaffa = {
  gameHash: string;
  serial: string;
  gameId: string;
  bundleHash: string;
  status: string;
  label?: string | null;
  createdAt: string;
  contract?: string | null;
  buyer?: string | null;
  roundNumber?: number | null;
  matchScore?: number[] | null;
};

export function renderJobReferencesResearch(
  job: RenderJob,
  resultCode: string,
): boolean {
  return renderJobReferencesSource(job, 'research_scenario_result', resultCode)
    || renderJobReferencesSource(job, 'research_result', resultCode);
}

export function renderJobReferencesSource(
  job: RenderJob,
  kind: string,
  code: string,
): boolean {
  const requestItems = Array.isArray(job.request?.items) ? job.request.items : [];
  return requestItems.some((value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
    const source = value.source;
    return Boolean(
      source
      && typeof source === 'object'
      && !Array.isArray(source)
      && String(source.kind || '') === kind
      && String(source.code || '') === code
    );
  });
}

export function renderBatchMediaEntries(job: RenderJob): RenderMediaEntry[] {
  const result = record(job.result);
  const outputBaseUrl = typeof result.outputBaseUrl === 'string'
    && /^\/outputs\/render-batches\/batch-[a-f0-9]+$/i.test(result.outputBaseUrl)
    ? result.outputBaseUrl
    : '';
  if (!outputBaseUrl) return [];
  const readyItems = Array.isArray(result.readyItems) ? result.readyItems : [];
  if (readyItems.length) {
    const requestItems = Array.isArray(job.request?.items) ? job.request.items : [];
    return readyItems.flatMap((value) => {
      const ready = record(value);
      const ordinal = ready.ordinal;
      const basename = typeof ready.basename === 'string'
        && /^[a-z0-9][a-z0-9._-]{0,79}$/i.test(ready.basename)
        ? ready.basename
        : '';
      const manifestName = typeof ready.manifestName === 'string'
        ? ready.manifestName
        : '';
      if (
        !Number.isInteger(ordinal)
        || Number(ordinal) < 0
        || Number(ordinal) >= requestItems.length
        || !basename
        || manifestName !== `${basename}-manifest.json`
        || !/^[a-f0-9]{64}$/i.test(String(ready.manifestSha256 || ''))
      ) return [];
      const requestItem = record(requestItems[Number(ordinal)]);
      const source = record(requestItem.source);
      const locator = record(requestItem.locator);
      return [{
        ...(source.auditOnly === true ? {auditOnly: true as const} : {}),
        basename,
        manifestUrl: `${outputBaseUrl}/${manifestName}`,
        playerHref: challengePlayerHref(locator.universalLink),
        posterUrl: `${outputBaseUrl}/${basename}.jpg`,
        sourceCode: typeof source.code === 'string' ? source.code : '',
        videoUrl: `${outputBaseUrl}/${basename}.mp4`,
      }];
    });
  }
  if (job.status !== 'completed') return [];
  const batch = record(result.batch);
  const batchItems = Array.isArray(batch.items) ? batch.items : [];
  const requestItems = Array.isArray(job.request?.items) ? job.request.items : [];
  const manifests = Array.isArray(result.manifests) ? result.manifests : [];
  return batchItems.flatMap((value, index) => {
    const item = record(value);
    const basename = typeof item.basename === 'string'
      && /^[a-z0-9][a-z0-9._-]{0,79}$/i.test(item.basename)
      ? item.basename
      : '';
    if (!basename) return [];
    const requestItem = record(requestItems[index]);
    const source = record(requestItem.source);
    const locator = record(requestItem.locator ?? item.locator);
    const manifest = manifests[index];
    return [{
      ...(source.auditOnly === true ? {auditOnly: true as const} : {}),
      basename,
      manifestUrl: typeof manifest === 'string' && manifest.startsWith(`${outputBaseUrl}/`)
        ? manifest
        : '',
      playerHref: challengePlayerHref(locator.universalLink),
      posterUrl: `${outputBaseUrl}/${basename}.jpg`,
      sourceCode: typeof source.code === 'string' ? source.code : '',
      videoUrl: `${outputBaseUrl}/${basename}.mp4`,
    }];
  });
}

export function renderBatchPlayerEntries(job: RenderJob): RenderPlayerEntry[] {
  const requestItems = Array.isArray(job.request?.items) ? job.request.items : [];
  return requestItems.flatMap((value, ordinal) => {
    const item = record(value);
    const source = record(item.source);
    const locator = record(item.locator);
    const playerHref = challengePlayerHref(locator.universalLink);
    return playerHref ? [{
      ordinal,
      playerHref,
      sourceCode: typeof source.code === 'string' ? source.code : '',
    }] : [];
  });
}

export function challengePlayerHref(value: unknown): string {
  return typeof value === 'string' && /^\/c\/[A-Za-z0-9_-]+$/.test(value)
    ? value
    : '';
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}
