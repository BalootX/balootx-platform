# Local publisher

Portable publisher source and mock tests live here. Live upload uses the operator's
own signed-in local Chrome and separately installed helper and kernel-backed lock.
There is no hosted browser, account login, session export, or CI posting path.
Importing these modules does not connect to a browser. The hosted workflow runs
synthetic Node unit tests only, including for fork pull requests, with read-only
repository permission and no secrets.

Run the focused tests with Node 22 or newer:

```sh
node --test --test-concurrency=1 tests/publisher/*.test.mjs
```

## Local configuration

All runtime files stay outside the source tree. `operator.mjs` requires an absolute
`--config` path; live commands reject `CI` and `GITHUB_ACTIONS`. No default path,
account, browser target, media, ledger, or schedule is shipped. Configuration has:

| Field | Operator-supplied value |
| --- | --- |
| `localOnly` | `true` |
| `bindingPath` | Absolute path to the current release binding |
| `helperPath`, `helperSha256` | Installed single-tab helper and exact source hash |
| `lockModulePath`, `lockModuleSha256` | Installed exclusive-lock module and source hash |
| `lockPath`, `lockOwner` | Local kernel-backed claim path and expected owner |
| `controlUrl` | Explicit HTTP IPv4 loopback URL with a chosen port |
| `protectedTargetIds` | Local list of targets the helper must never use |
| `allowedOrigins` | Exact parsed origins allowed for navigation |
| `startUrl`, `profileCheckUrl`, `profileSuffix` | Local profile verification inputs |
| `profileUrls`, `uploadUrls` | Exact URLs keyed by platform |
| `applicationAckTexts` | Observed application acknowledgment text arrays keyed by platform |
| `operatorAdapterPath`, `operatorAdapterSha256` | Pinned local scheduling entry point |
| `schedulerAdapterPath`, `schedulerAdapterSha256` | Optional pinned local registration adapter |

The binding carries `platforms`, `account`, `dealNumber`, `deal`, `slotUtc`,
`scheduledTaskId`, `mediaPath`, `mediaName`, `mediaSha256`, `mediaBytes`,
`approvalPath`, `originalCaptionSha256`, `manifestPath`, `captionPath`,
`captionSha256`, `captionCharacters`, `captionUtf8Bytes`, `captionPoolIndex`,
`releasePath`, `releaseSha256`, `contentRequestPath`, `contentRequestSha256`,
`contentHandoffPath`, `contentHandoffSha256`, `receivingPath`, `executionPinPath`, `evidenceRoot`,
`ledgerPath`, `admissionProofPaths`, `opener`, `familyCases`, `knownMinimumIds`,
and `inputPins`. File paths are absolute local paths. The reviewed execution pin
binds the exact configuration, binding, current release, task, slot, platforms,
and every publisher module's source hash (`./filename.mjs`). It uses status
`REVIEWED_EXECUTION_FREEZE` and `liveEnabled: true`; this tool never grants it.
The release/request/manifest checks retain the qualified publisher contracts.

`captionPath` names an approved pool with status `APPROVED_CAPTION_POOL` and
`captions: [{index, caption, platforms}]`. Pool indices must be unique. The
selected caption is checked against the released SHA-256, JavaScript character
count and UTF-8 byte length; no trimming, rewriting, or generated fallback occurs.
Existing approved caption text can be placed in that local pool unchanged.

The ledger contract is [publisher-ledger-v1.json](../../schemas/publisher-ledger-v1.json).
It contains every known post, including removed posts, plus the released rows.
Local history imports must retain their post identity, caption hash, observed
opening cards, fingerprint, scenario family and actual posting time. Confirmed
new rows also carry the application acknowledgment and actual caption hash;
existing confirmed history can retain its original fields.
Do not remove history to admit a repeated opening. No live ledger is shipped.

## Operator commands

Pass the explicit local configuration to every command:

```text
node scripts/publisher/operator.mjs --config <absolute-local-config> --action schedule
node scripts/publisher/operator.mjs --config <absolute-local-config> --action register
node scripts/publisher/operator.mjs --config <absolute-local-config> --action helper
node scripts/publisher/operator.mjs --config <absolute-local-config> --platform <platform> --action <step>
```

`schedule` emits one due-time command per released platform with one attempt.
`register` passes that plan to the pinned local scheduler adapter's default
function. Neither command starts Chrome. `dispatch` calls the pinned operator
adapter's default function with `{client, runAction}`; the adapter integrates the
existing local scheduler and UI navigation. It must run the public commands under
the installed exclusive-lock wrapper, which supplies `CHROME_LOCK_PATH`,
`CHROME_LOCK_CLAIM_SHA256` and `CHROME_LOCK_HOLDER_PID`. No lock is acquired or
stolen by this code. `helper` extends the pinned helper in memory, retains its one
WebSocket and one owned tab, and checks the local claim on every request. It
accepts no cross-origin requests and requires the claim hash in the request.

Steps are `init`, `account`, `attach`, `fill`, `share`, and `readback` (the last
also requires `--url <actual-new-permalink>`). `observe` is read-only. The local
adapter owns platform navigation between steps; no global text click fallback
is provided. Every browser operation checks due time, same-day expiry, exact
source/current release authority and the local claim. Upload/final admission
also checks retained media approval, caption, current ledger row, full history,
and a fresh due-time actual predecessor observation from the owned session.
The second platform may reuse the opener only for the exact current released
pair; a planned predecessor never counts as the actual predecessor.

Caption entry uses native `Input.insertText`. Posting requires the exact active
composer, actual page counter and enabled submit control, then a fresh viewport
pointer and same-target hit test. Readiness waits at most 20 seconds and stops
immediately at sign-in/challenge prompts. `share` reserves an exclusive intent
before clicking once. Protocol acknowledgment is saved separately from the
application acknowledgment. `readback` checks the new permalink, author-bound
exact caption and 120-second deadline before atomically updating the ledger.
Unknown outcomes retain the intent and require operator reconciliation; there
is no automatic retry, deletion, or caption-edit fallback.

Source/unit acceptance does not establish live browser or posting qualification.
Local configuration and adapters must be bound to the admitted source before
an operator switches scheduling. This repository change activates no schedule.
