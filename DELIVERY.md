Follows engineering standard v1.2.

# BalootX delivery

## CI

[The CI workflow](.github/workflows/ci.yml) runs on GitHub-hosted runners.

| Event or lane | Checks |
| --- | --- |
| Ready pull request | Preflight, dependency review and Python version smoke. [The CI selector](scripts/ci_scope.py) adds affected Python, SDK, Foundry and worker checks. Unrecognized paths or an unresolved diff select the full profile. |
| Main push and merge group | Full profile with the Python wheelhouse, native build and sanitizers, fast, backend and heavy Python shards with exact coverage verification, SDK and Foundry checks, Remotion and video-worker parity, media rendering and release smoke. The merge group verifies its event head before running the profile. |
| Manual dispatch | Full diagnostics under `CI diagnostic result`, or the dedicated viewer-seat proof cohort when selected. |
| [Nightly at 00:30 UTC](.github/workflows/nightly-candidates.yml) | Searches seeded synthetic deals for up to four clips and retains the candidate index, exact viewer-seat receipts, calibrated hook evidence and passing media for seven days. The score is provisional while publication history and publication QA are pending. |
| Media verification | The full profile builds the governed inputs, renders shards, checks repeatability and playback parity, decodes the sample media, and generates media and supply-chain evidence. |

The **ci-gate** check compares the producer results to the plan and fails on missing, failed, cancelled or unexpectedly skipped work. During the ruleset cutover, **CI result** also reports the gate's verdict. See [the current limits in the workflow](.github/workflows/ci.yml) and the [historical timing samples](ci/TIMINGS.md).

Concurrency groups use the PR number for pull requests and a unique run ID for other events.

## Merge

- Use the `main` merge queue. Enable auto-merge against the exact head after green PR checks and, for [risk paths](AGENTS.md#risk-paths), one independent Opus full-diff review recorded on the PR at that head. The merge rule requires zero human approvals, and the Opus review is the only additional review requirement for risk paths.
- Disable auto-merge and dequeue whenever a PR head moves or a review sends it back. Requalify the replacement head before re-enabling auto-merge.

## Media release

- Clips are published outside this repository. Each clip requires individual release approval until the clip release gate is reviewed and Portfolio approves its activation. Once active, the gate releases passing clips automatically.
- Nightly artifacts contain candidate evidence only. Use the run's `candidate-index.json` to resolve the exact source tree, per-candidate reasons and file hashes. Only IDs in `qualified` have passed the hosted candidate checks.
- The clip gate's audit rows live in a private evidence store on the posting machine. Each row records the gate and ledger hashes.
- Require independent media QA, [engine-interest and viewer-seat proof](docs/RESEARCH_SCENARIO_V4.md#engine-interest-v1), the [approved clip presentation](DESIGN.md#rendered-clips), caption policy and a current per-platform publication history. Hold a failing or unverified clip and record the reason.

## Rollback

- For a bad code release, start a revert branch at current `main`, run `git revert <squash-merge-sha>` and submit the revert through the [merge queue](#merge).
- For a media release regression, stop affected releases and restore the last approved release configuration and artifact pins. Reconcile any completed or uncertain publish in the audit ledger before resuming.

## Deviations

| Rule | BalootX reason | Review date |
| --- | --- | --- |
| Required check named `ci-gate` | The workflow publishes both `CI result` and `ci-gate`. The ruleset requires `CI result` until settings step `PUT-01` is applied and verified. | 2026-12-22 |
| Gate job runs without a skip condition | The result job skips the manual viewer-seat proof lane, which runs its own proof checks. | 2026-12-22 |
| Measured per-job limits | Per-job p95 for setting the timeouts is not yet measured. The timing samples above cover earlier workflow runs. | 2026-12-22 |
| Release tags and GitHub Releases | There are no release tags or GitHub Releases, and the tag lock blocks tag creation. The release unit is a clip. | 2026-12-22 |
| GitHub deployment environments | No GitHub environments are configured. The [clip release path](#media-release) runs outside GitHub. | 2026-12-22 |
| One attested artifact across environments | No single attested artifact is promoted across environments. Clip releases use individually verified media artifacts. | 2026-12-22 |
| Committed Python lockfile and locked CI installs | There is no Python lockfile. CI resolves dependencies from the ranges in [pyproject.toml](pyproject.toml) for installs and wheel builds. | 2026-12-22 |
| Tracked pre-commit and pre-push hooks | The repository ships neither hook. Hosted preflight runs lint and focused tests, and the [privacy guard](AGENTS.md#privacy-and-public-pushes) checks pushes. | 2026-12-22 |
| High-severity code-scanning ruleset | The active main ruleset has no code-scanning requirement. Review how to add the rule with its check producer. | 2026-12-22 |
| Privacy deny check in CI on every branch push and merge group | The tracked CI workflow has no equivalent to the pre-push privacy scan. | 2026-12-22 |
