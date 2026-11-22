Follows engineering standard v1.2.

# BalootX delivery

## CI

[The CI workflow](.github/workflows/ci.yml) runs on GitHub-hosted runners.

| Event or lane | Checks |
| --- | --- |
| Ready pull request | Preflight, dependency review and Python version smoke. [The CI selector](scripts/ci_scope.py) adds affected Python, SDK, Foundry and worker checks. Unrecognized paths or an unresolved diff select the full profile. |
| Main push and merge group | Full profile with the Python wheelhouse, native build and sanitizers, fast, backend and heavy Python shards with exact coverage verification, SDK and Foundry checks, Remotion and video-worker parity, media rendering and release smoke. The merge group verifies its event head before running the profile. |
| Manual dispatch | Full diagnostics under `CI diagnostic result`, or the dedicated viewer-seat proof cohort when selected. |
| Media verification | The full profile builds the governed inputs, renders shards, checks repeatability and playback parity, decodes the sample media, and generates media and supply-chain evidence. |

The required status check is **CI result**. The check compares the producer results to the plan and fails on missing, failed, cancelled or unexpectedly skipped work. See [the current limits in the workflow](.github/workflows/ci.yml) and the [historical timing samples](ci/TIMINGS.md).

## Merge

- Use the `main` merge queue with **SQUASH** and **ALLGREEN**. Enable auto-merge against the exact head after green PR checks and, for [risk paths](AGENTS.md#risk-paths), one independent Opus full-diff review recorded on the PR at that head. The merge rule requires zero human approvals, and the Opus review is the only additional review requirement for risk paths.
- Disable auto-merge and dequeue whenever a PR head moves or a review sends it back. Requalify the replacement head before re-enabling auto-merge.

## Media release

- Map release-candidate tags to dev for validation. The public CI media lane verifies artifacts. TikTok publishing uses the release gate.
- Review the TikTok release gate once and obtain one Portfolio GO before activation. The gate then releases each passing clip automatically and records the result and the publication ledger hash in an audit row. A new kind of release requires a new GO.
- Require independent media QA, [engine-interest and viewer-seat proof](docs/RESEARCH_SCENARIO_V4.md#engine-interest-v1), the [approved clip presentation](DESIGN.md#rendered-clips), caption policy and a current per-platform publication history. Hold a failing or unverified clip and record the reason. Keep publication evidence in the release system.

## Rollback

- For a bad code release, start a revert branch at current `main`, run `git revert <squash-merge-sha>` and submit the revert through the [merge queue](#merge).
- For a media release regression, stop affected releases and restore the last approved release configuration and artifact pins. Reconcile any completed or uncertain publish in the audit ledger before resuming.

## Deviations

| Rule | BalootX reason | Review date |
| --- | --- | --- |
| Required check named `ci-gate` | `CI result` is the existing protected context. Rename the workflow check and its ruleset binding together. | 2026-12-22 |
| Measured per-job limits | Per-job p95 for setting the timeouts is not yet measured. The timing samples above cover earlier workflow runs. | 2026-12-22 |
| Staging environment | Release candidates use dev. The project has no separate staging deploy. | 2026-12-22 |
| High-severity code-scanning ruleset | The active main ruleset has no code-scanning requirement. Review how to add the rule with its check producer. | 2026-12-22 |
| Privacy deny check in CI on every branch push and merge group | The [public push guard](AGENTS.md#privacy-and-public-pushes) scans candidate commits. The tracked CI workflow has no equivalent denial step. | 2026-12-22 |
