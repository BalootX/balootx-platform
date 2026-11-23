Follows engineering standard v1.10.

# BalootX delivery

## CI

[The CI workflow](.github/workflows/ci.yml) runs on GitHub-hosted runners.

| Event or lane | Checks |
| --- | --- |
| Ready pull request | Preflight, dependency review and Python version smoke. [The CI selector](scripts/ci_scope.py) adds affected Python, SDK, Foundry and worker checks. Unrecognized paths or an unresolved diff select the full profile. |
| Main push and merge group | Full profile with the Python wheelhouse, native build and sanitizers, fast, backend and heavy Python shards with exact coverage verification, SDK and Foundry checks, Remotion and video-worker parity, media rendering and release smoke. The merge group verifies its event head before running the profile. |
| Manual dispatch | Full diagnostics under `CI diagnostic result`, or the dedicated viewer-seat proof cohort when selected. |
| [Nightly at 00:30 UTC or manual trigger](.github/workflows/nightly-candidates.yml) | Searches seeded synthetic deals for up to four clips and retains the candidate index, exact viewer-seat receipts, calibrated hook evidence and passing media for seven days. The provisional score gives full variety credit when publication history is unavailable. Publication QA is pending. |
| Media verification | The full profile builds the governed inputs, renders shards, checks repeatability and playback parity, decodes the sample media, and generates media and supply-chain evidence. |

The required **ci-gate** check compares the producer results to the plan and fails on missing, failed, cancelled or unexpectedly skipped work. See [the current limits in the workflow](.github/workflows/ci.yml) and the [historical timing samples](ci/TIMINGS.md).

The main ruleset's CodeQL rule evaluates each PR at admission and blocks new high or critical alerts on lines the PR changes. The main-push scan surfaces alerts that arise only in the combined commit.

The concurrency key uses the PR number to keep same-named branches from different forks in separate groups. A new run supersedes only the same PR's older run. Each push, merge-group and manual CI run has its own group and cannot cancel another run.

Fix failed checks at the source. Do not rerun failures, widen deadlines or skip tests to get CI green. Keep authorization, privacy, migration and concurrency tests. Removing a test that only detects change requires two weeks of evidence. Follow the [test data and service rules](AGENTS.md#data-safety). Pin actions by full commit SHA, keep workflow defaults read-only, and grant writes only to jobs that need them. Keep secrets off fork runs and untrusted code off privileged triggers. A new `tests/test_*.py` file needs both `seconds` and `shardSeconds` entries in [.github/python-test-durations.json](.github/python-test-durations.json).

## Merge

- The `main` merge queue is the only path to trunk. Use a short-lived PR branch and push without rewriting history. Merge, split or close an open PR within two days of its first push. Enable auto-merge against the exact head after green PR checks. For [risk paths](AGENTS.md#risk-paths), first record an independent full-diff review at that head by the designated risk-path reviewer. The ruleset requires zero human approvals.
- Disable auto-merge and dequeue whenever a PR head moves or a review sends it back. Requalify the replacement head before re-enabling auto-merge.
- Use `Closes #n` in the PR body only if merging meets the item's whole Done-when. Otherwise use `Refs #n`.

## Media release

- Live clip publication requires individual release approval until the clip release gate is approved. Once approved, the gate can admit clips for publication. A production deploy requires the project owner's explicit approval.
- Nightly artifacts contain candidate evidence only. Use the run's `candidate-index.json` to resolve the exact source tree, per-candidate reasons and file hashes. Only IDs in `qualified` have passed the hosted candidate checks.
- The clip gate keeps its audit rows in a private evidence store. Each row records the gate and ledger hashes.
- Require independent media QA, [engine-interest and viewer-seat proof](docs/RESEARCH_SCENARIO_V4.md#engine-interest-v1), the [approved clip presentation](DESIGN.md#rendered-clips), caption policy and a current per-platform publication history. Hold a failing or unverified clip and record the reason.

## Rollback

- Revert the change that broke main or the queue if the failure remains after 30 minutes. For a bad code release, start a revert branch at current `main`, run `git revert <squash-merge-sha>` and submit the revert through the [merge queue](#merge).
- For a media release regression, stop affected releases and restore the last approved release configuration and artifact pins. Reconcile any completed or uncertain publish in the audit ledger before resuming.

## Deviations

| Rule | BalootX reason | Review date |
| --- | --- | --- |
| `ci-gate` runs without a skip condition | The job skips the manual viewer-seat proof lane, which runs its own proof checks. | 2026-12-22 |
| Measured per-job limits | Per-job p95 for setting the timeouts is not yet measured. The timing samples above cover earlier workflow runs. | 2026-12-22 |
| Release tags and GitHub Releases | There are no release tags or GitHub Releases, and the tag lock blocks tag creation. The release unit is a clip. | 2026-12-22 |
| GitHub deployment environments | No GitHub environments are configured. The [clip release path](#media-release) runs outside GitHub. | 2026-12-22 |
| One attested artifact across environments | No single attested artifact is promoted across environments. Clip releases use individually verified media artifacts. | 2026-12-22 |
| Committed Python lockfile and locked CI installs | There is no Python lockfile. CI resolves dependencies from the ranges in [pyproject.toml](pyproject.toml) for installs and wheel builds. | 2026-12-22 |
| Tracked pre-commit and pre-push hooks | The repository ships neither hook. Hosted preflight runs lint and focused tests, and the [privacy guard](AGENTS.md#privacy-and-public-pushes) checks pushes. | 2026-12-22 |
| Privacy deny check in CI on every branch push and merge group | The push guard relies on a private deny list. Hosted enforcement needs a way to check fork changes while keeping that list private. | 2026-12-22 |
