Follows engineering standard v1.20.

# BalootX delivery

## CI

[The CI workflow](.github/workflows/ci.yml) runs on GitHub-hosted runners.

| Event or lane | Checks |
| --- | --- |
| Ready pull request | Preflight, dependency review and Python version smoke. [The CI selector](scripts/ci_scope.py) adds affected Python, SDK, Foundry and worker checks. Unrecognized paths or an unresolved diff select the full profile. |
| Main push and merge group | Full profile with the Python wheelhouse, native build and sanitizers, fast, backend and heavy Python shards with exact coverage verification, SDK and Foundry checks, Remotion and video-worker parity, media rendering and release smoke. The merge group verifies its event head before running the profile. |
| Manual dispatch | Full diagnostics under `CI diagnostic result`, or the dedicated viewer-seat proof cohort when selected. |
| [Nightly at 00:17 UTC or manual trigger](.github/workflows/nightly-candidates.yml) | Generates and proves synthetic deals on GitHub-hosted runners. A successful run seals at least 14 fully checked clips across at least three mechanisms and retains the pool for three days. Live variety and caption checks run on the PC. |
| Media verification | The full profile builds the governed inputs, renders shards, checks repeatability and playback parity, decodes the sample media, and generates media and supply-chain evidence. |

The required **ci-gate** check compares the producer results to the plan and fails on missing, failed, cancelled or unexpectedly skipped work. See [the current limits in the workflow](.github/workflows/ci.yml) and the [historical timing samples](ci/TIMINGS.md).

The main ruleset's CodeQL rule evaluates each PR at admission and blocks new high or critical alerts on lines the PR changes. The main-push scan surfaces alerts that arise only in the combined commit.

The concurrency key uses the PR number to keep same-named branches from different forks in separate groups. A new run supersedes only the same PR's older run. Each push, merge-group and manual CI run has its own group and cannot cancel another run.

Fix failed checks at the source. Do not rerun failures, widen deadlines or skip tests to get CI green. Keep authorization, privacy, migration and concurrency tests. Removing a test that only detects change requires two weeks of evidence. Follow the [test data and service rules](AGENTS.md#data-safety). Pin actions by full commit SHA, keep workflow defaults read-only, and grant writes only to jobs that need them. Keep secrets off fork runs and untrusted code off privileged triggers. A new `tests/test_*.py` file needs both `seconds` and `shardSeconds` entries in [.github/python-test-durations.json](.github/python-test-durations.json).

## Merge

- The `main` merge queue is the only path to trunk and uses squash merges only. Workers push their own short-lived branches without rewriting history and open PRs under the project's App identity. Merge, split or close an open PR within two days of its first push. The ruleset requires zero human approvals.
- Only critical changes get a review. Critical means production, secrets, real data, making anything public, host or network changes, repo settings, and release gates. See [critical paths](AGENTS.md#risk-paths). Get one independent full-diff review. The worker enables auto-merge only after the reviewer posts a verdict comment naming the reviewed head SHA. Turn off auto-merge before any later push to the PR. A later push touching a critical path needs a new verdict on that head. For other pushes, the recorded verdict carries forward. Verify fixes to review findings with tests.
- For everything else, the thread that made the change opens its own PR and enables auto-merge at once. The PR merges through the queue once the required CI checks are green, with no review.
- Delivery owns the queue, CI health and speed, releases, deploys, rollback and alerts. Workers handle each PR.
- Use `Closes #n` in the PR body only if merging meets the item's whole Done-when. Otherwise use `Refs #n`.

## GitHub Releases

- `public-switch-lock-tag` blocks creating, moving or deleting tags outside `v*`. `release-tags-create` lets only the project's App create `v*` tags. `release-tags-immutable` blocks everyone from moving or deleting them.
- The App creates an annotated `vX.Y.Z` tag at the milestone's approved `main` commit, then publishes the GitHub Release on that tag with `gh release create --verify-tag`. Link the milestone in the release notes.

## Media release

- Live clip publication requires individual release approval until the clip release gate is approved. Once approved, the gate can admit clips for publication. A production deploy requires the project owner's explicit approval.
- The `nightly-synthetic-candidates` artifact contains synthetic render evidence. Check `SHA256SUMS.txt` for the complete pool and use `candidate-index.json` to bind the exact source tree and qualified media. Each render uses the checkout's locked renderer and the [shared synthetic name pool](examples/synthetic-clip-identity.json). The hosted pool is marked `NEVER_POST` and carries no posting ledger or caption pool.
- The PC task runs `python -B -m scripts.pick_nightly_candidates` from the checkout with the package installed. Pass `--ledger`, `--caption-pool`, `--gate`, `--policy`, and `--output` from the existing private release configuration. Set `GH_TOKEN` only for this process using the repository-scoped, Actions-read App token. The picker verifies the newest successful scheduled pool from main, uses the pinned release gate against the live ledger, and writes a private two-day pick with distinct reserves. For manual evidence, pass a downloaded pool directory with `--artifact`. Both paths produce `NEVER_POST` picks, which the next stage must validate against its own receiving rules.
- The clip gate keeps its audit rows in a private evidence store. Each row records the gate and ledger hashes.
- Require independent media QA, [engine-interest and viewer-seat proof](docs/RESEARCH_SCENARIO_V4.md#engine-interest-v1), the [approved clip presentation](DESIGN.md#rendered-clips), caption policy and a current per-platform publication history. Hold a failing or unverified clip and record the reason.

## Rollback

- Revert the change that broke main or the queue if the failure remains after 30 minutes. For a bad code release, start a revert branch at current `main`, run `git revert <squash-merge-sha>` and submit the revert through the [merge queue](#merge).
- For a media release regression, stop affected releases and restore the last approved release configuration and artifact pins. Reconcile any completed or uncertain publish in the audit ledger before resuming.

## Deviations

| Rule | BalootX reason | Review date |
| --- | --- | --- |
| `ci-gate` runs without a skip condition | The job skips the manual viewer-seat proof lane, which runs its own proof checks. | 2026-12-22 |
| Measured per-job limits in `.github/ci-timings.json` | Per-job p95 is still pending. The timing samples above cover earlier workflow runs. | 2026-12-04 |
| Release tags and GitHub Releases | The first milestone release tag and GitHub Release are pending. Clips follow the [media release path](#media-release). | 2026-12-22 |
| GitHub deployment environments | No GitHub environments are configured. The [clip release path](#media-release) runs outside GitHub. | 2026-12-22 |
| One attested artifact across environments | No single attested artifact is promoted across environments. Clip releases use individually verified media artifacts. | 2026-12-22 |
| Committed Python lockfile and locked CI installs | There is no Python lockfile. CI resolves dependencies from the ranges in [pyproject.toml](pyproject.toml) for installs and wheel builds. | 2026-12-22 |
| Tracked pre-commit and pre-push hooks | The repository ships neither hook. Hosted preflight runs lint and focused tests, and the [privacy guard](AGENTS.md#privacy-and-public-pushes) checks pushes. | 2026-12-22 |
| Privacy deny check in CI on every branch push and merge group | The push guard relies on a private deny list. Hosted enforcement needs a way to check fork changes while keeping that list private. | 2026-12-22 |
| PR size targets | PR size p50 and p95 are pending. Generated files lack size-exclusion markers. | 2026-12-04 |
| Scoped URL and IP literal check | The push guard checks private values. The source literal check and committed allowlist are pending. | 2026-12-04 |
| Configured no-reply domains for commit identities and trailers | The push guard checks commit metadata with a fixed address policy. | 2026-12-04 |
| Affected CI on draft PRs | The affected checks start when a PR is ready. | 2026-12-04 |
| Critical review coverage | The risk-path list leaves gaps for retention, audit logs and backups that affect real data. | 2026-12-04 |
