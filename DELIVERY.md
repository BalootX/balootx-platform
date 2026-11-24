Follows engineering standard v1.26.

# BalootX delivery

## CI

[The CI workflow](.github/workflows/ci.yml) runs on GitHub-hosted runners.

| Event or lane | Checks |
| --- | --- |
| Ready pull request | Preflight, dependency review and Python version smoke. [The CI selector](scripts/ci_scope.py) adds affected Python, SDK, Foundry and worker checks. Unrecognized paths or an unresolved diff select the full profile. |
| Main push and merge group | Full profile with the Python wheelhouse, native build and sanitizers, fast, backend and heavy Python shards with exact coverage verification, SDK and Foundry checks, Remotion and video-worker parity, media rendering and release smoke. The merge group verifies its event head before running the profile. |
| Manual dispatch | Full diagnostics under `CI diagnostic result`, or the dedicated viewer-seat proof cohort when selected. |
| [Nightly at 00:17 UTC or manual trigger](.github/workflows/nightly-candidates.yml) | Generates and proves synthetic deals on GitHub-hosted runners. A successful run seals at least 14 fully checked clips across at least three mechanisms and retains the pool for three days. The [viewer-seat proof cohort](.github/workflows/viewer-seat-proof.yml) also runs and retains one receipt per pinned fixture for 30 days. Run live variety and caption checks on a server runner. |
| Media verification | The full profile builds the governed inputs, renders shards, checks repeatability and playback parity, decodes the sample media, and generates media and supply-chain evidence. |

The required **ci-gate** check compares the producer results to the plan and fails on missing, failed, cancelled or unexpectedly skipped work. See [the current limits in the workflow](.github/workflows/ci.yml) and the [historical timing samples](ci/TIMINGS.md).

The `docker-constraints` job dry-runs pip with the Python base image, exact constraints and extras shared by the Docker test and verifier stages. It runs when its Docker, dependency or CI inputs change on a PR, when CI cannot resolve the diff, and on main and merge groups.

The Foundry job discovers the Vitest suite under `tests/frontend` using the same configuration as the release verifier. It runs the Playwright suite against the synthetic operator server and the dashboard built from the checkout. The `foundry-test-receipts` artifact contains per-case JUnit results for both runners and the browser latency samples.

The main ruleset's CodeQL rule evaluates each PR at admission and blocks new high or critical alerts on lines the PR changes. The main-push scan surfaces alerts that arise only in the combined commit.

The concurrency key uses the PR number to keep same-named branches from different forks in separate groups. A new run supersedes only the same PR's older run. Each push, merge-group and manual CI run has its own group and cannot cancel another run.

Fix failed checks at the source. Do not rerun failures, widen deadlines or skip tests to get CI green. Keep authorization, privacy, migration and concurrency tests. Removing a test that only detects change requires two weeks of evidence. Follow the [test data and service rules](AGENTS.md#data-safety). Pin actions by full commit SHA, keep workflow defaults read-only, and grant writes only to jobs that need them. Keep secrets off fork runs and untrusted code off privileged triggers. A new `tests/test_*.py` file needs both `seconds` and `shardSeconds` entries in [.github/python-test-durations.json](.github/python-test-durations.json).

### Nightly full suite

The full suite runs daily at 20:00 UTC on the `Servers isolated $0 Linux test host (credential-free, not a GitHub runner)`, capped at 2h20m. A systemd timer starts each run with a fresh depth-1 checkout of public `main` and a per-run virtual environment. Nightly runs stay off the PC and outside the merge gate. [#109](https://github.com/BalootX/balootx-platform/issues/109)

Servers sends BalootX QA the `main` SHA, test counts, failing test IDs, a JUnit file and a scrubbed log tail. BalootX QA files a Kind Bug issue titled `Nightly full suite red: <date>` the same day for any red or missing run and links the run when available. Treat flaky failures as High-priority bugs and fix or revert them. [#140](https://github.com/BalootX/balootx-platform/issues/140)

### Test coverage

BalootX QA owns tests at every layer, the end-to-end journey catalogue, coverage, red and flaky tests, nightly results, and the pre-release sweep. QA routes pipeline faults to BalootX Delivery.

- Add property and edge-case tests for code that handles untrusted or varied input, including parsers, limits, time zones and daylight saving transitions, schedules, slot sequencing and caption windows. Put long runs in the nightly profile.
- Every bug that reached `main` adds its failing case as a regression test or a golden-case row.

### CI-only PC rule

- Keep PC work to Git edits and commits plus the required pre-push deny check and privacy screen. Close shells when their step ends.
- Run tests, lint, scrubs, builds and renders through CI. On the PC, `run-locked.ps1` returns `SKIP_TO_CI`.
- Use free GitHub-hosted runners for public repo checks. Use zero-cost server runners for private repos, deploys and jobs that use secrets.
- If a check has no CI job, add one in the same PR.

## Work scheduling

A Due is the latest finish. Start Ready work at once and ship as soon as its checks pass. Every hold names its blocker and what clears it.

## Merge

- The `main` merge queue is the only path to trunk and uses squash merges only. Workers push their own short-lived branches without rewriting history and open PRs under the project's App identity. Merge, split or close an open PR within two days of its first push. The ruleset requires zero human approvals.
- Only changes in the [critical classes mapped in AGENTS.md](AGENTS.md#critical-paths) get one independent full-diff review in parallel with CI. The worker enables auto-merge after the verdict comment names the current head SHA. Turn off auto-merge before any later push. An existing verdict carries forward unless the push adds a new critical change. Verify fixes to review findings with tests.
- For everything else, the thread that made the change opens its own PR and enables auto-merge at once. The PR merges through the queue once the required CI checks are green, with no review.
- Delivery owns the queue, admission gates, pipeline faults, CI health and speed, releases, deploys, rollback and alerts. Workers handle each PR.
- Use `Closes #n` in the PR body only if merging meets the item's whole Done-when. Otherwise use `Refs #n`.

## GitHub Releases

- `public-switch-lock-tag` blocks creating, moving or deleting tags outside `v*`. `release-tags-create` lets only the project's App create `v*` tags. `release-tags-immutable` blocks everyone from moving or deleting them.
- The App creates an annotated `vX.Y.Z` tag at the milestone's approved `main` commit, then publishes the GitHub Release on that tag with `gh release create --verify-tag`. Link the milestone in the release notes.

## Media release

- The flagship and decision Remotion commands claim a new output directory for each invocation. An existing directory fails immediately. The Python drivers stage each render separately and move its completed files to the expected artifact paths.
- Live clip publication requires individual release approval until the clip release gate is approved. Once approved, the gate can admit clips for publication. A production deploy requires the project owner's explicit approval.
- The `nightly-synthetic-candidates` artifact contains synthetic render evidence. Check `SHA256SUMS.txt` for the complete pool and use `candidate-index.json` to bind the exact source tree and qualified media. Each render uses the checkout's locked renderer and the [shared synthetic name pool](examples/synthetic-clip-identity.json). The hosted pool is marked `NEVER_POST` and carries no posting ledger or caption pool.
- On a server runner, run `python -B -m scripts.pick_nightly_candidates` from the checkout with the package installed. Pass `--ledger`, `--caption-pool`, `--gate`, `--policy`, and `--output` from the existing private release configuration. Set `GH_TOKEN` only for this process using the repository-scoped, Actions-read App token. The picker verifies the newest successful scheduled pool from main, uses the pinned release gate against the live ledger, and writes a private two-day pick with distinct reserves. For manual evidence, pass a downloaded pool directory with `--artifact`. Both paths produce `NEVER_POST` picks, which the next stage must validate against its own receiving rules.
- Build a private shadow day with `python -B -m scripts.build_nightly_shadow_day`, using the resulting `--pick`, current `--ledger`, pinned `--caption-pool`, `--gate` and `--policy`, and a `--reference` slot manifest and its `--reference-sha`. Give it a new private `--output` directory. It maps the picks to the reference's four UTC slots, copies the primaries and reserves, and verifies synthetic names, the approved red notice and layout, omitted justification, exact viewer-seat proof at 90% or higher, and loudness evidence bound to each exact video. It checks captions against the last eight posts and the approved pool and retired-phrase rules, screens every clip against the canonical ledger, and records Policy C for each reachable primary, reserve and no-post branch in `SLOT-COMBINATION-CHECKS.json`. `shadow-day.json` and `SHA256SUMS.txt` record the checks. Unresolved canonical history holds the sealed day and returns a nonzero exit status. Every output is marked `NEVER_POST`.
- The clip gate keeps its audit rows in a private evidence store. Each row records the gate and ledger hashes.
- Require independent media QA, [engine-interest and viewer-seat proof](docs/RESEARCH_SCENARIO_V4.md#engine-interest-v1), the [approved clip presentation](DESIGN.md#rendered-clips), caption policy and a current per-platform publication history. Hold a failing or unverified clip and record the reason.

### Manual render kickoff

At each batch kickoff, invoke the [shared preflight](scripts/render_kickoff.py) from a trusted public-main checkout with the project dependencies installed. Give it the clean, full-history checkout for the selected inputs, the selected build receipt, and the image ID observed by the launcher. Capture stdout as the batch's kickoff receipt and proceed only with a `passed` verdict.

```sh
python -I -B scripts/render_kickoff.py --checkout "$SOURCE_CHECKOUT" \
  --source-head "$SOURCE_HEAD" --source-tree "$SOURCE_TREE" \
  --build-receipt "$BUILD_RECEIPT" --image-id "$IMAGE_ID" > "$KICKOFF_RECEIPT"
```

The command resolves public `main` outside any Git checkout and fetches the observed commit into the selected checkout's Git object store if needed. It checks its own script, policy and schema against that commit. The selected checkout must be clean and its pinned tree must match public `main` across the repository, subject to the [reviewed exclusions](scripts/render-kickoff.json). If the content differs, the verdict identifies a missing merged commit when available. The verdict binds the input commit and tree to the selected worker image.

Stdout always carries one JSON verdict with `schemaVersion` (`1`), `status`, `sourceHead`, `sourceTree`, `imageId`, `packageLockSha256`, `mainHead`, `checkedAt` (UTC), `missingRendererCommit`, and `reason`. Unavailable evidence is `null`. Exit `0` carries `passed` or `help`, exit `1` carries a refusal, and exit `2` carries a usage error. Keep the receipt read-only for the batch. Run `render_research_notice --public` from the pinned Git checkout with `BALOOT_RENDER_KICKOFF_RECEIPT`, `BALOOT_RENDER_BUILD_RECEIPT`, and `BALOOT_RENDER_IMAGE_ID`. The producer checks its own commit and tree and rejects future-dated receipts or receipts older than the configured limit. Run kickoff again before starting a producer with an expired receipt.

Bind-mount a pinned checkout owned by UID `10001` into the video worker and run the producer as that UID. The hosted `remotion-image-parity` job exercises the Git check with a read-only mount and the image's default user.

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
