# Hosted CI timing

Baseline from [Actions run 44068917758](https://github.com/BalootX/balootx-platform/actions/runs/44068917758), attempt 1. Times use the job API and retained execution receipts. Workflow span from the first job start to the final job completion was 36m 19s. The API run span including setup and finalization was 36m 21s.

| Metric | Baseline | Replacement |
| --- | ---: | --- |
| Workflow API span | 36m 21s | 13m 39s |
| Workflow job span | 2,179 s | 816 s |
| Job duration p50 / p90 (33 baseline, 42 replacement jobs) | 553 s / 889 s | 352 s / 609 s |
| Created-to-start p90, including matrix cap | 944 s | 2 s |
| Slot-release-to-start, four waiting fast legs | 1-3 s | NOT_RUN |
| Slowest fast producer | 1,065 s | 763 s |
| Slowest backend producer | 867 s | NOT_RUN |
| Slowest heavy producer | 889 s | NOT_RUN |
| Media producer / render step | 735 s / 575 s | NOT_RUN |
| Fast single-file shard / other fast shards | 1.02 / 2.92-3.24 cores | NOT_RUN |
| Heavy shard average CPU occupancy | 1.02 of 4 cores | NOT_RUN |
| Sampled package cache lookups | 0 hits / 13 | NOT_RUN |

The previous concurrency cap held late Python shards in the queue. Full lanes now qualify in parallel, with a cap of 10 per Python matrix. The file worker pool retains four cores and the node runners use four isolated subprocesses. Recorded file and node phase timings balance the work. The unified executor and four other slow files with complete passing phase receipts join the existing two-version node matrix. The golden resource test and the timing-sensitive serial lanes keep their existing file-level checks. Native release and Python wheels are built once per applicable toolchain. Media outputs fan out from the same generated bundles and retain parity, decode and release-evidence checks.

The replacement figures come from [run 44170428047](https://github.com/BalootX/balootx-platform/actions/runs/44170428047) at `951ecbc4`. That run failed qualification. All 22 Python producers started within two seconds, and the slowest job was Python 3.12 fast shard 1.

Use the next exact-head run to recalculate this table. Compare job durations, queue time and per-job CPU use before adding shards. The repository cache budget is 10 GB. Each key has one producer that saves on main pushes, while PR and shard jobs restore.

Report scoped PR timings by change type. A docs-only run exercises the smoke path, while SDK and Foundry runs exercise their named producers and the Python 3.12 and 3.13 affected-test jobs. Record the full fallback separately because workflow changes select it.

Each PR has one CI concurrency group. A new run for that PR supersedes the previous run. Main pushes, workflow dispatches and merge-group runs each use a distinct run-ID group with cancellation disabled. Track cancelled runs and their runner minutes separately from completed PR gate timings.

CI builds the committed Dockerfile's full `video-worker` target on main pushes and on PRs that change worker inputs, including Foundry UI changes copied with `src`. It verifies that the built image's recorded SHA-256 and installed Remotion production dependencies match the checkout's lockfile. Main pushes populate the BuildKit cache, and PRs can restore it.

The worker build uses the digest in [the BuildKit image declaration](Dockerfile.buildkit). The Python base image digest also selects the Debian archive snapshot recorded by that image. Dockerfile Python installs use [the exact constraints](../requirements-docker.txt), and worker parity checks the installed Python versions alongside Remotion. The existing Docker and pip Dependabot updates cover these inputs. When a Python dependency changes its requirements, refresh the affected constraints together so the hosted build can verify the result.

For the main merge queue, GitHub requests checks on the temporary merge-group ref. CI verifies the checkout against the event's head SHA and runs the same full producer and Python shard profile as a main push. The gate reports on that SHA. PRs keep scoped feedback, and main pushes keep the full post-merge run. Merge groups can restore caches. Only main pushes save them.

The render host preflight checks deployed images for staleness. Compare the worker's `/opt/baloot-remotion/.lock-sha256` with the SHA-256 of the Remotion lockfile for the source being rendered. Run `python -P scripts/remotion_image_parity.py /opt/baloot-remotion/node_modules <lockfile>` with that same lockfile.

CODEOWNERS protects the CI control plane, including the gate, plan, coverage guard, launchers, and their startup and import paths. Media and release verifiers are product checks reviewed with the product. Existing ownership of verifier scripts governs review of changes to those files.
