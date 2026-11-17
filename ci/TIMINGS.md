# Hosted CI timing

Baseline from [Actions run 44068917758](https://github.com/BalootX/balootx-platform/actions/runs/44068917758), attempt 1. Times use the job API and retained execution receipts. Workflow span from the first job start to the final job completion was 36m 19s. The API run span including setup and finalization was 36m 21s.

| Metric | Baseline | Replacement |
| --- | ---: | --- |
| Workflow job span | 2,179 s | NOT_RUN |
| Job duration p50 / p90, 33 jobs | 553 s / 889 s | NOT_RUN |
| Dependency-ready-to-start p50 / p90 / max, including matrix cap | 2 s / 944 s / 1,203 s | NOT_RUN |
| Slot-release-to-start, four waiting fast legs | 1-3 s | NOT_RUN |
| Slowest fast producer | 1,065 s | NOT_RUN |
| Slowest backend producer | 867 s | NOT_RUN |
| Slowest heavy producer | 889 s | NOT_RUN |
| Media producer / render step | 735 s / 575 s | NOT_RUN |
| Fast single-file shard / other fast shards | 1.02 / 2.92-3.24 cores | NOT_RUN |
| Heavy shard average CPU occupancy | 1.02 of 4 cores | NOT_RUN |
| Sampled package cache lookups | 0 hits / 13 | NOT_RUN |

The previous concurrency cap held late Python shards in the queue. Full lanes now qualify in parallel, with a cap of 10 per Python matrix. The file worker pool retains four cores and the node runners use four isolated subprocesses. Recorded file and node phase timings balance the work. The unified executor and four other slow files with complete passing phase receipts join the existing two-version node matrix. The golden resource test and the timing-sensitive serial lanes keep their existing file-level checks. Native release and Python wheels are built once per applicable toolchain. Media outputs fan out from the same generated bundles and retain parity, decode and release-evidence checks.

Use the next exact-head run to recalculate this table. The goals are approximately five minutes for affected feature feedback after the queue cutover and ten minutes for full qualification. Compare job durations, queue time and per-job CPU use before adding shards. The repository cache budget is 10 GB. Each key has one producer that saves on main pushes, while PR and shard jobs restore. The owner-controlled cutover pairs the ci-gate context with the dev merge queue, so the bootstrap continues to report the existing CI result and full coverage.
