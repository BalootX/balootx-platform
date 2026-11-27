# Configuration revision schedules

Profile and bundle creation accept optional effective bounds in both REST API
versions. Use an ISO date-time with `Z` or an explicit UTC offset for
`effectiveFrom`, `effectiveUntil` and `atTime`. The v1 registry retains its
legacy UTC interpretation of offsetless input and stores new bounds in UTC.
The v2 creation routes require an explicit offset for absolute bounds. Use
the ISO `T` separator for local wall times.

For a local schedule, supply an IANA zone and at least one wall-time bound in
`localSchedule`. The configuration studio sends this shape directly.
Omit `localSchedule` for instant-only or unscheduled revisions.

```json
{
  "localSchedule": {
    "timeZone": "America/New_York",
    "effectiveFrom": "2026-07-01T12:00"
  }
}
```

Supply each bound once. The start is inclusive and the end is exclusive.
Nonexistent and ambiguous local times are rejected with the field name. For a
repeated hour, choose an unambiguous wall time or specify the intended instant
with its explicit offset.

A bounded bundle publication keeps the existing publication for the previous
release and records the scheduled revision in the same scope. In a scope used
by packaged runtime bootstrap, publish a verified replacement with no time
bounds and 100% rollout before publishing or promoting a bounded revision.
If that underlying publication is missing or unusable, the store refuses the
bounded publication before changing the bundle or its audit history. The current
release checks the scheduled revision and its predecessors in publication
order and uses the first one whose window includes the requested instant.
The selected revision remains subject to its rollout policy. An immediate,
unscheduled publication ends the schedule. The previous release continues to
use the underlying publication before, during and after a scheduled window.
The current service reads the anchor, linked history, selected components,
audit verification, and invalidation epoch in one database snapshot. Concurrent
retirement can select the schedule from before or after the commit, with all
these values from the same state.
Packaged Research scopes retain their packaged publication for the previous
service during immediate replacements as well. The current service can use a
replacement in that same scope. Both Research scopes are registered before the
global startup check, so authenticated recovery cannot publish a Research
replacement that would strand the previous service. Restore an unbounded,
full-rollout revision with the packaged content before replacing it in a scope
that lacks a usable packaged publication.
Processes load a verified runtime bundle at startup. Restart them at a
scheduled boundary to load the newly effective bundle. Startup selects within
its configured scope and has no rollout subject, so a partial rollout cannot
become effective for startup. Historical replay can still read the exact
revision by ID.

New revisions retain the wall times and zone in `localSchedule` and continue
returning UTC instants in `effectiveFrom` and `effectiveUntil`. Identical
content and schedules reuse the existing revision. A different schedule for
existing content returns HTTP 409 and identifies the conflicting revision.
The gRPC create messages have no schedule fields and keep their original
content-only reuse behavior.

The server reads zone rules from the pinned IANA data package and records the
rule version in `localSchedule.zoneDataVersion`. On resolution it checks that
the current rules still yield the saved UTC instants. A failed bundle or
component revision check blocks that bundle's resolution, including when the
failing revision is outside its effective window. REST reports HTTP 503 and
gRPC reports `UNAVAILABLE`. An authenticated configuration administrator can
see the failed revision or research environment in the REST diagnosis. Other
callers receive a safe message and a request ID. Startup logs record the
revision ID and environment as structured fields.

Profile lookups without `atTime` apply the current clock at the next schedule
boundary, including when a higher-priority profile takes effect. Legacy Research
replay requires a unique verified match across retained published history. A
scheduled handoff preserves the underlying publication's saved recipe identity.
An unverifiable schedule on an unrelated bundle does not prevent that match.

If no runtime bundle is effective or its verification fails at startup, HTTP
serves liveness, readiness, metrics and the authenticated configuration-admin
recovery routes.
Readiness returns HTTP 503 when configuration is unavailable. The
`baloot_research_configuration_unavailable` gauge counts affected research
environments. A research-only failure leaves the verified runtime's HTTP
routes available. The gRPC service exits before binding a port and the video
worker exits before claiming a job when either startup check fails.

To recover, authenticate as a configuration administrator and inspect
`/readyz` and the configuration registry's profile and bundle lists. Use
the v1 registry or v2 configuration create routes with `publish: true` to
publish a currently effective replacement bundle with verified component
revisions. A refused publication leaves the active revision and audit history
intact. Replace any published bundle that uses a component before retiring
that component. This includes the underlying and reachable scheduled revisions.
A retirement request for a scheduled bundle is refused if a later publication
has already displaced it, even if the revision's ETag is unchanged.
Bundle retirement commits the schedule change and both audit events together.
If the reason cannot be recorded, the current schedule stays in effect and the
same request can be retried.
To restore known-good content, use the v1 registry's
`POST /bundles/{revisionId}:rollback`
with a new version. It publishes a fresh, unscheduled revision and records the
target and replaced publication in the same transaction.
Restart affected processes after publishing and verify readiness.

Apply PostgreSQL migrations 033 and 034 with the current migration runner before
starting new service code. They follow the two identity index migrations. The
runner needs permission to create its receipt schema and records checksums so
later runs can verify previously recorded migrations. SQLite adds the nullable
column on open and converts historical empty bounds to NULL. Historical bounds
retain their effective instants and carry no local schedule. SQLite reads legacy
text without a UTC offset as UTC and an empty bound as absent. Older timed
publications without a recorded predecessor enter the recovery flow when their
window has no effective bundle.

The schema retains content-only uniqueness. The PostgreSQL CI job writes
schedules with the new code, runs the previous revision's migration runner,
and opens both databases with the previous stores to verify their UTC reads.
It also starts both service releases before, during and after a scheduled
window on each store.
