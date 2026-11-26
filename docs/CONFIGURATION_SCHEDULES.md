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

If the runtime bundle fails verification at startup, HTTP serves liveness,
readiness, metrics and the authenticated configuration-admin recovery routes.
Readiness returns HTTP 503 for runtime and research schedule failures. The
`baloot_research_configuration_unavailable` gauge counts affected research
environments. A research-only failure leaves the verified runtime's HTTP
routes available. The gRPC service exits before binding a port and the video
worker exits before claiming a job when either startup check fails.

To recover, authenticate as a configuration administrator and inspect
`/readyz` and the configuration registry's profile and bundle lists. Use
the v1 registry or v2 configuration create routes with `publish: true` to
publish a replacement bundle with verified component revisions. To restore
known-good content, use the v1 registry's `POST /bundles/{revisionId}:rollback`
with a new version. It publishes a fresh, unscheduled revision.
Restart affected processes after publishing and verify readiness.

Apply the PostgreSQL migration with the current migration runner before starting
new service code. The runner records checksums so the historical content-only
index is applied once. SQLite adds the nullable column on open and converts
historical empty bounds to NULL. Historical bounds retain their effective
instants and carry no local schedule. SQLite reads legacy text without a UTC
offset as UTC.

The schema retains content-only uniqueness. The PostgreSQL CI job writes
schedules with the new code, runs the previous revision's migration runner,
and opens both databases with the previous stores to verify their UTC reads.
