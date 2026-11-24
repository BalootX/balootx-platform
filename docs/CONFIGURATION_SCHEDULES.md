# Configuration revision schedules

Profile and bundle creation accept optional effective bounds in both configuration
API versions. For an absolute instant, use `effectiveFrom` or `effectiveUntil`
with `Z` or an explicit UTC offset. When resolving as of a specific instant,
`atTime` also requires an explicit offset. Use the ISO `T` separator for
both instants and wall times.

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
content and schedules reuse the existing revision. The same content with a new
schedule creates a new revision without changing the content hash.

The server reads zone rules from the pinned IANA data package and records the
rule version in `localSchedule.zoneDataVersion`. On resolution it checks that
the current rules still yield the saved UTC instants. If the mapping changes,
the time becomes ambiguous or nonexistent, or the zone is unavailable, the
affected resolution returns HTTP 503. Create a replacement revision with a
verified schedule. If the runtime bundle fails verification at startup, the
API serves liveness and metrics and reports the schedule error on readiness.

Apply the PostgreSQL migration with the current migration runner before starting
new service code. The runner records checksums so the historical content-only
index is applied once. SQLite adds the nullable column on open and converts
historical empty bounds to NULL. Historical bounds retain their effective
instants and carry no local schedule.

Prior PostgreSQL readers can still read the persisted UTC bounds. Once duplicate
content exists under distinct schedules, the previous migration runner and
SQLite initializer try to restore content-only uniqueness and cannot open that
database. Rehearse application rollback before using schedule variants.
