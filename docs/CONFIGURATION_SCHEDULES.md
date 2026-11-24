# Configuration revision schedules

Profile and bundle creation accept optional effective bounds in both configuration
API versions. For an absolute instant, use `effectiveFrom` or `effectiveUntil`
with `Z` or an explicit UTC offset. When resolving as of a specific instant,
`atTime` also requires an explicit offset.

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
returning UTC instants in `effectiveFrom` and `effectiveUntil`. When a creation
request matches existing content, its schedule must match too. Use different
revision content to change the schedule.

The server reads zone rules from the pinned IANA data package and records the
rule version in `localSchedule.zoneDataVersion`. On resolution it checks that
the current rules still yield the saved UTC instants. If the mapping changes,
the time becomes ambiguous or nonexistent, or the zone is unavailable, create
a replacement revision with a verified schedule.

Apply the additive PostgreSQL migration before starting new service code. SQLite
adds the nullable column on open and converts historical empty bounds to NULL.
Historical bounds retain their effective instants and carry no local schedule.
Both schemas remain readable by the prior service version.
