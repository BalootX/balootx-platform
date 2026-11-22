# TikTok slot sequencer

Use the sequencer after preparation has released clips and built slot-specific packets. It plans four UTC slots for one day and writes a private handoff for the posting controller. It uses the reviewed r7 gate's exact candidate score, Policy C repeat checks and caption rules. The controller still makes the slot-time decision against the actual ledger and Studio predecessor.

Run from an environment with the package installed.

```text
python -B -m baloot_platform.research.slot_sequencer
  --slots <slot-1-utc> <slot-2-utc> <slot-3-utc> <slot-4-utc>
  --buffer <released-buffer-json> --buffer-sha <trusted-buffer-sha256>
  --evidence <trusted-gate-catalog> --evidence-sha <trusted-catalog-sha256>
  --ledger <canonical-ledger> --caption-pool <approved-pool-json> --account <verified-account>
  --gate <approved-gate-script> --policy <approved-gate-policy> --output <new-private-plan-file>
```

Pass the arguments as one command. Supply the four exact, increasing slot timestamps from the sealed requests. The ledger, approved pool, gate installation, account and output location are runtime inputs. Keep these values and the handoff in the private evidence store. Use the buffer and catalog digests from the trusted preparation handoff.

## Released buffer

The buffer has a `schemaVersion` of `1` and a `routes` list. Each released route supplies `status: "RELEASED"`, a `releasedDay` in `YYYY-MM-DD` format, and `packet`, the path to the sealed packet directory. List the same clip at multiple slots only when the catalog has a separately bound packet for each slot. The clip's released day stays the same across its routes. Entries with another status never enter the plan.

The sequencer reads the catalog-pinned bindings, release requests, candidate index and viewer proof. Captions are already sealed into each route. The sequencer checks the exact text against the approved pool and screens each route against r7's cross-platform caption history, including removed posts. The canonical ledger is read-only. The planner models a possible post in memory to test later routes. It scores a primary assuming the earlier primaries posted. It scores a later fallback assuming the immediately preceding window ended with zero submits. The first-slot fallback is scored against the same base history as the first primary. The controller retains its first-slot authorization and prior-outcome rules.

## Results

Stdout contains a single JSON report with counts grouped by released day, the number of primary and fallback slots assigned from each day, and both approved gate digests. The private output file contains each slot's `primary` and `fallback` as named arguments for the documented standing gate command. The controller supplies the fresh `studio-post-id` and `studio-observed-at` at admission and checks the audit against the exact packet, ledger and approved digests.

`PLAN_REJECTED` exits nonzero without producing a handoff. `INSUFFICIENT_DISTINCT_MECHANISMS` means the day's released options offer fewer than two mechanisms. `SLOT_NO_DIFFERENT_MECHANISM_FALLBACK` identifies the slot that prevented a complete four-slot plan. Changes to either approved gate digest also close the plan.

## Read-only archive check

The reviewed sequential replay is scoped to the sealed 23 November archive. For that replay, use its archive evidence catalog and add `--archive-day <sealed-day-directory> --day-seal-sha <trusted-seal-sha256>`. The sequencer runs the documented replay with no publication side effects and accepts its nonzero process exit only with a bound, completed report and `WOULD_RELEASE` for each selected route on its corresponding archive trajectory. `ARCHIVE_CHECKED` records archive evidence. Future days use the `STANDING_TIKTOK` catalog and the slot-time controller handoff.
