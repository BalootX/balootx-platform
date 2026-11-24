from __future__ import annotations

from datetime import datetime, timezone
from functools import cache
from importlib.resources import files
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzdata import IANA_VERSION


class ConfigurationIntegrityError(Exception):
    """A saved configuration schedule cannot be verified."""


def _parse_datetime(value: str) -> datetime:
    if "T" not in value:
        raise ValueError
    return datetime.fromisoformat(value)


def parse_instant(value: str | None, field: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = _parse_datetime(value)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid ISO date-time with an explicit UTC offset") from exc
    raise ValueError(f"{field} requires an explicit UTC offset")


@cache
def _iana_zones() -> frozenset[str]:
    return frozenset(files("tzdata").joinpath("zones").read_text(encoding="utf-8").splitlines())


@cache
def _zone(name: str) -> ZoneInfo:
    if name not in _iana_zones():
        raise ZoneInfoNotFoundError(name)
    with files("tzdata.zoneinfo").joinpath(*name.split("/")).open("rb") as resource:
        return ZoneInfo.from_file(resource, key=name)


def _local_instant(value: str, zone: ZoneInfo, field: str) -> tuple[datetime, str]:
    try:
        wall_time = _parse_datetime(value)
        if wall_time.tzinfo is not None:
            raise ValueError
        # Both folds round-trip during an overlap, and neither during a gap.
        instants = set()
        for fold in (0, 1):
            instant = wall_time.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
            if instant.astimezone(zone).replace(tzinfo=None) == wall_time:
                instants.add(instant)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid local ISO date-time without an offset") from exc
    if not instants:
        raise ValueError(f"{field} does not exist in this time zone. Choose a different local time")
    if len(instants) > 1:
        raise ValueError(f"{field} is ambiguous in this time zone. Choose a different local time or use an explicit offset")
    return instants.pop(), wall_time.isoformat()


def normalize_schedule(
    effective_from: str | None,
    effective_until: str | None,
    local_schedule: Mapping[str, str] | None,
) -> tuple[str | None, str | None, dict[str, str] | None]:
    fields = ("effectiveFrom", "effectiveUntil")
    normalized = None
    if local_schedule is not None:
        if not isinstance(local_schedule, Mapping) or set(local_schedule) - {"timeZone", *fields}:
            raise ValueError("localSchedule accepts timeZone, effectiveFrom and effectiveUntil")
        zone_name = local_schedule.get("timeZone")
        try:
            if not isinstance(zone_name, str):
                raise ZoneInfoNotFoundError
            zone = _zone(zone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("localSchedule.timeZone requires a valid IANA time zone") from exc
        normalized = {"timeZone": zone_name, "zoneDataVersion": IANA_VERSION}
        if all(local_schedule.get(field) is None for field in fields):
            raise ValueError("localSchedule requires effectiveFrom or effectiveUntil")

    instants = []
    for field, value in zip(fields, (effective_from, effective_until), strict=True):
        local_value = local_schedule.get(field) if local_schedule is not None else None
        if local_value is None:
            instant = parse_instant(value, field)
        else:
            if value is not None:
                raise ValueError(f"{field} and localSchedule.{field} cannot both be specified")
            assert normalized is not None
            instant, normalized[field] = _local_instant(local_value, zone, f"localSchedule.{field}")
        instants.append(instant)
    start, end = instants
    if start is not None and end is not None and end <= start:
        raise ValueError("effectiveUntil must be later than effectiveFrom")
    return (
        start.isoformat() if start is not None else None,
        end.isoformat() if end is not None else None,
        normalized,
    )


def verify_schedule(
    revision_id: str,
    effective_from: str | None,
    effective_until: str | None,
    local_schedule: Mapping[str, str] | None,
) -> None:
    try:
        bounds = {
            "effectiveFrom": parse_instant(effective_from, "effectiveFrom"),
            "effectiveUntil": parse_instant(effective_until, "effectiveUntil"),
        }
        if local_schedule is None:
            return
        recorded_version = local_schedule["zoneDataVersion"]
        if not recorded_version or not local_schedule.keys() & {"effectiveFrom", "effectiveUntil"}:
            raise ValueError("localSchedule is missing its zone-data version or effective bounds")
        zone = _zone(local_schedule["timeZone"])
        for field, saved in bounds.items():
            if field in local_schedule:
                instant, _ = _local_instant(local_schedule[field], zone, f"localSchedule.{field}")
                if instant != saved:
                    raise ValueError(
                        f"localSchedule.{field} maps to a different UTC instant than under IANA {recorded_version}"
                    )
    except (KeyError, TypeError, ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigurationIntegrityError(
            f"{revision_id}: the saved schedule cannot be verified with IANA {IANA_VERSION}. "
            f"Create a replacement revision with a verified schedule. {exc}"
        ) from exc
