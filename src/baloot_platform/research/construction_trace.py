"""Canonical trace-prefix serialization shared by native construction paths."""

from baloot_platform.core.enums import Seat
from .scenario_contracts import ExactTracePrefixV1


def construction_trace_prefix_v1(trace) -> ExactTracePrefixV1:
    """Normalize one complete CP-SAT construction for an all-decision audit."""

    project_actions = [
        dict(row) for row in trace["projects"].get("declarationActions", ())
    ]
    declarations = sorted(
        (row for row in trace["projects"]["projects"] if row["declared"]),
        key=lambda row: int(row["declarationOrder"]),
    )
    if not project_actions:
        declared_by_owner = {
            row["owner"]: row for row in declarations
        }
        first_declarer = Seat.parse(trace["dealer"]).next()
        declaration_seats = (
            first_declarer,
            first_declarer.next(),
            first_declarer.next().next(),
            Seat.parse(trace["dealer"]),
        )
        project_actions = [
            (
                {
                    "actor": seat.name.lower(),
                    "action": "declare",
                    "projectId": declared_by_owner[seat.name.lower()]["projectId"],
                }
                if seat.name.lower() in declared_by_owner
                else {"actor": seat.name.lower(), "action": "pass"}
            )
            for seat in declaration_seats
        ]
    return ExactTracePrefixV1.build({
        "schemaVersion": 1,
        "deckOrder": list(trace["deckOrder"]),
        "dealer": trace["dealer"],
        "roundNumber": int(trace.get("roundNumber", 1)),
        "matchScore": list(trace.get("matchScore", (0, 0))),
        "auctionActions": [
            {key: row[key] for key in ("actor", "action", "trumpSuit")
             if row.get(key) is not None}
            for row in trace["auction"]["completion"]["actions"]
        ],
        "escalationActions": [
            {key: row[key] for key in ("actor", "action", "requestedMode")
             if row.get(key) is not None}
            for row in trace["escalation"]["observation"]["actions"]
        ],
        **({"projectActions": project_actions} if project_actions else {}),
        "projectDeclarations": [] if project_actions else [
            {key: row[key] for key in (
                "owner", "kind", "value", "highRank", "declarationOrder"
            )}
            for row in declarations
        ],
        "projectAssertions": [],
        "playActions": [
            {"actor": row["actor"], "card": row["card"],
             "ekka": bool(row.get("ekka", False))}
            for row in trace["play"]["plays"]
        ],
        "stateAssertions": {},
    })
