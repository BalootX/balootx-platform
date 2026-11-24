"""Plan four TikTok slots from trusted released inventory and prepare the r7 handoff."""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import ModuleType

from .calibration.score import engine_interest_rank_key_v1

GATE_SHA = "5af4fa770edea3f1fb18e8bc1f9f319cfbf7d1725a226ad34989449c6e300e22"
POLICY_SHA = "d060b4615d9d45cb44670ba94e03bf30292ca0549cea2d76c332c9a44cb108bb"
SLOT_COUNT = 4


class PlanError(Exception):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def require(condition, reason):
    if not condition:
        raise PlanError(reason)


def pinned_bytes(path, expected, reason):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, reason)
    return raw


def load_gate(script, policy):
    source = pinned_bytes(script, GATE_SHA, "GATE_SCRIPT_DIGEST_MISMATCH")
    pinned_bytes(policy, POLICY_SHA, "GATE_POLICY_DIGEST_MISMATCH")
    # Use the checked bytes and avoid creating Python cache files in the gate installation.
    gate = ModuleType("approved_release_gate")
    gate.__file__ = str(script.resolve())
    exec(compile(source, "<approved release gate>", "exec"), gate.__dict__)
    require(gate.POLICY.resolve() == policy.resolve(), "GATE_POLICY_PATH_MISMATCH")
    return gate


@dataclass(frozen=True)
class Route:
    packet: Path
    slot: datetime
    binding: dict
    clip: dict
    released_day: str
    base: Fraction
    decision: Fraction
    surprise: int
    difficulty: float
    caption_allowed: bool

    @property
    def mechanism(self):
        return self.binding["opener"]["mechanism"]


def released_routes(gate, inputs, buffer, evidence, pool, private, account):
    require(
        buffer["schemaVersion"] == 1 and type(buffer["routes"]) is list,
        "INVALID_BUFFER",
    )
    clips = inputs.pinned(evidence["files"]["releaseRequests"])["clips"]
    ranks = inputs.pinned(evidence["files"]["candidateIndex"])["ranking"]
    summaries = inputs.pinned(evidence["files"]["a2Index"])["receipts"]
    routes, seen, days = [], set(), {}
    for item in buffer["routes"]:
        if item["status"] != "RELEASED":
            continue
        day = date.fromisoformat(item["releasedDay"]).isoformat()
        packet = Path(item["packet"]).resolve(strict=True)
        b = inputs.pinned(
            evidence["packets"][packet.name], path=packet / "binding.json"
        )
        slot = gate.utc(b["slotUtc"])
        require(
            b["packetId"]
            == packet.name
            == slot.strftime("%H%M") + "-" + b["dealNumber"]
            and day <= slot.date().isoformat()
            and packet not in seen
            and b["account"] == account
            and b["canonicalLedgerReadOnly"] == gate.LEDGER.as_posix()
            and b["platforms"] == b["authorizedPlatforms"] == ["TikTok"]
            and b["opener"]["mechanism"] in gate.MECHANISMS,
            "INVALID_RELEASED_ROUTE",
        )
        clip = gate.unique(clips, "clipId", b["dealNumber"])
        require(
            any(
                (entry["packetId"], entry["slotUtc"]) == (packet.name, b["slotUtc"])
                for entry in clip["slots"]
            )
            and (
                clip["exactStateId"],
                clip["deal"],
                clip["dealHash"],
                clip["mechanism"],
                clip["family"],
                clip["visibleOpenerSha256"],
                clip["mp4Sha256"],
                clip["captionSha256"],
            )
            == (
                b["exactStateId"],
                b["deal"],
                b["dealHash"],
                b["opener"]["mechanism"],
                b["opener"]["family"],
                gate.opener_hash(b["opener"]),
                b["mediaSha256"],
                b["captionSha256"],
            )
            and b["opener"]["sha256"] == clip["visibleOpenerSha256"]
            and days.setdefault(b["exactStateId"], day) == day,
            "RELEASED_CLIP_BINDING_MISMATCH",
        )
        seen.add(packet)
        caption = gate.bound_asset(inputs, packet, b, "captionPath")["caption"]
        require(
            gate.sha_text(caption) == b["captionSha256"], "CAPTION_IDENTITY_MISMATCH"
        )
        caption_reasons = gate.check_caption(caption, private, pool)
        rank = gate.unique(ranks, "shortId", b["dealNumber"])
        proof = inputs.pinned(clip["R"]["receipt"])
        summary = gate.unique(summaries, "id", b["dealNumber"])
        require(
            summary["receiptSha256"] == clip["R"]["receipt"]["sha256"],
            "VIEWER_PROOF_BINDING_MISMATCH",
        )
        values = gate.check_viewer(
            proof,
            summary,
            rank["fixtureSha256"],
            clip,
            rank,
            b["familyEvidence"]["selectedAction"],
        )
        d, s, base = gate.score_components(rank, values)
        require(
            base == Fraction(clip["unroundedScore"]["baseExact"]),
            "SCORE_EVIDENCE_CONFLICT",
        )
        routes.append(
            Route(
                packet,
                slot,
                b,
                clip,
                day,
                base,
                d,
                s,
                rank["B"]["difficultyPrior"],
                not caption_reasons,
            )
        )
    return routes


def simulated_post(gate, route, post_id):
    b = route.binding
    opener = b["opener"]
    return {
        "platform": "TikTok",
        "status": gate.CONFIRMED,
        "id": post_id,
        "url": f"https://www.tiktok.com/@{b['account']}/video/{post_id}",
        "postedAtUtc": b["slotUtc"],
        "deal": b["deal"],
        "dealHash": b["dealHash"],
        "exactStateId": b["exactStateId"],
        "artifactSha256": b["mediaSha256"],
        "captionSha256": b["captionSha256"],
        "visibleOpener": opener,
        "visibleOpenerSha256": opener["sha256"],
        "visibleFingerprintSha256": opener["sha256"],
        "scenarioFamily": opener["family"],
        "mechanism": route.mechanism,
        "contract": b["contract"],
        "trickBand": b["trickBand"],
    }


def eligible_rank(gate, inputs, ledger, route, slot):
    if not route.caption_allowed:
        return None
    b = route.binding
    known, captions, recent, forbidden, chronology, unresolved = gate.history(
        ledger, b, slot, inputs
    )
    require(not unresolved, "LEDGER_HISTORY_UNRESOLVED")
    previous = recent[-1]
    cases = [
        case
        for case in b["familyCases"]
        if case["deal"] == previous["deal"]
        and case["visibleFingerprintSha256"] == previous["visibleOpenerSha256"]
    ]
    if (
        len(cases) != 1
        or cases[0]["mechanism"] != previous["mechanism"]
        or cases[0]["family"] != previous["scenarioFamily"]
        or gate.opener_hash(cases[0]["visibleOpener"])
        != previous["visibleOpenerSha256"]
    ):
        return None
    inputs.pinned(cases[0]["fixture"], decode=False)
    reasons, variety = gate.check_history(
        known, captions, recent, forbidden, b, chronology
    )
    if reasons or gate.check_score(route.base, variety, route.decision, route.surprise):
        return None
    # Use the exact total for the first ranking key, including for very close rational scores.
    total = route.base + variety
    ranking = engine_interest_rank_key_v1(
        {
            "parts": {"S": route.surprise, "D": route.decision},
            "difficultyPrior": route.difficulty,
            "mechanismFrequency": sum(
                row["mechanism"] == route.mechanism
                for _, _, row in sorted(chronology, key=lambda item: item[1])[-10:]
            ),
            "exactStateId": b["exactStateId"],
        }
    )
    return -total, *ranking[1:], route.packet.name


def choose_slots(routes, slots, rank):
    if len({route.mechanism for route in routes}) < 2:
        raise PlanError("INSUFFICIENT_DISTINCT_MECHANISMS")
    by_slot = defaultdict(list)
    for route in routes:
        by_slot[route.slot].append(route)

    def eligible(slot, prior):
        ranked = [(rank(route, prior), route) for route in by_slot[slot]]
        return [
            route
            for key, route in sorted(
                (item for item in ranked if item[0] is not None),
                key=lambda item: item[0],
            )
        ]

    furthest = 0

    def search(plan, primaries):
        nonlocal furthest
        i = len(plan)
        if i == len(slots):
            return plan
        furthest = max(furthest, i)
        slot = slots[i]
        primary = eligible(slot, primaries)
        # A later fallback is considered after the preceding window ended with zero submits.
        fallback = eligible(slot, primaries[:-1]) if primaries else primary
        for first in primary:
            other = next(
                (route for route in fallback if route.mechanism != first.mechanism),
                None,
            )
            if other is not None:
                result = search([*plan, (first, other)], (*primaries, first))
                if result is not None:
                    return result
        return None

    result = search([], ())
    if result is None:
        raise PlanError(
            "SLOT_NO_DIFFERENT_MECHANISM_FALLBACK",
            slotUtc=slots[furthest].isoformat().replace("+00:00", "Z"),
        )
    return result


def coverage_by_day(routes, plan):
    coverage = []
    for day in sorted({route.released_day for route in routes}):
        inventory = [route for route in routes if route.released_day == day]
        coverage.append(
            {
                "releasedDay": day,
                "releasedClips": len(
                    {route.binding["exactStateId"] for route in inventory}
                ),
                "distinctMechanisms": len({route.mechanism for route in inventory}),
                "candidateSlots": len({route.slot for route in inventory}),
                "primarySlots": sum(primary.released_day == day for primary, _ in plan),
                "fallbackSlots": sum(
                    fallback.released_day == day for _, fallback in plan
                ),
            }
        )
    return coverage


def gate_args(args, route):
    return {
        "packet": str(route.packet),
        "ledger": str(args.ledger),
        "evidence": str(args.evidence),
        "evidence-sha": args.evidence_sha,
        "qa": route.clip["opusMediaQa"]["receipt"]["path"],
        "proof": route.clip["R"]["receipt"]["path"],
        "slot": route.binding["slotUtc"],
        "studio-account": args.account,
    }


def build_plan(args, gate):
    require(
        args.ledger.resolve() == gate.LEDGER.resolve(), "CANONICAL_LEDGER_PATH_REQUIRED"
    )
    output = args.output.resolve()
    require(
        not output.is_relative_to(args.policy.resolve().parent)
        and not any((parent / ".git").exists() for parent in output.parents),
        "INVALID_OUTPUT_PATH",
    )
    inputs = gate.Inputs()
    ledger_bytes = args.ledger.read_bytes()
    ledger_sha = sha256(ledger_bytes).hexdigest()
    ledger = gate.parse(ledger_bytes)
    pool = gate.parse(inputs.read(args.caption_pool, gate.CAPTION_POOL_SHA))
    private = gate.parse(inputs.read(gate.PRIVATE, gate.PRIVATE_SHA))
    evidence = gate.parse(inputs.read(args.evidence, args.evidence_sha))
    require(evidence["scope"] == "STANDING_TIKTOK", "INVALID_GATE_EVIDENCE_SCOPE")
    buffer = gate.parse(inputs.read(args.buffer, args.buffer_sha))
    routes = released_routes(
        gate, inputs, buffer, evidence, pool, private, args.account
    )
    slots = [gate.utc(slot) for slot in args.slots]
    require(
        slots == sorted(set(slots)) and len({slot.date() for slot in slots}) == 1,
        "INVALID_DAILY_SLOTS",
    )
    daily = [route for route in routes if route.slot in slots]

    ids = {row.get("id") for row in ledger["records"]}
    simulated_ids = [
        f"{index:019d}"
        for index in range(1, len(ids) + SLOT_COUNT + 1)
        if f"{index:019d}" not in ids
    ][:SLOT_COUNT]

    def rank(route, prior):
        projected = {
            **ledger,
            "records": [
                *ledger["records"],
                *(
                    simulated_post(gate, posted, post_id)
                    for posted, post_id in zip(prior, simulated_ids)
                ),
            ],
        }
        return eligible_rank(gate, inputs, projected, route, route.slot)

    try:
        plan = choose_slots(daily, slots, rank)
    except PlanError as exc:
        exc.details["coverageByReleasedDay"] = coverage_by_day(routes, [])
        raise
    pinned_bytes(args.policy, POLICY_SHA, "GATE_POLICY_DIGEST_MISMATCH")
    pinned_bytes(args.ledger, ledger_sha, "CANONICAL_LEDGER_CHANGED_DURING_PLANNING")
    report = {
        "status": "PLAN_READY",
        "platform": "TikTok",
        "dayUtc": slots[0].date().isoformat(),
        "gateScriptSha256": GATE_SHA,
        "gatePolicySha256": POLICY_SHA,
        "evidenceScope": "STANDING_TIKTOK",
        "plannedSlots": SLOT_COUNT,
        "coverageByReleasedDay": coverage_by_day(routes, plan),
    }
    handoff = {
        **report,
        "schemaVersion": 1,
        "canonicalLedgerSha256": ledger_sha,
        "bufferSha256": args.buffer_sha,
        "gateScript": str(args.gate.resolve()),
        "slots": [
            {
                "slotUtc": primary.binding["slotUtc"],
                "primary": gate_args(args, primary),
                "fallback": gate_args(args, fallback),
            }
            for primary, fallback in plan
        ],
    }
    return report, handoff


def write_handoff(path, handoff):
    payload = json.dumps(handoff, sort_keys=True, allow_nan=False).encode() + b"\n"
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        scratch = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.close()
            os.link(scratch, path)
        finally:
            temporary.close()
            scratch.unlink()


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise PlanError("INVALID_ARGUMENTS")


def main(argv=None):
    parser = Parser(description=__doc__)
    for name in (
        "buffer",
        "ledger",
        "caption-pool",
        "gate",
        "policy",
        "evidence",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("account", "evidence-sha", "buffer-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--slots", nargs=SLOT_COUNT, required=True)
    try:
        args = parser.parse_args(argv)
        gate = load_gate(args.gate, args.policy)
        try:
            report, handoff = build_plan(args, gate)
        except gate.Reject as exc:
            raise PlanError(str(exc)) from exc
        write_handoff(args.output, handoff)
    except PlanError as exc:
        print(
            json.dumps(
                {"status": "PLAN_REJECTED", "reason": exc.reason, **exc.details},
                sort_keys=True,
            )
        )
        return 1
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        RecursionError,
        ArithmeticError,
    ):
        print(
            json.dumps(
                {
                    "status": "PLAN_REJECTED",
                    "reason": "MISSING_OR_INVALID_PLANNING_INPUT",
                }
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
