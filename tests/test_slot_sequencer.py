import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from baloot_platform.research import slot_sequencer as sequencer

SLOTS = [f"2030-01-02T{hour:02d}:00:00Z" for hour in (9, 12, 15, 18)]
MECHANISMS = ("LEAD", "FOLLOW_SUIT_CAPTURE")


def route(slot, mechanism, label, binding=None):
    return sequencer.Route(
        Path(f"packet-{label}"),
        {
            "slotUtc": slot,
            "exactStateId": label,
            "account": "synthetic-user",
            "opener": {"mechanism": mechanism},
            **(binding or {}),
        },
        {
            "opusMediaQa": {"receipt": {"path": "qa.json"}},
            "R": {"receipt": {"path": "proof.json"}},
        },
        "2030-01-01",
        Fraction(75 if mechanism == MECHANISMS[0] else 70),
        Fraction(25),
        25,
        50,
        True,
    )


@pytest.fixture
def cli(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"records": []}', encoding="utf-8")
    policy = tmp_path / "gate" / "policy.md"
    policy.parent.mkdir()
    policy.write_text("synthetic policy", encoding="utf-8")
    gate_path = policy.with_name("gate.py")
    gate_path.write_text("synthetic gate", encoding="utf-8")
    payloads = {
        "buffer": {"routes": []},
        "pool": {},
        "private": {},
        "evidence": {"scope": "STANDING_TIKTOK"},
    }
    gate = SimpleNamespace(
        LEDGER=ledger,
        Inputs=lambda: SimpleNamespace(
            read=lambda path, _: json.dumps(payloads[path.name]).encode()
        ),
        CAPTION_POOL_SHA="pool-digest",
        PRIVATE=Path("private"),
        PRIVATE_SHA="private-digest",
        parse=json.loads,
        utc=datetime.fromisoformat,
        CONFIRMED="POSTED_ACTUAL_CAPTION_CONFIRMED",
    )
    monkeypatch.setattr(
        sequencer, "GATE_SHA", sha256(gate_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        sequencer, "POLICY_SHA", sha256(policy.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(sequencer, "load_gate", lambda *args: gate)
    inventory = [
        route(slot, mechanism, f"{i}-{j}")
        for i, slot in enumerate(SLOTS)
        for j, mechanism in enumerate(MECHANISMS)
    ]
    for candidate in inventory:
        candidate.binding.update(
            {
                "deal": "deal",
                "dealHash": "deal-digest",
                "mediaSha256": "media-digest",
                "captionSha256": "caption-digest",
                "contract": "sun",
                "trickBand": "1-8",
            }
        )
        candidate.binding["opener"].update(sha256="opener-digest", family="family")
    monkeypatch.setattr(sequencer, "released_routes", lambda *args: inventory)

    def eligible_rank(_gate, _inputs, history, candidate, _slot):
        posts = history["records"]
        if posts and posts[-1]["mechanism"] == candidate.mechanism:
            return None
        return (-candidate.base,)

    monkeypatch.setattr(sequencer, "eligible_rank", eligible_rank)
    output = tmp_path / "plan.json"
    argv = [
        "--buffer",
        "buffer",
        "--buffer-sha",
        "buffer-digest",
        "--ledger",
        str(ledger),
        "--caption-pool",
        "pool",
        "--gate",
        str(gate_path),
        "--policy",
        str(policy),
        "--account",
        "synthetic-user",
        "--evidence",
        "evidence",
        "--evidence-sha",
        "evidence-digest",
        "--output",
        str(output),
        "--slots",
        *SLOTS,
    ]
    return inventory, output, argv


@pytest.mark.parametrize(
    ("remove", "reason", "failed_slot"),
    [
        ([1, 3, 5, 7], "INSUFFICIENT_DISTINCT_MECHANISMS", None),
        ([1], "SLOT_NO_DIFFERENT_MECHANISM_FALLBACK", SLOTS[0]),
    ],
)
def test_daily_hard_fail(cli, capsys, remove, reason, failed_slot):
    inventory, output, argv = cli
    inventory[:] = [row for index, row in enumerate(inventory) if index not in remove]
    assert sequencer.main(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == reason
    assert report.get("slotUtc") == failed_slot
    assert report["coverageByReleasedDay"][0]["primarySlots"] == 0
    assert not output.exists()


def test_valid_plan_publishes_coverage_and_standing_gate_arguments(cli, capsys):
    _, output, argv = cli
    assert sequencer.main(argv) == 0
    public_report = capsys.readouterr().out
    assert "synthetic-user" not in public_report and "packet-" not in public_report
    report = json.loads(public_report)
    assert report == {
        "status": "PLAN_READY",
        "platform": "TikTok",
        "dayUtc": "2030-01-02",
        "gateScriptSha256": sequencer.GATE_SHA,
        "gatePolicySha256": sequencer.POLICY_SHA,
        "archiveDryRun": "NOT_RUN",
        "evidenceScope": "STANDING_TIKTOK",
        "plannedSlots": 4,
        "coverageByReleasedDay": [
            {
                "releasedDay": "2030-01-01",
                "releasedClips": 8,
                "distinctMechanisms": 2,
                "candidateSlots": 4,
                "primarySlots": 4,
                "fallbackSlots": 4,
            }
        ],
    }
    plan = json.loads(output.read_bytes())
    assert plan["gatePolicySha256"] == sequencer.POLICY_SHA
    assert plan["gateScriptSha256"] == sequencer.GATE_SHA
    assert [row["slotUtc"] for row in plan["slots"]] == SLOTS
    for i, row in enumerate(plan["slots"]):
        for label, j in (("primary", i % 2), ("fallback", 1 - i % 2)):
            assert row[label] == {
                "packet": f"packet-{i}-{j}",
                "slot": SLOTS[i],
                "qa": "qa.json",
                "proof": "proof.json",
                "ledger": argv[argv.index("--ledger") + 1],
                "studio-account": "synthetic-user",
                "evidence": "evidence",
                "evidence-sha": "evidence-digest",
            }
    assert sequencer.main(argv) == 1
    assert json.loads(output.read_bytes()) == plan


def test_search_chooses_highest_ranked_primary_that_can_complete_the_day():
    inventory = [
        route(slot, mechanism, f"{i}-{j}")
        for i, slot in enumerate(SLOTS)
        for j, mechanism in enumerate(MECHANISMS)
    ]

    def rank(candidate, prior):
        if (
            candidate.binding["slotUtc"] == SLOTS[1]
            and prior
            and prior[0].mechanism == MECHANISMS[0]
        ):
            return None
        return (-candidate.base,)

    plan = sequencer.choose_slots(inventory, SLOTS, rank)
    assert plan[0] == (inventory[1], inventory[0])


@pytest.mark.parametrize("which", ["SCRIPT", "POLICY"])
def test_gate_digest_mismatch_rejected_before_execution(which, monkeypatch, tmp_path):
    script, policy = tmp_path / "gate.py", tmp_path / "policy.md"
    script.write_text(
        "raise AssertionError('unapproved code executed')", encoding="utf-8"
    )
    policy.write_text("changed policy", encoding="utf-8")
    if which == "POLICY":
        monkeypatch.setattr(
            sequencer, "GATE_SHA", sha256(script.read_bytes()).hexdigest()
        )
    with pytest.raises(sequencer.PlanError, match=f"^GATE_{which}_DIGEST_MISMATCH$"):
        sequencer.load_gate(script, policy)


@pytest.mark.parametrize("changed_audit", [False, True])
def test_read_only_archive_handoff(monkeypatch, tmp_path, changed_audit):
    plan = [
        tuple(
            route(slot, mechanism, f"{i}-{j}") for j, mechanism in enumerate(MECHANISMS)
        )
        for i, slot in enumerate(SLOTS)
    ]
    args = SimpleNamespace(
        gate=tmp_path / "release_gate.py",
        archive_day=tmp_path / "archive",
        day_seal_sha="seal",
        ledger=tmp_path / "ledger",
        evidence=tmp_path / "catalog",
        evidence_sha="catalog",
    )
    runner = args.gate.with_name("dry_run_gate.py")
    runner.write_bytes(b"synthetic archive runner")
    monkeypatch.setattr(
        sequencer, "ARCHIVE_RUNNER_SHA", sha256(runner.read_bytes()).hexdigest()
    )
    replay = {
        "status": "COMPLETED_READ_ONLY",
        "dryRun": True,
        "daySealSha256": "seal",
        "canonicalLedgerSha256Before": "ledger",
        "canonicalLedgerSha256After": "ledger",
        "scenarios": {},
    }
    for j, scenario in enumerate(("primary", "fallback")):
        slots = []
        for i, pair in enumerate(plan):
            selected = pair[j]
            audit = {
                "decision": "WOULD_RELEASE",
                "dryRun": True,
                "reasons": [],
                "gateScriptSha256": sequencer.GATE_SHA,
                "policySha256": sequencer.POLICY_SHA,
                "packetId": selected.packet.name,
                "slotUtc": SLOTS[i],
                "evidenceExpectedSha256": "catalog",
                "canonicalLedgerSha256": "ledger",
                "ledgerReadbackSha256": "ledger",
            }
            slots.append(
                {
                    "slotUtc": SLOTS[i],
                    "assumedPostedPacketId": selected.packet.name,
                    "routes": [
                        {
                            "packetId": selected.packet.name,
                            "decision": "WOULD_RELEASE",
                            "dryRun": True,
                            "predecessorMatched": True,
                            "agreesWithSealedEligibility": True,
                            "reasons": [],
                            "auditRow": audit,
                        }
                    ],
                }
            )
        replay["scenarios"][scenario] = {"slots": slots}
    if changed_audit:
        audit["policySha256"] = "mismatch"

    def run(argv, **kwargs):
        assert argv[1:] == [
            "-I",
            "-B",
            str(runner),
            "--sequential-dry-run",
            str(args.archive_day),
            "--day-seal-sha",
            "seal",
            "--ledger",
            str(args.ledger),
            "--evidence",
            str(args.evidence),
            "--evidence-sha",
            "catalog",
        ]
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(argv, 1, stdout=json.dumps(replay).encode())

    monkeypatch.setattr(subprocess, "run", run)
    if changed_audit:
        with pytest.raises(sequencer.PlanError, match="^ARCHIVE_ROUTE_DID_NOT_PASS$"):
            sequencer.check_archive(args, plan, "ledger")
    else:
        sequencer.check_archive(args, plan, "ledger")


@pytest.fixture(scope="module")
def r7():
    path = os.environ.get("BALOOT_R7_GATE_TEST")
    if not path:
        pytest.skip(
            "approved external gate installation is required for the r7 compatibility tests"
        )
    script = Path(path)
    return sequencer.load_gate(script, script.parents[1] / "RELEASE-GATE-POLICY.md")


@pytest.mark.parametrize(
    ("index", "latest_tiktok", "eligible"),
    [(6, 2, False), (5, 2, True), (4, 2, True), (13, 2, False), (5, 5, False)],
    ids=[
        "seven-later",
        "eight-later",
        "nine-later",
        "cross-platform-consecutive",
        "latest-tiktok",
    ],
)
def test_r7_caption_window(r7, tmp_path, index, latest_tiktok, eligible):
    caption_sha = r7.sha_text("Synthetic caption for the planned clip.")
    inputs = r7.Inputs()
    fixture = tmp_path / "fixture.json"
    fixture.write_bytes(b"{}")
    rows = []
    for i, card in enumerate(
        (
            "C7",
            "C8",
            "C9",
            "C10",
            "CJ",
            "CQ",
            "CK",
            "CA",
            "D7",
            "D8",
            "D9",
            "D10",
            "DJ",
            "DQ",
        )
    ):
        opener = {
            "hand": [card],
            "table": [],
            "family": f"family-{i}",
            "mechanism": MECHANISMS[1],
        }
        opener["sha256"] = r7.opener_hash(opener)
        instant = datetime(2030, 1, 1, tzinfo=UTC) + timedelta(minutes=i)
        post_id = f"{100 + i:019d}"
        tiktok = i in (0, 1, latest_tiktok)
        rows.append(
            {
                "id": post_id,
                "url": f"https://www.tiktok.com/@synthetic-user/video/{post_id}",
                "platform": "TikTok" if tiktok else "Instagram",
                "status": r7.CONFIRMED if tiktok else "REMOVED_ON_OWNER_WORD",
                "postedAtUtc": instant.isoformat(),
                "deal": f"deal-{i}",
                "exactStateId": f"state-{i}",
                "dealHash": r7.sha_text(f"deal-{i}"),
                "artifactSha256": r7.sha_text(f"media-{i}"),
                "captionSha256": caption_sha
                if i == index
                else r7.sha_text(f"Synthetic caption {i}."),
                "visibleOpener": opener,
                "visibleOpenerSha256": opener["sha256"],
                "visibleFingerprintSha256": opener["sha256"],
                "mechanism": opener["mechanism"],
                "scenarioFamily": opener["family"],
                "contract": "sun",
                "trickBand": "1-8",
            }
        )
    previous = rows[latest_tiktok]
    opener = {
        "hand": ["SA"],
        "table": [],
        "family": "new-family",
        "mechanism": MECHANISMS[0],
    }
    opener["sha256"] = r7.opener_hash(opener)
    candidate = route(
        SLOTS[0],
        MECHANISMS[0],
        "new-state",
        {
            "deal": "new-deal",
            "dealHash": r7.sha_text("new-deal"),
            "mediaSha256": r7.sha_text("new-media"),
            "opener": opener,
            "captionSha256": caption_sha,
            "contract": "hokum",
            "trickBand": "9-16",
            "knownMinimumIds": [row["id"] for row in rows],
            "familyCases": [
                {
                    "deal": previous["deal"],
                    "visibleFingerprintSha256": previous["visibleOpenerSha256"],
                    "family": previous["scenarioFamily"],
                    "mechanism": previous["mechanism"],
                    "visibleOpener": previous["visibleOpener"],
                    "fixture": {"path": str(fixture), "sha256": r7.digest(b"{}")},
                }
            ],
        },
    )
    rank = sequencer.eligible_rank(
        r7, inputs, {"schemaVersion": 1, "records": rows}, candidate, r7.utc(SLOTS[0])
    )
    assert (rank is not None) is eligible
