import json
import os
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
        datetime.fromisoformat(slot),
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

    class Reject(ValueError):
        pass

    def read(path, expected):
        if path.name == "evidence" and expected != "evidence-digest":
            raise Reject("INPUT_SHA_MISMATCH")
        return json.dumps(payloads[path.name]).encode()

    gate = SimpleNamespace(
        LEDGER=ledger,
        Inputs=lambda: SimpleNamespace(read=read),
        Reject=Reject,
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

    plan = sequencer.choose_slots(
        inventory, list(map(datetime.fromisoformat, SLOTS)), rank
    )
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


def test_equivalent_utc_slots_keep_the_exact_bound_gate_argument(cli, capsys):
    inventory, output, argv = cli
    argv[argv.index("--slots") + 1] = SLOTS[0].replace("Z", "+00:00")
    bound_slot = SLOTS[1].replace("Z", "+00:00")
    inventory[2].binding["slotUtc"] = bound_slot
    assert sequencer.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["plannedSlots"] == 4
    assert report["coverageByReleasedDay"][0]["candidateSlots"] == 4
    plan = json.loads(output.read_bytes())
    assert plan["slots"][1]["primary"]["slot"] == SLOTS[1]
    assert plan["slots"][1]["fallback"]["slot"] == bound_slot


def test_cli_preserves_gate_rejection_reason(cli, capsys):
    _, output, argv = cli
    argv[argv.index("--evidence-sha") + 1] = "changed-digest"
    assert sequencer.main(argv) == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "PLAN_REJECTED",
        "reason": "INPUT_SHA_MISMATCH",
    }
    assert not output.exists()


@pytest.fixture(scope="module")
def r7():
    path = os.environ.get("BALOOT_R7_GATE_TEST")
    if not path:
        pytest.skip(
            "approved external gate installation is required for the r7 compatibility tests"
        )
    script = Path(path)
    return sequencer.load_gate(script, script.parents[1] / "RELEASE-GATE-POLICY.md")


@pytest.fixture
def r7_history(r7):
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
        tiktok = i < 3
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
                "captionSha256": r7.sha_text(f"Synthetic caption {i}."),
                "visibleOpener": opener,
                "visibleOpenerSha256": opener["sha256"],
                "visibleFingerprintSha256": opener["sha256"],
                "mechanism": opener["mechanism"],
                "scenarioFamily": opener["family"],
                "contract": "sun",
                "trickBand": "1-8",
            }
        )
    return rows


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
def test_r7_caption_window(r7, r7_history, tmp_path, index, latest_tiktok, eligible):
    rows = r7_history
    caption_sha = rows[index]["captionSha256"]
    inputs = r7.Inputs()
    fixture = tmp_path / "fixture.json"
    fixture.write_bytes(b"{}")
    rows[2].update(platform="Instagram", status="REMOVED_ON_OWNER_WORD")
    rows[latest_tiktok].update(platform="TikTok", status=r7.CONFIRMED)
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


def test_r7_builds_four_slots_with_valid_primary_and_fallback(
    r7, r7_history, monkeypatch, tmp_path
):
    def write(name, value):
        path = tmp_path / name
        data = json.dumps(value).encode()
        path.write_bytes(data)
        return path, sha256(data).hexdigest()

    ledger, _ = write("ledger.json", {"schemaVersion": 1, "records": r7_history})
    fixture, fixture_sha = write("fixture.json", {})
    captions = [f"Synthetic example {i}." for i in range(8)]
    pool_data = {
        **dict.fromkeys(
            "status recordedAtUtcDateU ownerVerbatim source derivedFromApprovals rules".split()
        ),
        "transform": '"retired phrase" to "replacement phrase"',
        "captions": [
            {"text": caption, "sha256": r7.sha_text(caption)} for caption in captions
        ],
    }
    private_data = {
        **dict.fromkeys(
            "LICENSE_only_allow_input_ids common_generic_synthetic_pool_not_denied old_ID_hash_prefix_policy".split()
        ),
        "entries": [
            dict(input_id="synthetic", origins=[], value="excluded phrase", variants=[])
        ],
    }
    pool, pool_sha = write("pool.json", pool_data)
    private, private_sha = write("private.json", private_data)
    evidence, evidence_sha = write("evidence.json", {"scope": "STANDING_TIKTOK"})
    buffer, buffer_sha = write("buffer.json", {})
    monkeypatch.setattr(r7, "LEDGER", ledger)
    monkeypatch.setattr(r7, "CAPTION_POOL_SHA", pool_sha)
    monkeypatch.setattr(r7, "PRIVATE", private)
    monkeypatch.setattr(r7, "PRIVATE_SHA", private_sha)

    lead, capture = MECHANISMS
    underplay = "FOLLOW_SUIT_UNDERPLAY"
    pairs = [
        (lead, underplay),
        (underplay, lead),
        (capture, underplay),
        (lead, capture),
    ]
    inventory = []
    cards = ("H7", "H8", "H9", "H10", "HJ", "HQ", "HK", "HA")
    known_ids = [row["id"] for row in r7_history]
    for i, (slot, pair) in enumerate(zip(SLOTS, pairs, strict=True)):
        for j, mechanism in enumerate(pair):
            label = f"{i}-{j}"
            opener = {
                "hand": [cards[2 * i + j]],
                "table": [],
                "mechanism": mechanism,
                "family": f"new-family-{label}",
            }
            opener["sha256"] = r7.opener_hash(opener)
            inventory.append(
                route(
                    slot,
                    mechanism,
                    label,
                    {
                        "deal": label,
                        "dealHash": r7.sha_text(f"deal {label}"),
                        "mediaSha256": r7.sha_text(f"media {label}"),
                        "captionSha256": r7.sha_text(captions[2 * i + j]),
                        "opener": opener,
                        "contract": "hokum",
                        "trickBand": "9-16",
                        "knownMinimumIds": known_ids,
                        "familyCases": [],
                    },
                )
            )

    previous = r7_history[2]
    predecessors = [(previous["deal"], previous["visibleOpener"])]
    for primary, fallback in zip(inventory[::2], inventory[1::2], strict=True):
        for option in (primary, fallback):
            option.binding["familyCases"].extend(
                {
                    "deal": deal,
                    "visibleFingerprintSha256": opener["sha256"],
                    "family": opener["family"],
                    "mechanism": opener["mechanism"],
                    "visibleOpener": opener,
                    "fixture": {"path": str(fixture), "sha256": fixture_sha},
                }
                for deal, opener in predecessors
            )
        predecessors.append((primary.binding["deal"], primary.binding["opener"]))

    # Isolate released-media intake. Planning, history and scoring use the verified r7 module.
    monkeypatch.setattr(sequencer, "released_routes", lambda *args: inventory)
    args = SimpleNamespace(
        ledger=ledger,
        buffer=buffer,
        buffer_sha=buffer_sha,
        evidence=evidence,
        evidence_sha=evidence_sha,
        caption_pool=pool,
        gate=Path(r7.__file__),
        policy=r7.POLICY,
        output=tmp_path / "plan.json",
        slots=SLOTS,
        account="synthetic-user",
    )
    report, plan = sequencer.build_plan(args, r7)
    assert report["status"] == "PLAN_READY"
    assert report["plannedSlots"] == len(plan["slots"]) == 4
    assert (
        report["gatePolicySha256"] == plan["gatePolicySha256"] == sequencer.POLICY_SHA
    )
    for i, slot in enumerate(plan["slots"]):
        for j, mode in enumerate(("primary", "fallback")):
            candidate = inventory[2 * i + j]
            assert slot[mode] == sequencer.gate_args(args, candidate)
            assert r7.check_caption(captions[2 * i + j], private_data, pool_data) == []
