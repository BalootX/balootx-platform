import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import nightly_candidates as nightly


@pytest.fixture(scope="module")
def sample(rules, tmp_path_factory):
    engine = nightly.BalootEngine(rules)
    profile = nightly.notice.notice_profile(public=True)
    source = "a" * 40
    seed = 140250512009021371
    prefixes = list(nightly.synthetic_prefixes(engine, seed, nightly.Contract.SUN))
    assert prefixes == list(
        nightly.synthetic_prefixes(engine, seed, nightly.Contract.SUN)
    )
    prefix = prefixes[1]
    exact = nightly.ExactStateNormalizerV1(rules).normalize(prefix)
    service = nightly.UnifiedScenarioProofServiceV1.from_repository(nightly.notice.ROOT)
    scenario_id = "synthetic-candidate"
    fixture, wins = nightly.winning_fixture(
        exact, prefix, service, nightly.canonical_sha256(profile), scenario_id
    )
    assert fixture["expected"]["selectedAction"] == "HA"
    assert wins == {"HK": -1, "HA": 1}
    raw = nightly.viewer.receipt_bytes(fixture)
    digest = nightly.write_json(
        tmp_path_factory.mktemp("scenario") / "fixture.json", fixture
    )["sha256"]
    receipt = nightly.viewer.prove_fixture(
        raw,
        fixture_sha=digest,
        source_sha=source,
        maximum_allocations=4096,
        maximum_nodes=2_000_000,
        cp_work=1,
        backend="ortools",
    )
    assert receipt["status"] == "exact_complete" and receipt["R"] == 1
    margins, _ = nightly.viewer.cp_sat_values(engine, exact.round_state, 2_000_000, 1)
    return SimpleNamespace(
        engine=engine,
        profile=profile,
        source=source,
        prefix=prefix,
        exact=exact,
        service=service,
        scenario_id=scenario_id,
        raw=raw,
        digest=digest,
        receipt=receipt,
        margins=margins,
    )


def test_viewer_score_requires_exact_status_and_matching_rules(sample, rules):
    assert (
        nightly.exact_viewer_inputs(sample.receipt, rules_sha=rules.sha256)
        == sample.receipt["scorerInputs"]
    )
    with pytest.raises(nightly.Rejected, match="viewer_proof_binding_failed"):
        nightly.exact_viewer_inputs(sample.receipt, rules_sha="d" * 64)
    with pytest.raises(nightly.Rejected, match="maximum_nodes"):
        nightly.exact_viewer_inputs(
            {"status": "INCOMPLETE", "reason": "maximum_nodes"}, rules_sha=rules.sha256
        )


@pytest.mark.parametrize(
    "failure", ["input", "hook_layout", "media_preflight", "time_budget", None]
)
def test_only_passing_media_enters_the_artifact(tmp_path, monkeypatch, sample, failure):
    def prove(raw, **options):
        assert (raw, options["fixture_sha"], options["source_sha"]) == (
            sample.raw,
            sample.digest,
            sample.source,
        )
        return sample.receipt

    def hook(bundle_file, bundle_sha, output):
        assert json.loads(bundle_file.read_bytes())["bundleSha256"] == bundle_sha
        nightly.write_json(output / "hook-layout-evidence.json", {"synthetic": True})
        if failure == "hook_layout":
            raise subprocess.CalledProcessError(1, "synthetic-hook")

    def media(bundle, renderer, scratch, ffmpeg):
        video = scratch / "candidate.mp4"
        video.write_bytes(b"synthetic media")
        if failure == "media_preflight":
            raise ValueError("synthetic failed decode")
        if failure == "time_budget":
            raise nightly.SearchBudgetExhausted
        return video, [], {"bundleSha256": bundle.bundle_sha256, "status": "passed"}

    monkeypatch.setattr(nightly.viewer, "prove_fixture", prove)
    monkeypatch.setattr(nightly, "check_hook", hook)
    monkeypatch.setattr(nightly, "render_candidate", media)
    solver = SimpleNamespace(
        solve=lambda *args, **kwargs: {
            "status": "exact",
            "rootActions": [
                {"action": key, "value": value} for key, value in sample.margins.items()
            ],
        }
    )
    prefix = sample.prefix if failure != "input" else {**sample.prefix, "deckOrder": []}
    row = {"id": sample.scenario_id}

    def qualify():
        return nightly.qualify(
            row,
            prefix,
            tmp_path,
            engine=sample.engine,
            solver=solver,
            profile=sample.profile,
            proof_service=sample.service,
            renderer=None,
            source=sample.source,
            library=None,
            ffmpeg=None,
        )

    if failure == "time_budget":
        with pytest.raises(nightly.SearchBudgetExhausted):
            qualify()
        assert row["status"] == "rejected" and row["reasons"] == ["time_budget"]
        assert row["stage"] == "media_preflight"
        assert not (tmp_path / "candidate.mp4").exists()
        return
    passed = qualify()

    assert passed is (failure is None)
    assert (tmp_path / "candidate.mp4").exists() is passed
    assert row["status"] == ("passed" if passed else "rejected")
    assert row["stage"] == (failure or "media_preflight")
    if failure != "input":
        assert row["scorecard"]["score"] >= 60
        assert row["scorecard"]["postAllowed"] is False
        assert row["scorecard"]["exactActorProofSha256"] == row["viewerProof"]["sha256"]
        assert {entry["file"] for entry in row["hookLayout"]["files"]} == {
            "hook-layout-evidence.json"
        }
    if passed:
        assert row["video"] == nightly.file_receipt(tmp_path / "candidate.mp4")


def test_hook_check_requires_the_receipt_for_the_exact_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.json"
    bundle.write_bytes(b"synthetic bundle")
    hook = tmp_path / "hook"
    table = nightly.REMOTION / "scripts/approved-layout-adjacencies.json"
    receipt = {
        "scope": "hook",
        "status": "stills_passed",
        "postingAuthorized": False,
        "binding": {
            "mode": "qualification",
            "bundleSha256": "a" * 64,
            "bundleFileSha256": nightly.file_receipt(bundle)["sha256"],
            "tableSha256": nightly.APPROVED_LAYOUT_TABLE_SHA256,
        },
    }

    def run(args, **kwargs):
        assert args[2:] == ["--check-hook", str(bundle), str(hook)]
        hook.mkdir(exist_ok=True)
        (hook / "layout-table.json").write_bytes(table.read_bytes())
        png = hook / "layout-hook.png"
        png.write_bytes(b"synthetic png")
        receipt["items"] = [
            {
                "phase": "hook",
                "source": png.name,
                "pngSha256": nightly.file_receipt(png)["sha256"],
                "validation": {"status": "passed"},
            }
        ]
        nightly.write_json(hook / "hook-layout-evidence.json", receipt)

    monkeypatch.setattr(nightly.subprocess, "run", run)
    nightly.check_hook(bundle, "a" * 64, hook)
    bundle.write_bytes(b"changed")
    with pytest.raises(nightly.Rejected, match="hook_layout_binding_failed"):
        nightly.check_hook(bundle, "a" * 64, hook)


@pytest.fixture
def batch(tmp_path, monkeypatch, rules):
    output = tmp_path / "candidates"
    monkeypatch.delenv("BALOOT_NIGHTLY_BUDGET_SECONDS", raising=False)
    monkeypatch.delenv("BALOOT_NIGHTLY_JOB_DEADLINE", raising=False)
    monkeypatch.setattr(nightly, "signal", SimpleNamespace())
    monkeypatch.setattr(nightly.RuleBook, "load", lambda path: rules)
    monkeypatch.setattr(nightly.notice, "notice_profile", lambda **kwargs: {})
    monkeypatch.setattr(
        nightly.notice, "RemotionProfessionalRenderer", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        nightly.UnifiedScenarioProofServiceV1, "from_repository", lambda path: None
    )
    monkeypatch.setattr(nightly, "media_tool", lambda *args: None)
    monkeypatch.setattr(
        nightly.viewer,
        "DoubleDummySolver",
        lambda *args: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        nightly,
        "synthetic_prefixes",
        lambda *args: [
            {"playActions": []},
            {"playActions": [{}]},
        ],
    )

    def run():
        code = nightly.generate(
            output,
            run_id=1,
            source="a" * 40,
            tree="b" * 40,
            library=None,
            rounds=3,
            target=2,
        )
        return code, json.loads((output / "candidate-index.json").read_bytes())

    return run


@pytest.mark.parametrize("qualified", [False, True])
def test_budget_stops_before_the_next_candidate(batch, monkeypatch, qualified):
    clock = 0
    attempts = []
    monkeypatch.setattr(nightly, "time", SimpleNamespace(monotonic=lambda: clock))
    assert nightly.budget_seconds() == 18 * 60

    def qualify(row, *args, **kwargs):
        nonlocal clock
        attempts.append(row["id"])
        clock += 18 * 60
        row["status"] = "passed" if qualified else "rejected"
        return qualified

    monkeypatch.setattr(nightly, "qualify", qualify)
    code, index = batch()
    assert len(attempts) == 1
    assert code == (0 if qualified else 1)
    assert index["status"] == ("passed" if qualified else "failed")
    assert index["stopReason"] == "time_budget"
    assert index["qualified"] == (attempts if qualified else [])


def test_job_deadline_reserves_cleanup_time(monkeypatch):
    monkeypatch.setenv("BALOOT_NIGHTLY_BUDGET_SECONDS", "20")
    monkeypatch.setenv("BALOOT_NIGHTLY_JOB_DEADLINE", "130")
    monkeypatch.setattr(nightly, "time", SimpleNamespace(time=lambda: 100))
    assert nightly.budget_seconds() == 15
    monkeypatch.setenv("BALOOT_NIGHTLY_BUDGET_SECONDS", "10")
    assert nightly.budget_seconds() == 10


@pytest.mark.parametrize("stage", ["setup", "search"])
def test_unexpected_failure_finalizes_index(batch, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise AssertionError("synthetic failure")

    if stage == "setup":
        monkeypatch.setattr(nightly, "media_tool", fail)
    else:
        monkeypatch.setattr(nightly, "qualify", fail)
    code, index = batch()
    assert code == 1
    assert index["status"] == "failed"
    assert index["stage"] == stage
    assert index["reason"] == "AssertionError"
