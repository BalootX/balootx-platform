import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import render_research_notice as notice


def fake_layout_contract(output, video, files, source_identity):
    """Mock renderer boundary only; real geometry is tested by the Node validator."""
    layout = output / "mock-layout.json"
    layout.write_text(json.dumps({"mock": True}), encoding="utf-8")
    for phase in ("hook", "question", "reveal"):
        files[f"layout:{phase}"] = files[f"playback:{phase}"]
    return {"status": "passed", "path": layout.name, "sha256": notice._sha256(layout),
            "mp4Sha256": notice._sha256(video), "binding": source_identity}


def approved_test_design(root, monkeypatch):
    """Synthetic test approval only; never writes a production baseline."""
    source = notice.ROOT
    for name in (*notice.DESIGN_INPUTS, "configs/rules/saudi_common_v1.yaml"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    monkeypatch.setattr(notice, "ROOT", root)
    profile = notice.yaml.safe_load((root / notice.DESIGN_INPUTS[0]).read_text(encoding="utf-8"))
    baseline = {
        "schemaVersion": 1,
        "status": "owner_approved_design_baseline",
        "approvedDesignBinding": profile["approved_design"],
        "approvedDesignInputs": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in notice.DESIGN_INPUTS
        },
    }
    path = root / notice.DESIGN_BASELINE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline), encoding="utf-8")
    return baseline


def test_notice_post_step_uses_source_manifest_without_git(tmp_path, monkeypatch):
    rules = notice.RuleBook.load(notice.ROOT / "configs/rules/saudi_common_v1.yaml")
    source_fixture = notice.FIXTURE.read_bytes()
    root = tmp_path / "archive"
    approved_test_design(root, monkeypatch)
    (root / "release/source-manifest.json").write_text(
        json.dumps({"source": {"revision": "a" * 40, "tree": "b" * 40}}),
        encoding="utf-8",
    )
    fixture = root / "examples/fixture.json"
    fixture.parent.mkdir()
    fixture.write_bytes(source_fixture)
    output = root / "outputs"
    output.mkdir()
    gate_calls = []
    monkeypatch.setattr(notice, "ROOT", root)
    monkeypatch.setattr(notice, "FIXTURE", fixture)
    monkeypatch.setattr(
        notice,
        "RuleBook",
        SimpleNamespace(load=lambda path: rules),
    )
    original_profile = notice.notice_profile

    def profile(**kwargs):
        gate_calls.append("design")
        return original_profile(**kwargs)

    monkeypatch.setattr(notice, "notice_profile", profile)
    bundle = SimpleNamespace(
        value={
            "timeline": {
                "segments": [
                    {"stage": stage, "durationFrames": 30, "startFrame": index * 30}
                    for index, stage in enumerate(notice.STAGES)
                ]
            },
            "render": {"fps": 30},
            "localizations": {"video.notice.title": "title", "video.notice.sentence": "sentence"},
            "identity": {"players": {"north": "N", "east": "E", "south": "S", "west": "W"}},
            "source": {"decisionReference": "fixture:1"},
        },
        bundle_sha256="bundle-sha",
    )
    bundle.value["identity"]["players"] = notice.seeded_player_names(
        notice.yaml.safe_load((root / notice.DESIGN_INPUTS[0]).read_text(encoding="utf-8"))["identity"],
        notice.DEAL_SOURCES[0][0],
    )
    monkeypatch.setattr(notice, "build_notice_bundle", lambda rules, profile, **kwargs: bundle)
    monkeypatch.setattr(notice, "engine_round_result", lambda rules, fixture=None: {
        "matchScore": [0, 46],
        "roundScore": {
            **{key: 0 for key in ("cardGamePoints", "projectPoints", "balootPoints", "awardedPoints")},
            "winner": "south",
        },
    })
    monkeypatch.setattr(
        notice,
        "validate_notice_bundle",
        lambda bundle, engine_result, *, public=False, fixture=None: gate_calls.append("validation"),
    )
    monkeypatch.setattr(notice, "timeline_outline", lambda value: [])
    monkeypatch.setattr(notice, "notice_sentence_fit", lambda text: {"widthPx": 1, "availablePx": 1})

    class FakeRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, bundle, *, output_dir, **kwargs):
            gate_calls.append("ffprobe")
            output_dir.mkdir(parents=True, exist_ok=True)
            video = output_dir / "clip.mp4"
            video.write_bytes(b"video")
            files = {}
            evidence = []
            for index, phase in enumerate(notice.PHASES):
                name = f"{phase}.png"
                Image.new("RGB", (1080, 1920)).save(output_dir / name)
                files[f"playback:{phase}"] = {"name": name}
                evidence.append({"checkpointPhase": phase, "frame": index})
            manifest = output_dir / "manifest.json"
            manifest.write_text(
                json.dumps({"files": files, "playbackEvidence": evidence,
                            "layoutGeometry": fake_layout_contract(output_dir, video, files, kwargs["source_identity"])}),
                encoding="utf-8",
            )
            return SimpleNamespace(video=video, manifest=manifest)

    monkeypatch.setattr(notice, "RemotionProfessionalRenderer", FakeRenderer)
    original_provenance = notice._write_provenance_notice

    def record_provenance(provenance_output, *args, **kwargs):
        assert (provenance_output / notice.GATE_RESULT).is_file()
        assert not (provenance_output / "SHA256SUMS.txt").exists()
        gate_calls.append("provenance")
        return original_provenance(provenance_output, *args, **kwargs)

    monkeypatch.setattr(notice, "_write_provenance_notice", record_provenance)

    copied = tmp_path / "copied"
    notice.main(output_dir=output, copy_to=copied)

    receipt = json.loads((output / "owner-approval.json").read_text(encoding="utf-8"))
    assert receipt["sourceHead"] == "a" * 40
    assert receipt["sourceTree"] == "b" * 40
    assert receipt["status"] == "awaiting_owner_approval"
    assert receipt["postingAuthorized"] is False
    provenance = json.loads((output / "provenance-notice.json").read_text(encoding="utf-8"))
    assert provenance["checks"] == {"source_identity_declaration": {"status": "passed"}}
    assert provenance["scope"] == "source identity declaration validation only"
    assert provenance["sourceTreeContentsVerified"] is False
    assert receipt["measuredDesignInputs"] == {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in notice.DESIGN_INPUTS
    }
    assert receipt["approvedDesignInputs"] == receipt["measuredDesignInputs"]
    assert "approvedDesignCommit" not in receipt
    assert "approvedDesignChange" not in receipt
    gate = json.loads((output / notice.GATE_RESULT).read_text(encoding="utf-8"))
    assert gate["status"] == "passed"
    assert gate["kind"] == "render_validation_only"
    assert gate["postingAuthorized"] is False
    assert "exact MP4" in gate["ownerReleaseRequired"]
    assert gate["sourceHead"] == receipt["sourceHead"]
    assert gate["sourceTree"] == receipt["sourceTree"]
    assert gate["fixtureSha256"] == notice._sha256(fixture)
    assert gate["mp4Sha256"] == notice._sha256(output / "clip.mp4")
    assert gate["manifestSha256"] == notice._sha256(output / "manifest.json")
    assert set(gate["checks"]) == {
        "notice_profile", "validate_design_input_integrity", "build_notice_bundle",
        "engine_round_result", "validate_notice_bundle", "RemotionProfessionalRenderer.execute",
        "render_input_integrity", "render_receipt_content", "provenance_notice",
        "validate_rendered_layout_geometry",
    }
    assert all(check["status"] == "passed" for check in gate["checks"].values())
    assert receipt["renderGate"]["sha256"] == notice._sha256(output / notice.GATE_RESULT)
    for name, digest in receipt["files"].items():
        assert notice._sha256(output / name) == digest
    checksum_bytes = (output / "SHA256SUMS.txt").read_bytes()
    assert b"\r" not in checksum_bytes
    assert checksum_bytes.endswith(b"\n")
    assert (copied / "SHA256SUMS.txt").read_bytes() == checksum_bytes
    checker = shutil.which("sha256sum")
    if checker is None:
        checker = str(Path(shutil.which("git")).parent.parent / "usr/bin/sha256sum.exe")
    subprocess.run(
        [checker, "-c", "SHA256SUMS.txt"], cwd=copied,
        capture_output=True, text=True, check=True, timeout=20,
    )
    sums = checksum_bytes.decode("utf-8").splitlines()
    sealed_names = set()
    for row in sums:
        digest, name = row.split("  ", 1)
        sealed_names.add(name)
        assert notice._sha256(output / name) == digest
        assert notice._sha256(copied / name) == digest
    assert sealed_names == {*receipt["files"], "owner-approval.json"}
    assert notice.GATE_RESULT in sealed_names
    assert "provenance-notice.json" in sealed_names
    assert (output / "SHA256SUMS.txt").is_file()
    assert gate_calls == ["design", "validation", "ffprobe", "provenance"]
    assert not (root / ".git").exists()

    gate_calls.clear()
    (root / "release/source-manifest.json").unlink()
    failed_output = root / "failed-output"
    with pytest.raises(SystemExit, match="1"):
        notice.main(output_dir=failed_output)
    failed_provenance = json.loads(
        (failed_output / "provenance-notice.json").read_text(encoding="utf-8")
    )
    assert failed_provenance["status"] == "failed"
    failed_gate = json.loads((failed_output / notice.GATE_RESULT).read_text(encoding="utf-8"))
    assert failed_gate["status"] == "failed"
    assert failed_gate["checks"]["provenance_notice"] == failed_provenance
    assert (failed_output / "owner-approval.json").is_file()
    assert (failed_output / "SHA256SUMS.txt").is_file()
    assert gate_calls == ["design", "validation", "ffprobe", "provenance"]

    (root / "release/source-manifest.json").write_text(
        json.dumps({"source": {"revision": "a" * 40, "tree": "b" * 40}}),
        encoding="utf-8",
    )
    (root / ".git").write_text("gitdir: unavailable", encoding="utf-8")
    gate_calls.clear()
    checkout_output = root / "checkout-output"
    with pytest.raises(SystemExit, match="1"):
        notice.main(output_dir=checkout_output)
    checkout_provenance = json.loads(
        (checkout_output / "provenance-notice.json").read_text(encoding="utf-8")
    )
    assert checkout_provenance["status"] == "failed"
    assert gate_calls == ["design", "validation", "ffprobe", "provenance"]


def test_notice_selector_binds_deal_and_name_draw_to_manifest(tmp_path):
    with pytest.raises(ValueError, match="unknown governed deal identifier"):
        notice.select_deal("sun-hundred-project-no-salvage-v1")

    first = notice.select_deal(notice.DEAL_SOURCES[0][0])
    second = notice.select_deal(notice.DEAL_SOURCES[1][0])
    assert first["identifier"] != second["identifier"]
    assert first["fixture"] != second["fixture"]

    profile = {
        "identity": {"player_name_pool": list("abcdefgh"), "default_player_names": {}}
    }
    manifests = []
    for selection in (first, second):
        identity_seed = str(selection["identifier"])
        players = notice.seeded_player_names(profile["identity"], identity_seed)
        bundle = SimpleNamespace(
            value={"identity": {"gameSerial": identity_seed, "players": players}}
        )
        manifest_path = tmp_path / f"{identity_seed}.json"
        manifest_path.write_text(
            json.dumps({"identity": bundle.value["identity"]}), encoding="utf-8"
        )
        manifests.append(notice._record_deal_manifest(manifest_path, selection, bundle, profile))

    assert manifests[0]["dealIdentifier"] != manifests[1]["dealIdentifier"]
    assert manifests[0]["dealSource"] != manifests[1]["dealSource"]
    assert manifests[0]["dealSeed"] is None and manifests[1]["dealSeed"] is None
    assert manifests[0]["nameDraw"] != manifests[1]["nameDraw"]
    assert manifests[0]["manifestSha256"] != manifests[1]["manifestSha256"]

    first_trace = json.loads(first["fixture"].read_text(encoding="utf-8"))["exactTracePrefix"]
    second_trace = json.loads(second["fixture"].read_text(encoding="utf-8"))["exactTracePrefix"]
    assert first_trace != second_trace

    seeds = []
    for value in range(8):
        try:
            seeds.append(notice.select_deal(seed=str(value))["identifier"])
        except ValueError as error:
            assert "viewer=buyer" in str(error)
    assert len(set(seeds)) > 1
    assert notice.select_deal(seed="stable")["identifier"] == notice.select_deal(seed="stable")["identifier"]


def test_notice_selectors_buildable_and_invalid_fixtures_rejected(tmp_path, monkeypatch):
    rules = notice.RuleBook.load(notice.ROOT / "configs/rules/saudi_common_v1.yaml")
    for identifier, _, _ in notice.DEAL_SOURCES:
        if identifier == "sun-single-winner-20261116-04-v1":
            with pytest.raises(ValueError, match="viewer=buyer"):
                notice.select_deal(identifier)
            continue
        selection = notice.select_deal(identifier)
        fixture = json.loads(selection["fixture"].read_text(encoding="utf-8"))
        prefix = fixture["exactTracePrefix"]
        plays = [
            {"seat": row["actor"], "card": {"code": row["card"]}}
            for row in prefix["playActions"]
        ]
        for card in fixture["expected"]["principalVariation"]:
            raw = [{"actor": play["seat"], "card": play["card"]["code"]} for play in plays]
            actor = notice.ExactStateNormalizerV1(rules).normalize(
                {**prefix, "playActions": raw, "stateAssertions": {}}
            ).next_actor
            plays.append({"seat": actor, "card": {"code": card}})
        assert notice.complete_trace(rules, plays, selection["fixture"])[
            "completeExactNormalization"
        ]["phase"] == "scored"

    identifier = "invalid-governed-deal-v1"
    (tmp_path / "invalid.json").write_text(
        json.dumps({
            "scenarioId": identifier,
            "exactTracePrefix": {"playActions": []},
            "expected": {"selectedAction": "DA", "winningRootActionCount": 1},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(notice, "ROOT", tmp_path)
    monkeypatch.setattr(
        notice,
        "DEAL_SOURCES",
        ((identifier, "invalid.json", "invalid-governed-deal"),),
    )
    with pytest.raises(ValueError, match="not buildable"):
        notice.select_deal(identifier)


@pytest.mark.parametrize("selector", [{"deal_id": "fixture"}, {"seed": "1"}])
def test_notice_selectors_require_public_in_main_and_cli(monkeypatch, capsys, selector):
    with pytest.raises(ValueError, match="require --public"):
        notice.main(**selector)

    flag, value = ("--deal-id", selector["deal_id"]) if "deal_id" in selector else ("--seed", selector["seed"])
    monkeypatch.setattr(sys, "argv", ["render_research_notice.py", flag, value])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(notice.ROOT / "scripts/render_research_notice.py"), run_name="__main__")
    assert error.value.code == 2
    assert "require --public" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [None, "content", "ffprobe", "design_drift", "fixture_drift", "baseline_drift", "thumbnail", "layout_missing", "layout_source", "layout_drift"])
def test_notice_selected_main_reaches_gates_with_selected_fixture(tmp_path, monkeypatch, failure):
    selection = notice.select_deal(notice.DEAL_SOURCES[1][0])
    expected_trace = json.loads(selection["fixture"].read_text(encoding="utf-8"))["exactTracePrefix"]
    source = tmp_path / "source"
    approved_test_design(source, monkeypatch)
    fixture_copy = source / selection["source"]
    fixture_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selection["fixture"], fixture_copy)
    selection["fixture"] = fixture_copy
    monkeypatch.setattr(notice, "select_deal", lambda *args: selection)
    output = tmp_path / "output"
    output.mkdir()
    events = []
    monkeypatch.setattr(notice, "RuleBook", SimpleNamespace(load=lambda path: object()))
    original_profile = notice.notice_profile

    def profile(**kwargs):
        events.append("design")
        return original_profile(**kwargs)

    monkeypatch.setattr(notice, "notice_profile", profile)

    bundle = SimpleNamespace(
        value={
            "timeline": {
                "segments": [
                    {"stage": stage, "durationFrames": 30, "startFrame": index * 30}
                    for index, stage in enumerate(notice.STAGES)
                ]
            },
            "render": {"fps": 30},
            "localizations": {"video.notice.title": "title", "video.notice.sentence": "sentence"},
            "identity": {
                "gameSerial": selection["identifier"],
                "players": {"north": "N", "east": "E", "south": "S", "west": "W"},
            },
            "source": {"decisionReference": "fixture:1"},
        },
        bundle_sha256="bundle-sha",
    )

    bundle.value["identity"]["players"] = notice.seeded_player_names(
        notice.yaml.safe_load((source / notice.DESIGN_INPUTS[0]).read_text(encoding="utf-8"))["identity"],
        selection["identifier"],
    )

    def build(rules, profile, *, fixture, golden):
        events.append("build")
        assert fixture == selection["fixture"]
        assert golden == selection["golden"]
        assert json.loads(fixture.read_text(encoding="utf-8"))["exactTracePrefix"] == expected_trace
        return bundle

    def engine(rules, fixture):
        events.append("engine")
        assert fixture == selection["fixture"]
        return {
            "matchScore": [0, 46],
            "roundScore": {
                **{key: 0 for key in ("cardGamePoints", "projectPoints", "balootPoints", "awardedPoints")},
                "winner": "south",
            },
        }

    monkeypatch.setattr(notice, "build_notice_bundle", build)
    monkeypatch.setattr(notice, "engine_round_result", engine)
    def validate(bundle, engine_result, *, public=False, fixture=None):
        events.append("validation")
        assert fixture == selection["fixture"]
        if failure == "content":
            raise ValueError("content proof failed")

    monkeypatch.setattr(notice, "validate_notice_bundle", validate)
    monkeypatch.setattr(notice, "timeline_outline", lambda value: [])
    monkeypatch.setattr(notice, "notice_sentence_fit", lambda text: {"widthPx": 1, "availablePx": 1})

    class FakeRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, bundle, *, output_dir, **kwargs):
            events.append("ffprobe")
            if failure == "ffprobe":
                raise RuntimeError("ffprobe dimensions invalid")
            output_dir.mkdir(parents=True, exist_ok=True)
            video = output_dir / "clip.mp4"
            video.write_bytes(b"video")
            files = {}
            evidence = []
            for index, phase in enumerate(notice.PHASES):
                name = f"{phase}.png"
                size = (1, 1) if failure == "thumbnail" and phase == "hook" else (1080, 1920)
                Image.new("RGB", size).save(output_dir / name)
                files[f"playback:{phase}"] = {"name": name}
                evidence.append({"checkpointPhase": phase, "frame": index})
            manifest = output_dir / "manifest.json"
            layout = fake_layout_contract(output_dir, video, files, kwargs["source_identity"])
            if failure == "layout_missing":
                layout = None
            elif failure == "layout_source":
                layout["binding"] = {"sourceHead": "c" * 40, "sourceTree": "b" * 40}
            elif failure == "layout_drift":
                with (output_dir / layout["path"]).open("a", encoding="utf-8") as handle:
                    handle.write("\n")
            manifest.write_text(
                json.dumps({
                    "identity": bundle.value["identity"],
                    "files": files,
                    "playbackEvidence": evidence,
                    "layoutGeometry": layout,
                }),
                encoding="utf-8",
            )
            if failure == "design_drift":
                with (source / notice.DESIGN_INPUTS[-1]).open("a", encoding="utf-8") as handle:
                    handle.write("\n/* changed during render */\n")
            if failure == "baseline_drift":
                with (source / notice.DESIGN_BASELINE).open("a", encoding="utf-8") as handle:
                    handle.write("\n")
            if failure == "fixture_drift":
                fixture_copy.write_bytes(fixture_copy.read_bytes() + b"\n")
            return SimpleNamespace(video=video, manifest=manifest)

    monkeypatch.setattr(notice, "RemotionProfessionalRenderer", FakeRenderer)
    original_provenance = notice._write_provenance_notice

    def provenance(*args, **kwargs):
        events.append("provenance")
        return original_provenance(*args, **kwargs)

    monkeypatch.setattr(notice, "_write_provenance_notice", provenance)

    arguments = dict(
        output_dir=output,
        public=True,
        deal_id=selection["identifier"],
        source_head="a" * 40,
        source_tree="b" * 40,
    )

    if failure is not None:
        stage, message, error_type = {
            "content": ("validate_notice_bundle", "content proof failed", ValueError),
            "ffprobe": ("RemotionProfessionalRenderer.execute", "ffprobe dimensions invalid", RuntimeError),
            "design_drift": ("render_input_integrity", "render inputs changed during rendering", ValueError),
            "fixture_drift": ("render_input_integrity", "render inputs changed during rendering", ValueError),
            "baseline_drift": ("render_input_integrity", "render inputs changed during rendering", ValueError),
            "thumbnail": ("render_receipt_content", "thumbnail must be the portrait paused decision frame", ValueError),
            "layout_missing": ("validate_rendered_layout_geometry", "measured layout geometry gate is missing or unbound to MP4", ValueError),
            "layout_source": ("validate_rendered_layout_geometry", "layout geometry source binding mismatch", ValueError),
            "layout_drift": ("validate_rendered_layout_geometry", "layout geometry result hash mismatch", ValueError),
        }[failure]
        with pytest.raises(error_type, match=message):
            notice.main(**arguments)
        gate = json.loads((output / notice.GATE_RESULT).read_text(encoding="utf-8"))
        assert gate["status"] == "failed"
        assert gate["checks"][stage] == {"status": "failed", "error": message, "errorType": error_type.__name__}
        assert "provenance" not in events
        assert not (output / "owner-approval.json").exists()
        assert not (output / "SHA256SUMS.txt").exists()
        return

    notice.main(**arguments)

    assert events == ["design", "build", "engine", "validation", "ffprobe", "provenance"]
    receipt = json.loads((output / "owner-approval.json").read_text(encoding="utf-8"))
    assert receipt["dealIdentifier"] == selection["identifier"]
    assert receipt["fixture"] == selection["source"]
    gate = json.loads((output / notice.GATE_RESULT).read_text(encoding="utf-8"))
    assert gate["checks"]["validate_approved_v4_public_profile"] == {"status": "passed"}
    assert gate["dealIdentifier"] == selection["identifier"]
    assert gate["measuredDesignInputs"] == receipt["measuredDesignInputs"]
    assert receipt["approvedDesignInputs"] == receipt["measuredDesignInputs"] == {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in notice.DESIGN_INPUTS
    }


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("explicit_output", [False, True])
@pytest.mark.parametrize("markers", [
    ("owner-approval.json",), ("SHA256SUMS.txt",), ("owner-approval.json", "SHA256SUMS.txt"),
])
def test_notice_refuses_existing_packet_before_any_write(tmp_path, monkeypatch, public, explicit_output, markers):
    monkeypatch.setattr(notice, "ROOT", tmp_path)
    output = tmp_path / ("explicit" if explicit_output else "outputs/approved-v4-public" if public else "outputs/research-notice-4")
    output.mkdir(parents=True)
    (output / "old.mp4").write_bytes(b"previous encoded MP4\x00\xff")
    (output / "manifest.json").write_bytes(b'{"status":"passed"}\r\n')
    if "owner-approval.json" in markers:
        (output / "owner-approval.json").write_bytes(b'{"status":"awaiting_owner_approval","postingAuthorized":false}\r\n')
    if "SHA256SUMS.txt" in markers:
        names = ["old.mp4", "manifest.json", *[name for name in markers if name != "SHA256SUMS.txt"]]
        (output / "SHA256SUMS.txt").write_bytes(
            "".join(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n" for name in names).encode("utf-8")
        )

    def snapshot():
        return {path.relative_to(output): path.read_bytes() if path.is_file() else None for path in output.rglob("*")}

    before = snapshot()
    monkeypatch.setattr(notice, "_json_write", lambda *args: pytest.fail("existing packet reached an output write"))
    copied = tmp_path / "copied"
    with pytest.raises(ValueError, match="use a fresh output directory"):
        notice.main(output_dir=output if explicit_output else None, public=public, copy_to=copied)
    assert snapshot() == before
    assert not (output / notice.GATE_RESULT).exists()
    assert not copied.exists()


@pytest.mark.parametrize("public", [False, True])
def test_notice_profile_measures_actual_bytes_and_guards_both_modes(tmp_path, monkeypatch, public):
    baseline = approved_test_design(tmp_path, monkeypatch)
    evidence = {}
    profile = notice.notice_profile(public=public, design_evidence=evidence)
    measured = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in notice.DESIGN_INPUTS
    }
    assert evidence["measuredDesignInputs"] == measured
    assert evidence["approvedDesignInputs"] == baseline["approvedDesignInputs"] == measured
    assert evidence["approvedDesignBinding"] == profile["approved_design"]
    assert evidence["approvedDesignBindingStatus"] == "matched_retained_baseline"
    path = tmp_path / notice.DESIGN_INPUTS[-1]
    path.write_bytes(path.read_bytes() + b"\n/* unapproved change */\n")
    with pytest.raises(ValueError, match="approved Majlis design inputs changed"):
        notice.notice_profile(public=public, design_evidence=evidence)
    assert evidence["measuredDesignInputs"][notice.DESIGN_INPUTS[-1]] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence["measuredDesignInputs"] != baseline["approvedDesignInputs"]


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("defect", ["missing", "malformed", "empty", "version", "status", "binding", "hashes", "mismatch", "reference", "missing_input"])
def test_notice_gate_fails_closed_with_durable_measured_result(tmp_path, monkeypatch, public, defect):
    baseline = approved_test_design(tmp_path / "source", monkeypatch)
    path = notice.ROOT / notice.DESIGN_BASELINE
    if defect == "missing":
        path.unlink()
    elif defect == "malformed":
        path.write_text("{", encoding="utf-8")
    elif defect == "missing_input":
        (notice.ROOT / notice.DESIGN_INPUTS[-1]).unlink()
    else:
        if defect == "empty":
            baseline = {}
        elif defect == "version":
            baseline["schemaVersion"] = True
        elif defect == "status":
            baseline["status"] = "pending"
        elif defect == "binding":
            baseline["approvedDesignBinding"]["source_tree"] = "0" * 40
        elif defect == "hashes":
            baseline["approvedDesignInputs"].pop(notice.DESIGN_INPUTS[0])
        elif defect == "mismatch":
            baseline["approvedDesignInputs"][notice.DESIGN_INPUTS[0]] = "0" * 64
        elif defect == "reference":
            config = notice.ROOT / notice.DESIGN_INPUTS[0]
            profile = notice.yaml.safe_load(config.read_text(encoding="utf-8"))
            profile["approved_design"]["reference_mp4_sha256"] = "0" * 64
            config.write_text(notice.yaml.safe_dump(profile), encoding="utf-8")
            baseline["approvedDesignBinding"] = profile["approved_design"]
            baseline["approvedDesignInputs"][notice.DESIGN_INPUTS[0]] = hashlib.sha256(config.read_bytes()).hexdigest()
        path.write_text(json.dumps(baseline), encoding="utf-8")
    output = tmp_path / "failed"
    with pytest.raises((ValueError, FileNotFoundError)):
        notice.main(output_dir=output, public=public, source_head="a" * 40, source_tree="b" * 40)
    gate = json.loads((output / notice.GATE_RESULT).read_text(encoding="utf-8"))
    assert gate["status"] == "failed"
    assert gate["postingAuthorized"] is False
    assert gate["approvedDesignBindingStatus"] == "not_validated"
    assert gate["checks"]["notice_profile"]["status"] == "failed"
    assert "RemotionProfessionalRenderer.execute" not in gate["checks"]
    assert gate["measuredDesignInputs"][notice.DESIGN_INPUTS[0]] == hashlib.sha256((notice.ROOT / notice.DESIGN_INPUTS[0]).read_bytes()).hexdigest()
    assert not (output / "owner-approval.json").exists()
    assert not (output / "SHA256SUMS.txt").exists()


def test_gate_persistence_error_preserves_original_stage_failure(tmp_path, monkeypatch):
    calls = []

    def write(path, value):
        calls.append(value["status"])
        if value["status"] == "failed":
            raise OSError("storage unavailable")

    monkeypatch.setattr(notice, "_json_write", write)
    gate = {"status": "running", "checks": {}}
    with pytest.raises(ValueError, match="content proof failed") as error:
        with notice._gate_check(gate, tmp_path / "gate.json", "validate_notice_bundle"):
            raise ValueError("content proof failed")
    assert calls == ["running", "failed"]
    assert "storage unavailable" in error.value.__notes__[0]


@pytest.mark.parametrize("head, tree, status", [
    ("a" * 40, "b" * 40, "passed"),
    ("a" * 64, "b" * 64, "passed"),
    ("A" * 40, "b" * 40, "failed"),
    ("a" * 40, "bad", "failed"),
    ("a" * 40, None, "failed"),
])
def test_provenance_reports_exact_checked_scope(tmp_path, head, tree, status):
    result = notice._write_provenance_notice(tmp_path, source_head=head, source_tree=tree)
    assert result["status"] == status
    assert result["checks"] == {"source_identity_declaration": {"status": status}}
    assert result["scope"] == "source identity declaration validation only"
    assert "lowercase 40- or 64-digit hexadecimal" in result["checkDescription"]
    assert result["sourceTreeContentsVerified"] is False
    assert json.loads((tmp_path / "provenance-notice.json").read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("public, selector", [(True, "id"), (True, "seed"), (True, "default"), (False, "default")])
def test_viewer_buyer_real_deal04_refused_before_profile_or_render(tmp_path, monkeypatch, public, selector):
    fixture = notice.ROOT / "examples/research-scenario-v4-single-winner-20261116-04.json"
    monkeypatch.setattr(notice, "notice_profile", lambda **kwargs: pytest.fail("barred deal reached profile/render"))
    monkeypatch.setattr(notice, "RemotionProfessionalRenderer", lambda *args, **kwargs: pytest.fail("barred deal reached renderer"))
    kwargs = {"deal_id": "sun-single-winner-20261116-04-v1"} if selector == "id" else {"seed": "7"} if selector == "seed" else {}
    if selector == "default":
        monkeypatch.setattr(notice, "FIXTURE", fixture)
        monkeypatch.setattr(notice, "GOLDEN", "sun-single-winner-20261116-04")
    with pytest.raises(ValueError, match="viewer=buyer"):
        notice.main(public=public, output_dir=tmp_path / "unused", **kwargs)
    assert not (tmp_path / "unused").exists()


def test_viewer_buyer_direct_builder_and_actual_catalog_census(monkeypatch):
    rules = notice.RuleBook.load(notice.ROOT / "configs/rules/saudi_common_v1.yaml")
    affected = []
    for identifier, relative, golden in notice.DEAL_SOURCES:
        fixture = notice.ROOT / relative
        normalized = notice.ExactStateNormalizerV1(rules).normalize(json.loads(fixture.read_text(encoding="utf-8"))["exactTracePrefix"])
        if normalized.next_actor == normalized.state["buyer"]:
            affected.append(identifier)
            monkeypatch.setattr(notice, "RemotionProfessionalRenderer", lambda *args, **kwargs: pytest.fail("barred direct builder reached renderer"))
            with pytest.raises(ValueError, match="viewer=buyer"):
                notice.build_notice_bundle(rules, fixture=fixture, golden=golden)
        else:
            admission = notice.validate_notice_fixture_admission(fixture, rules)
            assert admission["viewer"] != admission["buyer"]
    assert affected == ["sun-single-winner-20261116-04-v1"]


@pytest.mark.parametrize("viewer,buyer", [(seat, seat) for seat in ("north", "west", "south", "east")] + [(None, "north"), ("west", None), ("unknown", "north")])
def test_viewer_buyer_bundle_guard_fails_closed(viewer, buyer):
    value = {"observerSeat": viewer, "scenes": [{"layers": [{"id": "public-lifecycle", "content": {"buyer": buyer}}]}]}
    with pytest.raises(ValueError, match="viewer[=/]buyer"):
        notice.validate_decision_clip_admission(value)
