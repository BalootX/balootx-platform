import json
import runpy
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import render_research_notice as notice


def test_notice_post_step_uses_source_manifest_without_git(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    (root / "release").mkdir(parents=True)
    (root / "release/source-manifest.json").write_text(
        json.dumps({"source": {"revision": "a" * 40, "tree": "b" * 40}}),
        encoding="utf-8",
    )
    fixture = root / "examples/fixture.json"
    fixture.parent.mkdir()
    fixture.write_text("{}", encoding="utf-8")
    output = root / "outputs"
    output.mkdir()
    gate_calls = []
    monkeypatch.setattr(notice, "ROOT", root)
    monkeypatch.setattr(notice, "FIXTURE", fixture)
    monkeypatch.setattr(
        notice,
        "RuleBook",
        SimpleNamespace(load=lambda path: object()),
    )
    monkeypatch.setattr(
        notice,
        "notice_profile",
        lambda *, public=False: gate_calls.append("design") or {},
    )
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
                json.dumps({"files": files, "playbackEvidence": evidence}),
                encoding="utf-8",
            )
            return SimpleNamespace(video=video, manifest=manifest)

    monkeypatch.setattr(notice, "RemotionProfessionalRenderer", FakeRenderer)
    original_provenance = notice._write_provenance_notice

    def record_provenance(provenance_output, *args, **kwargs):
        assert (provenance_output / "owner-approval.json").is_file()
        assert (provenance_output / "SHA256SUMS.txt").is_file()
        gate_calls.append("provenance")
        return original_provenance(provenance_output, *args, **kwargs)

    monkeypatch.setattr(notice, "_write_provenance_notice", record_provenance)

    notice.main(output_dir=output)

    receipt = json.loads((output / "owner-approval.json").read_text(encoding="utf-8"))
    assert receipt["sourceHead"] == "a" * 40
    assert receipt["sourceTree"] == "b" * 40
    assert receipt["status"] == "awaiting_owner_approval"
    assert receipt["postingAuthorized"] is False
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

    seeds = [notice.select_deal(seed=str(value))["identifier"] for value in range(8)]
    assert len(set(seeds)) > 1
    assert notice.select_deal(seed="stable")["identifier"] == notice.select_deal(seed="stable")["identifier"]


def test_notice_selectors_buildable_and_invalid_fixtures_rejected(tmp_path, monkeypatch):
    rules = notice.RuleBook.load(notice.ROOT / "configs/rules/saudi_common_v1.yaml")
    for identifier, _, _ in notice.DEAL_SOURCES:
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


def test_notice_selected_main_reaches_gates_with_selected_fixture(tmp_path, monkeypatch):
    selection = notice.select_deal(notice.DEAL_SOURCES[1][0])
    expected_trace = json.loads(selection["fixture"].read_text(encoding="utf-8"))["exactTracePrefix"]
    output = tmp_path / "output"
    output.mkdir()
    events = []
    monkeypatch.setattr(notice, "RuleBook", SimpleNamespace(load=lambda path: object()))
    monkeypatch.setattr(notice, "notice_profile", lambda *, public=False: events.append("design") or {})

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

    monkeypatch.setattr(notice, "validate_notice_bundle", validate)
    monkeypatch.setattr(notice, "timeline_outline", lambda value: [])
    monkeypatch.setattr(notice, "notice_sentence_fit", lambda text: {"widthPx": 1, "availablePx": 1})

    class FakeRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, bundle, *, output_dir, **kwargs):
            events.append("ffprobe")
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
                json.dumps({
                    "identity": bundle.value["identity"],
                    "files": files,
                    "playbackEvidence": evidence,
                }),
                encoding="utf-8",
            )
            return SimpleNamespace(video=video, manifest=manifest)

    monkeypatch.setattr(notice, "RemotionProfessionalRenderer", FakeRenderer)
    original_provenance = notice._write_provenance_notice

    def provenance(*args, **kwargs):
        events.append("provenance")
        return original_provenance(*args, **kwargs)

    monkeypatch.setattr(notice, "_write_provenance_notice", provenance)

    notice.main(
        output_dir=output,
        public=True,
        deal_id=selection["identifier"],
        source_head="a" * 40,
        source_tree="b" * 40,
    )

    assert events == ["design", "build", "engine", "validation", "ffprobe", "provenance"]
    receipt = json.loads((output / "owner-approval.json").read_text(encoding="utf-8"))
    assert receipt["dealIdentifier"] == selection["identifier"]
    assert receipt["fixture"] == selection["source"]
