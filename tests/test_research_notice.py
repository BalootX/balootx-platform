import hashlib
import json
import runpy
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import render_research_notice as notice


@pytest.fixture
def approved_layout_bytes(root):
    return (root / "src/baloot_platform/video/remotion/scripts/approved-layout-adjacencies.json").read_bytes()


@pytest.mark.parametrize("legacy_title", [None, "ملاحظة بحثية"], ids=["title-absent", "legacy-title"])
def test_research_notice_renderer_accepts_optional_title(root, legacy_title):
    profile = notice.notice_profile(public=True)
    title_key = "video.notice.title"
    if legacy_title is not None:
        profile["localizations"][title_key] = legacy_title
    renderer = notice.RemotionProfessionalRenderer(profile, repository_root=root)
    assert renderer.profile["localizations"].get(title_key) == legacy_title


def test_public_notice_profile_builds_a_qualified_bundle_with_the_exact_notice():
    profile = notice.notice_profile(public=True)
    design = profile["approved_design"]
    assert design["variant"] == "public_v4_no_justification_red_notice"
    assert design["reference_variant"] == "public_v4_no_justification_no_notice"
    assert design["reference_mp4_sha256"] == "15492d3f389fcb07be423fc97a6e46fe0b8e385b5beb173f01073e6d11a87bf9"

    rules = notice.RuleBook.load(notice.ROOT / "configs/rules/saudi_common_v1.yaml")
    bundle = notice.build_notice_bundle(rules, profile)
    notice.validate_notice_bundle(bundle, notice.engine_round_result(rules), public=True)
    sentence = bundle.value["localizations"]["video.notice.sentence"]
    assert "video.notice.title" not in profile["localizations"]
    assert "video.notice.title" not in bundle.value["localizations"]
    assert sentence == "هذا الفيديو لأغراض البحث فقط، ويُمنع تداوله أو نشره أو مشاركته"
    assert len(sentence.encode("utf-8")) == 114
    assert hashlib.sha256(sentence.encode("utf-8")).hexdigest() == "290e2a2d03424ea7846a71afc80e019140fac3d56e465b4766a89d57a3c3b8ac"
    assert bundle.value["presentation"]["majlis"]["research_notice"] is True
    presentation = bundle.value["presentation"]["presentation"]
    assert presentation["show_justification"] is presentation["show_justification_text"] is False


@pytest.mark.parametrize("override", [None, "BALOOTX_LAYOUT_TABLE", "BALOOTX_LAYOUT_TABLE_SHA256"])
def test_renderer_rejects_unapproved_layout_table_before_render(
    root, tmp_path, monkeypatch, approved_layout_bytes, override
):
    profile = notice.notice_profile(public=True)
    rules = notice.RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    bundle = notice.build_notice_bundle(rules, profile)
    renderer = notice.RemotionProfessionalRenderer(profile, repository_root=root)
    renderer.renderer_root = tmp_path
    table = tmp_path / "scripts" / "approved-layout-adjacencies.json"
    table.parent.mkdir()
    table.write_bytes(approved_layout_bytes if override else approved_layout_bytes + b"\n")
    if override:
        monkeypatch.setenv(override, "unapproved")
    monkeypatch.setattr(renderer, "_run_renderer", lambda **_: pytest.fail("renderer started"))

    error = "overrides require NEVER_POST" if override else "approved layout table SHA-256 mismatch"
    with pytest.raises(ValueError, match=error):
        renderer.execute(bundle, export_preset="decision_clip", output_dir=tmp_path)


@pytest.mark.parametrize(("section", "key", "value", "error"), [
    ("majlis", "research_notice", False, "research notice must match"),
    ("localizations", "video.notice.sentence", "هذا الفيديو لأغراض البحث فقط، ويُمنع تداوله.", "sentence hash mismatch"),
    ("localizations", "video.notice.sentence", "changed", "sentence hash mismatch"),
    ("approved_design", "reference_variant", "public_v4_no_justification_red_notice", "binding mismatch: reference_variant"),
    ("approved_design", "reference_mp4_sha256", "0" * 64, "binding mismatch: reference_mp4_sha256"),
    ("presentation", "show_justification", True, "disable all justification text"),
    ("presentation", "show_justification_text", True, "disable all justification text"),
])
def test_public_notice_profile_rejects_unapproved_changes(section, key, value, error):
    profile = notice.notice_profile(public=True)
    profile[section][key] = value
    with pytest.raises(ValueError, match=error):
        notice.validate_approved_v4_public_profile(profile)


def _fake_qualification(video, bundle_sha, files, table_bytes):
    output = video.parent
    table = output / "layout-table.json"
    table.write_bytes(table_bytes)
    table_sha = notice._sha256(table)
    video_sha = notice._sha256(video)
    (output / "layout-evidence.json").write_text(json.dumps({
        "status": "passed", "videoSha256": video_sha,
        "binding": {"mode": "qualification", "bundleSha256": bundle_sha, "tableSha256": table_sha},
    }), encoding="utf-8")
    phases = ("hook", "question", "reveal")
    names = {
        **{f"layout:{phase}": f"{phase}.png" for phase in phases},
        "layoutTable": "layout-table.json",
        "layoutGeometry": "layout-evidence.json",
    }
    files.update({key: {"name": name, "sha256": notice._sha256(output / name)} for key, name in names.items()})
    return {
        "mode": "qualification", "status": "passed", "phases": dict.fromkeys(phases, "passed"),
        "bundleSha256": bundle_sha, "rendererVideoSha256": video_sha, "videoSha256": video_sha,
        "tableSha256": table_sha, "evidenceSha256": files["layoutGeometry"]["sha256"],
    }


def test_notice_post_step_uses_source_manifest_without_git(tmp_path, monkeypatch, approved_layout_bytes):
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
            "localizations": {"video.notice.sentence": "sentence"},
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
                json.dumps({
                    "files": files, "playbackEvidence": evidence,
                    "layoutGeometry": _fake_qualification(video, bundle.bundle_sha256, files, approved_layout_bytes),
                }),
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


@pytest.mark.parametrize("stale", [None, "image", "source", "lockfile", "dependencies"])
def test_notice_selected_main_reaches_gates_with_selected_fixture(
    tmp_path, monkeypatch, approved_layout_bytes, stale
):
    selection = notice.select_deal(notice.DEAL_SOURCES[1][0])
    expected_trace = json.loads(selection["fixture"].read_text(encoding="utf-8"))["exactTracePrefix"]
    design = notice.notice_profile(public=True)["approved_design"]
    root = tmp_path / "source"
    renderer_root = root / "src/baloot_platform/video/remotion"
    dependency = renderer_root / "node_modules/renderer/package.json"
    dependency.parent.mkdir(parents=True)
    dependency.write_text(
        json.dumps({"version": "1.0.0" if stale == "dependencies" else "2.0.0"}),
        encoding="utf-8",
    )
    lockfile = renderer_root / "package-lock.json"
    lockfile.write_text(
        '{"lockfileVersion":3,"packages":{"node_modules/renderer":{"version":"2.0.0"}}}',
        encoding="utf-8",
    )
    lock_digest = notice._sha256(lockfile)
    stamp = tmp_path / ".lock-sha256"
    stamp_digest = "0" * 64 if stale == "lockfile" else lock_digest
    stamp.write_bytes(f"{stamp_digest}  package-lock.json\n".encode("ascii"))
    image_id = "sha256:" + "3" * 64
    build_receipt = tmp_path / "BUILD-RECEIPT.json"
    build_receipt.write_text(
        json.dumps({
            "status": "PASSED",
            "target": "video-worker",
            "sourceHead": "c" * 40 if stale == "source" else "a" * 40,
            "sourceTree": "b" * 40,
            "imageId": image_id,
            "packageLockSha256": lock_digest,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(notice, "ROOT", root)
    monkeypatch.setattr(notice, "WORKER_LOCK_STAMP", stamp)
    monkeypatch.setattr(notice, "select_deal", lambda *_: selection)
    monkeypatch.setenv("BALOOT_RENDER_BUILD_RECEIPT", str(build_receipt))
    monkeypatch.setenv("BALOOT_RENDER_IMAGE_ID", "sha256:" + "4" * 64 if stale == "image" else image_id)
    output = tmp_path / "output"
    events = []
    monkeypatch.setattr(notice, "RuleBook", SimpleNamespace(load=lambda path: object()))
    monkeypatch.setattr(notice, "notice_profile", lambda *, public=False: events.append("design") or {"approved_design": design})

    bundle = SimpleNamespace(
        value={
            "timeline": {
                "segments": [
                    {"stage": stage, "durationFrames": 30, "startFrame": index * 30}
                    for index, stage in enumerate(notice.STAGES)
                ]
            },
            "render": {"fps": 30},
            "localizations": {"video.notice.sentence": "sentence"},
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
            self.renderer_root = renderer_root

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
                    "layoutGeometry": _fake_qualification(video, bundle.bundle_sha256, files, approved_layout_bytes),
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

    inputs = dict(
        output_dir=output,
        public=True,
        deal_id=selection["identifier"],
        source_head="a" * 40,
        source_tree="b" * 40,
    )
    if stale:
        with pytest.raises(ValueError, match=rf"public render (build )?{stale} .*\b(does|do) not match"):
            notice.main(**inputs)
        assert events == ["design"]
        assert not output.exists()
        return

    notice.main(**inputs)

    assert events == ["design", "build", "engine", "validation", "ffprobe", "provenance"]
    receipt = json.loads((output / "owner-approval.json").read_text(encoding="utf-8"))
    provenance_receipt = json.loads((output / "provenance-notice.json").read_text(encoding="utf-8"))
    for document in (receipt, provenance_receipt):
        assert document["imageId"] == image_id
        assert document["packageLockSha256"] == lock_digest
    assert receipt["dealIdentifier"] == selection["identifier"]
    assert receipt["fixture"] == selection["source"]
    assert receipt["publicResearchNoticeShown"] is True
    assert "title" not in receipt["headerText"]
    assert receipt["headerText"]["sentence"] == bundle.value["localizations"]["video.notice.sentence"]
    assert receipt["approvedDesignBinding"]["variant"] == "public_v4_no_justification_red_notice"
    assert receipt["layoutGeometry"]["tableSha256"] == notice._sha256(output / "layout-table.json")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    video = output / "clip.mp4"
    measurement = {**manifest, "layoutGeometry": {**manifest["layoutGeometry"], "mode": "measurement"}}
    with pytest.raises(ValueError, match="permanently NEVER_POST"):
        notice._release_layout(measurement, video, bundle.bundle_sha256)
    (output / "layout-table.json").write_bytes(b"{ }\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        notice._release_layout(manifest, video, bundle.bundle_sha256)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("widened-table", "approved layout table SHA-256"),
        ("table-file", "release layout evidence SHA-256 mismatch"),
        ("final-video", "bound to the exact MP4"),
        ("evidence-table", "checkpoint evidence does not match"),
        ("evidence-video", "checkpoint evidence does not match"),
    ],
)
def test_release_layout_rejects_mismatched_receipt_pins(tmp_path, approved_layout_bytes, change, error):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"qualified MP4")
    for phase in ("hook", "question", "reveal"):
        (tmp_path / f"{phase}.png").write_bytes(phase.encode("utf-8"))
    files = {}
    table_bytes = approved_layout_bytes
    if change == "widened-table":
        altered = json.loads(table_bytes)
        altered["rows"][0]["observedMaxWidth"] = 200
        table_bytes = json.dumps(altered).encode("utf-8")
    geometry = _fake_qualification(video, "bundle-sha", files, table_bytes)
    manifest = {"files": files, "layoutGeometry": geometry}
    if change != "widened-table":
        notice._release_layout(manifest, video, "bundle-sha")
    if change == "final-video":
        video.write_bytes(b"different final MP4")
    elif change == "table-file":
        path = tmp_path / files["layoutTable"]["name"]
        path.write_bytes(approved_layout_bytes + b"\n")
        files["layoutTable"]["sha256"] = notice._sha256(path)
    elif change.startswith("evidence-"):
        path = tmp_path / files["layoutGeometry"]["name"]
        evidence = json.loads(path.read_bytes())
        if change == "evidence-table":
            evidence["binding"]["tableSha256"] = "0" * 64
        else:
            evidence["videoSha256"] = "0" * 64
        path.write_text(json.dumps(evidence), encoding="utf-8")
        files["layoutGeometry"]["sha256"] = geometry["evidenceSha256"] = notice._sha256(path)

    with pytest.raises(ValueError, match=error):
        notice._release_layout(manifest, video, "bundle-sha")
