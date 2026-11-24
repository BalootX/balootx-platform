from datetime import UTC, datetime, timedelta
import hashlib
import json
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import render_kickoff, render_research_notice as notice


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


def fixture_git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@users.noreply.github.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, text=True, timeout=10,
    ).strip()


@pytest.fixture
def kickoff_clock(monkeypatch):
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    monkeypatch.setattr(render_kickoff, "utc_now", lambda: now)
    return now


@pytest.fixture
def kickoff_inputs(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    fixture_git(public, "init", "--initial-branch=main")
    layout = public / "scene/layout.txt"
    layout.parent.mkdir(parents=True)
    layout.write_text("before layout fix\n", encoding="utf-8")
    scripts = public / "scripts"
    scripts.mkdir()
    (scripts / "render_kickoff.py").write_bytes(
        (render_kickoff.TRUSTED_ROOT / "scripts/render_kickoff.py").read_bytes()
    )
    policy = json.loads(render_kickoff.POLICY.read_bytes())
    policy.update(publicRemote=public.as_uri(), maxReceiptAgeSeconds=60)
    (scripts / "render-kickoff.json").write_text(json.dumps(policy), encoding="utf-8")
    schema = json.loads(render_kickoff.POLICY_SCHEMA.read_bytes())
    schema["properties"]["publicRemote"]["pattern"] = "^file://"
    (scripts / "render-kickoff.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (public / ".gitignore").write_bytes((render_kickoff.TRUSTED_ROOT / ".gitignore").read_bytes())
    fixture_git(public, "add", ".")
    fixture_git(public, "commit", "-m", "4fd9b56a pre-fix renderer")
    checkout = tmp_path / "source"
    trusted = tmp_path / "kickoff"
    fixture_git(tmp_path, "clone", "--no-local", str(public), str(checkout))
    fixture_git(tmp_path, "clone", "--no-local", str(public), str(trusted))
    policy_path = trusted / "scripts/render-kickoff.json"
    monkeypatch.setattr(render_kickoff, "TRUSTED_ROOT", trusted)
    monkeypatch.setattr(render_kickoff, "POLICY", policy_path)
    monkeypatch.setattr(render_kickoff, "POLICY_SCHEMA", policy_path.with_suffix(".schema.json"))
    return public, checkout, layout, policy_path


def kickoff_cli_args(checkout, build, *, head=None, tree=None, image_id=None, build_edits=None):
    actual_head, actual_tree = fixture_git(checkout, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    head = actual_head if head is None else head
    tree = actual_tree if tree is None else tree
    image_id = image_id if image_id is not None else "sha256:" + "3" * 64
    build.write_text(json.dumps({
        "status": "PASSED", "target": "video-worker", "sourceHead": head,
        "sourceTree": tree, "imageId": image_id, "packageLockSha256": "4" * 64,
        **(build_edits or {}),
    }), encoding="utf-8")
    return [
        "--checkout", str(checkout), "--source-head", head, "--source-tree", tree,
        "--build-receipt", str(build), "--image-id", image_id,
    ]


@pytest.mark.parametrize("pin", [
    "4fd9b56a", "diverged", "diverged-equal", "current", "newer",
    "after-fix", "after-fix-newer", "reverted", "main-reverted",
])
def test_render_kickoff_checks_live_public_renderer_history(tmp_path, monkeypatch, capsys, kickoff_inputs, kickoff_clock, pin):
    public, checkout, layout, policy = kickoff_inputs
    before_fix = fixture_git(public, "rev-parse", "HEAD")
    layout.write_text("after layout fix\n", encoding="utf-8")
    fixture_git(public, "commit", "-am", "3e073db layout fix")
    fix = fixture_git(public, "rev-parse", "HEAD")
    if pin == "main-reverted":
        fixture_git(public, "revert", "--no-edit", fix)
    docs = public / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("review notes\n", encoding="utf-8")
    fixture_git(public, "add", "docs")
    fixture_git(public, "commit", "-m", "unrelated documentation change")
    main_head = fixture_git(public, "rev-parse", "HEAD")
    assert fixture_git(checkout, "rev-parse", "origin/main") == before_fix

    if pin in {"current", "newer", "after-fix", "after-fix-newer", "reverted"}:
        fixture_git(checkout, "fetch", "origin")
        fixture_git(checkout, "checkout", "--detach", fix if pin.startswith("after-fix") else "origin/main")
    if pin == "current":
        (checkout / "outputs").mkdir()
        (checkout / "outputs/render.json").write_bytes(b"{}")
    if pin in {"diverged", "newer", "after-fix-newer"}:
        fixture_git(checkout, "commit", "--allow-empty", "-m", "newer candidate change")
    if pin == "diverged-equal":
        (checkout / "scene/layout.txt").write_bytes(layout.read_bytes())
        fixture_git(checkout, "commit", "-am", "same layout on different history")
        assert fixture_git(checkout, "rev-parse", "HEAD^{tree}") == fixture_git(public, "rev-parse", f"{fix}^{{tree}}")
        assert fixture_git(checkout, "rev-parse", "HEAD^") == before_fix
    if pin == "reverted":
        fixture_git(checkout, "revert", "--no-edit", fix)
    if pin == "4fd9b56a":
        alternate = tmp_path / "alternate"
        fixture_git(tmp_path, "clone", "--branch=main", str(checkout), str(alternate))
        directive = f"url.{alternate.as_uri()}.insteadOf"
        trusted = policy.parent.parent
        fixture_git(trusted, "config", directive, public.as_uri())
        monkeypatch.setattr(render_kickoff, "TemporaryDirectory", lambda: TemporaryDirectory(dir=trusted))
        global_config = tmp_path / "global.cfg"
        fixture_git(checkout, "config", "--file", str(global_config), directive, public.as_uri())
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    args = kickoff_cli_args(checkout, tmp_path / "build.json")
    status = render_kickoff.main(args)
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["mainHead"] == main_head
    assert verdict["checkedAt"] == kickoff_clock.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert (verdict["sourceHead"], verdict["sourceTree"], verdict["imageId"]) == (args[3], args[5], args[9])
    if pin in {"4fd9b56a", "diverged", "diverged-equal"}:
        assert status == 1
        assert verdict["status"] == "failed"
        assert verdict["missingRendererCommit"] == fix
        assert "repin and rebuild" in verdict["reason"]
    elif pin == "reverted":
        assert status == 1
        assert verdict["status"] == "failed"
        assert verdict["missingRendererCommit"] is None
        assert "differs from public main" in verdict["reason"]
    else:
        assert status == 0
        assert verdict["status"] == "passed"
        assert verdict["missingRendererCommit"] is None
        assert verdict["reason"] is None


@pytest.mark.parametrize("path", [
    "src/baloot_platform/core/rules.py",
    "src/baloot_platform/research/exact_state.py",
    "configs/rules/saudi_common_v1.yaml",
    "examples/synthetic-clip-identity.json",
])
def test_render_kickoff_refuses_pins_missing_merged_inputs(tmp_path, capsys, kickoff_inputs, path):
    public, checkout, _, _ = kickoff_inputs
    changed = public / path
    changed.parent.mkdir(parents=True)
    changed.write_text("synthetic renderer input\n", encoding="utf-8")
    fixture_git(public, "add", path)
    fixture_git(public, "commit", "-m", "fix a renderer input")
    fix = fixture_git(public, "rev-parse", "HEAD")

    assert render_kickoff.main(kickoff_cli_args(checkout, tmp_path / "build.json")) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["mainHead"] == fix and verdict["missingRendererCommit"] == fix


@pytest.mark.parametrize("filename", ["render_kickoff.py", "render-kickoff.json", "render-kickoff.schema.json"])
def test_render_kickoff_requires_its_own_authority_from_main(tmp_path, capsys, kickoff_inputs, filename):
    _, checkout, _, policy = kickoff_inputs
    stale = policy.with_name(filename)
    stale.write_bytes(stale.read_bytes() + b"\n")

    assert render_kickoff.main(kickoff_cli_args(checkout, tmp_path / "build.json")) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["mainHead"] is not None
    assert f"scripts/{filename} differs from public main" in verdict["reason"]


@pytest.mark.parametrize(("argv", "code", "status"), [([], 2, "failed"), (["--help"], 0, "help")])
def test_render_kickoff_usage_writes_json(capsys, argv, code, status):
    assert render_kickoff.main(argv) == code
    captured = capsys.readouterr()
    verdict = json.loads(captured.out)
    assert verdict["status"] == status
    assert verdict["mainHead"] is None and verdict["checkedAt"] is None
    assert "usage:" in captured.err


@pytest.mark.parametrize(("case", "reason"), [
    ("head", "does not match the pinned source"), ("tree", "does not match the pinned source"),
    ("subdirectory", "does not match the pinned source"), ("dirty", "requires a clean checkout"),
    ("untracked", "requires a clean checkout"), ("shallow", "requires a complete Git history"),
    ("build", "requires a passed video-worker build"), ("build-source", "build source does not match"),
    ("policy", "excludedPaths"), ("remote", "git ls-remote"),
    ("ids", "requires full source commit and tree IDs"), ("unexpected", "unexpected render kickoff error"),
])
def test_render_kickoff_refuses_invalid_inputs(tmp_path, monkeypatch, capsys, kickoff_inputs, case, reason):
    public, checkout, _, policy_path = kickoff_inputs
    if case == "shallow":
        checkout = tmp_path / "shallow"
        fixture_git(tmp_path, "clone", "--depth=1", public.as_uri(), str(checkout))
    args = kickoff_cli_args(
        checkout, tmp_path / "build.json", head="1" * 40 if case == "head" else "bad" if case == "ids" else None,
        tree="2" * 40 if case == "tree" else "bad" if case == "ids" else None,
        image_id="bad" if case == "ids" else None,
        build_edits={"status": "FAILED"} if case == "build" else {"sourceHead": "0" * 40} if case == "build-source" else None,
    )
    if case == "subdirectory":
        args[1] = str(checkout / "scene")
    if case == "dirty":
        (checkout / "scene/layout.txt").write_bytes(b"dirty")
    if case == "untracked":
        (checkout / "yaml.py").write_bytes(b"untracked module")
    if case in {"policy", "remote"}:
        policy = json.loads(policy_path.read_bytes())
        if case == "policy":
            policy["excludedPaths"] = ["../scene"]
        else:
            policy["publicRemote"] = (tmp_path / "unreachable").as_uri()
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
    if case == "unexpected":
        def fail():
            raise RuntimeError("internal error")
        monkeypatch.setattr(render_kickoff, "load_policy", fail)

    assert render_kickoff.main(args) == 1
    captured = capsys.readouterr()
    verdict = json.loads(captured.out)
    assert verdict["status"] == "failed" and reason in verdict["reason"]
    assert verdict["mainHead"] is None and verdict["checkedAt"] is None
    if case == "ids":
        assert verdict["sourceHead"] is None and verdict["sourceTree"] is None and verdict["imageId"] is None
    if case == "unexpected":
        assert "Traceback" in captured.err and "RuntimeError: internal error" in captured.err


@pytest.mark.parametrize(("edits", "reason"), [
    ({"status": "failed"}, "requires a passed"),
    ({"missingRendererCommit": "1" * 40}, "requires a passed"),
    ({"reason": "refused"}, "requires a passed"),
    ({"sourceHead": "2" * 40}, "does not match"),
    ({"sourceTree": "3" * 40}, "does not match"),
    ({"packageLockSha256": "4" * 64}, "does not match"),
    ({"schemaVersion": True}, "requires a passed"),
    ({"schemaVersion": 1.0}, "requires a passed"),
    ({"extra": True}, "requires a passed"),
    ({"checkedAt": None}, "requires a UTC checkedAt"),
    ({"checkedAt": "2026-02-30T12:00:00Z"}, "requires a valid checkedAt"),
    ({"checkedAt": "2026-01-01T12:00:01Z"}, "future-dated"),
    ({"checkedAt": "2026-01-01T11:58:59Z"}, "has expired"),
    ([], "must be a JSON object"),
])
def test_public_renderer_refuses_invalid_kickoff_receipts(
    tmp_path, kickoff_inputs, kickoff_clock, edits, reason
):
    _, checkout, _, _ = kickoff_inputs
    source_head, source_tree = fixture_git(checkout, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    image_id = "sha256:" + "6" * 64
    lock_digest = "7" * 64
    verdict = {
        **render_kickoff.new_verdict(source_head, source_tree, image_id),
        "status": "passed", "mainHead": source_head, "packageLockSha256": lock_digest,
        "checkedAt": (kickoff_clock - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    receipt = tmp_path / "kickoff.json"
    receipt.write_text(json.dumps(verdict), encoding="utf-8")
    args = (receipt, source_head, source_tree, image_id, lock_digest)
    assert render_kickoff.require_kickoff_receipt(*args, checkout=checkout) == (source_head, verdict["checkedAt"])
    receipt.write_text(json.dumps({**verdict, **edits} if isinstance(edits, dict) else edits), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        render_kickoff.require_kickoff_receipt(*args, checkout=checkout)


@pytest.mark.parametrize("stale", [
    None, "image", "source", "kickoff receipt", "missing kickoff",
    "checkout", "lockfile", "dependencies", "dependency path",
])
def test_notice_selected_main_reaches_gates_with_selected_fixture(
    tmp_path, monkeypatch, approved_layout_bytes, kickoff_clock, stale
):
    selection = notice.select_deal(notice.DEAL_SOURCES[1][0])
    expected_trace = json.loads(selection["fixture"].read_text(encoding="utf-8"))["exactTracePrefix"]
    design = notice.notice_profile(public=True)["approved_design"]
    root = tmp_path / "source"
    renderer_root = root / "src/baloot_platform/video/remotion"
    worker = tmp_path / "worker"
    dependency = worker / "node_modules/renderer/package.json"
    dependency.parent.mkdir(parents=True)
    dependency.write_text(
        json.dumps({"version": "1.0.0" if stale == "dependencies" else "2.0.0"}),
        encoding="utf-8",
    )
    lockfile = renderer_root / "package-lock.json"
    lockfile.parent.mkdir(parents=True)
    lockfile.write_text(
        '{"lockfileVersion":3,"packages":{"node_modules/renderer":{"version":"2.0.0"}}}',
        encoding="utf-8",
    )
    (worker / "package-lock.json").write_bytes(lockfile.read_bytes())
    fixture_git(root, "init")
    fixture_git(root, "add", ".")
    fixture_git(root, "commit", "-m", "pinned producer checkout")
    source_head, source_tree = fixture_git(root, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    if stale == "checkout":
        fixture_git(root, "commit", "--allow-empty", "-m", "wrong producer checkout")
    lock_digest = notice._sha256(lockfile)
    stamp = worker / ".lock-sha256"
    stamp_digest = "0" * 64 if stale == "lockfile" else lock_digest
    stamp.write_bytes(f"{stamp_digest}  package-lock.json\n".encode("ascii"))
    image_id = "sha256:" + "3" * 64
    build_receipt = tmp_path / "BUILD-RECEIPT.json"
    build_receipt.write_text(
        json.dumps({
            "status": "PASSED",
            "target": "video-worker",
            "sourceHead": "c" * 40 if stale == "source" else source_head,
            "sourceTree": source_tree,
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
    kickoff = tmp_path / "kickoff.json"
    kickoff.write_text(json.dumps({
        "schemaVersion": 1, "status": "passed", "mainHead": "d" * 40,
        "sourceHead": source_head, "sourceTree": source_tree,
        "imageId": "sha256:" + "4" * 64 if stale == "kickoff receipt" else image_id,
        "checkedAt": kickoff_clock.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "packageLockSha256": lock_digest, "missingRendererCommit": None, "reason": None,
    }), encoding="utf-8")
    if stale == "missing kickoff":
        monkeypatch.delenv("BALOOT_RENDER_KICKOFF_RECEIPT", raising=False)
    else:
        monkeypatch.setenv("BALOOT_RENDER_KICKOFF_RECEIPT", str(kickoff))
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
        def __init__(self, *args, use_worker_renderer, **kwargs):
            assert use_worker_renderer
            self.renderer_root = renderer_root if stale == "dependency path" else worker

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
        source_head=source_head,
        source_tree=source_tree,
    )
    if stale:
        mismatch = {
            "missing kickoff": "requires a passed render kickoff receipt",
            "checkout": "checkout does not match the pinned source",
            "dependency path": "dependencies do not match",
        }.get(stale, rf"public render (build )?{stale} .*\b(does|do) not match")
        with pytest.raises(ValueError, match=mismatch):
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
        assert document["kickoffMainHead"] == "d" * 40
        assert document["checkedAt"] == kickoff_clock.strftime("%Y-%m-%dT%H:%M:%SZ")
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
