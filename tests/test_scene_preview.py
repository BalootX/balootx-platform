from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_scene_preview import build


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    flagship_dir = tmp_path / "professional-video-flagship"
    decision_dir = tmp_path / "professional-video-decision"
    prototype_dir = tmp_path / "professional-video-prototypes"
    _write(flagship_dir / "manifest.json", {"schemaVersion": 1})
    _write(decision_dir / "manifest.json", {"schemaVersion": 1})
    (flagship_dir / "balootx-complete-laffah-audit.mp4").write_bytes(b"flagship")
    (decision_dir / "balootx-decision-clip.mp4").write_bytes(b"decision")
    common = {
        "render": {"width": 1080, "height": 1920, "fps": 60},
        "identity": {"sceneId": "scene-question"},
        "scenes": [{
            "stage": "question",
            "layers": [
                {"kind": "table", "disclosure": "public"},
                {"kind": "cards", "disclosure": "observer_private"},
            ],
        }],
    }
    _write(
        prototype_dir / "render-bundle-hooked_flagship.json",
        {
            **common,
            "exportPreset": "hooked_flagship",
            "timeline": {"segments": [{"stage": "hook", "startFrame": 0}]},
        },
    )
    _write(
        prototype_dir / "render-bundle-decision.json",
        {
            **common,
            "exportPreset": "decision_clip",
            "timeline": {"segments": [{"stage": "hook", "startFrame": 0}]},
        },
    )
    return tmp_path


def test_static_preview_is_observer_safe_and_has_no_domain_authority(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    output = tmp_path / "preview"
    build(source, output)
    data_text = (output / "qa-data.js").read_text(encoding="utf-8")
    data = json.loads(data_text.removeprefix("window.BALOOTX_QA=").removesuffix(";\n"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    profile = yaml.safe_load((ROOT / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    assert data["domainCalculations"] is False
    assert data["copy"] == profile["preview_copy"]
    assert len(manifest["copySha256"]) == 64
    assert all(item["observerSafe"] for item in data["artifacts"].values())
    assert data["artifacts"]["flagship"]["width"] * 16 == data["artifacts"]["flagship"]["height"] * 9
    assert data["artifacts"]["hitl"]["width"] * 16 == data["artifacts"]["hitl"]["height"] * 9
    assert manifest["domainCalculations"] is False
    assert manifest["observerSafe"] is True




def test_preview_refresh_is_bounded_and_checkable(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    output = tmp_path / "preview"
    build(source, output)
    build(source, output, refresh=True)
    build(source, output, check=True)
    (output / "unexpected.txt").write_text("not managed", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        build(source, output, refresh=True)


def test_preview_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(_fixture(tmp_path), tmp_path / "preview", check=True, refresh=True)


def test_preview_is_keyboard_operable_and_contains_required_controls() -> None:
    html = (ROOT / "src/baloot_platform/service/static/scene-preview/index.html").read_text(encoding="utf-8")
    script = (ROOT / "src/baloot_platform/service/static/scene-preview/app.js").read_text(encoding="utf-8")
    assert 'id="artifact"' in html
    assert 'id="scene"' in html
    assert 'id="safeArea"' in html
    assert 'id="disclosure"' in html
    assert 'addEventListener("keydown"' in script
    assert "ArrowRight" in script and "ArrowLeft" in script
    for forbidden in ("legal_actions(", "apply(", "terminal_result(", "score("):
        assert forbidden not in script


def test_preview_rejects_non_vertical_release_media(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    path = source / "professional-video-prototypes/render-bundle-hooked_flagship.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["render"]["width"] = 1920
    manifest["render"]["height"] = 1080
    _write(path, manifest)
    with pytest.raises(ValueError, match="1080x1920"):
        build(source, tmp_path / "preview")
