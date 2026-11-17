from __future__ import annotations

from pathlib import Path

from baloot_platform.config import build_bundle
from scripts.generate_platform_evidence import configuration_evidence


def test_current_release_scripts_exist(root: Path):
    for name in (
        "package_release.py",
        "verify_release.py",
        "verify_archive.py",
        "render_release.py",
    ):
        assert (root / "scripts" / name).is_file()


def test_v18_active_release_metadata_has_no_v17_version_drift(root: Path):
    active = [root / "pyproject.toml", root / "src/baloot_platform/version.py", root / "configs/bundle.yaml", root / "sdk/typescript/package.json", root / "native/CMakeLists.txt"]
    joined = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in active)
    assert "17.0.0" not in joined
    assert "21.1.0" in joined


def test_configuration_evidence_tracks_the_packaged_bundle_version(root: Path):
    evidence = configuration_evidence()

    assert evidence["status"] == "passed"
    assert evidence["resolvedBundle"]["version"] == build_bundle(root).to_dict()[
        "profile"
    ]["version"]


def test_hitl_privacy_contract_and_no_automatic_mutation(root: Path):
    service = (root / "src/baloot_platform/hitl/service.py").read_text(encoding="utf-8")
    renderer = (root / "src/baloot_platform/video/hitl_review_v18.py").read_text(encoding="utf-8")
    assert "automaticProductionPromotion=False" in service
    assert "automaticProductionApply=False" in service
    assert '"productionRecommendationBeforeSubmission": False' in renderer
    assert '"hiddenHandsIncluded": False' in renderer
    assert "private_information_reference" in service
