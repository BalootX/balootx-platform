from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from baloot_platform.benchmark import canonical_sha256


@pytest.mark.parametrize(
    "script_name",
    (
        "assemble_agent_qualification.py",
        "benchmark_agent_symmetry.py",
        "benchmark_paired_games.py",
        "benchmark_permanent_holdout.py",
        "engine_conformance.py",
        "generate_holdout_candidates.py",
        "generate_rules_candidate.py",
        "holdout_review.py",
        "label_exact_holdout.py",
        "run_local_finalist_preanalysis.py",
    ),
)
def test_release_review_commands_support_clean_source_help(
    root: Path,
    script_name: str,
):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, "-B", str(root / "scripts" / script_name), "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def _module(root: Path):
    path = root / "scripts/holdout_review.py"
    spec = importlib.util.spec_from_file_location(
        "holdout_review",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidates() -> dict:
    positions = [
        {
            "candidateId": f"candidate-{index}",
            "archetype": "endgame",
            "legalActions": ["A", "B"],
        }
        for index in range(2)
    ]
    return {
        "datasetId": "candidate-test",
        "datasetSha256": canonical_sha256(positions),
        "candidateCount": len(positions),
        "positions": positions,
    }


def test_review_template_is_blind_and_seals_only_complete_legal_answers(
    root: Path,
):
    module = _module(root)
    candidates = _candidates()
    draft = module.prepare_template(
        candidates,
        reviewer_id="reviewer-a",
        authority_id="authority-a",
        attestation="I reviewed independently.",
    )

    assert draft["instructions"]["blindReview"] is True
    assert draft["instructions"]["hiddenCardsAllowed"] is False
    assert draft["instructions"]["engineRecommendationAllowed"] is False
    assert draft["submissionHash"] is None
    with pytest.raises(ValueError, match="action must be legal"):
        module.seal_submission(candidates, draft)

    for item in draft["labels"]:
        item["action"] = "A"
        item["confidence"] = 0.8
    with pytest.raises(ValueError, match="regrets must be numeric"):
        module.seal_submission(candidates, draft)

    for item in draft["labels"]:
        item["actionRegrets"] = {"A": 0.0, "B": 1.0}
    sealed = module.seal_submission(candidates, draft)

    assert len(sealed["submissionHash"]) == 64
    assert all(
        set(item) == {
            "candidateId",
            "action",
            "confidence",
            "actionRegrets",
        }
        for item in sealed["labels"]
    )
    assert "instructions" not in sealed

    tampered = module.prepare_template(
        candidates,
        reviewer_id="reviewer-a",
        authority_id="authority-a",
        attestation="I reviewed independently.",
    )
    tampered["reviewer"]["independenceAttestation"] = "changed"
    for item in tampered["labels"]:
        item["action"] = "A"
        item["confidence"] = 0.8
        item["actionRegrets"] = {"A": 0.0, "B": 1.0}
    with pytest.raises(ValueError, match="attestation hash mismatch"):
        module.seal_submission(candidates, tampered)


def test_candidate_jsonl_loader_rejects_tampering(root: Path, tmp_path: Path):
    module = _module(root)
    candidates = _candidates()
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        module.serialize_holdout(candidates),
        encoding="utf-8",
    )
    loaded = module.load_candidate_jsonl(path)
    assert loaded == candidates

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace('"A"', '"X"', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.load_candidate_jsonl(path)


def test_reviewer_package_is_deterministic_blind_and_self_describing(
    root: Path, tmp_path: Path,
):
    module = _module(root)
    candidates = _candidates()
    kwargs = {
        "reviewer_id": "reviewer-a",
        "authority_id": "authority-a",
        "attestation": "I reviewed independently.",
    }
    left = tmp_path / "left.zip"
    right = tmp_path / "right.zip"
    first = module.build_reviewer_package(candidates, output=left, **kwargs)
    second = module.build_reviewer_package(candidates, output=right, **kwargs)

    assert left.read_bytes() == right.read_bytes()
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(left) as archive:
        assert archive.namelist() == [
            "README.md",
            "candidates.jsonl",
            "manifest.json",
            "review-template.json",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        draft = json.loads(archive.read("review-template.json"))
        combined = b"\n".join(archive.read(name) for name in archive.namelist())
    assert manifest["candidateCount"] == 2
    assert manifest["reviewer"]["id"] == "reviewer-a"
    assert all(item["action"] is None for item in draft["labels"])
    for forbidden in (
        b"trusted-replay-recipes",
        b"exact-labels",
        b"referenceAction",
        b"recommendedAction",
    ):
        assert forbidden not in combined
