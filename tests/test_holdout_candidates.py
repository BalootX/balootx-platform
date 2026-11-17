from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from baloot_platform.benchmark import benchmark_configuration_sha256
from baloot_platform.config import build_bundle

def _module(root: Path):
    path = root / "scripts/generate_holdout_candidates.py"
    spec = importlib.util.spec_from_file_location(
        "generate_holdout_candidates",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_candidates_are_deterministic_observer_safe_and_unlabelled(
    root: Path,
):
    module = _module(root)
    first = module.generate(per_archetype=2, root=root)
    second = module.generate(per_archetype=2, root=root)

    assert first == second
    assert first["engineVersion"] == module.__version__
    assert first["candidateCount"] == 18
    assert first["labelStatus"] == "pending_governed_adjudication"
    assert len(first["datasetSha256"]) == 64
    assert {
        item["archetype"] for item in first["positions"]
    } == set(module.ARCHETYPES)
    for item in first["positions"]:
        assert len(item["legalActions"]) >= 2
        assert item["labelStatus"] == "pending_governed_adjudication"
        assert "referenceAction" not in item
        assert "predictedAction" not in item
        assert "seed" not in item
        assert "seed" not in item["observerState"]
        assert "shuffle" not in item["observerState"]
        assert "dealTrace" not in item["observerState"]
        assert "events" not in item["observerState"]
        observer = item["observer"]
        hands = item["observerState"]["hands"]
        assert isinstance(hands[observer], list)
        for seat, hand in hands.items():
            if seat != observer:
                assert set(hand) == {"count"}


def test_trusted_replay_recipes_are_separate_from_public_candidates(
    root: Path,
):
    module = _module(root)
    public, trusted = module.generate_artifacts(
        per_archetype=1,
        root=root,
    )

    assert trusted["recipeCount"] == public["candidateCount"] == 9
    assert trusted["datasetSha256"] == public["datasetSha256"]
    assert len(trusted["sidecarSha256"]) == 64
    for item in public["positions"]:
        recipe = trusted["recipes"][item["candidateId"]]
        assert "seed" in recipe
        assert "seed" not in item


def test_strength_configuration_ignores_video_only_changes(
    root: Path,
):
    bundle = build_bundle(root)
    altered_video = replace(
        bundle.components["video"],
        sha256="f" * 64,
    )
    altered = replace(
        bundle,
        components={
            **bundle.components,
            "video": altered_video,
        },
        sha256="e" * 64,
    )

    assert (
        benchmark_configuration_sha256(bundle)
        == benchmark_configuration_sha256(altered)
    )


def test_checked_in_synthetic_fixture_is_complete_and_reproducible(
    root: Path,
    monkeypatch,
):
    module = _module(root)
    stored = module.load_public(
        root / "tests/fixtures/synthetic-candidates.jsonl"
    )
    # Regenerate the independent small synthetic fixture; this is not holdout qualification.
    monkeypatch.setattr(module, "__version__", stored["engineVersion"])
    generated = module.generate(per_archetype=1, root=root)

    assert stored == generated
    assert stored["purpose"] == "independent_synthetic_test_fixture_not_holdout_evidence"
    assert stored["candidateCount"] == 9
    assert stored["perArchetype"] == 1
    assert len(
        {item["candidateId"] for item in stored["positions"]}
    ) == 9
    assert len(
        {item["stateSha256"] for item in stored["positions"]}
    ) == 9
    assert len(
        {item["dealHash"] for item in stored["positions"]}
    ) == 9
