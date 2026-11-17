from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module(root: Path, relative: str, name: str):
    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_holdout_labels_replay_and_preserve_all_optimal_ties(
    root: Path,
):
    generator = _module(
        root,
        "scripts/generate_holdout_candidates.py",
        "generate_holdout_candidates_for_exact_labels",
    )
    labeler = _module(
        root,
        "scripts/label_exact_holdout.py",
        "label_exact_holdout",
    )
    candidates, trusted = generator.generate_artifacts(
        per_archetype=1,
        root=root,
    )
    endgame = next(
        item
        for item in candidates["positions"]
        if item["archetype"] == "endgame"
    )

    first = labeler.label_exact_positions(
        candidates,
        trusted,
        root=root,
        candidate_ids={endgame["candidateId"]},
    )
    second = labeler.label_exact_positions(
        candidates,
        trusted,
        root=root,
        candidate_ids={endgame["candidateId"]},
    )

    assert first == second
    assert first["exactLabelCount"] == 1
    assert first["positions"][0]["searchCompletion"] == "exact"
    assert (
        first["positions"][0]["informationSet"]["method"]
        == "exhaustive_constrained_hidden_deals"
    )
    assert set(first["positions"][0]["referenceActions"]).issubset(
        endgame["legalActions"]
    )
    best = max(
        item["expectedTeamMargin"]
        for item in first["positions"][0]["actionValues"]
    )
    assert first["positions"][0]["referenceActions"] == sorted(
        item["action"]
        for item in first["positions"][0]["actionValues"]
        if item["expectedTeamMargin"] == best
    )
    assert len(first["evidenceSha256"]) == 64


def test_exact_holdout_labels_reject_tampered_trusted_sidecar(root: Path):
    generator = _module(
        root,
        "scripts/generate_holdout_candidates.py",
        "generate_holdout_candidates_for_tamper",
    )
    labeler = _module(
        root,
        "scripts/label_exact_holdout.py",
        "label_exact_holdout_for_tamper",
    )
    candidates, trusted = generator.generate_artifacts(
        per_archetype=1,
        root=root,
    )
    trusted["recipes"][
        next(iter(trusted["recipes"]))
    ]["seed"] += 1

    try:
        labeler.label_exact_positions(
            candidates,
            trusted,
            root=root,
        )
    except ValueError as exc:
        assert "sidecar hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered trusted sidecar was accepted")


