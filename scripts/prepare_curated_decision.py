#!/usr/bin/env python3
# ruff: noqa: E402
"""Freeze one new seeded legal endgame before the separately admitted render."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from baloot_platform.config import build_bundle
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.history.provenance_v4 import canonical_hash, game_serial

try:
    from .generate_professional_video_prototypes import generate, _write
except ImportError:
    from generate_professional_video_prototypes import generate, _write


def prepare(output: Path, seed: int) -> Path:
    configuration = build_bundle(ROOT)
    overrides = {
        "seed": seed, "match_score": [140, 140], "require_project": False,
        "pause_trick_min": 5, "pause_trick_max": 6,
        "requested_samples": 16, "exact_allocation_limit": 2000,
        "exact_search_max_remaining": 4, "maximum_sampling_attempts": 4000,
        "exact_search_max_nodes": 20000, "exact_search_max_seconds": 2.0,
        "max_seed_attempts": 64, "max_candidates_per_seed": 2,
        "minimum_hidden_deals": 1, "minimum_expected_margin_gap": 3.0,
        "minimum_win_probability_gap": 0.1,
    }
    request = ScenarioRequest.governed(
        yaml.safe_load((ROOT / "configs/challenge/progressive_mining_v1.yaml").read_text(encoding="utf-8")),
        **overrides,
    )
    deadline = time.monotonic() + 90
    episode = generate_episode(
        configuration.rules, configuration.convention,
        bundle_hash=configuration.sha256, request=request,
        copy=configuration.video["localizations"],
        cancel_check=lambda: time.monotonic() >= deadline,
    )
    game_hash = canonical_hash(episode.final_state.to_dict(reveal_hands=True))
    decision_id = "DEC-" + canonical_hash({
        "game": game_hash, "pause": episode.pause_before_play_index,
    })[:20].upper()
    episode = replace(
        episode, schema_version=7, game_serial=game_serial(game_hash),
        game_record_hash=game_hash, decision_id=decision_id,
        decision_index=tuple((stage, decision_id) for stage in ("question", "reveal", "continuation")),
    )
    source = output / "curated-source.json"
    _write(source, episode.to_dict())
    replayed = load_episode_spec(source, configuration.rules)
    if (
        [(p.seat, p.card) for p in replayed.final_state.played_cards]
        != [(p.seat, p.card) for p in episode.final_state.played_cards]
        or replayed.final_state.round_score.awarded_points != episode.final_state.round_score.awarded_points
        or replayed.final_state.match_score != episode.final_state.match_score
    ):
        raise ValueError("curated source differs from authoritative legal replay")
    _, path = generate(output, source=source, seed=seed)
    _write(output / "curation.json", {
        "schemaVersion": 1, "requestOverrides": overrides,
        "actualDealSeed": episode.final_state.seed,
        "configurationSha256": configuration.sha256,
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gameRecordHash": game_hash, "legalReplay": "passed",
        "analysis": episode.analysis.to_dict(),
        "story": "A late legal choice in a 140–140 match with a close round finish",
        "qualification": "candidate_for_media_and_editorial_review",
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.seed))


if __name__ == "__main__":
    main()
