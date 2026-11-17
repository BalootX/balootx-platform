#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate exact-state inputs for the professional Remotion approval gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.config import build_bundle
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.video.render_bundle import build_export_timeline, build_render_bundle
from baloot_platform.video.scene_adapter import build_legacy_episode_scene_package

SOURCE = ROOT / "examples/v7-outcome-flip-source.json"
PROFILE = ROOT / "configs/video/premium.yaml"
OUTPUT = ROOT / "outputs/professional-video-prototypes"
SAUDI_PLAYER_NAMES = ('Player 01', 'Player 02', 'Player 03', 'Player 04', 'Player 05', 'Player 06', 'Player 07', 'Player 08', 'Player 09', 'Player 10', 'Player 11', 'Player 12', 'Player 13', 'Player 14', 'Player 15', 'Player 16')
def _write(path: Path, value: dict[str, object]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def generate(
    output: Path = OUTPUT, *, source: Path | None = None, seed: int = 0,
) -> tuple[Path, Path]:
    bundle = build_bundle(ROOT)
    source = source or SOURCE
    episode = load_episode_spec(source, bundle.rules)
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    scenes = build_legacy_episode_scene_package(episode, profile)
    question_scene = next(scene for scene in scenes.scenes if scene.value["stage"] == "question")
    game_serial = str(episode.game_serial or episode.episode_id)
    identity = {
        "gameSerial": game_serial,
        "sceneId": question_scene.scene_id,
        "enrollmentUri": str(profile["identity"]["enrollment_uri_template"]).format(game_serial=game_serial),
        "players": dict(zip(
            ("north", "west", "east", "south"),
            random.Random(seed).sample(SAUDI_PLAYER_NAMES, 4), strict=True,
        )),
    }
    paths: list[Path] = []
    for preset in ("hooked_flagship", "decision_clip"):
        timeline = build_export_timeline(
            scenes=scenes.scenes, profile=profile, export_preset=preset,
        )
        render_bundle = build_render_bundle(
            scenes=scenes.scenes,
            timeline=timeline,
            profile=profile,
            export_preset=preset,
            localizations=profile["localizations"],
            identity=identity,
            repository_root=ROOT,
            decision_index=episode.decision_index,
        )
        path = output / f"render-bundle-{preset.removesuffix('_clip')}.json"
        _write(path, render_bundle.to_dict())
        paths.append(path)
    _write(output / "source.json", json.loads(source.read_text(encoding="utf-8")))
    _write(output / "source-manifest.json", {
        "schemaVersion": 1,
        "sourceSha256": hashlib.sha256((output / "source.json").read_bytes()).hexdigest(),
        "inputSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "configurationSha256": bundle.sha256,
        "identitySeed": seed,
        "players": identity["players"],
        "legalReplay": "passed",
        "sourcePlays": len(episode.final_state.played_cards),
        "pauseBeforePlayIndex": episode.pause_before_play_index,
        "roundScore": episode.final_state.round_score.to_dict(),
        "bundles": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    })
    return paths[0], paths[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in generate(args.output, source=args.source, seed=args.seed):
        print(path)


if __name__ == "__main__":
    main()
