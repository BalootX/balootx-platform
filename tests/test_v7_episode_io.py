from __future__ import annotations

import json
from pathlib import Path

from baloot_platform.core.episode_io import load_episode_spec


def test_v7_episode_can_be_reopened_and_legally_replayed(root: Path, rules) -> None:
    episode = load_episode_spec(root / "examples/v7-outcome-flip-source.json", rules)
    assert episode.pause_before_play_index == 26
    assert len(episode.final_state.completed_tricks) == 8
    assert len(episode.final_state.played_cards) == 32
    assert episode.state_before_pause.current_player.name.lower() == "south"
    assert {action.card.code for action in episode.analysis.actions} == {"D10", "CA"}
    assert episode.analysis.actions[0].match_win_probability == 1.0
    assert episode.analysis.actions[1].match_win_probability == 0.0
    assert episode.final_state.round_score is not None
    assert episode.final_state.match_score == [158, 140]


def test_v7_episode_import_rejects_tampered_play(root: Path, rules, tmp_path: Path) -> None:
    source = json.loads((root / "examples/v7-outcome-flip-source.json").read_text(encoding="utf-8"))
    source["completedTricks"][0]["plays"][0]["card"] = source["completedTricks"][0]["plays"][1]["card"]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    try:
        load_episode_spec(path, rules)
    except (ValueError, AssertionError):
        return
    raise AssertionError("tampered EpisodeSpec was accepted")
