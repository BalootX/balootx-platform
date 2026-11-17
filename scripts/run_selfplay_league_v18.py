#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baloot_platform.config import build_bundle
from baloot_platform.ml.actions import ACTION_COUNT
from baloot_platform.ml.league_v18 import LeagueAgent, LeagueMatch, SelfPlayLeague
from baloot_platform.ml.models import ModelOutput
from baloot_platform.ml.selfplay import play_round


class LowestLegalModel:
    id = "lowest_legal_baseline_v18"
    version = "18.0.0"
    def infer(self, observation, legal_mask):
        logits = np.full(ACTION_COUNT, -1e9, dtype=np.float32)
        legal = [i for i, value in enumerate(legal_mask) if value]
        logits[min(legal)] = 1.0
        return ModelOutput(logits, 0.0)


class HighestLegalModel:
    id = "highest_legal_baseline_v18"
    version = "18.0.0"
    def infer(self, observation, legal_mask):
        logits = np.full(ACTION_COUNT, -1e9, dtype=np.float32)
        legal = [i for i, value in enumerate(legal_mask) if value]
        logits[max(legal)] = 1.0
        return ModelOutput(logits, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=16)
    parser.add_argument("--output", default="release/selfplay-league-v18.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rules = build_bundle(root).rules
    models = (LowestLegalModel(), HighestLegalModel())
    scores: list[float] = []
    decisions = 0
    board_rows = []
    for seed in range(17_000, 17_000 + max(2, args.boards)):
        results = [play_round(rules, seed=seed, model=model) for model in models]
        margins = [result.final_match_score[0] - result.final_match_score[1] for result in results]
        decisions += sum(result.steps for result in results)
        scores.append(1.0 if margins[0] > margins[1] else 0.0 if margins[0] < margins[1] else 0.5)
        board_rows.append({"seed": seed, "margins": margins, "steps": [item.steps for item in results]})
    league = SelfPlayLeague()
    agents = [
        LeagueAgent.create(model_id=model.id, version=model.version, artifact_hash=model.id.encode().hex().ljust(64, "0")[:64])
        for model in models
    ]
    for agent in agents:
        league.register(agent)
    score_a = sum(scores) / len(scores)
    league.record(LeagueMatch.create(
        board_set_hash="v18-identical-board-set-" + str(args.boards),
        agent_a=agents[0].agent_id, agent_b=agents[1].agent_id,
        score_a=score_a, score_b=1.0 - score_a, decisions=decisions,
        metadata={"identicalBoardSet": True, "boards": len(board_rows)},
    ))
    payload = {
        "schemaVersion": 1,
        "status": "passed",
        "strengthClaim": "baseline_pipeline_validation_only",
        "boards": board_rows,
        "league": league.snapshot(),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
