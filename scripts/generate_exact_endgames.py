#!/usr/bin/env python3
"""Generate frozen independent-oracle fixtures for exact endgames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baloot_platform.analysis.search import terminal_margin
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit, Team


OUTPUT = Path("tests/fixtures/synthetic-exact-endgames.json")


def prepare(engine: BalootEngine, case: dict):
    state = engine.create_round(
        seed=int(case["seed"]),
        dealer=Seat.parse(case["dealer"]),
    )
    engine.shuffle_and_initial_deal(
        state,
        cut_position=int(case["cutPosition"]),
    )
    contract = Contract(str(case["contract"]))
    engine.bid_and_complete_deal(
        state,
        forced_contract=contract,
        forced_buyer=Seat.parse(case["buyer"]),
        forced_trump=(
            Suit.parse(case["trumpSuit"])
            if case.get("trumpSuit") is not None
            else None
        ),
    )
    engine.prepare_projects(state)
    for action_id in case.get("actionPrefix", ()):
        engine.play_card(state, int(action_id))
    return state


def clone_oracle(state, engine: BalootEngine, team: Team) -> float:
    if state.current_player is None:
        return terminal_margin(state, team)
    values = []
    for card in engine.legal_actions(state):
        child = state.clone()
        engine.play_card(child, card)
        values.append(clone_oracle(child, engine, team))
    return (
        max(values)
        if state.current_player.team == team
        else min(values)
    )


def generate() -> dict:
    bundle = build_bundle(platform_root())
    engine = BalootEngine(bundle.rules)
    cases = []
    for index in range(12):
        contract = Contract.SUN if index % 2 == 0 else Contract.HOKUM
        source = {
            "id": f"exact-endgame-{index + 1:02d}",
            "seed": 61_000_000 + index,
            "dealer": Seat(index % 4).name.lower(),
            "cutPosition": (index * 7 % 31) + 1,
            "contract": contract.value,
            "buyer": Seat((index * 3 + 2) % 4).name.lower(),
            "trumpSuit": (
                Suit((index * 5 + 1) % 4).name.lower()
                if contract == Contract.HOKUM
                else None
            ),
            "actionPrefix": [],
        }
        state = prepare(engine, source)
        bot = HeuristicBot(
            engine,
            BotProfile(deterministic_seed=71_000_000 + index),
        )
        remaining_target = 4 + index % 2
        while (
            sum(mask.bit_count() for mask in state.hands)
            > remaining_target
        ):
            action = bot.choose(state)
            source["actionPrefix"].append(action.id)
            engine.play_card(state, action)
        source.update(
            {
                "remainingCards": remaining_target,
                "rootActions": [card.code for card in engine.legal_actions(state)],
                "observerTeam": "north_south",
                "stateSha256": engine.state_hash(state),
                "expectedMargin": clone_oracle(
                    state,
                    engine,
                    Team.NORTH_SOUTH,
                ),
                "oracle": "independent_clone_minimax_v1",
                "externallyCertified": False,
            }
        )
        cases.append(source)
    return {
        "schemaVersion": 1,
        "rulesSha256": bundle.rules.sha256,
        "provenance": {
            "kind": "independent_synthetic_software_oracle_fixture",
            "externalCertification": False,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = platform_root()
    output = root / OUTPUT
    text = json.dumps(
        generate(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != text:
            raise SystemExit("exact-endgame fixture corpus is stale")
        return
    output.write_text(text, encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
