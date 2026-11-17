#!/usr/bin/env python3
"""Deterministic release property and native-parity checks.

This is intentionally runnable outside pytest so deployment pipelines can use it
as a bounded conformance gate. It exercises the complete round lifecycle across
Sun, Hokum, and Ashkal and compares every playable position with the optional
native hot-path accelerator.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.validation import validate_completed_round
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable


def run(
    rounds: int,
    native_library: str | None = None,
    *,
    minimum_native_positions: int = 0,
) -> dict:
    root = platform_root()
    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    native = None
    if native_library:
        native = NativeAccelerator(bundle.rules, native_library)
    else:
        try:
            native = NativeAccelerator(bundle.rules)
        except (NativeUnavailable, OSError):
            pass

    started = time.perf_counter()
    contracts = (Contract.SUN, Contract.HOKUM, Contract.ASHKAL)
    completed = 0
    plays = 0
    native_compared = 0
    project_rounds = 0
    contract_counts = {item.value: 0 for item in contracts}
    score_totals = [0, 0]

    for index in range(rounds):
        contract = contracts[index % len(contracts)]
        dealer = Seat(index % 4)
        if contract == Contract.ASHKAL:
            order = [dealer.next(), dealer.next().next(), dealer.next().next().next(), dealer]
            bidder = order[2 + (index % 2)]
            buyer = bidder.partner
        else:
            buyer = Seat((index * 3 + 2) % 4)
        trump = Suit((index * 5 + 1) % 4) if contract == Contract.HOKUM else None
        state = engine.create_round(seed=10_000 + index, dealer=dealer, round_number=index + 1, match_score=tuple(score_totals))
        engine.shuffle_and_initial_deal(state, cut_position=(index * 7 % 31) + 1)
        accepted = engine.bid_and_complete_deal(
            state,
            forced_contract=contract,
            forced_buyer=buyer,
            forced_trump=trump,
        )
        if not accepted:
            raise AssertionError("forced contract unexpectedly passed")
        engine.prepare_projects(state)
        if state.projects:
            project_rounds += 1
        bot = HeuristicBot(engine, BotProfile(90_000 + index))
        while state.current_player is not None:
            legal = engine.legal_moves(state)
            if not legal:
                raise AssertionError("nonterminal state has no legal move")
            if native is not None:
                native_legal = native.legal_moves(state)
                if set(native_legal) != set(legal):
                    raise AssertionError(
                        f"native parity mismatch at round {index}: "
                        f"python={[c.code for c in legal]} native={[c.code for c in native_legal]}"
                    )
                native_compared += 1
            before = sum(hand.bit_count() for hand in state.hands)
            card = bot.choose(state)
            if card not in legal:
                raise AssertionError("bot selected an illegal card")
            engine.play_card(state, card)
            after = sum(hand.bit_count() for hand in state.hands)
            if after != before - 1:
                raise AssertionError("playing a card did not reduce remaining hand count by one")
            plays += 1
        errors = validate_completed_round(state)
        if errors:
            raise AssertionError(f"round {index} failed validation: {errors}")
        if state.round_score is None:
            raise AssertionError("round score missing")
        score_totals = list(state.match_score)
        completed += 1
        contract_counts[contract.value] += 1

    elapsed = time.perf_counter() - started
    result = {
        "schemaVersion": 1,
        "rounds": completed,
        "plays": plays,
        "contracts": contract_counts,
        "projectRounds": project_rounds,
        "nativeEnabled": native is not None,
        "nativePositionsCompared": native_compared,
        "elapsedSeconds": round(elapsed, 6),
        "roundsPerSecond": round(completed / elapsed, 2) if elapsed else None,
        "finalRollingScore": score_totals,
        "bundleSha256": bundle.sha256,
    }
    if native_compared < int(minimum_native_positions):
        raise AssertionError(
            "native parity position gate failed: "
            f"{native_compared} < {int(minimum_native_positions)}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--native-library")
    parser.add_argument("--output")
    parser.add_argument("--minimum-native-positions", type=int, default=0)
    args = parser.parse_args()
    result = run(
        args.rounds,
        args.native_library,
        minimum_native_positions=args.minimum_native_positions,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
