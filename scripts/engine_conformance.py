#!/usr/bin/env python3
# ruff: noqa: E402
"""Deterministic rule-kernel conformance evidence.

Routine tests use an explicitly small sample. Stable-release verification reads
the governed minimums from the active certification profile and must provide
the native accelerator when native parity is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.validation import validate_completed_round
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable


def _prepared_round(engine: BalootEngine, index: int):
    contracts = (Contract.SUN, Contract.HOKUM, Contract.ASHKAL)
    contract = contracts[index % len(contracts)]
    dealer = Seat(index % 4)
    if contract == Contract.ASHKAL:
        order = [
            dealer.next(),
            dealer.next().next(),
            dealer.next().next().next(),
            dealer,
        ]
        bidder = order[2 + (index % 2)]
        buyer = bidder.partner
    else:
        buyer = Seat((index * 3 + 2) % 4)
    trump = Suit((index * 5 + 1) % 4) if contract == Contract.HOKUM else None
    state = engine.create_round(
        seed=30_000 + index,
        dealer=dealer,
        round_number=index + 1,
    )
    engine.shuffle_and_initial_deal(
        state,
        cut_position=(index * 7 % 31) + 1,
    )
    accepted = engine.bid_and_complete_deal(
        state,
        forced_contract=contract,
        forced_buyer=buyer,
        forced_trump=trump,
    )
    if not accepted:
        raise AssertionError("forced contract unexpectedly passed")
    engine.prepare_projects(state)
    return state


def _invalid_action(index: int, engine: BalootEngine, state) -> Any:
    category = index % 4
    if category == 0:
        return -1 - index
    if category == 1:
        return 32 + index
    if category == 2:
        return f"invalid-card-{index}"
    legal = {card.id for card in engine.legal_actions(state)}
    return next(card_id for card_id in range(32) if card_id not in legal)


def _load_gates(bundle) -> dict[str, Any]:
    certification = bundle.components["certification"].data
    gates = dict(certification.get("software_conformance", {}))
    required = {
        "minimum_executable_fixtures",
        "minimum_seeded_replay_rounds",
        "minimum_python_native_positions",
        "minimum_invalid_input_fuzz_cases",
        "require_apply_undo_byte_identity",
        "require_observer_noninterference",
    }
    missing = required - set(gates)
    if missing:
        raise ValueError(f"active certification profile lacks gates: {sorted(missing)}")
    return gates


def run(
    *,
    rounds: int,
    invalid_cases: int,
    native_library: str | None = None,
    enforce_profile: bool = False,
) -> dict[str, Any]:
    if rounds <= 0 or invalid_cases <= 0:
        raise ValueError("rounds and invalid_cases must be positive")

    bundle = build_bundle(platform_root())
    gates = _load_gates(bundle)
    if enforce_profile:
        if rounds < int(gates["minimum_seeded_replay_rounds"]):
            raise ValueError("round count is below the governed conformance minimum")
        if invalid_cases < int(gates["minimum_invalid_input_fuzz_cases"]):
            raise ValueError("invalid-case count is below the governed conformance minimum")

    engine = BalootEngine(bundle.rules)
    native = None
    if native_library:
        native = NativeAccelerator(bundle.rules, native_library)
    else:
        try:
            native = NativeAccelerator(bundle.rules)
        except (NativeUnavailable, OSError):
            pass

    native_positions = 0
    plays = 0
    replay_rounds = 0
    undo_checks = 0
    observer_checks = 0
    invalid_rejections = 0
    invalid_categories: dict[str, int] = {}
    final_hashes: list[str] = []

    for index in range(rounds):
        state = _prepared_round(engine, index)
        initial_hash = engine.state_hash(state)

        for seat in Seat:
            view = engine.observer_view(state, seat)
            if len(view["hands"][seat.name.lower()]) != 8:
                raise AssertionError("observer hand projection is incomplete")
            if any(
                view["hands"][other.name.lower()] != {"count": 8}
                for other in Seat
                if other != seat
            ):
                raise AssertionError("observer projection disclosed a hidden hand")
            observer_checks += 1

        if index < invalid_cases:
            invalid = _invalid_action(index, engine, state)
            before = engine.canonical_state(state)
            category = ("negative", "overflow", "malformed", "not_legal")[index % 4]
            try:
                engine.apply(state, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid action accepted: {invalid!r}")
            if engine.canonical_state(state) != before:
                raise AssertionError("rejected invalid action mutated state")
            invalid_rejections += 1
            invalid_categories[category] = invalid_categories.get(category, 0) + 1

        first = engine.legal_actions(state)[0]
        before = engine.canonical_state(state)
        token = engine.apply(state, first)
        engine.undo(state, token)
        if engine.canonical_state(state) != before:
            raise AssertionError("apply/undo did not restore byte-identical state")
        undo_checks += 1

        bot = HeuristicBot(engine, BotProfile(130_000 + index))
        actions: list[int] = []
        while state.current_player is not None:
            legal = engine.legal_actions(state)
            if native is not None:
                native_legal = tuple(
                    sorted(native.legal_moves(state), key=lambda card: card.id)
                )
                if native_legal != legal:
                    raise AssertionError(
                        f"native parity mismatch at round {index}, play {len(actions)}"
                    )
                native_positions += 1
            action = bot.choose(state)
            if action not in legal:
                raise AssertionError("deterministic bot selected an illegal action")
            engine.play_card(state, action)
            state.validate_conservation()
            actions.append(action.id)
            plays += 1

        errors = validate_completed_round(state)
        if errors:
            raise AssertionError(f"round {index} failed validation: {errors}")

        replay = _prepared_round(engine, index)
        if engine.state_hash(replay) != initial_hash:
            raise AssertionError("replay initial state identity mismatch")
        for action_id in actions:
            engine.apply(replay, action_id)
        if engine.canonical_state(replay) != engine.canonical_state(state):
            raise AssertionError(f"replay identity mismatch at round {index}")
        replay_rounds += 1
        final_hashes.append(engine.state_hash(state))

    while invalid_rejections < invalid_cases:
        index = invalid_rejections
        state = _prepared_round(engine, index % rounds)
        invalid = _invalid_action(index, engine, state)
        before = engine.canonical_state(state)
        category = ("negative", "overflow", "malformed", "not_legal")[index % 4]
        try:
            engine.apply(state, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid action accepted: {invalid!r}")
        if engine.canonical_state(state) != before:
            raise AssertionError("rejected invalid action mutated state")
        invalid_rejections += 1
        invalid_categories[category] = invalid_categories.get(category, 0) + 1

    required_native = (
        int(gates["minimum_python_native_positions"]) if enforce_profile else 0
    )
    if native_positions < required_native:
        raise AssertionError(
            f"native parity position gate failed: {native_positions} < {required_native}"
        )

    semantic = {
        "schemaVersion": 1,
        "bundleSha256": bundle.sha256,
        "rulesSha256": bundle.rules.sha256,
        "certificationProfile": bundle.components["certification"].data["profile"],
        "profileEnforced": bool(enforce_profile),
        "rounds": rounds,
        "plays": plays,
        "replayRounds": replay_rounds,
        "nativeEnabled": native is not None,
        "nativePositionsCompared": native_positions,
        "applyUndoByteIdentityChecks": undo_checks,
        "observerBoundaryChecks": observer_checks,
        "invalidCasesRejected": invalid_rejections,
        "invalidCategories": dict(sorted(invalid_categories.items())),
        "finalStateSetSha256": hashlib.sha256(
            "\n".join(final_hashes).encode("ascii")
        ).hexdigest(),
        "externalCertification": False,
    }
    semantic["evidenceSha256"] = hashlib.sha256(
        json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--invalid-cases", type=int, default=100)
    parser.add_argument("--native-library")
    parser.add_argument("--enforce-profile", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(
        rounds=args.rounds,
        invalid_cases=args.invalid_cases,
        native_library=args.native_library,
        enforce_profile=args.enforce_profile,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
