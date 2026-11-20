#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate frozen, observer-safe strength holdout candidates.

This stage deliberately does not label positions. Exact software-oracle and
independent adjudication evidence are attached by a separate governed step so
candidate generation cannot silently create its own answers.
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
from baloot_platform.benchmark import benchmark_configuration_sha256, validate_trusted_sidecar
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.version import __version__


OUTPUT = Path("tests/fixtures/synthetic-candidates.jsonl")
DEFAULT_TRUSTED = Path(
    ".release-runtime/strength-holdout/trusted-replay-recipes.json"
)
SYNTHETIC_PURPOSE = "independent_synthetic_test_fixture_not_holdout_evidence"
ARCHETYPES = (
    "bidding",
    "opening_lead",
    "play_obligation",
    "projects",
    "midgame",
    "endgame",
    "contract_rescue",
    "risk",
    "outcome_flip",
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def serialize_public(value: dict[str, Any]) -> str:
    manifest = {
        key: item for key, item in value.items() if key != "positions"
    }
    lines = [manifest, *value["positions"]]
    return "".join(
        json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in lines
    )


def load_public(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("holdout candidate corpus is empty")
    return {**rows[0], "positions": rows[1:]}


def _deal_hash(state) -> str:
    return _hash(
        {
            "rulesSha256": state.rules_hash,
            "seed": state.seed,
            "dealer": state.dealer.name.lower(),
            "cutPosition": (
                state.shuffle.cut_position if state.shuffle is not None else None
            ),
            "initialHands": list(state.initial_hands),
            "purchaseCard": (
                state.purchase_card.id
                if state.purchase_card is not None
                else None
            ),
        }
    )


def _observer_snapshot(view: dict[str, Any]) -> dict[str, Any]:
    """Return only information an acting player is authorized to receive."""
    allowed = (
        "rulesHash",
        "roundNumber",
        "dealer",
        "phase",
        "matchScore",
        "purchaseCard",
        "hands",
        "biddingHistory",
        "contract",
        "winningBidder",
        "buyer",
        "contractOwner",
        "purchaseRecipient",
        "declarer",
        "trumpSuit",
        "escalation",
        "escalationHistory",
        "locked",
        "firstLeader",
        "currentPlayer",
        "currentTrick",
        "completedTricks",
        "projects",
        "projectWinningTeam",
        "balootTeams",
        "trumpBroken",
        "roundScore",
        "matchWinner",
        "annulledBy",
        "annulReason",
        "observer",
        "bidding",
    )
    return {key: view[key] for key in allowed if key in view}


def _base(
    *,
    archetype: str,
    index: int,
    state,
    engine: BalootEngine,
    bundle_sha256: str,
    observer: Seat,
    legal_actions: list[str],
    observer_state: dict[str, Any],
) -> dict[str, Any]:
    seed_group = f"holdout-seed-{archetype}-{index:04d}"
    return {
        "candidateId": f"candidate-{archetype}-{index:04d}",
        "split": "permanent_holdout_candidate",
        "archetype": archetype,
        "observer": observer.name.lower(),
        "observerState": _observer_snapshot(observer_state),
        "legalActions": legal_actions,
        "stateSha256": engine.state_hash(state),
        "dealHash": _deal_hash(state),
        "episodeId": f"holdout-episode-{archetype}-{index:04d}",
        "sourceGroup": f"deterministic-{archetype}",
        "seedGroup": seed_group,
        "engineVersion": __version__,
        "rulesSha256": state.rules_hash,
        "configurationSha256": bundle_sha256,
        "labelStatus": "pending_governed_adjudication",
        "proofBoundary": (
            "Observer-safe candidate generation only; no exact, strength, "
            "promotion, or external-adjudication claim."
        ),
    }


def _bidding_candidate(
    engine: BalootEngine,
    bundle_sha256: str,
    *,
    seed: int,
    index: int,
) -> dict[str, Any]:
    dealer = Seat(index % 4)
    state = engine.create_round(seed=seed, dealer=dealer)
    engine.shuffle_and_initial_deal(
        state,
        cut_position=(index * 11 % 31) + 1,
    )
    if state.purchase_card is None:
        raise AssertionError("initial deal has no purchase card")
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(engine.rules.data),
    )
    observer = machine.actor
    if observer is None:
        raise AssertionError("new bidding machine has no actor")
    view = engine.observer_view(state, observer)
    view["bidding"] = {
        "round": machine.round_index,
        "actor": observer.name.lower(),
        "legalActions": list(machine.legal_actions()),
        "history": [],
    }
    candidate = _base(
        archetype="bidding",
        index=index,
        state=state,
        engine=engine,
        bundle_sha256=bundle_sha256,
        observer=observer,
        legal_actions=list(machine.legal_actions()),
        observer_state=view,
    )
    candidate["_trustedRecipe"] = {
        "seed": seed,
        "dealer": dealer.name.lower(),
        "cutPosition": (index * 11 % 31) + 1,
        "actionPrefix": [],
        "stage": "initial_bidding",
    }
    return candidate


def _play_candidate(
    engine: BalootEngine,
    bundle_sha256: str,
    *,
    archetype: str,
    seed: int,
    index: int,
    target_plays: int,
) -> dict[str, Any] | None:
    dealer = Seat(index % 4)
    state = engine.create_round(
        seed=seed,
        dealer=dealer,
        match_score=(
            (140, 140)
            if archetype == "contract_rescue"
            else (0, 0)
        ),
    )
    engine.shuffle_and_initial_deal(
        state,
        cut_position=(index * 13 % 31) + 1,
    )
    contract = Contract.SUN if index % 2 == 0 else Contract.HOKUM
    buyer = Seat((index * 3 + 1) % 4)
    trump = (
        None
        if contract == Contract.SUN
        else Suit((index * 5 + 1) % 4)
    )
    engine.bid_and_complete_deal(
        state,
        forced_contract=contract,
        forced_buyer=buyer,
        forced_trump=trump,
    )
    engine.prepare_projects(state)
    bot = HeuristicBot(
        engine,
        BotProfile(deterministic_seed=seed ^ 0xB4100C),
    )
    action_prefix: list[int] = []
    while (
        state.current_player is not None
        and len(state.played_cards) < target_plays
    ):
        action = bot.choose(state)
        action_prefix.append(action.id)
        engine.play_card(state, action)
    if state.current_player is None:
        return None
    legal = engine.legal_actions(state)
    if len(legal) < 2:
        return None
    if archetype == "projects" and not state.projects:
        return None
    observer = state.current_player
    candidate = _base(
        archetype=archetype,
        index=index,
        state=state,
        engine=engine,
        bundle_sha256=bundle_sha256,
        observer=observer,
        legal_actions=[card.code for card in legal],
        observer_state=engine.observer_view(state, observer),
    )
    candidate["_trustedRecipe"] = {
        "seed": seed,
        "dealer": dealer.name.lower(),
        "cutPosition": (index * 13 % 31) + 1,
        "contract": contract.value,
        "buyer": buyer.name.lower(),
        "trumpSuit": (
            trump.name.lower() if trump is not None else None
        ),
        "matchScore": list(state.match_score),
        "actionPrefix": action_prefix,
        "stage": "card_play",
    }
    return candidate


def generate_artifacts(
    *,
    per_archetype: int = 1,
    root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if per_archetype < 1:
        raise ValueError("per_archetype must be positive")
    resolved_root = root or platform_root()
    bundle = build_bundle(resolved_root)
    configuration_sha256 = benchmark_configuration_sha256(bundle)
    engine = BalootEngine(bundle.rules)
    positions: list[dict[str, Any]] = []
    recipes: dict[str, dict[str, Any]] = {}
    target_plays = {
        "opening_lead": 0,
        "play_obligation": 1,
        "projects": 0,
        "midgame": 12,
        "endgame": 24,
        "contract_rescue": 20,
        "risk": 16,
        "outcome_flip": 24,
    }
    for archetype_index, archetype in enumerate(ARCHETYPES):
        accepted = 0
        attempt = 0
        while accepted < per_archetype:
            if attempt >= per_archetype * 200:
                raise RuntimeError(
                    f"could not generate enough {archetype} candidates"
                )
            index = archetype_index * 1_000_000 + attempt
            seed = 31_000_000 + index
            if archetype == "bidding":
                candidate = _bidding_candidate(
                    engine,
                    configuration_sha256,
                    seed=seed,
                    index=accepted,
                )
            else:
                candidate = _play_candidate(
                    engine,
                    configuration_sha256,
                    archetype=archetype,
                    seed=seed,
                    index=accepted,
                    target_plays=target_plays[archetype],
                )
            attempt += 1
            if candidate is None:
                continue
            recipe = dict(candidate.pop("_trustedRecipe"))
            recipes[str(candidate["candidateId"])] = recipe
            positions.append(candidate)
            accepted += 1
    positions.sort(key=lambda item: str(item["candidateId"]))
    body_sha256 = _hash(positions)
    public = {
        "schemaVersion": 1,
        "datasetId": f"synthetic-smoke-candidates-{body_sha256[:20]}",
        "datasetSha256": body_sha256,
        "engineVersion": __version__,
        "rulesSha256": bundle.rules.sha256,
        "configurationSha256": configuration_sha256,
        "perArchetype": per_archetype,
        "candidateCount": len(positions),
        "purpose": SYNTHETIC_PURPOSE,
        "archetypes": list(ARCHETYPES),
        "labelStatus": "pending_governed_adjudication",
        "positions": positions,
    }
    trusted = {
        "schemaVersion": 1,
        "datasetId": public["datasetId"],
        "datasetSha256": public["datasetSha256"],
        "rulesSha256": bundle.rules.sha256,
        "configurationSha256": configuration_sha256,
        "recipeCount": len(recipes),
        "recipes": dict(sorted(recipes.items())),
        "classification": "trusted_hidden_state_reconstruction",
        "consumerBoundary": (
            "Never expose this sidecar to candidate agents, renderers, "
            "observer APIs, or public benchmark inputs."
        ),
    }
    trusted["sidecarSha256"] = _hash(trusted)
    return public, trusted


def generate(
    *,
    per_archetype: int = 1,
    root: Path | None = None,
) -> dict[str, Any]:
    public, _ = generate_artifacts(
        per_archetype=per_archetype,
        root=root,
    )
    return public


def load_replay_recipes(
    candidates: dict[str, Any], *, root: Path, path: Path | None = None,
) -> dict[str, Any]:
    """Use supplied governed recipes or reconstruct matching synthetic recipes in memory."""
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    legacy_path = root / DEFAULT_TRUSTED
    if candidates.get("purpose") != SYNTHETIC_PURPOSE:
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    per_archetype = candidates.get("perArchetype")
    if (
        type(per_archetype) is not int
        or per_archetype < 1
        or per_archetype * len(ARCHETYPES) != len(candidates["positions"])
    ):
        raise ValueError("invalid synthetic holdout archetype count")
    _, recipes = generate_artifacts(per_archetype=per_archetype, root=root)
    validate_trusted_sidecar(candidates, recipes)
    return recipes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--per-archetype", type=int, default=1)
    args = parser.parse_args()
    root = platform_root()
    output = root / OUTPUT
    text = serialize_public(generate(per_archetype=args.per_archetype, root=root))
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != text:
            raise SystemExit("holdout candidate corpus is stale")
        return
    output.write_text(text, encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
