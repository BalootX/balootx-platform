from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit

from .adjudication import canonical_sha256


def benchmark_configuration_sha256(bundle: Any) -> str:
    """Hash only governed inputs that can affect the frozen strength corpus.

    Presentation, media, deployment, and other unrelated profile changes must
    not invalidate engine-strength evidence.
    """

    return canonical_sha256(
        {
            "schemaVersion": 1,
            "rulesSha256": bundle.rules.sha256,
            "benchmarkProfileSha256": bundle.components[
                "benchmark"
            ].sha256,
        }
    )


def load_candidate_jsonl(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("candidate corpus is empty")
    value = {**rows[0], "positions": rows[1:]}
    if int(value.get("candidateCount", -1)) != len(value["positions"]):
        raise ValueError("candidate manifest count mismatch")
    if canonical_sha256(value["positions"]) != str(
        value.get("datasetSha256", "")
    ):
        raise ValueError("candidate corpus hash mismatch")
    return value


def load_adjudicated_holdout_jsonl(path: Path) -> dict[str, Any]:
    """Load and checksum a finalized permanent-holdout artifact."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("permanent holdout is empty")
    value = {**rows[0], "positions": rows[1:]}
    if int(value.get("positionCount", -1)) != len(value["positions"]):
        raise ValueError("permanent holdout count mismatch")
    if canonical_sha256(value["positions"]) != str(value.get("datasetSha256", "")):
        raise ValueError("permanent holdout dataset hash mismatch")
    manifest_hash = str(value.get("manifestSha256", ""))
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"positions", "manifestSha256"}
    }
    payload["positions"] = value["positions"]
    if canonical_sha256(payload) != manifest_hash:
        raise ValueError("permanent holdout manifest hash mismatch")
    return value


def validate_trusted_sidecar(
    candidates: dict[str, Any],
    trusted: dict[str, Any],
) -> None:
    for key in (
        "datasetId",
        "datasetSha256",
        "rulesSha256",
        "configurationSha256",
    ):
        if str(trusted.get(key, "")) != str(candidates.get(key, "")):
            raise ValueError(f"trusted sidecar {key} mismatch")
    recipes = dict(trusted.get("recipes") or {})
    if int(trusted.get("recipeCount", -1)) != len(recipes):
        raise ValueError("trusted sidecar recipe count mismatch")
    payload = {
        key: value
        for key, value in trusted.items()
        if key != "sidecarSha256"
    }
    if str(trusted.get("sidecarSha256", "")) != canonical_sha256(payload):
        raise ValueError("trusted sidecar hash mismatch")


def reconstruct_state(
    engine: BalootEngine,
    candidate: dict[str, Any],
    recipe: dict[str, Any],
):
    if str(recipe.get("stage", "")) != "card_play":
        raise ValueError("trusted recipe is not a card-play state")
    state = engine.create_round(
        seed=int(recipe["seed"]),
        dealer=Seat.parse(recipe["dealer"]),
        match_score=tuple(int(value) for value in recipe["matchScore"]),
    )
    engine.shuffle_and_initial_deal(
        state,
        cut_position=int(recipe["cutPosition"]),
    )
    contract = Contract(str(recipe["contract"]))
    engine.bid_and_complete_deal(
        state,
        forced_contract=contract,
        forced_buyer=Seat.parse(recipe["buyer"]),
        forced_trump=(
            Suit.parse(recipe["trumpSuit"])
            if recipe.get("trumpSuit") is not None
            else None
        ),
    )
    engine.prepare_projects(state)
    for action_id in recipe.get("actionPrefix", ()):
        engine.play_card(state, int(action_id))
    if engine.state_hash(state) != str(candidate["stateSha256"]):
        raise ValueError("trusted replay did not reproduce candidate state")
    if (
        state.current_player is None
        or state.current_player.name.lower() != str(candidate["observer"])
    ):
        raise ValueError("candidate observer is not the acting player")
    legal = [card.code for card in engine.legal_actions(state)]
    if legal != list(candidate["legalActions"]):
        raise ValueError("trusted replay legal actions mismatch")
    return state
