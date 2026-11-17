"""Stable game/video/model provenance identifiers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def game_serial(game_hash: str) -> str:
    clean = "".join(ch for ch in game_hash.lower() if ch in "0123456789abcdef")
    if len(clean) < 16:
        raise ValueError("game hash must contain at least 16 hexadecimal characters")
    return f"BLT-{clean[:4].upper()}-{clean[4:8].upper()}-{clean[8:12].upper()}-{clean[12:16].upper()}"


@dataclass(frozen=True)
class ProvenanceLink:
    game_serial: str
    game_hash: str
    state_hash: str
    rules_bundle_hash: str
    engine_version: str
    challenge_id: str | None = None
    analysis_hash: str | None = None
    renderer_id: str | None = None
    renderer_version: str | None = None
    model_ids: tuple[str, ...] = ()
    parent_game_hash: str | None = None

    @property
    def short_hash(self) -> str:
        return self.game_hash[:12]

    @property
    def artifact_identity(self) -> str:
        return canonical_hash({
            "gameSerial": self.game_serial,
            "gameHash": self.game_hash,
            "stateHash": self.state_hash,
            "rulesBundleHash": self.rules_bundle_hash,
            "engineVersion": self.engine_version,
            "challengeId": self.challenge_id,
            "analysisHash": self.analysis_hash,
            "rendererId": self.renderer_id,
            "rendererVersion": self.renderer_version,
            "modelIds": self.model_ids,
            "parentGameHash": self.parent_game_hash,
        })
