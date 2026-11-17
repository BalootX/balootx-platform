from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .enums import Contract, Seat, Suit, Team
from .scenario import EpisodeSpec, ScenarioRequest, generate_episode
from .rules import RuleBook


@dataclass(frozen=True, slots=True)
class MatchScenarioRequest:
    seed: int
    dealer: Seat
    viewer: Seat
    maximum_rounds: int
    contract: Contract | None
    trump_suit: Suit | None
    require_project: bool
    requested_samples: int
    exact_search_max_remaining: int

    @classmethod
    def governed(
        cls,
        profile: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> "MatchScenarioRequest":
        from baloot_platform.config import platform_root
        import yaml

        if profile is None:
            profile = yaml.safe_load(
                (
                    platform_root()
                    / "configs/challenge/progressive_mining_v1.yaml"
                ).read_text(encoding="utf-8")
            )
        raw = dict(profile["match_defaults"])
        unknown = set(overrides) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown match scenario request keys: {sorted(unknown)}"
            )
        raw.update(overrides)
        raw["dealer"] = (
            raw["dealer"] if isinstance(raw["dealer"], Seat)
            else Seat[str(raw["dealer"]).upper()]
        )
        raw["viewer"] = (
            raw["viewer"] if isinstance(raw["viewer"], Seat)
            else Seat[str(raw["viewer"]).upper()]
        )
        raw["contract"] = (
            raw["contract"] if isinstance(raw["contract"], Contract)
            else None if raw["contract"] is None
            else Contract(str(raw["contract"]).lower())
        )
        raw["trump_suit"] = (
            raw["trump_suit"] if isinstance(raw["trump_suit"], Suit)
            else None if raw["trump_suit"] is None
            else Suit[str(raw["trump_suit"]).upper()]
        )
        if int(raw["maximum_rounds"]) < 1:
            raise ValueError("match maximum rounds must be positive")
        if int(raw["requested_samples"]) < 1:
            raise ValueError("match requested samples must be positive")
        if not 0 <= int(raw["exact_search_max_remaining"]) <= 32:
            raise ValueError("match exact-search remainder must be in [0, 32]")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class MatchEpisodeSpec:
    schema_version: int
    match_id: str
    bundle_hash: str
    rounds: tuple[EpisodeSpec, ...]
    final_score: tuple[int, int]
    winner: Team | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "matchId": self.match_id,
            "bundleHash": self.bundle_hash,
            "rounds": [episode.to_dict() for episode in self.rounds],
            "finalScore": list(self.final_score),
            "winner": self.winner.name.lower() if self.winner is not None else None,
        }


def generate_match_episode(
    rules: RuleBook,
    convention: dict[str, Any],
    *,
    bundle_hash: str,
    request: MatchScenarioRequest,
    copy: Mapping[str, Any] | None = None,
) -> MatchEpisodeSpec:
    score = (0, 0)
    dealer = request.dealer
    episodes: list[EpisodeSpec] = []
    winner: Team | None = None
    target = int(rules.data["scoring"]["match"]["target"])
    for round_index in range(1, request.maximum_rounds + 1):
        episode = generate_episode(
            rules,
            convention,
            bundle_hash=bundle_hash,
            copy=copy,
            request=ScenarioRequest.governed(
                seed=request.seed + (round_index - 1) * 1000,
                viewer=request.viewer,
                dealer=dealer,
                round_number=round_index,
                contract=request.contract,
                trump_suit=request.trump_suit,
                match_score=score,
                require_project=request.require_project,
                requested_samples=request.requested_samples,
                exact_search_max_remaining=request.exact_search_max_remaining,
                max_seed_attempts=200,
            ),
        )
        episodes.append(episode)
        score = tuple(episode.final_state.match_score)
        winner = episode.final_state.match_winner
        if winner is not None or (max(score) >= target and score[0] != score[1]):
            winner = Team.NORTH_SOUTH if score[0] > score[1] else Team.WEST_EAST
            break
        dealer = rules.structural_domain.next_dealer(dealer)
    match_id = f"match-{request.seed}-{bundle_hash[:10]}-{len(episodes)}"
    return MatchEpisodeSpec(1, match_id, bundle_hash, tuple(episodes), score, winner)
