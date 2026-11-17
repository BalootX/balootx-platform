"""Deterministic complete natural-Sakka generation and evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.core.enums import Seat
from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import canonical_sha256
from .sakka_context import NaturalSakkaV1
from .simulation import (
    NaturalLaffaSimulationResultV1,
    NaturalLaffaSimulatorV1,
    derive_foundry_seed_v1,
)


@dataclass(frozen=True, slots=True)
class NaturalSakkaGenerationProfileV1:
    profile_id: str
    revision: str
    target: int
    maximum_rounds: int
    seed_policy: str
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "NaturalSakkaGenerationProfileV1":
        raw = dict(value)
        allowed = {"schemaVersion", "id", "revision", "target", "maximumRounds", "seedPolicy"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown natural Sakka generation keys: {sorted(unknown)}")
        if int(raw.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported natural Sakka generation schema version")
        profile_id = str(raw.get("id") or "").strip()
        revision = str(raw.get("revision") or "").strip()
        target = int(raw.get("target", 0))
        maximum_rounds = int(raw.get("maximumRounds", 0))
        seed_policy = str(raw.get("seedPolicy") or "")
        if not profile_id or not revision:
            raise ValueError("natural Sakka generation identity is required")
        if target < 1:
            raise ValueError("natural Sakka target must be positive")
        if not 1 <= maximum_rounds <= 1_000:
            raise ValueError("natural Sakka maximum rounds must be in [1, 1000]")
        if seed_policy != "sha256_round_ordinal_v1":
            raise ValueError("unsupported natural Sakka seed policy")
        core = {
            "schemaVersion": 1,
            "id": profile_id,
            "revision": revision,
            "target": target,
            "maximumRounds": maximum_rounds,
            "seedPolicy": seed_policy,
        }
        return cls(
            profile_id,
            revision,
            target,
            maximum_rounds,
            seed_policy,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "target": self.target,
            "maximumRounds": self.maximum_rounds,
            "seedPolicy": self.seed_policy,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class NaturalSakkaGenerationResultV1:
    root_seed: int
    initial_dealer: Seat
    profile_hash: str
    sakka: NaturalSakkaV1
    rounds: tuple[NaturalLaffaSimulationResultV1, ...]
    round_seeds: tuple[int, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        root_seed: int,
        initial_dealer: Seat | str,
        profile_hash: str,
        sakka: NaturalSakkaV1,
        rounds: Iterable[NaturalLaffaSimulationResultV1],
        round_seeds: Iterable[int],
    ) -> "NaturalSakkaGenerationResultV1":
        values = tuple(rounds)
        seeds = tuple(int(item) for item in round_seeds)
        dealer = Seat.parse(initial_dealer) if isinstance(initial_dealer, str) else initial_dealer
        if int(root_seed) < 0:
            raise ValueError("natural Sakka root seed must be non-negative")
        if not values or len(values) != len(seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("natural Sakka generation requires one unique seed per round")
        if any(not item.publishable_laffa for item in values):
            raise ValueError("natural Sakka generation contains an unplayed Laffa")
        if tuple(item.final_trace.sha256 for item in values) != tuple(
            str(item["traceHash"]) for item in sakka.rounds
        ):
            raise ValueError("natural Sakka generation trace order mismatch")
        profile = str(profile_hash).strip().lower()
        if len(profile) != 64 or any(char not in "0123456789abcdef" for char in profile):
            raise ValueError("natural Sakka generation profile hash is invalid")
        core = {
            "schemaVersion": 1,
            "rootSeed": int(root_seed),
            "initialDealer": dealer.name.lower(),
            "profileHash": profile,
            "sakkaHash": sakka.sha256,
            "roundSeeds": list(seeds),
            "roundResults": [item.sha256 for item in values],
            "attemptTraceHashes": [
                [trace.sha256 for trace in item.attempt_traces] for item in values
            ],
        }
        return cls(
            int(root_seed), dealer, profile, sakka, values, seeds, canonical_sha256(core)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "rootSeed": self.root_seed,
            "initialDealer": self.initial_dealer.name.lower(),
            "profileHash": self.profile_hash,
            "sakka": self.sakka.to_dict(),
            "roundSeeds": list(self.round_seeds),
            "roundResults": [item.to_summary_dict() for item in self.rounds],
            "sha256": self.sha256,
        }


class NaturalSakkaGeneratorV1:
    descriptor = PluginDescriptor(
        id="natural_sakka_generator_v1",
        kind="sakka_composer",
        version="1.0.0",
        description="Generate one deterministic uninterrupted natural Sakka",
        capabilities=(
            "complete_match",
            "complete_auction_lineage",
            "score_continuity",
            "dealer_continuity",
            "equal_agent_release",
            "deterministic",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4_096,
    )

    def __init__(
        self,
        simulator: NaturalLaffaSimulatorV1,
        profile: NaturalSakkaGenerationProfileV1,
    ) -> None:
        rules_target = int(simulator.rules.data["scoring"]["match"]["target"])
        if profile.target != rules_target:
            raise ValueError("natural Sakka generation target must match the active rules")
        self.simulator = simulator
        self.profile = profile

    def generate(
        self,
        root_seed: int,
        *,
        initial_dealer: Seat | str = Seat.EAST,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NaturalSakkaGenerationResultV1:
        seed = int(root_seed)
        if seed < 0:
            raise ValueError("natural Sakka root seed must be non-negative")
        dealer = Seat.parse(initial_dealer) if isinstance(initial_dealer, str) else initial_dealer
        starting_dealer = dealer
        score = (0, 0)
        round_number = 1
        rounds: list[NaturalLaffaSimulationResultV1] = []
        round_seeds: list[int] = []
        for ordinal in range(1, self.profile.maximum_rounds + 1):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("natural Sakka generation cancelled")
            round_seed = derive_foundry_seed_v1(
                "natural_sakka",
                self.profile.seed_policy,
                self.profile.sha256,
                seed,
                ordinal,
            )
            result = self.simulator.simulate(
                round_seed,
                dealer=dealer,
                match_score=score,
                round_number=round_number,
                cancel_check=cancel_check,
            )
            if not result.publishable_laffa:
                raise RuntimeError("natural Sakka generation exhausted the auction-attempt bound")
            rounds.append(result)
            round_seeds.append(round_seed)
            trace = result.final_trace
            score = tuple(int(item) for item in trace.trusted_final_state["matchScore"])
            dealer = self.simulator.structural_domain.next_dealer(
                Seat.parse(str(trace.trusted_final_state["dealer"]))
            )
            round_number = int(trace.trusted_final_state["roundNumber"]) + 1
            if progress_callback is not None:
                progress_callback(ordinal, self.profile.maximum_rounds)
            if max(score) >= self.profile.target:
                sakka = NaturalSakkaV1.build(
                    (item.final_trace for item in rounds),
                    target=self.profile.target,
                    attempt_trace_groups=(item.attempt_traces for item in rounds),
                )
                return NaturalSakkaGenerationResultV1.build(
                    root_seed=seed,
                    initial_dealer=starting_dealer,
                    profile_hash=self.profile.sha256,
                    sakka=sakka,
                    rounds=rounds,
                    round_seeds=round_seeds,
                )
        raise RuntimeError("natural Sakka did not finish within the governed round bound")

    def compose(
        self,
        root_seed: int,
        *,
        initial_dealer: Seat | str = Seat.EAST,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NaturalSakkaGenerationResultV1:
        """Expose natural generation through the governed composer boundary."""

        return self.generate(
            root_seed,
            initial_dealer=initial_dealer,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
