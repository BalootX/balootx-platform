"""Particle belief state for observer-safe Baloot analysis."""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from baloot_platform.config import platform_root
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import RoundState

from .hidden_deals import HiddenDealSet, generate_hidden_deals
from .models import InformationState, SignalObservation
from .posterior import deal_likelihood, effective_sample_size

# Process-wide safety ceilings prevent a malformed caller from turning a
# governed tuning override into unbounded CPU or memory use. Product defaults
# remain owned by the active versioned belief profile.
MAXIMUM_HIDDEN_DEAL_ATTEMPTS = 100_000
MAXIMUM_EXACT_ALLOCATION_LIMIT = 100_000


def hidden_deal_attempt_budget(
    requested_particles: int,
    *,
    multiplier: int,
    minimum: int,
) -> int:
    requested = max(1, int(requested_particles))
    return min(
        MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
        max(requested, int(minimum), requested * max(1, int(multiplier))),
    )


@dataclass(frozen=True, slots=True)
class BeliefPolicy:
    profile_id: str
    profile_version: str
    requested_particles: int
    maximum_particles: int
    exact_allocation_limit: int
    maximum_exact_allocation_limit: int
    minimum_effective_fraction: float

    @classmethod
    def from_profile(
        cls,
        profile: dict[str, Any] | None = None,
        *,
        root: str | Path | None = None,
    ) -> "BeliefPolicy":
        if profile is None:
            return _active_belief_policy(
                str(platform_root(root))
            )
        metadata = dict(profile.get("profile") or {})
        particles = dict(profile.get("particles") or {})
        construction = dict(profile.get("construction") or {})
        policy = cls(
            profile_id=str(metadata.get("id", "")).strip(),
            profile_version=str(metadata.get("version", "")).strip(),
            requested_particles=int(particles.get("requested", 0)),
            maximum_particles=int(particles.get("maximum", 0)),
            exact_allocation_limit=int(
                construction.get("exact_when_allocations_below", 0)
            ),
            maximum_exact_allocation_limit=(
                MAXIMUM_EXACT_ALLOCATION_LIMIT
            ),
            minimum_effective_fraction=float(
                particles.get("minimum_effective_fraction", 0.0)
            ),
        )
        if not policy.profile_id or not policy.profile_version:
            raise ValueError("belief profile identity is required")
        if not (
            1
            <= policy.requested_particles
            <= policy.maximum_particles
            <= MAXIMUM_HIDDEN_DEAL_ATTEMPTS
        ):
            raise ValueError("belief particle bounds are invalid")
        if not (
            1
            <= policy.exact_allocation_limit
            <= policy.maximum_exact_allocation_limit
        ):
            raise ValueError("belief exact-allocation bounds are invalid")
        if not 0.0 < policy.minimum_effective_fraction <= 1.0:
            raise ValueError(
                "belief effective-sample fraction must be in (0, 1]"
            )
        if str(particles.get("resampling", "")).strip() != "systematic":
            raise ValueError("only deterministic systematic resampling is supported")
        return policy

    def resolve(
        self,
        *,
        particles: int | None,
        exact_allocation_limit: int | None,
        resample_threshold: float | None,
        maximum_attempts: int | None,
    ) -> tuple[int, int, float, int]:
        requested = (
            self.requested_particles
            if particles is None
            else int(particles)
        )
        exact_limit = (
            self.exact_allocation_limit
            if exact_allocation_limit is None
            else int(exact_allocation_limit)
        )
        threshold = (
            self.minimum_effective_fraction
            if resample_threshold is None
            else float(resample_threshold)
        )
        attempts = (
            MAXIMUM_HIDDEN_DEAL_ATTEMPTS
            if maximum_attempts is None
            else int(maximum_attempts)
        )
        if not 1 <= requested <= self.maximum_particles:
            raise ValueError(
                "requested belief particles exceed the governed maximum"
            )
        if not 1 <= exact_limit <= self.maximum_exact_allocation_limit:
            raise ValueError(
                "exact-allocation limit exceeds the safety maximum"
            )
        if not 0.0 < threshold <= 1.0:
            raise ValueError("resample threshold must be in (0, 1]")
        if not requested <= attempts <= MAXIMUM_HIDDEN_DEAL_ATTEMPTS:
            raise ValueError(
                "hidden-deal attempts violate the governed safety bounds"
            )
        return requested, exact_limit, threshold, attempts


@lru_cache(maxsize=8)
def _active_belief_policy(resolved_root: str) -> BeliefPolicy:
    root = Path(resolved_root).resolve()
    bundle = yaml.safe_load(
        (root / "configs/bundle.yaml").read_text(encoding="utf-8")
    )
    relative = str(
        dict(bundle.get("components") or {}).get("belief", "")
    ).strip()
    if not relative:
        raise ValueError("active bundle has no belief profile")
    config_root = (root / "configs").resolve()
    profile_path = (config_root / relative).resolve()
    try:
        profile_path.relative_to(config_root)
    except ValueError as exc:
        raise ValueError(
            "belief profile path escapes the configuration root"
        ) from exc
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    return BeliefPolicy.from_profile(profile)


@dataclass(frozen=True, slots=True)
class BeliefParticle:
    state: RoundState
    weight: float


@dataclass(frozen=True, slots=True)
class ParticleBelief:
    particles: tuple[BeliefParticle, ...]
    method: str
    attempted: int
    effective_sample_size: float
    resampled: bool
    seed: int
    profile_id: str
    profile_version: str

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        total = sum(max(0.0, item.weight) for item in self.particles)
        if total <= 0:
            return tuple(1.0 / len(self.particles) for _ in self.particles)
        return tuple(max(0.0, item.weight) / total for item in self.particles)

    def card_location_probabilities(self) -> dict[str, dict[str, float]]:
        from baloot_platform.core.cards import FULL_DECK
        result: dict[str, dict[str, float]] = {card.code: {seat.name.lower(): 0.0 for seat in Seat} for card in FULL_DECK}
        for particle, weight in zip(self.particles, self.normalized_weights):
            for seat in Seat:
                mask = particle.state.hands[int(seat)]
                for card in FULL_DECK:
                    if mask & (1 << card.id):
                        result[card.code][seat.name.lower()] += weight
        return result

    def to_dict(self, *, include_card_locations: bool = True) -> dict[str, Any]:
        return {
            "method": self.method,
            "particleCount": len(self.particles),
            "attempted": self.attempted,
            "effectiveSampleSize": self.effective_sample_size,
            "resampled": self.resampled,
            "seed": self.seed,
            "profileId": self.profile_id,
            "profileVersion": self.profile_version,
            "cardLocations": self.card_location_probabilities() if include_card_locations else None,
        }


def _systematic_resample(particles: tuple[BeliefParticle, ...], seed: int) -> tuple[BeliefParticle, ...]:
    count = len(particles)
    if count == 0:
        return ()
    weights = [max(0.0, item.weight) for item in particles]
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / count] * count
    else:
        weights = [weight / total for weight in weights]
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running)
    rng = random.Random(seed)
    start = rng.random() / count
    indexes: list[int] = []
    cursor = 0
    for index in range(count):
        point = start + index / count
        while cursor < count - 1 and point > cumulative[cursor]:
            cursor += 1
        indexes.append(cursor)
    uniform = 1.0 / count
    return tuple(BeliefParticle(particles[index].state.clone(), uniform) for index in indexes)


def build_particle_belief(
    info: InformationState,
    engine: BalootEngine,
    observations: tuple[SignalObservation, ...] = (),
    convention_profile: dict[str, Any] | None = None,
    *,
    particles: int | None = None,
    exact_allocation_limit: int | None = None,
    maximum_attempts: int | None = None,
    seed: int = 1,
    resample_threshold: float | None = None,
    belief_profile: dict[str, Any] | None = None,
    belief_policy: BeliefPolicy | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ParticleBelief:
    if belief_policy is not None and belief_profile is not None:
        raise ValueError(
            "provide either a compiled belief policy or a belief profile, not both"
        )
    policy = (
        belief_policy
        if belief_policy is not None
        else BeliefPolicy.from_profile(belief_profile)
    )
    requested, exact_limit, threshold, attempts = policy.resolve(
        particles=particles,
        exact_allocation_limit=exact_allocation_limit,
        resample_threshold=resample_threshold,
        maximum_attempts=maximum_attempts,
    )
    deals: HiddenDealSet = generate_hidden_deals(
        info, engine, requested_samples=requested,
        exact_allocation_limit=exact_limit,
        maximum_attempts=attempts,
        seed=int(seed),
        cancel_check=cancel_check,
    )
    raw = tuple(
        BeliefParticle(
            state,
            deal_likelihood(
                state,
                info,
                observations,
                convention_profile,
                play_domain=engine.rules.play_domain,
            ),
        )
        for state in deals.states
    )
    ess = effective_sample_size([item.weight for item in raw])
    should_resample = len(raw) > 1 and ess < threshold * len(raw)
    final = _systematic_resample(raw, seed + 0xB41007) if should_resample else raw
    final_ess = effective_sample_size([item.weight for item in final])
    return ParticleBelief(
        final,
        deals.method,
        deals.attempted,
        final_ess,
        should_resample,
        int(seed),
        policy.profile_id,
        policy.profile_version,
    )


def calibration_brier(
    forecasts: list[dict[str, dict[str, float]]],
    truths: list[dict[str, str]],
) -> float:
    """Multiclass Brier score for held-out card-location forecasts."""
    if len(forecasts) != len(truths) or not forecasts:
        raise ValueError("matching non-empty forecasts and truths are required")
    total = 0.0
    count = 0
    for forecast, truth in zip(forecasts, truths):
        for card, actual_seat in truth.items():
            probs = forecast[card]
            for seat in ("north", "west", "south", "east"):
                total += (float(probs.get(seat, 0.0)) - float(seat == actual_seat)) ** 2
                count += 1
    return total / max(1, count)
