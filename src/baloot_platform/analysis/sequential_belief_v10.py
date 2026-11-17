"""Sequential observer-safe particle belief updates for Baloot.

The V7 builder creates a legal initial particle cohort.  This module advances
that same cohort after each public card instead of rebuilding independent
hidden deals from scratch.  Hard rules eliminate impossible particles;
optional convention likelihoods may only reweight surviving legal particles.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat

from .belief_v7 import (
    BeliefParticle,
    BeliefPolicy,
    ParticleBelief,
    build_particle_belief,
)
from .models import InformationState, SignalObservation
from .posterior import deal_likelihood, effective_sample_size


@dataclass(frozen=True, slots=True)
class SequentialBeliefUpdate:
    step: int
    actor: Seat
    card: Card
    before_particles: int
    surviving_particles: int
    effective_sample_size: float
    resampled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "actor": self.actor.name.lower(),
            "card": self.card.to_dict(),
            "beforeParticles": self.before_particles,
            "survivingParticles": self.surviving_particles,
            "effectiveSampleSize": self.effective_sample_size,
            "resampled": self.resampled,
        }


@dataclass(slots=True)
class SequentialParticleBelief:
    observer: Seat
    particles: tuple[BeliefParticle, ...]
    seed: int
    profile_id: str
    profile_version: str
    minimum_effective_fraction: float
    step: int = 0
    history: tuple[SequentialBeliefUpdate, ...] = ()

    @classmethod
    def initialize(
        cls,
        info: InformationState,
        engine: BalootEngine,
        observations: tuple[SignalObservation, ...] = (),
        convention_profile: dict[str, Any] | None = None,
        *,
        particles: int | None = None,
        exact_allocation_limit: int | None = None,
        seed: int = 1,
        belief_profile: dict[str, Any] | None = None,
    ) -> "SequentialParticleBelief":
        policy = BeliefPolicy.from_profile(belief_profile)
        belief = build_particle_belief(
            info,
            engine,
            observations,
            convention_profile,
            particles=particles,
            exact_allocation_limit=exact_allocation_limit,
            seed=seed,
            belief_profile=belief_profile,
        )
        return cls(
            info.observer,
            belief.particles,
            int(seed),
            policy.profile_id,
            policy.profile_version,
            policy.minimum_effective_fraction,
        )

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        if not self.particles:
            return ()
        total = sum(max(0.0, item.weight) for item in self.particles)
        if total <= 0:
            return tuple(1.0 / len(self.particles) for _ in self.particles)
        return tuple(max(0.0, item.weight) / total for item in self.particles)

    def as_particle_belief(self) -> ParticleBelief:
        ess = effective_sample_size([p.weight for p in self.particles])
        return ParticleBelief(
            self.particles,
            "sequential_constrained_particle_belief_v2",
            len(self.particles),
            ess,
            bool(self.history and self.history[-1].resampled),
            self.seed,
            self.profile_id,
            self.profile_version,
        )

    @staticmethod
    def _systematic_resample(
        particles: tuple[BeliefParticle, ...],
        *,
        seed: int,
        target_count: int,
    ) -> tuple[BeliefParticle, ...]:
        if not particles:
            return ()
        weights = [max(0.0, p.weight) for p in particles]
        total = sum(weights)
        if total <= 0:
            weights = [1.0 / len(particles)] * len(particles)
        else:
            weights = [w / total for w in weights]
        cumulative: list[float] = []
        running = 0.0
        for weight in weights:
            running += weight
            cumulative.append(running)
        rng = random.Random(seed)
        start = rng.random() / target_count
        cursor = 0
        uniform = 1.0 / target_count
        output: list[BeliefParticle] = []
        for index in range(target_count):
            point = start + index / target_count
            while cursor < len(cumulative) - 1 and point > cumulative[cursor]:
                cursor += 1
            output.append(BeliefParticle(particles[cursor].state.clone(), uniform))
        return tuple(output)

    def observe_card(
        self,
        engine: BalootEngine,
        *,
        actor: Seat,
        card: Card | str | int | dict[str, Any],
        ekka: bool = False,
        updated_information: InformationState | None = None,
        observations: tuple[SignalObservation, ...] = (),
        convention_profile: dict[str, Any] | None = None,
        resample_fraction: float | None = None,
    ) -> SequentialBeliefUpdate:
        """Advance every compatible particle through one public card action.

        The observed action is applied only where it was legal from the acting
        player's hidden hand.  This makes legal information authoritative.
        Behavioral likelihoods are applied only after the legal filter and can
        never resurrect an impossible particle.
        """
        typed = Card.parse(card)
        before = len(self.particles)
        advanced: list[BeliefParticle] = []
        for particle in self.particles:
            state = particle.state.clone()
            if state.current_player != actor:
                continue
            legal_ids = {item.id for item in engine.legal_moves(state)}
            if typed.id not in legal_ids:
                continue
            engine.play_card(state, typed, ekka=ekka)
            state.events = []
            weight = max(0.0, float(particle.weight))
            if updated_information is not None:
                # Reweight only using observer-safe public information.  The
                # deal-likelihood function itself is disabled unless the active
                # convention profile explicitly allows posterior weighting.
                weight *= deal_likelihood(
                    state,
                    updated_information,
                    observations,
                    convention_profile,
                    play_domain=engine.rules.play_domain,
                )
            advanced.append(BeliefParticle(state, weight))
        if not advanced:
            raise RuntimeError("observed card is incompatible with every belief particle")

        raw = tuple(advanced)
        ess = effective_sample_size([item.weight for item in raw])
        fraction = (
            self.minimum_effective_fraction
            if resample_fraction is None
            else float(resample_fraction)
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("resample fraction must be in (0, 1]")
        threshold = max(1.0, fraction * len(raw))
        resampled = ess < threshold and len(raw) > 1
        if resampled:
            final = self._systematic_resample(
                raw,
                seed=self.seed + self.step * 65_537 + typed.id,
                target_count=max(1, before),
            )
            ess = effective_sample_size([item.weight for item in final])
        else:
            final = raw

        self.step += 1
        self.particles = final
        update = SequentialBeliefUpdate(
            self.step,
            actor,
            typed,
            before,
            len(final),
            ess,
            resampled,
        )
        self.history = (*self.history, update)
        return update

    def card_location_probabilities(self) -> dict[str, dict[str, float]]:
        return self.as_particle_belief().card_location_probabilities()

    def to_dict(self, *, include_card_locations: bool = True) -> dict[str, Any]:
        belief = self.as_particle_belief()
        return {
            "observer": self.observer.name.lower(),
            "method": belief.method,
            "particleCount": len(self.particles),
            "effectiveSampleSize": belief.effective_sample_size,
            "step": self.step,
            "profileId": self.profile_id,
            "profileVersion": self.profile_version,
            "history": [item.to_dict() for item in self.history],
            "cardLocations": self.card_location_probabilities() if include_card_locations else None,
        }
