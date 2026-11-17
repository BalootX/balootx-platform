"""Pure deterministic natural-Laffa generation for Foundry workers.

This module composes the existing authoritative engine and observer-safe agent
policies.  It intentionally does not implement a second set of Baloot rules.
Every auction attempt is retained as its own complete trace; an all-pass deal
is never mislabeled as a playable Laffa.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.agents.profiles import (
    AdaptiveBiddingStrategy,
    AdaptivePlayStrategy,
    AgentProfile,
    AgentProfileRegistry,
    actor_observer_bid_public_state_v1,
    canonical_plain_profile,
    primary_bid_decision_score_authority_v1,
    primary_play_decision_score_authority_v1,
    validate_primary_bid_decision_score_trace_v1,
    validate_primary_play_decision_score_trace_v1,
)
from baloot_platform.agents.qualification import FoundryAgentReleaseV1
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.bidding import legal_bid_choices
from baloot_platform.analysis.evaluator import AnalysisPolicy
from baloot_platform.analysis.search_backend import ExactSearchBackend
from baloot_platform.config import build_bundle
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.structural_ir import StructuralDomainIRV1
from baloot_platform.plugins.base import (
    PluginDescriptor,
    ResearchAuctionSelectorPlugin,
    ResearchPlaySelectorPlugin,
)
from baloot_platform.version import __version__

from .decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from .evidence_v3 import DecisionLedgerV1, canonical_sha256
from .finalist_agent_agreement import (
    FinalistAgentAgreementV1,
    trace_analysis_play_budget_v1,
    validate_finalist_primary_profile_v1,
    validate_trace_analysis_authority_v1,
)
from .trace import CompleteGameTraceV1, TraceEventV1


_SEED_POLICY = "foundry_sha256_uint64_v1"


def derive_foundry_seed_v1(*parts: Any) -> int:
    """Derive a stable unsigned 64-bit seed without process-global RNG state."""
    return int(canonical_sha256({"policy": _SEED_POLICY, "parts": list(parts)})[:16], 16)


def bidding_machine_projection_v1(machine: BiddingMachine) -> dict[str, Any]:
    return {
        "round": machine.round_index,
        "actor": machine.actor.name.lower() if machine.actor is not None else None,
        "inRecall": machine.in_recall,
        "legalActions": [item.to_dict() for item in legal_bid_choices(machine)],
        "history": [item.to_dict() for item in machine.records],
    }


def observer_safe_bid_state_v1(
    state: RoundState, machine: BiddingMachine, actor: Seat
) -> dict[str, Any]:
    return actor_observer_bid_public_state_v1(state, machine, actor)


def canonical_bid_complete_state_v1(
    state: RoundState, machine: BiddingMachine
) -> dict[str, Any]:
    return {
        "state": state.to_dict(reveal_hands=True),
        "bidding": bidding_machine_projection_v1(machine),
    }


def _play_public_state(state: RoundState, actor: Seat) -> dict[str, Any]:
    from baloot_platform.analysis.models import InformationState

    return InformationState.from_complete(state, actor).to_dict()


@dataclass(frozen=True, slots=True)
class NaturalLaffaSimulationResultV1:
    root_seed: int
    attempt_traces: tuple[CompleteGameTraceV1, ...]
    publishable_laffa: bool
    coverage_cells: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        root_seed: int,
        attempt_traces: Iterable[CompleteGameTraceV1],
        publishable_laffa: bool,
        coverage_cells: Iterable[str],
    ) -> "NaturalLaffaSimulationResultV1":
        traces = tuple(attempt_traces)
        if not traces:
            raise ValueError("a natural simulation must retain at least one auction trace")
        publishable = bool(publishable_laffa)
        final_state = traces[-1].trusted_final_state
        if publishable and str(final_state.get("phase")) != GamePhase.SCORED.value:
            raise ValueError("a publishable Laffa must end in an authoritative scored state")
        if not publishable and str(final_state.get("phase")) != GamePhase.COMPLETE.value:
            raise ValueError("a non-publishable simulation must end in an all-pass complete state")
        coverage = tuple(sorted(set(str(item) for item in coverage_cells if str(item))))
        core = {
            "schemaVersion": 1,
            "rootSeed": int(root_seed),
            "attemptTraceHashes": [item.sha256 for item in traces],
            "publishableLaffa": publishable,
            "coverageCells": list(coverage),
        }
        return cls(int(root_seed), traces, publishable, coverage, canonical_sha256(core))

    @property
    def final_trace(self) -> CompleteGameTraceV1:
        return self.attempt_traces[-1]

    def to_batch_output(self) -> dict[str, Any]:
        """Return the stable callback shape consumed by the batch executor."""
        return {
            "traceHash": self.final_trace.sha256,
            "traceHashes": [item.sha256 for item in self.attempt_traces],
            "candidateHashes": (),
            "coverageCells": self.coverage_cells,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Return compact lineage without duplicating content-addressed traces."""
        return {
            "schemaVersion": 1,
            "rootSeed": self.root_seed,
            "attemptCount": len(self.attempt_traces),
            "attemptTraceHashes": [item.sha256 for item in self.attempt_traces],
            "publishableLaffa": self.publishable_laffa,
            "coverageCells": list(self.coverage_cells),
            "sha256": self.sha256,
        }

    def to_dict(self, *, trusted: bool = False) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "rootSeed": self.root_seed,
            "attemptCount": len(self.attempt_traces),
            "attemptTraces": [item.to_dict(trusted=trusted) for item in self.attempt_traces],
            "publishableLaffa": self.publishable_laffa,
            "coverageCells": list(self.coverage_cells),
            "sha256": self.sha256,
        }


class NaturalLaffaSimulatorV1:
    """Generate one legal natural Laffa with equal, frozen seat policies.

    This is a throughput/reference generator, not an optimal-play claim.  The
    supplied agent release is used identically for every seat and every action
    still passes through the engine's legal-action boundary.
    """

    descriptor = PluginDescriptor(
        id="natural_laffa_generator_v1",
        kind="generator",
        version="1.2.0",
        description="Deterministic full-auction and full-play natural Laffa generator",
        capabilities=(
            "complete_bidding",
            "complete_play",
            "all_seats",
            "equal_agent_release",
            "all_pass_lineage",
            "complete_trace",
            "deterministic",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=2048,
    )

    def __init__(
        self,
        rules: RuleBook,
        convention: Mapping[str, Any],
        *,
        bundle_hash: str,
        configuration_snapshot: Mapping[str, Any],
        structural_domain: StructuralDomainIRV1 | None = None,
        play_domain: PlayDomainIRV1 | None = None,
        agent_profile: AgentProfile,
        agent_release: str | FoundryAgentReleaseV1 | None = None,
        generation_tier: str = "throughput_reference",
        engine_version: str = __version__,
        maximum_deal_attempts: int = 8,
        all_pass_dealer_policy: str = "counter_clockwise_next",
        exact_search_backend: ExactSearchBackend | None = None,
        decision_agreement: FinalistAgentAgreementV1 | None = None,
        analysis_policy: AnalysisPolicy | None = None,
        belief_policy: BeliefPolicy | None = None,
        match_value_profile: Mapping[str, Any] | None = None,
        trace_analysis_profile: Any | None = None,
        analysis_identity_hash: str | None = None,
    ):
        if len(str(bundle_hash)) != 64:
            raise ValueError("natural simulation requires an exact bundle SHA-256")
        snapshot = copy.deepcopy(dict(configuration_snapshot))
        declared_snapshot_hash = str(snapshot.pop("sha256", ""))
        if int(snapshot.get("schemaVersion", 0)) != 1:
            raise ValueError("natural simulation requires configuration snapshot schema version 1")
        if str(snapshot.get("bundleHash") or "") != str(bundle_hash):
            raise ValueError("configuration snapshot does not match the simulation bundle")
        rules_component = dict(dict(snapshot.get("components") or {}).get("rules") or {})
        if (
            str(rules_component.get("sha256") or "") != rules.sha256
            or canonical_sha256(rules_component.get("payload")) != rules.sha256
        ):
            raise ValueError("configuration snapshot does not contain the exact simulation rules")
        calculated_snapshot_hash = canonical_sha256(snapshot)
        if declared_snapshot_hash != calculated_snapshot_hash:
            raise ValueError("configuration snapshot checksum mismatch")
        snapshot["sha256"] = declared_snapshot_hash
        if all(
            item is not None
            for item in (
                analysis_policy,
                belief_policy,
                match_value_profile,
            )
        ):
            assert match_value_profile is not None
            plain_match_value_profile = canonical_plain_profile(
                match_value_profile
            )
            components = dict(snapshot.get("components") or {})
            missing_policy_components = sorted(
                {"analysis", "belief", "match_value"} - set(components)
            )
            if missing_policy_components:
                raise ValueError(
                    "explicit simulation policy is absent from the frozen "
                    f"configuration snapshot: {missing_policy_components}"
                )
            policy_payloads: dict[str, dict[str, Any]] = {}
            for family in ("analysis", "belief", "match_value"):
                component = dict(components[family] or {})
                payload = copy.deepcopy(dict(component.get("payload") or {}))
                if str(component.get("sha256") or "") != canonical_sha256(
                    payload
                ):
                    raise ValueError(
                        f"frozen {family} policy component checksum mismatch"
                    )
                policy_payloads[family] = payload
            frozen_belief = BeliefPolicy.from_profile(
                policy_payloads["belief"]
            )
            frozen_analysis = AnalysisPolicy.from_profile(
                policy_payloads["analysis"],
                belief_policy=frozen_belief,
            )
            if belief_policy != frozen_belief:
                raise ValueError(
                    "explicit belief policy does not match the frozen bundle"
                )
            if analysis_policy != frozen_analysis:
                raise ValueError(
                    "explicit analysis policy does not match the frozen bundle"
                )
            if canonical_sha256(
                plain_match_value_profile
            ) != canonical_sha256(policy_payloads["match_value"]):
                raise ValueError(
                    "explicit match-value profile does not match the frozen bundle"
                )
            match_value_profile = plain_match_value_profile
        if int(maximum_deal_attempts) < 1:
            raise ValueError("maximum deal attempts must be positive")
        if all_pass_dealer_policy not in {"counter_clockwise_next", "same"}:
            raise ValueError("unsupported all-pass dealer policy")
        if generation_tier not in {
            "throughput_reference",
            "balanced_discovery",
            "expert_validation",
        }:
            raise ValueError("unsupported Foundry generation tier")
        (
            frozen_analysis_profile,
            trace_analysis_profile_hash,
            frozen_analysis_identity_hash,
        ) = validate_trace_analysis_authority_v1(
            trace_analysis_profile,
            analysis_identity_hash,
            exact_search_backend=exact_search_backend,
        )
        if frozen_analysis_profile is not None:
            validate_finalist_primary_profile_v1(
                agent_profile,
                frozen_analysis_profile,
            )
        self.rules = rules
        compiled_structural = rules.structural_domain.require_integrity()
        if structural_domain is not None:
            structural_domain.require_integrity()
        if (
            structural_domain is not None
            and structural_domain.sha256 != compiled_structural.sha256
        ):
            raise ValueError("simulation structural domain does not match its rules")
        self.structural_domain = structural_domain or compiled_structural
        self.structural_domain.validate_legacy_all_pass_dealer_policy(
            all_pass_dealer_policy
        )
        compiled_play = rules.play_domain.require_integrity(
            self.structural_domain
        )
        if play_domain is not None:
            play_domain.require_integrity(self.structural_domain)
        if play_domain is not None and play_domain != compiled_play:
            raise ValueError("simulation play domain does not match its rules")
        self.play_domain = play_domain or compiled_play
        self.convention = dict(convention)
        self.bundle_hash = str(bundle_hash)
        self.configuration_snapshot = snapshot
        self.configuration_snapshot_hash = declared_snapshot_hash
        self.agent_profile = agent_profile
        self.trace_analysis_profile = frozen_analysis_profile
        self.trace_analysis_profile_hash = trace_analysis_profile_hash
        self.analysis_identity_hash = frozen_analysis_identity_hash
        self._primary_analysis_budget = (
            trace_analysis_play_budget_v1(frozen_analysis_profile)
            if frozen_analysis_profile is not None
            else None
        )
        self.profile_hash = canonical_sha256(agent_profile.to_dict())
        self.agent_qualification: FoundryAgentReleaseV1 | None = None
        if isinstance(agent_release, FoundryAgentReleaseV1):
            if agent_release.profile_hash != self.profile_hash:
                raise ValueError("agent release profile does not match the simulator profile")
            self.agent_qualification = agent_release
            self.agent_release = agent_release.sha256
        else:
            self.agent_release = str(agent_release or f"{agent_profile.id}@{self.profile_hash[:16]}")
        if not self.agent_release.strip():
            raise ValueError("agent release identity cannot be empty")
        if generation_tier == "expert_validation" and (
            self.agent_qualification is None
            or not self.agent_qualification.expert_generation_allowed
        ):
            raise ValueError("expert validation generation requires a fully qualified agent release")
        self.generation_tier = generation_tier
        self.engine_version = str(engine_version)
        self.maximum_deal_attempts = int(maximum_deal_attempts)
        self.all_pass_dealer_policy = all_pass_dealer_policy
        self.exact_search_backend = exact_search_backend
        analysis_convention = (
            self.convention
            if frozen_analysis_profile is None
            or bool(frozen_analysis_profile.use_conventions)
            else {}
        )
        self.play_strategy = AdaptivePlayStrategy(
            self.agent_profile,
            analysis_convention,
            exact_search_backend=self.exact_search_backend,
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            match_value_profile=match_value_profile,
        )
        self.analysis_policy = self.play_strategy.analysis_policy
        self.belief_policy = self.play_strategy.belief_policy
        self._match_value_profile = copy.deepcopy(
            self.play_strategy.match_value_profile
        )
        self._analysis_policy_authority = copy.deepcopy(
            self.play_strategy.policy_authority
        )
        if (
            decision_agreement is not None
            and decision_agreement.profile.base_agent_profile_hash != self.profile_hash
        ):
            raise ValueError("decision agreement does not govern the simulator agent profile")
        if decision_agreement is not None:
            agreement_pair = (
                decision_agreement.trace_analysis_profile_hash,
                decision_agreement.analysis_identity_hash,
            )
            simulator_pair = (
                self.trace_analysis_profile_hash,
                self.analysis_identity_hash,
            )
            if agreement_pair != simulator_pair:
                raise ValueError(
                    "decision agreement does not use the simulator trace-analysis authority"
                )
            if (
                self._primary_analysis_budget is not None
                and dict(decision_agreement.primary_analysis_budget)
                != self._primary_analysis_budget
            ):
                raise ValueError(
                    "decision agreement does not use the simulator trace-analysis budget"
                )
        self.decision_agreement = decision_agreement

    @classmethod
    def from_root(
        cls,
        root: str | Path | None = None,
        *,
        profile_id: str = "strong",
        maximum_deal_attempts: int = 8,
    ) -> "NaturalLaffaSimulatorV1":
        bundle = build_bundle(Path(root) if root is not None else None)
        registry = AgentProfileRegistry.from_mapping(
            bundle.component("agents")
        )
        game = bundle.component("game")
        belief_policy = BeliefPolicy.from_profile(bundle.component("belief"))
        analysis_policy = AnalysisPolicy.from_profile(
            bundle.component("analysis"),
            belief_policy=belief_policy,
        )
        return cls(
            bundle.rules,
            bundle.convention,
            bundle_hash=bundle.sha256,
            configuration_snapshot=bundle.configuration_snapshot(),
            agent_profile=registry.get(profile_id),
            maximum_deal_attempts=maximum_deal_attempts,
            all_pass_dealer_policy=str(
                game.get("all_pass", {}).get("dealer_policy", "counter_clockwise_next")
            ),
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            match_value_profile=bundle.component("match_value"),
        )

    @property
    def equal_seat_agents(self) -> tuple[str, str, str, str]:
        return (self.agent_release,) * 4

    @property
    def match_value_profile(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._match_value_profile)

    @property
    def analysis_policy_authority(self) -> dict[str, Any]:
        return copy.deepcopy(self._analysis_policy_authority)

    @property
    def primary_analysis_budget(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._primary_analysis_budget)

    def primary_decision_seed(
        self,
        *,
        public_prefix_hash: str,
        decision_kind: str,
        acting_seat: Seat | str,
    ) -> int:
        if self.analysis_identity_hash is None:
            raise ValueError("primary actor-observer seed requires frozen analysis authority")
        return derive_actor_observer_cohort_seed_v1(
            public_prefix_hash=public_prefix_hash,
            decision_kind=decision_kind,
            acting_seat=acting_seat,
            cohort_authority_hash=self.analysis_identity_hash,
            cohort_index=0,
            governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
        )

    def _decision_seed(
        self,
        *,
        root_seed: int,
        attempt_index: int,
        ordinal: int,
        decision_kind: str,
        acting_seat: Seat,
        public_prefix_hash: str,
    ) -> int:
        if self.analysis_identity_hash is not None:
            return self.primary_decision_seed(
                public_prefix_hash=public_prefix_hash,
                decision_kind=decision_kind,
                acting_seat=acting_seat,
            )
        return derive_foundry_seed_v1(
            root_seed,
            attempt_index,
            ordinal,
            decision_kind,
            int(acting_seat),
        )

    def _policy_with_analysis_identity(
        self,
        policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(policy)
        if self.trace_analysis_profile_hash is None:
            return result
        assert self.analysis_identity_hash is not None
        for key, expected in (
            ("traceAnalysisProfileHash", self.trace_analysis_profile_hash),
            ("analysisIdentityHash", self.analysis_identity_hash),
        ):
            actual = result.get(key)
            if actual is not None and str(actual) != expected:
                raise ValueError(f"decision policy contains drifted {key}")
            result[key] = expected
        return result

    def _choose_primary_play(
        self,
        strategy: AdaptivePlayStrategy,
        engine: BalootEngine,
        state: RoundState,
        legal_cards: tuple[Card, ...],
        *,
        seed: int,
        cancel_check: Callable[[], bool] | None,
        public_prefix_hash: str,
    ) -> tuple[Card, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "seed": seed,
            "cancel_check": cancel_check,
        }
        score_authority: dict[str, Any] | None = None
        if self._primary_analysis_budget is not None:
            kwargs["analysis_budget"] = self._primary_analysis_budget
            if state.current_player is None:
                raise ValueError("primary play evidence requires an active actor")
            assert self.trace_analysis_profile is not None
            assert self.trace_analysis_profile_hash is not None
            assert self.analysis_identity_hash is not None
            score_authority = primary_play_decision_score_authority_v1(
                actor=state.current_player,
                public_prefix_hash=public_prefix_hash,
                legal_cards=legal_cards,
                rules_hash=engine.rules.sha256,
                analysis_identity_hash=self.analysis_identity_hash,
                trace_analysis_profile_hash=self.trace_analysis_profile_hash,
                cohort_index=0,
                cohort_seed=seed,
                objective=str(self.trace_analysis_profile.objective),
                backend=str(self.trace_analysis_profile.search_backend_id),
            )
            kwargs["primary_decision_score_authority"] = score_authority
        card, trace = strategy.choose_with_trace(
            engine,
            state,
            legal_cards,
            **kwargs,
        )
        if score_authority is not None:
            validate_primary_play_decision_score_trace_v1(
                trace,
                score_authority,
                selected_action=card.to_dict(),
                forced=len(legal_cards) == 1,
            )
        return card, trace

    def _choose_primary_bid(
        self,
        strategy: AdaptiveBiddingStrategy,
        state: RoundState,
        machine: BiddingMachine,
        legal_choices: tuple[Any, ...],
        *,
        seed: int,
        cancel_check: Callable[[], bool] | None,
        public_prefix_hash: str,
    ) -> tuple[str, Any | None, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "seed": seed,
            "cancel_check": cancel_check,
        }
        score_authority: dict[str, Any] | None = None
        if self.trace_analysis_profile is not None:
            if machine.actor is None:
                raise ValueError("primary bid evidence requires an active actor")
            assert self.trace_analysis_profile_hash is not None
            assert self.analysis_identity_hash is not None
            score_authority = primary_bid_decision_score_authority_v1(
                actor=machine.actor,
                public_prefix_hash=public_prefix_hash,
                legal_choices=legal_choices,
                rules_hash=self.rules.sha256,
                analysis_identity_hash=self.analysis_identity_hash,
                trace_analysis_profile_hash=self.trace_analysis_profile_hash,
                cohort_index=0,
                cohort_seed=seed,
                backend=str(self.trace_analysis_profile.search_backend_id),
            )
            kwargs["primary_decision_score_authority"] = score_authority
        action, trump, trace = strategy.choose_with_trace(
            state,
            machine,
            self.rules,
            **kwargs,
        )
        if score_authority is not None:
            selected = {
                "action": action,
                "trumpSuit": trump.name.lower() if trump is not None else None,
            }
            validate_primary_bid_decision_score_trace_v1(
                trace,
                score_authority,
                selected_action=selected,
                forced=len(legal_choices) == 1,
            )
        return action, trump, trace

    def generate(self, seed: int, **kwargs: Any) -> NaturalLaffaSimulationResultV1:
        return self.simulate(seed, **kwargs)

    def resimulate_exact_deal(
        self,
        parent: CompleteGameTraceV1,
        *,
        match_score: tuple[int, int],
        round_number: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CompleteGameTraceV1:
        """Replay one frozen deal under a new match context.

        The original root/attempt seed policy is preserved so policy randomness
        and dealt cards remain comparable.  An all-pass result is returned as a
        complete trace and must not be treated as a playable contextualized
        Laffa by callers.
        """
        generation = dict(parent.trusted_final_state.get("generation") or {})
        required = {
            "rootSeed", "attemptIndex", "attemptSeed", "dealer",
            "rulesHash", "bundleHash", "agentRelease", "agentProfileHash",
            "configurationSnapshotHash",
        }
        missing = sorted(required - set(generation))
        if missing:
            raise ValueError(f"parent trace lacks exact generation provenance: {missing}")
        if str(generation["rulesHash"]) != self.rules.sha256:
            raise ValueError("parent trace rules do not match the context simulator")
        if str(generation["bundleHash"]) != self.bundle_hash:
            raise ValueError("parent trace bundle does not match the context simulator")
        if str(generation.get("configurationSnapshotHash") or "") != self.configuration_snapshot_hash:
            raise ValueError("parent trace configuration snapshot does not match the context simulator")
        if str(generation["agentRelease"]) != self.agent_release:
            raise ValueError("parent trace agent release does not match the context simulator")
        if str(generation["agentProfileHash"]) != self.profile_hash:
            raise ValueError("parent trace agent profile does not match the context simulator")
        trace, _purchased, _coverage = self._simulate_attempt(
            root_seed=int(generation["rootSeed"]),
            attempt_index=int(generation["attemptIndex"]),
            attempt_seed=int(generation["attemptSeed"]),
            dealer=Seat.parse(str(generation["dealer"])),
            match_score=tuple(int(item) for item in match_score),
            round_number=int(round_number),
            cancel_check=cancel_check,
            deck_order=self._constructed_deck_order(parent),
            deal_generator_provenance=generation.get("dealGenerator"),
        )
        original_commitment = str(parent.seed_provenance.get("deckCommitment") or "")
        if trace.seed_provenance.get("deckCommitment") != original_commitment:
            raise AssertionError("exact context replay changed the dealt cards")
        return trace

    def resimulate_exact_deal_as_finalist_descendant(
        self,
        parent: CompleteGameTraceV1,
        *,
        match_score: tuple[int, int],
        round_number: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CompleteGameTraceV1:
        """Replay one frozen deal with this validator agent as a signed descendant.

        Unlike context replay, the validator agent is intentionally allowed to
        differ from the discovery agent. Rules, the exact governed bundle,
        configuration snapshot, seed, dealer, and dealt cards may not differ.
        """
        return self._resimulate_exact_deal_as_agent_descendant(
            parent,
            match_score=match_score,
            round_number=round_number,
            cancel_check=cancel_check,
            lineage_kind="finalist_validation_descendant",
        )

    def resimulate_exact_deal_as_candidate_descendant(
        self,
        parent: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CompleteGameTraceV1:
        """Replay a retained deal with a stronger governed discovery policy."""

        generation = dict(parent.trusted_final_state.get("generation") or {})
        match_score = tuple(
            int(item) for item in generation.get("matchScore", (0, 0))
        )
        if len(match_score) != 2:
            raise ValueError("candidate resimulation requires a valid match score")
        return self._resimulate_exact_deal_as_agent_descendant(
            parent,
            match_score=(match_score[0], match_score[1]),
            round_number=int(generation.get("roundNumber", 1)),
            cancel_check=cancel_check,
            lineage_kind="candidate_resimulation_descendant",
        )

    def resimulate_constructed_play_as_counterfactual_descendant(
        self,
        parent: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CompleteGameTraceV1:
        """Keep a constructed public setup fixed and vary only play policy.

        This is a contextualized counterfactual, not a natural replay. The
        constructed deck, public auction, escalation/project prefix, and first
        play information state are checksum-locked before lineups are compared.
        """

        generation = dict(parent.trusted_final_state.get("generation") or {})
        if not generation.get("dealGenerator"):
            raise ValueError(
                "play-only counterfactual requires a constructed complete-play parent"
            )
        self._constructed_play_source_key(generation)
        required = {
            "rootSeed", "attemptIndex", "attemptSeed", "dealer",
            "rulesHash", "bundleHash", "configurationSnapshotHash",
        }
        missing = sorted(required - set(generation))
        if missing:
            raise ValueError(
                f"constructed parent lacks exact generation provenance: {missing}"
            )
        if str(generation["rulesHash"]) != self.rules.sha256:
            raise ValueError("constructed parent rules do not match the simulator")
        if str(generation["bundleHash"]) != self.bundle_hash:
            raise ValueError("constructed parent bundle does not match the simulator")
        if (
            str(generation["configurationSnapshotHash"])
            != self.configuration_snapshot_hash
        ):
            raise ValueError(
                "constructed parent configuration snapshot does not match the simulator"
            )

        scripted_bids = tuple(
            {
                "actor": str(event["seat"]),
                "roundIndex": int(event["round"]),
                "action": str(event["action"]),
                "trumpSuit": event.get("trumpSuit"),
            }
            for event in parent.trusted_final_state.get("events") or ()
            if event.get("type") == "bid_made"
        )
        if not scripted_bids:
            raise ValueError("constructed parent has no authoritative auction prefix")
        parent_prefix_hash = self._constructed_pre_play_prefix_hash(parent)
        match_score = tuple(int(item) for item in generation.get("matchScore", (0, 0)))
        if len(match_score) != 2:
            raise ValueError("constructed parent has an invalid match score")
        trace, _purchased, _coverage = self._simulate_attempt(
            root_seed=int(generation["rootSeed"]),
            attempt_index=int(generation["attemptIndex"]),
            attempt_seed=int(generation["attemptSeed"]),
            dealer=Seat.parse(str(generation["dealer"])),
            match_score=(match_score[0], match_score[1]),
            round_number=int(generation.get("roundNumber", 1)),
            cancel_check=cancel_check,
            parent_trace_hash=parent.sha256,
            parent_game_hash=parent.game_hash,
            lineage_kind="constructed_play_counterfactual_descendant",
            deck_order=self._constructed_deck_order(parent),
            deal_generator_provenance=generation.get("dealGenerator"),
            scripted_bids=scripted_bids,
            scripted_escalations=tuple(
                copy.deepcopy(dict(item))
                for item in generation.get("scriptedEscalations") or ()
            ),
            counterfactual_provenance={
                "schemaVersion": 1,
                "scope": "play_policy_only",
                "parentPrePlayPrefixHash": parent_prefix_hash,
            },
        )
        child_prefix_hash = self._constructed_pre_play_prefix_hash(trace)
        if child_prefix_hash != parent_prefix_hash:
            raise AssertionError(
                "play-only counterfactual changed its auction/escalation/project prefix"
            )
        if trace.seed_provenance.get("deckCommitment") != parent.seed_provenance.get(
            "deckCommitment"
        ):
            raise AssertionError("play-only counterfactual changed the dealt cards")
        return trace

    @staticmethod
    def _constructed_play_source_key(generation: Mapping[str, Any]) -> str:
        """Return the one governed play-construction provenance key.

        Complete scripted play and the actor-safe strategic selector are the
        only construction modes currently compatible with this replay helper.
        Their provenance shapes stay distinct and an ambiguous or malformed
        parent fails closed.
        """

        compatible_keys = ("scriptedPlayCompiler", "strategicPlaySelector")
        present = tuple(key for key in compatible_keys if key in generation)
        if len(present) != 1:
            raise ValueError(
                "play-only counterfactual requires exactly one governed play source"
            )
        key = present[0]
        value = generation[key]
        if not isinstance(value, Mapping) or int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("constructed parent play-source provenance is invalid")

        if key == "scriptedPlayCompiler":
            required = ("pluginId", "modelHash", "planHash")
            if any(not str(value.get(field) or "").strip() for field in required):
                raise ValueError("constructed parent scripted-play provenance is incomplete")
            return key

        plugin = value.get("plugin")
        profile = value.get("profile")
        if not isinstance(plugin, Mapping) or not isinstance(profile, Mapping):
            raise ValueError("constructed parent strategic-play provenance is incomplete")
        if (
            str(plugin.get("kind") or "") != "research_play_selector"
            or not str(plugin.get("id") or "").strip()
            or not str(plugin.get("version") or "").strip()
            or plugin.get("deterministic") is not True
            or int(profile.get("schemaVersion", 0)) != 1
            or not str(profile.get("sha256") or "").strip()
        ):
            raise ValueError("constructed parent strategic-play provenance is invalid")
        return key

    @staticmethod
    def _constructed_pre_play_prefix_hash(trace: CompleteGameTraceV1) -> str:
        events: list[dict[str, Any]] = []
        for event in trace.events:
            if event.event_type == "card_played":
                break
            events.append(event.to_dict())
        first_play = next(
            (
                record
                for record in trace.decision_ledger.records
                if record.decision_kind == "play"
            ),
            None,
        )
        if first_play is None:
            raise ValueError("play-only counterfactual requires a played Laffa")
        return canonical_sha256({
            "schemaVersion": 2,
            "eventsBeforePlay": events,
            "firstPlay": {
                "ordinal": first_play.ordinal,
                "actingSeat": first_play.acting_seat.name.lower(),
                "publicPrefixHash": first_play.public_prefix_hash,
                "completeStateHash": first_play.complete_state_hash,
                "legalActions": [dict(item) for item in first_play.legal_actions],
            },
        })

    def _resimulate_exact_deal_as_agent_descendant(
        self,
        parent: CompleteGameTraceV1,
        *,
        match_score: tuple[int, int],
        round_number: int,
        cancel_check: Callable[[], bool] | None,
        lineage_kind: str,
    ) -> CompleteGameTraceV1:
        if lineage_kind not in {
            "candidate_resimulation_descendant",
            "finalist_validation_descendant",
        }:
            raise ValueError("unsupported same-deal descendant lineage")
        generation = dict(parent.trusted_final_state.get("generation") or {})
        required = {
            "rootSeed", "attemptIndex", "attemptSeed", "dealer",
            "rulesHash", "bundleHash", "configurationSnapshotHash",
        }
        missing = sorted(required - set(generation))
        if missing:
            raise ValueError(
                f"parent trace lacks exact generation provenance: {missing}"
            )
        if str(generation["rulesHash"]) != self.rules.sha256:
            raise ValueError("parent trace rules do not match the descendant simulator")
        if str(generation["bundleHash"]) != self.bundle_hash:
            raise ValueError("parent trace bundle does not match the descendant simulator")
        if str(generation["configurationSnapshotHash"]) != self.configuration_snapshot_hash:
            raise ValueError(
                "parent trace configuration snapshot does not match the descendant simulator"
            )
        trace, _purchased, _coverage = self._simulate_attempt(
            root_seed=int(generation["rootSeed"]),
            attempt_index=int(generation["attemptIndex"]),
            attempt_seed=int(generation["attemptSeed"]),
            dealer=Seat.parse(str(generation["dealer"])),
            match_score=tuple(int(item) for item in match_score),
            round_number=int(round_number),
            cancel_check=cancel_check,
            parent_trace_hash=parent.sha256,
            parent_game_hash=parent.game_hash,
            lineage_kind=lineage_kind,
            deck_order=self._constructed_deck_order(parent),
            deal_generator_provenance=generation.get("dealGenerator"),
        )
        if trace.seed_provenance.get("deckCommitment") != parent.seed_provenance.get(
            "deckCommitment"
        ):
            raise AssertionError("same-deal descendant changed the dealt cards")
        return trace

    @staticmethod
    def _constructed_deck_order(
        parent: CompleteGameTraceV1,
    ) -> tuple[int, ...] | None:
        generation = dict(parent.trusted_final_state.get("generation") or {})
        if not generation.get("dealGenerator"):
            return None
        shuffle = dict(parent.trusted_final_state.get("shuffle") or {})
        order = tuple(int(item) for item in shuffle.get("deckOrder") or ())
        if len(order) != 32 or sorted(order) != list(range(32)):
            raise ValueError("constructed parent trace lacks its complete deck order")
        return order

    def simulate_preordered_deck(
        self,
        seed: int,
        deck_order: tuple[int, ...] | list[int],
        *,
        deal_generator_provenance: Mapping[str, Any],
        dealer: Seat = Seat.EAST,
        match_score: tuple[int, int] = (0, 0),
        round_number: int = 1,
        scripted_bids: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
        scripted_escalations: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
        scripted_plays: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
        scripted_play_provenance: Mapping[str, Any] | None = None,
        bid_selector: ResearchAuctionSelectorPlugin | None = None,
        play_selector: ResearchPlaySelectorPlugin | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> NaturalLaffaSimulationResultV1:
        """Run a governed constructed deck through the full engine and agents."""
        root_seed = int(seed)
        if root_seed < 0:
            raise ValueError("simulation seed must be non-negative")
        provenance = copy.deepcopy(dict(deal_generator_provenance))
        if not provenance or not provenance.get("profileHash"):
            raise ValueError("constructed simulation requires generator provenance")
        trace, purchased, attempt_coverage = self._simulate_attempt(
            root_seed=root_seed,
            attempt_index=0,
            attempt_seed=root_seed,
            dealer=dealer,
            match_score=match_score,
            round_number=round_number,
            cancel_check=cancel_check,
            lineage_kind="guided_constraint_attempt",
            deck_order=tuple(int(item) for item in deck_order),
            deal_generator_provenance=provenance,
            scripted_bids=(
                tuple(copy.deepcopy(dict(item)) for item in scripted_bids)
                if scripted_bids is not None else None
            ),
            scripted_escalations=(
                tuple(copy.deepcopy(dict(item)) for item in scripted_escalations)
                if scripted_escalations is not None else None
            ),
            scripted_plays=(
                tuple(copy.deepcopy(dict(item)) for item in scripted_plays)
                if scripted_plays is not None else None
            ),
            scripted_play_provenance=(
                copy.deepcopy(dict(scripted_play_provenance))
                if scripted_play_provenance is not None else None
            ),
            bid_selector=bid_selector,
            play_selector=play_selector,
        )
        coverage = {
            *attempt_coverage,
            "source:guided_constraint",
            f"agent:{self.agent_release}",
            f"generation_tier:{self.generation_tier}",
            f"deal_generator:{provenance.get('pluginId', 'unknown')}",
        }
        return NaturalLaffaSimulationResultV1.build(
            root_seed=root_seed,
            attempt_traces=(trace,),
            publishable_laffa=purchased,
            coverage_cells=coverage,
        )

    def simulate(
        self,
        seed: int,
        *,
        dealer: Seat = Seat.EAST,
        match_score: tuple[int, int] = (0, 0),
        round_number: int = 1,
        cancel_check: Callable[[], bool] | None = None,
    ) -> NaturalLaffaSimulationResultV1:
        root_seed = int(seed)
        if root_seed < 0:
            raise ValueError("simulation seed must be non-negative")
        attempts: list[CompleteGameTraceV1] = []
        coverage: set[str] = {
            "source:organic",
            f"agent:{self.agent_release}",
            f"generation_tier:{self.generation_tier}",
        }
        current_dealer = dealer

        for attempt_index in range(self.maximum_deal_attempts):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("natural Laffa simulation cancelled")
            attempt_seed = root_seed + attempt_index
            trace, purchased, attempt_coverage = self._simulate_attempt(
                root_seed=root_seed,
                attempt_index=attempt_index,
                attempt_seed=attempt_seed,
                dealer=current_dealer,
                match_score=match_score,
                round_number=round_number + attempt_index,
                cancel_check=cancel_check,
            )
            attempts.append(trace)
            coverage.update(attempt_coverage)
            if purchased:
                coverage.add(f"all_pass_attempts:{attempt_index}")
                return NaturalLaffaSimulationResultV1.build(
                    root_seed=root_seed,
                    attempt_traces=attempts,
                    publishable_laffa=True,
                    coverage_cells=coverage,
                )
            coverage.add("auction:all_pass")
            if self.structural_domain.all_pass_redeal:
                current_dealer = self.structural_domain.next_dealer(
                    current_dealer
                )

        coverage.add(f"all_pass_attempts:{self.maximum_deal_attempts}")
        return NaturalLaffaSimulationResultV1.build(
            root_seed=root_seed,
            attempt_traces=attempts,
            publishable_laffa=False,
            coverage_cells=coverage,
        )

    def _simulate_attempt(
        self,
        *,
        root_seed: int,
        attempt_index: int,
        attempt_seed: int,
        dealer: Seat,
        match_score: tuple[int, int],
        round_number: int,
        cancel_check: Callable[[], bool] | None,
        parent_trace_hash: str | None = None,
        parent_game_hash: str | None = None,
        lineage_kind: str = "natural_laffa_attempt",
        deck_order: tuple[int, ...] | None = None,
        deal_generator_provenance: Mapping[str, Any] | None = None,
        scripted_bids: tuple[Mapping[str, Any], ...] | None = None,
        scripted_escalations: tuple[Mapping[str, Any], ...] | None = None,
        scripted_plays: tuple[Mapping[str, Any], ...] | None = None,
        scripted_play_provenance: Mapping[str, Any] | None = None,
        bid_selector: ResearchAuctionSelectorPlugin | None = None,
        play_selector: ResearchPlaySelectorPlugin | None = None,
        counterfactual_provenance: Mapping[str, Any] | None = None,
    ) -> tuple[CompleteGameTraceV1, bool, tuple[str, ...]]:
        if scripted_plays is not None:
            if len(scripted_plays) != 32:
                raise ValueError("scripted play requires exactly 32 card actions")
            if not scripted_play_provenance:
                raise ValueError("scripted play requires compiler provenance")
        if scripted_plays is not None and play_selector is not None:
            raise ValueError("scripted play and a strategic play selector are exclusive")
        if scripted_bids is not None and bid_selector is not None:
            raise ValueError("scripted auction and a strategic auction selector are exclusive")
        engine = BalootEngine(
            self.rules,
            structural_domain=self.structural_domain,
            play_domain=self.play_domain,
        )
        state = engine.create_round(
            seed=attempt_seed,
            dealer=dealer,
            round_number=round_number,
            match_score=match_score,
        )
        if deck_order is None:
            if deal_generator_provenance is not None:
                raise ValueError("deal generator provenance requires a constructed deck")
            engine.shuffle_and_initial_deal(state)
        else:
            if not deal_generator_provenance:
                raise ValueError("a constructed deck requires generator provenance")
            engine.initial_deal_from_preordered(state, deck_order)
        assert state.purchase_card is not None
        state.phase = GamePhase.BIDDING
        machine = BiddingMachine(
            state.dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(self.rules.data),
            state.purchase_card.rank,
        )
        bidding = AdaptiveBiddingStrategy(
            self.agent_profile,
            exact_search_backend=self.exact_search_backend,
        )
        play = self.play_strategy
        raw_decisions: list[dict[str, Any]] = []
        policy_traces: list[dict[str, Any]] = []
        ordinal = 1
        agreement_ordinal = 0
        unresolved_agreements = 0
        scripted_bid_index = 0
        scripted_play_index = 0

        while machine.result is None:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("natural Laffa simulation cancelled")
            actor = machine.actor
            if actor is None:
                raise AssertionError("unfinished auction has no actor")
            legal_choices = tuple(legal_bid_choices(machine))
            legal = tuple(item.to_dict() for item in legal_choices)
            public = observer_safe_bid_state_v1(state, machine, actor)
            public_prefix_hash = canonical_sha256(public)
            complete = canonical_bid_complete_state_v1(state, machine)
            decision_seed = self._decision_seed(
                root_seed=root_seed,
                attempt_index=attempt_index,
                ordinal=ordinal,
                decision_kind="bid",
                acting_seat=actor,
                public_prefix_hash=public_prefix_hash,
            )
            if scripted_bids is not None:
                if scripted_bid_index >= len(scripted_bids):
                    raise ValueError("scripted auction ended before the runtime auction")
                scripted = dict(scripted_bids[scripted_bid_index])
                scripted_bid_index += 1
                if str(scripted.get("actor")) != actor.name.lower():
                    raise ValueError("scripted auction actor differs from runtime")
                if int(scripted.get("roundIndex", 0)) != machine.round_index:
                    raise ValueError("scripted auction round differs from runtime")
                action = str(scripted.get("action") or "")
                trump = (
                    Suit.parse(scripted["trumpSuit"])
                    if scripted.get("trumpSuit") is not None else None
                )
                if action not in {item.action for item in legal_choices}:
                    raise ValueError("scripted auction action is illegal in the runtime")
                if action == "hokum" and not any(
                    item.action == action and item.trump_suit == trump for item in legal_choices
                ):
                    raise ValueError("scripted auction trump is illegal in the runtime")
                policy = {
                    "schemaVersion": 1,
                    "method": "ortools_cp_sat_auction_v1",
                    "selectionReason": "compiled_legal_auction",
                    "qualificationBlocked": False,
                }
            elif bid_selector is not None and len(legal_choices) > 1:
                action, trump, policy = bid_selector.choose_with_trace(
                    state,
                    machine,
                    self.rules,
                    seed=decision_seed,
                    cancel_check=cancel_check,
                )
            elif self.decision_agreement is not None and len(legal_choices) > 1:
                agreement_ordinal += 1
                action, trump, policy = self.decision_agreement.choose_bid(
                    bidding,
                    state,
                    machine,
                    self.rules,
                    legal_choices,
                    seed=decision_seed,
                    agreement_ordinal=agreement_ordinal,
                    cancel_check=cancel_check,
                    public_prefix_hash=public_prefix_hash,
                )
            else:
                action, trump, policy = self._choose_primary_bid(
                    bidding,
                    state,
                    machine,
                    legal_choices,
                    seed=decision_seed,
                    cancel_check=cancel_check,
                    public_prefix_hash=public_prefix_hash,
                )
            policy = self._policy_with_analysis_identity(policy)
            unresolved_agreements += int(bool(policy.get("qualificationBlocked")))
            selected = {
                "action": action,
                "trumpSuit": trump.name.lower() if trump is not None else None,
            }
            raw_decisions.append({
                "ordinal": ordinal,
                "actor": actor.name.lower(),
                "decisionKind": "bid",
                "phase": GamePhase.BIDDING.value,
                "publicState": public,
                "completeStateHash": canonical_sha256(complete),
                "legalActions": legal,
                "selectedAction": selected,
                "resultingEventSequence": None,
            })
            policy_traces.append({"ordinal": ordinal, "trace": policy})
            machine.submit(action, trump=trump)
            ordinal += 1

        assert machine.result is not None
        if scripted_bids is not None and scripted_bid_index != len(scripted_bids):
            raise ValueError("scripted auction contains actions after the runtime auction completed")
        completed = engine.complete_deal_from_bid_result(state, machine.result)
        bid_event_sequences = [event.sequence for event in state.events if event.type == "bid_made"]
        bid_records = [item for item in raw_decisions if item["decisionKind"] == "bid"]
        if len(bid_event_sequences) != len(bid_records):
            raise AssertionError("every auction decision must produce one canonical bid event")
        for item, event_sequence in zip(bid_records, bid_event_sequences, strict=True):
            item["resultingEventSequence"] = event_sequence

        if completed:
            for scripted in scripted_escalations or ():
                engine.apply_escalation(
                    state,
                    seat=Seat.parse(str(scripted["actor"])),
                    action=str(scripted["action"]),
                    mode=(
                        str(scripted["requestedMode"])
                        if scripted.get("requestedMode") is not None
                        else None
                    ),
                )
            engine.prepare_projects(state)
            while state.current_player is not None:
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("natural Laffa simulation cancelled")
                actor = state.current_player
                legal_cards = engine.legal_moves(state, actor)
                if not legal_cards:
                    raise AssertionError("active player has no legal cards")
                public = _play_public_state(state, actor)
                public_prefix_hash = canonical_sha256(public)
                complete_hash = canonical_sha256(state.to_dict(reveal_hands=True))
                decision_seed = self._decision_seed(
                    root_seed=root_seed,
                    attempt_index=attempt_index,
                    ordinal=ordinal,
                    decision_kind="play",
                    acting_seat=actor,
                    public_prefix_hash=public_prefix_hash,
                )
                scripted_ekka: bool | None = None
                if scripted_plays is not None:
                    scripted = dict(scripted_plays[scripted_play_index])
                    scripted_play_index += 1
                    if str(scripted.get("actor")) != actor.name.lower():
                        raise ValueError("scripted play actor differs from runtime")
                    card = Card.parse(scripted.get("card"))
                    if card not in legal_cards:
                        raise ValueError("scripted play card is illegal in the runtime")
                    scripted_ekka = (
                        bool(scripted["ekka"])
                        if scripted.get("ekka") is not None
                        else None
                    )
                    policy = {
                        "schemaVersion": 1,
                        "method": "ortools_cp_sat_complete_play_v1",
                        "selectionReason": "compiled_legal_play",
                        "qualificationBlocked": False,
                        "compilerModelHash": str(
                            scripted_play_provenance.get("modelHash") or ""
                        ),
                        "compilerPlanHash": str(
                            scripted_play_provenance.get("planHash") or ""
                        ),
                    }
                elif play_selector is not None and len(legal_cards) > 1:
                    strategic = play_selector.choose(
                        engine=engine,
                        state=state,
                        legal_cards=legal_cards,
                        seed=decision_seed,
                        cancel_check=cancel_check,
                    )
                    if strategic is None:
                        if self.decision_agreement is not None:
                            agreement_ordinal += 1
                            card, policy = self.decision_agreement.choose_play(
                                play,
                                engine,
                                state,
                                legal_cards,
                                seed=decision_seed,
                                agreement_ordinal=agreement_ordinal,
                                cancel_check=cancel_check,
                                public_prefix_hash=public_prefix_hash,
                            )
                        else:
                            card, policy = self._choose_primary_play(
                                play,
                                engine,
                                state,
                                legal_cards,
                                seed=decision_seed,
                                cancel_check=cancel_check,
                                public_prefix_hash=public_prefix_hash,
                            )
                    else:
                        strategic_card, strategic_policy = strategic
                        if strategic_card is None:
                            if self.decision_agreement is not None:
                                agreement_ordinal += 1
                                card, fallback_policy = self.decision_agreement.choose_play(
                                    play,
                                    engine,
                                    state,
                                    legal_cards,
                                    seed=decision_seed,
                                    agreement_ordinal=agreement_ordinal,
                                    cancel_check=cancel_check,
                                    public_prefix_hash=public_prefix_hash,
                                )
                            else:
                                card, fallback_policy = self._choose_primary_play(
                                    play,
                                    engine,
                                    state,
                                    legal_cards,
                                    seed=decision_seed,
                                    cancel_check=cancel_check,
                                    public_prefix_hash=public_prefix_hash,
                                )
                            policy = {
                                **dict(fallback_policy),
                                "strategicSelection": dict(strategic_policy),
                                "qualificationBlocked": True,
                            }
                        else:
                            card = Card.parse(strategic_card)
                            if card not in legal_cards:
                                raise ValueError(
                                    "strategic play selector returned an illegal card"
                                )
                            policy = dict(strategic_policy)
                elif self.decision_agreement is not None and len(legal_cards) > 1:
                    agreement_ordinal += 1
                    card, policy = self.decision_agreement.choose_play(
                        play,
                        engine,
                        state,
                        legal_cards,
                        seed=decision_seed,
                        agreement_ordinal=agreement_ordinal,
                        cancel_check=cancel_check,
                        public_prefix_hash=public_prefix_hash,
                    )
                else:
                    card, policy = self._choose_primary_play(
                        play,
                        engine,
                        state,
                        legal_cards,
                        seed=decision_seed,
                        cancel_check=cancel_check,
                        public_prefix_hash=public_prefix_hash,
                    )
                policy = self._policy_with_analysis_identity(policy)
                unresolved_agreements += int(bool(policy.get("qualificationBlocked")))
                before_events = len(state.events)
                selected = card.to_dict()
                raw_decisions.append({
                    "ordinal": ordinal,
                    "actor": actor.name.lower(),
                    "decisionKind": "play",
                    "phase": GamePhase.PLAYING.value,
                    "publicState": public,
                    "completeStateHash": complete_hash,
                    "legalActions": tuple(item.to_dict() for item in legal_cards),
                    "selectedAction": selected,
                    "resultingEventSequence": None,
                })
                policy_traces.append({"ordinal": ordinal, "trace": policy})
                engine.play_card(state, card, ekka=scripted_ekka)
                matching = [
                    event.sequence
                    for event in state.events[before_events:]
                    if event.type == "card_played"
                    and event.data.get("seat") == actor.name.lower()
                    and event.data.get("card") == selected
                ]
                if len(matching) != 1:
                    raise AssertionError("every play decision must produce one canonical card event")
                raw_decisions[-1]["resultingEventSequence"] = matching[0]
                ordinal += 1
            if scripted_plays is not None and scripted_play_index != len(scripted_plays):
                raise ValueError("scripted play contains actions after the round completed")
            if state.phase != GamePhase.SCORED or state.round_score is None:
                raise AssertionError("a complete played Laffa must be authoritatively scored")
        elif state.phase != GamePhase.COMPLETE:
            raise AssertionError("an all-pass auction must terminate as a complete redeal")

        event_chain = TraceEventV1.chain(item.to_dict() for item in state.events)
        event_hashes = {item.sequence: item.event_hash for item in event_chain}
        attempt_identity = {
            "schemaVersion": 2,
            "kind": str(lineage_kind),
            "rootSeed": root_seed,
            "attemptIndex": attempt_index,
            "attemptSeed": attempt_seed,
            "dealer": dealer.name.lower(),
            "roundNumber": round_number,
            "matchScore": list(match_score),
            "rulesHash": self.rules.sha256,
            "bundleHash": self.bundle_hash,
            "configurationSnapshotHash": self.configuration_snapshot_hash,
            "engineVersion": self.engine_version,
            "agentRelease": self.agent_release,
            "agentProfileHash": self.profile_hash,
            "generationTier": self.generation_tier,
            "agentQualificationHash": (
                self.agent_qualification.sha256 if self.agent_qualification is not None else None
            ),
            "seedPolicy": _SEED_POLICY,
            "decisionAgreementProfileHash": (
                self.decision_agreement.profile.sha256
                if self.decision_agreement is not None else None
            ),
            "decisionAgreementDescriptor": (
                self.decision_agreement.descriptor.to_dict()
                if self.decision_agreement is not None else None
            ),
            "decisionAgreementDecisionCount": agreement_ordinal,
            "decisionAgreementUnresolvedCount": unresolved_agreements,
            "analysisPolicyAuthority": copy.deepcopy(
                self._analysis_policy_authority
            ),
        }
        if self.trace_analysis_profile_hash is not None:
            assert self.analysis_identity_hash is not None
            attempt_identity.update({
                "traceAnalysisProfileHash": self.trace_analysis_profile_hash,
                "analysisIdentityHash": self.analysis_identity_hash,
            })
        if parent_trace_hash is not None:
            attempt_identity["parentTraceHash"] = str(parent_trace_hash)
        if deal_generator_provenance is not None:
            attempt_identity["dealGenerator"] = copy.deepcopy(
                dict(deal_generator_provenance)
            )
        if scripted_escalations is not None:
            attempt_identity["scriptedEscalations"] = [
                copy.deepcopy(dict(item)) for item in scripted_escalations
            ]
        if scripted_plays is not None:
            attempt_identity["scriptedPlayCompiler"] = copy.deepcopy(
                dict(scripted_play_provenance or {})
            )
        if play_selector is not None:
            attempt_identity["strategicPlaySelector"] = copy.deepcopy(
                play_selector.provenance()
            )
        if counterfactual_provenance is not None:
            attempt_identity["counterfactual"] = copy.deepcopy(
                dict(counterfactual_provenance)
            )
        game_id = f"SIM-{canonical_sha256(attempt_identity)[:20].upper()}"
        trusted_final = {
            **state.to_dict(reveal_hands=True),
            "gameId": game_id,
            "bundleHash": self.bundle_hash,
            "configurationSnapshot": copy.deepcopy(self.configuration_snapshot),
            "engineVersion": self.engine_version,
            "agentProfiles": {seat.name.lower(): self.agent_release for seat in Seat},
            "agentProfileHash": self.profile_hash,
            "generationTier": self.generation_tier,
            "agentQualification": (
                self.agent_qualification.to_dict() if self.agent_qualification is not None else None
            ),
            "modelReleases": [],
            "generation": attempt_identity,
            "decisionPolicyTraces": policy_traces,
            "decisionAgreement": {
                "descriptor": (
                    self.decision_agreement.descriptor.to_dict()
                    if self.decision_agreement is not None else None
                ),
                "profile": (
                    self.decision_agreement.profile.to_dict()
                    if self.decision_agreement is not None else None
                ),
                **(
                    {
                        "traceAnalysisProfileHash": self.trace_analysis_profile_hash,
                        "analysisIdentityHash": self.analysis_identity_hash,
                    }
                    if self.trace_analysis_profile_hash is not None
                    else {}
                ),
                "decisionCount": agreement_ordinal,
                "unresolvedCount": unresolved_agreements,
                "qualifiedAtGeneration": unresolved_agreements == 0,
            },
        }
        state_hash = canonical_sha256(trusted_final)
        game_identity = {
            "gameId": game_id,
            "version": 1,
            "stateHash": state_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "parentGameHash": parent_game_hash,
        }
        game_hash = canonical_sha256(game_identity)
        decisions: list[dict[str, Any]] = []
        for raw in raw_decisions:
            prefix_hash = canonical_sha256(raw["publicState"])
            identity = {
                "gameHash": game_hash,
                "ordinal": raw["ordinal"],
                "decisionKind": raw["decisionKind"],
                "actor": raw["actor"],
                "publicPrefixHash": prefix_hash,
                "completeStateHash": raw["completeStateHash"],
                "selectedAction": raw["selectedAction"],
            }
            decisions.append({
                **raw,
                "decisionId": f"DEC-{canonical_sha256(identity)[:20].upper()}",
                "publicPrefixHash": prefix_hash,
                "resultingEventHash": event_hashes[int(raw["resultingEventSequence"])],
            })
        ledger = DecisionLedgerV1.build(game_hash, decisions)
        trace = CompleteGameTraceV1.build(
            game_record={
                "gameHash": game_hash,
                "gameId": game_id,
                "stateHash": state_hash,
                "bundleHash": self.bundle_hash,
                "engineVersion": self.engine_version,
            },
            trusted_final_state=trusted_final,
            decision_ledger=ledger,
        )
        coverage = {
            f"seat:{item.acting_seat.name.lower()}" for item in ledger.records
        }
        coverage.update(f"decision:{item.decision_kind}" for item in ledger.records)
        coverage.add(f"dealer:{dealer.name.lower()}")
        if completed:
            assert state.contract is not None and state.buyer is not None
            coverage.update({
                f"contract:{state.contract.value}",
                f"buyer:{state.buyer.name.lower()}",
                f"trump:{state.trump_suit.name.lower() if state.trump_suit else 'none'}",
                f"purchase_round:{state.winning_bid.round_index if state.winning_bid else 'unknown'}",
            })
            coverage.update(f"project:{project.kind}" for project in state.projects)
        if self.decision_agreement is not None:
            coverage.add("agent_agreement:stable" if unresolved_agreements == 0 else "agent_agreement:unresolved")
        return trace, completed, tuple(sorted(coverage))
