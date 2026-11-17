"""Engine-witnessed enumeration of reachable pre-target Sakka scores."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .evidence_v3 import canonical_sha256
from .replay import CompleteTraceReplayerV1
from .sakka_context import SakkaContextV1
from .trace import CompleteGameTraceV1


@dataclass(frozen=True, slots=True)
class ScoreReachabilityProfileV1:
    profile_id: str
    revision: str
    maximum_witness_traces: int
    witnesses_per_outcome: int
    maximum_rounds: int
    maximum_states: int
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScoreReachabilityProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "maximumWitnessTraces",
            "witnessesPerOutcome", "maximumRounds", "maximumStates",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown score reachability profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported score reachability profile schema")
        witnesses = int(value.get("maximumWitnessTraces", 0))
        per_outcome = int(value.get("witnessesPerOutcome", 0))
        rounds = int(value.get("maximumRounds", 0))
        states = int(value.get("maximumStates", 0))
        if not 1 <= witnesses <= 1_000_000:
            raise ValueError("score reachability witness bound is out of range")
        if not 1 <= per_outcome <= 100:
            raise ValueError("score reachability witnesses per outcome is out of range")
        if not 1 <= rounds <= 10_000:
            raise ValueError("score reachability round bound is out of range")
        if not 1 <= states <= 10_000_000:
            raise ValueError("score reachability state bound is out of range")
        core = {
            "schemaVersion": 1,
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "maximumWitnessTraces": witnesses,
            "witnessesPerOutcome": per_outcome,
            "maximumRounds": rounds,
            "maximumStates": states,
        }
        if not core["id"] or not core["revision"]:
            raise ValueError("score reachability profile identity is required")
        return cls(
            core["id"], core["revision"], witnesses, per_outcome, rounds, states,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "maximumWitnessTraces": self.maximum_witness_traces,
            "witnessesPerOutcome": self.witnesses_per_outcome,
            "maximumRounds": self.maximum_rounds,
            "maximumStates": self.maximum_states,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TransferableRoundOutcomeV1:
    trace_hash: str
    dealer: Seat
    awarded_points: tuple[int, int]
    deal_commitment: str
    witness_context_replay_hash: str
    sha256: str
    trace: CompleteGameTraceV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "traceHash": self.trace_hash,
            "dealer": self.dealer.name.lower(),
            "awardedPoints": list(self.awarded_points),
            "dealCommitment": self.deal_commitment,
            "witnessContextReplayHash": self.witness_context_replay_hash,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthoritativeRoundOutcomeCatalogV1:
    rules_hash: str
    bundle_hash: str
    engine_version: str
    profile_hash: str
    outcomes: tuple[TransferableRoundOutcomeV1, ...]
    rejected: tuple[Mapping[str, str], ...]
    sha256: str

    @classmethod
    def build(
        cls,
        traces: Iterable[CompleteGameTraceV1],
        *,
        rules: RuleBook,
        profile: ScoreReachabilityProfileV1,
    ) -> "AuthoritativeRoundOutcomeCatalogV1":
        values = tuple(sorted(traces, key=lambda item: item.sha256))
        if not values or len(values) > profile.maximum_witness_traces:
            raise ValueError("score outcome witness count is outside policy")
        bundles = {item.bundle_hash for item in values}
        engines = {item.engine_version for item in values}
        if len(bundles) != 1 or len(engines) != 1:
            raise ValueError("score outcome witnesses must use one bundle and engine")
        replayer = CompleteTraceReplayerV1(rules)
        candidates: list[TransferableRoundOutcomeV1] = []
        rejected: list[dict[str, str]] = []
        for trace in values:
            try:
                replayer.replay(trace)
                state = trace.trusted_final_state
                if str(state.get("phase")) != GamePhase.SCORED.value:
                    raise ValueError("not_scored")
                if str(state.get("escalation") or "normal") != "normal":
                    raise ValueError("score_dependent_escalation")
                score = state.get("roundScore")
                if not isinstance(score, Mapping) or bool(score.get("gahwa")):
                    raise ValueError("gahwa_or_missing_score")
                generation = dict(state.get("generation") or {})
                start_values = tuple(int(item) for item in generation.get("matchScore") or ())
                if len(start_values) != 2:
                    raise ValueError("missing_start_score")
                proof = replayer.replay_in_context(
                    trace,
                    match_score=start_values,
                    round_number=int(generation.get("roundNumber", 1)),
                )
                awarded = proof.awarded_points
                if min(awarded) < 0 or awarded == (0, 0):
                    raise ValueError("non_advancing_score")
                core = {
                    "schemaVersion": 1,
                    "traceHash": trace.sha256,
                    "dealer": proof.dealer.name.lower(),
                    "awardedPoints": list(awarded),
                    "dealCommitment": str(trace.seed_provenance["deckCommitment"]),
                    "witnessContextReplayHash": proof.sha256,
                }
                candidates.append(TransferableRoundOutcomeV1(
                    trace.sha256,
                    proof.dealer,
                    awarded,
                    core["dealCommitment"],
                    proof.sha256,
                    canonical_sha256(core),
                    trace,
                ))
            except ValueError as exc:
                rejected.append({"traceHash": trace.sha256, "reason": str(exc)})
        # Retain bounded independent witnesses so a path never repeats a deal.
        unique: dict[
            tuple[Seat, tuple[int, int], str], TransferableRoundOutcomeV1
        ] = {}
        for item in candidates:
            unique.setdefault((item.dealer, item.awarded_points, item.deal_commitment), item)
        grouped: dict[tuple[Seat, tuple[int, int]], list[TransferableRoundOutcomeV1]] = {}
        for item in sorted(
            unique.values(),
            key=lambda value: (
                int(value.dealer), value.awarded_points,
                value.deal_commitment, value.trace_hash,
            ),
        ):
            group = grouped.setdefault((item.dealer, item.awarded_points), [])
            if len(group) < profile.witnesses_per_outcome:
                group.append(item)
        outcomes = tuple(
            item
            for key in sorted(grouped, key=lambda value: (int(value[0]), value[1]))
            for item in grouped[key]
        )
        if not outcomes:
            raise ValueError("score outcome catalog contains no transferable witnesses")
        core = {
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "bundleHash": next(iter(bundles)),
            "engineVersion": next(iter(engines)),
            "profileHash": profile.sha256,
            "outcomeHashes": [item.sha256 for item in outcomes],
            "rejected": sorted(rejected, key=lambda item: (item["traceHash"], item["reason"])),
        }
        return cls(
            rules.sha256,
            core["bundleHash"],
            core["engineVersion"],
            profile.sha256,
            outcomes,
            tuple(core["rejected"]),
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "profileHash": self.profile_hash,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "rejected": [dict(item) for item in self.rejected],
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ScoreReachabilityStepV1:
    ordinal: int
    witness_trace_hash: str
    deal_commitment: str
    dealer: Seat
    starting_score: tuple[int, int]
    awarded_points: tuple[int, int]
    ending_score: tuple[int, int]
    context_replay_hash: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreReachabilityStepV1":
        return cls(
            int(value["ordinal"]),
            str(value["witnessTraceHash"]),
            str(value["dealCommitment"]),
            Seat.parse(str(value["dealer"])),
            tuple(int(item) for item in value["startingScore"]),
            tuple(int(item) for item in value["awardedPoints"]),
            tuple(int(item) for item in value["endingScore"]),
            str(value["contextReplayHash"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "witnessTraceHash": self.witness_trace_hash,
            "dealCommitment": self.deal_commitment,
            "dealer": self.dealer.name.lower(),
            "startingScore": list(self.starting_score),
            "awardedPoints": list(self.awarded_points),
            "endingScore": list(self.ending_score),
            "contextReplayHash": self.context_replay_hash,
        }


@dataclass(frozen=True, slots=True)
class ScoreReachabilityProofV1:
    catalog_hash: str
    profile_hash: str
    rules_hash: str
    bundle_hash: str
    engine_version: str
    target: int
    initial_dealer: Seat
    target_score: tuple[int, int]
    target_dealer: Seat
    round_number: int
    steps: tuple[ScoreReachabilityStepV1, ...]
    explored_states: int
    sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreReachabilityProofV1":
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported score reachability proof schema version")
        proof = cls(
            str(value["catalogHash"]),
            str(value["profileHash"]),
            str(value["rulesHash"]),
            str(value["bundleHash"]),
            str(value["engineVersion"]),
            int(value["target"]),
            Seat.parse(str(value["initialDealer"])),
            tuple(int(item) for item in value["targetScore"]),
            Seat.parse(str(value["targetDealer"])),
            int(value["roundNumber"]),
            tuple(ScoreReachabilityStepV1.from_dict(item) for item in value["steps"]),
            int(value["exploredStates"]),
            str(value["sha256"]),
        )
        proof.verify()
        return proof

    def verify(self) -> None:
        if self.target < 1:
            raise ValueError("score reachability proof target must be positive")
        if (
            len(self.target_score) != 2
            or min(self.target_score) < 0
            or max(self.target_score) >= self.target
        ):
            raise ValueError("score reachability proof target score is invalid")
        if self.round_number != len(self.steps) + 1 or self.explored_states < 1:
            raise ValueError("score reachability proof bounds are inconsistent")
        score = (0, 0)
        dealer = self.initial_dealer
        commitments: set[str] = set()
        for ordinal, step in enumerate(self.steps, start=1):
            if step.ordinal != ordinal or step.starting_score != score:
                raise ValueError("score reachability proof path is discontinuous")
            if step.dealer != dealer:
                raise ValueError("score reachability proof dealer path is inconsistent")
            if min(step.awarded_points) < 0 or step.awarded_points == (0, 0):
                raise ValueError("score reachability proof contains a non-advancing round")
            expected = (
                score[0] + step.awarded_points[0],
                score[1] + step.awarded_points[1],
            )
            if step.ending_score != expected or max(expected) >= self.target:
                raise ValueError("score reachability proof score path is inconsistent")
            if step.deal_commitment in commitments:
                raise ValueError("score reachability proof repeats a dealt-card commitment")
            for value, field in (
                (step.witness_trace_hash, "witness trace hash"),
                (step.deal_commitment, "deal commitment"),
                (step.context_replay_hash, "context replay hash"),
            ):
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ValueError(f"score reachability proof {field} is invalid")
            commitments.add(step.deal_commitment)
            score = expected
            dealer = dealer.next()
        if score != self.target_score or dealer != self.target_dealer:
            raise ValueError("score reachability proof does not reach its declared context")
        core = {
            "schemaVersion": 1,
            "catalogHash": self.catalog_hash,
            "profileHash": self.profile_hash,
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "target": self.target,
            "initialDealer": self.initial_dealer.name.lower(),
            "targetScore": list(self.target_score),
            "targetDealer": self.target_dealer.name.lower(),
            "roundNumber": self.round_number,
            "steps": [item.to_dict() for item in self.steps],
            "exploredStates": self.explored_states,
        }
        if canonical_sha256(core) != self.sha256:
            raise ValueError("score reachability proof hash is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "catalogHash": self.catalog_hash,
            "profileHash": self.profile_hash,
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "target": self.target,
            "initialDealer": self.initial_dealer.name.lower(),
            "targetScore": list(self.target_score),
            "targetDealer": self.target_dealer.name.lower(),
            "roundNumber": self.round_number,
            "steps": [item.to_dict() for item in self.steps],
            "exploredStates": self.explored_states,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedEnumeratedContextV1:
    context: SakkaContextV1
    proof: ScoreReachabilityProofV1
    sha256: str

    @classmethod
    def build(cls, proof: ScoreReachabilityProofV1) -> "VerifiedEnumeratedContextV1":
        proof.verify()
        context = SakkaContextV1.build(
            starting_score=proof.target_score,
            dealer=proof.target_dealer,
            round_number=proof.round_number,
            target=proof.target,
            reachability_mode="verified_enumeration",
            reachability_proof_hash=proof.sha256,
        )
        core = {
            "schemaVersion": 1,
            "contextHash": context.sha256,
            "proofHash": proof.sha256,
            "rulesHash": proof.rules_hash,
            "bundleHash": proof.bundle_hash,
            "engineVersion": proof.engine_version,
        }
        return cls(context, proof, canonical_sha256(core))

    def verify(self) -> None:
        self.proof.verify()
        expected_context = SakkaContextV1.build(
            starting_score=self.proof.target_score,
            dealer=self.proof.target_dealer,
            round_number=self.proof.round_number,
            target=self.proof.target,
            reachability_mode="verified_enumeration",
            reachability_proof_hash=self.proof.sha256,
        )
        if self.context != expected_context:
            raise ValueError("enumerated context does not match its reachability proof")
        core = {
            "schemaVersion": 1,
            "contextHash": self.context.sha256,
            "proofHash": self.proof.sha256,
            "rulesHash": self.proof.rules_hash,
            "bundleHash": self.proof.bundle_hash,
            "engineVersion": self.proof.engine_version,
        }
        if canonical_sha256(core) != self.sha256:
            raise ValueError("enumerated context hash is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "context": self.context.to_dict(),
            "proof": self.proof.to_dict(),
            "sha256": self.sha256,
        }


class AuthoritativeScoreReachabilityV1:
    descriptor = PluginDescriptor(
        id="authoritative_score_reachability_v1",
        kind="context_search",
        version="1.0.0",
        description="Prove exact pre-target scores from engine-replayed normal-round witnesses",
        capabilities=(
            "verified_enumeration", "exact_score_path", "dealer_progression",
            "context_replay", "bounded_search",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        catalog: AuthoritativeRoundOutcomeCatalogV1,
        rules: RuleBook,
        profile: ScoreReachabilityProfileV1,
    ):
        if catalog.rules_hash != rules.sha256 or catalog.profile_hash != profile.sha256:
            raise ValueError("score reachability catalog identity mismatch")
        self.catalog = catalog
        self.rules = rules
        self.profile = profile

    def prove(
        self,
        target_score: tuple[int, int],
        *,
        target_dealer: Seat | str,
        initial_dealer: Seat | str = Seat.EAST,
    ) -> ScoreReachabilityProofV1:
        goal = tuple(int(item) for item in target_score)
        target = int(self.rules.data["scoring"]["match"]["target"])
        if len(goal) != 2 or min(goal) < 0 or max(goal) >= target:
            raise ValueError("reachability target must be a nonnegative pre-match score")
        desired_dealer = Seat.parse(target_dealer) if isinstance(target_dealer, str) else target_dealer
        first_dealer = Seat.parse(initial_dealer) if isinstance(initial_dealer, str) else initial_dealer
        start_state = ((0, 0), first_dealer, 1, ())
        queue = deque([start_state])
        parents: dict[
            tuple[tuple[int, int], Seat, int, tuple[str, ...]],
            tuple[
                tuple[tuple[int, int], Seat, int, tuple[str, ...]],
                TransferableRoundOutcomeV1,
            ] | None,
        ] = {start_state: None}
        found: tuple[tuple[int, int], Seat, int, tuple[str, ...]] | None = None
        outcomes_by_dealer = {
            seat: tuple(item for item in self.catalog.outcomes if item.dealer == seat)
            for seat in Seat
        }
        while queue:
            current = queue.popleft()
            score, dealer, round_number, used_commitments = current
            if score == goal and dealer == desired_dealer:
                found = current
                break
            if round_number > self.profile.maximum_rounds:
                continue
            for outcome in outcomes_by_dealer[dealer]:
                if outcome.deal_commitment in used_commitments:
                    continue
                ending = (
                    score[0] + outcome.awarded_points[0],
                    score[1] + outcome.awarded_points[1],
                )
                if ending[0] > goal[0] or ending[1] > goal[1] or max(ending) >= target:
                    continue
                state = (
                    ending,
                    dealer.next(),
                    round_number + 1,
                    tuple(sorted((*used_commitments, outcome.deal_commitment))),
                )
                if state in parents:
                    continue
                parents[state] = (current, outcome)
                if len(parents) > self.profile.maximum_states:
                    raise RuntimeError("score reachability exceeded its governed state bound")
                queue.append(state)
        if found is None:
            raise LookupError("no verified score path exists in the current witness catalog")
        path: list[TransferableRoundOutcomeV1] = []
        cursor = found
        while parents[cursor] is not None:
            previous, outcome = parents[cursor]
            path.append(outcome)
            cursor = previous
        path.reverse()
        score = (0, 0)
        dealer = first_dealer
        replayer = CompleteTraceReplayerV1(self.rules)
        steps: list[ScoreReachabilityStepV1] = []
        for index, outcome in enumerate(path, start=1):
            if outcome.dealer != dealer:
                raise AssertionError("score proof dealer path is inconsistent")
            replay = replayer.replay_in_context(
                outcome.trace,
                match_score=score,
                round_number=index,
            )
            steps.append(ScoreReachabilityStepV1(
                index,
                outcome.trace_hash,
                outcome.deal_commitment,
                dealer,
                score,
                replay.awarded_points,
                replay.ending_score,
                replay.sha256,
            ))
            score = replay.ending_score
            dealer = dealer.next()
        if score != goal or dealer != desired_dealer:
            raise AssertionError("verified score path did not reach its target")
        if len({item.deal_commitment for item in steps}) != len(steps):
            raise AssertionError("verified score path repeated a dealt-card commitment")
        core = {
            "schemaVersion": 1,
            "catalogHash": self.catalog.sha256,
            "profileHash": self.profile.sha256,
            "rulesHash": self.catalog.rules_hash,
            "bundleHash": self.catalog.bundle_hash,
            "engineVersion": self.catalog.engine_version,
            "target": target,
            "initialDealer": first_dealer.name.lower(),
            "targetScore": list(goal),
            "targetDealer": desired_dealer.name.lower(),
            "roundNumber": len(steps) + 1,
            "steps": [item.to_dict() for item in steps],
            "exploredStates": len(parents),
        }
        return ScoreReachabilityProofV1(
            self.catalog.sha256,
            self.profile.sha256,
            self.catalog.rules_hash,
            self.catalog.bundle_hash,
            self.catalog.engine_version,
            target,
            first_dealer,
            goal,
            desired_dealer,
            len(steps) + 1,
            tuple(steps),
            len(parents),
            canonical_sha256(core),
        )


class ScoreReachabilityCatalogFactoryV1:
    """Registered factory for row-selected witness trace catalogs."""

    descriptor = PluginDescriptor(
        id="score_reachability_catalog_factory_v1",
        kind="context_search_factory",
        version="1.0.0",
        description="Build engine-witnessed score reachability catalogs",
        capabilities=("verified_enumeration", "row_selected_witnesses"),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(self, rules: RuleBook, profile: ScoreReachabilityProfileV1):
        self.rules = rules
        self.profile = profile

    def build(
        self,
        traces: Iterable[CompleteGameTraceV1],
    ) -> AuthoritativeScoreReachabilityV1:
        catalog = AuthoritativeRoundOutcomeCatalogV1.build(
            traces,
            rules=self.rules,
            profile=self.profile,
        )
        return AuthoritativeScoreReachabilityV1(catalog, self.rules, self.profile)
