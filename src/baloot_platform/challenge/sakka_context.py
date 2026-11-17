"""Natural and contextualized Sakka provenance contracts."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from baloot_platform.core.enums import Seat

from .evidence_v3 import canonical_sha256
from .trace import CompleteGameTraceV1


def _sha256(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def _start(trace: CompleteGameTraceV1) -> dict[str, Any]:
    generation = dict(trace.trusted_final_state.get("generation") or {})
    return {
        "score": tuple(int(item) for item in generation.get("matchScore") or ()),
        "dealer": str(generation.get("dealer") or ""),
        "roundNumber": int(generation.get("roundNumber", 0)),
    }


def _finish(trace: CompleteGameTraceV1) -> tuple[int, int]:
    score = tuple(int(item) for item in trace.trusted_final_state.get("matchScore") or ())
    if len(score) != 2:
        raise ValueError("scored Laffa trace must contain a two-team match score")
    return score


def _scored(trace: CompleteGameTraceV1) -> None:
    CompleteGameTraceV1.from_dict(trace.to_dict(trusted=True))
    if str(trace.trusted_final_state.get("phase")) != "scored":
        raise ValueError("Sakka provenance accepts only complete scored Laffa traces")
    if not isinstance(trace.trusted_final_state.get("roundScore"), Mapping):
        raise ValueError("scored Laffa trace is missing authoritative round points")


@dataclass(frozen=True, slots=True)
class SakkaContextV1:
    starting_score: tuple[int, int]
    dealer: Seat
    round_number: int
    target: int
    reachability_mode: str
    reachability_proof_hash: str
    previous_round_hashes: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        starting_score: tuple[int, int],
        dealer: Seat | str,
        round_number: int,
        target: int,
        reachability_mode: str,
        reachability_proof_hash: str,
        previous_round_hashes: Iterable[str] = (),
    ) -> "SakkaContextV1":
        score = tuple(int(item) for item in starting_score)
        if len(score) != 2 or min(score) < 0:
            raise ValueError("Sakka context requires two nonnegative team scores")
        goal = int(target)
        if goal < 1 or max(score) >= goal:
            raise ValueError("Sakka context must begin before the declared target")
        number = int(round_number)
        if number < 1:
            raise ValueError("Sakka context round number must be positive")
        mode = str(reachability_mode)
        if mode not in {"natural_history", "verified_enumeration"}:
            raise ValueError("unsupported Sakka context reachability mode")
        proof = _sha256(reachability_proof_hash, "reachabilityProofHash")
        previous = tuple(_sha256(item, "previousRoundHash") for item in previous_round_hashes)
        if mode == "natural_history" and len(previous) != number - 1:
            raise ValueError("natural Sakka context must reference every preceding round")
        core = {
            "schemaVersion": 1,
            "startingScore": list(score),
            "dealer": Seat.parse(dealer).name.lower() if isinstance(dealer, str) else dealer.name.lower(),
            "roundNumber": number,
            "target": goal,
            "reachabilityMode": mode,
            "reachabilityProofHash": proof,
            "previousRoundHashes": list(previous),
        }
        return cls(
            score,
            Seat.parse(dealer) if isinstance(dealer, str) else dealer,
            number,
            goal,
            mode,
            proof,
            previous,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "startingScore": list(self.starting_score),
            "dealer": self.dealer.name.lower(),
            "roundNumber": self.round_number,
            "target": self.target,
            "reachabilityMode": self.reachability_mode,
            "reachabilityProofHash": self.reachability_proof_hash,
            "previousRoundHashes": list(self.previous_round_hashes),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextualizedLaffaV1:
    parent_trace_hash: str
    descendant_trace_hash: str
    context_hash: str
    original_decision_id: str
    descendant_decision_id: str
    analysis_hash: str
    context_amplification: Mapping[str, float]
    strategy_changed: bool
    recommendation_stable: bool
    provenance: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        parent: CompleteGameTraceV1,
        descendant: CompleteGameTraceV1,
        context: SakkaContextV1,
        original_decision_id: str,
        descendant_decision_id: str,
        analysis_hash: str,
        context_amplification: Mapping[str, float],
        strategy_changed: bool,
        recommendation_stable: bool,
    ) -> "ContextualizedLaffaV1":
        _scored(parent)
        _scored(descendant)
        if parent.rules_hash != descendant.rules_hash or parent.bundle_hash != descendant.bundle_hash:
            raise ValueError("contextualized Laffa must preserve rules and bundle identity")
        parent_commitment = str(parent.seed_provenance.get("deckCommitment") or "")
        descendant_commitment = str(descendant.seed_provenance.get("deckCommitment") or "")
        if not parent_commitment or parent_commitment != descendant_commitment:
            raise ValueError("contextualized Laffa must preserve the exact dealt cards")
        start = _start(descendant)
        if (
            start["score"] != context.starting_score
            or Seat.parse(start["dealer"]) != context.dealer
            or start["roundNumber"] != context.round_number
        ):
            raise ValueError("descendant Laffa was not re-simulated under the declared context")
        parent_ids = {item.decision_id for item in parent.decision_ledger.records}
        descendant_ids = {item.decision_id for item in descendant.decision_ledger.records}
        if original_decision_id not in parent_ids or descendant_decision_id not in descendant_ids:
            raise ValueError("contextualized Laffa critical decision identity is missing")
        if not recommendation_stable:
            raise ValueError("unstable recommendations cannot become contextualized challenges")
        amplification = {
            str(key): float(value) for key, value in context_amplification.items()
        }
        if not amplification or any(not -100.0 <= value <= 100.0 for value in amplification.values()):
            raise ValueError("context amplification requires bounded declared dimensions")
        analysis = _sha256(analysis_hash, "analysisHash")
        core = {
            "schemaVersion": 1,
            "parentTraceHash": parent.sha256,
            "descendantTraceHash": descendant.sha256,
            "contextHash": context.sha256,
            "originalDecisionId": original_decision_id,
            "descendantDecisionId": descendant_decision_id,
            "analysisHash": analysis,
            "contextAmplification": amplification,
            "strategyChanged": bool(strategy_changed),
            "recommendationStable": True,
            "provenance": "contextualized_laffa",
        }
        return cls(
            parent.sha256,
            descendant.sha256,
            context.sha256,
            str(original_decision_id),
            str(descendant_decision_id),
            analysis,
            amplification,
            bool(strategy_changed),
            True,
            "contextualized_laffa",
            canonical_sha256(core),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextualizedLaffaV1":
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported contextualized Laffa schema version")
        amplification = {
            str(key): float(item)
            for key, item in dict(value.get("contextAmplification") or {}).items()
        }
        if not amplification or any(
            not -100.0 <= item <= 100.0 for item in amplification.values()
        ):
            raise ValueError("context amplification requires bounded declared dimensions")
        if value.get("provenance") != "contextualized_laffa":
            raise ValueError("invalid contextualized Laffa provenance")
        if value.get("recommendationStable") is not True:
            raise ValueError("unstable recommendations cannot become contextualized challenges")
        if not isinstance(value.get("strategyChanged"), bool):
            raise ValueError("contextualized Laffa strategyChanged must be boolean")
        core = {
            "schemaVersion": 1,
            "parentTraceHash": _sha256(value.get("parentTraceHash"), "parentTraceHash"),
            "descendantTraceHash": _sha256(
                value.get("descendantTraceHash"), "descendantTraceHash"
            ),
            "contextHash": _sha256(value.get("contextHash"), "contextHash"),
            "originalDecisionId": str(value.get("originalDecisionId") or ""),
            "descendantDecisionId": str(value.get("descendantDecisionId") or ""),
            "analysisHash": _sha256(value.get("analysisHash"), "analysisHash"),
            "contextAmplification": amplification,
            "strategyChanged": value["strategyChanged"],
            "recommendationStable": True,
            "provenance": "contextualized_laffa",
        }
        if not core["originalDecisionId"] or not core["descendantDecisionId"]:
            raise ValueError("contextualized Laffa decision identities are required")
        identity = canonical_sha256(core)
        if _sha256(value.get("sha256"), "sha256") != identity:
            raise ValueError("contextualized Laffa identity mismatch")
        return cls(
            core["parentTraceHash"],
            core["descendantTraceHash"],
            core["contextHash"],
            core["originalDecisionId"],
            core["descendantDecisionId"],
            core["analysisHash"],
            amplification,
            core["strategyChanged"],
            True,
            "contextualized_laffa",
            identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "parentTraceHash": self.parent_trace_hash,
            "descendantTraceHash": self.descendant_trace_hash,
            "contextHash": self.context_hash,
            "originalDecisionId": self.original_decision_id,
            "descendantDecisionId": self.descendant_decision_id,
            "analysisHash": self.analysis_hash,
            "contextAmplification": dict(self.context_amplification),
            "strategyChanged": self.strategy_changed,
            "recommendationStable": self.recommendation_stable,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class NaturalSakkaV1:
    target: int
    rules_hash: str
    bundle_hash: str
    engine_version: str
    agent_releases: tuple[str, ...]
    rounds: tuple[Mapping[str, Any], ...]
    starting_score: tuple[int, int]
    final_score: tuple[int, int]
    winner: str
    lead_changes: int
    provenance: str
    sha256: str

    @classmethod
    def build(
        cls,
        traces: Iterable[CompleteGameTraceV1],
        *,
        target: int,
        attempt_trace_groups: Iterable[Iterable[CompleteGameTraceV1]] | None = None,
    ) -> "NaturalSakkaV1":
        values = tuple(traces)
        if not values:
            raise ValueError("natural Sakka requires at least one scored Laffa")
        goal = int(target)
        if goal < 1:
            raise ValueError("natural Sakka target must be positive")
        for trace in values:
            _scored(trace)
        if len({item.rules_hash for item in values}) != 1:
            raise ValueError("natural Sakka rounds must use one rules revision")
        if len({item.bundle_hash for item in values}) != 1:
            raise ValueError("natural Sakka rounds must use one configuration bundle")
        if len({item.engine_version for item in values}) != 1:
            raise ValueError("natural Sakka rounds must use one engine version")
        releases = tuple(sorted({profile for trace in values for _, profile in trace.agent_profiles}))
        if len(releases) != 1:
            raise ValueError("natural Sakka must use one equal frozen agent release")

        lineage_groups: tuple[tuple[CompleteGameTraceV1, ...], ...] | None = None
        if attempt_trace_groups is not None:
            lineage_groups = tuple(tuple(group) for group in attempt_trace_groups)
            if len(lineage_groups) != len(values) or any(not group for group in lineage_groups):
                raise ValueError("natural Sakka auction lineage must match its scored Laffat")
            if tuple(group[-1].sha256 for group in lineage_groups) != tuple(
                trace.sha256 for trace in values
            ):
                raise ValueError("natural Sakka auction lineage must end at each scored Laffa")
            attempts = tuple(trace for group in lineage_groups for trace in group)
            if len({trace.rules_hash for trace in attempts}) != 1:
                raise ValueError("natural Sakka auction attempts must use one rules revision")
            if len({trace.bundle_hash for trace in attempts}) != 1:
                raise ValueError("natural Sakka auction attempts must use one configuration bundle")
            if len({trace.engine_version for trace in attempts}) != 1:
                raise ValueError("natural Sakka auction attempts must use one engine version")
            attempt_releases = {
                profile for trace in attempts for _, profile in trace.agent_profiles
            }
            if attempt_releases != set(releases):
                raise ValueError("natural Sakka auction attempts must use the frozen agent release")
            expected_attempt_score = _start(attempts[0])["score"]
            previous_attempt_dealer: Seat | None = None
            previous_attempt_round: int | None = None
            for group in lineage_groups:
                for attempt_index, attempt in enumerate(group):
                    CompleteGameTraceV1.from_dict(attempt.to_dict(trusted=True))
                    start = _start(attempt)
                    if start["score"] != expected_attempt_score:
                        raise ValueError("natural Sakka auction-attempt score continuity is broken")
                    dealer = Seat.parse(start["dealer"])
                    if (
                        previous_attempt_dealer is not None
                        and dealer != previous_attempt_dealer.next()
                    ):
                        raise ValueError("natural Sakka auction-attempt dealer progression is broken")
                    if (
                        previous_attempt_round is not None
                        and start["roundNumber"] != previous_attempt_round + 1
                    ):
                        raise ValueError("natural Sakka auction-attempt round progression is broken")
                    is_group_final = attempt_index == len(group) - 1
                    phase = str(attempt.trusted_final_state.get("phase"))
                    if is_group_final:
                        _scored(attempt)
                        expected_attempt_score = _finish(attempt)
                    elif (
                        phase != "complete"
                        or attempt.trusted_final_state.get("roundScore") is not None
                    ):
                        raise ValueError(
                            "natural Sakka auction lineage contains a non-final scored attempt"
                        )
                    elif tuple(
                        int(item) for item in attempt.trusted_final_state["matchScore"]
                    ) != expected_attempt_score:
                        raise ValueError("all-pass auction attempt changed the match score")
                    previous_attempt_dealer = Seat.parse(
                        str(attempt.trusted_final_state["dealer"])
                    )
                    previous_attempt_round = int(
                        attempt.trusted_final_state["roundNumber"]
                    )

        rounds: list[dict[str, Any]] = []
        expected_score = _start(values[0])["score"]
        previous_dealer: Seat | None = None
        previous_round_number: int | None = None
        leaders: list[int | None] = []
        for index, trace in enumerate(values):
            start = _start(trace)
            finish = _finish(trace)
            if start["score"] != expected_score:
                raise ValueError("natural Sakka score continuity is broken")
            dealer = Seat.parse(start["dealer"])
            if previous_dealer is not None and previous_round_number is not None:
                round_delta = start["roundNumber"] - previous_round_number
                if round_delta < 1 or (lineage_groups is None and round_delta != 1):
                    raise ValueError("natural Sakka round progression is broken")
                expected_dealer = previous_dealer
                for _ in range(round_delta):
                    expected_dealer = expected_dealer.next()
                if dealer != expected_dealer:
                    raise ValueError("natural Sakka dealer progression is broken")
            match_winner = trace.trusted_final_state["roundScore"].get("matchWinner")
            if index < len(values) - 1 and (max(finish) >= goal or match_winner is not None):
                raise ValueError("natural Sakka contains rounds after match termination")
            leader = 0 if finish[0] > finish[1] else 1 if finish[1] > finish[0] else None
            leaders.append(leader)
            round_evidence = {
                "ordinal": index + 1,
                "traceHash": trace.sha256,
                "startingScore": list(start["score"]),
                "awardedPoints": list(trace.trusted_final_state["roundScore"]["awardedPoints"]),
                "finalScore": list(finish),
                "dealer": dealer.name.lower(),
                "roundNumber": start["roundNumber"],
            }
            if lineage_groups is not None:
                round_evidence["auctionAttemptTraceHashes"] = [
                    attempt.sha256 for attempt in lineage_groups[index]
                ]
                round_evidence["allPassCount"] = len(lineage_groups[index]) - 1
            rounds.append(round_evidence)
            expected_score = finish
            previous_dealer = Seat.parse(str(trace.trusted_final_state["dealer"]))
            previous_round_number = int(trace.trusted_final_state["roundNumber"])
        if max(expected_score) < goal:
            raise ValueError("natural Sakka does not reach its declared target")
        winner_index = 0 if expected_score[0] > expected_score[1] else 1
        expected_winner = "north_south" if winner_index == 0 else "west_east"
        if values[-1].trusted_final_state["roundScore"].get("matchWinner") != expected_winner:
            raise ValueError("natural Sakka target does not match the engine's terminal outcome")
        changes = sum(
            1 for left, right in zip(leaders, leaders[1:])
            if left is not None and right is not None and left != right
        )
        core = {
            "schemaVersion": 1,
            "target": goal,
            "rulesHash": values[0].rules_hash,
            "bundleHash": values[0].bundle_hash,
            "engineVersion": values[0].engine_version,
            "agentReleases": list(releases),
            "rounds": rounds,
            "startingScore": list(_start(values[0])["score"]),
            "finalScore": list(expected_score),
            "winner": expected_winner,
            "leadChanges": changes,
            "provenance": "natural_sakka",
        }
        return cls(
            goal,
            values[0].rules_hash,
            values[0].bundle_hash,
            values[0].engine_version,
            releases,
            tuple(rounds),
            _start(values[0])["score"],
            expected_score,
            core["winner"],
            changes,
            "natural_sakka",
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "target": self.target,
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "agentReleases": list(self.agent_releases),
            "rounds": [dict(item) for item in self.rounds],
            "startingScore": list(self.starting_score),
            "finalScore": list(self.final_score),
            "winner": self.winner,
            "leadChanges": self.lead_changes,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CuratedSakkaSignerV1:
    secret: bytes
    key_id: str
    algorithm: str = "hmac-sha256"

    @classmethod
    def from_secret(cls, secret: bytes) -> "CuratedSakkaSignerV1":
        if len(secret) < 32:
            raise ValueError("curated Sakka signing secret must contain at least 32 bytes")
        return cls(bytes(secret), hashlib.sha256(secret).hexdigest()[:20])

    def sign_hash(self, composition_hash: str) -> str:
        digest = _sha256(composition_hash, "compositionHash")
        return hmac.new(self.secret, digest.encode("ascii"), hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class CuratedSakkaV1:
    target: int
    rules_hash: str
    bundle_hash: str
    engine_version: str
    agent_releases: tuple[str, ...]
    rounds: tuple[Mapping[str, Any], ...]
    starting_score: tuple[int, int]
    final_score: tuple[int, int]
    winner: str
    provenance: str
    composition_hash: str
    signature_algorithm: str
    signature_key_id: str
    signature: str

    @staticmethod
    def _core(
        *,
        target: int,
        rules_hash: str,
        bundle_hash: str,
        engine_version: str,
        agent_releases: tuple[str, ...],
        rounds: tuple[Mapping[str, Any], ...],
        starting_score: tuple[int, int],
        final_score: tuple[int, int],
        winner: str,
    ) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "target": int(target),
            "rulesHash": rules_hash,
            "bundleHash": bundle_hash,
            "engineVersion": engine_version,
            "agentReleases": list(agent_releases),
            "rounds": [dict(item) for item in rounds],
            "startingScore": list(starting_score),
            "finalScore": list(final_score),
            "winner": winner,
            "provenance": "curated_sakka",
        }

    @classmethod
    def build(
        cls,
        contextualized_laffat: Iterable[ContextualizedLaffaV1],
        descendants: Iterable[CompleteGameTraceV1],
        *,
        target: int,
        signer: CuratedSakkaSignerV1,
    ) -> "CuratedSakkaV1":
        contexts = tuple(contextualized_laffat)
        traces = tuple(descendants)
        if not contexts or len(contexts) != len(traces):
            raise ValueError("curated Sakka requires one contextualized record per descendant")
        goal = int(target)
        if goal < 1:
            raise ValueError("curated Sakka target must be positive")
        for trace in traces:
            _scored(trace)
        if any(context.descendant_trace_hash != trace.sha256 for context, trace in zip(contexts, traces)):
            raise ValueError("curated Sakka descendant identity mismatch")
        if any(context.provenance != "contextualized_laffa" for context in contexts):
            raise ValueError("curated Sakka accepts only contextualized Laffa descendants")
        if len({trace.rules_hash for trace in traces}) != 1:
            raise ValueError("curated Sakka rounds must use one rules revision")
        if len({trace.bundle_hash for trace in traces}) != 1:
            raise ValueError("curated Sakka rounds must use one configuration bundle")
        if len({trace.engine_version for trace in traces}) != 1:
            raise ValueError("curated Sakka rounds must use one engine version")
        releases = tuple(sorted({profile for trace in traces for _, profile in trace.agent_profiles}))
        if len(releases) != 1:
            raise ValueError("curated Sakka must use one equal frozen agent release")
        commitments = tuple(
            str(trace.seed_provenance.get("deckCommitment") or "")
            for trace in traces
        )
        if not all(commitments) or len(set(commitments)) != len(commitments):
            raise ValueError("curated Sakka cannot reuse a dealt-card fingerprint")

        expected_score = _start(traces[0])["score"]
        starting_score = expected_score
        previous_dealer: Seat | None = None
        previous_round_number: int | None = None
        rounds: list[dict[str, Any]] = []
        for index, (context, trace) in enumerate(zip(contexts, traces)):
            start = _start(trace)
            finish = _finish(trace)
            if start["score"] != expected_score:
                raise ValueError("curated Sakka score continuity is broken")
            dealer = Seat.parse(start["dealer"])
            if previous_dealer is not None and dealer != previous_dealer.next():
                raise ValueError("curated Sakka dealer progression is broken")
            if previous_round_number is not None and start["roundNumber"] != previous_round_number + 1:
                raise ValueError("curated Sakka round progression is broken")
            if index < len(traces) - 1 and (
                max(finish) >= goal
                or trace.trusted_final_state["roundScore"].get("matchWinner") is not None
            ):
                raise ValueError("curated Sakka contains rounds after match termination")
            rounds.append({
                "ordinal": index + 1,
                "contextualizedLaffaHash": context.sha256,
                "contextHash": context.context_hash,
                "descendantTraceHash": trace.sha256,
                "analysisHash": context.analysis_hash,
                "startingScore": list(start["score"]),
                "awardedPoints": list(trace.trusted_final_state["roundScore"]["awardedPoints"]),
                "finalScore": list(finish),
                "dealer": dealer.name.lower(),
                "roundNumber": start["roundNumber"],
            })
            expected_score = finish
            previous_dealer = dealer
            previous_round_number = start["roundNumber"]
        if max(expected_score) < goal:
            raise ValueError("curated Sakka does not reach its declared target")
        winner_index = 0 if expected_score[0] > expected_score[1] else 1
        winner = "north_south" if winner_index == 0 else "west_east"
        if traces[-1].trusted_final_state["roundScore"].get("matchWinner") != winner:
            raise ValueError("curated Sakka target does not match the engine terminal outcome")
        round_values = tuple(rounds)
        core = cls._core(
            target=goal,
            rules_hash=traces[0].rules_hash,
            bundle_hash=traces[0].bundle_hash,
            engine_version=traces[0].engine_version,
            agent_releases=releases,
            rounds=round_values,
            starting_score=starting_score,
            final_score=expected_score,
            winner=winner,
        )
        composition_hash = canonical_sha256(core)
        return cls(
            goal,
            traces[0].rules_hash,
            traces[0].bundle_hash,
            traces[0].engine_version,
            releases,
            round_values,
            starting_score,
            expected_score,
            winner,
            "curated_sakka",
            composition_hash,
            signer.algorithm,
            signer.key_id,
            signer.sign_hash(composition_hash),
        )

    def verify(self, signer: CuratedSakkaSignerV1) -> bool:
        if signer.algorithm != self.signature_algorithm or signer.key_id != self.signature_key_id:
            return False
        core = self._core(
            target=self.target,
            rules_hash=self.rules_hash,
            bundle_hash=self.bundle_hash,
            engine_version=self.engine_version,
            agent_releases=self.agent_releases,
            rounds=self.rounds,
            starting_score=self.starting_score,
            final_score=self.final_score,
            winner=self.winner,
        )
        digest = canonical_sha256(core)
        return hmac.compare_digest(digest, self.composition_hash) and hmac.compare_digest(
            signer.sign_hash(digest),
            self.signature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._core(
                target=self.target,
                rules_hash=self.rules_hash,
                bundle_hash=self.bundle_hash,
                engine_version=self.engine_version,
                agent_releases=self.agent_releases,
                rounds=self.rounds,
                starting_score=self.starting_score,
                final_score=self.final_score,
                winner=self.winner,
            ),
            "compositionHash": self.composition_hash,
            "signatureAlgorithm": self.signature_algorithm,
            "signatureKeyId": self.signature_key_id,
            "signature": self.signature,
        }
