from __future__ import annotations

import copy
import json
import re
from importlib.resources import files
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from baloot_platform.analysis.models import InformationState, SignalObservation
from baloot_platform.analysis.posterior import deal_likelihood
from baloot_platform.analysis.signals import (
    PRODUCTION_SIGNAL_KINDS,
    detect_signals,
)
from baloot_platform.config import is_historical_conventions_profile
from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.enums import Contract, GamePhase, Rank, Seat, Suit
from baloot_platform.core.models import PlayedCard, RoundState, Trick
from baloot_platform.core.rules import RuleBook

from .contracts import canonical_sha256


_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SEMANTIC_VERSION = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_PROFILE_STATUSES = frozenset(
    {"draft", "validated", "published", "research_default", "retired"}
)


def validate_convention_profile(
    value: Mapping[str, Any],
    *,
    _historical_three_signal_schema: bool = False,
) -> None:
    """Validate convention syntax, provenance, and safe probability bounds."""

    schema_path = files("baloot_platform.resources").joinpath(
        "schemas/profiles/conventions.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if _historical_three_signal_schema or is_historical_conventions_profile(value):
        historical_signals = {
            "high_then_low",
            "low_then_high",
            "partner_return",
        }
        signal_schema = schema["properties"]["signals"]
        signal_schema["properties"] = {
            key: item
            for key, item in signal_schema["properties"].items()
            if key in historical_signals
        }
        signal_schema["required"] = sorted(historical_signals)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(value)),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ValueError(
            f"convention profile is outside the governed schema at {location}: "
            f"{first.message}"
        )

    profile = dict(value["profile"])
    if not _PROFILE_ID.fullmatch(str(profile["id"])):
        raise ValueError("convention profile id must be a stable snake-case identifier")
    if not _SEMANTIC_VERSION.fullmatch(str(profile["version"])):
        raise ValueError("convention profile version must be semantic versioning")
    if str(profile["status"]) not in _PROFILE_STATUSES:
        raise ValueError("convention profile status is unsupported")

    forcedness = dict(value["forcedness"])
    ordered = tuple(
        float(forcedness[key])
        for key in ("forced", "near_forced", "meaningful_choice", "highly_optional")
    )
    if ordered != tuple(sorted(ordered)):
        raise ValueError("convention forcedness weights must be monotonic")


def _replace_path(payload: dict[str, Any], primitive_id: str, value: Any) -> None:
    parts = primitive_id.split(".")[1:]
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _actor_for_relationship(observer: Seat, relationship: str) -> Seat:
    if relationship == "self":
        return observer
    if relationship == "partner":
        return observer.partner
    if relationship == "right_opponent":
        return observer.next()
    if relationship == "left_opponent":
        return observer.previous()
    raise ValueError(f"unsupported relationship: {relationship}")


def _scenario(
    rules: RuleBook,
    *,
    signal: str,
    relationship: str = "partner",
    legal_count: int = 5,
    branch: str = "default",
) -> tuple[RoundState, InformationState, SignalObservation]:
    observer = Seat.SOUTH
    actor = _actor_for_relationship(observer, relationship)
    state = RoundState(rules.sha256, 48017, 1, Seat.EAST)
    state.phase = GamePhase.PLAYING
    state.contract = Contract.HOKUM
    state.buyer = Seat.SOUTH
    state.declarer = Seat.SOUTH
    state.trump_suit = Suit.CLUBS
    state.first_leader = actor
    state.current_player = observer
    def alternatives(played: Card) -> tuple[Card, ...]:
        return (
            played,
            Card.of(Suit.CLUBS, Rank.EIGHT),
            Card.of(Suit.DIAMONDS, Rank.NINE),
            Card.of(Suit.SPADES, Rank.TEN),
            Card.of(Suit.CLUBS, Rank.QUEEN),
        )[:legal_count]

    if signal == "low_then_high":
        played = Card.of(Suit.HEARTS, Rank.SEVEN)
        plays = (
            PlayedCard(
                actor,
                played,
                0,
                0,
                alternatives(played),
            ),
            PlayedCard(actor, Card.of(Suit.HEARTS, Rank.EIGHT), 1, 0),
        )
    elif signal == "high_then_low":
        played = Card.of(Suit.HEARTS, Rank.ACE)
        plays = (
            PlayedCard(
                actor,
                played,
                0,
                0,
                alternatives(played),
            ),
            PlayedCard(actor, Card.of(Suit.HEARTS, Rank.TEN), 1, 0),
        )
    elif signal == "partner_return":
        played = Card.of(Suit.HEARTS, Rank.SEVEN)
        plays = (
            PlayedCard(
                actor.partner,
                Card.of(Suit.HEARTS, Rank.EIGHT),
                0,
                0,
            ),
            PlayedCard(
                actor,
                played,
                1,
                0,
                alternatives(played),
            ),
        )
    elif signal in {"minimum_winner", "high_expenditure"}:
        if actor != observer:
            raise ValueError(
                f"{signal} is observer-safe only for the observer's own choice"
            )
        legal_winners = (
            (
                Card.of(Suit.CLUBS, Rank.KING),
                Card.of(Suit.CLUBS, Rank.TEN),
                Card.of(Suit.CLUBS, Rank.JACK),
            )
            if branch == "error_or_deception"
            else (
                Card.of(Suit.CLUBS, Rank.TEN),
                Card.of(Suit.CLUBS, Rank.ACE),
                Card.of(Suit.CLUBS, Rank.JACK),
            )
        )
        if signal == "minimum_winner":
            chosen = legal_winners[0]
        elif branch == "error_or_deception":
            chosen = legal_winners[1]
        else:
            chosen = legal_winners[-1]
        plays = (
            PlayedCard(Seat.EAST, Card.of(Suit.CLUBS, Rank.SEVEN), 0, 0),
            PlayedCard(Seat.NORTH, Card.of(Suit.CLUBS, Rank.EIGHT), 0, 1),
            PlayedCard(Seat.WEST, Card.of(Suit.CLUBS, Rank.QUEEN), 0, 2),
            PlayedCard(observer, chosen, 0, 3, legal_winners),
        )
    else:
        raise ValueError(f"no production detector scenario for {signal}")
    state.completed_tricks = []
    state.current_trick = Trick(
        0,
        plays[0].seat,
        list(plays),
    )
    state.hands = [0, 0, 0, 0]
    remaining: tuple[Card, ...] = ()
    if branch in {"encourage_suit", "support_partner_suit"}:
        remaining = (Card.of(Suit.HEARTS, Rank.ACE),)
    elif branch == "discourage_suit":
        remaining = (Card.of(Suit.HEARTS, Rank.EIGHT),)
    elif branch == "tactical_return":
        remaining = (Card.of(Suit.HEARTS, Rank.NINE),)
    state.hands[int(actor)] = cards_to_mask(remaining)
    state.purchase_card = None
    state.projects = []
    information = InformationState.from_complete(state, observer)
    if any(
        play.seat != observer and play.legal_cards for play in information.plays
    ):
        raise AssertionError("observer projection leaked another player's alternatives")
    matched = tuple(
        observation
        for observation in detect_signals(information, rules.play_domain)
        if observation.kind == signal and observation.seat == actor
    )
    if len(matched) != 1:
        raise ValueError(
            f"production detector did not yield exactly one {signal} observation"
        )
    observation = matched[0]
    return state, information, observation


def _counter_value(primitive_id: str, value: float) -> float:
    if ".likelihood_ratios." in primitive_id:
        return 1.23 if value != 1.23 else 0.91
    if ".relationship_reliability." in primitive_id:
        return 0.4 if value != 0.4 else 0.7
    if ".forcedness." in primitive_id:
        return {
            "forced": 0.10,
            "near_forced": 0.25,
            "meaningful_choice": 0.55,
            "highly_optional": 0.80,
        }[primitive_id.rsplit(".", 1)[-1]]
    raise ValueError(f"no numeric counterfactual for {primitive_id}")


def convention_behavioral_proofs(
    *, conventions: Mapping[str, Any], rules: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Prove each convention leaf without promoting soft evidence to fact."""

    source = copy.deepcopy(dict(conventions))
    validate_convention_profile(source)
    historical_three_signal_schema = is_historical_conventions_profile(source)

    def validate_proof_variant(value: Mapping[str, Any]) -> None:
        validate_convention_profile(
            value,
            _historical_three_signal_schema=historical_three_signal_schema,
        )

    unsupported = {
        str(kind)
        for kind, settings in dict(source["signals"]).items()
        if bool(dict(settings)["enabled"])
    } - PRODUCTION_SIGNAL_KINDS
    if unsupported:
        raise ValueError(
            "enabled probability conventions lack production weighting: "
            f"{sorted(unsupported)}"
        )
    rules_book = RuleBook(dict(rules))
    rules_hash = rules_book.sha256
    profile_hash = canonical_sha256(source)
    proofs: dict[str, dict[str, Any]] = {}

    metadata_counterexamples = {
        "conventions.schema_version": 2,
        "conventions.profile.id": "INVALID ID",
        "conventions.profile.version": "version-one",
        "conventions.profile.status": "untracked",
    }
    for primitive_id, counterexample in metadata_counterexamples.items():
        mutated = copy.deepcopy(source)
        _replace_path(mutated, primitive_id, counterexample)
        try:
            validate_proof_variant(mutated)
        except ValueError as exc:
            core = {
                "schemaVersion": 1,
                "primitiveId": primitive_id,
                "status": "proven_invariant",
                "proofKind": "convention_schema_or_provenance_rejection",
                "rulesHash": rules_hash,
                "profileHash": profile_hash,
                "counterexampleHash": canonical_sha256(counterexample),
                "rejectionType": type(exc).__name__,
                "targetsObserved": ["policy", "documentation"],
                "proofBoundary": "closed_schema_and_profile_provenance",
                "softEvidenceOnly": True,
            }
            proofs[primitive_id] = {
                **core,
                "proofHash": canonical_sha256(core),
            }
        else:
            raise ValueError(
                f"convention metadata counterexample was accepted for {primitive_id}"
            )

    ratio_scenarios = {
        "conventions.signals.low_then_high.likelihood_ratios.encourage_suit": (
            "low_then_high",
            "encourage_suit",
        ),
        "conventions.signals.low_then_high.likelihood_ratios.no_signal_or_deception": (
            "low_then_high",
            "no_signal_or_deception",
        ),
        "conventions.signals.high_then_low.likelihood_ratios.short_suit": (
            "high_then_low",
            "short_suit",
        ),
        "conventions.signals.high_then_low.likelihood_ratios.discourage_suit": (
            "high_then_low",
            "discourage_suit",
        ),
        "conventions.signals.partner_return.likelihood_ratios.support_partner_suit": (
            "partner_return",
            "support_partner_suit",
        ),
        "conventions.signals.partner_return.likelihood_ratios.tactical_return": (
            "partner_return",
            "tactical_return",
        ),
        "conventions.signals.minimum_winner.likelihood_ratios.preserve_strength": (
            "minimum_winner",
            "preserve_strength",
        ),
        "conventions.signals.high_expenditure.likelihood_ratios.urgency_or_signal": (
            "high_expenditure",
            "urgency_or_signal",
        ),
        "conventions.signals.high_expenditure.likelihood_ratios.error_or_deception": (
            "high_expenditure",
            "error_or_deception",
        ),
    }

    behavioral: list[tuple[str, str, str, int, str]] = [
        (
            "conventions.apply_to_probability",
            "low_then_high",
            "partner",
            5,
            "encourage_suit",
        )
    ]
    for relationship in ("self", "partner", "left_opponent", "right_opponent"):
        behavioral.append(
            (
                f"conventions.relationship_reliability.{relationship}",
                "low_then_high",
                relationship,
                5,
                "encourage_suit",
            )
        )
    for forcedness, count in (
        ("forced", 1),
        ("near_forced", 2),
        ("meaningful_choice", 3),
        ("highly_optional", 5),
    ):
        behavioral.append(
            (
                f"conventions.forcedness.{forcedness}",
                "low_then_high",
                "partner",
                count,
                "encourage_suit",
            )
        )
    for signal in sorted(source["signals"]):
        branch = {
            "low_then_high": "encourage_suit",
            "high_then_low": "discourage_suit",
            "partner_return": "support_partner_suit",
            "minimum_winner": "preserve_strength",
            "high_expenditure": "urgency_or_signal",
        }.get(signal, "default")
        relationship = (
            "self"
            if signal in {"minimum_winner", "high_expenditure"}
            else "partner"
        )
        behavioral.append(
            (
                f"conventions.signals.{signal}.enabled",
                signal,
                relationship,
                5,
                branch,
            )
        )
    behavioral.extend(
        (
            primitive_id,
            signal,
            (
                "self"
                if signal in {"minimum_winner", "high_expenditure"}
                else "partner"
            ),
            5,
            branch,
        )
        for primitive_id, (signal, branch) in sorted(ratio_scenarios.items())
        if signal in source["signals"]
    )
    counterexample_signal = {
        "high_expenditure": "minimum_winner",
        "high_then_low": "low_then_high",
        "low_then_high": "high_then_low",
        "minimum_winner": "high_expenditure",
        "partner_return": "low_then_high",
    }

    for primitive_id, signal, relationship, legal_count, branch in behavioral:
        baseline = copy.deepcopy(source)
        if primitive_id != "conventions.apply_to_probability":
            baseline["apply_to_probability"] = True
        counterfactual = copy.deepcopy(baseline)
        parts = primitive_id.split(".")[1:]
        cursor: Any = counterfactual
        source_cursor: Any = baseline
        for part in parts[:-1]:
            cursor = cursor[part]
            source_cursor = source_cursor[part]
        leaf = parts[-1]
        if primitive_id == "conventions.apply_to_probability":
            cursor[leaf] = not bool(source_cursor[leaf])
        elif primitive_id.endswith(".enabled"):
            cursor[leaf] = not bool(source_cursor[leaf])
        else:
            cursor[leaf] = _counter_value(primitive_id, float(source_cursor[leaf]))
        validate_proof_variant(baseline)
        validate_proof_variant(counterfactual)
        state, information, observation = _scenario(
            rules_book,
            signal=signal,
            relationship=relationship,
            legal_count=legal_count,
            branch=branch,
        )
        comparison_signal = counterexample_signal[signal]
        comparison_state, comparison_information, _comparison_observation = _scenario(
            rules_book,
            signal=comparison_signal,
            relationship=(
                "self"
                if comparison_signal in {"minimum_winner", "high_expenditure"}
                else relationship
            ),
            legal_count=legal_count,
            branch=(
                "urgency_or_signal"
                if comparison_signal == "high_expenditure"
                else "default"
            ),
        )
        comparison_kinds = tuple(
            item.kind
            for item in detect_signals(
                comparison_information,
                rules_book.play_domain,
            )
        )
        if signal in comparison_kinds:
            raise AssertionError(
                f"detector counterexample unexpectedly produced {signal}"
            )
        fixtures = (state, comparison_state)
        fixture_hashes_before = tuple(
            canonical_sha256(item.to_dict(reveal_hands=True)) for item in fixtures
        )
        baseline_weight = deal_likelihood(
            state,
            information,
            (observation,),
            baseline,
            play_domain=rules_book.play_domain,
        )
        counterfactual_weight = deal_likelihood(
            state,
            information,
            (observation,),
            counterfactual,
            play_domain=rules_book.play_domain,
        )
        if baseline_weight == counterfactual_weight:
            raise ValueError(
                f"convention counterfactual did not influence {primitive_id}"
            )
        fixture_hashes_after = tuple(
            canonical_sha256(item.to_dict(reveal_hands=True)) for item in fixtures
        )
        if fixture_hashes_after != fixture_hashes_before:
            raise AssertionError("convention weighting mutated a detector fixture")
        public_hash = canonical_sha256(information.to_dict())
        core = {
            "schemaVersion": 1,
            "primitiveId": primitive_id,
            "status": "proven_behavioral",
            "proofKind": "observer_safe_detector_fixture_likelihood_counterfactual",
            "rulesHash": rules_hash,
            "profileHash": profile_hash,
            "activationProfileHash": canonical_sha256(baseline),
            "counterfactualProfileHash": canonical_sha256(counterfactual),
            "observerInformationHash": public_hash,
            "detectorFixtureStateHashes": list(fixture_hashes_before),
            "observationKind": signal,
            "observationHash": canonical_sha256(observation.to_dict()),
            "counterexampleObservationKinds": list(comparison_kinds),
            "productionDetector": "analysis.signals.detect_signals",
            "baselineWeight": baseline_weight,
            "counterfactualWeight": counterfactual_weight,
            "targetsObserved": ["policy"],
            "observerProjectionSafe": True,
            "softEvidenceOnly": True,
            "candidateStateMutation": False,
            "legalCandidateSetChanged": False,
            "proofBoundary": "observer_safe_detector_fixture_soft_reweighting",
        }
        proofs[primitive_id] = {
            **core,
            "proofHash": canonical_sha256(core),
        }

    expected = {
        primitive_id
        for path, _leaf in _leaf_paths(source)
        for primitive_id in ("conventions." + ".".join(path),)
    }
    if set(proofs) != expected:
        missing = sorted(expected - set(proofs))
        extra = sorted(set(proofs) - expected)
        raise ValueError(
            f"convention proof coverage mismatch: missing={missing} extra={extra}"
        )
    return proofs


def _leaf_paths(
    value: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    rows: list[tuple[tuple[str, ...], Any]] = []
    for key, item in sorted(value.items()):
        path = (*prefix, str(key))
        if isinstance(item, Mapping):
            rows.extend(_leaf_paths(item, path))
        else:
            rows.append((path, item))
    return tuple(rows)
