from __future__ import annotations

from types import SimpleNamespace

import pytest

from baloot_platform.challenge.evidence_v3 import TruthMode
from baloot_platform.challenge.funnel import FoundryAnalysisFunnelV1
from baloot_platform.challenge.oracle import (
    ActualDealActionOracleV1,
    OracleTrickFactV1,
    derive_exact_line_causality,
)
from baloot_platform.challenge.tactical_evidence import (
    CausalActionLineV1,
    CausalTacticalDetectorV1,
    TacticalDetectorProfileV1,
    TacticalKind,
)
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit


def _fact(
    ordinal,
    *,
    leader,
    winner,
    lead,
    winning,
    strength,
    points,
    actor_trumps,
    opponent_trumps,
    plays,
):
    return OracleTrickFactV1(
        ordinal,
        leader,
        winner,
        lead,
        winning,
        strength,
        tuple(plays),
        points,
        actor_trumps,
        opponent_trumps,
    )


def _action(card, *, strength, immediate, final, facts, digest):
    return ActualDealActionOracleV1(
        Card.parse(card).to_dict(),
        strength,
        immediate,
        final,
        True,
        False,
        False,
        100,
        (card,),
        tuple(facts),
        "exact",
        digest * 64,
    )


def _profile(**overrides):
    value = {
        "schemaVersion": 1,
        "id": "causal_detector_test_v1",
        "revision": "1",
        "lowCardMaxStrength": 35.0,
        "highCardMinStrength": 70.0,
        "minimumDelayedGain": 8.0,
        "minimumCausalGain": 0.01,
        "kaboutProbabilityDelta": 0.20,
        "contractProbabilityDelta": 0.20,
        "projectConsequenceDelta": 5.0,
        "debateMarginGap": 2.0,
        "debateDisagreement": 0.25,
    }
    value.update(overrides)
    return TacticalDetectorProfileV1.build(value)


def test_paired_exact_trick_facts_prove_delayed_causal_mechanisms():
    selected = _action(
        "H7",
        strength=10,
        immediate=-5,
        final=20,
        digest="a",
        facts=(
            _fact(
                1,
                leader=Seat.EAST,
                winner=Seat.EAST,
                lead="H10",
                winning="HJ",
                strength=90,
                points=-5,
                actor_trumps=2,
                opponent_trumps=1,
                plays=(("east", "H10"), ("south", "H7"), ("west", "H9"), ("north", "HJ")),
            ),
            _fact(
                2,
                leader=Seat.EAST,
                winner=Seat.NORTH,
                lead="H10",
                winning="H8",
                strength=20,
                points=15,
                actor_trumps=1,
                opponent_trumps=0,
                plays=(("east", "H10"), ("south", "S7"), ("west", "H9"), ("north", "H8")),
            ),
        ),
    )
    alternative = _action(
        "HA",
        strength=100,
        immediate=5,
        final=0,
        digest="b",
        facts=(
            _fact(
                1,
                leader=Seat.EAST,
                winner=Seat.EAST,
                lead="H10",
                winning="HJ",
                strength=90,
                points=-5,
                actor_trumps=1,
                opponent_trumps=1,
                plays=(("east", "H10"), ("south", "HA"), ("west", "H9"), ("north", "HJ")),
            ),
            _fact(
                2,
                leader=Seat.EAST,
                winner=Seat.WEST,
                lead="S10",
                winning="SA",
                strength=100,
                points=-15,
                actor_trumps=0,
                opponent_trumps=1,
                plays=(("east", "S10"), ("south", "S7"), ("west", "SA"), ("north", "S8")),
            ),
        ),
    )
    causality = derive_exact_line_causality(
        selected,
        alternative,
        acting_seat=Seat.SOUTH,
        trump_suit=Suit.HEARTS,
    )
    assert causality.payoff_depth == 2
    assert set(causality.mechanisms) == {
        "entry",
        "forced_return",
        "partner_enablement",
        "promotion",
        "trump_exhaustion",
        "trump_extraction",
        "trump_preservation",
    }

    selected_line = CausalActionLineV1(
        "H7", 10, -5, 20, True, 0.0, 0.0, causality.payoff_depth,
        causality.mechanisms, ("H7", "HJ", "H8"),
        causal_source_hash=causality.sha256,
    )
    alternative_line = CausalActionLineV1(
        "HA", 100, 5, 0, False, 0.0, 0.0, 0, (), ("HA", "HJ", "SA"),
    )
    evidence = CausalTacticalDetectorV1(_profile()).detect(
        selected_line, (alternative_line,)
    )
    assert CausalTacticalDetectorV1(_profile()).descriptor.version == "1.1.0"
    kinds = {item.kind for item in evidence}
    assert TacticalKind.SMALL_CARD_BIG_GAIN in kinds
    assert TacticalKind.ENTRY_UNBLOCKING in kinds
    assert TacticalKind.PARTNER_ENABLEMENT in kinds
    assert TacticalKind.TRUMP_CONTROL in kinds
    assert all(item.criteria for item in evidence)
    assert all(item.causal_proof_hash for item in evidence)


def test_shared_detector_profile_preserves_legacy_identity_and_worker_transport():
    import pickle
    from dataclasses import replace
    from baloot_platform.tactics import TacticalDetectorProfileV1 as SharedProfile
    from baloot_platform.challenge.evidence_v3 import canonical_sha256

    assert SharedProfile is TacticalDetectorProfileV1
    profile = _profile()
    raw = profile.to_dict()
    digest = raw.pop('sha256')
    assert digest == canonical_sha256(raw)
    assert SharedProfile.build(raw).to_dict() == profile.to_dict()
    assert pickle.loads(pickle.dumps(profile)) == profile
    profile.validate()
    with pytest.raises(ValueError, match='hash mismatch'):
        replace(profile, low_card_max_strength=36).validate()


def test_detector_profile_defaults_keep_exact_legacy_document_hash_and_constructor():
    digest = "797bc2543b4da73c5352256a4c9bbfd6eb09e1186bab2f63fd0a552f31dfc98e"
    expected = {
        "schemaVersion": 1,
        "id": "causal_tactical_detector_default_v1",
        "revision": "1",
        "lowCardMaxStrength": 35.0,
        "highCardMinStrength": 70.0,
        "minimumDelayedGain": 8.0,
        "minimumCausalGain": 0.01,
        "kaboutProbabilityDelta": 0.20,
        "contractProbabilityDelta": 0.20,
        "projectConsequenceDelta": 5.0,
        "debateMarginGap": 2.0,
        "debateDisagreement": 0.25,
        "sha256": digest,
    }
    profile = TacticalDetectorProfileV1.defaults()
    assert profile.to_dict() == expected
    assert profile.native_delayed_gain_metric is None
    assert profile.native_delayed_minimum_gain is None
    assert TacticalDetectorProfileV1(
        "causal_tactical_detector_default_v1", "1", 35.0, 70.0, 8.0, 0.01,
        0.20, 0.20, 5.0, 2.0, 0.25, digest,
    ) == profile
    profile.validate()


@pytest.mark.parametrize("metric", ["raw_card_points", "awarded_point_margin"])
@pytest.mark.parametrize("minimum_gain", [0, 8, 8.5, 100])
def test_native_delayed_payoff_profile_round_trips_and_crosses_worker_transport(metric, minimum_gain):
    import pickle
    from baloot_platform.challenge.evidence_v3 import canonical_sha256

    profile = _profile(nativeDelayedPayoff={"gainMetric": metric, "minimumGain": minimum_gain})
    raw = profile.to_dict()
    assert raw["nativeDelayedPayoff"] == {"gainMetric": metric, "minimumGain": float(minimum_gain)}
    assert profile.native_delayed_gain_metric == metric
    assert type(profile.native_delayed_minimum_gain) is float
    digest = raw.pop("sha256")
    assert digest == canonical_sha256(raw)
    assert TacticalDetectorProfileV1.build(raw) == profile
    assert pickle.loads(pickle.dumps(profile)) == profile
    profile.validate()


def test_native_delayed_payoff_profile_is_immutable_and_hash_binds_settings():
    from dataclasses import FrozenInstanceError, replace

    native = {"gainMetric": "raw_card_points", "minimumGain": 8}
    profile = _profile(nativeDelayedPayoff=native)
    native["minimumGain"] = 9
    raw = profile.to_dict()
    raw["nativeDelayedPayoff"]["gainMetric"] = "awarded_point_margin"
    assert profile.to_dict()["nativeDelayedPayoff"] == {"gainMetric": "raw_card_points", "minimumGain": 8.0}
    assert len({
        _profile().sha256,
        profile.sha256,
        _profile(nativeDelayedPayoff={"gainMetric": "awarded_point_margin", "minimumGain": 8}).sha256,
        _profile(nativeDelayedPayoff={"gainMetric": "raw_card_points", "minimumGain": 9}).sha256,
    }) == 4
    with pytest.raises(FrozenInstanceError):
        profile.native_delayed_minimum_gain = 9
    for change in ({"native_delayed_minimum_gain": 9}, {"native_delayed_gain_metric": "awarded_point_margin"}):
        with pytest.raises(ValueError, match="hash mismatch"):
            replace(profile, **change).validate()
    profile.validate()


@pytest.mark.parametrize("native", [
    None, True, 8, "raw_card_points", [], {},
    {"gainMetric": "raw_card_points"},
    {"minimumGain": 8},
    {"gainMetric": "raw_card_points", "minimumGain": 8, "enabled": True},
])
def test_native_delayed_payoff_profile_rejects_non_objects_missing_and_unknown_keys(native):
    with pytest.raises(ValueError, match="requires only gainMetric and minimumGain"):
        _profile(nativeDelayedPayoff=native)


@pytest.mark.parametrize("metric", [None, True, 0, "raw_point_margin", "raw_card_points ", []])
def test_native_delayed_payoff_profile_rejects_unknown_or_non_string_metrics(metric):
    with pytest.raises(ValueError, match="gainMetric is unsupported"):
        _profile(nativeDelayedPayoff={"gainMetric": metric, "minimumGain": 8})


@pytest.mark.parametrize("minimum_gain", [
    None, True, False, "8", [], -0.01, 100.01, float("nan"), float("inf"), -float("inf"),
])
def test_native_delayed_payoff_profile_rejects_non_numeric_nonfinite_or_out_of_range_gain(minimum_gain):
    with pytest.raises(ValueError, match="minimumGain must be a finite number"):
        _profile(nativeDelayedPayoff={"gainMetric": "raw_card_points", "minimumGain": minimum_gain})


def test_detector_profile_blocks_unsupported_labels_and_rejects_unknown_keys():
    selected = CausalActionLineV1(
        "H7", 10, -5, 20, True, 0.0, 0.0, 2,
        ("entry",), ("H7", "H8"), causal_source_hash="c" * 64,
    )
    alternative = CausalActionLineV1(
        "HA", 100, 5, 0, False, 0.0, 0.0, 0, (), ("HA",),
    )
    strict = CausalTacticalDetectorV1(_profile(minimumDelayedGain=25.0))
    kinds = {item.kind for item in strict.detect(selected, (alternative,))}
    assert TacticalKind.SMALL_CARD_BIG_GAIN not in kinds
    assert TacticalKind.ENTRY_UNBLOCKING in kinds

    no_gain = CausalActionLineV1(
        "H7", 10, 0, 0, True, 0.0, 0.0, 2,
        ("entry", "trump_preservation"), ("H7",), causal_source_hash="d" * 64,
    )
    equal = CausalActionLineV1(
        "H8", 20, 0, 0, True, 0.0, 0.0, 0, (), ("H8",),
    )
    false_positive_kinds = {
        item.kind for item in CausalTacticalDetectorV1(_profile()).detect(no_gain, (equal,))
    }
    assert TacticalKind.ENTRY_UNBLOCKING not in false_positive_kinds
    assert TacticalKind.TRUMP_CONTROL not in false_positive_kinds

    with pytest.raises(ValueError, match="unknown tactical detector"):
        TacticalDetectorProfileV1.build({
            **_profile().to_dict(),
            "sha256": "e" * 64,
            "unsupported": True,
        })


def test_tactical_detector_rejects_unproven_and_wrong_direction_labels():
    profile = _profile()
    detector = CausalTacticalDetectorV1(profile)
    alternative = CausalActionLineV1(
        "HA", 100, 0, 0, False, 0.0, 0.0, 0, (), ("HA",),
        project_consequence=0.0,
        hidden_deal_disagreement=0.05,
    )

    unproven = CausalActionLineV1(
        "H7", 10, -4, 20, True, 0.0, 0.0, 3,
        ("entry", "partner_enablement", "trump_preservation"), ("H7", "H8"),
    )
    unproven_kinds = {item.kind for item in detector.detect(unproven, (alternative,))}
    assert TacticalKind.SMALL_CARD_BIG_GAIN not in unproven_kinds
    assert TacticalKind.ENTRY_UNBLOCKING not in unproven_kinds
    assert TacticalKind.PARTNER_ENABLEMENT not in unproven_kinds
    assert TacticalKind.TRUMP_CONTROL not in unproven_kinds

    immediate_winner = CausalActionLineV1(
        "H7", 10, 20, 24, True, 0.0, 0.0, 2,
        ("promotion",), ("H7", "H8"), causal_source_hash="f" * 64,
    )
    immediate_kinds = {
        item.kind for item in detector.detect(immediate_winner, (alternative,))
    }
    assert TacticalKind.SMALL_CARD_BIG_GAIN not in immediate_kinds

    project_loss = CausalActionLineV1(
        "pass", 0, 0, 1, True, 0.0, 0.0, 0, (), ("pass",),
        project_consequence=-10.0,
    )
    assert TacticalKind.PROJECT_BIDDING not in {
        item.kind for item in detector.detect(project_loss, (alternative,))
    }

    close = CausalActionLineV1(
        "H8", 20, 0, 10, True, 0.0, 0.0, 0, (), ("H8",),
        hidden_deal_disagreement=0.05,
    )
    unrelated_uncertain = CausalActionLineV1(
        "H9", 30, 0, -20, False, 0.0, 0.0, 0, (), ("H9",),
        hidden_deal_disagreement=0.90,
    )
    selected = CausalActionLineV1(
        "H10", 40, 0, 11, True, 0.0, 0.0, 0, (), ("H10",),
        hidden_deal_disagreement=0.05,
    )
    assert TacticalKind.DEBATE not in {
        item.kind
        for item in detector.detect(selected, (close, unrelated_uncertain))
    }


def test_funnel_tactical_quota_filters_post_trace_blocked_union_first():
    def snapshot(decision_id: str, ordinal: int):
        return SimpleNamespace(
            decision=SimpleNamespace(
                decision_id=decision_id,
                decision_kind="play",
                forced=False,
                ordinal=ordinal,
            ),
            state=SimpleNamespace(hands=(1, 1, 1, 1)),
        )

    snapshots = (
        snapshot("PLAY-UNRESOLVED", 30),
        snapshot("PLAY-COMPETENT", 29),
        snapshot("PLAY-ELIGIBLE", 1),
    )
    verified: list[str] = []
    funnel = object.__new__(FoundryAnalysisFunnelV1)
    funnel.maximum_oracle_decisions_per_trace = 1
    funnel.selection_profile = object()
    funnel.scorer = SimpleNamespace(
        score=lambda evidence, profile: SimpleNamespace(score=100.0)
    )
    funnel.tactical_verifier = SimpleNamespace(
        maximum_remaining_cards=32,
        verify_decision=lambda item, analysis, **kwargs: (
            verified.append(item.decision.decision_id)
            or item.decision.decision_id
        ),
    )
    analysis = SimpleNamespace(
        decisions=tuple(
            SimpleNamespace(decision_id=item.decision.decision_id)
            for item in snapshots
        ),
        laffa_evidence=SimpleNamespace(
            critical_decision_ids=("PLAY-UNRESOLVED",),
            perspectives=tuple(
                SimpleNamespace(
                    decision_id=item.decision.decision_id,
                    truth_mode=TruthMode.ACTOR_OBSERVER,
                    peak=100.0,
                )
                for item in snapshots
            ),
        ),
        adjudication_evidence={
            "competentDecisionIds": ["PLAY-COMPETENT"],
            "unresolvedDecisionIds": ["PLAY-UNRESOLVED"],
        },
        unresolved_decision_ids=("PLAY-UNRESOLVED",),
    )
    store = SimpleNamespace(replay=SimpleNamespace(decisions=snapshots))

    results = funnel._verify(None, analysis, store, None)

    assert results == ("PLAY-ELIGIBLE",)
    assert verified == ["PLAY-ELIGIBLE"]
