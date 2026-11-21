import json
from pathlib import Path

import pytest

from baloot_platform.research.calibration.score import engine_interest_rank_key_v1, engine_interest_v1
from baloot_platform.research.lesson_tags import classify_public_mechanism_v1


@pytest.fixture
def candidate():
    return {
        "exactStateId": "candidate", "action": "SA", "legal": ["SA", "H8", "D7"],
        "roundWin": {"SA": 1, "H8": -1, "D7": -1}, "ddGap": 30, "ddSwing": 50,
        "worldCoverageComplete": True, "bestShare": 1, "strictShare": 1, "worldMeanGap": 20,
        "baselineRegret": 20, "naiveRegret": 20, "interestingnessV4": {"difficulty_prior": 50},
        "legalReplayVerified": True, "hardDeductions": 1, "exactActorProofVerified": True,
        "finalMargin": 40, "playIndex": 27, "auctionSetupBeats": 10, "durationSeconds": 44.25,
        "publicationGateReasons": [], "scenarioFamily": "family-1", "visibleOpenerSha256": "a" * 64,
        "captionSha256": "b" * 64, "mechanism": "LEAD", "contract": "sun", "trickBand": "24-31",
    }


def test_calibrated_ranking_and_screen_rejections():
    # Cached world measurements avoid running 16 exhaustive solvers in the fast CI lane.
    path = Path(__file__).with_name("fixtures") / "engine_interest_v1.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    cards = {name: engine_interest_v1({**row, "exactStateId": name}) for name, row in rows.items()}
    rejections = {
        "18-03": ["G3_viewer_unfair"], "16-05": ["G3_viewer_unfair"], "19-03": ["G3_viewer_unfair"],
        "unique-saving": ["G2_no_decision", "G3_viewer_unfair"],
        "golden": ["G2_no_decision", "G3_viewer_unfair"],
    }
    for name, card in cards.items():
        assert [reason for reason in card["gateReasons"]
                if reason in {"G1_clarity", "G2_no_decision", "G3_viewer_unfair"}] == rejections.get(name, [])
        assert "G3_exact_actor_proof_unverified" in card["gateReasons"]
        assert "V_history_unavailable" in card["gateReasons"]
        assert card["postAllowed"] is False
    ranked = sorted(cards, key=lambda name: (name in rejections, engine_interest_rank_key_v1(cards[name])))
    assert [(name, cards[name]["score"]) for name in ranked] == [
        ("16-02", 79), ("two-aces", 75), ("16-01", 68.6), ("18-01", 66.8),
        ("ace-seven", 66.3), ("16-04", 61.3), ("21-01", 61.3), ("16-03", 59.7),
        ("18-04", 59), ("16-06", 55.9), ("18-02", 53.3), ("18-03", 69.2),
        ("16-05", 60.2), ("19-03", 60.2), ("unique-saving", 35.3), ("golden", 24),
    ]
    assert {name for name, card in cards.items() if name not in rejections
            and "below_minimum_score" in card["gateReasons"]} == {"16-03", "16-06", "18-02", "18-04"}


def test_complete_card_is_versioned_deterministic_and_scores_100(candidate):
    card = engine_interest_v1(candidate, posted_history=[])
    assert card == engine_interest_v1(dict(reversed(list(candidate.items()))), posted_history=[])
    assert card["scoreVersion"] == 1
    assert card["weightsSha256"] == "a0c9961a2ce7edd28a994afbda94472650d9639c1214a1def2cde8eea4495e66"
    assert card["policySha256"] == "5f3aad6f3ac59703eaf11aa045a95749a72fc87e0a8460befab7b79722821779"
    assert card["score"] == 100
    assert card["parts"] == {"D": 25, "R": 10, "S": 25, "K": 15, "C": 10, "B": 0, "V": 15}
    assert card["gateStatus"] == {"T": True, "F": True} and card["action"] == candidate["action"]
    assert card["eligible"] and card["postAllowed"] and card["gateReasons"] == []
    with pytest.raises(ValueError, match="frozen policy"):
        engine_interest_v1(candidate, {"schemaVersion": 1, "fairness": {"minimumBestShare": 0.89}})


@pytest.mark.parametrize(("updates", "reason"), [
    ({"legalReplayVerified": False}, "G0_legal_replay_unverified"),
    ({"legal": ["SA", "SA+ekka"]}, "G1_clarity"),
    ({"legal": ["SA", "H8", "D7", "S7", "C8"]}, "G1_clarity"),
    ({"hardDeductions": 0}, "G1_clarity"),
    ({"roundWin": {"SA": 1, "H8": 1, "D7": -1}}, "G2_no_decision"),
    ({"roundWin": {"SA": -1, "H8": 1, "D7": -1}}, "G2_no_decision"),
    ({"roundWin": {"SA": -1, "H8": -1, "D7": -1}}, "G2_no_decision"),
    ({"roundWin": {"SA": 1, "H8": -1}}, "G2_no_decision"),
    ({"worldCoverageComplete": False}, "G3_viewer_unfair"),
    ({"bestShare": 0.899999, "strictShare": 0.85}, "G3_viewer_unfair"),
    ({"worldMeanGap": 1.999999}, "G3_viewer_unfair"),
    ({"exactActorProofVerified": False}, "G3_exact_actor_proof_unverified"),
    ({"finalMargin": 41}, "G4_final_margin"),
    ({"finalMargin": None}, "G4_final_margin"),
    ({"playIndex": 28}, "G6_timing"),
    ({"auctionSetupBeats": 11}, "G6_timing"),
    ({"durationSeconds": 43.749}, "G6_timing"),
    ({"durationSeconds": 44.251}, "G6_timing"),
    ({"durationSeconds": None}, "G6_timing"),
    ({"publicationGateReasons": None}, "publication_qa_unverified"),
    ({"publicationGateReasons": ""}, "publication_qa_unverified"),
    ({"publicationGateReasons": ["layoutGeometry"]}, "layoutGeometry"),
])
def test_each_hard_gate_blocks_even_a_high_score(candidate, updates, reason):
    card = engine_interest_v1({**candidate, **updates}, posted_history=[])
    assert reason in card["gateReasons"] and not card["eligible"] and not card["postAllowed"]


def test_inclusive_viewer_and_timing_boundaries(candidate):
    row = {**candidate, "ddGap": 0, "worldMeanGap": 2, "strictShare": 0.90, "bestShare": 0.90, "durationSeconds": 43.75}
    card = engine_interest_v1(row, posted_history=[])
    assert card["postAllowed"] and card["gateReasons"] == []
    assert card["parts"]["D"] == 2.5 and card["parts"]["R"] == 9


@pytest.mark.parametrize("updates", [{"bestShare": True}, {"worldMeanGap": float("nan")}, {"ddSwing": float("inf")}])
def test_invalid_measurements_fail_closed(candidate, updates):
    with pytest.raises(ValueError, match="invalid ENGINE-INTEREST"):
        engine_interest_v1({**candidate, **updates}, posted_history=[])


def test_score_floor_uses_unrounded_total(candidate):
    row = {**candidate, "worldMeanGap": 16, "strictShare": 0, "baselineRegret": 19, "naiveRegret": 19}
    assert engine_interest_v1(row, posted_history=[])["score"] == 60
    assert engine_interest_v1(row, posted_history=[])["postAllowed"]
    card = engine_interest_v1({**row, "worldMeanGap": 15.999, "naiveRegret": 20, "ddSwing": 20,
                               "interestingnessV4": {"difficulty_prior": 76}, "strictShare": 0.5}, posted_history=[])
    assert card["score"] == 60 and card["eligible"] and not card["postAllowed"]
    assert card["gateReasons"] == ["below_minimum_score"]


def test_interest_floor_needs_surprise_or_decision(candidate):
    row = {**candidate, "worldMeanGap": 15, "baselineRegret": None, "naiveRegret": 19}
    card = engine_interest_v1(row, posted_history=[])
    assert card["score"] > 60 and card["eligible"] and card["gateReasons"] == ["below_interest_floor"]
    assert not card["postAllowed"]
    assert engine_interest_v1({**row, "naiveRegret": 20}, posted_history=[])["postAllowed"]


@pytest.mark.parametrize(("prior", "points"), [(24.999, -10), (25, 0), (75, 0), (75.001, -10)])
def test_difficulty_band(candidate, prior, points):
    card = engine_interest_v1({**candidate, "interestingnessV4": {"difficulty_prior": prior}}, posted_history=[])
    assert card["parts"]["B"] == points


def test_clarity_counts_cards_and_round_win_accepts_ekka_variants_of_the_same_card(candidate):
    row = {**candidate, "legal": [*candidate["legal"], "C7", "SA+ekka"],
           "roundWin": {**candidate["roundWin"], "C7": -1, "SA+ekka": 1}}
    card = engine_interest_v1(row, posted_history=[])
    assert card["postAllowed"] and card["parts"]["C"] == 6


def test_variety_uses_actual_latest_previous_and_stale_cell(candidate):
    other = {**candidate, "scenarioFamily": "family-2", "visibleOpenerSha256": "c" * 64,
             "captionSha256": "d" * 64, "mechanism": "FOLLOW_SUIT_CAPTURE"}
    same_mechanism = {**other, "mechanism": "LEAD"}
    other_cell = {**other, "trickBand": "16-23"}
    latest = engine_interest_v1(candidate, posted_history=[same_mechanism, other])
    assert latest["gateReasons"] == ["V_same_as_latest_mechanism"] and latest["parts"]["V"] == 0
    recent = engine_interest_v1(candidate, posted_history=[other, same_mechanism])
    assert recent["postAllowed"] and recent["parts"]["V"] == 7.5 and recent["mechanismFrequency"] == 1
    stale = engine_interest_v1(candidate, posted_history=[other, same_mechanism, other])
    assert stale["postAllowed"] and stale["parts"]["V"] == 0
    fresh = engine_interest_v1(candidate, posted_history=[other, other, other_cell, *([other] * 7), same_mechanism])
    assert fresh["postAllowed"] and fresh["parts"]["V"] == 15 and fresh["mechanismFrequency"] == 0


@pytest.mark.parametrize("key", ["scenarioFamily", "visibleOpenerSha256", "captionSha256"])
def test_repeats_block_even_outside_the_last_ten_posts(candidate, key):
    other = {**candidate, "scenarioFamily": "family-2", "visibleOpenerSha256": "c" * 64,
             "captionSha256": "d" * 64, "mechanism": "FOLLOW_SUIT_CAPTURE"}
    card = engine_interest_v1(candidate, posted_history=[other] * 10 + [{**other, key: candidate[key]}])
    assert card["gateReasons"] == [f"V_repeat_{key}"] and not card["postAllowed"]


@pytest.mark.parametrize(("first", "second"), [
    ({"score": 81}, {"parts": {"S": 11, "D": 6}}),
    ({"parts": {"S": 11, "D": 5}}, {"parts": {"S": 10, "D": 6}}),
    ({"parts": {"S": 10, "D": 6}}, {"difficultyPrior": 52}),
    ({"difficultyPrior": 52}, {"mechanismFrequency": 1}),
    ({"mechanismFrequency": 1}, {"exactStateId": "a"}),
    ({"exactStateId": "a"}, {}),
])
def test_rank_breaks_ties_in_exact_order(first, second):
    card = {"score": 80, "parts": {"S": 10, "D": 5}, "difficultyPrior": 47,
            "mechanismFrequency": 2, "exactStateId": "b"}
    assert engine_interest_rank_key_v1({**card, **first}) < engine_interest_rank_key_v1({**card, **second})


@pytest.mark.parametrize(("table", "contract", "trump", "action", "mechanism"), [
    ([], "sun", None, "SA", "LEAD"),
    (["S7", "HA"], "sun", None, "S10", "FOLLOW_SUIT_CAPTURE"),
    (["S10"], "sun", None, "SJ", "FOLLOW_SUIT_UNDERPLAY"),
    (["S7", "H7"], "hokum", "hearts", "SA+ekka", "FOLLOW_SUIT_UNDERPLAY"),
    (["H10"], "hokum", "hearts", "H9", "FOLLOW_SUIT_CAPTURE"),
    (["S7"], "sun", None, "H7", "VOID_OFF_SUIT_DISCARD"),
    (["S7"], "hokum", "hearts", "H7", "TRUMP_RUFF"),
])
def test_mechanism_uses_visible_cards_and_contract(rules, table, contract, trump, action, mechanism):
    assert classify_public_mechanism_v1(rules=rules, current_trick={"plays": [{"card": card} for card in table]},
                                       contract=contract, trump=trump, action=action) == mechanism
