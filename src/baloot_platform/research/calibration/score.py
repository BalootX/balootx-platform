"""ENGINE-INTEREST v1 scoring over evidence supplied by the pipeline stages."""

from math import fsum, isfinite

from ..contracts import canonical_sha256


_POLICY = {
    "schemaVersion": 1,
    "weights": {"D": 25, "R": 10, "S": 25, "K": 15, "C": 10, "V": 15},
    "decisionGapFullAt": 20, "surpriseRegret": 20, "baselineSurpriseWeight": 15,
    "fairness": {"minimumBestShare": 0.90, "minimumWorldMeanGap": 2},
    "stakes": {"outcomeFlipWeight": 10, "swingStart": 20, "swingRange": 30},
    "clarityByLegalCount": {2: 0.7, 3: 1.0, 4: 0.6},
    "difficultyBand": {"min": 25, "max": 75, "penalty": 10},
    "minimumScore": 60, "interestFloor": {"minimumS": 10, "minimumD": 20},
    "variety": {"lastN": 10, "mechanismLookback": 2, "staleCellLookback": 3},
    "baselinePolicyId": "observer_safe_public_two_trick_control_v6",
    "maximumFinalMargin": 40, "maximumPlayIndex": 27, "maximumAuctionBeats": 10,
    "durationSeconds": 44.0, "durationToleranceSeconds": 0.25,
}
_POLICY_SHA = canonical_sha256(_POLICY)
_WEIGHTS_SHA = canonical_sha256(_POLICY["weights"])
_REPEAT_KEYS = ("scenarioFamily", "visibleOpenerSha256", "captionSha256")


def _variety(row, posted_history, reasons):
    """Use confirmed posts for this platform in newest-first order."""
    keys = (*_REPEAT_KEYS, "mechanism", "contract", "trickBand")
    if not isinstance(posted_history, (list, tuple)) or any(
        entry.get(key) in (None, "") for entry in (row, *posted_history) for key in keys
    ):
        reasons.append("V_history_unavailable")
        return 1.0, None  # Provisional calibration score.
    for key in _REPEAT_KEYS:
        if any(post[key] == row[key] for post in posted_history):
            reasons.append(f"V_repeat_{key}")
    recent = _POLICY["variety"]
    mechanism = row["mechanism"]
    count = sum(post["mechanism"] == mechanism for post in posted_history[:recent["lastN"]])
    mechanisms = [post["mechanism"] for post in posted_history[:recent["mechanismLookback"]]]
    same_latest = bool(mechanisms) and mechanism == mechanisms[0]
    if same_latest:
        reasons.append("V_same_as_latest_mechanism")
    cell = (row["contract"], row["trickBand"])
    stale = recent["staleCellLookback"]
    if len(posted_history) >= stale and all(
        (post["contract"], post["trickBand"]) == cell for post in posted_history[:stale]
    ):
        return 0.0, count
    if mechanism in mechanisms[1:]:
        return 0.5, count
    return float(not same_latest and "V_repeat_scenarioFamily" not in reasons), count


def engine_interest_v1(row, policy=None, *, posted_history=None):
    """Score engine measurements and require verified evidence before posting.

    The caller binds replay, exact actor proof, and publication QA to this candidate.
    Publication QA supplies its §2 gate failures, or an empty list after validation.
    For admission, the caller supplies fresh confirmed history for the target platform.
    """
    if policy is not None and canonical_sha256(policy) != _POLICY_SHA:
        raise ValueError("ENGINE-INTEREST v1 requires the frozen policy")
    reasons = []
    legal, wins = row["legal"], row["roundWin"]
    action, gap, strict, best = row["action"], row["worldMeanGap"], row["strictShare"], row["bestShare"]
    cards = {move.removesuffix("+ekka") for move in legal}
    clarity = _POLICY["clarityByLegalCount"].get(len(cards), 0)
    difficulty = row["interestingnessV4"]["difficulty_prior"]
    baseline_regret = row["baselineRegret"] if row["baselineRegret"] is not None else 0
    measurements = (gap, strict, best, difficulty, row["ddGap"], row["ddSwing"], baseline_regret, row["naiveRegret"])
    if (any(type(value) not in (int, float) or not isfinite(value) for value in measurements)
            or not (0 <= strict <= best <= 1 and 0 <= difficulty <= 100)):
        raise ValueError("invalid ENGINE-INTEREST v1 measurements")
    win_cards = {move.removesuffix("+ekka") for move, value in wins.items() if value == 1}
    flip = wins.get(action) == 1 and -1 in wins.values()
    fairness = _POLICY["fairness"]
    replay_ok = row.get("legalReplayVerified") is True
    if not replay_ok:
        reasons.append("G0_legal_replay_unverified")
    if not clarity or row.get("hardDeductions", 0) < 1:
        reasons.append("G1_clarity")
    if (set(wins) != set(legal) or any(type(value) is not int or abs(value) != 1 for value in wins.values())
            or wins.get(action) != 1 or len(win_cards) != 1
            or not (flip or row["ddGap"] >= _POLICY["decisionGapFullAt"])):
        reasons.append("G2_no_decision")
    viewer_ok = (row.get("worldCoverageComplete") is True and best >= fairness["minimumBestShare"]
                 and gap >= fairness["minimumWorldMeanGap"])
    if not viewer_ok:
        reasons.append("G3_viewer_unfair")
    proof_ok = row.get("exactActorProofVerified") is True
    if not proof_ok:
        reasons.append("G3_exact_actor_proof_unverified")
    margin, beats, duration = (row.get(key) for key in ("finalMargin", "auctionSetupBeats", "durationSeconds"))
    margin_ok = type(margin) in (int, float) and 0 <= margin <= _POLICY["maximumFinalMargin"]
    if not margin_ok:
        reasons.append("G4_final_margin")
    timing_ok = (type(row["playIndex"]) is int and 0 <= row["playIndex"] <= _POLICY["maximumPlayIndex"]
                 and type(beats) is int and 0 <= beats <= _POLICY["maximumAuctionBeats"]
                 and type(duration) in (int, float)
                 and abs(duration - _POLICY["durationSeconds"]) <= _POLICY["durationToleranceSeconds"])
    if not timing_ok:
        reasons.append("G6_timing")
    qa = row.get("publicationGateReasons")
    if not isinstance(qa, (list, tuple)) or any(not isinstance(reason, str) or not reason for reason in qa):
        reasons.append("publication_qa_unverified")
    else:
        reasons.extend(qa)
    variety, mechanism_count = _variety(row, posted_history, reasons)

    w, stakes, band = (_POLICY[key] for key in ("weights", "stakes", "difficultyBand"))
    regret, baseline_weight = (_POLICY[key] for key in ("surpriseRegret", "baselineSurpriseWeight"))
    parts = {
        "D": w["D"] * min(1, max(0, gap) / _POLICY["decisionGapFullAt"]),
        "R": w["R"] * strict,
        "S": baseline_weight * (baseline_regret >= regret)
        + (w["S"] - baseline_weight) * (row["naiveRegret"] >= regret),
        "K": stakes["outcomeFlipWeight"] * flip + (w["K"] - stakes["outcomeFlipWeight"])
        * min(1, max(0, row["ddSwing"] - stakes["swingStart"]) / stakes["swingRange"]),
        "C": w["C"] * clarity, "B": 0 if band["min"] <= difficulty <= band["max"] else -band["penalty"],
        "V": w["V"] * variety,
    }
    total = fsum(parts.values())
    eligible = not reasons
    if total < _POLICY["minimumScore"]:
        reasons.append("below_minimum_score")
    floor = _POLICY["interestFloor"]
    if parts["S"] < floor["minimumS"] and parts["D"] < floor["minimumD"]:
        reasons.append("below_interest_floor")
    return {"scoreVersion": _POLICY["schemaVersion"], "policySha256": _POLICY_SHA, "weightsSha256": _WEIGHTS_SHA,
            "score": round(total, 1), "parts": parts, "gateReasons": reasons, "eligible": eligible,
            "gateStatus": {"T": timing_ok, "F": replay_ok and viewer_ok and proof_ok and margin_ok},
            "postAllowed": not reasons, "difficultyPrior": difficulty, "mechanismFrequency": mechanism_count,
            "exactStateId": row["exactStateId"], "action": action}


def engine_interest_rank_key_v1(card):
    """Sort ascending to rank first by score, then by the versioned tie-breaks."""
    return (-card["score"], -card["parts"]["S"], -card["parts"]["D"], abs(card["difficultyPrior"] - 50),
            card["mechanismFrequency"] or 0, card["exactStateId"])
