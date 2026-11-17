from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

from baloot_platform.audit.models import canonical_action
from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.history.provenance_v4 import canonical_hash, game_serial


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = root / "release" / "v6-challenge-runtime"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    bundle = build_bundle(root)
    request = ScenarioRequest.governed(
        seed=27,
        dealer=Seat.EAST,
        contract=Contract.HOKUM,
        buyer=Seat.SOUTH,
        trump_suit=Suit.HEARTS,
        require_project=True,
        pause_trick_min=5,
        pause_trick_max=5,
        requested_samples=48,
        exact_allocation_limit=12_000,
        exact_search_max_remaining=10,
        maximum_sampling_attempts=60_000,
        max_seed_attempts=1,
        max_candidates_per_seed=4,
        minimum_hidden_deals=20,
        minimum_expected_margin_gap=3.0,
        minimum_win_probability_gap=0.08,
        require_exact_analysis=False,
        target_debate_score=55,
    )
    episode = generate_episode(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        request=request,
        copy=bundle.video["localizations"],
    )
    if len(episode.analysis.actions) < 2:
        raise RuntimeError("challenge analysis did not produce at least two alternatives")
    best, alternative = episode.analysis.actions[0], episode.analysis.actions[1]
    game_hash = canonical_hash(episode.final_state.to_dict(reveal_hands=True))
    decision_id = "DEC-" + hashlib.sha256(f"{game_hash}:{episode.pause_before_play_index}".encode()).hexdigest()[:20].upper()
    audit_identity = canonical_hash({
        "gameHash": game_hash,
        "decisionId": decision_id,
        "alternative": alternative.card.code,
        "best": best.card.code,
        "bundleHash": bundle.sha256,
    })
    audit_summary = {
        "auditId": "DAU-" + audit_identity[:20].upper(),
        "qualityLabel": "counterfactual",
        "qualityScore": 100.0,
        "actualAction": canonical_action(alternative.card),
        "bestAction": canonical_action(best.card),
        "expectedRegret": best.expected_match_margin - alternative.expected_match_margin,
        "winProbabilityDrop": best.win_probability - alternative.win_probability,
        "catastrophicCrossing": (
            best.expected_match_margin > 0 > alternative.expected_match_margin
            or best.win_probability >= 0.5 > alternative.win_probability
        ),
        "actualIntent": {
            "status": "counterfactual_not_observed",
            "statement": "This is a comparison branch, not a claim about a real player's intention.",
        },
    }
    episode = replace(
        episode,
        schema_version=max(7, episode.schema_version),
        game_serial=game_serial(game_hash),
        game_record_hash=game_hash,
        challenge_id="CHL-REPLAY-AUDIT-V6",
        analysis_id="ANL-" + canonical_hash(episode.analysis.to_dict())[:20].upper(),
        audit_id=audit_summary["auditId"],
        decision_id=decision_id,
        decision_index=(("question", decision_id), ("reveal", decision_id), ("continuation", decision_id)),
        audit_summary=audit_summary,
        playback_mode="challenge_branch",
        title_ar=str(bundle.video["localizations"]["video.decision.challenge_title"]),
        question_ar=str(bundle.video["localizations"]["video.decision.challenge_question"]),
    )
    (runtime / "episode.pkl").write_bytes(pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL))
    summary = {
        "schemaVersion": 1,
        "gameHash": game_hash,
        "gameSerial": episode.game_serial,
        "decisionId": decision_id,
        "auditId": episode.audit_id,
        "analysisId": episode.analysis_id,
        "bestCard": best.card.to_dict(),
        "alternativeCard": alternative.card.to_dict(),
        "expectedRegret": audit_summary["expectedRegret"],
        "winProbabilityDrop": audit_summary["winProbabilityDrop"],
        "catastrophicCrossing": audit_summary["catastrophicCrossing"],
        "difficulty": episode.analysis.difficulty,
        "debateScore": episode.analysis.debate_score,
        "hiddenDeals": episode.analysis.hidden_deals,
    }
    (runtime / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
