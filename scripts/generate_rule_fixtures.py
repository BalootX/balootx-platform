#!/usr/bin/env python3
"""Generate executable Baloot rule fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.core.rules import resolved_disputed_options  # noqa: E402


RULES_PATH = ROOT / "configs/rules/saudi_common_v1.yaml"
OUTPUT = ROOT / "examples/rule-fixtures.json"
SUITS = ("spades", "hearts", "diamonds", "clubs")
SUIT_CODES = ("S", "H", "D", "C")
RANKS = ("seven", "eight", "nine", "ten", "jack", "queen", "king", "ace")
RANK_CODES = ("7", "8", "9", "10", "J", "Q", "K", "A")


def _profile_hash(profile: dict) -> str:
    encoded = json.dumps(
        profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base(
    *,
    case_id: str,
    clause: str,
    profile_version: str,
    initial_state: dict,
    action: dict,
    resulting_state: dict,
    score: dict,
    rationale: str,
) -> dict:
    return {
        "caseId": case_id,
        "clauseId": clause,
        "provenance": {
            "kind": "published_profile_software_conformance",
            "source": "configs/rules/saudi_common_v1.yaml",
            "externalCertification": False,
        },
        "profileRevision": profile_version,
        "initialState": initial_state,
        "legalMask": [],
        "action": action,
        "events": [],
        "resultingState": resulting_state,
        "score": score,
        "rationale": rationale,
    }


def _card_cases(profile: dict) -> list[dict]:
    version = str(profile["profile"]["version"])
    tables = profile["card_rules"]
    cases: list[dict] = []
    modes = (
        (
            "sun",
            None,
            "sun_strength",
            "sun_points",
            "card-table.sun",
        ),
        (
            "hokum",
            "different",
            "hokum_non_trump_strength",
            "hokum_non_trump_points",
            "card-table.hokum-non-trump",
        ),
        (
            "hokum",
            "same",
            "hokum_trump_strength",
            "hokum_trump_points",
            "card-table.hokum-trump",
        ),
    )
    for contract, trump_mode, strength_key, points_key, clause in modes:
        for suit_index, (suit, suit_code) in enumerate(
            zip(SUITS, SUIT_CODES)
        ):
            trump = None
            if trump_mode == "same":
                trump = suit
            elif trump_mode == "different":
                trump = SUITS[(suit_index + 1) % len(SUITS)]
            for rank_index, (rank, rank_code) in enumerate(
                zip(RANKS, RANK_CODES)
            ):
                card = f"{suit_code}{rank_code}"
                cases.append(
                    _base(
                        case_id=(
                            f"CARD-{contract.upper()}-"
                            f"{trump_mode or 'NO-TRUMP'}-{card}"
                        ),
                        clause=clause,
                        profile_version=version,
                        initial_state={
                            "contract": contract,
                            "trumpSuit": trump,
                        },
                        action={"kind": "evaluate_card", "card": card},
                        resulting_state={
                            "strength": int(
                                tables[strength_key][rank_index]
                            )
                        },
                        score={
                            "rawCardPoints": int(
                                tables[points_key][rank_index]
                            )
                        },
                        rationale=(
                            f"Locks {strength_key} and {points_key} for "
                            f"{rank} of {suit}."
                        ),
                    )
                )
    return cases


def _sun_points(raw: int) -> int:
    units = raw % 10
    rounded = raw if units == 5 else round(raw / 10) * 10
    return int(rounded // 5)


def _hokum_points(raw: int) -> int:
    tens, units = divmod(raw, 10)
    return tens + (1 if units >= 6 else 0)


def _conversion_cases(profile: dict) -> list[dict]:
    version = str(profile["profile"]["version"])
    cases: list[dict] = []
    for contract, maximum, converter, clause in (
        ("sun", 130, _sun_points, "score.conversion.sun"),
        ("hokum", 162, _hokum_points, "score.conversion.hokum"),
    ):
        for raw in range(maximum + 1):
            cases.append(
                _base(
                    case_id=f"SCORE-{contract.upper()}-{raw:03d}",
                    clause=clause,
                    profile_version=version,
                    initial_state={"contract": contract, "rawPoints": raw},
                    action={"kind": "convert_raw_score"},
                    resulting_state={"rawPoints": raw},
                    score={"gamePoints": converter(raw)},
                    rationale=(
                        f"Locks the published {contract} rounding boundary "
                        f"for raw score {raw}."
                    ),
                )
            )
    return cases


def _disputed_option_cases(profile: dict) -> list[dict]:
    version = str(profile["profile"]["version"])
    cases: list[dict] = []
    for option, contract in sorted(resolved_disputed_options(profile).items()):
        values = [contract["selected"], *contract["alternatives"]]
        for index, value in enumerate(values):
            case = _base(
                case_id=f"VARIANT-{option.upper().replace('_', '-')}-{index}",
                clause="variants.disputed-options",
                profile_version=version,
                initial_state={
                    "option": option,
                    "publishedSelection": contract["selected"],
                },
                action={"kind": "resolve_disputed_option", "value": value},
                resulting_state={"resolvedValue": value},
                score={},
                rationale=(
                    f"Locks the explicit {option} branch without treating it as "
                    "universal Baloot law."
                ),
            )
            case["legalMask"] = values
            cases.append(case)
    return cases


def _governed_authority_cases(
    profile: dict, adjudication: dict | None
) -> list[dict]:
    """Bind the corrected public clauses and referee boundary to executable IDs."""

    profile_id = str(profile["profile"]["id"])
    if profile_id not in {
        "saudi_federation_2023_v2",
        "saudi_federation_2023_v8",
        "saudi_federation_2023_sweep",
        "saudi_project_optional_unratified_v1",
    }:
        return []
    version = str(profile["profile"]["version"])

    def case(
        case_id: str,
        clause: str,
        action: str,
        initial_state: dict,
        resulting_state: dict,
        rationale: str,
    ) -> dict:
        return _base(
            case_id=case_id,
            clause=clause,
            profile_version=version,
            initial_state=initial_state,
            action={"kind": action},
            resulting_state=resulting_state,
            score={},
            rationale=rationale,
        )

    bidding = profile["bidding"]
    escalation = profile["escalation"]
    projects = profile["projects"]
    scoring = profile["scoring"]
    cases = [
        case(
            "AUTH-AUCTION-EXPOSED-ACE",
            "section_4.clause_4_1_and_section_3.clause_3_5",
            "verify_exposed_ace_recall",
            {
                "dealer": "east",
                "purchaseSuit": "hearts",
                "purchaseRank": "ace",
                "auctionPrefix": ["pass", "pass", "hokum", "pass"],
            },
            {
                "legalActions": (
                    ["pass"]
                    if bidding["first_round"][
                        "exposed_ace_hokum_sun_overcall"
                    ]
                    == "dealer_right_only"
                    else ["pass", "sun"]
                )
            },
            "Exercises the rank-aware exposed-Ace second-recall restriction.",
        ),
        case(
            "AUTH-AUCTION-ASHKAL-ROLES",
            "general_points.item_2",
            "verify_ashkal_roles",
            {"actor": "south", "purchaseSuit": "hearts"},
            {
                "buyer": "south"
                if bidding["ashkal"]["contract_owner"] == "bidder"
                else "north",
                "purchaseRecipient": "north",
            },
            "Exercises the configured buyer/card-recipient distinction for Ashkal.",
        ),
        case(
            "AUTH-ESCALATION-SUN-100-INCLUSIVE",
            "section_7.clause_7_2",
            "verify_sun_double_boundary",
            {"declarer": "south", "matchScore": [101, 100]},
            {"level": "double" if escalation["sun_lower_score_inclusive"] else "rejected"},
            "Exercises the public Sun Double boundary where the lower side equals 100.",
        ),
        case(
            "AUTH-PROJECT-HUNDRED-PRIORITY",
            "section_5.clause_5_2",
            "verify_hundred_priority",
            {},
            {
                "winnerKind": (
                    "hundred_sequence"
                    if projects["comparison"]["category_priorities"][
                        "hundred_sequence"
                    ]
                    > projects["comparison"]["category_priorities"][
                        "hundred_four_kind"
                    ]
                    else "hundred_four_kind"
                )
            },
            "Exercises sequential Hundred versus four-of-a-kind Hundred comparison.",
        ),
        case(
            "AUTH-PROJECT-BALOOT-IN-HUNDRED",
            "section_5.clauses_5_6_and_5_8",
            "verify_baloot_in_hundred",
            {"owner": "north", "trumpSuit": "hearts"},
            {
                "counts": projects["baloot"]["hundred_inclusion"] == "automatic",
                "verbalDeclarationAuthorized": False,
            },
            "Separates automatic Baloot scoring from the prohibited verbal declaration.",
        ),
        case(
            "AUTH-SCORING-DOUBLED-KABOUT",
            "section_7.clause_7_7",
            "verify_doubled_kabout",
            {},
            {
                "sun": scoring["kabout"]["base_game_points"]["sun"],
                "hokum": scoring["kabout"]["base_game_points"]["hokum"],
            },
            "Exercises the non-remultiplied doubled-Kabout bases.",
        ),
        case(
            "AUTH-PROJECT-DECLARATION-MODE",
            "configured_profile.projects.declaration_mode",
            "verify_project_declaration_mode",
            {},
            {"mode": projects["declaration_mode"]},
            "Binds the explicit mandatory or optional project-declaration variant.",
        ),
    ]
    if scoring["kabout"].get("winner") == "trick_sweeper":
        cases.append(case(
            "AUTH-SCORING-KABOUT-SWEEPER",
            "configured_profile.scoring.kabout.winner",
            "verify_kabout_sweeper",
            {"contract": "sun", "rawPoints": [0, 130],
             "projectPoints": [44, 0], "trickWinnerTeams": [1] * 8},
            {"roundWinner": "west_east", "awardedPoints": [0, 44]},
            "Configured, unratified correction: losing-side projects cannot reverse a sweep.",
        ))
    if adjudication is not None:
        for event_id, event in sorted(adjudication["events"].items()):
            cases.append(
                case(
                    f"AUTH-ADJUDICATION-{event_id.upper().replace('_', '-')}",
                    str(event["sourceClause"]),
                    "verify_adjudication_boundary",
                    {"eventId": event_id},
                    {
                        "ordinaryTrace": event["ordinaryTrace"],
                        "hardPredicate": event["hardPredicate"],
                        "requiresTableReferee": event["requiresTableReferee"],
                        "remedies": event["availableRemedies"],
                    },
                    "Binds the source-clause remedy without turning referee evidence into a legal trace action.",
                )
            )
    return cases


def build(
    rules_path: Path = RULES_PATH,
    adjudication_path: Path | None = None,
) -> dict:
    profile = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    adjudication = (
        yaml.safe_load(adjudication_path.read_text(encoding="utf-8"))
        if adjudication_path is not None
        else None
    )
    cases = (
        _card_cases(profile)
        + _conversion_cases(profile)
        + _disputed_option_cases(profile)
        + _governed_authority_cases(profile, adjudication)
    )
    source = rules_path.resolve().relative_to(ROOT).as_posix()
    for case in cases:
        case["provenance"]["source"] = source
    return {
        "schemaVersion": 1,
        "rulesProfile": str(profile["profile"]["id"]),
        "rulesVersion": str(profile["profile"]["version"]),
        "rulesSha256": _profile_hash(profile),
        "certificationStatus": "candidate_external_review_required",
        "caseCount": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rules-path", type=Path, default=RULES_PATH)
    parser.add_argument("--adjudication-path", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rendered = json.dumps(
        build(args.rules_path, args.adjudication_path),
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("executable rules fixture corpus is stale")
        return 0
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
