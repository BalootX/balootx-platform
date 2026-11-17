from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from baloot_platform.core.rules import resolved_disputed_options

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Suit
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring import _hokum_game_points, _sun_game_points


def test_v21_fixture_corpus_is_frozen_complete_and_current(root: Path):
    result = subprocess.run(
        [sys.executable, "scripts/generate_rule_fixtures.py", "--check"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    corpus = json.loads(
        (root / "examples/rule-fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    assert corpus["caseCount"] == len(corpus["cases"])
    assert corpus["caseCount"] >= 250
    assert corpus["certificationStatus"] == "candidate_external_review_required"
    required = {
        "caseId", "clauseId", "provenance", "profileRevision",
        "initialState", "legalMask", "action", "events", "resultingState",
        "score", "rationale",
    }
    assert all(set(case) == required for case in corpus["cases"])
    assert len({case["caseId"] for case in corpus["cases"]}) == len(
        corpus["cases"]
    )
    assert all(
        case["provenance"]["externalCertification"] is False
        for case in corpus["cases"]
    )


def test_v21_card_and_score_fixtures_match_authoritative_engine(root: Path):
    corpus = json.loads(
        (root / "examples/rule-fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    rules = RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    assert corpus["rulesSha256"] == rules.sha256
    assert corpus["rulesVersion"] == rules.version
    for case in corpus["cases"]:
        action = case["action"]
        state = case["initialState"]
        if action["kind"] == "evaluate_card":
            card = Card.parse(action["card"])
            contract = Contract.parse(state["contract"])
            trump = (
                Suit.parse(state["trumpSuit"])
                if state["trumpSuit"] is not None
                else None
            )
            assert rules.strength(card, contract, trump) == case[
                "resultingState"
            ]["strength"]
            assert rules.points(card, contract, trump) == case["score"][
                "rawCardPoints"
            ]
        elif action["kind"] == "convert_raw_score":
            converter = (
                _sun_game_points
                if state["contract"] == "sun"
                else _hokum_game_points
            )
            assert converter(int(state["rawPoints"])) == case["score"][
                "gamePoints"
            ]
        elif action["kind"] == "resolve_disputed_option":
            assert case["resultingState"]["resolvedValue"] in case["legalMask"]
        else:
            raise AssertionError(f"unknown fixture action: {action['kind']}")


def test_v21_disputed_rule_branches_are_explicit_fixtures(root: Path):
    corpus = json.loads(
        (root / "examples/rule-fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    rules = yaml.safe_load(
        (root / "configs/rules/saudi_common_v1.yaml").read_text(encoding="utf-8")
    )
    cases = [
        case
        for case in corpus["cases"]
        if case["clauseId"] == "variants.disputed-options"
    ]
    actual = {
        (case["initialState"]["option"], case["resultingState"]["resolvedValue"])
        for case in cases
    }
    expected = {
        (option, value)
        for option, contract in resolved_disputed_options(rules).items()
        for value in [contract["selected"], *contract["alternatives"]]
    }

    assert actual == expected
    assert all(case["legalMask"] for case in cases)


def test_v21_rule_conformance_scale_gates_are_governed(root: Path):
    profile = yaml.safe_load(
        (root / "configs/certification/rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    gates = profile["software_conformance"]
    assert gates["minimum_executable_fixtures"] >= 250
    assert gates["minimum_seeded_replay_rounds"] >= 10000
    assert gates["minimum_python_native_positions"] >= 100000
    assert gates["minimum_invalid_input_fuzz_cases"] >= 10000
    assert gates["require_apply_undo_byte_identity"] is True
    assert gates["require_observer_noninterference"] is True
    assert gates["parity_claim"] == "semantic_equivalence"
    assert gates["external_certification_substitute"] is False
    bundle = yaml.safe_load(
        (root / "configs/bundle.yaml").read_text(encoding="utf-8")
    )
    assert bundle["components"]["certification"] == (
        "certification/rules.yaml"
    )
