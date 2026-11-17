from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from baloot_platform.config import build_bundle
from baloot_platform.research.service import ResearchFoundryServiceV1
from scripts.run_combined_strategy_pilot import _actor_safe_mismatch_partition


def _recipe(*, contract: str = "sun") -> SimpleNamespace:
    return SimpleNamespace(
        situation={
            "buyer": None,
            "contracts": [contract],
            "trumpSuit": None,
            "purchaseRound": None,
            "projects": [],
            "escalations": [],
            "dealer": "east",
            "matchScore": [0, 0],
        }
    )


def _simulation(*, contract: str = "ashkal", rules_hash: str = "rules") -> Any:
    return SimpleNamespace(
        publishable_laffa=True,
        final_trace=SimpleNamespace(
            trusted_final_state={
                "buyer": "north",
                "contract": contract,
                "trumpSuit": None,
                "biddingHistory": [],
                "projects": [],
                "escalation": "normal",
                "rulesHash": rules_hash,
            }
        ),
    )


def test_actor_safe_exact_contract_miss_is_not_semantic_integrity() -> None:
    service = object.__new__(ResearchFoundryServiceV1)
    service.domain = SimpleNamespace(rules_hash="rules")
    recipe = _recipe(contract="sun")
    simulation = _simulation(contract="ashkal")

    assert service._policy_target_mismatches(recipe, simulation) == (
        "contract_target_miss",
    )
    assert service._semantic_mismatches(
        recipe,
        simulation,
        include_policy_target_mismatches=False,
    ) == []
    assert service._semantic_mismatches(recipe, simulation) == [
        "compiler_parity_defect"
    ]


def test_actor_safe_partition_keeps_compiled_replay_defects_fatal(root) -> None:
    calls: list[bool] = []
    compiled_calls: list[dict[str, Any]] = []

    class _Research:
        @staticmethod
        def _policy_target_mismatches(*_args: Any) -> tuple[str, ...]:
            return ("contract_target_miss",)

        @staticmethod
        def _semantic_mismatches(
            *_args: Any,
            include_policy_target_mismatches: bool,
            **_kwargs: Any,
        ) -> list[str]:
            calls.append(include_policy_target_mismatches)
            return []

        @staticmethod
        def _compiled_play_mismatches(
            *_args: Any,
            **kwargs: Any,
        ) -> tuple[str, ...]:
            compiled_calls.append(kwargs)
            return ("compiler_parity_defect",)

    rules = build_bundle(root).rules
    candidate = SimpleNamespace(
        deck_order=tuple(range(32)),
        candidate_hash="c" * 64,
    )
    project_plan = object()
    targets, defects = _actor_safe_mismatch_partition(
        research=_Research(),
        recipe=_recipe(),
        simulation=_simulation(),
        project_plan=project_plan,
        play_plan=object(),
        candidate=candidate,
        rules=rules,
        context=object(),
    )

    assert calls == [False]
    assert len(compiled_calls) == 1
    assert compiled_calls[0]["project_plan"] is project_plan
    assert compiled_calls[0]["expected_deck_order"] == tuple(range(32))
    assert compiled_calls[0]["expected_candidate_hash"] == "c" * 64
    assert compiled_calls[0]["expected_dealer"] == "east"
    assert compiled_calls[0]["expected_round_number"] == 1
    assert compiled_calls[0]["starting_match_score"] == (0, 0)
    assert targets == ["contract_target_miss"]
    assert defects == ["compiler_parity_defect"]
