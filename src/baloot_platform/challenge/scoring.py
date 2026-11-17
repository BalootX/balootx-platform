"""Profile-driven challenge scoring plugin.

Evidence contracts own facts. Published profiles own weights and gates. This
plugin is the single runtime boundary that applies one to the other; it never
changes engine truth or turns failed evidence into passing evidence.
"""
from __future__ import annotations

from typing import Any

from baloot_platform.plugins.base import PluginDescriptor


class ProfileChallengeScorerV1:
    descriptor = PluginDescriptor(
        id="profile_challenge_scorer_v1",
        kind="challenge_scorer",
        version="1.0.0",
        description="Apply a governed score profile to immutable challenge evidence",
        capabilities=(
            "decision_content_selection",
            "laffa_content_selection",
            "hard_truth_gates",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=30,
        maximum_memory_mb=256,
        contract_id="research_objective",
        contract_version="1",
        safety_guards=(
            "observer_safe",
            "hard_gate_preservation",
            "frozen_provenance",
        ),
    )

    @staticmethod
    def score(evidence: Any, profile: Any, **_: Any) -> Any:
        scorer = getattr(profile, "score", None)
        if not callable(scorer):
            raise TypeError("challenge score profile must provide score(evidence)")
        result = scorer(evidence)
        if not hasattr(result, "score_family") or not hasattr(result, "sha256"):
            raise TypeError("challenge score profile returned an unsupported result")
        return result
