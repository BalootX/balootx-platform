from __future__ import annotations

from scripts.run_tactical_precursor_pilot import _pilot_recipe


class _ResearchStub:
    @staticmethod
    def capabilities():
        return {
            "sha256": "1" * 64,
            "bundleHash": "2" * 64,
            "rulesHash": "3" * 64,
            "conventionHash": "4" * 64,
            "research": {
                "defaultAgentRelease": "foundry_balanced@test",
                "analysisProfiles": {
                    "screening": "5" * 64,
                    "deep": "7" * 64,
                },
                "defaultScoreProfileHash": "6" * 64,
            },
        }


def test_precursor_pilot_changes_only_the_governed_tactical_profile():
    baseline = _pilot_recipe(
        _ResearchStub(),
        attempts=2,
        seed_start=202608090001,
        precursor_enabled=False,
    )
    candidate = _pilot_recipe(
        _ResearchStub(),
        attempts=2,
        seed_start=202608090001,
        precursor_enabled=True,
    )
    tactical = candidate["resources"].pop("tacticalConstruction")
    assert baseline == candidate
    assert tactical["enabled"] is True
    assert tactical["minimumLaterRawPoints"] == 10
    assert baseline["tactics"][0]["mode"] == "qualification_gate"
    assert baseline["situation"]["buyer"] == "south"
    assert baseline["qualification"]["minimumTruth"] == "robust_best_known"
    assert baseline["resources"]["analysisMode"] == "deep"
