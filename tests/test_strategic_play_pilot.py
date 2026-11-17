from __future__ import annotations

from scripts.run_strategic_play_pilot import _decision, _metrics


def _lane(
    *,
    qualified: int,
    cpu_seconds: float,
    quality: float,
    exact_decisions: int,
) -> dict[str, object]:
    return {
        "status": "completed",
        "semanticDefectCount": 0,
        "strategicExactDecisions": exact_decisions,
        "meanSurroundingQuality": quality,
        "qualifiedPerCpuHour": qualified * 3600.0 / cpu_seconds,
        "cpuSecondsPerDistinctQualified": (
            cpu_seconds / qualified if qualified else None
        ),
        "distinctQualifiedCount": qualified,
        "cpuSeconds": cpu_seconds,
    }


def test_zero_yield_cannot_pass_the_cost_or_promotion_gate() -> None:
    baseline = _lane(
        qualified=0,
        cpu_seconds=5.0,
        quality=0.0,
        exact_decisions=0,
    )
    strategic = _lane(
        qualified=0,
        cpu_seconds=2.0,
        quality=10.0,
        exact_decisions=4,
    )

    evidence = _decision(baseline, strategic)

    assert evidence["costReduction"] is None
    assert evidence["costGatePassed"] is False
    assert evidence["yieldGatePassed"] is False
    assert evidence["promoteStrategicMode"] is False
    assert evidence["recommendation"] == "keep_experimental"


def test_distinct_qualified_yield_can_pass_the_promotion_gate() -> None:
    baseline = _lane(
        qualified=0,
        cpu_seconds=10.0,
        quality=20.0,
        exact_decisions=0,
    )
    strategic = _lane(
        qualified=1,
        cpu_seconds=10.0,
        quality=25.0,
        exact_decisions=2,
    )

    evidence = _decision(baseline, strategic)

    assert evidence["yieldGatePassed"] is True
    assert evidence["promoteStrategicMode"] is True
    assert evidence["recommendation"] == "promote"


def test_pilot_metrics_retain_zero_yield_qualification_failures() -> None:
    class _Research:
        @staticmethod
        def list_results(**_kwargs):
            return {
                "items": [
                    {
                        "resultCode": "LAF-ONE",
                        "resultHash": "a" * 64,
                        "gameHash": "b" * 64,
                        "score": 42.0,
                        "disposition": "rejected",
                        "truthLabel": "rejected",
                        "qualification": {
                            "minimumSurroundingQuality": 0,
                            "failures": [
                                "surrounding_agent_regret",
                                "surrounding_quality_below_gate",
                            ],
                        },
                    },
                    {
                        "resultCode": "LAF-TWO",
                        "resultHash": "c" * 64,
                        "gameHash": "d" * 64,
                        "score": 45.0,
                        "disposition": "rejected",
                        "truthLabel": "rejected",
                        "qualification": {
                            "minimumSurroundingQuality": 30,
                            "failures": ["surrounding_agent_regret"],
                        },
                    },
                ]
            }

    evidence = _metrics(
        _Research(),
        {
            "runCode": "RUN-TEST",
            "status": "completed",
            "attemptedSeeds": 2,
            "elapsedSeconds": 1,
            "cpuSeconds": 1,
        },
    )

    assert evidence["distinctQualifiedCount"] == 0
    assert evidence["qualificationFailureCounts"] == {
        "surrounding_agent_regret": 2,
        "surrounding_quality_below_gate": 1,
    }
    assert evidence["resultSummaries"][0]["qualificationFailures"] == [
        "surrounding_agent_regret",
        "surrounding_quality_below_gate",
    ]
