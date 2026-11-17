import copy

import pytest

from baloot_platform.research.result_authority import _native_observed_bounds_semantics
from baloot_platform.research.scenario_contracts import canonical_sha256


def hashed(value):
    return {**value, "sha256": canonical_sha256(value)}


@pytest.mark.parametrize("kind", ["native_counterexample_prefix_bound_v1", "native_counterexample_prefix_threshold_v1",
    "native_counterexample_prefix_minimax_v1",
    "native_bid_policy_bounds_v1", "native_bid_branch_preparation_v1", "native_incumbent_threshold_floor_v1"])
def test_observed_proof_comparison_preserves_content_and_reference_integrity(kind):
    def document(seconds, bound=7, reference=None):
        clock = ({"maximumSeconds": seconds} if kind == "native_bid_policy_bounds_v1"
                 else {"limits": {"seconds": seconds, "conflicts": 100}})
        prefix = hashed({"kind": kind, **clock, "upperBound": bound})
        refinement = hashed({"kind": "native_symbolic_round_refinement_v1", "queries": [prefix],
            "prefixBoundQueryHash": reference or prefix["sha256"]})
        return hashed({"kind": "native_symbolic_round_policy_bounds_v2", "limits": {"seconds": 10},
            "finalBranchProofs": [{"refinementHash": refinement["sha256"]}],
            "refinements": {refinement["sha256"]: refinement}})

    normalize = lambda value: _native_observed_bounds_semantics(value, maximum_seconds=10)
    original = document(8)
    retained = copy.deepcopy(original)
    assert normalize(original) == normalize(document(6))
    assert original == retained  # Comparison never edits immutable evidence.
    assert normalize(original) != normalize(document(6, bound=8))
    assert normalize(original) != normalize(document(6, reference="0" * 64))
    for seconds in (True, 0, -1, 11):
        with pytest.raises(ValueError, match="allowance"):
            normalize(document(seconds))
    corrupted = document(8)
    refinement = next(iter(corrupted["refinements"].values()))
    forged = copy.deepcopy(refinement["queries"][0])
    forged["upperBound"] += 1  # Duplicate digest must not hide different content.
    refinement["queries"].append(forged)
    with pytest.raises(ValueError, match="checksum"):
        normalize(corrupted)
    with pytest.raises(ValueError, match="checksum"):
        normalize([refinement["queries"][0], forged])


def test_observed_proof_comparison_checks_caller_allowance():
    for seconds in (True, 0, -1, float("inf"), None):
        with pytest.raises(ValueError, match="finite positive"):
            _native_observed_bounds_semantics({}, maximum_seconds=seconds)
