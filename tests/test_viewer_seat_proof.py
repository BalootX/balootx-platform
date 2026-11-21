import hashlib
import os

import pytest

from scripts.viewer_seat_proof import prove_fixture, receipt_bytes


@pytest.fixture
def proof_input(root):
    raw = (root / "examples/research-scenario-v4-ace-sacrifice-seven.json").read_bytes()
    return raw, {
        "fixture_sha": hashlib.sha256(raw).hexdigest(), "source_sha": "a" * 40,
        "maximum_allocations": 4096, "maximum_nodes": 2000, "cp_work": 1, "backend": "ortools",
    }


def test_exact_viewer_seat_receipt_matches_the_two_hand_checkable_outcomes(proof_input):
    raw, options = proof_input
    receipt = prove_fixture(raw, **options)
    assert receipt["status"] == "exact_complete" and receipt["coverageComplete"]
    assert receipt["dealsSolved"] == receipt["dealsTotal"] == 12
    # DA scores 44 in all twelve worlds. H10 scores 44 in six, and -2 in six.
    assert (receipt["R"], receipt["D"], receipt["uniqueBest"]) == (1, 23, True)
    assert receipt["scorerInputs"] == {"worldsComplete": True, "bestShare": 1, "strictShare": 0.5, "worldMeanGap": 23}
    assert [(move["action"], move["bestDeals"], move["strictBestDeals"], move["meanGapNumerator"])
            for move in receipt["moves"]] == [("DA", 12, 6, 276), ("H10", 6, 0, -276)]
    assert receipt_bytes(receipt) == receipt_bytes(prove_fixture(raw, **options))


@pytest.mark.parametrize("limit,reason", [
    ({"maximum_allocations": 11}, "maximum_allocations"),
    ({"maximum_nodes": 100}, "maximum_nodes"),
])
def test_incomplete_viewer_seat_proof_withholds_scores(proof_input, limit, reason):
    raw, options = proof_input
    receipt = prove_fixture(raw, **{**options, **limit})
    assert receipt["status"] == "INCOMPLETE" and receipt["reason"] == reason
    assert receipt["coverageComplete"] is False
    assert receipt["R"] is None and receipt["D"] is None and not receipt["moves"]
    assert "dealRootValuesSha256" not in receipt and "scorerInputs" not in receipt
    if reason == "maximum_nodes":
        assert 0 < receipt["dealsSolved"] < receipt["dealsTotal"]
    assert receipt_bytes(receipt) == receipt_bytes(prove_fixture(raw, **{**options, **limit}))


def test_fixture_hash_mismatch_does_not_enter_the_solver(proof_input):
    raw, options = proof_input
    with pytest.raises(ValueError, match="scenario SHA-256 mismatch"):
        prove_fixture(raw + b"\n", **options)


@pytest.mark.skipif(not os.environ.get("BALOOT_NATIVE_LIBRARY"), reason="native test library not configured")
def test_node_only_native_receipt_matches_cp_sat_and_is_repeatable(proof_input):
    raw, options = proof_input
    reference = prove_fixture(raw, **options)
    native_options = {**options, "backend": "native"}
    actual = prove_fixture(raw, **native_options)
    assert actual["status"] == "exact_complete"
    for key in ("R", "D", "moves", "dealRootValuesSha256"):
        assert actual[key] == reference[key]
    assert receipt_bytes(actual) == receipt_bytes(prove_fixture(raw, **native_options))
    stopped = prove_fixture(raw, **{**native_options, "maximum_nodes": 1})
    assert (stopped["status"], stopped["reason"], stopped["dealsSolved"]) == ("INCOMPLETE", "maximum_nodes", 0)
