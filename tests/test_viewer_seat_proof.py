import hashlib
import json

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
    assert receipt["schemaVersion"] == 2
    assert receipt["dealsSolved"] == receipt["dealsTotal"] == 12
    assert receipt["allocationsAttempted"] == 12 and receipt["allocationRejectionsByReason"] == {}
    for phrase in ("rule-consistent hidden deals", "Each deal counts equally",
                   "bids are not used as constraints", "R is not a probability"):
        assert phrase in receipt["rMeaning"]
    # DA scores 44 in all twelve worlds. H10 scores 44 in six, and -2 in six.
    assert (receipt["R"], receipt["D"], receipt["uniqueBest"]) == (1, 23, True)
    assert receipt["scorerInputs"] == {"worldsComplete": True, "bestShare": 1, "strictShare": 0.5, "worldMeanGap": 23}
    assert [(move["action"], move["bestDeals"], move["strictBestDeals"], move["meanGapNumerator"])
            for move in receipt["moves"]] == [("DA", 12, 6, 276), ("H10", 6, 0, -276)]
    assert receipt_bytes(receipt) == receipt_bytes(prove_fixture(raw, **options))


def test_viewer_seat_receipt_counts_rejected_public_project_histories(root, proof_input):
    _, options = proof_input
    raw = (root / "examples/research-scenario-v4-single-winner-20261116-05.json").read_bytes()
    receipt = prove_fixture(raw, **{**options, "fixture_sha": hashlib.sha256(raw).hexdigest()})
    assert receipt["status"] == "exact_complete"
    assert (receipt["R"], receipt["D"]) == (12 / 21, -56 / 21)
    assert (receipt["allocationsAttempted"], receipt["dealsTotal"]) == (60, 21)
    assert receipt["allocationRejectionsByReason"] == {"public_project_history_mismatch": 39}


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
        assert receipt["allocationRejectionsByReason"] == {}
    else:
        assert "allocationRejectionsByReason" not in receipt
    assert receipt_bytes(receipt) == receipt_bytes(prove_fixture(raw, **{**options, **limit}))


def test_fixture_hash_mismatch_does_not_enter_the_solver(proof_input):
    raw, options = proof_input
    with pytest.raises(ValueError, match="scenario SHA-256 mismatch"):
        prove_fixture(raw + b"\n", **options)


@pytest.mark.parametrize("acting_seat", ["east", None])
def test_fixture_viewer_seat_must_match_replay(proof_input, acting_seat):
    raw, options = proof_input
    fixture = json.loads(raw)
    fixture["rulesMeaning"]["actingSeat"] = acting_seat
    raw = receipt_bytes(fixture)
    with pytest.raises(ValueError, match=r"rulesMeaning\.actingSeat"):
        prove_fixture(raw, **{**options, "fixture_sha": hashlib.sha256(raw).hexdigest()})


def test_viewer_seat_proof_rejects_single_legal_card(proof_input):
    raw, options = proof_input
    fixture = json.loads(raw)
    fixture["exactTracePrefix"]["playActions"].extend([
        {"actor": "west", "card": "DA"}, {"actor": "south", "card": "H8"},
    ])
    fixture["rulesMeaning"]["actingSeat"] = "east"
    fixture["expected"]["selectedAction"] = "H7"
    raw = receipt_bytes(fixture)
    with pytest.raises(ValueError, match="D is undefined when only one card is legal"):
        prove_fixture(raw, **{**options, "fixture_sha": hashlib.sha256(raw).hexdigest()})


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
