import pytest

from baloot_platform.research.result_authority import _require_matching_exact_observed_reproof_v1
from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped


def test_readback_exhaustion_is_incomplete_not_a_claim_disagreement():
    with pytest.raises(InformationSetCompilationStopped) as caught:
        _require_matching_exact_observed_reproof_v1(
            {"proof": {"completionStatus": "exact_complete"}, "proofModel": {}},
            {"completionStatus": "proof_incomplete"}, {})
    assert caught.value.reason == "proof_incomplete"
    assert caught.value.partial_receipt["stage"] == "exact_observed_readback"


def test_complete_disagreement_still_fails_without_disclosing_values():
    original = {"proof": {"completionStatus": "exact_complete", "secret": "SA"}, "proofModel": {"nodes": 2}}
    with pytest.raises(ValueError) as caught:
        _require_matching_exact_observed_reproof_v1(original,
            {"completionStatus": "exact_complete", "secret": "HK"}, {"nodes": 3})
    assert "proof.secret" in str(caught.value) and "proofModel.nodes" in str(caught.value)
    assert "SA" not in str(caught.value) and "HK" not in str(caught.value)


def test_identical_reproof_and_presence_checks():
    proof = {"completionStatus": "exact_complete"}
    _require_matching_exact_observed_reproof_v1({"proof": proof, "proofModel": {}}, proof, {})
    with pytest.raises(ValueError):
        _require_matching_exact_observed_reproof_v1({"proof": proof, "proofModel": {}},
                                                   dict(proof, added=None), {})


def test_optional_root_witnesses_are_reproved_without_rewriting_historical_proofs():
    proof = {"completionStatus": "exact_complete"}
    witnesses = [{"action": "SA", "worldId": "private-world", "principalVariation": ["SA", "HK"]}]
    historical = {"proof": proof, "proofModel": {}}
    _require_matching_exact_observed_reproof_v1(historical, proof, {}, fresh_root_witnesses=witnesses)
    current = {**historical, "rootWitnesses": witnesses}
    _require_matching_exact_observed_reproof_v1(current, proof, {}, fresh_root_witnesses=witnesses)
    for changed in (None, [], [{**witnesses[0], "principalVariation": ["SA", "H7"]}],
                    [{**witnesses[0], "worldId": "substituted-world"}]):
        with pytest.raises(ValueError) as caught:
            _require_matching_exact_observed_reproof_v1(current, proof, {}, fresh_root_witnesses=changed)
        assert str(caught.value).endswith("rootWitnesses")
        assert "private-world" not in str(caught.value) and "SA" not in str(caught.value)
