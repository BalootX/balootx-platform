import copy

import pytest

from baloot_platform.research.scenario_contracts import ScenarioOriginV1


def _origin():
    return {"kind": "exact_trace_prefix", "exactTracePrefix": {
        "schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
    }}


def test_recorded_opportunities_preserve_absence_order_and_identity():
    legacy = ScenarioOriginV1.build(_origin())
    assert "recordedEscalationOpportunities" not in legacy.to_dict()
    assert ScenarioOriginV1.build(legacy.to_dict()).sha256 == legacy.sha256
    raw = {**_origin(), "recordedEscalationOpportunities": []}
    empty = ScenarioOriginV1.build(raw)
    assert empty.sha256 != legacy.sha256
    raw["recordedEscalationOpportunities"] = [
        {"actor": "west", "actionCode": "decline"},
        {"actor": "east", "actionCode": "double:open"},
    ]
    authored = ScenarioOriginV1.build(raw)
    assert ScenarioOriginV1.build(authored.to_dict()).sha256 == authored.sha256
    changed = copy.deepcopy(authored.to_dict())
    changed["recordedEscalationOpportunities"].reverse()
    with pytest.raises(ValueError, match="checksum"):
        ScenarioOriginV1.build(changed)


@pytest.mark.parametrize("events", [None, "decline", [{}],
    [{"actor": True, "actionCode": "decline"}],
    [{"actor": "west", "actionCode": 1}],
    [{"actor": "west", "actionCode": " decline"}],
    [{"actor": "west", "actionCode": "decline", "visible": True}],
])
def test_recorded_opportunities_reject_coercion_and_extra_fields(events):
    with pytest.raises((ValueError, TypeError)):
        ScenarioOriginV1.build({**_origin(), "recordedEscalationOpportunities": events})


def test_search_cannot_inject_recorded_opportunities():
    with pytest.raises(ValueError, match="require exact"):
        ScenarioOriginV1.build({"kind": "search", "recordedEscalationOpportunities": []})


def test_api_recorded_opportunities_use_canonical_origin_validation():
    from baloot_platform.service.api import ScenarioOriginV1Request
    raw = {**_origin(), "recordedEscalationOpportunities": [{"actor": "west", "actionCode": "decline"}]}
    request = ScenarioOriginV1Request.model_validate(raw)
    assert ScenarioOriginV1.build(request.model_dump(exclude_none=True, exclude_unset=True)).sha256 == ScenarioOriginV1.build(raw).sha256
    for invalid in (None, [{"actor": "west", "actionCode": " decline"}], [{"actor": "west", "actionCode": False}]):
        with pytest.raises(ValueError):
            ScenarioOriginV1Request.model_validate({**_origin(), "recordedEscalationOpportunities": invalid})
