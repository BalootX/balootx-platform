from types import SimpleNamespace

import pytest

from baloot_platform.research.contracts import build_research_configuration_selector_v1
from baloot_platform.research.service import ResearchFoundryServiceV1


@pytest.mark.parametrize("configuration", [
    {"schemaVersion": True, "bundleRevisionId": "CBR-PREVIEW"},
    {"schemaVersion": 1.0, "bundleRevisionId": "CBR-PREVIEW"},
    {"schemaVersion": "1", "bundleRevisionId": "CBR-PREVIEW"},
    {"schemaVersion": 1, "bundleRevisionId": 123},
    {"schemaVersion": 1, "bundleRevisionId": True},
    {"schemaVersion": 1, "bundleRevisionId": ""},
    {"schemaVersion": 1, "bundleRevisionId": "C" * 201},
    [["schemaVersion", 1], ["bundleRevisionId", "CBR-PREVIEW"]],
])
def test_exact_preview_rejects_coercive_selector_before_freezing(configuration):
    def forbidden(**_kwargs):
        pytest.fail("invalid configuration reached bundle resolution")

    service = SimpleNamespace(
        execution_context_factory=SimpleNamespace(freeze_selector=forbidden),
    )
    with pytest.raises(ValueError, match="configuration"):
        ResearchFoundryServiceV1.preview_exact_trace_legal_actions(service, {
            "schemaVersion": 1,
            "configuration": configuration,
            "exactTracePrefix": {},
        })
    with pytest.raises(ValueError, match="configuration"):
        ResearchFoundryServiceV1._build_recipe_and_context(service, {
            "schemaVersion": 3,
            "configuration": configuration,
        })


@pytest.mark.parametrize("revision_id", ("CBR-PREVIEW", "C" * 200))
def test_configuration_selector_preserves_identity_and_names_unknown_fields(revision_id):
    original = {"schemaVersion": 1, "bundleRevisionId": revision_id}
    built = build_research_configuration_selector_v1(original)
    assert built == original
    assert built is not original
    with pytest.raises(ValueError, match="rulesHash"):
        build_research_configuration_selector_v1({**original, "rulesHash": "private-value"})
    assert original == {"schemaVersion": 1, "bundleRevisionId": revision_id}
