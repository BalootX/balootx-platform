"""Public adjudication contracts with lazy service/storage imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AdjudicationCase": (".models", "AdjudicationCase"),
    "CaseStatus": (".models", "CaseStatus"),
    "EvidenceItem": (".models", "EvidenceItem"),
    "Ruling": (".models", "Ruling"),
    "RulingOutcome": (".models", "RulingOutcome"),
    "ViolationCode": (".models", "ViolationCode"),
    "AdjudicationProfile": (".service", "AdjudicationProfile"),
    "AdjudicationService": (".service", "AdjudicationService"),
    "SQLiteAdjudicationStore": (".store", "SQLiteAdjudicationStore"),
    "PostgresAdjudicationStore": (".store", "PostgresAdjudicationStore"),
    "AdjudicationEventEvidenceV1": (".research", "AdjudicationEventEvidenceV1"),
    "AdjudicationResearchAuthorityV1": (
        ".research",
        "AdjudicationResearchAuthorityV1",
    ),
    "AdjudicationResearchCapabilityV1": (
        ".research",
        "AdjudicationResearchCapabilityV1",
    ),
    "AdjudicationResearchEventKindV1": (
        ".research",
        "AdjudicationResearchEventKindV1",
    ),
    "AdjudicationResearchReportV1": (
        ".research",
        "AdjudicationResearchReportV1",
    ),
    "AuthoredAdjudicationEventV1": (".research", "AuthoredAdjudicationEventV1"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
