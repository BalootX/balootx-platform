from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module(root: Path):
    path = root / "scripts/engine_conformance.py"
    spec = importlib.util.spec_from_file_location("engine_conformance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v21_rule_conformance_smoke_is_deterministic(root: Path):
    module = _module(root)
    first = module.run(rounds=3, invalid_cases=12)
    second = module.run(rounds=3, invalid_cases=12)
    assert first == second
    assert first["rounds"] == first["replayRounds"] == 3
    assert first["plays"] == 96
    assert first["applyUndoByteIdentityChecks"] == 3
    assert first["observerBoundaryChecks"] == 12
    assert first["invalidCasesRejected"] == 12
    assert first["invalidCategories"] == {
        "malformed": 3,
        "negative": 3,
        "not_legal": 3,
        "overflow": 3,
    }
    assert first["externalCertification"] is False


def test_v21_rule_conformance_refuses_under_scale_profile_claim(root: Path):
    module = _module(root)
    with pytest.raises(ValueError, match="round count"):
        module.run(rounds=3, invalid_cases=12, enforce_profile=True)


