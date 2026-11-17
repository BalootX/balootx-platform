from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _check_fresh_imports(script: str) -> None:
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )


def test_challenge_public_exports_are_lazy_and_preserve_identity() -> None:
    _check_fresh_imports("""
import importlib
import sys
import baloot_platform.challenge as challenge

original_exports = {
    "InterestingnessVector", "MiningRequest",
    "ProgressiveChallengeMiner", "score_interestingness",
}
exports = challenge._EXPORT_MODULES
assert len(exports) == 153
assert len(challenge.__all__) == 149
assert set(challenge.__all__) == set(exports) - original_exports
assert set(exports).isdisjoint(vars(challenge))
assert set(exports) <= set(dir(challenge))
assert not any(name.startswith("baloot_platform.challenge.") for name in sys.modules)
for name, module in exports.items():
    namespace = {}
    exec(f"from baloot_platform.challenge import {name}", namespace)
    value = namespace[name]
    assert value is getattr(importlib.import_module(module), name), name
    assert vars(challenge)[name] is value, name
    assert getattr(challenge, name) is value, name
namespace = {}
exec("from baloot_platform.challenge import *", namespace)
assert set(namespace) - {"__builtins__"} == set(challenge.__all__)
try:
    challenge.not_a_public_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown exports must raise AttributeError")
""")


def test_pure_challenge_scoring_does_not_initialize_legacy_engine() -> None:
    _check_fresh_imports("""
import sys
from baloot_platform.challenge import InterestingnessVector, score_interestingness
from baloot_platform.challenge.interestingness_v4 import InterestingnessVector as direct_vector
from baloot_platform.challenge.scoring import ProfileChallengeScorerV1

assert InterestingnessVector is direct_vector
assert callable(score_interestingness)
for name in sys.modules:
    assert name != "baloot_platform.core.engine", name
    assert name != "baloot_platform.core.bots", name
    assert name != "baloot_platform.agents" and not name.startswith("baloot_platform.agents."), name
    assert name != "baloot_platform.challenge.miner", name
""")
