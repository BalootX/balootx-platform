from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

# Establish a fresh, process-local runtime before any test module imports the
# FastAPI application or durable stores. This keeps the suite hermetic and
# prevents previous runs from growing or locking the default .runtime database.
_TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="baloot-pytest-"))
os.environ.setdefault("BALOOT_ENV", "test")
os.environ.setdefault("BALOOT_API_KEY", "synthetic-pytest-service")
os.environ.setdefault("BALOOT_DATA_DIR", str(_TEST_RUNTIME))
os.environ.setdefault("BALOOT_RUNTIME", str(_TEST_RUNTIME))
os.environ.setdefault("BALOOT_ROOT", str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from baloot_platform.core.rules import RuleBook

# The minimal CI preflight does not install the dev extra or run property tests.
try:
    from hypothesis import settings
except ModuleNotFoundError as exc:
    if exc.name != "hypothesis":
        raise
else:
    settings.register_profile(
        "balootx", max_examples=20, derandomize=True, deadline=None
    )
    settings.register_profile(
        "nightly", parent=settings.get_profile("balootx"), max_examples=500
    )
    settings.load_profile("balootx")

_WORK_FILE_STARTED = pytest.StashKey[float]()


def pytest_configure(config):
    config.addinivalue_line("markers", "controlled_clock: test supplies scripted clock readings")


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    from deterministic_work import deterministic_work_clock_v1, solver_test_module

    if isinstance(collector, pytest.Module):
        collector.stash[_WORK_FILE_STARTED] = time.monotonic()
        if solver_test_module(collector.path):
            with deterministic_work_clock_v1():
                return (yield)
    return (yield)


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rules(root: Path) -> RuleBook:
    return RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")


@pytest.fixture(scope="session")
def convention(root: Path):
    return yaml.safe_load((root / "configs/conventions/conservative_v1.yaml").read_text())


@pytest.fixture(scope="module")
def _work_file_started(request):
    if os.environ.get("BALOOT_PYTEST_FILE_BUDGET") == "1":
        return request.node.stash[_WORK_FILE_STARTED]
    return None


@pytest.fixture(autouse=True)
def _solver_work_default(request):
    from deterministic_work import solver_test_module

    if solver_test_module(request.path):
        request.getfixturevalue("deterministic_work")


@pytest.fixture
def deterministic_work(request, _work_file_started):
    """Solver modules default to deterministic work; real_clock opts out."""
    from deterministic_work import deterministic_work_clock_v1, file_work_safety_seconds

    if request.node.get_closest_marker("real_clock") is not None:
        yield None
        return
    with deterministic_work_clock_v1(file_work_safety_seconds(_work_file_started)) as clock:
        yield clock


@pytest.fixture(scope="module")
def fresh_professional_bundles(tmp_path_factory):
    import json

    from baloot_platform.video.render_bundle import RenderBundleV1
    from scripts.generate_professional_video_prototypes import generate

    paths = generate(tmp_path_factory.mktemp("professional-bundles"))
    return {
        preset: RenderBundleV1(json.loads(path.read_text(encoding="utf-8")))
        for preset, path in zip(("hooked_flagship", "decision_clip"), paths, strict=True)
    }


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    # The FastAPI module owns a process-local rendering executor.  Shut it down
    # explicitly so per-file and clean-room test lanes terminate deterministically.
    import sys
    api_module = sys.modules.get("baloot_platform.service.api")
    container = getattr(api_module, "CONTAINER", None) if api_module else None
    shutdown = getattr(container, "shutdown", None)
    if callable(shutdown):
        shutdown()
    shutil.rmtree(_TEST_RUNTIME, ignore_errors=True)
