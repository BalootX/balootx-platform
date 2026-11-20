"""Opt-in deterministic work clock for generation-path tests (v1).

See docs/DETERMINISTIC_TEST_BUDGETS.md. Test-only: production code is untouched.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
import threading
import time
import types
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path

from scripts.run_isolated_pytest import FILE_SAFETY_SECONDS

VERSION = 1
# Below the 900 s per-file CI timeout, so a seconds-only loop fails cleanly instead of being killed.
SAFETY_SECONDS = 600.0
_CLOCKS = ("monotonic", "perf_counter", "process_time", "thread_time")
_REAL = {name: getattr(time, name) for name in _CLOCKS}
_PATCHED_PREFIXES = ("baloot_platform", "scripts")
_SOLVER_PACKAGES = ("baloot_platform.analysis", "baloot_platform.challenge", "baloot_platform.research")


@lru_cache(maxsize=None)
def solver_test_module(path: Path) -> bool:
    """Select direct solver imports, including imports inside test functions."""
    if path.name in {"test_exact_holdout_labels.py", "test_v9_shared_explain_review.py"}:
        return True  # These consumers load solver code indirectly.
    source = path.read_text(encoding="utf-8-sig")
    for node in ast.walk(ast.parse(source)):
        imports = ([item.name for item in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if any(name == "ortools" or name.startswith(("ortools.",
                "baloot_platform.research", "baloot_platform.analysis")) for name in imports):
            return True
    return False


def file_work_safety_seconds(started: float | None) -> float:
    if started is None:
        return SAFETY_SECONDS
    remaining = FILE_SAFETY_SECONDS - (_REAL["monotonic"]() - started)
    if remaining <= 0:
        raise TimeoutError("deterministic test file exhausted its 840-second safety budget")
    return min(SAFETY_SECONDS, remaining)


class DeterministicWorkClockV1:
    """Time that advances only by native CP-SAT deterministic work."""

    version = VERSION

    def __init__(self, safety_seconds: float = SAFETY_SECONDS):
        self.safety_seconds = float(safety_seconds)
        self.work = 0.0
        self._lock = threading.Lock()
        self._bases = {name: clock() for name, clock in _REAL.items()}

    def real_remaining(self) -> float:
        return self.safety_seconds - (_REAL["monotonic"]() - self._bases["monotonic"])

    def read(self, name: str) -> float:
        if self.real_remaining() <= 0:
            return self._bases[name] + 1e9  # Safety stop: every budget trips; never hang.
        with self._lock:
            return self._bases[name] + self.work

    def charge(self, deterministic_time: float) -> None:
        with self._lock:
            self.work += max(0.0, float(deterministic_time))

    def time_module(self) -> types.ModuleType:
        proxy = types.ModuleType("time")
        proxy.__dict__.update(time.__dict__)
        for name in _CLOCKS:
            proxy.__dict__[name] = lambda name=name: self.read(name)
            proxy.__dict__[name + "_ns"] = lambda name=name: int(self.read(name) * 1e9)
        return proxy


@lru_cache(maxsize=1)
def _import_solver_packages() -> None:
    # Lazily imported solver modules must already be loaded to be patched.
    for package in _SOLVER_PACKAGES:
        root = importlib.import_module(package)
        for info in pkgutil.walk_packages(root.__path__, package + "."):
            try:
                importlib.import_module(info.name)
            except ImportError:  # Optional extras are not solver dependencies.
                pass


@contextmanager
def deterministic_work_clock_v1(safety_seconds: float = SAFETY_SECONDS):
    """Make in-process solve budgets depend on deterministic work only."""
    from ortools.sat.python import cp_model

    _import_solver_packages()
    clock = DeterministicWorkClockV1(safety_seconds)
    proxy = clock.time_module()
    replacements = {time: proxy, **{_REAL[name]: proxy.__dict__[name] for name in _CLOCKS}}
    restore = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(_PATCHED_PREFIXES):
            continue
        for attribute, value in list(vars(module).items()):
            try:
                replacement = replacements.get(value)
            except TypeError:  # Unhashable module attribute.
                continue
            if replacement is not None:
                restore.append((module, attribute, value))
                setattr(module, attribute, replacement)

    original_solve = cp_model.CpSolver.solve
    active = True

    @wraps(original_solve)
    def solve(self, model, solution_callback=None):
        if not active:
            return original_solve(self, model, solution_callback)
        parameters = self.parameters
        saved = (parameters.max_time_in_seconds, parameters.max_deterministic_time,
                 parameters.num_workers, parameters.num_search_workers)
        parameters.max_deterministic_time = min(saved[0], saved[1])
        parameters.max_time_in_seconds = max(clock.real_remaining(), 0.001)
        if not saved[2] and not saved[3]:
            parameters.num_workers = 1
        try:
            status = original_solve(self, model, solution_callback)
        finally:
            (parameters.max_time_in_seconds, parameters.max_deterministic_time,
             parameters.num_workers, parameters.num_search_workers) = saved
        clock.charge(self.deterministic_time)
        return status

    def release():
        """Restore real clocks early, e.g. before a test checks real elapsed time."""
        nonlocal active
        active = False
        cp_model.CpSolver.solve = original_solve
        while restore:
            module, attribute, value = restore.pop()
            setattr(module, attribute, value)

    clock.release = release
    cp_model.CpSolver.solve = solve
    try:
        yield clock
    finally:
        release()


def set_monotonic(monkeypatch, module, *timestamps):
    """Supply the prescribed clock readings and hold the clock at the final tick."""
    ticks = iter(timestamps)
    monkeypatch.setattr(module, "time", types.SimpleNamespace(
        **{**vars(module.time), "monotonic": lambda: next(ticks, timestamps[-1])}))
