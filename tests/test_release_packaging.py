from __future__ import annotations

import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(params=("package_release.py", "package_release_v20.py"))
def packager(request, tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location("release_packaging", SCRIPTS / request.param)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_packaging_excludes_scratch_work(packager, tmp_path: Path) -> None:
    scratch = tmp_path / "work/audit-notes.txt"
    scratch.parent.mkdir()
    scratch.write_bytes(b"local audit notes\n")

    assert not packager.include(scratch)


def test_packaging_excludes_sqlite_databases_and_sidecars(packager, tmp_path: Path) -> None:
    for suffix in (".sqlite", ".sqlite3"):
        database = tmp_path / f"runtime{suffix}"
        database.write_bytes(b"local database")
        assert not packager.include(database), database.name
    for database_suffix in (".sqlite", ".sqlite3", ".db"):
        for sidecar in ("shm", "wal", "journal"):
            path = tmp_path / f"runtime{database_suffix}-{sidecar}"
            path.write_bytes(b"transient database state")
            assert not packager.include(path), path.name


def test_packaging_preserves_source_and_possible_database_fixtures(
    packager, tmp_path: Path
) -> None:
    for relative in ("src/module.py", "tests/fixtures/reference.db", "docs/database.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reviewable repository content\n")
        assert packager.include(path), relative


def test_sqlite_wal_lifecycle_cannot_change_the_package_inventory(
    packager, tmp_path: Path
) -> None:
    source = tmp_path / "module.py"
    source.write_bytes(b"VALUE = 1\n")
    database = tmp_path / "runtime.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("CREATE TABLE fixture (value TEXT)")
        connection.commit()
        sidecars = [Path(f"{database}-{suffix}") for suffix in ("shm", "wal")]
        assert all(path.exists() for path in sidecars)
        while_open = sorted(
            path.relative_to(tmp_path).as_posix()
            for path in tmp_path.rglob("*")
            if packager.include(path)
        )
    assert all(not path.exists() for path in sidecars)
    after_close = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if packager.include(path)
    )

    assert while_open == after_close == ["module.py"]



@pytest.mark.parametrize(
    "script",
    ("package_release.py", "package_release_v17.py", "package_release_v18.py",
     "package_release_v19.py", "package_release_v20.py"),
)
def test_all_packagers_exclude_private_qualification_inputs_and_state(
    script, tmp_path: Path, monkeypatch,
) -> None:
    spec = importlib.util.spec_from_file_location("private_boundary_packager", SCRIPTS / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for prefix in ("", "nested/"):
        for directory in ("private-inputs", ".local-state", ".codex-local"):
            for filename in ("candidates.jsonl", "evidence.json"):
                path = tmp_path / prefix / directory / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"private qualification input\n")
                assert not module.include(path), path.relative_to(tmp_path)
    public = tmp_path / "src" / "example.py"
    public.parent.mkdir(exist_ok=True)
    public.write_bytes(b"VALUE = 1\n")
    assert module.include(public)
