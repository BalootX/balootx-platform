from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from baloot_platform.storage.artifact_store import (
    LocalContentAddressedArtifactStore,
    S3ContentAddressedArtifactStore,
)


@pytest.fixture
def s3_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class ClientError(Exception):
        def __init__(self, code: str):
            self.response = {"Error": {"Code": code}}

    class Client:
        exceptions = SimpleNamespace(ClientError=ClientError)

        def __init__(self):
            self.objects = {}
            self.puts = []
            self.read_error = None

        def head_object(self, *, Bucket, Key):
            if (Bucket, Key) not in self.objects:
                raise ClientError("404")
            return {"Metadata": self.objects[Bucket, Key][1]}

        def get_object(self, *, Bucket, Key):
            if self.read_error:
                raise ClientError(self.read_error)
            return {"Body": io.BytesIO(self.objects[Bucket, Key][0])}

        def put_object(self, **request):
            self.puts.append(request)
            self.objects[request["Bucket"], request["Key"]] = request["Body"], request["Metadata"]

    client = Client()
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: client))
    store = S3ContentAddressedArtifactStore(bucket="private-test", cache_dir=tmp_path / "cache")
    return store, client


@pytest.mark.parametrize("metadata_present", [True, False])
def test_s3_put_reuses_verified_bytes_with_or_without_metadata(s3_store, metadata_present):
    store, client = s3_store
    data = b"baloot-research-evidence"
    digest = hashlib.sha256(data).hexdigest()
    client.objects[store.bucket, store._key(digest)] = data, ({"sha256": digest} if metadata_present else {})
    reference = store.put(data, media_type="application/json", name="proof.json")
    assert reference["sha256"] == digest
    assert reference["bytes"] == len(data)
    assert store.get(reference["uri"]) == data
    assert client.puts == []


@pytest.mark.parametrize("metadata_present", [True, False])
@pytest.mark.parametrize("corrupt", [b"bad", b"baloot-research-evidencf"])
def test_s3_put_rejects_corrupt_existing_bytes_without_overwriting(s3_store, metadata_present, corrupt):
    store, client = s3_store
    data = b"baloot-research-evidence"
    digest = hashlib.sha256(data).hexdigest()
    key = store.bucket, store._key(digest)
    client.objects[key] = corrupt, ({"sha256": digest} if metadata_present else {})
    with pytest.raises(RuntimeError, match="checksum verification failed"):
        store.put(data, media_type="application/json")
    assert client.objects[key][0] == corrupt
    assert client.puts == []


@pytest.mark.parametrize("read_error", ["AccessDenied", "404", "NoSuchKey", "NotFound"])
def test_s3_put_does_not_trust_metadata_when_object_cannot_be_read(s3_store, read_error):
    store, client = s3_store
    data = b"baloot-research-evidence"
    digest = hashlib.sha256(data).hexdigest()
    client.objects[store.bucket, store._key(digest)] = data, {"sha256": digest}
    client.read_error = read_error
    with pytest.raises(client.exceptions.ClientError):
        store.put(data, media_type="application/json")
    assert client.puts == []


def test_s3_put_file_creates_missing_artifact_and_roundtrips(s3_store, tmp_path):
    store, client = s3_store
    source = tmp_path / "proof.json"
    source.write_bytes(b"baloot-research-evidence")
    reference = store.put_file(source, media_type="application/json")
    assert len(client.puts) == 1
    assert client.puts[0]["Metadata"]["sha256"] == reference["sha256"]
    assert reference["name"] == source.name
    assert store.get(reference["uri"]) == source.read_bytes()


def test_s3_concurrent_materialize_uses_unique_temporary_files(s3_store, monkeypatch):
    store, _client = s3_store
    data = b"baloot-research-evidence"
    reference = store.put(data, media_type="application/json")
    target = store.cache_dir / f"{reference['sha256']}.json"
    worker_count = 4
    publishers = Barrier(worker_count)
    original_replace = os.replace
    original_rename = os.rename

    def synchronized_replace(source, destination):
        if Path(destination) == target:
            publishers.wait(timeout=10)
        return original_replace(source, destination)

    def synchronized_rename(source, destination):
        if Path(destination) == target:
            publishers.wait(timeout=10)
        return original_rename(source, destination)

    monkeypatch.setattr(os, "replace", synchronized_replace)
    monkeypatch.setattr(os, "rename", synchronized_rename)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        paths = list(executor.map(lambda _: store.materialize(reference["uri"], suffix=".json"),
                                  range(worker_count)))
    assert paths == [target] * worker_count
    assert all(path.read_bytes() == data for path in paths)
    assert list(store.cache_dir.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["download", "install"])
def test_s3_failed_materialize_leaves_no_partial_cache(s3_store, monkeypatch, failure):
    store, client = s3_store
    reference = store.put(b"baloot-research-evidence", media_type="application/json")
    if failure == "download":
        client.read_error = "AccessDenied"
        error = client.exceptions.ClientError
    else:
        def failed_install(source, destination):
            raise OSError("cache install failed")
        monkeypatch.setattr(os, "replace", failed_install)
        monkeypatch.setattr(os, "rename", failed_install)
        error = OSError
    with pytest.raises(error):
        store.materialize(reference["uri"], suffix=".json")
    assert not list(store.cache_dir.iterdir())


def test_local_content_addressed_artifacts_are_deduplicated_and_verified(tmp_path: Path) -> None:
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    first = store.put(b"baloot-learning-artifact", media_type="application/octet-stream", name="one.bin")
    second = store.put(b"baloot-learning-artifact", media_type="application/octet-stream", name="two.bin")
    assert first["sha256"] == second["sha256"]
    assert first["uri"] == second["uri"]
    assert store.get(first["uri"]) == b"baloot-learning-artifact"
    source = store.materialize(first["uri"])
    assert source.read_bytes() == b"baloot-learning-artifact"
    materialized = store.materialize(first["uri"], suffix=".bin", cache_dir=tmp_path / "cache")
    assert materialized.read_bytes() == b"baloot-learning-artifact"
    assert materialized.name.endswith(".bin")
    source.write_bytes(b"corrupted-artifact")
    with pytest.raises(RuntimeError, match="artifact checksum verification failed"):
        store.materialize(first["uri"])


@pytest.mark.parametrize("failure", ["copy", "install", "checksum"])
def test_local_failed_materialize_leaves_no_partial_cache(tmp_path, monkeypatch, failure):
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    data = b"baloot-research-evidence"
    reference = store.put(data, media_type="application/json")
    source = store.materialize(reference["uri"])
    cache = tmp_path / "cache"
    with monkeypatch.context() as patch:
        if failure == "copy":
            def failed_copy(source, destination):
                Path(destination).write_bytes(b"partial")
                raise OSError("cache copy interrupted")
            patch.setattr(shutil, "copyfile", failed_copy)
            error = OSError
        elif failure == "install":
            def failed_install(source, destination):
                raise OSError("cache install failed")
            patch.setattr(os, "rename", failed_install)
            error = OSError
        else:
            source.write_bytes(b"corrupt")
            error = RuntimeError
        with pytest.raises(error):
            store.materialize(reference["uri"], suffix=".json", cache_dir=cache)
        assert not list(cache.iterdir())
    source.write_bytes(data)
    restored = store.materialize(reference["uri"], suffix=".json", cache_dir=cache)
    assert restored.read_bytes() == data
    assert list(cache.iterdir()) == [restored]


def test_local_concurrent_materialize_never_exposes_partial_bytes(tmp_path, monkeypatch):
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    data = b"baloot-research-evidence"
    reference = store.put(data, media_type="application/json")
    cache = tmp_path / "cache"
    target = cache / f"{reference['sha256']}.json"
    copying = Barrier(4)
    original_copy = shutil.copyfile

    def partial_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        assert not target.exists()
        copying.wait(timeout=10)
        return original_copy(source, destination)

    monkeypatch.setattr(shutil, "copyfile", partial_copy)
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(
            lambda _: store.materialize(reference["uri"], suffix=".json", cache_dir=cache),
            range(4),
        ))
    assert paths == [target] * 4
    assert target.read_bytes() == data
    assert list(cache.iterdir()) == [target]


def test_concurrent_identical_artifact_puts_use_unique_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    store = LocalContentAddressedArtifactStore(root)
    data = b"baloot-concurrent-artifact" * 4096
    digest = hashlib.sha256(data).hexdigest()
    worker_count = 16
    starts = Barrier(worker_count)
    legacy_writes = Barrier(worker_count)
    original_write_bytes = Path.write_bytes

    def synchronized_legacy_write(path: Path, payload: bytes) -> int:
        written = original_write_bytes(path, payload)
        if ".tmp-" in path.name:
            legacy_writes.wait(timeout=10)
        return written

    def put_identical(_ordinal: int) -> dict[str, object]:
        starts.wait(timeout=10)
        return store.put(
            data,
            media_type="application/octet-stream",
            name="concurrent.bin",
        )

    monkeypatch.setattr(Path, "write_bytes", synchronized_legacy_write)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        references = list(executor.map(put_identical, range(worker_count)))

    assert {reference["sha256"] for reference in references} == {digest}
    assert {reference["uri"] for reference in references} == {
        f"cas://sha256/{digest}"
    }
    assert store.get(references[0]["uri"]) == data
    assert len([path for path in root.rglob(digest) if path.is_file()]) == 1
    assert not [path for path in root.rglob("*") if ".tmp-" in path.name]
    metadata = json.loads(
        (store._payload_path(digest).with_suffix(".metadata.json")).read_text(
            encoding="utf-8"
        )
    )
    assert metadata == {
        "schemaVersion": 1,
        "sha256": digest,
        "bytes": len(data),
        "mediaType": "application/octet-stream",
        "name": "concurrent.bin",
    }




def test_storage_profile_uses_current_artifact_plugin(root: Path) -> None:
    text = (root / "configs/storage/local_v1.yaml").read_text(encoding="utf-8")
    assert "local_content_addressed_v2" in text


def test_postgres_learning_migration_and_runtime_contract_exist(root: Path) -> None:
    migration = (root / "migrations/002_learning.sql").read_text(encoding="utf-8")
    required = {
        "archived_games",
        "learning_decisions",
        "learning_feedback",
        "learning_datasets",
        "learning_models",
        "learning_challenges",
        "learning_artifacts",
    }
    assert all(name in migration for name in required)
    source = (root / "src/baloot_platform/storage/postgres_learning.py").read_text(encoding="utf-8")
    assert "class PostgresLearningStore" in source
    assert "ConnectionPool" in source
    assert "trusted_payload" in source
