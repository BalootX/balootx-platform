from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from baloot_platform.config import ProfileBundle, data_root
from baloot_platform.plugins.base import PluginDescriptor


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    uri: str
    sha256: str
    size: int
    media_type: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "bytes": self.size,
            "mediaType": self.media_type,
            "name": self.name,
        }


class LocalContentAddressedArtifactStore:
    """Immutable local SHA-256 artifact repository.

    Metadata and bytes are addressed by digest rather than caller-provided
    filenames.  This makes datasets, models, replays and videos deduplicated,
    tamper-evident and safe to reference from SQLite or PostgreSQL.
    """

    descriptor = PluginDescriptor(
        "local_content_addressed_v2",
        "artifact_store",
        "2.0.0",
        "Local immutable SHA-256 repository with materialization support",
        ("immutable", "deduplicated", "materialize", "offline"),
        resource_class="storage",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._payload_lock = Lock()
        self._metadata_lock = Lock()

    def _payload_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            raise ValueError("artifact digest must be a SHA-256 hexadecimal string")
        return self.root / "sha256" / digest[:2] / digest[2:4] / digest

    @staticmethod
    def _digest_from_uri(value: str) -> str:
        if value.startswith("cas://sha256/"):
            return value.rsplit("/", 1)[-1].lower()
        if len(value) == 64:
            return value.lower()
        raise ValueError(f"unsupported local artifact reference: {value}")

    def put(self, data: bytes, *, media_type: str, name: str | None = None) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        payload = self._payload_path(digest)
        payload.parent.mkdir(parents=True, exist_ok=True)
        with self._payload_lock:
            payload_exists = payload.exists()
            if (
                payload_exists
                and hashlib.sha256(payload.read_bytes()).hexdigest() != digest
            ):
                raise RuntimeError("content-addressed artifact collision")
        if not payload_exists:
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=payload.parent,
                    prefix=f".{payload.name}.tmp-",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                with self._payload_lock:
                    if payload.exists():
                        if hashlib.sha256(payload.read_bytes()).hexdigest() != digest:
                            raise RuntimeError("content-addressed artifact collision")
                    else:
                        os.replace(temporary, payload)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        metadata = {
            "schemaVersion": 1,
            "sha256": digest,
            "bytes": len(data),
            "mediaType": media_type,
            "name": name,
        }
        metadata_path = payload.with_suffix(".metadata.json")
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
        with self._metadata_lock:
            if (
                metadata_path.exists()
                and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata
            ):
                # Names are descriptive only; retain the first immutable metadata record.
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if existing.get("sha256") != digest or int(
                    existing.get("bytes", -1)
                ) != len(data):
                    raise RuntimeError("artifact metadata collision")
            else:
                metadata_path.write_text(encoded, encoding="utf-8")
        return ArtifactReference(f"cas://sha256/{digest}", digest, len(data), media_type, name).to_dict()

    def put_file(self, path: str | Path, *, media_type: str, name: str | None = None) -> dict[str, Any]:
        source = Path(path)
        return self.put(source.read_bytes(), media_type=media_type, name=name or source.name)

    def get(self, reference: str) -> bytes:
        digest = self._digest_from_uri(reference)
        payload = self._payload_path(digest)
        data = payload.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("artifact checksum verification failed")
        return data

    def materialize(self, reference: str, *, suffix: str = "", cache_dir: str | Path | None = None) -> Path:
        digest = self._digest_from_uri(reference)
        source = self._payload_path(digest)
        if not source.is_file():
            raise FileNotFoundError(reference)
        if not suffix:
            self.get(reference)
            return source
        cache = Path(cache_dir or self.root / "materialized").expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"{digest}{suffix}"
        if not target.exists():
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=cache, prefix=f".{target.name}.tmp-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                shutil.copyfile(source, temporary)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                    raise RuntimeError("materialized artifact checksum verification failed")
                try:
                    os.rename(temporary, target)
                except FileExistsError:
                    # Another worker filled the cache; verify it below.
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError("materialized artifact checksum verification failed")
        return target


class S3ContentAddressedArtifactStore:
    """S3-compatible immutable artifact repository with a verified local cache."""

    descriptor = PluginDescriptor(
        "s3_content_addressed_v1",
        "artifact_store",
        "1.0.0",
        "S3-compatible SHA-256 repository with local verified materialization cache",
        ("immutable", "deduplicated", "distributed", "materialize"),
        resource_class="storage",
        deterministic=False,
    )

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "baloot",
        cache_dir: str | Path,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError('Install with: pip install "baloot-platform[s3]"') from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)

    def _key(self, digest: str) -> str:
        stem = f"sha256/{digest[:2]}/{digest[2:4]}/{digest}"
        return f"{self.prefix}/{stem}" if self.prefix else stem

    @staticmethod
    def _parse(reference: str) -> tuple[str, str]:
        parsed = urlparse(reference)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"unsupported S3 artifact reference: {reference}")
        return parsed.netloc, parsed.path.lstrip("/")

    def put(self, data: bytes, *, media_type: str, name: str | None = None) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        key = self._key(digest)
        uri = f"s3://{self.bucket}/{key}"
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            remote_digest = (head.get("Metadata") or {}).get("sha256")
            if remote_digest and remote_digest != digest:
                raise RuntimeError("remote content-addressed artifact collision")
        except self.client.exceptions.ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=media_type,
                Metadata={"sha256": digest, "name": name or ""},
            )
        else:
            # Matching metadata alone does not verify the stored bytes.
            self.get(uri)
        return ArtifactReference(uri, digest, len(data), media_type, name).to_dict()

    def put_file(self, path: str | Path, *, media_type: str, name: str | None = None) -> dict[str, Any]:
        source = Path(path)
        return self.put(source.read_bytes(), media_type=media_type, name=name or source.name)

    def get(self, reference: str) -> bytes:
        bucket, key = self._parse(reference)
        response = self.client.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        expected = key.rsplit("/", 1)[-1]
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("downloaded S3 artifact checksum verification failed")
        return data

    def materialize(self, reference: str, *, suffix: str = "", cache_dir: str | Path | None = None) -> Path:
        _bucket, key = self._parse(reference)
        digest = key.rsplit("/", 1)[-1]
        cache = Path(cache_dir or self.cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"{digest}{suffix}"
        if not target.exists():
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=cache, prefix=f".{target.name}.tmp-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(self.get(reference))
                try:
                    os.rename(temporary, target)
                except FileExistsError:
                    # Another worker filled the cache; verify it below.
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            target.unlink(missing_ok=True)
            raise RuntimeError("cached S3 artifact checksum verification failed")
        return target


def build_artifact_store(bundle: ProfileBundle, *, writable_root: str | Path | None = None) -> Any:
    """Construct the configured artifact plugin.

    Environment variables intentionally override profile locations/secrets so
    immutable published profiles never contain production credentials.
    """
    storage = bundle.component("storage")
    configured = storage.get("artifacts", {})
    plugin = os.environ.get("BALOOT_ARTIFACT_PLUGIN", str(configured.get("plugin", "local_content_addressed_v2")))
    base = Path(writable_root or data_root())
    if plugin in {"local_content_addressed_v1", "local_content_addressed_v2"}:
        root = os.environ.get("BALOOT_ARTIFACT_ROOT", str(base / "artifacts"))
        return LocalContentAddressedArtifactStore(root)
    if plugin == "s3_content_addressed_v1":
        bucket = os.environ.get("BALOOT_ARTIFACT_BUCKET", str(configured.get("bucket", ""))).strip()
        if not bucket:
            raise ValueError("BALOOT_ARTIFACT_BUCKET is required for S3 artifact storage")
        return S3ContentAddressedArtifactStore(
            bucket=bucket,
            prefix=os.environ.get("BALOOT_ARTIFACT_PREFIX", str(configured.get("prefix", "baloot"))),
            cache_dir=os.environ.get("BALOOT_ARTIFACT_CACHE", str(base / "artifact-cache")),
            endpoint_url=os.environ.get("BALOOT_S3_ENDPOINT") or configured.get("endpoint_url"),
            region_name=os.environ.get("AWS_REGION") or configured.get("region_name"),
        )
    raise ValueError(f"unsupported artifact plugin {plugin}")
