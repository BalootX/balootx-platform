from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .actions import ACTION_SCHEMA_VERSION
from .features import FEATURE_SCHEMA_VERSION


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    id: str
    examples_path: Path
    manifest_path: Path
    example_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "exampleCount": self.example_count,
            "sha256": self.sha256,
            "examples": str(self.examples_path),
            "manifest": str(self.manifest_path),
        }


class DatasetWriter:
    """Write immutable, content-addressed JSONL training datasets.

    Examples contain only explicitly supplied features/targets. The writer
    records rules, feature and action schema identities so models cannot be
    silently trained against a different engine contract.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        examples: Iterable[dict[str, Any]],
        *,
        rules_hash: str,
        bundle_hash: str,
        purpose: str,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetArtifact:
        rows = [dict(item) for item in examples]
        body = b"".join(_canonical(row) + b"\n" for row in rows)
        digest = hashlib.sha256(body).hexdigest()
        dataset_id = f"ds_{digest[:20]}"
        directory = self.root / digest[:2] / digest
        directory.mkdir(parents=True, exist_ok=True)
        examples_path = directory / "examples.jsonl"
        if not examples_path.exists():
            examples_path.write_bytes(body)
        manifest = {
            "schemaVersion": 1,
            "id": dataset_id,
            "sha256": digest,
            "purpose": purpose,
            "exampleCount": len(rows),
            "rulesHash": rules_hash,
            "bundleHash": bundle_hash,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
            "actionSchemaVersion": ACTION_SCHEMA_VERSION,
            "metadata": metadata or {},
        }
        manifest_path = directory / "manifest.json"
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("content-addressed dataset manifest collision")
        manifest_path.write_text(encoded, encoding="utf-8")
        return DatasetArtifact(dataset_id, examples_path, manifest_path, len(rows), digest)
