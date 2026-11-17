from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from .features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ModelOutput:
    policy_logits: np.ndarray
    value: float
    beliefs: np.ndarray | None = None
    intentions: np.ndarray | None = None


@runtime_checkable
class ModelPlugin(Protocol):
    id: str
    version: str
    def infer(self, observation: np.ndarray, legal_mask: tuple[int, ...]) -> ModelOutput: ...


def rank_legal_actions(
    output: ModelOutput,
    legal_action_ids: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    """Validate one model response and rank only authoritative legal actions."""
    logits = np.asarray(output.policy_logits, dtype=np.float32)
    if logits.shape != (ACTION_COUNT,):
        raise ValueError("model policy output shape mismatch")
    if not math.isfinite(float(output.value)):
        raise ValueError("model value output must be finite")
    legal = tuple(dict.fromkeys(int(identifier) for identifier in legal_action_ids))
    if not legal:
        raise ValueError("at least one legal action is required")
    if any(identifier < 0 or identifier >= ACTION_COUNT for identifier in legal):
        raise ValueError("legal action is outside the stable action space")
    ranked: list[tuple[int, float]] = []
    for identifier in legal:
        score = float(logits[identifier])
        if not math.isfinite(score):
            raise ValueError("legal model policy scores must be finite")
        ranked.append((identifier, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return tuple(ranked)


def model_runtime_identity(model: ModelPlugin | None) -> dict[str, Any] | None:
    """Return stable model provenance without exposing local artifact paths."""
    if model is None:
        return None
    plugin_id = str(getattr(model, "id", model.__class__.__name__))
    version = str(getattr(model, "version", ""))
    return {
        "modelId": str(getattr(model, "model_id", plugin_id)),
        "pluginId": plugin_id,
        "pluginVersion": version,
        "artifactSha256": getattr(model, "artifact_sha256", None),
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "actionSchemaVersion": ACTION_SCHEMA_VERSION,
        "runtime": str(getattr(model, "runtime_identity", "python")),
    }


class UniformLegalModel:
    id = "uniform_legal_v1"
    version = "1.0.0"
    model_id = "uniform_legal_v1"
    artifact_sha256 = None
    runtime_identity = f"numpy@{np.__version__}"

    def infer(self, observation: np.ndarray, legal_mask: tuple[int, ...]) -> ModelOutput:
        if observation.shape != (FEATURE_COUNT,):
            raise ValueError("unexpected observation shape")
        logits = np.full(ACTION_COUNT, -1e9, dtype=np.float32)
        for index, legal in enumerate(legal_mask):
            if legal:
                logits[index] = 0.0
        return ModelOutput(logits, 0.0)


class OnnxModel:
    """Optional ONNX Runtime adapter with strict engine-contract validation."""

    def __init__(self, path: str | Path, manifest_path: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install the optional dependency with: pip install "baloot-platform[ml]"') from exc
        self.path = Path(path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
            raise ValueError("model feature schema is incompatible")
        if manifest.get("actionSchemaVersion") != ACTION_SCHEMA_VERSION:
            raise ValueError("model action schema is incompatible")
        actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if manifest.get("sha256") and manifest["sha256"] != actual:
            raise ValueError("model checksum mismatch")
        self.id = str(manifest["id"])
        self.version = str(manifest["version"])
        self.model_id = str(manifest.get("modelId", self.id))
        self.artifact_sha256 = actual
        self.runtime_identity = f"onnxruntime@{ort.__version__}"
        self._session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def infer(self, observation: np.ndarray, legal_mask: tuple[int, ...]) -> ModelOutput:
        array = np.asarray(observation, dtype=np.float32).reshape(1, FEATURE_COUNT)
        outputs = self._session.run(None, {self._input: array})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(ACTION_COUNT)
        for index, legal in enumerate(legal_mask):
            if not legal:
                logits[index] = -1e9
        value = float(np.asarray(outputs[1]).reshape(-1)[0]) if len(outputs) > 1 else 0.0
        beliefs = np.asarray(outputs[2]).reshape(-1) if len(outputs) > 2 else None
        intentions = np.asarray(outputs[3]).reshape(-1) if len(outputs) > 3 else None
        return ModelOutput(logits, value, beliefs, intentions)
