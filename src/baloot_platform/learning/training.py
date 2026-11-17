from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from baloot_platform.ml.actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from baloot_platform.ml.features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION
from baloot_platform.ml.models import ModelOutput, rank_legal_actions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 30.0))
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True, slots=True)
class TrainResult:
    model_id: str
    plugin_id: str
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    metrics: dict[str, Any]


class NumpyLinearPolicyValueModel:
    """Small deterministic baseline for supervised policy/value learning.

    This is intentionally simple and auditable. It is useful for validating the
    end-to-end learning path before introducing larger PyTorch/ONNX models.
    Legal masks remain authoritative at inference time.
    """

    id = "numpy_linear_policy_value_v1"
    version = "1.0.0"

    def __init__(self, artifact_path: str | Path, manifest_path: str | Path | None = None):
        self.artifact_path = Path(artifact_path)
        self.artifact_sha256 = hashlib.sha256(
            self.artifact_path.read_bytes()
        ).hexdigest()
        self.model_id = self.artifact_path.stem
        self.runtime_identity = f"numpy@{np.__version__}"
        data = np.load(self.artifact_path)
        self.policy_weights = np.asarray(data["policy_weights"], dtype=np.float32)
        self.policy_bias = np.asarray(data["policy_bias"], dtype=np.float32)
        self.value_weights = np.asarray(data["value_weights"], dtype=np.float32)
        self.value_bias = float(np.asarray(data["value_bias"]).reshape(-1)[0])
        if self.policy_weights.shape != (FEATURE_COUNT, ACTION_COUNT):
            raise ValueError("incompatible policy weight shape")
        if self.value_weights.shape != (FEATURE_COUNT,):
            raise ValueError("incompatible value weight shape")
        if manifest_path is not None:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if manifest.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
                raise ValueError("feature schema mismatch")
            if manifest.get("actionSchemaVersion") != ACTION_SCHEMA_VERSION:
                raise ValueError("action schema mismatch")
            if manifest.get("artifactSha256") != self.artifact_sha256:
                raise ValueError("model artifact checksum mismatch")
            self.id = str(manifest.get("pluginId", self.id))
            self.version = str(manifest.get("version", self.version))
            self.model_id = str(manifest.get("modelId", self.model_id))

    def infer(self, observation: np.ndarray, legal_mask: tuple[int, ...]) -> ModelOutput:
        vector = np.asarray(observation, dtype=np.float32).reshape(FEATURE_COUNT)
        if len(legal_mask) != ACTION_COUNT:
            raise ValueError("legal mask length mismatch")
        logits = vector @ self.policy_weights + self.policy_bias
        logits = np.asarray(logits, dtype=np.float32)
        for index, legal in enumerate(legal_mask):
            if not legal:
                logits[index] = -1e9
        value = float(np.tanh(vector @ self.value_weights + self.value_bias))
        return ModelOutput(logits, value)


def _arrays(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = list(rows)
    if not values:
        return (
            np.zeros((0, FEATURE_COUNT), dtype=np.float32),
            np.zeros((0, ACTION_COUNT), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    features = np.asarray([row["features"] for row in values], dtype=np.float32)
    legal = np.asarray([row["legalMask"] for row in values], dtype=np.float32)
    targets = np.asarray([int(row["targetAction"]) for row in values], dtype=np.int64)
    value_targets = np.asarray([float(row.get("valueTarget", 0.0)) for row in values], dtype=np.float32)
    if features.shape[1:] != (FEATURE_COUNT,):
        raise ValueError("dataset feature contract mismatch")
    if legal.shape[1:] != (ACTION_COUNT,):
        raise ValueError("dataset action contract mismatch")
    if np.any(targets < 0) or np.any(targets >= ACTION_COUNT):
        raise ValueError("dataset contains action outside stable action space")
    if np.any(legal[np.arange(len(targets)), targets] <= 0):
        raise ValueError("dataset target action is not legal")
    return features, legal, targets, value_targets


def evaluate_rows(model: Any, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    if not items:
        return {
            "exampleCount": 0,
            "top1Accuracy": 0.0,
            "top3Accuracy": 0.0,
            "weightedTop1Accuracy": 0.0,
            "illegalActionRate": 0.0,
            "crossEntropy": None,
            "weightedCrossEntropy": None,
            "valueMae": None,
            "preferencePairCount": 0,
            "preferencePairAccuracy": None,
        }
    top1 = 0
    top3 = 0
    illegal = 0
    weighted_top1 = 0.0
    total_weight = 0.0
    losses: list[float] = []
    weighted_losses = 0.0
    value_errors: list[float] = []
    pair_count = 0
    pair_correct = 0
    for row in items:
        obs = np.asarray(row["features"], dtype=np.float32)
        mask = tuple(int(v) for v in row["legalMask"])
        target = int(row["targetAction"])
        weight = max(1e-6, float(row.get("sampleWeight", 1.0)))
        output = model.infer(obs, mask)
        ranked = rank_legal_actions(
            output,
            tuple(index for index, allowed in enumerate(mask) if allowed),
        )
        order = tuple(identifier for identifier, _score in ranked)
        prediction = order[0]
        hit = prediction == target
        top1 += hit
        top3 += target in set(order[:3])
        illegal += not bool(mask[prediction])
        weighted_top1 += weight * float(hit)
        total_weight += weight
        valid_logits = np.full(
            (1, ACTION_COUNT),
            -1.0e9,
            dtype=np.float32,
        )
        for identifier, score in ranked:
            valid_logits[0, identifier] = score
        probs = _softmax(valid_logits)[0]
        loss = float(-np.log(max(1e-12, probs[target])))
        losses.append(loss)
        weighted_losses += weight * loss
        value_errors.append(abs(float(output.value) - float(row.get("valueTarget", 0.0))))
        pair = row.get("preferencePair") or {}
        actual = pair.get("actualAction")
        preferred = pair.get("preferredAction", target)
        if actual is not None and int(actual) != int(preferred) and 0 <= int(actual) < ACTION_COUNT:
            pair_count += 1
            pair_correct += float(output.policy_logits[int(preferred)]) > float(output.policy_logits[int(actual)])
    n = len(items)
    return {
        "exampleCount": n,
        "top1Accuracy": top1 / n,
        "top3Accuracy": top3 / n,
        "weightedTop1Accuracy": weighted_top1 / max(total_weight, 1e-6),
        "illegalActionRate": illegal / n,
        "crossEntropy": float(np.mean(losses)),
        "weightedCrossEntropy": weighted_losses / max(total_weight, 1e-6),
        "valueMae": float(np.mean(value_errors)),
        "preferencePairCount": pair_count,
        "preferencePairAccuracy": None if pair_count == 0 else pair_correct / pair_count,
    }


def train_numpy_linear(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    epochs: int = 40,
    learning_rate: float = 0.03,
    l2: float = 1e-4,
    batch_size: int = 64,
    seed: int = 1,
    preference_loss_weight: float = 0.25,
) -> TrainResult:
    rows = _read_jsonl(dataset_path)
    train_rows = [row for row in rows if row.get("split", "train") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not train_rows:
        train_rows = rows
    x, legal, y, value_y = _arrays(train_rows)
    if len(x) == 0:
        raise ValueError("dataset contains no trainable examples")
    weights = np.asarray([max(1e-6, float(row.get("sampleWeight", 1.0))) for row in train_rows], dtype=np.float32)
    actual_actions = np.asarray([
        int((row.get("preferencePair") or {}).get("actualAction", -1))
        if (row.get("preferencePair") or {}).get("actualAction") is not None else -1
        for row in train_rows
    ], dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    policy_weights = rng.normal(0.0, 0.005, size=(FEATURE_COUNT, ACTION_COUNT)).astype(np.float32)
    policy_bias = np.zeros(ACTION_COUNT, dtype=np.float32)
    value_weights = np.zeros(FEATURE_COUNT, dtype=np.float32)
    value_bias = np.float32(0.0)

    batch_size = max(1, min(int(batch_size), len(x)))
    for _epoch in range(max(1, int(epochs))):
        indices = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            idx = indices[start : start + batch_size]
            xb = x[idx]
            mb = legal[idx]
            yb = y[idx]
            vb = value_y[idx]

            batch_weights = weights[idx]
            weight_sum = max(float(batch_weights.sum()), 1e-6)
            logits = xb @ policy_weights + policy_bias
            logits = np.where(mb > 0, logits, -1e9)
            probs = _softmax(logits)
            grad_logits = probs.copy()
            grad_logits[np.arange(len(idx)), yb] -= 1.0
            grad_logits *= (batch_weights / weight_sum).reshape(-1, 1)

            if preference_loss_weight > 0:
                actual = actual_actions[idx]
                valid_pair = (actual >= 0) & (actual < ACTION_COUNT) & (actual != yb)
                for row_index in np.flatnonzero(valid_pair):
                    preferred_index = int(yb[row_index]); actual_index = int(actual[row_index])
                    difference = float(logits[row_index, preferred_index] - logits[row_index, actual_index])
                    derivative = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, difference))))
                    scale = float(preference_loss_weight) * float(batch_weights[row_index]) / weight_sum
                    grad_logits[row_index, preferred_index] -= scale * derivative
                    grad_logits[row_index, actual_index] += scale * derivative

            grad_w = xb.T @ grad_logits + l2 * policy_weights
            grad_b = np.sum(grad_logits, axis=0)
            policy_weights -= learning_rate * grad_w.astype(np.float32)
            policy_bias -= learning_rate * grad_b.astype(np.float32)

            prediction = np.tanh(xb @ value_weights + value_bias)
            grad_value = 2.0 * (prediction - vb) * (1.0 - prediction * prediction) * batch_weights / weight_sum
            value_weights -= learning_rate * (xb.T @ grad_value + l2 * value_weights).astype(np.float32)
            value_bias -= np.float32(learning_rate * np.sum(grad_value))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "model.tmp.npz"
    np.savez_compressed(
        temporary,
        policy_weights=policy_weights,
        policy_bias=policy_bias,
        value_weights=value_weights,
        value_bias=np.asarray([value_bias], dtype=np.float32),
    )
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    model_id = f"mdl_{digest[:20]}"
    artifact = output_dir / f"{model_id}.npz"
    temporary.replace(artifact)

    model = NumpyLinearPolicyValueModel(artifact)
    metrics = {
        "train": evaluate_rows(model, train_rows),
        "validation": evaluate_rows(model, validation_rows),
        "test": evaluate_rows(model, test_rows),
    }
    manifest = {
        "schemaVersion": 1,
        "modelId": model_id,
        "pluginId": NumpyLinearPolicyValueModel.id,
        "version": NumpyLinearPolicyValueModel.version,
        "datasetId": dataset_id,
        "artifactSha256": digest,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "actionSchemaVersion": ACTION_SCHEMA_VERSION,
        "training": {
            "backend": "numpy",
            "epochs": int(epochs),
            "learningRate": float(learning_rate),
            "l2": float(l2),
            "batchSize": int(batch_size),
            "seed": int(seed),
            "preferenceLossWeight": float(preference_loss_weight),
        },
        "metrics": metrics,
    }
    manifest_path = output_dir / f"{model_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return TrainResult(model_id, NumpyLinearPolicyValueModel.id, artifact, manifest_path, manifest, metrics)


class TorchScriptPolicyValueModel:
    """Small residual TorchScript policy/value model with authoritative masking."""

    id = "torch_residual_policy_value_v1"
    version = "1.0.0"

    def __init__(self, artifact_path: str | Path, manifest_path: str | Path | None = None):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install the optional dependency with: pip install "baloot-platform[ml]"') from exc
        self.artifact_path = Path(artifact_path)
        self._torch = torch
        self.artifact_sha256 = hashlib.sha256(
            self.artifact_path.read_bytes()
        ).hexdigest()
        self.model_id = self.artifact_path.stem
        self.runtime_identity = f"torch@{torch.__version__}"
        self.model = torch.jit.load(str(self.artifact_path), map_location="cpu")
        self.model.eval()
        if manifest_path is not None:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if manifest.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
                raise ValueError("feature schema mismatch")
            if manifest.get("actionSchemaVersion") != ACTION_SCHEMA_VERSION:
                raise ValueError("action schema mismatch")
            if manifest.get("artifactSha256") != self.artifact_sha256:
                raise ValueError("model artifact checksum mismatch")
            self.id = str(manifest.get("pluginId", self.id))
            self.version = str(manifest.get("version", self.version))
            self.model_id = str(manifest.get("modelId", self.model_id))

    def infer(self, observation: np.ndarray, legal_mask: tuple[int, ...]) -> ModelOutput:
        torch = self._torch
        vector = np.asarray(observation, dtype=np.float32).reshape(1, FEATURE_COUNT)
        mask = np.asarray(legal_mask, dtype=np.float32).reshape(1, ACTION_COUNT)
        with torch.no_grad():
            logits, value = self.model(torch.from_numpy(vector), torch.from_numpy(mask))
        return ModelOutput(
            logits.detach().cpu().numpy().reshape(ACTION_COUNT).astype(np.float32),
            float(value.detach().cpu().numpy().reshape(-1)[0]),
        )


def train_torch_policy_value(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    l2: float = 1e-5,
    batch_size: int = 64,
    seed: int = 1,
    hidden_size: int = 256,
    residual_blocks: int = 2,
    preference_loss_weight: float = 0.25,
) -> TrainResult:
    """Train a compact CPU-friendly residual policy/value network.

    The network always receives the legal mask and applies it inside the
    exported TorchScript graph. This makes illegal-action prevention part of
    the serving artifact rather than a UI convention.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('PyTorch is unavailable; select the NumPy backend or install baloot-platform[ml]') from exc

    rows = _read_jsonl(dataset_path)
    train_rows = [row for row in rows if row.get("split", "train") == "train"] or rows
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    x, legal, y, value_y = _arrays(train_rows)
    if len(x) == 0:
        raise ValueError("dataset contains no trainable examples")
    weights = np.asarray([max(1e-6, float(row.get("sampleWeight", 1.0))) for row in train_rows], dtype=np.float32)
    actual_actions = np.asarray([
        int((row.get("preferencePair") or {}).get("actualAction", -1))
        if (row.get("preferencePair") or {}).get("actualAction") is not None else -1
        for row in train_rows
    ], dtype=np.int64)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    class ResidualBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
            self.activation = nn.ReLU()

        def forward(self, value):
            return self.activation(value + self.net(value))

    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.Linear(FEATURE_COUNT, hidden_size), nn.ReLU())
            self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(max(1, residual_blocks))])
            self.policy = nn.Linear(hidden_size, ACTION_COUNT)
            self.value = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1), nn.Tanh())

        def forward(self, features, legal_mask):
            hidden = self.input(features)
            for block in self.blocks:
                hidden = block(hidden)
            logits = self.policy(hidden)
            logits = torch.where(legal_mask > 0.0, logits, torch.full_like(logits, -1.0e9))
            return logits, self.value(hidden)

    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(l2))
    features = torch.from_numpy(x)
    masks = torch.from_numpy(legal)
    targets = torch.from_numpy(y)
    values = torch.from_numpy(value_y).reshape(-1, 1)
    sample_weights = torch.from_numpy(weights)
    actual_targets = torch.from_numpy(actual_actions)
    batch_size = max(1, min(int(batch_size), len(features)))
    generator = torch.Generator().manual_seed(int(seed))
    model.train()
    for _epoch in range(max(1, int(epochs))):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            idx = order[start:start + batch_size]
            logits, predicted_values = model(features[idx], masks[idx])
            policy_loss = nn.functional.cross_entropy(logits, targets[idx], reduction="none")
            value_loss = (predicted_values.reshape(-1) - values[idx].reshape(-1)).pow(2)
            batch_weights = sample_weights[idx]
            per_row_loss = policy_loss + 0.35 * value_loss
            if preference_loss_weight > 0:
                actual = actual_targets[idx]
                valid_pair = (actual >= 0) & (actual < ACTION_COUNT) & (actual != targets[idx])
                if bool(valid_pair.any()):
                    preferred_logits = logits[torch.arange(len(idx)), targets[idx]]
                    safe_actual = actual.clamp(min=0, max=ACTION_COUNT - 1)
                    actual_logits = logits[torch.arange(len(idx)), safe_actual]
                    pair_loss = nn.functional.softplus(-(preferred_logits - actual_logits))
                    per_row_loss = per_row_loss + float(preference_loss_weight) * pair_loss * valid_pair.float()
            loss = (per_row_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "torch-model.tmp.pt"
    example_x = torch.zeros((1, FEATURE_COUNT), dtype=torch.float32)
    example_m = torch.ones((1, ACTION_COUNT), dtype=torch.float32)
    traced = torch.jit.trace(model, (example_x, example_m))
    traced.save(str(temporary))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    model_id = f"mdl_{digest[:20]}"
    artifact = output_dir / f"{model_id}.pt"
    temporary.replace(artifact)
    loaded = TorchScriptPolicyValueModel(artifact)
    metrics = {
        "train": evaluate_rows(loaded, train_rows),
        "validation": evaluate_rows(loaded, validation_rows),
        "test": evaluate_rows(loaded, test_rows),
    }
    manifest = {
        "schemaVersion": 1,
        "modelId": model_id,
        "pluginId": TorchScriptPolicyValueModel.id,
        "version": TorchScriptPolicyValueModel.version,
        "datasetId": dataset_id,
        "artifactSha256": digest,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "actionSchemaVersion": ACTION_SCHEMA_VERSION,
        "training": {
            "backend": "torch",
            "epochs": int(epochs),
            "learningRate": float(learning_rate),
            "l2": float(l2),
            "batchSize": int(batch_size),
            "seed": int(seed),
            "hiddenSize": int(hidden_size),
            "residualBlocks": int(residual_blocks),
            "preferenceLossWeight": float(preference_loss_weight),
        },
        "metrics": metrics,
    }
    manifest_path = output_dir / f"{model_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return TrainResult(model_id, TorchScriptPolicyValueModel.id, artifact, manifest_path, manifest, metrics)
