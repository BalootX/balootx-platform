from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from baloot_platform.ml.actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from baloot_platform.ml.features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION
from baloot_platform.ml.models import (
    ModelOutput,
    UniformLegalModel,
    model_runtime_identity,
    rank_legal_actions,
)
from baloot_platform.learning.training import (
    NumpyLinearPolicyValueModel,
    train_numpy_linear,
)
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore


def test_v21_model_ranking_uses_only_legal_actions_and_stable_ties():
    logits = np.zeros(ACTION_COUNT, dtype=np.float32)
    logits[75] = 1.0e9
    ranked = rank_legal_actions(ModelOutput(logits, 0.0), (8, 3))
    assert ranked == ((3, 0.0), (8, 0.0))


@pytest.mark.parametrize(
    ("logits", "value", "message"),
    (
        (
            np.zeros(ACTION_COUNT - 1, dtype=np.float32),
            0.0,
            "shape",
        ),
        (
            np.full(ACTION_COUNT, np.nan, dtype=np.float32),
            0.0,
            "scores must be finite",
        ),
        (
            np.zeros(ACTION_COUNT, dtype=np.float32),
            float("nan"),
            "value output must be finite",
        ),
    ),
)
def test_v21_model_ranking_rejects_invalid_output(logits, value, message):
    with pytest.raises(ValueError, match=message):
        rank_legal_actions(ModelOutput(logits, value), (1,))


def test_v21_model_runtime_identity_is_structured_and_path_free():
    identity = model_runtime_identity(UniformLegalModel())
    assert identity == {
        "modelId": "uniform_legal_v1",
        "pluginId": "uniform_legal_v1",
        "pluginVersion": "1.0.0",
        "artifactSha256": None,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "actionSchemaVersion": ACTION_SCHEMA_VERSION,
        "runtime": identity["runtime"],
    }
    assert identity["runtime"].startswith("numpy@")
    assert "\\" not in identity["runtime"]
    assert "/" not in identity["runtime"]


def test_v21_loaded_model_identity_is_bound_to_manifest(tmp_path):
    artifact = tmp_path / "materialized.npz"
    np.savez_compressed(
        artifact,
        policy_weights=np.zeros(
            (FEATURE_COUNT, ACTION_COUNT),
            dtype=np.float32,
        ),
        policy_bias=np.zeros(ACTION_COUNT, dtype=np.float32),
        value_weights=np.zeros(FEATURE_COUNT, dtype=np.float32),
        value_bias=np.zeros(1, dtype=np.float32),
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "materialized.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "modelId": "mdl_governed",
                "pluginId": NumpyLinearPolicyValueModel.id,
                "version": NumpyLinearPolicyValueModel.version,
                "artifactSha256": digest,
                "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
                "actionSchemaVersion": ACTION_SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )
    identity = model_runtime_identity(
        NumpyLinearPolicyValueModel(artifact, manifest)
    )
    assert identity["modelId"] == "mdl_governed"
    assert identity["artifactSha256"] == digest
    assert identity["runtime"].startswith("numpy@")


def test_preference_training_increases_preferred_action_margin(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    legal_mask = [int(index in (0, 1, 2)) for index in range(ACTION_COUNT)]
    dataset.write_text(
        json.dumps({
            "features": [0.0] * FEATURE_COUNT,
            "legalMask": legal_mask,
            "targetAction": 1,
            "preferencePair": {"preferredAction": 1, "actualAction": 2},
        }) + "\n",
        encoding="utf-8",
    )
    models = []
    for weight in (0.0, 1.0):
        result = train_numpy_linear(
            dataset,
            tmp_path / f"model-{weight}",
            dataset_id="preference-regression",
            epochs=1,
            learning_rate=0.1,
            preference_loss_weight=weight,
        )
        models.append(NumpyLinearPolicyValueModel(result.artifact_path, result.manifest_path))

    ordinary, preferred = [
        model.infer(np.zeros(FEATURE_COUNT, dtype=np.float32), tuple(legal_mask))
        for model in models
    ]
    ordinary_margin = ordinary.policy_logits[1] - ordinary.policy_logits[2]
    preferred_margin = preferred.policy_logits[1] - preferred.policy_logits[2]
    assert preferred_margin > ordinary_margin
    assert preferred.policy_logits[0] == ordinary.policy_logits[0]
    assert np.isfinite(preferred.policy_logits).all()
    assert rank_legal_actions(preferred, (0, 1, 2))[0][0] == 1


def _register_candidate(store, model_id):
    return store.register_model(
        model_id=model_id,
        plugin_id=NumpyLinearPolicyValueModel.id,
        version=NumpyLinearPolicyValueModel.version,
        artifact_path=f"{model_id}.npz",
        manifest={"modelId": model_id},
        dataset_id="held-out-v21",
        metrics={
            "test": {
                "exampleCount": 20,
                "top3Accuracy": 0.60,
                "illegalActionRate": 0.0,
            }
        },
    )


def _passing_shadow_evidence():
    return {
        "decisionCount": 10_000,
        "boardSets": 20,
        "scoreRate": 0.53,
        "scoreRateLowerBound": 0.51,
        "calibrationError": 0.07,
        "p95LatencyMs": 49.0,
        "oodRecall": 0.76,
        "legalViolationRate": 0.0,
    }


def test_v21_model_promotion_requires_shadow_and_rollback_evidence(
    root,
    tmp_path,
):
    store = LearningStore(tmp_path / "learning.sqlite3")
    service = LearningService(
        root,
        tmp_path / "learning",
        store=store,
    )
    _register_candidate(store, "mdl_first")
    with pytest.raises(ValueError, match="shadow promotion evidence"):
        service.promote_model("mdl_first")
    non_finite = _passing_shadow_evidence()
    non_finite["calibrationError"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        service.promote_model(
            "mdl_first",
            shadow_evidence=non_finite,
            rollback_model_id="deterministic_heuristic_fallback_v1",
        )
    weak = _passing_shadow_evidence()
    weak["scoreRateLowerBound"] = 0.49
    with pytest.raises(ValueError, match="scoreRateLowerBound"):
        service.promote_model(
            "mdl_first",
            shadow_evidence=weak,
            rollback_model_id="deterministic_heuristic_fallback_v1",
        )
    promoted = service.promote_model(
        "mdl_first",
        shadow_evidence=_passing_shadow_evidence(),
        rollback_model_id="deterministic_heuristic_fallback_v1",
    )
    assert promoted["status"] == "production"
    proof = promoted["metrics"]["promotionEvidence"]
    assert proof["candidateModelId"] == "mdl_first"
    assert len(proof["sha256"]) == 64

    _register_candidate(store, "mdl_second")
    with pytest.raises(ValueError, match="rollback model"):
        service.promote_model(
            "mdl_second",
            shadow_evidence=_passing_shadow_evidence(),
            rollback_model_id="deterministic_heuristic_fallback_v1",
        )
    second = service.promote_model(
        "mdl_second",
        shadow_evidence=_passing_shadow_evidence(),
        rollback_model_id="mdl_first",
    )
    assert second["status"] == "production"
    assert store.get_model("mdl_first")["status"] == "validated"


def test_governed_dataset_training_fails_closed_after_consent_withdrawal(root, tmp_path):
    lineage = [{
        "candidateId": "TRN-1",
        "caseId": "HITL-1",
        "decisionId": "DEC-1",
        "consentSubjects": ["REVIEWER-1"],
    }]

    class DatasetStore:
        def get_dataset(self, dataset_id):
            return {
                "id": dataset_id,
                "artifactPath": str(tmp_path / "never-read.jsonl"),
                "manifest": {"metadata": {"governedHitlLineage": lineage}},
            }

    class GovernanceStore:
        def get_candidate(self, candidate_id):
            return {"candidateId": candidate_id, "status": "consent_withdrawn"}

        def get_consent(self, subject_id):
            return {"subjectId": subject_id, "allowTraining": False, "revokedAt": "2026-08-03T00:00:00+00:00"}

    service = LearningService(root, tmp_path / "learning-lineage", store=DatasetStore())
    service.hitl_store = GovernanceStore()
    with pytest.raises(ValueError, match="withdrawn or ineligible"):
        service.train_model("DS-GOVERNED", backend="numpy")
