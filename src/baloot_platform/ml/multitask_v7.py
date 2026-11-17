"""V7 observer-safe multi-task ML contracts and a reproducible NumPy baseline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from .features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION

BELIEF_SIZE = 32 * 4
INTENTION_SIZE = 16
MODEL_SCHEMA_VERSION = "9.0.0"


@dataclass(frozen=True, slots=True)
class MultiTaskOutput:
    policy_logits: np.ndarray
    belief_logits: np.ndarray
    contract_success_logit: float
    round_value: float
    match_value_logit: float
    intention_logits: np.ndarray
    uncertainty: float

    def legal_policy(self, legal_mask: tuple[int, ...]) -> np.ndarray:
        logits=np.asarray(self.policy_logits,dtype=np.float64).copy()
        if logits.shape!=(ACTION_COUNT,) or len(legal_mask)!=ACTION_COUNT:raise ValueError('action contract mismatch')
        for i,legal in enumerate(legal_mask):
            if not legal:logits[i]=-1e30
        maximum=float(np.max(logits)); exp=np.exp(logits-maximum); exp=np.where(np.asarray(legal_mask,dtype=bool),exp,0.0)
        total=float(exp.sum())
        if total<=0:raise ValueError('legal mask contains no action')
        return (exp/total).astype(np.float32)

    def belief_probabilities(self)->np.ndarray:
        logits=np.asarray(self.belief_logits,dtype=np.float64).reshape(32,4)
        logits-=logits.max(axis=1,keepdims=True); values=np.exp(logits); values/=values.sum(axis=1,keepdims=True)
        return values.astype(np.float32)


@runtime_checkable
class MultiTaskModelPlugin(Protocol):
    id:str; version:str
    def infer_multitask(self,observation:np.ndarray,legal_mask:tuple[int,...])->MultiTaskOutput:...


class NumpyMultiTaskLinearModel:
    """Small deterministic baseline suitable for tests, distillation, and edge deployment.

    It is intentionally not claimed to be a strong Baloot model.  The same
    interface is used by Torch/ONNX models after training and promotion.
    """
    id="numpy_multitask_linear_v1"; version="1.0.0"
    def __init__(self,seed:int=7001):
        rng=np.random.default_rng(seed); scale=1.0/np.sqrt(FEATURE_COUNT)
        self.policy=rng.normal(0,scale,(FEATURE_COUNT,ACTION_COUNT)).astype(np.float32)
        self.belief=rng.normal(0,scale,(FEATURE_COUNT,BELIEF_SIZE)).astype(np.float32)
        self.values=rng.normal(0,scale,(FEATURE_COUNT,3)).astype(np.float32)
        self.intentions=rng.normal(0,scale,(FEATURE_COUNT,INTENTION_SIZE)).astype(np.float32)
    def infer_multitask(self,observation:np.ndarray,legal_mask:tuple[int,...])->MultiTaskOutput:
        x=np.asarray(observation,dtype=np.float32).reshape(FEATURE_COUNT)
        logits=x@self.policy
        for i,legal in enumerate(legal_mask):
            if not legal:logits[i]=-1e9
        values=x@self.values; intent=x@self.intentions; belief=x@self.belief
        uncertainty=float(1.0/(1.0+np.linalg.norm(x)))
        return MultiTaskOutput(logits,belief,float(values[0]),float(np.tanh(values[1])),float(values[2]),intent,uncertainty)


def binary_probability(logit:float)->float:
    if logit>=0:return float(1.0/(1.0+np.exp(-logit)))
    z=np.exp(logit);return float(z/(1.0+z))


def model_manifest(*,model_id:str,version:str,artifact:Path,training_dataset_id:str,metrics:dict[str,Any],heads:tuple[str,...]=( "policy","belief","contract_success","round_value","match_value","intentions","uncertainty"))->dict[str,Any]:
    body=artifact.read_bytes()
    return {"schemaVersion":MODEL_SCHEMA_VERSION,"id":model_id,"version":version,"sha256":hashlib.sha256(body).hexdigest(),"featureSchemaVersion":FEATURE_SCHEMA_VERSION,"actionSchemaVersion":ACTION_SCHEMA_VERSION,"featureCount":FEATURE_COUNT,"actionCount":ACTION_COUNT,"beliefSize":BELIEF_SIZE,"intentionSize":INTENTION_SIZE,"heads":list(heads),"trainingDatasetId":training_dataset_id,"metrics":dict(metrics)}


def validate_manifest(manifest:dict[str,Any])->None:
    required={"policy","belief","contract_success","round_value","match_value","intentions","uncertainty"}
    if manifest.get('featureSchemaVersion')!=FEATURE_SCHEMA_VERSION:raise ValueError('incompatible feature schema')
    if manifest.get('actionSchemaVersion')!=ACTION_SCHEMA_VERSION:raise ValueError('incompatible action schema')
    if not required.issubset(set(manifest.get('heads',()))):raise ValueError('required V7 model heads are missing')
