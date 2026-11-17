"""Diversity-aware active-learning selection for Baloot decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True,slots=True)
class ActiveLearningCandidate:
    decision_id:str
    features:tuple[float,...]
    uncertainty:float
    impact:float
    disagreement:float
    novelty:float=0.0
    metadata:dict[str,Any]|None=None


@dataclass(frozen=True,slots=True)
class SelectedCandidate:
    decision_id:str
    priority:float
    uncertainty:float
    impact:float
    disagreement:float
    diversity_distance:float

    def to_dict(self)->dict[str,Any]:
        return {"decisionId":self.decision_id,"priority":self.priority,"uncertainty":self.uncertainty,"impact":self.impact,"disagreement":self.disagreement,"diversityDistance":self.diversity_distance}


def _unit(value:np.ndarray)->np.ndarray:
    norm=float(np.linalg.norm(value));return value/norm if norm>1e-12 else value


def select_for_labeling(candidates:Iterable[ActiveLearningCandidate],*,limit:int=100,uncertainty_weight:float=0.35,impact_weight:float=0.30,disagreement_weight:float=0.20,diversity_weight:float=0.15)->tuple[SelectedCandidate,...]:
    pool=list(candidates)
    if not pool or limit<=0:return ()
    vectors={item.decision_id:_unit(np.asarray(item.features,dtype=np.float64)) for item in pool}
    selected:list[SelectedCandidate]=[]; chosen:set[str]=set()
    while len(selected)<min(limit,len(pool)):
        best=None
        for item in pool:
            if item.decision_id in chosen:continue
            if not selected:distance=1.0
            else:distance=min(1.0-float(np.dot(vectors[item.decision_id],vectors[other.decision_id])) for other in selected)
            priority=(uncertainty_weight*max(0,min(1,item.uncertainty))+impact_weight*max(0,min(1,item.impact))+disagreement_weight*max(0,min(1,item.disagreement))+diversity_weight*max(0,min(1,distance)))
            row=SelectedCandidate(item.decision_id,priority,item.uncertainty,item.impact,item.disagreement,distance)
            if best is None or (row.priority,row.decision_id)>(best.priority,best.decision_id):best=row
        assert best is not None;selected.append(best);chosen.add(best.decision_id)
    return tuple(selected)
