"""Configurable project declaration strategy.

The policy cannot suppress a declaration when the active rules profile makes
it mandatory.  Optional-concealment modes are available only for training,
research, or rule profiles that explicitly allow them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random


class ProjectPolicy(str, Enum):
    RULE_COMPLIANT = "rule_compliant"
    MAXIMIZE_SCORE = "maximize_score"
    INFORMATION_AWARE = "information_aware"
    DECEPTIVE_OPTIONAL = "deceptive_optional"


@dataclass(frozen=True)
class ProjectContext:
    owner: str
    project_id: str
    project_points: int
    estimated_win_value: float
    reveal_information_cost: float
    declaration_required: bool
    declaration_allowed: bool
    team_score: int = 0
    opponent_score: int = 0
    target_score: int = 152


@dataclass(frozen=True)
class ProjectPolicyConfig:
    plugin_id: str = "project_information_aware_v4"
    policy: ProjectPolicy = ProjectPolicy.RULE_COMPLIANT
    minimum_optional_net_value: float = 0.0
    deception_probability: float = 0.0
    mistake_probability: float = 0.0
    deterministic_seed: int = 1


@dataclass(frozen=True)
class ProjectDecision:
    declare: bool
    net_value: float
    reason_codes: tuple[str, ...]
    plugin_id: str


def decide_project(context: ProjectContext, config: ProjectPolicyConfig | None = None) -> ProjectDecision:
    config = config or ProjectPolicyConfig()
    if not context.declaration_allowed:
        return ProjectDecision(False, 0.0, ("declaration_not_allowed_by_rules",), config.plugin_id)
    if context.declaration_required:
        return ProjectDecision(True, context.estimated_win_value, ("mandatory_declaration",), config.plugin_id)
    if config.policy is ProjectPolicy.RULE_COMPLIANT:
        return ProjectDecision(True, context.estimated_win_value, ("default_optional_declaration",), config.plugin_id)

    net = context.estimated_win_value + context.project_points / 100.0 - context.reveal_information_cost
    if config.policy is ProjectPolicy.MAXIMIZE_SCORE:
        declare = context.project_points > 0
    else:
        declare = net >= config.minimum_optional_net_value

    rng = Random(config.deterministic_seed + sum(map(ord, context.owner + context.project_id)))
    if config.policy is ProjectPolicy.DECEPTIVE_OPTIONAL and rng.random() < config.deception_probability:
        declare = not declare
    if rng.random() < config.mistake_probability:
        declare = not declare

    reasons = [f"optional_net_value={net:.3f}"]
    reasons.append("declare" if declare else "conceal_to_reduce_information_cost")
    return ProjectDecision(declare, net, tuple(reasons), config.plugin_id)
