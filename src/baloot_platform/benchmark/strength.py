from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _key(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mean_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = statistics.fmean(values)
    if len(values) == 1:
        return (mean, mean)
    margin = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
    return (mean - margin, mean + margin)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model_id: str
    examples: int
    policy_accuracy: float
    top3_accuracy: float
    expected_regret: float
    match_value_loss: float
    brier_score: float
    illegal_action_rate: float
    archetypes: Mapping[str, Mapping[str, float]]
    promotion_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1, "modelId": self.model_id, "examples": self.examples,
            "policyAccuracy": self.policy_accuracy, "top3Accuracy": self.top3_accuracy,
            "expectedRegret": self.expected_regret, "matchValueLoss": self.match_value_loss,
            "brierScore": self.brier_score, "illegalActionRate": self.illegal_action_rate,
            "archetypes": {k: dict(v) for k, v in self.archetypes.items()},
            "promotionEligible": self.promotion_eligible,
        }
        value["resultHash"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return value


@dataclass(frozen=True, slots=True)
class PermanentHoldoutResult:
    model_id: str
    dataset_id: str
    dataset_sha256: str
    examples: int
    policy_accuracy: float
    top3_accuracy: float
    expected_regret: float
    brier_score: float
    expected_calibration_error: float
    illegal_action_rate: float
    exact_labels: int
    adjudicated_labels: int
    accuracy_interval: tuple[float, float]
    regret_interval: tuple[float, float]
    archetypes: Mapping[str, Mapping[str, float]]
    search_completion: Mapping[str, int]
    workload: Mapping[str, float]
    proof_boundary: str = "permanent_holdout_local_evidence"

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 2,
            "modelId": self.model_id,
            "datasetId": self.dataset_id,
            "datasetSha256": self.dataset_sha256,
            "examples": self.examples,
            "policyAccuracy": self.policy_accuracy,
            "top3Accuracy": self.top3_accuracy,
            "expectedRegret": self.expected_regret,
            "brierScore": self.brier_score,
            "expectedCalibrationError": self.expected_calibration_error,
            "illegalActionRate": self.illegal_action_rate,
            "labelProvenance": {
                "exact": self.exact_labels,
                "independentlyAdjudicated": self.adjudicated_labels,
            },
            "confidence95": {
                "policyAccuracy": list(self.accuracy_interval),
                "expectedRegret": list(self.regret_interval),
            },
            "archetypes": {
                key: dict(item)
                for key, item in sorted(self.archetypes.items())
            },
            "searchCompletion": dict(sorted(self.search_completion.items())),
            "workload": dict(sorted(self.workload.items())),
            "groupLeakageDetected": False,
            "automaticPromotion": False,
            "proofBoundary": self.proof_boundary,
        }
        value["resultHash"] = _hash(value)
        return value


@dataclass(frozen=True, slots=True)
class PairedMatchResult:
    candidate_id: str
    baseline_id: str
    pairs: int
    wins: int
    losses: int
    ties: int
    decisive_win_rate: float
    win_rate_interval: tuple[float, float]
    mean_paired_margin: float
    margin_interval: tuple[float, float]
    regret_improvement: float
    promotion_target_met: bool

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "candidateId": self.candidate_id,
            "baselineId": self.baseline_id,
            "pairs": self.pairs,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "decisiveWinRate": self.decisive_win_rate,
            "confidence95": {
                "decisiveWinRate": list(self.win_rate_interval),
                "pairedMargin": list(self.margin_interval),
            },
            "meanPairedMargin": self.mean_paired_margin,
            "regretImprovement": self.regret_improvement,
            "promotionTargetMet": self.promotion_target_met,
            "automaticPromotion": False,
            "proofBoundary": "symmetric_paired_local_evidence",
        }
        value["resultHash"] = _hash(value)
        return value


class StrengthBenchmark:
    """Leakage-safe, policy-agnostic benchmark contract.

    It evaluates supplied predictions but never promotes a production model.
    Promotion is a separate governed release decision.
    """

    def __init__(self, profile: Mapping[str, Any]):
        self.profile = dict(profile)

    def evaluate(self, model_id: str, examples: Iterable[Mapping[str, Any]]) -> BenchmarkResult:
        rows = list(examples)
        if not rows:
            raise ValueError("benchmark requires examples")
        correct = top3 = illegal = 0
        regret_total = value_loss = brier = 0.0
        groups: dict[str, list[tuple[bool, float]]] = {}
        for row in rows:
            legal = {_key(x) for x in row.get("legalActions", [])}
            truth = _key(row.get("referenceAction"))
            prediction = _key(row.get("predictedAction"))
            ranked = [_key(x) for x in row.get("rankedActions", [])]
            probability = max(0.0, min(1.0, float(row.get("referenceProbability", 0.0))))
            y = 1.0 if prediction == truth else 0.0
            correct += int(y)
            top3 += int(truth in ranked[:3] or prediction == truth)
            illegal += int(prediction not in legal)
            regret = max(0.0, float(row.get("predictedRegret", 0.0)))
            loss = max(0.0, float(row.get("matchValueLoss", regret)))
            regret_total += regret
            value_loss += loss
            brier += (probability - y) ** 2
            archetype = str(row.get("archetype", "other"))
            groups.setdefault(archetype, []).append((bool(y), regret))
        n = len(rows)
        archetypes = {
            name: {"examples": float(len(items)), "accuracy": sum(int(x[0]) for x in items) / len(items),
                   "expectedRegret": sum(x[1] for x in items) / len(items)}
            for name, items in sorted(groups.items())
        }
        thresholds = dict(self.profile.get("promotion_thresholds") or {})
        eligible = (
            n >= int(thresholds.get("minimum_examples", 5000))
            and correct / n >= float(thresholds.get("minimum_policy_accuracy", 0.60))
            and regret_total / n <= float(thresholds.get("maximum_expected_regret", 1.5))
            and illegal == 0
        )
        # Result reports eligibility only. It never performs promotion.
        return BenchmarkResult(model_id, n, correct / n, top3 / n, regret_total / n,
                                  value_loss / n, brier / n, illegal / n, archetypes, eligible)

    def evaluate_permanent_holdout(
        self,
        model_id: str,
        examples: Iterable[Mapping[str, Any]],
        predictions: Iterable[Mapping[str, Any]],
        *,
        dataset_id: str,
        excluded_groups: Mapping[str, Iterable[str]],
    ) -> PermanentHoldoutResult:
        """Evaluate a frozen leakage-safe holdout without changing defaults."""
        rows = [dict(row) for row in examples]
        prediction_rows = [dict(row) for row in predictions]
        prediction_by_position: dict[str, dict[str, Any]] = {}
        for prediction in prediction_rows:
            position_id = str(prediction.get("positionId", "")).strip()
            if not position_id or position_id in prediction_by_position:
                raise ValueError(
                    "prediction position identities must be unique"
                )
            prediction_by_position[position_id] = prediction
        governance = dict(self.profile.get("permanent_holdout") or {})
        minimum_positions = int(governance.get("minimum_positions", 500))
        minimum_per_archetype = int(
            governance.get("minimum_per_archetype", 1)
        )
        strata = tuple(str(value) for value in self.profile.get("strata") or ())
        if minimum_positions < 1 or minimum_per_archetype < 1 or not strata:
            raise ValueError("permanent holdout policy is invalid")
        required_group_keys = (
            "dealHash",
            "episodeId",
            "sourceGroup",
            "seedGroup",
        )
        excluded = {
            key: {str(value) for value in excluded_groups.get(key, ())}
            for key in required_group_keys
        }
        if set(excluded_groups) != set(required_group_keys):
            raise ValueError(
                "excluded groups must declare deal, episode, source, and seed"
            )
        if len(rows) < minimum_positions:
            raise ValueError("permanent holdout has insufficient positions")
        rows.sort(key=lambda row: str(row.get("positionId", "")))
        if len(prediction_rows) != len(rows):
            raise ValueError("predictions must cover the complete holdout")

        position_ids: set[str] = set()
        correct = top3 = illegal = exact_labels = adjudicated_labels = 0
        regrets: list[float] = []
        confidences: list[tuple[float, int]] = []
        completion: dict[str, int] = {}
        groups: dict[str, list[tuple[int, float]]] = {}
        latencies: list[float] = []
        memories: list[float] = []
        nodes: list[float] = []

        for row in rows:
            position_id = str(row.get("positionId", "")).strip()
            if not position_id or position_id in position_ids:
                raise ValueError(
                    "permanent holdout position identities must be unique"
                )
            position_ids.add(position_id)
            prediction = prediction_by_position.get(position_id)
            if prediction is None:
                raise ValueError("prediction is missing for holdout position")
            if str(row.get("split", "")) != "permanent_holdout":
                raise ValueError("benchmark row is not in permanent holdout")
            _require_sha256(row.get("stateSha256"), "stateSha256")
            _require_sha256(row.get("rulesSha256"), "rulesSha256")
            _require_sha256(
                row.get("configurationSha256"),
                "configurationSha256",
            )
            if not str(row.get("engineVersion", "")).strip():
                raise ValueError("benchmark row is missing engineVersion")
            if str(row.get("observer", "")).strip() not in {
                "north",
                "west",
                "south",
                "east",
            }:
                raise ValueError("benchmark observer is invalid")
            for key in required_group_keys:
                value = str(row.get(key, "")).strip()
                if not value:
                    raise ValueError(f"benchmark row is missing {key}")
                if key == "dealHash":
                    _require_sha256(value, key)
                if value in excluded[key]:
                    raise ValueError(
                        f"permanent holdout leakage detected for {key}"
                    )
            archetype = str(row.get("archetype", "")).strip()
            if archetype not in strata:
                raise ValueError(
                    f"unsupported benchmark archetype: {archetype}"
                )
            label_source = str(row.get("labelSource", "")).strip()
            if not str(row.get("labelAuthorityId", "")).strip():
                raise ValueError("benchmark label authority is missing")
            _require_sha256(
                row.get("labelAuthoritySha256"),
                "labelAuthoritySha256",
            )
            if label_source == "exact":
                exact_labels += 1
            elif label_source == "independent_adjudication":
                adjudicators = [
                    str(value).strip()
                    for value in row.get("adjudicators", ())
                    if str(value).strip()
                ]
                if (
                    int(row.get("adjudicatorCount", 0)) < 2
                    or len(set(adjudicators)) < 2
                    or len(adjudicators)
                    != int(row.get("adjudicatorCount", 0))
                ):
                    raise ValueError(
                        "adjudicated labels require at least two reviewers"
                    )
                if float(row.get("adjudicationAgreement", 0.0)) < float(
                    governance.get("minimum_adjudication_agreement", 0.8)
                ):
                    raise ValueError(
                        "adjudicated label agreement is below policy"
                    )
                adjudicated_labels += 1
            else:
                raise ValueError("benchmark label provenance is unsupported")

            legal = tuple(_key(value) for value in row.get("legalActions", ()))
            if len(set(legal)) < 2:
                raise ValueError(
                    "permanent holdout positions require two legal choices"
                )
            raw_references = row.get("referenceActions")
            if raw_references is None:
                raw_references = (row.get("referenceAction"),)
            references = tuple(
                dict.fromkeys(_key(value) for value in raw_references)
            )
            predicted_action = _key(prediction.get("predictedAction"))
            ranked = tuple(
                _key(value) for value in prediction.get("rankedActions", ())
            )
            if not references or any(
                value not in legal for value in references
            ):
                raise ValueError("reference actions must be legal")
            action_regrets = {
                _key(action): float(regret)
                for action, regret in dict(
                    row.get("actionRegrets") or {}
                ).items()
            }
            if (
                set(action_regrets) != set(legal)
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in action_regrets.values()
                )
                or min(action_regrets.values()) != 0.0
                or any(action_regrets[value] != 0.0 for value in references)
                or {
                    action
                    for action, value in action_regrets.items()
                    if value == 0.0
                }
                != set(references)
            ):
                raise ValueError(
                    "label-owned action regrets are invalid"
                )
            hit = int(predicted_action in references)
            correct += hit
            top3 += int(
                any(value in ranked[:3] for value in references)
            )
            illegal += int(predicted_action not in legal)
            confidence = float(
                prediction.get("predictedConfidence", 0.0)
            )
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("predicted confidence must be in [0, 1]")
            confidences.append((confidence, hit))
            regret = action_regrets.get(
                predicted_action,
                max(action_regrets.values()) + 1.0,
            )
            regrets.append(regret)
            groups.setdefault(archetype, []).append((hit, regret))
            status = str(
                prediction.get("searchCompletion", "")
            ).strip()
            if status not in {
                "exact",
                "heuristic_complete",
                "budget_exhausted",
                "cancelled",
                "failed",
                "not_applicable",
            }:
                raise ValueError("search completion status is unsupported")
            completion[status] = completion.get(status, 0) + 1
            for key, target in (
                ("latencyMs", latencies),
                ("memoryBytes", memories),
                ("nodes", nodes),
            ):
                value = float(prediction.get(key, math.nan))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{key} must be finite and nonnegative")
                target.append(value)

        missing = [
            archetype
            for archetype in strata
            if len(groups.get(str(archetype), ())) < minimum_per_archetype
        ]
        if missing:
            raise ValueError(
                "permanent holdout lacks archetype coverage: "
                + ", ".join(sorted(missing))
            )
        n = len(rows)
        calibration_bins = max(
            2,
            min(50, int(governance.get("calibration_bins", 10))),
        )
        ece = 0.0
        for index in range(calibration_bins):
            lower = index / calibration_bins
            upper = (index + 1) / calibration_bins
            selected = [
                item
                for item in confidences
                if lower <= item[0] < upper
                or (index == calibration_bins - 1 and item[0] == 1.0)
            ]
            if selected:
                mean_confidence = statistics.fmean(
                    item[0] for item in selected
                )
                empirical = statistics.fmean(item[1] for item in selected)
                ece += len(selected) / n * abs(mean_confidence - empirical)
        accuracy_interval = _wilson(correct, n)
        regret_interval = _mean_interval(regrets)
        dataset_sha256 = _hash(rows)
        archetypes = {
            name: {
                "examples": len(items),
                "policyAccuracy": statistics.fmean(
                    item[0] for item in items
                ),
                "expectedRegret": statistics.fmean(
                    item[1] for item in items
                ),
            }
            for name, items in sorted(groups.items())
        }
        workload = {
            "latencyP50Ms": _percentile(latencies, 0.50),
            "latencyP95Ms": _percentile(latencies, 0.95),
            "latencyP99Ms": _percentile(latencies, 0.99),
            "memoryPeakBytes": max(memories),
            "nodesMean": statistics.fmean(nodes),
        }
        brier = statistics.fmean(
            (confidence - hit) ** 2 for confidence, hit in confidences
        )
        return PermanentHoldoutResult(
            model_id=model_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            examples=n,
            policy_accuracy=correct / n,
            top3_accuracy=top3 / n,
            expected_regret=statistics.fmean(regrets),
            brier_score=brier,
            expected_calibration_error=ece,
            illegal_action_rate=illegal / n,
            exact_labels=exact_labels,
            adjudicated_labels=adjudicated_labels,
            accuracy_interval=accuracy_interval,
            regret_interval=regret_interval,
            archetypes=archetypes,
            search_completion=completion,
            workload=workload,
        )

    def evaluate_symmetric_pairs(
        self,
        candidate_id: str,
        baseline_id: str,
        pairs: Iterable[Mapping[str, Any]],
        *,
        candidate_expected_regret: float,
        baseline_expected_regret: float,
    ) -> PairedMatchResult:
        """Evaluate duplicate deals with seats and teams swapped."""
        if not candidate_id.strip() or not baseline_id.strip():
            raise ValueError("paired comparison identities are required")
        if candidate_id == baseline_id:
            raise ValueError("candidate and baseline must be different")
        rows = [dict(row) for row in pairs]
        governance = dict(self.profile.get("paired_comparison") or {})
        minimum_pairs = int(governance.get("minimum_pairs", 100))
        if minimum_pairs < 1:
            raise ValueError("paired comparison policy is invalid")
        if len(rows) < minimum_pairs:
            raise ValueError("paired comparison has insufficient pairs")
        rows.sort(key=lambda row: str(row.get("pairId", "")))
        pair_ids: set[str] = set()
        comparison_units: set[tuple[str, str, str]] = set()
        margins: list[float] = []
        wins = losses = ties = 0
        for row in rows:
            pair_id = str(row.get("pairId", "")).strip()
            if not pair_id or pair_id in pair_ids:
                raise ValueError("paired comparison identities must be unique")
            pair_ids.add(pair_id)
            if str(row.get("candidateId", "")).strip() != candidate_id:
                raise ValueError("paired row candidate identity mismatch")
            if str(row.get("baselineId", "")).strip() != baseline_id:
                raise ValueError("paired row baseline identity mismatch")
            deal_hash = _require_sha256(row.get("dealHash"), "dealHash")
            configuration_hash = _require_sha256(
                row.get("configurationHash"),
                "configurationHash",
            )
            _require_sha256(row.get("rulesSha256"), "rulesSha256")
            if not str(row.get("engineVersion", "")).strip():
                raise ValueError("paired comparison requires an engine version")
            seed_group = str(row.get("seedGroup", "")).strip()
            if not seed_group:
                raise ValueError("paired comparison requires a seed group")
            comparison_unit = (deal_hash, configuration_hash, seed_group)
            if comparison_unit in comparison_units:
                raise ValueError("paired comparison contains a duplicate unit")
            comparison_units.add(comparison_unit)
            games = list(row.get("games") or ())
            if len(games) != 2:
                raise ValueError("paired comparison requires two swapped games")
            teams = [
                str(game.get("candidateTeam", "")).strip()
                for game in games
            ]
            if sorted(teams) != ["north_south", "west_east"]:
                raise ValueError(
                    "paired comparison must swap candidate teams"
                )
            margin = 0.0
            for game in games:
                candidate = float(game.get("candidateScore", math.nan))
                baseline = float(game.get("baselineScore", math.nan))
                if not math.isfinite(candidate) or not math.isfinite(baseline):
                    raise ValueError("paired scores must be finite")
                margin += candidate - baseline
            margins.append(margin)
            if margin > 0.0:
                wins += 1
            elif margin < 0.0:
                losses += 1
            else:
                ties += 1
        decisive = wins + losses
        decisive_rate = wins / decisive if decisive else 0.0
        win_interval = _wilson(wins, decisive)
        margin_interval = _mean_interval(margins)
        if not math.isfinite(candidate_expected_regret) or not math.isfinite(
            baseline_expected_regret
        ):
            raise ValueError("regret evidence must be finite")
        if candidate_expected_regret < 0.0 or baseline_expected_regret <= 0.0:
            raise ValueError("regret evidence is outside valid bounds")
        regret_improvement = (
            baseline_expected_regret - candidate_expected_regret
        ) / baseline_expected_regret
        target_rate = float(governance.get("minimum_win_rate", 0.55))
        target_lower = float(
            governance.get("minimum_win_rate_lower_bound", 0.50)
        )
        target_regret = float(
            governance.get("minimum_regret_improvement", 0.25)
        )
        promotion_target_met = (
            decisive_rate >= target_rate
            and win_interval[0] > target_lower
            and regret_improvement >= target_regret
        )
        return PairedMatchResult(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            pairs=len(rows),
            wins=wins,
            losses=losses,
            ties=ties,
            decisive_win_rate=decisive_rate,
            win_rate_interval=win_interval,
            mean_paired_margin=statistics.fmean(margins),
            margin_interval=margin_interval,
            regret_improvement=regret_improvement,
            promotion_target_met=promotion_target_met,
        )


# Compatibility aliases for previously published Python imports. New code uses
# the responsibility-based names above; the aliases can be removed under the
# documented API deprecation policy.
BenchmarkResultV18 = BenchmarkResult
PermanentHoldoutResultV21 = PermanentHoldoutResult
PairedMatchResultV21 = PairedMatchResult
IntelligenceBenchmarkV18 = StrengthBenchmark
