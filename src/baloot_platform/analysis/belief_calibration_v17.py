from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_frequency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower, "upper": self.upper, "count": self.count,
            "meanConfidence": self.mean_confidence, "empiricalFrequency": self.empirical_frequency,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    examples: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float
    top1_accuracy: float
    bins: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "brierScore": self.brier_score,
            "logLoss": self.log_loss,
            "expectedCalibrationError": self.expected_calibration_error,
            "maximumCalibrationError": self.maximum_calibration_error,
            "top1Accuracy": self.top1_accuracy,
            "bins": [item.to_dict() for item in self.bins],
        }


def evaluate_multiclass_calibration(
    predictions: Iterable[Mapping[str, float]],
    labels: Iterable[str],
    *,
    bins: int = 10,
    epsilon: float = 1e-12,
) -> CalibrationReport:
    pairs = list(zip(predictions, labels, strict=True))
    if not pairs:
        raise ValueError("at least one prediction is required")
    bins = max(2, min(50, int(bins)))
    brier_total = 0.0
    loss_total = 0.0
    correct = 0
    confidence_rows: list[tuple[float, int]] = []
    for raw, label in pairs:
        probabilities = {str(key): max(0.0, float(value)) for key, value in raw.items()}
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("prediction probabilities must have positive mass")
        probabilities = {key: value / total for key, value in probabilities.items()}
        if label not in probabilities:
            probabilities[label] = 0.0
        keys = set(probabilities)
        brier_total += sum((probabilities[key] - (1.0 if key == label else 0.0)) ** 2 for key in keys)
        loss_total += -math.log(max(epsilon, probabilities[label]))
        predicted, confidence = max(probabilities.items(), key=lambda item: item[1])
        hit = int(predicted == label)
        correct += hit
        confidence_rows.append((confidence, hit))

    calibration_bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    count = len(pairs)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [row for row in confidence_rows if lower <= row[0] < upper or (index == bins - 1 and row[0] == 1.0)]
        if not selected:
            calibration_bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        mean_confidence = sum(row[0] for row in selected) / len(selected)
        empirical = sum(row[1] for row in selected) / len(selected)
        gap = abs(mean_confidence - empirical)
        ece += len(selected) / count * gap
        mce = max(mce, gap)
        calibration_bins.append(CalibrationBin(lower, upper, len(selected), mean_confidence, empirical))
    return CalibrationReport(
        examples=count,
        brier_score=brier_total / count,
        log_loss=loss_total / count,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        top1_accuracy=correct / count,
        bins=tuple(calibration_bins),
    )


def flatten_card_location_examples(
    predictions: Iterable[Mapping[str, Mapping[str, float]]],
    true_locations: Iterable[Mapping[str, str]],
) -> tuple[list[Mapping[str, float]], list[str]]:
    rows: list[Mapping[str, float]] = []
    labels: list[str] = []
    for predicted, truth in zip(predictions, true_locations, strict=True):
        for card, location in truth.items():
            if card not in predicted:
                raise ValueError(f"missing card prediction: {card}")
            rows.append(predicted[card]); labels.append(location)
    return rows, labels
