from __future__ import annotations

import math
from dataclasses import dataclass

from statsmodels.stats.power import TTestIndPower

from .domain import MetricAnalysisTransform, MetricDefinition, ThresholdType


DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
SOFT_SHOT_WARNING_LIMIT = 100
HARD_SHOT_LIMIT = 1000


@dataclass(frozen=True)
class PowerAnalysis:
    """Sample-size recommendation for an experiment's primary metric."""

    expected_effect: float
    expected_standard_deviation: float
    standardized_effect: float
    raw_shots_per_configuration: float
    recommended_shots_per_configuration: int
    alpha: float
    target_power: float
    block_size: int
    analysis_basis: str
    achieved_power: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_effect": self.expected_effect,
            "expected_standard_deviation": self.expected_standard_deviation,
            "standardized_effect": self.standardized_effect,
            "raw_shots_per_configuration": self.raw_shots_per_configuration,
            "recommended_shots_per_configuration": self.recommended_shots_per_configuration,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "block_size": self.block_size,
            "analysis_basis": self.analysis_basis,
            "achieved_power": self.achieved_power,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PowerAnalysis":
        return cls(
            expected_effect=float(data["expected_effect"]),
            expected_standard_deviation=float(data["expected_standard_deviation"]),
            standardized_effect=float(data["standardized_effect"]),
            raw_shots_per_configuration=float(data["raw_shots_per_configuration"]),
            recommended_shots_per_configuration=int(data["recommended_shots_per_configuration"]),
            alpha=float(data["alpha"]),
            target_power=float(data["target_power"]),
            block_size=int(data["block_size"]),
            analysis_basis=str(data["analysis_basis"]),
            achieved_power=float(data["achieved_power"]),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
        )

    def __post_init__(self) -> None:
        numeric_fields = (
            self.expected_effect,
            self.expected_standard_deviation,
            self.standardized_effect,
            self.raw_shots_per_configuration,
            self.alpha,
            self.target_power,
            self.achieved_power,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric_fields
        ):
            raise ValueError("Power-analysis numeric fields must be finite numbers")
        if self.expected_effect <= 0:
            raise ValueError("Expected effect must be greater than zero")
        if (
            not math.isfinite(self.expected_standard_deviation)
            or self.expected_standard_deviation <= 0
        ):
            raise ValueError("Expected standard deviation must be greater than zero")
        if not math.isfinite(self.standardized_effect) or self.standardized_effect <= 0:
            raise ValueError("Standardized effect must be greater than zero")
        if (
            isinstance(self.recommended_shots_per_configuration, bool)
            or not isinstance(self.recommended_shots_per_configuration, int)
            or self.recommended_shots_per_configuration <= 0
        ):
            raise ValueError("Recommended sample size must be greater than zero")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("Block size must be a positive integer")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between zero and one")
        if not 0 < self.target_power < 1:
            raise ValueError("Target power must be between zero and one")
        if not 0 < self.achieved_power <= 1:
            raise ValueError("Achieved power must be between zero and one")
        if self.recommended_shots_per_configuration % self.block_size != 0:
            raise ValueError("Recommended sample size must match the block size")


def _expected_effect(
    metric: MetricDefinition,
    *,
    expected_baseline_mean: float | None,
) -> tuple[float, str]:
    if metric.threshold_type is ThresholdType.ABSOLUTE:
        if metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET:
            return (
                metric.meaningful_threshold,
                "reduction in absolute error from the target",
            )
        return metric.meaningful_threshold, "absolute change in the primary metric"

    if expected_baseline_mean is None:
        raise ValueError(
            "Expected baseline mean is required for percentage thresholds"
        )
    if not math.isfinite(expected_baseline_mean) or expected_baseline_mean == 0:
        raise ValueError(
            "Expected baseline mean must be finite and non-zero for percentage thresholds"
        )
    return (
        abs(expected_baseline_mean) * metric.meaningful_threshold / 100,
        (
            "percentage reduction in absolute error from the target"
            if metric.analysis_transform
            is MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET
            else "percentage change in the transformed primary metric converted to its measurement units"
        ),
    )


def calculate_power_analysis(
    metric: MetricDefinition,
    *,
    expected_standard_deviation: float,
    expected_baseline_mean: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    block_size: int = 5,
) -> PowerAnalysis:
    """Estimate equal A/B sample sizes for a metric.

    This uses an independent two-sample means approximation. The practical
    threshold is treated as the effect worth detecting, and the user-provided
    standard deviation describes the analysis quantity for the metric.
    """

    if (
        isinstance(expected_standard_deviation, bool)
        or not isinstance(expected_standard_deviation, (int, float))
        or not math.isfinite(expected_standard_deviation)
        or expected_standard_deviation <= 0
    ):
        raise ValueError("Expected standard deviation must be finite and greater than zero")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("Alpha must be between zero and one")
    if (
        isinstance(target_power, bool)
        or not isinstance(target_power, (int, float))
        or not math.isfinite(target_power)
        or not 0 < target_power < 1
    ):
        raise ValueError("Target power must be between zero and one")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")

    effect, analysis_basis = _expected_effect(
        metric,
        expected_baseline_mean=expected_baseline_mean,
    )
    standardized_effect = effect / expected_standard_deviation
    raw_shots = float(
        TTestIndPower().solve_power(
            effect_size=standardized_effect,
            alpha=alpha,
            power=target_power,
            ratio=1.0,
            alternative="two-sided",
        )
    )
    if not math.isfinite(raw_shots):
        raise ValueError("Power analysis could not produce a finite sample size")

    recommended_shots = max(1, math.ceil(raw_shots / block_size) * block_size)
    if recommended_shots > HARD_SHOT_LIMIT:
        raise ValueError(
            f"Recommended sample size exceeds the safety limit of {HARD_SHOT_LIMIT} shots per configuration"
        )
    achieved_power = float(
        TTestIndPower().power(
            effect_size=standardized_effect,
            nobs1=recommended_shots,
            alpha=alpha,
            ratio=1.0,
            alternative="two-sided",
        )
    )
    warnings = [
        "Power analysis is an estimate, not a guarantee. It assumes independent observations with a common variance; repeated golf shots may violate this assumption."
    ]
    if recommended_shots > SOFT_SHOT_WARNING_LIMIT:
        warnings.append(
            f"This plan requires more than {SOFT_SHOT_WARNING_LIMIT} valid shots per configuration and may be operationally demanding."
        )
    return PowerAnalysis(
        expected_effect=effect,
        expected_standard_deviation=expected_standard_deviation,
        standardized_effect=standardized_effect,
        raw_shots_per_configuration=raw_shots,
        recommended_shots_per_configuration=recommended_shots,
        alpha=alpha,
        target_power=target_power,
        block_size=block_size,
        analysis_basis=analysis_basis,
        achieved_power=achieved_power,
        warnings=tuple(warnings),
    )


def calculate_minimum_detectable_effect(
    *,
    expected_standard_deviation: float,
    shots_per_configuration: int,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
) -> float:
    """Calculate the smallest absolute effect detectable at a sample size."""

    if (
        isinstance(expected_standard_deviation, bool)
        or not isinstance(expected_standard_deviation, (int, float))
        or not math.isfinite(expected_standard_deviation)
        or expected_standard_deviation <= 0
    ):
        raise ValueError("Expected standard deviation must be finite and greater than zero")
    if (
        isinstance(shots_per_configuration, bool)
        or not isinstance(shots_per_configuration, int)
        or shots_per_configuration < 2
    ):
        raise ValueError("Shots per configuration must be an integer of at least two")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("Alpha must be between zero and one")
    if (
        isinstance(target_power, bool)
        or not isinstance(target_power, (int, float))
        or not math.isfinite(target_power)
        or not 0 < target_power < 1
    ):
        raise ValueError("Target power must be between zero and one")

    standardized_effect = float(
        TTestIndPower().solve_power(
            effect_size=None,
            nobs1=shots_per_configuration,
            alpha=alpha,
            power=target_power,
            ratio=1.0,
            alternative="two-sided",
        )
    )
    if not math.isfinite(standardized_effect) or standardized_effect <= 0:
        raise ValueError("Minimum detectable effect could not be calculated")
    return standardized_effect * expected_standard_deviation
