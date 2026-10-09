from __future__ import annotations

import math
from dataclasses import dataclass

from statsmodels.stats.power import TTestPower

from .domain import MetricAnalysisTransform, MetricDefinition, ThresholdType


DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
SOFT_SHOT_WARNING_LIMIT = 100
HARD_SHOT_LIMIT = 1000
POWER_METHOD = "paired_block_t"
LEGACY_POWER_METHOD = "independent_two_sample_t"


@dataclass(frozen=True)
class PowerAnalysis:
    """Sample-size recommendation for an experiment's primary metric."""

    expected_effect: float
    expected_standard_deviation: float | None
    standardized_effect: float
    raw_shots_per_configuration: float
    recommended_shots_per_configuration: int
    alpha: float
    target_power: float
    block_size: int
    analysis_basis: str
    achieved_power: float
    warnings: tuple[str, ...] = ()
    method: str = POWER_METHOD
    expected_block_difference_standard_deviation: float | None = None
    variability_source: str = "shot_sd_approximation"
    raw_block_pairs: float | None = None
    recommended_block_pairs: int | None = None

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
            "method": self.method,
            "variability_source": self.variability_source,
            "expected_block_difference_standard_deviation": self.expected_block_difference_standard_deviation,
            "raw_block_pairs": self.raw_block_pairs,
            "recommended_block_pairs": self.recommended_block_pairs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PowerAnalysis":
        block_size = int(data["block_size"])
        recommended_shots = int(data["recommended_shots_per_configuration"])
        method = str(data.get("method", LEGACY_POWER_METHOD))
        expected_standard_deviation = (
            float(data["expected_standard_deviation"])
            if data.get("expected_standard_deviation") is not None else None
        )
        expected_block_sd = data.get("expected_block_difference_standard_deviation")
        if expected_block_sd is None:
            if expected_standard_deviation is None:
                raise ValueError("A planning SD is required")
            expected_block_sd = (
                expected_standard_deviation * math.sqrt(2 / block_size)
                if method == POWER_METHOD
                else expected_standard_deviation
            )
        raw_shots = float(data["raw_shots_per_configuration"])
        raw_block_pairs = data.get("raw_block_pairs")
        if raw_block_pairs is None:
            raw_block_pairs = raw_shots / block_size
        recommended_block_pairs = data.get("recommended_block_pairs")
        if recommended_block_pairs is None:
            recommended_block_pairs = recommended_shots // block_size
        return cls(
            expected_effect=float(data["expected_effect"]),
            expected_standard_deviation=expected_standard_deviation,
            standardized_effect=float(data["standardized_effect"]),
            raw_shots_per_configuration=raw_shots,
            recommended_shots_per_configuration=recommended_shots,
            alpha=float(data["alpha"]),
            target_power=float(data["target_power"]),
            block_size=block_size,
            analysis_basis=str(data["analysis_basis"]),
            achieved_power=float(data["achieved_power"]),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            method=method,
            variability_source=str(data.get("variability_source", "shot_sd_approximation")),
            expected_block_difference_standard_deviation=float(expected_block_sd),
            raw_block_pairs=float(raw_block_pairs),
            recommended_block_pairs=int(recommended_block_pairs),
        )

    def __post_init__(self) -> None:
        numeric_fields = (
            self.expected_effect,
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
        if self.variability_source not in {"shot_sd_approximation", "pilot_block_difference_sd"}:
            raise ValueError("Unknown variability source")
        if self.expected_standard_deviation is not None:
            _validate_sd(self.expected_standard_deviation, "Expected shot SD")
        if self.variability_source == "shot_sd_approximation" and self.expected_standard_deviation is None:
            raise ValueError("Expected shot SD is required for the approximation")
        if self.variability_source == "pilot_block_difference_sd" and self.expected_block_difference_standard_deviation is None:
            raise ValueError("Pilot block-difference SD is required")
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
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Power-analysis method cannot be empty")
        if self.expected_block_difference_standard_deviation is None:
            derived_block_sd = (
                self.expected_standard_deviation * math.sqrt(2 / self.block_size)
                if self.method == POWER_METHOD
                else self.expected_standard_deviation
            )
            object.__setattr__(
                self,
                "expected_block_difference_standard_deviation",
                derived_block_sd,
            )
        if (
            not isinstance(self.expected_block_difference_standard_deviation, (int, float))
            or isinstance(self.expected_block_difference_standard_deviation, bool)
            or not math.isfinite(self.expected_block_difference_standard_deviation)
            or self.expected_block_difference_standard_deviation <= 0
        ):
            raise ValueError("Expected block-difference standard deviation must be positive and finite")
        if self.raw_block_pairs is None:
            object.__setattr__(self, "raw_block_pairs", self.raw_shots_per_configuration / self.block_size)
        if self.recommended_block_pairs is None:
            object.__setattr__(self, "recommended_block_pairs", self.recommended_shots_per_configuration // self.block_size)
        if (
            not isinstance(self.raw_block_pairs, (int, float))
            or isinstance(self.raw_block_pairs, bool)
            or not math.isfinite(self.raw_block_pairs)
            or self.raw_block_pairs <= 0
        ):
            raise ValueError("Raw block-pair sample size must be positive and finite")
        if (
            not isinstance(self.recommended_block_pairs, int)
            or isinstance(self.recommended_block_pairs, bool)
            or self.recommended_block_pairs < 2
        ):
            raise ValueError("Recommended block-pair sample size must be at least two")


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


def estimate_block_difference_standard_deviation(
    expected_shot_standard_deviation: float,
    block_size: int,
) -> float:
    """Estimate SD of a paired block difference from individual-shot SD.

    This approximation assumes equal A/B shot variance, independent shots
    within each block, and zero covariance between paired block means.
    """

    if (
        isinstance(expected_shot_standard_deviation, bool)
        or not isinstance(expected_shot_standard_deviation, (int, float))
        or not math.isfinite(expected_shot_standard_deviation)
        or expected_shot_standard_deviation <= 0
    ):
        raise ValueError("Expected shot standard deviation must be finite and greater than zero")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")
    return float(expected_shot_standard_deviation * math.sqrt(2 / block_size))


def _validate_sd(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _planning_sd(shot_sd: float | None, block_sd: float | None, block_size: int) -> float:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")
    if shot_sd is not None:
        _validate_sd(shot_sd, "Expected shot SD")
    if block_sd is not None:
        _validate_sd(block_sd, "Pilot block-difference SD")
        return float(block_sd)
    if shot_sd is None:
        raise ValueError("Provide an expected shot SD or a pilot block-difference SD")
    return estimate_block_difference_standard_deviation(shot_sd, block_size)


def calculate_power_analysis(
    metric: MetricDefinition,
    *,
    expected_standard_deviation: float | None = None,
    expected_block_difference_standard_deviation: float | None = None,
    expected_baseline_mean: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    block_size: int = 5,
) -> PowerAnalysis:
    """Estimate paired A/B block sample sizes for a metric.

    Use a directly supplied pilot block-difference SD when available.
    Otherwise approximate it from shot SD. Both paths use a one-sample
    t power model on independent paired differences.
    """

    block_difference_sd = _planning_sd(
        expected_standard_deviation, expected_block_difference_standard_deviation, block_size
    )

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
    standardized_effect = effect / block_difference_sd
    raw_block_pairs = float(
        TTestPower().solve_power(
            effect_size=standardized_effect,
            alpha=alpha,
            power=target_power,
            alternative="two-sided",
        )
    )
    if not math.isfinite(raw_block_pairs):
        raise ValueError("Power analysis could not produce a finite block-pair sample size")

    recommended_block_pairs = max(2, math.ceil(raw_block_pairs))
    recommended_shots = recommended_block_pairs * block_size
    achieved_power = float(
        TTestPower().power(
            effect_size=standardized_effect,
            nobs=recommended_block_pairs,
            alpha=alpha,
            alternative="two-sided",
        )
    )
    source = ("pilot_block_difference_sd" if expected_block_difference_standard_deviation is not None
              else "shot_sd_approximation")
    warnings = [
        "Power is conditional on the assumed effect and SD, not a guarantee. Paired differences must be independent across pairs."
    ]
    if source == "shot_sd_approximation":
        warnings.append(
            "Planning approximation: shot SD × sqrt(2 / block size) assumes equal A/B shot variance, "
            "independent shots within blocks, and zero covariance between paired A/B block means. "
            "Actual block-difference SD may be higher or lower; use comparable pilot block differences when available."
        )
    else:
        warnings.append(
            "Pilot-based planning: use the sample SD of B-block mean minus A-block mean from a separate pilot "
            "with the same block size, order design, metric transform, and comparable conditions. A small pilot gives an uncertain SD; "
            "check a larger SD before committing to the plan."
        )
    if recommended_shots > SOFT_SHOT_WARNING_LIMIT:
        warnings.append(
            f"This plan requires more than {SOFT_SHOT_WARNING_LIMIT} valid shots per configuration and may be operationally demanding."
        )
    if recommended_shots > HARD_SHOT_LIMIT:
        warnings.append(
            f"The powered recommendation requires {recommended_shots} shots per configuration, exceeding the operational limit of {HARD_SHOT_LIMIT}; confirmatory mode is unavailable."
        )
    return PowerAnalysis(
        expected_effect=effect,
        expected_standard_deviation=expected_standard_deviation,
        standardized_effect=standardized_effect,
        raw_shots_per_configuration=raw_block_pairs * block_size,
        recommended_shots_per_configuration=recommended_shots,
        alpha=alpha,
        target_power=target_power,
        block_size=block_size,
        analysis_basis=analysis_basis,
        achieved_power=achieved_power,
        warnings=tuple(warnings),
        method=POWER_METHOD,
        expected_block_difference_standard_deviation=block_difference_sd,
        variability_source=source,
        raw_block_pairs=raw_block_pairs,
        recommended_block_pairs=recommended_block_pairs,
    )


def calculate_minimum_detectable_effect(
    *,
    expected_standard_deviation: float | None = None,
    expected_block_difference_standard_deviation: float | None = None,
    shots_per_configuration: int,
    block_size: int = 5,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
) -> float:
    """Calculate the smallest absolute effect detectable at a sample size."""

    block_difference_sd = _planning_sd(
        expected_standard_deviation, expected_block_difference_standard_deviation, block_size
    )

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

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")
    if shots_per_configuration % block_size != 0:
        raise ValueError("Shots per configuration must be divisible by block size")
    block_pairs = shots_per_configuration // block_size
    if block_pairs < 2:
        raise ValueError("At least two paired blocks per configuration are required")
    standardized_effect = float(
        TTestPower().solve_power(
            effect_size=None,
            nobs=block_pairs,
            alpha=alpha,
            power=target_power,
            alternative="two-sided",
        )
    )
    if not math.isfinite(standardized_effect) or standardized_effect <= 0:
        raise ValueError("Minimum detectable effect could not be calculated")
    return standardized_effect * block_difference_sd
