import math

import pytest

from golf_performance.domain import MetricDefinition, MetricDirection, ThresholdType
from golf_performance.power import (
    POWER_METHOD,
    calculate_minimum_detectable_effect,
    calculate_power_analysis,
    estimate_block_difference_standard_deviation,
)


def metric(
    *,
    direction: MetricDirection = MetricDirection.HIGHER,
    threshold: float = 3.0,
    threshold_type: ThresholdType = ThresholdType.ABSOLUTE,
) -> MetricDefinition:
    return MetricDefinition(
        key="metric",
        display_name="Metric",
        unit="units",
        direction=direction,
        meaningful_threshold=threshold,
        threshold_type=threshold_type,
        target_value=10.0 if direction is MetricDirection.TARGET else None,
        is_primary=True,
    )


def test_power_analysis_uses_practical_threshold_as_effect() -> None:
    result = calculate_power_analysis(
        metric(), expected_standard_deviation=6.0, block_size=5
    )

    assert result.expected_effect == 3.0
    assert result.expected_block_difference_standard_deviation == pytest.approx(
        estimate_block_difference_standard_deviation(6.0, 5)
    )
    assert result.standardized_effect == pytest.approx(
        3.0 / estimate_block_difference_standard_deviation(6.0, 5)
    )
    assert result.method == "paired_block_t"
    assert result.recommended_block_pairs >= 2
    assert result.recommended_shots_per_configuration == result.recommended_block_pairs * 5
    assert result.recommended_shots_per_configuration % 5 == 0
    assert result.recommended_shots_per_configuration >= math.ceil(
        result.raw_shots_per_configuration
    )
    assert result.method == POWER_METHOD
    assert result.method == "paired_block_t"


def test_minimum_detectable_effect_is_consistent_with_power_model() -> None:
    standard_deviation = 6.0
    shots_per_configuration = 40
    mde = calculate_minimum_detectable_effect(
        expected_standard_deviation=standard_deviation,
        shots_per_configuration=shots_per_configuration,
        alpha=0.05,
        target_power=0.80,
    )

    result = calculate_power_analysis(
        metric(threshold=mde),
        expected_standard_deviation=standard_deviation,
        block_size=5,
    )

    assert result.raw_shots_per_configuration <= shots_per_configuration + 1


@pytest.mark.parametrize("direction", [MetricDirection.HIGHER, MetricDirection.LOWER])
def test_direction_does_not_change_required_sample_size(direction: MetricDirection) -> None:
    result = calculate_power_analysis(
        metric(direction=direction), expected_standard_deviation=6.0
    )

    assert result.expected_effect == 3.0


def test_percentage_threshold_is_converted_to_measurement_units() -> None:
    result = calculate_power_analysis(
        metric(
            direction=MetricDirection.LOWER,
            threshold=10.0,
            threshold_type=ThresholdType.PERCENTAGE,
        ),
        expected_standard_deviation=4.0,
        expected_baseline_mean=30.0,
    )

    assert result.expected_effect == pytest.approx(3.0)


def test_target_metric_powers_reduction_in_absolute_error() -> None:
    result = calculate_power_analysis(
        metric(direction=MetricDirection.TARGET),
        expected_standard_deviation=2.0,
    )

    assert result.expected_effect == 3.0
    assert "absolute error" in result.analysis_basis


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_standard_deviation": 0.0},
        {"expected_standard_deviation": -1.0},
        {"expected_standard_deviation": float("nan")},
        {"expected_standard_deviation": 2.0, "alpha": 0.0},
        {"expected_standard_deviation": 2.0, "target_power": 1.0},
        {"expected_standard_deviation": 2.0, "block_size": 0},
    ],
)
def test_invalid_power_inputs_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        calculate_power_analysis(metric(), **kwargs)


def test_percentage_threshold_requires_nonzero_baseline_mean() -> None:
    percentage_metric = metric(
        direction=MetricDirection.LOWER,
        threshold_type=ThresholdType.PERCENTAGE,
        threshold=10.0,
    )

    with pytest.raises(ValueError, match="baseline mean"):
        calculate_power_analysis(
            percentage_metric,
            expected_standard_deviation=4.0,
        )

    with pytest.raises(ValueError, match="baseline mean"):
        calculate_power_analysis(
            percentage_metric,
            expected_standard_deviation=4.0,
            expected_baseline_mean=0.0,
        )


def test_minimum_detectable_effect_rejects_boolean_parameters() -> None:
    from golf_performance.power import calculate_minimum_detectable_effect

    with pytest.raises(ValueError):
        calculate_minimum_detectable_effect(
            expected_standard_deviation=2.0,
            shots_per_configuration=10,
            target_power=True,
        )


def test_power_analysis_rejects_boolean_numeric_inputs() -> None:
    with pytest.raises(ValueError):
        calculate_power_analysis(metric(), expected_standard_deviation=True)

    with pytest.raises(ValueError):
        calculate_power_analysis(metric(), expected_standard_deviation=2.0, block_size=True)


def test_pilot_sd_controls_sample_size_power_and_mde() -> None:
    from scipy.stats import nct, t
    from golf_performance.power import PowerAnalysis

    result = calculate_power_analysis(metric(), expected_block_difference_standard_deviation=8.0)
    k = result.recommended_block_pairs
    critical = t.ppf(0.975, k - 1)
    noncentrality = 3.0 / 8.0 * math.sqrt(k)
    independent_power = nct.sf(critical, k - 1, noncentrality) + nct.cdf(-critical, k - 1, noncentrality)
    assert result.achieved_power == pytest.approx(independent_power)
    assert result.achieved_power >= 0.8
    assert result.expected_standard_deviation is None
    assert result.variability_source == "pilot_block_difference_sd"
    assert PowerAnalysis.from_dict(result.to_dict()) == result
    mde = calculate_minimum_detectable_effect(
        expected_block_difference_standard_deviation=8.0,
        shots_per_configuration=k * 5,
    )
    assert mde <= 3.0
    assert mde > 2.9
    larger = calculate_power_analysis(metric(), expected_block_difference_standard_deviation=12.0)
    assert larger.recommended_block_pairs > k
    # A pilot SD is already for this block size; do not divide it by sqrt(block size).
    other = calculate_power_analysis(metric(), expected_block_difference_standard_deviation=8.0, block_size=10)
    assert other.recommended_block_pairs == k


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_pilot_sd_rejected(value) -> None:
    with pytest.raises(ValueError, match="SD"):
        calculate_power_analysis(metric(), expected_block_difference_standard_deviation=value)
    with pytest.raises(ValueError, match="SD"):
        calculate_minimum_detectable_effect(expected_block_difference_standard_deviation=value, shots_per_configuration=30)


def test_old_power_payload_retains_shot_approximation() -> None:
    from golf_performance.power import PowerAnalysis
    result = calculate_power_analysis(metric(), expected_standard_deviation=6.0)
    payload = result.to_dict()
    payload.pop("variability_source")
    assert PowerAnalysis.from_dict(payload) == result
    assert any("zero covariance" in warning for warning in result.warnings)
