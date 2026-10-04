import pandas as pd
import pytest

from golf_performance.domain import (
    Experiment,
    MetricAnalysisTransform,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)
from golf_performance.statistics import analyze_experiment, analyze_metric


def metric(
    key: str = "carry",
    *,
    direction: MetricDirection = MetricDirection.HIGHER,
    transform: MetricAnalysisTransform = MetricAnalysisTransform.RAW,
    target_value: float | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        key=key,
        display_name=key.title(),
        unit="units",
        direction=direction,
        meaningful_threshold=3.0,
        threshold_type=ThresholdType.ABSOLUTE,
        target_value=target_value,
        is_primary=True,
        analysis_transform=transform,
    )


def test_analyze_metric_reports_descriptive_and_welch_statistics() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 4 + ["B"] * 4,
            "carry": [100.0, 102.0, 98.0, 101.0, 105.0, 107.0, 104.0, 106.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.n == 4
    assert analysis.treatment.n == 4
    assert analysis.difference == pytest.approx(5.25)
    assert analysis.percentage_difference == pytest.approx(5.25 / 100.25 * 100)
    assert analysis.confidence_interval_lower < analysis.difference < analysis.confidence_interval_upper
    assert analysis.p_value is not None
    assert analysis.hedges_g is not None


def test_analysis_transform_is_applied_before_comparison() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 4 + ["B"] * 4,
            "offline": [-10.0, 10.0, -8.0, 8.0, -5.0, 5.0, -4.0, 4.0],
        }
    )

    analysis = analyze_metric(
        results,
        metric(
            key="offline",
            direction=MetricDirection.LOWER,
            transform=MetricAnalysisTransform.ABSOLUTE_VALUE,
        ),
    )

    assert analysis.baseline.mean == pytest.approx(9.0)
    assert analysis.treatment.mean == pytest.approx(4.5)
    assert analysis.difference == pytest.approx(-4.5)


def test_target_analysis_uses_absolute_error_from_target() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 3 + ["B"] * 3,
            "launch": [14.0, 16.0, 15.5, 14.5, 15.5, 15.0],
        }
    )

    analysis = analyze_metric(
        results,
        metric(
            key="launch",
            direction=MetricDirection.TARGET,
            transform=MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET,
            target_value=15.0,
        ),
    )

    assert analysis.analysis_basis == "absolute error from target"
    assert analysis.baseline.mean == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert analysis.treatment.mean == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    assert analysis.difference < 0


def test_incomplete_groups_keep_descriptive_results_but_warn_about_inference() -> None:
    results = pd.DataFrame({"configuration": ["A", "B"], "carry": [100.0, 105.0]})

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.mean == 100.0
    assert analysis.treatment.mean == 105.0
    assert analysis.p_value is None
    assert analysis.confidence_interval_lower is None
    assert any("two valid observations" in warning for warning in analysis.warnings)


def test_invalid_and_missing_values_are_excluded_from_analysis() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "A", "B", "B", "B"],
            "carry": [100.0, float("nan"), "bad", 105.0, 106.0, float("inf")],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.n == 1
    assert analysis.treatment.n == 2
    assert analysis.p_value is None
    assert any("excluded" in warning for warning in analysis.warnings)
    assert any("independent observations" in warning for warning in analysis.warnings)


def test_zero_variance_difference_does_not_claim_a_valid_p_value() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 3 + ["B"] * 3,
            "carry": [100.0, 100.0, 100.0, 105.0, 105.0, 105.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.p_value is None
    assert analysis.confidence_interval_lower is None
    assert analysis.hedges_g is None
    assert any("zero observed variability" in warning for warning in analysis.warnings)


def test_unexpected_configurations_are_reported_and_ignored() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B", "C"],
            "carry": [100.0, 101.0, 105.0, 106.0, 999.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.n == 2
    assert analysis.treatment.n == 2
    assert any("unexpected configurations" in warning for warning in analysis.warnings)


def test_analyze_experiment_preserves_metric_order() -> None:
    metrics = [metric(), metric("offline", direction=MetricDirection.LOWER, transform=MetricAnalysisTransform.ABSOLUTE_VALUE)]
    experiment = Experiment(
        name="Test",
        changed_variable="Loft",
        baseline_value="9",
        treatment_value="10",
        primary_goal="Carry and dispersion",
        metrics=[metrics[0], MetricDefinition(**{**metrics[1].__dict__, "is_primary": False})],
    )
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B"],
            "carry": [100.0, 101.0, 105.0, 106.0],
            "offline": [-10.0, 10.0, -5.0, 5.0],
        }
    )

    analyses = analyze_experiment(results, experiment)

    assert [analysis.metric_key for analysis in analyses] == ["carry", "offline"]
