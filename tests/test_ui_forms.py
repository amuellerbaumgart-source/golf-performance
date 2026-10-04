import pytest

from golf_performance.domain import (
    MetricAnalysisTransform,
    MetricDirection,
    ThresholdType,
)
from golf_performance.ui.forms import build_experiment, build_metric_definition, metric_display_name


def test_build_metric_definition_preserves_ui_metric_semantics() -> None:
    metric = build_metric_definition(
        key="offline",
        threshold=10.0,
        threshold_type=ThresholdType.PERCENTAGE,
        is_primary=True,
    )

    assert metric.direction is MetricDirection.LOWER
    assert metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_VALUE
    assert metric.threshold_type is ThresholdType.PERCENTAGE


def test_build_target_metric_requires_and_preserves_target() -> None:
    metric = build_metric_definition(
        key="launch_angle",
        threshold=1.0,
        threshold_type=ThresholdType.ABSOLUTE,
        target_value=15.0,
        is_primary=True,
    )

    assert metric.direction is MetricDirection.TARGET
    assert metric.target_value == 15.0
    assert metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET


def test_build_metric_definition_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        build_metric_definition(
            key="unknown",
            threshold=1.0,
            threshold_type=ThresholdType.ABSOLUTE,
        )


def test_build_experiment_creates_valid_domain_object() -> None:
    metric = build_metric_definition(
        key="carry",
        threshold=3.0,
        threshold_type=ThresholdType.ABSOLUTE,
        is_primary=True,
    )
    experiment = build_experiment(
        name="Loft test",
        changed_variable="Loft",
        baseline_value="9 degrees",
        treatment_value="10 degrees",
        primary_goal="Increase carry",
        metrics=[metric],
    )

    assert experiment.primary_metric.key == "carry"


def test_metric_display_name_accepts_keys_and_labels() -> None:
    assert metric_display_name("carry") == "Carry"
    assert metric_display_name("Carry") == "Carry"
