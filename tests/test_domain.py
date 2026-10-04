from datetime import datetime, timezone

import pytest

from golf_performance.domain import (
    Experiment,
    ExperimentStatus,
    MetricAnalysisTransform,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)


def carry_metric(*, primary: bool = True) -> MetricDefinition:
    return MetricDefinition(
        key="carry",
        display_name="Carry",
        unit="yards",
        direction=MetricDirection.HIGHER,
        meaningful_threshold=3.0,
        is_primary=primary,
    )


def valid_experiment() -> Experiment:
    return Experiment(
        name="Driver loft test",
        changed_variable="Driver loft",
        baseline_value="9 degrees",
        treatment_value="10 degrees",
        primary_goal="Increase carry without worsening dispersion",
        metrics=[carry_metric()],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_valid_experiment_can_be_created() -> None:
    experiment = valid_experiment()

    assert experiment.status is ExperimentStatus.DRAFT
    assert experiment.metrics[0].key == "carry"


@pytest.mark.parametrize("field_name", [
    "name",
    "changed_variable",
    "baseline_value",
    "treatment_value",
    "primary_goal",
])
def test_required_text_fields_cannot_be_empty(field_name: str) -> None:
    values = {
        "name": "Experiment",
        "changed_variable": "Loft",
        "baseline_value": "9 degrees",
        "treatment_value": "10 degrees",
        "primary_goal": "More carry",
        "metrics": [carry_metric()],
    }
    values[field_name] = " "

    with pytest.raises(ValueError, match="cannot be empty"):
        Experiment(**values)


def test_baseline_and_treatment_must_differ() -> None:
    with pytest.raises(ValueError, match="must be different"):
        Experiment(
            name="Test",
            changed_variable="Loft",
            baseline_value="10 degrees",
            treatment_value="10 degrees",
            primary_goal="More carry",
            metrics=[carry_metric()],
        )


def test_exactly_one_primary_metric_is_required() -> None:
    with pytest.raises(ValueError, match="Exactly one primary"):
        Experiment(
            name="Test",
            changed_variable="Loft",
            baseline_value="9 degrees",
            treatment_value="10 degrees",
            primary_goal="More carry",
            metrics=[carry_metric(primary=False)],
        )


def test_target_metric_requires_target_value() -> None:
    with pytest.raises(ValueError, match="require a target"):
        MetricDefinition(
            key="launch",
            display_name="Launch angle",
            unit="degrees",
            direction=MetricDirection.TARGET,
            meaningful_threshold=1.0,
        )


def test_experiment_serialization_round_trip() -> None:
    original = valid_experiment()

    restored = Experiment.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_percentage_threshold_is_supported() -> None:
    metric = MetricDefinition(
        key="offline",
        display_name="Offline",
        unit="yards",
        direction=MetricDirection.LOWER,
        meaningful_threshold=10.0,
        threshold_type=ThresholdType.PERCENTAGE,
        is_primary=True,
    )

    assert metric.threshold_type is ThresholdType.PERCENTAGE


def test_target_metric_defaults_to_absolute_error_transform() -> None:
    metric = MetricDefinition(
        key="launch",
        display_name="Launch angle",
        unit="degrees",
        direction=MetricDirection.TARGET,
        meaningful_threshold=1.0,
        target_value=15.0,
        is_primary=True,
    )

    assert metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET


def test_target_metric_rejects_incompatible_transform() -> None:
    with pytest.raises(ValueError, match="absolute error"):
        MetricDefinition(
            key="launch",
            display_name="Launch angle",
            unit="degrees",
            direction=MetricDirection.TARGET,
            meaningful_threshold=1.0,
            target_value=15.0,
            analysis_transform=MetricAnalysisTransform.RAW,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricDefinition(
            key="carry",
            display_name="Carry",
            unit="yards",
            direction=MetricDirection.HIGHER,
            meaningful_threshold=value,
            is_primary=True,
        )


def test_experiment_lifecycle_transitions_are_constrained() -> None:
    experiment = valid_experiment()

    experiment.transition_to(ExperimentStatus.PROTOCOL_READY)
    experiment.transition_to(ExperimentStatus.COLLECTING)

    with pytest.raises(ValueError, match="Cannot transition"):
        experiment.transition_to(ExperimentStatus.ANALYZED)


def test_malformed_experiment_data_has_domain_error() -> None:
    with pytest.raises(ValueError, match="Invalid experiment data"):
        Experiment.from_dict({"name": "incomplete"})


def test_experiment_rejects_non_datetime_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        Experiment(
            name="Test",
            changed_variable="Loft",
            baseline_value="9 degrees",
            treatment_value="10 degrees",
            primary_goal="More carry",
            metrics=[carry_metric()],
            created_at="2026-01-01",  # type: ignore[arg-type]
        )
