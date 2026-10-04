from __future__ import annotations

from dataclasses import dataclass

from ..domain import (
    Experiment,
    MetricAnalysisTransform,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)


@dataclass(frozen=True)
class MetricPreset:
    key: str
    display_name: str
    unit: str
    direction: MetricDirection
    transform: MetricAnalysisTransform


METRIC_PRESETS: tuple[MetricPreset, ...] = (
    MetricPreset("carry", "Carry", "yards", MetricDirection.HIGHER, MetricAnalysisTransform.RAW),
    MetricPreset("ball_speed", "Ball speed", "mph", MetricDirection.HIGHER, MetricAnalysisTransform.RAW),
    MetricPreset("offline", "Offline distance", "yards", MetricDirection.LOWER, MetricAnalysisTransform.ABSOLUTE_VALUE),
    MetricPreset("launch_angle", "Launch angle", "degrees", MetricDirection.TARGET, MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET),
    MetricPreset("spin", "Spin", "rpm", MetricDirection.TARGET, MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET),
)

PRESETS_BY_KEY = {preset.key: preset for preset in METRIC_PRESETS}


def metric_display_name(value: str) -> str:
    """Return a stable display label for either a preset key or label."""

    preset = PRESETS_BY_KEY.get(value)
    if preset is not None:
        return preset.display_name
    for preset in METRIC_PRESETS:
        if preset.display_name == value:
            return preset.display_name
    raise ValueError(f"Unknown metric value: {value}")


def build_metric_definition(
    *,
    key: str,
    threshold: float,
    threshold_type: ThresholdType,
    target_value: float | None = None,
    is_primary: bool = False,
) -> MetricDefinition:
    """Build a validated metric definition from UI-friendly values."""

    if key not in PRESETS_BY_KEY:
        raise ValueError("Unknown metric preset")
    preset = PRESETS_BY_KEY[key]
    return MetricDefinition(
        key=preset.key,
        display_name=preset.display_name,
        unit=preset.unit,
        direction=preset.direction,
        meaningful_threshold=threshold,
        threshold_type=threshold_type,
        target_value=target_value,
        is_primary=is_primary,
        analysis_transform=preset.transform,
    )


def build_experiment(
    *,
    name: str,
    changed_variable: str,
    baseline_value: str,
    treatment_value: str,
    primary_goal: str,
    metrics: list[MetricDefinition],
) -> Experiment:
    """Build the domain experiment used by the Streamlit workflow."""

    return Experiment(
        name=name,
        changed_variable=changed_variable,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        primary_goal=primary_goal,
        metrics=metrics,
    )
