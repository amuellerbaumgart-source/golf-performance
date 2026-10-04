import pandas as pd
import pytest

from golf_performance.decision import (
    DecisionCategory,
    evaluate_experiment,
    evaluate_metric_decision,
)
from golf_performance.domain import (
    Experiment,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)
from golf_performance.statistics import analyze_experiment


def build_metric(
    key: str = "carry",
    *,
    direction: MetricDirection = MetricDirection.HIGHER,
    threshold: float = 3.0,
    threshold_type: ThresholdType = ThresholdType.ABSOLUTE,
) -> MetricDefinition:
    return MetricDefinition(
        key=key,
        display_name=key.title(),
        unit="units",
        direction=direction,
        meaningful_threshold=threshold,
        threshold_type=threshold_type,
        is_primary=True,
    )


def build_experiment(metric: MetricDefinition) -> Experiment:
    return Experiment(
        name="Decision test",
        changed_variable="Loft",
        baseline_value="9",
        treatment_value="10",
        primary_goal="Improve performance",
        metrics=[metric],
    )


def test_statistical_and_practical_significance_produce_strong_decision() -> None:
    metric = build_metric()
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 10 + ["B"] * 10,
            "carry": [100, 101, 99, 100, 101, 99, 100, 101, 99, 100]
            + [105, 106, 104, 105, 106, 104, 105, 106, 104, 105],
        }
    )

    analysis = analyze_experiment(results, build_experiment(metric))[0]
    decision = evaluate_metric_decision(analysis, metric)

    assert decision.category is DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT
    assert decision.statistically_significant is True
    assert decision.practically_meaningful is True


def test_lower_is_better_uses_the_correct_improvement_direction() -> None:
    metric = build_metric(key="offline", direction=MetricDirection.LOWER, threshold=10.0)
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 10 + ["B"] * 10,
            "offline": [20, 21, 19, 20, 21, 19, 20, 21, 19, 20]
            + [15, 16, 14, 15, 16, 14, 15, 16, 14, 15],
        }
    )

    analysis = analyze_experiment(results, build_experiment(metric))[0]
    decision = evaluate_metric_decision(analysis, metric)

    assert decision.observed_improvement == pytest.approx(5.0)
    assert decision.practically_meaningful is False


def test_percentage_threshold_uses_direction_adjusted_percentage() -> None:
    metric = build_metric(
        key="offline",
        direction=MetricDirection.LOWER,
        threshold=10.0,
        threshold_type=ThresholdType.PERCENTAGE,
    )
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 10 + ["B"] * 10,
            "offline": [20.0] * 10 + [17.0] * 10,
        }
    )

    analysis = analyze_experiment(results, build_experiment(metric))[0]
    decision = evaluate_metric_decision(analysis, metric)

    assert decision.observed_improvement_percentage == pytest.approx(15.0)
    assert decision.practically_meaningful is True


def test_non_significant_practically_large_effect_is_promising_uncertain() -> None:
    metric = build_metric()
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B"],
            "carry": [100.0, 120.0, 104.0, 124.0],
        }
    )

    analysis = analyze_experiment(results, build_experiment(metric))[0]
    decision = evaluate_metric_decision(analysis, metric)

    assert decision.category is DecisionCategory.PROMISING_BUT_UNCERTAIN


def test_experiment_decision_reports_secondary_tradeoffs() -> None:
    primary = build_metric()
    secondary = build_metric(key="offline", direction=MetricDirection.LOWER, threshold=3.0)
    secondary = MetricDefinition(**{**secondary.__dict__, "is_primary": False})
    experiment = Experiment(
        name="Tradeoff test",
        changed_variable="Loft",
        baseline_value="9",
        treatment_value="10",
        primary_goal="More carry without more offline",
        metrics=[primary, secondary],
    )
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 10 + ["B"] * 10,
            "carry": [99, 100, 101, 99, 100, 101, 99, 100, 101, 100]
            + [104, 105, 106, 104, 105, 106, 104, 105, 106, 105],
            "offline": [4, 5, 6, 4, 5, 6, 4, 5, 6, 5]
            + [9, 10, 11, 9, 10, 11, 9, 10, 11, 10],
        }
    )

    decision = evaluate_experiment(analyze_experiment(results, experiment), experiment)

    assert decision.primary.category is DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT
    assert decision.tradeoff_metric_names == ("Offline",)
    assert "tradeoff" in decision.conclusion.lower()
    secondary_decision = next(item for item in decision.metrics if item.metric_key == "offline")
    assert secondary_decision.confirmatory is False
    assert any("not confirmatory" in warning for warning in secondary_decision.warnings)


def test_experiment_decision_rejects_duplicate_analysis_records() -> None:
    metric = build_metric()
    experiment = build_experiment(metric)
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B"],
            "carry": [100.0, 101.0, 105.0, 106.0],
        }
    )
    analysis = analyze_experiment(results, experiment)[0]

    with pytest.raises(ValueError, match="exactly"):
        evaluate_experiment((analysis, analysis), experiment)


def test_metric_decision_rejects_invalid_alpha() -> None:
    metric = build_metric()
    experiment = build_experiment(metric)
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B"],
            "carry": [100.0, 101.0, 105.0, 106.0],
        }
    )
    analysis = analyze_experiment(results, experiment)[0]

    with pytest.raises(ValueError, match="Alpha"):
        evaluate_metric_decision(analysis, metric, alpha=True)
