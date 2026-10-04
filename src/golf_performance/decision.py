from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .domain import Experiment, MetricDefinition, MetricDirection, ThresholdType
from .statistics import MetricAnalysis


DEFAULT_ALPHA = 0.05


class DecisionCategory(StrEnum):
    STRONG_MEANINGFUL_IMPROVEMENT = "strong_meaningful_improvement"
    MEASURABLE_NOT_PRACTICALLY_MEANINGFUL = "measurable_not_practically_meaningful"
    PROMISING_BUT_UNCERTAIN = "promising_but_uncertain"
    NO_USEFUL_IMPROVEMENT = "no_useful_improvement"
    POTENTIALLY_DETRIMENTAL = "potentially_detrimental"
    DETRIMENTAL_BUT_UNCERTAIN = "detrimental_but_uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class MetricDecision:
    metric_key: str
    metric_name: str
    category: DecisionCategory
    statistically_significant: bool | None
    practically_meaningful: bool | None
    observed_improvement: float
    observed_improvement_percentage: float | None
    practical_threshold: float
    threshold_type: ThresholdType
    interpretation: str
    warnings: tuple[str, ...] = ()
    confirmatory: bool = True


@dataclass(frozen=True)
class ExperimentDecision:
    primary: MetricDecision
    metrics: tuple[MetricDecision, ...]
    conclusion: str
    tradeoff_metric_names: tuple[str, ...] = ()


def _improvement_sign(metric: MetricDefinition) -> float:
    return 1.0 if metric.direction is MetricDirection.HIGHER else -1.0


def _category_interpretation(
    category: DecisionCategory,
    *,
    metric_name: str,
    observed_improvement: float,
    practical_threshold: float,
    threshold_type: ThresholdType,
) -> str:
    threshold_label = (
        f"{practical_threshold:g}"
        + ("%" if threshold_type is ThresholdType.PERCENTAGE else " units")
    )
    if category is DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT:
        return f"{metric_name} shows statistical evidence of an improvement that exceeds the predefined practical threshold of {threshold_label}."
    if category is DecisionCategory.MEASURABLE_NOT_PRACTICALLY_MEANINGFUL:
        return f"{metric_name} shows a statistically detectable change, but the observed improvement ({observed_improvement:.2f}) is below the predefined practical threshold of {threshold_label}."
    if category is DecisionCategory.PROMISING_BUT_UNCERTAIN:
        return f"{metric_name} exceeded the practical threshold, but the statistical evidence is uncertain; more data may be needed."
    if category is DecisionCategory.POTENTIALLY_DETRIMENTAL:
        return f"{metric_name} changed in an unfavorable direction by more than the practical threshold of {threshold_label}, with statistical evidence of deterioration."
    if category is DecisionCategory.DETRIMENTAL_BUT_UNCERTAIN:
        return f"{metric_name} changed in a potentially unfavorable direction by more than the practical threshold, but the evidence is uncertain."
    if category is DecisionCategory.INSUFFICIENT_DATA:
        return f"There is not enough valid data to evaluate {metric_name} statistically and practically."
    return f"The observed change in {metric_name} does not provide evidence of a useful improvement beyond the predefined threshold."


def evaluate_metric_decision(
    analysis: MetricAnalysis,
    metric: MetricDefinition,
    *,
    alpha: float = DEFAULT_ALPHA,
    confirmatory: bool = True,
) -> MetricDecision:
    """Separate statistical significance from direction-aware practical significance."""

    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("Alpha must be between zero and one")
    if analysis.metric_key != metric.key:
        raise ValueError("Analysis and metric definitions must refer to the same metric")
    if not isinstance(confirmatory, bool):
        raise ValueError("Confirmatory flag must be boolean")

    sign = _improvement_sign(metric)
    observed_improvement = analysis.difference * sign
    observed_improvement_percentage = (
        analysis.percentage_difference * sign
        if analysis.percentage_difference is not None
        else None
    )
    statistically_significant = (
        analysis.p_value < alpha if analysis.p_value is not None else None
    )
    warnings: list[str] = []
    if not confirmatory:
        warnings.append(
            "Secondary metric result is exploratory; its nominal p-value is not confirmatory evidence."
        )

    if metric.threshold_type is ThresholdType.ABSOLUTE:
        practical_value = observed_improvement
    else:
        if observed_improvement_percentage is None:
            practical_value = None
            warnings.append(
                "Percentage practical significance cannot be evaluated because the baseline analysis mean is zero."
            )
        else:
            practical_value = observed_improvement_percentage

    practically_meaningful: bool | None
    if practical_value is None:
        practically_meaningful = None
    else:
        practically_meaningful = practical_value >= metric.meaningful_threshold

    adverse = (
        practical_value is not None
        and practical_value <= -metric.meaningful_threshold
    )
    if statistically_significant is None or practically_meaningful is None:
        category = DecisionCategory.INSUFFICIENT_DATA
    elif adverse and statistically_significant:
        category = DecisionCategory.POTENTIALLY_DETRIMENTAL
    elif adverse:
        category = DecisionCategory.DETRIMENTAL_BUT_UNCERTAIN
    elif practically_meaningful and statistically_significant:
        category = DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT
    elif practically_meaningful:
        category = DecisionCategory.PROMISING_BUT_UNCERTAIN
    elif statistically_significant:
        category = DecisionCategory.MEASURABLE_NOT_PRACTICALLY_MEANINGFUL
    else:
        category = DecisionCategory.NO_USEFUL_IMPROVEMENT

    return MetricDecision(
        metric_key=metric.key,
        metric_name=metric.display_name,
        category=category,
        statistically_significant=statistically_significant,
        practically_meaningful=practically_meaningful,
        observed_improvement=observed_improvement,
        observed_improvement_percentage=observed_improvement_percentage,
        practical_threshold=metric.meaningful_threshold,
        threshold_type=metric.threshold_type,
        interpretation=(
            "Exploratory result: "
            if not confirmatory
            else ""
        ) + _category_interpretation(
            category,
            metric_name=metric.display_name,
            observed_improvement=observed_improvement,
            practical_threshold=metric.meaningful_threshold,
            threshold_type=metric.threshold_type,
        ),
        warnings=tuple(warnings),
        confirmatory=confirmatory,
    )


def evaluate_experiment(
    analyses: tuple[MetricAnalysis, ...],
    experiment: Experiment,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> ExperimentDecision:
    """Evaluate metrics in experiment order and summarize primary-metric tradeoffs."""

    experiment.validate()
    if len(analyses) != len(experiment.metrics):
        raise ValueError("Analyses must contain exactly the experiment's configured metrics")
    by_key = {analysis.metric_key: analysis for analysis in analyses}
    if set(by_key) != {metric.key for metric in experiment.metrics}:
        raise ValueError("Analyses must contain exactly the experiment's configured metrics")

    metric_decisions = tuple(
        evaluate_metric_decision(
            by_key[metric.key],
            metric,
            alpha=alpha,
            confirmatory=metric.key == experiment.primary_metric.key,
        )
        for metric in experiment.metrics
    )
    primary = next(
        decision for decision in metric_decisions if decision.metric_key == experiment.primary_metric.key
    )
    tradeoffs = tuple(
        decision.metric_name
        for decision in metric_decisions
        if decision.metric_key != primary.metric_key
        and decision.category
        in {
            DecisionCategory.POTENTIALLY_DETRIMENTAL,
            DecisionCategory.DETRIMENTAL_BUT_UNCERTAIN,
        }
    )

    conclusion = primary.interpretation
    if tradeoffs:
        conclusion += " Potential tradeoffs were identified in: " + ", ".join(tradeoffs) + "."
    return ExperimentDecision(
        primary=primary,
        metrics=metric_decisions,
        conclusion=conclusion,
        tradeoff_metric_names=tradeoffs,
    )
