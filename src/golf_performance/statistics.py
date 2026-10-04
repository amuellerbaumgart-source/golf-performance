from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats

from .domain import (
    Experiment,
    MetricAnalysisTransform,
    MetricDefinition,
)


CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class ConditionSummary:
    configuration: str
    n: int
    mean: float
    median: float
    standard_deviation: float | None


@dataclass(frozen=True)
class MetricAnalysis:
    metric_key: str
    metric_name: str
    unit: str
    analysis_basis: str
    baseline: ConditionSummary
    treatment: ConditionSummary
    difference: float
    percentage_difference: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    p_value: float | None
    hedges_g: float | None
    warnings: tuple[str, ...] = ()

    @property
    def inferential_statistics_available(self) -> bool:
        return (
            self.p_value is not None
            and self.confidence_interval_lower is not None
            and self.confidence_interval_upper is not None
        )


def _analysis_values(values: pd.Series, metric: MetricDefinition) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric.notna() & numeric.map(math.isfinite)]
    if metric.analysis_transform is MetricAnalysisTransform.RAW:
        return numeric.astype(float)
    if metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_VALUE:
        return numeric.abs().astype(float)
    if metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET:
        if metric.target_value is None:
            raise ValueError(f"Metric {metric.key} requires a target value")
        return (numeric - metric.target_value).abs().astype(float)
    raise ValueError(f"Unsupported analysis transform for metric {metric.key}")


def _summary(configuration: str, values: pd.Series) -> ConditionSummary:
    if values.empty:
        raise ValueError(f"Configuration {configuration} has no valid observations")
    standard_deviation = float(values.std(ddof=1)) if len(values) >= 2 else None
    return ConditionSummary(
        configuration=configuration,
        n=int(len(values)),
        mean=float(values.mean()),
        median=float(values.median()),
        standard_deviation=standard_deviation,
    )


def _welch_statistics(
    baseline: pd.Series,
    treatment: pd.Series,
) -> tuple[float | None, float | None, float | None, float | None, tuple[str, ...]]:
    """Return CI, p-value, and Hedges' g for treatment minus baseline."""

    if len(baseline) < 2 or len(treatment) < 2:
        return (
            None,
            None,
            None,
            None,
            ("At least two valid observations per configuration are required for inferential statistics.",),
        )

    baseline_mean = float(baseline.mean())
    treatment_mean = float(treatment.mean())
    difference = treatment_mean - baseline_mean
    baseline_variance = float(baseline.var(ddof=1))
    treatment_variance = float(treatment.var(ddof=1))
    standard_error = math.sqrt(
        baseline_variance / len(baseline) + treatment_variance / len(treatment)
    )

    warnings: list[str] = []
    if standard_error == 0:
        if difference == 0:
            return 0.0, 0.0, 1.0, 0.0, ()
        warnings.append(
            "Both groups have zero observed variability, so conventional Welch inferential statistics are undefined."
        )
        return None, None, None, None, tuple(warnings)

    degrees_of_freedom_numerator = (
        baseline_variance / len(baseline) + treatment_variance / len(treatment)
    ) ** 2
    degrees_of_freedom_denominator = (
        (baseline_variance / len(baseline)) ** 2 / (len(baseline) - 1)
        + (treatment_variance / len(treatment)) ** 2 / (len(treatment) - 1)
    )
    degrees_of_freedom = degrees_of_freedom_numerator / degrees_of_freedom_denominator
    critical_value = float(stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, degrees_of_freedom))
    margin = critical_value * standard_error
    _, p_value = stats.ttest_ind(baseline, treatment, equal_var=False, alternative="two-sided")

    pooled_degrees_of_freedom = len(baseline) + len(treatment) - 2
    pooled_variance = (
        ((len(baseline) - 1) * baseline_variance)
        + ((len(treatment) - 1) * treatment_variance)
    ) / pooled_degrees_of_freedom
    pooled_standard_deviation = math.sqrt(pooled_variance)
    if pooled_standard_deviation == 0:
        hedges_g = None
        warnings.append("The pooled standard deviation is zero, so Hedges' g is undefined.")
    else:
        cohen_d = difference / pooled_standard_deviation
        correction = 1 - 3 / (4 * pooled_degrees_of_freedom - 1)
        hedges_g = cohen_d * correction

    return (
        difference - margin,
        difference + margin,
        float(p_value) if math.isfinite(float(p_value)) else None,
        hedges_g,
        tuple(warnings),
    )


def analyze_metric(results: pd.DataFrame, metric: MetricDefinition) -> MetricAnalysis:
    """Analyze one metric as treatment minus baseline using Welch's t-test."""

    required_columns = {"configuration", metric.key}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Results are missing required columns: {missing}")

    baseline = _analysis_values(results.loc[results["configuration"] == "A", metric.key], metric)
    treatment = _analysis_values(results.loc[results["configuration"] == "B", metric.key], metric)
    warnings: list[str] = [
        "Welch statistics treat shots as independent observations; repeated shots from one golfer may violate that assumption."
    ]
    observed_configurations = set(results["configuration"].dropna().astype(str))
    unexpected_configurations = sorted(observed_configurations.difference({"A", "B"}))
    if unexpected_configurations:
        warnings.append(
            "Rows with unexpected configurations were ignored: "
            + ", ".join(unexpected_configurations)
            + "."
        )
    for configuration, raw_values, valid_values in (
        ("A", results.loc[results["configuration"] == "A", metric.key], baseline),
        ("B", results.loc[results["configuration"] == "B", metric.key], treatment),
    ):
        if len(raw_values) > len(valid_values):
            warnings.append(
                f"{len(raw_values) - len(valid_values)} non-finite or non-numeric {configuration} observation(s) were excluded."
            )
    baseline_summary = _summary("A", baseline)
    treatment_summary = _summary("B", treatment)
    difference = treatment_summary.mean - baseline_summary.mean
    percentage_difference = (
        difference / abs(baseline_summary.mean) * 100
        if baseline_summary.mean != 0
        else None
    )
    ci_lower, ci_upper, p_value, hedges_g, inferential_warnings = _welch_statistics(
        baseline, treatment
    )
    warnings.extend(inferential_warnings)
    if baseline_summary.mean == 0:
        warnings.append(
            "Percentage difference is unavailable because the baseline mean is zero."
        )

    if metric.analysis_transform is MetricAnalysisTransform.RAW:
        analysis_basis = "raw metric values"
    elif metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_VALUE:
        analysis_basis = "absolute metric values"
    else:
        analysis_basis = "absolute error from target"

    return MetricAnalysis(
        metric_key=metric.key,
        metric_name=metric.display_name,
        unit=metric.unit,
        analysis_basis=analysis_basis,
        baseline=baseline_summary,
        treatment=treatment_summary,
        difference=difference,
        percentage_difference=percentage_difference,
        confidence_interval_lower=ci_lower,
        confidence_interval_upper=ci_upper,
        p_value=p_value,
        hedges_g=hedges_g,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def analyze_experiment(results: pd.DataFrame, experiment: Experiment) -> tuple[MetricAnalysis, ...]:
    """Analyze every configured metric in experiment order."""

    experiment.validate()
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame")
    return tuple(analyze_metric(results, metric) for metric in experiment.metrics)
