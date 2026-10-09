from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from .domain import (
    Experiment,
    MetricAnalysisTransform,
    MetricDefinition,
)

if TYPE_CHECKING:
    from .protocols import TestingProtocol


CONFIDENCE_LEVEL = 0.95
STATISTICAL_METHOD = "welch_independent_two_sample_t"
ANALYSIS_UNIT = "individual_shot"
PAIRED_STATISTICAL_METHOD = "paired_block_t"
PAIRED_ANALYSIS_UNIT = "paired_block"
BOOTSTRAP_MIN_PAIRS = 5
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 2025


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
    statistical_method: str = STATISTICAL_METHOD
    analysis_unit: str = ANALYSIS_UNIT
    n_pairs: int = 0
    paired_difference_standard_deviation: float | None = None
    bootstrap_confidence_interval_lower: float | None = None
    bootstrap_confidence_interval_upper: float | None = None

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


def transform_metric_values(values: pd.Series, metric: MetricDefinition) -> pd.Series:
    """Return finite values on the same analysis scale used by the statistics engine."""

    return _analysis_values(values, metric)


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


def _paired_statistics(
    paired_differences: pd.Series,
) -> tuple[float | None, float | None, float | None, float | None, tuple[str, ...]]:
    """Return a paired-t CI, p-value, and paired Hedges' g."""

    if len(paired_differences) < 2:
        return (
            None,
            None,
            None,
            None,
            ("At least two complete A/B block pairs are required for inferential statistics.",),
        )

    effect_size_available = len(paired_differences) >= 3
    effect_size_warnings = (() if effect_size_available else (
        "Paired Hedges' g is unavailable: at least three complete A/B block pairs are required. "
        "With two pairs, the small-sample correction collapses to zero and is not a valid effect-size estimate.",
    ))
    difference = float(paired_differences.mean())
    standard_deviation = float(paired_differences.std(ddof=1))
    if standard_deviation == 0:
        if difference == 0:
            return 0.0, 0.0, 1.0, (0.0 if effect_size_available else None), effect_size_warnings
        return (
            None,
            None,
            None,
            None,
            ("Paired block differences have zero variability, so conventional paired inferential statistics are undefined.",) + effect_size_warnings,
        )

    degrees_of_freedom = len(paired_differences) - 1
    standard_error = standard_deviation / math.sqrt(len(paired_differences))
    t_statistic = difference / standard_error
    critical_value = float(
        stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, degrees_of_freedom)
    )
    margin = critical_value * standard_error
    p_value = float(2 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
    hedges_g = None
    if effect_size_available:
        cohen_d_z = difference / standard_deviation
        correction = 1 - 3 / (4 * degrees_of_freedom - 1)
        corrected_effect = cohen_d_z * correction
        hedges_g = corrected_effect if math.isfinite(corrected_effect) else None
    return (
        difference - margin,
        difference + margin,
        p_value if math.isfinite(p_value) else None,
        hedges_g,
        effect_size_warnings,
    )


def _paired_bootstrap_interval(
    paired_differences: pd.Series,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a reproducible percentile bootstrap interval for paired means."""

    if len(paired_differences) < BOOTSTRAP_MIN_PAIRS:
        raise ValueError(
            f"At least {BOOTSTRAP_MIN_PAIRS} complete block pairs are required for bootstrap sensitivity analysis"
        )
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("Bootstrap resamples must be an integer of at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Bootstrap seed must be an integer")
    values = paired_differences.to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    sampled_means = generator.choice(
        values,
        size=(resamples, len(values)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.percentile(sampled_means, [2.5, 97.5])
    return float(lower), float(upper)


def _paired_block_differences(
    results: pd.DataFrame,
    metric: MetricDefinition,
    protocol: "TestingProtocol",
) -> tuple[pd.Series, tuple[str, ...]]:
    """Calculate B-minus-A means for complete adjacent protocol block pairs."""

    if len(results) != protocol.total_shots:
        raise ValueError(
            f"Results must contain exactly {protocol.total_shots} rows for the selected protocol"
        )
    configurations = results["configuration"].astype("string").tolist()
    if configurations != list(protocol.sequence):
        raise ValueError("Results configuration order does not match the selected protocol")

    differences: list[float] = []
    warnings: list[str] = []
    for pair_index in range(protocol.blocks_per_configuration):
        first_start = pair_index * 2 * protocol.block_size
        first_end = first_start + protocol.block_size
        second_start = first_end
        second_end = second_start + protocol.block_size
        first_configuration = protocol.sequence[first_start]
        second_configuration = protocol.sequence[second_start]
        first_values = _analysis_values(
            results.iloc[first_start:first_end][metric.key], metric
        )
        second_values = _analysis_values(
            results.iloc[second_start:second_end][metric.key], metric
        )
        if len(first_values) != protocol.block_size or len(second_values) != protocol.block_size:
            warnings.append(
                f"Block pair {pair_index + 1} was excluded because both blocks need {protocol.block_size} valid {metric.display_name} observations."
            )
            continue
        if first_configuration == "A" and second_configuration == "B":
            baseline_mean = float(first_values.mean())
            treatment_mean = float(second_values.mean())
        elif first_configuration == "B" and second_configuration == "A":
            treatment_mean = float(first_values.mean())
            baseline_mean = float(second_values.mean())
        else:
            raise ValueError("Protocol must contain adjacent A/B block pairs")
        differences.append(treatment_mean - baseline_mean)
    return pd.Series(differences, dtype="float64"), tuple(warnings)


def _analysis_basis(metric: MetricDefinition) -> str:
    if metric.analysis_transform is MetricAnalysisTransform.RAW:
        return "raw metric values"
    if metric.analysis_transform is MetricAnalysisTransform.ABSOLUTE_VALUE:
        return "absolute metric values"
    return "absolute error from target"


def analyze_metric(
    results: pd.DataFrame,
    metric: MetricDefinition,
    protocol: "TestingProtocol | None" = None,
) -> MetricAnalysis:
    """Analyze one metric, using paired blocks when a protocol is supplied.

    Without a protocol this retains the legacy independent-shot Welch analysis
    for compatibility with previously stored or directly analyzed data.
    """

    required_columns = {"configuration", metric.key}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Results are missing required columns: {missing}")

    baseline = _analysis_values(results.loc[results["configuration"] == "A", metric.key], metric)
    treatment = _analysis_values(results.loc[results["configuration"] == "B", metric.key], metric)
    warnings: list[str] = []
    if "exclusion_note" in results.columns:
        note_mask = results["exclusion_note"].fillna("").astype(str).str.strip().ne("")
        noted_shots = int(note_mask.sum())
        if noted_shots:
            warnings.append(
                f"{noted_shots} shot(s) have exclusion notes; review those observations before interpreting the result."
            )
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
    shot_difference = treatment_summary.mean - baseline_summary.mean
    difference = shot_difference
    if protocol is None:
        warnings.append(
            "Welch statistics treat shots as independent observations; repeated shots from one golfer may violate that assumption."
        )
        ci_lower, ci_upper, p_value, hedges_g, inferential_warnings = _welch_statistics(
            baseline, treatment
        )
        statistical_method = STATISTICAL_METHOD
        analysis_unit = ANALYSIS_UNIT
        n_pairs = 0
        paired_difference_standard_deviation = None
        bootstrap_confidence_interval_lower = None
        bootstrap_confidence_interval_upper = None
    else:
        paired_differences, block_warnings = _paired_block_differences(
            results, metric, protocol
        )
        warnings.extend(block_warnings)
        if not paired_differences.empty:
            difference = float(paired_differences.mean())
        ci_lower, ci_upper, p_value, hedges_g, inferential_warnings = _paired_statistics(
            paired_differences
        )
        statistical_method = PAIRED_STATISTICAL_METHOD
        analysis_unit = PAIRED_ANALYSIS_UNIT
        n_pairs = int(len(paired_differences))
        paired_difference_standard_deviation = (
            float(paired_differences.std(ddof=1)) if n_pairs >= 2 else None
        )
        if n_pairs >= BOOTSTRAP_MIN_PAIRS:
            (
                bootstrap_confidence_interval_lower,
                bootstrap_confidence_interval_upper,
            ) = _paired_bootstrap_interval(paired_differences)
        else:
            bootstrap_confidence_interval_lower = None
            bootstrap_confidence_interval_upper = None
            warnings.append(
                f"Bootstrap sensitivity interval unavailable: at least {BOOTSTRAP_MIN_PAIRS} complete A/B block pairs are required."
            )
    warnings.extend(inferential_warnings)
    percentage_difference = (
        difference / abs(baseline_summary.mean) * 100
        if baseline_summary.mean != 0
        else None
    )
    if baseline_summary.mean == 0:
        warnings.append(
            "Percentage difference is unavailable because the baseline mean is zero."
        )

    return MetricAnalysis(
        metric_key=metric.key,
        metric_name=metric.display_name,
        unit=metric.unit,
        analysis_basis=_analysis_basis(metric),
        baseline=baseline_summary,
        treatment=treatment_summary,
        difference=difference,
        percentage_difference=percentage_difference,
        confidence_interval_lower=ci_lower,
        confidence_interval_upper=ci_upper,
        p_value=p_value,
        hedges_g=hedges_g,
        warnings=tuple(dict.fromkeys(warnings)),
        statistical_method=statistical_method,
        analysis_unit=analysis_unit,
        n_pairs=n_pairs,
        paired_difference_standard_deviation=paired_difference_standard_deviation,
        bootstrap_confidence_interval_lower=bootstrap_confidence_interval_lower,
        bootstrap_confidence_interval_upper=bootstrap_confidence_interval_upper,
    )


def analyze_experiment(
    results: pd.DataFrame,
    experiment: Experiment,
    protocol: "TestingProtocol | None" = None,
) -> tuple[MetricAnalysis, ...]:
    """Analyze every configured metric in experiment order."""

    experiment.validate()
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame")
    return tuple(analyze_metric(results, metric, protocol) for metric in experiment.metrics)
