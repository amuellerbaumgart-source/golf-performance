from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_collection import completed_shots, normalize_saved_results, validate_results
from .decision import ExperimentDecision, evaluate_experiment
from .domain import Experiment
from .protocols import TestingProtocol
from .statistics import MetricAnalysis, analyze_experiment


class ReportNotReadyError(ValueError):
    """Raised when a complete, protocol-matching report cannot be built."""


@dataclass(frozen=True)
class ExperimentReport:
    experiment: Experiment
    protocol: TestingProtocol
    analyses: tuple[MetricAnalysis, ...]
    decision: ExperimentDecision
    total_valid_shots: int


def build_experiment_report(
    experiment: Experiment,
    protocol: TestingProtocol,
    results: pd.DataFrame,
) -> ExperimentReport:
    """Build a report from complete results and existing analysis services."""

    metric_keys = [metric.key for metric in experiment.metrics]
    results = normalize_saved_results(results, protocol, metric_keys)
    validation_errors = validate_results(results, protocol, metric_keys)
    if validation_errors:
        raise ReportNotReadyError("; ".join(validation_errors))
    valid_shots = completed_shots(results, metric_keys)
    if valid_shots != protocol.total_shots:
        raise ReportNotReadyError(
            f"Report requires {protocol.total_shots} complete shots; found {valid_shots}."
        )

    analyses = analyze_experiment(results, experiment, protocol)
    decision = evaluate_experiment(analyses, experiment)
    return ExperimentReport(
        experiment=experiment,
        protocol=protocol,
        analyses=analyses,
        decision=decision,
        total_valid_shots=valid_shots,
    )


def build_report_export(report: ExperimentReport) -> pd.DataFrame:
    """Create one CSV-friendly row per analyzed metric."""

    experiment = report.experiment
    decision_by_key = {item.metric_key: item for item in report.decision.metrics}
    rows: list[dict[str, object]] = []
    for analysis in report.analyses:
        decision = decision_by_key[analysis.metric_key]
        rows.append(
            {
                "experiment_id": str(experiment.experiment_id),
                "experiment_name": experiment.name,
                "changed_variable": experiment.changed_variable,
                "baseline_value": experiment.baseline_value,
                "treatment_value": experiment.treatment_value,
                "protocol_mode": report.protocol.label,
                "metric": analysis.metric_name,
                "unit": analysis.unit,
                "analysis_basis": analysis.analysis_basis,
                "baseline_n": analysis.baseline.n,
                "treatment_n": analysis.treatment.n,
                "analysis_unit": analysis.analysis_unit,
                "statistical_method": analysis.statistical_method,
                "paired_block_count": analysis.n_pairs,
                "baseline_mean": analysis.baseline.mean,
                "treatment_mean": analysis.treatment.mean,
                "difference": analysis.difference,
                "percentage_difference": analysis.percentage_difference,
                "confidence_interval_lower": analysis.confidence_interval_lower,
                "confidence_interval_upper": analysis.confidence_interval_upper,
                "bootstrap_confidence_interval_lower": analysis.bootstrap_confidence_interval_lower,
                "bootstrap_confidence_interval_upper": analysis.bootstrap_confidence_interval_upper,
                "p_value": analysis.p_value,
                "hedges_g": analysis.hedges_g,
                "practical_threshold": decision.practical_threshold,
                "threshold_type": decision.threshold_type.value,
                "statistically_significant": decision.statistically_significant,
                "practically_meaningful": decision.practically_meaningful,
                "evidence_role": "Confirmatory" if decision.confirmatory else "Exploratory",
                "decision": decision.category.value,
            }
        )
    return pd.DataFrame(rows)
