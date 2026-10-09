import pandas as pd
import pytest

from golf_performance.data_collection import build_results_template
from golf_performance.domain import (
    Experiment,
    ExperimentDesignMode,
    MetricDefinition,
    MetricDirection,
)
from golf_performance.protocols import generate_protocol_options
from golf_performance.reporting import (
    ReportNotReadyError,
    build_experiment_report,
    build_report_export,
)


def experiment() -> Experiment:
    return Experiment(
        name="Loft report test",
        changed_variable="Driver loft",
        baseline_value="9 degrees",
        treatment_value="10 degrees",
        primary_goal="Increase carry",
        metrics=[
            MetricDefinition(
                key="carry",
                display_name="Carry",
                unit="yards",
                direction=MetricDirection.HIGHER,
                meaningful_threshold=3.0,
                is_primary=True,
            )
        ],
    )


def selected_protocol(experiment: Experiment):
    return generate_protocol_options(
        experiment,
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=10,
        block_size=5,
    ).select(ExperimentDesignMode.EXPLORATORY).selected


def complete_results(protocol) -> pd.DataFrame:
    results = build_results_template(protocol, ["carry"])
    results["carry"] = [
        200.0 + (5.0 if configuration == "B" else 0.0)
        for configuration in results["configuration"]
    ]
    return results


def test_complete_experiment_builds_report_and_export() -> None:
    current = experiment()
    protocol = selected_protocol(current)
    report = build_experiment_report(current, protocol, complete_results(protocol))

    export = build_report_export(report)

    assert report.total_valid_shots == protocol.total_shots
    assert report.decision.primary.metric_key == "carry"
    assert list(export["metric"]) == ["Carry"]
    assert export.loc[0, "protocol_mode"] == "Exploratory"
    assert "decision" in export.columns
    assert export.loc[0, "evidence_role"] == "Exploratory"
    assert report.decision.primary.confirmatory is False
    assert report.decision.conclusion.startswith("Exploratory result:")
    assert "bootstrap_confidence_interval_lower" in export.columns


def test_incomplete_results_cannot_build_final_report() -> None:
    current = experiment()
    protocol = selected_protocol(current)
    results = build_results_template(protocol, ["carry"])
    results.loc[0, "carry"] = 200.0

    with pytest.raises(ReportNotReadyError, match="complete shots"):
        build_experiment_report(current, protocol, results)


def test_legacy_results_can_build_report_after_normalization() -> None:
    current = experiment()
    protocol = selected_protocol(current)
    current_results = complete_results(protocol)
    legacy_results = current_results[["shot_id", "configuration", "carry"]]

    report = build_experiment_report(current, protocol, legacy_results)

    assert report.total_valid_shots == protocol.total_shots


def test_two_pair_report_exports_missing_effect_size_and_warning() -> None:
    current = experiment()
    protocol = selected_protocol(current)
    results = complete_results(protocol)
    for i, configuration in enumerate(protocol.sequence):
        results.loc[i, "carry"] = 200 + ((2 if i < 10 else 4) if configuration == "B" else 0)
    report = build_experiment_report(current, protocol, results)
    analysis = report.analyses[0]
    assert analysis.n_pairs == 2
    assert analysis.hedges_g is None
    assert analysis.p_value is not None
    assert analysis.confidence_interval_lower is not None
    assert any("at least three" in warning for warning in analysis.warnings)
    export = build_report_export(report)
    assert pd.isna(export.loc[0, "hedges_g"])
    assert "at least three" in export.loc[0, "analysis_warnings"]
    assert export.loc[0, "decision_label"] == report.decision.primary.label
    assert export.loc[0, "protocol_order_method"] == protocol.order_method
