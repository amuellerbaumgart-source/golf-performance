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


def test_incomplete_results_cannot_build_final_report() -> None:
    current = experiment()
    protocol = selected_protocol(current)
    results = build_results_template(protocol, ["carry"])
    results.loc[0, "carry"] = 200.0

    with pytest.raises(ReportNotReadyError, match="complete shots"):
        build_experiment_report(current, protocol, results)
