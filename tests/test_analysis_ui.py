import pytest

from golf_performance.ui.analysis import build_metric_comparison_table
from golf_performance.statistics import MetricAnalysis, ConditionSummary


def test_metric_comparison_table_contains_report_fields() -> None:
    analysis = MetricAnalysis(
        metric_key="carry",
        metric_name="Carry",
        unit="yards",
        analysis_basis="raw metric values",
        baseline=ConditionSummary("A", 3, 200.0, 200.0, 2.0),
        treatment=ConditionSummary("B", 3, 204.0, 204.0, 2.5),
        difference=4.0,
        percentage_difference=2.0,
        confidence_interval_lower=1.0,
        confidence_interval_upper=7.0,
        p_value=0.04,
        hedges_g=1.2,
    )

    table = build_metric_comparison_table((analysis,))

    assert list(table["Metric"]) == ["Carry"]
    assert table.loc[0, "B − A"] == pytest.approx(4.0)
    assert table.loc[0, "95% CI for B − A"] == "[1.00, 7.00]"
    assert table.loc[0, "p-value"] == pytest.approx(0.04)


@pytest.mark.parametrize("page", ["analysis", "report"])
@pytest.mark.parametrize("mode", ["exploratory", "confirmatory"])
@pytest.mark.parametrize("lower", [False, True])
def test_analysis_page_renders_roles_and_direction(tmp_path, mode, lower, page) -> None:
    from streamlit.testing.v1 import AppTest

    script = f'''
from golf_performance.domain import Experiment, MetricDefinition, MetricDirection, ExperimentDesignMode
from golf_performance.protocols import generate_protocol_options
from golf_performance.storage import FileSystemStorage
from golf_performance.data_collection import build_results_template
from golf_performance.ui.{page} import render_{page}
metric = MetricDefinition("carry", "Carry", "yards", MetricDirection.{"LOWER" if lower else "HIGHER"}, 3, is_primary=True)
experiment = Experiment("Analysis preview", "Setup", "Original", "New", "Improve", [metric])
recommendation = generate_protocol_options(experiment, expected_standard_deviation=4).select(ExperimentDesignMode.{mode.upper()})
protocol = recommendation.selected
results = build_results_template(protocol, ["carry"])
results["carry"] = [100 + ({-1 if lower else 1} * (4 + (i // 10) % 3) if c == "B" else 0) for i, c in enumerate(protocol.sequence)]
storage = FileSystemStorage({str(tmp_path)!r})
storage.save_experiment(experiment, recommendation)
storage.save_results(experiment.experiment_id, results)
render_{page}(experiment, recommendation, protocol, storage)
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    if page == "report":
        assert any("Experiment report" in item.value for item in app.header)
        assert app.dataframe[0].value.loc[0, "evidence_role"] == mode.capitalize()
        assert app.get_by_key("download_report_csv")
        if mode == "exploratory":
            assert not app.success
        return
    assert any("Your result" in item.value for item in app.subheader)
    assert any("Full statistical details" in item.label for item in [*app.expander, *app.status])
    improvement = next(item for item in app.metric if item.label == "Observed improvement")
    assert improvement.value.startswith("+")
    evidence_table = app.dataframe[1].value
    assert evidence_table.loc[0, "Evidence role"] == mode.capitalize()
    assert app.get_by_key("view_report")
    if mode == "exploratory":
        assert not app.success
        assert any("All results from this plan are exploratory" in item.value for item in app.info)
