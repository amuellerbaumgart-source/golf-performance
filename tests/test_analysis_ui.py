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
