import pandas as pd
import pytest
from uuid import UUID

from golf_performance.domain import (
    Experiment,
    ExperimentDesignMode,
    MetricAnalysisTransform,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)
from golf_performance.data_collection import build_results_template
from golf_performance.protocols import generate_protocol_options
from golf_performance.statistics import (
    ANALYSIS_UNIT,
    STATISTICAL_METHOD,
    analyze_experiment,
    analyze_metric,
)


def metric(
    key: str = "carry",
    *,
    direction: MetricDirection = MetricDirection.HIGHER,
    transform: MetricAnalysisTransform = MetricAnalysisTransform.RAW,
    target_value: float | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        key=key,
        display_name=key.title(),
        unit="units",
        direction=direction,
        meaningful_threshold=3.0,
        threshold_type=ThresholdType.ABSOLUTE,
        target_value=target_value,
        is_primary=True,
        analysis_transform=transform,
    )


def test_analyze_metric_reports_descriptive_and_welch_statistics() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 4 + ["B"] * 4,
            "carry": [100.0, 102.0, 98.0, 101.0, 105.0, 107.0, 104.0, 106.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.n == 4
    assert analysis.treatment.n == 4
    assert analysis.difference == pytest.approx(5.25)
    assert analysis.percentage_difference == pytest.approx(5.25 / 100.25 * 100)
    assert analysis.confidence_interval_lower < analysis.difference < analysis.confidence_interval_upper
    assert analysis.p_value is not None
    assert analysis.hedges_g is not None
    assert analysis.statistical_method == STATISTICAL_METHOD
    assert analysis.analysis_unit == ANALYSIS_UNIT


def test_analysis_warns_when_shots_have_exclusion_notes() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 4 + ["B"] * 4,
            "exclusion_note": ["mishit", "", "", "", "", "", "", ""],
            "carry": [100.0, 102.0, 98.0, 101.0, 105.0, 107.0, 104.0, 106.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert any("exclusion notes" in warning for warning in analysis.warnings)


def test_current_analysis_explicitly_identifies_individual_shot_method() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 3 + ["B"] * 3,
            "carry": [100.0, 101.0, 99.0, 104.0, 105.0, 103.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.statistical_method == "welch_independent_two_sample_t"
    assert analysis.analysis_unit == "individual_shot"
    assert any("independent observations" in warning for warning in analysis.warnings)


def test_analysis_transform_is_applied_before_comparison() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 4 + ["B"] * 4,
            "offline": [-10.0, 10.0, -8.0, 8.0, -5.0, 5.0, -4.0, 4.0],
        }
    )

    analysis = analyze_metric(
        results,
        metric(
            key="offline",
            direction=MetricDirection.LOWER,
            transform=MetricAnalysisTransform.ABSOLUTE_VALUE,
        ),
    )

    assert analysis.baseline.mean == pytest.approx(9.0)
    assert analysis.treatment.mean == pytest.approx(4.5)
    assert analysis.difference == pytest.approx(-4.5)


def test_target_analysis_uses_absolute_error_from_target() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 3 + ["B"] * 3,
            "launch": [14.0, 16.0, 15.5, 14.5, 15.5, 15.0],
        }
    )

    analysis = analyze_metric(
        results,
        metric(
            key="launch",
            direction=MetricDirection.TARGET,
            transform=MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET,
            target_value=15.0,
        ),
    )

    assert analysis.analysis_basis == "absolute error from target"
    assert analysis.baseline.mean == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert analysis.treatment.mean == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    assert analysis.difference < 0


def test_incomplete_groups_keep_descriptive_results_but_warn_about_inference() -> None:
    results = pd.DataFrame({"configuration": ["A", "B"], "carry": [100.0, 105.0]})

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.mean == 100.0
    assert analysis.treatment.mean == 105.0
    assert analysis.p_value is None
    assert analysis.confidence_interval_lower is None
    assert any("two valid observations" in warning for warning in analysis.warnings)


def test_invalid_and_missing_values_are_excluded_from_analysis() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "A", "B", "B", "B"],
            "carry": [100.0, float("nan"), "bad", 105.0, 106.0, float("inf")],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.n == 1
    assert analysis.treatment.n == 2
    assert analysis.p_value is None
    assert any("excluded" in warning for warning in analysis.warnings)
    assert any("independent observations" in warning for warning in analysis.warnings)


def test_zero_variance_difference_does_not_claim_a_valid_p_value() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A"] * 3 + ["B"] * 3,
            "carry": [100.0, 100.0, 100.0, 105.0, 105.0, 105.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.p_value is None
    assert analysis.confidence_interval_lower is None
    assert analysis.hedges_g is None
    assert any("zero observed variability" in warning for warning in analysis.warnings)


def test_unexpected_configurations_are_reported_and_ignored() -> None:
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B", "C"],
            "carry": [100.0, 101.0, 105.0, 106.0, 999.0],
        }
    )

    analysis = analyze_metric(results, metric())

    assert analysis.baseline.n == 2
    assert analysis.treatment.n == 2
    assert any("unexpected configurations" in warning for warning in analysis.warnings)


def test_analyze_experiment_preserves_metric_order() -> None:
    metrics = [metric(), metric("offline", direction=MetricDirection.LOWER, transform=MetricAnalysisTransform.ABSOLUTE_VALUE)]
    experiment = Experiment(
        name="Test",
        changed_variable="Loft",
        baseline_value="9",
        treatment_value="10",
        primary_goal="Carry and dispersion",
        metrics=[metrics[0], MetricDefinition(**{**metrics[1].__dict__, "is_primary": False})],
    )
    results = pd.DataFrame(
        {
            "configuration": ["A", "A", "B", "B"],
            "carry": [100.0, 101.0, 105.0, 106.0],
            "offline": [-10.0, 10.0, -5.0, 5.0],
        }
    )

    analyses = analyze_experiment(results, experiment)

    assert [analysis.metric_key for analysis in analyses] == ["carry", "offline"]


def _paired_protocol(
    metric_definition: MetricDefinition,
    *,
    start_with_b: bool = False,
    exploratory_shots: int = 10,
):
    experiment = Experiment(
        name="Paired test",
        changed_variable="Loft",
        baseline_value="9",
        treatment_value="10",
        primary_goal="Improve carry",
        metrics=[metric_definition],
        experiment_id=UUID(int=1 if start_with_b else 2),
    )
    return generate_protocol_options(
        experiment,
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=exploratory_shots,
        block_size=5,
    ).select(ExperimentDesignMode.EXPLORATORY).selected


def test_protocol_analysis_uses_complete_paired_blocks_as_inferential_units() -> None:
    current_metric = metric()
    protocol = _paired_protocol(current_metric)
    results = build_results_template(protocol, [current_metric.key])
    block_means = {"A": [100.0, 110.0], "B": [104.0, 116.0]}
    for block_index in range(protocol.blocks_per_configuration * 2):
        start = block_index * protocol.block_size
        configuration = protocol.sequence[start]
        mean = block_means[configuration][block_index // 2]
        results.loc[start : start + protocol.block_size - 1, "carry"] = mean

    analysis = analyze_metric(results, current_metric, protocol)

    assert analysis.analysis_unit == "paired_block"
    assert analysis.statistical_method == "paired_block_t"
    assert analysis.n_pairs == 2
    assert analysis.difference == pytest.approx(5.0)
    assert analysis.paired_difference_standard_deviation == pytest.approx(2**0.5)
    assert analysis.confidence_interval_lower < analysis.difference < analysis.confidence_interval_upper


def test_paired_analysis_handles_protocol_starting_with_b() -> None:
    current_metric = metric()
    protocol = _paired_protocol(current_metric, start_with_b=True)
    results = build_results_template(protocol, [current_metric.key])
    for block_index in range(protocol.blocks_per_configuration * 2):
        start = block_index * protocol.block_size
        configuration = protocol.sequence[start]
        mean = 100.0 if configuration == "A" else 105.0
        results.loc[start : start + protocol.block_size - 1, "carry"] = mean

    analysis = analyze_metric(results, current_metric, protocol)

    assert analysis.difference == pytest.approx(5.0)
    assert analysis.n_pairs == protocol.blocks_per_configuration


def test_incomplete_block_pair_is_excluded_from_paired_inference() -> None:
    current_metric = metric()
    protocol = _paired_protocol(current_metric)
    results = build_results_template(protocol, [current_metric.key])
    results["carry"] = 100.0
    results.loc[results["configuration"] == "B", "carry"] = 105.0
    results.loc[0, "carry"] = float("nan")

    analysis = analyze_metric(results, current_metric, protocol)

    assert analysis.n_pairs == 1
    assert analysis.p_value is None
    assert any("block pair" in warning for warning in analysis.warnings)


def test_paired_bootstrap_interval_is_reproducible_when_enough_pairs_exist() -> None:
    current_metric = metric()
    protocol = _paired_protocol(current_metric, exploratory_shots=25)
    results = build_results_template(protocol, [current_metric.key])
    pair_differences = [4.0, 5.0, 6.0, 7.0, 8.0]
    for pair_index, difference in enumerate(pair_differences):
        first_start = pair_index * 2 * protocol.block_size
        second_start = first_start + protocol.block_size
        results.loc[first_start : first_start + protocol.block_size - 1, "carry"] = 100.0
        results.loc[second_start : second_start + protocol.block_size - 1, "carry"] = 100.0 + difference

    first = analyze_metric(results, current_metric, protocol)
    second = analyze_metric(results, current_metric, protocol)

    assert first.n_pairs == 5
    assert first.bootstrap_confidence_interval_lower is not None
    assert first.bootstrap_confidence_interval_upper is not None
    assert first.bootstrap_confidence_interval_lower == second.bootstrap_confidence_interval_lower
    assert first.bootstrap_confidence_interval_upper == second.bootstrap_confidence_interval_upper
