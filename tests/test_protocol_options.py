import pytest

from golf_performance.domain import (
    Experiment,
    ExperimentDesignMode,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)
from golf_performance.protocols import generate_protocol_options


def experiment() -> Experiment:
    return Experiment(
        name="Driver loft test",
        changed_variable="Driver loft",
        baseline_value="9 degrees",
        treatment_value="10 degrees",
        primary_goal="Improve carry",
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


def test_protocol_options_expose_confirmatory_and_exploratory_choices() -> None:
    recommendation = generate_protocol_options(
        experiment(),
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=30,
        seconds_per_shot=30.0,
    )

    assert recommendation.selected is None
    assert recommendation.confirmatory.design_mode is ExperimentDesignMode.CONFIRMATORY
    assert recommendation.exploratory.design_mode is ExperimentDesignMode.EXPLORATORY
    assert recommendation.confirmatory.label == "Confirmatory"
    assert recommendation.exploratory.label == "Exploratory"
    assert recommendation.confirmatory.total_shots == (
        recommendation.confirmatory.shots_per_configuration * 2
    )
    assert recommendation.exploratory.total_shots == 60
    assert recommendation.exploratory.estimated_total_minutes == pytest.approx(30.0)
    assert len(recommendation.exploratory.sequence) == 60
    assert recommendation.exploratory.minimum_detectable_effect > 0
    assert any("Exploratory plan" in warning for warning in recommendation.exploratory.instructions)


def test_user_can_select_the_plan_to_use() -> None:
    recommendation = generate_protocol_options(
        experiment(),
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=30,
    )

    selected = recommendation.select(ExperimentDesignMode.EXPLORATORY)

    assert selected.selected is selected.exploratory
    assert selected.selected_mode is ExperimentDesignMode.EXPLORATORY
    assert recommendation.selected is None


def test_exploratory_cap_must_match_blocks() -> None:
    with pytest.raises(ValueError, match="divisible"):
        generate_protocol_options(
            experiment(),
            expected_standard_deviation=8.0,
            exploratory_shots_per_configuration=31,
            block_size=5,
        )


def test_protocol_options_share_start_order_but_have_exact_sequences() -> None:
    recommendation = generate_protocol_options(
        experiment(),
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=20,
        block_size=5,
    )

    assert recommendation.confirmatory.start_configuration == recommendation.exploratory.start_configuration
    assert recommendation.confirmatory.sequence[0] == recommendation.confirmatory.start_configuration
    assert recommendation.exploratory.sequence[0] == recommendation.exploratory.start_configuration
    assert recommendation.confirmatory.blocks_per_configuration * 5 == recommendation.confirmatory.shots_per_configuration


def test_percentage_option_exposes_mde_in_percentage_terms() -> None:
    percentage_experiment = Experiment(
        name="Dispersion test",
        changed_variable="Tee height",
        baseline_value="Normal",
        treatment_value="Higher",
        primary_goal="Reduce offline dispersion",
        metrics=[
            MetricDefinition(
                key="offline",
                display_name="Offline",
                unit="yards",
                direction=MetricDirection.LOWER,
                meaningful_threshold=10.0,
                threshold_type=ThresholdType.PERCENTAGE,
                is_primary=True,
            )
        ],
    )
    recommendation = generate_protocol_options(
        percentage_experiment,
        expected_standard_deviation=10.0,
        expected_baseline_mean=20.0,
        exploratory_shots_per_configuration=30,
    )

    assert recommendation.exploratory.minimum_detectable_effect_percentage is not None


def test_infeasible_confirmatory_plan_is_reported_without_crashing() -> None:
    recommendation = generate_protocol_options(
        experiment(),
        expected_standard_deviation=100.0,
        exploratory_shots_per_configuration=30,
        block_size=5,
    )

    assert recommendation.confirmatory is None
    assert recommendation.confirmatory_shots_required > 1000
    assert recommendation.exploratory is not None
    assert any("operational limit" in warning for warning in recommendation.exploratory.power_analysis.warnings)
    with pytest.raises(ValueError, match="operational shot limit"):
        recommendation.select(ExperimentDesignMode.CONFIRMATORY)
