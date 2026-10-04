import pytest

from golf_performance.domain import (
    Experiment,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)
from golf_performance.power import calculate_power_analysis
from golf_performance.protocols import TestingProtocol as Protocol, generate_protocol


def experiment(metric: MetricDefinition | None = None) -> Experiment:
    return Experiment(
        name="Driver loft test",
        changed_variable="Driver loft",
        baseline_value="9 degrees",
        treatment_value="10 degrees",
        primary_goal="Improve carry",
        metrics=[
            metric
            or MetricDefinition(
                key="carry",
                display_name="Carry",
                unit="yards",
                direction=MetricDirection.HIGHER,
                meaningful_threshold=3.0,
                is_primary=True,
            )
        ],
    )


def test_protocol_uses_power_recommendation_and_alternates_blocks() -> None:
    protocol = generate_protocol(
        experiment(),
        expected_standard_deviation=8.0,
        block_size=5,
    )

    shots = protocol.shots_per_configuration
    assert shots == protocol.power_analysis.recommended_shots_per_configuration
    assert shots % 5 == 0
    assert len(protocol.sequence) == shots * 2
    first_configuration = protocol.start_configuration
    second_configuration = "B" if first_configuration == "A" else "A"
    assert protocol.sequence[:5] == (first_configuration,) * 5
    assert protocol.sequence[5:10] == (second_configuration,) * 5
    assert protocol.sequence.count(first_configuration) == shots
    assert protocol.sequence.count(second_configuration) == shots


def test_protocol_is_deterministic() -> None:
    same_experiment = experiment()
    first = generate_protocol(
        same_experiment, expected_standard_deviation=8.0, block_size=3
    )
    second = generate_protocol(
        same_experiment, expected_standard_deviation=8.0, block_size=3
    )

    assert first == second


def test_higher_variability_produces_more_shots() -> None:
    low_variability = generate_protocol(
        experiment(), expected_standard_deviation=4.0, block_size=5
    )
    high_variability = generate_protocol(
        experiment(), expected_standard_deviation=8.0, block_size=5
    )

    assert high_variability.shots_per_configuration > low_variability.shots_per_configuration


def test_protocol_instructions_describe_power_assumptions() -> None:
    protocol = generate_protocol(
        experiment(), expected_standard_deviation=8.0, block_size=5
    )

    assert "TEST VARIABLE: Driver loft" in protocol.instructions
    assert "A (baseline): 9 degrees" in protocol.instructions
    assert "B (treatment): 10 degrees" in protocol.instructions
    assert "PRIMARY METRIC: Carry (yards)" in protocol.instructions
    assert any("TARGET POWER: 80%" in instruction for instruction in protocol.instructions)
    assert any("not a guarantee" in instruction for instruction in protocol.instructions)


def test_percentage_metric_requires_and_uses_expected_baseline_mean() -> None:
    metric = MetricDefinition(
        key="offline",
        display_name="Offline",
        unit="yards",
        direction=MetricDirection.LOWER,
        meaningful_threshold=10.0,
        threshold_type=ThresholdType.PERCENTAGE,
        is_primary=True,
    )

    protocol = generate_protocol(
        experiment(metric),
        expected_standard_deviation=10.0,
        expected_baseline_mean=20.0,
        block_size=5,
    )

    assert protocol.power_analysis.expected_effect == pytest.approx(2.0)

    with pytest.raises(ValueError, match="baseline mean"):
        generate_protocol(
            experiment(metric), expected_standard_deviation=10.0, block_size=5
        )


def test_target_metric_explains_absolute_error_basis() -> None:
    metric = MetricDefinition(
        key="launch",
        display_name="Launch angle",
        unit="degrees",
        direction=MetricDirection.TARGET,
        meaningful_threshold=1.0,
        target_value=15.0,
        is_primary=True,
    )

    protocol = generate_protocol(
        experiment(metric), expected_standard_deviation=3.0, block_size=5
    )

    assert "absolute error" in protocol.power_analysis.analysis_basis


def test_protocol_model_rejects_malformed_sequence() -> None:
    power_analysis = calculate_power_analysis(
        experiment().primary_metric,
        expected_standard_deviation=8.0,
        block_size=1,
    )
    with pytest.raises(ValueError, match="unexpected number"):
        Protocol(
            shots_per_configuration=power_analysis.recommended_shots_per_configuration,
            block_size=1,
            sequence=("A",),
            instructions=(),
            power_analysis=power_analysis,
        )


def test_protocol_model_rejects_mismatched_power_recommendation() -> None:
    power_analysis = calculate_power_analysis(
        experiment().primary_metric,
        expected_standard_deviation=8.0,
        block_size=5,
    )
    with pytest.raises(ValueError, match="shot count"):
        Protocol(
            shots_per_configuration=power_analysis.recommended_shots_per_configuration + 5,
            block_size=5,
            sequence=("A",) * (power_analysis.recommended_shots_per_configuration + 5)
            + ("B",) * (power_analysis.recommended_shots_per_configuration + 5),
            instructions=(),
            power_analysis=power_analysis,
            start_configuration="A",
        )
