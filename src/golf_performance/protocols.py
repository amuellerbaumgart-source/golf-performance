from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .domain import Experiment, ExperimentDesignMode
from .power import (
    HARD_SHOT_LIMIT,
    PowerAnalysis,
    calculate_minimum_detectable_effect,
    calculate_power_analysis,
)


DEFAULT_EXPLORATORY_SHOTS_PER_CONFIGURATION = 30
DEFAULT_SECONDS_PER_SHOT = 30.0


@dataclass(frozen=True)
class TestingProtocol:
    """A concrete shot-collection plan selected for an experiment."""

    shots_per_configuration: int
    block_size: int
    sequence: tuple[str, ...]
    instructions: tuple[str, ...]
    power_analysis: PowerAnalysis
    start_configuration: str = "A"
    design_mode: ExperimentDesignMode = ExperimentDesignMode.CONFIRMATORY
    seconds_per_shot: float = DEFAULT_SECONDS_PER_SHOT
    minimum_detectable_effect: float | None = None
    minimum_detectable_effect_percentage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shots_per_configuration": self.shots_per_configuration,
            "block_size": self.block_size,
            "sequence": list(self.sequence),
            "instructions": list(self.instructions),
            "power_analysis": self.power_analysis.to_dict(),
            "start_configuration": self.start_configuration,
            "design_mode": self.design_mode.value,
            "seconds_per_shot": self.seconds_per_shot,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "minimum_detectable_effect_percentage": self.minimum_detectable_effect_percentage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestingProtocol":
        return cls(
            shots_per_configuration=int(data["shots_per_configuration"]),
            block_size=int(data["block_size"]),
            sequence=tuple(str(item) for item in data["sequence"]),
            instructions=tuple(str(item) for item in data["instructions"]),
            power_analysis=PowerAnalysis.from_dict(data["power_analysis"]),
            start_configuration=str(data.get("start_configuration", "A")),
            design_mode=ExperimentDesignMode(data.get("design_mode", ExperimentDesignMode.CONFIRMATORY.value)),
            seconds_per_shot=float(data.get("seconds_per_shot", DEFAULT_SECONDS_PER_SHOT)),
            minimum_detectable_effect=(
                float(data["minimum_detectable_effect"])
                if data.get("minimum_detectable_effect") is not None
                else None
            ),
            minimum_detectable_effect_percentage=(
                float(data["minimum_detectable_effect_percentage"])
                if data.get("minimum_detectable_effect_percentage") is not None
                else None
            ),
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.shots_per_configuration, bool)
            or not isinstance(self.shots_per_configuration, int)
            or self.shots_per_configuration <= 0
        ):
            raise ValueError("Shots per configuration must be a positive integer")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("Block size must be a positive integer")
        if self.start_configuration not in {"A", "B"}:
            raise ValueError("Start configuration must be A or B")
        if not isinstance(self.power_analysis, PowerAnalysis):
            raise ValueError("Power analysis is required")
        if not isinstance(self.design_mode, ExperimentDesignMode):
            raise ValueError("Design mode must be confirmatory or exploratory")
        if (
            isinstance(self.seconds_per_shot, bool)
            or not isinstance(self.seconds_per_shot, (int, float))
            or not math.isfinite(self.seconds_per_shot)
            or self.seconds_per_shot <= 0
        ):
            raise ValueError("Seconds per shot must be finite and greater than zero")
        if self.power_analysis.block_size != self.block_size:
            raise ValueError("Protocol and power analysis block sizes must match")
        if (
            self.power_analysis.recommended_shots_per_configuration
            != self.shots_per_configuration
            and self.design_mode is ExperimentDesignMode.CONFIRMATORY
        ):
            raise ValueError("Confirmatory shot count must match the power recommendation")
        if self.shots_per_configuration > HARD_SHOT_LIMIT:
            raise ValueError("Protocol exceeds the maximum shot safety limit")
        if self.shots_per_configuration % self.block_size != 0:
            raise ValueError(
                "Shots per configuration must be divisible by the block size"
            )
        if self.minimum_detectable_effect is None:
            object.__setattr__(
                self,
                "minimum_detectable_effect",
                calculate_minimum_detectable_effect(
                    expected_standard_deviation=self.power_analysis.expected_standard_deviation,
                    shots_per_configuration=self.shots_per_configuration,
                    alpha=self.power_analysis.alpha,
                    target_power=self.power_analysis.target_power,
                ),
            )
        if (
            isinstance(self.minimum_detectable_effect, bool)
            or not isinstance(self.minimum_detectable_effect, (int, float))
            or not math.isfinite(self.minimum_detectable_effect)
            or self.minimum_detectable_effect <= 0
        ):
            raise ValueError("Minimum detectable effect must be finite and greater than zero")
        if self.minimum_detectable_effect_percentage is not None and (
            isinstance(self.minimum_detectable_effect_percentage, bool)
            or not isinstance(self.minimum_detectable_effect_percentage, (int, float))
            or not math.isfinite(self.minimum_detectable_effect_percentage)
            or self.minimum_detectable_effect_percentage <= 0
        ):
            raise ValueError("Minimum detectable effect percentage must be positive and finite")

        expected_shots = self.shots_per_configuration * 2
        if len(self.sequence) != expected_shots:
            raise ValueError("Protocol sequence has an unexpected number of shots")
        if set(self.sequence) != {"A", "B"}:
            raise ValueError("Protocol sequence must contain only A and B")
        if self.sequence[0] != self.start_configuration:
            raise ValueError("Protocol sequence must start with the configured starting condition")
        for block_index in range(self.blocks_per_configuration * 2):
            expected_configuration = (
                self.start_configuration
                if block_index % 2 == 0
                else "B" if self.start_configuration == "A" else "A"
            )
            start = block_index * self.block_size
            end = start + self.block_size
            if self.sequence[start:end] != (expected_configuration,) * self.block_size:
                raise ValueError("Protocol sequence must alternate complete blocks")

    @property
    def total_shots(self) -> int:
        return len(self.sequence)

    @property
    def blocks_per_configuration(self) -> int:
        return self.shots_per_configuration // self.block_size

    @property
    def estimated_total_minutes(self) -> float:
        return self.total_shots * self.seconds_per_shot / 60

    @property
    def label(self) -> str:
        return self.design_mode.value.capitalize()


@dataclass(frozen=True)
class ProtocolRecommendation:
    """UI-ready alternatives from which the golfer selects one plan."""

    confirmatory: TestingProtocol
    exploratory: TestingProtocol
    selected_mode: ExperimentDesignMode | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmatory": self.confirmatory.to_dict(),
            "exploratory": self.exploratory.to_dict(),
            "selected_mode": self.selected_mode.value if self.selected_mode else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolRecommendation":
        selected_mode = data.get("selected_mode")
        return cls(
            confirmatory=TestingProtocol.from_dict(data["confirmatory"]),
            exploratory=TestingProtocol.from_dict(data["exploratory"]),
            selected_mode=ExperimentDesignMode(selected_mode) if selected_mode else None,
        )

    def __post_init__(self) -> None:
        if self.confirmatory.design_mode is not ExperimentDesignMode.CONFIRMATORY:
            raise ValueError("Confirmatory option must use confirmatory mode")
        if self.exploratory.design_mode is not ExperimentDesignMode.EXPLORATORY:
            raise ValueError("Exploratory option must use exploratory mode")
        if self.confirmatory.start_configuration != self.exploratory.start_configuration:
            raise ValueError("Protocol options must use the same starting configuration")
        if self.selected_mode is not None and not isinstance(
            self.selected_mode, ExperimentDesignMode
        ):
            raise ValueError("Selected mode must be confirmatory or exploratory")

    @property
    def options(self) -> tuple[TestingProtocol, TestingProtocol]:
        return self.confirmatory, self.exploratory

    @property
    def selected(self) -> TestingProtocol | None:
        if self.selected_mode is None:
            return None
        return (
            self.confirmatory
            if self.selected_mode is ExperimentDesignMode.CONFIRMATORY
            else self.exploratory
        )

    def select(self, mode: ExperimentDesignMode) -> ProtocolRecommendation:
        if not isinstance(mode, ExperimentDesignMode):
            raise ValueError("Selected mode must be confirmatory or exploratory")
        return replace(self, selected_mode=mode)


def _build_sequence(
    *,
    shots_per_configuration: int,
    block_size: int,
    start_configuration: str,
) -> tuple[str, ...]:
    blocks_per_configuration = shots_per_configuration // block_size
    second_configuration = "B" if start_configuration == "A" else "A"
    sequence: list[str] = []
    for block_index in range(blocks_per_configuration * 2):
        configuration = (
            start_configuration if block_index % 2 == 0 else second_configuration
        )
        sequence.extend([configuration] * block_size)
    return tuple(sequence)


def _build_protocol(
    experiment: Experiment,
    *,
    mode: ExperimentDesignMode,
    shots_per_configuration: int,
    block_size: int,
    power_analysis: PowerAnalysis,
    seconds_per_shot: float,
    expected_baseline_mean: float | None,
) -> TestingProtocol:
    primary_metric = experiment.primary_metric
    start_configuration = "A" if experiment.experiment_id.int % 2 == 0 else "B"
    minimum_detectable_effect = calculate_minimum_detectable_effect(
        expected_standard_deviation=power_analysis.expected_standard_deviation,
        shots_per_configuration=shots_per_configuration,
        alpha=power_analysis.alpha,
        target_power=power_analysis.target_power,
    )
    minimum_detectable_effect_percentage = None
    if expected_baseline_mean is not None and expected_baseline_mean != 0:
        minimum_detectable_effect_percentage = (
            minimum_detectable_effect / abs(expected_baseline_mean) * 100
        )

    warnings = list(power_analysis.warnings)
    if mode is ExperimentDesignMode.EXPLORATORY:
        warnings.insert(
            0,
            "Exploratory plan: use this sample for trends and large effects, not definitive confirmation.",
        )
        if shots_per_configuration >= power_analysis.recommended_shots_per_configuration:
            warnings.append(
                "This exploratory cap meets or exceeds the powered recommendation; confirmatory mode may be appropriate."
            )

    instructions = (
        f"TEST VARIABLE: {experiment.changed_variable}",
        f"A (baseline): {experiment.baseline_value}",
        f"B (treatment): {experiment.treatment_value}",
        f"DESIGN MODE: {mode.value.capitalize()}",
        f"PRIMARY METRIC: {primary_metric.display_name} ({primary_metric.unit})",
        f"SHOTS: {shots_per_configuration} per configuration ({shots_per_configuration * 2} total)",
        f"ESTIMATED TIME: {shots_per_configuration * 2 * seconds_per_shot / 60:.1f} minutes",
        f"MINIMUM DETECTABLE EFFECT: {minimum_detectable_effect:g} {primary_metric.unit}",
        f"TARGET POWER: {power_analysis.target_power:.0%} at alpha={power_analysis.alpha:g}",
        f"STARTING CONFIGURATION: {start_configuration}",
        f"ANALYSIS TRANSFORM: {primary_metric.analysis_transform.value}",
        "Keep everything else constant where reasonably possible.",
        "Use the same ball model and target throughout the test.",
        "Warm up before recording valid shots.",
        "Apply exclusion rules before reviewing the results.",
    ) + tuple(warnings)

    return TestingProtocol(
        shots_per_configuration=shots_per_configuration,
        block_size=block_size,
        sequence=_build_sequence(
            shots_per_configuration=shots_per_configuration,
            block_size=block_size,
            start_configuration=start_configuration,
        ),
        instructions=instructions,
        power_analysis=power_analysis,
        start_configuration=start_configuration,
        design_mode=mode,
        seconds_per_shot=seconds_per_shot,
        minimum_detectable_effect=minimum_detectable_effect,
        minimum_detectable_effect_percentage=minimum_detectable_effect_percentage,
    )


def generate_protocol_options(
    experiment: Experiment,
    *,
    expected_standard_deviation: float,
    exploratory_shots_per_configuration: int = DEFAULT_EXPLORATORY_SHOTS_PER_CONFIGURATION,
    expected_baseline_mean: float | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    block_size: int = 5,
    seconds_per_shot: float = DEFAULT_SECONDS_PER_SHOT,
) -> ProtocolRecommendation:
    """Generate confirmatory and exploratory alternatives for user selection."""

    experiment.validate()
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("Block size must be a positive integer")
    if (
        isinstance(exploratory_shots_per_configuration, bool)
        or not isinstance(exploratory_shots_per_configuration, int)
        or exploratory_shots_per_configuration <= 0
    ):
        raise ValueError("Exploratory shot cap must be a positive integer")
    if exploratory_shots_per_configuration % block_size != 0:
        raise ValueError("Exploratory shot cap must be divisible by the block size")
    if exploratory_shots_per_configuration > HARD_SHOT_LIMIT:
        raise ValueError("Exploratory shot cap exceeds the maximum shot safety limit")
    if (
        isinstance(seconds_per_shot, bool)
        or not isinstance(seconds_per_shot, (int, float))
        or seconds_per_shot <= 0
        or not math.isfinite(seconds_per_shot)
    ):
        raise ValueError("Seconds per shot must be finite and greater than zero")

    power_analysis = calculate_power_analysis(
        experiment.primary_metric,
        expected_standard_deviation=expected_standard_deviation,
        expected_baseline_mean=expected_baseline_mean,
        alpha=alpha,
        target_power=target_power,
        block_size=block_size,
    )
    confirmatory = _build_protocol(
        experiment,
        mode=ExperimentDesignMode.CONFIRMATORY,
        shots_per_configuration=power_analysis.recommended_shots_per_configuration,
        block_size=block_size,
        power_analysis=power_analysis,
        seconds_per_shot=seconds_per_shot,
        expected_baseline_mean=expected_baseline_mean,
    )
    exploratory = _build_protocol(
        experiment,
        mode=ExperimentDesignMode.EXPLORATORY,
        shots_per_configuration=exploratory_shots_per_configuration,
        block_size=block_size,
        power_analysis=power_analysis,
        seconds_per_shot=seconds_per_shot,
        expected_baseline_mean=expected_baseline_mean,
    )
    return ProtocolRecommendation(confirmatory=confirmatory, exploratory=exploratory)


def generate_protocol(
    experiment: Experiment,
    *,
    expected_standard_deviation: float,
    expected_baseline_mean: float | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    block_size: int = 5,
    seconds_per_shot: float = DEFAULT_SECONDS_PER_SHOT,
) -> TestingProtocol:
    """Backward-compatible helper returning the confirmatory option."""

    return generate_protocol_options(
        experiment,
        expected_standard_deviation=expected_standard_deviation,
        expected_baseline_mean=expected_baseline_mean,
        alpha=alpha,
        target_power=target_power,
        block_size=block_size,
        seconds_per_shot=seconds_per_shot,
    ).confirmatory
