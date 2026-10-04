from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, ClassVar
from uuid import UUID, uuid4


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    PROTOCOL_READY = "protocol_ready"
    COLLECTING = "collecting"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    ANALYZED = "analyzed"


class ExperimentDesignMode(StrEnum):
    CONFIRMATORY = "confirmatory"
    EXPLORATORY = "exploratory"


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"
    TARGET = "target"


class ThresholdType(StrEnum):
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"


class MetricAnalysisTransform(StrEnum):
    RAW = "raw"
    ABSOLUTE_VALUE = "absolute_value"
    ABSOLUTE_ERROR_FROM_TARGET = "absolute_error_from_target"


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    display_name: str
    unit: str
    direction: MetricDirection
    meaningful_threshold: float
    threshold_type: ThresholdType = ThresholdType.ABSOLUTE
    target_value: float | None = None
    is_primary: bool = False
    analysis_transform: MetricAnalysisTransform | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("Metric key cannot be empty")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("Metric display name cannot be empty")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("Metric unit cannot be empty")
        if not isinstance(self.direction, MetricDirection):
            raise ValueError("Metric direction is invalid")
        if not isinstance(self.threshold_type, ThresholdType):
            raise ValueError("Threshold type is invalid")
        if self.analysis_transform is not None and not isinstance(
            self.analysis_transform, MetricAnalysisTransform
        ):
            raise ValueError("Metric analysis transform is invalid")
        if (
            isinstance(self.meaningful_threshold, bool)
            or not isinstance(self.meaningful_threshold, (int, float))
            or not math.isfinite(self.meaningful_threshold)
        ):
            raise ValueError("Meaningful threshold must be finite and numeric")
        if self.meaningful_threshold <= 0:
            raise ValueError("Meaningful threshold must be greater than zero")
        if self.direction is MetricDirection.TARGET and self.target_value is None:
            raise ValueError("Target-based metrics require a target value")
        if self.direction is not MetricDirection.TARGET and self.target_value is not None:
            raise ValueError("Only target-based metrics may define a target value")
        if self.target_value is not None and (
            isinstance(self.target_value, bool)
            or
            not isinstance(self.target_value, (int, float))
            or not math.isfinite(self.target_value)
        ):
            raise ValueError("Target value must be finite and numeric")
        if self.analysis_transform is None:
            default_transform = (
                MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET
                if self.direction is MetricDirection.TARGET
                else MetricAnalysisTransform.RAW
            )
            object.__setattr__(self, "analysis_transform", default_transform)
        if (
            self.direction is MetricDirection.TARGET
            and self.analysis_transform
            is not MetricAnalysisTransform.ABSOLUTE_ERROR_FROM_TARGET
        ):
            raise ValueError("Target metrics must use absolute error from target")


@dataclass
class Experiment:
    name: str
    changed_variable: str
    baseline_value: str
    treatment_value: str
    primary_goal: str
    metrics: list[MetricDefinition]
    experiment_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ExperimentStatus = ExperimentStatus.DRAFT
    schema_version: int = 1

    _ALLOWED_TRANSITIONS: ClassVar[dict[ExperimentStatus, frozenset[ExperimentStatus]]] = {
        ExperimentStatus.DRAFT: frozenset({ExperimentStatus.PROTOCOL_READY}),
        ExperimentStatus.PROTOCOL_READY: frozenset({ExperimentStatus.COLLECTING}),
        ExperimentStatus.COLLECTING: frozenset({ExperimentStatus.READY_FOR_ANALYSIS}),
        ExperimentStatus.READY_FOR_ANALYSIS: frozenset({ExperimentStatus.ANALYZED}),
        ExperimentStatus.ANALYZED: frozenset(),
    }

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.experiment_id, UUID):
            raise ValueError("experiment_id must be a UUID")
        if not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime")
        if not isinstance(self.status, ExperimentStatus):
            raise ValueError("status must be a valid experiment status")
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must be an integer")
        required_text = {
            "name": self.name,
            "changed_variable": self.changed_variable,
            "baseline_value": self.baseline_value,
            "treatment_value": self.treatment_value,
            "primary_goal": self.primary_goal,
        }
        for field_name, value in required_text.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")

        if self.baseline_value.strip() == self.treatment_value.strip():
            raise ValueError("Baseline and treatment values must be different")
        if not isinstance(self.metrics, list) or not self.metrics:
            raise ValueError("At least one metric is required")
        if any(not isinstance(metric, MetricDefinition) for metric in self.metrics):
            raise ValueError("metrics must contain MetricDefinition objects")
        if sum(metric.is_primary for metric in self.metrics) != 1:
            raise ValueError("Exactly one primary metric is required")
        if len({metric.key for metric in self.metrics}) != len(self.metrics):
            raise ValueError("Metric keys must be unique")
        if self.schema_version < 1:
            raise ValueError("Unsupported schema version")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")

    @property
    def primary_metric(self) -> MetricDefinition:
        return next(metric for metric in self.metrics if metric.is_primary)

    def transition_to(self, new_status: ExperimentStatus) -> None:
        """Move the experiment through its allowed lifecycle states."""

        if not isinstance(new_status, ExperimentStatus):
            raise ValueError("new_status must be a valid experiment status")
        if new_status not in self._ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(
                f"Cannot transition experiment from {self.status.value} to {new_status.value}"
            )
        self.status = new_status

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["experiment_id"] = str(self.experiment_id)
        data["created_at"] = self.created_at.isoformat()
        data["status"] = self.status.value
        data["metrics"] = [
            {
                **asdict(metric),
                "direction": metric.direction.value,
                "threshold_type": metric.threshold_type.value,
                "analysis_transform": metric.analysis_transform.value,
            }
            for metric in self.metrics
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        try:
            metrics = []
            for metric in data["metrics"]:
                metric_data = dict(metric)
                metric_data["direction"] = MetricDirection(metric_data["direction"])
                metric_data["threshold_type"] = ThresholdType(metric_data["threshold_type"])
                transform = metric_data.get("analysis_transform")
                metric_data["analysis_transform"] = (
                    MetricAnalysisTransform(transform) if transform is not None else None
                )
                metrics.append(MetricDefinition(**metric_data))

            return cls(
                name=data["name"],
                changed_variable=data["changed_variable"],
                baseline_value=data["baseline_value"],
                treatment_value=data["treatment_value"],
                primary_goal=data["primary_goal"],
                metrics=metrics,
                experiment_id=UUID(data["experiment_id"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                status=ExperimentStatus(data["status"]),
                schema_version=data.get("schema_version", 1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid experiment data") from exc
