"""Core domain package for the golf performance experimentation app."""

from .domain import (
    Experiment,
    ExperimentDesignMode,
    ExperimentStatus,
    MetricAnalysisTransform,
    MetricDefinition,
    MetricDirection,
    ThresholdType,
)
from .protocols import (
    ProtocolRecommendation,
    TestingProtocol,
    generate_protocol,
    generate_protocol_options,
)
from .power import (
    PowerAnalysis,
    calculate_minimum_detectable_effect,
    calculate_power_analysis,
)
from .storage import ExperimentSummary, FileSystemStorage, StoredExperiment
from .decision import (
    DecisionCategory,
    ExperimentDecision,
    MetricDecision,
    evaluate_experiment,
    evaluate_metric_decision,
)
from .reporting import (
    ExperimentReport,
    ReportNotReadyError,
    build_experiment_report,
    build_report_export,
)

__all__ = [
    "Experiment",
    "ExperimentDesignMode",
    "ExperimentStatus",
    "MetricAnalysisTransform",
    "MetricDefinition",
    "MetricDirection",
    "ThresholdType",
    "TestingProtocol",
    "ProtocolRecommendation",
    "generate_protocol",
    "generate_protocol_options",
    "PowerAnalysis",
    "calculate_minimum_detectable_effect",
    "calculate_power_analysis",
    "ExperimentSummary",
    "FileSystemStorage",
    "StoredExperiment",
    "DecisionCategory",
    "MetricDecision",
    "ExperimentDecision",
    "evaluate_metric_decision",
    "evaluate_experiment",
    "ExperimentReport",
    "ReportNotReadyError",
    "build_experiment_report",
    "build_report_export",
]
