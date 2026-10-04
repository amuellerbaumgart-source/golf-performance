from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd

from .domain import Experiment, ExperimentDesignMode, ExperimentStatus
from .protocols import ProtocolRecommendation


STORAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredExperiment:
    experiment: Experiment
    protocol_recommendation: ProtocolRecommendation | None = None

    @property
    def selected_mode(self) -> ExperimentDesignMode | None:
        return self.protocol_recommendation.selected_mode if self.protocol_recommendation else None


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: UUID
    name: str
    status: ExperimentStatus
    created_at: datetime


class FileSystemStorage:
    """Local JSON/CSV storage for resumable experiments."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured_root = root or os.environ.get("GOLF_PERFORMANCE_DATA_DIR", "data/experiments")
        self.root = Path(configured_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _experiment_directory(self, experiment_id: UUID | str) -> Path:
        try:
            parsed_id = experiment_id if isinstance(experiment_id, UUID) else UUID(str(experiment_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid experiment ID") from exc
        return self.root / str(parsed_id)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = temporary_file.name
            os.replace(temporary_path, path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def save_experiment(
        self,
        experiment: Experiment,
        protocol_recommendation: ProtocolRecommendation | None = None,
    ) -> None:
        experiment.validate()
        payload = {
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
            "experiment": experiment.to_dict(),
            "protocol_recommendation": (
                protocol_recommendation.to_dict() if protocol_recommendation else None
            ),
        }
        directory = self._experiment_directory(experiment.experiment_id)
        self._atomic_write_text(
            directory / "experiment.json",
            json.dumps(payload, indent=2, sort_keys=True),
        )

    def create_experiment(
        self,
        experiment: Experiment,
        protocol_recommendation: ProtocolRecommendation | None = None,
    ) -> Experiment:
        self.save_experiment(experiment, protocol_recommendation)
        return experiment

    def load_experiment(self, experiment_id: UUID | str) -> StoredExperiment:
        directory = self._experiment_directory(experiment_id)
        metadata_path = directory / "experiment.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} was not found")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("storage_schema_version") != STORAGE_SCHEMA_VERSION:
                raise ValueError("Unsupported storage schema version")
            experiment = Experiment.from_dict(payload["experiment"])
            raw_recommendation = payload.get("protocol_recommendation")
            recommendation = (
                ProtocolRecommendation.from_dict(raw_recommendation)
                if raw_recommendation
                else None
            )
            return StoredExperiment(experiment, recommendation)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Experiment {experiment_id} contains invalid metadata") from exc

    def list_experiments(self) -> list[ExperimentSummary]:
        summaries: list[ExperimentSummary] = []
        for metadata_path in self.root.glob("*/experiment.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if payload.get("storage_schema_version") != STORAGE_SCHEMA_VERSION:
                    continue
                experiment = Experiment.from_dict(payload["experiment"])
                summaries.append(
                    ExperimentSummary(
                        experiment_id=experiment.experiment_id,
                        name=experiment.name,
                        status=experiment.status,
                        created_at=experiment.created_at,
                    )
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(summaries, key=lambda summary: summary.created_at, reverse=True)

    def save_results(self, experiment_id: UUID | str, results: pd.DataFrame) -> None:
        if not isinstance(results, pd.DataFrame):
            raise TypeError("results must be a pandas DataFrame")
        directory = self._experiment_directory(experiment_id)
        if not (directory / "experiment.json").exists():
            raise FileNotFoundError(f"Experiment {experiment_id} was not found")
        csv_content = results.to_csv(index=False)
        self._atomic_write_text(directory / "results.csv", csv_content)

    def load_results(self, experiment_id: UUID | str) -> pd.DataFrame:
        directory = self._experiment_directory(experiment_id)
        if not (directory / "experiment.json").exists():
            raise FileNotFoundError(f"Experiment {experiment_id} was not found")
        results_path = directory / "results.csv"
        if not results_path.exists():
            return pd.DataFrame()
        return pd.read_csv(results_path)
