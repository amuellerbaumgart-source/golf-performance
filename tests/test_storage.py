import pandas as pd
import pytest

from golf_performance.domain import Experiment, ExperimentDesignMode, MetricDefinition, MetricDirection
from golf_performance.protocols import generate_protocol_options
from golf_performance.storage import FileSystemStorage


def experiment() -> Experiment:
    return Experiment(
        name="Driver loft test",
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


def test_experiment_and_selected_protocol_round_trip(tmp_path) -> None:
    storage = FileSystemStorage(tmp_path / "experiments")
    current = experiment()
    recommendation = generate_protocol_options(
        current,
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=30,
    ).select(ExperimentDesignMode.EXPLORATORY)

    storage.save_experiment(current, recommendation)
    restored = storage.load_experiment(current.experiment_id)

    assert restored.experiment.to_dict() == current.to_dict()
    assert restored.protocol_recommendation is not None
    assert restored.selected_mode is ExperimentDesignMode.EXPLORATORY
    assert restored.protocol_recommendation.selected is restored.protocol_recommendation.exploratory


def test_results_round_trip_and_missing_results_are_empty(tmp_path) -> None:
    storage = FileSystemStorage(tmp_path / "experiments")
    current = experiment()
    storage.save_experiment(current)

    assert storage.load_results(current.experiment_id).empty

    results = pd.DataFrame(
        {
            "shot_id": [1, 2],
            "configuration": ["A", "B"],
            "carry": [245.0, 250.0],
        }
    )
    storage.save_results(current.experiment_id, results)

    restored_results = storage.load_results(current.experiment_id)
    pd.testing.assert_frame_equal(restored_results, results)


def test_list_experiments_returns_summaries(tmp_path) -> None:
    storage = FileSystemStorage(tmp_path / "experiments")
    current = experiment()
    storage.save_experiment(current)

    summaries = storage.list_experiments()

    assert len(summaries) == 1
    assert summaries[0].experiment_id == current.experiment_id
    assert summaries[0].name == current.name
    assert summaries[0].changed_variable == current.changed_variable
    assert summaries[0].primary_metric_name == current.primary_metric.display_name


def test_storage_rejects_unknown_results_experiment(tmp_path) -> None:
    storage = FileSystemStorage(tmp_path / "experiments")

    with pytest.raises(FileNotFoundError):
        storage.save_results(
            "00000000-0000-0000-0000-000000000000",
            pd.DataFrame({"carry": [1.0]}),
        )


def test_malformed_metadata_is_reported(tmp_path) -> None:
    storage = FileSystemStorage(tmp_path / "experiments")
    current = experiment()
    directory = storage._experiment_directory(current.experiment_id)
    directory.mkdir(parents=True)
    (directory / "experiment.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid metadata"):
        storage.load_experiment(current.experiment_id)
