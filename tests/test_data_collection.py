import pandas as pd

from golf_performance.data_collection import (
    build_results_template,
    completed_shots,
    merge_saved_results,
    normalize_saved_results,
    validate_results,
)
from golf_performance.domain import (
    Experiment,
    MetricDefinition,
    MetricDirection,
)
from golf_performance.protocols import generate_protocol_options


def experiment() -> Experiment:
    return Experiment(
        name="Entry test",
        changed_variable="Loft",
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


def protocol():
    recommendation = generate_protocol_options(
        experiment(),
        expected_standard_deviation=8.0,
        exploratory_shots_per_configuration=10,
        block_size=5,
    )
    return recommendation.exploratory


def test_template_locks_protocol_rows_and_allows_blank_metrics() -> None:
    current = protocol()
    results = build_results_template(current, ["carry"])

    assert list(results.columns) == ["shot_id", "block_id", "configuration", "exclusion_note", "carry"]
    assert len(results) == current.total_shots
    assert results["block_id"].tolist() == [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
    assert results["configuration"].tolist() == list(current.sequence)
    assert completed_shots(results, ["carry"]) == 0
    assert validate_results(results, current, ["carry"]) == []


def test_saved_values_are_merged_by_shot_id() -> None:
    current = protocol()
    template = build_results_template(current, ["carry"])
    saved = pd.DataFrame({"shot_id": [2], "carry": [245.5], "exclusion_note": ["top edge"]})

    merged = merge_saved_results(template, saved)

    assert merged.loc[1, "carry"] == 245.5
    assert merged.loc[1, "exclusion_note"] == "top edge"
    assert pd.isna(merged.loc[0, "carry"])


def test_legacy_saved_results_are_normalized_to_current_schema() -> None:
    current = protocol()
    legacy = pd.DataFrame({"shot_id": [1], "configuration": ["B"], "carry": [245.5]})

    normalized = normalize_saved_results(legacy, current, ["carry"])

    assert list(normalized.columns) == ["shot_id", "block_id", "configuration", "exclusion_note", "carry"]
    assert normalized.loc[0, "configuration"] == current.sequence[0]
    assert normalized.loc[0, "carry"] == 245.5


def test_results_validation_rejects_non_numeric_values() -> None:
    current = protocol()
    results = build_results_template(current, ["carry"])
    results["carry"] = results["carry"].astype(object)
    results.loc[0, "carry"] = "not a number"

    errors = validate_results(results, current, ["carry"])

    assert any("non-numeric" in error for error in errors)


def test_results_validation_rejects_modified_block_ids() -> None:
    current = protocol()
    results = build_results_template(current, ["carry"])
    results.loc[0, "block_id"] = 99

    errors = validate_results(results, current, ["carry"])

    assert any("Block IDs" in error for error in errors)


def test_complete_results_count_and_validation() -> None:
    current = protocol()
    results = build_results_template(current, ["carry"])
    results["carry"] = 245.0

    assert validate_results(results, current, ["carry"]) == []
    assert completed_shots(results, ["carry"]) == current.total_shots
