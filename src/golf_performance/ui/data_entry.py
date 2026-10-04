from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data_collection import (
    build_results_template,
    completed_shots,
    merge_saved_results,
    validate_results,
)
from ..domain import ExperimentStatus
from ..protocols import ProtocolRecommendation, TestingProtocol
from ..storage import FileSystemStorage


def _metric_columns(experiment) -> list[str]:
    return [metric.key for metric in experiment.metrics]


def _save_results(
    experiment,
    recommendation: ProtocolRecommendation,
    protocol: TestingProtocol,
    results: pd.DataFrame,
    storage: FileSystemStorage,
    collection_notes: str,
) -> None:
    metric_keys = _metric_columns(experiment)
    errors = validate_results(results, protocol, metric_keys)
    if errors:
        st.error("Please fix the results table before saving.")
        for error in errors:
            st.caption(f":material/error: {error}")
        return

    try:
        experiment.collection_notes = collection_notes.strip()
        storage.save_results(experiment.experiment_id, results)
        if experiment.status is ExperimentStatus.PROTOCOL_READY:
            experiment.transition_to(ExperimentStatus.COLLECTING)
        is_complete = completed_shots(results, metric_keys) == protocol.total_shots
        if is_complete and experiment.status is ExperimentStatus.COLLECTING:
            experiment.transition_to(ExperimentStatus.READY_FOR_ANALYSIS)
        storage.save_experiment(experiment, recommendation)
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Could not save shot data: {error}")
        return

    st.success("Shot data saved. You can close the app and resume this experiment later.")
    st.session_state.experiment = experiment
    if is_complete:
        st.session_state.workflow_step = "analysis"
        st.rerun()


def render_data_entry(
    experiment,
    recommendation: ProtocolRecommendation,
    protocol: TestingProtocol,
    storage: FileSystemStorage,
) -> None:
    st.header("Enter shot results", icon=":material/edit_note:")
    st.caption("Enter only the measured values. Shot order and A/B assignments are locked to the selected protocol.")

    metric_keys = _metric_columns(experiment)
    template = build_results_template(protocol, metric_keys)
    try:
        saved_results = storage.load_results(experiment.experiment_id)
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not load saved shot data: {error}")
        saved_results = pd.DataFrame()
    results = merge_saved_results(template, saved_results)
    complete_count = completed_shots(results, metric_keys)

    summary_columns = st.columns(3)
    summary_columns[0].metric("Completed shots", f"{complete_count} / {protocol.total_shots}")
    summary_columns[1].metric("Per configuration", protocol.shots_per_configuration)
    summary_columns[2].metric("Plan", protocol.label)

    with st.container(border=True):
        st.markdown(f"**{experiment.changed_variable}:** {experiment.baseline_value} (A) vs {experiment.treatment_value} (B)")
        st.caption("Blank cells are allowed while collecting. A shot is complete only when every selected metric has a finite value.")
        st.caption("Block IDs and configurations are locked. Add an exclusion note when a shot is affected by a predefined rule, mishit, interruption, or unusual condition.")
        if experiment.exclusion_rules:
            st.info(f"Predefined exclusion rules: {experiment.exclusion_rules}", icon=":material/rule:")

    column_config = {
        "shot_id": st.column_config.NumberColumn("Shot", disabled=True, pinned=True, format="%d"),
        "block_id": st.column_config.NumberColumn("Block", disabled=True, pinned=True, format="%d"),
        "configuration": st.column_config.TextColumn("Configuration", disabled=True, pinned=True),
        "exclusion_note": st.column_config.TextColumn(
            "Exclusion note",
            help="Optional context for an excluded or unusual shot.",
        ),
    }
    for metric in experiment.metrics:
        column_config[metric.key] = st.column_config.NumberColumn(
            metric.display_name,
            help=f"Enter {metric.display_name} in {metric.unit}.",
            format="%.2f",
            step=0.01,
        )

    edited_results = st.data_editor(
        results,
        key=f"results_editor_{experiment.experiment_id}",
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        disabled=["shot_id", "block_id", "configuration"],
        column_config=column_config,
    )

    collection_notes = st.text_area(
        "Collection notes",
        value=experiment.collection_notes,
        placeholder="Record fatigue, interruptions, weather changes, or order concerns.",
        height=80,
        key=f"collection_notes_{experiment.experiment_id}",
    )
    with st.container(horizontal=True, horizontal_alignment="left"):
        save = st.button("Save progress", type="primary", icon=":material/save:", key="save_results")
        back = st.button("Back to protocol", icon=":material/arrow_back:", key="back_to_protocol")
    if save:
        _save_results(experiment, recommendation, protocol, edited_results, storage, collection_notes)
    if back:
        st.session_state.workflow_step = "protocol"
        st.rerun()
