from __future__ import annotations

import streamlit as st

from ..domain import ExperimentStatus
from ..storage import FileSystemStorage


def workflow_step_for_status(status: ExperimentStatus) -> str:
    if status is ExperimentStatus.COLLECTING:
        return "data_entry"
    if status is ExperimentStatus.READY_FOR_ANALYSIS:
        return "analysis"
    if status is ExperimentStatus.ANALYZED:
        return "report"
    if status is ExperimentStatus.PROTOCOL_READY:
        return "protocol"
    return "define"


def open_saved_experiment(storage: FileSystemStorage, experiment_id: str) -> None:
    """Load a saved experiment into the current Streamlit workflow."""

    stored = storage.load_experiment(experiment_id)
    st.session_state.experiment = stored.experiment
    st.session_state.protocol_recommendation = stored.protocol_recommendation
    st.session_state.selected_protocol = (
        stored.protocol_recommendation.selected
        if stored.protocol_recommendation
        else None
    )
    st.session_state.workflow_step = workflow_step_for_status(stored.experiment.status)

