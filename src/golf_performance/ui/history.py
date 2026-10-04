from __future__ import annotations

import pandas as pd
import streamlit as st

from ..domain import ExperimentStatus
from ..storage import ExperimentSummary, FileSystemStorage
from .workflow import open_saved_experiment


_STATUS_LABELS = {
    "All": None,
    "In progress": {
        ExperimentStatus.DRAFT,
        ExperimentStatus.PROTOCOL_READY,
        ExperimentStatus.COLLECTING,
        ExperimentStatus.READY_FOR_ANALYSIS,
    },
    "Analyzed": {ExperimentStatus.ANALYZED},
}


def _status_label(status: ExperimentStatus) -> str:
    return {
        ExperimentStatus.DRAFT: "Draft",
        ExperimentStatus.PROTOCOL_READY: "Protocol ready",
        ExperimentStatus.COLLECTING: "Collecting data",
        ExperimentStatus.READY_FOR_ANALYSIS: "Ready for analysis",
        ExperimentStatus.ANALYZED: "Analyzed",
    }[status]


def _summary_frame(summaries: list[ExperimentSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Experiment": summary.name,
                "Variable": summary.changed_variable,
                "Baseline A": summary.baseline_value,
                "Treatment B": summary.treatment_value,
                "Primary metric": summary.primary_metric_name,
                "Status": _status_label(summary.status),
                "Created": summary.created_at.astimezone().strftime("%Y-%m-%d"),
            }
            for summary in summaries
        ]
    )


def render_history(storage: FileSystemStorage) -> None:
    st.header("Experiment history", icon=":material/history:")
    st.caption("Review saved experiments and reopen any design, data-entry, analysis, or report workflow.")

    summaries = storage.list_experiments()
    if not summaries:
        st.info("No saved experiments yet. Create your first experiment to start building a useful history.")
        if st.button("Create an experiment", type="primary", icon=":material/add:", key="create_from_history"):
            st.session_state.workflow_step = "define"
            st.rerun()
        return

    status_filter = st.selectbox(
        "Show",
        options=list(_STATUS_LABELS),
        key="history_status_filter",
    )
    allowed_statuses = _STATUS_LABELS[status_filter]
    filtered = [
        summary for summary in summaries
        if allowed_statuses is None or summary.status in allowed_statuses
    ]

    counts = {
        "Total experiments": len(summaries),
        "In progress": sum(summary.status is not ExperimentStatus.ANALYZED for summary in summaries),
        "Analyzed": sum(summary.status is ExperimentStatus.ANALYZED for summary in summaries),
    }
    with st.container(horizontal=True):
        for label, value in counts.items():
            st.metric(label, value, border=True)

    if not filtered:
        st.info("No experiments match this filter.")
        return

    with st.container(border=True):
        st.subheader(f"{len(filtered)} saved experiment{'s' if len(filtered) != 1 else ''}")
        st.dataframe(
            _summary_frame(filtered),
            width="stretch",
            hide_index=True,
            alt="Saved golf experiments",
        )

    options = [str(summary.experiment_id) for summary in filtered]
    labels = {
        str(summary.experiment_id): f"{summary.name} · {_status_label(summary.status)}"
        for summary in filtered
    }
    selected_id = st.selectbox(
        "Open experiment",
        options=options,
        format_func=lambda value: labels[value],
        key="history_selected_experiment",
    )
    if st.button(
        "Open selected experiment",
        type="primary",
        icon=":material/folder_open:",
        key="open_history_experiment",
    ):
        try:
            open_saved_experiment(storage, selected_id)
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(f"Could not open experiment: {error}")
            return
        st.rerun()

