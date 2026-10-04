import streamlit as st

from golf_performance.domain import ExperimentStatus
from golf_performance.storage import FileSystemStorage
from golf_performance.ui.define_experiment import render_define_experiment
from golf_performance.ui.data_entry import render_data_entry
from golf_performance.ui.analysis import render_analysis
from golf_performance.ui.protocol_review import render_protocol_review
from golf_performance.ui.report import render_report


st.set_page_config(
    page_title="Golf performance experiments",
    page_icon=":material/sports_golf:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    st.session_state.setdefault("workflow_step", "define")
    st.session_state.setdefault("experiment", None)
    st.session_state.setdefault("protocol_recommendation", None)
    st.session_state.setdefault("selected_protocol", None)


def reset_workflow() -> None:
    st.session_state.workflow_step = "define"
    st.session_state.experiment = None
    st.session_state.protocol_recommendation = None
    st.session_state.selected_protocol = None
    st.session_state.pop("saved_experiment_id", None)


def render_sidebar(storage: FileSystemStorage) -> None:
    with st.sidebar:
        st.markdown("### Experiment workflow")
        current_step = st.session_state.workflow_step
        steps = (
            ("define", "Define experiment"),
            ("protocol", "Choose testing plan"),
            ("data_entry", "Enter shot results"),
            ("analysis", "Analyze results"),
            ("report", "Experiment report"),
        )
        step_order = [step_key for step_key, _ in steps]
        current_index = step_order.index(current_step) if current_step in step_order else 0
        for index, (step_key, label) in enumerate(steps):
            if index < current_index:
                icon = ":material/check_circle:"
            elif index == current_index:
                icon = ":material/radio_button_checked:"
            else:
                icon = ":material/lock:"
            st.markdown(f"{icon} {label}")
        if st.button("New experiment", key="new_experiment", icon=":material/add:"):
            reset_workflow()
            st.rerun()

        summaries = storage.list_experiments()
        if summaries:
            st.subheader("Saved experiments")
            summary_ids = [str(summary.experiment_id) for summary in summaries]
            summary_labels = {
                str(summary.experiment_id): f"{summary.name} · {summary.status.value}"
                for summary in summaries
            }
            selected_id = st.selectbox(
                "Open experiment",
                options=summary_ids,
                format_func=lambda value: summary_labels[value],
                index=None,
                placeholder="Choose an experiment",
                key="saved_experiment_id",
            )
            if st.button(
                "Open selected experiment",
                key="open_saved_experiment",
                icon=":material/folder_open:",
                disabled=selected_id is None,
            ):
                try:
                    stored = storage.load_experiment(selected_id)
                except (FileNotFoundError, OSError, ValueError) as error:
                    st.error(f"Could not open experiment: {error}")
                    return
                st.session_state.experiment = stored.experiment
                st.session_state.protocol_recommendation = stored.protocol_recommendation
                st.session_state.selected_protocol = (
                    stored.protocol_recommendation.selected
                    if stored.protocol_recommendation
                    else None
                )
                if not stored.protocol_recommendation:
                    workflow_step = "define"
                elif stored.experiment.status is ExperimentStatus.COLLECTING:
                    workflow_step = "data_entry"
                elif stored.experiment.status is ExperimentStatus.READY_FOR_ANALYSIS:
                    workflow_step = "analysis"
                elif stored.experiment.status is ExperimentStatus.ANALYZED:
                    workflow_step = "report"
                else:
                    workflow_step = "protocol"
                st.session_state.workflow_step = workflow_step
                st.rerun()
        st.space("small")
        st.caption("One variable changes. Everything else stays as constant as reasonably possible.")


initialize_session_state()
storage = FileSystemStorage()
render_sidebar(storage)

st.title("Golf performance experiments", icon=":material/sports_golf:")
st.caption("Design a controlled A/B test for one golf variable at a time.")

if st.session_state.workflow_step == "protocol":
    render_protocol_review(
        st.session_state.experiment,
        st.session_state.protocol_recommendation,
        storage,
    )
elif st.session_state.workflow_step == "data_entry":
    render_data_entry(
        st.session_state.experiment,
        st.session_state.protocol_recommendation,
        st.session_state.selected_protocol,
        storage,
    )
elif st.session_state.workflow_step == "analysis":
    render_analysis(
        st.session_state.experiment,
        st.session_state.protocol_recommendation,
        st.session_state.selected_protocol,
        storage,
    )
elif st.session_state.workflow_step == "report":
    render_report(
        st.session_state.experiment,
        st.session_state.protocol_recommendation,
        st.session_state.selected_protocol,
        storage,
    )
else:
    render_define_experiment(storage)
