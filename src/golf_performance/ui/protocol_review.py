from __future__ import annotations

import pandas as pd
import streamlit as st

from ..domain import ExperimentDesignMode, ExperimentStatus
from ..protocols import ProtocolRecommendation, TestingProtocol
from ..storage import FileSystemStorage


def _render_option_card(
    protocol: TestingProtocol,
    key: str,
    experiment,
    storage: FileSystemStorage,
) -> None:
    with st.container(border=True, height="stretch"):
        divider = "blue" if protocol.design_mode is ExperimentDesignMode.CONFIRMATORY else "orange"
        st.subheader(protocol.label, divider=divider)
        if protocol.design_mode is ExperimentDesignMode.CONFIRMATORY:
            st.caption("Powered plan for formal evaluation.")
            st.badge("Best for confirmation", icon=":material/verified:", color="blue")
        else:
            st.caption("A practical cap for learning trends before committing to a larger test.")
            st.badge("Best for exploration", icon=":material/explore:", color="orange")

        metrics = st.columns(3)
        metrics[0].metric("Shots / configuration", protocol.shots_per_configuration)
        metrics[1].metric("Total shots", protocol.total_shots)
        metrics[2].metric("Estimated time", f"{protocol.estimated_total_minutes:.1f} min")
        st.caption(
            f"{protocol.blocks_per_configuration} paired A/B blocks per configuration · "
            f"Planning block-difference SD: "
            f"{protocol.power_analysis.expected_block_difference_standard_deviation:.2f} {experiment.primary_metric.unit}"
        )

        st.metric(
            "Minimum detectable effect",
            f"{protocol.minimum_detectable_effect:.2f}",
            help=f"Based on {protocol.power_analysis.analysis_basis}.",
        )
        if protocol.minimum_detectable_effect_percentage is not None:
            st.caption(
                f"Equivalent minimum detectable change: {protocol.minimum_detectable_effect_percentage:.1f}%"
            )

        with st.expander("View assumptions and warnings", icon=":material/info:"):
            st.caption(f"Target power: {protocol.power_analysis.target_power:.0%}")
            st.caption(f"Alpha: {protocol.power_analysis.alpha:g}")
            st.caption(f"Start with: configuration {protocol.start_configuration}")
            warnings = list(protocol.power_analysis.warnings)
            warnings.extend(
                instruction
                for instruction in protocol.instructions
                if instruction.startswith("Exploratory plan:")
                or instruction.startswith("This exploratory cap")
            )
            for warning in dict.fromkeys(warnings):
                st.warning(warning, icon=":material/warning:")

        st.button(
            f"Use {protocol.label} plan",
            key=key,
            type="primary",
            icon=":material/check:",
            on_click=_select_protocol,
            args=(protocol, experiment, storage),
        )


def _select_protocol(
    protocol: TestingProtocol,
    experiment,
    storage: FileSystemStorage,
) -> None:
    try:
        selected_recommendation = st.session_state.protocol_recommendation.select(
            protocol.design_mode
        )
        if experiment.status is ExperimentStatus.DRAFT:
            experiment.transition_to(ExperimentStatus.PROTOCOL_READY)
        storage.save_experiment(experiment, selected_recommendation)
    except (OSError, TypeError, ValueError) as error:
        st.session_state.protocol_save_error = str(error)
        return
    st.session_state.selected_protocol = protocol
    st.session_state.protocol_recommendation = selected_recommendation


def render_protocol_review(
    experiment,
    recommendation: ProtocolRecommendation,
    storage: FileSystemStorage,
) -> None:
    protocol_save_error = st.session_state.pop("protocol_save_error", None)
    if protocol_save_error:
        st.error(f"Could not save selected protocol: {protocol_save_error}")
    st.header("Choose your testing plan", icon=":material/compare_arrows:")
    st.caption("Both plans test the same A/B comparison. Choose how much evidence you want before moving forward.")

    summary_columns = st.columns(3)
    summary_columns[0].metric("Variable", experiment.changed_variable)
    summary_columns[1].metric("Baseline A", experiment.baseline_value)
    summary_columns[2].metric("Treatment B", experiment.treatment_value)
    with st.container(border=True):
        st.markdown(f"**Primary goal**  \n{experiment.primary_goal}")

    columns = st.columns(2)
    with columns[0]:
        if recommendation.confirmatory is None:
            with st.container(border=True, height="stretch"):
                st.subheader("Confirmatory", divider="blue")
                st.caption("The powered plan is above the operational limit.")
                st.badge("Unavailable at this cap", icon=":material/warning:", color="orange")
                st.warning(
                    f"The powered recommendation requires "
                    f"{recommendation.confirmatory_shots_required} shots per configuration, "
                    "so choose the exploratory plan to collect a practical sample.",
                    icon=":material/warning:",
                )
                st.caption("The exploratory result can show trends and large effects, but it will not provide definitive confirmation.")
        else:
            _render_option_card(recommendation.confirmatory, "select_confirmatory", experiment, storage)
    with columns[1]:
        _render_option_card(recommendation.exploratory, "select_exploratory", experiment, storage)

    selected = recommendation.selected
    if selected is not None:
        st.success(f"Selected plan: {selected.label}")
        st.subheader("Exact protocol sequence", icon=":material/list:")
        sequence_frame = pd.DataFrame(
            {
                "shot_number": range(1, selected.total_shots + 1),
                "configuration": selected.sequence,
            }
        )
        st.dataframe(
            sequence_frame,
            width="stretch",
            hide_index=True,
            height=min(420, 36 + selected.total_shots * 35),
        )
        st.caption("The selected sequence will drive the future shot-entry table.")
        if st.button("Continue to shot entry", type="primary", icon=":material/edit_note:", key="continue_to_data_entry"):
            st.session_state.workflow_step = "data_entry"
            st.rerun()

    with st.container(horizontal=True, horizontal_alignment="left"):
        back = st.button("Back to experiment definition", icon=":material/arrow_back:")
    if back:
        st.session_state.workflow_step = "define"
        st.session_state.selected_protocol = None
        st.rerun()
