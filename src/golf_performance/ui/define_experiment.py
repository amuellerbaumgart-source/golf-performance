from __future__ import annotations

import streamlit as st

from ..domain import MetricDirection, ThresholdType
from ..storage import FileSystemStorage
from .forms import (
    METRIC_PRESETS,
    build_experiment,
    build_metric_definition,
)


def render_define_experiment(storage: FileSystemStorage) -> None:
    st.header("Define your experiment", icon=":material/tune:")
    st.caption("Start with the question you want the experiment to answer, then define exactly one change.")

    st.subheader("Experiment setup")
    setup_columns = st.columns(2)
    with setup_columns[0]:
        name = st.text_input("Experiment name", placeholder="Driver loft test", key="experiment_name")
    with setup_columns[1]:
        changed_variable = st.text_input("Single variable being changed", placeholder="Driver loft", key="changed_variable")

    value_columns = st.columns(2)
    with value_columns[0]:
        baseline_value = st.text_input("Baseline A", placeholder="9 degrees", key="baseline_value")
    with value_columns[1]:
        treatment_value = st.text_input("Treatment B", placeholder="10 degrees", key="treatment_value")

    primary_goal = st.text_area(
        "Performance goal",
        placeholder="Increase carry without worsening dispersion",
        height=80,
        key="primary_goal",
    )

    selected_metric_labels = st.pills(
        "Metrics to measure",
        options=[preset.display_name for preset in METRIC_PRESETS],
        default=["Carry"],
        selection_mode="multi",
        wrap=True,
        key="selected_metrics",
    )
    selected_keys = [
        next(preset.key for preset in METRIC_PRESETS if preset.display_name == label)
        for label in (selected_metric_labels or [])
    ]

    st.subheader("What will you measure?")
    st.caption("Choose one primary metric. Additional metrics help identify tradeoffs.")
    primary_metric_key = None
    if selected_keys:
        primary_metric_label = st.selectbox(
            "Primary metric",
            options=[
                next(preset.display_name for preset in METRIC_PRESETS if preset.key == key)
                for key in selected_keys
            ],
            key="primary_metric",
        )
        primary_metric_key = next(
            preset.key
            for preset in METRIC_PRESETS
            if preset.display_name == primary_metric_label
        )
    metric_inputs: dict[str, dict[str, float | str | None]] = {}
    for index, key in enumerate(selected_keys):
        preset = next(preset for preset in METRIC_PRESETS if preset.key == key)
        with st.expander(
            f"{'Primary metric · ' if key == primary_metric_key else ''}{preset.display_name}",
            expanded=key == primary_metric_key,
        ):
            st.caption(
                f"Direction: {preset.direction.value}. Analysis quantity: {preset.transform.value}."
            )
            threshold_type = st.selectbox(
                "Meaningful threshold",
                options=[ThresholdType.ABSOLUTE, ThresholdType.PERCENTAGE],
                format_func=lambda value: "Absolute" if value is ThresholdType.ABSOLUTE else "Percentage",
                key=f"threshold_type_{key}",
            )
            threshold = st.number_input(
                f"Minimum meaningful improvement ({preset.unit if threshold_type is ThresholdType.ABSOLUTE else '%'})",
                min_value=0.0001,
                value=3.0 if key == "carry" else 1.0,
                step=0.5,
                key=f"threshold_{key}",
            )
            target_value = None
            if preset.direction is MetricDirection.TARGET:
                target_value = st.number_input(
                    f"Target value ({preset.unit})",
                    value=15.0 if key == "launch_angle" else 2500.0,
                    step=1.0,
                    key=f"target_{key}",
                )
            metric_inputs[key] = {
                "threshold": threshold,
                "threshold_type": threshold_type,
                "target_value": target_value,
            }

    expected_baseline_mean = None
    if primary_metric_key and metric_inputs[primary_metric_key]["threshold_type"] is ThresholdType.PERCENTAGE:
        expected_baseline_mean = st.number_input(
            "Expected baseline mean of the primary analysis quantity",
            min_value=0.0001,
            value=100.0,
            step=1.0,
            help="Required to convert the percentage threshold into measurement units for power analysis.",
        )

    st.subheader("Testing assumptions")
    st.caption("These assumptions determine how much data the app recommends collecting.")
    power_columns = st.columns(3)
    with power_columns[0]:
        expected_standard_deviation = st.number_input(
            "Expected SD of primary analysis quantity",
            min_value=0.0001,
            value=8.0,
            step=0.5,
            key="expected_standard_deviation",
            help="Estimate the shot-to-shot standard deviation in the transformed quantity being analyzed.",
        )
    with power_columns[1]:
        exploratory_cap = st.number_input(
            "Exploratory shots per configuration",
            min_value=5,
            value=30,
            step=5,
            key="exploratory_cap",
        )
    with power_columns[2]:
        seconds_per_shot = st.number_input(
            "Estimated seconds per shot",
            min_value=1.0,
            value=30.0,
            step=5.0,
            key="seconds_per_shot",
        )

    block_size = st.number_input("Shots per protocol block", min_value=1, value=5, step=1, key="block_size")
    submitted = st.button(
        "Generate protocol options",
        type="primary",
        icon=":material/arrow_forward:",
        key="generate_protocol_options",
    )

    if not submitted:
        return

    try:
        if not selected_keys:
            raise ValueError("Select at least one metric")

        metrics = [
            build_metric_definition(
                key=key,
                threshold=float(metric_inputs[key]["threshold"]),
                threshold_type=metric_inputs[key]["threshold_type"],  # type: ignore[arg-type]
                target_value=metric_inputs[key]["target_value"],  # type: ignore[arg-type]
                is_primary=key == primary_metric_key,
            )
            for index, key in enumerate(selected_keys)
        ]

        experiment = build_experiment(
            name=name,
            changed_variable=changed_variable,
            baseline_value=baseline_value,
            treatment_value=treatment_value,
            primary_goal=primary_goal,
            metrics=metrics,
        )
        from ..protocols import generate_protocol_options

        recommendation = generate_protocol_options(
            experiment,
            expected_standard_deviation=float(expected_standard_deviation),
            exploratory_shots_per_configuration=int(exploratory_cap),
            block_size=int(block_size),
            seconds_per_shot=float(seconds_per_shot),
            expected_baseline_mean=expected_baseline_mean,
        )
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return

    try:
        storage.save_experiment(experiment, recommendation)
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Could not save experiment: {error}")
        return

    st.session_state.experiment = experiment
    st.session_state.protocol_recommendation = recommendation
    st.session_state.selected_protocol = None
    st.session_state.workflow_step = "protocol"
    st.rerun()
