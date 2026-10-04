from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ..decision import DecisionCategory
from ..domain import ExperimentStatus
from ..reporting import (
    ExperimentReport,
    ReportNotReadyError,
    build_experiment_report,
    build_report_export,
)
from ..statistics import transform_metric_values
from ..storage import FileSystemStorage


def _build_distribution_chart(report: ExperimentReport, results: pd.DataFrame) -> alt.Chart:
    metric = report.experiment.primary_metric
    values = transform_metric_values(results[metric.key], metric)
    frame = pd.DataFrame(
        {
            "configuration": results.loc[values.index, "configuration"].astype(str).values,
            "value": values.values,
        }
    )
    boxplot = (
        alt.Chart(frame)
        .mark_boxplot(size=55)
        .encode(
            x=alt.X("configuration:N", title="Configuration"),
            y=alt.Y("value:Q", title=f"{metric.display_name} ({metric.unit})"),
            color=alt.Color("configuration:N", title="Configuration"),
        )
    )
    points = (
        alt.Chart(frame)
        .mark_circle(size=35, opacity=0.45)
        .encode(
            x=alt.X("configuration:N", title="Configuration"),
            y=alt.Y("value:Q", title=f"{metric.display_name} ({metric.unit})"),
            color=alt.Color("configuration:N", title="Configuration"),
            tooltip=["configuration", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    return (boxplot + points).properties(
        title=f"{metric.display_name}: baseline A versus treatment B"
    )


def _build_difference_chart(report: ExperimentReport) -> alt.Chart:
    rows = []
    for analysis in report.analyses:
        if (
            analysis.confidence_interval_lower is None
            or analysis.confidence_interval_upper is None
        ):
            continue
        rows.append(
            {
                "metric": analysis.metric_name,
                "difference": analysis.difference,
                "lower": analysis.confidence_interval_lower,
                "upper": analysis.confidence_interval_upper,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return (
            alt.Chart(pd.DataFrame({"message": ["Confidence intervals unavailable"]}))
            .mark_text(size=16)
            .encode(text="message:N")
            .properties(title="Observed differences with 95% confidence intervals")
        )
    interval = (
        alt.Chart(frame)
        .mark_rule(size=4)
        .encode(
            x=alt.X("lower:Q", title="Difference (B − A)"),
            x2="upper:Q",
            y=alt.Y("metric:N", title=None),
            tooltip=["metric", "difference", "lower", "upper"],
        )
    )
    points = (
        alt.Chart(frame)
        .mark_point(size=90, filled=True)
        .encode(
            x=alt.X("difference:Q", title="Difference (B − A)"),
            y=alt.Y("metric:N", title=None),
            tooltip=["metric", "difference", "lower", "upper"],
        )
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="zero:Q")
    return alt.layer(interval, points, zero).properties(
        title="Observed differences with 95% confidence intervals"
    )


def _render_conclusion(report: ExperimentReport) -> None:
    category = report.decision.primary.category
    if category is DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT:
        st.success(report.decision.conclusion)
    elif category in {
        DecisionCategory.PROMISING_BUT_UNCERTAIN,
        DecisionCategory.MEASURABLE_NOT_PRACTICALLY_MEANINGFUL,
    }:
        st.warning(report.decision.conclusion)
    elif category is DecisionCategory.POTENTIALLY_DETRIMENTAL:
        st.error(report.decision.conclusion)
    else:
        st.info(report.decision.conclusion)


def render_report(experiment, recommendation, protocol, storage: FileSystemStorage) -> None:
    st.header("Experiment report", icon=":material/assignment:")
    st.caption("A reproducible summary of the saved experiment, statistical evidence, and practical golf relevance.")

    if experiment is None or recommendation is None or protocol is None:
        st.error("This experiment does not have a selected protocol.")
        return
    try:
        results = storage.load_results(experiment.experiment_id)
        report = build_experiment_report(experiment, protocol, results)
    except (FileNotFoundError, OSError, ReportNotReadyError, TypeError, ValueError) as error:
        st.error(f"The report is not ready: {error}")
        if st.button("Back to analysis", icon=":material/arrow_back:", key="report_back_error"):
            st.session_state.workflow_step = "analysis"
            st.rerun()
        return

    with st.container(border=True):
        st.subheader(report.experiment.name)
        st.markdown(
            f"**Variable tested:** {report.experiment.changed_variable}  \n"
            f"**Baseline A:** {report.experiment.baseline_value}  \n"
            f"**Treatment B:** {report.experiment.treatment_value}  \n"
            f"**Primary goal:** {report.experiment.primary_goal}"
        )
        st.caption(
            f"{report.protocol.label} protocol · {report.total_valid_shots} valid shots · "
            f"{report.protocol.shots_per_configuration} per configuration"
        )

    with st.container(border=True):
        st.subheader("Conclusion", icon=":material/flag:")
        _render_conclusion(report)
        for warning in report.decision.primary.warnings:
            st.warning(warning, icon=":material/warning:")

    primary = report.decision.primary
    primary_analysis = next(item for item in report.analyses if item.metric_key == primary.metric_key)
    with st.container(horizontal=True):
        st.metric("Primary metric", primary.metric_name, border=True)
        st.metric("Observed improvement", f"{primary.observed_improvement:+.2f}", border=True)
        st.metric(
            "Primary p-value",
            f"{primary_analysis.p_value:.4g}" if primary_analysis.p_value is not None else "Unavailable",
            border=True,
        )

    with st.container(border=True):
        st.subheader("Primary metric distribution")
        st.altair_chart(
            _build_distribution_chart(report, results),
            width="stretch",
            alt="Primary metric distributions for baseline A and treatment B",
        )

    with st.container(border=True):
        st.subheader("Difference and uncertainty")
        st.altair_chart(
            _build_difference_chart(report),
            width="stretch",
            alt="Metric differences with 95 percent confidence intervals",
        )

    with st.container(border=True):
        st.subheader("Metric results")
        export_frame = build_report_export(report)
        st.dataframe(export_frame, width="stretch", hide_index=True, alt="Experiment report metric results")
        st.download_button(
            "Download CSV summary",
            data=export_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{report.experiment.name.replace(' ', '_').lower()}_report.csv",
            mime="text/csv",
            icon=":material/download:",
            key="download_report_csv",
        )

    with st.container(border=True):
        st.subheader("Limitations")
        st.markdown(
            "- Primary inference uses complete paired A/B blocks rather than treating every shot as independent.\n"
            "- Repeated shots from one golfer may still be affected by fatigue, learning, and order effects.\n"
            "- The primary result uses paired A/B blocks; secondary metrics are exploratory evidence.\n"
            "- Bootstrap intervals are sensitivity checks and require at least five complete block pairs.\n"
            "- Practical significance is based on the thresholds defined before data review."
        )

    with st.container(horizontal=True, horizontal_alignment="left"):
        if st.button("Back to analysis", icon=":material/arrow_back:", key="report_back"):
            st.session_state.workflow_step = "analysis"
            st.rerun()
        if experiment.status is ExperimentStatus.READY_FOR_ANALYSIS and st.button(
            "Mark experiment analyzed",
            icon=":material/check_circle:",
            key="mark_analyzed",
        ):
            try:
                experiment.transition_to(ExperimentStatus.ANALYZED)
                storage.save_experiment(experiment, recommendation)
                st.session_state.experiment = experiment
                st.success("Experiment marked as analyzed.")
            except (OSError, TypeError, ValueError) as error:
                st.error(f"Could not mark experiment as analyzed: {error}")
