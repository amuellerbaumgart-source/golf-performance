from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data_collection import completed_shots, validate_results
from ..decision import DecisionCategory, ExperimentDecision, evaluate_experiment
from ..domain import ThresholdType
from ..statistics import MetricAnalysis, analyze_experiment
from ..storage import FileSystemStorage


def build_metric_comparison_table(analyses: tuple[MetricAnalysis, ...]) -> pd.DataFrame:
    """Build a readable, UI-facing comparison table from analysis results."""

    rows: list[dict[str, object]] = []
    for analysis in analyses:
        confidence_interval = "Unavailable"
        if (
            analysis.confidence_interval_lower is not None
            and analysis.confidence_interval_upper is not None
        ):
            confidence_interval = (
                f"[{analysis.confidence_interval_lower:.2f}, "
                f"{analysis.confidence_interval_upper:.2f}]"
            )
        rows.append(
            {
                "Metric": analysis.metric_name,
                "Basis": analysis.analysis_basis,
                "A n": analysis.baseline.n,
                "A mean": analysis.baseline.mean,
                "A median": analysis.baseline.median,
                "A SD": analysis.baseline.standard_deviation,
                "B n": analysis.treatment.n,
                "B mean": analysis.treatment.mean,
                "B median": analysis.treatment.median,
                "B SD": analysis.treatment.standard_deviation,
                "Paired blocks": analysis.n_pairs,
                "B − A": analysis.difference,
                "% difference": analysis.percentage_difference,
                "95% CI for B − A": confidence_interval,
                "Bootstrap 95% CI": (
                    f"[{analysis.bootstrap_confidence_interval_lower:.2f}, "
                    f"{analysis.bootstrap_confidence_interval_upper:.2f}]"
                    if analysis.bootstrap_confidence_interval_lower is not None
                    and analysis.bootstrap_confidence_interval_upper is not None
                    else "Unavailable"
                ),
                "p-value": analysis.p_value,
                "Hedges' g": analysis.hedges_g,
            }
        )
    return pd.DataFrame(rows)


def build_decision_table(decision: ExperimentDecision) -> pd.DataFrame:
    """Build a compact table separating statistical and practical significance."""

    rows = []
    for metric_decision in decision.metrics:
        rows.append(
            {
                "Metric": metric_decision.metric_name,
                "Evidence role": "Confirmatory" if metric_decision.confirmatory else "Exploratory",
                "Observed improvement": metric_decision.observed_improvement,
                "Practical threshold": metric_decision.practical_threshold,
                "Threshold type": (
                    "%" if metric_decision.threshold_type is ThresholdType.PERCENTAGE else "absolute"
                ),
                "Statistically significant": (
                    "Yes" if metric_decision.statistically_significant is True
                    else "No" if metric_decision.statistically_significant is False
                    else "Unavailable"
                ),
                "Practically meaningful": (
                    "Yes" if metric_decision.practically_meaningful is True
                    else "No" if metric_decision.practically_meaningful is False
                    else "Unavailable"
                ),
                "Decision": metric_decision.category.value.replace("_", " ").title(),
            }
        )
    return pd.DataFrame(rows)


def _render_readiness_block(experiment, protocol, results: pd.DataFrame) -> bool:
    metric_keys = [metric.key for metric in experiment.metrics]
    validation_errors = validate_results(results, protocol, metric_keys)
    complete_count = completed_shots(results, metric_keys)
    if validation_errors:
        st.error("Saved results do not match the selected protocol.")
        for error in validation_errors:
            st.caption(f":material/error: {error}")
        return False
    if complete_count < protocol.total_shots:
        st.warning(
            f"Analysis is locked until all {protocol.total_shots} protocol shots have valid values. "
            f"Current progress: {complete_count} complete shots.",
            icon=":material/hourglass_top:",
        )
        return False
    return True


def render_analysis(experiment, recommendation, protocol, storage: FileSystemStorage) -> None:
    st.header("Analyze results", icon=":material/analytics:")
    st.caption("Statistical significance and practical golf significance are evaluated separately.")

    if experiment is None or recommendation is None or protocol is None:
        st.error("This experiment does not have a selected protocol.")
        return

    try:
        results = storage.load_results(experiment.experiment_id)
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not load saved results: {error}")
        return

    with st.container(border=True):
        st.markdown(
            f"**{experiment.name}**  \n"
            f"{experiment.changed_variable}: {experiment.baseline_value} (A) vs {experiment.treatment_value} (B)"
        )
        st.caption(f"Selected protocol: {protocol.label} · {protocol.total_shots} total shots")

    if not _render_readiness_block(experiment, protocol, results):
        with st.container(horizontal=True, horizontal_alignment="left"):
            if st.button("Back to shot entry", icon=":material/arrow_back:", key="analysis_back_incomplete"):
                st.session_state.workflow_step = "data_entry"
                st.rerun()
        return

    try:
        analyses = analyze_experiment(results, experiment, protocol)
    except (TypeError, ValueError) as error:
        st.error(f"Analysis could not be completed: {error}")
        return

    decision = evaluate_experiment(analyses, experiment)

    primary_analysis = next(
        analysis for analysis in analyses if analysis.metric_key == experiment.primary_metric.key
    )
    with st.container(horizontal=True):
        st.metric("Valid shots", sum(analysis.baseline.n + analysis.treatment.n for analysis in analyses[:1]))
        st.metric(
            f"Primary change ({primary_analysis.unit})",
            f"{primary_analysis.difference:+.2f}",
            delta_description="Treatment B minus baseline A",
        )
        st.metric(
            "Primary p-value",
            f"{primary_analysis.p_value:.4g}"
            if primary_analysis.p_value is not None
            else "Unavailable",
        )

    with st.container(border=True):
        st.subheader("Metric comparison")
        st.dataframe(
            build_metric_comparison_table(analyses),
            width="stretch",
            hide_index=True,
            alt="Baseline and treatment statistical comparison by metric",
        )

    with st.container(border=True):
        st.subheader("Decision summary", icon=":material/flag:")
        if decision.primary.category is DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT:
            st.success(decision.conclusion)
        elif decision.primary.category in {
            DecisionCategory.PROMISING_BUT_UNCERTAIN,
            DecisionCategory.MEASURABLE_NOT_PRACTICALLY_MEANINGFUL,
        }:
            st.warning(decision.conclusion)
        elif decision.primary.category is DecisionCategory.POTENTIALLY_DETRIMENTAL:
            st.error(decision.conclusion)
        else:
            st.info(decision.conclusion)
        st.dataframe(
            build_decision_table(decision),
            width="stretch",
            hide_index=True,
            alt="Statistical and practical significance decision by metric",
        )
        if st.button(
            "View experiment report",
            type="primary",
            icon=":material/assignment:",
            key="view_report",
        ):
            st.session_state.workflow_step = "report"
            st.rerun()

    warning_count = 0
    for analysis in analyses:
        if not analysis.warnings:
            continue
        with st.expander(f"Analysis notes · {analysis.metric_name}", icon=":material/info:"):
            for warning in analysis.warnings:
                st.warning(warning, icon=":material/warning:")
                warning_count += 1
    if warning_count == 0:
        st.success("No statistical warnings were generated for the selected metrics.")

    with st.container(horizontal=True, horizontal_alignment="left"):
        if st.button("Back to shot entry", icon=":material/arrow_back:", key="analysis_back"):
            st.session_state.workflow_step = "data_entry"
            st.rerun()
