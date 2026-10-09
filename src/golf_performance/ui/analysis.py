from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data_collection import completed_shots, normalize_saved_results, validate_results
from ..decision import DecisionCategory, ExperimentDecision, evaluate_experiment
from ..domain import MetricDirection, ThresholdType
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
                "Observed estimate meets threshold": (
                    "Yes" if metric_decision.practically_meaningful is True
                    else "No" if metric_decision.practically_meaningful is False
                    else "Unavailable"
                ),
                "Decision": metric_decision.label,
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
    results = normalize_saved_results(results, protocol, [metric.key for metric in experiment.metrics])

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

    decision = evaluate_experiment(analyses, experiment, design_mode=protocol.design_mode, alpha=protocol.power_analysis.alpha)

    primary_analysis = next(
        analysis for analysis in analyses if analysis.metric_key == experiment.primary_metric.key
    )
    primary = decision.primary
    metric = experiment.primary_metric
    st.subheader(f"Your result · {primary_analysis.metric_name}")
    st.badge(
        "Confirmatory primary result" if primary.confirmatory else "Exploratory result",
        icon=":material/science:", color="blue" if primary.confirmatory else "orange",
    )
    if not primary.confirmatory:
        st.info("Use this experiment to identify promising changes and plan a follow-up test. All results from this plan are exploratory.")
    if primary.category is DecisionCategory.POTENTIALLY_DETRIMENTAL:
        st.error(decision.conclusion)
    elif primary.category is DecisionCategory.STRONG_MEANINGFUL_IMPROVEMENT and primary.confirmatory:
        st.success(decision.conclusion)
    else:
        st.info(decision.conclusion)

    threshold_unit = "%" if metric.threshold_type is ThresholdType.PERCENTAGE else primary_analysis.unit
    with st.container(horizontal=True):
        st.metric("Observed improvement", f"{primary.observed_improvement:+.2f} {primary_analysis.unit}",
                  border=True, help="Positive favors B; negative favors A. Target metrics compare distance from the target.")
        st.metric("Your worthwhile improvement", f"{metric.meaningful_threshold:g} {threshold_unit}", border=True)
        st.metric("Complete block pairs", str(primary_analysis.n_pairs), border=True,
                  help="The number of paired comparisons used to estimate uncertainty.")

    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("How certain is the change?", icon=":material/query_stats:")
        lo, hi = primary_analysis.confidence_interval_lower, primary_analysis.confidence_interval_upper
        if lo is not None and hi is not None:
            if metric.direction is not MetricDirection.HIGHER:
                lo, hi = -hi, -lo
            st.metric("95% interval for improvement", f"{lo:+.2f} to {hi:+.2f} {primary_analysis.unit}")
            st.caption("Positive values favor B. An interval crossing zero leaves both improvement and deterioration plausible. A narrower interval means a more precise estimate.")
        else:
            st.write("An uncertainty interval is unavailable.")
            st.caption("Review the analysis notes before drawing a conclusion.")
        if primary_analysis.p_value is not None:
            st.caption(f"p-value: {primary_analysis.p_value:.4g} · significance cutoff: {protocol.power_analysis.alpha:g}. This tests zero change, not your practical threshold.")
    with right.container(border=True):
        st.subheader("Is it large enough to matter?", icon=":material/flag:")
        st.write("The observed estimate meets your threshold." if primary.practically_meaningful is True
                 else "The observed estimate is below your threshold." if primary.practically_meaningful is False
                 else "The practical comparison is unavailable.")
        if metric.threshold_type is ThresholdType.PERCENTAGE and primary.observed_improvement_percentage is not None:
            st.write(f"Observed improvement: {primary.observed_improvement_percentage:+.2f}% · goal: {metric.meaningful_threshold:g}%")
        st.caption("This compares your goal with the estimate. It does not establish that the true improvement exceeds your goal.")
        st.write(f"A average: {primary_analysis.baseline.mean:.2f} {primary_analysis.unit}")
        st.write(f"B average: {primary_analysis.treatment.mean:.2f} {primary_analysis.unit}")
        st.caption(f"Measured as {primary_analysis.analysis_basis}.")

    secondary = tuple(item for item in analyses if item.metric_key != metric.key)
    if secondary:
        st.subheader("Other metrics & tradeoffs")
        st.caption("These results are exploratory, even in a confirmatory plan.")
        for item in secondary:
            item_decision = next(d for d in decision.metrics if d.metric_key == item.metric_key)
            with st.expander(item.metric_name):
                st.write(item_decision.interpretation)
                st.metric("Observed improvement", f"{item_decision.observed_improvement:+.2f} {item.unit}")

    with st.expander("Full statistical details", icon=":material/table_chart:"):
        st.caption("The tables retain B − A signs. For lower-is-better metrics, a negative difference is an improvement.")
        st.caption("Baseline and treatment statistics with confidence intervals")
        st.dataframe(build_metric_comparison_table(analyses), width="stretch", hide_index=True)
        st.caption("Evidence roles and observed practical threshold comparisons")
        st.dataframe(build_decision_table(decision), width="stretch", hide_index=True)

    notes = list(dict.fromkeys(
        f"{item.metric_name}: {warning}"
        for item in analyses for warning in item.warnings
    ))
    if notes:
        with st.expander(f"Analysis notes ({len(notes)})", icon=":material/info:"):
            for note in notes:
                st.write(note)

    with st.container(horizontal=True):
        if st.button("View experiment report", type="primary", icon=":material/assignment:", key="view_report"):
            st.session_state.workflow_step = "report"
            st.rerun()
        if st.button("Back to shot entry", icon=":material/arrow_back:", key="analysis_back"):
            st.session_state.workflow_step = "data_entry"
            st.rerun()
