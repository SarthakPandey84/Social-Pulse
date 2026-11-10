"""Page 4: A/B Testing Framework.

Provides rigorous statistical inference for digital creative experiments:
1. Hypothesis testing across conversion rates (proportions) and engagement metrics (continuous).
2. Cluster-defensible post-level comparisons vs aggregate exposure comparisons.
3. Multiple testing corrections (Holm-Bonferroni FWER & Benjamini-Hochberg FDR).
4. Statistical effect sizes (Cohen's h, Hedges' g) and Agresti-Caffo / Welch-Satterthwaite confidence intervals.
5. Pre-experiment sample size planning and approximate achieved statistical power calculations.
"""

from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.components.ui_widgets import render_header, render_metric_card
from src.statistics.ab_testing import ABTestingFramework, ComparisonResult
from src.data_ingestion.schemas import ExperimentStatus, ExperimentType, StatisticalUnit

render_header(
    title="Module 4: A/B Testing Framework",
    subtitle="Rigorous parametric and non-parametric hypothesis testing for creative optimization"
)

# Provenance & Pipeline Demonstration Notice
st.info(
    "ℹ️ **Dataset Provenance & Demonstration Notice:** This module operates strictly on **synthetic benchmark experiment fixtures** "
    "designed to validate statistical power analysis, Welch's t-tests, two-proportion z-tests, Fisher's exact tests, "
    "cluster-collapsed post-level comparisons, and multiple testing corrections. These fixtures are for pipeline validation only "
    "and do NOT represent live production campaign results. Observational comparisons are explicitly labeled to avoid causal over-claiming."
)

# 1. Load Data
exp_path = root_dir / "data" / "sample" / "sample_experiments.csv"
obs_path = root_dir / "data" / "processed" / "processed_ab_tests.parquet"

if exp_path.exists() and obs_path.exists():
    df_experiments = pd.read_csv(exp_path)
    df_observations = pd.read_parquet(obs_path)
else:
    st.warning("Experiment datasets not found. Please run the fixture generation script.")
    st.stop()

framework = ABTestingFramework(alpha=0.05)

# 2. Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Active Experiments & Testing",
    "📊 Statistical Visualizations",
    "📈 Multiple Testing & Corrections",
    "🧮 Sample Size & Power Planning"
])

# --- TAB 1: ACTIVE EXPERIMENTS & TESTING ---
with tab1:
    st.subheader("1. Select or Configure Experiment (Benchmark Fixture)")

    exp_options = {
        row["experiment_id"]: f"{row['experiment_id']}: {row['experiment_name']}"
        for _, row in df_experiments.iterrows()
    }
    selected_exp_id = st.selectbox("Select Benchmark Experiment:", list(exp_options.keys()), format_func=lambda x: exp_options[x])

    exp_meta = df_experiments[df_experiments["experiment_id"] == selected_exp_id].iloc[0]
    obs_subset = df_observations[df_observations["experiment_id"] == selected_exp_id].copy()

    # Experiment Info Banner
    is_randomized = exp_meta["experiment_type"] == ExperimentType.RANDOMIZED_EXPERIMENT.value
    exp_type_badge = "🟢 Randomized Controlled Trial (A/B Test)" if is_randomized else "🟡 Observational Cohort Comparison (Non-Randomized)"

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
            <div style="font-size: 13px; font-weight: 700; color: #475569;">
                EXPERIMENT REGISTRATION: <span style="background-color: #E2E8F0; padding: 2px 8px; border-radius: 4px;">{selected_exp_id}</span> | {exp_type_badge}
            </div>
            <div style="font-size: 14px; margin-top: 6px;"><b>Hypothesis:</b> <i>"{exp_meta['hypothesis']}"</i></div>
            <div style="font-size: 13px; color: #64748B; margin-top: 4px;">
                Platform: <b>{exp_meta['platform'].title()}</b> | 
                Control (A): <b>{exp_meta['variant_a_label']}</b> | 
                Treatment (B): <b>{exp_meta['variant_b_label']}</b> | 
                Observation Horizon: <b>7-day settled post observations (deduplicated)</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not is_randomized:
        st.warning(
            "⚠️ **Observational Comparison Notice (Non-Randomized):** Treatments were observed historically rather than assigned via controlled audience randomization. "
            "Confounders (posting time, audience demographics, algorithmic favoring) are NOT controlled. "
            "Statistical differences indicate correlation only and must NOT be interpreted as causal proof."
        )

    # Metric Selection & Alpha Configuration
    cfg_col1, cfg_col2 = st.columns([2, 1])
    with cfg_col1:
        available_metrics = ["share_rate", "retention_rate", "views", "saves", "shares", "relatability_pct"]
        default_idx = available_metrics.index(exp_meta["primary_metric"]) if exp_meta["primary_metric"] in available_metrics else 0
        selected_metric = st.selectbox("Select Test Metric:", available_metrics, index=default_idx)
    with cfg_col2:
        alpha_val = st.slider("Significance Level (α):", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

    var_a = exp_meta["variant_a_label"]
    var_b = exp_meta["variant_b_label"]
    is_prop = selected_metric in ["share_rate", "save_rate", "comment_rate"]

    # Cluster-Defense Mode Selection for Proportion Metrics
    if is_prop:
        mode_choice = st.radio(
            "Statistical Unit / Clustering Mode:",
            [
                "Cluster-Robust Post-Level Analysis (Recommended: Compares independent post rates via Welch's t-test)",
                "Aggregate Exposure Analysis (Unclustered: Treats reach as independent Bernoulli trials; ignores intra-post clustering)"
            ]
        )
        use_cluster_collapsed = "Cluster-Robust" in mode_choice
    else:
        use_cluster_collapsed = True

    # Execute Test
    if is_prop and not use_cluster_collapsed:
        # Aggregate exposure path
        sub_a = obs_subset[obs_subset["variant"] == var_a]
        sub_b = obs_subset[obs_subset["variant"] == var_b]

        base_metric = selected_metric.replace("_rate", "s") if selected_metric != "comment_rate" else "comments_count"
        cnt_a = int(sub_a[base_metric].sum()) if base_metric in sub_a else int(sub_a[selected_metric].sum() * 100)
        cnt_b = int(sub_b[base_metric].sum()) if base_metric in sub_b else int(sub_b[selected_metric].sum() * 100)
        n_a = int(sub_a["reach"].sum()) if "reach" in sub_a else len(sub_a) * 1000
        n_b = int(sub_b["reach"].sum()) if "reach" in sub_b else len(sub_b) * 1000

        test_res = framework.compare_proportions(
            count_a=cnt_a, nobs_a=n_a, count_b=cnt_b, nobs_b=n_b,
            metric_name=selected_metric, variant_a_label=var_a, variant_b_label=var_b,
            alpha=alpha_val, experiment_type=exp_meta["experiment_type"]
        )
    elif is_prop and use_cluster_collapsed:
        # Cluster-collapsed post-level rate comparison
        test_res = framework.compare_post_level_rates(
            df=obs_subset,
            variant_col="variant",
            variant_a=var_a,
            variant_b=var_b,
            rate_col=selected_metric,
            metric_name=selected_metric,
            alpha=alpha_val,
            experiment_type=exp_meta["experiment_type"]
        )
    else:
        sample_a = obs_subset[obs_subset["variant"] == var_a][selected_metric].dropna().values
        sample_b = obs_subset[obs_subset["variant"] == var_b][selected_metric].dropna().values
        test_res = framework.compare_continuous(
            sample_a=sample_a, sample_b=sample_b,
            metric_name=selected_metric, variant_a_label=var_a, variant_b_label=var_b,
            alpha=alpha_val, experiment_type=exp_meta["experiment_type"]
        )

    # Render Outcome KPI Cards
    st.markdown("### Hypothesis Test Outcome")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_metric_card(f"Control ({var_a[:15]})", f"{test_res.value_a:.4f}" if is_prop else f"{test_res.value_a:,.2f}")
    with kpi_col2:
        render_metric_card(f"Treatment ({var_b[:15]})", f"{test_res.value_b:.4f}" if is_prop else f"{test_res.value_b:,.2f}")
    with kpi_col3:
        lift_sign = "+" if test_res.relative_lift_pct > 0 else ""
        render_metric_card("Relative Lift", f"{lift_sign}{test_res.relative_lift_pct:.2f}%")
    with kpi_col4:
        p_val_display = f"{test_res.p_value:.5f}" if test_res.p_value >= 0.0001 else "< 0.0001"
        render_metric_card(f"p-value ({test_res.primary_method})", p_val_display)

    # Comprehensive Verdict Card
    badge_color = "#10B981" if test_res.is_statistically_significant else "#64748B"
    cluster_badge = "✅ Handled (Cluster-Collapsed)" if test_res.clustering_handled else "⚠️ Ignored (Unclustered Exposure Assumption)"

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border-left: 5px solid {badge_color}; padding: 16px; border-radius: 8px; margin-top: 12px;">
            <div style="font-size: 13px; color: #64748B; font-weight: 600;">STATISTICAL VERDICT (α = {alpha_val})</div>
            <div style="font-size: 20px; font-weight: 700; color: {badge_color}; margin-top: 4px;">{test_res.verdict}</div>
            <div style="font-size: 13px; color: #475569; margin-top: 8px;">
                <b>Primary Method:</b> {test_res.primary_method} | 
                <b>Test Statistic:</b> {test_res.test_statistic:.4f} | 
                <b>Effect Size:</b> {test_res.effect_size_type} = {test_res.effect_size:+.3f} | 
                <b>Confidence Interval:</b> [{test_res.confidence_interval[0]:.4f}, {test_res.confidence_interval[1]:.4f}] ({test_res.ci_method})<br>
                <b>Unit of Analysis:</b> {test_res.unit_of_analysis} | 
                <b>Clustering Handled:</b> {cluster_badge} | 
                <b>Data Provenance:</b> Synthetic Benchmark Fixture
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Additional Non-Parametric & Diagnostic Cards
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        if test_res.mann_whitney_p_value is not None:
            mw_sig = "Significant" if test_res.mann_whitney_p_value < alpha_val else "Not Significant"
            st.markdown(
                f"""
                <div style="background-color: #F1F5F9; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 13px;">
                    <b>Non-Parametric Diagnostic (Mann-Whitney U):</b> p = {test_res.mann_whitney_p_value:.5f} ({mw_sig})<br>
                    <span style="color: #64748B;">Evaluates rank distribution stochastic dominance P(X > Y) + 0.5P(X = Y) = 0.5; does not assume equal shapes or medians.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    with diag_col2:
        if test_res.diagnostic_p_value is not None and test_res.primary_method == "Fisher's Exact Test":
            st.markdown(
                f"""
                <div style="background-color: #F1F5F9; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 13px;">
                    <b>Secondary Diagnostic (Asymptotic z-test):</b> z = {test_res.diagnostic_test_statistic:.3f}, p = {test_res.diagnostic_p_value:.5f}<br>
                    <span style="color: #64748B;">Retained for diagnostic reference; verdict was governed strictly by Fisher's Exact Test due to sparse cell count.</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    if test_res.warning_messages:
        for w in test_res.warning_messages:
            st.info(f"ℹ️ {w}")


# --- TAB 2: STATISTICAL VISUALIZATIONS ---
with tab2:
    st.subheader("2. Metric Distributions & Confidence Intervals (Benchmark Fixtures)")
    st.caption(
        f"Unit of Analysis: {test_res.unit_of_analysis}. "
        "Each observation represents a settled post observation (deduplicated; no repeated snapshot pseudo-replication)."
    )

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown(f"#### Distribution Comparison: {selected_metric}")
        fig_box = px.box(
            obs_subset[obs_subset["variant"].isin([var_a, var_b])],
            x="variant",
            y=selected_metric,
            color="variant",
            points="all",
            title=f"Sample Post Observations per Variant (N = {len(obs_subset)} posts)"
        )
        fig_box.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with v_col2:
        st.markdown(f"#### Estimated Difference & {int((1-alpha_val)*100)}% CI ({test_res.ci_method})")
        diff_val = test_res.absolute_difference
        ci_low, ci_high = test_res.confidence_interval

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=[diff_val],
            y=[selected_metric],
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci_high - diff_val],
                arrayminus=[diff_val - ci_low],
                thickness=3,
                width=10,
                color="#3B82F6"
            ),
            mode='markers',
            marker=dict(size=12, color="#1E3A8A"),
            name="Point Estimate"
        ))
        fig_ci.add_vline(x=0.0, line_dash="dash", line_color="red", annotation_text="Null (H0: Δ = 0)")
        fig_ci.update_layout(
            title=f"Absolute Difference (B - A): {diff_val:+.4f}",
            xaxis_title="Difference (Treatment - Control)",
            yaxis_title="",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_ci, use_container_width=True)


# --- TAB 3: MULTIPLE TESTING & CORRECTIONS ---
with tab3:
    st.subheader("3. Multiple Hypotheses & Family-Wise Error Rate Control (Benchmark Fixtures)")
    st.markdown(
        "When an experimentation team tests multiple metrics simultaneously (e.g. evaluating views, retention, shares, "
        "and saves in the same experiment), the probability of at least one false positive (Type I error) inflates rapidly: "
        r"$\text{FWER} = 1 - (1 - \alpha)^m$. "
        "We apply **Holm-Bonferroni (step-down FWER control)** and **Benjamini-Hochberg (FDR control)** across post-level outcomes."
    )

    secondary_metrics_list = ["share_rate", "retention_rate", "views", "saves", "shares", "relatability_pct"]
    raw_results = []

    for m in secondary_metrics_list:
        if m in obs_subset.columns:
            s_a = obs_subset[obs_subset["variant"] == var_a][m].dropna().values
            s_b = obs_subset[obs_subset["variant"] == var_b][m].dropna().values
            if len(s_a) > 2 and len(s_b) > 2:
                res_m = framework.compare_continuous(s_a, s_b, metric_name=m, variant_a_label=var_a, variant_b_label=var_b, alpha=alpha_val)
                raw_results.append(res_m)

    if raw_results:
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            correction_method = st.radio("Correction Method:", ["Holm-Bonferroni (FWER Control)", "Benjamini-Hochberg (FDR Control)"])
            method_key = "holm" if "Holm" in correction_method else "bh"

        corrected_results = ABTestingFramework.apply_multiple_testing_correction(raw_results, method=method_key, alpha=alpha_val)

        table_rows = []
        for cr in corrected_results:
            table_rows.append({
                "Metric": cr.metric_name,
                "Control Mean": f"{cr.value_a:.4f}",
                "Treatment Mean": f"{cr.value_b:.4f}",
                "Relative Lift": f"{cr.relative_lift_pct:+.2f}%",
                "Effect Size (g)": f"{cr.effect_size:+.3f}",
                "Raw p-value": f"{cr.p_value:.5f}",
                "Adjusted p-value": f"{cr.p_value_adjusted:.5f}",
                "Significant (Adj)": "✅ Yes" if cr.is_statistically_significant else "❌ No"
            })
        st.table(pd.DataFrame(table_rows))
        st.caption(f"Adjusted significance evaluated at family-wise α = {alpha_val}. Tested across m = {len(raw_results)} simultaneous metrics.")


# --- TAB 4: SAMPLE SIZE & POWER PLANNING ---
with tab4:
    st.subheader("4. Pre-Experiment Sample Size & Power Planning")
    st.markdown(
        "Calculate the minimum sample size required per variant before launching an experiment "
        "to ensure adequate statistical power ($1 - \beta = 80\%$) and avoid inconclusive under-powered tests. "
        "All calculations use standard asymptotic normal approximations."
    )

    calc_type = st.radio("Metric Type to Plan:", ["Conversion / Proportion Metric (e.g. Share Rate)", "Continuous Metric (e.g. Retention Rate, Watch Time)"])

    if "Proportion" in calc_type:
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            base_p = st.number_input("Baseline Rate (%):", min_value=0.1, max_value=50.0, value=3.0, step=0.5) / 100.0
        with p_c2:
            mde_pct = st.number_input("Minimum Detectable Lift (%):", min_value=1.0, max_value=100.0, value=20.0, step=1.0) / 100.0
        with p_c3:
            calc_alpha = st.selectbox("Alpha (α):", [0.01, 0.05, 0.10], index=1)
        with p_c4:
            calc_power = st.selectbox("Power (1 - β):", [0.80, 0.90, 0.95], index=0)

        plan = ABTestingFramework.calculate_sample_size_proportions(base_p, mde_pct, alpha=calc_alpha, power=calc_power)

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            render_metric_card("Sample Size per Variant", f"{plan['sample_size_per_variant']:,}")
        with res_c2:
            render_metric_card("Total Sample Size Needed", f"{plan['total_sample_size_required']:,}")

        st.caption(f"**Assumptions Disclosed:** {plan['assumptions_disclosed']}")

    else:
        c_c1, c_c2, c_c3, c_c4 = st.columns(4)
        with c_c1:
            base_mean = st.number_input("Baseline Mean:", min_value=0.1, max_value=1000.0, value=100.0, step=5.0)
        with c_c2:
            base_std = st.number_input("Baseline Std Dev:", min_value=0.1, max_value=500.0, value=25.0, step=2.0)
        with c_c3:
            mde_c_pct = st.number_input("Desired Relative MDE (%):", min_value=1.0, max_value=100.0, value=10.0, step=1.0) / 100.0
        with c_c4:
            power_c = st.selectbox("Desired Power:", [0.80, 0.90, 0.95], index=0)

        plan_c = ABTestingFramework.calculate_sample_size_continuous(base_mean, base_std, mde_c_pct, alpha=0.05, power=power_c)

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            render_metric_card("Sample Size per Variant", f"{plan_c['sample_size_per_variant']:,}")
        with res_c2:
            render_metric_card("Total Sample Size Needed", f"{plan_c['total_sample_size_required']:,}")

        st.caption(f"**Assumptions Disclosed:** {plan_c['assumptions_disclosed']}")

    # Interactive Approximate Achieved Power Calculator Section
    st.markdown("---")
    st.markdown("### Approximate Achieved Power Calculator (Post-Hoc Diagnostic)")
    st.markdown(
        "Evaluate whether an observed experiment had sufficient statistical power to detect an effect of a given magnitude. "
        "*(Note: Power is computed using standard normal approximation to the non-central distribution and is explicitly labeled **approximate**).*"
    )
    pow_c1, pow_c2, pow_c3, pow_c4 = st.columns(4)
    with pow_c1:
        n_a_input = st.number_input("Sample Size Variant A:", min_value=5, max_value=100000, value=60, step=5)
    with pow_c2:
        n_b_input = st.number_input("Sample Size Variant B:", min_value=5, max_value=100000, value=60, step=5)
    with pow_c3:
        eff_input = st.number_input("Standardized Effect Size (d / h):", min_value=0.01, max_value=3.0, value=0.40, step=0.05)
    with pow_c4:
        alpha_input = st.selectbox("Alpha Level (α):", [0.01, 0.05, 0.10], index=1, key="pow_alpha")

    achieved_power_val = ABTestingFramework.calculate_achieved_power(n_a_input, n_b_input, eff_input, alpha=alpha_input)
    pow_kpi1, pow_kpi2 = st.columns(2)
    with pow_kpi1:
        render_metric_card("Approximate Achieved Power", f"{achieved_power_val * 100.0:.1f}%")
    with pow_kpi2:
        pow_verdict = "Adequately Powered (≥ 80%)" if achieved_power_val >= 0.80 else "Under-Powered (< 80%)"
        render_metric_card("Power Status", pow_verdict)

    st.caption("Approximate achieved power calculated using normal approximation of non-central distribution with harmonic mean sample size.")
