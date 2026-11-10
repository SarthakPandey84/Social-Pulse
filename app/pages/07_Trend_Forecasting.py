"""Page 7: Trend Forecasting Engine.

Implements rigorous, time-aware forecasting for publication cohort performance indicators
and emerging relatable struggle topics:
1. Forecasting Overview, Target Semantics & Coverage Status (Section A)
2. Historical Publication-Cohort Series Visualizer (Section B)
3. Forecast Trajectory & Empirical Uncertainty Range (Section C)
4. Rolling-Origin Model Evaluation & MASE Comparison Table (Section D)
5. Emerging Struggle Topics Tracking (Section E)
6. Methodology, Empirical Uncertainty Disclosures & Limitations (Section F)
"""

from pathlib import Path
import sys
from typing import List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.data_loader import (
    DatasetLoadError,
    SYNTHETIC_DEMO_DATASET_ID,
    get_trend_forecaster,
    list_available_datasets,
    load_posts_dataset,
    load_selected_daily_summary_dataset,
    load_selected_posts_dataset,
    load_trends_dataset,
)
from app.components.ui_widgets import (
    SOCIALPULSE_PALETTE,
    apply_plotly_theme,
    render_dataset_selector,
    render_footer,
    render_header,
    render_metric_card,
    render_provenance_banner,
    render_section_header,
)

from src.modeling.trend_forecaster import (
    ForecastModelType,
    TrendForecastReport,
    ValidationStatus,
)

# Page configuration
st.set_page_config(
    page_title="Trend Forecasting | SocialPulse",
    page_icon="📈",
    layout="wide",
)

# Branded Header
render_header(
    title="Module 7: Trend Forecasting Engine",
    subtitle="Publication-cohort performance projections, rolling-origin validation, and emerging struggle trajectory tracking",
)

# Compact Summary Banner + Complete Expandable Methodological Boundaries
st.warning(
    "⚠️ **Methodological Summary:** Series reflect **publication-cohort lifetime totals** (exploratory descriptive analysis with "
    "**~2–63 day exposure-age disparity**), **UNVERIFIED** source ingestion coverage, strict **platform isolation** "
    "(YouTube lacks post saves), **non-causal** baselines, and **uncalibrated** empirical residual intervals."
)

with st.expander("📋 View Full Methodological Audit Disclosures & Modeling Boundaries (7 Guardrails)", expanded=False):
    st.markdown(
        "- **Publication-Cohort Target Semantics:** One time-series observation at time $t$ represents the **lifetime cumulative performance "
        "of content assets published during period $t$ (Publication Cohort Performance)**. It does NOT represent calendar-day "
        "traffic across the entire account profile.\n"
        "- **Exposure-Age Accumulation Bias:** Older posts (published early Sept) had up to ~63 days of accumulation before collection on Nov 2–3, "
        "while late-October cohorts had only ~2–3 days. Multi-snapshot daily tracking is absent in the source data to compute age-standardized metrics (e.g. Day 7 views). "
        "Projections are strictly exploratory descriptive analyses of lifetime views at collection time.\n"
        "- **Unverified Source Coverage:** Continuous calendar date spans do not verify whether zero-post intervals reflect true editorial inactivity "
        "or unobserved/dropped API records. Source crawl completeness is classified as **UNVERIFIED**.\n"
        "- **Platform Isolation & YouTube Interaction Semantics:** Platforms are strictly modeled separately. YouTube views (30s playback standard) "
        "and Instagram video exposures are never pooled. Post saves are not tracked on YouTube; four-component total engagement is rejected on YouTube, "
        "and three-component `youtube_available_interactions` (Likes + Comments + Shares) is provided.\n"
        "- **Missing-Data Distinction:** Periods with no published content (`post_count = 0`) yield zero cohort volume, while "
        "engagement rate is undefined (NaN) and rejected from time-series models rather than converted to a fake 0.0% rate.\n"
        "- **Uncalibrated Empirical Intervals:** Prediction bounds reflect empirical out-of-sample error quantiles across rolling origins; "
        "nominal 95% coverage is illustrative and NOT mathematically calibrated or guaranteed. Horizons beyond the validation cutoff or with sparse residuals ($n < 3$) use square-root horizon expansion (evaluation-horizon expansion, step-1 expansion, or MAE fallback as reported per step).\n"
        "- **Non-Causal Status:** Status-quo projections assume observational continuity and do NOT predict causal returns from creative interventions."
    )


# Cached Analytical Computations (using shared forecaster resource)
def get_forecaster():
    return get_trend_forecaster(rolling_window=7)


@st.cache_data(show_spinner=False)
def _run_cohort_forecast_cached(
    posts_df: pd.DataFrame,
    target_metric: str,
    platform: str,
    cadence: str,
    horizon: int,
    selected_model_value: Optional[str],
    allow_tiktok: bool = False,
) -> TrendForecastReport:
    override = (
        next((m for m in ForecastModelType if m.value == selected_model_value), None)
        if selected_model_value is not None
        else None
    )
    return get_forecaster().forecast_metric(
        df=posts_df,
        target_metric=target_metric,
        platform=platform,
        cadence=cadence,
        horizon=horizon,
        selected_model=override,
        allow_tiktok=allow_tiktok,
    )


@st.cache_data(show_spinner=False)
def _run_daily_increment_forecast_cached(
    daily_df: pd.DataFrame,
    metric_col: str,
    platform: str,
    horizon: int,
    selected_model_value: Optional[str],
) -> TrendForecastReport:
    override = (
        next((m for m in ForecastModelType if m.value == selected_model_value), None)
        if selected_model_value is not None
        else None
    )
    return get_forecaster().forecast_daily_increment_series(
        daily_df=daily_df,
        metric_col=metric_col,
        platform=platform,
        horizon=horizon,
        selected_model=override,
        segment_policy="longest_contiguous",
    )


@st.cache_data(show_spinner=False)
def _run_keyword_forecast_cached(
    trends_df: pd.DataFrame,
    keyword: str,
    horizon: int,
    selected_model_value: Optional[str],
) -> TrendForecastReport:
    override = (
        next((m for m in ForecastModelType if m.value == selected_model_value), None)
        if selected_model_value is not None
        else None
    )
    return get_forecaster().forecast_keyword_trend(
        trend_df=trends_df,
        keyword=keyword,
        horizon=horizon,
        selected_model=override,
    )


@st.cache_data(show_spinner=False)
def _get_available_keywords_cached(trends_df: pd.DataFrame) -> List[str]:
    return sorted(trends_df["keyword"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _get_rising_struggles_cached(
    trends_df: pd.DataFrame,
    min_velocity: float = 5.0,
) -> pd.DataFrame:
    fc = get_forecaster()
    df_velocity = fc.compute_trend_velocity(trends_df)
    return fc.identify_rising_struggles(df_velocity, min_velocity=min_velocity)


@st.cache_data(show_spinner=False)
def _build_historical_chart_cached(
    hist_df: pd.DataFrame,
    target_metric: str,
    platform: str,
    cadence: str,
) -> go.Figure:
    metric_title = target_metric.replace("_", " ").title()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_df["timestamp"],
            y=hist_df["actual"],
            mode="lines+markers",
            name=metric_title,
            line=dict(color=SOCIALPULSE_PALETTE["primary_accent"], width=2.5),
            marker=dict(size=6, color=SOCIALPULSE_PALETTE["primary_accent"]),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.08)",
        )
    )
    fig.update_layout(
        title=f"Historical {metric_title} ({platform.title()} | {cadence} Cadence)",
        hovermode="x unified",
        xaxis_title="Publication Date",
        yaxis_title=metric_title,
        showlegend=False,
        margin=dict(l=56, r=24, t=58, b=66),
    )
    return apply_plotly_theme(fig, height=375)


@st.cache_data(show_spinner=False)
def _build_topics_chart_cached(trends_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        trends_df,
        x="date",
        y="mention_volume",
        color="keyword",
        title="Topic Volume (Observed Daily Continuity)",
        labels={"date": "Date", "mention_volume": "Daily Volume", "keyword": "Topic"},
        color_discrete_sequence=SOCIALPULSE_PALETTE["series"],
    )
    fig.update_traces(line=dict(width=2.2))
    fig.update_layout(
        title=dict(y=0.97, yanchor="top"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title_text="",
        ),
        margin=dict(l=54, r=20, t=98, b=64),
    )
    return apply_plotly_theme(fig, height=415)


# Unified Workspace Dataset Selector
all_datasets = list_available_datasets(root_dir=root_dir)
posts_datasets = list_available_datasets(root_dir=root_dir, required_table="posts")
selected_dataset_id = render_dataset_selector(
    posts_datasets,
    all_dataset_ids=[d["dataset_id"] for d in all_datasets],
    key="sp_dataset_selector_p7",
    location="sidebar",
)

# Load Analytical Datasets
daily_summary_df = None
active_meta = None
if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    posts_path = root_dir / "data" / "processed" / "processed_posts.parquet"
    trends_path = root_dir / "data" / "sample" / "sample_trends.csv"

    posts_df, _ = load_posts_dataset(posts_path, sample_path=None)
    trends_df = load_trends_dataset(trends_path)

    if posts_df is None and trends_df is None:
        st.error("❌ Required datasets not found. Please ensure data/processed/ or data/sample/ exist.")
        st.stop()
else:
    try:
        posts_df, active_meta = load_selected_posts_dataset(
            selected_dataset_id,
            root_dir=root_dir,
            columns=[
                "post_id",
                "platform",
                "timestamp",
                "views",
                "likes",
                "comments_count",
                "shares",
                "saves",
                "collect_count",
                "topic",
                "provenance",
            ],
        )
        daily_summary_df, _ = load_selected_daily_summary_dataset(selected_dataset_id, root_dir=root_dir)
    except DatasetLoadError as exc:
        st.error(f"❌ Dataset Load Error: {exc}")
        st.stop()
    trends_df = None
    render_provenance_banner(active_meta)

# Determine longitudinal readiness for the active corpus
has_daily_panel = daily_summary_df is not None and not daily_summary_df.empty
has_cohort_ts = (
    posts_df is not None
    and not posts_df.empty
    and "timestamp" in posts_df.columns
    and pd.to_datetime(posts_df["timestamp"], utc=True, errors="coerce").notna().any()
    and "views" in posts_df.columns
    and posts_df["views"].notna().any()
)

if selected_dataset_id != SYNTHETIC_DEMO_DATASET_ID and not has_daily_panel and not has_cohort_ts:
    st.info(
        f"ℹ️ **Longitudinal Forecasting Unavailable for `{selected_dataset_id}`:** "
        "This corpus is a cross-sectional snapshot (lacks sequential daily snapshots or valid publication-cohort "
        "timestamps with observed views). Select `Synthetic Demo Benchmark` or `TikTok Video Engagement 200K` "
        "to run rolling-origin walk-forward forecasts."
    )
    render_footer("SocialPulse Forecasting Engine · Rolling-Origin Walk-Forward Validation")
    st.stop()

# --- SIDEBAR: FORECASTING CONTROLS ---
st.sidebar.markdown("### 📈 Forecasting Controls")

if has_daily_panel:
    domain_options = ["Daily Panel Increments (Longest Contiguous Segment)", "Publication-Cohort Performance"]
elif selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    domain_options = ["Publication-Cohort Performance", "Struggle Keyword Trajectories"]
else:
    domain_options = ["Publication-Cohort Performance"]

domain_choice = st.sidebar.radio(
    "Select Forecasting Domain:",
    domain_options,
    index=0,
)

report: Optional[TrendForecastReport] = None

model_options = [
    "Automatic (Champion Selection)",
    ForecastModelType.NAIVE.value,
    ForecastModelType.MOVING_AVERAGE.value,
    ForecastModelType.EXPONENTIAL_SMOOTHING.value,
    ForecastModelType.AUTOREGRESSIVE.value,
    ForecastModelType.SEASONAL_NAIVE.value,
]

if domain_choice == "Daily Panel Increments (Longest Contiguous Segment)":
    daily_metric_labels = {
        "daily_views": "Daily Incremental Views (1-Day Non-Negative Snapshot Deltas)",
        "daily_likes": "Daily Incremental Likes",
        "daily_comments": "Daily Incremental Comments",
        "daily_shares": "Daily Incremental Shares",
        "daily_collects": "Daily Incremental Bookmark Collects (TikTok Proxy Metric)",
    }
    avail_daily_keys = [k for k in daily_metric_labels if k in daily_summary_df.columns]
    selected_daily_metric = st.sidebar.selectbox(
        "Target Daily Increment Metric:",
        options=avail_daily_keys,
        format_func=lambda k: daily_metric_labels[k],
        index=0,
    )
    daily_platform = (
        str(daily_summary_df["platform"].dropna().iloc[0]).strip().lower()
        if "platform" in daily_summary_df.columns and daily_summary_df["platform"].notna().any()
        else "tiktok"
    )
    cadence_code = "D"
    horizon_periods = st.sidebar.slider(
        "Forecast Horizon (Days Ahead):",
        min_value=1,
        max_value=14,
        value=7,
        step=1,
    )
    chosen_model_str = st.sidebar.selectbox("Forecasting Model:", model_options, index=0)
    selected_model_value = None if chosen_model_str == "Automatic (Champion Selection)" else chosen_model_str
    try:
        with st.spinner("Extracting contiguous segment and executing rolling-origin walk-forward validation..."):
            report = _run_daily_increment_forecast_cached(
                daily_df=daily_summary_df,
                metric_col=selected_daily_metric,
                platform=daily_platform,
                horizon=horizon_periods,
                selected_model_value=selected_model_value,
            )
    except ValueError as ve:
        st.error(f"⚠️ Validation Constraint: {ve}")
        st.stop()

elif domain_choice == "Publication-Cohort Performance":
    if posts_df is None or len(posts_df) == 0:
        st.error("Processed posts dataset not found.")
        st.stop()

    raw_plats = (
        sorted(posts_df["platform"].dropna().astype(str).str.strip().str.lower().unique().tolist())
        if "platform" in posts_df.columns
        else ["instagram"]
    )
    plat_display_map = {"instagram": "Instagram", "youtube": "YouTube", "tiktok": "TikTok"}
    platform_options = [plat_display_map.get(p, p.title()) for p in raw_plats]
    if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
        platform_options = ["Instagram", "YouTube"]
    selected_platform = st.sidebar.selectbox("Platform Scope (Isolated):", platform_options, index=0)

    if selected_platform == "Instagram":
        metric_labels = {
            "cohort_views": "Publication Cohort Views (Lifetime views of posts published in interval t)",
            "cohort_total_engagement": "Publication Cohort Total Interactions (Likes + Comments + Shares + Saves)",
            "cohort_likes": "Publication Cohort Likes",
            "cohort_comments": "Publication Cohort Comments",
            "cohort_shares": "Publication Cohort Shares",
            "cohort_saves": "Publication Cohort Saves",
            "views_engagement_rate": "Cohort Engagement Rate (% - undefined on zero-post intervals)",
        }
    elif selected_platform == "TikTok":
        metric_labels = {
            "cohort_views": "Publication Cohort Views (Lifetime views of posts published in interval t)",
            "available_interactions": "TikTok Available Interactions (Likes + Comments + Shares; Saves unavailable)",
            "cohort_likes": "Publication Cohort Likes",
            "cohort_comments": "Publication Cohort Comments",
            "cohort_shares": "Publication Cohort Shares",
            "cohort_collects": "TikTok Bookmark Collects (collect_count proxy)",
            "views_engagement_rate": "Cohort Engagement Rate (% - undefined on zero-post intervals)",
        }
    else:
        metric_labels = {
            "cohort_views": "Publication Cohort Views (Lifetime views of posts published in interval t)",
            "youtube_available_interactions": "YouTube Available Interactions (Likes + Comments + Shares; Saves unavailable)",
            "cohort_likes": "Publication Cohort Likes",
            "cohort_comments": "Publication Cohort Comments",
            "cohort_shares": "Publication Cohort Shares",
            "views_engagement_rate": "Cohort Engagement Rate (% - undefined on zero-post intervals)",
        }

    selected_metric_key = st.sidebar.selectbox(
        "Target Cohort Metric:",
        options=list(metric_labels.keys()),
        format_func=lambda k: metric_labels[k],
        index=0,
    )

    cadence_choice = st.sidebar.radio(
        "Aggregation Cadence:",
        ["Daily ('D')", "Weekly ('W')"],
        index=0,
    )
    cadence_code = "D" if "Daily" in cadence_choice else "W"

    max_h = 14 if cadence_code == "D" else 4
    horizon_periods = st.sidebar.slider(
        "Forecast Horizon (Steps Ahead):",
        min_value=1,
        max_value=max_h,
        value=7 if cadence_code == "D" else 3,
        step=1,
    )

    chosen_model_str = st.sidebar.selectbox("Forecasting Model:", model_options, index=0)
    selected_model_value = None if chosen_model_str == "Automatic (Champion Selection)" else chosen_model_str

    try:
        with st.spinner("Executing rolling-origin walk-forward validation and model estimation..."):
            report = _run_cohort_forecast_cached(
                posts_df=posts_df,
                target_metric=selected_metric_key,
                platform=selected_platform.lower(),
                cadence=cadence_code,
                horizon=horizon_periods,
                selected_model_value=selected_model_value,
                allow_tiktok=(selected_platform.lower() == "tiktok"),
            )
    except ValueError as ve:
        st.error(f"⚠️ Validation Constraint: {ve}")
        st.stop()

else:
    if trends_df is None or len(trends_df) == 0:
        st.error("Sample trends dataset not found.")
        st.stop()

    available_keywords = _get_available_keywords_cached(trends_df)
    selected_kw = st.sidebar.selectbox("Struggle Keyword:", available_keywords, index=0)

    cadence_code = "D"
    horizon_periods = st.sidebar.slider(
        "Forecast Horizon (Days Ahead):",
        min_value=1,
        max_value=14,
        value=7,
        step=1,
    )

    chosen_model_str = st.sidebar.selectbox("Forecasting Model:", model_options, index=0)
    selected_model_value = None if chosen_model_str == "Automatic (Champion Selection)" else chosen_model_str

    try:
        with st.spinner("Verifying coverage and evaluating rolling origins on keyword trajectory..."):
            report = _run_keyword_forecast_cached(
                trends_df=trends_df,
                keyword=selected_kw,
                horizon=horizon_periods,
                selected_model_value=selected_model_value,
            )
    except ValueError as ve:
        st.error(f"⚠️ Validation Constraint: {ve}")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("SocialPulse Forecasting Engine · Modules 1–7 Pipeline")


# --- SECTION A: FORECASTING OVERVIEW, TARGET SEMANTICS & COVERAGE STATUS ---
render_section_header(
    title="Section A: Forecasting Overview & Coverage Status",
    subtitle="Active series scope, observation activity, validation design, and selected champion model",
    badge=f"{report.platform.title()} · {'Daily' if report.cadence == 'D' else 'Weekly'}",
    badge_tone="accent",
)

if report.status not in [ValidationStatus.VALID]:
    st.error(f"❌ Forecasting halted: {report.uncertainty_description} (Status: {report.status.value})")
    if report.model_evaluations:
        st.subheader("Rolling-Origin Candidate Validation Summary")
        diag_rows = [
            {
                "Status": ev.evaluation_status,
                "Forecasting Model": ev.model_name,
                "Successful Origins": ev.successful_origins,
                "Failed Origins": ev.failed_origins,
                "MAE": f"{ev.mae:,.2f}" if ev.mae is not None else "Unavailable",
                "Notes": ev.notes,
            }
            for ev in report.model_evaluations
        ]
        st.dataframe(pd.DataFrame(diag_rows), width="stretch")
    st.stop()

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
step_label = "days" if report.cadence == "D" else "weeks"

with kpi_c1:
    render_metric_card(
        "Historical Window",
        f"{report.history_start} → {report.history_end}",
        delta=f"{report.platform.title()} · {'Daily' if report.cadence == 'D' else 'Weekly'}",
    )

with kpi_c2:
    obs_val = f"{report.history_points} {step_label}"
    obs_delta = f"{report.active_observation_days} active · {report.zero_publication_days} zero-post"
    render_metric_card(
        "History & Post Activity",
        obs_val,
        delta=obs_delta,
    )

with kpi_c3:
    render_metric_card(
        "Forecast Horizon",
        f"{report.forecast_horizon} {step_label} ahead",
        delta=f"{report.validation_origins_count} rolling origins",
    )

with kpi_c4:
    champ_label = report.champion_model
    render_metric_card(
        "Validated Champion",
        champ_label,
        delta=f"{report.validation_observations_count} OOS predictions",
    )

# Clean side-by-side Domain & Target Semantics Callouts
disc_col1, disc_col2 = st.columns(2)
with disc_col1:
    if domain_choice == "Daily Panel Increments (Longest Contiguous Segment)":
        st.info(
            f"📅 **Contiguous Panel Continuity Status:** **{report.coverage_status}**.\n\n"
            "• Missing crawler/snapshot outage dates are never silently zero-imputed or interpolated.\n"
            "• Negative inter-day cumulative deltas (user un-likes/deletions/resets) are quarantined prior to daily aggregation."
        )
    elif domain_choice == "Publication-Cohort Performance":
        st.warning(
            "⚠️ **EXPOSURE-AGE ACCUMULATION BIAS & UNVERIFIED COVERAGE DISCLOSURE:**\n\n"
            "- **Exposure Duration Disparity:** Posts published earlier in the collection window accumulated lifetime views over "
            "more days than recently published cohorts. "
            "Intermediate daily tracking snapshots are absent in single-snapshot exports, precluding standardized post-age metrics (e.g. Day 7 views). "
            "Projections represent exploratory descriptive analysis of lifetime views at collection time.\n"
            f"- **Source Ingestion Coverage Status:** **{report.coverage_status}**. "
            "Continuous calendar spans do not verify whether zero-post intervals reflect genuine editorial downtime or unobserved/dropped API records."
        )
    else:
        st.info(
            f"📅 **Keyword Observed Calendar-Date Continuity:** **{report.coverage_pct}%** ({report.coverage_status}).\n\n"
            "Unbroken presence across daily calendar dates reflects observation continuity within the sample dataset; "
            "source/API ingestion completeness remains unverified without crawler polling logs."
        )

with disc_col2:
    st.info(
        f"📌 **Forecasting Target Definition:** `{report.target_description}`\n\n"
        f"🔄 **Validation Design:** Evaluated via **rolling-origin walk-forward validation** across "
        f"**{report.validation_origins_count} distinct forecast origins** ({report.validation_observations_count} total out-of-sample evaluations). "
        "Models are refit at each origin using strictly preceding history to eliminate lookahead bias."
    )

if report.is_manual_override:
    st.warning(
        f"⚙️ **Manual Model Override Active:** Final forecast generated using **{report.selected_model}** "
        f"(User Selected). The walk-forward validation champion was **{report.champion_model}**."
    )


# --- SECTION B: HISTORICAL PUBLICATION-COHORT SERIES VISUALIZER ---
render_section_header(
    title="Section B: Historical Publication-Cohort Series Visualizer",
    subtitle="Observed cohort trajectory prior to walk-forward forecasting",
    badge=f"{report.history_points} {step_label}",
    badge_tone="teal",
)

hist_df = report.historical_df
if not hist_df.empty:
    fig_hist = _build_historical_chart_cached(
        hist_df=hist_df,
        target_metric=report.target_metric,
        platform=report.platform,
        cadence=report.cadence,
    )
    st.plotly_chart(fig_hist, width="stretch")


# --- SECTION C: FORECAST TRAJECTORY & EMPIRICAL UNCERTAINTY RANGE ---
render_section_header(
    title="Section C: Forecast Trajectory & Empirical Uncertainty Range",
    subtitle="Out-of-sample continuation paired with uncalibrated empirical residual bands",
    badge=report.selected_model,
    badge_tone="warning" if report.is_manual_override else "positive",
)

f_df = report.forecast_df
if not f_df.empty:
    fig_fc = go.Figure()

    # 1. Historical Actuals (tail for visual clarity)
    plot_hist = hist_df.tail(30)
    fig_fc.add_trace(
        go.Scatter(
            x=plot_hist["timestamp"],
            y=plot_hist["actual"],
            mode="lines+markers",
            name="Historical Actuals",
            line=dict(color=SOCIALPULSE_PALETTE["primary_bg"], width=2.5),
            marker=dict(size=5, color=SOCIALPULSE_PALETTE["primary_bg"]),
        )
    )

    # Bridge connection
    last_hist_x = plot_hist["timestamp"].iloc[-1]
    last_hist_y = plot_hist["actual"].iloc[-1]

    forecast_x = [last_hist_x] + f_df["timestamp"].tolist()
    forecast_y = [last_hist_y] + f_df["point_forecast"].tolist()
    ci_lower_y = [last_hist_y] + f_df["ci_lower"].tolist()
    ci_upper_y = [last_hist_y] + f_df["ci_upper"].tolist()

    # 2. Upper Uncertainty Bound
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_x,
            y=ci_upper_y,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 3. Lower Uncertainty Bound with Shaded Fan
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_x,
            y=ci_lower_y,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(2, 132, 199, 0.14)",
            name="Empirical Residual Range (Uncalibrated)",
        )
    )

    # 4. Point Forecast
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines+markers",
            name=f"Forecast ({report.selected_model})",
            line=dict(color=SOCIALPULSE_PALETTE["warning"], width=2.6, dash="dash"),
            marker=dict(size=7, symbol="diamond", color=SOCIALPULSE_PALETTE["warning"]),
        )
    )

    fig_fc.update_layout(
        title=dict(
            text=f"Publication-Cohort Forecast with Empirical Uncertainty Spread ({report.forecast_horizon}-Step Horizon)",
            y=0.96,
            yanchor="top",
        ),
        hovermode="x unified",
        xaxis_title="Timeline",
        yaxis_title=report.target_metric.replace("_", " ").title(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=56, r=24, t=88, b=62),
    )
    apply_plotly_theme(fig_fc, height=410)

    st.plotly_chart(fig_fc, width="stretch")

    # Uncertainty disclosure callout
    st.caption(f"📌 **Uncertainty Bounds Methodology:** {report.uncertainty_description}")

    # Forecast Detail Table
    st.subheader("Forecast Estimates & Empirical Residual Spread")
    st.info(
        "🔍 **Uncertainty Estimation Method Hierarchy:** Per forecast step $s$ (validation cutoff $H_{\\text{eval}} = \\min(H, 5)$): "
        "(1) **Direct Empirical Quantile** (90th percentile of $|e_s|$) when $s \\le H_{\\text{eval}}$ and $n_s \\ge 3$; "
        "(2) **Evaluation-Horizon Expansion** ($\\text{Spread}_{H_{\\text{eval}}} \\times \\sqrt{s / H_{\\text{eval}}}$) when $s > H_{\\text{eval}}$ and $n_{H_{\\text{eval}}} \\ge 3$; "
        "(3) **Step-1 Expansion Fallback** ($\\text{Spread}_1 \\times \\sqrt{s}$) when step-$s$ or cutoff residuals have $n < 3$ and $n_1 \\ge 3$; "
        "(4) **MAE Fallback** ($\\text{MAE} \\times 1.645 \\times \\sqrt{s}$) when $n_1 < 3$. "
        "Quantile estimates based on few observations ($n < 10$) are statistically unstable and exploratory."
    )
    cols_to_show = ["timestamp", "point_forecast", "ci_lower", "ci_upper", "empirical_error_spread"]
    col_names = [
        "Forecast Date",
        "Point Forecast",
        "Empirical Lower Bound",
        "Empirical Upper Bound",
        "Empirical Error Spread (±)",
    ]
    if "uncertainty_method" in f_df.columns:
        cols_to_show.append("uncertainty_method")
        col_names.append("Uncertainty Estimation Method")
    if "residual_count" in f_df.columns:
        cols_to_show.append("residual_count")
        col_names.append("Residual Observations (n)")

    display_f_df = f_df[cols_to_show].copy()
    display_f_df.insert(0, "Horizon Step", [f"Step +{i}" for i in range(1, len(display_f_df) + 1)])
    display_f_df.columns = ["Horizon Step"] + col_names

    st.dataframe(display_f_df, width="stretch")

    # Download Button
    csv_bytes = display_f_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast Table (CSV)",
        data=csv_bytes,
        file_name=f"cohort_forecast_{report.target_metric}_{report.cadence}_{report.history_end}.csv",
        mime="text/csv",
    )


# --- SECTION D: ROLLING-ORIGIN MODEL COMPARISON & VALIDATION ---
render_section_header(
    title="Section D: Rolling-Origin Model Comparison & Validation Metrics",
    subtitle="Walk-forward candidate benchmarks excluding partial-failure or under-evaluated candidates from Champion eligibility",
    badge=f"Champion: {report.champion_model}",
    badge_tone="positive",
)

st.markdown(
    "To eliminate lookahead bias, all candidate forecasters were evaluated via **rolling-origin walk-forward validation** "
    "across multiple expanding historical windows. Out-of-sample predictions were compared against the Naive persistence baseline. "
    "Any model with failed rolling origins (`Failed Origins > 0`) or insufficient successful evaluations (`< 3` origins) is excluded "
    "from Champion eligibility without manufacturing synthetic error penalties."
)

st.warning(
    "⚠️ **Model-Selection Optimism Bias Disclosure:** The champion model was designated by selecting the minimum out-of-sample MAE "
    "across eligible candidates evaluated across all rolling origins. Because the champion's reported evaluation metrics (MAE, RMSE, MASE) are derived from the "
    "exact same validation windows used for model selection, they carry model-selection optimism and do not represent an independent, "
    "unbiased estimate of future out-of-sample generalization error."
)

eval_rows = []
for ev in report.model_evaluations:
    status_label = "🏆 Champion" if ev.is_champion else ev.evaluation_status
    mae_str = f"{ev.mae:,.2f}" if ev.mae is not None else "Unavailable"
    rmse_str = f"{ev.rmse:,.2f}" if ev.rmse is not None else "Unavailable"

    if ev.mase is not None:
        mase_str = f"{ev.mase:.3f}"
    elif ev.mae is None:
        mase_str = "Unavailable"
    else:
        mase_str = "N/A (Zero In-Sample Scale)"

    if ev.relative_mae_to_naive is None:
        rel_mae_str = "Unavailable"
        pct_impr = "Unavailable"
    elif np.isinf(ev.relative_mae_to_naive):
        rel_mae_str = "∞ (Naive MAE=0)"
        pct_impr = "Worse than zero-error Naive"
    else:
        rel_mae_str = f"{ev.relative_mae_to_naive:.2f}x"
        pct_impr = (
            f"{(1.0 - ev.relative_mae_to_naive) * 100:.1f}% reduction"
            if ev.relative_mae_to_naive <= 1.0
            else f"+{(ev.relative_mae_to_naive - 1.0) * 100:.1f}% error"
        )

    eval_rows.append(
        {
            "Status": status_label,
            "Forecasting Model": ev.model_name,
            "Out-of-Sample MAE": mae_str,
            "Out-of-Sample RMSE": rmse_str,
            "MASE": mase_str,
            "Relative MAE vs. Naive": rel_mae_str,
            "Error vs Persistence": pct_impr,
            "Successful Origins": ev.successful_origins,
            "Failed Origins": ev.failed_origins,
            "Out-of-Sample Predictions": ev.validation_observations,
            "Notes": ev.notes,
        }
    )

eval_df = pd.DataFrame(eval_rows)
st.dataframe(eval_df, width="stretch")

# Methodology alert on why MASE is used
st.info(
    "💡 **Methodological Note: Why MASE instead of MAPE?**\n"
    "Mean Absolute Percentage Error (MAPE) divides by the observed value $|y_t|$. In social media publication series, "
    "periods frequently contain zero post uploads ($y_t = 0$), causing MAPE to divide by zero or diverge toward infinity. "
    "Mean Absolute Scaled Error (MASE) scales out-of-sample MAE against the in-sample mean absolute 1-step naive difference: "
    "$$\\text{MASE} = \\frac{\\text{MAE}_{\\text{test}}}{\\frac{1}{T-1}\\sum_{t=2}^T |y_t - y_{t-1}|}$$ "
    "A MASE $< 1.0$ indicates that the model outperforms the naive persistence baseline."
)


# --- SECTION E: EMERGING STRUGGLE TOPICS TRACKING ---
render_section_header(
    title="Section E: Emerging Struggle Keywords Tracking",
    subtitle="External discourse acceleration & topic calendar continuity monitoring without silent zero-imputation",
    badge="Keyword Intelligence",
    badge_tone="teal",
)

if trends_df is not None and not trends_df.empty:
    rising_struggles = _get_rising_struggles_cached(trends_df, min_velocity=5.0)

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.subheader("High-Velocity Relatable Topics")
        if not rising_struggles.empty:
            st.dataframe(rising_struggles, width="stretch")
        else:
            st.write("No topics currently exceed the growth velocity threshold.")

    with col_e2:
        st.subheader("Volume Trajectories Across Struggle Topics")
        fig_topics = _build_topics_chart_cached(trends_df)
        st.plotly_chart(fig_topics, width="stretch")
else:
    st.info("ℹ️ Struggle-keyword trajectory table (`sample_trends.csv`) is scoped to the Synthetic Demo Benchmark workspace.")


# --- SECTION F: METHODOLOGY, STATISTICAL LIMITATIONS & FORMULATIONS ---
render_section_header(
    title="Section F: Methodology & Mathematical Formulations",
    subtitle="Formal specifications for cohort targets, candidate models, rolling evaluation, and empirical spreads",
)

with st.expander("🔬 View Model Specifications, Validation Design & Calibration Disclosures", expanded=False):
    st.markdown(
        """
### 1. Publication-Cohort Target Semantics
Social media analytics platforms do not record daily time-series views accrued across an entire account profile in historical API exports.
Instead, API logs record the **lifetime cumulative performance accrued by specific published content assets**.
When resampled to daily or weekly intervals, one observation at time $t$ represents the **publication cohort performance**:
$$y_t = \\sum_{i \\in \\text{Posts}_t} \\text{Views}_i$$
Days where no content was published have $y_t = 0.0$ and $\\text{post\\_count} = 0$. For rate metrics (e.g. engagement rate per view),
periods with zero views are mathematically **Undefined (NaN)** and are never imputed as $0.0\\%$.

### 2. Candidate Time-Series Models
1. **Naive Persistence Baseline**: $\\hat{y}_{T+h} = y_T$.
2. **Seasonal Naive**: $\\hat{y}_{T+h} = y_{T+h - m(k+1)}$ with period $m=7$ (daily) or $m=4$ (weekly). Requires $N \\ge 2m$; falls back to persistence if history is insufficient.
3. **Moving Average**: $\\hat{y}_{T+h} = \\frac{1}{k}\\sum_{i=0}^{k-1} y_{T-i}$ with dynamically clipped window $k \\le N$.
4. **Holt's Exponential Smoothing**: Level $l_t$, Trend $b_t$, with multi-step damping factor $\\phi = 0.95$ and bounded L-BFGS-B SSE parameter optimization.
5. **AutoRegressive Ridge (AR)**: Lagged autoregressive features estimated via regularized $L_2$ Ridge regression.

### 3. Rolling-Origin Walk-Forward Validation & Model-Selection Optimism
Unlike a single train/test split, rolling-origin validation iterates across $K$ chronological cutoffs $t \\in \\{N-K, \\dots, N-1\\}$:
- At each cutoff $t$, models fit strictly on $y_{0:t}$.
- Predictions are generated out-of-sample for $h_t$ steps ahead ($y_{t:t+h_t}$).
- Prediction residuals $e_{t, j} = y_{t+j-1} - \\hat{y}_{t, j}$ are collected strictly from successful out-of-sample origin predictions (never fabricating artificial penalty residuals when a model fails).
- **Champion Eligibility Rule:** A candidate model must succeed at all attempted rolling origins (`failed_origins == 0` and `successful_origins >= 3`) to compete for Champion. Models with fewer than 3 successful origins are marked `Insufficiently Evaluated` or `Unavailable` with omitted error metrics; models with partial failures (`>= 3` successes, `failed_origins > 0`) report metrics strictly from valid origins and are marked `Excluded (Partial Failure)`.
- Among eligible candidates, the model minimizing total out-of-sample MAE is designated Champion.
- **Model-Selection Optimism Caveat:** Because the champion model is selected by evaluating candidate errors on these exact same origins, the champion's reported MAE/RMSE is selection-dependent and does not represent an independent, unbiased estimate of future generalization error.

### 4. Empirical Uncertainty Spread & Fallback Hierarchy
Prediction bounds reflect empirical out-of-sample residual spreads across four mutually distinct paths per horizon step $s$ (validation cutoff $H_{\\text{eval}} = \\min(H, 5)$):
1. **Direct Empirical Quantile** (when $s \\le H_{\\text{eval}}$ and $n_s \\ge 3$ residuals):
   $$\\text{Spread}_s = \\text{Percentile}_{90}\\left(\\{|e_{t, s}|\\}\\right)$$
2. **Evaluation-Horizon Expansion** (when $s > H_{\\text{eval}}$ and $n_{H_{\\text{eval}}} \\ge 3$):
   $$\\text{Spread}_s = \\text{Spread}_{H_{\\text{eval}}} \\times \\sqrt{\\frac{s}{H_{\\text{eval}}}}$$
3. **Step-1 Expansion Fallback** (when step-$s$ or cutoff residuals have $n < 3$, and step-1 has $n_1 \\ge 3$):
   $$\\text{Spread}_s = \\text{Spread}_1 \\times \\sqrt{s}$$
4. **MAE Fallback** (when even step-1 has $n_1 < 3$):
   $$\\text{Spread}_s = \\text{MAE} \\times 1.645 \\times \\sqrt{s}$$
- The interval enforces a non-negative floor:
  $$\\text{CI}_s = \\left[ \\max\\left(0.0, \\hat{y}_{T+s} - \\text{Spread}_s\\right), \\quad \\hat{y}_{T+s} + \\text{Spread}_s \\right]$$
**Calibration Caveat:** Because sample sizes across historical origins are finite ($K \\approx 10$) and time-series data may exhibit residual autocorrelation and non-stationarity, nominal 95% coverage is **NOT calibrated or guaranteed**. Quantile estimates based on few observations ($n < 10$) are statistically unstable and exploratory.

### 5. Platform Isolation
Instagram and YouTube data are modeled independently. Cross-platform pooling of raw views is prohibited due to differing platform playback definitions.
        """
    )

render_footer("SocialPulse Forecasting Engine · Publication-Cohort Walk-Forward Validation")
