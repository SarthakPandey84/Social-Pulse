"""Page 6: Growth Visualization Dashboard.

Integrates Modules 1-5 into a unified, interactive growth intelligence dashboard:
1. Growth Overview KPIs with strict denominator integrity and period comparisons.
2. Interactive time-series performance trends.
3. Normalized cross-platform comparisons.
4. Multidimensional content performance breakdowns with sample size disclosure.
5. Audience sentiment and relatability insights (Module 3).
6. Controlled experiment evaluation and effect-size forest plot (Module 4).
7. Prioritized strategic optimization recommendations (Module 5).
"""

from datetime import datetime
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from app.components.ui_widgets import render_header, render_metric_card
except (ImportError, ModuleNotFoundError):
    from components.ui_widgets import render_header, render_metric_card
from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.recommender.schemas import ConfidenceLevel, RecommendationCategory, EvidenceSource

# Branded Header
render_header(
    title="Module 6: Growth Visualization Dashboard",
    subtitle="Unified cross-module descriptive analytics, experiment tracking, and prescriptive strategy"
)

# Prominent Synthetic Benchmark Provenance Banner
st.warning(
    "⚠️ **SYNTHETIC DEMONSTRATION DATA:** These visualizations demonstrate analytical functionality across "
    "Modules 1–5 using synthetic benchmark fixtures and do not represent verified real-world creator performance. "
    "Observational correlations are strictly separated from experimental evidence and are never presented as proven causal relationships."
)

# Initialize Backend Engine
@st.cache_resource
def get_backend():
    return GrowthDashboardBackend()

backend = get_backend()

if backend.posts_df is None or len(backend.posts_df) == 0:
    st.error("❌ No analytical datasets found. Please verify `data/processed/` or run ingestion pipeline.")
    st.stop()

# --- SIDEBAR: GLOBAL INTERACTIVE FILTERS ---
st.sidebar.title("🔍 Global Dashboard Filters")

# Date range bounds from data
ts_col = "timestamp_dt" if "timestamp_dt" in backend.posts_df.columns else "timestamp"
all_dates = pd.to_datetime(backend.posts_df[ts_col], utc=True, errors="coerce").dropna()
min_data_date = all_dates.min().date() if not all_dates.empty else datetime(2026, 9, 1).date()
max_data_date = all_dates.max().date() if not all_dates.empty else datetime(2026, 10, 31).date()

# Date range selector
date_range = st.sidebar.date_input(
    "📅 Date Range:",
    value=(min_data_date, max_data_date),
    min_value=min_data_date,
    max_value=max_data_date
)

start_date = date_range[0] if isinstance(date_range, (list, tuple)) and len(date_range) > 0 else min_data_date
end_date = date_range[1] if isinstance(date_range, (list, tuple)) and len(date_range) > 1 else max_data_date

# Platform filter
platform_options = ["All Platforms", "Instagram", "YouTube"]
selected_plat_raw = st.sidebar.selectbox("📱 Platform:", platform_options, index=0)
selected_platform = None if selected_plat_raw == "All Platforms" else selected_plat_raw.lower()

# Content format filter
format_options = ["All Formats"] + sorted(list(backend.posts_df["content_type"].dropna().unique())) if "content_type" in backend.posts_df.columns else ["All Formats"]
selected_fmt_raw = st.sidebar.selectbox("🎬 Content Format:", format_options, index=0)
selected_format = None if selected_fmt_raw == "All Formats" else selected_fmt_raw

# Topic filter
topic_options = ["All Topics"] + sorted(list(backend.posts_df["topic"].dropna().unique())) if "topic" in backend.posts_df.columns else ["All Topics"]
selected_topic_raw = st.sidebar.selectbox("💡 Struggle Topic:", topic_options, index=0)
selected_topic = None if selected_topic_raw == "All Topics" else selected_topic_raw

# Reset filters button
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data Sources: Modules 1–5 Pipelines | Parquet / CSV Fixtures")

# Apply Global Filters
filtered_posts = backend.filter_posts(
    start_date=start_date,
    end_date=end_date,
    platform=selected_platform,
    content_type=selected_format,
    topic=selected_topic
)

# Empty State Guard
if len(filtered_posts) == 0:
    st.warning("⚠️ No posts match the selected filter combination. Please broaden your date range or filter criteria.")
    st.stop()

# --- SECTION A: GROWTH OVERVIEW ---
st.markdown("## 📊 Section A: High-Level Growth Overview")

kpis = backend.calculate_growth_overview_kpis(filtered_posts)

row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
with row1_col1:
    render_metric_card("Analyzed Posts", f"{kpis['total_posts']:,}")
with row1_col2:
    render_metric_card("Total Views", f"{kpis['total_views']:,}")
with row1_col3:
    render_metric_card("Total Impressions", f"{kpis['total_impressions']:,}")
with row1_col4:
    render_metric_card("New Followers", f"{kpis['total_new_followers']:,}")

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
with row2_col1:
    render_metric_card("Total Likes", f"{kpis['total_likes']:,}")
with row2_col2:
    render_metric_card("Total Comments", f"{kpis['total_comments']:,}")
with row2_col3:
    render_metric_card("Total Shares", f"{kpis['total_shares']:,}")
with row2_col4:
    render_metric_card("Total Saves (IG)", f"{kpis['total_saves']:,}")

row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
with row3_col1:
    render_metric_card("Pooled Views Eng. Rate", f"{kpis['pooled_views_engagement_rate']:.2f}%")
with row3_col2:
    render_metric_card("Mean Post-Level Views Rate", f"{kpis['mean_post_level_views_engagement_rate']:.2f}%")
with row3_col3:
    render_metric_card("Reach Eng. Rate (IG Pooled)", f"{kpis['pooled_reach_engagement_rate']:.2f}%" if kpis['pooled_reach_engagement_rate'] > 0 else "N/A (YouTube)")
with row3_col4:
    render_metric_card("Avg Views / Post", f"{kpis['avg_views_per_post']:,.0f}")

# Period & Comparison Disclosure
with st.container():
    p_col1, p_col2 = st.columns([2, 3])
    with p_col1:
        st.info(f"📅 **Reporting Period:** `{kpis['reporting_period']}` ({kpis['period_days']} calendar days, {kpis['active_posting_days']} active posting days)")
    with p_col2:
        if kpis["has_previous_period_comparison"]:
            deltas = kpis["comparison_deltas"]
            v_d = deltas.get("views_delta_pct")
            e_d = deltas.get("eng_delta_pct")
            v_str = f"{v_d:+.1f}%" if v_d is not None else "N/A"
            e_str = f"{e_d:+.1f}%" if e_d is not None else "N/A"
            st.success(f"📈 **Period-over-Period Delta:** Views: **{v_str}** | Total Engagement: **{e_str}** ({kpis['comparison_note']})")
        else:
            st.caption(f"ℹ️ {kpis['comparison_note']}")

# Metric Definition Disclosure
st.caption(
    "📐 **Metric Definitions & Denominators:** "
    "• **Pooled Views Rate (Overview KPI):** `Total Interactions / Total Views × 100%` (Volume-weighted across all posts in window). "
    "• **Mean Post-Level Views Rate:** `(1/N) × Σ (Post Interactions / Post Views × 100%)` (Unweighted arithmetic mean treating each post equally). "
    "• **Reach-Based Rate (Instagram only):** Evaluated against Reach (unique accounts); YouTube Data API does not expose post-level reach. "
    "• **Save Metrics:** Tracked on Instagram; YouTube API lacks post-level save data. Missing saves are never imputed as zero."
)

st.markdown("---")

# --- SECTION B: PERFORMANCE TRENDS OVER TIME ---
st.markdown("## 📈 Section B: Performance Trends Over Time")

freq_col1, freq_col2 = st.columns([1, 4])
with freq_col1:
    trend_freq = st.selectbox("Interval Cadence:", ["Weekly", "Daily", "Monthly"], index=0)

time_df = backend.calculate_time_series_trends(filtered_posts, freq=trend_freq)

if len(time_df) > 0:
    trend_tabs = st.tabs(["👁️ Views & Impressions", "❤️ Engagement & High-Intent Shares", "📊 Engagement Rates (%)", "📱 Platform Volume"])

    with trend_tabs[0]:
        fig_views = go.Figure()
        fig_views.add_trace(go.Scatter(
            x=time_df["period_label"], y=time_df["views"],
            mode="lines+markers", name="Views", line=dict(color="#2563EB", width=3)
        ))
        fig_views.add_trace(go.Scatter(
            x=time_df["period_label"], y=time_df["impressions"],
            mode="lines+markers", name="Impressions", line=dict(color="#9333EA", width=2, dash="dash")
        ))
        fig_views.update_layout(
            title=f"{trend_freq} Views & Impressions Trend",
            xaxis_title="Period Ending", yaxis_title="Count",
            hovermode="x unified", template="plotly_white",
            height=405,
            legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=11)),
            margin=dict(t=52, b=100, l=56, r=28),
        )
        fig_views.update_xaxes(automargin=True, title_standoff=6)
        fig_views.update_yaxes(automargin=True)
        st.plotly_chart(fig_views, use_container_width=True)

    with trend_tabs[1]:
        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["likes"], name="Likes", marker_color="#3B82F6"))
        fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["shares"], name="Shares", marker_color="#10B981"))
        fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["saves"], name="Saves", marker_color="#F59E0B"))
        fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["comments"], name="Comments", marker_color="#6366F1"))
        fig_eng.update_layout(
            barmode="stack", title=f"{trend_freq} Engagement Volume Breakdown",
            xaxis_title="Period Ending", yaxis_title="Interactions",
            template="plotly_white", hovermode="x unified",
            height=405,
            legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=11)),
            margin=dict(t=52, b=104, l=56, r=28),
        )
        fig_eng.update_xaxes(automargin=True, title_standoff=6)
        fig_eng.update_yaxes(automargin=True)
        st.plotly_chart(fig_eng, use_container_width=True)

    with trend_tabs[2]:
        fig_rates = go.Figure()
        fig_rates.add_trace(go.Scatter(
            x=time_df["period_label"], y=time_df["views_engagement_rate"],
            mode="lines+markers", name="Views Engagement Rate (%)", line=dict(color="#059669", width=3)
        ))
        fig_rates.add_trace(go.Scatter(
            x=time_df["period_label"], y=time_df["share_rate"],
            mode="lines+markers", name="Share Rate (%)", line=dict(color="#2563EB", width=2)
        ))
        fig_rates.add_trace(go.Scatter(
            x=time_df["period_label"], y=time_df["save_rate"],
            mode="lines+markers", name="Save Rate (%)", line=dict(color="#D97706", width=2)
        ))
        fig_rates.update_layout(
            title=f"{trend_freq} Engagement Rates (%) — Continuous Non-Zero Intervals",
            xaxis_title="Period Ending", yaxis_title="Percentage (%)",
            template="plotly_white", hovermode="x unified",
            height=415,
            legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=10.5)),
            margin=dict(t=52, b=118, l=56, r=28),
        )
        fig_rates.update_xaxes(automargin=True, title_standoff=6)
        fig_rates.update_yaxes(automargin=True)
        st.plotly_chart(fig_rates, use_container_width=True)

    with trend_tabs[3]:
        plat_trend_df = backend.calculate_platform_time_series(filtered_posts, freq=trend_freq)
        if len(plat_trend_df) > 0:
            fig_plat = px.bar(
                plat_trend_df, x="period_label", y="views", color="platform", barmode="group",
                title=f"{trend_freq} Views by Platform",
                labels={"period_label": "Period Ending", "views": "Total Views", "platform": "Platform"},
                template="plotly_white", color_discrete_map={"Instagram": "#E1306C", "Youtube": "#FF0000"}
            )
            fig_plat.update_layout(
                height=405,
                legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, title_text=None, font=dict(size=11)),
                margin=dict(t=52, b=100, l=56, r=28),
            )
            fig_plat.update_xaxes(automargin=True, title_standoff=6)
            fig_plat.update_yaxes(automargin=True)
            st.plotly_chart(fig_plat, use_container_width=True)
else:
    st.info("ℹ️ Insufficient data points to construct time-series trend lines for this filter range.")

st.markdown("---")

# --- SECTION C: PLATFORM COMPARISON ---
st.markdown("## 📱 Section C: Normalized Cross-Platform Comparison")

plat_comp = backend.calculate_platform_comparisons(filtered_posts)
st.dataframe(plat_comp["summary_table"], use_container_width=True)

for disc in plat_comp["disclosures"]:
    st.caption(f"ℹ️ {disc}")

if len(plat_comp["summary_table"]) > 1:
    p_chart_col1, p_chart_col2 = st.columns(2)
    with p_chart_col1:
        fig_p_eng = px.bar(
            plat_comp["summary_table"], x="Platform",
            y=["Normalized Eng Rate (%)", "Full Eng Rate (%)"],
            barmode="group", title="Normalized vs Full Engagement Rate (%) by Platform",
            labels={"value": "Rate (%)", "variable": "Rate Type"},
            template="plotly_white", color_discrete_sequence=["#3B82F6", "#10B981"]
        )
        fig_p_eng.update_layout(
            height=410,
            title_font_size=13,
            legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, title_text=None, font=dict(size=10.5)),
            margin=dict(t=54, b=118, l=52, r=28),
        )
        fig_p_eng.update_xaxes(automargin=True, title_standoff=6)
        fig_p_eng.update_yaxes(automargin=True)
        st.plotly_chart(fig_p_eng, use_container_width=True)

    with p_chart_col2:
        if len(plat_comp["format_distribution"]) > 0:
            fig_p_fmt = px.bar(
                plat_comp["format_distribution"], x="platform", y="post_count", color="content_type",
                title="Content Format Distribution by Platform",
                labels={"platform": "Platform", "post_count": "Post Count", "content_type": "Format"},
                barmode="stack", template="plotly_white"
            )
            fig_p_fmt.update_layout(
                height=410,
                title_font_size=13,
                legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, title_text=None, font=dict(size=10.5)),
                margin=dict(t=54, b=118, l=52, r=28),
            )
            fig_p_fmt.update_xaxes(automargin=True, title_standoff=6)
            fig_p_fmt.update_yaxes(automargin=True)
            st.plotly_chart(fig_p_fmt, use_container_width=True)

st.markdown("---")

# --- SECTION D: CONTENT PERFORMANCE BREAKDOWNS ---
st.markdown("## 🎯 Section D: Content Performance Analysis")
st.caption("🔬 *Observational Analysis Notice: The following distributions represent historical descriptive associations. Confounding variables (topic selection, script pacing, creator schedule) prevent causal interpretation.*")

content_data = backend.calculate_content_breakdowns(filtered_posts)

c_row1_col1, c_row1_col2 = st.columns(2)
with c_row1_col1:
    if len(content_data["by_topic"]) > 0:
        fig_topic = px.bar(
            content_data["by_topic"], x="avg_views", y="topic", orientation="h",
            text="post_count", title="Average Views by Struggle Topic (N = Post Count)",
            labels={"avg_views": "Average Views", "topic": "Struggle Topic"},
            template="plotly_white", color="avg_views", color_continuous_scale="Viridis"
        )
        fig_topic.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
        max_topic_views = float(pd.to_numeric(content_data["by_topic"]["avg_views"], errors="coerce").max() or 0)
        if max_topic_views > 0:
            fig_topic.update_xaxes(range=[0, max_topic_views * 1.28], automargin=True)
        else:
            fig_topic.update_xaxes(automargin=True)
        fig_topic.update_yaxes(automargin=True)
        fig_topic.update_layout(
            height=485,
            title_font_size=13,
            margin=dict(t=56, b=64, l=20, r=58),
            coloraxis_colorbar=dict(
                title=dict(text="Average Views", side="right"),
                thickness=13,
                len=0.80,
            ),
        )
        st.plotly_chart(fig_topic, use_container_width=True)

with c_row1_col2:
    if "shares" in filtered_posts.columns and "views" in filtered_posts.columns:
        scatter_df = filtered_posts.copy()
        scatter_df["saves_display"] = scatter_df["saves"].fillna(0) if "saves" in scatter_df.columns else 0
        scatter_df["views_size"] = pd.to_numeric(scatter_df["views"], errors="coerce").fillna(1).clip(lower=1)
        fig_scatter = px.scatter(
            scatter_df, x="shares", y="saves_display", color="topic" if "topic" in scatter_df.columns else None,
            size="views_size", hover_data=["post_id", "platform", "content_type"],
            title="Saves vs Shares Bubble Chart (Size = Views)",
            labels={"shares": "Total Shares", "saves_display": "Total Saves (Bookmarks)"},
            template="plotly_white"
        )
        fig_scatter.update_xaxes(automargin=True, title_standoff=4)
        fig_scatter.update_yaxes(automargin=True)
        fig_scatter.update_layout(
            height=485,
            title_font_size=13,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.33,
                xanchor="center",
                x=0.5,
                title_text=None,
                font=dict(size=10),
            ),
            margin=dict(t=56, b=196, l=56, r=36),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

c_row2_col1, c_row2_col2 = st.columns(2)
with c_row2_col1:
    if len(content_data["by_duration"]) > 0:
        fig_dur = px.bar(
            content_data["by_duration"], x="duration_bin", y="avg_retention",
            text="post_count", title="Video Duration Tier vs Average Retention Rate (%)",
            labels={"duration_bin": "Duration Window", "avg_retention": "Mean Retention (%)"},
            template="plotly_white", color="avg_retention", color_continuous_scale="Blues"
        )
        fig_dur.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
        max_dur_ret = float(pd.to_numeric(content_data["by_duration"]["avg_retention"], errors="coerce").max() or 0)
        if max_dur_ret > 0:
            fig_dur.update_yaxes(range=[0, max_dur_ret * 1.18], automargin=True)
        else:
            fig_dur.update_yaxes(automargin=True)
        fig_dur.update_xaxes(automargin=True)
        fig_dur.update_layout(
            height=420,
            title_font_size=13,
            margin=dict(t=56, b=102, l=56, r=64),
            coloraxis_colorbar=dict(
                title=dict(text="Mean Retention (%)", side="right"),
                thickness=13,
                len=0.78,
            ),
        )
        st.plotly_chart(fig_dur, use_container_width=True)

with c_row2_col2:
    if len(content_data["by_hook"]) > 0:
        fig_hook = px.bar(
            content_data["by_hook"], x="hook_type", y="share_rate",
            text="post_count", title="Hook Style vs Observational Share Rate (%)",
            labels={"hook_type": "Hook Style", "share_rate": "Mean Share Rate (%)"},
            template="plotly_white", color="share_rate", color_continuous_scale="Teal"
        )
        fig_hook.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
        max_hook_sr = float(pd.to_numeric(content_data["by_hook"]["share_rate"], errors="coerce").max() or 0)
        if max_hook_sr > 0:
            fig_hook.update_yaxes(range=[0, max_hook_sr * 1.18], automargin=True)
        else:
            fig_hook.update_yaxes(automargin=True)
        fig_hook.update_xaxes(automargin=True)
        fig_hook.update_layout(
            height=420,
            title_font_size=13,
            margin=dict(t=56, b=102, l=56, r=64),
            coloraxis_colorbar=dict(
                title=dict(text="Mean Share Rate (%)", side="right"),
                thickness=13,
                len=0.78,
            ),
        )
        st.plotly_chart(fig_hook, use_container_width=True)

# Timing Heatmap
if not content_data["timing_matrix"].empty and "mean" in content_data["timing_matrix"]:
    st.subheader("🗓️ Posting Schedule Matrix: Average Views by Hour & Day of Week")
    mean_views_matrix = content_data["timing_matrix"]["mean"].fillna(0)
    fig_heat = px.imshow(
        mean_views_matrix,
        labels=dict(x="Day of Week", y="Posting Hour (UTC)", color="Mean Views"),
        color_continuous_scale="YlGnBu",
        title="Heatmap: Mean Views by Upload Window"
    )
    fig_heat.update_xaxes(automargin=True)
    fig_heat.update_yaxes(automargin=True)
    fig_heat.update_layout(
        height=430,
        margin=dict(t=56, b=76, l=56, r=68),
        coloraxis_colorbar=dict(
            title=dict(text="Mean Views", side="right"),
            thickness=13,
            len=0.82,
        ),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# --- SECTION E: AUDIENCE INSIGHTS (MODULE 3 INTEGRATION) ---
st.markdown("## 🧠 Section E: Audience Insights & Relatability (Module 3)")
st.caption("ℹ️ *NLP Sentiment & Relatability Model Disclosure: Predictions are generated by the hybrid TF-IDF + rule-based pipeline (Module 3). They represent algorithmic classifications on synthetic comments, NOT verified ground-truth human emotion.*")

aud_data = backend.calculate_audience_insights()

if aud_data["has_data"]:
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        fig_sent = px.pie(
            aud_data["sentiment_dist"], names="sentiment", values="count",
            title=f"Audience Sentiment Valence Distribution (N = {aud_data['total_comments']})",
            color="sentiment",
            color_discrete_map={"Positive": "#10B981", "Neutral": "#94A3B8", "Negative": "#EF4444", "Mixed": "#F59E0B"},
            hole=0.45
        )
        fig_sent.update_layout(
            height=360,
            title_font_size=13,
            legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5, font=dict(size=11)),
            margin=dict(t=52, b=68, l=20, r=20),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with a_col2:
        fig_rel = px.pie(
            aud_data["relatability_dist"], names="relatability", values="count",
            title=f"Audience Relatability Classification (N = {aud_data['total_comments']})",
            color="relatability",
            color_discrete_map={"Relatable": "#059669", "Neutral": "#64748B", "Uncertain": "#F59E0B"},
            hole=0.45
        )
        fig_rel.update_layout(
            height=360,
            title_font_size=13,
            legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5, font=dict(size=11)),
            margin=dict(t=52, b=68, l=20, r=20),
        )
        st.plotly_chart(fig_rel, use_container_width=True)

    if len(aud_data["topic_relatability"]) > 0:
        fig_topic_rel = px.bar(
            aud_data["topic_relatability"], x="post_relatability_rate", y="topic", orientation="h",
            title="Percentage of Comments Tagged 'Relatable' by Struggle Topic",
            labels={"post_relatability_rate": "Relatable Comment Rate (%)", "topic": "Struggle Topic"},
            template="plotly_white", color="post_relatability_rate", color_continuous_scale="Greens"
        )
        fig_topic_rel.update_xaxes(automargin=True)
        fig_topic_rel.update_yaxes(automargin=True)
        fig_topic_rel.update_layout(
            height=400,
            margin=dict(t=56, b=58, l=20, r=68),
            coloraxis_colorbar=dict(
                title=dict(text="Relatable Rate (%)", side="right"),
                thickness=13,
                len=0.78,
            ),
        )
        st.plotly_chart(fig_topic_rel, use_container_width=True)

    st.info(f"📊 **Empirical Correlation Disclosure:** {aud_data['correlation_text']}")
else:
    st.info("ℹ️ Audience comment dataset is unavailable.")

st.markdown("---")

# --- SECTION F: EXPERIMENT PERFORMANCE (MODULE 4 INTEGRATION) ---
st.markdown("## 🧪 Section F: A/B Testing & Controlled Experiments (Module 4)")
st.caption("ℹ️ *Statistical Inference Notice: Evaluated at independent post units (which addresses intra-post exposure dependence, though creator-level clustering across posts is uncorrected). Synthetic benchmark experiments.*")

exp_data = backend.calculate_experiment_performance()

if exp_data["experiments"]:
    # Forest Plot
    if not exp_data["forest_plot_data"].empty:
        fp_df = exp_data["forest_plot_data"]
        fig_forest = go.Figure()
        for _, row in fp_df.iterrows():
            marker_color = "#059669" if row["is_significant"] and row["type"] == "Randomized RCT" else ("#DC2626" if not row["is_significant"] else "#3B82F6")
            fig_forest.add_trace(go.Scatter(
                x=[row["effect_size"]], y=[row["experiment"]],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[max(0, row["ci_high"] - row["effect_size"])],
                    arrayminus=[max(0, row["effect_size"] - row["ci_low"])],
                    color=marker_color, width=4
                ),
                mode="markers",
                marker=dict(size=12, color=marker_color),
                name=f"{row['experiment']} ({row['type']})"
            ))
        fig_forest.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Null Effect (0)")
        fig_forest.update_layout(
            title="A/B Experiment Effect Sizes & 95% Confidence Intervals (Forest Plot)",
            title_font_size=12,
            xaxis_title="Standardized Effect Size<br>(Hedges' g / Cohen's h)",
            yaxis_title="Experiment", template="plotly_white", showlegend=False,
            margin=dict(t=48, b=72, l=36, r=42)
        )
        fig_forest.update_xaxes(automargin=True, title_standoff=14, title_font_size=10)
        fig_forest.update_yaxes(automargin=True)
        st.plotly_chart(fig_forest, use_container_width=True)

    # Detailed Experiment Cards
    for exp_res in exp_data["experiments"]:
        badge_color = "#10B981" if exp_res["is_randomized"] and exp_res["is_significant"] else ("#F59E0B" if not exp_res["is_randomized"] else "#EF4444")
        badge_label = "🟢 RANDOMIZED RCT (STAT. SIG)" if exp_res["is_randomized"] and exp_res["is_significant"] else ("🟡 OBSERVATIONAL COHORT (NON-RANDOM)" if not exp_res["is_randomized"] else "🔴 INCONCLUSIVE")

        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border-left: 5px solid {badge_color}; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b>{exp_res['experiment_id']}: {exp_res['name']}</b>
                    <span style="font-size: 11px; font-weight: 700; color: {badge_color};">{badge_label}</span>
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
                    Platform: <b>{exp_res['platform']}</b> | 
                    Sample Size: <b>N = {exp_res['sample_size']} posts</b> | 
                    Unit: <b>{exp_res['unit_of_analysis']}</b> | 
                    Test: <b>{exp_res['test_name']}</b>
                </div>
                <div style="font-size: 13px; margin-top: 6px; color: #1E293B;">
                    • <b>Variant A:</b> {exp_res['variant_a']} (Mean: {exp_res['value_a']:.4f})<br>
                    • <b>Variant B:</b> {exp_res['variant_b']} (Mean: {exp_res['value_b']:.4f})<br>
                    • <b>Observed Relative Lift:</b> <b>{exp_res['relative_lift_pct']:+.2f}%</b> | 
                    Effect Size: <b>{exp_res['effect_size']:.3f} ({exp_res['effect_size_type']})</b>
                </div>
                <div style="font-size: 12px; margin-top: 6px; color: #0284C7;">
                    <b>Statistical Inference:</b> Raw p = {exp_res['raw_p_value']:.2e} | 
                    <b>Holm-Bonferroni Adj. p = {exp_res['holm_adj_p_value']:.2e}</b> | 
                    95% CI: [{exp_res['confidence_interval'][0]:,.2f}, {exp_res['confidence_interval'][1]:,.2f}] ({exp_res['ci_method']})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("ℹ️ Experiment dataset is unavailable.")

st.markdown("---")

# --- SECTION G: STRATEGIC OPTIMIZATION & GROWTH OPPORTUNITIES (MODULE 5) ---
st.markdown("## 🚀 Section G: Strategic Recommendations & Opportunities (Module 5)")
st.caption("ℹ️ *Prescriptive Engine Notice: Priorities are ranked using the deterministic 7-component formula. The priority score is a strategic decision index and does NOT represent a calibrated probability of virality or success.*")

recs_report = backend.calculate_recommendation_priorities()

if recs_report.recommendations:
    # Ranked Bar Chart of Priority Scores
    rec_bar_data = []
    for r in recs_report.recommendations:
        rec_bar_data.append({
            "title": f"#{r.recommendation_id}: {r.title[:35]}...",
            "priority_score": r.priority_score,
            "confidence": r.confidence_level.value,
            "category": r.category.value if hasattr(r.category, "value") else str(r.category)
        })
    rec_bar_df = pd.DataFrame(rec_bar_data).sort_values("priority_score", ascending=True)

    fig_recs = px.bar(
        rec_bar_df, x="priority_score", y="title", orientation="h", color="confidence",
        title="Top Optimization Recommendations Ranked by Priority Score",
        labels={"priority_score": "Deterministic Priority<br>Score (0-1)", "title": "Recommendation"},
        template="plotly_white",
        color_discrete_map={"HIGH": "#10B981", "MEDIUM": "#F59E0B", "LOW": "#64748B", "EXPLORATORY": "#64748B"}
    )
    fig_recs.update_xaxes(automargin=True, title_standoff=16, title_font_size=9.5)
    fig_recs.update_yaxes(automargin=True, tickfont=dict(size=9))
    fig_recs.update_layout(
        height=460,
        title_font_size=11.5,
        legend=dict(orientation="h", yanchor="top", y=-0.29, xanchor="center", x=0.5, title_text="Confidence:", font=dict(size=11)),
        margin=dict(t=56, b=114, l=20, r=36),
    )
    st.plotly_chart(fig_recs, use_container_width=True)

    # Cards
    for idx, r in enumerate(recs_report.recommendations[:5], 1):
        c_badge = "🟢 HIGH EVIDENCE" if r.confidence_level == ConfidenceLevel.HIGH else "⚪ EXPLORATORY / LOW"
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>#{idx}. {r.title}</b>
                    <span style="font-size: 11px; font-weight: 700;">{c_badge} (Score: {r.priority_score:.4f})</span>
                </div>
                <div style="font-size: 13px; color: #334155; margin-top: 4px;"><b>Suggested Action:</b> {r.suggested_action}</div>
                <div style="font-size: 12px; color: #0284C7; margin-top: 4px;"><b>Expected Impact:</b> {r.expected_impact_direction} | <b>Recommended Validation:</b> {r.suggested_validation_method}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Conflicting Signals Resolution
    if recs_report.conflicting_signals:
        with st.expander("⚖️ Conflicting Evidence Resolution (Click to View)", expanded=False):
            for conf in recs_report.conflicting_signals:
                st.markdown(f"**Conflict:** {conf['area']}")
                st.markdown(f"- *Observational Signal:* {conf['observational_signal']}")
                st.markdown(f"- *Experimental Evidence:* {conf['experimental_evidence']}")
                st.info(f"💡 *Resolution:* {conf['resolution']}")

st.markdown("---")
st.caption("✅ Unlox Data-Driven Social Engagement Initiative — Module 6 Growth Visualization Dashboard Finalized.")

