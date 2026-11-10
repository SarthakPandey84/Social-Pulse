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

from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.data_loader import (
    DatasetLoadError,
    SYNTHETIC_DEMO_DATASET_ID,
    get_dashboard_experiment_performance,
    get_dashboard_recommendation_priorities,
    get_growth_dashboard_backend,
    list_available_datasets,
)
from app.components.ui_widgets import (
    build_dashboard_experiment_card_html,
    build_dashboard_recommendation_card_html,
    render_dataset_selector,
    render_footer,
    render_header,
    render_metric_card,
    render_provenance_banner,
)
from src.recommender.schemas import ConfidenceLevel

# Branded Header
render_header(
    title="Module 6: Growth Visualization Dashboard",
    subtitle="Unified cross-module descriptive analytics, experiment tracking, and prescriptive strategy"
)

# Unified Workspace Dataset Selector
all_datasets = list_available_datasets(root_dir=root_dir)
posts_datasets = list_available_datasets(root_dir=root_dir, required_table="posts")
selected_dataset_id = render_dataset_selector(
    posts_datasets,
    all_dataset_ids=[d["dataset_id"] for d in all_datasets],
    key="sp_dataset_selector_p6",
    location="sidebar",
)

# Initialize Backend Engine (keyed by workspace + dataset_id + underlying dataset signatures)
try:
    backend = get_growth_dashboard_backend(root_dir, dataset_id=selected_dataset_id)
except DatasetLoadError as exc:
    st.error(f"❌ Dataset Load Error: {exc}")
    st.stop()

if backend.posts_df is None or len(backend.posts_df) == 0:
    st.error("❌ No analytical datasets found for the selected corpus.")
    st.stop()

if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    st.warning(
        "⚠️ **SYNTHETIC DEMONSTRATION DATA:** These visualizations demonstrate analytical functionality across "
        "Modules 1–5 using synthetic benchmark fixtures and do not represent verified real-world creator performance. "
        "Observational correlations are strictly separated from experimental evidence and are never presented as proven causal relationships."
    )
else:
    active_meta = backend.dataset_meta or next(
        (d for d in posts_datasets if d["dataset_id"] == selected_dataset_id),
        {"dataset_id": selected_dataset_id},
    )
    render_provenance_banner(active_meta)

# --- SIDEBAR: GLOBAL INTERACTIVE FILTERS ---
st.sidebar.title("🔍 Global Dashboard Filters")

# Date range bounds from data (only when valid timestamps are present)
ts_col = (
    "timestamp_dt"
    if "timestamp_dt" in backend.posts_df.columns
    else ("timestamp" if "timestamp" in backend.posts_df.columns else None)
)
all_dates = (
    pd.to_datetime(backend.posts_df[ts_col], utc=True, errors="coerce").dropna()
    if ts_col is not None
    else pd.Series(dtype="datetime64[ns, UTC]")
)

start_date = None
end_date = None
if not all_dates.empty:
    min_data_date = all_dates.min().date()
    max_data_date = all_dates.max().date()
    if max_data_date > min_data_date:
        date_range = st.sidebar.date_input(
            "📅 Date Range:",
            value=(min_data_date, max_data_date),
            min_value=min_data_date,
            max_value=max_data_date,
        )
        start_date = date_range[0] if isinstance(date_range, (list, tuple)) and len(date_range) > 0 else min_data_date
        end_date = date_range[1] if isinstance(date_range, (list, tuple)) and len(date_range) > 1 else max_data_date
    else:
        st.sidebar.caption(f"📅 Observed Date: `{min_data_date}`")
else:
    st.sidebar.caption("📅 Timestamp filtering unavailable (single-snapshot corpus).")

# Platform filter (dynamic across Instagram / YouTube / TikTok)
dynamic_plats = (
    sorted(backend.posts_df["platform"].dropna().astype(str).str.strip().str.title().unique().tolist())
    if "platform" in backend.posts_df.columns
    else []
)
platform_options = ["All Platforms"] + dynamic_plats
selected_plat_raw = st.sidebar.selectbox("📱 Platform:", platform_options, index=0)
selected_platform = None if selected_plat_raw == "All Platforms" else selected_plat_raw.lower()

# Content format filter
format_options = (
    ["All Formats"] + sorted(list(backend.posts_df["content_type"].dropna().astype(str).unique()))
    if "content_type" in backend.posts_df.columns
    else ["All Formats"]
)
selected_fmt_raw = st.sidebar.selectbox("🎬 Content Format:", format_options, index=0)
selected_format = None if selected_fmt_raw == "All Formats" else selected_fmt_raw

# Topic filter
topic_options = (
    ["All Topics"] + sorted(list(backend.posts_df["topic"].dropna().astype(str).unique()))
    if "topic" in backend.posts_df.columns
    else ["All Topics"]
)
selected_topic_raw = st.sidebar.selectbox("💡 Struggle / Content Topic:", topic_options, index=0)
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
avail = kpis.get("metric_availability", {})

row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
with row1_col1:
    render_metric_card("Analyzed Posts", f"{kpis['total_posts']:,}")
with row1_col2:
    _tot_views = kpis.get("total_views")
    render_metric_card("Total Views", f"{_tot_views:,}" if (avail.get("views", False) and _tot_views is not None) else "N/A (Unavailable)")
with row1_col3:
    _tot_imp = kpis.get("total_impressions")
    render_metric_card("Total Impressions", f"{_tot_imp:,}" if (avail.get("impressions", False) and _tot_imp is not None) else "N/A (Unavailable)")
with row1_col4:
    _tot_nf = kpis.get("total_new_followers")
    render_metric_card("New Followers", f"{_tot_nf:,}" if (avail.get("new_followers", False) and _tot_nf is not None) else "N/A (Unavailable)")

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
with row2_col1:
    _tot_likes = kpis.get("total_likes")
    render_metric_card("Total Likes", f"{_tot_likes:,}" if (avail.get("likes", False) and _tot_likes is not None) else "N/A (Unavailable)")
with row2_col2:
    _tot_comm = kpis.get("total_comments")
    render_metric_card("Total Comments", f"{_tot_comm:,}" if (avail.get("comments_count", False) and _tot_comm is not None) else "N/A (Unavailable)")
with row2_col3:
    _tot_shares = kpis.get("total_shares")
    render_metric_card("Total Shares", f"{_tot_shares:,}" if (avail.get("shares", False) and _tot_shares is not None) else "N/A (Unavailable)")
with row2_col4:
    _tot_saves = kpis.get("total_saves")
    _tot_collects = kpis.get("total_collect_count_proxy")
    if avail.get("saves", False) and _tot_saves is not None:
        render_metric_card("Total Saves (IG)", f"{_tot_saves:,}")
    elif avail.get("collect_count", False) and _tot_collects is not None:
        render_metric_card("Collects (TikTok Proxy)", f"{_tot_collects:,}")
    else:
        render_metric_card("Total Saves (IG)", "N/A (Unavailable)")

row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
with row3_col1:
    _pooled_vr = kpis.get("pooled_views_engagement_rate")
    render_metric_card(
        "Pooled Views Eng. Rate",
        f"{_pooled_vr:.2f}%" if (avail.get("views", False) and _pooled_vr is not None) else "N/A (Unavailable)",
    )
with row3_col2:
    _mean_vr = kpis.get("mean_post_level_views_engagement_rate")
    render_metric_card(
        "Mean Post-Level Views Rate",
        f"{_mean_vr:.2f}%" if (avail.get("views", False) and _mean_vr is not None) else "N/A (Unavailable)",
    )
with row3_col3:
    _pooled_rr = kpis.get("pooled_reach_engagement_rate")
    render_metric_card(
        "Reach Eng. Rate (IG Pooled)",
        f"{_pooled_rr:.2f}%" if (avail.get("reach", False) and _pooled_rr is not None) else "N/A (Platform)",
    )
with row3_col4:
    _avg_views = kpis.get("avg_views_per_post")
    render_metric_card(
        "Avg Views / Post",
        f"{_avg_views:,.0f}" if (avail.get("views", False) and _avg_views is not None) else "N/A (Unavailable)",
    )

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
        if avail.get("views", False):
            fig_views.add_trace(go.Scatter(
                x=time_df["period_label"], y=time_df["views"],
                mode="lines+markers", name="Views", line=dict(color="#2563EB", width=3)
            ))
        if avail.get("impressions", False):
            fig_views.add_trace(go.Scatter(
                x=time_df["period_label"], y=time_df["impressions"],
                mode="lines+markers", name="Impressions", line=dict(color="#9333EA", width=2, dash="dash")
            ))
        if len(fig_views.data) > 0:
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
            st.plotly_chart(fig_views, width="stretch")
        else:
            st.info("ℹ️ Views & Impressions time-series data are unavailable in the selected corpus.")

    with trend_tabs[1]:
        fig_eng = go.Figure()
        if avail.get("likes", False):
            fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["likes"], name="Likes", marker_color="#3B82F6"))
        if avail.get("shares", False):
            fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["shares"], name="Shares", marker_color="#10B981"))
        if avail.get("saves", False):
            fig_eng.add_trace(go.Bar(x=time_df["period_label"], y=time_df["saves"], name="Saves", marker_color="#F59E0B"))
        if avail.get("comments_count", False):
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
        st.plotly_chart(fig_eng, width="stretch")

    with trend_tabs[2]:
        if avail.get("views", False):
            fig_rates = go.Figure()
            fig_rates.add_trace(go.Scatter(
                x=time_df["period_label"], y=time_df["views_engagement_rate"],
                mode="lines+markers", name="Views Engagement Rate (%)", line=dict(color="#059669", width=3)
            ))
            if avail.get("shares", False):
                fig_rates.add_trace(go.Scatter(
                    x=time_df["period_label"], y=time_df["share_rate"],
                    mode="lines+markers", name="Share Rate (%)", line=dict(color="#2563EB", width=2)
                ))
            if avail.get("saves", False):
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
            st.plotly_chart(fig_rates, width="stretch")
        else:
            st.info("ℹ️ Engagement rates per view require non-null views.")

    with trend_tabs[3]:
        plat_trend_df = backend.calculate_platform_time_series(filtered_posts, freq=trend_freq)
        if len(plat_trend_df) > 0 and avail.get("views", False):
            fig_plat = px.bar(
                plat_trend_df, x="period_label", y="views", color="platform", barmode="group",
                title=f"{trend_freq} Views by Platform",
                labels={"period_label": "Period Ending", "views": "Total Views", "platform": "Platform"},
                template="plotly_white",
                color_discrete_map={"Instagram": "#E1306C", "Youtube": "#FF0000", "Tiktok": "#0F172A"},
            )
            fig_plat.update_layout(
                height=405,
                legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, title_text=None, font=dict(size=11)),
                margin=dict(t=52, b=100, l=56, r=28),
            )
            fig_plat.update_xaxes(automargin=True, title_standoff=6)
            fig_plat.update_yaxes(automargin=True)
            st.plotly_chart(fig_plat, width="stretch")
        else:
            st.info("ℹ️ Platform view time-series unavailable in this corpus.")
else:
    st.info("ℹ️ Insufficient data points to construct time-series trend lines for this filter range.")

st.markdown("---")

# --- SECTION C: PLATFORM COMPARISON ---
st.markdown("## 📱 Section C: Normalized Cross-Platform Comparison")

plat_comp = backend.calculate_platform_comparisons(filtered_posts)
st.dataframe(plat_comp["summary_table"], width="stretch")

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
        st.plotly_chart(fig_p_eng, width="stretch")

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
            st.plotly_chart(fig_p_fmt, width="stretch")

st.markdown("---")

# --- SECTION D: CONTENT PERFORMANCE BREAKDOWNS ---
st.markdown("## 🎯 Section D: Content Performance Analysis")
st.caption("🔬 *Observational Analysis Notice: The following distributions represent historical descriptive associations. Confounding variables (topic selection, script pacing, creator schedule) prevent causal interpretation.*")

content_data = backend.calculate_content_breakdowns(filtered_posts)

c_row1_col1, c_row1_col2 = st.columns(2)
with c_row1_col1:
    by_topic_df = content_data["by_topic"]
    has_topic_views = len(by_topic_df) > 0 and "avg_views" in by_topic_df.columns and by_topic_df["avg_views"].notna().any()
    has_topic_likes = len(by_topic_df) > 0 and "avg_likes" in by_topic_df.columns and by_topic_df["avg_likes"].notna().any()
    if has_topic_views or has_topic_likes:
        metric_col = "avg_views" if has_topic_views else "avg_likes"
        metric_title = "Average Views" if has_topic_views else "Average Likes"
        plot_topic_df = by_topic_df.dropna(subset=[metric_col]).head(15)
        fig_topic = px.bar(
            plot_topic_df, x=metric_col, y="topic", orientation="h",
            text="post_count", title=f"{metric_title} by Content Topic (Top {len(plot_topic_df)}, N = Post Count)",
            labels={metric_col: metric_title, "topic": "Topic"},
            template="plotly_white", color=metric_col, color_continuous_scale="Viridis"
        )
        fig_topic.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
        max_topic_val_s = pd.to_numeric(plot_topic_df[metric_col], errors="coerce").dropna()
        max_topic_val = float(max_topic_val_s.max()) if not max_topic_val_s.empty else 0.0
        if max_topic_val > 0:
            fig_topic.update_xaxes(range=[0, max_topic_val * 1.28], automargin=True)
        else:
            fig_topic.update_xaxes(automargin=True)
        fig_topic.update_yaxes(automargin=True)
        fig_topic.update_layout(
            height=485,
            title_font_size=13,
            margin=dict(t=56, b=64, l=20, r=58),
            coloraxis_colorbar=dict(
                title=dict(text=metric_title, side="right"),
                thickness=13,
                len=0.80,
            ),
        )
        st.plotly_chart(fig_topic, width="stretch")
    else:
        st.info("ℹ️ Topic breakdown is unavailable for the selected corpus.")

with c_row1_col2:
    can_scatter = (
        "shares" in filtered_posts.columns
        and "views" in filtered_posts.columns
        and filtered_posts["shares"].notna().any()
        and filtered_posts["views"].notna().any()
    )
    if can_scatter:
        scatter_source = (
            filtered_posts.sample(n=5000, random_state=42)
            if len(filtered_posts) > 5000
            else filtered_posts
        ).copy()
        if "saves" in scatter_source.columns and scatter_source["saves"].notna().any():
            y_col = "saves_display"
            scatter_source[y_col] = pd.to_numeric(scatter_source["saves"], errors="coerce").fillna(0)
            y_label = "Total Saves (Bookmarks)"
            chart_title = "Saves vs Shares Bubble Chart (Size = Views)"
        elif "collect_count" in scatter_source.columns and scatter_source["collect_count"].notna().any():
            y_col = "collects_display"
            scatter_source[y_col] = pd.to_numeric(scatter_source["collect_count"], errors="coerce").fillna(0)
            y_label = "Collects (TikTok Bookmark Proxy)"
            chart_title = "Collects (Proxy) vs Shares Bubble Chart (Size = Views)"
        else:
            y_col = "likes_display"
            scatter_source[y_col] = pd.to_numeric(scatter_source.get("likes", 0), errors="coerce").fillna(0)
            y_label = "Total Likes"
            chart_title = "Likes vs Shares Bubble Chart (Size = Views)"

        scatter_source["views_size"] = pd.to_numeric(scatter_source["views"], errors="coerce").fillna(1).clip(lower=1)
        hover_cols = [c for c in ["post_id", "platform", "content_type"] if c in scatter_source.columns]
        color_col = "topic" if ("topic" in scatter_source.columns and scatter_source["topic"].nunique(dropna=True) <= 15 and scatter_source["topic"].notna().any()) else None
        if len(filtered_posts) > 5000:
            chart_title += " [5,000-Post Sample]"
        fig_scatter = px.scatter(
            scatter_source, x="shares", y=y_col, color=color_col,
            size="views_size", hover_data=hover_cols or None,
            title=chart_title,
            labels={"shares": "Total Shares", y_col: y_label},
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
        st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.info("ℹ️ Scatter interaction breakdown requires non-null views and shares.")

c_row2_col1, c_row2_col2 = st.columns(2)
with c_row2_col1:
    by_dur_df = content_data["by_duration"]
    if len(by_dur_df) > 0:
        has_ret = "avg_retention" in by_dur_df.columns and (pd.to_numeric(by_dur_df["avg_retention"], errors="coerce").fillna(0) > 0).any()
        has_dur_views = "avg_views" in by_dur_df.columns and by_dur_df["avg_views"].notna().any()
        if has_ret or has_dur_views:
            dur_y = "avg_retention" if has_ret else "avg_views"
            dur_y_label = "Mean Retention (%)" if has_ret else "Average Views"
            dur_title = "Video Duration Tier vs Average Retention Rate (%)" if has_ret else "Video Duration Tier vs Average Views"
            fig_dur = px.bar(
                by_dur_df, x="duration_bin", y=dur_y,
                text="post_count", title=dur_title,
                labels={"duration_bin": "Duration Window", dur_y: dur_y_label},
                template="plotly_white", color=dur_y, color_continuous_scale="Blues"
            )
            fig_dur.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
            max_dur_s = pd.to_numeric(by_dur_df[dur_y], errors="coerce").dropna()
            max_dur_val = float(max_dur_s.max()) if not max_dur_s.empty else 0.0
            if max_dur_val > 0:
                fig_dur.update_yaxes(range=[0, max_dur_val * 1.18], automargin=True)
            else:
                fig_dur.update_yaxes(automargin=True)
            fig_dur.update_xaxes(automargin=True)
            fig_dur.update_layout(
                height=420,
                title_font_size=13,
                margin=dict(t=56, b=102, l=56, r=64),
                coloraxis_colorbar=dict(
                    title=dict(text=dur_y_label, side="right"),
                    thickness=13,
                    len=0.78,
                ),
            )
            st.plotly_chart(fig_dur, width="stretch")

with c_row2_col2:
    by_hook_df = content_data["by_hook"]
    if len(by_hook_df) > 0 and (pd.to_numeric(by_hook_df["share_rate"], errors="coerce").fillna(0) > 0).any():
        fig_hook = px.bar(
            by_hook_df, x="hook_type", y="share_rate",
            text="post_count", title="Hook Style vs Observational Share Rate (%)",
            labels={"hook_type": "Hook Style", "share_rate": "Mean Share Rate (%)"},
            template="plotly_white", color="share_rate", color_continuous_scale="Teal"
        )
        fig_hook.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
        max_hook_s = pd.to_numeric(by_hook_df["share_rate"], errors="coerce").dropna()
        max_hook_sr = float(max_hook_s.max()) if not max_hook_s.empty else 0.0
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
        st.plotly_chart(fig_hook, width="stretch")

# Timing Heatmap
if not content_data["timing_matrix"].empty and "mean" in content_data["timing_matrix"] and content_data["timing_matrix"]["mean"].notna().any().any():
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
    st.plotly_chart(fig_heat, width="stretch")

st.markdown("---")

# --- SECTION E: AUDIENCE INSIGHTS (MODULE 3 INTEGRATION) ---
st.markdown("## 🧠 Section E: Audience Insights & Relatability (Module 3)")
if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    st.caption("ℹ️ *NLP Sentiment & Relatability Model Disclosure: Predictions are generated by the hybrid TF-IDF + rule-based pipeline (Module 3). They represent algorithmic classifications on synthetic comments, NOT verified ground-truth human emotion.*")
else:
    st.caption("ℹ️ *NLP Sentiment & Relatability Model Disclosure: Predictions are generated by the hybrid TF-IDF + rule-based pipeline (Module 3) on external public comments.*")

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
        st.plotly_chart(fig_sent, width="stretch")

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
        st.plotly_chart(fig_rel, width="stretch")

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
        st.plotly_chart(fig_topic_rel, width="stretch")

    st.info(f"📊 **Empirical Correlation Disclosure:** {aud_data['correlation_text']}")
else:
    st.info("ℹ️ Audience comment dataset is unavailable for the selected corpus.")

st.markdown("---")

# --- SECTION F: EXPERIMENT PERFORMANCE (MODULE 4 INTEGRATION) ---
st.markdown("## 🧪 Section F: A/B Testing & Controlled Experiments (Module 4)")
if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    st.caption("ℹ️ *Statistical Inference Notice: Evaluated at independent post units (which addresses intra-post exposure dependence, though creator-level clustering across posts is uncorrected). Synthetic benchmark experiments.*")
else:
    st.caption("ℹ️ *Statistical Inference Notice: Synthetic A/B experiment tables are separated from external corpora. Use Module 04 (A/B Testing) to analyze creator-clustered observational cohorts on Zenodo / InstaArt.*")

exp_data = get_dashboard_experiment_performance(backend)

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
        st.plotly_chart(fig_forest, width="stretch")

    # Detailed Experiment Cards
    for exp_res in exp_data["experiments"]:
        st.markdown(
            build_dashboard_experiment_card_html(
                is_randomized=bool(exp_res["is_randomized"]),
                is_significant=bool(exp_res["is_significant"]),
                experiment_id=exp_res["experiment_id"],
                name=exp_res["name"],
                platform=exp_res["platform"],
                sample_size=exp_res["sample_size"],
                unit_of_analysis=exp_res["unit_of_analysis"],
                test_name=exp_res["test_name"],
                variant_a=exp_res["variant_a"],
                value_a=exp_res["value_a"],
                variant_b=exp_res["variant_b"],
                value_b=exp_res["value_b"],
                relative_lift_pct=exp_res["relative_lift_pct"],
                effect_size=exp_res["effect_size"],
                effect_size_type=exp_res["effect_size_type"],
                raw_p_value=exp_res["raw_p_value"],
                holm_adj_p_value=exp_res["holm_adj_p_value"],
                ci_low=exp_res["confidence_interval"][0],
                ci_high=exp_res["confidence_interval"][1],
                ci_method=exp_res["ci_method"],
            ),
            unsafe_allow_html=True,
        )
else:
    st.info("ℹ️ Experiment dataset is unavailable.")

st.markdown("---")

# --- SECTION G: STRATEGIC OPTIMIZATION & GROWTH OPPORTUNITIES (MODULE 5) ---
st.markdown("## 🚀 Section G: Strategic Recommendations & Opportunities (Module 5)")
st.caption("ℹ️ *Prescriptive Engine Notice: Priorities are ranked using the deterministic 7-component formula. The priority score is a strategic decision index and does NOT represent a calibrated probability of virality or success.*")

recs_report = get_dashboard_recommendation_priorities(backend)

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
    st.plotly_chart(fig_recs, width="stretch")

    # Cards
    for idx, r in enumerate(recs_report.recommendations[:5], 1):
        conf_val = r.confidence_level.value if isinstance(r.confidence_level, ConfidenceLevel) else str(r.confidence_level)
        st.markdown(
            build_dashboard_recommendation_card_html(
                idx=idx,
                confidence_level=conf_val,
                title=r.title,
                priority_score=r.priority_score,
                suggested_action=r.suggested_action,
                expected_impact_direction=r.expected_impact_direction,
                suggested_validation_method=r.suggested_validation_method,
            ),
            unsafe_allow_html=True,
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
render_footer("SocialPulse Data-Driven Social Engagement Initiative — Module 6 Growth Visualization Dashboard")

