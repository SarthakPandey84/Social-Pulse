"""Page 1: Content Performance Tracker.

Consumes validated, cleaned, and feature-engineered datasets from Module 1.
Displays provenance metadata, missing value distributions, historical snapshots, and platform-specific metrics.
"""

from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.components.data_loader import (
    DatasetLoadError,
    SYNTHETIC_DEMO_DATASET_ID,
    list_available_datasets,
    load_posts_dataset,
    load_selected_posts_dataset,
)
from app.components.ui_widgets import (
    render_dataset_selector,
    render_header,
    render_metric_card,
    render_provenance_banner,
)
from src.preprocessing.cleaning import get_latest_posts_view

render_header(
    title="Module 1: Content Performance Tracker",
    subtitle="Unified cross-platform ingestion, schema validation, and feature-engineered engagement signals"
)

all_datasets = list_available_datasets(root_dir=root_dir)
posts_datasets = list_available_datasets(root_dir=root_dir, required_table="posts")
selected_dataset_id = render_dataset_selector(
    posts_datasets,
    all_dataset_ids=[d["dataset_id"] for d in all_datasets],
    key="sp_dataset_selector_p1",
    location="sidebar",
)

# Load Module 1 dataset via shared cached loader (fail-loud on corrupt external dataset)
processed_path = root_dir / "data" / "processed" / "processed_posts.parquet"
sample_path = root_dir / "data" / "sample" / "sample_posts.csv"
active_meta = {"dataset_id": selected_dataset_id, "provenance": "synthetic_data"}

if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    df, source_kind = load_posts_dataset(processed_path, sample_path)
    if source_kind == "processed" and df is not None:
        data_source_badge = "✅ Source: Processed Data Pipeline (data/processed/)"
    elif source_kind == "sample" and df is not None:
        data_source_badge = "⚠️ Source: Sample Offline Benchmark (data/sample/)"
    else:
        st.warning("No ingested datasets found. Please run the ingestion pipeline.")
        st.stop()
else:
    projected_cols = [
        "platform", "post_id", "creator_pseudo_id", "timestamp", "collection_timestamp",
        "content_type", "topic", "views", "impressions", "reach", "likes",
        "comments_count", "shares", "saves", "collect_count", "retention_rate",
        "high_intent_actions", "high_intent_completeness", "save_to_share_ratio",
        "save_share_ratio_status", "data_completeness_pct", "provenance", "source_dataset",
    ]
    try:
        df, active_meta = load_selected_posts_dataset(
            selected_dataset_id,
            root_dir=root_dir,
            columns=projected_cols,
        )
    except DatasetLoadError as exc:
        st.error(f"❌ Dataset Load Error: {exc}")
        st.stop()
    render_provenance_banner(active_meta)
    data_source_badge = f"✅ Source: External Corpus (`{selected_dataset_id}` | {active_meta.get('provenance', 'external')})"

st.caption(data_source_badge)

# Ingestion Summary & Quality KPIs
total_snapshots = len(df)
latest_posts_df = get_latest_posts_view(df) if "post_id" in df.columns else df
unique_posts_count = len(latest_posts_df)

platforms = sorted(str(p) for p in df["platform"].dropna().unique()) if "platform" in df.columns else ["N/A"]
if "data_completeness_pct" in df.columns and df["data_completeness_pct"].notna().any():
    completeness_str = f"{round(float(df['data_completeness_pct'].mean()), 1)}%"
else:
    canonical_observed = [c for c in ["likes", "comments_count", "views", "shares", "saves", "reach"] if c in df.columns and df[c].notna().any()]
    completeness_str = f"{len(canonical_observed)}/6 core cols"

has_views = "views" in df.columns and df["views"].notna().any()
views_str = f"{int(df['views'].dropna().sum()):,}" if has_views else "N/A (Platform)"

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    render_metric_card("Total Snapshots (History)", f"{total_snapshots:,}", delta=f"{unique_posts_count:,} unique posts")
with kpi_col2:
    render_metric_card("Active Platforms", ", ".join(platforms).title())
with kpi_col3:
    render_metric_card("Applicable Completeness", completeness_str)
with kpi_col4:
    render_metric_card("Cumulative Video Views", views_str)

st.markdown("---")

# Filters & Historical Snapshot Toggle
col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
with col_filter1:
    platform_options = ["All"] + sorted(list(df["platform"].dropna().unique())) if "platform" in df.columns else ["All"]
    selected_platform = st.selectbox("Filter Platform:", platform_options)
with col_filter2:
    topic_options = ["All"] + sorted(list(df["topic"].dropna().unique())) if "topic" in df.columns else ["All"]
    selected_topic = st.selectbox("Filter Struggle Topic:", topic_options)
with col_filter3:
    view_mode = st.radio("Snapshot View Mode:", ["Latest Observations Only", "All Historical Snapshots"], horizontal=True)

active_df = latest_posts_df if view_mode == "Latest Observations Only" else df

filtered_df = active_df.copy()
if selected_platform != "All" and "platform" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["platform"] == selected_platform]
if selected_topic != "All" and "topic" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["topic"] == selected_topic]

# Unified Tabular Explorer
st.subheader("Unified Dataset Records")
st.markdown("Preserving missing-value semantics: unavailable metrics display as `<NA>` without zero-imputation.")

display_columns = [
    c for c in [
        "platform", "post_id", "timestamp", "collection_timestamp", "content_type",
        "views", "impressions", "reach", "likes", "comments_count",
        "shares", "saves", "collect_count", "retention_rate", "high_intent_actions",
        "high_intent_completeness", "save_to_share_ratio", "save_share_ratio_status"
    ] if c in filtered_df.columns
]
max_table_rows = 500
if len(filtered_df) > max_table_rows:
    st.caption(f"Showing preview of first {max_table_rows:,} of {len(filtered_df):,} filtered records.")
st.dataframe(filtered_df[display_columns].head(max_table_rows), width="stretch")

# Visualizations (sample deterministically for huge corpora so browser SVG stays responsive)
plot_df = filtered_df if len(filtered_df) <= 5000 else filtered_df.sample(n=5000, random_state=42)
if len(filtered_df) > 5000:
    st.caption(f"Charts render a deterministic 5,000-post visual sample out of {len(filtered_df):,} filtered rows; all KPIs above use the complete corpus.")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Views vs. Impressions by Content Type")
    chart_df = plot_df.dropna(subset=["content_type"]).copy() if "content_type" in plot_df.columns else pd.DataFrame()
    if (
        not chart_df.empty
        and "views" in chart_df.columns
        and "impressions" in chart_df.columns
        and chart_df["views"].notna().any()
        and chart_df["impressions"].notna().any()
    ):
        fig_views = px.scatter(
            chart_df,
            x="impressions",
            y="views",
            color="content_type",
            hover_data=[c for c in ["post_id", "platform"] if c in chart_df.columns],
            title="Video Plays (Views) vs. Screen Displays (Impressions)"
        )
        st.plotly_chart(fig_views, width="stretch")
    else:
        st.info("Views + Impressions pair not jointly observed in this platform/corpus (no synthetic imputation applied).")

with col_chart2:
    st.subheader("High-Intent Actions vs. Passive Likes")
    has_hi_pair = (
        "high_intent_actions" in plot_df.columns
        and "likes" in plot_df.columns
        and plot_df["high_intent_actions"].notna().any()
        and plot_df["likes"].notna().any()
    )
    has_proxy_pair = (
        "shares" in plot_df.columns
        and "likes" in plot_df.columns
        and plot_df["shares"].notna().any()
        and plot_df["likes"].notna().any()
    )
    if has_hi_pair:
        fig_intent = px.scatter(
            plot_df.dropna(subset=["high_intent_actions", "likes"]),
            x="likes",
            y="high_intent_actions",
            color="high_intent_completeness" if "high_intent_completeness" in plot_df.columns else None,
            hover_data=[c for c in ["post_id", "platform"] if c in plot_df.columns],
            title="High-Intent Actions (Saves+Shares) vs Likes by Completeness"
        )
        st.plotly_chart(fig_intent, width="stretch")
    elif has_proxy_pair:
        y_col = "collect_count" if ("collect_count" in plot_df.columns and plot_df["collect_count"].notna().any()) else "shares"
        y_title = "Collect Count Proxy (TikTok Bookmark)" if y_col == "collect_count" else "Shares"
        fig_proxy = px.scatter(
            plot_df.dropna(subset=[y_col, "likes"]),
            x="likes",
            y=y_col,
            color="platform" if "platform" in plot_df.columns else None,
            hover_data=[c for c in ["post_id"] if c in plot_df.columns],
            title=f"{y_title} vs. Passive Likes",
        )
        st.plotly_chart(fig_proxy, width="stretch")
    else:
        st.info("High-intent metrics not available for current slice.")
