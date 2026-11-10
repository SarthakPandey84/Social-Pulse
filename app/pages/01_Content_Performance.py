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

from app.components.ui_widgets import render_header, render_metric_card
from src.data_ingestion.schemas import UNIFIED_POST_FIELDS
from src.preprocessing.cleaning import get_latest_posts_view

render_header(
    title="Module 1: Content Performance Tracker",
    subtitle="Unified cross-platform ingestion, schema validation, and feature-engineered engagement signals"
)

# Load Module 1 Processed dataset or fallback to sample
processed_path = root_dir / "data" / "processed" / "processed_posts.parquet"
sample_path = root_dir / "data" / "sample" / "sample_posts.csv"

if processed_path.exists():
    df = pd.read_parquet(processed_path)
    data_source_badge = "✅ Source: Processed Data Pipeline (data/processed/)"
elif sample_path.exists():
    df = pd.read_csv(sample_path)
    data_source_badge = "⚠️ Source: Sample Offline Benchmark (data/sample/)"
else:
    st.warning("No ingested datasets found. Please run the ingestion pipeline.")
    st.stop()

st.caption(data_source_badge)

# Ingestion Summary & Quality KPIs
total_snapshots = len(df)
latest_posts_df = get_latest_posts_view(df)
unique_posts_count = len(latest_posts_df)

platforms = list(df["platform"].dropna().unique()) if "platform" in df.columns else ["N/A"]
avg_completeness = round(float(df["data_completeness_pct"].mean()), 1) if "data_completeness_pct" in df.columns else 100.0
total_views = int(df["views"].dropna().sum()) if "views" in df.columns else 0

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    render_metric_card("Total Snapshots (History)", f"{total_snapshots:,}", delta=f"{unique_posts_count} unique posts")
with kpi_col2:
    render_metric_card("Active Platforms", ", ".join(platforms).title())
with kpi_col3:
    render_metric_card("Applicable Completeness", f"{avg_completeness}%")
with kpi_col4:
    render_metric_card("Cumulative Video Views", f"{total_views:,}")

st.markdown("---")

# Filters & Historical Snapshot Toggle
col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
with col_filter1:
    platform_options = ["All"] + list(df["platform"].dropna().unique()) if "platform" in df.columns else ["All"]
    selected_platform = st.selectbox("Filter Platform:", platform_options)
with col_filter2:
    topic_options = ["All"] + list(df["topic"].dropna().unique()) if "topic" in df.columns else ["All"]
    selected_topic = st.selectbox("Filter Struggle Topic:", topic_options)
with col_filter3:
    view_mode = st.radio("Snapshot View Mode:", ["Latest Observations Only", "All Historical Snapshots"], horizontal=True)

active_df = latest_posts_df if view_mode == "Latest Observations Only" else df

filtered_df = active_df.copy()
if selected_platform != "All":
    filtered_df = filtered_df[filtered_df["platform"] == selected_platform]
if selected_topic != "All":
    filtered_df = filtered_df[filtered_df["topic"] == selected_topic]

# Unified Tabular Explorer
st.subheader("Unified Dataset Records")
st.markdown("Preserving missing-value semantics: unavailable metrics display as `<NA>` without zero-imputation.")

display_columns = [
    c for c in [
        "platform", "post_id", "timestamp", "collection_timestamp", "content_type",
        "views", "impressions", "reach", "likes", "comments_count",
        "shares", "saves", "retention_rate", "high_intent_actions",
        "high_intent_completeness", "save_to_share_ratio", "save_share_ratio_status"
    ] if c in filtered_df.columns
]
st.dataframe(filtered_df[display_columns], width="stretch")

# Visualizations
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Views vs. Impressions by Content Type")
    chart_df = filtered_df.dropna(subset=["content_type"]).copy()
    if "views" in chart_df.columns and "impressions" in chart_df.columns:
        fig_views = px.scatter(
            chart_df,
            x="impressions",
            y="views",
            color="content_type",
            hover_data=["post_id", "platform"],
            title="Video Plays (Views) vs. Screen Displays (Impressions)"
        )
        st.plotly_chart(fig_views, width="stretch")
    else:
        st.info("Views and impressions data not available.")

with col_chart2:
    st.subheader("High-Intent Actions vs. Passive Likes")
    if "high_intent_actions" in filtered_df.columns and "likes" in filtered_df.columns:
        fig_intent = px.scatter(
            filtered_df.dropna(subset=["high_intent_actions", "likes"]),
            x="likes",
            y="high_intent_actions",
            color="high_intent_completeness",
            hover_data=["post_id"],
            title="High-Intent Actions (Saves+Shares) vs Likes by Completeness"
        )
        st.plotly_chart(fig_intent, width="stretch")
    else:
        st.info("High-intent metrics not available for current slice.")
