"""SocialPulse — Growth Intelligence Hub (Home View)."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.ui_widgets import (
    render_footer,
    render_header,
    render_info_banner,
    render_metric_card,
    render_section_header,
    render_status_badge,
)
from src.analytics.engagement_metrics import EngagementAnalytics

st.set_page_config(
    page_title="SocialPulse Hub — Social Engagement Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_data(show_spinner=False)
def _load_home_summary_kpis(path_str: str, mtime_ns: int, file_size: int) -> Optional[Dict[str, Any]]:
    """Cache sample dataset KPI summary keyed by file path, mtime, and size."""
    df_posts = pd.read_csv(path_str)
    if df_posts.empty:
        return None
    return EngagementAnalytics.calculate_summary_kpis(df_posts)


# Load sample data KPIs
data_path = root_dir / "data" / "sample" / "sample_posts.csv"
if data_path.exists():
    data_stat = data_path.stat()
    kpis = _load_home_summary_kpis(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)
else:
    kpis = None

# Hero Header
render_header(
    title="SocialPulse",
    subtitle="An Intelligent Social Media Analytics & Engagement Optimization Platform decoding high-intent audience resonance through statistical modeling, orthogonal NLP, and walk-forward forecasting.",
)

render_info_banner(
    title="Benchmark Environment & Methodological Transparency",
    body=(
        "SocialPulse replaces vanity-metric guesswork by prioritizing high-intent engagement (Shares & Saves) over passive likes. "
        "Metrics below summarize the included offline benchmark dataset (data/sample/sample_posts.csv) to demonstrate pipeline mechanics cleanly without fabricating live social API connections."
    ),
    tone="teal",
)

# Section 1: Genuine Platform Benchmark KPIs
render_section_header(
    title="Benchmark Dataset Overview",
    subtitle="Key engagement indicators computed directly from the baseline benchmark post corpus",
    badge="Offline Benchmark Corpus",
    badge_tone="accent",
)

if kpis:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Analyzed Posts", f"{kpis['total_posts']:,}", delta="Multi-platform sample")
    with col2:
        render_metric_card("Cumulative Reach / Views", f"{kpis['total_views']:,}", delta="Instagram + YouTube")
    with col3:
        render_metric_card("Average Retention Rate", f"{kpis['avg_retention_pct']}%", delta="Video completion avg")
    with col4:
        render_metric_card("Save-to-Share Ratio", f"{kpis['overall_save_to_share_ratio']}x", delta="High-intent balance")
else:
    st.warning("Sample benchmark dataset not found at `data/sample/sample_posts.csv`.")

# Section 2: Interactive Modules Directory
render_section_header(
    title="Analytical Modules & Workspace",
    subtitle="Seven specialized engines spanning data ingestion, predictive modeling, NLP sentiment, experimentation, and forecasting",
    badge="7 Active Modules",
    badge_tone="teal",
)

nav_pages = st.session_state.get("_socialpulse_pages", {})

modules = [
    {
        "key": "m1",
        "code": "MODULE 01",
        "domain": "Data & Metrics",
        "tone": "accent",
        "is_teal": False,
        "title": "01. Content Performance Tracker",
        "desc": "Unified Instagram and YouTube schema ingestion, snapshot history deduplication, missing-value preservation, and high-intent signal profiling.",
        "tags": ["Schema Validation", "High-Intent KPIs", "Parquet Pipeline"],
        "path": "app/pages/01_Content_Performance.py",
    },
    {
        "key": "m2",
        "code": "MODULE 02",
        "domain": "Predictive ML",
        "tone": "teal",
        "is_teal": True,
        "title": "02. Virality Prediction Engine",
        "desc": "Pre-publication Engagement Virality Index (EVI) estimator built strictly on leak-free pre-publish features with temporal holdout comparison.",
        "tags": ["Ridge Regressor", "Leakage Guards", "Topic Ranking"],
        "path": "app/pages/02_Virality_Prediction.py",
    },
    {
        "key": "m3",
        "code": "MODULE 03",
        "domain": "Audience NLP",
        "tone": "accent",
        "is_teal": False,
        "title": "03. Audience Sentiment & Relatability",
        "desc": "Orthogonal NLP architecture separating emotional valence (Sentiment) from shared struggle resonance (Relatability) with language guards.",
        "tags": ["TF-IDF + Rules", "Language Safety", "Interactive Tester"],
        "path": "app/pages/03_Sentiment_Analysis.py",
    },
    {
        "key": "m4",
        "code": "MODULE 04",
        "domain": "Experimentation",
        "tone": "teal",
        "is_teal": True,
        "title": "04. A/B Testing Framework",
        "desc": "Hypothesis testing with cluster-robust post-level comparisons, Holm-Bonferroni & FDR corrections, confidence intervals, and power planning.",
        "tags": ["Welch / Fisher", "FWER & FDR", "Power Calculator"],
        "path": "app/pages/04_AB_Testing.py",
    },
    {
        "key": "m5",
        "code": "MODULE 05",
        "domain": "Prescriptive AI",
        "tone": "accent",
        "is_teal": False,
        "title": "05. Engagement Optimization Recommender",
        "desc": "Deterministic 7-component recommendation engine synthesizing Modules 1–4 into tiered strategy actions while surfacing evidence conflicts.",
        "tags": ["Evidence Hierarchy", "Conflict Detection", "Priority Index"],
        "path": "app/pages/05_Recommendations.py",
    },
    {
        "key": "m6",
        "code": "MODULE 06",
        "domain": "Unified BI",
        "tone": "teal",
        "is_teal": True,
        "title": "06. Growth Visualization Dashboard",
        "desc": "Cross-module intelligence workspace uniting KPIs, time-series trends, normalized platform metrics, heatmaps, experiments, and priorities.",
        "tags": ["Cross-Platform", "Schedule Matrix", "Forest Plot"],
        "path": "app/pages/06_Growth_Dashboard.py",
    },
    {
        "key": "m7",
        "code": "MODULE 07",
        "domain": "Forecasting",
        "tone": "accent",
        "is_teal": False,
        "title": "07. Trend Forecasting Engine",
        "desc": "Rolling-origin walk-forward forecasting for isolated publication cohorts and struggle keyword trajectories with horizon-specific uncertainty.",
        "tags": ["Walk-Forward CV", "MASE Benchmark", "Empirical Spreads"],
        "path": "app/pages/07_Trend_Forecasting.py",
    },
]

for row_start in range(0, len(modules), 2):
    row_items = modules[row_start : row_start + 2]
    cols = st.columns(2) if len(row_items) == 2 else [st.container()]
    for col, mod in zip(cols, row_items):
        with col:
            badge_left = render_status_badge(mod["code"], tone=mod["tone"])
            badge_right = render_status_badge(mod["domain"], tone="neutral")
            tags_html = "".join(f'<span class="sp-tag">{tag}</span>' for tag in mod["tags"])
            card_cls = "sp-module-card is-teal" if mod["is_teal"] else "sp-module-card"
            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div>
                        <div class="sp-module-head">
                            {badge_left}
                            {badge_right}
                        </div>
                        <div class="sp-module-title">{mod['title']}</div>
                        <p class="sp-module-desc">{mod['desc']}</p>
                    </div>
                    <div class="sp-module-tags">{tags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            page_target = nav_pages.get(mod["key"])
            if page_target is not None:
                st.page_link(page_target, label=f"Open {mod['title'].split('. ', 1)[1]} →", width="stretch")

# Section 3: How SocialPulse Works
render_section_header(
    title="How SocialPulse Works",
    subtitle="Core analytical principles shared across every dashboard view and modeling module",
)

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    st.markdown(
        """
        <div class="sp-pillar-card">
            <div class="sp-pillar-title">1 · Signal Integrity &amp; Platform Isolation</div>
            <div class="sp-pillar-body">Preserves native platform schema differences (no zero-imputation on unavailable YouTube saves/reach) and blocks invalid raw cross-platform volume pooling.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p_col2:
    st.markdown(
        """
        <div class="sp-pillar-card">
            <div class="sp-pillar-title">2 · Orthogonal NLP &amp; Causal Guardrails</div>
            <div class="sp-pillar-body">Separates negative emotion from authentic audience relatability and distinguishes randomized experimental lift from confounded historical correlations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p_col3:
    st.markdown(
        """
        <div class="sp-pillar-card">
            <div class="sp-pillar-title">3 · Walk-Forward &amp; Prescriptive Rigor</div>
            <div class="sp-pillar-body">Evaluates forecasters strictly out-of-sample across expanding origins via MASE and surfaces calibrated-honesty uncertainty caveats on every projection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_footer("SocialPulse v1.0 · The Data-Driven Social Engagement Initiative · Synthetic Benchmark Mode")
