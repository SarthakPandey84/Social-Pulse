"""SocialPulse — Growth Intelligence Hub (Home View)."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.data_loader import (
    DatasetLoadError,
    SYNTHETIC_DEMO_DATASET_ID,
    list_available_datasets,
    load_home_summary_kpis,
    load_selected_comments_dataset,
    load_selected_posts_dataset,
)
from app.components.ui_widgets import (
    render_dataset_selector,
    render_footer,
    render_header,
    render_info_banner,
    render_metric_card,
    render_provenance_banner,
    render_section_header,
    render_status_badge,
)

st.set_page_config(
    page_title="SocialPulse Hub — Social Engagement Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)


all_datasets = list_available_datasets(root_dir=root_dir)
all_dataset_ids = [d["dataset_id"] for d in all_datasets]
selected_dataset_id = render_dataset_selector(
    all_datasets,
    all_dataset_ids=all_dataset_ids,
    key="sp_dataset_selector_home",
    location="sidebar",
)

# Load KPIs / metadata for active dataset (fail-loud on corrupt external selection)
data_path = root_dir / "data" / "sample" / "sample_posts.csv"
active_meta: Dict[str, Any] = next(
    (d for d in all_datasets if d["dataset_id"] == selected_dataset_id),
    {"dataset_id": selected_dataset_id, "tables": ["posts"]},
)
kpis: Optional[Dict[str, Any]] = None
comments_summary: Optional[Dict[str, Any]] = None

if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    kpis = load_home_summary_kpis(data_path)
else:
    try:
        if "posts" in active_meta.get("tables", []) or active_meta.get("meta_error"):
            ext_posts, active_meta = load_selected_posts_dataset(
                selected_dataset_id,
                root_dir=root_dir,
                columns=["post_id", "platform", "views", "likes", "comments_count", "shares", "saves", "collect_count", "retention_rate", "content_type", "provenance"],
            )
            tot_posts = int(ext_posts["post_id"].nunique()) if "post_id" in ext_posts.columns else int(len(ext_posts))
            has_views = "views" in ext_posts.columns and ext_posts["views"].notna().any()
            tot_views = int(ext_posts["views"].dropna().sum()) if has_views else None
            has_shares = "shares" in ext_posts.columns and ext_posts["shares"].notna().any()
            tot_shares = int(ext_posts["shares"].dropna().sum()) if has_shares else 0
            has_saves = "saves" in ext_posts.columns and ext_posts["saves"].notna().any()
            tot_saves = int(ext_posts["saves"].dropna().sum()) if has_saves else None
            has_collect = "collect_count" in ext_posts.columns and ext_posts["collect_count"].notna().any()
            tot_collects = int(ext_posts["collect_count"].dropna().sum()) if has_collect else None
            has_ret = "retention_rate" in ext_posts.columns and ext_posts["retention_rate"].notna().any()
            if has_ret:
                _mean_ret = float(ext_posts["retention_rate"].dropna().mean())
                avg_ret = round(_mean_ret * 100.0 if _mean_ret <= 1.0 else _mean_ret, 1)
            else:
                avg_ret = None
            kpis = {
                "total_posts": tot_posts,
                "total_views": tot_views,
                "total_likes": int(ext_posts["likes"].dropna().sum()) if "likes" in ext_posts.columns and ext_posts["likes"].notna().any() else None,
                "total_shares": tot_shares if has_shares else None,
                "total_saves": tot_saves,
                "total_collects_proxy": tot_collects,
                "avg_retention_pct": avg_ret,
                "overall_save_to_share_ratio": (
                    round(tot_saves / max(tot_shares, 1), 2) if (has_saves and has_shares and tot_saves is not None) else None
                ),
            }
        else:
            ext_comments, active_meta = load_selected_comments_dataset(selected_dataset_id, root_dir=root_dir)
            comments_summary = {
                "total_comments": int(len(ext_comments)),
                "platforms": ", ".join(str(p).title() for p in active_meta.get("platforms") or ["Instagram"]),
            }
    except DatasetLoadError as exc:
        render_header(
            title="SocialPulse",
            subtitle="An Intelligent Social Media Analytics & Engagement Optimization Platform.",
        )
        st.error(f"❌ Dataset Load Error: {exc}")
        st.stop()

# Hero Header
render_header(
    title="SocialPulse",
    subtitle="An Intelligent Social Media Analytics & Engagement Optimization Platform decoding high-intent audience resonance through statistical modeling, orthogonal NLP, and walk-forward forecasting.",
)

if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    render_info_banner(
        title="Benchmark Environment & Methodological Transparency",
        body=(
            "SocialPulse replaces vanity-metric guesswork by prioritizing high-intent engagement (Shares & Saves) over passive likes. "
            "Metrics below summarize the included offline benchmark dataset (data/sample/sample_posts.csv) to demonstrate pipeline mechanics cleanly without fabricating live social API connections."
        ),
        tone="teal",
    )
else:
    render_provenance_banner(active_meta)

# Section 1: Active Corpus KPIs
render_section_header(
    title="Benchmark Dataset Overview" if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID else f"Active Corpus Overview · {active_meta.get('display_name', selected_dataset_id)}",
    subtitle="Key engagement indicators computed directly from the active SocialPulse corpus",
    badge="Offline Benchmark Corpus" if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID else str(active_meta.get("provenance", "external_data")),
    badge_tone="accent" if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID else "teal",
)

if kpis:
    platforms_label = " + ".join(str(p).title() for p in (active_meta.get("platforms") or ["Instagram", "YouTube"]))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Analyzed Posts", f"{kpis['total_posts']:,}", delta=platforms_label)
    with col2:
        if kpis.get("total_views") is not None:
            render_metric_card("Cumulative Reach / Views", f"{kpis['total_views']:,}", delta=platforms_label)
        elif kpis.get("total_likes") is not None:
            render_metric_card("Cumulative Observed Likes", f"{kpis['total_likes']:,}", delta="Views unavailable (public scrape)")
        else:
            render_metric_card("Cumulative Reach / Views", "N/A", delta="Not exposed in corpus")
    with col3:
        if kpis.get("avg_retention_pct") is not None:
            render_metric_card("Average Retention Rate", f"{kpis['avg_retention_pct']}%", delta="Video completion avg")
        elif kpis.get("total_shares") is not None:
            render_metric_card("Cumulative Shares", f"{kpis['total_shares']:,}", delta="Retention N/A in public scrape")
        else:
            render_metric_card("Average Retention Rate", "N/A", delta="Unavailable in public scrape")
    with col4:
        if kpis.get("overall_save_to_share_ratio") is not None:
            render_metric_card("Save-to-Share Ratio", f"{kpis['overall_save_to_share_ratio']}x", delta="High-intent balance")
        elif kpis.get("total_collects_proxy") is not None:
            render_metric_card("Collect Count (Proxy)", f"{kpis['total_collects_proxy']:,}", delta="TikTok proxy (not IG saves)")
        else:
            render_metric_card("Save-to-Share Ratio", "N/A", delta="Saves N/A on platform")
elif comments_summary:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Analyzed Comments", f"{comments_summary['total_comments']:,}", delta="Module 3 NLP Corpus")
    with c2:
        render_metric_card("Platform", comments_summary["platforms"], delta="Comments-only dataset")
    with c3:
        render_metric_card("Post Views / Reach", "N/A", delta="Posts table absent")
    with c4:
        render_metric_card("Provenance", str(active_meta.get("provenance", "real_public_data")), delta=str(active_meta.get("license_name", "")))
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

footer_mode = (
    "Synthetic Benchmark Mode"
    if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID
    else f"Active Corpus: {active_meta.get('display_name', selected_dataset_id)} ({active_meta.get('provenance', 'external')})"
)
render_footer(f"SocialPulse v1.0 · The Data-Driven Social Engagement Initiative · {footer_mode}")
