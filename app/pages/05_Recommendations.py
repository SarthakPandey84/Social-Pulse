"""Page 5: Engagement Optimization Recommender.

Provides evidence-backed, explainable content strategy recommendations:
1. Translates descriptive metrics, virality models, NLP sentiment, and A/B test outcomes into actionable suggestions.
2. Ranks recommendations deterministically by evidence strength, potential impact, and sample adequacy.
3. Classifies confidence into High (Randomized Trials), Medium (Historical/Predictive), and Low (Observational/Heuristic).
4. Highlights conflicting evidence and proposes controlled A/B experiments for uncertain areas.
"""

from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.components.ui_widgets import render_header, render_metric_card
from src.recommender.engine import OptimizationRecommender
from src.recommender.schemas import ConfidenceLevel, RecommendationCategory, EvidenceSource

render_header(
    title="Module 5: Engagement Optimization Recommender",
    subtitle="Evidence-based prescriptive intelligence for strategic content creation"
)

# Provenance & Pipeline Demonstration Notice
st.info(
    "ℹ️ **Dataset Provenance & Demonstration Notice:** This recommendation engine operates on **synthetic benchmark fixtures** "
    "from Modules 1–4 to demonstrate evidence-based prioritization, confidence classification, and experimental proposal generation. "
    "Observational correlations are strictly separated from experimental evidence and are never presented as proven causal relationships."
)

# 1. Load Multi-Module Datasets
posts_path = root_dir / "data" / "processed" / "processed_posts.parquet"
comments_path = root_dir / "data" / "processed" / "processed_comments.parquet"
exp_path = root_dir / "data" / "sample" / "sample_experiments.csv"
obs_path = root_dir / "data" / "processed" / "processed_ab_tests.parquet"

posts_df = pd.read_parquet(posts_path) if posts_path.exists() else None
comments_df = pd.read_parquet(comments_path) if comments_path.exists() else None
exp_df = pd.read_csv(exp_path) if exp_path.exists() else None
obs_df = pd.read_parquet(obs_path) if obs_path.exists() else None

if posts_df is None and comments_df is None and exp_df is None:
    st.warning("⚠️ No analytical datasets found. Please run the data ingestion and preprocessing pipelines first.")
    st.stop()

# Initialize Engine
recommender = OptimizationRecommender(
    posts_df=posts_df,
    comments_df=comments_df,
    experiments_df=exp_df,
    observations_df=obs_df
)

# 2. Sidebar / Top Controls
st.subheader("⚙️ Filter Strategy Recommendations")
f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    platform_choice = st.selectbox(
        "Target Platform:",
        ["All Platforms", "Instagram", "YouTube"],
        index=0
    )
    selected_platform = None if platform_choice == "All Platforms" else platform_choice.lower()

with f_col2:
    category_options = {
        "All Categories": None,
        "Format & Hook Structure": RecommendationCategory.CONTENT_FORMAT_STRUCTURE,
        "Caption & Storytelling": RecommendationCategory.CAPTION_DEPTH_STYLE,
        "Topic Selection": RecommendationCategory.TOPIC_SELECTION,
        "Audience Relatability": RecommendationCategory.AUDIENCE_RELATABILITY,
        "Posting Timing": RecommendationCategory.POSTING_TIMING,
        "A/B Experimentation Agenda": RecommendationCategory.EXPERIMENTATION_AGENDA
    }
    category_choice = st.selectbox("Recommendation Category:", list(category_options.keys()), index=0)
    selected_category = category_options[category_choice]

with f_col3:
    confidence_choice = st.selectbox(
        "Minimum Evidence Level:",
        ["All Levels (High, Med, Low)", "Medium & High Evidence", "High Evidence Only (Randomized Trials)"],
        index=0
    )
    if "High Evidence Only" in confidence_choice:
        selected_min_conf = ConfidenceLevel.HIGH
    elif "Medium" in confidence_choice:
        selected_min_conf = ConfidenceLevel.MEDIUM
    else:
        selected_min_conf = None

# Generate Recommendations Report
report = recommender.generate_recommendations(
    platform=selected_platform,
    category=selected_category,
    min_confidence=selected_min_conf,
    max_recommendations=20
)

# 3. Executive KPI Overview
st.markdown("### 📊 Recommendation Overview & Evidence Breakdown")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    render_metric_card("Total Suggestions", str(report.total_recommendations))
with kpi2:
    render_metric_card("High Confidence (RCTs)", str(report.high_confidence_count))
with kpi3:
    render_metric_card("Medium Confidence", str(report.medium_confidence_count))
with kpi4:
    render_metric_card("Exploratory / Low", str(report.low_confidence_count))

# Evidence Hierarchy Card
st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px; margin-top: 10px; margin-bottom: 20px; font-size: 13px;">
        <b>EVIDENCE HIERARCHY & METHODOLOGICAL DISCLOSURES:</b><br>
        <span style="color: #059669; font-weight: 700;">🟢 HIGH CONFIDENCE (Demonstration RCT):</span> Supported by adequately powered randomized A/B trials (α = 0.05, power ≥ 80%, Holm-adjusted p < 0.05). Evaluated at independent post units (which addresses intra-post exposure dependence, though across-post creator/campaign clustering remains uncorrected).<br>
        <span style="color: #D97706; font-weight: 700;">🟡 MEDIUM CONFIDENCE:</span> Supported by controlled observational comparisons with matched cohorts or temporally validated predictive models. Not assigned merely for reaching 15 observational posts.<br>
        <span style="color: #475569; font-weight: 700;">⚪ LOW / EXPLORATORY:</span> Observational associations, exploratory NLP patterns, or proposed A/B experiments. Observational correlation is NOT causal proof.
    </div>
    """,
    unsafe_allow_html=True
)

if report.insufficient_data_warnings:
    for w in report.insufficient_data_warnings:
        st.caption(f"⚠️ {w}")

# 4. Conflicting Signals Section
if report.conflicting_signals:
    with st.expander("⚖️ Conflicting Evidence & Analytical Resolution (Click to expand)", expanded=True):
        for conf in report.conflicting_signals:
            st.markdown(f"#### Contradiction: {conf['area']}")
            st.markdown(f"- **Observational Finding:** {conf['observational_signal']}")
            st.markdown(f"- **Experimental Finding:** {conf['experimental_evidence']}")
            st.info(f"💡 **Recommender Resolution:** {conf['resolution']}")
            st.caption(f"Impact on Strategy: {conf['confidence_impact']}")

# 5. Ranked Recommendations Display
st.markdown("### 🎯 Prescribed Content Recommendations (Ranked by Evidence & Impact)")
st.caption("ℹ️ *Notice: Priority Score is a transparent deterministic ranking index for decision prioritization and does NOT represent a calibrated probability of virality or creator success.*")

if not report.recommendations:
    st.info("ℹ️ No recommendations match the selected filters. Try broadening your platform or confidence filter.")
else:
    for idx, rec in enumerate(report.recommendations, start=1):
        # Color coding by confidence
        if rec.confidence_level == ConfidenceLevel.HIGH:
            border_color = "#10B981"
            conf_badge = '<span style="background-color: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">🟢 HIGH EVIDENCE (DEMO RCT)</span>'
        elif rec.confidence_level == ConfidenceLevel.MEDIUM:
            border_color = "#F59E0B"
            conf_badge = '<span style="background-color: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">🟡 MEDIUM EVIDENCE</span>'
        else:
            border_color = "#94A3B8"
            conf_badge = '<span style="background-color: #F1F5F9; color: #475569; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">⚪ EXPLORATORY / LOW</span>'

        src_label = rec.evidence_source.value.replace("_", " ").title() if isinstance(rec.evidence_source, EvidenceSource) else str(rec.evidence_source)
        plat_label = rec.relevant_platform.title()

        ci_str = f"[{rec.confidence_interval[0]:,.2f}, {rec.confidence_interval[1]:,.2f}]" if rec.confidence_interval else "N/A"
        adj_p_str = f"{rec.p_value_adjusted:.2e}" if rec.p_value_adjusted is not None else "N/A"
        demo_pill = '<span style="background-color: #EDE9FE; color: #5B21B6; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-left: 6px;">SYNTHETIC DEMO</span>' if rec.is_synthetic_demonstration else ''

        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border-left: 6px solid {border_color}; border: 1px solid #E2E8F0; border-left-width: 6px; border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 16px; font-weight: 700; color: #0F172A;">
                        #{idx}. {rec.title} {demo_pill}
                    </div>
                    <div>{conf_badge}</div>
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
                    Category: <b>{rec.category.value if isinstance(rec.category, RecommendationCategory) else rec.category}</b> | 
                    Platform: <b>{plat_label}</b> | 
                    Source: <b>{src_label}</b> | 
                    Priority Score: <b>{rec.priority_score:.4f}</b>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px; margin-top: 10px; font-size: 14px;">
                    <b>Suggested Action:</b><br>{rec.suggested_action}
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #334155;">
                    <b>Reasoning:</b> {rec.reasoning}
                </div>
                <div style="margin-top: 6px; font-size: 13px; color: #1E293B;">
                    <b>Supporting Evidence:</b> {rec.supporting_evidence}
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #0284C7; background-color: #F0F9FF; padding: 6px 10px; border-radius: 4px;">
                    <b>Statistical Metrics:</b> Sample Size = <b>N={rec.sample_size}</b> | 
                    Unit = <b>{rec.unit_of_analysis}</b> | 
                    Holm Adj. p = <b>{adj_p_str}</b> | 
                    95% CI = <b>{ci_str}</b>
                </div>
                <div style="margin-top: 6px; font-size: 13px; color: #0369A1;">
                    <b>Expected Impact:</b> <b>{rec.expected_impact_direction}</b>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #B45309; background-color: #FFFBEB; padding: 8px; border-radius: 4px;">
                    <b>Limitations & Caveats:</b> {rec.limitations_and_caveats}
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #047857;">
                    <b>Recommended Validation:</b> {rec.suggested_validation_method}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(f"🔍 Priority Score Breakdown for #{idx} ({rec.recommendation_id})", expanded=False):
            bd = rec.scoring_breakdown
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                st.caption(f"• Evidence Weight (30%): **{bd.get('evidence_score', 0):.2f}**")
                st.caption(f"• Impact Weight (20%): **{bd.get('impact_score', 0):.2f}**")
            with b_col2:
                st.caption(f"• Sample & Uncertainty (15%): **{bd.get('sample_score', 0):.2f}**")
                st.caption(f"• Platform Relevance (15%): **{bd.get('relevance_score', 0):.2f}**")
            with b_col3:
                st.caption(f"• Evidence Recency (10%): **{bd.get('recency_score', 0):.2f}**")
                st.caption(f"• Data Quality (10%): **{bd.get('quality_score', 0):.2f}**")
                st.caption(f"• Conflict Penalty: **{bd.get('conflict_penalty', 1.0):.2f}x**")
            st.caption(f"**Deterministic Priority Score:** `{rec.priority_score:.4f}` *(Decision ranking index only; not a success probability)*")
