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

from app.components.ui_widgets import (
    build_recommendation_detail_card_html,
    render_header,
    render_metric_card,
)
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
        src_label = rec.evidence_source.value.replace("_", " ").title() if isinstance(rec.evidence_source, EvidenceSource) else str(rec.evidence_source)
        plat_label = rec.relevant_platform.title()

        ci_str = f"[{rec.confidence_interval[0]:,.2f}, {rec.confidence_interval[1]:,.2f}]" if rec.confidence_interval else "N/A"
        adj_p_str = f"{rec.p_value_adjusted:.2e}" if rec.p_value_adjusted is not None else "N/A"
        conf_val = rec.confidence_level.value if isinstance(rec.confidence_level, ConfidenceLevel) else str(rec.confidence_level)
        cat_val = rec.category.value if isinstance(rec.category, RecommendationCategory) else str(rec.category)

        st.markdown(
            build_recommendation_detail_card_html(
                idx=idx,
                confidence_level=conf_val,
                is_synthetic_demonstration=bool(rec.is_synthetic_demonstration),
                title=rec.title,
                category=cat_val,
                platform_label=plat_label,
                source_label=src_label,
                priority_score=rec.priority_score,
                suggested_action=rec.suggested_action,
                reasoning=rec.reasoning,
                supporting_evidence=rec.supporting_evidence,
                sample_size=rec.sample_size,
                unit_of_analysis=rec.unit_of_analysis,
                adj_p_str=adj_p_str,
                ci_str=ci_str,
                expected_impact_direction=rec.expected_impact_direction,
                limitations_and_caveats=rec.limitations_and_caveats,
                suggested_validation_method=rec.suggested_validation_method,
            ),
            unsafe_allow_html=True,
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
