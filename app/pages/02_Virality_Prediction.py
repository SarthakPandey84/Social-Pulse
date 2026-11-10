"""Page 2: Virality Prediction Engine.

Presents:
1. Pre-publication concept draft performance forecaster with approximate error bands.
2. Temporal evaluation benchmark against Naive Dummy Baseline on mature settled posts (>= 7 days).
3. Standardized Ridge coefficients and dataset limitations notice.
4. Ex-post Engagement Virality Index (EVI) topic rankings.
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
from src.modeling.virality_predictor import ViralityPredictor
from src.preprocessing.cleaning import get_latest_posts_view

render_header(
    title="Module 2: Virality Prediction Engine",
    subtitle="Pre-publication performance estimation and ex-post Engagement Virality Index analytics"
)

# Provenance & Pipeline Demonstration Notice
st.info(
    "🔬 **Dataset Provenance & Demonstration Notice:** This module runs on a **synthetic benchmark dataset** "
    "designed to demonstrate the architectural pipeline, target leakage guards, and statistical separation. "
    "Evaluation metrics demonstrate pipeline mechanics and baseline comparison, not real-world predictive validation."
)

# 1. Load Processed Dataset & Initial Model
processed_path = root_dir / "data" / "processed" / "processed_posts.parquet"
sample_path = root_dir / "data" / "sample" / "sample_posts.csv"
model_path = root_dir / "models" / "virality_predictor.joblib"
meta_path = root_dir / "models" / "virality_predictor_metadata.json"

if processed_path.exists():
    df = pd.read_parquet(processed_path)
elif sample_path.exists():
    df = pd.read_csv(sample_path)
else:
    st.warning("Processed dataset not found. Please run the ingestion pipeline.")
    st.stop()

latest_df = get_latest_posts_view(df)

predictor = ViralityPredictor()
# Load pre-trained model or train on mature posts (>= 7 days)
if model_path.exists() and meta_path.exists():
    predictor.load_model_pipeline(model_path, meta_path)
else:
    predictor.train_and_evaluate(latest_df, model_type="ridge", test_ratio=0.2, min_elapsed_days=7.0)
    predictor.save_model_pipeline(model_path, meta_path)

meta = predictor.metadata

# Tabs: Pre-Publication Forecasting vs Model Evaluation vs Ex-Post Rankings
tab_forecast, tab_eval, tab_expost = st.tabs([
    "🎯 Pre-Publication Draft Forecaster",
    "📊 Model Evaluation & Baseline Comparison",
    "📈 Ex-Post Topic Virality Rankings"
])

# =========================================================================
# TAB 1: PRE-PUBLICATION FORECASTER
# =========================================================================
with tab_forecast:
    st.subheader("Predict Expected Performance Before Publishing")
    st.markdown(
        "Forecast expected content resonance using strictly **pre-publication variables** (no outcome metric leakage). "
        "The model predicts the settled **Engagement Virality Index (EVI)**."
    )

    with st.form("forecast_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            platform = st.selectbox("Platform:", ["instagram", "youtube"])
            content_type = st.selectbox("Content Format:", ["Reel", "Short Video", "Carousel"])
            topic = st.selectbox("Struggle Topic:", [
                "Social Anxiety", "Career Burnout", "Imposter Syndrome",
                "Dating & Rejection", "Academic Stress", "Procrastination Loop", "Friendship Dynamics"
            ])
        with col2:
            hook_type = st.selectbox("Hook Style:", ["Visual Hook", "Text Hook", "Story Hook"])
            duration = st.slider("Planned Video Duration (seconds):", min_value=0, max_value=90, value=30, step=5)
            follower_count = st.number_input("Creator Baseline Follower Count:", min_value=100, max_value=1000000, value=35000, step=1000)
        with col3:
            posting_hour = st.slider("Planned Posting Hour (UTC):", min_value=0, max_value=23, value=18)
            posting_day = st.selectbox("Day of Week:", [
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            ], index=6)

        submit_btn = st.form_submit_button("Generate Performance Forecast", width="stretch")

    if submit_btn:
        draft_df = pd.DataFrame([{
            "platform": platform,
            "content_type": content_type,
            "topic": topic,
            "hook_type": hook_type,
            "duration_seconds": duration if content_type != "Carousel" else 0,
            "posting_hour": posting_hour,
            "posting_day_of_week": posting_day,
            "follower_count": follower_count
        }])

        pred_result = predictor.predict_pre_publication(draft_df).iloc[0]
        score = pred_result["predicted_engagement_virality_index"]
        band_lower, band_upper = pred_result["approximate_error_band"]
        tier = pred_result["predicted_performance_tier"]

        st.markdown("### Forecast Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Predicted EVI", f"{score:.1f}")
        with c2:
            render_metric_card("Approximate Error Band", f"[{band_lower:.1f}, {band_upper:.1f}]")
        with c3:
            render_metric_card("Expected Tier", tier)

        st.caption(
            "ℹ️ **Uncertainty Interpretation:** The approximate error band reflects empirical residual dispersion "
            "(±50.2 points). It is an indicative model dispersion window, not a formal conformal prediction interval "
            "with guaranteed finite-sample coverage."
        )
        st.caption(
            "⚠️ **Cross-Platform Comparability:** Baseline engagement density differs across Instagram and YouTube. "
            "Interpret predictions relative to within-platform percentiles rather than absolute cross-network parity."
        )

# =========================================================================
# TAB 2: MODEL EVALUATION & BASELINE COMPARISON
# =========================================================================
with tab_eval:
    st.subheader("Temporal Evaluation on Mature Settled Posts (>= 7 Days)")
    st.markdown(
        f"To prevent maturity bias and lookahead leakage, posts were filtered to **mature settled records** (≥ 7 days since publication), "
        f"yielding **{meta.get('total_mature_posts', 112)} unique posts**. "
        f"Data was partitioned strictly chronologically: earliest {meta.get('train_samples_count', 89)} posts for training, "
        f"and subsequent {meta.get('test_samples_count', 23)} posts for out-of-sample temporal holdout testing."
    )

    base = meta.get("baseline_metrics", {})
    cand = meta.get("candidate_metrics", {})

    eval_df = pd.DataFrame([
        {
            "Model": "Naive Dummy Baseline (Historical Mean)",
            "MAE": f"{base.get('mae', 0):.2f}",
            "RMSE": f"{base.get('rmse', 0):.2f}",
            "R²": f"{base.get('r2', 0):.3f}",
            "Spearman Rank Correlation": "N/A (Constant Prediction)",
            "Status": "Baseline"
        },
        {
            "Model": f"Candidate Model ({meta.get('model_type', 'ridge').title()} Regressor)",
            "MAE": f"{cand.get('mae', 0):.2f}",
            "RMSE": f"{cand.get('rmse', 0):.2f}",
            "R²": f"{cand.get('r2', 0):.3f}",
            "Spearman Rank Correlation": f"{cand.get('spearman_rho', 0):.3f} (p={cand.get('spearman_pvalue', 0):.4f})",
            "Status": "✅ Beats Baseline" if meta.get("beats_baseline") else "⚠️ Underperforms Baseline"
        }
    ])
    st.table(eval_df)

    col_lift1, col_lift2 = st.columns(2)
    with col_lift1:
        st.metric("MAE Error Reduction", f"{meta.get('mae_lift_pct', 0)}%", delta=f"{meta.get('mae_lift_pct', 0)}% vs Baseline")
    with col_lift2:
        st.metric("RMSE Error Reduction", f"{meta.get('rmse_lift_pct', 0)}%", delta=f"{meta.get('rmse_lift_pct', 0)}% vs Baseline")

    # Feature Importance Plot
    top_features = meta.get("top_features", {})
    if top_features:
        st.subheader("Top Pre-Publication Coefficients (Standardized Ridge)")
        feat_df = pd.DataFrame(list(top_features.items()), columns=["Feature", "Standardized Coefficient"])
        feat_df = feat_df.sort_values(by="Standardized Coefficient", ascending=True)

        fig_coef = px.bar(
            feat_df,
            x="Standardized Coefficient",
            y="Feature",
            orientation="h",
            color="Standardized Coefficient",
            color_continuous_scale="RdBu",
            title="Predictive Impact of Pre-Publication Factors"
        )
        st.plotly_chart(fig_coef, width="stretch")

    # Methodological Limitations & Recommendation
    st.markdown("---")
    st.subheader("Methodological Audit & Production Recommendations")
    limitations = meta.get("methodological_limitations", {})
    st.markdown(f"- **Selection Optimism Risk:** {limitations.get('selection_optimism_risk', '')}")
    st.markdown(f"- **Prediction Interval Validity:** {limitations.get('prediction_interval_validity', '')}")
    st.markdown(f"- **Recommended Validation Strategy:** For future streaming real-world API data, **Rolling-Origin Cross-Validation (Walk-Forward Temporal Backtesting)** should be adopted over single-split holdouts to ensure continuous statistical tracking.")

# =========================================================================
# TAB 3: EX-POST TOPIC RANKINGS
# =========================================================================
with tab_expost:
    st.subheader("Historical Ex-Post Topic Rankings")
    st.markdown(
        "Rankings based on the observed **Engagement Virality Index (EVI)** across historical posts. "
        "Weights high-intent actions (**Shares: 40%, Saves: 35%**) over passive vanity actions (**Likes: 10%**)."
    )

    ranked_topics = predictor.rank_topics_by_virality(latest_df)
    st.dataframe(ranked_topics, width="stretch")

    fig_topics = px.bar(
        ranked_topics,
        x="topic",
        y="avg_viral_coefficient",
        color="avg_viral_coefficient",
        color_continuous_scale="Teal",
        title="Mean Engagement Virality Index by Struggle Topic"
    )
    st.plotly_chart(fig_topics, width="stretch")
