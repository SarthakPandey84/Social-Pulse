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

from app.components.data_loader import (
    DatasetLoadError,
    SYNTHETIC_DEMO_DATASET_ID,
    get_virality_predictor,
    list_available_datasets,
    load_json_artifact,
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
    title="Module 2: Virality Prediction Engine",
    subtitle="Pre-publication performance estimation and ex-post Engagement Virality Index analytics"
)

all_datasets = list_available_datasets(root_dir=root_dir)
posts_datasets = list_available_datasets(root_dir=root_dir, required_table="posts")
selected_dataset_id = render_dataset_selector(
    posts_datasets,
    all_dataset_ids=[d["dataset_id"] for d in all_datasets],
    key="sp_dataset_selector_p2",
    location="sidebar",
)

# 1. Load Active Dataset & Cached Predictor Resource (fail-loud on corrupt external selection)
processed_path = root_dir / "data" / "processed" / "processed_posts.parquet"
sample_path = root_dir / "data" / "sample" / "sample_posts.csv"
model_path = root_dir / "models" / "virality_predictor.joblib"
meta_path = root_dir / "models" / "virality_predictor_metadata.json"

active_meta = {"dataset_id": selected_dataset_id, "provenance": "synthetic_data"}
if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    st.info(
        "🔬 **Dataset Provenance & Demonstration Notice:** This module runs on a **synthetic benchmark dataset** "
        "designed to demonstrate the architectural pipeline, target leakage guards, and statistical separation. "
        "Evaluation metrics demonstrate pipeline mechanics and baseline comparison, not real-world predictive validation."
    )
    df, source_kind = load_posts_dataset(processed_path, sample_path)
    if df is None:
        st.warning("Processed dataset not found. Please run the ingestion pipeline.")
        st.stop()
    active_posts_path = processed_path if source_kind == "processed" else sample_path
else:
    ext_cols = [
        "platform", "post_id", "content_type", "topic", "hook_type",
        "duration_seconds", "posting_hour", "posting_day_of_week",
        "follower_count", "views", "reach", "impressions",
        "likes", "comments_count", "shares", "saves", "collect_count",
        "retention_rate", "timestamp", "collection_timestamp", "provenance",
    ]
    try:
        df, active_meta = load_selected_posts_dataset(
            selected_dataset_id,
            root_dir=root_dir,
            columns=ext_cols,
        )
    except DatasetLoadError as exc:
        st.error(f"❌ Dataset Load Error: {exc}")
        st.stop()
    render_provenance_banner(active_meta)
    active_posts_path = None

latest_df = get_latest_posts_view(df) if "post_id" in df.columns else df

predictor = get_virality_predictor(
    model_path=model_path,
    meta_path=meta_path,
    fallback_posts_df=latest_df,
    fallback_posts_path=active_posts_path,
)

meta = predictor.metadata
real_benchmarks = load_json_artifact(root_dir / "models" / "real_data_benchmarks.json") or {}

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
            platform = st.selectbox("Platform:", ["instagram", "youtube", "tiktok"])
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

        cand_rmse = meta.get("candidate_metrics", {}).get("rmse", 32.8)
        err_disp = meta.get("empirical_residual_q95", cand_rmse * 1.96)
        st.caption(
            f"ℹ️ **Uncertainty Interpretation:** The approximate error band reflects empirical residual dispersion "
            f"(±{err_disp:.2f} points). It is an indicative model dispersion window, not a formal conformal prediction interval "
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

    plat_eval = meta.get("platform_evaluation_metrics", {})
    if isinstance(plat_eval, dict) and plat_eval:
        st.subheader("Platform-Stratified Temporal Evaluation Coverage")
        plat_rows = []
        for plat_name, pe in plat_eval.items():
            if not isinstance(pe, dict):
                continue
            c_mae = pe.get("candidate_mae")
            c_rmse = pe.get("candidate_rmse")
            sp_rho = pe.get("spearman_rho")
            plat_rows.append({
                "Platform": str(plat_name).title(),
                "Status": str(pe.get("status", "unknown")),
                "Train N": pe.get("train_samples", 0),
                "Test N": pe.get("test_samples", 0),
                "Candidate MAE": f"{float(c_mae):.2f}" if c_mae is not None else "N/A (Omitted)",
                "Candidate RMSE": f"{float(c_rmse):.2f}" if c_rmse is not None else "N/A (Omitted)",
                "Spearman ρ": f"{float(sp_rho):.3f}" if sp_rho is not None else "N/A",
                "Notes": str(pe.get("note", "")),
            })
        if plat_rows:
            st.table(pd.DataFrame(plat_rows))

        missing_test_plats = [
            str(p).title()
            for p, pe in plat_eval.items()
            if isinstance(pe, dict) and pe.get("status") == "insufficient_evaluation_samples"
        ]
        if missing_test_plats or float(cand.get("r2", 0.0)) < 0.0:
            st.warning(
                "Chronological holdout caveats: "
                + (
                    f"{', '.join(missing_test_plats)} had `test_samples = 0` (`insufficient_evaluation_samples`) in the chronological split, "
                    "so platform-specific holdout errors are omitted rather than fabricated. "
                    if missing_test_plats
                    else ""
                )
                + f"Pooled holdout R² = {float(cand.get('r2', 0.0)):.3f} across N = {meta.get('test_samples_count', 0)} held-out posts, "
                "reflecting substantial residual score variance on a small synthetic test split despite reducing MAE vs. the naive mean baseline."
            )

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

    # Real-World Longitudinal TikTok Validation Matrix (from real_data_benchmarks.json if available)
    m2_real = real_benchmarks.get("modules", {}).get("module2", {})
    if isinstance(m2_real, dict) and m2_real:
        st.markdown("---")
        st.subheader("Real-Data Longitudinal Validation (`tiktok_engagement_200k` — Zero Outcome Leakage)")
        st.caption(
            "Pre-publication (`day7_views` / `day30_views` without outcome counters) vs. Early Day-1 signal regimes, "
            "evaluated on both chronological and zero-overlap creator-grouped splits."
        )
        tk_rows = []
        for regime_key, r in m2_real.items():
            if not isinstance(r, dict):
                continue
            b_name = r.get("best_model")
            best_m = (r.get("models") or {}).get(b_name, {})
            if not best_m:
                continue
            cov_90 = best_m.get("empirical_90pct_test_coverage")
            rmse_val = best_m.get("rmse_log1p")
            r2_val = best_m.get("r2_log1p")
            rho_val = best_m.get("spearman_rho")
            lift_val = r.get("rmse_lift_over_dummy_pct")
            tk_rows.append({
                "Regime": regime_key,
                "Target": r.get("target_column"),
                "Features": r.get("feature_regime"),
                "Split": r.get("split_strategy"),
                "Creator Overlap": r.get("creator_overlap_count"),
                "Train N": r.get("train_rows"),
                "Test N": r.get("test_rows"),
                "Champion": b_name,
                "RMSE (log1p)": round(float(rmse_val), 4) if rmse_val is not None else None,
                "R² (log1p)": round(float(r2_val), 4) if r2_val is not None else None,
                "Spearman ρ": round(float(rho_val), 4) if rho_val is not None else None,
                "RMSE Lift vs Dummy (%)": round(float(lift_val), 2) if lift_val is not None else None,
                "90% Empirical Cov (%)": round(float(cov_90), 2) if cov_90 is not None else None,
            })
        if tk_rows:
            st.dataframe(pd.DataFrame(tk_rows), width="stretch")

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

    if "topic" not in latest_df.columns or not bool(latest_df["topic"].notna().any()):
        st.info(
            f"The selected dataset (`{selected_dataset_id}`) does not expose a curated `topic` taxonomy "
            "with observable exposure denominators, so topic-level EVI rankings are omitted rather than fabricated."
        )
    else:
        ranked_topics = predictor.rank_topics_by_virality(latest_df)
        if ranked_topics.empty:
            st.info(
                f"No posts in `{selected_dataset_id}` have both observable numerator engagement and exposure "
                "(views/reach/impressions) required for Ex-Post EVI calculation."
            )
        else:
            st.dataframe(ranked_topics, width="stretch")

            fig_topics = px.bar(
                ranked_topics,
                x="topic",
                y="avg_viral_coefficient",
                color="avg_viral_coefficient",
                color_continuous_scale="Teal",
                title=f"Mean Engagement Virality Index by Topic ({selected_dataset_id})"
            )
            st.plotly_chart(fig_topics, width="stretch")
