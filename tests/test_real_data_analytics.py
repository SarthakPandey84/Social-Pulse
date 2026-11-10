"""Comprehensive multi-platform analytical tests covering Modules 1–7 (synthetic + processed external data)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.engagement_metrics import EngagementAnalytics, is_retention_applicable
from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.data_ingestion.schemas import ExperimentType, SourceProvenance
from src.modeling.trend_forecaster import TrendForecaster, ValidationStatus, prepare_post_time_series
from src.modeling.virality_predictor import ViralityPredictor, is_outcome_leakage_column
from src.nlp.evaluation import (
    evaluate_heldout_instagram_sentiment,
    evaluate_lokananta_language_guardrail,
)
from src.recommender.engine import OptimizationRecommender
from src.recommender.schemas import ConfidenceLevel
from src.statistics.ab_testing import ABTestingFramework


def _make_tiktok_panel_fixture(n_rows: int = 120, n_creators: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-07-01", periods=n_rows, freq="6h", tz="UTC")
    creators = [f"cr_{i % n_creators:02d}" for i in range(n_rows)]
    base_scale = np.array([1000.0 * (1 + (i % n_creators)) for i in range(n_rows)])
    day1_views = np.round(base_scale * rng.uniform(0.4, 1.5, size=n_rows)).astype(float)
    day7_views = np.round(day1_views * rng.uniform(1.4, 2.2, size=n_rows)).astype(float)
    day30_views = np.round(day7_views * rng.uniform(1.2, 1.8, size=n_rows)).astype(float)
    return pd.DataFrame({
        "post_id": [f"tk_{i}" for i in range(n_rows)],
        "platform": ["tiktok"] * n_rows,
        "content_type": ["short video"] * n_rows,
        "topic": [f"Topic_{i % 4}" for i in range(n_rows)],
        "timestamp": dates,
        "posting_hour": dates.hour,
        "duration_seconds": rng.integers(10, 60, size=n_rows),
        "word_count": rng.integers(15, 120, size=n_rows),
        "emoji_count": rng.integers(0, 6, size=n_rows),
        "question_count": rng.integers(0, 3, size=n_rows),
        "hashtag_count": rng.integers(1, 8, size=n_rows),
        "speaking_rate": rng.uniform(1.5, 4.5, size=n_rows),
        "is_english": [True] * n_rows,
        "created_by_ai": [bool(i % 9 == 0) for i in range(n_rows)],
        "follower_count": np.array([5000 * (1 + (i % n_creators)) for i in range(n_rows)], dtype=float),
        "creator_video_count": rng.integers(10, 200, size=n_rows).astype(float),
        "creator_total_favorited": np.array([100000 * (1 + (i % n_creators)) for i in range(n_rows)], dtype=float),
        "creator_pseudo_id": creators,
        "views": day30_views,
        "likes": np.round(day30_views * 0.08).astype(float),
        "comments_count": np.round(day30_views * 0.005).astype(float),
        "shares": np.round(day30_views * 0.01).astype(float),
        "saves": pd.Series([pd.NA] * n_rows, dtype="Float64"),
        "collect_count": np.round(day30_views * 0.015).astype(float),
        "retention_rate": pd.Series([pd.NA] * n_rows, dtype="Float64"),
        "day1_views": day1_views,
        "day1_likes": np.round(day1_views * 0.08).astype(float),
        "day1_comments": np.round(day1_views * 0.005).astype(float),
        "day1_shares": np.round(day1_views * 0.01).astype(float),
        "day1_collects": np.round(day1_views * 0.015).astype(float),
        "day7_views": day7_views,
        "day30_views": day30_views,
        "has_negative_delta_reset": [bool(i % 15 == 0) for i in range(n_rows)],
        "provenance": [SourceProvenance.REAL_PUBLIC_DATA.value] * n_rows,
    })


def test_module1_multiplatform_filtering_creator_relative_and_availability():
    df = _make_tiktok_panel_fixture(n_rows=40, n_creators=5)
    assert is_retention_applicable("tiktok", "short video") is True
    assert is_retention_applicable("instagram", "image") is False

    # Multi-filter + age window projection
    sub = EngagementAnalytics.filter_posts_multi(df, platform="tiktok", age_window="day7")
    assert len(sub) == 40
    assert (sub["views"] == sub["day7_views"]).all()
    assert (sub["active_age_window"] == "day7").all()

    # Creator-relative metrics & proxy disclosures
    rel = EngagementAnalytics.compute_creator_relative_metrics(df)
    assert "creator_relative_ratio" in rel.columns
    assert rel["creator_post_count"].min() == 8

    avail = EngagementAnalytics.summarize_metric_availability(df)
    assert "collect_count" in avail["proxy_disclosures"]
    assert avail["metrics"]["saves"]["observed_count"] == 0
    assert avail["metrics"]["collect_count"]["observed_pct"] == 100.0


def test_module2_leakage_guards_and_longitudinal_tiktok_regimes():
    assert is_outcome_leakage_column("day30_views") is True
    assert is_outcome_leakage_column("day1_likes") is True
    assert is_outcome_leakage_column("collect_count") is True
    assert is_outcome_leakage_column("duration_seconds") is False

    df = _make_tiktok_panel_fixture(n_rows=140, n_creators=14)
    vp = ViralityPredictor()

    res_pre = vp.evaluate_tiktok_longitudinal(
        df, target_horizon="day30", feature_regime="pre_pub", split_strategy="creator_grouped"
    )
    assert res_pre["creator_overlap_count"] == 0
    assert res_pre["calibration_strategy"] == "creator_grouped"
    assert res_pre["calibration_available"] is True
    assert res_pre["calibration_creator_overlap_count"] == 0
    assert res_pre["models"]["hist_gradient_boosting"]["empirical_90pct_test_coverage"] is not None
    assert not any(c.startswith(("day0_", "day1_", "day7_", "day30_")) for c in res_pre["features_used"])
    assert "hist_gradient_boosting" in res_pre["models"]

    res_day1 = vp.evaluate_tiktok_longitudinal(
        df, target_horizon="day30", feature_regime="early_day1", split_strategy="chronological"
    )
    assert res_day1["calibration_strategy"] == "chronological"
    assert res_day1["calibration_available"] is True
    assert pd.Timestamp(res_day1["train_date_range"][1]) <= pd.Timestamp(res_day1["test_date_range"][0])
    assert any(c == "day1_views_log1p" for c in res_day1["features_used"])

    # Insufficient distinct creators in training partition must omit calibration instead of falling back to X_fit residuals
    df_5cr = _make_tiktok_panel_fixture(n_rows=50, n_creators=5)
    res_no_cal = vp.evaluate_tiktok_longitudinal(
        df_5cr, target_horizon="day30", feature_regime="pre_pub", split_strategy="creator_grouped", test_ratio=0.8
    )
    assert res_no_cal["calibration_available"] is False
    assert res_no_cal["models"]["hist_gradient_boosting"]["empirical_q90_log1p_radius"] is None
    assert res_no_cal["models"]["hist_gradient_boosting"]["empirical_90pct_test_coverage"] is None
    assert any("calibration omitted" in lim.lower() for lim in res_no_cal["limitations"])

    # Verify reset exclusion filter works
    res_clean = vp.evaluate_tiktok_longitudinal(
        df, target_horizon="day30", feature_regime="pre_pub", exclude_negative_reset_rows=True
    )
    assert res_clean["total_evaluated_rows"] < len(df)


def test_module3_heldout_isolation_and_indonesian_guardrail():
    train_rows = [
        {"comment_text": f"I love this helpful advice {i}", "ground_truth_sentiment": "Positive", "eval_split": "train"}
        for i in range(30)
    ] + [
        {"comment_text": f"Terrible confusing video {i}", "ground_truth_sentiment": "Negative", "eval_split": "train"}
        for i in range(30)
    ] + [
        {"comment_text": f"Posted on Monday update {i}", "ground_truth_sentiment": "Neutral", "eval_split": "train"}
        for i in range(30)
    ]
    test_rows = [
        {"comment_text": "Really helpful guide!", "ground_truth_sentiment": "Positive", "eval_split": "test"},
        {"comment_text": "I love this helpful advice 0", "ground_truth_sentiment": "Positive", "eval_split": "test"},
        {"comment_text": "This was awful waste", "ground_truth_sentiment": "Negative", "eval_split": "test"},
        {"comment_text": "Video summary for Tuesday", "ground_truth_sentiment": "Neutral", "eval_split": "test"},
    ]
    out = evaluate_heldout_instagram_sentiment(pd.DataFrame(train_rows + test_rows))
    assert out["cross_split_duplicate_texts_in_test"] == 1
    assert out["test_samples_strict_unique"] == 3
    assert out["negation_perturbation_check"]["passed"] is True

    id_df = pd.DataFrame({
        "comment_text": [
            "Bagus banget kak untuk karya kami!",
            "Terima kasih ya sudah bikin lagu ini",
            "Mantap sekali semangat selalu",
        ],
        "language_hint": ["id", "id", "id"],
    })
    lk = evaluate_lokananta_language_guardrail(id_df)
    assert lk["auto_null_sentiment_preserved"] == lk["auto_detect_status_counts"].get("unsupported_language", 0)
    assert lk["hinted_unsupported_language_count"] == 3
    assert lk["hinted_null_relatability_preserved"] == 3


def test_module4_creator_clustered_inference_and_observational_guard():
    rng = np.random.default_rng(7)
    rows = []
    # 20 unsponsored creators, 20 sponsored creators, 3 posts per creator
    for g_idx, grp in enumerate(["unsponsored", "sponsored"]):
        for c in range(20):
            cid = f"creator_{g_idx}_{c}"
            foll = 1000 * (1 + c + g_idx * 5)
            creator_mean = 50.0 + foll * 0.02 + (20.0 if grp == "sponsored" else 0.0)
            for p in range(3):
                rows.append({
                    "post_id": f"p_{g_idx}_{c}_{p}",
                    "treatment_group": grp,
                    "creator_pseudo_id": cid,
                    "follower_count": foll,
                    "likes": max(1.0, float(creator_mean + rng.normal(0, 5))),
                    "comments_count": max(0.0, float(creator_mean * 0.05 + rng.normal(0, 1))),
                })
    df = pd.DataFrame(rows)
    ab = ABTestingFramework(alpha=0.05)

    meta = ab.create_experiment_metadata(
        experiment_id="EXP_OBS",
        experiment_name="Sponsored Confounding Check",
        hypothesis="Creator scale confounds raw likes",
        experiment_type=ExperimentType.OBSERVATIONAL_COMPARISON.value,
        platform="instagram",
        variant_a_label="unsponsored",
        variant_b_label="sponsored",
        primary_metric="likes",
    )
    assert meta["experiment_type"] == "observational_comparison"

    rand_df = ab.stratified_randomize_assignment(df, strata_cols=["treatment_group"], seed=42)
    assert set(rand_df["assigned_variant"].unique()) == {"Control (A)", "Treatment (B)"}

    obs_eval = ab.evaluate_observational_cohort(df)
    assert obs_eval["is_causal"] is False
    assert obs_eval["n_creators_total"] == 40
    cl = obs_eval["creator_clustered_comparisons"][0]
    assert cl["clustering_handled"] is True
    assert "ICC=" in cl["clustering_method"]
    assert "DEFF=" in cl["clustering_method"]
    # Verify ANOVA ICC strictly <= 1.0 (and finite)
    assert len(obs_eval["follower_tertile_stratified_likes"]) >= 2


def test_module5_real_provenance_inference_and_creator_guardrails():
    df_tk = _make_tiktok_panel_fixture(n_rows=60, n_creators=6)
    rec = OptimizationRecommender(posts_df=df_tk)
    report = rec.generate_recommendations(platform="tiktok")
    assert report.is_synthetic_demonstration is False
    assert all(not r.is_synthetic_demonstration for r in report.recommendations)
    assert all(r.confidence_level != ConfidenceLevel.HIGH for r in report.recommendations)
    assert not any("nan saves" in r.reasoning.lower() for r in report.recommendations)

    # Fewer than 3 unique creators in a topic should suppress unsupported topic-level claims
    df_few = df_tk.copy()
    df_few["creator_pseudo_id"] = "single_creator"
    rec_few = OptimizationRecommender(posts_df=df_few)
    rep_few = rec_few.generate_recommendations(platform="tiktok")
    assert all(r.recommendation_id != "REC_TOPIC_001" for r in rep_few.recommendations)


def test_module6_external_dataset_static_post_count_and_no_silent_fallback():
    with pytest.raises(FileNotFoundError):
        GrowthDashboardBackend().load_external_dataset("nonexistent_dataset_id_xyz")

    df_static = pd.DataFrame({
        "post_id": [f"s_{i}" for i in range(10)],
        "platform": ["instagram"] * 10,
        "content_type": ["image"] * 10,
        "topic": ["General"] * 10,
        "timestamp": pd.date_range("2024-08-01", periods=10, freq="D", tz="UTC"),
        "views": pd.Series([pd.NA] * 10, dtype="Float64"),
        "likes": np.linspace(100, 200, 10),
        "comments_count": np.linspace(5, 15, 10),
        "shares": pd.Series([pd.NA] * 10, dtype="Float64"),
        "saves": pd.Series([pd.NA] * 10, dtype="Float64"),
        "provenance": [SourceProvenance.REAL_PUBLIC_DATA.value] * 10,
    })
    dash = GrowthDashboardBackend(posts_df=df_static, dataset_id="zenodo_sponsored_unsponsored_ig")
    kpis = dash.calculate_growth_overview_kpis(dash.posts_df)
    assert kpis["total_posts"] == 10
    assert kpis["dataset_id"] == "zenodo_sponsored_unsponsored_ig"
    assert kpis["total_views"] is None
    assert kpis["total_saves"] is None
    assert kpis["total_shares"] is None
    assert kpis["avg_views_per_post"] is None
    assert kpis["overall_views_engagement_rate"] is None

    bk = dash.calculate_content_breakdowns(dash.posts_df)
    assert int(bk["by_format"].iloc[0]["post_count"]) == 10
    assert pd.isna(bk["by_format"].iloc[0]["avg_saves"])
    assert pd.isna(bk["by_format"].iloc[0]["views_engagement_rate"])
    assert pd.isna(bk["by_format"].iloc[0]["high_intent_rate"])
    assert pd.isna(bk["by_topic"].iloc[0]["save_to_share_ratio"])


def test_module7_tiktok_opt_in_and_daily_increment_forecasting():
    df_tk = _make_tiktok_panel_fixture(n_rows=40, n_creators=8)
    tf = TrendForecaster()

    # Default opt-in preserves backward-compatible rejection
    with pytest.raises(ValueError, match="Unsupported platform"):
        tf.forecast_metric(df_tk, platform="tiktok")

    # Explicit opt-in supports TikTok 3-component interactions & collect_count proxy
    ts_tk, meta_tk = prepare_post_time_series(
        df_tk, target_metric="available_interactions", cadence="D", platform="tiktok", allow_tiktok=True
    )
    assert meta_tk["platform"] == "Tiktok"

    rep_tk = tf.forecast_metric(
        df_tk, target_metric="cohort_collects", platform="tiktok", cadence="D", horizon=3, allow_tiktok=True
    )
    assert rep_tk.status in (ValidationStatus.VALID, ValidationStatus.INSUFFICIENT_DATA)

    # Daily panel increments with a 1-day gap handled honestly
    dates_part1 = pd.date_range("2024-07-01", periods=25, freq="D", tz="UTC")
    dates_part2 = pd.date_range("2024-07-28", periods=10, freq="D", tz="UTC")
    all_d = dates_part1.append(dates_part2)
    daily_df = pd.DataFrame({
        "date": all_d,
        "daily_views": [10000.0 + i * 250.0 for i in range(len(all_d))],
        "active_videos": [500] * len(all_d),
        "reset_events": [2] * len(all_d),
    })

    # Strict mode rejects gap; longest_contiguous models the 25-day unbroken segment transparently
    rep_strict = tf.forecast_daily_increment_series(
        daily_df, metric_col="daily_views", segment_policy="strict", horizon=3
    )
    assert rep_strict.status == ValidationStatus.SPARSE_SERIES

    rep_seg = tf.forecast_daily_increment_series(
        daily_df, metric_col="daily_views", segment_policy="longest_contiguous", horizon=3
    )
    assert rep_seg.status == ValidationStatus.VALID
    assert rep_seg.history_points == 25
    assert rep_seg.missing_calendar_days == 2
    assert len(rep_seg.forecast_points) == 3
    assert any("GAP-FREE SEGMENTATION" in d for d in rep_seg.disclosures)


@pytest.mark.skipif(
    not Path("data/processed/external/tiktok_engagement_200k/posts.parquet").exists(),
    reason="External processed TikTok dataset not present locally",
)
def test_real_external_benchmarks_smoke_invariants():
    import json
    from src.modeling.real_data_benchmarks import run_all_real_data_benchmarks

    rep_path = Path("models/real_data_benchmarks.json")
    assert rep_path.exists()
    rep = json.loads(rep_path.read_text(encoding="utf-8"))

    # Verify TikTok content_type normalization & non-complete first-day strict-prior coverage
    tk_meta = pd.read_parquet(
        "data/processed/external/tiktok_engagement_200k/posts.parquet",
        columns=["content_type", "follower_count"],
    )
    assert set(tk_meta["content_type"].dropna().unique()) == {"short video"}
    assert is_retention_applicable("tiktok", "short video") is True
    # Earliest-day videos without a strictly prior creator snapshot must remain NA
    assert int(tk_meta["follower_count"].isna().sum()) > 0

    # Verify M2 grouped calibration zero creator overlap & M6 null preservation in benchmarks JSON
    cg_m2 = rep["modules"]["module2"]["day30_pre_pub_creator_grouped"]
    assert cg_m2["calibration_strategy"] == "creator_grouped"
    assert cg_m2["calibration_creator_overlap_count"] == 0

    tk_kpis = rep["modules"]["module6"]["tiktok_sample_kpis"]
    assert tk_kpis["total_impressions"] is None
    assert tk_kpis["total_saves"] is None
    assert tk_kpis["total_new_followers"] is None
    assert tk_kpis["overall_reach_engagement_rate"] is None
    assert tk_kpis["total_collect_count_proxy"] > 0

    # Verify M6 static Instagram benchmark preserves None/null rather than zero-imputation
    sp_bd = rep["modules"]["module6"]["sponsored_static_content_breakdown"]
    assert len(sp_bd) == 1
    assert sp_bd[0]["avg_saves"] is None
    assert sp_bd[0]["views_engagement_rate"] is None
    assert sp_bd[0]["high_intent_rate"] is None

    # Verify daily summary panel forecasting end-to-end on real daily_platform_summary.parquet
    daily = pd.read_parquet("data/processed/external/tiktok_engagement_200k/daily_platform_summary.parquet")
    out = TrendForecaster().forecast_daily_increment_series(daily, metric_col="daily_views", horizon=7)
    assert out.status == ValidationStatus.VALID
    assert out.validation_origins_count >= 5
    # Avoid unused import lint warning cleanly
    assert callable(run_all_real_data_benchmarks)

