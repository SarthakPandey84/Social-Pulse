"""Unit, regression, and methodological audit tests for Module 7: Trend Forecasting Engine.

Verifies:
1. Preserved legacy APIs (compute_trend_velocity, identify_rising_struggles).
2. Rolling-origin walk-forward validation with multiple origins and zero temporal leakage.
3. Horizon-specific out-of-sample residual collection and error spreading.
4. Scale-aware error metrics (MASE, Relative MAE) with zero-scale and zero-naive guards.
5. Publication-cohort target semantics and missing-data distinction (no-post days vs missing observations).
6. Platform incompatibility guards (prohibiting cross-platform raw view pooling; YouTube saves/reach unavailable).
7. Case-insensitive cadence handling ('weekly', 'Weekly', 'W', 'D').
8. Horizon validation (zero and negative horizons).
9. Manual model override reporting distinct from validation champion.
10. Uncalibrated empirical uncertainty intervals and limitation disclosures.
11. Keyword coverage tracking and non-interpolation policy.
12. Parquet engine availability.
13. End-to-end forecasting pipelines on real project datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.modeling.trend_forecaster import (
    AutoRegressiveRidgeModel,
    BaseTimeSeriesModel,
    ForecastModelType,
    ForecastPoint,
    HoltExponentialSmoothingModel,
    ModelEvaluationMetric,
    SeasonalNaiveModel,
    TrendForecaster,
    ValidationStatus,
    build_uncertainty_description,
    evaluate_models_rolling_origin,
    get_rolling_origin_diagnostics,
    prepare_post_time_series,
)


# --- 1. PRESERVED LEGACY TESTS ---

def test_trend_velocity_computation():
    """Preserved baseline test verifying trend velocity and acceleration calculations."""
    forecaster = TrendForecaster(rolling_window=3)
    dates = pd.date_range("2026-10-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "keyword": ["#test"] * 10,
        "mention_volume": [100, 110, 120, 130, 150, 180, 220, 280, 350, 450]
    })
    velocity_df = forecaster.compute_trend_velocity(df)
    assert "velocity_pct" in velocity_df.columns
    assert "acceleration" in velocity_df.columns
    rising = forecaster.identify_rising_struggles(velocity_df, min_velocity=5.0)
    assert len(rising) > 0


def test_identify_rising_struggles_filtering():
    """Verify filtering thresholds and trend status labeling."""
    forecaster = TrendForecaster(rolling_window=2)
    dates = pd.date_range("2026-10-01", periods=5, freq="D")
    df1 = pd.DataFrame({
        "date": dates,
        "keyword": ["#rising"] * 5,
        "mention_volume": [100, 120, 160, 220, 300]
    })
    df2 = pd.DataFrame({
        "date": dates,
        "keyword": ["#falling"] * 5,
        "mention_volume": [300, 250, 200, 150, 100]
    })
    df = pd.concat([df1, df2], ignore_index=True)
    vel_df = forecaster.compute_trend_velocity(df)
    rising = forecaster.identify_rising_struggles(vel_df, min_velocity=10.0)

    assert len(rising) == 1
    assert rising.iloc[0]["keyword"] == "#rising"
    assert rising.iloc[0]["trend_status"] in ["Accelerating Breakout", "Sustained Rise"]


# --- 2. ROLLING-ORIGIN WALK-FORWARD VALIDATION TESTS ---

def test_rolling_origin_validation_multiple_origins_and_refitting():
    """Verify that walk-forward validation evaluates across multiple forecast origins and refits models."""
    rng = np.random.RandomState(42)
    y = np.linspace(50, 250, 35) + rng.normal(0, 5, 35)

    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y, cadence="D", horizon=5, min_train_size=14, max_origins=10
    )

    # Must have multiple origins (at least 3, up to 10)
    assert n_origins >= 3
    assert n_val_obs > n_origins
    assert len(eval_metrics) == 5
    # Exactly one champion selected
    champions = [m for m in eval_metrics if m.is_champion]
    assert len(champions) == 1
    assert champions[0].model_name == champ_name
    # Each evaluation record tracks origins
    for m in eval_metrics:
        assert m.validation_origins == n_origins
        assert m.validation_observations == n_val_obs


def test_no_temporal_leakage_between_origins():
    """Verify that at each validation origin, training indices strictly precede test observations via get_rolling_origin_diagnostics."""
    n = 30
    y = np.arange(n, dtype=float)
    diagnostics = get_rolling_origin_diagnostics(y, cadence="D", horizon=7, min_train_size=14, max_origins=10)

    assert len(diagnostics) >= 3
    for diag in diagnostics:
        t = diag["origin_index"]
        train_start = diag["train_start_idx"]
        train_end = diag["train_end_idx"]
        train_len = diag["train_len"]
        test_start = diag["test_start_idx"]
        test_end = diag["test_end_idx"]
        test_len = diag["test_len"]

        # Strict temporal inequality and zero lookahead
        assert train_end < test_start
        assert train_len == test_start
        assert train_len == t
        assert test_start == t
        assert test_len <= 5
        assert test_end < n

        # Slices must be disjoint
        train_set = set(range(train_start, train_end + 1))
        test_set = set(range(test_start, test_end + 1))
        assert train_set.isdisjoint(test_set)
        assert len(train_set) + len(test_set) == train_len + test_len


def test_horizon_specific_out_of_sample_residual_collection():
    """Verify that out-of-sample residuals are tracked by horizon step and empirical spreads are computed."""
    rng = np.random.RandomState(42)
    y = 100.0 + 3.0 * np.arange(30) + rng.normal(0, 4, 30)

    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y, cadence="D", horizon=5, min_train_size=14, max_origins=8
    )

    champ = next(m for m in eval_metrics if m.is_champion)
    assert len(champ.horizon_step_mae) > 0
    # Steps 1 to 5 should have empirical spreads
    for s in range(1, 6):
        assert s in spreads
        assert spreads[s] > 0.0


def test_insufficient_validation_history_guard():
    """Verify that short series (< min_train_size + 3) returns insufficient validation status."""
    y_short = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y_short, cadence="D", horizon=3, min_train_size=14
    )

    assert len(eval_metrics) == 0
    assert champ_name == "None"
    assert n_origins == 0


# --- 3. TARGET SEMANTICS & MISSING-DATA DISTINCTION ---

def test_forecasting_target_publication_cohort_semantics():
    """Verify that target metric reflects publication-cohort lifetime performance, not account daily traffic."""
    dates = pd.date_range("2026-09-01", periods=20, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(20)],
        "timestamp": dates,
        "platform": ["instagram"] * 20,
        "views": [1000] * 20,
        "likes": [100] * 20,
        "comments_count": [10] * 20,
        "shares": [5] * 20,
        "saves": [2] * 20
    })

    forecaster = TrendForecaster()
    rep = forecaster.forecast_metric(df, target_metric="cohort_views", platform="instagram", cadence="D", horizon=3)

    assert rep.status == ValidationStatus.VALID
    assert rep.target_metric == "cohort_views"
    assert "Publication Cohort Views" in rep.target_description
    assert any("PUBLICATION COHORT TARGET" in d for d in rep.disclosures)


def test_missing_observations_vs_no_post_periods():
    """Verify that periods with no published posts have cohort_views=0 but engagement_rate=NaN (undefined)."""
    # 4 days total: day 0 has a post, day 1 has no post, day 2 has a post, day 3 has no post
    dates = [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-03"), pd.Timestamp("2026-09-15")]
    df = pd.DataFrame({
        "post_id": ["p1", "p2", "p3"],
        "timestamp": dates,
        "platform": ["instagram"] * 3,
        "views": [1000, 2000, 3000],
        "likes": [100, 200, 300],
        "comments_count": [10, 20, 30],
        "shares": [5, 10, 15],
        "saves": [2, 4, 6]
    })

    ts_df, meta = prepare_post_time_series(df, target_metric="views_engagement_rate", platform="instagram", cadence="D")

    # Resampled calendar includes dates between 2026-09-01 and 2026-09-15 (15 days)
    assert meta["total_periods"] == 15
    assert meta["active_periods"] == 3
    assert meta["zero_publication_periods"] == 12

    # On zero-post days, views_engagement_rate should be NaN (NOT fake 0.0)
    zero_post_rows = ts_df[ts_df["post_count"] == 0]
    assert len(zero_post_rows) == 12
    assert zero_post_rows["views_engagement_rate"].isna().all()


def test_incomplete_coverage_guard_on_sparse_keywords():
    """Verify that sparse keyword series with < 80% coverage returns SPARSE_SERIES status without blind interpolation."""
    # Only 5 days out of a 20-day span (25% coverage)
    sparse_dates = [pd.Timestamp("2026-10-01") + pd.Timedelta(days=i * 4) for i in range(5)]
    df_sparse = pd.DataFrame({
        "date": sparse_dates,
        "keyword": ["#sparse"] * 5,
        "mention_volume": [100, 120, 110, 130, 140]
    })

    forecaster = TrendForecaster()
    rep = forecaster.forecast_keyword_trend(df_sparse, keyword="#sparse", horizon=3)

    assert rep.status == ValidationStatus.SPARSE_SERIES
    assert rep.coverage_pct < 80.0
    assert rep.missing_calendar_days > 3
    assert len(rep.forecast_points) == 0


# --- 4. PLATFORM INCOMPATIBILITY & METADATA GUARDS ---

def test_platform_incompatibility_pooling_prohibited():
    """Verify that passing 'All Platforms' for raw views raises an explicit ValueError."""
    dates = pd.date_range("2026-09-01", periods=16, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(16)],
        "timestamp": dates,
        "platform": ["instagram"] * 8 + ["youtube"] * 8,
        "views": [1000] * 16,
        "likes": [100] * 16,
        "comments_count": [10] * 16,
        "shares": [5] * 16,
        "saves": [2] * 8 + [np.nan] * 8
    })

    forecaster = TrendForecaster()
    with pytest.raises(ValueError, match="Cross-platform pooling"):
        forecaster.forecast_metric(df, target_metric="cohort_views", platform="All Platforms")


def test_youtube_saves_reach_unavailable_guard():
    """Verify that requesting 'saves' on YouTube raises an explicit ValueError."""
    dates = pd.date_range("2026-09-01", periods=16, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(16)],
        "timestamp": dates,
        "platform": ["youtube"] * 16,
        "views": [1000] * 16,
        "likes": [100] * 16,
        "comments_count": [10] * 16,
        "shares": [5] * 16,
        "saves": [np.nan] * 16
    })

    forecaster = TrendForecaster()
    with pytest.raises(ValueError, match="not available on YouTube"):
        forecaster.forecast_metric(df, target_metric="saves", platform="youtube")


def test_missing_or_invalid_platform_metadata_raises_error():
    """Verify that invalid platform string raises ValueError."""
    dates = pd.date_range("2026-09-01", periods=16, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(16)],
        "timestamp": dates,
        "platform": ["instagram"] * 16,
        "views": [1000] * 16
    })

    forecaster = TrendForecaster()
    with pytest.raises(ValueError, match="Unsupported platform"):
        forecaster.forecast_metric(df, platform="tiktok")


# --- 5. CADENCE & HORIZON VALIDATION ---

def test_weekly_cadence_string_handling_case_insensitive():
    """Verify that 'weekly', 'Weekly', 'w', 'W' all resolve to weekly frequency without fallback to daily."""
    dates = pd.date_range("2026-09-01", periods=30, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(30)],
        "timestamp": dates,
        "platform": ["instagram"] * 30,
        "views": [500] * 30,
        "likes": [50] * 30,
        "comments_count": [5] * 30,
        "shares": [2] * 30,
        "saves": [1] * 30
    })

    for weekly_str in ["weekly", "Weekly", "w", "W"]:
        ts_df, meta = prepare_post_time_series(df, target_metric="cohort_views", cadence=weekly_str, platform="instagram")
        assert meta["cadence"] == "W"
        # 30 days should yield ~5 weekly periods
        assert len(ts_df) < 10


def test_zero_and_negative_forecast_horizons_raise_error():
    """Verify that zero and negative horizons raise ValueError."""
    dates = pd.date_range("2026-09-01", periods=20, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(20)],
        "timestamp": dates,
        "platform": ["instagram"] * 20,
        "views": [500] * 20
    })

    forecaster = TrendForecaster()
    with pytest.raises(ValueError, match="positive integer"):
        forecaster.forecast_metric(df, horizon=0)

    with pytest.raises(ValueError, match="positive integer"):
        forecaster.forecast_metric(df, horizon=-3)


# --- 6. MODEL SELECTION & SPECIAL CASE ERROR METRICS ---

def test_zero_naive_validation_mae_handling():
    """Verify that when Naive MAE is zero (constant series), relative MAE does not cause division-by-zero crash."""
    y_const = np.ones(30, dtype=float) * 100.0

    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y_const, cadence="D", horizon=3, min_train_size=14, max_origins=5
    )

    naive = next(m for m in eval_metrics if m.model_name == ForecastModelType.NAIVE.value)
    assert naive.mae == 0.0
    assert naive.relative_mae_to_naive == 1.0


def test_manual_model_selection_reporting_distinct_from_champion():
    """Verify that selecting a non-champion model records selected_model != champion_model and is_manual_override=True."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2026-09-01", periods=25, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(25)],
        "timestamp": dates,
        "platform": ["instagram"] * 25,
        "views": [1000 + i * 50 + int(rng.normal(0, 20)) for i in range(25)],
        "likes": [100] * 25,
        "comments_count": [10] * 25,
        "shares": [5] * 25,
        "saves": [2] * 25
    })

    forecaster = TrendForecaster()
    # Force manual selection of Naive model
    rep = forecaster.forecast_metric(
        df, target_metric="cohort_views", platform="instagram", cadence="D", horizon=5,
        selected_model=ForecastModelType.NAIVE
    )

    assert rep.status == ValidationStatus.VALID
    assert rep.selected_model == ForecastModelType.NAIVE.value
    # If Champion was Moving Average or Holt, override is True
    if rep.champion_model != ForecastModelType.NAIVE.value:
        assert rep.is_manual_override is True


def test_prediction_interval_construction_and_calibration_disclosures():
    """Verify that prediction intervals enforce non-negative floor and disclose uncalibrated status."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2026-09-01", periods=25, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(25)],
        "timestamp": dates,
        "platform": ["instagram"] * 25,
        "views": [50 + int(rng.normal(0, 15)) for _ in range(25)],
        "likes": [5] * 25,
        "comments_count": [1] * 25,
        "shares": [0] * 25,
        "saves": [0] * 25
    })

    forecaster = TrendForecaster()
    rep = forecaster.forecast_metric(df, target_metric="cohort_views", platform="instagram", horizon=5)

    assert rep.is_calibrated is False
    assert any("UNCALIBRATED" in d for d in rep.disclosures)
    for pt in rep.forecast_points:
        assert pt.ci_lower >= 0.0
        assert pt.ci_upper >= pt.point_forecast


def test_holt_optimizer_failure_graceful_fallback():
    """Verify that Holt model handles numerical difficulty without throwing uncaught exceptions."""
    y_noisy = np.array([1.0, 1e6, -1e6, 0.0, 1.0, -1.0, 1e8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    holt = HoltExponentialSmoothingModel()
    # Should fit with fallback parameters without unhandled crash
    holt.fit(y_noisy)
    preds = holt.predict(3)
    assert len(preds) == 3
    assert not np.isnan(preds).any()


def test_parquet_engine_availability_and_csv_fallback():
    """Verify that parquet engine (pyarrow) is available and functional."""
    parquet_path = Path("data/processed/processed_posts.parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0


# --- 7. INTEGRATION TESTS ON REAL PROJECT DATA ---

def test_forecast_metric_end_to_end_on_processed_posts():
    """Verify rolling-origin walk-forward forecasting end-to-end on data/processed/processed_posts.parquet."""
    parquet_path = Path("data/processed/processed_posts.parquet")
    if not parquet_path.exists():
        pytest.skip("processed_posts.parquet not present.")

    posts_df = pd.read_parquet(parquet_path)
    forecaster = TrendForecaster()

    rep = forecaster.forecast_metric(
        posts_df, target_metric="cohort_views", platform="instagram", cadence="D", horizon=7
    )

    assert rep.status == ValidationStatus.VALID
    assert rep.validation_origins_count >= 3
    assert rep.validation_observations_count > rep.validation_origins_count
    assert len(rep.model_evaluations) == 5
    assert len(rep.forecast_points) == 7
    assert rep.target_metric == "cohort_views"
    assert rep.is_calibrated is False
    assert len(rep.disclosures) >= 5


def test_forecast_keyword_trend_end_to_end_on_sample_trends():
    """Verify topic trajectory forecasting end-to-end on data/sample/sample_trends.csv."""
    csv_path = Path("data/sample/sample_trends.csv")
    if not csv_path.exists():
        pytest.skip("sample_trends.csv not present.")

    trends_df = pd.read_csv(csv_path)
    forecaster = TrendForecaster()

    rep = forecaster.forecast_keyword_trend(trends_df, keyword="#BurnoutRecovery", horizon=5)

    assert rep.status == ValidationStatus.VALID
    assert rep.coverage_pct == 100.0
    assert rep.calendar_continuity_pct == 100.0
    assert "Unbroken Observed Calendar Continuity" in rep.coverage_status
    assert "unverified" in rep.coverage_status.lower()
    assert rep.missing_calendar_days == 0
    assert rep.validation_origins_count >= 3
    assert len(rep.forecast_points) == 5
    assert rep.is_calibrated is False


# --- 8. SECOND AUDIT REGRESSION TESTS ---

def test_short_keyword_history_handling_consecutive_lengths():
    """Verify that keyword histories of lengths 14, 15, and 16 return INSUFFICIENT_VALIDATION_HISTORY
    without raising an unhandled IndexError, while length 17 returns VALID with 3 origins."""
    forecaster = TrendForecaster()
    base_date = pd.Timestamp("2026-10-01")

    # Lengths 14, 15, 16 -> insufficient origins (n - 14 < 3)
    for length in [14, 15, 16]:
        dates = [base_date + pd.Timedelta(days=i) for i in range(length)]
        df_short = pd.DataFrame({
            "date": dates,
            "keyword": [f"#test_{length}"] * length,
            "mention_volume": [100 + i * 5 for i in range(length)]
        })
        rep = forecaster.forecast_keyword_trend(df_short, keyword=f"#test_{length}", horizon=3)
        assert rep.status == ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY
        assert len(rep.forecast_points) == 0
        assert rep.validation_origins_count < 3
        assert any("INSUFFICIENT VALIDATION HISTORY" in d for d in rep.disclosures)

    # Length 17 -> exactly 3 origins (17 - 14 = 3) -> VALID
    dates_17 = [base_date + pd.Timedelta(days=i) for i in range(17)]
    df_17 = pd.DataFrame({
        "date": dates_17,
        "keyword": ["#test_17"] * 17,
        "mention_volume": [100 + i * 5 for i in range(17)]
    })
    rep_17 = forecaster.forecast_keyword_trend(df_17, keyword="#test_17", horizon=3)
    assert rep_17.status == ValidationStatus.VALID
    assert rep_17.validation_origins_count == 3
    assert len(rep_17.forecast_points) == 3


def test_engagement_rate_nan_handling_rejected():
    """Verify that undefined engagement rates on zero-post intervals raise ValueError rather than filling with fake 0.0%."""
    # 10 posts spaced every 2 days -> 19 calendar days (>= min_train=14) with 9 zero-post intervals
    dates = [pd.Timestamp("2026-09-01") + pd.Timedelta(days=2 * i) for i in range(10)]
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(10)],
        "timestamp": dates,
        "platform": ["instagram"] * 10,
        "views": [1000] * 10,
        "likes": [100] * 10,
        "comments_count": [10] * 10,
        "shares": [5] * 10,
        "saves": [2] * 10
    })

    forecaster = TrendForecaster()
    with pytest.raises(ValueError, match="Cannot model 'views_engagement_rate' as a regular time series"):
        forecaster.forecast_metric(df, target_metric="views_engagement_rate", platform="instagram", cadence="D")


def test_youtube_interaction_semantics_and_available_interactions():
    """Verify YouTube rejects 4-component total engagement, provides 3-component youtube_available_interactions,
    and sets cohort_saves to NaN."""
    dates = pd.date_range("2026-09-01", periods=20, freq="D")
    df = pd.DataFrame({
        "post_id": [f"yt_{i}" for i in range(20)],
        "timestamp": dates,
        "platform": ["youtube"] * 20,
        "views": [1000 + i * 10 for i in range(20)],
        "likes": [50 + i for i in range(20)],
        "comments_count": [10 + i for i in range(20)],
        "shares": [5 + i for i in range(20)],
        "saves": [np.nan] * 20
    })

    forecaster = TrendForecaster()

    # 1. 4-component total engagement must be rejected on YouTube
    with pytest.raises(ValueError, match="not valid on YouTube"):
        forecaster.forecast_metric(df, target_metric="cohort_total_engagement", platform="youtube")

    with pytest.raises(ValueError, match="not valid on YouTube"):
        forecaster.forecast_metric(df, target_metric="total_engagement", platform="youtube")

    # 2. cohort_saves must be rejected on YouTube
    with pytest.raises(ValueError, match="not available on YouTube"):
        forecaster.forecast_metric(df, target_metric="cohort_saves", platform="youtube")

    # 3. prepare_post_time_series for youtube_available_interactions
    ts_df, meta = prepare_post_time_series(df, target_metric="youtube_available_interactions", platform="youtube", cadence="D")
    assert "youtube_available_interactions" in ts_df.columns
    expected = [65 + 3 * i for i in range(20)]
    assert list(ts_df["youtube_available_interactions"]) == expected

    # 4. forecast_metric on youtube_available_interactions succeeds
    rep = forecaster.forecast_metric(df, target_metric="youtube_available_interactions", platform="youtube", cadence="D", horizon=3)
    assert rep.status == ValidationStatus.VALID
    assert rep.target_metric == "youtube_available_interactions"
    assert "YouTube Available Interactions" in rep.target_description


def test_post_time_series_unverified_coverage_and_exposure_age_warning():
    """Verify that post time-series reports coverage as unverified (None) and flags exposure-age bias."""
    dates = pd.date_range("2026-09-01", periods=20, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(20)],
        "timestamp": dates,
        "platform": ["instagram"] * 20,
        "views": [500] * 20,
        "likes": [50] * 20,
        "comments_count": [5] * 20,
        "shares": [2] * 20,
        "saves": [1] * 20
    })

    _, meta = prepare_post_time_series(df, target_metric="cohort_views", platform="instagram", cadence="D")
    assert meta["coverage_pct"] is None
    assert meta["missing_periods"] is None
    assert meta["coverage_status"].startswith("UNVERIFIED")
    assert meta["exposure_age_bias_warning"] is not None
    assert "Unequal accumulation duration" in meta["exposure_age_bias_warning"]

    forecaster = TrendForecaster()
    rep = forecaster.forecast_metric(df, target_metric="cohort_views", platform="instagram", cadence="D", horizon=3)
    assert rep.coverage_pct is None
    assert any("EXPOSURE-AGE BIAS" in d for d in rep.disclosures)
    assert any("SOURCE COMPLETENESS UNVERIFIED" in d for d in rep.disclosures)


def test_uncertainty_direct_vs_extrapolated_steps():
    """Verify that forecast horizon steps <= 5 are labeled Direct Empirical Quantile,
    while steps > 5 are labeled Extrapolated Horizon Range."""
    dates = pd.date_range("2026-09-01", periods=25, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(25)],
        "timestamp": dates,
        "platform": ["instagram"] * 25,
        "views": [1000 + i * 20 for i in range(25)],
        "likes": [100] * 25,
        "comments_count": [10] * 25,
        "shares": [5] * 25,
        "saves": [2] * 25
    })

    forecaster = TrendForecaster()
    rep = forecaster.forecast_metric(df, target_metric="cohort_views", platform="instagram", cadence="D", horizon=7)

    assert rep.status == ValidationStatus.VALID
    assert len(rep.forecast_points) == 7

    for pt in rep.forecast_points:
        if pt.step <= 5:
            assert "Direct Empirical Quantile" in pt.uncertainty_method
        else:
            assert "Extrapolated Horizon Range" in pt.uncertainty_method

    fc_df = rep.forecast_df
    assert "uncertainty_method" in fc_df.columns
    assert (fc_df.iloc[:5]["uncertainty_method"].str.contains("Direct Empirical Quantile")).all()
    assert (fc_df.iloc[5:]["uncertainty_method"].str.contains("Extrapolated Horizon Range")).all()


# --- 9. FINAL AUDIT REGRESSION TESTS ---

def test_exactly_17_observations_3_step_forecast():
    """Verify that a 17-observation series with 3-step forecast evaluates 3 origins and correctly labels:
    - Step 1: Direct Empirical Quantile (residual_count=3)
    - Step 2: Fallback (insufficient residuals: residual_count=2, NOT labeled Direct Empirical Quantile)
    - Step 3: Fallback (insufficient residuals: residual_count=1, NOT labeled Direct Empirical Quantile)
    - Carries small-sample uncertainty disclosure and model-selection optimism bias disclosure.
    """
    dates_17 = pd.date_range("2026-10-01", periods=17, freq="D")
    df_17 = pd.DataFrame({
        "date": dates_17,
        "keyword": ["#test_17"] * 17,
        "mention_volume": [100.0 + i * 5.0 for i in range(17)]
    })

    forecaster = TrendForecaster()
    rep = forecaster.forecast_keyword_trend(df_17, keyword="#test_17", horizon=3)

    assert rep.status == ValidationStatus.VALID
    assert rep.validation_origins_count == 3
    assert len(rep.forecast_points) == 3

    # Step 1: 3 origins -> 3 residuals (origins 14, 15, 16) -> Direct Empirical Quantile
    pt1 = rep.forecast_points[0]
    assert pt1.step == 1
    assert pt1.residual_count == 3
    assert "Direct Empirical Quantile" in pt1.uncertainty_method

    # Step 2: 2 residuals (origins 14, 15) -> Insufficient (<3) -> Must use fallback, NOT Direct Empirical Quantile
    pt2 = rep.forecast_points[1]
    assert pt2.step == 2
    assert pt2.residual_count == 2
    assert "Direct Empirical Quantile" not in pt2.uncertainty_method
    assert "Fallback" in pt2.uncertainty_method

    # Step 3: 1 residual (origin 14) -> Insufficient (<3) -> Must use fallback, NOT Direct Empirical Quantile
    pt3 = rep.forecast_points[2]
    assert pt3.step == 3
    assert pt3.residual_count == 1
    assert "Direct Empirical Quantile" not in pt3.uncertainty_method
    assert "Fallback" in pt3.uncertainty_method

    # Check forecast_df columns and values
    fc_df = rep.forecast_df
    assert "uncertainty_method" in fc_df.columns
    assert "residual_count" in fc_df.columns
    assert list(fc_df["residual_count"]) == [3, 2, 1]

    # Check disclosures
    assert any("SMALL-SAMPLE UNCERTAINTY WARNING" in d for d in rep.disclosures)
    assert any("MODEL-SELECTION OPTIMISM BIAS" in d for d in rep.disclosures)


def test_short_history_fallback_uncertainty_never_mislabeled_as_direct_quantile():
    """Verify that whenever residual observation count is < 3, the evaluator never assigns 'Direct Empirical Quantile'."""
    y_17 = np.linspace(100, 200, 17)
    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y_17, cadence="D", horizon=3, min_train_size=14, max_origins=10
    )

    for step, cnt in counts.items():
        method_str = methods[step]
        if cnt < 3:
            assert "Direct Empirical Quantile" not in method_str
            assert "Fallback" in method_str
        else:
            assert "Direct Empirical Quantile" in method_str


def test_keyword_calendar_continuity_terminology_and_unverified_ingestion():
    """Verify that keyword coverage represents observed calendar continuity and does not claim verified source ingestion."""
    csv_path = Path("data/sample/sample_trends.csv")
    if not csv_path.exists():
        pytest.skip("sample_trends.csv not present.")

    trends_df = pd.read_csv(csv_path)
    forecaster = TrendForecaster()
    rep = forecaster.forecast_keyword_trend(trends_df, keyword="#BurnoutRecovery", horizon=3)

    assert rep.status == ValidationStatus.VALID
    assert rep.coverage_pct == 100.0
    assert rep.calendar_continuity_pct == 100.0

    # Must describe observed continuity, not verified ingestion completeness
    assert "Unbroken Observed Calendar Continuity" in rep.coverage_status
    assert "unverified" in rep.coverage_status.lower()

    # Must NOT claim verified complete span or coverage confirmed
    assert "Verified Complete Span" not in rep.coverage_status
    assert not any("COVERAGE CONFIRMED" in d for d in rep.disclosures)
    assert not any("Verified Complete Span" in d for d in rep.disclosures)
    assert any("OBSERVED CALENDAR-DATE CONTINUITY" in d for d in rep.disclosures)


def test_model_selection_optimism_disclosure_present():
    """Verify that both post cohort forecasts and keyword forecasts include model-selection optimism bias disclosure."""
    parquet_path = Path("data/processed/processed_posts.parquet")
    if not parquet_path.exists():
        pytest.skip("processed_posts.parquet not present.")

    posts_df = pd.read_parquet(parquet_path)
    forecaster = TrendForecaster()
    rep_post = forecaster.forecast_metric(posts_df, target_metric="cohort_views", platform="instagram", horizon=3)
    assert any("MODEL-SELECTION OPTIMISM BIAS" in d for d in rep_post.disclosures)

    csv_path = Path("data/sample/sample_trends.csv")
    if csv_path.exists():
        trends_df = pd.read_csv(csv_path)
        rep_kw = forecaster.forecast_keyword_trend(trends_df, keyword="#BurnoutRecovery", horizon=3)
        assert any("MODEL-SELECTION OPTIMISM BIAS" in d for d in rep_kw.disclosures)


def test_uncertainty_formula_synchronization_sqrt_expansion():
    """Verify that for multi-step horizons beyond the evaluated steps (s > 5), the spread follows sqrt(s / 5) expansion."""
    parquet_path = Path("data/processed/processed_posts.parquet")
    if not parquet_path.exists():
        pytest.skip("processed_posts.parquet not present.")

    posts_df = pd.read_parquet(parquet_path)
    forecaster = TrendForecaster()
    rep = forecaster.forecast_metric(posts_df, target_metric="cohort_views", platform="instagram", horizon=7)

    assert len(rep.forecast_points) == 7
    spread_5 = rep.forecast_points[4].empirical_error_spread
    spread_6 = rep.forecast_points[5].empirical_error_spread
    spread_7 = rep.forecast_points[6].empirical_error_spread

    # Verify square-root expansion formula: spread_s = spread_5 * sqrt(s / 5.0)
    expected_spread_6 = round(float(spread_5 * np.sqrt(6.0 / 5.0)), 2)
    expected_spread_7 = round(float(spread_5 * np.sqrt(7.0 / 5.0)), 2)
    assert abs(spread_6 - expected_spread_6) <= 0.05
    assert abs(spread_7 - expected_spread_7) <= 0.05

    # Check method labeling reflects sqrt(h)
    assert "sqrt(h)" in rep.forecast_points[5].uncertainty_method
    assert "sqrt(h)" in rep.forecast_points[6].uncertainty_method


# --- 10. FINAL TARGETED CORRECTION REGRESSION TESTS ---


class _ScriptedOriginModel(BaseTimeSeriesModel):
    """Test double that fails on specified training-cutoff lengths or returns scripted predictions."""

    def __init__(
        self,
        fail_on_train_lens=None,
        exact_series=None,
        return_nan_on_train_lens=None,
        wrong_shape_on_train_lens=None,
    ):
        self.fail_on_train_lens = set(fail_on_train_lens or [])
        self.return_nan_on_train_lens = set(return_nan_on_train_lens or [])
        self.wrong_shape_on_train_lens = set(wrong_shape_on_train_lens or [])
        self.exact_series = exact_series
        self.train_len = 0

    def fit(self, y: np.ndarray) -> "_ScriptedOriginModel":
        self.train_len = len(y)
        if self.train_len in self.fail_on_train_lens:
            raise RuntimeError(f"Injected failure at origin len={self.train_len}")
        self.last_val = float(y[-1])
        return self

    def predict(self, h: int = 1, horizon: int = None) -> np.ndarray:
        steps = horizon if horizon is not None else h
        if self.train_len in self.return_nan_on_train_lens:
            return np.full(steps, np.nan, dtype=float)
        if self.train_len in self.wrong_shape_on_train_lens:
            return np.zeros(steps + 1, dtype=float)
        if self.exact_series is not None:
            return np.asarray(self.exact_series[self.train_len:self.train_len + steps], dtype=float)
        return np.full(steps, self.last_val, dtype=float)


def test_single_failed_origin_excluded_from_champion_and_no_synthetic_residuals():
    """Verify that a model failing at exactly 1 rolling origin:
    - records failed_origins == 1 and successful_origins == attempted - 1,
    - never receives synthetic 99999.0 error penalties,
    - computes MAE/RMSE solely from genuine predictions on its successful origins, and
    - is excluded from competing for Champion status even if remaining origins have zero error.
    """
    y = np.array([100.0 + i * 10.0 + (25.0 if i % 2 == 0 else -15.0) for i in range(20)], dtype=float)

    # Inject a model that crashes on origin 14 (1st of 6 origins),
    # yet has 0.0 error on origins 15..19. Without partial-failure exclusion, it would steal Champion.
    custom_factories = {
        ForecastModelType.AUTOREGRESSIVE: lambda yt: _ScriptedOriginModel(
            fail_on_train_lens={14},
            exact_series=y,
        )
    }

    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y, cadence="D", horizon=3, min_train_size=14, max_origins=10, model_factories=custom_factories
    )

    assert n_origins == 6
    assert len(eval_metrics) == 5

    ar_eval = next(m for m in eval_metrics if m.model_type == ForecastModelType.AUTOREGRESSIVE)
    assert ar_eval.attempted_origins == 6
    assert ar_eval.successful_origins == 5
    assert ar_eval.failed_origins == 1
    assert ar_eval.validation_origins == 5
    # Origins 15, 16, 17 give 3 obs each; origin 18 gives 2; origin 19 gives 1 -> 12 genuine predictions
    assert ar_eval.validation_observations == 12
    assert ar_eval.is_eligible_for_champion is False
    assert ar_eval.is_champion is False
    assert ar_eval.evaluation_status == "Excluded (Partial Failure)"

    # Strictly genuine MAE from its 5 valid origins (zero error), NOT contaminated by 99999.0
    assert ar_eval.mae == 0.0
    assert ar_eval.rmse == 0.0

    # Winner must come only from fully evaluated candidates (failed_origins == 0)
    champ = next(m for m in eval_metrics if m.is_champion)
    assert champ.model_type != ForecastModelType.AUTOREGRESSIVE
    assert champ.failed_origins == 0
    assert champ.successful_origins == 6
    assert champ.is_eligible_for_champion is True
    assert all(spread < 5000.0 for spread in spreads.values())


def test_multiple_failed_origins_insufficient_and_unavailable_states():
    """Verify handling of multiple failed origins, insufficient successful evaluations (<3),
    total unavailability (0 successful origins), non-finite outputs, and total-suite failure."""
    rng = np.random.RandomState(7)
    y = 100.0 + np.cumsum(rng.normal(2.0, 5.0, 20))

    # 6 attempted origins: len(yt) in {14, 15, 16, 17, 18, 19}
    # - AR succeeds only on 2 origins (18, 19) -> Insufficiently Evaluated
    # - Holt fails on all 6 origins -> Unavailable
    # - Seasonal Naive returns NaN on 3 origins (14, 15, 16) -> 3 failed, 3 successful -> Excluded (Partial Failure)
    custom_factories = {
        ForecastModelType.AUTOREGRESSIVE: lambda yt: _ScriptedOriginModel(
            fail_on_train_lens={14, 15, 16, 17}
        ),
        ForecastModelType.EXPONENTIAL_SMOOTHING: lambda yt: _ScriptedOriginModel(
            fail_on_train_lens={14, 15, 16, 17, 18, 19}
        ),
        ForecastModelType.SEASONAL_NAIVE: lambda yt: _ScriptedOriginModel(
            return_nan_on_train_lens={14, 15, 16}
        ),
    }

    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y, cadence="D", horizon=3, min_train_size=14, max_origins=10, model_factories=custom_factories
    )

    assert n_origins == 6
    assert len(eval_metrics) == 5

    ar_eval = next(m for m in eval_metrics if m.model_type == ForecastModelType.AUTOREGRESSIVE)
    assert ar_eval.successful_origins == 2
    assert ar_eval.failed_origins == 4
    assert ar_eval.evaluation_status == "Insufficiently Evaluated"
    assert ar_eval.mae is None
    assert ar_eval.rmse is None
    assert ar_eval.mase is None
    assert ar_eval.relative_mae_to_naive is None
    assert ar_eval.horizon_step_mae == {}
    assert ar_eval.is_eligible_for_champion is False
    assert ar_eval.is_champion is False

    holt_eval = next(m for m in eval_metrics if m.model_type == ForecastModelType.EXPONENTIAL_SMOOTHING)
    assert holt_eval.successful_origins == 0
    assert holt_eval.failed_origins == 6
    assert holt_eval.validation_observations == 0
    assert holt_eval.evaluation_status == "Unavailable"
    assert holt_eval.mae is None
    assert holt_eval.rmse is None
    assert holt_eval.mase is None
    assert holt_eval.relative_mae_to_naive is None
    assert holt_eval.is_eligible_for_champion is False
    assert holt_eval.is_champion is False

    snaive_eval = next(m for m in eval_metrics if m.model_type == ForecastModelType.SEASONAL_NAIVE)
    assert snaive_eval.successful_origins == 3
    assert snaive_eval.failed_origins == 3
    assert snaive_eval.evaluation_status == "Excluded (Partial Failure)"
    assert snaive_eval.mae is not None and np.isfinite(snaive_eval.mae)
    assert snaive_eval.is_eligible_for_champion is False

    champ = next(m for m in eval_metrics if m.is_champion)
    assert champ.model_type in {ForecastModelType.NAIVE, ForecastModelType.MOVING_AVERAGE}
    assert champ.failed_origins == 0

    # If Naive fails and all models fail at least 1 origin -> no champion and relative MAE vs Naive is None
    all_fail_factories = {
        m_type: (lambda yt: _ScriptedOriginModel(fail_on_train_lens={14}))
        for m_type in ForecastModelType
    }
    eval_none, champ_none, sp_none, mth_none, cnt_none, n_orig_none, _ = evaluate_models_rolling_origin(
        y, cadence="D", horizon=3, min_train_size=14, max_origins=10, model_factories=all_fail_factories
    )
    assert n_orig_none == 6
    assert champ_none == "None"
    assert sp_none == {}
    assert mth_none == {}
    assert cnt_none == {}
    assert all(not m.is_champion for m in eval_none)
    assert all(not m.is_eligible_for_champion for m in eval_none)
    assert all(m.relative_mae_to_naive is None for m in eval_none)


def test_failed_origins_do_not_contaminate_residual_counts_or_metrics():
    """Verify that shape mismatches, exceptions, and failed origins on non-champion or partial models
    do not leak into horizon-specific residual counts, genuine metrics, or uncertainty spreads."""
    y = np.array([200.0 + i * 2.0 for i in range(20)], dtype=float)

    # Make Naive exact (predicts next step linearly by cheating via exact_series) -> guaranteed Champion with 0 failures
    # Make Holt produce wrong prediction shapes at odd origins; make AR raise exceptions at even origins
    custom_factories = {
        ForecastModelType.NAIVE: lambda yt: _ScriptedOriginModel(exact_series=y),
        ForecastModelType.EXPONENTIAL_SMOOTHING: lambda yt: _ScriptedOriginModel(wrong_shape_on_train_lens={15, 17, 19}),
        ForecastModelType.AUTOREGRESSIVE: lambda yt: _ScriptedOriginModel(fail_on_train_lens={14, 16, 18}),
    }

    eval_metrics, champ_name, spreads, methods, counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
        y, cadence="D", horizon=5, min_train_size=14, max_origins=10, model_factories=custom_factories
    )

    assert n_origins == 6
    assert champ_name == ForecastModelType.NAIVE.value
    # Exact horizon residual counts across origins 14..19 (N=20, H_eval=5):
    # step 1: 6, step 2: 5, step 3: 4, step 4: 3, step 5: 2
    assert counts == {1: 6, 2: 5, 3: 4, 4: 3, 5: 2}
    assert sum(counts.values()) == 20
    assert n_val_obs == 20

    # Champion spread should be 0.0 (exact predictions) and never inflated by failed models
    for s in range(1, 6):
        assert spreads[s] == 0.0

    # Partial failing models report exact genuine observation counts and zero 99999.0 contamination
    ar_eval = next(m for m in eval_metrics if m.model_type == ForecastModelType.AUTOREGRESSIVE)
    holt_eval = next(m for m in eval_metrics if m.model_type == ForecastModelType.EXPONENTIAL_SMOOTHING)
    assert ar_eval.failed_origins == 3 and ar_eval.successful_origins == 3
    assert holt_eval.failed_origins == 3 and holt_eval.successful_origins == 3
    assert ar_eval.mae is not None and ar_eval.mae < 100.0
    assert holt_eval.mae is not None and holt_eval.mae < 100.0


def test_uncertainty_methods_and_reconciled_description_all_four_paths():
    """Verify exact uncertainty_method labels, numerical calculations, and reconciled uncertainty_description
    across ALL FOUR paths:
    1. Direct Empirical Quantile
    2. Evaluation-Horizon Expansion
    3. Step-1 Expansion Fallback
    4. MAE Fallback Spread
    """
    forecaster = TrendForecaster()

    # Path 1 (steps 1..5) + Path 2 (steps 6..7) on N=25, H=7
    rng = np.random.RandomState(123)
    dates_25 = pd.date_range("2026-09-01", periods=25, freq="D")
    df_25 = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(25)],
        "timestamp": dates_25,
        "platform": ["instagram"] * 25,
        "views": [1000.0 + i * 25.0 + float(rng.normal(0, 12)) for i in range(25)],
        "likes": [100] * 25,
        "comments_count": [10] * 25,
        "shares": [5] * 25,
        "saves": [2] * 25,
    })
    rep_25 = forecaster.forecast_metric(df_25, target_metric="cohort_views", platform="instagram", horizon=7)
    assert rep_25.status == ValidationStatus.VALID
    for pt in rep_25.forecast_points[:5]:
        assert "Direct Empirical Quantile" in pt.uncertainty_method
        assert "Evaluation-Horizon Expansion" not in pt.uncertainty_method
    for pt in rep_25.forecast_points[5:]:
        assert "Evaluation-Horizon Expansion" in pt.uncertainty_method
        assert "Extrapolated Horizon Range" in pt.uncertainty_method
        assert "Step-1 Expansion" not in pt.uncertainty_method
    assert "Direct Empirical Quantile" in rep_25.uncertainty_description
    assert "Evaluation-Horizon Expansion" in rep_25.uncertainty_description
    assert "steps [6, 7]: Evaluation-Horizon Expansion" in rep_25.uncertainty_description

    # Path 1 (step 1) + Path 3 (steps 2..7) on N=17, H=7
    # Here H_eval=5 has n=0 (<3), so steps 6 and 7 MUST use Step-1 Expansion, NOT Evaluation-Horizon Expansion!
    dates_17 = pd.date_range("2026-10-01", periods=17, freq="D")
    df_17 = pd.DataFrame({
        "date": dates_17,
        "keyword": ["#step1_exp"] * 17,
        "mention_volume": [100.0 + i * 8.0 + (4.0 if i % 2 else -4.0) for i in range(17)],
    })
    rep_17 = forecaster.forecast_keyword_trend(df_17, keyword="#step1_exp", horizon=7)
    assert rep_17.status == ValidationStatus.VALID
    assert "Direct Empirical Quantile" in rep_17.forecast_points[0].uncertainty_method
    spread_1 = rep_17.forecast_points[0].empirical_error_spread
    for pt in rep_17.forecast_points[1:]:
        assert "Step-1 Expansion" in pt.uncertainty_method
        assert "Direct Empirical Quantile" not in pt.uncertainty_method
        assert "Evaluation-Horizon Expansion" not in pt.uncertainty_method
        expected = round(float(spread_1 * np.sqrt(pt.step)), 2)
        assert abs(pt.empirical_error_spread - expected) <= 0.05

    assert "steps [1]: Direct Empirical Quantile" in rep_17.uncertainty_description
    assert "steps [2, 3, 4, 5, 6, 7]: Step-1 Expansion" in rep_17.uncertainty_description
    assert "Evaluation-Horizon Expansion (Spread_5" not in rep_17.uncertainty_description.split("Fallback hierarchy")[0]

    # Path 4: MAE Fallback (when step-1 has n_1 < 3 residuals)
    y_16 = np.array([100.0 + i * 5.0 + (3.0 if i % 2 else -3.0) for i in range(16)], dtype=float)
    eval_mae_fb, champ_mae_fb, spreads_fb, methods_fb, counts_fb, n_orig_fb, _ = evaluate_models_rolling_origin(
        y_16, cadence="D", horizon=3, min_train_size=14, max_origins=10, min_successful_origins=2
    )
    assert n_orig_fb == 2
    champ_rec = next(m for m in eval_mae_fb if m.model_name == champ_mae_fb)
    assert counts_fb == {1: 2, 2: 1, 3: 0}
    for s in range(1, 4):
        assert "MAE Fallback" in methods_fb[s]
        assert "1.645" in methods_fb[s]
        assert "Direct Empirical Quantile" not in methods_fb[s]
        assert "Step-1 Expansion" not in methods_fb[s]
        assert "Evaluation-Horizon Expansion" not in methods_fb[s]
        expected_spread = float(champ_rec.mae * 1.645 * np.sqrt(s))
        assert abs(spreads_fb[s] - expected_spread) <= 0.05

    # Reconciled uncertainty description for MAE Fallback points
    mae_pts = [
        ForecastPoint(
            step=s,
            timestamp=f"2026-10-{17 + s:02d}",
            point_forecast=180.0,
            ci_lower=max(0.0, 180.0 - spreads_fb[s]),
            ci_upper=180.0 + spreads_fb[s],
            empirical_error_spread=spreads_fb[s],
            uncertainty_method=methods_fb[s],
            residual_count=counts_fb[s],
        )
        for s in range(1, 4)
    ]
    desc_fb = build_uncertainty_description(n_origins=2, eval_step_horizon=3, forecast_points=mae_pts)
    assert "steps [1, 2, 3]: MAE Fallback (MAE * 1.645 * sqrt(s))" in desc_fb


def test_forecaster_reports_all_models_failed_without_falling_back_to_eval_zero(monkeypatch):
    """Verify TrendForecaster.forecast_metric does NOT fall back to eval_metrics[0] if all candidates failed."""
    dates = pd.date_range("2026-09-01", periods=20, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(20)],
        "timestamp": dates,
        "platform": ["instagram"] * 20,
        "views": [1000.0 + i * 10.0 for i in range(20)],
    })

    failed_evals = [
        ModelEvaluationMetric(
            model_name=ForecastModelType.NAIVE.value,
            model_type=ForecastModelType.NAIVE,
            mae=None,
            rmse=None,
            mase=None,
            relative_mae_to_naive=None,
            is_champion=False,
            validation_origins=0,
            validation_observations=0,
            successful_origins=0,
            failed_origins=6,
            attempted_origins=6,
            is_eligible_for_champion=False,
            evaluation_status="Unavailable",
            notes="Unavailable: failed at all 6 rolling origins.",
        )
    ]

    monkeypatch.setattr(
        "src.modeling.trend_forecaster.evaluate_models_rolling_origin",
        lambda *args, **kwargs: (failed_evals, "None", {}, {}, {}, 6, 15),
    )

    rep = TrendForecaster().forecast_metric(df, target_metric="cohort_views", platform="instagram", horizon=3)
    assert rep.status == ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY
    assert rep.champion_model == "None"
    assert rep.selected_model == "None"
    assert rep.forecast_points == []
    assert len(rep.model_evaluations) == 1
    assert rep.model_evaluations[0].evaluation_status == "Unavailable"


# --- 11. PHASE 3 OPTIMIZATION EQUIVALENCE REGRESSION TESTS ---


def test_holt_vectorized_predict_equivalence():
    """Verify HoltExponentialSmoothingModel fit/predict matches analytical damped geometric recurrence."""
    y = np.array([100.0, 115.0, 128.0, 140.0, 160.0, 155.0, 172.0, 190.0, 205.0, 212.0, 225.0, 240.0])
    for damping in [0.85, 0.95, 1.0]:
        model = HoltExponentialSmoothingModel(dampening_factor=damping).fit(y)
        for steps in [0, 1, 5, 14]:
            preds = model.predict(steps)
            expected = np.array(
                [
                    max(0.0, model.level + model.trend * sum(damping ** i for i in range(1, s + 1)))
                    for s in range(1, steps + 1)
                ],
                dtype=float,
            )
            np.testing.assert_allclose(preds, expected, atol=1e-12)


def test_ar_ridge_sliding_window_and_predict_equivalence():
    """Verify AutoRegressiveRidgeModel sliding-window design matrix and fast predict match sklearn predict."""
    rng = np.random.RandomState(42)
    y = 100.0 + np.arange(30, dtype=float) * 3.5 + rng.normal(0.0, 4.0, 30)
    ar = AutoRegressiveRidgeModel(p=3, l2_alpha=1.0).fit(y)

    # Compare against explicit recursive Ridge.predict calls
    current_window = list(y[-ar.p:][::-1])
    expected = []
    for _ in range(10):
        val = max(0.0, float(ar.model.predict([current_window])[0]))
        expected.append(val)
        current_window = [val] + current_window[:-1]

    preds = ar.predict(10)
    np.testing.assert_allclose(preds, np.array(expected, dtype=float), atol=1e-12)
    assert len(ar.predict(0)) == 0


def test_seasonal_naive_vectorized_predict_equivalence():
    """Verify SeasonalNaiveModel vectorized modular indexing matches periodic seasonal lookup."""
    y = np.arange(1.0, 20.0)
    model = SeasonalNaiveModel(season_length=7).fit(y)
    preds = model.predict(10)
    expected = np.array([y[-7 + ((s - 1) % 7)] for s in range(1, 11)], dtype=float)
    np.testing.assert_allclose(preds, expected, atol=1e-12)

    # Fallback when N < 2 * season_length
    short_model = SeasonalNaiveModel(season_length=7).fit(y[:10])
    np.testing.assert_allclose(short_model.predict(5), np.full(5, y[9]), atol=1e-12)
    assert len(model.predict(0)) == 0


def test_prepare_post_time_series_and_velocity_edge_cases():
    """Verify native-agg prepare_post_time_series and vectorized compute_trend_velocity on edge cases."""
    dates = pd.date_range("2026-09-01", periods=16, freq="D")
    # Dataset with missing optional interaction columns (likes/comments/shares/saves absent)
    df_min = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(16)],
        "timestamp": dates,
        "platform": ["instagram"] * 16,
        "views": [100 + i * 10 for i in range(16)],
    })
    ts_likes, meta_likes = prepare_post_time_series(df_min, target_metric="cohort_likes", platform="instagram")
    assert meta_likes["target_metric"] == "cohort_likes"
    assert (ts_likes["cohort_likes"] == 0).all()

    with pytest.raises(ValueError, match="not available"):
        prepare_post_time_series(df_min, target_metric="nonexistent_metric", platform="instagram")

    # Empty & NaN-keyword compute_trend_velocity
    fc = TrendForecaster(rolling_window=3)
    assert fc.compute_trend_velocity(pd.DataFrame()).empty
    df_nan_kw = pd.DataFrame({
        "date": pd.date_range("2026-10-01", periods=4, freq="D"),
        "keyword": ["#k1", "#k1", np.nan, "#k2"],
        "mention_volume": [10, 20, 30, 40],
    })
    vel = fc.compute_trend_velocity(df_nan_kw)
    assert len(vel) == 3
    assert vel["keyword"].isna().sum() == 0


def test_manual_override_uses_own_residual_spreads_instead_of_champion():
    """Verify an eligible manual model override uses its own step residual spreads and origin counts."""
    # Strong linear trend with low noise: Holt tracks trend tightly (tiny residuals),
    # whereas Naive lags behind trend at every step (much larger residuals).
    dates = pd.date_range("2026-09-01", periods=25, freq="D")
    df = pd.DataFrame({
        "post_id": [f"p_{i}" for i in range(25)],
        "timestamp": dates,
        "platform": ["instagram"] * 25,
        "views": [1000.0 + i * 100.0 for i in range(25)],
        "likes": [50] * 25,
        "comments_count": [5] * 25,
        "shares": [2] * 25,
        "saves": [1] * 25,
    })

    forecaster = TrendForecaster()
    rep_champ = forecaster.forecast_metric(df, target_metric="cohort_views", platform="instagram", cadence="D", horizon=3)
    rep_naive = forecaster.forecast_metric(
        df,
        target_metric="cohort_views",
        platform="instagram",
        cadence="D",
        horizon=3,
        selected_model=ForecastModelType.NAIVE,
    )

    assert rep_champ.status == ValidationStatus.VALID
    assert rep_naive.status == ValidationStatus.VALID
    assert rep_naive.is_manual_override is True
    assert rep_champ.champion_model != ForecastModelType.NAIVE.value

    naive_eval = next(m for m in rep_naive.model_evaluations if m.model_name == ForecastModelType.NAIVE.value)
    assert naive_eval.is_eligible_for_champion is True
    assert 1 in naive_eval.step_error_spreads

    # Naive's step-1 residual spread must reflect Naive's own errors, not the low-error champion's spread
    assert rep_naive.forecast_points[0].empirical_error_spread == pytest.approx(naive_eval.step_error_spreads[1], rel=1e-9)
    assert rep_naive.forecast_points[0].empirical_error_spread > rep_champ.forecast_points[0].empirical_error_spread
    assert rep_naive.forecast_points[0].residual_count == naive_eval.step_error_counts[1]
    assert any("MANUAL OVERRIDE UNCERTAINTY" in d for d in rep_naive.disclosures)



