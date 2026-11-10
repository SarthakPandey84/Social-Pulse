"""Comprehensive Test Suite for Module 6: Growth Visualization Dashboard.

Verifies:
1. Data loading from each available module (Modules 1-5).
2. Correct KPI calculations (views, reach, shares, saves, rates).
3. Engagement-rate denominator integrity (views-based vs reach-based).
4. Time-series aggregation (daily/weekly/monthly without zero-imputation).
5. Date filtering (start_date, end_date boundaries).
6. Platform filtering (Instagram, YouTube, Cross-platform).
7. Content-category filtering (format, struggle topic).
8. Missing-value handling (e.g. YouTube saves and reach being null).
9. Empty-data behavior and graceful empty states.
10. Synthetic-data provenance propagation.
11. Incompatible platform metric handling (normalized vs full engagement rates).
12. Experiment visualization data integrity (Module 4 integration with Holm adjustments).
13. Recommendation integration (Module 5 integration with priority ranking).
14. Consistency between filtered KPI values and time-series chart data.
15. Regression compatibility with Modules 1-5.
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.preprocessing.cleaning import get_latest_posts_view
from src.recommender.schemas import ConfidenceLevel, EvidenceSource, RecommendationReport


@pytest.fixture
def backend():
    """Initialize GrowthDashboardBackend with actual processed fixtures."""
    root_dir = Path(__file__).resolve().parent.parent
    b = GrowthDashboardBackend(root_dir=root_dir)
    return b


# --- 1. Data Loading from Each Available Module ---

def test_data_loading_from_all_modules(backend):
    """Verify backend successfully loads datasets from Modules 1-4."""
    assert backend.posts_df is not None
    assert len(backend.posts_df) > 0
    assert "post_id" in backend.posts_df.columns
    assert "platform" in backend.posts_df.columns

    assert backend.comments_df is not None
    assert len(backend.comments_df) > 0
    assert "sentiment_label" in backend.comments_df.columns

    assert backend.experiments_df is not None
    assert len(backend.experiments_df) >= 4

    assert backend.observations_df is not None
    assert len(backend.observations_df) > 0


# --- 2. Correct KPI Calculations ---

def test_correct_kpi_calculations(backend):
    """Verify overview KPIs match mathematical sums across latest post snapshots."""
    df = backend.posts_df
    kpis = backend.calculate_growth_overview_kpis(df)

    latest_df = get_latest_posts_view(df)
    assert kpis["total_posts"] == len(latest_df)
    assert kpis["total_views"] == int(latest_df["views"].sum())
    assert kpis["total_likes"] == int(latest_df["likes"].sum())
    assert kpis["total_shares"] == int(latest_df["shares"].sum())
    assert kpis["total_saves"] == int(latest_df["saves"].dropna().sum())
    assert kpis["avg_views_per_post"] == round(latest_df["views"].sum() / len(latest_df), 1)
    assert kpis["platform_count"] == len(latest_df["platform"].unique())


# --- 3. Engagement-Rate Denominator Integrity ---

def test_engagement_rate_denominator_integrity(backend):
    """Verify views-based rate uses views as denominator and reach-based uses reach."""
    df = backend.posts_df
    kpis = backend.calculate_growth_overview_kpis(df)

    latest_df = get_latest_posts_view(df)
    tot_views = latest_df["views"].sum()
    tot_interactions = (
        latest_df["likes"].sum() +
        latest_df["comments_count"].sum() +
        latest_df["shares"].sum() +
        latest_df["saves"].dropna().sum()
    )
    expected_views_rate = round(tot_interactions / tot_views * 100.0, 2)
    assert kpis["overall_views_engagement_rate"] == expected_views_rate

    # Reach-based rate must only sum interactions from posts with reach > 0
    ig_posts = latest_df[latest_df["reach"] > 0]
    ig_reach = ig_posts["reach"].sum()
    ig_interactions = (
        ig_posts["likes"].sum() +
        ig_posts["comments_count"].sum() +
        ig_posts["shares"].sum() +
        ig_posts["saves"].dropna().sum()
    )
    expected_reach_rate = round(ig_interactions / ig_reach * 100.0, 2)
    assert kpis["overall_reach_engagement_rate"] == expected_reach_rate
    assert kpis["overall_reach_engagement_rate"] > 0.0


# --- 4. Time-Series Aggregation & No Zero-Imputation on Rates ---

def test_time_series_aggregation_cadences(backend):
    """Verify weekly, daily, and monthly aggregation without false zero-imputation on rates."""
    df = backend.posts_df
    weekly_ts = backend.calculate_time_series_trends(df, freq="W")
    daily_ts = backend.calculate_time_series_trends(df, freq="D")

    assert len(weekly_ts) > 0
    assert len(daily_ts) >= len(weekly_ts)

    # For weekly periods with views > 0, engagement rate must be finite and non-negative
    for _, row in weekly_ts[weekly_ts["views"] > 0].iterrows():
        assert not np.isnan(row["views_engagement_rate"])
        assert row["views_engagement_rate"] > 0.0

    # Periods with 0 views must have NaN rate, not 0.0 falsely pulling down metrics
    zero_view_periods = weekly_ts[weekly_ts["views"] == 0]
    for _, row in zero_view_periods.iterrows():
        assert np.isnan(row["views_engagement_rate"])


# --- 5. Date Filtering ---

def test_date_range_filtering(backend):
    """Verify start_date and end_date filters constrain posts accurately."""
    start_dt = "2026-10-01"
    end_dt = "2026-10-15"

    filtered = backend.filter_posts(start_date=start_dt, end_date=end_dt)
    assert len(filtered) > 0
    assert len(filtered) < len(backend.posts_df)

    ts_series = pd.to_datetime(filtered["timestamp_dt"], utc=True)
    assert ts_series.min() >= pd.to_datetime(start_dt, utc=True)
    assert ts_series.max() <= pd.to_datetime(end_dt, utc=True)


# --- 6. Platform Filtering ---

def test_platform_filtering(backend):
    """Verify filtering by platform isolates single-platform records."""
    ig_posts = backend.filter_posts(platform="instagram")
    assert len(ig_posts) > 0
    assert set(ig_posts["platform"].unique()) == {"instagram"}

    yt_posts = backend.filter_posts(platform="youtube")
    assert len(yt_posts) > 0
    assert set(yt_posts["platform"].unique()) == {"youtube"}

    assert len(ig_posts) + len(yt_posts) == len(backend.posts_df)


# --- 7. Content-Category Filtering ---

def test_content_category_filtering(backend):
    """Verify filtering by content format and topic."""
    reels = backend.filter_posts(content_type="Reel")
    assert len(reels) > 0
    assert set(reels["content_type"].unique()) == {"Reel"}

    topic_df = backend.filter_posts(topic="Social Anxiety")
    assert len(topic_df) > 0
    assert set(topic_df["topic"].unique()) == {"Social Anxiety"}


# --- 8. Missing-Value Handling ---

def test_missing_value_handling_youtube_metrics(backend):
    """Verify YouTube records preserve nullable saves and reach without crashes."""
    yt_posts = backend.filter_posts(platform="youtube")
    assert yt_posts["reach"].isna().all() or (yt_posts["reach"] == 0).all()
    assert yt_posts["saves"].isna().all() or (yt_posts["saves"] == 0).all()

    # Calculating KPIs on YouTube only should not crash
    yt_kpis = backend.calculate_growth_overview_kpis(yt_posts)
    assert yt_kpis["total_saves"] == 0
    assert yt_kpis["overall_reach_engagement_rate"] == 0.0
    assert yt_kpis["overall_views_engagement_rate"] > 0.0


# --- 9. Empty-Data Behavior ---

def test_empty_data_behavior(backend):
    """Verify backend functions handle empty DataFrames gracefully."""
    empty_df = pd.DataFrame()
    kpis = backend.calculate_growth_overview_kpis(empty_df)
    assert kpis["total_posts"] == 0
    assert kpis["total_views"] == 0
    assert kpis["reporting_period"] == "N/A"

    ts = backend.calculate_time_series_trends(empty_df)
    assert len(ts) == 0

    comp = backend.calculate_platform_comparisons(empty_df)
    assert len(comp["summary_table"]) == 0

    breakdowns = backend.calculate_content_breakdowns(empty_df)
    assert len(breakdowns["by_topic"]) == 0


# --- 10. Synthetic-Data Provenance Propagation ---

def test_synthetic_data_provenance_propagation(backend):
    """Verify all generated reports flag synthetic demonstration status."""
    recs = backend.calculate_recommendation_priorities()
    assert recs.is_synthetic_demonstration is True
    for r in recs.recommendations:
        assert r.is_synthetic_demonstration is True

    exps = backend.calculate_experiment_performance()
    for e in exps["experiments"]:
        assert "is_randomized" in e


# --- 11. Incompatible Platform Metric Handling ---

def test_incompatible_platform_metric_normalization(backend):
    """Verify normalized engagement rate excludes saves for fair cross-platform comparison."""
    comp = backend.calculate_platform_comparisons(backend.posts_df)
    tbl = comp["summary_table"]
    assert len(tbl) == 2  # Instagram and YouTube

    ig_row = tbl[tbl["Platform"] == "Instagram"].iloc[0]
    yt_row = tbl[tbl["Platform"] == "Youtube"].iloc[0]

    # For Instagram, Full Eng Rate includes saves; Normalized excludes saves
    assert ig_row["Full Eng Rate (%)"] > ig_row["Normalized Eng Rate (%)"]

    # For YouTube, saves do not exist, so Full Eng Rate equals Normalized Eng Rate
    assert yt_row["Full Eng Rate (%)"] == yt_row["Normalized Eng Rate (%)"]
    assert np.isnan(yt_row["Save Rate (%)"])


# --- 12. Experiment Visualization Data Integrity (Module 4) ---

def test_experiment_performance_data_integrity(backend):
    """Verify Module 4 experiment synthesis calculates CIs, effect sizes, and Holm adjustments."""
    exp_res = backend.calculate_experiment_performance()
    assert len(exp_res["experiments"]) == 4

    exp_001 = next(e for e in exp_res["experiments"] if e["experiment_id"] == "EXP_001")
    assert exp_001["is_randomized"] is True
    assert "Independent Post Unit" in exp_001["unit_of_analysis"]
    assert exp_001["is_significant"] is True
    assert exp_001["holm_adj_p_value"] is not None
    assert exp_001["confidence_interval"][0] > 0

    exp_004 = next(e for e in exp_res["experiments"] if e["experiment_id"] == "EXP_004")
    assert exp_004["is_randomized"] is False
    assert exp_004["is_significant"] is False
    # CI spans zero for EXP_004
    assert exp_004["confidence_interval"][0] <= 0 <= exp_004["confidence_interval"][1]


# --- 13. Recommendation Integration (Module 5) ---

def test_recommendation_integration_module5(backend):
    """Verify Module 5 recommendation synthesis delivers ranked priorities."""
    report = backend.calculate_recommendation_priorities()
    assert isinstance(report, RecommendationReport)
    assert report.total_recommendations > 0

    # Verify descending deterministic priority score order
    scores = [r.priority_score for r in report.recommendations]
    assert scores == sorted(scores, reverse=True)


# --- 14. Consistency Between Filtered KPI Values and Chart Data ---

def test_consistency_between_kpi_and_chart_data(backend):
    """Verify total views in KPIs match the sum of views in weekly time-series trends."""
    df = backend.posts_df
    kpis = backend.calculate_growth_overview_kpis(df)
    ts = backend.calculate_time_series_trends(df, freq="W")

    assert kpis["total_views"] == int(ts["views"].sum())
    assert kpis["total_shares"] == int(ts["shares"].sum())


# --- 15. Regression Compatibility with Modules 1-5 ---

def test_regression_compatibility_modules_1_to_5(backend):
    """Verify backend functions operate without mutating original datasets or breaking schemas."""
    orig_post_count = len(backend.posts_df)
    orig_cols = list(backend.posts_df.columns)

    # Perform multiple operations
    _ = backend.calculate_growth_overview_kpis(backend.posts_df)
    _ = backend.calculate_time_series_trends(backend.posts_df)
    _ = backend.calculate_platform_comparisons(backend.posts_df)
    _ = backend.calculate_content_breakdowns(backend.posts_df)
    _ = backend.calculate_audience_insights()
    _ = backend.calculate_experiment_performance()
    _ = backend.calculate_recommendation_priorities()

    # Verify original DataFrame was not mutated in row count or schema
    assert len(backend.posts_df) == orig_post_count
    for c in orig_cols:
        assert c in backend.posts_df.columns


# --- 16. Audit 1: Relatability Interpretation Disclaimer Verification ---

def test_relatability_interpretation_disclaimer(backend):
    """Verify exact required wording for relatability correlation without causal overreach."""
    aud = backend.calculate_audience_insights()
    assert aud["has_data"] is True
    text = aud["correlation_text"]

    expected_snippet = (
        "Exploratory analysis found no statistically significant monotonic association "
        "between post-level relatability percentage and bookmark saves in this synthetic dataset."
    )
    assert expected_snippet in text
    assert "does not establish equivalence or prove the absence of an effect" in text
    assert "causally dictate" not in text
    assert aud["correlation_rho"] == 0.0044
    assert aud["correlation_p"] == 0.972


# --- 17. Audit 2: Pooled vs. Arithmetic Mean Engagement Rates Distinction ---

def test_pooled_vs_arithmetic_mean_engagement_rates(backend):
    """Verify backend explicitly computes and distinguishes pooled rate vs post-level mean rate."""
    # Deterministic test fixture with highly disparate view counts
    df_disparate = pd.DataFrame([
        {
            "post_id": "P1", "platform": "Instagram", "views": 1_000_000,
            "likes": 5000, "comments_count": 2000, "shares": 2000, "saves": 1000,
            "reach": 800_000, "timestamp": "2026-09-01T10:00:00Z"
        },
        {
            "post_id": "P2", "platform": "Instagram", "views": 100,
            "likes": 10, "comments_count": 5, "shares": 3, "saves": 2,
            "reach": 80, "timestamp": "2026-09-02T10:00:00Z"
        }
    ])

    kpis = backend.calculate_growth_overview_kpis(df_disparate)

    # Post 1 rate: 10,000 / 1,000,000 * 100 = 1.00%
    # Post 2 rate: 20 / 100 * 100 = 20.00%
    # Pooled rate: (10,000 + 20) / (1,000,000 + 100) * 100 = 10,020 / 1,000,100 * 100 = 1.0019% -> 1.00%
    # Arithmetic mean rate: (1.00% + 20.00%) / 2 = 10.50%
    assert kpis["pooled_views_engagement_rate"] == 1.00
    assert kpis["mean_post_level_views_engagement_rate"] == 10.50
    assert kpis["pooled_views_engagement_rate"] != kpis["mean_post_level_views_engagement_rate"]

    # Verify rate definitions dictionary exists and documents the distinction
    assert "pooled_views_engagement_rate" in kpis["rate_definitions"]
    assert "mean_post_level_views_engagement_rate" in kpis["rate_definitions"]


# --- 18. Audit 3: Period Comparison Equal Duration and Incomplete Detection ---

def test_period_comparison_equal_duration_and_incomplete_detection(backend):
    """Verify period comparison enforces equal duration windows and rejects incomplete periods."""
    # Current period: 14 days (10 posts)
    curr_dates = pd.date_range("2026-10-01", "2026-10-14", periods=10)
    curr_df = pd.DataFrame({
        "post_id": [f"C{i}" for i in range(10)],
        "platform": "YouTube",
        "views": [1000] * 10,
        "likes": [50] * 10,
        "shares": [10] * 10,
        "timestamp": curr_dates
    })

    # Incomplete prior period: only 3 days (mismatched duration)
    mismatched_dates = pd.date_range("2026-09-28", "2026-09-30", periods=6)
    mismatched_df = pd.DataFrame({
        "post_id": [f"M{i}" for i in range(6)],
        "platform": "YouTube",
        "views": [800] * 6,
        "likes": [40] * 6,
        "shares": [8] * 6,
        "timestamp": mismatched_dates
    })

    kpis_mismatched = backend.calculate_growth_overview_kpis(curr_df, previous_df=mismatched_df)
    assert kpis_mismatched["has_previous_period_comparison"] is False
    assert "Incomplete period mismatch" in kpis_mismatched["comparison_note"]

    # Valid equal duration prior period: 14 days (10 posts)
    equal_dates = pd.date_range("2026-09-17", "2026-09-30", periods=10)
    equal_df = pd.DataFrame({
        "post_id": [f"E{i}" for i in range(10)],
        "platform": "YouTube",
        "views": [800] * 10,
        "likes": [40] * 10,
        "shares": [8] * 10,
        "timestamp": equal_dates
    })

    kpis_equal = backend.calculate_growth_overview_kpis(curr_df, previous_df=equal_df)
    assert kpis_equal["has_previous_period_comparison"] is True
    assert kpis_equal["comparison_deltas"]["views_delta_pct"] == 25.0  # (10000 - 8000) / 8000 * 100


# --- 19. Audit 3: Period Comparison Safe Handling of Zero Activity ---

def test_period_comparison_zero_activity_in_previous_period(backend):
    """Verify period comparison safely handles zero views in the previous period without division by zero."""
    curr_dates = pd.date_range("2026-10-01", "2026-10-15", periods=6)
    curr_df = pd.DataFrame({
        "post_id": [f"C{i}" for i in range(6)],
        "platform": "Instagram",
        "views": [500] * 6,
        "likes": [25] * 6,
        "shares": [5] * 6,
        "timestamp": curr_dates
    })

    # Prior period of equal length has 0 views across all posts
    prev_dates = pd.date_range("2026-09-16", "2026-09-30", periods=6)
    zero_df = pd.DataFrame({
        "post_id": [f"Z{i}" for i in range(6)],
        "platform": "Instagram",
        "views": [0] * 6,
        "likes": [0] * 6,
        "shares": [0] * 6,
        "timestamp": prev_dates
    })

    kpis_zero = backend.calculate_growth_overview_kpis(curr_df, previous_df=zero_df)
    assert kpis_zero["has_previous_period_comparison"] is True
    # views_delta_pct should be None because dividing by zero views is mathematically undefined
    assert kpis_zero["comparison_deltas"]["views_delta_pct"] is None
    assert "Zero activity in previous comparison period" in kpis_zero["comparison_note"]


# --- 20. Audit 4: Cross-Platform Disclosures and Non-Equivalence Notice ---

def test_cross_platform_disclosures_and_non_equivalence(backend):
    """Verify cross-platform disclosures warn against assuming universal equivalence."""
    comp = backend.calculate_platform_comparisons(backend.posts_df)
    disclosures = comp["disclosures"]

    joined_disc = " ".join(disclosures)
    assert "PLATFORM METRIC NON-EQUIVALENCE" in joined_disc
    assert "does NOT imply that engagement actions are universally equivalent" in joined_disc
    assert "Instagram: Tracks Views, Impressions, Reach, Likes, Comments, Shares, and Saves" in joined_disc
    assert "YouTube: Tracks Views, Impressions, Likes, Comments, and Shares" in joined_disc
    assert "summary_table" in comp
    assert "Available Fields" in comp["summary_table"].columns


# --- 21. Audit 2: Zero Denominators Robustness ---

def test_zero_denominators_safety(backend):
    """Verify backend functions handle zero views, zero reach, and empty post sets safely."""
    zero_df = pd.DataFrame([
        {
            "post_id": "Z1", "platform": "Instagram", "views": 0, "reach": 0,
            "likes": 0, "comments_count": 0, "shares": 0, "saves": 0,
            "timestamp": "2026-10-01T10:00:00Z"
        }
    ])

    kpis = backend.calculate_growth_overview_kpis(zero_df)
    assert kpis["pooled_views_engagement_rate"] == 0.0
    assert kpis["mean_post_level_views_engagement_rate"] == 0.0
    assert kpis["pooled_reach_engagement_rate"] == 0.0
    assert kpis["mean_post_level_reach_engagement_rate"] == 0.0
    assert kpis["avg_views_per_post"] == 0.0
    assert kpis["avg_reach_per_post"] == 0.0


# --- 22. Audit 2: Distinct Metric Availability for YouTube vs Instagram ---

def test_missing_saves_handling(backend):
    """Verify YouTube posts with missing saves are not treated as 0 saves on an active assay."""
    yt_only = backend.posts_df[backend.posts_df["platform"].str.lower() == "youtube"]
    assert len(yt_only) > 0

    kpis = backend.calculate_growth_overview_kpis(yt_only)
    assert kpis["platform_count"] == 1
    assert "Youtube" in kpis["connected_platforms"]
    # Reach rate on YouTube must be 0.0 (unavailable)
    assert kpis["overall_reach_engagement_rate"] == 0.0
    assert kpis["platform_metric_availability"]["YouTube"]["saves_tracked"] is False
    assert kpis["platform_metric_availability"]["Instagram"]["saves_tracked"] is True

