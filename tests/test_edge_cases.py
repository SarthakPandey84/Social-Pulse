"""Edge-case tests for missing values, zero views/shares, invalid timestamps, duplicates, and unavailable metrics."""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from src.modeling.virality_predictor import ViralityPredictor
from src.data_ingestion.data_validator import DataValidator
from src.preprocessing.cleaning import clean_posts_dataset, get_latest_posts_view
from src.preprocessing.feature_engineering import add_engagement_features


def test_unavailable_metrics_renormalization():
    """Verify weights re-normalize over available metrics and do not assume missing metrics are zero."""
    predictor = ViralityPredictor()

    # Case A: Full metrics available (shares, saves, comments, likes)
    row_full = pd.Series({
        "views": 1000,
        "shares": 50,
        "saves": 40,
        "comments_count": 20,
        "likes": 100,
        "retention_rate": 0.60
    })
    score_full = predictor.calculate_single_post_score(row_full)
    assert score_full["confidence_tier"] == "high"
    assert "shares" in score_full["metrics_contributed"]
    assert "saves" in score_full["metrics_contributed"]
    assert score_full["weight_coverage"] == 1.0

    # Case B: YouTube or Public IG (saves and shares are NaN / unavailable)
    row_partial = pd.Series({
        "views": 1000,
        "shares": np.nan,
        "saves": np.nan,
        "comments_count": 20,
        "likes": 100,
        "retention_rate": 0.60
    })
    score_partial = predictor.calculate_single_post_score(row_partial)
    assert score_partial["confidence_tier"] == "low"
    assert "shares" not in score_partial["metrics_contributed"]
    assert "saves" not in score_partial["metrics_contributed"]
    # Available weights sum = 0.15 (comments) + 0.10 (likes) = 0.25
    assert score_partial["weight_coverage"] == 0.25
    assert not np.isnan(score_partial["viral_coefficient"])


def test_all_engagement_metrics_missing():
    """Verify that when all engagement metrics are missing, score returns NaN rather than 0."""
    predictor = ViralityPredictor()
    row_empty = pd.Series({
        "views": 1000,
        "shares": np.nan,
        "saves": np.nan,
        "comments_count": np.nan,
        "likes": np.nan,
        "retention_rate": 0.50
    })
    res = predictor.calculate_single_post_score(row_empty)
    assert np.isnan(res["viral_coefficient"])
    assert res["confidence_tier"] == "unmeasured"


def test_zero_views_division_by_zero_prevention():
    """Verify that posts with 0 views do not crash or produce infinity due to Laplace smoothing."""
    predictor = ViralityPredictor()
    row_zero_views = pd.Series({
        "views": 0,
        "reach": 0,
        "shares": 5,
        "saves": 5,
        "comments_count": 2,
        "likes": 10,
        "retention_rate": 0.5
    })
    res = predictor.calculate_single_post_score(row_zero_views)
    assert not np.isnan(res["viral_coefficient"])
    assert not np.isinf(res["viral_coefficient"])
    assert res["viral_coefficient"] > 0


def test_zero_shares_handling():
    """Verify explicit handling when shares == 0 in feature engineering (no silent substitution of 1)."""
    df = pd.DataFrame([{
        "views": 1000,
        "likes": 50,
        "comments_count": 5,
        "shares": 0,
        "saves": 25,
        "retention_rate": 0.5
    }])
    df_feat = add_engagement_features(df)
    # When shares == 0 and saves > 0, ratio is mathematically undefined (pd.NA with status undefined_zero_shares)
    assert pd.isna(df_feat.iloc[0]["save_to_share_ratio"])
    assert df_feat.iloc[0]["save_share_ratio_status"] == "undefined_zero_shares"
    assert df_feat.iloc[0]["high_intent_actions"] == 25


def test_invalid_timestamps_detection():
    """Verify data validator detects future dates and malformed strings."""
    now_utc = datetime.now(timezone.utc)
    future_date = (now_utc + timedelta(days=5)).isoformat()

    df_future = pd.DataFrame([{
        "platform": "instagram",
        "post_id": "P_FUTURE",
        "timestamp": future_date,
        "content_type": "reel",
        "views": 100,
        "impressions": 120,
        "reach": 80,
        "likes": 10,
        "comments_count": 2,
        "shares": 1,
        "saves": 1,
        "follower_count": 1000,
        "retention_rate": 0.5,
        "collection_timestamp": now_utc.isoformat()
    }])

    report = DataValidator.validate_unified_posts(df_future)
    assert report["is_valid"] is False
    assert any("future" in err.lower() for err in report["errors"])


def test_duplicate_snapshot_detection_and_cleaning():
    """Verify cleaning preserves historical snapshots while deduplicating exact identical snapshots."""
    # 2 historical snapshots over time (different collection_timestamp)
    # plus 1 redundant identical snapshot (same collection_timestamp)
    df_snapshots = pd.DataFrame([
        {
            "platform": "instagram",
            "post_id": "POST_DUP_1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 100,
            "impressions": 120,
            "reach": 80,
            "likes": 10,
            "comments_count": 2,
            "shares": 1,
            "saves": 1,
            "follower_count": 1000,
            "retention_rate": 0.5,
            "collection_timestamp": "2026-10-01T13:00:00Z"
        },
        {
            "platform": "instagram",
            "post_id": "POST_DUP_1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 500,  # Newer observation
            "impressions": 600,
            "reach": 400,
            "likes": 50,
            "comments_count": 10,
            "shares": 8,
            "saves": 12,
            "follower_count": 1000,
            "retention_rate": 0.6,
            "collection_timestamp": "2026-10-02T13:00:00Z"
        },
        {
            "platform": "instagram",
            "post_id": "POST_DUP_1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 500,  # Redundant exact duplicate of row 2
            "impressions": 600,
            "reach": 400,
            "likes": 50,
            "comments_count": 10,
            "shares": 8,
            "saves": 12,
            "follower_count": 1000,
            "retention_rate": 0.6,
            "collection_timestamp": "2026-10-02T13:00:00Z"
        }
    ])

    report = DataValidator.validate_unified_posts(df_snapshots)
    assert report["duplicate_snapshot_count"] == 2

    # Cleaning must eliminate the redundant snapshot, leaving 2 historical snapshots
    cleaned = clean_posts_dataset(df_snapshots)
    assert len(cleaned) == 2

    # Latest record view selects strictly the most recent observation (views = 500)
    latest = get_latest_posts_view(cleaned)
    assert len(latest) == 1
    assert latest.iloc[0]["views"] == 500


def test_reach_vs_views_denominator_selection():
    """Verify that reach is prioritized when available in auto mode."""
    predictor = ViralityPredictor()
    row = pd.Series({
        "views": 10000,
        "reach": 5000,
        "shares": 100,
        "saves": 100,
        "comments_count": 20,
        "likes": 200,
        "retention_rate": 0.5
    })
    res_auto = predictor.calculate_single_post_score(row, denominator_mode="auto")
    res_views = predictor.calculate_single_post_score(row, denominator_mode="views")

    assert res_auto["denominator_used"] == "reach"
    assert res_views["denominator_used"] == "views"
    # Smaller denominator (reach = 5000) gives higher score than views = 10000
    assert res_auto["viral_coefficient"] > res_views["viral_coefficient"]


def test_retention_rate_boundaries():
    """Verify retention rate boundary enforcement."""
    df_invalid_ret = pd.DataFrame([{
        "platform": "instagram",
        "post_id": "P_INVALID_RET",
        "timestamp": "2026-10-01T12:00:00Z",
        "content_type": "reel",
        "views": 100,
        "impressions": 120,
        "reach": 80,
        "likes": 10,
        "comments_count": 2,
        "shares": 1,
        "saves": 1,
        "follower_count": 1000,
        "retention_rate": 1.75,  # Invalid: > 1.0
        "collection_timestamp": "2026-10-01T13:00:00Z"
    }])
    report = DataValidator.validate_unified_posts(df_invalid_ret)
    assert report["is_valid"] is False
    assert any("retention_rate" in err for err in report["errors"])

    # Cleaning clips out-of-bounds retention rate to 1.0
    cleaned = clean_posts_dataset(df_invalid_ret)
    assert cleaned.iloc[0]["retention_rate"] == 1.0


def test_minimum_training_samples_guard():
    """Verify that predictive ML model refuses training when under minimum sample threshold (50)."""
    predictor = ViralityPredictor()
    tiny_df = pd.DataFrame([
        {"post_id": f"P{i}", "views": 100, "shares": 5, "saves": 5, "comments_count": 1, "likes": 5, "viral_coefficient": 10.0}
        for i in range(10)
    ])
    with pytest.raises(ValueError, match="Insufficient historical data"):
        predictor.train_predictive_model(tiny_df)
