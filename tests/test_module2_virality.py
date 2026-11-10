"""Comprehensive unit and integration test suite for Module 2: Virality Prediction Engine."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from src.modeling.virality_predictor import (
    ViralityPredictor,
    PRE_PUBLICATION_FEATURES,
    OUTCOME_METRICS
)


def create_mock_historical_data(n_samples: int = 60) -> pd.DataFrame:
    """Helper to generate mock chronological post data for testing."""
    start_time = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    topics = ["Social Anxiety", "Career Burnout", "Dating & Rejection"]
    platforms = ["instagram", "youtube"]
    content_types = ["reel", "short", "carousel"]
    hooks = ["Visual Hook", "Text Hook", "Story Hook"]

    rows = []
    for i in range(n_samples):
        ts = start_time + timedelta(days=i)
        platform = platforms[i % len(platforms)]
        c_type = content_types[i % len(content_types)]
        topic = topics[i % len(topics)]
        views = 5000 + i * 50
        reach = int(views * 0.8)
        likes = 200 + i * 5
        shares = 50 + i * 2
        saves = 60 + i * 2 if platform == "instagram" else np.nan
        retention = 0.65 if c_type in ["reel", "short"] else np.nan

        # Set collection timestamp 14 days later so all mock posts are mature (>= 7 days)
        coll_ts = ts + timedelta(days=14)

        rows.append({
            "platform": platform,
            "post_id": f"POST_{i:04d}",
            "timestamp": ts.isoformat(),
            "content_type": c_type,
            "topic": topic,
            "hook_type": hooks[i % len(hooks)],
            "duration_seconds": 30 if c_type != "carousel" else 0,
            "posting_hour": 18,
            "posting_day_of_week": ts.strftime("%A"),
            "follower_count": 35000 + i * 100,
            "views": views,
            "impressions": int(views * 1.15),
            "reach": reach,
            "likes": likes,
            "comments_count": 25,
            "shares": shares,
            "saves": saves,
            "retention_rate": retention,
            "collection_timestamp": coll_ts.isoformat()
        })
    return pd.DataFrame(rows)


def test_pre_publication_feature_restrictions():
    """Verify that canonical pre-publication features contain NO post-publication outcome metrics."""
    for feat in PRE_PUBLICATION_FEATURES:
        assert feat not in OUTCOME_METRICS, f"Feature '{feat}' is an outcome metric and must not be pre-publication!"


def test_target_leakage_guard_raises_error():
    """Verify that passing outcome metrics into prediction raises a ValueError."""
    predictor = ViralityPredictor()
    df_leak = pd.DataFrame([{
        "platform": "instagram",
        "topic": "Social Anxiety",
        "views": 10000  # Outcome metric!
    }])
    with pytest.raises(ValueError, match="Target leakage detected"):
        predictor.predict_pre_publication(df_leak)


def test_chronological_train_test_splitting():
    """Verify temporal train/test split strictly partitions past from future."""
    df = create_mock_historical_data(n_samples=60)
    train_df, test_df = ViralityPredictor.temporal_train_test_split(df, test_ratio=0.2)

    assert len(train_df) == 48
    assert len(test_df) == 12

    max_train_ts = pd.to_datetime(train_df["timestamp"]).max()
    min_test_ts = pd.to_datetime(test_df["timestamp"]).min()
    assert max_train_ts < min_test_ts, "Lookahead leakage! Train timestamps overlap with test timestamps."


def test_immature_posts_filtered_from_training():
    """Verify posts with < 7 days elapsed between publication and collection are excluded from training."""
    now_utc = datetime.now(timezone.utc)
    df = create_mock_historical_data(n_samples=60)

    # Make last 5 posts immature (< 7 days)
    for idx in range(55, 60):
        df.loc[idx, "timestamp"] = (now_utc - timedelta(days=3)).isoformat()
        df.loc[idx, "collection_timestamp"] = now_utc.isoformat()

    predictor = ViralityPredictor()
    prepared = predictor.prepare_dataset_for_modeling(df, min_elapsed_days=7.0)
    assert len(prepared) == 55
    # Assert all prepared posts have >= 7 elapsed days
    assert (prepared["elapsed_days"] >= 7.0).all()


def test_insufficient_training_data_guard():
    """Verify model refuses training when unique mature records are under the minimum threshold (50)."""
    predictor = ViralityPredictor()
    df_small = create_mock_historical_data(n_samples=25)

    with pytest.raises(ValueError, match="Insufficient historical data"):
        predictor.train_and_evaluate(df_small, model_type="ridge")


def test_missing_input_features_handling():
    """Verify imputer and one-hot encoder gracefully handle missing or unseen values during inference."""
    df = create_mock_historical_data(n_samples=60)
    predictor = ViralityPredictor()
    predictor.train_and_evaluate(df, model_type="ridge")

    # Input with missing optional values and unseen category
    draft = pd.DataFrame([{
        "platform": "instagram",
        "content_type": "reel",
        "topic": "Unseen New Topic",  # Unseen category
        "hook_type": "Visual Hook",
        "duration_seconds": np.nan,    # Missing numeric
        "posting_hour": 18,
        "posting_day_of_week": "Friday",
        "follower_count": np.nan       # Missing numeric
    }])

    res = predictor.predict_pre_publication(draft)
    assert len(res) == 1
    assert pd.notna(res.iloc[0]["predicted_engagement_virality_index"])
    assert res.iloc[0]["predicted_engagement_virality_index"] > 0
    assert "approximate_error_band" in res.columns
    assert res.iloc[0]["error_band_type"] == "empirical_residual_dispersion_band"


def test_baseline_comparison_structure():
    """Verify train_and_evaluate produces structured comparison against DummyRegressor baseline."""
    df = create_mock_historical_data(n_samples=60)
    predictor = ViralityPredictor()
    meta = predictor.train_and_evaluate(df, model_type="ridge", test_ratio=0.2)

    assert "baseline_metrics" in meta
    assert "candidate_metrics" in meta
    assert "mae" in meta["baseline_metrics"]
    assert "rmse" in meta["baseline_metrics"]
    assert "mae" in meta["candidate_metrics"]
    assert "spearman_rho" in meta["candidate_metrics"]
    assert "mae_lift_pct" in meta
    assert "rmse_lift_pct" in meta
    assert meta["dataset_provenance"] == "synthetic_illustrative_benchmark"


def test_model_serialization_and_loading(tmp_path):
    """Verify joblib model pipeline and JSON metadata serialization and loading."""
    df = create_mock_historical_data(n_samples=60)
    predictor = ViralityPredictor()
    predictor.train_and_evaluate(df, model_type="ridge")

    model_file = tmp_path / "virality_model.joblib"
    meta_file = tmp_path / "virality_meta.json"

    predictor.save_model_pipeline(model_file, meta_file)
    assert model_file.exists()
    assert meta_file.exists()

    new_predictor = ViralityPredictor()
    new_predictor.load_model_pipeline(model_file, meta_file)
    assert new_predictor.model is not None
    assert new_predictor.metadata["model_type"] == "ridge"

    draft = pd.DataFrame([{
        "platform": "youtube",
        "content_type": "short",
        "topic": "Career Burnout",
        "hook_type": "Story Hook",
        "duration_seconds": 45,
        "posting_hour": 12,
        "posting_day_of_week": "Sunday",
        "follower_count": 40000
    }])
    preds = new_predictor.predict_pre_publication(draft)
    assert len(preds) == 1
    assert preds.iloc[0]["predicted_engagement_virality_index"] > 0


def test_within_platform_target_normalization():
    """Verify platform-specific target normalization calculates percentiles within each platform."""
    predictor = ViralityPredictor()
    df = pd.DataFrame([
        {"platform": "instagram", "post_id": "P1", "views": 1000, "shares": 20, "likes": 50},
        {"platform": "instagram", "post_id": "P2", "views": 1000, "shares": 100, "likes": 50},
        {"platform": "youtube", "post_id": "P3", "views": 5000, "shares": 10, "likes": 20},
        {"platform": "youtube", "post_id": "P4", "views": 5000, "shares": 50, "likes": 20}
    ])
    pcts = predictor.normalize_target_by_platform(df)
    assert len(pcts) == 4
    # P2 is top of Instagram, P4 is top of YouTube
    assert pcts.iloc[1] == 100.0
    assert pcts.iloc[3] == 100.0
