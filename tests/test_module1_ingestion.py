"""Integration and unit tests for Module 1: Content Performance Tracker."""

import pandas as pd

from src.data_ingestion.schemas import UNIFIED_POST_FIELDS
from src.data_ingestion.sample_ingestion import SampleDataIngestion
from src.data_ingestion.api_connectors import InstagramGraphAPIConnector, YouTubeDataAPIConnector
from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.storage import DataStorageManager
from src.preprocessing.cleaning import clean_posts_dataset, get_latest_posts_view
from src.preprocessing.feature_engineering import add_engagement_features


def test_sample_ingestion_loads_valid_schema():
    """Verify SampleDataIngestion produces a dataset with all 14 unified fields."""
    loader = SampleDataIngestion()
    df, provenance = loader.load_sample_posts()

    assert len(df) > 0
    assert provenance.access_tier == "sample_fixture"
    for field in UNIFIED_POST_FIELDS:
        assert field in df.columns

    report = DataValidator.validate_unified_posts(df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0


def test_api_connectors_contract():
    """Verify Instagram and YouTube connectors fulfill BaseAPIConnector contract."""
    ig = InstagramGraphAPIConnector()
    df_ig = ig.fetch_posts("dummy_account")
    for field in UNIFIED_POST_FIELDS:
        assert field in df_ig.columns
    assert ig.get_provenance().source_name == "Instagram Graph API"

    yt = YouTubeDataAPIConnector()
    df_yt = yt.fetch_posts("dummy_channel")
    for field in UNIFIED_POST_FIELDS:
        assert field in df_yt.columns
    assert yt.get_provenance().source_name == "YouTube Data API v3"


def test_cleaning_preserves_nullable_metrics():
    """Verify cleaning does NOT impute missing values to zero."""
    df_raw = pd.DataFrame([{
        "platform": "instagram",
        "post_id": "P_NULL_CHECK",
        "timestamp": "2026-10-01T12:00:00Z",
        "content_type": "reel",
        "views": 1000,
        "impressions": 1200,
        "reach": pd.NA,
        "likes": 50,
        "comments_count": 10,
        "shares": pd.NA,
        "saves": pd.NA,
        "follower_count": 5000,
        "retention_rate": pd.NA,
        "collection_timestamp": "2026-10-01T13:00:00Z"
    }])

    cleaned = clean_posts_dataset(df_raw)
    assert pd.isna(cleaned.iloc[0]["reach"])
    assert pd.isna(cleaned.iloc[0]["shares"])
    assert pd.isna(cleaned.iloc[0]["saves"])
    assert pd.isna(cleaned.iloc[0]["retention_rate"])
    assert cleaned.iloc[0]["views"] == 1000
    assert cleaned.iloc[0]["impressions"] == 1200
    assert cleaned.iloc[0]["likes"] == 50


def test_historical_snapshots_survive_cleaning():
    """Verify multiple snapshots of the same (platform, post_id) survive cleaning."""
    df_history = pd.DataFrame([
        {
            "platform": "instagram",
            "post_id": "P_HIST_1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 100,
            "likes": 10,
            "collection_timestamp": "2026-10-01T14:00:00Z"
        },
        {
            "platform": "instagram",
            "post_id": "P_HIST_1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 850,
            "likes": 80,
            "collection_timestamp": "2026-10-02T14:00:00Z"
        }
    ])

    cleaned = clean_posts_dataset(df_history)
    # Both snapshots MUST survive cleaning
    assert len(cleaned) == 2

    # Latest record view selects strictly the most recent observation
    latest_view = get_latest_posts_view(cleaned)
    assert len(latest_view) == 1
    assert latest_view.iloc[0]["views"] == 850
    assert latest_view.iloc[0]["likes"] == 80


def test_sqlite_snapshot_retention_and_latest_view(tmp_path):
    """Verify DataStorageManager retains snapshots and creates the latest-record view."""
    storage = DataStorageManager(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        db_path=tmp_path / "test_hist.db"
    )

    df_test = pd.DataFrame([
        {
            "platform": "instagram",
            "post_id": "P_HIST_SQL",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 200,
            "likes": 15,
            "collection_timestamp": "2026-10-01T13:00:00Z"
        },
        {
            "platform": "instagram",
            "post_id": "P_HIST_SQL",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 1200,
            "likes": 95,
            "collection_timestamp": "2026-10-02T13:00:00Z"
        }
    ])

    persisted_count = storage.persist_to_sqlite(df_test, table_name="posts")
    assert persisted_count == 2

    # All snapshots in base table
    all_rows = storage.load_from_sqlite(table_name="posts")
    assert len(all_rows) == 2

    # Latest view returns strictly the latest record
    latest_rows = storage.load_latest_from_sqlite(table_name="posts")
    assert len(latest_rows) == 1
    assert int(latest_rows.iloc[0]["views"]) == 1200

    # Longitudinal history query
    hist_post = storage.load_history_for_post("instagram", "P_HIST_SQL", table_name="posts")
    assert len(hist_post) == 2


def test_feature_engineering_explicit_zeros_and_derivations():
    """Verify feature engineering handles zero shares, zero likes, and distinct rates."""
    df = pd.DataFrame([
        # Post 1: Valid metrics
        {
            "platform": "instagram",
            "content_type": "reel",
            "views": 1000,
            "reach": 800,
            "likes": 100,
            "comments_count": 20,
            "shares": 50,
            "saves": 100,
            "retention_rate": 0.60
        },
        # Post 2: Zero shares (should NOT substitute 1; status undefined_zero_shares)
        {
            "platform": "instagram",
            "content_type": "reel",
            "views": 1000,
            "reach": 800,
            "likes": 100,
            "comments_count": 20,
            "shares": 0,
            "saves": 50,
            "retention_rate": 0.50
        },
        # Post 3: Zero likes (should NOT substitute 1; status undefined_zero_likes)
        {
            "platform": "instagram",
            "content_type": "reel",
            "views": 1000,
            "reach": 800,
            "likes": 0,
            "comments_count": 20,
            "shares": 10,
            "saves": 20,
            "retention_rate": 0.50
        }
    ])

    feat = add_engagement_features(df)

    # Post 1 checks
    assert feat.iloc[0]["high_intent_actions"] == 150
    assert feat.iloc[0]["high_intent_completeness"] == "complete"
    assert feat.iloc[0]["save_to_share_ratio"] == 2.0
    assert feat.iloc[0]["save_share_ratio_status"] == "valid_ratio"
    assert feat.iloc[0]["views_engagement_rate"] == 0.27  # (100+20+50+100) / 1000
    assert feat.iloc[0]["reach_engagement_rate"] == 0.3375  # (100+20+50+100) / 800
    assert feat.iloc[0]["equivalent_full_watch_views"] == 600.0  # 1000 * 0.60

    # Post 2 zero shares check
    assert pd.isna(feat.iloc[1]["save_to_share_ratio"])
    assert feat.iloc[1]["save_share_ratio_status"] == "undefined_zero_shares"

    # Post 3 zero likes check
    assert pd.isna(feat.iloc[2]["intent_to_passive_ratio"])
    assert feat.iloc[2]["intent_passive_ratio_status"] == "undefined_zero_likes"


def test_data_completeness_instagram_vs_youtube():
    """Verify data completeness denominator adapts to applicable platform metrics."""
    df_yt = pd.DataFrame([{
        "platform": "youtube",
        "content_type": "short",
        "views": 5000,
        "impressions": 6000,
        "reach": 4000,
        "likes": 200,
        "comments_count": 30,
        "shares": 80,
        "saves": pd.NA,  # Inapplicable on YouTube
        "follower_count": 10000,
        "retention_rate": 0.65
    }])

    feat_yt = add_engagement_features(df_yt, access_tier="owner_oauth")
    # On YouTube, saves is structurally not applicable and NOT counted against completeness
    assert feat_yt.iloc[0]["data_completeness_pct"] == 100.0
    assert feat_yt.iloc[0]["high_intent_completeness"] == "partial_shares_only"
