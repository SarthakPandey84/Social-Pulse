"""Regression and parity tests for Phase 3 optimizations & cache invalidation safety."""

from pathlib import Path
import warnings
import pandas as pd

from src.data_ingestion.sample_ingestion import SampleDataIngestion
from src.nlp.text_preprocessor import TextPreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_text_preprocessor_contractions_and_normalization_parity():
    """Verify compiled TextPreprocessor rules match expected contraction expansion and edge guards."""
    prep = TextPreprocessor()

    res = prep.preprocess("I don't think you're wrong!!! Can't wait???")
    assert res.is_valid is True
    assert res.cleaned_text == "I don't think you're wrong!! Can't wait??"
    assert "do not" in res.normalized_tokens
    assert "you are" in res.normalized_tokens
    assert "cannot" in res.normalized_tokens

    # Edge guards must still catch insufficient context and empty strings without regression
    assert prep.preprocess("ok").is_valid is False
    assert prep.preprocess("🔥🔥").is_valid is False
    assert prep.preprocess("   ").is_valid is False
    assert prep.preprocess(None).is_valid is False


def test_sample_ingestion_null_safe_impressions_parity(tmp_path):
    """Verify vectorized null-safe Int64 impressions calculation handles both valid integers and <NA>."""
    csv_file = tmp_path / "custom_posts.csv"
    pd.DataFrame([
        {
            "post_id": "P1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 100,
            "likes": 10,
            "comments_count": 2,
            "shares": 3,
            "saves": 4,
            "retention_rate": 0.6,
        },
        {
            "post_id": "P2",
            "timestamp": "2026-10-02T12:00:00Z",
            "content_type": "carousel",
            "views": pd.NA,
            "likes": 5,
            "comments_count": 1,
            "shares": 1,
            "saves": 2,
            "retention_rate": pd.NA,
        },
    ]).to_csv(csv_file, index=False)

    loader = SampleDataIngestion(sample_dir=tmp_path)
    df, _ = loader.load_sample_posts(filename="custom_posts.csv")
    assert str(df["impressions"].dtype) == "Int64"
    assert df.loc[0, "impressions"] == int(100 * 1.15)
    assert pd.isna(df.loc[1, "impressions"])


def test_page7_cache_invalidation_and_zero_deprecation_warnings():
    """Verify Page 7 responds accurately to parameter changes (no stale cache) and emits no use_container_width warnings."""
    from streamlit.testing.v1 import AppTest

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at = AppTest.from_file(str(REPO_ROOT / "app" / "pages" / "07_Trend_Forecasting.py"), default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        # Switch horizon slider 7 -> 10: forecast detail table must reflect 10 steps, not stale 7
        slider = next(s for s in at.sidebar.slider if "Horizon" in s.label)
        slider.set_value(10).run()
        assert len(at.exception) == 0
        forecast_df = next(d.value for d in at.dataframe if "Horizon Step" in d.value.columns)
        assert len(forecast_df) == 10

        # Switch model selector: Automatic -> Naive (Persistence) (distinct non-champion override)
        model_sb = next(sb for sb in at.sidebar.selectbox if "Model" in sb.label)
        model_sb.set_value("Naive (Persistence)").run()
        assert len(at.exception) == 0
        assert any("Naive (Persistence)" in w.value for w in at.warning)

        # Switch domain: Publication-Cohort Performance -> Struggle Keyword Trajectories
        domain_radio = next(r for r in at.sidebar.radio if "Domain" in r.label)
        domain_radio.set_value("Struggle Keyword Trajectories").run()
        assert len(at.exception) == 0
        assert any("Keyword Observed Calendar-Date Continuity" in elem.value for elem in at.info)

    ucw_warnings = [w for w in caught if "use_container_width" in str(w.message)]
    assert len(ucw_warnings) == 0, f"Unexpected use_container_width warnings: {ucw_warnings}"


def test_home_page_loads_cleanly_without_exceptions():
    """Verify cached Home entrypoint renders SocialPulse KPIs and 7 module cards without exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    rendered_html = "\n".join(m.value for m in at.markdown)
    assert "SocialPulse" in rendered_html
    assert "sp-module-card" in rendered_html
    assert rendered_html.count("MODULE 0") == 7


def test_vectorized_feature_engineering_and_evi_rowwise_parity():
    """Verify vectorized add_engagement_features and EVI match strict scalar row-by-row semantics."""
    import numpy as np
    from src.data_ingestion.schemas import get_applicable_metrics
    from src.modeling.virality_predictor import ViralityPredictor
    from src.preprocessing.feature_engineering import add_engagement_features

    df = pd.DataFrame([
        {
            "post_id": "P_IG_REEL",
            "platform": "instagram",
            "content_type": "Reel",
            "views": 1000,
            "reach": 850,
            "impressions": 1150,
            "likes": 120,
            "comments_count": 15,
            "shares": 25,
            "saves": 40,
            "retention_rate": 0.62,
            "follower_count": 5000,
        },
        {
            "post_id": "P_IG_CAROUSEL",
            "platform": "instagram",
            "content_type": "Carousel",
            "views": pd.NA,
            "reach": 500,
            "impressions": 700,
            "likes": 50,
            "comments_count": 5,
            "shares": 0,
            "saves": 12,
            "retention_rate": 0.75,  # Non-video static format must remain phi=1.0 and N/A effective_watch_views
            "follower_count": 5000,
        },
        {
            "post_id": "P_YT_SHORT",
            "platform": "youtube",
            "content_type": "Short Video",
            "views": 2500,
            "reach": pd.NA,
            "impressions": pd.NA,
            "likes": 0,
            "comments_count": 3,
            "shares": 10,
            "saves": pd.NA,
            "retention_rate": 0.85,
        },
        {
            "post_id": "P_TK_VIDEO",
            "platform": "tiktok",
            "content_type": "Short Video",
            "views": 0,
            "reach": pd.NA,
            "impressions": pd.NA,
            "likes": 4,
            "comments_count": 1,
            "shares": 0,
            "saves": pd.NA,
            "retention_rate": pd.NA,
        },
        {
            "post_id": "P_UNMEASURED",
            "platform": "instagram",
            "content_type": "Reel",
            "views": pd.NA,
            "reach": pd.NA,
            "impressions": pd.NA,
            "likes": 10,
            "comments_count": pd.NA,
            "shares": pd.NA,
            "saves": pd.NA,
            "retention_rate": 1.25,  # Out-of-range retention_rate falls back to neutral phi=1.0
        },
    ])

    feat_df = add_engagement_features(df)

    # Row 0 (IG Reel)
    assert int(feat_df.loc[0, "high_intent_actions"]) == 65
    assert feat_df.loc[0, "high_intent_completeness"] == "complete"
    assert float(feat_df.loc[0, "save_to_share_ratio"]) == 1.60
    assert feat_df.loc[0, "save_share_ratio_status"] == "valid_ratio"
    assert float(feat_df.loc[0, "intent_to_passive_ratio"]) == round(65 / 120, 3)
    assert float(feat_df.loc[0, "views_engagement_rate"]) == round(200 / 1000, 4)
    assert float(feat_df.loc[0, "reach_engagement_rate"]) == round(200 / 850, 4)
    assert float(feat_df.loc[0, "equivalent_full_watch_views"]) == 620.0
    assert feat_df.loc[0, "metric_unavailability_audit"] == "all_applicable_observed"

    # Row 1 (IG Carousel): 0 shares & +saves -> undefined_zero_shares, static format -> not_applicable_non_video
    assert pd.isna(feat_df.loc[1, "save_to_share_ratio"])
    assert feat_df.loc[1, "save_share_ratio_status"] == "undefined_zero_shares"
    assert pd.isna(feat_df.loc[1, "equivalent_full_watch_views"])
    assert feat_df.loc[1, "equivalent_full_watch_status"] == "not_applicable_non_video"
    app_carousel = get_applicable_metrics("instagram", "Carousel", "owner_oauth")
    assert float(feat_df.loc[1, "data_completeness_pct"]) == round((len(app_carousel) / len(app_carousel)) * 100.0, 1)

    # Row 2 (YT Short): partial_shares_only + 0 likes -> undefined_zero_likes, no saves/reach
    assert int(feat_df.loc[2, "high_intent_actions"]) == 10
    assert feat_df.loc[2, "high_intent_completeness"] == "partial_shares_only"
    assert feat_df.loc[2, "intent_passive_ratio_status"] == "undefined_zero_likes"
    assert "saves:not_applicable_youtube" in feat_df.loc[2, "metric_unavailability_audit"]

    # Row 3 (TikTok 0 views): views_engagement_rate must remain <NA> (never inf or 0.0)
    assert pd.isna(feat_df.loc[3, "views_engagement_rate"])

    # Vectorized EVI vs element-wise calculate_single_post_score exact parity
    predictor = ViralityPredictor()
    evi_df = predictor.calculate_engagement_virality_index_detailed(df)
    for idx in range(len(df)):
        ref = predictor.calculate_single_post_score(df.iloc[idx])
        got_evi = evi_df.loc[idx, "engagement_virality_index"]
        if pd.isna(ref["engagement_virality_index"]):
            assert pd.isna(got_evi), f"Row {idx} expected NaN EVI, got {got_evi}"
        else:
            assert np.isclose(float(got_evi), float(ref["engagement_virality_index"]), atol=1e-9)
        assert evi_df.loc[idx, "metrics_contributed"] == ref["metrics_contributed"]
        assert float(evi_df.loc[idx, "weight_coverage"]) == float(ref["weight_coverage"])
        assert evi_df.loc[idx, "denominator_used"] == ref["denominator_used"]
        assert evi_df.loc[idx, "confidence_tier"] == ref["confidence_tier"]
