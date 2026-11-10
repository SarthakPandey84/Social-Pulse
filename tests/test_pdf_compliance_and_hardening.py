"""Regression tests for PDF specification compliance, analytical hardening, governance tiers, and Deliverables 4–5."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.ui_widgets import format_dataset_option_label, render_provenance_banner
from scripts.generate_strategy_report import generate_strategy_report
from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.data_ingestion.api_connectors import InstagramGraphAPIConnector, SocialMediaScraper
from src.data_ingestion.external_adapters import _write_dataset_meta
from src.data_ingestion.external_datasets import (
    DATASET_REGISTRY,
    GOVERNANCE_TIERS,
    classify_governance_tier,
)
from src.nlp.evaluation import evaluate_heldout_instagram_sentiment
from src.recommender.engine import ContentRecommender, OptimizationRecommender


def test_env_example_and_instagram_connector_alignment(monkeypatch):
    """Verify .env.example and InstagramGraphAPIConnector synchronize META_* and YOUTUBE_CHANNEL_ID cleanly."""
    env_text = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    for expected_key in (
        "META_ACCESS_TOKEN=",
        "META_IG_BUSINESS_ACCOUNT_ID=",
        "META_GRAPH_API_VERSION=v21.0",
        "YOUTUBE_API_KEY=",
        "YOUTUBE_CHANNEL_ID=",
    ):
        assert expected_key in env_text

    # 1. Default without overrides uses v21.0
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("META_IG_BUSINESS_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("META_GRAPH_API_VERSION", raising=False)
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("INSTAGRAM_PAGE_ID", raising=False)
    default_conn = InstagramGraphAPIConnector()
    assert default_conn.base_url == "https://graph.facebook.com/v21.0"
    assert default_conn.sandbox_mode is True

    # 2. Canonical META_* variables take precedence over legacy INSTAGRAM_* fallbacks
    monkeypatch.setenv("META_ACCESS_TOKEN", "meta-tok")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "legacy-tok")
    monkeypatch.setenv("META_IG_BUSINESS_ACCOUNT_ID", "meta-ig-id")
    monkeypatch.setenv("INSTAGRAM_PAGE_ID", "legacy-page-id")
    monkeypatch.setenv("META_GRAPH_API_VERSION", "22.0")
    meta_conn = InstagramGraphAPIConnector()
    assert meta_conn.access_token == "meta-tok"
    assert meta_conn.page_id == "meta-ig-id"
    assert meta_conn.base_url == "https://graph.facebook.com/v22.0"
    assert meta_conn.sandbox_mode is False

    # 3. Placeholder scraper contract remains unchanged
    scraper = SocialMediaScraper()
    assert scraper.scrape_post_metrics("https://example.com")["status"] == "placeholder"


def test_nlp_evaluation_overlap_count_is_dynamic():
    """Verify cross-split duplicate overlap count in limitations is computed dynamically (never hardcoded)."""
    rows = []
    # Train: 18 unique balanced comments
    for i in range(6):
        rows.append({"eval_split": "train", "comment_text": f"I genuinely love this workflow example {i}", "ground_truth_sentiment": "Positive"})
        rows.append({"eval_split": "train", "comment_text": f"Honestly this approach is awful and broken {i}", "ground_truth_sentiment": "Negative"})
        rows.append({"eval_split": "train", "comment_text": f"Here is a timestamp note for clip {i}", "ground_truth_sentiment": "Neutral"})
    # Test: 2 exact overlap duplicates with train + 4 unique test comments
    rows.extend([
        {"eval_split": "test", "comment_text": "I genuinely love this workflow example 0", "ground_truth_sentiment": "Positive"},
        {"eval_split": "test", "comment_text": "Honestly this approach is awful and broken 0", "ground_truth_sentiment": "Negative"},
        {"eval_split": "test", "comment_text": "Unique positive sentiment validation string", "ground_truth_sentiment": "Positive"},
        {"eval_split": "test", "comment_text": "Unique negative terrible experience string", "ground_truth_sentiment": "Negative"},
        {"eval_split": "test", "comment_text": "Neutral documentation reference note", "ground_truth_sentiment": "Neutral"},
        {"eval_split": "test", "comment_text": "Another clear joyful helpful recommendation", "ground_truth_sentiment": "Positive"},
    ])
    rep = evaluate_heldout_instagram_sentiment(pd.DataFrame(rows))
    assert rep["train_samples"] == 18
    assert rep["test_samples_full"] == 6
    assert rep["cross_split_duplicate_texts_in_test"] == 2
    assert rep["test_samples_strict_unique"] == 4
    assert "N=18 train / 6 test" in rep["limitations"][0]
    assert "with 2 exact text duplicates" in rep["limitations"][0]

    # Verify checked real_data_benchmarks.json if present matches dynamically
    bench_path = ROOT_DIR / "models" / "real_data_benchmarks.json"
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        ig_sent = (bench.get("modules") or {}).get("module3", {}).get("instagram_sentiment_heldout")
        if ig_sent:
            overlap = ig_sent["cross_split_duplicate_texts_in_test"]
            assert f"with {overlap} exact text duplicates" in ig_sent["limitations"][0]
            assert "with 3 exact text duplicates" not in ig_sent["limitations"][0]


def test_real_data_benchmarks_sponsored_ig_schema_projection():
    """Verify real_data_benchmarks.json projects follower_count and provenance for sponsored IG."""
    bench_path = ROOT_DIR / "models" / "real_data_benchmarks.json"
    if not bench_path.is_file():
        pytest.skip("real_data_benchmarks.json not built")
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    sp_avail = (bench.get("modules") or {}).get("module1", {}).get("sponsored_ig_metric_availability", {})
    if sp_avail:
        sp_fc = (sp_avail.get("metrics") or {}).get("follower_count", {})
        assert sp_fc["present_in_schema"] is True
        assert sp_fc["observed_count"] > 0
        assert len(sp_avail.get("provenances", [])) >= 1


def test_module5_real_corpus_wording_retention_scale_and_missing_views_guards():
    """Verify M5 labels non-synthetic corpora properly, scales [0,1] retention to %, and avoids views=1.0 distortion."""
    posts_df = pd.DataFrame([
        {
            "platform": "instagram",
            "post_id": f"real_{i}",
            "timestamp": f"2026-09-0{(i % 6) + 1}T12:00:00Z",
            "collection_timestamp": "2026-10-01T00:00:00Z",
            "content_type": "reel",
            "topic": "Visual Art" if i < 6 else "Sketching",
            "hook_type": "Visual Hook",
            "duration_seconds": 35 if i < 6 else 70,
            "posting_hour": 18,
            "views": pd.NA,
            "likes": 200 + i * 10,
            "comments_count": 20,
            "shares": 30 + i,
            "saves": pd.NA,
            "retention_rate": 0.77 if i < 6 else 0.52,
            "creator_pseudo_id": f"cr_{i}",
            "provenance": "real_public_data",
        }
        for i in range(12)
    ])
    comments_df = pd.DataFrame([
        {
            "post_id": f"real_{i}",
            "comment_id": f"c_{i}",
            "comment_text": "I felt this so much honestly!",
            "sentiment_label": "Positive",
            "relatability_label": "Relatable",
            "provenance": "real_public_data",
        }
        for i in range(12)
    ])

    rec_engine = OptimizationRecommender(posts_df=posts_df, comments_df=comments_df, experiments_df=None)
    assert rec_engine._is_synthetic_demo is False
    report = rec_engine.generate_recommendations()
    by_id = {r.recommendation_id: r for r in report.recommendations}

    # 1. NLP evidence must not mislabel public corpus as Synthetic Comment Corpus
    assert "Synthetic Comment Corpus" not in by_id["REC_NLP_001"].supporting_evidence
    assert "Analyzed Public Comment Corpus" in by_id["REC_NLP_001"].supporting_evidence

    # 2. Missing views must not silently compute intent_density by dividing by 1.0
    assert "REC_TOPIC_001" in by_id
    assert "intent density per view = N/A" in by_id["REC_TOPIC_001"].supporting_evidence
    assert "Intent density =" not in by_id["REC_TOPIC_001"].supporting_evidence

    # 3. [0, 1] retention must scale to percentage (77.0%, never 0.8%)
    assert "REC_FORMAT_001" in by_id
    assert "77.0%" in by_id["REC_FORMAT_001"].reasoning
    assert "0.8%" not in by_id["REC_FORMAT_001"].reasoning

    # 4. ContentRecommender.recommend_weekly_strategy preserves NaN when likes/shares/saves are all NA
    na_df = pd.DataFrame([
        {
            "post_id": "p1",
            "topic": "Focus",
            "views": pd.NA,
            "likes": pd.NA,
            "comments_count": pd.NA,
            "shares": pd.NA,
            "saves": pd.NA,
        }
    ])
    strat = ContentRecommender().recommend_weekly_strategy(na_df)
    assert strat["recommended_topics"] == {}


def test_module6_backend_preserves_nan_and_scales_retention():
    """Verify M6 keeps absent metrics as NaN, scales [0,1] retention %, and omits missing-saves correlation."""
    posts_df = pd.DataFrame([
        {
            "platform": "instagram",
            "post_id": f"ia_{i}",
            "timestamp": f"2026-09-0{(i % 5) + 1}T10:00:00Z",
            "collection_timestamp": "2026-10-01T00:00:00Z",
            "content_type": "reel",
            "topic": "Art",
            "hook_type": "Visual",
            "duration_seconds": 35,
            "posting_hour": 14,
            "views": pd.NA,
            "impressions": pd.NA,
            "reach": pd.NA,
            "likes": 120 + i,
            "comments_count": 10,
            "shares": 15,
            "saves": pd.NA,
            "retention_rate": 0.77,
            "provenance": "real_public_data",
        }
        for i in range(12)
    ])
    comments_df = pd.DataFrame([
        {
            "post_id": f"ia_{i}",
            "comment_id": f"ic_{i}",
            "comment_text": "Relatable moment",
            "sentiment_label": "Positive",
            "relatability_label": "Relatable",
        }
        for i in range(12)
    ])

    backend = GrowthDashboardBackend(posts_df=posts_df, comments_df=comments_df, dataset_id="instaart_human_ai")

    ts = backend.calculate_time_series_trends(posts_df, freq="W")
    assert ts["views"].isna().all()
    assert ts["saves"].isna().all()
    assert ts["views_engagement_rate"].isna().all()
    assert ts["share_rate"].isna().all()
    assert ts["save_rate"].isna().all()

    pts = backend.calculate_platform_time_series(posts_df, freq="W")
    assert pts["views"].isna().all()
    assert pts["engagement_rate"].isna().all()

    comp = backend.calculate_platform_comparisons(posts_df)["summary_table"].iloc[0]
    assert np.isnan(comp["Total Views"])
    assert np.isnan(comp["Avg Views/Post"])
    assert np.isnan(comp["Normalized Eng Rate (%)"])
    assert np.isnan(comp["Full Eng Rate (%)"])
    assert np.isnan(comp["Share Rate (%)"])
    assert np.isnan(comp["Save Rate (%)"])
    assert comp["Avg Retention (%)"] == 77.0
    assert "Views" not in comp["Available Fields"]
    assert "Saves" not in comp["Available Fields"]
    assert "Likes" in comp["Available Fields"]

    breakdowns = backend.calculate_content_breakdowns(posts_df)
    assert float(breakdowns["by_duration"]["avg_retention"].dropna().iloc[0]) == pytest.approx(77.0)
    assert float(breakdowns["by_topic"]["avg_retention"].dropna().iloc[0]) == pytest.approx(77.0)

    aud = backend.calculate_audience_insights()
    assert aud["has_data"] is True
    assert aud["correlation_r"] is None
    assert aud["correlation_rho"] is None
    assert aud["correlation_p"] is None
    assert "synthetic dataset" not in aud["correlation_text"].lower()


def test_governance_tiers_taxonomy_metadata_and_ui_banners(tmp_path):
    """Verify standardized 6-tier governance taxonomy across registry, metadata writer, and UI formatters."""
    for ds_id, spec in DATASET_REGISTRY.items():
        tier = classify_governance_tier(spec)
        assert tier in GOVERNANCE_TIERS, ds_id
        summary = spec.to_summary_dict()
        assert summary["governance_tier"] == tier

    assert classify_governance_tier(DATASET_REGISTRY["zenodo_sponsored_unsponsored_ig"]) == "VERIFIED_COMMERCIAL_SAFE"
    assert classify_governance_tier(DATASET_REGISTRY["tiktok_engagement_200k"]) == "NON_COMMERCIAL_RESEARCH_ONLY"
    assert classify_governance_tier(DATASET_REGISTRY["instagram_comments_sentiment_hf"]) == "PENDING_REVIEW_WEAKLY_LABELLED"
    assert classify_governance_tier(DATASET_REGISTRY["instagram_analytics_github_alt"]) == "EXTERNAL_BENCHMARK_UNVERIFIED"
    assert classify_governance_tier(DATASET_REGISTRY["social_instagram_marketing_hf"]) == "SYNTHETIC_BENCHMARK"
    assert classify_governance_tier(DATASET_REGISTRY["youtube_trending_kaggle"]) == "BLOCKED_MISSING_CREDENTIALS"

    meta_out = _write_dataset_meta(
        tmp_path / "tiktok_engagement_200k",
        {"dataset_id": "tiktok_engagement_200k", "display_name": "TikTok 200K"},
    )
    loaded = json.loads(meta_out.read_text(encoding="utf-8"))
    assert loaded["governance_tier"] == "NON_COMMERCIAL_RESEARCH_ONLY"

    label = format_dataset_option_label({
        "dataset_id": "tiktok_engagement_200k",
        "display_name": "TikTok 200K",
        "provenance": "real_public_data",
        "governance_tier": "NON_COMMERCIAL_RESEARCH_ONLY",
        "row_counts": {"posts": 209543},
    })
    assert "Research-Only" in label
    assert callable(render_provenance_banner)


def test_page02_tab2_surfaces_platform_evaluation_coverage():
    """Verify Page 02 Tab 2 displays Instagram test_samples=0 and negative pooled R² warnings cleanly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "02_Virality_Prediction.py"), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    subheaders = "\n".join(s.value for s in at.subheader)
    assert "Platform-Stratified Temporal Evaluation Coverage" in subheaders
    warnings_text = "\n".join(w.value for w in at.warning)
    assert "insufficient_evaluation_samples" in warnings_text
    assert "test_samples = 0" in warnings_text
    assert "R² = -0.136" in warnings_text


def test_deliverables_4_and_5_exist_and_strategy_report_is_deterministic(tmp_path):
    """Verify Deliverable 5 template exists and Deliverable 4 is byte-identical across clean & populated roots."""
    import shutil

    template_path = ROOT_DIR / "docs" / "CONTENT_SERIES_TEMPLATE.md"
    assert template_path.is_file()
    tmpl = template_path.read_text(encoding="utf-8")
    for required in ("PILOT_EXP_01", "collection_timestamp", "retention_rate", "saves", "collect_count"):
        assert required in tmpl

    out1 = tmp_path / "r1.md"
    out2 = tmp_path / "r2.md"
    generate_strategy_report(root_dir=ROOT_DIR, output_path=out1)
    generate_strategy_report(root_dir=ROOT_DIR, output_path=out2)
    b1 = out1.read_bytes()
    b2 = out2.read_bytes()
    assert b1 == b2

    # Build an isolated clean root containing ONLY tracked inputs (zero gitignored external/processed files)
    clean_root = tmp_path / "clean_repo"
    for rel in (
        "models/real_data_benchmarks.json",
        "models/virality_predictor_metadata.json",
        "models/relatability_metadata.json",
        "models/performance_benchmarks.json",
        "data/sample/sample_experiments.csv",
    ):
        src = ROOT_DIR / rel
        if src.is_file():
            dst = clean_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    out_clean = tmp_path / "r_clean.md"
    generate_strategy_report(root_dir=clean_root, output_path=out_clean)
    assert out_clean.read_bytes() == b1

    text = b1.decode("utf-8")
    assert "VERIFIED_COMMERCIAL_SAFE" in text
    assert "SYNTHETIC_BENCHMARK" in text
    assert "Technical Stack" in text
    assert "CONTENT_SERIES_TEMPLATE.md" in text
    assert "Historical Pre-Batch-5A Reference" in text
    # Kaggle blocked datasets must NEVER be mislabeled as "Not present locally"
    for line in text.splitlines():
        if "youtube_trending_kaggle" in line or "instagram_analytics_kaggle" in line or "youtube_kaggle_candidate_h" in line:
            assert "Not present locally" not in line
            assert "blocked" in line.lower()
            if line.startswith("| `"):
                assert "| Blocked (" in line

    # Local-state opt-in surfaces optional host availability without corrupting blocked Kaggle status
    out_local = tmp_path / "r_local.md"
    generate_strategy_report(root_dir=ROOT_DIR, output_path=out_local, include_local_state=True)
    local_text = out_local.read_text(encoding="utf-8")
    assert "Host Runtime State Enabled" in local_text
    for line in local_text.splitlines():
        if "| `youtube_trending_kaggle`" in line:
            assert "Not present locally" not in line

    # Verify real-data benchmarks section contains zero broken N/A placeholders when models/real_data_benchmarks.json is present
    if (ROOT_DIR / "models" / "real_data_benchmarks.json").is_file():
        sec4 = text.split("## 4. Empirical Findings")[1].split("## 5. Offline Synthetic")[0]
        assert "N/A" not in sec4
        assert "Metric `None`" not in sec4
