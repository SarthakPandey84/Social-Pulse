"""Tests for the 12-stage data quality pipeline and multi-platform normalization adapters."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.engagement_metrics import is_retention_applicable
from src.data_ingestion.api_connectors import (
    InstagramGraphAPIConnector,
    YouTubeDataAPIConnector,
)
from src.data_ingestion.external_adapters import (
    adapt_youtube_trending_dataframe,
    prepare_instaart_human_ai,
    prepare_instagram_comments_sentiment_hf,
    prepare_tiktok_engagement_200k,
)
from src.data_ingestion.external_datasets import (
    DATASET_REGISTRY,
    list_xlsx_sheet_names,
    read_xlsx_sheet_to_dataframe,
)
from src.data_ingestion.schemas import MetricAvailabilityStatus, SourceProvenance, get_applicable_metrics
from src.preprocessing.feature_engineering import add_engagement_features
from src.preprocessing.quality_pipeline import (
    CleaningConfig,
    PII_HMAC_KEY_ENV,
    fix_mojibake_and_whitespace,
    parse_compact_count_series,
    pseudonymize_identifier,
    redact_pii_text,
    run_cleaning_pipeline,
)


def test_quality_pipeline_all_12_stages_and_proxy_honesty() -> None:
    df = pd.DataFrame({
        "platform": ["tiktok", "tiktok", "tiktok", "tiktok"],
        "post_id": ["v1", "v1", "v1", "v2"],
        "days_since_post": [0, 1, 2, 0],
        "timestamp": ["2024-07-01 10:00:00", "2024-07-01 10:00:00", "2024-07-01 10:00:00", "17:59"],
        "views": ["10.5K", "12K", "11K", "N/A"],
        "likes": [100, 120, 115, -5],
        "comments_count": [10, 10, 10, 0],
        "shares": [5, 5, 5, 1],
        "saves": [pd.NA, pd.NA, pd.NA, pd.NA],
        "retention_rate": [0.45, 1.5, -0.2, pd.NA],
        "author_id": ["creator_raw_1", "creator_raw_1", "creator_raw_1", "creator_raw_2"],
        "caption": [
            "Check out @alice_creator and https://example.com! ðŸ˜¤",
            "Check out @alice_creator and https://example.com! ðŸ˜¤",
            "Email test@example.org for collab",
            "Clean text",
        ],
        "eval_split": ["train", "train", "test", "test"],
        "performance_bucket_label": ["high", "high", "medium", "low"],
    })

    cfg = CleaningConfig(
        dataset_id="unit_test_ds",
        table_name="posts",
        platform="tiktok",
        provenance=SourceProvenance.REAL_PUBLIC_DATA,
        access_tier="public_api",
        composite_key_cols=("platform", "post_id", "days_since_post"),
        mandatory_cols=("platform", "post_id"),
        shorthand_int_cols=("views",),
        int_cols=("likes", "comments_count", "shares", "saves"),
        timestamp_cols=("timestamp",),
        non_negative_cols=("likes",),
        rate_0_to_1_cols=("retention_rate",),
        monotonic_group_col="post_id",
        monotonic_order_col="days_since_post",
        monotonic_counter_cols=("views", "likes"),
        pii_id_cols=("author_id",),
        drop_raw_pii_cols=("author_id",),
        pii_text_cols=("caption",),
        mojibake_text_cols=("caption",),
        split_col="eval_split",
        split_dedup_text_col="caption",
        quarantine_leakage_cols=("performance_bucket_label",),
        proxy_metrics_used=("tiktok_collect_count",),
        recompute_engagement_rate=True,
    )

    clean_df, report = run_cleaning_pipeline(df, cfg)

    # Composite key preserves v1 days 0, 1, 2
    assert len(clean_df) == 4
    assert report["stage_1_encoding_fixes"] >= 1
    assert "😤" in clean_df.loc[0, "caption"]
    assert "[USER]" in clean_df.loc[0, "caption"]
    assert "[URL]" in clean_df.loc[0, "caption"]
    assert "[EMAIL]" in clean_df.loc[2, "caption"]
    assert "author_id" not in clean_df.columns
    assert str(clean_df.loc[0, "creator_pseudo_id"]).startswith("cr_")

    # Shorthand expansion without zero-imputation
    assert int(clean_df.loc[0, "views"]) == 10500
    assert pd.isna(clean_df.loc[3, "views"])
    assert pd.isna(clean_df.loc[3, "likes"])  # negative raw count masked to NA, NOT clamped to 0

    # Cumulative counter reset detected cleanly on day 2 drop without deleting day 2
    assert bool(clean_df.loc[2, "has_negative_delta_reset"]) is True
    assert report["stage_5_anomalies"]["cumulative_counter_negative_deltas"]["views"] == 1
    assert report["stage_5_anomalies"]["cumulative_counter_negative_deltas"]["likes"] == 1

    # Partial time-only timestamp classified cleanly instead of coerced to today's date
    assert clean_df.loc[3, "timestamp_precision"] == "time_only"
    assert pd.isna(clean_df.loc[3, "timestamp"])

    # Retention clipped safely & NA preserved
    assert float(clean_df.loc[1, "retention_rate"]) == 1.0
    assert float(clean_df.loc[2, "retention_rate"]) == 0.0
    assert pd.isna(clean_df.loc[3, "retention_rate"])

    # Proxy & saves availability strictly separated
    assert report["stage_7_metric_availability"]["saves"] == MetricAvailabilityStatus.NOT_APPLICABLE.value
    assert "tiktok_collect_count" in report["stage_7_proxy_definitions"]
    assert "performance_bucket_label" in report["stage_10_leakage_quarantined_columns"]


def test_mojibake_shorthand_and_pseudonym_helpers() -> None:
    recovered, chg = fix_mojibake_and_whitespace("Keren banget ðŸ”¥ðŸ˜\x8d  ")
    assert chg is True
    assert recovered == "Keren banget 🔥😍"

    parsed = parse_compact_count_series(pd.Series(["30.5K", "1.2M", "500", "-", None]))
    assert list(parsed.iloc[:3]) == [30500.0, 1200000.0, 500.0]
    assert pd.isna(parsed.iloc[3]) and pd.isna(parsed.iloc[4])

    assert pseudonymize_identifier("user_abc") == pseudonymize_identifier("user_abc")
    assert pseudonymize_identifier("user_abc") != pseudonymize_identifier("user_xyz")
    txt, counts = redact_pii_text("Hi @friend see https://t.co/x me@foo.bar")
    assert "[USER]" in txt and "[URL]" in txt and "[EMAIL]" in txt
    assert counts == {"handles": 1, "emails": 1, "urls": 1}


def test_tiktok_adapter_preserves_proxy_reset_and_strict_prior_creator_join(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()

    videos = pd.DataFrame({
        "video_id": ["tk_before", "tk0", "tk1", "tk2"],
        "author_id": ["auth_a", "auth_a", "auth_a", "auth_a"],
        "create_time": [
            "2024-06-30 12:00:00",
            "2024-07-01 12:00:00",
            "2024-07-02 12:00:00",
            "2024-07-05 12:00:00",
        ],
        "create_date": ["2024-06-30", "2024-07-01", "2024-07-02", "2024-07-05"],
        "duration": [18.0, 22.0, 25.0, 40.0],
        "ratio": ["720p", "720p", "720p", "1080p"],
        "desc_language": ["en", "en", "en", "en"],
        "is_english": [1, 1, 1, 1],
        "desc": ["video before", "video zero", "video one", "video two"],
        "hashtags": [[], [], [{"hashtag_id": "1", "hashtag_name": "fyp"}], []],
        "created_by_ai": [0, 0, 0, 0],
        "is_ads": ["0", "0", "0", "0"],
        "music_id": ["m0", "m00", "m1", "m2"],
        "music_title": ["song b", "song 0", "song 1", "song 2"],
        "word_count": [8.0, 12.0, 10.0, 20.0],
        "emoji_count": [0.0, 0.0, 1.0, 0.0],
        "question_count": [0.0, 0.0, 0.0, 1.0],
        "hashtag_count": [0.0, 0.0, 1.0, 0.0],
        "speaking_rate": [2.0, 2.5, 2.0, 3.0],
        "topic": ["Lifestyle", "Lifestyle", "Lifestyle", "Cooking_Food"],
        "anger": [0.01, 0.01, 0.01, 0.02],
        "joy": [0.70, 0.75, 0.80, 0.60],
        "surprise": [0.05, 0.04, 0.05, 0.02],
        "sadness": [0.01, 0.01, 0.01, 0.01],
        "disgust": [0.01, 0.01, 0.01, 0.01],
        "fear": [0.01, 0.01, 0.01, 0.01],
    })
    videos.to_parquet(raw / "videos.parquet", index=False)

    creator_daily = pd.DataFrame({
        "author_id": ["auth_a", "auth_a", "auth_a"],
        "date": ["2024-07-01", "2024-07-05", "2024-07-20"],
        "follower_count": [1000.0, 1500.0, 99999.0],  # same-day (2024-07-05) and future (2024-07-20) must NOT leak into prior-day joins
        "following_count": [50.0, 50.0, 50.0],
        "total_favorited": [5000, 8000, 500000],
        "video_count": [10.0, 11.0, 20.0],
        "enterprise_verified": [np.nan, np.nan, np.nan],
    })
    creator_daily.to_parquet(raw / "creator_daily.parquet", index=False)

    ed = pd.DataFrame({
        "video_id": ["tk_before", "tk0", "tk1", "tk1", "tk1", "tk2"],
        "date": ["2024-06-30", "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-09", "2024-07-05"],
        "days_since_post": [0, 0, 0, 1, 7, 0],
        "play_count": [90, 120, 100, 250, 240, 500],  # reset between day 1 and day 7 for tk1
        "like_count": [9, 12, 10, 25, 30, 50],
        "comment_count": [1, 1, 1, 2, 3, 5],
        "share_count": [0, 0, 0, 1, 2, 4],
        "collect_count": [1, 2, 2, 5, 6, 9],
        "download_count": [0, 0, 0, 1, 1, 2],
        "whatsapp_share_count": [0, 0, 0, 0, 1, 1],
    })
    ed.to_parquet(raw / "engagement_daily.parquet", index=False)

    meta = prepare_tiktok_engagement_200k(raw_dir=raw, out_dir=out)
    assert meta["row_counts"]["posts"] == 4
    posts = pd.read_parquet(out / "posts.parquet").set_index("post_id")

    # Pre-history (tk_before) & earliest-day (tk0) have NO strictly prior snapshot -> must retain pd.NA (no same-day or +7d future backfill)
    assert pd.isna(posts.loc["tk_before", "follower_count"])
    assert pd.isna(posts.loc["tk_before", "creator_total_favorited"])
    assert pd.isna(posts.loc["tk0", "follower_count"])
    assert pd.isna(posts.loc["tk0", "creator_total_favorited"])

    # Strictly prior snapshot: both tk1 (2024-07-02) and tk2 (2024-07-05) join 2024-07-01 (1000), NEVER same-day 2024-07-05 (1500) or future (99999)
    assert int(posts.loc["tk1", "follower_count"]) == 1000
    assert int(posts.loc["tk1", "creator_total_favorited"]) == 5000
    assert int(posts.loc["tk2", "follower_count"]) == 1000
    assert int(posts.loc["tk2", "creator_total_favorited"]) == 5000

    # Canonical TikTok content_type normalization & proxy/reset preservation
    assert posts.loc["tk1", "content_type"] == "short video"
    assert posts.loc["tk2", "content_type"] == "short video"
    assert pd.isna(posts.loc["tk1", "saves"])
    assert int(posts.loc["tk1", "collect_count"]) == 6
    assert bool(posts.loc["tk1", "has_negative_delta_reset"]) is True
    assert bool(posts.loc["tk2", "has_negative_delta_reset"]) is False
    assert int(posts.loc["tk1", "day0_views"]) == 100
    assert int(posts.loc["tk1", "day7_views"]) == 240


def test_short_video_canonical_and_underscore_parity() -> None:
    assert is_retention_applicable("tiktok", "short video") is True
    assert is_retention_applicable("tiktok", "short_video") is True
    assert is_retention_applicable("youtube", "short_video") is True
    assert "views" in get_applicable_metrics("youtube", "short_video", "owner_oauth")
    assert "retention_rate" in get_applicable_metrics("youtube", "short_video", "owner_oauth")

    df = pd.DataFrame([
        {
            "post_id": "TK_CANON",
            "platform": "tiktok",
            "content_type": "short video",
            "views": 1000,
            "likes": 80,
            "comments_count": 5,
            "shares": 10,
            "saves": pd.NA,
            "retention_rate": pd.NA,
        },
        {
            "post_id": "TK_LEGACY_UNDERSCORE",
            "platform": "tiktok",
            "content_type": "short_video",
            "views": 1000,
            "likes": 80,
            "comments_count": 5,
            "shares": 10,
            "saves": pd.NA,
            "retention_rate": 0.45,
        },
    ])
    feat = add_engagement_features(df)
    assert feat.loc[0, "equivalent_full_watch_status"] == "unmeasured_missing_views_or_retention"
    assert feat.loc[1, "equivalent_full_watch_status"] == "computed"
    assert float(feat.loc[1, "equivalent_full_watch_views"]) == pytest.approx(450.0)


def test_instaart_and_ig_sentiment_adapters(tmp_path: Path) -> None:
    # 1. InstaArt wide unpivot & rank vs chronological index
    ia_raw = tmp_path / "ia_raw"
    ia_out = tmp_path / "ia_out"
    ia_raw.mkdir()

    pd.DataFrame({
        "human_post_id": ["H1"],
        "human_post_link": ["https://instagram.com/p/H1/"],
        "human_type": ["Post"],
        "human_art_description": ["portrait"],
        "human_instagram_account": ["@human_artist"],
        "human_followers": [5000],
        "human_total_posts": [50],
        "human_likes": [300],
        "human_total_comments": [2],
        "human_reposts": [4],
        "human_shares": [10],
        "ai_post_id": ["A1"],
        "ai_post_link": ["https://instagram.com/p/A1/"],
        "ai_type": ["Reel"],
        "ai_art_description": ["cyberpunk"],
        "ai_instagram_account": ["@ai_artist"],
        "ai_followers": [8000],
        "ai_total_posts": [80],
        "ai_likes": [500],
        "ai_total_comments": [2],
        "ai_reposts": [6],
        "ai_shares": [20],
        "discovery_hashtag": ["#aiart"],
    }).to_csv(ia_raw / "Posts_Primary.csv", index=False)

    # Deliberately store rank 1 with LATER timestamp than rank 2 to prove rank != chronological
    pd.DataFrame({"post_id": ["H1"], "comment_1": ["Top ranked later"], "comment_2": ["Older comment"]}).to_csv(
        ia_raw / "Human_Comments_Data.csv", index=False
    )
    pd.DataFrame({"post_id": ["H1"], "comment_1": ["2024-08-20 20:00:00"], "comment_2": ["2024-08-20 10:00:00"]}).to_csv(
        ia_raw / "Human_CommentDate.csv", index=False
    )
    pd.DataFrame({"post_id": ["A1"], "comment_1": ["Nice AI piece @user1"], "comment_2": [np.nan]}).to_csv(
        ia_raw / "AI_Comments_Data.csv", index=False
    )
    pd.DataFrame({"post_id": ["A1"], "comment_1": ["2024-08-21 12:00:00"], "comment_2": [np.nan]}).to_csv(
        ia_raw / "AI_CommentDate_Data.csv", index=False
    )

    ia_meta = prepare_instaart_human_ai(raw_dir=ia_raw, out_dir=ia_out)
    assert ia_meta["row_counts"] == {"posts": 2, "comments": 3}
    comments = pd.read_parquet(ia_out / "comments.parquet")
    h_r1 = comments.loc[(comments["post_id"] == "H1") & (comments["comment_rank_index"] == 1)].iloc[0]
    h_r2 = comments.loc[(comments["post_id"] == "H1") & (comments["comment_rank_index"] == 2)].iloc[0]
    assert int(h_r2["chronological_order_index"]) == 1
    assert int(h_r1["chronological_order_index"]) == 2
    assert "[USER]" in comments.loc[comments["post_id"] == "A1", "comment_text"].iloc[0]

    # 2. IG Comments Sentiment HF PII & train/test isolation
    igs_raw = tmp_path / "igs_raw"
    igs_out = tmp_path / "igs_out"
    igs_raw.mkdir()
    pd.DataFrame({
        "comment_id": ["1001"], "user_id": ["999"], "username": ["john_doe"],
        "text": ["@friend awesome work"], "created_at": ["2019-07-22 08:00:00"],
        "likes": [3], "category": ["human"], "shortcode": ["XYZ"], "split": ["train"],
        "sentiment_label_rater1": ["positive"],
    }).to_csv(igs_raw / "train.csv", index=False)
    pd.DataFrame({
        "comment_id": ["1002"], "user_id": ["888"], "username": ["jane_doe"],
        "text": ["Not sure about this"], "created_at": ["2019-07-23 09:00:00"],
        "likes": [0], "category": ["ai"], "shortcode": ["ABC"], "split": ["test"],
        "sentiment_label_rater1": ["neutral"],
    }).to_csv(igs_raw / "test.csv", index=False)

    igs_meta = prepare_instagram_comments_sentiment_hf(raw_dir=igs_raw, out_dir=igs_out)
    assert igs_meta["row_counts"] == {"comments": 2, "train": 1, "test": 1}
    igs_df = pd.read_parquet(igs_out / "comments.parquet")
    assert "raw_username" not in igs_df.columns
    assert "[USER]" in igs_df.loc[igs_df["eval_split"] == "train", "comment_text"].iloc[0]


def test_youtube_trending_adapter_composite_key_and_disabled_flags() -> None:
    yt_raw = pd.DataFrame({
        "video_id": ["yt1", "yt1", "yt2"],
        "trending_date": ["20.12.08", "20.13.08", "20.14.08"],
        "publishedAt": ["2020-08-10T10:00:00Z", "2020-08-10T10:00:00Z", "2020-08-11T10:00:00Z"],
        "channelId": ["ch_A", "ch_A", "ch_B"],
        "categoryId": [24, 24, 10],
        "view_count": [10000, 25000, 5000],
        "likes": [500, 900, 0],
        "comment_count": [50, 0, 20],
        "comments_disabled": [False, True, False],
        "ratings_disabled": [False, False, True],
        "title": ["Video 1", "Video 1", "Video 2"],
    })
    clean_yt, rep = adapt_youtube_trending_dataframe(yt_raw, region="US")
    assert len(clean_yt) == 3  # both days for yt1 preserved via composite key
    assert pd.isna(clean_yt.loc[1, "comments_count"])  # comments_disabled=True -> NA, NEVER 0
    assert pd.isna(clean_yt.loc[2, "likes"])  # ratings_disabled=True -> NA, NEVER 0
    assert clean_yt["saves"].isna().all()
    assert rep["stage_7_metric_availability"]["saves"] == MetricAvailabilityStatus.NOT_APPLICABLE.value
    assert "youtube_trending_snapshot" in rep["stage_7_proxy_definitions"]


def test_stdlib_xlsx_named_sheet_selector(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "mini.xlsx"
    with zipfile.ZipFile(xlsx_path, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Guide" sheetId="1" r:id="rId1"/><sheet name="Analysis" sheetId="2" r:id="rId2"/></sheets>'
            "</workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Target="worksheets/sheet2.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1" t="inlineStr"><is><t>GuideOnly</t></is></c></row>'
            "</sheetData></worksheet>",
        )
        zf.writestr(
            "xl/worksheets/sheet2.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1" t="inlineStr"><is><t>Metric</t></is></c></row>'
            '<row r="2"><c r="A2"><v>42</v></c></row>'
            "</sheetData></worksheet>",
        )

    assert list_xlsx_sheet_names(xlsx_path) == ["Guide", "Analysis"]
    df = read_xlsx_sheet_to_dataframe(xlsx_path, sheet_index="Analysis")
    assert list(df.columns) == ["Metric"]
    assert int(df.iloc[0]["Metric"]) == 42


def test_live_api_payload_normalizers() -> None:
    ig = InstagramGraphAPIConnector(sandbox_mode=True)
    df_pub = ig.normalize_api_payload(
        [{"id": "123", "media_type": "VIDEO", "like_count": 50, "comments_count": 5, "timestamp": "2026-01-01T00:00:00Z"}],
        mode="public",
        follower_count=1200,
    )
    assert pd.isna(df_pub.loc[0, "saves"])
    assert pd.isna(df_pub.loc[0, "reach"])

    df_own = ig.normalize_api_payload(
        [{"id": "123", "media_type": "VIDEO", "like_count": 50, "comments_count": 5, "saved": 12, "reach": 800, "timestamp": "2026-01-01T00:00:00Z"}],
        mode="owner",
    )
    assert int(df_own.loc[0, "saves"]) == 12
    assert int(df_own.loc[0, "reach"]) == 800

    yt = YouTubeDataAPIConnector(sandbox_mode=True)
    df_yt = yt.normalize_api_payload(
        [{"id": "yt_1", "snippet": {"title": "Demo", "publishedAt": "2026-01-01T00:00:00Z"}, "statistics": {"viewCount": "900", "likeCount": "40", "commentCount": "3"}, "averageViewPercentage": 62.5}],
        mode="owner",
    )
    assert int(df_yt.loc[0, "views"]) == 900
    assert float(df_yt.loc[0, "retention_rate"]) == pytest.approx(0.625)
    assert pd.isna(df_yt.loc[0, "saves"])


def test_pseudonymization_hmac_mode_fail_closed_and_terminology(monkeypatch) -> None:
    """Verify keyed HMAC-SHA256 mode, compatibility boundary, and accurate Stage 8 disclosures."""
    monkeypatch.delenv(PII_HMAC_KEY_ENV, raising=False)

    legacy_tok = pseudonymize_identifier("creator_42")
    assert legacy_tok.startswith("cr_") and len(legacy_tok) == 15

    # Explicit HMAC key produces cr_<16-hex>, distinct from legacy public salt
    hmac_tok = pseudonymize_identifier("creator_42", hmac_key="unit-test-hmac-secret-1")
    assert hmac_tok.startswith("cr_") and len(hmac_tok) == 19
    assert hmac_tok != legacy_tok
    assert hmac_tok == pseudonymize_identifier("creator_42", hmac_key="unit-test-hmac-secret-1")
    assert hmac_tok != pseudonymize_identifier("creator_42", hmac_key="unit-test-hmac-secret-2")

    # Fail-closed when allow_legacy_public_salt=False without key/custom salt
    with pytest.raises(ValueError, match="allow_legacy_public_salt=False"):
        pseudonymize_identifier("creator_42", allow_legacy_public_salt=False)

    # Environment variable upgrades default pseudonymization cleanly
    monkeypatch.setenv(PII_HMAC_KEY_ENV, "env-hmac-secret-xyz")
    assert pseudonymize_identifier("creator_42") == pseudonymize_identifier(
        "creator_42", hmac_key="env-hmac-secret-xyz", allow_legacy_public_salt=False
    )

    df = pd.DataFrame({
        "platform": ["instagram", "instagram"],
        "post_id": ["p1", "p2"],
        "author_id": ["creator_42", "creator_43"],
    })
    cfg = CleaningConfig(
        dataset_id="hmac_test",
        table_name="posts",
        platform="instagram",
        provenance=SourceProvenance.REAL_PUBLIC_DATA,
        composite_key_cols=("platform", "post_id"),
        mandatory_cols=("platform", "post_id"),
        pii_id_cols=("author_id",),
        drop_raw_pii_cols=("author_id",),
        allow_legacy_public_salt=False,
    )
    clean_df, rep = run_cleaning_pipeline(df, cfg)
    assert "author_id" not in clean_df.columns
    assert clean_df.loc[0, "creator_pseudo_id"] == pseudonymize_identifier("creator_42")
    s8 = rep["stage_8_pii_protection"]
    assert s8["protection_mechanism"] == "pseudonymization"
    assert s8["pseudonymization_mode"] == "hmac_sha256"
    assert s8["is_secret_keyed"] is True
    assert s8["irreversibility_claim"] is False

    # Dataset limitations must not mislabel deterministic pseudonymization as anonymization
    sp_lim = " ".join(DATASET_REGISTRY["zenodo_sponsored_unsponsored_ig"].limitations).lower()
    assert "anonymized" not in sp_lim
    assert "pseudonymized" in sp_lim

