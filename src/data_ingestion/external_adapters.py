"""Multi-platform data normalization adapters mapping external datasets to SocialPulse schemas.

Each adapter:
1. Reads verified files under `data/external/<dataset_id>/raw/` (without modifying raw inputs).
2. Normalizes platform-specific columns into SocialPulse canonical schemas (`posts.parquet`,
   `comments.parquet`, and compact aggregated/daily tables where applicable).
3. Runs the 12-stage `run_cleaning_pipeline()` auditing provenance, proxy semantics, PII redaction,
   split isolation, cumulative counter resets, and target-leakage quarantine.
4. Writes normalized Parquet outputs, quality reports, and `dataset_meta.json` under
   `data/processed/external/<dataset_id>/`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_ingestion.external_datasets import (
    DATASET_REGISTRY,
    DEFAULT_EXTERNAL_ROOT,
    classify_governance_tier,
    get_dataset_spec,
    read_xlsx_sheet_to_dataframe,
)
from src.data_ingestion.schemas import (
    ExperimentType,
    PROXY_METRIC_DEFINITIONS,
    SourceProvenance,
)
from src.preprocessing.quality_pipeline import (
    CleaningConfig,
    run_cleaning_pipeline,
    save_clean_artifacts,
)


DEFAULT_PROCESSED_EXTERNAL_ROOT = Path("data") / "processed" / "external"


def _write_dataset_meta(out_dir: Path, meta: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_id = str(meta.get("dataset_id") or "")
    if "governance_tier" not in meta and ds_id in DATASET_REGISTRY:
        meta["governance_tier"] = classify_governance_tier(DATASET_REGISTRY[ds_id])
    meta_path = out_dir / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return meta_path


def _extract_hashtag_names(val: Any) -> str:
    """Flatten TikTok list-of-dicts / numpy.ndarray hashtags cleanly into pipe-delimited string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, (np.ndarray, list, tuple)):
        names: List[str] = []
        for item in val:
            if isinstance(item, dict):
                nm = str(item.get("hashtag_name") or "").strip()
                if nm:
                    names.append(nm)
            elif item is not None:
                nm = str(item).strip()
                if nm:
                    names.append(nm)
        return "|".join(names)
    return str(val)


def prepare_tiktok_engagement_200k(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `tiktok_engagement_200k` into canonical posts + age-standardized panel + daily aggregates."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "tiktok_engagement_200k" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "tiktok_engagement_200k")
    spec = get_dataset_spec("tiktok_engagement_200k")

    # 1. Read videos metadata cleanly (avoid carrying unhashable list/ndarray columns directly)
    video_cols = [
        "video_id", "author_id", "create_time", "create_date", "duration", "ratio",
        "desc_language", "is_english", "desc", "hashtags", "created_by_ai", "is_ads",
        "music_id", "music_title", "word_count", "emoji_count", "question_count",
        "hashtag_count", "speaking_rate", "topic",
        "anger", "joy", "surprise", "sadness", "disgust", "fear",
    ]
    videos = pd.read_parquet(raw / "videos.parquet", columns=video_cols)
    videos["video_id"] = videos["video_id"].astype("string")
    videos["author_id"] = videos["author_id"].astype("string")
    videos["hashtag_names"] = videos["hashtags"].map(_extract_hashtag_names).astype("string")
    videos = videos.drop(columns=["hashtags"])

    # Parse creation timestamp safely
    create_dt = pd.to_datetime(videos["create_time"], format="mixed", utc=True, errors="coerce")
    fallback_dt = pd.to_datetime(videos["create_date"], format="mixed", utc=True, errors="coerce")
    videos["timestamp"] = create_dt.fillna(fallback_dt)
    videos["create_day"] = pd.to_datetime(videos["create_date"], format="mixed", utc=True, errors="coerce")

    # 2. Extract creator baseline strictly BEFORE video publication date (creator_day < create_day)
    # Same-day creator_daily rows contain end-of-day engagement accumulated AFTER publication on create_day.
    creators = pd.read_parquet(raw / "creator_daily.parquet")
    creators["author_id"] = creators["author_id"].astype("string")
    creators["creator_day"] = pd.to_datetime(creators["date"], format="mixed", utc=True, errors="coerce")
    creators = creators.dropna(subset=["author_id", "creator_day"]).sort_values(["creator_day", "author_id"]).reset_index(drop=True)

    videos_sorted = videos.sort_values(["create_day", "author_id"]).reset_index(drop=True)
    videos_joined = pd.merge_asof(
        videos_sorted,
        creators[["author_id", "creator_day", "follower_count", "video_count", "total_favorited"]].rename(
            columns={"video_count": "creator_video_count", "total_favorited": "creator_total_favorited"}
        ),
        left_on="create_day",
        right_on="creator_day",
        by="author_id",
        direction="backward",
        allow_exact_matches=False,
    )

    # 3. Scan engagement_daily.parquet for cumulative reset flags, age-standardized snapshots, and clean daily increments
    ed_cols = [
        "video_id", "date", "days_since_post", "play_count", "like_count",
        "comment_count", "share_count", "collect_count", "download_count", "whatsapp_share_count",
    ]
    ed = pd.read_parquet(raw / "engagement_daily.parquet", columns=ed_cols)
    ed["video_id"] = ed["video_id"].astype("string")
    ed = ed.sort_values(["video_id", "days_since_post"], kind="mergesort").reset_index(drop=True)

    same_vid = ed["video_id"].eq(ed["video_id"].shift(1)).fillna(False).to_numpy()
    day_step = (ed["days_since_post"] - ed["days_since_post"].shift(1)).fillna(0).to_numpy()
    consec = same_vid & (day_step == 1)

    counter_cols = ["play_count", "like_count", "comment_count", "share_count", "collect_count"]
    reset_counts: Dict[str, int] = {}
    any_reset_row = np.zeros(len(ed), dtype=bool)
    inc_arrays: Dict[str, np.ndarray] = {}

    for col in counter_cols:
        arr = pd.to_numeric(ed[col], errors="coerce").to_numpy(dtype="float64")
        prev = np.empty_like(arr)
        prev[0] = np.nan
        prev[1:] = arr[:-1]
        diff = arr - prev
        neg_mask = same_vid & (diff < 0)
        reset_counts[col] = int(np.nansum(neg_mask))
        any_reset_row |= neg_mask

        # Daily new increment: day 0 cumulative count == day 0 increment; subsequent consecutive positive deltas == daily increment;
        # reset rows masked to NaN rather than distorting totals with spurious negative spikes.
        inc = np.full_like(arr, np.nan)
        day0_mask = ed["days_since_post"].to_numpy() == 0
        inc[day0_mask] = np.where(arr[day0_mask] >= 0, arr[day0_mask], np.nan)
        valid_step = consec & (diff >= 0)
        inc[valid_step] = diff[valid_step]
        inc_arrays[col] = inc

    ed["has_negative_delta_reset"] = any_reset_row
    vid_has_reset = ed.groupby("video_id", sort=False)["has_negative_delta_reset"].any().rename("has_negative_delta_reset")

    # Build age-standardized horizon snapshots (Day 0, Day 1, Day 7, Day 30, and latest observed snapshot)
    latest_ed = ed.drop_duplicates(subset=["video_id"], keep="last").set_index("video_id")

    snap_frames: List[pd.DataFrame] = []
    for target_day in (0, 1, 7, 30):
        sub = ed.loc[ed["days_since_post"] == target_day, [
            "video_id", "play_count", "like_count", "comment_count", "share_count", "collect_count", "download_count"
        ]].drop_duplicates(subset=["video_id"], keep="last").set_index("video_id")
        sub = sub.rename(columns={
            "play_count": f"day{target_day}_views",
            "like_count": f"day{target_day}_likes",
            "comment_count": f"day{target_day}_comments",
            "share_count": f"day{target_day}_shares",
            "collect_count": f"day{target_day}_collects",
            "download_count": f"day{target_day}_downloads",
        })
        snap_frames.append(sub)

    # Assemble canonical post table (1 row per unique video_id, latest cumulative outcome + horizon snapshots)
    posts_raw = videos_joined.set_index("video_id")
    posts_raw["platform"] = "tiktok"
    posts_raw["post_id"] = posts_raw.index.astype("string")
    posts_raw["content_type"] = "short video"
    posts_raw["views"] = latest_ed["play_count"].reindex(posts_raw.index)
    posts_raw["impressions"] = pd.NA
    posts_raw["reach"] = pd.NA
    posts_raw["likes"] = latest_ed["like_count"].reindex(posts_raw.index)
    posts_raw["comments_count"] = latest_ed["comment_count"].reindex(posts_raw.index)
    posts_raw["shares"] = latest_ed["share_count"].reindex(posts_raw.index)
    posts_raw["saves"] = pd.NA  # Explicit structural NA — TikTok collect_count is proxy only!
    posts_raw["retention_rate"] = pd.NA
    posts_raw["collect_count"] = latest_ed["collect_count"].reindex(posts_raw.index)
    posts_raw["download_count"] = latest_ed["download_count"].reindex(posts_raw.index)
    posts_raw["whatsapp_share_count"] = latest_ed["whatsapp_share_count"].reindex(posts_raw.index)
    posts_raw["latest_days_since_post"] = latest_ed["days_since_post"].reindex(posts_raw.index)
    posts_raw["collection_timestamp"] = pd.to_datetime(latest_ed["date"].reindex(posts_raw.index), format="mixed", utc=True, errors="coerce")
    posts_raw["has_negative_delta_reset"] = vid_has_reset.reindex(posts_raw.index).fillna(False)

    for sf in snap_frames:
        posts_raw = posts_raw.join(sf, how="left")

    posts_raw["duration_seconds"] = pd.to_numeric(posts_raw["duration"], errors="coerce")
    posts_raw["provenance"] = spec.provenance.value
    posts_raw["source_dataset"] = spec.dataset_id
    posts_raw = posts_raw.reset_index(drop=True)

    # Drop raw author_id via cleaning pipeline after hashing to creator_pseudo_id and redact PII text
    post_int_cols = (
        "views", "likes", "comments_count", "shares", "follower_count",
        "collect_count", "download_count", "whatsapp_share_count", "latest_days_since_post",
        "creator_video_count", "creator_total_favorited",
        "day0_views", "day0_likes", "day0_comments", "day0_shares", "day0_collects", "day0_downloads",
        "day1_views", "day1_likes", "day1_comments", "day1_shares", "day1_collects", "day1_downloads",
        "day7_views", "day7_likes", "day7_comments", "day7_shares", "day7_collects", "day7_downloads",
        "day30_views", "day30_likes", "day30_comments", "day30_shares", "day30_collects", "day30_downloads",
    )
    post_cfg = CleaningConfig(
        dataset_id=spec.dataset_id,
        table_name="posts",
        platform="tiktok",
        provenance=spec.provenance,
        access_tier="public_api",
        composite_key_cols=("platform", "post_id"),
        mandatory_cols=("platform", "post_id", "timestamp", "collection_timestamp"),
        int_cols=post_int_cols,
        float_cols=("duration_seconds", "retention_rate", "word_count", "emoji_count", "question_count", "hashtag_count", "speaking_rate", "anger", "joy", "surprise", "sadness", "disgust", "fear"),
        string_cols=("platform", "post_id", "content_type", "topic", "desc_language", "hashtag_names", "provenance", "source_dataset"),
        timestamp_cols=("timestamp", "collection_timestamp"),
        non_negative_cols=("views", "likes", "comments_count", "shares", "collect_count", "follower_count", "duration_seconds"),
        pii_id_cols=("author_id",),
        drop_raw_pii_cols=("author_id", "desc", "creator_day", "create_day", "create_time", "create_date", "duration"),
        mojibake_text_cols=("music_title", "hashtag_names"),
        quarantine_leakage_cols=(
            "views", "likes", "comments_count", "shares", "collect_count", "download_count",
            "day7_views", "day7_likes", "day7_comments", "day7_shares", "day7_collects",
            "day30_views", "day30_likes", "day30_comments", "day30_shares", "day30_collects",
            "engagement_rate_recomputed", "share_rate_recomputed",
        ),
        proxy_metrics_used=("tiktok_collect_count", "tiktok_whatsapp_share_count"),
        recompute_engagement_rate=True,
        recompute_share_rate=True,
        notes=(
            "209,543 videos joined strictly with prior-day creator_daily snapshot (creator_day < create_day; zero same-day or future follower/total_favorited leakage; first-day posts without prior history retain pd.NA).",
            f"Cumulative daily panel negative-delta reset counts audited across 6.07M rows: {reset_counts}.",
            "`saves` retained as explicit pd.NA (`not_applicable` on TikTok); `collect_count` kept as proxy.",
        ),
    )
    posts_clean, posts_report = run_cleaning_pipeline(posts_raw, post_cfg)
    posts_report["stage_5_anomalies"]["cumulative_counter_negative_deltas"] = reset_counts
    save_clean_artifacts(posts_clean, posts_report, out, "posts")

    # 4. Build compact daily longitudinal & platform-level aggregates for M6 / M7 forecasting without 6M reload overhead
    vid_topic_map = videos.set_index("video_id")["topic"]
    daily_work = pd.DataFrame({
        "date": ed["date"],
        "days_since_post": ed["days_since_post"],
        "topic": ed["video_id"].map(vid_topic_map).fillna("Unknown"),
        "inc_views": inc_arrays["play_count"],
        "inc_likes": inc_arrays["like_count"],
        "inc_comments": inc_arrays["comment_count"],
        "inc_shares": inc_arrays["share_count"],
        "inc_collects": inc_arrays["collect_count"],
        "cum_views": pd.to_numeric(ed["play_count"], errors="coerce"),
        "cum_likes": pd.to_numeric(ed["like_count"], errors="coerce"),
        "cum_comments": pd.to_numeric(ed["comment_count"], errors="coerce"),
        "cum_shares": pd.to_numeric(ed["share_count"], errors="coerce"),
        "cum_collects": pd.to_numeric(ed["collect_count"], errors="coerce"),
        "is_reset": any_reset_row,
    })

    daily_summary = (
        daily_work.groupby("date", as_index=False)
        .agg(
            active_videos=("days_since_post", "count"),
            reset_events=("is_reset", "sum"),
            daily_views=("inc_views", "sum"),
            daily_likes=("inc_likes", "sum"),
            daily_comments=("inc_comments", "sum"),
            daily_shares=("inc_shares", "sum"),
            daily_collects=("inc_collects", "sum"),
            median_cum_views=("cum_views", "median"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_summary["date"] = pd.to_datetime(daily_summary["date"], format="mixed", utc=True, errors="coerce")
    daily_summary["platform"] = "tiktok"
    daily_summary["provenance"] = spec.provenance.value
    daily_summary.to_parquet(out / "daily_platform_summary.parquet", index=False)

    age_curve = (
        daily_work.groupby("days_since_post", as_index=False)
        .agg(
            observations=("inc_views", "count"),
            median_cum_views=("cum_views", "median"),
            mean_cum_views=("cum_views", "mean"),
            median_cum_likes=("cum_likes", "median"),
            mean_inc_views=("inc_views", "mean"),
            mean_inc_likes=("inc_likes", "mean"),
            reset_events=("is_reset", "sum"),
        )
        .sort_values("days_since_post")
        .reset_index(drop=True)
    )
    age_curve.to_parquet(out / "age_trajectory_curve.parquet", index=False)

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "tiktok",
        "platforms": ["tiktok"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {
            "posts": "posts.parquet",
            "daily_platform_summary": "daily_platform_summary.parquet",
            "age_trajectory_curve": "age_trajectory_curve.parquet",
        },
        "row_counts": {
            "posts": int(len(posts_clean)),
            "daily_platform_summary": int(len(daily_summary)),
            "age_trajectory_curve": int(len(age_curve)),
            "raw_engagement_daily_audited": int(len(ed)),
        },
        "metric_availability": posts_report["stage_7_metric_availability"],
        "proxy_metrics": {k: PROXY_METRIC_DEFINITIONS[k] for k in ("tiktok_collect_count", "tiktok_whatsapp_share_count")},
        "leakage_quarantined_columns": posts_report["stage_10_leakage_quarantined_columns"],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M1", "M2", "M5", "M6", "M7"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def prepare_instaart_human_ai(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `instaart_human_ai` paired Human vs AI Instagram art posts & unpivoted comments."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "instaart_human_ai" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "instaart_human_ai")
    spec = get_dataset_spec("instaart_human_ai")

    pp = pd.read_csv(raw / "Posts_Primary.csv")
    pp["pair_id"] = np.arange(len(pp), dtype=np.int64)

    def _map_type(val: Any) -> str:
        s = str(val).strip().lower()
        return "reel" if "reel" in s else "image"

    human_posts = pd.DataFrame({
        "platform": "instagram",
        "post_id": pp["human_post_id"].astype("string"),
        "pair_id": pp["pair_id"],
        "account_group": "human",
        "discovery_hashtag": pp["discovery_hashtag"].astype("string"),
        "content_type": pp["human_type"].map(_map_type),
        "art_description": pp["human_art_description"].astype("string"),
        "raw_account_ref": pp["human_instagram_account"].fillna(pp["human_post_id"]).astype("string"),
        "follower_count": pp["human_followers"],
        "creator_total_posts": pp["human_total_posts"],
        "likes": pp["human_likes"],
        "comments_count": pp["human_total_comments"],
        "reposts": pp["human_reposts"],
        "shares": pp["human_shares"],
    })
    ai_posts = pd.DataFrame({
        "platform": "instagram",
        "post_id": pp["ai_post_id"].astype("string"),
        "pair_id": pp["pair_id"],
        "account_group": "ai",
        "discovery_hashtag": pp["discovery_hashtag"].astype("string"),
        "content_type": pp["ai_type"].map(_map_type),
        "art_description": pp["ai_art_description"].astype("string"),
        "raw_account_ref": pp["ai_instagram_account"].fillna(pp["ai_post_id"]).astype("string"),
        "follower_count": pp["ai_followers"],
        "creator_total_posts": pp["ai_total_posts"],
        "likes": pp["ai_likes"],
        "comments_count": pp["ai_total_comments"],
        "reposts": pp["ai_reposts"],
        "shares": pp["ai_shares"],
    })
    posts_raw = pd.concat([human_posts, ai_posts], ignore_index=True)
    posts_raw["views"] = pd.NA
    posts_raw["impressions"] = pd.NA
    posts_raw["reach"] = pd.NA
    posts_raw["saves"] = pd.NA
    posts_raw["retention_rate"] = pd.NA
    posts_raw["provenance"] = spec.provenance.value
    posts_raw["source_dataset"] = spec.dataset_id

    # Unpivot wide Comment_1..Comment_50 columns and align matching dates
    def _unpivot_group(text_file: Path, date_file: Path, group_name: str) -> pd.DataFrame:
        tdf = pd.read_csv(text_file)
        ddf = pd.read_csv(date_file)
        comment_cols = [c for c in tdf.columns if c != "post_id"]
        t_long = tdf.melt(id_vars=["post_id"], value_vars=comment_cols, var_name="rank_col", value_name="comment_text")
        d_long = ddf.melt(id_vars=["post_id"], value_vars=comment_cols, var_name="rank_col", value_name="comment_timestamp_raw")
        merged = t_long.merge(d_long, on=["post_id", "rank_col"], how="left")
        merged["comment_rank_index"] = pd.to_numeric(merged["rank_col"].str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
        merged = merged.dropna(subset=["comment_text"]).copy()
        merged["comment_text"] = merged["comment_text"].astype("string")
        merged = merged[merged["comment_text"].str.strip().str.len() > 0].reset_index(drop=True)
        merged["account_group"] = group_name
        merged["platform"] = "instagram"
        merged["post_id"] = merged["post_id"].astype("string")
        merged["comment_id"] = (
            merged["post_id"].astype(str) + "_" + group_name + "_r" + merged["comment_rank_index"].astype(str)
        )
        merged["comment_timestamp"] = merged["comment_timestamp_raw"]
        return merged.drop(columns=["rank_col", "comment_timestamp_raw"])

    c_human = _unpivot_group(raw / "Human_Comments_Data.csv", raw / "Human_CommentDate.csv", "human")
    c_ai = _unpivot_group(raw / "AI_Comments_Data.csv", raw / "AI_CommentDate_Data.csv", "ai")
    comments_raw = pd.concat([c_human, c_ai], ignore_index=True)

    # Derive post-level timestamp span from unpivoted comments when available
    comm_ts_parsed = pd.to_datetime(comments_raw["comment_timestamp"], format="mixed", utc=True, errors="coerce")
    comments_raw["comment_timestamp_dt"] = comm_ts_parsed
    comments_raw = comments_raw.sort_values(["post_id", "comment_timestamp_dt", "comment_rank_index"]).reset_index(drop=True)
    comments_raw["chronological_order_index"] = comments_raw.groupby("post_id").cumcount() + 1
    earliest_by_post = comments_raw.groupby("post_id")["comment_timestamp_dt"].min()
    latest_by_post = comments_raw.groupby("post_id")["comment_timestamp_dt"].max()
    comments_raw = comments_raw.drop(columns=["comment_timestamp_dt"])
    comments_raw["provenance"] = spec.provenance.value
    comments_raw["source_dataset"] = spec.dataset_id
    comments_raw["eval_split"] = "unsplit"

    posts_raw["timestamp"] = posts_raw["post_id"].map(earliest_by_post).fillna(pd.Timestamp("2025-01-01T00:00:00Z"))
    posts_raw["collection_timestamp"] = posts_raw["post_id"].map(latest_by_post).fillna(pd.Timestamp("2026-03-01T00:00:00Z"))

    posts_clean, posts_report = run_cleaning_pipeline(
        posts_raw,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="posts",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "post_id"),
            mandatory_cols=("platform", "post_id", "timestamp", "collection_timestamp"),
            int_cols=("pair_id", "follower_count", "creator_total_posts", "likes", "comments_count", "reposts", "shares", "views", "impressions", "reach", "saves"),
            float_cols=("retention_rate",),
            string_cols=("platform", "post_id", "account_group", "discovery_hashtag", "content_type", "art_description", "provenance", "source_dataset"),
            timestamp_cols=("timestamp", "collection_timestamp"),
            non_negative_cols=("follower_count", "likes", "comments_count", "reposts", "shares"),
            pii_id_cols=("raw_account_ref",),
            drop_raw_pii_cols=("raw_account_ref",),
            pii_text_cols=("art_description",),
            notes=(
                "40 matched Human vs AI Instagram art pairs (80 posts).",
                "Reposts included; reach/impressions/saves/views/retention_rate retained as explicit pd.NA.",
            ),
        ),
    )
    save_clean_artifacts(posts_clean, posts_report, out, "posts")

    comments_clean, comments_report = run_cleaning_pipeline(
        comments_raw,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="comments",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "comment_id"),
            mandatory_cols=("platform", "comment_id", "post_id", "comment_text"),
            int_cols=("comment_rank_index", "chronological_order_index"),
            string_cols=("platform", "comment_id", "post_id", "account_group", "comment_text", "provenance", "source_dataset", "eval_split"),
            timestamp_cols=("comment_timestamp",),
            pii_text_cols=("comment_text",),
            mojibake_text_cols=("comment_text",),
            notes=(
                "Unpivoted from wide `comment_1..comment_50` tables; preserves BOTH scrape display `comment_rank_index` and sorted `chronological_order_index`.",
            ),
        ),
    )
    save_clean_artifacts(comments_clean, comments_report, out, "comments")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"posts": "posts.parquet", "comments": "comments.parquet"},
        "row_counts": {"posts": int(len(posts_clean)), "comments": int(len(comments_clean))},
        "metric_availability": posts_report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": [],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M1", "M3", "M6", "M7"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def prepare_instagram_comments_sentiment_hf(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `instagram_comments_sentiment_hf` preserving strict train/test isolation & PII redaction."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "instagram_comments_sentiment_hf" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "instagram_comments_sentiment_hf")
    spec = get_dataset_spec("instagram_comments_sentiment_hf")

    tr = pd.read_csv(raw / "train.csv", dtype={"comment_id": str, "user_id": str})
    tr["eval_split"] = "train"
    te = pd.read_csv(raw / "test.csv", dtype={"comment_id": str, "user_id": str})
    te["eval_split"] = "test"
    combined = pd.concat([tr, te], ignore_index=True)

    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    raw_df = pd.DataFrame({
        "platform": "instagram",
        "comment_id": combined["comment_id"].astype("string"),
        "post_id": combined["shortcode"].astype("string"),
        "raw_user_id": combined["user_id"].astype("string"),
        "raw_username": combined["username"].astype("string"),
        "comment_text": combined["text"].astype("string"),
        "comment_timestamp": combined["created_at"],
        "likes": combined["likes"],
        "category": combined["category"].astype("string"),
        "eval_split": combined["eval_split"].astype("string"),
        "ground_truth_sentiment": combined["sentiment_label_rater1"].astype(str).str.strip().str.lower().map(label_map).fillna("Neutral"),
        "provenance": spec.provenance.value,
        "source_dataset": spec.dataset_id,
    })

    clean_df, report = run_cleaning_pipeline(
        raw_df,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="comments",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "comment_id", "eval_split"),
            mandatory_cols=("platform", "comment_id", "post_id", "comment_text", "eval_split"),
            int_cols=("likes",),
            string_cols=("platform", "comment_id", "post_id", "comment_text", "category", "eval_split", "ground_truth_sentiment", "provenance", "source_dataset"),
            timestamp_cols=("comment_timestamp",),
            non_negative_cols=("likes",),
            pii_id_cols=("raw_user_id",),
            drop_raw_pii_cols=("raw_user_id", "raw_username"),
            pii_text_cols=("comment_text",),
            mojibake_text_cols=("comment_text",),
            split_col="eval_split",
            split_dedup_text_col="comment_text",
            quarantine_leakage_cols=("ground_truth_sentiment",),
            notes=(
                "Raw `user_id` pseudonymized and `username` dropped; inline `@handle` replaced with `[USER]`.",
                "`eval_split == 'test'` (56 rows) strictly isolated for held-out M3 evaluation.",
            ),
        ),
    )
    save_clean_artifacts(clean_df, report, out, "comments")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"comments": "comments.parquet"},
        "row_counts": {
            "comments": int(len(clean_df)),
            "train": int((clean_df["eval_split"] == "train").sum()),
            "test": int((clean_df["eval_split"] == "test").sum()),
        },
        "metric_availability": report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": report["stage_10_leakage_quarantined_columns"],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M3"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def prepare_zenodo_lokananta_comments_ig(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `zenodo_lokananta_comments_ig` recovering UTF-8/CP1252 mojibake cleanly."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "zenodo_lokananta_comments_ig" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "zenodo_lokananta_comments_ig")
    spec = get_dataset_spec("zenodo_lokananta_comments_ig")

    src_df = pd.read_csv(raw / "lokananta_instagram_comments.csv", encoding="utf-8")
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    raw_df = pd.DataFrame({
        "platform": "instagram",
        "comment_id": [f"loka_{i:04d}" for i in range(len(src_df))],
        "post_id": "lokananta_heritage_ig",
        "comment_text": src_df["Komentar"].astype("string"),
        "ground_truth_sentiment": src_df["Label"].astype(str).str.strip().str.lower().map(label_map).fillna("Neutral"),
        "language_hint": "id",
        "eval_split": "test",
        "provenance": spec.provenance.value,
        "source_dataset": spec.dataset_id,
    })

    clean_df, report = run_cleaning_pipeline(
        raw_df,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="comments",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "comment_id"),
            mandatory_cols=("platform", "comment_id", "post_id", "comment_text"),
            string_cols=("platform", "comment_id", "post_id", "comment_text", "ground_truth_sentiment", "language_hint", "eval_split", "provenance", "source_dataset"),
            pii_text_cols=("comment_text",),
            mojibake_text_cols=("comment_text",),
            split_col="eval_split",
            quarantine_leakage_cols=("ground_truth_sentiment",),
            notes=(
                "Reversed UTF-8/CP1252/Latin-1 double-encoded emoji mojibake (e.g., `ðŸ˜¤` -> `😤`).",
                "Predominantly Indonesian comments; intended to exercise M3 non-English handling.",
            ),
        ),
    )
    save_clean_artifacts(clean_df, report, out, "comments")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"comments": "comments.parquet"},
        "row_counts": {"comments": int(len(clean_df))},
        "metric_availability": report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": report["stage_10_leakage_quarantined_columns"],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M3"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def prepare_zenodo_sponsored_unsponsored_ig(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `zenodo_sponsored_unsponsored_ig` as an observational non-causal cohort."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "zenodo_sponsored_unsponsored_ig" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "zenodo_sponsored_unsponsored_ig")
    spec = get_dataset_spec("zenodo_sponsored_unsponsored_ig")

    sp = pd.read_csv(raw / "sponsored_posts.csv", dtype=str)
    sp["treatment_group"] = "sponsored"
    un = pd.read_csv(raw / "unsponsored_posts.csv", dtype=str)
    un["treatment_group"] = "unsponsored"
    combined = pd.concat([sp, un], ignore_index=True)

    # Derive approximate IG Snowflake creation timestamp? No — leave collection_timestamp explicit and don't synthesize fake timestamps!
    raw_df = pd.DataFrame({
        "platform": "instagram",
        "post_id": combined["id"].astype("string"),
        "raw_user_id": combined["user_id"].astype("string"),
        "content_type": "image",
        "treatment_group": combined["treatment_group"].astype("string"),
        "experiment_type": ExperimentType.OBSERVATIONAL_COMPARISON.value,
        "follower_count": combined["user_followers"],
        "likes": combined["likes"],
        "comments_count": combined["comments"],
        "views": pd.NA,
        "impressions": pd.NA,
        "reach": pd.NA,
        "shares": pd.NA,
        "saves": pd.NA,
        "retention_rate": pd.NA,
        "provenance": spec.provenance.value,
        "source_dataset": spec.dataset_id,
    })
    raw_df["collection_timestamp"] = pd.Timestamp("2021-07-13T00:00:00Z")

    clean_df, report = run_cleaning_pipeline(
        raw_df,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="posts",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "post_id", "treatment_group"),
            mandatory_cols=("platform", "post_id", "collection_timestamp"),
            int_cols=("follower_count", "likes", "comments_count", "views", "impressions", "reach", "shares", "saves"),
            float_cols=("retention_rate",),
            string_cols=("platform", "post_id", "content_type", "treatment_group", "experiment_type", "provenance", "source_dataset"),
            timestamp_cols=("collection_timestamp",),
            non_negative_cols=("follower_count", "likes", "comments_count"),
            pii_id_cols=("raw_user_id",),
            drop_raw_pii_cols=("raw_user_id",),
            notes=(
                "Read with string IDs to avoid float64 19-digit precision truncation on Instagram media IDs.",
                "Tagged `observational_comparison`; strictly non-causal without randomization.",
            ),
        ),
    )
    save_clean_artifacts(clean_df, report, out, "posts")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"posts": "posts.parquet"},
        "row_counts": {
            "posts": int(len(clean_df)),
            "sponsored": int((clean_df["treatment_group"] == "sponsored").sum()),
            "unsponsored": int((clean_df["treatment_group"] == "unsponsored").sum()),
        },
        "metric_availability": report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": [],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M1", "M4", "M5"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def _parse_excel_mmss_duration_seconds(val: Any) -> float | Any:
    """Decode Zenodo 23301344 Excel duration values (fractional days encoding MM:SS as HH:MM, or '41sec')."""
    if val is None or val is pd.NA:
        return np.nan
    if isinstance(val, (int, float, np.number)):
        f = float(val)
        if np.isnan(f):
            return np.nan
        # Excel stored MM:SS entries like 1:39 as 1h39m -> fractional days * 1440 = 99 seconds
        return round(f * 1440.0, 2) if 0.0 <= f < 1.0 else round(f, 2)
    s = str(val).strip().lower()
    if s.endswith("sec"):
        try:
            return float(s.replace("sec", "").strip())
        except ValueError:
            return np.nan
    return np.nan


def prepare_zenodo_educational_reels_ig(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `zenodo_educational_reels_ig` from worksheet `Analysis` using stdlib XML reader."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "zenodo_educational_reels_ig" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "zenodo_educational_reels_ig")
    spec = get_dataset_spec("zenodo_educational_reels_ig")

    xlsx_path = raw / "Reels analysis.xlsx"
    sheet_df = read_xlsx_sheet_to_dataframe(xlsx_path, sheet_index="Analysis")
    subheaders = sheet_df.iloc[0].fillna(sheet_df.columns.to_series()).tolist()
    sheet_df = sheet_df.iloc[1:].reset_index(drop=True)
    # Build unique clean header names
    clean_cols: List[str] = []
    for i, sh in enumerate(subheaders):
        name = str(sh).strip() if sh is not None and str(sh).strip() else f"col_{i}"
        if name in clean_cols:
            name = f"{name}_{i}"
        clean_cols.append(name)
    sheet_df.columns = clean_cols

    # Extract shortcode from Instagram Reel URL without persisting the URL/Author handle
    url_shortcodes = sheet_df["URL"].astype(str).str.extract(r"/reel/([^/?]+)")[0]
    vid_ids = url_shortcodes.fillna("edureel_" + pd.to_numeric(sheet_df["Video ID"], errors="coerce").fillna(-1).astype(int).astype(str))

    raw_df = pd.DataFrame({
        "platform": "instagram",
        "post_id": vid_ids.astype("string"),
        "raw_author": sheet_df["Author"].astype("string"),
        "content_type": "reel",
        "timestamp": sheet_df["Publication date"],
        "collection_timestamp": sheet_df["Analysis date"],
        "follower_count": sheet_df["Followers"],
        "views": sheet_df["Views"],
        "comments_count": sheet_df["Comments"],
        "impressions": pd.NA,
        "reach": pd.NA,
        "likes": pd.NA,
        "shares": pd.NA,
        "saves": pd.NA,
        "retention_rate": pd.NA,
        "duration_seconds": [ _parse_excel_mmss_duration_seconds(v) for v in sheet_df["Lenght"] ],
        "hook_duration_seconds": [ _parse_excel_mmss_duration_seconds(v) for v in sheet_df["Duration"] ],
        "topic": sheet_df["Topic"].astype("string"),
        "hook_type": sheet_df["Type"].astype("string"),
        "delivery_style": sheet_df["Delivery style"].astype("string"),
        "cta_type": sheet_df["CTA"].astype("string"),
        "education_value_score": sheet_df["Education value"],
        "attention_potential_score": sheet_df["Attention retention potential"],
        "execution_quality_score": sheet_df["Content execution quality"],
        "provenance": spec.provenance.value,
        "source_dataset": spec.dataset_id,
    })

    clean_df, report = run_cleaning_pipeline(
        raw_df,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="posts",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "post_id"),
            mandatory_cols=("platform", "post_id", "timestamp", "collection_timestamp"),
            shorthand_int_cols=("follower_count", "views"),
            int_cols=("comments_count", "impressions", "reach", "likes", "shares", "saves"),
            float_cols=("retention_rate", "duration_seconds", "hook_duration_seconds", "education_value_score", "attention_potential_score", "execution_quality_score"),
            string_cols=("platform", "post_id", "content_type", "topic", "hook_type", "delivery_style", "cta_type", "provenance", "source_dataset"),
            timestamp_cols=("timestamp", "collection_timestamp"),
            excel_serial_date_cols=("timestamp", "collection_timestamp"),
            non_negative_cols=("follower_count", "views", "comments_count", "duration_seconds"),
            pii_id_cols=("raw_author",),
            drop_raw_pii_cols=("raw_author",),
            notes=(
                "Parsed natively from `Analysis` worksheet using stdlib XML `.xlsx` reader.",
                "Compact follower/view strings (`30.5K`, `1.2M`) expanded deterministically without zero-imputing missing rows.",
            ),
        ),
    )
    save_clean_artifacts(clean_df, report, out, "posts")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"posts": "posts.parquet"},
        "row_counts": {"posts": int(len(clean_df))},
        "metric_availability": report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": [],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M1", "M5"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def prepare_instagram_benchmark_source(
    dataset_id: str,
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `instagram_engagement_eda_hf` or `instagram_analytics_github_alt` with leakage quarantine."""
    if dataset_id not in {"instagram_engagement_eda_hf", "instagram_analytics_github_alt"}:
        raise ValueError(f"Unsupported benchmark dataset_id: {dataset_id}")
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / dataset_id / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / dataset_id)
    spec = get_dataset_spec(dataset_id)

    df_src = pd.read_csv(raw / "Instagram_Analytics.csv")
    ts_raw = df_src["post_datetime"] if "post_datetime" in df_src.columns else df_src["upload_date"]
    raw_df = pd.DataFrame({
        "platform": "instagram",
        "post_id": df_src["post_id"].astype("string"),
        "timestamp": ts_raw,
        "collection_timestamp": ts_raw,
        "content_type": df_src["media_type"].astype(str).str.strip().str.lower(),
        "views": pd.NA,
        "impressions": df_src["impressions"],
        "reach": df_src["reach"],
        "likes": df_src["likes"],
        "comments_count": df_src["comments"],
        "shares": df_src["shares"],
        "saves": df_src["saves"],
        "follower_count": df_src["follower_count"] if "follower_count" in df_src.columns else pd.NA,
        "retention_rate": pd.NA,
        "content_category": df_src["content_category"].astype("string"),
        "traffic_source": df_src["traffic_source"].astype("string"),
        "caption_length": df_src["caption_length"],
        "hashtags_count": df_src["hashtags_count"],
        "followers_gained": df_src["followers_gained"],
        "engagement_rate_reported": df_src["engagement_rate"],
        "provenance": spec.provenance.value,
        "source_dataset": spec.dataset_id,
    })
    if "account_id" in df_src.columns:
        raw_df["raw_account_id"] = df_src["account_id"].astype("string")
    if "has_call_to_action" in df_src.columns:
        raw_df["has_call_to_action"] = df_src["has_call_to_action"]
    if "performance_bucket_label" in df_src.columns:
        raw_df["performance_bucket_label"] = df_src["performance_bucket_label"].astype("string")

    # Audit post_id overlap with sibling benchmark if present
    sibling_id = "instagram_analytics_github_alt" if dataset_id == "instagram_engagement_eda_hf" else "instagram_engagement_eda_hf"
    sibling_csv = base / DEFAULT_EXTERNAL_ROOT / sibling_id / "raw" / "Instagram_Analytics.csv"
    overlap_note = None
    if sibling_csv.exists():
        sib_ids = set(pd.read_csv(sibling_csv, usecols=["post_id"])["post_id"].astype(str))
        shared = len(set(raw_df["post_id"].astype(str)) & sib_ids)
        overlap_note = f"Empirical overlap with `{sibling_id}`: {shared:,} shared post_ids."

    clean_df, report = run_cleaning_pipeline(
        raw_df,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="posts",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="owner_oauth",
            composite_key_cols=("platform", "post_id"),
            mandatory_cols=("platform", "post_id", "timestamp", "collection_timestamp"),
            int_cols=("views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count", "caption_length", "hashtags_count", "followers_gained"),
            float_cols=("retention_rate", "engagement_rate_reported"),
            string_cols=("platform", "post_id", "content_type", "content_category", "traffic_source", "provenance", "source_dataset"),
            timestamp_cols=("timestamp", "collection_timestamp"),
            non_negative_cols=("impressions", "reach", "likes", "comments_count", "shares", "saves"),
            pii_id_cols=("raw_account_id",) if "raw_account_id" in raw_df.columns else (),
            drop_raw_pii_cols=("raw_account_id",) if "raw_account_id" in raw_df.columns else (),
            quarantine_leakage_cols=("performance_bucket_label", "engagement_rate_reported", "engagement_rate_recomputed", "followers_gained"),
            recompute_engagement_rate=True,
            recompute_save_rate=True,
            recompute_share_rate=True,
            notes=tuple(x for x in (overlap_note, "Quarantined `performance_bucket_label` and `engagement_rate` from pre-publication ML inputs.") if x),
        ),
    )
    save_clean_artifacts(clean_df, report, out, "posts")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"posts": "posts.parquet"},
        "row_counts": {"posts": int(len(clean_df))},
        "metric_availability": report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": report["stage_10_leakage_quarantined_columns"],
        "limitations": list(spec.limitations) + ([overlap_note] if overlap_note else []),
        "recommended_modules": ["M1_benchmark", "M2_benchmark"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def prepare_social_instagram_marketing_hf(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize `social_instagram_marketing_hf` clearly tagged as `SYNTHETIC_DATA`."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw = Path(raw_dir) if raw_dir is not None else (base / DEFAULT_EXTERNAL_ROOT / "social_instagram_marketing_hf" / "raw")
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "social_instagram_marketing_hf")
    spec = get_dataset_spec("social_instagram_marketing_hf")

    src = pd.read_parquet(raw / "social_dataset.parquet")
    fmt_map = {"feed post": "image", "story": "story"}
    raw_df = pd.DataFrame({
        "platform": "instagram",
        "post_id": [f"ig_mkt_syn_{i:05d}" for i in range(len(src))],
        "timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
        "collection_timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
        "content_type": src["format"].astype(str).str.strip().str.lower().map(fmt_map).fillna("image"),
        "business_type": src["business_type"].astype("string"),
        "brand_voice": src["brand_voice"].astype("string"),
        "occasion": src["occasion"].astype("string"),
        "caption": src["caption"].astype("string"),
        "hashtags": src["hashtags"].astype("string"),
        "cta": src["cta"].astype("string"),
        "views": pd.NA,
        "impressions": pd.NA,
        "reach": pd.NA,
        "likes": pd.NA,
        "comments_count": pd.NA,
        "shares": pd.NA,
        "saves": pd.NA,
        "follower_count": pd.NA,
        "retention_rate": pd.NA,
        "provenance": spec.provenance.value,
        "source_dataset": spec.dataset_id,
    })

    clean_df, report = run_cleaning_pipeline(
        raw_df,
        CleaningConfig(
            dataset_id=spec.dataset_id,
            table_name="posts",
            platform="instagram",
            provenance=spec.provenance,
            access_tier="public_api",
            composite_key_cols=("platform", "post_id"),
            mandatory_cols=("platform", "post_id", "timestamp", "collection_timestamp"),
            int_cols=("views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count"),
            float_cols=("retention_rate",),
            string_cols=("platform", "post_id", "content_type", "business_type", "brand_voice", "occasion", "caption", "hashtags", "cta", "provenance", "source_dataset"),
            timestamp_cols=("timestamp", "collection_timestamp"),
            pii_text_cols=("caption",),
            mojibake_text_cols=("caption",),
            notes=("Explicitly marked `SYNTHETIC_DATA`; campaign text corpus without real engagement outcomes.",),
        ),
    )
    save_clean_artifacts(clean_df, report, out, "posts")

    meta = {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "platform": "instagram",
        "platforms": ["instagram"],
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "pinned_revision": spec.pinned_revision,
        "primary_tables": {"posts": "posts.parquet"},
        "row_counts": {"posts": int(len(clean_df))},
        "metric_availability": report["stage_7_metric_availability"],
        "proxy_metrics": {},
        "leakage_quarantined_columns": [],
        "limitations": list(spec.limitations),
        "recommended_modules": ["M1_synthetic", "M5_synthetic"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_dataset_meta(out, meta)
    return meta


def adapt_youtube_trending_dataframe(
    df: pd.DataFrame,
    *,
    region: str = "UNKNOWN",
    dataset_id: str = "youtube_trending_kaggle",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize a YouTube Trending CSV chunk/table while preserving `(video_id, region, trending_date)` snapshots.

    Honesty Guarantees:
    - Never maps `comments_disabled=True` -> `comments_count = 0`; preserves explicit `pd.NA`.
    - Never maps `ratings_disabled=True` -> `likes = 0`; preserves explicit `pd.NA`.
    - Leaves `saves = pd.NA` (`not_applicable` on YouTube) and `impressions/reach/shares/retention_rate = pd.NA`.
    """
    work = df.copy()
    vid_col = "video_id" if "video_id" in work.columns else "id"
    tr_col = "trending_date" if "trending_date" in work.columns else "collection_timestamp"
    pub_col = "publishedAt" if "publishedAt" in work.columns else ("publish_time" if "publish_time" in work.columns else "timestamp")
    chan_col = "channelId" if "channelId" in work.columns else ("channelTitle" if "channelTitle" in work.columns else None)
    views_col = "view_count" if "view_count" in work.columns else "views"
    comm_col = "comment_count" if "comment_count" in work.columns else "comments_count"

    # Kaggle trending_date uses formats like '20.12.08' (%y.%d.%m) or ISO
    tr_raw = work[tr_col].astype("string")
    tr_iso = pd.to_datetime(tr_raw, format="%y.%d.%m", utc=True, errors="coerce")
    tr_iso = tr_iso.fillna(pd.to_datetime(tr_raw, format="mixed", utc=True, errors="coerce"))

    likes_series = pd.to_numeric(work["likes"], errors="coerce").astype("Float64")
    if "ratings_disabled" in work.columns:
        rd_mask = work["ratings_disabled"].astype(str).str.lower().isin({"true", "1", "yes"})
        likes_series = likes_series.mask(rd_mask, pd.NA)

    comms_series = pd.to_numeric(work[comm_col], errors="coerce").astype("Float64")
    if "comments_disabled" in work.columns:
        cd_mask = work["comments_disabled"].astype(str).str.lower().isin({"true", "1", "yes"})
        comms_series = comms_series.mask(cd_mask, pd.NA)

    norm = pd.DataFrame({
        "platform": "youtube",
        "post_id": work[vid_col].astype("string"),
        "region": str(region).upper(),
        "trending_date": tr_raw.astype("string"),
        "timestamp": work[pub_col],
        "collection_timestamp": tr_iso,
        "content_type": "video",
        "views": work[views_col],
        "impressions": pd.NA,
        "reach": pd.NA,
        "likes": likes_series,
        "comments_count": comms_series,
        "shares": pd.NA,
        "saves": pd.NA,
        "follower_count": pd.NA,
        "retention_rate": pd.NA,
        "category_id": work["categoryId"] if "categoryId" in work.columns else pd.NA,
        "title": work["title"].astype("string") if "title" in work.columns else pd.NA,
        "provenance": SourceProvenance.REAL_PUBLIC_DATA.value,
        "source_dataset": dataset_id,
    })
    if chan_col is not None:
        norm["raw_channel_ref"] = work[chan_col].astype("string")

    cfg = CleaningConfig(
        dataset_id=dataset_id,
        table_name="posts",
        platform="youtube",
        provenance=SourceProvenance.REAL_PUBLIC_DATA,
        access_tier="public_api",
        composite_key_cols=("platform", "post_id", "region", "trending_date"),
        mandatory_cols=("platform", "post_id", "region", "trending_date", "timestamp", "collection_timestamp"),
        int_cols=("views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count"),
        float_cols=("retention_rate",),
        string_cols=("platform", "post_id", "region", "trending_date", "content_type", "title", "provenance", "source_dataset"),
        timestamp_cols=("timestamp", "collection_timestamp"),
        non_negative_cols=("views", "likes", "comments_count"),
        pii_id_cols=("raw_channel_ref",) if "raw_channel_ref" in norm.columns else (),
        drop_raw_pii_cols=("raw_channel_ref",) if "raw_channel_ref" in norm.columns else (),
        pii_text_cols=("title",),
        proxy_metrics_used=("youtube_trending_snapshot",),
        notes=(
            "Composite snapshot key `(platform, post_id, region, trending_date)` preserves longitudinal trending observations.",
            "`comments_disabled`/`ratings_disabled` map cleanly to `pd.NA` (never false zeros).",
        ),
    )
    return run_cleaning_pipeline(norm, cfg)


def prepare_all_installed_external_datasets(
    *,
    root_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run adapters for every downloaded dataset present on disk under `data/external/`."""
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw_root = base / DEFAULT_EXTERNAL_ROOT

    adapters: List[Tuple[str, Callable[..., Dict[str, Any]], Tuple[str, ...]]] = [
        ("tiktok_engagement_200k", prepare_tiktok_engagement_200k, ("videos.parquet", "engagement_daily.parquet", "creator_daily.parquet")),
        ("instaart_human_ai", prepare_instaart_human_ai, ("Posts_Primary.csv", "Human_Comments_Data.csv", "AI_Comments_Data.csv")),
        ("instagram_comments_sentiment_hf", prepare_instagram_comments_sentiment_hf, ("train.csv", "test.csv")),
        ("zenodo_lokananta_comments_ig", prepare_zenodo_lokananta_comments_ig, ("lokananta_instagram_comments.csv",)),
        ("zenodo_sponsored_unsponsored_ig", prepare_zenodo_sponsored_unsponsored_ig, ("sponsored_posts.csv", "unsponsored_posts.csv")),
        ("zenodo_educational_reels_ig", prepare_zenodo_educational_reels_ig, ("Reels analysis.xlsx",)),
        ("instagram_engagement_eda_hf", lambda **kw: prepare_instagram_benchmark_source("instagram_engagement_eda_hf", **kw), ("Instagram_Analytics.csv",)),
        ("instagram_analytics_github_alt", lambda **kw: prepare_instagram_benchmark_source("instagram_analytics_github_alt", **kw), ("Instagram_Analytics.csv",)),
        ("social_instagram_marketing_hf", prepare_social_instagram_marketing_hf, ("social_dataset.parquet",)),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for ds_id, fn, req_files in adapters:
        ds_raw = raw_root / ds_id / "raw"
        if all((ds_raw / rf).exists() for rf in req_files):
            results[ds_id] = fn(root_dir=base)
        else:
            results[ds_id] = {"dataset_id": ds_id, "status": "NOT_DOWNLOADED"}

    # Check Kaggle-gated YouTube Trending dynamically if CSVs were later placed on disk
    yt_raw = raw_root / "youtube_trending_kaggle" / "raw"
    yt_csvs = sorted(yt_raw.glob("*trending*.csv")) if yt_raw.exists() else []
    if yt_csvs:
        yt_frames: List[pd.DataFrame] = []
        last_rep: Dict[str, Any] = {}
        for csv_p in yt_csvs:
            region = csv_p.name.split("_")[0].upper() if "_" in csv_p.name else "UNKNOWN"
            df_chunk = pd.read_csv(csv_p, low_memory=False)
            c_df, last_rep = adapt_youtube_trending_dataframe(df_chunk, region=region)
            yt_frames.append(c_df)
        merged_yt = pd.concat(yt_frames, ignore_index=True)
        yt_out = base / DEFAULT_PROCESSED_EXTERNAL_ROOT / "youtube_trending_kaggle"
        save_clean_artifacts(merged_yt, last_rep, yt_out, "posts")
        yt_spec = get_dataset_spec("youtube_trending_kaggle")
        yt_meta = {
            "dataset_id": "youtube_trending_kaggle",
            "display_name": yt_spec.display_name,
            "platform": "youtube",
            "platforms": ["youtube"],
            "provenance": yt_spec.provenance.value,
            "license_name": yt_spec.license_name,
            "license_status": yt_spec.license_status,
            "pinned_revision": yt_spec.pinned_revision,
            "primary_tables": {"posts": "posts.parquet"},
            "row_counts": {"posts": int(len(merged_yt))},
            "metric_availability": last_rep.get("stage_7_metric_availability", {}),
            "proxy_metrics": {"youtube_trending_snapshot": PROXY_METRIC_DEFINITIONS["youtube_trending_snapshot"]},
            "leakage_quarantined_columns": [],
            "limitations": list(yt_spec.limitations),
            "recommended_modules": ["M1", "M2", "M6", "M7"],
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_dataset_meta(yt_out, yt_meta)
        results["youtube_trending_kaggle"] = yt_meta
    else:
        results["youtube_trending_kaggle"] = {
            "dataset_id": "youtube_trending_kaggle",
            "status": "BLOCKED_MISSING_CREDENTIALS",
        }

    return results

