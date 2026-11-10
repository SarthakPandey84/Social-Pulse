"""Engagement Metrics and Growth Correlation Analytics."""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.data_ingestion.schemas import PROXY_METRIC_DEFINITIONS
from src.preprocessing.cleaning import get_latest_posts_view

VIDEO_CONTENT_TYPES = {"reel", "short", "video", "short video", "short_video"}
PLATFORM_INCOMPATIBLE_CONTENT_TYPES = {
    ("instagram", "short"),
    ("youtube", "reel"),
}


def is_retention_applicable(platform: Any, content_type: Any) -> bool:
    """Return True iff content_type is a valid, platform-compatible video format.

    Structural rules (aligned with unified post schema):
    - Only explicit video formats ('reel', 'short', 'video', 'short video', 'short_video') support retention_rate.
    - Static/multi-slide formats ('carousel', 'image', 'text'), unknown/missing content_type,
      and platform-incompatible pairings (e.g. YouTube 'reel', Instagram 'short') return False.
    """
    if pd.isna(content_type) or content_type is None:
        return False
    c_norm = str(content_type).strip().lower()
    if c_norm not in VIDEO_CONTENT_TYPES:
        return False
    if pd.notna(platform) and platform is not None:
        p_norm = str(platform).strip().lower()
        if (p_norm, c_norm) in PLATFORM_INCOMPATIBLE_CONTENT_TYPES:
            return False
    return True


def _vectorized_retention_applicable_mask(df: pd.DataFrame) -> np.ndarray:
    """Fast vectorized check matching `is_retention_applicable` row-by-row."""
    if "content_type" not in df.columns:
        return np.ones(len(df), dtype=bool)
    c_raw = df["content_type"]
    c_norm = c_raw.astype("string").str.strip().str.lower()
    is_vid = c_raw.notna().to_numpy() & c_norm.isin(VIDEO_CONTENT_TYPES).fillna(False).to_numpy()
    if "platform" not in df.columns:
        return is_vid
    p_raw = df["platform"]
    p_norm = p_raw.astype("string").str.strip().str.lower()
    incompat = (
        p_raw.notna().to_numpy()
        & (
            ((p_norm == "instagram").fillna(False).to_numpy() & (c_norm == "short").fillna(False).to_numpy())
            | ((p_norm == "youtube").fillna(False).to_numpy() & (c_norm == "reel").fillna(False).to_numpy())
        )
    )
    return is_vid & (~incompat)


class EngagementAnalytics:
    """Computes global performance indicators, multi-source filters, creator baselines, and growth correlations."""

    @staticmethod
    def calculate_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate platform-wide key performance indicators on deduplicated latest snapshots."""
        if df is None or len(df) == 0:
            return {
                "total_posts": 0,
                "total_views": 0,
                "total_shares": 0,
                "total_saves": 0,
                "avg_retention_pct": 0.0,
                "overall_save_to_share_ratio": 0.0,
            }

        work_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        total_posts = len(work_df)
        total_views = int(work_df["views"].dropna().sum()) if "views" in work_df.columns else 0
        total_shares = int(work_df["shares"].dropna().sum()) if "shares" in work_df.columns else 0
        total_saves = int(work_df["saves"].dropna().sum()) if "saves" in work_df.columns else 0

        if "retention_rate" in work_df.columns:
            if "content_type" in work_df.columns:
                video_mask = _vectorized_retention_applicable_mask(work_df)
                ret_series = work_df.loc[video_mask, "retention_rate"].dropna()
            else:
                ret_series = work_df["retention_rate"].dropna()
            avg_retention = float(ret_series.mean()) if len(ret_series) > 0 else 0.0
        else:
            avg_retention = 0.0

        save_share_ratio = round(total_saves / max(total_shares, 1), 2)

        return {
            "total_posts": total_posts,
            "total_views": total_views,
            "total_shares": total_shares,
            "total_saves": total_saves,
            "avg_retention_pct": round(avg_retention * 100, 1),
            "overall_save_to_share_ratio": save_share_ratio
        }

    @staticmethod
    def filter_posts_multi(
        df: Optional[pd.DataFrame],
        *,
        source_dataset: Optional[str] = None,
        platform: Optional[str] = None,
        content_type: Optional[str] = None,
        topic: Optional[str] = None,
        region: Optional[str] = None,
        creator_id: Optional[str] = None,
        date_range: Optional[Tuple[Any, Any]] = None,
        age_window: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter multi-platform datasets cleanly across source, platform, region, creator, topic, date, and age slice."""
        if df is None or len(df) == 0:
            return pd.DataFrame() if df is None else df.copy()

        mask = np.ones(len(df), dtype=bool)
        if source_dataset and str(source_dataset).lower() != "all" and "source_dataset" in df.columns:
            mask &= (df["source_dataset"].astype("string").str.strip() == str(source_dataset).strip()).fillna(False).to_numpy()
        if platform and str(platform).lower() != "all" and "platform" in df.columns:
            mask &= (df["platform"].astype("string").str.strip().str.lower() == str(platform).strip().lower()).fillna(False).to_numpy()
        if content_type and str(content_type).lower() != "all" and "content_type" in df.columns:
            mask &= (df["content_type"].astype("string").str.strip().str.lower() == str(content_type).strip().lower()).fillna(False).to_numpy()
        if topic and str(topic).lower() != "all" and "topic" in df.columns:
            mask &= (df["topic"].astype("string").str.strip().str.lower() == str(topic).strip().lower()).fillna(False).to_numpy()
        if region and str(region).lower() != "all" and "region" in df.columns:
            mask &= (df["region"].astype("string").str.strip().str.upper() == str(region).strip().upper()).fillna(False).to_numpy()

        creator_col = "creator_pseudo_id" if "creator_pseudo_id" in df.columns else ("author_id" if "author_id" in df.columns else None)
        if creator_id and str(creator_id).lower() != "all" and creator_col is not None:
            mask &= (df[creator_col].astype("string").str.strip() == str(creator_id).strip()).fillna(False).to_numpy()

        if date_range is not None and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            start_dt, end_dt = date_range
            if start_dt is not None:
                mask &= (ts >= pd.to_datetime(start_dt, utc=True)).fillna(False).to_numpy()
            if end_dt is not None:
                mask &= (ts <= pd.to_datetime(end_dt, utc=True)).fillna(False).to_numpy()

        out = df.loc[mask].copy()
        if age_window and str(age_window).lower() in {"day0", "day1", "day7", "day30"}:
            prefix = str(age_window).lower()
            remap = {
                f"{prefix}_views": "views",
                f"{prefix}_likes": "likes",
                f"{prefix}_comments": "comments_count",
                f"{prefix}_shares": "shares",
                f"{prefix}_collects": "collect_count",
            }
            for src_col, dst_col in remap.items():
                if src_col in out.columns:
                    out[dst_col] = out[src_col]
            out["active_age_window"] = prefix
        return out

    @staticmethod
    def compute_creator_relative_metrics(
        df: pd.DataFrame,
        creator_col: str = "creator_pseudo_id",
        value_col: str = "views",
    ) -> pd.DataFrame:
        """Compute creator-relative normalization (`value / creator_median_value`) cleanly without imputation."""
        if df is None or len(df) == 0:
            return pd.DataFrame() if df is None else df.copy()
        out = df.copy()
        c_col = creator_col if creator_col in out.columns else ("author_id" if "author_id" in out.columns else None)
        if c_col is None or value_col not in out.columns:
            out["creator_baseline_median"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
            out["creator_relative_ratio"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
            out["creator_post_count"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
            return out

        vals = pd.to_numeric(out[value_col], errors="coerce")
        grp_median = vals.groupby(out[c_col], sort=False).transform("median")
        grp_count = vals.groupby(out[c_col], sort=False).transform("count")

        valid_ratio = vals.notna() & grp_median.notna() & (grp_median > 0)
        ratio = np.full(len(out), np.nan, dtype="float64")
        ratio[valid_ratio.to_numpy()] = (vals[valid_ratio] / grp_median[valid_ratio]).round(3).to_numpy()

        out["creator_baseline_median"] = grp_median.astype("Float64")
        out["creator_relative_ratio"] = pd.Series(ratio, index=out.index, dtype="Float64")
        out["creator_post_count"] = grp_count.astype("Int64")
        return out

    @staticmethod
    def summarize_metric_availability(df: pd.DataFrame) -> Dict[str, Any]:
        """Audit exact non-null metric coverage and disclose explicit platform proxy columns."""
        n = int(len(df)) if df is not None else 0
        fields = ["views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count", "retention_rate", "collect_count"]
        metrics_info: Dict[str, Dict[str, Any]] = {}
        for col in fields:
            if df is None or col not in df.columns or n == 0:
                metrics_info[col] = {"present_in_schema": bool(df is not None and col in df.columns), "observed_count": 0, "observed_pct": 0.0}
            else:
                obs = int(df[col].notna().sum())
                metrics_info[col] = {
                    "present_in_schema": True,
                    "observed_count": obs,
                    "observed_pct": round((obs / n) * 100.0, 2),
                }

        proxies: Dict[str, str] = {}
        if df is not None and "collect_count" in df.columns and df["collect_count"].notna().any():
            proxies["collect_count"] = PROXY_METRIC_DEFINITIONS["tiktok_collect_count"]
        if df is not None and "trending_date" in df.columns and df["trending_date"].notna().any():
            proxies["trending_date"] = PROXY_METRIC_DEFINITIONS["youtube_trending_snapshot"]

        provenances = sorted(df["provenance"].dropna().astype(str).unique().tolist()) if (df is not None and "provenance" in df.columns) else []
        platforms = sorted(df["platform"].dropna().astype(str).str.lower().unique().tolist()) if (df is not None and "platform" in df.columns) else []

        return {
            "row_count": n,
            "platforms": platforms,
            "provenances": provenances,
            "metrics": metrics_info,
            "proxy_disclosures": proxies,
        }

    @staticmethod
    def topic_follower_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """Map the relationship between struggle topics and new follower growth."""
        if df is None or len(df) == 0 or "new_followers" not in df.columns or "topic" not in df.columns:
            return pd.DataFrame()

        work_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        summary = work_df.groupby("topic").agg(
            posts_count=("post_id", "count"),
            total_new_followers=("new_followers", "sum"),
            avg_followers_per_post=("new_followers", "mean"),
            avg_saves=("saves", "mean"),
            avg_shares=("shares", "mean")
        ).reset_index()

        return summary.sort_values(by="total_new_followers", ascending=False).reset_index(drop=True)

