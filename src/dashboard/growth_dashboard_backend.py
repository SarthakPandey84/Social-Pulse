"""Backend Data Engine for Module 6: Growth Visualization Dashboard.

Provides decoupled, high-performance data loading, filtering, KPI calculations,
period-over-period comparisons, time-series aggregations, platform normalizations,
and multi-module synthesis (Modules 1-5).
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing.cleaning import get_latest_posts_view
from src.statistics.ab_testing import ABTestingFramework
from src.recommender.engine import OptimizationRecommender
from src.recommender.schemas import RecommendationReport


class GrowthDashboardBackend:
    """Analytical data engine powering the Module 6 Growth Visualization Dashboard."""

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        dataset_id: Optional[str] = None,
        posts_df: Optional[pd.DataFrame] = None,
        comments_df: Optional[pd.DataFrame] = None,
        experiments_df: Optional[pd.DataFrame] = None,
        observations_df: Optional[pd.DataFrame] = None,
        daily_summary_df: Optional[pd.DataFrame] = None,
        dataset_meta: Optional[Dict[str, Any]] = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parent.parent.parent
        self.dataset_id: str = str(dataset_id or "synthetic_demo").strip()
        self.posts_df: Optional[pd.DataFrame] = posts_df.copy() if posts_df is not None else None
        self.comments_df: Optional[pd.DataFrame] = comments_df.copy() if comments_df is not None else None
        self.experiments_df: Optional[pd.DataFrame] = experiments_df.copy() if experiments_df is not None else None
        self.observations_df: Optional[pd.DataFrame] = observations_df.copy() if observations_df is not None else None
        self.daily_summary_df: Optional[pd.DataFrame] = daily_summary_df.copy() if daily_summary_df is not None else None
        self.dataset_meta: Dict[str, Any] = dict(dataset_meta or {})

        if self.posts_df is None and self.comments_df is None:
            if self.dataset_id and self.dataset_id != "synthetic_demo":
                self.load_external_dataset(self.dataset_id)
            else:
                self.load_datasets()
        elif self.posts_df is not None and "timestamp" in self.posts_df.columns and "timestamp_dt" not in self.posts_df.columns:
            self.posts_df["timestamp_dt"] = pd.to_datetime(self.posts_df["timestamp"], utc=True, errors="coerce")

    def load_external_dataset(self, dataset_id: str) -> None:
        """Load an installed curated external dataset strictly from Parquet (zero silent fallback)."""
        ext_dir = self.root_dir / "data" / "processed" / "external" / dataset_id
        if not ext_dir.exists():
            raise FileNotFoundError(
                f"Installed external dataset '{dataset_id}' not found at {ext_dir}. "
                "Download and prepare the dataset first via scripts/download_external_datasets.py and scripts/prepare_external_datasets.py."
            )
        meta_path = ext_dir / "dataset_meta.json"
        if meta_path.exists():
            self.dataset_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        posts_path = ext_dir / "posts.parquet"
        comments_path = ext_dir / "comments.parquet"
        daily_path = ext_dir / "daily_platform_summary.parquet"

        self.posts_df = pd.read_parquet(posts_path) if posts_path.exists() else None
        self.comments_df = pd.read_parquet(comments_path) if comments_path.exists() else None
        self.daily_summary_df = pd.read_parquet(daily_path) if daily_path.exists() else None
        self.experiments_df = None
        self.observations_df = None

        if self.posts_df is not None and "timestamp" in self.posts_df.columns:
            self.posts_df["timestamp_dt"] = pd.to_datetime(self.posts_df["timestamp"], utc=True, errors="coerce")

    def load_datasets(self) -> None:
        """Load analytical datasets across Modules 1-4."""
        posts_path = self.root_dir / "data" / "processed" / "processed_posts.parquet"
        comments_path = self.root_dir / "data" / "processed" / "processed_comments.parquet"
        exp_path = self.root_dir / "data" / "sample" / "sample_experiments.csv"
        obs_path = self.root_dir / "data" / "processed" / "processed_ab_tests.parquet"

        # Fallbacks
        sample_posts_path = self.root_dir / "data" / "sample" / "sample_posts.csv"

        if posts_path.exists():
            self.posts_df = pd.read_parquet(posts_path)
        elif sample_posts_path.exists():
            self.posts_df = pd.read_csv(sample_posts_path)

        if comments_path.exists():
            self.comments_df = pd.read_parquet(comments_path)

        if exp_path.exists():
            self.experiments_df = pd.read_csv(exp_path)

        if obs_path.exists():
            self.observations_df = pd.read_parquet(obs_path)

        # Ensure datetime format for timestamps
        if self.posts_df is not None and "timestamp" in self.posts_df.columns:
            self.posts_df["timestamp_dt"] = pd.to_datetime(self.posts_df["timestamp"], utc=True, errors="coerce")

    # --- 1. FILTERING ENGINE ---

    def filter_posts(
        self,
        df: Optional[pd.DataFrame] = None,
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        platform: Optional[str] = None,
        content_type: Optional[str] = None,
        topic: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter posts multi-dimensionally while preserving schema and snapshot integrity."""
        source_df = df if df is not None else self.posts_df
        if source_df is None or len(source_df) == 0:
            return pd.DataFrame()

        filtered = source_df.copy()

        # Date range filtering
        if "timestamp_dt" not in filtered.columns and "timestamp" in filtered.columns:
            filtered["timestamp_dt"] = pd.to_datetime(filtered["timestamp"], utc=True, errors="coerce")

        if "timestamp_dt" in filtered.columns:
            if start_date is not None:
                start_dt = pd.to_datetime(start_date, utc=True)
                filtered = filtered[filtered["timestamp_dt"] >= start_dt]
            if end_date is not None:
                end_dt = pd.to_datetime(end_date, utc=True)
                filtered = filtered[filtered["timestamp_dt"] <= end_dt]

        # Platform filter
        if platform and platform.lower() != "all" and "platform" in filtered.columns:
            filtered = filtered[
                (filtered["platform"].astype("string").str.strip().str.lower() == platform.lower().strip()).fillna(False)
            ]

        # Content type filter
        if content_type and content_type.lower() != "all" and "content_type" in filtered.columns:
            filtered = filtered[
                (filtered["content_type"].astype("string").str.strip().str.lower() == content_type.lower().strip()).fillna(False)
            ]

        # Topic filter
        if topic and topic.lower() != "all" and "topic" in filtered.columns:
            filtered = filtered[
                (filtered["topic"].astype("string").str.strip().str.lower() == topic.lower().strip()).fillna(False)
            ]

        return filtered

    # --- 2. SECTION A: GROWTH OVERVIEW KPIS ---

    def calculate_growth_overview_kpis(
        self,
        df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Calculate high-level KPIs with strict denominator integrity and period comparison.

        Distinguishes between:
        1. Pooled engagement rate (total interactions / total views * 100).
        2. Arithmetic mean of post-level engagement rates (average of individual post rates).
        """
        if df is None or len(df) == 0:
            return {
                "total_posts": 0,
                "total_views": None,
                "total_impressions": None,
                "total_likes": None,
                "total_comments": None,
                "total_shares": None,
                "total_saves": None,
                "total_new_followers": None,
                "avg_views_per_post": None,
                "avg_reach_per_post": None,
                "overall_views_engagement_rate": None,
                "pooled_views_engagement_rate": None,
                "mean_post_level_views_engagement_rate": None,
                "overall_reach_engagement_rate": None,
                "pooled_reach_engagement_rate": None,
                "mean_post_level_reach_engagement_rate": None,
                "rate_definitions": {
                    "pooled_views_engagement_rate": "Volume-weighted total interactions divided by total views.",
                    "mean_post_level_views_engagement_rate": "Arithmetic mean of individual post-level views engagement rates.",
                    "pooled_reach_engagement_rate": "Volume-weighted total Instagram interactions divided by total Instagram reach.",
                    "mean_post_level_reach_engagement_rate": "Arithmetic mean of individual Instagram post reach rates."
                },
                "platform_metric_availability": {
                    "Instagram": {"saves_tracked": True, "reach_tracked": True, "denominators": ["views", "reach"]},
                    "YouTube": {"saves_tracked": False, "reach_tracked": False, "denominators": ["views"]},
                    "TikTok": {"saves_tracked": False, "collect_count_proxy_tracked": True, "reach_tracked": False, "denominators": ["views"]},
                },
                "total_collect_count_proxy": None,
                "connected_platforms": [],
                "platform_count": 0,
                "reporting_period": "N/A",
                "period_days": 0,
                "active_posting_days": 0,
                "missing_posting_dates": 0,
                "has_previous_period_comparison": False,
                "comparison_deltas": {},
                "comparison_note": "No data available.",
                "period_details": {},
                "metric_availability": {
                    "views": False,
                    "impressions": False,
                    "likes": False,
                    "comments_count": False,
                    "shares": False,
                    "saves": False,
                    "collect_count": False,
                    "new_followers": False,
                    "reach": False,
                },
            }

        # Deduplicate to latest post snapshots to avoid multiple snapshot inflation
        latest_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        n_posts = len(latest_df)

        # Totals & explicit observed-signal availability flags
        has_views = "views" in latest_df.columns and bool(latest_df["views"].notna().any())
        has_impressions = "impressions" in latest_df.columns and bool(latest_df["impressions"].notna().any())
        has_likes = "likes" in latest_df.columns and bool(latest_df["likes"].notna().any())
        has_comments = "comments_count" in latest_df.columns and bool(latest_df["comments_count"].notna().any())
        has_shares = "shares" in latest_df.columns and bool(latest_df["shares"].notna().any())
        has_saves = "saves" in latest_df.columns and bool(latest_df["saves"].notna().any())
        has_collects = "collect_count" in latest_df.columns and bool(latest_df["collect_count"].notna().any())
        has_new_followers = "new_followers" in latest_df.columns and bool(latest_df["new_followers"].notna().any())
        has_reach = "reach" in latest_df.columns and bool(latest_df["reach"].notna().any())

        views_series = latest_df["views"].dropna() if has_views else pd.Series(dtype=float)
        tot_views: Optional[int] = int(views_series.sum()) if has_views else None
        tot_impressions: Optional[int] = int(latest_df["impressions"].dropna().sum()) if has_impressions else None
        tot_likes: Optional[int] = int(latest_df["likes"].dropna().sum()) if has_likes else None
        tot_comments: Optional[int] = int(latest_df["comments_count"].dropna().sum()) if has_comments else None
        tot_shares: Optional[int] = int(latest_df["shares"].dropna().sum()) if has_shares else None
        tot_saves: Optional[int] = int(latest_df["saves"].dropna().sum()) if has_saves else None
        tot_new_followers: Optional[int] = int(latest_df["new_followers"].dropna().sum()) if has_new_followers else None
        tot_collect_count_proxy: Optional[int] = int(latest_df["collect_count"].dropna().sum()) if has_collects else None

        # Reach (Instagram insights)
        reach_series = latest_df["reach"].dropna() if has_reach else pd.Series(dtype=float)
        positive_reach_series = reach_series[reach_series > 0]
        if not has_reach:
            avg_reach_per_post: Optional[float] = None
        elif len(positive_reach_series) > 0:
            avg_reach_per_post = round(float(positive_reach_series.sum()) / len(positive_reach_series), 1)
        else:
            # Reach was genuinely measured as 0 across observed rows
            avg_reach_per_post = 0.0

        avg_views_per_post: Optional[float] = (
            round(float(tot_views) / len(views_series), 1) if (has_views and tot_views is not None and len(views_series) > 0) else None
        )

        # Engagement rates with explicit denominator integrity (None when numerator or denominator unavailable;
        # genuine 0.0 preserved when measured denominator == 0 or measured interactions == 0).
        has_any_interaction = has_likes or has_comments or has_shares or has_saves

        pooled_views_eng_rate: Optional[float]
        mean_post_views_eng_rate: Optional[float]
        if not has_views or not has_any_interaction:
            pooled_views_eng_rate = None
            mean_post_views_eng_rate = None
        else:
            valid_views_mask = latest_df["views"].notna() & (latest_df["views"] > 0)
            if valid_views_mask.any():
                sub_df = latest_df.loc[valid_views_mask]
                sub_has_int = any(
                    col in sub_df.columns and bool(sub_df[col].notna().any())
                    for col in ("likes", "comments_count", "shares", "saves")
                )
                if sub_has_int:
                    sub_views_sum = float(sub_df["views"].sum())
                    post_interactions = (
                        (sub_df["likes"].fillna(0) if "likes" in sub_df.columns else 0) +
                        (sub_df["comments_count"].fillna(0) if "comments_count" in sub_df.columns else 0) +
                        (sub_df["shares"].fillna(0) if "shares" in sub_df.columns else 0) +
                        (sub_df["saves"].fillna(0) if "saves" in sub_df.columns else 0)
                    )
                    pooled_views_eng_rate = round(float(post_interactions.sum()) / sub_views_sum * 100.0, 2)
                    post_views_rates = (post_interactions / sub_df["views"].astype(float)) * 100.0
                    mean_post_views_eng_rate = round(float(post_views_rates.mean()), 2)
                else:
                    pooled_views_eng_rate = None
                    mean_post_views_eng_rate = None
            else:
                # Measured views are all genuine 0 across observed rows -> 0.0
                pooled_views_eng_rate = 0.0
                mean_post_views_eng_rate = 0.0

        # 3. Reach-based rates (Instagram only)
        pooled_reach_eng_rate: Optional[float]
        mean_post_reach_eng_rate: Optional[float]
        if not has_reach or not has_any_interaction:
            pooled_reach_eng_rate = None
            mean_post_reach_eng_rate = None
        else:
            valid_reach_mask = latest_df["reach"].notna() & (latest_df["reach"] > 0)
            if valid_reach_mask.any():
                ig_posts = latest_df.loc[valid_reach_mask]
                ig_has_int = any(
                    col in ig_posts.columns and bool(ig_posts[col].notna().any())
                    for col in ("likes", "comments_count", "shares", "saves")
                )
                if ig_has_int:
                    ig_reach_sum = float(ig_posts["reach"].sum())
                    post_reach_interactions = (
                        (ig_posts["likes"].fillna(0) if "likes" in ig_posts.columns else 0) +
                        (ig_posts["comments_count"].fillna(0) if "comments_count" in ig_posts.columns else 0) +
                        (ig_posts["shares"].fillna(0) if "shares" in ig_posts.columns else 0) +
                        (ig_posts["saves"].fillna(0) if "saves" in ig_posts.columns else 0)
                    )
                    pooled_reach_eng_rate = round(float(post_reach_interactions.sum()) / ig_reach_sum * 100.0, 2)
                    post_reach_rates = (post_reach_interactions / ig_posts["reach"].astype(float)) * 100.0
                    mean_post_reach_eng_rate = round(float(post_reach_rates.mean()), 2)
                else:
                    pooled_reach_eng_rate = None
                    mean_post_reach_eng_rate = None
            else:
                # Reach genuinely measured as 0 across observed rows -> 0.0
                pooled_reach_eng_rate = 0.0
                mean_post_reach_eng_rate = 0.0

        # Platforms represented
        platforms = sorted(list(latest_df["platform"].dropna().astype(str).str.title().unique())) if "platform" in latest_df.columns else []

        # Reporting period dates & missing date tracking
        ts_col = "timestamp_dt" if "timestamp_dt" in latest_df.columns else "timestamp"
        ts_valid = (
            pd.to_datetime(latest_df[ts_col], utc=True, errors="coerce").dropna()
            if ts_col in latest_df.columns
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        if not ts_valid.empty:
            min_date = ts_valid.min().strftime("%Y-%m-%d")
            max_date = ts_valid.max().strftime("%Y-%m-%d")
            period_str = f"{min_date} to {max_date}"
            period_days = max(1, (ts_valid.max().date() - ts_valid.min().date()).days + 1)
            active_posting_days = int(ts_valid.dt.date.nunique())
            missing_posting_dates = period_days - active_posting_days
        else:
            period_str = "Unknown"
            period_days = 0
            active_posting_days = 0
            missing_posting_dates = 0

        # Period-over-period comparison with strict equal-duration windows and sample sufficiency
        has_comparison = False
        comp_deltas: Dict[str, Optional[float]] = {}
        comp_note = ""
        period_details: Dict[str, Any] = {}

        MIN_POSTS_PER_PERIOD = 5
        MIN_DAYS_PER_PERIOD = 14

        prev_half: Optional[pd.DataFrame] = None
        curr_half: Optional[pd.DataFrame] = None

        if previous_df is not None:
            # Case A: Explicit previous_df provided
            prev_work_df = get_latest_posts_view(previous_df) if "post_id" in previous_df.columns else previous_df.copy()
            prev_ts = pd.to_datetime(prev_work_df[ts_col], utc=True, errors="coerce").dropna() if ts_col in prev_work_df.columns else pd.Series(dtype="datetime64[ns, UTC]")

            if prev_ts.empty or ts_valid.empty:
                comp_note = "Missing timestamp data in current or comparison period."
            else:
                prev_min, prev_max = prev_ts.min(), prev_ts.max()
                prev_duration = (prev_max - prev_min).total_seconds()
                curr_duration = (ts_valid.max() - ts_valid.min()).total_seconds()

                # Verify equal duration within 1-day tolerance (86400s)
                if abs(prev_duration - curr_duration) > 86400:
                    comp_note = (
                        f"Incomplete period mismatch: current period duration ({period_days} days) "
                        f"does not match comparison period duration ({max(1, (prev_max.date() - prev_min.date()).days + 1)} days). "
                        "Incomplete periods are not presented as equivalent."
                    )
                elif len(prev_work_df) < MIN_POSTS_PER_PERIOD or n_posts < MIN_POSTS_PER_PERIOD:
                    comp_note = (
                        f"Insufficient post sample size for statistical reliability "
                        f"(requires >= {MIN_POSTS_PER_PERIOD} posts per period; found N_curr={n_posts}, N_prev={len(prev_work_df)})."
                    )
                else:
                    prev_half = prev_work_df
                    curr_half = latest_df
                    has_comparison = True
        else:
            # Case B: Internal equal-duration split into two halves
            if period_days >= MIN_DAYS_PER_PERIOD and n_posts >= (MIN_POSTS_PER_PERIOD * 2) and not ts_valid.empty:
                midpoint = ts_valid.min() + (ts_valid.max() - ts_valid.min()) / 2
                prev_half = latest_df[ts_valid < midpoint]
                curr_half = latest_df[ts_valid >= midpoint]

                if len(prev_half) < MIN_POSTS_PER_PERIOD or len(curr_half) < MIN_POSTS_PER_PERIOD:
                    comp_note = (
                        f"Insufficient post volume in one or both equal halves "
                        f"(found N_prev={len(prev_half)}, N_curr={len(curr_half)}; requires >= {MIN_POSTS_PER_PERIOD})."
                    )
                else:
                    has_comparison = True
            elif period_days < MIN_DAYS_PER_PERIOD:
                comp_note = f"Period shorter than {MIN_DAYS_PER_PERIOD} days; historical period comparison omitted."
            else:
                comp_note = f"Fewer than {MIN_POSTS_PER_PERIOD * 2} posts; historical period comparison omitted."

        if has_comparison and prev_half is not None and curr_half is not None:
            prev_has_views = "views" in prev_half.columns and bool(prev_half["views"].notna().any())
            curr_has_views = "views" in curr_half.columns and bool(curr_half["views"].notna().any())
            if not (prev_has_views and curr_has_views):
                prev_views: Optional[int] = int(prev_half["views"].dropna().sum()) if prev_has_views else None
                curr_views: Optional[int] = int(curr_half["views"].dropna().sum()) if curr_has_views else None
                comp_deltas["views_delta_pct"] = None
            else:
                prev_views = int(prev_half["views"].dropna().sum())
                curr_views = int(curr_half["views"].dropna().sum())
                # Safe zero-activity handling (only when views were actually observed)
                if prev_views == 0:
                    comp_deltas["views_delta_pct"] = 0.0 if curr_views == 0 else None
                    comp_note = "Zero activity in previous comparison period (views = 0); percentage delta omitted to avoid distortion."
                else:
                    comp_deltas["views_delta_pct"] = round(((curr_views - prev_views) / prev_views * 100.0), 1)

            int_cols = ("likes", "comments_count", "shares", "saves")
            prev_has_int = any(c in prev_half.columns and bool(prev_half[c].notna().any()) for c in int_cols)
            curr_has_int = any(c in curr_half.columns and bool(curr_half[c].notna().any()) for c in int_cols)
            if not (prev_has_int and curr_has_int):
                prev_eng: Optional[int] = None
                curr_eng: Optional[int] = None
                comp_deltas["eng_delta_pct"] = None
            else:
                prev_eng = int(
                    (prev_half["likes"].dropna().sum() if "likes" in prev_half.columns else 0) +
                    (prev_half["comments_count"].dropna().sum() if "comments_count" in prev_half.columns else 0) +
                    (prev_half["shares"].dropna().sum() if "shares" in prev_half.columns else 0) +
                    (prev_half["saves"].dropna().sum() if "saves" in prev_half.columns else 0)
                )
                curr_eng = int(
                    (curr_half["likes"].dropna().sum() if "likes" in curr_half.columns else 0) +
                    (curr_half["comments_count"].dropna().sum() if "comments_count" in curr_half.columns else 0) +
                    (curr_half["shares"].dropna().sum() if "shares" in curr_half.columns else 0) +
                    (curr_half["saves"].dropna().sum() if "saves" in curr_half.columns else 0)
                )
                if prev_eng == 0:
                    comp_deltas["eng_delta_pct"] = 0.0 if curr_eng == 0 else None
                else:
                    comp_deltas["eng_delta_pct"] = round(((curr_eng - prev_eng) / prev_eng * 100.0), 1)

            prev_posts_cnt = len(prev_half)
            curr_posts_cnt = len(curr_half)
            comp_deltas["posts_delta_pct"] = round(((curr_posts_cnt - prev_posts_cnt) / prev_posts_cnt * 100.0), 1) if prev_posts_cnt > 0 else None

            if not comp_note:
                comp_note = f"Compared against equal-duration prior period ({len(prev_half)} vs {len(curr_half)} posts, equal time span)."

            period_details = {
                "equal_duration_verified": True,
                "prev_posts_count": int(prev_posts_cnt),
                "curr_posts_count": int(curr_posts_cnt),
                "prev_views": prev_views,
                "curr_views": curr_views,
                "prev_engagement": prev_eng,
                "curr_engagement": curr_eng,
                "active_posting_days": int(active_posting_days),
                "missing_posting_dates": int(missing_posting_dates)
            }

        return {
            "total_posts": n_posts,
            "total_views": tot_views,
            "total_impressions": tot_impressions,
            "total_likes": tot_likes,
            "total_comments": tot_comments,
            "total_shares": tot_shares,
            "total_saves": tot_saves,
            "total_new_followers": tot_new_followers,
            "avg_views_per_post": avg_views_per_post,
            "avg_reach_per_post": avg_reach_per_post,
            "overall_views_engagement_rate": pooled_views_eng_rate,
            "pooled_views_engagement_rate": pooled_views_eng_rate,
            "mean_post_level_views_engagement_rate": mean_post_views_eng_rate,
            "overall_reach_engagement_rate": pooled_reach_eng_rate,
            "pooled_reach_engagement_rate": pooled_reach_eng_rate,
            "mean_post_level_reach_engagement_rate": mean_post_reach_eng_rate,
            "rate_definitions": {
                "pooled_views_engagement_rate": "Volume-weighted: Total interactions divided by total views.",
                "mean_post_level_views_engagement_rate": "Arithmetic Mean: Average of individual post-level views engagement rates.",
                "pooled_reach_engagement_rate": "Volume-weighted: Total Instagram interactions divided by total Instagram reach.",
                "mean_post_level_reach_engagement_rate": "Arithmetic Mean: Average of individual Instagram post reach rates."
            },
            "platform_metric_availability": {
                "Instagram": {"saves_tracked": True, "reach_tracked": True, "denominators": ["views", "reach"]},
                "YouTube": {"saves_tracked": False, "reach_tracked": False, "denominators": ["views"]},
                "TikTok": {"saves_tracked": False, "collect_count_proxy_tracked": True, "reach_tracked": False, "denominators": ["views"]},
            },
            "metric_availability": {
                "views": has_views,
                "impressions": has_impressions,
                "likes": has_likes,
                "comments_count": has_comments,
                "shares": has_shares,
                "saves": has_saves,
                "collect_count": has_collects,
                "new_followers": has_new_followers,
                "reach": has_reach,
            },
            "total_collect_count_proxy": tot_collect_count_proxy,
            "dataset_id": getattr(self, "dataset_id", "synthetic_demo"),
            "provenances": sorted(latest_df["provenance"].dropna().astype(str).unique().tolist()) if "provenance" in latest_df.columns else ["synthetic_data"],
            "connected_platforms": platforms,
            "platform_count": len(platforms),
            "reporting_period": period_str,
            "period_days": period_days,
            "active_posting_days": active_posting_days,
            "missing_posting_dates": missing_posting_dates,
            "has_previous_period_comparison": has_comparison,
            "comparison_deltas": comp_deltas,
            "comparison_note": comp_note,
            "period_details": period_details
        }

    # --- 3. SECTION B: PERFORMANCE TRENDS ---

    def calculate_time_series_trends(
        self,
        df: pd.DataFrame,
        freq: str = "W"
    ) -> pd.DataFrame:
        """Aggregate performance trends over time without falsely zero-filling continuous rates."""
        if df is None or len(df) == 0:
            return pd.DataFrame()

        work_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        ts_col = "timestamp_dt" if "timestamp_dt" in work_df.columns else "timestamp"
        if ts_col not in work_df.columns:
            return pd.DataFrame()
        work_df["ts"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce")
        work_df = work_df.dropna(subset=["ts"]).sort_values("ts")

        if len(work_df) == 0:
            return pd.DataFrame()

        # Group by frequency ('D' for daily, 'W' for weekly, 'ME' for monthly)
        valid_freqs = {"daily": "D", "weekly": "W", "monthly": "ME", "D": "D", "W": "W", "ME": "ME"}
        chosen_freq = valid_freqs.get(freq.lower() if isinstance(freq, str) else "W", "W")

        # Period grouping
        grouped = work_df.set_index("ts").resample(chosen_freq)

        # Aggregate sums using native Cython groupbys (preserve np.nan for globally absent columns)
        has_col = {
            "views": "views" in work_df.columns and bool(work_df["views"].notna().any()),
            "impressions": "impressions" in work_df.columns and bool(work_df["impressions"].notna().any()),
            "likes": "likes" in work_df.columns and bool(work_df["likes"].notna().any()),
            "comments": "comments_count" in work_df.columns and bool(work_df["comments_count"].notna().any()),
            "shares": "shares" in work_df.columns and bool(work_df["shares"].notna().any()),
            "saves": "saves" in work_df.columns and bool(work_df["saves"].notna().any()),
            "new_followers": "new_followers" in work_df.columns and bool(work_df["new_followers"].notna().any()),
        }
        count_col = "post_id" if "post_id" in work_df.columns else ("views" if has_col["views"] else work_df.columns[0])
        agg_spec: Dict[str, Tuple[str, str]] = {
            "post_count": (count_col, "count"),
        }
        src_col_map = {
            "views": "views",
            "impressions": "impressions",
            "likes": "likes",
            "comments": "comments_count",
            "shares": "shares",
            "saves": "saves",
            "new_followers": "new_followers",
        }
        for dst_col, src_col in src_col_map.items():
            if has_col[dst_col]:
                agg_spec[dst_col] = (src_col, "sum")

        agg_df = grouped.agg(**agg_spec).reset_index()
        for col in ("views", "impressions", "likes", "comments", "shares", "saves", "new_followers"):
            if col not in agg_df.columns:
                agg_df[col] = np.nan

        has_any_int = any(has_col[k] for k in ("likes", "comments", "shares", "saves"))
        if has_any_int:
            agg_df["total_engagement"] = (
                agg_df["likes"].fillna(0.0)
                + agg_df["comments"].fillna(0.0)
                + agg_df["shares"].fillna(0.0)
                + agg_df["saves"].fillna(0.0)
            )
        else:
            agg_df["total_engagement"] = np.nan

        agg_views = agg_df["views"].fillna(0.0).to_numpy(dtype="float64") if has_col["views"] else np.zeros(len(agg_df), dtype="float64")
        valid_agg_views = has_col["views"] & (agg_views > 0)
        agg_shares = agg_df["shares"].fillna(0.0).to_numpy(dtype="float64") if has_col["shares"] else np.zeros(len(agg_df), dtype="float64")
        valid_agg_shares = has_col["shares"] & (agg_shares > 0)
        agg_posts = agg_df["post_count"].fillna(0.0).to_numpy(dtype="float64")
        valid_agg_posts = agg_posts > 0

        agg_tot_eng = agg_df["total_engagement"].fillna(0.0).to_numpy(dtype="float64") if has_any_int else np.zeros(len(agg_df), dtype="float64")
        agg_saves = agg_df["saves"].fillna(0.0).to_numpy(dtype="float64") if has_col["saves"] else np.zeros(len(agg_df), dtype="float64")

        valid_views_eng = valid_agg_views & has_any_int
        valid_share_rate = valid_agg_views & has_col["shares"]
        valid_save_rate = valid_agg_views & has_col["saves"]
        valid_save_share_ratio = valid_agg_shares & has_col["saves"]
        valid_avg_views = valid_agg_posts & has_col["views"]

        agg_df["views_engagement_rate"] = np.where(
            valid_views_eng,
            (agg_tot_eng / np.where(valid_views_eng, agg_views, 1.0) * 100.0).round(2),
            np.nan,
        )
        agg_df["share_rate"] = np.where(
            valid_share_rate,
            (agg_shares / np.where(valid_share_rate, agg_views, 1.0) * 100.0).round(3),
            np.nan,
        )
        agg_df["save_rate"] = np.where(
            valid_save_rate,
            (agg_saves / np.where(valid_save_rate, agg_views, 1.0) * 100.0).round(3),
            np.nan,
        )
        agg_df["save_to_share_ratio"] = np.where(
            valid_save_share_ratio,
            (agg_saves / np.where(valid_save_share_ratio, agg_shares, 1.0)).round(2),
            np.nan,
        )
        agg_df["avg_views_per_post"] = np.where(
            valid_avg_views,
            (agg_views / np.where(valid_avg_views, agg_posts, 1.0)).round(1),
            np.nan,
        )

        # Format period label
        agg_df["period_label"] = agg_df["ts"].dt.strftime("%Y-%m-%d")

        return agg_df

    def calculate_platform_time_series(self, df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
        """Calculate time-series trends segmented by platform."""
        if df is None or len(df) == 0 or "platform" not in df.columns:
            return pd.DataFrame()

        work_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        ts_col = "timestamp_dt" if "timestamp_dt" in work_df.columns else "timestamp"
        if ts_col not in work_df.columns:
            return pd.DataFrame()
        work_df["ts"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce")
        work_df = work_df.dropna(subset=["ts"]).sort_values("ts")

        valid_freqs = {"daily": "D", "weekly": "W", "monthly": "ME", "D": "D", "W": "W", "ME": "ME"}
        chosen_freq = valid_freqs.get(freq.lower() if isinstance(freq, str) else "W", "W")

        # Group by platform and resample cleanly without Python lambdas
        rows = []
        for plat, p_df in work_df.groupby("platform"):
            p_has = {
                c: (c in p_df.columns and bool(p_df[c].notna().any()))
                for c in ("views", "shares", "likes", "saves")
            }
            p_count_col = "post_id" if "post_id" in p_df.columns else ("views" if p_has["views"] else p_df.columns[0])
            p_agg_spec: Dict[str, Tuple[str, str]] = {
                "post_count": (p_count_col, "count"),
            }
            for c in ("views", "shares", "likes", "saves"):
                if p_has[c]:
                    p_agg_spec[c] = (c, "sum")
            p_grouped = p_df.set_index("ts").resample(chosen_freq).agg(**p_agg_spec).reset_index()
            for c in ("views", "shares", "likes", "saves"):
                if c not in p_grouped.columns:
                    p_grouped[c] = np.nan
            p_grouped["platform"] = str(plat).title()
            p_grouped["period_label"] = p_grouped["ts"].dt.strftime("%Y-%m-%d")

            p_views = p_grouped["views"].fillna(0.0).to_numpy(dtype="float64") if p_has["views"] else np.zeros(len(p_grouped), dtype="float64")
            p_has_any_int = any(p_has[k] for k in ("likes", "shares", "saves"))
            valid_p_eng = p_has["views"] & p_has_any_int & (p_views > 0)
            p_eng = (
                p_grouped["likes"].fillna(0.0)
                + p_grouped["shares"].fillna(0.0)
                + p_grouped["saves"].fillna(0.0)
            ).to_numpy(dtype="float64")
            p_grouped["engagement_rate"] = np.where(
                valid_p_eng,
                (p_eng / np.where(valid_p_eng, p_views, 1.0) * 100.0).round(2),
                np.nan,
            )
            rows.append(p_grouped)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # --- 4. SECTION C: PLATFORM COMPARISON ---

    def calculate_platform_comparisons(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute normalized, fair cross-platform comparison with metric definition disclosure."""
        if df is None or len(df) == 0 or "platform" not in df.columns:
            return {"summary_table": pd.DataFrame(), "format_distribution": pd.DataFrame(), "disclosures": []}

        work_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()

        summary_rows = []
        for plat, p_df in work_df.groupby("platform"):
            p_name = str(plat).strip().lower()
            n_posts = len(p_df)
            has_views = "views" in p_df.columns and bool(p_df["views"].notna().any())
            has_impressions = "impressions" in p_df.columns and bool(p_df["impressions"].notna().any())
            has_reach = "reach" in p_df.columns and bool(p_df["reach"].notna().any())
            has_likes = "likes" in p_df.columns and bool(p_df["likes"].notna().any())
            has_comments = "comments_count" in p_df.columns and bool(p_df["comments_count"].notna().any())
            has_shares = "shares" in p_df.columns and bool(p_df["shares"].notna().any())
            has_saves = "saves" in p_df.columns and bool(p_df["saves"].notna().any())
            has_collect = "collect_count" in p_df.columns and bool(p_df["collect_count"].notna().any())

            tot_views = int(p_df["views"].dropna().sum()) if has_views else np.nan
            avg_views = round(float(tot_views) / n_posts, 1) if (has_views and n_posts > 0) else np.nan

            tot_likes = int(p_df["likes"].dropna().sum()) if has_likes else 0
            tot_comments = int(p_df["comments_count"].dropna().sum()) if has_comments else 0
            tot_shares = int(p_df["shares"].dropna().sum()) if has_shares else 0
            tot_saves = int(p_df["saves"].dropna().sum()) if has_saves else 0

            valid_views_denom = bool(has_views and tot_views > 0)
            has_comparable_int = has_likes or has_comments or has_shares
            comparable_interactions = tot_likes + tot_comments + tot_shares

            norm_eng_rate = (
                round(comparable_interactions / tot_views * 100.0, 2)
                if (valid_views_denom and has_comparable_int)
                else np.nan
            )

            full_interactions = comparable_interactions + (tot_saves if (p_name == "instagram" and has_saves) else 0)
            plat_eng_rate = (
                round(full_interactions / tot_views * 100.0, 2)
                if (valid_views_denom and (has_comparable_int or (p_name == "instagram" and has_saves)))
                else np.nan
            )

            share_rate = (
                round(tot_shares / tot_views * 100.0, 3)
                if (valid_views_denom and has_shares)
                else np.nan
            )
            save_rate = (
                round(tot_saves / tot_views * 100.0, 3)
                if (valid_views_denom and p_name == "instagram" and has_saves)
                else np.nan
            )

            ret_nonnull = (
                pd.to_numeric(p_df["retention_rate"], errors="coerce").dropna()
                if "retention_rate" in p_df.columns
                else pd.Series(dtype=float)
            )
            if len(ret_nonnull) > 0:
                ret_mean = float(ret_nonnull.mean())
                if float(ret_nonnull.max()) <= 1.0:
                    ret_mean *= 100.0
                avg_retention = round(ret_mean, 1)
            else:
                avg_retention = np.nan

            obs_fields: List[str] = []
            if has_views:
                obs_fields.append("Views")
            if has_impressions:
                obs_fields.append("Impressions")
            if has_reach:
                obs_fields.append("Reach")
            if has_likes:
                obs_fields.append("Likes")
            if has_comments:
                obs_fields.append("Comments")
            if has_shares:
                obs_fields.append("Shares")
            if has_saves:
                obs_fields.append("Saves")
            if has_collect:
                obs_fields.append("+ collect_count bookmark proxy")
            avail_fields = ", ".join(obs_fields) if obs_fields else "None Observed"

            summary_rows.append({
                "Platform": str(plat).title(),
                "Posts": n_posts,
                "Total Views": tot_views,
                "Avg Views/Post": avg_views,
                "Normalized Eng Rate (%)": norm_eng_rate,
                "Full Eng Rate (%)": plat_eng_rate,
                "Share Rate (%)": share_rate,
                "Save Rate (%)": save_rate,
                "Avg Retention (%)": avg_retention,
                "Available Fields": avail_fields,
            })

        summary_table = pd.DataFrame(summary_rows)

        # Format distribution by platform
        if "content_type" in work_df.columns:
            fmt_dist = work_df.groupby(["platform", "content_type"]).size().reset_index(name="post_count")
            fmt_dist["platform"] = fmt_dist["platform"].astype(str).str.title()
        else:
            fmt_dist = pd.DataFrame()

        disclosures = [
            "PLATFORM METRIC NON-EQUIVALENCE: Aligning metrics into a common mathematical formula does NOT imply that engagement actions are universally equivalent across platforms. User intent, distribution algorithms, and session contexts differ significantly across Instagram, YouTube, and TikTok.",
            "AVAILABLE INTERACTION FIELDS & DENOMINATORS BY PLATFORM:",
            "  • Instagram: Tracks Views, Impressions, Reach, Likes, Comments, Shares, and Saves (Bookmarks). Denominators can be Views (total playback counts) or Reach (unique accounts exposed).",
            "  • YouTube: Tracks Views, Impressions, Likes, Comments, and Shares. Neither post-level Reach (unique viewers) nor Saves/Bookmarks are exposed in the standard YouTube Data API v3 schema. Denominator is restricted to Views.",
            "  • TikTok: Tracks Views (`play_count`), Likes, Comments, Shares, and explicit proxy `collect_count` (never pooled directly into Instagram `saves`); retention and reach are unavailable in public panel schemas.",
            "NORMALIZED FORMULA DEFINITION: Normalized Engagement Rate = (Likes + Comments + Shares) / Views × 100%. Excludes saves to restrict comparison to commonly tracked interaction fields, but should not be interpreted as proving identical behavioral value.",
            "FULL PLATFORM FORMULA DEFINITION: Instagram Full Engagement Rate = (Likes + Comments + Shares + Saves) / Views × 100%. YouTube & TikTok Full Engagement Rate = (Likes + Comments + Shares) / Views × 100%."
        ]

        return {
            "summary_table": summary_table,
            "format_distribution": fmt_dist,
            "disclosures": disclosures
        }

    # --- 5. SECTION D: CONTENT PERFORMANCE BREAKDOWNS ---

    def calculate_content_breakdowns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Aggregate performance across content formats, topics, duration bins, and hooks."""
        if df is None or len(df) == 0:
            return {
                "by_format": pd.DataFrame(),
                "by_topic": pd.DataFrame(),
                "by_duration": pd.DataFrame(),
                "by_hook": pd.DataFrame(),
                "timing_heatmap": pd.DataFrame()
            }

        work_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        ret_series_global = (
            pd.to_numeric(work_df["retention_rate"], errors="coerce").dropna()
            if "retention_rate" in work_df.columns
            else pd.Series(dtype=float)
        )
        ret_scale = 100.0 if (not ret_series_global.empty and float(ret_series_global.max()) <= 1.0) else 1.0

        count_col = (
            "post_id"
            if "post_id" in work_df.columns
            else ("views" if "views" in work_df.columns else work_df.columns[0])
        )

        # 1. By Content Format
        if "content_type" in work_df.columns and work_df["content_type"].notna().any():
            fmt_agg: Dict[str, Tuple[str, str]] = {
                "post_count": (count_col, "count"),
            }
            if "views" in work_df.columns:
                fmt_agg["avg_views"] = ("views", "mean")
            if "likes" in work_df.columns:
                fmt_agg["avg_likes"] = ("likes", "mean")
            if "shares" in work_df.columns:
                fmt_agg["avg_shares"] = ("shares", "mean")
            if "retention_rate" in work_df.columns:
                fmt_agg["avg_retention"] = ("retention_rate", "mean")
            if "comments_count" in work_df.columns:
                fmt_agg["avg_comments"] = ("comments_count", "mean")
            if "saves" in work_df.columns:
                fmt_agg["avg_saves"] = ("saves", "mean")
            by_format = work_df.groupby("content_type").agg(**fmt_agg).reset_index()
            for col in ("avg_views", "avg_likes", "avg_shares", "avg_retention", "avg_comments", "avg_saves"):
                if col not in by_format.columns:
                    by_format[col] = np.nan
                else:
                    by_format[col] = pd.to_numeric(by_format[col], errors="coerce").astype("float64")
            if ret_scale != 1.0:
                by_format["avg_retention"] = by_format["avg_retention"] * ret_scale

            views_arr = by_format["avg_views"].to_numpy(dtype="float64")
            likes_s = by_format["avg_likes"]
            comm_s = by_format["avg_comments"]
            shares_s = by_format["avg_shares"]
            saves_s = by_format["avg_saves"]

            valid_views = np.isfinite(views_arr) & (views_arr > 0)
            has_any_int = (likes_s.notna() | comm_s.notna() | shares_s.notna() | saves_s.notna()).to_numpy(dtype=bool)
            has_any_high_int = (shares_s.notna() | saves_s.notna()).to_numpy(dtype=bool)

            tot_int = (
                likes_s.fillna(0.0) +
                comm_s.fillna(0.0) +
                shares_s.fillna(0.0) +
                saves_s.fillna(0.0)
            ).to_numpy(dtype="float64")
            high_int = (
                shares_s.fillna(0.0) +
                saves_s.fillna(0.0)
            ).to_numpy(dtype="float64")

            valid_views_eng = valid_views & has_any_int
            valid_high_int = valid_views & has_any_high_int
            by_format["views_engagement_rate"] = np.where(
                valid_views_eng,
                np.round(tot_int / np.where(valid_views_eng, views_arr, 1.0) * 100.0, 2),
                np.nan,
            )
            by_format["high_intent_rate"] = np.where(
                valid_high_int,
                np.round(high_int / np.where(valid_high_int, views_arr, 1.0) * 100.0, 2),
                np.nan,
            )
        else:
            by_format = pd.DataFrame()

        # 2. By Struggle Topic
        if "topic" in work_df.columns and work_df["topic"].notna().any():
            top_agg: Dict[str, Tuple[str, str]] = {
                "post_count": (count_col, "count"),
            }
            if "views" in work_df.columns:
                top_agg["avg_views"] = ("views", "mean")
            if "likes" in work_df.columns:
                top_agg["avg_likes"] = ("likes", "mean")
            if "shares" in work_df.columns:
                top_agg["avg_shares"] = ("shares", "mean")
            if "retention_rate" in work_df.columns:
                top_agg["avg_retention"] = ("retention_rate", "mean")
            if "saves" in work_df.columns:
                top_agg["avg_saves"] = ("saves", "mean")
            if "new_followers" in work_df.columns:
                top_agg["new_followers"] = ("new_followers", "sum")
            by_topic = work_df.groupby("topic").agg(**top_agg).reset_index()
            for col in ("avg_views", "avg_likes", "avg_shares", "avg_retention", "avg_saves"):
                if col not in by_topic.columns:
                    by_topic[col] = np.nan
                else:
                    by_topic[col] = pd.to_numeric(by_topic[col], errors="coerce").astype("float64")
            if ret_scale != 1.0:
                by_topic["avg_retention"] = by_topic["avg_retention"] * ret_scale
            if "new_followers" not in by_topic.columns:
                by_topic["new_followers"] = np.nan
            else:
                nf_obs = work_df["new_followers"].notna().groupby(work_df["topic"]).sum()
                has_nf = by_topic["topic"].map(nf_obs).fillna(0).to_numpy() > 0
                nf_vals = pd.to_numeric(by_topic["new_followers"], errors="coerce").to_numpy(dtype="float64")
                by_topic["new_followers"] = np.where(has_nf, nf_vals, np.nan)

            shares_arr = by_topic["avg_shares"].to_numpy(dtype="float64")
            saves_arr = by_topic["avg_saves"].to_numpy(dtype="float64")
            valid_ratio = np.isfinite(shares_arr) & (shares_arr > 0) & np.isfinite(saves_arr)
            by_topic["save_to_share_ratio"] = np.where(
                valid_ratio,
                np.round(
                    np.where(valid_ratio, saves_arr, 0.0) / np.where(valid_ratio, shares_arr, 1.0),
                    2,
                ),
                np.nan,
            )
            sort_col = (
                "avg_views"
                if by_topic["avg_views"].notna().any()
                else ("avg_likes" if by_topic["avg_likes"].notna().any() else "post_count")
            )
            by_topic = by_topic.sort_values(sort_col, ascending=False, na_position="last")
        else:
            by_topic = pd.DataFrame()

        # 3. By Duration Bin
        if "duration_seconds" in work_df.columns and work_df["duration_seconds"].notna().any():
            bins = [0, 25, 50, 90, 3600]
            labels = ["Short (<25s)", "Medium (25-50s)", "Long (50-90s)", "Extended (>90s)"]
            work_df["duration_bin"] = pd.cut(pd.to_numeric(work_df["duration_seconds"], errors="coerce"), bins=bins, labels=labels)
            dur_agg: Dict[str, Tuple[str, str]] = {
                "post_count": (count_col, "count"),
            }
            if "views" in work_df.columns:
                dur_agg["avg_views"] = ("views", "mean")
            if "shares" in work_df.columns:
                dur_agg["avg_shares"] = ("shares", "mean")
            if "retention_rate" in work_df.columns:
                dur_agg["avg_retention"] = ("retention_rate", "mean")
            by_duration = work_df.groupby("duration_bin", observed=False).agg(**dur_agg).reset_index()
            for col in ("avg_views", "avg_shares", "avg_retention"):
                if col not in by_duration.columns:
                    by_duration[col] = np.nan
                else:
                    by_duration[col] = pd.to_numeric(by_duration[col], errors="coerce").astype("float64")
            if ret_scale != 1.0:
                by_duration["avg_retention"] = by_duration["avg_retention"] * ret_scale
        else:
            by_duration = pd.DataFrame()

        # 4. By Hook Type
        if "hook_type" in work_df.columns and work_df["hook_type"].notna().any():
            hook_agg: Dict[str, Tuple[str, str]] = {
                "post_count": (count_col, "count"),
            }
            if "views" in work_df.columns:
                hook_agg["avg_views"] = ("views", "mean")
            if "shares" in work_df.columns:
                hook_agg["avg_shares"] = ("shares", "mean")
            if "retention_rate" in work_df.columns:
                hook_agg["avg_retention"] = ("retention_rate", "mean")
            by_hook = work_df.groupby("hook_type").agg(**hook_agg).reset_index()
            for col in ("avg_views", "avg_shares", "avg_retention"):
                if col not in by_hook.columns:
                    by_hook[col] = np.nan
                else:
                    by_hook[col] = pd.to_numeric(by_hook[col], errors="coerce").astype("float64")
            if ret_scale != 1.0:
                by_hook["avg_retention"] = by_hook["avg_retention"] * ret_scale
            hook_views = by_hook["avg_views"].to_numpy(dtype="float64")
            hook_shares = by_hook["avg_shares"].to_numpy(dtype="float64")
            valid_hrate = np.isfinite(hook_views) & (hook_views > 0) & np.isfinite(hook_shares)
            by_hook["share_rate"] = np.where(
                valid_hrate,
                np.round(
                    np.where(valid_hrate, hook_shares, 0.0) / np.where(valid_hrate, hook_views, 1.0) * 100.0,
                    2,
                ),
                np.nan,
            )
        else:
            by_hook = pd.DataFrame()

        # 5. Timing Matrix (Posting Hour vs Day of Week)
        ts_col = "timestamp_dt" if "timestamp_dt" in work_df.columns else "timestamp"
        if (
            ts_col in work_df.columns
            and "posting_hour" in work_df.columns
            and work_df["posting_hour"].notna().any()
            and "views" in work_df.columns
            and work_df["views"].notna().any()
        ):
            work_df["day_name"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce").dt.day_name()
            timing_matrix = work_df.pivot_table(
                index="posting_hour",
                columns="day_name",
                values="views",
                aggfunc=["mean", "count"]
            )
        else:
            timing_matrix = pd.DataFrame()

        return {
            "by_format": by_format,
            "by_topic": by_topic,
            "by_duration": by_duration,
            "by_hook": by_hook,
            "timing_matrix": timing_matrix
        }

    # --- 6. SECTION E: AUDIENCE INSIGHTS (MODULE 3 SYNTHESIS) ---

    def calculate_audience_insights(self) -> Dict[str, Any]:
        """Aggregate audience sentiment, relatability, and empirical post correlation."""
        if self.comments_df is None or len(self.comments_df) == 0:
            return {
                "has_data": False,
                "total_comments": 0,
                "sentiment_dist": pd.DataFrame(),
                "relatability_dist": pd.DataFrame(),
                "topic_relatability": pd.DataFrame(),
                "matched_correlation_posts": 0,
                "correlation_r": None,
                "correlation_rho": None,
                "correlation_p": None,
                "correlation_text": "No comment data available."
            }

        if "sentiment_label" not in self.comments_df.columns or "relatability_label" not in self.comments_df.columns:
            return {
                "has_data": False,
                "total_comments": int(len(self.comments_df)),
                "sentiment_dist": pd.DataFrame(),
                "relatability_dist": pd.DataFrame(),
                "topic_relatability": pd.DataFrame(),
                "matched_correlation_posts": 0,
                "correlation_r": None,
                "correlation_rho": None,
                "correlation_p": None,
                "correlation_text": "Comments present without precomputed NLP sentiment/relatability labels."
            }

        c_df = self.comments_df.copy()
        tot_comments = len(c_df)

        # 1. Sentiment Distribution
        sent_counts = c_df["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        sent_counts["percentage"] = (sent_counts["count"] / tot_comments * 100.0).round(1)

        # 2. Relatability Distribution
        rel_counts = c_df["relatability_label"].value_counts().reset_index()
        rel_counts.columns = ["relatability", "count"]
        rel_counts["percentage"] = (rel_counts["count"] / tot_comments * 100.0).round(1)

        # 3. Post-level merge for correlation with saves/engagement
        topic_rel = pd.DataFrame()
        r_val: Optional[float] = None
        p_val: Optional[float] = None
        matched_corr_n = 0

        if "post_id" in c_df.columns and self.posts_df is not None and len(self.posts_df) > 0:
            latest_posts = get_latest_posts_view(self.posts_df) if "post_id" in self.posts_df.columns else self.posts_df.copy()
            valid_c = c_df.dropna(subset=["post_id"]).copy()
            rel_by_post = (
                valid_c.assign(_is_rel=(valid_c["relatability_label"] == "Relatable"))
                .groupby("post_id", as_index=False)["_is_rel"]
                .mean()
                .rename(columns={"_is_rel": "post_relatability_rate"})
            )

            merged = pd.merge(latest_posts, rel_by_post, on="post_id", how="inner")
            if len(merged) > 0 and "topic" in merged.columns:
                topic_rel = merged.groupby("topic")["post_relatability_rate"].mean().reset_index()
                topic_rel["post_relatability_rate"] = (topic_rel["post_relatability_rate"] * 100.0).round(1)
                topic_rel = topic_rel.sort_values("post_relatability_rate", ascending=False)

            # Empirical Spearman correlation on deduplicated latest post records
            if "saves" in merged.columns:
                clean_corr = merged.dropna(subset=["post_relatability_rate", "saves"])
                matched_corr_n = len(clean_corr)
                if matched_corr_n >= 10:
                    sp_r, sp_p = stats.spearmanr(clean_corr["post_relatability_rate"], clean_corr["saves"])
                    if np.isfinite(sp_r) and np.isfinite(sp_p):
                        r_val = float(sp_r)
                        p_val = float(sp_p)

        is_synth = self.dataset_id == "synthetic_demo"
        if r_val is not None and p_val is not None:
            r_round = round(float(r_val), 4)
            p_round = round(float(p_val), 3)
            if is_synth:
                corr_text = (
                    f"Spearman rank correlation: rho = {r_round:.4f} (p = {p_round:.3f}). "
                    "Exploratory analysis found no statistically significant monotonic association between post-level relatability percentage and bookmark saves in this synthetic dataset. "
                    "This exploratory result does not establish equivalence or prove the absence of an effect in production settings."
                )
            else:
                sig_desc = "a statistically significant" if p_val < 0.05 else "no statistically significant"
                corr_text = (
                    f"Spearman rank correlation across N = {matched_corr_n} matched posts: rho = {r_round:.4f} (p = {p_round:.3f}). "
                    f"Exploratory analysis found {sig_desc} monotonic association between post-level relatability percentage and bookmark saves in this corpus. "
                    "Observational correlation does not establish causal effects or prove equivalence."
                )
        else:
            r_round = None
            p_round = None
            corr_text = (
                "Spearman rank correlation with bookmark saves was omitted "
                f"(requires at least 10 matched posts with observed `saves`; matched observed pairs N = {matched_corr_n} in this corpus)."
            )

        return {
            "has_data": True,
            "total_comments": tot_comments,
            "sentiment_dist": sent_counts,
            "relatability_dist": rel_counts,
            "topic_relatability": topic_rel,
            "matched_correlation_posts": int(matched_corr_n),
            "correlation_r": r_round,
            "correlation_rho": r_round,
            "correlation_p": p_round,
            "correlation_text": corr_text
        }

    # --- 7. SECTION F: EXPERIMENT PERFORMANCE (MODULE 4 SYNTHESIS) ---

    def calculate_experiment_performance(self) -> Dict[str, Any]:
        """Synthesize Module 4 A/B experiment outcomes, forest plot data, and Holm-adjusted p-values."""
        if self.experiments_df is None or self.observations_df is None:
            return {"experiments": [], "forest_plot_data": pd.DataFrame()}

        framework = ABTestingFramework(alpha=0.05)
        res_list = []
        meta_list = []

        for exp in self.experiments_df.to_dict("records"):
            e_id = exp["experiment_id"]
            p_metric = exp["primary_metric"]
            sub = self.observations_df[self.observations_df["experiment_id"] == e_id]
            va_label = exp["variant_a_label"]
            vb_label = exp["variant_b_label"]
            va = sub[sub["variant"] == va_label]
            vb = sub[sub["variant"] == vb_label]

            if len(va) == 0 or len(vb) == 0:
                continue

            if p_metric in ["views", "relatability_pct"]:
                res = framework.compare_continuous(
                    va[p_metric].dropna().values,
                    vb[p_metric].dropna().values,
                    metric_name=p_metric,
                    variant_a_label=va_label,
                    variant_b_label=vb_label,
                    experiment_type=exp["experiment_type"]
                )
            else:
                res = framework.compare_post_level_rates(
                    df=sub,
                    variant_col="variant",
                    variant_a=va_label,
                    variant_b=vb_label,
                    rate_col=p_metric if p_metric in sub.columns else None,
                    numerator_col="shares" if "shares" in sub.columns else None,
                    denominator_col="views" if "views" in sub.columns else None,
                    metric_name=p_metric,
                    experiment_type=exp["experiment_type"]
                )
            res_list.append(res)
            meta_list.append(exp)

        # Apply Holm-Bonferroni step-down correction
        adj_results = ABTestingFramework.apply_multiple_testing_correction(res_list, method="holm")

        exp_summaries = []
        forest_rows = []

        for exp_meta, res in zip(meta_list, adj_results):
            e_id = exp_meta["experiment_id"]
            is_rct = exp_meta["experiment_type"] == "randomized_experiment"
            n_tot = res.sample_size_a + res.sample_size_b

            exp_summaries.append({
                "experiment_id": e_id,
                "name": exp_meta["experiment_name"],
                "experiment_type": exp_meta["experiment_type"],
                "is_randomized": is_rct,
                "platform": exp_meta["platform"].title(),
                "primary_metric": res.metric_name,
                "sample_size": n_tot,
                "variant_a": f"{res.variant_a_label} (N={res.sample_size_a})",
                "variant_b": f"{res.variant_b_label} (N={res.sample_size_b})",
                "value_a": res.value_a,
                "value_b": res.value_b,
                "relative_lift_pct": res.relative_lift_pct,
                "effect_size": res.effect_size,
                "effect_size_type": res.effect_size_type,
                "test_statistic": res.test_statistic,
                "test_name": res.test_name,
                "raw_p_value": res.p_value,
                "holm_adj_p_value": res.p_value_adjusted,
                "confidence_interval": res.confidence_interval,
                "ci_method": res.ci_method,
                "is_significant": res.is_statistically_significant,
                "unit_of_analysis": res.unit_of_analysis
            })

            # Forest plot data (standardized effect size and normalized error bars)
            # Bound g for visualization
            eff_disp = min(max(res.effect_size, -3.0), 3.0)
            ci_low = max(res.confidence_interval[0], -5.0)
            ci_high = min(res.confidence_interval[1], 5.0)
            forest_rows.append({
                "experiment": f"{e_id}: {exp_meta['experiment_name'][:28]}...",
                "effect_size": round(eff_disp, 3),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "is_significant": res.is_statistically_significant,
                "type": "Randomized RCT" if is_rct else "Observational Cohort"
            })

        return {
            "experiments": exp_summaries,
            "forest_plot_data": pd.DataFrame(forest_rows)
        }

    # --- 8. SECTION G: OPTIMIZATION OPPORTUNITIES (MODULE 5 SYNTHESIS) ---

    def calculate_recommendation_priorities(self) -> RecommendationReport:
        """Synthesize prescriptive recommendations from Module 5 OptimizationRecommender."""
        recommender = OptimizationRecommender(
            posts_df=self.posts_df,
            comments_df=self.comments_df,
            experiments_df=self.experiments_df,
            observations_df=self.observations_df
        )
        return recommender.generate_recommendations(max_recommendations=10)

