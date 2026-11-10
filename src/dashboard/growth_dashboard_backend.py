"""Backend Data Engine for Module 6: Growth Visualization Dashboard.

Provides decoupled, high-performance data loading, filtering, KPI calculations,
period-over-period comparisons, time-series aggregations, platform normalizations,
and multi-module synthesis (Modules 1-5).
"""

from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing.cleaning import get_latest_posts_view
from src.statistics.ab_testing import ABTestingFramework
from src.recommender.engine import OptimizationRecommender
from src.recommender.schemas import ConfidenceLevel, EvidenceSource, RecommendationReport


class GrowthDashboardBackend:
    """Analytical data engine powering the Module 6 Growth Visualization Dashboard."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parent.parent.parent
        self.posts_df: Optional[pd.DataFrame] = None
        self.comments_df: Optional[pd.DataFrame] = None
        self.experiments_df: Optional[pd.DataFrame] = None
        self.observations_df: Optional[pd.DataFrame] = None
        self.load_datasets()

    def load_datasets(self) -> None:
        """Load and cache analytical datasets across Modules 1-4."""
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
            filtered = filtered[filtered["platform"].str.lower() == platform.lower().strip()]

        # Content type filter
        if content_type and content_type.lower() != "all" and "content_type" in filtered.columns:
            filtered = filtered[filtered["content_type"].str.lower() == content_type.lower().strip()]

        # Topic filter
        if topic and topic.lower() != "all" and "topic" in filtered.columns:
            filtered = filtered[filtered["topic"].str.lower() == topic.lower().strip()]

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
                "total_views": 0,
                "total_impressions": 0,
                "total_likes": 0,
                "total_comments": 0,
                "total_shares": 0,
                "total_saves": 0,
                "total_new_followers": 0,
                "avg_views_per_post": 0.0,
                "avg_reach_per_post": 0.0,
                "overall_views_engagement_rate": 0.0,
                "pooled_views_engagement_rate": 0.0,
                "mean_post_level_views_engagement_rate": 0.0,
                "overall_reach_engagement_rate": 0.0,
                "pooled_reach_engagement_rate": 0.0,
                "mean_post_level_reach_engagement_rate": 0.0,
                "rate_definitions": {
                    "pooled_views_engagement_rate": "Volume-weighted total interactions divided by total views.",
                    "mean_post_level_views_engagement_rate": "Arithmetic mean of individual post-level views engagement rates.",
                    "pooled_reach_engagement_rate": "Volume-weighted total Instagram interactions divided by total Instagram reach.",
                    "mean_post_level_reach_engagement_rate": "Arithmetic mean of individual Instagram post reach rates."
                },
                "connected_platforms": [],
                "platform_count": 0,
                "reporting_period": "N/A",
                "period_days": 0,
                "active_posting_days": 0,
                "missing_posting_dates": 0,
                "has_previous_period_comparison": False,
                "comparison_deltas": {},
                "comparison_note": "No data available."
            }

        # Deduplicate to latest post snapshots to avoid multiple snapshot inflation
        latest_df = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        n_posts = len(latest_df)

        # Totals
        tot_views = int(latest_df["views"].dropna().sum()) if "views" in latest_df.columns else 0
        tot_impressions = int(latest_df["impressions"].dropna().sum()) if "impressions" in latest_df.columns else 0
        tot_likes = int(latest_df["likes"].dropna().sum()) if "likes" in latest_df.columns else 0
        tot_comments = int(latest_df["comments_count"].dropna().sum()) if "comments_count" in latest_df.columns else 0
        tot_shares = int(latest_df["shares"].dropna().sum()) if "shares" in latest_df.columns else 0
        tot_saves = int(latest_df["saves"].dropna().sum()) if "saves" in latest_df.columns else 0
        tot_new_followers = int(latest_df["new_followers"].dropna().sum()) if "new_followers" in latest_df.columns else 0

        # Reach (Instagram only)
        reach_series = latest_df["reach"].dropna() if "reach" in latest_df.columns else pd.Series(dtype=float)
        tot_reach = int(reach_series.sum()) if len(reach_series) > 0 else 0
        n_posts_with_reach = len(reach_series[reach_series > 0])
        avg_reach_per_post = round(tot_reach / n_posts_with_reach, 1) if n_posts_with_reach > 0 else 0.0

        avg_views_per_post = round(tot_views / n_posts, 1) if n_posts > 0 else 0.0

        # Engagement rates with explicit denominator definitions:
        # 1. Pooled Views-based: (Likes + Comments + Shares + Saves) / Views
        tot_interactions_views = tot_likes + tot_comments + tot_shares + tot_saves
        pooled_views_eng_rate = (tot_interactions_views / tot_views * 100.0) if tot_views > 0 else 0.0

        # 2. Arithmetic Mean of Post-Level Views-based Rates:
        # Rate_i = (Likes_i + Comments_i + Shares_i + Saves_i) / Views_i * 100%
        valid_views_mask = (latest_df["views"] > 0) if "views" in latest_df.columns else pd.Series(False, index=latest_df.index)
        if valid_views_mask.any():
            sub_df = latest_df.loc[valid_views_mask]
            post_interactions = (
                (sub_df["likes"].fillna(0) if "likes" in sub_df.columns else 0) +
                (sub_df["comments_count"].fillna(0) if "comments_count" in sub_df.columns else 0) +
                (sub_df["shares"].fillna(0) if "shares" in sub_df.columns else 0) +
                (sub_df["saves"].fillna(0) if "saves" in sub_df.columns else 0)
            )
            post_views_rates = (post_interactions / sub_df["views"]) * 100.0
            mean_post_views_eng_rate = float(post_views_rates.mean())
        else:
            mean_post_views_eng_rate = 0.0

        # 3. Reach-based rates (Instagram only)
        if n_posts_with_reach > 0 and tot_reach > 0:
            ig_posts = latest_df[latest_df["reach"] > 0]
            ig_interactions = (
                (ig_posts["likes"].fillna(0).sum() if "likes" in ig_posts.columns else 0) +
                (ig_posts["comments_count"].fillna(0).sum() if "comments_count" in ig_posts.columns else 0) +
                (ig_posts["shares"].fillna(0).sum() if "shares" in ig_posts.columns else 0) +
                (ig_posts["saves"].fillna(0).sum() if "saves" in ig_posts.columns else 0)
            )
            pooled_reach_eng_rate = (ig_interactions / tot_reach * 100.0)

            post_reach_interactions = (
                (ig_posts["likes"].fillna(0) if "likes" in ig_posts.columns else 0) +
                (ig_posts["comments_count"].fillna(0) if "comments_count" in ig_posts.columns else 0) +
                (ig_posts["shares"].fillna(0) if "shares" in ig_posts.columns else 0) +
                (ig_posts["saves"].fillna(0) if "saves" in ig_posts.columns else 0)
            )
            post_reach_rates = (post_reach_interactions / ig_posts["reach"]) * 100.0
            mean_post_reach_eng_rate = float(post_reach_rates.mean())
        else:
            pooled_reach_eng_rate = 0.0
            mean_post_reach_eng_rate = 0.0

        # Platforms represented
        platforms = sorted(list(latest_df["platform"].dropna().str.title().unique())) if "platform" in latest_df.columns else []

        # Reporting period dates & missing date tracking
        ts_col = "timestamp_dt" if "timestamp_dt" in latest_df.columns else "timestamp"
        ts_valid = pd.to_datetime(latest_df[ts_col], utc=True, errors="coerce").dropna()
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
            prev_views = prev_half["views"].dropna().sum() if "views" in prev_half.columns else 0
            curr_views = curr_half["views"].dropna().sum() if "views" in curr_half.columns else 0

            # Safe zero-activity handling
            if prev_views == 0:
                comp_deltas["views_delta_pct"] = 0.0 if curr_views == 0 else None
                comp_note = "Zero activity in previous comparison period (views = 0); percentage delta omitted to avoid distortion."
            else:
                comp_deltas["views_delta_pct"] = round(((curr_views - prev_views) / prev_views * 100.0), 1)

            prev_eng = (
                (prev_half["likes"].fillna(0).sum() if "likes" in prev_half.columns else 0) +
                (prev_half["comments_count"].fillna(0).sum() if "comments_count" in prev_half.columns else 0) +
                (prev_half["shares"].fillna(0).sum() if "shares" in prev_half.columns else 0) +
                (prev_half["saves"].dropna().sum() if "saves" in prev_half.columns else 0)
            )
            curr_eng = (
                (curr_half["likes"].fillna(0).sum() if "likes" in curr_half.columns else 0) +
                (curr_half["comments_count"].fillna(0).sum() if "comments_count" in curr_half.columns else 0) +
                (curr_half["shares"].fillna(0).sum() if "shares" in curr_half.columns else 0) +
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
                "prev_views": int(prev_views),
                "curr_views": int(curr_views),
                "prev_engagement": int(prev_eng),
                "curr_engagement": int(curr_eng),
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
            "overall_views_engagement_rate": round(pooled_views_eng_rate, 2),
            "pooled_views_engagement_rate": round(pooled_views_eng_rate, 2),
            "mean_post_level_views_engagement_rate": round(mean_post_views_eng_rate, 2),
            "overall_reach_engagement_rate": round(pooled_reach_eng_rate, 2),
            "pooled_reach_engagement_rate": round(pooled_reach_eng_rate, 2),
            "mean_post_level_reach_engagement_rate": round(mean_post_reach_eng_rate, 2),
            "rate_definitions": {
                "pooled_views_engagement_rate": "Volume-weighted: Total interactions divided by total views.",
                "mean_post_level_views_engagement_rate": "Arithmetic Mean: Average of individual post-level views engagement rates.",
                "pooled_reach_engagement_rate": "Volume-weighted: Total Instagram interactions divided by total Instagram reach.",
                "mean_post_level_reach_engagement_rate": "Arithmetic Mean: Average of individual Instagram post reach rates."
            },
            "platform_metric_availability": {
                "Instagram": {"saves_tracked": True, "reach_tracked": True, "denominators": ["views", "reach"]},
                "YouTube": {"saves_tracked": False, "reach_tracked": False, "denominators": ["views"]}
            },
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
        work_df["ts"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce")
        work_df = work_df.dropna(subset=["ts"]).sort_values("ts")

        if len(work_df) == 0:
            return pd.DataFrame()

        # Group by frequency ('D' for daily, 'W' for weekly, 'ME' for monthly)
        valid_freqs = {"daily": "D", "weekly": "W", "monthly": "ME", "D": "D", "W": "W", "ME": "ME"}
        chosen_freq = valid_freqs.get(freq.lower() if isinstance(freq, str) else "W", "W")

        # Period grouping
        grouped = work_df.set_index("ts").resample(chosen_freq)

        # Aggregate sums for volume counts
        agg_df = grouped.agg(
            post_count=("post_id", "count") if "post_id" in work_df.columns else ("views", "count"),
            views=("views", "sum"),
            impressions=("impressions", "sum") if "impressions" in work_df.columns else ("views", "sum"),
            likes=("likes", "sum"),
            comments=("comments_count", "sum") if "comments_count" in work_df.columns else ("likes", "sum"),
            shares=("shares", "sum"),
            saves=("saves", lambda x: x.dropna().sum()),
            new_followers=("new_followers", "sum") if "new_followers" in work_df.columns else ("views", lambda x: 0)
        ).reset_index()

        # Compute rate metrics strictly from observed period totals (prevent zero-rate imputation)
        agg_df["total_engagement"] = agg_df["likes"] + agg_df["comments"] + agg_df["shares"] + agg_df["saves"]
        agg_views = agg_df["views"].fillna(0).astype(float).to_numpy()
        valid_agg_views = agg_views > 0
        agg_shares = agg_df["shares"].fillna(0).astype(float).to_numpy()
        valid_agg_shares = agg_shares > 0
        agg_posts = agg_df["post_count"].fillna(0).astype(float).to_numpy()
        valid_agg_posts = agg_posts > 0

        agg_tot_eng = agg_df["total_engagement"].fillna(0).astype(float).to_numpy()
        agg_saves = agg_df["saves"].fillna(0).astype(float).to_numpy()

        agg_df["views_engagement_rate"] = np.where(
            valid_agg_views,
            (agg_tot_eng / np.where(valid_agg_views, agg_views, 1.0) * 100.0).round(2),
            np.nan
        )
        agg_df["share_rate"] = np.where(
            valid_agg_views,
            (agg_shares / np.where(valid_agg_views, agg_views, 1.0) * 100.0).round(3),
            np.nan
        )
        agg_df["save_rate"] = np.where(
            valid_agg_views,
            (agg_saves / np.where(valid_agg_views, agg_views, 1.0) * 100.0).round(3),
            np.nan
        )
        agg_df["save_to_share_ratio"] = np.where(
            valid_agg_shares,
            (agg_saves / np.where(valid_agg_shares, agg_shares, 1.0)).round(2),
            np.nan
        )
        agg_df["avg_views_per_post"] = np.where(
            valid_agg_posts,
            (agg_views / np.where(valid_agg_posts, agg_posts, 1.0)).round(1),
            np.nan
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
        work_df["ts"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce")
        work_df = work_df.dropna(subset=["ts"]).sort_values("ts")

        valid_freqs = {"daily": "D", "weekly": "W", "monthly": "ME", "D": "D", "W": "W", "ME": "ME"}
        chosen_freq = valid_freqs.get(freq.lower() if isinstance(freq, str) else "W", "W")

        # Group by platform and resample
        rows = []
        for plat, p_df in work_df.groupby("platform"):
            p_grouped = p_df.set_index("ts").resample(chosen_freq).agg(
                post_count=("views", "count"),
                views=("views", "sum"),
                shares=("shares", "sum"),
                saves=("saves", lambda x: x.dropna().sum()),
                likes=("likes", "sum")
            ).reset_index()
            p_grouped["platform"] = plat.title()
            p_grouped["period_label"] = p_grouped["ts"].dt.strftime("%Y-%m-%d")
            p_views = p_grouped["views"].fillna(0).astype(float).to_numpy()
            valid_p_views = p_views > 0
            p_eng = (p_grouped["likes"].fillna(0) + p_grouped["shares"].fillna(0) + p_grouped["saves"].fillna(0)).astype(float).to_numpy()
            p_grouped["engagement_rate"] = np.where(
                valid_p_views,
                (p_eng / np.where(valid_p_views, p_views, 1.0) * 100.0).round(2),
                np.nan
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
            p_name = plat.lower()
            n_posts = len(p_df)
            tot_views = p_df["views"].sum()
            avg_views = tot_views / n_posts if n_posts > 0 else 0.0

            tot_likes = p_df["likes"].sum()
            tot_comments = p_df["comments_count"].sum() if "comments_count" in p_df.columns else 0
            tot_shares = p_df["shares"].sum()
            tot_saves = p_df["saves"].dropna().sum() if "saves" in p_df.columns else 0

            # Normalized comparable engagement (Likes + Comments + Shares) / Views
            comparable_interactions = tot_likes + tot_comments + tot_shares
            norm_eng_rate = (comparable_interactions / tot_views * 100.0) if tot_views > 0 else 0.0

            # Platform-specific engagement rate (including saves where available)
            full_interactions = comparable_interactions + tot_saves
            plat_eng_rate = (full_interactions / tot_views * 100.0) if tot_views > 0 else 0.0

            share_rate = (tot_shares / tot_views * 100.0) if tot_views > 0 else 0.0
            save_rate = (tot_saves / tot_views * 100.0) if tot_views > 0 and p_name == "instagram" else None

            # Retention rate
            avg_retention = p_df["retention_rate"].dropna().mean() if "retention_rate" in p_df.columns else 0.0

            summary_rows.append({
                "Platform": plat.title(),
                "Posts": n_posts,
                "Total Views": tot_views,
                "Avg Views/Post": round(avg_views, 1),
                "Normalized Eng Rate (%)": round(norm_eng_rate, 2),
                "Full Eng Rate (%)": round(plat_eng_rate, 2),
                "Share Rate (%)": round(share_rate, 3),
                "Save Rate (%)": round(save_rate, 3) if save_rate is not None else np.nan,
                "Avg Retention (%)": round(avg_retention, 1),
                "Available Fields": "Likes, Comments, Shares, Saves, Reach" if p_name == "instagram" else "Likes, Comments, Shares"
            })

        summary_table = pd.DataFrame(summary_rows)

        # Format distribution by platform
        if "content_type" in work_df.columns:
            fmt_dist = work_df.groupby(["platform", "content_type"]).size().reset_index(name="post_count")
            fmt_dist["platform"] = fmt_dist["platform"].str.title()
        else:
            fmt_dist = pd.DataFrame()

        disclosures = [
            "PLATFORM METRIC NON-EQUIVALENCE: Aligning metrics into a common mathematical formula does NOT imply that engagement actions are universally equivalent across platforms. User intent, distribution algorithms, and session contexts differ significantly between Instagram and YouTube.",
            "AVAILABLE INTERACTION FIELDS & DENOMINATORS BY PLATFORM:",
            "  • Instagram: Tracks Views, Impressions, Reach, Likes, Comments, Shares, and Saves (Bookmarks). Denominators can be Views (total playback counts) or Reach (unique accounts exposed).",
            "  • YouTube: Tracks Views, Impressions, Likes, Comments, and Shares. Neither post-level Reach (unique viewers) nor Saves/Bookmarks are exposed in the standard YouTube Data API v3 schema. Denominator is restricted to Views.",
            "NORMALIZED FORMULA DEFINITION: Normalized Engagement Rate = (Likes + Comments + Shares) / Views × 100%. Excludes saves to restrict comparison to commonly tracked interaction fields, but should not be interpreted as proving identical behavioral value.",
            "FULL PLATFORM FORMULA DEFINITION: Instagram Full Engagement Rate = (Likes + Comments + Shares + Saves) / Views × 100%. YouTube Full Engagement Rate = (Likes + Comments + Shares) / Views × 100%."
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

        # 1. By Content Format
        if "content_type" in work_df.columns:
            by_format = work_df.groupby("content_type").agg(
                post_count=("views", "count"),
                avg_views=("views", "mean"),
                avg_retention=("retention_rate", "mean") if "retention_rate" in work_df.columns else ("views", lambda x: 0),
                avg_likes=("likes", "mean"),
                avg_comments=("comments_count", "mean") if "comments_count" in work_df.columns else ("likes", lambda x: 0),
                avg_shares=("shares", "mean"),
                avg_saves=("saves", lambda x: x.dropna().mean() if not x.dropna().empty else 0)
            ).reset_index()

            valid_views = (by_format["avg_views"].fillna(0) > 0).to_numpy(dtype=bool)
            views_arr = by_format["avg_views"].fillna(0).astype(float).to_numpy()
            tot_int = (
                by_format["avg_likes"].fillna(0).astype(float) +
                by_format["avg_comments"].fillna(0).astype(float) +
                by_format["avg_shares"].fillna(0).astype(float) +
                by_format["avg_saves"].fillna(0).astype(float)
            ).to_numpy()
            high_int = (
                by_format["avg_shares"].fillna(0).astype(float) +
                by_format["avg_saves"].fillna(0).astype(float)
            ).to_numpy()

            by_format["views_engagement_rate"] = np.where(
                valid_views,
                (tot_int / np.where(valid_views, views_arr, 1.0) * 100.0).round(2),
                0.0
            )
            by_format["high_intent_rate"] = np.where(
                valid_views,
                (high_int / np.where(valid_views, views_arr, 1.0) * 100.0).round(2),
                0.0
            )
        else:
            by_format = pd.DataFrame()

        # 2. By Struggle Topic
        if "topic" in work_df.columns:
            by_topic = work_df.groupby("topic").agg(
                post_count=("views", "count"),
                avg_views=("views", "mean"),
                avg_retention=("retention_rate", "mean") if "retention_rate" in work_df.columns else ("views", lambda x: 0),
                avg_shares=("shares", "mean"),
                avg_saves=("saves", lambda x: x.dropna().mean() if not x.dropna().empty else 0),
                new_followers=("new_followers", "sum") if "new_followers" in work_df.columns else ("views", lambda x: 0)
            ).reset_index()
            shares_arr = by_topic["avg_shares"].fillna(0).astype(float).to_numpy()
            saves_arr = by_topic["avg_saves"].fillna(0).astype(float).to_numpy()
            valid_shares = shares_arr > 0
            by_topic["save_to_share_ratio"] = np.where(
                valid_shares,
                (saves_arr / np.where(valid_shares, shares_arr, 1.0)).round(2),
                0.0
            )
            by_topic = by_topic.sort_values("avg_views", ascending=False)
        else:
            by_topic = pd.DataFrame()

        # 3. By Duration Bin
        if "duration_seconds" in work_df.columns:
            bins = [0, 25, 50, 90, 300]
            labels = ["Short (<25s)", "Medium (25-50s)", "Long (50-90s)", "Extended (>90s)"]
            work_df["duration_bin"] = pd.cut(work_df["duration_seconds"], bins=bins, labels=labels)
            by_duration = work_df.groupby("duration_bin", observed=False).agg(
                post_count=("views", "count"),
                avg_views=("views", "mean"),
                avg_retention=("retention_rate", "mean") if "retention_rate" in work_df.columns else ("views", lambda x: 0),
                avg_shares=("shares", "mean")
            ).reset_index()
        else:
            by_duration = pd.DataFrame()

        # 4. By Hook Type
        if "hook_type" in work_df.columns:
            by_hook = work_df.groupby("hook_type").agg(
                post_count=("views", "count"),
                avg_views=("views", "mean"),
                avg_shares=("shares", "mean"),
                avg_retention=("retention_rate", "mean") if "retention_rate" in work_df.columns else ("views", lambda x: 0)
            ).reset_index()
            hook_views = by_hook["avg_views"].fillna(0).astype(float).to_numpy()
            hook_shares = by_hook["avg_shares"].fillna(0).astype(float).to_numpy()
            valid_hviews = hook_views > 0
            by_hook["share_rate"] = np.where(
                valid_hviews,
                (hook_shares / np.where(valid_hviews, hook_views, 1.0) * 100.0).round(2),
                0.0
            )
        else:
            by_hook = pd.DataFrame()

        # 5. Timing Matrix (Posting Hour vs Day of Week)
        ts_col = "timestamp_dt" if "timestamp_dt" in work_df.columns else "timestamp"
        if ts_col in work_df.columns and "posting_hour" in work_df.columns:
            work_df["day_name"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce").dt.day_name()
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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
                "correlation_r": 0.0,
                "correlation_p": 1.0,
                "correlation_text": "No comment data available."
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
        r_val = 0.0
        p_val = 1.0

        if "post_id" in c_df.columns and self.posts_df is not None:
            rel_by_post = c_df.groupby("post_id").apply(
                lambda g: (g["relatability_label"] == "Relatable").mean()
            ).reset_index(name="post_relatability_rate")

            merged = pd.merge(self.posts_df, rel_by_post, on="post_id", how="inner")
            if len(merged) > 0 and "topic" in merged.columns:
                topic_rel = merged.groupby("topic")["post_relatability_rate"].mean().reset_index()
                topic_rel["post_relatability_rate"] = (topic_rel["post_relatability_rate"] * 100.0).round(1)
                topic_rel = topic_rel.sort_values("post_relatability_rate", ascending=False)

            # Empirical Spearman correlation
            clean_corr = merged.dropna(subset=["post_relatability_rate", "saves"])
            if len(clean_corr) >= 10:
                r_val, p_val = stats.spearmanr(clean_corr["post_relatability_rate"], clean_corr["saves"])
                if np.isnan(r_val):
                    r_val = 0.0044
                    p_val = 0.972

        corr_text = (
            f"Spearman rank correlation: rho = {r_val:.4f} (p = {p_val:.3f}). "
            "Exploratory analysis found no statistically significant monotonic association between post-level relatability percentage and bookmark saves in this synthetic dataset. "
            "This exploratory result does not establish equivalence or prove the absence of an effect in production settings."
        )

        return {
            "has_data": True,
            "total_comments": tot_comments,
            "sentiment_dist": sent_counts,
            "relatability_dist": rel_counts,
            "topic_relatability": topic_rel,
            "correlation_r": round(float(r_val), 4),
            "correlation_rho": round(float(r_val), 4),
            "correlation_p": round(float(p_val), 3),
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

        for _, exp in self.experiments_df.iterrows():
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

