"""Engagement Metrics and Growth Correlation Analytics."""

from typing import Any, Dict
import pandas as pd
import numpy as np


class EngagementAnalytics:
    """Computes global performance indicators and correlation with follower growth."""

    @staticmethod
    def calculate_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate platform-wide key performance indicators."""
        total_posts = len(df)
        total_views = int(df["views"].sum()) if "views" in df.columns else 0
        total_shares = int(df["shares"].sum()) if "shares" in df.columns else 0
        total_saves = int(df["saves"].sum()) if "saves" in df.columns else 0
        avg_retention = float(df["retention_rate"].mean()) if "retention_rate" in df.columns else 0.0

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
    def topic_follower_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """Map the relationship between struggle topics and new follower growth."""
        if "new_followers" not in df.columns or "topic" not in df.columns:
            return pd.DataFrame()

        summary = df.groupby("topic").agg(
            posts_count=("post_id", "count"),
            total_new_followers=("new_followers", "sum"),
            avg_followers_per_post=("new_followers", "mean"),
            avg_saves=("saves", "mean"),
            avg_shares=("shares", "mean")
        ).reset_index()

        return summary.sort_values(by="total_new_followers", ascending=False).reset_index(drop=True)

