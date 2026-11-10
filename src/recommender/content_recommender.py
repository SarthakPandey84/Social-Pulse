"""Engagement Optimization Recommender.

Prescriptive analytics engine recommending optimal topics, video duration,
hook styles, and posting windows for the upcoming week based on historical win-rates
and multi-module empirical evidence.
"""

from typing import Any, Dict, Optional
import pandas as pd

from src.recommender.engine import OptimizationRecommender
from src.recommender.schemas import RecommendationReport


class ContentRecommender:
    """Generates prescriptive content strategies from empirical performance data (Backwards-Compatible API)."""

    def __init__(self, target_metric: str = "viral_coefficient"):
        self.target_metric = target_metric
        self.engine = OptimizationRecommender()

    def recommend_weekly_strategy(self, df: pd.DataFrame, top_k: int = 3) -> Dict[str, Any]:
        """Synthesize high-probability content combinations from historical data."""
        df_eval = df.copy()
        if self.target_metric not in df_eval.columns:
            # Fallback to total interactions if virality metric isn't yet present
            df_eval[self.target_metric] = df_eval["shares"] * 2 + df_eval["saves"] * 2 + df_eval["likes"]

        # 1. Best performing topics
        top_topics = (
            df_eval.groupby("topic")[self.target_metric]
            .mean()
            .sort_values(ascending=False)
            .head(top_k)
            .to_dict()
        )

        # 2. Optimal video length bin
        if "duration_seconds" in df_eval.columns:
            bins = [0, 20, 45, 90, 300]
            labels = ["Short (<20s)", "Medium (20-45s)", "Long (45-90s)", "Extended (>90s)"]
            df_eval["duration_bin"] = pd.cut(df_eval["duration_seconds"], bins=bins, labels=labels)
            best_duration = (
                df_eval.groupby("duration_bin", observed=False)[self.target_metric]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
        else:
            best_duration = "Medium (30-60s)"

        # 3. Best hook type
        if "hook_type" in df_eval.columns:
            best_hook = (
                df_eval.groupby("hook_type")[self.target_metric]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
        else:
            best_hook = "Problem-Focused Question Hook"

        # 4. Best posting hour
        if "posting_hour" in df_eval.columns:
            best_hour = int(
                df_eval.groupby("posting_hour")[self.target_metric]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
            posting_window = f"{best_hour:02d}:00 - {(best_hour+2)%24:02d}:00"
        else:
            posting_window = "18:00 - 21:00"

        # 5. Save-to-Share high resonance topic
        if "save_to_share_ratio" in df_eval.columns:
            high_resonance_topic = (
                df_eval.groupby("topic")["save_to_share_ratio"]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
        else:
            high_resonance_topic = list(top_topics.keys())[0] if top_topics else "N/A"

        return {
            "recommended_topics": top_topics,
            "optimal_duration_bin": str(best_duration),
            "optimal_hook_style": str(best_hook),
            "optimal_posting_window": posting_window,
            "deep_resonance_topic": str(high_resonance_topic),
            "confidence_statement": "Based on empirical high-intent actions (Shares, Saves, Retention) over passive likes."
        }

    def generate_evidence_based_recommendations(
        self,
        platform: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: Optional[str] = None
    ) -> RecommendationReport:
        """Forward to the full OptimizationRecommender engine."""
        return self.engine.generate_recommendations(
            platform=platform,
            category=category,
            min_confidence=min_confidence
        )
