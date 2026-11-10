"""Real-Data Empirical Validation Suite across Modules 1–7.

Runs strictly isolated, leakage-free benchmarks against verified processed external datasets:
- M1: Metric availability & creator-relative aggregation on TikTok 200K & Instagram cohorts.
- M2: Longitudinal TikTok virality prediction (`evaluate_tiktok_longitudinal`) comparing
      strict `pre_pub` vs `early_day1` feature regimes across `chronological` & `creator_grouped`
      splits on mature `day30` (and `day7`) targets, plus empirical 90% coverage checks.
- M3: Held-out Instagram sentiment benchmark (`eval_split == 'train'` -> `'test'` only),
      Lokananta Indonesian non-English guardrail, and InstaArt human-vs-AI art comments.
- M4: Observational cohort evaluation on `zenodo_sponsored_unsponsored_ig` (post-level vs
      creator-clustered DEFF/ICC, follower-tertile stratified check, strict non-causal cap).
- M5: Recommendation engine on real TikTok posts verifying real-provenance inference and
      creator-support guardrails (`MIN_CREATORS_FOR_REC = 3`).
- M6: Dashboard backend loading `tiktok_engagement_200k` and static `zenodo_sponsored_unsponsored_ig`.
- M7: Rolling-origin walk-forward validation on `tiktok_engagement_200k/daily_platform_summary.parquet`
      (`daily_views` and `daily_collects`) via gap-free contiguous segment handling.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.analytics.engagement_metrics import EngagementAnalytics
from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.modeling.trend_forecaster import TrendForecaster
from src.modeling.virality_predictor import ViralityPredictor
from src.nlp.evaluation import (
    evaluate_heldout_instagram_sentiment,
    evaluate_instaart_comments,
    evaluate_lokananta_language_guardrail,
)
from src.recommender.engine import OptimizationRecommender
from src.statistics.ab_testing import ABTestingFramework


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalars and dataclasses cleanly for JSON serialization."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_json_safe(obj.model_dump())
    if hasattr(obj, "value"):
        return _to_json_safe(obj.value)
    return str(obj)


def run_all_real_data_benchmarks(
    processed_external_root: Path = Path("data/processed/external"),
    output_path: Optional[Path] = Path("models/real_data_benchmarks.json"),
    include_volatile_timestamps: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end empirical validation across all available processed external datasets."""
    root = Path(processed_external_root)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        root_display = root.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        root_display = root.as_posix()
    report: Dict[str, Any] = {
        "processed_external_root": root_display,
        "benchmark_reproducibility": {
            "artifact_type": "deterministic_real_data_benchmark",
            "intentionally_variable_fields_excluded": ["generated_at_utc"],
            "runtime_provenance_location": "models/build_provenance.json",
        },
        "modules": {},
    }
    if include_volatile_timestamps:
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    tk_posts_path = root / "tiktok_engagement_200k" / "posts.parquet"
    tk_daily_path = root / "tiktok_engagement_200k" / "daily_platform_summary.parquet"
    ig_sent_path = root / "instagram_comments_sentiment_hf" / "comments.parquet"
    lokananta_path = root / "zenodo_lokananta_comments_ig" / "comments.parquet"
    instaart_posts_path = root / "instaart_human_ai" / "posts.parquet"
    instaart_comments_path = root / "instaart_human_ai" / "comments.parquet"
    sponsored_path = root / "zenodo_sponsored_unsponsored_ig" / "posts.parquet"

    tk_posts: Optional[pd.DataFrame] = None
    if tk_posts_path.exists():
        tk_posts = pd.read_parquet(tk_posts_path)

    # --- M1: Metric Availability & Creator-Relative Aggregation ---
    m1_res: Dict[str, Any] = {}
    if tk_posts is not None:
        m1_res["tiktok_metric_availability"] = EngagementAnalytics.summarize_metric_availability(tk_posts)
        rel_df = EngagementAnalytics.compute_creator_relative_metrics(
            tk_posts[["post_id", "creator_pseudo_id", "views"]].dropna(subset=["views"]),
            creator_col="creator_pseudo_id",
            value_col="views",
        )
        m1_res["tiktok_creator_relative_views"] = {
            "rows_evaluated": int(len(rel_df)),
            "eligible_creators_min_5_posts": int(rel_df.loc[rel_df["creator_post_count"] >= 5, "creator_pseudo_id"].nunique()),
            "median_relative_ratio": round(float(rel_df["creator_relative_ratio"].dropna().median()), 4),
            "p90_relative_ratio": round(float(rel_df["creator_relative_ratio"].dropna().quantile(0.90)), 4),
        }
    if sponsored_path.exists():
        import pyarrow.parquet as pq

        avail_sp_cols = pq.read_schema(sponsored_path).names
        wanted_sp_cols = [
            col
            for col in (
                "post_id",
                "platform",
                "content_type",
                "views",
                "likes",
                "comments_count",
                "shares",
                "saves",
                "follower_count",
                "provenance",
            )
            if col in avail_sp_cols
        ]
        sp_meta_df = pd.read_parquet(sponsored_path, columns=wanted_sp_cols)
        m1_res["sponsored_ig_metric_availability"] = EngagementAnalytics.summarize_metric_availability(sp_meta_df)
    report["modules"]["module1"] = m1_res

    # --- M2: TikTok Longitudinal Virality Prediction ---
    m2_res: Dict[str, Any] = {}
    if tk_posts is not None:
        vp = ViralityPredictor()
        m2_res["day30_pre_pub_chronological"] = vp.evaluate_tiktok_longitudinal(
            tk_posts, target_horizon="day30", feature_regime="pre_pub", split_strategy="chronological", random_state=42
        )
        m2_res["day30_pre_pub_creator_grouped"] = vp.evaluate_tiktok_longitudinal(
            tk_posts, target_horizon="day30", feature_regime="pre_pub", split_strategy="creator_grouped", random_state=42
        )
        m2_res["day30_early_day1_chronological"] = vp.evaluate_tiktok_longitudinal(
            tk_posts, target_horizon="day30", feature_regime="early_day1", split_strategy="chronological", random_state=42
        )
        m2_res["day30_early_day1_creator_grouped"] = vp.evaluate_tiktok_longitudinal(
            tk_posts, target_horizon="day30", feature_regime="early_day1", split_strategy="creator_grouped", random_state=42
        )
        m2_res["day7_pre_pub_creator_grouped"] = vp.evaluate_tiktok_longitudinal(
            tk_posts, target_horizon="day7", feature_regime="pre_pub", split_strategy="creator_grouped", random_state=42
        )
    report["modules"]["module2"] = m2_res

    # --- M3: NLP Held-Out & Language Guardrail Benchmarks ---
    m3_res: Dict[str, Any] = {}
    if ig_sent_path.exists():
        m3_res["instagram_sentiment_heldout"] = evaluate_heldout_instagram_sentiment(pd.read_parquet(ig_sent_path))
    if lokananta_path.exists():
        m3_res["lokananta_indonesian_guardrail"] = evaluate_lokananta_language_guardrail(pd.read_parquet(lokananta_path))
    if instaart_comments_path.exists():
        m3_res["instaart_comments"] = evaluate_instaart_comments(pd.read_parquet(instaart_comments_path))
    report["modules"]["module3"] = m3_res

    # --- M4: Sponsored vs Unsponsored Creator-Clustered Observational Cohort ---
    m4_res: Dict[str, Any] = {}
    if sponsored_path.exists():
        sp_df = pd.read_parquet(sponsored_path)
        ab = ABTestingFramework(alpha=0.05)
        m4_res["sponsored_unsponsored_observational"] = ab.evaluate_observational_cohort(sp_df)
    if instaart_posts_path.exists():
        ia_posts = pd.read_parquet(instaart_posts_path)
        m4_res["instaart_group_counts"] = {
            "total_posts": int(len(ia_posts)),
            "unique_creators": int(ia_posts["creator_pseudo_id"].dropna().nunique()) if "creator_pseudo_id" in ia_posts.columns else 0,
            "account_group_counts": ia_posts["account_group"].value_counts(dropna=False).to_dict() if "account_group" in ia_posts.columns else {},
            "unique_pairs": int(ia_posts["pair_id"].dropna().nunique()) if "pair_id" in ia_posts.columns else 0,
        }
    report["modules"]["module4"] = m4_res

    # --- M5: Recommender Engine on Real TikTok Data ---
    m5_res: Dict[str, Any] = {}
    if tk_posts is not None:
        # Take mature day30 slice or sample up to 25,000 recent posts for crisp recommendation latency benchmark
        rec_sample = tk_posts.head(25_000).copy()
        recommender = OptimizationRecommender(posts_df=rec_sample, comments_df=None, experiments_df=None)
        rec_report = recommender.generate_recommendations(platform="tiktok", max_recommendations=12)
        m5_res["tiktok_sample_recommendations"] = {
            "sample_rows": int(len(rec_sample)),
            "unique_creators": int(rec_sample["creator_pseudo_id"].dropna().nunique()),
            "is_synthetic_demonstration": bool(rec_report.is_synthetic_demonstration),
            "all_recs_marked_real": all(not r.is_synthetic_demonstration for r in rec_report.recommendations),
            "total_recommendations": int(len(rec_report.recommendations)),
            "recommendation_ids": [r.recommendation_id for r in rec_report.recommendations],
            "confidence_levels": {r.recommendation_id: r.confidence_level.value for r in rec_report.recommendations},
            "evidence_breakdown": rec_report.evidence_breakdown,
        }
    report["modules"]["module5"] = m5_res

    # --- M6: Dashboard Backend with External Datasets ---
    m6_res: Dict[str, Any] = {}
    if tk_posts is not None:
        tk_sample = tk_posts.head(25_000).copy()
        dash_tk = GrowthDashboardBackend(posts_df=tk_sample, dataset_id="tiktok_engagement_200k")
        kpis_tk = dash_tk.calculate_growth_overview_kpis(dash_tk.posts_df)
        plats_tk = dash_tk.calculate_platform_comparisons(dash_tk.posts_df)
        m6_res["tiktok_sample_kpis"] = kpis_tk
        m6_res["tiktok_platform_comparison"] = (
            plats_tk["summary_table"].to_dict("records")
            if plats_tk.get("summary_table") is not None and not plats_tk["summary_table"].empty
            else []
        )
    if sponsored_path.exists():
        sp_sub = pd.read_parquet(sponsored_path).head(10_000)
        dash_sp = GrowthDashboardBackend(posts_df=sp_sub, dataset_id="zenodo_sponsored_unsponsored_ig")
        breakdowns_sp = dash_sp.calculate_content_breakdowns(dash_sp.posts_df)
        fmt_df = breakdowns_sp.get("by_format", pd.DataFrame())
        m6_res["sponsored_static_content_breakdown"] = (
            fmt_df.to_dict("records")
            if fmt_df is not None and not fmt_df.empty
            else []
        )
    report["modules"]["module6"] = m6_res

    # --- M7: Trend Forecasting on Real TikTok Daily Increments ---
    m7_res: Dict[str, Any] = {}
    if tk_daily_path.exists():
        daily_df = pd.read_parquet(tk_daily_path)
        tf = TrendForecaster()
        for m_col in ("daily_views", "daily_collects"):
            rep = tf.forecast_daily_increment_series(
                daily_df, metric_col=m_col, platform="tiktok", horizon=7, segment_policy="longest_contiguous"
            )
            m7_res[m_col] = {
                "status": rep.status.value,
                "history_start": rep.history_start,
                "history_end": rep.history_end,
                "history_points": int(rep.history_points),
                "missing_calendar_days_full_span": rep.missing_calendar_days,
                "overall_coverage_pct": rep.coverage_pct,
                "coverage_status": rep.coverage_status,
                "champion_model": rep.champion_model,
                "validation_origins_count": int(rep.validation_origins_count),
                "validation_observations_count": int(rep.validation_observations_count),
                "is_calibrated": bool(rep.is_calibrated),
                "disclosures_count": len(rep.disclosures),
                "evaluations": [
                    {
                        "model_name": e.model_name,
                        "mae": e.mae,
                        "rmse": e.rmse,
                        "mase": e.mase,
                        "relative_mae_to_naive": e.relative_mae_to_naive,
                        "is_champion": e.is_champion,
                    }
                    for e in rep.model_evaluations
                ],
                "horizon_forecast_step_1": rep.forecast_points[0].point_forecast if rep.forecast_points else None,
                "horizon_forecast_step_7": rep.forecast_points[-1].point_forecast if rep.forecast_points else None,
            }
    report["modules"]["module7"] = m7_res

    safe_report = _to_json_safe(report)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(safe_report, handle, indent=2)
            handle.write("\n")

    return safe_report
