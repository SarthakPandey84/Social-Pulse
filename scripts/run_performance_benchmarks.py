"""Reproducible runtime and memory benchmark runner for SocialPulse."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.modeling.virality_predictor import ViralityPredictor
from src.preprocessing.feature_engineering import add_engagement_features


# Historical baseline recorded on an earlier local Python 3.11.9 workstation prior to Batch 5A vectorization.
# Kept strictly as a fixed historical comparison reference; NOT re-executed on HEAD.
PRE_OPTIMIZATION_BASELINE = {
    "source_note": (
        "Historical pre-Batch-5A baseline recorded prior to vectorization on a local Python 3.11.9 "
        "workstation (fixed reference for comparison ratios; not re-executed on HEAD)"
    ),
    "evi_5k_seconds": 1.8833,
    "feature_engineering_5k_seconds": 3.3271,
    "dashboard_synthesis_seconds": 0.4129,
    "peak_memory_mb": 15.53,
}


def _time_and_trace(fn) -> tuple[Any, float, float]:
    t0 = time.perf_counter()
    res = fn()
    elapsed = time.perf_counter() - t0
    tracemalloc.start()
    _ = fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, round(elapsed, 4), round(peak_bytes / (1024.0 * 1024.0), 2)


def run_benchmarks(
    root_dir: Path = PROJECT_ROOT,
    output_path: Path | None = None,
    include_volatile_timestamps: bool = False,
) -> Dict[str, Any]:
    processed_posts_path = root_dir / "data" / "processed" / "processed_posts.parquet"
    sample_posts_path = root_dir / "data" / "sample" / "sample_posts.csv"
    if processed_posts_path.is_file():
        base_df = pd.read_parquet(processed_posts_path)
    else:
        base_df = pd.read_csv(sample_posts_path)

    # Construct deterministic ~5,000-row benchmark frame
    reps = max(1, (5000 + len(base_df) - 1) // len(base_df))
    df_5k = pd.concat([base_df] * reps, ignore_index=True).iloc[:5000].copy()
    df_5k["post_id"] = [f"BENCH_{i}" for i in range(len(df_5k))]

    predictor = ViralityPredictor()

    # 1. 5,000-row EVI benchmark
    _, evi_5k_sec, evi_5k_mem = _time_and_trace(
        lambda: predictor.calculate_engagement_virality_index_detailed(df_5k)
    )

    # 2. 5,000-row Feature Engineering benchmark
    _, fe_5k_sec, fe_5k_mem = _time_and_trace(
        lambda: add_engagement_features(df_5k)
    )

    # 3. Dashboard synthesis benchmark (both warm methods and cold init+synth)
    warm_backend = GrowthDashboardBackend(root_dir=root_dir)

    def _dash_warm_synth():
        kpis = warm_backend.calculate_growth_overview_kpis(warm_backend.posts_df)
        ts = warm_backend.calculate_time_series_trends(warm_backend.posts_df, "D")
        bd = warm_backend.calculate_content_breakdowns(warm_backend.posts_df)
        aud = warm_backend.calculate_audience_insights()
        exp = warm_backend.calculate_experiment_performance()
        return (kpis, ts, bd, aud, exp)

    def _dash_cold_synth():
        backend = GrowthDashboardBackend(root_dir=root_dir)
        kpis = backend.calculate_growth_overview_kpis(backend.posts_df)
        ts = backend.calculate_time_series_trends(backend.posts_df, "D")
        bd = backend.calculate_content_breakdowns(backend.posts_df)
        aud = backend.calculate_audience_insights()
        exp = backend.calculate_experiment_performance()
        return (kpis, ts, bd, aud, exp)

    _, dash_sec, dash_mem = _time_and_trace(_dash_warm_synth)
    _, dash_cold_sec, dash_cold_mem = _time_and_trace(_dash_cold_synth)

    peak_after_mb = max(evi_5k_mem, fe_5k_mem, dash_mem)

    report: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "benchmark_reproducibility": {
            "artifact_type": "deterministic_performance_benchmark",
            "baseline_origin": "hardcoded_historical_pre_batch5a_reference",
            "measured_timing_method": "time.perf_counter() and tracemalloc on current HEAD (includes warm-backend and cold-init dashboard paths)",
            "intentionally_variable_fields_excluded": ["generated_at_utc"],
            "runtime_provenance_location": "models/build_provenance.json",
        },
        "pre_optimization_baseline": PRE_OPTIMIZATION_BASELINE,
        "post_optimization_measured": {
            "evi_5k_seconds": evi_5k_sec,
            "evi_5k_peak_mb": evi_5k_mem,
            "evi_5k_speedup_x": round(PRE_OPTIMIZATION_BASELINE["evi_5k_seconds"] / max(evi_5k_sec, 1e-6), 1),
            "feature_engineering_5k_seconds": fe_5k_sec,
            "feature_engineering_5k_peak_mb": fe_5k_mem,
            "feature_engineering_5k_speedup_x": round(PRE_OPTIMIZATION_BASELINE["feature_engineering_5k_seconds"] / max(fe_5k_sec, 1e-6), 1),
            "dashboard_synthesis_seconds": dash_sec,
            "dashboard_synthesis_peak_mb": dash_mem,
            "dashboard_synthesis_speedup_x": round(PRE_OPTIMIZATION_BASELINE["dashboard_synthesis_seconds"] / max(dash_sec, 1e-6), 1),
            "dashboard_cold_init_and_synthesis_seconds": dash_cold_sec,
            "dashboard_cold_init_and_synthesis_peak_mb": dash_cold_mem,
            "peak_memory_mb": peak_after_mb,
            "peak_memory_reduction_pct": round(
                (1.0 - peak_after_mb / max(PRE_OPTIMIZATION_BASELINE["peak_memory_mb"], 1e-6)) * 100.0,
                1,
            ),
        },
        "external_scale_benchmarks": {},
    }

    tk_path = root_dir / "data" / "processed" / "external" / "tiktok_engagement_200k" / "posts.parquet"
    if tk_path.is_file():
        t_load0 = time.perf_counter()
        tk_cols = [
            "post_id", "platform", "content_type", "topic",
            "views", "reach", "impressions", "likes",
            "comments_count", "shares", "saves", "collect_count", "retention_rate",
        ]
        tk_df = pd.read_parquet(tk_path, columns=tk_cols)
        tk_load_sec = round(time.perf_counter() - t_load0, 4)

        _, tk_fe_sec, tk_fe_mem = _time_and_trace(lambda: add_engagement_features(tk_df, access_tier="public_scrape"))
        _, tk_evi_sec, tk_evi_mem = _time_and_trace(lambda: predictor.calculate_engagement_virality_index_detailed(tk_df))
        _, tk_rank_sec, _ = _time_and_trace(lambda: predictor.rank_topics_by_virality(tk_df))

        report["external_scale_benchmarks"]["tiktok_engagement_200k"] = {
            "row_count": int(len(tk_df)),
            "projected_parquet_read_seconds": tk_load_sec,
            "add_engagement_features_seconds": tk_fe_sec,
            "add_engagement_features_peak_mb": tk_fe_mem,
            "evi_detailed_seconds": tk_evi_sec,
            "evi_detailed_peak_mb": tk_evi_mem,
            "rank_topics_seconds": tk_rank_sec,
        }

    if include_volatile_timestamps:
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    return report


def _update_build_provenance(provenance_path: Path, key: str, value: str) -> None:
    """Persist runtime benchmark timestamp into gitignored build provenance."""
    payload: Dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload[key] = value
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(provenance_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    out_path = PROJECT_ROOT / "models" / "performance_benchmarks.json"
    report = run_benchmarks(PROJECT_ROOT, output_path=out_path, include_volatile_timestamps=False)
    _update_build_provenance(
        PROJECT_ROOT / "models" / "build_provenance.json",
        "performance_benchmarks_generated_at_utc",
        datetime.now(timezone.utc).isoformat(),
    )
    print(f"Performance benchmarks written to: {out_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
