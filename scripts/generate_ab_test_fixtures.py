"""Generate benchmark experiments and observations for Module 4 A/B Testing."""

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data_ingestion.schemas import (
    ExperimentStatus,
    ExperimentType,
    UNIFIED_EXPERIMENT_FIELDS,
    UNIFIED_OBSERVATION_FIELDS,
)


def get_experiment_definitions() -> List[Dict[str, Any]]:
    """Return canonical benchmark experiment metadata records."""
    return [
        {
            "experiment_id": "EXP_001",
            "experiment_name": "Reels Hook Architecture: Story Hook vs Visual Hook",
            "hypothesis": "Visual action hooks will achieve a statistically significant higher 7-day share rate than narrative story hooks.",
            "experiment_type": ExperimentType.RANDOMIZED_EXPERIMENT.value,
            "platform": "instagram",
            "variant_a_label": "Story Hook",
            "variant_b_label": "Visual Hook",
            "primary_metric": "share_rate",
            "secondary_metrics": json.dumps(["retention_rate", "save_rate", "engagement_virality_index"]),
            "start_date": "2026-10-01T00:00:00Z",
            "end_date": "2026-10-21T23:59:59Z",
            "experiment_status": ExperimentStatus.COMPLETED.value,
        },
        {
            "experiment_id": "EXP_002",
            "experiment_name": "YouTube Shorts Title Framing: Question vs Bold Statement",
            "hypothesis": "Bold assertive claim titles will drive higher average views and retention compared to exploratory question titles.",
            "experiment_type": ExperimentType.RANDOMIZED_EXPERIMENT.value,
            "platform": "youtube",
            "variant_a_label": "Question Hook",
            "variant_b_label": "Bold Statement Hook",
            "primary_metric": "views",
            "secondary_metrics": json.dumps(["retention_rate", "share_rate"]),
            "start_date": "2026-10-05T00:00:00Z",
            "end_date": "2026-10-25T23:59:59Z",
            "experiment_status": ExperimentStatus.COMPLETED.value,
        },
        {
            "experiment_id": "EXP_003",
            "experiment_name": "Caption Depth: Short Punchy vs Long Vulnerable Story",
            "hypothesis": "In-depth vulnerable narrative captions will yield higher audience relatability percentage and bookmark saves.",
            "experiment_type": ExperimentType.RANDOMIZED_EXPERIMENT.value,
            "platform": "instagram",
            "variant_a_label": "Short Punchy Caption",
            "variant_b_label": "Long Vulnerable Story",
            "primary_metric": "relatability_pct",
            "secondary_metrics": json.dumps(["save_rate", "comment_rate"]),
            "start_date": "2026-10-10T00:00:00Z",
            "end_date": "2026-10-31T23:59:59Z",
            "experiment_status": ExperimentStatus.COMPLETED.value,
        },
        {
            "experiment_id": "EXP_004",
            "experiment_name": "Posting Window Cohort Comparison: Weekday Morning vs Weekend Evening",
            "hypothesis": "Weekend evening posts will achieve higher overall reach than weekday morning posts (Observational Comparison).",
            "experiment_type": ExperimentType.OBSERVATIONAL_COMPARISON.value,
            "platform": "instagram",
            "variant_a_label": "Weekday Morning (8-10 AM)",
            "variant_b_label": "Weekend Evening (6-8 PM)",
            "primary_metric": "views",
            "secondary_metrics": json.dumps(["shares", "saves"]),
            "start_date": "2026-09-15T00:00:00Z",
            "end_date": "2026-10-30T23:59:59Z",
            "experiment_status": ExperimentStatus.COMPLETED.value,
        },
    ]


def validate_experiments_dataframe(df_experiments: pd.DataFrame) -> None:
    """Validate that an experiments DataFrame adheres to the canonical A/B metadata schema."""
    if df_experiments.empty:
        raise ValueError("sample_experiments.csv is empty; expected experiment definitions.")

    missing_cols = [c for c in UNIFIED_EXPERIMENT_FIELDS if c not in df_experiments.columns]
    if missing_cols:
        raise ValueError(f"sample_experiments.csv is missing required fields: {missing_cols}")

    mandatory_non_null = [
        "experiment_id",
        "experiment_name",
        "experiment_type",
        "platform",
        "variant_a_label",
        "variant_b_label",
        "primary_metric",
        "experiment_status",
    ]
    for col in mandatory_non_null:
        if df_experiments[col].isna().any():
            raise ValueError(f"sample_experiments.csv contains nulls in mandatory column '{col}'.")

    valid_types = {e.value for e in ExperimentType}
    bad_types = set(df_experiments["experiment_type"].astype(str)) - valid_types
    if bad_types:
        raise ValueError(f"sample_experiments.csv contains invalid experiment_type values: {sorted(bad_types)}")

    valid_statuses = {e.value for e in ExperimentStatus}
    bad_statuses = set(df_experiments["experiment_status"].astype(str)) - valid_statuses
    if bad_statuses:
        raise ValueError(f"sample_experiments.csv contains invalid experiment_status values: {sorted(bad_statuses)}")

    expected_ids = {"EXP_001", "EXP_002", "EXP_003", "EXP_004"}
    present_ids = set(df_experiments["experiment_id"].astype(str))
    missing_ids = expected_ids - present_ids
    if missing_ids:
        raise ValueError(f"sample_experiments.csv is missing required benchmark experiments: {sorted(missing_ids)}")


def generate_synthetic_observations(seed: int = 42) -> pd.DataFrame:
    """Generate deterministic post-level observations for EXP_001 through EXP_004."""
    random.seed(seed)
    np.random.seed(seed)

    observations: List[Dict[str, Any]] = []
    obs_id = 1

    # EXP_001: 60 posts per variant (Story vs Visual Hook)
    # Story Hook: baseline share_rate ~ 2.4%, retention ~ 0.58
    # Visual Hook: treatment share_rate ~ 3.8% (+58% lift), retention ~ 0.65
    for i in range(60):
        reach_a = int(np.random.lognormal(mean=9.5, sigma=0.5))
        views_a = int(reach_a * np.random.uniform(1.1, 1.4))
        shares_a = int(np.random.binomial(n=reach_a, p=0.024))
        saves_a = int(np.random.binomial(n=reach_a, p=0.035))
        ret_a = float(np.clip(np.random.normal(loc=0.58, scale=0.08), 0.2, 0.95))
        rel_a = float(np.clip(np.random.normal(loc=42.0, scale=12.0), 5.0, 95.0))

        observations.append({
            "experiment_id": "EXP_001",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Story Hook",
            "post_id": f"EXP1_A_{i+1:03d}",
            "platform": "instagram",
            "collection_timestamp": "2026-10-22T12:00:00Z",
            "views": views_a,
            "reach": reach_a,
            "shares": shares_a,
            "saves": saves_a,
            "likes": int(reach_a * 0.08),
            "comments_count": int(reach_a * 0.015),
            "retention_rate": round(ret_a, 4),
            "relatability_pct": round(rel_a, 2),
            "share_rate": round(shares_a / reach_a, 5),
            "save_rate": round(saves_a / reach_a, 5),
        })
        obs_id += 1

        reach_b = int(np.random.lognormal(mean=9.6, sigma=0.5))
        views_b = int(reach_b * np.random.uniform(1.2, 1.5))
        shares_b = int(np.random.binomial(n=reach_b, p=0.038))
        saves_b = int(np.random.binomial(n=reach_b, p=0.039))
        ret_b = float(np.clip(np.random.normal(loc=0.65, scale=0.07), 0.2, 0.95))
        rel_b = float(np.clip(np.random.normal(loc=46.0, scale=11.0), 5.0, 95.0))

        observations.append({
            "experiment_id": "EXP_001",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Visual Hook",
            "post_id": f"EXP1_B_{i+1:03d}",
            "platform": "instagram",
            "collection_timestamp": "2026-10-22T12:00:00Z",
            "views": views_b,
            "reach": reach_b,
            "shares": shares_b,
            "saves": saves_b,
            "likes": int(reach_b * 0.09),
            "comments_count": int(reach_b * 0.018),
            "retention_rate": round(ret_b, 4),
            "relatability_pct": round(rel_b, 2),
            "share_rate": round(shares_b / reach_b, 5),
            "save_rate": round(saves_b / reach_b, 5),
        })
        obs_id += 1

    # EXP_002: YouTube Shorts Question vs Bold Statement (50 per variant)
    for i in range(50):
        v_a = int(np.random.normal(loc=14500, scale=3200))
        v_b = int(np.random.normal(loc=19200, scale=3800))
        r_a = float(np.clip(np.random.normal(loc=0.62, scale=0.06), 0.3, 0.95))
        r_b = float(np.clip(np.random.normal(loc=0.69, scale=0.05), 0.3, 0.95))

        observations.append({
            "experiment_id": "EXP_002",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Question Hook",
            "post_id": f"EXP2_A_{i+1:03d}",
            "platform": "youtube",
            "collection_timestamp": "2026-10-26T12:00:00Z",
            "views": v_a,
            "reach": v_a,
            "shares": int(v_a * 0.015),
            "saves": 0,  # YouTube doesn't expose saves
            "likes": int(v_a * 0.06),
            "comments_count": int(v_a * 0.01),
            "retention_rate": round(r_a, 4),
            "relatability_pct": round(float(np.random.normal(38, 10)), 2),
            "share_rate": 0.015,
            "save_rate": 0.0,
        })
        obs_id += 1

        observations.append({
            "experiment_id": "EXP_002",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Bold Statement Hook",
            "post_id": f"EXP2_B_{i+1:03d}",
            "platform": "youtube",
            "collection_timestamp": "2026-10-26T12:00:00Z",
            "views": v_b,
            "reach": v_b,
            "shares": int(v_b * 0.022),
            "saves": 0,
            "likes": int(v_b * 0.07),
            "comments_count": int(v_b * 0.012),
            "retention_rate": round(r_b, 4),
            "relatability_pct": round(float(np.random.normal(44, 9)), 2),
            "share_rate": 0.022,
            "save_rate": 0.0,
        })
        obs_id += 1

    # EXP_003: Caption Depth (Short Punchy vs Long Vulnerable Story) - 45 per variant
    for i in range(45):
        rel_a = float(np.clip(np.random.normal(loc=31.5, scale=8.2), 5.0, 80.0))
        rel_b = float(np.clip(np.random.normal(loc=54.2, scale=9.1), 10.0, 95.0))
        reach = int(np.random.lognormal(mean=9.2, sigma=0.4))

        observations.append({
            "experiment_id": "EXP_003",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Short Punchy Caption",
            "post_id": f"EXP3_A_{i+1:03d}",
            "platform": "instagram",
            "collection_timestamp": "2026-11-01T12:00:00Z",
            "views": int(reach * 1.2),
            "reach": reach,
            "shares": int(reach * 0.02),
            "saves": int(reach * 0.025),
            "likes": int(reach * 0.06),
            "comments_count": int(reach * 0.008),
            "retention_rate": 0.55,
            "relatability_pct": round(rel_a, 2),
            "share_rate": 0.02,
            "save_rate": 0.025,
        })
        obs_id += 1

        observations.append({
            "experiment_id": "EXP_003",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Long Vulnerable Story",
            "post_id": f"EXP3_B_{i+1:03d}",
            "platform": "instagram",
            "collection_timestamp": "2026-11-01T12:00:00Z",
            "views": int(reach * 1.25),
            "reach": reach,
            "shares": int(reach * 0.028),
            "saves": int(reach * 0.052),
            "likes": int(reach * 0.08),
            "comments_count": int(reach * 0.022),
            "retention_rate": 0.58,
            "relatability_pct": round(rel_b, 2),
            "share_rate": 0.028,
            "save_rate": 0.052,
        })
        obs_id += 1

    # EXP_004: Observational Posting Window (35 per cohort)
    for i in range(35):
        v_a = int(np.random.normal(loc=11200, scale=2800))
        v_b = int(np.random.normal(loc=13100, scale=3100))

        observations.append({
            "experiment_id": "EXP_004",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Weekday Morning (8-10 AM)",
            "post_id": f"EXP4_A_{i+1:03d}",
            "platform": "instagram",
            "collection_timestamp": "2026-10-31T18:00:00Z",
            "views": v_a,
            "reach": int(v_a * 0.85),
            "shares": int(v_a * 0.021),
            "saves": int(v_a * 0.03),
            "likes": int(v_a * 0.07),
            "comments_count": int(v_a * 0.01),
            "retention_rate": 0.54,
            "relatability_pct": 39.0,
            "share_rate": 0.021,
            "save_rate": 0.03,
        })
        obs_id += 1

        observations.append({
            "experiment_id": "EXP_004",
            "observation_id": f"OBS_{obs_id:05d}",
            "variant": "Weekend Evening (6-8 PM)",
            "post_id": f"EXP4_B_{i+1:03d}",
            "platform": "instagram",
            "collection_timestamp": "2026-10-31T18:00:00Z",
            "views": v_b,
            "reach": int(v_b * 0.88),
            "shares": int(v_b * 0.024),
            "saves": int(v_b * 0.034),
            "likes": int(v_b * 0.08),
            "comments_count": int(v_b * 0.012),
            "retention_rate": 0.57,
            "relatability_pct": 42.0,
            "share_rate": 0.024,
            "save_rate": 0.034,
        })
        obs_id += 1

    return pd.DataFrame(observations)[UNIFIED_OBSERVATION_FIELDS]


def generate_ab_fixtures(
    base_dir: Optional[Path] = None,
    overwrite_sample: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate processed A/B observation fixtures without mutating tracked sample files when overwrite_sample=False."""
    target_root = Path(base_dir) if base_dir is not None else root_dir
    exp_csv_path = target_root / "data" / "sample" / "sample_experiments.csv"

    if overwrite_sample:
        df_experiments = pd.DataFrame(get_experiment_definitions())
        exp_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_experiments.to_csv(exp_csv_path, index=False)
    else:
        if not exp_csv_path.exists():
            raise FileNotFoundError(f"Required experiment definitions file not found at: {exp_csv_path}")
        df_experiments = pd.read_csv(exp_csv_path)
        validate_experiments_dataframe(df_experiments)

    df_obs = generate_synthetic_observations(seed=seed)

    processed_dir = target_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    obs_parquet_path = processed_dir / "processed_ab_tests.parquet"
    obs_csv_path = processed_dir / "processed_ab_tests.csv"

    df_obs.to_parquet(obs_parquet_path, index=False)
    df_obs.to_csv(obs_csv_path, index=False)

    return {
        "experiments_count": len(df_experiments),
        "observations_count": len(df_obs),
        "exp_csv_path": exp_csv_path,
        "obs_parquet_path": obs_parquet_path,
        "obs_csv_path": obs_csv_path,
    }


def main() -> None:
    res = generate_ab_fixtures(base_dir=root_dir, overwrite_sample=True, seed=42)
    print(f"Saved {res['experiments_count']} experiment definitions to {res['exp_csv_path']}")
    print(f"Saved {res['observations_count']} observations to {res['obs_parquet_path']} and {res['obs_csv_path']}")


if __name__ == "__main__":
    main()
