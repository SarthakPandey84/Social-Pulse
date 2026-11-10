"""Idempotent end-to-end artifact build and verification pipeline for SocialPulse.

Rebuilds all derived runtime artifacts (processed datasets, SQLite warehouse,
and serialized ML/NLP models + metadata) strictly from checked-in inputs in
``data/sample/`` without modifying or overwriting any source sample CSVs.

Usage:
    python scripts/build_all_artifacts.py
    python scripts/build_all_artifacts.py --validate-only
"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import sys
import threading
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import sklearn

# Ensure project root is on sys.path before package imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.sample_ingestion import SampleDataIngestion
from src.data_ingestion.schemas import (
    UNIFIED_COMMENT_FIELDS,
    UNIFIED_OBSERVATION_FIELDS,
    UNIFIED_POST_FIELDS,
)
from src.data_ingestion.storage import DataStorageManager
from src.modeling.virality_predictor import ViralityPredictor
from src.nlp.relatability_classifier import RelatabilityClassifier
from src.preprocessing.cleaning import clean_posts_dataset
from src.preprocessing.feature_engineering import add_engagement_features
from scripts.generate_ab_test_fixtures import (
    generate_ab_fixtures,
    validate_experiments_dataframe,
)
from scripts.train_and_benchmark_nlp import (
    run_nlp_pipeline,
    validate_sample_comments_dataframe,
)


REQUIRED_ENGINEERED_POST_COLUMNS = [
    "high_intent_actions",
    "high_intent_completeness",
    "save_to_share_ratio",
    "intent_to_passive_ratio",
    "views_engagement_rate",
    "reach_engagement_rate",
    "overall_engagement_rate",
    "equivalent_full_watch_views",
    "data_completeness_pct",
]

REQUIRED_POST_MODELING_CONTEXT_COLUMNS = [
    "topic",
    "hook_type",
    "duration_seconds",
    "posting_hour",
]

REQUIRED_TREND_COLUMNS = ["date", "keyword", "mention_volume"]

REQUIRED_ARTIFACT_RELATIVE_PATHS: Tuple[Path, ...] = (
    Path("data") / "processed" / "processed_posts.parquet",
    Path("data") / "processed" / "processed_posts.csv",
    Path("data") / "processed" / "processed_comments.parquet",
    Path("data") / "processed" / "processed_comments.csv",
    Path("data") / "processed" / "processed_ab_tests.parquet",
    Path("data") / "processed" / "processed_ab_tests.csv",
    Path("database") / "social_analytics.db",
    Path("models") / "virality_predictor.joblib",
    Path("models") / "virality_predictor_metadata.json",
    Path("models") / "relatability_classifier.joblib",
    Path("models") / "relatability_metadata.json",
)

_BUILD_LOCKS_GUARD = threading.Lock()
_BUILD_LOCKS: Dict[Path, threading.RLock] = {}


def get_build_lock(root_dir: Optional[Path] = None) -> threading.RLock:
    """Return the process-wide re-entrant lock coordinating builds for ``root_dir``."""
    target_root = Path(root_dir).resolve() if root_dir is not None else ROOT_DIR
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.get(target_root)
        if lock is None:
            lock = threading.RLock()
            _BUILD_LOCKS[target_root] = lock
        return lock


def artifacts_present_and_nonempty(root_dir: Optional[Path] = None) -> bool:
    """Return True iff every required generated artifact exists on disk and is non-empty."""
    target_root = Path(root_dir).resolve() if root_dir is not None else ROOT_DIR
    for rel_path in REQUIRED_ARTIFACT_RELATIVE_PATHS:
        candidate = target_root / rel_path
        try:
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _validate_sample_posts(root_dir: Path) -> pd.DataFrame:
    """Validate sample_posts.csv upfront before any artifact writes occur."""
    sample_dir = root_dir / "data" / "sample"
    posts_csv = sample_dir / "sample_posts.csv"
    if not posts_csv.exists():
        raise FileNotFoundError(f"Required input file not found: {posts_csv}")

    raw_df = pd.read_csv(posts_csv)
    if raw_df.empty:
        raise ValueError(f"Input dataset is empty: {posts_csv}")

    missing_context = [c for c in REQUIRED_POST_MODELING_CONTEXT_COLUMNS if c not in raw_df.columns]
    if missing_context:
        raise ValueError(f"sample_posts.csv missing required modeling context columns: {missing_context}")

    schema_ok, schema_errors = DataValidator.validate_posts_schema(raw_df)
    if not schema_ok:
        raise ValueError(f"sample_posts.csv schema validation failed: {'; '.join(schema_errors)}")

    try:
        ingested_df, _ = SampleDataIngestion(sample_dir=sample_dir).load_sample_posts("sample_posts.csv")
    except Exception as exc:
        raise ValueError(f"Failed loading and typing sample_posts.csv: {exc}") from exc

    unified_report = DataValidator.validate_unified_posts(ingested_df)
    if not unified_report["is_valid"]:
        raise ValueError(
            f"sample_posts.csv unified schema validation failed: {'; '.join(unified_report['errors'])}"
        )

    cleaned_df = clean_posts_dataset(ingested_df)
    featured_df = add_engagement_features(cleaned_df)
    mature_df = ViralityPredictor().prepare_dataset_for_modeling(featured_df, min_elapsed_days=7.0)
    if len(mature_df) < ViralityPredictor.MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"sample_posts.csv contains only {len(mature_df)} mature unique posts (>= 7 days), "
            f"below minimum required {ViralityPredictor.MIN_TRAINING_SAMPLES}."
        )

    return raw_df


def _validate_sample_comments(root_dir: Path) -> pd.DataFrame:
    """Validate sample_comments.csv upfront."""
    comments_csv = root_dir / "data" / "sample" / "sample_comments.csv"
    if not comments_csv.exists():
        raise FileNotFoundError(f"Required input file not found: {comments_csv}")

    df_comments = pd.read_csv(comments_csv)
    valid_ok, schema_errors = DataValidator.validate_comments_schema(df_comments)
    if not valid_ok:
        raise ValueError(f"sample_comments.csv validation failed: {'; '.join(schema_errors)}")

    validate_sample_comments_dataframe(df_comments)

    non_null_ts = df_comments["comment_timestamp"].dropna()
    if not non_null_ts.empty:
        parsed_ts = pd.to_datetime(non_null_ts, utc=True, errors="coerce")
        bad_ts = int(parsed_ts.isna().sum())
        if bad_ts > 0:
            raise ValueError(f"sample_comments.csv contains {bad_ts} unparseable comment_timestamp values.")

    return df_comments


def _validate_sample_experiments(root_dir: Path) -> pd.DataFrame:
    """Validate sample_experiments.csv upfront."""
    exp_csv = root_dir / "data" / "sample" / "sample_experiments.csv"
    if not exp_csv.exists():
        raise FileNotFoundError(f"Required input file not found: {exp_csv}")

    df_exp = pd.read_csv(exp_csv)
    validate_experiments_dataframe(df_exp)
    return df_exp


def _validate_sample_trends(root_dir: Path) -> pd.DataFrame:
    """Validate sample_trends.csv upfront."""
    trends_csv = root_dir / "data" / "sample" / "sample_trends.csv"
    if not trends_csv.exists():
        raise FileNotFoundError(f"Required input file not found: {trends_csv}")

    df_trends = pd.read_csv(trends_csv)
    if df_trends.empty:
        raise ValueError(f"Input dataset is empty: {trends_csv}")

    missing_cols = [c for c in REQUIRED_TREND_COLUMNS if c not in df_trends.columns]
    if missing_cols:
        raise ValueError(f"sample_trends.csv is missing required columns: {missing_cols}")

    if df_trends["date"].isna().any():
        raise ValueError("sample_trends.csv contains missing dates.")
    parsed_dates = pd.to_datetime(df_trends["date"], errors="coerce")
    bad_dates = int(parsed_dates.isna().sum())
    if bad_dates > 0:
        raise ValueError(f"sample_trends.csv contains {bad_dates} unparseable date values.")

    if df_trends["keyword"].isna().any() or (df_trends["keyword"].astype(str).str.strip() == "").any():
        raise ValueError("sample_trends.csv contains null or blank keyword values.")

    volumes = pd.to_numeric(df_trends["mention_volume"], errors="coerce")
    if volumes.isna().any():
        raise ValueError("sample_trends.csv contains non-numeric or missing mention_volume values.")
    if (volumes < 0).any():
        raise ValueError("sample_trends.csv contains negative mention_volume values.")

    return df_trends


def validate_source_inputs(root_dir: Path) -> Dict[str, Any]:
    """Fail-fast validation of all required read-only inputs in ``data/sample/``."""
    df_posts = _validate_sample_posts(root_dir)
    df_comments = _validate_sample_comments(root_dir)
    df_exp = _validate_sample_experiments(root_dir)
    df_trends = _validate_sample_trends(root_dir)

    return {
        "sample_posts_rows": len(df_posts),
        "sample_comments_rows": len(df_comments),
        "sample_experiments_rows": len(df_exp),
        "sample_trends_rows": len(df_trends),
    }


def _assert_file_present_and_nonempty(file_path: Path) -> None:
    if not file_path.exists():
        raise RuntimeError(f"Expected generated artifact is missing: {file_path}")
    if file_path.stat().st_size <= 0:
        raise RuntimeError(f"Generated artifact is empty (0 bytes): {file_path}")


def verify_generated_artifacts(
    root_dir: Path,
    expected_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Verify presence, schema conformance, DB integrity, and model reloadability of all built artifacts."""
    processed_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"
    db_path = root_dir / "database" / "social_analytics.db"

    for rel_path in REQUIRED_ARTIFACT_RELATIVE_PATHS:
        _assert_file_present_and_nonempty(root_dir / rel_path)

    # 1. Verify processed_posts
    df_posts_pq = pd.read_parquet(processed_dir / "processed_posts.parquet")
    df_posts_csv = pd.read_csv(processed_dir / "processed_posts.csv")
    if df_posts_pq.empty or len(df_posts_pq) != len(df_posts_csv):
        raise RuntimeError(
            f"processed_posts Parquet ({len(df_posts_pq)}) and CSV ({len(df_posts_csv)}) mismatch or empty."
        )
    missing_post_cols = [
        c for c in (UNIFIED_POST_FIELDS + REQUIRED_ENGINEERED_POST_COLUMNS) if c not in df_posts_pq.columns
    ]
    if missing_post_cols:
        raise RuntimeError(f"processed_posts.parquet missing expected columns: {missing_post_cols}")
    posts_val = DataValidator.validate_unified_posts(df_posts_pq)
    if not posts_val["is_valid"]:
        raise RuntimeError(f"processed_posts.parquet failed unified validation: {posts_val['errors']}")
    if expected_counts and "posts_rows" in expected_counts and len(df_posts_pq) != expected_counts["posts_rows"]:
        raise RuntimeError(
            f"processed_posts row count mismatch: expected {expected_counts['posts_rows']}, got {len(df_posts_pq)}"
        )

    # 2. Verify processed_comments
    df_cmt_pq = pd.read_parquet(processed_dir / "processed_comments.parquet")
    df_cmt_csv = pd.read_csv(processed_dir / "processed_comments.csv")
    sample_cmt_df = pd.read_csv(root_dir / "data" / "sample" / "sample_comments.csv")
    if df_cmt_pq.empty or len(df_cmt_pq) != len(df_cmt_csv) or len(df_cmt_pq) != len(sample_cmt_df):
        raise RuntimeError(
            f"processed_comments row count mismatch (parquet={len(df_cmt_pq)}, csv={len(df_cmt_csv)}, sample={len(sample_cmt_df)})."
        )
    missing_cmt_cols = [c for c in UNIFIED_COMMENT_FIELDS if c not in df_cmt_pq.columns]
    if missing_cmt_cols:
        raise RuntimeError(f"processed_comments.parquet missing expected columns: {missing_cmt_cols}")
    if expected_counts and "comments_rows" in expected_counts and len(df_cmt_pq) != expected_counts["comments_rows"]:
        raise RuntimeError(
            f"processed_comments row count mismatch: expected {expected_counts['comments_rows']}, got {len(df_cmt_pq)}"
        )

    # 3. Verify processed_ab_tests
    df_ab_pq = pd.read_parquet(processed_dir / "processed_ab_tests.parquet")
    df_ab_csv = pd.read_csv(processed_dir / "processed_ab_tests.csv")
    sample_exp_df = pd.read_csv(root_dir / "data" / "sample" / "sample_experiments.csv")
    if df_ab_pq.empty or len(df_ab_pq) != len(df_ab_csv):
        raise RuntimeError(
            f"processed_ab_tests Parquet ({len(df_ab_pq)}) and CSV ({len(df_ab_csv)}) mismatch or empty."
        )
    missing_ab_cols = [c for c in UNIFIED_OBSERVATION_FIELDS if c not in df_ab_pq.columns]
    if missing_ab_cols:
        raise RuntimeError(f"processed_ab_tests.parquet missing expected columns: {missing_ab_cols}")
    unknown_exp_ids = set(df_ab_pq["experiment_id"].astype(str)) - set(sample_exp_df["experiment_id"].astype(str))
    if unknown_exp_ids:
        raise RuntimeError(f"processed_ab_tests references unknown experiment IDs: {sorted(unknown_exp_ids)}")
    if expected_counts and "ab_rows" in expected_counts and len(df_ab_pq) != expected_counts["ab_rows"]:
        raise RuntimeError(
            f"processed_ab_tests row count mismatch: expected {expected_counts['ab_rows']}, got {len(df_ab_pq)}"
        )

    # 4. Verify SQLite database
    with closing(sqlite3.connect(db_path)) as conn:
        db_objs = dict(conn.execute("SELECT name, type FROM sqlite_master").fetchall())
        if db_objs.get("posts") != "table":
            raise RuntimeError("SQLite database missing required 'posts' table.")
        if db_objs.get("latest_posts") != "view":
            raise RuntimeError("SQLite database missing required 'latest_posts' view.")
        db_posts_count = int(conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0])
        db_latest_count = int(conn.execute("SELECT COUNT(*) FROM latest_posts").fetchone()[0])
        if db_posts_count != len(df_posts_pq):
            raise RuntimeError(
                f"SQLite 'posts' count ({db_posts_count}) does not match processed_posts ({len(df_posts_pq)})."
            )
        if db_latest_count <= 0 or db_latest_count > db_posts_count:
            raise RuntimeError(f"Invalid SQLite 'latest_posts' row count: {db_latest_count}")

    # 5. Verify Virality model & metadata reload and predict
    with open(models_dir / "virality_predictor_metadata.json", "r", encoding="utf-8") as f:
        vir_meta = json.load(f)
    for key in ["model_type", "baseline_metrics", "candidate_metrics", "total_mature_posts", "empirical_residual_q95"]:
        if key not in vir_meta:
            raise RuntimeError(f"virality_predictor_metadata.json missing required key: '{key}'")

    predictor = ViralityPredictor().load_model_pipeline(
        models_dir / "virality_predictor.joblib",
        models_dir / "virality_predictor_metadata.json",
    )
    sample_draft = pd.DataFrame([{
        "platform": "instagram",
        "content_type": "Reel",
        "topic": "Social Anxiety",
        "hook_type": "Visual Hook",
        "duration_seconds": 30,
        "posting_hour": 18,
        "posting_day_of_week": "Sunday",
        "follower_count": 35000,
    }])
    vir_pred = predictor.predict_pre_publication(sample_draft)
    if vir_pred.empty or pd.isna(vir_pred.iloc[0]["predicted_engagement_virality_index"]):
        raise RuntimeError("Reloaded virality predictor returned empty or NaN smoke prediction.")

    # 6. Verify Relatability NLP model & metadata reload and predict
    with open(models_dir / "relatability_metadata.json", "r", encoding="utf-8") as f:
        nlp_meta = json.load(f)
    for key in ["model_version", "models_benchmarked", "strictly_unique_text_benchmark"]:
        if key not in nlp_meta:
            raise RuntimeError(f"relatability_metadata.json missing required key: '{key}'")

    clf = RelatabilityClassifier(model_path=models_dir / "relatability_classifier.joblib")
    if clf.pipeline is None:
        raise RuntimeError("Reloaded RelatabilityClassifier has pipeline=None.")
    rel_smoke = clf.classify_comment("Literally me every single morning.")
    neu_smoke = clf.classify_comment("What camera did you use to film this video?")
    if rel_smoke["relatability_label"] != "Relatable" or neu_smoke["relatability_label"] != "Neutral":
        raise RuntimeError(
            f"Reloaded NLP classifier smoke check failed: rel={rel_smoke['relatability_label']}, neu={neu_smoke['relatability_label']}"
        )

    return {
        "processed_posts_rows": len(df_posts_pq),
        "processed_comments_rows": len(df_cmt_pq),
        "processed_ab_tests_rows": len(df_ab_pq),
        "sqlite_posts_rows": db_posts_count,
        "sqlite_latest_posts_rows": db_latest_count,
        "virality_mature_posts": vir_meta["total_mature_posts"],
        "nlp_model_version": nlp_meta["model_version"],
    }


def build_all_artifacts(
    root_dir: Optional[Path] = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    """Build (or validate) all SocialPulse derived data, SQLite, and model artifacts deterministically."""
    target_root = Path(root_dir).resolve() if root_dir is not None else ROOT_DIR

    if validate_only:
        input_summary = validate_source_inputs(target_root)
        verification = verify_generated_artifacts(target_root)
        return {
            "mode": "validate_only",
            "root_dir": str(target_root),
            "inputs": input_summary,
            "verification": verification,
        }

    with get_build_lock(target_root):
        # Fail-fast pre-flight validation BEFORE mutating any output artifacts
        input_summary = validate_source_inputs(target_root)

        # Stage 1: Module 1 Ingestion, Cleaning & Feature Engineering -> data/processed/processed_posts.{parquet,csv}
        sample_dir = target_root / "data" / "sample"
        df_posts_raw, _ = SampleDataIngestion(sample_dir=sample_dir).load_sample_posts("sample_posts.csv")
        df_posts_clean = clean_posts_dataset(df_posts_raw)
        df_posts_feat = add_engagement_features(df_posts_clean)

        storage = DataStorageManager(
            raw_dir=target_root / "data" / "raw",
            processed_dir=target_root / "data" / "processed",
            db_path=target_root / "database" / "social_analytics.db",
        )
        posts_pq_path, posts_csv_path = storage.save_processed_dataset(df_posts_feat, base_name="processed_posts")

        # Stage 2: SQLite Warehouse Persistence (idempotent replace)
        sqlite_rows = storage.persist_to_sqlite(df_posts_feat, table_name="posts", if_exists="replace")

        # Stage 3: Module 2 Virality Prediction Model & Metadata
        predictor = ViralityPredictor()
        vir_meta = predictor.train_and_evaluate(
            df_posts_feat,
            model_type="ridge",
            test_ratio=0.2,
            min_elapsed_days=7.0,
        )
        vir_model_path = target_root / "models" / "virality_predictor.joblib"
        vir_meta_path = target_root / "models" / "virality_predictor_metadata.json"
        predictor.save_model_pipeline(vir_model_path, vir_meta_path, include_volatile_timestamps=False)

        # Stage 4: Module 3 NLP Classifier, Metadata & Processed Comments (read-only on sample_comments.csv)
        nlp_res = run_nlp_pipeline(base_dir=target_root, overwrite_sample=False, seed=42)

        # Stage 5: Module 4 A/B Test Observations (read-only on sample_experiments.csv)
        ab_res = generate_ab_fixtures(base_dir=target_root, overwrite_sample=False, seed=42)

        # Stage 5b: Persist truthful runtime build provenance separately in gitignored manifest
        sample_hashes = {
            csv_path.name: hashlib.sha256(csv_path.read_bytes()).hexdigest()
            for csv_path in sorted((target_root / "data" / "sample").glob("*.csv"))
        }
        provenance_path = target_root / "models" / "build_provenance.json"
        existing_provenance: Dict[str, Any] = {}
        if provenance_path.is_file():
            try:
                loaded_prov = json.loads(provenance_path.read_text(encoding="utf-8"))
                if isinstance(loaded_prov, dict):
                    existing_provenance = loaded_prov
            except (OSError, json.JSONDecodeError):
                existing_provenance = {}
        provenance_payload = {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "virality_model_trained_at_utc": vir_meta.get("trained_at"),
            "nlp_model_trained_at_utc": nlp_res.get("runtime_provenance", {}).get("trained_at"),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "source_dataset_sha256": sample_hashes,
            "notes": (
                "Runtime build provenance containing intentionally variable wall-clock timestamps. "
                "Excluded from git tracking to keep models/*_metadata.json and models/*_benchmarks.json deterministic."
            ),
        }
        for bench_key in ("performance_benchmarks_generated_at_utc", "real_data_benchmarks_generated_at_utc"):
            if bench_key in existing_provenance:
                provenance_payload[bench_key] = existing_provenance[bench_key]
        with open(provenance_path, "w", encoding="utf-8", newline="\n") as pf:
            json.dump(provenance_payload, pf, indent=2)
            pf.write("\n")

        # Stage 6: Strict Post-Build Verification
        verification = verify_generated_artifacts(
            target_root,
            expected_counts={
                "posts_rows": len(df_posts_feat),
                "comments_rows": nlp_res["comments_count"],
                "ab_rows": ab_res["observations_count"],
            },
        )

        return {
            "mode": "build",
            "root_dir": str(target_root),
            "inputs": input_summary,
            "outputs": {
                "processed_posts": [str(posts_pq_path), str(posts_csv_path)],
                "sqlite_db": str(storage.db_path),
                "sqlite_rows": sqlite_rows,
                "virality_model": str(vir_model_path),
                "virality_metadata": str(vir_meta_path),
                "virality_metrics": vir_meta["candidate_metrics"],
                "nlp_model": str(nlp_res["model_path"]),
                "nlp_metadata": str(nlp_res["metadata_path"]),
                "build_provenance": str(provenance_path),
                "processed_comments": [str(nlp_res["processed_parquet_path"]), str(nlp_res["processed_csv_path"])],
                "processed_ab_tests": [str(ab_res["obs_parquet_path"]), str(ab_res["obs_csv_path"])],
            },
            "verification": verification,
        }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify SocialPulse datasets, SQLite DB, and serialized models."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DIR,
        help="Target project root directory (defaults to repository root).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate source inputs and existing generated artifacts without rebuilding.",
    )
    args = parser.parse_args(argv)

    try:
        result = build_all_artifacts(root_dir=args.root, validate_only=args.validate_only)
    except Exception as exc:
        print(f"[ERROR] Artifact pipeline failed: {exc}", file=sys.stderr)
        return 1

    v = result["verification"]
    print(
        f"[OK] SocialPulse artifact pipeline completed (mode={result['mode']}) "
        f"| posts={v['processed_posts_rows']} (latest={v['sqlite_latest_posts_rows']}) "
        f"| comments={v['processed_comments_rows']} "
        f"| ab_obs={v['processed_ab_tests_rows']} "
        f"| mature_virality_posts={v['virality_mature_posts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
