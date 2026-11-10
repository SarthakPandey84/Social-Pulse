"""Shared Streamlit data-loading, artifact-caching, and resource-management layer.

Centralizes ``@st.cache_data`` and ``@st.cache_resource`` boundaries across the
SocialPulse UI so repeated widget interactions avoid redundant disk I/O,
model deserialization, and backend reconstruction while remaining strictly
fresh whenever underlying artifacts change on disk.

All ``src/`` modules remain completely decoupled from Streamlit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from scripts.build_all_artifacts import (
    REQUIRED_ARTIFACT_RELATIVE_PATHS,
    artifacts_present_and_nonempty,  # noqa: F401 - re-exported for streamlit_app.py bootstrap check
    build_all_artifacts,
    get_build_lock,
    verify_generated_artifacts,
)
from src.analytics.engagement_metrics import EngagementAnalytics
from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.modeling.trend_forecaster import TrendForecaster
from src.modeling.virality_predictor import ViralityPredictor
from src.nlp.relatability_classifier import RelatabilityClassifier
from src.recommender.engine import OptimizationRecommender
from src.recommender.schemas import RecommendationReport

DEFAULT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DEMO_DATASET_ID = "synthetic_demo"


class DatasetLoadError(RuntimeError):
    """Raised explicitly when a selected dataset is missing, incomplete, or corrupt (never masks via fallback)."""


_SYNTHETIC_DATASET_META: Dict[str, Any] = {
    "dataset_id": SYNTHETIC_DEMO_DATASET_ID,
    "display_name": "Synthetic Demo Benchmark (Offline IG + YT)",
    "provenance": "synthetic_data",
    "governance_tier": "SYNTHETIC_BENCHMARK",
    "platforms": ["instagram", "youtube"],
    "license_name": "Internal Synthetic Fixture",
    "license_status": "VERIFIED",
    "pinned_revision": "data/sample/sample_*.csv",
    "tables": ["posts", "comments", "experiments", "observations", "trends"],
    "row_counts": {"posts": 120, "comments": 500, "experiments": 4, "trends": 170},
    "proxy_metrics": {},
    "limitations": [
        "Offline synthetic benchmark fixtures demonstrating Modules 1–7 pipeline mechanics without live platform APIs.",
    ],
    "is_external": False,
}

FileSignature = Tuple[str, bool, int, int]
DataFrameSignature = Tuple[Tuple[str, ...], Tuple[int, int], int]
CallableSignature = Tuple[Any, ...]
_UNSET: Any = object()


def _callable_signature(fn: Any) -> CallableSignature:
    """Return a deterministic, address-recycling-safe fingerprint for a callable."""
    if fn is None:
        return ("None",)
    target = getattr(fn, "__func__", fn)
    code = getattr(target, "__code__", None)
    if code is not None:
        return (
            getattr(target, "__module__", ""),
            getattr(target, "__qualname__", ""),
            code.co_code,
            repr(code.co_consts),
        )
    return (
        getattr(target, "__module__", ""),
        getattr(target, "__qualname__", repr(target)),
    )


def _resolve_root(root_dir: Optional[Union[str, Path]] = None) -> Path:
    return Path(root_dir).resolve() if root_dir is not None else DEFAULT_ROOT_DIR


def file_signature(path: Optional[Union[str, Path]]) -> FileSignature:
    """Return ``(resolved_path, exists, mtime_ns, size_bytes)`` for cache keying."""
    if path is None:
        return ("", False, 0, 0)
    resolved = Path(path).resolve()
    try:
        stat_res = resolved.stat()
    except OSError:
        return (str(resolved), False, 0, 0)
    return (str(resolved), True, int(stat_res.st_mtime_ns), int(stat_res.st_size))


def _dataframe_signature(df: Optional[pd.DataFrame]) -> DataFrameSignature:
    """Compute a deterministic structural + content hash signature for in-memory DataFrames.

    Uses full element-wise hashing up to 10,000 rows and a fast O(N) vectorized
    numeric + boundary-sample fingerprint above 10,000 rows so repeated Streamlit
    reruns remain sub-15ms even on 200K+ external dataset panels while still
    invalidating on structural, sample, null-pattern, or numeric cell edits.
    """
    if df is None:
        return ((), (0, 0), 0)
    cols = tuple(str(c) for c in df.columns)
    shape = (int(df.shape[0]), int(df.shape[1]))
    if df.empty:
        return (cols, shape, 0)
    if len(df) <= 10_000:
        try:
            hashed = pd.util.hash_pandas_object(df, index=True)
            digest = int(hashed.sum())
        except TypeError:
            digest = int(pd.util.hash_pandas_object(df.astype(str), index=True).sum())
        return (cols, shape, digest)

    # Fast exact numeric + sliced structural fingerprint for large (>10k row) DataFrames
    n = len(df)
    mid_start = max(0, (n // 2) - 1000)
    sample_df = pd.concat([df.iloc[:1500], df.iloc[mid_start : mid_start + 1500], df.iloc[-1500:]])
    try:
        sample_hash = int(pd.util.hash_pandas_object(sample_df, index=True).sum())
    except TypeError:
        sample_hash = int(pd.util.hash_pandas_object(sample_df.astype(str), index=True).sum())
    null_counts = tuple(int(c) for c in df.isna().sum().tolist())
    num_df = df.select_dtypes(include="number")
    num_sum = int(round(float(num_df.sum(numeric_only=True).fillna(0.0).sum()) * 1000.0)) if not num_df.empty else 0
    digest = int(
        hashlib.blake2b(
            f"{sample_hash}:{null_counts}:{num_sum}".encode("utf-8"),
            digest_size=8,
        ).hexdigest(),
        16,
    )
    return (cols, shape, digest)


# =========================================================================
# 1. TABULAR & JSON ARTIFACT LOADERS (@st.cache_data)
# =========================================================================


@st.cache_data(show_spinner=False)
def _read_parquet_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    reader_sig: CallableSignature,
) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@st.cache_data(show_spinner=False)
def _read_csv_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    reader_sig: CallableSignature,
) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _read_json_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _compute_home_kpis_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    reader_sig: CallableSignature,
) -> Optional[Dict[str, Any]]:
    df_posts = pd.read_csv(path_str)
    if df_posts.empty:
        return None
    return EngagementAnalytics.calculate_summary_kpis(df_posts)


def load_parquet_table(path: Optional[Union[str, Path]]) -> Optional[pd.DataFrame]:
    """Load a Parquet artifact with file-signature cache invalidation."""
    path_str, exists, mtime_ns, file_size = file_signature(path)
    if not exists:
        return None
    return _read_parquet_cached(path_str, mtime_ns, file_size, _callable_signature(pd.read_parquet))


def load_csv_table(path: Optional[Union[str, Path]]) -> Optional[pd.DataFrame]:
    """Load a CSV artifact with file-signature cache invalidation."""
    path_str, exists, mtime_ns, file_size = file_signature(path)
    if not exists:
        return None
    return _read_csv_cached(path_str, mtime_ns, file_size, _callable_signature(pd.read_csv))


def load_json_artifact(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load a JSON metadata artifact with file-signature cache invalidation."""
    path_str, exists, mtime_ns, file_size = file_signature(path)
    if not exists:
        return {}
    return _read_json_cached(path_str, mtime_ns, file_size)


def load_posts_dataset(
    processed_path: Any = _UNSET,
    sample_path: Any = _UNSET,
    root_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Load the unified posts corpus (Parquet primary, CSV sample fallback).

    Returns ``(df, source_kind)`` where ``source_kind`` is one of
    ``'processed'``, ``'sample'``, or ``'missing'``.
    """
    root = _resolve_root(root_dir)
    proc = root / "data" / "processed" / "processed_posts.parquet" if processed_path is _UNSET else processed_path
    samp = root / "data" / "sample" / "sample_posts.csv" if sample_path is _UNSET else sample_path

    df_proc = load_parquet_table(proc)
    if df_proc is not None:
        return df_proc, "processed"

    df_samp = load_csv_table(samp)
    if df_samp is not None:
        return df_samp, "sample"

    return None, "missing"


def load_comments_dataset(
    processed_path: Any = _UNSET,
    sample_path: Any = _UNSET,
    root_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Load the processed comments corpus (Parquet primary, CSV fallback)."""
    root = _resolve_root(root_dir)
    proc = root / "data" / "processed" / "processed_comments.parquet" if processed_path is _UNSET else processed_path
    samp = root / "data" / "sample" / "sample_comments.csv" if sample_path is _UNSET else sample_path

    df_proc = load_parquet_table(proc)
    if df_proc is not None:
        return df_proc, "processed"

    df_samp = load_csv_table(samp)
    if df_samp is not None:
        return df_samp, "sample"

    return None, "missing"


def load_experiments_dataset(
    exp_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """Load the A/B experiment metadata registry table."""
    target = (
        Path(exp_path)
        if exp_path is not None
        else _resolve_root(root_dir) / "data" / "sample" / "sample_experiments.csv"
    )
    return load_csv_table(target)


def load_ab_observations_dataset(
    obs_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """Load the processed A/B experiment post-level observations Parquet table."""
    target = (
        Path(obs_path)
        if obs_path is not None
        else _resolve_root(root_dir) / "data" / "processed" / "processed_ab_tests.parquet"
    )
    return load_parquet_table(target)


def load_trends_dataset(
    trends_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """Load the struggle-keyword trend trajectory dataset."""
    target = (
        Path(trends_path)
        if trends_path is not None
        else _resolve_root(root_dir) / "data" / "sample" / "sample_trends.csv"
    )
    return load_csv_table(target)


def load_home_summary_kpis(
    sample_posts_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute and cache Home Hub summary KPIs keyed by sample CSV signature."""
    target = (
        Path(sample_posts_path)
        if sample_posts_path is not None
        else _resolve_root(root_dir) / "data" / "sample" / "sample_posts.csv"
    )
    path_str, exists, mtime_ns, file_size = file_signature(target)
    if not exists:
        return None
    return _compute_home_kpis_cached(path_str, mtime_ns, file_size, _callable_signature(pd.read_csv))


# =========================================================================
# 1B. UNIFIED DATASET REGISTRY & FAIL-LOUD EXTERNAL PARQUET LOADERS
# =========================================================================


def _safe_dataset_dir(root: Path, dataset_id: str) -> Path:
    """Resolve `<root>/data/processed/external/<dataset_id>` safely without path traversal."""
    ds_id = str(dataset_id or "").strip()
    if not ds_id or "/" in ds_id or "\\" in ds_id or ".." in ds_id:
        raise DatasetLoadError(f"Invalid dataset identifier: {dataset_id!r}")
    base = (root / "data" / "processed" / "external").resolve()
    candidate = (base / ds_id).resolve()
    if candidate.parent != base:
        raise DatasetLoadError(f"Invalid external dataset directory path: {dataset_id!r}")
    return candidate


def _infer_tables_from_dir(ds_dir: Path, meta: Dict[str, Any]) -> List[str]:
    tables: List[str] = []
    primary = meta.get("primary_tables")
    if isinstance(primary, dict):
        tables.extend(str(k) for k in primary.keys())
    for fname, tname in (
        ("posts.parquet", "posts"),
        ("comments.parquet", "comments"),
        ("daily_platform_summary.parquet", "daily_platform_summary"),
        ("age_trajectory_curve.parquet", "age_trajectory_curve"),
    ):
        if (ds_dir / fname).is_file() and tname not in tables:
            tables.append(tname)
    return tables


def list_available_datasets(
    root_dir: Optional[Union[str, Path]] = None,
    required_table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover Synthetic Demo plus any prepared datasets under ``data/processed/external/``.

    Parameters
    ----------
    required_table : str, optional
        Filter to corpora supporting ``"posts"``, ``"comments"``, ``"cohorts"`` (for Module 4
        observational cohorts alongside synthetic RCT fixtures), or ``"daily_platform_summary"``.
    """
    root = _resolve_root(root_dir)
    entries: List[Dict[str, Any]] = []
    if required_table in (None, "posts", "comments", "cohorts", "experiments", "trends"):
        entries.append(dict(_SYNTHETIC_DATASET_META))

    ext_root = root / "data" / "processed" / "external"
    if ext_root.is_dir():
        for ds_dir in sorted(p for p in ext_root.iterdir() if p.is_dir()):
            meta_file = ds_dir / "dataset_meta.json"
            meta: Dict[str, Any] = {}
            meta_error: Optional[str] = None
            if meta_file.is_file():
                try:
                    meta = load_json_artifact(meta_file)
                except Exception as exc:
                    meta_error = f"Corrupt dataset_meta.json ({type(exc).__name__}: {exc})"

            tables = _infer_tables_from_dir(ds_dir, meta)
            if not tables and meta_error is None:
                continue

            item = {
                "dataset_id": ds_dir.name,
                "display_name": str(meta.get("display_name") or ds_dir.name),
                "provenance": str(meta.get("provenance") or "external_benchmark_unverified"),
                "governance_tier": str(meta.get("governance_tier") or "EXTERNAL_BENCHMARK_UNVERIFIED"),
                "platforms": list(meta.get("platforms") or ([meta["platform"]] if meta.get("platform") else [])),
                "license_name": str(meta.get("license_name") or "UNKNOWN"),
                "license_status": str(meta.get("license_status") or "PENDING_REVIEW"),
                "pinned_revision": str(meta.get("pinned_revision") or "unknown"),
                "tables": tables,
                "row_counts": dict(meta.get("row_counts") or {}),
                "proxy_metrics": dict(meta.get("proxy_metrics") or {}),
                "limitations": list(meta.get("limitations") or []),
                "recommended_modules": list(meta.get("recommended_modules") or []),
                "metric_availability": dict(meta.get("metric_availability") or {}),
                "is_external": True,
                "meta_error": meta_error,
            }

            if required_table is None:
                entries.append(item)
            elif required_table == "cohorts":
                if (
                    "posts" in tables
                    and (
                        "M4" in item["recommended_modules"]
                        or ds_dir.name in {"zenodo_sponsored_unsponsored_ig", "instaart_human_ai"}
                    )
                ):
                    entries.append(item)
            elif required_table in tables:
                entries.append(item)

    return entries


def _load_external_meta(ds_dir: Path, dataset_id: str) -> Dict[str, Any]:
    if not ds_dir.is_dir():
        raise DatasetLoadError(
            f"Selected external dataset '{dataset_id}' was not found at '{ds_dir}'. "
            "Run `scripts/download_external_datasets.py` and `scripts/prepare_external_datasets.py` first."
        )
    meta_file = ds_dir / "dataset_meta.json"
    meta: Dict[str, Any] = {}
    if meta_file.is_file():
        try:
            with open(meta_file, "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            if not isinstance(parsed, dict):
                raise ValueError("dataset_meta.json root must be a JSON object.")
            meta = parsed
        except Exception as exc:
            raise DatasetLoadError(
                f"Corrupt metadata file for external dataset '{dataset_id}' ({type(exc).__name__}: {exc})."
            ) from exc
    meta.setdefault("dataset_id", dataset_id)
    meta.setdefault("display_name", dataset_id)
    meta.setdefault("provenance", "external_benchmark_unverified")
    meta.setdefault("governance_tier", "EXTERNAL_BENCHMARK_UNVERIFIED")
    meta.setdefault("platforms", [meta["platform"]] if meta.get("platform") else [])
    meta.setdefault("tables", _infer_tables_from_dir(ds_dir, meta))
    meta["is_external"] = True
    return meta


@st.cache_data(show_spinner=False)
def _read_parquet_projected_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    cols_key: Tuple[str, ...],
    row_limit: Optional[int],
    reader_sig: CallableSignature,
) -> pd.DataFrame:
    df = pd.read_parquet(path_str, columns=list(cols_key) if cols_key else None)
    if row_limit is not None and row_limit > 0 and len(df) > row_limit:
        df = df.iloc[:row_limit].copy()
    return df


def _read_parquet_fail_loud(
    pq_path: Path,
    dataset_id: str,
    table_label: str,
    columns: Optional[Sequence[str]] = None,
    row_limit: Optional[int] = None,
) -> pd.DataFrame:
    path_str, exists, mtime_ns, file_size = file_signature(pq_path)
    if not exists or file_size <= 0:
        raise DatasetLoadError(
            f"External dataset '{dataset_id}' is missing required table '{table_label}' ({pq_path.name})."
        )
    try:
        cols_key: Tuple[str, ...] = ()
        if columns is not None:
            available_names = set(pq.read_schema(path_str).names)
            selected = [str(c) for c in columns if str(c) in available_names]
            cols_key = tuple(selected)
        return _read_parquet_projected_cached(
            path_str,
            mtime_ns,
            file_size,
            cols_key,
            row_limit,
            _callable_signature(pd.read_parquet),
        ).copy()
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to read '{table_label}' for external dataset '{dataset_id}': {type(exc).__name__}: {exc}"
        ) from exc


def load_selected_posts_dataset(
    dataset_id: str = SYNTHETIC_DEMO_DATASET_ID,
    root_dir: Optional[Union[str, Path]] = None,
    columns: Optional[Sequence[str]] = None,
    row_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the posts table for ``dataset_id`` strictly (never silently falls back on external errors)."""
    root = _resolve_root(root_dir)
    ds_id = str(dataset_id or SYNTHETIC_DEMO_DATASET_ID).strip()
    if ds_id == SYNTHETIC_DEMO_DATASET_ID:
        df, source_kind = load_posts_dataset(root_dir=root)
        if df is None:
            raise DatasetLoadError("Baseline synthetic posts dataset not found in data/processed/ or data/sample/.")
        out = df.copy()
        if columns is not None:
            keep = [c for c in columns if c in out.columns]
            if keep:
                out = out[keep].copy()
        meta = dict(_SYNTHETIC_DATASET_META)
        meta["source_kind"] = source_kind
        return out, meta

    ds_dir = _safe_dataset_dir(root, ds_id)
    meta = _load_external_meta(ds_dir, ds_id)
    df = _read_parquet_fail_loud(
        ds_dir / "posts.parquet",
        dataset_id=ds_id,
        table_label="posts",
        columns=columns,
        row_limit=row_limit,
    )
    meta["source_kind"] = "external_parquet"
    return df, meta


@st.cache_data(show_spinner=False)
def _classify_external_comments_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    model_sig: FileSignature,
    pipeline_sig: CallableSignature,
) -> pd.DataFrame:
    df_raw = pd.read_parquet(path_str)
    required_nlp = {"classification_status", "relatability_label", "sentiment_label"}
    if required_nlp.issubset(df_raw.columns):
        return df_raw
    classifier = RelatabilityClassifier(model_path=model_sig[0] if model_sig[1] else None)
    lang_col = (
        "language"
        if "language" in df_raw.columns
        else ("language_hint" if "language_hint" in df_raw.columns else None)
    )
    scored = classifier.process_comments_pipeline(df_raw, language_col=lang_col)
    # Preserve useful external metadata columns (e.g., source ground_truth_sentiment, language_hint, account_group)
    for extra_col in ("ground_truth_sentiment", "language_hint", "account_group", "provenance", "source_dataset", "eval_split"):
        if extra_col in df_raw.columns and extra_col not in scored.columns:
            scored[extra_col] = df_raw[extra_col].values
    return scored


def load_selected_comments_dataset(
    dataset_id: str = SYNTHETIC_DEMO_DATASET_ID,
    root_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and (if raw external) classify comments for ``dataset_id`` cleanly with no silent fallback."""
    root = _resolve_root(root_dir)
    ds_id = str(dataset_id or SYNTHETIC_DEMO_DATASET_ID).strip()
    if ds_id == SYNTHETIC_DEMO_DATASET_ID:
        df, source_kind = load_comments_dataset(root_dir=root)
        if df is None:
            raise DatasetLoadError("Baseline synthetic comments dataset not found in data/processed/ or data/sample/.")
        meta = dict(_SYNTHETIC_DATASET_META)
        meta["source_kind"] = source_kind
        return df, meta

    ds_dir = _safe_dataset_dir(root, ds_id)
    meta = _load_external_meta(ds_dir, ds_id)
    pq_path = ds_dir / "comments.parquet"
    path_str, exists, mtime_ns, file_size = file_signature(pq_path)
    if not exists or file_size <= 0:
        raise DatasetLoadError(f"External dataset '{ds_id}' is missing 'comments.parquet'.")
    model_sig = file_signature(root / "models" / "relatability_classifier.joblib")
    try:
        df_scored = _classify_external_comments_cached(
            path_str,
            mtime_ns,
            file_size,
            model_sig,
            _callable_signature(RelatabilityClassifier.process_comments_pipeline),
        ).copy()
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to load/classify comments for external dataset '{ds_id}': {type(exc).__name__}: {exc}"
        ) from exc
    meta["source_kind"] = "external_parquet"
    return df_scored, meta


def load_selected_daily_summary_dataset(
    dataset_id: str = SYNTHETIC_DEMO_DATASET_ID,
    root_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Load pre-aggregated ``daily_platform_summary.parquet`` for external panel datasets when available."""
    root = _resolve_root(root_dir)
    ds_id = str(dataset_id or SYNTHETIC_DEMO_DATASET_ID).strip()
    if ds_id == SYNTHETIC_DEMO_DATASET_ID:
        return None, dict(_SYNTHETIC_DATASET_META)

    ds_dir = _safe_dataset_dir(root, ds_id)
    meta = _load_external_meta(ds_dir, ds_id)
    daily_path = ds_dir / "daily_platform_summary.parquet"
    expects_daily = "daily_platform_summary" in meta.get("tables", [])
    if not daily_path.is_file():
        if expects_daily:
            raise DatasetLoadError(
                f"External dataset '{ds_id}' declares 'daily_platform_summary' but '{daily_path.name}' is missing."
            )
        return None, meta
    df_daily = _read_parquet_fail_loud(
        daily_path,
        dataset_id=ds_id,
        table_label="daily_platform_summary",
    )
    return df_daily, meta


# =========================================================================
# 2. MODEL & BACKEND RESOURCE LOADERS (@st.cache_resource)
# =========================================================================


@st.cache_resource(show_spinner=False)
def _build_virality_predictor(
    model_sig: FileSignature,
    meta_sig: FileSignature,
    cls_sig: CallableSignature,
    load_method_sig: CallableSignature,
) -> ViralityPredictor:
    predictor = ViralityPredictor()
    predictor.load_model_pipeline(model_sig[0], meta_sig[0])
    return predictor


@st.cache_resource(show_spinner=False)
def _build_virality_predictor_from_fallback(
    root_str: str,
    model_sig: FileSignature,
    meta_sig: FileSignature,
    fallback_file_sig: FileSignature,
    fallback_df_sig: DataFrameSignature,
    cls_sig: CallableSignature,
    save_to_disk: bool,
    _fallback_posts_df: pd.DataFrame,
) -> ViralityPredictor:
    predictor = ViralityPredictor()
    predictor.train_and_evaluate(_fallback_posts_df, model_type="ridge", test_ratio=0.2, min_elapsed_days=7.0)
    if save_to_disk and model_sig[0] and meta_sig[0]:
        with get_build_lock(root_str):
            predictor.save_model_pipeline(
                model_sig[0],
                meta_sig[0],
                include_volatile_timestamps=False,
            )
    return predictor


def get_virality_predictor(
    model_path: Optional[Union[str, Path]] = None,
    meta_path: Optional[Union[str, Path]] = None,
    fallback_posts_df: Optional[pd.DataFrame] = None,
    fallback_posts_path: Optional[Union[str, Path]] = None,
    save_if_trained: bool = True,
    root_dir: Optional[Union[str, Path]] = None,
) -> ViralityPredictor:
    """Return a cached fitted ``ViralityPredictor``.

    When serialized model/metadata artifacts exist on disk, caches by their
    file signatures. When they are absent and ``fallback_posts_df`` is provided,
    caches fallback training keyed on both ``fallback_posts_path`` signature and
    ``fallback_posts_df`` content signature so changes to fallback data
    invalidate the resource cleanly even if artifacts remain absent on disk.
    """
    root = _resolve_root(root_dir)
    m_path = Path(model_path) if model_path is not None else root / "models" / "virality_predictor.joblib"
    j_path = Path(meta_path) if meta_path is not None else root / "models" / "virality_predictor_metadata.json"

    model_sig = file_signature(m_path)
    meta_sig = file_signature(j_path)
    if model_sig[1] and meta_sig[1]:
        return _build_virality_predictor(
            model_sig,
            meta_sig,
            _callable_signature(ViralityPredictor),
            _callable_signature(ViralityPredictor.load_model_pipeline),
        )

    if fallback_posts_df is not None and not fallback_posts_df.empty:
        lock_root = str(Path(model_sig[0]).resolve().parent.parent) if model_path is not None else str(root)
        return _build_virality_predictor_from_fallback(
            lock_root,
            model_sig,
            meta_sig,
            file_signature(fallback_posts_path),
            _dataframe_signature(fallback_posts_df),
            _callable_signature(ViralityPredictor),
            bool(save_if_trained),
            fallback_posts_df,
        )

    return ViralityPredictor()


@st.cache_resource(show_spinner=False)
def _build_relatability_classifier(
    model_sig: FileSignature,
    cls_sig: CallableSignature,
) -> RelatabilityClassifier:
    model_arg = model_sig[0] if model_sig[1] else None
    return RelatabilityClassifier(model_path=model_arg)


def get_relatability_classifier(
    model_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> RelatabilityClassifier:
    """Return a cached ``RelatabilityClassifier`` invalidated when the model artifact changes."""
    root = _resolve_root(root_dir)
    m_path = Path(model_path) if model_path is not None else root / "models" / "relatability_classifier.joblib"
    return _build_relatability_classifier(file_signature(m_path), _callable_signature(RelatabilityClassifier))


@st.cache_resource(show_spinner=False)
def _build_trend_forecaster(rolling_window: int, cls_sig: CallableSignature) -> TrendForecaster:
    return TrendForecaster(rolling_window=rolling_window)


def get_trend_forecaster(rolling_window: int = 7) -> TrendForecaster:
    """Return a cached ``TrendForecaster`` configured with the requested rolling window."""
    return _build_trend_forecaster(int(rolling_window), _callable_signature(TrendForecaster))


@st.cache_resource(show_spinner=False)
def _build_optimization_recommender(
    posts_sig: FileSignature,
    comments_sig: FileSignature,
    exp_sig: FileSignature,
    obs_sig: FileSignature,
    df_sigs: Tuple[DataFrameSignature, DataFrameSignature, DataFrameSignature, DataFrameSignature],
    alpha: float,
    cls_sig: CallableSignature,
    _posts_df: Optional[pd.DataFrame],
    _comments_df: Optional[pd.DataFrame],
    _experiments_df: Optional[pd.DataFrame],
    _observations_df: Optional[pd.DataFrame],
) -> OptimizationRecommender:
    return OptimizationRecommender(
        posts_df=_posts_df,
        comments_df=_comments_df,
        experiments_df=_experiments_df,
        observations_df=_observations_df,
        alpha=alpha,
    )


def get_optimization_recommender(
    posts_df: Optional[pd.DataFrame] = None,
    comments_df: Optional[pd.DataFrame] = None,
    experiments_df: Optional[pd.DataFrame] = None,
    observations_df: Optional[pd.DataFrame] = None,
    *,
    posts_path: Optional[Union[str, Path]] = None,
    comments_path: Optional[Union[str, Path]] = None,
    exp_path: Optional[Union[str, Path]] = None,
    obs_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.05,
    root_dir: Optional[Union[str, Path]] = None,
) -> OptimizationRecommender:
    """Return a cached ``OptimizationRecommender`` keyed by dataset signatures and ``alpha``."""
    root = _resolve_root(root_dir)
    p_sig = file_signature(posts_path if posts_path is not None else root / "data" / "processed" / "processed_posts.parquet")
    c_sig = file_signature(comments_path if comments_path is not None else root / "data" / "processed" / "processed_comments.parquet")
    e_sig = file_signature(exp_path if exp_path is not None else root / "data" / "sample" / "sample_experiments.csv")
    o_sig = file_signature(obs_path if obs_path is not None else root / "data" / "processed" / "processed_ab_tests.parquet")
    df_sigs = (
        _dataframe_signature(posts_df) if posts_df is not None else (),
        _dataframe_signature(comments_df) if comments_df is not None else (),
        _dataframe_signature(experiments_df) if experiments_df is not None else (),
        _dataframe_signature(observations_df) if observations_df is not None else (),
    )
    return _build_optimization_recommender(
        p_sig,
        c_sig,
        e_sig,
        o_sig,
        df_sigs,
        float(alpha),
        _callable_signature(OptimizationRecommender),
        posts_df,
        comments_df,
        experiments_df,
        observations_df,
    )


@st.cache_resource(show_spinner=False)
def _build_growth_dashboard_backend(
    root_str: str,
    dataset_id: str,
    posts_sig: FileSignature,
    sample_posts_sig: FileSignature,
    comments_sig: FileSignature,
    exp_sig: FileSignature,
    obs_sig: FileSignature,
    ext_meta_sig: FileSignature,
    ext_daily_sig: FileSignature,
    reader_sigs: Tuple[CallableSignature, CallableSignature],
    cls_sig: CallableSignature,
) -> GrowthDashboardBackend:
    try:
        backend = GrowthDashboardBackend(root_dir=root_str, dataset_id=dataset_id)
        if (
            dataset_id != SYNTHETIC_DEMO_DATASET_ID
            and backend.comments_df is not None
            and not backend.comments_df.empty
            and (
                "sentiment_label" not in backend.comments_df.columns
                or "relatability_label" not in backend.comments_df.columns
            )
        ):
            comments_path = Path(root_str) / "data" / "processed" / "external" / dataset_id / "comments.parquet"
            c_sig = file_signature(comments_path)
            model_sig = file_signature(Path(root_str) / "models" / "relatability_classifier.joblib")
            if c_sig[1]:
                backend.comments_df = _classify_external_comments_cached(
                    c_sig[0],
                    c_sig[2],
                    c_sig[3],
                    model_sig,
                    _callable_signature(RelatabilityClassifier.process_comments_pipeline),
                ).copy()
        return backend
    except Exception as exc:
        if dataset_id != SYNTHETIC_DEMO_DATASET_ID:
            raise DatasetLoadError(
                f"Failed to initialize GrowthDashboardBackend for '{dataset_id}': {type(exc).__name__}: {exc}"
            ) from exc
        raise


def get_growth_dashboard_backend(
    root_dir: Optional[Union[str, Path]] = None,
    dataset_id: str = SYNTHETIC_DEMO_DATASET_ID,
) -> GrowthDashboardBackend:
    """Return a cached ``GrowthDashboardBackend`` keyed by workspace path, ``dataset_id``, and dataset signatures."""
    root = _resolve_root(root_dir)
    ds_id = str(dataset_id or SYNTHETIC_DEMO_DATASET_ID).strip()
    if ds_id == SYNTHETIC_DEMO_DATASET_ID:
        return _build_growth_dashboard_backend(
            str(root),
            ds_id,
            file_signature(root / "data" / "processed" / "processed_posts.parquet"),
            file_signature(root / "data" / "sample" / "sample_posts.csv"),
            file_signature(root / "data" / "processed" / "processed_comments.parquet"),
            file_signature(root / "data" / "sample" / "sample_experiments.csv"),
            file_signature(root / "data" / "processed" / "processed_ab_tests.parquet"),
            ("", False, 0, 0),
            ("", False, 0, 0),
            (_callable_signature(pd.read_parquet), _callable_signature(pd.read_csv)),
            _callable_signature(GrowthDashboardBackend),
        )

    ext_dir = _safe_dataset_dir(root, ds_id)
    _ = _load_external_meta(ext_dir, ds_id)
    ext_posts = ext_dir / "posts.parquet"
    if not ext_posts.is_file() or ext_posts.stat().st_size <= 0:
        raise DatasetLoadError(f"External dataset '{ds_id}' is missing a valid posts.parquet file.")
    return _build_growth_dashboard_backend(
        str(root),
        ds_id,
        file_signature(ext_posts),
        ("", False, 0, 0),
        file_signature(ext_dir / "comments.parquet"),
        ("", False, 0, 0),
        ("", False, 0, 0),
        file_signature(ext_dir / "dataset_meta.json"),
        file_signature(ext_dir / "daily_platform_summary.parquet"),
        (_callable_signature(pd.read_parquet), _callable_signature(pd.read_csv)),
        _callable_signature(GrowthDashboardBackend),
    )


# =========================================================================
# 3. DASHBOARD UNFILTERED CROSS-MODULE SYNTHESIS CACHES (@st.cache_data)
# =========================================================================


def _dashboard_backend_signature(backend: GrowthDashboardBackend) -> Tuple[Any, ...]:
    """Return a deterministic, address-recycling-safe signature for a dashboard backend.

    Combines resolved ``backend.root_dir``, ``backend.dataset_id``, and file signatures for
    all backing source datasets (synthetic workspace OR external Parquet bundle). When a source
    file is absent on disk, includes full structural + content DataFrame signatures via
    ``_dataframe_signature()``.
    """
    root = Path(backend.root_dir).resolve()
    ds_id = str(getattr(backend, "dataset_id", SYNTHETIC_DEMO_DATASET_ID) or SYNTHETIC_DEMO_DATASET_ID)
    if ds_id == SYNTHETIC_DEMO_DATASET_ID:
        file_sigs: Tuple[FileSignature, ...] = (
            file_signature(root / "data" / "processed" / "processed_posts.parquet"),
            file_signature(root / "data" / "sample" / "sample_posts.csv"),
            file_signature(root / "data" / "processed" / "processed_comments.parquet"),
            file_signature(root / "data" / "sample" / "sample_experiments.csv"),
            file_signature(root / "data" / "processed" / "processed_ab_tests.parquet"),
        )
    else:
        ext_dir = root / "data" / "processed" / "external" / ds_id
        file_sigs = (
            file_signature(ext_dir / "posts.parquet"),
            file_signature(ext_dir / "comments.parquet"),
            file_signature(ext_dir / "daily_platform_summary.parquet"),
            file_signature(ext_dir / "dataset_meta.json"),
            ("", False, 0, 0),
        )

    posts_df = getattr(backend, "posts_df", None)
    comments_df = getattr(backend, "comments_df", None)
    experiments_df = getattr(backend, "experiments_df", None)
    observations_df = getattr(backend, "observations_df", None)
    in_memory_sigs = (
        _dataframe_signature(posts_df) if posts_df is not None else (),
        _dataframe_signature(comments_df) if comments_df is not None else (),
        _dataframe_signature(experiments_df) if experiments_df is not None else (),
        _dataframe_signature(observations_df) if observations_df is not None else (),
    )

    return (
        str(root),
        ds_id,
        *file_sigs,
        in_memory_sigs,
    )


@st.cache_data(show_spinner=False)
def _compute_dashboard_exp_perf_cached(
    backend_sig: Tuple[Any, ...],
    method_sig: CallableSignature,
    _backend: GrowthDashboardBackend,
) -> Dict[str, Any]:
    return _backend.calculate_experiment_performance()


def get_dashboard_experiment_performance(backend: GrowthDashboardBackend) -> Dict[str, Any]:
    """Cache unfiltered Module 4 experiment synthesis for Page 06 across sidebar filter reruns."""
    method = getattr(backend, "calculate_experiment_performance", None)
    return _compute_dashboard_exp_perf_cached(
        _dashboard_backend_signature(backend),
        _callable_signature(method),
        backend,
    )


@st.cache_data(show_spinner=False)
def _compute_dashboard_recs_cached(
    backend_sig: Tuple[Any, ...],
    method_sig: CallableSignature,
    _backend: GrowthDashboardBackend,
) -> RecommendationReport:
    return _backend.calculate_recommendation_priorities()


def get_dashboard_recommendation_priorities(backend: GrowthDashboardBackend) -> RecommendationReport:
    """Cache unfiltered Module 5 recommendation synthesis for Page 06 across sidebar filter reruns."""
    method = getattr(backend, "calculate_recommendation_priorities", None)
    return _compute_dashboard_recs_cached(
        _dashboard_backend_signature(backend),
        _callable_signature(method),
        backend,
    )


# =========================================================================
# 4. CLEAN-CHECKOUT STARTUP ARTIFACT BOOTSTRAP
# =========================================================================

_ACTIVE_BOOTSTRAP_GUARD = threading.Lock()
_ACTIVE_BOOTSTRAP_ROOTS: Set[Path] = set()


def _artifact_set_signature(target_root: Path) -> Tuple[FileSignature, ...]:
    """Return ordered file signatures covering every required generated artifact."""
    return tuple(
        file_signature(target_root / rel_path)
        for rel_path in REQUIRED_ARTIFACT_RELATIVE_PATHS
    )


@st.cache_resource(show_spinner=False)
def _verify_artifacts_cached(
    root_str: str,
    artifact_sigs: Tuple[FileSignature, ...],
    verifier_sig: CallableSignature,
) -> Dict[str, Any]:
    """Verify generated artifacts once per exact root path and artifact-signature set."""
    return verify_generated_artifacts(Path(root_str))


def _try_verify_existing_artifacts(target_root: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Fast pre-check: return (True, verification) iff all required artifacts exist and pass validation."""
    with _ACTIVE_BOOTSTRAP_GUARD:
        if target_root in _ACTIVE_BOOTSTRAP_ROOTS:
            return False, None

    artifact_sigs = _artifact_set_signature(target_root)
    if not all(sig[1] and sig[3] > 0 for sig in artifact_sigs):
        return False, None

    try:
        verification = _verify_artifacts_cached(
            str(target_root),
            artifact_sigs,
            _callable_signature(verify_generated_artifacts),
        )
        return True, verification
    except Exception:
        return False, None


def ensure_artifacts_ready(root_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Ensure all required processed datasets, SQLite store, and models are present and valid.

    Uses a lock-free fast path keyed by workspace root and artifact file signatures
    after startup verification, and a double-checked process-wide re-entrant lock
    around ``build_all_artifacts()`` so concurrent Streamlit sessions never trigger
    overlapping builds. Failed builds are never cached or masked.
    """
    target_root = _resolve_root(root_dir)

    ready, verification = _try_verify_existing_artifacts(target_root)
    if ready:
        return {
            "status": "ready",
            "built": False,
            "root_dir": str(target_root),
            "verification": verification,
        }

    lock = get_build_lock(target_root)
    with lock:
        ready, verification = _try_verify_existing_artifacts(target_root)
        if ready:
            return {
                "status": "ready",
                "built": False,
                "root_dir": str(target_root),
                "verification": verification,
            }

        with _ACTIVE_BOOTSTRAP_GUARD:
            _ACTIVE_BOOTSTRAP_ROOTS.add(target_root)
        try:
            build_result = build_all_artifacts(root_dir=target_root)
            post_sigs = _artifact_set_signature(target_root)
            if not all(sig[1] and sig[3] > 0 for sig in post_sigs):
                raise RuntimeError(
                    f"Artifact build for '{target_root}' did not produce all required output files."
                )
            verification = _verify_artifacts_cached(
                str(target_root),
                post_sigs,
                _callable_signature(verify_generated_artifacts),
            )
            return {
                "status": "built",
                "built": True,
                "root_dir": str(target_root),
                "verification": verification,
                "build_mode": build_result.get("mode", "build"),
            }
        finally:
            with _ACTIVE_BOOTSTRAP_GUARD:
                _ACTIVE_BOOTSTRAP_ROOTS.discard(target_root)
