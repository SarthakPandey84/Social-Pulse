"""Data cleaning and tabular preprocessing pipelines.

Preserves historical snapshots across collection intervals.
Deduplication strictly targets identical snapshots on (platform, post_id, collection_timestamp).
Provides a dedicated view function get_latest_posts_view() for latest-state analysis.
Preserves missing-value semantics using pandas nullable types (Int64, Float64).
"""

from datetime import datetime, timezone
import re
from typing import Optional
import pandas as pd
from src.data_ingestion.schemas import UNIFIED_PANDAS_DTYPES


def clean_text(text: str) -> str:
    """Normalize text by stripping URLs, emojis, excessive spaces, and special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_posts_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean tabular posts dataset while preserving historical snapshots and nullable types.

    Preserves multiple snapshots of the same post over time (identified by collection_timestamp).
    Only eliminates exact duplicate extractions on (platform, post_id, collection_timestamp).
    """
    if df.empty:
        return df.copy()

    df_clean = df.copy()

    # 1. Standardize Timestamps to UTC
    if "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], utc=True, errors="coerce")

    if "collection_timestamp" in df_clean.columns:
        df_clean["collection_timestamp"] = pd.to_datetime(df_clean["collection_timestamp"], utc=True, errors="coerce")
    else:
        df_clean["collection_timestamp"] = datetime.now(timezone.utc)

    # 2. Deduplication of identical snapshots ONLY (idempotent re-ingestion guard)
    if "platform" in df_clean.columns and "post_id" in df_clean.columns and "collection_timestamp" in df_clean.columns:
        # Sort chronologically and drop only identical snapshots
        df_clean = df_clean.sort_values(by=["platform", "post_id", "collection_timestamp"]).reset_index(drop=True)
        df_clean = df_clean.drop_duplicates(
            subset=["platform", "post_id", "collection_timestamp"],
            keep="last"
        ).reset_index(drop=True)

    # 3. Cast numeric fields to nullable types WITHOUT filling with 0
    integer_fields = ["views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count"]
    for col in integer_fields:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype("Int64")

    # 4. Retention rate casting and boundary validation
    if "retention_rate" in df_clean.columns:
        ret_numeric = pd.to_numeric(df_clean["retention_rate"], errors="coerce")
        # Clip invalid out-of-bounds numbers while keeping true NaNs as NaN
        df_clean["retention_rate"] = ret_numeric.apply(
            lambda x: min(max(float(x), 0.0), 1.0) if pd.notna(x) else pd.NA
        ).astype("Float64")

    # 5. String type enforcement
    string_fields = ["platform", "post_id", "content_type"]
    for col in string_fields:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("string")

    return df_clean


def get_latest_posts_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered view containing strictly the latest observation per (platform, post_id).

    Preserves full snapshot history in the parent dataset while providing latest-state slices.
    """
    if df.empty or "platform" not in df.columns or "post_id" not in df.columns:
        return df.copy()

    df_sorted = df.copy()
    if "collection_timestamp" in df_sorted.columns:
        df_sorted["collection_timestamp"] = pd.to_datetime(df_sorted["collection_timestamp"], utc=True, format="ISO8601")
        df_sorted = df_sorted.sort_values(by="collection_timestamp", ascending=False)

    latest_view = df_sorted.drop_duplicates(subset=["platform", "post_id"], keep="first").reset_index(drop=True)
    return latest_view
