"""Sample Dataset Ingestion Engine for Offline Development and Testing."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from .schemas import UNIFIED_POST_FIELDS, UNIFIED_PANDAS_DTYPES, DataProvenance
from utils.helpers import get_project_root


class SampleDataIngestion:
    """Loads, validates, and normalizes benchmark datasets without requiring API keys."""

    def __init__(self, sample_dir: Optional[Path] = None):
        self.sample_dir = sample_dir or (get_project_root() / "data" / "sample")

    def get_provenance(self) -> DataProvenance:
        return DataProvenance(
            source_name="Offline Sample Benchmark Dataset",
            access_tier="sample_fixture",
            notes="Curated multi-platform dataset for local development, tests, and demonstration."
        )

    def load_sample_posts(self, filename: str = "sample_posts.csv") -> Tuple[pd.DataFrame, DataProvenance]:
        """Load and normalize sample post data into the 14-field unified schema."""
        file_path = self.sample_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Sample file not found at: {file_path}")

        df = pd.read_csv(file_path)

        # Standardize missing unified columns
        now_iso = datetime.now(timezone.utc).isoformat()
        if "platform" not in df.columns:
            df["platform"] = "instagram"
        if "impressions" not in df.columns:
            # Derived from views if available, else null (vectorized null-safe nullable Int64 truncation)
            views_num = pd.to_numeric(df["views"], errors="coerce")
            df["impressions"] = pd.Series(
                np.trunc(views_num * 1.15), index=df.index, dtype="Float64"
            ).astype("Int64")
        if "reach" not in df.columns:
            df["reach"] = pd.NA
        if "follower_count" not in df.columns:
            df["follower_count"] = pd.NA
        if "collection_timestamp" not in df.columns:
            df["collection_timestamp"] = now_iso

        # Format datatypes
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"], utc=True)

        for col, dtype in UNIFIED_PANDAS_DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # Ensure all 14 columns are present in canonical order (preserving extra context columns like topic)
        canonical_cols = UNIFIED_POST_FIELDS
        extra_cols = [c for c in df.columns if c not in canonical_cols]
        df = df[canonical_cols + extra_cols]

        return df, self.get_provenance()
