"""Data validation engine ensuring schema conformance, timestamp validity, and statistical integrity."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import pandas as pd
from .schemas import UNIFIED_POST_FIELDS, NULLABLE_SPECIFICATION


class DataValidator:
    """Validates raw and ingested datasets before processing."""

    REQUIRED_POST_COLUMNS = [
        "post_id", "timestamp", "topic", "views", "likes", 
        "comments_count", "shares", "saves", "retention_rate"
    ]

    REQUIRED_COMMENT_COLUMNS = [
        "post_id", "comment_id", "comment_text"
    ]

    @classmethod
    def validate_posts_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify presence of required columns and data ranges for legacy/post metrics."""
        errors = []
        for col in cls.REQUIRED_POST_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required post column: {col}")

        if not errors:
            numeric_cols = ["views", "likes", "comments_count", "shares", "saves"]
            for col in numeric_cols:
                # Range check only for non-null values
                non_null_vals = df[col].dropna()
                if (non_null_vals < 0).any():
                    errors.append(f"Column '{col}' contains negative numbers.")
            
            non_null_ret = df["retention_rate"].dropna()
            if ((non_null_ret < 0.0) | (non_null_ret > 1.0)).any():
                errors.append("Column 'retention_rate' contains values outside range [0, 1].")

        return (len(errors) == 0, errors)

    @classmethod
    def validate_comments_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify presence of required columns for comments dataset."""
        errors = []
        for col in cls.REQUIRED_COMMENT_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required comment column: {col}")
        return (len(errors) == 0, errors)

    @classmethod
    def validate_unified_posts(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform full 14-field unified validation with detailed error and nullability reporting."""
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Column Presence Check
        missing_cols = [c for c in UNIFIED_POST_FIELDS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required unified fields: {missing_cols}")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "total_rows": len(df),
                "null_distributions": {},
                "duplicate_snapshot_count": 0
            }

        # 2. Non-Nullable Field Check
        for col, is_nullable in NULLABLE_SPECIFICATION.items():
            if not is_nullable:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    errors.append(f"Mandatory non-nullable field '{col}' contains {null_count} nulls.")

        # 3. Numeric Bound Checks (Only evaluated on non-null values)
        numeric_counts = ["views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count"]
        for col in numeric_counts:
            if col in df.columns:
                non_null = df[col].dropna()
                if (non_null < 0).any():
                    errors.append(f"Metric '{col}' contains negative numbers.")

        # 4. Retention Rate Bounds [0.0, 1.0]
        if "retention_rate" in df.columns:
            non_null_ret = df["retention_rate"].dropna()
            if ((non_null_ret < 0.0) | (non_null_ret > 1.0)).any():
                errors.append("Column 'retention_rate' has values outside [0.0, 1.0].")

        # 5. Timestamp Parsing & Future Date Validation
        try:
            parsed_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            invalid_ts_count = parsed_ts.isna().sum()
            if invalid_ts_count > 0:
                errors.append(f"Timestamp column contains {invalid_ts_count} unparseable values.")
            else:
                # Allow max 24 hour clock skew into future
                now_utc = datetime.now(timezone.utc)
                future_dates = (parsed_ts > (now_utc + pd.Timedelta(days=1))).sum()
                if future_dates > 0:
                    errors.append(f"Timestamp column contains {future_dates} timestamps set in the future.")
        except Exception as e:
            errors.append(f"Timestamp validation failed with error: {str(e)}")

        # 6. Snapshot Identity & Deduplication Audit
        # Historical snapshots of the same (platform, post_id) are valid over time.
        # Exact duplicate snapshots on (platform, post_id, collection_timestamp) represent redundant extractions.
        if "platform" in df.columns and "post_id" in df.columns and "collection_timestamp" in df.columns:
            exact_dupe_mask = df.duplicated(subset=["platform", "post_id", "collection_timestamp"], keep=False)
            exact_dupe_count = int(exact_dupe_mask.sum())
            if exact_dupe_count > 0:
                warnings.append(f"Detected {exact_dupe_count} redundant identical snapshots on (platform, post_id, collection_timestamp).")
        else:
            exact_dupe_count = 0

        # 7. Null Distribution Summary
        null_dist = {col: int(df[col].isna().sum()) for col in UNIFIED_POST_FIELDS if col in df.columns}

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_rows": len(df),
            "null_distributions": null_dist,
            "duplicate_snapshot_count": exact_dupe_count
        }
