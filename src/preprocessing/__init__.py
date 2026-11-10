"""Preprocessing module for text cleaning, snapshot views, quality pipeline, and feature engineering."""

from .cleaning import clean_text, clean_posts_dataset, get_latest_posts_view
from .feature_engineering import add_engagement_features
from .quality_pipeline import (
    LEGACY_PUBLIC_SALT,
    PII_HMAC_KEY_ENV,
    CleaningConfig,
    fix_mojibake_and_whitespace,
    parse_compact_count_series,
    pseudonymize_identifier,
    redact_pii_text,
    resolve_pseudonymization_mode,
    run_cleaning_pipeline,
    save_clean_artifacts,
)

__all__ = [
    "clean_text",
    "clean_posts_dataset",
    "get_latest_posts_view",
    "add_engagement_features",
    "LEGACY_PUBLIC_SALT",
    "PII_HMAC_KEY_ENV",
    "CleaningConfig",
    "fix_mojibake_and_whitespace",
    "parse_compact_count_series",
    "pseudonymize_identifier",
    "redact_pii_text",
    "resolve_pseudonymization_mode",
    "run_cleaning_pipeline",
    "save_clean_artifacts",
]
