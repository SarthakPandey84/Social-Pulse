"""Tests for Data Validation module."""

import pandas as pd
from src.data_ingestion.data_validator import DataValidator


def test_validate_posts_schema_valid():
    df = pd.DataFrame([{
        "post_id": "P1",
        "timestamp": "2026-10-01 12:00:00",
        "topic": "Social Anxiety",
        "views": 1000,
        "likes": 50,
        "comments_count": 10,
        "shares": 30,
        "saves": 40,
        "retention_rate": 0.65
    }])
    is_valid, errors = DataValidator.validate_posts_schema(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_posts_schema_invalid():
    df = pd.DataFrame([{
        "post_id": "P1",
        "views": -5,
        "retention_rate": 1.5
    }])
    is_valid, errors = DataValidator.validate_posts_schema(df)
    assert is_valid is False
    assert len(errors) > 0

