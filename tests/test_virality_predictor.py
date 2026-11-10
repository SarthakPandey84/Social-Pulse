"""Tests for the Virality Prediction Engine."""

import pandas as pd
from src.modeling.virality_predictor import ViralityPredictor


def test_calculate_viral_coefficient():
    predictor = ViralityPredictor()
    df = pd.DataFrame([{
        "post_id": "P1",
        "views": 1000,
        "shares": 50,
        "saves": 40,
        "comments_count": 20,
        "likes": 100,
        "retention_rate": 0.60
    }])
    vc = predictor.calculate_viral_coefficient(df)
    assert len(vc) == 1
    assert vc.iloc[0] > 0


def test_rank_topics_by_virality():
    predictor = ViralityPredictor()
    df = pd.DataFrame([
        {"post_id": "P1", "topic": "Social Anxiety", "views": 1000, "shares": 100, "saves": 80, "comments_count": 30, "likes": 50, "retention_rate": 0.7},
        {"post_id": "P2", "topic": "Dating", "views": 1000, "shares": 10, "saves": 5, "comments_count": 5, "likes": 20, "retention_rate": 0.3}
    ])
    ranked = predictor.rank_topics_by_virality(df)
    assert len(ranked) == 2
    assert ranked.iloc[0]["topic"] == "Social Anxiety"

