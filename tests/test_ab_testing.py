"""Tests for A/B Testing statistical module."""

import numpy as np
import pandas as pd
from src.statistics.ab_testing import ABTestingFramework


def test_ab_comparison():
    tester = ABTestingFramework(alpha=0.05)
    np.random.seed(42)
    group_a = np.random.normal(loc=10, scale=2, size=50)
    group_b = np.random.normal(loc=15, scale=2, size=50)

    df = pd.DataFrame({
        "variant": ["A"] * 50 + ["B"] * 50,
        "score": np.concatenate([group_a, group_b])
    })

    res = tester.compare_two_variants(df, "variant", "A", "B", "score")
    assert res["is_statistically_significant"] is True
    assert "Variant B" in res["winner"]
    assert res["lift_percent"] > 0

