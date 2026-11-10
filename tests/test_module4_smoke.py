"""Separate and Dedicated Statistical Smoke Tests for Module 4.

Explicitly verifies:
1. Proportion testing with actual integer successes (counts) and exposure denominators.
2. Continuous comparison of independent post-level rates.
3. Sparse proportion fallback where Fisher's exact test strictly dictates the verdict.
4. Repeated post snapshot handling (pseudo-replication deduplication).
5. Cluster-aware post-level rates comparison (resolving intra-cluster exposure dependence).
"""

import numpy as np
import pandas as pd
import pytest

from src.statistics.ab_testing import ABTestingFramework


@pytest.fixture
def framework():
    return ABTestingFramework(alpha=0.05)


def test_smoke_proportion_testing_with_actual_counts_and_denominators(framework):
    """Smoke test 1: Validates two-proportion comparison using actual integer successes and exposures."""
    # Control: 1,500 shares out of 50,000 reach (3.0%)
    # Treatment: 2,400 shares out of 60,000 reach (4.0%)
    res = framework.compare_proportions(
        count_a=1500, nobs_a=50000,
        count_b=2400, nobs_b=60000,
        metric_name="share_rate",
        variant_a_label="Control Hook",
        variant_b_label="Treatment Hook"
    )
    assert res.primary_method == "Two-Proportion Pooled z-Test"
    assert res.metric_type == "proportion"
    assert res.value_a == pytest.approx(0.03, abs=1e-4)
    assert res.value_b == pytest.approx(0.04, abs=1e-4)
    assert res.relative_lift_pct == pytest.approx(33.33, abs=1e-1)
    assert res.p_value < 0.0001
    assert res.is_statistically_significant is True
    assert "Treatment Hook" in res.verdict
    assert res.clustering_handled is False
    assert any("clustering limitation" in w.lower() for w in res.warning_messages)


def test_smoke_continuous_comparison_independent_post_level_rates(framework):
    """Smoke test 2: Validates continuous comparison of independent post-level rates via Welch's t-test."""
    # 30 independent post-level retention rates (%) per variant
    np.random.seed(101)
    rates_a = np.random.normal(loc=42.0, scale=4.0, size=30)
    rates_b = np.random.normal(loc=48.5, scale=5.5, size=30)

    res = framework.compare_continuous(
        sample_a=rates_a,
        sample_b=rates_b,
        metric_name="retention_rate",
        variant_a_label="Variant A",
        variant_b_label="Variant B"
    )
    assert res.primary_method == "Welch's t-test"
    assert res.metric_type == "continuous"
    assert res.sample_size_a == 30
    assert res.sample_size_b == 30
    assert res.p_value < 0.001
    assert res.mann_whitney_p_value is not None
    assert res.mann_whitney_p_value < 0.001
    assert res.clustering_handled is True
    assert "Independent Post Unit" in res.unit_of_analysis


def test_smoke_sparse_proportion_fallback_and_verdict_selection(framework):
    """Smoke test 3: Proves Fisher's exact test fallback is selected and governs the verdict when sparse."""
    # 2 successes / 30 vs 8 successes / 30
    # Asymptotic z-test p = 0.0377 (claims significance)
    # Fisher exact test p = 0.0797 (correctly rejects significance at alpha=0.05)
    res = framework.compare_proportions(
        count_a=2, nobs_a=30,
        count_b=8, nobs_b=30,
        metric_name="click_rate",
        alpha=0.05
    )
    assert res.primary_method == "Fisher's Exact Test"
    assert res.test_name == "Fisher's Exact Test"
    assert res.p_value == pytest.approx(0.0797, abs=1e-3)
    assert res.is_statistically_significant is False
    assert "No Statistically Significant Evidence" in res.verdict
    assert res.diagnostic_p_value == pytest.approx(0.0377, abs=1e-3)


def test_smoke_repeated_post_snapshot_deduplication(framework):
    """Smoke test 4: Proves repeated post snapshots are deduplicated to protect unit integrity."""
    df_snapshots = pd.DataFrame({
        "post_id": ["P101", "P101", "P101", "P102", "P201", "P201", "P202"],
        "variant": ["Control", "Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
        "collection_timestamp": [
            "2026-10-01T00:00:00Z", "2026-10-03T00:00:00Z", "2026-10-07T00:00:00Z",
            "2026-10-07T00:00:00Z",
            "2026-10-01T00:00:00Z", "2026-10-07T00:00:00Z",
            "2026-10-07T00:00:00Z"
        ],
        "saves": [10, 15, 20, 25, 30, 45, 50]
    })
    res_dict = framework.compare_two_variants(df_snapshots, "variant", "Control", "Treatment", "saves")
    # 2 unique posts in Control (P101, P102) and 2 unique posts in Treatment (P201, P202)
    assert res_dict["sample_size_a"] == 2
    assert res_dict["sample_size_b"] == 2
    assert any("pseudo-replication" in w.lower() for w in res_dict["warning_messages"])


def test_smoke_cluster_aware_post_level_rates_method(framework):
    """Smoke test 5: Proves cluster-collapsed post-level rate comparison correctly resolves exposure clustering."""
    np.random.seed(202)
    df_experiment = pd.DataFrame({
        "post_id": [f"EXP_POST_{i:03d}" for i in range(40)],
        "variant": ["Hook_Story"] * 20 + ["Hook_Action"] * 20,
        "reach": list(np.random.randint(5000, 20000, 40)),
        "shares": list(np.random.randint(100, 300, 20)) + list(np.random.randint(400, 900, 20))
    })

    res = framework.compare_post_level_rates(
        df_experiment,
        variant_col="variant",
        variant_a="Hook_Story",
        variant_b="Hook_Action",
        numerator_col="shares",
        denominator_col="reach",
        metric_name="share_rate"
    )
    assert res.clustering_handled is True
    assert "Cluster-Collapsed" in res.unit_of_analysis
    assert res.primary_method == "Welch's t-test on Post-Level Rates"
    assert res.sample_size_a == 20
    assert res.sample_size_b == 20
    assert res.is_statistically_significant is True
    assert res.relative_lift_pct > 50.0

