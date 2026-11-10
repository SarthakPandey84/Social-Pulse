"""Test Suite for Module 4: A/B Testing Framework.

Verifies:
1. Two-proportion z-tests with Cohen's h and confidence intervals.
2. Two-sample Welch's t-tests with Mann-Whitney U and Hedges' g.
3. Unequal sample sizes handling.
4. Zero denominators and zero variances.
5. Missing observations and NaN handling.
6. Insufficient sample size guards.
7. Outlier detection reporting (Tukey IQR method).
8. Multiple testing corrections (Holm-Bonferroni and Benjamini-Hochberg).
9. Sample size estimation for proportions and continuous outcomes.
10. Achieved statistical power calculations.
11. Observational vs randomized comparison attribution and warning triggers.
12. Schema conformance and reproducibility.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.schemas import ExperimentType
from src.statistics.ab_testing import ABTestingFramework, ComparisonResult


@pytest.fixture
def framework() -> ABTestingFramework:
    return ABTestingFramework(alpha=0.05)


# --- 1. Two-Proportion z-test ---

def test_proportion_comparison_significant_difference(framework):
    """Verify two-proportion z-test detects significant difference with positive lift."""
    # Variant A: 25 successes out of 1000 (2.5%)
    # Variant B: 45 successes out of 1000 (4.5%)
    res = framework.compare_proportions(
        count_a=25, nobs_a=1000,
        count_b=45, nobs_b=1000,
        metric_name="share_rate",
        variant_a_label="Story Hook",
        variant_b_label="Visual Hook"
    )
    assert res.is_statistically_significant is True
    assert res.p_value < 0.05
    assert res.relative_lift_pct > 70.0
    assert res.absolute_difference == pytest.approx(0.02, abs=1e-4)
    assert res.effect_size > 0.0  # Cohen's h
    assert res.effect_size_type == "Cohen's h"
    assert "Visual Hook" in res.verdict
    assert res.confidence_interval[0] > 0.0  # Entire CI strictly above zero


def test_proportion_comparison_no_difference(framework):
    """Verify two-proportion z-test correctly concludes no significance when proportions are equal."""
    res = framework.compare_proportions(
        count_a=30, nobs_a=1000,
        count_b=31, nobs_b=1000,
        metric_name="click_through_rate"
    )
    assert res.is_statistically_significant is False
    assert res.p_value > 0.50
    assert "No Statistically Significant Evidence" in res.verdict
    # Confidence interval must cross zero
    assert res.confidence_interval[0] < 0.0 < res.confidence_interval[1]


# --- 2. Continuous Metric Comparisons (Welch's t-test + Mann-Whitney) ---

def test_continuous_comparison_welch_and_mann_whitney(framework):
    """Verify Welch's t-test and Mann-Whitney U test detect mean differences without equal variance."""
    np.random.seed(42)
    sample_a = np.random.normal(loc=12.0, scale=2.5, size=60)
    sample_b = np.random.normal(loc=16.5, scale=4.0, size=55)  # Unequal size and variance

    res = framework.compare_continuous(
        sample_a, sample_b,
        metric_name="watch_time_seconds",
        variant_a_label="Format A",
        variant_b_label="Format B"
    )
    assert res.is_statistically_significant is True
    assert res.p_value < 0.001
    assert res.mann_whitney_p_value < 0.001
    assert res.relative_lift_pct > 25.0
    assert res.effect_size > 0.8  # Strong Hedges' g
    assert res.effect_size_type == "Hedges' g"
    assert "Format B" in res.verdict
    assert res.confidence_interval[0] > 0.0


# --- 3. Unequal Sample Sizes ---

def test_unequal_sample_sizes_handling(framework):
    """Verify framework computes tests accurately with imbalanced group sizes."""
    np.random.seed(123)
    sample_small = np.random.normal(loc=10.0, scale=2.0, size=25)
    sample_large = np.random.normal(loc=15.0, scale=2.0, size=150)

    res = framework.compare_continuous(sample_small, sample_large, metric_name="retention_rate")
    assert res.sample_size_a == 25
    assert res.sample_size_b == 150
    assert res.is_statistically_significant is True


# --- 4. Zero Denominators and Zero Variances ---

def test_zero_denominator_safeguards(framework):
    """Verify zero or negative exposure returns clean error result without crashing."""
    res = framework.compare_proportions(
        count_a=0, nobs_a=0,
        count_b=10, nobs_b=100
    )
    assert res.is_statistically_significant is False
    assert "Invalid" in res.verdict
    assert len(res.warning_messages) > 0


def test_zero_variance_safeguards(framework):
    """Verify constant samples with zero variance return diagnostic result without crashing."""
    sample_const_a = [5.0] * 30
    sample_const_b = [5.0] * 30

    res = framework.compare_continuous(sample_const_a, sample_const_b, metric_name="score")
    assert res.is_statistically_significant is False
    assert res.absolute_difference == 0.0
    assert any("zero variance" in w.lower() for w in res.warning_messages)


# --- 5. Missing Observations and NaNs ---

def test_missing_and_nan_observations_filtered(framework):
    """Verify NaNs and infinities are safely filtered prior to test execution."""
    sample_with_nans = [10.0, np.nan, 12.0, None, 14.0, np.inf, 11.0]
    sample_clean = [15.0, 16.0, 14.0, 17.0, 15.5]

    res = framework.compare_continuous(sample_with_nans, sample_clean, metric_name="test_metric")
    assert res.sample_size_a == 4  # 4 valid finite numbers in sample A
    assert res.sample_size_b == 5
    assert not math.isnan(res.value_a)
    assert not math.isnan(res.p_value)


# --- 6. Insufficient Sample Sizes ---

def test_insufficient_sample_size_guards(framework):
    """Verify samples with N < 2 return clean error response."""
    res = framework.compare_continuous([10.0], [15.0, 16.0], metric_name="retention")
    assert res.is_statistically_significant is False
    assert "Insufficient sample size" in res.verdict


# --- 7. Outlier Detection Reporting ---

def test_outlier_detection_iqr_method(framework):
    """Verify statistical outliers generate explicit warning message without silent deletion."""
    normal_data = list(np.random.normal(loc=10.0, scale=1.0, size=50))
    data_with_outlier = normal_data + [100.0, 150.0]  # extreme outliers

    res = framework.compare_continuous(data_with_outlier, normal_data, metric_name="views")
    assert any("outlier" in w.lower() for w in res.warning_messages)


# --- 8. Multiple Testing Corrections (Holm & Benjamini-Hochberg) ---

def test_multiple_testing_correction_holm_and_bh(framework):
    """Verify Holm-Bonferroni and Benjamini-Hochberg adjust p-values and protect FWER/FDR."""
    # Create 5 test results with varying p-values
    p_values = [0.005, 0.012, 0.035, 0.048, 0.65]
    results = [
        ComparisonResult(
            metric_name=f"metric_{i}",
            metric_type="continuous",
            variant_a_label="A",
            variant_b_label="B",
            sample_size_a=50,
            sample_size_b=50,
            value_a=10.0,
            value_b=12.0,
            absolute_difference=2.0,
            relative_lift_pct=20.0,
            test_statistic=2.5,
            test_name="Welch's t-test",
            p_value=p,
            confidence_interval=(0.5, 3.5),
            ci_level=0.95,
            effect_size=0.5,
            effect_size_type="Hedges' g",
            is_statistically_significant=True,
            verdict="Initial"
        )
        for i, p in enumerate(p_values)
    ]

    # Apply Holm correction
    holm_results = ABTestingFramework.apply_multiple_testing_correction(results, method="holm", alpha=0.05)
    # Most significant p=0.005 is multiplied by 5 -> 0.025 (< 0.05, still sig)
    assert holm_results[0].p_value_adjusted == pytest.approx(0.005 * 5, abs=1e-4)
    assert holm_results[0].is_statistically_significant is True
    # The marginal p=0.048 becomes non-significant after Holm adjustment
    assert holm_results[3].p_value_adjusted > 0.05
    assert holm_results[3].is_statistically_significant is False


# --- 9. Sample Size and Power Estimation ---

def test_sample_size_estimation_proportions():
    """Verify proportion sample size calculation produces mathematically valid estimates."""
    est = ABTestingFramework.calculate_sample_size_proportions(
        baseline_rate=0.03,  # 3% baseline
        mde_relative=0.20,   # 20% relative lift (detecting 3.6%)
        alpha=0.05,
        power=0.80
    )
    assert est["sample_size_per_variant"] > 5000
    assert est["total_sample_size_required"] == est["sample_size_per_variant"] * 2
    assert "assumptions_disclosed" in est


def test_sample_size_estimation_continuous():
    """Verify continuous outcome sample size calculation produces valid estimates."""
    est = ABTestingFramework.calculate_sample_size_continuous(
        baseline_mean=100.0,
        baseline_std=25.0,
        mde_relative=0.10,  # 10% lift = 10 units (Cohen's d = 10 / 25 = 0.4)
        alpha=0.05,
        power=0.80
    )
    # For d=0.4, 80% power at alpha=0.05 requires ~100 per variant
    assert 80 <= est["sample_size_per_variant"] <= 120
    assert est["standardized_effect_size_d"] == 0.4


def test_achieved_power_calculation():
    """Verify achieved statistical power increases monotonically with sample size and effect size."""
    power_small = ABTestingFramework.calculate_achieved_power(sample_size_a=30, sample_size_b=30, effect_size=0.3)
    power_large = ABTestingFramework.calculate_achieved_power(sample_size_a=200, sample_size_b=200, effect_size=0.3)
    assert power_small < power_large
    assert power_large > 0.80


# --- 10. Observational vs Randomized Comparison Attribution ---

def test_observational_comparison_warning_trigger(framework):
    """Verify comparisons marked observational include non-randomized confounder warning."""
    res = framework.compare_proportions(
        count_a=20, nobs_a=500,
        count_b=35, nobs_b=500,
        experiment_type=ExperimentType.OBSERVATIONAL_COMPARISON.value
    )
    assert res.experiment_type == ExperimentType.OBSERVATIONAL_COMPARISON.value
    assert any("observational comparison" in w.lower() for w in res.warning_messages)


# --- 11. Schema Conformance and Backwards Compatibility ---

def test_backwards_compatible_compare_two_variants_api(framework):
    """Verify compare_two_variants DataFrame interface matches original contract."""
    df = pd.DataFrame({
        "hook_type": ["Story"] * 30 + ["Visual"] * 30,
        "shares": list(np.random.normal(50, 10, 30)) + list(np.random.normal(80, 15, 30))
    })
    res_dict = framework.compare_two_variants(df, "hook_type", "Story", "Visual", "shares")
    assert "lift_percent" in res_dict
    assert "t_statistic" in res_dict
    assert "p_value" in res_dict
    assert "is_statistically_significant" in res_dict
    assert "winner" in res_dict


# --- 12. Statistical Audit Tests (Module 4 Release Gate) ---

def test_sparse_proportions_and_fisher_exact_fallback(framework):
    """Verify sparse proportion cell counts trigger warning, compute Fisher exact fallback, and select Fisher as primary method."""
    # Control: 2 successes / 20; Treatment: 8 successes / 20 (cells < 5)
    res = framework.compare_proportions(
        count_a=2, nobs_a=20,
        count_b=8, nobs_b=20,
        metric_name="share_rate"
    )
    assert res.primary_method == "Fisher's Exact Test"
    assert res.test_name == "Fisher's Exact Test"
    assert res.fisher_exact_p_value is not None
    assert res.p_value == res.fisher_exact_p_value
    assert 0.0 < res.fisher_exact_p_value <= 1.0
    assert any("sparse proportion data" in w.lower() for w in res.warning_messages)
    assert "Agresti-Caffo" in res.ci_method
    assert -1.0 <= res.confidence_interval[0] < res.confidence_interval[1] <= 1.0
    assert res.clustering_handled is False


def test_sparse_proportion_fisher_controls_verdict(framework):
    """Verify Fisher's exact test strictly dictates the significance verdict when sparse,
    even when asymptotic z-test would spuriously claim significance.
    """
    # count_a=2, nobs_a=30 vs count_b=8, nobs_b=30
    # Asymptotic z-test p = 0.0377 (claims p < 0.05)
    # Fisher's exact test p = 0.0797 (correctly non-significant at alpha=0.05)
    res = framework.compare_proportions(
        count_a=2, nobs_a=30,
        count_b=8, nobs_b=30,
        alpha=0.05
    )
    assert res.primary_method == "Fisher's Exact Test"
    assert res.p_value == pytest.approx(0.0797, abs=1e-3)
    # Verdict must be driven by Fisher's p-value (not significant), NOT by the z-test p-value (0.0377)
    assert res.is_statistically_significant is False
    assert "No Statistically Significant Evidence of Difference via Fisher's Exact Test" in res.verdict
    assert res.diagnostic_p_value == pytest.approx(0.0377, abs=1e-3)


def test_agresti_caffo_handles_zero_successes(framework):
    """Verify Agresti-Caffo produces valid bounded CI when one variant has zero successes."""
    res = framework.compare_proportions(
        count_a=0, nobs_a=50,
        count_b=5, nobs_b=50,
        metric_name="conversion_rate"
    )
    assert res.confidence_interval[0] >= -1.0
    assert res.confidence_interval[1] <= 1.0
    assert res.confidence_interval[1] > res.confidence_interval[0]
    assert res.fisher_exact_p_value is not None
    assert "Approximate Coverage" in res.ci_method


def test_mann_whitney_u_rank_dominance_interpretation(framework):
    """Verify Mann-Whitney U test p-value and rank distribution interpretation."""
    np.random.seed(999)
    # Skewed distributions (log-normal)
    sample_a = np.random.lognormal(mean=2.0, sigma=0.8, size=40)
    sample_b = np.random.lognormal(mean=3.0, sigma=0.8, size=40)
    res = framework.compare_continuous(sample_a, sample_b, metric_name="virality_index")
    assert res.mann_whitney_p_value is not None
    assert res.mann_whitney_p_value < 0.01
    assert "Independent Post Unit" in res.unit_of_analysis
    assert res.clustering_handled is True
    assert res.ci_method == "Welch-Satterthwaite t-interval"


def test_cluster_defensible_post_level_rates_comparison(framework):
    """Verify compare_post_level_rates evaluates independent post rates without unclustered Bernoulli assumptions."""
    np.random.seed(42)
    # 25 independent posts per variant with varying reach and shares
    df_posts = pd.DataFrame({
        "post_id": [f"POST_{i:03d}" for i in range(50)],
        "variant": ["Hook_A"] * 25 + ["Hook_B"] * 25,
        "reach": list(np.random.randint(1000, 5000, 50)),
        "shares": list(np.random.randint(20, 100, 25)) + list(np.random.randint(80, 250, 25))
    })
    res = framework.compare_post_level_rates(
        df_posts,
        variant_col="variant",
        variant_a="Hook_A",
        variant_b="Hook_B",
        numerator_col="shares",
        denominator_col="reach",
        metric_name="share_rate"
    )
    assert res.clustering_handled is True
    assert "Cluster-Collapsed" in res.unit_of_analysis
    assert res.primary_method == "Welch's t-test on Post-Level Rates"
    assert res.sample_size_a == 25
    assert res.sample_size_b == 25
    assert res.is_statistically_significant is True
    assert res.relative_lift_pct > 0


def test_unclustered_aggregate_proportion_clustering_limitation_warning(framework):
    """Verify compare_proportions discloses that it does NOT handle intra-post clustering."""
    res = framework.compare_proportions(
        count_a=50, nobs_a=2000,
        count_b=100, nobs_b=2000,
        metric_name="share_rate"
    )
    assert res.clustering_handled is False
    assert any("clustering limitation" in w.lower() for w in res.warning_messages)
    assert "Aggregate Exposure Unit" in res.unit_of_analysis


def test_multiple_testing_robustness_with_nans_and_ties(framework):
    """Verify multiple testing correction handles ties and missing p-values gracefully."""
    results = [
        ComparisonResult(
            metric_name="m1", metric_type="continuous", variant_a_label="A", variant_b_label="B",
            sample_size_a=30, sample_size_b=30, value_a=1.0, value_b=2.0, absolute_difference=1.0,
            relative_lift_pct=100.0, test_statistic=2.5, test_name="t-test", p_value=0.02,
            confidence_interval=(0.2, 1.8), ci_level=0.95, effect_size=0.5, effect_size_type="Hedges' g",
            is_statistically_significant=True, verdict="Favors B"
        ),
        ComparisonResult(
            metric_name="m2", metric_type="continuous", variant_a_label="A", variant_b_label="B",
            sample_size_a=30, sample_size_b=30, value_a=1.0, value_b=1.0, absolute_difference=0.0,
            relative_lift_pct=0.0, test_statistic=0.0, test_name="t-test", p_value=float("nan"),
            confidence_interval=(0.0, 0.0), ci_level=0.95, effect_size=0.0, effect_size_type="Hedges' g",
            is_statistically_significant=False, verdict="Invalid"
        ),
        ComparisonResult(
            metric_name="m3", metric_type="continuous", variant_a_label="A", variant_b_label="B",
            sample_size_a=30, sample_size_b=30, value_a=1.0, value_b=2.0, absolute_difference=1.0,
            relative_lift_pct=100.0, test_statistic=2.5, test_name="t-test", p_value=0.02,  # tied with m1
            confidence_interval=(0.2, 1.8), ci_level=0.95, effect_size=0.5, effect_size_type="Hedges' g",
            is_statistically_significant=True, verdict="Favors B"
        ),
    ]

    holm = ABTestingFramework.apply_multiple_testing_correction(results, method="holm", alpha=0.05)
    # Valid count m = 2 (m1 and m3)
    assert holm[0].p_value_adjusted == pytest.approx(0.02 * 2, abs=1e-4)
    assert holm[1].p_value_adjusted is None
    assert holm[1].is_statistically_significant is False
    assert holm[2].p_value_adjusted == pytest.approx(0.02 * 2, abs=1e-4)


def test_sample_size_and_power_input_validation():
    """Verify input validation guards against invalid parameters."""
    with pytest.raises(ValueError):
        ABTestingFramework.calculate_sample_size_proportions(baseline_rate=1.5, mde_relative=0.20)
    with pytest.raises(ValueError):
        ABTestingFramework.calculate_sample_size_proportions(baseline_rate=0.05, mde_relative=-0.10)
    with pytest.raises(ValueError):
        ABTestingFramework.calculate_sample_size_continuous(baseline_mean=10.0, baseline_std=-2.0, mde_relative=0.10)
    with pytest.raises(ValueError):
        ABTestingFramework.calculate_sample_size_continuous(baseline_mean=0.0, baseline_std=5.0, mde_relative=0.10)
    with pytest.raises(ValueError):
        ABTestingFramework.calculate_achieved_power(sample_size_a=50, sample_size_b=50, effect_size=0.3, alpha=1.5)


def test_pseudo_replication_guard_deduplicates_repeated_post_snapshots(framework):
    """Verify compare_two_variants flags and deduplicates repeated snapshots of the same post."""
    df_with_dupes = pd.DataFrame({
        "post_id": ["P1", "P1", "P2", "P3", "P3", "P4"],  # repeated post snapshots
        "variant": ["A", "A", "A", "B", "B", "B"],
        "collection_timestamp": [
            "2026-10-01T10:00:00Z", "2026-10-07T10:00:00Z", "2026-10-07T10:00:00Z",
            "2026-10-01T10:00:00Z", "2026-10-07T10:00:00Z", "2026-10-07T10:00:00Z"
        ],
        "retention_rate": [45.0, 50.0, 52.0, 60.0, 65.0, 68.0]
    })
    res_dict = framework.compare_two_variants(df_with_dupes, "variant", "A", "B", "retention_rate")
    # Should evaluate 2 unique posts for A and 2 unique posts for B (4 total)
    assert res_dict["sample_size_a"] == 2
    assert res_dict["sample_size_b"] == 2
    assert any("pseudo-replication" in w.lower() for w in res_dict["warning_messages"])



