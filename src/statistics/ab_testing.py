"""A/B Testing and Statistical Experimentation Framework.

Provides rigorous parametric and non-parametric hypothesis testing for social media experiments:
- Two-proportion tests with observed/expected cell verification, Fisher's exact test fallback
  as the primary inferential method when sparse, Agresti-Caffo adjusted Wald confidence intervals,
  and Cohen's h effect sizes.
- Two-sample Welch's t-tests with Mann-Whitney U checks (rank distribution stochastic dominance),
  Welch-Satterthwaite degrees of freedom, and Hedges' g effect sizes.
- Cluster-defensible post-level rate comparisons (compare_post_level_rates) to eliminate
  intra-post exposure clustering bias.
- Multiple testing adjustments (Holm-Bonferroni FWER step-down and Benjamini-Hochberg FDR step-up)
  with robust handling of ties and missing p-values.
- Pre-experiment sample size planning and approximate achieved statistical power calculations.
- Explicit guards against pseudo-replication (repeated snapshots of the same post),
  zero denominators, zero variance, and confounding observational bias.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.data_ingestion.schemas import (
    ExperimentStatus,
    ExperimentType,
    UNIFIED_EXPERIMENT_FIELDS,
)
from src.preprocessing.cleaning import get_latest_posts_view


@dataclass
class ComparisonResult:
    """Encapsulates comprehensive statistical outputs and diagnostics for an A/B test."""
    metric_name: str
    metric_type: str                  # 'proportion' | 'continuous'
    variant_a_label: str
    variant_b_label: str
    sample_size_a: int
    sample_size_b: int
    value_a: float                    # mean or proportion
    value_b: float                    # mean or proportion
    absolute_difference: float        # value_b - value_a
    relative_lift_pct: float          # (value_b - value_a) / value_a * 100
    test_statistic: float             # primary z, t, or odds ratio
    test_name: str
    p_value: float                    # primary inferential p-value
    confidence_interval: Tuple[float, float]
    ci_level: float
    effect_size: float                # Cohen's d/g or Cohen's h
    effect_size_type: str
    is_statistically_significant: bool
    verdict: str
    primary_method: str = ""
    diagnostic_test_statistic: Optional[float] = None
    diagnostic_p_value: Optional[float] = None
    p_value_adjusted: Optional[float] = None
    mann_whitney_p_value: Optional[float] = None
    fisher_exact_p_value: Optional[float] = None
    ci_method: str = "Agresti-Caffo Adjusted Wald Interval (Approximate Coverage)"
    unit_of_analysis: str = "Post"
    clustering_handled: bool = False
    clustering_method: str = "None"
    warning_messages: List[str] = field(default_factory=list)
    experiment_type: str = ExperimentType.RANDOMIZED_EXPERIMENT.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialisation and UI presentation."""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "variant_a_label": self.variant_a_label,
            "variant_b_label": self.variant_b_label,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
            "variant_a_mean": round(self.value_a, 4),
            "variant_b_mean": round(self.value_b, 4),
            "absolute_difference": round(self.absolute_difference, 4),
            "lift_percent": round(self.relative_lift_pct, 2),
            "test_statistic": round(self.test_statistic, 4),
            "t_statistic": round(self.test_statistic, 4),
            "test_name": self.test_name,
            "primary_method": self.primary_method,
            "p_value": round(self.p_value, 5),
            "diagnostic_test_statistic": round(self.diagnostic_test_statistic, 4) if self.diagnostic_test_statistic is not None else None,
            "diagnostic_p_value": round(self.diagnostic_p_value, 5) if self.diagnostic_p_value is not None else None,
            "p_value_adjusted": round(self.p_value_adjusted, 5) if self.p_value_adjusted is not None else None,
            "mann_whitney_p_value": round(self.mann_whitney_p_value, 5) if self.mann_whitney_p_value is not None else None,
            "fisher_exact_p_value": round(self.fisher_exact_p_value, 5) if self.fisher_exact_p_value is not None else None,
            "ci_diff": (round(self.confidence_interval[0], 4), round(self.confidence_interval[1], 4)),
            "ci_95_diff": (round(self.confidence_interval[0], 4), round(self.confidence_interval[1], 4)),
            "ci_level": self.ci_level,
            "ci_method": self.ci_method,
            "effect_size": round(self.effect_size, 4),
            "effect_size_type": self.effect_size_type,
            "is_statistically_significant": self.is_statistically_significant,
            "winner": self.verdict,
            "unit_of_analysis": self.unit_of_analysis,
            "clustering_handled": self.clustering_handled,
            "clustering_method": self.clustering_method,
            "warning_messages": self.warning_messages,
            "experiment_type": self.experiment_type
        }


class ABTestingFramework:
    """Comprehensive statistical inference engine for digital content A/B testing."""

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)

    # --- 1. PROPORTION / CONVERSION COMPARISON (TWO-PROPORTION Z-TEST + FISHER FALLBACK) ---

    def compare_proportions(
        self,
        count_a: int,
        nobs_a: int,
        count_b: int,
        nobs_b: int,
        metric_name: str = "conversion_rate",
        variant_a_label: str = "Control (A)",
        variant_b_label: str = "Treatment (B)",
        alpha: Optional[float] = None,
        experiment_type: str = ExperimentType.RANDOMIZED_EXPERIMENT.value
    ) -> ComparisonResult:
        """Execute two-proportion comparison with pooled variance, Fisher's exact test fallback
        as primary inferential method when sparse, Agresti-Caffo confidence interval, and Cohen's h.

        Method Selection & Clustering Notes:
        - Evaluates expected and observed contingency cells under H0.
        - When sparse (min observed or expected count < 5), Fisher's exact test is selected as the
          PRIMARY inferential test controlling the final p-value and significance verdict.
        - Agresti–Caffo adjusted Wald interval, which generally improves coverage compared with
          the conventional Wald interval, particularly for small samples and boundary proportions.
          Coverage remains approximate.
        - Unit of Analysis: Aggregated Exposure Unit (unclustered reach/views). Ignores intra-post
          clustering (ICC > 0). For creative comparisons with multiple posts, use compare_post_level_rates().
        """
        sig_alpha = float(alpha) if alpha is not None else self.alpha
        if not (0.0 < sig_alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {sig_alpha}")

        warnings = []
        unit_of_analysis = "Aggregate Exposure Unit (Unclustered reach/views)"

        # Explicit Clustering Limitation Disclosure
        warnings.append(
            "Clustering Limitation: This two-proportion test treats exposure units (reach/views) as independent "
            "Bernoulli trials. It does NOT correct for intra-post clustering (ICC > 0). For creative experiments "
            "spanning multiple posts, use compare_post_level_rates() to evaluate independent post-level rates."
        )

        if experiment_type == ExperimentType.OBSERVATIONAL_COMPARISON.value:
            warnings.append(
                "Observational comparison: Audience allocation was not randomized; confounders (timing, algorithmic delivery) may exist. "
                "Differences indicate correlation only and cannot be interpreted as causal proof."
            )

        # Defensive Zero / Negative / Non-Integer Sample Checks
        if nobs_a <= 0 or nobs_b <= 0:
            return ComparisonResult(
                metric_name=metric_name,
                metric_type="proportion",
                variant_a_label=variant_a_label,
                variant_b_label=variant_b_label,
                sample_size_a=nobs_a,
                sample_size_b=nobs_b,
                value_a=0.0,
                value_b=0.0,
                absolute_difference=0.0,
                relative_lift_pct=0.0,
                test_statistic=0.0,
                test_name="Two-Proportion z-test",
                primary_method="N/A",
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                ci_level=1.0 - sig_alpha,
                ci_method="N/A",
                effect_size=0.0,
                effect_size_type="Cohen's h",
                is_statistically_significant=False,
                verdict="Invalid: Denominator (exposure / sample size) must be greater than zero.",
                unit_of_analysis=unit_of_analysis,
                clustering_handled=False,
                clustering_method="Unclustered (Assumes independent exposures; intra-post clustering ignored)",
                warning_messages=["Denominator count is zero or negative; comparison cannot be computed."],
                experiment_type=experiment_type
            )

        if count_a < 0 or count_b < 0 or count_a > nobs_a or count_b > nobs_b:
            return ComparisonResult(
                metric_name=metric_name,
                metric_type="proportion",
                variant_a_label=variant_a_label,
                variant_b_label=variant_b_label,
                sample_size_a=nobs_a,
                sample_size_b=nobs_b,
                value_a=0.0,
                value_b=0.0,
                absolute_difference=0.0,
                relative_lift_pct=0.0,
                test_statistic=0.0,
                test_name="Two-Proportion z-test",
                primary_method="N/A",
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                ci_level=1.0 - sig_alpha,
                ci_method="N/A",
                effect_size=0.0,
                effect_size_type="Cohen's h",
                is_statistically_significant=False,
                verdict="Invalid: Success count cannot be negative or exceed sample size.",
                unit_of_analysis=unit_of_analysis,
                clustering_handled=False,
                clustering_method="Unclustered (Assumes independent exposures; intra-post clustering ignored)",
                warning_messages=["Count must be within [0, nobs]."],
                experiment_type=experiment_type
            )

        # Compute empirical proportions
        p_a = max(0.0, min(1.0, count_a / nobs_a))
        p_b = max(0.0, min(1.0, count_b / nobs_b))
        diff = p_b - p_a

        # Pooled proportion under Null Hypothesis H0: p_A = p_B
        p_pool = (count_a + count_b) / (nobs_a + nobs_b)

        # Expected counts under H0
        exp_succ_a = nobs_a * p_pool
        exp_fail_a = nobs_a * (1.0 - p_pool)
        exp_succ_b = nobs_b * p_pool
        exp_fail_b = nobs_b * (1.0 - p_pool)

        obs_cells = [count_a, nobs_a - count_a, count_b, nobs_b - count_b]
        exp_cells = [exp_succ_a, exp_fail_a, exp_succ_b, exp_fail_b]
        min_obs = min(obs_cells)
        min_exp = min(exp_cells)
        is_sparse = bool(min_obs < 5 or min_exp < 5)

        # Compute asymptotic z-statistic
        se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / nobs_a + 1.0 / nobs_b)) if 0.0 < p_pool < 1.0 else 0.0
        if se_pool > 0.0:
            z_stat = diff / se_pool
            z_pval = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            z_pval = 1.0

        # Compute Fisher's Exact Test
        odds_ratio = None
        fisher_pval = None
        try:
            contingency_table = [[int(count_a), int(nobs_a - count_a)], [int(count_b), int(nobs_b - count_b)]]
            or_val, f_p = stats.fisher_exact(contingency_table, alternative="two-sided")
            odds_ratio = float(or_val)
            fisher_pval = float(f_p)
        except Exception as e:
            warnings.append(f"Fisher's exact test computation failed: {e}")

        # PRIMARY METHOD SELECTION AND VERDICT GOVERNANCE
        if is_sparse and fisher_pval is not None:
            primary_method = "Fisher's Exact Test"
            test_name = "Fisher's Exact Test"
            primary_p_val = fisher_pval
            primary_test_stat = odds_ratio if (odds_ratio is not None and np.isfinite(odds_ratio)) else 0.0
            diag_stat = z_stat
            diag_pval = z_pval
            warnings.append(
                f"Sparse proportion data: Expected or observed cell count < 5 under H0 "
                f"(min expected = {min_exp:.2f}, min observed = {min_obs}). "
                f"Fisher's exact test is selected as the primary inferential method (p = {primary_p_val:.5f}). "
                f"Asymptotic z-test (z = {diag_stat:.2f}, p = {diag_pval:.5f}) retained for diagnostic reference only."
            )
        else:
            primary_method = "Two-Proportion Pooled z-Test"
            test_name = "Two-Proportion Pooled z-Test"
            primary_p_val = z_pval
            primary_test_stat = z_stat
            diag_stat = odds_ratio if (odds_ratio is not None and np.isfinite(odds_ratio)) else None
            diag_pval = fisher_pval

        # Agresti–Caffo adjusted Wald interval, which generally improves coverage compared with
        # the conventional Wald interval, particularly for small samples and boundary proportions.
        # Coverage remains approximate.
        z_crit = stats.norm.ppf(1.0 - sig_alpha / 2.0)
        n_tilde_a = nobs_a + 2.0
        n_tilde_b = nobs_b + 2.0
        p_tilde_a = (count_a + 1.0) / n_tilde_a
        p_tilde_b = (count_b + 1.0) / n_tilde_b
        diff_tilde = p_tilde_b - p_tilde_a
        se_ac = math.sqrt((p_tilde_a * (1.0 - p_tilde_a) / n_tilde_a) + (p_tilde_b * (1.0 - p_tilde_b) / n_tilde_b))
        ci_lower = max(-1.0, float(diff_tilde - z_crit * se_ac))
        ci_upper = min(1.0, float(diff_tilde + z_crit * se_ac))

        # Relative lift (%)
        rel_lift = (diff / p_a * 100.0) if p_a > 0.0 else 0.0

        # Cohen's h effect size: 2 * (arcsin(sqrt(p_b)) - arcsin(sqrt(p_a)))
        # Positive sign convention indicates Treatment (B) outperforms Control (A).
        h = 2.0 * (math.asin(math.sqrt(p_b)) - math.asin(math.sqrt(p_a)))

        # Verdict strictly dictated by selected primary inferential p-value
        is_sig = bool(primary_p_val < sig_alpha)

        if is_sig:
            direction = f"Variant B ({variant_b_label})" if diff > 0 else f"Variant A ({variant_a_label})"
            verdict = f"Statistically Significant Evidence of Improvement via {primary_method} (Favors {direction})"
        else:
            verdict = f"No Statistically Significant Evidence of Difference via {primary_method}"

        return ComparisonResult(
            metric_name=metric_name,
            metric_type="proportion",
            variant_a_label=variant_a_label,
            variant_b_label=variant_b_label,
            sample_size_a=nobs_a,
            sample_size_b=nobs_b,
            value_a=p_a,
            value_b=p_b,
            absolute_difference=diff,
            relative_lift_pct=rel_lift,
            test_statistic=primary_test_stat,
            test_name=test_name,
            primary_method=primary_method,
            p_value=primary_p_val,
            diagnostic_test_statistic=diag_stat,
            diagnostic_p_value=diag_pval,
            confidence_interval=(ci_lower, ci_upper),
            ci_level=1.0 - sig_alpha,
            ci_method="Agresti-Caffo Adjusted Wald Interval (Approximate Coverage)",
            effect_size=h,
            effect_size_type="Cohen's h",
            is_statistically_significant=is_sig,
            verdict=verdict,
            mann_whitney_p_value=None,
            fisher_exact_p_value=fisher_pval,
            unit_of_analysis=unit_of_analysis,
            clustering_handled=False,
            clustering_method="Unclustered (Assumes independent exposures; intra-post clustering ignored)",
            warning_messages=warnings,
            experiment_type=experiment_type
        )

    # --- 2. CONTINUOUS METRIC COMPARISON (WELCH'S T-TEST + MANN-WHITNEY U) ---

    def compare_continuous(
        self,
        sample_a: Union[List[float], np.ndarray, pd.Series],
        sample_b: Union[List[float], np.ndarray, pd.Series],
        metric_name: str = "retention_rate",
        variant_a_label: str = "Control (A)",
        variant_b_label: str = "Treatment (B)",
        alpha: Optional[float] = None,
        experiment_type: str = ExperimentType.RANDOMIZED_EXPERIMENT.value
    ) -> ComparisonResult:
        """Execute Welch's t-test and Mann-Whitney U test with Hedges' g effect size.

        Statistical notes:
        - Welch's t-test handles unequal sample sizes and unequal variances using Welch-Satterthwaite df.
        - Mann-Whitney U test evaluates the null hypothesis of stochastic equality P(X > Y) + 0.5*P(X == Y) = 0.5.
          It tests rank distribution difference / stochastic dominance, NOT a pure test of medians unless
          both distributions have identical shape and variance.
        - Hedges' g provides a small-sample bias correction to Cohen's d with positive sign indicating B > A.
        """
        sig_alpha = float(alpha) if alpha is not None else self.alpha
        if not (0.0 < sig_alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {sig_alpha}")

        warnings = []
        unit_of_analysis = "Independent Post Unit (Cluster-Collapsed, N = posts)"

        if experiment_type == ExperimentType.OBSERVATIONAL_COMPARISON.value:
            warnings.append(
                "Observational comparison: Content variants were not delivered under controlled audience randomization. "
                "Differences indicate correlation only and cannot be interpreted as causal proof."
            )

        # Clean NaNs and infinite values
        clean_a = np.asarray(sample_a, dtype=float)
        clean_b = np.asarray(sample_b, dtype=float)
        clean_a = clean_a[np.isfinite(clean_a)]
        clean_b = clean_b[np.isfinite(clean_b)]

        n_a = len(clean_a)
        n_b = len(clean_b)

        if n_a < 2 or n_b < 2:
            return ComparisonResult(
                metric_name=metric_name,
                metric_type="continuous",
                variant_a_label=variant_a_label,
                variant_b_label=variant_b_label,
                sample_size_a=n_a,
                sample_size_b=n_b,
                value_a=float(np.mean(clean_a)) if n_a > 0 else 0.0,
                value_b=float(np.mean(clean_b)) if n_b > 0 else 0.0,
                absolute_difference=0.0,
                relative_lift_pct=0.0,
                test_statistic=0.0,
                test_name="Welch's t-test",
                primary_method="Welch's t-test",
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                ci_level=1.0 - sig_alpha,
                ci_method="N/A",
                effect_size=0.0,
                effect_size_type="Hedges' g",
                is_statistically_significant=False,
                verdict="Insufficient sample size to compute hypothesis test (need >= 2 observations per group).",
                unit_of_analysis=unit_of_analysis,
                clustering_handled=True,
                clustering_method="Independent Post Units",
                warning_messages=["Sample size too small to evaluate."],
                experiment_type=experiment_type
            )

        mean_a = float(np.mean(clean_a))
        mean_b = float(np.mean(clean_b))
        var_a = float(np.var(clean_a, ddof=1)) if n_a > 1 else 0.0
        var_b = float(np.var(clean_b, ddof=1)) if n_b > 1 else 0.0
        diff = mean_b - mean_a

        # Check for zero variance
        if var_a == 0.0 and var_b == 0.0:
            warnings.append("Both samples have zero variance (identical values).")
            return ComparisonResult(
                metric_name=metric_name,
                metric_type="continuous",
                variant_a_label=variant_a_label,
                variant_b_label=variant_b_label,
                sample_size_a=n_a,
                sample_size_b=n_b,
                value_a=mean_a,
                value_b=mean_b,
                absolute_difference=diff,
                relative_lift_pct=0.0,
                test_statistic=0.0,
                test_name="Welch's t-test",
                primary_method="Welch's t-test",
                p_value=1.0 if diff == 0.0 else 0.0,
                confidence_interval=(diff, diff),
                ci_level=1.0 - sig_alpha,
                ci_method="Zero-Variance Interval",
                effect_size=0.0,
                effect_size_type="Hedges' g",
                is_statistically_significant=False,
                verdict="Inconclusive: Zero variance observed in both groups.",
                unit_of_analysis=unit_of_analysis,
                clustering_handled=True,
                clustering_method="Independent Post Units",
                warning_messages=warnings,
                experiment_type=experiment_type
            )

        # Welch's t-test (robust against unequal variances; sign aligned with B - A so positive t means B > A)
        raw_t_stat, t_pval = stats.ttest_ind(clean_a, clean_b, equal_var=False)
        t_stat = (-float(raw_t_stat) + 0.0) if np.isfinite(raw_t_stat) else 0.0

        # Mann-Whitney U test (non-parametric safeguard against skewed social distributions)
        # Tests stochastic dominance: P(X > Y) + 0.5*P(X == Y) = 0.5
        try:
            u_stat, u_pval = stats.mannwhitneyu(clean_a, clean_b, alternative="two-sided")
        except Exception:
            u_pval = None

        # Degrees of freedom via Welch-Satterthwaite equation
        se_a = var_a / n_a
        se_b = var_b / n_b
        se_diff = math.sqrt(se_a + se_b)

        if (se_a + se_b) > 0.0:
            dof = ((se_a + se_b)**2) / ((se_a**2 / (n_a - 1)) + (se_b**2 / (n_b - 1)))
            t_crit = stats.t.ppf(1.0 - sig_alpha / 2.0, df=dof)
            ci_lower = diff - t_crit * se_diff
            ci_upper = diff + t_crit * se_diff
        else:
            ci_lower = diff
            ci_upper = diff

        # Relative lift (%)
        rel_lift = (diff / abs(mean_a) * 100.0) if abs(mean_a) > 1e-9 else 0.0

        # Effect Size: Cohen's d with pooled standard deviation, then Hedges' g correction
        s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)) if (n_a + n_b > 2) else se_diff
        cohens_d = diff / s_pooled if s_pooled > 0.0 else 0.0
        # Hedges' g correction factor for small samples (positive sign means B > A)
        hedges_g = cohens_d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)) if (n_a + n_b > 3) else cohens_d

        # Outlier Detection via Interquartile Range (Tukey Method) — observations reported without silent deletion
        for s_name, s_data in [("Variant A", clean_a), ("Variant B", clean_b)]:
            q25, q75 = np.percentile(s_data, 25), np.percentile(s_data, 75)
            iqr = q75 - q25
            outliers = s_data[(s_data < q25 - 1.5 * iqr) | (s_data > q75 + 1.5 * iqr)]
            if len(outliers) > 0:
                warnings.append(
                    f"{s_name} contains {len(outliers)} statistical outlier(s) (IQR method). "
                    f"Retained in dataset to preserve distribution integrity."
                )

        is_sig = bool(t_pval < sig_alpha)

        if is_sig:
            direction = f"Variant B ({variant_b_label})" if diff > 0 else f"Variant A ({variant_a_label})"
            verdict = f"Statistically Significant Evidence of Difference (Favors {direction})"
        else:
            verdict = "No Statistically Significant Evidence of Difference"

        return ComparisonResult(
            metric_name=metric_name,
            metric_type="continuous",
            variant_a_label=variant_a_label,
            variant_b_label=variant_b_label,
            sample_size_a=n_a,
            sample_size_b=n_b,
            value_a=mean_a,
            value_b=mean_b,
            absolute_difference=diff,
            relative_lift_pct=rel_lift,
            test_statistic=float(t_stat),
            test_name="Welch's t-test",
            primary_method="Welch's t-test",
            p_value=float(t_pval),
            confidence_interval=(ci_lower, ci_upper),
            ci_level=1.0 - sig_alpha,
            ci_method="Welch-Satterthwaite t-interval",
            effect_size=float(hedges_g),
            effect_size_type="Hedges' g",
            is_statistically_significant=is_sig,
            verdict=verdict,
            mann_whitney_p_value=float(u_pval) if u_pval is not None else None,
            fisher_exact_p_value=None,
            unit_of_analysis=unit_of_analysis,
            clustering_handled=True,
            clustering_method="Independent Post Units",
            warning_messages=warnings,
            experiment_type=experiment_type
        )

    # --- 3. CLUSTER-DEFENSIBLE POST-LEVEL RATE COMPARISON ---

    def compare_post_level_rates(
        self,
        df: pd.DataFrame,
        variant_col: str,
        variant_a: Any,
        variant_b: Any,
        rate_col: Optional[str] = None,
        numerator_col: Optional[str] = None,
        denominator_col: Optional[str] = None,
        metric_name: str = "share_rate",
        alpha: Optional[float] = None,
        experiment_type: str = ExperimentType.RANDOMIZED_EXPERIMENT.value
    ) -> ComparisonResult:
        """Compare independent post-level rates between variants (cluster-collapsed).

        Methodology & Cluster Defense:
        - Resolves intra-post exposure clustering by collapsing exposures into settled post rates:
          r_i = numerator_i / denominator_i for each post i.
        - Each post represents 1 independent experimental unit (N_A, N_B = number of posts).
        - Evaluates rate differences using Welch's t-test (unequal variances via Welch-Satterthwaite df)
          and Mann-Whitney U test (rank distribution stochastic dominance).
        - Eliminates the pseudo-replication and Type I error inflation of unclustered exposure aggregates.
        """
        df_filtered = df[df[variant_col].isin([variant_a, variant_b])].copy()

        # Deduplicate repeated post snapshots to guarantee 1 independent observation per post
        if "post_id" in df_filtered.columns and df_filtered["post_id"].duplicated().any():
            if "collection_timestamp" in df_filtered.columns:
                df_filtered = df_filtered.sort_values("collection_timestamp").groupby("post_id", as_index=False).last()
            else:
                df_filtered = df_filtered.drop_duplicates(subset=["post_id"], keep="last")

        if rate_col and rate_col in df_filtered.columns:
            df_filtered["_computed_rate"] = df_filtered[rate_col].astype(float)
        elif numerator_col and denominator_col:
            denom = df_filtered[denominator_col].astype(float).replace(0, np.nan)
            df_filtered["_computed_rate"] = df_filtered[numerator_col].astype(float) / denom
        else:
            raise ValueError("Must provide either rate_col or both numerator_col and denominator_col.")

        sample_a = df_filtered[df_filtered[variant_col] == variant_a]["_computed_rate"].dropna().values
        sample_b = df_filtered[df_filtered[variant_col] == variant_b]["_computed_rate"].dropna().values

        res = self.compare_continuous(
            sample_a=sample_a,
            sample_b=sample_b,
            metric_name=metric_name,
            variant_a_label=str(variant_a),
            variant_b_label=str(variant_b),
            alpha=alpha,
            experiment_type=experiment_type
        )

        res.unit_of_analysis = f"Independent Post Unit (Cluster-Collapsed, N_A={len(sample_a)}, N_B={len(sample_b)} posts)"
        res.clustering_handled = True
        res.clustering_method = "Cluster-Collapsed Post-Level Summary (Independent Post Rates)"
        res.primary_method = "Welch's t-test on Post-Level Rates"
        res.warning_messages.append(
            "Cluster-Robust Analysis: Evaluated rate variation across independent posts rather than unclustered exposure units."
        )
        return res

    # --- 4. HIGH-LEVEL DATAFRAME COMPARISON (BACKWARDS COMPATIBILITY) ---

    def compare_two_variants(
        self,
        df: pd.DataFrame,
        variant_col: str,
        variant_a: Any,
        variant_b: Any,
        metric_col: str,
        is_proportion: bool = False,
        exposure_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """High-level dispatch function maintaining full backwards compatibility.

        Guards against pseudo-replication: If 'post_id' is present and contains duplicates,
        deduplicates to the latest settled snapshot per post (by collection_timestamp if available).
        """
        df_filtered = df[df[variant_col].isin([variant_a, variant_b])].copy()

        # Guard against pseudo-replication from repeated post snapshots
        pseudo_warning = None
        if "post_id" in df_filtered.columns and df_filtered["post_id"].duplicated().any():
            initial_count = len(df_filtered)
            if "collection_timestamp" in df_filtered.columns:
                df_filtered = df_filtered.sort_values("collection_timestamp").groupby("post_id", as_index=False).last()
            else:
                df_filtered = df_filtered.drop_duplicates(subset=["post_id"], keep="last")
            dedup_count = len(df_filtered)
            pseudo_warning = (
                f"Pseudo-replication guard: Detected {initial_count - dedup_count} repeated snapshot(s) of identical post_id. "
                f"Deduplicated to latest snapshot per post ({dedup_count} unique posts evaluated)."
            )

        if is_proportion and exposure_col:
            # Aggregate counts and exposures
            sub_a = df_filtered[df_filtered[variant_col] == variant_a]
            sub_b = df_filtered[df_filtered[variant_col] == variant_b]
            cnt_a = int(sub_a[metric_col].sum())
            n_a = int(sub_a[exposure_col].sum())
            cnt_b = int(sub_b[metric_col].sum())
            n_b = int(sub_b[exposure_col].sum())
            res = self.compare_proportions(
                cnt_a, n_a, cnt_b, n_b,
                metric_name=metric_col,
                variant_a_label=str(variant_a),
                variant_b_label=str(variant_b)
            )
            if pseudo_warning:
                res.warning_messages.append(pseudo_warning)
            res_dict = res.to_dict()

            # Attach cluster-collapsed post-level rate comparison if multiple posts exist
            if len(sub_a) >= 2 and len(sub_b) >= 2:
                post_res = self.compare_post_level_rates(
                    df_filtered, variant_col, variant_a, variant_b,
                    numerator_col=metric_col, denominator_col=exposure_col, metric_name=metric_col
                )
                res_dict["cluster_robust_post_comparison"] = post_res.to_dict()

            return res_dict

        # Continuous metric path
        sample_a = df_filtered[df_filtered[variant_col] == variant_a][metric_col].dropna().values
        sample_b = df_filtered[df_filtered[variant_col] == variant_b][metric_col].dropna().values
        res = self.compare_continuous(
            sample_a, sample_b,
            metric_name=metric_col,
            variant_a_label=str(variant_a),
            variant_b_label=str(variant_b)
        )
        if pseudo_warning:
            res.warning_messages.append(pseudo_warning)
        return res.to_dict()

    # --- 5. MULTIPLE TESTING CORRECTION (HOLM & BENJAMINI-HOCHBERG) ---

    @staticmethod
    def apply_multiple_testing_correction(
        results: List[ComparisonResult],
        method: str = "holm",
        alpha: float = 0.05
    ) -> List[ComparisonResult]:
        """Apply FWER (Holm-Bonferroni step-down) or FDR (Benjamini-Hochberg step-up) correction across multiple metrics.

        Robustness guarantees:
        - Safely isolates missing/NaN p-values without distorting valid ranks.
        - Handles tied p-values consistently and monotonically.
        - Strictly bounds adjusted p-values in [0.0, 1.0].
        """
        if not results:
            return []

        # Separate results with valid finite p-values from invalid/NaN ones
        valid_indices = []
        valid_pvals = []
        for idx, res in enumerate(results):
            if res.p_value is not None and np.isfinite(res.p_value):
                valid_indices.append(idx)
                valid_pvals.append(float(res.p_value))
            else:
                res.p_value_adjusted = None
                res.is_statistically_significant = False
                res.verdict = "Inconclusive: Invalid or missing raw p-value."

        m = len(valid_indices)
        if m == 0:
            return results
        if m == 1:
            idx = valid_indices[0]
            results[idx].p_value_adjusted = valid_pvals[0]
            results[idx].is_statistically_significant = bool(valid_pvals[0] < alpha)
            return results

        # Deterministic sorting by p-value then original index
        sorted_pairs = sorted(zip(valid_indices, valid_pvals), key=lambda x: (x[1], x[0]))
        sorted_indices = [p[0] for p in sorted_pairs]
        sorted_pvals = [p[1] for p in sorted_pairs]

        adj_pvals = [0.0] * m
        method_clean = method.lower().strip()

        if method_clean in ["holm", "holm-bonferroni"]:
            # Holm-Bonferroni step-down: p_adj_i = min(1, max_{j<=i} ((m - j) * p_j))
            running_max = 0.0
            for rank, p in enumerate(sorted_pvals):
                multiplier = m - rank
                adj_p = min(1.0, max(0.0, p * multiplier))
                running_max = max(running_max, adj_p)
                adj_pvals[rank] = min(1.0, running_max)

        elif method_clean in ["fdr_bh", "benjamini-hochberg", "bh"]:
            # Benjamini-Hochberg step-up: p_adj_i = min_{j>=i} (p_j * m / (j + 1))
            p_bh = [min(1.0, max(0.0, p * m / (rank + 1))) for rank, p in enumerate(sorted_pvals)]
            running_min = 1.0
            bh_adjusted = [0.0] * m
            for i in range(m - 1, -1, -1):
                running_min = min(running_min, p_bh[i])
                bh_adjusted[i] = running_min
            adj_pvals = bh_adjusted
        else:
            raise ValueError(f"Unsupported correction method: '{method}'. Choose 'holm' or 'bh'.")

        # Map back to original results preserving initial ordering
        for rank, orig_idx in enumerate(sorted_indices):
            res = results[orig_idx]
            adj_p = float(adj_pvals[rank])
            res.p_value_adjusted = adj_p
            res.is_statistically_significant = bool(adj_p < alpha)
            if res.is_statistically_significant:
                direction = res.variant_b_label if res.absolute_difference > 0 else res.variant_a_label
                res.verdict = f"Statistically Significant after {method.upper()} correction (Favors {direction})"
            else:
                res.verdict = f"Not Statistically Significant after {method.upper()} multiple testing correction"

        return results

    # --- 6. SAMPLE SIZE & STATISTICAL POWER CALCULATIONS ---

    @staticmethod
    def calculate_sample_size_proportions(
        baseline_rate: float,
        mde_relative: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> Dict[str, Any]:
        """Estimate minimum sample size per variant for two-proportion tests.

        Validates all inputs and documents underlying statistical assumptions:
        - Two-sided hypothesis test with balanced 1:1 allocation.
        - Asymptotic normal approximation to independent Bernoulli trials.
        """
        if not (0.0 < float(baseline_rate) < 1.0):
            raise ValueError(f"baseline_rate must be between 0 and 1, got {baseline_rate}")
        if float(mde_relative) <= 0.0:
            raise ValueError(f"mde_relative must be positive, got {mde_relative}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not (0.0 < float(power) < 1.0):
            raise ValueError(f"power must be between 0 and 1, got {power}")

        p1 = float(baseline_rate)
        p2_target = p1 * (1.0 + float(mde_relative))
        p2 = min(0.9999, max(0.0001, p2_target))

        z_alpha = stats.norm.ppf(1.0 - float(alpha) / 2.0)
        z_beta = stats.norm.ppf(float(power))

        p_avg = (p1 + p2) / 2.0
        delta = abs(p2 - p1)

        if delta == 0.0:
            raise ValueError("Minimum detectable effect delta cannot be zero.")

        # Standard sample size formula for two proportions
        n = (z_alpha * math.sqrt(2.0 * p_avg * (1.0 - p_avg)) + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2)))**2 / (delta**2)
        n_per_variant = int(math.ceil(n))

        return {
            "baseline_conversion_rate": round(p1, 4),
            "expected_treatment_rate": round(p2, 4),
            "mde_relative_pct": round(float(mde_relative) * 100.0, 2),
            "mde_absolute": round(delta, 4),
            "significance_alpha": float(alpha),
            "statistical_power": float(power),
            "sample_size_per_variant": n_per_variant,
            "total_sample_size_required": n_per_variant * 2,
            "planning_method": "Two-Proportion Normal Approximation (1:1 balanced allocation, two-sided)",
            "assumptions_disclosed": (
                "Formula assumes balanced 1:1 allocation, independent Bernoulli exposures, "
                "two-sided hypothesis testing, and normal approximation. "
                "Organic social media exposures exhibit intra-post clustering; actual required sample size may be higher."
            )
        }

    @staticmethod
    def calculate_sample_size_continuous(
        baseline_mean: float,
        baseline_std: float,
        mde_relative: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> Dict[str, Any]:
        """Estimate minimum sample size per variant for continuous outcomes (t-test).

        Validates all inputs and documents underlying statistical assumptions:
        - Two-sided Welch/Student t-test with balanced 1:1 allocation.
        - Asymptotic normal approximation to independent observations.
        """
        if float(baseline_std) <= 0.0:
            raise ValueError(f"baseline_std must be positive, got {baseline_std}")
        if float(baseline_mean) == 0.0:
            raise ValueError("baseline_mean cannot be zero when using relative MDE")
        if float(mde_relative) <= 0.0:
            raise ValueError(f"mde_relative must be positive, got {mde_relative}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not (0.0 < float(power) < 1.0):
            raise ValueError(f"power must be between 0 and 1, got {power}")

        delta = abs(float(baseline_mean) * float(mde_relative))
        z_alpha = stats.norm.ppf(1.0 - float(alpha) / 2.0)
        z_beta = stats.norm.ppf(float(power))

        effect_size = delta / float(baseline_std)
        n = 2.0 * ((z_alpha + z_beta)**2) / (effect_size**2)
        n_per_variant = int(math.ceil(n))

        return {
            "baseline_mean": round(float(baseline_mean), 4),
            "baseline_std": round(float(baseline_std), 4),
            "mde_relative_pct": round(float(mde_relative) * 100.0, 2),
            "mde_absolute": round(delta, 4),
            "standardized_effect_size_d": round(effect_size, 4),
            "significance_alpha": float(alpha),
            "statistical_power": float(power),
            "sample_size_per_variant": n_per_variant,
            "total_sample_size_required": n_per_variant * 2,
            "planning_method": "Two-Sample t-Test Normal Approximation (1:1 balanced allocation, two-sided)",
            "assumptions_disclosed": (
                "Formula assumes balanced 1:1 allocation, independent normally distributed experimental units, "
                "and homoscedastic variance. Highly skewed virality distributions require larger sample sizes."
            )
        }

    @staticmethod
    def calculate_achieved_power(
        sample_size_a: int,
        sample_size_b: int,
        effect_size: float,
        alpha: float = 0.05
    ) -> float:
        """Calculate approximate achieved statistical power (1 - beta) for an observed effect size.

        Note: Uses the standard normal approximation of the non-central distribution.
        """
        if int(sample_size_a) <= 0 or int(sample_size_b) <= 0 or float(effect_size) == 0.0:
            return 0.0
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        n_harmonic = 2.0 * (sample_size_a * sample_size_b) / (sample_size_a + sample_size_b)
        z_crit = stats.norm.ppf(1.0 - float(alpha) / 2.0)
        non_centrality = abs(float(effect_size)) * math.sqrt(n_harmonic / 2.0)
        z_power = non_centrality - z_crit
        power = float(stats.norm.cdf(z_power))
        return round(power, 4)

    @staticmethod
    def calculate_approximate_achieved_power_report(
        sample_size_a: int,
        sample_size_b: int,
        effect_size: float,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Generate full diagnostic report for approximate achieved power."""
        power = ABTestingFramework.calculate_achieved_power(sample_size_a, sample_size_b, effect_size, alpha=alpha)
        return {
            "approximate_achieved_power": power,
            "approximation_method": "Standard normal approximation of non-central distribution",
            "sample_size_a": sample_size_a,
            "sample_size_b": sample_size_b,
            "effect_size": effect_size,
            "significance_alpha": alpha,
            "adequately_powered": bool(power >= 0.80)
        }

    # --- 7. EXPERIMENT DESIGN, CREATOR-CLUSTERED INFERENCE & OBSERVATIONAL COHORT EVALUATION ---

    @staticmethod
    def create_experiment_metadata(
        *,
        experiment_id: str,
        experiment_name: str,
        hypothesis: str,
        experiment_type: str,
        platform: str,
        variant_a_label: str,
        variant_b_label: str,
        primary_metric: str,
        secondary_metrics: str = "",
        start_date: str = "",
        end_date: str = "",
        experiment_status: str = ExperimentStatus.RUNNING.value,
    ) -> Dict[str, Any]:
        """Construct a validated metadata record conforming to UNIFIED_EXPERIMENT_FIELDS."""
        valid_types = {e.value for e in ExperimentType}
        e_type = str(experiment_type).strip()
        if e_type not in valid_types:
            raise ValueError(f"Unsupported experiment_type '{experiment_type}'. Expected one of {sorted(valid_types)}.")
        valid_statuses = {s.value for s in ExperimentStatus}
        e_stat = str(experiment_status).strip()
        if e_stat not in valid_statuses:
            raise ValueError(f"Unsupported experiment_status '{experiment_status}'. Expected one of {sorted(valid_statuses)}.")
        if not experiment_id or not experiment_name or not variant_a_label or not variant_b_label or not primary_metric:
            raise ValueError("experiment_id, experiment_name, variant labels, and primary_metric must be non-empty.")

        record = {
            "experiment_id": str(experiment_id).strip(),
            "experiment_name": str(experiment_name).strip(),
            "hypothesis": str(hypothesis).strip(),
            "experiment_type": e_type,
            "platform": str(platform).strip().lower(),
            "variant_a_label": str(variant_a_label).strip(),
            "variant_b_label": str(variant_b_label).strip(),
            "primary_metric": str(primary_metric).strip(),
            "secondary_metrics": str(secondary_metrics).strip(),
            "start_date": str(start_date).strip(),
            "end_date": str(end_date).strip(),
            "experiment_status": e_stat,
        }
        return {k: record[k] for k in UNIFIED_EXPERIMENT_FIELDS}

    @staticmethod
    def stratified_randomize_assignment(
        df: pd.DataFrame,
        *,
        strata_cols: Optional[List[str]] = None,
        variant_a_label: str = "Control (A)",
        variant_b_label: str = "Treatment (B)",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Assign balanced A/B variants via permuted-block randomization within strata."""
        if df is None or len(df) == 0:
            return pd.DataFrame() if df is None else df.copy()

        rng = np.random.default_rng(seed)
        out = df.copy()
        assigned = np.empty(len(out), dtype=object)

        valid_strata = [c for c in (strata_cols or []) if c in out.columns]
        if not valid_strata:
            indices = np.arange(len(out))
            perm = rng.permutation(indices)
            half = len(perm) // 2
            assigned[perm[:half]] = variant_a_label
            assigned[perm[half:]] = variant_b_label
        else:
            # Map positional indices cleanly per stratum
            out_reset = out.reset_index(drop=True)
            for _, grp in out_reset.groupby(valid_strata, dropna=False, sort=True):
                pos_idx = grp.index.to_numpy()
                perm = rng.permutation(pos_idx)
                for rank, p in enumerate(perm):
                    assigned[p] = variant_a_label if (rank % 2 == 0) else variant_b_label

        out["assigned_variant"] = assigned
        return out

    def compare_creator_clustered_rates(
        self,
        df: pd.DataFrame,
        *,
        variant_col: str,
        variant_a: Any,
        variant_b: Any,
        value_col: str,
        creator_col: str = "creator_pseudo_id",
        metric_name: Optional[str] = None,
        alpha: Optional[float] = None,
        experiment_type: str = ExperimentType.RANDOMIZED_EXPERIMENT.value,
    ) -> ComparisonResult:
        """Aggregate posts to unique creators per variant before comparing means to avoid creator pseudoreplication."""
        if value_col not in df.columns or variant_col not in df.columns:
            raise ValueError(f"Required columns '{variant_col}' and '{value_col}' missing from DataFrame.")

        sub = df[df[variant_col].isin([variant_a, variant_b])].copy()
        if "post_id" in sub.columns and sub["post_id"].duplicated().any():
            if "collection_timestamp" in sub.columns:
                sub = sub.sort_values("collection_timestamp").groupby("post_id", as_index=False).last()
            else:
                sub = sub.drop_duplicates(subset=["post_id"], keep="last")

        sub["_val"] = pd.to_numeric(sub[value_col], errors="coerce")
        sub = sub.dropna(subset=["_val"])

        has_creator = creator_col in sub.columns and sub[creator_col].notna().any()
        m_name = metric_name or value_col
        if not has_creator:
            res = self.compare_continuous(
                sub[sub[variant_col] == variant_a]["_val"].to_numpy(),
                sub[sub[variant_col] == variant_b]["_val"].to_numpy(),
                metric_name=m_name,
                variant_a_label=str(variant_a),
                variant_b_label=str(variant_b),
                alpha=alpha,
                experiment_type=experiment_type,
            )
            res.clustering_handled = False
            res.clustering_method = f"Fallback Post-Level (Missing '{creator_col}')"
            res.warning_messages.append(f"Creator clustering column '{creator_col}' unavailable; fell back to post level.")
            return res

        sub_c = sub.dropna(subset=[creator_col]).copy()
        a_sub = sub_c[sub_c[variant_col] == variant_a]
        b_sub = sub_c[sub_c[variant_col] == variant_b]

        a_creators = a_sub.groupby(creator_col, sort=False)["_val"].mean().to_numpy()
        b_creators = b_sub.groupby(creator_col, sort=False)["_val"].mean().to_numpy()
        overlapping_creators = len(set(a_sub[creator_col].unique()) & set(b_sub[creator_col].unique()))

        # Exact unbalanced one-way random-effects ANOVA ICC estimator (Donner & Koval / Searle)
        grp = sub_c.groupby(creator_col, sort=False)["_val"]
        grp_stats = grp.agg(n="count", mean="mean")
        k_clusters = int(len(grp_stats))
        n_total_obs = int(len(sub_c))
        m_bar = float(grp_stats["n"].mean()) if k_clusters > 0 else 1.0
        if k_clusters > 1 and n_total_obs > k_clusters:
            n_i = grp_stats["n"].to_numpy(dtype=float)
            means_i = grp_stats["mean"].to_numpy(dtype=float)
            grand_mean = float(sub_c["_val"].mean())
            ss_between = float(np.sum(n_i * ((means_i - grand_mean) ** 2)))
            ss_total = float(np.sum((sub_c["_val"].to_numpy(dtype=float) - grand_mean) ** 2))
            ss_within = max(0.0, ss_total - ss_between)
            ms_between = ss_between / (k_clusters - 1)
            ms_within = ss_within / (n_total_obs - k_clusters)
            m0 = (n_total_obs - float(np.sum(n_i ** 2)) / n_total_obs) / (k_clusters - 1)
            sigma_b2 = max(0.0, (ms_between - ms_within) / max(m0, 1e-12))
            denom_var = sigma_b2 + ms_within
            icc_est = max(0.0, min(1.0, sigma_b2 / denom_var)) if denom_var > 0 else 0.0
        else:
            icc_est = 0.0
        deff = 1.0 + max(0.0, m_bar - 1.0) * icc_est

        res = self.compare_continuous(
            a_creators,
            b_creators,
            metric_name=m_name,
            variant_a_label=str(variant_a),
            variant_b_label=str(variant_b),
            alpha=alpha,
            experiment_type=experiment_type,
        )
        res.unit_of_analysis = (
            f"Creator-Cluster Collapsed Unit (N_A={len(a_creators)}, N_B={len(b_creators)} creators "
            f"across {len(a_sub)} + {len(b_sub)} posts)"
        )
        res.clustering_handled = True
        res.clustering_method = (
            f"Creator-Level Mean Aggregation (mean posts/creator={m_bar:.2f}, est. ICC={icc_est:.3f}, DEFF={deff:.2f})"
        )
        if overlapping_creators > 0:
            res.warning_messages.append(
                f"Creator Overlap Disclosure: {overlapping_creators} creator(s) contributed posts to both arms; "
                "creator-arm means are treated as unpaired unless analyzed via within-creator paired differences."
            )
        return res

    def evaluate_observational_cohort(
        self,
        df: pd.DataFrame,
        *,
        treatment_col: str = "treatment_group",
        variant_a: str = "unsponsored",
        variant_b: str = "sponsored",
        creator_col: str = "creator_pseudo_id",
        strata_col: str = "follower_count",
    ) -> Dict[str, Any]:
        """Evaluate observational A/B cohort (e.g. Sponsored vs Unsponsored) with strict non-causal & clustering guards."""
        if df is None or len(df) == 0:
            return {"status": "empty", "comparisons": []}

        work = get_latest_posts_view(df) if "post_id" in df.columns else df.copy()
        work = work[work[treatment_col].isin([variant_a, variant_b])].copy()
        if len(work) == 0:
            return {"status": "empty", "comparisons": []}

        if "likes" in work.columns and "follower_count" in work.columns:
            foll = pd.to_numeric(work["follower_count"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            likes = pd.to_numeric(work["likes"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (likes / foll) * 1000.0
            valid_foll = np.isfinite(foll) & (foll > 0.0) & np.isfinite(likes)
            work["likes_per_1k_followers"] = np.where(valid_foll, ratio, np.nan)

        metrics_to_eval = [m for m in ("likes", "comments_count", "likes_per_1k_followers") if m in work.columns and work[m].notna().any()]

        post_level_results: List[ComparisonResult] = []
        creator_level_results: List[ComparisonResult] = []
        for m in metrics_to_eval:
            sa = pd.to_numeric(work.loc[work[treatment_col] == variant_a, m], errors="coerce").dropna().to_numpy()
            sb = pd.to_numeric(work.loc[work[treatment_col] == variant_b, m], errors="coerce").dropna().to_numpy()
            post_level_results.append(
                self.compare_continuous(
                    sa,
                    sb,
                    metric_name=m,
                    variant_a_label=variant_a,
                    variant_b_label=variant_b,
                    experiment_type=ExperimentType.OBSERVATIONAL_COMPARISON.value,
                )
            )
            creator_level_results.append(
                self.compare_creator_clustered_rates(
                    work,
                    variant_col=treatment_col,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    value_col=m,
                    creator_col=creator_col,
                    metric_name=f"{m} (creator-clustered)",
                    experiment_type=ExperimentType.OBSERVATIONAL_COMPARISON.value,
                )
            )

        if post_level_results:
            self.apply_multiple_testing_correction(post_level_results, method="holm", alpha=self.alpha)
        if creator_level_results:
            self.apply_multiple_testing_correction(creator_level_results, method="holm", alpha=self.alpha)

        # Follower-tertile stratification to surface scale confounding
        tertile_summaries: List[Dict[str, Any]] = []
        if strata_col in work.columns and "likes" in work.columns:
            fvals = pd.to_numeric(work[strata_col], errors="coerce").astype("float64")
            valid_mask = fvals.notna().to_numpy()
            valid_f = work.loc[valid_mask].copy()
            if len(valid_f) >= 30 and valid_f[strata_col].nunique() >= 3:
                valid_f["_tertile"] = pd.qcut(fvals.loc[valid_mask], q=3, labels=["Small", "Mid", "Large"], duplicates="drop")
                for t_lbl, t_df in valid_f.groupby("_tertile", observed=False):
                    ta = pd.to_numeric(t_df.loc[t_df[treatment_col] == variant_a, "likes"], errors="coerce").dropna()
                    tb = pd.to_numeric(t_df.loc[t_df[treatment_col] == variant_b, "likes"], errors="coerce").dropna()
                    if len(ta) >= 5 and len(tb) >= 5:
                        t_res = self.compare_continuous(
                            ta.to_numpy(),
                            tb.to_numpy(),
                            metric_name=f"likes ({t_lbl} follower tier)",
                            variant_a_label=variant_a,
                            variant_b_label=variant_b,
                            experiment_type=ExperimentType.OBSERVATIONAL_COMPARISON.value,
                        )
                        tertile_summaries.append({
                            "follower_tier": str(t_lbl),
                            "n_a": int(t_res.sample_size_a),
                            "n_b": int(t_res.sample_size_b),
                            "mean_a": round(t_res.value_a, 2),
                            "mean_b": round(t_res.value_b, 2),
                            "relative_lift_pct": round(t_res.relative_lift_pct, 2),
                            "hedges_g": round(t_res.effect_size, 4),
                            "p_value": round(t_res.p_value, 6),
                        })

        return {
            "experiment_type": ExperimentType.OBSERVATIONAL_COMPARISON.value,
            "is_causal": False,
            "n_posts_total": int(len(work)),
            "n_creators_total": int(work[creator_col].nunique(dropna=True)) if creator_col in work.columns else None,
            "post_level_comparisons": [r.to_dict() for r in post_level_results],
            "creator_clustered_comparisons": [r.to_dict() for r in creator_level_results],
            "follower_tertile_stratified_likes": tertile_summaries,
            "power_reports": {
                r.metric_name: self.calculate_approximate_achieved_power_report(
                    r.sample_size_a, r.sample_size_b, r.effect_size, alpha=self.alpha
                )
                for r in creator_level_results
            },
            "disclosure": (
                "STRICT NON-CAUSAL OBSERVATIONAL COHORT: Sponsored vs unsponsored posts are self-selected by creators "
                "and brand partnerships, not randomized. Follower-tier stratification and creator clustering mitigate "
                "pseudoreplication but cannot eliminate unmeasured confounding."
            ),
        }
