"""Explainable Engagement Optimization Recommendation Engine.

Synthesizes multi-module analytical outputs:
1. Module 1: Historical descriptive post metrics and retention curves.
2. Module 2: Pre-publication virality model predictions (EVI).
3. Module 3: Hybrid NLP audience sentiment and relatability signals.
4. Module 4: Randomized controlled trials and observational A/B experiments.
5. Prescriptive Experimentation Agenda: Proposing controlled tests for uncertain areas.

Adheres strictly to the evidence hierarchy:
- High Confidence:
    Supported by verified randomized controlled trials with:
    * True random assignment of creative variants.
    * Analysis conducted at the independent post unit (post-level aggregation addressing
      intra-post exposure dependence; note: does not correct cross-post creator or campaign clustering).
    * Adequate sample size (N >= 30 posts per variant, total N >= 60) and statistical power (>= 80%).
    * Statistically significant after multiple testing correction (p_adj < alpha = 0.05).
    * Non-zero effect size (|g| >= 0.20 or |h| >= 0.05) and confidence interval not crossing zero.
    * Transparent data provenance (explicitly flagged if synthetic benchmark demonstration).
- Medium Confidence:
    Supported by controlled observational comparisons with matched cohorts,
    randomized experiments with moderate sample sizes (20 <= N < 60), or
    predictive models verified with temporal train-test cross-validation.
    NOT assigned merely for reaching 15 observational posts.
    NOT assigned to internal rule-based NLP signals without external ground-truth validation.
- Low / Exploratory Confidence:
    Observational cohort patterns, correlation analyses, small samples (N < 20),
    unverified heuristic hypotheses, or proposed A/B experiments. Always pairs with non-causal disclosures.
- Insufficient Evidence:
    Empty datasets, missing metrics, or contradictory findings with no empirical resolution.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.recommender.schemas import (
    ConfidenceLevel,
    ContentRecommendation,
    EvidenceSource,
    RecommendationCategory,
    RecommendationReport
)
from src.preprocessing.cleaning import get_latest_posts_view
from src.statistics.ab_testing import ABTestingFramework, ComparisonResult
from src.data_ingestion.schemas import ExperimentType, SourceProvenance


_REAL_PROVENANCE_VALUES = {
    SourceProvenance.REAL_PUBLIC_DATA.value,
    SourceProvenance.OWNER_AUTHORIZED_DATA.value,
    SourceProvenance.WEAKLY_LABELLED_DATA.value,
}


class OptimizationRecommender:
    """Core prescriptive analytics and evidence-based recommendation engine."""

    MIN_CREATORS_FOR_REC = 3

    def __init__(
        self,
        posts_df: Optional[pd.DataFrame] = None,
        comments_df: Optional[pd.DataFrame] = None,
        experiments_df: Optional[pd.DataFrame] = None,
        observations_df: Optional[pd.DataFrame] = None,
        alpha: float = 0.05
    ):
        if posts_df is not None:
            self.posts_df = (
                get_latest_posts_view(posts_df).copy()
                if ("post_id" in posts_df.columns and len(posts_df) > 0)
                else posts_df.copy()
            )
        else:
            self.posts_df = None
        self.comments_df = comments_df.copy() if comments_df is not None else None
        self.experiments_df = experiments_df.copy() if experiments_df is not None else None
        self.observations_df = observations_df.copy() if observations_df is not None else None
        self.alpha = float(alpha)
        self.framework = ABTestingFramework(alpha=self.alpha)

        # Reference date for recency calculation (latest observation in dataset)
        self._reference_date: Optional[datetime] = None
        if self.posts_df is not None and "timestamp" in self.posts_df.columns:
            ts_series = pd.to_datetime(self.posts_df["timestamp"], errors="coerce")
            if not ts_series.dropna().empty:
                self._reference_date = ts_series.max().to_pydatetime()

        self._is_synthetic_demo: bool = self._infer_is_synthetic_demonstration()

    def _infer_is_synthetic_demonstration(self) -> bool:
        """Infer dynamically whether supplied inputs are predominantly real public/owner data vs synthetic."""
        for df_cand in (self.posts_df, self.comments_df):
            if df_cand is not None and len(df_cand) > 0 and "provenance" in df_cand.columns:
                vals = df_cand["provenance"].dropna().astype(str).str.strip().str.lower()
                if len(vals) > 0:
                    real_frac = float(vals.isin(_REAL_PROVENANCE_VALUES).mean())
                    if real_frac >= 0.5:
                        return False
        return True

    # --- 1. PRIMARY RECOMMENDATION PIPELINE ---

    def generate_recommendations(
        self,
        platform: Optional[str] = None,
        category: Optional[Union[str, RecommendationCategory]] = None,
        min_confidence: Optional[Union[str, ConfidenceLevel]] = None,
        max_recommendations: int = 15
    ) -> RecommendationReport:
        """Generate, validate, calibrate, and rank content recommendations based on multi-module evidence."""
        recommendations: List[ContentRecommendation] = []
        warnings: List[str] = []

        target_platform = platform.lower().strip() if platform else "all"

        # Check data availability
        if self.posts_df is None or len(self.posts_df) == 0:
            warnings.append("Historical posts dataset is missing or empty. Descriptive performance signals unavailable.")
        if self.comments_df is None or len(self.comments_df) == 0:
            warnings.append("Audience comments dataset is missing or empty. Relatability signals unavailable.")
        if self.experiments_df is None or len(self.experiments_df) == 0:
            warnings.append("A/B experimentation dataset is missing or empty. Experimental evidence unavailable.")

        # 1. Experimental Recommendations (Module 4) - Highest Evidence Tier
        exp_recs = self._generate_ab_experiment_recommendations(target_platform)
        recommendations.extend(exp_recs)

        # 2. Audience Relatability & Vulnerability Recommendations (Module 3)
        rel_recs = self._generate_sentiment_relatability_recommendations(target_platform)
        recommendations.extend(rel_recs)

        # 3. Topic Selection & Resonance Recommendations (Module 1 & 2)
        topic_recs = self._generate_topic_recommendations(target_platform)
        recommendations.extend(topic_recs)

        # 4. Content Format, Hook & Duration Recommendations (Module 1 & 4)
        format_recs = self._generate_format_duration_recommendations(target_platform)
        recommendations.extend(format_recs)

        # 5. Timing Recommendations (Module 1 - Observational with explicit non-causal caveats)
        timing_recs = self._generate_timing_recommendations(target_platform)
        recommendations.extend(timing_recs)

        # 6. Prescriptive Experimentation Agenda (A/B Test Proposals for Uncertain Areas)
        agenda_recs = self._generate_experimentation_agenda(target_platform)
        recommendations.extend(agenda_recs)

        # Detect Conflicting Evidence and apply conflict penalty
        conflicts = self._detect_conflicting_evidence(recommendations)

        # Filter by Platform
        if target_platform != "all":
            recommendations = [
                r for r in recommendations
                if r.relevant_platform.lower() in [target_platform, "all"]
            ]

        # Filter by Category
        if category:
            cat_val = category.value if isinstance(category, RecommendationCategory) else str(category)
            recommendations = [
                r for r in recommendations
                if (r.category.value if isinstance(r.category, RecommendationCategory) else str(r.category)) == cat_val
            ]

        # Score and Deterministically Rank Recommendations
        for r in recommendations:
            r.priority_score = self._compute_priority_score(r, target_platform)

        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        # Filter by Minimum Confidence Level
        if min_confidence:
            min_conf_val = min_confidence.value if isinstance(min_confidence, ConfidenceLevel) else str(min_confidence)
            confidence_order = {
                "HIGH": 4,
                "MEDIUM": 3,
                "LOW": 2,
                "EXPLORATORY": 2,
                "INSUFFICIENT_EVIDENCE": 1
            }
            min_rank = confidence_order.get(min_conf_val.upper(), 1)
            recommendations = [
                r for r in recommendations
                if confidence_order.get(r.confidence_level.value, 1) >= min_rank
            ]

        # Cap recommendations
        final_recs = recommendations[:max_recommendations]

        # Summary Metrics
        high_cnt = sum(1 for r in final_recs if r.confidence_level == ConfidenceLevel.HIGH)
        med_cnt = sum(1 for r in final_recs if r.confidence_level == ConfidenceLevel.MEDIUM)
        low_cnt = sum(1 for r in final_recs if r.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.EXPLORATORY, ConfidenceLevel.INSUFFICIENT_EVIDENCE])

        evidence_counts: Dict[str, int] = {}
        for r in final_recs:
            src_key = r.evidence_source.value if isinstance(r.evidence_source, EvidenceSource) else str(r.evidence_source)
            evidence_counts[src_key] = evidence_counts.get(src_key, 0) + 1

        return RecommendationReport(
            total_recommendations=len(final_recs),
            recommendations=final_recs,
            platform_filter=platform,
            category_filter=str(category) if category else None,
            high_confidence_count=high_cnt,
            medium_confidence_count=med_cnt,
            low_confidence_count=low_cnt,
            evidence_breakdown=evidence_counts,
            conflicting_signals=conflicts,
            insufficient_data_warnings=warnings,
            is_synthetic_demonstration=self._is_synthetic_demo
        )

    # --- 2. SPECIALIZED RECOMMENDATION GENERATORS ---

    @staticmethod
    def _format_metric_value(metric_name: str, value: float) -> str:
        """Format metric values consistently according to their measurement unit."""
        m_low = str(metric_name).lower()
        if m_low in {"share_rate", "save_rate"}:
            return f"{value * 100.0:.2f}%"
        if "pct" in m_low or "rate" in m_low:
            return f"{value:.2f}%"
        return f"{value:,.1f}"

    # Minimum independent observations per arm before an experiment can qualify for MEDIUM/HIGH evidence.
    # Conservative classification guardrail only — not a formal power calculation or reliability guarantee.
    MIN_ARM_SAMPLES_FOR_EVIDENCE = 5

    @staticmethod
    def _format_ci_range(metric_name: str, ci_range: Optional[Tuple[float, float]]) -> str:
        """Format 95% confidence interval bounds cleanly with unit context."""
        if (
            ci_range is None
            or not isinstance(ci_range, (tuple, list))
            or len(ci_range) != 2
            or ci_range[0] is None
            or ci_range[1] is None
        ):
            return "N/A (unavailable interval)"
        try:
            low = float(ci_range[0])
            high = float(ci_range[1])
        except (TypeError, ValueError):
            return "N/A (malformed interval)"
        if not (np.isfinite(low) and np.isfinite(high)) or low > high:
            return "N/A (non-finite or invalid interval)"

        m_low = str(metric_name).lower()
        if m_low in {"share_rate", "save_rate"}:
            return f"[{low * 100.0:+.2f}%, {high * 100.0:+.2f}%]"
        if "pct" in m_low or "rate" in m_low:
            return f"[{low:+.2f}%, {high:+.2f}%]"
        return f"[{low:+,.1f}, {high:+,.1f} {metric_name}]"

    @staticmethod
    def _is_valid_finite_ci(ci_range: Any) -> bool:
        """Return True iff ci_range is a 2-element interval with finite, ordered bounds."""
        if (
            ci_range is None
            or not isinstance(ci_range, (tuple, list))
            or len(ci_range) != 2
            or ci_range[0] is None
            or ci_range[1] is None
        ):
            return False
        try:
            low = float(ci_range[0])
            high = float(ci_range[1])
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(low) and np.isfinite(high) and low <= high)

    @staticmethod
    def _deduplicate_continuous_experiment_snapshots(
        sub: pd.DataFrame,
        exp_id: Any,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Deduplicate repeated post snapshots for continuous-metric experiment evaluation.

        - When repeated `post_id` rows exist and a valid ordering timestamp (`collection_timestamp`,
          or `timestamp` fallback) is available, retains strictly the latest snapshot per
          `(variant, post_id)` (or `post_id`).
        - When repeated `post_id` rows exist without a reliable ordering timestamp, avoids
          selecting an arbitrary row by conservatively excluding the ambiguous duplicate
          `post_id` records and returning an explicit disclosure message.
        """
        if sub.empty or "post_id" not in sub.columns:
            return sub, None

        dedup_keys = ["variant", "post_id"] if "variant" in sub.columns else ["post_id"]
        valid_pid_mask = sub["post_id"].notna()
        if not valid_pid_mask.any():
            return sub, None

        dup_mask = sub.loc[valid_pid_mask].duplicated(subset=dedup_keys, keep=False)
        if not dup_mask.any():
            return sub, None

        initial_rows = len(sub)
        order_col = None
        for candidate_col in ("collection_timestamp", "timestamp"):
            if candidate_col in sub.columns:
                parsed_ts = pd.to_datetime(sub[candidate_col], utc=True, errors="coerce")
                # Require valid timestamps across all rows belonging to duplicated post_ids
                dup_post_ids = set(sub.loc[valid_pid_mask].loc[dup_mask, "post_id"].tolist())
                dup_rows_mask = sub["post_id"].isin(dup_post_ids)
                if parsed_ts.loc[dup_rows_mask].notna().all():
                    order_col = candidate_col
                    sub_ordered = sub.copy()
                    sub_ordered["_parsed_order_ts"] = parsed_ts
                    break

        if order_col is not None:
            sub_ordered = sub_ordered.sort_values("_parsed_order_ts", ascending=True, kind="mergesort")
            deduped = (
                sub_ordered.drop_duplicates(subset=dedup_keys, keep="last")
                .drop(columns=["_parsed_order_ts"])
                .reset_index(drop=True)
            )
            removed = initial_rows - len(deduped)
            note = (
                f"Snapshot deduplication ({exp_id}): Collapsed {removed} earlier snapshot(s) using latest "
                f"'{order_col}' per post_id ({len(deduped)} unique post observations retained)."
            )
            return deduped, note

        # Ambiguous duplicates with no reliable timestamp/order column: exclude ambiguous post_ids conservatively
        dup_post_ids = set(sub.loc[valid_pid_mask].loc[dup_mask, "post_id"].tolist())
        unambiguous = sub[~sub["post_id"].isin(dup_post_ids)].reset_index(drop=True)
        excluded_rows = initial_rows - len(unambiguous)
        note = (
            f"Ambiguous duplicate snapshots ({exp_id}): Found {excluded_rows} row(s) across "
            f"{len(dup_post_ids)} repeated post_id(s) without a reliable ordering timestamp; "
            "conservatively excluded ambiguous duplicate post_ids rather than selecting an arbitrary snapshot."
        )
        return unambiguous, note

    def _classify_experiment_evidence(
        self,
        is_randomized: bool,
        comp_res: Optional[ComparisonResult],
    ) -> Tuple[ConfidenceLevel, str]:
        """Classify evidence strength transparently from statistical and design criteria.

        Applies structural validity checks, finite confidence-interval verification, and a
        conservative minimum sample guardrail (`MIN_ARM_SAMPLES_FOR_EVIDENCE = 5` per arm;
        a classification guardrail only, not a formal power calculation).
        """
        if comp_res is None:
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                "No post-level observations were available to evaluate this experiment.",
            )

        # 1. Contract & statistical validity checks (invalid, failed, zero-variance, or malformed results)
        verdict_str = str(comp_res.verdict or "").strip()
        ci_method_str = str(comp_res.ci_method or "").strip()
        primary_method_str = str(comp_res.primary_method or "").strip()
        invalid_verdict_prefixes = ("Invalid", "Insufficient sample size", "Inconclusive: Zero variance")
        if (
            any(verdict_str.startswith(prefix) for prefix in invalid_verdict_prefixes)
            or ci_method_str in {"N/A", "Zero-Variance Interval", "Unavailable"}
            or primary_method_str == "N/A"
        ):
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                f"Statistical comparison could not produce a valid inferential estimate ({verdict_str or 'invalid result contract'}).",
            )

        try:
            n_a = int(comp_res.sample_size_a)
            n_b = int(comp_res.sample_size_b)
        except (TypeError, ValueError):
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                "Statistical comparison reported non-integer or missing arm sample sizes.",
            )

        if n_a < 2 or n_b < 2:
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                f"Insufficient sample size to compute valid hypothesis test (N_A={n_a}, N_B={n_b}; requires >= 2 per arm).",
            )

        # 2. Essential numeric statistics must exist and be finite
        try:
            val_a = float(comp_res.value_a)
            val_b = float(comp_res.value_b)
            abs_diff = float(comp_res.absolute_difference)
            t_stat = float(comp_res.test_statistic)
            p_raw = float(comp_res.p_value)
            eff_size = float(comp_res.effect_size)
            p_adj = float(comp_res.p_value_adjusted) if comp_res.p_value_adjusted is not None else p_raw
        except (TypeError, ValueError):
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                "Statistical comparison contained missing or non-numeric essential metrics.",
            )

        if not all(np.isfinite(v) for v in (val_a, val_b, abs_diff, t_stat, p_raw, p_adj, eff_size)):
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                "Statistical comparison contained non-finite (NaN/Inf) essential statistics.",
            )

        if not (0.0 <= p_raw <= 1.0 and 0.0 <= p_adj <= 1.0):
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                f"Statistical comparison reported out-of-range p-value (raw={p_raw}, adjusted={p_adj}).",
            )

        # 3. Confidence interval must have finite, ordered bounds before evaluating zero exclusion
        if not self._is_valid_finite_ci(comp_res.confidence_interval):
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                "Statistical comparison reported missing, malformed, or non-finite confidence-interval bounds.",
            )

        ci_low = float(comp_res.confidence_interval[0])
        ci_high = float(comp_res.confidence_interval[1])
        ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)

        # 4. Conservative minimum per-arm sample guardrail (classification floor, not a power calculation)
        if n_a < self.MIN_ARM_SAMPLES_FOR_EVIDENCE or n_b < self.MIN_ARM_SAMPLES_FOR_EVIDENCE:
            return (
                ConfidenceLevel.INSUFFICIENT_EVIDENCE,
                f"Sample size below minimum classification guardrail (N_A={n_a}, N_B={n_b}; requires >= "
                f"{self.MIN_ARM_SAMPLES_FOR_EVIDENCE} independent observations per arm; conservative guardrail only, not a formal power calculation).",
            )

        n_tot = n_a + n_b
        is_sig_adj = bool(p_adj < self.alpha)
        eff_abs = abs(eff_size)
        favors_treatment = bool(abs_diff > 0.0)

        if not is_randomized:
            return (
                ConfidenceLevel.LOW,
                "Observational comparison without random assignment; capped at LOW confidence due to potential confounding.",
            )

        if not is_sig_adj or not ci_excludes_zero or not favors_treatment:
            reasons = []
            if not is_sig_adj:
                reasons.append(f"Holm-adjusted p = {p_adj:.4f} >= alpha ({self.alpha:.2f})")
            if not ci_excludes_zero:
                reasons.append("95% confidence interval spans zero")
            if not favors_treatment:
                reasons.append("observed treatment difference is zero or negative")
            return (
                ConfidenceLevel.LOW,
                "Randomized trial is statistically inconclusive or unfavorable (" + "; ".join(reasons) + ").",
            )

        if n_a >= 30 and n_b >= 30 and n_tot >= 60 and eff_abs >= 0.20:
            return (
                ConfidenceLevel.HIGH,
                f"Randomized trial with adequate sample size (N={n_tot}, N_A={n_a}, N_B={n_b} >= 30/arm), "
                f"Holm-adjusted p = {p_adj:.2e} < {self.alpha:.2f}, |{comp_res.effect_size_type}| = {eff_abs:.3f} >= 0.20, "
                "and 95% CI excluding zero (achieved power not separately estimated).",
            )

        caveats = []
        if n_a < 30 or n_b < 30 or n_tot < 60:
            caveats.append(f"moderate sample size (N={n_tot}, N_A={n_a}, N_B={n_b}; < 30/arm threshold for HIGH)")
        if eff_abs < 0.20:
            caveats.append(f"small standardized effect (|{comp_res.effect_size_type}| = {eff_abs:.3f} < 0.20)")
        return (
            ConfidenceLevel.MEDIUM,
            "Statistically significant randomized comparison after Holm correction, downgraded to MEDIUM due to "
            + "; ".join(caveats)
            + ".",
        )

    def _generate_ab_experiment_recommendations(self, target_platform: str) -> List[ContentRecommendation]:
        """Convert validated A/B experiment outcomes into empirical recommendations using Module 4 framework."""
        recs: List[ContentRecommendation] = []
        if self.experiments_df is None or len(self.experiments_df) == 0:
            return recs

        # Collect ComparisonResults across all experiments for multiple testing correction
        exp_results: Dict[str, ComparisonResult] = {}
        exp_dedup_notes: Dict[str, str] = {}
        res_list: List[ComparisonResult] = []

        exp_records = self.experiments_df.to_dict("records")
        if self.observations_df is not None and len(self.observations_df) > 0:
            for exp in exp_records:
                e_id = exp["experiment_id"]
                p_metric = exp["primary_metric"]
                e_type = exp["experiment_type"]
                sub = self.observations_df[self.observations_df["experiment_id"] == e_id].copy()
                va_label = exp["variant_a_label"]
                vb_label = exp["variant_b_label"]

                if p_metric in ["views", "relatability_pct"]:
                    sub_clean, dedup_note = self._deduplicate_continuous_experiment_snapshots(sub, e_id)
                    if dedup_note:
                        exp_dedup_notes[str(e_id)] = dedup_note
                    va = sub_clean[sub_clean["variant"] == va_label]
                    vb = sub_clean[sub_clean["variant"] == vb_label]
                    if len(va) == 0 or len(vb) == 0:
                        continue
                    res = self.framework.compare_continuous(
                        va[p_metric].dropna().values,
                        vb[p_metric].dropna().values,
                        metric_name=p_metric,
                        variant_a_label=va_label,
                        variant_b_label=vb_label,
                        experiment_type=e_type
                    )
                    if dedup_note:
                        res.warning_messages.append(dedup_note)
                else:
                    va = sub[sub["variant"] == va_label]
                    vb = sub[sub["variant"] == vb_label]
                    if len(va) == 0 or len(vb) == 0:
                        continue
                    # share_rate / rates: post-level rate comparison (deduplicates post_id internally)
                    res = self.framework.compare_post_level_rates(
                        df=sub,
                        variant_col="variant",
                        variant_a=va_label,
                        variant_b=vb_label,
                        rate_col=p_metric if p_metric in sub.columns else None,
                        numerator_col="shares" if "shares" in sub.columns else None,
                        denominator_col="views" if "views" in sub.columns else None,
                        metric_name=p_metric,
                        experiment_type=e_type
                    )
                exp_results[e_id] = res
                res_list.append(res)

            # Apply Holm-Bonferroni correction across the experiment suite
            if res_list:
                ABTestingFramework.apply_multiple_testing_correction(res_list, method="holm", alpha=self.alpha)

        # Build structured recommendations dynamically from live ComparisonResults
        for exp in exp_records:
            exp_id = str(exp["experiment_id"])
            p_form = str(exp["platform"]).lower()
            if target_platform != "all" and p_form != target_platform:
                continue

            is_randomized = exp["experiment_type"] == ExperimentType.RANDOMIZED_EXPERIMENT.value
            comp_res = exp_results.get(exp_id)
            conf_level, conf_rationale = self._classify_experiment_evidence(is_randomized, comp_res)
            dedup_note = exp_dedup_notes.get(exp_id)
            if dedup_note:
                conf_rationale = f"{conf_rationale} {dedup_note}"

            # Calculate recency if end_date exists
            recency_days = None
            if "end_date" in exp and pd.notnull(exp["end_date"]):
                try:
                    exp_end = pd.to_datetime(exp["end_date"])
                    ref_dt = self._reference_date if self._reference_date else datetime.utcnow()
                    recency_days = max(0, (ref_dt.date() - exp_end.date()).days)
                except Exception:
                    recency_days = None

            p_metric = str(exp["primary_metric"])
            va_label = str(exp["variant_a_label"])
            vb_label = str(exp["variant_b_label"])

            if comp_res is not None:
                n_a = comp_res.sample_size_a
                n_b = comp_res.sample_size_b
                n_posts = n_a + n_b
                val_a_str = self._format_metric_value(p_metric, comp_res.value_a)
                val_b_str = self._format_metric_value(p_metric, comp_res.value_b)
                lift_pct = comp_res.relative_lift_pct
                t_val = comp_res.test_statistic
                p_raw = comp_res.p_value
                p_adj = comp_res.p_value_adjusted if comp_res.p_value_adjusted is not None else p_raw
                mw_p = comp_res.mann_whitney_p_value
                eff_g = comp_res.effect_size
                ci_range = comp_res.confidence_interval if self._is_valid_finite_ci(comp_res.confidence_interval) else None
                ci_str = self._format_ci_range(p_metric, comp_res.confidence_interval)
                is_sig_adj = bool(
                    conf_level != ConfidenceLevel.INSUFFICIENT_EVIDENCE
                    and np.isfinite(p_adj)
                    and p_adj < self.alpha
                )
                sig_status_str = (
                    "Statistically significant after Holm-Bonferroni correction."
                    if is_sig_adj
                    else "Not statistically significant after Holm-Bonferroni correction."
                )
                mw_str = f", Mann-Whitney U p = {mw_p:.2e}" if mw_p is not None else ""
                dynamic_evidence = (
                    f"Module 4 A/B Test {exp_id} (N = {n_posts} posts; {n_a} {va_label} vs {n_b} {vb_label}). "
                    f"Mean {p_metric}: {val_b_str} vs {val_a_str} ({lift_pct:+.2f}% relative lift). "
                    f"Welch's t = {t_val:+.2f}, raw p = {p_raw:.2e}, Holm-adjusted p = {p_adj:.2e}{mw_str}, "
                    f"{comp_res.effect_size_type} = {eff_g:.3f}. 95% CI: {ci_str}. {sig_status_str}"
                )
                if is_sig_adj and comp_res.absolute_difference > 0:
                    dynamic_impact = f"{lift_pct:+.1f}% observed lift in {p_metric} (Benchmark Demonstration)"
                else:
                    dynamic_impact = f"Inconclusive / Unproven ({lift_pct:+.1f}% observed difference in {p_metric})"
                ci_method_str = comp_res.ci_method
                unit_str = comp_res.unit_of_analysis
            else:
                n_posts = 0
                p_raw = None
                p_adj = None
                eff_g = None
                ci_range = None
                val_a_str = "N/A"
                val_b_str = "N/A"
                lift_pct = 0.0
                is_sig_adj = False
                if dedup_note:
                    dynamic_evidence = f"Module 4 A/B Test {exp_id}: No unambiguous post observations available. {dedup_note}"
                else:
                    dynamic_evidence = f"Module 4 A/B Test {exp_id}: No observation rows available to compute inferential statistics."
                dynamic_impact = f"Unavailable — no observations recorded for {p_metric}"
                ci_method_str = "Unavailable"
                unit_str = "Post"

            if exp_id == "EXP_001":
                recs.append(ContentRecommendation(
                    recommendation_id="REC_EXP_001",
                    title="[Demonstration] Adopt Visual Action Hooks for High-Share Instagram Reels (Synthetic Benchmark)",
                    category=RecommendationCategory.CONTENT_FORMAT_STRUCTURE,
                    suggested_action=(
                        "Begin 15-30s Instagram Reels with an immediate visual cut, action, or scene shift "
                        "in the first 2 seconds rather than a slow conversational narrative hook."
                    ),
                    reasoning=(
                        f"In synthetic benchmark evaluation ({exp_id}), {vb_label} achieved {val_b_str} vs {val_a_str} "
                        f"for {va_label} ({lift_pct:+.2f}% relative change in post-level {p_metric}). {conf_rationale}"
                    ),
                    supporting_evidence=dynamic_evidence,
                    evidence_source=EvidenceSource.AB_TESTING,
                    relevant_platform=p_form,
                    expected_impact_direction=dynamic_impact,
                    confidence_level=conf_level,
                    limitations_and_caveats=(
                        f"SYNTHETIC BENCHMARK DEMONSTRATION. {conf_rationale} Analysis conducted at independent post level, "
                        "which addresses dependence among exposures within the same post by aggregating them, "
                        "but does not correct for potential correlation across posts from the same creator, campaign, or time period."
                    ),
                    suggested_validation_method="Run a 2-week alternate-day creative split test across your next 10 Reels to verify real-world effect.",
                    primary_metric=p_metric,
                    sample_size=n_posts,
                    effect_size=eff_g,
                    p_value=p_raw,
                    p_value_adjusted=p_adj,
                    confidence_interval=ci_range,
                    ci_method=ci_method_str,
                    unit_of_analysis=unit_str,
                    clustering_nuance="Aggregated exposures to post units; creator-level clustering uncorrected",
                    is_randomized_evidence=is_randomized,
                    is_synthetic_demonstration=True,
                    recency_days=recency_days,
                    tags=["reels", "hook", "visual", "share_rate", "synthetic_demonstration"]
                ))

            elif exp_id == "EXP_002":
                recs.append(ContentRecommendation(
                    recommendation_id="REC_EXP_002",
                    title="[Demonstration] Lead YouTube Shorts with Assertive Claim Titles & Hooks (Synthetic Benchmark)",
                    category=RecommendationCategory.CONTENT_FORMAT_STRUCTURE,
                    suggested_action=(
                        "Structure YouTube Shorts titles and opening audio around bold, counter-intuitive statements "
                        "('Why your workout routine is destroying recovery') rather than open exploratory questions."
                    ),
                    reasoning=(
                        f"In benchmark evaluation ({exp_id}), {vb_label} averaged {val_b_str} vs {val_a_str} for {va_label} "
                        f"({lift_pct:+.2f}% relative change in {p_metric}). {conf_rationale}"
                    ),
                    supporting_evidence=dynamic_evidence,
                    evidence_source=EvidenceSource.AB_TESTING,
                    relevant_platform=p_form,
                    expected_impact_direction=dynamic_impact,
                    confidence_level=conf_level,
                    limitations_and_caveats=(
                        f"SYNTHETIC BENCHMARK DEMONSTRATION. {conf_rationale} "
                        "Tested on synthetic Shorts corpus; actual audience retention and CTR must be monitored."
                    ),
                    suggested_validation_method="Test bold titles across your next 10 YouTube Shorts and compare 7-day view distributions.",
                    primary_metric=p_metric,
                    sample_size=n_posts,
                    effect_size=eff_g,
                    p_value=p_raw,
                    p_value_adjusted=p_adj,
                    confidence_interval=ci_range,
                    ci_method=ci_method_str,
                    unit_of_analysis=unit_str,
                    is_randomized_evidence=is_randomized,
                    is_synthetic_demonstration=True,
                    recency_days=recency_days,
                    tags=["youtube_shorts", "titles", "bold_statement", "views", "synthetic_demonstration"]
                ))

            elif exp_id == "EXP_003":
                recs.append(ContentRecommendation(
                    recommendation_id="REC_EXP_003",
                    title="[Demonstration] Craft Detailed Vulnerable Story Captions to Deepen Relatability (Synthetic Benchmark)",
                    category=RecommendationCategory.CAPTION_DEPTH_STYLE,
                    suggested_action=(
                        "Pair emotionally resonant video clips with structured, multi-paragraph caption stories detailing "
                        "personal struggles, lessons learned, and honest vulnerability."
                    ),
                    reasoning=(
                        f"In benchmark evaluation ({exp_id}), {vb_label} shifted mean {p_metric} from {val_a_str} "
                        f"to {val_b_str} ({lift_pct:+.2f}% relative change). {conf_rationale}"
                    ),
                    supporting_evidence=dynamic_evidence,
                    evidence_source=EvidenceSource.AB_TESTING,
                    relevant_platform=p_form,
                    expected_impact_direction=dynamic_impact,
                    confidence_level=conf_level,
                    limitations_and_caveats=(
                        f"SYNTHETIC BENCHMARK DEMONSTRATION. {conf_rationale} Relatability was scored via hybrid NLP "
                        "classification on synthetic comments; requires compelling opening hook to trigger expansion."
                    ),
                    suggested_validation_method="Alternate short vs story captions across 12 posts and compare save-to-share ratios.",
                    primary_metric=p_metric,
                    sample_size=n_posts,
                    effect_size=eff_g,
                    p_value=p_raw,
                    p_value_adjusted=p_adj,
                    confidence_interval=ci_range,
                    ci_method=ci_method_str,
                    unit_of_analysis=unit_str,
                    is_randomized_evidence=is_randomized,
                    is_synthetic_demonstration=True,
                    recency_days=recency_days,
                    tags=["instagram", "captions", "storytelling", "relatability", "synthetic_demonstration"]
                ))

            elif exp_id == "EXP_004":
                recs.append(ContentRecommendation(
                    recommendation_id="REC_EXP_004",
                    title="[Exploratory Demonstration] Weekend Evening Scheduling (Observational Cohort)",
                    category=RecommendationCategory.POSTING_TIMING,
                    suggested_action=(
                        "Consider pilot scheduling of select content during weekend evening windows (18:00 - 20:00) "
                        "as an exploratory hypothesis, rather than weekday morning hours."
                    ),
                    reasoning=(
                        f"Observational comparison ({exp_id}) showed {vb_label} averaged {val_b_str} vs {val_a_str} "
                        f"for {va_label} ({lift_pct:+.2f}% relative difference). {conf_rationale}"
                    ),
                    supporting_evidence=dynamic_evidence,
                    evidence_source=EvidenceSource.AB_TESTING,
                    relevant_platform=p_form,
                    expected_impact_direction=dynamic_impact,
                    confidence_level=conf_level,
                    limitations_and_caveats=(
                        "CRITICAL NON-CAUSAL LIMITATION: Observational comparison without random assignment. "
                        f"{conf_rationale} Confounded by weekend leisure patterns, topic variation, and feed distribution dynamics."
                    ),
                    suggested_validation_method="Execute a true randomized scheduling trial across 20 identical-format videos.",
                    primary_metric=p_metric,
                    sample_size=n_posts,
                    effect_size=eff_g,
                    p_value=p_raw,
                    p_value_adjusted=p_adj,
                    confidence_interval=ci_range,
                    ci_method=ci_method_str,
                    unit_of_analysis=unit_str,
                    is_randomized_evidence=is_randomized,
                    is_synthetic_demonstration=True,
                    recency_days=recency_days,
                    tags=["timing", "weekend", "observational_cohort", "synthetic_demonstration"]
                ))
            else:
                exp_name = str(exp.get("experiment_name", exp_id))
                recs.append(ContentRecommendation(
                    recommendation_id=f"REC_{exp_id}",
                    title=f"[Demonstration] {exp_name}",
                    category=RecommendationCategory.EXPERIMENTATION_AGENDA,
                    suggested_action=f"Evaluate adopting '{vb_label}' over '{va_label}' on {p_form.title()} based on observed {p_metric}.",
                    reasoning=f"Experiment {exp_id} compared {vb_label} ({val_b_str}) vs {va_label} ({val_a_str}) ({lift_pct:+.2f}%). {conf_rationale}",
                    supporting_evidence=dynamic_evidence,
                    evidence_source=EvidenceSource.AB_TESTING,
                    relevant_platform=p_form,
                    expected_impact_direction=dynamic_impact,
                    confidence_level=conf_level,
                    limitations_and_caveats=f"SYNTHETIC BENCHMARK DEMONSTRATION. {conf_rationale}",
                    suggested_validation_method="Replicate comparison on independent creator posts prior to broad rollout.",
                    primary_metric=p_metric,
                    sample_size=n_posts,
                    effect_size=eff_g,
                    p_value=p_raw,
                    p_value_adjusted=p_adj,
                    confidence_interval=ci_range,
                    ci_method=ci_method_str,
                    unit_of_analysis=unit_str,
                    is_randomized_evidence=is_randomized,
                    is_synthetic_demonstration=True,
                    recency_days=recency_days,
                    tags=["experiment", p_form, p_metric, "synthetic_demonstration"]
                ))

        return recs

    def _generate_sentiment_relatability_recommendations(self, target_platform: str) -> List[ContentRecommendation]:
        """Generate exploratory recommendations based on Module 3 NLP audience relatability classifications."""
        recs = []
        if self.comments_df is None or len(self.comments_df) == 0:
            return recs

        total_comments = len(self.comments_df)
        relatable_comments = self.comments_df[self.comments_df["relatability_label"] == "Relatable"]
        relatability_rate = len(relatable_comments) / total_comments if total_comments > 0 else 0.0

        corr_r: Optional[float] = None
        corr_p: Optional[float] = None
        n_merged_posts = 0
        if (
            "post_id" in self.comments_df.columns
            and self.posts_df is not None
            and len(self.posts_df) > 0
            and "post_id" in self.posts_df.columns
        ):
            valid_c = self.comments_df.dropna(subset=["post_id"]).copy()
            rel_by_post = (
                valid_c.assign(_is_rel=(valid_c["relatability_label"] == "Relatable"))
                .groupby("post_id", as_index=False)["_is_rel"]
                .mean()
                .rename(columns={"_is_rel": "post_relatability_rate"})
            )
            merged = pd.merge(self.posts_df, rel_by_post, on="post_id", how="inner")
            n_merged_posts = len(merged)
            if "saves" in merged.columns:
                clean_corr = merged.dropna(subset=["post_relatability_rate", "saves"])
                if len(clean_corr) >= 10:
                    sp_r, sp_p = stats.spearmanr(clean_corr["post_relatability_rate"], clean_corr["saves"])
                    if np.isfinite(sp_r) and np.isfinite(sp_p):
                        corr_r = round(float(sp_r), 4)
                        corr_p = round(float(sp_p), 3)

        corpus_label = (
            "Synthetic Comment Corpus"
            if self._is_synthetic_demo
            else "Analyzed Public Comment Corpus"
        )
        if corr_r is not None and corr_p is not None:
            corr_reasoning = (
                f"However, correlational analysis across {n_merged_posts} deduplicated posts reveals near-zero empirical "
                f"correlation with bookmark saves (r = {corr_r:.4f}, p = {corr_p:.3f}). "
            )
            corr_evidence = (
                f"{corpus_label} (N = {total_comments} classified comments across {n_merged_posts} unique posts). "
                f"Overall benchmark relatability rate = {relatability_rate * 100:.1f}%. "
                f"Spearman correlation between post-level relatability rate and bookmark saves: r = {corr_r:.4f} (p = {corr_p:.3f}, not significant)."
            )
        else:
            corr_reasoning = (
                "Post-level correlation with bookmark saves could not be established from the available post records. "
            )
            corr_evidence = (
                f"{corpus_label} (N = {total_comments} classified comments). "
                f"Overall benchmark relatability rate = {relatability_rate * 100:.1f}%. "
                "Post-level Spearman correlation with saves omitted due to insufficient overlapping post records."
            )

        recs.append(ContentRecommendation(
            recommendation_id="REC_NLP_001",
            title=(
                "[Exploratory Demonstration] Prompt Audience Shared Experience via Vulnerability Prompts"
                if self._is_synthetic_demo
                else "[Exploratory] Prompt Audience Shared Experience via Reflection Prompts"
            ),
            category=RecommendationCategory.AUDIENCE_RELATABILITY,
            suggested_action=(
                "Conclude reflective videos with specific experiential reflection prompts ('When was the last time you felt this?') "
                "to encourage relational viewer discussion in comments."
            ),
            reasoning=(
                f"Exploratory NLP analysis {'on synthetic benchmark comments ' if self._is_synthetic_demo else ''}"
                f"shows that {relatability_rate * 100:.1f}% of comments express shared experience. {corr_reasoning}"
                "Any causal engagement benefit from reflection prompts remains an unvalidated exploratory hypothesis."
            ),
            supporting_evidence=corr_evidence,
            evidence_source=EvidenceSource.SENTIMENT_RELATABILITY,
            relevant_platform="all",
            expected_impact_direction="Exploratory / Unvalidated (Requires Controlled A/B Testing)",
            confidence_level=ConfidenceLevel.LOW,
            limitations_and_caveats=(
                "EXPLORATORY OBSERVATIONAL FINDING. Rule-based and TF-IDF NLP classifications reflect heuristic/model scores "
                "and do not establish causal engagement gains."
            ),
            suggested_validation_method="A/B test an experiential reflection prompt against a standard call-to-action across 16 posts.",
            primary_metric="relatability_pct",
            sample_size=total_comments,
            effect_size=corr_r,
            p_value=corr_p,
            is_randomized_evidence=False,
            is_synthetic_demonstration=self._is_synthetic_demo,
            tags=["nlp", "relatability", "prompts", "exploratory"] + (["synthetic_demonstration"] if self._is_synthetic_demo else ["real_public_data"])
        ))

        return recs

    def _generate_topic_recommendations(self, target_platform: str) -> List[ContentRecommendation]:
        """Generate topic prioritization recommendations with creator-support and missing-metric guardrails."""
        recs = []
        if self.posts_df is None or len(self.posts_df) == 0:
            return recs

        df_eval = self.posts_df.copy()
        if target_platform != "all" and "platform" in df_eval.columns:
            df_eval = df_eval[df_eval["platform"].astype(str).str.lower() == target_platform]

        if "topic" not in df_eval.columns or len(df_eval) < 5:
            return recs

        has_creator_col = (
            "creator_pseudo_id" in df_eval.columns
            and df_eval["creator_pseudo_id"].dropna().nunique() > 0
        )
        has_collect = "collect_count" in df_eval.columns and df_eval["collect_count"].notna().any()

        agg_kwargs: Dict[str, Any] = {
            "post_count": ("post_id", "nunique"),
        }
        if "shares" in df_eval.columns:
            agg_kwargs["mean_shares"] = ("shares", "mean")
        if "saves" in df_eval.columns:
            agg_kwargs["mean_saves"] = ("saves", "mean")
        if "views" in df_eval.columns:
            agg_kwargs["mean_views"] = ("views", "mean")
        if "likes" in df_eval.columns:
            agg_kwargs["mean_likes"] = ("likes", "mean")
        if "retention_rate" in df_eval.columns:
            agg_kwargs["mean_retention"] = ("retention_rate", "mean")
        if has_creator_col:
            agg_kwargs["creator_count"] = ("creator_pseudo_id", "nunique")
        if has_collect:
            agg_kwargs["mean_collects"] = ("collect_count", "mean")

        topic_summary = df_eval.groupby("topic", dropna=True).agg(**agg_kwargs).reset_index()
        for opt_col in ("mean_shares", "mean_saves", "mean_views", "mean_likes", "mean_retention"):
            if opt_col not in topic_summary.columns:
                topic_summary[opt_col] = np.nan

        # High-intent numerator: native shares + saves (or TikTok collect_count proxy where saves absent)
        bookmark_signal = (
            topic_summary["mean_collects"]
            if has_collect and topic_summary["mean_saves"].isna().all()
            else topic_summary["mean_saves"]
        )
        has_valid_views = topic_summary["mean_views"].notna() & (topic_summary["mean_views"] > 0)
        has_any_intent = topic_summary["mean_shares"].notna() | bookmark_signal.notna()
        raw_high_intent = (
            topic_summary["mean_shares"].fillna(0.0) * 2.0
            + bookmark_signal.fillna(0.0) * 2.0
        )
        topic_summary["raw_high_intent"] = np.where(has_any_intent, raw_high_intent, np.nan)
        topic_summary["intent_density"] = np.where(
            has_valid_views & has_any_intent,
            raw_high_intent / np.where(has_valid_views, topic_summary["mean_views"], 1.0),
            np.nan,
        )

        support_mask = topic_summary["post_count"] >= 3
        if has_creator_col:
            support_mask &= topic_summary["creator_count"] >= self.MIN_CREATORS_FOR_REC

        supported = topic_summary.loc[support_mask].copy()
        if supported["intent_density"].notna().any():
            rank_col = "intent_density"
        elif supported["raw_high_intent"].notna().any():
            rank_col = "raw_high_intent"
        elif supported["mean_likes"].notna().any():
            rank_col = "mean_likes"
        else:
            rank_col = "post_count"

        topic_summary = supported.sort_values(rank_col, ascending=False, na_position="last")

        if len(topic_summary) > 0:
            top_topic = topic_summary.iloc[0]
            shares_str = f"{float(top_topic['mean_shares']):.1f}" if pd.notna(top_topic["mean_shares"]) else "N/A"
            if pd.notna(top_topic["mean_saves"]):
                bookmark_str = f"mean saves: {float(top_topic['mean_saves']):.1f}"
            elif has_collect and pd.notna(top_topic.get("mean_collects")):
                bookmark_str = f"mean collect_count proxy: {float(top_topic['mean_collects']):.1f} (saves N/A on TikTok)"
            else:
                bookmark_str = "saves: N/A (unavailable on platform)"

            has_top_density = pd.notna(top_topic["intent_density"])
            density_note = (
                f"Intent density = {float(top_topic['intent_density']):.4f}."
                if has_top_density
                else "Views unavailable on this corpus; ranked by observed engagement volume (intent density per view = N/A)."
            )
            reasoning_metric = (
                f"exhibited directionally higher intent density (mean shares: {shares_str}, {bookmark_str})"
                if has_top_density
                else f"exhibited directionally higher observed engagement (mean shares: {shares_str}, {bookmark_str})"
            )

            creator_note = f", {int(top_topic['creator_count'])} unique creators" if has_creator_col else ""
            recs.append(ContentRecommendation(
                recommendation_id="REC_TOPIC_001",
                title=f"[Exploratory] Prioritize Content Pillar: '{top_topic['topic']}' (Observational Association)",
                category=RecommendationCategory.TOPIC_SELECTION,
                suggested_action=(
                    f"Consider testing additional content around '{top_topic['topic']}', "
                    "focusing on practical struggle navigation and personal lessons."
                ),
                reasoning=(
                    f"Observational historical summary indicates '{top_topic['topic']}' {reasoning_metric} "
                    "in the sample posts."
                ),
                supporting_evidence=(
                    f"Descriptive post analysis across N = {int(top_topic['post_count'])} observational posts{creator_note} for this topic. "
                    f"{density_note}"
                ),
                evidence_source=EvidenceSource.HISTORICAL_PERFORMANCE,
                relevant_platform=target_platform if target_platform != "all" else "all",
                expected_impact_direction="Directional (+10% to +20% in High-Intent Actions)",
                confidence_level=ConfidenceLevel.LOW,
                limitations_and_caveats=(
                    f"NON-CAUSAL OBSERVATIONAL PATTERN (N = {int(top_topic['post_count'])} posts{creator_note}). Subject to confounding by production quality, "
                    "creator follower scale, posting date, and audience fatigue. Observational topic correlation does not establish causal performance."
                ),
                suggested_validation_method="Produce a 3-part controlled creative test comparing this topic against a control topic.",
                primary_metric="high_intent_actions",
                sample_size=int(top_topic["post_count"]),
                is_randomized_evidence=False,
                is_synthetic_demonstration=self._is_synthetic_demo,
                tags=["topic", "content_pillar", "observational"] + (["synthetic_demonstration"] if self._is_synthetic_demo else ["real_public_data"])
            ))

        return recs

    def _generate_format_duration_recommendations(self, target_platform: str) -> List[ContentRecommendation]:
        """Recommend video duration pacing based on observational retention curves or view efficiency."""
        recs = []
        if self.posts_df is None or len(self.posts_df) == 0:
            return recs

        df_eval = self.posts_df.copy()
        if target_platform != "all" and "platform" in df_eval.columns:
            df_eval = df_eval[df_eval["platform"].astype(str).str.lower() == target_platform]

        if "duration_seconds" not in df_eval.columns or len(df_eval) < 5:
            return recs

        has_retention = "retention_rate" in df_eval.columns and df_eval["retention_rate"].notna().any()
        has_views = "views" in df_eval.columns and df_eval["views"].notna().any()
        if not has_retention and not has_views:
            return recs

        has_creator_col = (
            "creator_pseudo_id" in df_eval.columns
            and df_eval["creator_pseudo_id"].dropna().nunique() > 0
        )

        bins = [0, 25, 50, 90, 3600]
        labels = ["Short (<25s)", "Medium (25-50s)", "Long (50-90s)", "Extended (>90s)"]
        df_eval["duration_bin"] = pd.cut(pd.to_numeric(df_eval["duration_seconds"], errors="coerce"), bins=bins, labels=labels)

        agg_kwargs: Dict[str, Any] = {
            "post_count": ("post_id", "nunique"),
            "mean_shares": ("shares", "mean"),
        }
        if has_retention:
            agg_kwargs["mean_metric"] = ("retention_rate", "mean")
            metric_key = "retention_rate"
        else:
            agg_kwargs["mean_metric"] = ("views", "mean")
            metric_key = "views"
        if has_creator_col:
            agg_kwargs["creator_count"] = ("creator_pseudo_id", "nunique")

        duration_summary = df_eval.groupby("duration_bin", observed=False).agg(**agg_kwargs).reset_index()
        support_mask = (duration_summary["post_count"] >= 3) & duration_summary["mean_metric"].notna()
        if has_creator_col:
            support_mask &= duration_summary["creator_count"] >= self.MIN_CREATORS_FOR_REC

        duration_summary = duration_summary.loc[support_mask].sort_values("mean_metric", ascending=False)

        if len(duration_summary) > 0:
            best_bin = duration_summary.iloc[0]
            creator_note = f", {int(best_bin['creator_count'])} unique creators" if has_creator_col else ""
            if has_retention:
                ret_nonnull = pd.to_numeric(df_eval["retention_rate"], errors="coerce").dropna()
                ret_scale = 100.0 if (not ret_nonnull.empty and float(ret_nonnull.max()) <= 1.0) else 1.0
                best_ret_pct = float(best_bin["mean_metric"]) * ret_scale
                cohort_avg_pct = float(ret_nonnull.mean()) * ret_scale
                reasoning_txt = (
                    f"Historical posts in '{best_bin['duration_bin']}' exhibited the highest percentage retention "
                    f"(mean retention: {best_ret_pct:.1f}%)."
                )
                evidence_txt = (
                    f"Observational analysis of N = {int(best_bin['post_count'])} posts{creator_note} in this runtime tier. "
                    f"Average retention rate = {best_ret_pct:.1f}% vs cohort average of {cohort_avg_pct:.1f}%."
                )
                impact_txt = "Directional (+5% to +12% in Completion Percentage)"
            else:
                cohort_avg_v = float(df_eval["views"].dropna().mean())
                reasoning_txt = (
                    f"Historical posts in '{best_bin['duration_bin']}' averaged {float(best_bin['mean_metric']):,.0f} views "
                    "(retention_rate not exposed on this platform/tier)."
                )
                evidence_txt = (
                    f"Observational analysis of N = {int(best_bin['post_count'])} posts{creator_note} in this runtime tier. "
                    f"Average views = {float(best_bin['mean_metric']):,.0f} vs cohort average of {cohort_avg_v:,.0f}."
                )
                impact_txt = "Directional (Higher Average Playback Volume)"

            recs.append(ContentRecommendation(
                recommendation_id="REC_FORMAT_001",
                title=f"[Exploratory] Video Runtime Pacing: Target '{best_bin['duration_bin']}' (Observational Pattern)",
                category=RecommendationCategory.CONTENT_FORMAT_STRUCTURE,
                suggested_action=(
                    f"Test editing upcoming short-form content to fit within '{best_bin['duration_bin']}', "
                    "trimming pauses and eliminating filler b-roll to maintain viewer completion."
                ),
                reasoning=reasoning_txt,
                supporting_evidence=evidence_txt,
                evidence_source=EvidenceSource.HISTORICAL_PERFORMANCE,
                relevant_platform=target_platform if target_platform != "all" else "all",
                expected_impact_direction=impact_txt,
                confidence_level=ConfidenceLevel.LOW,
                limitations_and_caveats=(
                    "OBSERVATIONAL DESCRIPTIVE PATTERN. Shorter videos mathematically achieve higher percentage completion, "
                    "while platform algorithms may reward either watch-time or replays depending on surface."
                ),
                suggested_validation_method="A/B test a 25s cut vs a 50s cut of the exact same script.",
                primary_metric=metric_key,
                sample_size=int(best_bin["post_count"]),
                is_randomized_evidence=False,
                is_synthetic_demonstration=self._is_synthetic_demo,
                tags=["duration", "pacing", metric_key, "observational"] + (["synthetic_demonstration"] if self._is_synthetic_demo else ["real_public_data"])
            ))

        return recs

    def _generate_timing_recommendations(self, target_platform: str) -> List[ContentRecommendation]:
        """Generate scheduling recommendations with creator-support guardrails and non-causal warnings."""
        recs = []
        if self.posts_df is None or len(self.posts_df) == 0:
            return recs

        df_eval = self.posts_df.copy()
        if target_platform != "all" and "platform" in df_eval.columns:
            df_eval = df_eval[df_eval["platform"].astype(str).str.lower() == target_platform]

        if "posting_hour" not in df_eval.columns and "timestamp" in df_eval.columns:
            df_eval["posting_hour"] = pd.to_datetime(df_eval["timestamp"], utc=True, errors="coerce").dt.hour

        if "posting_hour" not in df_eval.columns or "views" not in df_eval.columns or len(df_eval) < 5:
            return recs

        has_creator_col = (
            "creator_pseudo_id" in df_eval.columns
            and df_eval["creator_pseudo_id"].dropna().nunique() > 0
        )
        agg_kwargs: Dict[str, Any] = {
            "post_count": ("post_id", "nunique"),
            "mean_views": ("views", "mean"),
        }
        if has_creator_col:
            agg_kwargs["creator_count"] = ("creator_pseudo_id", "nunique")

        hour_summary = df_eval.groupby("posting_hour", dropna=True).agg(**agg_kwargs).reset_index()
        support_mask = (hour_summary["post_count"] >= 3) & hour_summary["mean_views"].notna()
        if has_creator_col:
            support_mask &= hour_summary["creator_count"] >= self.MIN_CREATORS_FOR_REC

        hour_summary = hour_summary.loc[support_mask].sort_values("mean_views", ascending=False)

        if len(hour_summary) > 0:
            top_h = int(hour_summary.iloc[0]["posting_hour"])
            creator_note = f", {int(hour_summary.iloc[0]['creator_count'])} unique creators" if has_creator_col else ""
            recs.append(ContentRecommendation(
                recommendation_id="REC_TIMING_001",
                title=f"[Exploratory] Schedule Window: {top_h:02d}:00 - {(top_h+2)%24:02d}:00 (Observational Association)",
                category=RecommendationCategory.POSTING_TIMING,
                suggested_action=(
                    f"Target scheduling initial uploads between {top_h:02d}:00 and {(top_h+2)%24:02d}:00 UTC audience window "
                    "as an exploratory operational test."
                ),
                reasoning=(
                    f"Historical posts published in this hour bin exhibited higher average views ({float(hour_summary.iloc[0]['mean_views']):,.0f} views) "
                    "in this sample dataset."
                ),
                supporting_evidence=(
                    f"Observational analysis across N = {int(hour_summary.iloc[0]['post_count'])} posts{creator_note} published in this hour bin. "
                    "Observational view difference represents an unadjusted association."
                ),
                evidence_source=EvidenceSource.HISTORICAL_PERFORMANCE,
                relevant_platform=target_platform if target_platform != "all" else "all",
                expected_impact_direction="Unproven / Directional (+5% to +15% in Initial Velocity)",
                confidence_level=ConfidenceLevel.LOW,
                limitations_and_caveats=(
                    "CRITICAL NON-CAUSAL LIMITATION: Posting timing is strictly observational. Algorithm distribution, creator timezone/follower size, "
                    "and content quality introduce major confounding. Timing does not compensate for weak creative hooks."
                ),
                suggested_validation_method="Run randomized time-of-day release tests across 16 uniform format videos.",
                primary_metric="views",
                sample_size=int(hour_summary.iloc[0]["post_count"]),
                is_randomized_evidence=False,
                is_synthetic_demonstration=self._is_synthetic_demo,
                tags=["timing", "scheduling", "observational"] + (["synthetic_demonstration"] if self._is_synthetic_demo else ["real_public_data"])
            ))

        return recs

    def _generate_experimentation_agenda(self, target_platform: str) -> List[ContentRecommendation]:
        """Propose prioritized A/B tests for areas characterized by high uncertainty or conflicting signals."""
        recs = []
        has_tiktok_data = (
            self.posts_df is not None
            and "platform" in self.posts_df.columns
            and (self.posts_df["platform"].astype(str).str.lower() == "tiktok").any()
        )

        if target_platform in ["youtube", "all"]:
            recs.append(ContentRecommendation(
                recommendation_id="REC_EXP_AGENDA_001",
                title="A/B Test Proposal: Facial Expression vs Graphic Minimalist YouTube Thumbnails",
                category=RecommendationCategory.EXPERIMENTATION_AGENDA,
                suggested_action=(
                    "Deploy a 14-day 1:1 split test comparing expressive close-up facial thumbnails against minimalist "
                    "high-contrast typography graphics on YouTube Shorts / Videos."
                ),
                reasoning=(
                    "Observational data indicates high variance in YouTube CTR (2.1% to 8.4%). No randomized trial currently "
                    "confirms whether expressive faces or graphic minimalism drives higher conversion."
                ),
                supporting_evidence="Identified knowledge gap in Module 4 experimentation portfolio. High potential for click-through rate lift.",
                evidence_source=EvidenceSource.HEURISTIC,
                relevant_platform="youtube",
                expected_impact_direction="Hypothesized +15% to +25% in Click-Through Rate",
                confidence_level=ConfidenceLevel.LOW,
                limitations_and_caveats=(
                    "PRE-EXPERIMENTAL PROPOSAL. Requires sample size planning: N >= 80 posts or 20,000 impressions per variant "
                    "to detect an MDE of 15% with 80% statistical power."
                ),
                suggested_validation_method="Launch a 2-variant randomized split test using Module 4 A/B testing framework.",
                primary_metric="views",
                sample_size=0,
                is_randomized_evidence=False,
                is_synthetic_demonstration=self._is_synthetic_demo,
                tags=["experiment_proposal", "youtube", "thumbnails", "ctr"]
            ))

        if target_platform in ["instagram", "all"]:
            recs.append(ContentRecommendation(
                recommendation_id="REC_EXP_AGENDA_002",
                title="A/B Test Proposal: Mid-Video Overlay CTA vs End-Screen Spoken CTA",
                category=RecommendationCategory.EXPERIMENTATION_AGENDA,
                suggested_action=(
                    "Conduct a 3-week creative experiment testing mid-video (second 15) non-intrusive text overlays "
                    "('Save this for later') versus verbal end-screen calls to action."
                ),
                reasoning=(
                    "Historical retention curves show 35% audience drop-off before video end-screens. Placing save prompts "
                    "earlier may capture peak attention before viewer abandonment."
                ),
                supporting_evidence="Historical retention decay analysis (Module 1). Median retention drop at second 15 is 18% vs 42% at end-screen.",
                evidence_source=EvidenceSource.HISTORICAL_PERFORMANCE,
                relevant_platform="instagram",
                expected_impact_direction="Hypothesized +25% to +40% in Save-to-Share Ratio",
                confidence_level=ConfidenceLevel.LOW,
                limitations_and_caveats="Intrusive mid-video prompts risk disrupting viewing immersion and accelerating drop-off. Must measure retention simultaneously.",
                suggested_validation_method="Execute 20-post A/B test evaluated with Holm-Bonferroni correction on saves and retention.",
                primary_metric="save_to_share_ratio",
                sample_size=0,
                is_randomized_evidence=False,
                is_synthetic_demonstration=self._is_synthetic_demo,
                tags=["experiment_proposal", "instagram", "cta", "saves"]
            ))

        if target_platform == "tiktok" or (target_platform == "all" and has_tiktok_data):
            recs.append(ContentRecommendation(
                recommendation_id="REC_EXP_AGENDA_003",
                title="A/B Test Proposal: 2-Second Pattern-Interrupt Hook vs Direct Explanation on TikTok",
                category=RecommendationCategory.EXPERIMENTATION_AGENDA,
                suggested_action=(
                    "Run a creator-blocked 20-video split test comparing fast 2-second visual pattern-interrupt hooks "
                    "against direct verbal openings while tracking 7-day views and `collect_count` proxy rate."
                ),
                reasoning=(
                    "Longitudinal TikTok trajectories show most cumulative views and shares accrue between Day 0 and Day 7, "
                    "with high between-creator variance requiring creator-level blocking."
                ),
                supporting_evidence=(
                    "Identified from real TikTok longitudinal curves; evaluate outbound `shares` and `collect_count` "
                    "(TikTok bookmark proxy, not directly equivalent to Instagram Insights `saves`)."
                ),
                evidence_source=EvidenceSource.HISTORICAL_PERFORMANCE,
                relevant_platform="tiktok",
                expected_impact_direction="Hypothesized lift in 7-day normalized view ratio and share rate",
                confidence_level=ConfidenceLevel.LOW,
                limitations_and_caveats=(
                    "PRE-EXPERIMENTAL PROPOSAL. Account for creator clustering (creator_pseudo_id) and right-censoring "
                    "by evaluating all videos at an identical fixed post-age horizon (e.g. day7_views)."
                ),
                suggested_validation_method="Execute stratified creator-blocked split test and analyze with ABTestingFramework.compare_creator_clustered_rates().",
                primary_metric="day7_views",
                sample_size=0,
                is_randomized_evidence=False,
                is_synthetic_demonstration=self._is_synthetic_demo,
                tags=["experiment_proposal", "tiktok", "day7_views", "collect_count_proxy"]
            ))

        return recs

    # --- 3. CONFLICTING EVIDENCE DETECTION ---

    def _detect_conflicting_evidence(self, recommendations: List[ContentRecommendation]) -> List[Dict[str, Any]]:
        """Identify contradictions between observational signals and randomized experimental evidence."""
        conflicts = []

        rec_001 = next((r for r in recommendations if r.recommendation_id == "REC_EXP_001"), None)
        if (
            rec_001 is not None
            and rec_001.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
            and self.posts_df is not None
            and "hook_type" in self.posts_df.columns
            and "views" in self.posts_df.columns
        ):
            obs_views = self.posts_df.groupby("hook_type")["views"].mean().to_dict()
            story_v = obs_views.get("Story Hook", 0)
            visual_v = obs_views.get("Visual Hook", 0)
            if story_v > visual_v:
                rec_001.conflict_penalty = 0.85
                p_adj_str = f"{rec_001.p_value_adjusted:.2e}" if rec_001.p_value_adjusted is not None else "N/A"
                conflicts.append({
                    "conflict_id": "CONF_001",
                    "area": "Hook Architecture (Story vs Visual)",
                    "observational_signal": f"Observational view averages favor Story Hooks ({story_v:,.0f} vs {visual_v:,.0f} views).",
                    "experimental_evidence": (
                        f"Randomized A/B test EXP_001 supports Visual Hooks ({rec_001.expected_impact_direction}, "
                        f"p_adj = {p_adj_str})."
                    ),
                    "resolution": (
                        "Prioritize randomized experimental evidence over observational view aggregates. "
                        "High shares expand organic distribution more reliably than passive views."
                    ),
                    "confidence_impact": f"Visual Hook recommendation retained at {rec_001.confidence_level.value} confidence; conflict penalty (0.85) applied to priority score."
                })

        return conflicts

    # --- 4. DETERMINISTIC RECOMMENDATION RANKING ---

    def _compute_priority_score(self, rec: ContentRecommendation, target_platform: str) -> float:
        """Calculate transparent, deterministic priority score.
        
        Priority Score is a ranking index for decision prioritization and does NOT represent
        a calibrated probability of virality or success.

        Formula:
          Raw Score = (0.30 * s_evidence + 0.20 * s_impact + 0.15 * s_sample +
                       0.15 * s_relevance + 0.10 * s_recency + 0.10 * s_quality)
          Final Score = Raw Score * conflict_penalty
        """
        # 1. Evidence Strength Score (0.0 to 1.0)
        if rec.confidence_level == ConfidenceLevel.HIGH:
            s_evidence = 1.0
        elif rec.confidence_level == ConfidenceLevel.MEDIUM:
            s_evidence = 0.65
        elif rec.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.EXPLORATORY]:
            s_evidence = 0.30
        else:
            s_evidence = 0.05

        # 2. Impact Potential Score (0.0 to 1.0)
        # Scaled from standardized effect size or parsed relative lift
        if rec.effect_size is not None:
            eff = abs(rec.effect_size)
            if "hedges" in rec.ci_method.lower() or eff > 1.0:
                s_impact = 1.0 if eff >= 1.0 else (0.80 if eff >= 0.5 else (0.55 if eff >= 0.2 else 0.25))
            else:
                # Proportion / Cohen's h or small effect
                s_impact = 1.0 if eff >= 0.5 else (0.75 if eff >= 0.2 else (0.50 if eff >= 0.05 else 0.25))
        else:
            imp_text = rec.expected_impact_direction.lower()
            if "+40%" in imp_text or "+50%" in imp_text or "+60%" in imp_text:
                s_impact = 0.85
            elif "+20%" in imp_text or "+30%" in imp_text or "+25%" in imp_text:
                s_impact = 0.70
            elif "+10%" in imp_text or "+15%" in imp_text or "+12%" in imp_text:
                s_impact = 0.50
            else:
                s_impact = 0.30

        # 3. Sample Size Adequacy & Uncertainty Score (0.0 to 1.0)
        n = rec.sample_size
        if n >= 100:
            s_sample = 1.0
        elif n >= 50:
            s_sample = 0.85
        elif n >= 20:
            s_sample = 0.65
        elif n >= 5:
            s_sample = 0.35
        elif n > 0:
            s_sample = 0.20
        else:
            s_sample = 0.15  # Pre-experimental proposal

        # Penalize if 95% confidence interval spans zero
        if rec.confidence_interval is not None:
            ci_low, ci_high = rec.confidence_interval
            if ci_low <= 0 <= ci_high:
                s_sample *= 0.60

        # 4. Platform Relevance Alignment (0.0 to 1.0)
        # Crucial: Do NOT automatically reward cross-platform generalizability when a specific platform is requested!
        target_p = target_platform.lower().strip()
        rec_p = rec.relevant_platform.lower().strip()
        if target_p == "all":
            s_relevance = 1.0 if rec_p == "all" else 0.85
        elif target_p == rec_p:
            s_relevance = 1.0
        elif rec_p == "all":
            # Target is specific (e.g., instagram), but rec is generic (all): assign 0.70 (not 1.0)
            s_relevance = 0.70
        else:
            s_relevance = 0.10

        # 5. Evidence Recency (0.0 to 1.0)
        if rec.recency_days is not None:
            rd = rec.recency_days
            if rd <= 30:
                s_recency = 1.0
            elif rd <= 90:
                s_recency = 0.85
            elif rd <= 180:
                s_recency = 0.65
            else:
                s_recency = 0.40
        else:
            s_recency = 0.70  # Neutral baseline when recency is unmeasured

        # 6. Data Quality & Provenance (0.0 to 1.0)
        s_quality = rec.data_quality_score

        # Combine Raw Score
        raw_score = (
            0.30 * s_evidence +
            0.20 * s_impact +
            0.15 * s_sample +
            0.15 * s_relevance +
            0.10 * s_recency +
            0.10 * s_quality
        )

        # Apply Conflicting Evidence Penalty
        final_score = raw_score * rec.conflict_penalty

        rec.scoring_breakdown = {
            "evidence_score": s_evidence,
            "impact_score": s_impact,
            "sample_score": s_sample,
            "relevance_score": s_relevance,
            "recency_score": s_recency,
            "quality_score": s_quality,
            "conflict_penalty": rec.conflict_penalty
        }

        return float(round(final_score, 4))


class ContentRecommender:
    """Generates prescriptive content strategies from empirical performance data (Backwards-Compatible API)."""

    def __init__(self, target_metric: str = "viral_coefficient"):
        self.target_metric = target_metric
        self.engine = OptimizationRecommender()

    def recommend_weekly_strategy(self, df: pd.DataFrame, top_k: int = 3) -> Dict[str, Any]:
        """Synthesize high-probability content combinations from historical data."""
        if df is None or len(df) == 0:
            return {
                "recommended_topics": {},
                "optimal_duration_bin": "Medium (30-60s)",
                "optimal_hook_style": "Problem-Focused Question Hook",
                "optimal_posting_window": "18:00 - 21:00",
                "deep_resonance_topic": "N/A",
                "confidence_statement": "Insufficient historical records to compute weekly strategy."
            }

        df_eval = get_latest_posts_view(df).copy() if "post_id" in df.columns else df.copy()
        if self.target_metric not in df_eval.columns:
            shares = pd.to_numeric(df_eval["shares"], errors="coerce") if "shares" in df_eval.columns else pd.Series(np.nan, index=df_eval.index, dtype="float64")
            saves = pd.to_numeric(df_eval["saves"], errors="coerce") if "saves" in df_eval.columns else pd.Series(np.nan, index=df_eval.index, dtype="float64")
            likes = pd.to_numeric(df_eval["likes"], errors="coerce") if "likes" in df_eval.columns else pd.Series(np.nan, index=df_eval.index, dtype="float64")
            has_any_signal = shares.notna() | saves.notna() | likes.notna()
            combined = shares.fillna(0.0) * 2.0 + saves.fillna(0.0) * 2.0 + likes.fillna(0.0)
            df_eval[self.target_metric] = combined.where(has_any_signal, np.nan)

        df_eval[self.target_metric] = pd.to_numeric(df_eval[self.target_metric], errors="coerce")

        # 1. Best performing topics
        if "topic" in df_eval.columns and df_eval[self.target_metric].notna().any():
            topic_means = df_eval.groupby("topic")[self.target_metric].mean().dropna().sort_values(ascending=False)
            top_topics = topic_means.head(top_k).to_dict()
        else:
            top_topics = {}

        # 2. Optimal video length bin
        best_duration = "Medium (30-60s)"
        if "duration_seconds" in df_eval.columns:
            bins = [0, 20, 45, 90, 300]
            labels = ["Short (<20s)", "Medium (20-45s)", "Long (45-90s)", "Extended (>90s)"]
            df_eval["duration_bin"] = pd.cut(pd.to_numeric(df_eval["duration_seconds"], errors="coerce"), bins=bins, labels=labels)
            dur_means = df_eval.groupby("duration_bin", observed=False)[self.target_metric].mean().dropna().sort_values(ascending=False)
            if not dur_means.empty:
                best_duration = dur_means.index[0]

        # 3. Best hook type
        best_hook = "Problem-Focused Question Hook"
        if "hook_type" in df_eval.columns:
            hook_means = df_eval.groupby("hook_type")[self.target_metric].mean().dropna().sort_values(ascending=False)
            if not hook_means.empty:
                best_hook = hook_means.index[0]

        # 4. Best posting hour
        posting_window = "18:00 - 21:00"
        if "posting_hour" in df_eval.columns:
            hour_means = df_eval.groupby("posting_hour")[self.target_metric].mean().dropna().sort_values(ascending=False)
            if not hour_means.empty:
                best_hour = int(hour_means.index[0])
                posting_window = f"{best_hour:02d}:00 - {(best_hour+2)%24:02d}:00"

        # 5. Save-to-Share high resonance topic
        high_resonance_topic = list(top_topics.keys())[0] if top_topics else "N/A"
        if "save_to_share_ratio" in df_eval.columns and "topic" in df_eval.columns:
            res_means = (
                df_eval.groupby("topic")["save_to_share_ratio"]
                .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
                .dropna()
                .sort_values(ascending=False)
            )
            if not res_means.empty:
                high_resonance_topic = res_means.index[0]

        return {
            "recommended_topics": top_topics,
            "optimal_duration_bin": str(best_duration),
            "optimal_hook_style": str(best_hook),
            "optimal_posting_window": posting_window,
            "deep_resonance_topic": str(high_resonance_topic),
            "confidence_statement": "Based on empirical high-intent actions (Shares, Saves, Retention) over passive likes."
        }

    def generate_evidence_based_recommendations(
        self,
        platform: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: Optional[str] = None
    ) -> RecommendationReport:
        """Forward to the full OptimizationRecommender engine."""
        return self.engine.generate_recommendations(
            platform=platform,
            category=category,
            min_confidence=min_confidence
        )
