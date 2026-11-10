"""Comprehensive Test Suite for Module 5: Engagement Optimization Recommender.

Verifies:
1. Recommendation generation across format, topic, caption, relatability, timing, and experimentation.
2. Deterministic recommendation ranking by priority score.
3. Handling of missing, null, and empty datasets without runtime crashes.
4. Conflicting evidence detection and analytical resolution.
5. Insufficient sample size states and defensive fallbacks.
6. Platform filtering (Instagram, YouTube, Cross-platform).
7. Category filtering across recommendation types.
8. Evidence-source attribution (A/B testing, historical, NLP relatability, heuristic).
9. Confidence level classification (HIGH, MEDIUM, LOW) according to evidence hierarchy.
10. Observational versus randomized evidence distinction (non-causal warnings).
11. Backwards compatibility with the legacy ContentRecommender API.
12. Synthetic benchmark demonstration notice preservation.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.recommender import ContentRecommender, OptimizationRecommender
from src.recommender.schemas import (
    ConfidenceLevel,
    ContentRecommendation,
    EvidenceSource,
    RecommendationCategory,
    RecommendationReport
)


@pytest.fixture
def sample_datasets():
    """Load processed test fixtures for Module 5 integration testing."""
    root_dir = Path(__file__).resolve().parent.parent
    posts_path = root_dir / "data" / "processed" / "processed_posts.parquet"
    comments_path = root_dir / "data" / "processed" / "processed_comments.parquet"
    exp_path = root_dir / "data" / "sample" / "sample_experiments.csv"
    obs_path = root_dir / "data" / "processed" / "processed_ab_tests.parquet"

    posts_df = pd.read_parquet(posts_path) if posts_path.exists() else None
    comments_df = pd.read_parquet(comments_path) if comments_path.exists() else None
    exp_df = pd.read_csv(exp_path) if exp_path.exists() else None
    obs_df = pd.read_parquet(obs_path) if obs_path.exists() else None

    return {
        "posts_df": posts_df,
        "comments_df": comments_df,
        "experiments_df": exp_df,
        "observations_df": obs_df
    }


@pytest.fixture
def recommender(sample_datasets):
    return OptimizationRecommender(
        posts_df=sample_datasets["posts_df"],
        comments_df=sample_datasets["comments_df"],
        experiments_df=sample_datasets["experiments_df"],
        observations_df=sample_datasets["observations_df"]
    )


# --- 1. Recommendation Generation & Report Structure ---

def test_generate_recommendations_structure(recommender):
    """Verify recommendation report contains valid structured recommendations with all required fields."""
    report = recommender.generate_recommendations()
    assert isinstance(report, RecommendationReport)
    assert report.total_recommendations > 0
    assert len(report.recommendations) == report.total_recommendations
    assert report.is_synthetic_demonstration is True

    # Validate schema of every generated recommendation
    for rec in report.recommendations:
        assert isinstance(rec, ContentRecommendation)
        assert rec.recommendation_id.startswith("REC_")
        assert len(rec.title) > 5
        assert isinstance(rec.category, RecommendationCategory)
        assert len(rec.suggested_action) > 10
        assert len(rec.reasoning) > 10
        assert len(rec.supporting_evidence) > 10
        assert isinstance(rec.evidence_source, EvidenceSource)
        assert rec.relevant_platform in ["instagram", "youtube", "all"]
        assert len(rec.expected_impact_direction) > 0
        assert isinstance(rec.confidence_level, ConfidenceLevel)
        assert len(rec.limitations_and_caveats) > 10
        assert len(rec.suggested_validation_method) > 10
        assert 0.0 <= rec.priority_score <= 1.0


# --- 2. Deterministic Ranking Mechanism ---

def test_deterministic_recommendation_ranking(recommender):
    """Verify recommendation ranking is strictly descending by priority score and fully reproducible."""
    report1 = recommender.generate_recommendations()
    report2 = recommender.generate_recommendations()

    # Reproducibility check
    assert [r.recommendation_id for r in report1.recommendations] == [r.recommendation_id for r in report2.recommendations]
    assert [r.priority_score for r in report1.recommendations] == [r.priority_score for r in report2.recommendations]

    # Monotonic descending order check
    scores = [r.priority_score for r in report1.recommendations]
    assert scores == sorted(scores, reverse=True)


# --- 3. Confidence Classification & Evidence Hierarchy ---

def test_confidence_level_classification_hierarchy(recommender):
    """Verify evidence hierarchy: Randomized trials are HIGH confidence; observational timing is LOW/MEDIUM."""
    report = recommender.generate_recommendations()

    high_recs = [r for r in report.recommendations if r.confidence_level == ConfidenceLevel.HIGH]
    assert len(high_recs) >= 3

    # Randomized A/B test recommendations must be marked HIGH confidence with randomized flag
    for hr in high_recs:
        if hr.evidence_source == EvidenceSource.AB_TESTING and hr.recommendation_id in ["REC_EXP_001", "REC_EXP_002", "REC_EXP_003"]:
            assert hr.is_randomized_evidence is True
            assert hr.p_value is not None
            assert hr.p_value < 0.001

    # Observational timing comparison (EXP_004) must be LOW confidence with non-causal caveats
    timing_rec = next((r for r in report.recommendations if r.recommendation_id == "REC_EXP_004"), None)
    if timing_rec:
        assert timing_rec.confidence_level == ConfidenceLevel.LOW
        assert timing_rec.is_randomized_evidence is False
        assert "non-randomized" in timing_rec.limitations_and_caveats.lower() or "observational" in timing_rec.limitations_and_caveats.lower()


# --- 4. Platform Filtering ---

def test_platform_filtering(recommender):
    """Verify platform filter restricts recommendations to target platform and cross-platform suggestions."""
    ig_report = recommender.generate_recommendations(platform="instagram")
    for r in ig_report.recommendations:
        assert r.relevant_platform in ["instagram", "all"]

    yt_report = recommender.generate_recommendations(platform="youtube")
    for r in yt_report.recommendations:
        assert r.relevant_platform in ["youtube", "all"]


# --- 5. Category Filtering ---

def test_category_filtering(recommender):
    """Verify category filter isolates specific recommendation domains."""
    cat_report = recommender.generate_recommendations(category=RecommendationCategory.CONTENT_FORMAT_STRUCTURE)
    for r in cat_report.recommendations:
        assert r.category == RecommendationCategory.CONTENT_FORMAT_STRUCTURE


# --- 6. Minimum Confidence Filter ---

def test_min_confidence_filter(recommender):
    """Verify filtering by minimum confidence level."""
    high_only = recommender.generate_recommendations(min_confidence=ConfidenceLevel.HIGH)
    for r in high_only.recommendations:
        assert r.confidence_level == ConfidenceLevel.HIGH

    med_plus = recommender.generate_recommendations(min_confidence=ConfidenceLevel.MEDIUM)
    for r in med_plus.recommendations:
        assert r.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]


# --- 7. Conflicting Evidence Detection ---

def test_conflicting_evidence_detection(recommender):
    """Verify conflicting signals between observational view aggregates and experimental share rate are reported."""
    report = recommender.generate_recommendations()
    # Conflicting signals list should capture the Story vs Visual Hook discrepancy if present
    if report.conflicting_signals:
        conf = report.conflicting_signals[0]
        assert "conflict_id" in conf
        assert "observational_signal" in conf
        assert "experimental_evidence" in conf
        assert "resolution" in conf


# --- 8. Missing and Empty Datasets Handling ---

def test_missing_datasets_graceful_handling():
    """Verify engine handles complete lack of data without throwing uncaught exceptions."""
    empty_recommender = OptimizationRecommender(
        posts_df=None,
        comments_df=None,
        experiments_df=None,
        observations_df=None
    )
    report = empty_recommender.generate_recommendations()
    assert isinstance(report, RecommendationReport)
    assert len(report.insufficient_data_warnings) >= 3
    # Only A/B agenda heuristic proposals should be generated
    for r in report.recommendations:
        assert r.confidence_level == ConfidenceLevel.LOW


def test_empty_dataframe_inputs():
    """Verify empty DataFrames do not cause division-by-zero or crashes."""
    empty_posts = pd.DataFrame(columns=["post_id", "platform", "shares", "saves", "views"])
    recommender_empty = OptimizationRecommender(posts_df=empty_posts)
    report = recommender_empty.generate_recommendations()
    assert isinstance(report, RecommendationReport)


# --- 9. Prescriptive Experimentation Agenda ---

def test_experimentation_agenda_recommendations(recommender):
    """Verify engine produces concrete A/B testing proposals for uncertain areas."""
    agenda_report = recommender.generate_recommendations(category=RecommendationCategory.EXPERIMENTATION_AGENDA)
    assert len(agenda_report.recommendations) >= 1
    for r in agenda_report.recommendations:
        assert r.category == RecommendationCategory.EXPERIMENTATION_AGENDA
        assert "a/b test proposal" in r.title.lower() or "test proposal" in r.title.lower()
        assert r.confidence_level == ConfidenceLevel.LOW
        assert "validation" in r.suggested_validation_method.lower() or "test" in r.suggested_validation_method.lower()


# --- 10. Backwards Compatibility with Legacy ContentRecommender ---

def test_backwards_compatible_content_recommender(sample_datasets):
    """Verify legacy ContentRecommender works across all 3 import paths and handles edge cases safely."""
    from src.recommender.engine import ContentRecommender as EngineContentRecommender
    from src.recommender.content_recommender import ContentRecommender as ShimContentRecommender

    assert ContentRecommender is EngineContentRecommender
    assert ShimContentRecommender is EngineContentRecommender

    legacy_rec = ContentRecommender(target_metric="viral_coefficient")
    posts_df = sample_datasets["posts_df"]

    strategy = legacy_rec.recommend_weekly_strategy(posts_df)
    assert isinstance(strategy, dict)
    assert set(strategy.keys()) == {
        "recommended_topics",
        "optimal_duration_bin",
        "optimal_hook_style",
        "optimal_posting_window",
        "deep_resonance_topic",
        "confidence_statement",
    }

    # Empty DataFrame edge case
    empty_strategy = legacy_rec.recommend_weekly_strategy(pd.DataFrame())
    assert empty_strategy["recommended_topics"] == {}
    assert empty_strategy["deep_resonance_topic"] == "N/A"

    # Missing optional columns + duplicate post snapshots deduplication
    dup_df = pd.DataFrame([
        {
            "post_id": "P_DUP_1",
            "platform": "instagram",
            "collection_timestamp": "2026-10-01T00:00:00Z",
            "topic": "Topic_Old_Snapshot",
            "shares": 1000,
            "saves": 1000,
            "likes": 5000,
        },
        {
            "post_id": "P_DUP_1",
            "platform": "instagram",
            "collection_timestamp": "2026-10-02T00:00:00Z",
            "topic": "Topic_Latest_Snapshot",
            "shares": 5,
            "saves": 5,
            "likes": 10,
        },
        {
            "post_id": "P_UNIQUE_2",
            "platform": "instagram",
            "collection_timestamp": "2026-10-02T00:00:00Z",
            "topic": "Topic_Winner",
            "shares": 50,
            "saves": 50,
            "likes": 100,
        },
    ])
    dedup_strategy = legacy_rec.recommend_weekly_strategy(dup_df)
    assert "Topic_Old_Snapshot" not in dedup_strategy["recommended_topics"]
    assert list(dedup_strategy["recommended_topics"].keys())[0] == "Topic_Winner"
    assert dedup_strategy["optimal_duration_bin"] == "Medium (30-60s)"
    assert dedup_strategy["optimal_hook_style"] == "Problem-Focused Question Hook"
    assert dedup_strategy["optimal_posting_window"] == "18:00 - 21:00"

    # Test forward bridge to new engine
    rep = legacy_rec.generate_evidence_based_recommendations()
    assert isinstance(rep, RecommendationReport)
    assert rep.total_recommendations >= 1


# --- 11. Serialization to Dict ---

def test_recommendation_serialization(recommender):
    """Verify to_dict produces clean serializable dictionaries for API/UI."""
    report = recommender.generate_recommendations()
    report_dict = report.to_dict()
    assert "total_recommendations" in report_dict
    assert "recommendations" in report_dict
    assert "evidence_breakdown" in report_dict
    assert isinstance(report_dict["recommendations"], list)
    if report_dict["recommendations"]:
        first_r = report_dict["recommendations"][0]
        assert "recommendation_id" in first_r
        assert "priority_score" in first_r
        assert "confidence_level" in first_r
        assert "scoring_breakdown" in first_r


# --- 12. Targeted Audit Tests: Contradictory Evidence Penalty ---

def test_contradictory_evidence_penalty(recommender):
    """Verify that conflicting observational vs experimental evidence triggers an explicit penalty."""
    report = recommender.generate_recommendations()
    rec_001 = next((r for r in report.recommendations if r.recommendation_id == "REC_EXP_001"), None)
    assert rec_001 is not None
    assert rec_001.conflict_penalty == 0.85
    assert rec_001.scoring_breakdown["conflict_penalty"] == 0.85

    # Priority score must reflect the 0.85 multiplier on raw weighted sum
    raw_score = sum([
        0.30 * rec_001.scoring_breakdown["evidence_score"],
        0.20 * rec_001.scoring_breakdown["impact_score"],
        0.15 * rec_001.scoring_breakdown["sample_score"],
        0.15 * rec_001.scoring_breakdown["relevance_score"],
        0.10 * rec_001.scoring_breakdown["recency_score"],
        0.10 * rec_001.scoring_breakdown["quality_score"],
    ])
    expected_score = round(raw_score * 0.85, 4)
    assert abs(rec_001.priority_score - expected_score) <= 0.001


# --- 13. Targeted Audit Tests: Missing & Known Recency Handling ---

def test_missing_and_known_recency_handling(sample_datasets):
    """Verify that missing recency uses neutral baseline (0.70) while recent experiments get 1.0."""
    rec = OptimizationRecommender(
        posts_df=sample_datasets["posts_df"],
        experiments_df=sample_datasets["experiments_df"],
        observations_df=sample_datasets["observations_df"]
    )
    report = rec.generate_recommendations()

    # EXP_001 has end_date in October 2026 (<= 30 days relative to data reference date) -> 1.0
    rec_001 = next(r for r in report.recommendations if r.recommendation_id == "REC_EXP_001")
    assert rec_001.scoring_breakdown["recency_score"] == 1.0

    # Agenda proposals have recency_days = None -> neutral baseline 0.70
    agenda = next(r for r in report.recommendations if r.category == RecommendationCategory.EXPERIMENTATION_AGENDA)
    assert agenda.recency_days is None
    assert agenda.scoring_breakdown["recency_score"] == 0.70


# --- 14. Targeted Audit Tests: Weak Sample Sizes & CI Crossing Zero ---

def test_weak_sample_size_and_ci_crossing_zero_penalties(recommender):
    """Verify sample size scoring and uncertainty penalty when 95% CI spans zero."""
    report = recommender.generate_recommendations()

    # Pre-experimental agenda proposals (N = 0) get weak sample score (0.15)
    agenda = next(r for r in report.recommendations if r.category == RecommendationCategory.EXPERIMENTATION_AGENDA)
    assert agenda.sample_size == 0
    assert agenda.scoring_breakdown["sample_score"] == 0.15

    # EXP_004 (N = 70, base score 0.85), but CI spans zero [-71.8, 2720.8] -> 0.85 * 0.60 = 0.51
    rec_004 = next(r for r in report.recommendations if r.recommendation_id == "REC_EXP_004")
    assert rec_004.confidence_interval is not None
    assert rec_004.confidence_interval[0] <= 0 <= rec_004.confidence_interval[1]
    assert abs(rec_004.scoring_breakdown["sample_score"] - 0.51) <= 0.01


# --- 15. Targeted Audit Tests: Platform Relevance & No Automatic Reward for Cross-Platform ---

def test_platform_relevance_no_automatic_reward_for_cross_platform(recommender):
    """Verify that targeting a specific platform assigns 1.0 to exact matches and 0.70 to generic 'all'."""
    ig_report = recommender.generate_recommendations(platform="instagram")
    for r in ig_report.recommendations:
        if r.relevant_platform == "instagram":
            assert r.scoring_breakdown["relevance_score"] == 1.0
        elif r.relevant_platform == "all":
            # Must NOT be automatically rewarded with 1.0
            assert r.scoring_breakdown["relevance_score"] == 0.70


# --- 16. Targeted Audit Tests: Provenance & Synthetic Flag ---

def test_provenance_synthetic_demonstration_flag(recommender):
    """Verify that all recommendations and reports are explicitly flagged as synthetic demonstrations."""
    report = recommender.generate_recommendations()
    assert report.is_synthetic_demonstration is True
    for r in report.recommendations:
        assert r.is_synthetic_demonstration is True
        if r.recommendation_id in ["REC_EXP_001", "REC_EXP_002", "REC_EXP_003"]:
            assert "demonstration" in r.title.lower() or "synthetic" in r.title.lower()


# --- 17. Targeted Audit Tests: NLP Unsupported Claims Stripped ---

def test_nlp_unsupported_claims_stripped(recommender):
    """Verify that fabricated claims (2.8x saves, r=0.68) have been removed from NLP recommendations."""
    report = recommender.generate_recommendations()
    nlp_rec = next((r for r in report.recommendations if r.recommendation_id == "REC_NLP_001"), None)
    if nlp_rec:
        assert nlp_rec.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.EXPLORATORY]
        assert "2.8x" not in nlp_rec.reasoning
        assert "0.68" not in nlp_rec.supporting_evidence
        # Real deduplicated Spearman correlation r = -0.0109 (p = 0.932) must be stated
        assert "-0.0109" in nlp_rec.supporting_evidence
        assert "0.932" in nlp_rec.supporting_evidence


# --- 18. Targeted Audit Tests: Multiple-Testing Holm Adjustment Integration ---

def test_multiple_testing_holm_adjustment_integration(recommender):
    """Verify that multiple testing adjusted p-values are calculated and distinct from raw p-values."""
    report = recommender.generate_recommendations()
    r1 = next(r for r in report.recommendations if r.recommendation_id == "REC_EXP_001")
    assert r1.p_value is not None
    assert r1.p_value_adjusted is not None
    assert r1.p_value_adjusted >= r1.p_value


# --- 19. B2 Regression Tests: Dynamic Experiment Metrics & Snapshot Deduplication ---

def test_experiment_recommendations_derive_metrics_and_confidence_dynamically(sample_datasets):
    """Verify experiment recommendation metrics and ConfidenceLevel adapt dynamically to observations_df."""
    exps_df = sample_datasets["experiments_df"].copy()
    obs_df = sample_datasets["observations_df"].copy()

    # Modify EXP_001 observations to have small sample size (N=16 total) and overlapping noisy values
    non_exp1 = obs_df[obs_df["experiment_id"] != "EXP_001"].copy()
    exp1_orig = obs_df[obs_df["experiment_id"] == "EXP_001"].copy()
    variants = sorted(exp1_orig["variant"].dropna().unique().tolist())
    assert len(variants) == 2

    part_a = exp1_orig[exp1_orig["variant"] == variants[0]].head(8).copy()
    part_b = exp1_orig[exp1_orig["variant"] == variants[1]].head(8).copy()
    part_a["shares"] = [40.0, 42.0, 38.0, 41.0, 39.0, 43.0, 37.0, 40.0]
    part_a["reach"] = 1000
    part_a["share_rate"] = part_a["shares"] / part_a["reach"]
    part_b["shares"] = [41.0, 43.0, 39.0, 42.0, 40.0, 44.0, 38.0, 41.0]
    part_b["reach"] = 1000
    part_b["share_rate"] = part_b["shares"] / part_b["reach"]
    modified_obs = pd.concat([non_exp1, part_a, part_b], ignore_index=True)

    rec = OptimizationRecommender(
        posts_df=sample_datasets["posts_df"],
        comments_df=sample_datasets["comments_df"],
        experiments_df=exps_df,
        observations_df=modified_obs,
    )
    report = rec.generate_recommendations()
    rec_001 = next(r for r in report.recommendations if r.recommendation_id == "REC_EXP_001")

    # Metrics must reflect modified observations rather than hardcoded 54.4% / t=8.50 / HIGH
    assert rec_001.sample_size == 16
    assert rec_001.p_value is not None and rec_001.p_value > 0.05
    assert rec_001.confidence_level != ConfidenceLevel.HIGH
    assert "+54.4%" not in rec_001.title
    assert "t = 8.50" not in rec_001.supporting_evidence


def test_recommender_deduplicates_posts_df_snapshots(sample_datasets):
    """Verify OptimizationRecommender deduplicates repeated post snapshots upon initialization."""
    posts_df = sample_datasets["posts_df"]
    rec = OptimizationRecommender(
        posts_df=posts_df,
        comments_df=sample_datasets["comments_df"],
        experiments_df=sample_datasets["experiments_df"],
        observations_df=sample_datasets["observations_df"],
    )
    assert len(posts_df) == 121
    assert len(rec.posts_df) == 120


# --- 20. B2.2 Regression Tests: M5 Evidence Classification Hardening ---

def _make_synthetic_comparison_result(
    sample_size_a: int = 35,
    sample_size_b: int = 35,
    value_a: float = 100.0,
    value_b: float = 140.0,
    p_value: float = 0.001,
    p_value_adjusted: float = 0.003,
    confidence_interval=(20.0, 60.0),
    effect_size: float = 0.75,
    verdict: str = "Statistically Significant Evidence of Difference (Favors Variant B)",
    ci_method: str = "Welch-Satterthwaite t-interval",
    primary_method: str = "Welch's t-test",
):
    """Helper constructing a ComparisonResult for deterministic evidence-classification tests."""
    from src.statistics.ab_testing import ComparisonResult

    diff = float(value_b - value_a)
    lift = (diff / abs(value_a) * 100.0) if abs(value_a) > 1e-9 else 0.0
    return ComparisonResult(
        metric_name="views",
        metric_type="continuous",
        variant_a_label="Control",
        variant_b_label="Treatment",
        sample_size_a=sample_size_a,
        sample_size_b=sample_size_b,
        value_a=value_a,
        value_b=value_b,
        absolute_difference=diff,
        relative_lift_pct=lift,
        test_statistic=3.5,
        test_name="Welch's t-test",
        primary_method=primary_method,
        p_value=p_value,
        p_value_adjusted=p_value_adjusted,
        confidence_interval=confidence_interval,
        ci_level=0.95,
        ci_method=ci_method,
        effect_size=effect_size,
        effect_size_type="Hedges' g",
        is_statistically_significant=bool(p_value_adjusted < 0.05),
        verdict=verdict,
    )


def test_invalid_comparison_results_classified_as_insufficient_evidence():
    """Verify invalid, unrunnable, or zero-variance ComparisonResult objects return INSUFFICIENT_EVIDENCE."""
    from src.statistics.ab_testing import ABTestingFramework

    fw = ABTestingFramework(alpha=0.05)
    rec = OptimizationRecommender(alpha=0.05)

    # 1. Insufficient sample size from ABTestingFramework (N_A=1, N_B=2)
    res_n1 = fw.compare_continuous([10.0], [15.0, 16.0], metric_name="views")
    level_n1, _ = rec._classify_experiment_evidence(True, res_n1)
    assert level_n1 == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # 2. Zero-variance result from ABTestingFramework
    res_zero_var = fw.compare_continuous([10.0] * 35, [20.0] * 35, metric_name="views")
    level_zv, _ = rec._classify_experiment_evidence(True, res_zero_var)
    assert level_zv == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # 3. Invalid proportion result (zero denominator)
    res_bad_prop = fw.compare_proportions(count_a=0, nobs_a=0, count_b=5, nobs_b=20)
    level_bp, _ = rec._classify_experiment_evidence(True, res_bad_prop)
    assert level_bp == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # 4. Non-finite essential statistics (NaN p-value or effect size)
    res_nan_p = _make_synthetic_comparison_result(p_value=float("nan"), p_value_adjusted=float("nan"))
    assert rec._classify_experiment_evidence(True, res_nan_p)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    res_nan_eff = _make_synthetic_comparison_result(effect_size=float("nan"))
    assert rec._classify_experiment_evidence(True, res_nan_eff)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE


def test_nan_and_malformed_confidence_intervals_classified_as_insufficient_evidence():
    """Verify NaN, infinite, missing, or reversed confidence intervals return INSUFFICIENT_EVIDENCE."""
    rec = OptimizationRecommender(alpha=0.05)

    # NaN bounds must never satisfy ci_excludes_zero
    res_nan_ci = _make_synthetic_comparison_result(confidence_interval=(float("nan"), float("nan")))
    assert rec._classify_experiment_evidence(True, res_nan_ci)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    res_half_nan = _make_synthetic_comparison_result(confidence_interval=(10.0, float("nan")))
    assert rec._classify_experiment_evidence(True, res_half_nan)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # Infinite bounds
    res_inf_ci = _make_synthetic_comparison_result(confidence_interval=(-float("inf"), float("inf")))
    assert rec._classify_experiment_evidence(True, res_inf_ci)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    res_pos_inf = _make_synthetic_comparison_result(confidence_interval=(5.0, float("inf")))
    assert rec._classify_experiment_evidence(True, res_pos_inf)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # Missing or malformed CI tuples
    res_none_ci = _make_synthetic_comparison_result(confidence_interval=None)
    assert rec._classify_experiment_evidence(True, res_none_ci)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    res_reversed_ci = _make_synthetic_comparison_result(confidence_interval=(50.0, 10.0))
    assert rec._classify_experiment_evidence(True, res_reversed_ci)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # Finite interval crossing zero remains LOW (statistically inconclusive, not INSUFFICIENT_EVIDENCE)
    res_cross_zero = _make_synthetic_comparison_result(confidence_interval=(-5.0, 45.0))
    assert rec._classify_experiment_evidence(True, res_cross_zero)[0] == ConfidenceLevel.LOW


def test_minimum_arm_sample_guardrail_and_high_confidence_thresholds():
    """Verify <5/arm returns INSUFFICIENT_EVIDENCE, ==5/arm can reach MEDIUM, and HIGH requires >=30/arm."""
    rec = OptimizationRecommender(alpha=0.05)

    # Fewer than 5 observations in either arm -> INSUFFICIENT_EVIDENCE even with favorable statistics
    res_4_4 = _make_synthetic_comparison_result(sample_size_a=4, sample_size_b=4)
    level_4_4, rationale_4_4 = rec._classify_experiment_evidence(True, res_4_4)
    assert level_4_4 == ConfidenceLevel.INSUFFICIENT_EVIDENCE
    assert "not a formal power calculation" in rationale_4_4.lower()

    res_4_40 = _make_synthetic_comparison_result(sample_size_a=4, sample_size_b=40)
    assert rec._classify_experiment_evidence(True, res_4_40)[0] == ConfidenceLevel.INSUFFICIENT_EVIDENCE

    # Exactly 5 per arm qualifies for MEDIUM only when randomized, significant, CI excludes zero, and diff > 0
    res_5_5_valid = _make_synthetic_comparison_result(sample_size_a=5, sample_size_b=5)
    assert rec._classify_experiment_evidence(True, res_5_5_valid)[0] == ConfidenceLevel.MEDIUM

    # At 5 per arm, failing significance, crossing zero, or non-randomized design downgrades to LOW
    res_5_5_nonsig = _make_synthetic_comparison_result(
        sample_size_a=5, sample_size_b=5, p_value=0.04, p_value_adjusted=0.12
    )
    assert rec._classify_experiment_evidence(True, res_5_5_nonsig)[0] == ConfidenceLevel.LOW

    res_5_5_cross = _make_synthetic_comparison_result(
        sample_size_a=5, sample_size_b=5, confidence_interval=(-2.0, 30.0)
    )
    assert rec._classify_experiment_evidence(True, res_5_5_cross)[0] == ConfidenceLevel.LOW
    assert rec._classify_experiment_evidence(False, res_5_5_valid)[0] == ConfidenceLevel.LOW

    # Existing HIGH-confidence requirements (>=30/arm, >=60 total, |g| >= 0.20, randomized, p_adj < 0.05, CI > 0)
    res_29_31 = _make_synthetic_comparison_result(sample_size_a=29, sample_size_b=31, effect_size=0.80)
    assert rec._classify_experiment_evidence(True, res_29_31)[0] == ConfidenceLevel.MEDIUM

    res_30_30_small_eff = _make_synthetic_comparison_result(sample_size_a=30, sample_size_b=30, effect_size=0.15)
    assert rec._classify_experiment_evidence(True, res_30_30_small_eff)[0] == ConfidenceLevel.MEDIUM

    res_30_30_high = _make_synthetic_comparison_result(sample_size_a=30, sample_size_b=30, effect_size=0.20)
    assert rec._classify_experiment_evidence(True, res_30_30_high)[0] == ConfidenceLevel.HIGH


def test_continuous_metric_snapshot_deduplication_and_rate_preservation(sample_datasets):
    """Verify repeated post_id snapshots in continuous metrics are deduplicated cleanly without inflating N."""
    exps_df = sample_datasets["experiments_df"].copy()
    obs_df = sample_datasets["observations_df"].copy()

    # 1. With valid collection_timestamp: duplicate 10 EXP_002 observations with earlier timestamps
    # and inflated stale views; latest snapshot must be selected and sample_size must remain 100.
    exp2_rows = obs_df[obs_df["experiment_id"] == "EXP_002"].copy()
    assert len(exp2_rows) == 100
    stale_dups = exp2_rows.head(10).copy()
    stale_dups["collection_timestamp"] = "2026-01-01T00:00:00Z"
    stale_dups["views"] = 999999999

    obs_with_ordered_dups = pd.concat([obs_df, stale_dups], ignore_index=True)
    rec_ordered = OptimizationRecommender(
        posts_df=sample_datasets["posts_df"],
        comments_df=sample_datasets["comments_df"],
        experiments_df=exps_df,
        observations_df=obs_with_ordered_dups,
    )
    rep_ordered = rec_ordered.generate_recommendations()
    rec_002 = next(r for r in rep_ordered.recommendations if r.recommendation_id == "REC_EXP_002")
    rec_001 = next(r for r in rep_ordered.recommendations if r.recommendation_id == "REC_EXP_001")

    assert rec_002.sample_size == 100  # Not 110
    assert rec_002.confidence_level == ConfidenceLevel.HIGH
    assert "Snapshot deduplication (EXP_002)" in rec_002.limitations_and_caveats
    # Rate-metric recommendation (EXP_001 share_rate) remains completely intact
    assert rec_001.sample_size == 120
    assert rec_001.confidence_level == ConfidenceLevel.HIGH

    # 2. Without reliable ordering timestamp: ambiguous duplicate post_ids are excluded conservatively
    # rather than arbitrarily choosing a row.
    obs_no_ts = obs_df.drop(columns=["collection_timestamp"]).copy()
    exp2_no_ts = obs_no_ts[obs_no_ts["experiment_id"] == "EXP_002"].copy()
    ambig_dups = exp2_no_ts.head(4).copy()  # 4 post_ids become ambiguous duplicates
    ambig_dups["views"] = 123456789
    obs_with_ambig_dups = pd.concat([obs_no_ts, ambig_dups], ignore_index=True)

    rec_ambig = OptimizationRecommender(
        posts_df=sample_datasets["posts_df"],
        comments_df=sample_datasets["comments_df"],
        experiments_df=exps_df,
        observations_df=obs_with_ambig_dups,
    )
    rep_ambig = rec_ambig.generate_recommendations()
    rec_002_ambig = next(r for r in rep_ambig.recommendations if r.recommendation_id == "REC_EXP_002")
    assert rec_002_ambig.sample_size == 96  # 100 - 4 ambiguous duplicate post_ids excluded
    assert "Ambiguous duplicate snapshots (EXP_002)" in rec_002_ambig.limitations_and_caveats

