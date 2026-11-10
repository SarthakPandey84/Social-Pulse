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

from src.recommender.engine import OptimizationRecommender
from src.recommender.content_recommender import ContentRecommender
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
    """Verify legacy ContentRecommender API continues to function without breaking existing callers."""
    legacy_rec = ContentRecommender(target_metric="viral_coefficient")
    posts_df = sample_datasets["posts_df"]

    strategy = legacy_rec.recommend_weekly_strategy(posts_df)
    assert isinstance(strategy, dict)
    assert "recommended_topics" in strategy
    assert "optimal_duration_bin" in strategy
    assert "optimal_hook_style" in strategy
    assert "optimal_posting_window" in strategy
    assert "deep_resonance_topic" in strategy
    assert "confidence_statement" in strategy

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
        # Real Spearman correlation r ~ 0.0044 must be stated
        assert "0.004" in nlp_rec.supporting_evidence
        assert "0.97" in nlp_rec.supporting_evidence


# --- 18. Targeted Audit Tests: Multiple-Testing Holm Adjustment Integration ---

def test_multiple_testing_holm_adjustment_integration(recommender):
    """Verify that multiple testing adjusted p-values are calculated and distinct from raw p-values."""
    report = recommender.generate_recommendations()
    r1 = next(r for r in report.recommendations if r.recommendation_id == "REC_EXP_001")
    assert r1.p_value is not None
    assert r1.p_value_adjusted is not None
    assert r1.p_value_adjusted >= r1.p_value

