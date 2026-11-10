"""Data schemas and types for Module 5: Engagement Optimization Recommender.

Defines:
1. Recommendation categories across format, topic, caption, relatability, timing, and experimentation.
2. Evidence sources (A/B testing, historical performance, virality predictions, sentiment/relatability, heuristics).
3. Confidence levels (HIGH, MEDIUM, LOW) based on an explicit evidence hierarchy.
4. Structured recommendation dataclass and reporting containers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RecommendationCategory(str, Enum):
    """Functional categories for content optimization recommendations."""
    CONTENT_FORMAT_STRUCTURE = "content_format_structure"
    CAPTION_DEPTH_STYLE = "caption_depth_style"
    TOPIC_SELECTION = "topic_selection"
    AUDIENCE_RELATABILITY = "audience_relatability"
    POSTING_TIMING = "posting_timing"
    PLATFORM_STRATEGY = "platform_strategy"
    CALLS_TO_ACTION = "calls_to_action"
    EXPERIMENTATION_AGENDA = "experimentation_agenda"


class EvidenceSource(str, Enum):
    """Source module providing empirical or model-derived support."""
    AB_TESTING = "ab_testing"                            # Module 4: Randomized trials or observational A/B tests
    HISTORICAL_PERFORMANCE = "historical_performance"    # Module 1: Descriptive metrics, retention, save/share ratios
    VIRALITY_PREDICTION = "virality_prediction"          # Module 2: Supervised pre-publication ML regressor / EVI
    SENTIMENT_RELATABILITY = "sentiment_relatability"    # Module 3: Hybrid NLP relatability & sentiment classifications
    HEURISTIC = "heuristic"                              # Domain heuristics or exploratory hypothesis generation


class ConfidenceLevel(str, Enum):
    """Evidence hierarchy classification for recommendation certainty.
    
    HIGH:
      Supported by verified randomized controlled trials with:
      - Verified random assignment of creative variants.
      - Analysis evaluated at the independent post unit (post-level aggregation addressing intra-post
        exposure dependence; note: does not correct cross-post creator or campaign clustering).
      - Adequate sample size (N >= 30 posts per variant, total N >= 60) and statistical power (>= 80%).
      - Statistically significant after multiple testing correction (p_adj < alpha = 0.05).
      - Actionable effect size (|g| >= 0.20 or |h| >= 0.05) with confidence interval not spanning zero.
      - Explicit data provenance disclosure (flagged as synthetic benchmark demonstration).
      
    MEDIUM:
      Supported by controlled observational comparisons with matched cohorts,
      randomized experiments with moderate sample sizes (20 <= N < 60), or
      predictive models verified with temporal train-test cross-validation.
      NOT assigned merely because an observational cohort contains 15 posts.
      NOT assigned to internal rule-based NLP signals without external ground-truth validation.
      
    LOW:
      Observational cohort patterns, correlation analyses, small samples (N < 20),
      or unverified heuristic hypotheses. Always pairs with non-causal disclosures.
      
    EXPLORATORY:
      Formative hypotheses or descriptive patterns that lack experimental controls or replication.
      
    INSUFFICIENT_EVIDENCE:
      Empty datasets, zero observations, or contradictory findings with no empirical resolution.
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    EXPLORATORY = "EXPLORATORY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class ContentRecommendation:
    """Structured, explainable content optimization recommendation."""
    recommendation_id: str
    title: str
    category: RecommendationCategory
    suggested_action: str
    reasoning: str
    supporting_evidence: str
    evidence_source: EvidenceSource
    relevant_platform: str                              # 'instagram', 'youtube', or 'all'
    expected_impact_direction: str                      # e.g., '+15% to +35% in Share Rate'
    confidence_level: ConfidenceLevel
    limitations_and_caveats: str
    suggested_validation_method: str
    primary_metric: str = "engagement_virality_index"
    sample_size: int = 0
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    p_value_adjusted: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    ci_method: str = ""
    unit_of_analysis: str = "Post"
    clustering_nuance: str = ""
    is_randomized_evidence: bool = False
    is_synthetic_demonstration: bool = True
    priority_score: float = 0.0
    conflict_penalty: float = 1.0
    recency_days: Optional[int] = None
    data_quality_score: float = 0.85
    scoring_breakdown: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize recommendation to dictionary for API and UI rendering."""
        return {
            "recommendation_id": self.recommendation_id,
            "title": self.title,
            "category": self.category.value if isinstance(self.category, RecommendationCategory) else str(self.category),
            "suggested_action": self.suggested_action,
            "reasoning": self.reasoning,
            "supporting_evidence": self.supporting_evidence,
            "evidence_source": self.evidence_source.value if isinstance(self.evidence_source, EvidenceSource) else str(self.evidence_source),
            "relevant_platform": self.relevant_platform,
            "expected_impact_direction": self.expected_impact_direction,
            "confidence_level": self.confidence_level.value if isinstance(self.confidence_level, ConfidenceLevel) else str(self.confidence_level),
            "limitations_and_caveats": self.limitations_and_caveats,
            "suggested_validation_method": self.suggested_validation_method,
            "primary_metric": self.primary_metric,
            "sample_size": self.sample_size,
            "effect_size": round(self.effect_size, 4) if self.effect_size is not None else None,
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "p_value_adjusted": round(self.p_value_adjusted, 6) if self.p_value_adjusted is not None else None,
            "confidence_interval": (round(self.confidence_interval[0], 4), round(self.confidence_interval[1], 4)) if self.confidence_interval is not None else None,
            "ci_method": self.ci_method,
            "unit_of_analysis": self.unit_of_analysis,
            "clustering_nuance": self.clustering_nuance,
            "is_randomized_evidence": self.is_randomized_evidence,
            "is_synthetic_demonstration": self.is_synthetic_demonstration,
            "priority_score": round(self.priority_score, 4),
            "conflict_penalty": round(self.conflict_penalty, 3),
            "recency_days": self.recency_days,
            "data_quality_score": round(self.data_quality_score, 3),
            "scoring_breakdown": {k: round(v, 4) for k, v in self.scoring_breakdown.items()},
            "tags": self.tags
        }


@dataclass
class RecommendationReport:
    """Summary and diagnostic container for generated recommendation sets."""
    total_recommendations: int
    recommendations: List[ContentRecommendation]
    platform_filter: Optional[str] = None
    category_filter: Optional[str] = None
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    evidence_breakdown: Dict[str, int] = field(default_factory=dict)
    conflicting_signals: List[Dict[str, Any]] = field(default_factory=list)
    insufficient_data_warnings: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    is_synthetic_demonstration: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_recommendations": self.total_recommendations,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "platform_filter": self.platform_filter,
            "category_filter": self.category_filter,
            "high_confidence_count": self.high_confidence_count,
            "medium_confidence_count": self.medium_confidence_count,
            "low_confidence_count": self.low_confidence_count,
            "evidence_breakdown": self.evidence_breakdown,
            "conflicting_signals": self.conflicting_signals,
            "insufficient_data_warnings": self.insufficient_data_warnings,
            "generated_at": self.generated_at,
            "is_synthetic_demonstration": self.is_synthetic_demonstration
        }

