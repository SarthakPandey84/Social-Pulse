"""Unified Data Schemas for Social Engagement Intelligence.

Defines the 14-field unified schema, type definitions, nullability constraints,
metric availability classifications, and provenance metadata across Instagram, YouTube, and scrapers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set
import pandas as pd


# The 14 canonical unified fields required across all platforms
# Note: 'views' and 'impressions' are strictly separate media metrics:
# - views: video plays / starts (at least 3s on IG, watch event on YT)
# - impressions: total screen exposures / displays on screen
UNIFIED_POST_FIELDS = [
    "platform",
    "post_id",
    "timestamp",
    "content_type",
    "views",
    "impressions",
    "reach",
    "likes",
    "comments_count",
    "shares",
    "saves",
    "follower_count",
    "retention_rate",
    "collection_timestamp"
]

# Field nullability specifications
# True = Metric is allowed to be missing / null (untracked on platform or owner-only)
# False = Strict mandatory field required for identity and time indexing
NULLABLE_SPECIFICATION = {
    "platform": False,
    "post_id": False,
    "timestamp": False,
    "content_type": False,
    "views": True,             # Video plays; null for static image/carousel
    "impressions": True,       # Total screen displays; null if untracked
    "reach": True,             # Unique accounts reached; owner-only analytics
    "likes": True,             # Positive clicks; null if hidden by creator
    "comments_count": True,    # Comment count; null if comments disabled
    "shares": True,            # Outbound shares; owner-only analytics
    "saves": True,             # Bookmarks; IG owner-only; N/A on YouTube
    "follower_count": True,    # Creator followers snapshot
    "retention_rate": True,    # Watch percentage; video owner-only
    "collection_timestamp": False
}

# Pandas nullable data types mapping
UNIFIED_PANDAS_DTYPES = {
    "platform": "string",
    "post_id": "string",
    "content_type": "string",
    "views": "Int64",
    "impressions": "Int64",
    "reach": "Int64",
    "likes": "Int64",
    "comments_count": "Int64",
    "shares": "Int64",
    "saves": "Int64",
    "follower_count": "Int64",
    "retention_rate": "Float64"
}


class MetricAvailabilityStatus(str, Enum):
    """Rigorous classification of why a metric value is present or absent."""
    OBSERVED = "observed"              # Successfully collected, including a genuine zero
    MISSING = "missing"                # Expected and applicable, but failed to collect
    UNAVAILABLE = "unavailable"        # Not exposed by platform/tier (e.g. competitor saves via public API)
    NOT_APPLICABLE = "not_applicable"  # Structurally inapplicable (e.g. saves on YouTube, retention on static image)


class SourceProvenance(str, Enum):
    """Strict provenance taxonomy distinguishing real, synthetic, owner-authorized, and unverified sources."""
    REAL_PUBLIC_DATA = "real_public_data"
    OWNER_AUTHORIZED_DATA = "owner_authorized_data"
    SYNTHETIC_DATA = "synthetic_data"
    SIMULATED_DATA = "simulated_data"
    EXTERNAL_BENCHMARK_UNVERIFIED = "external_benchmark_unverified"
    WEAKLY_LABELLED_DATA = "weakly_labelled_data"


SUPPORTED_PLATFORMS = ("instagram", "youtube", "tiktok")

PROXY_METRIC_DEFINITIONS: Dict[str, str] = {
    "tiktok_collect_count": (
        "TikTok video favorite/collect counter (`collect_count`). Distinct from Instagram Insights "
        "`saves`; retained as an explicit proxy metric and never silently pooled with Instagram saves."
    ),
    "tiktok_whatsapp_share_count": (
        "TikTok outbound WhatsApp share counter (`whatsapp_share_count`), subset/companion to total `shares`."
    ),
    "youtube_trending_snapshot": (
        "Point-in-time YouTube Trending tab snapshot indexed by `(video_id, region, trending_date)`. "
        "Subject to platform curation/selection bias and not representative of channel-wide averages."
    ),
}


def get_applicable_metrics(platform: str, content_type: str, access_tier: str = "owner_oauth") -> Set[str]:
    """Return the exact set of canonical metrics that are applicable and collectible.

    Structural Rules:
    1. saves: Only applicable to Instagram; NOT applicable to YouTube.
    2. retention_rate: Only applicable to video content ('reel', 'short', 'video'); NOT applicable to static 'image' or 'carousel'.
    3. views: Video play events; NOT applicable to static 'image' or 'carousel' (which have impressions/reach instead).
    4. Access Tier restrictions:
       - If access_tier == 'public_api': 'reach', 'saves', 'shares', 'retention_rate', 'impressions' are UNAVAILABLE.
    """
    platform_norm = str(platform).lower() if pd.notna(platform) else "unknown"
    content_norm = str(content_type).lower() if pd.notna(content_type) else "unknown"
    tier_norm = str(access_tier).lower() if pd.notna(access_tier) else "owner_oauth"

    applicable = {"likes", "comments_count", "follower_count"}

    # Views vs Impressions by content type
    is_video = content_norm in {"reel", "short", "video", "short video", "short_video"}
    if is_video:
        applicable.add("views")
        if tier_norm == "owner_oauth":
            applicable.add("retention_rate")
            applicable.add("impressions")
            applicable.add("reach")
            applicable.add("shares")
    else:
        # Static image / carousel has impressions and reach, but NOT video views or video retention
        if tier_norm == "owner_oauth":
            applicable.add("impressions")
            applicable.add("reach")
            applicable.add("shares")

    # Platform-specific features
    if platform_norm == "instagram" and tier_norm == "owner_oauth":
        applicable.add("saves")

    if platform_norm == "youtube" and tier_norm == "owner_oauth":
        applicable.add("shares")

    if platform_norm == "tiktok":
        applicable = {"views", "likes", "comments_count", "shares", "follower_count"}
        if is_video or content_norm in {"short_video", "video", "unknown"}:
            applicable.add("views")

    # Public API limitations
    if tier_norm == "public_api":
        if is_video:
            applicable.add("views")
        # In public tier: reach, shares, saves, retention_rate, impressions are unavailable

    return applicable


@dataclass
class DataProvenance:
    """Tracks origin and acquisition method of ingested records."""
    source_name: str
    access_tier: str          # "public_api", "owner_oauth", "web_scraper", "sample_fixture"
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "access_tier": self.access_tier,
            "extracted_at": self.extracted_at,
            "notes": self.notes
        }


# --- Module 3: Comment Schema & Classification Enums ---

UNIFIED_COMMENT_FIELDS = [
    "comment_id",
    "post_id",
    "platform",
    "comment_text",
    "comment_timestamp",
    "language",
    "sentiment_label",
    "sentiment_confidence",
    "sentiment_score",
    "relatability_label",
    "relatability_confidence",
    "classification_status",
    "model_version",
    "analysis_timestamp"
]

NULLABLE_COMMENT_SPECIFICATION = {
    "comment_id": False,
    "post_id": False,
    "platform": False,
    "comment_text": False,
    "comment_timestamp": True,
    "language": True,
    "sentiment_label": True,
    "sentiment_confidence": True,
    "sentiment_score": True,
    "relatability_label": True,
    "relatability_confidence": True,
    "classification_status": False,
    "model_version": False,
    "analysis_timestamp": False
}

UNIFIED_COMMENT_DTYPES = {
    "comment_id": "string",
    "post_id": "string",
    "platform": "string",
    "comment_text": "string",
    "language": "string",
    "sentiment_label": "string",
    "sentiment_confidence": "Float64",
    "sentiment_score": "Float64",
    "relatability_label": "string",
    "relatability_confidence": "Float64",
    "classification_status": "string",
    "model_version": "string"
}


class ClassificationStatus(str, Enum):
    """Execution status for audience comment classification."""
    SUCCESS = "success"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    EMPTY_TEXT = "empty_text"
    UNCERTAIN = "uncertain"
    INFERENCE_FAILED = "inference_failed"


class SentimentLabel(str, Enum):
    """Four-way sentiment polarity dimension."""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"


class RelatabilityLabel(str, Enum):
    """Audience identification dimension."""
    RELATABLE = "Relatable"
    NEUTRAL = "Neutral"
    UNCERTAIN = "Uncertain"


# --- Module 4: A/B Testing & Experimentation Schemas ---

class ExperimentType(str, Enum):
    """Classification of experimental control."""
    RANDOMIZED_EXPERIMENT = "randomized_experiment"          # Controlled split with random delivery / balanced audience
    OBSERVATIONAL_COMPARISON = "observational_comparison"    # Post-hoc comparison of historical content without randomization


class StatisticalUnit(str, Enum):
    """Statistical unit of analysis for digital content experiments.
    
    1. POST_LEVEL_INDEPENDENT:
       The post is the experimental unit. Continuous outcomes or post-level rates
       (e.g., share_rate = shares / reach per post) are compared across independent posts.
       This cluster-collapsed approach resolves intra-post exposure dependence.
       
    2. USER_LEVEL_RANDOMIZED:
       Individual audience members are randomly assigned to variants (e.g. ad auctions).
       The user is the unit of randomization. Note: Individual user IDs are NOT provided
       by standard public or creator-level social media APIs.
       
    3. EXPOSURE_LEVEL_AGGREGATE:
       All reach/views and action counts are summed across posts. Treats each exposure
       as an independent Bernoulli trial. Note: Ignores intra-post clustering (ICC > 0),
       which inflates Type I error rates if multiple posts exist per variant.
    """
    POST_LEVEL_INDEPENDENT = "post_level_independent"
    USER_LEVEL_RANDOMIZED = "user_level_randomized"
    EXPOSURE_LEVEL_AGGREGATE = "exposure_level_aggregate"


class ExperimentStatus(str, Enum):
    """Lifecycle and conclusion status of an experiment."""
    RUNNING = "running"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_SAMPLE = "insufficient_sample"


UNIFIED_EXPERIMENT_FIELDS = [
    "experiment_id",
    "experiment_name",
    "hypothesis",
    "experiment_type",
    "platform",
    "variant_a_label",
    "variant_b_label",
    "primary_metric",
    "secondary_metrics",
    "start_date",
    "end_date",
    "experiment_status"
]

UNIFIED_OBSERVATION_FIELDS = [
    "experiment_id",
    "observation_id",
    "variant",
    "post_id",
    "platform",
    "collection_timestamp",
    "views",
    "reach",
    "shares",
    "saves",
    "likes",
    "comments_count",
    "retention_rate",
    "relatability_pct",
    "share_rate",
    "save_rate"
]


