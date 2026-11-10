"""Reproducible external dataset registry, verifier, resumable downloader, and raw quality inspector.

Supports non-interactive pinned downloads from Hugging Face, GitHub, and Zenodo
using only Python standard library HTTP/hash/archive primitives (zero additional
runtime dependency requirements), plus gated Kaggle CLI preflight checks.

Safety & Integrity Guarantees:
- All Hugging Face and GitHub sources are pinned to exact commit SHAs.
- All Zenodo sources are pinned to exact versioned record DOIs and MD5 checksums.
- Only explicitly allowlisted tabular/documentation files are fetched (large raw
  video/image blobs such as `.mp4` or >1 GB image folders are never fetched).
- Downloads write to `.part` files, support HTTP Range resume when offered, retry
  transient errors with exponential backoff, verify size + cryptographic digests
  before atomic promotion, and redact credentials/tokens in all diagnostics.
- Raw quality inspection scans schema, duplicates, nulls, date spans, potential
  PII patterns, and post-outcome target-leakage columns, saving JSON/Markdown
  reports under `data/external/reports/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import numpy as np

from src.data_ingestion.schemas import SourceProvenance


DEFAULT_EXTERNAL_ROOT = Path("data") / "external"

# Regexes used solely for security redaction & aggregate PII auditing (never stored verbatim)
_SECRET_REDACT_PATTERNS = (
    re.compile(r"(KAGGLE_(?:KEY|API_TOKEN|USERNAME)\s*[=:]\s*)[^\s,;\"']+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"([?&](?:key|token|access_token|api_key)=)[^&\s]+", re.IGNORECASE),
    re.compile(r"(gh[ps]_[A-Za-z0-9_]+|hf_[A-Za-z0-9_]+|AIza[0-9A-Za-z\-_]+)"),
)
_HANDLE_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_.]{2,30}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

_LEAKAGE_COLUMN_HINTS = {
    "performance_bucket_label": "Post-outcome binned engagement label — target leakage if used as feature.",
    "engagement_rate": "Ratio computed from post-outcome likes/comments/shares — excluded from pre-pub X.",
    "viral_coefficient": "Ex-post Virality metric derived from outcome counters.",
    "engagement_virality_index": "Ex-post EVI composite derived from outcome counters.",
    "sentiment_label": "Derived/annotated target label — excluded from NLP feature inputs.",
}

_RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

GOVERNANCE_TIERS: Tuple[str, ...] = (
    "VERIFIED_COMMERCIAL_SAFE",
    "NON_COMMERCIAL_RESEARCH_ONLY",
    "PENDING_REVIEW_WEAKLY_LABELLED",
    "EXTERNAL_BENCHMARK_UNVERIFIED",
    "SYNTHETIC_BENCHMARK",
    "BLOCKED_MISSING_CREDENTIALS",
)

_GOVERNANCE_RESTRICTIONS: Dict[str, str] = {
    "VERIFIED_COMMERCIAL_SAFE": "Cleared license & verified origin; observational caveats still apply (non-causal unless randomized).",
    "NON_COMMERCIAL_RESEARCH_ONLY": "Non-commercial / academic research license only; prohibited for commercial product redistribution.",
    "PENDING_REVIEW_WEAKLY_LABELLED": "License undeclared or weakly supervised labels; permitted for bounded evaluation/prototyping with explicit caveats.",
    "EXTERNAL_BENCHMARK_UNVERIFIED": "External benchmark of unverified origin; isolated from verified organic real-world performance claims.",
    "SYNTHETIC_BENCHMARK": "Synthetic fixture/benchmark; demonstrates pipeline mechanics only — never cite as empirical real-world results.",
    "BLOCKED_MISSING_CREDENTIALS": "Gated source awaiting user credentials/slug verification; skipped automatically in offline builds.",
}


class DatasetRegistryError(KeyError):
    """Raised when an unknown dataset identifier is requested."""


class IntegrityVerificationError(RuntimeError):
    """Raised when downloaded file size, digest, or revision fails verification."""


class LicenseReviewRequiredError(PermissionError):
    """Raised when downloading a dataset whose license is unverified/pending review without override."""


class KaggleCredentialsMissingError(RuntimeError):
    """Raised or caught cleanly when Kaggle API credentials are absent."""


def redact_secrets(text: Any) -> str:
    """Redact potential API tokens, bearer credentials, or keys from text."""
    msg = "" if text is None else str(text)
    for pat in _SECRET_REDACT_PATTERNS:
        if pat.groups >= 1:
            msg = pat.sub(r"\1[REDACTED]", msg)
        else:
            msg = pat.sub("[REDACTED]", msg)
    return msg


@dataclass(frozen=True)
class DatasetFileSpec:
    """Allowlisted file specification with upstream size and integrity digests."""

    url: str
    local_filename: str
    expected_size: Optional[int] = None
    expected_sha256: Optional[str] = None
    expected_md5: Optional[str] = None
    expected_git_sha1: Optional[str] = None
    role: str = "data"  # "data" | "documentation" | "license"


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable metadata and integrity contract for an external dataset."""

    dataset_id: str
    display_name: str
    source_host: str  # "huggingface" | "github" | "zenodo" | "kaggle"
    source_url: str
    pinned_revision: str
    platforms: Tuple[str, ...]
    license_name: str
    license_status: str  # "VERIFIED" | "RESEARCH_ONLY" | "PENDING_REVIEW" | "COMMERCIAL_RESTRICTED"
    provenance: SourceProvenance
    intended_modules: Tuple[str, ...]
    files: Tuple[DatasetFileSpec, ...] = ()
    requires_kaggle_auth: bool = False
    kaggle_slug: Optional[str] = None
    contains_pii_risk: bool = False
    leakage_warnings: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    notes: str = ""
    governance_tier: Optional[str] = None

    @property
    def effective_governance_tier(self) -> str:
        return classify_governance_tier(self)

    def to_summary_dict(self) -> Dict[str, Any]:
        tier = self.effective_governance_tier
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "source_host": self.source_host,
            "source_url": self.source_url,
            "pinned_revision": self.pinned_revision,
            "platforms": list(self.platforms),
            "license_name": self.license_name,
            "license_status": self.license_status,
            "governance_tier": tier,
            "governance_restriction": _GOVERNANCE_RESTRICTIONS.get(tier, ""),
            "provenance": self.provenance.value,
            "intended_modules": list(self.intended_modules),
            "requires_kaggle_auth": self.requires_kaggle_auth,
            "kaggle_slug": self.kaggle_slug,
            "contains_pii_risk": self.contains_pii_risk,
            "leakage_warnings": list(self.leakage_warnings),
            "limitations": list(self.limitations),
            "file_count": len(self.files),
            "total_expected_bytes": sum(f.expected_size or 0 for f in self.files),
            "notes": self.notes,
        }


def classify_governance_tier(spec: DatasetSpec) -> str:
    """Derive standardized six-tier data governance classification from registry contract."""
    if spec.governance_tier in GOVERNANCE_TIERS:
        return spec.governance_tier
    if spec.requires_kaggle_auth:
        return "BLOCKED_MISSING_CREDENTIALS"
    if spec.provenance == SourceProvenance.SYNTHETIC_DATA:
        return "SYNTHETIC_BENCHMARK"
    if spec.provenance == SourceProvenance.WEAKLY_LABELLED_DATA:
        return "PENDING_REVIEW_WEAKLY_LABELLED"
    if spec.provenance == SourceProvenance.EXTERNAL_BENCHMARK_UNVERIFIED:
        return "EXTERNAL_BENCHMARK_UNVERIFIED"
    if (
        "NC" in spec.license_name.upper()
        or "NON_COMMERCIAL" in spec.license_status.upper()
        or "RESEARCH" in spec.license_status.upper()
    ):
        return "NON_COMMERCIAL_RESEARCH_ONLY"
    if spec.license_status == "VERIFIED" and spec.provenance in (
        SourceProvenance.REAL_PUBLIC_DATA,
        SourceProvenance.OWNER_AUTHORIZED_DATA,
    ):
        return "VERIFIED_COMMERCIAL_SAFE"
    return "PENDING_REVIEW_WEAKLY_LABELLED"


def _hf_url(repo: str, rev: str, path: str) -> str:
    quoted = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{quoted}"


def _gh_url(repo: str, rev: str, path: str) -> str:
    quoted = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{repo}/{rev}/{quoted}"


def _build_registry() -> Dict[str, DatasetSpec]:
    specs: List[DatasetSpec] = [
        # --- DATASET A: TikTok 200K Longitudinal Video Engagement Panel ---
        DatasetSpec(
            dataset_id="tiktok_engagement_200k",
            display_name="TikTok Video Engagement 200K Longitudinal Panel (lingbow)",
            source_host="huggingface",
            source_url="https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k",
            pinned_revision="ef0da889e05dea58754bc0ce2d0e2ed6b697fdf2",
            platforms=("tiktok",),
            license_name="CC-BY-NC-4.0",
            license_status="NON_COMMERCIAL_CC_BY_NC_4_0",
            provenance=SourceProvenance.REAL_PUBLIC_DATA,
            intended_modules=("M1", "M2", "M5", "M6", "M7"),
            files=(
                DatasetFileSpec(
                    url=_hf_url("lingbow/tiktok-video-engagement-200k", "ef0da889e05dea58754bc0ce2d0e2ed6b697fdf2", "README.md"),
                    local_filename="README.md",
                    expected_size=9264,
                    expected_git_sha1="f5d0c424fcb0f64adbeb7d23e8055205cdb7309f",
                    role="documentation",
                ),
                DatasetFileSpec(
                    url=_hf_url("lingbow/tiktok-video-engagement-200k", "ef0da889e05dea58754bc0ce2d0e2ed6b697fdf2", "creator_daily.parquet"),
                    local_filename="creator_daily.parquet",
                    expected_size=2322214,
                    expected_sha256="e41423dd49770a67c692c0b3ef6c0b7770c8e1206a423a5d2776a3552686d4d9",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("lingbow/tiktok-video-engagement-200k", "ef0da889e05dea58754bc0ce2d0e2ed6b697fdf2", "engagement_daily.parquet"),
                    local_filename="engagement_daily.parquet",
                    expected_size=26288243,
                    expected_sha256="746e98af1c06ebf041251b34505b13f54bb18374a4fb61a0f39fc15731697f82",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("lingbow/tiktok-video-engagement-200k", "ef0da889e05dea58754bc0ce2d0e2ed6b697fdf2", "videos.parquet"),
                    local_filename="videos.parquet",
                    expected_size=106922687,
                    expected_sha256="368a9c92a700fa9968bbfc40d8fda3c5a2921d24f17fb88e673e48d9bbb5a853",
                    role="data",
                ),
            ),
            limitations=(
                "Non-commercial license (CC-BY-NC-4.0).",
                "TikTok `collect_count` is a platform bookmark proxy distinct from Instagram Insights `saves`.",
                "Cumulative daily trajectories may contain small negative deltas due to user un-likes/deletion or counter resets.",
                "Public scrape cohort; `impressions`, `reach`, and exact `retention_rate` are unavailable.",
            ),
            notes="Primary real longitudinal panel (~209.6K videos, ~6.07M daily snapshots, ~10K creator snapshots).",
        ),
        # --- DATASET B: Instagram Engagement EDA (HF mirror; license undeclared) ---
        DatasetSpec(
            dataset_id="instagram_engagement_eda_hf",
            display_name="Instagram Engagement EDA Mirror (maayan890/instagram-engagement-eda)",
            source_host="huggingface",
            source_url="https://huggingface.co/datasets/maayan890/instagram-engagement-eda",
            pinned_revision="15d719771d6b7c403be626c0f1261b899db71091",
            platforms=("instagram",),
            license_name="NONE_DECLARED",
            license_status="PENDING_REVIEW",
            provenance=SourceProvenance.EXTERNAL_BENCHMARK_UNVERIFIED,
            intended_modules=("M1_benchmark", "M2_benchmark", "audit"),
            files=(
                DatasetFileSpec(
                    url=_hf_url("maayan890/instagram-engagement-eda", "15d719771d6b7c403be626c0f1261b899db71091", "Instagram_Analytics.csv"),
                    local_filename="Instagram_Analytics.csv",
                    expected_size=4130524,
                    expected_git_sha1="06abeb23fa18f03812985eac9fa41997dee15f18",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("maayan890/instagram-engagement-eda", "15d719771d6b7c403be626c0f1261b899db71091", "REDME.md"),
                    local_filename="REDME.md",
                    expected_size=1565,
                    expected_git_sha1="942f33c6cdf1c4e1a356cb9081d7f56077155065",
                    role="documentation",
                ),
            ),
            leakage_warnings=(
                "Exclude `performance_bucket_label`, `engagement_rate`, and post-outcome metrics from pre-publication features.",
            ),
            limitations=(
                "License undeclared (`PENDING_REVIEW`) — excluded from strict verified real-world claims.",
                "Suspected synthetic or stripped-handle classroom benchmark; retains `.mp4`/`.ipynb` upstream exclusions.",
            ),
            notes="Excludes upstream 37.6 MB `.mp4` and `.ipynb` files by allowlist.",
        ),
        # --- DATASET G_alt: CodeMaverick-143 GitHub Instagram_Analytics (MIT) ---
        DatasetSpec(
            dataset_id="instagram_analytics_github_alt",
            display_name="Instagram Analytics Benchmark (GitHub CodeMaverick-143)",
            source_host="github",
            source_url="https://github.com/CodeMaverick-143/Instagram_Analytics",
            pinned_revision="bf881939eda6de7f92ca2a67d7f865fa40fb57e3",
            platforms=("instagram",),
            license_name="MIT",
            license_status="VERIFIED_MIT_ORIGIN_UNVERIFIED",
            provenance=SourceProvenance.EXTERNAL_BENCHMARK_UNVERIFIED,
            intended_modules=("M1_benchmark", "M2_benchmark", "audit"),
            files=(
                DatasetFileSpec(
                    url=_gh_url("CodeMaverick-143/Instagram_Analytics", "bf881939eda6de7f92ca2a67d7f865fa40fb57e3", "Instagram_Analytics.csv"),
                    local_filename="Instagram_Analytics.csv",
                    expected_size=3409374,
                    expected_git_sha1="6ca35dc5b82d7cb5e182aa4f50f442a8a7ab6912",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_gh_url("CodeMaverick-143/Instagram_Analytics", "bf881939eda6de7f92ca2a67d7f865fa40fb57e3", "LICENSE"),
                    local_filename="LICENSE",
                    expected_size=1069,
                    expected_git_sha1="7ac0632615c600d4102b2ce3b6ccd7b266034244",
                    role="license",
                ),
                DatasetFileSpec(
                    url=_gh_url("CodeMaverick-143/Instagram_Analytics", "bf881939eda6de7f92ca2a67d7f865fa40fb57e3", "README.md"),
                    local_filename="README.md",
                    expected_size=4361,
                    expected_git_sha1="c7e8443ebdbace2f60b4b10c35e707386fe8dcba",
                    role="documentation",
                ),
            ),
            leakage_warnings=(
                "Audit overlap with `instagram_engagement_eda_hf` and exclude derived engagement/bucket labels from pre-publication features.",
            ),
            limitations=(
                "Repository includes MIT LICENSE, but underlying post rows exhibit synthetic/standardized properties; treated as external benchmark rather than verified organic platform telemetry.",
            ),
            notes="Verified MIT GitHub repository alternative to Kaggle `kundanbedmutha/instagram-analytics-dataset`.",
        ),
        # --- DATASET C: InstaArt Human vs AI Instagram Posts & Comments ---
        DatasetSpec(
            dataset_id="instaart_human_ai",
            display_name="InstaArt Human vs AI Art Instagram Posts & Comments (sparklessszzz)",
            source_host="huggingface",
            source_url="https://huggingface.co/datasets/sparklessszzz/InstaArt-HumanAI",
            pinned_revision="9516b48e688929d62b98193a25b57c5661cc7d6b",
            platforms=("instagram",),
            license_name="OTHER_ACADEMIC_RESEARCH",
            license_status="RESEARCH_ONLY",
            provenance=SourceProvenance.REAL_PUBLIC_DATA,
            intended_modules=("M1", "M3", "M6", "M7"),
            files=(
                DatasetFileSpec(
                    url=_hf_url("sparklessszzz/InstaArt-HumanAI", "9516b48e688929d62b98193a25b57c5661cc7d6b", "Posts_Primary.csv"),
                    local_filename="Posts_Primary.csv",
                    expected_size=14725,
                    expected_git_sha1="01787f20f7cfb8129d2f1636ec35bc3481a02c1a",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("sparklessszzz/InstaArt-HumanAI", "9516b48e688929d62b98193a25b57c5661cc7d6b", "Human_Comments_Data.csv"),
                    local_filename="Human_Comments_Data.csv",
                    expected_size=64918,
                    expected_sha256="c8570ac8c0ca271d41ece6f38c91d75cc4bd941394a52169dab3d3641387d8a8",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("sparklessszzz/InstaArt-HumanAI", "9516b48e688929d62b98193a25b57c5661cc7d6b", "Human_CommentDate.csv"),
                    local_filename="Human_CommentDate.csv",
                    expected_size=32783,
                    expected_sha256="17717452bb3cc6b1407b669ff6c40647970ec6f80a3c703e6f5a657938635a1f",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("sparklessszzz/InstaArt-HumanAI", "9516b48e688929d62b98193a25b57c5661cc7d6b", "AI_Comments_Data.csv"),
                    local_filename="AI_Comments_Data.csv",
                    expected_size=46207,
                    expected_sha256="1b3c4f13aca16ce01c93d5497cf25ebdc95fbe81a1c7ce790b96c16b272ec8b2",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("sparklessszzz/InstaArt-HumanAI", "9516b48e688929d62b98193a25b57c5661cc7d6b", "AI_CommentDate_Data.csv"),
                    local_filename="AI_CommentDate_Data.csv",
                    expected_size=28888,
                    expected_sha256="00bd97354d2f0655e073fd22936aa68c2b07f025b2d2f632f959773eb4ce04c4",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("sparklessszzz/InstaArt-HumanAI", "9516b48e688929d62b98193a25b57c5661cc7d6b", "README.md"),
                    local_filename="README.md",
                    expected_size=10475,
                    expected_git_sha1="4386df5339757f1e19e7a96c31a4699952f6e878",
                    role="documentation",
                ),
            ),
            contains_pii_risk=True,
            limitations=(
                "Academic research license only (`other`).",
                "Wide `Comment_1..Comment_N` columns represent scrape display order (`comment_rank_index`), NOT chronological posting order unless explicitly joined with date CSVs.",
                "Public scrape metrics only (`saves`, `shares`, `reach`, `impressions` absent).",
            ),
            notes="Real Instagram visual art cohort comparing Human vs AI accounts across posts & timestamped comments.",
        ),
        # --- DATASET D: Instagram Comments Sentiment (HF, PII + Weakly/Held-out) ---
        DatasetSpec(
            dataset_id="instagram_comments_sentiment_hf",
            display_name="Instagram Comments Sentiment (pgurazada1/instagram-comments-sentiment)",
            source_host="huggingface",
            source_url="https://huggingface.co/datasets/pgurazada1/instagram-comments-sentiment",
            pinned_revision="a6b5d2b30a6497edd4d2fa02e7920f769af3c5e2",
            platforms=("instagram",),
            license_name="NONE_DECLARED",
            license_status="PENDING_REVIEW",
            provenance=SourceProvenance.WEAKLY_LABELLED_DATA,
            intended_modules=("M3",),
            files=(
                DatasetFileSpec(
                    url=_hf_url("pgurazada1/instagram-comments-sentiment", "a6b5d2b30a6497edd4d2fa02e7920f769af3c5e2", "train.csv"),
                    local_filename="train.csv",
                    expected_size=28708,
                    expected_git_sha1="db0324c68050d4c9d79d1552e1ce214be4fdbf8d",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("pgurazada1/instagram-comments-sentiment", "a6b5d2b30a6497edd4d2fa02e7920f769af3c5e2", "test.csv"),
                    local_filename="test.csv",
                    expected_size=8751,
                    expected_git_sha1="471148df54f40c725f0c6867d3c85bbcc8a57d51",
                    role="data",
                ),
            ),
            contains_pii_risk=True,
            limitations=(
                "License undeclared (`PENDING_REVIEW`) and contains raw user identifiers/handles.",
                "Labels may be GPT/weakly supervised (`WEAKLY_LABELLED_DATA`).",
                "`test.csv` is strictly reserved for out-of-sample evaluation and NEVER used during training or tuning.",
            ),
            notes="Small 256-comment labeled sentiment corpus (200 train / 56 test) requiring PII pseudonymization.",
        ),
        # --- DATASET E: Synthetic Social Instagram Marketing (CC-BY-4.0, tabular only) ---
        DatasetSpec(
            dataset_id="social_instagram_marketing_hf",
            display_name="Social Instagram Marketing Tabular Subset (viswapriyan33, Synthetic)",
            source_host="huggingface",
            source_url="https://huggingface.co/datasets/viswapriyan33/social-instagram-marketing",
            pinned_revision="b1f798e6a4444b15d39964f03ad6d6b1d4c7740d",
            platforms=("instagram",),
            license_name="CC-BY-4.0",
            license_status="VERIFIED",
            provenance=SourceProvenance.SYNTHETIC_DATA,
            intended_modules=("M1_synthetic", "M5_synthetic"),
            files=(
                DatasetFileSpec(
                    url=_hf_url("viswapriyan33/social-instagram-marketing", "b1f798e6a4444b15d39964f03ad6d6b1d4c7740d", "social_dataset.parquet"),
                    local_filename="social_dataset.parquet",
                    expected_size=178022,
                    expected_sha256="242f949e52d3b1cded3311a58d1a595dff0600191302b5d9e9b4dc0f711773ff",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("viswapriyan33/social-instagram-marketing", "b1f798e6a4444b15d39964f03ad6d6b1d4c7740d", "social_dataset.csv"),
                    local_filename="social_dataset.csv",
                    expected_size=482401,
                    expected_git_sha1="9adbe89a22f12013997186a55588aff48f650a04",
                    role="data",
                ),
                DatasetFileSpec(
                    url=_hf_url("viswapriyan33/social-instagram-marketing", "b1f798e6a4444b15d39964f03ad6d6b1d4c7740d", "README.md"),
                    local_filename="README.md",
                    expected_size=5928,
                    expected_git_sha1="08eb8980c8e85f9231785157620218e17d6cb7aa",
                    role="documentation",
                ),
            ),
            limitations=(
                "Explicitly synthetic campaign/caption dataset — MUST NOT be represented as real organic Instagram performance.",
                "Upstream 1,000+ PNG images (>1 GB) intentionally excluded by allowlist.",
            ),
            notes="1,000 synthetic Instagram marketing posts with captions, hashtags, formats, and campaign attributes.",
        ),
        # --- ZENODO 5082114: Sponsored vs Unsponsored Instagram Posts (CC-BY-4.0) ---
        DatasetSpec(
            dataset_id="zenodo_sponsored_unsponsored_ig",
            display_name="Sponsored and Unsponsored Posts on Instagram (Zenodo 5082114)",
            source_host="zenodo",
            source_url="https://zenodo.org/records/5082114",
            pinned_revision="doi:10.5281/zenodo.5082114",
            platforms=("instagram",),
            license_name="CC-BY-4.0",
            license_status="VERIFIED",
            provenance=SourceProvenance.REAL_PUBLIC_DATA,
            intended_modules=("M1", "M4_observational", "M5"),
            files=(
                DatasetFileSpec(
                    url="https://zenodo.org/api/records/5082114/files/sponsored_posts.csv/content",
                    local_filename="sponsored_posts.csv",
                    expected_size=1217896,
                    expected_md5="b480cf9a19554739a965a1c212733177",
                    role="data",
                ),
                DatasetFileSpec(
                    url="https://zenodo.org/api/records/5082114/files/unsponsored_posts.csv/content",
                    local_filename="unsponsored_posts.csv",
                    expected_size=1153368,
                    expected_md5="544dd9bdca66bc33f389a7c1cfc0ae45",
                    role="data",
                ),
            ),
            contains_pii_risk=True,
            limitations=(
                "Observational comparison (sponsored vs unsponsored creator posts) — strictly non-causal without randomization.",
                "Raw creator/sponsor handles pseudonymized (`creator_pseudo_id`) and dropped before analytics ingestion (pseudonymization, not irreversible anonymization).",
            ),
            notes="Peer-archived real Instagram creator post comparison under CC-BY-4.0.",
        ),
        # --- ZENODO 23301344: Educational Instagram Reels (CC-BY-4.0) ---
        DatasetSpec(
            dataset_id="zenodo_educational_reels_ig",
            display_name="Educational Instagram Reels 300-Post Coded Dataset (Zenodo 23301344)",
            source_host="zenodo",
            source_url="https://zenodo.org/records/23301344",
            pinned_revision="doi:10.5281/zenodo.23301344",
            platforms=("instagram",),
            license_name="CC-BY-4.0",
            license_status="VERIFIED",
            provenance=SourceProvenance.REAL_PUBLIC_DATA,
            intended_modules=("M1", "M5"),
            files=(
                DatasetFileSpec(
                    url="https://zenodo.org/api/records/23301344/files/Reels%20analysis.xlsx/content",
                    local_filename="Reels analysis.xlsx",
                    expected_size=120166,
                    expected_md5="b86000d0de934c01774bc9e164ea33d2",
                    role="data",
                ),
                DatasetFileSpec(
                    url="https://zenodo.org/api/records/23301344/files/Educational%20Instagram%20Reels%20Statistical%20Analysis%20and%20Findings.pdf/content",
                    local_filename="Educational Instagram Reels Statistical Analysis and Findings.pdf",
                    expected_size=887938,
                    expected_md5="39423073f32b7c2bcb8fccb7c4d5c149",
                    role="documentation",
                ),
            ),
            limitations=(
                "N=300 Reels across 21 educational accounts; domain-specific short-video sample.",
            ),
            notes="Coded dataset parsed natively via Python stdlib XML `.xlsx` reader (no openpyxl required).",
        ),
        # --- ZENODO 15638548: Lokananta Cultural Heritage Instagram Comments (CC-BY-4.0) ---
        DatasetSpec(
            dataset_id="zenodo_lokananta_comments_ig",
            display_name="Lokananta Cultural Heritage Instagram Comments (Zenodo 15638548)",
            source_host="zenodo",
            source_url="https://zenodo.org/records/15638548",
            pinned_revision="doi:10.5281/zenodo.15638548",
            platforms=("instagram",),
            license_name="CC-BY-4.0",
            license_status="VERIFIED",
            provenance=SourceProvenance.REAL_PUBLIC_DATA,
            intended_modules=("M3_multilingual_edge",),
            files=(
                DatasetFileSpec(
                    url="https://zenodo.org/api/records/15638548/files/Instagram%20Comment%20Data%20From%20Lokananta%20Cultural%20Heritage%20For%20Sentiment%20Analysis%20.csv/content",
                    local_filename="lokananta_instagram_comments.csv",
                    expected_size=28414,
                    expected_md5="a14d218839fb69303dbe4b8e99c88ab8",
                    role="data",
                ),
            ),
            contains_pii_risk=True,
            limitations=(
                "Predominantly Indonesian/multilingual cultural heritage comments — explicitly exercises non-English & spam handling in M3.",
            ),
            notes="Multilingual & emoji/spam robustness audit corpus for Module 3.",
        ),
        # --- KAGGLE GATED SPECS (Require user Kaggle CLI credentials) ---
        DatasetSpec(
            dataset_id="youtube_trending_kaggle",
            display_name="YouTube Trending Video Multi-Region Dataset (Kaggle rsrishav)",
            source_host="kaggle",
            source_url="https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset",
            pinned_revision="kaggle:latest-upon-auth",
            platforms=("youtube",),
            license_name="CC0-1.0_KAGGLE",
            license_status="REQUIRES_KAGGLE_AUTH",
            provenance=SourceProvenance.REAL_PUBLIC_DATA,
            intended_modules=("M1", "M2", "M6", "M7"),
            requires_kaggle_auth=True,
            kaggle_slug="rsrishav/youtube-trending-video-dataset",
            limitations=(
                "Requires Kaggle API authentication (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`+`KAGGLE_KEY`).",
                "Subject to YouTube Trending tab curation/selection bias — NOT representative of average YouTube channels.",
                "Rows are `(video_id, region, trending_date)` snapshots and must preserve region/date composite keys.",
            ),
            notes="Ingestion adapter & schema contract implemented; acquisition awaits Kaggle credentials.",
        ),
        DatasetSpec(
            dataset_id="instagram_analytics_kaggle",
            display_name="Instagram Analytics Dataset (Kaggle kundanbedmutha)",
            source_host="kaggle",
            source_url="https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset",
            pinned_revision="kaggle:latest-upon-auth",
            platforms=("instagram",),
            license_name="UNKNOWN_REQUIRES_KAGGLE_AUTH",
            license_status="REQUIRES_KAGGLE_AUTH",
            provenance=SourceProvenance.EXTERNAL_BENCHMARK_UNVERIFIED,
            intended_modules=("M1_benchmark", "audit"),
            requires_kaggle_auth=True,
            kaggle_slug="kundanbedmutha/instagram-analytics-dataset",
            limitations=(
                "Requires Kaggle API credentials; compare schema/SHA overlap against `instagram_analytics_github_alt` once downloaded.",
            ),
            notes="Kaggle-gated counterpart; GitHub `instagram_analytics_github_alt` serves as public inspectable benchmark.",
        ),
        DatasetSpec(
            dataset_id="youtube_kaggle_candidate_h",
            display_name="YouTube Video Performance Candidate (Aviral Trivedi — Unverified Host/Slug)",
            source_host="kaggle",
            source_url="https://www.kaggle.com/",
            pinned_revision="unverified",
            platforms=("youtube",),
            license_name="UNKNOWN",
            license_status="BLOCKED_MISSING_CREDENTIALS_AND_UNVERIFIED_SLUG",
            provenance=SourceProvenance.EXTERNAL_BENCHMARK_UNVERIFIED,
            intended_modules=("audit",),
            requires_kaggle_auth=True,
            kaggle_slug=None,
            limitations=(
                "Exact Kaggle slug unverified and Kaggle credentials absent; never guesses or fabricates dataset URLs.",
            ),
            notes="Tracked transparently as blocked pending exact slug verification & Kaggle authentication.",
        ),
    ]
    return {spec.dataset_id: spec for spec in specs}


DATASET_REGISTRY: Dict[str, DatasetSpec] = _build_registry()


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """Return the immutable `DatasetSpec` for `dataset_id` or raise `DatasetRegistryError`."""
    if dataset_id not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise DatasetRegistryError(f"Unknown dataset '{dataset_id}'. Available datasets: {known}")
    return DATASET_REGISTRY[dataset_id]


def list_dataset_specs(*, include_kaggle: bool = True) -> List[DatasetSpec]:
    """Return registered datasets in deterministic order."""
    specs = list(DATASET_REGISTRY.values())
    if not include_kaggle:
        specs = [s for s in specs if not s.requires_kaggle_auth]
    return specs


def _compute_file_digests(path: Path) -> Dict[str, Any]:
    sha256_h = hashlib.sha256()
    md5_h = hashlib.md5()
    size = path.stat().st_size
    git_sha1_h = hashlib.sha1()
    git_sha1_h.update(f"blob {size}\0".encode("ascii"))

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha256_h.update(chunk)
            md5_h.update(chunk)
            git_sha1_h.update(chunk)

    return {
        "size": int(size),
        "sha256": sha256_h.hexdigest(),
        "md5": md5_h.hexdigest(),
        "git_sha1": git_sha1_h.hexdigest(),
    }


def verify_downloaded_file(path: Path, spec: DatasetFileSpec) -> Dict[str, Any]:
    """Verify exact byte size and cryptographic digests of a downloaded file."""
    if not path.exists():
        raise IntegrityVerificationError(f"File does not exist for verification: {path}")

    digests = _compute_file_digests(path)
    if spec.expected_size is not None and digests["size"] != spec.expected_size:
        raise IntegrityVerificationError(
            f"Size mismatch for {path.name}: expected {spec.expected_size} bytes, got {digests['size']} bytes."
        )
    if spec.expected_sha256 is not None and digests["sha256"].lower() != spec.expected_sha256.lower():
        raise IntegrityVerificationError(
            f"SHA-256 mismatch for {path.name}: expected {spec.expected_sha256}, got {digests['sha256']}."
        )
    if spec.expected_md5 is not None and digests["md5"].lower() != spec.expected_md5.lower():
        raise IntegrityVerificationError(
            f"MD5 mismatch for {path.name}: expected {spec.expected_md5}, got {digests['md5']}."
        )
    if spec.expected_git_sha1 is not None and digests["git_sha1"].lower() != spec.expected_git_sha1.lower():
        raise IntegrityVerificationError(
            f"Git blob SHA-1 mismatch for {path.name}: expected {spec.expected_git_sha1}, got {digests['git_sha1']}."
        )
    return digests


def download_file_resumable(
    spec: DatasetFileSpec,
    dest_path: Path,
    *,
    force: bool = False,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    timeout: float = 60.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Download a pinned file with HTTP Range resume, exponential backoff, and atomic integrity promotion."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")

    # Fast reuse when intact target file already passes verification
    if not force and dest_path.exists():
        try:
            digests = verify_downloaded_file(dest_path, spec)
            return {
                "filename": spec.local_filename,
                "path": str(dest_path),
                "url": spec.url,
                "status": "cached_verified",
                **digests,
            }
        except IntegrityVerificationError:
            # Remove corrupt/stale target and re-fetch cleanly
            dest_path.unlink(missing_ok=True)

    if force and part_path.exists():
        part_path.unlink(missing_ok=True)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            existing_bytes = part_path.stat().st_size if part_path.exists() else 0
            headers = {"User-Agent": "SocialPulse-DatasetDownloader/1.0"}
            if existing_bytes > 0 and (spec.expected_size is None or existing_bytes < spec.expected_size):
                headers["Range"] = f"bytes={existing_bytes}-"

            req = urllib.request.Request(spec.url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = getattr(resp, "status", None) or resp.getcode() or 200
                mode = "ab" if (status_code == 206 and existing_bytes > 0) else "wb"
                with open(part_path, mode) as out_handle:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        out_handle.write(chunk)

            digests = verify_downloaded_file(part_path, spec)
            os.replace(part_path, dest_path)
            return {
                "filename": spec.local_filename,
                "path": str(dest_path),
                "url": spec.url,
                "status": "downloaded_verified",
                "resumed_from_bytes": existing_bytes if mode == "ab" else 0,
                **digests,
            }
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 416 and part_path.exists():
                # Existing .part already reached/exceeded EOF — reset .part and retry once
                part_path.unlink(missing_ok=True)
            elif exc.code not in _RETRYABLE_HTTP_CODES or attempt >= max_retries:
                raise RuntimeError(
                    f"HTTP {exc.code} while downloading '{spec.local_filename}': {redact_secrets(exc.reason)}"
                ) from exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Network failure while downloading '{spec.local_filename}': {redact_secrets(exc)}"
                ) from exc
        except IntegrityVerificationError as exc:
            # Remove bad .part so corrupted fragments aren't resumed indefinitely
            part_path.unlink(missing_ok=True)
            raise IntegrityVerificationError(
                f"Integrity check failed for '{spec.local_filename}': {redact_secrets(exc)}"
            ) from exc

        if attempt < max_retries:
            sleep_fn(backoff_base * (2 ** attempt))

    raise RuntimeError(
        f"Failed downloading '{spec.local_filename}' after {max_retries + 1} attempts: {redact_secrets(last_error)}"
    )


def check_kaggle_credentials(kaggle_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Check whether Kaggle CLI/API credentials are configured without reading/returning secrets."""
    kdir = kaggle_dir if kaggle_dir is not None else (Path.home() / ".kaggle")
    json_path = kdir / "kaggle.json"
    has_json = json_path.exists() and json_path.is_file() and json_path.stat().st_size > 0
    has_env_pair = bool(os.environ.get("KAGGLE_USERNAME", "").strip() and os.environ.get("KAGGLE_KEY", "").strip())
    has_env_token = bool(os.environ.get("KAGGLE_API_TOKEN", "").strip())
    cli_path = shutil.which("kaggle")

    return {
        "has_kaggle_json": has_json,
        "has_env_pair": has_env_pair,
        "has_env_token": has_env_token,
        "has_credentials": bool(has_json or has_env_pair or has_env_token),
        "cli_available": bool(cli_path),
        "kaggle_json_path": str(json_path),
    }


def download_kaggle_dataset(
    spec: DatasetSpec,
    dest_dir: Path,
    *,
    raise_on_missing_creds: bool = False,
    kaggle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Attempt Kaggle dataset acquisition via Kaggle CLI when authenticated, otherwise report blocked cleanly."""
    creds = check_kaggle_credentials(kaggle_dir=kaggle_dir)
    manual_instructions = (
        f"Configure ~/.kaggle/kaggle.json (chmod 600) or set KAGGLE_USERNAME & KAGGLE_KEY, then run: "
        f"`kaggle datasets download -d {spec.kaggle_slug} -p \"{dest_dir}\" --unzip`"
        if spec.kaggle_slug
        else "Exact Kaggle dataset slug has not been verified yet."
    )

    if not spec.kaggle_slug:
        msg = f"[{spec.dataset_id}] BLOCKED: Exact Kaggle dataset slug is unverified."
        if raise_on_missing_creds:
            raise KaggleCredentialsMissingError(msg)
        return {
            "dataset_id": spec.dataset_id,
            "status": "BLOCKED_UNVERIFIED_SLUG",
            "credentials": creds,
            "message": msg,
            "instructions": manual_instructions,
        }

    if not creds["has_credentials"] or not creds["cli_available"]:
        reason = (
            "Kaggle API credentials missing (~/.kaggle/kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY / KAGGLE_API_TOKEN)"
            if not creds["has_credentials"]
            else "`kaggle` CLI binary not installed on PATH"
        )
        msg = f"[{spec.dataset_id}] BLOCKED_MISSING_CREDENTIALS: {reason}. {manual_instructions}"
        if raise_on_missing_creds:
            raise KaggleCredentialsMissingError(redact_secrets(msg))
        return {
            "dataset_id": spec.dataset_id,
            "status": "BLOCKED_MISSING_CREDENTIALS",
            "credentials": creds,
            "message": redact_secrets(msg),
            "instructions": manual_instructions,
        }

    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", spec.kaggle_slug, "-p", str(dest_dir), "--unzip"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err_clean = redact_secrets(proc.stderr or proc.stdout or "Kaggle CLI execution failed.")
        return {
            "dataset_id": spec.dataset_id,
            "status": "FAILED",
            "credentials": creds,
            "message": err_clean,
            "instructions": manual_instructions,
        }

    files_info = [
        {"filename": p.name, "path": str(p), **_compute_file_digests(p)}
        for p in sorted(dest_dir.rglob("*"))
        if p.is_file() and not p.name.endswith(".zip")
    ]
    return {
        "dataset_id": spec.dataset_id,
        "status": "downloaded_verified",
        "files": files_info,
    }


def _col_letters_to_index(col_ref: str) -> int:
    idx = 0
    for ch in col_ref:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(idx - 1, 0)


def list_xlsx_sheet_names(xlsx_path: Path) -> List[str]:
    """Return ordered worksheet names from an `.xlsx` workbook using stdlib XML."""
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    names: List[str] = []
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        if "xl/workbook.xml" in zf.namelist():
            wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
            for sh in wb_root.findall(f".//{{{ns_main}}}sheet"):
                names.append(sh.attrib.get("name", f"sheet_{len(names)}"))
    return names


def read_xlsx_sheet_to_dataframe(xlsx_path: Path, sheet_index: int | str = 0) -> pd.DataFrame:
    """Parse a worksheet (by 0-based index or exact sheet name) using Python stdlib XML + zipfile.

    Avoids introducing an optional `openpyxl` runtime dependency for clean cloud deployments
    while extracting typed rows cleanly from Excel spreadsheets such as Zenodo 23301344.
    """
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_ss.findall(f"{{{ns_main}}}si"):
                parts = [t.text or "" for t in si.iter(f"{{{ns_main}}}t")]
                shared_strings.append("".join(parts))

        # Resolve ordered sheet XML target from workbook.xml + workbook.xml.rels
        sheet_paths: List[str] = []
        sheet_name_map: Dict[str, str] = {}
        if "xl/workbook.xml" in zf.namelist() and "xl/_rels/workbook.xml.rels" in zf.namelist():
            wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
            rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rel_map = {
                rel.attrib.get("Id", ""): rel.attrib.get("Target", "")
                for rel in rels_root
            }
            for sh in wb_root.findall(f".//{{{ns_main}}}sheet"):
                rid = sh.attrib.get(f"{{{ns_rel}}}id", "")
                sh_name = sh.attrib.get("name", "")
                target = rel_map.get(rid, "").lstrip("/")
                if target and not target.startswith("xl/"):
                    target = f"xl/{target}"
                if target in zf.namelist():
                    sheet_paths.append(target)
                    if sh_name:
                        sheet_name_map[sh_name] = target

        if not sheet_paths:
            sheet_paths = sorted(p for p in zf.namelist() if p.startswith("xl/worksheets/sheet") and p.endswith(".xml"))

        if not sheet_paths:
            return pd.DataFrame()
        if isinstance(sheet_index, str):
            if sheet_index not in sheet_name_map:
                raise KeyError(f"Worksheet '{sheet_index}' not found in {xlsx_path.name}: {list(sheet_name_map)}")
            chosen_sheet = sheet_name_map[sheet_index]
        else:
            chosen_sheet = sheet_paths[min(max(int(sheet_index), 0), len(sheet_paths) - 1)]
        sh_root = ET.fromstring(zf.read(chosen_sheet))

    raw_rows: List[Dict[int, Any]] = []
    max_col = 0
    for row_el in sh_root.findall(f".//{{{ns_main}}}sheetData/{{{ns_main}}}row"):
        row_map: Dict[int, Any] = {}
        fallback_col = 0
        for cell in row_el.findall(f"{{{ns_main}}}c"):
            ref = cell.attrib.get("r", "")
            col_letters = "".join(ch for ch in ref if "A" <= ch <= "Z")
            col_idx = _col_letters_to_index(col_letters) if col_letters else fallback_col
            fallback_col = col_idx + 1
            max_col = max(max_col, col_idx + 1)

            cell_type = cell.attrib.get("t", "")
            val_el = cell.find(f"{{{ns_main}}}v")
            inline_el = cell.find(f"{{{ns_main}}}is")
            val: Any = None
            if cell_type == "s" and val_el is not None and val_el.text is not None:
                ss_i = int(val_el.text)
                val = shared_strings[ss_i] if 0 <= ss_i < len(shared_strings) else val_el.text
            elif cell_type == "inlineStr" and inline_el is not None:
                val = "".join(t.text or "" for t in inline_el.iter(f"{{{ns_main}}}t"))
            elif cell_type == "b" and val_el is not None:
                val = val_el.text == "1"
            elif val_el is not None and val_el.text is not None:
                txt = val_el.text
                try:
                    num = float(txt)
                    val = int(num) if num.is_integer() and "." not in txt and "e" not in txt.lower() else num
                except ValueError:
                    val = txt
            row_map[col_idx] = val
        if row_map:
            raw_rows.append(row_map)

    if not raw_rows:
        return pd.DataFrame()

    header_row = raw_rows[0]
    headers = [
        str(header_row.get(c) if header_row.get(c) not in (None, "") else f"col_{c}").strip()
        for c in range(max_col)
    ]
    data_rows = [[r.get(c) for c in range(max_col)] for r in raw_rows[1:]]
    return pd.DataFrame(data_rows, columns=headers).dropna(how="all").reset_index(drop=True)


def _read_tabular_file(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        return None
    if suffix == ".xlsx":
        names = list_xlsx_sheet_names(path)
        target_sheet: int | str = "Analysis" if "Analysis" in names else 0
        return read_xlsx_sheet_to_dataframe(path, sheet_index=target_sheet)
    return None


def inspect_dataframe_quality(df: pd.DataFrame, table_name: str, source_spec: DatasetSpec) -> Dict[str, Any]:
    """Compute structural, duplicate, missingness, date span, PII heuristic, and leakage audit stats."""
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    nested_cols: List[str] = []
    if n_rows > 0:
        for col in df.columns:
            sample_vals = df[col].dropna().head(10)
            if any(isinstance(v, (np.ndarray, list, dict, set)) for v in sample_vals):
                nested_cols.append(str(col))

    try:
        dup_rows = int(df.duplicated().sum()) if n_rows > 0 else 0
    except TypeError:
        hashable_cols = [c for c in df.columns if str(c) not in nested_cols]
        dup_rows = int(df.duplicated(subset=hashable_cols).sum()) if (n_rows > 0 and hashable_cols) else 0

    col_stats: Dict[str, Dict[str, Any]] = {}
    date_spans: Dict[str, Dict[str, str]] = {}
    pii_columns: List[str] = []
    pii_pattern_counts = {"handles": 0, "emails": 0, "urls": 0}
    leakage_flags: Dict[str, str] = {}

    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower().strip()
        series = df[col]
        null_cnt = int(series.isna().sum())
        null_pct = round((null_cnt / n_rows) * 100.0, 2) if n_rows > 0 else 0.0
        col_stats[col_str] = {
            "dtype": str(series.dtype),
            "null_count": null_cnt,
            "null_pct": null_pct,
        }

        if any(k in col_lower for k in ("user_id", "username", "author", "handle", "owner", "account_name")):
            pii_columns.append(col_str)
        if col_lower in _LEAKAGE_COLUMN_HINTS:
            leakage_flags[col_str] = _LEAKAGE_COLUMN_HINTS[col_lower]

        if any(tok in col_lower for tok in ("date", "time", "created", "publish")) and n_rows > 0:
            sample = series.dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            # Numeric epoch vs ISO/string timestamps
            if pd.api.types.is_numeric_dtype(sample):
                med = float(sample.median()) if not sample.empty else 0.0
                if 1_000_000_000 <= med <= 2_500_000_000:
                    parsed = pd.to_datetime(sample, unit="s", utc=True, errors="coerce").dropna()
                else:
                    parsed = pd.Series(dtype="datetime64[ns, UTC]")
            else:
                parsed = pd.to_datetime(sample, format="mixed", utc=True, errors="coerce").dropna()
            if not parsed.empty:
                date_spans[col_str] = {
                    "min": str(parsed.min()),
                    "max": str(parsed.max()),
                    "sampled_valid": int(len(parsed)),
                }

        # Bounded regex scan on string/object columns for handles/emails/URLs
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            text_sample = series.dropna().astype(str).head(1000)
            for txt in text_sample:
                if "@" in txt:
                    pii_pattern_counts["handles"] += len(_HANDLE_RE.findall(txt))
                    pii_pattern_counts["emails"] += len(_EMAIL_RE.findall(txt))
                if "http" in txt.lower():
                    pii_pattern_counts["urls"] += len(_URL_RE.findall(txt))

    return {
        "table_name": table_name,
        "dataset_id": source_spec.dataset_id,
        "provenance": source_spec.provenance.value,
        "rows": n_rows,
        "columns": n_cols,
        "duplicate_rows": dup_rows,
        "nested_array_columns": nested_cols,
        "column_stats": col_stats,
        "date_spans": date_spans,
        "pii_audit": {
            "flagged_identifier_columns": sorted(set(pii_columns)),
            "sample_pattern_counts": pii_pattern_counts,
        },
        "leakage_audit": {
            "flagged_columns": leakage_flags,
            "registry_warnings": list(source_spec.leakage_warnings),
        },
    }


def generate_raw_quality_report(
    dataset_id: str,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect all downloaded tabular files for `dataset_id` and save JSON + Markdown quality reports."""
    spec = get_dataset_spec(dataset_id)
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw_dir = base / DEFAULT_EXTERNAL_ROOT / dataset_id / "raw"
    reports_dir = base / DEFAULT_EXTERNAL_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    table_reports: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*")) if raw_dir.exists() else []:
        if not path.is_file() or path.suffix.lower() not in {".parquet", ".csv", ".xlsx"}:
            continue
        df = _read_tabular_file(path)
        if df is not None:
            table_reports.append(inspect_dataframe_quality(df, path.name, spec))

    report = {
        "dataset_id": dataset_id,
        "display_name": spec.display_name,
        "provenance": spec.provenance.value,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "governance_tier": spec.effective_governance_tier,
        "pinned_revision": spec.pinned_revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": table_reports,
        "limitations": list(spec.limitations),
    }

    json_out = reports_dir / f"{dataset_id}_quality.json"
    with open(json_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    md_lines = [
        f"# Raw Dataset Quality Report — `{dataset_id}`",
        "",
        f"- **Display Name:** {spec.display_name}",
        f"- **Provenance:** `{spec.provenance.value}`",
        f"- **License:** `{spec.license_name}` (`{spec.license_status}`)",
        f"- **Pinned Revision / DOI:** `{spec.pinned_revision}`",
        "",
    ]
    if not table_reports:
        md_lines.append("_No tabular files present on disk to inspect._")
    for tbl in table_reports:
        md_lines.extend([
            f"## Table `{tbl['table_name']}`",
            f"- Rows: `{tbl['rows']:,}` | Columns: `{tbl['columns']}` | Exact duplicate rows: `{tbl['duplicate_rows']:,}`",
            f"- Flagged identifier/PII columns: `{tbl['pii_audit']['flagged_identifier_columns']}`",
            f"- Sample text PII pattern hits (first 1K non-null strings): `{tbl['pii_audit']['sample_pattern_counts']}`",
        ])
        if tbl["date_spans"]:
            md_lines.append(f"- Date spans: `{tbl['date_spans']}`")
        if tbl["leakage_audit"]["flagged_columns"]:
            md_lines.append(f"- Target-leakage flags: `{tbl['leakage_audit']['flagged_columns']}`")
        md_lines.append("")
    if spec.limitations:
        md_lines.append("## Documented Limitations")
        for lim in spec.limitations:
            md_lines.append(f"- {lim}")

    md_out = reports_dir / f"{dataset_id}_quality.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def load_or_init_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass
    return {
        "schema_version": "1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }


def write_manifest(manifest: Dict[str, Any], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def download_dataset(
    dataset_id: str,
    *,
    root_dir: Optional[Path] = None,
    include_pending_review: bool = False,
    force: bool = False,
    generate_report: bool = True,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Download, verify, record in `manifest.json`, and inspect one registered external dataset.

    Datasets with ``license_status == 'PENDING_REVIEW'`` are rejected by default
    (default-deny governance) unless ``include_pending_review=True`` is passed
    explicitly.
    """
    spec = get_dataset_spec(dataset_id)
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    raw_dir = base / DEFAULT_EXTERNAL_ROOT / dataset_id / "raw"
    manifest_path = base / DEFAULT_EXTERNAL_ROOT / "manifest.json"
    manifest = load_or_init_manifest(manifest_path)

    if spec.requires_kaggle_auth:
        entry = download_kaggle_dataset(spec, raw_dir, raise_on_missing_creds=False)
        entry.update({
            "source_host": spec.source_host,
            "source_url": spec.source_url,
            "pinned_revision": spec.pinned_revision,
            "license_name": spec.license_name,
            "license_status": spec.license_status,
            "governance_tier": spec.effective_governance_tier,
            "provenance": spec.provenance.value,
        })
        manifest["datasets"][dataset_id] = entry
        write_manifest(manifest, manifest_path)
        return entry

    if spec.license_status == "PENDING_REVIEW" and not include_pending_review:
        raise LicenseReviewRequiredError(
            f"Dataset '{dataset_id}' has license_status='PENDING_REVIEW' and was not explicitly permitted."
        )

    downloaded_files: List[Dict[str, Any]] = []
    for file_spec in spec.files:
        target = raw_dir / file_spec.local_filename
        res = download_file_resumable(
            file_spec,
            target,
            force=force,
            max_retries=max_retries,
            backoff_base=backoff_base,
            timeout=timeout,
        )
        res["role"] = file_spec.role
        downloaded_files.append(res)

    entry = {
        "dataset_id": dataset_id,
        "display_name": spec.display_name,
        "status": "downloaded_verified",
        "source_host": spec.source_host,
        "source_url": spec.source_url,
        "pinned_revision": spec.pinned_revision,
        "license_name": spec.license_name,
        "license_status": spec.license_status,
        "governance_tier": spec.effective_governance_tier,
        "provenance": spec.provenance.value,
        "intended_modules": list(spec.intended_modules),
        "raw_dir": str(raw_dir),
        "files": downloaded_files,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["datasets"][dataset_id] = entry
    write_manifest(manifest, manifest_path)

    if generate_report:
        report = generate_raw_quality_report(dataset_id, root_dir=base)
        entry["quality_report_tables"] = len(report["tables"])

    return entry


def download_all_accessible(
    *,
    root_dir: Optional[Path] = None,
    include_pending_review: bool = False,
    attempt_kaggle: bool = True,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch approved non-Kaggle datasets (default-denying PENDING_REVIEW unless opted in)."""
    results: Dict[str, Dict[str, Any]] = {}
    for spec in list_dataset_specs(include_kaggle=attempt_kaggle):
        if spec.license_status == "PENDING_REVIEW" and not include_pending_review:
            results[spec.dataset_id] = {
                "dataset_id": spec.dataset_id,
                "status": "SKIPPED_PENDING_REVIEW",
                "license_status": spec.license_status,
                "governance_tier": spec.effective_governance_tier,
                "requires_explicit_opt_in": True,
            }
            continue
        results[spec.dataset_id] = download_dataset(
            spec.dataset_id,
            root_dir=root_dir,
            include_pending_review=include_pending_review,
            force=force,
        )
    return results
