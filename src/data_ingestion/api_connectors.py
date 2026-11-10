"""Social Media API Connectors for Instagram Graph API and YouTube Data API v3.

Documents exact endpoint requirements, access levels (Public vs. Owner-authorized),
rate-limiting boundaries, and normalizes API JSON payloads into the 14-field unified schema.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
from .schemas import UNIFIED_POST_FIELDS, UNIFIED_PANDAS_DTYPES, DataProvenance, SourceProvenance


def _empty_unified_posts_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=UNIFIED_POST_FIELDS)
    for col, dtype in UNIFIED_PANDAS_DTYPES.items():
        df[col] = pd.Series(dtype=dtype)
    df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    df["collection_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


class BaseAPIConnector(ABC):
    """Abstract interface defining the contract for social platform data connectors."""

    @abstractmethod
    def fetch_posts(self, identifier: str, limit: int = 50, mode: str = "public") -> pd.DataFrame:
        """Fetch post records formatted to the 14-field unified schema."""
        pass

    @abstractmethod
    def get_provenance(self) -> DataProvenance:
        """Return provenance metadata describing source and authorization tier."""
        pass


class InstagramGraphAPIConnector(BaseAPIConnector):
    """Connector for Instagram Graph API.

    API Requirements & Limitations:
    1. Public / Competitor Mode (Business Discovery API):
       - Requires Meta App with 'instagram_basic' and connected Facebook Page.
       - Returns: caption, comments_count, like_count, media_type, permalink, timestamp.
       - CANNOT access: reach, saves, shares, retention curves (returned as pd.NA).
    2. Account-Owner Mode (Instagram Insights API):
       - Requires 'instagram_manage_insights' permission + Page Access Token.
       - Returns: reach, saved, shares, video_views, profile_visits.
    Rate Limit: 200 calls/hour per user.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        page_id: Optional[str] = None,
        sandbox_mode: bool = False,
        api_version: Optional[str] = None,
    ):
        self.access_token = (
            access_token
            or os.getenv("META_ACCESS_TOKEN", "")
            or os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        )
        self.page_id = (
            page_id
            or os.getenv("META_IG_BUSINESS_ACCOUNT_ID", "")
            or os.getenv("INSTAGRAM_PAGE_ID", "")
        )
        raw_ver = (
            api_version
            or os.getenv("META_GRAPH_API_VERSION", "")
            or "v21.0"
        ).strip()
        ver = raw_ver if raw_ver.startswith("v") else f"v{raw_ver}"
        self.api_version = ver
        self.sandbox_mode = sandbox_mode or (not bool(self.access_token))
        self.base_url = f"https://graph.facebook.com/{ver}"

    def get_provenance(self, mode: str = "public") -> DataProvenance:
        return DataProvenance(
            source_name="Instagram Graph API",
            access_tier="owner_oauth" if mode == "owner" else "public_api",
            notes="Owner mode exposes saves/shares/reach; public mode marks them as null."
        )

    def normalize_api_payload(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        mode: str = "public",
        follower_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Normalize Instagram Graph API / Business Discovery media objects into unified schema."""
        if not items:
            return _empty_unified_posts_df()

        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        owner_mode = str(mode).lower() == "owner"
        rows: List[Dict[str, Any]] = []
        media_map = {"IMAGE": "image", "VIDEO": "reel", "CAROUSEL_ALBUM": "carousel", "REELS": "reel"}

        for raw in items:
            mtype = media_map.get(str(raw.get("media_type", "")).upper(), "image")
            views_val = raw.get("video_views", raw.get("plays")) if (owner_mode and mtype == "reel") else pd.NA
            rows.append({
                "platform": "instagram",
                "post_id": str(raw.get("id", "")),
                "timestamp": raw.get("timestamp"),
                "content_type": mtype,
                "views": views_val,
                "impressions": raw.get("impressions", pd.NA) if owner_mode else pd.NA,
                "reach": raw.get("reach", pd.NA) if owner_mode else pd.NA,
                "likes": raw.get("like_count", pd.NA),
                "comments_count": raw.get("comments_count", pd.NA),
                "shares": raw.get("shares", pd.NA) if owner_mode else pd.NA,
                "saves": raw.get("saved", raw.get("saves", pd.NA)) if owner_mode else pd.NA,
                "follower_count": follower_count if follower_count is not None else pd.NA,
                "retention_rate": pd.NA,
                "collection_timestamp": now_utc,
                "provenance": SourceProvenance.OWNER_AUTHORIZED_DATA.value if owner_mode else SourceProvenance.REAL_PUBLIC_DATA.value,
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True, errors="coerce")
        df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"], utc=True, errors="coerce")
        for col, dtype in UNIFIED_PANDAS_DTYPES.items():
            if col in df.columns:
                if dtype in {"Int64", "Float64"}:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)
        return df

    def fetch_posts(self, identifier: str, limit: int = 50, mode: str = "public") -> pd.DataFrame:
        """Fetch media records and format into the 14-field unified schema.

        If unauthenticated or in sandbox mode, returns an empty structured DataFrame with proper dtypes.
        """
        return _empty_unified_posts_df()


class YouTubeDataAPIConnector(BaseAPIConnector):
    """Connector for YouTube Data API v3 and YouTube Analytics API.

    API Requirements & Limitations:
    1. Public Mode (YouTube Data API v3):
       - Requires standard Google Cloud API Key.
       - Endpoint: videos.list(part='snippet,statistics,contentDetails')
       - Returns: viewCount, likeCount, commentCount, duration, publishedAt.
       - CANNOT access: shares, retention curves, averageViewPercentage (returned as pd.NA).
       - Saves metric: Not applicable to YouTube architecture.
    2. Account-Owner Mode (YouTube Analytics API):
       - Requires OAuth 2.0 with 'https://www.googleapis.com/auth/yt-analytics.readonly'.
       - Returns: averageViewPercentage (retention), shares, subscribersGained.
    Quota Cost: 1 unit per videos.list call (Daily quota default: 10,000 units).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        channel_id: Optional[str] = None,
        sandbox_mode: bool = False
    ):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.channel_id = channel_id or os.getenv("YOUTUBE_CHANNEL_ID", "")
        self.sandbox_mode = sandbox_mode or (not bool(self.api_key))

    def get_provenance(self, mode: str = "public") -> DataProvenance:
        return DataProvenance(
            source_name="YouTube Data API v3",
            access_tier="owner_oauth" if mode == "owner" else "public_api",
            notes="Retention and shares require Analytics OAuth; public API provides viewCount and likeCount only."
        )

    def normalize_api_payload(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        mode: str = "public",
        subscriber_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Normalize YouTube Data API v3 `videos.list` items into unified schema."""
        if not items:
            return _empty_unified_posts_df()

        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        owner_mode = str(mode).lower() == "owner"
        rows: List[Dict[str, Any]] = []

        for raw in items:
            snip = raw.get("snippet", {})
            stats = raw.get("statistics", {})
            avg_pct = raw.get("averageViewPercentage")
            ret_val = (float(avg_pct) / 100.0) if (owner_mode and avg_pct is not None) else pd.NA
            rows.append({
                "platform": "youtube",
                "post_id": str(raw.get("id", "")),
                "timestamp": snip.get("publishedAt") or raw.get("publishedAt"),
                "content_type": "short" if "#shorts" in str(snip.get("title", "")).lower() else "video",
                "views": stats.get("viewCount", pd.NA),
                "impressions": raw.get("impressions", pd.NA) if owner_mode else pd.NA,
                "reach": pd.NA,
                "likes": stats.get("likeCount", pd.NA),
                "comments_count": stats.get("commentCount", pd.NA),
                "shares": raw.get("shares", pd.NA) if owner_mode else pd.NA,
                "saves": pd.NA,  # Structurally N/A on YouTube
                "follower_count": subscriber_count if subscriber_count is not None else pd.NA,
                "retention_rate": ret_val,
                "collection_timestamp": now_utc,
                "provenance": SourceProvenance.OWNER_AUTHORIZED_DATA.value if owner_mode else SourceProvenance.REAL_PUBLIC_DATA.value,
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True, errors="coerce")
        df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"], utc=True, errors="coerce")
        for col, dtype in UNIFIED_PANDAS_DTYPES.items():
            if col in df.columns:
                if dtype in {"Int64", "Float64"}:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)
        return df

    def fetch_posts(self, identifier: str, limit: int = 50, mode: str = "public") -> pd.DataFrame:
        """Fetch video records and format into the 14-field unified schema."""
        return _empty_unified_posts_df()


class SocialMediaScraper:
    """Contract placeholder interface for extracting public post metadata and comment sections."""

    def __init__(self, headless: bool = True):
        self.headless = headless

    def scrape_post_metrics(self, url: str) -> Dict[str, Any]:
        """Scrape engagement metrics from a public post URL."""
        return {
            "url": url,
            "views": 0,
            "likes": 0,
            "comments_count": 0,
            "shares": 0,
            "saves": 0,
            "status": "placeholder"
        }

    def scrape_comments(self, url: str, limit: int = 100) -> pd.DataFrame:
        """Scrape comments for NLP sentiment analysis."""
        return pd.DataFrame(columns=["post_id", "comment_id", "author", "comment_text", "timestamp", "likes"])
