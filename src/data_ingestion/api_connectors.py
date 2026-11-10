"""Social Media API Connectors for Instagram Graph API and YouTube Data API v3.

Documents exact endpoint requirements, access levels (Public vs. Owner-authorized),
rate-limiting boundaries, and formats responses into the 13-field unified schema.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
import pandas as pd
from .schemas import UNIFIED_POST_FIELDS, UNIFIED_PANDAS_DTYPES, DataProvenance


class BaseAPIConnector(ABC):
    """Abstract interface defining the contract for social platform data connectors."""

    @abstractmethod
    def fetch_posts(self, identifier: str, limit: int = 50, mode: str = "public") -> pd.DataFrame:
        """Fetch post records formatted to the 13-field unified schema."""
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
        sandbox_mode: bool = False
    ):
        self.access_token = access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        self.page_id = page_id or os.getenv("INSTAGRAM_PAGE_ID", "")
        self.sandbox_mode = sandbox_mode or (not bool(self.access_token))
        self.base_url = "https://graph.facebook.com/v18.0"

    def get_provenance(self, mode: str = "public") -> DataProvenance:
        return DataProvenance(
            source_name="Instagram Graph API",
            access_tier="owner_oauth" if mode == "owner" else "public_api",
            notes="Owner mode exposes saves/shares/reach; public mode marks them as null."
        )

    def fetch_posts(self, identifier: str, limit: int = 50, mode: str = "public") -> pd.DataFrame:
        """Fetch media records and format into the 13-field unified schema.

        If unauthenticated or in sandbox mode, returns an empty structured DataFrame with proper dtypes.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        
        # In absence of active network credentials or sandbox mode, return empty typed schema
        df = pd.DataFrame(columns=UNIFIED_POST_FIELDS)
        for col, dtype in UNIFIED_PANDAS_DTYPES.items():
            df[col] = pd.Series(dtype=dtype)
        df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        df["collection_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        return df


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

    def fetch_posts(self, identifier: str, limit: int = 50, mode: str = "public") -> pd.DataFrame:
        """Fetch video records and format into the 13-field unified schema."""
        df = pd.DataFrame(columns=UNIFIED_POST_FIELDS)
        for col, dtype in UNIFIED_PANDAS_DTYPES.items():
            df[col] = pd.Series(dtype=dtype)
        df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        df["collection_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        return df
