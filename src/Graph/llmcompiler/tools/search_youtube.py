
from __future__ import annotations
import os
import re
import requests
from typing import List, Dict, Any, Optional
from langchain_core.tools import StructuredTool

from src.helpers.config import get_settings

_YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def _iso8601_duration_to_seconds(s: str) -> int:
    """
    Convert ISO 8601 duration (e.g., 'PT1H2M10S') to total seconds.
    Handles hours/minutes/seconds; ignores days/weeks/months/years for YouTube.
    """
    m = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        s.strip()
    )
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + sec


def _seconds_to_hms(total: int) -> str:
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class YouTubeRecipeSearchTool:
    """
    Search recipes on YouTube and return enriched video results.
    Exportable as a LangChain StructuredTool.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        region_code: str = "US",
        relevance_language: Optional[str] = "en",
        safe_search: str = "moderate",  # "none" | "moderate" | "strict"
    ):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("YouTube API key missing. Set YOUTUBE_API_KEY or pass api_key.")
        self.max_results = max(1, min(max_results, 50))
        self.region_code = region_code
        self.relevance_language = relevance_language
        self.safe_search = safe_search

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Call search.list and return raw items."""
        params = {
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": self.max_results,
            "regionCode": self.region_code,
            "safeSearch": self.safe_search,
            "key": self.api_key,
        }
        if self.relevance_language:
            params["relevanceLanguage"] = self.relevance_language
        r = requests.get(_YT_SEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("items", [])

    def _videos_details(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Call videos.list and map id -> {duration, views}."""
        if not video_ids:
            return {}
        params = {
            "part": "contentDetails,statistics",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        r = requests.get(_YT_VIDEOS_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out: Dict[str, Dict[str, Any]] = {}
        for it in data.get("items", []):
            vid = it.get("id")
            dur = it.get("contentDetails", {}).get("duration", "PT0S")
            views = it.get("statistics", {}).get("viewCount")
            sec = _iso8601_duration_to_seconds(dur)
            out[vid] = {
                "duration_sec": sec,
                "duration_hms": _seconds_to_hms(sec),
                "views": int(views) if views is not None else None,
            }
        return out

    def run(self, query: str) -> Dict[str, Any]:
        """Public entry. Returns {'query': str, 'results': [ ... ]}."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        raw = self._search(query.strip())
        vids = [it.get("id", {}).get("videoId") for it in raw if it.get("id")]
        vids = [v for v in vids if v]
        details = self._videos_details(vids)

        results = []
        for it in raw:
            id_ = it.get("id", {}).get("videoId")
            sn = it.get("snippet", {}) or {}
            if not id_:
                continue
            url = f"https://www.youtube.com/watch?v={id_}"
            thumbs = sn.get("thumbnails", {}) or {}
            best_thumb = (
                thumbs.get("maxres")
                or thumbs.get("standard")
                or thumbs.get("high")
                or thumbs.get("medium")
                or thumbs.get("default")
                or {}
            )
            d = details.get(id_, {})
            results.append(
                {
                    "title": sn.get("title"),
                    "channel": sn.get("channelTitle"),
                    "published_at": sn.get("publishedAt"),
                    "video_id": id_,
                    "url": url,
                    "duration_sec": d.get("duration_sec"),
                    "duration_hms": d.get("duration_hms"),
                    "views": d.get("views"),
                    "thumbnail_url": best_thumb.get("url"),
                }
            )
        return {"query": query, "results": results}

    def youtube_as_tool(self) -> StructuredTool:
        """Export as a LangChain StructuredTool."""
        return StructuredTool.from_function(
            func=self.run,
            name="youtube_recipe_search",
            description='youtube_recipe_search(query="recipe or dish") → JSON with YouTube videos.',
        )




if __name__ == "__main__":
    app_settings = get_settings()
    os.environ['YOUTUBE_API_KEY'] = app_settings.YOUTUBE_API_KEY

    tool = YouTubeRecipeSearchTool(max_results=5)
    resp = tool.run("chicken biryani recipe")
    for v in resp["results"]:
        print(v["title"], v["duration_hms"], v["url"])