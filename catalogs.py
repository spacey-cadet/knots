"""
search_content: matches extracted topics against real content catalogs.

Three live sources, each with a graceful offline fallback to small,
hand-written mock results so the whole server runs and demos cleanly with
no API keys configured and no network access. Set the env vars in .env to
switch a source from mock to live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Literal

import httpx

ContentType = Literal["movie", "doc", "video", "paper"]

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
ARXIV_API_URL = "http://export.arxiv.org/api/query"


@dataclass
class ContentItem:
    title: str
    content_type: ContentType
    source_url: str
    topic: str  # which extracted topic produced this candidate
    topic_overlap_score: float
    blurb: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock catalogs — used when no API key is set, or a live call fails/times out.
# Keep these small and hand-curated so the demo always has something sane to
# show, even fully offline.
# ---------------------------------------------------------------------------

MOCK_MOVIES = {
    "default": [
        {"title": "Pantheon", "url": "https://www.themoviedb.org/tv/206559-pantheon",
         "blurb": "Uploaded consciousness and simulation-theory themes."},
        {"title": "Dirty Money", "url": "https://www.themoviedb.org/tv/85400-dirty-money",
         "blurb": "Documentary series on financial schemes and speculative markets."},
    ],
}

MOCK_VIDEOS = {
    "default": [
        {"title": "How memecoins actually work (explainer)",
         "url": "https://www.youtube.com/results?search_query=how+memecoins+work",
         "blurb": "Accessible breakdown of tokenomics and speculative dynamics."},
    ],
}

MOCK_PAPERS = {
    "default": [
        {"title": "Speculative Bubbles and Social Media-Driven Asset Pricing",
         "url": "https://arxiv.org/abs/2201.00000",
         "blurb": "Academic model of social-media-amplified speculative assets."},
    ],
}


def _mock_lookup(table: dict, topic: str) -> list[dict]:
    return table.get(topic, table["default"])


# ---------------------------------------------------------------------------
# Live clients
# ---------------------------------------------------------------------------

async def _search_tmdb(topic: str, client: httpx.AsyncClient) -> list[ContentItem]:
    if not TMDB_API_KEY:
        results = _mock_lookup(MOCK_MOVIES, topic)
        return [
            ContentItem(title=r["title"], content_type="movie", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.6, blurb=r["blurb"])
            for r in results
        ]
    try:
        resp = await client.get(
            TMDB_SEARCH_URL,
            params={"api_key": TMDB_API_KEY, "query": topic},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        items = []
        for r in data.get("results", [])[:3]:
            items.append(ContentItem(
                title=r.get("title", "Untitled"),
                content_type="movie",
                source_url=f"https://www.themoviedb.org/movie/{r.get('id')}",
                topic=topic,
                topic_overlap_score=0.8,
                blurb=(r.get("overview") or "")[:140],
            ))
        return items or [ContentItem(**{**r, "content_type": "movie", "topic": topic, "topic_overlap_score": 0.5})
                          for r in []]
    except Exception:
        results = _mock_lookup(MOCK_MOVIES, topic)
        return [
            ContentItem(title=r["title"], content_type="movie", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, blurb=r["blurb"] + " (offline fallback)")
            for r in results
        ]


async def _search_youtube(topic: str, client: httpx.AsyncClient) -> list[ContentItem]:
    if not YOUTUBE_API_KEY:
        results = _mock_lookup(MOCK_VIDEOS, topic)
        return [
            ContentItem(title=r["title"], content_type="video", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.6, blurb=r["blurb"])
            for r in results
        ]
    try:
        resp = await client.get(
            YOUTUBE_SEARCH_URL,
            params={
                "part": "snippet", "q": topic, "type": "video",
                "maxResults": 3, "key": YOUTUBE_API_KEY,
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        items = []
        for r in data.get("items", []):
            vid = r["id"]["videoId"]
            items.append(ContentItem(
                title=r["snippet"]["title"],
                content_type="video",
                source_url=f"https://www.youtube.com/watch?v={vid}",
                topic=topic,
                topic_overlap_score=0.8,
                blurb=r["snippet"].get("description", "")[:140],
            ))
        return items
    except Exception:
        results = _mock_lookup(MOCK_VIDEOS, topic)
        return [
            ContentItem(title=r["title"], content_type="video", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, blurb=r["blurb"] + " (offline fallback)")
            for r in results
        ]


async def _search_arxiv(topic: str, client: httpx.AsyncClient) -> list[ContentItem]:
    # arXiv needs no API key, but still fall back to mocks offline / on error.
    try:
        resp = await client.get(
            ARXIV_API_URL,
            params={"search_query": f"all:{topic}", "start": 0, "max_results": 2},
            timeout=5,
        )
        resp.raise_for_status()
        text = resp.text
        # Minimal, dependency-free Atom-feed title scrape (avoids adding an
        # XML parsing dependency for a v1 server).
        import re as _re
        titles = _re.findall(r"<title>(.*?)</title>", text, _re.DOTALL)[1:]  # skip feed title
        ids = _re.findall(r"<id>(http://arxiv\.org/abs/.*?)</id>", text)
        items = []
        for title, url in zip(titles, ids):
            items.append(ContentItem(
                title=title.strip().replace("\n", " "),
                content_type="paper",
                source_url=url,
                topic=topic,
                topic_overlap_score=0.8,
            ))
        if items:
            return items
        raise ValueError("no arxiv results")
    except Exception:
        results = _mock_lookup(MOCK_PAPERS, topic)
        return [
            ContentItem(title=r["title"], content_type="paper", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, blurb=r["blurb"] + " (offline fallback)")
            for r in results
        ]


async def search_content(topics: list[str]) -> list[ContentItem]:
    """Fan out each topic to all three catalogs, in parallel."""
    import asyncio
    async with httpx.AsyncClient() as client:
        tasks = []
        for topic in topics:
            tasks.append(_search_tmdb(topic, client))
            tasks.append(_search_youtube(topic, client))
            tasks.append(_search_arxiv(topic, client))
        results = await asyncio.gather(*tasks)
    flat: list[ContentItem] = []
    for group in results:
        flat.extend(group)
    return flat
