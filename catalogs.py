"""
search_content: matches extracted interests against content catalogs.

Roles (see README, "Design"):
  - role="recommendation": films ("movie") and documentaries ("doc").
    These are what the person is actually recommended.
  - role="anchor": papers (arXiv). They ground a concept ("this is the
    research this film is about") and are attached to cards as evidence,
    never ranked against films. Optional videos (INCLUDE_VIDEOS=1) are also
    recommendations.

Why TMDB is queried by *keyword*, not by title search:
TMDB's /search/movie matches film titles, so "free will" finds films
literally named "Free Will", not films about free will. TMDB's keyword
index (/search/keyword -> /discover/movie?with_keywords=) matches the
thematic tags editors attach to films, which is what we want. That is why
the Bedrock extractor is asked for film-tag-style keywords.

Literal vs lateral: for the same keywords we run discover twice, once
restricted to the Documentary genre (literal pick: a documentary about it)
and once excluding it (lateral pick: fiction that dramatizes it).

Everything has an offline mock fallback so the server runs with zero keys.
"""

from __future__ import annotations

import asyncio
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Literal

import httpx

from interests import Interest

ContentType = Literal["movie", "doc", "video", "paper"]
Role = Literal["recommendation", "anchor"]

TMDB_BASE = "https://api.themoviedb.org/3"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
DOCUMENTARY_GENRE = 99

MAX_SEARCH_INTERESTS = 3     # latency cap: only the top interests hit live catalogs
MAX_KEYWORDS_PER_INTEREST = 4
MIN_VOTE_COUNT = 100         # drop obscure / junk entries from discover results
HTTP_TIMEOUT = 5


def _tmdb_key() -> str:  # read at call time so tests / .env changes take effect
    return os.environ.get("TMDB_API_KEY", "")


def _youtube_key() -> str:
    return os.environ.get("YOUTUBE_API_KEY", "")


@dataclass
class ContentItem:
    title: str
    content_type: ContentType
    source_url: str
    topic: str                      # which interest produced this candidate
    topic_overlap_score: float
    blurb: str = ""
    role: Role = "recommendation"
    year: str = ""
    matched_keywords: list[str] = field(default_factory=list)  # evidence: which terms hit
    popularity: int = 0             # vote_count, used only as a tie-break

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock catalogs — used when no key is set, or a live call fails/times out.
# ---------------------------------------------------------------------------

MOCK_MOVIES = [
    {"title": "Pantheon", "url": "https://www.themoviedb.org/tv/206559-pantheon",
     "blurb": "Uploaded consciousness and simulation-theory themes.", "type": "movie"},
    {"title": "Dirty Money", "url": "https://www.themoviedb.org/tv/85400-dirty-money",
     "blurb": "Documentary series on financial schemes and speculative markets.", "type": "doc"},
]

MOCK_VIDEOS = [
    {"title": "How memecoins actually work (explainer)",
     "url": "https://www.youtube.com/results?search_query=how+memecoins+work",
     "blurb": "Accessible breakdown of tokenomics and speculative dynamics."},
]

MOCK_PAPERS = [
    {"title": "Speculative Bubbles and Social Media-Driven Asset Pricing",
     "url": "https://arxiv.org/abs/2201.00000",
     "blurb": "Academic model of social-media-amplified speculative assets."},
]


def _mock_films(topic: str, note: str = "") -> list[ContentItem]:
    return [ContentItem(title=r["title"], content_type=r["type"], source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, blurb=r["blurb"] + note)
            for r in MOCK_MOVIES]


# ---------------------------------------------------------------------------
# TMDB: keyword -> discover, literal (doc) and lateral (film)
# ---------------------------------------------------------------------------

async def _tmdb_get(client: httpx.AsyncClient, path: str, params: dict) -> dict:
    resp = await client.get(f"{TMDB_BASE}{path}", params={"api_key": _tmdb_key(), **params},
                            timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


async def _resolve_keyword(client: httpx.AsyncClient, term: str) -> tuple[str, int] | None:
    """Map a free-text term to a TMDB keyword id (best exact-ish match)."""
    data = await _tmdb_get(client, "/search/keyword", {"query": term})
    results = data.get("results", [])
    if not results:
        return None
    exact = next((r for r in results if r.get("name", "").lower() == term), None)
    return term, (exact or results[0])["id"]


async def _discover(client: httpx.AsyncClient, kw_id: int, doc: bool) -> list[dict]:
    params = {
        "with_keywords": str(kw_id),
        "sort_by": "vote_count.desc",
        "vote_count.gte": MIN_VOTE_COUNT,
        "include_adult": "false",
    }
    params["with_genres" if doc else "without_genres"] = DOCUMENTARY_GENRE
    return (await _tmdb_get(client, "/discover/movie", params)).get("results", [])[:10]


async def _search_tmdb(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    if not _tmdb_key():
        return _mock_films(interest.topic)
    try:
        terms = interest.search_terms(limit=MAX_KEYWORDS_PER_INTEREST * 2)
        resolved = [r for r in await asyncio.gather(*[_resolve_keyword(client, t) for t in terms]) if r]
        # de-dup by keyword id, keep the extractor's priority order, cap
        by_id: dict[int, str] = {}
        for term, kid in resolved:
            by_id.setdefault(kid, term)
        kw = list(by_id.items())[:MAX_KEYWORDS_PER_INTEREST]
        if not kw:
            return []  # no thematic hook in TMDB: better to return nothing than noise

        jobs = [(kid, term, doc) for kid, term in kw for doc in (True, False)]
        batches = await asyncio.gather(*[_discover(client, kid, doc) for kid, _, doc in jobs])

        found: dict[int, ContentItem] = {}
        for (kid, term, doc), results in zip(jobs, batches):
            for r in results:
                item = found.get(r["id"])
                if item is None:
                    item = found[r["id"]] = ContentItem(
                        title=r.get("title", "Untitled"),
                        content_type="doc" if doc else "movie",
                        source_url=f"https://www.themoviedb.org/movie/{r['id']}",
                        topic=interest.topic,
                        topic_overlap_score=0.0,
                        blurb=(r.get("overview") or "")[:180],
                        year=(r.get("release_date") or "")[:4],
                        popularity=int(r.get("vote_count") or 0),
                    )
                if term not in item.matched_keywords:
                    item.matched_keywords.append(term)
        for item in found.values():
            # share of the interest's resolved keywords this film carries,
            # scaled by how sure the extractor was about the reading
            share = len(item.matched_keywords) / len(kw)
            item.topic_overlap_score = round((0.4 + 0.6 * share) * (0.7 + 0.3 * interest.confidence), 3)
        return list(found.values())
    except Exception:
        return _mock_films(interest.topic, " (offline fallback)")


# ---------------------------------------------------------------------------
# arXiv: papers are anchors (concept grounding), not recommendations
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"


async def _search_arxiv(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    query = (interest.concepts[0] if interest.concepts else interest.topic)
    try:
        resp = await client.get(
            ARXIV_API_URL,
            params={"search_query": f'all:"{query}"', "start": 0, "max_results": 1},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = []
        for entry in root.findall(f"{_ATOM}entry"):
            title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            url = (entry.findtext(f"{_ATOM}id") or "").strip()
            if title and url:
                items.append(ContentItem(
                    title=title, content_type="paper", source_url=url, topic=interest.topic,
                    topic_overlap_score=0.8, role="anchor",
                    blurb=" ".join((entry.findtext(f"{_ATOM}summary") or "").split())[:180],
                ))
        if items:
            return items
        raise ValueError("no arxiv results")
    except Exception:
        return [ContentItem(title=r["title"], content_type="paper", source_url=r["url"],
                            topic=interest.topic, topic_overlap_score=0.5, role="anchor",
                            blurb=r["blurb"] + " (offline fallback)")
                for r in MOCK_PAPERS]


# ---------------------------------------------------------------------------
# YouTube (optional, off by default: the product is films + docs)
# ---------------------------------------------------------------------------

async def _search_youtube(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    if not _youtube_key():
        return [ContentItem(title=r["title"], content_type="video", source_url=r["url"],
                            topic=interest.topic, topic_overlap_score=0.5, blurb=r["blurb"])
                for r in MOCK_VIDEOS]
    try:
        resp = await client.get(
            YOUTUBE_SEARCH_URL,
            # a concept plus "explained" finds explainers; raw film-tag keywords find trailers
            params={"part": "snippet",
                    "q": f"{(interest.concepts or [interest.topic])[0]} explained",
                    "type": "video", "maxResults": 3, "key": _youtube_key()},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return [ContentItem(
            title=r["snippet"]["title"], content_type="video",
            source_url=f"https://www.youtube.com/watch?v={r['id']['videoId']}",
            topic=interest.topic, topic_overlap_score=0.7,
            blurb=r["snippet"].get("description", "")[:140]) for r in resp.json().get("items", [])]
    except Exception:
        return []


async def search_content(interests: list[Interest],
                         client: httpx.AsyncClient | None = None,
                         include_videos: bool | None = None) -> list[ContentItem]:
    """Fan out interests to the catalogs in parallel. `client` is injectable for tests.
    `include_videos=None` defers to the INCLUDE_VIDEOS env flag."""
    if include_videos is None:
        include_videos = os.environ.get("INCLUDE_VIDEOS") == "1"
    owns = client is None
    client = client or httpx.AsyncClient()
    try:
        tasks = []
        for interest in interests[:MAX_SEARCH_INTERESTS]:
            tasks.append(_search_tmdb(interest, client))
            tasks.append(_search_arxiv(interest, client))
            if include_videos:
                tasks.append(_search_youtube(interest, client))
        groups = await asyncio.gather(*tasks)
    finally:
        if owns:
            await client.aclose()
    return [item for g in groups for item in g]
