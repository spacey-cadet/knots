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

Live failures return nothing, never made-up items. The fixed mock catalogs are
used only in explicit offline mode (KNOTS_OFFLINE=1), for demos and tests with
no network. A missing key or a failed call otherwise yields an empty list and a
logged warning, so a fake title can never be shown as a real recommendation.

Ranking: relevance (how many of the interest's keywords a film carries) decides.
Quality is a floor, not a score: see `passes_quality`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Literal

import httpx

from interests import Interest

log = logging.getLogger("knots.catalogs")

ContentType = Literal["movie", "doc", "video", "paper"]
Role = Literal["recommendation", "anchor"]

TMDB_BASE = "https://api.themoviedb.org/3"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
ARXIV_API_URL = "https://export.arxiv.org/api/query"   # http:// 301-redirects to https://
DOCUMENTARY_GENRE = 99

MAX_SEARCH_INTERESTS = 3     # latency cap: only the top interests hit live catalogs
MAX_KEYWORDS_PER_INTEREST = 4
HTTP_TIMEOUT = 5

# Quality floor (a gate, not a ranking signal). Fame is not quality, and
# relevance, not rating, decides order among films that pass.
MIN_VOTE_COUNT = 100         # also sent to TMDB so discover drops junk early
MIN_RATING = 6.5
OLD_FILM_BEFORE = 1990       # older films must clear a higher bar
OLD_MIN_RATING = 7.5
OLD_MIN_VOTES = 1000
DISCOVER_PER_CALL = 20       # TMDB returns 20 per page; gate first, then rank
GEM_MAX_VOTES = 3000         # "lesser-known": few votes ...
GEM_MIN_RATING = 7.0         # ... but highly rated


def passes_quality(rating: float, votes: int, year: str) -> bool:
    """Floor for a TMDB film: rating plus a minimum vote count, higher for old films."""
    try:
        old = bool(year) and int(year) < OLD_FILM_BEFORE
    except ValueError:
        old = False
    if old:
        return rating >= OLD_MIN_RATING and votes >= OLD_MIN_VOTES
    return rating >= MIN_RATING and votes >= MIN_VOTE_COUNT


def is_gem(item: "ContentItem") -> bool:
    """Lesser-known but highly rated film/doc (the reserved slot in the picks)."""
    return (item.content_type in ("movie", "doc") and item.popularity <= GEM_MAX_VOTES
            and item.rating >= GEM_MIN_RATING)


def offline_mode() -> bool:
    """Mock catalogs are allowed only when explicitly requested."""
    return os.environ.get("KNOTS_OFFLINE") == "1"


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
    popularity: int = 0             # TMDB vote_count; evidence of obscurity, not a ranking signal
    rating: float = 0.0             # TMDB vote_average (0-10)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock catalogs — used ONLY in explicit offline mode (KNOTS_OFFLINE=1).
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


def _mock_films(topic: str) -> list[ContentItem]:
    return [ContentItem(title=r["title"], content_type=r["type"], source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, blurb=r["blurb"])
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
        "sort_by": "vote_average.desc",   # best-rated first, so famous films don't crowd out the rest
        "vote_count.gte": MIN_VOTE_COUNT,
        "vote_average.gte": MIN_RATING,
        "include_adult": "false",
    }
    params["with_genres" if doc else "without_genres"] = DOCUMENTARY_GENRE
    return (await _tmdb_get(client, "/discover/movie", params)).get("results", [])[:DISCOVER_PER_CALL]


async def _search_tmdb(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    if not _tmdb_key():
        if offline_mode():
            return _mock_films(interest.topic)
        log.warning("TMDB_API_KEY is not set: no films returned (set KNOTS_OFFLINE=1 for mock data)")
        return []
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
                year = (r.get("release_date") or "")[:4]
                rating, votes = float(r.get("vote_average") or 0), int(r.get("vote_count") or 0)
                if not passes_quality(rating, votes, year):
                    continue
                item = found.get(r["id"])
                if item is None:
                    item = found[r["id"]] = ContentItem(
                        title=r.get("title", "Untitled"),
                        content_type="doc" if doc else "movie",
                        source_url=f"https://www.themoviedb.org/movie/{r['id']}",
                        topic=interest.topic,
                        topic_overlap_score=0.0,
                        blurb=(r.get("overview") or "")[:180],
                        year=year, popularity=votes, rating=rating,
                    )
                if term not in item.matched_keywords:
                    item.matched_keywords.append(term)
        for item in found.values():
            # share of the interest's resolved keywords this film carries,
            # scaled by how sure the extractor was about the reading
            share = len(item.matched_keywords) / len(kw)
            item.topic_overlap_score = round((0.4 + 0.6 * share) * (0.7 + 0.3 * interest.confidence), 3)
        return list(found.values())
    except Exception as e:  # noqa: BLE001
        log.warning("TMDB search failed for %r: %s: %s", interest.topic, type(e).__name__, e)
        return _mock_films(interest.topic) if offline_mode() else []


def _mock_papers(topic: str) -> list[ContentItem]:
    return [ContentItem(title=r["title"], content_type="paper", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, role="anchor", blurb=r["blurb"])
            for r in MOCK_PAPERS]


# ---------------------------------------------------------------------------
# arXiv: papers are anchors (concept grounding), not recommendations
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"


async def _search_arxiv(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    query = (interest.concepts[0] if interest.concepts else interest.topic)
    if offline_mode():
        return _mock_papers(interest.topic)
    try:
        resp = await client.get(
            ARXIV_API_URL,
            params={"search_query": f'all:"{query}"', "start": 0, "max_results": 1},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
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
        return items          # no match is a legitimate empty answer
    except Exception as e:  # noqa: BLE001
        log.warning("arXiv search failed for %r: %s: %s", query, type(e).__name__, e)
        return []


# ---------------------------------------------------------------------------
# YouTube (optional, off by default: the product is films + docs)
# ---------------------------------------------------------------------------

async def _search_youtube(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    if not _youtube_key():
        if offline_mode():
            return [ContentItem(title=r["title"], content_type="video", source_url=r["url"],
                                topic=interest.topic, topic_overlap_score=0.5, blurb=r["blurb"])
                    for r in MOCK_VIDEOS]
        log.warning("YOUTUBE_API_KEY is not set: no videos returned")
        return []
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
    except Exception as e:  # noqa: BLE001
        log.warning("YouTube search failed for %r: %s: %s", interest.topic, type(e).__name__, e)
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
