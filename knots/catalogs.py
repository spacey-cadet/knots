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
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Literal

import httpx

from . import signals
from .extraction import STOPWORDS, Interest, allowed_mediums
from .quality import (MIN_RATING, MIN_VOTE_COUNT, TV_MIN_RATING, TV_MIN_VOTE_COUNT,
                      passes_quality)

log = logging.getLogger("knots.catalogs")

ContentType = Literal["movie", "series", "doc", "video", "paper"]
Role = Literal["recommendation", "anchor"]

TMDB_BASE = "https://api.themoviedb.org/3"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
ARXIV_API_URL = "https://export.arxiv.org/api/query"   # http:// 301-redirects to https://
DOCUMENTARY_GENRE = 99

MAX_SEARCH_INTERESTS = 3     # latency cap: only the top interests hit live catalogs
MAX_KEYWORDS_PER_INTEREST = 4
DISCOVER_PER_CALL = 20       # TMDB returns 20 per page; gate first, then rank
HTTP_TIMEOUT = 5              # seconds; override with KNOTS_HTTP_TIMEOUT (the eval uses a longer one)


def _timeout() -> httpx.Timeout:
    try:
        t = float(os.environ.get("KNOTS_HTTP_TIMEOUT", HTTP_TIMEOUT))
    except ValueError:
        t = float(HTTP_TIMEOUT)
    return httpx.Timeout(t, pool=None)      # queueing behind the connection cap is not a failure


# Every failed live catalog call is recorded here, so callers (the eval) can tell a
# session with missing results from one that genuinely had none.
FAILURES: list[str] = []


def _fail(source: str, what: str, err: object) -> None:
    code = getattr(getattr(err, "response", None), "status_code", None)
    reason = code or type(err).__name__
    FAILURES.append(f"{source}:{what}:{reason}")
    log.warning("%s search failed for %r: %s", source, what, reason)

# YouTube explainers. A search costs 100 quota units (10,000/day by default), a details
# lookup 1, so only the top interests are searched and the details are batched.
YOUTUBE_MAX_INTERESTS = 2
YOUTUBE_CANDIDATES = 10
YOUTUBE_PER_INTEREST = 2
VIDEO_MIN_SECONDS = 180      # no shorts or trailers
VIDEO_MAX_SECONDS = 3600     # an explainer, not a lecture series
VIDEO_MIN_VIEWS = 5000       # floor against junk uploads; views are not a ranking signal

# arXiv asks for at most one request every ~3 s and answers bursts with 429.
ARXIV_MIN_INTERVAL = 5.0
ARXIV_MAX_INTERESTS = 2       # papers are anchors; searching every interest only adds wait
ARXIV_RETRY_AFTER_DEFAULT = 20.0

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
    # --- richer candidate facts: embeddings and ranking need more than a 180-char blurb ---
    abstract: str = ""              # film/series overview, paper abstract, or video description (longer than blurb)
    citations: int | None = None    # papers: OpenAlex cited_by_count; None = unknown (never guessed)
    is_survey: bool = False         # papers: a review/overview, the accessible way into a field
    status: str = ""                # papers: "preprint" or "published" (only when a journal reference is listed)
    venue: str = ""                 # papers: the journal reference arXiv lists, if any
    duration_s: int = 0             # videos
    challenge: str = ""             # set when found by the opt-in "challenge me" search: the counterpoint it raises

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock catalogs — used ONLY in explicit offline mode (KNOTS_OFFLINE=1).
# ---------------------------------------------------------------------------

MOCK_MOVIES = [
    {"title": "Pantheon", "url": "https://www.themoviedb.org/tv/206559-pantheon",
     "blurb": "Uploaded consciousness and simulation-theory themes.", "type": "series"},
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


def mock_urls() -> set[str]:
    """Every fixed mock URL, so the eval can tell offline mock rows from live ones."""
    return {r["url"] for r in (*MOCK_MOVIES, *MOCK_VIDEOS, *MOCK_PAPERS)}


def _mock_films(topic: str) -> list[ContentItem]:
    return [ContentItem(title=r["title"], content_type=r["type"], source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, blurb=r["blurb"])
            for r in MOCK_MOVIES]


# ---------------------------------------------------------------------------
# TMDB: keyword -> discover, literal (doc) and lateral (film)
# ---------------------------------------------------------------------------

async def _tmdb_get(client: httpx.AsyncClient, path: str, params: dict) -> dict:
    for attempt in (1, 2, 3):
        try:
            resp = await client.get(f"{TMDB_BASE}{path}", params={"api_key": _tmdb_key(), **params},
                                    timeout=_timeout())
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                await asyncio.sleep(attempt)       # brief backoff, then retry
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt == 3:
                raise
            await asyncio.sleep(attempt)
    raise RuntimeError("unreachable")


async def _resolve_keyword(client: httpx.AsyncClient, term: str) -> tuple[str, int] | None:
    """Map a free-text term to a TMDB keyword id (best exact-ish match)."""
    data = await _tmdb_get(client, "/search/keyword", {"query": term})
    results = data.get("results", [])
    if not results:
        return None
    exact = next((r for r in results if r.get("name", "").lower() == term), None)
    return term, (exact or results[0])["id"]


async def _discover(client: httpx.AsyncClient, kw_id: int, doc: bool, sort: str = "vote_average.desc") -> list[dict]:
    """Best-rated first surfaces the strongest titles, vote count first the widely seen ones;
    both lists are quality-gated and relevance decides the order afterwards."""
    params = {
        "with_keywords": str(kw_id),
        "sort_by": sort,
        "vote_count.gte": MIN_VOTE_COUNT,
        "vote_average.gte": MIN_RATING,
        "include_adult": "false",
    }
    params["with_genres" if doc else "without_genres"] = DOCUMENTARY_GENRE
    return (await _tmdb_get(client, "/discover/movie", params)).get("results", [])[:DISCOVER_PER_CALL]


async def _discover_tv(client: httpx.AsyncClient, kw_id: int, sort: str = "vote_average.desc") -> list[dict]:
    """Series tagged with the keyword. Documentary series are left to the film/doc path.
    Sorting by rating alone surfaces niche, highly rated shows (and anime); sorting by vote
    count surfaces the established ones. Both are gated, and relevance decides the order."""
    params = {
        "with_keywords": str(kw_id),
        "sort_by": sort,
        "vote_count.gte": TV_MIN_VOTE_COUNT,
        "vote_average.gte": TV_MIN_RATING,
        "without_genres": str(DOCUMENTARY_GENRE),
    }
    return (await _tmdb_get(client, "/discover/tv", params)).get("results", [])[:DISCOVER_PER_CALL]


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

        # kind: "doc" (documentary film), "movie" (fiction film), "series" (scripted TV)
        # "-known" jobs sort by vote count, so widely seen titles are candidates too
        jobs = [(kid, term, kind) for kid, term in kw
                for kind in ("doc", "movie", "movie-known", "series", "series-known")]
        sort_for = lambda kind: "vote_count.desc" if kind.endswith("-known") else "vote_average.desc"  # noqa: E731
        batches = await asyncio.gather(*[
            _discover_tv(client, kid, sort_for(kind)) if kind.startswith("series")
            else _discover(client, kid, kind == "doc", sort_for(kind))
            for kid, _, kind in jobs])

        found: dict[tuple[str, int], ContentItem] = {}
        for (kid, term, kind), results in zip(jobs, batches):
            kind = kind.split("-")[0]
            tv = kind == "series"
            for r in results:
                date = (r.get("first_air_date") if tv else r.get("release_date")) or ""
                year = date[:4]
                rating, votes = float(r.get("vote_average") or 0), int(r.get("vote_count") or 0)
                if not passes_quality(rating, votes, year, series=tv):
                    continue
                key = ("tv" if tv else "movie", r["id"])     # TMDB ids are only unique per medium
                item = found.get(key)
                if item is None:
                    item = found[key] = ContentItem(
                        title=(r.get("name") if tv else r.get("title")) or "Untitled",
                        content_type=kind,
                        source_url=f"https://www.themoviedb.org/{'tv' if tv else 'movie'}/{r['id']}",
                        topic=interest.topic,
                        topic_overlap_score=0.0,
                        blurb=(r.get("overview") or "")[:180],
                        abstract=(r.get("overview") or "")[:900],
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
        _fail("TMDB", interest.topic, e)
        return _mock_films(interest.topic) if offline_mode() else []


def _mock_papers(topic: str) -> list[ContentItem]:
    return [ContentItem(title=r["title"], content_type="paper", source_url=r["url"],
                        topic=topic, topic_overlap_score=0.5, role="anchor", blurb=r["blurb"])
            for r in MOCK_PAPERS]


# ---------------------------------------------------------------------------
# arXiv: papers are anchors (concept grounding), not recommendations
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"


_arxiv_next_slot = 0.0                      # monotonic time the next arXiv request may start
_arxiv_cache: dict[str, list[ContentItem]] = {}   # query -> parsed result (successes only)
_arxiv_disk_loaded = False


def _arxiv_cache_path() -> str:
    """Optional on-disk cache (KNOTS_ARXIV_CACHE=path.json) so repeated eval runs do not
    re-ask arXiv for queries it already answered."""
    return os.environ.get("KNOTS_ARXIV_CACHE", "")


def _arxiv_cache_load() -> None:
    global _arxiv_disk_loaded
    path = _arxiv_cache_path()
    if _arxiv_disk_loaded or not path:
        return
    _arxiv_disk_loaded = True
    try:
        import json
        with open(path, encoding="utf-8") as f:
            for q, rows in json.load(f).items():
                _arxiv_cache.setdefault(q, [ContentItem(**r) for r in rows])
    except (OSError, ValueError, TypeError):
        pass


def _arxiv_cache_save() -> None:
    path = _arxiv_cache_path()
    if not path:
        return
    try:
        import json
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({q: [asdict(i) for i in v] for q, v in _arxiv_cache.items()}, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


async def _arxiv_wait_turn() -> None:
    """Space requests ARXIV_MIN_INTERVAL apart across all concurrent searches.
    No lock: the slot is claimed with no await in between, so it is race-free
    inside one event loop, and it survives the eval's one-loop-per-session use."""
    global _arxiv_next_slot
    now = time.monotonic()
    slot = max(now, _arxiv_next_slot)
    _arxiv_next_slot = slot + ARXIV_MIN_INTERVAL
    if slot > now:
        await asyncio.sleep(slot - now)


_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _parse_arxiv(text: str, topic: str) -> list[ContentItem]:
    items = []
    for entry in ET.fromstring(text).findall(f"{_ATOM}entry"):
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        url = (entry.findtext(f"{_ATOM}id") or "").strip()
        if title and url:
            summary = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
            venue = " ".join((entry.findtext(f"{_ARXIV_NS}journal_ref") or "").split())[:200]
            items.append(ContentItem(
                title=title, content_type="paper", source_url=url, topic=topic,
                topic_overlap_score=0.8, role="anchor",
                blurb=summary[:180], abstract=summary[:900],
                is_survey=signals.is_survey(title, summary),
                # arXiv is a preprint server; only a listed journal reference says it was published
                status="published" if venue else "preprint", venue=venue,
            ))
    return items


async def _search_arxiv(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    query = (interest.concepts[0] if interest.concepts else interest.topic)
    if offline_mode():
        return _mock_papers(interest.topic)
    _arxiv_cache_load()
    cached = _arxiv_cache.get(query)
    if cached is not None:
        return [ContentItem(**{**asdict(i), "topic": interest.topic}) for i in cached]
    try:
        for attempt in (1, 2):
            await _arxiv_wait_turn()
            resp = await client.get(
                ARXIV_API_URL,
                params={"search_query": f'all:"{query}"', "start": 0, "max_results": 1},
                timeout=_timeout(),
                follow_redirects=True,
            )
            if resp.status_code == 429 and attempt == 1:     # rate limited: wait once, retry once
                try:
                    wait = min(float(resp.headers.get("Retry-After", "")), 60.0)
                except ValueError:
                    wait = ARXIV_RETRY_AFTER_DEFAULT
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            break
        items = _parse_arxiv(resp.text, interest.topic)
        _arxiv_cache[query] = items       # a genuine "no match" is cached too
        _arxiv_cache_save()
        return items
    except Exception as e:  # noqa: BLE001
        _fail("arXiv", query, e)
        return []


# ---------------------------------------------------------------------------
# YouTube (optional, off by default: the product is films + docs)
# ---------------------------------------------------------------------------

_ISO_DURATION = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


def _duration_seconds(iso: str) -> int:
    m = _ISO_DURATION.match(iso or "")
    if not m:
        return 0
    h, mi, sec = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + sec


# Textbook and tutorial formats are not "explaining the idea you were curious about".
_COURSEWARE = re.compile(
    r"\b(lesson|lecture|bootcamp|course|tutorial|walkthrough|for beginners|template|compilation|"
    r"behind the scenes|full episode|reaction|podcast)\b|\bepisode \d+|\bpart \d+|#\d+", re.I)


def _youtube_query(interest: Interest) -> str:
    """The concept, plus the topic's extra words so a bare concept cannot change meaning
    ("memory consolidation" alone finds CPU caches; with "dementia" it does not)."""
    concept = (interest.concepts or [interest.topic])[0]
    have = set(re.findall(r"[a-z0-9]+", concept.lower()))
    extra = [w for w in re.findall(r"[a-z0-9]+", interest.topic.lower())
             if w not in have and w not in STOPWORDS and len(w) > 2 and w not in ("and", "the")]
    return " ".join([concept, *extra[:4], "explained"])


def _title_matches(concept: str, title: str) -> bool:
    """The video title must mention at least one meaningful word of the concept."""
    words = [w for w in re.findall(r"[a-z0-9]+", concept.lower()) if len(w) > 2 and w not in STOPWORDS]
    low = title.lower()
    return not words or any(w in low for w in words)


async def _search_youtube(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    if not _youtube_key():
        if offline_mode():
            return [ContentItem(title=r["title"], content_type="video", source_url=r["url"],
                                topic=interest.topic, topic_overlap_score=0.5, blurb=r["blurb"])
                    for r in MOCK_VIDEOS]
        log.warning("YOUTUBE_API_KEY is not set: no videos returned")
        return []
    concept = (interest.concepts or [interest.topic])[0]
    try:
        resp = await client.get(
            YOUTUBE_SEARCH_URL,
            # a concept plus "explained" finds explainers; raw film-tag keywords find trailers
            params={"part": "snippet", "q": _youtube_query(interest), "type": "video",
                    "maxResults": YOUTUBE_CANDIDATES, "relevanceLanguage": "en",
                    "safeSearch": "moderate", "videoEmbeddable": "true", "key": _youtube_key()},
            timeout=_timeout(),
        )
        resp.raise_for_status()
        hits = [r for r in resp.json().get("items", [])
                if r.get("id", {}).get("videoId") and _title_matches(concept, r["snippet"].get("title", ""))
                and not _COURSEWARE.search(r["snippet"].get("title", ""))]
        if not hits:
            return []
        # one cheap batched lookup for length and views, so shorts and junk can be dropped
        details = await client.get(
            YOUTUBE_VIDEOS_URL,
            params={"part": "contentDetails,statistics", "key": _youtube_key(),
                    "id": ",".join(r["id"]["videoId"] for r in hits)},
            timeout=_timeout(),
        )
        details.raise_for_status()
        info = {d["id"]: d for d in details.json().get("items", [])}
        out = []
        for r in hits:                                     # keep YouTube's relevance order
            d = info.get(r["id"]["videoId"])
            if not d:
                continue
            seconds = _duration_seconds(d.get("contentDetails", {}).get("duration", ""))
            views = int(d.get("statistics", {}).get("viewCount", 0) or 0)
            if not (VIDEO_MIN_SECONDS <= seconds <= VIDEO_MAX_SECONDS) or views < VIDEO_MIN_VIEWS:
                continue
            out.append(ContentItem(
                title=r["snippet"]["title"], content_type="video",
                source_url=f"https://www.youtube.com/watch?v={r['id']['videoId']}",
                topic=interest.topic, topic_overlap_score=0.7,
                blurb=f"{r['snippet'].get('channelTitle', '')}: {r['snippet'].get('description', '')}"[:160],
                abstract=f"{r['snippet'].get('title', '')}. {r['snippet'].get('description', '')}"[:600],
                duration_s=seconds,
                year=(r["snippet"].get("publishedAt") or "")[:4], popularity=views,
                matched_keywords=[concept]))
            if len(out) >= YOUTUBE_PER_INTEREST:
                break
        return out
    except Exception as e:  # noqa: BLE001
        _fail("YouTube", interest.topic, e)
        return []


OPENALEX_URL = "https://api.openalex.org/works"


def _title_key(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


async def _openalex_enrich(items: list[ContentItem], client: httpx.AsyncClient) -> None:
    """Add citation counts to papers. A match must have a near-identical title, so a wrong
    paper's count is never attached. Any failure leaves `citations` as None (unknown)."""
    async def one(item: ContentItem) -> None:
        params = {"search": item.title, "per-page": 1, "select": "title,cited_by_count"}
        if os.environ.get("OPENALEX_MAILTO"):
            params["mailto"] = os.environ["OPENALEX_MAILTO"]
        try:
            resp = await client.get(OPENALEX_URL, params=params, timeout=_timeout())
            resp.raise_for_status()
            hit = (resp.json().get("results") or [None])[0]
            if not hit:
                return
            a, b = _title_key(item.title), _title_key(hit.get("title") or "")
            if a and b and len(a & b) / len(a | b) >= 0.85 and isinstance(hit.get("cited_by_count"), int):
                item.citations = hit["cited_by_count"]
        except Exception as e:  # noqa: BLE001
            _fail("OpenAlex", item.title[:40], e)

    todo = [i for i in items if i.content_type == "paper" and i.citations is None]
    if todo:
        await asyncio.gather(*[one(i) for i in todo[:8]])


async def _challenge_search(interest: Interest, client: httpx.AsyncClient) -> list[ContentItem]:
    """The opt-in 'challenge me' slot: films/series/docs found with the extractor's counter-keywords,
    i.e. a view that complicates the person's own. Tagged so ranking can reserve one slot."""
    probe = Interest(topic=interest.topic, depth=interest.depth, sentiment=interest.sentiment, mentions=1,
                     keywords=list(interest.counter_keywords), confidence=interest.confidence)
    out = await _search_tmdb(probe, client)
    for item in out:
        item.challenge = interest.counterpoint
    return out


def challenge_enabled(flag: bool | None) -> bool:
    return flag if flag is not None else os.environ.get("KNOTS_CHALLENGE") == "1"


async def search_content(interests: list[Interest],
                         client: httpx.AsyncClient | None = None,
                         include_videos: bool | None = None,
                         challenge: bool | None = None) -> list[ContentItem]:
    """Fan out interests to the catalogs in parallel. `client` is injectable for tests.
    `include_videos=None` defers to the INCLUDE_VIDEOS env flag; `challenge=None` to KNOTS_CHALLENGE."""
    if include_videos is None:
        include_videos = os.environ.get("INCLUDE_VIDEOS") == "1"
    owns = client is None
    client = client or httpx.AsyncClient(limits=httpx.Limits(max_connections=6))
    try:
        tasks = []
        for n, interest in enumerate(interests[:MAX_SEARCH_INTERESTS]):
            allowed = allowed_mediums(interest)           # skip formats that would feel forced
            if "film" in allowed:
                tasks.append(_search_tmdb(interest, client))
            if "paper" in allowed and n < ARXIV_MAX_INTERESTS:
                tasks.append(_search_arxiv(interest, client))
            if "video" in allowed and include_videos and n < YOUTUBE_MAX_INTERESTS:
                tasks.append(_search_youtube(interest, client))
        if challenge_enabled(challenge) and not offline_mode():
            lead = next((i for i in interests[:MAX_SEARCH_INTERESTS] if i.counter_keywords and i.counterpoint), None)
            if lead is not None:
                tasks.append(_challenge_search(lead, client))
        groups = await asyncio.gather(*tasks)
        items = [item for g in groups for item in g]
        if not offline_mode() and os.environ.get("KNOTS_OPENALEX", "1") != "0":
            await _openalex_enrich(items, client)
    finally:
        if owns:
            await client.aclose()
    return items
