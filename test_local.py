"""
Offline tests: no network, no AWS credentials, no API keys.

    python test_local.py     # runs the tests, then prints a sample partner session
    pytest -q                # same tests under pytest

Bedrock is exercised through a fake client that returns the same response shape
the real Converse API does, so the parsing, validation and fallback logic that
runs in production is what gets tested here.
"""

import asyncio
import json

import httpx
import pytest

import catalogs as C
import interests as I
from cards import rank_and_render
from catalogs import ContentItem, passes_quality, search_content
import history as H
from datetime import datetime, timedelta, timezone
from partner import build_session_plan

SAMPLE_SESSION = [
    "Alexa, what's a memecoin and why do people buy them?",
    "Can you explain memecoin tokenomics to me, like liquidity pools and market cap?",
    "I keep seeing memecoin threads on my Reddit feed, is this all just a bubble?",
    "Tell me more about the research behind speculative crypto assets.",
]

MODEL_OUTPUT = {"interests": [{
    "topic": "memecoins",
    "concepts": ["collective belief", "reflexivity"],
    "implicit_themes": ["trust without institutions", "fear of missing out"],
    "keywords": ["financial crisis", "stock market", "greed"],
    "rationale": "Asking why people buy them and whether it's a bubble points at belief creating value.",
    "depth": "deep", "sentiment": "skeptical", "confidence": 0.8,
}]}


class FakeBedrock:
    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output, error, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"output": {"message": {"content": [
            {"toolUse": {"name": "record_interests", "input": self.output}}]}}}


def _use(monkeypatch, fake, mode="auto"):
    monkeypatch.setenv("EXTRACTOR", mode)
    monkeypatch.setenv("BEDROCK_MODEL_ID", "test-model")
    monkeypatch.delenv("EXTRACTOR_STRICT", raising=False)
    monkeypatch.setattr(I, "_client", fake)
    I._cache.clear()


@pytest.fixture(autouse=True)
def _fast_arxiv(monkeypatch):
    monkeypatch.setattr(C, "ARXIV_MIN_INTERVAL", 0)
    monkeypatch.setattr(C, "ARXIV_RETRY_AFTER_DEFAULT", 0)
    C._arxiv_cache.clear()
    C._arxiv_disk_loaded = False
    C.FAILURES.clear()
    monkeypatch.delenv("KNOTS_ARXIV_CACHE", raising=False)


def _offline_catalogs(monkeypatch):
    """Explicit offline mode: the fixed mock catalogs are allowed, nothing else is."""
    monkeypatch.setenv("KNOTS_OFFLINE", "1")
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))


def _topic(name="memecoins"):
    return I.Interest(topic=name, depth="casual", sentiment="curious", mentions=1,
                      concepts=[name], keywords=[name])


def _film(title, rating, votes, year="2015", score=0.8, topic="memecoins"):
    return ContentItem(title=title, content_type="movie", source_url=f"https://x/{title}",
                       topic=topic, topic_overlap_score=score, year=year,
                       popularity=votes, rating=rating)


def test_heuristic_only(monkeypatch):
    monkeypatch.setenv("EXTRACTOR", "heuristic")
    found, used = I.extract_sync(SAMPLE_SESSION)
    assert used == "heuristic" and found and found[0].topic


def test_bedrock_extracts_underlying_concepts(monkeypatch):
    fake = FakeBedrock(MODEL_OUTPUT)
    _use(monkeypatch, fake)
    found, used = I.extract_sync(SAMPLE_SESSION)
    assert used == "bedrock"
    top = found[0]
    assert "reflexivity" in top.concepts
    assert "trust without institutions" in top.implicit_themes
    assert top.search_terms()[0] == "financial crisis"      # keywords lead, shaped for TMDB
    sent = fake.calls[0]["messages"][0]["content"][0]["text"]
    assert sent.startswith("<session>") and "memecoin" in sent  # untrusted text is fenced


def test_falls_back_on_any_bedrock_failure(monkeypatch):
    _use(monkeypatch, FakeBedrock(error=RuntimeError("AccessDenied")))
    found, used = I.extract_sync(SAMPLE_SESSION)
    assert used == "heuristic (bedrock failed)" and found


def test_falls_back_on_malformed_model_output(monkeypatch):
    _use(monkeypatch, FakeBedrock({"interests": "not a list"}))
    _, used = I.extract_sync(SAMPLE_SESSION)
    assert used == "heuristic (bedrock failed)"


def test_model_output_is_clamped(monkeypatch):
    messy = {"interests": [{"topic": "  X  " * 40, "concepts": ["A", "a", 7, None, "b"],
                            "keywords": ["k"] * 30, "confidence": 99, "depth": "bogus"}]}
    _use(monkeypatch, FakeBedrock(messy))
    top = I.extract_sync(["anything"])[0][0]
    assert len(top.topic) <= 60 and top.concepts == ["a", "b"]
    assert top.confidence == 1.0 and top.depth == "casual" and len(top.keywords) == 1


def test_partner_plan_groups_picks_with_reasons(monkeypatch):
    _use(monkeypatch, FakeBedrock(MODEL_OUTPUT))
    client = _offline_catalogs(monkeypatch)
    found, used = I.extract_sync(SAMPLE_SESSION)
    items = asyncio.run(search_content(found, client=client, include_videos=True))
    plan = build_session_plan(found, items, extractor=used)

    assert plan["extractor"] == "bedrock"
    assert plan["threads"][0]["underneath"] == ["trust without institutions", "fear of missing out"]
    assert plan["picks"]["watch"] and plan["picks"]["read"] and plan["picks"]["explore"]
    assert all(c["reasoning"] for group in plan["picks"].values() for c in group)
    assert "trust without institutions" in plan["spoken"]
    assert plan["ask"].endswith("?")
    urls = [c["url"] for group in plan["picks"].values() for c in group]
    assert len(urls) == len(set(urls))                      # no title offered twice


def test_live_failure_returns_nothing_not_fake_items(monkeypatch):
    # keys present, every request fails, NOT in offline mode -> no mock titles may appear
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("TMDB_API_KEY", "k")
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    items = asyncio.run(search_content([_topic()], client=client, include_videos=True))
    assert items == []


def test_missing_keys_return_nothing_unless_offline(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert asyncio.run(search_content([_topic()], client=client, include_videos=True)) == []
    monkeypatch.setenv("KNOTS_OFFLINE", "1")
    assert asyncio.run(search_content([_topic()], client=client, include_videos=True))


def test_arxiv_follows_http_redirect(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    atom = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.00001v1</id>'
            '<title>Reflexivity in markets</title><summary>A real paper.</summary></entry></feed>')
    hits = []

    def handler(request):
        hits.append(str(request.url))
        if request.url.scheme == "http":
            return httpx.Response(301, headers={"location": str(request.url).replace("http://", "https://", 1)})
        return httpx.Response(200, text=atom)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert C.ARXIV_API_URL.startswith("https://")      # the shipped default needs no redirect
    # and even if it were http://, the redirect must be followed rather than swallowed
    monkeypatch.setattr(C, "ARXIV_API_URL", "http://export.arxiv.org/api/query")
    items = asyncio.run(C._search_arxiv(_topic(), client))
    assert [i.title for i in items] == ["Reflexivity in markets"] and len(hits) == 2


ATOM = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.00001v1</id>'
        '<title>Reflexivity in markets</title><summary>A real paper.</summary></entry></feed>')


def test_arxiv_retries_once_on_429_then_caches(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "0"}) if len(calls) == 1 else httpx.Response(200, text=ATOM)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first = asyncio.run(C._search_arxiv(_topic(), client))
    assert [i.title for i in first] == ["Reflexivity in markets"] and len(calls) == 2   # 429, then success
    asyncio.run(C._search_arxiv(_topic(), client))
    assert len(calls) == 2                                  # same query: served from cache
    asyncio.run(C._search_arxiv(_topic("grief"), client))
    assert len(calls) == 3                                  # different query: one new request


def test_arxiv_gives_up_after_second_429_without_fake_paper(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    assert asyncio.run(C._search_arxiv(_topic(), client)) == []


def test_arxiv_requests_are_spaced(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setattr(C, "ARXIV_MIN_INTERVAL", 0.2)
    C._arxiv_next_slot = 0.0
    times = []

    def handler(request):
        times.append(__import__("time").monotonic())
        return httpx.Response(200, text=ATOM)

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await asyncio.gather(*[C._search_arxiv(_topic(f"t{i}"), client) for i in range(3)])

    asyncio.run(go())
    gaps = [b - a for a, b in zip(sorted(times), sorted(times)[1:])]
    assert len(times) == 3 and all(g >= 0.15 for g in gaps)


def test_failures_are_recorded_so_incomplete_runs_are_visible(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    asyncio.run(C._search_arxiv(_topic(), client))
    assert C.FAILURES and C.FAILURES[0].startswith("arXiv:memecoins:429")


def test_tmdb_retries_connect_timeouts(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "k")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectTimeout("slow")
        return httpx.Response(200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert asyncio.run(C._tmdb_get(client, "/search/keyword", {"query": "x"})) == {"results": []}
    assert len(calls) == 3


def test_arxiv_disk_cache_survives_restart(monkeypatch, tmp_path):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("KNOTS_ARXIV_CACHE", str(tmp_path / "c.json"))
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, text=ATOM)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    asyncio.run(C._search_arxiv(_topic(), client))
    C._arxiv_cache.clear()
    C._arxiv_disk_loaded = False                      # simulate a new process
    again = asyncio.run(C._search_arxiv(_topic(), client))
    assert len(calls) == 1 and again[0].title == "Reflexivity in markets"


def test_quality_gate_and_age_rule():
    assert passes_quality(7.0, 500, "2015")
    assert not passes_quality(5.9, 50_000, "2015")        # famous but poorly rated
    assert not passes_quality(8.0, 40, "2015")            # too few votes to trust
    assert not passes_quality(7.0, 5_000, "1975")         # old films need 7.5+
    assert not passes_quality(8.0, 600, "1975")           # ... and 1,000+ votes
    assert passes_quality(7.6, 1_200, "1975")
    assert passes_quality(7.0, 500, "")                   # unknown year = no old-film penalty


def test_tmdb_drops_films_below_the_gate(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("TMDB_API_KEY", "k")

    def film(i, title, rating, votes, date):
        return {"id": i, "title": title, "vote_average": rating, "vote_count": votes,
                "release_date": date, "overview": ""}

    def handler(request):
        path = request.url.path
        if path.endswith("/search/keyword"):
            return httpx.Response(200, json={"results": [{"id": 1, "name": "memecoins"}]})
        if path.endswith("/discover/tv") or "99" in request.url.params.get("with_genres", ""):
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": [
            film(1, "Good Modern", 7.4, 800, "2018-01-01"),
            film(2, "Famous Flop", 4.8, 90_000, "2016-01-01"),
            film(3, "Tiny Cult", 8.0, 30, "2019-01-01"),
            film(4, "Old Meh", 7.0, 9_000, "1971-01-01"),
        ]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    items = asyncio.run(C._search_tmdb(_topic(), client))
    assert [i.title for i in items] == ["Good Modern"]
    assert items[0].rating == 7.4 and items[0].popularity == 800


def test_series_are_discovered_gated_and_linked_to_tv(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("TMDB_API_KEY", "k")

    def show(i, name, rating, votes, date):
        return {"id": i, "name": name, "vote_average": rating, "vote_count": votes,
                "first_air_date": date, "overview": ""}

    def handler(request):
        path = request.url.path
        if path.endswith("/search/keyword"):
            return httpx.Response(200, json={"results": [{"id": 1, "name": "memecoins"}]})
        if path.endswith("/discover/tv"):
            assert request.url.params["without_genres"] == "99"     # documentaries stay on the doc path
            return httpx.Response(200, json={"results": [
                show(7, "Severance", 8.2, 4000, "2022-02-18"),
                show(8, "Meh Show", 6.4, 900, "2019-01-01"),        # below the TV floor
                show(9, "Obscure", 8.5, 20, "2021-01-01"),          # too few votes
                show(10, "Old Series", 7.2, 5000, "1975-01-01"),    # old: needs 7.5+
            ]})
        return httpx.Response(200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    items = asyncio.run(C._search_tmdb(_topic(), client))
    assert [(i.title, i.content_type) for i in items] == [("Severance", "series")]
    assert items[0].source_url == "https://www.themoviedb.org/tv/7"


def test_same_tmdb_id_for_a_film_and_a_series_are_kept_apart(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("TMDB_API_KEY", "k")

    def handler(request):
        path = request.url.path
        if path.endswith("/search/keyword"):
            return httpx.Response(200, json={"results": [{"id": 1, "name": "memecoins"}]})
        if path.endswith("/discover/tv"):
            return httpx.Response(200, json={"results": [{"id": 5, "name": "The Show", "vote_average": 8.0,
                                                          "vote_count": 500, "first_air_date": "2020-01-01"}]})
        if "99" in request.url.params.get("with_genres", ""):
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": [{"id": 5, "title": "The Film", "vote_average": 7.5,
                                                      "vote_count": 500, "release_date": "2020-01-01"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    items = asyncio.run(C._search_tmdb(_topic(), client))
    assert sorted(i.title for i in items) == ["The Film", "The Show"]


def test_series_card_and_watch_grouping():
    show = ContentItem(title="Severance", content_type="series", source_url="https://x/tv/7", topic="memecoins",
                       topic_overlap_score=0.9, year="2022", popularity=4000, rating=8.2)
    film = _film("Margin Call", rating=7.1, votes=60_000, score=0.8)
    plan = build_session_plan([_topic()], [show, film], extractor="bedrock")
    titles = [c["title"] for c in plan["picks"]["watch"]]
    assert titles == ["Severance", "Margin Call"]
    assert "series" in plan["picks"]["watch"][0]["reasoning"]


def _yt_handler(videos, details):
    def handler(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"items": [
                {"id": {"videoId": vid}, "snippet": {"title": title, "channelTitle": "Chan",
                                                     "description": "d", "publishedAt": "2023-05-01T00:00:00Z"}}
                for vid, title in videos]})
        assert request.url.path.endswith("/videos")
        return httpx.Response(200, json={"items": [
            {"id": vid, "contentDetails": {"duration": dur}, "statistics": {"viewCount": str(views)}}
            for vid, dur, views in details]})
    return handler


def test_youtube_filters_shorts_junk_and_off_topic(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")
    videos = [("a", "Reflexivity explained"), ("b", "Reflexivity in 30 seconds"), ("c", "Reflexivity lecture series"),
              ("d", "Reflexivity unknown channel"), ("e", "Cat videos compilation"), ("f", "Reflexivity, Soros and markets")]
    details = [("a", "PT12M30S", 250_000), ("b", "PT0M30S", 9_000_000),   # short: dropped
               ("c", "PT2H10M5S", 80_000),                                # too long: dropped
               ("d", "PT9M", 800),                                        # too few views: dropped
               ("e", "PT10M", 5_000_000),                                 # off topic: dropped before lookup
               ("f", "PT1H", 40_000)]
    client = httpx.AsyncClient(transport=httpx.MockTransport(_yt_handler(videos, details)))
    topic = I.Interest(topic="markets", depth="deep", sentiment="curious", mentions=1,
                       concepts=["reflexivity"], keywords=["reflexivity"])
    items = asyncio.run(C._search_youtube(topic, client))
    assert [i.title for i in items] == ["Reflexivity explained", "Reflexivity, Soros and markets"]
    assert items[0].source_url == "https://www.youtube.com/watch?v=a" and items[0].popularity == 250_000


def test_youtube_failure_returns_nothing(monkeypatch):
    monkeypatch.delenv("KNOTS_OFFLINE", raising=False)
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    assert asyncio.run(C._search_youtube(_topic(), client)) == []
    assert C.FAILURES[-1].startswith("YouTube:")


def test_iso_duration_parsing():
    assert C._duration_seconds("PT1H2M3S") == 3723 and C._duration_seconds("PT45S") == 45
    assert C._duration_seconds("P1DT0S") == 0 and C._duration_seconds("") == 0   # unparseable = excluded


def test_relevance_decides_not_popularity():
    famous = _film("Famous", rating=7.0, votes=90_000, score=0.5)
    fitting = _film("Fitting", rating=7.0, votes=400, score=0.9, topic="grief")
    cards = rank_and_render([famous, fitting], [_topic(), _topic("grief")], limit=6)
    assert cards[0]["title"] == "Fitting"


def test_one_slot_is_reserved_for_a_lesser_known_gem():
    popular = [_film(f"Big{i}", rating=7.8, votes=50_000, score=0.9 - i * 0.01, topic=f"t{i}") for i in range(4)]
    gem = _film("Hidden", rating=7.6, votes=600, score=0.6, topic="t0")
    topics = [_topic(f"t{i}") for i in range(4)]
    cards = rank_and_render(popular + [gem], topics, limit=4)
    assert any(c["title"] == "Hidden" and c["gem"] for c in cards)
    assert [c["title"] for c in cards].index("Hidden") <= 2
    # no gem available -> nothing forced in
    assert not any(c["gem"] for c in rank_and_render(popular, topics, limit=4))


def test_empty_catalogs_yield_an_honest_note(monkeypatch):
    plan = build_session_plan([_topic()], [], extractor="heuristic")
    assert not any(plan["picks"].values()) and "not guessing" in plan["note"]


def test_label_server_accepts_reason_only_on_not_good_picks(monkeypatch, tmp_path):
    import importlib.util
    import threading
    import urllib.request
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parent / "eval" / "run_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    labels = tmp_path / "labels.csv"
    row = {c: "" for c in ev.LABEL_COLS} | {"session_id": "s1", "extractor": "x", "url": "u", "title": "T"}
    ev.write_csv(labels, ev.LABEL_COLS, [row])
    monkeypatch.setattr(ev, "LABELS", labels)
    monkeypatch.setitem(ev.SHEETS, "pick", (labels, ev.LABEL_COLS, ("session_id", "extractor", "url"), "good"))
    monkeypatch.setattr(ev, "THEMES", tmp_path / "themes.csv")

    # drive the handler through a real server on a free port
    port = 8799
    args = type("A", (), {"port": port, "no_browser": True})()
    t = threading.Thread(target=ev.cmd_label, args=(args,), daemon=True)
    t.start()
    import time
    time.sleep(0.5)
    rid = ev.row_id("pick", row)

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/label", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    assert post({"kind": "pick", "id": rid, "value": "y", "reason": "quality"}) == 400   # y has no reason
    assert post({"kind": "pick", "id": rid, "value": "n", "reason": "bogus"}) == 400
    assert post({"kind": "pick", "id": rid, "value": "n", "reason": "quality"}) == 200
    saved = ev.read_csv(labels)[0]
    assert saved["good"] == "n" and saved["reason"] == "quality"


# ---------------------------------------------------------------------------
# input scope: one conversation vs many
# ---------------------------------------------------------------------------

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def _conv(text, days_ago=None, cid=""):
    ts = (NOW - timedelta(days=days_ago)).isoformat() if days_ago is not None else None
    return H.Conversation(turns=[text], timestamp=ts, id=cid)


def _it(topic, concepts=None, conf=0.8, themes=None, keywords=None):
    return I.Interest(topic=topic, depth="casual", sentiment="curious", mentions=1,
                      concepts=concepts or [topic], keywords=keywords or [topic],
                      implicit_themes=themes or [], confidence=conf)


def test_bedrock_input_cap_keeps_the_latest_text(monkeypatch):
    fake = FakeBedrock(MODEL_OUTPUT)
    _use(monkeypatch, fake)
    long = ["OLDEST " + "x" * 5000, "y" * 9000, "NEWEST question about memecoins"]
    I.bedrock_extract(long)
    sent = fake.calls[0]["messages"][0]["content"][0]["text"]
    assert "NEWEST" in sent and "OLDEST" not in sent


def test_normalize_orders_trims_and_caps():
    old = _conv("from long ago", days_ago=200)                    # older than the window: dropped
    many = [_conv(f"c{i}", days_ago=i + 1) for i in range(12)]    # more than MAX_CONVERSATIONS
    newest = _conv("right now", days_ago=0)
    out = H.normalize_input(None, [old, *many, newest], now=NOW)
    assert len(out) == H.MAX_CONVERSATIONS
    assert out[-1][0].turns == ["right now"]                       # newest last
    assert all("long ago" not in c.turns[0] for c, _ in out)
    ages = [a for _, a in out]
    assert ages == sorted(ages, reverse=True)                      # oldest -> newest


def test_newest_conversation_is_kept_even_if_old_or_untimestamped():
    out = H.normalize_input(None, [_conv("only one", days_ago=400)], now=NOW)
    assert len(out) == 1
    flat = H.normalize_input(["a", "b"], None, now=NOW)            # old flat input = one conversation
    assert len(flat) == 1 and flat[0][0].turns == ["a", "b"]
    assert H.normalize_input([], [], now=NOW) == [] and H.normalize_input(["  "], None, now=NOW) == []


def test_untimestamped_conversations_age_by_position():
    out = H.normalize_input(None, [H.Conversation(["a"]), H.Conversation(["b"]), H.Conversation(["c"])], now=NOW)
    assert [a for _, a in out] == [2.0, 1.0, 0.0]


def test_long_conversation_keeps_its_latest_turns():
    turns = [f"turn {i} " + "z" * 800 for i in range(10)]
    out = H.normalize_input(None, [H.Conversation(turns)], now=NOW)[0][0].turns
    assert out[-1].startswith("turn 9") and not any(t.startswith("turn 0") for t in out)
    assert sum(len(t) for t in out) <= H.MAX_CHARS_PER_CONVERSATION + 800


def test_recurring_topic_outweighs_a_one_off_and_is_counted():
    convs = [(_conv("a", 20), 20.0, [_it("grief", ["loss"]), _it("chess")]),
             (_conv("b", 10), 10.0, [_it("grief", ["loss"])]),
             (_conv("c", 3), 3.0, [_it("grief"), _it("memecoins", ["reflexivity"])]),
             (_conv("d", 0), 0.0, [_it("sourdough")])]
    merged = H.merge_interests(convs)
    by = {i.topic: i for i in merged}
    assert by["grief"].conversations == 3 and by["chess"].conversations == 1
    assert [i.topic for i in merged].index("grief") < [i.topic for i in merged].index("chess")
    assert by["grief"].in_current is False and by["sourdough"].in_current is True
    assert by["grief"].confidence > 0.8                             # recurrence raises confidence


def test_newest_conversations_strongest_thread_stays_in_the_top_two():
    old = [(_conv(str(i), 30), 30.0, [_it("grief", ["loss"], conf=1.0), _it("chess", conf=1.0),
                                       _it("crypto", ["reflexivity"], conf=1.0)]) for i in range(3)]
    now = [(_conv("now", 0), 0.0, [_it("sourdough", conf=0.3)])]
    merged = H.merge_interests(old + now)
    assert "sourdough" in [i.topic for i in merged[:2]]


def test_threads_merge_by_shared_concept_and_keep_the_union_of_detail():
    a = _it("memecoins", ["reflexivity"], themes=["fear of missing out"], keywords=["financial crisis"])
    b = _it("crypto bubbles", ["reflexivity", "belief"], themes=["trust without institutions"], keywords=["stock market"])
    merged = H.merge_interests([(_conv("a", 5), 5.0, [a]), (_conv("b", 0), 0.0, [b])])
    assert len(merged) == 1 and merged[0].conversations == 2
    assert set(merged[0].implicit_themes) == {"fear of missing out", "trust without institutions"}
    assert {"financial crisis", "stock market"} <= set(merged[0].keywords)


def test_unrelated_topics_are_not_merged():
    merged = H.merge_interests([(_conv("a", 5), 5.0, [_it("memory loss", ["dementia"])]),
                                (_conv("b", 0), 0.0, [_it("memecoins", ["reflexivity"])])])
    assert len(merged) == 2


def test_extract_history_runs_one_call_per_conversation_and_merges(monkeypatch):
    fake = FakeBedrock(MODEL_OUTPUT)
    _use(monkeypatch, fake)
    convs = [_conv("what is a memecoin?", 4), _conv("is crypto just a bubble?", 1), _conv("memecoins again", 0)]
    found, used = asyncio.run(H.extract_history(None, convs, now=NOW))
    assert used == "bedrock" and len(fake.calls) == 3
    assert found[0].topic == "memecoins" and found[0].conversations == 3


def test_extract_history_single_conversation_is_the_old_path(monkeypatch):
    fake = FakeBedrock(MODEL_OUTPUT)
    _use(monkeypatch, fake)
    found, used = asyncio.run(H.extract_history(SAMPLE_SESSION))
    assert used == "bedrock" and len(fake.calls) == 1 and found[0].conversations == 1


def test_extract_history_reports_mixed_extractors(monkeypatch):
    class Flaky(FakeBedrock):
        def converse(self, **kw):
            if "second" in kw["messages"][0]["content"][0]["text"]:
                raise RuntimeError("throttled")
            return super().converse(**kw)

    _use(monkeypatch, Flaky(MODEL_OUTPUT))
    found, used = asyncio.run(H.extract_history(None, [_conv("first memecoin chat", 2), _conv("second chat", 0)], now=NOW))
    assert used.startswith("mixed") and found


def test_tools_accept_conversations_and_the_plan_says_it_came_back(monkeypatch):
    import server
    fake = FakeBedrock(MODEL_OUTPUT)
    _use(monkeypatch, fake)
    _offline_catalogs(monkeypatch)
    out = asyncio.run(server.mcp.call_tool("thinking_partner", {"conversations": [
        {"turns": ["what is a memecoin?"], "timestamp": "2026-10-10T08:00:00Z"},
        {"turns": ["is crypto a bubble?"], "timestamp": "2026-10-14T08:00:00Z"},
        {"turns": ["tokenomics of memecoins"], "timestamp": "2026-10-15T08:00:00Z"}]}))
    plan = json.loads(out[0][0].text) if isinstance(out, tuple) else json.loads(out[0].text)
    assert plan["threads"][0]["conversations"] == 3
    assert "across 3 conversations" in plan["spoken"]


def test_tools_still_accept_the_old_flat_input(monkeypatch):
    import server
    _use(monkeypatch, FakeBedrock(MODEL_OUTPUT))
    _offline_catalogs(monkeypatch)
    out = asyncio.run(server.mcp.call_tool("extract_interests", {"session_texts": SAMPLE_SESSION}))
    data = json.loads(out[0][0].text) if isinstance(out, tuple) else json.loads(out[0].text)
    assert data["extractor"] == "bedrock" and data["interests"]


def test_server_exposes_tools():
    import server
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"thinking_partner", "get_recommendations", "extract_interests", "search_content"} <= names


if __name__ == "__main__":
    import os
    import sys

    import pytest
    code = pytest.main(["-q", __file__])
    if code == 0:
        os.environ["EXTRACTOR"] = "heuristic"
        os.environ["KNOTS_OFFLINE"] = "1"   # sample output uses the explicit offline mock catalogs
        os.environ.pop("TMDB_API_KEY", None)
        found, used = I.extract_sync(SAMPLE_SESSION)
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        items = asyncio.run(search_content(found, client=client, include_videos=True))
        print("\n=== Sample thinking_partner output (offline, heuristic) ===")
        print(json.dumps(build_session_plan(found, items, extractor=used), indent=2))
    sys.exit(code)
