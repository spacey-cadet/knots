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

import interests as I
from catalogs import search_content
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


def _offline_catalogs(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    # arXiv has no key, so give it a transport that always fails -> mock fallback
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))


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
        os.environ.pop("TMDB_API_KEY", None)
        found, used = I.extract_sync(SAMPLE_SESSION)
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        items = asyncio.run(search_content(found, client=client, include_videos=True))
        print("\n=== Sample thinking_partner output (offline, heuristic) ===")
        print(json.dumps(build_session_plan(found, items, extractor=used), indent=2))
    sys.exit(code)
