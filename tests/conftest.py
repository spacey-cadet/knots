import pytest


@pytest.fixture(autouse=True)
def _isolated_event_log(monkeypatch, tmp_path):
    """Tools that show picks log impressions; no test may write into the repository's data/ folder."""
    from knots import feedback
    monkeypatch.setenv("KNOTS_EVENT_LOG", str(tmp_path / "events.jsonl"))
    feedback._log = None
    yield
    feedback._log = None


@pytest.fixture(autouse=True)
def _fresh_embedder_state(monkeypatch):
    """The Titan circuit breaker and the embedding cache are process-wide; isolate every test from them."""
    from knots import catalogs, embeddings, fitmodel
    catalogs._youtube_reset_for_tests()              # cache, usage counter and the quota circuit breaker
    monkeypatch.delenv("KNOTS_YOUTUBE_CACHE", raising=False)
    monkeypatch.delenv("KNOTS_YOUTUBE_BUDGET", raising=False)
    embeddings.reset_breaker()
    fitmodel._EMBED_CACHE.clear()
    monkeypatch.setenv("TITAN_RETRIES", "1")        # no sleeping on simulated failures
    monkeypatch.setattr(embeddings, "_TITAN_RETRIES", 1)
    monkeypatch.setattr(embeddings, "_TITAN_RPM", 0.0)      # no pacing sleeps; the pacing tests set their own
    monkeypatch.setattr(embeddings, "_COHERE_RPM", 0.0)
    yield
    embeddings.reset_breaker()
    catalogs._youtube_reset_for_tests()
