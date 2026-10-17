import pytest


@pytest.fixture(autouse=True)
def _fresh_embedder_state(monkeypatch):
    """The Titan circuit breaker and the embedding cache are process-wide; isolate every test from them."""
    from knots import embeddings, fitmodel
    embeddings.reset_breaker()
    fitmodel._EMBED_CACHE.clear()
    monkeypatch.setenv("TITAN_RETRIES", "1")        # no sleeping on simulated failures
    monkeypatch.setattr(embeddings, "_TITAN_RETRIES", 1)
    monkeypatch.setattr(embeddings, "_TITAN_RPM", 0.0)      # no pacing sleeps; the pacing test sets its own
    yield
    embeddings.reset_breaker()
