"""
Text embeddings for Knots' memory.

Three embedders behind `embed_sync(text)` and the batched `embed_many(texts)`, each returning (vectors, name):

  - "titan"  : Amazon Titan Text Embeddings v2 on Bedrock (256 dimensions,
               normalised). Semantic: "what is a memecoin" lands near "crypto
               speculation". Used when Bedrock is configured.
  - "cohere" : Cohere Embed v4 on Bedrock (256 dimensions, normalised), selected by EMBEDDER=cohere or a Cohere
               EMBED_MODEL_ID. Takes up to 96 texts per request, so a whole plan is one call.
  - "hashed" : an offline bag-of-words hash embedding (unigrams + bigrams,
               signed hashing, L2-normalised). Lexical only, but deterministic,
               free and needs no keys, so the server and tests always work.

EMBEDDER env: auto (default: Cohere if EMBED_MODEL_ID names it, else Titan if Bedrock is configured, else hashed)
| titan | cohere | hashed.
The embedder's name is stored next to every vector. Vectors from different
embedders are never compared (their spaces are unrelated).

Embeddings are not raw text, but they are not anonymous either: they keep
enough of the meaning of what was said that they must be treated as personal
data (see README, "Memory").
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time

log = logging.getLogger("knots.embeddings")

DIM = 256
DEFAULT_TITAN_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_COHERE_MODEL = "cohere.embed-v4:0"   # on-demand (10 RPM); use a global/geo cross-region profile id for 20+
MAX_EMBED_CHARS = 8000

_client = None


def _bedrock_client():
    global _client
    if _client is None:
        import boto3  # lazy: offline use must not need boto3
        from botocore.config import Config
        _client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
            config=Config(retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=5,
                          read_timeout=int(os.environ.get("BEDROCK_TIMEOUT_S", "20"))),
        )
    return _client


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


_WORD = re.compile(r"[a-z0-9]+")


def hashed_embedding(text: str, dim: int = DIM) -> list[float]:
    words = _WORD.findall(text.lower())
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    vec = [0.0] * dim
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    return _normalise(vec)


def titan_embedding(text: str) -> list[float]:
    model_id = os.environ.get("EMBED_MODEL_ID", "").strip() or DEFAULT_TITAN_MODEL
    resp = _bedrock_client().invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text[:MAX_EMBED_CHARS], "dimensions": DIM, "normalize": True}),
        accept="application/json", contentType="application/json",
    )
    body = resp["body"]
    data = json.loads(body.read() if hasattr(body, "read") else body)
    vec = data["embedding"]
    if not isinstance(vec, list) or not vec or not all(isinstance(x, (int, float)) for x in vec):
        raise ValueError("embedding response has no usable vector")
    return _normalise([float(x) for x in vec])


_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}
_TITAN_SLOTS = threading.BoundedSemaphore(int(os.environ.get("TITAN_CONCURRENCY", "3")))   # shared by all Bedrock embedders
_TITAN_RETRIES = int(os.environ.get("TITAN_RETRIES", "6"))
_BREAKER_THRESHOLD = int(os.environ.get("TITAN_BREAKER_FAILS", "3"))
_BREAKER_COOLDOWN_S = float(os.environ.get("TITAN_BREAKER_COOLDOWN_S", "30"))
# Titan Text Embeddings V2 is capped at 60 requests/minute and AWS does not allow an increase, so pace calls below
# that (default 50 leaves headroom). Cohere Embed v4 through a cross-region profile is 20/minute but takes up to 96
# texts per request, so the default is 15. Set either to 0 to disable pacing.
_TITAN_RPM = float(os.environ.get("TITAN_RPM", "50"))
_COHERE_RPM = float(os.environ.get("COHERE_RPM", "15"))
COHERE_BATCH = 96                                  # Cohere's maximum number of texts per request
_pace_lock = threading.Lock()
_next_slots: dict[str, float] = {}
_breaker_lock = threading.Lock()
_consecutive_failures = 0
_open_until = 0.0
_last_warned = 0.0


def _is_throttle(e: Exception) -> bool:
    code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
    return code in _THROTTLE_CODES


def _pace(key: str, rpm: float) -> None:
    """Blocks until the next request slot for this embedder; every call (including retries) goes through here."""
    if rpm <= 0:
        return
    gap = 60.0 / rpm
    with _pace_lock:
        now = time.monotonic()
        slot = max(now, _next_slots.get(key, 0.0))
        _next_slots[key] = slot + gap
    if slot > now:
        time.sleep(slot - now)


def _with_backoff(fn, key: str, rpm: float):
    """Paced call; throttling is retried with exponential backoff + jitter, other errors raise immediately."""
    for attempt in range(_TITAN_RETRIES):
        try:
            with _TITAN_SLOTS:
                _pace(key, rpm)
                return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_throttle(e) or attempt == _TITAN_RETRIES - 1:
                raise
            time.sleep(min(20.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError("unreachable")


def cohere_embeddings(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """One Cohere Embed v4 request for up to COHERE_BATCH texts. Returns one normalised vector per text."""
    model_id = os.environ.get("EMBED_MODEL_ID", "").strip() or DEFAULT_COHERE_MODEL
    resp = _bedrock_client().invoke_model(
        modelId=model_id,
        body=json.dumps({"input_type": input_type, "texts": [t[:MAX_EMBED_CHARS] or " " for t in texts],
                         "embedding_types": ["float"], "output_dimension": DIM, "truncate": "RIGHT"}),
        accept="application/json", contentType="application/json",
    )
    body = resp["body"]
    data = json.loads(body.read() if hasattr(body, "read") else body)
    vecs = data["embeddings"]
    if isinstance(vecs, dict):                       # keyed by type when several types are requested
        vecs = vecs.get("float")
    if (not isinstance(vecs, list) or len(vecs) != len(texts)
            or not all(isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) for v in vecs)):
        raise ValueError("embedding response has no usable vectors")
    return [_normalise([float(x) for x in v]) for v in vecs]


def reset_breaker() -> None:
    """Close the circuit breaker and clear pacing (used by tests and long-lived servers after a quota change)."""
    global _consecutive_failures, _open_until, _last_warned
    with _breaker_lock:
        _consecutive_failures, _open_until, _last_warned = 0, 0.0, 0.0
    with _pace_lock:
        _next_slots.clear()


def _breaker_open() -> bool:
    return time.monotonic() < _open_until


def _record(ok: bool) -> None:
    global _consecutive_failures, _open_until
    with _breaker_lock:
        if ok:
            _consecutive_failures = 0
            return
        _consecutive_failures += 1
        if _consecutive_failures >= _BREAKER_THRESHOLD:
            _open_until = time.monotonic() + _BREAKER_COOLDOWN_S
            _consecutive_failures = 0


def _provider() -> str | None:
    """"cohere", "titan" or None (hashed). A Cohere EMBED_MODEL_ID selects Cohere in auto and titan modes too."""
    mode = os.environ.get("EMBEDDER", "auto").lower()
    if mode not in ("auto", "titan", "cohere"):
        return None
    if mode == "auto" and not os.environ.get("BEDROCK_MODEL_ID"):
        return None
    return "cohere" if mode == "cohere" or "cohere" in os.environ.get("EMBED_MODEL_ID", "").lower() else "titan"


def embed_many(texts: list[str], input_type: str = "search_document") -> tuple[list[list[float]], str]:
    """Vectors for texts from ONE embedder (all-or-nothing: if any call fails, every text is embedded hashed).

    Cohere sends up to 96 texts per request, which is what makes it usable under a requests-per-minute cap.
    `input_type` ("search_document" or "search_query") only matters to Cohere; Titan ignores it."""
    global _last_warned
    texts = [(t or "").strip() for t in texts]
    if not texts:
        return [], "hashed"
    provider = _provider()
    if provider and not _breaker_open():
        try:
            if provider == "cohere":
                vecs: list[list[float]] = []
                for i in range(0, len(texts), COHERE_BATCH):
                    chunk = texts[i:i + COHERE_BATCH]
                    vecs += _with_backoff(lambda c=chunk: cohere_embeddings(c, input_type), "cohere", _COHERE_RPM)
            else:
                vecs = [_with_backoff(lambda t=t: titan_embedding(t), "titan", _TITAN_RPM) for t in texts]
            _record(True)
            return vecs, provider
        except Exception as e:  # noqa: BLE001
            _record(False)
            if os.environ.get("EMBEDDER", "auto").lower() in ("titan", "cohere") and os.environ.get("EXTRACTOR_STRICT") == "1":
                raise
            now = time.monotonic()
            if now - _last_warned > 10:               # one warning per 10 s, not one per text
                _last_warned = now
                log.warning("%s embedding failed (%s: %s); using the hashed embedder",
                            provider.capitalize(), type(e).__name__, e)
    return [hashed_embedding(t) for t in texts], "hashed"


def embed_sync(text: str) -> tuple[list[float], str]:
    """Returns (vector, embedder_name). Fallbacks to hashed are marked by the name, so callers can avoid caching them."""
    vecs, name = embed_many([text])
    return vecs[0], name


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))      # both sides are L2-normalised
