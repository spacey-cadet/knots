"""
Text embeddings for Knots' memory.

Two embedders behind one function, `embed(text) -> (vector, name)`:

  - "titan"  : Amazon Titan Text Embeddings v2 on Bedrock (256 dimensions,
               normalised). Semantic: "what is a memecoin" lands near "crypto
               speculation". Used when Bedrock is configured.
  - "hashed" : an offline bag-of-words hash embedding (unigrams + bigrams,
               signed hashing, L2-normalised). Lexical only, but deterministic,
               free and needs no keys, so the server and tests always work.

EMBEDDER env: auto (default: titan, else hashed) | titan | hashed.
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
_TITAN_SLOTS = threading.BoundedSemaphore(int(os.environ.get("TITAN_CONCURRENCY", "3")))
_TITAN_RETRIES = int(os.environ.get("TITAN_RETRIES", "6"))
_BREAKER_THRESHOLD = int(os.environ.get("TITAN_BREAKER_FAILS", "3"))
_BREAKER_COOLDOWN_S = float(os.environ.get("TITAN_BREAKER_COOLDOWN_S", "30"))
# Titan Text Embeddings V2 is capped at 60 requests/minute and AWS does not allow an increase, so pace calls
# below that (default 50 RPM leaves headroom for other users of the account). Set TITAN_RPM=0 to disable.
_TITAN_RPM = float(os.environ.get("TITAN_RPM", "50"))
_pace_lock = threading.Lock()
_next_slot = 0.0
_breaker_lock = threading.Lock()
_consecutive_failures = 0
_open_until = 0.0
_last_warned = 0.0


def _is_throttle(e: Exception) -> bool:
    code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
    return code in _THROTTLE_CODES


def _pace() -> None:
    """Blocks until the next request slot; every Titan call (including retries) goes through here."""
    global _next_slot
    if _TITAN_RPM <= 0:
        return
    gap = 60.0 / _TITAN_RPM
    with _pace_lock:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + gap
    if wait:
        time.sleep(wait)


def _titan_with_backoff(text: str) -> list[float]:
    """Retries throttling with exponential backoff + jitter; other errors raise immediately."""
    for attempt in range(_TITAN_RETRIES):
        try:
            with _TITAN_SLOTS:
                _pace()
                return titan_embedding(text)
        except Exception as e:  # noqa: BLE001
            if not _is_throttle(e) or attempt == _TITAN_RETRIES - 1:
                raise
            time.sleep(min(20.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError("unreachable")


def reset_breaker() -> None:
    """Close the circuit breaker (used by tests and long-lived servers after a quota change)."""
    global _consecutive_failures, _open_until, _last_warned, _next_slot
    with _breaker_lock:
        _consecutive_failures, _open_until, _last_warned = 0, 0.0, 0.0
    with _pace_lock:
        _next_slot = 0.0


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


def embed_sync(text: str) -> tuple[list[float], str]:
    """Returns (vector, embedder_name). Fallbacks to hashed are marked by the name, so callers can avoid caching them."""
    global _last_warned
    text = (text or "").strip()
    mode = os.environ.get("EMBEDDER", "auto").lower()
    if mode in ("auto", "titan") and (mode == "titan" or os.environ.get("BEDROCK_MODEL_ID")):
        if not _breaker_open():
            try:
                vec = _titan_with_backoff(text)
                _record(True)
                return vec, "titan"
            except Exception as e:  # noqa: BLE001
                _record(False)
                if mode == "titan" and os.environ.get("EXTRACTOR_STRICT") == "1":
                    raise
                now = time.monotonic()
                if now - _last_warned > 10:       # one warning per 10 s, not one per text
                    _last_warned = now
                    log.warning("Titan embedding failed (%s: %s); using the hashed embedder", type(e).__name__, e)
    return hashed_embedding(text), "hashed"


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))      # both sides are L2-normalised
