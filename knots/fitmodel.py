"""
How well does a candidate fit what this person is actually asking? One small linear model.

    score = sum(weight[f] * feature[f])

    sim_session  similarity of the candidate (title + abstract + matched tags) to the session text
    sim_concept  similarity to the interest it was retrieved for (topic + concepts)
    overlap      the catalog's tag-overlap score (how many of the interest's keywords it carries)
    quality      TMDB rating / 10; neutral for papers and videos (no rating, so no penalty)
    medium       how well the format suits this interest (0 best, negative worse)
    authority    papers only: citations (log) and survey status

The two similarities and the overlap are z-scored within the candidate set, so the weights do not
depend on which embedder ran (Titan or the offline hashed one).

Weights: `DEFAULT_WEIGHTS` are hand-set priors. `eval/train_rank.py` fits them from your pick
labels and writes knots/brain/data/rank_weights.json (refusing to do so with too few sessions),
including a calibration fitted on held-out scores. Without a shipped calibration a card carries NO
confidence number: an uncalibrated probability would be invented.

Popularity is not a feature. Neither is anything the catalogs do not tell us.
"""
from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import calibration, signals
from .embeddings import cosine, embed_sync, hashed_embedding

log = logging.getLogger("knots.fitmodel")

FEATURES = ("sim_session", "sim_concept", "overlap", "quality", "medium", "authority")
DEFAULT_WEIGHTS = {"sim_session": 1.0, "sim_concept": 0.6, "overlap": 0.5,
                   "quality": 0.4, "medium": 3.0, "authority": 0.5}
NEUTRAL_QUALITY = 0.75
WEIGHTS_FILE = Path(__file__).parent / "brain" / "data" / "rank_weights.json"


@lru_cache(maxsize=1)
def _artifact() -> dict:
    try:
        return json.loads(WEIGHTS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def reload() -> None:
    _artifact.cache_clear()


def weights() -> dict[str, float]:
    w = _artifact().get("weights")
    try:
        return {f: float(w[f]) for f in FEATURES}
    except (TypeError, KeyError, ValueError):
        return dict(DEFAULT_WEIGHTS)


def trained() -> bool:
    return bool(_artifact().get("weights"))


def candidate_text(item) -> str:
    return " ".join(p for p in (item.title, item.abstract or item.blurb, " ".join(item.matched_keywords)) if p)


_EMBED_CACHE: dict[str, tuple[tuple[float, ...], str]] = {}


def _embed(text: str) -> tuple[tuple[float, ...], str]:
    """Cache only real Titan vectors; a throttled fallback must not be remembered, or the text stays hashed forever."""
    hit = _EMBED_CACHE.get(text)
    if hit is not None:
        return hit
    vec, name = embed_sync(text)
    out = (tuple(vec), name)
    if name != "hashed" or os.environ.get("EMBEDDER", "auto").lower() == "hashed":
        if len(_EMBED_CACHE) >= 4096:
            _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
        _EMBED_CACHE[text] = out
    return out


def embed_all(texts: list[str]) -> tuple[list[list[float]], str]:
    """Vectors for texts, all from ONE embedder (if any call fell back, redo them all hashed).

    Duplicate texts (e.g. the same interest's concept string repeated for every item) are embedded once:
    Titan is rate-limited to ~1 call/second, so every avoided call matters."""
    unique = list(dict.fromkeys(texts))
    with ThreadPoolExecutor(max_workers=int(os.environ.get("EMBED_WORKERS", "3"))) as pool:
        got = dict(zip(unique, pool.map(_embed, unique)))
    names = {n for _, n in got.values()}
    if len(names) > 1:
        return [hashed_embedding(t) for t in texts], "hashed"
    return [list(got[t][0]) for t in texts], (names.pop() if names else "hashed")


def zscore(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return [0.0] * len(xs) if sd < 1e-9 else [(x - m) / sd for x in xs]


def quality_of(rating: float | None) -> float:
    return rating / 10.0 if rating else NEUTRAL_QUALITY


@dataclass
class Scored:
    """Features, scores and vectors for a candidate set, computed once and shared by ranking and pairing."""
    items: list
    features: list[dict]          # per item: feature name -> value (z-scored where described above)
    scores: list[float]
    sims: list[float]             # raw session similarity per item
    vecs: list[list[float]]       # item embedding per item
    embedder: str
    query_used: bool

    def index(self, item) -> int:
        return next(i for i, it in enumerate(self.items) if it is item)

    def confidence(self, i: int) -> float | None:
        """Calibrated P(good) for item i, or None when no calibration has been fitted."""
        cal = calibration.from_meta(_artifact())
        return None if cal is None or not self.query_used else round(calibration.apply(self.scores[i], *cal), 2)


def score_items(query: str, items: list, interests: list, medium_weight) -> Scored:
    """`medium_weight(item, interest)` is ranking.medium_weight (passed in to avoid a cycle)."""
    by_topic = {i.topic: i for i in interests}
    q = (query or "").strip()
    texts = [candidate_text(i) for i in items]
    concept_texts = [" ".join([by_topic[i.topic].topic, *by_topic[i.topic].concepts]) if i.topic in by_topic else i.topic
                     for i in items]
    vecs, name = embed_all(texts + concept_texts + ([q] if q else [])) if items else ([], "hashed")
    n = len(items)
    iv, cv = vecs[:n], vecs[n:2 * n]
    qv = vecs[2 * n] if q else None
    sims = [cosine(qv, v) if qv is not None else 0.0 for v in iv]
    csims = [cosine(a, b) for a, b in zip(iv, cv)]
    z_s, z_c = zscore(sims), zscore(csims)
    z_o = zscore([i.topic_overlap_score for i in items])
    feats = []
    for k, it in enumerate(items):
        interest = by_topic.get(it.topic)
        feats.append({
            "sim_session": z_s[k], "sim_concept": z_c[k], "overlap": z_o[k],
            "quality": quality_of(it.rating),
            "medium": (medium_weight(it, interest) - 1.0) if interest else 0.0,
            "authority": signals.authority(it),
        })
    w = weights()
    scores = [sum(w[f] * x[f] for f in FEATURES) for x in feats]
    return Scored(items, feats, scores, sims, iv, name, bool(q))
