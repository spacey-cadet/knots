"""
Beyond-accuracy metrics for a list of recommended picks: diversity, novelty, coverage, serendipity.

Pure functions on pick dicts (the cards a plan already carries), with no I/O, so the offline eval
(`eval/eval_beyond.py`) and the online monitor (`knots/online.py`) compute exactly the same numbers.

What is and is not known. A pick carries its format, matched topic, a popularity count (TMDB votes for films,
series and documentaries, views for videos, citations for papers) and two "how close to the person's own words"
features (`fit_features.overlap`, keyword overlap, and `fit_features.sim_session`, similarity to the session).
It does NOT carry genres, item vectors or the candidate pool the ranker chose from. So:

  * Anything that cannot be computed from what a pick carries is None / NaN, never guessed (a paper with no
    citation count has no novelty; a list of one pick has no diversity).
  * "Rare" and "far from the person's words" are PERCENTILES against a reference pool of picks (`Pool`), so a
    number means "more or less than a typical shown pick", and changes if the pool changes. Popularity is only
    compared inside one source family (votes, views and citations are not on one scale).
  * Coverage has no catalog denominator (TMDB, YouTube and arXiv are not enumerable here), so it reports how
    spread out and how repeated the shown items are, not what fraction of a catalog was reached.

Definitions (each in 0..1 unless stated; higher is "more" of the named thing, not "better"):

  ild              intra-list diversity: mean pairwise cosine DISTANCE between the picks' vectors.
  format_spread    Shannon entropy of the picks' formats over log(list length): 1 = every pick a different format.
  topic_spread     distinct matched topics / picks.
  novelty          1 - popularity percentile (within the source family): how rarely seen the pick is.
  unexpectedness   1 - mean percentile of (keyword overlap, session similarity): how far the pick sits from
                   what the person literally said. The obvious recommendation scores 0.
  serendipity      mean over picks of  good * unexpectedness      (valuable AND not obvious; needs labels)
  serendipitous_share   share of the GOOD picks with unexpectedness >= 0.5

Distance is never rewarded as a goal. Serendipity, novelty and unexpectedness are DESCRIPTIONS of what was shown and
are not to be maximised: they say nothing about whether a link is real. What is "hard work" is measured by
`knots/grounding.py` (the evidence behind a link), and the overreach rate is the gate that stops reaching further
from being mistaken for improvement. An earlier "creativity" metric (good * sqrt(novelty * unexpectedness)) paid for
distance, which rewards far-fetched picks, and was removed.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter

NAN = float("nan")
MIN_POOL = 5            # fewer reference picks than this in a group: the percentile is unknown, not guessed
HALF = 0.5              # "more novel / more unexpected than the typical pick"

_SOURCE = {"movie": "tmdb", "series": "tmdb", "doc": "tmdb", "video": "video", "paper": "paper"}


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return None if v != v else float(v)


def source_family(card: dict) -> str:
    return _SOURCE.get(card.get("type"), card.get("type") or "other")


def popularity(card: dict) -> float | None:
    """log(1 + count) from the source's own popularity count; None when the source did not give one."""
    v = _num(card.get("citations") if card.get("type") == "paper" else card.get("votes"))
    return None if v is None else math.log1p(max(0.0, v))


def _closeness(card: dict) -> list[float]:
    ff = card.get("fit_features") or {}
    return [x for x in (_num(ff.get("overlap")), _num(ff.get("sim_session"))) if x is not None]


def _pct(sorted_vals: list[float], x: float) -> float:
    """Mid-rank percentile of x among sorted_vals, in 0..1 (ties count half)."""
    lo, hi = bisect_left(sorted_vals, x), bisect_right(sorted_vals, x)
    return (lo + 0.5 * (hi - lo)) / len(sorted_vals)


class Pool:
    """Reference distributions from a set of shown picks (each distinct item counted once)."""

    def __init__(self, cards: list[dict], min_pool: int = MIN_POOL):
        seen: dict[str, dict] = {}
        for c in cards:
            seen.setdefault(c.get("url") or c.get("title") or str(id(c)), c)
        self.items = len(seen)
        self.min_pool = min_pool
        self._pop: dict[str, list[float]] = {}
        self._overlap: list[float] = []
        self._sim: list[float] = []
        for c in seen.values():
            p = popularity(c)
            if p is not None:
                self._pop.setdefault(source_family(c), []).append(p)
            ff = c.get("fit_features") or {}
            if _num(ff.get("overlap")) is not None:
                self._overlap.append(float(ff["overlap"]))
            if _num(ff.get("sim_session")) is not None:
                self._sim.append(float(ff["sim_session"]))
        for v in (*self._pop.values(), self._overlap, self._sim):
            v.sort()

    MAX_STORED = 2000   # per distribution; longer ones are thinned evenly (the percentile barely moves)

    def to_dict(self) -> dict:
        """A JSON-safe copy, so a reference pool can be frozen (at promotion time) and compared against later."""
        def thin(v: list[float]) -> list[float]:
            if len(v) <= self.MAX_STORED:
                return list(v)
            step = (len(v) - 1) / (self.MAX_STORED - 1)
            return [v[round(i * step)] for i in range(self.MAX_STORED)]
        return {"items": self.items, "min_pool": self.min_pool, "pop": {k: thin(v) for k, v in self._pop.items()},
                "overlap": thin(self._overlap), "sim": thin(self._sim)}

    @classmethod
    def from_dict(cls, d: dict) -> "Pool":
        p = cls([], d.get("min_pool", MIN_POOL))
        p.items = int(d.get("items", 0))
        p._pop = {k: sorted(float(x) for x in v) for k, v in (d.get("pop") or {}).items()}
        p._overlap = sorted(float(x) for x in d.get("overlap") or [])
        p._sim = sorted(float(x) for x in d.get("sim") or [])
        return p

    def describe(self) -> dict:
        return {"distinct_items": self.items, "popularity_reference": {k: len(v) for k, v in sorted(self._pop.items())},
                "overlap_reference": len(self._overlap), "session_similarity_reference": len(self._sim)}

    def novelty(self, card: dict) -> float | None:
        p, ref = popularity(card), self._pop.get(source_family(card), [])
        return None if p is None or len(ref) < self.min_pool else round(1.0 - _pct(ref, p), 4)

    def unexpectedness(self, card: dict) -> float | None:
        ff = card.get("fit_features") or {}
        parts = []
        for key, ref in (("overlap", self._overlap), ("sim_session", self._sim)):
            v = _num(ff.get(key))
            if v is not None and len(ref) >= self.min_pool:
                parts.append(_pct(ref, v))
        return round(1.0 - sum(parts) / len(parts), 4) if parts else None


def ild(vectors) -> float | None:
    """Mean pairwise cosine distance (1 - cos). None for fewer than two vectors."""
    vs = [v for v in vectors if v]
    if len(vs) < 2:
        return None
    import numpy as np
    m = np.asarray(vs, float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.where(norms == 0, 1.0, norms)
    sims = m @ m.T
    iu = np.triu_indices(len(vs), 1)
    return round(float(np.mean(1.0 - sims[iu])), 4)


def format_spread(cards: list[dict]) -> float | None:
    if len(cards) < 2:
        return None
    counts = Counter(c.get("type") or "?" for c in cards)
    n = len(cards)
    h = -sum(k / n * math.log(k / n) for k in counts.values())
    return round(h / math.log(n), 4)


def topic_spread(cards: list[dict]) -> float | None:
    if not cards:
        return None
    return round(len({c.get("matched_topic") or c.get("url") for c in cards}) / len(cards), 4)


def interest_coverage(cards: list[dict], topics: list[str]) -> float | None:
    """Share of the session's own interests that at least one pick answers. None when there were no interests."""
    topics = [t for t in topics if t]
    if not topics:
        return None
    got = {c.get("matched_topic") for c in cards}
    return round(sum(1 for t in topics if t in got) / len(topics), 4)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else NAN


def list_metrics(cards: list[dict], rels: list[int] | None = None, pool: Pool | None = None,
                 vectors: list | None = None) -> dict[str, float]:
    """Metrics for ONE ordered, already-truncated list. `rels` are 0/1 labels aligned with `cards` (omit for the
    label-free metrics only). `vectors` are aligned with `cards` (omit to skip ild). Missing values are NaN."""
    pool = pool or Pool(cards)
    out: dict[str, float] = {"format_spread": _nan(format_spread(cards)), "topic_spread": _nan(topic_spread(cards)),
                             "ild": _nan(ild(vectors)) if vectors else NAN}
    nov = [pool.novelty(c) for c in cards]
    unx = [pool.unexpectedness(c) for c in cards]
    out["novelty"] = _mean([v for v in nov if v is not None])
    out["unexpectedness"] = _mean([v for v in unx if v is not None])
    if rels is None:
        return out
    rels = [1 if r else 0 for r in rels]
    out["serendipity"] = _mean([r * u for r, u in zip(rels, unx) if u is not None])
    good_unx = [u for r, u in zip(rels, unx) if r and u is not None]
    out["serendipitous_share"] = _mean([1.0 if u >= HALF else 0.0 for u in good_unx])
    return out


def _nan(v) -> float:
    return NAN if v is None else float(v)


def gini(counts) -> float:
    """Inequality of how often each shown item appears: 0 = every item shown equally often."""
    xs = sorted(c for c in counts if c > 0)
    n = len(xs)
    if n < 2:
        return 0.0
    total = sum(xs)
    return round(sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * total), 4)


def coverage(lists: list[list[dict]]) -> dict[str, float]:
    """Population-level spread over many lists (all the lists shown in the evaluation or in a monitoring window)."""
    urls = [c.get("url") or c.get("title") for lst in lists for c in lst]
    n = len(urls)
    if not n:
        return {"slots": 0, "distinct_items": 0, "unique_share": NAN, "gini": NAN, "top_item_share": NAN,
                "formats_seen": 0}
    counts = Counter(urls)
    return {"slots": n, "distinct_items": len(counts), "unique_share": round(len(counts) / n, 4),
            "gini": gini(counts.values()), "top_item_share": round(max(counts.values()) / n, 4),
            "formats_seen": len({c.get("type") for lst in lists for c in lst})}


def card_text(card: dict) -> str:
    """The text a pick is embedded as for ild: its title plus the catalog summary receipt when it has one."""
    summary = next((r.get("text", "") for r in card.get("receipts") or [] if r.get("kind") == "summary"), "")
    return f"{card.get('title', '')}. {summary}".strip()
