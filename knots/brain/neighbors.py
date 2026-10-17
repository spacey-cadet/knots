"""
Nearest-labelled-theme judge (few-shot, no hand-written word lists).

A new theme is scored by the labelled themes it most resembles: character n-grams of the theme
phrase, TF-IDF weighted, cosine similarity, similarity-weighted vote of the k nearest
(+1 accepted, -1 rejected). The only "training" is storing the labelled examples, so every new
label immediately changes the judgement, and it can be evaluated honestly by leaving a whole
session out (`exclude_group`).

Pure Python on purpose: it runs at serving time with no numpy and no model call.
"""
from __future__ import annotations

import math
import re
from collections import Counter

K = 5
_NGRAMS = (3, 4, 5)


def _grams(text: str) -> Counter:
    c: Counter = Counter()
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        padded = f" {w} "
        for n in _NGRAMS:
            for i in range(len(padded) - n + 1):
                c[padded[i:i + n]] += 1
    return c


class Neighbors:
    def __init__(self, themes: list[str], accepted: list[int], groups: list[str] | None = None):
        self.themes, self.accepted = list(themes), list(accepted)
        self.groups = list(groups) if groups else [""] * len(themes)
        counts = [_grams(t) for t in self.themes]
        df: Counter = Counter()
        for c in counts:
            df.update(c.keys())
        n = max(1, len(counts))
        self.idf = {g: math.log((1 + n) / (1 + d)) + 1.0 for g, d in df.items()}
        self.vecs = [self._vec(c) for c in counts]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {g: (1.0 + math.log(c)) * self.idf[g] for g, c in counts.items() if g in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def score(self, theme: str, exclude_group: str | None = None, k: int = K) -> float | None:
        """Weighted vote in [-1, 1] (1 = its neighbours were all accepted), or None with no examples."""
        q = self._vec(_grams(theme))
        sims = []
        for v, y, g in zip(self.vecs, self.accepted, self.groups):
            if exclude_group is not None and g == exclude_group:
                continue
            a, b = (q, v) if len(q) <= len(v) else (v, q)
            sims.append((sum(x * b.get(t, 0.0) for t, x in a.items()), y))
        if not sims:
            return None
        top = sorted(sims, key=lambda s: -s[0])[:k]
        w = sum(s + 1e-6 for s, _ in top)
        return sum((s + 1e-6) * (2 * y - 1) for s, y in top) / w
