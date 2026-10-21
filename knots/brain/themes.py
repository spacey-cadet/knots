"""
Theme curation: at most N themes for the WHOLE conversation, each one resting on a quote.

The model proposes themes with an `evidence` quote. Three filters follow, cheapest first:

  1. quote check   the quote must really appear in the person's text (case, punctuation and spacing
                   ignored; "..." may skip words). A theme with no quote, or a quote that is not there, is
                   dropped. This is mechanical: it cannot be talked round, and it removes the commonest
                   overreach (a feeling or term the person never used has no sentence to point at).
  2. judge         the learned judge (brain.assess) drops themes it expects the author would reject.
  3. cap + rank    what is left is ranked by judge score (calibrated probability when a calibration is
                   shipped, else the raw score; ties keep the model's order) and cut to N for the session.

Dropped themes are kept on the interest with a reason, for audit; they are never shown or labelled.
"""
from __future__ import annotations

import os
import re

from . import judge

DEFAULT_MAX_THEMES = 5
MIN_QUOTE_WORDS = 3
_ELLIPSIS = re.compile(r"\.{2,}|…")


def max_themes() -> int:
    try:
        return max(1, int(os.environ.get("KNOTS_MAX_THEMES", DEFAULT_MAX_THEMES)))
    except ValueError:
        return DEFAULT_MAX_THEMES


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_supported(quote: str, text: str) -> bool:
    """True when `quote` is a real excerpt of `text` and long enough to mean something."""
    if not isinstance(quote, str) or not quote.strip():
        return False
    parts = [_norm(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if p]
    if not parts or sum(len(p.split()) for p in parts) < MIN_QUOTE_WORDS:
        return False
    hay, pos = _norm(text), 0
    for p in parts:                                  # segments must appear, in order
        i = hay.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def _score(text: str, theme: str, conf: dict) -> float:
    if theme in conf:
        return conf[theme]
    raw = judge.raw_score(text, theme)
    return raw if raw is not None else 0.5


def curate(interests: list, text: str, cap: int | None = None) -> list:
    """Applies the three filters in place and returns the interests. Sets on each interest:
    implicit_themes (kept, best first), dropped_themes, theme_dropped_why, theme_confidence, theme_rank."""
    cap = cap or max_themes()
    pool: list[tuple[float, int, object, str]] = []
    seen: set[str] = set()
    order = 0
    for it in interests:
        proposed, it.implicit_themes = list(it.implicit_themes), []
        it.dropped_themes = list(it.dropped_themes)
        why = dict(it.theme_dropped_why)
        survivors = []
        for t in proposed:
            if t in seen:
                continue                                     # the same theme under two interests: first one wins
            seen.add(t)
            q = it.theme_evidence.get(t, "")
            if not quote_supported(q, text):
                it.dropped_themes.append(t)
                why[t] = "no quote from the conversation" if not q else "quote not found in the conversation"
                continue
            survivors.append(t)
        kept, dropped, conf = judge.assess(text, survivors)
        for t in dropped:
            it.dropped_themes.append(t)
            why[t] = "the judge expects this would be rejected"
        it.theme_confidence = conf
        it.theme_dropped_why = why
        for t in kept:
            pool.append((_score(text, t, conf), order, it, t))
            order += 1
    pool.sort(key=lambda x: (-x[0], x[1]))
    for rank, (_, _, it, t) in enumerate(pool):
        if rank < cap:
            it.implicit_themes.append(t)
            it.theme_rank[t] = rank + 1
        else:
            it.dropped_themes.append(t)
            it.theme_dropped_why[t] = f"over the limit of {cap} themes for the conversation"
            it.theme_confidence.pop(t, None)
    return interests


def ranked(interests: list, limit: int | None = None) -> list[dict]:
    """The session's themes, best first: [{theme, topic, evidence, confidence, rank}]. Themes from the current
    conversation come before recalled ones; within each group the stored rank decides."""
    rows = []
    for idx, it in enumerate(interests):
        for t in it.implicit_themes:
            rows.append((0 if getattr(it, "in_current", True) else 1, it.theme_rank.get(t, 99), idx, {
                "theme": t, "topic": it.topic, "evidence": it.theme_evidence.get(t, ""),
                "confidence": it.theme_confidence.get(t)}))
    rows.sort(key=lambda r: r[:3])
    out = []
    for n, (_, _, _, row) in enumerate(rows, 1):
        out.append({**row, "rank": n})
    return out[:limit] if limit else out
