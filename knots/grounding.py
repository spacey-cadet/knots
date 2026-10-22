"""
Grounding: how much independent, checkable evidence backs the link between a pick and what the person said.

It replaces "creativity", which paid for DISTANCE from the person's words and so rewarded far-fetched picks. Grounding
pays for support and never for distance: a pick that is near or far scores the same, so what moves it is how many
independent kinds of evidence stand behind the connection.

Evidence types (a pick earns each at most once):

  keyword        a TMDB keyword matched the session's topic or concept. Films, series and documentaries only:
                 for a video the stored "match" is just the concept it was searched for, and a paper stores none,
                 so neither is evidence that the item itself carries the idea.
  theme_quote    the interest the pick answers has a theme the judge kept AND that rests on a verbatim quote of
                 the person (the quote check runs before the judge, see knots/brain/themes.py), and the card states
                 that theme (films, series, documentaries). This separates interests, not picks within one
                 interest, so on its own it cannot rank two picks for the same topic.
  overview_sim   the pick's overview or description is at least SIM_MIN similar to the session text (cosine). The
                 value depends on the embedder, so it is an untuned default (0.30), NOT fitted to any labels, and the
                 eval reports a sensitivity check around it.
  co_mention     reserved for outside co-mention (Reddit). No source exists yet, so it never fires.

  Grounding(d)   = good(d) * supported(d) * min(E(d), 3) / 3      E(d) = number of evidence types
  Grounding@k    = mean of Grounding over the top k picks
  overreach rate = share of shown picks labelled not supported. A gate, not a metric to maximise: a change that
                   raises diversity but also raises overreach is rejected (see `retrain.gate`).

With co_mention absent, a pick needs all three live types to reach the cap, so today the metric is deliberately
hard to max out. Every number depends on what the stored data can show; the eval says how many picks it could link.
"""
from __future__ import annotations

import os

TYPES = ("keyword", "theme_quote", "overview_sim", "co_mention")
CAP = 3
FILM_TYPES = ("movie", "series", "doc")


def sim_min() -> float:
    try:
        return float(os.environ.get("KNOTS_GROUNDING_SIM_MIN", "0.30"))
    except ValueError:
        return 0.30


def quoted_theme(interest) -> bool:
    """True if the interest has a kept theme that rests on a non-empty verbatim quote (stored by the theme step)."""
    quotes = getattr(interest, "theme_evidence", None) or {}
    return any((quotes.get(t) or "").strip() for t in (getattr(interest, "implicit_themes", None) or []))


def evidence_types(content_type: str, matched_keywords, sim, has_quoted_theme: bool,
                   threshold: float | None = None) -> list[str]:
    """The evidence types that fired for one pick, in a fixed order. Missing facts never fire a type."""
    threshold = sim_min() if threshold is None else threshold
    out = []
    film = content_type in FILM_TYPES
    if film and matched_keywords:
        out.append("keyword")
    if film and has_quoted_theme:
        out.append("theme_quote")
    if isinstance(sim, (int, float)) and not isinstance(sim, bool) and sim == sim and sim >= threshold:
        out.append("overview_sim")
    return out


def card_types(card: dict, quoted: bool | None = None, threshold: float | None = None) -> list[str]:
    """Evidence types for a stored card, recomputed from its own fields so a different similarity threshold can be
    tried on old results. `quoted` overrides the stored theme_quote flag (the eval supplies it when it can link
    a pick to a human-supported theme and the card predates stored types)."""
    stored = card.get("evidence_types")
    has_quote = (quoted if quoted is not None else ("theme_quote" in stored if stored is not None else False))
    return evidence_types(card.get("type", ""), card.get("evidence"), card.get("sim"), has_quote, threshold)


def score(n_types: int) -> float:
    """min(E, 3) / 3: one kind of evidence is a third, three kinds is the maximum, distance earns nothing."""
    return min(max(n_types, 0), CAP) / CAP


def pick_grounding(good: int, supported: int, n_types: int) -> float:
    return good * supported * score(n_types)
