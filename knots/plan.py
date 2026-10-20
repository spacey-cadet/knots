"""
Thinking-partner mode: the "what are we doing today?" experience.

Instead of returning a flat ranked list, this answers like a person you think
with: it says what it thinks you're circling (including what's underneath),
then offers things to watch, read and explore, each with its reason, and ends
with a question that keeps the conversation going.

Output is shaped for both surfaces Alexa+ has:
  - `spoken`  : a short, voice-friendly summary Alexa can read aloud
  - the rest  : structured data for a screen (threads + grouped, reasoned picks)
"""

from __future__ import annotations

from .brain.themes import ranked as ranked_themes
from .catalogs import ContentItem
from .extraction import Interest
from .pairing import build_pairs
from .ranking import rank_and_render

PER_KIND = {"watch": 4, "read": 2, "explore": 2}


def _join(words: list[str]) -> str:
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _threads(interests: list[Interest]) -> list[dict]:
    return [{
        "topic": i.topic,
        "concepts": i.concepts,
        "underneath": i.implicit_themes,   # what the text implies but never says
        "underneath_confidence": i.theme_confidence,   # calibrated P(supported) per theme; {} if uncalibrated
        "underneath_evidence": {t: i.theme_evidence.get(t, "") for t in i.implicit_themes},   # the quote each rests on
        "withheld_themes": len(i.dropped_themes),     # themes the judge held back rather than overreach
        "why_i_think_so": i.rationale,
        "confidence": round(i.confidence, 2),
        "best_medium": i.mediums[0] if i.mediums else "film",
        "conversations": i.conversations,     # how many separate conversations raised it
        "last_seen": i.last_seen,
        "in_current": i.in_current,
    } for i in interests]


def _spoken(interests: list[Interest], picks: dict[str, list[dict]], pairs: list[dict] | None = None,
            themes: list[dict] | None = None) -> str:
    lead = interests[0]
    sentence = f"It sounds like you're circling {_join([i.topic for i in interests[:2]])}."
    if lead.conversations > 1:
        sentence += f" You've come back to {lead.topic} across {lead.conversations} conversations."
    top = (themes[0]["theme"] if themes else (lead.implicit_themes[0] if lead.implicit_themes else ""))
    if top:
        sentence += f" Underneath it, I hear {top}."
    # lead with the format that suits this person's lead thread, then the others
    phrasing = {
        "film": ("watch", "To see it on screen, try {t}."),
        "video": ("explore", "For a clear explainer, try {t}."),
        "paper": ("read", "And if you want the research, {t}."),
    }
    order = [m for m in lead.mediums if m in phrasing] or list(phrasing)
    for n, medium in enumerate(order):
        bucket, line = phrasing[medium]
        if picks[bucket]:
            sentence += " " + (line if n == 0 else line.replace("And if", "If")).format(t=picks[bucket][0]["title"])
    if pairs:
        p = pairs[0]
        sentence += f" For a double feature, watch {p['watch']['title']} and then {p['learn']['title']}."
    return sentence


def _ask(interests: list[Interest], themes: list[dict] | None = None) -> str:
    lead = interests[0]
    top = (themes[0]["theme"] if themes else (lead.implicit_themes[0] if lead.implicit_themes else ""))
    if top:
        return (f"Is it {lead.topic} itself that's pulling you in, "
                f"or the question of {top}?")
    return f"Do you want to go deeper into {lead.topic}, or see where it shows up elsewhere?"


def build_session_plan(interests: list[Interest], items: list[ContentItem],
                       extractor: str = "", query: str = "") -> dict:
    """Group ranked, reasoned cards into watch / read / explore around the
    person's current threads."""
    cards = rank_and_render(items, interests, limit=len(items) or 1, query=query)
    picks: dict[str, list[dict]] = {"watch": [], "read": [], "explore": []}
    seen_urls: set[str] = set()      # the same title can match several topics; show it once
    for c in cards:
        bucket = picks[c["kind"]]
        if c["url"] in seen_urls or len(bucket) >= PER_KIND[c["kind"]]:
            continue
        seen_urls.add(c["url"])
        bucket.append(c)
    note = "" if any(picks.values()) else (
        "No catalog results right now (catalogs unavailable or no keys set), so I'm not guessing.")
    pairs = build_pairs(items, interests, query)
    themes = ranked_themes(interests, 5)
    return {
        "extractor": extractor,
        "threads": _threads(interests),
        "themes": themes,        # at most 5 for the conversation, best first, each with the quote it rests on
        "picks": picks,
        "pairs": pairs,          # watch-and-learn double features, only when both halves are sourced
        "note": note,
        "spoken": _spoken(interests, picks, pairs, themes),
        "ask": _ask(interests, themes),
    }
