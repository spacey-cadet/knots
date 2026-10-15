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

from cards import rank_and_render
from catalogs import ContentItem
from interests import Interest

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
        "why_i_think_so": i.rationale,
        "confidence": round(i.confidence, 2),
    } for i in interests]


def _spoken(interests: list[Interest], picks: dict[str, list[dict]]) -> str:
    lead = interests[0]
    sentence = f"It sounds like you're circling {_join([i.topic for i in interests[:2]])}."
    if lead.implicit_themes:
        sentence += f" Underneath it, I hear {lead.implicit_themes[0]}."
    if picks["watch"]:
        sentence += f" To see it on screen, try {picks['watch'][0]['title']}."
    if picks["read"]:
        sentence += f" And if you want the research, {picks['read'][0]['title']}."
    return sentence


def _ask(interests: list[Interest]) -> str:
    lead = interests[0]
    if lead.implicit_themes:
        return (f"Is it {lead.topic} itself that's pulling you in, "
                f"or the question of {lead.implicit_themes[0]}?")
    return f"Do you want to go deeper into {lead.topic}, or see where it shows up elsewhere?"


def build_session_plan(interests: list[Interest], items: list[ContentItem],
                       extractor: str = "") -> dict:
    """Group ranked, reasoned cards into watch / read / explore around the
    person's current threads."""
    cards = rank_and_render(items, interests, limit=len(items) or 1)
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
    return {
        "extractor": extractor,
        "threads": _threads(interests),
        "picks": picks,
        "note": note,
        "spoken": _spoken(interests, picks),
        "ask": _ask(interests),
    }
