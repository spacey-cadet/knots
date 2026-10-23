"""
The note on a pick: one or two sentences that say what it is, how known it is, and why it fits.

Built ONLY from catalog data (description / abstract, rating and vote count, citations, preprint status) and from
the person's own words (the quote a theme rests on). No model writes facts here: an LLM may rephrase a note
(`rephrase=` hook) but the result is rejected unless every number in it still traces to a data field. That is
the whole faithfulness rule, and `check_numbers` is what tests it.
"""
from __future__ import annotations

import re
from typing import Callable

from .catalogs import ContentItem

_SENT = re.compile(r"(?<=[.!?])\s+")
_NUM = re.compile(r"\d[\d,\.]*\d|\d")


def first_sentence(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    s = _SENT.split(text)[0]
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return s if s[-1] in ".!?…" else s + "."


def _n(v: int) -> str:
    return f"{v:,}"


def facts(item: ContentItem) -> list[dict]:
    """Verifiable facts about the item, each with where it came from. Unknown stays absent, never guessed."""
    out = []
    if item.year:
        out.append({"label": "year", "value": str(item.year), "source": "catalog"})
    if item.rating and item.popularity:
        out.append({"label": "rating", "value": f"{item.rating:.1f}/10 from {_n(item.popularity)} ratings", "source": "TMDB"})
    if item.citations is not None:
        out.append({"label": "citations", "value": f"{_n(item.citations)} citations", "source": "OpenAlex"})
    if item.status:
        out.append({"label": "status", "value": item.status, "source": "arXiv"})
    if item.is_survey:
        out.append({"label": "kind", "value": "a survey of the field", "source": "arXiv"})
    return out


def popularity_phrase(item: ContentItem) -> str:
    if item.content_type in ("movie", "series", "doc") and item.rating and item.popularity:
        v = item.popularity
        scale = "widely seen" if v >= 10000 else "well known" if v >= 2000 else "a lesser-known pick" if v < 500 else "moderately known"
        return f"{scale}: rated {item.rating:.1f}/10 by {_n(v)} people"
    if item.content_type == "paper":
        if item.citations is None:
            return "preprint, citation count not available" if item.status == "preprint" else ""
        scale = "highly cited" if item.citations >= 1000 else "well cited" if item.citations >= 100 else "little cited so far"
        tail = ", still a preprint" if item.status == "preprint" else ""
        return f"{scale}: {_n(item.citations)} citations{tail}"
    return ""


def build_note(item: ContentItem, theme: dict | None = None, rephrase: Callable[[str], str] | None = None,
               said: str = "", personal: bool = True) -> str:
    """The faithful fallback note (used when no model-written note is available or one is rejected).

    Order matters: lead with the link to THIS person (their own words and the idea underneath), then what the
    thing is, then how well regarded it is. Everything comes from catalog fields and the person's own quote.

    The lead says the pick fits them ("this goes straight at it"), so it is only written when `personal` is true:
    the caller passes that only for a pick that cleared the full quality floor (fitmodel.cleared_floor). Otherwise the
    note is the plain description and standing, which claims nothing about the person."""
    parts = []
    said = (said or "").removesuffix(" ...").strip()
    quote = (theme or {}).get("evidence") or said
    if personal and theme and theme.get("evidence"):
        parts.append(f"You touched on “{theme['evidence']}”, and this speaks to {theme['theme']}.")
    elif personal and said:
        parts.append(f"You said “{said}”, and this goes straight at it.")
    parts.append(first_sentence(item.abstract or item.blurb))
    pop = popularity_phrase(item)
    if pop:
        parts.append(pop[0].upper() + pop[1:] + ".")
    note = " ".join(p for p in parts if p)
    if rephrase and note:
        try:
            alt = rephrase(note)
        except Exception:  # noqa: BLE001 - a failed rewrite just keeps the faithful original
            alt = ""
        if alt and check_numbers(alt, allowed_text(item, theme) + " " + quote):
            return alt
    return note


def allowed_text(item: ContentItem, theme: dict | None = None) -> str:
    return " ".join(["10", item.abstract, item.blurb, str(item.year), str(item.rating or ""), str(item.popularity or ""),
                     "" if item.citations is None else str(item.citations), _n(item.popularity or 0),
                     _n(item.citations or 0), f"{item.rating:.1f}" if item.rating else "",
                     (theme or {}).get("evidence", "")])


def check_numbers(note: str, allowed: str) -> bool:
    """Every number in the note must appear in the allowed source text (commas ignored)."""
    pool = {x.replace(",", "") for x in _NUM.findall(allowed)}
    return all(x.replace(",", "").rstrip(".") in pool for x in _NUM.findall(note))
