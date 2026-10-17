"""
Receipts: every reason Knots gives points at evidence it actually holds.

  you_said   a sentence the person said that shares terms with the interest (never paraphrased)
  tags       the catalog tags that matched the item
  summary    the first sentence(s) of the item's own overview or abstract
  status     papers: "Preprint, not peer reviewed" or the journal reference arXiv lists
  citations  papers: OpenAlex citation count, when found

A card with no `tags` and no `summary` has nothing to stand on, so it is not shown. `you_said` is
optional evidence: an unstated theme may be legitimate even when the person never used its words.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {"the", "and", "for", "that", "this", "with", "what", "why", "how", "are", "can", "does", "about",
         "from", "have", "you", "your", "tell", "more", "its", "into", "like", "really", "just"}


def _stems(text: str) -> set[str]:
    return {w[:5] for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


def your_words(query: str, terms: list[str], max_words: int = 25) -> str:
    """The sentence of `query` sharing the most term stems with `terms`; '' if none shares any."""
    want = set().union(*[_stems(t) for t in terms]) if terms else set()
    if not want or not query.strip():
        return ""
    best, best_n = "", 0
    for sent in re.split(r"(?<=[.?!])\s+|\n+", query):
        n = len(want & _stems(sent))
        if n > best_n:
            best, best_n = sent.strip(), n
    words = best.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "") if best_n else ""


def _first_sentences(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > 60 else cut.rsplit(" ", 1)[0] + " ..."


def build(item, interest, query: str = "") -> list[dict]:
    r: list[dict] = []
    said = your_words(query, [interest.topic, *interest.concepts, *interest.keywords]) if interest else ""
    if said:
        r.append({"kind": "you_said", "text": said})
    if item.matched_keywords:
        r.append({"kind": "tags", "text": ", ".join(item.matched_keywords[:5])})
    summary = _first_sentences(item.abstract or item.blurb)
    if summary:
        r.append({"kind": "summary", "text": summary})
    if item.content_type == "paper":
        if item.status == "published" and item.venue:
            r.append({"kind": "status", "text": f"Published: {item.venue}"})
        elif item.status:
            r.append({"kind": "status", "text": "Preprint, not peer reviewed"})
        if item.citations is not None:
            r.append({"kind": "citations", "text": f"{item.citations:,} citations (OpenAlex)"})
    return r


def sourced(receipts: list[dict]) -> bool:
    """True if the card has something to stand on besides the person's own words."""
    return any(x["kind"] in ("tags", "summary") for x in receipts)
