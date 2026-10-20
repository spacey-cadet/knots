"""
The response contract: one shape every surface (Alexa skill, MCP tool, screen) renders from.

    {
      "version": 1,
      "status": "ok" | "relaxed" | "empty",   # relaxed: fewer checks passed than usual, say so
      "say": "...",                           # what a voice reads: <= 2 picks, no URLs
      "picks": [{title, type, year, note, why, facts[{label,value,source}], confidence, url, matched_topic}],
      "themes": [{theme, evidence, rank}],    # <= 5, each with the person's words
      "ask": "...",                           # one follow-up question
      "mode": {"learn_vs_play": 0..1, "label": "learn" | "play" | "mixed"}   # when a profile supplied it
    }

At least one pick whenever the catalogs returned anything at all. If the normal ranking (which demands receipts)
yields none, the relaxation ladder widens step by step and `status` and `confidence` say so honestly. If the
catalogs returned nothing there is nothing true to recommend: status is "empty", picks is [] and `say` asks a
question instead of inventing a title.
"""
from __future__ import annotations

from .catalogs import ContentItem
from .notes import build_note, facts
from .ranking import KIND, render_card

VERSION = 1
MAX_PICKS = 5
SPOKEN_PICKS = 2
_LABELS = ("strong match", "possible match", "stretch")


def _flatten(plan: dict) -> list[dict]:
    p = plan["picks"]
    return [c for k in ("watch", "read", "explore") for c in p.get(k, [])]


def _confidence(card: dict, level: int) -> str:
    if level >= 2:
        return "stretch"
    if level == 1:
        return "possible match"
    rel = card.get("relevance") or 0
    return "strong match" if rel >= 0.5 else "possible match"


def _relaxed(items: list[ContentItem], interests: list, n: int) -> list[dict]:
    by_topic = {i.topic: i for i in interests}
    ranked = sorted(items, key=lambda it: (-it.topic_overlap_score, -(it.rating or 0)))
    out, seen = [], set()
    for it in ranked:
        if it.source_url in seen:
            continue
        seen.add(it.source_url)
        out.append(render_card(it, by_topic.get(it.topic) or interests[0]))
        if len(out) >= n:
            break
    return out


def build_response(plan: dict, items: list[ContentItem], interests: list, mode: dict | None = None,
                   rephrase=None) -> dict:
    by_url = {it.source_url: it for it in items}
    themes = [{"theme": t["theme"], "evidence": t["evidence"], "rank": t["rank"]} for t in plan.get("themes", [])]
    cards, level = _flatten(plan)[:MAX_PICKS], 0
    if not cards and items and interests:
        cards, level = _relaxed(items, interests, 3), 1
        if all((c.get("relevance") or 0) < 0.2 for c in cards):
            level = 2
    picks = []
    for c in cards:
        it = by_url.get(c["url"])
        theme = next((t for t in plan.get("themes", []) if t["topic"] == c.get("matched_topic")), None)
        picks.append({
            "title": c["title"], "type": c["type"], "year": c.get("year", ""), "url": c["url"],
            "matched_topic": c.get("matched_topic", ""),
            "note": build_note(it, theme, rephrase) if it else c.get("reasoning", ""),
            "why": c.get("reasoning", ""),
            "facts": facts(it) if it else [],
            "confidence": _confidence(c, level),
        })
    status = "empty" if not picks else "relaxed" if level else "ok"
    resp = {"version": VERSION, "status": status, "say": "", "picks": picks, "themes": themes[:5],
            "ask": plan.get("ask", ""), "extractor": plan.get("extractor", "")}
    if mode:
        resp["mode"] = mode
    resp["say"] = _say(resp, plan)
    return resp


def _say(resp: dict, plan: dict) -> str:
    if resp["status"] == "empty":
        return ("I couldn't find anything I'd stand behind right now, so I won't guess. "
                + (resp["ask"] or "What would you like to dig into?"))
    lead = ""
    if resp["themes"]:
        lead = f"Underneath it, I hear {resp['themes'][0]['theme']}. "
    bits = []
    for n, p in enumerate(resp["picks"][:SPOKEN_PICKS]):
        kind = {"movie": "film", "series": "series", "doc": "documentary", "video": "video", "paper": "paper"}.get(p["type"], p["type"])
        yr = f" from {p['year']}" if p["year"] else ""
        bits.append(f"{'Try' if n == 0 else 'Or'} {p['title']}, a {kind}{yr}. {p['note']}")
    hedge = "" if resp["status"] == "ok" else "These are looser matches than usual. "
    return (lead + hedge + " ".join(bits) + (" " + resp["ask"] if resp["ask"] else "")).strip()
