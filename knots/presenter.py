"""
What the person is shown, as opposed to what Knots holds.

Knots keeps an evidence ledger behind every card (`receipts`: the sentence the person said, the matched
catalog tags, the item's own summary, a paper's preprint status, its fit features). The ledger is how the
system stays honest: a card with nothing to stand on is never built, the grounding metric is computed from it,
and every number, name and quote in a note is checked against the same data.

It is NOT what the person hears or reads. The persuading is done by the note ("why you would love this").
This module is the one place that decides what leaves the server:

  default        an allowlist of fields a person can use: title, kind, year, link, the note, and a few labels.
                 No receipts, no fit features, no templated "matched on" reasoning.
  explain=True   the full cards, ledger included. For "why did you pick that?" and for debugging.

An allowlist, not a denylist, so a field added to a card later stays internal until someone chooses to show it.
The engine, the eval and the feedback log all work on the full cards; only the transport (server.py) calls this.
"""
from __future__ import annotations

# Fields of a card a person may see. `note` / `spoken_note` are filled in from the response contract.
PUBLIC_CARD_KEYS = (
    "title", "type", "kind", "year", "url", "matched_topic", "concept",
    "note", "spoken_note", "confidence", "gem", "blockbuster", "challenge", "status",
)
# `get_recommendations` cards carry the same written note as a `knots_partner` pick (contract.note_cards), so they
# show the same fields. The templated one-line `reasoning` is part of the ledger now: only with explain=True.
PUBLIC_RANKED_KEYS = PUBLIC_CARD_KEYS
# A pick in the response contract: drop `why` (the templated reasoning); keep the note and the sourced facts.
PUBLIC_RESPONSE_PICK_DROP = ("why",)


def _notes_by_url(plan: dict) -> dict[str, dict]:
    return {p["url"]: p for p in (plan.get("response") or {}).get("picks", []) if p.get("url")}


def _public_card(card: dict, notes: dict[str, dict], keys: tuple[str, ...]) -> dict:
    merged = dict(card)
    n = notes.get(card.get("url", ""))
    if n:
        merged["note"] = n.get("note", "")
        merged["spoken_note"] = n.get("spoken_note", "")
        merged["confidence"] = n.get("confidence", merged.get("confidence"))
    return {k: merged[k] for k in keys if merged.get(k) not in (None, "", [], False)}


def public_plan(plan: dict, explain: bool = False) -> dict:
    """The `knots_partner` payload. Returns a new dict; the plan passed in is not changed."""
    if explain:
        return plan
    notes = _notes_by_url(plan)
    out = {k: v for k, v in plan.items() if k not in ("picks", "pairs", "response")}
    out["picks"] = {bucket: [_public_card(c, notes, PUBLIC_CARD_KEYS) for c in cards]
                    for bucket, cards in plan.get("picks", {}).items()}
    out["pairs"] = [{"concept": p.get("concept", ""),
                     "watch": _public_card(p["watch"], notes, PUBLIC_CARD_KEYS),
                     "learn": _public_card(p["learn"], notes, PUBLIC_CARD_KEYS),
                     "bridge": p.get("bridge", "")} for p in plan.get("pairs", [])]
    resp = plan.get("response")
    if resp is not None:
        out["response"] = {**resp, "picks": [{k: v for k, v in p.items() if k not in PUBLIC_RESPONSE_PICK_DROP}
                                              for p in resp.get("picks", [])]}
    return out


def public_cards(result: dict, explain: bool = False) -> dict:
    """The `get_recommendations` payload: ranked cards, evidence ledger removed unless explain=True."""
    if explain or "cards" not in result:
        return result
    return {**result, "cards": [_public_card(c, {}, PUBLIC_RANKED_KEYS) for c in result["cards"]]}
