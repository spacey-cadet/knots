"""Adapter: contract response -> Alexa custom-skill response. Limits per Alexa docs: speech 8000 chars, card 8000."""
from __future__ import annotations

from xml.sax.saxutils import escape

SPEECH_LIMIT = 8000
CARD_LIMIT = 8000


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[:end + 1] if end > 0 else cut.rsplit(" ", 1)[0]).strip()


def to_alexa(resp: dict, end_session: bool = False) -> dict:
    wrap = len("<speak></speak>")
    say = _cut(resp["say"], SPEECH_LIMIT)
    while len(escape(say)) + wrap > SPEECH_LIMIT:         # the limit applies to the escaped SSML, not the raw text
        say = _cut(say, int(len(say) * 0.9))
    ssml = f"<speak>{escape(say)}</speak>"
    card_lines = [f"{p['title']} ({p['year']}) - {p['note']}" if p["year"] else f"{p['title']} - {p['note']}"
                  for p in resp["picks"]]
    out = {"version": "1.0", "response": {"outputSpeech": {"type": "SSML", "ssml": ssml},
                                           "shouldEndSession": end_session}}
    if card_lines:
        out["response"]["card"] = {"type": "Simple", "title": "Knots", "content": _cut("\n\n".join(card_lines), CARD_LIMIT)}
    if resp.get("ask") and not end_session:
        out["response"]["reprompt"] = {"outputSpeech": {"type": "PlainText", "text": resp["ask"]}}
    return out
