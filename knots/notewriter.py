"""
The model-written note: the line that makes the case. On each pick it says why THIS person would love it,
written from the catalog's own facts and the person's own words. (The evidence behind the pick is a separate,
internal ledger; the note is the only thing a person is asked to be persuaded by.)

`notes.build_note` is the faithful, templated note. This module lets a model (Claude on Bedrock, the
same account as the extractor) rewrite it with some voice. One call covers every pick in a response, so it
adds one round trip, not one per pick, and it never blocks a response: a timeout, an error or a bad note
quietly keeps the templated one.

Faithfulness is the rule, not the hope. The model is handed only what the catalogs said (description,
rating, citations, year) and the person's own quote. Its note is then REJECTED, entry by entry, unless
  - every number in it appears in that data (notes.check_numbers),
  - every capitalised name in it appears in that data, the title, the topic or the quote (`check_names`),
  - anything it puts in quotation marks is a verbatim piece of the person's own words (the quote a theme rests
    on, or the sentence they said that matches the topic),
  - it carries no URL and stays short.
A rejected note falls back to the templated one. Nothing here researches beyond what the catalogs returned;
that is deliberate, because a model asked to "research" a title from memory is the thing that invents a cast.

Env: NOTES_LLM=off disables it (the eval does); NOTE_MODEL_ID picks the model (default BEDROCK_MODEL_ID);
NOTE_TIMEOUT_S bounds the call (default 8, so an Alexa turn is not blown).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

from .notes import allowed_text, check_numbers, facts

log = logging.getLogger("knots.notewriter")

MAX_NOTE, MAX_SPOKEN = 330, 280

_SYSTEM = """You write the short note under each recommendation in a "what should I watch or read" assistant. \
The note is where the case gets made: its whole job is to make this ONE person want to press play, and to \
believe the pick was chosen for them. Write like a well-read friend who has just thought of the perfect thing: \
specific, warm, a little playful, never salesy.

Hard rules (a note that breaks one is thrown away):
- Use ONLY the facts, description and words given for that item. Never add plot points, characters, cast, \
directors, awards, dates, rankings or numbers that are not given. If the description is thin, say less, but \
still say why it fits them.
- No spoilers. No URLs.
- Copy any number exactly as given. Do not round it or convert it.
- If you quote the person, use their exact words (`their_words`), once, and only if it is natural.

Craft:
- `note`: one or two sentences, at most 45 words. Start from THEM, not from the item: the question they are \
sitting with, the idea underneath (`their_theme`, `about`), or their own words. Then give the one concrete \
thing the item will do for that: make it felt, make it clear, test it, complicate it. Close on a reason to \
start it now. Do not summarise the plot or list facts: the facts are already shown separately.
- `spoken`: what the assistant SAYS aloud, so write it for the ear, not the page: one or two short sentences, \
at most 40 words in all, no quotation marks, no digits, no titles in quotes. The voice has already introduced \
the pick by title, kind and year ("Let's start with ..., a film from ..." or "And if you want the research \
behind it, there's ..., a paper."), so never repeat the title, the kind or the year: pick up right after that \
introduction. Say why this one is for THEM first, then the one thing it will do for them.
- Never explain how you chose it (no "matched on", "based on your", "the algorithm", "evidence").
- Vary your openings across items. Do not start with "If you", "Since you", "Because you" or the title. \
Avoid clichés such as rollercoaster, gripping, must-watch, deep dive, rabbit hole, journey.
- Match the medium: a film is something to sink into, an explainer to learn from, a paper to dig into.

Voice (for `spoken` only):
- Talk like a person across a table: contractions, plain words, and a little wit when it comes naturally. At \
most one light touch per line. Never forced, never an exclamation mark, no filler such as "great pick".
- The lines of one response should sound like one conversation: a later line may pick up the thread of an \
earlier one ("same idea, other angle"), but never name another item, and never add a fact you were not given."""

_TOOL = {"toolSpec": {
    "name": "record_notes",
    "description": "Return the note and spoken line for each item.",
    "inputSchema": {"json": {"type": "object", "required": ["notes"], "properties": {"notes": {
        "type": "array", "items": {"type": "object", "required": ["id", "note", "spoken"], "properties": {
            "id": {"type": "integer"}, "note": {"type": "string"}, "spoken": {"type": "string"}}}}}}}}}

_cache: dict[str, dict[int, tuple[str, str]]] = {}
_client = None


def model_id() -> str:
    return (os.environ.get("NOTE_MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID") or "").strip()


def enabled() -> bool:
    return (os.environ.get("NOTES_LLM", "on").lower() != "off" and bool(model_id())
            and os.environ.get("KNOTS_OFFLINE") != "1")


def _bedrock():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
            config=Config(retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=4,
                          read_timeout=int(os.environ.get("NOTE_TIMEOUT_S", "8"))))
    return _client


_QUOTED = re.compile(r"[“\"]([^”\"]{2,})[”\"]")
_CAPS = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z’'\-]{2,}\b")
_WORD = re.compile(r"[a-z0-9’'\-]+")


def check_names(note: str, allowed: str) -> bool:
    """Every capitalised word that is not opening a sentence must occur in the allowed text (case-insensitive)."""
    pool = set(_WORD.findall(allowed.lower()))
    return all(w.lower() in pool for w in _CAPS.findall(note))


def check_quotes(note: str, quote: str) -> bool:
    """Whatever the note puts in quotation marks must be a verbatim piece of the person's own words."""
    q = re.sub(r"\s+", " ", (quote or "").lower())
    return all(re.sub(r"\s+", " ", m.lower()).strip(" .,") in q for m in _QUOTED.findall(note))


def valid(text: str, limit: int, item, theme: dict | None, *, spoken: bool = False, said: str = "") -> bool:
    if not isinstance(text, str) or not (20 <= len(text.strip()) <= limit) or re.search(r"https?:|www\.", text):
        return False
    quote = " ".join(x for x in ((theme or {}).get("evidence", ""), said or "") if x)
    allowed = " ".join([allowed_text(item, theme), item.title, item.topic, (theme or {}).get("theme", ""), quote])
    if spoken and (re.search(r"\d", text) or _QUOTED.search(text)):
        return False
    return check_numbers(text, allowed) and check_names(text, allowed) and check_quotes(text, quote)


def _their_words(entry: dict) -> str:
    """The person's own words to build the note around: the quote an unstated theme rests on, else the sentence
    they said that matches the topic (from the internal ledger)."""
    return ((entry.get("theme") or {}).get("evidence") or entry.get("said") or "").strip()


def _payload(entries: list[dict]) -> list[dict]:
    out = []
    for k, e in enumerate(entries):
        it, theme = e["item"], e.get("theme") or {}
        out.append({"id": k, "title": it.title, "type": it.content_type, "year": it.year or None,
                    "description": re.sub(r"\s+", " ", it.abstract or it.blurb or "").strip()[:500],
                    "facts": [f"{f['label']}: {f['value']}" for f in facts(it)],
                    "about": e.get("concept") or it.topic,
                    "their_theme": theme.get("theme", ""), "their_words": _their_words(e)})
    return out


def write_batch(entries: list[dict]) -> list[tuple[str, str] | None]:
    """entries: [{"item": ContentItem, "theme": {...}|None, "concept": str}]. Returns, per entry, (note, spoken)
    or None to keep the templated note. Never raises."""
    if not entries or not enabled():
        return [None] * len(entries)
    try:
        payload = _payload(entries)
        key = hashlib.sha256((model_id() + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        got = _cache.get(key)
        if got is None:
            resp = _bedrock().converse(
                modelId=model_id(), system=[{"text": _SYSTEM}],
                messages=[{"role": "user", "content": [{"text": "<items>\n" + json.dumps(payload, ensure_ascii=False) + "\n</items>"}]}],
                toolConfig={"tools": [_TOOL], "toolChoice": {"tool": {"name": "record_notes"}}},
                inferenceConfig={"maxTokens": 1200, "temperature": 0.8})
            got = {}
            for b in resp.get("output", {}).get("message", {}).get("content", []):
                tu = b.get("toolUse")
                if tu and tu.get("name") == "record_notes":
                    for n in tu["input"].get("notes", []):
                        got[int(n["id"])] = (str(n.get("note", "")).strip(), str(n.get("spoken", "")).strip())
            if len(_cache) > 128:
                _cache.clear()
            _cache[key] = got
        out: list[tuple[str, str] | None] = []
        for k, e in enumerate(entries):
            note, spoken = got.get(k, ("", ""))
            ok_note = valid(note, MAX_NOTE, e["item"], e.get("theme"), said=e.get("said", ""))
            ok_spoken = valid(spoken, MAX_SPOKEN, e["item"], None, spoken=True)
            out.append((note, spoken if ok_spoken else "") if ok_note else None)
            if not ok_note:
                log.info("note for %r rejected; keeping the templated one", e["item"].title)
        return out
    except Exception as ex:  # noqa: BLE001 - any failure keeps the faithful templated notes
        log.warning("note writer failed (%s: %s); using templated notes", type(ex).__name__, ex)
        return [None] * len(entries)
