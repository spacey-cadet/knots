"""
Input scope: one conversation, or many.

What the person is asking right now and what they keep coming back to are
different signals, and Knots uses both:

  - NOW      the newest conversation. Decides what the first pick is about.
  - ENDURING topics that recur across several conversations. A topic raised in
             four separate conversations is a stronger signal than one raised
             four times in one sitting.

How it works
  1. Each conversation is extracted on its own (one small Bedrock call each,
     in parallel, cached by content, so only NEW conversations cost anything).
     Mixing conversations into one prompt blurs unrelated topics together.
  2. The per-conversation interests are merged into threads: two interests are
     the same thread when they share a topic or concept (or most of a topic's
     words).
  3. Each thread is weighted by recency (half-life HALF_LIFE_DAYS) and by how
     many different conversations it appeared in. The newest conversation gets
     a boost, and its strongest thread is always kept in the top two.

Limits (so cost, latency and the prompt-injection surface stay bounded):
  at most MAX_CONVERSATIONS conversations, none older than MAX_AGE_DAYS (the
  newest is always kept), each trimmed to its latest MAX_CHARS_PER_CONVERSATION.

A single conversation (or the old flat `session_texts`) takes the same path as
before: one extraction, no merging.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import interests as I
from interests import Interest

MAX_CONVERSATIONS = 8
MAX_AGE_DAYS = 60
MAX_CHARS_PER_CONVERSATION = 3000
HALF_LIFE_DAYS = 14.0
NOW_BOOST = 1.5               # weight multiplier for the newest conversation
RECURRENCE_CONF_BONUS = 0.05  # per extra conversation a thread appears in
MAX_PARALLEL_EXTRACTIONS = 4  # keeps Bedrock throttling out of the picture


@dataclass
class Conversation:
    turns: list[str]
    timestamp: str | None = None      # ISO 8601; optional
    id: str = ""


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _trim_to_latest(turns: list[str], limit: int) -> list[str]:
    """Keep the most recent turns that fit in `limit` characters (never the oldest)."""
    kept, used = [], 0
    for t in reversed(turns):
        if used + len(t) > limit and kept:
            break
        kept.append(t[-limit:] if len(t) > limit else t)
        used += len(t)
    return list(reversed(kept))


def normalize_input(session_texts: list[str] | None,
                    conversations: list[Conversation] | None,
                    now: datetime | None = None) -> list[tuple[Conversation, float]]:
    """Return [(conversation, age_in_days)] oldest -> newest, trimmed and capped.

    `session_texts` (the original flat input) counts as one conversation. A
    conversation without a timestamp is aged by its position instead: the
    newest is "now", each earlier one a day older."""
    now = now or datetime.now(timezone.utc)
    convs = [c for c in (conversations or []) if any((t or "").strip() for t in c.turns)]
    if session_texts and any((t or "").strip() for t in session_texts):
        convs.append(Conversation(turns=list(session_texts)))
    if not convs:
        return []

    parsed = [_parse_time(c.timestamp) for c in convs]
    if any(parsed):                       # order by time when we have it, else keep the given order
        order = sorted(range(len(convs)), key=lambda i: (parsed[i] or datetime.min.replace(tzinfo=timezone.utc)))
        convs, parsed = [convs[i] for i in order], [parsed[i] for i in order]

    out = []
    for idx, (c, ts) in enumerate(zip(convs, parsed)):
        age = max(0.0, (now - ts).total_seconds() / 86400) if ts else float(len(convs) - 1 - idx)
        turns = _trim_to_latest([t.strip() for t in c.turns if t and t.strip()], MAX_CHARS_PER_CONVERSATION)
        out.append((Conversation(turns=turns, timestamp=c.timestamp, id=c.id), age))

    newest = out[-1]
    out = [x for x in out[:-1] if x[1] <= MAX_AGE_DAYS] + [newest]    # the newest always stays
    return out[-MAX_CONVERSATIONS:]


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _signature(it: Interest) -> set[str]:
    return {it.topic.lower().strip(), *(c.lower().strip() for c in it.concepts)} - {""}


def _same_thread(a: Interest, b: Interest) -> bool:
    if _signature(a) & _signature(b):
        return True
    wa, wb = set(_WORD.findall(a.topic.lower())), set(_WORD.findall(b.topic.lower()))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def _ordered_union(lists: list[list[str]], cap: int) -> list[str]:
    seen, out = set(), []
    for lst in lists:
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out[:cap]


def merge_interests(per_conversation: list[tuple[Conversation, float, list[Interest]]]) -> list[Interest]:
    """per_conversation is oldest -> newest. Returns merged threads, strongest first."""
    newest = len(per_conversation) - 1
    groups: list[dict] = []        # each: {"members": [(conv_idx, age, Interest, weight)]}
    for idx in range(newest, -1, -1):                       # newest first, so groups are named by recent wording
        conv, age, found = per_conversation[idx]
        recency = 0.5 ** (age / HALF_LIFE_DAYS) * (NOW_BOOST if idx == newest else 1.0)
        for it in found:
            w = recency * it.confidence
            home = next((g for g in groups if _same_thread(g["lead"], it)), None)
            if home is None:
                groups.append({"lead": it, "members": [(idx, conv, it, w)]})
            else:
                home["members"].append((idx, conv, it, w))

    merged = []
    for g in groups:
        members = sorted(g["members"], key=lambda m: m[3], reverse=True)
        best = members[0][2]
        conv_ids = {m[0] for m in members}
        n = len(conv_ids)
        newest_member = max(members, key=lambda m: m[0])
        merged.append((sum(m[3] for m in members), newest in conv_ids, Interest(
            topic=best.topic,
            depth="deep" if any(m[2].depth == "deep" for m in members) else "casual",
            sentiment=newest_member[2].sentiment,
            mentions=sum(m[2].mentions for m in members),
            concepts=_ordered_union([m[2].concepts for m in members], 4),
            implicit_themes=_ordered_union([m[2].implicit_themes for m in members], 4),
            keywords=_ordered_union([m[2].keywords for m in members], 8),
            rationale=best.rationale,
            confidence=min(1.0, best.confidence + RECURRENCE_CONF_BONUS * (n - 1)),
            conversations=n,
            last_seen=newest_member[1].timestamp or "",
            in_current=newest in conv_ids,
        )))

    merged.sort(key=lambda m: m[0], reverse=True)
    ranked = [m[2] for m in merged]
    # the newest conversation's strongest thread is always among the first two
    cur = next((i for i, it in enumerate(ranked) if it.in_current), None)
    if cur is not None and cur > 1:
        ranked.insert(1, ranked.pop(cur))
    return ranked[:I.MAX_INTERESTS]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

async def extract_history(session_texts: list[str] | None = None,
                          conversations: list[Conversation] | None = None,
                          now: datetime | None = None) -> tuple[list[Interest], str]:
    """Like interests.extract, for one conversation or many. Returns (interests, extractor_used)."""
    convs = normalize_input(session_texts, conversations, now)
    if not convs:
        return [], "none"
    if len(convs) == 1:
        return await I.extract(convs[0][0].turns)

    gate = asyncio.Semaphore(MAX_PARALLEL_EXTRACTIONS)

    async def one(conv: Conversation):
        async with gate:
            return await I.extract(conv.turns)

    results = await asyncio.gather(*[one(c) for c, _ in convs])
    used = {u for _, u in results}
    label = next(iter(used)) if len(used) == 1 else "mixed (" + ", ".join(sorted(used)) + ")"
    merged = merge_interests([(c, age, found) for (c, age), (found, _) in zip(convs, results)])
    return merged, label
