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
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import embeddings as E
import interests as I
from interests import Interest
from memory import ProfileStore, get_store

log = logging.getLogger("knots.history")

MAX_CONVERSATIONS = 8
MAX_AGE_DAYS = 60
MAX_CHARS_PER_CONVERSATION = 3000
HALF_LIFE_DAYS = 14.0
NOW_BOOST = 1.5               # weight multiplier for the newest conversation
RECURRENCE_CONF_BONUS = 0.05  # per extra conversation a thread appears in
MAX_PARALLEL_EXTRACTIONS = 4  # keeps Bedrock throttling out of the picture

# memory recall (see recall_and_extract)
SIM_THRESHOLD = {"titan": 0.35, "hashed": 0.12}   # min cosine similarity to count as "related"
MAX_RECALLED = 6              # past conversations pulled in for one request
PROFILE_RECORDS = 30          # how much history "profile mode" looks at


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
            mediums=list(best.mediums),
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
# entry points
# ---------------------------------------------------------------------------

async def _extract_each(convs: list[tuple[Conversation, float]]):
    """One extraction per conversation (parallel, bounded). Returns
    ([(conversation, age, interests)], extractor_label)."""
    gate = asyncio.Semaphore(MAX_PARALLEL_EXTRACTIONS)

    async def one(conv: Conversation):
        async with gate:
            return await I.extract(conv.turns)

    results = await asyncio.gather(*[one(c) for c, _ in convs])
    used = {u for _, u in results}
    label = next(iter(used)) if len(used) == 1 else "mixed (" + ", ".join(sorted(used)) + ")"
    return [(c, age, found) for (c, age), (found, _) in zip(convs, results)], label


async def extract_history(session_texts: list[str] | None = None,
                          conversations: list[Conversation] | None = None,
                          now: datetime | None = None) -> tuple[list[Interest], str]:
    """Like interests.extract, for one conversation or many. Returns (interests, extractor_used).
    Stateless: nothing is stored. See recall_and_extract for the version with memory."""
    convs = normalize_input(session_texts, conversations, now)
    if not convs:
        return [], "none"
    per, label = await _extract_each(convs)
    return (per[0][2] if len(per) == 1 else merge_interests(per)), label


# ---------------------------------------------------------------------------
# memory: recall by similarity and recency, then remember
# ---------------------------------------------------------------------------

def conversation_key(conv: Conversation) -> str:
    """Stable identity, so sending the same conversation twice is not counted twice."""
    if conv.id:
        return "id:" + conv.id[:100]
    return "h:" + hashlib.sha256("\n".join(conv.turns).encode()).hexdigest()[:32]


async def recall_and_extract(user_id: str,
                             session_texts: list[str] | None = None,
                             conversations: list[Conversation] | None = None,
                             store: ProfileStore | None = None,
                             now: datetime | None = None) -> tuple[list[Interest], str, dict]:
    """Extract from what was passed now, bring in RELEVANT past conversations from memory,
    merge, and remember what is new. Returns (interests, extractor_used, memory_info).

    - Past conversations are chosen by embedding similarity to the newest conversation,
      discounted by age. Only their stored interests are used (no raw text exists).
    - With nothing passed (the person just asks "what should I explore?"), it works from
      the most recent history alone ("profile" mode).
    - Memory trouble never breaks the answer: any store error is logged and the request
      continues without it."""
    now = now or datetime.now(timezone.utc)
    info = {"recalled": 0, "recalled_topics": [], "stored_new": 0, "embedder": "", "profile_size": 0}
    try:
        store = store or get_store()
    except Exception as e:  # noqa: BLE001
        log.warning("memory unavailable (%s: %s)", type(e).__name__, e)
        interests, used = await extract_history(session_texts, conversations, now)
        return interests, used, {**info, "error": "memory unavailable"}

    convs = normalize_input(session_texts, conversations, now)

    if not convs:                                           # profile mode
        try:
            recs = await asyncio.to_thread(store.records, user_id, PROFILE_RECORDS)
        except Exception as e:  # noqa: BLE001
            log.warning("memory read failed (%s: %s)", type(e).__name__, e)
            return [], "none", {**info, "error": "memory unavailable"}
        if not recs:
            return [], "none", info
        per = [(Conversation(turns=[], timestamp=r.timestamp), max(0.0, (now - r.time()).total_seconds() / 86400),
                r.interests) for r in reversed(recs)]
        merged = merge_interests(per)
        return merged, "profile", {**info, "recalled": len(recs), "recalled_topics": [m.topic for m in merged[:5]],
                                   "profile_size": len(recs)}

    per_current, label = await _extract_each(convs)
    newest = convs[-1][0]
    embedded = await asyncio.gather(*[asyncio.to_thread(E.embed_sync, "\n".join(c.turns)) for c, _ in convs])
    qvec, ename = embedded[-1]
    info["embedder"] = ename

    recalled = []
    try:
        keys = {conversation_key(c) for c, _ in convs}
        thr = SIM_THRESHOLD.get(ename, 0.2)
        scored = []
        for r in await asyncio.to_thread(store.records, user_id):
            if r.key in keys or r.embedder != ename:        # vectors from other embedders are not comparable
                continue
            sim = E.cosine(qvec, r.vector)
            if sim < thr:
                continue
            age = max(0.0, (now - r.time()).total_seconds() / 86400)
            scored.append((sim * 0.5 ** (age / HALF_LIFE_DAYS), age, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        recalled = sorted(scored[:MAX_RECALLED], key=lambda x: x[1], reverse=True)    # oldest first
    except Exception as e:  # noqa: BLE001
        log.warning("memory read failed (%s: %s)", type(e).__name__, e)
        info["error"] = "memory unavailable"

    past = [(Conversation(turns=[], timestamp=r.timestamp), age, r.interests) for _, age, r in recalled]
    everything = past + per_current
    interests = per_current[0][2] if len(everything) == 1 else merge_interests(everything)
    info["recalled"] = len(recalled)
    info["recalled_topics"] = list(dict.fromkeys(i.topic for _, _, r in recalled for i in r.interests))[:5]

    try:                                                     # remember what is new
        for (conv, _age, found), (vec, en) in zip(per_current, embedded):
            ts = conv.timestamp or now.isoformat()
            if await asyncio.to_thread(store.save, user_id, conversation_key(conv), ts, found, vec, en):
                info["stored_new"] += 1
        info["profile_size"] = await asyncio.to_thread(store.count, user_id)
    except Exception as e:  # noqa: BLE001
        log.warning("memory write failed (%s: %s)", type(e).__name__, e)
        info["error"] = "memory unavailable"
    return interests, label, info


def profile_summary(user_id: str, store: ProfileStore | None = None, now: datetime | None = None) -> dict:
    """What Knots remembers about a person: derived interests only, never conversation text."""
    now = now or datetime.now(timezone.utc)
    store = store or get_store()
    recs = store.records(user_id, PROFILE_RECORDS)
    if not recs:
        return {"remembered_conversations": 0, "stores": "interests and embeddings, never your words",
                "threads": []}
    per = [(Conversation(turns=[], timestamp=r.timestamp), max(0.0, (now - r.time()).total_seconds() / 86400),
            r.interests) for r in reversed(recs)]
    return {
        "remembered_conversations": store.count(user_id),
        "since": min(r.timestamp for r in recs), "latest": max(r.timestamp for r in recs),
        "stores": "interests and embeddings, never your words",
        "threads": [{"topic": t.topic, "concepts": t.concepts, "underneath": t.implicit_themes,
                     "conversations": t.conversations, "last_seen": t.last_seen}
                    for t in merge_interests(per)],
    }
