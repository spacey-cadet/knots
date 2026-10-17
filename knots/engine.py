"""
The Knots recommendation algorithm, behind one door.

    from knots import engine
    plan = await engine.plan(session_texts=[...], user_id="alex")

Everything between "what the person said" and "what to watch, read or explore" lives in this
package. The MCP server (server.py) and the evaluation harness (eval/run_eval.py) are thin
callers of the functions below and contain no recommendation logic of their own.

The pipeline, and where each step lives:

    input scope        context.py     normalise one or many conversations, merge into threads
    memory             context.py     recall related past conversations (memory.py, embeddings.py)
    understanding      extraction.py  topics, concepts, unstated themes, best mediums (Bedrock)
    retrieval          catalogs.py    TMDB films, series and docs; YouTube explainers; arXiv papers
    quality gate       quality.py     what is good enough to recommend at all
    ranking            ranking.py     relevance first; reserved slots; grounded reasons
    presentation       plan.py        watch / read / explore, spoken line, follow-up question

Public functions (all async except where noted):

    run(...)          the whole pipeline, returning every intermediate result (used by the eval)
    plan(...)         "what are we doing today?": threads plus grouped, reasoned picks
    recommend(...)    ranked cards
    extract(...)      only the understanding step
    search(topics)    only retrieval, for plain topic strings
    what_remembered(user_id), forget(user_id)    the person's control over their memory
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .catalogs import ContentItem, search_content
from .context import (Conversation, extract_history, profile_summary, recall_and_extract)
from .extraction import Interest
from .memory import get_store
from .plan import build_session_plan
from .ranking import rank_and_render

NOTHING_YET = "I don't have anything to go on yet. What have you been curious about lately?"


def clean_user_id(user_id: str | None) -> str | None:
    """A usable id, or None (= stateless: nothing is read or stored)."""
    uid = (user_id or "").strip()
    return uid[:128] if uid else None


@dataclass
class Run:
    """Everything the pipeline produced, so callers can show or score any stage."""
    interests: list[Interest]
    extractor: str
    items: list[ContentItem] = field(default_factory=list)
    memory: dict | None = None
    query: str = ""          # what the person said (newest kept): ranking compares candidates to it

    def plan(self) -> dict:
        """The thinking-partner view. Always has `threads`, `picks`, `spoken` and `ask`."""
        if not self.interests:
            out = {"extractor": self.extractor, "threads": [],
                   "picks": {"watch": [], "read": [], "explore": []},
                   "spoken": NOTHING_YET, "ask": "What have you been curious about lately?"}
        else:
            out = build_session_plan(self.interests, self.items, extractor=self.extractor, query=self.query)
        if self.memory is not None:
            out["memory"] = self.memory
        return out

    def cards(self, limit: int = 6) -> dict:
        """The ranked-cards view."""
        if not self.interests:
            return {"extractor": self.extractor, "interests": [], "cards": [],
                    "note": "No topics detected in the conversation text."}
        out = {"extractor": self.extractor, "interests": [i.to_dict() for i in self.interests],
               "cards": rank_and_render(self.items, self.interests, limit=limit, query=self.query)}
        if self.memory is not None:
            out["memory"] = self.memory
        return out


async def _understand(session_texts, conversations, user_id):
    """(interests, extractor_used, memory_info_or_None)."""
    uid = clean_user_id(user_id)
    convs = list(conversations or [])
    if uid:
        return await recall_and_extract(uid, session_texts, convs)
    interests, used = await extract_history(session_texts, convs)
    return interests, used, None


QUERY_CHARS = 4000


def query_text(session_texts: list[str] | None, conversations: list[Conversation] | None) -> str:
    """The person's own words, oldest first, keeping the NEWEST if it must be cut."""
    turns = [t for c in (conversations or []) for t in c.turns] + list(session_texts or [])
    return "\n".join(t.strip() for t in turns if t and t.strip())[-QUERY_CHARS:]


async def run(session_texts: list[str] | None = None,
              conversations: list[Conversation] | None = None,
              user_id: str | None = None,
              include_videos: bool | None = None,
              search: bool = True,
              challenge: bool | None = None) -> Run:
    """The whole pipeline. `include_videos=None` defers to the INCLUDE_VIDEOS env flag.
    `search=False` stops after understanding (no catalog calls)."""
    interests, used, memory = await _understand(session_texts, conversations, user_id)
    items = (await search_content(interests, include_videos=include_videos, challenge=challenge)
             if (interests and search) else [])
    return Run(interests=interests, extractor=used, items=items, memory=memory,
               query=query_text(session_texts, conversations))


async def plan(session_texts: list[str] | None = None,
               conversations: list[Conversation] | None = None,
               user_id: str | None = None,
               include_videos: bool = True,
               challenge: bool | None = None) -> dict:
    return (await run(session_texts, conversations, user_id, include_videos, challenge=challenge)).plan()


async def recommend(session_texts: list[str] | None = None,
                    conversations: list[Conversation] | None = None,
                    user_id: str | None = None,
                    limit: int = 6,
                    challenge: bool | None = None) -> dict:
    return (await run(session_texts, conversations, user_id, challenge=challenge)).cards(limit)


async def extract(session_texts: list[str] | None = None,
                  conversations: list[Conversation] | None = None,
                  user_id: str | None = None) -> dict:
    r = await run(session_texts, conversations, user_id, search=False)
    out = {"extractor": r.extractor, "interests": [i.to_dict() for i in r.interests]}
    if r.memory is not None:
        out["memory"] = r.memory
    return out


async def search(topics: list[str], include_videos: bool | None = None) -> list[dict]:
    """Retrieval only, for plain topic strings. Deep interests, so every medium is searched."""
    interests = [Interest(topic=t, depth="deep", sentiment="curious", mentions=1,
                          concepts=[t], keywords=[t]) for t in topics if t.strip()]
    return [i.to_dict() for i in await search_content(interests, include_videos=include_videos)]


async def what_remembered(user_id: str) -> dict:
    uid = clean_user_id(user_id)
    if not uid:
        return {"remembered_conversations": 0, "threads": [], "note": "No user_id given."}
    return await asyncio.to_thread(profile_summary, uid)


async def forget(user_id: str) -> dict:
    uid = clean_user_id(user_id)
    if not uid:
        return {"deleted": 0, "note": "No user_id given."}
    return {"deleted": await asyncio.to_thread(get_store().forget, uid)}
