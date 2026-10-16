"""
Knots — turns what you research into films, papers and videos that make it concrete (MCP server for Alexa+)

Exposes four tools over Streamable HTTP (MCP spec 2025-11-25+):
  - thinking_partner(session_texts)      -> "what are we doing today?": threads,
                                            what's underneath them, and things to
                                            watch / read / explore, each with a reason
  - get_recommendations(session_texts)   -> the full pipeline as ranked cards
  - extract_interests(session_texts)     -> topics, concepts, unstated themes, keywords
  - search_content(topics)               -> candidate titles from the real catalogs

Interest extraction uses Amazon Bedrock when configured (BEDROCK_MODEL_ID) and
falls back to an offline heuristic on any failure, so the server always answers.

Run:
    python server.py
Then point an MCP client (or Alexa+ MCP config) at:
    http://localhost:8000/mcp
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()  # before the imports below read any env-driven config

from cards import rank_and_render  # noqa: E402
from catalogs import search_content as _search_content  # noqa: E402
from history import Conversation, profile_summary, recall_and_extract, extract_history  # noqa: E402
from memory import get_store  # noqa: E402
from interests import Interest  # noqa: E402
from partner import build_session_plan  # noqa: E402

mcp = FastMCP("knots", stateless_http=True)


class ConversationIn(BaseModel):
    """One conversation (or one Reddit session) as the assistant remembers it."""
    turns: list[str] = Field(description="The person's messages in this conversation, oldest first.")
    timestamp: str | None = Field(default=None, description="When it happened, ISO 8601. Optional.")
    id: str | None = Field(default=None, description="Any stable id. Optional.")


def _user(user_id: str | None) -> str | None:
    """A usable id, or None (= stateless: nothing is read or stored)."""
    uid = (user_id or "").strip()
    return uid[:128] if uid else None


async def _interests(session_texts: list[str] | None, conversations: list[ConversationIn] | None,
                     user_id: str | None = None):
    """Returns (interests, extractor_used, memory_info_or_None)."""
    convs = [Conversation(turns=c.turns, timestamp=c.timestamp, id=c.id or "") for c in conversations or []]
    uid = _user(user_id)
    if uid:
        return await recall_and_extract(uid, session_texts, convs)
    interests, used = await extract_history(session_texts, convs)
    return interests, used, None


@mcp.tool()
async def extract_interests(session_texts: list[str] | None = None,
                            conversations: list[ConversationIn] | None = None,
                            user_id: str | None = None) -> dict:
    """
    Extract what the person is really interested in from recent Alexa+
    conversation turns and/or Reddit activity text. For each interest returns
    the surface topic, the abstract concepts at stake, unstated themes the text
    implies, and film-tag-style keywords. Also reports which extractor ran
    ("bedrock" or a heuristic fallback).

    Pass the current conversation as `session_texts`. If you also have earlier
    conversations, pass them all as `conversations` (newest last, timestamps if
    known): topics that recur across conversations are weighted up, and the
    newest conversation still leads.
    """
    interests, used, memory = await _interests(session_texts, conversations, user_id)
    out = {"extractor": used, "interests": [i.to_dict() for i in interests]}
    if memory is not None:
        out["memory"] = memory
    return out


@mcp.tool()
async def search_content(topics: list[str]) -> list[dict]:
    """
    Match plain topic strings against TMDB (films and documentaries), arXiv
    (papers) and optionally YouTube. A missing key or failed call returns nothing
    for that source (mock results only when KNOTS_OFFLINE=1).
    """
    interests = [Interest(topic=t, depth="deep", sentiment="curious", mentions=1,   # deep = all three mediums
                          concepts=[t], keywords=[t]) for t in topics if t.strip()]
    return [i.to_dict() for i in await _search_content(interests)]


@mcp.tool()
async def get_recommendations(session_texts: list[str] | None = None, limit: int = 6,
                              conversations: list[ConversationIn] | None = None,
                              user_id: str | None = None) -> dict:
    """
    Full pipeline: extract interests from session text, match them against
    the content catalogs, rank, and return ready-to-render cards, each with the
    reason it was chosen. Accepts one conversation (`session_texts`) or several
    (`conversations`), as in extract_interests.
    """
    interests, used, memory = await _interests(session_texts, conversations, user_id)
    if not interests:
        return {"extractor": used, "interests": [], "cards": [],
                "note": "No topics detected in the conversation text."}
    items = await _search_content(interests)
    out = {
        "extractor": used,
        "interests": [i.to_dict() for i in interests],
        "cards": rank_and_render(items, interests, limit=limit),
    }
    if memory is not None:
        out["memory"] = memory
    return out


@mcp.tool()
async def thinking_partner(session_texts: list[str] | None = None, include_videos: bool = True,
                           conversations: list[ConversationIn] | None = None,
                           user_id: str | None = None) -> dict:
    """
    Call this when the person asks things like "what are we doing today?",
    "what should I explore?", "what should I watch or read?", or opens a
    session with no specific request. Acts as an intellectual partner: says what
    they seem to be circling (including what's underneath), then suggests films
    and documentaries to watch, papers to read and videos to explore, each with
    its reason. `spoken` is a short line Alexa can read aloud; `ask` is a
    follow-up question to keep the conversation going.

    Pass the current conversation as `session_texts`. If you can also see recent
    earlier conversations (up to about 8, last 60 days), pass them as
    `conversations` so recurring interests count for more.

    Pass `user_id` (a stable id for this person) to give Knots a memory: it keeps
    the interests and an embedding of each conversation (never the words), brings
    back related past conversations, and can answer with no conversation text at
    all ("what should I explore?") from what it remembers.
    """
    interests, used, memory = await _interests(session_texts, conversations, user_id)
    if not interests:
        return {"extractor": used, "threads": [], "picks": {"watch": [], "read": [], "explore": []},
                "spoken": "I don't have anything to go on yet. What have you been curious about lately?",
                "ask": "What have you been curious about lately?",
                **({"memory": memory} if memory is not None else {})}
    items = await _search_content(interests, include_videos=include_videos)
    plan = build_session_plan(interests, items, extractor=used)
    if memory is not None:
        plan["memory"] = memory
    return plan


@mcp.tool()
async def what_do_you_remember(user_id: str) -> dict:
    """
    Call when the person asks what Knots knows or remembers about them. Returns the
    interest threads it holds and how many conversations they come from. Knots keeps
    interests and embeddings only, never conversation text.
    """
    uid = _user(user_id)
    if not uid:
        return {"remembered_conversations": 0, "threads": [], "note": "No user_id given."}
    return await asyncio.to_thread(profile_summary, uid)


@mcp.tool()
async def forget_me(user_id: str) -> dict:
    """
    Call when the person asks Knots to forget them. Permanently deletes everything
    it stores for that user_id and returns how many conversation records were removed.
    """
    uid = _user(user_id)
    if not uid:
        return {"deleted": 0, "note": "No user_id given."}
    return {"deleted": await asyncio.to_thread(get_store().forget, uid)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MCP server on http://0.0.0.0:{port}/mcp (Streamable HTTP)")
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
