"""
Knots — turns what you research into films, series, explainers and papers that make it concrete
(MCP server for Alexa+).

This file is transport only: it declares the MCP tools and hands every call to `knots.engine`,
where the whole recommendation algorithm lives. There is no recommendation logic here.

Tools (Streamable HTTP, MCP spec 2025-11-25+):
  - thinking_partner      "what are we doing today?": threads, what's underneath them, and things
                          to watch / read / explore, each with a reason
  - get_recommendations   the full pipeline as ranked cards
  - extract_interests     topics, concepts, unstated themes, keywords, best mediums
  - search_content        candidate titles from the real catalogs
  - what_do_you_remember  what Knots holds about a person (interests and embeddings, never words)
  - forget_me             delete it

Interest extraction uses Amazon Bedrock when configured (BEDROCK_MODEL_ID) and falls back to an
offline heuristic on any failure, so the server always answers.

Run:
    python server.py
Then point an MCP client (or Alexa+ MCP config) at:
    http://localhost:8000/mcp
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()  # before the import below reads any env-driven config

from knots import Conversation, engine  # noqa: E402

mcp = FastMCP("knots", stateless_http=True)


class ConversationIn(BaseModel):
    """One conversation (or one Reddit session) as the assistant remembers it."""
    turns: list[str] = Field(description="The person's messages in this conversation, oldest first.")
    timestamp: str | None = Field(default=None, description="When it happened, ISO 8601. Optional.")
    id: str | None = Field(default=None, description="Any stable id. Optional.")


def _convs(conversations: list[ConversationIn] | None) -> list[Conversation]:
    return [Conversation(turns=c.turns, timestamp=c.timestamp, id=c.id or "") for c in conversations or []]


@mcp.tool()
async def extract_interests(session_texts: list[str] | None = None,
                            conversations: list[ConversationIn] | None = None,
                            user_id: str | None = None) -> dict:
    """
    Extract what the person is really interested in from recent Alexa+
    conversation turns and/or Reddit activity text. For each interest returns
    the surface topic, the abstract concepts at stake, unstated themes the text
    implies, film-tag-style keywords and which formats (film, video, paper) suit it
    best. Also reports which extractor ran ("bedrock" or a heuristic fallback).

    Pass the current conversation as `session_texts`. If you also have earlier
    conversations, pass them all as `conversations` (newest last, timestamps if
    known): topics that recur across conversations are weighted up, and the
    newest conversation still leads.
    """
    return await engine.extract(session_texts, _convs(conversations), user_id)


@mcp.tool()
async def search_content(topics: list[str]) -> list[dict]:
    """
    Match plain topic strings against TMDB (films, series and documentaries), arXiv
    (papers) and optionally YouTube. A missing key or failed call returns nothing
    for that source (mock results only when KNOTS_OFFLINE=1).
    """
    return await engine.search(topics)


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
    return await engine.recommend(session_texts, _convs(conversations), user_id, limit)


@mcp.tool()
async def thinking_partner(session_texts: list[str] | None = None, include_videos: bool = True,
                           conversations: list[ConversationIn] | None = None,
                           user_id: str | None = None) -> dict:
    """
    Call this when the person asks things like "what are we doing today?",
    "what should I explore?", "what should I watch or read?", or opens a
    session with no specific request. Acts as an intellectual partner: says what
    they seem to be circling (including what's underneath), then suggests films,
    series and documentaries to watch, papers to read and videos to explore, each
    with its reason. `spoken` is a short line Alexa can read aloud; `ask` is a
    follow-up question to keep the conversation going.

    Pass the current conversation as `session_texts`. If you can also see recent
    earlier conversations (up to about 8, last 60 days), pass them as
    `conversations` so recurring interests count for more.

    Pass `user_id` (a stable id for this person) to give Knots a memory: it keeps
    the interests and an embedding of each conversation (never the words), brings
    back related past conversations, and can answer with no conversation text at
    all ("what should I explore?") from what it remembers.
    """
    return await engine.plan(session_texts, _convs(conversations), user_id, include_videos)


@mcp.tool()
async def what_do_you_remember(user_id: str) -> dict:
    """
    Call when the person asks what Knots knows or remembers about them. Returns the
    interest threads it holds and how many conversations they come from. Knots keeps
    interests and embeddings only, never conversation text.
    """
    return await engine.what_remembered(user_id)


@mcp.tool()
async def forget_me(user_id: str) -> dict:
    """
    Call when the person asks Knots to forget them. Permanently deletes everything
    it stores for that user_id and returns how many conversation records were removed.
    """
    return await engine.forget(user_id)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MCP server on http://0.0.0.0:{port}/mcp (Streamable HTTP)")
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
