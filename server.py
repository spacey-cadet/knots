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

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()  # before the imports below read any env-driven config

from cards import rank_and_render  # noqa: E402
from catalogs import search_content as _search_content  # noqa: E402
from interests import Interest, extract  # noqa: E402
from partner import build_session_plan  # noqa: E402

mcp = FastMCP("knots", stateless_http=True)


@mcp.tool()
async def extract_interests(session_texts: list[str]) -> dict:
    """
    Extract what the person is really interested in from recent Alexa+
    conversation turns and/or Reddit activity text. For each interest returns
    the surface topic, the abstract concepts at stake, unstated themes the text
    implies, and film-tag-style keywords. Also reports which extractor ran
    ("bedrock" or a heuristic fallback).
    """
    interests, used = await extract(session_texts)
    return {"extractor": used, "interests": [i.to_dict() for i in interests]}


@mcp.tool()
async def search_content(topics: list[str]) -> list[dict]:
    """
    Match plain topic strings against TMDB (films and documentaries), arXiv
    (papers) and optionally YouTube. Falls back to small offline mock results
    per source if no key is set or a live call fails.
    """
    interests = [Interest(topic=t, depth="casual", sentiment="curious", mentions=1,
                          concepts=[t], keywords=[t]) for t in topics if t.strip()]
    return [i.to_dict() for i in await _search_content(interests)]


@mcp.tool()
async def get_recommendations(session_texts: list[str], limit: int = 6) -> dict:
    """
    Full pipeline: extract interests from session text, match them against
    the content catalogs, rank, and return ready-to-render cards, each with the
    reason it was chosen.
    """
    interests, used = await extract(session_texts)
    if not interests:
        return {"extractor": used, "interests": [], "cards": [],
                "note": "No topics detected in session_texts."}
    items = await _search_content(interests)
    return {
        "extractor": used,
        "interests": [i.to_dict() for i in interests],
        "cards": rank_and_render(items, interests, limit=limit),
    }


@mcp.tool()
async def thinking_partner(session_texts: list[str], include_videos: bool = True) -> dict:
    """
    Call this when the person asks things like "what are we doing today?",
    "what should I explore?", "what should I watch or read?", or opens a
    session with no specific request. Acts as an intellectual partner: says what
    they seem to be circling (including what's underneath), then suggests films
    and documentaries to watch, papers to read and videos to explore, each with
    its reason. `spoken` is a short line Alexa can read aloud; `ask` is a
    follow-up question to keep the conversation going.
    """
    interests, used = await extract(session_texts)
    if not interests:
        return {"extractor": used, "threads": [], "picks": {"watch": [], "read": [], "explore": []},
                "spoken": "I don't have anything to go on yet. What have you been curious about lately?",
                "ask": "What have you been curious about lately?"}
    items = await _search_content(interests, include_videos=include_videos)
    return build_session_plan(interests, items, extractor=used)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MCP server on http://0.0.0.0:{port}/mcp (Streamable HTTP)")
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
