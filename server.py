"""
Alexa+ Research-to-Media Recommendation Engine — MCP server (v1, runnable)

Exposes four tools over Streamable HTTP (MCP spec 2025-11-25+):
  - extract_interests(session_texts)      -> topics
  - search_content(topics)                -> candidate titles from real catalogs
  - explain_recommendation(title, topic)  -> grounded one-line "why"
  - get_recommendations(session_texts)    -> the full pipeline, ready-to-render cards

`get_recommendations` is the one Alexa+ actually calls end-to-end; the other
three are exposed individually too, so they're independently testable/demoable
and so the AWS Builder swap (heuristic -> Bedrock in interests.py) only
touches one function.

Run:
    python server.py
Then point an MCP client (or Alexa+ MCP config) at:
    http://localhost:8000/mcp
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from cards import rank_and_render
from catalogs import search_content as _search_content
from interests import heuristic_extract

load_dotenv()

mcp = FastMCP("research-to-media-recs", stateless_http=True)


@mcp.tool()
def extract_interests(session_texts: list[str]) -> list[dict]:
    """
    Extract topics from recent Alexa+ conversation turns and/or Reddit
    activity text. Returns [{topic, depth, sentiment, mentions}, ...].
    """
    return [i.to_dict() for i in heuristic_extract(session_texts)]


@mcp.tool()
async def search_content(topics: list[str]) -> list[dict]:
    """
    Match a list of topics against TMDB (movies/docs), YouTube (videos),
    and arXiv (papers). Falls back to small offline mock results per
    source if no API key is set or a live call fails.
    """
    items = await _search_content(topics)
    return [i.to_dict() for i in items]


@mcp.tool()
async def get_recommendations(session_texts: list[str], limit: int = 6) -> dict:
    """
    Full pipeline: extract interests from session text, match against real
    content catalogs, rank, and return ready-to-render visual cards.

    This is the single tool Alexa+ calls for "what should I watch based on
    what I've been researching?"
    """
    interests = heuristic_extract(session_texts)
    if not interests:
        return {"interests": [], "cards": [], "note": "No topics detected in session_texts."}

    topics = [i.topic for i in interests]
    items = await _search_content(topics)
    cards = rank_and_render(items, interests, limit=limit)

    return {
        "interests": [i.to_dict() for i in interests],
        "cards": cards,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MCP server on http://0.0.0.0:{port}/mcp (Streamable HTTP)")
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
