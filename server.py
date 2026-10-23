"""
Knots — turns what you research into films, series, explainers and papers that make it concrete
(MCP server for Alexa+, plus an Alexa custom-skill endpoint).

This file is transport only: it declares the MCP tools and hands every call to `knots.engine`,
where the whole recommendation algorithm lives. There is no recommendation logic here.

Tools (Streamable HTTP, MCP spec 2025-11-25+):
  - knots_partner      "what are we doing today?": threads, what's underneath them, and things
                          to watch / read / explore, each with a reason. Returns a `session_id`.
  - rate_pick             the person said whether a pick was good: the feedback that drives retraining
  - get_recommendations   the full pipeline as ranked cards
  - extract_interests     topics, concepts, unstated themes, keywords, best mediums
  - search_content        candidate titles from the real catalogs
  - what_do_you_remember  what Knots holds about a person (interests and embeddings, never words)
  - forget_me             delete it (memory and logged events)

Other routes (outside the MCP bearer-token check, each protected its own way):
  GET  /health   liveness for the load balancer and Docker
  POST /alexa    Alexa custom skill requests (Amazon's signature is verified; see knots/alexa_skill.py)
  GET  /explore  a Swagger-style page for these tools (off unless KNOTS_EXPLORER=on; see knots/explorer.py)

Identity. A `user_id` argument is NOT trusted: who is calling comes from a verified bearer token (set
KNOTS_JWKS_URL or KNOTS_JWT_SECRET, see knots/identity.py) and is turned into an opaque salted id. With no auth
configured the tools run stateless: nothing is remembered and no feedback is accepted.

Interest extraction uses Amazon Bedrock when configured (BEDROCK_MODEL_ID) and falls back to an
offline heuristic on any failure, so the server always answers.

Run:
    python server.py
Then point an MCP client at http://localhost:8000/mcp. In production run it behind a TLS-terminating proxy
(docs/DEPLOY_EC2.md) with KNOTS_HOST=0.0.0.0 and KNOTS_ALLOWED_HOSTS set to the public hostname.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

load_dotenv()  # before the import below reads any env-driven config

from knots import Conversation, engine, feedback, identity  # noqa: E402
from knots.ratelimit import RateLimiter  # noqa: E402

limiter = RateLimiter()


class ConversationIn(BaseModel):
    """One conversation (or one Reddit session) as the assistant remembers it."""
    turns: list[str] = Field(description="The person's messages in this conversation, oldest first.")
    timestamp: str | None = Field(default=None, description="When it happened, ISO 8601. Optional.")
    id: str | None = Field(default=None, description="Any stable id. Optional.")


def _convs(conversations: list[ConversationIn] | None) -> list[Conversation]:
    return [Conversation(turns=c.turns, timestamp=c.timestamp, id=c.id or "") for c in conversations or []]


def _caller(user_id: str | None = None) -> str | None:
    """The verified caller (or None), after the rate limit. A supplied `user_id` is ignored unless development
    mode is on. Raises ValueError, which the MCP layer returns as a tool error, when the caller is over the limit."""
    user = identity.resolve_user(user_id)
    if not limiter.allow(user or "anonymous"):
        raise ValueError("Too many requests right now. Please try again in a minute.")
    return user


# ----------------------------------------------------------------------------------------------- tools
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
    newest conversation still leads. `user_id` is ignored: the caller's identity
    comes from their sign-in.
    """
    return await engine.extract(session_texts, _convs(conversations), _caller(user_id))


async def search_content(topics: list[str]) -> list[dict]:
    """
    Match plain topic strings against TMDB (films, series and documentaries), arXiv
    (papers) and optionally YouTube. A missing key or failed call returns nothing
    for that source (mock results only when KNOTS_OFFLINE=1).
    """
    _caller()
    return await engine.search(topics)


async def get_recommendations(session_texts: list[str] | None = None, limit: int = 6,
                              conversations: list[ConversationIn] | None = None,
                              user_id: str | None = None,
                              challenge_me: bool = False) -> dict:
    """
    Full pipeline: extract interests from session text, match them against
    the content catalogs, rank, and return ready-to-render cards, each with the
    reason it was chosen. Accepts one conversation (`session_texts`) or several
    (`conversations`), as in extract_interests.
    """
    return await engine.recommend(session_texts, _convs(conversations), _caller(user_id), limit,
                                  challenge=challenge_me or None)


async def knots_partner(session_texts: list[str] | None = None, include_videos: bool = True,
                           conversations: list[ConversationIn] | None = None,
                           user_id: str | None = None,
                           challenge_me: bool = False) -> dict:
    """
    Call this when the person asks things like "what are we doing today?",
    "what should I explore?", "what should I watch or read?", or opens a
    session with no specific request. Acts as an intellectual partner: says what
    they seem to be circling (including what's underneath), then suggests films,
    series and documentaries to watch, papers to read and videos to explore, each
    with its reason. `response.say` is a short line to read aloud; `response.ask`
    is a follow-up question to keep the conversation going.

    Pass the current conversation as `session_texts`. If you can also see recent
    earlier conversations (up to about 8, last 60 days), pass them as
    `conversations` so recurring interests count for more. When the person is
    signed in, Knots remembers their interests (an embedding of each conversation,
    never the words) and can answer with no conversation text at all.

    The result carries a `session_id`. If the person then says whether a pick was
    good or not, call `rate_pick` with that `session_id` and the pick's `url`.

    Also returns `pairs`: watch-and-learn double features (a film or series plus the explainer or
    paper on the same idea), each card with its receipts. Set `challenge_me` to reserve one pick
    that complicates the person's own view.
    """
    user = _caller(user_id)
    plan = await engine.plan(session_texts, _convs(conversations), user, include_videos,
                             challenge=challenge_me or None)
    plan["session_id"] = feedback.new_session_id()
    await asyncio.to_thread(feedback.record_impression, plan, user, plan["session_id"])
    return plan


async def rate_pick(session_id: str, url: str, good: bool) -> dict:
    """
    Call when the person says whether a pick you gave them was good ("that was a good pick", "not for
    me", "I liked it", "that wasn't what I meant"). Pass the `session_id` from the knots_partner result and
    the `url` of the pick they mean (use the first pick if they say "that one" about the first thing you read out).
    `good` is true for a good pick and false for one that missed. Only works for a signed-in person and only for
    a pick that was actually shown to them in that session.
    """
    user = _caller()
    return await asyncio.to_thread(feedback.record_feedback, user, session_id, url, bool(good))


async def what_do_you_remember(user_id: str | None = None) -> dict:
    """
    Call when the person asks what Knots knows or remembers about them. Returns the
    interest threads it holds and how many conversations they come from. Knots keeps
    interests and embeddings only, never conversation text.
    """
    user = _caller(user_id)
    if not user:
        return {"remembered_conversations": 0, "threads": [],
                "note": "Knots only remembers people who are signed in, so nothing is held for this request."}
    return await engine.what_remembered(user)


async def forget_me(user_id: str | None = None) -> dict:
    """
    Call when the person asks Knots to forget them. Permanently deletes everything
    it stores for them (remembered conversations and logged feedback) and returns how many
    conversation records were removed.
    """
    user = _caller(user_id)
    if not user:
        return {"deleted": 0, "note": "Nothing is stored for an unsigned request."}
    out = await engine.forget(user)
    out["events_deleted"] = await asyncio.to_thread(feedback.forget_events, user)
    return out


TOOLS = (knots_partner, rate_pick, get_recommendations, extract_interests, search_content,
         what_do_you_remember, forget_me)


# ----------------------------------------------------------------------------------------------- routes
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def alexa_route(request: Request):
    from knots import alexa_skill
    return await alexa_skill.handle_http(request, limiter)


def _hosts() -> list[str]:
    return [h.strip() for h in os.environ.get("KNOTS_ALLOWED_HOSTS", "").split(",") if h.strip()]


def create_mcp() -> FastMCP:
    """Build the server from the environment. A function so tests can build it under different settings."""
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.transport_security import TransportSecuritySettings

    kwargs: dict = {"stateless_http": True, "host": os.environ.get("KNOTS_HOST", "127.0.0.1"),
                    "port": int(os.environ.get("PORT", 8000))}
    hosts = _hosts()
    if hosts:                       # DNS-rebinding protection for the public hostname(s); Host and Origin are checked
        kwargs["transport_security"] = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[x for h in hosts for x in (h, f"{h}:*")],
            allowed_origins=[f"https://{h}" for h in hosts])
    elif kwargs["host"] not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: listening beyond localhost with no KNOTS_ALLOWED_HOSTS: Host headers are not restricted.")
    if identity.auth_configured():
        base = os.environ.get("KNOTS_JWT_AUDIENCE", "")
        kwargs["token_verifier"] = identity.JwtVerifier()
        kwargs["auth"] = AuthSettings(
            issuer_url=os.environ.get("KNOTS_JWT_ISSUER") or os.environ.get("KNOTS_PUBLIC_URL") or base,
            resource_server_url=base, validate_token_resource=False)
        if not identity.salt():
            print("WARNING: KNOTS_USER_SALT is not set, so signed-in callers have no stable identity "
                  "(nothing will be remembered and feedback is refused).")
    server = FastMCP("knots", **kwargs)
    for fn in TOOLS:
        server.tool()(fn)
    server.custom_route("/health", methods=["GET"])(health)
    server.custom_route("/alexa", methods=["POST"])(alexa_route)
    from knots import explorer
    explorer.register(server, limiter)        # inert (404) unless KNOTS_EXPLORER=on
    return server


mcp = create_mcp()


if __name__ == "__main__":
    s = mcp.settings
    print(f"Starting MCP server on http://{s.host}:{s.port}/mcp (Streamable HTTP); "
          f"auth {'on' if identity.auth_configured() else 'OFF (stateless)'}")
    mcp.run(transport="streamable-http")
