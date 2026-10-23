"""
Knots Explorer: a Swagger-style page for the MCP tools, so anyone can see what goes in and what comes out.

    GET  /explore                 the page (knots/explorer_ui/index.html)
    GET  /explore/api/tools       the tools as the MCP server declares them (name, description, JSON schema)
    GET  /explore/api/status      what is switched on: floor, notes, trained weights, memory, extractor
    POST /explore/api/call        {"tool": "...", "arguments": {...}} -> {"ok", "result", "ms", "request"}

It calls the same tools through the same FastMCP server as any MCP client, so what the page shows is what Alexa+
gets. It is a public door onto a server that spends Bedrock, TMDB and YouTube quota, so:

  * It is OFF unless KNOTS_EXPLORER=on (every route then answers 404).
  * KNOTS_EXPLORER_PASSCODE, if set, must be sent as the X-Passcode header on the API routes.
  * A visitor is a random id in a signed, HttpOnly, SameSite=Strict cookie. It becomes an opaque per-visitor user
    (memory and ratings work, and nobody can read another visitor's). With no KNOTS_USER_SALT there is no stable
    identity, so the page works but remembers nothing. A caller-supplied `user_id` argument is always dropped.
  * Each visitor is rate limited (KNOTS_EXPLORER_RATE calls per minute, default 12) on top of the global limit.
  * POST needs `Content-Type: application/json` and an `X-Knots-Explorer` header, which a cross-site form cannot send.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from . import fitmodel, identity, notewriter
from .ratelimit import RateLimiter

PAGE = Path(__file__).parent / "explorer_ui" / "index.html"
PITCH = Path(__file__).parent / "pitch.html"
COOKIE = "knots_visitor"
MAX_ARGS_BYTES = 60_000
KIND = {"knots_partner": "read", "get_recommendations": "read", "extract_interests": "read", "search_content": "read",
        "what_do_you_remember": "read", "rate_pick": "write", "forget_me": "delete"}
GROUP = {"extract_interests": "Understand", "search_content": "Understand", "knots_partner": "Recommend",
         "get_recommendations": "Recommend", "rate_pick": "Remember", "what_do_you_remember": "Remember",
         "forget_me": "Remember"}


def enabled() -> bool:
    return os.environ.get("KNOTS_EXPLORER", "off").lower() == "on"


def _key() -> bytes:
    return identity.salt() or b""


def _sign(visitor_id: str) -> str:
    return hmac.new(_key(), visitor_id.encode(), hashlib.sha256).hexdigest()[:20]


def visitor_from(request: Request) -> tuple[str, bool]:
    """(visitor id, is_new). A cookie with a bad signature is treated as no cookie."""
    raw = request.cookies.get(COOKIE, "")
    vid, _, sig = raw.partition(".")
    if vid and sig and hmac.compare_digest(sig, _sign(vid)):
        return vid, False
    return secrets.token_urlsafe(18), True


def _with_cookie(request: Request, resp: Response, vid: str, new: bool) -> Response:
    if new:
        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        resp.set_cookie(COOKIE, f"{vid}.{_sign(vid)}", max_age=60 * 60 * 24 * 90, httponly=True, samesite="strict", secure=secure)
    return resp


def _denied(request: Request) -> Response | None:
    if not enabled():
        return Response("Not found", status_code=404)
    want = os.environ.get("KNOTS_EXPLORER_PASSCODE", "")
    if want and not hmac.compare_digest(request.headers.get("x-passcode", ""), want):
        return JSONResponse({"ok": False, "error": "This explorer needs the passcode. Enter it at the top right."}, status_code=401)
    return None


def _result_of(out):
    """The tool's dict, whichever shape FastMCP hands back."""
    if isinstance(out, tuple) and len(out) == 2:
        out = out[1] if isinstance(out[1], dict) else out[0]
    if isinstance(out, dict):
        return out.get("result", out) if set(out) == {"result"} else out
    if isinstance(out, (list, tuple)):
        texts = [getattr(b, "text", None) for b in out]
        if len(texts) == 1 and texts[0]:
            try:
                return json.loads(texts[0])
            except ValueError:
                return {"text": texts[0]}
        parsed = []
        for t in texts:
            try:
                parsed.append(json.loads(t))
            except (TypeError, ValueError):
                parsed.append(t)
        return parsed
    return out


def register(server, limiter) -> None:
    per_visitor = RateLimiter(per_key=int(os.environ.get("KNOTS_EXPLORER_RATE", "12")), overall=0)

    async def page(request: Request) -> Response:
        if not enabled():
            return Response("Not found", status_code=404)
        vid, new = visitor_from(request)
        resp = HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
        return _with_cookie(request, resp, vid, new)

    async def pitch(request: Request) -> Response:
        return HTMLResponse(PITCH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})

    async def tools(request: Request) -> Response:
        if (d := _denied(request)) is not None:
            return d
        listed = await server.list_tools()
        out = [{"name": t.name, "description": (t.description or "").strip(), "inputSchema": t.inputSchema,
                "kind": KIND.get(t.name, "read"), "group": GROUP.get(t.name, "Other")} for t in listed]
        return JSONResponse({"tools": out})

    async def status(request: Request) -> Response:
        if (d := _denied(request)) is not None:
            return d
        f = fitmodel.floor() if fitmodel.floor_active() else None
        return JSONResponse({
            "passcode_required": bool(os.environ.get("KNOTS_EXPLORER_PASSCODE")),
            "remembers_visitors": bool(identity.salt()),
            "extractor": "bedrock" if os.environ.get("BEDROCK_MODEL_ID") else "heuristic",
            "ranker": "trained" if fitmodel.trained() else "hand-set weights",
            "floor": ({"similarity": f["sim"], "fit": f["score"], "embedder": f["embedder"]} if f else None),
            "notes": "model-written" if notewriter.enabled() else "templated",
            "offline_mock_catalogs": os.environ.get("KNOTS_OFFLINE") == "1"})

    async def call(request: Request) -> Response:
        if (d := _denied(request)) is not None:
            return d
        if "application/json" not in request.headers.get("content-type", "") or not request.headers.get("x-knots-explorer"):
            return JSONResponse({"ok": False, "error": "Send JSON with an X-Knots-Explorer header."}, status_code=400)
        raw = await request.body()
        if len(raw) > MAX_ARGS_BYTES:
            return JSONResponse({"ok": False, "error": "That input is too large for the explorer."}, status_code=413)
        try:
            body = json.loads(raw or b"{}")
            name, args = str(body["tool"]), dict(body.get("arguments") or {})
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"ok": False, "error": 'Body must look like {"tool": "knots_partner", "arguments": {}}.'}, status_code=400)
        known = {t.name for t in await server.list_tools()}
        if name not in known:
            return JSONResponse({"ok": False, "error": f"Unknown tool {name!r}."}, status_code=404)
        args.pop("user_id", None)                       # identity is the visitor cookie, never an argument
        vid, new = visitor_from(request)
        if not per_visitor.allow(vid):
            return _with_cookie(request, JSONResponse({"ok": False, "error": "Slow down: too many runs a minute. Try again shortly."}, status_code=429), vid, new)
        envelope = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
        user = identity.hash_user(vid, "explorer")
        t0 = time.perf_counter()
        try:
            with identity.acting_as(user):
                out = await server.call_tool(name, args)
            payload = {"ok": True, "result": _result_of(out)}
        except Exception as e:  # noqa: BLE001 - show the tool's own error, as an MCP client would see it
            msg = str(e)
            payload = {"ok": False, "error": msg[:600] or type(e).__name__}
        payload.update(ms=round((time.perf_counter() - t0) * 1000), request=envelope, remembered_as="visitor" if user else "nobody")
        return _with_cookie(request, JSONResponse(payload), vid, new)

    server.custom_route("/", methods=["GET"])(pitch)
    server.custom_route("/explore", methods=["GET"])(page)
    server.custom_route("/explore/api/tools", methods=["GET"])(tools)
    server.custom_route("/explore/api/status", methods=["GET"])(status)
    server.custom_route("/explore/api/call", methods=["POST"])(call)
