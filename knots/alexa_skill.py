"""
The Alexa custom-skill endpoint: POST /alexa.

Alexa sends every request to a public HTTPS URL, so anyone on the internet can POST to it. Amazon's rules for
trusting a request are all enforced here, and a request that fails any of them gets HTTP 400 before it can cost a
Bedrock call:

  1. SignatureCertChainUrl must be https://s3.amazonaws.com[:443]/echo.api/... (after normalising the path)
  2. the certificate chain it points to must be inside its validity dates, name echo-api.amazon.com, and chain to a
     trusted root (the `certifi` bundle)
  3. the Signature-256 header (RSA-SHA256, base64) must verify against the exact request body with that
     certificate's key (the older SHA-1 `Signature` header is accepted only when Signature-256 is absent)
  4. the request timestamp must be within 150 seconds of now (replay protection)
  5. the request must be for OUR skill: `applicationId` must equal KNOTS_ALEXA_SKILL_ID, or any other skill could
     use this endpoint

The person is `session.user.userId` from a request that passed 1-4, turned into an opaque salted id
(knots/identity.py). That id is scoped to the skill, which is why no account linking is needed for memory.

Alexa allows about 8 seconds for a reply and the recommendation pipeline (Bedrock, three catalogs, embeddings)
can take longer. The work therefore runs under a deadline (KNOTS_ALEXA_TIMEOUT_S, default 6.5). If it is missed the
person is told to ask again, the work keeps running, and the same question within two minutes gets the finished
answer immediately instead of starting over. There are no progressive responses yet.

Feedback: after picks the reprompt asks whether the first pick was good; "good pick" / "not for me" record a
verdict through `feedback.record_feedback`, which is what retraining learns from.

  KNOTS_ALEXA_SKILL_ID      amzn1.ask.skill.... (Alexa developer console -> your skill). Without it the route answers 503.
  KNOTS_ALEXA_TIMEOUT_S     deadline for the pipeline (default 6.5)
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import posixpath
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

from . import engine, feedback, identity
from .alexa import to_alexa

log = logging.getLogger("knots.alexa")

CERT_HOST = "s3.amazonaws.com"
CERT_PATH_PREFIX = "/echo.api/"
CERT_SAN = "echo-api.amazon.com"
TIMESTAMP_TOLERANCE_S = 150
MAX_BODY_BYTES = 256 * 1024
MAX_CHAIN_BYTES = 100 * 1024
PENDING_TTL_S = 120
MAX_PENDING = 200

WELCOME = ("This is Knots, your partner for what to watch and read. Tell me what you have been curious about, "
           "or ask me what you should explore.")
HELP = ("I suggest films, series, videos and papers that fit what you have been thinking about. "
        "Try: ask me about memecoins. Or say, what should I explore. "
        "After I suggest something, you can say good pick, or not for me, and I will learn from it.")
REPROMPT = "What have you been curious about lately?"
SLOW = ("That is taking me longer than usual. Ask me again in a moment and I should have it ready.")
BROKE = "Sorry, something went wrong on my side. Please try again."
BUSY = "I am getting a lot of requests right now. Please try again in a minute."


class AlexaVerificationError(Exception):
    """The request is not provably from Alexa for this skill. The route turns this into HTTP 400."""


# ----------------------------------------------------------------------------------- request verification
def cert_url_ok(url: str) -> bool:
    """Amazon's rules for SignatureCertChainUrl, applied after normalising dot segments and repeated slashes."""
    try:
        parts = urlsplit((url or "").strip())
        port = parts.port
    except ValueError:
        return False
    if parts.scheme.lower() != "https" or (parts.hostname or "").lower() != CERT_HOST or port not in (None, 443):
        return False
    if parts.username or parts.password:
        return False
    path = posixpath.normpath(parts.path)
    return path.startswith(CERT_PATH_PREFIX.rstrip("/") + "/")


def _roots():
    import certifi
    from cryptography import x509
    with open(certifi.where(), "rb") as f:
        return x509.load_pem_x509_certificates(f.read())


def verify_chain_and_signature(chain_pem: bytes, signature_b64: str, body: bytes, sha256: bool = True,
                               roots=None, now: datetime | None = None) -> None:
    """Raises AlexaVerificationError unless `body` was signed by a valid echo-api.amazon.com certificate."""
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.x509.verification import PolicyBuilder, Store, VerificationError
    now = now or datetime.now(timezone.utc)
    try:
        certs = x509.load_pem_x509_certificates(chain_pem)
        signature = base64.b64decode(signature_b64, validate=True)
    except ValueError as e:
        raise AlexaVerificationError(f"unreadable certificate or signature: {e}") from e
    if not certs:
        raise AlexaVerificationError("no certificate in the chain")
    leaf, inter = certs[0], certs[1:]
    if not (leaf.not_valid_before_utc <= now <= leaf.not_valid_after_utc):
        raise AlexaVerificationError("signing certificate is outside its validity dates")
    try:
        names = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName)
    except x509.ExtensionNotFound:
        names = []
    if CERT_SAN not in names:
        raise AlexaVerificationError(f"signing certificate does not name {CERT_SAN}")
    try:
        verifier = (PolicyBuilder().store(Store(roots if roots is not None else _roots())).time(now)
                    .build_server_verifier(x509.DNSName(CERT_SAN)))
        verifier.verify(leaf, inter)
    except (VerificationError, ValueError) as e:
        raise AlexaVerificationError(f"certificate chain is not trusted: {e}") from e
    try:
        leaf.public_key().verify(signature, body, padding.PKCS1v15(), hashes.SHA256() if sha256 else hashes.SHA1())
    except (InvalidSignature, TypeError, ValueError) as e:
        raise AlexaVerificationError("signature does not match the request body") from e


_chain_cache: dict[str, tuple[float, bytes]] = {}


async def fetch_chain(url: str) -> bytes:
    """Download the certificate chain (cached an hour: Alexa reuses a few URLs, and each fetch is a network call)."""
    hit = _chain_cache.get(url)
    if hit and time.monotonic() - hit[0] < 3600:
        return hit[1]
    import httpx
    async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
        r = await client.get(url)
    if r.status_code != 200 or len(r.content) > MAX_CHAIN_BYTES:
        raise AlexaVerificationError("could not fetch the signing certificate")
    if len(_chain_cache) > 20:
        _chain_cache.clear()
    _chain_cache[url] = (time.monotonic(), r.content)
    return r.content


def check_timestamp(body: dict, now: float | None = None) -> None:
    raw = ((body.get("request") or {}).get("timestamp")) or ""
    try:
        sent = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError as e:
        raise AlexaVerificationError("request has no valid timestamp") from e
    if abs((time.time() if now is None else now) - sent) > TIMESTAMP_TOLERANCE_S:
        raise AlexaVerificationError("request timestamp is outside the 150 second window")


def application_id(body: dict) -> str:
    return (((body.get("session") or {}).get("application") or {}).get("applicationId")
            or (((body.get("context") or {}).get("System") or {}).get("application") or {}).get("applicationId") or "")


async def verify_request(headers, raw: bytes, body: dict, fetch=None, roots=None, now: float | None = None,
                         skill_id: str | None = None) -> None:
    skill_id = skill_id if skill_id is not None else os.environ.get("KNOTS_ALEXA_SKILL_ID", "")
    url = headers.get("signaturecertchainurl", "")
    sig, sha256 = headers.get("signature-256", ""), True
    if not sig:
        sig, sha256 = headers.get("signature", ""), False
    if not url or not sig:
        raise AlexaVerificationError("missing signature headers")
    if not cert_url_ok(url):
        raise AlexaVerificationError("SignatureCertChainUrl is not an Amazon certificate location")
    check_timestamp(body, now)
    chain = await (fetch or fetch_chain)(url)
    verify_chain_and_signature(chain, sig, raw, sha256, roots,
                               datetime.fromtimestamp(now, timezone.utc) if now else None)
    if not skill_id or application_id(body) != skill_id:
        raise AlexaVerificationError("request is for a different skill")


# ----------------------------------------------------------------------------------------------- the skill
def say(text: str, reprompt: str | None = None, end: bool = False, attrs: dict | None = None) -> dict:
    out = {"version": "1.0", "sessionAttributes": attrs or {},
           "response": {"outputSpeech": {"type": "PlainText", "text": text}, "shouldEndSession": end}}
    if reprompt and not end:
        out["response"]["reprompt"] = {"outputSpeech": {"type": "PlainText", "text": reprompt}}
    return out


def _slot(intent: dict, name: str) -> str:
    v = (((intent.get("slots") or {}).get(name)) or {}).get("value")
    return (v or "").strip()[:300]


def _user(body: dict) -> str | None:
    return identity.hash_user((((body.get("session") or {}).get("user")) or {}).get("userId"), "alexa")


_pending: dict[str, tuple[float, asyncio.Task]] = {}


async def _plan_by_deadline(key: str, texts: list[str], user: str | None, timeout: float):
    """(plan, session_id) or None if the deadline passed. The work outlives the deadline so a retry finds it done."""
    now = time.monotonic()
    for k in [k for k, (t, _) in _pending.items() if now - t > PENDING_TTL_S]:
        _pending.pop(k, None)
    hit = _pending.get(key)
    if hit is None:
        if len(_pending) >= MAX_PENDING:
            _pending.clear()

        async def work():
            plan = await engine.plan(texts, [], user)
            sid = feedback.new_session_id()
            plan["session_id"] = sid
            await asyncio.to_thread(feedback.record_impression, plan, user, sid)
            return plan, sid
        hit = (now, asyncio.ensure_future(work()))
        _pending[key] = hit
    try:
        return await asyncio.wait_for(asyncio.shield(hit[1]), timeout)
    except asyncio.TimeoutError:
        return None
    except Exception:                                              # noqa: BLE001
        _pending.pop(key, None)
        raise


def _recommendation(plan: dict, session_id: str) -> dict:
    resp = plan.get("response") or {"say": plan.get("spoken", ""), "picks": [], "ask": plan.get("ask", "")}
    out = to_alexa(resp)
    attrs = {"session_id": session_id, "last_url": "", "last_title": ""}
    if resp["picks"]:
        first = resp["picks"][0]
        attrs.update(last_url=first.get("url", ""), last_title=first.get("title", ""))
        out["response"]["reprompt"] = {"outputSpeech": {
            "type": "PlainText", "text": f"Was {first['title']} a good pick? Say good pick, or not for me."}}
    out["sessionAttributes"] = attrs
    return out


async def _rate(body: dict, good: bool) -> dict:
    attrs = ((body.get("session") or {}).get("attributes")) or {}
    user = _user(body)
    if not attrs.get("last_url") or not attrs.get("session_id"):
        return say("I do not have a pick for you to rate yet. Ask me what to explore first.", REPROMPT)
    if not user:
        return say("Thanks. I cannot save feedback for this request, but I heard you.", REPROMPT, attrs=attrs)
    result = await asyncio.to_thread(feedback.record_feedback, user, attrs["session_id"], attrs["last_url"], good)
    text = ("Thanks, I will remember that. " if result["recorded"] else "Thanks. ") + "What else are you curious about?"
    return say(text, REPROMPT, attrs={**attrs, "last_url": "", "last_title": ""} if result["recorded"] else attrs)


async def handle_alexa(body: dict) -> dict:
    """One verified Alexa request in, one Alexa response out. Never raises."""
    try:
        req = body.get("request") or {}
        kind = req.get("type")
        if kind == "LaunchRequest":
            return say(WELCOME, REPROMPT)
        if kind == "SessionEndedRequest":
            return {"version": "1.0", "response": {}}
        if kind != "IntentRequest":
            return say(HELP, REPROMPT)
        intent = req.get("intent") or {}
        name = intent.get("name", "")
        if name == "KnotsPartnerIntent":
            user, topic = _user(body), _slot(intent, "topic")
            key = f"{user or 'anon'}|{topic.lower()}"
            got = await _plan_by_deadline(key, [topic] if topic else [], user,
                                          float(os.environ.get("KNOTS_ALEXA_TIMEOUT_S", "6.5")))
            return say(SLOW, REPROMPT) if got is None else _recommendation(*got)
        if name == "GoodPickIntent":
            return await _rate(body, True)
        if name == "BadPickIntent":
            return await _rate(body, False)
        if name in ("AMAZON.StopIntent", "AMAZON.CancelIntent"):
            return say("Goodbye.", end=True)
        return say(HELP, REPROMPT)                                 # help, fallback, navigate home, anything unknown
    except Exception:                                              # noqa: BLE001
        log.exception("alexa request failed")
        return say(BROKE, REPROMPT)


async def handle_http(request, limiter, fetch=None, roots=None, now: float | None = None):
    """The Starlette route body for POST /alexa."""
    import json
    from starlette.responses import JSONResponse, Response
    if not os.environ.get("KNOTS_ALEXA_SKILL_ID"):
        return Response("Alexa skill is not configured", status_code=503)
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        return Response("request too large", status_code=413)
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return Response("request too large", status_code=413)
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("not an object")
        await verify_request(request.headers, raw, body, fetch, roots, now)
    except AlexaVerificationError as e:
        log.warning("alexa request rejected: %s", e)
        return Response("invalid request", status_code=400)
    except ValueError:
        return Response("invalid request", status_code=400)
    except Exception:                                              # noqa: BLE001  (certificate fetch failed, etc.)
        log.exception("alexa verification could not complete")
        return Response("invalid request", status_code=400)
    if not limiter.allow(_user(body) or "alexa-anonymous"):
        return JSONResponse(say(BUSY, end=True))
    return JSONResponse(await handle_alexa(body))
