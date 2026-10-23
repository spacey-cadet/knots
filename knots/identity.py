"""
Who is calling, established from something the caller cannot simply type.

Two sources, both verified before they are trusted:

  * MCP: a bearer token (a JWT) checked against the issuer's keys, audience and expiry.
  * Alexa: `session.user.userId` on a request whose Amazon signature has been verified (knots/alexa_skill.py).

Neither raw value is ever stored or logged. `hash_user` turns it into an opaque, salted id (HMAC-SHA256), so the
memory store and the event log hold no Amazon id or token subject, and the same person always maps to the same id.
Without a salt there is no stable pseudonym, so there is no identity: callers are treated as anonymous (stateless).

A `user_id` argument supplied by a caller is NOT an identity. `resolve_user` ignores it unless the operator sets
KNOTS_ALLOW_CLIENT_USER_ID=1 (local development only: with it on, anyone can read anyone's profile).

Environment:
  KNOTS_USER_SALT        required for any identity: a long random secret (openssl rand -hex 32). Changing it
                         orphans every stored profile, so treat it like a key.
  KNOTS_JWT_ISSUER       issuer (`iss`) tokens must carry
  KNOTS_JWT_AUDIENCE     audience (`aud`) tokens must carry: this server's public URL, e.g. https://knots.example.com/mcp
  KNOTS_JWKS_URL         the issuer's public keys (RS256/ES256), e.g. Cognito's or Auth0's jwks.json
  KNOTS_JWT_SECRET       OR a shared secret (HS256), for a token you mint yourself; at least 32 characters
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import hmac
import logging
import os
import time

log = logging.getLogger("knots.identity")

MIN_SECRET_CHARS = 32
ASYMMETRIC = ("RS256", "RS384", "RS512", "ES256", "ES384", "PS256")


def salt() -> bytes:
    return os.environ.get("KNOTS_USER_SALT", "").encode()


def hash_user(subject: str | None, namespace: str = "") -> str | None:
    """An opaque stable id for a verified subject, or None when there is no subject or no salt.

    `namespace` keeps the same string from two sources (an Alexa id and a token subject) from colliding."""
    subject = (subject or "").strip()
    key = salt()
    if not subject or not key:
        return None
    digest = hmac.new(key, f"{namespace}\x00{subject}".encode(), hashlib.sha256).hexdigest()
    return "u_" + digest[:32]


def auth_configured() -> bool:
    return bool(os.environ.get("KNOTS_JWKS_URL") or os.environ.get("KNOTS_JWT_SECRET"))


def allow_client_user_id() -> bool:
    return os.environ.get("KNOTS_ALLOW_CLIENT_USER_ID") == "1"


_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("knots_user_override", default=None)


@contextlib.contextmanager
def acting_as(user: str | None):
    """Run a block as an already-verified, opaque user id. Only code that has itself verified the caller (the
    explorer's signed visitor cookie) may use this; nothing a client sends reaches it."""
    token = _override.set(user)
    try:
        yield
    finally:
        _override.reset(token)


def current_user() -> str | None:
    """The verified caller of the MCP request being handled, as an opaque id, or None (anonymous)."""
    if _override.get():
        return _override.get()
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
    except ImportError:                                           # pragma: no cover
        return None
    token = get_access_token()
    return hash_user(getattr(token, "subject", None), "mcp") if token else None


def resolve_user(client_supplied: str | None = None) -> str | None:
    """The id to read and write memory under. Verified identity always wins; a caller-supplied id counts only in
    explicit development mode."""
    verified = current_user()
    if verified:
        return verified
    if allow_client_user_id() and (client_supplied or "").strip():
        return hash_user(client_supplied, "client") or (client_supplied or "").strip()[:128]
    return None


class JwtVerifier:
    """Verifies bearer tokens for the MCP endpoint (implements the MCP SDK's TokenVerifier protocol).

    Returns None, never raises, for anything that is not a valid, unexpired token from the configured issuer for
    this audience, so a bad token is a 401 and not a crash. `subject` is the token's `sub`."""

    def __init__(self, issuer: str | None = None, audience: str | None = None, jwks_url: str | None = None,
                 secret: str | None = None, algorithms: list[str] | None = None, leeway_s: int = 30):
        self.issuer = issuer or os.environ.get("KNOTS_JWT_ISSUER") or None
        self.audience = audience or os.environ.get("KNOTS_JWT_AUDIENCE") or None
        self.jwks_url = jwks_url or os.environ.get("KNOTS_JWKS_URL") or None
        self.secret = secret or os.environ.get("KNOTS_JWT_SECRET") or None
        if not (self.jwks_url or self.secret):
            raise ValueError("JwtVerifier needs KNOTS_JWKS_URL or KNOTS_JWT_SECRET")
        if self.secret and not self.jwks_url and len(self.secret) < MIN_SECRET_CHARS:
            raise ValueError(f"KNOTS_JWT_SECRET must be at least {MIN_SECRET_CHARS} characters")
        if not self.audience:
            raise ValueError("KNOTS_JWT_AUDIENCE is required: a token for another service must not be accepted here")
        self.algorithms = algorithms or (list(ASYMMETRIC) if self.jwks_url else ["HS256"])
        self.leeway_s = leeway_s
        self._jwks = None

    def _key(self, token: str):
        if self.jwks_url:
            import jwt
            if self._jwks is None:
                self._jwks = jwt.PyJWKClient(self.jwks_url, cache_keys=True, lifespan=3600)
            return self._jwks.get_signing_key_from_jwt(token).key
        return self.secret

    def decode(self, token: str) -> dict | None:
        import jwt
        try:
            claims = jwt.decode(token, self._key(token), algorithms=self.algorithms, audience=self.audience,
                                issuer=self.issuer, leeway=self.leeway_s,
                                options={"require": ["exp", "sub"], "verify_iss": bool(self.issuer)})
        except Exception as e:                                    # noqa: BLE001  (every failure is "not valid")
            log.info("bearer token rejected: %s", type(e).__name__)
            return None
        return claims if claims.get("sub") else None

    async def verify_token(self, token: str):
        from mcp.server.auth.provider import AccessToken
        claims = await asyncio.to_thread(self.decode, token)       # the JWKS fetch is blocking network I/O
        if claims is None:
            return None
        aud = claims.get("aud")
        scopes = claims.get("scope", "")
        scopes = scopes.split() if isinstance(scopes, str) else list(scopes or [])
        return AccessToken(token=token, client_id=str(claims.get("client_id") or claims.get("azp") or ""),
                           scopes=scopes, expires_at=int(claims.get("exp") or time.time()),
                           resource=self.audience if self.audience in ([aud] if isinstance(aud, str) else aud or []) else None,
                           subject=str(claims["sub"]), claims=claims)
