"""Mint a bearer token for testing the MCP endpoint yourself (only for the shared-secret mode, KNOTS_JWT_SECRET).

    KNOTS_JWT_SECRET=... KNOTS_JWT_AUDIENCE=https://your-host/mcp python scripts/mint_token.py --sub me --days 7

The token is a login to your server: treat it like a password. For real listeners use an identity provider
(KNOTS_JWKS_URL) instead of minting tokens."""
import argparse
import os
import sys
import time

try:
    import jwt
except ImportError:
    sys.exit("pip install 'pyjwt[crypto]'")

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True, help="who the token is for (becomes the person's identity)")
ap.add_argument("--days", type=float, default=7)
args = ap.parse_args()
secret, aud = os.environ.get("KNOTS_JWT_SECRET", ""), os.environ.get("KNOTS_JWT_AUDIENCE", "")
if len(secret) < 32 or not aud:
    sys.exit("set KNOTS_JWT_SECRET (32+ characters) and KNOTS_JWT_AUDIENCE")
claims = {"sub": args.sub, "aud": aud, "exp": int(time.time() + args.days * 86400)}
if os.environ.get("KNOTS_JWT_ISSUER"):
    claims["iss"] = os.environ["KNOTS_JWT_ISSUER"]
print(jwt.encode(claims, secret, algorithm="HS256"))
