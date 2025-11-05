"""Authentication and rate limiting for PineScript RAG Server.

Provides JWT verification using python-jose and rate limiting middleware.
Supports HS256 via a shared secret (`JWT_SECRET`) or RS256 via a JWKS URL.
"""
from typing import Optional, Dict
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from server.config import get_config
import logging
import requests
import time

from jose import jwt, jwk
from jose.exceptions import JWTError

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Simple JWKS cache
_JWKS_CACHE: Dict[str, any] = {"jwks": None, "fetched_at": 0}
_JWKS_TTL = 60 * 60  # 1 hour


def verify_admin_key(api_key: Optional[str]) -> bool:
    """Verify admin API key for /internal/* endpoints.

    Raises HTTPException if invalid or not configured.
    """
    config = get_config()

    if not config.admin_api_key:
        logger.warning("ADMIN_API_KEY not configured, denying access")
        raise HTTPException(status_code=503, detail="Admin access not configured")

    if not api_key or api_key != config.admin_api_key:
        logger.warning("Invalid admin API key attempt")
        raise HTTPException(status_code=401, detail="Invalid admin API key")

    return True


def _fetch_jwks(url: str) -> dict:
    now = int(time.time())
    if _JWKS_CACHE["jwks"] and (now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL):
        return _JWKS_CACHE["jwks"]

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        jwks = resp.json()
        _JWKS_CACHE["jwks"] = jwks
        _JWKS_CACHE["fetched_at"] = now
        return jwks
    except Exception as e:
        logger.error("Failed to fetch JWKS from %s: %s", url, e)
        raise HTTPException(status_code=503, detail="Failed to fetch JWKS")


async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, any]:
    """Verify JWT token for authenticated endpoints.

    Supports HS256 (shared secret) and RS256 via JWKS. Returns decoded token
    payload on success, raises HTTPException on failure.
    """
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    config = get_config()

    # Prefer HS256 with configured secret
    try:
        if config.jwt_secret and config.jwt_algorithm.upper().startswith("HS"):
            # Use shared secret
            payload = jwt.decode(
                token,
                config.jwt_secret,
                algorithms=[config.jwt_algorithm],
                audience=config.jwt_audience or None,
                issuer=config.jwt_issuer or None,
            )
            logger.debug("JWT verified with HS secret: sub=%s", payload.get("sub"))
            return payload

        # Otherwise, try JWKS flow for RS256
        if config.jwks_url:
            # Get header to find kid
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            jwks = _fetch_jwks(config.jwks_url)
            keys = jwks.get("keys", [])
            key = None
            for k in keys:
                if k.get("kid") == kid:
                    key = k
                    break

            if not key:
                logger.error("No matching JWK for kid=%s", kid)
                raise HTTPException(status_code=401, detail="Invalid token")

            public_key = jwk.construct(key)
            pem = public_key.to_pem().decode() if hasattr(public_key, "to_pem") else None

            # If jwk.construct provides PEM-like bytes, use them; otherwise pass the JWK dict
            decode_key = pem or key

            payload = jwt.decode(
                token,
                decode_key,
                algorithms=[config.jwt_algorithm],
                audience=config.jwt_audience or None,
                issuer=config.jwt_issuer or None,
            )
            logger.debug("JWT verified via JWKS: sub=%s", payload.get("sub"))
            return payload

        # No verification method configured
        logger.warning("No JWT verification configured (no secret or jwks_url)")
        raise HTTPException(status_code=503, detail="Authentication not configured")

    except JWTError as e:
        logger.warning("JWT verification failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid authentication token")


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key from request.

    Uses remote address by default. In production this should use `sub` from
    JWT payload to provide per-user limits.
    """
    # Prefer per-user rate limiting using verified JWT `sub` when possible.
    try:
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(None, 1)[1].strip()
            config = get_config()

            # Try HS verification first if secret is configured
            if config.jwt_secret and config.jwt_algorithm.upper().startswith("HS"):
                try:
                    payload = jwt.decode(
                        token,
                        config.jwt_secret,
                        algorithms=[config.jwt_algorithm],
                        audience=config.jwt_audience or None,
                        issuer=config.jwt_issuer or None,
                    )
                    sub = payload.get("sub")
                    if sub:
                        return f"user:{sub}"
                except Exception:
                    # Fall back to remote address if verification fails
                    pass

            # Try JWKS/RS verification if configured
            if config.jwks_url:
                try:
                    header = jwt.get_unverified_header(token)
                    kid = header.get("kid")
                    jwks = _fetch_jwks(config.jwks_url)
                    keys = jwks.get("keys", [])
                    key = None
                    for k in keys:
                        if k.get("kid") == kid:
                            key = k
                            break

                    if key:
                        public_key = jwk.construct(key)
                        pem = public_key.to_pem().decode() if hasattr(public_key, "to_pem") else None
                        decode_key = pem or key
                        payload = jwt.decode(
                            token,
                            decode_key,
                            algorithms=[config.jwt_algorithm],
                            audience=config.jwt_audience or None,
                            issuer=config.jwt_issuer or None,
                        )
                        sub = payload.get("sub")
                        if sub:
                            return f"user:{sub}"
                except Exception:
                    pass

    except Exception:
        # Conservative fallback on any unexpected error
        pass

    # Default to remote address
    return get_remote_address(request)
