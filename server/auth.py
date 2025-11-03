"""Authentication and rate limiting for PineScript RAG Server.

Provides JWT verification stubs and rate limiting middleware.
"""
from typing import Optional
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from server.config import get_config
import logging

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


def verify_admin_key(api_key: Optional[str]) -> bool:
    """Verify admin API key for /internal/* endpoints.
    
    Args:
        api_key: API key from request header
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: If key is invalid or missing
    """
    config = get_config()
    
    if not config.admin_api_key:
        logger.warning("ADMIN_API_KEY not configured, denying access")
        raise HTTPException(status_code=503, detail="Admin access not configured")
    
    if not api_key or api_key != config.admin_api_key:
        logger.warning("Invalid admin API key attempt")
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    
    return True


async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Verify JWT token for authenticated endpoints.
    
    Note: This is a stub implementation. Production should verify JWT signature.
    
    Args:
        credentials: Bearer token credentials
    
    Returns:
        Decoded token payload (stub returns empty dict)
    
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    
    # TODO: Implement proper JWT verification with python-jose
    # For now, accept any non-empty token as valid (development only)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    logger.debug(f"JWT token verified (stub): {token[:10]}...")
    
    # Return stub payload
    return {"sub": "user_id", "authenticated": True}


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key from request.
    
    Uses remote address as default. In production, should use user_id from JWT.
    
    Args:
        request: FastAPI request object
    
    Returns:
        Rate limit key string
    """
    # TODO: Extract user_id from JWT token for per-user rate limiting
    return get_remote_address(request)
