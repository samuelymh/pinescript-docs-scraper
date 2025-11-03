"""FastAPI application for PineScript RAG Server.

Provides REST API endpoints for document indexing and chat functionality.
"""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import logging

from server.config import get_config
from server.models import ChatRequest, ChatResponse
from server.auth import limiter, verify_admin_key, verify_jwt_token
from server.utils import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    config = get_config()
    setup_logging(config.log_level)
    logger.info("PineScript RAG Server starting up")
    logger.info(f"Configuration: model={config.llm_model}, embedding={config.embedding_model}")
    
    yield
    
    # Shutdown
    logger.info("PineScript RAG Server shutting down")


# Initialize FastAPI app
app = FastAPI(
    title="PineScript RAG Server",
    description="Retrieval-Augmented Generation API for PineScript documentation",
    version="0.1.0",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
@limiter.limit("30/minute")
async def get_status(request: Request):
    """Health and status endpoint.
    
    Args:
        request: FastAPI request object (required by slowapi)
    
    Returns:
        Status information including health, configuration, and indexing stats
    """
    config = get_config()
    
    return {
        "status": "healthy",
        "version": "0.1.0",
        "configuration": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "max_context_docs": config.max_context_docs,
            "chunk_token_threshold": config.chunk_token_threshold
        },
        "indexing": {
            "documents_count": None,  # TODO: Query from Supabase in Step 2
            "last_index_time": None   # TODO: Query from manifest table in Step 2
        }
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(verify_jwt_token)
):
    """Chat endpoint for PineScript queries.
    
    Note: This is a placeholder returning 501. Full implementation in Step 4.
    
    Args:
        request: Starlette request object (required by slowapi)
        chat_request: Chat request with query and options
        user: Authenticated user from JWT token
    
    Returns:
        Chat response with generated code and sources
    """
    logger.info(f"Chat request from user {user.get('sub')}: {chat_request.query[:50]}...")
    
    # TODO: Implement in Step 4
    # 1. Generate query embedding
    # 2. Retrieve relevant documents
    # 3. Assemble context
    # 4. Call LLM
    # 5. Post-process and return response
    
    raise HTTPException(
        status_code=501,
        detail="Chat endpoint not yet implemented. Will be available in Step 4."
    )


@app.post("/internal/index")
@limiter.limit("5/hour")
async def trigger_indexing(
    request: Request,
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
    full: bool = False
):
    """Trigger document indexing (admin only).
    
    Note: This is a placeholder returning 501. Full implementation in Step 2.
    
    Args:
        request: FastAPI request object (required by slowapi)
        x_admin_key: Admin API key from header
        full: If True, perform full re-index; otherwise incremental
    
    Returns:
        Indexing results with counts
    """
    verify_admin_key(x_admin_key)
    
    logger.info(f"Indexing triggered: full={full}")
    
    # TODO: Implement in Step 2
    # 1. Scan processed files
    # 2. Check manifest for changes
    # 3. Parse and chunk documents
    # 4. Generate embeddings
    # 5. Upsert to Supabase
    
    raise HTTPException(
        status_code=501,
        detail="Indexing endpoint not yet implemented. Will be available in Step 2."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
