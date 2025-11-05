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
from server.auth import verify_admin_key, verify_jwt_token, get_rate_limit_key
from slowapi import Limiter
from server.utils import setup_logging
from server.supabase_client import get_document_stats
from server.ingest import index_documents
import asyncio

# Retrieval / LLM wiring
from fastapi.concurrency import run_in_threadpool
from server.embed_client import generate_single_embedding
from server.retriever import hybrid_search, assemble_context
from server.llm_client import create_chat_completion

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

# Create a rate limiter using per-user key function and attach to app
limiter = Limiter(key_func=get_rate_limit_key)
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
    
    # Get document stats from Supabase
    doc_stats = get_document_stats()
    
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
            "documents_count": doc_stats["documents_count"],
            "last_index_time": doc_stats["last_index_time"]
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

    try:
        # 1) Generate query embedding (run in threadpool to avoid blocking)
        query_embedding = await run_in_threadpool(generate_single_embedding, chat_request.query)

        # 2) Retrieve relevant documents (hybrid search)
        retrieved = await run_in_threadpool(
            hybrid_search, query_embedding, chat_request.query, chat_request.max_context_docs
        )

        # 3) Assemble context to respect token budgets
        selected_docs, prompt_tokens = await run_in_threadpool(assemble_context, retrieved)

        # Convert docs to plain dicts for LLM client
        context_docs = [d.model_dump() for d in selected_docs]

        # 4) Call LLM to create chat completion
        llm_result = await run_in_threadpool(
            create_chat_completion, chat_request.query, context_docs, float(chat_request.temperature)
        )

        # 5) Build response model with provenance
        resp_text = llm_result.get("response", "") or ""
        model_used = llm_result.get("model", "") or ""
        tokens = llm_result.get("tokens", {})
        sources = llm_result.get("sources", [])

        return ChatResponse(
            response=resp_text,
            sources=sources,
            tokens_used=tokens,
            model=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat request failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/internal/index")
@limiter.limit("5/hour")
async def trigger_indexing(
    request: Request,
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
    full: bool = False,
    background: bool = False
):
    """Trigger document indexing (admin only).
    
    Scans processed markdown files, detects changes, generates embeddings,
    and upserts to Supabase with manifest tracking.
    
    Args:
        request: FastAPI request object (required by slowapi)
        x_admin_key: Admin API key from header
        full: If True, perform full re-index; otherwise incremental
    
    Returns:
        Indexing results with counts and statistics
    """
    verify_admin_key(x_admin_key)
    
    logger.info(f"Indexing triggered: full={full}")
    
    try:
        if background:
            # Schedule indexing asynchronously and return immediately
            logger.info("Scheduling background indexing task")
            asyncio.create_task(index_documents(full_reindex=full))
            return {"started": True, "background": True}

        # Run indexing pipeline synchronously
        results = await index_documents(full_reindex=full)

        if results["success"]:
            logger.info(f"Indexing completed successfully: {results}")
            return results
        else:
            logger.error(f"Indexing failed: {results.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Indexing failed: {results.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Indexing failed with exception: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
