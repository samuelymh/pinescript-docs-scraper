"""LLM client wrapper for constructing prompts and calling OpenAI.

This module encapsulates prompt assembly, model selection logic, and a thin
wrapper around the OpenAI client so tests can mock the low-level calls.
"""
from typing import Dict, Any, List, Tuple, Optional
import logging

from server.config import get_config
from server.embed_client import init_openai_client
from server.utils import estimate_prompt_tokens, count_tokens

logger = logging.getLogger(__name__)


def build_system_prompt() -> str:
    """Return a consistent system prompt for PineScript assistant."""
    return (
        "You are an expert Pine Script assistant. Provide concise, correct Pine "
        "Script code examples and explain differences between versions when relevant. "
        "Always include provenance for code samples by listing filenames and similarity."
    )


def choose_model(prompt_tokens: int) -> str:
    """Choose primary or fallback model based on prompt token size.

    If the prompt is too large for the preferred model, return the fallback.
    """
    config = get_config()
    # Simple heuristic: prefer primary, but fall back if prompt exceeds budget
    if prompt_tokens > config.prompt_token_budget:
        logger.info("Prompt tokens %d exceed budget %d, selecting fallback model",
                    prompt_tokens, config.prompt_token_budget)
        return config.llm_model_fallback
    return config.llm_model_primary


def create_chat_completion(
    user_query: str,
    context_docs: List[Dict[str, Any]],
    temperature: float = 0.1,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Assemble prompt and call OpenAI to produce a response.

    Returns a dict with keys: `response`, `model`, `tokens` (dict of counts).
    """
    system_prompt = build_system_prompt()

    # Build context text from docs
    context_parts = [f"Source: {d['source_filename']}\n{d['content']}" for d in context_docs]

    # Compose parts used for token estimation: system, optional history, context, query
    parts: List[str] = [system_prompt]
    if conversation_history:
        for m in conversation_history:
            parts.append(m.get("content", ""))
    parts.extend(context_parts)
    parts.append(user_query)

    prompt_tokens = estimate_prompt_tokens(parts)

    # If prompt is over budget, trim oldest conversation history entries first
    config = get_config()
    budget = getattr(config, "prompt_token_budget", None)
    if budget is not None and prompt_tokens > budget and conversation_history:
        # Work on a mutable copy of the history
        trimmed_history = list(conversation_history)
        # Remove oldest entries until we are within budget or have removed all history
        while trimmed_history and prompt_tokens > budget:
            trimmed_history.pop(0)
            # rebuild parts and recompute tokens
            parts = [system_prompt] + [m.get("content", "") for m in trimmed_history] + context_parts + [user_query]
            prompt_tokens = estimate_prompt_tokens(parts)

        if prompt_tokens > budget:
            # Even after removing history we're still over budget; log and continue
            logger = logging.getLogger(__name__)
            logger.info("Prompt still exceeds token budget after trimming history: %d > %d", prompt_tokens, budget)
        else:
            # Replace conversation_history with the trimmed version for the messages payload later
            conversation_history = trimmed_history

    model = choose_model(prompt_tokens)
    client = init_openai_client()

    # Wrap the request in a try/except so tests can assert on request payload
    try:
        # Use the Responses API if available; fall back to chat completions shape.
        # Build the chat-style messages payload, including conversation history if present
        messages_payload = [
            {"role": "system", "content": system_prompt}
        ]
        if conversation_history:
            for m in conversation_history:
                role = m.get("role", "user")
                content = m.get("content", "")
                messages_payload.append({"role": role, "content": content})

        # Append a final user message containing the retrieved context + the current query
        messages_payload.append({"role": "user", "content": "\n\n".join(context_parts + [user_query])})

        resp = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            temperature=float(temperature),
            max_tokens=get_config().llm_max_completion_tokens
        )

        # Extract text and token usage if present
        text = ""
        tokens = {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens}

        if hasattr(resp, "choices") and resp.choices:
            first = resp.choices[0]
            msg = getattr(first, "message", None)
            if isinstance(msg, dict):
                text = msg.get("content")
            else:
                # Some SDKs expose a ChatCompletionMessage with `.content`
                text = getattr(msg, "content", None) or str(first)
            # Some SDKs include usage info
            usage = getattr(resp, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    tokens = {
                        "prompt": usage.get("prompt_tokens", prompt_tokens),
                        "completion": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", prompt_tokens)
                    }
                else:
                    # usage may be an object with attributes
                    tokens = {
                        "prompt": getattr(usage, "prompt_tokens", prompt_tokens),
                        "completion": getattr(usage, "completion_tokens", 0),
                        "total": getattr(usage, "total_tokens", prompt_tokens)
                    }

        # Build structured provenance sources from context_docs input where possible
        sources = []
        for d in context_docs:
            sources.append({
                "filename": d.get("source_filename") or d.get("source_filename", d.get("source", None)),
                "similarity_score": d.get("similarity_score") or d.get("similarity", None) or d.get("score", None),
                "excerpt": (d.get("content", "")[:200] if d.get("content") else d.get("excerpt", ""))
            })

        return {"response": text, "model": model, "tokens": tokens, "sources": sources}

    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise
