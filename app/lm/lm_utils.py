"""Factory and cache for OpenAI-compatible chat model clients."""

from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional
import os

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.lm.request_options import non_thinking_body

# Cache clients by model and response mode to avoid repeated initialization.
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """Return a cached chat client configured for an OpenAI-compatible endpoint."""
    target_model = model or lm_config.llm_model or "qwen3-32b"
    cache_key = (target_model, json_mode)

    if cache_key in _llm_client_cache:
        logger.debug(f"LLM client cache hit: model={target_model}, json_mode={json_mode}")
        return _llm_client_cache[cache_key]

    if not lm_config.api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if not lm_config.base_url:
        raise ValueError("OPENAI_BASE_URL or OPENAI_API_BASE is required")
    logger.info(f"Initializing LLM client: model={target_model}, json_mode={json_mode}")

    extra_body = non_thinking_body(lm_config.base_url)
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("LLM client configured for JSON responses")

    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )
    except LangChainException as e:
        raise Exception(f"Failed to initialize LLM client for {target_model}: {e}") from e

    _llm_client_cache[cache_key] = llm_client
    logger.info(f"LLM client initialized and cached: model={target_model}, json_mode={json_mode}")

    return llm_client
