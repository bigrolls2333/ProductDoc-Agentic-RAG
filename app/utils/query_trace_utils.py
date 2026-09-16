"""Build and persist compact traces for query observability and evaluation."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.core.logger import logger


DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[2] / "logs" / "query_trace.jsonl"


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _summarize_docs(docs: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    summary = []
    for doc in (docs or [])[:limit]:
        if not isinstance(doc, dict):
            continue
        summary.append({
            "chunk_id": doc.get("chunk_id") or doc.get("doc_id") or doc.get("id"),
            "source": doc.get("source") or "",
            "title": doc.get("title") or doc.get("item_name") or "",
            "score": doc.get("score"),
        })
    return summary


def _resolve_trace_path() -> Path:
    configured_path = os.getenv("QUERY_TRACE_PATH", "").strip()
    if configured_path:
        return Path(configured_path)
    return DEFAULT_TRACE_PATH


def build_query_trace(state: Dict[str, Any], status: str = "completed", error: str = "") -> Dict[str, Any]:
    trace_started_at = state.get("trace_started_at")
    duration_ms = None
    if isinstance(trace_started_at, (int, float)):
        duration_ms = int((time.time() - trace_started_at) * 1000)

    embedding_chunks = state.get("embedding_chunks") or []
    hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []
    web_search_docs = state.get("web_search_docs") or []
    rrf_chunks = state.get("rrf_chunks") or []
    reranked_docs = state.get("reranked_docs") or []
    answer = state.get("answer") or ""

    trace = {
        "trace_version": "agent_trace_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "session_id": state.get("session_id", ""),
        "original_query": state.get("original_query", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "item_names": state.get("item_names") or [],
        "query_route": state.get("query_route", ""),
        "doc_need_route": state.get("doc_need_route", ""),
        "retrieval_plan": state.get("retrieval_plan") or {},
        "counts": {
            "embedding": _safe_len(embedding_chunks),
            "hyde": _safe_len(hyde_embedding_chunks),
            "web": _safe_len(web_search_docs),
            "rrf": _safe_len(rrf_chunks),
            "rerank": _safe_len(reranked_docs),
        },
        "top_reranked_docs": _summarize_docs(reranked_docs),
        "answer_length": len(answer),
        "answer_preview": answer[:300],
        "duration_ms": duration_ms,

        "agent_trace": {
            "context_bundle_summary": _build_context_bundle_summary(state),
            "planner": {
                "query_route": state.get("query_route", ""),
                "doc_need_route": state.get("doc_need_route", ""),
                "doc_need_reason": state.get("doc_need_reason", ""),
                "doc_need_confidence": state.get("doc_need_confidence", 0),
                "retrieval_plan": state.get("retrieval_plan") or {},
            },
            "tool_calls": {
                "embedding": _build_tool_call_info(
                    True, _safe_len(embedding_chunks)
                ),
                "hyde": _build_tool_call_info(
                    (state.get("retrieval_plan") or {}).get("use_hyde", False),
                    _safe_len(hyde_embedding_chunks),
                ),
                "web": _build_tool_call_info(
                    (state.get("retrieval_plan") or {}).get("use_web", False),
                    _safe_len(web_search_docs),
                ),
            },
            "self_eval": {
                "status": state.get("self_eval_status", ""),
                "retry_count": state.get("self_eval_retry_count", 0),
                "detail": state.get("self_eval_detail") or {},
            },
            "checkpoint": {
                "enabled": bool(state.get("checkpoint_enabled", True)),
                "backend": "sqlite" if state.get("checkpoint_enabled", True) else "none",
                "resumed": bool(state.get("resume_from_checkpoint", False)),
                "thread_id": state.get("thread_id") or state.get("session_id", ""),
                "run_id": state.get("run_id", ""),
            },
            "clarification": {
                "resolved": bool(state.get("clarification_resolved", False)),
                "has_pending": bool(
                    (state.get("context_bundle") or {}).get("pending_clarification")
                ),
            },
        },
    }
    return trace


def _build_context_bundle_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract bounded context-bundle metadata without persisting full content."""
    bundle = state.get("context_bundle") or {}
    return {
        "history_msg_count": _safe_len(bundle.get("history_messages") or []),
        "confirmed_items": bundle.get("confirmed_items") or [],
        "evidence_doc_count": _safe_len(bundle.get("recent_evidence_docs") or []),
        "has_pending_clarification": bool(bundle.get("pending_clarification")),
        "has_last_answer_summary": bool(bundle.get("last_answer_summary")),
    }


def _build_tool_call_info(enabled: bool, result_count: int) -> Dict[str, Any]:
    """Build a compact tool-call record for the trace."""
    return {
        "enabled": bool(enabled),
        "result_count": result_count,
    }


def append_query_trace(state: Dict[str, Any], status: str = "completed", error: str = "") -> None:
    try:
        trace_path = _resolve_trace_path()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = build_query_trace(state, status=status, error=error)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        logger.info(f"Query trace written: {trace_path}")
    except Exception as e:
        logger.error(f"Query trace write failed: {e}", exc_info=True)
