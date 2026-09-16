"""Assemble bounded conversation, memory, and evidence context for a query run."""

import re
import sys
import time
from typing import Dict, List, Any

from app.query_process.agent.state import QueryGraphState
from app.clients.mongo_history_utils import get_recent_messages
from app.clients.mongo_session_memory_utils import get_session_memory
from app.clients.mongo_product_memory_utils import get_product_memory, search_product_memory
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.eval_flag_utils import is_eval_enabled
from app.core.logger import logger


# Context budgets keep checkpoint and model payloads bounded.
MAX_HISTORY_MESSAGES = 6       # Maximum conversation messages included in context.
MAX_EVIDENCE_DOCS = 6          # Maximum evidence documents included in context.
MAX_CHARS_PER_MESSAGE = 500    # Maximum characters per conversation message.
MAX_CHARS_PER_DOC = 500        # Maximum characters per evidence preview.
MAX_ANSWER_SUMMARY_CHARS = 500 # Maximum characters in the answer summary.


CLARIFICATION_RESOLUTION_PATTERNS = [
    r"^(第)?[一二三四五12345]个$",
    r"^(第)?[一二三四五12345]种$",
    r"^选第?[一二三四五12345]个?$",
    r"^第一个$", r"^第二个$", r"^第三个$", r"^第四个$", r"^第五个$",
    r"^就是它$", r"^就这个$",
]

ORDINAL_MAP = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    "first": 0, "second": 1, "third": 2,
}


def _is_clarification_resolution(query: str) -> bool:
    """Return whether a query selects one of the pending candidates."""
    clean = (query or "").strip()
    return any(re.match(pattern, clean) for pattern in CLARIFICATION_RESOLUTION_PATTERNS)


def _resolve_candidate_selection(query: str, candidates: list) -> str | None:
    """Resolve an ordinal selection against the pending candidate list."""
    clean = (query or "").strip().lower()
    for word, idx in ORDINAL_MAP.items():
        if word in clean:
            if idx < len(candidates):
                return candidates[idx].get("name")
    if clean in {"就是它", "就这个"} and candidates:
        return candidates[0].get("name")
    return None


def _limit_messages(messages: list, max_count: int = MAX_HISTORY_MESSAGES) -> list:
    """Keep the latest messages within the configured character budget."""
    result = []
    for msg in (messages or [])[-max_count:]:
        truncated = {
            "role": msg.get("role", ""),
            "text": (msg.get("text") or "")[:MAX_CHARS_PER_MESSAGE],
            "item_names": msg.get("item_names") or [],
        }
        result.append(truncated)
    return result


def _sanitize_for_checkpoint(value):
    """Convert external identifiers into checkpoint-serializable values."""
    if isinstance(value, list):
        return [_sanitize_for_checkpoint(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_for_checkpoint(item) for key, item in value.items()}
    if value.__class__.__name__ == "ObjectId":
        return str(value)
    return value


def _limit_evidence(docs: list, max_count: int = MAX_EVIDENCE_DOCS) -> list:
    """Keep a bounded summary of the most recent evidence documents."""
    result = []
    for doc in (docs or [])[:max_count]:
        d = {
            "chunk_id": doc.get("chunk_id"),
            "title": doc.get("title") or doc.get("file_title") or "",
            "canonical_name": doc.get("canonical_name") or "",
            "text_preview": (doc.get("text_preview") or "")[:MAX_CHARS_PER_DOC],
            "score": doc.get("score"),
        }
        result.append(d)
    return result


def _truncate(text: str, max_chars: int = MAX_ANSWER_SUMMARY_CHARS) -> str:
    """Truncate text to the configured character limit."""
    return (text or "")[:max_chars]


def _item_name_text(item) -> str:
    if isinstance(item, dict):
        return (
            item.get("canonical_name")
            or item.get("name")
            or item.get("alias_name")
            or item.get("item_name")
            or ""
        )
    return str(item or "")


def _normalize_confirmed_items(items: list) -> list:
    normalized = []
    for item in items or []:
        if isinstance(item, dict):
            canonical_name = _item_name_text(item)
            if canonical_name:
                normalized.append({
                    "product_id": item.get("product_id", ""),
                    "canonical_name": canonical_name,
                    "alias_name": item.get("alias_name", ""),
                    "brand": item.get("brand", ""),
                    "model_code": item.get("model_code", ""),
                    "category": item.get("category", ""),
                    "confidence": item.get("confidence", item.get("score", 1.0)),
                    "source": item.get("source", "session_memory"),
                })
            continue
        name = str(item or "").strip()
        if name:
            normalized.append({
                "product_id": "",
                "canonical_name": name,
                "alias_name": "",
                "brand": "",
                "model_code": "",
                "category": "",
                "confidence": 1.0,
                "source": "session_memory",
            })
    return normalized


def _load_product_memory_hints(query: str, confirmed_items: list) -> dict:
    """Load product memory by confirmed item first, then fall back to alias search."""
    if is_eval_enabled("EVAL_DISABLE_PRODUCT_MEMORY"):
        return {
            "matched": False,
            "memories": [],
            "aliases": [],
            "doc_ids": [],
            "common_questions": [],
        }

    memories = []
    for item in confirmed_items or []:
        name = _item_name_text(item)
        if not name:
            continue
        mem = get_product_memory(name)
        if mem:
            memories.append(mem)

    if not memories and query:
        memories = search_product_memory(query, limit=3)

    common_questions = []
    aliases = []
    doc_ids = []
    for mem in memories or []:
        aliases.extend(mem.get("aliases") or [])
        doc_ids.extend(mem.get("doc_ids") or [])
        common_questions.extend(mem.get("common_questions") or [])

    common_questions.sort(key=lambda x: x.get("frequency", 0), reverse=True)
    return {
        "matched": bool(memories),
        "memories": memories[:3],
        "aliases": list(dict.fromkeys(aliases))[:20],
        "doc_ids": list(dict.fromkeys(doc_ids))[:20],
        "common_questions": common_questions[:10],
    }


def node_build_context(state: QueryGraphState) -> QueryGraphState:
    """Load session context, resolve pending clarification, and update state."""
    node_name = sys._getframe().f_code.co_name
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    logger.info("node_build_context started")
    add_running_task(session_id, node_name, is_stream)

    query = (state.get("original_query") or "").strip()

    if is_eval_enabled("EVAL_DISABLE_CONTEXT_BUNDLE"):
        state["history"] = []
        state["context_bundle"] = {
            "history_messages": [],
            "confirmed_items": [],
            "recent_evidence_docs": [],
            "last_answer_summary": "",
            "pending_clarification": None,
            "last_route_meta": {},
            "assembled_at": time.time(),
            "disabled_by_eval": True,
        }
        state["product_memory_hints"] = {}
        logger.info("node_build_context skipped by EVAL_DISABLE_CONTEXT_BUNDLE")
        add_done_task(session_id, node_name, is_stream)
        return state

    mem = {} if is_eval_enabled("EVAL_DISABLE_SESSION_MEMORY") else (get_session_memory(session_id) or {})
    logger.info(
        f"node_build_context: session memory loaded, "
        f"confirmed_items={mem.get('last_confirmed_items')}, "
        f"has_pending_clarification={bool(mem.get('pending_clarification'))}"
    )

    history = _sanitize_for_checkpoint(get_recent_messages(session_id, limit=10))
    state["history"] = history
    logger.info(f"node_build_context: history_messages={len(history)}")

    confirmed_items = _normalize_confirmed_items(mem.get("last_confirmed_items") or [])
    bundle = {
        "history_messages": _limit_messages(history),
        "confirmed_items": confirmed_items,
        "recent_evidence_docs": _limit_evidence(mem.get("last_evidence_docs") or []),
        "last_answer_summary": _truncate(mem.get("last_answer_summary") or ""),
        "pending_clarification": mem.get("pending_clarification"),
        "last_route_meta": mem.get("last_route_meta") or {},
        "assembled_at": time.time(),
    }
    state["context_bundle"] = bundle
    state["product_memory_hints"] = _load_product_memory_hints(query, confirmed_items)

    pending = bundle.get("pending_clarification")
    if pending and _is_clarification_resolution(query):
        candidates = pending.get("candidates") or []
        resolved = _resolve_candidate_selection(query, candidates)
        if resolved:
            logger.info(f"node_build_context: clarification resolved, product={resolved}")
            restored_query = (
                pending.get("original_query")
                or pending.get("query")
                or state.get("original_query")
                or query
            )
            state["clarification_resolved"] = True
            state["item_names"] = [resolved]
            state["original_query"] = restored_query
            state["rewritten_query"] = restored_query
            bundle["resolved_clarification_query"] = query
            bundle["restored_original_query"] = restored_query
        else:
            logger.warning(f"node_build_context: clarification selection unresolved, query={query}, candidates={candidates}")

    logger.info(
        f"node_build_context completed: history={len(bundle['history_messages'])}, "
        f"confirmed_items={len(bundle['confirmed_items'])}, "
        f"evidence={len(bundle['recent_evidence_docs'])}, "
        f"pending_clarification={bool(bundle['pending_clarification'])}"
    )
    add_done_task(session_id, node_name, is_stream)
    return state
