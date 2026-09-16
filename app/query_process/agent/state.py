"""State contract shared by the query graph nodes."""

from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """Typed state exchanged by the query workflow."""
    session_id: str  # Stable conversation identifier
    original_query: str  # User-provided query

    # Retrieval results
    embedding_chunks: list  # Dense/sparse retrieval results
    hyde_embedding_chunks: list  # HyDE retrieval results
    web_search_docs: list  # Web search results

    # Ranking results
    rrf_chunks: list  # RRF-fused chunks
    reranked_docs: list  # Final reranked Top-K documents
    rerank_trace: dict  # Rerank and truncation summary

    # Generation fields
    prompt: str  # Rendered answer prompt
    answer: str  # Final answer

    # Routing and context
    item_names: List[str]  # Resolved product names
    rewritten_query: str  # Normalized query
    history: list  # Conversation history
    is_stream: bool  # Whether the response is streamed
    query_route: str  # product_doc_qa / direct_answer / clarification / out_of_scope
    direct_answer: str  # Direct response from intent routing
    doc_need_route: str  # need_retrieval / direct_answer
    doc_need_reason: str  # Routing rationale
    doc_need_confidence: float  # Routing confidence
    retrieval_plan: dict  # Enabled retrieval branches
    context_evidence_docs: list  # Evidence reused from prior turns
    trace_started_at: float  # Trace start timestamp

    # Evidence evaluation and bounded retry
    self_eval_status: str  # "" | "sufficient" | "insufficient_retry" | "needs_clarification"
    self_eval_retry_count: int  # Bounded web retry count
    self_eval_detail: dict  # Structured evaluation result

    # Memory, checkpoint, and run metadata
    context_bundle: dict  # Assembled conversation and evidence context
    thread_id: str  # LangGraph thread identifier
    checkpoint_enabled: bool  # Whether checkpoint persistence is enabled
    resume_from_checkpoint: bool  # Whether this invocation resumes state
    product_memory_hints: dict  # Product-memory signals for query expansion
    clarification_resolved: bool  # Whether the pending clarification was resolved
    pending_clarification_created: bool  # Whether this turn created a clarification
    run_id: str  # Run metadata identifier


# Empty defaults keep state construction deterministic.
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "rerank_trace": {},
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "query_route": "",
    "direct_answer": "",
    "doc_need_route": "",
    "doc_need_reason": "",
    "doc_need_confidence": 0.0,
    "retrieval_plan": {},
    "context_evidence_docs": [],
    "trace_started_at": 0.0,
    "self_eval_status": "",
    "self_eval_retry_count": 0,
    "self_eval_detail": {},
    "context_bundle": {},
    "thread_id": "",
    "checkpoint_enabled": False,
    "resume_from_checkpoint": False,
    "product_memory_hints": {},
    "clarification_resolved": False,
    "pending_clarification_created": False,
    "run_id": "",
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """Create an isolated query state and apply optional field overrides."""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """Return an isolated copy of the default query state."""
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """Copy a query state without sharing mutable values with the source."""
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state
