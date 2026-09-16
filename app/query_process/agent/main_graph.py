"""LangGraph orchestration for the query pipeline."""

from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
from app.utils.eval_flag_utils import is_eval_enabled
from app.query_process.agent.nodes.node_build_context import node_build_context
from app.query_process.agent.nodes.node_query_intent import node_query_intent
from app.query_process.agent.nodes.node_direct_answer import node_direct_answer
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_doc_need_router import node_doc_need_router
from app.query_process.agent.nodes.node_retrieval_plan import node_retrieval_plan
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_self_eval import node_self_eval
from app.query_process.agent.nodes.node_retry_web import node_retry_web
from app.clients.run_state_utils import update_run_node, checkpoint_run


def _tracked_node(node_name, func):
    def _wrapped(state):
        run_id = state.get("run_id") if isinstance(state, dict) else ""
        if run_id:
            update_run_node(run_id, node_name)
        result = func(state)
        if run_id:
            checkpoint_run(run_id)
        return result
    return _wrapped

# Build the graph once during service startup.
builder = StateGraph(QueryGraphState)

# Register business nodes and explicit fan-out/fan-in markers.
builder.add_node("node_build_context", _tracked_node("node_build_context", node_build_context))  # Context assembly
builder.add_node("node_query_intent", _tracked_node("node_query_intent", node_query_intent))  # Intent classification
builder.add_node("node_direct_answer", _tracked_node("node_direct_answer", node_direct_answer))  # Direct response
builder.add_node("node_item_name_confirm", _tracked_node("node_item_name_confirm", node_item_name_confirm))  # Product resolution
builder.add_node("node_doc_need_router", _tracked_node("node_doc_need_router", node_doc_need_router))  # Retrieval routing
builder.add_node("node_retrieval_plan", _tracked_node("node_retrieval_plan", node_retrieval_plan))  # Retrieval planning
builder.add_node("node_multi_search", _tracked_node("node_multi_search", lambda x: x))  # Fan-out marker
builder.add_node("node_search_embedding", _tracked_node("node_search_embedding", node_search_embedding))  # Dense vector retrieval
builder.add_node("node_search_embedding_hyde", _tracked_node("node_search_embedding_hyde", node_search_embedding_hyde))
builder.add_node("node_web_search_mcp", _tracked_node("node_web_search_mcp", node_web_search_mcp))
builder.add_node("node_join", _tracked_node("node_join", lambda x: {}))  # Fan-in marker
builder.add_node("node_rrf", _tracked_node("node_rrf", node_rrf))  # Rank fusion
builder.add_node("node_rerank", _tracked_node("node_rerank", node_rerank))  # Cross-encoder rerank
builder.add_node("node_self_eval", _tracked_node("node_self_eval", node_self_eval))  # Retrieval sufficiency
builder.add_node("node_retry_web", _tracked_node("node_retry_web", node_retry_web))  # Web retry
builder.add_node("node_answer_output", _tracked_node("node_answer_output", node_answer_output))  # Answer generation

builder.set_entry_point("node_build_context")
builder.add_edge("node_build_context", "node_query_intent")


def route_after_query_intent(state: QueryGraphState):
    if state.get("query_route") == "product_doc_qa":
        return "node_item_name_confirm"
    return "node_direct_answer"


def route_after_item_confirm(state: QueryGraphState):
    """Route clarification or rejection responses directly to answer output."""
    if state.get("answer"):
        return "node_answer_output"
    return "node_doc_need_router"


def route_after_doc_need_router(state: QueryGraphState):
    """Route retrieval requests to planning and other outcomes to output."""
    if state.get("doc_need_route") != "need_retrieval":
        return "node_answer_output"
    return "node_retrieval_plan"


def route_after_self_eval(state: QueryGraphState):
    """Route based on evidence sufficiency and the bounded retry budget."""
    status = state.get("self_eval_status", "")
    retry_count = int(state.get("self_eval_retry_count", 0))
    if status == "sufficient":
        return "node_answer_output"
    if (
        status == "insufficient_retry"
        and retry_count < 1
        and not state.get("_no_web_override")
        and not is_eval_enabled("EVAL_DISABLE_SELF_EVAL_RETRY")
        and not is_eval_enabled("EVAL_DISABLE_WEB")
    ):
        return "node_retry_web"
    return "node_answer_output"


builder.add_conditional_edges(
    "node_query_intent",
    route_after_query_intent
)
builder.add_edge("node_direct_answer", "node_answer_output")

builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm
)

builder.add_conditional_edges(
    "node_doc_need_router",
    route_after_doc_need_router
)
builder.add_edge("node_retrieval_plan", "node_multi_search")

# Execute the retrieval branches in parallel.
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")

# Merge retrieval branches before ranking.
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")

# Rank, rerank, evaluate, and optionally retry once before answering.
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_self_eval")
builder.add_conditional_edges("node_self_eval", route_after_self_eval)
builder.add_edge("node_retry_web", "node_rrf")
builder.add_edge("node_answer_output", END)

# Compile with the configured checkpointer when available.
from app.conf.checkpoint_config import get_checkpointer

checkpointer = get_checkpointer()
if checkpointer:
    query_app = builder.compile(checkpointer=checkpointer)
else:
    query_app = builder.compile()
