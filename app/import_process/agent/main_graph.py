"""LangGraph orchestration for the document-ingestion pipeline."""

from app.core.logger import logger


def _load_graph_dependencies():
    """Load optional native dependencies at workflow-construction time."""
    from dotenv import load_dotenv
    from langgraph.graph import StateGraph, END
    from app.import_process.agent.state import ImportGraphState
    from app.import_process.agent.nodes.node_entry import node_entry
    from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
    from app.import_process.agent.nodes.node_md_img import node_md_img
    from app.import_process.agent.nodes.node_document_split import node_document_split
    from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
    from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
    from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
    load_dotenv()

    return {
        "StateGraph": StateGraph,
        "END": END,
        "ImportGraphState": ImportGraphState,
        "node_entry": node_entry,
        "node_pdf_to_md": node_pdf_to_md,
        "node_md_img": node_md_img,
        "node_document_split": node_document_split,
        "node_item_name_recognition": node_item_name_recognition,
        "node_bge_embedding": node_bge_embedding,
        "node_import_milvus": node_import_milvus,
    }

def _build_workflow():
    """Build the document-ingestion workflow after dependencies are available."""
    deps = _load_graph_dependencies()
    StateGraph = deps["StateGraph"]
    ImportGraphState = deps["ImportGraphState"]
    END = deps["END"]

    workflow = StateGraph(ImportGraphState)

    # Register the ingestion stages in execution order.
    workflow.add_node("node_entry", deps["node_entry"])  # Validate input and initialize state
    workflow.add_node("node_pdf_to_md", deps["node_pdf_to_md"])  # Convert PDF to Markdown
    workflow.add_node("node_md_img", deps["node_md_img"])  # Resolve document images
    workflow.add_node("node_document_split", deps["node_document_split"])  # Split document content
    workflow.add_node("node_item_name_recognition", deps["node_item_name_recognition"])  # Identify product entity
    workflow.add_node("node_bge_embedding", deps["node_bge_embedding"])  # Generate embeddings
    workflow.add_node("node_import_milvus", deps["node_import_milvus"])  # Persist vectors

    workflow.set_entry_point("node_entry")

    # Select the parser path from the input format flags.
    workflow.add_conditional_edges(
        "node_entry",
        route_after_entry,
        {
            "node_md_img": "node_md_img",
            "node_pdf_to_md": "node_pdf_to_md",
            END: END
        }
    )

    workflow.add_edge("node_pdf_to_md", "node_md_img")
    workflow.add_edge("node_md_img", "node_document_split")
    workflow.add_edge("node_document_split", "node_item_name_recognition")
    workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
    workflow.add_edge("node_bge_embedding", "node_import_milvus")
    workflow.add_edge("node_import_milvus", END)

    return workflow

def route_after_entry(state) -> str:
    """Select the parser branch from the input-format flags."""
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    else:
        from langgraph.graph import END
        return END

# Compile lazily so service startup does not initialize native dependencies.
_kb_import_app = None
_workflow = None


def get_kb_import_app():
    """Return the cached compiled ingestion graph, building it on first use."""
    global _kb_import_app
    global _workflow

    if _kb_import_app is None:
        logger.info("Building import workflow on first access")
        _workflow = _build_workflow()
        logger.info("Import workflow definition built; compiling")
        _kb_import_app = _workflow.compile()
        logger.info("Import workflow compiled")

    return _kb_import_app
