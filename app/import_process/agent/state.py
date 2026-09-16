"""State contract shared by document-ingestion workflow nodes."""

from typing import TypedDict
import copy

class ImportGraphState(TypedDict):
    """Typed state exchanged by the document-ingestion workflow."""
    task_id: str          # Task identifier used for trace correlation

    # Input routing
    is_md_read_enabled: bool   # Whether Markdown ingestion is enabled
    is_pdf_read_enabled: bool  # Whether PDF ingestion is enabled

    # File locations and identity
    local_dir: str        # Working or output directory
    local_file_path: str  # Original input path
    doc_id: str           # Content-derived document identifier
    file_title: str       # Filename without extension
    pdf_path: str         # PDF input path
    md_path: str          # Normalized Markdown path

    # Normalized content
    md_content: str       # Full Markdown content
    chunks: list          # Chunk records with metadata
    item_name: str        # Resolved product name

    # Vector-store payload
    embeddings_content: list # Records prepared for Milvus


# Empty defaults keep state construction deterministic.
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "doc_id": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """Create an isolated ingestion state and apply optional overrides."""
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state

def get_default_state() -> ImportGraphState:
    """Return an isolated copy of the default ingestion state."""
    return copy.deepcopy(graph_default_state)
