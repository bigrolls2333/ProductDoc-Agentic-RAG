"""Persist embedded document chunks in Milvus with stable chunk identifiers."""

import sys
import hashlib
from typing import List, Dict, Any
from pymilvus import DataType
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config

# Resolve the collection name from the runtime configuration.
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

REQUIRED_SCHEMA_FIELDS = {
    "chunk_id",
    "doc_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    "canonical_name",
    "product_id",
    "brand",
    "category",
    "model_code",
    "sparse_vector",
    "dense_vector",
}
def _build_chunk_id(chunk: Dict[str, Any]) -> str:
    doc_id = str(chunk.get("doc_id", "")).strip()
    part = str(chunk.get("part", "")).strip()
    title = str(chunk.get("title", "")).strip()
    parent_title = str(chunk.get("parent_title", "")).strip()
    content = str(chunk.get("content", "")).strip()
    base = "\n".join([doc_id, part, title, parent_title, content])
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:20]
    return f"chunk_{digest}"


def _write_chunk_rows(client, collection_name: str, rows: List[Dict[str, Any]]):
    if hasattr(client, "upsert"):
        return client.upsert(collection_name=collection_name, data=rows)
    return client.insert(collection_name=collection_name, data=rows)


def node_import_milvus(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate chunks, prepare the collection, and upsert the document vectors."""
    current_node = sys._getframe().f_code.co_name
    logger.info(f"{current_node} started")
    add_running_task(state["task_id"], current_node)
    logger.info("Milvus chunk persistence started")

    try:
        chunks_json_data, vector_dimension = step_1_check_input(state)
        client = step_2_prepare_collection(vector_dimension)
        updated_chunks = step_4_insert_data(client, chunks_json_data)
        state["chunks"] = updated_chunks

        logger.info(f"Milvus chunk persistence completed: chunks={len(updated_chunks)}")
    except Exception as e:
        logger.error(f"Milvus chunk persistence failed: {e}", exc_info=True)
        raise ValueError(f"Milvus import failed: {e}")

    return state

def step_1_check_input(state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    """Validate chunks and return the dense-vector dimension."""
    chunks_json_data = state.get("chunks")
    if not chunks_json_data:
        logger.error("Milvus input validation failed: chunks is empty")
        raise ValueError("chunks is required for Milvus import")
    if not isinstance(chunks_json_data, list) or len(chunks_json_data) == 0:
        logger.error("Milvus input validation failed: chunks must be a non-empty list")
        raise ValueError("chunks must be a non-empty list")
    first_chunk = chunks_json_data[0]
    if 'dense_vector' not in first_chunk:
        logger.error("Milvus input validation failed: dense_vector is missing")
        raise ValueError("dense_vector is required for Milvus import")
    if not str(first_chunk.get("doc_id", "")).strip():
        logger.error("Milvus input validation failed: doc_id is missing")
        raise ValueError("doc_id is required for Milvus import")

    vector_dimension = len(first_chunk['dense_vector'])
    item_name = first_chunk.get('item_name', '-')
    doc_id = first_chunk.get('doc_id', '')
    logger.info(
        f"Milvus input validated: chunks={len(chunks_json_data)}, dimension={vector_dimension}, "
        f"item_name={item_name}, doc_id={doc_id or '-'}"
    )

    return chunks_json_data, vector_dimension


def create_collection(client, collection_name: str, vector_dimension: int):
    """Create the chunk collection schema and its dense/sparse indexes."""
    schema = client.create_schema(auto_id=False, enable_dynamic_fields=True)

    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="canonical_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="product_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="model_code", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
    )

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"Milvus collection created: name={collection_name}, dimension={vector_dimension}")


def step_2_prepare_collection(vector_dimension: int):
    """Connect to Milvus and create the target collection when required."""
    logger.info(f"Preparing Milvus collection: {CHUNKS_COLLECTION_NAME}")
    client = get_milvus_client()
    if client is None:
        logger.error("Milvus client is unavailable")
        raise ValueError("Milvus client is unavailable")
    if not CHUNKS_COLLECTION_NAME:
        logger.error("CHUNKS_COLLECTION is not configured")
        raise ValueError("CHUNKS_COLLECTION is required")

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.info(f"Milvus collection not found; creating: {CHUNKS_COLLECTION_NAME}")
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
    else:
        logger.info(f"Using existing Milvus collection: {CHUNKS_COLLECTION_NAME}")

    return client


def step_4_insert_data(client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate stable IDs, upsert chunks, and write IDs back to import state."""
    data_to_insert = []
    for item in chunks_json_data:
        item_copy = item.copy()
        item_copy["chunk_id"] = _build_chunk_id(item_copy)
        data_to_insert.append(item_copy)

    logger.info(f"Milvus upsert started: chunks={len(data_to_insert)}")
    write_result = _write_chunk_rows(client, CHUNKS_COLLECTION_NAME, data_to_insert)

    logger.info(f"Milvus upsert completed: chunks={len(data_to_insert)}, result={write_result}")

    for idx, item in enumerate(chunks_json_data):
        item["chunk_id"] = data_to_insert[idx]["chunk_id"]
    logger.info("Stable chunk IDs synchronized to import state")

    return chunks_json_data
