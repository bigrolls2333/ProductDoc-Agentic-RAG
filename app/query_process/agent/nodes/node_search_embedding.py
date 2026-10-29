"""Hybrid vector retrieval node for product-document evidence."""

import sys
import os
import re
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def _normalize_name_key(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or ""))
    return compact.upper()


def node_search_embedding(state):
    """Run filtered dense-and-sparse retrieval against the Milvus collection."""
    from app.query_process.context_selection import context_policy, context_per_item_k
    from app.query_process.product_retrieval import search_per_item, canonical_filter
    logger.info("node_search_embedding started")
    add_running_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    retrieval_plan = state.get("retrieval_plan") or {}
    if retrieval_plan and not retrieval_plan.get("use_embedding", True):
        logger.info("Embedding retrieval disabled by retrieval plan")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": []}

    query = state.get("rewritten_query")
    item_names = state.get("item_names")
    
    logger.info(f"Retrieval inputs: query='{query}', item_names={item_names}")

    logger.info(f"Generating query embeddings: {query[:50]}..." if len(query) > 50 else f"Generating query embeddings: {query}")
    embeddings = generate_embeddings([query])
    
    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]
    logger.debug(f"Query embeddings generated: dense_dim={len(dense_vec)}, sparse_terms={len(sparse_vec)}")

    collection_name = os.environ.get("CHUNKS_COLLECTION")
    logger.info(f"Preparing Milvus collection: {collection_name}")

    if not item_names:
        logger.warning("Embedding retrieval skipped because item_names is empty")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": []}
        
    normalized_item_names = list(dict.fromkeys([
        normalized for v in item_names if (normalized := _normalize_name_key(v))
    ]))
    expr = canonical_filter(normalized_item_names)
    logger.info(f"Milvus filter expression: {expr}")

    logger.info("Executing Milvus hybrid search")
    client = get_milvus_client()
    fields = ["chunk_id", "content", "canonical_name", "item_name", "doc_id", "file_title", "title"]
    if context_policy(state) == 'item_balanced':
        res = search_per_item(item_names=normalized_item_names, per_item_k=context_per_item_k(state),
                              dense_vector=dense_vec, sparse_vector=sparse_vec, client=client,
                              collection_name=collection_name, create_requests=create_hybrid_search_requests,
                              search=hybrid_search, output_fields=fields)
    else:
        reqs = create_hybrid_search_requests(dense_vector=dense_vec, sparse_vector=sparse_vec, expr=expr, limit=10)
        res = hybrid_search(client=client, collection_name=collection_name, reqs=reqs,
                            ranker_weights=(0.8, 0.2), norm_score=True, limit=5, output_fields=fields)

    hit_count = len(res[0]) if res and len(res) > 0 else 0
    logger.info(f"node_search_embedding completed: hits={hit_count}")
    if hit_count > 0:
        logger.debug(f"Top retrieval hit: {res[0][0]}")
        
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"embedding_chunks": res[0] if res else []}
