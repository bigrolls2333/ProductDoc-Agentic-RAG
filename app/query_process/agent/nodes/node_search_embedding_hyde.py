"""HyDE retrieval node for query expansion and hybrid vector search."""

import os
import sys
import re
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.eval_flag_utils import is_eval_enabled
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def _normalize_name_key(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or ""))
    return compact.upper()


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """Generate a hypothetical answer used to expand the retrieval query."""
    if not rewritten_query:
        logger.error("HyDE generation skipped because rewritten_query is empty")
        raise ValueError("rewritten_query is required")

    logger.info(f"HyDE generation started: query={rewritten_query}")

    try:
        llm = get_llm_client()
        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        logger.debug(f"HyDE prompt loaded: length={len(hyde_prompt)}")

        response = llm.invoke(hyde_prompt)
        hyde_doc = response.content
        
        logger.info(f"HyDE generation completed: length={len(hyde_doc)}")
        logger.debug(f"HyDE document preview: {hyde_doc[:50]}...")
        
        return hyde_doc

    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        raise e


def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),  # Favor dense retrieval in the default blend.
    norm_score: bool = True,    # Normalize scores by default.
    output_fields=["chunk_id", "content", "canonical_name", "item_name", "doc_id", "file_title", "title"],
    per_item_k=None,
):
    """Search Milvus with embeddings generated from the query and HyDE document."""
    from app.query_process.product_retrieval import search_per_item, canonical_filter
    if not rewritten_query:
        raise ValueError("rewritten_query is required")
    if not hyde_doc:
        raise ValueError("hyde_doc is required")

    combined_text = rewritten_query + " " + hyde_doc
    logger.info(f"HyDE retrieval input assembled: length={len(combined_text)}")

    logger.info("Generating dense and sparse retrieval vectors")
    embeddings = generate_embeddings([combined_text])
    
    # Prepare the Milvus search request.
    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("CHUNKS_COLLECTION is not configured")
        return []
        
    logger.info(f"Preparing HyDE search: collection={collection_name}")

    expr = None
    if item_names:
        normalized_item_names = list(dict.fromkeys([
            normalized for v in item_names if (normalized := _normalize_name_key(v))
        ]))
        expr = canonical_filter(normalized_item_names)
        logger.info(f"HyDE filter expression: {expr}")
    else:
        logger.info("HyDE retrieval has no product filter")

    try:
        client = get_milvus_client()
        if not client:
            logger.error("Milvus client is unavailable")
            return []

        if per_item_k is not None and item_names:
            return search_per_item(item_names=item_names, per_item_k=per_item_k,
                dense_vector=embeddings.get('dense')[0], sparse_vector=embeddings.get('sparse')[0],
                client=client, collection_name=collection_name, create_requests=create_hybrid_search_requests,
                search=hybrid_search, output_fields=output_fields, ranker_weights=ranker_weights,
                norm_score=norm_score, req_limit=req_limit)

        reqs = create_hybrid_search_requests(dense_vector=embeddings.get('dense')[0],
            sparse_vector=embeddings.get('sparse')[0], expr=expr, limit=req_limit)

        logger.info(f"Executing HyDE hybrid search: weights={ranker_weights}, top_k={top_k}")
        res = hybrid_search(
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=top_k,
            output_fields=list(output_fields),
        )
        
        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"HyDE retrieval completed: hits={hit_count}")
        
        return res

    except Exception as e:
        logger.error(f"HyDE retrieval failed: {e}")
        return []


def node_search_embedding_hyde(state):
    """Generate a HyDE document, run hybrid retrieval, and update query state."""
    from app.query_process.context_selection import context_policy, context_per_item_k
    logger.info("node_search_embedding_hyde started")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    retrieval_plan = state.get("retrieval_plan") or {}
    if is_eval_enabled("EVAL_DISABLE_HYDE") or not retrieval_plan.get("use_hyde", False):
        logger.info("retrieval_plan.use_hyde=False; skipping HyDE retrieval")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        rewritten_query = state.get("original_query")
    
    if not rewritten_query:
        logger.error("HyDE retrieval skipped because no query is available")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    item_names = state.get("item_names")
    logger.info(f"HyDE retrieval inputs: query='{rewritten_query}', item_names={item_names}")

    hyde_doc = ""
    try:
        logger.info("HyDE generation started")
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        logger.info(f"HyDE document ready: length={len(hyde_doc)}")
        logger.debug(f"HyDE document preview: {hyde_doc[:100]}...")
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}", exc_info=True)
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    try:
        logger.info("HyDE retrieval started")
        res = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
            per_item_k=context_per_item_k(state) if context_policy(state) == 'item_balanced' else None,
        )
        
        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"HyDE retrieval completed: hits={hit_count}")
        
        if hit_count > 0:
            first_hit = res[0][0]
            score = first_hit.get("distance")
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]
            logger.debug(f"Top retrieval hit: score={score}, content='{content_preview}...'")

        return {
            "hyde_embedding_chunks": res[0] if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception as e:
        logger.error(f"HyDE retrieval failed: {e}", exc_info=True)
        return {"hyde_embedding_chunks": [], "hyde_doc": hyde_doc}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        logger.info("node_search_embedding_hyde completed")
