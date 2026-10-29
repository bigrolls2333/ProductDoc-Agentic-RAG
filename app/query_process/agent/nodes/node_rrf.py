"""Reciprocal rank fusion for local retrieval branches."""

import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.utils.eval_flag_utils import is_eval_enabled


def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    """Normalize Milvus hits and mapping-like results to entity dictionaries."""
    out: List[Dict[str, Any]] = []
    for doc in (state_list or []):
        if not doc:
            continue
        
        final_ent = {}
        
        if hasattr(doc, "entity") and hasattr(doc, "id"):
            entity_content = doc.entity
            if hasattr(entity_content, "to_dict"):
                 final_ent = entity_content.to_dict()
            elif isinstance(entity_content, dict):
                 final_ent = entity_content.copy()
            else:
                 # Support SDK versions that expose an iterable entity object.
                 try:
                     final_ent = dict(entity_content)
                 except:
                     pass
            
            if "id" not in final_ent and "chunk_id" not in final_ent:
                final_ent["id"] = doc.id
            if hasattr(doc, "distance"):
                final_ent["score"] = doc.distance

        elif isinstance(doc, dict):
             if "entity" in doc:
                 ent = doc["entity"]
                 if isinstance(ent, dict):
                     final_ent = ent.copy()
                 if "id" in doc and "id" not in final_ent:
                     final_ent["id"] = doc["id"]
                 if "distance" in doc:
                     final_ent["score"] = doc["distance"]
             else:
                 final_ent = doc

        elif hasattr(doc, "get"):
             ent = doc.get("entity") or doc
             if isinstance(ent, dict):
                 final_ent = ent
        
        if final_ent and isinstance(final_ent, dict):
            out.append(final_ent)
            
    return out


def reciprocal_rank_fusion(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:
    """Fuse ranked document lists with optional per-source weights."""
    score_map = {}
    chunk_map = {}

    for docs, weight in source_weights:
        for rank, item in enumerate(docs, start=1):
            chunk_id = item.get("chunk_id") or item.get("id")
            
            if not chunk_id:
                logger.warning(
                    "RRF item missing chunk_id/id: "
                    f"{list(item.keys()) if isinstance(item, dict) else item}"
                )
                continue

            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))
            chunk_map.setdefault(chunk_id, item)

    merged = []
    for chunk_id, score in score_map.items():
        doc_item = chunk_map[chunk_id]
        merged.append((doc_item, score))
    
    merged.sort(key=lambda x: x[1], reverse=True)
    if max_results is not None:
        merged = merged[:max_results]
        
    return merged


def node_rrf(state):
    """Fuse embedding and HyDE retrieval results into a bounded ranked list."""
    from app.query_process.context_selection import context_policy, context_per_item_k, group_by_item, interleave_groups
    from app.query_process.subquery_retrieval import subquery_enabled, weighted_rrf
    per_item_mode = context_policy(state) == 'item_balanced' and bool(state.get('item_names'))
    logger.info("node_rrf started")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))
    subquery_lists = ([_as_entity_list(rows) for rows in state.get('subquery_chunks', [])]
                      if subquery_enabled(state) else [])

    logger.info(
        f"RRF inputs: embedding={len(embedding_chunks)}, "
        f"hyde={len(hyde_embedding_chunks)}"
    )

    if is_eval_enabled("EVAL_DISABLE_RRF"):
        seen = set()
        merged = []
        for doc in embedding_chunks + hyde_embedding_chunks + [doc for rows in subquery_lists for doc in rows]:
            doc_id = doc.get("chunk_id") or doc.get("id") or str(len(merged))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(doc)
            if not per_item_mode and len(merged) >= 10:
                break
        if per_item_mode:
            merged = interleave_groups(group_by_item(merged, state['item_names'], context_per_item_k(state)))[:50]
        logger.info(f"RRF disabled for evaluation: output={len(merged)}")
        add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"rrf_chunks": merged}
    
    if embedding_chunks:
        logger.debug(f"Embedding retrieval IDs: {[c.get('chunk_id') for c in embedding_chunks[:5]]}")
    if hyde_embedding_chunks:
        logger.debug(f"HyDE retrieval IDs: {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}")

    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    if subquery_enabled(state):
        routes = [('main', embedding_chunks, 1.0), ('hyde', hyde_embedding_chunks, 1.0)]
        routes.extend((f'subquery:{i}', docs, 1/len(subquery_lists)) for i, docs in enumerate(subquery_lists))
        rrf_res = weighted_rrf(routes, k=60, max_results=None if per_item_mode else 10)
    else:
        rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=None if per_item_mode else 10)

    rrf_chunks = [doc for doc, score in rrf_res]
    if per_item_mode:
        rrf_chunks = interleave_groups(group_by_item(rrf_chunks, state['item_names'], context_per_item_k(state)))[:50]
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info(f"node_rrf completed: output={len(rrf_chunks)}")
    return {"rrf_chunks": rrf_chunks}
