"""Optional parallel recall branch executing the existing planner's subqueries."""

import os
import time

from app.query_process.subquery_retrieval import subquery_enabled, validate_subqueries
from app.query_process.context_selection import context_per_item_k, normalize_item_names
from app.query_process.product_retrieval import search_per_item


def node_search_subqueries(state):
    started = time.perf_counter()
    trace = {'status': 'disabled', 'search_calls': 0, 'queries': []}
    output = {'subquery_chunks': [], 'subquery_trace': trace}
    if not subquery_enabled(state):
        return output
    try:
        names = normalize_item_names(state.get('item_names'))
        query = state.get('rewritten_query') or state.get('original_query') or ''
        queries = validate_subqueries((state.get('retrieval_plan') or {}).get('subqueries', []), names, query)
        trace.update(status='no_expansion', queries=queries)
        if not queries:
            return output
        from app.lm.embedding_utils import generate_embeddings
        from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
        from app.query_process.agent.nodes.node_rrf import _as_entity_list
        texts = list(dict.fromkeys(row['query'] for row in queries))
        embeddings = generate_embeddings(texts)
        client = get_milvus_client()
        if client is None:
            raise RuntimeError('Milvus unavailable for subquery recall')
        results = []
        for row in queries:
            index = texts.index(row['query'])
            items = [names[i] for i in row['item_indices']]
            hits = search_per_item(item_names=items, per_item_k=context_per_item_k(state),
                dense_vector=embeddings['dense'][index], sparse_vector=embeddings['sparse'][index],
                client=client, collection_name=os.environ.get('CHUNKS_COLLECTION'),
                create_requests=create_hybrid_search_requests, search=hybrid_search,
                output_fields=['chunk_id', 'content', 'canonical_name', 'item_name', 'doc_id', 'file_title', 'title'])
            results.append(_as_entity_list(hits[0] if hits else []))
            trace['search_calls'] += len(items)
        output['subquery_chunks'] = results
        trace.update(status='completed', hit_counts=[len(rows) for rows in results], encoded_queries=len(texts))
    except Exception as exc:
        # Failed optional expansion must not discard the ordinary recall routes.
        trace.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        trace['latency_ms'] = (time.perf_counter() - started) * 1000
    return output
