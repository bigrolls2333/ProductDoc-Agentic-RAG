"""Per-item retrieval with shared query vectors and explicit canonical filters."""

import json
from app.query_process.context_selection import normalize_item_names


def canonical_filter(item_names):
    return 'canonical_name in [' + ', '.join(json.dumps(name, ensure_ascii=False) for name in normalize_item_names(item_names)) + ']'


def search_per_item(*, item_names, per_item_k, dense_vector, sparse_vector, client, collection_name,
                    create_requests, search, output_fields, ranker_weights=(0.8, 0.2), norm_score=True,
                    req_limit=10):
    """Return the Milvus list-of-query-results shape; do not compare per-item scores."""
    from itertools import zip_longest
    groups = []
    for name in normalize_item_names(item_names):
        requests = create_requests(dense_vector=dense_vector, sparse_vector=sparse_vector,
                                   expr=canonical_filter([name]), limit=max(req_limit, per_item_k))
        result = search(client=client, collection_name=collection_name, reqs=requests,
                        ranker_weights=ranker_weights, norm_score=norm_score,
                        limit=per_item_k, output_fields=list(output_fields))
        groups.append((result[0] if result else [])[:per_item_k])
    hits = [hit for round_hits in zip_longest(*groups) for hit in round_hits if hit is not None]
    return [hits]
