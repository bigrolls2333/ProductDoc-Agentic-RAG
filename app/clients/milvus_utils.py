"""Shared Milvus client and hybrid-search helpers."""

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# Process-local client cache.
_milvus_client = None


def get_milvus_client():
    """Return the cached Milvus client or ``None`` when initialization fails."""
    try:
        global _milvus_client
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            if not milvus_uri:
                logger.error("Milvus client unavailable: MILVUS_URL is not configured")
                return None
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus client initialized")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus client initialization failed: {e}", exc_info=True)
        return None


def _normalize_primary_ids(ids):
    """Normalize chunk identifiers and return valid and invalid values."""
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        value = str(x).strip()
        if not value:
            bad.append(x)
            continue
        ok.append(value)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """Fetch chunk records by primary key with batched get/query fallback."""
    if client is None:
        return []
    if not collection_name:
        return []
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    ok_ids, bad_ids = _normalize_primary_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"Skipping invalid chunk identifiers: {bad_ids}")

    if not ok_ids:
        return []

    results = []
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get failed; falling back to query: {e}")

        try:
            quoted_ids = ", ".join(f'"{str(x).replace(chr(34), chr(92) + chr(34))}"' for x in batch)
            expr = f"chunk_id in [{quoted_ids}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query failed for chunk identifiers: {e}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """Build dense and sparse Milvus search requests for hybrid retrieval."""
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """Run weighted dense/sparse hybrid search and return its raw result."""
    try:
        if hasattr(client, "load_collection"):
            try:
                client.load_collection(collection_name=collection_name)
            except Exception as load_error:
                logger.warning(
                    f"Milvus collection load failed; attempting search anyway: "
                    f"collection={collection_name}, error={load_error}"
                )

        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        if output_fields is None:
            output_fields = ["item_name"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus hybrid search completed: collection={collection_name}, hits={len(res[0])}")
        return res
    except Exception as e:
        logger.error(f"Milvus hybrid search failed: collection={collection_name}, error={e}", exc_info=True)
        return None
