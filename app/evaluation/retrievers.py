"""Independent retrieval backends with explicit filters and no Agent side effects."""

import json
import math
import re
from collections import Counter


def eligible_ids(corpus, query, scope):
    if scope == "all":
        return list(corpus)
    ids, names = set(query.get("product_ids", [])), set(query.get("canonical_names", []))
    if not ids and not names:
        raise ValueError(f"product scope requires reviewed product_ids or canonical_names: {query['query_id']}")
    return [cid for cid, doc in corpus.items()
            if (doc.get("product_id") in ids if ids else doc.get("canonical_name") in names)]


def tokenize(text):
    """Latin/model tokens and overlapping Chinese character bigrams.

    Kept deterministic and dependency-free; this is an explicit benchmark
    tokenizer, not a claim of parity with a language-specific BM25 service.
    """
    tokens = []
    for term in re.findall(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", term):
            tokens.extend(term[i:i + 2] for i in range(len(term) - 1))
            if len(term) == 1:
                tokens.append(term)
        else:
            tokens.append(term)
    return tokens


class BM25Retriever:
    metadata = {"backend": "bm25", "tokenizer": "latin-model+han-bigram.v1", "k1": 1.5, "b": 0.75}

    def __init__(self, corpus):
        self.corpus = corpus
        self.terms = {cid: Counter(tokenize(doc["content"])) for cid, doc in corpus.items()}
        self.lengths = {cid: sum(terms.values()) for cid, terms in self.terms.items()}
        self.avg_length = sum(self.lengths.values()) / max(len(corpus), 1) or 1
        self.df = Counter(term for terms in self.terms.values() for term in terms)

    def search(self, query, eligible, mode, limit):
        if mode != "bm25":
            raise ValueError("BM25 backend only supports bm25")
        n = len(self.corpus)
        query_terms = set(tokenize(query))
        scored = []
        for cid in eligible:
            score = 0.0
            for term in query_terms:
                frequency = self.terms[cid].get(term, 0)
                if frequency:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * self.lengths[cid] / self.avg_length))
            if score > 0:
                scored.append({"chunk_id": cid, "score": score})
        return sorted(scored, key=lambda item: (-item["score"], item["chunk_id"]))[:limit]


class MilvusRetriever:
    """Read-only backend that verifies live records against the frozen corpus."""

    def __init__(self, corpus, collection, query_encoder='encode_queries'):
        from app.clients.milvus_utils import get_milvus_client
        from app.lm.embedding_utils import get_bge_m3_ef
        from app.conf.embedding_config import embedding_config

        self.client = get_milvus_client()
        if self.client is None:
            raise RuntimeError("Milvus is unavailable; no empty-success fallback is allowed")
        self.collection = collection
        self.client.load_collection(collection_name=collection)
        live = export_corpus(self.client, collection)
        frozen = {cid: (d["doc_id"], d["content"], d.get("product_id", ""), d.get("canonical_name", ""))
                  for cid, d in corpus.items()}
        actual = {d["chunk_id"]: (d["doc_id"], d["content"], d.get("product_id", ""), d.get("canonical_name", "")) for d in live}
        if frozen != actual:
            raise ValueError("live collection differs from corpus snapshot; export a new benchmark version")
        self.model = get_bge_m3_ef()
        if query_encoder not in {'encode_queries', 'encode_documents'}:
            raise ValueError('unknown query encoder')
        self.query_encoder = query_encoder
        self.vectors = {}
        self.metadata = {"backend": "milvus", "collection": collection,
                         "embedding_model": embedding_config.bge_m3_path or "BAAI/bge-m3",
                         "device": embedding_config.bge_device, "fp16": embedding_config.bge_fp16,
                         "query_encoder": query_encoder, "dense_weight": 0.8, "sparse_weight": 0.2,
                         "snapshot_verified": True}

    def search(self, query, eligible, mode, limit):
        from pymilvus import AnnSearchRequest, WeightedRanker

        if not eligible:
            return []
        if query not in self.vectors:
            encoded = getattr(self.model, self.query_encoder)([query])
            sparse = encoded["sparse"].tocsr()
            self.vectors[query] = (encoded["dense"][0].tolist(),
                                   dict(zip(sparse.indices.tolist(), sparse.data.tolist())))
        dense, sparse = self.vectors[query]
        expr = "chunk_id in " + json.dumps(eligible, ensure_ascii=False)
        if mode == "hybrid":
            requests = [AnnSearchRequest(data=[vector], anns_field=field, param={"metric_type": metric}, expr=expr, limit=limit)
                        for vector, field, metric in [(dense, "dense_vector", "COSINE"), (sparse, "sparse_vector", "IP")]]
            results = self.client.hybrid_search(collection_name=self.collection, reqs=requests,
                                                ranker=WeightedRanker(0.8, 0.2), limit=limit,
                                                output_fields=["chunk_id"], consistency_level="Strong")
        else:
            vector, field, metric = (dense, "dense_vector", "COSINE") if mode == "dense" else (sparse, "sparse_vector", "IP")
            results = self.client.search(collection_name=self.collection, data=[vector], anns_field=field,
                                         filter=expr, limit=limit, search_params={"metric_type": metric},
                                         output_fields=["chunk_id"], consistency_level="Strong")
        return [{"chunk_id": str(hit.get("entity", {}).get("chunk_id") or hit["id"]),
                 "score": float(hit["distance"])} for hit in results[0]]


def export_corpus(client, collection):
    fields = ["chunk_id", "doc_id", "content", "title", "parent_title", "file_title", "product_id", "canonical_name"]
    iterator = client.query_iterator(collection_name=collection, filter="", output_fields=fields,
                                     batch_size=500, consistency_level="Strong")
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    return sorted(rows, key=lambda row: row["chunk_id"])
