"""
retrieval_benchmark_utils.py - weak-supervision IR metrics for retrieval/rerank eval.

The metrics are derived from validation-set source_hint fields. They are suitable
for comparing evidence ordering across retrieval/rerank variants, while remaining
separate from a fully human-labeled corpus-level benchmark.
"""
from math import log2
from typing import Any, Dict, List, Tuple


RETRIEVAL_BENCHMARK_KS = (1, 3, 5, 8, 10)


def normalize_text(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("（", "(").replace("）", ")")
    return "".join(value.upper().split())


def _doc_text_for_benchmark(doc: Dict[str, Any]) -> str:
    title = normalize_text(doc.get("title", "") or "")
    text = normalize_text(doc.get("text", "") or doc.get("content", "") or "")
    chunk_id = normalize_text(doc.get("chunk_id", "") or doc.get("doc_id", "") or "")
    return " ".join(x for x in [title, text, chunk_id] if x)


def _matched_hints(
    doc: Dict[str, Any],
    expected_titles: List[str],
    expected_keywords: List[str],
) -> Tuple[List[str], List[str]]:
    haystack = _doc_text_for_benchmark(doc)
    title_matches = [
        hint for hint in expected_titles
        if normalize_text(hint) and normalize_text(hint) in haystack
    ]
    keyword_matches = [
        hint for hint in expected_keywords
        if normalize_text(hint) and normalize_text(hint) in haystack
    ]
    return title_matches, keyword_matches


def _dcg(relevances: List[float], k: int) -> float:
    return sum(((2 ** rel) - 1) / log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def build_retrieval_benchmark(
    reranked_docs: List[Dict[str, Any]],
    expected_titles: List[str],
    expected_keywords: List[str],
) -> Dict[str, Any]:
    """
    Build source_hint-based retrieval metrics.

    source_hint_recall@K measures expected title/keyword coverage in top K.
    nDCG@K uses graded relevance from matched source_hint count among retrieved candidates.
    """
    docs = reranked_docs or []
    expected_titles = [str(x) for x in (expected_titles or []) if str(x).strip()]
    expected_keywords = [str(x) for x in (expected_keywords or []) if str(x).strip()]
    hint_total = len(expected_titles) + len(expected_keywords)

    rank_details = []
    relevances: List[float] = []
    for idx, doc in enumerate(docs, start=1):
        title_matches, keyword_matches = _matched_hints(doc, expected_titles, expected_keywords)
        matched = [f"title:{x}" for x in title_matches] + [f"keyword:{x}" for x in keyword_matches]
        relevance = float(len(set(matched)))
        relevances.append(relevance)
        rank_details.append({
            "rank": idx,
            "chunk_id": doc.get("chunk_id") or doc.get("doc_id") or "",
            "source": doc.get("source") or "",
            "score": doc.get("score"),
            "matched_titles": title_matches,
            "matched_keywords": keyword_matches,
            "relevance": relevance,
        })

    covered_all = set()
    for detail in rank_details:
        covered_all.update(f"title:{x}" for x in detail["matched_titles"])
        covered_all.update(f"keyword:{x}" for x in detail["matched_keywords"])

    first_relevant_rank = next((i for i, rel in enumerate(relevances, start=1) if rel > 0), None)
    ideal_relevances = sorted(relevances, reverse=True)
    metrics: Dict[str, Any] = {
        "benchmark_type": "source_hint_weak_supervision",
        "hint_total": hint_total,
        "candidate_count": len(docs),
        "relevant_candidate_count": sum(1 for rel in relevances if rel > 0),
        "first_relevant_rank": first_relevant_rank,
        "observed_source_hint_recall": round(len(covered_all) / hint_total, 6) if hint_total else None,
        "rank_details": rank_details[:10],
    }

    for k in RETRIEVAL_BENCHMARK_KS:
        top_details = rank_details[:k]
        covered = set()
        for detail in top_details:
            covered.update(f"title:{x}" for x in detail["matched_titles"])
            covered.update(f"keyword:{x}" for x in detail["matched_keywords"])

        idcg = _dcg(ideal_relevances, k)
        dcg = _dcg(relevances, k)
        metrics[f"hit@{k}"] = 1.0 if any(rel > 0 for rel in relevances[:k]) else 0.0
        metrics[f"source_hint_recall@{k}"] = round(len(covered) / hint_total, 6) if hint_total else None
        metrics[f"mrr@{k}"] = round(1.0 / first_relevant_rank, 6) if first_relevant_rank and first_relevant_rank <= k else 0.0
        metrics[f"ndcg@{k}"] = round(dcg / idcg, 6) if idcg > 0 else 0.0

    return metrics
