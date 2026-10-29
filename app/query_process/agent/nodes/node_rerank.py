"""Cross-source reranking and adaptive evidence selection."""

from app.utils.task_utils import *
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.utils.eval_flag_utils import is_eval_enabled
import sys
import os

# Rerank and TopK defaults.
RERANK_MAX_TOPK: int = 8
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.5
RERANK_DISABLE_GAP: bool = True


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={raw}; using default={default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float for {name}={raw}; using default={default}")
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_rerank_config():
    max_topk = max(1, _env_int("EVAL_RERANK_MAX_TOPK", RERANK_MAX_TOPK))
    min_topk = max(1, _env_int("EVAL_RERANK_MIN_TOPK", RERANK_MIN_TOPK))
    min_topk = min(min_topk, max_topk)
    return {
        "max_topk": max_topk,
        "min_topk": min_topk,
        "gap_ratio": max(0.0, _env_float("EVAL_RERANK_GAP_RATIO", RERANK_GAP_RATIO)),
        "gap_abs": max(0.0, _env_float("EVAL_RERANK_GAP_ABS", RERANK_GAP_ABS)),
        "disable_gap": _env_bool("EVAL_RERANK_DISABLE_GAP", RERANK_DISABLE_GAP),
    }

def step_1_merge_docs(state):
    """Normalize local and web evidence into the reranker input schema."""
    
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []
    
    logger.info(f"Evidence normalization started: local={len(rrf_docs)}, web={len(web_docs)}")
    doc_items = []
    for i, doc in enumerate(rrf_docs):
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc
        if not isinstance(entity, dict):
            logger.warning(f"Local evidence has an unsupported shape: index={i}, type={type(entity)}")
            continue
            
        content = entity.get("content")
        if not content:
            logger.debug(f"Local evidence skipped because content is empty: index={i}")
            continue

        chunk_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name") or ""

        doc_items.append({
            "text": content,
            "doc_id": entity.get("doc_id"),
            "chunk_id": chunk_id,
            "canonical_name": entity.get("canonical_name") or entity.get("item_name") or "",
            "product_id": entity.get("product_id") or "",
            "file_title": entity.get("file_title") or "",
            "title": title,
            "url": "",
            "source": "local",
            "retrieval_sources": entity.get('retrieval_sources', []),
        })

    for i, doc in enumerate(web_docs):
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        
        if not text:
            logger.debug(f"Web evidence skipped because text is empty: index={i}")
            continue
            
        doc_items.append({
            "text": text,
            "doc_id": None,
            "chunk_id": None,
            "title": title,
            "url": url,
            "source": "web",
        })

    logger.info(f"Evidence normalization completed: output={len(doc_items)}")
    return doc_items


def step_2_rerank_docs(state, doc_items):
    """Score normalized evidence against the rewritten query."""
    question = state.get("rewritten_query") or state.get("original_query") or ""

    if not doc_items or not question:
        logger.warning("Reranking skipped because evidence or query is empty")
        return []

    logger.info(f"Reranking started: candidates={len(doc_items)}")
    
    texts = [x["text"] for x in doc_items]
    try:
        reranker = get_reranker_model()

        sentence_pairs = [[question, t] for t in texts]
        logger.info("Computing reranker scores")
        scores = reranker.compute_score(sentence_pairs)
        scored_docs = []
        for item, text, score in zip(doc_items, texts, scores):
            score_val = float(score)
            scored_docs.append(
                {
                    "text": text,
                    "score": score_val,
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                    "canonical_name": item.get("canonical_name") or "",
                    "product_id": item.get("product_id") or "",
                    "file_title": item.get("file_title") or "",
                    "retrieval_sources": item.get('retrieval_sources', []),
                }
            )
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs
    except Exception as e:
        logger.error(f"Reranking failed; preserving candidate order: {e}", exc_info=True)
        fallback_docs = [
            {
                "text": x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
                "canonical_name": x.get("canonical_name") or "",
                "product_id": x.get("product_id") or "",
                "file_title": x.get("file_title") or "",
                "retrieval_sources": x.get('retrieval_sources', []),
            }
            for x in doc_items
        ]
        return fallback_docs

def _source_counts(docs):
    counts = {}
    for doc in docs or []:
        source = doc.get("source") or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _apply_topk(scored_docs):
    """Apply configured bounds and optional score-gap truncation."""
    config = _get_rerank_config()
    max_topk = min(config["max_topk"], len(scored_docs))
    min_topk = min(config["min_topk"], max_topk) if max_topk else 0
    gap_ratio = config["gap_ratio"]
    gap_abs = config["gap_abs"]
    disable_gap = config["disable_gap"]

    topk = max_topk
    gap_triggered = False
    gap_index = None
    gap_detail = {}
    if not disable_gap and topk > min_topk:
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")

            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(f"Score-gap truncation triggered: index={i}, gap={gap:.4f}")
                topk = i + 1
                gap_triggered = True
                gap_index = i
                gap_detail = {
                    "score_before": round(float(s1), 6),
                    "score_after": round(float(s2), 6),
                    "gap": round(float(gap), 6),
                    "relative_gap": round(float(rel), 6),
                }
                break

    topk_docs = scored_docs[:topk]
    
    logger.info(f"TopK selection completed: output={len(topk_docs)}, top_k={topk}")
    
    if topk_docs:
        preview = ", ".join([f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})" for d in topk_docs[:3]])
        logger.debug(f"Top evidence preview: {preview}")
        
    trace = {
        "input_docs": len(scored_docs or []),
        "output_docs": len(topk_docs),
        "max_topk": config["max_topk"],
        "min_topk": config["min_topk"],
        "effective_max_topk": max_topk,
        "effective_min_topk": min_topk,
        "gap_ratio": gap_ratio,
        "gap_abs": gap_abs,
        "disable_gap": disable_gap,
        "gap_triggered": gap_triggered,
        "gap_index": gap_index,
        "gap_detail": gap_detail,
        "top_scores": [round(float(d.get("score") or 0.0), 6) for d in (scored_docs or [])[:10]],
        "input_source_counts": _source_counts(scored_docs),
        "output_source_counts": _source_counts(topk_docs),
    }
    return topk_docs, trace


def step_3_topk(scored_docs):
    topk_docs, _ = _apply_topk(scored_docs)
    return topk_docs


def node_rerank(state):
  """Rerank evidence and select the configured top-k candidates."""
  logger.info("node_rerank started")
  add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

  doc_items = step_1_merge_docs(state)
  if is_eval_enabled("EVAL_DISABLE_RERANK"):
      logger.info("Reranker disabled for evaluation; preserving upstream order")
      scored_docs = [
          {
              "text": x.get("text"),
              "score": 0.0,
              "source": x.get("source") or "",
              "chunk_id": x.get("chunk_id"),
              "doc_id": x.get("doc_id"),
              "url": x.get("url") or "",
              "title": x.get("title") or "",
              "canonical_name": x.get("canonical_name") or "",
              "product_id": x.get("product_id") or "",
              "file_title": x.get("file_title") or "",
              "retrieval_sources": x.get('retrieval_sources', []),
          }
          for x in doc_items
      ]
      topk_docs, rerank_trace = _apply_topk(scored_docs)
      rerank_trace["mode"] = "disabled"
      add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
      return {"reranked_docs": topk_docs, "context_candidate_docs": scored_docs[:50], "rerank_trace": rerank_trace}
  scored_docs = step_2_rerank_docs(state, doc_items)
  topk_docs, rerank_trace = _apply_topk(scored_docs)
  rerank_trace["mode"] = "bge"
  
  logger.info(f"node_rerank completed: output={len(topk_docs)}")

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  return {"reranked_docs": topk_docs, "context_candidate_docs": scored_docs[:50], "rerank_trace": rerank_trace}
