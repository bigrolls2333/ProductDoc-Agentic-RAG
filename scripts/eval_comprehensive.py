"""
eval_comprehensive.py — Comprehensive Agentic RAG evaluation script.

Runs the full query graph per sample and evaluates ALL downstream dimensions:
- Retrieval quality (reranked_docs vs source_hint)
- Self-eval accuracy (self_eval_status)
- Answer quality (must_include / should_not_include)
- Retry effectiveness

Usage:
    python scripts/eval_comprehensive.py
    python scripts/eval_comprehensive.py --limit 10
    python scripts/eval_comprehensive.py --dimensions retrieval,self_eval,answer
    python scripts/eval_comprehensive.py --answer-method llm --concurrency 4
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mongo_history_utils import clear_history
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import update_task_status, TASK_STATUS_COMPLETED
from eval_utils import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    ItemResolver,
    compare_item_identity,
    compare_retrieval_plan,
    expected_item_names,
    expected_product_ids,
    expected_self_eval_status,
    filter_samples,
    is_sample_ready,
    load_jsonl,
    make_session_id,
    normalize_key,
    ordered_unique,
    print_summary,
    rate,
    seed_history,
)
from retrieval_benchmark_utils import build_retrieval_benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ALL_DIMENSIONS = {"routing", "retrieval", "self_eval", "answer", "retry"}
DEFAULT_CONCURRENCY = 1
ANSWER_EVAL_TEXT_PREVIEW_CHARS = 500
DEFAULT_ARTIFACT_TEXT_CHARS = 4000


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive Agentic RAG evaluation")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to validation_set.jsonl")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to logs/eval_comprehensive_*.json")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N samples (0 = all)")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--ids", default="", help="Comma-separated sample ID filter")
    parser.add_argument("--include-not-ready", action="store_true", help="Include samples marked not-ready")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first exception")
    parser.add_argument("--no-web", action="store_true", help="Skip MCP web search (offline mode)")
    parser.add_argument(
        "--dimensions",
        default="all",
        help="Comma-separated dimensions: routing,retrieval,self_eval,answer,retry (default: all)",
    )
    parser.add_argument(
        "--answer-method",
        default="string",
        choices=["string", "llm", "hybrid"],
        help="Answer quality eval method: string=fast (default), llm=LLM-as-judge, hybrid=string+LLM fallback",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Parallel graph invocations")
    parser.add_argument("--skip-item-names", action="store_true", help="Skip item identity checks")
    parser.add_argument("--keep-history", action="store_true", help="Keep seeded eval history in Mongo for debugging")
    parser.add_argument("--variant-name", default="", help="Ablation variant name for report metadata")
    parser.add_argument(
        "--artifact-text-chars",
        type=int,
        default=DEFAULT_ARTIFACT_TEXT_CHARS,
        help="Max chars per retained document text in judge_artifacts (0 = keep full text)",
    )
    parser.add_argument("--no-judge-artifacts", action="store_true", help="Do not store judge_artifacts in output JSON")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dimension: Retrieval quality
# ---------------------------------------------------------------------------
def evaluate_retrieval_quality(
    sample: Dict[str, Any], final_state: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Check if reranked_docs contain expected source_hint evidence."""
    expected = sample.get("expected", {})
    source_hint = expected.get("source_hint", {})
    expected_titles = source_hint.get("doc_title_contains", [])
    expected_keywords = source_hint.get("chunk_keywords", [])
    plan = expected.get("retrieval_plan", {})

    # Skip if embedding retrieval not expected
    if not plan.get("use_embedding", False):
        return True, {"skipped": True, "reason": "no_embedding_expected"}
    if not expected_titles and not expected_keywords:
        return True, {"skipped": True, "reason": "no_source_hint"}

    reranked = final_state.get("reranked_docs") or []
    # Build title candidates from available fields
    # Note: reranked_docs from node_rerank only has: text, doc_id, chunk_id, title, url, source
    # file_title/item_name/canonical_name are NOT passed through from Milvus chunks
    top_k_titles_norm = [
        normalize_key(d.get("title", "") or "") for d in reranked
    ]
    top_k_texts = [normalize_key(d.get("text", "") or d.get("content", "") or "") for d in reranked]
    # Also check text content for expected titles (doc titles often appear in chunk text)
    top_k_texts_and_titles = [
        t + " " + txt for t, txt in zip(top_k_titles_norm, top_k_texts)
    ]

    # Title hits: check title field AND text content (doc titles appear in chunk text)
    title_hits: Dict[str, bool] = {}
    for et in expected_titles:
        et_norm = normalize_key(et)
        title_hits[et] = any(et_norm in t for t in top_k_titles_norm) or any(
            et_norm in t for t in top_k_texts
        )

    # Keyword hits
    keyword_hits: Dict[str, bool] = {}
    for kw in expected_keywords:
        kw_norm = normalize_key(kw)
        keyword_hits[kw] = any(kw_norm in t for t in top_k_texts)

    title_pass_rate = sum(title_hits.values()) / max(len(title_hits), 1) if title_hits else 1.0
    keyword_pass_rate = sum(keyword_hits.values()) / max(len(keyword_hits), 1) if keyword_hits else 1.0

    all_titles_matched = all(title_hits.values()) if title_hits else True
    keywords_pass = (keyword_pass_rate >= 0.5) if keyword_hits else True
    passed = all_titles_matched and keywords_pass

    return passed, {
        "title_hits": title_hits,
        "keyword_hits": keyword_hits,
        "title_pass_rate": round(title_pass_rate, 3),
        "keyword_pass_rate": round(keyword_pass_rate, 3),
        "combined_score": round(title_pass_rate * 0.4 + keyword_pass_rate * 0.6, 3),
        "total_reranked": len(reranked),
        "benchmark": build_retrieval_benchmark(reranked, expected_titles, expected_keywords),
    }


# ---------------------------------------------------------------------------
# Dimension: Self-eval accuracy
# ---------------------------------------------------------------------------
def evaluate_self_eval_accuracy(
    sample: Dict[str, Any], final_state: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Check if self_eval_status matches expected."""
    expected = sample.get("expected", {})
    expected_status = expected_self_eval_status(sample)
    actual_status = str(final_state.get("self_eval_status", ""))

    if expected_status == "not_applicable":
        return True, {"skipped": True, "reason": "self_eval_not_applicable"}

    passed = actual_status == expected_status

    # Categorize failure type
    failure_type: Optional[str] = None
    if not passed:
        if expected_status == "sufficient" and actual_status == "insufficient_retry":
            failure_type = "false_positive_retry"
        elif expected_status == "sufficient" and actual_status == "needs_clarification":
            failure_type = "false_negative_sufficient"
        elif expected_status == "insufficient_retry" and actual_status != "insufficient_retry":
            failure_type = "missed_retry_need"
        elif expected_status == "needs_clarification" and actual_status == "sufficient":
            failure_type = "overconfident"
        elif expected_status == "needs_clarification" and actual_status == "insufficient_retry":
            failure_type = "retry_on_clarification"
        else:
            failure_type = f"mismatch:{expected_status}->{actual_status}"

    detail = final_state.get("self_eval_detail", {})
    return passed, {
        "expected": expected_status,
        "actual": actual_status,
        "retry_count": int(final_state.get("self_eval_retry_count", 0)),
        "failure_type": failure_type,
        "eval_reason": (detail or {}).get("reason", ""),
        "eval_confidence": (detail or {}).get("confidence", 0.0),
    }


# ---------------------------------------------------------------------------
# Dimension: Answer quality (string matching)
# ---------------------------------------------------------------------------
def _check_answer_string(answer: str, must_include: List[str], should_not_include: List[str]) -> Dict[str, Any]:
    """Phase 1: fast string matching for answer quality."""
    answer_norm = normalize_key(answer)

    must_results: Dict[str, str] = {}
    for item in must_include:
        item_norm = normalize_key(item)
        must_results[item] = "found" if item_norm in answer_norm else "missing"

    should_not_results: Dict[str, str] = {}
    for item in should_not_include:
        item_norm = normalize_key(item)
        should_not_results[item] = "found_violation" if item_norm in answer_norm else "absent"

    all_must_pass = all(v == "found" for v in must_results.values())
    all_should_not_pass = all(v == "absent" for v in should_not_results.values())

    return {
        "method": "string_match",
        "must_include": must_results,
        "should_not_include": should_not_results,
        "all_must_pass": all_must_pass,
        "all_should_not_pass": all_should_not_pass,
        "overall_string_pass": all_must_pass and all_should_not_pass,
    }


def _check_answer_llm(
    query: str, answer: str, must_include: List[str], should_not_include: List[str]
) -> Dict[str, Any]:
    """Phase 2: LLM-as-judge for answer quality."""
    must_str = "\n".join(f"  - {item}" for item in must_include)
    should_not_str = "\n".join(f"  - {item}" for item in should_not_include)

    system_prompt = (
        "你是产品文档问答系统的答案质量评估器。"
        "你的任务是判断系统回答是否覆盖了必须包含的内容，且没有包含禁止出现的内容。"
        "必须只输出JSON，不要输出解释文本。"
    )
    user_prompt = f"""
用户问题：
{query}

系统回答：
{answer}

必须包含的内容：
{must_str or "无"}

不能包含的内容：
{should_not_str or "无"}

请评估回答质量。对于"必须包含"的每一项，判断回答是否覆盖了该要点（可以是不同措辞）。
对于"不能包含"的每一项，判断回答中是否出现了该内容。

返回JSON格式：
{{
  "must_include": {{"<内容>": "pass"|"fail", ...}},
  "should_not_include": {{"<内容>": "pass"|"fail", ...}},
  "all_must_pass": true/false,
  "all_should_not_pass": true/false,
  "reason": "简短评估理由"
}}
"""
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        raw = (getattr(response, "content", "") or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        must_results = {k: v for k, v in result.get("must_include", {}).items()}
        should_not_results = {k: v for k, v in result.get("should_not_include", {}).items()}
        return {
            "method": "llm_judge",
            "must_include": must_results,
            "should_not_include": should_not_results,
            "all_must_pass": result.get("all_must_pass", False),
            "all_should_not_pass": result.get("all_should_not_pass", False),
            "overall_llm_pass": result.get("all_must_pass", False) and result.get("all_should_not_pass", False),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {
            "method": "llm_judge",
            "error": str(e),
            "must_include": {},
            "should_not_include": {},
            "overall_llm_pass": False,
        }


def evaluate_answer_quality(
    sample: Dict[str, Any], final_state: Dict[str, Any], method: str = "string"
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate answer quality: must_include coverage and should_not_include avoidance."""
    expected = sample.get("expected", {})
    must_include = [str(x) for x in (expected.get("answer_must_include") or [])]
    should_not_include = [str(x) for x in (expected.get("answer_should_not_include") or [])]
    answer = (final_state.get("answer") or "").strip()

    if not must_include and not should_not_include:
        return True, {"skipped": True, "reason": "no_answer_checks"}
    if not answer:
        return False, {"skipped": False, "error": "empty_answer", "method": method}

    # Phase 1: always run string matching
    string_result = _check_answer_string(answer, must_include, should_not_include)

    # Phase 2: LLM-as-judge when needed
    if method == "string":
        passed = string_result["overall_string_pass"]
        return passed, string_result

    if method == "llm" or (method == "hybrid" and not string_result["overall_string_pass"]):
        llm_result = _check_answer_llm(
            sample.get("query", ""), answer, must_include, should_not_include
        )
        if method == "llm":
            passed = llm_result.get("overall_llm_pass", False)
            return passed, llm_result
        # hybrid: LLM overrides string when it disagrees
        passed = llm_result.get("overall_llm_pass", string_result["overall_string_pass"])
        llm_result["string_failed"] = not string_result["overall_string_pass"]
        return passed, llm_result

    # string only
    passed = string_result["overall_string_pass"]
    return passed, string_result


# ---------------------------------------------------------------------------
# Dimension: Retry effectiveness
# ---------------------------------------------------------------------------
def evaluate_retry_effectiveness(
    sample: Dict[str, Any], final_state: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate whether web retry improved results."""
    retry_count = int(final_state.get("self_eval_retry_count", 0))

    if retry_count == 0:
        return True, {"skipped": True, "reason": "no_retry_triggered"}

    web_docs = final_state.get("web_search_docs") or []
    reranked = final_state.get("reranked_docs") or []
    self_eval_status = final_state.get("self_eval_status", "")

    has_web_results = len(web_docs) > 0
    improved = self_eval_status == "sufficient"
    web_sourced_in_reranked = sum(1 for d in reranked if d.get("source") == "web")
    expected_plan = sample.get("expected", {}).get("retrieval_plan", {})
    expected_web = expected_plan.get("use_web", False)

    passed = has_web_results or improved

    return passed, {
        "retry_count": retry_count,
        "expected_use_web": expected_web,
        "has_web_results": has_web_results,
        "web_doc_count": len(web_docs),
        "web_sourced_in_reranked": web_sourced_in_reranked,
        "final_self_eval_status": self_eval_status,
        "improved_after_retry": improved,
    }


# ---------------------------------------------------------------------------
# Per-sample evaluation (full graph invocation)
# ---------------------------------------------------------------------------
def evaluate_sample(
    sample: Dict[str, Any],
    dimensions: set,
    resolver: ItemResolver,
    answer_method: str,
    no_web: bool,
    include_not_ready: bool,
    include_judge_artifacts: bool = True,
    artifact_text_chars: int = DEFAULT_ARTIFACT_TEXT_CHARS,
) -> Dict[str, Any]:
    """Run full query graph for one sample and evaluate all requested dimensions."""
    sample_id = sample.get("id", "")
    expected = sample.get("expected", {})
    ready, skip_reason = is_sample_ready(sample)

    result: Dict[str, Any] = {
        "id": sample_id,
        "category": sample.get("category", ""),
        "query": sample.get("query", ""),
        "readiness": {"ready": ready, "skip_reason": skip_reason},
        "session_id": "",
        "status": "passed",
        "primary_failure": "",
        "error": "",
        "timings_ms": {},
        "dimension_results": {},
        "expected": {
            "query_route": expected.get("query_route"),
            "doc_need_route": expected.get("doc_need_route"),
            "retrieval_plan": expected.get("retrieval_plan", {}),
            "item_names": expected_item_names(expected),
            "self_eval_status": expected_self_eval_status(sample),
            "source_hint": expected.get("source_hint", {}),
            "answer_must_include": expected.get("answer_must_include", []),
            "answer_should_not_include": expected.get("answer_should_not_include", []),
        },
        "answer_preview": "",
        "answer": "",
        "trace_summary": {},
        "judge_artifacts": {},
    }

    if not ready and not include_not_ready:
        result["status"] = "skipped"
        result["primary_failure"] = "not_ready"
        return result

    session_id = make_session_id(sample_id)
    result["session_id"] = session_id
    clear_history(session_id)
    seed_history(session_id, sample)

    state = create_query_default_state(
        session_id=session_id,
        thread_id=session_id,
        original_query=sample.get("query", ""),
        is_stream=False,
        trace_started_at=time.time(),
    )

    # If --no-web, force all use_web to False in retrieval plan
    if no_web:
        state["_no_web_override"] = True

    graph_start = time.perf_counter()

    try:
        final_state = query_app.invoke(
            state,
            {
                "recursion_limit": 50,
                "configurable": {"thread_id": session_id},
            },
        )
        result["timings_ms"]["graph_total"] = int((time.perf_counter() - graph_start) * 1000)

        # Answer preview
        answer = final_state.get("answer") or ""
        result["answer"] = answer
        result["answer_preview"] = answer[:ANSWER_EVAL_TEXT_PREVIEW_CHARS]
        result["trace_summary"] = _build_trace_summary(final_state)
        if include_judge_artifacts:
            result["judge_artifacts"] = _build_judge_artifacts(sample, final_state, artifact_text_chars)

        # Evaluate each requested dimension
        dr = result["dimension_results"]

        # --- routing (query_route + item_identity + doc_need_route + retrieval_plan) ---
        if "routing" in dimensions:
            dr["query_route"] = _eval_query_route(expected, final_state)
            dr["item_identity"] = _eval_item_identity(expected, final_state, resolver)
            dr["doc_need_route"] = _eval_doc_need(expected, final_state)
            dr["retrieval_plan"] = _eval_retrieval_plan_check(expected, final_state)

        # --- retrieval quality ---
        if "retrieval" in dimensions:
            rp, rd = evaluate_retrieval_quality(sample, final_state)
            rd["passed"] = rp
            dr["retrieval_quality"] = rd

        # --- self-eval accuracy ---
        if "self_eval" in dimensions:
            sep, sed = evaluate_self_eval_accuracy(sample, final_state)
            sed["passed"] = sep
            dr["self_eval_accuracy"] = sed

        # --- answer quality ---
        if "answer" in dimensions:
            ap, ad = evaluate_answer_quality(sample, final_state, method=answer_method)
            ad["passed"] = ap
            dr["answer_quality"] = ad

        # --- retry effectiveness ---
        if "retry" in dimensions:
            rep, red = evaluate_retry_effectiveness(sample, final_state)
            red["passed"] = rep
            dr["retry_effectiveness"] = red

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result["primary_failure"] = "exception"
        return result
    finally:
        if session_id:
            update_task_status(session_id, TASK_STATUS_COMPLETED, False)
            if not getattr(evaluate_sample, "_keep_history", False):
                clear_history(session_id)

    # Determine overall pass/fail
    non_skipped = [v for v in result["dimension_results"].values() if not v.get("skipped")]
    if non_skipped:
        all_pass = all(v.get("passed", False) for v in non_skipped)
        result["status"] = "passed" if all_pass else "failed"
        result["primary_failure"] = _find_primary_failure(result["dimension_results"])
    else:
        result["status"] = "passed"

    return result


# ---------------------------------------------------------------------------
# Sub-evaluators for routing dimension
# ---------------------------------------------------------------------------
def _eval_query_route(expected: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    actual = state.get("query_route", "")
    exp = expected.get("query_route", "")
    passed = actual == exp
    return {"passed": passed, "expected": exp, "actual": actual}


def _eval_item_identity(expected: Dict[str, Any], state: Dict[str, Any], resolver: ItemResolver) -> Dict[str, Any]:
    actual_names = state.get("item_names") or []
    passed, detail = compare_item_identity(expected, actual_names, resolver)
    return {"passed": passed, **detail}


def _eval_doc_need(expected: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    actual = state.get("doc_need_route", "")
    exp = expected.get("doc_need_route", "")
    passed = actual == exp
    return {"passed": passed, "expected": exp, "actual": actual}


def _eval_retrieval_plan_check(expected: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    actual_plan = state.get("retrieval_plan", {}) or {}
    exp_plan = expected.get("retrieval_plan", {}) or {}
    passed, detail = compare_retrieval_plan(exp_plan, actual_plan)
    return {"passed": passed, **detail}


def _find_primary_failure(dimension_results: Dict[str, Any]) -> str:
    """Return the first failing dimension name."""
    for dim_name in ["query_route", "item_identity", "doc_need_route", "retrieval_plan",
                      "retrieval_quality", "self_eval_accuracy", "answer_quality", "retry_effectiveness"]:
        d = dimension_results.get(dim_name)
        if d and not d.get("skipped") and not d.get("passed"):
            return dim_name
    return ""


def _safe_len(value: Any) -> int:
    try:
        return len(value or [])
    except TypeError:
        return 0


def _truncate_for_artifact(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars and max_chars > 0 and len(value) > max_chars:
        return value[:max_chars] + f"\n...[truncated {len(value) - max_chars} chars]"
    return value


def _as_plain_doc(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        entity = doc.get("entity") if isinstance(doc.get("entity"), dict) else doc
        merged = dict(entity)
        for key in ["score", "distance", "source", "url", "title", "chunk_id", "doc_id"]:
            if key in doc and key not in merged:
                merged[key] = doc.get(key)
        return merged
    return {"raw": str(doc)}


def _artifact_doc(doc: Any, rank: int, max_chars: int) -> Dict[str, Any]:
    item = _as_plain_doc(doc)
    text = item.get("text") or item.get("content") or item.get("snippet") or ""
    return {
        "rank": rank,
        "source": item.get("source") or "",
        "score": item.get("score"),
        "chunk_id": item.get("chunk_id") or item.get("doc_id") or item.get("id") or "",
        "doc_id": item.get("doc_id") or item.get("id") or "",
        "title": item.get("title") or item.get("file_title") or item.get("item_name") or "",
        "canonical_name": item.get("canonical_name") or "",
        "item_name": item.get("item_name") or "",
        "url": item.get("url") or "",
        "text": _truncate_for_artifact(text, max_chars),
        "text_chars": len(str(text or "")),
    }


def _artifact_docs(docs: Any, max_chars: int, limit: int = 20) -> List[Dict[str, Any]]:
    if not isinstance(docs, list):
        return []
    return [_artifact_doc(doc, idx, max_chars) for idx, doc in enumerate(docs[:limit], start=1)]


def _build_judge_artifacts(sample: Dict[str, Any], state: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    expected = sample.get("expected", {}) or {}
    answer = state.get("answer") or ""
    return {
        "artifact_schema": "judge_artifacts.v1",
        "sample": {
            "id": sample.get("id", ""),
            "category": sample.get("category", ""),
            "query": sample.get("query", ""),
            "history": sample.get("history", []),
            "notes": sample.get("notes", ""),
        },
        "expected": {
            "query_route": expected.get("query_route", ""),
            "doc_need_route": expected.get("doc_need_route", ""),
            "retrieval_plan": expected.get("retrieval_plan", {}),
            "item_names": expected_item_names(expected),
            "source_hint": expected.get("source_hint", {}),
            "answer_must_include": expected.get("answer_must_include", []),
            "answer_should_not_include": expected.get("answer_should_not_include", []),
            "self_eval_status": expected_self_eval_status(sample),
        },
        "actual": {
            "query_route": state.get("query_route", ""),
            "doc_need_route": state.get("doc_need_route", ""),
            "retrieval_plan": state.get("retrieval_plan") or {},
            "item_names": state.get("item_names") or [],
            "rewritten_query": state.get("rewritten_query", ""),
            "context_bundle": state.get("context_bundle") or {},
            "product_memory_hints": state.get("product_memory_hints") or {},
            "self_eval_status": state.get("self_eval_status", ""),
            "self_eval_detail": state.get("self_eval_detail") or {},
            "answer": answer,
            "answer_chars": len(answer),
        },
        "retrieval": {
            "embedding_chunks": _artifact_docs(state.get("embedding_chunks"), max_chars),
            "hyde_embedding_chunks": _artifact_docs(state.get("hyde_embedding_chunks"), max_chars),
            "web_search_docs": _artifact_docs(state.get("web_search_docs"), max_chars),
            "rrf_chunks": _artifact_docs(state.get("rrf_chunks"), max_chars),
            "reranked_docs": _artifact_docs(state.get("reranked_docs"), max_chars),
            "context_evidence_docs": _artifact_docs(state.get("context_evidence_docs"), max_chars),
            "rerank_trace": state.get("rerank_trace") or {},
        },
    }


def _build_trace_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("retrieval_plan") or {}
    detail = state.get("self_eval_detail") or {}
    return {
        "thread_id": state.get("thread_id", ""),
        "run_id": state.get("run_id", ""),
        "checkpoint_enabled": bool(state.get("checkpoint_enabled", False)),
        "query_route": state.get("query_route", ""),
        "doc_need_route": state.get("doc_need_route", ""),
        "doc_need_reason": state.get("doc_need_reason", ""),
        "retrieval_plan": {
            "use_embedding": bool(plan.get("use_embedding", False)),
            "use_hyde": bool(plan.get("use_hyde", False)),
            "use_web": bool(plan.get("use_web", False)),
            "reason": plan.get("reason", ""),
            "decided_by": plan.get("decided_by", ""),
        },
        "doc_counts": {
            "embedding": _safe_len(state.get("embedding_chunks")),
            "hyde": _safe_len(state.get("hyde_embedding_chunks")),
            "web": _safe_len(state.get("web_search_docs")),
            "rrf": _safe_len(state.get("rrf_chunks")),
            "reranked": _safe_len(state.get("reranked_docs")),
            "context_evidence": _safe_len(state.get("context_evidence_docs")),
        },
        "self_eval": {
            "status": state.get("self_eval_status", ""),
            "retry_count": int(state.get("self_eval_retry_count", 0)),
            "reason": detail.get("reason", ""),
            "confidence": detail.get("confidence", 0),
        },
        "rerank": state.get("rerank_trace") or {},
    }


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------
def build_comprehensive_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a comprehensive summary with by_category and by_dimension breakdowns."""
    status_counter = Counter(r["status"] for r in results)
    passed_total = status_counter.get("passed", 0)
    failed_total = status_counter.get("failed", 0)
    error_total = status_counter.get("error", 0)

    # Per-category breakdown
    cat_data: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in cat_data:
            cat_data[cat] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0}
        cat_data[cat]["total"] += 1
        cat_data[cat][r["status"]] += 1

    by_category = {}
    for cat, d in sorted(cat_data.items()):
        evaluated = d["passed"] + d["failed"]
        by_category[cat] = {
            "total": d["total"],
            "pass_rate": rate(d["passed"], d["failed"] + d["error"]),
            "quality_pass_rate": rate(d["passed"], d["failed"]),
            "completion_rate": rate(d["passed"] + d["failed"], d["error"]),
            "status": {"passed": d["passed"], "failed": d["failed"], "skipped": d["skipped"], "error": d["error"]},
        }

    # Per-dimension breakdown
    dim_data: Dict[str, Counter] = {}
    for r in results:
        for dim_name, dim_result in r.get("dimension_results", {}).items():
            dim_data.setdefault(dim_name, Counter())
            if r["status"] == "error":
                dim_data[dim_name]["error"] += 1
            elif dim_result.get("skipped"):
                dim_data[dim_name]["skipped"] += 1
            elif dim_result.get("passed"):
                dim_data[dim_name]["passed"] += 1
            else:
                dim_data[dim_name]["failed"] += 1

    by_dimension = {}
    for name, counter in sorted(dim_data.items()):
        p = counter.get("passed", 0)
        f = counter.get("failed", 0)
        by_dimension[name] = {
            "accuracy": rate(p, f),
            "passed": p,
            "failed": f,
            "skipped": counter.get("skipped", 0),
            "error": counter.get("error", 0),
        }

    # Primary failure analysis
    primary_counter = Counter(
        r.get("primary_failure") or "none" for r in results if r.get("status") in ("failed", "error")
    )

    retrieval_benchmark = _aggregate_retrieval_benchmark(results)

    return {
        "total": len(results),
        "evaluated": passed_total + failed_total + status_counter.get("error", 0),
        "status": dict(status_counter),
        "pass_rate": rate(passed_total, failed_total + error_total),
        "quality_pass_rate": rate(passed_total, failed_total),
        "completion_rate": rate(passed_total + failed_total, error_total),
        "by_category": by_category,
        "by_dimension": by_dimension,
        "retrieval_benchmark": retrieval_benchmark,
        "primary_failures": dict(primary_counter),
    }


def _aggregate_retrieval_benchmark(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[float]] = {}
    evaluated = 0
    for result in results:
        benchmark = (
            result.get("dimension_results", {})
            .get("retrieval_quality", {})
            .get("benchmark", {})
        )
        if not benchmark:
            continue
        evaluated += 1
        for key, value in benchmark.items():
            if key in {"benchmark_type", "rank_details"}:
                continue
            if isinstance(value, (int, float)) and value is not None:
                metric_values.setdefault(key, []).append(float(value))

    averages = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(metric_values.items())
        if values
    }
    return {
        "benchmark_type": "source_hint_weak_supervision",
        "evaluated": evaluated,
        "averages": averages,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve() if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve dimensions
    if args.dimensions == "all":
        dimensions = _ALL_DIMENSIONS
    else:
        dimensions = set(d.strip() for d in args.dimensions.split(",") if d.strip())
        invalid = dimensions - _ALL_DIMENSIONS
        if invalid:
            print(f"[ERROR] Unknown dimensions: {invalid}. Valid: {_ALL_DIMENSIONS}")
            sys.exit(1)

    samples = load_jsonl(dataset_path)
    filtered = filter_samples(
        samples=samples,
        categories=args.categories.split(",") if args.categories else [],
        ids=args.ids.split(",") if args.ids else [],
        limit=args.limit,
        include_not_ready=args.include_not_ready,
    )

    if not filtered:
        print("[WARN] No samples matched filters. Check --categories/--ids/--limit.")
        return

    print(f"Evaluating {len(filtered)} samples across dimensions: {dimensions}")
    print(f"Answer method: {args.answer_method}, Concurrency: {args.concurrency}, No-web: {args.no_web}")
    if args.variant_name:
        print(f"Variant: {args.variant_name}")

    resolver = ItemResolver()
    results: List[Dict[str, Any]] = []
    evaluate_sample._keep_history = args.keep_history

    if args.concurrency > 1 and len(filtered) > 1:
        _run_concurrent(filtered, args, dimensions, resolver, results)
    else:
        _run_sequential(filtered, args, dimensions, resolver, results)

    summary = build_comprehensive_summary(results)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "command_args": vars(args),
        "variant_name": args.variant_name,
        "summary": summary,
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print_summary(summary, output_path, title="Comprehensive Eval Summary")

    # Per-dimension print
    print(f"\n{'Dimension':<25} {'Accuracy':<10} {'Passed':<8} {'Failed':<8}")
    print("-" * 55)
    for name, ds in sorted(summary.get("by_dimension", {}).items()):
        acc = f"{ds['accuracy']:.2%}" if ds["accuracy"] is not None else "N/A"
        print(f"{name:<25} {acc:<10} {ds['passed']:<8} {ds['failed']:<8}")


def _run_sequential(filtered, args, dimensions, resolver, results):
    for sample in filtered:
        sid = sample.get("id", "?")
        print(f"\n[{sid}] {sample.get('query', '')[:60]}...")
        result = evaluate_sample(
            sample,
            dimensions,
            resolver,
            args.answer_method,
            args.no_web,
            args.include_not_ready,
            include_judge_artifacts=not args.no_judge_artifacts,
            artifact_text_chars=args.artifact_text_chars,
        )
        results.append(result)
        _print_result_line(result)
        if result["status"] == "error" and args.fail_fast:
            print(f"[FAIL-FAST] Stopping on {sid}: {result.get('error', '')}")
            break


def _run_concurrent(filtered, args, dimensions, resolver, results):
    """Run samples concurrently using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        fut_map = {}
        for sample in filtered:
            sid = sample.get("id", "?")
            fut = executor.submit(
                evaluate_sample,
                sample,
                dimensions,
                resolver,
                args.answer_method,
                args.no_web,
                args.include_not_ready,
                not args.no_judge_artifacts,
                args.artifact_text_chars,
            )
            fut_map[fut] = sid

        for fut in as_completed(fut_map):
            sid = fut_map[fut]
            try:
                result = fut.result()
                results.append(result)
                print(f"\n[{sid}] {result.get('query', '')[:60]}...")
                _print_result_line(result)
            except Exception as e:
                print(f"\n[{sid}] CONCURRENT ERROR: {e}")
                if args.fail_fast:
                    raise


def _print_result_line(result: Dict[str, Any]) -> None:
    status = result["status"]
    pf = result.get("primary_failure", "")
    ts = result.get("timings_ms", {}).get("graph_total", 0)
    marker = "PASS" if status == "passed" else ("FAIL" if status == "failed" else "ERRR")
    print(f"  {marker} status={status} failure={pf} time={ts}ms")


def _default_output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = DEFAULT_OUTPUT_DIR / "eval_comprehensive"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"report_{ts}.json"


if __name__ == "__main__":
    main()
