"""
eval_utils.py — Shared utility functions for Agentic RAG evaluation scripts.

Extracted from eval_routing.py to avoid duplication across eval_comprehensive.py,
eval_routing.py, and other future eval scripts.
"""
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = PROJECT_ROOT / "eval" / "validation_set.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs"
NOT_APPLICABLE = "not_applicable"
NOT_READY_STATUSES = {"missing_document", "not_ingested", "not_ready", "disabled", "skip"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return rows


def is_sample_ready(sample: Dict[str, Any]) -> Tuple[bool, str]:
    if sample.get("enabled") is False:
        return False, sample.get("skip_reason") or "enabled_false"

    expected = sample.get("expected") or {}
    status = str(sample.get("ingestion_status") or expected.get("ingestion_status") or "ready").strip()
    if status in NOT_READY_STATUSES:
        return False, sample.get("skip_reason") or expected.get("skip_reason") or status
    return True, ""


def filter_samples(
    samples: List[Dict[str, Any]],
    categories: List[str],
    ids: List[str],
    limit: int,
    include_not_ready: bool,
) -> List[Dict[str, Any]]:
    id_set = {x.strip() for x in ids if x.strip()}
    category_set = {x.strip() for x in categories if x.strip()}

    result = []
    for sample in samples:
        if id_set and sample.get("id") not in id_set:
            continue
        if category_set and sample.get("category") not in category_set:
            continue
        ready, _ = is_sample_ready(sample)
        if not ready and not include_not_ready:
            continue
        result.append(sample)

    if limit > 0:
        result = result[:limit]
    return result


def normalize_key(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("（", "(").replace("）", ")")
    return "".join(value.upper().split())


def model_tokens(text: str) -> List[str]:
    value = normalize_key(text)
    tokens = []
    for token in re.findall(r"[A-Z]{0,8}\d{2,8}[A-Z0-9-]*", value):
        if len(token) >= 3:
            tokens.append(token)
    return ordered_unique(tokens)


def ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def expected_item_names(expected: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ["canonical_name", "item_name"]:
        if expected.get(key):
            names.append(str(expected[key]))
    names.extend(expected.get("item_names") or [])
    for item in expected.get("items") or []:
        if isinstance(item, dict):
            names.extend([item.get("canonical_name", ""), item.get("item_name", "")])
            names.extend(item.get("aliases") or [])
        elif item:
            names.append(str(item))
    return ordered_unique(names)


def expected_product_ids(expected: Dict[str, Any]) -> List[str]:
    product_ids: List[str] = []
    if expected.get("product_id"):
        product_ids.append(str(expected["product_id"]))
    for item in expected.get("items") or []:
        if isinstance(item, dict) and item.get("product_id"):
            product_ids.append(str(item["product_id"]))
    return ordered_unique(product_ids)


def expected_self_eval_status(sample: Dict[str, Any]) -> str:
    """Return the expected final self-eval status for a sample.

    Explicit labels win. Older validation rows do not contain this field, so
    infer only stable final-state semantics: non-retrieval paths are not
    applicable, no-answer cases should end in clarification, and normal
    retrieval cases should finish with sufficient evidence. Retry triggering
    is evaluated separately.
    """
    expected = sample.get("expected") or {}
    explicit = str(expected.get("self_eval_status") or "").strip()
    if explicit:
        return explicit

    plan = expected.get("retrieval_plan") or {}
    if (
        expected.get("query_route") != "product_doc_qa"
        or expected.get("doc_need_route") != "need_retrieval"
        or not plan.get("use_embedding", False)
    ):
        return "not_applicable"
    if sample.get("category") == "no_answer":
        return "needs_clarification"
    return "sufficient"


class ItemResolver:
    def __init__(self) -> None:
        self.collection_name = os.environ.get("ITEM_NAME_COLLECTION", "")
        self.client = None
        self.cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.loaded = False

    def _lazy_init(self):
        if self.client is None and self.collection_name:
            from app.clients.milvus_utils import get_milvus_client

            self.client = get_milvus_client()

    def _load_once(self) -> None:
        if self.loaded or not self.client or not self.collection_name:
            return
        self.client.load_collection(self.collection_name)
        self.loaded = True

    def resolve(self, name: str) -> Optional[Dict[str, Any]]:
        key = normalize_key(name)
        if not key:
            return None
        if key in self.cache:
            return self.cache[key]

        self._lazy_init()
        if not self.client or not self.collection_name:
            self.cache[key] = None
            return None

        self._load_once()
        from app.utils.escape_milvus_string_utils import escape_milvus_string

        safe_key = escape_milvus_string(key)
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=(
                f'alias_name=="{safe_key}" or '
                f'item_name=="{safe_key}" or '
                f'canonical_name=="{safe_key}" or '
                f'model_code=="{safe_key}"'
            ),
            output_fields=[
                "product_id",
                "canonical_name",
                "alias_name",
                "item_name",
                "brand",
                "category",
                "model_code",
                "alias_type",
                "file_title",
            ],
            limit=5,
        )
        row = rows[0] if rows else None
        self.cache[key] = row
        return row


def resolve_many(resolver: ItemResolver, names: List[str]) -> List[Dict[str, Any]]:
    resolved = []
    for raw_name in ordered_unique(names):
        row = resolver.resolve(raw_name)
        resolved.append({
            "raw": raw_name,
            "normalized": normalize_key(raw_name),
            "resolved": bool(row),
            "product_id": row.get("product_id", "") if row else "",
            "canonical_name": row.get("canonical_name", "") if row else "",
            "alias_name": row.get("alias_name", "") if row else "",
            "alias_type": row.get("alias_type", "") if row else "",
            "brand": row.get("brand", "") if row else "",
            "category": row.get("category", "") if row else "",
            "model_code": row.get("model_code", "") if row else "",
            "file_title": row.get("file_title", "") if row else "",
        })
    return resolved


def compare_item_identity(
    expected: Dict[str, Any], actual_names: List[str], resolver: ItemResolver
) -> Tuple[bool, Dict[str, Any]]:
    expected_names = expected_item_names(expected)
    expected_ids = set(expected_product_ids(expected))
    actual_names = ordered_unique(actual_names or [])

    expected_resolved = resolve_many(resolver, expected_names)
    actual_resolved = resolve_many(resolver, actual_names)

    expected_resolved_ids = {x["product_id"] for x in expected_resolved if x.get("product_id")}
    actual_resolved_ids = {x["product_id"] for x in actual_resolved if x.get("product_id")}
    expected_all_ids = expected_ids | expected_resolved_ids

    expected_canonicals = {normalize_key(x["canonical_name"]) for x in expected_resolved if x.get("canonical_name")}
    if expected.get("canonical_name"):
        expected_canonicals.add(normalize_key(expected.get("canonical_name")))
    actual_canonicals = {normalize_key(x["canonical_name"]) for x in actual_resolved if x.get("canonical_name")}
    actual_aliases = {normalize_key(x["alias_name"]) for x in actual_resolved if x.get("alias_name")}

    expected_raw_norm = {normalize_key(x) for x in expected_names}
    actual_raw_norm = {normalize_key(x) for x in actual_names}
    actual_identity_keys = actual_raw_norm | actual_canonicals | actual_aliases
    expected_model_tokens = set()
    for name in expected_names:
        expected_model_tokens.update(model_tokens(name))
    actual_model_tokens = set()
    for key in actual_identity_keys:
        actual_model_tokens.update(model_tokens(key))
    for row in actual_resolved:
        actual_model_tokens.update(model_tokens(row.get("model_code", "")))

    matched_expected: List[str] = []
    missing_expected: List[str] = []
    expected_resolution_by_raw = {x["raw"]: x for x in expected_resolved}

    for expected_name in expected_names:
        expected_key = normalize_key(expected_name)
        resolved = expected_resolution_by_raw.get(expected_name) or {}
        expected_id = resolved.get("product_id", "")
        expected_canonical = normalize_key(resolved.get("canonical_name", ""))
        tokens = set(model_tokens(expected_name))
        tokens.update(model_tokens(resolved.get("model_code", "")))

        matched = False
        if expected_id and expected_id in actual_resolved_ids:
            matched = True
        elif expected_canonical and expected_canonical in actual_canonicals:
            matched = True
        elif expected_key and any(
            expected_key in actual_key or actual_key in expected_key
            for actual_key in actual_identity_keys
        ):
            matched = True
        elif tokens and tokens & actual_model_tokens:
            matched = True

        (matched_expected if matched else missing_expected).append(expected_name)

    passed = not missing_expected if expected_names else not actual_names
    match_type = "all_expected_items" if expected_names else "expected_empty"

    return passed, {
        "match_type": match_type,
        "expected_raw": expected_names,
        "actual_raw": actual_names,
        "expected_product_ids": sorted(expected_all_ids),
        "actual_product_ids": sorted(actual_resolved_ids),
        "expected_canonical_keys": sorted(expected_canonicals),
        "actual_canonical_keys": sorted(actual_canonicals),
        "actual_alias_keys": sorted(actual_aliases),
        "expected_model_tokens": sorted(expected_model_tokens),
        "actual_model_tokens": sorted(actual_model_tokens),
        "matched_expected_items": matched_expected,
        "missing_expected_items": missing_expected,
        "expected_item_recall": round(len(matched_expected) / len(expected_names), 4) if expected_names else 1.0,
        "expected_resolved": expected_resolved,
        "actual_resolved": actual_resolved,
        "missing_raw_keys": sorted(expected_raw_norm - actual_raw_norm),
        "unexpected_raw_keys": sorted(actual_raw_norm - expected_raw_norm),
    }


def compare_retrieval_plan(expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    hard_keys = ["use_embedding", "use_hyde", "use_web"]
    soft_keys: List[str] = []
    hard_diff = {}
    soft_diff = {}

    for key in hard_keys:
        if expected.get(key) != actual.get(key):
            hard_diff[key] = {"expected": expected.get(key), "actual": actual.get(key)}
    for key in soft_keys:
        if expected.get(key) != actual.get(key):
            soft_diff[key] = {"expected": expected.get(key), "actual": actual.get(key)}

    return len(hard_diff) == 0, {
        "hard_diff": hard_diff,
        "soft_diff": soft_diff,
        "reason": {"expected": expected.get("reason"), "actual": actual.get("reason")},
    }


def seed_history(session_id: str, sample: Dict[str, Any]) -> None:
    from app.clients.mongo_history_utils import save_chat_message

    for msg in sample.get("history") or []:
        save_chat_message(
            session_id=session_id,
            role=msg.get("role", "user"),
            text=msg.get("content") or msg.get("text") or "",
            rewritten_query=msg.get("rewritten_query", ""),
            item_names=msg.get("item_names") or [],
            image_urls=msg.get("image_urls") or [],
            evidence_docs=msg.get("evidence_docs") or [],
            route_meta=msg.get("route_meta") or {},
        )


def make_session_id(sample_id: str) -> str:
    return f"eval-{sample_id}-{uuid.uuid4().hex[:8]}"


def timed_call(name: str, timings: Dict[str, int], fn, state: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    new_state = fn(state)
    timings[name] = int((time.perf_counter() - start) * 1000)
    return new_state


def first_primary_failure(checks: Dict[str, Dict[str, Any]]) -> str:
    for name in ["query_route", "item_identity", "doc_need_route", "retrieval_plan"]:
        check = checks.get(name)
        if check and not check.get("skipped") and not check.get("passed"):
            return name
    return ""


def rate(passed: int, failed: int) -> Optional[float]:
    denom = passed + failed
    return round(passed / denom, 4) if denom else None


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counter = Counter(r["status"] for r in results)
    primary_counter = Counter(r.get("primary_failure") or "none" for r in results if r.get("status") == "failed")
    check_counter: Dict[str, Counter] = {}

    for result in results:
        for check_name, check in result.get("checks", {}).items():
            check_counter.setdefault(check_name, Counter())
            if result["status"] == "error":
                check_counter[check_name]["error"] += 1
            elif check.get("skipped"):
                check_counter[check_name]["skipped"] += 1
            elif check.get("passed"):
                check_counter[check_name]["passed"] += 1
            else:
                check_counter[check_name]["failed"] += 1

    check_summary = {}
    for name, counter in check_counter.items():
        passed = counter.get("passed", 0)
        failed = counter.get("failed", 0)
        check_summary[name] = {**dict(counter), "accuracy": rate(passed, failed)}

    passed_total = status_counter.get("passed", 0)
    failed_total = status_counter.get("failed", 0)
    return {
        "total": len(results),
        "evaluated": passed_total + failed_total + status_counter.get("error", 0),
        "status": dict(status_counter),
        "pass_rate": rate(passed_total, failed_total),
        "checks": check_summary,
        "primary_failures": dict(primary_counter),
    }


def print_summary(summary: Dict[str, Any], output_path: Path, title: str = "Eval Summary") -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"total:     {summary.get('total', 0)}")
    print(f"evaluated: {summary.get('evaluated', 0)}")
    print(f"status:    {summary.get('status', {})}")
    print(f"pass_rate: {summary.get('pass_rate')}")
    if "checks" in summary:
        print(f"checks:    {summary.get('checks', {})}")
    if "primary_failures" in summary:
        print(f"primary:   {summary.get('primary_failures', {})}")
    print(f"output:    {output_path}")


def default_output_path(subdir: str = "eval_routing") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / subdir / f"report_{ts}.json"
