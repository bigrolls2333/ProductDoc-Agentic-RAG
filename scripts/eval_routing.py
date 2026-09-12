"""
scripts/eval_routing.py — Query Graph routing evaluation (query_route, item_names, doc_need_route, retrieval_plan).

Run individual nodes step-by-step (no full graph invocation, cheap):
    python scripts/eval_routing.py
    python scripts/eval_routing.py --intent-only
    python scripts/eval_routing.py --limit 10 --categories smalltalk,abuse_or_invalid
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mongo_history_utils import clear_history
from app.query_process.agent.nodes.node_doc_need_router import node_doc_need_router
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_intent import node_query_intent
from app.query_process.agent.nodes.node_retrieval_plan import node_retrieval_plan
from app.query_process.agent.state import create_query_default_state
from eval_utils import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    NOT_APPLICABLE,
    ItemResolver,
    build_summary,
    compare_item_identity,
    compare_retrieval_plan,
    default_output_path,
    expected_item_names,
    expected_product_ids,
    filter_samples,
    first_primary_failure,
    is_sample_ready,
    load_jsonl,
    make_session_id,
    print_summary,
    seed_history,
    timed_call,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run routing evaluation for the Query Graph.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to validation_set.jsonl")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to logs/eval_routing_*.json")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N active samples after filtering")
    parser.add_argument("--categories", default="", help="Comma-separated categories to include")
    parser.add_argument("--ids", default="", help="Comma-separated sample ids to include")
    parser.add_argument("--intent-only", action="store_true", help="Only evaluate query_route")
    parser.add_argument("--skip-item-names", action="store_true", help="Skip item identity comparison but still run downstream routing")
    parser.add_argument("--inject-expected-items", action="store_true", help="Inject expected item_names after item node to isolate doc_need/retrieval_plan")
    parser.add_argument("--include-not-ready", action="store_true", help="Evaluate samples marked as missing_document/not_ready instead of skipping them")
    parser.add_argument("--keep-history", action="store_true", help="Keep seeded eval history in Mongo for debugging")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first exception")
    return parser.parse_args()


def evaluate_sample(sample: Dict[str, Any], args: argparse.Namespace, resolver: ItemResolver) -> Dict[str, Any]:
    sample_id = sample.get("id", "")
    expected = sample.get("expected") or {}
    ready, skip_reason = is_sample_ready(sample)

    result: Dict[str, Any] = {
        "id": sample_id,
        "category": sample.get("category", ""),
        "query": sample.get("query", ""),
        "readiness": {"ready": ready, "skip_reason": skip_reason},
        "session_id": "",
        "expected": {
            "query_route": expected.get("query_route"),
            "doc_need_route": expected.get("doc_need_route"),
            "retrieval_plan": expected.get("retrieval_plan", {}),
            "item_names": expected_item_names(expected),
            "product_ids": expected_product_ids(expected),
            "canonical_name": expected.get("canonical_name", ""),
        },
        "actual": {
            "query_route": "",
            "doc_need_route": "",
            "retrieval_plan": {},
            "item_names": [],
        },
        "checks": {},
        "primary_failure": "",
        "timings_ms": {},
        "status": "passed",
        "error": "",
    }

    if not ready and not args.include_not_ready:
        result["status"] = "skipped"
        result["primary_failure"] = "not_ready"
        return result

    session_id = make_session_id(sample_id)
    result["session_id"] = session_id
    clear_history(session_id)
    seed_history(session_id, sample)

    state = create_query_default_state(
        session_id=session_id,
        original_query=sample.get("query", ""),
        is_stream=False,
    )

    try:
        state = timed_call("query_intent", result["timings_ms"], node_query_intent, state)
        actual_query_route = state.get("query_route", "")
        result["actual"]["query_route"] = actual_query_route
        result["checks"]["query_route"] = {
            "passed": actual_query_route == expected.get("query_route"),
            "expected": expected.get("query_route"),
            "actual": actual_query_route,
        }

        if args.intent_only:
            result["status"] = "passed" if result["checks"]["query_route"]["passed"] else "failed"
            result["primary_failure"] = first_primary_failure(result["checks"])
            return result

        if actual_query_route == "product_doc_qa":
            state = timed_call("item_name_confirm", result["timings_ms"], node_item_name_confirm, state)
            actual_item_names = state.get("item_names", []) or []
            result["actual"]["item_names"] = actual_item_names

            if args.skip_item_names:
                result["checks"]["item_identity"] = {"passed": True, "skipped": True, "reason": "skip_item_names"}
            else:
                item_passed, item_detail = compare_item_identity(expected, actual_item_names, resolver)
                result["checks"]["item_identity"] = {"passed": item_passed, **item_detail}

            if args.inject_expected_items:
                injected = expected.get("item_names") or expected_item_names(expected)
                state["item_names"] = injected
                result["actual"]["item_names_after_injection"] = injected

            state = timed_call("doc_need_router", result["timings_ms"], node_doc_need_router, state)
            actual_doc_need = state.get("doc_need_route", "")
            result["actual"]["doc_need_route"] = actual_doc_need
            expected_doc_need = expected.get("doc_need_route")
            result["checks"]["doc_need_route"] = {
                "passed": actual_doc_need == expected_doc_need,
                "expected": expected_doc_need,
                "actual": actual_doc_need,
            }

            if actual_doc_need == "need_retrieval":
                state = timed_call("retrieval_plan", result["timings_ms"], node_retrieval_plan, state)
                actual_plan = state.get("retrieval_plan", {}) or {}
                result["actual"]["retrieval_plan"] = actual_plan
                plan_passed, plan_detail = compare_retrieval_plan(expected.get("retrieval_plan", {}) or {}, actual_plan)
                result["checks"]["retrieval_plan"] = {"passed": plan_passed, **plan_detail}
            else:
                result["checks"]["retrieval_plan"] = {
                    "passed": expected_doc_need != "need_retrieval",
                    "skipped": True,
                    "reason": "doc_need_route_skips_new_retrieval",
                    "expected_doc_need_route": expected_doc_need,
                    "actual_doc_need_route": actual_doc_need,
                }
        else:
            expected_doc_need = expected.get("doc_need_route")
            result["checks"]["item_identity"] = {"passed": True, "skipped": True, "reason": "non_product_doc_qa_route"}
            result["checks"]["doc_need_route"] = {
                "passed": expected_doc_need == NOT_APPLICABLE,
                "expected": expected_doc_need,
                "actual": NOT_APPLICABLE,
                "skipped": True,
            }
            result["checks"]["retrieval_plan"] = {"passed": True, "skipped": True, "reason": "non_product_doc_qa_route"}

    except Exception as e:
        result["status"] = "error"
        result["error"] = repr(e)
        result["primary_failure"] = "exception"
        if args.fail_fast:
            raise
    finally:
        if not args.keep_history and session_id:
            clear_history(session_id)

    if result["status"] != "error":
        active_checks = [v for v in result["checks"].values() if not v.get("skipped")]
        result["status"] = "passed" if active_checks and all(v.get("passed") for v in active_checks) else "failed"
        result["primary_failure"] = first_primary_failure(result["checks"])
    return result


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output_path("eval_routing")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(dataset_path)
    filtered = filter_samples(
        samples=samples,
        categories=args.categories.split(",") if args.categories else [],
        ids=args.ids.split(",") if args.ids else [],
        limit=args.limit,
        include_not_ready=args.include_not_ready,
    )

    resolver = ItemResolver()
    results = [evaluate_sample(sample, args, resolver) for sample in filtered]
    summary = build_summary(results)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "filters": {
            "categories": args.categories,
            "ids": args.ids,
            "limit": args.limit,
            "intent_only": args.intent_only,
            "skip_item_names": args.skip_item_names,
            "inject_expected_items": args.inject_expected_items,
            "include_not_ready": args.include_not_ready,
        },
        "summary": summary,
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print_summary(summary, output_path, title="Routing Eval Summary")


if __name__ == "__main__":
    # Need datetime import here since it's used inline in main()
    from datetime import datetime
    main()
