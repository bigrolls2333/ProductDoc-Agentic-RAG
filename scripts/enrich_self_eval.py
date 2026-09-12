"""
enrich_self_eval.py — Infer expected.self_eval_status for each sample in the validation set.

Usage:
    python scripts/enrich_self_eval.py
    python scripts/enrich_self_eval.py --input eval/validation_set.jsonl --output eval/validation_set_enriched.jsonl
    python scripts/enrich_self_eval.py --manual-overrides '{"VAL-099":"sufficient","VAL-100":"needs_clarification"}'
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_utils import load_jsonl


def infer_self_eval_status(sample: Dict[str, Any]) -> str:
    """Infer expected self_eval_status from existing sample fields."""
    expected = sample.get("expected", {})
    route = expected.get("query_route", "")
    doc_need = expected.get("doc_need_route", "")
    retrieval_plan = expected.get("retrieval_plan", {})
    must_include = [str(x) for x in (expected.get("answer_must_include") or [])]
    source_hint = expected.get("source_hint", {})
    category = sample.get("category", "")

    # Rule 1: Not applicable — self-eval won't run meaningfully
    if route != "product_doc_qa" or doc_need != "need_retrieval":
        return "not_applicable"
    if not retrieval_plan.get("use_embedding", False):
        return "not_applicable"

    # Rule 2: Needs clarification — no-answer / rejection samples
    no_answer_markers = ["文档未提及", "无法确认", "拒绝回答", "找不到", "不支持", "未找到"]
    if any(marker in " ".join(must_include) for marker in no_answer_markers):
        return "needs_clarification"

    # Rule 3: Insufficient retry — web-required samples
    if retrieval_plan.get("use_web", False) or category == "web_required":
        return "insufficient_retry"

    # Rule 4: Sufficient — normal product QA with source_hint
    if source_hint.get("doc_title_contains") or source_hint.get("chunk_keywords"):
        return "sufficient"

    # Fallback
    return "sufficient"


def parse_manual_overrides(raw: str) -> Dict[str, str]:
    """Parse --manual-overrides JSON string like '{"VAL-099":"sufficient"}'."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] Failed to parse --manual-overrides: {e}. Ignoring.")
        return {}


def print_validation_table(samples: List[Dict[str, Any]]) -> None:
    """Print a table of inferred self_eval_status for manual review."""
    print()
    print(f"{'Sample':<12} {'Category':<20} {'Inferred Status':<20} {'Rule':<10}")
    print("-" * 65)
    for s in samples:
        status = s.get("expected", {}).get("self_eval_status", "?")
        route = s.get("expected", {}).get("query_route", "")
        doc_need = s.get("expected", {}).get("doc_need_route", "")
        category = s.get("category", "")
        rule = "6-fallback"
        if status == "not_applicable":
            if route != "product_doc_qa":
                rule = "1a-non_product"
            elif doc_need != "need_retrieval":
                rule = "1b-no_retrieval"
            else:
                rule = "1c-no_embedding"
        elif status == "needs_clarification":
            rule = "2-no_answer"
        elif status == "insufficient_retry":
            rule = "3-web_req"
        elif status == "sufficient":
            rule = "4-normal_qa"
        print(f"{s.get('id','?'):<12} {category:<20} {status:<20} {rule:<10}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer expected.self_eval_status for validation set")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "eval" / "validation_set.jsonl"), help="Input JSONL path")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "eval" / "validation_set_enriched.jsonl"), help="Output JSONL path")
    parser.add_argument("--manual-overrides", default="", help='JSON string of manual overrides e.g. \'{"VAL-099":"sufficient"}\'')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    overrides = parse_manual_overrides(args.manual_overrides)

    samples = load_jsonl(input_path)

    updated = 0
    for sample in samples:
        expected = sample.setdefault("expected", {})
        old_status = expected.get("self_eval_status")
        sample_id = sample.get("id", "")

        # Apply manual override first
        if sample_id in overrides:
            expected["self_eval_status"] = overrides[sample_id]
        else:
            expected["self_eval_status"] = infer_self_eval_status(sample)

        if expected["self_eval_status"] != old_status:
            updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Enriched {len(samples)} samples → {output_path}")
    print(f"Updated self_eval_status: {updated}")
    print(f"Manual overrides applied: {len(overrides)}")

    # Show distribution
    counts: Dict[str, int] = {}
    for s in samples:
        st = s.get("expected", {}).get("self_eval_status", "unknown")
        counts[st] = counts.get(st, 0) + 1
    print(f"Distribution: {json.dumps(counts, ensure_ascii=False)}")

    print_validation_table(samples)


if __name__ == "__main__":
    main()
