"""Summarize failed dimensions from a comprehensive evaluation report."""

import argparse
import json
from collections import Counter
from pathlib import Path


def _failed(results: list[dict], dimension: str) -> list[dict]:
    """Return non-skipped samples that failed a requested dimension."""
    return [
        result
        for result in results
        if (details := result.get("dimension_results", {}).get(dimension, {})).get("passed") is False
        and not details.get("skipped")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize failed evaluation dimensions.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("logs/eval_comprehensive/report_rejudged.json"),
        help="Path to a comprehensive evaluation report.",
    )
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    results = data["results"]

    retrieval_failures = _failed(results, "retrieval_quality")
    self_eval_failures = _failed(results, "self_eval_accuracy")
    answer_failures = _failed(results, "answer_quality")

    failure_types = Counter(
        result["dimension_results"]["self_eval_accuracy"].get("failure_type", "?")
        for result in self_eval_failures
    )
    print(f"retrieval_quality: {len(retrieval_failures)} failures")
    print(f"self_eval_accuracy: {len(self_eval_failures)} failures")
    print(f"  types: {dict(failure_types)}")
    print(f"answer_quality: {len(answer_failures)} failures")

    print("\n--- retrieval_quality examples ---")
    for result in retrieval_failures[:3]:
        details = result["dimension_results"]["retrieval_quality"]
        print(f"  [{result['id']}] {result['query'][:40]}")
        print(f"    title_hits: {details.get('title_hits', {})}")
        print(f"    keyword_hits: {details.get('keyword_hits', {})}")

    print("\n--- self_eval_accuracy examples ---")
    for result in self_eval_failures[:5]:
        details = result["dimension_results"]["self_eval_accuracy"]
        print(f"  [{result['id']}] {result['query'][:40]}")
        print(
            f"    expected={details.get('expected')} actual={details.get('actual')} "
            f"type={details.get('failure_type')}"
        )

    print("\n--- answer_quality examples ---")
    for result in answer_failures[:5]:
        details = result["dimension_results"]["answer_quality"]
        missing = {
            key: value
            for key, value in details.get("must_include", {}).items()
            if value != "found"
        }
        violations = {
            key: value
            for key, value in details.get("should_not_include", {}).items()
            if value == "found_violation"
        }
        print(f"  [{result['id']}] {result['query'][:40]}")
        if missing:
            print(f"    missing: {missing}")
        if violations:
            print(f"    violations: {violations}")

    categories = Counter(
        result.get("category", "?")
        for result in results
        if result.get("status") == "failed"
    )
    print("\n--- failures by category ---")
    for category, count in sorted(categories.items(), key=lambda item: -item[1]):
        total = sum(1 for result in results if result.get("category") == category)
        print(f"  {category}: {count}/{total}")


if __name__ == "__main__":
    main()
