"""Pure legacy scoring helpers, also usable for offline rescoring of saved runs."""

import copy
from collections import Counter, defaultdict


FAILURE_ORDER = ("query_route", "item_identity", "doc_need_route", "retrieval_plan",
                 "retrieval_quality", "self_eval_accuracy", "answer_quality", "retry_effectiveness")


def compare_doc_need(expected, actual):
    if expected == "not_applicable" and actual in {"", None, "not_applicable"}:
        return {"passed": True, "skipped": True, "expected": expected, "actual": actual,
                "reason": "document_router_not_applicable"}
    return {"passed": actual == expected, "expected": expected, "actual": actual}


def compare_plan(expected, actual):
    differences = {}
    for key in ("use_embedding", "use_hyde", "use_web"):
        left, right = bool(expected.get(key, False)), bool(actual.get(key, False))
        if left != right:
            differences[key] = {"expected": expected.get(key), "actual": actual.get(key)}
    return not differences, {"hard_diff": differences, "soft_diff": {},
                             "reason": {"expected": expected.get("reason"), "actual": actual.get("reason")}}


def summarize(results):
    def ratio(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None
    status = Counter(row["status"] for row in results)
    categories, dimensions = defaultdict(Counter), defaultdict(Counter)
    metrics = defaultdict(list)
    benchmark_count = 0
    for row in results:
        categories[row.get("category", "unknown")][row["status"]] += 1
        for name, check in row.get("dimension_results", {}).items():
            outcome = "error" if row["status"] == "error" else "skipped" if check.get("skipped") else "passed" if check.get("passed") else "failed"
            dimensions[name][outcome] += 1
        benchmark = row.get("dimension_results", {}).get("retrieval_quality", {}).get("benchmark", {})
        if benchmark:
            benchmark_count += 1
            for key, value in benchmark.items():
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
    by_category = {}
    for name, counts in sorted(categories.items()):
        completed = counts["passed"] + counts["failed"]
        by_category[name] = {"total": sum(counts.values()), "status": {key: counts[key] for key in ("passed", "failed", "error", "skipped")},
                             "pass_rate": ratio(counts["passed"], completed + counts["error"]),
                             "quality_pass_rate": ratio(counts["passed"], completed),
                             "completion_rate": ratio(completed, completed + counts["error"])}
    by_dimension = {name: {"accuracy": ratio(counts["passed"], counts["passed"] + counts["failed"]),
                            **{key: counts[key] for key in ("passed", "failed", "error", "skipped")}}
                    for name, counts in sorted(dimensions.items())}
    completed = status["passed"] + status["failed"]
    return {"total": len(results), "evaluated": completed + status["error"], "status": dict(status),
            "pass_rate": ratio(status["passed"], completed + status["error"]),
            "quality_pass_rate": ratio(status["passed"], completed),
            "completion_rate": ratio(completed, completed + status["error"]),
            "by_category": by_category, "by_dimension": by_dimension,
            "primary_failures": dict(Counter(row.get("primary_failure") or "none" for row in results if row["status"] in {"failed", "error"})),
            "retrieval_benchmark": {"benchmark_type": "source_hint_weak_supervision", "evaluated": benchmark_count,
                                     "averages": {key: round(sum(values) / len(values), 6) for key, values in sorted(metrics.items())}}}


def rescore(report):
    output = copy.deepcopy(report)
    adjusted = 0
    for row in output["results"]:
        if row["status"] in {"error", "skipped"}:
            continue
        checks = row["dimension_results"]
        before = copy.deepcopy(checks)
        if "doc_need_route" in checks:
            check = checks["doc_need_route"]
            checks["doc_need_route"] = compare_doc_need(check["expected"], check["actual"])
        if "retrieval_plan" in checks:
            actual = row.get("judge_artifacts", {}).get("actual", {}).get("retrieval_plan")
            if actual is None:
                raise ValueError("offline rescore requires retained actual retrieval_plan artifacts")
            passed, detail = compare_plan(row["expected"]["retrieval_plan"], actual)
            checks["retrieval_plan"] = dict(detail, passed=passed)
        if checks != before:
            row["scoring_adjustments"] = {key: value for key, value in before.items() if value != checks.get(key)}
            adjusted += 1
        row["original_status"] = row["status"]
        active = [check for check in checks.values() if not check.get("skipped")]
        row["status"] = "passed" if active and all(check.get("passed") for check in active) else "failed"
        row["primary_failure"] = next((name for name in FAILURE_ORDER if name in checks and not checks[name].get("skipped") and not checks[name].get("passed")), "")
    output["original_summary"] = report["summary"]
    output["summary"] = summarize(output["results"])
    output["rescoring"] = {"version": "legacy-routing-semantics.v1", "adjusted_samples": adjusted,
                           "model_calls": 0, "reason": "Skip inapplicable document routes; absent tool flags mean disabled. No answer or retrieval labels changed."}
    return output
