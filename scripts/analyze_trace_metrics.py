"""
Aggregate trace_summary fields from eval_comprehensive or eval_ablation reports.

Examples:
    python scripts/analyze_trace_metrics.py --input logs/eval_ablation/run_xxx/baseline.json
    python scripts/analyze_trace_metrics.py --input logs/eval_ablation/run_xxx/*.json
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trace metrics from eval reports.")
    parser.add_argument("--input", nargs="+", required=True, help="Report JSON paths or glob patterns")
    parser.add_argument("--output", default="", help="Output JSON path")
    return parser.parse_args()


def _expand_inputs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m).resolve() for m in matches)
        else:
            paths.append(Path(pattern).resolve())
    return sorted(dict.fromkeys(paths))


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(num: int, denom: int):
    return round(num / denom, 4) if denom else None


def _latency(values: List[int]) -> Dict[str, Any]:
    values = sorted(v for v in values if v >= 0)
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None}
    p95_idx = min(len(values) - 1, int(round(len(values) * 0.95)) - 1)
    return {
        "avg_ms": round(sum(values) / len(values), 2),
        "p50_ms": int(median(values)),
        "p95_ms": values[p95_idx],
    }


def analyze_report(path: Path) -> Dict[str, Any]:
    report = _load(path)
    results = report.get("results") or []
    route_counter = Counter()
    planner_counter = Counter()
    self_eval_counter = Counter()
    primary_counter = Counter()
    doc_counts = Counter()
    latencies = []
    retry_total = 0
    evaluated = 0
    rerank_input_total = 0
    rerank_output_total = 0
    rerank_gap_triggered = 0
    rerank_modes = Counter()
    rerank_output_counts = Counter()

    for row in results:
        if row.get("status") in {"skipped"}:
            continue
        evaluated += 1
        trace = row.get("trace_summary") or {}
        route_counter[trace.get("doc_need_route", "")] += 1
        plan = trace.get("retrieval_plan") or {}
        planner_counter[plan.get("decided_by", "") or "unknown"] += 1
        self_eval = trace.get("self_eval") or {}
        self_eval_counter[self_eval.get("status", "") or "unknown"] += 1
        retry_total += int(self_eval.get("retry_count") or 0)
        if row.get("primary_failure"):
            primary_counter[row.get("primary_failure")] += 1
        counts = trace.get("doc_counts") or {}
        for key in ["embedding", "hyde", "web", "rrf", "reranked", "context_evidence"]:
            doc_counts[f"{key}_docs"] += int(counts.get(key) or 0)
        rerank = trace.get("rerank") or {}
        if rerank:
            rerank_input_total += int(rerank.get("input_docs") or 0)
            rerank_output_total += int(rerank.get("output_docs") or 0)
            rerank_gap_triggered += 1 if rerank.get("gap_triggered") else 0
            rerank_modes[rerank.get("mode") or "unknown"] += 1
            rerank_output_counts[str(rerank.get("output_docs") or 0)] += 1
        timing = row.get("timings_ms") or {}
        if timing.get("graph_total") is not None:
            latencies.append(int(timing.get("graph_total") or 0))

    return {
        "path": str(path),
        "variant_name": report.get("variant_name") or (report.get("command_args") or {}).get("variant_name", ""),
        "evaluated": evaluated,
        "pass_rate": (report.get("summary") or {}).get("pass_rate"),
        "latency": _latency(latencies),
        "avg_retry_count": round(retry_total / evaluated, 3) if evaluated else None,
        "doc_need_route": dict(route_counter),
        "planner": dict(planner_counter),
        "self_eval_status": dict(self_eval_counter),
        "primary_failures": dict(primary_counter),
        "avg_doc_counts": {
            key: round(value / evaluated, 3) if evaluated else None
            for key, value in doc_counts.items()
        },
        "checkpoint_thread_rate": _rate(
            sum(1 for row in results if (row.get("trace_summary") or {}).get("thread_id")),
            evaluated,
        ),
        "rerank": {
            "avg_input_docs": round(rerank_input_total / evaluated, 3) if evaluated else None,
            "avg_output_docs": round(rerank_output_total / evaluated, 3) if evaluated else None,
            "gap_trigger_rate": _rate(rerank_gap_triggered, evaluated),
            "mode": dict(rerank_modes),
            "output_doc_count_distribution": dict(rerank_output_counts),
        },
    }


def main() -> None:
    args = parse_args()
    paths = _expand_inputs(args.input)
    excluded = {"summary.json", "trace_metrics.json"}
    summaries = [analyze_report(path) for path in paths if path.exists() and path.name not in excluded]
    payload = {"reports": summaries}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
