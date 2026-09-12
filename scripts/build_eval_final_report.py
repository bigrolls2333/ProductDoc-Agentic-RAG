"""
Build a final Markdown report from eval_ablation variant JSON files.

Usage:
    python scripts/build_eval_final_report.py --run-dir logs/eval_ablation/run_YYYYMMDD_HHMMSS
"""
import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


VARIANT_ORDER = [
    "baseline",
    "no_rerank",
    "no_hyde",
    "no_self_eval",
    "no_retry",
    "no_model_guard",
    "history_only",
    "no_context_memory",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final evaluation report.")
    parser.add_argument("--run-dir", required=True, help="logs/eval_ablation/run_* directory")
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _pp(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:+.2f}pp"


def _latency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = sorted(
        int((r.get("timings_ms") or {}).get("graph_total") or 0)
        for r in results
        if r.get("status") in {"passed", "failed"} and (r.get("timings_ms") or {}).get("graph_total") is not None
    )
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None}
    p95_idx = min(len(values) - 1, int(round(len(values) * 0.95)) - 1)
    return {"avg_ms": round(sum(values) / len(values), 2), "p50_ms": int(median(values)), "p95_ms": values[p95_idx]}


def _variant_row(name: str, report: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary") or {}
    dims = summary.get("by_dimension") or {}
    bench = ((summary.get("retrieval_benchmark") or {}).get("averages") or {})
    base_pass = (baseline.get("summary") or {}).get("pass_rate")
    pass_rate = summary.get("pass_rate")
    delta = None
    if pass_rate is not None and base_pass is not None:
        delta = round(float(pass_rate) - float(base_pass), 4)
    return {
        "variant": name,
        "pass_rate": pass_rate,
        "delta_vs_baseline": delta,
        "status": summary.get("status", {}),
        "routing": _combined_routing_accuracy(dims),
        "retrieval_accuracy": (dims.get("retrieval_quality") or {}).get("accuracy"),
        "hit_at_1": bench.get("hit@1"),
        "source_hint_recall_at_5": bench.get("source_hint_recall@5"),
        "mrr_at_5": bench.get("mrr@5"),
        "ndcg_at_5": bench.get("ndcg@5"),
        "answer_accuracy": (dims.get("answer_quality") or {}).get("accuracy"),
        "self_eval_accuracy": (dims.get("self_eval_accuracy") or {}).get("accuracy"),
        "retry_accuracy": (dims.get("retry_effectiveness") or {}).get("accuracy"),
        "latency": _latency(report.get("results") or []),
        "primary_failures": summary.get("primary_failures", {}),
    }


def _combined_routing_accuracy(dims: Dict[str, Any]) -> Any:
    keys = ["query_route", "item_identity", "doc_need_route", "retrieval_plan"]
    passed = failed = 0
    for key in keys:
        data = dims.get(key) or {}
        passed += int(data.get("passed") or 0)
        failed += int(data.get("failed") or 0)
    return round(passed / (passed + failed), 4) if passed + failed else None


def _load_reports(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    reports = {}
    for name in VARIANT_ORDER:
        path = run_dir / f"{name}.json"
        if path.exists():
            reports[name] = _load(path)
    return reports


def _write_summary_files(run_dir: Path, reports: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    summary = {
        "run_dir": str(run_dir),
        "variants": list(reports.keys()),
        "rows": rows,
        "reports": {name: str(run_dir / f"{name}.json") for name in reports},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Agentic RAG Ablation Summary",
        "",
        "| Variant | Pass Rate | Delta | Routing | Retrieval | Hit@1 | Recall@5 | MRR@5 | nDCG@5 | Answer | Self-Eval | Retry | P50 Latency | P95 Latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        latency = row.get("latency") or {}
        lines.append(
            "| {variant} | {pass_rate} | {delta} | {routing} | {retrieval} | {hit1} | {recall5} | {mrr5} | {ndcg5} | {answer} | {self_eval} | {retry} | {p50}ms | {p95}ms |".format(
                variant=row["variant"],
                pass_rate=_pct(row.get("pass_rate")),
                delta=_pp(row.get("delta_vs_baseline")),
                routing=_pct(row.get("routing")),
                retrieval=_pct(row.get("retrieval_accuracy")),
                hit1=_pct(row.get("hit_at_1")),
                recall5=_pct(row.get("source_hint_recall_at_5")),
                mrr5=_pct(row.get("mrr_at_5")),
                ndcg5=_pct(row.get("ndcg_at_5")),
                answer=_pct(row.get("answer_accuracy")),
                self_eval=_pct(row.get("self_eval_accuracy")),
                retry=_pct(row.get("retry_accuracy")),
                p50=latency.get("p50_ms") if latency.get("p50_ms") is not None else "N/A",
                p95=latency.get("p95_ms") if latency.get("p95_ms") is not None else "N/A",
            )
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_category_lines(baseline: Dict[str, Any]) -> List[str]:
    lines = [
        "## Baseline By Category",
        "",
        "| Category | Total | Pass Rate | Passed | Failed | Error | Skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cat, data in sorted(((baseline.get("summary") or {}).get("by_category") or {}).items()):
        status = data.get("status") or {}
        lines.append(
            f"| {cat} | {data.get('total', 0)} | {_pct(data.get('pass_rate'))} | "
            f"{status.get('passed', 0)} | {status.get('failed', 0)} | {status.get('error', 0)} | {status.get('skipped', 0)} |"
        )
    return lines


def _baseline_dimension_lines(baseline: Dict[str, Any]) -> List[str]:
    lines = [
        "## Baseline By Dimension",
        "",
        "| Dimension | Accuracy | Passed | Failed | Error | Skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dim, data in sorted(((baseline.get("summary") or {}).get("by_dimension") or {}).items()):
        lines.append(
            f"| {dim} | {_pct(data.get('accuracy'))} | {data.get('passed', 0)} | "
            f"{data.get('failed', 0)} | {data.get('error', 0)} | {data.get('skipped', 0)} |"
        )
    return lines


def _failure_lines(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Primary Failures",
        "",
        "| Variant | Failure Counts |",
        "| --- | --- |",
    ]
    for row in rows:
        failures = row.get("primary_failures") or {}
        text = ", ".join(f"{k}: {v}" for k, v in sorted(failures.items())) or "none"
        lines.append(f"| {row['variant']} | {text} |")
    return lines


def _trace_lines(run_dir: Path) -> List[str]:
    path = run_dir / "trace_metrics.json"
    if not path.exists():
        return []
    data = _load(path)
    lines = [
        "## Trace Metrics",
        "",
        "| Variant | Evaluated | Avg Retry | Checkpoint Thread Rate | Planner | Self-Eval Status | Avg Doc Counts |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for report in data.get("reports") or []:
        planner = ", ".join(f"{k}: {v}" for k, v in sorted((report.get("planner") or {}).items()))
        status = ", ".join(f"{k}: {v}" for k, v in sorted((report.get("self_eval_status") or {}).items()))
        counts = ", ".join(f"{k}: {v}" for k, v in sorted((report.get("avg_doc_counts") or {}).items()))
        lines.append(
            f"| {report.get('variant_name')} | {report.get('evaluated')} | "
            f"{report.get('avg_retry_count')} | {_pct(report.get('checkpoint_thread_rate'))} | "
            f"{planner} | {status} | {counts} |"
        )
    return lines


def _write_final_report(run_dir: Path, reports: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    baseline = reports.get("baseline") or {}
    lines = [
        "# ProductDoc-Agent Evaluation Report",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "## Ablation Overview",
        "",
        "| Variant | Pass Rate | Delta | Routing | Retrieval | Hit@1 | Recall@5 | MRR@5 | nDCG@5 | Answer | Self-Eval | Retry | P50 Latency | P95 Latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        latency = row.get("latency") or {}
        lines.append(
            "| {variant} | {pass_rate} | {delta} | {routing} | {retrieval} | {hit1} | {recall5} | {mrr5} | {ndcg5} | {answer} | {self_eval} | {retry} | {p50}ms | {p95}ms |".format(
                variant=row["variant"],
                pass_rate=_pct(row.get("pass_rate")),
                delta=_pp(row.get("delta_vs_baseline")),
                routing=_pct(row.get("routing")),
                retrieval=_pct(row.get("retrieval_accuracy")),
                hit1=_pct(row.get("hit_at_1")),
                recall5=_pct(row.get("source_hint_recall_at_5")),
                mrr5=_pct(row.get("mrr_at_5")),
                ndcg5=_pct(row.get("ndcg_at_5")),
                answer=_pct(row.get("answer_accuracy")),
                self_eval=_pct(row.get("self_eval_accuracy")),
                retry=_pct(row.get("retry_accuracy")),
                p50=latency.get("p50_ms") if latency.get("p50_ms") is not None else "N/A",
                p95=latency.get("p95_ms") if latency.get("p95_ms") is not None else "N/A",
            )
        )
    lines.extend(["", "Note: baseline was rerun with concurrency=1 after the initial concurrent baseline file was missing; other core variants were produced by the full core ablation run with concurrency=2. Accuracy metrics are comparable across variants. Latency is useful as a runtime reference and should be interpreted with the concurrency difference in mind.", ""])
    lines.extend(_baseline_dimension_lines(baseline))
    lines.extend([""])
    lines.extend(_baseline_category_lines(baseline))
    lines.extend([""])
    lines.extend(_failure_lines(rows))
    trace = _trace_lines(run_dir)
    if trace:
        lines.extend([""])
        lines.extend(trace)
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    reports = _load_reports(run_dir)
    if "baseline" not in reports:
        raise SystemExit("baseline.json not found")
    baseline = reports["baseline"]
    rows = [_variant_row(name, reports[name], baseline) for name in reports]
    _write_summary_files(run_dir, reports, rows)
    _write_final_report(run_dir, reports, rows)
    print(run_dir / "final_report.md")


if __name__ == "__main__":
    main()
