"""
Run Agentic RAG ablation variants on the same validation set.

Examples:
    python scripts/eval_ablation.py --limit 20
    python scripts/eval_ablation.py --categories multi_turn,web_required,no_answer
    python scripts/eval_ablation.py --variants baseline,no_rerank,no_hyde,no_context_memory
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "validation_set.jsonl"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "eval_ablation"


VARIANTS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    "no_rerank": {"EVAL_DISABLE_RERANK": "1"},
    "upstream_top1": {"EVAL_DISABLE_RERANK": "1", "EVAL_RERANK_DISABLE_GAP": "1", "EVAL_RERANK_MAX_TOPK": "1"},
    "upstream_top3": {"EVAL_DISABLE_RERANK": "1", "EVAL_RERANK_DISABLE_GAP": "1", "EVAL_RERANK_MAX_TOPK": "3"},
    "upstream_top5": {"EVAL_DISABLE_RERANK": "1", "EVAL_RERANK_DISABLE_GAP": "1", "EVAL_RERANK_MAX_TOPK": "5"},
    "upstream_top8": {"EVAL_DISABLE_RERANK": "1", "EVAL_RERANK_DISABLE_GAP": "1", "EVAL_RERANK_MAX_TOPK": "8"},
    "rerank_min3": {"EVAL_RERANK_MIN_TOPK": "3"},
    "rerank_min5": {"EVAL_RERANK_MIN_TOPK": "5"},
    "rerank_no_gap_top5": {"EVAL_RERANK_DISABLE_GAP": "1", "EVAL_RERANK_MAX_TOPK": "5"},
    "rerank_no_gap_top8": {"EVAL_RERANK_DISABLE_GAP": "1", "EVAL_RERANK_MAX_TOPK": "8"},
    "rerank_soft_gap": {
        "EVAL_RERANK_MIN_TOPK": "3",
        "EVAL_RERANK_GAP_ABS": "1.0",
        "EVAL_RERANK_GAP_RATIO": "0.45",
    },
    "no_hyde": {"EVAL_DISABLE_HYDE": "1"},
    "no_rrf": {"EVAL_DISABLE_RRF": "1"},
    "no_self_eval": {"EVAL_DISABLE_SELF_EVAL": "1"},
    "no_retry": {"EVAL_DISABLE_SELF_EVAL_RETRY": "1"},
    "no_model_guard": {"EVAL_DISABLE_MODEL_GUARD": "1"},
    "no_web": {"EVAL_DISABLE_WEB": "1"},
    "rule_planner": {"EVAL_FORCE_PLANNER": "rule"},
    "embedding_only": {"EVAL_FORCE_PLANNER": "embedding_only", "EVAL_DISABLE_HYDE": "1", "EVAL_DISABLE_WEB": "1"},
    "history_only": {"EVAL_DISABLE_SESSION_MEMORY": "1", "EVAL_DISABLE_PRODUCT_MEMORY": "1"},
    "no_context_memory": {
        "EVAL_DISABLE_CONTEXT_BUNDLE": "1",
        "EVAL_DISABLE_SESSION_MEMORY": "1",
        "EVAL_DISABLE_PRODUCT_MEMORY": "1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive eval across ablation variants.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to validation JSONL")
    parser.add_argument("--output-dir", default="", help="Output dir. Defaults to logs/eval_ablation/run_<timestamp>")
    parser.add_argument("--variants", default="core", help="Comma-separated variants, or core/all")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N samples")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--ids", default="", help="Comma-separated sample ID filter")
    parser.add_argument("--dimensions", default="routing,retrieval,self_eval,answer,retry", help="Comprehensive eval dimensions")
    parser.add_argument("--answer-method", default="string", choices=["string", "llm", "hybrid"], help="Answer judge method")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel graph invocations per variant")
    parser.add_argument("--include-not-ready", action="store_true", help="Include samples marked not-ready")
    parser.add_argument("--fail-fast", action="store_true", help="Stop when one variant exits non-zero")
    parser.add_argument("--keep-judge-artifacts", action="store_true", help="Keep judge_artifacts in each variant JSON")
    parser.add_argument("--artifact-text-chars", type=int, default=4000, help="Forwarded to eval_comprehensive when artifacts are kept")
    return parser.parse_args()


def _selected_variants(raw: str) -> List[str]:
    if raw == "core":
        return ["baseline", "no_rerank", "no_hyde", "no_self_eval", "no_retry", "no_model_guard", "history_only", "no_context_memory"]
    if raw == "all":
        return list(VARIANTS.keys())
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Valid: {sorted(VARIANTS)}")
    return names


def _python() -> str:
    return sys.executable or "python"


def _run_variant(name: str, env_flags: Dict[str, str], args: argparse.Namespace, output_path: Path) -> int:
    cmd = [
        _python(),
        str(PROJECT_ROOT / "scripts" / "eval_comprehensive.py"),
        f"--dataset={args.dataset}",
        f"--output={output_path}",
        f"--dimensions={args.dimensions}",
        f"--answer-method={args.answer_method}",
        f"--concurrency={args.concurrency}",
        f"--variant-name={name}",
    ]
    if args.limit:
        cmd.append(f"--limit={args.limit}")
    if args.categories:
        cmd.append(f"--categories={args.categories}")
    if args.ids:
        cmd.append(f"--ids={args.ids}")
    if args.include_not_ready:
        cmd.append("--include-not-ready")
    if env_flags.get("EVAL_DISABLE_WEB") == "1":
        cmd.append("--no-web")
    if args.keep_judge_artifacts:
        cmd.append(f"--artifact-text-chars={args.artifact_text_chars}")
    else:
        cmd.append("--no-judge-artifacts")

    env = os.environ.copy()
    env.update(env_flags)
    print("\n" + "=" * 78)
    print(f"[{name}] env={env_flags or '{}'}")
    print(" ".join(cmd))
    print("=" * 78)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def _load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"_error": "missing_report", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _latency_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = sorted(
        int((r.get("timings_ms") or {}).get("graph_total") or 0)
        for r in results
        if r.get("status") in {"passed", "failed"} and (r.get("timings_ms") or {}).get("graph_total") is not None
    )
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None}
    p95_idx = min(len(values) - 1, int(round(len(values) * 0.95)) - 1)
    return {
        "avg_ms": round(sum(values) / len(values), 2),
        "p50_ms": int(median(values)),
        "p95_ms": values[p95_idx],
    }


def _variant_row(name: str, report: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary") or {}
    dims = summary.get("by_dimension") or {}
    bench = ((summary.get("retrieval_benchmark") or {}).get("averages") or {})
    base_summary = baseline.get("summary") or {}
    base_pass = base_summary.get("pass_rate")
    pass_rate = summary.get("pass_rate")
    delta = None
    if pass_rate is not None and base_pass is not None:
        delta = round(float(pass_rate) - float(base_pass), 4)
    return {
        "variant": name,
        "pass_rate": pass_rate,
        "delta_vs_baseline": delta,
        "status": summary.get("status", {}),
        "retrieval_accuracy": (dims.get("retrieval_quality") or {}).get("accuracy"),
        "answer_accuracy": (dims.get("answer_quality") or {}).get("accuracy"),
        "self_eval_accuracy": (dims.get("self_eval_accuracy") or {}).get("accuracy"),
        "retry_accuracy": (dims.get("retry_effectiveness") or {}).get("accuracy"),
        "hit_at_1": bench.get("hit@1"),
        "source_hint_recall_at_5": bench.get("source_hint_recall@5"),
        "mrr_at_5": bench.get("mrr@5"),
        "ndcg_at_5": bench.get("ndcg@5"),
        "primary_failures": summary.get("primary_failures", {}),
        "latency": _latency_stats(report.get("results") or []),
    }


def _write_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Agentic RAG Ablation Summary",
        "",
        "| Variant | Pass Rate | Delta | Retrieval | Hit@1 | Recall@5 | MRR@5 | nDCG@5 | Answer | Self-Eval | Retry | P50 Latency | P95 Latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row.get("delta_vs_baseline")
        delta_text = "N/A" if delta is None else f"{delta * 100:+.2f}pp"
        latency = row.get("latency") or {}
        lines.append(
            "| {variant} | {pass_rate} | {delta} | {retrieval} | {hit1} | {recall5} | {mrr5} | {ndcg5} | {answer} | {self_eval} | {retry} | {p50}ms | {p95}ms |".format(
                variant=row["variant"],
                pass_rate=_percent(row.get("pass_rate")),
                delta=delta_text,
                retrieval=_percent(row.get("retrieval_accuracy")),
                hit1=_percent(row.get("hit_at_1")),
                recall5=_percent(row.get("source_hint_recall_at_5")),
                mrr5=_percent(row.get("mrr_at_5")),
                ndcg5=_percent(row.get("ndcg_at_5")),
                answer=_percent(row.get("answer_accuracy")),
                self_eval=_percent(row.get("self_eval_accuracy")),
                retry=_percent(row.get("retry_accuracy")),
                p50=latency.get("p50_ms") if latency.get("p50_ms") is not None else "N/A",
                p95=latency.get("p95_ms") if latency.get("p95_ms") is not None else "N/A",
            )
        )
    lines.extend([
        "",
        "## Resume Metrics",
        "",
        "- Rerank delta: compare `baseline` with `no_rerank` on retrieval and answer accuracy.",
        "- HyDE delta: compare `baseline` with `no_hyde` on troubleshooting and multi-turn categories.",
        "- Context/Memory delta: compare `baseline` with `history_only` and `no_context_memory` on multi-turn samples.",
        "- Self-Eval/Retry/Guard delta: compare `baseline` with `no_self_eval`, `no_retry`, and `no_model_guard` on no-answer and web-required samples.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    variant_names = _selected_variants(args.variants)
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = DEFAULT_LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Dict[str, Any]] = {}
    for name in variant_names:
        report_path = output_dir / f"{name}.json"
        code = _run_variant(name, VARIANTS[name], args, report_path)
        reports[name] = _load_report(report_path)
        if code != 0:
            reports[name]["_error"] = f"exit_code={code}"
            if args.fail_fast:
                break

    baseline = reports.get("baseline") or next(iter(reports.values()), {})
    rows = [_variant_row(name, reports[name], baseline) for name in reports]
    summary = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(Path(args.dataset).resolve()),
        "command_args": vars(args),
        "variants": variant_names,
        "rows": rows,
        "reports": {name: str(output_dir / f"{name}.json") for name in reports},
    }
    summary_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(rows, md_path)

    print("\n" + "#" * 78)
    print("Ablation Summary")
    print("#" * 78)
    for row in rows:
        delta = row.get("delta_vs_baseline")
        delta_text = "N/A" if delta is None else f"{delta * 100:+.2f}pp"
        print(f"{row['variant']:<20} pass={_percent(row.get('pass_rate')):<8} delta={delta_text}")
    print(f"\nJSON: {summary_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
