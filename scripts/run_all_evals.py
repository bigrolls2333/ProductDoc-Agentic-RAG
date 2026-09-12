"""
run_all_evals.py — One-command orchestrator for all evaluation scripts.

Usage:
    python scripts/run_all_evals.py                        # Full evaluation
    python scripts/run_all_evals.py --limit 10             # Smoke test (10 samples)
    python scripts/run_all_evals.py --routing-only         # Only routing checks
    python scripts/run_all_evals.py --no-web               # Offline mode (skip MCP)
    python scripts/run_all_evals.py --output ./eval_report # Custom output dir
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all evaluation scripts and merge reports.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N samples (0 = all)")
    parser.add_argument("--routing-only", action="store_true", help="Only run routing evaluation (skip full graph)")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--ids", default="", help="Comma-separated sample ID filter")
    parser.add_argument("--no-web", action="store_true", help="Skip MCP web search during eval")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first eval script failure")
    parser.add_argument("--keep-history", action="store_true", help="Keep seeded eval history in Mongo for debugging")
    parser.add_argument("--include-not-ready", action="store_true", help="Include not-ready samples")
    parser.add_argument(
        "--output", default="", help="Output directory for eval reports (default: logs/eval_all_<timestamp>)"
    )
    return parser.parse_args()


def _script_path(name: str) -> Path:
    return PROJECT_ROOT / "scripts" / name


def _python() -> str:
    return sys.executable or "python"


def _run_script(script_name: str, extra_args: List[str]) -> Dict[str, Any]:
    """Run a Python script and return the parsed JSON report."""
    script = _script_path(script_name)
    if not script.exists():
        print(f"[WARN] Script not found: {script}")
        return {}

    cmd = [_python(), str(script)] + extra_args
    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {script_name} exited with code {result.returncode}")
        return {"_error": f"exit_code={result.returncode}"}
    return {}


def _find_latest_report(subdir: str) -> Optional[Path]:
    """Find the most recent JSON report in logs/<subdir>/."""
    report_dir = PROJECT_ROOT / "logs" / subdir
    if not report_dir.exists():
        return None
    candidates = sorted(report_dir.glob("report_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _merge_summaries(summaries: List[Dict[str, Any]], routing_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple eval summaries into one comprehensive report."""
    merged = {
        "total": 0,
        "evaluated": 0,
        "status": {},
        "pass_rate": None,
        "by_dimension": {},
        "by_category": {},
        "primary_failures": {},
    }

    for s in summaries:
        if not s:
            continue
        for key in ["total", "evaluated"]:
            merged[key] += s.get(key, 0)

        for st, count in s.get("status", {}).items():
            merged["status"][st] = merged["status"].get(st, 0) + count

        for dim_name, dim_data in s.get("by_dimension", {}).items():
            if dim_name not in merged["by_dimension"]:
                merged["by_dimension"][dim_name] = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
            for k in ["passed", "failed", "skipped", "error"]:
                merged["by_dimension"][dim_name][k] += dim_data.get(k, 0)

        for cat, cat_data in s.get("by_category", {}).items():
            if cat not in merged["by_category"]:
                merged["by_category"][cat] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0}
            merged["by_category"][cat]["total"] += cat_data.get("total", 0)
            for st in ["passed", "failed", "skipped", "error"]:
                merged["by_category"][cat][st] += cat_data.get("status", {}).get(st, 0)

        for pf, count in s.get("primary_failures", {}).items():
            merged["primary_failures"][pf] = merged["primary_failures"].get(pf, 0) + count

    # If routing summary also has routing-specific checks, add them as dimensions
    if routing_summary:
        for check_name, check_data in routing_summary.get("checks", {}).items():
            if check_name not in merged["by_dimension"]:
                merged["by_dimension"][check_name] = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
            for k in ["passed", "failed", "skipped", "error"]:
                merged["by_dimension"][check_name][k] += check_data.get(k, 0)

    # Recalculate pass rates
    for dim_name, dd in merged["by_dimension"].items():
        p, f = dd.get("passed", 0), dd.get("failed", 0)
        denom = p + f
        dd["accuracy"] = round(p / denom, 4) if denom else None

    for cat, cd in merged["by_category"].items():
        p, f = cd.get("passed", 0), cd.get("failed", 0)
        denom = p + f
        cd["pass_rate"] = round(p / denom, 4) if denom else None

    passed = merged["status"].get("passed", 0)
    failed = merged["status"].get("failed", 0)
    denom = passed + failed
    merged["pass_rate"] = round(passed / denom, 4) if denom else None
    merged["evaluated"] = passed + failed + merged["status"].get("error", 0)

    return merged


def main() -> None:
    args = parse_args()

    # Build output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / f"eval_all_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common args for sub-scripts.
    # Do NOT pass --output; each script writes to its default location
    # (logs/eval_routing/ and logs/eval_comprehensive/) and we discover
    # the latest report via _find_latest_report().
    common_args = [
        f"--limit={args.limit}",
    ]
    if args.categories:
        common_args.append(f"--categories={args.categories}")
    if args.ids:
        common_args.append(f"--ids={args.ids}")
    if args.include_not_ready:
        common_args.append("--include-not-ready")
    if args.fail_fast:
        common_args.append("--fail-fast")

    all_summaries: List[Dict[str, Any]] = []
    routing_summary: Optional[Dict[str, Any]] = None

    # Step 1: Routing eval (step-by-step, cheap).
    routing_args = list(common_args)
    if args.keep_history:
        routing_args.append("--keep-history")
    _run_script("eval_routing.py", routing_args)
    routing_report = _find_latest_report("eval_routing")
    if routing_report:
        try:
            data = json.loads(routing_report.read_text(encoding="utf-8"))
            routing_summary = data.get("summary", {})
            all_summaries.append(routing_summary)
        except Exception as e:
            print(f"[WARN] Failed to load routing report: {e}")

    # Step 2: Comprehensive eval (full graph, expensive)
    if not args.routing_only:
        comp_args = list(common_args)
        if args.no_web:
            comp_args.append("--no-web")
        if args.keep_history:
            comp_args.append("--keep-history")
        _run_script("eval_comprehensive.py", comp_args)
        comp_report = _find_latest_report("eval_comprehensive")
        if comp_report:
            try:
                data = json.loads(comp_report.read_text(encoding="utf-8"))
                all_summaries.append(data.get("summary", {}))
            except Exception as e:
                print(f"[WARN] Failed to load comprehensive report: {e}")

    # Merge and save unified report
    merged_summary = _merge_summaries(all_summaries, routing_summary)

    unified = {
        "generated_at": datetime.now().isoformat(),
        "command_args": vars(args),
        "output_dir": str(output_dir),
        "summary": merged_summary,
    }

    unified_path = output_dir / "unified_report.json"
    with unified_path.open("w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)

    # Print unified summary
    print(f"\n{'#' * 72}")
    print(f"# Unified Eval Report")
    print(f"{'#' * 72}")
    print(f"total:       {merged_summary.get('total', 0)}")
    print(f"evaluated:   {merged_summary.get('evaluated', 0)}")
    print(f"status:      {merged_summary.get('status', {})}")
    print(f"pass_rate:   {merged_summary.get('pass_rate')}")
    print(f"\n--- Per-Dimension ---")
    for dim, dd in sorted(merged_summary.get("by_dimension", {}).items()):
        acc = f"{dd['accuracy']:.2%}" if dd.get("accuracy") is not None else "N/A"
        print(f"  {dim:<25} {acc:<10} passed={dd.get('passed',0)} failed={dd.get('failed',0)}")
    print(f"\n--- Per-Category ---")
    for cat, cd in sorted(merged_summary.get("by_category", {}).items()):
        pr = f"{cd['pass_rate']:.2%}" if cd.get("pass_rate") is not None else "N/A"
        print(f"  {cat:<25} {pr:<10} total={cd.get('total',0)}")
    print(f"\nOutput: {unified_path}")


if __name__ == "__main__":
    main()
