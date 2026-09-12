"""
Run eval_comprehensive.py one category at a time and merge completed reports.

This runner keeps comprehensive evaluation recoverable on long or unstable
agentic RAG runs: each category writes its own JSON report, and the merged
report is rebuilt from completed category reports.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.eval_comprehensive import build_comprehensive_summary
from scripts.eval_utils import DEFAULT_DATASET, filter_samples, load_jsonl


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "eval_ablation" / "full_260_benchmark_batches"
DEFAULT_MERGED_OUTPUT = PROJECT_ROOT / "logs" / "eval_ablation" / "full_260_benchmark_merged.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive eval by category.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to validation JSONL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Per-category output directory.")
    parser.add_argument("--merged-output", default=str(DEFAULT_MERGED_OUTPUT), help="Merged JSON output path.")
    parser.add_argument("--categories", default="", help="Comma-separated category filter. Defaults to all active categories.")
    parser.add_argument("--answer-method", default="hybrid", choices=["string", "llm", "hybrid"])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--samples-per-category", type=int, default=0, help="Run only the first N active samples in each category (0 = all).")
    parser.add_argument("--category-timeout-seconds", type=int, default=1800)
    parser.add_argument("--no-judge-artifacts", action="store_true", default=True)
    parser.add_argument("--keep-judge-artifacts", action="store_true", help="Keep judge_artifacts in per-category JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun categories with existing output files.")
    parser.add_argument("--no-run", action="store_true", help="Only merge existing category reports.")
    return parser.parse_args()


def _active_categories(dataset_path: Path, categories_arg: str) -> List[str]:
    if categories_arg.strip():
        return [x.strip() for x in categories_arg.split(",") if x.strip()]

    rows = load_jsonl(dataset_path)
    active = filter_samples(rows, categories=[], ids=[], limit=0, include_not_ready=False)
    categories: List[str] = []
    seen = set()
    for row in active:
        category = str(row.get("category", "")).strip()
        if category and category not in seen:
            seen.add(category)
            categories.append(category)
    return categories


def _safe_name(category: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", category).strip("_") or "category"


def _run_category(args: argparse.Namespace, category: str, output_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_comprehensive.py"),
        "--dataset",
        str(Path(args.dataset).resolve()),
        "--output",
        str(output_path),
        "--categories",
        category,
        "--answer-method",
        args.answer_method,
        "--concurrency",
        str(args.concurrency),
        "--variant-name",
        f"full_260_benchmark_{category}",
    ]
    if args.samples_per_category > 0:
        cmd.extend(["--limit", str(args.samples_per_category)])
    if not args.keep_judge_artifacts:
        cmd.append("--no-judge-artifacts")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    started_at = datetime.now().isoformat()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.category_timeout_seconds,
        )
        return {
            "category": category,
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "output": str(output_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "category": category,
            "status": "timeout",
            "returncode": None,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "timeout_seconds": args.category_timeout_seconds,
            "output": str(output_path),
        }


def _load_completed_reports(paths: List[Path]) -> List[Dict[str, Any]]:
    reports = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def _expected_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(Path(args.dataset).resolve())
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    active = filter_samples(rows, categories=categories, ids=[], limit=0, include_not_ready=False)
    if args.samples_per_category <= 0:
        return active

    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for row in active:
        category = str(row.get("category", ""))
        if counts.get(category, 0) >= args.samples_per_category:
            continue
        selected.append(row)
        counts[category] = counts.get(category, 0) + 1
    return selected


def _merge_reports(args: argparse.Namespace, statuses: List[Dict[str, Any]], report_paths: List[Path]) -> Dict[str, Any]:
    reports = _load_completed_reports(report_paths)
    dataset_path = Path(args.dataset).resolve()
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    results = []
    for report in reports:
        if Path(report.get("dataset", "")).resolve() != dataset_path:
            raise ValueError(f"Report dataset mismatch: {report.get('dataset')}")
        if report.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Report dataset hash mismatch; rerun with --overwrite")
        report_args = report.get("command_args") or {}
        if report_args.get("answer_method") != args.answer_method:
            raise ValueError("Report answer-method mismatch; rerun with --overwrite")
        results.extend(report.get("results") or [])

    expected = _expected_samples(args)
    expected_ids = [str(row.get("id", "")) for row in expected]
    benchmark_expected_total = (
        len(expected_ids)
        if args.categories.strip() or args.samples_per_category > 0
        else 260
    )
    result_ids = [str(row.get("id", "")) for row in results]
    result_id_counts = Counter(result_ids)
    duplicate_ids = sorted(sample_id for sample_id, count in result_id_counts.items() if count > 1)
    missing_ids = sorted(set(expected_ids) - set(result_ids))
    unexpected_ids = sorted(set(result_ids) - set(expected_ids))
    failed_categories = [
        row for row in statuses
        if row.get("status") not in {"completed", "skipped_existing", "not_run"}
    ]
    complete = (
        len(expected_ids) == benchmark_expected_total
        and not duplicate_ids
        and not missing_ids
        and not unexpected_ids
        and not failed_categories
    )

    summary = build_comprehensive_summary(results)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "command_args": vars(args),
        "variant_name": "full_260_benchmark_by_category",
        "category_run_status": statuses,
        "completed_category_reports": [str(path) for path in report_paths if path.exists()],
        "coverage": {
            "complete": complete,
            "benchmark_expected_total": benchmark_expected_total,
            "expected_total": len(expected_ids),
            "result_total": len(result_ids),
            "duplicate_ids": duplicate_ids,
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "failed_categories": failed_categories,
        },
        "summary": summary,
        "results": results,
    }

    merged_output = Path(args.merged_output).resolve()
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    with merged_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = _active_categories(dataset_path, args.categories)
    report_paths = [output_dir / f"{_safe_name(category)}.json" for category in categories]
    statuses: List[Dict[str, Any]] = []

    for category, output_path in zip(categories, report_paths):
        if args.no_run:
            statuses.append({"category": category, "status": "not_run", "output": str(output_path)})
            continue
        if output_path.exists() and not args.overwrite:
            statuses.append({"category": category, "status": "skipped_existing", "output": str(output_path)})
            print(f"[SKIP] {category}: {output_path}")
            continue

        print(f"[RUN] {category} -> {output_path}")
        status = _run_category(args, category, output_path)
        statuses.append(status)
        print(f"[{status['status'].upper()}] {category}")

    payload = _merge_reports(args, statuses, report_paths)
    summary = payload.get("summary", {})
    print(f"[MERGED] {Path(args.merged_output).resolve()}")
    print(f"total={summary.get('total')} pass_rate={summary.get('pass_rate')}")
    print(f"status={summary.get('status')}")
    coverage = payload.get("coverage", {})
    print(f"coverage={coverage}")
    if not coverage.get("complete"):
        raise SystemExit("Evaluation is incomplete; merged report is partial and must not be used as the final benchmark.")


if __name__ == "__main__":
    main()
