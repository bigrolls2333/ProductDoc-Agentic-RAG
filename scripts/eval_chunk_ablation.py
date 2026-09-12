"""
Orchestrate chunking strategy ablations with isolated Milvus collections.

Default mode is dry-run. Use --run-import to rebuild temporary collections.

Examples:
    python scripts/eval_chunk_ablation.py --docs doc/a.md,doc/b.pdf --ids VAL-001,VAL-002
    python scripts/eval_chunk_ablation.py --docs doc/a.md --run-import --run-eval --ids VAL-001
"""
import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "eval_chunk_ablation"


STRATEGIES: Dict[str, Dict[str, str]] = {
    "fixed_300_overlap_50": {"EVAL_CHUNK_STRATEGY": "fixed", "EVAL_CHUNK_MAX_LEN": "300", "EVAL_CHUNK_OVERLAP": "50"},
    "fixed_600_overlap_100": {"EVAL_CHUNK_STRATEGY": "fixed", "EVAL_CHUNK_MAX_LEN": "600", "EVAL_CHUNK_OVERLAP": "100"},
    "fixed_1000_overlap_150": {"EVAL_CHUNK_STRATEGY": "fixed", "EVAL_CHUNK_MAX_LEN": "1000", "EVAL_CHUNK_OVERLAP": "150"},
    "semantic_title_1000": {"EVAL_CHUNK_STRATEGY": "semantic_title", "EVAL_CHUNK_MAX_LEN": "1000"},
    "semantic_title_2000": {"EVAL_CHUNK_STRATEGY": "semantic_title", "EVAL_CHUNK_MAX_LEN": "2000"},
    "semantic_title_no_merge": {
        "EVAL_CHUNK_STRATEGY": "semantic_title",
        "EVAL_CHUNK_MAX_LEN": "2000",
        "EVAL_CHUNK_DISABLE_SHORT_MERGE": "1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunk strategy ablation with isolated collections.")
    parser.add_argument("--docs", default="", help="Comma-separated PDF/MD paths to import for each strategy")
    parser.add_argument("--strategies", default="all", help="Comma-separated strategy names or all")
    parser.add_argument("--output-dir", default="", help="Output directory")
    parser.add_argument("--run-import", action="store_true", help="Actually import docs into temporary collections")
    parser.add_argument("--run-eval", action="store_true", help="Run eval_comprehensive after import")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "eval" / "validation_set.jsonl"))
    parser.add_argument("--ids", default="", help="Eval sample ids")
    parser.add_argument("--categories", default="", help="Eval categories")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--answer-method", default="string", choices=["string", "llm", "hybrid"])
    parser.add_argument("--_run_one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_strategy", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_collection_suffix", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def _selected(raw: str) -> List[str]:
    if raw == "all":
        return list(STRATEGIES.keys())
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies: {unknown}. Valid: {sorted(STRATEGIES)}")
    return names


def _docs(raw: str) -> List[Path]:
    return [Path(x.strip()).resolve() for x in raw.split(",") if x.strip()]


def _python() -> str:
    return sys.executable or "python"


def _collection_names(strategy: str, suffix: str) -> Dict[str, str]:
    safe = "".join(ch if ch.isalnum() else "_" for ch in strategy.lower())
    return {
        "CHUNKS_COLLECTION": f"kb_chunks_eval_{safe}_{suffix}",
        "ITEM_NAME_COLLECTION": f"kb_items_eval_{safe}_{suffix}",
    }


def _run_import_worker(strategy: str, docs: List[Path], output_dir: Path) -> None:
    from app.import_process.api.file_import_service import run_graph_task

    for doc in docs:
        task_id = f"chunk-eval-{strategy}-{uuid.uuid4().hex[:8]}"
        local_dir = output_dir / strategy / doc.stem
        local_dir.mkdir(parents=True, exist_ok=True)
        run_graph_task(
            task_id=task_id,
            local_dir=str(local_dir),
            local_file_path=str(doc),
            doc_id=f"{strategy}:{doc.stem}",
        )


def _run_eval(strategy: str, args: argparse.Namespace, output_dir: Path, env: Dict[str, str]) -> int:
    report_path = output_dir / strategy / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _python(),
        str(PROJECT_ROOT / "scripts" / "eval_comprehensive.py"),
        f"--dataset={args.dataset}",
        f"--output={report_path}",
        "--dimensions=routing,retrieval,self_eval,answer",
        f"--answer-method={args.answer_method}",
        f"--variant-name=chunk_{strategy}",
    ]
    if args.ids:
        cmd.append(f"--ids={args.ids}")
    if args.categories:
        cmd.append(f"--categories={args.categories}")
    if args.limit:
        cmd.append(f"--limit={args.limit}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode


def _run_one(args: argparse.Namespace) -> None:
    strategy = args._strategy
    output_dir = Path(args.output_dir).resolve()
    docs = _docs(args.docs)
    if args.run_import:
        _run_import_worker(strategy, docs, output_dir)
    if args.run_eval:
        code = _run_eval(strategy, args, output_dir, os.environ.copy())
        if code != 0:
            raise SystemExit(code)


def main() -> None:
    args = parse_args()
    if args._run_one:
        _run_one(args)
        return

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / f"run_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = _selected(args.strategies)
    docs = _docs(args.docs)
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "dry_run": not args.run_import and not args.run_eval,
        "docs": [str(p) for p in docs],
        "strategies": {},
    }

    for strategy in strategies:
        env_flags = {**STRATEGIES[strategy], **_collection_names(strategy, suffix)}
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            _python(),
            str(Path(__file__).resolve()),
            "--_run_one",
            f"--_strategy={strategy}",
            f"--output-dir={output_dir}",
            f"--docs={args.docs}",
            f"--dataset={args.dataset}",
            f"--answer-method={args.answer_method}",
        ]
        if args.run_import:
            cmd.append("--run-import")
        if args.run_eval:
            cmd.append("--run-eval")
        if args.ids:
            cmd.append(f"--ids={args.ids}")
        if args.categories:
            cmd.append(f"--categories={args.categories}")
        if args.limit:
            cmd.append(f"--limit={args.limit}")

        manifest["strategies"][strategy] = {
            "env": env_flags,
            "command": cmd,
            "status": "planned",
        }
        if args.run_import or args.run_eval:
            env = os.environ.copy()
            env.update(env_flags)
            print("\n" + "=" * 78)
            print(f"[{strategy}] env={env_flags}")
            print(" ".join(cmd))
            print("=" * 78)
            code = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode
            manifest["strategies"][strategy]["status"] = "completed" if code == 0 else f"exit_code={code}"
            if code != 0:
                break

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunk ablation manifest: {manifest_path}")
    if not args.run_import and not args.run_eval:
        print("Dry-run only. Add --run-import and --run-eval to execute.")


if __name__ == "__main__":
    main()
