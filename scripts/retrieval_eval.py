"""CLI for frozen-corpus retrieval evaluation; no Agent invocation or history writes."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_benchmark, read_jsonl, write_jsonl
from app.evaluation.runner import VARIANTS, run_experiment, write_report
from app.evaluation.workflow import apply_reviews, build_pool, collect_cases, collect_legacy_cases, prepare_legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="migrate weak labels into a pending review draft")
    prepare.add_argument("--legacy", required=True)
    prepare.add_argument("--output", required=True)
    validate = commands.add_parser("validate", help="validate cross-file benchmark invariants")
    validate.add_argument("--dataset", required=True)
    export = commands.add_parser("export", help="read Milvus collection into a frozen corpus snapshot")
    export.add_argument("--dataset", required=True)
    export.add_argument("--collection", help="defaults to project CHUNKS_COLLECTION")
    export.add_argument("--corpus-version", required=True)
    export.add_argument("--index-version", required=True)
    run = commands.add_parser("run", help="evaluate reviewed queries")
    run.add_argument("--dataset", required=True)
    run.add_argument("--variants", default="bm25", help=",".join(VARIANTS))
    run.add_argument("--split", choices=["dev", "test", "regression"], default="dev")
    run.add_argument("--scope", choices=["product", "all"], default="product")
    run.add_argument("--candidate-k", type=int, default=50)
    run.add_argument("--context-k", type=int, default=5)
    run.add_argument("--context-chars", type=int, default=12000)
    run.add_argument("--ks", default="1,3,5,10,20,50")
    run.add_argument("--collection")
    run.add_argument("--annotation-only", action="store_true", help="retrieve pending queries for annotation; no quality scores")
    run.add_argument("--output", required=True)
    pool = commands.add_parser("pool", help="create an unjudged candidate annotation pool")
    pool.add_argument("--dataset", required=True)
    pool.add_argument("--reports", nargs="+", required=True)
    pool.add_argument("--top-k", type=int, default=20)
    pool.add_argument("--output", required=True)
    collect = commands.add_parser("collect", help="collect failures into a pending review queue")
    collect.add_argument("--report", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--legacy-dev", action="store_true", help="explicitly collect legacy development checks as weak observations")
    review = commands.add_parser("review", help="publish human labels into a new benchmark version")
    review.add_argument("--dataset", required=True)
    review.add_argument("--reviews", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_legacy(args.legacy, args.output)
    elif args.command == "collect":
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        result = collect_legacy_cases(report, args.output) if args.legacy_dev else collect_cases(report, args.output)
    elif args.command == "export":
        from app.clients.milvus_utils import get_milvus_client
        from app.conf.milvus_config import milvus_config
        from app.evaluation.retrievers import export_corpus
        root = Path(args.dataset)
        benchmark = load_benchmark(root)
        if benchmark.corpus or any(q["review_status"] == "approved" for q in benchmark.queries.values()):
            raise ValueError("export requires an empty, unapproved draft; copy to a new version before rebuilding")
        client = get_milvus_client()
        if client is None:
            raise RuntimeError("Milvus unavailable")
        collection = args.collection or milvus_config.chunks_collection
        if not collection:
            raise ValueError("CHUNKS_COLLECTION or --collection is required")
        rows = export_corpus(client, collection)
        if not rows:
            raise ValueError("collection is empty")
        write_jsonl(root / "corpus.jsonl", rows)
        manifest = dict(benchmark.manifest, corpus_version=args.corpus_version,
                        index_version=args.index_version, source_collection=collection)
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"chunks": len(rows), "dataset": str(root)}
    else:
        benchmark = load_benchmark(args.dataset)
        if args.command == "validate":
            result = {"queries": len(benchmark.queries), "approved": len(benchmark.select()), "chunks": len(benchmark.corpus),
                      "evidence": len(benchmark.evidence), "qrels": len(benchmark.judgments), "dataset_sha256": benchmark.fingerprint}
        elif args.command == "review":
            result = apply_reviews(benchmark, read_jsonl(args.reviews), args.output, args.version)
        elif args.command == "pool":
            rows = build_pool(benchmark, [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports], args.top_k)
            write_jsonl(args.output, rows)
            result = {"candidates": len(rows), "output": args.output}
        else:
            report = run_experiment(benchmark, [v.strip() for v in args.variants.split(",")], split=args.split,
                                    scope=args.scope, candidate_k=args.candidate_k, context_k=args.context_k,
                                    context_chars=args.context_chars, ks=tuple(int(k) for k in args.ks.split(",")), collection=args.collection,
                                    annotation_only=args.annotation_only)
            write_report(report, args.output)
            compact = {variant: {key: summary[key] for key in ("total", "completed", "errors", "completion_rate", "diagnosis_counts", "latency_ms")}
                       for variant, summary in report["summary"].items()}
            print(json.dumps({"output": args.output, "selection": report["selection"], "summary": compact}, ensure_ascii=False, indent=2))
            return 1 if any(row["status"] == "error" for row in report["results"]) else 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
