"""Score frozen retrieval predictions against a newly reviewed qrel version."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_benchmark
from app.evaluation.diagnosis import diagnose
from app.evaluation.metrics import average_metrics, evidence_metrics, retrieval_metrics
from app.evaluation.retrievers import eligible_ids
from app.evaluation.runner import git_revision, source_hashes, write_report


def replay(benchmark, source_benchmark, source_reports, split="dev"):
    selected = {q["query_id"]: q for q in benchmark.select(split)}
    if not selected:
        raise ValueError("no approved queries to score")
    results, variants, config, provenance = [], [], None, []
    for path in source_reports:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["dataset_sha256"] != source_benchmark.fingerprint:
            raise ValueError("source report does not match its source dataset")
        if report["file_sha256"]["corpus.jsonl"] != benchmark.file_hashes["corpus.jsonl"]:
            raise ValueError("corpus changed; frozen predictions cannot be reused")
        settings = {key: report["config"][key] for key in ("scope", "candidate_k", "context_k", "context_chars", "ks")}
        if config is not None and settings != config:
            raise ValueError("prediction configurations differ across reports")
        config = settings
        provenance.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "dataset_sha256": report["dataset_sha256"], "backends": report["backends"], "code": report["code"]})
        for variant in report["config"]["variants"]:
            if variant in variants:
                raise ValueError("duplicate prediction variant")
            variants.append(variant)
        for prediction in report["results"]:
            qid = prediction["query_id"]
            if qid not in selected:
                continue
            query, old_query = selected[qid], source_benchmark.queries[qid]
            if (query.get("retrieval_query") or query["query"]) != (old_query.get("retrieval_query") or old_query["query"]):
                raise ValueError("retrieval query changed; rerun retrieval rather than replaying")
            if config["scope"] == "product" and (query.get("product_ids"), query.get("canonical_names")) != (old_query.get("product_ids"), old_query.get("canonical_names")):
                raise ValueError("product filter changed; rerun retrieval")
            row = copy.deepcopy(prediction)
            judgments = {cid: judgment for (j_qid, cid), judgment in benchmark.judgments.items() if j_qid == qid}
            if row["status"] == "completed":
                eligible = eligible_ids(benchmark.corpus, query, config["scope"])
                row["stages"]["eligible"] = [{"chunk_id": cid} for cid in eligible if cid in judgments]
                row["answerable"] = query["answerable"]
                row["metrics"] = {"candidates": retrieval_metrics(row["stages"]["candidates"], benchmark.qrels[qid], config["ks"]),
                                  "rerank_order": retrieval_metrics(row["rerank_order"], benchmark.qrels[qid], config["ks"])}
                for stage in ("eligible", "candidates", "reranked", "context"):
                    row["metrics"][stage + "_evidence"] = evidence_metrics(row["stages"][stage], query["evidence_ids"], judgments, query.get("reference_facts", []))
                row["diagnosis"] = diagnose(query, row["stages"], judgments)
            results.append(row)
    keys = [(row["query_id"], row["variant"]) for row in results]
    if len(keys) != len(set(keys)) or len(keys) != len(selected) * len(variants):
        raise ValueError("predictions must cover every selected query/variant exactly once")
    stages = ("candidates", "rerank_order", "eligible_evidence", "candidates_evidence", "reranked_evidence", "context_evidence")
    summary = {}
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant]
        completed = [row for row in rows if row["status"] == "completed"]
        summary[variant] = {"total": len(rows), "completed": len(completed), "errors": len(rows) - len(completed),
                            "completion_rate": len(completed) / len(rows),
                            "metrics": {stage: average_metrics([row["metrics"][stage] for row in completed]) for stage in stages}}
    report = {"schema_version": "productdoc.experiment.v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "benchmark": benchmark.manifest, "dataset_sha256": benchmark.fingerprint, "file_sha256": benchmark.file_hashes,
              "config": dict(config, variants=variants, split=split, annotation_only=False, prediction_replay=True),
              "code": dict(git_revision(), source_sha256=source_hashes()), "prediction_provenance": provenance,
              "selection": {"selected": len(selected), "total": len(benchmark.queries)},
              "metric_policy": "fixed qrels; macro over answerable queries; unjudged zero gain with coverage reported",
              "latency_policy": "historical prediction timings retained; this replay makes zero model/database calls",
              "summary": summary, "results": results}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "regression"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = replay(load_benchmark(args.dataset), load_benchmark(args.source_dataset), args.predictions, args.split)
    write_report(report, args.output)
    print(json.dumps({"output": args.output, "queries": report["selection"]["selected"], "variants": report["config"]["variants"],
                      "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
