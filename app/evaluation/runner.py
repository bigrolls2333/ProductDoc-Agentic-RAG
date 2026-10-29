"""Run fixed-corpus retrieval experiments and retain stage-level artifacts."""

import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from app.evaluation.diagnosis import diagnose
from app.evaluation.metrics import average_metrics, evidence_metrics, retrieval_metrics
from app.evaluation.retrievers import BM25Retriever, MilvusRetriever, eligible_ids


VARIANTS = {"bm25": ("bm25", False), "dense": ("dense", False), "sparse": ("sparse", False),
            "hybrid": ("hybrid", False), "hybrid_rerank": ("hybrid", True)}


def git_revision():
    try:
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        # A dirty checkout must not be presented as an exact committed revision.
        return {"commit": revision, "dirty": bool(status.strip()), "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "app" / "evaluation").glob("*.py"))
    paths += [root / "scripts" / name for name in (
        "retrieval_eval.py", "eval_answer_trust.py", "with_eval_env.py", "replay_retrieval_eval.py",
        "build_source_index.py", "propose_qrels.py", "audit_qrel_sources.py", "prepare_source_decisions.py",
        "publish_source_diagnostic.py", "prepare_answer_records.py", "sweep_context_budget.py",
        "summarize_source_diagnostic.py")]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def select_context(rows, corpus, budget):
    selected, dropped, used = [], [], 0
    for row in rows:
        cost = len(corpus[row["chunk_id"]]["content"])
        if used + cost <= budget:
            selected.append(row)
            used += cost
        else:
            dropped.append(row["chunk_id"])
    return selected, {"content_chars": used, "budget_chars": budget, "dropped_chunk_ids": dropped}


def run_experiment(benchmark, variants, *, split="dev", scope="product", candidate_k=50,
                   context_k=5, context_chars=12000, ks=(1, 3, 5, 10, 20, 50), collection=None, annotation_only=False):
    if not variants or len(set(variants)) != len(variants) or set(variants) - VARIANTS.keys():
        raise ValueError("variants must be unique known names: " + ", ".join(VARIANTS))
    if candidate_k < 1 or context_k < 1 or context_k > candidate_k or context_chars < 1:
        raise ValueError("require candidate_k >= context_k >= 1 and context_chars >= 1")
    if scope not in {"all", "product"}:
        raise ValueError("scope must be all or product")
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    code_snapshot = dict(git_revision(), source_sha256=source_hashes())
    queries = ([q for q in benchmark.queries.values() if q["split"] == split]
               if annotation_only else benchmark.select(split))
    if not queries:
        raise ValueError(f"no approved queries in split={split}; review evidence and qrels first")
    if not benchmark.corpus:
        raise ValueError("benchmark corpus is empty")
    backends = {"bm25": BM25Retriever(benchmark.corpus)} if "bm25" in variants else {}
    if any(VARIANTS[v][0] != "bm25" for v in variants):
        name = collection or benchmark.manifest.get("source_collection")
        if not name:
            raise ValueError("live variants require --collection or manifest.source_collection")
        backends["milvus"] = MilvusRetriever(benchmark.corpus, name)
    reranker = None
    rerank_metadata = None
    if any(VARIANTS[v][1] for v in variants):
        from app.lm.reranker_utils import get_reranker_model
        from app.conf.reranker_config import reranker_config
        reranker = get_reranker_model()
        rerank_metadata = {"model": reranker_config.bge_reranker_large,
                           "device": reranker_config.bge_reranker_device,
                           "fp16": reranker_config.bge_reranker_fp16, "max_length": 512}
    results = []
    # Search results are shared across paired hybrid / hybrid_rerank experiments.
    cache = {}
    for query in queries:
        qid = query["query_id"]
        judgments = {cid: row for (row_qid, cid), row in benchmark.judgments.items() if row_qid == qid}
        for variant in variants:
            start = time.perf_counter()
            result = {"query_id": qid, "variant": variant, "category": query.get("category", "unknown"),
                      "answerable": query["answerable"], "status": "completed", "stages": {}, "metrics": {}}
            try:
                eligible = eligible_ids(benchmark.corpus, query, scope)
                # Full eligibility follows from the frozen corpus and filter.
                # Retain its judged projection instead of copying the entire
                # corpus ID list into every variant/query artifact.
                result["filter"] = {"scope": scope, "product_ids": query.get("product_ids", []),
                                    "canonical_names": query.get("canonical_names", []),
                                    "eligible_count": len(eligible), "stage_projection": "judged_chunks_only"}
                result["stages"]["eligible"] = [{"chunk_id": cid} for cid in eligible if cid in judgments]
                mode, do_rerank = VARIANTS[variant]
                key = (qid, mode)
                backend = backends["bm25" if mode == "bm25" else "milvus"]
                reused = key in cache
                if not reused:
                    search_start = time.perf_counter()
                    candidates = backend.search(query.get("retrieval_query") or query["query"], eligible, mode, candidate_k)
                    if any(row["chunk_id"] not in benchmark.corpus or row["chunk_id"] not in eligible for row in candidates):
                        raise ValueError("backend returned a chunk outside the frozen eligible corpus")
                    if any(not math.isfinite(row["score"]) for row in candidates):
                        raise ValueError("non-finite retrieval score")
                    cache[key] = (candidates, (time.perf_counter() - search_start) * 1000)
                candidates, search_ms = cache[key]
                result["stages"]["candidates"] = candidates
                rank_start = time.perf_counter()
                reranked = list(candidates)
                if do_rerank and candidates:
                    scores = reranker.compute_score([[query.get("retrieval_query") or query["query"], benchmark.corpus[row["chunk_id"]]["content"]] for row in candidates])
                    if len(scores) != len(candidates) or not all(math.isfinite(float(score)) for score in scores):
                        raise ValueError("reranker returned missing/non-finite scores")
                    reranked = sorted([dict(row, retrieval_score=row["score"], score=float(score)) for row, score in zip(candidates, scores)],
                                      key=lambda row: -row["score"])
                rank_ms = (time.perf_counter() - rank_start) * 1000
                # Keep the complete ordering, and expose the context selection separately.
                result["stages"]["reranked"] = reranked[:context_k]
                context, budget = select_context(reranked[:context_k], benchmark.corpus, context_chars)
                result["stages"]["context"] = context
                result["rerank_order"] = reranked
                result["context_budget"] = budget
                result["timings_ms"] = {"search": search_ms, "rerank": rank_ms,
                                         "pipeline": search_ms + rank_ms, "search_reused": reused,
                                         "wall": (time.perf_counter() - start) * 1000}
                if annotation_only:
                    result["diagnosis"] = [{"type": "annotation_pending", "status": "review_required"}]
                    results.append(result)
                    continue
                result["metrics"] = {"candidates": retrieval_metrics(candidates, benchmark.qrels[qid], ks),
                                      "rerank_order": retrieval_metrics(reranked, benchmark.qrels[qid], ks)}
                for stage in ("eligible", "candidates", "reranked", "context"):
                    result["metrics"][stage + "_evidence"] = evidence_metrics(result["stages"][stage], query.get("evidence_ids", []), judgments, query.get("reference_facts", []))
                result["diagnosis"] = diagnose(query, result["stages"], judgments)
            except Exception as exc:
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
                result["diagnosis"] = diagnose(query, result["stages"], judgments, error=result["error"])
            results.append(result)
    summaries = {}
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant]
        completed = [row for row in rows if row["status"] == "completed"]
        latencies = sorted(row["timings_ms"]["pipeline"] for row in completed)
        summaries[variant] = {"total": len(rows), "completed": len(completed), "errors": len(rows) - len(completed),
                              "completion_rate": len(completed) / len(rows),
                              "metrics": {stage: average_metrics([row["metrics"].get(stage, {}) for row in completed])
                                          for stage in ("candidates", "rerank_order", "eligible_evidence", "candidates_evidence", "reranked_evidence", "context_evidence")},
                              "diagnosis_counts": dict(Counter(d["type"] for row in rows for d in row["diagnosis"])),
                              "latency_ms": {"p50": median(latencies) if latencies else None,
                                             "p95": latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None}}
        summaries[variant]["by_category"] = {}
        for category in sorted({row["category"] for row in rows}):
            group = [row for row in rows if row["category"] == category]
            good = [row for row in group if row["status"] == "completed"]
            summaries[variant]["by_category"][category] = {
                "total": len(group), "completed": len(good),
                "context_evidence": average_metrics([row["metrics"].get("context_evidence", {}) for row in good])}
    comparisons = {}
    if not annotation_only:
        baseline = {row["query_id"]: row for row in results if row["variant"] == variants[0]}
        for variant in variants[1:]:
            pairs = [(baseline[row["query_id"]], row) for row in results if row["variant"] == variant
                     and row["status"] == "completed" and baseline[row["query_id"]]["status"] == "completed"]
            deltas = []
            for before, after in pairs:
                left = before["metrics"]["context_evidence"]["evidence_recall"]
                right = after["metrics"]["context_evidence"]["evidence_recall"]
                if left is not None and right is not None:
                    deltas.append({"delta_context_evidence_recall": right - left})
            comparisons[variant] = {"baseline": variants[0], "paired_completed": len(pairs),
                                    "excluded_error_pairs": len(queries) - len(pairs), "metrics": average_metrics(deltas)}
    return {"schema_version": "productdoc.experiment.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": benchmark.manifest, "dataset_sha256": benchmark.fingerprint, "file_sha256": benchmark.file_hashes,
            "code": dict(code_snapshot, source_changed_during_run=source_hashes() != code_snapshot["source_sha256"]),
            "python": platform.python_version(),
            "config": {"variants": variants, "split": split, "scope": scope, "candidate_k": candidate_k, "annotation_only": annotation_only,
                       "context_k": context_k, "context_chars": context_chars, "ks": list(ks), "reranker": rerank_metadata},
            "backends": {key: backend.metadata for key, backend in backends.items()},
            "selection": {"selected": len(queries), "pending": sum(q["review_status"] != "approved" for q in benchmark.queries.values()),
                          "total": len(benchmark.queries)},
            "metric_policy": "macro over completed answerable queries; unjudged treated as zero gain, judged_fraction reported; no-answer IR scores null; errors separate",
            "latency_policy": "search shared across paired variants; cached timing reused; model/backend initialization excluded; not serving latency",
            "paired_comparisons": comparisons, "summary": summaries, "results": results}


def write_report(report, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Retrieval experiment", "", f"Dataset SHA-256: `{report['dataset_sha256']}`", "",
             "Unjudged documents contribute zero gain; inspect judgment coverage before comparing variants.", "",
             "| Variant | Completed/Total | Recall@K candidates | nDCG@K ranking | Context span recall | Context fact recall | All facts in context |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    k = min(5, max(report["config"]["ks"]))
    if k not in report["config"]["ks"]:
        k = report["config"]["ks"][0]
    lines[6] = lines[6].replace("@K", f"@{k}")
    def metric(summary, stage, name):
        item = summary["metrics"][stage].get(name, {})
        return "N/A" if item.get("mean") is None else f"{item['mean']:.4f} (n={item['n']})"
    for variant, summary in report["summary"].items():
        lines.append(f"| {variant} | {summary['completed']}/{summary['total']} | {metric(summary, 'candidates', f'recall@{k}')} | {metric(summary, 'rerank_order', f'ndcg@{k}')} | {metric(summary, 'context_evidence', 'evidence_recall')} | {metric(summary, 'context_evidence', 'fact_recall')} | {metric(summary, 'context_evidence', 'all_facts')} |")
    if report["benchmark"].get("annotation_quality"):
        lines += ["", "Annotation provenance: " + json.dumps(report["benchmark"]["annotation_quality"], ensure_ascii=False),
                  "", "AI-reviewed labels require independent human review before being described as human gold."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
