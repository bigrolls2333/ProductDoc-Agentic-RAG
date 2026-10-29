"""Draft migration, annotation pools, and reviewed case collection."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from app.evaluation.dataset import FILES, load_benchmark, read_jsonl, write_jsonl


def prepare_legacy(source, output):
    """Create a draft directory; never infer approved gold labels."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"refusing to overwrite dataset directory: {output}")
    rows = read_jsonl(source)
    queries = []
    for row in rows:
        expected = row.get("expected", {})
        queries.append({"query_id": row["id"], "query": row["query"], "category": row.get("category", "unknown"),
                        "history": row.get("history", []), "split": "dev", "group_id": "",
                        "review_status": "pending", "answerable": None, "product_ids": [],
                        "canonical_names": expected.get("item_names", []), "evidence_ids": [], "reference_facts": [],
                        "legacy_hints": {"source_hint": expected.get("source_hint", {}),
                                         "answer_must_include": expected.get("answer_must_include", []),
                                         "ingestion_status": row.get("ingestion_status", "unknown"),
                                         "notes": row.get("notes", "")}})
    output.mkdir(parents=True)
    manifest = {"schema_version": "productdoc.benchmark.v1", "name": "productdoc", "version": "0.1-draft",
                "corpus_version": "pending-export", "index_version": "pending-export",
                "annotation_status": "draft", "source_dataset_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "relevance_scale": {"0": "irrelevant", "1": "helpful", "2": "direct evidence"}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output / "queries.jsonl", queries)
    for name in ("corpus", "evidence", "qrels"):
        write_jsonl(output / f"{name}.jsonl", [])
    return {"queries": len(queries), "approved": 0, "output": str(output)}


def build_pool(benchmark, reports, top_k=20):
    """Union candidates across experiments without promoting them to qrels."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    pooled = {}
    for report in reports:
        if report["dataset_sha256"] != benchmark.fingerprint:
            raise ValueError("pool report dataset does not match current benchmark")
        for row in report["results"]:
            if row["status"] != "completed":
                continue
            orders = {"candidates": row["stages"]["candidates"], "rerank_order": row.get("rerank_order", [])}
            for stage, docs in orders.items():
                for rank, doc in enumerate(docs[:top_k], 1):
                    key = row["query_id"], doc["chunk_id"]
                    entry = pooled.setdefault(key, {"query_id": key[0], "chunk_id": key[1],
                                                    "query": benchmark.queries[key[0]]["query"],
                                                    "doc_id": benchmark.corpus[key[1]]["doc_id"],
                                                    "content": benchmark.corpus[key[1]]["content"],
                                                    "relevance": None, "evidence_ids": [], "review_status": "pending", "retrieved_by": []})
                    origin = {"variant": row["variant"], "stage": stage, "rank": rank}
                    if origin not in entry["retrieved_by"]:
                        entry["retrieved_by"].append(origin)
    return list(pooled.values())


def collect_cases(report, output):
    """Idempotently collect observed failures for human review.

    Test queries are deliberately excluded from the development feedback loop.
    A case record is never silently turned into an approved benchmark label.
    """
    config = report.get("config", {})
    if config.get("split") == "test" or any(row.get("split") == "test" for row in report.get("results", [])):
        raise ValueError("test split cannot flow into the development case queue")
    if config.get("annotation_only"):
        raise ValueError("annotation-only retrieval is not a scored failure report")
    output = Path(output)
    existing = read_jsonl(output) if output.exists() else []
    seen = {row["case_id"] for row in existing}
    added = 0
    for result in report["results"]:
        issues = [item for item in result["diagnosis"] if item["type"] not in {"no_observed_evidence_loss", "unanswerable_case"}]
        if not issues:
            continue
        # Ignore timing differences when rerunning the identical observation.
        stable_result = {key: value for key, value in result.items() if key != "timings_ms"}
        identity = json.dumps([report["dataset_sha256"], config, stable_result], sort_keys=True)
        case_id = "case_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        if case_id in seen:
            continue
        existing.append({"case_id": case_id, "query_id": result["query_id"], "variant": result.get("variant", "answer"),
                         "dataset_sha256": report["dataset_sha256"], "review_status": "pending", "diagnosis": issues,
                         "observation": stable_result, "review": None})
        seen.add(case_id)
        added += 1
    write_jsonl(output, existing)
    return {"added": added, "total": len(existing)}


def collect_legacy_cases(report, output):
    """Keep legacy check failures as weak observations, never qrel diagnoses."""
    results = []
    for row in report["results"]:
        if row.get("status") == "error":
            issues = [{"type": "execution_error", "status": "observed", "reason": row.get("error", "")}]
        else:
            issues = [{"type": "legacy_check_failure", "dimension": name, "status": "weak_supervision_observation",
                       "reason": "Legacy check failed; this does not identify a root cause."}
                      for name, check in row.get("dimension_results", {}).items()
                      if not check.get("skipped") and check.get("passed") is False]
        results.append({"query_id": row["id"], "variant": report.get("variant_name", "legacy"),
                        "status": row["status"], "diagnosis": issues, "legacy_observation": row})
    converted = {"dataset_sha256": report["dataset_sha256"], "config": {"split": "dev", "legacy_weak_supervision": True},
                 "results": results}
    return collect_cases(converted, output)


def apply_reviews(benchmark, reviews, output, version):
    """Publish source-attributed review decisions into a new dataset version."""
    output = Path(output)
    if output.exists():
        raise ValueError("review output must be a new directory")
    if not reviews or not version.strip() or version == benchmark.manifest["version"]:
        raise ValueError("reviews and a new nonempty version are required")
    queries = dict(benchmark.queries)
    evidence = dict(benchmark.evidence)
    judgments = dict(benchmark.judgments)
    seen = set()
    for review in reviews:
        qid = review.get("query_id")
        if qid not in queries or qid in seen:
            raise ValueError("reviews must reference unique existing query IDs")
        seen.add(qid)
        if queries[qid]["split"] == "test" and queries[qid]["review_status"] == "approved":
            raise ValueError("frozen approved test queries cannot be edited through review feedback")
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError("review requires reviewer and reviewed_at provenance")
        updates = review.get("query", {})
        allowed = {"answerable", "group_id", "split", "product_ids", "canonical_names", "evidence_ids", "reference_facts", "retrieval_query"}
        if set(updates) - allowed:
            raise ValueError("unsupported reviewed query fields")
        queries[qid] = dict(queries[qid], **updates, review_status="approved",
                            review={"reviewer": review["reviewer"], "reviewed_at": review["reviewed_at"],
                                    "reviewer_type": review.get("reviewer_type", "unspecified"),
                                    "human_verified": review.get("human_verified", False),
                                    "source_case_id": review.get("source_case_id"),
                                    "notes": review.get("notes", ""),
                                    "review_scope": review.get("review_scope", "unspecified")})
        for item in review.get("evidence", []):
            eid = item["evidence_id"]
            if eid in evidence and evidence[eid] != item:
                raise ValueError("existing evidence anchors are immutable; use a new evidence_id")
            evidence[eid] = item
        judgments = {key: item for key, item in judgments.items() if key[0] != qid}
        for item in review.get("qrels", []):
            key = qid, item["chunk_id"]
            if key in judgments:
                raise ValueError("duplicate reviewed qrel")
            if item.get("query_id", qid) != qid:
                raise ValueError("review qrel query_id mismatch")
            judgments[key] = dict(item, query_id=qid)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        root = Path(directory) / "validated"
        root.mkdir()
        for name in FILES:
            shutil.copyfile(benchmark.root / name, root / name)
        approved = [q for q in queries.values() if q["review_status"] == "approved"]
        human_count = sum(q.get("review", {}).get("human_verified") is True for q in approved)
        manifest = dict(benchmark.manifest, version=version, parent_dataset_sha256=benchmark.fingerprint,
                        annotation_status="partially-reviewed", annotation_quality={**benchmark.manifest.get('annotation_quality', {}),
                            "approved_queries": len(approved), "human_verified_queries": human_count,
                            "reviewer_types": sorted({q.get("review", {}).get("reviewer_type", "unspecified") for q in approved})})
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(root / "queries.jsonl", queries.values())
        write_jsonl(root / "evidence.jsonl", evidence.values())
        write_jsonl(root / "qrels.jsonl", judgments.values())
        validated = load_benchmark(root)
        result = {"reviewed": len(reviews), "approved": len(validated.select()), "version": version}
        shutil.copytree(root, output)
    return result
