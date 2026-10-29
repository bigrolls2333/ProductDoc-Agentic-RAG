"""Consolidate a completed legacy baseline and unjudged retrieval campaign."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_benchmark, write_jsonl
from app.evaluation.workflow import build_pool, collect_legacy_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--retrieval-reports", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="actual model used for the legacy baseline")
    parser.add_argument("--legacy-memory-writes", action="store_true", help="record that the baseline retained legacy product-memory writes")
    args = parser.parse_args()
    benchmark = load_benchmark(args.dataset)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.retrieval_reports]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pool = build_pool(benchmark, reports)
    write_jsonl(output / "annotation_pool.jsonl", pool)
    cases = collect_legacy_cases(baseline, output / "case_queue.jsonl")
    answer_records = []
    for result in baseline["results"]:
        if result["status"] == "error" or not result.get("answer"):
            continue
        artifacts = result.get("judge_artifacts", {})
        if artifacts.get("answer_context_trace", {}).get("context_source") != "actual_prompt":
            continue
        # Explicitly leave abstention for review rather than guessing from text.
        answer_records.append({"record_id": result["session_id"], "query_id": result["id"],
                               "answer": result["answer"], "abstained": None, "review_status": "pending",
                               "context_source": "actual_prompt", "context_docs": [
                                   {"chunk_id": doc.get("chunk_id"), "text": doc.get("text", "")}
                                   for doc in artifacts.get("answer_context_docs", [])]})
    write_jsonl(output / "answers_for_review.jsonl", answer_records)
    by_query = {}
    for row in pool:
        by_query.setdefault(row["query_id"], []).append(row)
    priorities = []
    failed = {r["id"]: r for r in baseline["results"] if r["status"] in {"failed", "error"}}
    for qid, rows in by_query.items():
        methods = {origin["variant"] for row in rows for origin in row["retrieved_by"]}
        disagreement = sum(len({origin["variant"] for origin in row["retrieved_by"]}) == 1 for row in rows)
        priorities.append({"query_id": qid, "category": benchmark.queries[qid].get("category"),
                           "legacy_failed": qid in failed, "method_count": len(methods),
                           "single_method_candidates": disagreement, "candidate_count": len(rows)})
    priorities.sort(key=lambda row: (not row["legacy_failed"], -row["single_method_candidates"], row["query_id"]))
    write_jsonl(output / "annotation_priorities.jsonl", priorities)
    summary = {"dataset_sha256": benchmark.fingerprint, "corpus_chunks": len(benchmark.corpus),
               "corpus_documents": len({row['doc_id'] for row in benchmark.corpus.values()}),
               "queries": len(benchmark.queries), "approved_queries": len(benchmark.select()),
               "human_verified_queries": sum(q.get('review', {}).get('human_verified') is True for q in benchmark.select()),
               "annotation_pairs": len(pool), "legacy_cases": cases, "answers_pending_review": len(answer_records),
               "legacy_runtime": {"model": args.model, "no_web": baseline.get("command_args", {}).get("no_web"),
                                  "answer_method": baseline.get("command_args", {}).get("answer_method"),
                                  "concurrency": baseline.get("command_args", {}).get("concurrency"),
                                  "product_memory_writes": args.legacy_memory_writes,
                                  "evaluated": len(baseline["results"]),
                                  "excluded_from_legacy_run": len(benchmark.queries) - len(baseline["results"])},
               "legacy_summary": baseline["summary"],
               "rescoring": baseline.get("rescoring"),
               "retrieval_runs": {variant: {key: row[key] for key in ("total", "completed", "errors")}
                                  for report in reports for variant, row in report["summary"].items()},
               "note": "Legacy source_hint / answer checks are weak supervision. Annotation-only retrieval produces no standard quality scores."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 商品文档评测：本轮结果", "", "本轮包含真实语料的旧口径 Agent 基线和新评测的候选池。标准 qrels 尚待审核。", "",
             f"- 冻结语料：{summary['corpus_chunks']} 个 chunks / {summary['corpus_documents']} 份文档。",
             f"- 问题：{summary['queries']} 条；已批准评测样本：{summary['approved_queries']} 条，其中人工核验 {summary['human_verified_queries']} 条。",
             f"- Agent 基线：{len(baseline['results'])} 条；因入库就绪标记等过滤条件排除 {summary['legacy_runtime']['excluded_from_legacy_run']} 条。",
             f"- 合并候选：{len(pool)} 对；待审核实际上下文答案：{len(answer_records)} 条。",
             "", "## 全量 Agent 基线（旧口径）", "", "下面是现有规则检查的结果，不等同于标准检索 Recall 或事实支持率。", "",
             "| 检查项 | 通过/失败/错误/跳过 | 通过率 |", "| --- | ---: | ---: |"]
    for name, item in sorted(baseline["summary"].get("by_dimension", {}).items()):
        counts = "/".join(str(item.get(key, 0)) for key in ("passed", "failed", "error", "skipped"))
        accuracy = item.get("accuracy")
        score = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        lines.append(f"| {name} | {counts} | {score} |")
    if baseline.get("rescoring"):
        before = baseline["original_summary"]["pass_rate"]
        after = baseline["summary"]["pass_rate"]
        lines += ["", f"修正文档路由不适用与空工具计划的误判后，检查交集通过率从 {before:.2%} 变为 {after:.2%}。这是零模型调用的重新计分，不代表系统质量提升；原始报告与调整明细保留。"]
    lines += ["", "## 候选实验完成情况", "", "每个变体均为 annotation-only，不发布未审核的标准质量指标。", "",
              "| 变体 | 完成 | 错误 |", "| --- | ---: | ---: |"]
    for variant, row in summary["retrieval_runs"].items():
        lines.append(f"| {variant} | {row['completed']}/{row['total']} | {row['errors']} |")
    lines += ["", "## 下一步数据工作", "", "先审核 annotation_priorities.jsonl 中的失败与策略分歧样本，对照原文填写证据锚点与 qrels；通过 review 发布新版本后运行标准指标。", "",
              "case_queue.jsonl 保留旧检查失败观察，不能直接当作根因金标。answers_for_review.jsonl 的 abstained 需要审核；将 query 批准后再运行答案可信度评测。", ""]
    if args.legacy_memory_writes:
        lines += ["本轮旧 Agent 基线在加入评测写隔离之前启动，保留了商品长期记忆回写行为；它用于问题发现，不能当作冻结记忆的严格消融结果。后续包装器默认禁用这些回写。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"legacy_summary", "note"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
