"""Score saved answers against their actual context; errors never become passes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.answer_eval import JUDGE_INSTRUCTIONS, JUDGE_VERSION, judge_answer, summarize_judgment
from app.evaluation.dataset import load_benchmark, read_jsonl
from app.evaluation.metrics import average_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--judgments", help="optional saved judgments JSONL keyed by record_id; otherwise call LLM")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    benchmark = load_benchmark(args.dataset)
    answers = read_jsonl(args.answers)
    ids = [row.get("record_id") for row in answers]
    if not answers or any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("answers need unique nonempty record_id values")
    saved, saved_provenance = None, []
    if args.judgments:
        saved = {}
        for row in read_jsonl(args.judgments):
            if row["record_id"] in saved:
                raise ValueError("duplicate saved judgment record_id")
            saved[row["record_id"]] = row["judgment"]
            if row.get('judge') and row['judge'] not in saved_provenance:
                saved_provenance.append(row['judge'])
    model = None
    if saved is None:
        from app.conf.lm_config import lm_config
        model = lm_config.llm_model
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    def assess(answer):
        row = {"record_id": answer["record_id"], "query_id": answer.get("query_id"), "status": "completed",
               "split": benchmark.queries.get(answer.get("query_id"), {}).get("split"), "answer_record": answer}
        try:
            row["assessment"] = (summarize_judgment(benchmark, answer, saved[answer["record_id"]])
                                 if saved is not None else judge_answer(benchmark, answer))
            row["diagnosis"] = row["assessment"]["diagnosis"]
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            row["diagnosis"] = [{"type": "execution_error", "status": "observed", "reason": row["error"]}]
        return row
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(assess, answers))
    completed = [row["assessment"] for row in results if row["status"] == "completed"]
    answerable = [row for row in completed if row["answerable"]]
    no_answer = [row for row in completed if not row["answerable"]]
    report = {"schema_version": "productdoc.answer-eval.v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": benchmark.fingerprint,
              "answers_sha256": hashlib.sha256(Path(args.answers).read_bytes()).hexdigest(),
              "judge": {"version": JUDGE_VERSION, "model": model, "mode": "saved" if saved is not None else "llm",
                        "prompt_sha256": hashlib.sha256(JUDGE_INSTRUCTIONS.encode()).hexdigest(),
                        "saved_sha256": hashlib.sha256(Path(args.judgments).read_bytes()).hexdigest() if saved is not None else None},
              "saved_judge_provenance": saved_provenance,
              "summary": {"total": len(results), "completed": len(completed), "errors": len(results) - len(completed),
                          "metrics": average_metrics(completed),
                          "false_answer_rate": sum(not row["abstained"] for row in no_answer) / len(no_answer) if no_answer else None,
                          "unanswerable_n": len(no_answer),
                          "false_abstention_rate": sum(row["abstained"] for row in answerable) / len(answerable) if answerable else None,
                          "answerable_n": len(answerable)}, "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 1 if len(completed) != len(results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
