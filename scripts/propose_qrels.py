"""Create resumable LLM-assisted qrel drafts over all target-document chunks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.annotation import ANNOTATOR_PROMPT, ANNOTATOR_VERSION, clean_content, proposal_identity, verify_proposals
from app.evaluation.dataset import load_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.concurrency) < 1:
        raise ValueError("batch-size and concurrency must be positive")
    from app.conf.lm_config import lm_config
    from openai import OpenAI
    from app.lm.request_options import non_thinking_body

    benchmark = load_benchmark(args.dataset)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["queries"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=lm_config.api_key, base_url=lm_config.base_url, timeout=120, max_retries=1)
    model = lm_config.llm_model
    tasks = []
    for qid, doc_ids in selection.items():
        query = benchmark.queries[qid]
        chunks = [dict(chunk_id=cid, doc_id=row["doc_id"], content=clean_content(row["content"]))
                  for cid, row in benchmark.corpus.items() if row["doc_id"] in doc_ids]
        if not chunks:
            raise ValueError(f"target document missing: {qid}")
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start:start + args.batch_size]
            question = query.get('retrieval_query') or query['query']
            identity = proposal_identity(question, batch, model)
            tasks.append((qid, question, batch, identity))

    def annotate(task):
        qid, question, chunks, identity = task
        path = output / f"{qid}_{identity[:16]}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and existing.get("identity") == identity:
                verify_proposals(existing, chunks)
                return "cached"
        messages = [{"role": "system", "content": ANNOTATOR_PROMPT},
                    {"role": "user", "content": json.dumps({"question": question, "chunks": chunks}, ensure_ascii=False)}]
        error = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"},
                                                          messages=messages, max_tokens=7000, extra_body=non_thinking_body(lm_config.base_url))
                result = json.loads(response.choices[0].message.content)
                rows = verify_proposals(result, chunks)
                payload = {"query_id": qid, "identity": identity, "annotator_version": ANNOTATOR_VERSION,
                           "model": model, "status": "completed", "judgments": rows,
                           "requested_thinking_enabled": False,
                           "usage": response.usage.model_dump() if response.usage else None,
                           "human_verified": False}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(path)
                return "completed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                # Only include validation feedback, never credentials or raw API exception text.
                feedback=str(exc) if isinstance(exc,(ValueError,TypeError,KeyError)) else 'API request failed'
                messages.append({"role": "user", "content": f"前次输出未通过原文与JSON结构检查：{feedback}。请逐字复制quote，可缩短到一个直接支持事实的连续短句，保留HTML标签与标点；输出所有输入chunk，0相关facts为空，正相关facts不可为空。"})
        path.write_text(json.dumps({"query_id": qid, "identity": identity, "status": "error",
                                    "error_type": error.split(":", 1)[0],
                                    "validation_error": error if error.startswith("ValueError:") else None}, indent=2), encoding="utf-8")
        return "error"

    counts = {"completed": 0, "cached": 0, "error": 0}
    started = time.perf_counter()
    print(f"Annotation batches: {len(tasks)}, queries: {len(selection)}, model: {model}", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(annotate, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            counts[future.result()] += 1
            if index % 20 == 0 or index == len(tasks):
                print(json.dumps({"done": index, "total": len(tasks), **counts}), flush=True)
    summary = {"queries": len(selection), "batches": len(tasks), **counts,
               "seconds": time.perf_counter() - started, "dataset_sha256": benchmark.fingerprint,
               "model": model, "human_verified": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
