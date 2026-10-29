"""Verify annotation quotes against original PDFs and prepare reference drafts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.annotation import locate_quote
from app.evaluation.dataset import load_benchmark, read_jsonl, write_jsonl


REFERENCE_PROMPT = """你是商品文档参考答案审核员。只根据提供的已核验原文事实，选出回答问题所必要的最小要点集合。
候选facts可能有相关性误判或超出原文的归纳，你必须排除。不得利用外部知识，不得参考任何系统答案。
对故障问题，原文只给配置步骤时，应写成应核对的配置前提，不得捏造确定故障原因。
需要多商品时每个商品至少有一个要点。问题只是问设备类型时只保留类型及必要定位，不要增加无关详细参数。
步骤问题最多6个必要要点，其他问题最多4个。一个要点可由多个 evidence_id 联合支持，但不能遗漏用户问题中的必要方面。
如原文不足以支撑完整问题，输出 answerable=false 并说明缺失；不可自行补全。
若给出original_query和history，必须确认question与原始对话需求等价；多轮独立改写若遗漏或新增信息需求，输出answerable=false并说明。
返回JSON：{"answerable":true,"reference_facts":[{"text":"必要事实","evidence_ids":["实际ID"]}],"reason":"判定依据"}。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument('--allow-incomplete',action='store_true',help='retain failed/missing annotations as unknown, never as negative labels')
    args = parser.parse_args()
    from openai import OpenAI
    from app.conf.lm_config import lm_config
    from app.lm.request_options import non_thinking_body

    benchmark = load_benchmark(args.dataset)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["queries"]
    sources = {row["doc_id"]: row for row in read_jsonl(args.sources)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "reference_drafts"
    cache.mkdir(exist_ok=True)
    proposals = {qid: {} for qid in selection}
    failed_batches = []
    for path in Path(args.proposals).glob("VAL*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        qid = payload["query_id"]
        if payload["status"] != "completed":
            failed_batches.append(path.name)
            continue
        for row in payload["judgments"]:
            cid = row["chunk_id"]
            if cid in proposals[qid]:
                raise ValueError(f"duplicate judgments; clean obsolete proposal batches: {qid}/{cid}")
            proposals[qid][cid] = row
    if failed_batches and not args.allow_incomplete:
        raise ValueError(f"retry failed annotation batches before source audit: {len(failed_batches)}")
    audits = {}
    for qid, doc_ids in selection.items():
        target = {cid for cid, chunk in benchmark.corpus.items() if chunk["doc_id"] in doc_ids}
        if (set(proposals[qid])-target) or (target != proposals[qid].keys() and not args.allow_incomplete):
            raise ValueError(f"annotation batches incomplete: {qid}")
        verified, unverified, judgments = [], [], []
        for cid, row in proposals[qid].items():
            chunk = benchmark.corpus[cid]
            source = sources[chunk["doc_id"]]
            evidence_ids = []
            for fact in row["facts"]:
                if fact['quote'] not in chunk['content']:
                    unverified.append({'chunk_id':cid,**fact,'reason':'cleaned-text quote does not occur verbatim in original chunk'})
                    continue
                pages = locate_quote(fact["quote"], source)
                if not pages:
                    unverified.append({"chunk_id": cid, **fact})
                    continue
                identity = json.dumps([chunk["doc_id"], source["pdf_sha256"], fact["quote"]], ensure_ascii=False)
                eid = "ev_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
                evidence_ids.append(eid)
                verified.append({"evidence_id": eid, "text": fact["text"], "quote": fact["quote"],
                                 "doc_id": chunk["doc_id"], "chunk_id": cid, "source_revision": source["pdf_sha256"],
                                 "locator": {"path": source["path"], "pdf_pages": pages,
                                             "match": "NFKC_whitespace_markdown_structure_normalization"}})
            # A failed source match is unknown, never an automatic negative label.
            if row["relevance"] == 0 or evidence_ids:
                judgments.append({"query_id": qid, "chunk_id": cid, "relevance": row["relevance"],
                                  "evidence_ids": sorted(set(evidence_ids))})
        audits[qid] = {"query_id": qid, "query": benchmark.queries[qid].get('retrieval_query') or benchmark.queries[qid]["query"], "target_doc_ids": doc_ids,
                       "target_chunks": len(target), "judged_chunks": len(judgments),
                       'missing_chunk_ids':sorted(target-set(proposals[qid])),
                       "verified_facts": verified, "unverified_facts": unverified, "qrels": judgments}
    write_jsonl(output / "source_audits.jsonl", audits.values())
    client = OpenAI(api_key=lm_config.api_key, base_url=lm_config.base_url, timeout=120, max_retries=1)

    def reference(item):
        qid, audit = item
        facts = list({fact['evidence_id']:{key: fact[key] for key in ("evidence_id", "text", "quote", "doc_id")}
                      for fact in audit["verified_facts"]}.values())
        payload = {"question": audit["query"], "original_query":benchmark.queries[qid]['query'],
                   'history':benchmark.queries[qid].get('history',[]),
                   "required_document_ids": audit["target_doc_ids"], "facts": facts}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = cache / f"{qid}_{digest[:16]}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if not facts:
            draft = {"answerable": False, "reference_facts": [], "reason": "No source-verified answer evidence in full target-document scan."}
        else:
            response = client.chat.completions.create(model=lm_config.llm_model, temperature=0,
                messages=[{"role": "system", "content": REFERENCE_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"}, max_tokens=3000, extra_body=non_thinking_body(lm_config.base_url))
            draft = json.loads(response.choices[0].message.content)
        if type(draft.get("answerable")) is not bool or not isinstance(draft.get("reference_facts"), list):
            raise ValueError(f"invalid reference draft: {qid}")
        known = {fact["evidence_id"] for fact in facts}
        for fact in draft["reference_facts"]:
            if not fact.get("text") or not fact.get("evidence_ids") or not set(fact["evidence_ids"]) <= known:
                raise ValueError(f"reference draft invented evidence: {qid}")
        draft.update(query_id=qid, identity=digest, human_verified=False, model=lm_config.llm_model)
        path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        return draft

    drafts = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for future in as_completed([executor.submit(reference, item) for item in audits.items()]):
            drafts.append(future.result())
    write_jsonl(output / "reference_drafts.jsonl", sorted(drafts, key=lambda row: row["query_id"]))
    summary = {"queries": len(audits), "source_verified_facts": sum(len(r["verified_facts"]) for r in audits.values()),
               "unmatched_facts": sum(len(r["unverified_facts"]) for r in audits.values()),
               "answerable_drafts": sum(r["answerable"] for r in drafts), "human_verified": False}
    summary.update(failed_annotation_batches=len(failed_batches),
                   incomplete_queries=sum(bool(r['missing_chunk_ids']) for r in audits.values()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
