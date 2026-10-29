"""Load reviewed benchmarks without converting weak labels into ground truth."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


FILES = ("manifest.json", "corpus.jsonl", "queries.jsonl", "evidence.jsonl", "qrels.jsonl")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        _require(isinstance(value, str) and bool(value.strip()), f"missing/invalid {key}")
        _require(value not in result, f"duplicate {key}: {value}")
        result[value] = row
    return result


@dataclass
class Benchmark:
    root: Path
    manifest: dict
    corpus: dict
    queries: dict
    evidence: dict
    qrels: dict
    judgments: dict
    fingerprint: str
    file_hashes: dict

    def select(self, split=None):
        return [q for q in self.queries.values()
                if q["review_status"] == "approved" and (split is None or q["split"] == split)]


def load_benchmark(root):
    root = Path(root)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    _require(manifest.get("schema_version") == "productdoc.benchmark.v1", "unsupported benchmark schema")
    for field in ("name", "version", "corpus_version", "index_version"):
        _require(isinstance(manifest.get(field), str) and bool(manifest[field].strip()), f"manifest requires {field}")
    corpus = _index(read_jsonl(root / "corpus.jsonl"), "chunk_id")
    queries = _index(read_jsonl(root / "queries.jsonl"), "query_id")
    evidence = _index(read_jsonl(root / "evidence.jsonl"), "evidence_id")
    for chunk in corpus.values():
        _require(isinstance(chunk.get("doc_id"), str) and bool(chunk["doc_id"]), "corpus requires document doc_id")
        _require(isinstance(chunk.get("content"), str) and bool(chunk["content"].strip()), "corpus requires content")
    for item in evidence.values():
        for field in ("doc_id", "source_revision", "quote", "locator"):
            _require(bool(item.get(field)), f"evidence {item['evidence_id']} requires {field}")
    judgments, qrels = {}, {qid: {} for qid in queries}
    for row in read_jsonl(root / "qrels.jsonl"):
        qid, cid, grade = row.get("query_id"), row.get("chunk_id"), row.get("relevance")
        _require(qid in queries, f"qrels references unknown query: {qid}")
        _require(cid in corpus, f"qrels references unknown chunk: {cid}")
        _require(type(grade) is int and 0 <= grade <= 2, "relevance must be integer 0, 1 or 2")
        _require((qid, cid) not in judgments, f"duplicate qrel: {qid}/{cid}")
        ids = row.get("evidence_ids", [])
        _require(isinstance(ids, list), "qrel evidence_ids must be a list")
        for eid in ids:
            _require(eid in evidence, f"qrel references unknown evidence: {eid}")
            _require(evidence[eid]["doc_id"] == corpus[cid]["doc_id"], f"evidence/chunk document mismatch: {eid}/{cid}")
            _require(grade > 0, "irrelevant qrels cannot support evidence")
            _require(evidence[eid]["quote"] in corpus[cid]["content"],
                     f"mapped evidence quote is absent from chunk: {eid}/{cid}")
        judgments[qid, cid] = row
        qrels[qid][cid] = grade
    groups = {}
    for qid, query in queries.items():
        _require(isinstance(query.get("query"), str) and bool(query["query"].strip()), f"query text missing: {qid}")
        _require(query.get("split") in {"dev", "test", "regression"}, f"invalid split: {qid}")
        _require(query.get("review_status") in {"pending", "approved"}, f"invalid review_status: {qid}")
        _require(query.get("answerable") is None or type(query["answerable"]) is bool, f"invalid answerable: {qid}")
        for field in ("product_ids", "canonical_names", "evidence_ids", "reference_facts"):
            _require(isinstance(query.get(field, []), list), f"{field} must be a list: {qid}")
        for eid in query.get("evidence_ids", []):
            _require(eid in evidence, f"query references unknown evidence: {qid}/{eid}")
        fact_ids = []
        for index, fact in enumerate(query.get("reference_facts", [])):
            _require(isinstance(fact, dict) and bool(fact.get("text")), f"invalid reference fact: {qid}")
            fact_id = fact.get('fact_id', str(index))
            _require(isinstance(fact_id, str) and bool(fact_id.strip()) and fact_id not in fact_ids,
                     f"reference fact IDs must be unique nonempty strings: {qid}")
            fact_ids.append(fact_id)
            _require(bool(fact.get("evidence_ids")) and set(fact["evidence_ids"]) <= set(query.get("evidence_ids", [])),
                     f"reference facts need query evidence: {qid}")
            if "evidence_sets" in fact:
                _require(isinstance(fact["evidence_sets"], list) and bool(fact["evidence_sets"]), f"invalid evidence_sets: {qid}")
                for option in fact["evidence_sets"]:
                    _require(isinstance(option, list) and bool(option) and set(option) <= set(fact["evidence_ids"]),
                             f"evidence alternatives must reference fact evidence: {qid}")
        if query["review_status"] != "approved":
            continue
        if query.get("history"):
            _require(isinstance(query.get("retrieval_query"), str) and bool(query["retrieval_query"].strip()),
                     f"multi-turn retrieval benchmark requires a reviewed standalone retrieval_query: {qid}")
        _require(type(query.get("answerable")) is bool, f"approve answerability before evaluation: {qid}")
        group = query.get("group_id")
        _require(isinstance(group, str) and bool(group.strip()), f"approved query needs group_id: {qid}")
        _require(group not in groups or groups[group] == query["split"], f"group crosses splits: {group}")
        groups[group] = query["split"]
        positive = {cid for cid, grade in qrels[qid].items() if grade > 0}
        if query["answerable"]:
            _require(bool(positive), f"approved answerable query needs positive qrels: {qid}")
            _require(bool(query.get("evidence_ids")), f"approved query needs stable evidence: {qid}")
            mapped = {eid for cid in positive for eid in judgments[qid, cid].get("evidence_ids", [])}
            _require(set(query["evidence_ids"]) <= mapped, f"gold evidence not mapped to positive qrels: {qid}")
        else:
            _require(not positive and not query.get("evidence_ids"), f"unanswerable query has gold evidence: {qid}")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return Benchmark(root, manifest, corpus, queries, evidence, qrels, judgments, fingerprint, hashes)
