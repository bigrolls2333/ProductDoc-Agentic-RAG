"""Index original local PDFs for evidence verification without changing them."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_benchmark, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import fitz

    benchmark = load_benchmark(args.dataset)
    files = sorted([*ROOT.glob("doc/*.pdf"), *ROOT.glob("doc_eval/*.pdf")])
    by_stem = {}
    for path in files:
        by_stem.setdefault(path.stem, []).append(path)
    documents = {}
    for chunk in benchmark.corpus.values():
        documents.setdefault(chunk["doc_id"], chunk["file_title"])
    rows = []
    for doc_id, title in documents.items():
        paths = by_stem.get(title, [])
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if not paths or len(hashes) != 1:
            rows.append({"doc_id": doc_id, "title": title, "status": "missing_or_ambiguous"})
            continue
        path = paths[0]
        with fitz.open(path) as pdf:
            pages = [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(pdf)]
        rows.append({"doc_id": doc_id, "title": title, "status": "indexed", "path": str(path),
                     "pdf_sha256": next(iter(hashes)), "pages": pages})
    write_jsonl(args.output, rows)
    print(json.dumps({"documents": len(rows), "indexed": sum(r["status"] == "indexed" for r in rows),
                      "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
