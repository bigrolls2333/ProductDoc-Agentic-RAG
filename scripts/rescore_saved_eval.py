"""Correct routing-score semantics using saved artifacts, with zero model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.legacy import rescore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        raise ValueError("keep the raw report; use a different output path")
    report = rescore(json.loads(source.read_text(encoding="utf-8")))
    report["rescoring"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "rescoring": report["rescoring"], "status": report["summary"]["status"],
                      "pass_rate": report["summary"]["pass_rate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
