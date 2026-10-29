"""Extract actual prompt contexts using explicit reviewed abstention decisions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.answer_eval import validate_answer_record
from app.evaluation.dataset import load_benchmark, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'baseline', 'decisions', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    benchmark = load_benchmark(args.dataset)
    report = json.loads(Path(args.baseline).read_text(encoding='utf-8'))
    rows = {r['id']: r for r in report['results']}
    decisions = json.loads(Path(args.decisions).read_text(encoding='utf-8'))
    provenance = {key + '_sha256': hashlib.sha256(Path(value).read_bytes()).hexdigest()
                  for key, value in [('baseline', args.baseline), ('decisions', args.decisions)]}
    records = []
    for query in benchmark.select('dev'):
        qid = query['query_id']
        decision, row = decisions['queries'][qid], rows[qid]
        if type(decision.get('abstained')) is not bool or not decision.get('reason'):
            raise ValueError('each selected answer needs an explicit abstention review')
        artifacts = row['judge_artifacts']
        trace = artifacts.get('answer_context_trace', {})
        record = {'record_id': 'baseline:' + qid, 'query_id': qid, 'answer': row['answer'],
                  'abstained': decision['abstained'], 'context_source': trace.get('context_source', 'no_generation'),
                  'context_docs': artifacts.get('answer_context_docs', []),
                  'context_trace': trace, 'abstention_review': decision, 'provenance': provenance}
        validate_answer_record(benchmark, record)
        records.append(record)
    write_jsonl(args.output, records)
    print(json.dumps({'records': len(records), 'abstained': sum(r['abstained'] for r in records),
                      'no_generation': sum(r['context_source'] == 'no_generation' for r in records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
