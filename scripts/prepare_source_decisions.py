"""Apply an explicit source-review plan to cached AI reference drafts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.evaluation.dataset import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'drafts', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    decisions = {'reviewed_at': datetime.now(timezone.utc).isoformat(), 'extra_evidence': plan['extra_evidence'], 'queries': {}}
    for row in read_jsonl(args.drafts):
        qid = row['query_id']
        facts = plan['reference_overrides'].get(qid, row['reference_facts'])
        for i, fact in enumerate(facts):
            fact['fact_id'] = f'{qid}:f{i + 1}'
            if len(fact['evidence_ids']) == 1 and fact['evidence_ids'][0] in plan.get('equivalent_evidence', {}):
                options = plan['equivalent_evidence'][fact['evidence_ids'][0]]
                fact.update(evidence_ids=options, evidence_sets=[[eid] for eid in options])
        pending = qid in plan['pending_scope_refinement']
        decisions['queries'][qid] = {'status': 'pending' if pending else 'approved',
                                     'reason': 'scope requires refinement' if pending else 'AI PDF source review; no human verification',
                                     'answerable': row['answerable'], 'reference_facts': facts}
    path = Path(args.output)
    if path.exists():
        raise ValueError('refusing to overwrite source review decisions')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
