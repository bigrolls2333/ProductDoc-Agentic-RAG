"""Simulate context selection over frozen rankings; no retrieval or answer calls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.evaluation.dataset import load_benchmark
from app.evaluation.diagnosis import diagnose
from app.evaluation.metrics import average_metrics, evidence_metrics
from app.evaluation.runner import select_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--context-ks', default='5,10,20')
    parser.add_argument('--budgets', default='6000,12000,24000')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    benchmark = load_benchmark(args.dataset)
    report = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    if report['dataset_sha256'] != benchmark.fingerprint:
        raise ValueError('prediction dataset mismatch')
    ks, budgets = [int(v) for v in args.context_ks.split(',')], [int(v) for v in args.budgets.split(',')]
    if not ks or not budgets or min(ks + budgets) < 1 or max(ks) > report['config']['candidate_k']:
        raise ValueError('invalid context sweep bounds')
    results, summaries = [], []
    for variant in report['config']['variants']:
        for k in ks:
            for budget in budgets:
                group = []
                for row in report['results']:
                    if row['variant'] != variant or row['status'] != 'completed':
                        continue
                    query = benchmark.queries[row['query_id']]
                    judgments = {cid: j for (qid, cid), j in benchmark.judgments.items() if qid == row['query_id']}
                    selected = row['rerank_order'][:k]
                    context, usage = select_context(selected, benchmark.corpus, budget)
                    stages = dict(row['stages'], reranked=selected, context=context)
                    item = {'query_id': row['query_id'], 'variant': variant, 'context_k': k, 'budget': budget,
                            'metrics': evidence_metrics(context, query['evidence_ids'], judgments, query['reference_facts']),
                            'context_chars': usage['content_chars'], 'context_chunk_ids': [d['chunk_id'] for d in context],
                            'dropped_chunk_ids': usage['dropped_chunk_ids'], 'diagnosis': diagnose(query, stages, judgments)}
                    group.append(item)
                results.extend(group)
                summaries.append({'variant': variant, 'context_k': k, 'budget': budget, 'completed': len(group),
                                  'errors': sum(r['variant'] == variant and r['status'] != 'completed' for r in report['results']),
                                  'metrics': average_metrics([g['metrics'] for g in group]),
                                  'mean_context_chars': sum(g['context_chars'] for g in group) / len(group) if group else None,
                                  'diagnosis_counts': dict(Counter(d['type'] for g in group for d in g['diagnosis']))})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'dataset_sha256': benchmark.fingerprint,
                                 'predictions_sha256': hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
                                 'policy': 'fixed ranking + whole chunk character budget; simulation only, not answer quality or serving latency',
                                 'model_calls': 0, 'summary': summaries, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Context selection sweep', '', 'Frozen rankings; no new answers. Character counts exclude prompt metadata.', '',
             '| Variant | Top-K | Budget | Fact recall | All facts | Mean chars |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in summaries:
        m = item['metrics']
        lines.append(f"| {item['variant']} | {item['context_k']} | {item['budget']} | {m['fact_recall']['mean']:.4f} (n={m['fact_recall']['n']}) | {m['all_facts']['mean']:.4f} | {item['mean_context_chars']:.0f} |")
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'configurations': len(summaries), 'observations': len(results), 'model_calls': 0}))


if __name__ == '__main__':
    main()
