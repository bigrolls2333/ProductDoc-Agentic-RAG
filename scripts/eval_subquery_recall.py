"""Paired live recall ablation using saved actual upstream state, never gold products."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.dataset import load_benchmark
from app.evaluation.metrics import average_metrics, evidence_metrics, retrieval_metrics
from app.query_process.context_selection import normalize_item_names, select_context_documents, doc_item_name
from app.query_process.subquery_retrieval import SUBQUERY_VERSION, fuse_routes, validate_subqueries

VARIANTS = {'main': (False, False), 'main_hyde': (True, False),
            'main_subquery': (False, True), 'main_hyde_subquery': (True, True)}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--agent-report')
    inputs.add_argument('--reviewed-inputs',action='store_true',help='conditional retrieval benchmark with reviewed product scope and standalone query; not actual Agent inputs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--per-item-k', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.per_item_k <= 50 or not 1 <= args.concurrency <= 8:
        raise ValueError('invalid recall or concurrency limit')
    b = load_benchmark(args.dataset)
    selected = {q['query_id']: q for q in b.select('dev')}
    actual = {}
    source = json.loads(Path(args.agent_report).read_text(encoding='utf-8')) if args.agent_report else {'results':[
        {'id':q['query_id'],'query':q['query'],'judge_artifacts':{'actual':{
            'item_names':q['canonical_names'],'rewritten_query':q.get('retrieval_query') or q['query']}}} for q in selected.values()]}
    for row in source['results']:
        qid = row['id']
        if qid not in selected:
            continue
        if qid in actual or row['query'] != selected[qid]['query']:
            raise ValueError('saved upstream query mismatch or duplicate')
        state = row['judge_artifacts']['actual']
        names = normalize_item_names(state['item_names'])
        query = state.get('rewritten_query') or row['query']
        if not names or not isinstance(query, str) or not query.strip():
            raise ValueError('experiment requires saved confirmed items and query')
        actual[qid] = {'query': query, 'item_names': names}
    if actual.keys() != selected.keys():
        raise ValueError('saved upstream state does not cover approved dev queries')
    output = Path(args.output_dir)
    cache = output/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['EVAL_DISABLE_PRODUCT_MEMORY'] = '1'
    os.environ['EVAL_DISABLE_WEB'] = '1'
    # Load the Milvus/Arrow stack before chat SDK native extensions on Windows.
    # Loading Arrow later in this environment causes a native access violation.
    from app.evaluation.retrievers import MilvusRetriever
    backend = MilvusRetriever(b.corpus, b.manifest['source_collection'], query_encoder='encode_documents')
    from app.lm.reranker_utils import get_reranker_model
    from app.conf.reranker_config import reranker_config
    reranker = get_reranker_model()
    from app.conf.lm_config import lm_config
    from app.lm.lm_utils import get_llm_client
    from app.query_process.agent.nodes.node_retrieval_plan import _llm_decide_retrieval_plan
    from app.query_process.agent.nodes.node_search_embedding_hyde import step_1_create_hyde_doc
    get_llm_client(json_mode=True)
    get_llm_client()
    files = ['app/query_process/subquery_retrieval.py', 'app/query_process/agent/nodes/node_retrieval_plan.py',
             'prompts/hyde_prompt.prompt', 'scripts/eval_subquery_recall.py']
    source_sha = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    # Plan and HyDE caches depend on their inputs/prompts, not labels or downstream metrics.
    plan_source = {p: source_sha[p] for p in files[:2]}

    def prepare(qid):
        item = actual[qid]
        key = digest([item, plan_source, source_sha[files[2]], lm_config.llm_model,
                      lm_config.base_url, lm_config.llm_temperature])
        path = cache/f'plan_{qid}_{key[:16]}.json'
        if path.exists():
            return qid, json.loads(path.read_text(encoding='utf-8'))
        state = dict(item_names=item['item_names'], rewritten_query=item['query'], history=[],
                     subquery_retrieval_enabled=True, product_memory_hints={})
        start = time.perf_counter()
        plan = _llm_decide_retrieval_plan(state)
        plan_ms = (time.perf_counter()-start)*1000
        if plan.get('subquery_status') == 'invalid_plan':
            raise ValueError(f'{qid}: invalid subquery plan: {plan.get("subquery_error")}')
        validate_subqueries(plan.get('subqueries', []), item['item_names'], item['query'])
        start = time.perf_counter()
        hyde = step_1_create_hyde_doc(item['query'])
        value = {'identity':key, 'planner':plan, 'hyde':hyde,
                 'planner_ms':plan_ms, 'hyde_generation_ms':(time.perf_counter()-start)*1000}
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        return qid, value

    prepared = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for future in as_completed([executor.submit(prepare, qid) for qid in selected]):
            qid, value = future.result()
            prepared[qid] = value
            print(json.dumps({'phase':'planning', 'completed':len(prepared), 'total':len(selected)}, ensure_ascii=False), flush=True)

    results = []
    for qid, question in selected.items():
        item, planning = actual[qid], prepared[qid]
        names, query = item['item_names'], item['query']
        key = digest([b.fingerprint, item, planning['identity'], args.per_item_k,
                      backend.metadata, source_sha['app/query_process/subquery_retrieval.py'],
                      reranker_config.bge_reranker_large, reranker_config.bge_reranker_fp16,
                      'pool50_context20_chars12000.v1'])
        path = cache/f'retrieval_{qid}_{key[:16]}.json'
        if path.exists():
            rows = json.loads(path.read_text(encoding='utf-8'))
            results.extend(rows)
            print(json.dumps({'phase':'retrieval', 'query_id':qid, 'cached':True}), flush=True)
            continue
        eligible = {name:[cid for cid, doc in b.corpus.items() if doc_item_name(doc) == name] for name in names}
        judgments = {cid:j for (jqid,cid),j in b.judgments.items() if jqid == qid}
        grades = b.qrels[qid]

        def search(text, indices):
            groups = []
            start = time.perf_counter()
            for index in indices:
                hits = backend.search(text, eligible[names[index]], 'hybrid', args.per_item_k)
                groups.append(hits)
            from itertools import zip_longest
            docs = [dict(b.corpus[hit['chunk_id']], text=b.corpus[hit['chunk_id']]['content'])
                    for round_hits in zip_longest(*groups) for hit in round_hits if hit is not None]
            return docs, (time.perf_counter()-start)*1000

        main_docs, main_ms = search(query, list(range(len(names))))
        hyde_docs, hyde_ms = search(query+' '+planning['hyde'], list(range(len(names))))
        sub_lists, sub_ms, sub_calls = [], 0, 0
        for sub in planning['planner'].get('subqueries', []):
            docs, elapsed = search(sub['query'], sub['item_indices'])
            sub_lists.append(docs); sub_ms += elapsed; sub_calls += len(sub['item_indices'])
        candidates = {variant:fuse_routes(main_docs, hyde_docs if use_hyde else [],
                      sub_lists if use_sub else [], item_names=names, per_item_k=args.per_item_k)
                      for variant,(use_hyde,use_sub) in VARIANTS.items()}
        union = {d['chunk_id']:d for docs in candidates.values() for d in docs}
        started = time.perf_counter()
        scores = reranker.compute_score([[query,d['content']] for d in union.values()], max_length=512)
        rank_ms = (time.perf_counter()-started)*1000
        score_map = {cid:float(score) for cid,score in zip(union, scores)}
        rows = []
        main_ids, hyde_ids = {d['chunk_id'] for d in main_docs}, {d['chunk_id'] for d in hyde_docs}
        sub_ids = {d['chunk_id'] for docs in sub_lists for d in docs}
        for variant,(use_hyde,use_sub) in VARIANTS.items():
            fused = candidates[variant]
            ranking = sorted([dict(d,score=score_map[d['chunk_id']],source='local') for d in fused],
                             key=lambda d:d['score'], reverse=True)
            included, _, trace = select_context_documents(ranking, query, item_names=names,
                policy='item_balanced', per_item_k=args.per_item_k)
            compact = lambda docs:[{'chunk_id':d['chunk_id'],'score':d.get('score',d.get('fusion_score',0)),
                                   'retrieval_sources':d.get('retrieval_sources',[])} for d in docs]
            row = {'query_id':qid, 'variant':variant, 'status':'completed', 'category':question.get('category','unknown'),
                   'answerable':question['answerable'], 'actual_upstream':item, 'planner':planning['planner'],
                   'input_source':'reviewed_product_scope' if args.reviewed_inputs else 'saved_agent_output',
                   'cohort':question.get('cohort','original_diagnostic'),
                   'stages':{'candidates':compact(fused),'context':compact(included)},
                   'rerank_order':compact(ranking), 'context_trace':trace,
                   'metrics':{'candidates':retrieval_metrics(fused, grades),
                              'rerank_order':retrieval_metrics(ranking, grades),
                              'candidates_evidence':evidence_metrics(fused,question['evidence_ids'],judgments,question['reference_facts']),
                              'context_evidence':evidence_metrics(included,question['evidence_ids'],judgments,question['reference_facts'])},
                   'route_diagnostics':{'subquery_new_vs_main':sorted(sub_ids-main_ids),
                       'subquery_new_vs_main_hyde':sorted(sub_ids-main_ids-hyde_ids),
                       'raw_main_count':len(main_ids),'raw_hyde_count':len(hyde_ids),'raw_subquery_count':len(sub_ids)},
                   'timings_ms':{'main_search':main_ms,'hyde_search':hyde_ms if use_hyde else 0,
                       'subquery_search':sub_ms if use_sub else 0,'rerank_shared_union':rank_ms,
                       'planner':planning['planner_ms'] if use_sub else 0,
                       'hyde_generation':planning['hyde_generation_ms'] if use_hyde else 0},
                   'search_calls':len(names)*(1+int(use_hyde))+(sub_calls if use_sub else 0)}
            rows.append(row)
        path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        results.extend(rows)
        print(json.dumps({'phase':'retrieval','query_id':qid,'queries':planning['planner'].get('subqueries',[]),
                          'context_facts':{r['variant']:r['metrics']['context_evidence']['fact_recall'] for r in rows}},ensure_ascii=False),flush=True)

    summary = {}
    for variant in VARIANTS:
        rows = [r for r in results if r['variant']==variant]
        summary[variant] = {'total':len(rows),'completed':len(rows),'errors':0,
            'metrics':{stage:average_metrics([r['metrics'][stage] for r in rows])
                       for stage in ['candidates','rerank_order','candidates_evidence','context_evidence']},
            'mean_search_calls':sum(r['search_calls'] for r in rows)/len(rows),
            'mean_context_chars':sum(r['context_trace']['context_chars'] for r in rows)/len(rows)}
    comparisons = []
    for left,right in [('main','main_subquery'),('main_hyde','main_hyde_subquery')]:
        pairs = [(next(r for r in results if r['query_id']==qid and r['variant']==left),
                  next(r for r in results if r['query_id']==qid and r['variant']==right)) for qid in selected]
        comparisons.append({'baseline':left,'variant':right,
            'context_fact_wins':[a['query_id'] for a,z in pairs if a['answerable'] and z['metrics']['context_evidence']['fact_recall']>a['metrics']['context_evidence']['fact_recall']],
            'context_fact_losses':[a['query_id'] for a,z in pairs if a['answerable'] and z['metrics']['context_evidence']['fact_recall']<a['metrics']['context_evidence']['fact_recall']]})
    report = {'schema_version':'productdoc.subquery-experiment.v1','created_at':datetime.now(timezone.utc).isoformat(),
        'dataset_sha256':b.fingerprint, 'source_sha256':source_sha,
        'agent_report_sha256':hashlib.sha256(Path(args.agent_report).read_bytes()).hexdigest() if args.agent_report else None,
        'config':{'candidate_k':50,'per_item_k':args.per_item_k,'context_chars':12000,'context_max_docs':20,
                  'subquery_version':SUBQUERY_VERSION,'model':lm_config.llm_model,'backend':backend.metadata,
                  'scope':('reviewed product scope and standalone query; conditional retrieval only; excludes entity and rewrite errors'
                           if args.reviewed_inputs else 'saved actual canonical item_names; fresh live recall; no gold filters'),
                  'memory':'none','web':False,'ks':[1,3,5,10,20,50]},
        'latency_policy':'API generation and searches shared; reranking computed once over variant union, not serving latency',
        'summary':summary,'paired_comparisons':comparisons,'results':results}
    (output/'retrieval.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output/'retrieval.json'),'comparisons':comparisons,
         'fact_recall':{v:s['metrics']['context_evidence']['fact_recall'] for v,s in summary.items()}},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
