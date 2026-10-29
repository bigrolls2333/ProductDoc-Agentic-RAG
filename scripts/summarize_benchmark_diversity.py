"""Profile benchmark breadth and cohort-separated retrieval results without API calls."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from app.evaluation.dataset import load_benchmark
from app.evaluation.metrics import average_metrics
from app.evaluation.workflow import collect_cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--original',default='eval/local/productdoc-source-v2')
    parser.add_argument('--build-dir',default='eval/local/diverse_build_v1');parser.add_argument('--experiment')
    parser.add_argument('--case-dir',default='logs/source_diagnostic_diverse_v1')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    b,old=load_benchmark(args.dataset),load_benchmark(args.original)
    original=old.select();approved=b.select();new=[q for q in approved if q['query_id'] not in old.queries]
    pending=[q for q in b.queries.values() if q.get('cohort')=='diverse_source_v1' and q['review_status']!='approved']
    def profile(rows):
        counts=Counter(n for q in rows for n in q['canonical_names'])
        return {'queries':len(rows),'products':len(counts),'product_question_counts':dict(counts),
                'categories':dict(Counter(q['category'] for q in rows)),
                'styles':dict(Counter(q.get('question_style','legacy_unspecified') for q in rows)),
                'multi_product':sum(len(q['canonical_names'])>1 for q in rows),
                'multi_turn':sum(bool(q.get('history')) for q in rows),
                'table_seed':sum(q.get('facets',{}).get('table_evidence',False) for q in rows),
                'table_seed_labeled':sum('table_evidence' in q.get('facets',{}) for q in rows),
                'answerable':sum(q['answerable'] for q in rows),
                'unanswerable':sum(q['answerable'] is False for q in rows),
                'human_verified':sum(q.get('review',{}).get('human_verified') is True for q in rows)}
    profiles={label:profile(rows) for label,rows in [('original',original),('new',new),('all',approved)]}
    summary={'dataset_sha256':b.fingerprint,'profiles':profiles,'pending_new':len(pending),
             'legacy_pending':sum(q['review_status']=='pending' for q in old.queries.values()),
             'corpus_documents':len({d['doc_id'] for d in b.corpus.values()}),'corpus_chunks':len(b.corpus)}
    normalized=[''.join(q['query'].split()).casefold() for q in approved]
    summary['exact_duplicate_questions']=len(normalized)-len(set(normalized))
    # Source reuse is visible; variations do not count as independent documents.
    anchor_sets=[tuple(sorted((b.evidence[eid]['doc_id'],b.evidence[eid]['quote']) for eid in q['evidence_ids']))
                 for q in approved if q['answerable']]
    summary['identical_reference_anchor_sets']=sum(n-1 for n in Counter(anchor_sets).values() if n>1)
    pending_path=Path(args.build_dir)/'pending_review.json'
    seeds_path=Path(args.build_dir)/'seeds.json'
    if pending_path.exists() and seeds_path.exists():
        pending_records=json.loads(pending_path.read_text(encoding='utf-8'))
        seeds=json.loads(seeds_path.read_text(encoding='utf-8'))
        observations=[]
        for record in pending_records:
            qid=record['query_id']
            if qid not in b.queries:continue
            observations.append({'query_id':qid,'variant':'source_diversification','status':'pending',
                'query':b.queries[qid]['query'],'source_seed_evidence':seeds[qid]['seed_evidence'],
                'reference_review_reason':record.get('reference_reason'),
                'diagnosis':[{'type':'source_annotation_review','status':'review_required','reasons':record['reasons']}]})
        summary['pending_case_queue']=collect_cases({'dataset_sha256':b.fingerprint,'config':{'split':'dev','cohort':'diverse_source_v1'},
                                                     'results':observations},Path(args.case_dir)/'case_queue.jsonl')
    lines=['# 扩充开发评测集：商品与问题多样性','',
           f"当前版本共有{len(approved)}道已通过AI来源审核的开发题，其中保留原有{len(original)}题，新增{len(new)}题；另有{len(pending)}道新增题待审。",'',
           '**这是来源约束的合成开发集，不是人工金标或独立盲测集；题量增加不等于获得同等数量的独立真实用户问题。**','',
           '## 覆盖变化','',
           '| 维度 | 原有题 | 新增已审核题 | 合计 |','| --- | ---: | ---: | ---: |']
    for key,label in [('queries','题数'),('products','涉及商品数'),('multi_product','多商品题'),
                      ('multi_turn','多轮指代题'),('table_seed','含表格种子证据'),('answerable','可回答题'),
                      ('unanswerable','无答案题'),('human_verified','人工核验题')]:
        cells=[]
        for x in ['original','new','all']:
            value=str(profiles[x][key])
            if key=='table_seed':
                if not profiles[x]['table_seed_labeled']:value='未补标签'
                elif profiles[x]['table_seed_labeled']<profiles[x]['queries']:value='至少 '+value
            cells.append(value)
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','## 任务类型','', '| 类型 | 原有题 | 新增题 | 合计 |','| --- | ---: | ---: | ---: |']
    for category in sorted(profiles['all']['categories']):
        lines.append('| '+category+' | '+' | '.join(str(profiles[x]['categories'].get(category,0)) for x in ['original','new','all'])+' |')
    lines+=['','## 商品分布','', '| 商品标准名 | 原有题 | 新增题 | 合计 |','| --- | ---: | ---: | ---: |']
    for name in sorted(profiles['all']['product_question_counts']):
        lines.append('| '+name+' | '+' | '.join(str(profiles[x]['product_question_counts'].get(name,0)) for x in ['original','new','all'])+' |')
    lines+=['','## 数据如何产生','',
            '- 保持原有语料与25道已审核题不变，从全部28份手册的分散片段出题；按参数/表格、操作、约束或多方面、故障或指代分配任务，再增加同类商品对比。',
            '- 查询生成看文档，不看检索排名、系统答案或策略分数。新题包含正式、口语和上下文表达，并非只给旧问题换同义词。',
            '- 种子引文必须逐字存在于chunk，并能在原始PDF文本中按明确规范化规则定位；不匹配的出题任务只做来源修复，不按检索效果挑题。',
            '- 另一轮模型调用对目标商品全部chunks逐条标注相关性，每个正相关判断都要给原文引文；再次核验PDF。原文无法核验的正相关保留未知，不能自动改成负例。',
            '- 独立参考审核从这些原文事实提取最小参考要点；要求可完整回答、多商品证据齐全、目标文档标注覆盖至少90%，否则保持待审。',
            '- 其余商品的文档只在明确的商品手册任务范围下记为不相关；因此分数解释依赖商品范围。',
            '- 新题全部放在dev，并保守共享分组，不拆成随机训练/测试集；多轮题另存审核后的standalone retrieval_query。',
            '- 旧题没有重新补表格种子标签，不能把其未记录数当成不含表格；本轮未增加无答案题，仍保留原有5题。',
            f"- 版本还保留原有{summary['legacy_pending']}道未审核题；它们与新增待审题均不进入有效评测，所以工作台总题数会高于本报告的有效题数。",
            f"- 合成问题精确重复数：{summary['exact_duplicate_questions']}；可回答题中完全相同的参考引文集合复用数：{summary['identical_reference_anchor_sets']}。共享手册与相似事实仍存在相关性，不能当作独立统计样本。",'',
            '## 新题示例','']
    for category in sorted({q['category'] for q in new}):
        q=next(q for q in new if q['category']==category)
        prefix='；'.join(m.get('text','') for m in q.get('history',[]))
        lines.append(f"- {q['query_id']}（{category}）：{('历史：'+prefix+'；本轮：') if prefix else ''}{q['query']}")
    if args.experiment:
        report=json.loads(Path(args.experiment).read_text(encoding='utf-8'))
        if report['dataset_sha256']!=b.fingerprint:raise ValueError('experiment dataset fingerprint mismatch')
        summary['retrieval']={}
        lines+=['','## 召回对照：原有与新增分开报告','',
                '各组使用同一批审核后的商品范围及独立问题，排除实体识别和指代改写错误；这是给定正确商品的条件检索实验，不能视为Agent端到端成绩。',
                '主指标取融合候选的Recall@20、Precision@20，辅助看重排nDCG@10及标注覆盖率；不以最终上下文事实覆盖代替召回指标。','',
                '| 样本范围 | 策略 | 可回答题n | Recall@20 | Precision@20 | 重排nDCG@10 | 前20标注覆盖 |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for cohort,ids in [('original',{q['query_id'] for q in original}),('new',{q['query_id'] for q in new}),('all',{q['query_id'] for q in approved})]:
            summary['retrieval'][cohort]={}
            for variant in report['summary']:
                rows=[r for r in report['results'] if r['query_id'] in ids and r['variant']==variant]
                candidate=average_metrics([r['metrics']['candidates'] for r in rows]);ranking=average_metrics([r['metrics']['rerank_order'] for r in rows])
                summary['retrieval'][cohort][variant]={'candidates':candidate,'ranking':ranking,'total':len(rows)}
                n=candidate['recall@20']['n']
                lines.append(f"| {cohort} | {variant} | {n} | {candidate['recall@20']['mean']:.2%} | "
                    f"{candidate['precision@20']['mean']:.2%} | {ranking['ndcg@10']['mean']:.4f} | {candidate['judged_fraction@20']['mean']:.2%} |")
        lines+=['','题目结构与标注方式发生变化，不能把新集分数与旧集分数的差值当作系统提升。未标注结果按零收益计，需同时阅读标注覆盖；无答案题不计入Recall/Precision的平均值。',
                'Precision@20固定以20为分母，返回不足20条时空位计零；部分短手册的候选不足20，不能把这种固定K效应误判为噪声增加。']
        all_metrics=summary['retrieval']['all']
        no_hyde=all_metrics['main_subquery']['candidates']['recall@20']['mean']-all_metrics['main']['candidates']['recall@20']['mean']
        with_hyde=all_metrics['main_hyde_subquery']['candidates']['recall@20']['mean']-all_metrics['main_hyde']['candidates']['recall@20']['mean']
        lines+=['',f"本轮点估计：主查询加入子查询，Recall@20变化{100*no_hyde:+.2f}个百分点；已有HyDE时再加子查询，变化{100*with_hyde:+.2f}个百分点。"
                '这些是相关手册上的合成开发样本，标注也未经人工校准，不能据此宣称稳定的泛化增益。']
    lines+=['','## 复现','', '```powershell',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/expand_source_benchmark.py generate --output-dir eval/local/diverse_build_v1 --repair-source-spans --concurrency 3',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/propose_qrels.py --dataset eval/local/diverse_build_v1/draft --selection eval/local/diverse_build_v1/selection.json --output-dir eval/local/diverse_build_v1/proposals --batch-size 8 --concurrency 4',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/audit_qrel_sources.py --dataset eval/local/diverse_build_v1/draft --selection eval/local/diverse_build_v1/selection.json --sources eval/local/source_pdf_index.jsonl --proposals eval/local/diverse_build_v1/proposals --output-dir eval/local/diverse_build_v1/audit --allow-incomplete --concurrency 4',
            'python scripts/expand_source_benchmark.py publish --output-dir eval/local/diverse_build_v1 --publish-to eval/local/productdoc-diverse-v1',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_subquery_recall.py --dataset eval/local/productdoc-diverse-v1 --reviewed-inputs --output-dir logs/subquery_recall_diverse_v1 --concurrency 3',
            'python scripts/summarize_benchmark_diversity.py --dataset eval/local/productdoc-diverse-v1 --experiment logs/subquery_recall_diverse_v1/retrieval.json --output eval/DIVERSITY_RESULTS.md',
            '```','',f"数据指纹：`{b.fingerprint}`。原始PDF、真实题目、引用、模型缓存和完整报告保留在本地；仓库提供脚本与汇总。"]
    path=Path(args.output);path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    path.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(path),'profiles':profiles,'pending_new':len(pending)},ensure_ascii=False))


if __name__=='__main__':main()
