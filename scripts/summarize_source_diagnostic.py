"""Consolidate source-reviewed diagnostics and enqueue observed development failures."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.evaluation.dataset import load_benchmark, write_jsonl
from app.evaluation.workflow import collect_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'retrieval', 'sweep', 'answers', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    benchmark = load_benchmark(args.dataset)
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    retrieval, sweep, answers = read(args.retrieval), read(args.sweep), read(args.answers)
    if any(r['dataset_sha256'] != benchmark.fingerprint for r in (retrieval, sweep, answers)):
        raise ValueError('diagnostic artifacts must use the same dataset version')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    queue = output / 'case_queue.jsonl'
    collect_cases(retrieval, queue)
    collect_cases(answers, queue)
    selected = benchmark.select('dev')
    summary = {'dataset_sha256': benchmark.fingerprint, 'version': benchmark.manifest['version'],
               'approved': len(selected), 'answerable': sum(q['answerable'] for q in selected),
               'human_verified_queries': sum(q.get('review', {}).get('human_verified') is True for q in selected),
               'evidence': len(benchmark.evidence), 'qrels': len(benchmark.judgments),
               'positive_qrels': sum(j['relevance'] > 0 for j in benchmark.judgments.values()),
               'unjudged_pairs': len(selected) * len(benchmark.corpus) - len(benchmark.judgments),
               'reference_facts': sum(len(q['reference_facts']) for q in selected),
               'answer_summary': answers['summary'], 'annotation_quality': benchmark.manifest['annotation_quality'],
               'artifact_sha256': {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                                   for key, value in [('retrieval', args.retrieval), ('sweep', args.sweep), ('answers', args.answers)]}}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = {(r['query_id'], r['variant']): r for r in retrieval['results']}
    paired = []
    for query in selected:
        if not query['answerable']:
            continue
        qid = query['query_id']
        before, after = rows[qid, 'hybrid'], rows[qid, 'hybrid_rerank']
        if before['status'] != 'completed' or after['status'] != 'completed':
            continue
        delta = after['metrics']['context_evidence']['fact_recall'] - before['metrics']['context_evidence']['fact_recall']
        if delta:
            anchors = []
            for fact in query['reference_facts']:
                support = []
                for eid in fact['evidence_ids']:
                    for (j_qid, cid), judgment in benchmark.judgments.items():
                        if j_qid == qid and eid in judgment.get('evidence_ids', []):
                            rank = next((i for i, d in enumerate(after['rerank_order'], 1) if d['chunk_id'] == cid), None)
                            support.append({'evidence_id': eid, 'chunk_id': cid, 'reranked_rank': rank})
                anchors.append({'fact_id': fact['fact_id'], 'support': support})
            paired.append({'query_id': qid, 'query': query['query'], 'delta_fact_recall': delta,
                           'before': before['metrics']['context_evidence'], 'after': after['metrics']['context_evidence'],
                           'anchors': anchors, 'status': 'observed_fixed_candidate_ranking_difference',
                           'candidate_ids_equal': [d['chunk_id'] for d in before['stages']['candidates']] ==
                                                  [d['chunk_id'] for d in after['stages']['candidates']]})
    paired.sort(key=lambda r: r['delta_fact_recall'])
    write_jsonl(output / 'reranker_case_review.jsonl', paired)
    lines = ['# 商品文档检索与评测：来源核验诊断结果', '',
             f"数据版本 `{summary['version']}`；SHA-256 `{benchmark.fingerprint}`。", '',
             f"本轮发布 {len(selected)} 道诊断开发题：{summary['answerable']} 道可回答、{len(selected)-summary['answerable']} 道无文档答案，"
             f"{summary['reference_facts']} 个参考事实、{summary['evidence']} 个来源锚点、{summary['qrels']} 条相关性判断。"
             f"人工核验为 {summary['human_verified_queries']}；数据由 AI 提议、原 PDF 引文定位与显式 AI 来源审核形成。", '',
             '这是刻意选择的诊断集，未做盲测，也未做人类裁判校准。所有问题属于 dev；不得将指标写成全量人工金标测试成绩。', '',
             '原始候选与排序复用已完成的 271 题全量检索；本次标准计分限定批准的 25 题。冻结语料为 28 份文档、2216 个 chunks。', '',
             '## 检索消融', '',
             '统一条件：全库检索、候选50、上下文Top-5、content预算12000字符。IR均值只计算20道可回答题；标注覆盖率计算25题。', '',
             '| 方案 | Recall@50 | MRR@5 | nDCG@5 | 上下文事实覆盖 | 全事实覆盖 | Judged@5 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for variant, item in retrieval['summary'].items():
        m = item['metrics']
        values = [m['candidates']['recall@50']['mean'], m['rerank_order']['mrr@5']['mean'],
                  m['rerank_order']['ndcg@5']['mean'], m['context_evidence']['fact_recall']['mean'],
                  m['context_evidence']['all_facts']['mean'], m['rerank_order']['judged_fraction@5']['mean']]
        lines.append('| ' + variant + ' | ' + ' | '.join(f'{value:.2%}' for value in values) + ' |')
    lines += ['', '事实覆盖按参考事实的 OR/AND 证据组计算：同义原文任一命中即可，联合事实必须证据齐全。'
              '它仍是已标注证据覆盖，不等于语义裁判证明全部事实可推导。严格逐span召回另保留在JSON中。', '',
              f"未判断组合有 {summary['unjudged_pairs']} 对，保持 unknown；其他产品文档依照“指定商品权威手册”任务范围判为不相关。"
              '正相关qrels仍有AI相关性判断误差；不能以同商品命中替代相关性。', '',
              '## 固定排序的上下文消融', '',
              '以下只重放固定排序和整块字符预算，未重新调用模型、未生成新答案。分母均为20道可回答题。', '',
              '| 方案 | Top-K | 字符预算 | 事实覆盖 | 全事实覆盖 | 实际平均content字符 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in sweep['summary']:
        if item['variant'] not in {'hybrid', 'hybrid_rerank'}:
            continue
        if item['budget'] != 12000 and (item['context_k'], item['budget']) != (20, 24000):
            continue
        lines.append(f"| {item['variant']} | {item['context_k']} | {item['budget']} | {item['metrics']['fact_recall']['mean']:.2%} | "
                     f"{item['metrics']['all_facts']['mean']:.2%} | {item['mean_context_chars']:.0f} |")
    lines += ['', 'Top-K增大带来的覆盖改善不能直接当作答案正确率改善；需要随后固定模型、记忆和提示词生成答案验证。'
              '此消融用于决定下一轮工程投入，不能据此直接修改线上默认参数。', '', '## 答案可信度', '',
              '使用历史Agent真实prompt上下文；提前拒答标记no_generation，未伪造空prompt。'
              '裁判为deepseek-flash，判断逐条引用必须落在实际上下文；规范化仅允许可唯一对齐的空白/NFKC差异。'
              '23条首轮成功，2条引用校验失败后定向重试成功，最终25/25完成、0执行错误。', '']
    a = answers['summary']
    lines += [f"- 参考答案正确性：{a['metrics']['correct']['mean']:.2%}（n={a['metrics']['correct']['n']}）。",
              f"- 宏平均事实支持率：{a['metrics']['faithfulness']['mean']:.2%}（n={a['metrics']['faithfulness']['n']}，仅包含有事实声明的答案）。",
              f"- 引用精确率：{a['metrics']['citation_precision']['mean']:.2%}；引用完整率：{a['metrics']['citation_completeness']['mean']:.2%}（均n=19）。",
              f"- 可回答题误拒答：{a['false_abstention_rate']:.2%}（3/{a['answerable_n']}）；无答案题误回答：{a['false_answer_rate']:.2%}（0/{a['unanswerable_n']}）。", '',
              '生成模型和裁判来自同一服务，同源偏好与长答案拆分偏差尚未校准。高事实支持率只说明回答多数声明有上下文支持，不能掩盖误拒答或必要事实遗漏。', '',
              '## 可复核的失败与数据回流', '',
              'case_queue.jsonl保存检索与答案失败观察，去重且禁止测试集回流；reranker_case_review.jsonl保存相同候选下的前后覆盖及证据新排名。', '']
    for item in paired:
        lines.append(f"- {item['query_id']}：{item['query']} hybrid→rerank事实覆盖变化 {item['delta_fact_recall']:+.2%}；"
                     f"相同候选={item['candidate_ids_equal']}。")
    lines += ['', '下一轮优先处理：', '',
              '1. 对打印质量表格、跨商品问题检查reranker排序及上下文选择；先在开发集做表格结构保留和商品间证据配额实验。',
              '2. 对候选50仍缺事实的题补做查询分解/改写消融，不能靠增加context-k解决未召回事实。',
              '3. 审核提前拒答门控：已有语料可回答但被挡在生成前的3例应单独建回归样例。',
              '4. 扩充人工独立审核和新的文档/问题组后冻结test，再验证开发集选出的改动。', '',
              '本轮数据飞轮已具备候选池、来源核验、数据版本发布、标准计分、失败队列与复评入口；尚未证明某项线上修复带来独立测试集提升。', '',
              '## 复现', '', '固定语料版本与模型预测后，运行：', '', '```powershell',
              f'python scripts/replay_retrieval_eval.py --dataset {args.dataset} --source-dataset eval/local/productdoc-v1 --predictions logs/productdoc_bm25_pool_20261023.json logs/productdoc_neural_pool_20261023.json --output {args.retrieval}',
              f'python scripts/sweep_context_budget.py --dataset {args.dataset} --predictions {args.retrieval} --output {args.sweep}',
              f'python scripts/eval_answer_trust.py --dataset {args.dataset} --answers logs/source_diagnostic_v2/answers.jsonl --judgments logs/source_diagnostic_v2/saved_judgments.jsonl --output {args.answers}',
              '```', '', '以上重放不访问API或数据库。原始PDF、本地索引、API配置和运行日志均未加入Git。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'approved': len(selected), 'retrieval_observations': len(retrieval['results']),
                      'context_sweep_observations': len(sweep['results']), 'answer_completed': a['completed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
