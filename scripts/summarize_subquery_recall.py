"""Summarize the bounded subquery ablation, including regressions and pending cases."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.evaluation.workflow import collect_cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval',required=True)
    parser.add_argument('--baseline-answers')
    parser.add_argument('--expanded-answers')
    parser.add_argument('--output',required=True)
    parser.add_argument('--case-dir',required=True)
    args=parser.parse_args()
    report=json.loads(Path(args.retrieval).read_text(encoding='utf-8'))
    if report['schema_version']!='productdoc.subquery-experiment.v1':
        raise ValueError('subquery experiment required')
    pct=lambda value:'N/A' if value is None else f'{value:.2%}'
    base={r['query_id']:r for r in report['results'] if r['variant']=='main_hyde'}
    expanded={r['query_id']:r for r in report['results'] if r['variant']=='main_hyde_subquery'}
    observations=[]
    for row in report['results']:
        facts=row['metrics']['context_evidence']
        issues=[]
        if row['answerable'] and facts['all_facts']!=1:
            issues.append({'type':'insufficient_context_candidate','status':'review_required',
                           'missing_evidence_ids':facts['missing_evidence_ids']})
        if row['variant']=='main_hyde_subquery' and row['answerable']:
            before=base[row['query_id']]['metrics']['context_evidence']['fact_recall']
            if facts['fact_recall']<before:
                issues.append({'type':'subquery_context_regression','status':'observed',
                               'before':before,'after':facts['fact_recall']})
        observations.append(dict(row,diagnosis=issues,split='dev'))
    plans=[r['planner'] for r in expanded.values()]
    active=sum(bool(p.get('subqueries')) for p in plans)
    same_context=[qid for qid in base if base[qid]['context_trace']['context_sha256']==expanded[qid]['context_trace']['context_sha256']]
    lines=['# 子查询扩展召回实验','',
           '本实验比较主查询、HyDE、子查询及三路组合。新增分支默认关闭，是否保留以配对收益和退化为依据。','',
           '## 条件与限制','',
           '- 25道来源审核开发题：20道可回答，5道无文档答案。标注来自AI来源审核，尚未独立人工校准；本轮不修改标注。',
           '- 所有组读取同一历史Agent实际item_names与rewritten_query，不使用评测金标选择商品或生成子查询。',
           '- 同一套BGE-M3 dense+sparse混合检索、RRF与BGE reranker；每商品最多20条融合候选，全局最多50条。',
           '- 同一item_balanced选择器，最多20块、12000字符。子查询召回的原始候选可以更多，融合后与基线遵守相同上限。',
           '- 最多3条子查询，共享一路RRF权重1；主查询和HyDE各权重1。规划失败或无扩展则保留原路径。',
           '- 固定Web关闭、无记忆。绕过Agent上游路由与充分性门控，不是完整Agent端到端测试。',
           '- 规划和HyDE使用缓存；reranker对各组候选并集统一打分。耗时为共享批处理记录，不能当作线上延迟。',
           f"- 数据指纹：`{report['dataset_sha256']}`；规划模型：`{report['config']['model']}`。",'',
           f'25题中{active}题生成子查询，其余{25-active}题不扩展。','',
           '## 检索与上下文','',
           '| 组合 | 融合候选事实覆盖率 | 上下文事实覆盖率 | 全事实覆盖 | 平均搜索请求数 | 平均上下文字符 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for variant,summary in report['summary'].items():
        m=summary['metrics']; context=m['context_evidence']
        lines.append(f"| {variant} | {pct(m['candidates_evidence']['fact_recall']['mean'])} | "
                     f"{pct(context['fact_recall']['mean'])} | {pct(context['all_facts']['mean'])} | "
                     f"{summary['mean_search_calls']:.2f} | {summary['mean_context_chars']:.0f} |")
    lines+=['','事实覆盖对20道可回答题取宏平均；请求数、字符数对全部25题取平均。请求数按一次商品过滤的混合搜索计，一次内部含dense和sparse两个请求。','',
            f"主查询＋HyDE与三路组合有{len(same_context)}/25题的实际上下文完全相同，只有{25-len(same_context)}题改变了上下文。相同输入下答案生成或裁判的差异不能归因于检索策略。",'',
            '## 配对变化','']
    for comparison in report['paired_comparisons']:
        lines.append(f"- {comparison['baseline']} → {comparison['variant']}：上下文事实覆盖改善 "
                     f"{', '.join(comparison['context_fact_wins']) or '无'}；退化 {', '.join(comparison['context_fact_losses']) or '无'}。")
    primary=report['paired_comparisons'][1]
    before=report['summary']['main_hyde']['metrics']
    after=report['summary']['main_hyde_subquery']['metrics']
    lines+=['','## 本轮判断','',
            f"相对主查询＋HyDE，加入子查询的上下文事实覆盖变化为 "
            f"{100*(after['context_evidence']['fact_recall']['mean']-before['context_evidence']['fact_recall']['mean']):+.2f} 个百分点。"
            + ('本轮未观察到额外事实覆盖收益，保持实验分支默认关闭。' if not primary['context_fact_wins'] else '存在改善题，仍需复核退化与成本。'),
            f"主查询＋HyDE的融合候选事实覆盖为{pct(before['candidates_evidence']['fact_recall']['mean'])}，"
            f"最终上下文为{pct(before['context_evidence']['fact_recall']['mean'])}。后续应先检查排序、冗余和预算选择造成的证据损失。",'']
    for comparison in report['paired_comparisons']:
        for qid in comparison['context_fact_losses']:
            left=next(r for r in report['results'] if r['query_id']==qid and r['variant']==comparison['baseline'])
            right=next(r for r in report['results'] if r['query_id']==qid and r['variant']==comparison['variant'])
            lost=set(left['metrics']['candidates_evidence']['covered_fact_ids'])-set(right['metrics']['candidates_evidence']['covered_fact_ids'])
            lines.append(f"- {qid}：融合候选已丢失的事实ID：{', '.join(sorted(lost)) or '无'}。这是有限候选额度下的融合竞争，需与预算丢弃分开诊断。")
    lines+=['','| 问题 | 实际子查询 | 新增于主查询＋HyDE之外的chunk数 | 原上下文覆盖 | 扩展后覆盖 |',
            '| --- | --- | ---: | ---: | ---: |']
    for qid,row in expanded.items():
        queries=row['planner'].get('subqueries',[])
        if queries:
            query_text='；'.join(f"{q['query']} → 商品下标{q['item_indices']}" for q in queries).replace('|','\\|')
            lines.append(f"| {qid} | {query_text} | {len(row['route_diagnostics']['subquery_new_vs_main_hyde'])} | "
                         f"{pct(base[qid]['metrics']['context_evidence']['fact_recall'])} | {pct(row['metrics']['context_evidence']['fact_recall'])} |")
    if bool(args.baseline_answers)!=bool(args.expanded_answers):
        raise ValueError('provide both answer reports or neither')
    if args.baseline_answers:
        answers=[]
        for label,path,expected_variant in [('主查询＋HyDE',args.baseline_answers,'main_hyde'),
                                            ('主查询＋HyDE＋子查询',args.expanded_answers,'main_hyde_subquery')]:
            answer=json.loads(Path(path).read_text(encoding='utf-8'))
            if (answer['dataset_sha256']!=report['dataset_sha256'] or answer['config']['variant']!=expected_variant
                    or answer['prediction_sha256']!=__import__('hashlib').sha256(Path(args.retrieval).read_bytes()).hexdigest()):
                raise ValueError('answer report source mismatch')
            rows=answer['results'];summary=answer['summary']['item_balanced']
            answers.append({r['query_id']:r for r in rows})
            for row in rows:
                issues=row.get('assessment',{}).get('diagnosis',[])
                if row['status']=='error':
                    issues=issues+[{'type':'execution_error','status':'observed','reason':row['error']}]
                observations.append(dict(row,variant=expected_variant+'_answer',diagnosis=issues,split='dev'))
            if label=='主查询＋HyDE':
                lines+=['','## 受控答案对照','',
                        '| 组合 | 完成/总数 | LLM原判正确 | 平均输入token |',
                        '| --- | ---: | ---: | ---: |']
            score=summary['answer_metrics'].get('correct',{})
            lines.append(f"| {label} | {summary['completed']}/{summary['total']} | "
                         f"{pct(score.get('mean'))} (n={score.get('n',0)}) | {summary['mean_prompt_tokens']:.0f} |")
        wins,losses=[],[]
        for qid,left in answers[0].items():
            right=answers[1][qid]
            if left.get('assessment') and right.get('assessment'):
                a,z=left['assessment']['correct'],right['assessment']['correct']
                if not a and z:wins.append(qid)
                if a and not z:losses.append(qid)
        lines+=['',f"原判改善：{', '.join(wins) or '无'}；原判退化：{', '.join(losses) or '无'}。",
                '答案由同一生成模型、同一提示词生成，每题每组一次；裁判亦为模型，尚未人工校准，不能将原判比例作为稳定正确率或简历最终结论。']
        identical_differences=[qid for qid in same_context if answers[0][qid].get('assessment') and answers[1][qid].get('assessment')
                              and answers[0][qid]['assessment']['correct']!=answers[1][qid]['assessment']['correct']]
        lines+=['',f"其中，上下文完全相同却原判不同的题目：{', '.join(identical_differences) or '无'}。这些差异保留原始记录，按生成/评审波动处理。"]
    cases=collect_cases({'dataset_sha256':report['dataset_sha256'],'config':{'split':'dev','experiment':'subquery.v1'},
                         'results':observations},Path(args.case_dir)/'case_queue.jsonl')
    lines+=['','## 复现','',
            '加载本地原生库时，脚本先初始化Milvus/Arrow与BGE，再初始化聊天SDK，避免当前Windows环境中的DLL加载冲突。',
            '', '```powershell',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_subquery_recall.py --dataset eval/local/productdoc-source-v2 --agent-report logs/eval_full_20261023_rescored.json --output-dir logs/subquery_recall_v1 --concurrency 4',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/subquery_recall_v1/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --variant main_hyde --policies item_balanced --output-dir logs/context_policy_answers_subquery_v1_baseline --concurrency 4',
            'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/subquery_recall_v1/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --variant main_hyde_subquery --policies item_balanced --output-dir logs/context_policy_answers_subquery_v1_expanded --concurrency 4',
            'python scripts/summarize_subquery_recall.py --retrieval logs/subquery_recall_v1/retrieval.json --baseline-answers logs/context_policy_answers_subquery_v1_baseline/experiment.json --expanded-answers logs/context_policy_answers_subquery_v1_expanded/experiment.json --output eval/SUBQUERY_RESULTS.md --case-dir logs/source_diagnostic_subquery_v1',
            '```','',
            '将API_ENV_PATH替换为授权API配置文件路径。缓存保留同一输入的规划、HyDE和检索结果；答案评审失败重试复用已生成答案。',
            '', '## 在线试用','',
            '在启动查询服务前设置以下环境变量，或在查询状态传入subquery_retrieval_enabled=True：','',
            '```powershell',"$env:CONTEXT_SELECTION_POLICY = 'item_balanced'","$env:SUBQUERY_RETRIEVAL_ENABLED = 'True'",'```','',
            '关闭SUBQUERY_RETRIEVAL_ENABLED即可回到既有路径。Trace保存规划子查询、商品下标、搜索次数、命中数量和失败；chunk保留retrieval_sources。']
    Path(args.output).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':args.output,'cases':cases},ensure_ascii=False))


if __name__=='__main__':
    main()
