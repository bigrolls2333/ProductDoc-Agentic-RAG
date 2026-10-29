"""Report selector.v2 measurements without inheriting retired domain-rule scores."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.evaluation.workflow import collect_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case-dir', required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.experiment).read_text(encoding='utf-8'))
    config = report['config']
    if config.get('selector_version') != 'productdoc.context-selector.v2':
        raise ValueError('selector.v2 report required')
    def pct(value):
        return f'{value:.2%}' if value is not None else 'N/A'
    observations = []
    for row in report['results']:
        issues = list(row.get('assessment', {}).get('diagnosis', []))
        if row['status'] == 'error':
            issues.append({'type':'execution_error', 'status':'observed', 'reason':row['error']})
        if row['policy'] == 'item_balanced' and row['context_trace']['items_with_no_candidates']:
            issues.append({'type':'confirmed_item_without_candidates', 'status':'observed',
                           'item_names':row['context_trace']['items_with_no_candidates']})
        observations.append(dict(row, variant=f"v2_{row['policy']}", diagnosis=issues, split='dev'))
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    queued = collect_cases({'dataset_sha256':report['dataset_sha256'], 'config':dict(config,split='dev'),
                            'results':observations}, case_dir/'case_queue.jsonl')
    lines = ['# 上游商品名分组与轮询选择：selector.v2', '',
             '当前实现删除了打印关键词、表格章节特判及从query再次抽型号的逻辑。它复用上游item_names，将候选按标准商品名分组，每组取Top-K，按A1、B1、A2、B2逐轮放入预算。', '',
             '## 实现与边界', '',
             f"- 每商品K={config['per_item_k']}；全局候选上限{config['pool_k']}；最多{config['max_docs']}块；上下文预算{config['budget_chars']}字符（包含引用、标题等元数据）。",
             '- 商品名只统一空白与大小写，与canonical_name对应；旧数据可回退item_name，不对问题或文件标题再次模糊抽型号。',
             '- 片段过大时跳过并继续尝试其他商品/片段；某商品无候选则记录缺失，不用别的商品填充它的配额。',
             '- item_names为空时明确回退全局预算顺序；有确认商品时不自动归属缺少商品元数据的Web片段。',
             '- 新模式下，线上普通检索与HyDE对每个商品分别过滤召回，复用一次查询向量；RRF按商品保留候选，避免再次截成全局Top-10。', '',
             '## 实验条件', '',
             f"数据指纹 `{report['dataset_sha256']}`。",
             f"模型 `{config['model']}`，temperature={config['temperature']}，输入记忆为空、Web关闭。",
             '- 使用既有25道AI来源审核开发题（20道可回答、5道无文档答案），没有独立人工校准或冻结盲测。',
             '- 商品名读取历史Agent产物中的实际item_names，每道query核对一致；不使用expected或评测金标中的商品名。',
             '- 固定旧的候选50与reranker排序后重新选择并生成答案。本实验不重跑新的按商品召回，也不运行上游充分性门控，不能代表完整Agent端到端改进。',
             '- v1的80.50%包含已退役的领域表格特判，属于历史结果，不迁移到v2。', '',
             '## 实测结果', '',
             '| 策略 | 完成/总数 | 事实覆盖 | 全事实覆盖 | LLM原判正确 | 误拒答 | 无答案误回答 | 平均输入token |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for policy, summary in report['summary'].items():
        context, answer = summary['context_metrics'], summary['answer_metrics']
        correct = answer.get('correct', {})
        tokens = f"{summary['mean_prompt_tokens']:.0f}" if summary['mean_prompt_tokens'] is not None else 'N/A'
        lines.append(f"| {policy} | {summary['completed']}/{summary['total']} | {pct(context['fact_recall']['mean'])} | "
                     f"{pct(context['all_facts']['mean'])} | {pct(correct.get('mean'))} (n={correct.get('n',0)}) | "
                     f"{pct(summary['false_abstention_rate'])} | {pct(summary['false_answer_rate'])} | {tokens} |")
    lines += ['', '事实覆盖对20道可回答题取宏平均。LLM正确性保留原判，仍需人工复核，不能直接作为简历最终提升数字。', '',
              '## 多商品覆盖与缺失', '',
              '下面只统计上游实际识别了多个商品的问题；覆盖指至少入选一块同商品材料，不代表该材料已包含必要答案。', '',
              '| 问题 | 策略 | 每商品入选数量 | 无候选的商品 |', '| --- | --- | --- | --- |']
    for row in report['results']:
        trace = row['context_trace']
        if len(trace['item_names']) >= 2:
            counts = '；'.join(f'{name}: {count}' for name,count in trace['per_item_selected_count'].items())
            lines.append(f"| {row['query_id']} | {row['policy']} | {counts} | {', '.join(trace['items_with_no_candidates']) or '无'} |")
    lines += ['', '## 答案配对变化', '']
    for pair in report['paired_comparisons']:
        lines.append(f"- {pair['baseline']} → {pair['policy']}：配对{pair['n']}；原判改善 {', '.join(pair['correctness_wins']) or '无'}；原判退化 {', '.join(pair['correctness_losses']) or '无'}。")
    lines += ['', '逐商品Top-K是候选数量上限，不是强制装满；到达总预算、总块数或组内候选上限都会停止。轮询避免固定商品顺序连续占位，但整块长度不同，不能保证每个商品都能放入，更不保证已覆盖全部必要事实。', '',
              '## 复现', '', '```powershell',
              'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/source_diagnostic_v2/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --per-item-k 20 --output-dir logs/context_policy_answers_v2 --concurrency 4',
              'python scripts/summarize_item_balanced.py --experiment logs/context_policy_answers_v2/experiment.json --output eval/ITEM_BALANCED_RESULTS.md --case-dir logs/source_diagnostic_items_v2',
              '```', '', '第一次执行需要授权API环境；完成记录会缓存，失败裁判重试复用生成答案。摘要命令不调用API，失败队列去重。']
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(path),'case_queue':queued},ensure_ascii=False))


if __name__ == '__main__':
    main()
