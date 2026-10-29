"""Write qualified paired results and send answer/judge observations to review."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.evaluation.workflow import collect_cases
from app.evaluation.dataset import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--audit-cases', default='eval/judge_audit_cases.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--case-dir', required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.experiment).read_text(encoding='utf-8'))
    if report.get('config', {}).get('selector_version') != 'productdoc.context-selector.v1':
        raise ValueError('This historical summary is for selector.v1 only; use summarize_item_balanced.py for v2.')
    audits = json.loads(Path(args.audit_cases).read_text(encoding='utf-8'))
    audit_index = {(r['query_id'], r['policy']):r for r in audits['cases']}
    observations, audit_queue = [], []
    for row in report['results']:
        diagnosis = list(row.get('assessment', {}).get('diagnosis', []))
        if row['status'] == 'error':
            diagnosis.append({'type':'execution_error', 'status':'observed', 'reason':row['error']})
        audit = audit_index.get((row['query_id'],row['policy']))
        if audit:
            diagnosis.append({'type':'judge_reference_consistency_review', 'status':'review_required', 'reason':audit['reason']})
            audit_queue.append(dict(audit, reviewer_type=audits['reviewer_type'], human_verified=False,
                                    original_verdict=row.get('assessment',{}).get('correct'),
                                    answer=row.get('answer_record',{}).get('answer')))
        observations.append(dict(row, variant='context_' + row['policy'], diagnosis=diagnosis, split='dev'))
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    queue_result = collect_cases({'dataset_sha256':report['dataset_sha256'], 'config':dict(report['config'],split='dev'),
                                 'results':observations}, case_dir/'case_queue.jsonl')
    write_jsonl(case_dir/'judge_audit_queue.jsonl', audit_queue)
    lines = ['# 上下文选择：受控答案对照', '',
             '> 历史记录：本文对应selector.v1，其打印质量特判已移除。当前实现与结果见[按商品分组实验](ITEM_BALANCED_RESULTS.md)，旧版80.50%不能用于描述新实现。', '',
             '本轮在25道AI来源审核开发题上，对相同的hybrid_rerank候选排序比较三种上下文策略。'
             '每种策略都重新生成25条答案，合计75条；最终75/75完成，执行错误0。', '',
             '## 实验边界', '',
             '- 数据指纹：`' + report['dataset_sha256'] + '`。20道可回答题、5道无文档答案；人工核验数为0。',
             '- 生成模型和裁判均为deepseek-flash，temperature=0，关闭thinking；输入记忆为空、Web关闭。',
             '- 上游候选与排序固定。生成阶段直接接收选出的上下文，绕过商品识别、检索计划与证据充分性门控，因此不能与此前完整Agent的84%直接比较。',
             '- 各策略的上下文上限均为12000字符，计入引用ID、标题、分数和分隔符；实际token数量不同。',
             '- 仅在dev上开展的一次试验，启发式也基于dev失败设计；没有冻结盲测集和人工校准，不能作为独立测试收益。', '',
             '## 三种策略', '',
             '| 策略 | 行为 |', '| --- | --- |',
             '| ranked | 只使用前5条，遇到超预算片段即停止；作为受控基线。 |',
             '| budgeted | 查看前50条，在预算内继续选择整块证据；跳过放不下的片段，最多20块。 |',
             '| balanced | 使用相同前50条和预算；为问题文本匹配出的不同商品预留证据，对打印质量问题优先保留相关图像缺陷表格，再按排序填充。 |', '',
             '商品匹配来自问题与运行时canonical_name/file_title中的型号；表格规则来自查询意图与文档结构。选择函数不接收qrels、参考事实、正确商品标注或标准答案。', '',
             '## 结果', '',
             '事实覆盖的分母为20题；LLM裁判正确性分母为25题。下面的正确性保留裁判原判，尚有待人工复核的口径不一致。', '',
             '| 策略 | 事实覆盖 | 全事实覆盖 | LLM判正确 | 误拒答 | 无答案误回答 | 平均输入token | 平均输出token |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for policy,s in report['summary'].items():
        m=s['answer_metrics'];c=s['context_metrics']
        lines.append(f"| {policy} | {c['fact_recall']['mean']:.2%} | {c['all_facts']['mean']:.2%} | "
                     f"{m['correct']['mean']:.2%} ({round(m['correct']['mean']*m['correct']['n'])}/{m['correct']['n']}) | "
                     f"{s['false_abstention_rate']:.2%} | {s['false_answer_rate']:.2%} | {s['mean_prompt_tokens']:.0f} | {s['mean_completion_tokens']:.0f} |")
    lines += ['', 'balanced的标注事实覆盖由61.83%升到80.50%，提高18.67个百分点；完整覆盖由12/20升到14/20。'
              '平均输入token约为基线的2.14倍。LLM原判正确数由20/25变为22/25，但该数字尚未通过人工校准。', '',
              '## 成功与退化案例', '']
    for pair in report['paired_comparisons']:
        lines.append(f"- {pair['baseline']} → {pair['policy']}：配对完成 {pair['n']}；裁判判改善 {', '.join(pair['correctness_wins']) or '无'}；退化 {', '.join(pair['correctness_losses']) or '无'}。")
    lines += ['', 'VAL-044的图像缺陷表原先在reranker第21名。balanced将相关表格保留到实际prompt中，生成答案覆盖介质、粉盒和进纸通道排查。'
              '与此同时，VAL-043和VAL-149出现回答遗漏或不受支持的扩展，说明覆盖率提高不能保证每题回答更好。', '',
              '## 裁判一致性复核', '',
              '抽查发现下列原判与必要事实要求可能不一致，已保留原分数并进入独立待审队列：', '']
    for item in audits['cases']:
        lines.append(f"- {item['query_id']} / {item['policy']}：{item['reason']}")
    lines += ['', '这些是AI抽查提出的复核候选，尚不是人类裁决；可能需要修正参考事实粒度，也可能需要收紧裁判。'
              '最终简历中的答案提升数字应在人工复核和冻结测试完成后确定。', '',
              '## 工程交付', '',
              '- 选择器已接入答案prompt构造，并记录策略、候选范围、预留原因、实际字符数和上下文哈希。',
              '- 图内保存完整的有界重排候选，允许在Top-K之外选择证据；默认仍使用ranked，balanced作为显式可选策略。',
              '- 修正历史文本预算的累计方式：追加前检查单条新增文本，避免先越界追加、重复累计。',
              '- 工作台支持来源审核、版本发布、离线任务执行、实验对比和失败回流。',
              '- 新实验的答案失败与裁判一致性疑点已进入case_queue；实际答案、上下文、token和缓存全部保留。', '',
              '## 复现', '',
              '```powershell',
              'python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/source_diagnostic_v2/retrieval.json --output-dir logs/context_policy_answers_v1 --concurrency 4',
              'python scripts/summarize_context_answers.py --experiment logs/context_policy_answers_v1/experiment.json --output eval/CONTEXT_SELECTION_RESULTS.md --case-dir logs/source_diagnostic_context_v1',
              '```', '', '第一条命令只调用没有有效缓存的任务。首次运行需要API配置；缓存包含完整原答案，裁判失败后优先复用生成结果。'
              '生成输入不包含参考答案。第二条命令不访问API，可去重回流失败记录。']
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(path),'case_queue':queue_result,'judge_audit_candidates':len(audit_queue)},ensure_ascii=False))


if __name__ == '__main__':
    main()
