# ProductDoc-Agentic-RAG Evaluation Guide

本目录说明 Query Graph 的验证维度、数据格式和评测脚本。验证集属于运行时数据，公开仓库只保留方法说明，实际 JSONL 文件按项目环境单独准备。

## 文件

- `validation_set.jsonl`：本地人工标注样本，每行一条 JSON。该文件默认不进入公开仓库。
- `../scripts/eval_routing.py`：Query Graph 路由评测脚本，当前用于评测 `query_route / doc_need_route / retrieval_plan / item_names`。

## 评测对象

当前验证集按 Query Graph 的真实状态字段设计：

| 层级 | 对应字段 | 评测目标 |
| --- | --- | --- |
| 查询意图 | `expected.query_route` | 判断输入是否属于产品文档问答、直接回复或超范围问题 |
| 文档需求 | `expected.doc_need_route` | 判断产品相关问题是否需要查文档 |
| 检索计划 | `expected.retrieval_plan` | 判断是否启用 embedding、HyDE、Web |
| 商品确认 | `expected.item_names` | 判断别名、型号、多商品是否归一到正确商品 |
| 检索证据 | `expected.source_hint` | 判断 reranked docs 是否命中预期文档和关键词 |
| 最终回答 | `expected.answer_must_include` / `expected.answer_should_not_include` | 判断回答是否覆盖关键事实并避免明显幻觉 |

## 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 样本唯一编号 |
| `category` | string | 样本类别 |
| `query` | string | 用户输入 |
| `history` | array | 历史对话，空数组表示单轮问题 |
| `expected.query_route` | string | 期望查询路由，常用值：`product_doc_qa`、`direct_answer`、`out_of_scope` |
| `expected.doc_need_route` | string | 期望文档需求路由，常用值：`need_retrieval`、`direct_answer`、`not_applicable` |
| `expected.retrieval_plan` | object | 期望检索计划，包含 `use_embedding/use_hyde/use_web/reason` |
| `expected.item_names` | array | 期望确认出的商品名列表 |
| `expected.answer_must_include` | array | 答案必须覆盖的关键词或事实点 |
| `expected.answer_should_not_include` | array | 答案中不应出现的错误扩展 |
| `expected.source_hint` | object | 证据命中提示，包含 `doc_title_contains` 和 `chunk_keywords` |
| `notes` | string | 样本设计意图 |

## 类别枚举

| 类别 | 说明 |
| --- | --- |
| `product_fact` | 产品事实、参数、功能说明 |
| `operation_step` | 安装、配置、连接、初始化、使用步骤 |
| `troubleshooting` | 故障排查、异常现象、处理建议 |
| `image_grounded` | 需要结合图片、图示、拓扑、面板信息 |
| `alias_match` | 简称、型号、别名归一 |
| `multi_item` | 多商品同时提问 |
| `multi_turn` | 多轮上下文、指代、省略商品名 |
| `doc_not_needed` | 产品相关但无需查文档的问题 |
| `smalltalk` | 闲聊、问候、感谢 |
| `abuse_or_invalid` | 纯辱骂、挑衅、无意义输入 |
| `out_of_scope` | 非产品文档知识库范围问题 |
| `web_required` | 官网、价格、固件、最新版本等时效问题 |
| `no_answer` | 文档未覆盖或证据不足的问题 |

## 检索计划口径

- 普通文档事实和步骤问题：`use_embedding=true`、`use_hyde=false`、`use_web=false`、`reason=precise_doc_query`。
- 模糊故障、短问题、现象型问题：`use_embedding=true`、`use_hyde=true`、`use_web=false`、`reason=ambiguous_or_short_doc_query`。
- 官网、价格、购买、固件、最新版本、保修等时效问题：`use_embedding=true`、`use_hyde=false`、`use_web=true`、`reason=external_or_fresh_info_query`。
- 直接回复、闲聊、辱骂、超范围问题：`use_embedding=false`、`use_hyde=false`、`use_web=false`、`reason=not_applicable`。

## 第一版评测建议

1. 先只跑路由评测：比较 `query_route`、`doc_need_route`、`retrieval_plan`。
2. 再跑商品确认评测：比较 `item_names` 是否包含期望商品。
3. 再跑检索评测：检查 `reranked_docs` 的标题和文本是否命中 `source_hint`。
4. 最后跑答案评测：检查 `answer_must_include` 和 `answer_should_not_include`。

## 脚本用法

先跑只依赖 Query Intent Gate 的样本：

```bash
python scripts/eval_routing.py --intent-only
```

只跑无需导入文档也能稳定评测的类别：

```bash
python scripts/eval_routing.py --categories smalltalk,abuse_or_invalid,out_of_scope,doc_not_needed --intent-only
```

跑完整路由评测：

```bash
python scripts/eval_routing.py
```

只跑 H3C 相关样本或少量样本做冒烟：

```bash
python scripts/eval_routing.py --ids VAL-001,VAL-019,VAL-035,VAL-073
python scripts/eval_routing.py --limit 10
```

如果当前商品库还没有完整导入，可以先跳过 `item_names` 严格比对：

```bash
python scripts/eval_routing.py --skip-item-names
```

## 导入前置条件

- `smalltalk / abuse_or_invalid / out_of_scope / doc_not_needed` 这几类样本，不依赖本地 PDF 全量导入。
- `product_fact / operation_step / troubleshooting / image_grounded / alias_match / multi_item / multi_turn / web_required / no_answer` 这些样本，至少需要对应商品已经导入，尤其是 `item_names` 评测依赖 `ITEM_NAME_COLLECTION` 中已有 alias/canonical 数据。
- 第一轮不需要把所有 PDF 一次性全量导入。更稳的做法是先导入一小组高频文档，例如 `H3C LA2608 / H3C NR-1200W / H3C ER2100 / PixLab B5 / LJ2680DN`，先跑一批子集样本，确认链路稳定后再扩展。

## 来源和置信度展示建议

后续最终回答建议扩展为结构化输出：

```json
{
  "answer": "...",
  "matched_items": [
    {
      "canonical_name": "H3C LA2608室内无线网关",
      "matched_alias": "LA2608",
      "brand": "H3C",
      "category": "室内无线网关",
      "model_code": "LA2608",
      "confidence": "high"
    }
  ],
  "sources": [
    {
      "rank": 1,
      "doc_title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf",
      "chunk_id": "...",
      "score": 0.82,
      "source_type": "local_doc",
      "preview": "...",
      "image_urls": []
    }
  ],
  "confidence": {
    "level": "high",
    "reasons": [
      "商品名称高置信匹配",
      "Top文档来自目标商品手册",
      "多个证据片段支持同一结论"
    ]
  }
}
```

第一版置信度建议使用 `high/medium/low/unknown`，暂不输出未经校准的百分比。

## 消融评测

`../scripts/eval_ablation.py` 用于在同一份验证集上批量运行 baseline 与消融配置，输出每个 variant 的完整 JSON 报告和统一 Markdown 汇总。

推荐先跑核心消融：

```bash
python scripts/eval_ablation.py --variants core --categories multi_turn,web_required,no_answer,troubleshooting --answer-method string
```

常用 variant：

| Variant | 评测目标 |
| --- | --- |
| `baseline` | 当前完整 Agentic RAG 链路 |
| `no_rerank` | 去掉 BGE Rerank，评估重排对证据命中的贡献 |
| `no_hyde` | 去掉 HyDE，评估短查询和故障类问题的召回变化 |
| `no_self_eval` | 去掉 LLM 自评，评估证据不足时的幻觉风险 |
| `no_retry` | 保留自评，去掉自评触发的 Web 补搜，评估 retry recovery |
| `no_model_guard` | 去掉型号级 Web 证据校验，评估泛化网页误用风险 |
| `history_only` | 保留历史对话，去掉 session/product memory，评估结构化记忆收益 |
| `no_context_memory` | 去掉 ContextBundle 与记忆，评估多轮上下文收益 |
| `rule_planner` | 使用规则 planner，评估 LLM planner 的工具选择收益 |
| `embedding_only` | 只开本地 embedding，评估 planner 与多路召回收益 |

输出目录示例：

```text
logs/eval_ablation/run_YYYYMMDD_HHMMSS/
  baseline.json
  no_rerank.json
  summary.json
  summary.md
```

`summary.md` 可直接提取简历指标，重点比较 baseline 与对应消融项的通过率、检索命中率、答案通过率、自评准确率和 P50/P95 延迟。

## Trace 指标汇总

`../scripts/analyze_trace_metrics.py` 用于汇总 `eval_comprehensive.py` 输出中的 `trace_summary` 字段：

```bash
python scripts/analyze_trace_metrics.py --input logs/eval_ablation/run_YYYYMMDD_HHMMSS/*.json --output logs/eval_ablation/run_YYYYMMDD_HHMMSS/trace_metrics.json
```

汇总字段包括 planner 分布、doc_need_route 分布、self_eval_status 分布、平均召回数量、平均 retry 次数、P50/P95 延迟和 checkpoint thread 覆盖率。

## Chunk 切分消融

`../scripts/eval_chunk_ablation.py` 用于按不同 chunk 策略重建临时 Milvus collection，再运行同一批验证样本。默认只生成 dry-run manifest，不会写入 Milvus：

```bash
python scripts/eval_chunk_ablation.py --docs doc/a.md,doc/b.pdf --strategies all --ids VAL-001,VAL-002
```

确认文档路径和临时 collection 名称后再执行导入与评测：

```bash
python scripts/eval_chunk_ablation.py --docs doc/a.md,doc/b.pdf --strategies fixed_300_overlap_50,fixed_600_overlap_100,semantic_title_2000 --run-import --run-eval --ids VAL-001,VAL-002
```

内置策略：

| Strategy | 说明 |
| --- | --- |
| `fixed_300_overlap_50` | 固定窗口 300 字，50 overlap |
| `fixed_600_overlap_100` | 固定窗口 600 字，100 overlap |
| `fixed_1000_overlap_150` | 固定窗口 1000 字，150 overlap |
| `semantic_title_1000` | 标题语义切分，超长段 1000 字 |
| `semantic_title_2000` | 当前默认标题语义切分，超长段 2000 字 |
| `semantic_title_no_merge` | 标题语义切分，关闭短 chunk 合并 |

chunk 消融建议只选 3 到 5 份代表性文档和 20 到 30 条样本，重点记录 Recall@K、Top1 命中、答案通过率、chunk 数量、导入耗时和存储量。
