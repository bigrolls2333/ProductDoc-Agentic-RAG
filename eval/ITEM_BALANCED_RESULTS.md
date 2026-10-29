# 上游商品名分组与轮询选择：selector.v2

当前实现删除了打印关键词、表格章节特判及从query再次抽型号的逻辑。它复用上游item_names，将候选按标准商品名分组，每组取Top-K，按A1、B1、A2、B2逐轮放入预算。

## 实现与边界

- 每商品K=20；全局候选上限50；最多20块；上下文预算12000字符（包含引用、标题等元数据）。
- 商品名只统一空白与大小写，与canonical_name对应；旧数据可回退item_name，不对问题或文件标题再次模糊抽型号。
- 片段过大时跳过并继续尝试其他商品/片段；某商品无候选则记录缺失，不用别的商品填充它的配额。
- item_names为空时明确回退全局预算顺序；有确认商品时不自动归属缺少商品元数据的Web片段。
- 新模式下，线上普通检索与HyDE对每个商品分别过滤召回，复用一次查询向量；RRF按商品保留候选，避免再次截成全局Top-10。

## 实验条件

数据指纹 `ac9cac76feaf1ec7249906fe0c14424d016ea81edb6a4fb784c9e7b1627740cc`。
模型 `deepseek-flash`，temperature=0，输入记忆为空、Web关闭。
- 使用既有25道AI来源审核开发题（20道可回答、5道无文档答案），没有独立人工校准或冻结盲测。
- 商品名读取历史Agent产物中的实际item_names，每道query核对一致；不使用expected或评测金标中的商品名。
- 固定旧的候选50与reranker排序后重新选择并生成答案。本实验不重跑新的按商品召回，也不运行上游充分性门控，不能代表完整Agent端到端改进。

## 实测结果

| 策略 | 完成/总数 | 事实覆盖 | 全事实覆盖 | LLM原判正确 | 误拒答 | 无答案误回答 | 平均输入token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ranked | 25/25 | 61.83% | 60.00% | 76.00% (n=25) | 5.00% | 0.00% | 3051 |
| budgeted | 25/25 | 75.50% | 65.00% | 80.00% (n=25) | 0.00% | 0.00% | 6541 |
| item_balanced | 25/25 | 75.50% | 65.00% | 84.00% (n=25) | 0.00% | 0.00% | 4987 |

事实覆盖对20道可回答题取宏平均。LLM正确性保留原判，仍需人工复核，不能直接作为简历最终提升数字。

## 多商品覆盖与缺失

下面只统计上游实际识别了多个商品的问题；覆盖指至少入选一块同商品材料，不代表该材料已包含必要答案。

| 问题 | 策略 | 每商品入选数量 | 无候选的商品 |
| --- | --- | --- | --- |
| VAL-064 | ranked | H3CNR-1200W企业级无线宽带路由器: 3；H3CLA2608室内无线网关: 2 | 无 |
| VAL-064 | budgeted | H3CNR-1200W企业级无线宽带路由器: 9；H3CLA2608室内无线网关: 2 | 无 |
| VAL-064 | item_balanced | H3CNR-1200W企业级无线宽带路由器: 7；H3CLA2608室内无线网关: 4 | 无 |
| VAL-149 | ranked | PANTUMP3030黑白激光打印机: 2；HUAWEIPIXLABB5打印机: 3 | 无 |
| VAL-149 | budgeted | PANTUMP3030黑白激光打印机: 5；HUAWEIPIXLABB5打印机: 4 | 无 |
| VAL-149 | item_balanced | PANTUMP3030黑白激光打印机: 5；HUAWEIPIXLABB5打印机: 4 | 无 |
| VAL-151 | ranked | H3CLA2608室内无线网关: 1；H3CER2100企业级路由器: 4 | 无 |
| VAL-151 | budgeted | H3CLA2608室内无线网关: 3；H3CER2100企业级路由器: 10 | 无 |
| VAL-151 | item_balanced | H3CLA2608室内无线网关: 4；H3CER2100企业级路由器: 8 | 无 |
| VAL-155 | ranked | HUAWEIPIXLABB5打印机: 5；H3CLA2608室内无线网关: 0 | H3CLA2608室内无线网关 |
| VAL-155 | budgeted | HUAWEIPIXLABB5打印机: 9；H3CLA2608室内无线网关: 0 | H3CLA2608室内无线网关 |
| VAL-155 | item_balanced | HUAWEIPIXLABB5打印机: 9；H3CLA2608室内无线网关: 0 | H3CLA2608室内无线网关 |

## 答案配对变化

- ranked → budgeted：配对25；原判改善 VAL-037, VAL-038, VAL-041；原判退化 VAL-035, VAL-036。
- ranked → item_balanced：配对25；原判改善 VAL-037, VAL-038, VAL-041；原判退化 VAL-149。

逐商品Top-K是候选数量上限，不是强制装满；到达总预算、总块数或组内候选上限都会停止。轮询避免固定商品顺序连续占位，但整块长度不同，不能保证每个商品都能放入，更不保证已覆盖全部必要事实。

## 复现

```powershell
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/source_diagnostic_v2/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --per-item-k 20 --output-dir logs/context_policy_answers_v2 --concurrency 4
python scripts/summarize_item_balanced.py --experiment logs/context_policy_answers_v2/experiment.json --output eval/ITEM_BALANCED_RESULTS.md --case-dir logs/source_diagnostic_items_v2
```

第一次执行需要授权API环境；完成记录会缓存，失败裁判重试复用生成答案。摘要命令不调用API，失败队列去重。
