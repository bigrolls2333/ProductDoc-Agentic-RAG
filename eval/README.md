# 检索、答案与审核评测指南

本指南面向运行实验和维护标注的开发者。检索评测使用冻结语料与相关性标签；答案评测使用实际入模证据；完整 Agent 的路由检查单独报告。

## 离线运行公开样例

在项目根目录、已安装依赖的 Python 环境中执行。样例为虚构商品数据，不访问 API 或数据库。

```powershell
python scripts/retrieval_eval.py validate --dataset eval/examples/retrieval
python scripts/retrieval_eval.py run --dataset eval/examples/retrieval --variants bm25 --candidate-k 3 --context-k 1 --output logs/retrieval_example.json
python scripts/retrieval_eval.py pool --dataset eval/examples/retrieval --reports logs/retrieval_example.json --output logs/annotation_pool.jsonl
python scripts/retrieval_eval.py collect --report logs/retrieval_example.json --output logs/case_queue.jsonl
python scripts/eval_answer_trust.py --dataset eval/examples/retrieval --answers eval/examples/answers.jsonl --judgments eval/examples/judgments.jsonl --output logs/answer_example.json
```

校验返回2道批准题、3个chunks、2条证据和5条qrels。检索命令输出JSON及Markdown，答案命令重放保存的裁判结果。样例分数用于演示流程，不表示真实文档的检索效果。

## 数据与指标

每个数据版本包含 `manifest.json`、`corpus.jsonl`、`queries.jsonl`、`evidence.jsonl` 和 `qrels.jsonl`。相关性等级为0（无关）、1（有帮助）、2（直接证据）；未标注候选保持未知，并单独报告标注覆盖率。

| 评测对象 | 指标或输出 | 入口 |
| --- | --- | --- |
| 候选召回与排序 | Recall@K、Precision@K、MRR、nDCG@K、judged_fraction@K | `scripts/retrieval_eval.py` |
| 预算内证据选择 | 参考事实覆盖、完整覆盖、上下文字符成本 | `scripts/sweep_context_budget.py` |
| 生成答案 | 事实支持、矛盾、引用precision/completeness、误拒答与误回答 | `scripts/eval_answer_trust.py` |
| 完整Agent行为 | 查询路由、商品确认、工具计划、自评与重试 | `scripts/eval_comprehensive.py` |
| 失败定位 | 过滤、候选遗漏、排序选择、预算丢弃及执行错误 | `app/evaluation/diagnosis.py` |

无答案题不进入检索Recall/Precision分母，单独评测拒答。事实覆盖用于观察回答所需证据是否齐全，与chunk级检索指标分别报告。完整Agent的关键词命中和规则通过率不能替代qrels指标。

字段约束、分组隔离、指纹校验及指标定义见[标准评测参考](BENCHMARK.md)。

## 对比检索与扩展策略

固定数据版本、商品范围、候选上限和上下文预算后，可比较 `bm25,dense,sparse,hybrid,hybrid_rerank`。神经检索需要本地模型与Milvus；样例中的BM25无需这些服务。

`scripts/eval_subquery_recall.py` 比较原查询、原查询+HyDE、原查询+子查询及三路组合。子查询复用已有规划调用，最多3条，绑定确认商品并共享一路RRF权重。运行参数与分组结果见[开发集召回报告](DIVERSITY_RESULTS.md)。

- `--reviewed-inputs` 使用经审核的独立问题与商品范围，只评价检索模块。
- `--agent-report` 读取保存的Agent实际改写与商品识别结果，必须核对问题一致。
- 两种输入口径互斥，不能直接混用分数。扩展分支默认关闭。

上下文默认策略为 `ranked`，可选择 `budgeted` 或复用上游 `item_names` 的 `item_balanced`。配置与预算边界见[工作台指南](WORKBENCH.md#使用可选上下文策略)。

## 来源标注与审核回流

来源流程依次执行：文档约束出题、模型相关性提议、chunk与PDF引文核验、参考事实审核、数据新版本发布。问题生成不读取系统答案或检索排名；未通过来源核验的标注保留待审，不自动转成负例。

工作台提供原文证据、失败队列、标注修订、数据发布和持久化实验任务：

```powershell
python -m uvicorn app.evaluation.api:app --host 127.0.0.1 --port 8090
```

打开 <http://127.0.0.1:8090/>，无本地数据时选择 `example`。审核数据库保存在 `eval/local/workbench.sqlite3`，实验产物保存在 `logs/`。操作步骤见[审核工作台](WORKBENCH.md)。

每次修订保留审核来源；`approved` 代表符合版本准入条件，人工核验由显式字段单独记录。失败观察经复核后才进入新版本，冻结测试题禁止通过开发回流改写。

## 当前结果与复现边界

[开发集召回报告](DIVERSITY_RESULTS.md)包含28份文档、2216个chunks和130道AI来源审核开发题，完成四组召回对照。另有21道新增题待审，人工核验数为0。报告分别列出原有与新增问题，提供数据指纹、候选预算、标注覆盖和复现命令。

[商品分组选择实验](ITEM_BALANCED_RESULTS.md)和[子查询受控实验](SUBQUERY_RESULTS.md)说明特定输入条件下的证据选择与答案对照。不同数据版本、输入条件或标注口径之间的差值不作为策略提升。

真实PDF、评测集和模型缓存不公开；仓库提供协议、代码、虚构样例与结果摘要。需要在线裁判时可使用 `scripts/with_eval_env.py --env-file API_ENV_PATH ...` 注入自己的授权配置，包装器不修改项目 `.env`。
