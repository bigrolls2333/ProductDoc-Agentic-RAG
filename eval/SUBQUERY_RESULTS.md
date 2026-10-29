# 子查询扩展召回实验

本实验比较主查询、HyDE、子查询及三路组合。新增分支默认关闭，是否保留以配对收益和退化为依据。

## 条件与限制

- 25道来源审核开发题：20道可回答，5道无文档答案。标注来自AI来源审核，尚未独立人工校准；本轮不修改标注。
- 所有组读取同一历史Agent实际item_names与rewritten_query，不使用评测金标选择商品或生成子查询。
- 同一套BGE-M3 dense+sparse混合检索、RRF与BGE reranker；每商品最多20条融合候选，全局最多50条。
- 同一item_balanced选择器，最多20块、12000字符。子查询召回的原始候选可以更多，融合后与基线遵守相同上限。
- 最多3条子查询，共享一路RRF权重1；主查询和HyDE各权重1。规划失败或无扩展则保留原路径。
- 固定Web关闭、无记忆。绕过Agent上游路由与充分性门控，不是完整Agent端到端测试。
- 规划和HyDE使用缓存；reranker对各组候选并集统一打分。耗时为共享批处理记录，不能当作线上延迟。
- 数据指纹：`ac9cac76feaf1ec7249906fe0c14424d016ea81edb6a4fb784c9e7b1627740cc`；规划模型：`deepseek-flash`。

25题中12题生成子查询，其余13题不扩展。

## 检索与上下文

| 组合 | 融合候选事实覆盖率 | 上下文事实覆盖率 | 全事实覆盖 | 平均搜索请求数 | 平均上下文字符 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 92.17% | 79.67% | 65.00% | 1.16 | 9582 |
| main_hyde | 93.17% | 78.17% | 65.00% | 2.32 | 9578 |
| main_subquery | 89.67% | 78.83% | 65.00% | 1.92 | 9595 |
| main_hyde_subquery | 93.17% | 78.17% | 65.00% | 3.08 | 9578 |

事实覆盖对20道可回答题取宏平均；请求数、字符数对全部25题取平均。请求数按一次商品过滤的混合搜索计，一次内部含dense和sparse两个请求。

主查询＋HyDE与三路组合有21/25题的实际上下文完全相同，只有4题改变了上下文。相同输入下答案生成或裁判的差异不能归因于检索策略。

## 配对变化

- main → main_subquery：上下文事实覆盖改善 无；退化 VAL-038。
- main_hyde → main_hyde_subquery：上下文事实覆盖改善 无；退化 无。

## 本轮判断

相对主查询＋HyDE，加入子查询的上下文事实覆盖变化为 +0.00 个百分点。本轮未观察到额外事实覆盖收益，保持实验分支默认关闭。
主查询＋HyDE的融合候选事实覆盖为93.17%，最终上下文为78.17%。后续应先检查排序、冗余和预算选择造成的证据损失。

- VAL-038：融合候选已丢失的事实ID：VAL-038:f4。这是有限候选额度下的融合竞争，需与预算丢弃分开诊断。

| 问题 | 实际子查询 | 新增于主查询＋HyDE之外的chunk数 | 原上下文覆盖 | 扩展后覆盖 |
| --- | --- | ---: | ---: | ---: |
| VAL-020 | 如何为Host提供WiFi接入 → 商品下标[0] | 0 | 100.00% | 100.00% |
| VAL-035 | 室内无线网关无法上网的故障排查方法 → 商品下标[0]；室内无线网关联网配置与连接设置 → 商品下标[0] | 0 | 100.00% | 100.00% |
| VAL-036 | 无线网关没有WiFi信号的排查步骤 → 商品下标[0]；WiFi信号异常或无信号的故障处理方法 → 商品下标[0] | 0 | 100.00% | 100.00% |
| VAL-037 | 无线网络不稳定的排查步骤 → 商品下标[0]；无线信号干扰与信道配置检查 → 商品下标[0] | 10 | 0.00% | 0.00% |
| VAL-038 | 无法登录Web管理页面的排查与解决方法 → 商品下标[0] | 5 | 33.33% | 33.33% |
| VAL-041 | 打印机无法打印的故障排查方法 → 商品下标[0] | 9 | 66.67% | 66.67% |
| VAL-043 | M7208W Pro 无线网络连接失败排查步骤 → 商品下标[0] | 2 | 80.00% | 80.00% |
| VAL-064 | NR-1200W 是什么设备，产品定位与类型 → 商品下标[0]；LA2608 是什么设备，产品定位与类型 → 商品下标[1] | 6 | 100.00% | 100.00% |
| VAL-119 | 打印机卡纸的处理步骤 → 商品下标[0] | 5 | 100.00% | 100.00% |
| VAL-149 | PANTUMP3030 设备类型 → 商品下标[0]；HUAWEIPIXLABB5 设备类型 → 商品下标[1] | 8 | 0.00% | 0.00% |
| VAL-151 | LA2608 产品功能定位与主要用途 → 商品下标[0]；ER2100 产品功能定位与主要用途 → 商品下标[1] | 10 | 100.00% | 100.00% |
| VAL-155 | 产品类型与设备分类 → 商品下标[0, 1] | 10 | 50.00% | 50.00% |

## 受控答案对照

| 组合 | 完成/总数 | LLM原判正确 | 平均输入token |
| --- | ---: | ---: | ---: |
| 主查询＋HyDE | 25/25 | 100.00% (n=25) | 5273 |
| 主查询＋HyDE＋子查询 | 25/25 | 92.00% (n=25) | 5264 |

原判改善：无；原判退化：VAL-037, VAL-043。
答案由同一生成模型、同一提示词生成，每题每组一次；裁判亦为模型，尚未人工校准，不能将原判比例作为稳定正确率或简历最终结论。

其中，上下文完全相同却原判不同的题目：VAL-043。这些差异保留原始记录，按生成/评审波动处理。

## 复现

加载本地原生库时，脚本先初始化Milvus/Arrow与BGE，再初始化聊天SDK，避免当前Windows环境中的DLL加载冲突。

```powershell
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_subquery_recall.py --dataset eval/local/productdoc-source-v2 --agent-report logs/eval_full_20261023_rescored.json --output-dir logs/subquery_recall_v1 --concurrency 4
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/subquery_recall_v1/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --variant main_hyde --policies item_balanced --output-dir logs/context_policy_answers_subquery_v1_baseline --concurrency 4
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/subquery_recall_v1/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --variant main_hyde_subquery --policies item_balanced --output-dir logs/context_policy_answers_subquery_v1_expanded --concurrency 4
python scripts/summarize_subquery_recall.py --retrieval logs/subquery_recall_v1/retrieval.json --baseline-answers logs/context_policy_answers_subquery_v1_baseline/experiment.json --expanded-answers logs/context_policy_answers_subquery_v1_expanded/experiment.json --output eval/SUBQUERY_RESULTS.md --case-dir logs/source_diagnostic_subquery_v1
```

将API_ENV_PATH替换为授权API配置文件路径。缓存保留同一输入的规划、HyDE和检索结果；答案评审失败重试复用已生成答案。

## 在线试用

在启动查询服务前设置以下环境变量，或在查询状态传入subquery_retrieval_enabled=True：

```powershell
$env:CONTEXT_SELECTION_POLICY = 'item_balanced'
$env:SUBQUERY_RETRIEVAL_ENABLED = 'True'
```

关闭SUBQUERY_RETRIEVAL_ENABLED即可回到既有路径。Trace保存规划子查询、商品下标、搜索次数、命中数量和失败；chunk保留retrieval_sources。
