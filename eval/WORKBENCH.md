# 使用评测审核工作台

你可以核对参考事实和原文证据、修改相关性、发布数据新版本，并重放冻结检索实验。工作台独立启动，读取本地评测产物，无需加载BGE或连接Milvus、MongoDB。

## 启动

在项目根目录运行：

```powershell
.venv/Scripts/python.exe -m uvicorn app.evaluation.api:app --host 127.0.0.1 --port 8090
```

打开 <http://127.0.0.1:8090/>。启动日志显示 `Application startup complete`；公开仓库可以选择 `example` 查看虚构样例，有本地数据时选择对应版本。当前扩充开发集为 `productdoc-diverse-v1`，包含130道批准题、0道人工核验题。API参考位于 <http://127.0.0.1:8090/docs>。

工作台使用一个服务进程和一个串行任务线程。系统文件锁避免第二个进程同时恢复或派发任务。SQLite记录保存在 `eval/local/workbench.sqlite3`，实验产物保存在 `logs/workbench/<run_id>/`，均不进入Git。

## 审核问题

1. 选择数据版本和题目。勾选“仅看Bad Case”可筛选失败题。
2. 核对必要事实，避免要求问题范围以外的细节。
3. 打开原文证据中的PDF链接，核对引文与物理页码。服务返回PDF前检查SHA-256与来源索引一致。
4. 展开相关性标注和失败记录。列表包含全部正相关片段和失败案例的主要候选：0无关、1有帮助、2直接证据；其他标注保持原值。
5. 如需补充证据，从片段复制连续原文，填写PDF物理页码，再将证据勾选到对应参考事实。保存时检查引文存在于片段、参考证据映射到正相关片段。
6. 选择关联失败案例，填写审核人和说明。完成对应人工核对后勾选确认，再保存审核决定。

“编辑证据组合与分组”允许调整 `evidence_sets`。例如 `[["A"],["B","C"]]` 表示A或B+C可支持同一事实。普通证据勾选按替代关系设置；联合支持使用组合编辑。

每次保存保留新修订，遇到其他窗口已更新的记录返回409。保存前会校验完整数据版本，包括参考事实、引文、qrels和分组一致性。

人工核验标记的范围是问题、参考事实、正相关标注及证据定位。未逐项检查的AI负向标注保留其来源；不能将所有qrels描述为人工逐条标注。自动化验证只使用虚构DEMO数据，真实商品样本等待你亲自审核。

## 发布版本

1. 保存需要发布的审核决定。
2. 点击“发布已审核版本”，选择当前列表中最新、尚未发布的决定。
3. 填写新目录名称和版本标识，点击“发布新版本”。

新目录保留父数据指纹、审核人、说明和来源案例。已有目录、过时修订及重复发布会被拒绝。冻结测试题不能从此入口进入开发回流。

## 创建实验

1. 切换到“实验与对比”，选择“检索结果重评”。
2. 选择目标审核版本和原预测使用的数据版本。首次原预测版本为 `productdoc-v1`。
3. 多选 `logs/productdoc_bm25_pool_20261023.json` 和 `logs/productdoc_neural_pool_20261023.json`。
4. 点击“创建并运行”，查看任务状态和结果。

重评不调用模型。原预测指纹、目标语料、检索问题和产品过滤条件必须满足重放校验；改写检索问题后应重新检索。

选择“上下文选择消融”可使用与目标数据指纹一致的一份完整检索报告，填写Top-K和预算组合。该任务只模拟上下文，不生成答案。

任务状态保存为queued、running、completed、failed或interrupted。重启恢复待执行任务，将遗留running标为interrupted；失败和中断任务可重试。结果对比可选择两份同类型、同数据指纹的报告，口径不同会提示不能直接对照。

## 运行受控答案对照

工作台展示结果，API调用通过命令行与授权配置文件执行：

```powershell
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_context_answers.py --dataset eval/local/productdoc-source-v2 --predictions logs/source_diagnostic_v2/retrieval.json --agent-report logs/eval_full_20261023_rescored.json --per-item-k 20 --output-dir logs/context_policy_answers_v2 --concurrency 4
```

将 `API_ENV_PATH` 替换为自己的配置路径。包装器不复制密钥、不修改项目 `.env`。有效缓存会被复用；裁判失败时保留生成答案，只重试未完成任务。

实验固定排序、模型和提示词，比较ranked、budgeted、item_balanced，绕过上游Agent路由、自评和记忆，不能与完整Agent基线直接比较。`--agent-report`只读取历史Agent输出中的实际item_names，不读取expected或评测数据的canonical_names；每道问题文本必须一致。该条件下的结果见[按商品分组实验](ITEM_BALANCED_RESULTS.md)。

## 使用可选上下文策略

查询图默认使用ranked。在开发环境启动查询服务前可设置：

```powershell
$env:CONTEXT_SELECTION_POLICY = 'item_balanced'
$env:CONTEXT_PER_ITEM_K = '20'
```

也可构造状态时传入 `context_selection_policy="item_balanced", context_per_item_k=20`，状态值优先于环境变量。`balanced`保留为item_balanced的兼容别名，已没有旧的领域表格规则。budgeted按全局排序和预算选择。

item_balanced复用上游item_names，与候选canonical_name精确对应（统一大小写、忽略空白；旧数据可回退item_name）。每个商品保留组内排名前K条，按A1、B1、A2、B2逐轮尝试纳入；超预算片段跳过，直到预算或候选额度耗尽。不从原问题再抽型号，不使用打印关键词或章节特判。

同一策略启用时，普通检索与HyDE分别按每个商品过滤召回Top-K，复用该分支的一次查询向量；RRF也按商品保留候选，避免重新截成全局Top-10。全局候选上限为50，上下文上限为20块、12000字符。后面的商品仍可能因整块证据过大而放不下，Trace会区分“无候选”和“有候选但未入上下文”。

有确认商品时，不用其他商品或缺少商品归属的Web片段填充配额；空item_names明确回退全局预算选择，Trace记录回退原因。需要无商品归属的Web材料时可使用budgeted。该模式不会凭猜测把Web片段归给某个商品。

选择器记录上游商品名、每商品候选数/入选数、缺失商品、预算丢弃和上下文哈希。它不会绕过现有证据充分性门控；上游拒答仍需单独诊断。固定旧候选的实验只能评价上下文选择，不能代表新增的线上按商品召回已完成端到端评测。

数据协议见[标准评测参考](BENCHMARK.md)，首批来源审核结果见[诊断报告](RESULTS_20261023.md)。

## 试用子查询扩展召回

子查询是默认关闭的实验分支。规划器复用已有模型调用，最多输出3条查询及已确认商品下标，检索后共享一路RRF权重；无扩展、无效输出或分支失败时保留原召回路径。它不改变问题改写节点，也不增加领域关键词特判。

在启动查询服务前设置 `CONTEXT_SELECTION_POLICY=item_balanced` 和 `SUBQUERY_RETRIEVAL_ENABLED=True`，或在状态中传入 `subquery_retrieval_enabled=True`。Trace记录 `subquery_trace`；最终chunk保留 `retrieval_sources`，便于查明来源。

四组检索、受控答案对照、限制和复现命令见[子查询实验](SUBQUERY_RESULTS.md)。本轮没有额外事实覆盖收益，不建议默认开启；不要将更多候选或未经校准的LLM原判比例表述为系统准确率提升。

## 审核扩充后的样本

在数据版本下拉框选择 `productdoc-diverse-v1`，可查看130道有效开发题及保留的待审题。新增题的ID以 `VAL-DIV-` 开头；它们保留来源、参考事实和AI审核记录，多轮题另存独立检索问题。人工核验数保持为0，直到你实际审核并明确确认。

覆盖变化、原有/新增分组结果和复现步骤见[扩充开发集报告](DIVERSITY_RESULTS.md)。本轮不把待审题计入评测，也不把这批开发数据当成盲测。
