# 标准评测参考

本参考面向维护检索实验和标注数据的开发者。用 `scripts/retrieval_eval.py` 运行固定语料实验，用 `scripts/eval_answer_trust.py` 评测保存的答案；两者都不会调用完整 Agent 或写入会话历史。

## 数据目录

一个数据版本包含五个文件。公共样例在 `eval/examples/retrieval/`，产品样本和实验结果分别放入被 Git 忽略的 `eval/local/`、`logs/`。

| 文件 | 必要字段 | 含义 |
| --- | --- | --- |
| `manifest.json` | `schema_version,name,version,corpus_version,index_version` | schema 固定为 `productdoc.benchmark.v1`；数据、原文、切分索引分别标识版本 |
| `corpus.jsonl` | `chunk_id,doc_id,content` | 当前索引的冻结检索单元；产品过滤还需要 `product_id` 或 `canonical_name` |
| `queries.jsonl` | `query_id,query,split,review_status,answerable` | `split` 为 dev/test/regression；`review_status` 为 pending/approved；草稿 answerable 可为 null |
| `evidence.jsonl` | `evidence_id,doc_id,source_revision,quote,locator` | 独立于切分策略的证据锚点，locator 记录原文页码、章节或位置 |
| `qrels.jsonl` | `query_id,chunk_id,relevance,evidence_ids` | 0 无关、1 有帮助、2 直接证据；未标注结果保持 unjudged |

批准可回答问题前，填写 `group_id`、`evidence_ids`、正相关 qrels。每个证据锚点必须关联到正相关 chunk，原文 quote 必须存在于该 chunk。无答案问题没有正相关 qrels，IR 指标为 null，单独评测拒答。

`reference_facts` 使用 `[{"fact_id":"稳定事实ID","text":"必要事实","evidence_ids":["证据ID"]}]`。默认要求同一事实中的全部证据；可选 `evidence_sets` 表达替代路径，例如 `[["A"],["B","C"]]` 表示 A 或 B+C 足以支持事实。`evidence_ids` 包含这些路径的证据并集。报告同时保留严格 span recall 与按替代路径计算的 fact recall，避免把同义原文全部当成必须召回。

未提供标准事实时，答案正确性不计分，但仍可检查事实支持。多轮样本必须补充经过审核的独立 `retrieval_query`，避免把原始指代问题直接当成检索器输入。纯检索实验不评价历史理解或商品识别。

同一 `group_id` 不得跨 split。按商品、文档或问题家族人工安排 group，不能把同一问题的近似改写分入开发集和测试集。加载器检查 group 一致性，但不会自动判断两个不同 group 是否语义重复。

切分消融应保持 queries 和 evidence 不变，为新 chunks 重新审核 qrels；不要复用旧 chunk ID。证据 quote 取能独立支持一个事实的短原文，跨段组合事实使用多个 evidence ID。

## 离线运行

下面是可直接运行的虚构样例。结果只验证数据流，不表示真实商品文档的检索质量。命令在项目根目录执行，使用项目 Python 环境。

```powershell
python scripts/retrieval_eval.py validate --dataset eval/examples/retrieval
python scripts/retrieval_eval.py run --dataset eval/examples/retrieval --variants bm25 --candidate-k 3 --context-k 1 --output logs/retrieval_example.json
python scripts/retrieval_eval.py pool --dataset eval/examples/retrieval --reports logs/retrieval_example.json --output logs/annotation_pool.jsonl
python scripts/retrieval_eval.py collect --report logs/retrieval_example.json --output logs/case_queue.jsonl
```

validate 返回 2 个已批准问题、3 个 chunks、2 条稳定证据、5 条 qrels。run 写入 JSON 与同名 Markdown；`context-k 1` 刻意丢失双事实问题的一条证据，便于查看 `ranking_or_selection_loss`。collect 重复执行不会重复追加相同观察，测试集与 annotation-only 报告禁止回流。

## 真实数据与候选池

迁移旧样本只保留待审核提示，不能把标题、关键词和答案要点自动视为金标。prepare 拒绝覆盖已有目录。

```powershell
python scripts/retrieval_eval.py prepare --legacy eval/validation_set.jsonl --output eval/local/productdoc-v1
```

本地已迁移该目录。真实 corpus 需要从可用的 Milvus collection 导出。以下为命令格式，`COLLECTION_NAME` 替换为当前切片 collection，两个版本参数填写你冻结的原文与索引版本。

```powershell
python scripts/retrieval_eval.py export --dataset eval/local/productdoc-v1 --collection COLLECTION_NAME --corpus-version manuals-v1 --index-version title-2000-v1
python scripts/retrieval_eval.py run --dataset eval/local/productdoc-v1 --variants bm25 --annotation-only --scope all --output logs/productdoc_pool_run.json
python scripts/retrieval_eval.py pool --dataset eval/local/productdoc-v1 --reports logs/productdoc_pool_run.json --output logs/productdoc_pool.jsonl
```

导出仅执行查询，要求目标草稿 corpus 为空且没有已批准样本。annotation-only 包含待审核问题，保留候选但不产生质量分数。可从多个变体报告合并候选池；池中的 relevance 保持 null，先查原文再标注，也要补查检索器都漏掉的证据。

正式评测示例使用 `--variants bm25,dense,sparse,hybrid,hybrid_rerank`，不加 `--annotation-only`。product scope 使用审核后的标准产品身份；all scope 不加产品过滤。两者的实验含义不同，不能混为一个基线。BM25 使用 Latin/型号词元和中文二元字符，不代表中文分词 BM25 服务的全部能力。BGE sparse 与 BM25 分别报告。

外部 API 配置可通过 `scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_comprehensive.py ...` 注入。`API_ENV_PATH` 为你授权读取的 `.env` 路径。包装器只将 API 地址、密钥、温度和模型名称写入子进程环境，不修改本项目 `.env`；模型优先使用非空 `LLM_DEFAULT_MODEL`，否则使用 `MODEL_NAME`。`--no-web --no-checkpoint` 放在包装器的脚本路径之前，分别关闭图内 Web 和评测 Checkpoint。

包装器默认设置 `EVAL_DISABLE_PRODUCT_MEMORY_WRITE=1`，防止评测答案回写商品长期记忆；只读的已有记忆仍可能影响完整 Agent 的结果，因此做记忆消融时还需冻结输入记忆或使用隔离数据库。纯检索实验不访问 MongoDB。

## 人工审核与新版本

review 读取明确的审核决定，在新目录发布版本；输入目录保持原样。公共 `eval/examples/reviews.jsonl` 展示审核格式，下面命令生成一个独立样例版本。

```powershell
python scripts/retrieval_eval.py review --dataset eval/examples/retrieval --reviews eval/examples/reviews.jsonl --output logs/reviewed_example --version 2.0
```

每条决定包含 `query_id,reviewer,reviewed_at,query,evidence,qrels`。query 只更新 answerable、group、split、产品身份、证据 ID、标准事实或 retrieval_query。qrels 完整替换该问题的旧判断。已有 evidence ID 不能指向不同原文，冻结且已批准的 test 问题不能通过此入口修改。`source_case_id` 可追溯审核队列中的案例。

`reviewer_type` 和 `human_verified` 记录审核来源。`approved` 表示满足该版本的评测准入条件，不意味着人工金标；没有独立人工核验的AI审核数据必须单列其来源，manifest及报告记录人工核验条数。

发布前校验跨文件引用、原文 quote、相关性等级、分组泄漏和金标完整性。每份实验报告包含数据文件 SHA-256、代码版本与改动摘要、评测源码哈希、变体配置和分阶段结果。数据文件变化后，旧报告不能用于新版本候选池。

## 指标与归因

Recall 的分母为该问题全部正相关 qrels；nDCG 的 IDCG 来自固定 qrels 的理想排序。MRR、Hit、Precision 使用同一标注。重复 chunk 去重，缺少返回结果按未命中计算；运行错误单列。每个平均值带 `n`，无答案问题不进入 IR 分母。

未判断候选在 IR 计分时贡献零增益，同时报告 `judged_fraction@K` 与 `unjudged@K`。这是不完整标注下的约定，不意味着未判断等于不相关；不同方案标注覆盖差异大时应先扩充候选池。延迟是共享检索缓存的实验阶段耗时，排除模型初始化，不是线上 P95。

阶段依次为 eligible、candidates、reranked（选择 Top-K）、context（字符预算后的全文 chunks）。eligible 只保存已判断 chunks 的投影，完整集合可由冻结 corpus 与 filter 重建，避免每个问题重复存储整个语料 ID 列表。另保留 `rerank_order` 完整排序。上下文模拟预算只统计 content 字符，不包含生成提示词元数据；实际 Agent 的上下文单独记录在 QueryTrace。

归因输出是观察或待确认假设：过滤丢失、候选遗漏、排序/选择丢失、上下文预算丢失、执行错误。没有完整的阶段证据时不会补造中间状态。不能仅凭这些观察声称已证明解析错误或语料缺口；这两类仍需对照原文与解析产物。

## 答案事实与引用评测

答案输入 JSONL 的每条记录包含唯一 `record_id`、已批准的 `query_id`、完整 answer、显式 abstained、`context_source: actual_prompt`、实际 `context_docs`。每份上下文使用 `chunk_id,text`，必须来自冻结 corpus；不要提交完整候选集代替真实上下文。

在生成前被门控拒答的记录使用 `context_source: no_generation`，同时要求 `abstained: true` 和空 `context_docs`。仍然评测拒答是否合理，不能因为没有生成而从误拒答分母删除。

```powershell
python scripts/eval_answer_trust.py --dataset eval/examples/retrieval --answers eval/examples/answers.jsonl --judgments eval/examples/judgments.jsonl --output logs/answer_example.json
```

该命令重放保存的虚构裁判结果，不调用模型。删除 `--judgments` 后使用现有 LLM 配置调用裁判，记录模型、提示词版本和哈希；在依赖服务可用后执行。它不会生成待测答案，也不代表裁判已通过人工校准。

裁判区分 supported、contradicted、unsupported。支持和矛盾判断必须附真实上下文中的原文 quote；不存在的证据、遗漏的引用、无效 JSON 和调用异常计为错误。引用使用 `[chunk_id]` 或 `[chunk_id=ID]`，数字引用必须先映射为稳定 ID。当前固定语料评测不支持 Web 证据或图片语义；图片摘要中的错误需单独核验原图。

报告包含事实支持率、矛盾率、引用 precision/completeness、正确性，以及以可回答/无答案样本分别作分母的误拒答率/误回答率。零事实声明的拒答不会获得虚假的 100% faithfulness。裁判判断仍需抽样人工复核；原文 quote 校验只证明引用存在，不能替代语义审核。

线上 QueryTrace v3 现在保存实际上下文、完整答案和提示词哈希；旧综合评测的 judge_artifacts 也保留实际上下文。归档时注意文件体积。只复用旧报告中的预览或检索候选，无法恢复当时真实上下文。

## 已保存结果的重新计分

当修复旧评测的规则错误时，可以重用已保存的执行结果，不重复调用模型。下面命令保留原报告，并对不适用的文档路由、未调用工具的 false 开关做语义归一。

```powershell
python scripts/rescore_saved_eval.py --input logs/eval_full_20261023.json --output logs/eval_full_20261023_rescored.json
```

输出保留原 summary、原状态、各项调整和输入 SHA-256。缺少实际 retrieval_plan 产物时拒绝重评。重新计分导致的通过率变化属于评测口径修正，不能描述为系统质量提升。

## 来源审核诊断集

`eval/diagnostic_selection.json` 固定30道候选诊断题及权威产品手册范围；`eval/source_review_plan.json` 保存明确的AI来源审核修订与等价证据路径。当前25道批准为dev，5道范围过宽暂留待审。该集合用于定位问题，不能替代随机抽样或独立人工审核的test。

来源审核依次执行本地PDF定位、全目标文档AI相关性提议、原文引文核验、参考事实审核和新版本发布。图像自动摘要不作为来源金标；OCR无法精确匹配的引文保持待审，可在核对渲染原页后记录AI视觉审核及预览哈希。无答案表示指定冻结手册中没有支持答案，不能推断设备绝对不具备该功能。

```powershell
python scripts/prepare_source_decisions.py --plan eval/source_review_plan.json --drafts eval/local/source_audit_v1/reference_drafts.jsonl --output eval/local/source_review_decisions_v2.json
python scripts/publish_source_diagnostic.py --dataset eval/local/productdoc-v1 --audits eval/local/source_audit_v1/source_audits.jsonl --sources eval/local/source_pdf_index.jsonl --decisions eval/local/source_review_decisions_v2.json --output eval/local/productdoc-source-v2 --version 0.3-source-reviewed-ai
```

准备决定和发布均拒绝覆盖已有目标；重复审核应使用新文件及版本。PDF、模型提议和本地审核缓存不进入Git。`--decisions` 是明确审核决定，不会把模型提议静默转为人工标签。

```powershell
python scripts/replay_retrieval_eval.py --dataset eval/local/productdoc-source-v2 --source-dataset eval/local/productdoc-v1 --predictions logs/productdoc_bm25_pool_20261023.json logs/productdoc_neural_pool_20261023.json --output logs/source_diagnostic_v2/retrieval.json
python scripts/sweep_context_budget.py --dataset eval/local/productdoc-source-v2 --predictions logs/source_diagnostic_v2/retrieval.json --output logs/source_diagnostic_v2/context_sweep.json
```

重放要求原预测的数据指纹、语料字节哈希、检索问题及过滤条件一致；只更新标注时不用重复加载模型或访问数据库。上下文消融重放完整排序，不调用答案生成模型，因此只能报告证据覆盖和content字符成本。

`prepare_answer_records.py` 从历史综合评测抽取实际prompt，要求每道题有显式拒答审核。`eval_answer_trust.py --concurrency 4` 并发调用裁判；失败保持error，可只抽取失败答案重试，再用保存的判断离线合并。保存的判断应附原裁判模型、提示词哈希和来源数据指纹。

`summarize_source_diagnostic.py` 要求检索、上下文消融及答案报告使用同一数据版本，写入报告、可去重失败队列和同候选reranker案例复核记录。所有自动归因仍保留观察/待审核状态；队列中的失败不会自动变成批准标签或冻结测试题。
