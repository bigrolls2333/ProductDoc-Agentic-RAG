# ProductDoc-Agent：商品文档检索与评测系统

面向商品文档的检索与评测系统，保留 Agentic RAG 问答入口，覆盖文档导入、结构化检索、多轮上下文、证据自评和流式回答。

项目以 FastAPI 提供服务，以 LangGraph 编排有状态工作流，以 Milvus、MongoDB 和 MinIO 组成知识库基础设施。BGE-M3 dense+sparse、RRF 与 BGE Reranker 负责本地混合检索，Agent 按问题选择历史证据复用、HyDE、联网补搜或澄清。

评测使用版本化语料、分级相关性标注（qrels）与 PDF 原文证据锚点，分别观察候选召回、重排和实际入模上下文。[评测指南](eval/README.md)提供可离线运行的公开样例，[审核工作台](eval/WORKBENCH.md)支持原文复核、Bad Case 分析、标注修订和新版本发布。

当前开发集覆盖28份文档、2216个chunks、130道已通过AI来源审核的问题。四组召回对照的条件和结果见[评测报告](eval/DIVERSITY_RESULTS.md)。这些数据尚未经过独立人工核验，不作为人工金标或盲测成绩；真实文档、数据集、模型缓存及日志保留在本地。

子查询扩展复用检索规划器，生成绑定已确认商品范围的少量子查询，与原查询、HyDE 共同召回并共享融合预算。该分支默认关闭，配置方法见[工作台与检索策略指南](eval/WORKBENCH.md#试用子查询扩展召回)。

## 项目概览

面向读者：招聘方用于快速了解项目价值，技术面试官用于定位核心实现，开发者用于本地启动和扩展。

核心能力：

- 双工作流设计：文档导入图负责构建知识库，查询图负责完成 Agentic RAG 问答。
- 上下文管理：将历史消息、已确认商品、最近证据、答案摘要和待澄清事项整理为结构化 `context_bundle`。
- 工具选择：根据查询意图和时效性选择本地向量检索、HyDE 检索和 WebSearch MCP。
- 证据处理：使用 RRF 融合多路召回结果，再通过 BGE Reranker 进行精排。
- 受控检索扩展：复用上游 `item_names` 按商品召回，支持预算内轮询选择，以及最多3条子查询的可选召回分支。
- 标准化评测：比较 BM25、dense、sparse、hybrid 与 hybrid＋reranker，输出 Recall、Precision、MRR、nDCG 及标注覆盖率。
- 审核回流：定位过滤、召回、排序与预算损失，复核原文后发布标注新版本，再进行回归评测。
- Agent Loop：自评证据充分性，必要时触发一次联网补搜，再回到融合、重排和自评流程。
- 可靠执行：文件内容哈希生成稳定 `doc_id`，Redis 分布式锁抑制重复导入，RQ 提供可选后台队列，SQLite Checkpoint 支持图状态持久化。
- 可观测性：记录路由决策、工具调用、召回数量、重排结果、自评状态、Checkpoint 和耗时 Trace。

## 架构与数据流

```mermaid
flowchart LR
    U[用户] --> QAPI[Query API]
    QAPI --> QG[LangGraph Query Graph]
    QG --> C[Context Bundle]
    C --> R[意图识别与商品确认]
    R --> P[检索计划]
    P --> S1[向量检索]
    P --> S2[HyDE 检索]
    P --> S3[WebSearch MCP]
    P -. 可选 .-> S4[商品约束子查询]
    S1 --> F[RRF 融合]
    S2 --> F
    S3 --> F
    S4 --> F
    F --> RR[BGE Reranker]
    RR --> E[证据自评]
    E -->|充分| A[答案生成]
    E -->|需要补充| S3R[一次 Web 补搜]
    S3R --> F
    A --> O[SSE / JSON / History / Trace]

    IAPI[Import API] --> IG[LangGraph Import Graph]
    IG --> D[PDF/Markdown 解析]
    D --> M[图片与对象存储]
    M --> V[切分与向量化]
    V --> DB[(Milvus)]
    V --> PM[(MongoDB 商品记忆)]

    DB --> SNAP[冻结语料与证据锚点]
    SNAP --> EV[检索与答案评测]
    EV --> BC[Bad Case 审核]
    BC --> VER[标注版本发布]
    VER --> EV
```

### 文档导入链路

```text
Upload API
  → SHA-256 doc_id
  → Redis upload lock（可选）
  → PDF/Markdown 解析
  → 图片上传 MinIO
  → 标题语义切分
  → 商品名称识别
  → BGE-M3 向量化
  → Milvus collection
```

`doc_id` 由文件内容生成。同一份文件在文件名变化后仍然保持相同标识，便于去重、追踪和后续重建索引。

### 查询链路

```text
Query API
  → session memory / history
  → context_bundle
  → query intent
  → item confirmation
  → retrieval plan
  → embedding + HyDE + Web + 可选子查询并行召回
  → RRF + BGE Reranker
  → self-evaluation
  → optional Web retry
  → answer + evidence + trace
```

## 核心模块

| 模块 | 工程价值 | 主要实现 |
| --- | --- | --- |
| Agent Loop | 把单次生成变成可控的“计划—执行—检查—补充”闭环 | [`main_graph.py`](app/query_process/agent/main_graph.py)、[`node_self_eval.py`](app/query_process/agent/nodes/node_self_eval.py)、[`node_retry_web.py`](app/query_process/agent/nodes/node_retry_web.py) |
| 上下文管理 | 控制多轮对话输入规模，保留对当前问题真正有用的信息 | [`node_build_context.py`](app/query_process/agent/nodes/node_build_context.py)、[`state.py`](app/query_process/agent/state.py) |
| 短期与长期记忆 | 分离会话记忆与商品级经验，支持指代消解和检索策略复用 | [`mongo_session_memory_utils.py`](app/clients/mongo_session_memory_utils.py)、[`mongo_product_memory_utils.py`](app/clients/mongo_product_memory_utils.py) |
| Tool Calling | 让模型负责工具选择，代码负责工具边界和结果归一 | [`node_retrieval_plan.py`](app/query_process/agent/nodes/node_retrieval_plan.py)、[`node_web_search_mcp.py`](app/query_process/agent/nodes/node_web_search_mcp.py) |
| RAG 检索 | 组合本地知识、查询扩展和时效信息，提升召回覆盖率 | [`node_search_embedding.py`](app/query_process/agent/nodes/node_search_embedding.py)、[`node_search_embedding_hyde.py`](app/query_process/agent/nodes/node_search_embedding_hyde.py)、[`node_rrf.py`](app/query_process/agent/nodes/node_rrf.py)、[`node_rerank.py`](app/query_process/agent/nodes/node_rerank.py) |
| HITL 澄清 | 商品型号存在歧义时保留待澄清状态，让下一轮对话继续完成确认 | [`node_item_name_confirm.py`](app/query_process/agent/nodes/node_item_name_confirm.py)、[`mongo_session_memory_utils.py`](app/clients/mongo_session_memory_utils.py) |
| 可靠导入 | 通过稳定 ID、分布式锁、后台队列和状态轮询支撑长任务 | [`file_hash_utils.py`](app/utils/file_hash_utils.py)、[`redis_lock_utils.py`](app/utils/redis_lock_utils.py)、[`import_queue.py`](app/utils/import_queue.py) |
| Checkpoint 与 Trace | 保存工作流状态，记录可复盘的决策链路 | [`checkpoint_config.py`](app/conf/checkpoint_config.py)、[`query_trace_utils.py`](app/utils/query_trace_utils.py) |
| 检索规划与子查询 | 在已有规划调用中生成商品约束子查询，共享分支融合权重 | [`node_retrieval_plan.py`](app/query_process/agent/nodes/node_retrieval_plan.py)、[`subquery_retrieval.py`](app/query_process/subquery_retrieval.py) |
| 标准检索与答案评测 | 版本化语料、qrels、原文事实和引用支持评测 | [`app/evaluation/`](app/evaluation/)、[`scripts/retrieval_eval.py`](scripts/retrieval_eval.py)、[`scripts/eval_answer_trust.py`](scripts/eval_answer_trust.py) |
| 审核与实验工作台 | Bad Case 队列、修订冲突检查、版本发布与持久化任务 | [`api.py`](app/evaluation/api.py)、[`workbench.py`](app/evaluation/workbench.py)、[`eval/WORKBENCH.md`](eval/WORKBENCH.md) |

## 工程设计亮点

### 状态图替代隐式调用链

查询流程通过 `QueryGraphState` 传递结构化状态，每个节点只负责一个清晰阶段。路由、检索、排序、自评和回答都能单独观测，新增策略时只需要扩展节点和边。

导入流程使用独立的 `ImportGraphState`，把文件路径、解析结果、切片和向量化结果串成可追踪的数据流。导入图采用懒加载，服务启动时保持轻量，首次执行任务时再初始化重依赖。

### 记忆与上下文分层

- Session Memory 保存最近确认的商品、待澄清任务、最近证据和答案摘要。
- Product Memory 保存标准商品、别名、关联文档和高频问题模式。
- `context_bundle` 在每一轮查询开始时统一组装，后续节点读取稳定结构，减少跨模块拼接字符串。

### 检索与 Agent Loop

`node_retrieval_plan` 输出本地向量检索、HyDE 和 WebSearch 开关。启用子查询实验时，同一次规划调用额外生成最多3条子查询，绑定已确认商品的下标；子查询共享一路RRF权重，避免按扩展数量重复投票。多个分支并行执行，`node_rrf` 做名次融合，`node_rerank` 进行语义精排。

上下文默认使用 `ranked`；可选 `item_balanced` 复用上游商品识别结果，按商品分别召回与保留候选，再按 A1、B1、A2、B2 轮询纳入字符预算。超预算片段跳过并继续选择，Trace 记录缺失商品与预算丢弃。该策略不使用商品领域关键词或特定表格章节规则。

`node_self_eval` 根据问题、召回证据和当前重试次数判断证据是否充分。`insufficient_retry` 只允许触发一次 Web 补搜，补搜结果重新进入 RRF、Rerank 和自评，形成有上限的 Agent Loop。

### 幂等、并发与失败恢复

- 文件内容哈希生成稳定 `doc_id`，将导入幂等键从文件名提升到内容层。
- Redis `SET NX` + TTL 形成上传锁，同一文档的并发导入共享已有任务状态。
- RQ 以 `import:{task_id}` 作为 job ID；Redis 或 Worker 不可用时回退 FastAPI BackgroundTasks，保留本地开发路径。
- 每个查询节点执行前后更新运行状态，Checkpoint 使用 `thread_id=session_id` 关联同一会话。
- Trace 使用 JSONL 追加写入，记录路由、召回、重排、自评、重试、Checkpoint 和耗时摘要。

这些边界让长耗时任务、外部服务抖动和多轮追问都具备清晰的状态入口，便于继续接入更强的队列、数据库事务和分布式观测系统。

## API 入口

### Query Service：端口 8001

启动文件：[`app/query_process/api/query_service.py`](app/query_process/api/query_service.py)

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/query` | 同步或异步提交问题 |
| `GET` | `/stream/{session_id}` | 通过 SSE 获取流式进度与答案 |
| `GET` | `/history/{session_id}` | 获取会话历史 |
| `DELETE` | `/history/{session_id}` | 清理会话历史与会话记忆 |
| `GET` | `/chat.html` | 打开查询页面 |

同步查询示例：

```bash
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query":"如何配置产品的上网方式？","is_stream":false}'
```

流式查询先提交：

```bash
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query":"如何排查设备无法联网？","is_stream":true}'
```

响应中的 `session_id` 可用于订阅：

```bash
curl -N http://127.0.0.1:8001/stream/<session_id>
```

### Import Service：端口 8000

启动文件：[`app/import_process/api/file_import_service.py`](app/import_process/api/file_import_service.py)

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/upload` | 批量上传 PDF/Markdown 文档 |
| `GET` | `/status/{task_id}` | 查询导入任务状态和阶段进度 |
| `GET` | `/import.html` | 打开文档导入页面 |

上传示例：

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "files=@path/to/manual.pdf"
```

## 本地启动

### 1. 准备环境

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Milvus
- MongoDB
- MinIO
- OpenAI 兼容模型服务
- 可选：Redis、RQ Worker、MinerU API、WebSearch MCP

安装依赖：

```bash
uv sync
```

### 2. 配置环境变量

复制模板并填写本地服务地址与密钥：

```bash
copy .env.example .env
```

`.env` 已加入 Git 忽略规则。公开仓库只保留脱敏后的 [`.env.example`](.env.example) 模板，密钥、数据库连接串、模型路径和服务地址均通过环境变量注入。

### 3. 启动基础设施

Milvus 依赖 etcd 和 MinIO：

```bash
docker compose -f milvus/docker-compose.yml up -d
```

Redis 为可选组件：

```bash
docker compose -f docker-compose.redis.yml up -d
```

### 4. 启动服务

在两个终端分别运行：

```bash
uv run python -m app.import_process.api.file_import_service
```

```bash
uv run python -m app.query_process.api.query_service
```

浏览器访问：

- 查询页面：<http://127.0.0.1:8001/chat.html>
- 导入页面：<http://127.0.0.1:8000/import.html>
- Query Swagger：<http://127.0.0.1:8001/docs>
- Import Swagger：<http://127.0.0.1:8000/docs>

评测审核工作台可独立启动，无需初始化模型或连接上述数据库：

```bash
uv run python -m uvicorn app.evaluation.api:app --host 127.0.0.1 --port 8090
```

打开 <http://127.0.0.1:8090/>，选择 `example` 查看公开样例。真实数据版本放入 `eval/local/` 后会出现在列表中。

首次导入前，需要准备 BGE-M3 与 BGE Reranker 模型，并将路径写入 `.env`。WebSearch、MinerU 和模型服务属于可插拔的外部依赖，按需启用即可。

### 5. 启动 RQ Worker（可选）

当 `.env` 中设置 `RQ_ENABLED=True` 且 Redis 已启动时运行：

```bash
uv run python -m app.import_process.worker
```

## 配置分组

`.env.example` 按职责组织变量，常用配置如下：

| 分组 | 关键变量 | 作用 |
| --- | --- | --- |
| LLM | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_DEFAULT_MODEL` | OpenAI 兼容模型访问 |
| 解析 | `MINERU_BASE_URL`、`MINERU_API_TOKEN` | PDF/复杂文档解析服务 |
| Embedding | `BGE_M3_PATH`、`BGE_DEVICE`、`BGE_FP16` | 本地向量模型 |
| Reranker | `BGE_RERANKER_LARGE`、`BGE_RERANKER_DEVICE` | 召回结果精排 |
| Vector DB | `MILVUS_URL`、`CHUNKS_COLLECTION`、`ITEM_NAME_COLLECTION` | 切片和商品名称索引 |
| Document DB | `MONGO_URL`、`MONGO_DB_NAME` | 历史、会话记忆和商品记忆 |
| Object Storage | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` | 原始文件和图片存储 |
| Agent | `CHECKPOINT_ENABLED`、`QUERY_TRACE_PATH` | 状态恢复和 Trace 输出 |
| 检索实验 | `CONTEXT_SELECTION_POLICY`、`CONTEXT_PER_ITEM_K`、`SUBQUERY_RETRIEVAL_ENABLED` | 上下文选择、每商品候选数量及可选子查询分支 |
| Async jobs | `REDIS_*`、`RQ_*` | 任务状态、上传锁和后台队列 |

## 评测与观测

标准评测与完整 Agent 行为检查分开运行。检索评测使用冻结语料和 qrels，以 Recall@K、Precision@K、MRR、nDCG@K 和标注覆盖率比较策略；答案评测使用实际入模上下文，检查参考事实、忠实度、引用完整性与拒答。数据格式与完整命令见[评测指南](eval/README.md)及[协议参考](eval/BENCHMARK.md)。

以下公开样例不需要 API、模型权重或数据库：

```bash
uv run python scripts/retrieval_eval.py validate --dataset eval/examples/retrieval
uv run python scripts/retrieval_eval.py run --dataset eval/examples/retrieval --variants bm25 --candidate-k 3 --context-k 1 --output logs/retrieval_example.json
uv run python scripts/eval_answer_trust.py --dataset eval/examples/retrieval --answers eval/examples/answers.jsonl --judgments eval/examples/judgments.jsonl --output logs/answer_example.json
```

公开样例使用虚构数据，只演示协议与评测流程。真实数据在 `eval/local/`，完整实验产物在 `logs/`，均不进入公开仓库。完整 Agent 的路由与工具行为可另用 `scripts/eval_comprehensive.py` 检查，其规则通过率不等同于标准检索指标。

查询 Trace 默认写入 `logs/query_trace.jsonl`，可用于分析：

- 各路由和检索计划的分布
- 各召回分支的结果数量
- RRF 与 Rerank 后的证据变化
- 自评状态与 Web 重试次数
- Checkpoint 覆盖情况
- 查询总耗时及后续 P50/P95 统计

## 项目结构

```text
.
├── app/
│   ├── conf/                 # 环境变量驱动的配置
│   ├── clients/              # Milvus、MongoDB、MinIO、Redis 客户端
│   ├── evaluation/           # 数据协议、指标、归因与审核工作台
│   ├── import_process/       # 文档导入图与导入 API
│   ├── query_process/        # 查询图、Agent 节点与 Query API
│   ├── lm/                   # LLM、Embedding、Reranker 封装
│   ├── tool/                 # 模型下载等辅助工具
│   └── utils/                # 任务状态、锁、队列、Trace 等基础能力
├── prompts/                  # 可版本化的提示词模板
├── scripts/                  # 评测、分析与验证脚本
├── eval/                     # 评测协议、公开样例与结果摘要
├── tests/                    # 指标、证据、检索与审核一致性验证
├── milvus/                   # Milvus 本地部署编排
├── docker-compose.redis.yml  # Redis 本地部署编排
├── .env.example              # 脱敏环境变量模板
├── pyproject.toml            # 项目元数据与依赖
└── uv.lock                   # 可复现依赖锁定
```

## 技术选型

| 技术 | 选择理由 |
| --- | --- |
| FastAPI | 提供轻量 API、SSE 和自动化 Swagger 文档 |
| LangGraph | 用显式状态和边表达 Agent 路由、并行分支、重试与 Checkpoint |
| LangChain | 统一 LLM、消息和检索组件接口 |
| BGE-M3 + BGE Reranker | 组合高覆盖召回与精排能力，适合产品手册场景 |
| Milvus | 承载大规模文档切片向量和商品名称索引 |
| MongoDB | 保存会话历史、会话记忆和商品级长期记忆 |
| MinIO | 保存原始文件和文档图片 |
| Redis + RQ | 为长耗时导入任务提供可选的共享状态与异步队列 |
| SQLite Checkpoint | 为本地开发和单实例部署提供低成本状态恢复 |

## 工程化路线

当前代码边界已经为后续演进预留了清晰位置：

- 将 Checkpoint 从本地 SQLite 迁移到共享存储，支持多实例协作。
- 为模型、MCP 和数据库客户端补充统一超时、退避重试和熔断策略。
- 在 API 网关层加入鉴权、限流和租户隔离。
- 将 JSONL Trace 接入 OpenTelemetry 或集中式日志平台。
- 持续扩充脱敏验证集，比较 Recall@K、证据命中率、答案通过率与 P50/P95 延迟。

## 许可与使用

仓库中的示例代码用于展示 Agentic RAG 应用的工程组织方式。使用外部模型、解析服务和文档数据时，请遵守相应服务条款与数据授权要求。
