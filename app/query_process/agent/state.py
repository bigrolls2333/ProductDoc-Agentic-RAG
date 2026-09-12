from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档
    rerank_trace: dict  # Rerank 调参与截断过程摘要

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记
    query_route: str  # 查询路由：product_doc_qa / direct_answer / clarification / out_of_scope
    direct_answer: str  # 意图识别阶段生成的直接回复
    doc_need_route: str  # 文档检索需求路由：need_retrieval / direct_answer
    doc_need_reason: str  # 文档检索需求路由原因
    doc_need_confidence: float  # 文档检索需求路由置信度
    retrieval_plan: dict  # 检索计划：开启哪些召回分支
    context_evidence_docs: list  # 多轮追问复用的历史证据摘要
    trace_started_at: float  # 查询trace开始时间戳

    # Agentic RAG：自评与重试
    self_eval_status: str  # "" | "sufficient" | "insufficient_retry" | "needs_clarification"
    self_eval_retry_count: int  # 0 | 1（最多一次 web 重试）
    self_eval_detail: dict  # LLM 评估的完整结果

    # ===== Agentic RAG 改造：上下文记忆与 Checkpoint =====
    context_bundle: dict  # 上下文捆绑包（node_build_context 构建，包含历史消息/已确认商品/证据文档/答案摘要/待澄清任务）
    thread_id: str  # LangGraph thread_id（等于 session_id，用于 checkpoint 恢复）
    checkpoint_enabled: bool  # 当前运行是否启用 LangGraph checkpointer
    resume_from_checkpoint: bool  # 本次调用是否从 checkpoint 恢复
    product_memory_hints: dict  # 商品记忆命中（高频问题/检索策略/别名，用于 query expansion）
    clarification_resolved: bool  # 澄清是否已解决（用户选择候选后设为 True）
    pending_clarification_created: bool  # 本轮是否新创建待澄清任务
    run_id: str  # 项目级运行元数据 ID


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "rerank_trace": {},
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "query_route": "",
    "direct_answer": "",
    "doc_need_route": "",
    "doc_need_reason": "",
    "doc_need_confidence": 0.0,
    "retrieval_plan": {},
    "context_evidence_docs": [],
    "trace_started_at": 0.0,
    "self_eval_status": "",
    "self_eval_retry_count": 0,
    "self_eval_detail": {},
    # Agentic RAG 改造：新增默认值
    "context_bundle": {},
    "thread_id": "",
    "checkpoint_enabled": False,
    "resume_from_checkpoint": False,
    "product_memory_hints": {},
    "clarification_resolved": False,
    "pending_clarification_created": False,
    "run_id": "",
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


# ========================
# ✅ 状态复制函数（你要的）
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)

