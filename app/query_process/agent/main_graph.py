from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
from app.utils.eval_flag_utils import is_eval_enabled
# 导入所有节点函数
from app.query_process.agent.nodes.node_build_context import node_build_context
from app.query_process.agent.nodes.node_query_intent import node_query_intent
from app.query_process.agent.nodes.node_direct_answer import node_direct_answer
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_doc_need_router import node_doc_need_router
from app.query_process.agent.nodes.node_retrieval_plan import node_retrieval_plan
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_self_eval import node_self_eval
from app.query_process.agent.nodes.node_retry_web import node_retry_web
from app.clients.run_state_utils import update_run_node, checkpoint_run


def _tracked_node(node_name, func):
    def _wrapped(state):
        run_id = state.get("run_id") if isinstance(state, dict) else ""
        if run_id:
            update_run_node(run_id, node_name)
        result = func(state)
        if run_id:
            checkpoint_run(run_id)
        return result
    return _wrapped

# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
builder.add_node("node_build_context", _tracked_node("node_build_context", node_build_context))  # 【新】上下文捆绑包构建（入口节点）
builder.add_node("node_query_intent", _tracked_node("node_query_intent", node_query_intent))  # 查询意图识别
builder.add_node("node_direct_answer", _tracked_node("node_direct_answer", node_direct_answer))  # 非检索类问题直接回复
builder.add_node("node_item_name_confirm", _tracked_node("node_item_name_confirm", node_item_name_confirm))  # 确认商品
builder.add_node("node_doc_need_router", _tracked_node("node_doc_need_router", node_doc_need_router))  # 判断是否需要查文档
builder.add_node("node_retrieval_plan", _tracked_node("node_retrieval_plan", node_retrieval_plan))  # 检索策略计划
builder.add_node("node_multi_search", _tracked_node("node_multi_search", lambda x: x))  # 虚拟节点：多路搜索分叉点
builder.add_node("node_search_embedding", _tracked_node("node_search_embedding", node_search_embedding))  # 向量搜索
builder.add_node("node_search_embedding_hyde", _tracked_node("node_search_embedding_hyde", node_search_embedding_hyde))
builder.add_node("node_web_search_mcp", _tracked_node("node_web_search_mcp", node_web_search_mcp))
builder.add_node("node_join", _tracked_node("node_join", lambda x: {}))  # 虚拟节点：多路搜索合并点
builder.add_node("node_rrf", _tracked_node("node_rrf", node_rrf))  # 排序
builder.add_node("node_rerank", _tracked_node("node_rerank", node_rerank))  # 重排
builder.add_node("node_self_eval", _tracked_node("node_self_eval", node_self_eval))  # 结果自评（Agent 决策点）
builder.add_node("node_retry_web", _tracked_node("node_retry_web", node_retry_web))  # 自评不足时补搜 Web
builder.add_node("node_answer_output", _tracked_node("node_answer_output", node_answer_output))  # 生成

# 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑；
# lambda x:x 含义：接收 state 并原样返回，是最轻便的 “无逻辑传递” 方式；
# 普通函数替换：定义 def 函数名(state): return state 即可完全等价，优势是易扩展、易调试；

# 设置起点（改造后入口改为 node_build_context）
builder.set_entry_point("node_build_context")
builder.add_edge("node_build_context", "node_query_intent")


def route_after_query_intent(state: QueryGraphState):
    if state.get("query_route") == "product_doc_qa":
        return "node_item_name_confirm"
    return "node_direct_answer"


def route_after_item_confirm(state: QueryGraphState):
    # 如果已有答案（Branch B/C），直接跳到输出
    if state.get("answer"):
        """
        这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
        具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
        1. 多选一（反问用户） ：
        - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，且置信度都不足以直接确认。
        - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
        - 结果 ：此时不需要再去检索文档了，直接把这句话发给用户让他选。
        2. 查无此人（拒绝回答） ：

        - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，但库里只有华为的数据），或者评分过低（<0.6）。
        - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
        - 结果 ：同样不需要后续检索，直接结束流程。
        """
        return "node_answer_output"
    # 否则进入文档检索需求判断
    return "node_doc_need_router"


def route_after_doc_need_router(state: QueryGraphState):
    """
    文档需求路由后的分支：
    - need_retrieval → node_retrieval_plan（走检索流程）
    - reuse_context_evidence → node_answer_output（复用历史证据直接回答）
    - direct_answer / clarification → node_answer_output（直接回复/澄清）
    """
    if state.get("doc_need_route") != "need_retrieval":
        return "node_answer_output"
    return "node_retrieval_plan"


def route_after_self_eval(state: QueryGraphState):
    """
    Agent 结果自评路由：
    - sufficient：召回充分，直接进回答
    - insufficient_retry 且未达重试上限：进 node_retry_web 补搜
    - 其余（needs_clarification 或已达重试上限）：进回答（answer 已被填为拒答）
    """
    status = state.get("self_eval_status", "")
    retry_count = int(state.get("self_eval_retry_count", 0))
    if status == "sufficient":
        return "node_answer_output"
    if (
        status == "insufficient_retry"
        and retry_count < 1
        and not state.get("_no_web_override")
        and not is_eval_enabled("EVAL_DISABLE_SELF_EVAL_RETRY")
        and not is_eval_enabled("EVAL_DISABLE_WEB")
    ):
        return "node_retry_web"
    return "node_answer_output"


# 1. 查询意图识别 -> 产品文档问答 / 直接回复
builder.add_conditional_edges(
    "node_query_intent",
    route_after_query_intent
)
builder.add_edge("node_direct_answer", "node_answer_output")

# 2. 商品确认 -> (条件分叉) -> 是否需要查文档 / 答案输出
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm
)

# 3. 是否需要查文档 -> 检索计划 / 答案输出
builder.add_conditional_edges(
    "node_doc_need_router",
    route_after_doc_need_router
)
builder.add_edge("node_retrieval_plan", "node_multi_search")

# 4. 并发执行三路搜索
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")

# 5. 三路搜索 -> 结果合并
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")

# 6. 合并 -> 排序 -> 重排 -> 自评 -> (条件分叉) -> 回答/补搜 -> 结束
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_self_eval")
# 自评后按 sufficient / insufficient_retry / needs_clarification 分流
builder.add_conditional_edges("node_self_eval", route_after_self_eval)
# 补搜后回到 RRF 重新融合、重排、再自评（第二次自评 retry_count=1 不会再 retry）
builder.add_edge("node_retry_web", "node_rrf")
builder.add_edge("node_answer_output", END)

# 编译生成可执行的 Runnable 应用（带 Checkpointer 持久化）
from app.conf.checkpoint_config import get_checkpointer

checkpointer = get_checkpointer()
if checkpointer:
    query_app = builder.compile(checkpointer=checkpointer)
else:
    query_app = builder.compile()
