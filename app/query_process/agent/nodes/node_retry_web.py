import sys
import json
import asyncio

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.query_process.agent.nodes.node_web_search_mcp import mcp_call
from app.utils.task_utils import add_done_task, add_running_task


def _build_retry_query(state: QueryGraphState) -> str:
    """
    根据 self_eval 的 missing_aspects 和原 query 构造更精准的 web 查询。
    并上商品名，提高联网搜索命中率。
    """
    import re
    base = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    detail = state.get("self_eval_detail") or {}
    missing = detail.get("missing_aspects") or []
    item_names = state.get("item_names") or []

    parts = []
    if item_names:
        parts.append(" ".join(item_names[:1]))
    parts.append(base)
    if missing:
        # missing_aspects 可能是任意字符串，取前2条拼到查询里，截断控制长度
        miss_text = " ".join(str(m) for m in missing[:2])
        miss_text = re.sub(r"[\s,，;；]+", " ", miss_text).strip()
        if miss_text:
            parts.append(miss_text)
    return " ".join(p for p in parts if p)[:200]


def _parse_mcp_result(result) -> list:
    docs = []
    if not result or getattr(result, "isError", False) or not result.content:
        return docs
    raw_text = result.content[0].text
    try:
        data = json.loads(raw_text)
        pages = data.get("pages") or []
        for item in pages:
            snippet = (item.get("snippet") or "").strip()
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if not snippet:
                continue
            docs.append({"title": title, "url": url, "snippet": snippet})
    except json.JSONDecodeError:
        logger.error(f"[retry_web] MCP 结果解析失败: {(raw_text or '')[:100]}")
    return docs


def node_retry_web(state: QueryGraphState) -> QueryGraphState:
    """
    Self_eval 判定 insufficient_retry 后触发：强制补一次联网搜索，
    把结果写回 web_search_docs，并把 retrieval_plan.use_web 置为 True，
    自增 self_eval_retry_count，随后重新走 RRF / Rerank / self_eval。
    """
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)

    logger.info(">>> node_retry_web: self_eval 判定不足，开始补搜 Web")
    add_running_task(session_id, node_name, is_stream)

    query = _build_retry_query(state)
    docs = []
    if query:
        try:
            result = asyncio.run(mcp_call(query))
            docs = _parse_mcp_result(result)
        except Exception as e:
            logger.error(f"[retry_web] 联网补搜异常: {e}", exc_info=True)
    else:
        logger.warning("[retry_web] 构造的 web 查询为空，跳过补搜")

    # 强制更新检索计划，保证再次进入 RRF 时把 web 结果纳入融合
    plan = dict(state.get("retrieval_plan") or {})
    plan["use_web"] = True
    plan["reason"] = (plan.get("reason") or "") + " | forced_by_self_eval_retry"
    plan["decided_by"] = plan.get("decided_by", "llm") + "+retry"
    state["retrieval_plan"] = plan

    # 合并 web 结果：保留已有结果，追加本轮补搜（去重）
    existing = list(state.get("web_search_docs") or [])
    seen_urls = {d.get("url") for d in existing if d.get("url")}
    for d in docs:
        if d.get("url") and d["url"] in seen_urls:
            continue
        existing.append(d)
    state["web_search_docs"] = existing

    # 自增重试计数，防止无限重搜
    state["self_eval_retry_count"] = int(state.get("self_eval_retry_count", 0)) + 1
    # 重置上一次评估状态，避免残留影响下次路由
    state["self_eval_status"] = ""
    state["self_eval_detail"] = {}

    logger.info(
        f"node_retry_web: retry_count={state['self_eval_retry_count']} "
        f"web_docs_total={len(existing)} retry_query={query}"
    )
    add_done_task(session_id, node_name, is_stream)
    return state