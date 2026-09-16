"""Decide whether a query requires new evidence or can reuse context."""

import json
import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task
from app.clients.mongo_session_memory_utils import upsert_session_memory


NO_RETRIEVAL_PATTERNS = [
    r"^(谢谢|好的|好|明白了|知道了|收到|ok|OK|不用了|算了|不问了)[。.!！]?$",
    r".*(你能干什么|你可以干什么|你支持什么|你能查什么|你有哪些资料|怎么使用你).*",
    r".*(这是啥系统|你是什么|你是谁).*",
]
TOPIC_SHIFT_KEYWORDS = [
    "故障", "报错", "失败", "连不上", "通讯不上", "卡纸", "不对", "无法", "不能",
    "配置", "安装", "参数", "规格", "默认", "地址", "登录", "重置", "恢复出厂",
    "图片", "图示", "接线", "连接", "组网", "固件", "版本",
]
CONTEXT_REUSE_PATTERNS = [
    r".*(第一步|第二步|下一步|上一步|刚才|上面|前面|这句话|这段|总结|简单点|详细点|展开说|什么意思).*",
]

CAPABILITY_ANSWER = (
    "我主要用于回答已导入产品文档相关问题，可以查询设备配置、参数规格、安装连接、"
    "故障排查、使用步骤和文档中的图片说明。请提供具体产品型号和问题。"
)

ACK_ANSWER = "好的，你可以继续问具体产品的配置、参数、安装连接或故障排查问题。"


def _match_no_retrieval_answer(query: str) -> str:
    clean_query = (query or "").strip()
    for pattern in NO_RETRIEVAL_PATTERNS:
        if re.match(pattern, clean_query):
            if clean_query.lower() in {"谢谢", "好的", "好", "明白了", "知道了", "收到", "ok"}:
                return ACK_ANSWER
            return CAPABILITY_ANSWER
    return ""


def _parse_llm_json(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    elif raw.startswith("```"):
        raw = raw.replace("```", "").strip()
    return json.loads(raw)


def _recent_evidence_docs(history: list, max_docs: int = 6) -> list:
    docs = []
    for msg in reversed(history or []):
        for doc in msg.get("evidence_docs") or []:
            if doc:
                docs.append(doc)
                if len(docs) >= max_docs:
                    return docs
    return docs


def _latest_assistant_message(history: list) -> dict:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant" and (msg.get("text") or "").strip():
            return msg
    return {}


def _has_recent_answer(history: list) -> bool:
    return any((msg.get("role") == "assistant" and (msg.get("text") or "").strip()) for msg in reversed(history or []))


def _contains_topic_shift(query: str) -> bool:
    return any(keyword in query for keyword in TOPIC_SHIFT_KEYWORDS)


def _looks_context_reuse_query(query: str) -> bool:
    return any(re.match(pattern, query) for pattern in CONTEXT_REUSE_PATTERNS)


def _merge_evidence(bundle_evidence: list, history_evidence: list) -> list:
    """Merge and deduplicate bounded evidence from context and history."""
    seen_ids = set()
    merged = []
    for doc in (bundle_evidence or []) + (history_evidence or []):
        cid = doc.get("chunk_id")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)
        merged.append(doc)
        if len(merged) >= 6:
            break
    return merged


def _build_evidence_preview(evidence_docs: list) -> str:
    lines = []
    for idx, doc in enumerate(evidence_docs[:6], start=1):
        title = doc.get("title") or doc.get("file_title") or ""
        canonical_name = doc.get("canonical_name") or ""
        text = doc.get("text_preview") or doc.get("text") or ""
        lines.append(f"[{idx}] title={title} product={canonical_name}\n{text[:500]}")
    return "\n\n".join(lines)


def _llm_decide_context_sufficiency(state: QueryGraphState, evidence_docs: list) -> dict:
    query = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    original_query = (state.get("original_query") or "").strip()
    history = state.get("history") or []
    history_text = "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('text', '')} item_names={msg.get('item_names') or []}"
        for msg in history[-6:]
    )
    evidence_text = _build_evidence_preview(evidence_docs)
    system_prompt = (
        "你是产品文档问答系统的上下文充分性路由器。"
        "你的任务是判断当前问题是否需要新增检索，还是可以复用历史证据回答。"
        "必须只输出 JSON，不要输出解释文本。"
    )
    human_prompt = f"""
请判断当前问题的 doc_need_route：
- need_retrieval：需要查新的文档证据。适用于切换产品、切换主题、问新故障、新参数、新章节、新操作。
- answer_from_context：历史答案或上一轮 evidence 已足够回答当前追问。适用于"第一步是什么、简单点、这句话什么意思、总结一下"等追问。（对外路由会映射为 reuse_context_evidence）
- direct_answer：感谢、结束、能力说明、反馈类短回复，不需要文档证据。
- clarification：缺少产品或问题信息，需要用户补充。

返回 JSON：
{{
  "doc_need_route": "answer_from_context",
  "direct_answer": "",
  "confidence": 0.0,
  "reason": "history_evidence_sufficient"
}}

当前原始问题：
{original_query}

当前重写问题：
{query}

已确认商品：
{state.get("item_names") or []}

历史对话：
{history_text or "无"}

上一轮 evidence（含 session_memory 中的持久化证据）：
{evidence_text or "无"}
"""
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        result = _parse_llm_json(getattr(response, "content", ""))
        route = result.get("doc_need_route")
        confidence = float(result.get("confidence") or 0)
        if route == "answer_from_context" and confidence < 0.75:
            return {
                "doc_need_route": "need_retrieval",
                "direct_answer": "",
                "doc_need_reason": f"low_confidence_context_reuse:{confidence:.2f}",
                "doc_need_confidence": confidence,
            }
        if route in {"need_retrieval", "answer_from_context", "direct_answer", "clarification"}:
            return {
                "doc_need_route": route,
                "direct_answer": (result.get("direct_answer") or "").strip(),
                "doc_need_reason": (result.get("reason") or "llm_context_sufficiency").strip(),
                "doc_need_confidence": confidence,
            }
    except Exception as e:
        logger.error(f"Context sufficiency decision failed; routing to retrieval: {e}", exc_info=True)

    return {
        "doc_need_route": "need_retrieval",
        "direct_answer": "",
        "doc_need_reason": "llm_context_sufficiency_failed",
    }


def decide_doc_need(state: QueryGraphState) -> dict:
    query = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    original_query = (state.get("original_query") or "").strip()
    item_names = state.get("item_names") or []
    history = state.get("history") or []

    # Prefer bounded evidence assembled by the context node.
    context_bundle = state.get("context_bundle") or {}
    bundle_evidence = context_bundle.get("recent_evidence_docs") or []

    if _looks_context_reuse_query(original_query) and _has_recent_answer(history):
        latest_assistant = _latest_assistant_message(history)
        latest_evidence = latest_assistant.get("evidence_docs") or []
        if latest_evidence:
            result = _llm_decide_context_sufficiency(state, latest_evidence)
            if result["doc_need_route"] == "answer_from_context":
                result["context_evidence_docs"] = latest_evidence
                result["doc_need_route"] = "reuse_context_evidence"
            return result
        return {
            "doc_need_route": "direct_answer",
            "direct_answer": "上一轮回答没有可引用的文档证据；系统未在已导入产品文档中找到直接相关内容。",
            "doc_need_reason": "latest_answer_without_evidence",
        }

    history_evidence = _recent_evidence_docs(history)
    evidence_docs = _merge_evidence(bundle_evidence, history_evidence)

    direct_answer = _match_no_retrieval_answer(original_query) or _match_no_retrieval_answer(query)
    if direct_answer:
        return {
            "doc_need_route": "direct_answer",
            "direct_answer": direct_answer,
            "doc_need_reason": "product_related_direct_answer",
        }

    if not item_names:
        return {
            "doc_need_route": "clarification",
            "direct_answer": "请补充一下具体产品型号或设备名称，我再帮你查文档。",
            "doc_need_reason": "missing_confirmed_item",
        }

    if evidence_docs:
        result = _llm_decide_context_sufficiency(state, evidence_docs)
        if result["doc_need_route"] == "answer_from_context":
            result["context_evidence_docs"] = evidence_docs
            result["doc_need_route"] = "reuse_context_evidence"
        return result

    return {
        "doc_need_route": "need_retrieval",
        "direct_answer": "",
        "doc_need_reason": "need_document_evidence",
    }


def node_doc_need_router(state: QueryGraphState) -> QueryGraphState:
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)

    logger.info("node_doc_need_router started")
    add_running_task(session_id, node_name, is_stream)

    result = decide_doc_need(state)
    state["doc_need_route"] = result["doc_need_route"]
    state["doc_need_reason"] = result.get("doc_need_reason", "")
    state["doc_need_confidence"] = result.get("doc_need_confidence", 0)
    state["context_evidence_docs"] = result.get("context_evidence_docs", [])

    if result.get("direct_answer"):
        state["direct_answer"] = result["direct_answer"]
        state["answer"] = result["direct_answer"]

    logger.info(
        f"node_doc_need_router: route={state['doc_need_route']}, "
        f"reason={result.get('doc_need_reason')}"
    )
    upsert_session_memory(session_id, {
        "last_route_meta": {
            "query_route": state.get("query_route", ""),
            "doc_need_route": state.get("doc_need_route", ""),
            "doc_need_reason": state.get("doc_need_reason", ""),
            "doc_need_confidence": state.get("doc_need_confidence", 0),
            "retrieval_plan": state.get("retrieval_plan") or {},
            "rewritten_query": state.get("rewritten_query", ""),
            "item_names": state.get("item_names") or [],
        }
    })
    add_done_task(session_id, node_name, is_stream)
    return state
