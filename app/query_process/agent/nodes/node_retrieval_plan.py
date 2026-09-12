import re
import os
import re
import sys
import json
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task
from app.utils.eval_flag_utils import eval_mode, is_eval_enabled


# ======================================================================
# 旧的规则逻辑：保留作为 LLM 冱策失败时的 fallback
# ======================================================================
DOC_RETRIEVAL_KEYWORDS = [
    "怎么", "如何", "步骤", "配置", "安装", "接入", "连接", "设置", "重置",
    "参数", "规格", "支持", "默认", "地址", "指示灯", "故障", "报错",
    "异常", "不能", "失败", "图片", "图示", "表格", "页面", "菜单",
]

HYDE_HINT_KEYWORDS = [
    "怎么办", "怎么弄", "咋办", "不能用", "连不上", "失败", "异常", "故障",
    "一直闪", "没反应", "不行", "无法", "是什么",
]

WEB_HINT_KEYWORDS = [
    "最新", "官网", "价格", "报价", "购买", "售后", "固件", "版本",
    "发布日期", "现在", "目前", "哪里买", "保修",
]

MODEL_CODE_PATTERN = re.compile(r"\b[A-Za-z]{1,8}[- ]?\d{2,8}[A-Za-z0-9-]*\b")


def _contains_any(text: str, keywords) -> bool:
    lowered = (text or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _rule_based_plan(state: QueryGraphState) -> Dict:
    """
    规则兜底版检索计划（LLM 失败时使用）。
    保留原有行为：默认开本地向量检索，HyDE 用于模糊/短查询，Web 用于时效性查询。
    """
    query = (state.get("rewritten_query") or state.get("original_query") or "").strip()

    use_web = _contains_any(query, WEB_HINT_KEYWORDS)
    use_hyde = (
        len(query) <= 16
        or _contains_any(query, HYDE_HINT_KEYWORDS)
        or (not _contains_any(query, DOC_RETRIEVAL_KEYWORDS) and bool(MODEL_CODE_PATTERN.search(query)))
    )

    reason = "rule_fallback_precise_doc_query"
    if use_web:
        reason = "rule_fallback_external_or_fresh_info_query"
    elif use_hyde:
        reason = "rule_fallback_ambiguous_or_short_doc_query"

    return {
        "use_embedding": True,
        "use_hyde": use_hyde,
        "use_web": use_web,
        "reason": reason,
        "decided_by": "rule",
    }


def _parse_llm_json(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    elif raw.startswith("```"):
        raw = raw.replace("```", "").strip()
    return json.loads(raw)


def _build_signal_hints(query: str) -> dict:
    """
    给 LLM 的规则信号提示（参考 node_query_intent 的做法），
    让 LLM 决策时能兼顾关键词事实，避免漏掉明显特征。
    """
    return {
        "matched_web_hints": [kw for kw in WEB_HINT_KEYWORDS if kw.lower() in query.lower()],
        "matched_hyde_hints": [kw for kw in HYDE_HINT_KEYWORDS if kw.lower() in query.lower()],
        "query_length": len(query),
        "has_model_code": bool(MODEL_CODE_PATTERN.search(query)),
    }


def _llm_decide_retrieval_plan(state: QueryGraphState) -> Dict:
    """
    让 LLM 作为 Agent 的工具选择器，决定本轮召回开启哪些分支。
    输出严格 JSON，失败时由调用方走规则兜底。

    改造后新增：注入 product_memory_hints 作为历史检索经验参考。
    """
    query = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    item_names = state.get("item_names") or []
    history = state.get("history") or []
    history_text = "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('text', '')}"
        for msg in history[-4:]
    )

    # 【新】读取 ProductMemory 命中结果，用于 query expansion
    product_hints = {} if is_eval_enabled("EVAL_DISABLE_PRODUCT_MEMORY") else (state.get("product_memory_hints") or {})
    common_questions = product_hints.get("common_questions") or []
    product_hint_text = ""
    if common_questions:
        product_hint_text = "\n".join(
            f"- 高频问题: {cq.get('query_pattern', '')} | 历史检索策略: {cq.get('route_meta', {}).get('retrieval_plan', {})}"
            for cq in common_questions[:3]
        )

    signal = _build_signal_hints(query)

    system_prompt = (
        "你是产品文档问答系统的 Agent 工具选择器。"
        "你的任务是根据用户问题、已确认商品和历史上下文，决定本轮检索开启哪些召回工具。"
        "必须只输出 JSON，不要输出解释文本。"
    )
    human_prompt = f"""
请为当前查询选择召回工具组合，返回 use_embedding / use_hyde / use_web 三个布尔字段。

工具说明：
- use_embedding（本地向量检索）：默认 True。适合查询产品文档里的配置、参数、规格、安装、连接、故障、步骤、图片等事实性内容。
- use_hyde（假设性文档增强召回）：适合短查询、模糊查询、口语化故障描述（如“连不上”“一直闪”“咋办”）。问题越长越具体，越不需要 HyDE。
- use_web（联网搜索）：仅当问题需要时效性/外部信息时才开，例如最新固件版本、官方价格、发布日期、购买渠道、官方售后。能从本地手册查到的内容不要开 Web。

判定原则：
- 只要有产品文档相关问题，use_embedding 必须为 True。
- 一个工具不要无必要地开启，避免浪费成本和引入噪声。
- 规则信号只是参考，最终由你判断。

返回 JSON：
{{
  "use_embedding": true,
  "use_hyde": false,
  "use_web": false,
  "reason": "选择该组合的原因"
}}

规则信号：
{json.dumps(signal, ensure_ascii=False)}

已确认商品：
{json.dumps(item_names, ensure_ascii=False) or "无"}

历史对话：
{history_text or "无"}

当前查询：
{query}

{("该产品历史上高频问题及检索策略参考：\n" + product_hint_text) if product_hint_text else ""}
"""
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        result = _parse_llm_json(getattr(response, "content", ""))

        plan = {
            "use_embedding": bool(result.get("use_embedding", True)),
            "use_hyde": bool(result.get("use_hyde", False)),
            "use_web": bool(result.get("use_web", False)),
            "reason": (result.get("reason") or "llm_tool_selection").strip(),
            "decided_by": "llm",
        }
        # 本地向量检索是默认手段，LLM 漏选时强制保底
        if not plan["use_embedding"] and not plan["use_web"] and not plan["use_hyde"]:
            plan["use_embedding"] = True
            plan["reason"] = (plan["reason"] + " | fallback: no tool selected, force embedding")
        return plan
    except Exception as e:
        logger.error(f"LLM 检索计划决策失败，降级为规则兜底: {e}", exc_info=True)
        raise


def build_retrieval_plan(state: QueryGraphState) -> Dict:
    """
    构建检索计划：优先用 LLM 作为 Agent 工具选择器决策，
    LLM 失败时降级为规则逻辑，保证主链路稳定。
    """
    force_planner = eval_mode("EVAL_FORCE_PLANNER")
    if force_planner == "embedding_only":
        plan = {
            "use_embedding": True,
            "use_hyde": False,
            "use_web": False,
            "reason": "eval_force_embedding_only",
            "decided_by": "eval_force",
        }
    elif force_planner == "rule":
        plan = _rule_based_plan(state)
        plan["reason"] = (plan.get("reason") or "") + " | eval_force_rule_planner"
        plan["decided_by"] = "rule_eval_force"
    else:
        try:
            plan = _llm_decide_retrieval_plan(state)
        except Exception:
            plan = _rule_based_plan(state)

    if is_eval_enabled("EVAL_DISABLE_HYDE"):
        plan["use_hyde"] = False
        plan["reason"] = (plan.get("reason") or "") + " | eval_disable_hyde"
    if state.get("_no_web_override") or is_eval_enabled("EVAL_DISABLE_WEB"):
        plan["use_web"] = False
        plan["reason"] = (plan.get("reason") or "") + " | eval_disable_web"
    return plan


def node_retrieval_plan(state: QueryGraphState) -> QueryGraphState:
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)

    logger.info(">>> node_retrieval_plan: 开始处理（Agent 工具选择）")
    add_running_task(session_id, node_name, is_stream)

    plan = build_retrieval_plan(state)
    state["retrieval_plan"] = plan

    logger.info(
        f"node_retrieval_plan: decided_by={plan.get('decided_by')} "
        f"plan={plan}"
    )
    add_done_task(session_id, node_name, is_stream)
    return state
