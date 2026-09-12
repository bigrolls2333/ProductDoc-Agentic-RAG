import sys
import json
import re
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task
from app.utils.eval_flag_utils import is_eval_enabled


# 评估给 LLM 的文档预览条数与单条文本长度
EVAL_TOP_K_PREVIEW = 3
EVAL_TEXT_PREVIEW_CHARS = 300


def _parse_llm_json(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    elif raw.startswith("```"):
        raw = raw.replace("```", "").strip()
    return json.loads(raw)


def _build_evidence_preview(docs: list) -> str:
    lines = []
    for idx, doc in enumerate((docs or [])[:EVAL_TOP_K_PREVIEW], start=1):
        text = (doc.get("text") or doc.get("content") or doc.get("snippet") or "").strip()
        title = doc.get("title") or doc.get("file_title") or ""
        score = doc.get("score")
        source = doc.get("source") or ""
        head = f"[{idx}] source={source} title={title}"
        if score is not None:
            head += f" score={float(score):.3f}"
        lines.append(f"{head}\n{text[:EVAL_TEXT_PREVIEW_CHARS]}")
    return "\n\n".join(lines)


def _normalize_text(value: str) -> str:
    return "".join(str(value or "").split()).upper()


def _model_tokens(item_names: list) -> list:
    tokens = []
    for item in item_names or []:
        normalized = _normalize_text(item)
        for token in re.findall(r"[A-Z0-9]*[A-Z]+\d+[A-Z0-9]*", normalized):
            if len(token) < 4 or token in {"3G", "4G", "5G"}:
                continue
            tokens.append(token)
            if token.startswith("H3C") and len(token) > 3:
                tokens.append(token[3:])
    return list(dict.fromkeys(tokens))


def _doc_text_for_match(doc: dict) -> str:
    parts = [
        doc.get("title"),
        doc.get("file_title"),
        doc.get("canonical_name"),
        doc.get("text"),
        doc.get("content"),
        doc.get("snippet"),
        doc.get("url"),
    ]
    return _normalize_text(" ".join(str(part or "") for part in parts))


def _has_model_specific_evidence(item_names: list, docs: list) -> bool:
    tokens = _model_tokens(item_names)
    if not tokens:
        return True
    for doc in docs or []:
        doc_text = _doc_text_for_match(doc)
        if any(token in doc_text for token in tokens):
            return True
    return False


def _llm_eval_retrieval(state: QueryGraphState) -> Dict:
    """
    让 LLM 评估本轮召回的 reranked_docs 是否充分支撑回答用户问题，
    决定本轮是直接回答、补搜一次 Web，还是无答案澄清。
    """
    query = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    item_names = state.get("item_names") or []
    retrieval_plan = state.get("retrieval_plan") or {}
    reranked_docs = state.get("reranked_docs") or []
    retry_count = state.get("self_eval_retry_count", 0)

    # 没有任何召回结果：直接走澄清，不必调 LLM
    if not reranked_docs:
        return {
            "eval_status": "needs_clarification",
            "confidence": 0.0,
            "reason": "empty_reranked_docs",
            "missing_aspects": ["no_retrieval_result"],
        }

    evidence_text = _build_evidence_preview(reranked_docs)

    system_prompt = (
        "你是产品文档问答系统的 Agent 结果自评器。"
        "你的任务是判断当前召回的文档是否足以支撑对用户问题的回答，并决定下一步动作。"
        "必须只输出 JSON，不要输出解释文本。"
    )
    human_prompt = f"""
请评估当前召回的文档对回答用户问题是否充分，输出 eval_status：

- sufficient：召回内容已经能回答用户问题（包含关键事实、步骤或参数），可以直接生成答案。
- insufficient_retry：召回内容不够，但问题属于时效性/外部信息（如最新固件、价格、发布日期、官网），可以通过联网搜索补一次。
- needs_clarification：召回内容不够且无法通过补搜解决（例如本地手册没有该产品、问的不是手册内容、问题过于模糊），应该告诉用户未找到相关文档并请补充信息。

判定原则：
- 只看前几条 Top 文档是否贴合问题。如果 Top1 明显相关，判 sufficient。
- 如果召回文档明显跑题或为空，且问题确实需要外部时效信息，才判 insufficient_retry。
- 不要为了让答案“更完整”就轻易触发 retry；retry 有且仅有一次机会。
- 当前已是第 {retry_count} 次重试（>=1 时不能再判 insufficient_retry）。

返回 JSON：
{{
  "eval_status": "sufficient",
  "confidence": 0.0,
  "reason": "评估理由",
  "missing_aspects": ["缺失的方面，可为空"]
}}

当前查询：
{query}

已确认商品：
{json.dumps(item_names, ensure_ascii=False) or "无"}

本轮检索计划：
{json.dumps(retrieval_plan, ensure_ascii=False) or "{}"}

召回 Top 文档：
{evidence_text}
"""
    llm = get_llm_client(json_mode=True)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    result = _parse_llm_json(getattr(response, "content", ""))

    status = result.get("eval_status", "")
    if status not in {"sufficient", "insufficient_retry", "needs_clarification"}:
        status = "needs_clarification"

    # 已经用完重试机会，强制作澄清，避免无限重搜
    if status == "insufficient_retry" and retry_count >= 1:
        status = "needs_clarification"
        result["reason"] = (result.get("reason") or "") + " | max_retry_reached"

    if (
        status == "sufficient"
        and retry_count >= 1
        and any((doc.get("source") or "") == "web" for doc in reranked_docs[:EVAL_TOP_K_PREVIEW])
        and not is_eval_enabled("EVAL_DISABLE_MODEL_GUARD")
        and not _has_model_specific_evidence(item_names, reranked_docs[:EVAL_TOP_K_PREVIEW])
    ):
        status = "needs_clarification"
        result["reason"] = (
            (result.get("reason") or "")
            + " | web_evidence_missing_confirmed_model"
        )
        result["missing_aspects"] = list(dict.fromkeys(
            (result.get("missing_aspects") or []) + ["confirmed_model_not_in_web_evidence"]
        ))

    return {
        "eval_status": status,
        "confidence": float(result.get("confidence") or 0.0),
        "reason": (result.get("reason") or "").strip(),
        "missing_aspects": result.get("missing_aspects") or [],
    }


def _fallback_eval(state: QueryGraphState) -> Dict:
    """
    LLM 自评失败时的降级策略：
    - 有 reranked_docs 就算 sufficient，让主链路继续生成答案（基于 rerank 分数的最小判断）
    - 没有就走 needs_clarification
    """
    reranked_docs = state.get("reranked_docs") or []
    if reranked_docs:
        return {
            "eval_status": "sufficient",
            "confidence": 0.5,
            "reason": "llm_eval_failed_fallback_sufficient",
            "missing_aspects": [],
        }
    return {
        "eval_status": "needs_clarification",
        "confidence": 0.0,
        "reason": "llm_eval_failed_fallback_empty",
        "missing_aspects": ["no_retrieval_result"],
    }


CLARIFICATION_FALLBACK_ANSWER = (
    "抱歉，未能在已导入的产品文档中找到与您问题直接相关的内容。"
    "请补充具体产品型号或更准确的故障/配置描述，我会重新为您检索。"
)


def node_self_eval(state: QueryGraphState) -> QueryGraphState:
    """
    Rerank 后的结果自评节点：LLM 判断召回是否充分，
    返回 sufficient / insufficient_retry / needs_clarification，
    供 main_graph 的条件路由决定是直接回答、补搜 Web 还是澄清。
    """
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)

    logger.info(">>> node_self_eval: 开始评估召回充分性")
    add_running_task(session_id, node_name, is_stream)

    try:
        if is_eval_enabled("EVAL_DISABLE_SELF_EVAL"):
            detail = _fallback_eval(state)
            detail["reason"] = "eval_disable_self_eval"
        else:
            detail = _llm_eval_retrieval(state)
    except Exception as e:
        logger.error(f"LLM 结果自评失败，降级为兜底策略: {e}", exc_info=True)
        detail = _fallback_eval(state)

    if detail["eval_status"] == "insufficient_retry" and is_eval_enabled("EVAL_DISABLE_SELF_EVAL_RETRY"):
        detail = dict(detail)
        detail["eval_status"] = "needs_clarification"
        detail["reason"] = (detail.get("reason") or "") + " | eval_disable_self_eval_retry"

    state["self_eval_detail"] = detail
    state["self_eval_status"] = detail["eval_status"]

    # 澄清兜底：把拒答写入 answer，让下游 answer_output 走“已有答案”分支，
    # 避免在召回不充分时仍基于噪声文档生成幻觉答案
    if detail["eval_status"] == "needs_clarification":
        state["answer"] = CLARIFICATION_FALLBACK_ANSWER
        state["direct_answer"] = CLARIFICATION_FALLBACK_ANSWER

    logger.info(
        f"node_self_eval: status={detail['eval_status']} "
        f"confidence={detail['confidence']:.2f} reason={detail['reason']}"
    )
    add_done_task(session_id, node_name, is_stream)
    return state
