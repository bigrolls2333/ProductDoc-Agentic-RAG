import json
import re
import sys
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.clients.mongo_history_utils import get_recent_messages
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


PRODUCT_DOC_KEYWORDS = [
    "怎么", "如何", "配置", "安装", "连接", "接入", "参数", "规格", "故障", "报错",
    "支持", "手册", "说明书", "设备", "型号", "产品", "图片", "图示", "步骤",
    "重置", "登录", "管理", "组网", "无线", "网关", "控制器", "SIM", "WiFi",
    "3G", "4G", "5G", "VLAN", "IP", "LAN", "WAN",
]
PRODUCT_BRAND_KEYWORDS = [
    "华为", "HUAWEI", "H3C", "联想", "LENOVO", "兄弟", "BROTHER", "小米", "XIAOMI",
    "迅饶", "AOLYNK", "RS PRO", "RSPRO",
]
PRODUCT_ENTITY_KEYWORDS = [
    "平板", "打印机", "电脑", "笔记本", "台式机", "显示器", "网关", "路由器", "交换机",
    "控制器", "无线", "设备", "万用表", "烫金机", "手册", "指南", "说明书",
]
CLEAR_NON_DOC_KEYWORDS = [
    "写代码", "编程题", "股票", "基金", "天气", "新闻", "电影", "小说", "情书", "作文",
    "旅游", "菜谱", "减肥", "星座", "算命", "历史事件",
]
GENERIC_TECH_KEYWORDS = [
    "HTTP", "HTTPS", "404", "500", "PYTHON", "JAVA", "JAVASCRIPT", "SQL", "LINUX",
    "WINDOWS", "DOCKER", "K8S", "CUDA", "PYTORCH", "RTX", "NVIDIA", "GITHUB",
]

OUT_OF_SCOPE_KEYWORDS = [
    "天气", "股票", "基金", "新闻", "电影", "小说", "情书", "作文", "翻译",
    "旅游", "菜谱", "减肥", "星座", "算命", "历史事件", "写代码", "编程题",
]

INCOMPLETE_PATTERNS = [
    r"^(这个|那个|它|刚才那个|上面那个|前面那个)(怎么|如何|能不能|可以|是啥|是什么|呢|咋).*$",
    r"^(怎么弄|怎么搞|咋办|然后呢|这个呢|那它呢)$",
]
FOLLOW_UP_PATTERNS = [
    r"^(那|那么)?(这个|那个|它|刚才那个|上面那个|前面那个|另一个|其他的|别的).*$",
    r"^(那|那么)?[\u4e00-\u9fa5A-Za-z0-9\- ]{1,20}的呢[？?]?$",
    r"^(那|那么)?(怎么|如何|可以|能不能|支持|第一步|下一步|还有|再说|简单点|详细点).*$",
    r".*(太长了|简单点|详细点|没看懂|不对|重新说|总结一下).*",
    r".*(谢谢|不用了|算了|不问了).*",
]
MODEL_CODE_PATTERN = re.compile(r"\b[A-Za-z]{1,8}[- ]?\d{2,8}[A-Za-z0-9-]*\b")

DIRECT_ANSWERS = {
    "clarification": "请补充一下具体产品型号或你想查询的问题。",
    "smalltalk": "我主要用于回答已导入产品文档相关问题，你可以问设备配置、参数、故障排查或使用步骤。",
    "thanks_or_end": "好的，如需继续查询产品文档，可以再告诉我具体设备和问题。",
    "abuse_or_invalid": "我可以帮你查询产品手册里的配置、参数和故障处理内容。请描述具体设备和问题。",
    "out_of_scope": "这个问题不属于当前产品文档知识库范围。你可以问已导入设备的配置、参数、安装、连接或故障排查问题。",
}


def _normalize_query(query: str) -> str:
    return (query or "").strip()


def _is_invalid_query(query: str) -> bool:
    if not query:
        return True
    if len(query) <= 1:
        return True
    if re.fullmatch(r"[\W_]+", query, flags=re.UNICODE):
        return True
    return False


def _has_recent_item_context(history: List[Dict]) -> bool:
    for msg in reversed(history or []):
        item_names = msg.get("item_names") or []
        if item_names:
            return True
    return False


def _is_incomplete_reference(query: str) -> bool:
    return any(re.match(pattern, query) for pattern in INCOMPLETE_PATTERNS)


def _is_follow_up_query(query: str) -> bool:
    return any(re.match(pattern, query) for pattern in FOLLOW_UP_PATTERNS)


def _contains_any(query: str, keywords: List[str]) -> bool:
    query_lower = query.lower()
    return any(keyword.lower() in query_lower for keyword in keywords)


def _looks_like_product_entity_query(query: str) -> bool:
    if _contains_any(query, CLEAR_NON_DOC_KEYWORDS):
        return False
    has_brand = _contains_any(query, PRODUCT_BRAND_KEYWORDS)
    has_entity = _contains_any(query, PRODUCT_ENTITY_KEYWORDS)
    if has_brand and has_entity:
        return True
    if has_brand and MODEL_CODE_PATTERN.search(query):
        return True
    if has_entity and MODEL_CODE_PATTERN.search(query):
        return True
    compact = "".join(query.split())
    if MODEL_CODE_PATTERN.fullmatch(compact):
        return True
    return False


def _looks_like_generic_tech_without_product_anchor(query: str) -> bool:
    has_generic_tech = _contains_any(query, GENERIC_TECH_KEYWORDS)
    if not has_generic_tech:
        return False
    has_product_anchor = (
        _contains_any(query, PRODUCT_BRAND_KEYWORDS)
        or _contains_any(query, PRODUCT_ENTITY_KEYWORDS)
        or _looks_like_product_entity_query(query)
    )
    return not has_product_anchor


def _route(route: str, answer_key: str = "") -> Dict[str, str]:
    return {
        "query_route": route,
        "direct_answer": DIRECT_ANSWERS.get(answer_key or route, ""),
    }


def _parse_llm_json(content: str) -> Dict:
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    elif raw.startswith("```"):
        raw = raw.replace("```", "").strip()
    return json.loads(raw)


def _llm_classify_query_intent(query: str, history: List[Dict]) -> Dict[str, str]:
    signal = {
        "matched_product_keywords": [kw for kw in PRODUCT_DOC_KEYWORDS if kw.lower() in query.lower()],
        "matched_out_of_scope_keywords": [kw for kw in OUT_OF_SCOPE_KEYWORDS if kw.lower() in query.lower()],
        "matched_model_code": bool(MODEL_CODE_PATTERN.search(query)),
        "matched_product_brands": [kw for kw in PRODUCT_BRAND_KEYWORDS if kw.lower() in query.lower()],
        "matched_product_entities": [kw for kw in PRODUCT_ENTITY_KEYWORDS if kw.lower() in query.lower()],
        "matched_generic_tech_keywords": [kw for kw in GENERIC_TECH_KEYWORDS if kw.lower() in query.lower()],
        "looks_like_product_entity_query": _looks_like_product_entity_query(query),
        "looks_like_generic_tech_without_product_anchor": _looks_like_generic_tech_without_product_anchor(query),
    }
    history_text = "\n".join(
        (
            f"{msg.get('role', 'unknown')}: {msg.get('text', '')} "
            f"item_names={msg.get('item_names') or []} "
            f"route_meta={msg.get('route_meta') or {}}"
        )
        for msg in (history or [])[-6:]
    )
    system_prompt = (
        "你是产品文档问答系统的查询意图分类器。"
        "你的任务是像真人一样结合当前问题、历史上下文和规则信号，判断用户是否在问已导入产品文档相关问题。"
        "必须只输出 JSON，不要输出解释文本。"
    )
    human_prompt = f"""
请把用户输入分类为以下 intent 之一：
- product_doc_qa：用户在问产品、设备、手册、参数、配置、安装、连接、故障、图片或使用步骤。即使带有抱怨或粗话，只要包含真实产品问题，也归为这个类别。
- smalltalk：问候、闲聊、感谢、询问你是谁等，不需要检索产品文档。
- abuse_or_invalid：纯辱骂、挑衅、无意义输入，没有产品问题。
- out_of_scope：明确不是产品文档知识库范围的问题。

注意：
- 规则信号只作为参考，不是最终答案。
- 型号形态、VLAN、IP、404、Python 3.12、RTX 4060 等 token 不能单独证明它是产品文档问题。
- 用户只输入一个产品实体、品牌+品类、品牌+型号、型号简称，仍然应归为 product_doc_qa，让后续商品确认节点去确认或澄清。
- “华为平板”“华为平板怎么样”“华为平板C5”“HUAWEI PixLab B5”“LA2608”这类应归为 product_doc_qa。
- 如果用户是在问通用编程、写代码、股票、天气、新闻、购买推荐、通用技术科普，应归为 out_of_scope。
- 如果问题既包含产品/设备，又是在问该产品手册里的配置、参数、故障、安装、使用步骤，应归为 product_doc_qa。

返回 JSON 格式：
{{
  "intent": "product_doc_qa",
  "direct_answer": "",
  "confidence": 0.0,
  "reason": ""
}}

规则信号：
{json.dumps(signal, ensure_ascii=False)}

历史对话：
{history_text or "无"}

用户输入：
{query}
"""
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        result = _parse_llm_json(getattr(response, "content", ""))
        intent = result.get("intent", "")
        if intent == "product_doc_qa":
            return _route("product_doc_qa")
        if intent in {"smalltalk", "abuse_or_invalid", "out_of_scope"}:
            direct_answer = (result.get("direct_answer") or "").strip()
            return {
                "query_route": "direct_answer" if intent != "out_of_scope" else "out_of_scope",
                "direct_answer": direct_answer or DIRECT_ANSWERS[intent],
            }
    except Exception as e:
        logger.error(f"意图识别LLM分类失败，降级为直接提示: {e}", exc_info=True)

    return _route("direct_answer", "smalltalk")


def _classify_follow_up_query(query: str, history: List[Dict]) -> Dict[str, str]:
    history_text = "\n".join(
        (
            f"{msg.get('role', 'unknown')}: {msg.get('text', '')} "
            f"item_names={msg.get('item_names') or []}"
        )
        for msg in (history or [])[-6:]
    )
    system_prompt = (
        "你是产品文档问答系统的多轮追问分类器。"
        "你的任务是结合历史对话判断当前短句是否仍然延续产品文档任务。"
        "必须只输出 JSON，不要输出解释文本。"
    )
    human_prompt = f"""
请把用户当前输入分类为 follow_up_type 之一：
- continue_product_doc_task：当前输入是在延续产品文档问题，例如继续问步骤、参数、支持情况、同类产品、品牌补充、故障排查。
- clarification_needed：当前输入依赖指代，但历史不足以确定要问什么产品或问题。
- user_feedback：用户在评价上一轮回答，例如太长、没看懂、不对、重新说。
- thanks_or_end：感谢、结束、不再继续。
- smalltalk_or_meta：闲聊、问你是谁、问系统能力。
- out_of_scope：明确不是产品文档范围。

返回 JSON：
{{
  "follow_up_type": "continue_product_doc_task",
  "direct_answer": ""
}}

历史对话：
{history_text or "无"}

当前输入：
{query}
"""
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        result = _parse_llm_json(getattr(response, "content", ""))
        follow_up_type = (result.get("follow_up_type") or "").strip()
        direct_answer = (result.get("direct_answer") or "").strip()

        if follow_up_type == "continue_product_doc_task":
            return _route("product_doc_qa")
        if follow_up_type == "clarification_needed":
            return _route("clarification")
        if follow_up_type == "out_of_scope":
            return _route("out_of_scope")
        if follow_up_type == "user_feedback":
            if history:
                return _route("product_doc_qa")
            return {
                "query_route": "direct_answer",
                "direct_answer": direct_answer or DIRECT_ANSWERS["smalltalk"],
            }
        if follow_up_type in {"thanks_or_end", "smalltalk_or_meta"}:
            answer_key = "thanks_or_end" if follow_up_type == "thanks_or_end" else "smalltalk"
            return {
                "query_route": "direct_answer",
                "direct_answer": direct_answer or DIRECT_ANSWERS[answer_key],
            }
    except Exception as e:
        logger.error(f"多轮追问LLM分类失败，降级为澄清: {e}", exc_info=True)

    return _route("clarification")


def classify_query_intent(query: str, history: List[Dict]) -> Dict[str, str]:
    query = _normalize_query(query)

    if _is_invalid_query(query):
        return _route("direct_answer", "abuse_or_invalid")

    if _is_incomplete_reference(query):
        return _classify_follow_up_query(query, history)

    if _is_follow_up_query(query):
        return _classify_follow_up_query(query, history)

    if _looks_like_generic_tech_without_product_anchor(query):
        return _route("out_of_scope")

    if _looks_like_product_entity_query(query):
        return _route("product_doc_qa")

    return _llm_classify_query_intent(query, history)


def node_query_intent(state: QueryGraphState) -> QueryGraphState:
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    query = state.get("original_query", "")

    logger.info(">>> node_query_intent: 开始处理")
    add_running_task(session_id, node_name, is_stream)

    # 【改】优先使用 context_bundle 中的历史消息（已截断处理），减少重复 DB 查询
    context_bundle = state.get("context_bundle") or {}
    bundle_history = context_bundle.get("history_messages") or []

    if bundle_history:
        history = bundle_history
        logger.info(f"node_query_intent: 使用 context_bundle 历史消息，数量={len(history)}")
    else:
        history = get_recent_messages(session_id, limit=10)
        logger.info(f"node_query_intent: context_bundle 无历史消息，降级为直接查询 MongoDB，数量={len(history)}")

    result = classify_query_intent(query, history)
    state["query_route"] = result.get("query_route", "product_doc_qa")
    state["direct_answer"] = result.get("direct_answer", "")

    # 保存历史到 state（若 context_bundle 未提供完整格式，则用 MongoDB 数据补充）
    if not state.get("history"):
        state["history"] = history

    logger.info(
        f"node_query_intent: route={state['query_route']}, "
        f"direct_answer={bool(state['direct_answer'])}"
    )
    add_done_task(session_id, node_name, is_stream)
    return state
