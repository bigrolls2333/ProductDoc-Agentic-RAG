"""Resolve product references before document retrieval."""

import sys
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.clients.mongo_session_memory_utils import get_session_memory, upsert_session_memory, clear_pending_clarification
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def _normalize_name_key(value: str) -> str:
    compact = "".join(str(value or "").split())
    return compact.upper()


def _memory_item_name(item) -> str:
    if isinstance(item, dict):
        return _normalize_name_key(
            item.get("canonical_name")
            or item.get("name")
            or item.get("alias_name")
            or item.get("item_name")
            or ""
        )
    return _normalize_name_key(item)


def _structured_confirmed_items(names: List[str]) -> List[Dict[str, Any]]:
    structured = []
    for name in names or []:
        normalized = _normalize_name_key(name)
        if normalized:
            structured.append({
                "product_id": "",
                "canonical_name": normalized,
                "alias_name": "",
                "brand": "",
                "model_code": "",
                "category": "",
                "confidence": 1.0,
                "source": "node_item_name_confirm",
            })
    return structured


def _sanitize_for_checkpoint(value):
    """Convert external identifiers into checkpoint-serializable values."""
    if isinstance(value, list):
        return [_sanitize_for_checkpoint(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_for_checkpoint(item) for key, item in value.items()}
    if value.__class__.__name__ == "ObjectId":
        return str(value)
    return value


def _recent_history_item_names(history: List[Dict], limit: int = 3) -> List[str]:
    names = []
    for msg in reversed(history or []):
        for item in msg.get("item_names") or []:
            if item and item not in names:
                names.append(item)
                if len(names) >= limit:
                    return names
    return names


def _classify_history_item_reference(query: str, history: List[Dict]) -> Dict[str, Any]:
    history_item_names = _recent_history_item_names(history)
    if not history_item_names:
        return {"reuse_history_items": False, "item_names": [], "reason": "no_history_items"}

    history_text = ""
    for msg in (history or [])[-6:]:
        history_text += (
            f"{msg.get('role', 'unknown')}: {msg.get('text', '')} "
            f"item_names={msg.get('item_names') or []}\n"
        )

    prompt = f"""
你是产品文档问答系统的商品指代判断器。
当前用户问题没有明确抽取出商品名，请结合历史判断是否应该沿用上一轮已确认商品。

只能输出 JSON：
{{
  "reuse_history_items": true,
  "switch_product": false,
  "item_names": ["历史中的标准商品名"],
  "reason": ""
}}

判断原则：
- 用户说“那第一步是什么、简单点、这句话什么意思、总结一下、它支持吗”这类继续追问，且没有换产品迹象，可以沿用历史商品。
- 用户说“那华为的呢、另一个呢、不是这个、换一个、其他型号呢”这类换对象或否定当前商品，不要沿用历史商品。
- 历史商品不清楚或用户当前指代不清楚，不要沿用历史商品。

历史可用商品：
{history_item_names}

历史对话：
{history_text or "无"}

当前问题：
{query}
"""
    try:
        client = get_llm_client(json_mode=True)
        response = client.invoke([HumanMessage(content=prompt)])
        raw = getattr(response, "content", "{}")
        result = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
        if result.get("reuse_history_items") and not result.get("switch_product"):
            item_names = [
                item for item in result.get("item_names", [])
                if item in history_item_names
            ] or history_item_names[:1]
            return {
                "reuse_history_items": True,
                "item_names": item_names,
                "reason": result.get("reason", "llm_reuse_history_items"),
            }
    except Exception as e:
        logger.error(f"Historical product resolution failed; ignoring prior product context: {e}", exc_info=True)

    return {"reuse_history_items": False, "item_names": [], "reason": "do_not_reuse_history_items"}


def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """Extract product names and produce a standalone rewritten query."""
    logger.info("Product resolution: extracting query context")

    client = get_llm_client(json_mode=True)

    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"
    
    logger.debug(f"Product resolution: history context length={len(history_text)}")

    try:
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
        logger.debug(f"Product resolution: prompt loaded, length={len(prompt)}")
    except Exception as e:
        logger.error(f"Product resolution: prompt loading failed: {e}")
        return {"item_names": [], "rewritten_query": query}

    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]

    try:
        logger.debug("Product resolution: invoking extraction model")
        response = client.invoke(messages)
        content = response.content

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        
        result = json.loads(content)
        
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
        result["item_names"] = list(dict.fromkeys([
            normalized
            for item in result.get("item_names", [])
            if (normalized := _normalize_name_key(item))
        ]))
            
        logger.info(f"Product resolution: extracted_items={result['item_names']}")
        return result

    except Exception as e:
        logger.error(f"Product resolution: extraction failed: {e}")
        return {"item_names": [], "rewritten_query": query}


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    """Search the product-name collection using hybrid embeddings."""
    normalized_item_names = list(dict.fromkeys([
        normalized
        for item in item_names
        if (normalized := _normalize_name_key(item))
    ]))
    logger.info(f"Product resolution: searching items={normalized_item_names}")
    results = []
    
    client = get_milvus_client()
    if not client:
        logger.error("Product resolution: Milvus client unavailable")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Product resolution: ITEM_NAME_COLLECTION is not configured")
        return results

    try:
        logger.debug("Product resolution: generating embeddings")
        embeddings = generate_embeddings(normalized_item_names)
        logger.debug(f"Product resolution: searching collection={collection_name}")

        for i, name in enumerate(normalized_item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2), 
                    limit=5,
                    norm_score=True,
                    output_fields=["alias_name", "item_name", "canonical_name", "brand", "category", "model_code", "alias_type"]
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        alias_name = _normalize_name_key(entity.get("alias_name") or entity.get("item_name"))
                        canonical_name = _normalize_name_key(entity.get("canonical_name") or alias_name)
                        score = hit.get("distance")
                        
                        if alias_name:
                            matches.append({
                                "alias_name": alias_name,
                                "item_name": alias_name,  # Preserve the field for downstream compatibility.
                                "canonical_name": canonical_name,
                                "brand": entity.get("brand", ""),
                                "category": entity.get("category", ""),
                                "model_code": entity.get("model_code", ""),
                                "alias_type": entity.get("alias_type", ""),
                                "score": score
                            })
                            logger.debug(f"Product resolution match: {alias_name} -> {canonical_name}, score={score:.4f}")

                results.append({
                    "extracted_name": name,
                    "matches": matches
                })
                logger.info(f"Product resolution: item={name}, matches={len(matches)}")

            except Exception as inner_e:
                logger.error(f"Product resolution failed for item={name}: {inner_e}")
                results.append({"extracted_name": name, "matches": []})

    except Exception as e:
        logger.error(f"Product resolution search failed: {e}")

    return results


def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    """Convert product-name matches into confirmed names or clarification options."""
    logger.info("Product resolution: aligning candidates")
    
    confirmed_item_names = []
    options = []

    for res in query_results:
        extracted_name = _normalize_name_key(res.get("extracted_name", ""))
        matches = res.get("matches", []) or []
        
        if not matches:
            logger.info(f"Product resolution: no matches for item={extracted_name}")
            continue

        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        top_matches_log = ", ".join(
            [
                f"{m.get('alias_name') or m.get('item_name')}=>{m.get('canonical_name') or m.get('alias_name') or m.get('item_name')}({m['score']:.3f})"
                for m in matches[:3]
            ]
        )
        logger.debug(f"Product resolution: top matches for {extracted_name}: {top_matches_log}")

        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]
        high_canonical_names = list(dict.fromkeys([
            m.get("canonical_name") or m.get("alias_name") or m.get("item_name") for m in high
        ]))
        mid_canonical_names = list(dict.fromkeys([
            m.get("canonical_name") or m.get("alias_name") or m.get("item_name") for m in mid
        ]))

        if len(high_canonical_names) == 1:
            confirmed_name = high_canonical_names[0]
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Product resolution: confirmed high-confidence match={confirmed_name}")
            continue

        if len(high) > 1:
            picked = None
            if extracted_name:
                for m in high:
                    if (
                        _normalize_name_key(m.get("alias_name")) == extracted_name
                        or _normalize_name_key(m.get("item_name")) == extracted_name
                        or _normalize_name_key(m.get("canonical_name")) == extracted_name
                    ):
                        picked = m
                        logger.info(f"Product resolution: exact high-confidence match={picked.get('canonical_name')}")
                        break
            
            if not picked:
                picked = high[0]
                logger.info(f"Product resolution: selected highest-scoring match={picked.get('canonical_name')}")

            confirmed_item_names.append(picked.get("canonical_name") or picked.get("alias_name") or picked.get("item_name"))
            continue

        if len(mid) > 0:
            current_options = mid_canonical_names[:5]
            options.extend(current_options)
            logger.info(f"Product resolution: clarification candidates={current_options}")
            continue
        
        logger.info(f"Product resolution: no match above threshold for item={extracted_name}")

    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }
    logger.info(f"Product resolution alignment completed: {result}")
    return result


def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history: List[Dict], rewritten_query: str) -> Dict:
    """Apply confirmed product names or create a pending clarification."""
    logger.info("Product resolution: applying confirmation result")

    if align_result is None:
        align_result = {}

    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    if confirmed:
        logger.info(f"Product resolution: confirmed_items={confirmed}")

        upsert_session_memory(session_id, {
            "pending_clarification": None,
            "last_confirmed_items": _structured_confirmed_items(confirmed),
        })

        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))

        if ids_to_update:
            logger.info(f"Product resolution: updated history records={len(ids_to_update)}")
            update_message_item_names(ids_to_update, confirmed)

        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    if options:
        logger.info(f"Product resolution: creating clarification candidates={options}")

        upsert_session_memory(session_id, {
            "pending_clarification": {
                "original_query": state.get("original_query", ""),
                "candidates": [{"name": name, "score": 0.0} for name in options[:5]],
                "created_at": time.time(),
            }
        })

        options_str = "、".join(options[:3])
        answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
        state["answer"] = answer
        state["item_names"] = []
        state["pending_clarification_created"] = True
        return state

    logger.info("Product resolution: no confirmed product or candidate")
    state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
    state["item_names"] = []
    return state


def step_7_write_history(state: Dict, session_id: str, history: List[Dict], rewritten_query: str, message_id: str) -> Dict:
    """Persist the final assistant response and enriched user message."""
    logger.info("Product resolution: persisting conversation history")

    if state.get("answer"):
        logger.debug("Product resolution: persisting assistant response")
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            item_names=[]
        )

    logger.debug(f"Product resolution: updating user message id={message_id}")
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )

    return state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """Resolve product names from the current query and session context."""
    logger.info("node_item_name_confirm started")

    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, "node_item_name_confirm", is_stream)

    if state.get("clarification_resolved"):
        logger.info(f"Product resolution: restored_items={state.get('item_names')}")
        clear_pending_clarification(session_id)
        upsert_session_memory(session_id, {
            "last_confirmed_items": _structured_confirmed_items(state.get("item_names") or []),
            "pending_clarification": None,
        })
        save_chat_message(
            session_id=session_id,
            role="user",
            text=state.get("original_query", original_query),
            rewritten_query=state.get("rewritten_query", state.get("original_query", original_query)),
            item_names=state.get("item_names", []),
        )
        state["clarification_resolved"] = False
        add_done_task(session_id, "node_item_name_confirm", is_stream)
        return state

    # Reuse session-level product context before invoking the resolver.
    context_bundle = state.get("context_bundle") or {}
    confirmed_from_memory = context_bundle.get("confirmed_items") or []
    memory_item_names = [
        name for item in confirmed_from_memory
        if (name := _memory_item_name(item))
    ]
    if confirmed_from_memory:
        logger.info(f"Product resolution: context items={memory_item_names}")

    history = _sanitize_for_checkpoint(get_recent_messages(session_id, limit=10))
    logger.debug(f"Product resolution: history_messages={len(history)}")

    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Product resolution: initial user message id={message_id}")

    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)
    used_memory_items = False
    if not item_names:
        if memory_item_names:
            item_names = memory_item_names[:1]
            used_memory_items = True
            logger.info(f"Product resolution: reusing session product={item_names}")
        else:
            reference_result = _classify_history_item_reference(original_query, history)
            if reference_result.get("reuse_history_items"):
                item_names = reference_result.get("item_names", [])
                logger.info(
                    f"Product resolution: reusing history product={item_names}, "
                    f"reason={reference_result.get('reason')}"
                )
            else:
                logger.info(f"Product resolution: history product not reused, reason={reference_result.get('reason')}")

    state["rewritten_query"] = rewritten_query

    align_result = {}

    if used_memory_items:
        align_result = {"confirmed_item_names": item_names, "options": []}
    elif len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Product resolution: no product name extracted; skipping vector search")

    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    final_state["history"] = history

    add_done_task(session_id, "node_item_name_confirm", is_stream)

    logger.info(f"node_item_name_confirm completed: items={final_state.get('item_names')}")
    return final_state
