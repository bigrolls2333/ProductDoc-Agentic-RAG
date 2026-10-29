"""Generate answers, persist evidence, and publish streamed events."""

import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
from app.utils.query_trace_utils import append_query_trace
from app.clients.mongo_session_memory_utils import upsert_session_memory
import re
import hashlib

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000
EVIDENCE_TEXT_PREVIEW_CHARS = 500


def _summarize_evidence_docs(docs, limit: int = 8):
  evidence_docs = []
  for doc in (docs or [])[:limit]:
    text = (doc.get("text") or doc.get("content") or doc.get("snippet") or "").strip()
    evidence_docs.append({
      "chunk_id": doc.get("chunk_id"),
      "doc_id": doc.get("doc_id"),
      "file_title": doc.get("file_title") or doc.get("title") or "",
      "title": doc.get("title") or "",
      "canonical_name": doc.get("canonical_name") or doc.get("item_name") or "",
      "source": doc.get("source") or ("web" if doc.get("url") else "local"),
      "score": doc.get("score"),
      "url": doc.get("url") or "",
      "text_preview": text[:EVIDENCE_TEXT_PREVIEW_CHARS],
    })
  return evidence_docs


def _should_persist_evidence(state: QueryGraphState) -> bool:
  if state.get("doc_need_route") in ("answer_from_context", "reuse_context_evidence"):
    return True
  self_eval_status = state.get("self_eval_status")
  return not self_eval_status or self_eval_status == "sufficient"


def _docs_from_context_evidence(state: QueryGraphState):
  docs = []
  for doc in state.get("context_evidence_docs") or []:
    text = (doc.get("text_preview") or doc.get("text") or "").strip()
    if not text:
      continue
    docs.append({
      "chunk_id": doc.get("chunk_id"),
      "doc_id": doc.get("doc_id"),
      "file_title": doc.get("file_title") or "",
      "title": doc.get("title") or doc.get("file_title") or "",
      "canonical_name": doc.get("canonical_name") or "",
      "source": doc.get("source") or "history",
      "score": doc.get("score"),
      "url": doc.get("url") or "",
      "text": text,
    })
  return docs

def step_1_check_answer(state) -> bool:
  """Return whether an answer already exists and publish it when streaming."""
  answer = state.get("answer", None)
  is_stream = state.get("is_stream" )
  if answer:
    if is_stream:
      logger.debug("Answer already present; publishing existing response")
      push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    else:
      set_task_result(state["session_id"], "answer", answer)
    return True
  else:
    return False

def step_2_construct_prompt(state: QueryGraphState) -> str:
  """Build the answer prompt from query, bounded history, and ranked evidence."""
  from app.query_process.context_selection import select_context_documents, context_policy, context_per_item_k
  original_query = state.get("original_query", "")
  rewritten_query = state.get("rewritten_query", "")
  question = rewritten_query if rewritten_query else original_query
  history = state.get("history", [])
  item_names = state.get("item_names", [])
  reranked_docs = state.get("reranked_docs") or []
  doc_need_route = state.get("doc_need_route", "")
  if doc_need_route in ("answer_from_context", "reuse_context_evidence"):
    history_docs = _docs_from_context_evidence(state)
    if not history_docs:
      context_bundle = state.get("context_bundle") or {}
      bundle_evidence = context_bundle.get("recent_evidence_docs") or []
      if bundle_evidence:
        history_docs = [{
          "chunk_id": d.get("chunk_id"),
          "file_title": d.get("title", ""),
          "title": d.get("title", ""),
          "canonical_name": d.get("canonical_name", ""),
          "source": "history",
          "score": d.get("score"),
          "text": d.get("text_preview", ""),
        } for d in bundle_evidence]
    if history_docs:
      logger.info(f"Reusing context evidence: route={doc_need_route}, count={len(history_docs)}")
      reranked_docs = history_docs

  policy = context_policy(state)
  candidates = reranked_docs
  if policy != 'ranked' and doc_need_route not in ('answer_from_context', 'reuse_context_evidence'):
    candidates = state.get('context_candidate_docs') or reranked_docs
  included_docs, context_str, selection_trace = select_context_documents(
    candidates, question, policy=policy, budget=MAX_CONTEXT_CHARS,
    item_names=item_names, per_item_k=context_per_item_k(state),
    ranked_k=max(1, len(reranked_docs)), pool_k=50, max_docs=50 if policy == 'ranked' else 20)
  used = len(context_str)
  state["answer_context_docs"] = included_docs
  state["answer_context_trace"] = {
    **selection_trace,
    "context_source": "actual_prompt",
    "budget_chars": MAX_CONTEXT_CHARS,
    "context_chars": len(context_str),
    "included_count": len(included_docs),
    "omitted_count": len(candidates) - len(included_docs),
    "context_sha256": hashlib.sha256(context_str.encode("utf-8")).hexdigest(),
  }


  # Bound each history addition before appending; count only its incremental cost.
  history_parts = []
  for msg in history:
    role, text = msg.get('role'), msg.get('text')
    if role not in ('user', 'assistant') or not text:
      continue
    part = ('用户' if role == 'user' else '助手') + ': ' + text + '\n'
    if used + len(part) > MAX_CONTEXT_CHARS:
      break
    history_parts.append(part)
    used += len(part)
  history_str = ''.join(history_parts) if history_parts else '无历史对话'

  item_names_str = ", ".join(item_names) if item_names else "无指定商品"

  prompt = load_prompt("answer_out",
    context=context_str,
    history=history_str,
    item_names=item_names_str,
    question=question
  )

  logger.debug(f"Answer prompt assembled: length={len(prompt)}")
  state["answer_context_trace"]["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

  return prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
  """Generate an answer through the configured model in stream or batch mode."""
  logger.info("Answer generation started")
  logger.debug(f"Answer prompt length={len(prompt)}")

  llm = get_llm_client()

  session_id = state.get("session_id")
  is_stream = state.get("is_stream")

  if is_stream:
    logger.info(f"Answer generation mode=stream, session_id={session_id}")
    final_text = ""
    try:
      for chunk in llm.stream(prompt):
        delta = getattr(chunk, "content", "") or ""
        if delta:
          final_text += delta
          push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
      
      logger.info(f"Answer generation completed: length={len(final_text)}")

    except Exception as e:
      logger.error(f"Streaming answer generation failed: {e}", exc_info=True)
      push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
      
    state["answer"] = final_text
  else:
    logger.info(f"Answer generation mode=batch, session_id={session_id}")
    try:
      response = llm.invoke(prompt)
      content = response.content
      state["answer"] = content
      set_task_result(session_id, "answer", content)
      logger.info(f"Answer generation completed: length={len(content)}")
    except Exception as e:
      logger.error(f"Batch answer generation failed: {e}", exc_info=True)
      state["answer"] = "抱歉，生成回答时出现错误。"

  return state


def _extract_images_from_docs(docs):
    """Extract and deduplicate image URLs from local and web evidence."""
    images = []
    seen = set()
    if not docs:
        return []
    md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    logger.debug(f"Extracting image URLs from evidence_count={len(docs)}")

    for i, doc in enumerate(docs):
        url = (doc.get("url") or "").strip()
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in seen:
                    logger.debug(f"Evidence image URL found: index={i}, url={url}")
                    seen.add(url)
                    images.append(url)

        text = (doc.get("text") or "").strip()
        if text:
            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    logger.debug(f"Markdown image URL found: index={i}, url={img_url}")
                    seen.add(img_url)
                    images.append(img_url)

    logger.info(f"Evidence image extraction completed: unique_urls={len(images)}")
    return images


def step_4_write_history(state: QueryGraphState, image_urls = None, evidence_docs = None) -> QueryGraphState:
  """Persist the answer, evidence summary, and route metadata."""
  session_id = state.get("session_id", "default")
  answer = (state.get("answer") or "").strip()
  item_names = state.get("item_names") or []
  route_meta = {
    "query_route": state.get("query_route", ""),
    "doc_need_route": state.get("doc_need_route", ""),
    "doc_need_reason": state.get("doc_need_reason", ""),
    "doc_need_confidence": state.get("doc_need_confidence", 0),
    "retrieval_plan": state.get("retrieval_plan") or {},
    "rewritten_query": state.get("rewritten_query", ""),
    "item_names": item_names,
  }

  try:
    if answer:
       save_chat_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query="",
        item_names=item_names,
        image_urls=image_urls,
        evidence_docs=evidence_docs or [],
        route_meta=route_meta,
        message_id=None
      )
  except Exception as e:
    logger.error(f"Conversation history write failed: {e}", exc_info=True)

  return state


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
  """Generate, persist, and publish the final response for a query run."""
  logger.info("node_answer_output started")
  add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

  answer_exists = step_1_check_answer(state)

  if not answer_exists:
    prompt = step_2_construct_prompt(state)
    state["prompt"] = prompt

    step_3_generate_response(state, prompt)

  answer_context_docs = state.get("reranked_docs") or []
  doc_need_route = state.get("doc_need_route", "")
  if doc_need_route in ("answer_from_context", "reuse_context_evidence"):
    answer_context_docs = _docs_from_context_evidence(state)
  if not _should_persist_evidence(state):
    answer_context_docs = []
  image_urls = _extract_images_from_docs(answer_context_docs)
  evidence_docs = _summarize_evidence_docs(answer_context_docs)

  if state.get("answer"):
    logger.debug("Persisting answer and evidence metadata")
    step_4_write_history(state, image_urls=image_urls, evidence_docs=evidence_docs)

    _write_session_memory(state, evidence_docs)

    _maybe_write_product_memory(state, evidence_docs)

  append_query_trace(state, status="completed")

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
  logger.info(f"node_answer_output completed: image_urls={len(image_urls)}")
  if state.get("is_stream"):
    push_to_session(
        state['session_id'],
        SSEEvent.FINAL,
        {
            "answer": state["answer"],
            "status": "completed",
            "image_urls": image_urls,
        }
    )
  
  return state


def _write_session_memory(state: QueryGraphState, evidence_docs: list) -> None:
    """Persist bounded evidence, answer summary, and route metadata."""
    session_id = state.get("session_id", "")
    if not session_id:
        return

    answer = (state.get("answer") or "").strip()
    item_names = state.get("item_names") or []
    evidence_for_memory = evidence_docs or []
    if (
        state.get("doc_need_route") != "reuse_context_evidence"
        and state.get("self_eval_status")
        and state.get("self_eval_status") != "sufficient"
    ):
        evidence_for_memory = []

    updates = {
        "last_evidence_docs": evidence_for_memory,
        "last_answer_summary": answer[:500],
        "last_route_meta": {
            "query_route": state.get("query_route", ""),
            "doc_need_route": state.get("doc_need_route", ""),
            "doc_need_reason": state.get("doc_need_reason", ""),
            "doc_need_confidence": state.get("doc_need_confidence", 0),
            "retrieval_plan": state.get("retrieval_plan") or {},
            "rewritten_query": state.get("rewritten_query", ""),
            "item_names": item_names,
        },
    }
    if item_names:
        updates["last_confirmed_items"] = [
            {
                "product_id": "",
                "canonical_name": item,
                "alias_name": "",
                "brand": "",
                "model_code": "",
                "category": "",
                "confidence": 1.0,
                "source": "node_answer_output",
            }
            for item in item_names
        ]
    if not state.get("pending_clarification_created"):
        updates["pending_clarification"] = None
    upsert_session_memory(session_id, updates)


def _maybe_write_product_memory(state: QueryGraphState, evidence_docs: list) -> None:
    """Persist product memory only for resolved, evidence-backed answers."""
    from app.utils.eval_flag_utils import is_eval_enabled
    if is_eval_enabled("EVAL_DISABLE_PRODUCT_MEMORY_WRITE"):
        return
    if state.get("query_route") != "product_doc_qa":
        return
    if state.get("doc_need_route") not in ("need_retrieval", "reuse_context_evidence"):
        return
    if (
        state.get("doc_need_route") != "reuse_context_evidence"
        and state.get("self_eval_status") != "sufficient"
    ):
        return

    item_names = state.get("item_names") or []
    if not item_names:
        return

    query = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    answer = (state.get("answer") or "").strip()

    reranked = state.get("reranked_docs") or []
    chunk_ids = [d.get("chunk_id") for d in reranked[:3] if d.get("chunk_id")]

    if not chunk_ids and evidence_docs:
        chunk_ids = [d.get("chunk_id") for d in evidence_docs[:3] if d.get("chunk_id")]

    if not query or not answer:
        return

    try:
        from app.clients.mongo_product_memory_utils import upsert_product_memory

        for canonical_name in item_names[:1]:
            upsert_product_memory(
                canonical_name=canonical_name,
                query_pattern=query,
                answer_summary=answer[:500],
                evidence_chunk_ids=chunk_ids,
                route_meta={
                    "retrieval_plan": state.get("retrieval_plan") or {},
                },
            )
    except Exception as e:
        logger.warning(f"Product memory write failed: {e}")
