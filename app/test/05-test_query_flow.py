"""Exercise representative query-routing and retrieval scenarios."""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Allow the script to run from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import (
    get_task_result,
    update_task_status,
    TASK_STATUS_COMPLETED,
)


def run_one(query: str) -> dict:
    session_id = f"verify_{uuid.uuid4().hex[:8]}"
    initial_state = create_query_default_state(
        original_query=query,
        session_id=session_id,
        is_stream=False,
        trace_started_at=time.time(),
    )

    logger.info("=" * 70)
    logger.info(f"[INPUT ] query={query!r}  session_id={session_id}")

    try:
        final_state = query_app.invoke(initial_state, {"recursion_limit": 50})
    except Exception as e:
        logger.error(f"[ERROR ] graph invoke failed: {e}", exc_info=True)
        update_task_status(session_id, TASK_STATUS_COMPLETED, False)
        return {"error": str(e)}
    update_task_status(session_id, TASK_STATUS_COMPLETED, False)

    plan = final_state.get("retrieval_plan") or {}
    logger.info(
        f"[PLAN  ] decided_by={plan.get('decided_by')} "
        f"use_embedding={plan.get('use_embedding')} "
        f"use_hyde={plan.get('use_hyde')} use_web={plan.get('use_web')}"
    )
    logger.info(f"[PLAN  ] reason={plan.get('reason')}")

    logger.info(
        f"[EVAL  ] status={final_state.get('self_eval_status')} "
        f"retry_count={final_state.get('self_eval_retry_count')}"
    )
    detail = final_state.get("self_eval_detail") or {}
    if detail:
        logger.info(
            f"[EVAL  ] confidence={detail.get('confidence')} "
            f"reason={detail.get('reason')}"
        )
        if detail.get("missing_aspects"):
            logger.info(f"[EVAL  ] missing_aspects={detail.get('missing_aspects')}")

    emb = final_state.get("embedding_chunks") or []
    hyde = final_state.get("hyde_embedding_chunks") or []
    web = final_state.get("web_search_docs") or []
    rrf = final_state.get("rrf_chunks") or []
    reranked = final_state.get("reranked_docs") or []
    logger.info(
        f"[COUNT ] embedding={len(emb)} hyde={len(hyde)} web={len(web)} "
        f"rrf={len(rrf)} reranked={len(reranked)}"
    )

    answer = final_state.get("answer") or ""
    preview = answer.replace("\n", " ")[:200]
    logger.info(f"[ANSWER] len={len(answer)}  preview={preview!r}")

    # Check the task store as a compatibility path for non-streaming runs.
    task_answer = get_task_result(session_id, "answer", "")
    if task_answer and not answer:
        logger.info(f"[TASK  ] task_result.answer preview={str(task_answer)[:200]!r}")

    return final_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="", help="Run one custom query instead of the default scenarios.")
    args = ap.parse_args()

    if args.query:
        run_one(args.query)
        return

    scenarios = [
        ("local product query", "H3C LA2608 如何接入 3G/4G 网络？"),
        ("time-sensitive query with web retry",
         "H3C LA2608 2026年最新固件版本号是多少"),
        ("out-of-scope query", "今天天气怎么样"),
    ]
    for title, q in scenarios:
        logger.info("\n" + "#" * 70)
        logger.info(f"# {title}")
        logger.info("#" * 70)
        run_one(q)


if __name__ == "__main__":
    main()
