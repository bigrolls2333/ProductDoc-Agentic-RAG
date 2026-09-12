"""
Agentic RAG 改造验证脚本：跑真实 query_app，打印 retrieval_plan / self_eval / retry 三类场景的关键中间状态。

用法：
    .venv/Scripts/python.exe scripts/verify_agentic.py
    .venv/Scripts/python.exe scripts/verify_agentic.py --query "H3C LA2608 怎么接入4G网络"
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.query_process.agent.main_graph import query_app
from app.utils.task_utils import get_task_result, update_task_status, TASK_STATUS_COMPLETED


def run_one(query: str, is_stream: bool = False) -> dict:
    session_id = f"verify_{uuid.uuid4().hex[:8]}"
    default_state = {
        "original_query": query,
        "session_id": session_id,
        "is_stream": is_stream,
        "trace_started_at": time.time(),
    }
    print(f"\n{'='*70}")
    print(f"[INPUT ] query={query!r}  session_id={session_id}")
    try:
        final_state = query_app.invoke(default_state, {"recursion_limit": 50})
    except Exception as e:
        print(f"[ERROR ] graph invoke failed: {e}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        return {"error": str(e)}
    update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

    plan = final_state.get("retrieval_plan") or {}
    print(f"[PLAN  ] decided_by={plan.get('decided_by')} "
          f"use_embedding={plan.get('use_embedding')} "
          f"use_hyde={plan.get('use_hyde')} "
          f"use_web={plan.get('use_web')}")
    print(f"[PLAN  ] reason={plan.get('reason')}")

    print(f"[EVAL  ] status={final_state.get('self_eval_status')} "
          f"retry_count={final_state.get('self_eval_retry_count')}")
    detail = final_state.get("self_eval_detail") or {}
    if detail:
        print(f"[EVAL  ] confidence={detail.get('confidence')} "
              f"reason={detail.get('reason')}")
        if detail.get("missing_aspects"):
            print(f"[EVAL  ] missing_aspects={detail.get('missing_aspects')}")

    emb = final_state.get("embedding_chunks") or []
    hyde = final_state.get("hyde_embedding_chunks") or []
    web = final_state.get("web_search_docs") or []
    rrf = final_state.get("rrf_chunks") or []
    reranked = final_state.get("reranked_docs") or []
    print(f"[COUNT ] embedding={len(emb)} hyde={len(hyde)} web={len(web)} "
          f"rrf={len(rrf)} reranked={len(reranked)}")

    answer = final_state.get("answer") or ""
    preview = answer.replace("\n", " ")[:200]
    print(f"[ANSWER] len={len(answer)}  preview={preview!r}")

    final_state["session_id"] = session_id
    return final_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="", help="单条自定义查询；不传则跑 3 个预设场景")
    args = ap.parse_args()

    if args.query:
        run_one(args.query)
        return

    scenarios = [
        ("场景1-本地精确产品问题", "H3C LA2608 如何接入 3G/4G 网络？"),
        ("场景2-本地大概率没有的时效性查询(应触发web补搜)",
         "H3C LA2608 2026年最新固件版本号是多少"),
        ("场景3-完全不相关澄清类",
         "今天天气怎么样"),
    ]
    for title, q in scenarios:
        print("\n" + "#" * 70)
        print(f"# {title}")
        print("#" * 70)
        run_one(q)


if __name__ == "__main__":
    main()