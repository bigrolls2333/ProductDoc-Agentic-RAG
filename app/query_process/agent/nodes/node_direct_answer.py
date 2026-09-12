import sys

from app.clients.mongo_history_utils import save_chat_message
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def node_direct_answer(state: QueryGraphState) -> QueryGraphState:
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    original_query = state.get("original_query", "")
    direct_answer = (state.get("direct_answer") or "").strip()

    logger.info(">>> node_direct_answer: 开始处理")
    add_running_task(session_id, node_name, is_stream)

    if not direct_answer:
        direct_answer = "请补充一下具体产品型号或你想查询的问题。"

    state["answer"] = direct_answer
    state["item_names"] = []
    state["rewritten_query"] = original_query

    try:
        save_chat_message(
            session_id=session_id,
            role="user",
            text=original_query,
            rewritten_query=original_query,
            item_names=[],
        )
    except Exception as e:
        logger.error(f"node_direct_answer: 保存用户消息失败: {e}", exc_info=True)

    add_done_task(session_id, node_name, is_stream)
    logger.info("node_direct_answer: 直接回复已写入 state.answer")
    return state
