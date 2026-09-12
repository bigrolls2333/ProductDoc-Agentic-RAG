"""
Run State 运行元数据管理工具
============================
记录每次 LangGraph 图执行的元数据，支持监控、调试和 checkpoint 恢复。

存储方式：内存字典（轻量级，不依赖外部服务）
可选：同步写入 MongoDB run_state 集合（若可用）

记录字段：
- run_id: 运行唯一标识
- session_id: 会话标识
- thread_id: LangGraph thread_id
- graph_name: 图名称（query_app / import_app）
- current_node: 最后执行的节点名
- status: running / completed / failed / checkpointed
- checkpoint_backend: checkpoint 后端类型
- last_checkpoint_at: 最后一次 checkpoint 时间戳
- resume_from_checkpoint: 是否从 checkpoint 恢复
- error: 异常信息
"""

import time
import uuid
from typing import Dict, Any, Optional

from app.core.logger import logger

# 内存存储
_run_states: Dict[str, Dict[str, Any]] = {}

# 可选：尝试使用 MongoDB
_mongo_available: Optional[bool] = None


def _try_get_mongo_collection():
    """尝试获取 MongoDB run_state 集合，不可用时返回 None。"""
    global _mongo_available
    if _mongo_available is False:
        return None
    try:
        from app.clients.mongo_session_memory_utils import get_session_memory_mongo_tool
        tool = get_session_memory_mongo_tool()
        if tool is not None:
            return tool.db.get_collection("run_state")
    except Exception:
        pass
    _mongo_available = False
    return None


def create_run(
    session_id: str,
    graph_name: str = "query_app",
    checkpoint_backend: str = "sqlite",
    resume_from_checkpoint: bool = False,
) -> str:
    """
    创建新的运行记录。

    :param session_id: 会话标识
    :param graph_name: 图名称
    :param checkpoint_backend: Checkpoint 后端类型
    :param resume_from_checkpoint: 是否从 checkpoint 恢复
    :return: run_id
    """
    run_id = str(uuid.uuid4())[:12]
    now = time.time()

    doc = {
        "run_id": run_id,
        "session_id": session_id,
        "thread_id": session_id,
        "graph_name": graph_name,
        "current_node": "",
        "status": "running",
        "checkpoint_backend": checkpoint_backend,
        "last_checkpoint_at": None,
        "resume_from_checkpoint": resume_from_checkpoint,
        "error": None,
        "created_at": now,
        "completed_at": None,
    }

    # 内存存储
    _run_states[run_id] = doc

    # 可选：MongoDB 存储
    collection = _try_get_mongo_collection()
    if collection is not None:
        try:
            collection.insert_one(doc)
        except Exception as e:
            logger.debug(f"run_state MongoDB 写入失败（不影响主流程）: {e}")

    logger.info(f"Run [{run_id}] 已创建: session={session_id}, graph={graph_name}")
    return run_id


def update_run_node(run_id: str, node_name: str) -> None:
    """
    更新当前运行节点。

    :param run_id: 运行 ID
    :param node_name: 当前节点名
    """
    if run_id in _run_states:
        _run_states[run_id]["current_node"] = node_name
        _run_states[run_id]["updated_at"] = time.time()

    # 可选：更新 MongoDB
    collection = _try_get_mongo_collection()
    if collection is not None:
        try:
            collection.update_one(
                {"run_id": run_id},
                {"$set": {"current_node": node_name, "updated_at": time.time()}},
            )
        except Exception:
            pass


def checkpoint_run(run_id: str) -> None:
    """
    记录 checkpoint 时间戳。

    :param run_id: 运行 ID
    """
    now = time.time()
    if run_id in _run_states:
        _run_states[run_id]["last_checkpoint_at"] = now
        _run_states[run_id]["status"] = "checkpointed"

    collection = _try_get_mongo_collection()
    if collection is not None:
        try:
            collection.update_one(
                {"run_id": run_id},
                {"$set": {"last_checkpoint_at": now, "status": "checkpointed"}},
            )
        except Exception:
            pass


def complete_run(run_id: str, status: str = "completed", error: str = "") -> None:
    """
    标记运行完成或失败。

    :param run_id: 运行 ID
    :param status: completed / failed
    :param error: 错误信息（失败时）
    """
    now = time.time()
    if run_id in _run_states:
        _run_states[run_id]["status"] = status
        _run_states[run_id]["completed_at"] = now
        if error:
            _run_states[run_id]["error"] = error

    collection = _try_get_mongo_collection()
    if collection is not None:
        try:
            update = {"status": status, "completed_at": now}
            if error:
                update["error"] = error
            collection.update_one({"run_id": run_id}, {"$set": update})
        except Exception:
            pass

    logger.info(f"Run [{run_id}] {status}" + (f" error={error}" if error else ""))


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    获取运行记录。

    :param run_id: 运行 ID
    :return: 运行记录字典；未找到返回 None
    """
    return _run_states.get(run_id)


def get_session_runs(session_id: str, limit: int = 10) -> list:
    """
    获取指定会话最近的运行记录。

    :param session_id: 会话标识
    :param limit: 返回数量上限
    :return: 运行记录列表
    """
    results = [
        r for r in _run_states.values()
        if r.get("session_id") == session_id
    ]
    results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return results[:limit]
