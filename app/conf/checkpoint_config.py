"""
LangGraph Checkpointer 配置
===========================
使用 SQLite 作为 LangGraph 状态持久化后端，支持：
- 图执行中途异常后从 checkpoint 恢复
- 通过 CHECKPOINT_ENABLED 环境变量开关（默认 true）
- 初始化失败时降级为 None（无状态模式），不阻断主流程

Checkpoint 与 Trace 分工：
- Checkpoint：管理图执行状态（当前节点、错误、步数）——用于恢复
- Trace：记录 Agent 决策过程——用于观测和面试展示
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

from app.core.logger import logger

# 数据库文件路径：项目根目录 / data / checkpoints.db
CHECKPOINT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "checkpoints.db"

_checkpointer_instance: Optional[object] = None


def get_checkpointer() -> Optional[object]:
    """
    获取 LangGraph SQLite Checkpointer 实例。

    首次调用时初始化 SQLite 连接和 SqliteSaver。
    后续调用返回已缓存的实例。

    :return: SqliteSaver 实例；禁用或初始化失败时返回 None
    """
    global _checkpointer_instance

    # 检查环境变量开关
    enabled = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
    if not enabled:
        logger.info("Checkpoint 已通过 CHECKPOINT_ENABLED=false 禁用")
        return None

    # 返回已缓存的实例
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    # 尝试初始化
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
        _checkpointer_instance = SqliteSaver(conn)
        logger.info(f"SQLite Checkpointer 已初始化: {CHECKPOINT_DB_PATH}")
        return _checkpointer_instance

    except ImportError:
        logger.warning(
            "无法导入 langgraph.checkpoint.sqlite.SqliteSaver，"
            "请升级 langgraph 版本或安装 langgraph-checkpoint-sqlite。"
            "降级为无状态模式。"
        )
        return None
    except Exception as e:
        logger.warning(f"无法初始化 SQLite Checkpointer，降级为无状态模式: {e}")
        return None


def close_checkpointer() -> None:
    """关闭 Checkpointer 的数据库连接（服务关闭时调用）。"""
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        try:
            # SqliteSaver 可能通过 .conn 暴露底层连接
            conn = getattr(_checkpointer_instance, "conn", None)
            if conn:
                conn.close()
            _checkpointer_instance = None
            logger.info("Checkpointer 已关闭")
        except Exception as e:
            logger.warning(f"关闭 Checkpointer 时出错: {e}")


def is_checkpoint_enabled() -> bool:
    """检查 Checkpoint 是否已启用并初始化成功。"""
    return get_checkpointer() is not None
