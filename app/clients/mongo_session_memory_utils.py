"""
MongoDB 会话记忆（Session Memory）存取工具
===========================================
参照 mongo_history_utils.py 的单例模式，提供会话级别的短期记忆 CRUD。

Session Memory 存储每次查询后可复用的结构化上下文，包括：
- last_confirmed_items: 最近确认的商品名称（用于多轮指代消解）
- pending_clarification: 待处理澄清任务（支持"第一个"/"第二个"回复恢复）
- last_evidence_docs: 上一轮使用的证据文档摘要（用于追问复用）
- last_answer_summary: 上一轮回答摘要（用于"简单点说"等改写请求）
- last_route_meta: 上一轮路由决策元信息（用于 trace/debug）
- recent_message_refs: 最近消息引用

降级策略：任何 MongoDB 异常 catch 后 log warning，返回安全默认值，不阻断主流程。
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()


class SessionMemoryMongoTool:
    """
    MongoDB 会话记忆读写工具类（基于原生 PyMongo 实现）
    封装 MongoDB 的连接、集合初始化、索引创建，为上层提供统一的会话记忆存取入口。
    """

    def __init__(self):
        """初始化 MongoDB 连接，获取 session_memory 集合并创建索引。"""
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db["session_memory"]

            # session_id 唯一索引：每个会话只有一条记忆文档
            self.collection.create_index([("session_id", ASCENDING)], unique=True)
            # updated_at 索引：支持按更新时间查询和清理
            self.collection.create_index([("updated_at", ASCENDING)])

            logging.info(f"SessionMemoryMongoTool 已连接: {self.db_name}.session_memory")
        except Exception as e:
            logging.error(f"SessionMemoryMongoTool 初始化失败: {e}")
            raise


# ========================
# 单例模式
# ========================
_session_memory_mongo_tool: Optional[SessionMemoryMongoTool] = None

try:
    _session_memory_mongo_tool = SessionMemoryMongoTool()
except Exception as e:
    logging.warning(f"模块加载时无法初始化 SessionMemoryMongoTool: {e}")


def get_session_memory_mongo_tool() -> Optional[SessionMemoryMongoTool]:
    """获取 SessionMemoryMongoTool 单例实例（懒加载）。"""
    global _session_memory_mongo_tool
    if _session_memory_mongo_tool is None:
        try:
            _session_memory_mongo_tool = SessionMemoryMongoTool()
        except Exception as e:
            logging.warning(f"懒加载 SessionMemoryMongoTool 失败: {e}")
            return None
    return _session_memory_mongo_tool


# ========================
# CRUD 函数
# ========================

def get_session_memory(session_id: str) -> dict:
    """
    获取指定会话的 Session Memory。

    :param session_id: 会话唯一标识
    :return: 会话记忆字典；未找到或出错时返回空 dict（保证调用方无需判 None）
    """
    if not session_id:
        return {}

    tool = get_session_memory_mongo_tool()
    if tool is None:
        return {}

    try:
        doc = tool.collection.find_one({"session_id": session_id})
        if doc is None:
            return {}
        # 移除 MongoDB 的 _id 字段，保持返回纯业务数据
        doc.pop("_id", None)
        return doc
    except PyMongoError as e:
        logging.warning(f"获取 session_memory 失败，降级为空字典: session_id={session_id}, error={e}")
        return {}


def upsert_session_memory(session_id: str, updates: dict) -> bool:
    """
    写入或更新会话记忆（按字段合并，不覆盖未传入的字段）。

    :param session_id: 会话唯一标识
    :param updates: 要更新的字段字典（仅更新传入的字段）
    :return: True 表示写入成功，False 表示失败
    """
    if not session_id or not updates:
        return False

    tool = get_session_memory_mongo_tool()
    if tool is None:
        return False

    try:
        updates["updated_at"] = time.time()
        tool.collection.update_one(
            {"session_id": session_id},
            {"$set": updates},
            upsert=True,
        )
        return True
    except PyMongoError as e:
        logging.warning(f"写入 session_memory 失败: session_id={session_id}, error={e}")
        return False


def clear_session_memory(session_id: str) -> int:
    """
    清空指定会话的 Session Memory。

    :param session_id: 会话唯一标识
    :return: 实际删除的文档数量；失败返回 0
    """
    if not session_id:
        return 0

    tool = get_session_memory_mongo_tool()
    if tool is None:
        return 0

    try:
        result = tool.collection.delete_one({"session_id": session_id})
        return result.deleted_count
    except PyMongoError as e:
        logging.warning(f"删除 session_memory 失败: session_id={session_id}, error={e}")
        return 0


def clear_pending_clarification(session_id: str) -> bool:
    """
    清除待处理澄清状态（保留其他字段）。
    商品确认完成后调用，表示澄清已解决。

    :param session_id: 会话唯一标识
    :return: True 表示成功
    """
    return upsert_session_memory(session_id, {"pending_clarification": None})


# ========================
# 测试入口
# ========================
if __name__ == "__main__":
    test_sid = "test_session_memory_001"

    # 写入测试
    print("=== 写入 Session Memory ===")
    ok = upsert_session_memory(test_sid, {
        "last_confirmed_items": ["H3C LA2608室内无线网关"],
        "last_answer_summary": "LA2608 支持 3G/4G 和 WiFi 两种上网方式。",
        "last_route_meta": {
            "query_route": "product_doc_qa",
            "doc_need_route": "need_retrieval",
            "retrieval_plan": {"use_embedding": True, "use_hyde": False, "use_web": False},
        },
        "last_evidence_docs": [
            {"chunk_id": "chunk_001", "title": "操作手册", "text_preview": "LA2608 支持以下上网方式...", "score": 0.95}
        ],
        "pending_clarification": None,
    })
    print(f"写入结果: {ok}")

    # 读取测试
    print("\n=== 读取 Session Memory ===")
    mem = get_session_memory(test_sid)
    import json
    print(json.dumps(mem, ensure_ascii=False, indent=2, default=str))

    # 部分更新测试
    print("\n=== 部分更新 Session Memory ===")
    upsert_session_memory(test_sid, {"last_answer_summary": "更新后的摘要"})
    mem2 = get_session_memory(test_sid)
    print(f"last_answer_summary: {mem2.get('last_answer_summary')}")
    print(f"last_confirmed_items 应保留: {mem2.get('last_confirmed_items')}")

    # 清除测试
    print("\n=== 清除 Session Memory ===")
    deleted = clear_session_memory(test_sid)
    print(f"删除数量: {deleted}")
    mem3 = get_session_memory(test_sid)
    print(f"清除后查询: {mem3}")
