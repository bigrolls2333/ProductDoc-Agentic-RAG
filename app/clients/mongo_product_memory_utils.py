"""
MongoDB 商品记忆（Product Memory）存取工具
==========================================
参照 mongo_history_utils.py 的单例模式，提供跨会话的商品级长期记忆 CRUD。

Product Memory 存储每个商品的长期知识：
- aliases: 所有已知名称变体（用于商品确认阶段别名命中）
- doc_ids: 绑定的文档 ID
- common_questions: 高频问题记录（用于 query expansion 和检索策略复用）

写入条件（四个条件全满足才写入，避免低质量数据污染）：
1. query_route == "product_doc_qa"
2. doc_need_route in ("need_retrieval", "reuse_context_evidence")
3. self_eval_status == "sufficient"
4. item_names 命中明确的 product_id（canonical_name）

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

# common_questions 上限
MAX_COMMON_QUESTIONS = 20


class ProductMemoryMongoTool:
    """
    MongoDB 商品记忆读写工具类（基于原生 PyMongo 实现）
    封装 MongoDB 的连接、集合初始化、索引创建，为上层提供统一的商品记忆存取入口。
    """

    def __init__(self):
        """初始化 MongoDB 连接，获取 product_memory 集合并创建索引。"""
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db["product_memory"]

            # canonical_name 唯一索引
            self.collection.create_index([("canonical_name", ASCENDING)], unique=True)
            # aliases 索引（支持别名搜索）
            self.collection.create_index([("aliases", ASCENDING)])
            # updated_at 索引
            self.collection.create_index([("updated_at", ASCENDING)])

            logging.info(f"ProductMemoryMongoTool 已连接: {self.db_name}.product_memory")
        except Exception as e:
            logging.error(f"ProductMemoryMongoTool 初始化失败: {e}")
            raise


# ========================
# 单例模式
# ========================
_product_memory_mongo_tool: Optional[ProductMemoryMongoTool] = None

try:
    _product_memory_mongo_tool = ProductMemoryMongoTool()
except Exception as e:
    logging.warning(f"模块加载时无法初始化 ProductMemoryMongoTool: {e}")


def get_product_memory_mongo_tool() -> Optional[ProductMemoryMongoTool]:
    """获取 ProductMemoryMongoTool 单例实例（懒加载）。"""
    global _product_memory_mongo_tool
    if _product_memory_mongo_tool is None:
        try:
            _product_memory_mongo_tool = ProductMemoryMongoTool()
        except Exception as e:
            logging.warning(f"懒加载 ProductMemoryMongoTool 失败: {e}")
            return None
    return _product_memory_mongo_tool


# ========================
# CRUD 函数
# ========================

def get_product_memory(canonical_name: str) -> dict | None:
    """
    获取指定商品的 Product Memory。

    :param canonical_name: 商品标准名称
    :return: 商品记忆字典；未找到或出错返回 None
    """
    if not canonical_name:
        return None

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return None

    try:
        doc = tool.collection.find_one({"canonical_name": canonical_name})
        if doc is None:
            return None
        doc.pop("_id", None)
        return doc
    except PyMongoError as e:
        logging.warning(f"获取 product_memory 失败: canonical_name={canonical_name}, error={e}")
        return None


def upsert_product_memory(
    canonical_name: str,
    query_pattern: str,
    answer_summary: str,
    evidence_chunk_ids: list,
    route_meta: dict,
) -> bool:
    """
    写入或更新商品记忆中的 common_question 条目。

    - 若相同/相似 query_pattern 已存在 → 频率 +1，更新 last_asked_at
    - 若为新模式 → 追加新条目（上限 MAX_COMMON_QUESTIONS 条，超出淘汰最旧的）
    - 若 canonical_name 不存在 → 先创建基础文档再追加

    :param canonical_name: 商品标准名称
    :param query_pattern: 用户查询模式
    :param answer_summary: 答案摘要（截断 500 字）
    :param evidence_chunk_ids: 证据 chunk ID 列表
    :param route_meta: 检索计划元信息
    :return: True 表示成功
    """
    if not canonical_name or not query_pattern:
        return False

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return False

    try:
        now = time.time()
        summary = (answer_summary or "")[:500]

        # 尝试查找已存在的相同/相似 query_pattern
        existing = tool.collection.find_one({"canonical_name": canonical_name})
        if existing is None:
            # 新建文档
            tool.collection.insert_one({
                "canonical_name": canonical_name,
                "aliases": [canonical_name],
                "brand": "",
                "category": "",
                "doc_ids": [],
                "common_questions": [{
                    "query_pattern": query_pattern,
                    "answer_summary": summary,
                    "evidence_chunk_ids": evidence_chunk_ids or [],
                    "frequency": 1,
                    "last_asked_at": now,
                    "route_meta": route_meta or {},
                }],
                "created_at": now,
                "updated_at": now,
            })
            return True

        # 查找是否已存在相似 query_pattern
        common_questions = existing.get("common_questions") or []
        found = False
        for cq in common_questions:
            existing_pattern = cq.get("query_pattern", "")
            # 简单相似度判断：去除空格后一致，或一个包含另一个
            if _patterns_similar(query_pattern, existing_pattern):
                cq["frequency"] = cq.get("frequency", 0) + 1
                cq["last_asked_at"] = now
                cq["answer_summary"] = summary
                cq["route_meta"] = route_meta or {}
                found = True
                break

        if not found:
            # 追加新条目
            common_questions.append({
                "query_pattern": query_pattern,
                "answer_summary": summary,
                "evidence_chunk_ids": evidence_chunk_ids or [],
                "frequency": 1,
                "last_asked_at": now,
                "route_meta": route_meta or {},
            })
            # 超出上限，淘汰最旧的（按 last_asked_at 升序）
            if len(common_questions) > MAX_COMMON_QUESTIONS:
                common_questions.sort(key=lambda x: x.get("last_asked_at", 0))
                common_questions = common_questions[-MAX_COMMON_QUESTIONS:]

        # 更新数据库
        tool.collection.update_one(
            {"canonical_name": canonical_name},
            {"$set": {
                "common_questions": common_questions,
                "updated_at": now,
            }}
        )
        return True

    except PyMongoError as e:
        logging.warning(f"写入 product_memory 失败: canonical_name={canonical_name}, error={e}")
        return False


def _patterns_similar(pattern1: str, pattern2: str) -> bool:
    """简单相似度判断：去除空格后一致，或一个包含另一个。"""
    p1 = "".join((pattern1 or "").split()).lower()
    p2 = "".join((pattern2 or "").split()).lower()
    if not p1 or not p2:
        return False
    if p1 == p2:
        return True
    if p1 in p2 or p2 in p1:
        return True
    return False


def search_product_memory(query: str, limit: int = 5) -> list:
    """
    搜索 Product Memory：通过 canonical_name 和 aliases 匹配。

    :param query: 搜索关键词
    :param limit: 返回数量上限
    :return: 匹配的商品记忆列表
    """
    if not query:
        return []

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return []

    try:
        # 搜索 canonical_name 或 aliases 包含 query
        results = list(tool.collection.find({
            "$or": [
                {"canonical_name": {"$regex": query, "$options": "i"}},
                {"aliases": {"$regex": query, "$options": "i"}},
            ]
        }).limit(limit))
        for doc in results:
            doc.pop("_id", None)
        return results
    except PyMongoError as e:
        logging.warning(f"搜索 product_memory 失败: query={query}, error={e}")
        return []


def add_product_alias(canonical_name: str, alias: str) -> bool:
    """
    为商品添加别名。

    :param canonical_name: 商品标准名称
    :param alias: 新别名
    :return: True 表示成功
    """
    if not canonical_name or not alias:
        return False

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return False

    try:
        tool.collection.update_one(
            {"canonical_name": canonical_name},
            {"$addToSet": {"aliases": alias}, "$set": {"updated_at": time.time()}},
            upsert=True,
        )
        return True
    except PyMongoError as e:
        logging.warning(f"添加 product alias 失败: canonical_name={canonical_name}, error={e}")
        return False


def link_doc_to_product(canonical_name: str, doc_id: str) -> bool:
    """
    将文档关联到商品。

    :param canonical_name: 商品标准名称
    :param doc_id: 文档 ID
    :return: True 表示成功
    """
    if not canonical_name or not doc_id:
        return False

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return False

    try:
        tool.collection.update_one(
            {"canonical_name": canonical_name},
            {"$addToSet": {"doc_ids": doc_id}, "$set": {"updated_at": time.time()}},
            upsert=True,
        )
        return True
    except PyMongoError as e:
        logging.warning(f"link_doc_to_product 失败: canonical_name={canonical_name}, error={e}")
        return False


# ========================
# 测试入口
# ========================
if __name__ == "__main__":
    import json

    test_product = "H3C LA2608室内无线网关"

    # 写入测试
    print("=== 写入 Product Memory ===")
    ok = upsert_product_memory(
        canonical_name=test_product,
        query_pattern="怎么配置上网方式",
        answer_summary="LA2608 支持 3G/4G 和 WiFi 接入，需在 Web 管理页面配置...",
        evidence_chunk_ids=["chunk_001", "chunk_002"],
        route_meta={"retrieval_plan": {"use_embedding": True, "use_hyde": False, "use_web": False}},
    )
    print(f"写入结果: {ok}")

    # 重复写入（频率+1）
    ok2 = upsert_product_memory(
        canonical_name=test_product,
        query_pattern="怎么配置上网方式",
        answer_summary="LA2608 支持多种上网方式...",
        evidence_chunk_ids=["chunk_001"],
        route_meta={"retrieval_plan": {"use_embedding": True}},
    )
    print(f"重复写入（频率+1）: {ok2}")

    # 读取测试
    print("\n=== 读取 Product Memory ===")
    mem = get_product_memory(test_product)
    if mem:
        common = mem.get("common_questions") or []
        print(f"canonical_name: {mem.get('canonical_name')}")
        print(f"common_questions 数量: {len(common)}")
        for cq in common:
            print(f"  - pattern={cq.get('query_pattern')}, freq={cq.get('frequency')}")
    else:
        print("未找到")

    # 别名搜索测试
    print("\n=== 别名搜索 ===")
    results = search_product_memory("LA2608")
    print(f"搜索结果数: {len(results)}")
